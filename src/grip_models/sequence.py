import os
import numpy as np
import pandas
import random
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils import resample
import scipy
import re
import numpy as np
import pandas as pd
import random
from torch.utils.checkpoint import checkpoint
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.stats import rankdata
from datetime import datetime
torch.cuda.is_available()
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Tuple, Optional
from pathlib import Path
import sys
import functools
import logging
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .common_modules import Linear, LayerNorm
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from .evoformer_fast import Evoformer
import json 
from types import SimpleNamespace


####don't forget to add l2 @

logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    """Load config from JSON file and convert to object."""
    default_config = {
        "model": {
            "num_atom": 5,
            "num_recycle": 3,
            "embeddings": {
                "seq_info_dim": 21,
                "m_dim": 1024,
                "tf_dim": 768,
                "seq_channel": 128,
                "pair_channel": 64,
                "max_relative_feature": 32,
                "go_embed_dim": 64
            },
            "heads": {
                "seqhead": {
                    "enabled": True,
                },
                "pairhead": {
                    "enabled": True,
                },
                "losshead": {
                    "enabled": True,
                    "l1_weight": 0.01,
                    "l2_weight": 0.01
                }
            }
        },
        "loss": {
            "seq": {
                "config": {},
                "loss_fn": "seq_loss",
                "weight": 1.0
            },
            "mrf_pair": {
                "config": {
                    "pair_norm_weight": 0.01,
                    "site_norm_weight": 0.01
                },
                "loss_fn": "mrf_pair_loss",
                "weight": 1.0
            }
        },
        "data": {
            "batch_size": 1,
            "padding": 100,
            "n_channels_emb": 21,
            "n_channels_seq": 21,
            "n_channels_esm": 33,
            "go_embedding_dim": 64,
            "m": 1024,
            "z": 128
        }
    }

    if config_path:
        with open(config_path) as f:
            user_config = json.load(f)
            default_config = merge_dicts(default_config, user_config)

    def dict_to_namespace(d):
        if not isinstance(d, dict):
            return d  # Return non-dict objects as-is
        namespace = SimpleNamespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(namespace, key, dict_to_namespace(value))  # Recursive call for nested dict
            else:
                setattr(namespace, key, value)
        return namespace


    config = dict_to_namespace(default_config)
    return config


restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}

# Update SeqHead to use these constants
class SeqHead(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config
        
        self.norm = LayerNorm(num_in_channel)
        # Use len(restypes_with_x) instead of importing from residue_constants
        #self.proj = Linear(num_in_channel, len(restypes_with_x), init='final')
        self.proj = nn.Linear(num_in_channel, len(restypes_with_x), bias=True)
        self.config = config
        
    def forward(self, headers, representations, batch):
        if 'seq' not in representations:
            print("Error: 'seq' not in representations!")
            return None
        print(f"Linear weights (proj): {self.proj.weight}")
        print(f"Linear bias (proj): {self.proj.bias}")

        
        x = representations['seq']
        print(f"SeqHead input shape: {x.shape}")
        x = self.norm(x)
        print(f"Normed_x first 5 values: {x[..., :5]}")
        logits = self.proj(x)
        print(f"logits first 5 values: {logits}")
        print(f"SeqHead logits shape: {logits.shape}")
        return dict(logits=logits)


class PairHead(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config
        # Use defined constants instead of importing
        self.num_token = len(restypes_with_x)  # changed from residue_constants
        self.norm = LayerNorm(num_in_channel)
        self.proj = nn.Linear(num_in_channel, self.num_token * self.num_token, bias=True)
        self.config = config
        
    def forward(self, headers, representations, batch):
        x = representations['pair']
        x = self.norm(x)
        logits = self.proj(x)
        print(f"Initial logits shape: {logits.shape}")  
        logits = rearrange(logits, 'b i j (a c) -> b i j a c', a=self.num_token)
        print(f"After first rearrange: {logits.shape}")
        logits = rearrange(0.5 * (logits + rearrange(logits, 'b i j a c -> b j i c a')), 'b i j a c -> b i j (a c)')
        #logits = 0.5 * (logits + rearrange(logits, 'b i j a c -> b j i c a'))
        print(f"Final logits shape: {logits.shape}")
        print(f"logits pair values: {logits}") 
        return dict(logits=logits)

class LossHead(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        self.l1_weight = getattr(config, 'l1_weight', 0.01)  # For regularization
        self.l2_weight = getattr(config, 'l2_weight', 0.01)  # For regularization

    def _compute_single_loss(self, logits, labels):
        """Compute single sequence conservation loss hi(xi|si)."""
        print("\n=== _compute_single_loss ===")
        print(f"logits shape: {logits.shape}")
        print(f"labels shape: {labels.shape}")
        
        '''
        # Direct cross entropy for single residue prediction
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction='mean'
        )
        print(f"Single loss: {loss.item()}")
        return loss
        '''

        # Direct cross entropy for single residue prediction
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction='sum'
        )
        print(f"Single loss: {loss.item()}")
        return loss

    def compute_partition_function(self, seq_logits, pair_logits, labels, num_residues, batch_size):
        num_classes = seq_logits.shape[-1]
        device = seq_logits.device
        
        print("\nPartition Function Debug:")
        print(f"seq_logits shape: {seq_logits.shape}")  # [B, N, C]
        print(f"pair_logits initial shape: {pair_logits.shape}")  # [B, N, N, C, C]

        # Get all possible amino acid combinations
        a_poss = torch.arange(num_classes, device=device)
        i_amino, j_amino = torch.meshgrid(a_poss, a_poss, indexing='ij')
        i_amino = i_amino.reshape(-1)  # [C*C]
        j_amino = j_amino.reshape(-1)  # [C*C]
        print(f"Number of possible amino acid pairs: {len(i_amino)}")  # C*C

        # 1. Single site terms for all possible amino acids
        hi_terms = seq_logits[:, :, i_amino].unsqueeze(2)  # [B, N, 1, C*C]
        hj_terms = seq_logits[:, :, j_amino].unsqueeze(1)  # [B, 1, N, C*C]
        
        print(f"hi_terms shape: {hi_terms.shape}")
        print(f"hi_terms range: [{hi_terms.min():.4f}, {hi_terms.max():.4f}]")
        print(f"hj_terms shape: {hj_terms.shape}")
        print(f"hj_terms range: [{hj_terms.min():.4f}, {hj_terms.max():.4f}]")

        # 2. Direct coupling terms for all possible amino acid pairs
        eij_terms = pair_logits[
            torch.arange(batch_size, device=device)[:, None, None, None],
            torch.arange(num_residues, device=device)[None, :, None, None],
            torch.arange(num_residues, device=device)[None, None, :, None],
            i_amino[None, None, None, :],
            j_amino[None, None, None, :]
        ]  # [B, N, N, C*C]
        
        print(f"eij_terms shape: {eij_terms.shape}")
        print(f"eij_terms range: [{eij_terms.min():.4f}, {eij_terms.max():.4f}]")

        # 3. Coupling terms with observed amino acids at position k
        eik_list = []
        ejk_list = []
        for k in range(num_residues):
            xk = labels[:, k:k+1]  # [B, 1]
            # Get terms for position k with all possible amino acids i/j
            eik_term = pair_logits[
                torch.arange(batch_size, device=device)[:, None, None],
                torch.arange(num_residues, device=device)[None, :, None],
                k,
                i_amino[None, None, :],
                xk.unsqueeze(-1).expand(-1, -1, num_classes*num_classes)
            ]  # [B, N, C*C]
            
            ejk_term = pair_logits[
                torch.arange(batch_size, device=device)[:, None, None],
                torch.arange(num_residues, device=device)[None, :, None],
                k,
                j_amino[None, None, :],
                xk.unsqueeze(-1).expand(-1, -1, num_classes*num_classes)
            ]  # [B, N, C*C]
            
            eik_list.append(eik_term)
            ejk_list.append(ejk_term)

        # Stack along residue dimension
        eik = torch.stack(eik_list, dim=2)  # [B, N, N, C*C]
        ejk = torch.stack(ejk_list, dim=2)  # [B, N, N, C*C]
        
        print(f"eik shape: {eik.shape}")
        print(f"eik range: [{eik.min():.4f}, {eik.max():.4f}]")
        print(f"ejk shape: {ejk.shape}")
        print(f"ejk range: [{ejk.min():.4f}, {ejk.max():.4f}]")

        # 4. Combine all terms
        all_terms = hi_terms + hj_terms + eij_terms + eik + ejk
        print(f"all_terms shape before logsumexp: {all_terms.shape}")
        print(f"all_terms range: [{all_terms.min():.4f}, {all_terms.max():.4f}]")

        # 5. Compute partition function
        logZ = torch.logsumexp(all_terms, dim=-1)  # [B, N, N]
        print(f"logZ final shape: {logZ.shape}")
        print(f"logZ range: [{logZ.min():.4f}, {logZ.max():.4f}]")

        return logZ
    '''
    def _compute_pair_loss(self, pair_logits, seq_logits, labels, z_matrix=None):
        batch_size, num_residues = labels.shape
        num_classes = seq_logits.shape[-1]
        device = pair_logits.device

        print("\nPair Loss Debug:")
        print(f"labels shape: {labels.shape}")  # [1, 10]
        print(f"seq_logits shape: {seq_logits.shape}")  # [1, 1, 10, 21]
        print(f"pair_logits shape: {pair_logits.shape}")  # [1, 10, 10, 441]

        # Handle seq_logits dimensions first
        if seq_logits.dim() == 4:
            seq_logits = seq_logits.squeeze(1)  # Remove the extra dimension -> [1, 10, 21]
        print(f"seq_logits after squeeze: {seq_logits.shape}")

        # Upper triangular mask
        pair_mask = torch.triu(torch.ones(num_residues, num_residues), diagonal=1).bool()
        pair_mask = pair_mask.to(device)[None, :, :]

        # Reshape pair_logits
        pair_logits = pair_logits.view(batch_size, num_residues, num_residues, num_classes, num_classes)

        # ENERGY CALCULATION
        # 1. Set up observed amino acid indices
        i_obs = labels  # [1, 10]
        j_obs = labels  # [1, 10]


        # Fix single terms calculation
        # 1. Get observed values
        hi_terms_obs = torch.gather(seq_logits, -1, i_obs.unsqueeze(-1))  # [1, 10, 1]
        hj_terms_obs = torch.gather(seq_logits, -1, j_obs.unsqueeze(-1))  # [1, 10, 1]

        # 2. Reshape for broadcasting
        hi_terms_obs = hi_terms_obs.squeeze(-1).unsqueeze(2)  # [1, 10, 1]
        hj_terms_obs = hj_terms_obs.squeeze(-1).unsqueeze(1)  # [1, 1, 10]


        # Rest of the code remains the same...
        print(f"hi_terms_obs shape: {hi_terms_obs.shape}")
        print(f"hi_terms_obs range: [{hi_terms_obs.min():.4f}, {hi_terms_obs.max():.4f}]")
        print(f"hj_terms_obs shape: {hj_terms_obs.shape}")
        print(f"hj_terms_obs range: [{hj_terms_obs.min():.4f}, {hj_terms_obs.max():.4f}]")

        # 3. Direct coupling terms for observed amino acid pairs
        eij_terms_obs = pair_logits[
            torch.arange(batch_size, device=device)[:, None, None],
            torch.arange(num_residues, device=device)[None, :, None],
            torch.arange(num_residues, device=device)[None, None, :],
            i_obs.unsqueeze(-1).expand(-1, -1, num_residues),
            j_obs.unsqueeze(1).expand(-1, num_residues, -1)
        ]  # [B, N, N]
        
        print(f"eij_terms_obs shape: {eij_terms_obs.shape}")
        print(f"eij_terms_obs range: [{eij_terms_obs.min():.4f}, {eij_terms_obs.max():.4f}]")

        # 4. Coupling terms with observed amino acids
        eik_obs_list = []
        ejk_obs_list = []
        for k in range(num_residues):
            xk = labels[:, k:k+1]  # [B, 1]
            # Get terms for position k with observed amino acids i/j
            eik_term = pair_logits[
                torch.arange(batch_size, device=device)[:, None],
                torch.arange(num_residues, device=device)[None, :],
                k,
                i_obs,
                xk.expand(-1, num_residues)
            ]  # [B, N]
            
            ejk_term = pair_logits[
                torch.arange(batch_size, device=device)[:, None],
                torch.arange(num_residues, device=device)[None, :],
                k,
                j_obs,
                xk.expand(-1, num_residues)
            ]  # [B, N]
            
            eik_obs_list.append(eik_term)
            ejk_obs_list.append(ejk_term)

        # Stack along residue dimension
        eik_terms_obs = torch.stack(eik_obs_list, dim=2)  # [B, N, N]
        ejk_terms_obs = torch.stack(ejk_obs_list, dim=2)  # [B, N, N]
        
        print(f"eik_terms_obs shape: {eik_terms_obs.shape}")
        print(f"eik_terms_obs range: [{eik_terms_obs.min():.4f}, {eik_terms_obs.max():.4f}]")
        print(f"ejk_terms_obs shape: {ejk_terms_obs.shape}")
        print(f"ejk_terms_obs range: [{ejk_terms_obs.min():.4f}, {ejk_terms_obs.max():.4f}]")

        # 5. Combine all terms for Energy
        Energy = hi_terms_obs + hj_terms_obs + eij_terms_obs + eik_terms_obs + ejk_terms_obs
        #Energy = torch.log(torch.exp(hi_terms_obs + hj_terms_obs + eij_terms_obs + eik_terms_obs + ejk_terms_obs))


        print(f"Energy final shape: {Energy.shape}")  # [B, N, N]
        print(f"Energy range: [{Energy.min():.4f}, {Energy.max():.4f}]")

        # 6. Compute partition function
        logZ = self.compute_partition_function(seq_logits, pair_logits, labels, num_residues, batch_size)
        print(f"logZ shape: {logZ.shape}")
        print(f"logZ range: [{logZ.min():.4f}, {logZ.max():.4f}]")

        # Verify Energy <= logZ
        energy_logz_diff = (Energy - logZ).detach()
        max_violation = energy_logz_diff.max().item()
        if max_violation > 0:
            print(f"\nWarning: Found Energy > logZ! Max violation: {max_violation:.4f}")

        # Calculate loss
        num_pairs = num_residues * (num_residues - 1) / 2
        #pseudo_nll = -((Energy - logZ) * pair_mask.float()).sum() / (num_pairs * batch_size)

        pseudo_nll = -((Energy - logZ) * pair_mask.float()).sum() / (batch_size * 2)

        # Add regularization
        l1_reg = self.l1_weight * pair_logits.abs().mean()
        l2_reg = self.l2_weight * (pair_logits ** 2).mean()

        print("\nLoss Components:")
        print(f"pseudo_nll: {pseudo_nll.item():.4f}")
        print(f"l1_reg: {l1_reg.item():.4f}")
        print(f"l2_reg: {l2_reg.item():.4f}")
        print(f"Total loss: {(pseudo_nll + l1_reg + l2_reg).item():.4f}")

        return pseudo_nll + l1_reg + l2_reg
        '''
    
    def _compute_pair_loss(self, pair_logits, seq_logits, labels, z_matrix=None):
        batch_size, num_residues = labels.shape
        num_classes = seq_logits.shape[-1]
        device = pair_logits.device

        print("\nPair Loss Debug:")
        print(f"labels shape: {labels.shape}")  # [1, 13]
        print(f"seq_logits shape: {seq_logits.shape}")  # [1, 1, 13, 21]
        print(f"pair_logits shape: {pair_logits.shape}")  # [1, 13, 13, 441]

        # Handle seq_logits dimensions first
        if seq_logits.dim() == 4:
            seq_logits = seq_logits.squeeze(1)  # Remove the extra dimension -> [1, 13, 21]
        print(f"seq_logits after squeeze: {seq_logits.shape}")

        # Upper triangular mask
        pair_mask = torch.triu(torch.ones(num_residues, num_residues), diagonal=1).bool()
        pair_mask = pair_mask.to(device)[None, :, :]

        # Reshape pair_logits for energy calculations
        pair_logits_reshaped = pair_logits.view(batch_size, num_residues, num_residues, num_classes, num_classes)

        # ENERGY CALCULATION
        # 1. Set up observed amino acid indices
        i_obs = labels  # [1, 13]
        j_obs = labels  # [1, 13]

        # Fix single terms calculation
        # 1. Get observed values
        hi_terms_obs = torch.gather(seq_logits, -1, i_obs.unsqueeze(-1))  # [1, 13, 1]
        hj_terms_obs = torch.gather(seq_logits, -1, j_obs.unsqueeze(-1))  # [1, 13, 1]

        # 2. Reshape for broadcasting
        hi_terms_obs = hi_terms_obs.squeeze(-1).unsqueeze(2)  # [1, 13, 1]
        hj_terms_obs = hj_terms_obs.squeeze(-1).unsqueeze(1)  # [1, 1, 13]

        print(f"hi_terms_obs shape: {hi_terms_obs.shape}")
        print(f"hi_terms_obs range: [{hi_terms_obs.min():.4f}, {hi_terms_obs.max():.4f}]")
        print(f"hj_terms_obs shape: {hj_terms_obs.shape}")
        print(f"hj_terms_obs range: [{hj_terms_obs.min():.4f}, {hj_terms_obs.max():.4f}]")

        # 3. Direct coupling terms for observed amino acid pairs
        eij_terms_obs = pair_logits_reshaped[
            torch.arange(batch_size, device=device)[:, None, None],
            torch.arange(num_residues, device=device)[None, :, None],
            torch.arange(num_residues, device=device)[None, None, :],
            i_obs.unsqueeze(-1).expand(-1, -1, num_residues),
            j_obs.unsqueeze(1).expand(-1, num_residues, -1)
        ]  # [B, N, N]
        
        print(f"eij_terms_obs shape: {eij_terms_obs.shape}")
        print(f"eij_terms_obs range: [{eij_terms_obs.min():.4f}, {eij_terms_obs.max():.4f}]")

        # 4. Coupling terms with observed amino acids
        eik_obs_list = []
        ejk_obs_list = []
        for k in range(num_residues):
            xk = labels[:, k:k+1]  # [B, 1]
            # Get terms for position k with observed amino acids i/j
            eik_term = pair_logits_reshaped[
                torch.arange(batch_size, device=device)[:, None],
                torch.arange(num_residues, device=device)[None, :],
                k,
                i_obs,
                xk.expand(-1, num_residues)
            ]  # [B, N]
            
            ejk_term = pair_logits_reshaped[
                torch.arange(batch_size, device=device)[:, None],
                torch.arange(num_residues, device=device)[None, :],
                k,
                j_obs,
                xk.expand(-1, num_residues)
            ]  # [B, N]
            
            eik_obs_list.append(eik_term)
            ejk_obs_list.append(ejk_term)

        # Stack along residue dimension
        eik_terms_obs = torch.stack(eik_obs_list, dim=2)  # [B, N, N]
        ejk_terms_obs = torch.stack(ejk_obs_list, dim=2)  # [B, N, N]
        
        print(f"eik_terms_obs shape: {eik_terms_obs.shape}")
        print(f"eik_terms_obs range: [{eik_terms_obs.min():.4f}, {eik_terms_obs.max():.4f}]")
        print(f"ejk_terms_obs shape: {ejk_terms_obs.shape}")
        print(f"ejk_terms_obs range: [{ejk_terms_obs.min():.4f}, {ejk_terms_obs.max():.4f}]")

        # 5. Combine all terms for Energy
        Energy = hi_terms_obs + hj_terms_obs + eij_terms_obs + eik_terms_obs + ejk_terms_obs

        print(f"Energy final shape: {Energy.shape}")  # [B, N, N]
        print(f"Energy range: [{Energy.min():.4f}, {Energy.max():.4f}]")

        # Cross Entropy Loss Calculation
        # Create pair targets
        pair_targets = i_obs.unsqueeze(2) * num_classes + j_obs.unsqueeze(1)  # [B, N, N]

        # Apply masking to logits and get valid pairs
        pair_mask_expanded = pair_mask.unsqueeze(-1).expand(-1, -1, -1, num_classes * num_classes)
        valid_logits = pair_logits[pair_mask_expanded].reshape(-1, num_classes * num_classes)
        valid_targets = pair_targets[pair_mask].reshape(-1)

        # Calculate cross entropy loss
        pair_loss = F.cross_entropy(
            valid_logits,  # [num_valid_pairs, 441]
            valid_targets.long(),  # [num_valid_pairs]
            reduction='sum'
        ) / (batch_size * 2)  # Divide by 2 since we only use upper triangular

        # Add regularization
        l1_reg = self.l1_weight * pair_logits.abs().mean()
        l2_reg = self.l2_weight * (pair_logits ** 2).mean()

        print("\nLoss Components:")
        print(f"pair_loss: {pair_loss.item():.4f}")
        print(f"l1_reg: {l1_reg.item():.4f}")
        print(f"l2_reg: {l2_reg.item():.4f}")
        print(f"Total loss: {(pair_loss + l1_reg + l2_reg).item():.4f}")

        return pair_loss + l1_reg + l2_reg  
                
    def forward(self, headers, representations, batch):
        """
        Compute total loss following equation (4) in paper:
        L = Lsingle + Lpair
        """
        losses = {}
        
        # Get predictions
        seq_head_outputs = headers.get('seqhead', {})
        pair_head_outputs = headers.get('pairhead', {})
        
        seq_logits = seq_head_outputs.get('logits') if seq_head_outputs else None
        pair_logits = pair_head_outputs.get('logits') if pair_head_outputs else None
        
        # Get labels
        labels = batch.get('aatype')
        
        # Compute losses
        if seq_logits is not None and labels is not None:
            losses['single'] = self._compute_single_loss(seq_logits, labels)
        else:
            losses['single'] = torch.tensor(0.0, device=seq_logits.device if seq_logits is not None else 'cuda', requires_grad=True)
            
        if pair_logits is not None and seq_logits is not None and labels is not None:
            # Get z_matrix from representations if available
            z_matrix = representations.get('pair')
            losses['pair'] = self._compute_pair_loss(
                pair_logits=pair_logits,
                seq_logits=seq_logits,
                labels=labels,
                z_matrix=z_matrix
            )
        else:
            losses['pair'] = torch.tensor(0.0, device=pair_logits.device if pair_logits is not None else 'cuda', requires_grad=True)
            
        # Total loss
        losses['total'] = losses['single'] + losses['pair']
        
        return losses


class FoldingHead(nn.Module):
    """Head to predict 3d struct."""
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel):
        super().__init__()
        self.struct_module = folding.StructureModule(config, num_in_seq_channel, num_in_pair_channel)
        self.config = config
        
    def forward(self, headers, representations, batch):
        return self.struct_module(representations, batch)


class HeaderBuilder:
    @staticmethod
    def build(config, seq_channel, pair_channel, parent):
        print(f"\n=== HeaderBuilder.build ===")
        print(f"Input channels - seq: {seq_channel}, pair: {pair_channel}")

        # Access heads under config.model
        heads_config = config.model.heads  # Adjust path here

        # Create factory with all possible heads
        head_factory = OrderedDict(
            seqhead=functools.partial(SeqHead, num_in_channel=seq_channel),
            pairhead=functools.partial(PairHead, num_in_channel=pair_channel),
            losshead=functools.partial(LossHead, num_in_channel=seq_channel)
        )

        heads = {}
        for head_name, h in head_factory.items():
            # Check if this head type exists in heads_config
            if hasattr(heads_config, head_name):
                head_config = getattr(heads_config, head_name)
                head = h(config=head_config)

                # Add to parent if it's a Module
                if isinstance(parent, nn.Module):
                    parent.add_module(head_name, head)

                heads[head_name] = head
                print(f"Added {head_name} head to the model.")

        return heads

import pandas as pd
import os
from Bio import SeqIO


def create_hla_mapping(hla_fasta_path="data/hla_prot.fasta", mapping_cache_path="data/hla_mapping.csv"):
    """
    Create mapping between HLA alleles and their sequences from official IPD-IMGT/HLA database
    Args:
        hla_fasta_path: path to hla_prot.fasta from IPD-IMGT/HLA database
        mapping_cache_path: path to save/load the mapping
    """
    if os.path.exists(mapping_cache_path):
        print("Loading cached HLA mapping...")
        hla_mapping = pd.read_csv(mapping_cache_path, index_col=0).to_dict()['sequence']
        return hla_mapping
    
    print("Creating new HLA mapping from FASTA...")
    hla_mapping = {}
    
    # Example header: >HLA:HLA00001 A*01:01:01:01 365 bp
    for record in SeqIO.parse(hla_fasta_path, "fasta"):
        header_parts = record.description.split()
        if len(header_parts) < 2:
            continue
            
        # Get the HLA identifier and allele name
        hla_id = header_parts[0]  # e.g., HLA:HLA00001
        allele = header_parts[1]  # e.g., A*01:01:01:01
        
        # Check for valid HLA allele format
        if not any(allele.startswith(prefix) for prefix in ['A*', 'B*', 'C*']):
            continue
            
        # Get main allele designation (first two fields)
        allele_parts = allele.split(':')
        if len(allele_parts) < 2:
            continue
            
        # Extract the locus (A,B,C) and first two fields
        main_type = allele_parts[0]  # e.g., A*01
        subtype = allele_parts[1]    # e.g., 01
        standardized_allele = f"HLA-{main_type}:{subtype}"
        
        sequence = str(record.seq)
        
        # Validate sequence
        if not sequence or not all(aa in 'ARNDCQEGHILKMFPSTWYV' for aa in sequence):
            print(f"Warning: Invalid sequence for {standardized_allele}")
            continue
            
        # Store both with and without HLA- prefix
        hla_mapping[standardized_allele] = sequence
        hla_mapping[standardized_allele.replace("HLA-", "")] = sequence
    
    print(f"Loaded {len(hla_mapping)} valid HLA sequences")
    print("Sample alleles:", sorted(list(hla_mapping.keys()))[:5])
    
    if len(hla_mapping) == 0:
        raise ValueError("No valid HLA sequences found in FASTA file")
    
    # Save mapping
    pd.DataFrame({'sequence': hla_mapping}).to_csv(mapping_cache_path)
    
    return hla_mapping

# Example usage:
if __name__ == "__main__":
    try:
        hla_mapping = create_hla_mapping()
        
        # Print some statistics
        print("\nHLA allele statistics:")
        print(f"Total alleles: {len(hla_mapping)}")
        
        # Count by locus
        locus_counts = {}
        for allele in hla_mapping.keys():
            if allele.startswith("HLA-"):
                locus = allele.split("-")[1][0]  # Get A, B, or C
                locus_counts[locus] = locus_counts.get(locus, 0) + 1
                
        print("\nAlleles per locus:")
        for locus, count in sorted(locus_counts.items()):
            print(f"HLA-{locus}: {count} alleles")
            
    except Exception as e:
        print(f"Error creating HLA mapping: {e}")

class DataGenerator(Dataset):
    def __init__(self, list_IDs, labels, dataFrameIn, config, integrated_data_path="./data/integrated_tcr_antigen_data.csv"):
        super().__init__()
        c = config.data
        
        # Load data
        if integrated_data_path is not None:
            self.data = pd.read_csv(integrated_data_path)
        else:
            self.data = dataFrameIn
            
        # Load HLA mapping (needs to be before filter_by_length)
        #self.hla_mapping = create_hla_mapping()
        
        # Store residue mappings
        self.restype_order_with_x = restype_order_with_x
        
        # Filter out empty epitopes and process modifications
        self.data = self._preprocess_data(self.data)
        max_length = getattr(c, 'max_length', 500)
        self.data = self._filter_by_length(self.data, max_length)
        
        # Get parameters from config
        self.batch_size = c.batch_size
        self.n_channels_emb = c.n_channels_emb
        self.n_channels_seq = c.n_channels_seq
        self.m = c.m
        self.z = c.z
        
        self.labels = labels
        self.list_IDs = list_IDs
        self.dataFrameIn = dataFrameIn

    def _filter_by_length(self, df, max_length):
        """Filter sequences by maximum length"""
        print("\nApplying length filtering...")
        initial_count = len(df)
        
        # Filter MHC sequences - updated to use direct sequence
        mhc_mask = df['mhc'].apply(lambda x: 
            len(x) <= max_length if pd.notna(x) else True
        )
        
        # Filter TCR chain (beta chain)
        tcr_mask = df['cdr3_b'].apply(lambda x: 
            len(x) <= max_length if pd.notna(x) else True
        )
        
        # Filter epitope sequences
        epitope_mask = df['epitope_sequence'].apply(lambda x: 
            len(x) <= max_length if pd.notna(x) else True
        )
        
        # Apply all filters
        combined_mask = mhc_mask & tcr_mask & epitope_mask
        filtered_df = df[combined_mask]
        
        # Print filtering statistics
        print(f"\nLength filtering statistics (max length: {max_length}):")
        print(f"Original sequences: {initial_count}")
        print(f"Sequences removed:")
        print(f"  MHC: {len(df) - len(df[mhc_mask])}")
        print(f"  TCR beta: {len(df) - len(df[tcr_mask])}")
        print(f"  Epitope: {len(df) - len(df[epitope_mask])}")
        print(f"Final sequences: {len(filtered_df)}")
        
        return filtered_df

        
        # Load data
        if integrated_data_path is not None:
            self.data = pd.read_csv(integrated_data_path)
        else:
            self.data = dataFrameIn
            
        # Filter out empty epitopes and process modifications
        self.data = self._preprocess_data(self.data)
        max_length = getattr(c, 'max_length', 500)

        self.data = self._filter_by_length(self.data, max_length)
        
        # Load HLA mapping
        self.hla_mapping = create_hla_mapping()
        
        # Store residue mappings
        self.restype_order_with_x = restype_order_with_x
        
        # Get parameters from config
        self.batch_size = c.batch_size
        self.n_channels_emb = c.n_channels_emb
        self.n_channels_seq = c.n_channels_seq
        self.m = c.m
        self.z = c.z
        
        self.labels = labels
        self.list_IDs = list_IDs
        self.dataFrameIn = dataFrameIn
    

    def _preprocess_data(self, df):
        """
        Preprocess the data:
        1. Remove rows with empty epitopes
        2. Clean epitope sequences by removing modifications
        """
        # Remove rows where epitope is empty
        df = df[df['epitope_sequence'].notna()]
        df = df[df['epitope_sequence'].str.len() > 0]
        
        # Clean epitope sequences
        def clean_epitope(seq):
            """Remove modifications from epitope sequence"""
            if pd.isna(seq):
                return seq
                
            # Split on '+' and take the base sequence
            base_seq = seq.split('+')[0].strip()
            
            # Remove any remaining parentheses content
            base_seq = re.sub(r'\([^)]*\)', '', base_seq)
            
            # Keep only valid amino acids
            valid_aas = set('ARNDCQEGHILKMFPSTWYV')
            base_seq = ''.join(aa for aa in base_seq if aa in valid_aas)
            
            return base_seq
            
        df['epitope_sequence'] = df['epitope_sequence'].apply(clean_epitope)
        
        # Remove any rows where cleaning resulted in empty sequences
        df = df[df['epitope_sequence'].str.len() > 0]
        
        print(f"After preprocessing: {len(df)} valid sequences")
        return df

    def __len__(self):
        return len(self.data)

    def encode_sequence(self, seq, as_target=False):
        """
        Encode an amino acid sequence into one-hot format or target indices.
        
        Args:
            seq: Amino acid sequence string
            as_target: If True, return indices instead of one-hot encoding
            
        Returns:
            torch.Tensor: One-hot encoded sequence [seq_len, n_channels_seq] or
                         target indices [seq_len] if as_target=True
        """
        # Handle empty or missing sequences
        if pd.isna(seq) or len(seq) == 0:
            return torch.zeros((0, self.n_channels_seq), dtype=torch.float32) if not as_target \
                   else torch.zeros(0, dtype=torch.long)
        
        if as_target:
            # Return indices for target
            return torch.tensor([
                self.restype_order_with_x.get(aa, len(self.restype_order_with_x) - 1)
                for aa in seq
            ], dtype=torch.long)
        else:
            # Return one-hot encoding for input
            encoded = torch.zeros(len(seq), self.n_channels_seq, dtype=torch.float32)
            for index, aa in enumerate(seq):
                if aa in self.restype_order_with_x:
                    aa_index = self.restype_order_with_x[aa]
                    encoded[index, aa_index] = 1
                else:
                    # Unknown amino acid gets marked in last channel
                    encoded[index, -1] = 1
            return encoded

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get peptide length for position encoding only
        epitope_seq = row['epitope_sequence']
        epitope_len = len(epitope_seq)
        
        # Get MHC and TCR sequences
        mhc_seq = row['mhc'] if pd.notna(row['mhc']) else ""
        l_v = row['l_v']
        l_j = row['l_j']
        h_v = row['h_v']
        cdr3_b = row['cdr3_b']
        h_j = row['h_j']
        print(f"Sample {idx}: epitope='{epitope_seq}' mhc='{mhc_seq[:20]}...'")
        # Encode input sequences
        mhc_encoded = self.encode_sequence(mhc_seq)
        l_v_encoded = self.encode_sequence(l_v) if not pd.isna(l_v) else torch.zeros(0, self.n_channels_seq)
        l_j_encoded = self.encode_sequence(l_j) if not pd.isna(l_j) else torch.zeros(0, self.n_channels_seq)
        h_v_encoded = self.encode_sequence(h_v) if not pd.isna(h_v) else torch.zeros(0, self.n_channels_seq)
        cdr3_b_encoded = self.encode_sequence(cdr3_b) if not pd.isna(cdr3_b) else torch.zeros(0, self.n_channels_seq)
        h_j_encoded = self.encode_sequence(h_j) if not pd.isna(h_j) else torch.zeros(0, self.n_channels_seq)
        
        # Get lengths
        mhc_len = len(mhc_seq) if not pd.isna(mhc_seq) else 0
        l_v_len = len(l_v) if not pd.isna(l_v) else 0
        l_j_len = len(l_j) if not pd.isna(l_j) else 0
        h_v_len = len(h_v) if not pd.isna(h_v) else 0
        cdr3_b_len = len(cdr3_b) if not pd.isna(cdr3_b) else 0
        h_j_len = len(h_j) if not pd.isna(h_j) else 0
        
        item = {
            'mhc_encoded': mhc_encoded,
            'l_v_encoded': l_v_encoded,
            'l_j_encoded': l_j_encoded,
            'h_v_encoded': h_v_encoded,
            'cdr3_b_encoded': cdr3_b_encoded,
            'h_j_encoded': h_j_encoded,
            
            'mhc_ri': torch.arange(mhc_len) if mhc_len > 0 else torch.zeros(0),
            'l_v_ri': torch.arange(l_v_len) if l_v_len > 0 else torch.zeros(0),
            'l_j_ri': torch.arange(l_j_len) if l_j_len > 0 else torch.zeros(0),
            'h_v_ri': torch.arange(h_v_len) if h_v_len > 0 else torch.zeros(0),
            'cdr3_b_ri': torch.arange(cdr3_b_len) if cdr3_b_len > 0 else torch.zeros(0),
            'h_j_ri': torch.arange(h_j_len) if h_j_len > 0 else torch.zeros(0),
            'epitope_ri': torch.arange(epitope_len),  # Position indices for peptide
            
            'mhc_types': torch.zeros(mhc_len) if mhc_len > 0 else torch.zeros(0),
            'l_v_types': torch.full((l_v_len,), 1) if l_v_len > 0 else torch.zeros(0),
            'l_j_types': torch.full((l_j_len,), 2) if l_j_len > 0 else torch.zeros(0),
            'h_v_types': torch.full((h_v_len,), 3) if h_v_len > 0 else torch.zeros(0),
            'cdr3_b_types': torch.full((cdr3_b_len,), 4) if cdr3_b_len > 0 else torch.zeros(0),
            'h_j_types': torch.full((h_j_len,), 5) if h_j_len > 0 else torch.zeros(0),
            'epitope_types': torch.full((epitope_len,), 6),  # Type indicators for peptide
            
            'mhc_len': mhc_len,
            'l_v_len': l_v_len,
            'l_j_len': l_j_len,
            'h_v_len': h_v_len,
            'cdr3_b_len': cdr3_b_len,
            'h_j_len': h_j_len,
            'epitope_len': epitope_len
        }
        
        # Target is the actual peptide sequence
        target = self.encode_sequence(epitope_seq, as_target=True)
        
        return item, target

    
    def collate_fn(self, batch):
        """Collate function handling all TCR features"""
        batch_size = len(batch)
        
        # Get max lengths for all sequences
        max_mhc_len = max(x[0]['mhc_len'] for x in batch)
        max_epitope_len = max(x[0]['epitope_len'] for x in batch)
        max_l_v_len = max(x[0]['l_v_len'] for x in batch)
        max_l_j_len = max(x[0]['l_j_len'] for x in batch)
        max_h_v_len = max(x[0]['h_v_len'] for x in batch)
        max_cdr3_b_len = max(x[0]['cdr3_b_len'] for x in batch)
        max_h_j_len = max(x[0]['h_j_len'] for x in batch)
        
        # Initialize tensors for encoded sequences
        mhc_encoded = torch.zeros(batch_size, max_mhc_len, self.n_channels_seq) if max_mhc_len > 0 else \
                    torch.zeros(batch_size, 0, self.n_channels_seq)
        l_v_encoded = torch.zeros(batch_size, max_l_v_len, self.n_channels_seq) if max_l_v_len > 0 else \
                    torch.zeros(batch_size, 0, self.n_channels_seq)
        l_j_encoded = torch.zeros(batch_size, max_l_j_len, self.n_channels_seq) if max_l_j_len > 0 else \
                    torch.zeros(batch_size, 0, self.n_channels_seq)
        h_v_encoded = torch.zeros(batch_size, max_h_v_len, self.n_channels_seq) if max_h_v_len > 0 else \
                    torch.zeros(batch_size, 0, self.n_channels_seq)
        cdr3_b_encoded = torch.zeros(batch_size, max_cdr3_b_len, self.n_channels_seq) if max_cdr3_b_len > 0 else \
                        torch.zeros(batch_size, 0, self.n_channels_seq)
        h_j_encoded = torch.zeros(batch_size, max_h_j_len, self.n_channels_seq) if max_h_j_len > 0 else \
                    torch.zeros(batch_size, 0, self.n_channels_seq)
        
        # Initialize position indices
        mhc_ri = torch.zeros(batch_size, max_mhc_len) if max_mhc_len > 0 else torch.zeros(batch_size, 0)
        epitope_ri = torch.zeros(batch_size, max_epitope_len)
        l_v_ri = torch.zeros(batch_size, max_l_v_len) if max_l_v_len > 0 else torch.zeros(batch_size, 0)
        l_j_ri = torch.zeros(batch_size, max_l_j_len) if max_l_j_len > 0 else torch.zeros(batch_size, 0)
        h_v_ri = torch.zeros(batch_size, max_h_v_len) if max_h_v_len > 0 else torch.zeros(batch_size, 0)
        cdr3_b_ri = torch.zeros(batch_size, max_cdr3_b_len) if max_cdr3_b_len > 0 else torch.zeros(batch_size, 0)
        h_j_ri = torch.zeros(batch_size, max_h_j_len) if max_h_j_len > 0 else torch.zeros(batch_size, 0)
        
        # Initialize type indicators
        mhc_types = torch.zeros(batch_size, max_mhc_len) if max_mhc_len > 0 else torch.zeros(batch_size, 0)
        epitope_types = torch.zeros(batch_size, max_epitope_len)
        l_v_types = torch.zeros(batch_size, max_l_v_len) if max_l_v_len > 0 else torch.zeros(batch_size, 0)
        l_j_types = torch.zeros(batch_size, max_l_j_len) if max_l_j_len > 0 else torch.zeros(batch_size, 0)
        h_v_types = torch.zeros(batch_size, max_h_v_len) if max_h_v_len > 0 else torch.zeros(batch_size, 0)
        cdr3_b_types = torch.zeros(batch_size, max_cdr3_b_len) if max_cdr3_b_len > 0 else torch.zeros(batch_size, 0)
        h_j_types = torch.zeros(batch_size, max_h_j_len) if max_h_j_len > 0 else torch.zeros(batch_size, 0)
        
        # Prepare target tensor
        max_target_len = max(len(t) for _, t in batch)
        target_tensor = torch.zeros(batch_size, max_target_len, dtype=torch.long)
        
        # Fill batch tensors
        for i, (item, target) in enumerate(batch):
            # Fill target tensor
            target_len = len(target)
            target_tensor[i, :target_len] = target
            
            # Fill MHC tensors if present
            if item['mhc_len'] > 0:
                mhc_encoded[i, :item['mhc_len']] = item['mhc_encoded']
                mhc_ri[i, :item['mhc_len']] = item['mhc_ri']
                mhc_types[i, :item['mhc_len']] = item['mhc_types']
            
            # Fill epitope tensors
            if item['epitope_len'] > 0:
                epitope_ri[i, :item['epitope_len']] = item['epitope_ri']
                epitope_types[i, :item['epitope_len']] = item['epitope_types']
            
            # Fill TCR feature tensors
            if item['l_v_len'] > 0:
                l_v_encoded[i, :item['l_v_len']] = item['l_v_encoded']
                l_v_ri[i, :item['l_v_len']] = item['l_v_ri']
                l_v_types[i, :item['l_v_len']] = item['l_v_types']
                
            if item['l_j_len'] > 0:
                l_j_encoded[i, :item['l_j_len']] = item['l_j_encoded']
                l_j_ri[i, :item['l_j_len']] = item['l_j_ri']
                l_j_types[i, :item['l_j_len']] = item['l_j_types']
                
            if item['h_v_len'] > 0:
                h_v_encoded[i, :item['h_v_len']] = item['h_v_encoded']
                h_v_ri[i, :item['h_v_len']] = item['h_v_ri']
                h_v_types[i, :item['h_v_len']] = item['h_v_types']
                
            if item['cdr3_b_len'] > 0:
                cdr3_b_encoded[i, :item['cdr3_b_len']] = item['cdr3_b_encoded']
                cdr3_b_ri[i, :item['cdr3_b_len']] = item['cdr3_b_ri']
                cdr3_b_types[i, :item['cdr3_b_len']] = item['cdr3_b_types']
                
            if item['h_j_len'] > 0:
                h_j_encoded[i, :item['h_j_len']] = item['h_j_encoded']
                h_j_ri[i, :item['h_j_len']] = item['h_j_ri']
                h_j_types[i, :item['h_j_len']] = item['h_j_types']
        
        # Return all tensors in a dictionary
        return {
            'mhc_encoded': mhc_encoded,
            'l_v_encoded': l_v_encoded,
            'l_j_encoded': l_j_encoded,
            'h_v_encoded': h_v_encoded,
            'cdr3_b_encoded': cdr3_b_encoded,
            'h_j_encoded': h_j_encoded,
            
            'mhc_ri': mhc_ri,
            'epitope_ri': epitope_ri,
            'l_v_ri': l_v_ri,
            'l_j_ri': l_j_ri,
            'h_v_ri': h_v_ri,
            'cdr3_b_ri': cdr3_b_ri,
            'h_j_ri': h_j_ri,
            
            'mhc_types': mhc_types,
            'epitope_types': epitope_types,
            'l_v_types': l_v_types,
            'l_j_types': l_j_types,
            'h_v_types': h_v_types,
            'cdr3_b_types': cdr3_b_types,
            'h_j_types': h_j_types,
            
            'mhc_lengths': torch.tensor([x[0]['mhc_len'] for x in batch]),
            'epitope_lengths': torch.tensor([x[0]['epitope_len'] for x in batch]),
            'l_v_lengths': torch.tensor([x[0]['l_v_len'] for x in batch]),
            'l_j_lengths': torch.tensor([x[0]['l_j_len'] for x in batch]),
            'h_v_lengths': torch.tensor([x[0]['h_v_len'] for x in batch]),
            'cdr3_b_lengths': torch.tensor([x[0]['cdr3_b_len'] for x in batch]),
            'h_j_lengths': torch.tensor([x[0]['h_j_len'] for x in batch]),
            
            'aatype': target_tensor
        }, target_tensor

class myModel(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.use_checkpoint = True
        e = config.model.embeddings


        self.heads = HeaderBuilder.build(config, seq_channel=128, pair_channel=64, parent=self)
        self.loss_head = self.heads.get('losshead', None) 

        # Channels
        self.seq_channel = e.seq_channel  # 128
        self.pair_channel = e.pair_channel  # 64
        self.n_channels_seq = e.seq_info_dim  # 21
        self.tf_dim = e.tf_dim  # 768
        self.relpos_k = 32

        self.linear_single_projection = nn.Linear(self.seq_channel, self.seq_channel)
        self.linear_pair_projection = nn.Linear(self.pair_channel, self.pair_channel)


        self.evmodel = Evoformer(self.seq_channel, self.pair_channel)

        # Position encoding parameters
        self.num_frequencies = self.seq_channel // 2
        self.linear_abs_pos = nn.Linear(self.seq_channel, self.seq_channel)

        # Update number of types for new TCR features
        self.num_types = 7  # MHC, l_v, l_j, h_v, cdr3_b, h_j, epitope
        self.type_embedding = nn.Embedding(self.num_types, self.seq_channel)
        self.pair_type_embedding = nn.Embedding(self.num_types * self.num_types, self.pair_channel)
        
        # Linear layers for sequences
        self.linear_mhc_z_i = nn.Linear(self.n_channels_seq, self.pair_channel)
        self.linear_mhc_z_j = nn.Linear(self.n_channels_seq, self.pair_channel)
        self.linear_mhc_m = nn.Linear(self.n_channels_seq, self.seq_channel)
        
        # Linear layers for TCR components
        self.linear_tcr_z_i = nn.Linear(self.n_channels_seq, self.pair_channel)
        self.linear_tcr_z_j = nn.Linear(self.n_channels_seq, self.pair_channel)
        self.linear_tcr_m = nn.Linear(self.n_channels_seq, self.seq_channel)
        
        self.linear_combined_z_i = nn.Linear(self.n_channels_seq, self.pair_channel)
        self.linear_combined_z_j = nn.Linear(self.n_channels_seq, self.pair_channel)
        
        self.linear_relpos = nn.Linear(2 * self.relpos_k + 1, self.pair_channel)

    def add(self, x, y, inplace=False):
        """Helper method for adding tensors with optional inplace operation"""
        return x.add_(y) if inplace else x + y

    def get_sinusoidal_pos_encoding(self, positions):
        """Generate sinusoidal position encoding for sequences."""
        if len(positions.shape) == 1:
            positions = positions.unsqueeze(0)
        batch_size, seq_len = positions.shape
        
        freq = torch.exp(
            torch.arange(0, self.num_frequencies, device=positions.device) *
            (-math.log(10000.0) / self.num_frequencies)
        )
        
        pos = positions.unsqueeze(-1).float()
        pos_freq = pos * freq.unsqueeze(0).unsqueeze(0)
        
        pos_enc = torch.cat([
            torch.sin(pos_freq),
            torch.cos(pos_freq)
        ], dim=-1)
        
        if self.seq_channel % 2 == 1:
            pos_enc = F.pad(pos_enc, (0, 1), mode='constant', value=0)
        
        return self.linear_abs_pos(pos_enc)

    def relpos(self, ri: torch.Tensor, start_idx: int, end_idx: int):
        """
        Computes relative positional encodings for a sequence segment
        Args:
            ri: position indices tensor [batch, total_len]
            start_idx: start index of the sequence segment
            end_idx: end index of the sequence segment
        Returns:
            Relative position encodings for the segment [batch, segment_len, segment_len, pair_channel]
        """
        # Extract the relevant segment
        segment_ri = ri[..., start_idx:end_idx]  # [batch, segment_len]
        batch_size, segment_len = segment_ri.shape
        
        # Compute pairwise differences
        pos_i = segment_ri.unsqueeze(-1)  # [batch, segment_len, 1]
        pos_j = segment_ri.unsqueeze(-2)  # [batch, 1, segment_len]
        d = pos_i - pos_j  # [batch, segment_len, segment_len]
        
        # Create boundaries for binning
        boundaries = torch.arange(
            start=-self.relpos_k, 
            end=self.relpos_k + 1, 
            device=ri.device
        )
        
        # Compute bin assignments
        d = d.unsqueeze(-1) - boundaries.view(1, 1, 1, -1)  # [batch, segment_len, segment_len, num_bins]
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)  # [batch, segment_len, segment_len]
        
        # Convert to one-hot encoding
        d = nn.functional.one_hot(d, num_classes=2 * self.relpos_k + 1).float()  # [batch, segment_len, segment_len, num_bins]
        d = d.to(ri.dtype)
        d = d.float()
        # Project to pair channel dimension
        pos_emb = self.linear_relpos(d)  # [batch, segment_len, segment_len, pair_channel]
        
        return pos_emb

    def _create_pair_type_embeddings(self, type_tensor):
        """Creates pairwise type embeddings."""
        type_tensor = type_tensor.long()
        batch_size, seq_len = type_tensor.shape
        
        type_i = type_tensor.unsqueeze(2)
        type_j = type_tensor.unsqueeze(1)
        pair_type_idx = type_i * self.num_types + type_j
        
        return self.pair_type_embedding(pair_type_idx)

    def _run_evoformer(self, single, pair):
        """Wrapper for Evoformer forward pass to handle checkpointing"""
        def evoformer_function(s, p):
            return self.evmodel(s, p)
            
        if self.use_checkpoint and self.training:
            # Use checkpoint_sequential for the Evoformer
            return torch.utils.checkpoint.checkpoint(
                evoformer_function,
                single.detach(),  # Detach inputs to prevent double backward
                pair.detach(),
                preserve_rng_state=False,  # Disable RNG state preservation for better memory efficiency
                use_reentrant=False  # Critical: disable reentrant backwards
            )
        else:
            return self.evmodel(single, pair)
        
    

    def calculate_energies(self, headers, representations, batch):
        """Calculate energies for sequences"""
        energies = {}
        device = next(self.parameters()).device
        
        # Get logits from heads
        seq_head_outputs = headers.get('seqhead', {})
        pair_head_outputs = headers.get('pairhead', {})
        seq_logits = seq_head_outputs.get('logits')
        pair_logits = pair_head_outputs.get('logits')
        labels = batch.get('aatype')
        
        # Print shapes for debugging
        print("\nDebug shapes in energy calculation:")
        print(f"seq_logits shape: {seq_logits.shape if seq_logits is not None else 'None'}")
        print(f"labels shape: {labels.shape if labels is not None else 'None'}")
        
        if seq_logits is not None and labels is not None:
            # Handle various possible shapes of seq_logits
            if seq_logits.dim() == 4:  # [batch, extra_dim, seq_len, n_tokens]
                seq_logits = seq_logits.squeeze(1)  # Remove extra dimension
            
            # Ensure seq_logits is [batch, seq_len, n_tokens]
            if seq_logits.dim() != 3:
                print(f"Warning: Unexpected seq_logits shape: {seq_logits.shape}")
                return {
                    'single': torch.tensor(0.0, device=device, requires_grad=True),
                    'pair': torch.tensor(0.0, device=device, requires_grad=True),
                    'total': torch.tensor(0.0, device=device, requires_grad=True)
                }
                
            # Calculate single-site energies
            log_probs = F.log_softmax(seq_logits, dim=-1)  # [batch, seq_len, n_tokens]
            valid_len = min(log_probs.shape[1], labels.shape[1])  # Use the shorter sequence length
            
            # Truncate or pad tensors to valid length
            log_probs = log_probs[:, :valid_len, :]
            labels_valid = labels[:, :valid_len]
            
            seq_energy = -torch.gather(
                log_probs,
                -1,
                labels_valid.unsqueeze(-1)
            ).squeeze(-1)  # [batch, valid_len]
            
            # Sum over sequence positions
            #energies['single'] = seq_energy.sum(-1) / valid_len
            energies['single'] = seq_energy.sum(-1)
        else:
            energies['single'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        if pair_logits is not None and labels is not None:
            batch_size, seq_len = labels.shape
            n_classes = seq_logits.shape[-1]
            print(f"pair_logits shape before reshape: {pair_logits.shape}")
            # Reshape pair logits if needed
            if pair_logits.dim() == 4:  # [batch, seq_len, seq_len, n_classes*n_classes]
                try:
                    pair_logits = pair_logits.view(batch_size, seq_len, seq_len, n_classes, n_classes)
                except RuntimeError:
                    print(f"Warning: Could not reshape pair_logits from {pair_logits.shape}")
                    energies['pair'] = torch.tensor(0.0, device=device, requires_grad=True)
                    energies['total'] = energies['single'] + energies['pair']
                    return energies
            
            # Use valid length for pair calculations as well
            valid_len = min(pair_logits.shape[1], labels.shape[1])
            pair_logits = pair_logits[:, :valid_len, :valid_len]
            labels_valid = labels[:, :valid_len]
            
            # Get labels for both axes
            labels_i = labels_valid.view(batch_size, valid_len, 1, 1, 1)
            labels_j = labels_valid.view(batch_size, 1, valid_len, 1, 1)
            
            pair_log_probs = F.log_softmax(pair_logits.view(-1, n_classes * n_classes), dim=-1).view(batch_size, valid_len, valid_len, n_classes, n_classes)

            # Calculate pair energies
            pair_energy = -torch.gather(
                torch.gather(
                    pair_log_probs,
                    3,
                    labels_i.expand(-1, -1, valid_len, 1, n_classes)
                ),
                4,
                labels_j.expand(-1, valid_len, -1, 1, 1)
            ).squeeze(-1).squeeze(-1)  # [batch, valid_len, valid_len]
            
            #valid_pairs = valid_len * (valid_len - 1) / 2  
            energies['pair'] = pair_energy.sum((-2, -1)) / 2  # Only divide by 2 because we count each pair twice
            # Sum over all pairs and divide by 2 (since we count each pair twice)
            #energies['pair'] = pair_energy.sum((-2, -1)) / 2  # [batch]
            #energies['pair'] = pair_energy.sum((-2, -1)) / (2 * valid_pairs)
        else:
            energies['pair'] = torch.tensor(0.0, device=device, requires_grad=True)
            
        # Calculate total energy
        energies['total'] = energies['single'] + energies['pair']
        
        return energies


    def forward(self, batch, compute_loss=True, compute_energy=True):
        # Input shapes
        print("\n=== Input Shapes and Data ===")
        mhc_encoded = batch['mhc_encoded']
        l_v_encoded = batch['l_v_encoded']
        l_j_encoded = batch['l_j_encoded']
        h_v_encoded = batch['h_v_encoded']
        cdr3_b_encoded = batch['cdr3_b_encoded']
        h_j_encoded = batch['h_j_encoded']
        
        # Dimensions
        batch_size = mhc_encoded.shape[0]
        mhc_len = mhc_encoded.shape[1]
        l_v_len = l_v_encoded.shape[1]
        l_j_len = l_j_encoded.shape[1]
        h_v_len = h_v_encoded.shape[1]
        cdr3_b_len = cdr3_b_encoded.shape[1]
        h_j_len = h_j_encoded.shape[1]
        # Change this line to use epitope_lengths
        epitope_len = batch['epitope_lengths'][0]  # Since all lengths in batch are same
        
        # Type tensors
        device = mhc_encoded.device
        mhc_type = torch.zeros((batch_size, mhc_len), device=device)
        l_v_type = torch.ones((batch_size, l_v_len), device=device)
        l_j_type = torch.full((batch_size, l_j_len), 2, device=device)
        h_v_type = torch.full((batch_size, h_v_len), 3, device=device)
        cdr3_b_type = torch.full((batch_size, cdr3_b_len), 4, device=device)
        h_j_type = torch.full((batch_size, h_j_len), 5, device=device)
        epitope_type = torch.full((batch_size, epitope_len), 6, device=device)
        
        # Initialize empty sequences
        mhc_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        l_v_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        l_j_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        h_v_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        cdr3_b_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        h_j_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        
        # Process MHC if present
        if mhc_len > 0:
            mhc_type_emb = self.type_embedding(mhc_type.long())
            mhc_pos_enc = self.get_sinusoidal_pos_encoding(batch['mhc_ri'])
            mhc_single = self.linear_mhc_m(mhc_encoded) + mhc_type_emb + mhc_pos_enc
        
        # Process TCR components
        if l_v_len > 0:
            l_v_type_emb = self.type_embedding(l_v_type.long())
            l_v_pos_enc = self.get_sinusoidal_pos_encoding(batch['l_v_ri'])
            l_v_single = self.linear_tcr_m(l_v_encoded) + l_v_type_emb + l_v_pos_enc
            
        if l_j_len > 0:
            l_j_type_emb = self.type_embedding(l_j_type.long())
            l_j_pos_enc = self.get_sinusoidal_pos_encoding(batch['l_j_ri'])
            l_j_single = self.linear_tcr_m(l_j_encoded) + l_j_type_emb + l_j_pos_enc
            
        if h_v_len > 0:
            h_v_type_emb = self.type_embedding(h_v_type.long())
            h_v_pos_enc = self.get_sinusoidal_pos_encoding(batch['h_v_ri'])
            h_v_single = self.linear_tcr_m(h_v_encoded) + h_v_type_emb + h_v_pos_enc
            
        if cdr3_b_len > 0:
            cdr3_b_type_emb = self.type_embedding(cdr3_b_type.long())
            cdr3_b_pos_enc = self.get_sinusoidal_pos_encoding(batch['cdr3_b_ri'])
            cdr3_b_single = self.linear_tcr_m(cdr3_b_encoded) + cdr3_b_type_emb + cdr3_b_pos_enc
            
        if h_j_len > 0:
            h_j_type_emb = self.type_embedding(h_j_type.long())
            h_j_pos_enc = self.get_sinusoidal_pos_encoding(batch['h_j_ri'])
            h_j_single = self.linear_tcr_m(h_j_encoded) + h_j_type_emb + h_j_pos_enc
        
        # Process epitope position encoding only
        epitope_type_emb = self.type_embedding(epitope_type.long())
        epitope_pos_enc = self.get_sinusoidal_pos_encoding(batch['epitope_ri'])
        # Initialize epitope with only positional and type information
        epitope_single = epitope_type_emb + epitope_pos_enc
        
        # Combine sequences
        combined_singles = []
        combined_ris = []
        combined_types = []
        combined_encodeds = []
        
        # Add components in order
        components = [
            (mhc_len, mhc_single, batch['mhc_ri'], mhc_type, mhc_encoded),
            (l_v_len, l_v_single, batch['l_v_ri'], l_v_type, l_v_encoded),
            (l_j_len, l_j_single, batch['l_j_ri'], l_j_type, l_j_encoded),
            (h_v_len, h_v_single, batch['h_v_ri'], h_v_type, h_v_encoded),
            (cdr3_b_len, cdr3_b_single, batch['cdr3_b_ri'], cdr3_b_type, cdr3_b_encoded),
            (h_j_len, h_j_single, batch['h_j_ri'], h_j_type, h_j_encoded),
            (epitope_len, epitope_single, batch['epitope_ri'], epitope_type, 
            torch.zeros(batch_size, epitope_len, self.n_channels_seq, device=device))
        ]
        

        for length, single, ri, type_tensor, encoded in components:
            if length > 0:
                combined_singles.append(single)
                combined_ris.append(ri)
                combined_types.append(type_tensor)
                combined_encodeds.append(encoded)
        
        # Concatenate all sequences
        combined_single = torch.cat(combined_singles, dim=1)
        combined_ri = torch.cat(combined_ris, dim=1)
        combined_types = torch.cat(combined_types, dim=1)
        combined_encoded = torch.cat(combined_encodeds, dim=1)
        
        print(f"\nCombined shapes:")
        print(f"combined_single: {combined_single.shape}")
        print(f"combined_ri: {combined_ri.shape}")
        print(f"combined_types: {combined_types.shape}")
        
        # Initialize combined pair representation
        total_len = combined_single.shape[1]
        combined_pair = torch.zeros(
            batch_size, total_len, total_len, self.pair_channel, 
            device=device
        )
        
        # Process pair positions for each component
        current_idx = 0
        for length, _, _, _, _ in components:
            if length > 0:
                end_idx = current_idx + length
                component_pos = self.relpos(combined_ri, current_idx, end_idx)
                combined_pair[:, current_idx:end_idx, current_idx:end_idx] = component_pos
                current_idx = end_idx
        
        # Pair representations
        print(f"\nPair representation input shape: {combined_encoded.shape}")
        combined_z_i = self.linear_combined_z_i(combined_encoded)
        combined_z_j = self.linear_combined_z_j(combined_encoded)
        print(f"After linear transformations:")
        print(f"combined_z_i: {combined_z_i.shape}")
        print(f"combined_z_j: {combined_z_j.shape}")
        
        # Type pair embeddings
        type_pair_embeddings = self._create_pair_type_embeddings(combined_types)
        print(f"\nType pair embeddings: {type_pair_embeddings.shape}")
        
        # Add all information to pair representation
        combined_pair = combined_pair + type_pair_embeddings
        combined_pair = self.add(combined_pair, combined_z_i[..., None, :])
        combined_pair = self.add(combined_pair, combined_z_j[..., None, :, :])
        print(f"\nFinal combined pair shape: {combined_pair.shape}")
        
        combined_single = self.linear_single_projection(combined_single)
        combined_single = combined_single.unsqueeze(1)
        combined_pair = self.linear_pair_projection(combined_pair)

        
        # Evoformer
        print("\n=== Evoformer Input/Output ===")
        print(f"Evoformer input shapes:")
        print(f"single: {combined_single.shape}")
        print(f"pair: {combined_pair.shape}")
        
        single_out, pair_out = self._run_evoformer(combined_single, combined_pair)
        
        print(f"\nEvoformer output shapes:")
        print(f"single_out: {single_out.shape}")
        print(f"pair_out: {pair_out.shape}")
        
        # Calculate epitope position
        epitope_start = sum(length for length, _, _, _, _ in components[:-1] if length > 0)
        epitope_end = epitope_start + epitope_len
        
        # Debug shapes
        print(f"\nShape debugging before slicing:")
        print(f"single_out shape: {single_out.shape}")
        print(f"epitope_start: {epitope_start}, epitope_end: {epitope_end}")
        
        # Handle single_out slicing
        if single_out.dim() == 4:
            epitope_single = single_out[:, :, epitope_start:epitope_end, :]
        elif single_out.dim() == 3:
            epitope_single = single_out[:, epitope_start:epitope_end, :]
        else:
            raise ValueError(f"Unexpected single_out shape: {single_out.shape}")
        
        # Get pair representation
        epitope_pair = pair_out[:, epitope_start:epitope_end, epitope_start:epitope_end]
        
        print(f"\nShape debugging after slicing:")
        print(f"epitope_single shape: {epitope_single.shape}")
        print(f"epitope_pair shape: {epitope_pair.shape}")
        
        representations = {
            'seq': epitope_single,
            'pair': epitope_pair
        }
        
        # Process through heads
        outputs = {}
        headers = {}
        for head_name, head in self.heads.items():
            print(f"Processing head: {head_name}")
            out = head(headers, representations, batch)
            if out is not None:
                headers[head_name] = out
                print(f"{head_name} output added to headers.")
                if 'logits' in out:
                    outputs[f'{head_name}_pred'] = out['logits']
                    if head_name == 'seqhead':
                        outputs['pred'] = out['logits']
        
        if compute_loss and self.loss_head is not None:
            losses = self.loss_head(headers, representations, batch)
            outputs['losses'] = losses
            
        if compute_energy:
            energies = self.calculate_energies(headers, representations, batch)
            outputs['energies'] = energies
            print("\nEnergy values:")
            for k, v in energies.items():
                print(f"{k}: {v.item():.4f}")
        
        return outputs
