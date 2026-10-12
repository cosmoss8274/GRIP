"""Single-residue, pairwise and loss heads for the energy model."""

from collections import OrderedDict
import functools
import logging

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from ..data import restypes_with_x
from .layers import LayerNorm


logger = logging.getLogger(__name__)


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
            logger.debug("Error: 'seq' not in representations!")
            return None
        logger.debug(f"Linear weights (proj): {self.proj.weight}")
        logger.debug(f"Linear bias (proj): {self.proj.bias}")

        
        x = representations['seq']
        logger.debug(f"SeqHead input shape: {x.shape}")
        x = self.norm(x)
        logger.debug(f"Normed_x first 5 values: {x[..., :5]}")
        logits = self.proj(x)
        logger.debug(f"logits first 5 values: {logits}")
        logger.debug(f"SeqHead logits shape: {logits.shape}")
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
        logger.debug(f"Initial logits shape: {logits.shape}")  
        logits = rearrange(logits, 'b i j (a c) -> b i j a c', a=self.num_token)
        logger.debug(f"After first rearrange: {logits.shape}")
        logits = rearrange(0.5 * (logits + rearrange(logits, 'b i j a c -> b j i c a')), 'b i j a c -> b i j (a c)')
        #logits = 0.5 * (logits + rearrange(logits, 'b i j a c -> b j i c a'))
        logger.debug(f"Final logits shape: {logits.shape}")
        logger.debug(f"logits pair values: {logits}") 
        return dict(logits=logits)

class LossHead(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        self.l1_weight = getattr(config, 'l1_weight', 0.01)  # For regularization
        self.l2_weight = getattr(config, 'l2_weight', 0.01)  # For regularization

    def _compute_single_loss(self, logits, labels):
        """Compute single sequence conservation loss hi(xi|si)."""
        logger.debug("\n=== _compute_single_loss ===")
        logger.debug(f"logits shape: {logits.shape}")
        logger.debug(f"labels shape: {labels.shape}")
        
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
        logger.debug(f"Single loss: {loss.item()}")
        return loss

    def compute_partition_function(self, seq_logits, pair_logits, labels, num_residues, batch_size):
        num_classes = seq_logits.shape[-1]
        device = seq_logits.device
        
        logger.debug("\nPartition Function Debug:")
        logger.debug(f"seq_logits shape: {seq_logits.shape}")  # [B, N, C]
        logger.debug(f"pair_logits initial shape: {pair_logits.shape}")  # [B, N, N, C, C]

        # Get all possible amino acid combinations
        a_poss = torch.arange(num_classes, device=device)
        i_amino, j_amino = torch.meshgrid(a_poss, a_poss, indexing='ij')
        i_amino = i_amino.reshape(-1)  # [C*C]
        j_amino = j_amino.reshape(-1)  # [C*C]
        logger.debug(f"Number of possible amino acid pairs: {len(i_amino)}")  # C*C

        # 1. Single site terms for all possible amino acids
        hi_terms = seq_logits[:, :, i_amino].unsqueeze(2)  # [B, N, 1, C*C]
        hj_terms = seq_logits[:, :, j_amino].unsqueeze(1)  # [B, 1, N, C*C]
        
        logger.debug(f"hi_terms shape: {hi_terms.shape}")
        logger.debug(f"hi_terms range: [{hi_terms.min():.4f}, {hi_terms.max():.4f}]")
        logger.debug(f"hj_terms shape: {hj_terms.shape}")
        logger.debug(f"hj_terms range: [{hj_terms.min():.4f}, {hj_terms.max():.4f}]")

        # 2. Direct coupling terms for all possible amino acid pairs
        eij_terms = pair_logits[
            torch.arange(batch_size, device=device)[:, None, None, None],
            torch.arange(num_residues, device=device)[None, :, None, None],
            torch.arange(num_residues, device=device)[None, None, :, None],
            i_amino[None, None, None, :],
            j_amino[None, None, None, :]
        ]  # [B, N, N, C*C]
        
        logger.debug(f"eij_terms shape: {eij_terms.shape}")
        logger.debug(f"eij_terms range: [{eij_terms.min():.4f}, {eij_terms.max():.4f}]")

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
        
        logger.debug(f"eik shape: {eik.shape}")
        logger.debug(f"eik range: [{eik.min():.4f}, {eik.max():.4f}]")
        logger.debug(f"ejk shape: {ejk.shape}")
        logger.debug(f"ejk range: [{ejk.min():.4f}, {ejk.max():.4f}]")

        # 4. Combine all terms
        all_terms = hi_terms + hj_terms + eij_terms + eik + ejk
        logger.debug(f"all_terms shape before logsumexp: {all_terms.shape}")
        logger.debug(f"all_terms range: [{all_terms.min():.4f}, {all_terms.max():.4f}]")

        # 5. Compute partition function
        logZ = torch.logsumexp(all_terms, dim=-1)  # [B, N, N]
        logger.debug(f"logZ final shape: {logZ.shape}")
        logger.debug(f"logZ range: [{logZ.min():.4f}, {logZ.max():.4f}]")

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

        logger.debug("\nPair Loss Debug:")
        logger.debug(f"labels shape: {labels.shape}")  # [1, 13]
        logger.debug(f"seq_logits shape: {seq_logits.shape}")  # [1, 1, 13, 21]
        logger.debug(f"pair_logits shape: {pair_logits.shape}")  # [1, 13, 13, 441]

        # Handle seq_logits dimensions first
        if seq_logits.dim() == 4:
            seq_logits = seq_logits.squeeze(1)  # Remove the extra dimension -> [1, 13, 21]
        logger.debug(f"seq_logits after squeeze: {seq_logits.shape}")

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

        logger.debug(f"hi_terms_obs shape: {hi_terms_obs.shape}")
        logger.debug(f"hi_terms_obs range: [{hi_terms_obs.min():.4f}, {hi_terms_obs.max():.4f}]")
        logger.debug(f"hj_terms_obs shape: {hj_terms_obs.shape}")
        logger.debug(f"hj_terms_obs range: [{hj_terms_obs.min():.4f}, {hj_terms_obs.max():.4f}]")

        # 3. Direct coupling terms for observed amino acid pairs
        eij_terms_obs = pair_logits_reshaped[
            torch.arange(batch_size, device=device)[:, None, None],
            torch.arange(num_residues, device=device)[None, :, None],
            torch.arange(num_residues, device=device)[None, None, :],
            i_obs.unsqueeze(-1).expand(-1, -1, num_residues),
            j_obs.unsqueeze(1).expand(-1, num_residues, -1)
        ]  # [B, N, N]
        
        logger.debug(f"eij_terms_obs shape: {eij_terms_obs.shape}")
        logger.debug(f"eij_terms_obs range: [{eij_terms_obs.min():.4f}, {eij_terms_obs.max():.4f}]")

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
        
        logger.debug(f"eik_terms_obs shape: {eik_terms_obs.shape}")
        logger.debug(f"eik_terms_obs range: [{eik_terms_obs.min():.4f}, {eik_terms_obs.max():.4f}]")
        logger.debug(f"ejk_terms_obs shape: {ejk_terms_obs.shape}")
        logger.debug(f"ejk_terms_obs range: [{ejk_terms_obs.min():.4f}, {ejk_terms_obs.max():.4f}]")

        # 5. Combine all terms for Energy
        Energy = hi_terms_obs + hj_terms_obs + eij_terms_obs + eik_terms_obs + ejk_terms_obs

        logger.debug(f"Energy final shape: {Energy.shape}")  # [B, N, N]
        logger.debug(f"Energy range: [{Energy.min():.4f}, {Energy.max():.4f}]")

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

        logger.debug("\nLoss Components:")
        logger.debug(f"pair_loss: {pair_loss.item():.4f}")
        logger.debug(f"l1_reg: {l1_reg.item():.4f}")
        logger.debug(f"l2_reg: {l2_reg.item():.4f}")
        logger.debug(f"Total loss: {(pair_loss + l1_reg + l2_reg).item():.4f}")

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
        logger.debug(f"\n=== HeaderBuilder.build ===")
        logger.debug(f"Input channels - seq: {seq_channel}, pair: {pair_channel}")

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
                logger.debug(f"Added {head_name} head to the model.")

        return heads

