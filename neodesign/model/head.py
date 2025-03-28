import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from neodesign.utils.constant import RESTYPES_WITH_X
from neodesign.model.common_modules import LayerNorm, Linear

class SeqHead(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config
        
        self.norm = LayerNorm(num_in_channel)
        self.proj = nn.Linear(num_in_channel, len(RESTYPES_WITH_X), bias=True)
        self.config = config
        
    def forward(self, headers, representations, batch):
        if 'seq' not in representations:
            print("Error: 'seq' not in representations!")
            return None
            
        x = representations['seq']
        x = self.norm(x)
        logits = self.proj(x)
        return dict(logits=logits)


class PairHead(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config
        self.num_token = len(RESTYPES_WITH_X)
        self.norm = LayerNorm(num_in_channel)
        self.proj = nn.Linear(num_in_channel, self.num_token * self.num_token, bias=True)
        self.config = config
        
    def forward(self, headers, representations, batch):
        x = representations['pair']
        x = self.norm(x)
        logits = self.proj(x)
        logits = rearrange(logits, 'b i j (a c) -> b i j a c', a=self.num_token)
        logits = rearrange(0.5 * (logits + rearrange(logits, 'b i j a c -> b j i c a')), 'b i j a c -> b i j (a c)')
        return dict(logits=logits)


class LossHead(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        self.l1_weight = getattr(config, 'l1_weight', 0.01)
        self.l2_weight = getattr(config, 'l2_weight', 0.01)

    def _compute_single_loss(self, logits, labels):
        """Compute single sequence conservation loss hi(xi|si)."""
        # Direct cross entropy for single residue prediction
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction='sum'
        )
        return loss

    def compute_partition_function(self, seq_logits, pair_logits, labels, num_residues, batch_size):
        num_classes = seq_logits.shape[-1]
        device = seq_logits.device
        
        # Get all possible aa combinations
        a_poss = torch.arange(num_classes, device=device)
        i_amino, j_amino = torch.meshgrid(a_poss, a_poss, indexing='ij')
        i_amino = i_amino.reshape(-1)  # [C*C]
        j_amino = j_amino.reshape(-1)  # [C*C]

        # 1. Single site terms for all possible aa
        hi_terms = seq_logits[:, :, i_amino].unsqueeze(2)  # [B, N, 1, C*C]
        hj_terms = seq_logits[:, :, j_amino].unsqueeze(1)  # [B, 1, N, C*C]

        # 2. coupling terms for all possible aa pairs
        eij_terms = pair_logits[
            torch.arange(batch_size, device=device)[:, None, None, None],
            torch.arange(num_residues, device=device)[None, :, None, None],
            torch.arange(num_residues, device=device)[None, None, :, None],
            i_amino[None, None, None, :],
            j_amino[None, None, None, :]
        ]  # [B, N, N, C*C]

        # 3. Coupling terms with observed amino acids at position k
        eik_list = []
        ejk_list = []
        for k in range(num_residues):
            xk = labels[:, k:k+1]  # [B, 1]
            # Get terms for position k with all possible aa i/j
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

        all_terms = hi_terms + hj_terms + eij_terms + eik + ejk

        # 4. Compute partition function
        logZ = torch.logsumexp(all_terms, dim=-1)  # [B, N, N]
        return logZ

    def _compute_pair_loss(self, pair_logits, seq_logits, labels, z_matrix=None):
        batch_size, num_residues = labels.shape
        num_classes = seq_logits.shape[-1]
        device = pair_logits.device

        if seq_logits.dim() == 4:
            seq_logits = seq_logits.squeeze(1)

        # Upper triangular mask
        pair_mask = torch.triu(torch.ones(num_residues, num_residues), diagonal=1).bool()
        pair_mask = pair_mask.to(device)[None, :, :]
        pair_logits_reshaped = pair_logits.view(batch_size, num_residues, num_residues, num_classes, num_classes)

        # ENERGY CALCULATION
        i_obs = labels
        j_obs = labels
        hi_terms_obs = torch.gather(seq_logits, -1, i_obs.unsqueeze(-1))
        hj_terms_obs = torch.gather(seq_logits, -1, j_obs.unsqueeze(-1))

        hi_terms_obs = hi_terms_obs.squeeze(-1).unsqueeze(2)
        hj_terms_obs = hj_terms_obs.squeeze(-1).unsqueeze(1)

        eij_terms_obs = pair_logits_reshaped[
            torch.arange(batch_size, device=device)[:, None, None],
            torch.arange(num_residues, device=device)[None, :, None],
            torch.arange(num_residues, device=device)[None, None, :],
            i_obs.unsqueeze(-1).expand(-1, -1, num_residues),
            j_obs.unsqueeze(1).expand(-1, num_residues, -1)
        ]  # [B, N, N]

        eik_obs_list = []
        ejk_obs_list = []
        for k in range(num_residues):
            xk = labels[:, k:k+1]  # [B, 1]
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

        # 5. Combine all terms for Energy
        Energy = hi_terms_obs + hj_terms_obs + eij_terms_obs + eik_terms_obs + ejk_terms_obs

        # Create pair targets
        pair_targets = i_obs.unsqueeze(2) * num_classes + j_obs.unsqueeze(1)  # [B, N, N]

        pair_mask_expanded = pair_mask.unsqueeze(-1).expand(-1, -1, -1, num_classes * num_classes)
        valid_logits = pair_logits[pair_mask_expanded].reshape(-1, num_classes * num_classes)
        valid_targets = pair_targets[pair_mask].reshape(-1)

        # cross entropy loss
        pair_loss = F.cross_entropy(
            valid_logits,  # [num_valid_pairs, 441]
            valid_targets.long(),  # [num_valid_pairs]
            reduction='sum'
        ) / (batch_size * 2)  # only use upper triangular

        l1_reg = self.l1_weight * pair_logits.abs().mean()
        l2_reg = self.l2_weight * (pair_logits ** 2).mean()

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


class HeaderBuilder:
    @staticmethod
    def build(config, seq_channel, pair_channel, parent):
        import functools
        from collections import OrderedDict
        
        heads_config = config.model.heads
        
        head_factory = OrderedDict(
            seqhead=functools.partial(SeqHead, num_in_channel=seq_channel),
            pairhead=functools.partial(PairHead, num_in_channel=pair_channel),
            losshead=functools.partial(LossHead, num_in_channel=seq_channel)
        )

        heads = {}
        for head_name, h in head_factory.items():
            if hasattr(heads_config, head_name):
                head_config = getattr(heads_config, head_name)
                head = h(config=head_config)

                if isinstance(parent, nn.Module):
                    parent.add_module(head_name, head)

                heads[head_name] = head

        return heads