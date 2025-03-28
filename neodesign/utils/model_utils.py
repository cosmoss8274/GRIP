import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

def add(x, y, inplace=False):
    """Helper method for adding tensors with optional inplace operation"""
    return x.add_(y) if inplace else x + y

def get_sinusoidal_pos_encoding(positions, num_frequencies, seq_channel, linear_abs_pos):
    """Generate sinusoidal position encoding for sequences."""
    if len(positions.shape) == 1:
        positions = positions.unsqueeze(0)
    batch_size, seq_len = positions.shape
    
    freq = torch.exp(
        torch.arange(0, num_frequencies, device=positions.device) *
        (-math.log(10000.0) / num_frequencies)
    )
    
    pos = positions.unsqueeze(-1).float()
    pos_freq = pos * freq.unsqueeze(0).unsqueeze(0)
    
    pos_enc = torch.cat([
        torch.sin(pos_freq),
        torch.cos(pos_freq)
    ], dim=-1)
    
    if seq_channel % 2 == 1:
        pos_enc = F.pad(pos_enc, (0, 1), mode='constant', value=0)
    
    return linear_abs_pos(pos_enc)

def relpos(ri, start_idx, end_idx, relpos_k, linear_relpos):
    """
    Computes relative positional encodings for a sequence segment
    """
    segment_ri = ri[..., start_idx:end_idx]  # [batch, segment_len]
    batch_size, segment_len = segment_ri.shape
    
    pos_i = segment_ri.unsqueeze(-1)  # [batch, segment_len, 1]
    pos_j = segment_ri.unsqueeze(-2)  # [batch, 1, segment_len]
    d = pos_i - pos_j  # [batch, segment_len, segment_len]
    
    boundaries = torch.arange(
        start=-relpos_k, 
        end=relpos_k + 1, 
        device=ri.device
    )
    
    d = d.unsqueeze(-1) - boundaries.view(1, 1, 1, -1)
    d = torch.abs(d)
    d = torch.argmin(d, dim=-1)
    
    d = nn.functional.one_hot(d, num_classes=2 * relpos_k + 1).float()
    d = d.to(ri.dtype)
    d = d.float()
    pos_emb = linear_relpos(d)
    
    return pos_emb

def create_pair_type_embeddings(type_tensor, num_types, pair_type_embedding):
    """Creates pairwise type embeddings."""
    type_tensor = type_tensor.long()
    batch_size, seq_len = type_tensor.shape
    
    type_i = type_tensor.unsqueeze(2)
    type_j = type_tensor.unsqueeze(1)
    pair_type_idx = type_i * num_types + type_j
    
    return pair_type_embedding(pair_type_idx)

def run_evoformer(evmodel, single, pair, use_checkpoint, training):
    """Wrapper for Evoformer forward pass to handle checkpointing"""
    def evoformer_function(s, p):
        return evmodel(s, p)
        
    if use_checkpoint and training:
        return checkpoint(
            evoformer_function,
            single.detach(),
            pair.detach(),
            preserve_rng_state=False,
            use_reentrant=False
        )
    else:
        return evmodel(single, pair)

def calculate_energies(headers, representations, batch, device=None):
    """Calculate energies for sequences"""
    energies = {}
    
    seq_head_outputs = headers.get('seqhead', {})
    pair_head_outputs = headers.get('pairhead', {})
    seq_logits = seq_head_outputs.get('logits')
    pair_logits = pair_head_outputs.get('logits')
    labels = batch.get('aatype')

    if seq_logits is not None and labels is not None:
        if seq_logits.dim() == 4:  # [batch, extra_dim, seq_len, n_tokens]
            seq_logits = seq_logits.squeeze(1)
        
        if seq_logits.dim() != 3:
            print(f"Warning: Unexpected seq_logits shape: {seq_logits.shape}")
            return {
                'single': torch.tensor(0.0, device=device, requires_grad=True),
                'pair': torch.tensor(0.0, device=device, requires_grad=True),
                'total': torch.tensor(0.0, device=device, requires_grad=True)
            }
            
        log_probs = F.log_softmax(seq_logits, dim=-1)
        valid_len = min(log_probs.shape[1], labels.shape[1])
        
        log_probs = log_probs[:, :valid_len, :]
        labels_valid = labels[:, :valid_len]
        
        seq_energy = -torch.gather(
            log_probs,
            -1,
            labels_valid.unsqueeze(-1)
        ).squeeze(-1)
        
        energies['single'] = seq_energy.sum(-1)
    else:
        energies['single'] = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Pair energy calculation
    if pair_logits is not None and labels is not None:
        batch_size, seq_len = labels.shape
        n_classes = seq_logits.shape[-1]
        
        if pair_logits.dim() == 4:
            try:
                pair_logits = pair_logits.view(batch_size, seq_len, seq_len, n_classes, n_classes)
            except RuntimeError:
                print(f"Warning: Could not reshape pair_logits from {pair_logits.shape}")
                energies['pair'] = torch.tensor(0.0, device=device, requires_grad=True)
                energies['total'] = energies['single'] + energies['pair']
                return energies
        
        valid_len = min(pair_logits.shape[1], labels.shape[1])
        pair_logits = pair_logits[:, :valid_len, :valid_len]
        labels_valid = labels[:, :valid_len]
        
        labels_i = labels_valid.view(batch_size, valid_len, 1, 1, 1)
        labels_j = labels_valid.view(batch_size, 1, valid_len, 1, 1)
        
        pair_log_probs = F.log_softmax(pair_logits.view(-1, n_classes * n_classes), dim=-1).view(batch_size, valid_len, valid_len, n_classes, n_classes)

        pair_energy = -torch.gather(
            torch.gather(
                pair_log_probs,
                3,
                labels_i.expand(-1, -1, valid_len, 1, n_classes)
            ),
            4,
            labels_j.expand(-1, valid_len, -1, 1, 1)
        ).squeeze(-1).squeeze(-1)
        
        energies['pair'] = pair_energy.sum((-2, -1)) / 2
    else:
        energies['pair'] = torch.tensor(0.0, device=device, requires_grad=True)
        
    energies['total'] = energies['single'] + energies['pair']
    
    return energies