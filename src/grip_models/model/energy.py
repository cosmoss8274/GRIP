"""Core sequence energy network."""

import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

from .evoformer import Evoformer
from .heads import HeaderBuilder


logger = logging.getLogger(__name__)


class BaseEnergyModel(nn.Module):
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
        logger.debug("\nDebug shapes in energy calculation:")
        logger.debug(f"seq_logits shape: {seq_logits.shape if seq_logits is not None else 'None'}")
        logger.debug(f"labels shape: {labels.shape if labels is not None else 'None'}")

        if seq_logits is not None and labels is not None:
            # Handle various possible shapes of seq_logits
            if seq_logits.dim() == 4:  # [batch, extra_dim, seq_len, n_tokens]
                seq_logits = seq_logits.squeeze(1)  # Remove extra dimension

            # Ensure seq_logits is [batch, seq_len, n_tokens]
            if seq_logits.dim() != 3:
                logger.debug(f"Warning: Unexpected seq_logits shape: {seq_logits.shape}")
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
            logger.debug(f"pair_logits shape before reshape: {pair_logits.shape}")
            # Reshape pair logits if needed
            if pair_logits.dim() == 4:  # [batch, seq_len, seq_len, n_classes*n_classes]
                try:
                    pair_logits = pair_logits.view(batch_size, seq_len, seq_len, n_classes, n_classes)
                except RuntimeError:
                    logger.debug(f"Warning: Could not reshape pair_logits from {pair_logits.shape}")
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
        logger.debug("\n=== Input Shapes and Data ===")
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

        logger.debug(f"\nCombined shapes:")
        logger.debug(f"combined_single: {combined_single.shape}")
        logger.debug(f"combined_ri: {combined_ri.shape}")
        logger.debug(f"combined_types: {combined_types.shape}")

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
        logger.debug(f"\nPair representation input shape: {combined_encoded.shape}")
        combined_z_i = self.linear_combined_z_i(combined_encoded)
        combined_z_j = self.linear_combined_z_j(combined_encoded)
        logger.debug(f"After linear transformations:")
        logger.debug(f"combined_z_i: {combined_z_i.shape}")
        logger.debug(f"combined_z_j: {combined_z_j.shape}")

        # Type pair embeddings
        type_pair_embeddings = self._create_pair_type_embeddings(combined_types)
        logger.debug(f"\nType pair embeddings: {type_pair_embeddings.shape}")

        # Add all information to pair representation
        combined_pair = combined_pair + type_pair_embeddings
        combined_pair = self.add(combined_pair, combined_z_i[..., None, :])
        combined_pair = self.add(combined_pair, combined_z_j[..., None, :, :])
        logger.debug(f"\nFinal combined pair shape: {combined_pair.shape}")

        combined_single = self.linear_single_projection(combined_single)
        combined_single = combined_single.unsqueeze(1)
        combined_pair = self.linear_pair_projection(combined_pair)


        # Evoformer
        logger.debug("\n=== Evoformer Input/Output ===")
        logger.debug(f"Evoformer input shapes:")
        logger.debug(f"single: {combined_single.shape}")
        logger.debug(f"pair: {combined_pair.shape}")

        single_out, pair_out = self._run_evoformer(combined_single, combined_pair)

        logger.debug(f"\nEvoformer output shapes:")
        logger.debug(f"single_out: {single_out.shape}")
        logger.debug(f"pair_out: {pair_out.shape}")

        # Calculate epitope position
        epitope_start = sum(length for length, _, _, _, _ in components[:-1] if length > 0)
        epitope_end = epitope_start + epitope_len

        # Debug shapes
        logger.debug(f"\nShape debugging before slicing:")
        logger.debug(f"single_out shape: {single_out.shape}")
        logger.debug(f"epitope_start: {epitope_start}, epitope_end: {epitope_end}")

        # Handle single_out slicing
        if single_out.dim() == 4:
            epitope_single = single_out[:, :, epitope_start:epitope_end, :]
        elif single_out.dim() == 3:
            epitope_single = single_out[:, epitope_start:epitope_end, :]
        else:
            raise ValueError(f"Unexpected single_out shape: {single_out.shape}")

        # Get pair representation
        epitope_pair = pair_out[:, epitope_start:epitope_end, epitope_start:epitope_end]

        logger.debug(f"\nShape debugging after slicing:")
        logger.debug(f"epitope_single shape: {epitope_single.shape}")
        logger.debug(f"epitope_pair shape: {epitope_pair.shape}")

        representations = {
            'seq': epitope_single,
            'pair': epitope_pair
        }

        # Process through heads
        outputs = {}
        headers = {}
        for head_name, head in self.heads.items():
            logger.debug(f"Processing head: {head_name}")
            out = head(headers, representations, batch)
            if out is not None:
                headers[head_name] = out
                logger.debug(f"{head_name} output added to headers.")
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
            logger.debug("\nEnergy values:")
            for k, v in energies.items():
                logger.debug(f"{k}: {v.item():.4f}")

        return outputs
