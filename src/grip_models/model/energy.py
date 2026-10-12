"""Core sequence energy network."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .evoformer import Evoformer
from .heads import HeaderBuilder


class BaseEnergyModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_checkpoint = True
        e = config.model.embeddings
        self.heads = HeaderBuilder.build(
            config, seq_channel=128, pair_channel=64, parent=self
        )
        self.loss_head = self.heads.get("losshead", None)
        self.seq_channel = e.seq_channel
        self.pair_channel = e.pair_channel
        self.n_channels_seq = e.seq_info_dim
        self.tf_dim = e.tf_dim
        self.relpos_k = 32
        self.linear_single_projection = nn.Linear(self.seq_channel, self.seq_channel)
        self.linear_pair_projection = nn.Linear(self.pair_channel, self.pair_channel)
        self.evmodel = Evoformer(
            self.seq_channel,
            self.pair_channel,
            num_layers=getattr(config.model, "num_layers", 4),
        )
        self.num_frequencies = self.seq_channel // 2
        self.linear_abs_pos = nn.Linear(self.seq_channel, self.seq_channel)
        self.num_types = 7
        self.type_embedding = nn.Embedding(self.num_types, self.seq_channel)
        self.pair_type_embedding = nn.Embedding(
            self.num_types * self.num_types, self.pair_channel
        )
        self.linear_mhc_z_i = nn.Linear(self.n_channels_seq, self.pair_channel)
        self.linear_mhc_z_j = nn.Linear(self.n_channels_seq, self.pair_channel)
        self.linear_mhc_m = nn.Linear(self.n_channels_seq, self.seq_channel)
        self.linear_tcr_z_i = nn.Linear(self.n_channels_seq, self.pair_channel)
        self.linear_tcr_z_j = nn.Linear(self.n_channels_seq, self.pair_channel)
        self.linear_tcr_m = nn.Linear(self.n_channels_seq, self.seq_channel)
        self.linear_combined_z_i = nn.Linear(self.n_channels_seq, self.pair_channel)
        self.linear_combined_z_j = nn.Linear(self.n_channels_seq, self.pair_channel)
        self.linear_relpos = nn.Linear(2 * self.relpos_k + 1, self.pair_channel)

    def add(self, x, y, inplace=False):
        """Add tensors, optionally in place."""
        return x.add_(y) if inplace else x + y

    def get_sinusoidal_pos_encoding(self, positions):
        """Generate sinusoidal position encoding for sequences."""
        if len(positions.shape) == 1:
            positions = positions.unsqueeze(0)
        batch_size, seq_len = positions.shape
        freq = torch.exp(
            torch.arange(0, self.num_frequencies, device=positions.device)
            * (-math.log(10000.0) / self.num_frequencies)
        )
        pos = positions.unsqueeze(-1).float()
        pos_freq = pos * freq.unsqueeze(0).unsqueeze(0)
        pos_enc = torch.cat([torch.sin(pos_freq), torch.cos(pos_freq)], dim=-1)
        if self.seq_channel % 2 == 1:
            pos_enc = F.pad(pos_enc, (0, 1), mode="constant", value=0)
        return self.linear_abs_pos(pos_enc)

    def relpos(self, ri: torch.Tensor, start_idx: int, end_idx: int):
        """Embed relative positions within the segment ``ri[..., start_idx:end_idx]``."""
        segment_ri = ri[..., start_idx:end_idx]
        batch_size, segment_len = segment_ri.shape
        pos_i = segment_ri.unsqueeze(-1)
        pos_j = segment_ri.unsqueeze(-2)
        d = pos_i - pos_j
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=ri.device
        )
        d = d.unsqueeze(-1) - boundaries.view(1, 1, 1, -1)
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=2 * self.relpos_k + 1).float()
        d = d.to(ri.dtype)
        d = d.float()
        pos_emb = self.linear_relpos(d)
        return pos_emb

    def _create_pair_type_embeddings(self, type_tensor):
        """Embed ordered pairs of sequence-component types."""
        type_tensor = type_tensor.long()
        batch_size, seq_len = type_tensor.shape
        type_i = type_tensor.unsqueeze(2)
        type_j = type_tensor.unsqueeze(1)
        pair_type_idx = type_i * self.num_types + type_j
        return self.pair_type_embedding(pair_type_idx)

    def _run_evoformer(self, single, pair):
        """Run the Evoformer with optional activation checkpointing."""

        def evoformer_function(s, p):
            return self.evmodel(s, p)

        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                evoformer_function,
                single.detach(),
                pair.detach(),
                preserve_rng_state=False,
                use_reentrant=False,
            )
        else:
            return self.evmodel(single, pair)

    def calculate_energies(self, headers, representations, batch):
        """Compute single-site and joint residue-pair energies.

        E_single = -sum_i log p_i(x_i).
        E_pair = -1/2 sum_ij log p_ij(x_i, x_j), including diagonal terms."""
        seq_logits = headers["seqhead"]["logits"]
        pair_logits = headers["pairhead"]["logits"]
        labels = batch["aatype"]
        if seq_logits.ndim == 4:
            seq_logits = seq_logits.squeeze(1)
        batch_size, length, n_classes = seq_logits.shape
        if length == 0 or labels.shape != (batch_size, length):
            raise ValueError(
                "Sequence logits and labels must describe the same nonempty peptide."
            )
        if pair_logits.shape not in (
            (batch_size, length, length, n_classes * n_classes),
            (batch_size, length, length, n_classes, n_classes),
        ):
            raise ValueError(
                "Pair logits must match both peptide site axes and the residue alphabet."
            )

        log_probs = F.log_softmax(seq_logits, dim=-1)
        single = -torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
        single_energy = single.sum(-1)
        pair_log_probs = F.log_softmax(
            pair_logits.view(-1, n_classes * n_classes), dim=-1
        ).view(batch_size, length, length, n_classes, n_classes)
        labels_i = labels.view(batch_size, length, 1, 1, 1)
        labels_j = labels.view(batch_size, 1, length, 1, 1)
        pair = (
            -torch.gather(
                torch.gather(
                    pair_log_probs,
                    3,
                    labels_i.expand(-1, -1, length, 1, n_classes),
                ),
                4,
                labels_j.expand(-1, length, -1, 1, 1),
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        pair_energy = pair.sum((-2, -1)) / 2
        return {
            "single": single_energy,
            "pair": pair_energy,
            "total": single_energy + pair_energy,
        }

    def forward(self, batch, compute_loss=True, compute_energy=True):
        mhc_encoded = batch["mhc_encoded"]
        l_v_encoded = batch["l_v_encoded"]
        l_j_encoded = batch["l_j_encoded"]
        h_v_encoded = batch["h_v_encoded"]
        cdr3_b_encoded = batch["cdr3_b_encoded"]
        h_j_encoded = batch["h_j_encoded"]
        batch_size = mhc_encoded.shape[0]
        mhc_len = mhc_encoded.shape[1]
        l_v_len = l_v_encoded.shape[1]
        l_j_len = l_j_encoded.shape[1]
        h_v_len = h_v_encoded.shape[1]
        cdr3_b_len = cdr3_b_encoded.shape[1]
        h_j_len = h_j_encoded.shape[1]
        epitope_lengths = batch["epitope_lengths"]
        if epitope_lengths.numel() != batch_size or not torch.all(
            epitope_lengths == epitope_lengths[0]
        ):
            raise ValueError("A model batch must contain peptides of the same length.")
        epitope_len = epitope_lengths[0]
        device = mhc_encoded.device
        mhc_type = torch.zeros((batch_size, mhc_len), device=device)
        l_v_type = torch.ones((batch_size, l_v_len), device=device)
        l_j_type = torch.full((batch_size, l_j_len), 2, device=device)
        h_v_type = torch.full((batch_size, h_v_len), 3, device=device)
        cdr3_b_type = torch.full((batch_size, cdr3_b_len), 4, device=device)
        h_j_type = torch.full((batch_size, h_j_len), 5, device=device)
        epitope_type = torch.full((batch_size, epitope_len), 6, device=device)
        mhc_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        l_v_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        l_j_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        h_v_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        cdr3_b_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        h_j_single = torch.empty(batch_size, 0, self.seq_channel, device=device)
        if mhc_len > 0:
            mhc_type_emb = self.type_embedding(mhc_type.long())
            mhc_pos_enc = self.get_sinusoidal_pos_encoding(batch["mhc_ri"])
            mhc_single = self.linear_mhc_m(mhc_encoded) + mhc_type_emb + mhc_pos_enc
        if l_v_len > 0:
            l_v_type_emb = self.type_embedding(l_v_type.long())
            l_v_pos_enc = self.get_sinusoidal_pos_encoding(batch["l_v_ri"])
            l_v_single = self.linear_tcr_m(l_v_encoded) + l_v_type_emb + l_v_pos_enc
        if l_j_len > 0:
            l_j_type_emb = self.type_embedding(l_j_type.long())
            l_j_pos_enc = self.get_sinusoidal_pos_encoding(batch["l_j_ri"])
            l_j_single = self.linear_tcr_m(l_j_encoded) + l_j_type_emb + l_j_pos_enc
        if h_v_len > 0:
            h_v_type_emb = self.type_embedding(h_v_type.long())
            h_v_pos_enc = self.get_sinusoidal_pos_encoding(batch["h_v_ri"])
            h_v_single = self.linear_tcr_m(h_v_encoded) + h_v_type_emb + h_v_pos_enc
        if cdr3_b_len > 0:
            cdr3_b_type_emb = self.type_embedding(cdr3_b_type.long())
            cdr3_b_pos_enc = self.get_sinusoidal_pos_encoding(batch["cdr3_b_ri"])
            cdr3_b_single = (
                self.linear_tcr_m(cdr3_b_encoded) + cdr3_b_type_emb + cdr3_b_pos_enc
            )
        if h_j_len > 0:
            h_j_type_emb = self.type_embedding(h_j_type.long())
            h_j_pos_enc = self.get_sinusoidal_pos_encoding(batch["h_j_ri"])
            h_j_single = self.linear_tcr_m(h_j_encoded) + h_j_type_emb + h_j_pos_enc
        epitope_type_emb = self.type_embedding(epitope_type.long())
        epitope_pos_enc = self.get_sinusoidal_pos_encoding(batch["epitope_ri"])
        epitope_single = epitope_type_emb + epitope_pos_enc
        combined_singles = []
        combined_ris = []
        combined_types = []
        combined_encodeds = []
        components = [
            (mhc_len, mhc_single, batch["mhc_ri"], mhc_type, mhc_encoded),
            (l_v_len, l_v_single, batch["l_v_ri"], l_v_type, l_v_encoded),
            (l_j_len, l_j_single, batch["l_j_ri"], l_j_type, l_j_encoded),
            (h_v_len, h_v_single, batch["h_v_ri"], h_v_type, h_v_encoded),
            (
                cdr3_b_len,
                cdr3_b_single,
                batch["cdr3_b_ri"],
                cdr3_b_type,
                cdr3_b_encoded,
            ),
            (h_j_len, h_j_single, batch["h_j_ri"], h_j_type, h_j_encoded),
            (
                epitope_len,
                epitope_single,
                batch["epitope_ri"],
                epitope_type,
                torch.zeros(
                    batch_size, epitope_len, self.n_channels_seq, device=device
                ),
            ),
        ]
        for length, single, ri, type_tensor, encoded in components:
            if length > 0:
                combined_singles.append(single)
                combined_ris.append(ri)
                combined_types.append(type_tensor)
                combined_encodeds.append(encoded)
        combined_single = torch.cat(combined_singles, dim=1)
        combined_ri = torch.cat(combined_ris, dim=1)
        combined_types = torch.cat(combined_types, dim=1)
        combined_encoded = torch.cat(combined_encodeds, dim=1)
        total_len = combined_single.shape[1]
        combined_pair = torch.zeros(
            batch_size, total_len, total_len, self.pair_channel, device=device
        )
        current_idx = 0
        for length, _, _, _, _ in components:
            if length > 0:
                end_idx = current_idx + length
                component_pos = self.relpos(combined_ri, current_idx, end_idx)
                combined_pair[:, current_idx:end_idx, current_idx:end_idx] = (
                    component_pos
                )
                current_idx = end_idx
        combined_z_i = self.linear_combined_z_i(combined_encoded)
        combined_z_j = self.linear_combined_z_j(combined_encoded)
        type_pair_embeddings = self._create_pair_type_embeddings(combined_types)
        combined_pair = combined_pair + type_pair_embeddings
        combined_pair = self.add(combined_pair, combined_z_i[..., None, :])
        combined_pair = self.add(combined_pair, combined_z_j[..., None, :, :])
        combined_single = self.linear_single_projection(combined_single)
        combined_single = combined_single.unsqueeze(1)
        combined_pair = self.linear_pair_projection(combined_pair)
        single_out, pair_out = self._run_evoformer(combined_single, combined_pair)
        epitope_start = sum(
            (length for length, _, _, _, _ in components[:-1] if length > 0)
        )
        epitope_end = epitope_start + epitope_len
        if single_out.dim() == 4:
            epitope_single = single_out[:, :, epitope_start:epitope_end, :]
        elif single_out.dim() == 3:
            epitope_single = single_out[:, epitope_start:epitope_end, :]
        else:
            raise ValueError(f"Unexpected single_out shape: {single_out.shape}")
        epitope_pair = pair_out[:, epitope_start:epitope_end, epitope_start:epitope_end]
        representations = {"seq": epitope_single, "pair": epitope_pair}
        outputs = {}
        headers = {}
        for head_name, head in self.heads.items():
            if head_name == "losshead":
                continue
            out = head(headers, representations, batch)
            if out is not None:
                headers[head_name] = out
                if "logits" in out:
                    outputs[f"{head_name}_pred"] = out["logits"]
                    if head_name == "seqhead":
                        outputs["pred"] = out["logits"]
        if compute_loss and self.loss_head is not None:
            losses = self.loss_head(headers, representations, batch)
            outputs["losses"] = losses
        if compute_energy:
            energies = self.calculate_energies(headers, representations, batch)
            outputs["energies"] = energies
        return outputs
