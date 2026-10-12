"""GRIP-CLA heads and the two reference energy-normalization profiles."""

import torch
from torch import nn

from .energy import BaseEnergyModel


class EnergyNormalization(nn.Module):
    """Normalize single-site and pair energies by peptide length."""

    def __init__(self):
        super().__init__()
        # Retain the parameter name present in the fine-tuned checkpoints.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, energies, batch):
        lengths = batch["epitope_lengths"].float()
        single = energies["single"] / lengths
        pair = energies["pair"] / (lengths * lengths)
        return {"single": single, "pair": pair, "total": single + pair}


class ClassificationHead(nn.Module):
    """The trained 465-feature classification head."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(465, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, energies, seq_logits, pair_logits):
        if seq_logits.ndim == 4:
            seq_logits = seq_logits.squeeze(1)
        energy_features = torch.stack(
            [energies["single"], energies["pair"], energies["total"]], dim=-1
        )
        sequence_features = seq_logits.mean(dim=1)
        if pair_logits.shape[-1] == 441:
            pair_features = pair_logits.mean(dim=(1, 2))
        else:
            pair_features = pair_logits.view(seq_logits.shape[0], -1, 441).mean(dim=1)
        features = torch.cat(
            [energy_features, sequence_features, pair_features], dim=-1
        )
        return self.classifier(self.dropout(features)).squeeze(-1)


class ClassifierModel(BaseEnergyModel):
    """GRIP-CLA with a complete fine-tuned checkpoint-compatible architecture.

    ``profile='mhc'`` applies the MHC-I/MHC-II reference normalization once:
    single-site energy divided by L and pair energy divided by L squared.
    ``profile='tcr_legacy'`` preserves the two successive normalizations used
    in the original TCR benchmark. Neither profile selects weights or converts
    allele names to sequences.

    In evaluation mode, ``forward`` returns the base outputs (including raw
    ``energies``), the final head inputs in ``normalized_energies``, and
    ``logit`` and ``score`` tensors of shape [B]. ``score`` is sigmoid(logit).
    Reference evaluations use one record per batch, avoiding context padding.
    The historical classification training objectives are not implemented here.
    """

    def __init__(self, config, *, profile):
        if profile not in ("mhc", "tcr_legacy"):
            raise ValueError("profile must be 'mhc' or 'tcr_legacy'")
        super().__init__(config)
        self.profile = profile
        self.energy_head = EnergyNormalization()
        self.classification_head = ClassificationHead()

    def forward(self, batch):
        outputs = super().forward(batch, compute_loss=False, compute_energy=True)
        normalized = self.energy_head(outputs["energies"], batch)
        if self.profile == "tcr_legacy":
            normalized = self.energy_head(normalized, batch)
        logit = self.classification_head(
            normalized, outputs["seqhead_pred"], outputs["pairhead_pred"]
        )
        outputs.update(
            normalized_energies=normalized, logit=logit, score=torch.sigmoid(logit)
        )
        return outputs
