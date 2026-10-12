"""Single-residue, pairwise and loss heads for the energy model."""

from collections import OrderedDict
import functools
from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
from ..data import restypes_with_x
from .layers import LayerNorm


class SeqHead(nn.Module):

    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config
        self.norm = LayerNorm(num_in_channel)
        self.proj = nn.Linear(num_in_channel, len(restypes_with_x), bias=True)
        self.config = config

    def forward(self, headers, representations, batch):
        x = representations["seq"]
        x = self.norm(x)
        logits = self.proj(x)
        return dict(logits=logits)


class PairHead(nn.Module):

    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config
        self.num_token = len(restypes_with_x)
        self.norm = LayerNorm(num_in_channel)
        self.proj = nn.Linear(
            num_in_channel, self.num_token * self.num_token, bias=True
        )
        self.config = config

    def forward(self, headers, representations, batch):
        x = representations["pair"]
        x = self.norm(x)
        logits = self.proj(x)
        logits = rearrange(logits, "b i j (a c) -> b i j a c", a=self.num_token)
        logits = rearrange(
            0.5 * (logits + rearrange(logits, "b i j a c -> b j i c a")),
            "b i j a c -> b i j (a c)",
        )
        return dict(logits=logits)


class LossHead(nn.Module):

    def __init__(self, config, num_in_channel):
        super().__init__()
        self.l1_weight = getattr(config, "l1_weight", 0.01)
        self.l2_weight = getattr(config, "l2_weight", 0.01)

    def _compute_single_loss(self, logits, labels):
        """Compute single sequence conservation loss hi(xi|si)."""
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"
        )
        return loss

    def _compute_pair_loss(self, pair_logits, seq_logits, labels, z_matrix=None):
        """Upper-triangle pair cross-entropy and the trained regularizers."""
        batch_size, length = labels.shape
        n_classes = seq_logits.shape[-1]
        expected = (batch_size, length, length, n_classes * n_classes)
        if tuple(pair_logits.shape) != expected:
            raise ValueError(
                f"Expected pair logits {expected}, got {tuple(pair_logits.shape)}."
            )
        pair_mask = torch.triu(
            torch.ones(length, length, device=pair_logits.device, dtype=torch.bool),
            diagonal=1,
        )[None].expand(batch_size, -1, -1)
        pair_targets = labels.unsqueeze(2) * n_classes + labels.unsqueeze(1)
        pair_mask_expanded = pair_mask.unsqueeze(-1).expand(
            -1, -1, -1, n_classes * n_classes
        )
        valid_logits = pair_logits[pair_mask_expanded].reshape(
            -1, n_classes * n_classes
        )
        valid_targets = pair_targets[pair_mask].reshape(-1)
        pair_loss = F.cross_entropy(
            valid_logits, valid_targets.long(), reduction="sum"
        ) / (batch_size * 2)
        l1_reg = self.l1_weight * pair_logits.abs().mean()
        l2_reg = self.l2_weight * (pair_logits**2).mean()
        return pair_loss + l1_reg + l2_reg

    def forward(self, headers, representations, batch):
        """Return the trained single-site and pairwise objectives."""
        seq_logits = headers["seqhead"]["logits"]
        pair_logits = headers["pairhead"]["logits"]
        labels = batch["aatype"]
        single = self._compute_single_loss(seq_logits, labels)
        pair = self._compute_pair_loss(
            pair_logits=pair_logits,
            seq_logits=seq_logits,
            labels=labels,
            z_matrix=representations.get("pair"),
        )
        return {"single": single, "pair": pair, "total": single + pair}


class HeaderBuilder:

    @staticmethod
    def build(config, seq_channel, pair_channel, parent):
        heads_config = config.model.heads
        head_factory = OrderedDict(
            seqhead=functools.partial(SeqHead, num_in_channel=seq_channel),
            pairhead=functools.partial(PairHead, num_in_channel=pair_channel),
            losshead=functools.partial(LossHead, num_in_channel=seq_channel),
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
