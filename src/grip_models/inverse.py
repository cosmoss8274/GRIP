"""Structure-conditioned residue and residue-pair predictions for peptide design."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .checkpoints import load_weights

# Model configuration digest, independent of the local ESM path.
_MODEL_CONFIG_SHA256 = (
    "828a20bb2d6b56fcf4c8c78b748486fbb2e7b44f4ae4b93ad2beaf45c9f8725c"
)
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"


def _check_backbone(backbone):
    if not isinstance(backbone, nn.Module):
        raise TypeError("backbone must be an initialized structural model")
    config = backbone.config
    config = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    config = json.loads(json.dumps(config))
    lm = config["embeddings_and_seqformer"]["lm"]
    esm_path = Path(lm.pop("model_path")).expanduser()
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != _MODEL_CONFIG_SHA256:
        raise ValueError("Use the original GRIP inverse model configuration")
    regression = esm_path.with_name(esm_path.stem + "-contact-regression.pt")
    for path in (esm_path, regression):
        if not path.is_file():
            raise FileNotFoundError(f"Required local ESM resource: {path}")
    extractor = backbone.impl.seqformer.esm
    if (
        extractor.repr_layer != 33
        or extractor.model.num_layers != 33
        or extractor.model.embed_dim != 1280
    ):
        raise ValueError("The inverse model requires ESM2 t33 650M, layer 33")
    return extractor


def _check_features(batch):
    if {
        "heads",
        "representations",
        "prev_seq",
        "prev_pair",
        "is_recycling",
    } & batch.keys():
        raise ValueError("Pass fresh structure features without previous forward state")
    b, n = batch["seq"].shape
    mask, peptide = batch["mask"], batch["peptide_mask"]
    if mask.shape != (b, n) or peptide.shape != (b, n):
        raise ValueError("mask and peptide_mask must follow the full seq residue axes")
    if mask.dtype != torch.bool or peptide.dtype != torch.bool:
        raise ValueError("mask and peptide_mask must be boolean")
    if b == 0 or not peptide.any(dim=1).all():
        raise ValueError("Each example must identify at least one peptide residue")
    if (peptide & ~mask).any():
        raise ValueError("peptide_mask selects a masked or padded residue")
    return b, n


class _InverseModel(nn.Module):
    def __init__(self, backbone, extractor):
        super().__init__()
        self.carbondesign = backbone
        self.chain_type_embedding = nn.Embedding(4, 384)
        # The ESM extractor is managed separately from registered parameters.
        self._esm_extractor = extractor

    def forward(self, features):
        if any(module.training for module in self.modules()):
            raise ValueError("Call model.eval() for inverse-model inference")
        device = self.chain_type_embedding.weight.device
        b, n = _check_features(features)
        self._esm_extractor.model.eval()
        self._esm_extractor.model.to(device)
        self._esm_extractor.device = device
        # Isolate the recycling state from the caller's feature mapping.
        output = self.carbondesign(batch=dict(features), compute_loss=True)
        # The checkpoint includes this embedding; the prediction heads use
        # the unmodified sequence representation.
        _ = output["representations"]["seq"] + self.chain_type_embedding(
            features["chain_types"]
        )
        logits = output["heads"]["seqhead"]["logits"]
        pair_logits = output["heads"]["pairhead"]["logits"]
        for name, value, shape in (
            ("logits", logits, (b, n, 21)),
            ("pair_logits", pair_logits, (b, n, n, 441)),
        ):
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(
                    f"Invalid {name}; expected finite values with shape {shape}"
                )
        return {
            "logits": logits,
            "pair_logits": pair_logits,
            "peptide_mask": features["peptide_mask"],
        }


def load_inverse_model(
    backbone, checkpoint, *, sha256, state_dict_key=None, device="cpu"
):
    """Load inverse-model weights into a configured structure backbone.

    Requires the GRIP inverse configuration and local ESM2 t33 650M resources.
    ``state_dict_key`` selects a nested checkpoint state dictionary. Returns
    an evaluation-mode model, with its backbone and ESM extractor on ``device``."""
    extractor = _check_backbone(backbone)
    model = _InverseModel(backbone, extractor)
    load_weights(model, checkpoint, sha256=sha256, state_dict_key=state_dict_key)
    model.to(device).eval()
    extractor.model.to(device).eval()
    extractor.device = torch.device(device)
    return model


@torch.no_grad()
def inverse_logits(model, features):
    """Predict residue and residue-pair logits from structure features.

    Returns ``logits`` [B, L, 21], ``pair_logits`` [B, L, L, 441] and
    ``peptide_mask`` [B, L], where L spans all input residues. Supply a fresh
    feature batch prepared by the structure feature pipeline."""
    if not isinstance(model, _InverseModel):
        raise TypeError("model must be returned by load_inverse_model")
    return model(features)
