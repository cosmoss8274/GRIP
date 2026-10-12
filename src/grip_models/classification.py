"""Explicit checkpoint loading and single-record GRIP-CLA inference."""

from .checkpoints import load_weights
from .inference import _evaluation_device, load_config, prepare_record


def load_classifier(
    checkpoint_path,
    *,
    sha256,
    profile,
    config_path=None,
    state_dict_key=None,
    device="cpu",
):
    """Return a strictly loaded classifier in evaluation mode and its config.

    The caller supplies one local checkpoint, its expected SHA-256, and the
    ``mhc`` or ``tcr_legacy`` calculation profile. The complete fine-tuned
    state, including the backbone and classification head, is required. No model is
    selected, downloaded or substituted automatically. ``state_dict_key``
    selects an explicitly named checkpoint field; by default the file itself
    must be the state dictionary.
    """
    from .model.classifier import ClassifierModel

    config = load_config(config_path)
    model = ClassifierModel(config, profile=profile)
    load_weights(model, checkpoint_path, sha256=sha256, state_dict_key=state_dict_key)
    model.to(device).eval()
    return model, config


def classify_record(model, record, config):
    """Return ``{'logit': float, 'score': float}`` for one canonical record.

    Use the sequence fields accepted by ``prepare_record``. The receptor
    representation must be supplied explicitly; allele names and modified
    peptides are not rewritten or guessed. This entry point therefore does
    not reproduce historical benchmarks that encoded allele-label strings as
    amino acids. Such replays require their original encoded batches.

    The model and its submodules must already be in evaluation mode. The
    returned sigmoid score has no imposed threshold or calibration transform.
    """
    import torch
    from .model.classifier import ClassifierModel

    if not isinstance(model, ClassifierModel):
        raise TypeError("model must be a loaded ClassifierModel")
    device = _evaluation_device(model)
    batch = {
        name: value.to(device) for name, value in prepare_record(record, config).items()
    }
    with torch.inference_mode():
        outputs = model(batch)
    result = {}
    for name in ("logit", "score"):
        value = outputs[name]
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError(f"Classifier {name} must be one finite value")
        result[name] = float(value.cpu().item())
    return result
