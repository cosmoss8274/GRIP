"""GRIP classifier loading and sequence inference."""

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
    """Load a classifier checkpoint and return the model and configuration.

    ``profile`` selects ``mhc`` or ``tcr_legacy`` normalization. ``sha256``
    verifies the file; ``state_dict_key`` selects a nested state dictionary.
    The model is returned on ``device`` in evaluation mode."""
    from .model.classifier import ClassifierModel

    config = load_config(config_path)
    model = ClassifierModel(config, profile=profile)
    load_weights(model, checkpoint_path, sha256=sha256, state_dict_key=state_dict_key)
    model.to(device).eval()
    return model, config


def classify_record(model, record, config):
    """Classify one peptide–receptor sequence record.

    Uses the sequence fields accepted by ``prepare_record``. Returns scalar
    ``logit`` and ``score`` values, where ``score = sigmoid(logit)``. The model
    must be in evaluation mode."""
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
