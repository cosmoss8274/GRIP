"""GRIP energy, classification and inverse models."""

from importlib import import_module

__all__ = [
    "BaseEnergyModel",
    "ClassifierModel",
    "SequenceDataset",
    "TrainingDataGenerator",
    "load_config",
    "prepare_record",
    "score_record",
    "load_weights",
    "load_pretrained",
    "load_classifier",
    "classify_record",
    "load_inverse_model",
    "inverse_logits",
]

_EXPORTS = {
    "BaseEnergyModel": (".model.energy", "BaseEnergyModel"),
    "ClassifierModel": (".model.classifier", "ClassifierModel"),
    "SequenceDataset": (".data", "SequenceDataset"),
    "TrainingDataGenerator": (".data", "SequenceDataset"),
    "load_config": (".inference", "load_config"),
    "prepare_record": (".inference", "prepare_record"),
    "score_record": (".inference", "score_record"),
    "load_weights": (".checkpoints", "load_weights"),
    "load_pretrained": (".pretrained", "load_pretrained"),
    "load_classifier": (".classification", "load_classifier"),
    "classify_record": (".classification", "classify_record"),
    "load_inverse_model": (".inverse", "load_inverse_model"),
    "inverse_logits": (".inverse", "inverse_logits"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
