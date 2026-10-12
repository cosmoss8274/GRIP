"""GRIP core model, sequence encoding, and pretrained weights.

Importing this package does not import model dependencies or load checkpoints.
"""

from importlib import import_module

__all__ = [
    "BaseEnergyModel", "TrainingDataGenerator", "load_config",
    "prepare_record", "score_record", "load_weights", "load_pretrained",
]

_EXPORTS = {
    "BaseEnergyModel": (".model.energy", "BaseEnergyModel"),
    "TrainingDataGenerator": (".data", "SequenceDataset"),
    "load_config": (".inference", "load_config"),
    "prepare_record": (".inference", "prepare_record"),
    "score_record": (".inference", "score_record"),
    "load_weights": (".checkpoints", "load_weights"),
    "load_pretrained": (".pretrained", "load_pretrained"),
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
