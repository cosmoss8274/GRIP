"""Sequence encoding and energy-model inference."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from importlib import import_module
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .configuration import _default_config

_FIELDS = ("epitope_sequence", "mhc", "l_v", "l_j", "h_v", "cdr3_b", "h_j")
_AMINO_ACIDS = frozenset("ARNDCQEGHILKMFPSTWYV")
_FIXED_VALUES = {
    ("model", "embeddings", "seq_channel"): 128,
    ("model", "embeddings", "pair_channel"): 64,
    ("model", "embeddings", "seq_info_dim"): 21,
    ("data", "n_channels_seq"): 21,
}
# Match the dataset default.
_REFERENCE_MAX_LENGTH = 500


def _data_module():
    return import_module(".data", __package__)


def _plain(value):
    if isinstance(value, SimpleNamespace):
        return {key: _plain(item) for key, item in vars(value).items()}
    return deepcopy(value)


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    return deepcopy(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _finite(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, path + (str(index),))
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {'.'.join(path)}")


def _merge_known(default, supplied, path=(), *, complete=False):
    label = ".".join(path) or "configuration"
    if not isinstance(supplied, dict):
        raise ValueError(f"{label} must be an object")
    allowed = set(default)
    if path == ("data",):
        allowed.add("max_length")
    if path == ("model",):
        allowed.add("num_layers")
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(
            f"Unknown configuration key at {label}: {sorted(map(str, unknown))}"
        )
    if complete and set(default) - set(supplied):
        raise ValueError(
            f"Missing configuration keys at {label}: {sorted(set(default) - set(supplied))}"
        )
    result = deepcopy(default)
    for key, value in supplied.items():
        field_path = path + (key,)
        field = ".".join(field_path)
        if (path, key) in (
            (("data",), "max_length"),
            (("model",), "num_layers"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
            result[key] = value
            continue
        expected = default[key]
        if isinstance(expected, dict):
            result[key] = _merge_known(expected, value, field_path, complete=complete)
        else:
            if type(value) is not type(expected):
                raise ValueError(f"{field} must have type {type(expected).__name__}")
            result[key] = deepcopy(value)
    return result


def _fixed_architecture(config):
    for path, expected in _FIXED_VALUES.items():
        value = config
        for name in path:
            value = value[name]
        if value != expected:
            raise ValueError(
                f"{'.'.join(path)} is fixed at {expected} for the reference base model"
            )


def _validated_config(config):
    if not isinstance(config, SimpleNamespace):
        raise TypeError("config must be a configuration object returned by load_config")
    defaults = _default_config()
    supplied = _plain(config)
    _finite(supplied)
    merged = _merge_known(defaults, supplied, complete=True)
    _fixed_architecture(merged)
    return _namespace(merged)


def load_config(config_path=None):
    """Load model defaults with optional JSON overrides.

    ``model.num_layers`` sets the trunk depth (default: 4);
    ``data.max_length`` sets the input length limit. The model architecture
    must match the checkpoint."""
    supplied = None
    if config_path is not None:
        with Path(config_path).open("r", encoding="utf-8") as stream:
            supplied = json.load(
                stream,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        if not isinstance(supplied, dict):
            raise ValueError("Configuration JSON must be an object")
        _finite(supplied)
    defaults = _default_config()
    merged = defaults if supplied is None else _merge_known(defaults, supplied)
    _fixed_architecture(merged)
    return _namespace(merged)


def prepare_record(record, config):
    """Encode a peptide and its MHC/TCR context as a single CPU batch.

    Sequences use the 20 standard uppercase amino acids. Supply a nonempty
    ``epitope_sequence`` and at least one receptor component; omit unused
    components. Each sequence must fit ``data.max_length`` (default: 500)."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping of canonical sequence fields")
    unknown = set(record) - set(_FIELDS)
    if unknown:
        raise ValueError(f"Unknown record fields: {sorted(map(str, unknown))}")
    canonical = {}
    for field in _FIELDS:
        value = deepcopy(record.get(field))
        if field != "epitope_sequence" and value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a sequence string")
        if any(character not in _AMINO_ACIDS for character in value):
            raise ValueError(
                f"{field} must contain only standard uppercase amino acids"
            )
        canonical[field] = value
    if not canonical["epitope_sequence"]:
        raise ValueError("epitope_sequence must be nonempty")
    if not any(canonical[field] for field in _FIELDS[1:]):
        raise ValueError("At least one MHC or TCR component must be nonempty")

    checked_config = _validated_config(config)
    limit = getattr(checked_config.data, "max_length", _REFERENCE_MAX_LENGTH)
    for field, value in canonical.items():
        if len(value) > limit:
            raise ValueError(f"{field} exceeds max_length={limit}")

    import pandas as pd

    frame = pd.DataFrame([canonical], columns=_FIELDS)
    dataset = _data_module().SequenceDataset(
        [],
        None,
        frame,
        checked_config,
        integrated_data_path=None,
    )
    batch, _ = dataset.collate_fn([dataset[0]])
    return batch


def _evaluation_device(model):
    """Return the shared device of an evaluation-mode model."""
    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if any(module.training for module in model.modules()):
        raise ValueError(
            "model and all submodules must already be in evaluation mode; call model.eval() explicitly"
        )
    devices = {
        tensor.device for tensor in list(model.parameters()) + list(model.buffers())
    }
    if len(devices) != 1:
        raise ValueError("model parameters and buffers must identify one shared device")
    device = next(iter(devices))
    if device.type == "meta":
        raise ValueError("model parameters must reside on a materialized device")
    return device


def score_record(model, record, config):
    """Return single-site, pairwise and total energies for one record.

    The model must be in evaluation mode. Outputs are energy scores."""
    import torch

    device = _evaluation_device(model)
    batch = prepare_record(record, config)
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        outputs = model(batch, compute_loss=False, compute_energy=True)
    if not isinstance(outputs, Mapping) or not isinstance(
        outputs.get("energies"), Mapping
    ):
        raise ValueError("model must return an energies mapping")
    result = {}
    for name in ("single", "pair", "total"):
        value = outputs["energies"].get(name)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError(f"energy {name} must be a one-value tensor")
        if torch.is_complex(value) or not torch.isfinite(value).all().item():
            raise ValueError(f"energy {name} must be finite and real")
        result[name] = float(value.detach().cpu().item())
    return result
