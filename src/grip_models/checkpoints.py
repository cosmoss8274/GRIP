"""Explicit, checksum-verified loading for tensor-only model state dictionaries.

This module imports only the standard library until ``load_weights`` is called.
It neither identifies publication weights nor chooses a checkpoint for a caller.
"""

from collections.abc import Mapping
import hashlib
import hmac
import os
import re
import stat


def _digest_handle(handle):
    digest = hashlib.sha256()
    while True:
        block = handle.read(1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _file_signature(handle):
    info = os.fstat(handle.fileno())
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def load_weights(model, checkpoint_path, *, sha256, state_dict_key=None):
    """Load one explicitly named local checkpoint after complete preflight.

    ``sha256`` must be the expected 64-character hexadecimal digest supplied by
    the caller. ``state_dict_key=None`` means the checkpoint itself is the state
    dictionary; a nonempty string selects that exact top-level key. No schema,
    prefix, model head, checkpoint candidate or device is inferred.

    Deserialization uses ``torch.load(weights_only=True, map_location='cpu')``.
    The checksum and deserializer read the same open file handle. File metadata
    is checked for changes before any model state is assigned. All keys, tensor
    types, shapes, dtypes and layouts are checked before a strict state load.
    Meta tensors are rejected because they cannot provide concrete weights.

    The model's training flag and parameter identities are left unchanged.
    Returns the verified digest, absolute input path and selected key. Errors
    from safe deserialization propagate; there is no unsafe loading fallback.
    """
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
        raise ValueError("sha256 must be an expected 64-character hexadecimal digest")
    if state_dict_key is not None and (
        not isinstance(state_dict_key, str) or not state_dict_key
    ):
        raise TypeError(
            "state_dict_key must be None or a nonempty top-level key string"
        )

    local_path = os.fspath(checkpoint_path)
    if not isinstance(local_path, str) or not local_path:
        raise TypeError("checkpoint_path must name one local file using a string path")
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", local_path):
        raise ValueError("checkpoint_path must be a local file, not a URL")
    absolute_path = os.path.abspath(local_path)
    expected_digest = sha256.lower()

    # Nonblocking open prevents a FIFO supplied as a path from hanging before
    # the regular-file check. It has no effect on reads from regular files.
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(absolute_path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("checkpoint_path must name a regular local file")
        initial_signature = _file_signature(handle)
        actual_digest = _digest_handle(handle)
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise ValueError("Checkpoint SHA-256 does not match the expected digest")
        if _file_signature(handle) != initial_signature:
            raise RuntimeError("Checkpoint file changed while its checksum was read")

        import torch

        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        handle.seek(0)
        checkpoint = torch.load(handle, weights_only=True, map_location="cpu")
        if _file_signature(handle) != initial_signature:
            raise RuntimeError("Checkpoint file changed during deserialization")

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping, not an arbitrary object")
    if state_dict_key is None:
        candidate = checkpoint
    else:
        if state_dict_key not in checkpoint:
            raise KeyError(f"Checkpoint has no top-level key {state_dict_key!r}")
        candidate = checkpoint[state_dict_key]
    if not isinstance(candidate, Mapping):
        raise TypeError("Selected state dictionary must be a mapping")

    expected = model.state_dict()
    if not isinstance(expected, Mapping):
        raise TypeError("model.state_dict() must return a mapping")
    if any(not isinstance(key, str) for key in candidate):
        raise TypeError("Checkpoint state dictionary keys must be strings")
    if any(not isinstance(key, str) for key in expected):
        raise TypeError("Model state dictionary keys must be strings")
    missing = sorted(set(expected) - set(candidate))
    unexpected = sorted(set(candidate) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"State dictionary key mismatch: missing={missing}, unexpected={unexpected}"
        )

    for key, reference in expected.items():
        value = candidate[key]
        if not isinstance(reference, torch.Tensor) or not isinstance(
            value, torch.Tensor
        ):
            raise TypeError(
                f"State entry {key!r} must be a tensor in both dictionaries"
            )
        if reference.is_meta or value.is_meta:
            raise ValueError(
                f"State entry {key!r} must contain concrete, non-meta tensors"
            )
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(f"State entry {key!r} has a mismatched tensor shape")
        if value.dtype != reference.dtype:
            raise ValueError(f"State entry {key!r} has a mismatched tensor dtype")
        if value.layout != reference.layout:
            raise ValueError(f"State entry {key!r} has a mismatched tensor layout")

    model.load_state_dict(candidate, strict=True)
    return {
        "sha256": actual_digest,
        "path": absolute_path,
        "state_dict_key": state_dict_key,
    }
