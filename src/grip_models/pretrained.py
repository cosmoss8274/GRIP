"""Download and load released GRIP base-model weights."""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


_RELEASE = "v0.1.0"
_RELEASE_URL = f"https://github.com/cosmoss8274/GRIP/releases/download/{_RELEASE}"
_MODELS = {
    "mhc-i-energy": {
        "weights": (
            "grip-mhc-i-energy.pt",
            "411935d7595c5dbdfe090ea7a10b9d6b8acd325298427b9dabff239740d4e81e",
        ),
        "config": (
            "grip-mhc-i-energy.json",
            "02644a2fec914b6d126b2edb482fea1616a77cf7433a8d3e124d96b8c56eb7fd",
        ),
    },
}


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(cache, filename, expected, *, local_files_only=False):
    path = cache / filename
    if path.exists():
        if not path.is_file() or not hmac.compare_digest(_digest(path), expected):
            raise ValueError(f"Invalid cached pretrained file: {path}. Remove it and retry.")
        return path
    if local_files_only:
        raise FileNotFoundError(f"Pretrained file is not cached: {path}")

    temporary = None
    try:
        request = Request(f"{_RELEASE_URL}/{filename}", headers={"User-Agent": "GRIP/0.1.0"})
        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=cache, prefix=f".{filename}.",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                digest = hashlib.sha256()
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
                    digest.update(block)
        if not hmac.compare_digest(digest.hexdigest(), expected):
            raise ValueError(f"Downloaded pretrained file failed verification: {filename}")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_pretrained(name="mhc-i-energy", *, cache_dir=None, device="cpu",
                    local_files_only=False):
    """Return a pretrained base energy model in evaluation mode and its config.

    ``mhc-i-energy`` selects the MHC-I base energy checkpoint. Missing files
    are downloaded from the fixed GRIP v0.1.0 GitHub release and verified before
    use. ``cache_dir`` defaults to ``$XDG_CACHE_HOME/grip/v0.1.0`` or
    ``~/.cache/grip/v0.1.0``. An existing valid cache works offline. Set
    ``local_files_only=True`` to require cached files without network access.

    ``device`` is passed to ``model.to(device)`` after strict weight loading.
    The returned model can be used with ``score_record`` to obtain base-model
    energies; it does not include a classification head or peptide sampling.
    """
    if not isinstance(name, str) or name not in _MODELS:
        raise ValueError(f"Unknown pretrained model {name!r}. Available: mhc-i-energy")
    if cache_dir is None:
        cache_home = os.environ.get("XDG_CACHE_HOME")
        base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
        cache = base / "grip" / _RELEASE
    else:
        cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    entry = _MODELS[name]
    config_path = _asset(cache, *entry["config"], local_files_only=local_files_only)
    weights_path = _asset(cache, *entry["weights"], local_files_only=local_files_only)

    from .checkpoints import load_weights
    from .inference import load_config
    from .model.energy import BaseEnergyModel

    config = load_config(config_path)
    model = BaseEnergyModel(config)
    load_weights(model, weights_path, sha256=entry["weights"][1])
    model.to(device)
    model.eval()
    return model, config
