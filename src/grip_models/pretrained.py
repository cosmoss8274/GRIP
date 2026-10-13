"""Compatibility entry point for former pretrained downloads."""


def load_pretrained(
    name="mhc-i-energy", *, cache_dir=None, device="cpu", local_files_only=False
):
    """Direct existing callers to explicit local-checkpoint loading."""
    raise RuntimeError(
        "Pretrained downloads are not available. Load a local checkpoint with load_weights."
    )
