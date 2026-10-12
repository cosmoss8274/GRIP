"""Load the fixed original source files used by numerical regression tests."""

import hashlib
from pathlib import Path
import stat
from zipfile import ZipFile

ORIGINAL_HASHES = {
    "test_energy_nomean_newdata_fix_120.py": "90c87088363230a7fd25108a69943ac78e791052868217a672ac1e001dfb1700",
    "common_modules.py": "a607e30a78b66f9cf6e5f7ba14c5b1427dac84cfd16bc33949fc0c2336f547d1",
    "basic.py": "7e4b1a5e5eac450cd90b29aaff08a08aab8e308b8f93eef8d042719b5b3b7da9",
    "EvoMSA_flash.py": "5e15a4b34632f7440bfb4cdd89aae78fe9a013f8ec41aba8d2ce35f0a0c329e8",
    "EvoPair_flash.py": "9b8620bd8beabb048a44ac96f65811689b89f6ba83e79c9fa5acdc2c549dbff8",
    "evoformer_fast.py": "0df3f87be2d6c4aca738a2766f55b452d6d776b3853b11d30b1321f7d05d87df",
}
MODULE_FILES = (
    "src/grip_models/configuration.py",
    "src/grip_models/data.py",
    "src/grip_models/model/heads.py",
    "src/grip_models/model/energy.py",
    "src/grip_models/model/layers.py",
    "src/grip_models/model/primitives.py",
    "src/grip_models/model/msa.py",
    "src/grip_models/model/pair.py",
    "src/grip_models/model/evoformer.py",
)
REFERENCE_ARCHIVE = Path("tests/reference/original_sources.zip")


def validate_sources(root):
    """Return the six original files after checking names, types and hashes.

    The archive is independent of the current implementation. This validates
    reference bytes; normal-case numerical regression checks runtime behavior.
    """
    recovered = {}
    with ZipFile(Path(root) / REFERENCE_ARCHIVE) as archive:
        members = archive.infolist()
        if len(members) != len(ORIGINAL_HASHES) or {
            entry.filename for entry in members
        } != set(ORIGINAL_HASHES):
            raise ValueError(
                "Reference archive must contain exactly the six original files"
            )
        for entry in members:
            if (
                not stat.S_ISREG(entry.external_attr >> 16)
                or entry.file_size > 1024 * 1024
            ):
                raise ValueError(f"Invalid reference archive member: {entry.filename}")
            raw = archive.read(entry)
            if hashlib.sha256(raw).hexdigest() != ORIGINAL_HASHES[entry.filename]:
                raise ValueError(f"Original source hash mismatch: {entry.filename}")
            recovered[entry.filename] = raw
    return recovered, {
        "files_verified": len(recovered),
        "private_snapshot_required": False,
        "files": [
            {
                "source_filename": name,
                "original_source_sha256": ORIGINAL_HASHES[name],
                "original_bytes_preserved": True,
            }
            for name in sorted(recovered)
        ],
    }


def reconstruct_sources(root, destination):
    """Write only the verified, fixed source names; never extract ZIP paths."""
    recovered, report = validate_sources(root)
    for name, raw in recovered.items():
        (Path(destination) / name).write_bytes(raw)
    return report
