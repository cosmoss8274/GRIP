#!/usr/bin/env python3
"""Check source-relocation parity on three synthetic CPU inputs.

Run from a source checkout with the model dependencies installed::

    python tools/check_numerical_parity.py
    python tools/check_numerical_parity.py --report source-parity.json

The reference files are reconstructed in a temporary directory by reversing
the five import relocations recorded in src/grip_models/source_changes.json.
Their bytes must match all six recorded original SHA-256 hashes before import.
No private snapshot, external dataset, or trained weights are loaded. This is
an engineering equivalence check, not a biological benchmark or a validation
of trained-model accuracy. JSON is always printed to stdout; --report also
saves it to a file. A failed check exits with a nonzero status.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile


SOURCE_FILES = {
    "src/grip_models/sequence.py": "test_energy_nomean_newdata_fix_120.py",
    "src/grip_models/common_modules.py": "common_modules.py",
    "src/grip_models/basic.py": "basic.py",
    "src/grip_models/EvoMSA_flash.py": "EvoMSA_flash.py",
    "src/grip_models/EvoPair_flash.py": "EvoPair_flash.py",
    "src/grip_models/evoformer_fast.py": "evoformer_fast.py",
}
FIELDS = ("epitope_sequence", "mhc", "l_v", "l_j", "h_v", "cdr3_b", "h_j")
CASES = (
    {"epitope_sequence": "AAAAAAAA", "mhc": "ACDEFGHIK"},
    {"epitope_sequence": "CCCCCCCCC", "mhc": "ACDEFGHIK", "cdr3_b": "ACDE"},
    {"epitope_sequence": "GGGGGGGG", "mhc": "", "cdr3_b": "ACDE"},
)
SEED = 712


class ParityError(Exception):
    """A check failure whose message contains only public source labels."""


def require(condition, message):
    if not condition:
        raise ParityError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def reconstruct_sources(root, destination):
    """Read only published files; source manifest paths are labels, not inputs."""
    manifest_bytes = (root / "src/grip_models/source_changes.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest.get("schema_version") == 1, "Unsupported source manifest schema")
    records = manifest["files"]
    targets = [record["target"] for record in records]
    require(len(targets) == len(SOURCE_FILES) and set(targets) == set(SOURCE_FILES),
            "Source manifest must contain exactly the six expected files")
    require(sum(len(record["import_relocations"]) for record in records) == 5,
            "Source manifest must record exactly five import relocations")
    verified = []
    for record in records:
        target = record["target"]
        original_name = SOURCE_FILES[target]
        require(Path(record["source"]).name == original_name,
                f"Unexpected original source label for {target}")
        packaged = (root / target).read_bytes()
        require(sha256(packaged) == record["packaged_source_sha256"],
                f"Packaged SHA-256 mismatch: {target}")
        lines = packaged.splitlines(keepends=True)
        changed_lines = set()
        for relocation in record["import_relocations"]:
            index = relocation["line"] - 1
            require(0 <= index < len(lines) and index not in changed_lines,
                    f"Invalid import relocation line: {target}")
            changed_lines.add(index)
            content = lines[index].rstrip(b"\r\n")
            require(content == relocation["after"].encode("utf-8"),
                    f"Import relocation does not match source: {target}")
            newline = lines[index][len(content):]
            lines[index] = relocation["before"].encode("utf-8") + newline
        restored = b"".join(lines)
        require(sha256(restored) == record["original_source_sha256"],
                f"Restored original SHA-256 mismatch: {target}")
        (destination / original_name).write_bytes(restored)
        verified.append({
            "packaged_file": target,
            "restored_filename": original_name,
            "packaged_sha256": sha256(packaged),
            "restored_original_sha256": sha256(restored),
            "original_sha256_matches_manifest": True,
            "import_relocations_reversed": len(changed_lines),
        })
    return {
        "manifest": "src/grip_models/source_changes.json",
        "manifest_sha256": sha256(manifest_bytes),
        "private_snapshot_required": False,
        "files_verified": len(verified),
        "import_relocations_reversed": 5,
        "files": verified,
    }


def as_dict(value):
    if hasattr(value, "__dict__"):
        return {key: as_dict(item) for key, item in vars(value).items()}
    return value


def same(left, right, torch, location="output"):
    """Require exact, finite CPU tensors and the same nested structure."""
    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor), f"Tensor type mismatch at {location}")
        require(left.shape == right.shape, f"Tensor shape mismatch at {location}")
        require(left.dtype == right.dtype, f"Tensor dtype mismatch at {location}")
        require(left.device.type == right.device.type == "cpu",
                f"Non-CPU tensor at {location}")
        require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
                f"Nonfinite tensor at {location}")
        require(torch.equal(left, right), f"Tensor value mismatch at {location}")
        return 1
    if isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(),
                f"Mapping keys mismatch at {location}")
        return sum(same(left[key], right[key], torch, f"{location}.{key}") for key in left)
    if isinstance(left, (tuple, list)):
        require(type(left) is type(right) and len(left) == len(right),
                f"Sequence structure mismatch at {location}")
        return sum(same(a, b, torch, f"{location}.{index}")
                   for index, (a, b) in enumerate(zip(left, right)))
    require(type(left) is type(right) and left == right, f"Value mismatch at {location}")
    return 0


def compare_models(root, restored, report):
    # The reference selects a device at import. Keep this standalone process on
    # CPU even on a CUDA host; no inherited scientific source is modified.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available(), "CPU-only execution could not be established")
    report["runtime"] = {
        "python": sys.version.split()[0],
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "einops", "biopython")
        },
        "torch_threads": 1,
        "torch_interop_threads": 1,
    }
    old_path = sys.path[:]
    try:
        sys.path[:0] = [str(restored), str(root / "src")]
        original = importlib.import_module("test_energy_nomean_newdata_fix_120")
        packaged = importlib.import_module("grip_models.sequence")
        for target, name in SOURCE_FILES.items():
            original_module = importlib.import_module(Path(name).stem)
            package_module = importlib.import_module(f"grip_models.{Path(target).stem}")
            require(Path(original_module.__file__).resolve() == restored / name,
                    f"Unexpected restored module origin: {name}")
            require(Path(package_module.__file__).resolve() == (root / target).resolve(),
                    f"Unexpected packaged module origin: {target}")
        original_config = original.load_config(None)
        packaged_config = packaged.load_config(None)
        require(as_dict(original_config) == as_dict(packaged_config), "Default configuration mismatch")
        report["default_configurations_identical"] = True
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        original_model = original.myModel(original_config).cpu().eval()
        packaged_model = packaged.myModel(packaged_config).cpu().eval()
        packaged_model.load_state_dict(original_model.state_dict(), strict=True)
        report["state_tensors_identical"] = same(
            original_model.state_dict(), packaged_model.state_dict(), torch, "state_dict")
        report["cases"] = []
        for index, record in enumerate(CASES):
            full = {key: record.get(key, "") for key in FIELDS}
            old_data = original.DataGenerator(
                [], None, pd.DataFrame([full]), original_config, integrated_data_path=None)
            new_data = packaged.DataGenerator(
                [], None, pd.DataFrame([full]), packaged_config, integrated_data_path=None)
            old_batch, _ = old_data.collate_fn([old_data[0]])
            new_batch, _ = new_data.collate_fn([new_data[0]])
            batch_count = same(old_batch, new_batch, torch, "batch")
            with torch.inference_mode():
                old_output = original_model(old_batch, compute_loss=False, compute_energy=True)
                new_output = packaged_model(new_batch, compute_loss=False, compute_energy=True)
            output_count = same(old_output, new_output, torch)
            report["cases"].append({
                "index": index,
                "synthetic_input": full,
                "batch_tensors_identical": batch_count,
                "output_tensors_identical": output_count,
                "maximum_absolute_difference": 0.0,
            })
    finally:
        sys.path[:] = old_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, help="also save the JSON report to this file")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1,
        "purpose": "CPU source-relocation equivalence on three synthetic inputs",
        "trained_weights_used": False,
        "external_datasets_used": False,
        "biological_benchmark": False,
        "scientific_benchmark": False,
        "scope": "Exact comparison of default configuration, copied random model state, encoded batches, and forward outputs; no claim about biological accuracy or trained-model performance.",
        "device": "cpu",
        "seed": SEED,
        "comparison": "torch.equal with finite tensors; no numerical tolerance",
        "model_state": "seeded random initialization copied strictly from restored source to packaged model",
        "status": "failed",
    }
    stage = "source reconstruction"
    try:
        if args.report is not None:
            output = args.report.resolve()
            require(output != Path(__file__).resolve() and not output.is_relative_to(root / "src"),
                    "Report destination must not overwrite the checker or package source")
        with tempfile.TemporaryDirectory(prefix="grip-source-parity-") as directory:
            restored = Path(directory).resolve()
            report["source_reconstruction"] = reconstruct_sources(root, restored)
            stage = "CPU numerical comparison"
            # Inherited source prints debug information; stdout remains JSON.
            with contextlib.redirect_stdout(io.StringIO()):
                compare_models(root, restored, report)
        report["status"] = "passed"
    except Exception as error:
        report["failure"] = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error) if isinstance(error, ParityError) else "Check failed; verify the published source and model dependencies.",
        }
    if args.report is not None:
        try:
            require(args.report.resolve() != Path(__file__).resolve()
                    and not args.report.resolve().is_relative_to(root / "src"),
                    "Report destination must not overwrite the checker or package source")
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except Exception as error:
            report["status"] = "failed"
            report["failure"] = {
                "stage": "report writing",
                "type": type(error).__name__,
                "message": "Could not write the requested report file.",
            }
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
