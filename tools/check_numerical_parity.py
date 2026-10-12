#!/usr/bin/env python3
"""Check normal-input numerical behavior against six fixed original sources.

Run from a source checkout with the model dependencies installed::

    python tools/check_numerical_parity.py
    python tools/check_numerical_parity.py --report source-parity.json

The reference files come from a fixed test-only archive. Each original file's
hash is checked before import; the current implementation can change its error
handling independently of these normal-input numerical comparisons.
One valid single-record case also compares compute_loss=True and parameter
gradients, with both models in evaluation mode and without an optimizer step.
No private snapshot, external dataset, or trained weights are loaded. This is
an engineering equivalence check, not a biological benchmark or a validation
of trained-model accuracy. JSON is always printed to stdout; --report also
saves it to a file. A failed check exits with a nonzero status.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile

from source_reference import MODULE_FILES, ORIGINAL_HASHES, reconstruct_sources

FIELDS = ("epitope_sequence", "mhc", "l_v", "l_j", "h_v", "cdr3_b", "h_j")
CASES = (
    {"epitope_sequence": "AAAAAAAA", "mhc": "ACDEFGHIK"},
    {"epitope_sequence": "CCCCCCCCC", "mhc": "ACDEFGHIK", "cdr3_b": "ACDE"},
    {"epitope_sequence": "GGGGGGGG", "mhc": "", "cdr3_b": "ACDE"},
)
SEED = 712


class ParityError(Exception):
    """A numerical regression check failed."""


def require(condition, message):
    if not condition:
        raise ParityError(message)


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
        require(
            left.device.type == right.device.type == "cpu",
            f"Non-CPU tensor at {location}",
        )
        require(
            bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
            f"Nonfinite tensor at {location}",
        )
        require(torch.equal(left, right), f"Tensor value mismatch at {location}")
        return 1
    if isinstance(left, dict):
        require(
            isinstance(right, dict) and left.keys() == right.keys(),
            f"Mapping keys mismatch at {location}",
        )
        return sum(
            same(left[key], right[key], torch, f"{location}.{key}") for key in left
        )
    if isinstance(left, (tuple, list)):
        require(
            type(left) is type(right) and len(left) == len(right),
            f"Sequence structure mismatch at {location}",
        )
        return sum(
            same(a, b, torch, f"{location}.{index}")
            for index, (a, b) in enumerate(zip(left, right))
        )
    require(
        type(left) is type(right) and left == right, f"Value mismatch at {location}"
    )
    return 0


def compare_loss_gradients(original, packaged, old_batch, new_batch, torch):
    original.zero_grad(set_to_none=True)
    packaged.zero_grad(set_to_none=True)
    old_losses = original(old_batch, compute_loss=True, compute_energy=False)["losses"]
    new_losses = packaged(new_batch, compute_loss=True, compute_energy=False)["losses"]
    loss_count = same(old_losses, new_losses, torch, "losses")
    old_losses["total"].backward()
    new_losses["total"].backward()
    old_parameters = dict(original.named_parameters())
    new_parameters = dict(packaged.named_parameters())
    require(old_parameters.keys() == new_parameters.keys(), "Parameter names changed")
    gradient_count = absent_count = 0
    for name, parameter in old_parameters.items():
        old_gradient, new_gradient = parameter.grad, new_parameters[name].grad
        if old_gradient is None:
            require(new_gradient is None, f"Unexpected parameter gradient: {name}")
            absent_count += 1
        else:
            gradient_count += same(
                old_gradient, new_gradient, torch, f"gradient.{name}"
            )
    original.zero_grad(set_to_none=True)
    packaged.zero_grad(set_to_none=True)
    return {
        "model_mode": "eval",
        "compute_loss": True,
        "compute_energy": False,
        "loss_tensors_identical": loss_count,
        "parameter_gradients_identical": gradient_count,
        "parameters_without_gradients": absent_count,
        "maximum_absolute_difference": 0.0,
    }


def compare_models(root, restored, report):
    # The reference selects a device at import. Keep this standalone process on
    # CPU even on a CUDA host; no inherited scientific source is modified.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(
        not torch.cuda.is_available(), "CPU-only execution could not be established"
    )
    report["runtime"] = {
        "python": sys.version.split()[0],
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "einops",
                "biopython",
            )
        },
        "torch_threads": 1,
        "torch_interop_threads": 1,
    }
    old_path = sys.path[:]
    try:
        sys.path[:0] = [str(restored), str(root / "src")]
        original = importlib.import_module("test_energy_nomean_newdata_fix_120")
        packaged = importlib.import_module("grip_models")
        for name in ORIGINAL_HASHES:
            original_module = importlib.import_module(Path(name).stem)
            require(
                Path(original_module.__file__).resolve() == restored / name,
                f"Unexpected restored module origin: {name}",
            )
        for target in MODULE_FILES:
            module_name = ".".join(Path(target).with_suffix("").parts[1:])
            package_module = importlib.import_module(module_name)
            require(
                Path(package_module.__file__).resolve() == (root / target).resolve(),
                f"Unexpected packaged module origin: {target}",
            )
        original_config = original.load_config(None)
        packaged_config = packaged.load_config(None)
        require(
            as_dict(original_config) == as_dict(packaged_config),
            "Default configuration mismatch",
        )
        report["default_configurations_identical"] = True
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        original_model = original.myModel(original_config).cpu().eval()
        packaged_model = packaged.BaseEnergyModel(packaged_config).cpu().eval()
        packaged_model.load_state_dict(original_model.state_dict(), strict=True)
        report["state_tensors_identical"] = same(
            original_model.state_dict(),
            packaged_model.state_dict(),
            torch,
            "state_dict",
        )
        report["cases"] = []
        for index, record in enumerate(CASES):
            full = {key: record.get(key, "") for key in FIELDS}
            old_data = original.DataGenerator(
                [],
                None,
                pd.DataFrame([full]),
                original_config,
                integrated_data_path=None,
            )
            new_data = packaged.TrainingDataGenerator(
                [],
                None,
                pd.DataFrame([full]),
                packaged_config,
                integrated_data_path=None,
            )
            old_batch, _ = old_data.collate_fn([old_data[0]])
            new_batch, _ = new_data.collate_fn([new_data[0]])
            batch_count = same(old_batch, new_batch, torch, "batch")
            with torch.inference_mode():
                old_output = original_model(
                    old_batch, compute_loss=False, compute_energy=True
                )
                stdout, stderr = io.StringIO(), io.StringIO()
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    new_output = packaged_model(
                        new_batch, compute_loss=False, compute_energy=True
                    )
            output_count = same(old_output, new_output, torch)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                scores = packaged.score_record(packaged_model, full, packaged_config)
            require(
                scores
                == {
                    key: float(value.item())
                    for key, value in old_output["energies"].items()
                },
                "Public scoring values differ from the original output",
            )
            require(
                not stdout.getvalue() and not stderr.getvalue(),
                "Default inference emitted diagnostic output",
            )
            report["cases"].append(
                {
                    "index": index,
                    "synthetic_input": full,
                    "batch_tensors_identical": batch_count,
                    "output_tensors_identical": output_count,
                    "maximum_absolute_difference": 0.0,
                    "public_scores_identical": True,
                    "default_inference_quiet": True,
                }
            )
            if index == 0:
                report["loss_and_gradient_regression"] = compare_loss_gradients(
                    original_model, packaged_model, old_batch, new_batch, torch
                )
    finally:
        sys.path[:] = old_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report", type=Path, help="also save the JSON report to this file"
    )
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 2,
        "purpose": "CPU numerical regression on three valid synthetic inputs",
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
    stage = "reference archive verification"
    try:
        if args.report is not None:
            output = args.report.resolve()
            require(
                output != Path(__file__).resolve()
                and not output.is_relative_to(root / "src"),
                "Report destination must not overwrite the checker or package source",
            )
        with tempfile.TemporaryDirectory(prefix="grip-source-parity-") as directory:
            restored = Path(directory).resolve()
            report["reference_sources"] = reconstruct_sources(root, restored)
            stage = "CPU numerical comparison"
            # Inherited source prints debug information; stdout remains JSON.
            with contextlib.redirect_stdout(io.StringIO()):
                compare_models(root, restored, report)
        report["status"] = "passed"
    except Exception as error:
        report["failure"] = {
            "stage": stage,
            "type": type(error).__name__,
            "message": (
                str(error)
                if isinstance(error, ParityError)
                else "Check failed; verify the published source and model dependencies."
            ),
        }
    if args.report is not None:
        try:
            require(
                args.report.resolve() != Path(__file__).resolve()
                and not args.report.resolve().is_relative_to(root / "src"),
                "Report destination must not overwrite the checker or package source",
            )
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
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
