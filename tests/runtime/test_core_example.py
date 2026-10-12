"""Check the public example without trained weights or model inference."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "grip_core_example", ROOT / "examples" / "core.py"
)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


class CoreExampleTests(unittest.TestCase):
    def run_example(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = EXAMPLE.main(list(arguments))
            except SystemExit as error:
                code = error.code
        return SimpleNamespace(
            returncode=code, stdout=stdout.getvalue(), stderr=stderr.getvalue()
        )

    def test_default_encoding_is_usable_without_weights(self):
        result = self.run_example("encode")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["input_source"], "synthetic_example")
        self.assertEqual(payload["component_lengths"]["epitope"], 8)
        self.assertEqual(payload["component_lengths"]["mhc"], 9)
        self.assertEqual(payload["tensor_count"], 28)
        self.assertNotIn("total", payload)

    def test_encoding_returns_json_without_debug_output(self):
        with self.assertNoLogs("grip_models.data", level="DEBUG"):
            result = self.run_example("encode")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["input_source"], "synthetic_example")
        self.assertEqual(result.stderr, "")

    def test_supplied_record_is_encoded(self):
        with tempfile.TemporaryDirectory(prefix="grip-example-") as directory:
            record = Path(directory) / "record.json"
            record.write_text(
                json.dumps({"epitope_sequence": "AAAA", "cdr3_b": "ACDE"})
            )
            result = self.run_example("encode", "--record", str(record))
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["input_source"], "record_file")
        self.assertEqual(payload["component_lengths"]["epitope"], 4)
        self.assertEqual(payload["component_lengths"]["mhc"], 0)

    def test_scoring_requires_explicit_input(self):
        for arguments in (("score",), ("score", "--checkpoint", "model.pt")):
            with self.subTest(arguments=arguments):
                result = self.run_example(*arguments)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("required", result.stderr)

    def test_scoring_uses_pretrained_defaults_and_forwards_offline_options(self):
        model, config = object(), object()
        energies = {"single": 1.0, "pair": 2.0, "total": 3.0}
        with tempfile.TemporaryDirectory(prefix="grip-example-") as directory:
            record = Path(directory) / "record.json"
            record.write_text(json.dumps(EXAMPLE.SYNTHETIC_RECORD))
            with (
                patch(
                    "grip_models.load_pretrained", return_value=(model, config)
                ) as load,
                patch("grip_models.score_record", return_value=energies) as score,
            ):
                result = self.run_example("score", "--record", str(record))
                self.assertEqual(result.returncode, 0, result.stderr)
                load.assert_called_once_with(
                    "mhc-i-energy", cache_dir=None, device="cpu", local_files_only=False
                )
                score.assert_called_once_with(model, EXAMPLE.SYNTHETIC_RECORD, config)
                self.assertEqual(json.loads(result.stdout), energies)
                load.reset_mock()
                result = self.run_example(
                    "score",
                    "--record",
                    str(record),
                    "--cache-dir",
                    directory,
                    "--local-files-only",
                    "--device",
                    "test-device",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                load.assert_called_once_with(
                    "mhc-i-energy",
                    cache_dir=Path(directory),
                    device="test-device",
                    local_files_only=True,
                )

    def test_pretrained_failure_never_produces_a_score(self):
        with tempfile.TemporaryDirectory(prefix="grip-example-") as directory:
            record = Path(directory) / "record.json"
            record.write_text(json.dumps(EXAMPLE.SYNTHETIC_RECORD))
            with (
                patch(
                    "grip_models.load_pretrained",
                    side_effect=FileNotFoundError("not cached"),
                ),
                patch("grip_models.score_record") as score,
            ):
                result = self.run_example(
                    "score", "--record", str(record), "--local-files-only"
                )
                score.assert_not_called()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("not cached", result.stderr)

    def test_local_checkpoint_uses_matching_options(self):
        model = Mock()
        model.to.return_value = model
        with tempfile.TemporaryDirectory(prefix="grip-example-") as directory:
            record = Path(directory) / "record.json"
            record.write_text(json.dumps(EXAMPLE.SYNTHETIC_RECORD))
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"fixture, never deserialized")
            with (
                patch("grip_models.BaseEnergyModel", return_value=model),
                patch("grip_models.load_weights") as load,
                patch("grip_models.score_record", return_value={"total": 0.0}),
                patch("grip_models.load_pretrained") as pretrained,
            ):
                result = self.run_example(
                    "score",
                    "--record",
                    str(record),
                    "--checkpoint",
                    str(checkpoint),
                    "--state-dict-key",
                    "model_state_dict",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                load.assert_called_once_with(
                    model,
                    checkpoint,
                    sha256=EXAMPLE.file_sha256(checkpoint),
                    state_dict_key="model_state_dict",
                )
                pretrained.assert_not_called()
                model.to.assert_called_once_with("cpu")
                model.eval.assert_called_once_with()

    def test_incompatible_weight_options_are_rejected_before_reading_input(self):
        for options in (
            ("--config", "config.json"),
            ("--state-dict-key", "state"),
            ("--checkpoint", "model.pt", "--cache-dir", "cache"),
            ("--checkpoint", "model.pt", "--local-files-only"),
            ("--checkpoint", "model.pt", "--model", "mhc-i-energy"),
        ):
            with self.subTest(options=options):
                result = self.run_example("score", "--record", "unused.json", *options)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertNotIn("No such file", result.stderr)

    def test_missing_checkpoint_does_not_produce_a_score(self):
        with tempfile.TemporaryDirectory(prefix="grip-example-") as directory:
            record = Path(directory) / "record.json"
            record.write_text(json.dumps({"epitope_sequence": "AAAA", "mhc": "ACDE"}))
            result = self.run_example(
                "score",
                "--record",
                str(record),
                "--checkpoint",
                str(Path(directory) / "missing.pt"),
            )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("missing.pt", result.stderr)


if __name__ == "__main__":
    unittest.main()
