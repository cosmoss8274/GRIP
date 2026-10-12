"""Analytical GRIP-CLA checks; no research data or external weights required."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from grip_models.classification import classify_record, load_classifier
from grip_models.inference import load_config, prepare_record
from grip_models.model.classifier import (
    ClassificationHead,
    ClassifierModel,
    EnergyNormalization,
)
from grip_models.model.energy import BaseEnergyModel


class ClassifierMathTests(unittest.TestCase):
    def test_length_normalization_and_repeated_legacy_operation(self):
        head = EnergyNormalization()
        raw = {
            "single": torch.tensor([8.0, 27.0]),
            "pair": torch.tensor([16.0, 81.0]),
            "total": torch.tensor([24.0, 108.0]),
        }
        batch = {"epitope_lengths": torch.tensor([2, 3])}
        once = head(raw, batch)
        twice = head(once, batch)
        self.assertEqual(once["single"].tolist(), [4.0, 9.0])
        self.assertEqual(once["pair"].tolist(), [4.0, 9.0])
        self.assertEqual(once["total"].tolist(), [8.0, 18.0])
        self.assertEqual(twice["single"].tolist(), [2.0, 3.0])
        self.assertEqual(twice["pair"].tolist(), [1.0, 1.0])
        self.assertEqual(twice["total"].tolist(), [3.0, 4.0])
        self.assertEqual(raw["total"].tolist(), [24.0, 108.0])

    def test_feature_order_diagonal_pooling_and_equivalent_layouts(self):
        head = ClassificationHead().double().eval()
        seq = torch.arange(42, dtype=torch.float64).reshape(1, 2, 21)
        pair = torch.arange(1764, dtype=torch.float64).reshape(1, 2, 2, 441)
        energies = {
            key: torch.tensor([value], dtype=torch.float64)
            for key, value in [("single", 2.0), ("pair", 3.0), ("total", 5.0)]
        }
        # Route six independently weighted features through the three layers.
        features = [2.0, 3.0, 5.0, 10.5, 661.5, 1101.5]
        indices = [0, 1, 2, 3, 24, 464]
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
            for i, column in enumerate(indices):
                head.classifier[0].weight[0, column] = i + 1
            head.classifier[3].weight[0, 0] = 0.5
            head.classifier[6].weight[0, 0] = 2
            head.classifier[6].bias[0] = -7
        expected = sum((i + 1) * value for i, value in enumerate(features)) - 7
        flat = head(energies, seq, pair)
        expanded = head(energies, seq.unsqueeze(1), pair.reshape(1, 2, 2, 21, 21))
        self.assertEqual(flat.item(), expected)
        self.assertTrue(torch.equal(flat, expanded))

    def test_profile_controls_exactly_one_or_two_normalizations(self):
        config = load_config()
        raw = {
            "single": torch.tensor([8.0]),
            "pair": torch.tensor([16.0]),
            "total": torch.tensor([24.0]),
        }
        base_output = {
            "energies": raw,
            "seqhead_pred": torch.zeros(1, 2, 21),
            "pairhead_pred": torch.zeros(1, 2, 2, 441),
        }
        for profile, expected in [
            ("mhc", [4.0, 4.0, 8.0]),
            ("tcr_legacy", [2.0, 1.0, 3.0]),
        ]:
            with self.subTest(profile=profile):
                model = ClassifierModel(config, profile=profile).eval()
                with patch.object(
                    BaseEnergyModel, "forward", return_value=deepcopy(base_output)
                ) as base:
                    with patch.object(
                        model.classification_head,
                        "forward",
                        return_value=torch.tensor([-2.0]),
                    ) as head:
                        output = model({"epitope_lengths": torch.tensor([2])})
                self.assertEqual(
                    [
                        head.call_args.args[0][k].item()
                        for k in ("single", "pair", "total")
                    ],
                    expected,
                )
                self.assertEqual(output["energies"]["total"].item(), 24.0)
                self.assertEqual(output["logit"].item(), -2.0)
                self.assertTrue(
                    torch.equal(output["score"], torch.sigmoid(torch.tensor([-2.0])))
                )
                self.assertEqual(
                    base.call_args.kwargs,
                    {"compute_loss": False, "compute_energy": True},
                )


class ClassifierLoadingTests(unittest.TestCase):
    def test_profile_is_required_and_never_inferred(self):
        with self.assertRaises(TypeError):
            ClassifierModel(load_config())
        for profile in (None, "auto", "mhc-i", "tcr", 1):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                ClassifierModel(load_config(), profile=profile)

    def test_state_layout_and_strict_loading_with_synthetic_checkpoint(self):
        config = load_config()
        model = ClassifierModel(config, profile="mhc")
        state = model.state_dict()
        base_keys = set(BaseEnergyModel(config).state_dict())
        expected_extra = {"energy_head.dummy"} | {
            f"classification_head.classifier.{layer}.{part}"
            for layer in (0, 3, 6)
            for part in ("weight", "bias")
        }
        self.assertEqual(len(state), 297)
        self.assertEqual(set(state) - base_keys, expected_extra)
        self.assertEqual(
            tuple(state["classification_head.classifier.0.weight"].shape), (256, 465)
        )
        with tempfile.TemporaryDirectory(prefix="grip-classifier-") as directory:
            path = Path(directory) / "synthetic.pt"
            torch.save(state, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded, loaded_config = load_classifier(path, sha256=digest, profile="mhc")
            self.assertEqual(loaded_config, config)
            self.assertFalse(any(module.training for module in loaded.modules()))
            self.assertTrue(
                all(
                    torch.equal(value, loaded.state_dict()[key])
                    for key, value in state.items()
                )
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_classifier(path, sha256="0" * 64, profile="mhc")
            torch.save(
                {key: value for key, value in state.items() if key in base_keys}, path
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "key mismatch"):
                load_classifier(path, sha256=digest, profile="mhc")


class ClassifierRecordTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.model = ClassifierModel(self.config, profile="mhc").eval()
        self.record = {"epitope_sequence": "ARND", "mhc": "ACDE"}

    def test_record_adapter_preserves_encoding_and_uses_inference_mode(self):
        expected = prepare_record(self.record, self.config)
        before = deepcopy(self.record)

        def output(batch):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertEqual(set(batch), set(expected))
            self.assertTrue(
                all(torch.equal(batch[key], value) for key, value in expected.items())
            )
            return {
                "logit": torch.tensor([-2.0]),
                "score": torch.sigmoid(torch.tensor([-2.0])),
            }

        with patch.object(self.model, "forward", side_effect=output):
            actual = classify_record(self.model, self.record, self.config)
        self.assertEqual(set(actual), {"logit", "score"})
        self.assertEqual(actual["logit"], -2.0)
        self.assertAlmostEqual(actual["score"], 0.119202922, places=7)
        self.assertEqual(self.record, before)

    def test_training_mode_is_rejected_without_changing_it(self):
        self.model.classification_head.train()
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            classify_record(self.model, self.record, self.config)
        self.assertTrue(self.model.classification_head.training)

    def test_allele_names_and_modified_peptides_are_not_guessed(self):
        for record in (
            {**self.record, "mhc": "HLA-A*02:01"},
            {**self.record, "epitope_sequence": "ARND+mod"},
        ):
            with self.subTest(record=record), self.assertRaises(ValueError):
                classify_record(self.model, record, self.config)

    def test_nonfinite_output_is_not_returned_as_a_score(self):
        with patch.object(
            self.model,
            "forward",
            return_value={
                "logit": torch.tensor([float("nan")]),
                "score": torch.tensor([0.5]),
            },
        ):
            with self.assertRaisesRegex(ValueError, "finite"):
                classify_record(self.model, self.record, self.config)


if __name__ == "__main__":
    unittest.main()
