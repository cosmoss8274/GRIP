"""Analytical checks for energy values and malformed model outputs."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from grip_models import BaseEnergyModel, load_config, prepare_record
from grip_models.model.heads import LossHead


class EnergyCalculationTests(unittest.TestCase):
    def setUp(self):
        self.seq = torch.tensor([[[1.0, 2.0], [3.0, -1.0]]], dtype=torch.float64)
        self.pair = torch.arange(16, dtype=torch.float64).reshape(1, 2, 2, 4) / 7
        self.labels = torch.tensor([[0, 1]])

    def score(self, seq=None, pair=None, labels=None):
        return BaseEnergyModel.calculate_energies(
            None,
            {
                "seqhead": {"logits": self.seq if seq is None else seq},
                "pairhead": {"logits": self.pair if pair is None else pair},
            },
            {},
            {"aatype": self.labels if labels is None else labels},
        )

    def test_energy_matches_explicit_finite_alphabet_sum(self):
        single = sum(
            math.log(sum(math.exp(x) for x in row)) - row[aa]
            for row, aa in zip(self.seq[0].tolist(), [0, 1])
        )
        pair = 0.0
        for i, aa in enumerate([0, 1]):
            for j, bb in enumerate([0, 1]):
                logits = self.pair[0, i, j].tolist()
                pair += (
                    math.log(sum(math.exp(x) for x in logits)) - logits[2 * aa + bb]
                ) / 2
        actual = self.score()
        self.assertAlmostEqual(actual["single"].item(), single, places=12)
        self.assertAlmostEqual(actual["pair"].item(), pair, places=12)
        self.assertAlmostEqual(actual["total"].item(), single + pair, places=12)

    def test_equivalent_head_layouts(self):
        expected = self.score()
        actual = self.score(self.seq.unsqueeze(1), self.pair.reshape(1, 2, 2, 2, 2))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]))

    def test_invalid_dimensions_do_not_produce_zero_scores(self):
        cases = [
            {"seq": torch.zeros(1, 2)},
            {"seq": torch.zeros(1, 2, 2, 2)},
            {"seq": self.seq[:, :1]},
            {"labels": self.labels[:, :1]},
            {"pair": self.pair[..., :3]},
            {"pair": torch.zeros(1, 2, 1, 4)},
            {"pair": torch.zeros(1, 2, 2, 2, 3)},
        ]
        for case in cases:
            with self.subTest(case=tuple(case)), self.assertRaises(ValueError):
                self.score(**case)

    def test_missing_heads_and_labels_raise(self):
        for headers, batch in [
            ({}, {"aatype": self.labels}),
            ({"seqhead": {"logits": self.seq}}, {"aatype": self.labels}),
            ({"seqhead": {"logits": self.seq}, "pairhead": {"logits": self.pair}}, {}),
        ]:
            with self.subTest(headers=tuple(headers)), self.assertRaises(KeyError):
                BaseEnergyModel.calculate_energies(None, headers, {}, batch)

    def test_pair_training_objective_uses_upper_triangle(self):
        loss = LossHead(SimpleNamespace(l1_weight=0.0, l2_weight=0.0), 1)
        row = self.pair[0, 0, 1].tolist()
        expected = (math.log(sum(math.exp(x) for x in row)) - row[1]) / 2
        actual = loss._compute_pair_loss(self.pair, self.seq, self.labels)
        self.assertAlmostEqual(actual.item(), expected, places=12)
        two = loss._compute_pair_loss(
            self.pair.repeat(2, 1, 1, 1),
            self.seq.repeat(2, 1, 1),
            self.labels.repeat(2, 1),
        )
        self.assertAlmostEqual(two.item(), expected, places=12)

    def test_inference_does_not_call_training_loss(self):
        config = load_config()
        model = BaseEnergyModel(config).eval()
        batch = prepare_record({"epitope_sequence": "AAAAAAAA", "mhc": "ACDE"}, config)
        with patch.object(
            model.loss_head,
            "forward",
            side_effect=AssertionError("Training loss called"),
        ):
            with torch.inference_mode():
                result = model(batch, compute_loss=False)
        self.assertTrue(torch.isfinite(result["energies"]["total"]).all())


if __name__ == "__main__":
    unittest.main()
