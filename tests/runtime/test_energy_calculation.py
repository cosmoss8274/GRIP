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

    def test_energy_batch_matches_independent_records(self):
        seq = torch.cat([self.seq, self.seq.flip(1)])
        pair = torch.cat([self.pair, self.pair.flip(-1)])
        labels = torch.cat([self.labels, self.labels.flip(1)])
        together = self.score(seq, pair, labels)
        separate = [
            self.score(seq[i : i + 1], pair[i : i + 1], labels[i : i + 1])
            for i in range(2)
        ]
        for key in together:
            torch.testing.assert_close(
                together[key], torch.cat([item[key] for item in separate]),
                rtol=0, atol=0,
            )

    def test_energy_gradients_match_negative_log_probabilities(self):
        seq = self.seq.clone().requires_grad_()
        pair = self.pair.clone().requires_grad_()
        self.score(seq, pair)["total"].sum().backward()
        expected_single = seq.detach().softmax(-1)
        expected_pair = pair.detach().softmax(-1)
        for i, aa in enumerate(self.labels[0].tolist()):
            expected_single[0, i, aa] -= 1
            for j, bb in enumerate(self.labels[0].tolist()):
                expected_pair[0, i, j, 2 * aa + bb] -= 1
        torch.testing.assert_close(seq.grad, expected_single, rtol=0, atol=1e-15)
        torch.testing.assert_close(pair.grad, expected_pair / 2, rtol=0, atol=1e-15)

    def test_invalid_dimensions_do_not_produce_zero_scores(self):
        cases = [
            {"seq": torch.zeros(1, 2)},
            {"seq": torch.zeros(1, 2, 2, 2)},
            {"seq": self.seq[:, :1]},
            {"labels": self.labels[:, :1]},
            {"pair": self.pair[..., :3]},
            {"pair": torch.zeros(1, 2, 1, 4)},
            {"pair": torch.zeros(1, 2, 2, 2, 3)},
            {"pair": torch.zeros(1, 2, 2, 1, 4)},
            {"seq": self.seq[:, :0], "labels": self.labels[:, :0]},
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

    def test_loss_gradients_preserve_batch_scaling_and_regularizers(self):
        loss = LossHead(SimpleNamespace(l1_weight=0.03, l2_weight=0.07), 1)
        seq = self.seq.repeat(2, 1, 1).requires_grad_()
        pair = (self.pair.repeat(2, 1, 1, 1) - 0.4).requires_grad_()
        labels = torch.cat([self.labels, self.labels.flip(1)])
        result = loss(
            {"seqhead": {"logits": seq}, "pairhead": {"logits": pair}},
            {}, {"aatype": labels},
        )
        result["total"].backward()
        expected_single = seq.detach().softmax(-1)
        expected_pair = (
            0.03 * pair.detach().sign() + 2 * 0.07 * pair.detach()
        ) / pair.numel()
        for b, residues in enumerate(labels.tolist()):
            for i, aa in enumerate(residues):
                expected_single[b, i, aa] -= 1
                for j in range(i + 1, len(residues)):
                    probabilities = pair.detach()[b, i, j].softmax(-1)
                    probabilities[2 * aa + residues[j]] -= 1
                    expected_pair[b, i, j] += probabilities / (2 * len(labels))
        torch.testing.assert_close(seq.grad, expected_single, rtol=0, atol=1e-15)
        torch.testing.assert_close(pair.grad, expected_pair, rtol=0, atol=1e-15)

    def test_loss_rejects_reordered_axes_and_mismatched_pairs(self):
        loss = LossHead(SimpleNamespace(l1_weight=0.0, l2_weight=0.0), 1)
        cases = [
            (self.seq.transpose(0, 1), self.pair),
            (self.seq, self.pair[..., :3]),
            (self.seq, self.pair.transpose(0, 1)),
        ]
        for seq, pair in cases:
            with self.subTest(seq=seq.shape, pair=pair.shape):
                with self.assertRaises((ValueError, IndexError, RuntimeError)):
                    loss(
                        {"seqhead": {"logits": seq}, "pairhead": {"logits": pair}},
                        {}, {"aatype": self.labels},
                    )

    def test_single_site_has_diagonal_energy_but_no_off_diagonal_loss(self):
        seq, pair, labels = self.seq[:, :1], self.pair[:, :1, :1], self.labels[:, :1]
        energy = self.score(seq, pair, labels)
        row = pair[0, 0, 0].tolist()
        expected = (math.log(sum(math.exp(x) for x in row)) - row[0]) / 2
        self.assertAlmostEqual(energy["pair"].item(), expected, places=12)
        loss = LossHead(SimpleNamespace(l1_weight=0.0, l2_weight=0.0), 1)
        self.assertEqual(loss._compute_pair_loss(pair, seq, labels).item(), 0.0)

    def test_model_rejects_mixed_peptide_lengths_before_computation(self):
        config = load_config()
        model = BaseEnergyModel(config).eval()
        record = prepare_record({"epitope_sequence": "AAAAAAAA", "mhc": "ACDE"}, config)
        batch = {key: value.repeat(2, *([1] * (value.ndim - 1)))
                 for key, value in record.items()}
        batch["epitope_lengths"] = torch.tensor([8, 7])
        with patch.object(model, "_run_evoformer") as trunk:
            with self.assertRaises(ValueError):
                model(batch, compute_loss=False)
            trunk.assert_not_called()

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
