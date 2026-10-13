"""Direct model batches must not introduce unmasked receptor padding."""

import unittest
from unittest.mock import patch

import pandas as pd
import torch

from grip_models import BaseEnergyModel, ClassifierModel, load_config, prepare_record
from grip_models.data import SequenceDataset


RECEPTORS = ("mhc", "l_v", "l_j", "h_v", "cdr3_b", "h_j")


class ModelBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.config.model.num_layers = 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            cls.models = (
                BaseEnergyModel(cls.config).eval(),
                ClassifierModel(cls.config, profile="mhc").eval(),
            )

    def collate(self, records):
        fields = ("epitope_sequence",) + RECEPTORS
        frame = pd.DataFrame(
            [{field: record.get(field, "") for field in fields} for record in records]
        )
        dataset = SequenceDataset([], None, frame, self.config, integrated_data_path=None)
        return dataset.collate_fn([dataset[i] for i in range(len(dataset))])[0]

    def evaluate(self, model, batch):
        if isinstance(model, ClassifierModel):
            return model(batch)
        return model(batch, compute_loss=False)

    def test_collated_unequal_receptors_are_rejected_before_attention(self):
        for component in RECEPTORS:
            for short, long in (("ACDE", "ACDEFGHIK"), ("", "ACDE")):
                records = [
                    {"epitope_sequence": "ARND", "mhc": "ACDE", component: sequence}
                    for sequence in (short, long)
                ]
                batch = self.collate(records)
                self.assertEqual(batch["epitope_lengths"].tolist(), [4, 4])
                self.assertEqual(batch[f"{component}_lengths"].tolist(), [len(short), len(long)])
                for model in self.models:
                    with self.subTest(component=component, short=short, model=type(model).__name__):
                        with patch.object(model, "_run_evoformer") as trunk:
                            with self.assertRaisesRegex(ValueError, component):
                                self.evaluate(model, batch)
                            trunk.assert_not_called()

    def test_equal_length_batches_match_individual_predictions(self):
        records = [
            {"epitope_sequence": "ARND", "mhc": "ACDE", "l_v": "FG",
             "l_j": "HI", "h_v": "KL", "cdr3_b": "MN", "h_j": "PQ"},
            {"epitope_sequence": "CDEF", "mhc": "FGHI", "l_v": "KL",
             "l_j": "MN", "h_v": "PQ", "cdr3_b": "RS", "h_j": "TV"},
        ]
        batch = self.collate(records)
        with torch.inference_mode():
            for model in self.models:
                combined = self.evaluate(model, batch)
                for index, record in enumerate(records):
                    individual = self.evaluate(model, prepare_record(record, self.config))
                    for name in ("seqhead_pred", "pairhead_pred", "logit", "score"):
                        if name in combined:
                            torch.testing.assert_close(
                                combined[name][index:index + 1], individual[name],
                                atol=1e-5, rtol=1e-5,
                            )
                    for name in ("energies", "normalized_energies"):
                        if name in combined:
                            for term in ("single", "pair", "total"):
                                torch.testing.assert_close(
                                    combined[name][term][index:index + 1], individual[name][term],
                                    atol=1e-5, rtol=1e-5,
                                )


if __name__ == "__main__":
    unittest.main()
