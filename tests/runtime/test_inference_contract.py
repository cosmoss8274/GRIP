"""Real tensor/dataframe checks for the single-record adapter contract.

Model calls use a small protocol stub. No original model is trained or executed,
and no checkpoint is read. Encoding is compared with SequenceDataset.
"""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
import torch

from grip_models import inference
from grip_models.configuration import load_config as reference_config
from grip_models.data import SequenceDataset


ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("epitope_sequence", "mhc", "l_v", "l_j", "h_v", "cdr3_b", "h_j")
VALID = {"epitope_sequence": "ARND", "mhc": "ACDEFGHIK"}


class EnergyStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.seqhead = torch.nn.Identity()
        self.pairhead = torch.nn.Identity()
        self.heads = {"seqhead": self.seqhead, "pairhead": self.pairhead}
        self.calls = []
        self.return_value = None

    def forward(self, batch, *, compute_loss, compute_energy):
        self.calls.append((batch, compute_loss, compute_energy, torch.is_inference_mode_enabled()))
        if self.return_value is not None:
            return self.return_value
        return {"energies": {
            "single": torch.tensor([1.25], device=self.anchor.device),
            "pair": torch.tensor([2.5], device=self.anchor.device),
            "total": torch.tensor([3.75], device=self.anchor.device),
        }}


class ConfigContractTests(unittest.TestCase):
    def load_text(self, text):
        with tempfile.TemporaryDirectory(prefix="grip-config-") as directory:
            path = Path(directory) / "config.json"
            path.write_text(text, encoding="utf-8")
            return inference.load_config(path)

    def test_none_preserves_all_original_defaults(self):
        self.assertEqual(inference.load_config(), reference_config(None))
        self.assertEqual(inference.load_config(None), reference_config(None))
        self.assertFalse(hasattr(inference.load_config().data, "max_length"))

    def test_valid_partial_override_recursively_preserves_other_values(self):
        before = reference_config(None)
        actual = self.load_text(json.dumps({
            "data": {"batch_size": 2, "max_length": 30},
            "model": {"heads": {"losshead": {"l1_weight": 0.03}}},
        }))
        expected = deepcopy(before)
        expected.data.batch_size = 2
        expected.data.max_length = 30
        expected.model.heads.losshead.l1_weight = 0.03
        self.assertEqual(actual, expected)
        self.assertEqual(reference_config(None), before)

    def test_configuration_objects_do_not_share_mutable_defaults(self):
        first = inference.load_config()
        second = inference.load_config()
        first.data.batch_size = 7
        self.assertEqual(second, reference_config(None))

    def test_json_must_be_an_object(self):
        for text in ("[]", "null", "4", '"value"', "true"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load_text(text)

    def test_duplicate_keys_are_rejected_at_every_depth(self):
        for text in ('{"data": {}, "data": {}}',
                     '{"data": {"batch_size": 1, "batch_size": 2}}'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "Duplicate"):
                self.load_text(text)

    def test_nonfinite_numbers_are_rejected_including_overflow(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            text = '{"loss": {"seq": {"weight": ' + token + '}}}'
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "finite"):
                self.load_text(text)

    def test_unknown_keys_and_wrong_types_are_rejected(self):
        invalid = [
            {"unknown": 1},
            {"model": {"embeddings": {"seq_chanell": 128}}},
            {"data": {"batch_size": True}},
            {"data": {"batch_size": 1.0}},
            {"data": {"batch_size": "1"}},
            {"loss": {"seq": {"weight": 1}}},
            {"model": {"heads": {"seqhead": {"enabled": 1}}}},
            {"data": []},
            {"loss": {"seq": {"config": {"unrecognized": 1}}}},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load_text(json.dumps(value))

    def test_fixed_architecture_dimensions_are_rejected(self):
        invalid = [
            {"model": {"embeddings": {"seq_channel": 129}}},
            {"model": {"embeddings": {"pair_channel": 65}}},
            {"model": {"embeddings": {"seq_info_dim": 20}}},
            {"data": {"n_channels_seq": 20}},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fixed"):
                self.load_text(json.dumps(value))

    def test_max_length_is_positive_integer(self):
        for value in (0, -1, True, 20.0, "20", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_length"):
                self.load_text(json.dumps({"data": {"max_length": value}}))


class RecordContractTests(unittest.TestCase):
    def setUp(self):
        self.config = inference.load_config()

    def reference_batch(self, record):
        canonical = {field: record.get(field) for field in FIELDS}
        for field in FIELDS[1:]:
            if canonical[field] is None:
                canonical[field] = ""
        frame = pd.DataFrame([canonical], columns=FIELDS).copy(deep=True)
        dataset = SequenceDataset([], None, frame, self.config, integrated_data_path=None)
        return dataset.collate_fn([dataset[0]])[0]

    def test_encoding_is_identical_to_original_for_every_tensor(self):
        records = [
            VALID,
            {"epitope_sequence": "ARND", "mhc": None, "cdr3_b": "CASS"},
            {"epitope_sequence": "ARND", "mhc": "ACDE", "l_v": "FG",
             "l_j": "HI", "h_v": "KL", "cdr3_b": "MN", "h_j": "PQ"},
        ]
        for record in records:
            with self.subTest(record=record):
                before_record, before_config = deepcopy(record), deepcopy(self.config)
                actual = inference.prepare_record(record, self.config)
                expected = self.reference_batch(record)
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    self.assertEqual(actual[key].dtype, expected[key].dtype, key)
                    self.assertEqual(actual[key].device.type, "cpu", key)
                    self.assertTrue(torch.equal(actual[key], expected[key]), key)
                self.assertEqual(record, before_record)
                self.assertEqual(self.config, before_config)

    def test_aatype_and_missing_component_lengths_are_present(self):
        batch = inference.prepare_record(VALID, self.config)
        self.assertTrue(torch.equal(batch["aatype"], torch.tensor([[0, 1, 2, 3]])))
        self.assertEqual(batch["epitope_lengths"].tolist(), [4])
        for name in ("l_v", "l_j", "h_v", "cdr3_b", "h_j"):
            self.assertEqual(tuple(batch[name + "_encoded"].shape), (1, 0, 21))
            self.assertEqual(batch[name + "_lengths"].tolist(), [0])

    def test_dataframe_input_never_reads_default_data_file(self):
        with mock.patch.object(pd, "read_csv", side_effect=AssertionError("Unexpected file read")):
            batch = inference.prepare_record(VALID, self.config)
        self.assertEqual(batch["epitope_lengths"].tolist(), [4])

    def test_inputs_are_not_automatically_cleaned(self):
        for value in ("arnd", " ARND", "ARND ", "ARND+mod", "AR(N)D", "ARNX", "AR-ND", "AR\nND"):
            record = {**VALID, "epitope_sequence": value}
            with self.subTest(value=value), self.assertRaises(ValueError):
                inference.prepare_record(record, self.config)

    def test_invalid_optional_components_are_rejected(self):
        for field in FIELDS[1:]:
            for value in ("ac", "ACX", 1, False, float("nan"), ["AC"]):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    inference.prepare_record({**VALID, field: value}, self.config)

    def test_missing_epitope_context_unknown_keys_and_nonmapping_are_rejected(self):
        for record in ({"mhc": "AC"}, {**VALID, "epitope_sequence": ""},
                       {**VALID, "epitope_sequence": None},
                       {"epitope_sequence": "ARND"},
                       {"epitope_sequence": "ARND", "mhc": None},
                       {**VALID, "peptide": "ARND"}, {**VALID, 5: "AC"}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                inference.prepare_record(record, self.config)
        with self.assertRaises(TypeError):
            inference.prepare_record([VALID], self.config)

    def test_max_length_applies_to_all_components(self):
        config = deepcopy(self.config)
        config.data.max_length = 4
        valid = {"epitope_sequence": "ARND", "mhc": "ACDE"}
        for field in FIELDS:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "max_length"):
                inference.prepare_record({**valid, field: "A" * 5}, config)

    def test_no_unsupported_eight_to_fourteen_length_rule(self):
        for length in (1, 15):
            with self.subTest(length=length):
                batch = inference.prepare_record({"epitope_sequence": "A" * length, "mhc": "AC"}, self.config)
                self.assertEqual(batch["epitope_lengths"].tolist(), [length])

    def test_mutated_configuration_is_revalidated(self):
        for key, value in (("max_length", 0), ("n_channels_seq", 20), ("unrecognized", 1)):
            config = deepcopy(self.config)
            setattr(config.data, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                inference.prepare_record(VALID, config)
        config = deepcopy(self.config)
        config.loss.seq.weight = math.inf
        with self.assertRaises(ValueError):
            inference.prepare_record(VALID, config)


class ScoreContractTests(unittest.TestCase):
    def setUp(self):
        self.config = inference.load_config()
        self.model = EnergyStub().eval()

    def test_score_uses_exact_protocol_without_changing_mode_or_weights(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        result = inference.score_record(self.model, VALID, self.config)
        self.assertEqual(result, {"single": 1.25, "pair": 2.5, "total": 3.75})
        self.assertTrue(all(type(value) is float for value in result.values()))
        self.assertFalse(self.model.training)
        batch, compute_loss, compute_energy, inference_mode = self.model.calls[0]
        self.assertFalse(compute_loss)
        self.assertTrue(compute_energy)
        self.assertTrue(inference_mode)
        self.assertTrue(all(value.device == self.model.anchor.device for value in batch.values()))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_training_mode_is_rejected_without_changing_it(self):
        self.model.train()
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            inference.score_record(self.model, VALID, self.config)
        self.assertTrue(self.model.training)
        self.assertEqual(self.model.calls, [])

    def test_training_submodule_is_rejected_without_changing_it(self):
        self.model.seqhead.train()
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            inference.score_record(self.model, VALID, self.config)
        self.assertFalse(self.model.training)
        self.assertTrue(self.model.seqhead.training)
        self.assertEqual(self.model.calls, [])

    def test_missing_or_replaced_heads_are_rejected_before_forward(self):
        for name in ("seqhead", "pairhead"):
            model = EnergyStub().eval()
            delattr(model, name)
            with self.subTest(name=name), self.assertRaises(ValueError):
                inference.score_record(model, VALID, self.config)
            self.assertEqual(model.calls, [])
        self.model.heads["seqhead"] = torch.nn.Identity()
        with self.assertRaises(ValueError):
            inference.score_record(self.model, VALID, self.config)
        self.assertEqual(self.model.calls, [])

    def test_invalid_head_container_is_rejected_before_forward(self):
        self.model.heads = 5
        with self.assertRaisesRegex(ValueError, "mapping"):
            inference.score_record(self.model, VALID, self.config)
        self.assertEqual(self.model.calls, [])

    def test_meta_model_is_rejected_before_forward(self):
        self.model.to("meta")
        with self.assertRaisesRegex(ValueError, "materialized"):
            inference.score_record(self.model, VALID, self.config)
        self.assertEqual(self.model.calls, [])

    def test_mixed_device_model_is_rejected_before_forward(self):
        self.model.register_buffer("other_device", torch.empty(1, device="meta"))
        with self.assertRaisesRegex(ValueError, "shared device"):
            inference.score_record(self.model, VALID, self.config)
        self.assertEqual(self.model.calls, [])

    def test_invalid_record_is_rejected_before_forward(self):
        with self.assertRaises(ValueError):
            inference.score_record(self.model, {**VALID, "epitope_sequence": "invalid"}, self.config)
        self.assertEqual(self.model.calls, [])

    def test_missing_non_tensor_multiple_and_nonfinite_energies_are_rejected(self):
        valid = {"single": torch.tensor(1.0), "pair": torch.tensor(2.0), "total": torch.tensor(3.0)}
        invalid = [None, 1.0, torch.tensor([1.0, 2.0]), torch.tensor(float("nan")),
                   torch.tensor(float("inf")), torch.tensor(float("-inf")), torch.tensor(1 + 2j)]
        for name in valid:
            for value in invalid:
                model = EnergyStub().eval()
                model.return_value = {"energies": {**valid, name: value}}
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    inference.score_record(model, VALID, self.config)
        for output in ({}, {"energies": []}, {"energies": {"single": torch.tensor(1.0)}}):
            model = EnergyStub().eval()
            model.return_value = output
            with self.subTest(output=output), self.assertRaises(ValueError):
                inference.score_record(model, VALID, self.config)

    def test_importing_adapter_alone_does_not_load_heavy_dependencies(self):
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "import grip_models.inference\n"
            "assert 'grip_models.data' not in sys.modules\n"
            "assert not ({'torch', 'numpy', 'pandas'} & set(sys.modules))\n"
        )
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                                text=True, capture_output=True, check=False, env=os.environ.copy())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
