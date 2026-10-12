"""Real PyTorch tests using only tiny synthetic linear-layer checkpoints.

PyTorch is required: dependency absence is an import failure, not a passed or
skipped runtime check. No GRIP model module or biological checkpoint is used.
"""

import hashlib
import io
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from grip_models import checkpoints


class UnsupportedPayload:
    """A normal Python class which safe weights-only loading must reject."""


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "weights.pt"
        self.model = torch.nn.Linear(3, 2, dtype=torch.float32)
        with torch.no_grad():
            self.model.weight.fill_(-1.0)
            self.model.bias.fill_(-2.0)
        self.weights = {
            "weight": torch.full_like(self.model.weight, 3.0),
            "bias": torch.full_like(self.model.bias, 4.0),
        }

    def save(self, payload):
        torch.save(payload, self.path)
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def snapshot(self):
        return {name: tensor.detach().clone()
                for name, tensor in self.model.state_dict().items()}

    def assert_unchanged(self, before, training, identities):
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertEqual(self.model.training, training)
        self.assertEqual([id(p) for p in self.model.parameters()], identities)

    def reject(self, payload, exception, *, key=None):
        digest = self.save(payload)
        before = self.snapshot()
        training = self.model.training
        identities = [id(p) for p in self.model.parameters()]
        with mock.patch.object(self.model, "load_state_dict",
                               wraps=self.model.load_state_dict) as loader:
            with self.assertRaises(exception):
                checkpoints.load_weights(self.model, self.path,
                                         sha256=digest, state_dict_key=key)
            loader.assert_not_called()
        self.assert_unchanged(before, training, identities)

    def test_success_exact_weights_strict_load_and_training_state_preserved(self):
        for training in (True, False):
            with self.subTest(training=training):
                self.model.train(training)
                identities = [id(p) for p in self.model.parameters()]
                digest = self.save(self.weights)
                with mock.patch.object(self.model, "load_state_dict",
                                       wraps=self.model.load_state_dict) as loader:
                    result = checkpoints.load_weights(self.model, self.path,
                                                      sha256=digest.upper())
                    loader.assert_called_once()
                    self.assertEqual(loader.call_args.kwargs, {"strict": True})
                for name, tensor in self.model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, self.weights[name]))
                self.assertEqual(self.model.training, training)
                self.assertEqual([id(p) for p in self.model.parameters()], identities)
                self.assertEqual(result, {"sha256": digest, "path": str(self.path),
                                          "state_dict_key": None})

    def test_explicit_top_level_key_success(self):
        digest = self.save({"model_state_dict": self.weights, "epoch": 2})
        result = checkpoints.load_weights(self.model, self.path, sha256=digest,
                                          state_dict_key="model_state_dict")
        self.assertEqual(result["state_dict_key"], "model_state_dict")
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, self.weights[name]))

    def test_same_handle_is_hashed_rewound_and_safely_loaded(self):
        digest = self.save(self.weights)
        real_load = torch.load

        def safe_load(handle, **kwargs):
            self.assertIsInstance(handle, io.BufferedIOBase)
            self.assertEqual(handle.tell(), 0)
            self.assertEqual(kwargs, {"weights_only": True, "map_location": "cpu"})
            return real_load(handle, **kwargs)

        with mock.patch.object(checkpoints, "_digest_handle",
                               wraps=checkpoints._digest_handle) as hasher:
            with mock.patch.object(torch, "load", side_effect=safe_load) as loader:
                checkpoints.load_weights(self.model, self.path, sha256=digest)
        self.assertIs(hasher.call_args.args[0], loader.call_args.args[0])
        loader.assert_called_once()

    def test_wrong_hash_rejected_before_deserialization(self):
        digest = self.save(self.weights)
        before = self.snapshot()
        identities = [id(p) for p in self.model.parameters()]
        bad_digest = ("0" if digest[0] != "0" else "1") + digest[1:]
        with mock.patch.object(torch, "load") as loader:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                checkpoints.load_weights(self.model, self.path, sha256=bad_digest)
            loader.assert_not_called()
        self.assert_unchanged(before, True, identities)

    def test_hash_is_required_and_must_be_64_hex_characters(self):
        self.save(self.weights)
        with self.assertRaises(TypeError):
            checkpoints.load_weights(self.model, self.path)
        for digest in (None, "", "0" * 63, "0" * 65, "g" * 64, 123):
            with self.subTest(digest=digest):
                with self.assertRaises(ValueError):
                    checkpoints.load_weights(self.model, self.path, sha256=digest)

    def test_missing_key_rejected_without_mutation(self):
        self.reject({"weight": self.weights["weight"]}, ValueError)

    def test_extra_key_rejected_without_mutation(self):
        self.reject({**self.weights, "unexpected": torch.zeros(1)}, ValueError)

    def test_later_shape_mismatch_cannot_partially_load_first_parameter(self):
        self.reject({**self.weights, "bias": torch.zeros(3)}, ValueError)

    def test_later_dtype_mismatch_cannot_partially_load_first_parameter(self):
        self.reject({**self.weights, "bias": self.weights["bias"].double()}, ValueError)

    def test_non_tensor_rejected_without_mutation(self):
        self.reject({**self.weights, "bias": [4.0, 4.0]}, TypeError)

    def test_non_string_state_key_rejected_without_mutation(self):
        self.reject({**self.weights, 3: torch.zeros(1)}, TypeError)

    def test_invalid_or_missing_explicit_key_rejected_without_mutation(self):
        for key in ("missing", "model.state_dict", "", 3):
            with self.subTest(key=key):
                self.reject({"model_state_dict": self.weights},
                            (KeyError, TypeError), key=key)

    def test_nested_or_wrapped_state_is_never_guessed(self):
        self.reject({"model_state_dict": self.weights}, ValueError)
        self.reject({"bundle": {"state": self.weights}}, KeyError, key="bundle.state")

    def test_selected_state_must_be_a_mapping(self):
        self.reject({"weights": [1, 2]}, TypeError, key="weights")
        self.reject([1, 2], TypeError)

    def test_module_prefix_is_not_stripped(self):
        self.reject({"module." + name: value for name, value in self.weights.items()},
                    ValueError)

    def test_unsupported_python_payload_uses_no_unsafe_fallback(self):
        with mock.patch.object(torch, "load", wraps=torch.load) as loader:
            self.reject(UnsupportedPayload(), (pickle.UnpicklingError, RuntimeError))
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs,
                         {"weights_only": True, "map_location": "cpu"})

    def test_meta_tensor_rejected_before_any_parameter_copy(self):
        self.reject({**self.weights, "bias": torch.empty(2, device="meta")}, ValueError)

    def test_layout_mismatch_rejected_before_any_parameter_copy(self):
        self.reject({**self.weights, "bias": self.weights["bias"].to_sparse()}, ValueError)

    def test_changed_file_rejected_before_any_parameter_copy(self):
        digest = self.save(self.weights)
        before = self.snapshot()
        identities = [id(p) for p in self.model.parameters()]
        real_load = torch.load

        def change_after_loading(handle, **kwargs):
            payload = real_load(handle, **kwargs)
            with self.path.open("ab") as stream:
                stream.write(b"changed")
            return payload

        with mock.patch.object(torch, "load", side_effect=change_after_loading):
            with mock.patch.object(self.model, "load_state_dict") as loader:
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    checkpoints.load_weights(self.model, self.path, sha256=digest)
                loader.assert_not_called()
        self.assert_unchanged(before, True, identities)

    def test_only_explicit_local_regular_file_is_accepted(self):
        for path in ("https://example.invalid/weights.pt", "file:///tmp/weights.pt",
                     Path(self.temp.name), self.path):
            with self.subTest(path=path):
                with self.assertRaises((ValueError, FileNotFoundError, IsADirectoryError)):
                    checkpoints.load_weights(self.model, path, sha256="0" * 64)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO paths are POSIX-specific")
    def test_fifo_is_rejected_without_waiting_for_a_writer(self):
        fifo = Path(self.temp.name) / "pipe"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "regular local file"):
            checkpoints.load_weights(self.model, fifo, sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
