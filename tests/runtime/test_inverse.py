"""Inverse-model loading and inference tests with a synthetic backbone."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from grip_models import inverse


class TinyESM(nn.Module):
    num_layers = 33
    embed_dim = 1280


class TinyBackbone(nn.Module):
    def __init__(self, esm_path):
        super().__init__()
        self.config = {
            "num_recycle": 3,
            "embeddings_and_seqformer": {
                "lm": {"model_path": str(esm_path), "enabled": True}
            },
        }
        self.impl = nn.Module()
        self.impl.seqformer = nn.Module()
        self.impl.seqformer.esm = SimpleNamespace(
            model=TinyESM(), repr_layer=33, device=None
        )
        self.proj = nn.Linear(3, 21)
        self.pair = nn.Parameter(torch.arange(441, dtype=torch.float32))
        self.calls = 0

    def forward(self, batch, compute_loss=False):
        if not compute_loss:
            raise AssertionError("Pair/sidechain heads must be requested")
        self.calls += 1
        b, n = batch["seq"].shape
        # Model the upstream mutation of its top-level batch dictionary.
        batch["is_recycling"] = False
        batch["heads"] = {"from_previous_call": True}
        logits = self.proj(batch["left_gt_calpha3_frame_positions"])
        pair = self.pair.expand(b, n, n, 441) + 0.0
        return {
            "representations": {"seq": torch.zeros(b, n, 384)},
            "heads": {"seqhead": {"logits": logits}, "pairhead": {"logits": pair}},
        }


def features():
    b, n = 1, 4
    return {
        "seq": torch.tensor([[0, 1, 20, 20]]),
        "mask": torch.ones(b, n, dtype=torch.bool),
        "chain_id": torch.zeros(b, n, dtype=torch.int32),
        "chain_types": torch.tensor([[0, 1, 3, 3]]),
        "peptide_mask": torch.tensor([[False, False, True, True]]),
        "dist_one_hot": torch.zeros(b, n, n, 500),
        "pair_mask": torch.ones(b, n, n, dtype=torch.bool),
        "left_gt_calpha3_frame_positions": torch.arange(12).float().reshape(b, n, 3),
        "right_gt_calpha3_frame_positions": torch.zeros(b, n, 3),
        "rigidgroups_gt_frames": (
            torch.eye(3).expand(b, n, 8, 3, 3).clone(),
            torch.zeros(b, n, 8, 3),
        ),
    }


class InverseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        esm = self.directory / "esm2_t33_650M_UR50D.pt"
        esm.touch()
        esm.with_name(esm.stem + "-contact-regression.pt").touch()
        self.backbone = TinyBackbone(esm)
        config = copy.deepcopy(self.backbone.config)
        del config["embeddings_and_seqformer"]["lm"]["model_path"]
        digest = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        # Only this stand-in's configuration is accepted in these unit tests.
        patch = mock.patch.object(inverse, "_MODEL_CONFIG_SHA256", digest)
        patch.start()
        self.addCleanup(patch.stop)
        self.state = {
            "carbondesign." + k: v.detach().clone()
            for k, v in self.backbone.state_dict().items()
        }
        self.state["chain_type_embedding.weight"] = torch.ones(4, 384)
        self.path = self.directory / "inverse.pt"
        self.save(self.state)

    def save(self, payload):
        torch.save(payload, self.path)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def load(self, **kwargs):
        return inverse.load_inverse_model(
            self.backbone, self.path, sha256=self.digest, **kwargs
        )

    def test_raw_outputs_full_axes_fresh_mapping_and_repeat(self):
        model = self.load()
        batch = features()
        before = copy.deepcopy(batch)
        expected = self.backbone(dict(batch), compute_loss=True)
        output = inverse.inverse_logits(model, batch)
        repeated = inverse.inverse_logits(model, batch)
        self.assertFalse(model.training)
        self.assertFalse(self.backbone.impl.seqformer.esm.model.training)
        self.assertEqual(output["logits"].shape, (1, 4, 21))
        self.assertEqual(output["pair_logits"].shape, (1, 4, 4, 441))
        for key, head in (("logits", "seqhead"), ("pair_logits", "pairhead")):
            self.assertTrue(torch.equal(output[key], expected["heads"][head]["logits"]))
            self.assertTrue(torch.equal(output[key], repeated[key]))
            self.assertFalse(output[key].requires_grad)
        self.assertTrue(torch.equal(output["peptide_mask"], batch["peptide_mask"]))
        self.assertEqual(set(batch), set(before))
        for key in before:
            if isinstance(before[key], torch.Tensor):
                self.assertTrue(torch.equal(before[key], batch[key]), key)

    def test_unused_chain_embedding_does_not_change_reference_logits(self):
        model = self.load()
        before = inverse.inverse_logits(model, features())
        with torch.no_grad():
            model.chain_type_embedding.weight.fill_(123.0)
        after = inverse.inverse_logits(model, features())
        self.assertTrue(torch.equal(before["logits"], after["logits"]))
        self.assertTrue(torch.equal(before["pair_logits"], after["pair_logits"]))

    def test_complete_weights_required_without_prefix_or_missing_head_fallback(self):
        for missing in ("chain_type_embedding.weight", "carbondesign.pair"):
            with self.subTest(missing=missing):
                self.save({k: v for k, v in self.state.items() if k != missing})
                with self.assertRaisesRegex(ValueError, "key mismatch"):
                    self.load()
        self.assertEqual(self.backbone.calls, 0)

    def test_explicit_safe_training_key(self):
        self.save({"model_state_dict": self.state, "epoch": 21})
        model = self.load(state_dict_key="model_state_dict")
        self.assertEqual(set(model.state_dict()), set(self.state))

    def test_missing_checkpoint_and_hash_mismatch_do_not_fall_back(self):
        self.digest = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.load()
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_changed_recycling_or_disabled_esm_rejected(self):
        self.backbone.config["num_recycle"] = 0
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.load()
        self.backbone.config["num_recycle"] = 3
        self.backbone.config["embeddings_and_seqformer"]["lm"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.load()

    def test_wrong_esm_layer_and_missing_local_resource(self):
        self.backbone.impl.seqformer.esm.repr_layer = 32
        with self.assertRaisesRegex(ValueError, "ESM2"):
            self.load()
        self.backbone.impl.seqformer.esm.repr_layer = 33
        next(self.directory.glob("*-contact-regression.pt")).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "ESM"):
            self.load()

    def test_invalid_peptide_selection_or_recycled_state_rejected(self):
        model = self.load()
        cases = []
        batch = features()
        batch["heads"] = {}
        cases.append(batch)
        batch = features()
        batch["peptide_mask"] = torch.ones(1, 2, dtype=torch.bool)
        cases.append(batch)
        batch = features()
        batch["peptide_mask"] = batch["peptide_mask"].float()
        cases.append(batch)
        batch = features()
        batch["peptide_mask"].zero_()
        cases.append(batch)
        batch = features()
        batch["mask"][0, 3] = False
        cases.append(batch)
        for index, batch in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                inverse.inverse_logits(model, batch)
        self.assertEqual(self.backbone.calls, 0)

    def test_nonfinite_model_output_rejected(self):
        model = self.load()
        batch = features()
        batch["left_gt_calpha3_frame_positions"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite values"):
            inverse.inverse_logits(model, batch)

    def test_training_mode_rejected(self):
        model = self.load().train()
        with self.assertRaisesRegex(ValueError, "eval"):
            inverse.inverse_logits(model, features())


if __name__ == "__main__":
    unittest.main()
