"""Exercise release loading with small file fixtures and no network or model."""

from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from grip_models import pretrained


class PretrainedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="grip-pretrained-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.files = {
            "fixture.json": b'{"fixture": true}',
            "fixture.pt": b"unit-test checkpoint bytes, never deserialized",
        }
        self.entry = {
            "config": ("fixture.json", hashlib.sha256(self.files["fixture.json"]).hexdigest()),
            "weights": ("fixture.pt", hashlib.sha256(self.files["fixture.pt"]).hexdigest()),
        }
        registry = patch.dict(pretrained._MODELS, {"mhc-i-energy": self.entry}, clear=True)
        registry.start()
        self.addCleanup(registry.stop)

    def populate(self, cache=None):
        cache = self.cache if cache is None else cache
        cache.mkdir(parents=True)
        for filename, content in self.files.items():
            (cache / filename).write_bytes(content)

    @contextmanager
    def runtime(self):
        config = {"fixture": True}
        model = Mock()
        order = Mock()
        module = ModuleType("grip_models.model.energy")
        module.BaseEnergyModel = Mock(return_value=model)
        order.attach_mock(module.BaseEnergyModel, "construct")
        order.attach_mock(model.to, "to")
        order.attach_mock(model.eval, "eval")
        with patch.dict("sys.modules", {module.__name__: module}), \
                patch("grip_models.inference.load_config", return_value=config) as load_config, \
                patch("grip_models.checkpoints.load_weights") as load_weights:
            order.attach_mock(load_config, "config")
            order.attach_mock(load_weights, "weights")
            yield model, config, order

    def test_downloads_fixed_assets_then_loads_in_order_and_reuses_cache(self):
        requested = []

        def download(request, timeout):
            requested.append(request.full_url)
            self.assertEqual(timeout, 60)
            filename = request.full_url.rsplit("/", 1)[1]
            return io.BytesIO(self.files[filename])

        with self.runtime() as (model, config, order), \
                patch.object(pretrained, "urlopen", side_effect=download):
            result = pretrained.load_pretrained(cache_dir=self.cache, device="test-device")
            self.assertEqual(result, (model, config))
            self.assertEqual([call[0] for call in order.mock_calls],
                             ["config", "construct", "weights", "to", "eval"])
            order.weights.assert_called_once_with(
                model, self.cache / "fixture.pt", sha256=self.entry["weights"][1])
            order.to.assert_called_once_with("test-device")
            order.eval.assert_called_once_with()
            pretrained.load_pretrained(cache_dir=self.cache)
        self.assertEqual(requested, [pretrained._RELEASE_URL + "/fixture.json",
                                     pretrained._RELEASE_URL + "/fixture.pt"])
        self.assertEqual(sorted(path.name for path in self.cache.iterdir()), sorted(self.files))
        for filename, content in self.files.items():
            self.assertEqual((self.cache / filename).read_bytes(), content)

    def test_valid_cache_is_usable_without_network(self):
        self.populate()
        with self.runtime() as (model, config, _), \
                patch.object(pretrained, "urlopen", side_effect=AssertionError("network used")):
            for offline in (False, True):
                self.assertEqual(pretrained.load_pretrained(
                    cache_dir=self.cache, local_files_only=offline), (model, config))

    def test_missing_offline_file_never_attempts_network_or_model_loading(self):
        self.populate()
        (self.cache / "fixture.pt").unlink()
        with self.runtime() as (_, _, order), \
                patch.object(pretrained, "urlopen", side_effect=AssertionError("network used")):
            with self.assertRaisesRegex(FileNotFoundError, "fixture.pt"):
                pretrained.load_pretrained(cache_dir=self.cache, local_files_only=True)
            self.assertEqual(order.mock_calls, [])
            with self.assertRaisesRegex(FileNotFoundError, "fixture.json"):
                pretrained.load_pretrained(cache_dir=self.root / "empty", local_files_only=True)

    def test_corrupt_cache_is_rejected_before_model_loading(self):
        self.populate()
        (self.cache / "fixture.pt").write_bytes(b"damaged")
        with self.runtime() as (_, _, order), \
                patch.object(pretrained, "urlopen", side_effect=AssertionError("network used")):
            with self.assertRaisesRegex(ValueError, "Invalid cached pretrained file"):
                pretrained.load_pretrained(cache_dir=self.cache)
            self.assertEqual(order.mock_calls, [])
        self.assertEqual((self.cache / "fixture.pt").read_bytes(), b"damaged")

    def test_non_file_cache_entry_is_rejected(self):
        (self.cache / "fixture.json").mkdir(parents=True)
        with patch.object(pretrained, "urlopen", side_effect=AssertionError("network used")):
            with self.assertRaisesRegex(ValueError, "Invalid cached pretrained file"):
                pretrained.load_pretrained(cache_dir=self.cache)

    def test_corrupt_download_is_not_committed(self):
        with patch.object(pretrained, "urlopen", return_value=io.BytesIO(b"damaged")):
            with self.assertRaisesRegex(ValueError, "failed verification"):
                pretrained.load_pretrained(cache_dir=self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_interrupted_download_removes_partial_file(self):
        class Interrupted(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise OSError("connection interrupted")
                return super().read(4)

        with patch.object(pretrained, "urlopen", return_value=Interrupted(b"partial payload")):
            with self.assertRaisesRegex(OSError, "connection interrupted"):
                pretrained.load_pretrained(cache_dir=self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_atomic_replace_failure_removes_temporary_file(self):
        with patch.object(pretrained, "urlopen", return_value=io.BytesIO(self.files["fixture.json"])), \
                patch.object(pretrained.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                pretrained.load_pretrained(cache_dir=self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_unknown_model_rejected_without_cache_or_network(self):
        with patch.object(pretrained, "urlopen", side_effect=AssertionError("network used")):
            for name in ("unknown", "", None, []):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown pretrained model"):
                    pretrained.load_pretrained(name, cache_dir=self.cache)
        self.assertFalse(self.cache.exists())

    def test_cache_uses_xdg_home_and_release_directory(self):
        expected = self.root / "grip" / pretrained._RELEASE
        self.populate(expected)
        with self.runtime() as (_, _, order), \
                patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.root)}), \
                patch.object(pretrained, "urlopen", side_effect=AssertionError("network used")):
            pretrained.load_pretrained(local_files_only=True)
            order.config.assert_called_once_with(expected / "fixture.json")

    def test_default_cache_uses_home_when_xdg_is_unset(self):
        expected = self.root / ".cache" / "grip" / pretrained._RELEASE
        self.populate(expected)
        with self.runtime() as (_, _, order), patch.dict(os.environ, {}, clear=True), \
                patch.object(pretrained.Path, "home", return_value=self.root), \
                patch.object(pretrained, "urlopen", side_effect=AssertionError("network used")):
            pretrained.load_pretrained(local_files_only=True)
            order.config.assert_called_once_with(expected / "fixture.json")


if __name__ == "__main__":
    unittest.main()
