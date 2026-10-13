"""Legacy pretrained calls must fail without using downloads or cached weights."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from grip_models import load_pretrained


class PretrainedTests(unittest.TestCase):
    def test_legacy_signatures_fail_before_network_or_file_access(self):
        with patch("urllib.request.urlopen") as download, \
                patch("builtins.open") as open_file, \
                patch.object(Path, "open") as open_path, \
                patch.object(Path, "mkdir") as make_directory:
            for options in ({}, {"device": "cpu"}, {"local_files_only": True},
                            {"cache_dir": "legacy-cache", "local_files_only": False}):
                with self.subTest(options=options), self.assertRaisesRegex(
                    RuntimeError, "Load a local checkpoint with load_weights"
                ):
                    load_pretrained("mhc-i-energy", **options)
            download.assert_not_called()
            open_file.assert_not_called()
            open_path.assert_not_called()
            make_directory.assert_not_called()

    def test_existing_cache_never_bypasses_withdrawn_downloads(self):
        with tempfile.TemporaryDirectory(prefix="grip-legacy-cache-") as directory:
            cache = Path(directory)
            files = {
                "grip-mhc-i-energy.json": b'{"legacy": true}',
                "grip-mhc-i-energy.pt": b"legacy checkpoint fixture; never deserialized",
            }
            for name, content in files.items():
                (cache / name).write_bytes(content)
            for offline in (False, True):
                with self.assertRaisesRegex(RuntimeError, "Pretrained downloads are not available"):
                    load_pretrained(cache_dir=cache, local_files_only=offline)
            self.assertEqual({p.name: p.read_bytes() for p in cache.iterdir()}, files)

    def test_default_call_reports_local_loading(self):
        with self.assertRaisesRegex(RuntimeError, "load_weights"):
            load_pretrained()


if __name__ == "__main__":
    unittest.main()
