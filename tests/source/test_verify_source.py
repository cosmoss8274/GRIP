"""Source-integrity CLI tests; no scientific model is imported or executed."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parents[1] / "tools" / "verify_source.py"


class VerifySourceTests(unittest.TestCase):
    def setUp(self):
        # Keep all test artifacts inside this task's permitted tests directory.
        self.temporary = tempfile.TemporaryDirectory(prefix=".verify-source-", dir=HERE)
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.root = self.workspace / "project"
        (self.root / "provenance").mkdir(parents=True)
        self.entries = []

    def add_file(self, relative, content, **overrides):
        raw = content.encode("utf-8") if isinstance(content, str) else content
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        checksum = hashlib.sha256(raw).hexdigest()
        entry = {
            "path": relative,
            "source_path": "/original/" + relative,
            "sha256": checksum,
            "source_sha256": checksum,
            "role": "source",
            "status": "copied",
        }
        entry.update(overrides)
        self.entries.append(entry)
        return path

    def run_verify(self, *args):
        manifest = self.root / "provenance" / "source_manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "files": self.entries}))
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--root", str(self.root), "--json", *args],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.stderr, "")
        return result, json.loads(result.stdout)

    def test_valid_sources_and_original_hash_metadata(self):
        self.add_file("source/example.py", "answer = 42\n", source_sha256="0" * 64,
                      status="modified")
        self.add_file("notes.txt", "source inventory\n")
        result, report = self.run_verify()
        self.assertEqual(result.returncode, 0)
        self.assertTrue(report["ok"])
        self.assertEqual(report["checked_files"], 2)
        self.assertEqual(report["parsed_python_files"], 1)

    def test_hash_mismatch(self):
        self.add_file("example.py", "answer = 42\n", sha256="0" * 64)
        result, report = self.run_verify()
        self.assertEqual(result.returncode, 1)
        self.assertIn("sha256", [item["kind"] for item in report["errors"]])

    def test_syntax_error(self):
        self.add_file("example.py", "def unfinished(\n")
        result, report = self.run_verify()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["errors"][0]["kind"], "syntax")
        self.assertEqual(report["parsed_python_files"], 0)

    def test_absolute_and_parent_traversal_paths_are_rejected(self):
        outside = self.workspace / "outside.py"
        outside.write_text("answer = 42\n")
        for path in (str(outside), "../outside.py", "source/../../outside.py", "..\\outside.py"):
            with self.subTest(path=path):
                self.entries = [{"path": path, "sha256": "0" * 64}]
                result, report = self.run_verify()
                self.assertEqual(result.returncode, 1)
                self.assertEqual(report["errors"][0]["kind"], "path")
                self.assertEqual(report["checked_files"], 0)

    def test_symlink_escape_is_rejected(self):
        outside = self.workspace / "outside.py"
        outside.write_text("answer = 42\n")
        link = self.root / "linked.py"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are unavailable")
        self.entries = [{"path": "linked.py", "sha256": "0" * 64}]
        result, report = self.run_verify()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["errors"][0]["kind"], "path")
        self.assertEqual(report["checked_files"], 0)

    def test_parent_symlink_escape_is_rejected(self):
        self.add_file("example.py", "answer = 42\n")
        try:
            (self.root / "linked").symlink_to(self.workspace, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are unavailable")
        self.entries[0]["path"] = "linked/outside.py"
        (self.workspace / "outside.py").write_text("answer = 42\n")
        result, report = self.run_verify()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["errors"][0]["kind"], "path")

    def test_does_not_import_execute_or_create_bytecode(self):
        self.add_file(
            "side_effect.py",
            "from pathlib import Path\n"
            f"Path({str(self.root / 'EXECUTED')!r}).write_text('unsafe')\n"
            "raise RuntimeError('this module must never run')\n",
        )
        result, report = self.run_verify()
        self.assertEqual(result.returncode, 0)
        self.assertTrue(report["ok"])
        self.assertFalse((self.root / "EXECUTED").exists())
        self.assertEqual(list(self.root.rglob("__pycache__")), [])
        self.assertEqual(list(self.root.rglob("*.pyc")), [])

    def test_unlisted_work_and_data_are_not_scanned(self):
        self.add_file("example.py", "answer = 42\n")
        for name in (".work", "data"):
            directory = self.root / name
            directory.mkdir()
            (directory / "unlisted.py").write_text("this is invalid syntax !\n")
        result, report = self.run_verify()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report["checked_files"], 1)

    def test_report_can_only_be_written_inside_root(self):
        self.add_file("example.py", "answer = 42\n")
        outside = self.workspace / "outside-report.json"
        result, report = self.run_verify("--output", str(outside))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["errors"][-1]["kind"], "output")
        self.assertFalse(outside.exists())
        result, report = self.run_verify("--output", "report.json")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads((self.root / "report.json").read_text()), report)

    def test_report_cannot_overwrite_source_or_manifest(self):
        source = self.add_file("example.py", "answer = 42\n")
        for output in ("example.py", "provenance/source_manifest.json"):
            with self.subTest(output=output):
                result, report = self.run_verify("--output", output)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(report["errors"][-1]["kind"], "output")
                self.assertEqual(source.read_text(), "answer = 42\n")
                self.assertEqual(json.loads((self.root / "provenance/source_manifest.json").read_text())["schema_version"], 1)

    def test_report_symlink_is_rejected(self):
        self.add_file("example.py", "answer = 42\n")
        outside = self.workspace / "outside-report.json"
        outside.write_text("unchanged\n")
        try:
            (self.root / "report.json").symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are unavailable")
        result, report = self.run_verify("--output", "report.json")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["errors"][-1]["kind"], "output")
        self.assertEqual(outside.read_text(), "unchanged\n")

    def test_report_parent_symlink_escape_is_rejected(self):
        self.add_file("example.py", "answer = 42\n")
        try:
            (self.root / "linked").symlink_to(self.workspace, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are unavailable")
        result, report = self.run_verify("--output", "linked/outside-report.json")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["errors"][-1]["kind"], "output")
        self.assertFalse((self.workspace / "outside-report.json").exists())

    def test_report_hardlink_cannot_overwrite_outside_file(self):
        self.add_file("example.py", "answer = 42\n")
        outside = self.workspace / "outside-report.json"
        outside.write_text("unchanged\n")
        try:
            os.link(outside, self.root / "report.json")
        except (OSError, NotImplementedError):
            self.skipTest("hard links are unavailable")
        result, report = self.run_verify("--output", "report.json")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["errors"][-1]["kind"], "output")
        self.assertEqual(outside.read_text(), "unchanged\n")


if __name__ == "__main__":
    unittest.main()
