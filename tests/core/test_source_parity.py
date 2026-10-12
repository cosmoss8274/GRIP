"""Check fixed reference bytes and the public package's module structure."""

import ast
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from source_reference import (
    MODULE_FILES,
    ORIGINAL_HASHES,
    REFERENCE_ARCHIVE,
    reconstruct_sources,
    validate_sources,
)


class SourceParityTests(unittest.TestCase):
    def test_all_six_original_file_hashes_are_preserved(self):
        recovered, report = validate_sources(ROOT)
        self.assertEqual(set(recovered), set(ORIGINAL_HASHES))
        self.assertEqual(report["files_verified"], 6)
        for name, raw in recovered.items():
            self.assertEqual(hashlib.sha256(raw).hexdigest(), ORIGINAL_HASHES[name])
            ast.parse(raw.decode("utf-8"))

    def test_reference_is_independent_of_runtime_source(self):
        with tempfile.TemporaryDirectory(prefix="grip-reference-") as directory:
            root = Path(directory)
            (root / REFERENCE_ARCHIVE).parent.mkdir(parents=True)
            shutil.copyfile(ROOT / REFERENCE_ARCHIVE, root / REFERENCE_ARCHIVE)
            self.assertFalse((root / "src").exists())
            recovered, report = validate_sources(root)
            self.assertEqual(len(recovered), 6)
            self.assertFalse(report["private_snapshot_required"])

    def test_reference_reconstruction_writes_only_verified_files(self):
        with tempfile.TemporaryDirectory(prefix="grip-reference-") as directory:
            reconstruct_sources(ROOT, Path(directory))
            self.assertEqual(
                {path.name for path in Path(directory).iterdir()}, set(ORIGINAL_HASHES)
            )
            for path in Path(directory).iterdir():
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    ORIGINAL_HASHES[path.name],
                )

    def test_modified_source_bytes_are_rejected_before_writing(self):
        self.check_bad_archive("changed_bytes", "Original source hash mismatch")

    def test_unexpected_archive_paths_are_rejected_before_writing(self):
        self.check_bad_archive("unexpected_path", "exactly the six original files")

    def test_link_members_are_rejected_before_writing(self):
        self.check_bad_archive("link", "Invalid reference archive member")

    def check_bad_archive(self, variant, message):
        recovered, _ = validate_sources(ROOT)
        with tempfile.TemporaryDirectory(prefix="grip-reference-") as directory:
            root = Path(directory)
            archive = root / REFERENCE_ARCHIVE
            archive.parent.mkdir(parents=True)
            with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
                for index, (name, raw) in enumerate(sorted(recovered.items())):
                    if index == 0 and variant == "unexpected_path":
                        name = "../" + name
                    entry = ZipInfo(name)
                    entry.external_attr = (
                        0o120777 if index == 0 and variant == "link" else 0o100644
                    ) << 16
                    if index == 0 and variant == "changed_bytes":
                        raw += b"\n# modified fixture\n"
                    output.writestr(entry, raw)
            destination = root / "restored"
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, message):
                reconstruct_sources(root, destination)
            self.assertEqual(list(destination.iterdir()), [])

    def test_runtime_model_modules_do_not_print(self):
        for target in MODULE_FILES:
            tree = ast.parse((ROOT / target).read_text())
            self.assertFalse(
                any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                    for node in ast.walk(tree)
                ),
                target,
            )

    def test_module_names_are_snake_case_and_old_modules_are_removed(self):
        for target in MODULE_FILES:
            self.assertRegex(Path(target).stem, r"^[a-z][a-z0-9_]*$")
        for old in (
            "sequence.py",
            "common_modules.py",
            "basic.py",
            "EvoMSA_flash.py",
            "EvoPair_flash.py",
            "evoformer_fast.py",
        ):
            self.assertFalse((ROOT / "src/grip_models" / old).exists())

    def test_model_and_dataset_have_descriptive_class_names(self):
        for path, expected in (
            ("model/energy.py", "BaseEnergyModel"),
            ("data.py", "SequenceDataset"),
        ):
            tree = ast.parse((ROOT / "src/grip_models" / path).read_text())
            self.assertIn(
                expected,
                [node.name for node in tree.body if isinstance(node, ast.ClassDef)],
            )

    def test_fixed_reference_is_only_a_development_fixture(self):
        self.assertTrue((ROOT / REFERENCE_ARCHIVE).is_file())
        self.assertFalse((ROOT / "tests/reference/source_map.json").exists())
        self.assertFalse((ROOT / "src/grip_models/source_changes.json").exists())
        self.assertFalse(list((ROOT / "src").rglob("original_sources.zip")))


if __name__ == "__main__":
    unittest.main()
