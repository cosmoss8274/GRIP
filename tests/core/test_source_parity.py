"""Verify computation-preserving module moves and explicitly allowed renames."""

import ast
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from source_reference import MODULE_FILES, ORIGINAL_HASHES, normalized_ast, reconstruct_sources, validate_sources


class SourceParityTests(unittest.TestCase):
    def test_all_original_bytes_and_computational_asts_are_preserved(self):
        recovered, report = validate_sources(ROOT)
        self.assertEqual(set(recovered), set(ORIGINAL_HASHES))
        self.assertEqual(report["files_verified"], 6)
        self.assertEqual(report["modules_verified"], 9)

    def test_reference_reconstruction_needs_only_public_checkout(self):
        with tempfile.TemporaryDirectory(prefix="grip-source-map-") as directory:
            report = reconstruct_sources(ROOT, Path(directory))
            self.assertEqual({path.name for path in Path(directory).iterdir()}, set(ORIGINAL_HASHES))
            self.assertFalse(report["private_snapshot_required"])

    def test_rename_normalization_does_not_accept_computation_changes(self):
        old = "class myModel:\n    def forward(self, x):\n        return x * 2\n"
        moved = old.replace("myModel", "BaseEnergyModel")
        renames = {"myModel": "BaseEnergyModel"}
        self.assertEqual(normalized_ast(moved, renames), normalized_ast(old, {}))
        self.assertNotEqual(normalized_ast(moved.replace("x * 2", "x * 3"), renames), normalized_ast(old, {}))

    def test_diagnostic_normalization_preserves_argument_expressions(self):
        self.assertEqual(normalized_ast('logger.debug("value: %s" % value)', {}),
                         normalized_ast('print("value: %s" % value)', {}))
        self.assertEqual(normalized_ast('logger.debug("%s %s", "shape:", value.shape)', {}),
                         normalized_ast('print("shape:", value.shape)', {}))
        self.assertNotEqual(normalized_ast('logger.debug(value * 2)', {}),
                            normalized_ast('print(value * 3)', {}))

    def test_runtime_model_modules_do_not_print(self):
        for target in MODULE_FILES:
            tree = ast.parse((ROOT / target).read_text())
            self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                 and node.func.id == "print" for node in ast.walk(tree)), target)

    def test_module_names_are_snake_case_and_old_modules_are_removed(self):
        for target in MODULE_FILES:
            self.assertRegex(Path(target).stem, r"^[a-z][a-z0-9_]*$")
        for old in ("sequence.py", "common_modules.py", "basic.py", "EvoMSA_flash.py", "EvoPair_flash.py", "evoformer_fast.py"):
            self.assertFalse((ROOT / "src/grip_models" / old).exists())

    def test_model_and_dataset_have_descriptive_class_names(self):
        for path, expected in (("model/energy.py", "BaseEnergyModel"), ("data.py", "SequenceDataset")):
            tree = ast.parse((ROOT / "src/grip_models" / path).read_text())
            self.assertIn(expected, [node.name for node in tree.body if isinstance(node, ast.ClassDef)])

    def test_source_map_is_a_development_fixture(self):
        self.assertTrue((ROOT / "tests/reference/source_map.json").is_file())
        self.assertFalse((ROOT / "src/grip_models/source_changes.json").exists())


if __name__ == "__main__":
    unittest.main()
