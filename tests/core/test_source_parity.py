"""Verify exact source preservation using only the Python standard library.

The recorded original byte and AST fingerprints make these checks usable without
the research snapshot. When that snapshot is present, compare against it too.
Only five explicitly enumerated imports may differ; no AST nodes are discarded.
"""

import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "grip_models"
SOURCE_ROOT = ROOT / ".work" / "research_snapshot"

EXPECTED_FILES = {
    "test_energy_nomean_newdata_fix_120.py": "sequence.py",
    "common_modules.py": "common_modules.py",
    "basic.py": "basic.py",
    "EvoMSA_flash.py": "EvoMSA_flash.py",
    "EvoPair_flash.py": "EvoPair_flash.py",
    "evoformer_fast.py": "evoformer_fast.py",
}

ALLOWED_IMPORT_RELOCATIONS = {
    "sequence.py": [
        (31, "from common_modules import Linear, LayerNorm",
         "from .common_modules import Linear, LayerNorm"),
        (35, "from evoformer_fast import Evoformer",
         "from .evoformer_fast import Evoformer"),
    ],
    "common_modules.py": [],
    "basic.py": [],
    "EvoMSA_flash.py": [(5, "import basic", "from . import basic")],
    "EvoPair_flash.py": [(4, "import basic", "from . import basic")],
    "evoformer_fast.py": [
        (4, "import EvoPair_flash, EvoMSA_flash",
         "from . import EvoPair_flash, EvoMSA_flash"),
    ],
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def ast_text(tree):
    return ast.dump(tree, include_attributes=False)


class SourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((PACKAGE / "source_changes.json").read_text())
        cls.records = cls.manifest["files"]

    def test_manifest_has_exact_six_file_scope_and_five_import_changes(self):
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(len(self.records), len(EXPECTED_FILES))
        seen = set()
        for record in self.records:
            original = Path(record["source"]).name
            self.assertNotIn(original, seen)
            seen.add(original)
            self.assertIn(original, EXPECTED_FILES)
            destination = EXPECTED_FILES[original]
            self.assertEqual(record["source"], f".work/research_snapshot/{original}")
            self.assertEqual(record["target"], f"src/grip_models/{destination}")
            observed = [
                (change["line"], change["before"], change["after"])
                for change in record["import_relocations"]
            ]
            self.assertEqual(observed, ALLOWED_IMPORT_RELOCATIONS[destination])
        self.assertEqual(seen, set(EXPECTED_FILES))

    def test_original_bytes_recovered_by_only_whitelisted_import_reversal(self):
        for record in self.records:
            name = Path(record["target"]).name
            with self.subTest(file=name):
                packaged = (PACKAGE / name).read_bytes()
                self.assertEqual(sha256(packaged), record["packaged_source_sha256"])
                original = packaged
                for _, before, after in ALLOWED_IMPORT_RELOCATIONS[name]:
                    self.assertEqual(original.count(after.encode()), 1)
                    original = original.replace(after.encode(), before.encode(), 1)
                self.assertEqual(sha256(original), record["original_source_sha256"])
                snapshot = SOURCE_ROOT / Path(record["source"]).name
                if snapshot.is_file():
                    self.assertEqual(original, snapshot.read_bytes())

    def test_entire_ast_matches_original_after_exact_import_reversal(self):
        for record in self.records:
            name = Path(record["target"]).name
            with self.subTest(file=name):
                tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
                for line, before, after in ALLOWED_IMPORT_RELOCATIONS[name]:
                    indices = [i for i, node in enumerate(tree.body)
                               if node.lineno == line]
                    self.assertEqual(len(indices), 1)
                    index = indices[0]
                    self.assertEqual(ast_text(tree.body[index]),
                                     ast_text(ast.parse(after).body[0]))
                    tree.body[index] = ast.parse(before).body[0]
                complete_ast = ast_text(tree)
                # Verify the original bytes before using them as the AST
                # baseline. Parsing both trees here avoids treating new empty
                # AST fields in another Python version as source changes.
                original = (PACKAGE / name).read_bytes()
                for _, before, after in ALLOWED_IMPORT_RELOCATIONS[name]:
                    self.assertEqual(original.count(after.encode()), 1)
                    original = original.replace(after.encode(), before.encode(), 1)
                self.assertEqual(sha256(original), record["original_source_sha256"])
                self.assertEqual(complete_ast,
                                 ast_text(ast.parse(original.decode("utf-8"))))
                parser_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                if parser_version == self.manifest["ast_fingerprint_python"]:
                    self.assertEqual(sha256(complete_ast.encode("utf-8")),
                                     record["original_ast_sha256"])
                snapshot = SOURCE_ROOT / Path(record["source"]).name
                if snapshot.is_file():
                    self.assertEqual(complete_ast,
                                     ast_text(ast.parse(snapshot.read_text(encoding="utf-8"))))

    def test_local_imports_resolve_within_package(self):
        local_modules = {Path(name).stem for name in EXPECTED_FILES.values()}
        for name in EXPECTED_FILES.values():
            with self.subTest(file=name):
                tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertNotIn(alias.name.split(".")[0], local_modules)
                    elif isinstance(node, ast.ImportFrom):
                        if node.level:
                            self.assertEqual(node.level, 1)
                            modules = [node.module] if node.module else [
                                alias.name for alias in node.names
                            ]
                            for module in modules:
                                self.assertIn(module, local_modules)
                        else:
                            self.assertNotIn(node.module, local_modules)


if __name__ == "__main__":
    unittest.main()
