"""Reconstruct and verify the original six files for development checks."""

import ast
import hashlib
import json
from pathlib import Path
import sys


ORIGINAL_HASHES = {
    "test_energy_nomean_newdata_fix_120.py": "90c87088363230a7fd25108a69943ac78e791052868217a672ac1e001dfb1700",
    "common_modules.py": "a607e30a78b66f9cf6e5f7ba14c5b1427dac84cfd16bc33949fc0c2336f547d1",
    "basic.py": "7e4b1a5e5eac450cd90b29aaff08a08aab8e308b8f93eef8d042719b5b3b7da9",
    "EvoMSA_flash.py": "5e15a4b34632f7440bfb4cdd89aae78fe9a013f8ec41aba8d2ce35f0a0c329e8",
    "EvoPair_flash.py": "9b8620bd8beabb048a44ac96f65811689b89f6ba83e79c9fa5acdc2c549dbff8",
    "evoformer_fast.py": "0df3f87be2d6c4aca738a2766f55b452d6d776b3853b11d30b1321f7d05d87df",
}
MODULE_FILES = {
    "src/grip_models/configuration.py": {},
    "src/grip_models/data.py": {"DataGenerator": "SequenceDataset"},
    "src/grip_models/model/heads.py": {},
    "src/grip_models/model/energy.py": {"myModel": "BaseEnergyModel"},
    "src/grip_models/model/layers.py": {},
    "src/grip_models/model/primitives.py": {},
    "src/grip_models/model/msa.py": {"basic": "primitives"},
    "src/grip_models/model/pair.py": {"basic": "primitives"},
    "src/grip_models/model/evoformer.py": {"EvoPair_flash": "pair", "EvoMSA_flash": "msa"},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def ast_text(value):
    return ast.dump(value, include_attributes=False)


class RestoreNames(ast.NodeTransformer):
    def __init__(self, renames):
        self.reverse = {new: old for old, new in renames.items()}

    def visit_Name(self, node):
        node.id = self.reverse.get(node.id, node.id)
        return node

    def visit_ClassDef(self, node):
        node.name = self.reverse.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_Expr(self, node):
        # Whitespace in documentation strings is not computational source.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            node.value.value = "\n".join(line.rstrip() for line in node.value.value.split("\n"))
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "debug"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "logger"):
            require(not node.keywords and node.args, "Unexpected diagnostic call")
            if len(node.args) > 1:
                require(isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == " ".join(["%s"] * (len(node.args) - 1)),
                        "Unexpected diagnostic formatting")
                node.args = node.args[1:]
            node.func = ast.Name(id="print", ctx=ast.Load())
        return node

    def visit_ImportFrom(self, node):
        if node.level == 1 and node.module is None:
            aliases = [ast.alias(name=self.reverse.get(item.name, item.name), asname=item.asname)
                       for item in node.names]
            return ast.Import(names=aliases)
        return node


def normalized_ast(text, renames):
    return ast_text(RestoreNames(renames).visit(ast.parse(text)))


def is_docstring(node):
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


_SCAFFOLD_STATEMENTS = {
    ast_text(ast.parse(statement).body[0]) for statement in (
        "torch.cuda.is_available()",
        "logger = logging.getLogger(__name__)",
        'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    )
}
_LOGGER_STATEMENT = ast_text(ast.parse("logger = logging.getLogger(__name__)").body[0])


def validate_sources(root):
    root = Path(root)
    fixture = root / "tests/reference/source_map.json"
    manifest = json.loads(fixture.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 2, "Unexpected source-map schema")
    require(set(manifest["modules"]) == set(MODULE_FILES), "Unexpected packaged module set")
    require(len(manifest["files"]) == len(ORIGINAL_HASHES), "Unexpected original source count")
    require({item["source"] for item in manifest["files"]} == set(ORIGINAL_HASHES), "Unexpected original files")
    module_lines, occupied = {}, {}
    for target, details in manifest["modules"].items():
        raw = (root / target).read_bytes()
        require(sha256(raw) == details["sha256"], f"Packaged source mismatch: {target}")
        module_lines[target] = raw.decode("utf-8").splitlines(keepends=True)
        prefix = details["prefix_lines"]
        require(0 <= prefix <= len(module_lines[target]), f"Invalid prefix: {target}")
        for node in ast.parse("".join(module_lines[target][:prefix])).body:
            require(isinstance(node, (ast.Import, ast.ImportFrom)) or is_docstring(node)
                    or ast_text(node) == _LOGGER_STATEMENT,
                    f"Module prefix contains executable logic: {target}")
        occupied[target] = set(range(1, prefix + 1))

    recovered, verified = {}, []
    for item in manifest["files"]:
        name = item["source"]
        require(item["original_source_sha256"] == ORIGINAL_HASHES[name], f"Changed original fingerprint: {name}")
        pieces, next_line = [], 1
        for part in item["parts"]:
            require(part["source_start_line"] == next_line, f"Source gap or overlap: {name}")
            next_line = part["source_end_line"] + 1
            if "import_scaffolding" in part:
                restored = part["import_scaffolding"]
                for node in ast.parse(restored).body:
                    require(isinstance(node, (ast.Import, ast.ImportFrom)) or ast_text(node) in _SCAFFOLD_STATEMENTS,
                            f"Unmapped computational source: {name}")
            else:
                target = part["target"]
                require(target in MODULE_FILES, f"Unexpected target: {target}")
                allowed = MODULE_FILES[target]
                require(part["symbol_renames"] == allowed or (target.endswith("/data.py") and not part["symbol_renames"]),
                        f"Unexpected symbol rename: {target}")
                start, end = part["start_line"], part["end_line"]
                require(1 <= start <= end <= len(module_lines[target]), f"Invalid source range: {target}")
                indices = set(range(start, end + 1))
                require(not (indices & occupied[target]), f"Overlapping source mapping: {target}")
                occupied[target].update(indices)
                current = "".join(module_lines[target][start-1:end])
                lines = current.splitlines(keepends=True)
                changed = set()
                for change in part["replacements"]:
                    index = change["line"] - 1
                    require(0 <= index < len(lines) and index not in changed, f"Invalid replacement: {target}")
                    changed.add(index)
                    content = lines[index].rstrip("\r\n")
                    require(content == change["after"], f"Replacement mismatch: {target}")
                    lines[index] = change["before"] + lines[index][len(content):]
                trailing = part.get("trailing_whitespace", "")
                require(not trailing or trailing.isspace(), f"Non-whitespace source suffix: {target}")
                restored = "".join(lines) + trailing
                require(normalized_ast(current, part["symbol_renames"]) == normalized_ast(restored, {}),
                        f"Computation changed beyond allowed renames: {target}")
            require(len(restored.splitlines()) == part["source_end_line"]-part["source_start_line"]+1,
                    f"Source line count mismatch: {name}")
            pieces.append(restored)
        original = "".join(pieces).encode("utf-8")
        require(sha256(original) == ORIGINAL_HASHES[name], f"Original source hash mismatch: {name}")
        if f"{sys.version_info.major}.{sys.version_info.minor}" == manifest["ast_fingerprint_python"]:
            require(sha256(ast_text(ast.parse(original.decode("utf-8"))).encode()) == item["original_ast_sha256"],
                    f"Original AST mismatch: {name}")
        recovered[name] = original
        verified.append({"restored_filename": name, "original_source_sha256": sha256(original),
                         "original_bytes_and_ast_preserved": True})
    for target, lines in module_lines.items():
        require(occupied[target] == set(range(1, len(lines)+1)), f"Unmapped packaged source: {target}")
    return recovered, {"files_verified": len(verified), "modules_verified": len(module_lines),
                       "private_snapshot_required": False, "files": verified}


def reconstruct_sources(root, destination):
    recovered, report = validate_sources(root)
    for name, raw in recovered.items():
        (Path(destination) / name).write_bytes(raw)
    return report
