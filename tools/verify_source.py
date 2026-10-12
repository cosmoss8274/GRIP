#!/usr/bin/env python3
"""Verify a source manifest without importing or executing listed source files.

Only paths named by provenance/source_manifest.json are read. This tool checks
source integrity and Python syntax; it does not validate scientific results or
prove that a model, package, or service can run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys


MANIFEST_PATH = "provenance/source_manifest.json"
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


def _inside(path: Path, root: Path) -> bool:
    """Use path components, rather than a string prefix, for containment."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _source_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("path must be a nonempty relative string")
    portable = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        portable.is_absolute()
        or windows.drive
        or windows.root
        or "\\" in relative
        or ".." in portable.parts
        or portable == PurePosixPath(".")
        or "\x00" in relative
    ):
        raise ValueError("absolute paths, traversal, and ambiguous separators are forbidden")
    candidate = (root / portable).resolve(strict=True)
    if not _inside(candidate, root):
        raise ValueError("path resolves outside the project root")
    if not candidate.is_file():
        raise ValueError("path does not name a regular file")
    return candidate


def _error(report: dict, kind: str, path: object, message: str) -> None:
    report["errors"].append({"kind": kind, "path": str(path), "message": message})
    report["ok"] = False


def verify(root: Path) -> tuple[dict, set[Path]]:
    """Return a report and protected source paths; perform no writes."""
    report = {
        "schema_version": 1,
        "ok": True,
        "root": str(root),
        "manifest": MANIFEST_PATH,
        "listed_files": 0,
        "checked_files": 0,
        "parsed_python_files": 0,
        "errors": [],
    }
    protected = set()
    try:
        manifest_path = _source_path(root, MANIFEST_PATH)
        protected.add(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError) as exc:
        _error(report, "manifest", MANIFEST_PATH, str(exc))
        return report, protected
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest.get("files"), list)
    ):
        _error(report, "manifest", MANIFEST_PATH, "expected schema_version 1 and a files list")
        return report, protected

    report["listed_files"] = len(manifest["files"])
    for index, entry in enumerate(manifest["files"]):
        if not isinstance(entry, dict):
            _error(report, "entry", f"files[{index}]", "entry must be an object")
            continue
        relative = entry.get("path")
        try:
            source = _source_path(root, relative)
            protected.add(source)
        except (OSError, ValueError, RuntimeError) as exc:
            _error(report, "path", relative, str(exc))
            continue
        expected = entry.get("sha256")
        if not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(expected):
            _error(report, "entry", relative, "sha256 must contain 64 hexadecimal digits")
            continue
        is_python = PurePosixPath(relative).suffix.lower() == ".py"
        digest = hashlib.sha256()
        parts = [] if is_python else None
        try:
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    if parts is not None:
                        parts.append(chunk)
        except OSError as exc:
            _error(report, "read", relative, str(exc))
            continue
        report["checked_files"] += 1
        actual = digest.hexdigest()
        if actual != expected.lower():
            _error(report, "sha256", relative, f"expected {expected.lower()}, found {actual}")
        if is_python:
            try:
                # Parsing bytes respects Python encoding declarations and never
                # executes the source or produces a bytecode cache.
                ast.parse(b"".join(parts), filename=relative)
                report["parsed_python_files"] += 1
            except (SyntaxError, ValueError, UnicodeError) as exc:
                if isinstance(exc, SyntaxError):
                    detail = f"line {exc.lineno}: {exc.msg}"
                else:
                    detail = str(exc)
                _error(report, "syntax", relative, detail)
    return report, protected


def _write_report(output: str, root: Path, report: dict, protected: set[Path]) -> None:
    raw = Path(output)
    if ".." in raw.parts or "\\" in output or "\x00" in output:
        raise ValueError("output path must not contain traversal or ambiguous separators")
    requested = raw if raw.is_absolute() else root / raw
    if requested.is_symlink():
        raise ValueError("output must not be a symlink")
    target = requested.resolve(strict=False)
    if not _inside(target, root):
        raise ValueError("output path resolves outside the project root")
    if target in protected or target == (root / MANIFEST_PATH).resolve(strict=False):
        raise ValueError("output must not overwrite the manifest or a listed source file")
    if target.exists():
        if not target.is_file():
            raise ValueError("output must name a regular file")
        if target.stat().st_nlink > 1:
            raise ValueError("output must not overwrite a file with hard-link aliases")
    if not target.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    # Do not create directories, temporary files, or other implicit artifacts.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
        help="project root (default: parent of this tools directory)",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON report to stdout")
    parser.add_argument(
        "--output", help="also write JSON to this file inside root; its parent must exist",
    )
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("project root must be a directory")
    except (OSError, ValueError, RuntimeError) as exc:
        report = {"schema_version": 1, "ok": False, "errors": [
            {"kind": "root", "path": str(args.root), "message": str(exc)}
        ]}
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"FAIL: invalid project root: {exc}")
        return 1
    report, protected = verify(root)
    if args.output:
        try:
            _write_report(args.output, root, report, protected)
        except (OSError, ValueError, RuntimeError) as exc:
            _error(report, "output", args.output, str(exc))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        verdict = "PASS" if report["ok"] else "FAIL"
        print(
            f"{verdict}: {report['checked_files']}/{report['listed_files']} files checked; "
            f"{report['parsed_python_files']} Python files parsed; "
            f"{len(report['errors'])} errors."
        )
        for item in report["errors"]:
            print(f"  {item['kind']}: {item['path']}: {item['message']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
