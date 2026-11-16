"""Verify the numerical release payload against ``RELEASE_MANIFEST.json``."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ManifestVerificationError(RuntimeError):
    """Raised when a release manifest or payload is invalid."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_safe_relative_path(value: str) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return False
    if path.parts and re.fullmatch(r"[A-Za-z]:", path.parts[0]):
        return False
    return str(path) == value


def is_ignored_payload_path(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in {".git", ".venv", "outputs"}:
        return True
    if "__pycache__" in parts or relative.name == ".DS_Store":
        return True
    if relative.suffix in {".pyc", ".pyo"}:
        return True
    if parts[:3] == ("latex", "L4DC", "build"):
        return True
    if parts[:2] == ("latex", "L4DC") and len(parts) == 3:
        return (
            relative.suffix in {
                ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log",
                ".out", ".pdf",
            }
            or relative.name == "l4dc2027.cls"
            or relative.name.endswith(".synctex.gz")
        )
    return False


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestVerificationError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestVerificationError("manifest root must be a JSON object")
    return payload


def verify_manifest(root: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = (
        root / "RELEASE_MANIFEST.json"
        if manifest_path is None
        else Path(manifest_path).resolve()
    )
    try:
        manifest_relative = manifest_path.relative_to(root)
    except ValueError as error:
        raise ManifestVerificationError("manifest must be inside the release root") from error
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ManifestVerificationError(f"manifest is missing or not a regular file: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    hashes = manifest.get("sha256")
    file_count = manifest.get("file_count")
    if not isinstance(hashes, dict):
        raise ManifestVerificationError("manifest sha256 must be an object")
    if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count < 0:
        raise ManifestVerificationError("manifest file_count must be a nonnegative integer")
    if file_count != len(hashes):
        raise ManifestVerificationError(
            f"manifest file_count {file_count} does not match {len(hashes)} hash entries"
        )

    expected: dict[str, str] = {}
    errors: list[str] = []
    for name, expected_hash in hashes.items():
        if not is_safe_relative_path(name):
            errors.append(f"unsafe manifest path: {name!r}")
            continue
        if name == manifest_relative.as_posix():
            errors.append("manifest must not hash itself")
            continue
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
            errors.append(f"invalid SHA-256 for {name!r}")
            continue
        expected[name] = expected_hash

    actual_names = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative == manifest_relative or is_ignored_payload_path(relative):
            continue
        if path.is_symlink():
            errors.append(f"symlink is not allowed in release payload: {relative.as_posix()}")
        elif path.is_file():
            actual_names.add(relative.as_posix())

    expected_names = set(expected)
    for name in sorted(expected_names - actual_names):
        errors.append(f"missing declared file: {name}")
    for name in sorted(actual_names - expected_names):
        errors.append(f"unexpected undeclared file: {name}")
    for name in sorted(expected_names & actual_names):
        path = root / PurePosixPath(name)
        if path.is_symlink() or not path.is_file():
            errors.append(f"declared path is not a regular file: {name}")
            continue
        actual_hash = digest(path)
        if actual_hash != expected[name]:
            errors.append(
                f"SHA-256 mismatch for {name}: {actual_hash} != {expected[name]}"
            )

    if len(actual_names) != file_count:
        errors.append(
            f"payload file count {len(actual_names)} does not match manifest {file_count}"
        )
    if errors:
        raise ManifestVerificationError("\n".join(errors))
    return {
        "task": "verify_release_manifest",
        "status": "verified",
        "manifest": manifest_relative.as_posix(),
        "file_count": file_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        result = verify_manifest(args.root, args.manifest)
    except ManifestVerificationError as error:
        print(json.dumps({
            "task": "verify_release_manifest",
            "status": "failed",
            "error": str(error),
        }, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
