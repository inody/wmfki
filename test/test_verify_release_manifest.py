import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_release_manifest import (
    ManifestVerificationError,
    is_safe_relative_path,
    verify_manifest,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseManifestVerifierTests(unittest.TestCase):
    def write_release(self, root: Path, files: dict[str, bytes]) -> None:
        hashes = {}
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            hashes[name] = digest(path)
        (root / "RELEASE_MANIFEST.json").write_text(json.dumps({
            "task": "export_wmfki_numerical_release",
            "status": "complete",
            "content_scope": "numerical_reproducibility",
            "file_count": len(hashes),
            "sha256": hashes,
        }, indent=2, sort_keys=True) + "\n")

    def test_safe_relative_path_policy(self):
        self.assertTrue(is_safe_relative_path("reference/table1.json"))
        for value in (
            "", "/absolute", "../escape", "a/../b", "./file", "a\\b",
            "C:/tmp/file", "a//b",
        ):
            self.assertFalse(is_safe_relative_path(value), value)

    def test_verifies_declared_files_and_ignores_user_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_release(root, {
                "README.md": b"numerical release\n",
                "reference/table1.json": b"{}\n",
            })
            (root / "outputs").mkdir()
            (root / "outputs/result.json").write_text("{}\n")
            (root / ".venv").mkdir()
            (root / ".venv/keep.txt").write_text("environment\n")
            result = verify_manifest(root)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["file_count"], 2)

    def test_rejects_altered_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_release(root, {"README.md": b"original\n"})
            (root / "README.md").write_text("changed\n")
            with self.assertRaisesRegex(ManifestVerificationError, "SHA-256 mismatch"):
                verify_manifest(root)

    def test_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_release(root, {"README.md": b"content\n"})
            (root / "README.md").unlink()
            with self.assertRaisesRegex(ManifestVerificationError, "missing declared file"):
                verify_manifest(root)

    def test_rejects_unexpected_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_release(root, {"README.md": b"content\n"})
            (root / "undeclared.txt").write_text("extra\n")
            with self.assertRaisesRegex(ManifestVerificationError, "unexpected undeclared file"):
                verify_manifest(root)

    def test_rejects_unsafe_manifest_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "RELEASE_MANIFEST.json").write_text(json.dumps({
                "file_count": 1,
                "sha256": {"../escape": "0" * 64},
            }))
            with self.assertRaisesRegex(ManifestVerificationError, "unsafe manifest path"):
                verify_manifest(root)

    def test_rejects_inconsistent_count_and_invalid_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "RELEASE_MANIFEST.json").write_text(json.dumps({
                "file_count": 2,
                "sha256": {"README.md": "not-a-hash"},
            }))
            with self.assertRaisesRegex(ManifestVerificationError, "file_count"):
                verify_manifest(root)


if __name__ == "__main__":
    unittest.main()
