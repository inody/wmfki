"""Keep project test fixtures independent of CPython's optional test package."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectTestPackageTests(unittest.TestCase):
    def test_project_package_precedes_a_competing_standard_library_package(self):
        with tempfile.TemporaryDirectory() as directory:
            competitor = Path(directory)
            package = competitor / "test"
            package.mkdir()
            (package / "__init__.py").write_text("IS_COMPETING_PACKAGE = True\n")
            code = """
from pathlib import Path
import sys
root = Path.cwd().resolve()
sys.path.insert(1, sys.argv[1])
import test
assert Path(test.__file__).resolve() == root / 'test/__init__.py', test.__file__
assert not getattr(test, 'IS_COMPETING_PACKAGE', False)
from test import test_synthetic_protocol_fixtures as fixtures
assert Path(fixtures.__file__).resolve() == root / 'test/test_synthetic_protocol_fixtures.py'
assert fixtures.ROOT == root
"""
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTHONHOME", None)
            completed = subprocess.run(
                [sys.executable, "-S", "-c", code, str(competitor)],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )


if __name__ == "__main__":
    unittest.main()
