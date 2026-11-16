"""Independently invoke verification for a localized 10D P=50N production run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_localized10d_sampled_pair_production import digest, verify_output
from scripts.run_morse10d_sweeps import read_json, write_json


def verify(run_dir):
    run_dir = Path(run_dir)
    result = verify_output(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    source_manifest = read_json(run_dir / "source_manifest.json")["sha256"]
    frozen_checks = {}
    for relative, expected in source_manifest.items():
        frozen_path = run_dir / "source" / relative
        frozen_checks[relative] = frozen_path.is_file() and digest(frozen_path) == expected
    extra = {
        "manifest_verification_passed": manifest.get("verification_passed") is True,
        "saved_verification_hash_matches": (
            digest(run_dir / "verification_report.json")
            == manifest.get("verification_report_sha256")
        ),
        "all_frozen_source_snapshots_match": all(frozen_checks.values()),
    }
    failed = list(result["failed_checks"])
    failed.extend(name for name, passed in extra.items() if not passed)
    return {
        "passed": not failed,
        "failed_checks": sorted(failed),
        "runner_verification": result,
        "independent_checks": extra,
        "frozen_source_file_count": len(frozen_checks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = verify(args.run_dir)
    if args.write:
        write_json(args.run_dir / "independent_verification.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
