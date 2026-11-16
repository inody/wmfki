"""Verify the predator--prey strong FD-OLS observation-noise supplement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_predator_prey_strong_observation_noise import (
    SOURCE_FILES,
    digest,
    summarize,
    weak_pairing_check,
)
from scripts.run_predator_prey_sweeps import read_json


def verify(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    smoke = bool(manifest["smoke"])
    checks = {
        "status": manifest["status"] == (
            "smoke_complete" if smoke else "scientific_complete"
        )
    }
    for name, expected in manifest["artifact_sha256"].items():
        checks[f"artifact:{name}"] = digest(run_dir / name) == expected
    sources = read_json(run_dir / "source_manifest.json")["sha256"]
    checks["source_file_set"] = set(sources) == set(SOURCE_FILES)
    for relative, expected in sources.items():
        checks[f"source:{relative}"] = digest(ROOT / relative) == expected

    requested = read_json(run_dir / "requested_protocol.json")
    spec = read_json(run_dir / "effective_protocol.json")
    records = read_json(run_dir / "records.json")
    repetitions = int(spec["evaluation"]["scientific_repetitions"])
    checks["record_count"] = len(records) == repetitions * 5 == manifest["record_count"]
    checks["method"] = all(
        row["method"] == "strong_forward_difference_ols" for row in records
    )
    checks["repetitions"] = sorted({row["repetition"] for row in records}) \
        == list(range(repetitions))
    for value in requested["observation_noise_values"]:
        count = sum(float(row["condition"]["value"]) == float(value) for row in records)
        checks[f"noise:{value}"] = count == repetitions

    pairing = weak_pairing_check(records, requested)
    checks["pairing_recomputed"] = pairing == read_json(run_dir / "pairing_check.json")
    checks["pairing_passed"] = pairing["passed"] and manifest["pairing_passed"]
    checks["summary_recomputed"] = summarize(records) == read_json(
        run_dir / "summary.json"
    )
    quality = read_json(run_dir / "quality_report.json")
    checks["quality_recomputed"] = quality == {
        "records": len(records),
        "nonfinite": sum(row["status"] != "finite" for row in records),
        "rank_deficient_prey": sum(
            row["prey_system"]["rank_deficient"] for row in records
        ),
        "rank_deficient_predator": sum(
            row["predator_system"]["rank_deficient"] for row in records
        ),
        "pairing_passed": pairing["passed"],
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    return {
        "passed": not failed, "failed_checks": failed,
        "checks": len(checks), "records": len(records),
        "pairing": {"count": pairing["count"],
                    "passed_count": pairing["passed_count"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
