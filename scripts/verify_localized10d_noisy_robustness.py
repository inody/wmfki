"""Independently verify a fixed-noise localized 10D robustness run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_localized10d_noisy_robustness import (
    SOURCE_FILES,
    baseline_parity,
    digest,
)
from scripts.run_localized10d_pilot import summarize
from scripts.run_morse10d_sweeps import read_json


def verify(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    smoke = bool(manifest.get("smoke"))
    expected_status = "smoke_completed" if smoke else "scientific_completed"
    checks = {"status": manifest.get("status") == expected_status}

    for name, expected in manifest.get("artifact_sha256", {}).items():
        checks[f"artifact:{name}"] = digest(run_dir / name) == expected
    source_hashes = read_json(run_dir / "source_manifest.json")["sha256"]
    checks["source_file_set"] = set(source_hashes) == set(SOURCE_FILES)
    for relative, expected in source_hashes.items():
        checks[f"source:{relative}"] = digest(ROOT / relative) == expected

    spec = read_json(run_dir / "frozen_protocol.json")
    requested = read_json(run_dir / "requested_protocol.json")
    records = read_json(run_dir / "records.json")
    repetitions = int(spec["evaluation"]["scientific_repetitions"])
    expected_records = repetitions * 2 * sum(
        len(values) for values in spec["sweeps"].values()
    )
    checks["record_count"] = len(records) == expected_records == manifest.get("record_count")
    checks["observation_noise"] = all(
        float(row.get("observation_noise", -1.0))
        == float(requested["observation_noise"]) for row in records
    )
    checks["repetition_indices"] = sorted({row["repetition"] for row in records}) \
        == list(range(repetitions))
    checks["methods"] = {row["method"] for row in records} == {"lasso", "ols"}

    for sweep, values in spec["sweeps"].items():
        for value in values:
            for method in ("lasso", "ols"):
                count = sum(
                    row["sweep"] == sweep and float(row["sweep_value"]) == float(value)
                    and row["method"] == method for row in records
                )
                checks[f"group:{sweep}:{value}:{method}"] = count == repetitions

    recomputed_summary = summarize(records)
    checks["summary_recomputed"] = recomputed_summary == read_json(run_dir / "summary.json")
    quality = read_json(run_dir / "quality_report.json")
    checks["quality_record_count"] = quality["record_count"] == len(records)
    checks["quality_counts"] = quality == {
        "record_count": len(records),
        "rank_deficient": sum(bool(row["rank_deficient"]) for row in records),
        "nonconverged_records": sum(not bool(row["converged"]) for row in records),
        "boundary_selections": sum(bool(row["selected_at_grid_boundary"]) for row in records),
        "numerical_divergence": sum(bool(row["numerical_divergence"]) for row in records),
        "missing_coefficients": sum(row["coefficients"] is None for row in records),
        "baseline_parity_passed": None if smoke else True,
    }

    if smoke:
        checks["baseline_parity_not_applicable"] = read_json(
            run_dir / "baseline_parity.json"
        ) is None
        parity = None
    else:
        parity = baseline_parity(records, requested)
        saved_parity = read_json(run_dir / "baseline_parity.json")
        checks["baseline_parity_recomputed"] = parity == saved_parity
        checks["baseline_parity_passed"] = (
            parity["passed"] and parity["count"] == 40
            and parity["passed_count"] == 40
            and manifest.get("baseline_parity_passed") is True
        )

    failed = sorted(key for key, passed in checks.items() if not passed)
    result = {
        "passed": not failed,
        "failed_checks": failed,
        "checks": len(checks),
        "records": len(records),
        "baseline_parity": None if parity is None else {
            "count": parity["count"], "passed_count": parity["passed_count"]
        },
    }
    return result


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
