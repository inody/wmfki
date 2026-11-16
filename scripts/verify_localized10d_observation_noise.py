"""Verify localized-10D observation-noise sweep artifacts and clean-limit parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = {"weak_lasso", "weak_ols", "strong_forward_difference_ols"}
NOISE_VALUES = {0.0, 0.001, 0.003, 0.005, 0.01, 0.03}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    requested = json.loads((out / "requested_protocol.json").read_text())
    records = json.loads((out / "records.json").read_text())
    effective = json.loads((out / "effective_protocol.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    truth = np.asarray([0.0, 1.5, -1.0, 0.0], dtype=float)

    artifact_checks = {}
    for name, expected in manifest["artifact_sha256"].items():
        actual = digest(out / name)
        artifact_checks[name] = {"actual": actual, "expected": expected,
                                 "passed": actual == expected}
    source_checks = {}
    for name, expected in manifest["source_sha256"].items():
        actual = digest(name)
        source_checks[name] = {"actual": actual, "expected": expected,
                               "passed": actual == expected}

    coefficient_checks = []
    for row in records:
        theta = row.get("coefficients")
        if theta is None:
            coefficient_checks.append(row["coefficient_error_relative"] is None)
            continue
        recomputed = float(np.linalg.norm(np.asarray(theta) - truth) / np.linalg.norm(truth))
        coefficient_checks.append(bool(np.isclose(
            recomputed, row["coefficient_error_relative"], rtol=1e-12, atol=1e-12
        )))

    repetitions = int(manifest["repetitions"])
    group_checks = {}
    expected_noise = set(float(value) for value in effective["sweep"]["observation_noise_values"])
    for sigma_obs in sorted(expected_noise):
        for method in METHODS:
            subset = [row for row in records if row["method"] == method
                      and float(row["observation_noise"]) == sigma_obs]
            passed = len(subset) == repetitions
            group_checks[f"{sigma_obs:g}:{method}"] = {
                "count": len(subset), "expected": repetitions,
                "all_finite": all(row["status"] == "finite" for row in subset),
                "passed": passed,
            }

    summary_lookup = {(row["observation_noise"], row["method"]): row
                      for row in summary["groups"]}
    pairing_checks = {}
    paired_fields = (
        "training_seed", "validation_seed", "test_seed",
        "training_observation_noise_seed", "validation_observation_noise_seed",
        "training_pair_seed", "validation_pair_seed", "target_seed",
        "training_space_test_sha256", "validation_space_test_sha256",
    )
    for repetition in range(repetitions):
        rows = [row for row in records if row["repetition"] == repetition]
        field_counts = {field: len({row[field] for row in rows}) for field in paired_fields}
        pairing_checks[str(repetition)] = {
            "unique_value_counts": field_counts,
            "passed": all(count == 1 for count in field_counts.values()),
        }

    summary_checks = {}
    for key, group in summary_lookup.items():
        sigma_obs, method = key
        values = np.asarray([row["coefficient_error_relative"] for row in records
                             if row["method"] == method
                             and float(row["observation_noise"]) == float(sigma_obs)], dtype=float)
        recomputed = float(np.sqrt(np.mean(values ** 2)))
        stored = float(group["coefficient_error_relative"]["rmse"])
        summary_checks[f"{sigma_obs:g}:{method}"] = {
            "recomputed": recomputed, "stored": stored,
            "passed": bool(np.isclose(recomputed, stored, rtol=1e-12, atol=1e-12)),
        }

    parity_checks = []
    if not manifest["smoke"] and 0.0 in expected_noise:
        clean = json.loads(Path(requested["clean_reference_records"]).read_text())
        clean = [row for row in clean if row["sweep"] == "process_sigma"
                 and float(row["sweep_value"]) == 0.05]
        for repetition in range(repetitions):
            for source_method, new_method in (("lasso", "weak_lasso"), ("ols", "weak_ols")):
                reference = next(row for row in clean if row["repetition"] == repetition
                                 and row["method"] == source_method)
                candidate = next(row for row in records if row["repetition"] == repetition
                                 and row["method"] == new_method
                                 and float(row["observation_noise"]) == 0.0)
                parity_checks.append(bool(
                    np.allclose(reference["coefficients"], candidate["coefficients"],
                                rtol=1e-12, atol=1e-12)
                    and np.isclose(reference["coefficient_error_relative"],
                                   candidate["coefficient_error_relative"],
                                   rtol=1e-12, atol=1e-12)
                ))

    passed = (
        all(item["passed"] for item in artifact_checks.values())
        and all(item["passed"] for item in source_checks.values())
        and all(coefficient_checks)
        and all(item["passed"] for item in group_checks.values())
        and all(item["passed"] for item in pairing_checks.values())
        and all(item["passed"] for item in summary_checks.values())
        and all(parity_checks)
        and set(row["method"] for row in records) == METHODS
        and set(float(row["observation_noise"]) for row in records) == expected_noise
        and manifest["nominal_observation_noise"]
            == requested["nominal_observation_noise"]
    )
    report = {
        "passed": bool(passed), "artifact_checks": artifact_checks,
        "source_checks": source_checks,
        "coefficient_recomputations": {
            "count": len(coefficient_checks), "passed": int(sum(coefficient_checks))
        },
        "clean_limit_parity": {
            "count": len(parity_checks), "passed": int(sum(parity_checks))
        },
        "group_checks": group_checks, "pairing_checks": pairing_checks,
        "summary_checks": summary_checks,
    }
    (out / "independent_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if not passed:
        raise SystemExit("localized-10D observation-noise verification failed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    report = verify(args.out)
    print(json.dumps({
        "passed": report["passed"],
        "records": report["coefficient_recomputations"]["count"],
        "clean_parity": report["clean_limit_parity"],
    }))


if __name__ == "__main__":
    main()
