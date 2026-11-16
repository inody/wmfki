"""Independently verify a cross-example nominal-comparison output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = {
    "weak_lasso", "weak_ols", "strong_forward_difference_ols"
}
EXAMPLES = {"one_dimensional", "predator_prey", "localized_10d"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_error(theta, truth):
    return float(np.linalg.norm(np.asarray(theta) - truth) / np.linalg.norm(truth))


def verify(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    protocol = json.loads((out / "requested_protocol.json").read_text())
    records = json.loads((out / "records.json").read_text())
    summary = json.loads((out / "summary.json").read_text())

    artifact_checks = {}
    for name, expected in manifest["artifact_sha256"].items():
        actual = digest(out / name)
        artifact_checks[name] = {"expected": expected, "actual": actual,
                                 "passed": actual == expected}
    input_checks = {}
    for example in EXAMPLES:
        source = Path(protocol[example]["source_records"])
        actual = digest(source)
        expected = protocol[example]["source_sha256"]
        input_checks[example] = {"expected": expected, "actual": actual,
                                 "passed": actual == expected}

    truth = {
        "one_dimensional": np.asarray(protocol["one_dimensional"]["truth"], dtype=float),
        "predator_prey": np.asarray(
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
             0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.4, 0.0, 0.0], dtype=float
        ),
        "localized_10d": np.asarray([0.0, 1.5, -1.0, 0.0], dtype=float),
    }
    coefficient_checks = []
    for row in records:
        recomputed = relative_error(row["coefficients"], truth[row["example"]])
        stored = float(row["coefficient_error_relative"])
        coefficient_checks.append(bool(np.isclose(recomputed, stored, rtol=1e-12, atol=1e-12)))

    expected_repetitions = {
        "one_dimensional": 1 if manifest["smoke"] else protocol["one_dimensional"]["repetitions"],
        "predator_prey": 1 if manifest["smoke"] else protocol["predator_prey"]["repetitions"],
        "localized_10d": 1 if manifest["smoke"] else protocol["localized_10d"]["repetitions"],
    }
    group_checks = {}
    for example in EXAMPLES:
        for method in METHODS:
            rows = [row for row in records if row["example"] == example and row["method"] == method]
            key = f"{example}:{method}"
            group_checks[key] = {
                "count": len(rows), "expected": expected_repetitions[example],
                "finite": all(np.isfinite(row["coefficient_error_relative"]) for row in rows),
                "passed": len(rows) == expected_repetitions[example]
                          and all(np.isfinite(row["coefficient_error_relative"]) for row in rows),
            }

    summary_groups = {(row["example"], row["method"]): row for row in summary["groups"]}
    summary_checks = {}
    for example in EXAMPLES:
        for method in METHODS:
            values = np.asarray([
                row["coefficient_error_relative"] for row in records
                if row["example"] == example and row["method"] == method
            ], dtype=float)
            expected_rmse = float(np.sqrt(np.mean(values ** 2)))
            stored = float(summary_groups[(example, method)]["rmse"])
            summary_checks[f"{example}:{method}"] = {
                "recomputed_rmse": expected_rmse, "stored_rmse": stored,
                "passed": bool(np.isclose(expected_rmse, stored, rtol=1e-12, atol=1e-12)),
            }

    passed = (
        all(item["passed"] for item in artifact_checks.values())
        and all(item["passed"] for item in input_checks.values())
        and all(coefficient_checks)
        and all(item["passed"] for item in group_checks.values())
        and all(item["passed"] for item in summary_checks.values())
        and set(row["example"] for row in records) == EXAMPLES
        and set(row["method"] for row in records) == METHODS
    )
    report = {
        "passed": bool(passed), "artifact_checks": artifact_checks,
        "input_checks": input_checks,
        "coefficient_recomputations": {
            "count": len(coefficient_checks), "passed": int(sum(coefficient_checks)),
        },
        "group_checks": group_checks, "summary_checks": summary_checks,
    }
    (out / "independent_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if not passed:
        raise SystemExit("cross-example nominal comparison verification failed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    report = verify(args.out)
    print(json.dumps({"passed": report["passed"],
                      "records": report["coefficient_recomputations"]["count"]}))


if __name__ == "__main__":
    main()
