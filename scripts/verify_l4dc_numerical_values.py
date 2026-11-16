"""Verify generated L4DC Table 1 values against the frozen numerical reference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from scripts.build_cross_example_metric_table import format_three_significant_figures
except ModuleNotFoundError:  # Direct execution as scripts/verify_l4dc_numerical_values.py
    from build_cross_example_metric_table import format_three_significant_figures


EXAMPLES = (
    "one_dimensional",
    "predator_prey",
    "localized_10d",
)
METHODS = (
    "weak_lasso",
    "weak_ridge_validation_selected",
    "weak_ols",
    "strong_forward_difference_lasso_validation_selected",
    "strong_forward_difference_ridge_validation_selected",
    "strong_forward_difference_ols",
)
METRICS = (
    "coefficient_rmse_relative",
    "drift_rmse_relative",
)
GROUP_FIELDS = {"example", "method", "repetitions", *METRICS}


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _is_finite_nonnegative(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value)) and float(value) >= 0.0
    except OverflowError:
        return False


def _index_groups(
    groups: Any,
    *,
    label: str,
    errors: list[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(groups, list):
        errors.append(f"{label}.groups must be a list")
        return indexed
    expected_keys = {(example, method) for example in EXAMPLES for method in METHODS}
    for index, group in enumerate(groups):
        location = f"{label}.groups[{index}]"
        if not isinstance(group, dict):
            errors.append(f"{location} must be an object")
            continue
        fields = set(group)
        if fields != GROUP_FIELDS:
            errors.append(
                f"{location} fields changed: expected {sorted(GROUP_FIELDS)}, "
                f"found {sorted(fields)}"
            )
        key = (group.get("example"), group.get("method"))
        if not all(isinstance(value, str) for value in key):
            errors.append(f"{location} example and method must be strings")
            continue
        if key not in expected_keys:
            errors.append(f"{location} has unexpected group {key}")
            continue
        if key in indexed:
            errors.append(f"{label} has duplicate group {key}")
            continue
        indexed[key] = group
    missing = sorted(expected_keys - set(indexed))
    if missing:
        errors.append(f"{label} is missing groups: {missing}")
    return indexed


def verify_numerical_values(
    summary: Any,
    reference: Any,
) -> dict[str, Any]:
    """Return a complete, manuscript-independent numerical comparison report."""
    errors: list[str] = []
    comparisons: list[dict[str, Any]] = []

    if not isinstance(reference, dict):
        errors.append("reference must be an object")
        reference = {}
    if reference.get("task") != "l4dc_table1_numerical_reference":
        errors.append("reference task is invalid")
    if type(reference.get("protocol_version")) is not int or reference["protocol_version"] != 1:
        errors.append("reference protocol_version must be 1")
    if reference.get("significant_digits") != 3:
        errors.append("reference significant_digits must be 3")
    for field, expected in (
        ("examples", EXAMPLES),
        ("methods", METHODS),
        ("metrics", METRICS),
    ):
        value = reference.get(field)
        if not isinstance(value, list) or tuple(value) != expected:
            errors.append(f"reference {field} must equal {list(expected)}")

    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if summary.get("task") != "cross_example_metric_table":
        errors.append("summary task must be cross_example_metric_table")
    if summary.get("protocol_version") != 3:
        errors.append("summary protocol_version must be 3")
    if summary.get("status") != "complete":
        errors.append("summary status must be complete")

    reference_groups = _index_groups(
        reference.get("groups"), label="reference", errors=errors
    )
    summary_groups = _index_groups(
        summary.get("groups"), label="summary", errors=errors
    )

    for example in EXAMPLES:
        for method in METHODS:
            key = (example, method)
            expected_group = reference_groups.get(key)
            actual_group = summary_groups.get(key)
            if expected_group is None or actual_group is None:
                continue

            expected_repetitions = expected_group.get("repetitions")
            actual_repetitions = actual_group.get("repetitions")
            repetitions_valid = True
            for label, value in (
                ("reference", expected_repetitions),
                ("summary", actual_repetitions),
            ):
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    errors.append(f"{label} group {key} has invalid repetitions: {value!r}")
                    repetitions_valid = False
            repetitions_match = (
                repetitions_valid and actual_repetitions == expected_repetitions
            )
            if repetitions_valid and not repetitions_match:
                errors.append(
                    f"group {key} repetitions mismatch: "
                    f"{actual_repetitions} != {expected_repetitions}"
                )

            for metric in METRICS:
                expected_reported = expected_group.get(metric)
                actual_value = actual_group.get(metric)
                expected_value: float | None = None
                expected_valid = True
                if not isinstance(expected_reported, str):
                    errors.append(
                        f"reference group {key} metric {metric} must be a string"
                    )
                    expected_valid = False
                else:
                    try:
                        expected_value = float(expected_reported)
                    except ValueError:
                        errors.append(
                            f"reference group {key} metric {metric} is not numeric: "
                            f"{expected_reported!r}"
                        )
                        expected_valid = False
                    if expected_valid and not _is_finite_nonnegative(expected_value):
                        errors.append(
                            f"reference group {key} metric {metric} must be finite "
                            "and nonnegative"
                        )
                        expected_valid = False
                    if expected_valid:
                        canonical = format_three_significant_figures(expected_value)
                        if expected_reported != canonical:
                            errors.append(
                                f"reference group {key} metric {metric} has wrong "
                                f"precision: {expected_reported!r} != {canonical!r}"
                            )
                            expected_valid = False

                actual_valid = _is_finite_nonnegative(actual_value)
                if not actual_valid:
                    errors.append(
                        f"summary group {key} metric {metric} must be finite and "
                        f"nonnegative: {actual_value!r}"
                    )
                actual_reported = (
                    format_three_significant_figures(actual_value)
                    if actual_valid else None
                )
                value_match = (
                    expected_valid
                    and actual_valid
                    and actual_reported == expected_reported
                )
                if expected_valid and actual_valid and not value_match:
                    errors.append(
                        f"group {key} metric {metric} mismatch: "
                        f"{actual_reported!r} != {expected_reported!r}"
                    )
                comparisons.append({
                    "example": example,
                    "method": method,
                    "metric": metric,
                    "repetitions": actual_repetitions,
                    "expected_repetitions": expected_repetitions,
                    "repetitions_match": repetitions_match,
                    "actual_value": actual_value if actual_valid else None,
                    "actual_reported": actual_reported,
                    "expected_value": expected_value if expected_valid else None,
                    "expected_reported": expected_reported,
                    "value_match": value_match,
                    "matches": repetitions_match and value_match,
                })

    return {
        "task": "verify_l4dc_numerical_values",
        "protocol_version": 1,
        "status": "verified" if not errors else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_policy": {
            "significant_digits": 3,
            "formatter": "scripts.build_cross_example_metric_table.format_three_significant_figures",
        },
        "expected_group_count": len(EXAMPLES) * len(METHODS),
        "expected_value_count": len(EXAMPLES) * len(METHODS) * len(METRICS),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        summary = json.loads(args.summary.read_text())
        reference = json.loads(args.reference.read_text())
        result = verify_numerical_values(summary, reference)
    except Exception as error:
        result = {
            "task": "verify_l4dc_numerical_values",
            "protocol_version": 1,
            "status": "failed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "comparison_count": 0,
            "comparisons": [],
            "errors": [f"{type(error).__name__}: {error}"],
        }
    result.update({
        "summary": str(args.summary),
        "summary_sha256": digest(args.summary) if args.summary.is_file() else None,
        "reference": str(args.reference),
        "reference_sha256": digest(args.reference) if args.reference.is_file() else None,
    })
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
