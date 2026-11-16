import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_l4dc_numerical_values import (
    EXAMPLES,
    METHODS,
    METRICS,
    main,
    verify_numerical_values,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference/table1.json"


def load_reference():
    return json.loads(REFERENCE.read_text())


def matching_summary(reference):
    return {
        "task": "cross_example_metric_table",
        "protocol_version": 3,
        "status": "complete",
        "groups": [
            {
                "example": row["example"],
                "method": row["method"],
                "repetitions": row["repetitions"],
                **{metric: float(row[metric]) for metric in METRICS},
            }
            for row in reference["groups"]
        ],
    }


class L4dcNumericalValuesTests(unittest.TestCase):
    def setUp(self):
        self.reference = load_reference()
        self.summary = matching_summary(self.reference)

    def test_success_compares_all_36_reported_values(self):
        result = verify_numerical_values(self.summary, self.reference)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["comparison_count"], 36)
        self.assertTrue(all(row["matches"] for row in result["comparisons"]))
        rounded = next(
            row for row in result["comparisons"]
            if row["example"] == "predator_prey"
            and row["method"] == "strong_forward_difference_ridge_validation_selected"
            and row["metric"] == "coefficient_rmse_relative"
        )
        self.assertEqual(rounded["actual_reported"], "1740")

    def test_corrupted_value_fails_at_three_significant_figures(self):
        self.summary["groups"][0]["coefficient_rmse_relative"] = 2.48
        result = verify_numerical_values(self.summary, self.reference)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("mismatch" in error for error in result["errors"]))

    def test_missing_group_is_rejected(self):
        self.summary["groups"].pop()
        result = verify_numerical_values(self.summary, self.reference)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("missing groups" in error for error in result["errors"]))

    def test_duplicate_group_is_rejected(self):
        self.summary["groups"].append(copy.deepcopy(self.summary["groups"][0]))
        result = verify_numerical_values(self.summary, self.reference)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("duplicate group" in error for error in result["errors"]))

    def test_unexpected_group_is_rejected(self):
        self.summary["groups"][0]["example"] = "unexpected_example"
        result = verify_numerical_values(self.summary, self.reference)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("unexpected group" in error for error in result["errors"]))

    def test_nonfinite_and_negative_values_are_rejected(self):
        self.summary["groups"][0]["coefficient_rmse_relative"] = float("nan")
        self.summary["groups"][1]["drift_rmse_relative"] = -0.1
        result = verify_numerical_values(self.summary, self.reference)
        self.assertEqual(result["status"], "failed")
        self.assertGreaterEqual(
            sum("finite and nonnegative" in error for error in result["errors"]), 2
        )

    def test_invalid_group_identifiers_are_reported_without_raising(self):
        for field, value in (("example", []), ("method", {})):
            with self.subTest(field=field):
                summary = copy.deepcopy(self.summary)
                summary["groups"][0][field] = value
                result = verify_numerical_values(summary, self.reference)
                self.assertEqual(result["status"], "failed")
                self.assertTrue(any("must be strings" in error for error in result["errors"]))

    def test_overflowing_number_and_boolean_protocol_are_rejected(self):
        self.summary["groups"][0]["coefficient_rmse_relative"] = 10 ** 400
        reference = copy.deepcopy(self.reference)
        reference["protocol_version"] = True
        result = verify_numerical_values(self.summary, reference)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("finite and nonnegative" in error for error in result["errors"]))
        self.assertTrue(any("reference protocol_version" in error for error in result["errors"]))

    def test_malformed_reference_and_wrong_precision_are_rejected(self):
        changed = copy.deepcopy(self.reference)
        changed["metrics"] = ["coefficient_rmse_relative"]
        changed["groups"][0]["coefficient_rmse_relative"] = "2.470"
        result = verify_numerical_values(self.summary, changed)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("reference metrics" in error for error in result["errors"]))
        self.assertTrue(any("wrong precision" in error for error in result["errors"]))

    def test_wrong_protocol_repetitions_and_extra_metric_are_rejected(self):
        self.summary["protocol_version"] = 2
        self.summary["groups"][0]["repetitions"] += 1
        self.summary["groups"][0]["extra_metric"] = 1.0
        result = verify_numerical_values(self.summary, self.reference)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("protocol_version" in error for error in result["errors"]))
        self.assertTrue(any("repetitions mismatch" in error for error in result["errors"]))
        self.assertTrue(any("fields changed" in error for error in result["errors"]))

    def test_cli_writes_failure_report_and_returns_nonzero(self):
        self.summary["groups"].pop()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            summary = directory / "summary.json"
            output = directory / "verification.json"
            summary.write_text(json.dumps(self.summary) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main([
                    "--summary", str(summary),
                    "--reference", str(REFERENCE),
                    "--output", str(output),
                ])
            report = json.loads(output.read_text())
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["errors"])

    def test_reference_declares_exact_expected_sets(self):
        self.assertEqual(tuple(self.reference["examples"]), EXAMPLES)
        self.assertEqual(tuple(self.reference["methods"]), METHODS)
        self.assertEqual(tuple(self.reference["metrics"]), METRICS)


if __name__ == "__main__":
    unittest.main()
