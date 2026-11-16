import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from kernels import make_kernel_basis
from scripts.run_cross_example_nominal_comparison import assemble_localized_strong
from scripts.run_regularized_baseline_exploration import (
    DEFAULT_CONFIG,
    METHOD_ORDER,
    lasso_sufficient_statistic_system,
    load_baselines,
    paired_comparison_summary,
    protocol_parity_diagnostic,
    read_json,
    select_fd_lasso,
    select_ridge,
    summarize,
    validate_protocol,
    write_decision_report,
)
from scripts.run_morse10d_sweeps import _radial_drift_fast


class RegularizedBaselineExplorationTests(unittest.TestCase):
    def test_protocol_is_pinned_raw_and_current_manuscript(self):
        spec = read_json(DEFAULT_CONFIG)
        root = DEFAULT_CONFIG.parents[2]
        referenced = []
        for example in ("one_dimensional", "predator_prey", "localized_10d"):
            for key, value in spec[example].items():
                if key.endswith("_sha256"):
                    referenced.append(root / spec[example][key[:-7]])
        if all(path.is_file() for path in referenced):
            validate_protocol(spec)
        self.assertEqual(spec["study_status"], "manuscript_current")
        self.assertTrue(spec["regularization"]["raw_columns"])
        self.assertFalse(spec["regularization"]["fit_intercept"])
        self.assertFalse(spec["regularization"]["standardize"])
        self.assertTrue(spec["regularization"]["weak_ridge"]["include_zero_candidate"])
        self.assertTrue(spec["regularization"]["strong_forward_difference_ridge"]["include_zero_candidate"])
        self.assertEqual(spec["regularization"]["strong_forward_difference_lasso"]["alpha_min_ratio"], 1e-9)
        self.assertEqual(spec["predator_prey"]["channel_penalty_policy"],
                         "select_prey_system_and_predator_system_penalties_separately")

    def test_ridge_selection_matches_manual_raw_no_intercept_grid(self):
        A = np.array([[1.0, 10.0], [2.0, 0.0], [0.0, 3.0], [4.0, 1.0]])
        b = np.array([2.0, 1.0, -1.0, 3.0])
        Av = np.array([[3.0, 2.0], [0.5, 8.0], [2.0, -1.0]])
        bv = np.array([2.0, -0.5, 1.5])
        settings = {
            "alpha_count": 7,
            "include_zero_candidate": True,
            "alpha_min_ratio_to_spectral_scale": 1e-5,
            "alpha_max_ratio_to_spectral_scale": 10.0,
            "solver": "svd",
        }
        theta, selection = select_ridge(A, b, Av, bv, settings)
        selected = selection["selected"]
        manual = Ridge(alpha=selected["alpha"], fit_intercept=False, solver="svd").fit(A, b)
        np.testing.assert_allclose(theta, manual.coef_)
        self.assertAlmostEqual(
            selected["validation_mse"], float(np.mean((Av @ theta - bv) ** 2))
        )
        self.assertEqual(selection["candidate_count"], settings["alpha_count"] + 1)
        self.assertEqual(selection["path"][0]["alpha"], 0.0)

    def test_fd_lasso_uses_independent_validation_and_no_intercept(self):
        rng = np.random.default_rng(4)
        A = rng.normal(size=(60, 3))
        truth = np.array([1.0, 0.0, -0.5])
        b = A @ truth + 0.02 * rng.normal(size=60)
        Av = rng.normal(size=(40, 3))
        bv = Av @ truth + 0.02 * rng.normal(size=40)
        settings = {
            "alpha_count": 12, "alpha_min_ratio": 1e-5,
            "max_iter": 20000, "tol": 1e-9,
            "fit_intercept": False, "standardize": False,
            "selection": "cyclic", "report_nonzero_threshold": 1e-8,
        }
        theta, selection = select_fd_lasso(A, b, Av, bv, settings)
        self.assertEqual(theta.shape, (3,))
        self.assertEqual(selection["candidate_count"], 12)
        self.assertAlmostEqual(
            selection["selected"]["validation_mse"],
            float(np.mean((Av @ theta - bv) ** 2)),
        )
        compressed_A, compressed_b = lasso_sufficient_statistic_system(A, b)
        np.testing.assert_allclose(
            compressed_A.T @ compressed_A / compressed_A.shape[0],
            A.T @ A / A.shape[0], rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            compressed_A.T @ compressed_b / compressed_A.shape[0],
            A.T @ b / A.shape[0], rtol=1e-12, atol=1e-12,
        )
        self.assertEqual(selection["objective_compression"],
                         "exact_training_gram_and_score_for_raw_no_intercept_squared_loss")

    def test_localized_strong_system_recovers_forward_euler_coefficients(self):
        dictionary = {
            "basis": "gaussian_shell", "M": 4,
            "centers": [0.5, 1.0, 1.5, 2.0], "h": 0.3,
        }
        interaction = {
            "mode": "gaussian_shell", "centers": dictionary["centers"],
            "h": dictionary["h"], "coefficients": [0.2, -0.4, 0.7, 0.0],
        }
        basis, _ = make_kernel_basis(dictionary)
        x0 = np.random.default_rng(5).normal(size=(8, 2))
        dt = 1e-3
        x1 = x0 + dt * _radial_drift_fast(x0, interaction)
        X = np.stack([x0, x1])[None, ...]
        A, y = assemble_localized_strong(X, np.array([0.0, dt]), np.arange(8), basis)
        theta, *_ = np.linalg.lstsq(A, y, rcond=None)
        np.testing.assert_allclose(theta, interaction["coefficients"], rtol=1e-9, atol=1e-9)

    def test_smoke_baseline_loading_has_three_current_methods_per_example(self):
        spec = read_json(DEFAULT_CONFIG)
        required = [
            Path(spec["one_dimensional"]["baseline_records"]),
            Path(spec["predator_prey"]["baseline_records"]),
            Path(spec["predator_prey"]["strong_baseline_records"]),
            Path(spec["localized_10d"]["baseline_records"]),
        ]
        root = DEFAULT_CONFIG.parents[2]
        if not all((root / path).is_file() for path in required):
            self.skipTest("production baseline records are not distributed")
        records = load_baselines(spec, smoke=True)
        self.assertEqual(len(records), 9)
        for example in ("one_dimensional", "predator_prey", "localized_10d"):
            methods = {row["method"] for row in records if row["example"] == example}
            self.assertEqual(methods, set(METHOD_ORDER[:3]))

    def test_unique_protocol_parity_detects_match_and_mismatch(self):
        A = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        theta = np.array([1.0, -0.5])
        settings = read_json(DEFAULT_CONFIG)["protocol_parity"]
        frozen = {
            "method": "strong_forward_difference_ols",
            "coefficients": theta.tolist(),
            "coefficient_error_relative": 0.2,
            "heldout_drift_error_relative": 0.3,
        }
        matched = protocol_parity_diagnostic(
            example="one_dimensional", repetition=0, form="strong",
            system_label="global", A=A, reconstructed_theta=theta,
            reconstructed_metrics={
                "coefficient_error_relative": 0.2,
                "heldout_drift_error_relative": 0.3,
            }, frozen_record=frozen, settings=settings,
        )
        self.assertEqual(matched["classification"], "unique_system_match")
        self.assertTrue(matched["hard_smoke_passed"])
        mismatched = protocol_parity_diagnostic(
            example="one_dimensional", repetition=0, form="strong",
            system_label="global", A=A,
            reconstructed_theta=theta + np.array([0.1, 0.0]),
            reconstructed_metrics={
                "coefficient_error_relative": 0.4,
                "heldout_drift_error_relative": 0.5,
            }, frozen_record=frozen, settings=settings,
        )
        self.assertEqual(mismatched["classification"], "unique_system_mismatch")
        self.assertFalse(mismatched["hard_smoke_passed"])
        self.assertTrue(mismatched["actual_system_mismatch"])

    def test_rank_deficient_parity_distinguishes_nullspace_nonuniqueness(self):
        A = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        settings = read_json(DEFAULT_CONFIG)["protocol_parity"]
        frozen = {
            "method": "weak_ols", "coefficients": [1.0, 0.0],
            "coefficient_error_relative": 0.4,
            "heldout_drift_error_relative": 0.5,
        }
        parity = protocol_parity_diagnostic(
            example="one_dimensional", repetition=0, form="weak",
            system_label="global", A=A,
            reconstructed_theta=np.array([0.0, 1.0]),
            reconstructed_metrics={
                "coefficient_error_relative": 0.7,
                "heldout_drift_error_relative": 0.8,
            }, frozen_record=frozen, settings=settings,
        )
        self.assertEqual(
            parity["classification"],
            "rank_deficient_nonunique_equivalent_predictions",
        )
        self.assertTrue(parity["expected_nonuniqueness"])
        self.assertFalse(parity["actual_system_mismatch"])
        self.assertFalse(parity["coefficient_match"])

    def test_paired_comparisons_are_deterministic_and_report_requested_quantities(self):
        spec = read_json(DEFAULT_CONFIG)["paired_comparisons"]
        records = []
        values = {
            "weak_lasso": [1.0, 2.0, 3.0],
            "weak_ols": [1.5, 2.5, 3.5],
            "strong_forward_difference_ols": [2.0, 3.0, 4.0],
            "weak_ridge_validation_selected": [0.5, 2.5, 2.0],
            "strong_forward_difference_lasso_validation_selected": [1.5, 2.0, 2.5],
            "strong_forward_difference_ridge_validation_selected": [1.7, 2.3, 3.1],
        }
        for method, metric_values in values.items():
            for repetition, value in enumerate(metric_values):
                records.append({
                    "example": "one_dimensional", "method": method,
                    "repetition": repetition,
                    "coefficient_error_relative": value,
                    "heldout_drift_error_relative": value / 10.0,
                    "status": "finite",
                })
        for example in ("predator_prey", "localized_10d"):
            for row in list(records):
                if row["example"] == "one_dimensional":
                    records.append({**row, "example": example})
        first = paired_comparison_summary(records, spec)
        second = paired_comparison_summary(records, spec)
        self.assertEqual(first, second)
        self.assertEqual(len(first["comparisons"]), 42)
        row = next(item for item in first["comparisons"]
                   if item["example"] == "one_dimensional"
                   and item["metric"] == "coefficient_error_relative"
                   and item["left_method"] == "weak_lasso"
                   and item["right_method"] == "weak_ridge_validation_selected")
        self.assertEqual(row["win_counts"], {"left": 1, "right": 2, "ties": 0})
        self.assertIn("bootstrap_mean_95", row["paired_difference_right_minus_left"])
        self.assertIsNotNone(row["rmse_ratio_paired_bootstrap_95"])

    def test_summary_and_smoke_decision_report_retain_all_six_methods(self):
        records = []
        for example in ("one_dimensional", "predator_prey", "localized_10d"):
            for index, method in enumerate(METHOD_ORDER):
                records.append({
                    "example": example, "method": method, "repetition": 0,
                    "coefficient_error_relative": 0.1 + index,
                    "heldout_drift_error_relative": 0.2 + index,
                    "status": "finite",
                })
        summary = summarize(records)
        self.assertEqual(len(summary["groups"]), 18)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            paired = paired_comparison_summary(
                records, read_json(DEFAULT_CONFIG)["paired_comparisons"]
            )
            parity = []
            write_decision_report(out, summary, paired, parity, smoke=True)
            text = (out / "manuscript_decision_report.md").read_text()
            self.assertIn("do not add any new baseline", text)
            self.assertIn("one-repetition-per-example smoke run", text)


if __name__ == "__main__":
    unittest.main()
