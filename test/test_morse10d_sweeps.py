import unittest

import numpy as np

from simulate import make_true_kernel
from scripts.run_morse10d_sweeps import (
    DEFAULT_CONFIG,
    _morse_drift_fast,
    effective_protocol,
    evaluate_theta,
    fit_fixed_alpha,
    physical_conditions,
    read_json,
    repetition_plan,
    summarize,
    validate_protocol,
)


class TestMorse10DSweeps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read_json(DEFAULT_CONFIG)

    def test_scientific_protocol_is_frozen(self):
        validate_protocol(self.spec)
        self.assertEqual(self.spec["evaluation"]["scientific_repetitions"], 20)
        self.assertEqual(self.spec["estimator"]["alpha_count"], 60)
        self.assertEqual(
            self.spec["sweeps"]["process_sigma"], [0.0, 0.025, 0.05, 0.1, 0.2]
        )
        self.assertEqual(self.spec["sweeps"]["particle_number"], [100, 200, 350, 500])
        self.assertEqual(
            self.spec["sweeps"]["sampled_pairs"], [500, 1000, 3000, 5000, 10000]
        )
        self.assertFalse(self.spec["estimator"]["standardize"])
        self.assertFalse(self.spec["estimator"]["post_lasso_refit"])
        self.assertFalse(self.spec["estimator"]["coefficient_thresholding"])

    def test_smoke_is_separate_and_does_not_mutate_scientific_protocol(self):
        smoke = effective_protocol(self.spec, smoke=True)
        self.assertTrue(smoke["smoke_only"])
        self.assertEqual(smoke["evaluation"]["scientific_repetitions"], 1)
        self.assertLess(smoke["model"]["particles"], self.spec["model"]["particles"])
        self.assertLess(smoke["model"]["steps"], self.spec["model"]["steps"])
        self.assertEqual(self.spec["evaluation"]["scientific_repetitions"], 20)
        self.assertFalse(self.spec["smoke_only"])

    def test_seeds_and_baseline_are_paired_across_conditions(self):
        smoke = effective_protocol(self.spec, smoke=True)
        plan = repetition_plan(smoke)[0]
        self.assertNotEqual(plan["training_seed"], plan["validation_seed"])
        self.assertNotEqual(plan["training_seed"], plan["test_seed"])
        conditions = physical_conditions(smoke)
        self.assertEqual(conditions.count((24, 0.1)), 1)
        self.assertIn((24, 0.0), conditions)
        self.assertIn((12, 0.1), conditions)
        # One plan supplies the same trajectory/pair seeds to every condition.
        self.assertIsInstance(plan["training_pair_seed"], int)
        self.assertIsInstance(plan["validation_pair_seed"], int)

    def test_fast_morse_drift_matches_direct_all_pair_evaluation(self):
        rng = np.random.default_rng(421)
        x = rng.normal(size=(17, 10))
        interaction = self.spec["model"]["interaction"]
        kernel, _, _, _ = make_true_kernel(interaction)
        direct = kernel(x[:, None, :] - x[None, :, :]).mean(axis=1)
        np.testing.assert_allclose(
            _morse_drift_fast(x, interaction), direct, rtol=2e-13, atol=2e-13
        )

    def test_fixed_alpha_lasso_is_raw(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([1.0, -1.0, 0.0])
        theta, diagnostics = fit_fixed_alpha(A, b, 1e-8, self.spec["estimator"])
        warm_theta, warm_diagnostics = fit_fixed_alpha(
            A, b, 1e-8, self.spec["estimator"], initial_theta=[4.0, -4.0]
        )
        self.assertTrue(diagnostics["converged"])
        self.assertTrue(warm_diagnostics["converged"])
        np.testing.assert_allclose(theta, [1.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(warm_theta, theta, atol=1e-6)

    def test_heldout_metrics_are_relative_errors(self):
        theta = np.array([1.0, -2.0])
        A = np.eye(2)
        metrics = evaluate_theta(theta, A, theta, A, theta)
        self.assertEqual(metrics["latent_drift_relative_error"], 0.0)
        self.assertEqual(metrics["profile_relative_error"], 0.0)
        self.assertFalse(metrics["numerical_divergence"])

    def test_summary_retains_failures_and_boundaries(self):
        base = {
            "sweep": "sampled_pairs", "sweep_value": 20, "fit_id": "fit",
            "latent_drift_relative_error": 0.2, "profile_relative_error": 0.3,
            "rank_deficient": False, "selected_at_grid_boundary": False,
            "numerical_divergence": False,
        }
        records = [
            {**base, "method": "lasso", "converged": False,
             "selected_at_grid_boundary": True, "numerical_divergence": True,
             "latent_drift_relative_error": None},
            {**base, "method": "ols", "converged": True},
        ]
        result = summarize(records)
        self.assertEqual(result["diagnostics"]["records"], 2)
        self.assertEqual(result["diagnostics"]["nonconverged"], 1)
        self.assertEqual(result["diagnostics"]["boundary_selections"], 1)
        self.assertEqual(result["diagnostics"]["numerical_divergence"], 1)


if __name__ == "__main__":
    unittest.main()
