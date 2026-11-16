import unittest

import numpy as np

from scripts.run_qanr_matched_lasso_sweeps import (
    DEFAULT_CONFIG,
    apply_dictionary,
    effective_protocol,
    fit_fixed_alpha,
    read_json,
    summarize,
    validate_protocol,
)


class TestQANRMatchedLassoSweeps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read_json(DEFAULT_CONFIG)

    def test_protocol_is_fixed(self):
        validate_protocol(self.spec)
        self.assertEqual(self.spec["estimator"]["alpha_count"], 60)
        self.assertFalse(self.spec["estimator"]["truth_used_for_selection"])
        self.assertFalse(self.spec["estimator"]["post_lasso_refit"])

    def test_uniform_r100_protocol_is_fixed(self):
        spec = read_json("configs/experiments/qanr_matched_lasso_sweeps_R100.json")
        validate_protocol(spec)
        self.assertEqual(spec["protocol_variant"], "uniform_trajectory_R100")
        self.assertEqual(spec["observation_noise"]["repetitions"], 100)
        self.assertEqual(spec["process_noise"]["repetitions"], 100)
        self.assertEqual(spec["particle_number"]["repetitions"], 100)

    def test_redundant_six_atom_protocol_is_fixed(self):
        spec = read_json("configs/experiments/qanr_redundant6_lasso_sweeps_R100.json")
        validate_protocol(spec)
        config = apply_dictionary({"kernel_basis": {"powers": [1, 2]}}, spec)
        self.assertEqual(config["kernel_basis"]["basis"], "qanr_exp")
        self.assertEqual(config["kernel_basis"]["M"], 6)
        self.assertEqual(config["kernel_basis"]["ell_grid"], [0.25, 0.5, 1.0, 2.0])
        self.assertEqual(spec["true_theta"], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0])

    def test_smoke_is_not_scientific_protocol(self):
        smoke = effective_protocol(self.spec, smoke=True)
        self.assertTrue(smoke["smoke_only"])
        self.assertEqual(smoke["observation_noise"]["repetitions"], 1)
        self.assertEqual(smoke["pair_count"]["repetitions"], 2)
        self.assertEqual(smoke["time_step"]["stacked_trajectories"], 1)

    def test_fixed_alpha_fit_is_raw(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([1.0, -1.0, 0.0])
        theta, diagnostics = fit_fixed_alpha(A, b, 1e-8, self.spec["estimator"])
        self.assertTrue(diagnostics["converged"])
        np.testing.assert_allclose(theta, [1.0, -1.0], atol=1e-6)

    def test_summary_keeps_pair_failures(self):
        selected = {
            "selected_at_grid_boundary": False,
            "unconverged_candidates": 0,
        }
        observation = [{
            "sigma_obs": 0.0, "coefficient_error": 0.1,
            "ols_coefficient_error": 0.2, "euler_coefficient_error": 0.3, **selected,
        }]
        stochastic = [
            {"kind": "sigma", "value": 0.1, "coefficient_error": 0.1,
             "ols_coefficient_error": 0.2, **selected},
            {"kind": "particles", "value": 100.0, "coefficient_error": 0.1,
             "ols_coefficient_error": 0.2, **selected},
        ]
        dt_rows = [{
            "dt": 0.01, "coefficient_error": 0.1,
            "ols_coefficient_error": 0.2, **selected,
        }]
        pair_rows = [
            {"P": 100, "theta_to_full_sq": 0.2,
             "ols_theta_to_full_sq_recomputed": 0.3, "converged": True},
            {"P": 100, "theta_to_full_sq": 0.4,
             "ols_theta_to_full_sq_recomputed": 0.5, "converged": False},
        ]
        pair_selection = {
            "selected_at_grid_boundary": False,
            "path": [{"converged": True}],
        }
        summary = summarize(observation, stochastic, dt_rows, pair_rows, pair_selection)
        self.assertEqual(summary["diagnostics"]["pair_nonconverged"], 1)
        self.assertEqual(summary["pair_count"][0]["lasso_theta_to_full_sq"]["count"], 2)
        self.assertEqual(summary["pair_count"][0]["ols_theta_to_full_sq"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
