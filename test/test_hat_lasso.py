import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from kernels import make_kernel_basis, make_kernel_evaluator
from simulate import simulate_with_kernel
from scripts.compare_1d_hat_particles import check_protocols
from scripts.prepare_1d_hat_experiment import DEFAULT_CONFIG, basis_params, prepare
from scripts.run_1d_hat_lasso import (
    force_and_drift_errors, load_split, make_evaluation_kernel, run, select_lasso,
)


class HatLassoTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(DEFAULT_CONFIG.read_text())
        self.settings = {**self.spec["estimator"], "alpha_count": 8, "max_iter": 2000}

    def test_particle_comparison_rejects_other_protocol_changes(self):
        large = json.loads(DEFAULT_CONFIG.with_name("one_dimensional_hat_N1000.json").read_text())
        check_protocols(self.spec, large)
        self.assertEqual(large["simulation"]["N"], 1000)
        large["estimator"]["tol"] = 1e-6
        with self.assertRaises(ValueError):
            check_protocols(self.spec, large)
        with self.assertRaises(ValueError):
            check_protocols(self.spec, self.spec)

    def test_alpha_grid_scaling_and_validation_selection(self):
        A = np.eye(4)
        b = np.array([1.0, 0.0, 0.0, 0.0])
        # Avoid ties caused by the warm-start tolerance when comparing with the analytical solution.
        settings = {**self.settings, "tol": 1e-16}
        result, coefficients = select_lasso(A, b, A, b, settings)
        self.assertAlmostEqual(result["alpha_max"], 0.25)
        self.assertEqual(result["selected"]["index"], 7)
        self.assertEqual(result["selected"]["nonzero_reported"], 1)
        self.assertAlmostEqual(result["selected"]["lambda_summed_loss"], 4 * result["selected"]["alpha_sk"])
        np.testing.assert_allclose(result["theta"], coefficients[7])
        # Changing only the validation response while keeping the training data fixed selects stronger regularization.
        zero_validation, _ = select_lasso(A, b, A, np.zeros(4), self.settings)
        self.assertEqual(zero_validation["selected"]["index"], 0)
        np.testing.assert_array_equal(zero_validation["theta"], np.zeros(4))

    def test_unconverged_candidate_is_not_selected_even_with_low_validation_error(self):
        class FakeLasso:
            def __init__(self, **kwargs):
                self.count = 0
                self.dual_gap_ = 0.0
                self.n_iter_ = 1

            def fit(self, A, b):
                self.coef_ = np.array([1.0 if self.count == 0 else 0.5])
                if self.count == 0:
                    warnings.warn("not converged", ConvergenceWarning)
                self.count += 1

        with patch("scripts.run_1d_hat_lasso.Lasso", FakeLasso):
            result, _ = select_lasso(np.ones((2, 1)), np.ones(2), np.ones((2, 1)), np.ones(2), self.settings)
        self.assertEqual(result["path"][0]["validation_mse"], 0.0)
        self.assertFalse(result["path"][0]["converged"])
        self.assertEqual(result["selected"]["index"], 1)

    def test_force_and_drift_metrics_have_correct_scaling(self):
        X = np.array([[[[0.0], [1.0]], [[0.0], [1.0]]]])
        t = np.array([0.0, 1.0])
        truth = lambda z: 2.0 * z
        exact = force_and_drift_errors(X, t, truth, truth)
        self.assertEqual(exact["force_error_weighted_rel"], 0)
        self.assertEqual(exact["drift_error_rel"], 0)
        wrong = force_and_drift_errors(X, t, truth, lambda z: 3.0 * z)
        self.assertAlmostEqual(wrong["force_error_weighted_rel"], 0.5)
        self.assertAlmostEqual(wrong["drift_error_rel"], 0.5)

    def test_memory_bounded_evaluation_matches_original_basis_sum(self):
        rng = np.random.default_rng(173)
        candidates = [basis_params(self.spec, m) for m in self.spec["basis"]["sizes"]]
        candidates.append(self.spec["baseline"])
        for params in candidates:
            theta = rng.normal(size=params["M"])
            basis, _ = make_kernel_basis(params)
            reference = make_kernel_evaluator(theta, basis)
            bounded = make_evaluation_kernel(params, theta, chunk_size=7)
            for z in (np.array([0., 1e-10, 4., 5., -5.]),
                      rng.normal(size=(5, 7, 2)), np.empty((0, 1))):
                np.testing.assert_allclose(bounded(z), reference(z), atol=1e-14, rtol=1e-13)
            X0 = rng.normal(size=(1, 8, 1))
            noise = rng.normal(size=(1, 20, 8, 1))
            t = np.linspace(0, 0.1, 21)
            expected = simulate_with_kernel(reference, X0, t, sigma=0.1, noise=noise)
            actual = simulate_with_kernel(bounded, X0, t, sigma=0.1, noise=noise)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            for key, value in force_and_drift_errors(expected, t, reference, bounded).items():
                if key.endswith("_rel"):
                    self.assertLess(value, 1e-12)

    def test_smoke_fit_freezes_all_choices_before_loading_test_and_refuses_overwrite(self):
        spec = self.spec
        spec["estimator"] = {**self.settings, "alpha_count": 4}
        with tempfile.TemporaryDirectory() as folder:
            assembly, output = Path(folder) / "assembly", Path(folder) / "fit"
            manifest = prepare(spec, assembly, smoke=True)
            with self.assertRaises(ValueError):
                run(assembly, output)

            def guarded_load(directory, metadata, split, candidate):
                if split == "test":
                    frozen = json.loads((output / "frozen_selections.json").read_text())
                    self.assertEqual(set(frozen), set(metadata["splits"]["train"][0]["designs"]))
                return load_split(directory, metadata, split, candidate)

            with patch("scripts.run_1d_hat_lasso.load_split", side_effect=guarded_load):
                report = run(assembly, output, allow_smoke=True)
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["smoke_only"])
            self.assertFalse(report["ols_performed"])
            self.assertTrue((output / "pilot_summary.png").exists())
            self.assertTrue((output / "heldout_trajectories.png").exists())
            with self.assertRaises(FileExistsError):
                run(assembly, output, allow_smoke=True)
            path = assembly / "train" / str(manifest["splits"]["train"][0]["seed"]) / "hat_M4.npy"
            A = np.load(path)
            A[0, 0] += 1
            np.save(path, A)
            with self.assertRaises(ValueError):
                load_split(assembly, manifest, "train", "hat_M4")


if __name__ == "__main__":
    unittest.main()
