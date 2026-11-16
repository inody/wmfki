import unittest

import numpy as np

from features import (
    compute_A_full_timeweak,
    compute_A_timeweak,
    compute_increment_regression_system,
    make_trapz_weights,
)
from kernels import make_kernel_basis, make_kernel_evaluator, make_radial_basis
from pipeline import fit_kernel_from_particles
from scripts.run_sindy_comparison import (
    TRUE_THETA,
    minimum_pair_distance,
    reference_trajectory,
    sampled_drift_error,
    trajectory_errors,
)
from tests import make_test_functions
from time_tests import make_time_test_functions
from utils import make_pair_sampler


class WeakFeatureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.X = rng.normal(size=(1, 6, 5, 1))
        self.t = np.linspace(0.0, 1.0, self.X.shape[1])
        self.space = make_test_functions(
            {"type": "gaussian", "centers": [-0.5, 0.5], "s": 0.8}
        )
        self.time = make_time_test_functions(
            {"time_tests": {"type": "bump", "K": 2, "support": 0.6}},
            self.t,
        )
        self.basis, _ = make_kernel_basis(
            {"basis": "grad_abs_power", "powers": [1, 2], "eps": 0.1}
        )
        self.weights = make_trapz_weights(self.t)

    def test_all_pairs_sampler_matches_chunked_full_design(self):
        N = self.X.shape[2]
        sampler = make_pair_sampler(
            N * (N - 1),
            rng=np.random.default_rng(3),
            with_replacement=False,
            exclude_diagonal=True,
        )
        sampled = compute_A_timeweak(
            self.X,
            self.t,
            self.space,
            self.basis,
            sampler,
            self.time,
            self.weights,
        )
        full = compute_A_full_timeweak(
            self.X,
            self.t,
            self.space,
            self.basis,
            self.time,
            self.weights,
            pair_chunk_size=7,
        )
        np.testing.assert_allclose(sampled, full, rtol=1e-12, atol=1e-12)

    def test_fixed_pair_sampler_reuses_indices(self):
        fixed = make_pair_sampler(
            20,
            rng=np.random.default_rng(5),
            with_replacement=True,
            exclude_diagonal=True,
            resample_each_call=False,
        )
        first = fixed(8)
        second = fixed(8)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_sampled_design_is_unbiased(self):
        full = compute_A_full_timeweak(
            self.X,
            self.t,
            self.space,
            self.basis,
            self.time,
            self.weights,
        )
        estimates = []
        for seed in range(400):
            sampler = make_pair_sampler(
                20,
                rng=np.random.default_rng(seed),
                with_replacement=True,
                exclude_diagonal=True,
            )
            estimates.append(
                compute_A_timeweak(
                    self.X,
                    self.t,
                    self.space,
                    self.basis,
                    sampler,
                    self.time,
                    self.weights,
                )
            )
        mean_design = np.mean(estimates, axis=0)
        relative_error = np.linalg.norm(mean_design - full) / np.linalg.norm(full)
        self.assertLess(relative_error, 0.06)

    def test_compact_bump_derivatives(self):
        tests = make_test_functions(
            {"type": "bump", "centers": [[0.0, 0.0]], "s": 2.0}
        )
        x = np.array([[0.4, -0.3]])
        h = 1e-5
        grad_fd = []
        lap_fd = 0.0
        phi0 = tests["phi"](x)[0, 0]
        for coordinate in range(2):
            shift = np.zeros_like(x)
            shift[0, coordinate] = h
            plus = tests["phi"](x + shift)[0, 0]
            minus = tests["phi"](x - shift)[0, 0]
            grad_fd.append((plus - minus) / (2.0 * h))
            lap_fd += (plus - 2.0 * phi0 + minus) / (h * h)
        np.testing.assert_allclose(
            tests["grad_phi"](x)[0, 0], grad_fd, rtol=2e-5, atol=2e-6
        )
        self.assertAlmostEqual(tests["lap_phi"](x)[0, 0], lap_fd, places=4)
        outside = np.array([[2.1, 0.0]])
        self.assertEqual(tests["phi"](outside)[0, 0], 0.0)
        np.testing.assert_array_equal(tests["grad_phi"](outside)[0, 0], 0.0)

    def test_pipeline_returns_normalized_system(self):
        config = {
            "simulation": {"d": 1, "sigma": 0.0},
            "space_tests": {
                "type": "bump",
                "L": 2,
                "centers": [-0.5, 0.5],
                "s": 1.5,
                "layout": "grid",
            },
            "time_tests": {"type": "bump", "K": 2, "support": 0.6},
            "kernel_basis": {
                "basis": "grad_abs_power",
                "powers": [1, 2],
                "eps": 0.1,
            },
            "pair_sampling": {
                "total_pairs": 20,
                "with_replacement": False,
                "exclude_diagonal": True,
                "seed": 1,
            },
            "regression": {
                "method": "ridge",
                "alpha": 0.0,
                "standardize": False,
            },
        }
        result = fit_kernel_from_particles(
            config, self.X, self.t, return_system=True, pair_mode="full"
        )
        self.assertAlmostEqual(result["system"]["row_scale"], 0.5)
        self.assertEqual(result["system"]["A"].shape, (1, 2, 2, 2))
        self.assertEqual(result["system"]["b"].shape, (1, 2, 2))

    def test_qanr_exponential_overcomplete_basis(self):
        params = {
            "basis": "qanr_exp",
            "M": 4,
            "ell_grid": [0.5, 2.0],
            "eps": 0.1,
        }
        radial, meta = make_radial_basis(params)
        r = np.array([0.0, 1.0])
        expected = np.column_stack(
            [
                1.0 / (r + 0.1),
                np.ones_like(r),
                np.exp(-r / 0.5),
                np.exp(-r / 2.0),
            ]
        )
        np.testing.assert_allclose(radial(r), expected)
        self.assertEqual(meta["labels"], [
            "1/(r+eps)", "1", "exp(-r/0.5)", "exp(-r/2)"
        ])
        basis, _ = make_kernel_basis(params)
        self.assertEqual(basis(np.array([[1.0]])).shape, (1, 4, 1))
        with self.assertRaisesRegex(ValueError, "M = 2"):
            make_radial_basis({**params, "M": 5})

    def test_increment_system_uses_particle_drift_normalization(self):
        X = np.array(
            [[[[0.0], [1.0], [3.0]], [[-4.0 / 3.0], [2.0 / 3.0], [14.0 / 3.0]]]]
        )
        t = np.array([0.0, 1.0])

        def basis(z):
            return z[..., None, :]

        A, y = compute_increment_regression_system(X, t, basis, pair_chunk_size=2)
        expected = np.array([[-4.0 / 3.0], [-1.0 / 3.0], [5.0 / 3.0]])
        np.testing.assert_allclose(A, expected)
        np.testing.assert_allclose(y, expected[:, 0])


class SindyComparisonTests(unittest.TestCase):
    """Checks for the measurement-noise comparison runner."""

    SETTING = {"particles": 8, "horizon": 0.05, "step": 0.01, "sigma": 0.1}

    def setUp(self):
        self.X, self.t, self.X0, self.driving_noise, self.K_true = reference_trajectory(
            97, **self.SETTING
        )
        self.basis, _ = make_kernel_basis(
            {"basis": "grad_abs_power", "powers": [1, 2], "eps": 1e-8}
        )

    def test_reference_trajectory_is_reproducible_from_the_seed(self):
        X, t, X0, driving_noise, _ = reference_trajectory(97, **self.SETTING)
        np.testing.assert_array_equal(X, self.X)
        np.testing.assert_array_equal(t, self.t)
        np.testing.assert_array_equal(X0, self.X0)
        np.testing.assert_array_equal(driving_noise, self.driving_noise)

    def test_matched_coefficients_reproduce_the_reference_trajectory(self):
        """The trajectory metric must vanish when the estimate is exact."""
        K_hat = make_kernel_evaluator(TRUE_THETA, self.basis)
        errors = trajectory_errors(
            K_hat,
            self.X,
            self.X0,
            self.t,
            self.driving_noise,
            sigma=self.SETTING["sigma"],
        )
        self.assertFalse(errors["diverged"])
        self.assertLess(errors["trajectory_error_rel"], 1e-12)
        self.assertLess(errors["trajectory_error_final"], 1e-12)
        self.assertAlmostEqual(
            sampled_drift_error(self.K_true, K_hat, self.X, 5, samples=20), 0.0, places=12
        )

    def test_perturbed_coefficients_give_a_positive_trajectory_error(self):
        K_hat = make_kernel_evaluator(TRUE_THETA * 1.2, self.basis)
        errors = trajectory_errors(
            K_hat,
            self.X,
            self.X0,
            self.t,
            self.driving_noise,
            sigma=self.SETTING["sigma"],
        )
        self.assertFalse(errors["diverged"])
        self.assertGreater(errors["trajectory_error_rel"], 0.0)

    def test_divergent_kernel_is_flagged_rather_than_reported(self):
        def exploding_kernel(z):
            return 1e6 * np.asarray(z)

        errors = trajectory_errors(
            exploding_kernel,
            self.X,
            self.X0,
            self.t,
            self.driving_noise,
            sigma=self.SETTING["sigma"],
        )
        self.assertTrue(errors["diverged"])
        self.assertTrue(np.isnan(errors["trajectory_error_rel"]))

    def test_minimum_pair_distance_excludes_the_diagonal(self):
        X = np.array([[[[0.0], [0.25], [1.0]]]])
        self.assertAlmostEqual(minimum_pair_distance(X, stride=1), 0.25)


if __name__ == "__main__":
    unittest.main()
