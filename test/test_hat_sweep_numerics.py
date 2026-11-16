import unittest

import numpy as np

from features import compute_A_timeweak
from kernels import make_kernel_basis
from scripts.hat_sweep_numerics import (
    assemble_full, assemble_sampled, errors_from_statistics, evaluation_statistics,
    evaluate_model, hat_particle_sums, rollout_backend, simulate_hat,
)
from scripts.prepare_1d_hat_experiment import assemble_design, test_family
from scripts.run_1d_hat_lasso import force_and_drift_errors, make_evaluation_kernel
from simulate import make_true_kernel, simulate_with_kernel
from utils import make_pair_sampler


def params(M):
    return dict(basis="hat", M=M, knots=np.linspace(0, 4, M).tolist(), extrapolation="constant")


class TestSweepNumerics(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(8152)
        self.t = np.linspace(0, .2, 11)
        self.X = self.rng.normal(2., .9, (1, len(self.t), 13, 1))
        self.space, self.time = test_family(self.t)
        self.candidates = [(f"hat_M{M}", params(M)) for M in [9, 17, 33, 65]]

    def test_prefix_matches_pairs_at_knots_duplicates_and_extension(self):
        for x in [np.array([0., 0., .0625, .25, .5, 1., 4., 4., 7.]),
                  np.ones(8) * 3., self.rng.uniform(-5, 7, 100)]:
            for _, p in self.candidates:
                kernel, _ = make_kernel_basis(p)
                expected = kernel((x[:, None] - x[None, :])[..., None]).sum(axis=1)[..., 0]
                actual = hat_particle_sums(x, p["knots"])
                np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=1e-11)
                np.testing.assert_allclose(actual.sum(axis=0), 0, atol=1e-10)

    def test_prefix_matches_reference_assembly(self):
        actual = assemble_full(self.X, self.t, self.space, self.time, self.candidates)
        for name, p in self.candidates:
            expected = assemble_design(self.X, self.t, self.space, self.time, p, 31)
            np.testing.assert_allclose(actual[name], expected, atol=1e-15, rtol=1e-11)

    def test_sampled_matches_original_same_draws(self):
        actual = assemble_sampled(self.X, self.t, self.space, self.time, self.candidates, 193, 189)
        for name, p in self.candidates:
            sampler = make_pair_sampler(193, np.random.default_rng(189), exclude_diagonal=True)
            kernel, _ = make_kernel_basis(p)
            expected = compute_A_timeweak(self.X, self.t, self.space, kernel, sampler, self.time)
            expected = expected.reshape(-1, p["M"]) / np.sqrt(200)
            np.testing.assert_allclose(actual[name], expected, atol=1e-16, rtol=1e-12)

    def test_enumerated_sampled_pairs_reproduce_full_normalization(self):
        N = self.X.shape[2]
        i, j = np.where(~np.eye(N, dtype=bool))
        sampled = assemble_sampled(self.X, self.t, self.space, self.time, self.candidates,
                                   N * (N - 1), sampler=lambda n: (i, j))
        full = assemble_full(self.X, self.t, self.space, self.time, self.candidates)
        for name, _ in self.candidates:
            np.testing.assert_allclose(sampled[name], full[name], atol=1e-15, rtol=1e-11)

    def test_evaluation_quadratic_forms_match_direct_evaluation(self):
        truth = {"mode": "qanr", "eps": 1e-8, "r_max": 3.0}
        K, *_ = make_true_kernel(truth)
        stats = evaluation_statistics(self.X, self.t, self.candidates, truth)
        for name, p in self.candidates:
            theta = self.rng.normal(size=p["M"])
            direct = force_and_drift_errors(self.X, self.t, K, make_evaluation_kernel(p, theta))
            actual = errors_from_statistics(theta, stats[name])
            for metric in actual:
                self.assertAlmostEqual(actual[metric], direct[metric], places=11)

    def test_stiff_rollout_uses_identical_reference_arithmetic(self):
        p = params(9)
        theta = np.zeros(9)
        theta[-2] = 100.
        self.assertEqual(rollout_backend(p, theta, self.t)[0], "reference_pairwise")
        self.assertEqual(rollout_backend(p, np.ones(9) * .1, self.t)[0], "prefix_moments")
        noise = self.rng.normal(size=(1, len(self.t) - 1, self.X.shape[2], 1))
        data = dict(X=self.X, X0=self.X[:, 0], t=self.t, driving_noise=noise)
        stats = evaluation_statistics(self.X, self.t, [("hat", p)], {"mode": "qanr", "eps": 1e-8})
        metrics, actual = evaluate_model(data, p, theta, .1, stats["hat"])
        expected = simulate_with_kernel(make_evaluation_kernel(p, theta), data["X0"], self.t, sigma=.1, noise=noise)
        self.assertEqual(metrics["rollout_backend"], "reference_pairwise")
        np.testing.assert_array_equal(actual, expected)

    def test_rollout_matches_original(self):
        X0 = self.X[:, 0]
        noise = self.rng.normal(size=(1, len(self.t) - 1, X0.shape[1], 1))
        for _, p in self.candidates:
            theta = self.rng.normal(size=p["M"])
            expected = simulate_with_kernel(make_evaluation_kernel(p, theta), X0, self.t, sigma=.1, noise=noise)
            actual = simulate_hat(p, theta, X0, self.t, noise, .1)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
