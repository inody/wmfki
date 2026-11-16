import unittest

import numpy as np

from features import compute_increment_regression_system
from kernels import make_kernel_basis
from scripts.run_fig1c_strong_fd_lasso import (
    compressed_system,
    exact_hat_strong_features_1d,
    mse_from_statistics,
)


class Fig1cStrongFdLassoTests(unittest.TestCase):
    def test_sorted_prefix_features_match_exact_pair_assembly(self):
        x = np.array([-1.2, -0.1, 0.4, 1.7, 2.1])
        knots = np.array([0.0, 0.6, 1.4, 2.5])
        params = {
            "basis": "hat", "M": len(knots), "knots": knots.tolist(),
            "extrapolation": "constant",
        }
        basis, _ = make_kernel_basis(params)
        X = np.zeros((1, 2, len(x), 1))
        X[0, 0, :, 0] = x
        X[0, 1, :, 0] = x
        exact_A, _ = compute_increment_regression_system(X, np.array([0.0, 1.0]), basis)
        fast_A = exact_hat_strong_features_1d(x, knots)
        np.testing.assert_allclose(fast_A, exact_A, rtol=1e-12, atol=1e-12)

    def test_compressed_system_preserves_quadratic_up_to_constant(self):
        rng = np.random.default_rng(12)
        A = rng.normal(size=(50, 5))
        b = rng.normal(size=50)
        statistics = {
            "gram": A.T @ A / len(A),
            "score": A.T @ b / len(A),
            "response_square": float(b @ b / len(A)),
            "rows": len(A),
        }
        compressed_A, compressed_b = compressed_system(statistics)
        differences = []
        for _ in range(4):
            theta = rng.normal(size=5)
            direct = mse_from_statistics(theta, statistics)
            compressed = float(np.mean((compressed_A @ theta - compressed_b) ** 2))
            differences.append(direct - compressed)
        np.testing.assert_allclose(differences, differences[0], rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
