import tempfile
import unittest
from pathlib import Path

import numpy as np

from kernels import make_kernel_basis
from predator_prey import (
    channel_kernels,
    make_predator_prey_bases,
    predator_prey_velocity,
)
from scripts.run_cross_example_nominal_comparison import (
    assemble_localized_strong,
    bootstrap_rmse,
    localized_strong_features,
    predator_prey_strong_features,
    relative_error,
    summarize,
    write_tables,
)
from scripts.run_morse10d_sweeps import _radial_drift_fast


class CrossExampleNominalComparisonTests(unittest.TestCase):
    def test_predator_prey_features_reproduce_model_velocity(self):
        rng = np.random.default_rng(12)
        prey = rng.normal(size=(7, 2))
        predator = rng.normal(size=2)
        prey_basis, predator_basis, _ = make_predator_prey_bases(
            0.05, [0.25, 0.5, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        theta_xx = np.array([1.0, -1.0, 0.2, 0.0, 0.0, 0.0])
        theta_xz = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        theta_zx = np.array([0.0, 0.0, 0.4, 0.0, 0.0])
        feature_x, feature_z = predator_prey_strong_features(
            prey, predator, prey_basis, predator_basis
        )
        predicted_x = np.einsum(
            "imd,m->id", feature_x,
            np.concatenate([theta_xx, theta_xz]),
        )
        predicted_z = np.einsum("md,m->d", feature_z, theta_zx)
        expected_x, expected_z = predator_prey_velocity(
            prey, predator,
            channel_kernels(theta_xx, theta_xz, theta_zx, prey_basis, predator_basis),
        )
        np.testing.assert_allclose(predicted_x, expected_x, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(predicted_z, expected_z, rtol=1e-12, atol=1e-12)

    def test_localized_features_reproduce_radial_drift(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(11, 3))
        dictionary = {
            "basis": "gaussian_shell", "M": 4,
            "centers": [1.2, 1.8, 2.4, 3.0], "h": 0.2,
        }
        interaction = {
            "mode": "gaussian_shell", "centers": dictionary["centers"],
            "h": dictionary["h"], "coefficients": [0.0, 1.5, -1.0, 0.0],
        }
        basis, _ = make_kernel_basis(dictionary)
        targets = np.array([0, 3, 7])
        feature = localized_strong_features(x, targets, basis)
        predicted = np.einsum("imd,m->id", feature, interaction["coefficients"])
        expected = _radial_drift_fast(x, interaction)[targets]
        np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-12)

    def test_localized_assembly_recovers_deterministic_euler_coefficients(self):
        dictionary = {
            "basis": "gaussian_shell", "M": 4,
            "centers": [0.5, 1.0, 1.5, 2.0], "h": 0.3,
        }
        interaction = {
            "mode": "gaussian_shell", "centers": dictionary["centers"],
            "h": dictionary["h"], "coefficients": [0.2, -0.4, 0.7, 0.0],
        }
        basis, _ = make_kernel_basis(dictionary)
        rng = np.random.default_rng(5)
        x0 = rng.normal(size=(8, 2))
        dt = 1e-3
        x1 = x0 + dt * _radial_drift_fast(x0, interaction)
        X = np.stack([x0, x1], axis=0)[None, ...]
        A, y = assemble_localized_strong(X, np.array([0.0, dt]), np.arange(8), basis)
        theta, *_ = np.linalg.lstsq(A, y, rcond=None)
        np.testing.assert_allclose(theta, interaction["coefficients"], rtol=1e-9, atol=1e-9)

    def test_summary_and_tables_retain_all_methods(self):
        records = []
        for example in ("one_dimensional", "predator_prey", "localized_10d"):
            for repetition in range(3):
                for index, method in enumerate((
                    "weak_lasso", "weak_ols", "strong_forward_difference_ols"
                )):
                    records.append({
                        "example": example, "method": method,
                        "repetition": repetition,
                        "coefficient_error_relative": 0.1 * (index + 1),
                        "status": "finite",
                    })
        metric = {"bootstrap_draws": 100, "bootstrap_seed": 3}
        summary = summarize(records, metric)
        self.assertEqual(len(summary["groups"]), 9)
        self.assertTrue(all(value == "weak_lasso"
                            for value in summary["rmse_winner"].values()))
        with tempfile.TemporaryDirectory() as directory:
            write_tables(Path(directory), summary)
            text = (Path(directory) / "table_rmse_draft.md").read_text()
            self.assertIn("Strong FD-OLS", text)
            self.assertIn("1D overcomplete", text)

    def test_bootstrap_rmse_is_deterministic(self):
        first = bootstrap_rmse([1.0, 2.0, 3.0], 200, 11)
        second = bootstrap_rmse([1.0, 2.0, 3.0], 200, 11)
        self.assertEqual(first, second)
        self.assertAlmostEqual(relative_error([1.0, 2.0], [1.0, 1.0]), 1 / np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
