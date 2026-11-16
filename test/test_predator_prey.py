import unittest

import numpy as np

from predator_prey import (
    assemble_predator_prey_weak_systems,
    channel_kernels,
    make_predator_prey_bases,
    simulate_predator_prey,
)
from scripts.run_predator_prey_application import drift_error
from tests import make_test_functions
from time_tests import make_time_test_functions


class PredatorPreyTests(unittest.TestCase):
    def setUp(self):
        self.prey_basis, self.predator_basis, self.meta = make_predator_prey_bases(
            0.1, [0.25, 0.5, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0]
        )
        self.theta_xx = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        self.theta_xz = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.theta_zx = np.array([0.0, 0.0, 0.4, 0.0])
        self.kernels = channel_kernels(
            self.theta_xx,
            self.theta_xz,
            self.theta_zx,
            self.prey_basis,
            self.predator_basis,
        )

    def test_channel_basis_reproduces_regularized_model(self):
        displacement = np.array([[0.3, -0.4], [1.0, 0.0]])
        radius_sq = np.sum(displacement * displacement, axis=1)
        xx, xz, zx = self.kernels
        np.testing.assert_allclose(
            xx(displacement),
            displacement * (1.0 / (radius_sq + 0.01) - 1.0)[:, None],
        )
        np.testing.assert_allclose(
            xz(displacement),
            0.2 * displacement / (radius_sq + 0.01)[:, None],
        )
        np.testing.assert_allclose(
            zx(displacement),
            0.4 * displacement / (radius_sq + 0.01)[:, None] ** 1.5,
        )

    def test_simulation_shapes_and_translation_equivariance(self):
        rng = np.random.default_rng(12)
        initial_x = rng.uniform(-0.4, 0.4, size=(1, 12, 2))
        initial_z = rng.uniform(-0.2, 0.2, size=(1, 1, 2))
        x, z, time, diagnostics = simulate_predator_prey(
            particle_count=12,
            steps=8,
            dt=0.01,
            repetitions=1,
            kernels=self.kernels,
            seed=20,
            initial_prey=initial_x,
            initial_predator=initial_z,
        )
        shift = np.array([1.2, -0.7])
        shifted_x, shifted_z, _, shifted_diagnostics = simulate_predator_prey(
            particle_count=12,
            steps=8,
            dt=0.01,
            repetitions=1,
            kernels=self.kernels,
            seed=21,
            initial_prey=initial_x + shift,
            initial_predator=initial_z + shift,
        )
        self.assertEqual(x.shape, (1, 9, 12, 2))
        self.assertEqual(z.shape, (1, 9, 1, 2))
        self.assertEqual(time.shape, (9,))
        self.assertEqual(diagnostics["finite"], [True])
        self.assertEqual(shifted_diagnostics["finite"], [True])
        np.testing.assert_allclose(shifted_x, x + shift, atol=2e-12)
        np.testing.assert_allclose(shifted_z, z + shift, atol=2e-12)

    def test_species_balanced_drift_error_uses_full_sources_and_repetitions(self):
        x, z, _, diagnostics = simulate_predator_prey(
            particle_count=8,
            steps=4,
            dt=0.01,
            repetitions=2,
            kernels=self.kernels,
            seed=27,
        )
        self.assertTrue(all(diagnostics["finite"]))
        zero_kernels = channel_kernels(
            np.zeros_like(self.theta_xx),
            np.zeros_like(self.theta_xz),
            np.zeros_like(self.theta_zx),
            self.prey_basis,
            self.predator_basis,
        )
        result = drift_error(x, z, self.kernels, zero_kernels, stride=1)
        self.assertAlmostEqual(result["prey_relative"], 1.0)
        self.assertAlmostEqual(result["predator_relative"], 1.0)
        self.assertAlmostEqual(result["species_balanced_relative"], 1.0)

    def test_true_coefficients_reduce_weak_residual(self):
        x, z, time, diagnostics = simulate_predator_prey(
            particle_count=24,
            steps=120,
            dt=0.005,
            repetitions=2,
            kernels=self.kernels,
            seed=31,
        )
        self.assertTrue(all(diagnostics["finite"]))
        coordinates = np.linspace(-1.2, 1.2, 3)
        centers = np.array(np.meshgrid(coordinates, coordinates)).reshape(2, -1).T
        tests_x = make_test_functions(
            {"centers": centers, "s": 0.55, "type": "gaussian"}
        )
        tests_z = make_test_functions(
            {"centers": centers, "s": 0.75, "type": "gaussian"}
        )
        temporal = make_time_test_functions(
            {"time_tests": {"K": 4, "type": "bump", "s_factor": 1.8}},
            time,
        )
        systems = assemble_predator_prey_weak_systems(
            x,
            z,
            time,
            prey_tests=tests_x,
            predator_tests=tests_z,
            prey_basis=self.prey_basis,
            predator_basis=self.predator_basis,
            time_tests=temporal,
        )
        prey_truth = np.concatenate([self.theta_xx, self.theta_xz])
        predator_truth = self.theta_zx
        for name, truth in (("prey", prey_truth), ("predator", predator_truth)):
            A = systems[name]["A_flat"]
            b = systems[name]["b_flat"]
            truth_residual = np.linalg.norm(A @ truth - b)
            zero_residual = np.linalg.norm(b)
            self.assertLess(truth_residual, 0.35 * zero_residual)


if __name__ == "__main__":
    unittest.main()
