import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from features import make_trapz_weights
from kernels import make_kernel_basis, make_radial_basis
from pipeline import fit_kernel_from_particles
from scripts.prepare_1d_hat_experiment import (
    DEFAULT_CONFIG, array_hash, assemble_design, basis_params,
    distance_diagnostics, matrix_diagnostics, prepare, test_family, validate_spec,
)


class HatBasisTests(unittest.TestCase):
    def setUp(self):
        self.params = {
            "basis": "hat", "M": 4, "knots": [0.0, 0.4, 1.0, 2.0],
            "extrapolation": "constant",
        }

    def test_nodal_values_partition_and_extension(self):
        phi, meta = make_radial_basis(self.params)
        np.testing.assert_array_equal(phi(self.params["knots"]), np.eye(4))
        r = np.linspace(0, 3, 101)
        values = phi(r)
        self.assertTrue(np.all(values >= 0))
        np.testing.assert_allclose(values.sum(axis=-1), 1)
        np.testing.assert_array_equal(phi(3.0), [0, 0, 0, 1])
        self.assertEqual(meta["extrapolation"], "constant")

    def test_linear_interpolation_and_shapes(self):
        phi, _ = make_radial_basis(self.params)
        r = np.array([[0.1, 0.6], [1.2, 2.0]])
        coefficients = 3 * np.array(self.params["knots"]) + 2
        np.testing.assert_allclose(phi(r) @ coefficients, 3 * r + 2)
        self.assertEqual(phi(0.2).shape, (4,))
        self.assertEqual(phi(np.empty((0, 2))).shape, (0, 2, 4))

    def test_odd_vector_kernel_continuous_at_zero_and_outer_boundary(self):
        basis, _ = make_kernel_basis(self.params)
        z = np.array([[0.0, 0.0], [0.3, 0.4], [2.0, -1.0]])
        np.testing.assert_allclose(basis(-z), -basis(z))
        np.testing.assert_array_equal(basis(z[:1]), np.zeros((1, 4, 2)))
        epsilon = 1e-7
        left = basis(np.array([[2.0 - epsilon]]))
        right = basis(np.array([[2.0 + epsilon]]))
        self.assertLess(np.max(np.abs(left - right)), 5 * epsilon)

    def test_reject_invalid_definitions_and_radii(self):
        for knots in ([0, 1, 1, 2], [1, 2, 3, 4], [0, 2, 1, 3],
                      [0, 1, np.inf, 3], [0, 1, np.nan, 3], [0]):
            with self.subTest(knots=knots), self.assertRaises(ValueError):
                make_radial_basis({**self.params, "knots": knots})
        for replacement in ({"M": 3}, {"extrapolation": "zero"}):
            with self.assertRaises(ValueError):
                make_radial_basis({**self.params, **replacement})
        phi, _ = make_radial_basis(self.params)
        for value in (-0.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                phi(value)

    def test_old_dictionary_values_unchanged(self):
        phi, _ = make_radial_basis({
            "basis": "qanr_exp", "M": 6, "ell_grid": [0.25, 0.5, 1, 2], "eps": 1e-8,
        })
        r = np.array([0.02, 0.5, 2.0])
        expected = np.column_stack((1 / (r + 1e-8), np.ones(3),
                                    np.exp(-r[:, None] / np.array([0.25, 0.5, 1, 2]))))
        np.testing.assert_array_equal(phi(r), expected)


class HatPreparationTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(DEFAULT_CONFIG.read_text())

    def test_protocol_and_seed_validation(self):
        validate_spec(self.spec)
        for delta in (0, 1, -1):
            modified = json.loads(DEFAULT_CONFIG.read_text())
            modified["seeds"]["test"] = [modified["seeds"]["train"][0] + delta]
            with self.assertRaises(ValueError):
                validate_spec(modified)
        self.spec["true_kernel"]["eps"] = 1e-3
        with self.assertRaises(ValueError):
            validate_spec(self.spec)

    def test_particle_comparison_changes_only_approved_particle_count(self):
        large = json.loads(DEFAULT_CONFIG.with_name("one_dimensional_hat_N1000.json").read_text())
        validate_spec(large)
        self.assertEqual(large["simulation"]["N"], 1000)
        large["simulation"]["N"] = 100
        self.assertEqual(large, self.spec)
        for invalid in (1, 101, 1000.0, True):
            large["simulation"]["N"] = invalid
            with self.assertRaises(ValueError):
                validate_spec(large)
        large["simulation"].update(N=1000, sigma=0.0)
        with self.assertRaises(ValueError):
            validate_spec(large)

    def test_design_preserves_pair_and_row_normalization(self):
        rng = np.random.default_rng(1)
        Y = rng.normal(size=(1, 21, 4, 1))
        t = np.linspace(0, 0.1, 21)
        space, time = test_family(t)
        params = basis_params(self.spec, 4)
        A = assemble_design(Y, t, space, time, params, 3)
        basis, _ = make_kernel_basis(params)
        expected = np.zeros((25, 8, 4))
        for n, weight in enumerate(make_trapz_weights(t)):
            x = Y[0, n]
            gradients = space["grad_phi"](x)
            for i in range(4):
                for j in range(4):
                    if i == j:
                        continue
                    features = basis((x[i] - x[j])[None])[0, :, 0]
                    expected += (weight / 16 / np.sqrt(200)
                                 * gradients[i, :, 0][:, None, None]
                                 * time["psi"][:, n][None, :, None]
                                 * features[None, None, :])
        np.testing.assert_allclose(A, expected.reshape(200, 4), rtol=1e-12, atol=1e-14)

    def test_pipeline_forwards_hat_parameters_to_weighted_diagnostics(self):
        # API check with synthetic arrays; this test does not select an estimator for the experiment.
        rng = np.random.default_rng(2)
        Y = rng.normal(size=(1, 21, 4, 1))
        t = np.linspace(0.0, 0.1, 21)
        config = {
            "simulation": {"d": 1, "sigma": 0.1},
            "space_tests": {"L": 3, "centers": [-1.0, 0.0, 1.0], "s": 0.5},
            "time_tests": {"type": "bump", "K": 2, "support": 0.06},
            "kernel_basis": basis_params(self.spec, 4),
            "regression": {"method": "ridge", "alpha": 1e-6, "standardize": False},
            "diagnostics": {
                "method": "l2w", "total_pairs": 30, "bins": 8,
                "true_kernel": {"phi_true_fn": lambda r: np.ones_like(r)},
            },
        }
        result = fit_kernel_from_particles(config, Y, t, pair_mode="full")
        self.assertTrue(np.isfinite(result["diagnostics"]["kernel_L2w"]))
        self.assertEqual(result["theta_hat"].shape, (4,))

    def test_rank_diagnostics_do_not_change_matrix(self):
        for A, expected_rank in ((np.eye(4), 4), (np.zeros((5, 3)), 0), (np.ones((5, 3)), 1)):
            before = A.copy()
            info = matrix_diagnostics(A)
            np.testing.assert_array_equal(A, before)
            self.assertEqual(info["raw"]["rank"], expected_rank)
            if expected_rank < A.shape[1]:
                self.assertIsNone(info["raw"]["condition_number"])
            json.dumps(info, allow_nan=False)

    def test_distance_measure_and_extrapolation(self):
        Y = np.array([[[[0.0], [1.0], [5.0]], [[0.0], [1.0], [5.0]]]])
        info = distance_diagnostics(Y, np.array([0.0, 1.0]), 4.0, [4, 8])
        self.assertAlmostEqual(sum(info["bin_probability"]) + info["outside_basis_interval_fraction"], 1)
        self.assertAlmostEqual(info["outside_basis_interval_fraction"], 1 / 3)

    def test_smoke_is_reproducible_does_not_fit_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            with patch("regression.fit_theta", side_effect=AssertionError("must not fit")):
                result = prepare(self.spec, first, smoke=True)
                repeat = prepare(self.spec, second, smoke=True)
            self.assertEqual(result["status"], "smoke_passed")
            self.assertFalse(result["fitting_performed"])
            self.assertEqual(self.spec["simulation"]["N"], 100)
            for split in ("train", "validation", "test"):
                entry = result["splits"][split][0]
                again = repeat["splits"][split][0]
                self.assertEqual(entry["observation_sha256"], again["observation_sha256"])
                with np.load(first / split / str(entry["seed"]) / "data.npz") as data:
                    self.assertEqual(array_hash(data["Y"]), entry["observation_sha256"])
                for name in entry["designs"]:
                    self.assertEqual(entry["designs"][name]["shape"][0], 200)
                    self.assertEqual(entry["designs"][name]["sha256"], again["designs"][name]["sha256"])
            with self.assertRaises(FileExistsError):
                prepare(self.spec, first, smoke=True)


if __name__ == "__main__":
    unittest.main()
