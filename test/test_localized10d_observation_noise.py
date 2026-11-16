from copy import deepcopy
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.run_localized10d_observation_noise import (
    fixed_space_test_spec,
    observed_trajectory,
    summarize,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


class Localized10DObservationNoiseTests(unittest.TestCase):
    def setUp(self):
        self.sweep = json.loads((
            ROOT / "configs/experiments/localized10d_observation_noise.json"
        ).read_text())
        self.base = json.loads((ROOT / self.sweep["base_config"]).read_text())

    def test_protocol_and_predeclared_nominal_balance_without_archive(self):
        sweep = deepcopy(self.sweep)
        sweep["clean_reference_records"] = None
        sweep["clean_reference_sha256"] = None
        validate_protocol(sweep, self.base)
        h = self.base["model"]["dt"] * self.base["model"]["observation_stride"]
        sigma = self.base["model"]["sigma"]
        self.assertAlmostEqual(
            sweep["nominal_observation_noise"], sigma * np.sqrt(h / 2.0)
        )

    def test_fixed_noise_benchmark_protocol(self):
        fixed = json.loads((
            ROOT / "configs/experiments/localized10d_observation_noise_003.json"
        ).read_text())
        fixed["clean_reference_records"] = None
        fixed["clean_reference_sha256"] = None
        validate_protocol(fixed, self.base)
        self.assertEqual(fixed["observation_noise_values"], [0.03])
        self.assertEqual(
            fixed["spatial_test_construction"],
            "data_dependent_from_each_observed_training_or_validation_array",
        )

    def test_observation_noise_levels_share_standard_normal_array(self):
        latent = np.arange(12, dtype=float).reshape(1, 2, 3, 2)
        noise = np.linspace(-1.0, 1.0, latent.size).reshape(latent.shape)
        low = observed_trajectory(latent, 0.001, noise)
        high = observed_trajectory(latent, 0.01, noise)
        np.testing.assert_allclose((high - latent), 10.0 * (low - latent))
        np.testing.assert_array_equal(observed_trajectory(latent, 0.0, noise), latent)

    def test_summary_keeps_every_method_and_noise_level(self):
        sweep = json.loads(json.dumps(self.sweep))
        sweep["observation_noise_values"] = [0.0, 0.005]
        sweep["metric"]["bootstrap_draws"] = 100
        records = []
        for sigma_obs in sweep["observation_noise_values"]:
            for repetition in range(3):
                for index, method in enumerate((
                    "weak_lasso", "weak_ols", "strong_forward_difference_ols"
                )):
                    records.append({
                        "observation_noise": sigma_obs, "repetition": repetition,
                        "method": method, "coefficient_error_relative": index + 1.0,
                        "latent_drift_relative_error": index + 2.0,
                        "profile_relative_error": index + 3.0,
                        "status": "finite", "rank_deficient": False,
                        "selected_at_grid_boundary": False,
                        "unconverged_candidates": 0,
                    })
        summary = summarize(records, sweep)
        self.assertEqual(len(summary["groups"]), 6)
        self.assertEqual(summary["nominal_coefficient_rmse_winner"], "weak_lasso")
        self.assertTrue(all(group["coefficient_error_relative"]["count"] == 3
                            for group in summary["groups"]))

    def test_spatial_tests_are_fixed_from_latent_trajectory(self):
        rng = np.random.default_rng(3)
        latent = rng.normal(size=(1, 5, 7, 10))
        fixed = fixed_space_test_spec(self.base, latent, 123)
        noisy = observed_trajectory(latent, 0.03, rng.normal(size=latent.shape))
        self.assertEqual(fixed, fixed_space_test_spec(self.base, latent, 123))
        self.assertNotEqual(fixed, fixed_space_test_spec(self.base, noisy, 123))
        self.assertEqual(len(fixed["centers"]), self.base["weak_tests"]["spatial_count"])

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            observed_trajectory(np.zeros((2, 2)), 0.1, np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
