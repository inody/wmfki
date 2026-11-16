from copy import deepcopy
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.run_localized10d_noisy_robustness import (
    _add_observation_noise,
    effective_protocol,
    validate_protocol,
)
from test.test_synthetic_protocol_fixtures import (
    digest,
    relative,
    repository_temporary_directory,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


class Localized10DNoisyRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.requested = json.loads((
            ROOT / "configs/experiments/localized10d_noisy_robustness.json"
        ).read_text())
        self.base = json.loads((ROOT / self.requested["base_config"]).read_text())
        self.nominal = json.loads((ROOT / self.requested["nominal_config"]).read_text())

    def test_protocol_is_fixed_at_point_zero_three_with_synthetic_records(self):
        requested = deepcopy(self.requested)
        temporary = repository_temporary_directory(self)
        records = write_json(temporary / "nominal-records.json", [])
        requested["nominal_reference_records"] = relative(records)
        requested["nominal_reference_sha256"] = digest(records)
        validate_protocol(requested, self.base, self.nominal)
        spec = effective_protocol(requested, self.base)
        self.assertEqual(spec["observation_noise"], 0.03)
        self.assertEqual(spec["evaluation"]["scientific_repetitions"], 20)
        self.assertEqual(
            spec["evaluation"]["rollout_particles"], spec["model"]["particles"]
        )

    def test_noise_is_reproducible_and_nested_in_particle_number(self):
        spec = effective_protocol(self.requested, self.base)
        plan = {"index": 2}
        full = np.zeros((1, 4, spec["model"]["particles"], 10))
        small = np.zeros((1, 4, 100, 10))
        observed_full, seed_full = _add_observation_noise(
            spec, full, spec["model"]["particles"], plan, "training"
        )
        observed_repeat, seed_repeat = _add_observation_noise(
            spec, full, spec["model"]["particles"], plan, "training"
        )
        observed_small, seed_small = _add_observation_noise(
            spec, small, 100, plan, "training"
        )
        np.testing.assert_array_equal(observed_full, observed_repeat)
        np.testing.assert_array_equal(observed_small, observed_full[:, :, :100, :])
        self.assertEqual(seed_full, seed_repeat)
        self.assertEqual(seed_full, seed_small)

    def test_training_and_validation_noise_are_independent(self):
        spec = effective_protocol(self.requested, self.base, smoke=True)
        latent = np.zeros((1, 3, spec["model"]["particles"], 10))
        plan = {"index": 0}
        training, training_seed = _add_observation_noise(
            spec, latent, spec["model"]["particles"], plan, "training"
        )
        validation, validation_seed = _add_observation_noise(
            spec, latent, spec["model"]["particles"], plan, "validation"
        )
        self.assertNotEqual(training_seed, validation_seed)
        self.assertFalse(np.array_equal(training, validation))


if __name__ == "__main__":
    unittest.main()
