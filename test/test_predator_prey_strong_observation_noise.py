from copy import deepcopy
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.run_cross_example_nominal_comparison import assemble_predator_prey_strong
from scripts.run_predator_prey_strong_observation_noise import (
    effective_protocol,
    observation_conditions,
    validate_protocol,
)
from scripts.run_predator_prey_sweeps import model_objects
from test.test_synthetic_protocol_fixtures import (
    digest,
    relative,
    repository_temporary_directory,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


class PredatorPreyStrongObservationNoiseTests(unittest.TestCase):
    def setUp(self):
        self.requested = json.loads((
            ROOT / "configs/experiments/predator_prey_strong_observation_noise.json"
        ).read_text())
        self.base = json.loads((ROOT / self.requested["base_config"]).read_text())

    def test_protocol_uses_the_existing_noise_grid_with_synthetic_records(self):
        requested = deepcopy(self.requested)
        temporary = repository_temporary_directory(self)
        records = write_json(temporary / "weak-sweep-records.json", [])
        requested["weak_sweep_records"] = relative(records)
        requested["weak_sweep_records_sha256"] = digest(records)
        validate_protocol(requested, self.base)
        spec = effective_protocol(requested, self.base)
        conditions = observation_conditions(spec)
        self.assertEqual(
            [condition["value"] for condition in conditions],
            self.base["sweeps"]["observation_noise"],
        )
        self.assertTrue(all(condition["observation_dt"] == 0.01
                            for condition in conditions))

    def test_strong_system_has_twelve_and_five_columns(self):
        prey_basis, predator_basis, _, _, _ = model_objects(self.base)
        rng = np.random.default_rng(4)
        prey = rng.normal(size=(1, 4, 6, 2))
        predator = rng.normal(size=(1, 4, 1, 2))
        time = np.arange(4, dtype=float) * 0.01
        A_prey, b_prey, A_predator, b_predator = assemble_predator_prey_strong(
            prey, predator, time, prey_basis, predator_basis
        )
        self.assertEqual(A_prey.shape[1], 12)
        self.assertEqual(A_predator.shape[1], 5)
        self.assertEqual(A_prey.shape[0], b_prey.size)
        self.assertEqual(A_predator.shape[0], b_predator.size)


if __name__ == "__main__":
    unittest.main()
