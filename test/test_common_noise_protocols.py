from copy import deepcopy
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.run_localized10d_observation_noise import (
    validate_protocol as validate_localized_observation_protocol,
)
from scripts.run_localized10d_noisy_robustness import (
    effective_protocol as localized_robustness_effective_protocol,
    validate_protocol as validate_localized_robustness_protocol,
)
from scripts.run_predator_prey_application import (
    validate_protocol as validate_predator_application_protocol,
)
from scripts.run_predator_prey_sweeps import (
    condition_plan,
    condition_signature,
    validate_protocol as validate_predator_sweep_protocol,
)
from scripts.run_qanr_common_noise_pilot import (
    _stacked_trajectory_seed,
    simulate_positions,
    validate_pilot as validate_qanr_common_noise_pilot,
)
from test.test_synthetic_protocol_fixtures import (
    digest,
    relative,
    repository_temporary_directory,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


class CommonObservationNoiseProtocolTests(unittest.TestCase):
    def test_predator_application_accepts_common_noise_protocol(self):
        spec = read("configs/experiments/predator_prey_application_common_noise_001.json")
        validate_predator_application_protocol(spec)
        self.assertEqual(spec["observation"]["observation_noise"], 0.01)

    def test_predator_sweep_uses_common_noise_for_dt_and_particle_panels(self):
        spec = read("configs/experiments/predator_prey_sweeps_common_noise_001.json")
        validate_predator_sweep_protocol(spec)
        baseline = (100, 0.01, 0.01)
        conditions = condition_plan(spec)
        self.assertEqual(sum(condition_signature(row) == baseline for row in conditions), 3)
        self.assertTrue(all(
            row["observation_noise"] == 0.01
            for row in conditions
            if row["sweep"] in {"observation_dt", "prey_particles"}
        ))

    def test_localized_fixed_benchmark_accepts_common_noise_without_archive(self):
        fixed = deepcopy(read("configs/experiments/localized10d_observation_noise_001.json"))
        base = read(fixed["base_config"])
        fixed["clean_reference_records"] = None
        fixed["clean_reference_sha256"] = None
        validate_localized_observation_protocol(fixed, base)
        self.assertEqual(fixed["observation_noise_values"], [0.01])

    def test_localized_robustness_accepts_matching_synthetic_nominal_records(self):
        requested = deepcopy(read("configs/experiments/localized10d_noisy_robustness_001.json"))
        base = read(requested["base_config"])
        nominal = read(requested["nominal_config"])
        temporary = repository_temporary_directory(self)
        records = write_json(temporary / "nominal-records.json", [])
        requested["nominal_reference_records"] = relative(records)
        requested["nominal_reference_sha256"] = digest(records)
        validate_localized_robustness_protocol(requested, base, nominal)
        effective = localized_robustness_effective_protocol(requested, base)
        self.assertEqual(effective["observation_noise"], 0.01)
        self.assertEqual(effective["evaluation"]["scientific_repetitions"], 20)

    def test_qanr_common_noise_pilot_is_predeclared(self):
        requested = read("configs/experiments/qanr_redundant6_common_noise_pilot.json")
        base = read(requested["base_protocol"])
        validate_qanr_common_noise_pilot(requested, base)
        self.assertEqual(requested["observation_noise"], 0.01)
        self.assertEqual(requested["repetitions"], 10)

    def test_qanr_stacked_initial_states_are_paired_across_time_steps(self):
        requested = read("configs/experiments/qanr_redundant6_common_noise_pilot.json")
        base = read(requested["base_protocol"])
        seeds = [_stacked_trajectory_seed(1234, index) for index in range(3)]
        self.assertEqual(len(set(seeds)), 3)
        arrays = []
        for dt in (0.01, 0.005):
            _, positions, _ = simulate_positions(
                requested["time_step"], particles=12, horizon=0.02, dt=dt,
                sigma=0.0, seed=1234, solver="euler", repetitions=3, base=base,
            )
            arrays.append(positions[:, 0])
        np.testing.assert_array_equal(arrays[0], arrays[1])
        self.assertFalse(np.array_equal(arrays[0][0], arrays[0][1]))


if __name__ == "__main__":
    unittest.main()
