from copy import deepcopy
import unittest

import numpy as np

from scripts.run_predator_prey_sweeps import (
    DEFAULT_CONFIG,
    condition_plan,
    condition_signature,
    effective_protocol,
    observation_view,
    paired_inputs,
    read_json,
    species_balanced_drift_error,
    split_seed,
    validate_protocol,
)


class PredatorPreySweepTests(unittest.TestCase):
    def setUp(self):
        self.spec = read_json(DEFAULT_CONFIG)

    def test_approved_grid_and_one_factor_conditions(self):
        validate_protocol(self.spec)
        conditions = condition_plan(self.spec)
        self.assertEqual(len(conditions), 13)
        self.assertEqual(len({condition_signature(item) for item in conditions}), 11)
        baseline = (100, 0.01, 0.001)
        self.assertEqual(sum(condition_signature(item) == baseline for item in conditions), 3)
        for condition in conditions:
            changed = []
            if condition["prey_particles"] != baseline[0]:
                changed.append("prey_particles")
            if condition["observation_dt"] != baseline[1]:
                changed.append("observation_dt")
            if condition["observation_noise"] != baseline[2]:
                changed.append("observation_noise")
            self.assertEqual(changed, [] if condition_signature(condition) == baseline else [condition["sweep"]])

    def test_protocol_rejects_grid_or_dictionary_drift(self):
        changed = deepcopy(self.spec)
        changed["sweeps"]["prey_particles"][-1] = 201
        with self.assertRaises(ValueError):
            validate_protocol(changed)
        changed = deepcopy(self.spec)
        changed["dictionary"]["prey_prey_true"][0] = 0.9
        with self.assertRaises(ValueError):
            validate_protocol(changed)

    def test_smoke_is_explicit_and_scientific_defaults_remain_intact(self):
        scientific = effective_protocol(self.spec, False)
        smoke = effective_protocol(self.spec, True)
        self.assertFalse(scientific["smoke_only"])
        self.assertEqual(scientific["evaluation"]["scientific_repetitions"], 20)
        self.assertEqual(scientific["observation"]["training_trajectories"], 4)
        self.assertTrue(smoke["smoke_only"])
        self.assertEqual(smoke["evaluation"]["scientific_repetitions"], 1)
        self.assertEqual(smoke["observation"]["training_trajectories"], 1)
        self.assertEqual(smoke["estimator"]["alpha_count"], 60)

    def test_pairing_uses_nested_particles_and_independent_splits(self):
        smoke = effective_protocol(self.spec, True)
        training = paired_inputs(smoke, 2, "training")
        training_again = paired_inputs(smoke, 2, "training")
        validation = paired_inputs(smoke, 2, "validation")
        np.testing.assert_array_equal(training["initial_prey"], training_again["initial_prey"])
        np.testing.assert_array_equal(training["unit_prey_noise"], training_again["unit_prey_noise"])
        np.testing.assert_array_equal(
            training["initial_prey"][:, :25], training_again["initial_prey"][:, :25]
        )
        self.assertNotEqual(split_seed(smoke, 2, "training"), split_seed(smoke, 2, "validation"))
        self.assertFalse(np.array_equal(training["initial_predator"], validation["initial_predator"]))

    def test_observation_noise_and_dt_views_share_fine_arrays(self):
        prey = np.arange(1 * 9 * 3 * 2, dtype=float).reshape(1, 9, 3, 2)
        predator = np.arange(1 * 9 * 1 * 2, dtype=float).reshape(1, 9, 1, 2)
        prey_noise = np.linspace(-1.0, 1.0, prey.size).reshape(prey.shape)
        predator_noise = np.linspace(-0.5, 0.5, predator.size).reshape(predator.shape)
        clean = observation_view(prey, predator, 0.005, prey_noise, predator_noise, 0.005, 0.01, 0.0)
        noisy = observation_view(prey, predator, 0.005, prey_noise, predator_noise, 0.005, 0.01, 0.003)
        np.testing.assert_allclose(noisy[0] - clean[0], 0.003 * prey_noise[:, ::2])
        fine = observation_view(prey, predator, 0.005, prey_noise, predator_noise, 0.005, 0.005, 0.0)
        np.testing.assert_array_equal(clean[0], fine[0][:, ::2])
        np.testing.assert_array_equal(clean[1], fine[1][:, ::2])

    def test_held_out_drift_balances_species_not_particle_count(self):
        rng = np.random.default_rng(7)
        prey = rng.normal(size=(1, 3, 25, 2))
        predator = rng.normal(size=(1, 3, 1, 2))

        def identity(displacement):
            return np.asarray(displacement)

        def doubled(displacement):
            return 2.0 * np.asarray(displacement)

        metrics = species_balanced_drift_error(
            prey, predator, (identity, identity, identity),
            (doubled, doubled, doubled), stride=1,
        )
        self.assertAlmostEqual(metrics["prey_relative"], 1.0)
        self.assertAlmostEqual(metrics["predator_relative"], 1.0)
        self.assertAlmostEqual(metrics["species_balanced_relative"], 1.0)


if __name__ == "__main__":
    unittest.main()
