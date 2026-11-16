import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from predator_prey import (
    assemble_predator_prey_weak_systems,
    make_predator_prey_bases,
)
from scripts.run_predator_prey_sampled_pair_preview import (
    DEFAULT_CONFIG,
    assemble_sampled_predator_prey_weak_systems,
    effective_protocol,
    generate_preview_figures,
    make_time_independent_pair_provider,
    pair_seed,
    read_json,
    validate_preview_protocol,
)
from tests import make_test_functions
from time_tests import make_time_test_functions


ROOT = Path(__file__).resolve().parents[1]


class PredatorPreySampledPairPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requested = read_json(DEFAULT_CONFIG)
        cls.base = read_json(ROOT / cls.requested["base_config"])
        cls.spec = effective_protocol(cls.requested, cls.base)

    def test_preview_protocol_is_smoke_only(self):
        validate_preview_protocol(self.requested)
        self.assertTrue(self.spec["smoke_only"])
        self.assertEqual(self.spec["evaluation"]["scientific_repetitions"], 2)
        self.assertEqual(self.spec["observation"]["horizon"], 0.4)
        self.assertEqual(self.spec["pair_sampling"]["pairs_per_prey"], 10)

    def test_training_and_validation_pair_streams_are_independent(self):
        training_seed = pair_seed(self.spec, 1, "training")
        validation_seed = pair_seed(self.spec, 1, "validation")
        self.assertNotEqual(training_seed, validation_seed)
        training = make_time_independent_pair_provider(training_seed, 0.005)
        validation = make_time_independent_pair_provider(validation_seed, 0.005)
        train_i, train_j = training(12, 120, 0, 2, 0.01)
        valid_i, valid_j = validation(12, 120, 0, 2, 0.01)
        self.assertTrue(np.all(train_i != train_j))
        self.assertTrue(np.all(valid_i != valid_j))
        self.assertFalse(
            np.array_equal(train_i, valid_i) and np.array_equal(train_j, valid_j)
        )

    def test_pair_draws_are_time_independent_and_match_at_shared_physical_times(self):
        provider = make_time_independent_pair_provider(17, 0.005)
        first = provider(9, 90, 0, 1, 0.01)
        same_physical_time = provider(9, 90, 0, 99, 0.01)
        next_time = provider(9, 90, 0, 2, 0.015)
        np.testing.assert_array_equal(first[0], same_physical_time[0])
        np.testing.assert_array_equal(first[1], same_physical_time[1])
        self.assertFalse(
            np.array_equal(first[0], next_time[0])
            and np.array_equal(first[1], next_time[1])
        )

    def test_enumerating_all_ordered_off_diagonal_pairs_matches_exact_assembly(self):
        rng = np.random.default_rng(91)
        repetitions, time_count, particles = 2, 9, 7
        time = np.linspace(0.0, 0.08, time_count)
        prey = rng.normal(scale=0.3, size=(repetitions, time_count, particles, 2))
        predator = rng.normal(scale=0.2, size=(repetitions, time_count, 1, 2))
        prey_basis, predator_basis, _ = make_predator_prey_bases(
            0.05, [0.25, 0.5, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        coordinates = np.array([-0.5, 0.5])
        centers = np.array(np.meshgrid(coordinates, coordinates)).reshape(2, -1).T
        prey_tests = make_test_functions(
            {"centers": centers, "s": 0.5, "type": "gaussian"}
        )
        predator_tests = make_test_functions(
            {"centers": centers, "s": 0.7, "type": "gaussian"}
        )
        temporal = make_time_test_functions(
            {"time_tests": {"K": 3, "type": "bump", "s_factor": 1.8}}, time
        )
        first, second = np.where(~np.eye(particles, dtype=bool))

        def enumerate_pairs(particle_count, pair_count, trajectory_index, time_index, time_value):
            self.assertEqual(particle_count, particles)
            self.assertEqual(pair_count, particles * (particles - 1))
            return first, second

        sampled = assemble_sampled_predator_prey_weak_systems(
            prey,
            predator,
            time,
            prey_tests=prey_tests,
            predator_tests=predator_tests,
            prey_basis=prey_basis,
            predator_basis=predator_basis,
            time_tests=temporal,
            pair_count=particles * (particles - 1),
            pair_index_provider=enumerate_pairs,
        )
        exact = assemble_predator_prey_weak_systems(
            prey,
            predator,
            time,
            prey_tests=prey_tests,
            predator_tests=predator_tests,
            prey_basis=prey_basis,
            predator_basis=predator_basis,
            time_tests=temporal,
        )
        for species in ("prey", "predator"):
            np.testing.assert_allclose(
                sampled[species]["A"], exact[species]["A"], atol=2e-15, rtol=2e-13
            )
            np.testing.assert_array_equal(sampled[species]["b"], exact[species]["b"])
        self.assertEqual(sampled["pair_sampling"]["cross_species_sums"], "exact")
        self.assertEqual(
            sampled["pair_sampling"]["draw_count"], repetitions * time_count
        )

    def test_figure_mode_uses_records_only_and_writes_png_and_pdf(self):
        records = []
        sweep_values = {
            "observation_noise": [0.0, 0.01],
            "observation_dt": [0.01, 0.02],
            "prey_particles": [50, 100],
        }
        coefficients = {
            "prey_prey": [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            "prey_predator": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            "predator_prey": [0.0, 0.0, 0.4, 0.0, 0.0],
        }
        for repetition in range(2):
            for sweep, values in sweep_values.items():
                for index, value in enumerate(values):
                    condition = {
                        "id": f"{sweep}-{index:02d}",
                        "index": index,
                        "sweep": sweep,
                        "value": value,
                        "observation_noise": value if sweep == "observation_noise" else 0.01,
                        "observation_dt": value if sweep == "observation_dt" else 0.01,
                        "prey_particles": int(value) if sweep == "prey_particles" else 100,
                    }
                    methods = {}
                    for method, scale in (("lasso", 1.0), ("ols", 2.0)):
                        methods[method] = {
                            "held_out_drift": {
                                "species_balanced_relative": scale * (1.0 + repetition + index)
                            }
                        }
                    methods["lasso"]["channels"] = {
                        name: {"coefficients": values}
                        for name, values in coefficients.items()
                    }
                    records.append(
                        {
                            "repetition": repetition,
                            "condition": condition,
                            "methods": methods,
                        }
                    )
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            (run / "records.json").write_text(json.dumps(records))
            metadata = generate_preview_figures(run)
            self.assertEqual(metadata["record_count"], len(records))
            self.assertFalse(metadata["trajectory_generated"])
            self.assertEqual(len(metadata["files"]), 4)
            for name in metadata["files"]:
                self.assertTrue((run / "preview" / name).is_file())


if __name__ == "__main__":
    unittest.main()
