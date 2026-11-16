import json
from pathlib import Path
import unittest

from scripts.run_1d_hat_sampled_p50n_R100 import (
    job_plan,
    load_sources,
    reference_indices,
    validate_sources,
    validate_spec,
    worker_spec,
)
from test.test_synthetic_protocol_fixtures import (
    materialize_matched_dictionary_sources,
    relative,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/one_dimensional_hat_sampled_p50n_R100.json"


class OneDimensionalHatSampledP50NR100Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(CONFIG.read_text())

    def test_production_protocol_lock(self):
        validate_spec(self.spec)
        self.assertEqual(self.spec["pair_ratio"], 50)
        self.assertEqual(self.spec["pair_count"], 50000)
        self.assertEqual(self.spec["repetitions"], 100)
        self.assertEqual(reference_indices(self.spec), list(range(100)))
        self.assertTrue(self.spec["pair_sampling"]["independent_training_validation"])
        self.assertTrue(self.spec["production_artifacts_unchanged"])

    def test_synthetic_matched_sources_preserve_scientific_protocol(self):
        source, hashes = materialize_matched_dictionary_sources(self, 100)
        effective = worker_spec(self.spec)
        effective["source_lasso_batch"] = relative(source["lasso"])
        effective["source_ols_batch"] = relative(source["ols"])
        effective["source_lasso_frozen_protocol_sha256"] = hashes["lasso"]
        effective["source_ols_frozen_protocol_sha256"] = hashes["ols"]
        loaded = load_sources(effective)
        base = validate_sources(effective, loaded)
        self.assertEqual(base["simulation"], {
            "N": 1000, "horizon": 2.0, "dt": 0.005,
            "sigma": 0.1, "sigma_obs": 0.0,
        })
        self.assertEqual(base["basis"]["sizes"], self.spec["dictionary_sizes"])
        self.assertEqual(len(loaded["lasso_frozen"]["references"]), 100)

    def test_job_plan_has_all_predeclared_repetitions(self):
        jobs = job_plan(worker_spec(self.spec))
        self.assertEqual(len(jobs), 100)
        self.assertEqual({job["repetition"] for job in jobs.values()}, set(range(100)))
        self.assertEqual({job["ratio"] for job in jobs.values()}, {50})

    def test_pair_seed_schedule_is_unique_and_split_independent(self):
        pairs = set()
        for repetition in reference_indices(self.spec):
            base = self.spec["pair_seed_start"] + repetition * self.spec["repetition_seed_stride"]
            train = base + self.spec["training_pair_seed_offset"]
            validation = base + self.spec["validation_pair_seed_offset"]
            self.assertNotEqual(train, validation)
            self.assertNotIn((train, validation), pairs)
            pairs.add((train, validation))
        self.assertEqual(len(pairs), 100)


if __name__ == "__main__":
    unittest.main()
