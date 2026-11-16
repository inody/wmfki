import json
from pathlib import Path
import unittest

from scripts.run_1d_hat_sampled_pair_ratio_sweep_R5 import (
    job_plan,
    load_sources,
    pair_seed,
    spearman,
    validate_sources,
    validate_spec,
)
from test.test_synthetic_protocol_fixtures import (
    materialize_matched_dictionary_sources,
    relative,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/one_dimensional_hat_sampled_pair_ratio_sweep_R5.json"


class OneDimensionalHatSampledPairRatioSweepR5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(CONFIG.read_text())

    def test_protocol_lock(self):
        validate_spec(self.spec)
        self.assertEqual(self.spec["pair_ratios"], [20, 30, 50, 75, 100])
        self.assertEqual(self.spec["reference_indices"], [0, 1, 2, 3, 4])
        self.assertEqual(self.spec["acceptance"]["maximum_sampled_to_full_rms_ratio"], 1.25)
        self.assertTrue(self.spec["production_artifacts_unchanged"])

    def test_synthetic_matched_sources_preserve_authoritative_protocol(self):
        source, hashes = materialize_matched_dictionary_sources(self, 5)
        spec = json.loads(json.dumps(self.spec))
        spec["source_lasso_batch"] = relative(source["lasso"])
        spec["source_ols_batch"] = relative(source["ols"])
        spec["source_lasso_frozen_protocol_sha256"] = hashes["lasso"]
        spec["source_ols_frozen_protocol_sha256"] = hashes["ols"]
        loaded = load_sources(spec)
        base = validate_sources(spec, loaded)
        self.assertEqual(base["simulation"], {
            "N": 1000, "horizon": 2.0, "dt": 0.005,
            "sigma": 0.1, "sigma_obs": 0.0,
        })
        self.assertEqual(base["basis"]["sizes"], spec["dictionary_sizes"])
        self.assertEqual(base["pair_mode"], "full")
        for index in spec["reference_indices"]:
            key = f"run-{index}"
            self.assertEqual(
                loaded["lasso_frozen"]["references"][key],
                loaded["ols_frozen"]["references"][key],
            )

    def test_job_count_and_pair_counts(self):
        jobs = job_plan(self.spec)
        self.assertEqual(len(jobs), 25)
        self.assertEqual({job["ratio"] * 1000 for job in jobs.values()},
                         {20000, 30000, 50000, 75000, 100000})

    def test_pair_seeds_are_distinct(self):
        seen = set()
        for ratio_index, _ in enumerate(self.spec["pair_ratios"]):
            for repetition in self.spec["reference_indices"]:
                training = pair_seed(self.spec, repetition, ratio_index, "training")
                validation = pair_seed(self.spec, repetition, ratio_index, "validation")
                self.assertNotEqual(training, validation)
                self.assertNotIn(training, seen)
                self.assertNotIn(validation, seen)
                seen.update((training, validation))
        self.assertEqual(len(seen), 50)

    def test_spearman_trend_diagnostic(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)


if __name__ == "__main__":
    unittest.main()
