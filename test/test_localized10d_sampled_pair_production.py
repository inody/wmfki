import unittest

from scripts.run_localized10d_sampled_pair_production import (
    effective_protocol,
    pair_budget,
    pair_sampler_contract_audit,
    source_protocol_audit,
)
from test.test_synthetic_protocol_fixtures import (
    read_config,
    synthetic_localized_frozen,
)


class Localized10DSampledPairProductionTests(unittest.TestCase):
    def setUp(self):
        self.requested = read_config(
            "localized10d_sampled_pair_production_P50_R20.json"
        )
        self.source_frozen = synthetic_localized_frozen(0.01)

    def test_synthetic_source_protocol_is_P50_R20(self):
        spec = effective_protocol(self.requested, self.source_frozen)
        self.assertEqual(spec["observation_noise"], 0.01)
        self.assertEqual(spec["evaluation"]["scientific_repetitions"], 20)
        self.assertEqual(spec["pair_sampling"]["baseline_pairs"], 25000)
        self.assertEqual(spec["pair_sampling"]["mode"], "resample_each_time")
        self.assertEqual(
            spec["sweeps"]["sampled_pairs"],
            [3000, 5000, 10000, 20000, 25000, 30000],
        )
        self.assertTrue(source_protocol_audit(self.source_frozen, spec)["passed"])

    def test_pair_budgets_are_fifty_n(self):
        spec = effective_protocol(self.requested, self.source_frozen)
        self.assertEqual(pair_budget(spec, 100), 5000)
        self.assertEqual(pair_budget(spec, 200), 10000)
        self.assertEqual(pair_budget(spec, 350), 17500)
        self.assertEqual(pair_budget(spec, 500), 25000)
        self.assertEqual(spec["pair_count_penalty_policy"]["selection_pairs"], 25000)

    def test_pair_sampler_contract(self):
        audit = pair_sampler_contract_audit()
        self.assertTrue(audit["passed"], audit)
        self.assertTrue(all(audit["checks"].values()))


if __name__ == "__main__":
    unittest.main()
