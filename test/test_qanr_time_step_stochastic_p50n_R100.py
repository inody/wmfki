import json
from pathlib import Path
import unittest

from scripts.run_qanr_time_step_stochastic_p50n_R100 import (
    effective_spec,
    pair_seeds,
    validate_spec,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/qanr_redundant6_time_step_stochastic_p50n_R100.json"


class QANRStochasticTimeStepP50NR100Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(CONFIG.read_text())
        cls.base = json.loads((ROOT / cls.spec["base_protocol"]).read_text())

    def test_production_protocol(self):
        validate_spec(self.spec, self.base)
        self.assertEqual(self.spec["process_noise"], 0.1)
        self.assertEqual(self.spec["observation_noise"], 0.0)
        self.assertEqual(self.spec["simulation_step"], 0.001)
        self.assertEqual(self.spec["pair_sampling"]["pairs"], 50 * self.spec["particles"])

    def test_every_observation_interval_is_a_subsample(self):
        h = self.spec["simulation_step"]
        strides = [round(dt / h) for dt in self.spec["observation_intervals"]]
        self.assertEqual(strides, [100, 50, 20, 10, 5, 2])
        for dt, stride in zip(self.spec["observation_intervals"], strides):
            self.assertAlmostEqual(dt, stride * h)

    def test_pair_streams_are_split_independent(self):
        seen = set()
        for repetition in range(self.spec["repetitions"]):
            trajectory_seed = (
                self.spec["training_seed_start"]
                + self.spec["training_seed_stride"] * repetition
            )
            for condition_index, _ in enumerate(self.spec["observation_intervals"]):
                train, validation = pair_seeds(
                    self.spec, repetition, condition_index, trajectory_seed
                )
                self.assertNotEqual(train, validation)
                self.assertNotIn((train, validation), seen)
                seen.add((train, validation))

    def test_smoke_protocol_is_small_but_structurally_matched(self):
        smoke = effective_spec(self.spec, True)
        self.assertEqual(smoke["repetitions"], 2)
        self.assertEqual(smoke["particles"], 30)
        self.assertEqual(smoke["pair_sampling"]["ratio"], 50)
        self.assertEqual(smoke["pair_sampling"]["pairs"], 1500)
        self.assertEqual(smoke["simulation_step"], self.spec["simulation_step"])
        self.assertEqual(smoke["process_noise"], self.spec["process_noise"])


if __name__ == "__main__":
    unittest.main()
