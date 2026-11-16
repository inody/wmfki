import copy
import unittest

import numpy as np

from scripts.run_qanr_matched_lasso_sweeps import read_json
from scripts.run_qanr_time_step_repetitions import (
    DEFAULT_CONFIG,
    LEGACY_CONFIG,
    PAIRED_TASK,
    effective_protocol,
    simulate_paired_positions,
    summarize,
    validate_time_protocol,
)


class TestQANRTimeStepRepetitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_json(DEFAULT_CONFIG)
        cls.base = read_json(cls.protocol["base_protocol"])

    def test_scientific_protocol_is_fixed(self):
        validate_time_protocol(self.protocol, self.base)
        self.assertEqual(self.protocol["task"], PAIRED_TASK)
        self.assertEqual(self.protocol["protocol_version"], 2)
        self.assertEqual(self.protocol["run_class"], "production")
        self.assertEqual(self.protocol["time_step"]["repetitions"], 100)
        self.assertEqual(self.protocol["time_step"]["stacked_trajectories"], 3)
        self.assertEqual(self.protocol["time_step"]["sigma"], 0.0)
        self.assertEqual(self.protocol["time_step"]["solver"], "euler")
        self.assertTrue(
            self.protocol["time_step"]["trajectory_seed_policy"]
            ["paired_across_time_steps"]
        )

    def test_legacy_protocol_remains_readable_but_is_not_default(self):
        legacy = read_json(LEGACY_CONFIG)
        base = read_json(legacy["base_protocol"])
        validate_time_protocol(legacy, base)
        self.assertNotEqual(DEFAULT_CONFIG, LEGACY_CONFIG)

    def test_seed_policy_is_locked(self):
        changed = copy.deepcopy(self.protocol)
        changed["time_step"]["trajectory_seed_policy"]["namespace"] += 1
        with self.assertRaises(ValueError):
            validate_time_protocol(changed, self.base)

    def test_smoke_is_separate_and_retains_three_paired_trajectories(self):
        smoke = effective_protocol(self.protocol, smoke=True)
        self.assertEqual(smoke["time_step"]["repetitions"], 2)
        self.assertEqual(smoke["time_step"]["stacked_trajectories"], 3)
        self.assertNotEqual(smoke["task"], self.protocol["task"])

    def test_all_stacked_initial_states_are_bitwise_paired_across_dt(self):
        common = {
            "particles": 12,
            "repetitions": 3,
            "horizon": 0.04,
            "sigma": 0.0,
            "master_seed": 1234,
            "solver": "euler",
            "spec": self.base,
            "policy": self.protocol["time_step"]["trajectory_seed_policy"],
        }
        _, first, _, seeds_first, hashes_first = simulate_paired_positions(
            dt=0.01, **common
        )
        _, second, _, seeds_second, hashes_second = simulate_paired_positions(
            dt=0.02, **common
        )
        self.assertEqual(seeds_first, seeds_second)
        self.assertEqual(hashes_first, hashes_second)
        self.assertEqual(len(set(hashes_first)), 3)
        for trajectory in range(3):
            self.assertTrue(np.array_equal(
                first[trajectory, 0], second[trajectory, 0]
            ))

    def test_summary_retains_all_records(self):
        records = []
        for repetition, scale in enumerate((1.0, 2.0)):
            for dt in (0.01, 0.02):
                records.append({
                    "repetition": repetition,
                    "dt": dt,
                    "coefficient_error_relative": scale * dt,
                    "ols_coefficient_error_relative": 2.0 * scale * dt,
                    "selected_at_grid_boundary": repetition == 0,
                    "unconverged_candidates": repetition,
                    "ols_rank_deficient": False,
                })
        summary = summarize(records)
        self.assertEqual(len(summary["time_step"]), 2)
        self.assertEqual(summary["time_step"][0]["repetitions"], 2)
        self.assertEqual(summary["diagnostics"]["models"], 4)
        self.assertAlmostEqual(summary["diagnostics"]["lasso_log_log_slope_dt_le_0_02"], 1.0)


if __name__ == "__main__":
    unittest.main()
