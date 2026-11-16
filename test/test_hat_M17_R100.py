from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_1d_hat_M17_R100 import (
    effective_base,
    effective_sweep,
    job_plan,
    seed_lists,
    unique_cases,
    validate_protocol,
)
from scripts.summarize_1d_hat_M17_R100 import extended_statistics
from test.test_synthetic_protocol_fixtures import (
    ROOT,
    completed_status,
    read_config,
    relative,
    synthetic_hat_sweep_frozen,
    write_json,
)


class TestM17R100(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        cls.addClassCleanup(cls.temporary.cleanup)
        source = Path(cls.temporary.name) / "synthetic-sweep"
        cls.frozen = synthetic_hat_sweep_frozen()
        frozen_path = write_json(source / "frozen_protocol.json", cls.frozen)
        cls.status = completed_status(frozen_path, 20, job_prefix="source")
        write_json(source / "batch_status.json", cls.status)
        cls.spec = read_config("one_dimensional_hat_M17_R100.json")
        cls.spec["source_batch"] = relative(source)

    def test_protocol_has_20_conditions_and_1600_new_jobs(self):
        validate_protocol(self.spec, self.frozen, self.status)
        base = effective_base(self.frozen, self.spec)
        sweep = effective_sweep(self.frozen, self.spec)
        self.assertEqual(base["basis"]["sizes"], [17])
        self.assertEqual(base["replication"]["count"], 80)
        self.assertEqual(len(unique_cases(sweep, base)), 20)
        jobs = job_plan(sweep, base)
        self.assertEqual(len(jobs), 1600)
        self.assertEqual(sum(job["case"] == "base" for job in jobs.values()), 80)
        self.assertTrue(all(
            job["simulation"] == unique_cases(sweep, base)[job["case"]]["simulation"]
            for job in jobs.values()
        ))

    def test_only_one_parameter_changes(self):
        base = effective_base(self.frozen, self.spec)
        sweep = effective_sweep(self.frozen, self.spec)
        for case_id, case in unique_cases(sweep, base).items():
            changed = [
                key for key in case["simulation"]
                if case["simulation"][key] != base["simulation"][key]
            ]
            self.assertEqual(changed, [] if case_id == "base" else [case["parameter"]])

    def test_new_seed_families_are_disjoint(self):
        seeds = seed_lists()
        trajectory = {seed for values in seeds.values() for seed in values}
        brownian = {seed + 1 for seed in trajectory}
        observation = {self.spec["observation_seed_offset"] + seed for seed in trajectory}
        old = {
            seed + offset
            for values in self.frozen["base"]["seeds"].values()
            for seed in values for offset in (0, 1)
        }
        families = [trajectory, brownian, observation, old]
        self.assertEqual(sum(map(len, families)), len(set.union(*families)))
        self.assertEqual(len(trajectory), 240)

    def test_changes_to_count_grid_or_dictionary_are_rejected(self):
        for change in ("dictionary", "count", "grid"):
            spec = deepcopy(self.spec)
            if change == "dictionary":
                spec["dictionary_size"] = 16
            elif change == "count":
                spec["new_repetitions"] = 81
            else:
                spec["sweeps"]["N"] = [50, 100, 1000]
            with self.assertRaises(ValueError):
                validate_protocol(spec, self.frozen, self.status)

    def test_smoke_is_small_but_retains_all_four_axes(self):
        base = effective_base(self.frozen, self.spec, True)
        sweep = effective_sweep(self.frozen, self.spec, True)
        self.assertEqual(base["replication"]["count"], 2)
        self.assertEqual(set(sweep["sweeps"]), {"sigma_obs", "sigma", "N", "dt"})
        self.assertEqual(len(unique_cases(sweep, base)), 5)
        self.assertEqual(len(job_plan(sweep, base)), 10)

    def test_extended_statistics_does_not_average_missing_values(self):
        indices = np.array([[0, 1], [1, 0]])
        complete = extended_statistics([2.0, 4.0], indices)
        self.assertEqual(complete["mean"], 3.0)
        self.assertEqual(complete["minimum"], 2.0)
        self.assertEqual(complete["maximum"], 4.0)
        missing = extended_statistics([2.0, None], indices)
        self.assertIsNone(missing["mean"])
        self.assertEqual(missing["available_count"], 1)
        self.assertEqual(missing["minimum"], 2.0)


if __name__ == "__main__":
    unittest.main()
