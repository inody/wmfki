from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_1d_hat_M17_pair_R100 import job_plan, validate_protocol
from scripts.summarize_1d_hat_M17_pair_R100 import extended_statistics
from test.test_synthetic_protocol_fixtures import (
    ROOT,
    completed_status,
    read_config,
    relative,
    synthetic_hat_r100_frozen,
    synthetic_hat_sweep_frozen,
    write_json,
)


class TestM17PairR100(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        pair_batch = root / "synthetic-pair-r20"
        reference_batch = root / "synthetic-reference-r80"

        pair_frozen = synthetic_hat_sweep_frozen()
        pair_path = write_json(pair_batch / "frozen_protocol.json", pair_frozen)
        pair_status = completed_status(pair_path, 120, job_prefix="pair")
        write_json(pair_batch / "batch_status.json", pair_status)

        reference_frozen = synthetic_hat_r100_frozen()
        reference_path = write_json(
            reference_batch / "frozen_protocol.json", reference_frozen
        )
        reference_status = completed_status(reference_path, 80, job_prefix="base")
        write_json(reference_batch / "batch_status.json", reference_status)

        cls.spec = read_config("one_dimensional_hat_M17_pair_R100.json")
        cls.spec["source_pair_batch"] = relative(pair_batch)
        cls.spec["source_reference_batch"] = relative(reference_batch)
        cls.pair_frozen = pair_frozen
        cls.pair_status = pair_status
        cls.reference_frozen = reference_frozen
        cls.reference_status = reference_status

    def validate(self, spec):
        validate_protocol(
            spec,
            self.pair_frozen,
            self.pair_status,
            self.reference_frozen,
            self.reference_status,
        )

    def test_protocol_has_six_pair_counts_and_480_new_jobs(self):
        self.validate(self.spec)
        jobs = job_plan(self.spec)
        self.assertEqual(len(jobs), 480)
        self.assertEqual(sorted({job["P"] for job in jobs.values()}), self.spec["pair_counts"])
        self.assertEqual(sorted({job["global_index"] for job in jobs.values()}), list(range(20, 100)))
        self.assertTrue(all(
            job["reference_key"] == f"base-{job['local_index']}"
            for job in jobs.values()
        ))

    def test_new_pair_seed_family_extends_without_overlap(self):
        old = {2_000_000 + 1000 * k + 10 * pi for pi in range(6) for k in range(20)}
        new = {
            2_000_000 + 1000 * (20 + k) + 10 * pi
            for pi in range(6) for k in range(80)
        }
        self.assertFalse(old & new)
        self.assertEqual(len(old), 120)
        self.assertEqual(len(new), 480)

    def test_changes_to_count_grid_or_dictionary_are_rejected(self):
        for change in ("dictionary", "count", "grid"):
            spec = deepcopy(self.spec)
            if change == "dictionary":
                spec["dictionary_size"] = 16
            elif change == "count":
                spec["new_repetitions"] = 79
            else:
                spec["pair_counts"] = [100, 1000]
            with self.assertRaises(ValueError):
                self.validate(spec)

    def test_smoke_plan_is_small_and_uses_declared_reference_keys(self):
        jobs = job_plan(self.spec, smoke=True)
        self.assertEqual(len(jobs), 4)
        self.assertEqual(sorted({job["P"] for job in jobs.values()}), [100, 300])
        self.assertEqual(sorted({job["local_index"] for job in jobs.values()}), [0, 1])
        self.assertEqual({job["reference_key"] for job in jobs.values()}, {"base-0", "base-1"})

    def test_extended_statistics_never_averages_missing_values(self):
        indices = np.array([[0, 1], [1, 0]])
        complete = extended_statistics([2.0, 4.0], indices)
        self.assertEqual(complete["mean"], 3.0)
        self.assertEqual(complete["minimum"], 2.0)
        missing = extended_statistics([2.0, None], indices)
        self.assertIsNone(missing["mean"])
        self.assertEqual(missing["available_count"], 1)


if __name__ == "__main__":
    unittest.main()
