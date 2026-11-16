from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts.run_1d_hat_M17_observation_noise_extended import (
    combined_base,
    job_plan,
    validate,
)
from test.test_synthetic_protocol_fixtures import (
    ROOT,
    completed_status,
    read_config,
    relative,
    synthetic_hat_r100_frozen,
    synthetic_hat_sweep_frozen,
    write_json,
)


class TestM17ObservationNoiseExtended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        old = root / "synthetic-r20"
        total = root / "synthetic-r100"

        old_frozen = synthetic_hat_sweep_frozen()
        old_frozen_path = write_json(old / "frozen_protocol.json", old_frozen)
        old_status = completed_status(old_frozen_path, 580, job_prefix="old")
        write_json(old / "batch_status.json", old_status)

        total_frozen = synthetic_hat_r100_frozen()
        total_frozen["requested_protocol"] = deepcopy(
            read_config("one_dimensional_hat_M17_R100.json")
        )
        total_frozen["requested_protocol"]["source_batch"] = relative(old)
        total_frozen_path = write_json(total / "frozen_protocol.json", total_frozen)
        total_status = completed_status(total_frozen_path, 1600, job_prefix="total")
        write_json(total / "batch_status.json", total_status)

        cls.spec = read_config("one_dimensional_hat_M17_observation_noise_extended.json")
        cls.spec["source_total100_batch"] = relative(total)
        cls.state = (total, total_frozen, total_status, old, old_frozen, old_status)

    def test_protocol_and_job_count(self):
        validate(self.spec, self.state)
        base = combined_base(self.state)
        jobs = job_plan(self.spec, base)
        self.assertEqual(base["basis"]["sizes"], [17])
        self.assertEqual(base["replication"]["count"], 100)
        self.assertEqual(len(jobs), 400)
        self.assertEqual({job["value"] for job in jobs.values()}, {0.2, 0.4, 0.8, 1.0})

    def test_combined_seed_triplets_are_unique(self):
        base = combined_base(self.state)
        all_seeds = [
            seed for split in ("train", "validation", "test")
            for seed in base["seeds"][split]
        ]
        self.assertEqual(len(all_seeds), 300)
        self.assertEqual(len(set(all_seeds)), 300)
        self.assertFalse(set(all_seeds) & {seed + 1 for seed in all_seeds})

    def test_observation_draws_are_paired_with_original_families(self):
        jobs = job_plan(self.spec, combined_base(self.state))
        for job in jobs.values():
            expected = 1_000_000 if job["index"] < 20 else 10_000_000
            self.assertEqual(job["observation_seed_offset"], expected)

    def test_only_observation_noise_changes(self):
        base = combined_base(self.state)
        for job in job_plan(self.spec, base).values():
            changed = [
                key for key, value in job["simulation"].items()
                if value != base["simulation"][key]
            ]
            self.assertEqual(changed, ["sigma_obs"])


if __name__ == "__main__":
    unittest.main()
