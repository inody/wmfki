from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_1d_hat_dictionary_size_R100 import (
    effective_base,
    job_plan,
    validate_protocol,
)
from scripts.summarize_1d_hat_dictionary_size_R100 import extended_statistics
from test.test_synthetic_protocol_fixtures import (
    ROOT,
    completed_status,
    read_config,
    relative,
    synthetic_hat_r100_frozen,
    synthetic_hat_sweep_frozen,
    write_json,
)


class TestHatDictionarySizeR100(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        existing_batch = root / "synthetic-existing-r20"
        reference_batch = root / "synthetic-reference-r80"

        existing_frozen = synthetic_hat_sweep_frozen()
        existing_path = write_json(
            existing_batch / "frozen_protocol.json", existing_frozen
        )
        existing_status = completed_status(existing_path, 20, job_prefix="existing")
        write_json(existing_batch / "batch_status.json", existing_status)

        reference_frozen = synthetic_hat_r100_frozen()
        reference_path = write_json(
            reference_batch / "frozen_protocol.json", reference_frozen
        )
        reference_status = completed_status(reference_path, 80, job_prefix="base")
        write_json(reference_batch / "batch_status.json", reference_status)

        cls.spec = read_config("one_dimensional_hat_dictionary_size_R100.json")
        cls.spec["source_existing_batch"] = relative(existing_batch)
        cls.spec["source_new_reference_batch"] = relative(reference_batch)
        cls.existing_frozen = existing_frozen
        cls.existing_status = existing_status
        cls.reference_frozen = reference_frozen
        cls.reference_status = reference_status

    def validate(self, spec):
        validate_protocol(
            spec,
            self.existing_frozen,
            self.existing_status,
            self.reference_frozen,
            self.reference_status,
        )

    def test_protocol_reuses_M17_and_fits_240_new_models(self):
        self.validate(self.spec)
        base = effective_base(self.reference_frozen, self.spec)
        self.assertEqual(base["basis"]["sizes"], [9, 33, 65])
        self.assertEqual(base["replication"]["count"], 80)
        self.assertEqual(len(job_plan(self.spec)), 80)
        self.assertEqual(len(base["basis"]["sizes"]) * len(job_plan(self.spec)), 240)

    def test_every_job_uses_the_same_indexed_M17_reference(self):
        for index, job in enumerate(job_plan(self.spec).values()):
            self.assertEqual(job["index"], index)
            self.assertEqual(job["reference_key"], f"base-{index}")

    def test_changes_to_dictionary_or_counts_are_rejected(self):
        for change in ("grid", "new", "count"):
            spec = deepcopy(self.spec)
            if change == "grid":
                spec["dictionary_sizes"] = [9, 17, 65]
            elif change == "new":
                spec["newly_fitted_sizes"] = [9, 17, 33, 65]
            else:
                spec["new_repetitions"] = 79
            with self.assertRaises(ValueError):
                self.validate(spec)

    def test_smoke_plan_is_small(self):
        base = effective_base(self.reference_frozen, self.spec, smoke=True)
        self.assertEqual(base["basis"]["sizes"], [9])
        self.assertEqual(base["replication"]["count"], 2)
        self.assertEqual(len(job_plan(self.spec, smoke=True)), 2)

    def test_missing_values_invalidate_mean(self):
        indices = np.array([[0, 1], [1, 0]])
        complete = extended_statistics([1.0, 3.0], indices)
        self.assertEqual(complete["mean"], 2.0)
        missing = extended_statistics([1.0, None], indices)
        self.assertIsNone(missing["mean"])
        self.assertEqual(missing["available_count"], 1)


if __name__ == "__main__":
    unittest.main()
