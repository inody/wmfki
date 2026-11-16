from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_1d_hat_dictionary_grid_OLS_R100 import (
    effective_base,
    fit_ols,
    job_plan,
    validate_protocol,
)
from test.test_synthetic_protocol_fixtures import (
    ROOT,
    completed_status,
    digest,
    read_config,
    relative,
    synthetic_hat_r100_frozen,
    write_json,
)


class TestHatDictionaryGridOLSR100(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        cls.addClassCleanup(cls.temporary.cleanup)
        source = Path(cls.temporary.name) / "synthetic-lasso-grid"
        source_frozen = synthetic_hat_r100_frozen()
        source_frozen["requested_protocol"] = read_config(
            "one_dimensional_hat_dictionary_grid_R100.json"
        )
        source_frozen["references"] = {f"run-{index}": {} for index in range(100)}
        frozen_path = write_json(source / "frozen_protocol.json", source_frozen)
        source_status = completed_status(frozen_path, 100, job_prefix="source")
        write_json(source / "batch_status.json", source_status)
        summary = write_json(source / "summary/summary.json", {"groups": []})
        source_verification = {"status": "passed", "summary_sha256": digest(summary)}
        write_json(source / "verification/verification.json", source_verification)

        cls.spec = read_config("one_dimensional_hat_dictionary_grid_OLS_R100.json")
        cls.spec["source_lasso_grid_batch"] = relative(source)
        cls.source_batch = source
        cls.source_frozen = source_frozen
        cls.source_status = source_status
        cls.source_verification = source_verification

    def validate(self, spec):
        validate_protocol(
            spec,
            self.source_batch,
            self.source_frozen,
            self.source_status,
            self.source_verification,
        )

    def test_protocol_has_ten_sizes_and_one_thousand_models(self):
        self.validate(self.spec)
        self.assertEqual(
            self.spec["dictionary_sizes"], [9, 11, 14, 17, 22, 27, 33, 42, 52, 65]
        )
        self.assertEqual(len(job_plan(self.spec)), 100)
        self.assertEqual(len(self.spec["dictionary_sizes"]) * len(job_plan(self.spec)), 1000)

    def test_estimator_is_raw_minimum_norm_ols(self):
        estimator = self.spec["estimator"]
        self.assertEqual(estimator["implementation"], "numpy.linalg.lstsq")
        self.assertIsNone(estimator["rcond"])
        self.assertFalse(estimator["column_standardization"])
        self.assertFalse(estimator["column_removal"])
        self.assertFalse(estimator["coefficient_thresholding"])
        self.assertFalse(estimator["post_fit_refit"])

    def test_rank_deficient_system_uses_minimum_norm_solution(self):
        A = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        b = np.array([2.0, 4.0, 6.0])
        theta, fit = fit_ols(A, b)
        np.testing.assert_allclose(theta, [1.0, 1.0], rtol=1e-13, atol=1e-13)
        self.assertEqual(fit["rank"], 1)
        self.assertFalse(fit["full_column_rank"])
        self.assertIsNone(fit["condition_number"])
        self.assertEqual(fit["status"], "fitted")

    def test_changes_to_grid_counts_or_estimator_are_rejected(self):
        for field in ("dictionary_sizes", "models", "estimator"):
            spec = deepcopy(self.spec)
            if field == "dictionary_sizes":
                spec[field] = spec[field][:-1]
            elif field == "models":
                spec[field] = 999
            else:
                spec[field]["rcond"] = 1e-8
            with self.assertRaises(ValueError):
                self.validate(spec)

    def test_smoke_uses_two_sizes_and_two_repetitions(self):
        base = effective_base(self.source_frozen, self.spec, smoke=True)
        self.assertEqual(base["basis"]["sizes"], [9, 11])
        self.assertEqual(base["replication"]["count"], 2)
        self.assertEqual(len(job_plan(self.spec, smoke=True)), 2)


if __name__ == "__main__":
    unittest.main()
