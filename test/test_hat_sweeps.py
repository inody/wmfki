from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.prepare_1d_hat_experiment import array_hash
from scripts.run_1d_hat_lasso import select_lasso
from scripts.run_1d_hat_sweeps import (
    DEFAULT_CONFIG, ROOT, accuracy_job, cases, coverage, effective_protocol, fixed_penalty_path,
    job_plan, read_json, validate_protocol,
)
from scripts.summarize_1d_hat_sweeps import aggregate_models


class TestHatSweeps(unittest.TestCase):
    def setUp(self):
        self.spec = read_json(DEFAULT_CONFIG)
        self.base = read_json(ROOT / self.spec["base_config"])

    def test_common_baseline_and_only_one_changed_parameter(self):
        validate_protocol(self.spec, self.base)
        points = cases(self.spec, self.base)
        self.assertEqual(sum(point["id"] == "base" for point in points), 4)
        for point in points:
            changed = [key for key in point["simulation"] if point["simulation"][key] != self.base["simulation"][key]]
            self.assertEqual(changed, [] if point["id"] == "base" else [point["parameter"]])
        jobs = job_plan(self.spec, self.base)
        self.assertEqual(len(jobs), 580)
        self.assertEqual(sum(job["stage"] == "accuracy" for job in jobs.values()), 380)
        self.assertEqual(sum(job["stage"] == "pairs" for job in jobs.values()), 120)
        self.assertEqual(sum(job["stage"] == "runtime" for job in jobs.values()), 60)

    def test_missing_baseline_and_changed_settings_are_rejected(self):
        changed = deepcopy(self.spec)
        changed["sweeps"]["N"].remove(1000)
        with self.assertRaises(ValueError):
            validate_protocol(changed, self.base)
        changed_base = deepcopy(self.base)
        changed_base["simulation"]["horizon"] = 3.
        with self.assertRaises(ValueError):
            validate_protocol(self.spec, changed_base)

    def test_rng_families_do_not_collide(self):
        initial = {value for values in self.base["seeds"].values() for value in values}
        brownian = {value + 1 for value in initial}
        observation = {value + self.spec["observation_seed_offset"] for value in initial}
        pairs = {self.spec["pair_seed_base"] + 1000 * i + 10 * p for i in range(20) for p in range(6)}
        runtime = {self.spec["runtime_seed_base"] + 1000 * n + 10 * i + offset
                   for n in range(6) for i in range(10) for offset in range(3)}
        families = [initial, brownian, observation, pairs, runtime]
        self.assertEqual(sum(map(len, families)), len(set.union(*families)))

    def test_fixed_path_matches_same_full_path_without_reselection(self):
        rng = np.random.default_rng(4321)
        A = rng.normal(size=(50, 9))
        b = A @ rng.normal(size=9) + rng.normal(size=50) * .1
        settings = {**self.base["estimator"], "alpha_count": 8}
        selected, coefficients = select_lasso(A, b, A, b, settings)
        fixed, path = fixed_penalty_path(A, b, selected, settings)
        self.assertEqual(fixed["endpoint"]["index"], selected["selected"]["index"])
        self.assertEqual(fixed["endpoint"]["alpha_sk"], selected["selected"]["alpha_sk"])
        np.testing.assert_array_equal(path, coefficients[:len(path)])
        np.testing.assert_array_equal(fixed["theta"], selected["theta"])

    def test_nonconverged_finite_outputs_not_dropped_and_missing_mean_invalidated(self):
        bootstrap = np.array([[0, 1], [1, 0]])
        rows = [{"results": {"hat_M9": {"endpoint": {"converged": False},
                 "metrics": {"coefficient_difference_sq": 4.}}}},
                {"results": {"hat_M9": {"endpoint": {"converged": True},
                 "metrics": {"coefficient_difference_sq": 2.}}}}]
        actual = aggregate_models(rows, ["hat_M9"], bootstrap, pair=True)["hat_M9"]
        self.assertEqual(actual["metrics"]["coefficient_difference_sq"]["mean"], 3.)
        self.assertEqual(actual["unconverged_endpoints"], 1)
        self.assertIsNone(actual["metrics"]["trajectory_error_rel"]["mean"])
        rows[1] = {}
        actual = aggregate_models(rows, ["hat_M9"], bootstrap, pair=True)["hat_M9"]
        self.assertIsNone(actual["metrics"]["coefficient_difference_sq"]["mean"])
        self.assertEqual(actual["metrics"]["coefficient_difference_sq"]["available_count"], 1)

    def test_coverage_counts_exact_distances_at_boundary(self):
        x = np.array([0., 0., .5, 4., np.nextafter(4., np.inf), 7.])
        X = np.repeat(x[None, None, :, None], 3, axis=1)
        observed = coverage(X, np.array([0., .01, .02]))
        pairs = np.abs(x[:, None] - x[None, :])[~np.eye(len(x), dtype=bool)]
        self.assertEqual(observed["minimum"], float(pairs.min()))
        self.assertEqual(observed["maximum"], float(pairs.max()))
        self.assertAlmostEqual(observed["outside_basis_interval_fraction"], float(np.mean(pairs > 4)), places=14)

    def test_test_arrays_only_loaded_after_all_models_frozen(self):
        spec, base = effective_protocol(self.spec, self.base, True)
        with tempfile.TemporaryDirectory() as temp:
            batch = Path(temp)
            folder = batch / "job"
            folder.mkdir()
            from scripts.run_1d_hat_sweeps import make_split
            def checked(*args, **kwargs):
                split = args[3]
                if split == "test":
                    self.assertTrue((folder / "frozen_selections.json").is_file())
                    self.assertEqual(len(read_json(folder / "frozen_selections.json")), 4)
                return make_split(*args, **kwargs)
            with patch("scripts.run_1d_hat_sweeps.make_split", side_effect=checked):
                report = accuracy_job(batch, folder, dict(index=0, simulation=base["simulation"]), spec, base, None)
            self.assertEqual(len(report["results"]), 4)
            for split in ("train", "validation", "test"):
                saved = report["splits"][split]
                with np.load(batch / saved["data_file"]) as data:
                    self.assertEqual(saved["Y_sha256"], array_hash(data["X"]))


if __name__ == "__main__":
    unittest.main()
