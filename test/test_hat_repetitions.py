import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from kernels import make_radial_basis
from scripts.hat_full_pair_assembly import assemble_1d_designs
from scripts.run_1d_hat_repetitions import DEFAULT_CONFIG as REPLICATION_CONFIG, triplet_spec, validate_repetitions
from scripts.summarize_1d_hat_repetitions import METRICS, monotonicity, statistics
from scripts.plot_1d_hat_repetitions_log import plot as plot_log
from scripts.prepare_1d_hat_experiment import (
    DEFAULT_CONFIG, assemble_design, basis_params, prepare, test_family,
)


class CachedHatAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(DEFAULT_CONFIG.read_text())

    def test_nested_spaces_and_constant_extension(self):
        grid = np.r_[np.linspace(0, 4, 511), [4.0, 5.0]]
        for coarse, fine in ((9, 17), (17, 33), (33, 65)):
            coarse_params, fine_params = basis_params(self.spec, coarse), basis_params(self.spec, fine)
            coarse_phi, _ = make_radial_basis(coarse_params)
            fine_phi, _ = make_radial_basis(fine_params)
            np.testing.assert_array_equal(fine_params["knots"][::2], coarse_params["knots"])
            np.testing.assert_allclose(fine_phi(grid) @ coarse_phi(fine_params["knots"]), coarse_phi(grid), atol=1e-14)

    def test_cached_full_pairs_match_original_with_outside_distances_and_repetitions(self):
        rng = np.random.default_rng(582)
        Y = rng.normal(size=(2, 21, 9, 1)) * 3
        Y[:, :, 1] = Y[:, :, 0]  # Case with zero distance also off the diagonal.
        t = np.linspace(0, 0.1, 21)
        space, time = test_family(t)
        candidates = [(f"hat_M{m}", basis_params(self.spec, m)) for m in (9, 17, 33, 65)]
        candidates.append(("baseline", self.spec["baseline"]))
        actual, _ = assemble_1d_designs(Y, t, space, time, candidates)
        for name, params in candidates:
            expected = assemble_design(Y, t, space, time, params, 7)
            np.testing.assert_allclose(actual[name], expected, atol=1e-14, rtol=1e-12)
            self.assertEqual(actual[name].shape, (400, params["M"]))

    def test_cached_preparation_keeps_same_data_response_and_designs(self):
        with tempfile.TemporaryDirectory() as folder:
            a = prepare(self.spec, Path(folder) / "reference", smoke=True)
            b = prepare(self.spec, Path(folder) / "cached", smoke=True, backend="cached_1d")
            self.assertIn("scripts/hat_full_pair_assembly.py", b["source_sha256"])
            for split in a["splits"]:
                left, right = a["splits"][split][0], b["splits"][split][0]
                self.assertEqual(left["observation_sha256"], right["observation_sha256"])
                self.assertEqual(left["response_sha256"], right["response_sha256"])
                for name in left["designs"]:
                    paths = [Path(folder) / backend / split / str(left["seed"]) / f"{name}.npy"
                             for backend in ("reference", "cached")]
                    np.testing.assert_allclose(np.load(paths[0]), np.load(paths[1]), atol=1e-14, rtol=1e-12)


class ReplicationProtocolTests(unittest.TestCase):
    def test_predeclared_twenty_triplets_are_disjoint_and_not_stacked(self):
        spec = json.loads(REPLICATION_CONFIG.read_text())
        validate_repetitions(spec)
        seeds = []
        for index in range(20):
            single = triplet_spec(spec, index)
            self.assertEqual(single["basis"]["sizes"], [9, 17, 33, 65])
            for split in ("train", "validation", "test"):
                self.assertEqual(len(single["seeds"][split]), 1)
                seed = single["seeds"][split][0]
                seeds.extend([seed, seed + 1])
        self.assertEqual(len(set(seeds)), 120)
        spec["seeds"]["test"][19] = spec["seeds"]["train"][0] + 1
        with self.assertRaises(ValueError):
            validate_repetitions(spec)

    def test_failures_are_not_removed_from_all_run_average(self):
        indices = np.random.default_rng(19).integers(0, 3, size=(100, 3))
        result = statistics([1.0, None, 2.0], indices)
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["mean_ci95"])
        self.assertEqual(result["failure_count"], 1)
        self.assertEqual(result["planned_count"], 3)
        valid = statistics([1.0, 2.0, 3.0], indices)
        self.assertEqual(valid["mean"], 2.0)
        self.assertEqual(valid["sample_sd"], 1.0)

    def test_log_view_refuses_to_silently_drop_zero_or_missing_errors(self):
        for value, failure_count in ((0.0, 0), (None, 1)):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "summary.json"
                payload = {"dictionaries": {"hat_M9": {"metrics": {
                    key: {"failure_count": failure_count, "values": [value]}
                    for key in METRICS[:3]}}}}
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    plot_log(path, Path(folder) / "plot")
                self.assertFalse((Path(folder) / "plot").exists())

    def test_nonmonotone_results_are_reported_without_adjustment(self):
        values = np.array([[3.0, 2.0, 4.0], [5.0, 1.0, 3.0]])
        indices = np.random.default_rng(20).integers(0, 2, size=(100, 2))
        result = monotonicity(values, indices)
        self.assertFalse(result["mean_is_nonincreasing"])
        self.assertEqual(result["nonincreasing_repetitions"], 0)
        self.assertEqual(result["adjacent_differences_next_minus_previous"][1]["mean"], 2.0)
        missing = values.copy()
        missing[0, 0] = np.nan
        self.assertIsNone(monotonicity(missing, indices)["mean_is_nonincreasing"])


if __name__ == "__main__":
    unittest.main()
