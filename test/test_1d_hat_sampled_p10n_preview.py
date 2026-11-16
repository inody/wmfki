import json
from pathlib import Path
import unittest

from scripts.run_1d_hat_sampled_p10n_preview import source_context, validate_spec
from test.test_synthetic_protocol_fixtures import (
    completed_status,
    relative,
    repository_temporary_directory,
    synthetic_dictionary_grid_frozen,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/one_dimensional_hat_sampled_p10n_preview.json"


class OneDimensionalHatSampledP10NPreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(CONFIG.read_text())

    def test_protocol_lock(self):
        validate_spec(self.spec)
        self.assertEqual(self.spec["pair_ratio"], 10)
        self.assertTrue(self.spec["pair_sampling"]["independent_training_validation"])
        self.assertTrue(self.spec["production_artifacts_unchanged"])

    def test_synthetic_source_preserves_dictionary_grid_and_pair_budget(self):
        source = repository_temporary_directory(self) / "synthetic-grid"
        frozen = synthetic_dictionary_grid_frozen(100)
        frozen_path = write_json(source / "frozen_protocol.json", frozen)
        write_json(source / "batch_status.json", completed_status(frozen_path, 100))
        spec = dict(self.spec)
        spec["source_batch"] = relative(source)
        resolved, loaded, base = source_context(spec)
        self.assertEqual(resolved, source)
        self.assertEqual(len(loaded["references"]), 100)
        self.assertEqual(base["simulation"]["N"], 1000)
        self.assertEqual(base["basis"]["sizes"], spec["dictionary_sizes"])
        self.assertEqual(10 * base["simulation"]["N"], 10000)

    def test_training_and_validation_seeds_are_distinct(self):
        for index in (0, 1, 99):
            base = self.spec["pair_seed_start"] + self.spec["pair_seed_stride"] * index
            training = base + self.spec["training_pair_seed_offset"]
            validation = base + self.spec["validation_pair_seed_offset"]
            self.assertNotEqual(training, validation)


if __name__ == "__main__":
    unittest.main()
