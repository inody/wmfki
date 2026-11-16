import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_localized10d_sampled_pair_p10n_preview import (
    effective_protocol,
    generate_error_preview,
    pair_budget,
    pair_sampler_contract_audit,
    source_protocol_preservation_audit,
)
from scripts.run_morse10d_sweeps import repetition_plan
from test.test_synthetic_protocol_fixtures import (
    read_config,
    synthetic_localized_frozen,
)


class Localized10DSampledPairP10NPreviewTests(unittest.TestCase):
    def setUp(self):
        self.requested = read_config("localized10d_sampled_pair_p10n_preview.json")
        self.source_frozen = synthetic_localized_frozen(0.01)

    def test_synthetic_source_protocol_is_two_repetition_preview(self):
        spec = effective_protocol(self.requested, self.source_frozen)
        self.assertTrue(spec["preview_only"])
        self.assertEqual(spec["evaluation"]["scientific_repetitions"], 2)
        self.assertEqual(spec["model"]["particles"], 500)
        self.assertEqual(spec["observation_noise"], 0.01)
        preservation = source_protocol_preservation_audit(self.source_frozen, spec)
        self.assertTrue(preservation["passed"], preservation)

    def test_standard_condition_pair_budget_is_ten_n(self):
        spec = effective_protocol(self.requested, self.source_frozen)
        self.assertEqual(pair_budget(spec, 100), 1000)
        self.assertEqual(pair_budget(spec, 200), 2000)
        self.assertEqual(pair_budget(spec, 350), 3500)
        self.assertEqual(pair_budget(spec, 500), 5000)
        self.assertEqual(spec["pair_count_penalty_policy"]["selection_pairs"], 5000)

    def test_training_and_validation_pair_seeds_are_independent(self):
        spec = effective_protocol(self.requested, self.source_frozen)
        for plan in repetition_plan(spec):
            self.assertNotEqual(plan["training_pair_seed"], plan["validation_pair_seed"])

    def test_pair_sampler_contract(self):
        audit = pair_sampler_contract_audit()
        self.assertTrue(audit["passed"], audit)
        self.assertTrue(all(audit["checks"].values()))

    def test_figure_only_mode_reads_synthetic_records_and_writes_png_pdf(self):
        records = []
        sweep_values = {
            "process_sigma": [0.05, 0.1],
            "particle_number": [100, 500],
            "sampled_pairs": [3000, 5000],
        }
        for repetition in range(2):
            for method_index, method in enumerate(("lasso", "ols")):
                for sweep, values in sweep_values.items():
                    for value_index, value in enumerate(values):
                        error = 0.05 + 0.01 * repetition + 0.02 * method_index \
                            + 0.005 * value_index
                        records.append({
                            "repetition": repetition,
                            "method": method,
                            "sweep": sweep,
                            "sweep_value": value,
                            "coefficient_error_relative": error,
                            "latent_drift_relative_error": error * 0.8,
                            "profile_relative_error": error * 1.2,
                            "observation_noise": 0.01,
                            "converged": True,
                            "selected_at_grid_boundary": False,
                            "rank_deficient": False,
                            "numerical_divergence": False,
                        })
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "records.json").write_text(json.dumps(records))
            report = generate_error_preview(run_dir)
            preview = run_dir / "preview"
            self.assertTrue(report["records_only"])
            self.assertFalse(report["numerical_recomputation"])
            self.assertTrue((preview / "localized10d_sampled_pair_p10n_error_preview.png").is_file())
            self.assertTrue((preview / "localized10d_sampled_pair_p10n_error_preview.pdf").is_file())
            self.assertTrue((preview / "figure_report.json").is_file())
            with self.assertRaises(FileExistsError):
                generate_error_preview(run_dir)


if __name__ == "__main__":
    unittest.main()
