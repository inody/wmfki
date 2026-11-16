import json
from pathlib import Path
import tempfile
import unittest

from scripts.regenerate_l4dc_figures import validate_config as validate_figure_config
from scripts.reproduce_l4dc import ROOT, plan, relative
from scripts.run_localized10d_observation_noise import (
    validate_protocol as validate_localized10d_observation_protocol,
)
from scripts.run_1d_hat_dictionary_grid_from_scratch import (
    effective_base,
    seed_triplet,
    validate_protocol,
)
from scripts.build_cross_example_metric_table import format_three_significant_figures


class L4DCReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((
            ROOT / "configs/experiments/one_dimensional_hat_dictionary_grid_from_scratch_R100.json"
        ).read_text())
        cls.base = json.loads((ROOT / cls.protocol["base_config"]).read_text())

    def test_from_scratch_hat_protocol_is_self_contained(self):
        validate_protocol(self.protocol, self.base)
        self.assertEqual(self.protocol["reused_sizes"], [])
        self.assertEqual(self.protocol["new_models"], 1000)

    def test_seed_schedule_preserves_first_twenty_and_declares_new_eighty(self):
        for index in (0, 19):
            expected = {
                split: self.base["seeds"][split][index]
                for split in ("train", "validation", "test")
            }
            self.assertEqual(seed_triplet(self.base, index), expected)
        self.assertEqual(
            seed_triplet(self.base, 20),
            {"train": 8_000_000, "validation": 8_000_002, "test": 8_000_004},
        )
        self.assertEqual(
            seed_triplet(self.base, 99),
            {"train": 8_000_790, "validation": 8_000_792, "test": 8_000_794},
        )

    def test_smoke_reduces_repetitions_without_changing_requested_protocol(self):
        effective = effective_base(self.base, self.protocol, smoke=True)
        self.assertEqual(effective["replication"]["count"], 2)
        self.assertEqual(effective["basis"]["sizes"], [9, 11])
        self.assertEqual(self.protocol["repetitions"], 100)

    def test_plan_explicitly_excludes_historical_outputs(self):
        result = plan("production", 3)
        self.assertFalse(result["historical_outputs_required"])
        self.assertFalse(result["current_manuscript_figure_protocol"])
        self.assertEqual(result["current_figure_commands"]["check"], "make figures-check")
        self.assertIn("runtime_scaling", result["stages"])

    def test_fresh_localized10d_protocol_does_not_need_clean_archived_records(self):
        requested = json.loads((
            ROOT / "configs/experiments/localized10d_observation_noise_001.json"
        ).read_text())
        base = json.loads((ROOT / requested["base_config"]).read_text())
        requested["clean_reference_records"] = None
        requested["clean_reference_sha256"] = None
        validate_localized10d_observation_protocol(requested, base)

    def test_fresh_figure_protocol_accepts_unpinned_generated_hashes(self):
        names = {
            "one_dimensional_hat_practical.pdf", "validation_sweeps.pdf",
            "predator_prey_profiles.pdf", "predator_prey_dynamics.pdf",
            "predator_prey_robustness_sweeps.pdf", "localized10d_recovery.pdf",
            "localized10d_coupled_dynamics.pdf", "localized10d_robustness_sweeps.pdf",
        }
        sources = {
            key: "placeholder" for key in (
                "hat", "hat_noisy_comparison", "redundant6",
                "redundant6_time_step", "runtime_scaling",
                "qanr_strong_lasso_noise", "predator_sweeps",
                "predator_strong_lasso_noise", "localized10d",
                "regularized_baselines",
            )
        }
        validate_figure_config({
            "task": "reproduce_l4dc_figures_from_fresh_runs",
            "protocol_version": 2,
            "sources": sources,
            "expected_figure_sha256": {name: None for name in names},
        })

    def test_reproduction_output_must_be_inside_repository(self):
        self.assertEqual(relative(ROOT / "outputs/reproduction/example"),
                         "outputs/reproduction/example")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                relative(Path(directory))

    def test_table_parity_uses_the_numerical_three_significant_figure_policy(self):
        self.assertEqual(format_three_significant_figures(2.467136415726486), "2.47")
        self.assertEqual(format_three_significant_figures(0.044411183119162374), "0.0444")
        self.assertEqual(format_three_significant_figures(1743.344170170812), "1740")


if __name__ == "__main__":
    unittest.main()
