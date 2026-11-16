import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.manuscript_plot_style import apply_manuscript_style
from scripts.regenerate_l4dc_figures import (
    CURRENT_RENDERING_POLICY,
    EXPECTED_OUTPUTS,
    command_plan,
    configured_directories,
    staged_outputs,
    validate_config,
    verify_checked_in,
    verify_generated,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/l4dc_current_figures.json"


class RegenerateL4dcFiguresTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(CONFIG.read_text())

    def test_shared_plot_style_matches_manuscript_fonts(self):
        import matplotlib.pyplot as plt

        apply_manuscript_style()
        self.assertTrue(plt.rcParams["text.usetex"])
        self.assertEqual(plt.rcParams["font.family"], ["serif"])
        preamble = plt.rcParams["text.latex.preamble"]
        self.assertIn(r"\usepackage[T1]{fontenc}", preamble)
        self.assertIn(r"\usepackage{times}", preamble)

    def test_protocol_declares_exact_current_figure_set(self):
        validate_config(self.spec)
        self.assertEqual(
            set(self.spec["expected_figure_sha256"]), EXPECTED_OUTPUTS
        )
        self.assertEqual(len(EXPECTED_OUTPUTS), 8)

    def test_directory_resolution_never_requires_manuscript_sources(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temporary = Path(temporary)
            changed = json.loads(json.dumps(self.spec))
            changed.pop("manuscript_directory", None)
            changed["figure_directory"] = str(
                (temporary / "reference/figures").relative_to(ROOT)
            )
            changed["build_directory"] = str(
                (temporary / "outputs/figure-build").relative_to(ROOT)
            )
            validate_config(changed)
            figure_directory, build_directory = configured_directories(changed)
            self.assertEqual(figure_directory, temporary / "reference/figures")
            self.assertEqual(build_directory, temporary / "outputs/figure-build")

            changed.pop("build_directory")
            _, fallback_build = configured_directories(changed)
            self.assertEqual(fallback_build, temporary / "reference/build")

    def test_current_rendering_policy_pins_current_comparisons(self):
        self.assertEqual(self.spec["rendering_policy"], CURRENT_RENDERING_POLICY)
        sources = self.spec["sources"]
        self.assertIn("hat_noisy_comparison", sources)
        self.assertIn("regularized_baselines", sources)
        self.assertEqual(sources["redundant6"], sources["redundant6_time_step"])
        self.assertEqual(
            sources["runtime_scaling"],
            str(Path(sources["redundant6"]) / "summary_f.json"),
        )

    def test_current_protocol_rejects_stochastic_panel_d_source(self):
        changed = json.loads(json.dumps(self.spec))
        changed["sources"]["redundant6_time_step"] = "outputs/stochastic-time-step"
        with self.assertRaisesRegex(ValueError, "deterministic production-bundle"):
            validate_config(changed)

    def test_command_plan_covers_five_generation_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = command_plan(self.spec, Path(temporary))
        self.assertEqual(
            [label for label, _ in plan],
            [
                "hat", "redundant6", "nominal_overlays",
                "predator_robustness", "localized10d",
            ],
        )
        flattened = "\n".join(" ".join(command) for _, command in plan)
        for source in self.spec["sources"].values():
            self.assertIn(str(ROOT / source), flattened)

    def test_staged_output_mapping_and_hash_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            outputs = staged_outputs(stage)
            expected = {}
            for index, (name, path) in enumerate(outputs.items()):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"figure-{index}".encode())
                expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            actual = verify_generated(outputs, expected)
        self.assertEqual(actual, expected)

    def test_check_mode_verifies_checked_in_figure_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            expected = {}
            for index, name in enumerate(sorted(EXPECTED_OUTPUTS)):
                path = destination / name
                path.write_bytes(f"checked-in-{index}".encode())
                expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(verify_checked_in(destination, expected), expected)
            (destination / sorted(EXPECTED_OUTPUTS)[0]).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "checked-in figure hash mismatch"):
                verify_checked_in(destination, expected)


if __name__ == "__main__":
    unittest.main()
