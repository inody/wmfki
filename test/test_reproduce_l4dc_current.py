import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from scripts.manuscript_plot_style import apply_manuscript_style
from scripts.regenerate_l4dc_figures import validate_config as validate_figure_config
from scripts.run_localized10d_sampled_pair_production import source_files
from scripts.run_1d_hat_sampled_p50n_R100 import (
    load_reused_lasso_record,
    reset_postprocessing_outputs,
)
from scripts.run_predator_prey_sampled_pair_production import (
    validate_production as validate_predator_production,
)
from scripts.reproduce_l4dc_current import (
    CURRENT_FIGURES,
    CurrentReproducer,
    NUMERICAL_TABLE_REFERENCE,
    ROOT,
    STAGES,
    audit,
    build_cross_example_config,
    build_figure_config,
    build_hat_sampled_config,
    build_localized_sampled_config,
    build_metric_table_config,
    build_predator_application_config,
    build_predator_sampled_config,
    build_predator_strong_config,
    build_qanr_sampled_config,
    current_stage_paths,
    digest,
    plan,
    validate_stage_graph,
)


class CurrentL4DCReproductionPlanTests(unittest.TestCase):
    def test_current_executor_initializes_an_isolated_state(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "current-run"
            reproducer = CurrentReproducer(output, workers=2, resume=False)
            self.assertTrue(reproducer.state_path.is_file())
            self.assertEqual(reproducer.state["status"], "running")
            self.assertEqual(
                reproducer.state["source_contract"]["table1_reference"],
                {
                    "path": "reference/table1.json",
                    "sha256": digest(NUMERICAL_TABLE_REFERENCE),
                },
            )
            paths = current_stage_paths(output)
            self.assertEqual(paths["hat_full_lasso"], output / "data/hat_full_lasso")
            self.assertEqual(
                paths["localized_sampled_p50"],
                output / "data/localized_sampled_p50",
            )

    def test_resume_rejects_a_changed_numerical_source_contract(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "current-run"
            reproducer = CurrentReproducer(output, workers=2, resume=False)
            state = json.loads(reproducer.state_path.read_text())
            state["source_contract"]["table1_reference"]["sha256"] = "0" * 64
            reproducer.state_path.write_text(json.dumps(state) + "\n")
            with self.assertRaisesRegex(ValueError, "source contract mismatch"):
                CurrentReproducer(output, workers=2, resume=True)

    def test_stage_graph_is_topologically_ordered(self):
        names = validate_stage_graph()
        self.assertEqual(names[0], "hat_full_lasso")
        self.assertEqual(names[-1], "release_checks")
        self.assertEqual(len(names), len(STAGES))
        self.assertEqual(len(names), 25)

    def test_smoke_plot_style_can_disable_external_tex(self):
        previous = os.environ.get("WMFKI_PLOT_USETEX")
        try:
            os.environ["WMFKI_PLOT_USETEX"] = "0"
            apply_manuscript_style()
            self.assertFalse(plt.rcParams["text.usetex"])
            self.assertEqual(plt.rcParams["text.latex.preamble"], "")
        finally:
            if previous is None:
                os.environ.pop("WMFKI_PLOT_USETEX", None)
            else:
                os.environ["WMFKI_PLOT_USETEX"] = previous
            apply_manuscript_style()

    def test_makefile_exposes_current_plan_smoke_and_data_targets(self):
        makefile = (ROOT / "Makefile").read_text()
        self.assertIn("reproduce-current-plan:", makefile)
        self.assertIn("reproduce-current-smoke:", makefile)
        self.assertIn("reproduce-current-data:", makefile)
        self.assertIn("reproduce-current-artifacts:", makefile)
        self.assertIn("reproduce-current-all:", makefile)

    def test_plan_covers_current_eight_figures_and_three_modes(self):
        result = plan(workers=3)
        self.assertEqual(set(result["figures"]), set(CURRENT_FIGURES))
        self.assertEqual(
            set(result["execution_modes"]), {"frozen_figures", "smoke", "full"}
        )
        self.assertFalse(result["historical_outputs_required_by_target_workflow"])
        self.assertIn("eight reference PDFs", result["execution_modes"]["frozen_figures"])
        self.assertNotIn("seven", result["execution_modes"]["frozen_figures"])

    def test_plan_contains_only_numerical_release_metadata(self):
        result = plan()
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("protected", result)
        self.assertNotIn("manuscript", rendered)
        self.assertNotIn("github_repository_visibility", rendered)
        self.assertIn("numerical", result["scope"])

    def test_audit_confirms_archived_template_paths_are_rebound(self):
        result = audit()
        self.assertTrue(result["source_independent_full_run_ready"])
        self.assertEqual(result["blockers"], [])
        archived = [
            item for item in result["archived_inputs"]
            if item.get("is_archived_output_path")
        ]
        self.assertTrue(archived)
        self.assertTrue(all(item["rebound_by_current_executor"] for item in archived))
        self.assertEqual(set(result["working_trees"]), {"root"})

    def test_all_other_declared_stage_files_exist(self):
        result = audit()
        missing = [
            item for item in result["stage_files"]
            if not item["exists"]
        ]
        self.assertEqual(missing, [])

    def test_sampled_hat_resume_resets_only_derived_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            preserved = output / "runs/ratio-50-run-0/report.json"
            preserved.parent.mkdir(parents=True)
            preserved.write_text("{}\n")
            for name in ("figure1_candidate", "comparison_figure", "verification"):
                path = output / name / "partial.txt"
                path.parent.mkdir()
                path.write_text("partial\n")
            reset_postprocessing_outputs(output)
            self.assertTrue(preserved.is_file())
            self.assertFalse((output / "figure1_candidate").exists())
            self.assertFalse((output / "comparison_figure").exists())
            self.assertFalse((output / "verification").exists())

    def test_fresh_hat_grid_needs_no_archived_reused_model_records(self):
        frozen = {
            "source_mode": "generated_in_this_run",
            "imported_reused_models": {},
        }
        self.assertEqual(
            load_reused_lasso_record(frozen, "run-0", ROOT),
            {"results": {}, "source_model_records": []},
        )
        with self.assertRaises(KeyError):
            load_reused_lasso_record(
                {"source_mode": "archived", "imported_reused_models": {}},
                "run-0", ROOT,
            )

    def test_hat_sampled_config_can_bind_fresh_stage_outputs(self):
        template = json.loads((
            ROOT / "configs/experiments/one_dimensional_hat_sampled_p50n_R100.json"
        ).read_text())
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            lasso = directory / "lasso"
            ols = directory / "ols"
            for method, batch in (("lasso", lasso), ("ols", ols)):
                for relative in template[f"source_{method}_hashes"]:
                    path = batch / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"{method}:{relative}\n")
            output = directory / "configs/hat_sampled.json"
            result = build_hat_sampled_config(lasso, ols, output)
            saved = json.loads(output.read_text())
            self.assertEqual(saved, result)
            self.assertFalse(saved["source_lasso_batch"].startswith("outputs/validation"))
            self.assertEqual(
                saved["source_lasso_hashes"]["frozen_protocol.json"],
                digest(lasso / "frozen_protocol.json"),
            )
            self.assertEqual(saved["source_mode"], "fresh_full_pair_stage_outputs")
            self.assertEqual(saved["current_artifact_audit"], {})

    def test_qanr_sampled_config_can_bind_fresh_stage_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            full = directory / "full"
            time_step = directory / "time-step"
            for batch, names in (
                (full, ("summary.json", "observation_lasso.csv", "stochastic_lasso.csv")),
                (time_step, ("summary.json", "records.json")),
            ):
                batch.mkdir()
                for name in names:
                    (batch / name).write_text("{}\n")
            output = directory / "configs/qanr_sampled.json"
            result = build_qanr_sampled_config(full, time_step, output)
            saved = json.loads(output.read_text())
            self.assertEqual(saved, result)
            self.assertEqual(saved["matched_full_pair_batch"], str(full.relative_to(ROOT)))
            self.assertEqual(
                saved["matched_full_pair_time_step_batch"],
                str(time_step.relative_to(ROOT)),
            )
            self.assertNotIn(
                "outputs/validation/qanr_redundant6_lasso_R100/sweeps-20261022T052958Z",
                saved["protected_paths"],
            )
            self.assertEqual(saved["source_mode"], "fresh_full_pair_stage_outputs")

    def test_predator_sampled_config_can_bind_fresh_full_pair_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            full = directory / "predator-full"
            full.mkdir()
            for name in ("manifest.json", "records.json", "requested_protocol.json"):
                (full / name).write_text("{}\n")
            output = directory / "configs/predator-sampled.json"
            result = build_predator_sampled_config(full, output)
            saved = json.loads(output.read_text())
            self.assertEqual(saved, result)
            self.assertEqual(saved["reference_frozen_run"], str(full.relative_to(ROOT)))
            self.assertEqual(saved["source_mode"], "fresh_full_pair_stage_outputs")
            self.assertFalse(any(
                key.startswith("source_confirmation_") for key in saved
            ))
            validate_predator_production(saved)

    def test_predator_strong_config_can_bind_fresh_sampled_records(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            sampled = directory / "predator-sampled"
            sampled.mkdir()
            (sampled / "records.json").write_text("[]\n")
            output = directory / "configs/predator-strong.json"
            result = build_predator_strong_config(sampled, output)
            saved = json.loads(output.read_text())
            self.assertEqual(saved, result)
            self.assertEqual(
                saved["weak_sweep_records"],
                str((sampled / "records.json").relative_to(ROOT)),
            )
            self.assertEqual(
                saved["weak_sweep_records_sha256"],
                digest(sampled / "records.json"),
            )

    def test_integration_configs_bind_fresh_stage_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            full = directory / "predator-full"
            full.mkdir()
            for name in ("requested_protocol.json", "records.json"):
                (full / name).write_text("[]\n")
            application_config = directory / "configs/predator-application.json"
            application = build_predator_application_config(
                full, application_config
            )
            self.assertEqual(
                application["sweep_records"],
                str((full / "records.json").relative_to(ROOT)),
            )

            qanr = directory / "qanr"
            predator_application = directory / "predator-application"
            predator_strong = directory / "predator-strong"
            localized_nominal = directory / "localized-nominal"
            source_files_to_create = (
                qanr / "observation_lasso.csv",
                predator_application / "records.json",
                predator_strong / "records.json",
                localized_nominal / "records.json",
            )
            for path in source_files_to_create:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[]\n")
            cross_config = directory / "configs/cross.json"
            cross = build_cross_example_config(
                qanr, predator_application, predator_strong,
                localized_nominal, cross_config,
            )
            self.assertEqual(cross["source_mode"], "fresh_current_stage_outputs")
            self.assertEqual(
                cross["predator_prey"]["strong_source_sha256"],
                digest(predator_strong / "records.json"),
            )

            regularized = directory / "regularized"
            regularized.mkdir()
            (regularized / "summary.json").write_text("{\"groups\": []}\n")
            table_config = directory / "configs/table.json"
            table = build_metric_table_config(regularized, table_config)
            self.assertEqual(table["protocol_version"], 3)
            self.assertEqual(
                table["regularized_summary"],
                str((regularized / "summary.json").relative_to(ROOT)),
            )
            self.assertEqual(
                table["regularized_summary_sha256"],
                digest(regularized / "summary.json"),
            )

    def test_fresh_figure_config_covers_exactly_eight_figures(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            paths = current_stage_paths(output)
            required = (
                paths["hat_sampled_p50"] / "frozen_protocol.json",
                paths["hat_sampled_p50"] / "records.json",
                paths["hat_sampled_p50"] / "summary.json",
                paths["hat_sampled_p50"] / "figure1_candidate/metadata.json",
                paths["hat_sampled_p50"] / "figure1_candidate/representative_rollout.npz",
                paths["hat_noisy_comparison"] / "records.json",
                paths["hat_noisy_comparison"] / "summary.json",
                paths["qanr_sampled_p50"] / "requested_protocol.json",
                paths["qanr_sampled_p50"] / "summary_a_to_d.json",
                paths["qanr_sampled_p50"] / "summary_e.json",
                paths["qanr_sampled_p50"] / "summary_f.json",
                paths["qanr_strong_lasso_noise"] / "summary.json",
                paths["predator_sampled_p50"] / "effective_protocol.json",
                paths["predator_sampled_p50"] / "records.json",
                paths["predator_sampled_p50"] / "nominal_application/records.json",
                paths["predator_sampled_p50"] / "nominal_application/display_trajectory.npz",
                paths["predator_sampled_p50"] / "nominal_application/trajectory_report.json",
                paths["predator_strong_lasso_noise"] / "records.json",
                paths["localized_sampled_p50"] / "frozen_protocol.json",
                paths["localized_sampled_p50"] / "records.json",
                paths["localized_sampled_p50"] / "source_manifest.json",
                paths["regularized_baselines"] / "records.json",
                paths["regularized_baselines"] / "summary.json",
            )
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
            for key in (
                "hat_noisy_comparison", "qanr_strong_lasso_noise",
                "predator_strong_lasso_noise", "regularized_baselines",
            ):
                paths[key].mkdir(parents=True, exist_ok=True)
            destination = output / "configs/figures.json"
            spec = build_figure_config(output, paths, destination)
            validate_figure_config(spec)
            self.assertEqual(
                set(spec["expected_figure_sha256"]), set(CURRENT_FIGURES)
            )
            self.assertTrue(all(
                value is None
                for value in spec["expected_figure_sha256"].values()
            ))
            self.assertEqual(
                spec["figure_directory"],
                str((output / "figures").relative_to(ROOT)),
            )
            self.assertEqual(
                spec["build_directory"],
                str((output / "figure-build").relative_to(ROOT)),
            )
            self.assertNotIn("manuscript_directory", spec)
            self.assertEqual(
                spec["sources"]["regularized_baselines"],
                str(paths["regularized_baselines"].relative_to(ROOT)),
            )

    def test_integration_uses_numerical_table_verification_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "current-run"
            reproducer = CurrentReproducer(output, workers=2, resume=False)
            commands = {}

            def fake_command(label, arguments, expected, **_kwargs):
                commands[label] = arguments
                for path in expected:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}\n")

            reproducer.command = fake_command
            no_op = lambda *_args, **_kwargs: {}
            with patch.multiple(
                "scripts.reproduce_l4dc_current",
                build_cross_example_config=no_op,
                build_regularized_config=no_op,
                build_metric_table_config=no_op,
                build_figure_config=no_op,
            ):
                result = reproducer.run_integration_artifacts()

            verification = commands["metric_table_verify"]
            self.assertIn("scripts/verify_l4dc_numerical_values.py", verification)
            self.assertIn("--summary", verification)
            self.assertIn("reference/table1.json", verification)
            self.assertNotIn("--manuscript", verification)
            self.assertFalse(any("latex/" in token for token in verification))
            self.assertIn(
                "metric_table_numerical_verification", result["artifacts"]
            )
            self.assertIn("numerical_table_reference", result["artifacts"])
            self.assertNotIn("metric_table", result["artifacts"])

    def test_localized_sampled_config_can_bind_fresh_robustness_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            robustness = directory / "localized-robustness"
            for name in (
                "frozen_protocol.json",
                "requested_protocol.json",
                "source_manifest.json",
                "source/scripts/run_localized10d_observation_noise.py",
            ):
                path = robustness / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
            output = directory / "configs/localized-sampled.json"
            result = build_localized_sampled_config(robustness, output)
            saved = json.loads(output.read_text())
            self.assertEqual(saved, result)
            self.assertEqual(saved["source_mode"], "fresh_robustness_stage_output")
            self.assertEqual(
                saved["authoritative_frozen_protocol"],
                str((robustness / "frozen_protocol.json").relative_to(ROOT)),
            )
            expected_sources = source_files(saved)
            self.assertIn(saved["frozen_observation_runner"], expected_sources)
            self.assertNotIn(
                "outputs/validation/localized10d_noisy_robustness/"
                "sweeps-20261023T123949Z/frozen_protocol.json",
                expected_sources,
            )


if __name__ == "__main__":
    unittest.main()
