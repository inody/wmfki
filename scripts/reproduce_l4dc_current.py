"""Plan and audit the source-independent current L4DC numerical workflow.

This module implements the clean source-to-output workflow for the current
L4DC numerical results. The principal commands are::

    .venv/bin/python scripts/reproduce_l4dc_current.py plan
    .venv/bin/python scripts/reproduce_l4dc_current.py smoke
    .venv/bin/python scripts/reproduce_l4dc_current.py all

Fresh dependency-bound configs are generated inside each run so archived
``outputs/`` paths in the original production configs are never consumed by
the source-independent workflow.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "outputs/reproduction-current"
CURRENT_FIGURES = (
    "one_dimensional_hat_practical.pdf",
    "validation_sweeps.pdf",
    "predator_prey_profiles.pdf",
    "predator_prey_dynamics.pdf",
    "predator_prey_robustness_sweeps.pdf",
    "localized10d_recovery.pdf",
    "localized10d_coupled_dynamics.pdf",
    "localized10d_robustness_sweeps.pdf",
)
NUMERICAL_TABLE_REFERENCE = ROOT / "reference/table1.json"


@dataclass(frozen=True)
class Stage:
    name: str
    family: str
    dependencies: tuple[str, ...]
    runner: str
    config: str | None
    readiness: str
    purpose: str


STAGES = (
    Stage(
        "hat_full_lasso", "one_dimensional", (),
        "scripts/run_1d_hat_dictionary_grid_from_scratch.py",
        "configs/experiments/one_dimensional_hat_dictionary_grid_from_scratch_R100.json",
        "ready", "Generate declared trajectories and the full-pair LASSO grid.",
    ),
    Stage(
        "hat_full_ols", "one_dimensional", ("hat_full_lasso",),
        "scripts/run_1d_hat_dictionary_grid_OLS_R100.py",
        "configs/experiments/one_dimensional_hat_dictionary_grid_OLS_R100.json",
        "ready", "Fit the matching full-pair OLS grid.",
    ),
    Stage(
        "hat_sampled_p50", "one_dimensional", ("hat_full_lasso", "hat_full_ols"),
        "scripts/run_1d_hat_sampled_p50n_R100.py",
        "configs/experiments/one_dimensional_hat_sampled_p50n_R100.json",
        "requires_dependency_injection",
        "Run sampled-pair LASSO/OLS at P=50N and build Figure 1 records.",
    ),
    Stage(
        "hat_verify", "one_dimensional", ("hat_sampled_p50",),
        "scripts/finalize_1d_hat_sampled_p50n_R100.py", None,
        "requires_output_generalization",
        "Verify the fresh sampled-pair output and claim reports.",
    ),
    Stage(
        "hat_noisy_comparison", "one_dimensional", ("hat_verify",),
        "scripts/run_fig1c_noisy_three_method.py",
        "configs/experiments/fig1c_noisy_three_method.json",
        "requires_dependency_injection",
        "Run the noisy three-method comparison used by Figure 1.",
    ),
    Stage(
        "qanr_full", "qanr", (),
        "scripts/run_qanr_matched_lasso_sweeps.py",
        "configs/experiments/qanr_redundant6_lasso_sweeps_R100.json",
        "ready", "Run the full-pair six-atom reference sweeps.",
    ),
    Stage(
        "qanr_time_step", "qanr", (),
        "scripts/run_qanr_time_step_repetitions.py",
        "configs/experiments/qanr_redundant6_time_step_paired_R100.json",
        "ready", "Run the paired full-pair time-step reference.",
    ),
    Stage(
        "qanr_sampled_p50", "qanr", ("qanr_full", "qanr_time_step"),
        "scripts/run_qanr_sampled_p50n_production_R100.py",
        "configs/experiments/qanr_redundant6_sampled_p50n_production_R100.json",
        "requires_dependency_injection",
        "Run current Figure 2 panels (a)--(f) at P=50N.",
    ),
    Stage(
        "qanr_verify", "qanr", ("qanr_sampled_p50",),
        "scripts/verify_qanr_matched_lasso_sweeps.py", None,
        "requires_current_output_verifier",
        "Verify grids, repetitions, slopes, failures, and table metrics.",
    ),
    Stage(
        "qanr_strong_lasso_noise", "qanr", ("qanr_verify",),
        "scripts/run_qanr_strong_fd_lasso_observation_noise_R100.py",
        "configs/experiments/qanr_strong_fd_lasso_observation_noise_R100.json",
        "requires_dependency_injection",
        "Run the validation-selected strong FD--LASSO noise comparison.",
    ),
    Stage(
        "predator_full", "predator_prey", (),
        "scripts/run_predator_prey_sweeps.py",
        "configs/experiments/predator_prey_sweeps_common_noise_001.json",
        "ready", "Generate trajectories and the full-pair robustness reference.",
    ),
    Stage(
        "predator_sampled_p50", "predator_prey", ("predator_full",),
        "scripts/run_predator_prey_sampled_pair_production.py",
        "configs/experiments/predator_prey_sampled_pair_production_P50_R20.json",
        "fresh_mode_added_needs_smoke_validation",
        "Run the P=50N prey--prey block with exact cross-species sums.",
    ),
    Stage(
        "predator_strong", "predator_prey", ("predator_sampled_p50",),
        "scripts/run_predator_prey_strong_observation_noise.py",
        "configs/experiments/predator_prey_strong_observation_noise_common_noise_001.json",
        "requires_dependency_injection",
        "Run the strong forward-difference observation-noise comparison.",
    ),
    Stage(
        "predator_strong_lasso_noise", "predator_prey", ("predator_sampled_p50",),
        "scripts/run_predator_prey_strong_lasso_observation_noise.py",
        "configs/experiments/predator_prey_strong_lasso_observation_noise_common_noise_001.json",
        "requires_dependency_injection",
        "Run the validation-selected strong FD--LASSO observation-noise sweep.",
    ),
    Stage(
        "predator_application", "predator_prey", ("predator_full",),
        "scripts/build_predator_prey_nominal_application.py", None,
        "requires_dependency_injection",
        "Build nominal profiles and the representative coupled trajectory.",
    ),
    Stage(
        "predator_verify", "predator_prey",
        ("predator_sampled_p50", "predator_strong", "predator_strong_lasso_noise",
         "predator_application"),
        "scripts/verify_predator_prey_strong_observation_noise.py", None,
        "requires_current_output_verifier",
        "Verify sampled sweeps, paths, strong comparison, and nominal outputs.",
    ),
    Stage(
        "localized_nominal", "localized10d", (),
        "scripts/run_localized10d_observation_noise.py",
        "configs/experiments/localized10d_observation_noise_001.json",
        "ready", "Generate the source-independent nominal observation-noise data.",
    ),
    Stage(
        "localized_robustness", "localized10d", ("localized_nominal",),
        "scripts/run_localized10d_noisy_robustness.py",
        "configs/experiments/localized10d_noisy_robustness_001.json",
        "ready", "Run the source-independent localized-10D robustness protocol.",
    ),
    Stage(
        "localized_sampled_p50", "localized10d", ("localized_robustness",),
        "scripts/run_localized10d_sampled_pair_production.py",
        "configs/experiments/localized10d_sampled_pair_production_P50_R20.json",
        "requires_dependency_injection",
        "Run the P=50N nominal and pair-count experiments.",
    ),
    Stage(
        "localized_verify", "localized10d", ("localized_sampled_p50",),
        "scripts/verify_localized10d_sampled_pair_production.py", None,
        "ready", "Verify repetitions, trajectories, and sweep statistics.",
    ),
    Stage(
        "cross_example", "integration",
        ("qanr_sampled_p50", "predator_application", "predator_strong",
         "localized_sampled_p50"),
        "scripts/run_cross_example_nominal_comparison.py",
        "configs/experiments/cross_example_nominal_comparison_common_noise_001.json",
        "requires_dependency_injection",
        "Build and verify the common-noise cross-example comparison.",
    ),
    Stage(
        "regularized_baselines", "integration", ("cross_example",),
        "scripts/run_regularized_baseline_exploration.py",
        "configs/experiments/regularized_baseline_exploration.json",
        "requires_dependency_injection",
        "Run the six-method nominal comparison used by Table 1 and overlays.",
    ),
    Stage(
        "metric_table", "integration", ("regularized_baselines",),
        "scripts/build_cross_example_metric_table.py",
        "configs/experiments/cross_example_metric_table_common_noise_001.json",
        "requires_dependency_injection", "Rebuild the six-method numerical Table 1 summary.",
    ),
    Stage(
        "figures", "integration",
        ("hat_noisy_comparison", "qanr_strong_lasso_noise", "predator_verify",
         "regularized_baselines", "localized_verify"),
        "scripts/regenerate_l4dc_figures.py",
        "configs/experiments/l4dc_current_figures.json",
        "requires_fresh_run_config", "Generate the eight PDFs used by eight current figures.",
    ),
    Stage(
        "release_checks", "integration", ("metric_table", "figures"),
        "scripts/reproduce_l4dc_current.py", None,
        "scaffold", "Validate manifests, public-file policy, hashes, and release scope.",
    ),
)


# Config fields that currently bind the production workflow to archived outputs.
ARCHIVED_INPUT_FIELDS = (
    ("configs/experiments/one_dimensional_hat_sampled_p50n_R100.json",
     "source_lasso_batch"),
    ("configs/experiments/one_dimensional_hat_sampled_p50n_R100.json",
     "source_ols_batch"),
    ("configs/experiments/qanr_redundant6_sampled_p50n_production_R100.json",
     "matched_full_pair_batch"),
    ("configs/experiments/qanr_redundant6_sampled_p50n_production_R100.json",
     "matched_full_pair_time_step_batch"),
    ("configs/experiments/predator_prey_sampled_pair_production_P50_R20.json",
     "reference_frozen_run"),
    ("configs/experiments/predator_prey_sampled_pair_production_P50_R20.json",
     "source_confirmation_run"),
    ("configs/experiments/predator_prey_strong_observation_noise_common_noise_001.json",
     "weak_sweep_records"),
    ("configs/experiments/localized10d_sampled_pair_production_P50_R20.json",
     "authoritative_frozen_protocol"),
    ("configs/experiments/localized10d_sampled_pair_production_P50_R20.json",
     "authoritative_requested_protocol"),
    ("configs/experiments/localized10d_sampled_pair_production_P50_R20.json",
     "authoritative_source_manifest"),
    ("configs/experiments/localized10d_sampled_pair_production_P50_R20.json",
     "frozen_observation_runner"),
    ("configs/experiments/fig1c_noisy_three_method.json", "source_batch"),
    ("configs/experiments/qanr_strong_fd_lasso_observation_noise_R100.json",
     "current_summary"),
    ("configs/experiments/predator_prey_strong_lasso_observation_noise_common_noise_001.json",
     "weak_sweep_records"),
    ("configs/experiments/regularized_baseline_exploration.json",
     "one_dimensional.baseline_records"),
    ("configs/experiments/regularized_baseline_exploration.json",
     "predator_prey.baseline_records"),
    ("configs/experiments/regularized_baseline_exploration.json",
     "predator_prey.strong_baseline_records"),
    ("configs/experiments/regularized_baseline_exploration.json",
     "localized_10d.frozen_protocol"),
    ("configs/experiments/regularized_baseline_exploration.json",
     "localized_10d.baseline_records"),
    ("configs/experiments/regularized_baseline_exploration.json",
     "localized_10d.strong_baseline_records"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError as error:
        raise ValueError(f"reproduction input must be below {ROOT}: {resolved}") from error


def numerical_source_contract() -> dict[str, Any]:
    if not NUMERICAL_TABLE_REFERENCE.is_file():
        raise FileNotFoundError(NUMERICAL_TABLE_REFERENCE)
    return {
        "table1_reference": {
            "path": relative(NUMERICAL_TABLE_REFERENCE),
            "sha256": digest(NUMERICAL_TABLE_REFERENCE),
        }
    }


def build_hat_sampled_config(
    lasso_batch: Path,
    ols_batch: Path,
    destination: Path,
    *,
    template: Path | None = None,
) -> dict[str, Any]:
    """Bind the 1D P=50N protocol to fresh full-pair stage outputs."""
    template = template or (
        ROOT / "configs/experiments/one_dimensional_hat_sampled_p50n_R100.json"
    )
    spec = read_json(Path(template))
    batches = {
        "lasso": Path(lasso_batch).resolve(),
        "ols": Path(ols_batch).resolve(),
    }
    for method, batch in batches.items():
        if not batch.is_dir():
            raise FileNotFoundError(batch)
        hash_key = f"source_{method}_hashes"
        required = tuple(spec[hash_key])
        missing = [relative_name for relative_name in required
                   if not (batch / relative_name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{method} batch is missing required artifacts: {missing}"
            )
        spec[f"source_{method}_batch"] = relative(batch)
        spec[hash_key] = {
            relative_name: digest(batch / relative_name)
            for relative_name in required
        }
    # The private author template historically pinned manuscript artifacts.
    # Fresh numerical runs intentionally carry no paper-source dependency.
    spec["current_artifact_audit"] = {}
    spec["generated_by"] = "scripts/reproduce_l4dc_current.py"
    spec["source_mode"] = "fresh_full_pair_stage_outputs"
    write_json(destination, spec)
    return spec


def build_qanr_sampled_config(
    full_batch: Path,
    time_step_batch: Path,
    destination: Path,
    *,
    template: Path | None = None,
) -> dict[str, Any]:
    """Bind the six-atom P=50N protocol to fresh full-pair stage outputs."""
    template = template or (
        ROOT / "configs/experiments/qanr_redundant6_sampled_p50n_production_R100.json"
    )
    spec = read_json(Path(template))
    full_batch = Path(full_batch).resolve()
    time_step_batch = Path(time_step_batch).resolve()
    required = {
        full_batch: ("summary.json", "observation_lasso.csv", "stochastic_lasso.csv"),
        time_step_batch: ("summary.json", "records.json"),
    }
    for batch, names in required.items():
        if not batch.is_dir():
            raise FileNotFoundError(batch)
        missing = [name for name in names if not (batch / name).is_file()]
        if missing:
            raise FileNotFoundError(f"QANR batch {batch} is missing: {missing}")
    spec["matched_full_pair_batch"] = relative(full_batch)
    spec["matched_full_pair_time_step_batch"] = relative(time_step_batch)
    protected = [
        item for item in spec.get("protected_paths", [])
        if not item.startswith("outputs/")
    ]
    protected.extend((relative(full_batch), relative(time_step_batch)))
    spec["protected_paths"] = list(dict.fromkeys(protected))
    spec["generated_by"] = "scripts/reproduce_l4dc_current.py"
    spec["source_mode"] = "fresh_full_pair_stage_outputs"
    write_json(destination, spec)
    return spec


def build_predator_sampled_config(
    full_batch: Path,
    destination: Path,
    *,
    template: Path | None = None,
) -> dict[str, Any]:
    """Bind the predator--prey P=50N protocol to a fresh full-pair sweep."""
    template = template or (
        ROOT / "configs/experiments/predator_prey_sampled_pair_production_P50_R20.json"
    )
    spec = read_json(Path(template))
    full_batch = Path(full_batch).resolve()
    required = ("manifest.json", "records.json", "requested_protocol.json")
    if not full_batch.is_dir():
        raise FileNotFoundError(full_batch)
    missing = [name for name in required if not (full_batch / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"predator--prey full-pair batch {full_batch} is missing: {missing}"
        )
    spec["reference_frozen_run"] = relative(full_batch)
    spec["reference_manifest_sha256"] = digest(full_batch / "manifest.json")
    spec["reference_records_sha256"] = digest(full_batch / "records.json")
    spec["reference_requested_protocol_sha256"] = digest(
        full_batch / "requested_protocol.json"
    )
    spec["source_mode"] = "fresh_full_pair_stage_outputs"
    spec["generated_by"] = "scripts/reproduce_l4dc_current.py"
    for key in tuple(spec):
        if key.startswith("source_confirmation_"):
            spec.pop(key)
    write_json(destination, spec)
    return spec


def build_predator_strong_config(
    sampled_batch: Path,
    destination: Path,
    *,
    template: Path | None = None,
) -> dict[str, Any]:
    """Bind the predator strong-baseline protocol to fresh weak records."""
    template = template or (
        ROOT / "configs/experiments/"
        "predator_prey_strong_observation_noise_common_noise_001.json"
    )
    spec = read_json(Path(template))
    sampled_batch = Path(sampled_batch).resolve()
    records = sampled_batch / "records.json"
    if not records.is_file():
        raise FileNotFoundError(records)
    spec["weak_sweep_records"] = relative(records)
    spec["weak_sweep_records_sha256"] = digest(records)
    spec["source_mode"] = "fresh_sampled_weak_stage_output"
    spec["generated_by"] = "scripts/reproduce_l4dc_current.py"
    write_json(destination, spec)
    return spec


def build_localized_sampled_config(
    robustness_batch: Path,
    destination: Path,
    *,
    template: Path | None = None,
) -> dict[str, Any]:
    """Bind localized-10D P=50N production to a fresh robustness stage."""
    template = template or (
        ROOT / "configs/experiments/localized10d_sampled_pair_production_P50_R20.json"
    )
    spec = read_json(Path(template))
    robustness_batch = Path(robustness_batch).resolve()
    relative_files = {
        "authoritative_frozen_protocol": "frozen_protocol.json",
        "authoritative_requested_protocol": "requested_protocol.json",
        "authoritative_source_manifest": "source_manifest.json",
        "frozen_observation_runner": "source/scripts/run_localized10d_observation_noise.py",
    }
    if not robustness_batch.is_dir():
        raise FileNotFoundError(robustness_batch)
    missing = [name for name in relative_files.values()
               if not (robustness_batch / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"localized-10D robustness batch {robustness_batch} is missing: {missing}"
        )
    for key, name in relative_files.items():
        spec[key] = relative(robustness_batch / name)
    spec["authoritative_frozen_protocol_sha256"] = digest(
        robustness_batch / relative_files["authoritative_frozen_protocol"]
    )
    spec["authoritative_requested_protocol_sha256"] = digest(
        robustness_batch / relative_files["authoritative_requested_protocol"]
    )
    spec["authoritative_source_manifest_sha256"] = digest(
        robustness_batch / relative_files["authoritative_source_manifest"]
    )
    protected = [
        item for item in spec.get("protected_paths", [])
        if not item.startswith("outputs/")
    ]
    protected.append(relative(robustness_batch))
    spec["protected_paths"] = list(dict.fromkeys(protected))
    spec["source_mode"] = "fresh_robustness_stage_output"
    spec["generated_by"] = "scripts/reproduce_l4dc_current.py"
    write_json(destination, spec)
    return spec


def build_predator_application_config(
    full_batch: Path,
    destination: Path,
    *,
    template: Path | None = None,
) -> dict[str, Any]:
    """Bind the nominal full-pair application to a fresh predator sweep."""
    template = template or (
        ROOT / "configs/experiments/"
        "predator_prey_nominal_from_sweep_common_noise_001.json"
    )
    spec = read_json(Path(template))
    full_batch = Path(full_batch).resolve()
    application_config = ROOT / spec["application_config"]
    artifacts = {
        "sweep_protocol": full_batch / "requested_protocol.json",
        "sweep_records": full_batch / "records.json",
    }
    if not application_config.is_file():
        raise FileNotFoundError(application_config)
    missing = [path for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"predator--prey full-pair batch is missing: {missing}"
        )
    spec["application_config_sha256"] = digest(application_config)
    for key, path in artifacts.items():
        spec[key] = relative(path)
        spec[f"{key}_sha256"] = digest(path)
    spec["source_mode"] = "fresh_full_pair_stage_output"
    spec["generated_by"] = "scripts/reproduce_l4dc_current.py"
    write_json(destination, spec)
    return spec


def build_cross_example_config(
    qanr_full: Path,
    predator_application: Path,
    predator_strong: Path,
    localized_nominal: Path,
    destination: Path,
    *,
    template: Path | None = None,
) -> dict[str, Any]:
    """Bind the aligned comparison to fresh outputs from all examples."""
    template = template or (
        ROOT / "configs/experiments/"
        "cross_example_nominal_comparison_common_noise_001.json"
    )
    spec = read_json(Path(template))
    sources = {
        "one_dimensional": Path(qanr_full) / "observation_lasso.csv",
        "predator_prey": Path(predator_application) / "records.json",
        "localized_10d": Path(localized_nominal) / "records.json",
    }
    strong_records = Path(predator_strong) / "records.json"
    missing = [path for path in (*sources.values(), strong_records)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cross-example inputs are missing: {missing}")
    for family, path in sources.items():
        spec[family]["source_records"] = relative(path)
        spec[family]["source_sha256"] = digest(path)
    spec["predator_prey"]["strong_source_records"] = relative(strong_records)
    spec["predator_prey"]["strong_source_sha256"] = digest(strong_records)
    spec["source_mode"] = "fresh_current_stage_outputs"
    spec["generated_by"] = "scripts/reproduce_l4dc_current.py"
    write_json(destination, spec)
    return spec


def build_hat_noisy_config(source_batch: Path, destination: Path) -> dict[str, Any]:
    spec = read_json(ROOT / "configs/experiments/fig1c_noisy_three_method.json")
    source_batch = Path(source_batch)
    names = ("frozen_protocol.json", "summary.json", "records.json")
    missing = [source_batch / name for name in names if not (source_batch / name).is_file()]
    if missing:
        raise FileNotFoundError(f"noisy Figure 1 inputs are missing: {missing}")
    spec["source_batch"] = relative(source_batch)
    spec["source_hashes"] = {name: digest(source_batch / name) for name in names}
    spec["source_mode"] = "fresh_current_stage_outputs"
    spec["generated_by"] = "scripts/reproduce_l4dc_current.py"
    write_json(destination, spec)
    return spec


def build_qanr_strong_lasso_config(
    sampled_batch: Path, production_config: Path, destination: Path
) -> dict[str, Any]:
    spec = read_json(
        ROOT / "configs/experiments/qanr_strong_fd_lasso_observation_noise_R100.json"
    )
    summary = Path(sampled_batch) / "summary_a_to_d.json"
    production_config = Path(production_config)
    for path in (summary, production_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    spec["production_config"] = relative(production_config)
    spec["production_config_sha256"] = digest(production_config)
    spec["current_summary"] = relative(summary)
    spec["current_summary_sha256"] = digest(summary)
    spec["source_mode"] = "fresh_current_stage_outputs"
    spec["generated_by"] = "scripts/reproduce_l4dc_current.py"
    write_json(destination, spec)
    return spec


def build_predator_strong_lasso_config(
    full_batch: Path, destination: Path
) -> dict[str, Any]:
    spec = read_json(
        ROOT / "configs/experiments/"
        "predator_prey_strong_lasso_observation_noise_common_noise_001.json"
    )
    records = Path(full_batch) / "records.json"
    if not records.is_file():
        raise FileNotFoundError(records)
    spec["weak_sweep_records"] = relative(records)
    spec["weak_sweep_records_sha256"] = digest(records)
    spec["source_mode"] = "fresh_current_stage_outputs"
    spec["generated_by"] = "scripts/reproduce_l4dc_current.py"
    write_json(destination, spec)
    return spec


def build_regularized_config(
    qanr_sampled: Path,
    qanr_config: Path,
    predator_sampled: Path,
    predator_config: Path,
    localized_sampled: Path,
    localized_config: Path,
    cross_example: Path,
    destination: Path,
) -> dict[str, Any]:
    spec = read_json(
        ROOT / "configs/experiments/regularized_baseline_exploration.json"
    )
    one_d_records = Path(qanr_sampled) / "nominal_table_records.csv"
    predator_records = Path(predator_sampled) / "nominal_application/records.json"
    localized_records = Path(localized_sampled) / "records.json"
    localized_protocol = Path(localized_sampled) / "frozen_protocol.json"
    strong_records = Path(cross_example) / "records.json"
    required = (
        one_d_records, predator_records, localized_records, localized_protocol,
        strong_records, Path(qanr_config), Path(predator_config), Path(localized_config),
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"regularized comparison inputs are missing: {missing}")
    blocks = {
        "one_dimensional": {
            "production_config": Path(qanr_config),
            "baseline_records": one_d_records,
        },
        "predator_prey": {
            "production_config": Path(predator_config),
            "baseline_records": predator_records,
            "strong_baseline_records": strong_records,
        },
        "localized_10d": {
            "production_config": Path(localized_config),
            "frozen_protocol": localized_protocol,
            "baseline_records": localized_records,
            "strong_baseline_records": strong_records,
        },
    }
    for name, values in blocks.items():
        for key, path in values.items():
            spec[name][key] = relative(path)
            spec[name][f"{key}_sha256"] = digest(path)
    spec["source_mode"] = "fresh_current_stage_outputs"
    spec["generated_by"] = "scripts/reproduce_l4dc_current.py"
    write_json(destination, spec)
    return spec


def build_metric_table_config(
    regularized_run: Path,
    destination: Path,
    *,
    template: Path | None = None,
) -> dict[str, Any]:
    """Bind the six-method numerical table to a fresh regularized run."""
    summary = Path(regularized_run) / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    spec = {
        "task": "cross_example_metric_table",
        "protocol_version": 3,
        "methods": [
            "weak_lasso", "weak_ridge_validation_selected", "weak_ols",
            "strong_forward_difference_lasso_validation_selected",
            "strong_forward_difference_ridge_validation_selected",
            "strong_forward_difference_ols",
        ],
        "regularized_summary": relative(summary),
        "regularized_summary_sha256": digest(summary),
        "source_mode": "fresh_current_stage_outputs",
        "generated_by": "scripts/reproduce_l4dc_current.py",
    }
    write_json(destination, spec)
    return spec


def build_figure_config(
    output: Path,
    paths: dict[str, Path],
    destination: Path,
) -> dict[str, Any]:
    """Create a protocol-v2 figure config for one isolated fresh run."""
    output = Path(output).resolve()
    sources = {
        "hat": paths["hat_sampled_p50"],
        "hat_noisy_comparison": paths["hat_noisy_comparison"],
        "redundant6": paths["qanr_sampled_p50"],
        "redundant6_time_step": paths["qanr_sampled_p50"],
        "runtime_scaling": paths["qanr_sampled_p50"] / "summary_f.json",
        "qanr_strong_lasso_noise": paths["qanr_strong_lasso_noise"],
        "predator_sweeps": paths["predator_sampled_p50"],
        "predator_strong_lasso_noise": paths["predator_strong_lasso_noise"],
        "localized10d": paths["localized_sampled_p50"],
        "regularized_baselines": paths["regularized_baselines"],
    }
    artifacts = (
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
    missing_sources = [path for key, path in sources.items()
                       if not (path.is_file() if key == "runtime_scaling"
                               else path.is_dir())]
    missing_artifacts = [path for path in artifacts if not path.is_file()]
    if missing_sources or missing_artifacts:
        raise FileNotFoundError(
            f"figure inputs are missing: {missing_sources + missing_artifacts}"
        )
    spec = {
        "task": "reproduce_l4dc_figures_from_fresh_runs",
        "protocol_version": 2,
        "source_mode": "fresh_current_stage_outputs",
        "generated_by": "scripts/reproduce_l4dc_current.py",
        "figure_directory": relative(output / "figures"),
        "build_directory": relative(output / "figure-build"),
        "sources": {name: relative(path) for name, path in sources.items()},
        "required_artifact_sha256": {
            relative(path): digest(path) for path in artifacts
        },
        "expected_figure_sha256": {name: None for name in CURRENT_FIGURES},
        "rendering_policy": {
            "figure_text_font": "Times_via_LaTeX_times_package",
            "figure_math_font": "Computer_Modern",
            "figure1_observation_noise": 0.01,
            "figure1_panel_a": "M14_weak_lasso_median_with_10_90_band",
            "figure1_panel_c_metric": "drift_error",
            "figure1_panel_c_title": "dictionary_size",
            "figure2_panel_a_comparison": "weak_lasso_weak_ols_strong_fd_lasso",
            "nominal_overlay_comparison": "weak_lasso_weak_ols_strong_fd_lasso",
            "figure4_layout": "rows_true_weak_lasso_weak_ols_strong_fd_lasso_columns_t0_t1_t2_t3",
        },
    }
    write_json(destination, spec)
    return spec


def git_status(directory: Path) -> list[str] | None:
    if not (directory / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "status", "--short", "--branch"], cwd=directory,
        check=False, capture_output=True, text=True,
    )
    return result.stdout.splitlines()


def validate_stage_graph(stages: tuple[Stage, ...] = STAGES) -> list[str]:
    names = [stage.name for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError("stage names must be unique")
    known: set[str] = set()
    for stage in stages:
        missing = [name for name in stage.dependencies if name not in known]
        if missing:
            raise ValueError(
                f"stage {stage.name} has unordered or unknown dependencies: {missing}"
            )
        known.add(stage.name)
    return names


def plan(workers: int = 4) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    validate_stage_graph()
    return {
        "task": "reproduce_current_l4dc_from_scratch",
        "protocol_version": 1,
        "implementation_status": (
            "full_source_to_artifact_orchestrator_implemented_"
            "pending_scientific_run"
        ),
        "workers": workers,
        "scope": "current_L4DC_numerical_table_figures_and_claims",
        "historical_outputs_required_by_target_workflow": False,
        "execution_modes": {
            "frozen_figures": "regenerate eight reference PDFs from sanitized frozen data",
            "smoke": "reduced source-to-figure wiring check",
            "full": "fresh trajectories, fits, numerical verification, table summary, and figures",
        },
        "figures": list(CURRENT_FIGURES),
        "stages": [asdict(stage) for stage in STAGES],
    }


def archived_input_audit() -> list[dict[str, Any]]:
    records = []
    cache: dict[Path, Any] = {}
    for relative, field in ARCHIVED_INPUT_FIELDS:
        config = ROOT / relative
        if not config.is_file():
            records.append({
                "config": relative, "field": field, "config_exists": False,
                "value": None, "target_exists": False,
            })
            continue
        if config not in cache:
            cache[config] = read_json(config)
        value = cache[config]
        for part in field.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        target = ROOT / value if isinstance(value, str) else None
        records.append({
            "config": relative,
            "field": field,
            "config_exists": True,
            "value": value,
            "is_archived_output_path": isinstance(value, str) and value.startswith("outputs/"),
            "target_exists": bool(target and target.exists()),
        })
    return records


def absolute_path_audit(paths: tuple[str, ...]) -> list[dict[str, str]]:
    findings = []

    def walk(value: Any, location: tuple[str, ...]):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from walk(child, location + (str(key),))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, location + (str(index),))
        elif isinstance(value, str) and value.startswith("/"):
            yield ".".join(location), value

    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            continue
        for field, value in walk(read_json(path), ()):
            findings.append({"config": relative, "field": field, "value": value})
    return findings


def audit() -> dict[str, Any]:
    validate_stage_graph()
    stage_files = []
    for stage in STAGES:
        for kind, relative in (("runner", stage.runner), ("config", stage.config)):
            if relative is None:
                continue
            stage_files.append({
                "stage": stage.name,
                "kind": kind,
                "path": relative,
                "exists": (ROOT / relative).is_file(),
            })
    archived = archived_input_audit()
    config_paths = tuple(sorted({relative for relative, _ in ARCHIVED_INPUT_FIELDS}))
    missing_runners = [
        item for item in stage_files
        if item["kind"] == "runner" and not item["exists"]
    ]
    blockers = []
    blockers.extend(
        {"type": "missing_runner", "stage": item["stage"], "path": item["path"]}
        for item in missing_runners
    )
    return {
        "task": "audit_current_l4dc_reproduction_dependencies",
        "status": "blocked" if blockers else "ready",
        "source_independent_full_run_ready": not blockers,
        "stage_files": stage_files,
        "archived_inputs": [
            {**item, "rebound_by_current_executor": bool(item.get("is_archived_output_path"))}
            for item in archived
        ],
        "absolute_paths_in_production_configs": absolute_path_audit(config_paths),
        "blockers": blockers,
        "working_trees": {
            "root": git_status(ROOT),
        },
    }


def current_stage_paths(output: Path) -> dict[str, Path]:
    data = Path(output) / "data"
    return {
        "hat_full_lasso": data / "hat_full_lasso",
        "hat_full_ols": data / "hat_full_ols",
        "hat_sampled_p50": data / "hat_sampled_p50",
        "hat_noisy_comparison": data / "hat_noisy_comparison",
        "qanr_full": data / "qanr_full",
        "qanr_time_step": data / "qanr_time_step",
        "qanr_sampled_p50": data / "qanr_sampled_p50",
        "qanr_strong_lasso_noise": data / "qanr_strong_lasso_noise",
        "predator_full": data / "predator_full",
        "predator_application": data / "predator_application",
        "predator_sampled_p50": data / "predator_sampled_p50",
        "predator_strong": data / "predator_strong",
        "predator_strong_lasso_noise": data / "predator_strong_lasso_noise",
        "localized_nominal": data / "localized_nominal",
        "localized_robustness": data / "localized_robustness",
        "localized_sampled_p50": data / "localized_sampled_p50",
        "cross_example": data / "cross_example",
        "regularized_baselines": data / "regularized_baselines",
        "metric_table": Path(output) / "table",
        "figures": Path(output) / "figures",
    }


def run_wiring_smoke(output: Path, *, workers: int, resume: bool) -> dict[str, Any]:
    """Run reduced core experiments and materialize every current-stage config.

    Current P=50N production loops retain their full scientific dimensions, so
    this smoke run exercises the shared simulation, fitting, verification, and
    figure-input families with reduced protocols, then validates the dependency
    injection needed by the current production runners. It does not claim
    scientific parity or execute the R=100/R=20 production loops.
    """
    output = Path(output).resolve()
    relative(output)
    if output.exists() and not resume:
        raise FileExistsError(
            f"smoke output already exists: {output}; use --resume or a new path"
        )
    output.mkdir(parents=True, exist_ok=True)
    base = output / "base-smoke"
    log = output / "base-smoke.log"
    command = [
        sys.executable, "scripts/reproduce_l4dc.py", "smoke",
        "--output", relative(base), "--workers", str(workers),
        *(["--resume"] if resume else []),
    ]
    environment = os.environ.copy()
    environment.update(
        SOURCE_DATE_EPOCH="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", WMFKI_PLOT_USETEX="0",
    )
    with log.open("a" if resume else "w") as handle:
        subprocess.run(
            command, cwd=ROOT, env=environment, check=True,
            stdout=handle, stderr=subprocess.STDOUT,
        )
    data = base / "data"
    configs = output / "current-configs"
    configs.mkdir(exist_ok=True)
    generated = {
        "hat_sampled_p50": configs / "hat_sampled_p50.json",
        "qanr_sampled_p50": configs / "qanr_sampled_p50.json",
        "predator_sampled_p50": configs / "predator_sampled_p50.json",
        "predator_strong": configs / "predator_strong.json",
        "localized_sampled_p50": configs / "localized_sampled_p50.json",
    }
    build_hat_sampled_config(
        data / "hat_lasso", data / "hat_ols", generated["hat_sampled_p50"]
    )
    build_qanr_sampled_config(
        data / "redundant6", data / "redundant6_time_step",
        generated["qanr_sampled_p50"],
    )
    build_predator_sampled_config(
        data / "predator_sweeps", generated["predator_sampled_p50"]
    )
    # The reduced core smoke does not execute the fixed R=20 sampled production
    # loop. Its matched full-pair records are sufficient to validate this path.
    build_predator_strong_config(
        data / "predator_sweeps", generated["predator_strong"]
    )
    build_localized_sampled_config(
        data / "localized10d_robustness", generated["localized_sampled_p50"]
    )
    result = {
        "task": "current_l4dc_wiring_smoke",
        "status": "passed",
        "scientific_parity_claimed": False,
        "current_fixed_dimension_production_loops_executed": False,
        "base_smoke_state": relative(base / "state.json"),
        "base_smoke_state_sha256": digest(base / "state.json"),
        "log": relative(log),
        "generated_configs": {
            name: {"path": relative(path), "sha256": digest(path)}
            for name, path in generated.items()
        },
    }
    write_json(output / "current_smoke.json", result)
    return result


class CurrentReproducer:
    """Execute current scientific stages in one isolated output directory."""

    def __init__(self, output: Path, *, workers: int, resume: bool):
        self.output = Path(output).resolve()
        relative(self.output)
        self.workers = int(workers)
        self.resume = bool(resume)
        self.paths = current_stage_paths(self.output)
        self.configs = self.output / "configs"
        self.logs = self.output / "logs"
        self.state_path = self.output / "state.json"
        if self.output.exists() and not self.resume:
            raise FileExistsError(
                f"output already exists: {self.output}; use --resume or a new path"
            )
        self.output.mkdir(parents=True, exist_ok=True)
        self.configs.mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        source_contract = numerical_source_contract()
        if self.state_path.is_file():
            self.state = read_json(self.state_path)
            if self.state.get("source_contract") != source_contract:
                raise ValueError(
                    "resume source contract mismatch for frozen numerical references"
                )
        else:
            self.state = {
                "task": "reproduce_current_l4dc_from_scratch",
                "protocol_version": 1,
                "status": "running",
                "workers": self.workers,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_contract": source_contract,
                "stages": {},
            }
            write_json(self.state_path, self.state)

    def command(
        self,
        label: str,
        arguments: list[str],
        expected: tuple[Path, ...],
        *,
        restart_paths: tuple[Path, ...] = (),
    ) -> None:
        previous = self.state["stages"].get(label, {})
        if self.resume and previous.get("status") == "completed" \
                and all(path.is_file() for path in expected):
            print(f"[current-reproduce] {label}: already completed", flush=True)
            return
        if self.resume:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            for path in restart_paths:
                if path.exists():
                    archived = path.with_name(path.name + f"-interrupted-{stamp}")
                    path.rename(archived)
        log = self.logs / f"{label}.log"
        print(f"[current-reproduce] {label}", flush=True)
        started = perf_counter()
        self.state["stages"][label] = {
            "status": "running",
            "command": arguments,
            "log": relative(log),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(self.state_path, self.state)
        environment = os.environ.copy()
        matplotlib_cache = self.output / "matplotlib-cache"
        matplotlib_cache.mkdir(exist_ok=True)
        tex_bin = Path("/Library/TeX/texbin")
        if tex_bin.is_dir():
            environment["PATH"] = f"{tex_bin}:{environment.get('PATH', '')}"
        environment.update(
            SOURCE_DATE_EPOCH="0",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            MPLCONFIGDIR=str(matplotlib_cache),
        )
        try:
            with log.open("w") as handle:
                subprocess.run(
                    arguments, cwd=ROOT, env=environment, check=True,
                    stdout=handle, stderr=subprocess.STDOUT,
                )
            missing = [relative(path) for path in expected if not path.is_file()]
            if missing:
                raise RuntimeError(f"stage {label} did not create {missing}")
        except Exception as error:
            self.state["stages"][label].update(
                status="failed",
                elapsed_seconds=perf_counter() - started,
                error=f"{type(error).__name__}: {error}",
            )
            self.state["status"] = "failed"
            write_json(self.state_path, self.state)
            raise
        self.state["stages"][label].update(
            status="completed",
            elapsed_seconds=perf_counter() - started,
            artifacts={relative(path): digest(path) for path in expected},
        )
        write_json(self.state_path, self.state)

    def run_scientific_data(self) -> None:
        py = sys.executable
        p = self.paths

        self.command(
            "hat_full_lasso",
            [
                py, "scripts/run_1d_hat_dictionary_grid_from_scratch.py", "--run",
                "--workers", str(self.workers), "--out-dir", relative(p["hat_full_lasso"]),
                *(["--resume"] if self.resume and p["hat_full_lasso"].exists() else []),
            ],
            (p["hat_full_lasso"] / "summary/summary.json",
             p["hat_full_lasso"] / "batch_status.json"),
        )
        self.command(
            "hat_full_lasso_verify",
            [
                py, "scripts/verify_1d_hat_dictionary_grid_R100.py",
                relative(p["hat_full_lasso"]), "--out-dir",
                relative(p["hat_full_lasso"] / "verification"),
                "--workers", str(min(self.workers, 3)),
            ],
            (p["hat_full_lasso"] / "verification/verification.json",),
            restart_paths=(p["hat_full_lasso"] / "verification",),
        )
        ols_config = read_json(
            ROOT / "configs/experiments/one_dimensional_hat_dictionary_grid_OLS_R100.json"
        )
        ols_config["source_lasso_grid_batch"] = relative(p["hat_full_lasso"])
        ols_config_path = self.configs / "hat_full_ols.json"
        write_json(ols_config_path, ols_config)
        self.command(
            "hat_full_ols",
            [
                py, "scripts/run_1d_hat_dictionary_grid_OLS_R100.py", "--run",
                "--config", relative(ols_config_path), "--workers", str(self.workers),
                "--out-dir", relative(p["hat_full_ols"]),
                *(["--resume"] if self.resume and
                  (p["hat_full_ols"] / "frozen_protocol.json").is_file() else []),
            ],
            (p["hat_full_ols"] / "summary/summary.json",
             p["hat_full_ols"] / "batch_status.json"),
            restart_paths=(
                (p["hat_full_ols"],) if self.resume and p["hat_full_ols"].exists()
                and not (p["hat_full_ols"] / "frozen_protocol.json").is_file()
                else ()
            ),
        )
        self.command(
            "hat_full_ols_verify",
            [
                py, "scripts/verify_1d_hat_dictionary_grid_OLS_R100.py",
                relative(p["hat_full_ols"]), "--out-dir",
                relative(p["hat_full_ols"] / "verification"),
                "--workers", str(min(self.workers, 3)),
            ],
            (p["hat_full_ols"] / "verification/verification.json",),
            restart_paths=(p["hat_full_ols"] / "verification",),
        )
        hat_config = self.configs / "hat_sampled_p50.json"
        build_hat_sampled_config(
            p["hat_full_lasso"], p["hat_full_ols"], hat_config
        )
        self.command(
            "hat_sampled_p50",
            [
                py, "scripts/run_1d_hat_sampled_p50n_R100.py", "--run",
                "--config", relative(hat_config), "--workers", str(self.workers),
                "--out-dir", relative(p["hat_sampled_p50"]),
                *(["--resume"] if self.resume and
                  (p["hat_sampled_p50"] / "frozen_protocol.json").is_file()
                  else []),
            ],
            (p["hat_sampled_p50"] / "report.json",
             p["hat_sampled_p50"] / "verification/verification.json",
             p["hat_sampled_p50"] / "frozen_protocol.json",
             p["hat_sampled_p50"] / "records.json",
             p["hat_sampled_p50"] / "summary.json",
             p["hat_sampled_p50"] / "figure1_candidate/metadata.json"),
            restart_paths=(
                (p["hat_sampled_p50"],) if self.resume
                and p["hat_sampled_p50"].exists()
                and not (p["hat_sampled_p50"] / "frozen_protocol.json").is_file()
                else ()
            ),
        )

        hat_noisy_config = self.configs / "hat_noisy_comparison.json"
        build_hat_noisy_config(p["hat_sampled_p50"], hat_noisy_config)
        self.command(
            "hat_noisy_comparison",
            [
                py, "scripts/run_fig1c_noisy_three_method.py",
                "--config", relative(hat_noisy_config), "--workers", str(self.workers),
                "--out", relative(p["hat_noisy_comparison"]),
            ],
            (p["hat_noisy_comparison"] / "manifest.json",
             p["hat_noisy_comparison"] / "records.json",
             p["hat_noisy_comparison"] / "summary.json"),
            restart_paths=(p["hat_noisy_comparison"],),
        )

        self.command(
            "qanr_full",
            [
                py, "scripts/run_qanr_matched_lasso_sweeps.py", "--config",
                "configs/experiments/qanr_redundant6_lasso_sweeps_R100.json",
                "--workers", str(self.workers), "--out-dir", relative(p["qanr_full"]),
            ],
            (p["qanr_full"] / "summary.json", p["qanr_full"] / "report.json"),
            restart_paths=(p["qanr_full"],),
        )
        self.command(
            "qanr_full_verify",
            [py, "scripts/verify_qanr_matched_lasso_sweeps.py", relative(p["qanr_full"])],
            (p["qanr_full"] / "independent_verification.json",),
        )
        self.command(
            "qanr_time_step",
            [
                py, "scripts/run_qanr_time_step_repetitions.py", "--config",
                "configs/experiments/qanr_redundant6_time_step_paired_R100.json",
                "--workers", str(self.workers), "--output", relative(p["qanr_time_step"]),
                *(["--resume"] if self.resume and p["qanr_time_step"].exists() else []),
            ],
            (p["qanr_time_step"] / "manifest.json",
             p["qanr_time_step"] / "summary.json"),
        )
        self.command(
            "qanr_time_step_verify",
            [
                py, "scripts/verify_qanr_time_step_paired_R100.py",
                relative(p["qanr_time_step"]), "--skip-old-comparison",
            ],
            (p["qanr_time_step"] / "independent_verification.json",),
        )
        qanr_config = self.configs / "qanr_sampled_p50.json"
        build_qanr_sampled_config(p["qanr_full"], p["qanr_time_step"], qanr_config)
        self.command(
            "qanr_sampled_p50",
            [
                py, "scripts/run_qanr_sampled_p50n_production_R100.py",
                "--config", relative(qanr_config), "--workers", str(self.workers),
                "--output", relative(p["qanr_sampled_p50"]),
                *(["--resume"] if self.resume and p["qanr_sampled_p50"].exists() else []),
            ],
            (p["qanr_sampled_p50"] / "manifest.json",
             p["qanr_sampled_p50"] / "summary_a_to_d.json",
             p["qanr_sampled_p50"] / "summary_e.json",
             p["qanr_sampled_p50"] / "summary_f.json",
             p["qanr_sampled_p50"] / "nominal_table_metrics.json"),
        )

        predator_args = [
            py, "scripts/run_predator_prey_sweeps.py", "--config",
            "configs/experiments/predator_prey_sweeps_common_noise_001.json",
            "--workers", str(self.workers),
        ]
        if self.resume and p["predator_full"].exists():
            predator_args.extend(["--resume", relative(p["predator_full"])])
        else:
            predator_args.extend(["--output", relative(p["predator_full"])])
        qanr_strong_config = self.configs / "qanr_strong_lasso_noise.json"
        build_qanr_strong_lasso_config(
            p["qanr_sampled_p50"], qanr_config, qanr_strong_config
        )
        self.command(
            "qanr_strong_lasso_noise",
            [
                py, "scripts/run_qanr_strong_fd_lasso_observation_noise_R100.py",
                "--config", relative(qanr_strong_config), "--workers", str(self.workers),
                "--out", relative(p["qanr_strong_lasso_noise"]),
            ],
            (p["qanr_strong_lasso_noise"] / "manifest.json",
             p["qanr_strong_lasso_noise"] / "records.json",
             p["qanr_strong_lasso_noise"] / "summary.json"),
            restart_paths=(p["qanr_strong_lasso_noise"],),
        )

        self.command(
            "predator_full", predator_args,
            (p["predator_full"] / "records.json", p["predator_full"] / "manifest.json"),
        )
        predator_application_config = self.configs / "predator_application.json"
        build_predator_application_config(
            p["predator_full"], predator_application_config
        )
        self.command(
            "predator_application",
            [
                py, "scripts/build_predator_prey_nominal_application.py",
                "--config", relative(predator_application_config), "--output",
                relative(p["predator_application"]),
            ],
            (p["predator_application"] / "records.json",
             p["predator_application"] / "display_trajectory.npz",
             p["predator_application"] / "manifest.json"),
            restart_paths=(p["predator_application"],),
        )
        predator_config = self.configs / "predator_sampled_p50.json"
        build_predator_sampled_config(p["predator_full"], predator_config)
        self.command(
            "predator_sampled_p50",
            [
                py, "scripts/run_predator_prey_sampled_pair_production.py",
                "--config", relative(predator_config), "--workers",
                str(min(self.workers, 5)), "--output", relative(p["predator_sampled_p50"]),
            ],
            (p["predator_sampled_p50"] / "manifest.json",
             p["predator_sampled_p50"] / "effective_protocol.json",
             p["predator_sampled_p50"] / "records.json",
             p["predator_sampled_p50"] / "nominal_application/records.json",
             p["predator_sampled_p50"] / "nominal_application/display_trajectory.npz"),
            restart_paths=(p["predator_sampled_p50"],),
        )
        predator_strong_lasso_config = self.configs / "predator_strong_lasso_noise.json"
        build_predator_strong_lasso_config(
            p["predator_full"], predator_strong_lasso_config
        )
        self.command(
            "predator_strong_lasso_noise",
            [
                py, "scripts/run_predator_prey_strong_lasso_observation_noise.py",
                "--config", relative(predator_strong_lasso_config), "--workers",
                str(self.workers), "--out", relative(p["predator_strong_lasso_noise"]),
            ],
            (p["predator_strong_lasso_noise"] / "manifest.json",
             p["predator_strong_lasso_noise"] / "records.json",
             p["predator_strong_lasso_noise"] / "summary.json"),
            restart_paths=(p["predator_strong_lasso_noise"],),
        )

        predator_strong_config = self.configs / "predator_strong.json"
        build_predator_strong_config(
            p["predator_sampled_p50"], predator_strong_config
        )
        self.command(
            "predator_strong",
            [
                py, "scripts/run_predator_prey_strong_observation_noise.py",
                "--config", relative(predator_strong_config), "--workers",
                str(self.workers), "--out-dir", relative(p["predator_strong"]),
            ],
            (p["predator_strong"] / "records.json",
             p["predator_strong"] / "manifest.json"),
            restart_paths=(p["predator_strong"],),
        )
        self.command(
            "predator_strong_verify",
            [
                py, "scripts/verify_predator_prey_strong_observation_noise.py",
                relative(p["predator_strong"]),
            ],
            (p["predator_strong"] / "quality_report.json",),
        )

        nominal_config = read_json(
            ROOT / "configs/experiments/localized10d_observation_noise_001.json"
        )
        nominal_config.update(
            clean_reference_records=None,
            clean_reference_sha256=None,
            source_mode="generated_without_historical_clean_reference",
        )
        nominal_config_path = self.configs / "localized_nominal.json"
        write_json(nominal_config_path, nominal_config)
        self.command(
            "localized_nominal",
            [
                py, "scripts/run_localized10d_observation_noise.py", "--config",
                relative(nominal_config_path), "--workers", str(self.workers),
                "--out", relative(p["localized_nominal"]),
                *(["--resume"] if self.resume and p["localized_nominal"].exists() else []),
            ],
            (p["localized_nominal"] / "records.json",
             p["localized_nominal"] / "manifest.json"),
        )
        self.command(
            "localized_nominal_verify",
            [
                py, "scripts/verify_localized10d_observation_noise.py",
                relative(p["localized_nominal"]),
            ],
            (p["localized_nominal"] / "independent_verification.json",),
        )
        robustness_config = read_json(
            ROOT / "configs/experiments/localized10d_noisy_robustness_001.json"
        )
        robustness_config.update(
            nominal_reference_records=relative(p["localized_nominal"] / "records.json"),
            nominal_reference_sha256=digest(p["localized_nominal"] / "records.json"),
        )
        robustness_config_path = self.configs / "localized_robustness.json"
        write_json(robustness_config_path, robustness_config)
        self.command(
            "localized_robustness",
            [
                py, "scripts/run_localized10d_noisy_robustness.py", "--config",
                relative(robustness_config_path), "--workers", str(self.workers),
                "--out-dir", relative(p["localized_robustness"]),
                *(["--resume"] if self.resume and p["localized_robustness"].exists() else []),
            ],
            (p["localized_robustness"] / "records.json",
             p["localized_robustness"] / "manifest.json"),
        )
        self.command(
            "localized_robustness_verify",
            [
                py, "scripts/verify_localized10d_noisy_robustness.py",
                relative(p["localized_robustness"]),
            ],
            (p["localized_robustness"] / "quality_report.json",),
        )
        localized_config = self.configs / "localized_sampled_p50.json"
        build_localized_sampled_config(
            p["localized_robustness"], localized_config
        )
        self.command(
            "localized_sampled_p50",
            [
                py, "scripts/run_localized10d_sampled_pair_production.py",
                "--config", relative(localized_config), "--workers", str(self.workers),
                "--out-dir", relative(p["localized_sampled_p50"]),
                *(["--resume"] if self.resume and p["localized_sampled_p50"].exists() else []),
            ],
            (p["localized_sampled_p50"] / "records.json",
             p["localized_sampled_p50"] / "manifest.json",
             p["localized_sampled_p50"] / "frozen_protocol.json",
             p["localized_sampled_p50"] / "source_manifest.json"),
        )
        self.command(
            "localized_sampled_p50_verify",
            [
                py, "scripts/verify_localized10d_sampled_pair_production.py",
                relative(p["localized_sampled_p50"]), "--write",
            ],
            (p["localized_sampled_p50"] / "independent_verification.json",),
        )
        self.state["status"] = "scientific_data_complete"
        self.state["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(self.state_path, self.state)

    def run_integration_artifacts(self) -> dict[str, Any]:
        """Build the fresh comparison, numerical table, and eight figure PDFs."""
        py = sys.executable
        p = self.paths

        cross_config = self.configs / "cross_example.json"
        build_cross_example_config(
            p["qanr_full"], p["predator_application"], p["predator_strong"],
            p["localized_nominal"], cross_config,
        )
        self.command(
            "cross_example",
            [
                py, "scripts/run_cross_example_nominal_comparison.py",
                "--config", relative(cross_config), "--out",
                relative(p["cross_example"]), "--workers", str(self.workers),
            ],
            (p["cross_example"] / "records.json",
             p["cross_example"] / "summary.json",
             p["cross_example"] / "manifest.json"),
            restart_paths=(p["cross_example"],),
        )
        self.command(
            "cross_example_verify",
            [py, "scripts/verify_cross_example_nominal_comparison.py",
             relative(p["cross_example"])],
            (p["cross_example"] / "independent_verification.json",),
        )

        regularized_config = self.configs / "regularized_baselines.json"
        build_regularized_config(
            p["qanr_sampled_p50"], self.configs / "qanr_sampled_p50.json",
            p["predator_sampled_p50"], self.configs / "predator_sampled_p50.json",
            p["localized_sampled_p50"], self.configs / "localized_sampled_p50.json",
            p["cross_example"], regularized_config,
        )
        self.command(
            "regularized_baselines",
            [
                py, "scripts/run_regularized_baseline_exploration.py",
                "--config", relative(regularized_config), "--workers", str(self.workers),
                "--out", relative(p["regularized_baselines"]),
            ],
            (p["regularized_baselines"] / "manifest.json",
             p["regularized_baselines"] / "records.json",
             p["regularized_baselines"] / "summary.json",
             p["regularized_baselines"] / "summary_table.md"),
            restart_paths=(p["regularized_baselines"],),
        )

        table_config = self.configs / "metric_table.json"
        build_metric_table_config(p["regularized_baselines"], table_config)
        self.command(
            "metric_table",
            [
                py, "scripts/build_cross_example_metric_table.py",
                "--config", relative(table_config), "--output",
                relative(p["metric_table"]),
            ],
            (p["metric_table"] / "summary.json",
             p["metric_table"] / "table_draft.tex",
             p["metric_table"] / "artifact_sha256.json"),
            restart_paths=(p["metric_table"],),
        )
        self.command(
            "metric_table_verify",
            [
                py, "scripts/verify_l4dc_numerical_values.py", "--summary",
                relative(p["metric_table"] / "summary.json"),
                "--reference", relative(NUMERICAL_TABLE_REFERENCE), "--output",
                relative(p["metric_table"] / "numerical_verification.json"),
            ],
            (p["metric_table"] / "numerical_verification.json",),
        )

        figure_config = self.configs / "figures.json"
        build_figure_config(self.output, p, figure_config)
        figure_manifest = self.output / "figure-build/current_figure_regeneration.json"
        self.command(
            "figures",
            [py, "scripts/regenerate_l4dc_figures.py", "--config",
             relative(figure_config)],
            tuple(p["figures"] / name for name in CURRENT_FIGURES)
            + (figure_manifest,),
            restart_paths=(p["figures"], self.output / "figure-build"),
        )

        artifacts = {
            "cross_example_summary": p["cross_example"] / "summary.json",
            "metric_table_summary": p["metric_table"] / "summary.json",
            "metric_table_numerical_verification": (
                p["metric_table"] / "numerical_verification.json"
            ),
            "numerical_table_reference": NUMERICAL_TABLE_REFERENCE,
            "figure_manifest": figure_manifest,
            **{f"figure:{name}": p["figures"] / name for name in CURRENT_FIGURES},
        }
        result = {
            "task": "current_l4dc_integration_artifacts",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                name: {"path": relative(path), "sha256": digest(path)}
                for name, path in artifacts.items()
            },
        }
        write_json(self.output / "current_artifacts.json", result)
        self.state["status"] = "complete"
        self.state["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(self.state_path, self.state)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--workers", type=int, default=4)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument(
        "--strict", action="store_true",
        help="exit with status 1 while source-independent reproduction is blocked",
    )
    hat_parser = subparsers.add_parser("materialize-hat-config")
    hat_parser.add_argument("--lasso-batch", type=Path, required=True)
    hat_parser.add_argument("--ols-batch", type=Path, required=True)
    hat_parser.add_argument("--output", type=Path, required=True)
    qanr_parser = subparsers.add_parser("materialize-qanr-config")
    qanr_parser.add_argument("--full-batch", type=Path, required=True)
    qanr_parser.add_argument("--time-step-batch", type=Path, required=True)
    qanr_parser.add_argument("--output", type=Path, required=True)
    predator_parser = subparsers.add_parser("materialize-predator-config")
    predator_parser.add_argument("--full-batch", type=Path, required=True)
    predator_parser.add_argument("--output", type=Path, required=True)
    strong_parser = subparsers.add_parser("materialize-predator-strong-config")
    strong_parser.add_argument("--sampled-batch", type=Path, required=True)
    strong_parser.add_argument("--output", type=Path, required=True)
    localized_parser = subparsers.add_parser("materialize-localized-config")
    localized_parser.add_argument("--robustness-batch", type=Path, required=True)
    localized_parser.add_argument("--output", type=Path, required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--output", type=Path)
    smoke_parser.add_argument("--workers", type=int, default=2)
    smoke_parser.add_argument("--resume", action="store_true")
    data_parser = subparsers.add_parser("data")
    data_parser.add_argument("--output", type=Path)
    data_parser.add_argument("--workers", type=int, default=4)
    data_parser.add_argument("--resume", action="store_true")
    artifacts_parser = subparsers.add_parser("artifacts")
    artifacts_parser.add_argument("--output", type=Path, required=True)
    artifacts_parser.add_argument("--workers", type=int, default=4)
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--output", type=Path)
    all_parser.add_argument("--workers", type=int, default=4)
    all_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        try:
            result = plan(args.workers)
        except ValueError as error:
            parser.error(str(error))
    elif args.command == "audit":
        result = audit()
    elif args.command == "smoke":
        if not 1 <= args.workers <= 8:
            parser.error("workers must be between 1 and 8")
        if args.resume and args.output is None:
            parser.error("--resume requires --output")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = args.output or DEFAULT_PARENT / f"smoke-{stamp}"
        if not output.is_absolute():
            output = ROOT / output
        result = run_wiring_smoke(
            output, workers=args.workers, resume=args.resume
        )
    elif args.command in {"data", "artifacts", "all"}:
        if not 1 <= args.workers <= 8:
            parser.error("workers must be between 1 and 8")
        resume = args.command == "artifacts" or args.resume
        if args.command != "artifacts" and resume and args.output is None:
            parser.error("--resume requires --output")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = args.output or DEFAULT_PARENT / f"full-{stamp}"
        if not output.is_absolute():
            output = ROOT / output
        reproducer = CurrentReproducer(
            output, workers=args.workers, resume=resume
        )
        if args.command in {"data", "all"}:
            reproducer.run_scientific_data()
        integration = None
        if args.command in {"artifacts", "all"}:
            integration = reproducer.run_integration_artifacts()
        result = {
            "status": reproducer.state["status"],
            "output": relative(reproducer.output),
            "state": relative(reproducer.state_path),
            **({"integration": integration} if integration is not None else {}),
        }
    else:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        if args.command == "materialize-hat-config":
            spec = build_hat_sampled_config(
                args.lasso_batch if args.lasso_batch.is_absolute()
                else ROOT / args.lasso_batch,
                args.ols_batch if args.ols_batch.is_absolute()
                else ROOT / args.ols_batch,
                output,
            )
            inputs = {
                "source_lasso_batch": spec["source_lasso_batch"],
                "source_ols_batch": spec["source_ols_batch"],
            }
        elif args.command == "materialize-qanr-config":
            spec = build_qanr_sampled_config(
                args.full_batch if args.full_batch.is_absolute()
                else ROOT / args.full_batch,
                args.time_step_batch if args.time_step_batch.is_absolute()
                else ROOT / args.time_step_batch,
                output,
            )
            inputs = {
                "matched_full_pair_batch": spec["matched_full_pair_batch"],
                "matched_full_pair_time_step_batch": spec[
                    "matched_full_pair_time_step_batch"
                ],
            }
        elif args.command == "materialize-predator-config":
            spec = build_predator_sampled_config(
                args.full_batch if args.full_batch.is_absolute()
                else ROOT / args.full_batch,
                output,
            )
            inputs = {
                "reference_frozen_run": spec["reference_frozen_run"],
                "source_mode": spec["source_mode"],
            }
        elif args.command == "materialize-predator-strong-config":
            spec = build_predator_strong_config(
                args.sampled_batch if args.sampled_batch.is_absolute()
                else ROOT / args.sampled_batch,
                output,
            )
            inputs = {
                "weak_sweep_records": spec["weak_sweep_records"],
                "source_mode": spec["source_mode"],
            }
        else:
            spec = build_localized_sampled_config(
                args.robustness_batch if args.robustness_batch.is_absolute()
                else ROOT / args.robustness_batch,
                output,
            )
            inputs = {
                "authoritative_frozen_protocol": spec[
                    "authoritative_frozen_protocol"
                ],
                "source_mode": spec["source_mode"],
            }
        result = {
            "status": "written",
            "output": str(output.resolve()),
            **inputs,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "audit" and args.strict and result["blockers"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
