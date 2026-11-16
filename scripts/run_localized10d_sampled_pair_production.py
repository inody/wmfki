"""Run the production-candidate localized 10D study with P=50N sampled pairs.

The run is derived from the authoritative fixed-noise frozen protocol without
modifying current, frozen, manuscript, or P=10N preview artifacts.  Standard
conditions use P=50N; the pair-count sweep selects one absolute LASSO penalty
at P0=25000 on an independent sampled validation system and fixes it for all P.
"""

from __future__ import annotations

import argparse
import ast
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import compute_A_timeweak
from kernels import make_radial_basis
from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR,
    REFERENCE_COLOR,
    SECONDARY_COLOR,
    TRUE_COLOR,
    apply_manuscript_style,
    finish_axes,
)
from scripts.run_localized10d_observation_noise import (
    assemble_with_fixed_tests,
    fixed_space_test_spec,
    observed_trajectory,
)
from scripts.run_localized10d_pilot import _augment_record
from scripts.run_morse10d_sweeps import (
    _radial_drift_fast,
    _trajectory_path,
    design_diagnostics,
    latent_evaluation_system,
    load_or_simulate,
    method_records,
    physical_conditions,
    read_json,
    repetition_plan,
    select_raw_lasso,
    write_json,
)
from utils import make_pair_sampler

DEFAULT_CONFIG = ROOT / "configs/experiments/localized10d_sampled_pair_production_P50_R20.json"
BASE_SOURCE_FILES = (
    "configs/experiments/localized10d_sampled_pair_production_P50_R20.json",
    "scripts/run_localized10d_sampled_pair_production.py",
    "scripts/verify_localized10d_sampled_pair_production.py",
    "test/test_localized10d_sampled_pair_production.py",
    "scripts/run_localized10d_observation_noise.py",
    "scripts/run_localized10d_pilot.py",
    "scripts/run_morse10d_sweeps.py",
    "scripts/run_1d_hat_lasso.py",
    "scripts/manuscript_plot_style.py",
    "features.py", "kernels.py", "pipeline.py", "regression.py", "simulate.py",
    "tests.py", "time_tests.py", "utils.py",
)
LEGACY_AUTHORITY_FILES = (
    "outputs/validation/localized10d_noisy_robustness/sweeps-20261023T123949Z/requested_protocol.json",
    "outputs/validation/localized10d_noisy_robustness/sweeps-20261023T123949Z/frozen_protocol.json",
    "outputs/validation/localized10d_noisy_robustness/sweeps-20261023T123949Z/source_manifest.json",
    "outputs/validation/localized10d_noisy_robustness/sweeps-20261023T123949Z/source/scripts/run_localized10d_observation_noise.py",
)
CORE_FROZEN_SOURCES = (
    "features.py", "kernels.py", "pipeline.py", "regression.py", "simulate.py",
    "tests.py", "time_tests.py", "utils.py", "scripts/run_1d_hat_lasso.py",
    "scripts/run_localized10d_pilot.py", "scripts/run_morse10d_sweeps.py",
)
SWEEPS = ("process_sigma", "particle_number", "sampled_pairs")
METHODS = ("lasso", "ols")
METRICS = (
    "coefficient_error_relative",
    "latent_drift_relative_error",
    "profile_relative_error",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_status():
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-uall"], cwd=ROOT,
        check=False, capture_output=True, text=True,
    )
    return result.stdout.splitlines()


def function_ast_digest(path, names):
    tree = ast.parse(Path(path).read_text())
    functions = {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }
    if set(functions) != set(names):
        raise ValueError(f"missing audited functions in {path}: {sorted(set(names)-set(functions))}")
    return object_digest(functions)


def path_inventory(relative_paths):
    inventory = {}
    for relative in relative_paths:
        path = ROOT / relative
        if not path.exists():
            inventory[relative] = {"exists": False, "sha256": None, "files": 0}
            continue
        if path.is_file():
            inventory[relative] = {"exists": True, "sha256": digest(path), "files": 1}
            continue
        entries = []
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            entries.append((str(item.relative_to(path)), digest(item)))
        inventory[relative] = {
            "exists": True,
            "sha256": object_digest(entries),
            "files": len(entries),
        }
    return inventory


def validate_requested(requested, source_frozen, source_requested, source_manifest):
    if requested.get("task") != "localized10d_sampled_pair_production_P50_R20":
        raise ValueError("invalid production task")
    if requested.get("protocol_version") != 1:
        raise ValueError("unsupported production protocol version")
    if requested.get("study_status") != "production_candidate_distance_localized_P50_R20":
        raise ValueError("production-candidate designation changed")
    if int(requested.get("repetitions", -1)) != 20:
        raise ValueError("production run must use R=20")
    for path_key, hash_key in (
        ("authoritative_frozen_protocol", "authoritative_frozen_protocol_sha256"),
        ("authoritative_requested_protocol", "authoritative_requested_protocol_sha256"),
        ("authoritative_source_manifest", "authoritative_source_manifest_sha256"),
    ):
        path = ROOT / requested[path_key]
        if not path.is_file() or digest(path) != requested[hash_key]:
            raise ValueError(f"authoritative input changed: {requested[path_key]}")
    if source_frozen.get("observation_noise") != 0.01:
        raise ValueError("authoritative frozen protocol is not sigma_obs=0.01")
    if source_frozen["evaluation"].get("scientific_repetitions") != 20:
        raise ValueError("authoritative frozen protocol is not R=20")
    if source_requested.get("observation_noise") != 0.01:
        raise ValueError("authoritative requested protocol is not sigma_obs=0.01")
    if float(requested.get("observation_noise", -1.0)) != 0.01:
        raise ValueError("production observation noise must remain 0.01")
    if requested.get("pair_sampling") != {
        "mode": "resample_each_time",
        "with_replacement": True,
        "exclude_diagonal": True,
        "pairs_per_particle": 50,
        "nominal_particles": 500,
        "nominal_pairs": 25000,
    }:
        raise ValueError("P=50N pair-sampling contract changed")
    if requested.get("pair_count_sweep") != [3000, 5000, 10000, 20000, 25000, 30000]:
        raise ValueError("pair-count sweep changed")
    if requested.get("pair_count_penalty_policy") != {
        "selection_pairs": 25000,
        "selection_validation": "independent_sampled_validation_system_at_P0",
        "reuse": "fix_selected_absolute_alpha_across_all_pair_counts_within_repetition",
    }:
        raise ValueError("P0 penalty-selection policy changed")
    if not requested.get("protected_paths"):
        raise ValueError("protected paths are required")
    for relative in CORE_FROZEN_SOURCES:
        expected = source_manifest.get("sha256", {}).get(relative)
        if expected is None or digest(ROOT / relative) != expected:
            raise ValueError(f"active numerical source differs from authoritative source: {relative}")
    helpers = {"observed_trajectory", "fixed_space_test_spec", "assemble_with_fixed_tests"}
    active = function_ast_digest(ROOT / requested["active_observation_runner"], helpers)
    frozen = function_ast_digest(ROOT / requested["frozen_observation_runner"], helpers)
    if active != frozen:
        raise ValueError("active observation helpers differ from authoritative frozen source")


def effective_protocol(requested, source_frozen):
    spec = json.loads(json.dumps(source_frozen))
    spec["source_task"] = source_frozen["task"]
    spec["source_frozen_protocol"] = requested["authoritative_frozen_protocol"]
    spec["source_frozen_protocol_sha256"] = requested[
        "authoritative_frozen_protocol_sha256"
    ]
    spec["task"] = requested["task"]
    spec["protocol_version"] = requested["protocol_version"]
    spec["study_status"] = requested["study_status"]
    spec["evaluation"]["scientific_repetitions"] = int(requested["repetitions"])
    spec["pair_sampling"] = {
        "baseline_pairs": int(requested["pair_sampling"]["nominal_pairs"]),
        "mode": requested["pair_sampling"]["mode"],
        "with_replacement": requested["pair_sampling"]["with_replacement"],
        "exclude_diagonal": requested["pair_sampling"]["exclude_diagonal"],
        "pairs_per_particle": int(requested["pair_sampling"]["pairs_per_particle"]),
    }
    spec["sweeps"]["sampled_pairs"] = list(requested["pair_count_sweep"])
    spec["pair_penalty_policy"] = (
        "absolute_alpha_selected_at_P0_25000_by_independent_sampled_validation_"
        "then_fixed_across_all_P_within_repetition"
    )
    spec["pair_count_penalty_policy"] = requested["pair_count_penalty_policy"]
    spec["pair_index_policy"] = requested["pair_index_policy"]
    spec["condition_budget_policy"] = requested["condition_budget_policy"]
    spec["selection_policy"] = requested["selection_policy"]
    spec["source_preservation_policy"] = requested["source_preservation_policy"]
    spec["production_artifacts"] = requested["production_artifacts"]
    spec["output_policy"] = requested["output_policy"]
    spec["data_policy"] = (
        "R=20 production candidate derived from the authoritative sigma_obs=0.01 "
        "protocol. Training, validation, and latent test trajectories are separate. "
        "Pair indices are redrawn at every observed time, shared across all tests and "
        "atoms at that time, and independent between training and validation."
    )
    return spec


def pair_budget(spec, particles):
    return int(spec["pair_sampling"]["pairs_per_particle"]) * int(particles)


def noise_seed(spec, plan, split):
    seeds = spec["observation_noise_seeds"]
    return int(seeds[f"{split}_start"] + seeds["repetition_stride"] * plan["index"])


def add_observation_noise(spec, latent, particles, plan, split):
    if split == "test":
        return np.asarray(latent, dtype=float), None
    seed = noise_seed(spec, plan, split)
    latent = np.asarray(latent, dtype=float)
    maximum_particles = int(spec["model"]["particles"])
    full_shape = (latent.shape[0], latent.shape[1], maximum_particles, latent.shape[3])
    standard_normal = np.random.default_rng(seed).normal(size=full_shape)
    observed = observed_trajectory(
        latent, spec["observation_noise"], standard_normal[:, :, :particles, :]
    )
    return observed, seed


def assemble(spec, trajectory, time, particles, sigma, plan, split, pairs, tests):
    return assemble_with_fixed_tests(
        spec, trajectory, time, particles, sigma,
        plan[f"{split}_seed"], plan[f"{split}_pair_seed"], plan["space_seed"],
        int(pairs), tests,
    )


def tag_standard_records(records, spec, particles, sigma):
    tagged = []
    base_n = int(spec["model"]["particles"])
    base_sigma = float(spec["model"]["sigma"])
    if particles == base_n and sigma in spec["sweeps"]["process_sigma"]:
        tagged.extend({**row, "sweep": "process_sigma", "sweep_value": sigma}
                      for row in records)
    if sigma == base_sigma and particles in spec["sweeps"]["particle_number"]:
        tagged.extend({**row, "sweep": "particle_number", "sweep_value": particles}
                      for row in records)
    return tagged


def pair_sampler_contract_audit():
    sampler = make_pair_sampler(
        500, rng=np.random.default_rng(8128), with_replacement=True,
        exclude_diagonal=True, resample_each_call=True,
    )
    first = sampler(10)
    second = sampler(10)
    calls = []

    def recording_sampler(n):
        call = len(calls)
        i = np.array([(call + k) % n for k in range(7)], dtype=int)
        j = (i + 1) % n
        calls.append((i.copy(), j.copy()))
        return i, j

    X = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4, 1) / 10.0
    time = np.array([0.0, 0.5, 1.0])
    test_fns = {
        "phi": lambda x: np.ones((x.shape[0], 2), dtype=float),
        "grad_phi": lambda x: np.ones((x.shape[0], 2, 1), dtype=float),
    }

    def basis(z):
        return np.repeat(z[:, None, :], 3, axis=1)

    compute_A_timeweak(
        X, time, test_fns, basis, recording_sampler,
        {"psi": np.ones((2, 3), dtype=float)}, weights=np.ones(3, dtype=float),
    )
    pairs = list(zip(first[0].tolist(), first[1].tolist()))
    checks = {
        "ordered_off_diagonal": bool(np.all(first[0] != first[1]) and np.all(second[0] != second[1])),
        "with_replacement": len(set(pairs)) < len(pairs),
        "independent_between_time_calls": bool(
            not (np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1]))
        ),
        "one_draw_per_trajectory_time_shared_across_tests_and_atoms": len(calls) == 6,
    }
    return {"passed": all(checks.values()), "checks": checks}


def source_protocol_audit(source_frozen, spec):
    preserved = {
        section: spec.get(section) == source_frozen.get(section)
        for section in ("model", "dictionary", "weak_tests", "estimator", "seeds")
    }
    preserved.update({
        "process_sigma_grid": spec["sweeps"]["process_sigma"]
        == source_frozen["sweeps"]["process_sigma"],
        "particle_number_grid": spec["sweeps"]["particle_number"]
        == source_frozen["sweeps"]["particle_number"],
        "evaluation": spec["evaluation"] == source_frozen["evaluation"],
        "observation_noise": spec["observation_noise"]
        == source_frozen["observation_noise"] == 0.01,
        "observation_noise_seeds": spec["observation_noise_seeds"]
        == source_frozen["observation_noise_seeds"],
    })
    changes = {
        "pair_sampling": {
            "authoritative": source_frozen["pair_sampling"],
            "production": spec["pair_sampling"],
        },
        "sampled_pairs": {
            "authoritative": source_frozen["sweeps"]["sampled_pairs"],
            "production": spec["sweeps"]["sampled_pairs"],
        },
        "pair_penalty_policy": {
            "authoritative": source_frozen["pair_penalty_policy"],
            "production": spec["pair_penalty_policy"],
        },
    }
    return {"passed": all(preserved.values()), "preserved": preserved, "changes": changes}


def failed_lasso_record(A, common):
    return {
        **common, "method": "lasso", "coefficients": None,
        "nonzero_reported": None, **design_diagnostics(A), "converged": False,
        "unconverged_candidates": 1, "selected_at_grid_boundary": False,
        "alpha": None, "alpha_index": None, "alpha_over_max": None,
        "iterations": None, "fit_seconds": 0.0,
        "validation_weak_residual_mse": None, "latent_drift_relative_error": None,
        "profile_relative_error": None, "numerical_divergence": True,
        "ols_lstsq_rank": None, "ols_smallest_singular_value": None,
    }


def run_repetition(payload):
    out_string, spec, plan = payload
    out = Path(out_string)
    records, selections, trajectory_status = [], {}, []
    base_n = int(spec["model"]["particles"])
    base_sigma = float(spec["model"]["sigma"])
    target = np.asarray(spec["model"]["interaction"]["coefficients"], dtype=float)
    nominal = None

    with threadpool_limits(limits=1):
        for particles, sigma in physical_conditions(spec):
            split_data = {}
            simulation_started = perf_counter()
            for split in ("training", "validation", "test"):
                X, time, cached = load_or_simulate(out, spec, particles, sigma, split, plan)
                split_data[split] = (X, time)
                trajectory_status.append({
                    "particles": particles, "sigma": sigma, "split": split,
                    "cache_hit": cached,
                })
            simulation_seconds = perf_counter() - simulation_started
            stride = int(spec["model"]["observation_stride"])
            sampled = {
                split: (data[0][:, ::stride], data[1][::stride])
                for split, data in split_data.items()
            }
            observed, observation_seeds = {}, {}
            for split, (X, time) in sampled.items():
                observed_X, seed = add_observation_noise(spec, X, particles, plan, split)
                observed[split] = (observed_X, time)
                observation_seeds[split] = seed
            training_tests = fixed_space_test_spec(
                spec, observed["training"][0], plan["space_seed"]
            )
            validation_tests = fixed_space_test_spec(
                spec, observed["validation"][0], plan["space_seed"]
            )
            pairs = pair_budget(spec, particles)
            train_A, train_b, train_timing = assemble(
                spec, *observed["training"], particles, sigma, plan,
                "training", pairs, training_tests,
            )
            validation_A, validation_b, validation_timing = assemble(
                spec, *observed["validation"], particles, sigma, plan,
                "validation", pairs, validation_tests,
            )
            evaluation_system = latent_evaluation_system(
                spec, split_data["test"][0], plan["evaluation_seed"]
            )
            fit_id = f"rep{plan['index']:03d}-N{particles}-sigma{sigma:g}-P{pairs}"
            common = {
                "repetition": plan["index"], "fit_id": fit_id,
                "particles": particles, "sigma": sigma,
                "observation_noise": float(spec["observation_noise"]),
                "pairs": pairs, "pair_budget_rule": "50N",
                "pair_mode": spec["pair_sampling"]["mode"],
                "training_seed": plan["training_seed"],
                "validation_seed": plan["validation_seed"],
                "test_seed": plan["test_seed"],
                "training_observation_noise_seed": observation_seeds["training"],
                "validation_observation_noise_seed": observation_seeds["validation"],
                "training_pair_seed": plan["training_pair_seed"],
                "validation_pair_seed": plan["validation_pair_seed"],
                "training_space_test_sha256": object_digest(training_tests),
                "validation_space_test_sha256": object_digest(validation_tests),
                "simulation_and_cache_seconds": simulation_seconds,
                "training_assembly_seconds": train_timing["assembly_seconds"],
                "validation_assembly_seconds": validation_timing["assembly_seconds"],
            }
            fitted, paths = method_records(
                train_A, train_b, validation_A, validation_b, evaluation_system,
                spec["estimator"], common,
            )
            fitted = [_augment_record(row, target) for row in fitted]
            records.extend(tag_standard_records(fitted, spec, particles, sigma))
            selections[fit_id] = paths
            if particles == base_n and sigma == base_sigma:
                nominal = {
                    "observed": observed,
                    "evaluation_system": evaluation_system,
                    "training_tests": training_tests,
                    "validation_tests": validation_tests,
                    "train_A": train_A, "train_b": train_b,
                    "train_timing": train_timing,
                    "validation_A": validation_A, "validation_b": validation_b,
                    "validation_timing": validation_timing,
                }

        if nominal is None:
            raise RuntimeError("nominal N=500, sigma=0.05 condition was not generated")
        p0 = int(spec["pair_count_penalty_policy"]["selection_pairs"])
        selected_theta, selection, selected_info = select_raw_lasso(
            nominal["train_A"], nominal["train_b"],
            nominal["validation_A"], nominal["validation_b"], spec["estimator"],
        )
        selections[f"rep{plan['index']:03d}-P{p0}-lambda-selection"] = {
            "lasso": selection
        }
        alpha = selected_info["alpha"]
        for pairs in spec["sweeps"]["sampled_pairs"]:
            if int(pairs) == p0:
                A, b, train_timing = (
                    nominal["train_A"], nominal["train_b"], nominal["train_timing"]
                )
            else:
                A, b, train_timing = assemble(
                    spec, *nominal["observed"]["training"], base_n, base_sigma,
                    plan, "training", int(pairs), nominal["training_tests"],
                )
            common = {
                "repetition": plan["index"],
                "fit_id": f"rep{plan['index']:03d}-P{pairs}",
                "sweep": "sampled_pairs", "sweep_value": int(pairs),
                "particles": base_n, "sigma": base_sigma,
                "observation_noise": float(spec["observation_noise"]),
                "pairs": int(pairs),
                "pair_budget_rule": "declared_pair_count_sweep_value",
                "pair_mode": spec["pair_sampling"]["mode"],
                "training_seed": plan["training_seed"],
                "validation_seed": plan["validation_seed"],
                "test_seed": plan["test_seed"],
                "training_observation_noise_seed": noise_seed(spec, plan, "training"),
                "validation_observation_noise_seed": noise_seed(spec, plan, "validation"),
                "training_pair_seed": plan["training_pair_seed"],
                "validation_pair_seed": plan["validation_pair_seed"],
                "training_space_test_sha256": object_digest(nominal["training_tests"]),
                "validation_space_test_sha256": object_digest(nominal["validation_tests"]),
                "lambda_selection_pairs": p0,
                "validation_pairs": p0,
                "fixed_absolute_alpha": alpha,
                "simulation_and_cache_seconds": 0.0,
                "training_assembly_seconds": train_timing["assembly_seconds"],
                "validation_assembly_seconds": nominal["validation_timing"][
                    "assembly_seconds"
                ],
                "pair_design_relative_difference_from_P0": float(
                    np.linalg.norm(A - nominal["train_A"])
                    / max(np.linalg.norm(nominal["train_A"]), 1e-300)
                ),
            }
            if alpha is None:
                failed = failed_lasso_record(A, common)
                both, _ = method_records(
                    A, b, nominal["validation_A"], nominal["validation_b"],
                    nominal["evaluation_system"], spec["estimator"], common,
                    fixed_alpha=1.0,
                )
                fitted = [failed] + [row for row in both if row["method"] == "ols"]
            else:
                fitted, _ = method_records(
                    A, b, nominal["validation_A"], nominal["validation_b"],
                    nominal["evaluation_system"], spec["estimator"], common,
                    fixed_alpha=alpha, initial_theta=selected_theta,
                )
            records.extend(_augment_record(row, target) for row in fitted)

        for particles, sigma in physical_conditions(spec):
            for split in ("training", "validation", "test"):
                _trajectory_path(out, particles, sigma, split, plan["index"]).unlink(
                    missing_ok=True
                )
    return {
        "index": plan["index"], "records": records, "selections": selections,
        "trajectory_cache": trajectory_status, "pair_count_selected_alpha": alpha,
    }


def finite_stats(values):
    present = np.asarray([value for value in values if value is not None], dtype=float)
    finite = present[np.isfinite(present)]
    return {
        "count": len(values), "present_count": int(present.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "rmse": float(np.sqrt(np.mean(finite ** 2))) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "q25": float(np.quantile(finite, 0.25)) if finite.size else None,
        "q75": float(np.quantile(finite, 0.75)) if finite.size else None,
        "minimum": float(np.min(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def summarize(records):
    groups = []
    for sweep in SWEEPS:
        values = sorted({float(row["sweep_value"]) for row in records if row["sweep"] == sweep})
        for value in values:
            for method in METHODS:
                rows = [
                    row for row in records
                    if row["sweep"] == sweep
                    and float(row["sweep_value"]) == value
                    and row["method"] == method
                ]
                groups.append({
                    "sweep": sweep, "value": value, "method": method,
                    "repetitions": len(rows),
                    "pairs": sorted({int(row["pairs"]) for row in rows}),
                    **{metric: finite_stats([row[metric] for row in rows]) for metric in METRICS},
                    "condition_A": finite_stats([row["condition_A"] for row in rows]),
                    "rank_deficient": int(sum(bool(row["rank_deficient"]) for row in rows)),
                    "nonconverged": int(sum(not bool(row["converged"]) for row in rows)),
                    "boundary_selections": int(sum(
                        bool(row["selected_at_grid_boundary"]) for row in rows
                    )),
                    "numerical_divergence": int(sum(
                        bool(row["numerical_divergence"]) for row in rows
                    )),
                })
    return {"record_count": len(records), "groups": groups}


def write_csv(path, records):
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "coefficients"}
        coefficients = record.get("coefficients")
        for index in range(4):
            row[f"theta_{index + 1}"] = None if coefficients is None else coefficients[index]
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_lookup(summary):
    return {
        (row["sweep"], float(row["value"]), row["method"]): row
        for row in summary["groups"]
    }


def plot_robustness(summary, out):
    apply_manuscript_style()
    lookup = group_lookup(summary)
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 2.65), constrained_layout=True)
    labels = {
        "process_sigma": ("(a) Process noise", r"process noise $\sigma$"),
        "particle_number": ("(b) Particle number", r"particles $N$"),
        "sampled_pairs": ("(c) Pair count", r"sampled pairs $P$"),
    }
    colors = {"lasso": ESTIMATE_COLOR, "ols": TRUE_COLOR}
    markers = {"lasso": "^", "ols": "o"}
    for axis, sweep in zip(axes, SWEEPS):
        values = sorted({key[1] for key in lookup if key[0] == sweep})
        for method in METHODS:
            groups = [lookup[(sweep, value, method)] for value in values]
            median = np.asarray([row["coefficient_error_relative"]["median"] for row in groups])
            q25 = np.asarray([row["coefficient_error_relative"]["q25"] for row in groups])
            q75 = np.asarray([row["coefficient_error_relative"]["q75"] for row in groups])
            axis.plot(values, median, color=colors[method], marker=markers[method],
                      label=method.upper())
            axis.fill_between(values, q25, q75, color=colors[method], alpha=0.16, linewidth=0)
        axis.set_yscale("log")
        if sweep in {"particle_number", "sampled_pairs"}:
            axis.set_xscale("log")
        if sweep == "process_sigma":
            axis.set_xlim(left=0.0)
        axis.set_title(labels[sweep][0])
        axis.set_xlabel(labels[sweep][1])
        if axis is axes[0]:
            axis.set_ylabel("relative coefficient error")
        axis.legend(frameon=False, fontsize=7)
        finish_axes(axis)
    fig.suptitle("Localized 10D production candidate: P=50N, R=20", fontsize=9)
    paths = []
    for suffix, dpi in (("pdf", 300), ("png", 220)):
        path = out / f"localized10d_P50_robustness.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def nominal_rows(records, method):
    return sorted([
        row for row in records
        if row["sweep"] == "process_sigma"
        and np.isclose(float(row["sweep_value"]), 0.05)
        and row["method"] == method
    ], key=lambda row: row["repetition"])


def plot_nominal_profile(spec, records, out):
    basis, _ = make_radial_basis(spec["dictionary"])
    radius = np.linspace(*spec["evaluation"]["profile_radius"], 500)
    target = np.asarray(spec["model"]["interaction"]["coefficients"], dtype=float)
    truth = basis(radius) @ target
    apply_manuscript_style()
    fig, axis = plt.subplots(figsize=(4.35, 3.0), constrained_layout=True)
    axis.plot(radius, truth, color=REFERENCE_COLOR, linewidth=1.8, label="true")
    colors = {"lasso": ESTIMATE_COLOR, "ols": TRUE_COLOR}
    for method in METHODS:
        rows = nominal_rows(records, method)
        curves = np.asarray([
            basis(radius) @ np.asarray(row["coefficients"], dtype=float) for row in rows
        ])
        median = np.median(curves, axis=0)
        q25, q75 = np.quantile(curves, [0.25, 0.75], axis=0)
        axis.plot(radius, median, color=colors[method], label=f"{method.upper()} median")
        axis.fill_between(radius, q25, q75, color=colors[method], alpha=0.17, linewidth=0)
    axis.axhline(0.0, color="0.45", linewidth=0.7)
    axis.set(xlabel="distance $r$", ylabel="radial profile",
             title=r"Nominal $N=500$, $P=25000$, $\sigma=0.05$, $\sigma_{obs}=0.01$")
    axis.legend(frameon=False, fontsize=7)
    finish_axes(axis)
    paths = []
    for suffix, dpi in (("pdf", 300), ("png", 220)):
        path = out / f"localized10d_P50_nominal_profile.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def simulate_coupled(spec, coefficients, seed):
    model = spec["model"]
    particles = int(spec["evaluation"]["rollout_particles"])
    steps = min(
        int(round(spec["evaluation"]["rollout_horizon"] / model["dt"])),
        int(model["steps"]),
    )
    rng = np.random.default_rng(seed)
    scales = rng.choice(model["initial_scales"], size=particles, replace=True)
    initial = rng.normal(size=(particles, model["dimension"])) * scales[:, None]
    states = {"true": initial.copy()}
    interactions = {"true": model["interaction"]}
    for method, theta in coefficients.items():
        states[method] = initial.copy()
        interactions[method] = {
            **model["interaction"], "coefficients": np.asarray(theta, dtype=float).tolist()
        }
    radius = {name: np.empty(steps + 1) for name in states}
    numerator = {name: 0.0 for name in coefficients}
    denominator = 0.0

    def mean_radius(x):
        return float(np.linalg.norm(x - x.mean(axis=0, keepdims=True), axis=1).mean())

    for name, state in states.items():
        radius[name][0] = mean_radius(state)
    denominator += float(np.sum(states["true"] ** 2))
    for step in range(steps):
        increment = rng.normal(size=initial.shape)
        previous = {name: state.copy() for name, state in states.items()}
        for name in states:
            states[name] = (
                previous[name]
                + model["dt"] * _radial_drift_fast(previous[name], interactions[name])
                + model["sigma"] * np.sqrt(model["dt"]) * increment
            )
            radius[name][step + 1] = mean_radius(states[name])
        denominator += float(np.sum(states["true"] ** 2))
        for method in coefficients:
            numerator[method] += float(np.sum((states[method] - states["true"]) ** 2))
    tri = np.triu_indices(particles, k=1)
    distances = {
        name: np.linalg.norm(state[:, None, :] - state[None, :, :], axis=-1)[tri]
        for name, state in states.items()
    }
    errors = {
        method: float(np.sqrt(numerator[method] / max(denominator, 1e-300)))
        for method in coefficients
    }
    return np.arange(steps + 1) * model["dt"], radius, distances, errors


def plot_nominal_trajectory(spec, records, out):
    repetition = int(spec["production_artifacts"]["representative_repetition"])
    coefficients = {}
    for method in METHODS:
        row = next(row for row in nominal_rows(records, method) if row["repetition"] == repetition)
        if row["coefficients"] is None:
            raise ValueError(f"representative {method} coefficients are unavailable")
        coefficients[method] = row["coefficients"]
    plan = repetition_plan(spec)[repetition]
    time, radius, distances, errors = simulate_coupled(spec, coefficients, plan["test_seed"])
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), constrained_layout=True)
    colors = {"true": REFERENCE_COLOR, "lasso": ESTIMATE_COLOR, "ols": TRUE_COLOR}
    styles = {"true": "-", "lasso": "--", "ols": ":"}
    for name in ("true", "lasso", "ols"):
        axes[0].plot(time, radius[name], color=colors[name], linestyle=styles[name],
                     label=name.upper() if name != "true" else "true")
    axes[0].set(xlabel="time", ylabel="mean radius", title="(a) Coupled radius")
    axes[0].legend(frameon=False, fontsize=7)
    lower = min(np.min(values) for values in distances.values())
    upper = max(np.max(values) for values in distances.values())
    bins = np.linspace(lower, upper, 40)
    for name in ("true", "lasso", "ols"):
        axes[1].hist(distances[name], bins=bins, density=True, histtype="step",
                     color=colors[name], linestyle=styles[name],
                     label=name.upper() if name != "true" else "true")
    axes[1].set(xlabel="final pair distance", ylabel="density",
                title="(b) Final pair distances")
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        finish_axes(axis)
    paths = []
    for suffix, dpi in (("pdf", 300), ("png", 220)):
        path = out / f"localized10d_P50_nominal_trajectory.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    np.savez_compressed(
        out / "nominal_trajectory_rep000.npz", time=time,
        radius_true=radius["true"], radius_lasso=radius["lasso"],
        radius_ols=radius["ols"], final_distance_true=distances["true"],
        final_distance_lasso=distances["lasso"], final_distance_ols=distances["ols"],
    )
    report = {
        "repetition": repetition, "test_seed": plan["test_seed"],
        "coefficients": coefficients, "coupled_trajectory_relative_error": errors,
        "particles": int(spec["evaluation"]["rollout_particles"]),
        "horizon": float(time[-1]),
    }
    write_json(out / "nominal_trajectory_report.json", report)
    return paths, report


def log_log_slope(values, metrics):
    x = np.asarray(values, dtype=float)
    y = np.asarray(metrics, dtype=float)
    keep = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0]) if keep.sum() >= 2 else None


def build_table_metrics(spec, records, summary, trajectory_report):
    nominal = {}
    for method in METHODS:
        rows = nominal_rows(records, method)
        nominal[method] = {
            metric: finite_stats([row[metric] for row in rows]) for metric in METRICS
        }
        nominal[method]["coefficient_estimate"] = {
            "mean": np.mean([row["coefficients"] for row in rows], axis=0).tolist(),
            "median": np.median([row["coefficients"] for row in rows], axis=0).tolist(),
        }
    lookup = group_lookup(summary)
    sweeps = {}
    for sweep in SWEEPS:
        values = sorted({key[1] for key in lookup if key[0] == sweep})
        sweeps[sweep] = {"values": values, "methods": {}}
        for method in METHODS:
            rmse_values = [
                lookup[(sweep, value, method)]["coefficient_error_relative"]["rmse"]
                for value in values
            ]
            sweeps[sweep]["methods"][method] = {
                "coefficient_error_rmse": rmse_values,
                "first": {"value": values[0], "rmse": rmse_values[0]},
                "last": {"value": values[-1], "rmse": rmse_values[-1]},
                "log_log_slope": log_log_slope(values, rmse_values),
            }
    diagnostics = {
        "records": len(records),
        "rank_deficient": sum(bool(row["rank_deficient"]) for row in records),
        "nonconverged_records": sum(not bool(row["converged"]) for row in records),
        "boundary_selections": sum(bool(row["selected_at_grid_boundary"]) for row in records),
        "numerical_divergence": sum(bool(row["numerical_divergence"]) for row in records),
        "missing_coefficients": sum(row["coefficients"] is None for row in records),
    }
    return {
        "nominal_definition": {
            "particles": 500, "pairs": 25000, "process_sigma": 0.05,
            "observation_noise": 0.01, "repetitions": 20,
        },
        "nominal": nominal,
        "representative_trajectory": trajectory_report,
        "sweeps": sweeps,
        "diagnostics": diagnostics,
    }


def write_table_files(out, metrics):
    write_json(out / "table_metrics.json", metrics)
    rows = []
    for method, payload in metrics["nominal"].items():
        for metric in METRICS:
            stats = payload[metric]
            rows.append({
                "section": "nominal", "sweep": None, "method": method,
                "metric": metric, "value": None, "mean": stats["mean"],
                "rmse": stats["rmse"], "median": stats["median"],
                "q25": stats["q25"], "q75": stats["q75"], "slope": None,
            })
    for sweep, payload in metrics["sweeps"].items():
        for method, item in payload["methods"].items():
            for value, rmse_value in zip(payload["values"], item["coefficient_error_rmse"]):
                rows.append({
                    "section": "sweep", "sweep": sweep, "method": method,
                    "metric": "coefficient_error_relative", "value": value,
                    "mean": None, "rmse": rmse_value, "median": None,
                    "q25": None, "q75": None, "slope": item["log_log_slope"],
                })
    with (out / "table_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Localized 10D P=50N production-candidate table metrics", "",
        "Nominal condition: N=500, P=25000, sigma=0.05, sigma_obs=0.01, R=20.", "",
        "| Method | Relative coefficient RMSE | Latent drift RMSE | Profile RMSE | Median coefficient error [IQR] |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        payload = metrics["nominal"][method]
        coefficient = payload["coefficient_error_relative"]
        lines.append(
            f"| {method.upper()} | {coefficient['rmse']:.6g} | "
            f"{payload['latent_drift_relative_error']['rmse']:.6g} | "
            f"{payload['profile_relative_error']['rmse']:.6g} | "
            f"{coefficient['median']:.6g} [{coefficient['q25']:.6g}, {coefficient['q75']:.6g}] |"
        )
    lines.extend(["", "## Representative coupled trajectory", ""])
    for method in METHODS:
        value = metrics["representative_trajectory"]["coupled_trajectory_relative_error"][method]
        lines.append(f"- {method.upper()}: {value:.6g}")
    lines.extend(["", "## Sweep coefficient-RMSE endpoints and slopes", "",
                  "| Sweep | Method | First | Last | Log-log slope |", "|---|---|---:|---:|---:|"])
    for sweep in SWEEPS:
        for method in METHODS:
            item = metrics["sweeps"][sweep]["methods"][method]
            lines.append(
                f"| {sweep} | {method.upper()} | {item['first']['value']:g}: "
                f"{item['first']['rmse']:.6g} | {item['last']['value']:g}: "
                f"{item['last']['rmse']:.6g} | {item['log_log_slope']:.6g} |"
            )
    (out / "table_metrics.md").write_text("\n".join(lines) + "\n")


def source_files(requested):
    files = list(BASE_SOURCE_FILES)
    if requested.get("source_mode") == "fresh_robustness_stage_output":
        files.extend([
            requested["authoritative_requested_protocol"],
            requested["authoritative_frozen_protocol"],
            requested["authoritative_source_manifest"],
            requested["frozen_observation_runner"],
        ])
    else:
        files.extend(LEGACY_AUTHORITY_FILES)
    return tuple(dict.fromkeys(files))


def snapshot_sources(out, requested):
    hashes = {}
    for relative in source_files(requested):
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    write_json(out / "source_manifest.json", {"sha256": hashes})
    return hashes


def verify_output(out):
    out = Path(out)
    spec = read_json(out / "frozen_protocol.json")
    requested = read_json(out / "requested_protocol.json")
    records = read_json(out / "records.json")
    manifest = read_json(out / "manifest.json")
    repetitions = int(spec["evaluation"]["scientific_repetitions"])
    expected_records = repetitions * 2 * sum(len(spec["sweeps"][key]) for key in SWEEPS)
    checks = {
        "status": manifest.get("status") == "production_candidate_completed",
        "R20": repetitions == 20,
        "sigma_obs_0.01": float(spec["observation_noise"]) == 0.01,
        "record_count": len(records) == expected_records == manifest.get("record_count"),
        "methods": {row["method"] for row in records} == set(METHODS),
        "pair_mode": all(row.get("pair_mode") == "resample_each_time" for row in records),
        "independent_pair_seeds": all(
            row["training_pair_seed"] != row["validation_pair_seed"] for row in records
        ),
        "all_coefficients_present": all(row["coefficients"] is not None for row in records),
        "all_metrics_finite": all(
            row[metric] is not None and np.isfinite(float(row[metric]))
            for row in records for metric in METRICS
        ),
    }
    for sweep in SWEEPS:
        for value in spec["sweeps"][sweep]:
            for method in METHODS:
                rows = [
                    row for row in records if row["sweep"] == sweep
                    and float(row["sweep_value"]) == float(value)
                    and row["method"] == method
                ]
                checks[f"group:{sweep}:{value}:{method}"] = len(rows) == repetitions
                if sweep == "process_sigma":
                    budget = 25000
                elif sweep == "particle_number":
                    budget = 50 * int(value)
                else:
                    budget = int(value)
                checks[f"budget:{sweep}:{value}:{method}"] = all(
                    int(row["pairs"]) == budget for row in rows
                )
    checks["pair_grid_contains_nominal"] = 25000 in spec["sweeps"]["sampled_pairs"]
    for repetition in range(repetitions):
        rows = [
            row for row in records if row["repetition"] == repetition
            and row["sweep"] == "sampled_pairs" and row["method"] == "lasso"
        ]
        alphas = [row["alpha"] for row in rows]
        checks[f"pair_selection_P0:{repetition}"] = all(
            row["lambda_selection_pairs"] == 25000 and row["validation_pairs"] == 25000
            for row in rows
        )
        checks[f"pair_fixed_alpha:{repetition}"] = (
            len(alphas) == len(spec["sweeps"]["sampled_pairs"])
            and all(alpha == alphas[0] for alpha in alphas)
        )
    checks["pair_sampler_contract"] = read_json(
        out / "pair_sampler_contract_audit.json"
    )["passed"]
    checks["source_protocol_preserved"] = read_json(
        out / "source_protocol_audit.json"
    )["passed"]
    provenance = read_json(out / "provenance.json")
    checks["protected_paths_unchanged"] = provenance["protected_paths_unchanged"]
    checks["current_frozen_manuscript_P10_unmodified"] = (
        manifest.get("current_frozen_manuscript_P10_modified") is False
    )
    checks["summary_recomputed"] = summarize(records) == read_json(out / "summary.json")
    for name, expected in manifest.get("artifact_sha256", {}).items():
        checks[f"artifact:{name}"] = digest(out / name) == expected
    source_hashes = read_json(out / "source_manifest.json")["sha256"]
    checks["source_file_set"] = set(source_hashes) == set(source_files(requested))
    for relative, expected in source_hashes.items():
        checks[f"source:{relative}"] = digest(ROOT / relative) == expected
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failed, "failed_checks": failed,
        "check_count": len(checks), "record_count": len(records),
        "expected_record_count": expected_records,
    }


def run(config_path, out, *, workers=1, resume=False):
    requested = read_json(config_path)
    source_frozen = read_json(ROOT / requested["authoritative_frozen_protocol"])
    source_requested = read_json(ROOT / requested["authoritative_requested_protocol"])
    source_manifest = read_json(ROOT / requested["authoritative_source_manifest"])
    validate_requested(requested, source_frozen, source_requested, source_manifest)
    spec = effective_protocol(requested, source_frozen)
    out = Path(out)
    if out.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {out}")
    if not out.exists():
        out.mkdir(parents=True)
        write_json(out / "requested_protocol.json", requested)
        write_json(out / "frozen_protocol.json", spec)
        write_json(out / "pair_sampler_contract_audit.json", pair_sampler_contract_audit())
        write_json(out / "source_protocol_audit.json", source_protocol_audit(source_frozen, spec))
        start_inventory = path_inventory(requested["protected_paths"])
        write_json(out / "provenance_start.json", {
            "created_utc": utc_now(), "git_status": git_status(),
            "protected_paths": start_inventory,
        })
        source_hashes = snapshot_sources(out, requested)
        write_json(out / "manifest.json", {
            "task": spec["task"], "status": "running", "workers": workers,
            "authoritative_frozen_protocol": requested["authoritative_frozen_protocol"],
            "source_sha256": source_hashes, "jobs": [],
            "current_frozen_manuscript_P10_modified": False,
            "created_utc": utc_now(),
        })
    else:
        manifest = read_json(out / "manifest.json")
        current = {
            relative: digest(ROOT / relative) for relative in source_files(requested)
        }
        if manifest["source_sha256"] != current:
            raise ValueError("resume source hash mismatch")
        if read_json(out / "frozen_protocol.json") != spec:
            raise ValueError("resume protocol mismatch")

    plans = repetition_plan(spec)
    jobs_dir = out / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    completed = {
        int(path.stem.split("-")[-1]) for path in jobs_dir.glob("repetition-*.json")
    }
    pending = [plan for plan in plans if plan["index"] not in completed]
    payloads = [(str(out), spec, plan) for plan in pending]
    started = perf_counter()
    if workers == 1:
        generated, executor = map(run_repetition, payloads), None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        generated = executor.map(run_repetition, payloads)
    try:
        for result in generated:
            write_json(jobs_dir / f"repetition-{result['index']:03d}.json", result)
            print(f"localized10d P50 repetition {result['index'] + 1}/{len(plans)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    jobs = [read_json(path) for path in sorted(jobs_dir.glob("repetition-*.json"))]
    records = [row for job in jobs for row in job["records"]]
    records.sort(key=lambda row: (
        SWEEPS.index(row["sweep"]), float(row["sweep_value"]),
        METHODS.index(row["method"]), int(row["repetition"]),
    ))
    summary = summarize(records)
    write_json(out / "records.json", records)
    write_csv(out / "records.csv", records)
    write_json(out / "summary.json", summary)
    figure_paths = plot_robustness(summary, out)
    figure_paths.extend(plot_nominal_profile(spec, records, out))
    trajectory_paths, trajectory_report = plot_nominal_trajectory(spec, records, out)
    figure_paths.extend(trajectory_paths)
    table_metrics = build_table_metrics(spec, records, summary, trajectory_report)
    write_table_files(out, table_metrics)
    quality = table_metrics["diagnostics"]
    write_json(out / "quality_report.json", quality)

    start_provenance = read_json(out / "provenance_start.json")
    end_inventory = path_inventory(requested["protected_paths"])
    unchanged = start_provenance["protected_paths"] == end_inventory
    provenance = {
        "authoritative_protocol": requested["authoritative_frozen_protocol"],
        "authoritative_protocol_sha256": requested["authoritative_frozen_protocol_sha256"],
        "derivation": source_protocol_audit(source_frozen, spec),
        "pair_sampler_contract": read_json(out / "pair_sampler_contract_audit.json"),
        "protected_paths_start": start_provenance["protected_paths"],
        "protected_paths_end": end_inventory,
        "protected_paths_unchanged": unchanged,
        "git_status_at_start": start_provenance["git_status"],
        "git_status_at_end": git_status(),
        "completed_utc": utc_now(),
    }
    write_json(out / "provenance.json", provenance)
    if not unchanged:
        raise RuntimeError("a protected current/frozen/manuscript/P10 path changed during the run")

    artifact_names = [
        "requested_protocol.json", "frozen_protocol.json", "source_manifest.json",
        "source_protocol_audit.json", "pair_sampler_contract_audit.json",
        "provenance_start.json", "provenance.json", "records.json", "records.csv",
        "summary.json", "quality_report.json", "table_metrics.json",
        "table_metrics.csv", "table_metrics.md", "nominal_trajectory_report.json",
        "nominal_trajectory_rep000.npz",
        *[path.name for path in figure_paths],
    ]
    manifest = read_json(out / "manifest.json")
    manifest.update({
        "status": "production_candidate_completed",
        "jobs": [job["index"] for job in jobs], "record_count": len(records),
        "artifact_sha256": {name: digest(out / name) for name in artifact_names},
        "elapsed_seconds_this_invocation": perf_counter() - started,
        "updated_utc": utc_now(),
    })
    write_json(out / "manifest.json", manifest)
    verification = verify_output(out)
    write_json(out / "verification_report.json", verification)
    manifest["verification_passed"] = verification["passed"]
    manifest["verification_report_sha256"] = digest(out / "verification_report.json")
    write_json(out / "manifest.json", manifest)
    if not verification["passed"]:
        raise RuntimeError(f"production verification failed: {verification['failed_checks']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.verify_only is not None:
        result = verify_output(args.verify_only)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
        return
    if args.resume and args.out_dir is None:
        parser.error("--resume requires --out-dir")
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out_dir = (
            ROOT / "outputs/validation/localized10d_sampled_pair_production"
            / f"P50-R20-{stamp}"
        )
    result = run(args.config.resolve(), args.out_dir, workers=args.workers, resume=args.resume)
    print(json.dumps({
        "status": result["status"], "verification_passed": result["verification_passed"],
        "output": str(Path(args.out_dir).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
