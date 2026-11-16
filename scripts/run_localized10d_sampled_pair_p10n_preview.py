"""Run a two-repetition sampled-pair preview for the localized 10D study.

This preview leaves the existing scientific runner, frozen outputs, manuscript,
and current figures untouched.  Standard conditions use P=10N.  The pair-count
sweep selects LASSO at nominal P=5000 using an independent sampled validation
system and fixes that absolute alpha across all requested pair counts.
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
from scripts.run_localized10d_observation_noise import (
    assemble_with_fixed_tests,
    fixed_space_test_spec,
    observed_trajectory,
    validate_protocol as validate_observation_protocol,
)
from scripts.run_localized10d_pilot import (
    _augment_record,
    validate_protocol as validate_base_protocol,
)
from scripts.run_morse10d_sweeps import (
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

DEFAULT_CONFIG = ROOT / "configs/experiments/localized10d_sampled_pair_p10n_preview.json"
SOURCE_FILES = (
    "configs/experiments/localized10d_sampled_pair_p10n_preview.json",
    "configs/experiments/localized10d_scientific.json",
    "configs/experiments/localized10d_observation_noise_001.json",
    "outputs/validation/localized10d_noisy_robustness/sweeps-20261023T123949Z/requested_protocol.json",
    "outputs/validation/localized10d_noisy_robustness/sweeps-20261023T123949Z/frozen_protocol.json",
    "outputs/validation/localized10d_noisy_robustness/sweeps-20261023T123949Z/source_manifest.json",
    "outputs/validation/localized10d_noisy_robustness/sweeps-20261023T123949Z/source/scripts/run_localized10d_noisy_robustness.py",
    "outputs/validation/localized10d_noisy_robustness/sweeps-20261023T123949Z/source/scripts/run_localized10d_observation_noise.py",
    "scripts/run_localized10d_sampled_pair_p10n_preview.py",
    "scripts/run_localized10d_noisy_robustness.py",
    "scripts/run_localized10d_observation_noise.py",
    "scripts/run_localized10d_pilot.py",
    "scripts/run_morse10d_sweeps.py",
    "scripts/run_1d_hat_lasso.py",
    "test/test_localized10d_sampled_pair_p10n_preview.py",
    "features.py",
    "kernels.py",
    "pipeline.py",
    "regression.py",
    "simulate.py",
    "tests.py",
    "time_tests.py",
    "utils.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_status():
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-uall"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def _function_ast_digest(path, names):
    tree = ast.parse(Path(path).read_text())
    functions = {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }
    if set(functions) != set(names):
        missing = sorted(set(names) - set(functions))
        raise ValueError(f"missing audited functions in {path}: {missing}")
    return hashlib.sha256(
        json.dumps(functions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_requested(requested, base, observation, source_requested, source_frozen,
                       source_manifest):
    if requested.get("task") != "localized10d_sampled_pair_p10n_preview":
        raise ValueError("invalid sampled-pair preview task")
    if requested.get("protocol_version") != 2 or requested.get("preview_only") is not True:
        raise ValueError("sampled-pair preview designation changed")
    if int(requested.get("repetitions", -1)) != 2:
        raise ValueError("preview must use exactly two repetitions")
    validate_base_protocol(base)
    validate_observation_protocol(observation, base)
    for path_key, hash_key in (
        ("base_config", "base_config_sha256"),
        ("observation_config", "observation_config_sha256"),
        ("source_requested_protocol", "source_requested_protocol_sha256"),
        ("source_frozen_protocol", "source_frozen_protocol_sha256"),
        ("source_manifest", "source_manifest_sha256"),
        ("audited_runner", "audited_runner_sha256"),
        ("audited_frozen_runner", "audited_frozen_runner_sha256"),
        ("active_observation_runner", "active_observation_runner_sha256"),
        ("audited_frozen_observation_runner", "audited_frozen_observation_runner_sha256"),
    ):
        path = ROOT / requested[path_key]
        if not path.is_file():
            raise FileNotFoundError(path)
        if digest(path) != requested[hash_key]:
            raise ValueError(f"audited input changed: {requested[path_key]}")

    if source_requested.get("observation_noise") != 0.01:
        raise ValueError("authoritative requested protocol is not sigma_obs=0.01")
    if source_frozen.get("observation_noise") != 0.01:
        raise ValueError("authoritative frozen protocol is not sigma_obs=0.01")
    if float(requested.get("observation_noise", -1.0)) != 0.01:
        raise ValueError("preview observation noise must remain 0.01")
    if observation.get("observation_noise_values") != [0.01]:
        raise ValueError("observation config is not the fixed sigma_obs=0.01 protocol")
    if source_requested.get("nominal_config") != requested["observation_config"]:
        raise ValueError("authoritative requested protocol uses a different nominal config")
    if source_requested.get("nominal_config_sha256") != requested["observation_config_sha256"]:
        raise ValueError("authoritative nominal config hash differs")
    if source_frozen.get("observation_noise_seeds") != observation.get(
        "observation_noise_seeds"
    ):
        raise ValueError("authoritative observation-noise seeds differ")

    for section in ("model", "dictionary", "weak_tests", "sweeps", "estimator", "seeds"):
        if source_frozen.get(section) != base.get(section):
            raise ValueError(f"authoritative frozen {section} differs from base config")
    if source_manifest.get("sha256", {}).get("scripts/run_localized10d_noisy_robustness.py") \
            != requested["audited_frozen_runner_sha256"]:
        raise ValueError("authoritative source manifest runner hash differs")
    if source_manifest.get("sha256", {}).get(
        "configs/experiments/localized10d_observation_noise_001.json"
    ) != requested["observation_config_sha256"]:
        raise ValueError("authoritative source manifest observation config hash differs")

    numerical_helpers = {
        "observed_trajectory", "fixed_space_test_spec", "assemble_with_fixed_tests"
    }
    active_helper_digest = _function_ast_digest(
        ROOT / requested["active_observation_runner"], numerical_helpers
    )
    frozen_helper_digest = _function_ast_digest(
        ROOT / requested["audited_frozen_observation_runner"], numerical_helpers
    )
    if active_helper_digest != frozen_helper_digest:
        raise ValueError("active observation numerical helpers differ from frozen source")

    if requested.get("pair_sampling") != {
        "mode": "resample_each_time",
        "with_replacement": True,
        "exclude_diagonal": True,
        "pairs_per_particle": 10,
        "nominal_particles": 500,
        "nominal_pairs": 5000,
    }:
        raise ValueError("sampled-pair P=10N protocol changed")
    if requested.get("pair_count_penalty_policy") != {
        "selection_pairs": 5000,
        "selection_validation": "independent_sampled_validation_system_at_nominal_P",
        "reuse": "fix_selected_absolute_alpha_across_all_pair_counts_within_repetition",
    }:
        raise ValueError("pair-count penalty policy changed")


def effective_protocol(requested, source_frozen):
    spec = json.loads(json.dumps(source_frozen))
    spec["source_task"] = source_frozen["task"]
    spec["source_frozen_protocol"] = requested["source_frozen_protocol"]
    spec["source_frozen_protocol_sha256"] = requested["source_frozen_protocol_sha256"]
    spec["task"] = requested["task"]
    spec["protocol_version"] = requested["protocol_version"]
    spec["preview_only"] = True
    spec["evaluation"]["scientific_repetitions"] = int(requested["repetitions"])
    spec["pair_sampling"] = {
        "baseline_pairs": int(requested["pair_sampling"]["nominal_pairs"]),
        "mode": requested["pair_sampling"]["mode"],
        "with_replacement": requested["pair_sampling"]["with_replacement"],
        "exclude_diagonal": requested["pair_sampling"]["exclude_diagonal"],
        "pairs_per_particle": int(requested["pair_sampling"]["pairs_per_particle"]),
    }
    spec["pair_penalty_policy"] = (
        "absolute_alpha_selected_at_nominal_P5000_by_independent_sampled_validation_"
        "then_fixed_across_all_P_within_repetition"
    )
    spec["pair_count_penalty_policy"] = requested["pair_count_penalty_policy"]
    spec["pairing_policy"] = requested["pair_index_policy"]
    spec["pair_index_policy"] = requested["pair_index_policy"]
    spec["condition_budget_policy"] = requested["condition_budget_policy"]
    spec["selection_policy"] = requested["selection_policy"]
    spec["source_preservation_policy"] = requested["source_preservation_policy"]
    spec["supersedes_preview_output"] = requested["supersedes_preview_output"]
    spec["data_policy"] = (
        "Preview-only R=2 derivative of the authoritative 123949Z sigma_obs=0.01 "
        "protocol. Training, validation, and latent test trajectories remain separate; "
        "only pair sampling budgets/mode and pair-count alpha reuse are changed."
    )
    spec["output_policy"] = requested["output_policy"]
    return spec


def pair_budget(spec, particles):
    return int(spec["pair_sampling"]["pairs_per_particle"]) * int(particles)


def _noise_seed(spec, plan, split):
    seeds = spec["observation_noise_seeds"]
    return int(seeds[f"{split}_start"] + seeds["repetition_stride"] * plan["index"])


def _add_observation_noise(spec, latent, particles, plan, split):
    if split == "test":
        return np.asarray(latent, dtype=float), None
    seed = _noise_seed(spec, plan, split)
    latent = np.asarray(latent, dtype=float)
    max_particles = int(spec["model"]["particles"])
    full_shape = (latent.shape[0], latent.shape[1], max_particles, latent.shape[3])
    standard_normal = np.random.default_rng(seed).normal(size=full_shape)
    return observed_trajectory(
        latent,
        spec["observation_noise"],
        standard_normal[:, :, :particles, :],
    ), seed


def _assemble(spec, trajectory, time, particles, sigma, plan, split, pairs, tests):
    return assemble_with_fixed_tests(
        spec,
        trajectory,
        time,
        particles,
        sigma,
        plan[f"{split}_seed"],
        plan[f"{split}_pair_seed"],
        plan["space_seed"],
        pairs,
        tests,
    )


def _tag_standard_records(records, spec, particles, sigma):
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
        50,
        rng=np.random.default_rng(8128),
        with_replacement=True,
        exclude_diagonal=True,
        resample_each_call=True,
    )
    first = sampler(4)
    second = sampler(4)
    first_pairs = list(zip(first[0].tolist(), first[1].tolist()))

    calls = []

    def recording_sampler(n):
        call = len(calls)
        i = np.array([(call + k) % n for k in range(5)], dtype=int)
        j = (i + 1) % n
        calls.append((i.copy(), j.copy()))
        return i, j

    X = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4, 1) / 10.0
    t = np.array([0.0, 0.5, 1.0])
    test_fns = {
        "phi": lambda x: np.ones((x.shape[0], 2), dtype=float),
        "grad_phi": lambda x: np.ones((x.shape[0], 2, 1), dtype=float),
    }

    def basis(z):
        return np.repeat(z[:, None, :], 3, axis=1)

    compute_A_timeweak(
        X,
        t,
        test_fns,
        basis,
        recording_sampler,
        {"psi": np.ones((2, 3), dtype=float)},
        weights=np.ones(3, dtype=float),
    )
    checks = {
        "ordered_off_diagonal": bool(
            np.all(first[0] != first[1]) and np.all(second[0] != second[1])
        ),
        "with_replacement": len(set(first_pairs)) < len(first_pairs),
        "independent_between_time_calls": bool(
            not (np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1]))
        ),
        "one_pair_draw_shared_across_all_tests_and_atoms_per_trajectory_time": (
            len(calls) == X.shape[0] * X.shape[1]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def source_protocol_preservation_audit(source_frozen, spec):
    checks = {
        section: spec.get(section) == source_frozen.get(section)
        for section in ("model", "dictionary", "weak_tests", "sweeps", "estimator", "seeds")
    }
    checks.update({
        "observation_noise": spec.get("observation_noise") == source_frozen.get(
            "observation_noise"
        ) == 0.01,
        "observation_noise_seeds": spec.get("observation_noise_seeds")
        == source_frozen.get("observation_noise_seeds"),
        "evaluation_except_repetitions": {
            key: value for key, value in spec["evaluation"].items()
            if key != "scientific_repetitions"
        } == {
            key: value for key, value in source_frozen["evaluation"].items()
            if key != "scientific_repetitions"
        },
        "preview_repetitions": spec["evaluation"]["scientific_repetitions"] == 2,
        "pair_budget_changed_to_p10n": spec["pair_sampling"] == {
            "baseline_pairs": 5000,
            "mode": "resample_each_time",
            "with_replacement": True,
            "exclude_diagonal": True,
            "pairs_per_particle": 10,
        },
        "pair_penalty_changed_to_nominal_p5000": spec["pair_count_penalty_policy"] == {
            "selection_pairs": 5000,
            "selection_validation": "independent_sampled_validation_system_at_nominal_P",
            "reuse": "fix_selected_absolute_alpha_across_all_pair_counts_within_repetition",
        },
    })
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "authoritative_observation_noise": source_frozen["observation_noise"],
        "authoritative_source_task": source_frozen["task"],
        "scientific_changes": [
            "pair_sampling.baseline_pairs: 10000 -> 5000",
            "pair_sampling.mode: nested_resample_each_time -> resample_each_time",
            "particle-number and standard-condition pair budget: fixed P=10000 -> P=10N",
            "pair-count alpha selection: Pmax=30000 -> nominal P=5000",
        ],
        "preview_only_changes": [
            "task/provenance metadata",
            "evaluation.scientific_repetitions: 20 -> 2",
            "preview output policy",
        ],
    }


def audit_existing_protocol(requested, source_requested, source_frozen, source_manifest):
    helper_names = {
        "observed_trajectory", "fixed_space_test_spec", "assemble_with_fixed_tests"
    }
    active_helper_digest = _function_ast_digest(
        ROOT / requested["active_observation_runner"], helper_names
    )
    frozen_helper_digest = _function_ast_digest(
        ROOT / requested["audited_frozen_observation_runner"], helper_names
    )
    expected = {
        "authoritative_run": "sweeps-20261023T123949Z",
        "observation_noise": 0.01,
        "mode": "resample_each_time",
        "nominal_pairs": 5000,
        "particle_budget": "P=10N",
        "pair_count_lambda_selection_pairs": 5000,
    }
    findings = {
        "source_requested_protocol_sha256": digest(
            ROOT / requested["source_requested_protocol"]
        ),
        "source_frozen_protocol_sha256": digest(ROOT / requested["source_frozen_protocol"]),
        "source_manifest_sha256": digest(ROOT / requested["source_manifest"]),
        "source_observation_noise": source_frozen["observation_noise"],
        "source_pair_mode": source_frozen["pair_sampling"]["mode"],
        "source_baseline_pairs": source_frozen["pair_sampling"]["baseline_pairs"],
        "source_pair_count_penalty_policy": source_frozen["pair_penalty_policy"],
        "source_nominal_config": source_requested["nominal_config"],
        "source_manifest_runner_sha256": source_manifest["sha256"][
            "scripts/run_localized10d_noisy_robustness.py"
        ],
        "active_and_frozen_numerical_helper_ast_sha256": active_helper_digest,
        "active_numerical_helpers_match_frozen": active_helper_digest
        == frozen_helper_digest,
        "superseded_preview_output": requested["supersedes_preview_output"],
    }
    issues = [
        "The current 123949Z protocol uses baseline P=10000, which differs from the preview policy's nominal P=5000.",
        "The current particle-number sweep fixes P=10000, which differs from the preview policy P=10N.",
        "The current process-noise condition uses P=10000, which differs from the preview policy P=5000 for N=500.",
        "The current pair mode is nested_resample_each_time, which differs from the preview policy resample_each_time.",
        "The current pair-count sweep selects lambda at Pmax=30000, whereas the preview policy selects it at nominal P=5000.",
    ]
    passed = (
        source_frozen["observation_noise"] == 0.01
        and active_helper_digest == frozen_helper_digest
        and findings["source_manifest_runner_sha256"]
        == requested["audited_frozen_runner_sha256"]
    )
    return {
        "passed": passed,
        "expected_preview_protocol": expected,
        "findings": findings,
        "pair_policy_differences": issues,
        "existing_file_change_required_for_main_runner": True,
        "existing_files_modified": False,
        "superseded_output_modified": False,
        "resolution": (
            "Retained the current 123949Z protocol as the authoritative source and derived only the pair policy for the new preview."
        ),
    }


def _failed_lasso_record(A, common):
    return {
        **common,
        "method": "lasso",
        "coefficients": None,
        "nonzero_reported": None,
        **design_diagnostics(A),
        "converged": False,
        "unconverged_candidates": 1,
        "selected_at_grid_boundary": False,
        "alpha": None,
        "alpha_index": None,
        "alpha_over_max": None,
        "iterations": None,
        "fit_seconds": 0.0,
        "validation_weak_residual_mse": None,
        "latent_drift_relative_error": None,
        "profile_relative_error": None,
        "numerical_divergence": True,
        "ols_lstsq_rank": None,
        "ols_smallest_singular_value": None,
    }


def run_repetition(payload):
    out_string, spec, plan = payload
    out = Path(out_string)
    records, selections, trajectory_status = [], {}, []
    base_n = int(spec["model"]["particles"])
    base_sigma = float(spec["model"]["sigma"])
    target = np.asarray(spec["model"]["interaction"]["coefficients"], dtype=float)
    baseline_data = None

    with threadpool_limits(limits=1):
        for particles, sigma in physical_conditions(spec):
            split_data = {}
            simulation_started = perf_counter()
            for split in ("training", "validation", "test"):
                X, time, cached = load_or_simulate(out, spec, particles, sigma, split, plan)
                split_data[split] = (X, time)
                trajectory_status.append({
                    "particles": particles,
                    "sigma": sigma,
                    "split": split,
                    "cache_hit": cached,
                })
            simulation_seconds = perf_counter() - simulation_started
            stride = int(spec["model"]["observation_stride"])
            sampled = {
                split: (data[0][:, ::stride], data[1][::stride])
                for split, data in split_data.items()
            }
            observed, noise_seeds = {}, {}
            for split, (X, time) in sampled.items():
                observed_X, noise_seed = _add_observation_noise(
                    spec, X, particles, plan, split
                )
                observed[split] = (observed_X, time)
                noise_seeds[split] = noise_seed
            training_tests = fixed_space_test_spec(
                spec, observed["training"][0], plan["space_seed"]
            )
            validation_tests = fixed_space_test_spec(
                spec, observed["validation"][0], plan["space_seed"]
            )
            pairs = pair_budget(spec, particles)
            train_A, train_b, train_timing = _assemble(
                spec, *observed["training"], particles, sigma, plan,
                "training", pairs, training_tests,
            )
            validation_A, validation_b, validation_timing = _assemble(
                spec, *observed["validation"], particles, sigma, plan,
                "validation", pairs, validation_tests,
            )
            evaluation_system = latent_evaluation_system(
                spec, split_data["test"][0], plan["evaluation_seed"]
            )
            fit_id = f"rep{plan['index']:03d}-N{particles}-sigma{sigma:g}-P{pairs}"
            common = {
                "repetition": plan["index"],
                "fit_id": fit_id,
                "particles": particles,
                "sigma": sigma,
                "observation_noise": float(spec["observation_noise"]),
                "pairs": pairs,
                "pair_budget_rule": "10N",
                "pair_mode": spec["pair_sampling"]["mode"],
                "training_seed": plan["training_seed"],
                "validation_seed": plan["validation_seed"],
                "test_seed": plan["test_seed"],
                "training_observation_noise_seed": noise_seeds["training"],
                "validation_observation_noise_seed": noise_seeds["validation"],
                "training_pair_seed": plan["training_pair_seed"],
                "validation_pair_seed": plan["validation_pair_seed"],
                "simulation_and_cache_seconds": simulation_seconds,
                "training_assembly_seconds": train_timing["assembly_seconds"],
                "validation_assembly_seconds": validation_timing["assembly_seconds"],
            }
            fitted, paths = method_records(
                train_A,
                train_b,
                validation_A,
                validation_b,
                evaluation_system,
                spec["estimator"],
                common,
            )
            fitted = [_augment_record(row, target) for row in fitted]
            records.extend(_tag_standard_records(fitted, spec, particles, sigma))
            selections[fit_id] = paths
            if particles == base_n and sigma == base_sigma:
                baseline_data = {
                    "observed": observed,
                    "evaluation_system": evaluation_system,
                    "training_tests": training_tests,
                    "validation_tests": validation_tests,
                    "train_A": train_A,
                    "train_b": train_b,
                    "train_timing": train_timing,
                    "validation_A": validation_A,
                    "validation_b": validation_b,
                    "validation_timing": validation_timing,
                }

        if baseline_data is None:
            raise RuntimeError("nominal N=500, sigma=0.05 condition was not generated")
        nominal_pairs = int(spec["pair_count_penalty_policy"]["selection_pairs"])
        selected_theta, selection, selected_info = select_raw_lasso(
            baseline_data["train_A"],
            baseline_data["train_b"],
            baseline_data["validation_A"],
            baseline_data["validation_b"],
            spec["estimator"],
        )
        selections[f"rep{plan['index']:03d}-P{nominal_pairs}-lambda-selection"] = {
            "lasso": selection
        }
        alpha = selected_info["alpha"]
        for pairs in spec["sweeps"]["sampled_pairs"]:
            if int(pairs) == nominal_pairs:
                A = baseline_data["train_A"]
                b = baseline_data["train_b"]
                train_timing = baseline_data["train_timing"]
            else:
                A, b, train_timing = _assemble(
                    spec,
                    *baseline_data["observed"]["training"],
                    base_n,
                    base_sigma,
                    plan,
                    "training",
                    int(pairs),
                    baseline_data["training_tests"],
                )
            common = {
                "repetition": plan["index"],
                "fit_id": f"rep{plan['index']:03d}-P{pairs}",
                "sweep": "sampled_pairs",
                "sweep_value": int(pairs),
                "particles": base_n,
                "sigma": base_sigma,
                "observation_noise": float(spec["observation_noise"]),
                "pairs": int(pairs),
                "pair_budget_rule": "declared_pair_count_sweep_value",
                "pair_mode": spec["pair_sampling"]["mode"],
                "training_seed": plan["training_seed"],
                "validation_seed": plan["validation_seed"],
                "test_seed": plan["test_seed"],
                "training_observation_noise_seed": _noise_seed(spec, plan, "training"),
                "validation_observation_noise_seed": _noise_seed(spec, plan, "validation"),
                "training_pair_seed": plan["training_pair_seed"],
                "validation_pair_seed": plan["validation_pair_seed"],
                "lambda_selection_pairs": nominal_pairs,
                "validation_pairs": nominal_pairs,
                "fixed_absolute_alpha": alpha,
                "simulation_and_cache_seconds": 0.0,
                "training_assembly_seconds": train_timing["assembly_seconds"],
                "validation_assembly_seconds": baseline_data["validation_timing"][
                    "assembly_seconds"
                ],
                "pair_design_relative_difference_from_nominal_p5000": float(
                    np.linalg.norm(A - baseline_data["train_A"])
                    / max(np.linalg.norm(baseline_data["train_A"]), 1e-300)
                ),
            }
            if alpha is None:
                failed = _failed_lasso_record(A, common)
                both, _ = method_records(
                    A,
                    b,
                    baseline_data["validation_A"],
                    baseline_data["validation_b"],
                    baseline_data["evaluation_system"],
                    spec["estimator"],
                    common,
                    fixed_alpha=1.0,
                )
                fitted = [failed] + [row for row in both if row["method"] == "ols"]
            else:
                fitted, _ = method_records(
                    A,
                    b,
                    baseline_data["validation_A"],
                    baseline_data["validation_b"],
                    baseline_data["evaluation_system"],
                    spec["estimator"],
                    common,
                    fixed_alpha=alpha,
                    initial_theta=selected_theta,
                )
            records.extend(_augment_record(row, target) for row in fitted)

        for particles, sigma in physical_conditions(spec):
            for split in ("training", "validation", "test"):
                _trajectory_path(out, particles, sigma, split, plan["index"]).unlink(
                    missing_ok=True
                )
    return {
        "index": plan["index"],
        "records": records,
        "selections": selections,
        "trajectory_cache": trajectory_status,
        "pair_count_selected_alpha": alpha,
    }


def _finite_stats(values):
    present = np.asarray([value for value in values if value is not None], dtype=float)
    finite = present[np.isfinite(present)]
    return {
        "count": len(values),
        "present_count": int(present.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "minimum": float(np.min(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def summarize(records):
    groups = []
    for sweep in ("process_sigma", "particle_number", "sampled_pairs"):
        values = sorted({float(row["sweep_value"]) for row in records if row["sweep"] == sweep})
        for value in values:
            for method in ("lasso", "ols"):
                rows = [
                    row for row in records
                    if row["sweep"] == sweep
                    and float(row["sweep_value"]) == value
                    and row["method"] == method
                ]
                groups.append({
                    "sweep": sweep,
                    "value": value,
                    "method": method,
                    "repetitions": len(rows),
                    "pairs": sorted({int(row["pairs"]) for row in rows}),
                    "coefficient_error_relative": _finite_stats(
                        [row["coefficient_error_relative"] for row in rows]
                    ),
                    "latent_drift_relative_error": _finite_stats(
                        [row["latent_drift_relative_error"] for row in rows]
                    ),
                    "profile_relative_error": _finite_stats(
                        [row["profile_relative_error"] for row in rows]
                    ),
                    "rank_deficient": int(sum(bool(row["rank_deficient"]) for row in rows)),
                    "nonconverged": int(sum(not bool(row["converged"]) for row in rows)),
                    "boundary_selections": int(sum(
                        bool(row["selected_at_grid_boundary"]) for row in rows
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


def snapshot_sources(out):
    hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    write_json(out / "source_manifest.json", {"sha256": hashes})
    return hashes


def generate_error_preview(run_dir, preview_dir=None, *, overwrite=False):
    """Plot existing preview records without simulating, assembling, or refitting."""
    run_dir = Path(run_dir)
    records_path = run_dir / "records.json"
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    records = read_json(records_path)
    if not records:
        raise ValueError("records.json is empty")
    methods = ("lasso", "ols")
    if {row.get("method") for row in records} != set(methods):
        raise ValueError("preview records must contain exactly LASSO and OLS")

    preview_dir = Path(preview_dir) if preview_dir is not None else run_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    stem = "localized10d_sampled_pair_p10n_error_preview"
    png_path = preview_dir / f"{stem}.png"
    pdf_path = preview_dir / f"{stem}.pdf"
    report_path = preview_dir / "figure_report.json"
    existing = [path for path in (png_path, pdf_path, report_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite preview artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    observation_noises = sorted({float(row["observation_noise"]) for row in records})
    if len(observation_noises) != 1:
        raise ValueError("preview records must contain one observation-noise level")
    observation_noise = observation_noises[0]

    colors = {"lasso": "#0072B2", "ols": "#D55E00"}
    labels = {"lasso": "LASSO", "ols": "OLS"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)

    nominal = [
        row for row in records
        if row["sweep"] == "process_sigma"
        and np.isclose(float(row["sweep_value"]), 0.05)
    ]
    metric_specs = (
        ("coefficient_error_relative", "coefficient"),
        ("latent_drift_relative_error", "latent drift"),
        ("profile_relative_error", "radial profile"),
    )
    axis = axes[0, 0]
    x = np.arange(len(metric_specs), dtype=float)
    width = 0.36
    nominal_values = {}
    for method_index, method in enumerate(methods):
        method_rows = sorted(
            [row for row in nominal if row["method"] == method],
            key=lambda row: row["repetition"],
        )
        if not method_rows:
            raise ValueError(f"nominal records missing for {method}")
        values_by_metric = []
        for metric, _ in metric_specs:
            values = np.asarray([row[metric] for row in method_rows], dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"nominal {metric} contains nonpositive/nonfinite values")
            values_by_metric.append(values)
            nominal_values[f"{method}:{metric}"] = values.tolist()
        means = [float(np.mean(values)) for values in values_by_metric]
        positions = x + (method_index - 0.5) * width
        axis.bar(
            positions, means, width=width, color=colors[method], alpha=0.75,
            label=labels[method],
        )
        for metric_index, values in enumerate(values_by_metric):
            offsets = np.linspace(-0.045, 0.045, values.size)
            axis.scatter(
                np.full(values.size, positions[metric_index]) + offsets,
                values,
                color=colors[method], edgecolor="white", linewidth=0.5,
                s=28, zorder=3,
            )
    axis.set_xticks(x, [label for _, label in metric_specs])
    axis.set_yscale("log")
    axis.set_ylabel("relative error")
    axis.set_title(
        rf"(a) Nominal: $N=500$, $P=5000$, $\sigma=0.05$, "
        rf"$\sigma_{{\rm obs}}={observation_noise:g}$"
    )
    axis.grid(True, which="both", axis="y", alpha=0.25)
    axis.legend(frameon=False)

    sweep_specs = (
        ("process_sigma", axes[0, 1], "(b) Process-noise sweep", r"process noise $\sigma$", False),
        ("particle_number", axes[1, 0], "(c) Particle-number sweep", r"particles $N$", True),
        ("sampled_pairs", axes[1, 1], "(d) Pair-count sweep", r"sampled pairs $P$", True),
    )
    sweep_values = {}
    for sweep, axis, title, xlabel, log_x in sweep_specs:
        values = sorted({
            float(row["sweep_value"]) for row in records if row["sweep"] == sweep
        })
        if not values:
            raise ValueError(f"records missing sweep: {sweep}")
        for method in methods:
            method_rows = [
                row for row in records
                if row["sweep"] == sweep and row["method"] == method
            ]
            repetitions = sorted({int(row["repetition"]) for row in method_rows})
            per_repetition = []
            for repetition in repetitions:
                lookup = {
                    float(row["sweep_value"]): float(row["coefficient_error_relative"])
                    for row in method_rows if int(row["repetition"]) == repetition
                }
                if set(lookup) != set(values):
                    raise ValueError(
                        f"incomplete {sweep}/{method} records for repetition {repetition}"
                    )
                y = np.asarray([lookup[value] for value in values], dtype=float)
                if not np.all(np.isfinite(y)) or np.any(y <= 0.0):
                    raise ValueError(
                        f"{sweep}/{method} contains nonpositive/nonfinite coefficient errors"
                    )
                per_repetition.append(y)
                axis.plot(
                    values, y, color=colors[method], alpha=0.22,
                    linewidth=0.8, marker="o", markersize=3,
                )
            matrix = np.vstack(per_repetition)
            mean = matrix.mean(axis=0)
            sweep_values[f"{sweep}:{method}"] = {
                "values": values,
                "individual": matrix.tolist(),
                "mean": mean.tolist(),
            }
            axis.plot(
                values, mean, color=colors[method], linewidth=2.0,
                marker="o" if method == "ols" else "^", markersize=5,
                label=labels[method],
            )
        if log_x:
            axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("relative coefficient error")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)

    fig.suptitle(
        "Localized 10D sampled-pair preview (existing records only)\n"
        "Preview only, R=2: bars/solid lines are means; points/faint lines are repetitions.",
        fontsize=12,
    )
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "completed",
        "preview_only": True,
        "records_only": True,
        "numerical_recomputation": False,
        "source_records": str(records_path),
        "source_records_sha256": digest(records_path),
        "record_count": len(records),
        "repetitions": sorted({int(row["repetition"]) for row in records}),
        "nominal_definition": {
            "particles": 500,
            "pairs": 5000,
            "process_sigma": 0.05,
            "observation_noise": observation_noise,
        },
        "nominal_panel_metrics": [metric for metric, _ in metric_specs],
        "sweep_panel_metric": "coefficient_error_relative",
        "nominal_values": nominal_values,
        "sweep_values": sweep_values,
        "diagnostic_counts": {
            "nonconverged_records": sum(not bool(row["converged"]) for row in records),
            "boundary_selections": sum(
                bool(row["selected_at_grid_boundary"]) for row in records
            ),
            "rank_deficient": sum(bool(row["rank_deficient"]) for row in records),
            "numerical_divergence": sum(
                bool(row["numerical_divergence"]) for row in records
            ),
        },
        "caution": (
            "Two-repetition smoke preview only; means are descriptive and must not be "
            "used as manuscript uncertainty estimates or replace the frozen scientific run."
        ),
        "runner_sha256": digest(Path(__file__)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            png_path.name: digest(png_path),
            pdf_path.name: digest(pdf_path),
        },
    }
    write_json(report_path, report)
    return {**report, "figure_report_sha256": digest(report_path)}


def verify_output(out):
    out = Path(out)
    spec = read_json(out / "frozen_protocol.json")
    records = read_json(out / "records.json")
    manifest = read_json(out / "manifest.json")
    repetitions = int(spec["evaluation"]["scientific_repetitions"])
    expected_records = repetitions * 2 * sum(len(v) for v in spec["sweeps"].values())
    checks = {
        "status": manifest.get("status") == "preview_completed",
        "two_repetitions": repetitions == 2,
        "record_count": len(records) == expected_records == manifest.get("record_count"),
        "pair_mode": all(row.get("pair_mode") == "resample_each_time" for row in records),
        "training_validation_pair_seeds_independent": all(
            row["training_pair_seed"] != row["validation_pair_seed"] for row in records
        ),
        "observation_noise": all(
            float(row["observation_noise"]) == float(spec["observation_noise"])
            for row in records
        ),
    }
    for sweep, values in spec["sweeps"].items():
        for value in values:
            for method in ("lasso", "ols"):
                rows = [
                    row for row in records
                    if row["sweep"] == sweep
                    and float(row["sweep_value"]) == float(value)
                    and row["method"] == method
                ]
                checks[f"group:{sweep}:{value}:{method}"] = len(rows) == repetitions
                if sweep == "particle_number":
                    checks[f"budget:{sweep}:{value}:{method}"] = all(
                        int(row["pairs"]) == 10 * int(value) for row in rows
                    )
                elif sweep == "process_sigma":
                    checks[f"budget:{sweep}:{value}:{method}"] = all(
                        int(row["pairs"]) == 5000 for row in rows
                    )
                else:
                    checks[f"budget:{sweep}:{value}:{method}"] = all(
                        int(row["pairs"]) == int(value) for row in rows
                    )
    for repetition in range(repetitions):
        rows = [
            row for row in records
            if row["repetition"] == repetition
            and row["sweep"] == "sampled_pairs"
            and row["method"] == "lasso"
        ]
        alphas = [row["alpha"] for row in rows]
        checks[f"pair_alpha_selected_at_nominal:{repetition}"] = all(
            row["lambda_selection_pairs"] == 5000 and row["validation_pairs"] == 5000
            for row in rows
        )
        checks[f"pair_absolute_alpha_fixed:{repetition}"] = (
            len(alphas) == len(spec["sweeps"]["sampled_pairs"])
            and all(alpha == alphas[0] for alpha in alphas)
        )
    protocol_audit = read_json(out / "pair_sampler_contract_audit.json")
    checks["pair_sampler_contract"] = protocol_audit["passed"]
    audit = read_json(out / "existing_protocol_audit.json")
    checks["existing_protocol_audited_without_modification"] = (
        audit["passed"] and audit["existing_files_modified"] is False
    )
    preservation_path = out / "source_protocol_preservation_audit.json"
    if int(spec.get("protocol_version", 1)) >= 2:
        preservation = read_json(preservation_path)
        checks["authoritative_source_protocol_preserved"] = preservation["passed"]
        checks["authoritative_observation_noise_is_point_zero_one"] = (
            preservation["authoritative_observation_noise"] == 0.01
            and float(spec["observation_noise"]) == 0.01
        )
    for name, expected in manifest.get("artifact_sha256", {}).items():
        checks[f"artifact:{name}"] = digest(out / name) == expected
    failed = sorted(key for key, passed in checks.items() if not passed)
    return {
        "passed": not failed,
        "failed_checks": failed,
        "check_count": len(checks),
        "record_count": len(records),
    }


def run(config_path, out, *, workers=1, resume=False):
    requested = read_json(config_path)
    base = read_json(ROOT / requested["base_config"])
    observation = read_json(ROOT / requested["observation_config"])
    source_requested = read_json(ROOT / requested["source_requested_protocol"])
    source_frozen = read_json(ROOT / requested["source_frozen_protocol"])
    source_manifest = read_json(ROOT / requested["source_manifest"])
    validate_requested(
        requested, base, observation, source_requested, source_frozen, source_manifest
    )
    spec = effective_protocol(requested, source_frozen)
    out = Path(out)
    if out.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {out}")
    if not out.exists():
        out.mkdir(parents=True)
        write_json(out / "requested_protocol.json", requested)
        write_json(out / "frozen_protocol.json", spec)
        write_json(
            out / "existing_protocol_audit.json",
            audit_existing_protocol(
                requested, source_requested, source_frozen, source_manifest
            ),
        )
        write_json(
            out / "source_protocol_preservation_audit.json",
            source_protocol_preservation_audit(source_frozen, spec),
        )
        write_json(out / "pair_sampler_contract_audit.json", pair_sampler_contract_audit())
        hashes = snapshot_sources(out)
        write_json(out / "manifest.json", {
            "task": spec["task"],
            "status": "running",
            "preview_only": True,
            "authoritative_source_protocol": requested["source_frozen_protocol"],
            "authoritative_observation_noise": float(spec["observation_noise"]),
            "supersedes_preview_output": requested["supersedes_preview_output"],
            "superseded_output_modified": False,
            "source_sha256": hashes,
            "jobs": [],
            "worktree_status_at_start": _git_status(),
            "manuscript_modified": False,
            "current_figures_modified": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
    else:
        manifest = read_json(out / "manifest.json")
        current = {relative: digest(ROOT / relative) for relative in SOURCE_FILES}
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
            print(
                f"sampled-pair preview repetition {result['index'] + 1}/{len(plans)}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    jobs = [read_json(path) for path in sorted(jobs_dir.glob("repetition-*.json"))]
    records = [row for job in jobs for row in job["records"]]
    write_json(out / "records.json", records)
    write_csv(out / "records.csv", records)
    write_json(out / "summary.json", summarize(records))
    quality = {
        "record_count": len(records),
        "rank_deficient": sum(bool(row["rank_deficient"]) for row in records),
        "nonconverged_records": sum(not bool(row["converged"]) for row in records),
        "boundary_selections": sum(
            bool(row["selected_at_grid_boundary"]) for row in records
        ),
        "numerical_divergence": sum(bool(row["numerical_divergence"]) for row in records),
        "missing_coefficients": sum(row["coefficients"] is None for row in records),
    }
    write_json(out / "quality_report.json", quality)
    artifact_names = (
        "requested_protocol.json",
        "frozen_protocol.json",
        "source_manifest.json",
        "existing_protocol_audit.json",
        "source_protocol_preservation_audit.json",
        "pair_sampler_contract_audit.json",
        "records.json",
        "records.csv",
        "summary.json",
        "quality_report.json",
    )
    manifest = read_json(out / "manifest.json")
    manifest.update({
        "status": "preview_completed",
        "jobs": [job["index"] for job in jobs],
        "record_count": len(records),
        "artifact_sha256": {name: digest(out / name) for name in artifact_names},
        "elapsed_seconds_this_invocation": perf_counter() - started,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json(out / "manifest.json", manifest)
    verification = verify_output(out)
    write_json(out / "verification_report.json", verification)
    manifest["verification_passed"] = verification["passed"]
    manifest["verification_report_sha256"] = digest(out / "verification_report.json")
    write_json(out / "manifest.json", manifest)
    if not verification["passed"]:
        raise RuntimeError(f"preview verification failed: {verification['failed_checks']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--figure-only", "--plot-only", dest="figure_only", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--overwrite-preview", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.verify_only is not None and args.figure_only is not None:
        parser.error("--verify-only and --figure-only are mutually exclusive")
    if args.verify_only is not None:
        result = verify_output(args.verify_only)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
        return
    if args.figure_only is not None:
        result = generate_error_preview(
            args.figure_only,
            preview_dir=args.preview_dir,
            overwrite=args.overwrite_preview,
        )
        print(json.dumps({
            "status": result["status"],
            "records_only": result["records_only"],
            "numerical_recomputation": result["numerical_recomputation"],
            "preview": str(
                args.preview_dir if args.preview_dir is not None
                else args.figure_only / "preview"
            ),
        }))
        return
    if args.resume and args.out_dir is None:
        parser.error("--resume requires --out-dir")
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out_dir = (
            ROOT
            / "outputs/validation/localized10d_sampled_pair_p10n_preview"
            / f"preview-{stamp}"
        )
    result = run(
        args.config.resolve(), args.out_dir, workers=args.workers, resume=args.resume
    )
    print(json.dumps({
        "status": result["status"],
        "verification_passed": result["verification_passed"],
        "output": str(args.out_dir),
    }))


if __name__ == "__main__":
    main()
