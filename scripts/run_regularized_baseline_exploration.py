"""Run the current six-method weak/strong nominal comparison.

The runner preserves the declared simulation, split-seed, sampled-pair,
weak-system, strong-feature, independent-validation, and held-out-metric
protocols. Outputs are written beneath ``outputs/`` and never overwrite an
existing run.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import compute_increment_regression_system
from kernels import make_kernel_basis
from predator_prey import channel_kernels
from scripts.run_1d_hat_lasso import select_lasso
from scripts.run_cross_example_nominal_comparison import (
    assemble_localized_strong,
    assemble_predator_prey_strong,
    relative_error,
)
from scripts.run_localized10d_observation_noise import fixed_space_test_spec
from scripts.run_localized10d_sampled_pair_production import (
    add_observation_noise as localized_add_observation_noise,
    assemble as localized_assemble,
    effective_protocol as localized_effective_protocol,
)
from scripts.run_morse10d_sweeps import (
    _simulate_truth,
    design_diagnostics,
    evaluate_theta,
    latent_evaluation_system,
    repetition_plan,
)
from scripts.run_predator_prey_sampled_pair_production import (
    effective_production as predator_effective_protocol,
)
from scripts.run_predator_prey_sampled_pair_ratio_study import (
    assemble as predator_assemble,
    pair_seed as predator_pair_seed,
)
from scripts.run_predator_prey_sweeps import (
    _integer_ratio,
    coefficient_and_profile_metrics,
    condition_plan,
    latent_for_condition,
    model_objects as predator_model_objects,
    observation_view,
    paired_inputs,
    species_balanced_drift_error,
)
from scripts.run_qanr_matched_lasso_sweeps import apply_dictionary
from scripts.run_qanr_sampled_ratio_confirmation_R20 import pair_seeds
from scripts.run_qanr_sampled_ratio_screen import (
    PANEL_CODES,
    assemble_nested_prefix_designs,
    prepare_weak_system,
    sampled_drift_error,
)
from scripts.run_sindy_comparison import (
    DETERMINISTIC_SPACE_CENTERS,
    SPACE_TEST_WIDTH,
    TIME_TEST_SUPPORT,
    experiment_config as observation_config,
    reference_trajectory,
)


DEFAULT_CONFIG = ROOT / "configs/experiments/regularized_baseline_exploration.json"
OUTPUT_ROOT = ROOT / "outputs/validation/regularized_baseline_exploration"
METHOD_ORDER = (
    "weak_lasso",
    "weak_ols",
    "strong_forward_difference_ols",
    "weak_ridge_validation_selected",
    "strong_forward_difference_lasso_validation_selected",
    "strong_forward_difference_ridge_validation_selected",
)
METHOD_LABELS = {
    "weak_lasso": "Weak LASSO",
    "weak_ols": "Weak OLS",
    "strong_forward_difference_ols": "Strong FD–OLS",
    "weak_ridge_validation_selected": "Weak Ridge (validation selected)",
    "strong_forward_difference_lasso_validation_selected": "Strong FD–LASSO (validation selected)",
    "strong_forward_difference_ridge_validation_selected": "Strong FD–Ridge (validation selected)",
}
EXAMPLE_ORDER = ("one_dimensional", "predator_prey", "localized_10d")
EXAMPLE_LABELS = {
    "one_dimensional": "1D redundant six-atom",
    "predator_prey": "Predator--prey",
    "localized_10d": "Localized 10D",
}
SOURCE_FILES = (
    "configs/experiments/regularized_baseline_exploration.json",
    "scripts/run_regularized_baseline_exploration.py",
    "test/test_regularized_baseline_exploration.py",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify(path, expected):
    absolute = ROOT / path
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    actual = digest(absolute)
    if actual != expected:
        raise ValueError(f"pinned input changed: {path}: {actual} != {expected}")
    return absolute


def validate_protocol(spec):
    if spec.get("task") != "regularized_baseline_exploration" or spec.get("protocol_version") != 2:
        raise ValueError("invalid regularized-baseline protocol")
    if spec.get("study_status") != "manuscript_current":
        raise ValueError("regularized baselines must use the current manuscript protocol")
    if tuple(spec.get("methods", ())) != METHOD_ORDER:
        raise ValueError("method list changed")
    if float(spec.get("observation_noise")) != 0.01:
        raise ValueError("nominal observation noise must remain 0.01")
    regularization = spec["regularization"]
    if not regularization["raw_columns"] or regularization["fit_intercept"] \
            or regularization["standardize"]:
        raise ValueError("all new baselines must use raw columns and no intercept")
    if not regularization["weak_ridge"].get("include_zero_candidate"):
        raise ValueError("weak Ridge must include the exact unregularized candidate")
    if not regularization["strong_forward_difference_ridge"].get("include_zero_candidate"):
        raise ValueError("strong FD–Ridge must include the exact unregularized candidate")
    if regularization["strong_forward_difference_lasso"]["fit_intercept"] \
            or regularization["strong_forward_difference_lasso"]["standardize"]:
        raise ValueError("strong FD–LASSO preprocessing changed")
    parity = spec.get("protocol_parity", {})
    if parity.get("strong_unique_system_smoke_policy") != "hard_failure_on_mismatch":
        raise ValueError("strong-system smoke parity policy changed")
    if any(float(parity.get(key, 0.0)) <= 0.0 for key in (
        "coefficient_relative_tolerance", "coefficient_absolute_tolerance",
        "prediction_relative_tolerance", "metric_relative_tolerance",
        "metric_absolute_tolerance",
    )):
        raise ValueError("protocol-parity tolerances must be positive")
    paired = spec.get("paired_comparisons", {})
    if paired.get("comparisons") != [
        ["weak_lasso", "weak_ridge_validation_selected"],
        ["weak_lasso", "strong_forward_difference_lasso_validation_selected"],
        ["weak_ridge_validation_selected", "weak_ols"],
        ["strong_forward_difference_lasso_validation_selected", "strong_forward_difference_ols"],
        ["weak_ridge_validation_selected", "strong_forward_difference_ridge_validation_selected"],
        ["strong_forward_difference_lasso_validation_selected", "strong_forward_difference_ridge_validation_selected"],
        ["strong_forward_difference_ridge_validation_selected", "strong_forward_difference_ols"],
    ] or int(paired.get("bootstrap_draws", 0)) < 1:
        raise ValueError("paired-comparison protocol changed")
    expected_repetitions = {"one_dimensional": 100, "predator_prey": 20, "localized_10d": 20}
    for example, repetitions in expected_repetitions.items():
        if int(spec[example]["repetitions"]) != repetitions:
            raise ValueError(f"{example} repetition count changed")
        for key in tuple(spec[example]):
            if key.endswith("_sha256"):
                path_key = key[:-7]
                if path_key not in spec[example]:
                    raise ValueError(f"missing path for {example}.{key}")
                _verify(spec[example][path_key], spec[example][key])
    output = spec["output_policy"]
    if output != {
        "root": "outputs",
        "modify_existing_outputs": False,
        "modify_manuscript": False,
        "full_scientific_run_requires_explicit_followup": False,
        "manuscript_integration": "performed_by_separate_table_and_figure_generators",
    }:
        raise ValueError("output protection policy changed")


def system_diagnostics(A):
    A = np.asarray(A, dtype=float)
    singular = np.linalg.svd(A, compute_uv=False)
    rank = int(np.linalg.matrix_rank(A))
    condition = float(np.linalg.cond(A))
    return {
        "rows": int(A.shape[0]),
        "columns": int(A.shape[1]),
        "rank": rank,
        "rank_deficient": bool(rank < A.shape[1]),
        "condition_A": condition if np.isfinite(condition) else None,
        "smallest_singular_value": float(singular[-1]) if singular.size else None,
    }


def residual_diagnostics(theta, A, b, Av, bv):
    theta = np.asarray(theta, dtype=float)
    return {
        "training_residual_mse": float(np.mean((np.asarray(A) @ theta - np.asarray(b)) ** 2)),
        "validation_residual_mse": float(np.mean((np.asarray(Av) @ theta - np.asarray(bv)) ** 2)),
    }


def select_ridge(A, b, Av, bv, settings):
    """Select raw no-intercept Ridge solely by validation residual MSE."""
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    Av, bv = np.asarray(Av, dtype=float), np.asarray(bv, dtype=float)
    spectral_scale = float(np.linalg.norm(A, ord=2) ** 2)
    if not np.isfinite(spectral_scale) or spectral_scale <= 0.0:
        raise ValueError("Ridge spectral scale must be positive and finite")
    positive_ratios = np.logspace(
        np.log10(float(settings["alpha_min_ratio_to_spectral_scale"])),
        np.log10(float(settings["alpha_max_ratio_to_spectral_scale"])),
        int(settings["alpha_count"]),
    )
    ratios = ([0.0] if settings.get("include_zero_candidate") else []) + positive_ratios.tolist()
    path = []
    for index, ratio in enumerate(ratios):
        alpha = spectral_scale * float(ratio)
        started = perf_counter()
        model = Ridge(alpha=alpha, fit_intercept=False, solver=settings["solver"])
        model.fit(A, b)
        theta = np.asarray(model.coef_, dtype=float)
        path.append({
            "index": index,
            "alpha": alpha,
            "alpha_over_spectral_scale": float(ratio),
            "validation_mse": float(np.mean((Av @ theta - bv) ** 2)),
            "training_mse": float(np.mean((A @ theta - b) ** 2)),
            "fit_seconds": perf_counter() - started,
            "finite": bool(np.all(np.isfinite(theta))),
            "theta": theta.tolist(),
        })
    finite = [row for row in path if row["finite"] and np.isfinite(row["validation_mse"])]
    if not finite:
        raise RuntimeError("all Ridge candidates were nonfinite")
    selected = min(finite, key=lambda row: (row["validation_mse"], row["alpha"]))
    return np.asarray(selected["theta"], dtype=float), {
        "status": "selected",
        "selected": {key: value for key, value in selected.items() if key != "theta"},
        "selected_zero_penalty": bool(selected["alpha"] == 0.0),
        "selected_at_grid_boundary": bool(
            selected["alpha"] != 0.0
            and selected["index"] in {int(settings.get("include_zero_candidate", False)), len(path) - 1}
        ),
        "spectral_scale": spectral_scale,
        "candidate_count": len(path),
        "path": [{key: value for key, value in row.items() if key != "theta"} for row in path],
    }


def lasso_sufficient_statistic_system(A, b):
    """Compress a no-intercept squared-loss LASSO system without changing its objective."""
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    rows, columns = A.shape
    gram = (A.T @ A) / rows
    score = (A.T @ b) / rows
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    tolerance = max(float(np.max(eigenvalues)), 1.0) * np.finfo(float).eps * columns * 10.0
    eigenvalues = np.where(eigenvalues > tolerance, eigenvalues, 0.0)
    compressed_A = np.sqrt(columns) * (
        np.sqrt(eigenvalues)[:, None] * eigenvectors.T
    )
    compressed_b, *_ = np.linalg.lstsq(
        compressed_A.T, columns * score, rcond=None
    )
    return compressed_A, compressed_b


def select_fd_lasso(A, b, Av, bv, settings):
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    Av, bv = np.asarray(Av, dtype=float), np.asarray(bv, dtype=float)
    compressed_A, compressed_b = lasso_sufficient_statistic_system(A, b)
    selection, coefficient_path = select_lasso(
        compressed_A, compressed_b, Av, bv, settings
    )
    if selection["status"] != "selected":
        raise RuntimeError(f"FD–LASSO selection failed: {selection['status']}")
    selected = selection["selected"]
    selected_theta = np.asarray(selection["theta"], dtype=float)
    path = []
    for row, theta in zip(selection["path"], coefficient_path):
        updated = dict(row)
        if np.all(np.isfinite(theta)):
            updated["train_mse"] = float(np.mean((A @ theta - b) ** 2))
            updated["validation_mse"] = float(np.mean((Av @ theta - bv) ** 2))
        path.append(updated)
    diagnostics = {
        "status": selection["status"],
        "objective_compression": "exact_training_gram_and_score_for_raw_no_intercept_squared_loss",
        "original_training_rows": int(A.shape[0]),
        "compressed_training_rows": int(compressed_A.shape[0]),
        "selected": {
            "index": int(selected["index"]),
            "alpha": float(selected["alpha_sk"]),
            "alpha_over_max": float(selected["alpha_over_max"]),
            "validation_mse": float(np.mean((Av @ selected_theta - bv) ** 2)),
            "training_mse": float(np.mean((A @ selected_theta - b) ** 2)),
            "iterations": int(selected["iterations"]),
            "dual_gap": float(selected["dual_gap"]),
            "converged": bool(selected["converged"]),
            "nonzero_reported": int(selected["nonzero_reported"]),
        },
        "selected_at_grid_boundary": bool(selection["selected_at_grid_boundary"]),
        "candidate_count": len(path),
        "unconverged_candidates": int(sum(not row["converged"] for row in path)),
        "convergence_warnings": int(sum(row["convergence_warnings"] for row in path)),
        "path": path,
    }
    return selected_theta, diagnostics


def _record(example, method, repetition, theta, truth, drift, **extra):
    theta = np.asarray(theta, dtype=float)
    return {
        "example": example,
        "method": method,
        "repetition": int(repetition),
        "coefficients": theta.tolist(),
        "coefficient_error_relative": relative_error(theta, truth),
        "heldout_drift_error_relative": float(drift),
        "status": "finite" if np.all(np.isfinite(theta)) and np.isfinite(drift) else "failed",
        **extra,
    }


def fit_unregularized(A, b):
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    theta, _, rank, singular = np.linalg.lstsq(A, b, rcond=None)
    return np.asarray(theta, dtype=float), {
        **system_diagnostics(A),
        "lstsq_rank": int(rank),
        "singular_values": singular.tolist(),
        "finite": bool(np.all(np.isfinite(theta))),
    }


def protocol_parity_diagnostic(
    *, example, repetition, form, system_label, A, reconstructed_theta,
    reconstructed_metrics, frozen_record, settings,
):
    """Compare a reconstructed unregularized fit with its frozen counterpart."""
    A = np.asarray(A, dtype=float)
    reconstructed_theta = np.asarray(reconstructed_theta, dtype=float)
    frozen_theta = np.asarray(frozen_record["coefficients"], dtype=float)
    if reconstructed_theta.shape != frozen_theta.shape:
        raise ValueError("parity coefficient shapes differ")
    rank = int(np.linalg.matrix_rank(A))
    unique = rank == A.shape[1]
    coefficient_match = bool(np.allclose(
        reconstructed_theta, frozen_theta,
        rtol=float(settings["coefficient_relative_tolerance"]),
        atol=float(settings["coefficient_absolute_tolerance"]),
    ))
    reconstructed_prediction = A @ reconstructed_theta
    frozen_prediction = A @ frozen_theta
    prediction_denominator = max(
        float(np.linalg.norm(reconstructed_prediction)),
        float(np.linalg.norm(frozen_prediction)), 1e-300,
    )
    prediction_relative_difference = float(
        np.linalg.norm(reconstructed_prediction - frozen_prediction)
        / prediction_denominator
    )
    prediction_match = bool(
        prediction_relative_difference
        <= float(settings["prediction_relative_tolerance"])
    )
    metric_parity = {}
    for metric in ("coefficient_error_relative", "heldout_drift_error_relative"):
        reconstructed = float(reconstructed_metrics[metric])
        frozen = float(frozen_record[metric])
        matched = bool(np.isclose(
            reconstructed, frozen,
            rtol=float(settings["metric_relative_tolerance"]),
            atol=float(settings["metric_absolute_tolerance"]),
        ))
        metric_parity[metric] = {
            "reconstructed": reconstructed, "frozen": frozen,
            "difference": reconstructed - frozen, "matched": matched,
        }
    if unique:
        system_match = coefficient_match
        classification = "unique_system_match" if system_match else "unique_system_mismatch"
    else:
        system_match = prediction_match
        classification = (
            "rank_deficient_nonunique_equivalent_predictions"
            if system_match and not coefficient_match
            else "rank_deficient_match"
            if system_match
            else "rank_deficient_system_mismatch"
        )
    hard_smoke_required = form == "strong" and unique
    return {
        "example": example, "repetition": int(repetition), "form": form,
        "system_label": system_label, "frozen_method": frozen_record["method"],
        "rows": int(A.shape[0]), "columns": int(A.shape[1]), "rank": rank,
        "rank_deficient": not unique, "unique_coefficients": unique,
        "coefficient_match": coefficient_match,
        "coefficient_max_abs_difference": float(np.max(np.abs(
            reconstructed_theta - frozen_theta
        ))),
        "coefficient_relative_difference": float(
            np.linalg.norm(reconstructed_theta - frozen_theta)
            / max(float(np.linalg.norm(frozen_theta)), 1e-300)
        ),
        "prediction_relative_difference": prediction_relative_difference,
        "prediction_match": prediction_match, "metric_parity": metric_parity,
        "system_match": system_match, "classification": classification,
        "expected_nonuniqueness": bool(not unique and prediction_match),
        "actual_system_mismatch": bool(not system_match),
        "hard_smoke_required": hard_smoke_required,
        "hard_smoke_passed": bool(not hard_smoke_required or system_match),
        "reconstructed_coefficients": reconstructed_theta.tolist(),
        "frozen_coefficients": frozen_theta.tolist(),
    }


def load_baselines(spec, smoke):
    repetitions = {
        example: (1 if smoke else int(spec[example]["repetitions"]))
        for example in EXAMPLE_ORDER
    }
    records = []

    block = spec["one_dimensional"]
    with (ROOT / block["baseline_records"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        method, repetition = row["method"], int(row["repetition"])
        if method not in METHOD_ORDER[:3] or repetition >= repetitions["one_dimensional"]:
            continue
        records.append({
            "example": "one_dimensional", "method": method, "repetition": repetition,
            "coefficients": json.loads(row["coefficients"]),
            "coefficient_error_relative": float(row["coefficient_error_relative"]),
            "heldout_drift_error_relative": float(row["heldout_drift_error_relative"]),
            "status": "finite", "source": block["baseline_records"],
        })

    block = spec["predator_prey"]
    base = read_json(ROOT / block["base_config"])
    _, _, _, truth_channels, _ = predator_model_objects(base)
    truth = np.concatenate(truth_channels)
    for row in read_json(ROOT / block["baseline_records"]):
        repetition = int(row["repetition"])
        if repetition >= repetitions["predator_prey"]:
            continue
        for source_method, method in (("lasso", "weak_lasso"), ("ols", "weak_ols")):
            payload = row["methods"][source_method]
            theta = np.concatenate([
                np.asarray(payload["channels"][name]["coefficients"], dtype=float)
                for name in ("prey_prey", "prey_predator", "predator_prey")
            ])
            records.append(_record(
                "predator_prey", method, repetition, theta, truth,
                payload["held_out_drift"]["species_balanced_relative"],
                source=block["baseline_records"], systems=row.get("systems"),
            ))
    strong = [row for row in read_json(ROOT / block["strong_baseline_records"])
              if row["example"] == "predator_prey"
              and row["method"] == "strong_forward_difference_ols"]
    for row in strong:
        if int(row["repetition"]) < repetitions["predator_prey"]:
            records.append({**row, "source": block["strong_baseline_records"]})

    block = spec["localized_10d"]
    rows = read_json(ROOT / block["baseline_records"])
    for row in rows:
        if row["sweep"] != "process_sigma" or not np.isclose(row["sweep_value"], block["nominal_process_sigma"]):
            continue
        if int(row["particles"]) != int(block["nominal_particles"]) \
                or int(row["pairs"]) != int(block["nominal_pairs"]):
            continue
        if int(row["repetition"]) >= repetitions["localized_10d"]:
            continue
        method = {"lasso": "weak_lasso", "ols": "weak_ols"}[row["method"]]
        records.append({
            "example": "localized_10d", "method": method,
            "repetition": int(row["repetition"]), "coefficients": row["coefficients"],
            "coefficient_error_relative": float(row["coefficient_error_relative"]),
            "heldout_drift_error_relative": float(row["latent_drift_relative_error"]),
            "profile_error_relative": float(row["profile_relative_error"]),
            "status": "finite" if row["converged"] and not row["numerical_divergence"] else "failed",
            "source": block["baseline_records"],
            "system": {key: row.get(key) for key in (
                "design_rows", "design_columns", "design_rank", "rank_deficient",
                "condition_A", "condition_AtA",
            )},
        })
    strong = [row for row in read_json(ROOT / block["strong_baseline_records"])
              if row["example"] == "localized_10d"
              and row["method"] == "strong_forward_difference_ols"]
    for row in strong:
        if int(row["repetition"]) < repetitions["localized_10d"]:
            records.append({**row, "source": block["strong_baseline_records"]})

    records.sort(key=lambda row: (
        EXAMPLE_ORDER.index(row["example"]), row["repetition"], METHOD_ORDER.index(row["method"])
    ))
    expected = sum(repetitions[example] * 3 for example in EXAMPLE_ORDER)
    if len(records) != expected:
        raise RuntimeError(f"baseline record count {len(records)} != {expected}")
    return records


def run_one_dimensional(index, spec, baselines):
    block = spec["one_dimensional"]
    production = read_json(ROOT / block["production_config"])
    base = read_json(ROOT / block["base_config"])
    nominal = base["observation_noise"]
    sigma_obs = float(spec["observation_noise"])
    train_seed = int(nominal["training_seed_start"] + nominal["training_seed_stride"] * index)
    validation_seed = train_seed + int(production["validation_seed_offset"])
    with threadpool_limits(limits=1):
        X, time, *_ = reference_trajectory(train_seed, step=nominal["dt"], sigma=nominal["sigma"])
        Xv, tv, *_ = reference_trajectory(validation_seed, step=nominal["dt"], sigma=nominal["sigma"])
        observed = X + sigma_obs * np.random.default_rng(train_seed + 2).normal(size=X.shape)
        observed_v = Xv + sigma_obs * np.random.default_rng(validation_seed + 2).normal(size=Xv.shape)
        steps = int(round(nominal["horizon"] / nominal["dt"]))
        config = apply_dictionary(observation_config(
            nominal["particles"], steps, nominal["dt"], nominal["sigma"], train_seed,
            DETERMINISTIC_SPACE_CENTERS, SPACE_TEST_WIDTH, TIME_TEST_SUPPORT, "bump",
        ), base)
        condition_index = production["grids"]["observation_noise"].index(sigma_obs)
        train_pair_seed, validation_pair_seed = pair_seeds(
            production, "observation_noise", index, condition_index, train_seed
        )
        train_system = prepare_weak_system(config, observed, time, train_pair_seed)
        validation_config = json.loads(json.dumps(config))
        validation_config["simulation"]["seed"] += int(production["validation_seed_offset"])
        validation_system = prepare_weak_system(
            validation_config, observed_v, tv, validation_pair_seed
        )
        ratio = int(block["pair_ratio"])
        train_design, train_sampling = assemble_nested_prefix_designs(
            observed, time, train_system, [ratio], train_pair_seed
        )
        validation_design, validation_sampling = assemble_nested_prefix_designs(
            observed_v, tv, validation_system, [ratio], validation_pair_seed
        )
        A, b = train_design[ratio], np.asarray(train_system["b_flat"], dtype=float)
        Av, bv = validation_design[ratio], np.asarray(validation_system["b_flat"], dtype=float)
        theta_ridge, ridge_selection = select_ridge(A, b, Av, bv, spec["regularization"]["weak_ridge"])
        theta_weak_ols, weak_ols_system = fit_unregularized(A, b)

        basis = validation_system["kernel_basis"]
        strong_A, strong_b = compute_increment_regression_system(observed, time, basis)
        strong_Av, strong_bv = compute_increment_regression_system(observed_v, tv, basis)
        theta_lasso, lasso_selection = select_fd_lasso(
            strong_A, strong_b, strong_Av, strong_bv,
            spec["regularization"]["strong_forward_difference_lasso"],
        )
        theta_strong_ridge, strong_ridge_selection = select_ridge(
            strong_A, strong_b, strong_Av, strong_bv,
            spec["regularization"]["strong_forward_difference_ridge"],
        )
        theta_strong_ols, strong_ols_system = fit_unregularized(strong_A, strong_b)
        evaluation_seed = int(nominal["training_seed_start"]) + 2 * int(production["validation_seed_offset"])
        evaluation, _, *_ = reference_trajectory(
            evaluation_seed, particles=int(nominal["particles"]),
            horizon=float(nominal["horizon"]), step=float(nominal["dt"]),
            sigma=float(nominal["sigma"]),
        )
        drift_seed = evaluation_seed + int(production["heldout_table"]["drift_sampling_seed_offset"])
        samples = int(production["heldout_table"]["drift_target_time_samples"])
        truth = np.asarray(block["truth"], dtype=float)
        ridge_drift = sampled_drift_error(theta_ridge, evaluation, basis, truth, drift_seed, samples)
        lasso_drift = sampled_drift_error(theta_lasso, evaluation, basis, truth, drift_seed, samples)
        strong_ridge_drift = sampled_drift_error(
            theta_strong_ridge, evaluation, basis, truth, drift_seed, samples
        )
        weak_ols_drift = sampled_drift_error(
            theta_weak_ols, evaluation, basis, truth, drift_seed, samples
        )
        strong_ols_drift = sampled_drift_error(
            theta_strong_ols, evaluation, basis, truth, drift_seed, samples
        )
    parity = [
        protocol_parity_diagnostic(
            example="one_dimensional", repetition=index, form="weak",
            system_label="global", A=A, reconstructed_theta=theta_weak_ols,
            reconstructed_metrics={
                "coefficient_error_relative": relative_error(theta_weak_ols, truth),
                "heldout_drift_error_relative": weak_ols_drift,
            }, frozen_record=baselines["weak_ols"],
            settings=spec["protocol_parity"],
        ),
        protocol_parity_diagnostic(
            example="one_dimensional", repetition=index, form="strong",
            system_label="global", A=strong_A, reconstructed_theta=theta_strong_ols,
            reconstructed_metrics={
                "coefficient_error_relative": relative_error(theta_strong_ols, truth),
                "heldout_drift_error_relative": strong_ols_drift,
            }, frozen_record=baselines["strong_forward_difference_ols"],
            settings=spec["protocol_parity"],
        ),
    ]
    common = {
        "training_seed": train_seed, "validation_seed": validation_seed,
        "training_pair_seed": train_pair_seed, "validation_pair_seed": validation_pair_seed,
        "observation_noise": sigma_obs,
    }
    records = [
        _record(
            "one_dimensional", "weak_ridge_validation_selected", index,
            theta_ridge, truth, ridge_drift, **common,
            selected_penalties={"global": ridge_selection["selected"]["alpha"]},
            selection=ridge_selection, system=system_diagnostics(A),
            validation_system=system_diagnostics(Av),
            residuals=residual_diagnostics(theta_ridge, A, b, Av, bv),
            pair_sampling={"training": train_sampling, "validation": validation_sampling},
        ),
        _record(
            "one_dimensional", "strong_forward_difference_lasso_validation_selected", index,
            theta_lasso, truth, lasso_drift, **common,
            selected_penalties={"global": lasso_selection["selected"]["alpha"]},
            selection=lasso_selection, system=system_diagnostics(strong_A),
            validation_system=system_diagnostics(strong_Av),
            residuals=residual_diagnostics(theta_lasso, strong_A, strong_b, strong_Av, strong_bv),
        ),
        _record(
            "one_dimensional", "strong_forward_difference_ridge_validation_selected", index,
            theta_strong_ridge, truth, strong_ridge_drift, **common,
            selected_penalties={"global": strong_ridge_selection["selected"]["alpha"]},
            selection=strong_ridge_selection, system=system_diagnostics(strong_A),
            validation_system=system_diagnostics(strong_Av),
            residuals=residual_diagnostics(
                theta_strong_ridge, strong_A, strong_b, strong_Av, strong_bv
            ),
        ),
    ]
    return records, parity, {"weak": weak_ols_system, "strong": strong_ols_system}


def run_predator_prey(index, spec, baselines):
    block = spec["predator_prey"]
    requested = read_json(ROOT / block["production_config"])
    base = read_json(ROOT / block["base_config"])
    protocol = predator_effective_protocol(requested, base)
    condition = next(item for item in condition_plan(protocol)
                     if item["id"] == block["nominal_condition_id"])
    prey_basis, predator_basis, _, truth_channels, truth_kernels = predator_model_objects(protocol)
    train_inputs = paired_inputs(protocol, index, "training")
    validation_inputs = paired_inputs(protocol, index, "validation")
    with threadpool_limits(limits=1):
        (train_x, train_z, _), train_dt = latent_for_condition(
            protocol, condition, train_inputs, truth_kernels, {}
        )
        (validation_x, validation_z, _), validation_dt = latent_for_condition(
            protocol, condition, validation_inputs, truth_kernels, {}
        )
        observed_train = observation_view(
            train_x, train_z, train_dt, train_inputs["unit_prey_noise"],
            train_inputs["unit_predator_noise"], protocol["pairing"]["dt_common_fine"],
            condition["observation_dt"], condition["observation_noise"],
        )
        observed_validation = observation_view(
            validation_x, validation_z, validation_dt, validation_inputs["unit_prey_noise"],
            validation_inputs["unit_predator_noise"], protocol["pairing"]["dt_common_fine"],
            condition["observation_dt"], condition["observation_noise"],
        )
        train_pair_seed = predator_pair_seed(protocol, index, "training")
        validation_pair_seed = predator_pair_seed(protocol, index, "validation")
        training = predator_assemble(
            protocol, *observed_train, prey_basis, predator_basis, train_pair_seed
        )
        validation = predator_assemble(
            protocol, *observed_validation, prey_basis, predator_basis, validation_pair_seed
        )
        ratio = int(block["pair_ratio"])
        train_prey, validation_prey = training["prey"][ratio], validation["prey"][ratio]
        train_predator, validation_predator = training["predator"], validation["predator"]
        ridge_prey, ridge_prey_selection = select_ridge(
            train_prey["A_flat"], train_prey["b_flat"],
            validation_prey["A_flat"], validation_prey["b_flat"],
            spec["regularization"]["weak_ridge"],
        )
        ridge_predator, ridge_predator_selection = select_ridge(
            train_predator["A_flat"], train_predator["b_flat"],
            validation_predator["A_flat"], validation_predator["b_flat"],
            spec["regularization"]["weak_ridge"],
        )
        weak_ols_prey, weak_ols_prey_system = fit_unregularized(
            train_prey["A_flat"], train_prey["b_flat"]
        )
        weak_ols_predator, weak_ols_predator_system = fit_unregularized(
            train_predator["A_flat"], train_predator["b_flat"]
        )
        channel_count = truth_channels[0].size
        ridge_channels = (
            ridge_prey[:channel_count], ridge_prey[channel_count:], ridge_predator
        )

        strong_A_prey, strong_b_prey, strong_A_predator, strong_b_predator = \
            assemble_predator_prey_strong(*observed_train, prey_basis, predator_basis)
        strong_Av_prey, strong_bv_prey, strong_Av_predator, strong_bv_predator = \
            assemble_predator_prey_strong(*observed_validation, prey_basis, predator_basis)
        lasso_prey, lasso_prey_selection = select_fd_lasso(
            strong_A_prey, strong_b_prey, strong_Av_prey, strong_bv_prey,
            spec["regularization"]["strong_forward_difference_lasso"],
        )
        lasso_predator, lasso_predator_selection = select_fd_lasso(
            strong_A_predator, strong_b_predator, strong_Av_predator, strong_bv_predator,
            spec["regularization"]["strong_forward_difference_lasso"],
        )
        strong_ridge_prey, strong_ridge_prey_selection = select_ridge(
            strong_A_prey, strong_b_prey, strong_Av_prey, strong_bv_prey,
            spec["regularization"]["strong_forward_difference_ridge"],
        )
        strong_ridge_predator, strong_ridge_predator_selection = select_ridge(
            strong_A_predator, strong_b_predator, strong_Av_predator, strong_bv_predator,
            spec["regularization"]["strong_forward_difference_ridge"],
        )
        strong_ols_prey, strong_ols_prey_system = fit_unregularized(
            strong_A_prey, strong_b_prey
        )
        strong_ols_predator, strong_ols_predator_system = fit_unregularized(
            strong_A_predator, strong_b_predator
        )
        lasso_channels = (
            lasso_prey[:channel_count], lasso_prey[channel_count:], lasso_predator
        )
        strong_ridge_channels = (
            strong_ridge_prey[:channel_count], strong_ridge_prey[channel_count:],
            strong_ridge_predator,
        )

        test_inputs = paired_inputs(protocol, index, "test")
        (test_x, test_z, _), test_dt = latent_for_condition(
            protocol, condition, test_inputs, truth_kernels, {}
        )
        stride = _integer_ratio(protocol["evaluation"]["drift_evaluation_interval"], test_dt)
        ridge_drift = species_balanced_drift_error(
            test_x, test_z, truth_kernels,
            channel_kernels(*ridge_channels, prey_basis, predator_basis), stride,
        )
        lasso_drift = species_balanced_drift_error(
            test_x, test_z, truth_kernels,
            channel_kernels(*lasso_channels, prey_basis, predator_basis), stride,
        )
        strong_ridge_drift = species_balanced_drift_error(
            test_x, test_z, truth_kernels,
            channel_kernels(*strong_ridge_channels, prey_basis, predator_basis), stride,
        )
        weak_ols_channels = (
            weak_ols_prey[:channel_count], weak_ols_prey[channel_count:],
            weak_ols_predator,
        )
        strong_ols_channels = (
            strong_ols_prey[:channel_count], strong_ols_prey[channel_count:],
            strong_ols_predator,
        )
        weak_ols_drift = species_balanced_drift_error(
            test_x, test_z, truth_kernels,
            channel_kernels(*weak_ols_channels, prey_basis, predator_basis), stride,
        )
        strong_ols_drift = species_balanced_drift_error(
            test_x, test_z, truth_kernels,
            channel_kernels(*strong_ols_channels, prey_basis, predator_basis), stride,
        )
    truth = np.concatenate(truth_channels)
    ridge_theta, lasso_theta = np.concatenate(ridge_channels), np.concatenate(lasso_channels)
    strong_ridge_theta = np.concatenate(strong_ridge_channels)
    weak_ols_theta = np.concatenate(weak_ols_channels)
    strong_ols_theta = np.concatenate(strong_ols_channels)
    weak_frozen = baselines["weak_ols"]
    strong_frozen = baselines["strong_forward_difference_ols"]
    weak_metrics = {
        "coefficient_error_relative": relative_error(weak_ols_theta, truth),
        "heldout_drift_error_relative": weak_ols_drift["species_balanced_relative"],
    }
    strong_metrics = {
        "coefficient_error_relative": relative_error(strong_ols_theta, truth),
        "heldout_drift_error_relative": strong_ols_drift["species_balanced_relative"],
    }
    parity = [
        protocol_parity_diagnostic(
            example="predator_prey", repetition=index, form="weak",
            system_label="prey", A=train_prey["A_flat"],
            reconstructed_theta=weak_ols_prey,
            reconstructed_metrics=weak_metrics,
            frozen_record={**weak_frozen, "coefficients": weak_frozen["coefficients"][:2 * channel_count]},
            settings=spec["protocol_parity"],
        ),
        protocol_parity_diagnostic(
            example="predator_prey", repetition=index, form="weak",
            system_label="predator", A=train_predator["A_flat"],
            reconstructed_theta=weak_ols_predator,
            reconstructed_metrics=weak_metrics,
            frozen_record={**weak_frozen, "coefficients": weak_frozen["coefficients"][2 * channel_count:]},
            settings=spec["protocol_parity"],
        ),
        protocol_parity_diagnostic(
            example="predator_prey", repetition=index, form="strong",
            system_label="prey", A=strong_A_prey,
            reconstructed_theta=strong_ols_prey,
            reconstructed_metrics=strong_metrics,
            frozen_record={**strong_frozen, "coefficients": strong_frozen["coefficients"][:2 * channel_count]},
            settings=spec["protocol_parity"],
        ),
        protocol_parity_diagnostic(
            example="predator_prey", repetition=index, form="strong",
            system_label="predator", A=strong_A_predator,
            reconstructed_theta=strong_ols_predator,
            reconstructed_metrics=strong_metrics,
            frozen_record={**strong_frozen, "coefficients": strong_frozen["coefficients"][2 * channel_count:]},
            settings=spec["protocol_parity"],
        ),
    ]
    common = {
        "training_seed": train_inputs["seed"], "validation_seed": validation_inputs["seed"],
        "test_seed": test_inputs["seed"], "training_pair_seed": train_pair_seed,
        "validation_pair_seed": validation_pair_seed, "observation_noise": condition["observation_noise"],
    }
    records = [
        _record(
            "predator_prey", "weak_ridge_validation_selected", index,
            ridge_theta, truth, ridge_drift["species_balanced_relative"], **common,
            selected_penalties={
                "prey": ridge_prey_selection["selected"]["alpha"],
                "predator": ridge_predator_selection["selected"]["alpha"],
            },
            selection={"prey": ridge_prey_selection, "predator": ridge_predator_selection},
            systems={
                "prey": system_diagnostics(train_prey["A_flat"]),
                "predator": system_diagnostics(train_predator["A_flat"]),
            },
            validation_systems={
                "prey": system_diagnostics(validation_prey["A_flat"]),
                "predator": system_diagnostics(validation_predator["A_flat"]),
            },
            residuals={
                "prey": residual_diagnostics(ridge_prey, train_prey["A_flat"], train_prey["b_flat"], validation_prey["A_flat"], validation_prey["b_flat"]),
                "predator": residual_diagnostics(ridge_predator, train_predator["A_flat"], train_predator["b_flat"], validation_predator["A_flat"], validation_predator["b_flat"]),
            },
            channels=coefficient_and_profile_metrics(
                ridge_channels, truth_channels, protocol, prey_basis, predator_basis
            ),
            held_out_drift=ridge_drift,
        ),
        _record(
            "predator_prey", "strong_forward_difference_lasso_validation_selected", index,
            lasso_theta, truth, lasso_drift["species_balanced_relative"], **common,
            selected_penalties={
                "prey": lasso_prey_selection["selected"]["alpha"],
                "predator": lasso_predator_selection["selected"]["alpha"],
            },
            selection={"prey": lasso_prey_selection, "predator": lasso_predator_selection},
            systems={
                "prey": system_diagnostics(strong_A_prey),
                "predator": system_diagnostics(strong_A_predator),
            },
            validation_systems={
                "prey": system_diagnostics(strong_Av_prey),
                "predator": system_diagnostics(strong_Av_predator),
            },
            residuals={
                "prey": residual_diagnostics(lasso_prey, strong_A_prey, strong_b_prey, strong_Av_prey, strong_bv_prey),
                "predator": residual_diagnostics(lasso_predator, strong_A_predator, strong_b_predator, strong_Av_predator, strong_bv_predator),
            },
            channels=coefficient_and_profile_metrics(
                lasso_channels, truth_channels, protocol, prey_basis, predator_basis
            ),
            held_out_drift=lasso_drift,
        ),
        _record(
            "predator_prey", "strong_forward_difference_ridge_validation_selected", index,
            strong_ridge_theta, truth, strong_ridge_drift["species_balanced_relative"], **common,
            selected_penalties={
                "prey": strong_ridge_prey_selection["selected"]["alpha"],
                "predator": strong_ridge_predator_selection["selected"]["alpha"],
            },
            selection={
                "prey": strong_ridge_prey_selection,
                "predator": strong_ridge_predator_selection,
            },
            systems={
                "prey": system_diagnostics(strong_A_prey),
                "predator": system_diagnostics(strong_A_predator),
            },
            validation_systems={
                "prey": system_diagnostics(strong_Av_prey),
                "predator": system_diagnostics(strong_Av_predator),
            },
            residuals={
                "prey": residual_diagnostics(
                    strong_ridge_prey, strong_A_prey, strong_b_prey,
                    strong_Av_prey, strong_bv_prey,
                ),
                "predator": residual_diagnostics(
                    strong_ridge_predator, strong_A_predator, strong_b_predator,
                    strong_Av_predator, strong_bv_predator,
                ),
            },
            channels=coefficient_and_profile_metrics(
                strong_ridge_channels, truth_channels, protocol, prey_basis, predator_basis
            ),
            held_out_drift=strong_ridge_drift,
        ),
    ]
    return records, parity, {
        "weak_prey": weak_ols_prey_system,
        "weak_predator": weak_ols_predator_system,
        "strong_prey": strong_ols_prey_system,
        "strong_predator": strong_ols_predator_system,
    }


def run_localized_10d(index, spec, baselines):
    block = spec["localized_10d"]
    requested = read_json(ROOT / block["production_config"])
    frozen = read_json(ROOT / block["frozen_protocol"])
    protocol = localized_effective_protocol(requested, frozen)
    plan = repetition_plan(protocol)[index]
    particles = int(block["nominal_particles"])
    sigma = float(block["nominal_process_sigma"])
    pairs = int(block["nominal_pairs"])
    stride = int(protocol["model"]["observation_stride"])
    with threadpool_limits(limits=1):
        training_latent, full_time = _simulate_truth(protocol, particles, sigma, plan["training_seed"])
        validation_latent, _ = _simulate_truth(protocol, particles, sigma, plan["validation_seed"])
        test, _ = _simulate_truth(protocol, particles, sigma, plan["test_seed"])
        observed_time = full_time[::stride]
        training, training_noise_seed = localized_add_observation_noise(
            protocol, training_latent[:, ::stride], particles, plan, "training"
        )
        validation, validation_noise_seed = localized_add_observation_noise(
            protocol, validation_latent[:, ::stride], particles, plan, "validation"
        )
        training_tests = fixed_space_test_spec(protocol, training, plan["space_seed"])
        validation_tests = fixed_space_test_spec(protocol, validation, plan["space_seed"])
        A, b, train_timing = localized_assemble(
            protocol, training, observed_time, particles, sigma, plan,
            "training", pairs, training_tests,
        )
        Av, bv, validation_timing = localized_assemble(
            protocol, validation, observed_time, particles, sigma, plan,
            "validation", pairs, validation_tests,
        )
        ridge_theta, ridge_selection = select_ridge(
            A, b, Av, bv, spec["regularization"]["weak_ridge"]
        )
        weak_ols_theta, weak_ols_system = fit_unregularized(A, b)

        basis, _ = make_kernel_basis(protocol["dictionary"])
        target_count = int(block["strong_target_count"])
        training_target_seed = int(block["strong_training_target_seed_start"] + index)
        validation_target_seed = int(training_target_seed + block["strong_validation_target_seed_offset"])
        train_targets = np.sort(np.random.default_rng(training_target_seed).choice(
            particles, size=target_count, replace=False
        ))
        validation_targets = np.sort(np.random.default_rng(validation_target_seed).choice(
            particles, size=target_count, replace=False
        ))
        strong_A, strong_b = assemble_localized_strong(
            training, observed_time, train_targets, basis
        )
        strong_Av, strong_bv = assemble_localized_strong(
            validation, observed_time, validation_targets, basis
        )
        lasso_theta, lasso_selection = select_fd_lasso(
            strong_A, strong_b, strong_Av, strong_bv,
            spec["regularization"]["strong_forward_difference_lasso"],
        )
        strong_ridge_theta, strong_ridge_selection = select_ridge(
            strong_A, strong_b, strong_Av, strong_bv,
            spec["regularization"]["strong_forward_difference_ridge"],
        )
        strong_ols_theta, strong_ols_system = fit_unregularized(strong_A, strong_b)
        evaluation_system = latent_evaluation_system(protocol, test, plan["evaluation_seed"])
        ridge_metrics = evaluate_theta(ridge_theta, *evaluation_system)
        lasso_metrics = evaluate_theta(lasso_theta, *evaluation_system)
        strong_ridge_metrics = evaluate_theta(strong_ridge_theta, *evaluation_system)
        weak_ols_metrics = evaluate_theta(weak_ols_theta, *evaluation_system)
        strong_ols_metrics = evaluate_theta(strong_ols_theta, *evaluation_system)
    truth = np.asarray(protocol["model"]["interaction"]["coefficients"], dtype=float)
    parity = [
        protocol_parity_diagnostic(
            example="localized_10d", repetition=index, form="weak",
            system_label="global", A=A, reconstructed_theta=weak_ols_theta,
            reconstructed_metrics={
                "coefficient_error_relative": relative_error(weak_ols_theta, truth),
                "heldout_drift_error_relative": weak_ols_metrics["latent_drift_relative_error"],
            }, frozen_record=baselines["weak_ols"],
            settings=spec["protocol_parity"],
        ),
        protocol_parity_diagnostic(
            example="localized_10d", repetition=index, form="strong",
            system_label="global", A=strong_A,
            reconstructed_theta=strong_ols_theta,
            reconstructed_metrics={
                "coefficient_error_relative": relative_error(strong_ols_theta, truth),
                "heldout_drift_error_relative": strong_ols_metrics["latent_drift_relative_error"],
            }, frozen_record=baselines["strong_forward_difference_ols"],
            settings=spec["protocol_parity"],
        ),
    ]
    common = {
        "training_seed": plan["training_seed"], "validation_seed": plan["validation_seed"],
        "test_seed": plan["test_seed"], "training_pair_seed": plan["training_pair_seed"],
        "validation_pair_seed": plan["validation_pair_seed"],
        "training_observation_noise_seed": training_noise_seed,
        "validation_observation_noise_seed": validation_noise_seed,
        "observation_noise": float(protocol["observation_noise"]),
    }
    records = [
        _record(
            "localized_10d", "weak_ridge_validation_selected", index,
            ridge_theta, truth, ridge_metrics["latent_drift_relative_error"], **common,
            profile_error_relative=ridge_metrics["profile_relative_error"],
            selected_penalties={"global": ridge_selection["selected"]["alpha"]},
            selection=ridge_selection, system=design_diagnostics(A),
            validation_system=design_diagnostics(Av),
            residuals=residual_diagnostics(ridge_theta, A, b, Av, bv),
            timing={"training_assembly_seconds": train_timing["assembly_seconds"],
                    "validation_assembly_seconds": validation_timing["assembly_seconds"]},
        ),
        _record(
            "localized_10d", "strong_forward_difference_lasso_validation_selected", index,
            lasso_theta, truth, lasso_metrics["latent_drift_relative_error"], **common,
            profile_error_relative=lasso_metrics["profile_relative_error"],
            training_target_seed=training_target_seed,
            validation_target_seed=validation_target_seed,
            target_count=target_count,
            selected_penalties={"global": lasso_selection["selected"]["alpha"]},
            selection=lasso_selection, system=system_diagnostics(strong_A),
            validation_system=system_diagnostics(strong_Av),
            residuals=residual_diagnostics(lasso_theta, strong_A, strong_b, strong_Av, strong_bv),
        ),
        _record(
            "localized_10d", "strong_forward_difference_ridge_validation_selected", index,
            strong_ridge_theta, truth,
            strong_ridge_metrics["latent_drift_relative_error"], **common,
            profile_error_relative=strong_ridge_metrics["profile_relative_error"],
            training_target_seed=training_target_seed,
            validation_target_seed=validation_target_seed,
            target_count=target_count,
            selected_penalties={"global": strong_ridge_selection["selected"]["alpha"]},
            selection=strong_ridge_selection, system=system_diagnostics(strong_A),
            validation_system=system_diagnostics(strong_Av),
            residuals=residual_diagnostics(
                strong_ridge_theta, strong_A, strong_b, strong_Av, strong_bv
            ),
        ),
    ]
    return records, parity, {"weak": weak_ols_system, "strong": strong_ols_system}


def run_new_repetition(payload):
    example, index, spec, baselines = payload
    started = perf_counter()
    runners = {
        "one_dimensional": run_one_dimensional,
        "predator_prey": run_predator_prey,
        "localized_10d": run_localized_10d,
    }
    records, parity, unregularized_systems = runners[example](
        index, spec, baselines
    )
    return {
        "example": example, "repetition": index, "records": records,
        "protocol_parity": parity,
        "unregularized_system_diagnostics": unregularized_systems,
        "elapsed_seconds": perf_counter() - started,
    }


def metric_summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size), "finite_count": int(finite.size),
        "complete": bool(values.size > 0 and finite.size == values.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "rmse": float(np.sqrt(np.mean(finite ** 2))) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "q25": float(np.quantile(finite, 0.25)) if finite.size else None,
        "q75": float(np.quantile(finite, 0.75)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def summarize(records):
    groups = []
    for example in EXAMPLE_ORDER:
        for method in METHOD_ORDER:
            rows = [row for row in records if row["example"] == example and row["method"] == method]
            groups.append({
                "example": example, "method": method,
                "coefficient_error_relative": metric_summary(
                    [row["coefficient_error_relative"] for row in rows]
                ),
                "heldout_drift_error_relative": metric_summary(
                    [row["heldout_drift_error_relative"] for row in rows]
                ),
                "failed": int(sum(row["status"] != "finite" for row in rows)),
                "boundary_selections": int(sum(bool(row.get("selection", {}).get(
                    "selected_at_grid_boundary", False)) for row in rows)),
            })
    winners = {}
    for example in EXAMPLE_ORDER:
        rows = [row for row in groups if row["example"] == example]
        winners[example] = {
            metric: min(rows, key=lambda row: row[metric]["rmse"])["method"]
            for metric in ("coefficient_error_relative", "heldout_drift_error_relative")
        }
    return {"groups": groups, "winners": winners, "record_count": len(records)}


def paired_comparison_summary(records, paired_spec):
    """Build deterministic paired summaries; lower error is better."""
    results = []
    draws = int(paired_spec["bootstrap_draws"])
    base_seed = int(paired_spec["bootstrap_seed"])
    metrics = ("coefficient_error_relative", "heldout_drift_error_relative")
    for example_index, example in enumerate(EXAMPLE_ORDER):
        example_rows = [row for row in records if row["example"] == example]
        lookup = {
            (row["method"], int(row["repetition"])): row
            for row in example_rows
        }
        for comparison_index, (left_method, right_method) in enumerate(
            paired_spec["comparisons"]
        ):
            repetitions = sorted(
                set(row["repetition"] for row in example_rows
                    if row["method"] == left_method)
                & set(row["repetition"] for row in example_rows
                      if row["method"] == right_method)
            )
            for metric_index, metric in enumerate(metrics):
                left = np.asarray([
                    lookup[(left_method, int(repetition))][metric]
                    for repetition in repetitions
                ], dtype=float)
                right = np.asarray([
                    lookup[(right_method, int(repetition))][metric]
                    for repetition in repetitions
                ], dtype=float)
                finite = np.isfinite(left) & np.isfinite(right)
                left, right = left[finite], right[finite]
                differences = right - left
                ties = np.isclose(left, right, rtol=1e-12, atol=1e-15)
                left_wins = int(np.sum((left < right) & ~ties))
                right_wins = int(np.sum((right < left) & ~ties))
                tie_count = int(np.sum(ties))
                left_rmse = float(np.sqrt(np.mean(left ** 2)))
                right_rmse = float(np.sqrt(np.mean(right ** 2)))
                rmse_ratio = right_rmse / left_rmse if left_rmse > 0.0 else None
                rng = np.random.default_rng(
                    base_seed + 1000 * example_index
                    + 100 * comparison_index + metric_index
                )
                indices = rng.integers(0, left.size, size=(draws, left.size))
                sampled_left = left[indices]
                sampled_right = right[indices]
                sampled_difference = np.mean(sampled_right - sampled_left, axis=1)
                sampled_left_rmse = np.sqrt(np.mean(sampled_left ** 2, axis=1))
                sampled_right_rmse = np.sqrt(np.mean(sampled_right ** 2, axis=1))
                sampled_ratio = np.divide(
                    sampled_right_rmse, sampled_left_rmse,
                    out=np.full(draws, np.nan), where=sampled_left_rmse > 0.0,
                )
                finite_ratio = sampled_ratio[np.isfinite(sampled_ratio)]
                results.append({
                    "example": example, "metric": metric,
                    "left_method": left_method, "right_method": right_method,
                    "count": int(left.size),
                    "win_counts": {
                        "left": left_wins, "right": right_wins, "ties": tie_count,
                    },
                    "paired_difference_right_minus_left": {
                        "values": differences.tolist(),
                        "mean": float(np.mean(differences)),
                        "median": float(np.median(differences)),
                        "q25": float(np.quantile(differences, 0.25)),
                        "q75": float(np.quantile(differences, 0.75)),
                        "bootstrap_mean_95": [
                            float(np.quantile(sampled_difference, 0.025)),
                            float(np.quantile(sampled_difference, 0.975)),
                        ],
                    },
                    "rmse": {"left": left_rmse, "right": right_rmse},
                    "rmse_ratio_right_over_left": rmse_ratio,
                    "rmse_ratio_paired_bootstrap_95": (
                        [float(np.quantile(finite_ratio, 0.025)),
                         float(np.quantile(finite_ratio, 0.975))]
                        if finite_ratio.size else None
                    ),
                    "bootstrap_draws": draws,
                })
    return {
        "difference_definition": paired_spec["difference_definition"],
        "rmse_ratio_definition": paired_spec["rmse_ratio_definition"],
        "comparisons": results,
    }


def write_paired_tables(out, paired):
    lines = [
        "# Paired regularized-baseline comparisons\n\n",
        "Differences are right minus left; negative values favor the right method. "
        "RMSE ratios are right/left; values below one favor the right method.\n\n",
        "| Example | Metric | Left | Right | W-L-T (left/right/tie) | Mean diff. [95%] | RMSE ratio [95%] |\n",
        "|---|---|---|---|---:|---:|---:|\n",
    ]
    csv_rows = []
    for row in paired["comparisons"]:
        difference = row["paired_difference_right_minus_left"]
        ratio_ci = row["rmse_ratio_paired_bootstrap_95"]
        difference_ci = difference["bootstrap_mean_95"]
        ratio_text = (
            f"{row['rmse_ratio_right_over_left']:.6g} "
            f"[{ratio_ci[0]:.6g}, {ratio_ci[1]:.6g}]"
            if ratio_ci is not None and row["rmse_ratio_right_over_left"] is not None
            else "NA"
        )
        wins = row["win_counts"]
        lines.append(
            f"| {EXAMPLE_LABELS[row['example']]} | {row['metric']} | "
            f"{METHOD_LABELS[row['left_method']]} | {METHOD_LABELS[row['right_method']]} | "
            f"{wins['left']}/{wins['right']}/{wins['ties']} | "
            f"{difference['mean']:.6g} [{difference_ci[0]:.6g}, {difference_ci[1]:.6g}] | "
            f"{ratio_text} |\n"
        )
        csv_rows.append({
            "example": row["example"], "metric": row["metric"],
            "left_method": row["left_method"], "right_method": row["right_method"],
            "count": row["count"], "left_wins": wins["left"],
            "right_wins": wins["right"], "ties": wins["ties"],
            "mean_difference_right_minus_left": difference["mean"],
            "mean_difference_bootstrap_025": difference_ci[0],
            "mean_difference_bootstrap_975": difference_ci[1],
            "rmse_ratio_right_over_left": row["rmse_ratio_right_over_left"],
            "rmse_ratio_bootstrap_025": None if ratio_ci is None else ratio_ci[0],
            "rmse_ratio_bootstrap_975": None if ratio_ci is None else ratio_ci[1],
        })
    (out / "paired_comparisons.md").write_text("".join(lines))
    with (out / "paired_comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def write_parity_report(out, parity):
    lines = [
        "# Reconstructed-system protocol parity\n\n",
        "Unique full-rank systems require coefficient parity. Rank-deficient systems "
        "are classified by equality of fitted predictions on the reconstructed design, "
        "so null-space coefficient nonuniqueness is not mislabeled as a system mismatch.\n\n",
        "| Example | Form | System | Rank/columns | Classification | Coef. rel. diff. | Prediction rel. diff. | Hard smoke |\n",
        "|---|---|---|---:|---|---:|---:|---|\n",
    ]
    for row in parity:
        hard = "pass" if row["hard_smoke_required"] and row["hard_smoke_passed"] \
            else "FAIL" if row["hard_smoke_required"] else "n/a"
        lines.append(
            f"| {EXAMPLE_LABELS[row['example']]} | {row['form']} | {row['system_label']} | "
            f"{row['rank']}/{row['columns']} | {row['classification']} | "
            f"{row['coefficient_relative_difference']:.6g} | "
            f"{row['prediction_relative_difference']:.6g} | {hard} |\n"
        )
    (out / "protocol_parity.md").write_text("".join(lines))


def write_records_csv(path, records):
    fields = [
        "example", "method", "repetition", "coefficient_error_relative",
        "heldout_drift_error_relative", "status", "coefficients",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(row[key]) if key == "coefficients" else row.get(key)
                             for key in fields})


def write_summary_tables(out, summary):
    rows = summary["groups"]
    markdown = [
        "# Regularized baseline exploration summary\n\n",
        "| Example | Method | Coefficient RMSE | Held-out drift RMSE | n | Failures |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ]
    csv_rows = []
    for example in EXAMPLE_ORDER:
        for method in METHOD_ORDER:
            row = next(item for item in rows if item["example"] == example and item["method"] == method)
            coefficient = row["coefficient_error_relative"]["rmse"]
            drift = row["heldout_drift_error_relative"]["rmse"]
            markdown.append(
                f"| {EXAMPLE_LABELS[example]} | {METHOD_LABELS[method]} | {coefficient:.6g} | "
                f"{drift:.6g} | {row['coefficient_error_relative']['count']} | {row['failed']} |\n"
            )
            csv_rows.append({
                "example": example, "method": method,
                "coefficient_error_relative_rmse": coefficient,
                "heldout_drift_error_relative_rmse": drift,
                "count": row["coefficient_error_relative"]["count"],
                "failed": row["failed"],
            })
    (out / "summary_table.md").write_text("".join(markdown))
    with (out / "summary_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def write_decision_report(out, summary, paired, parity, smoke):
    lines = [
        "# Manuscript decision report: regularized baselines",
        "",
        "## Scope and decision status",
        "",
        "This experiment implements the six-method comparison used by the current manuscript.",
    ]
    if smoke:
        lines.extend([
            "This output is a one-repetition-per-example smoke run. It is suitable only for checking implementation, data flow, independent validation selection, diagnostics, and artifact generation.",
            "",
            "**Decision: do not add any new baseline to the manuscript from this smoke run.** A full predeclared R=100/20/20 run and review of failures, boundary selections, and paired RMSE distributions would be required.",
        ])
    else:
        lines.extend([
            "This output contains the full predeclared repetition counts and supports the current manuscript comparison after protocol-parity and artifact checks pass.",
            "",
            "A manuscript decision should consider both relative coefficient RMSE and held-out drift RMSE, together with convergence, boundary-selection, and rank diagnostics. A new baseline should not be included solely because it wins one metric in one example.",
        ])
    lines.extend(["", "## Current directional results", ""])
    for example in EXAMPLE_ORDER:
        winner = summary["winners"][example]
        lines.append(
            f"- {EXAMPLE_LABELS[example]}: coefficient-RMSE minimum = "
            f"{METHOD_LABELS[winner['coefficient_error_relative']]}; held-out-drift-RMSE minimum = "
            f"{METHOD_LABELS[winner['heldout_drift_error_relative']]} ."
        )
    hard_checks = [row for row in parity if row["hard_smoke_required"]]
    lines.extend([
        "", "## Protocol parity", "",
        f"- Reconstructed systems checked: {len(parity)}.",
        f"- Actual system mismatches: {sum(row['actual_system_mismatch'] for row in parity)}.",
        f"- Unique strong-system hard checks passed: "
        f"{sum(row['hard_smoke_passed'] for row in hard_checks)}/{len(hard_checks)}.",
        "- Rank-deficient systems are judged by fitted-prediction parity rather than coefficient identity.",
        "", "## Paired decision quantities", "",
        "The paired tables report left/right/tie counts, right-minus-left error differences, "
        "right/left RMSE ratios, and deterministic paired-bootstrap 95% intervals.",
    ])
    for row in paired["comparisons"]:
        if row["metric"] != "heldout_drift_error_relative":
            continue
        wins = row["win_counts"]
        ratio = row["rmse_ratio_right_over_left"]
        lines.append(
            f"- {EXAMPLE_LABELS[row['example']]}, {METHOD_LABELS[row['left_method']]} vs "
            f"{METHOD_LABELS[row['right_method']]}: wins {wins['left']}/{wins['right']}/"
            f"{wins['ties']}, right/left drift RMSE ratio {ratio:.6g}."
        )
    lines.extend([
        "", "## Required review before any manuscript use", "",
        "1. Run the full R=100/20/20 protocol only after explicit approval.",
        "2. Confirm no regularization path is systematically selected at a grid boundary.",
        "3. Review predator--prey prey and predator penalties separately.",
        "4. Check that any coefficient improvement is not accompanied by worse held-out drift.",
        "5. Treat the experiment as a baseline sensitivity analysis, not as a replacement for the current methods without a scientific rationale.",
        "",
    ])
    (out / "manuscript_decision_report.md").write_text("\n".join(lines))


def snapshot_sources(out):
    hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    return hashes


def run(config_path, out, *, workers=1, smoke=False):
    spec = read_json(config_path)
    validate_protocol(spec)
    out = Path(out).resolve()
    allowed = (ROOT / "outputs").resolve()
    if allowed not in out.parents:
        raise ValueError(f"output must be beneath {allowed}")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, out / "requested_protocol.json")
    write_json(out / "effective_protocol.json", {
        "requested": spec, "smoke": bool(smoke),
        "repetitions": {example: 1 if smoke else int(spec[example]["repetitions"])
                        for example in EXAMPLE_ORDER},
    })
    source_hashes = snapshot_sources(out)
    baseline_records = load_baselines(spec, smoke)
    baseline_lookup = {}
    for row in baseline_records:
        baseline_lookup.setdefault(
            (row["example"], int(row["repetition"])), {}
        )[row["method"]] = row
    jobs = [
        (example, index, spec, baseline_lookup[(example, index)])
        for example in EXAMPLE_ORDER
        for index in range(1 if smoke else int(spec[example]["repetitions"]))
    ]
    results = []
    if workers == 1:
        for job in jobs:
            result = run_new_repetition(job)
            results.append(result)
            print(f"completed {result['example']} repetition {result['repetition']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_new_repetition, job): job[:2] for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"completed {result['example']} repetition {result['repetition']}", flush=True)
    new_records = [row for result in results for row in result["records"]]
    records = baseline_records + new_records
    records.sort(key=lambda row: (
        EXAMPLE_ORDER.index(row["example"]), int(row["repetition"]),
        METHOD_ORDER.index(row["method"]),
    ))
    summary = summarize(records)
    paired = paired_comparison_summary(records, spec["paired_comparisons"])
    parity = [row for result in results for row in result["protocol_parity"]]
    parity.sort(key=lambda row: (
        EXAMPLE_ORDER.index(row["example"]), row["repetition"],
        0 if row["form"] == "weak" else 1, row["system_label"],
    ))
    hard_parity_failures = [
        row for row in parity
        if row["hard_smoke_required"] and not row["hard_smoke_passed"]
    ]
    penalties = [
        {"example": row["example"], "method": row["method"],
         "repetition": row["repetition"],
         "selected_penalties": row.get("selected_penalties"),
         "selection": row.get("selection")}
        for row in new_records
    ]
    selection_blocks = []
    for row in new_records:
        selection = row.get("selection", {})
        if "selected" in selection:
            selection_blocks.append(selection)
        else:
            selection_blocks.extend(
                item for item in selection.values()
                if isinstance(item, dict) and "selected" in item
            )
    diagnostics = {
        "new_records": len(new_records),
        "failed_new_records": sum(row["status"] != "finite" for row in new_records),
        "rank_deficient_training_systems": sum(
            bool(row.get("system", {}).get("rank_deficient", False))
            + sum(bool(item.get("rank_deficient", False))
                  for item in row.get("systems", {}).values())
            for row in new_records
        ),
        "boundary_selected_systems": sum(
            bool(item.get("selected_at_grid_boundary", False))
            for item in selection_blocks
        ),
        "zero_penalty_selected_systems": sum(
            bool(item.get("selected_zero_penalty", False))
            for item in selection_blocks
        ),
        "unconverged_lasso_candidates": sum(
            int(item.get("unconverged_candidates", 0)) for item in selection_blocks
        ),
        "lasso_convergence_warnings": sum(
            int(item.get("convergence_warnings", 0)) for item in selection_blocks
        ),
        "selected_system_count": len(selection_blocks),
        "protocol_parity": {
            "checks": len(parity),
            "actual_system_mismatches": int(sum(
                row["actual_system_mismatch"] for row in parity
            )),
            "rank_deficient_expected_nonuniqueness": int(sum(
                row["expected_nonuniqueness"] for row in parity
            )),
            "unique_strong_hard_checks": int(sum(
                row["hard_smoke_required"] for row in parity
            )),
            "unique_strong_hard_failures": len(hard_parity_failures),
        },
        "job_elapsed_seconds": [
            {"example": result["example"], "repetition": result["repetition"],
             "elapsed_seconds": result["elapsed_seconds"]}
            for result in sorted(results, key=lambda item: (EXAMPLE_ORDER.index(item["example"]), item["repetition"]))
        ],
    }
    write_json(out / "records.json", records)
    write_records_csv(out / "records.csv", records)
    write_json(out / "selected_penalties.json", penalties)
    write_json(out / "protocol_parity.json", parity)
    write_parity_report(out, parity)
    write_json(out / "diagnostics.json", diagnostics)
    write_json(out / "summary.json", summary)
    write_json(out / "paired_comparisons.json", paired)
    write_summary_tables(out, summary)
    write_paired_tables(out, paired)
    write_decision_report(out, summary, paired, parity, smoke)
    if smoke and hard_parity_failures:
        raise RuntimeError(
            "unique strong-system protocol parity failed: "
            + ", ".join(
                f"{row['example']}:{row['system_label']}"
                for row in hard_parity_failures
            )
        )
    artifacts = {
        name: digest(out / name) for name in (
            "requested_protocol.json", "effective_protocol.json", "records.json",
            "records.csv", "selected_penalties.json", "protocol_parity.json",
            "protocol_parity.md", "diagnostics.json", "summary.json",
            "paired_comparisons.json", "paired_comparisons.md",
            "paired_comparisons.csv", "summary_table.md", "summary_table.csv",
            "manuscript_decision_report.md",
        )
    }
    manifest = {
        "task": spec["task"], "protocol_version": spec["protocol_version"],
        "status": "smoke_complete" if smoke else "scientific_complete",
        "smoke": bool(smoke), "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records), "new_record_count": len(new_records),
        "protocol_parity_checks": len(parity),
        "unique_strong_parity_hard_failures": len(hard_parity_failures),
        "paired_comparison_rows": len(paired["comparisons"]),
        "source_sha256": source_hashes, "artifact_sha256": artifacts,
        "output_root_enforced": True, "manuscript_modified": False,
        "existing_outputs_modified": False,
    }
    write_json(out / "manifest.json", manifest)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = "smoke" if args.smoke else "scientific"
        args.out = OUTPUT_ROOT / f"{label}-{stamp}"
    result = run(args.config.resolve(), args.out, workers=args.workers, smoke=args.smoke)
    print(result)


if __name__ == "__main__":
    main()
