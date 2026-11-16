"""Build the production-candidate six-atom Figure 2 with sampled pairs at P=50N.

Panels (a)--(d) use the complete current grids and all 100 predeclared
repetitions. Training and validation pair streams are independent and redrawn
at every contributing time. Panel (e) selects one absolute LASSO penalty per
pair-sampling repetition at nominal P0=10000 using independent sampled
validation, then fixes it across all P and for the all-pair reference. Panel
(f) measures the operational sampled design at P=50N. Existing current,
manuscript, frozen, and R20 artifacts are read-only.
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
import subprocess
import sys
from time import perf_counter
import warnings

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from kernels import make_kernel_basis
from pipeline import fit_kernel_from_particles
from regression import fit_theta
from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR,
    REFERENCE_COLOR,
    SECONDARY_COLOR,
    TERTIARY_COLOR,
    TRUE_COLOR,
    apply_manuscript_style,
    finish_axes,
)
from scripts.run_qanr_matched_lasso_sweeps import (
    _selection_record,
    apply_dictionary,
    design_diagnostics,
    fit_fixed_alpha,
    read_json,
    validate_protocol,
)
from scripts.run_qanr_sampled_ratio_confirmation_R20 import run_repetition as run_ad_repetition
from scripts.run_qanr_sampled_ratio_screen import (
    PANEL_CODES,
    PAIR_NAMESPACE,
    current_r100_reference,
    derived_seed,
    digest,
    group_lookup,
    grouped_summary,
    matched_config,
    prepare_weak_system,
    simulate_standard,
    slope,
    write_csv,
    write_json,
)
from scripts.run_qanr_time_step_repetitions import validate_time_protocol
from scripts.run_random_pair_validation import (
    matched_config as timing_matched_config,
    rss_megabytes,
    synthetic_positions,
)
from scripts.run_sindy_comparison import reference_trajectory

DEFAULT_CONFIG = ROOT / "configs/experiments/qanr_redundant6_sampled_p50n_production_R100.json"
SOURCE_FILES = (
    "configs/experiments/qanr_redundant6_sampled_p50n_production_R100.json",
    "configs/experiments/qanr_redundant6_lasso_sweeps_R100.json",
    "configs/experiments/qanr_redundant6_time_step_paired_R100.json",
    "scripts/run_qanr_sampled_p50n_production_R100.py",
    "scripts/run_qanr_sampled_ratio_confirmation_R20.py",
    "scripts/run_qanr_sampled_ratio_screen.py",
    "scripts/run_qanr_matched_lasso_sweeps.py",
    "scripts/run_qanr_time_step_repetitions.py",
    "scripts/run_random_pair_validation.py",
    "scripts/run_full_pair_scaling.py",
    "scripts/run_sindy_comparison.py",
    "scripts/run_1d_hat_lasso.py",
    "scripts/r05_numerical_settings.py",
    "features.py", "pipeline.py", "regression.py", "utils.py", "kernels.py",
    "simulate.py", "tests.py", "time_tests.py",
)
PANEL_BOX_ASPECT = 3 / 4


def validate_production_protocol(protocol, base, time_protocol):
    validate_protocol(base)
    validate_time_protocol(time_protocol, base)
    if protocol.get("task") != "qanr_redundant6_sampled_p50n_production_R100":
        raise ValueError("unexpected production task")
    if protocol.get("protocol_version") != 1 or protocol.get("repetitions") != 100:
        raise ValueError("production protocol must remain version 1 with R=100")
    if protocol.get("ratios") != [50]:
        raise ValueError("production pair ratio must remain P/N=50")
    expected_sampling = {
        "mode": "nested_prefix_resample_each_time",
        "with_replacement": True,
        "exclude_diagonal": True,
        "training_validation_independent": True,
        "time_independent": True,
        "common_indices_across_tests_and_atoms": True,
        "pairs_per_condition": "P=50N",
        "penalty_selection": "independent_sampled_validation_at_P=50N",
    }
    if protocol.get("sampling") != expected_sampling:
        raise ValueError("P=50N sampling contract changed")
    grids = {
        "observation_noise": base["observation_noise"]["sigma_obs_values"],
        "process_noise": base["process_noise"]["values"],
        "particle_number": base["particle_number"]["values"],
        "time_step": time_protocol["time_step"]["values"],
    }
    for panel, expected in grids.items():
        if protocol["grids"].get(panel) != expected:
            raise ValueError(f"{panel} is not the complete current grid")
    target = protocol["time_step_rate_target"]
    if target.get("coefficient_rmse_log_log_slope") != 0.5 \
            or target.get("coefficient_mse_log_log_slope") != 1.0:
        raise ValueError("time-step sampled-pair rate target changed")
    pair = protocol["pair_count"]
    if pair.get("particles") != 200 or pair.get("nominal_selection_pairs") != 10000:
        raise ValueError("panel (e) nominal P0 must remain 50*200=10000")
    if pair.get("values") != [100, 300, 1000, 3000, 10000, 30000] \
            or pair.get("repetitions") != 100:
        raise ValueError("panel (e) production grid changed")
    if "used_for_the_all_pair_reference" not in pair.get("penalty_policy", ""):
        raise ValueError("panel (e) all-pair reference penalty policy changed")
    timing = protocol["timing"]
    if timing.get("sampled_ratio") != 50 or timing.get("repetitions") != 10:
        raise ValueError("panel (f) timing policy changed")
    for key in ("matched_full_pair_batch", "matched_full_pair_time_step_batch"):
        if not (ROOT / protocol[key]).is_dir():
            raise FileNotFoundError(ROOT / protocol[key])


def path_digest(path):
    path = Path(path)
    if path.is_file():
        return digest(path)
    if not path.is_dir():
        return None
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(item.relative_to(path)).encode())
        h.update(b"\0")
        h.update(hashlib.sha256(item.read_bytes()).digest())
    return h.hexdigest()


def protected_digests(protocol):
    return {relative: path_digest(ROOT / relative) for relative in protocol["protected_paths"]}


def snapshot_sources(out, config_path, base_path, time_path):
    files = list(SOURCE_FILES)
    for path in (config_path, base_path, time_path):
        relative = str(Path(path).resolve().relative_to(ROOT))
        if relative not in files:
            files.append(relative)
    hashes = {}
    for relative in files:
        source = ROOT / relative
        target = Path(out) / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    return hashes


def rmse_ci(values):
    values = np.asarray(values, dtype=float)
    squared = values ** 2
    estimate = float(np.sqrt(np.mean(squared)))
    if values.size < 2 or estimate == 0.0:
        se = 0.0
    else:
        se = float(np.std(squared, ddof=1) / np.sqrt(values.size) / (2 * estimate))
    return {
        "rmse": estimate,
        "standard_error": se,
        "ci95": [max(0.0, estimate - 1.96 * se), estimate + 1.96 * se],
        "count": int(values.size),
    }


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    estimate = float(np.mean(values))
    if values.size < 2:
        half = 0.0
    else:
        half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return {
        "mean": estimate,
        "ci95": [max(0.0, estimate - half), estimate + half],
        "count": int(values.size),
    }


def summarize_ad(records):
    summary = []
    keys = sorted({
        (row["panel"], float(row["value"]), row["ratio_label"], row["method"])
        for row in records
    })
    for panel, value, source, method in keys:
        subset = [
            row for row in records
            if row["panel"] == panel and float(row["value"]) == value
            and row["ratio_label"] == source and row["method"] == method
        ]
        coefficient = [float(row["relative_coefficient_error"]) for row in subset]
        drift = [float(row["validation_drift_error"]) for row in subset]
        summary.append({
            "panel": panel, "value": value, "source": source, "method": method,
            "coefficient": rmse_ci(coefficient),
            "validation_drift": rmse_ci(drift),
            "failed": int(sum(row["status"] != "selected" for row in subset)),
            "boundary_selections": int(sum(bool(row.get("selected_at_grid_boundary"))
                                             for row in subset)),
            "unconverged_path_candidates": int(sum(
                int(row.get("unconverged_candidates") or 0) for row in subset
            )),
        })
    return summary


def slope_values(protocol, panel, values):
    values = sorted(values)
    policy = protocol["slope_windows"][panel]
    if policy == "positive_values":
        return [value for value in values if value > 0]
    if policy == "values_at_most_0.02":
        return [value for value in values if value <= 0.02]
    return values


def paired_time_step_bootstrap(protocol, records):
    selected = [
        row for row in records
        if row["panel"] == "time_step" and row["ratio_label"] == "50"
        and row["method"] == "lasso"
    ]
    values = slope_values(protocol, "time_step", protocol["grids"]["time_step"])
    by_value = {
        value: np.asarray([
            float(row["relative_coefficient_error"])
            for row in sorted(selected, key=lambda item: item["repetition"])
            if float(row["value"]) == float(value)
        ]) for value in values
    }
    repetitions = protocol["repetitions"]
    if any(array.size != repetitions for array in by_value.values()):
        raise ValueError("time-step bootstrap input is incomplete")
    rng = np.random.default_rng(protocol["time_step_rate_target"]["bootstrap_seed"])
    count = protocol["time_step_rate_target"]["paired_bootstrap_resamples"]
    slopes = np.empty(count, dtype=float)
    x = np.asarray(values, dtype=float)
    for index in range(count):
        sampled = rng.integers(0, repetitions, size=repetitions)
        y = np.asarray([
            np.sqrt(np.mean(by_value[value][sampled] ** 2)) for value in values
        ])
        slopes[index] = np.polyfit(np.log(x), np.log(y), 1)[0]
    point_y = np.asarray([np.sqrt(np.mean(by_value[value] ** 2)) for value in values])
    point = float(np.polyfit(np.log(x), np.log(point_y), 1)[0])
    lower, upper = np.quantile(slopes, [0.025, 0.975])
    target = float(protocol["time_step_rate_target"]["coefficient_rmse_log_log_slope"])
    tolerance = float(protocol["time_step_rate_target"]["rmse_tolerance"])
    return {
        "values": values,
        "coefficient_rmse": point_y.tolist(),
        "coefficient_rmse_slope": point,
        "coefficient_rmse_slope_bootstrap_ci95": [float(lower), float(upper)],
        "coefficient_mse_slope": 2.0 * point,
        "target_rmse_slope": target,
        "target_mse_slope": 2.0 * target,
        "target_within_bootstrap_ci95": bool(lower <= target <= upper),
        "point_within_declared_tolerance": bool(abs(point - target) <= tolerance),
        "paired_bootstrap_resamples": count,
    }


def ad_diagnostics(protocol, records, summary, r100):
    lookup = {
        (row["panel"], float(row["value"]), row["source"], row["method"]): row
        for row in summary
    }
    slopes = []
    inflations = []
    rankings = []
    endpoints = []
    for panel in PANEL_CODES:
        values = sorted(protocol["grids"][panel])
        selected = slope_values(protocol, panel, values)
        current = {float(row["value"]): row for row in r100[panel]}
        for source in ("current_R100", "full", "50"):
            for method in ("lasso", "ols"):
                if source == "current_R100":
                    points = [(value, current[float(value)][method]) for value in selected]
                else:
                    points = [(value, lookup[(panel, float(value), source, method)][
                        "coefficient"]["rmse"]) for value in selected]
                slopes.append({
                    "panel": panel, "source": source, "method": method,
                    "values": selected, "log_log_slope": slope(points),
                })
        current_difference = (
            current[float(values[-1])]["lasso"] - current[float(values[0])]["lasso"]
        )
        sampled_first = lookup[(panel, float(values[0]), "50", "lasso")]["coefficient"]["rmse"]
        sampled_last = lookup[(panel, float(values[-1]), "50", "lasso")]["coefficient"]["rmse"]
        endpoints.append({
            "panel": panel,
            "current_direction": int(np.sign(current_difference)),
            "sampled_direction": int(np.sign(sampled_last - sampled_first)),
            "matches": int(np.sign(current_difference)) == int(np.sign(sampled_last - sampled_first)),
        })
        for value in values:
            methods = {
                method: lookup[(panel, float(value), "50", method)]["coefficient"]["rmse"]
                for method in ("lasso", "ols")
            }
            if panel == "observation_noise":
                methods["strong_fd_ols"] = lookup[
                    (panel, float(value), "full", "strong_fd_ols")
                ]["coefficient"]["rmse"]
            order = [key for key, _ in sorted(methods.items(), key=lambda item: item[1])]
            rankings.append({
                "panel": panel, "value": value, "order": order,
                "lasso_best": order[0] == "lasso",
            })
            for method in ("lasso", "ols"):
                sampled = lookup[(panel, float(value), "50", method)]["coefficient"]["rmse"]
                full = lookup[(panel, float(value), "full", method)]["coefficient"]["rmse"]
                sampled_drift = lookup[(panel, float(value), "50", method)][
                    "validation_drift"]["rmse"]
                full_drift = lookup[(panel, float(value), "full", method)][
                    "validation_drift"]["rmse"]
                inflations.append({
                    "panel": panel, "value": value, "method": method,
                    "coefficient_sampled": sampled, "coefficient_full": full,
                    "coefficient_inflation": sampled / full,
                    "validation_drift_sampled": sampled_drift,
                    "validation_drift_full": full_drift,
                    "validation_drift_inflation": sampled_drift / full_drift,
                })
    lasso_inflation = [row for row in inflations if row["method"] == "lasso"]
    time_rate = paired_time_step_bootstrap(protocol, records)
    threshold = float(protocol["comparison"]["coefficient_inflation_threshold"])
    worst_coefficient = max(lasso_inflation, key=lambda row: row["coefficient_inflation"])
    worst_drift = max(lasso_inflation, key=lambda row: row["validation_drift_inflation"])
    return {
        "slopes": slopes,
        "endpoint_directions": endpoints,
        "rankings": rankings,
        "inflations": inflations,
        "time_step_pair_sampling_rate": time_rate,
        "summary": {
            "endpoint_matches": int(sum(row["matches"] for row in endpoints)),
            "endpoint_total": len(endpoints),
            "lasso_best_conditions": int(sum(row["lasso_best"] for row in rankings)),
            "ranking_total": len(rankings),
            "lasso_coefficient_within_threshold": int(sum(
                row["coefficient_inflation"] <= threshold for row in lasso_inflation
            )),
            "lasso_coefficient_inflation_median": float(np.median([
                row["coefficient_inflation"] for row in lasso_inflation
            ])),
            "lasso_coefficient_inflation_max": float(worst_coefficient[
                "coefficient_inflation"]),
            "lasso_coefficient_worst_condition": worst_coefficient,
            "lasso_validation_drift_within_threshold": int(sum(
                row["validation_drift_inflation"] <= threshold for row in lasso_inflation
            )),
            "lasso_validation_drift_inflation_median": float(np.median([
                row["validation_drift_inflation"] for row in lasso_inflation
            ])),
            "lasso_validation_drift_inflation_max": float(worst_drift[
                "validation_drift_inflation"]),
            "lasso_validation_drift_worst_condition": worst_drift,
        },
    }


def heldout_nominal_metrics(protocol, base, records):
    block = base["observation_noise"]
    evaluation_seed = (
        int(block["training_seed_start"]) + 2 * int(protocol["validation_seed_offset"])
    )
    X, _, *_ = reference_trajectory(
        evaluation_seed, particles=int(block["particles"]),
        horizon=float(block["horizon"]), step=float(block["dt"]),
        sigma=float(block["sigma"]),
    )
    basis, _ = make_kernel_basis(base["dictionary"])
    truth = np.asarray(base["true_theta"], dtype=float)
    count = int(protocol["heldout_table"]["drift_target_time_samples"])
    rng = np.random.default_rng(
        evaluation_seed + int(protocol["heldout_table"]["drift_sampling_seed_offset"])
    )
    reps = rng.integers(0, X.shape[0], size=count)
    times = rng.integers(0, X.shape[1], size=count)
    targets = rng.integers(0, X.shape[2], size=count)
    design = np.empty((count, truth.size), dtype=float)
    for index, (rep, time_index, target) in enumerate(zip(reps, times, targets)):
        displacement = X[rep, time_index, target][None, :] - X[rep, time_index]
        design[index] = basis(displacement).mean(axis=0)[:, 0]
    true_drift = design @ truth
    denominator = float(np.sum(true_drift ** 2))
    definitions = (
        ("weak_lasso", "50", "lasso"),
        ("weak_ols", "50", "ols"),
        ("strong_forward_difference_ols", "full", "strong_fd_ols"),
        ("matched_full_weak_lasso_reference", "full", "lasso"),
        ("matched_full_weak_ols_reference", "full", "ols"),
    )
    nominal_value = float(protocol["heldout_table"]["value"])
    groups, detailed = [], []
    for label, source, method in definitions:
        subset = sorted([
            row for row in records
            if row["panel"] == "observation_noise"
            and np.isclose(float(row["value"]), nominal_value)
            and row["ratio_label"] == source and row["method"] == method
        ], key=lambda row: row["repetition"])
        coefficient_values, drift_values = [], []
        for row in subset:
            theta = np.asarray(row["coefficients"], dtype=float)
            coefficient = float(row["relative_coefficient_error"])
            drift = float(np.sqrt(np.sum((design @ theta - true_drift) ** 2) / denominator))
            coefficient_values.append(coefficient)
            drift_values.append(drift)
            detailed.append({
                "method": label, "repetition": int(row["repetition"]),
                "coefficient_error_relative": coefficient,
                "heldout_drift_error_relative": drift,
                "coefficients": theta.tolist(),
            })
        groups.append({
            "method": label, "repetitions": len(subset),
            "coefficient": rmse_ci(coefficient_values),
            "heldout_drift": rmse_ci(drift_values),
        })
    return {
        "panel": "observation_noise", "value": nominal_value,
        "evaluation_seed": evaluation_seed,
        "drift_target_time_samples": count,
        "selection_excluded_from_evaluation": True,
        "groups": groups,
    }, detailed


def assemble_budget_designs(positions, time, system, budgets, pair_seed):
    positions = np.asarray(positions, dtype=float)
    R, T1, N, dimension = positions.shape
    if dimension != 1:
        raise ValueError("panel (e) is specialized to the 1D benchmark")
    budgets = sorted(map(int, budgets))
    max_pairs = budgets[-1]
    tests = system["test_fns"]
    basis_fn = system["kernel_basis"]
    temporal = np.asarray(system["time_tests"]["psi"], dtype=float)
    weights = np.asarray(system["weights"], dtype=float)
    row_scale = float(system["row_scale"])
    L = tests["phi"](positions[0, 0]).shape[1]
    K = temporal.shape[0]
    M = basis_fn(np.zeros((1, 1))).shape[1]
    designs = {budget: np.zeros((R, L, K, M), dtype=float) for budget in budgets}
    sampled_times = 0
    for repetition in range(R):
        for time_index in range(T1):
            weight_k = weights[time_index] * temporal[:, time_index]
            if np.all(weight_k == 0.0):
                continue
            sampled_times += 1
            rng = np.random.default_rng(np.random.SeedSequence([
                int(pair_seed), repetition, time_index, N, PAIR_NAMESPACE
            ]))
            flat = rng.integers(0, N * (N - 1), size=max_pairs)
            i_idx = flat // (N - 1)
            j_idx = flat % (N - 1)
            j_idx = j_idx + (j_idx >= i_idx)
            x = positions[repetition, time_index]
            cumulative = np.zeros((L, M), dtype=float)
            start = 0
            for budget in budgets:
                stop = budget
                xi = x[i_idx[start:stop]]
                z = xi - x[j_idx[start:stop]]
                gradients = tests["grad_phi"](xi)[:, :, 0]
                atoms = basis_fn(z)[:, :, 0]
                cumulative += gradients.T @ atoms
                estimate = cumulative * (N - 1) / (N * stop)
                designs[budget][repetition] += estimate[:, None, :] * weight_k[None, :, None]
                start = stop
    return {
        budget: (design * row_scale).reshape(R * L * K, M)
        for budget, design in designs.items()
    }, {
        "particles": N, "budgets": budgets,
        "sampled_contributing_trajectory_times": sampled_times,
        "nested_prefixes": True, "resampled_each_contributing_time": True,
    }


def pair_count_worker(payload):
    repetition, protocol, base = payload
    block = protocol["pair_count"]
    particles = int(block["particles"])
    train_seed = int(block["training_seed"])
    validation_seed = train_seed + int(protocol["validation_seed_offset"])
    sim_block = {**block, "solver": "euler"}
    with threadpool_limits(limits=1):
        config, X, time = simulate_standard(
            sim_block, particles, float(block["sigma"]), train_seed, base
        )
        validation_config, Xv, tv = simulate_standard(
            sim_block, particles, float(block["sigma"]), validation_seed, base
        )
        train_pair_seed = derived_seed(
            protocol["pair_seed_start"] + train_seed, 5, repetition, 0
        )
        validation_pair_seed = derived_seed(
            protocol["pair_seed_start"] + protocol["pair_validation_seed_offset"]
            + train_seed, 5, repetition, 1
        )
        train_system = prepare_weak_system(config, X, time, train_pair_seed)
        validation_system = prepare_weak_system(
            validation_config, Xv, tv, validation_pair_seed
        )
        designs, train_audit = assemble_budget_designs(
            X, time, train_system, block["values"], train_pair_seed
        )
        p0 = int(block["nominal_selection_pairs"])
        validation_designs, validation_audit = assemble_budget_designs(
            Xv, tv, validation_system, [p0], validation_pair_seed
        )
        b = np.asarray(train_system["b_flat"], dtype=float)
        bv = np.asarray(validation_system["b_flat"], dtype=float)
        selection, theta_p0, selected = _selection_record(
            designs[p0], b, validation_designs[p0], bv, base["estimator"]
        )
        alpha = float(selected["alpha_sk"])
        rows = []
        for budget in block["values"]:
            A = designs[int(budget)]
            theta_lasso, convergence = fit_fixed_alpha(A, b, alpha, base["estimator"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                theta_ols, _ = fit_theta(A, b, method="ridge", alpha=0.0, standardize=False)
            rows.append({
                "P": int(budget), "pair_repetition": repetition,
                "alpha_sk": alpha,
                "lasso_coefficients": np.asarray(theta_lasso, dtype=float).tolist(),
                "ols_coefficients": np.asarray(theta_ols, dtype=float).tolist(),
                **design_diagnostics(A), **convergence,
            })
    return {
        "pair_repetition": repetition,
        "training_pair_seed": train_pair_seed,
        "validation_pair_seed": validation_pair_seed,
        "selected": selected,
        "theta_at_P0": theta_p0.tolist(),
        "rows": rows,
        "training_sampling": train_audit,
        "validation_sampling": validation_audit,
    }


def full_pair_count_system(protocol, base):
    block = protocol["pair_count"]
    particles = int(block["particles"])
    train_seed = int(block["training_seed"])
    sim_block = {**block, "solver": "euler"}
    config, X, time = simulate_standard(
        sim_block, particles, float(block["sigma"]), train_seed, base
    )
    fit = fit_kernel_from_particles(config, X, time, return_system=True, pair_mode="full")
    return (
        np.asarray(fit["system"]["A_flat"], dtype=float),
        np.asarray(fit["system"]["b_flat"], dtype=float),
        np.asarray(fit["theta_hat"], dtype=float),
    )


def finalize_pair_count(protocol, base, job_results):
    truth = np.asarray(base["true_theta"], dtype=float)
    truth_norm_sq = float(np.sum(truth ** 2))
    A_full, b_full, theta_full_ols = full_pair_count_system(protocol, base)
    rows, selections = [], []
    for result in sorted(job_results, key=lambda item: item["pair_repetition"]):
        alpha = float(result["selected"]["alpha_sk"])
        theta_full_lasso, full_convergence = fit_fixed_alpha(
            A_full, b_full, alpha, base["estimator"]
        )
        selections.append({
            "pair_repetition": result["pair_repetition"],
            "training_pair_seed": result["training_pair_seed"],
            "validation_pair_seed": result["validation_pair_seed"],
            "selected": result["selected"],
            "theta_at_P0": result["theta_at_P0"],
            "full_pair_alpha_sk": alpha,
            "full_pair_lasso_reference": theta_full_lasso.tolist(),
            "full_pair_ols_reference": theta_full_ols.tolist(),
            "full_pair_lasso_convergence": full_convergence,
            "training_sampling": result["training_sampling"],
            "validation_sampling": result["validation_sampling"],
        })
        for row in result["rows"]:
            lasso = np.asarray(row["lasso_coefficients"], dtype=float)
            ols = np.asarray(row["ols_coefficients"], dtype=float)
            rows.append({
                **row,
                "lasso_relative_mse_to_same_alpha_full": float(
                    np.sum((lasso - theta_full_lasso) ** 2) / truth_norm_sq
                ),
                "ols_relative_mse_to_full": float(
                    np.sum((ols - theta_full_ols) ** 2) / truth_norm_sq
                ),
                "lasso_relative_mse_to_truth": float(np.sum((lasso - truth) ** 2) / truth_norm_sq),
                "ols_relative_mse_to_truth": float(np.sum((ols - truth) ** 2) / truth_norm_sq),
            })
    rows.sort(key=lambda row: (row["P"], row["pair_repetition"]))
    return rows, selections


def summarize_pair_count(protocol, rows):
    summary = []
    for budget in protocol["pair_count"]["values"]:
        subset = [row for row in rows if row["P"] == budget]
        summary.append({
            "P": budget,
            "lasso": mean_ci([
                row["lasso_relative_mse_to_same_alpha_full"] for row in subset
            ]),
            "ols": mean_ci([row["ols_relative_mse_to_full"] for row in subset]),
            "lasso_nonconverged": int(sum(not row["converged"] for row in subset)),
        })
    p = np.asarray([row["P"] for row in summary], dtype=float)
    lasso = np.asarray([row["lasso"]["mean"] for row in summary])
    ols = np.asarray([row["ols"]["mean"] for row in summary])
    return {
        "conditions": summary,
        "slopes": {
            "lasso_mean_square": float(np.polyfit(np.log(p), np.log(lasso), 1)[0]),
            "ols_mean_square": float(np.polyfit(np.log(p), np.log(ols), 1)[0]),
            "target": float(protocol["pair_count"]["expected_mean_square_slope"]),
        },
    }


def runtime_worker(N, mode, repeat, ratio):
    steps = 20
    X, time = synthetic_positions(N, steps, seed=repeat)
    P = int(ratio) * int(N)
    config = timing_matched_config(N, steps, 1.0 / steps, 0.0, P, repeat)
    config["time_tests"].update(K=4, support=0.4)
    config["weak_form"]["full_pair_chunk_size"] = 100_000
    baseline = rss_megabytes()
    fit = fit_kernel_from_particles(config, X, time, pair_mode=mode, pair_seed=repeat)
    peak = rss_megabytes()
    payload = {
        "N": int(N), "mode": mode, "repeat": int(repeat),
        "P": int(N * (N - 1) if mode == "full" else P),
        "sampled_ratio": None if mode == "full" else int(ratio),
        "design_seconds": float(fit["diagnostics"]["timing"]["design_seconds"]),
        "baseline_rss_mb": float(baseline), "peak_rss_mb": float(peak),
        "peak_rss_delta_mb": float(max(0.0, peak - baseline)),
    }
    print(json.dumps(payload))


def run_timing_jobs(protocol, out, resume):
    block = protocol["timing"]
    job_dir = Path(out) / "jobs" / "timing"
    job_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    script = Path(__file__).resolve()
    jobs = [
        (N, mode, repeat)
        for N in block["particles"] for mode in block["modes"]
        for repeat in range(block["repetitions"])
    ]
    for index, (N, mode, repeat) in enumerate(jobs, 1):
        path = job_dir / f"N{N}-{mode}-r{repeat:02d}.json"
        if resume and path.exists():
            row = read_json(path)
        else:
            command = [
                sys.executable, str(script), "--runtime-worker", "--worker-N", str(N),
                "--worker-mode", mode, "--worker-repeat", str(repeat),
                "--worker-ratio", str(block["sampled_ratio"]),
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            row = json.loads(completed.stdout.strip().splitlines()[-1])
            write_json(path, row)
        rows.append(row)
        if index % 10 == 0 or index == len(jobs):
            print(f"timing: {index}/{len(jobs)}", flush=True)
    rows.sort(key=lambda row: (row["N"], row["mode"], row["repeat"]))
    return rows


def summarize_timing(protocol, rows):
    conditions = []
    for N in protocol["timing"]["particles"]:
        for mode in protocol["timing"]["modes"]:
            values = np.asarray([
                row["design_seconds"] for row in rows
                if row["N"] == N and row["mode"] == mode
            ])
            conditions.append({
                "N": N, "mode": mode, "count": int(values.size),
                "median_seconds": float(np.median(values)),
                "iqr_seconds": [float(np.quantile(values, .25)), float(np.quantile(values, .75))],
            })
    slopes = {}
    for mode in protocol["timing"]["modes"]:
        subset = [row for row in conditions if row["mode"] == mode]
        slopes[mode] = float(np.polyfit(
            np.log([row["N"] for row in subset]),
            np.log([row["median_seconds"] for row in subset]), 1
        )[0])
    maximum = max(protocol["timing"]["particles"])
    full = next(row for row in conditions if row["N"] == maximum and row["mode"] == "full")
    sampled = next(row for row in conditions if row["N"] == maximum
                   and row["mode"] == "resample_each_time")
    return {
        "conditions": conditions, "runtime_slopes": slopes,
        "speedup_at_maximum_N": full["median_seconds"] / sampled["median_seconds"],
        "maximum_N": maximum,
    }


def verify(protocol, records, audits, pair_rows, pair_selections, timing_rows,
           diagnostics, protected_before, protected_after):
    issues = []
    expected_records = protocol["repetitions"] * 95
    if len(records) != expected_records:
        issues.append(f"panel (a-d) record count {len(records)} != {expected_records}")
    if len(audits) != protocol["repetitions"] * 22:
        issues.append("panel (a-d) sampling audit count mismatch")
    for audit in audits:
        if audit["training_pair_seed"] == audit["validation_pair_seed"]:
            issues.append("panel (a-d) pair stream collision")
        for split in ("training_sampling", "validation_sampling"):
            sampling = audit[split]
            budget = sampling["budgets"].get("50", sampling["budgets"].get(50))
            if budget != 50 * sampling["particles"]:
                issues.append("panel (a-d) P != 50N")
    if any(row["status"] != "selected" for row in records):
        issues.append("panel (a-d) selected-model failure")
    expected_pair = protocol["pair_count"]["repetitions"] * len(protocol["pair_count"]["values"])
    if len(pair_rows) != expected_pair or len(pair_selections) != protocol["pair_count"]["repetitions"]:
        issues.append("panel (e) count mismatch")
    for selection in pair_selections:
        if selection["training_pair_seed"] == selection["validation_pair_seed"]:
            issues.append("panel (e) pair stream collision")
        repetition = selection["pair_repetition"]
        alphas = {float(row["alpha_sk"]) for row in pair_rows
                  if row["pair_repetition"] == repetition}
        selected_alpha = float(selection["selected"]["alpha_sk"])
        if len(alphas) != 1 or next(iter(alphas)) != selected_alpha:
            issues.append("panel (e) alpha not fixed across P")
        if float(selection["full_pair_alpha_sk"]) != selected_alpha:
            issues.append("panel (e) all-pair reference did not use the selected alpha")
        if not selection["full_pair_lasso_convergence"]["converged"]:
            issues.append("panel (e) all-pair fixed-alpha fit did not converge")
        if selection["validation_sampling"]["budgets"] != [10000]:
            issues.append("panel (e) validation selection budget is not P0=10000")
    if any(not np.isfinite(row["lasso_relative_mse_to_same_alpha_full"])
           for row in pair_rows):
        issues.append("panel (e) nonfinite LASSO metric")
    if any(not row["converged"] for row in pair_rows):
        issues.append("panel (e) sampled fixed-alpha fit did not converge")
    expected_timing = (
        len(protocol["timing"]["particles"]) * len(protocol["timing"]["modes"])
        * protocol["timing"]["repetitions"]
    )
    if len(timing_rows) != expected_timing:
        issues.append("panel (f) timing count mismatch")
    for row in timing_rows:
        if row["mode"] == "resample_each_time" and row["P"] != 50 * row["N"]:
            issues.append("panel (f) sampled timing P != 50N")
    if protected_before != protected_after:
        issues.append("protected current/manuscript/frozen/R20 artifact changed")
    time_rate = diagnostics["time_step_pair_sampling_rate"]
    return {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "records_a_to_d": len(records), "sampling_audits_a_to_d": len(audits),
        "records_e": len(pair_rows), "records_f": len(timing_rows),
        "all_a_to_d_use_P_equals_50N": not any("a-d) P" in item for item in issues),
        "independent_training_validation_pair_streams": not any("stream collision" in item for item in issues),
        "all_selected_models_finite": not any("selected-model failure" in item for item in issues),
        "panel_e_P0_equals_10000": not any("P0=10000" in item for item in issues),
        "panel_e_alpha_fixed_across_P_and_full_reference": not any(
            "alpha" in item for item in issues if "panel (e)" in item
        ),
        "panel_e_all_fixed_alpha_fits_converged": not any(
            "fixed-alpha fit did not converge" in item for item in issues
        ),
        "panel_f_uses_P_equals_50N": not any("f) sampled timing" in item for item in issues),
        "time_step_half_slope_point_within_tolerance": time_rate["point_within_declared_tolerance"],
        "time_step_half_slope_within_bootstrap_ci95": time_rate["target_within_bootstrap_ci95"],
        "protected_paths_unchanged": protected_before == protected_after,
        "protected_path_sha256": protected_after,
    }


def add_rmse(axis, rows, method, label, color, marker, linestyle):
    selected = [row for row in rows if row["method"] == method]
    selected.sort(key=lambda row: row["value"])
    x = np.asarray([row["value"] for row in selected], dtype=float)
    y = np.asarray([row["coefficient"]["rmse"] for row in selected])
    lower = np.asarray([row["coefficient"]["ci95"][0] for row in selected])
    upper = np.asarray([row["coefficient"]["ci95"][1] for row in selected])
    positive = x > 0
    axis.loglog(x[positive], y[positive], marker=marker, linestyle=linestyle,
                color=color, label=label)
    axis.fill_between(x[positive], np.maximum(lower[positive], np.finfo(float).tiny),
                      upper[positive], color=color, alpha=.10, linewidth=0)
    if np.any(~positive):
        axis.axhline(y[~positive][0], color=color, linestyle=":", linewidth=.9)


def plot_figure(protocol, ad_summary, pair_summary, timing_summary, destination):
    apply_manuscript_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.35), constrained_layout=True)
    axes = axes.ravel()
    titles = {
        "observation_noise": "(a) Observation noise",
        "process_noise": "(b) Process noise",
        "particle_number": "(c) Particle number",
        "time_step": "(d) Observation interval",
    }
    xlabels = {
        "observation_noise": r"observation noise $\sigma_{\rm obs}$",
        "process_noise": r"process noise $\sigma$",
        "particle_number": r"particles $N$",
        "time_step": r"observation interval $\Delta t$",
    }
    for axis, panel in zip(axes[:4], PANEL_CODES):
        rows = [row for row in ad_summary if row["panel"] == panel and row["source"] == "50"]
        add_rmse(axis, rows, "lasso", "weak LASSO", ESTIMATE_COLOR, "^", "--")
        add_rmse(axis, rows, "ols", "weak OLS", TRUE_COLOR, "o", "-")
        if panel == "observation_noise":
            strong = [row for row in ad_summary if row["panel"] == panel
                      and row["source"] == "full" and row["method"] == "strong_fd_ols"]
            add_rmse(axis, strong, "strong_fd_ols", "strong FD--OLS",
                     TERTIARY_COLOR, "s", "-.")
        values = np.asarray(sorted(protocol["grids"][panel]), dtype=float)
        if panel == "process_noise":
            selected = sorted([row for row in rows if row["method"] == "lasso"],
                              key=lambda row: row["value"])
            anchor = selected[2]
            reference = np.asarray([values.min(), values.max()])
            axis.loglog(reference, anchor["coefficient"]["rmse"] * reference / anchor["value"],
                        color=REFERENCE_COLOR, linestyle=":", label=r"$O(\sigma)$")
        elif panel == "particle_number":
            selected = sorted([row for row in rows if row["method"] == "lasso"],
                              key=lambda row: row["value"])
            anchor = selected[1]
            reference = np.asarray([values.min(), values.max()])
            axis.loglog(reference, anchor["coefficient"]["rmse"]
                        * (reference / anchor["value"]) ** -.5,
                        color=REFERENCE_COLOR, linestyle=":", label=r"$O(N^{-1/2})$")
        elif panel == "time_step":
            selected = sorted([row for row in rows if row["method"] == "lasso"],
                              key=lambda row: row["value"])
            anchor = next(row for row in selected if np.isclose(row["value"], .01))
            reference = np.asarray([values.min(), values.max()])
            axis.loglog(reference, anchor["coefficient"]["rmse"]
                        * (reference / anchor["value"]) ** .5,
                        color=REFERENCE_COLOR, linestyle=":", label=r"$O(\Delta t^{1/2})$")
        axis.set(title=titles[panel], xlabel=xlabels[panel],
                 ylabel=r"relative $\mathrm{RMSE}_{\theta}$")
        positive = values[values > 0]
        axis.set_xticks(positive, [f"{value:g}" for value in positive])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.legend(loc="best", fontsize=6.3)
        finish_axes(axis)

    conditions = pair_summary["conditions"]
    p = np.asarray([row["P"] for row in conditions], dtype=float)
    for key, label, color, marker, style in (
        ("lasso", "weak LASSO", ESTIMATE_COLOR, "^", "--"),
        ("ols", "weak OLS", TRUE_COLOR, "o", "-"),
    ):
        mean = np.asarray([row[key]["mean"] for row in conditions])
        lower = np.asarray([row[key]["ci95"][0] for row in conditions])
        upper = np.asarray([row[key]["ci95"][1] for row in conditions])
        axes[4].loglog(p, mean, marker=marker, linestyle=style, color=color, label=label)
        axes[4].fill_between(p, np.maximum(lower, np.finfo(float).tiny), upper,
                             color=color, alpha=.12, linewidth=0)
    anchor = conditions[2]["lasso"]["mean"]
    axes[4].loglog(p, anchor * conditions[2]["P"] / p, color=REFERENCE_COLOR,
                   linestyle=":", label=r"$P^{-1}$")
    axes[4].set(xlabel=r"sampled pairs $P$",
                ylabel="relative mean squared\ncoefficient difference",
                title="(e) Effect of pair sampling")
    axes[4].legend(loc="best", fontsize=6.3)
    finish_axes(axes[4])

    timing = timing_summary["conditions"]
    ns = protocol["timing"]["particles"]
    for mode, label, color, marker, style in (
        ("full", "all ordered pairs", TRUE_COLOR, "o", "-"),
        ("resample_each_time", r"sampled, $P=50N$", SECONDARY_COLOR, "s", "--"),
    ):
        selected = [next(row for row in timing if row["N"] == N and row["mode"] == mode)
                    for N in ns]
        median = np.asarray([row["median_seconds"] for row in selected])
        lower = np.asarray([row["iqr_seconds"][0] for row in selected])
        upper = np.asarray([row["iqr_seconds"][1] for row in selected])
        axes[5].loglog(ns, median, marker=marker, linestyle=style, color=color, label=label)
        axes[5].fill_between(ns, lower, upper, color=color, alpha=.16, linewidth=0)
    axes[5].set(xlabel=r"particles $N$", ylabel="feature time [s]",
                title="(f) Feature computation")
    axes[5].legend(loc="upper left", fontsize=6.3)
    finish_axes(axes[5])
    for axis in axes:
        axis.set_box_aspect(PANEL_BOX_ASPECT)
    fig.suptitle(protocol["output"]["watermark"], fontsize=7.5, color=REFERENCE_COLOR)
    destination.mkdir(parents=True, exist_ok=True)
    stem = protocol["output"]["figure_stem"] + "_figure2_candidate"
    pdf, png = destination / f"{stem}.pdf", destination / f"{stem}.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    return [pdf, png]


def plot_diagnostics(protocol, diagnostics, pair_summary, timing_summary, destination):
    apply_manuscript_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True)
    panels = list(PANEL_CODES)
    colors = dict(zip(panels, (ESTIMATE_COLOR, TRUE_COLOR, SECONDARY_COLOR, TERTIARY_COLOR)))
    for panel in panels:
        rows = [row for row in diagnostics["inflations"]
                if row["panel"] == panel and row["method"] == "lasso"]
        rows.sort(key=lambda row: row["value"])
        axes[0, 0].semilogy([row["value"] for row in rows],
                            [row["coefficient_inflation"] for row in rows],
                            marker="o", color=colors[panel], label=panel.replace("_", " "))
        axes[0, 1].semilogy([row["value"] for row in rows],
                            [row["validation_drift_inflation"] for row in rows],
                            marker="o", color=colors[panel], label=panel.replace("_", " "))
    for axis, title in ((axes[0, 0], "Coefficient RMSE: P50 / full"),
                        (axes[0, 1], "Validation-drift RMSE: P50 / full")):
        axis.axhline(protocol["comparison"]["coefficient_inflation_threshold"],
                    color=REFERENCE_COLOR, linestyle=":")
        axis.set(title=title, xlabel="sweep value", ylabel="inflation")
        finish_axes(axis)
    axes[0, 0].legend(fontsize=6.2)

    time_rate = diagnostics["time_step_pair_sampling_rate"]
    values = np.asarray(time_rate["values"])
    y = np.asarray(time_rate["coefficient_rmse"])
    axes[1, 0].loglog(values, y, "o-", color=ESTIMATE_COLOR, label="sampled P=50N")
    anchor = int(np.argmin(np.abs(values - .01)))
    axes[1, 0].loglog(values, y[anchor] * (values / values[anchor]) ** .5,
                      color=REFERENCE_COLOR, linestyle=":", label=r"$\Delta t^{1/2}$")
    axes[1, 0].set(title=f"Time-step RMSE slope = {time_rate['coefficient_rmse_slope']:.3f}",
                   xlabel=r"$\Delta t$", ylabel="relative coefficient RMSE")
    axes[1, 0].legend(fontsize=7)
    finish_axes(axes[1, 0])

    labels = ["pair LASSO", "pair OLS", "runtime full", "runtime P50"]
    values_bar = [pair_summary["slopes"]["lasso_mean_square"],
                  pair_summary["slopes"]["ols_mean_square"],
                  timing_summary["runtime_slopes"]["full"],
                  timing_summary["runtime_slopes"]["resample_each_time"]]
    axes[1, 1].bar(np.arange(4), values_bar,
                   color=(ESTIMATE_COLOR, TRUE_COLOR, TRUE_COLOR, SECONDARY_COLOR))
    axes[1, 1].axhline(-1, color=REFERENCE_COLOR, linestyle=":", linewidth=1)
    axes[1, 1].axhline(1, color="0.55", linestyle=":", linewidth=1)
    axes[1, 1].axhline(2, color="0.35", linestyle=":", linewidth=1)
    axes[1, 1].set_xticks(np.arange(4), labels, rotation=20, ha="right", fontsize=7)
    axes[1, 1].set(ylabel="log--log slope", title="Pair-count and timing slopes")
    finish_axes(axes[1, 1])
    stem = protocol["output"]["figure_stem"] + "_diagnostics"
    pdf, png = destination / f"{stem}.pdf", destination / f"{stem}.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    return [pdf, png]


def table_tex(nominal):
    labels = {
        "weak_lasso": "Weak LASSO",
        "weak_ols": "Weak OLS",
        "strong_forward_difference_ols": "Strong FD--OLS",
    }
    rows = [row for row in nominal["groups"] if row["method"] in labels]
    lines = [
        "\\begin{tabular}{lcc}\n", "\\toprule\n",
        "Method & coefficient RMSE & held-out drift RMSE \\\\\n",
        "\\midrule\n",
    ]
    for row in rows:
        lines.append(
            f"{labels[row['method']]} & {row['coefficient']['rmse']:.4g} & "
            f"{row['heldout_drift']['rmse']:.4g} \\\\\n"
        )
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    return "".join(lines)


def report_text(protocol, diagnostics, pair_summary, timing_summary, nominal, verification):
    time_rate = diagnostics["time_step_pair_sampling_rate"]
    summary = diagnostics["summary"]
    nominal_lookup = {row["method"]: row for row in nominal["groups"]}
    lines = [
        "# Six-atom P=50N sampled-pair production candidate (R=100)", "",
        "This output is a new production candidate. Existing current, manuscript, frozen, "
        "and R20 artifacts remain unchanged.", "",
        "## Panels (a)--(d)", "",
        f"- Endpoint directions matching current R=100: {summary['endpoint_matches']}/{summary['endpoint_total']}",
        f"- Conditions with weak LASSO ranked first: {summary['lasso_best_conditions']}/{summary['ranking_total']}",
        f"- LASSO coefficient inflation <=1.25: "
        f"{summary['lasso_coefficient_within_threshold']}/{summary['ranking_total']}",
        f"- LASSO coefficient-inflation median/max versus matched full pairs: "
        f"{summary['lasso_coefficient_inflation_median']:.3g}/"
        f"{summary['lasso_coefficient_inflation_max']:.3g}",
        f"- LASSO validation-drift inflation <=1.25: "
        f"{summary['lasso_validation_drift_within_threshold']}/{summary['ranking_total']}",
        f"- LASSO validation-drift inflation median/max: "
        f"{summary['lasso_validation_drift_inflation_median']:.3g}/"
        f"{summary['lasso_validation_drift_inflation_max']:.3g}",
        f"- Worst coefficient inflation: "
        f"{summary['lasso_coefficient_worst_condition']['panel']} at "
        f"{summary['lasso_coefficient_worst_condition']['value']:g}", "",
        "## LASSO slopes", "",
        "| sweep | current/full-pair R=100 | sampled P=50N | production target |",
        "|---|---:|---:|---:|",
    ]
    targets = {
        "observation_noise": "descriptive",
        "process_noise": "1",
        "particle_number": "-1/2",
        "time_step": "1/2",
    }
    for panel in PANEL_CODES:
        full_slope = next(row["log_log_slope"] for row in diagnostics["slopes"]
                          if row["panel"] == panel and row["source"] == "full"
                          and row["method"] == "lasso")
        sampled_slope = next(row["log_log_slope"] for row in diagnostics["slopes"]
                             if row["panel"] == panel and row["source"] == "50"
                             and row["method"] == "lasso")
        lines.append(
            f"| {panel.replace('_', ' ')} | {full_slope:.4f} | "
            f"{sampled_slope:.4f} | {targets[panel]} |"
        )
    lines.extend(["", "## Panel (d) sampled-pair rate", "",
        f"- coefficient RMSE slope for Delta t <= 0.02: {time_rate['coefficient_rmse_slope']:.4f}",
        f"- paired-bootstrap 95% interval: "
        f"[{time_rate['coefficient_rmse_slope_bootstrap_ci95'][0]:.4f}, "
        f"{time_rate['coefficient_rmse_slope_bootstrap_ci95'][1]:.4f}]",
        f"- corresponding coefficient MSE slope: {time_rate['coefficient_mse_slope']:.4f}",
        f"- target RMSE/MSE slopes from O(Delta t/P) mean square: 0.5/1.0",
        f"- point estimate within declared tolerance: {time_rate['point_within_declared_tolerance']}",
        f"- target 0.5 inside bootstrap interval: {time_rate['target_within_bootstrap_ci95']}", "",
        "## Panel (e)", "",
        "Each pair-sampling repetition selects lambda at P0=10000 from an independent "
        "sampled validation system. That absolute lambda is fixed across all P and is also "
        "used for the all-pair LASSO reference.",
        f"- LASSO mean-square slope: {pair_summary['slopes']['lasso_mean_square']:.4f}",
        f"- OLS mean-square slope: {pair_summary['slopes']['ols_mean_square']:.4f}", "",
        "## Panel (f)", "",
        f"- full-pair timing slope: {timing_summary['runtime_slopes']['full']:.4f}",
        f"- sampled P=50N timing slope: {timing_summary['runtime_slopes']['resample_each_time']:.4f}",
        f"- speedup at N={timing_summary['maximum_N']}: "
        f"{timing_summary['speedup_at_maximum_N']:.3g}x", "",
        "## Nominal table candidate (sigma_obs=0.01)", "",
        "| method | coefficient RMSE | held-out drift RMSE |",
        "|---|---:|---:|",
    ])
    for method in protocol["heldout_table"]["methods"]:
        row = nominal_lookup[method]
        lines.append(
            f"| {method} | {row['coefficient']['rmse']:.4g} | "
            f"{row['heldout_drift']['rmse']:.4g} |"
        )
    lines.extend(["", "## Verification", "",
                  f"- status: {verification['status']}",
                  f"- records (a--d/e/f): {verification['records_a_to_d']}/"
                  f"{verification['records_e']}/{verification['records_f']}",
                  f"- all (a--d) P=50N: {verification['all_a_to_d_use_P_equals_50N']}",
                  f"- independent train/validation pair streams: "
                  f"{verification['independent_training_validation_pair_streams']}",
                  f"- panel (e) fixed-lambda policy: "
                  f"{verification['panel_e_alpha_fixed_across_P_and_full_reference']}",
                  f"- panel (f) P=50N: {verification['panel_f_uses_P_equals_50N']}",
                  f"- protected artifacts unchanged: {verification['protected_paths_unchanged']}"])
    return "\n".join(lines) + "\n"


def run(config_path, out, workers=1, resume=False):
    protocol = read_json(config_path)
    base_path = ROOT / protocol["base_protocol"]
    time_path = ROOT / protocol["time_step_protocol"]
    base = read_json(base_path)
    time_protocol = read_json(time_path)
    validate_production_protocol(protocol, base, time_protocol)
    out = Path(out).resolve()
    if resume and (out / "requested_protocol.json").exists() \
            and read_json(out / "requested_protocol.json") != protocol:
        raise ValueError("resume protocol differs from saved protocol")
    out.mkdir(parents=True, exist_ok=resume)
    write_json(out / "requested_protocol.json", protocol)
    source_hashes = snapshot_sources(out, config_path, base_path, time_path)
    before_path = out / "protected_inputs_before.json"
    if resume and before_path.exists():
        protected_before = read_json(before_path)
    else:
        protected_before = protected_digests(protocol)
        write_json(before_path, protected_before)
    previous = read_json(out / "manifest.json") if resume and (out / "manifest.json").exists() else {}
    prior_elapsed = float(previous.get("elapsed_seconds", 0.0))
    started = perf_counter()
    manifest = {
        "task": protocol["task"], "status": "running",
        "started_utc": previous.get("started_utc", datetime.now(timezone.utc).isoformat()),
        "last_invocation_started_utc": datetime.now(timezone.utc).isoformat(),
        "workers": workers, "source_sha256": source_hashes,
        "current_manuscript_frozen_R20_are_read_only": True,
    }
    write_json(out / "manifest.json", manifest)

    ad_job_dir = out / "jobs" / "a_to_d"
    ad_job_dir.mkdir(parents=True, exist_ok=True)
    ad_results, pending = [], []
    for repetition in range(protocol["repetitions"]):
        path = ad_job_dir / f"repetition-{repetition:03d}.json"
        if resume and path.exists():
            ad_results.append(read_json(path))
        else:
            pending.append(((repetition, protocol, base, time_protocol), path))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_ad_repetition, payload): path for payload, path in pending}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                write_json(futures[future], result)
                ad_results.append(result)
                if index % 5 == 0 or index == len(pending):
                    print(f"panels a-d: {index}/{len(pending)} new repetitions", flush=True)
    ad_results.sort(key=lambda item: item["repetition"])
    records = [row for result in ad_results for row in result["records"]]
    audits = [row for result in ad_results for row in result["sampling_audits"]]
    records.sort(key=lambda row: (
        row["panel"], row["value"], row["ratio_label"], row["method"], row["repetition"]
    ))

    pair_job_dir = out / "jobs" / "pair_count"
    pair_job_dir.mkdir(parents=True, exist_ok=True)
    pair_results, pair_pending = [], []
    for repetition in range(protocol["pair_count"]["repetitions"]):
        path = pair_job_dir / f"repetition-{repetition:03d}.json"
        if resume and path.exists():
            pair_results.append(read_json(path))
        else:
            pair_pending.append(((repetition, protocol, base), path))
    if pair_pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(pair_count_worker, payload): path
                       for payload, path in pair_pending}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                write_json(futures[future], result)
                pair_results.append(result)
                if index % 5 == 0 or index == len(pair_pending):
                    print(f"panel e: {index}/{len(pair_pending)} new repetitions", flush=True)
    pair_rows, pair_selections = finalize_pair_count(protocol, base, pair_results)
    timing_rows = run_timing_jobs(protocol, out, resume)

    grouped = grouped_summary(records)
    ad_summary = summarize_ad(records)
    r100 = current_r100_reference(protocol)
    diagnostics = ad_diagnostics(protocol, records, ad_summary, r100)
    pair_summary = summarize_pair_count(protocol, pair_rows)
    timing_summary = summarize_timing(protocol, timing_rows)
    nominal, nominal_records = heldout_nominal_metrics(protocol, base, records)
    protected_after = protected_digests(protocol)
    verification = verify(
        protocol, records, audits, pair_rows, pair_selections, timing_rows,
        diagnostics, protected_before, protected_after,
    )
    if verification["status"] != "passed":
        raise ValueError("; ".join(verification["issues"]))
    conclusion = {
        "status": "production_candidate_ready_for_author_review",
        "fixed_pair_policy": "P=50N",
        "panels_a_to_d_endpoint_directions_preserved": (
            diagnostics["summary"]["endpoint_matches"]
            == diagnostics["summary"]["endpoint_total"]
        ),
        "panels_a_to_d_lasso_best_at_all_conditions": (
            diagnostics["summary"]["lasso_best_conditions"]
            == diagnostics["summary"]["ranking_total"]
        ),
        "time_step_rmse_slope": diagnostics["time_step_pair_sampling_rate"][
            "coefficient_rmse_slope"],
        "time_step_half_slope_verified": (
            diagnostics["time_step_pair_sampling_rate"][
                "point_within_declared_tolerance"]
            and diagnostics["time_step_pair_sampling_rate"][
                "target_within_bootstrap_ci95"]
        ),
        "panel_e_lasso_mean_square_slope": pair_summary["slopes"][
            "lasso_mean_square"],
        "panel_f_sampled_runtime_slope": timing_summary["runtime_slopes"][
            "resample_each_time"],
        "panel_f_speedup_at_maximum_N": timing_summary["speedup_at_maximum_N"],
        "protected_artifacts_unchanged": verification["protected_paths_unchanged"],
        "current_or_manuscript_integration_performed": False,
    }

    write_json(out / "records_a_to_d.json", records)
    write_csv(out / "records_a_to_d.csv", records)
    write_json(out / "sampling_audits_a_to_d.json", audits)
    write_json(out / "summary_a_to_d.json", ad_summary)
    write_json(out / "grouped_summary_a_to_d.json", grouped)
    write_json(out / "current_r100_reference.json", r100)
    write_json(out / "diagnostics.json", diagnostics)
    write_json(out / "records_e.json", pair_rows)
    write_csv(out / "records_e.csv", pair_rows)
    write_json(out / "panel_e_selections.json", pair_selections)
    write_json(out / "summary_e.json", pair_summary)
    write_json(out / "records_f.json", timing_rows)
    write_csv(out / "records_f.csv", timing_rows)
    write_json(out / "summary_f.json", timing_summary)
    write_json(out / "nominal_table_metrics.json", nominal)
    write_csv(out / "nominal_table_records.csv", nominal_records)
    (out / "nominal_table_candidate.tex").write_text(table_tex(nominal))
    write_json(out / "verification.json", verification)
    write_json(out / "conclusion.json", conclusion)
    (out / "report.md").write_text(
        report_text(protocol, diagnostics, pair_summary, timing_summary, nominal, verification)
    )
    figure_dir = out / "figures"
    figures = plot_figure(protocol, ad_summary, pair_summary, timing_summary, figure_dir)
    figures.extend(plot_diagnostics(
        protocol, diagnostics, pair_summary, timing_summary, figure_dir
    ))
    artifacts = [
        out / "requested_protocol.json", out / "protected_inputs_before.json",
        out / "records_a_to_d.json", out / "records_a_to_d.csv",
        out / "sampling_audits_a_to_d.json", out / "summary_a_to_d.json",
        out / "grouped_summary_a_to_d.json", out / "current_r100_reference.json",
        out / "diagnostics.json", out / "records_e.json", out / "records_e.csv",
        out / "panel_e_selections.json", out / "summary_e.json",
        out / "records_f.json", out / "records_f.csv", out / "summary_f.json",
        out / "nominal_table_metrics.json", out / "nominal_table_records.csv",
        out / "nominal_table_candidate.tex", out / "verification.json",
        out / "conclusion.json", out / "report.md", *figures,
    ]
    manifest.update({
        "status": "completed", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": prior_elapsed + perf_counter() - started,
        "counts": {"records_a_to_d": len(records), "sampling_audits": len(audits),
                   "records_e": len(pair_rows), "records_f": len(timing_rows)},
        "time_step_rmse_slope": diagnostics["time_step_pair_sampling_rate"][
            "coefficient_rmse_slope"],
        "time_step_half_slope_verified": diagnostics["time_step_pair_sampling_rate"][
            "point_within_declared_tolerance"],
        "protected_paths_unchanged": verification["protected_paths_unchanged"],
        "artifact_sha256": {str(path.relative_to(out)): digest(path) for path in artifacts},
    })
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runtime-worker", action="store_true")
    parser.add_argument("--worker-N", type=int, default=100)
    parser.add_argument("--worker-mode", default="full")
    parser.add_argument("--worker-repeat", type=int, default=0)
    parser.add_argument("--worker-ratio", type=int, default=50)
    args = parser.parse_args()
    if args.runtime_worker:
        runtime_worker(args.worker_N, args.worker_mode, args.worker_repeat, args.worker_ratio)
        return
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "outputs/validation/qanr_sampled_p50n_production" / f"R100-{stamp}"
    manifest = run(args.config.resolve(), args.output, workers=args.workers, resume=args.resume)
    print(json.dumps({
        "status": manifest["status"], "output": str(Path(args.output).resolve()),
        "time_step_rmse_slope": manifest["time_step_rmse_slope"],
    }, indent=2))


if __name__ == "__main__":
    main()
