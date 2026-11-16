"""Run matched-dictionary LASSO counterparts of the six-panel QANR validation figure.

The training trajectories and physical settings reproduce the archived OLS
sweeps.  Every LASSO penalty is selected from an independent validation weak
system without using the true coefficients or plotted evaluation errors.  For
the pair-count sweep, the absolute penalty selected by the full-pair systems is
held fixed across all sampled-pair counts.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import fit_kernel_from_particles
from scripts.run_1d_hat_lasso import select_lasso
from scripts.run_full_pair_scaling import matched_config as full_matched_config
from scripts.run_random_pair_validation import matched_config as pair_matched_config
from scripts.run_sindy_comparison import (
    DETERMINISTIC_SPACE_CENTERS,
    HORIZON as OBSERVATION_HORIZON,
    SPACE_TEST_WIDTH,
    TIME_TEST_SUPPORT,
    experiment_config as observation_config,
    fit_sindy_euler,
    reference_trajectory,
)
from simulate import simulate_particles

DEFAULT_CONFIG = ROOT / "configs/experiments/qanr_matched_lasso_sweeps.json"
SOURCE_FILES = (
    "scripts/run_qanr_matched_lasso_sweeps.py",
    "scripts/run_1d_hat_lasso.py",
    "scripts/run_full_pair_scaling.py",
    "scripts/run_random_pair_validation.py",
    "scripts/run_sindy_comparison.py",
    "scripts/r05_numerical_settings.py",
    "pipeline.py",
    "features.py",
    "kernels.py",
    "simulate.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_protocol(spec):
    if spec.get("task") not in {
        "qanr_matched_lasso_sweeps", "qanr_redundant6_lasso_sweeps"
    } or spec.get("smoke_only") is not False:
        raise ValueError("invalid task or scientific protocol marked as smoke")
    if spec["task"] == "qanr_matched_lasso_sweeps":
        if spec.get("true_theta") != [1.0, -1.0] or spec["dictionary"].get("powers") != [1, 2]:
            raise ValueError("matched QANR dictionary changed")
    else:
        expected_dictionary = {
            "basis": "qanr_exp", "M": 6,
            "ell_grid": [0.25, 0.5, 1.0, 2.0], "eps": 1e-8,
            "contains_true_qanr_atoms": True,
        }
        if spec.get("true_theta") != [1.0, -1.0, 0.0, 0.0, 0.0, 0.0]:
            raise ValueError("redundant-dictionary truth changed")
        if spec.get("dictionary") != expected_dictionary:
            raise ValueError("redundant six-atom dictionary changed")
    estimator = spec["estimator"]
    required_estimator = {
        "method": "raw_lasso", "alpha_count": 60, "alpha_min_ratio": 1e-7,
        "max_iter": 200000, "tol": 1e-8, "fit_intercept": False,
        "standardize": False, "selection": "cyclic",
        "selection_metric": "independent_validation_weak_residual_mse",
        "truth_used_for_selection": False, "test_error_used_for_selection": False,
        "post_lasso_refit": False, "coefficient_thresholding": False,
    }
    for key, value in required_estimator.items():
        if estimator.get(key) != value:
            raise ValueError(f"estimator setting changed: {key}")
    variant = spec.get("protocol_variant", "archived_repetition_counts")
    if variant == "archived_repetition_counts":
        trajectory_repetitions = {"observation_noise": 20, "process_noise": 30,
                                  "particle_number": 20}
    elif variant == "uniform_trajectory_R100":
        trajectory_repetitions = {"observation_noise": 100, "process_noise": 100,
                                  "particle_number": 100}
    else:
        raise ValueError(f"unknown protocol variant: {variant}")
    expected = {
        "observation_noise": ([0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                              trajectory_repetitions["observation_noise"]),
        "process_noise": ([0.01, 0.05, 0.1, 0.2, 0.4],
                          trajectory_repetitions["process_noise"]),
        "particle_number": ([50, 100, 200, 400],
                            trajectory_repetitions["particle_number"]),
        "pair_count": ([100, 300, 1000, 3000, 10000, 30000], 100),
    }
    for name, (values, repetitions) in expected.items():
        field = "sigma_obs_values" if name == "observation_noise" else "values"
        if spec[name].get(field) != values or spec[name].get("repetitions") != repetitions:
            raise ValueError(f"{name} sweep changed")
    if spec["time_step"].get("values") != [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
        raise ValueError("time-step sweep changed")
    if spec["pair_count"].get("penalty_policy") != (
        "absolute_alpha_selected_by_independent_full_pair_validation_then_fixed_across_P"
    ):
        raise ValueError("pair-count penalty policy changed")
    fixed = spec["fixed_tests"]
    if fixed != {
        "spatial_type": "gaussian", "spatial_count": 25,
        "spatial_interval": [0.0, 4.0], "spatial_width": 0.25,
        "temporal_type": "bump", "temporal_count": 8,
        "temporal_support_factor": 1.8,
    }:
        raise ValueError("fixed weak-test family changed")


def effective_protocol(spec, smoke=False):
    effective = json.loads(json.dumps(spec))
    effective["smoke_only"] = bool(smoke)
    if not smoke:
        return effective
    effective["observation_noise"]["repetitions"] = 1
    effective["observation_noise"]["sigma_obs_values"] = [0.0, 0.01]
    effective["process_noise"]["values"] = [0.1]
    effective["process_noise"]["repetitions"] = 1
    effective["particle_number"]["values"] = [100]
    effective["particle_number"]["repetitions"] = 1
    effective["time_step"]["values"] = [0.01]
    effective["time_step"]["stacked_trajectories"] = 1
    effective["pair_count"]["values"] = [100, 1000]
    effective["pair_count"]["repetitions"] = 2
    return effective


def true_theta(spec):
    return np.asarray(spec["true_theta"], dtype=float)


def apply_dictionary(config, spec):
    if spec["task"] == "qanr_redundant6_lasso_sweeps":
        config["kernel_basis"] = {
            key: value for key, value in spec["dictionary"].items()
            if key != "contains_true_qanr_atoms"
        }
    return config


def design_diagnostics(A):
    rank = int(np.linalg.matrix_rank(A))
    return {
        "ols_design_rank": rank,
        "ols_design_columns": int(A.shape[1]),
        "ols_rank_deficient": bool(rank < A.shape[1]),
    }


def _system(fit):
    return np.asarray(fit["system"]["A_flat"], dtype=float), np.asarray(
        fit["system"]["b_flat"], dtype=float
    )


def _fit_full(config, X, t):
    fit = fit_kernel_from_particles(config, X, t, return_system=True, pair_mode="full")
    A, b = _system(fit)
    return fit, A, b


def _selection_record(A, b, A_validation, b_validation, estimator):
    selection, _ = select_lasso(A, b, A_validation, b_validation, estimator)
    if selection["status"] != "selected":
        raise RuntimeError(f"no converged LASSO candidate: {selection['status']}")
    theta = np.asarray(selection["theta"], dtype=float)
    selected = selection["selected"]
    return selection, theta, {
        "alpha_sk": float(selected["alpha_sk"]),
        "alpha_over_max": float(selected["alpha_over_max"]),
        "alpha_index": int(selected["index"]),
        "selected_at_grid_boundary": bool(selection["selected_at_grid_boundary"]),
        "selected_iterations": int(selected["iterations"]),
        "selected_validation_mse": float(selected["validation_mse"]),
        "unconverged_candidates": int(sum(not row["converged"] for row in selection["path"])),
    }


def _simulate_full_system(*, particles, repetitions, horizon, dt, sigma, seed, solver, spec):
    steps = int(round(horizon / dt))
    config = full_matched_config(
        particles=particles,
        repetitions=repetitions,
        steps=steps,
        dt=dt,
        sigma=sigma,
        seed=seed,
        solver=solver,
        time_test_type="bump",
    )
    config = apply_dictionary(config, spec)
    X, t, *_ = simulate_particles(
        N=particles, T=steps, dt=dt, R=repetitions, sigma=sigma, d=1,
        init="gaussian_mixture", init_centers=[0.5, 2.0, 3.5], init_std=0.005,
        init_scale=1.0, kernel_params=config["simulation"]["interaction"],
        seed=seed, solver=solver,
    )
    fit, A, b = _fit_full(config, X, t)
    return fit, A, b


def observation_worker(payload):
    seed_index, spec = payload
    block = spec["observation_noise"]
    train_seed = block["training_seed_start"] + block["training_seed_stride"] * seed_index
    validation_seed = train_seed + spec["validation_seed_offset"]
    X, t, *_ = reference_trajectory(train_seed, step=block["dt"], sigma=block["sigma"])
    Xv, tv, *_ = reference_trajectory(validation_seed, step=block["dt"], sigma=block["sigma"])
    train_noise = np.random.default_rng(train_seed + 2).normal(0.0, 1.0, size=X.shape)
    validation_noise = np.random.default_rng(validation_seed + 2).normal(0.0, 1.0, size=Xv.shape)
    steps = int(round(block["horizon"] / block["dt"]))
    rows = []
    truth = true_theta(spec)
    with threadpool_limits(limits=1):
        for sigma_obs in block["sigma_obs_values"]:
            config = observation_config(
                block["particles"], steps, block["dt"], block["sigma"], train_seed,
                DETERMINISTIC_SPACE_CENTERS, SPACE_TEST_WIDTH, TIME_TEST_SUPPORT, "bump",
            )
            validation_config = observation_config(
                block["particles"], steps, block["dt"], block["sigma"], validation_seed,
                DETERMINISTIC_SPACE_CENTERS, SPACE_TEST_WIDTH, TIME_TEST_SUPPORT, "bump",
            )
            config = apply_dictionary(config, spec)
            validation_config = apply_dictionary(validation_config, spec)
            observed = X + sigma_obs * train_noise
            fit, A, b = _fit_full(config, observed, t)
            _, Av, bv = _fit_full(validation_config, Xv + sigma_obs * validation_noise, tv)
            selection, theta, selected = _selection_record(A, b, Av, bv, spec["estimator"])
            ols_theta = np.asarray(fit["theta_hat"], dtype=float)
            euler_theta, euler_condition, _, euler_equations = fit_sindy_euler(
                config, observed, t, fit["system"]["kernel_basis"]
            )
            rows.append({
                "seed_index": seed_index,
                "trajectory_seed": train_seed,
                "validation_seed": validation_seed,
                "sigma_obs": float(sigma_obs),
                **{f"theta_{index + 1}": float(value) for index, value in enumerate(theta)},
                "coefficient_error": float(np.linalg.norm(theta - truth)),
                **{f"ols_theta_{index + 1}_recomputed": float(value)
                   for index, value in enumerate(ols_theta)},
                "ols_coefficient_error": float(np.linalg.norm(ols_theta - truth)),
                **{f"euler_theta_{index + 1}": float(value)
                   for index, value in enumerate(euler_theta)},
                "euler_coefficient_error": float(np.linalg.norm(euler_theta - truth)),
                "euler_cond_ata": float(euler_condition),
                "ols_max_abs_coefficient": float(np.max(np.abs(ols_theta))),
                **design_diagnostics(A),
                "euler_equations": int(euler_equations),
                **selected,
            })
    return {"kind": "observation", "index": seed_index, "rows": rows}


def stochastic_worker(payload):
    kind, value, seed_index, spec = payload
    if kind == "sigma":
        block = spec["process_noise"]
        particles, sigma = block["particles"], float(value)
    elif kind == "particles":
        block = spec["particle_number"]
        particles, sigma = int(value), block["sigma"]
    else:
        raise ValueError(f"unknown stochastic kind: {kind}")
    train_seed = block["training_seed_start"] + block["training_seed_stride"] * seed_index
    validation_seed = train_seed + spec["validation_seed_offset"]
    truth = true_theta(spec)
    with threadpool_limits(limits=1):
        fit, A, b = _simulate_full_system(
            particles=particles, repetitions=1, horizon=block["horizon"], dt=block["dt"],
            sigma=sigma, seed=train_seed, solver=block["solver"], spec=spec,
        )
        _, Av, bv = _simulate_full_system(
            particles=particles, repetitions=1, horizon=block["horizon"], dt=block["dt"],
            sigma=sigma, seed=validation_seed, solver=block["solver"], spec=spec,
        )
        _, theta, selected = _selection_record(A, b, Av, bv, spec["estimator"])
    ols_theta = np.asarray(fit["theta_hat"], dtype=float)
    return {
        "kind": kind, "value": float(value), "seed_index": seed_index,
        "trajectory_seed": train_seed, "validation_seed": validation_seed,
        "particles": particles, "sigma": sigma,
        **{f"theta_{index + 1}": float(value) for index, value in enumerate(theta)},
        "coefficient_error": float(np.linalg.norm(theta - truth)),
        **{f"ols_theta_{index + 1}_recomputed": float(value)
           for index, value in enumerate(ols_theta)},
        "ols_coefficient_error": float(np.linalg.norm(ols_theta - truth)),
        "ols_max_abs_coefficient": float(np.max(np.abs(ols_theta))),
        **design_diagnostics(A),
        **selected,
    }


def time_step_worker(payload):
    dt, spec = payload
    block = spec["time_step"]
    train_seed = block["training_seed"]
    validation_seed = train_seed + spec["validation_seed_offset"]
    truth = true_theta(spec)
    with threadpool_limits(limits=1):
        fit, A, b = _simulate_full_system(
            particles=block["particles"], repetitions=block["stacked_trajectories"],
            horizon=block["horizon"], dt=float(dt), sigma=block["sigma"],
            seed=train_seed, solver=block["solver"], spec=spec,
        )
        _, Av, bv = _simulate_full_system(
            particles=block["particles"], repetitions=block["stacked_trajectories"],
            horizon=block["horizon"], dt=float(dt), sigma=block["sigma"],
            seed=validation_seed, solver=block["solver"], spec=spec,
        )
        _, theta, selected = _selection_record(A, b, Av, bv, spec["estimator"])
    ols_theta = np.asarray(fit["theta_hat"], dtype=float)
    return {
        "dt": float(dt), "trajectory_seed": train_seed,
        "validation_seed": validation_seed,
        **{f"theta_{index + 1}": float(value) for index, value in enumerate(theta)},
        "coefficient_error": float(np.linalg.norm(theta - truth)),
        **{f"ols_theta_{index + 1}_recomputed": float(value)
           for index, value in enumerate(ols_theta)},
        "ols_coefficient_error": float(np.linalg.norm(ols_theta - truth)),
        "ols_max_abs_coefficient": float(np.max(np.abs(ols_theta))),
        **design_diagnostics(A),
        **selected,
    }


def fit_fixed_alpha(A, b, alpha, estimator):
    model = Lasso(
        alpha=float(alpha), fit_intercept=False, max_iter=estimator["max_iter"],
        tol=estimator["tol"], selection="cyclic",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(A, b)
    theta = np.asarray(model.coef_, dtype=float)
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    return theta, {
        "converged": bool(np.all(np.isfinite(theta)) and warning_count == 0),
        "convergence_warnings": int(warning_count),
        "iterations": int(model.n_iter_),
        "dual_gap": float(model.dual_gap_) if np.isfinite(model.dual_gap_) else None,
    }


def run_pair_sweep(spec):
    block = spec["pair_count"]
    steps = int(round(block["horizon"] / block["dt"]))
    train_seed = block["training_seed"]
    validation_seed = train_seed + spec["validation_seed_offset"]
    truth = true_theta(spec)
    with threadpool_limits(limits=1):
        train_config = pair_matched_config(
            block["particles"], steps, block["dt"], block["sigma"],
            block["particles"] * (block["particles"] - 1), 0, "bump",
        )
        train_config = apply_dictionary(train_config, spec)
        X, t, *_ = simulate_particles(
            N=block["particles"], T=steps, dt=block["dt"], R=1,
            sigma=block["sigma"], d=1, init="gaussian_mixture",
            init_centers=[0.5, 2.0, 3.5], init_std=0.005,
            kernel_params=train_config["simulation"]["interaction"],
            seed=train_seed, solver="euler",
        )
        validation_config = pair_matched_config(
            block["particles"], steps, block["dt"], block["sigma"],
            block["particles"] * (block["particles"] - 1), 0, "bump",
        )
        validation_config = apply_dictionary(validation_config, spec)
        Xv, tv, *_ = simulate_particles(
            N=block["particles"], T=steps, dt=block["dt"], R=1,
            sigma=block["sigma"], d=1, init="gaussian_mixture",
            init_centers=[0.5, 2.0, 3.5], init_std=0.005,
            kernel_params=validation_config["simulation"]["interaction"],
            seed=validation_seed, solver="euler",
        )
        full_fit, A, b = _fit_full(train_config, X, t)
        _, Av, bv = _fit_full(validation_config, Xv, tv)
        selection, theta_full, selected = _selection_record(A, b, Av, bv, spec["estimator"])
        alpha = selected["alpha_sk"]
        rows = []
        for P in block["values"]:
            for pair_seed in range(block["repetitions"]):
                config = pair_matched_config(
                    block["particles"], steps, block["dt"], block["sigma"],
                    int(P), pair_seed, "bump",
                )
                config = apply_dictionary(config, spec)
                fit = fit_kernel_from_particles(
                    config, X, t, return_system=True,
                    pair_mode=block["pair_mode"], pair_seed=pair_seed,
                )
                Ap, bp = _system(fit)
                theta, convergence = fit_fixed_alpha(Ap, bp, alpha, spec["estimator"])
                ols_theta = np.asarray(fit["theta_hat"], dtype=float)
                rows.append({
                    "P": int(P), "pair_seed": pair_seed,
                    "alpha_sk": alpha,
                    **{f"theta_{index + 1}": float(value)
                       for index, value in enumerate(theta)},
                    "theta_to_full_sq": float(np.sum((theta - theta_full) ** 2)),
                    "theta_to_true_sq": float(np.sum((theta - truth) ** 2)),
                    **{f"ols_theta_{index + 1}_recomputed": float(value)
                       for index, value in enumerate(ols_theta)},
                    "ols_theta_to_full_sq_recomputed": float(np.sum(
                        (ols_theta - np.asarray(full_fit["theta_hat"], dtype=float)) ** 2
                    )),
                    "ols_max_abs_coefficient": float(np.max(np.abs(ols_theta))),
                    **design_diagnostics(Ap),
                    **convergence,
                })
    return {"selection": selection, "theta_full": theta_full.tolist(), "rows": rows}


def _job_file(out, kind, index):
    return Path(out) / "jobs" / kind / f"job-{index:04d}.json"


def execute_jobs(out, kind, payloads, worker, workers):
    results = [None] * len(payloads)
    missing = []
    for index, payload in enumerate(payloads):
        path = _job_file(out, kind, index)
        if path.exists():
            results[index] = read_json(path)
        else:
            missing.append((index, payload))
    if missing:
        if workers == 1:
            generated = map(worker, [payload for _, payload in missing])
            pool = None
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            generated = pool.map(worker, [payload for _, payload in missing])
        try:
            for count, ((index, _), result) in enumerate(zip(missing, generated), 1):
                write_json(_job_file(out, kind, index), result)
                results[index] = result
                if count % 10 == 0 or count == len(missing):
                    print(f"{kind}: {count}/{len(missing)} new jobs", flush=True)
        finally:
            if pool is not None:
                pool.shutdown()
    return results


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def rmse_with_ci(values):
    values = np.asarray(values, dtype=float)
    squared = values ** 2
    estimate = float(np.sqrt(np.mean(squared)))
    if values.size < 2 or estimate == 0.0:
        return {"rmse": estimate, "ci95": [estimate, estimate], "count": int(values.size)}
    standard_error = float(np.std(squared, ddof=1) / np.sqrt(values.size) / (2.0 * estimate))
    return {
        "rmse": estimate,
        "ci95": [max(0.0, estimate - 1.96 * standard_error), estimate + 1.96 * standard_error],
        "count": int(values.size),
        "ci_method": "delta_method_on_mean_squared_error",
    }


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return {"mean": mean, "ci95": [mean - half, mean + half], "count": int(values.size)}


def summarize(observation, stochastic, dt_rows, pair_rows, pair_selection):
    summary = {"observation_noise": [], "process_noise": [], "particle_number": [],
               "time_step": [], "pair_count": [], "diagnostics": {}}
    for level in sorted({row["sigma_obs"] for row in observation}):
        subset = [row for row in observation if row["sigma_obs"] == level]
        summary["observation_noise"].append({
            "sigma_obs": level,
            "lasso_rmse_theta": rmse_with_ci(
                [row["coefficient_error"] for row in subset]
            ),
            "ols_rmse_theta": rmse_with_ci(
                [row["ols_coefficient_error"] for row in subset]
            ),
            "euler_rmse_theta": rmse_with_ci(
                [row["euler_coefficient_error"] for row in subset]
            ),
            "repetitions": len(subset),
        })
    for kind, target in (("sigma", "process_noise"), ("particles", "particle_number")):
        for value in sorted({row["value"] for row in stochastic if row["kind"] == kind}):
            subset = [row for row in stochastic if row["kind"] == kind and row["value"] == value]
            summary[target].append({
                "value": value,
                "lasso_rmse_theta": rmse_with_ci(
                    [row["coefficient_error"] for row in subset]
                ),
                "ols_rmse_theta": rmse_with_ci(
                    [row["ols_coefficient_error"] for row in subset]
                ),
                "repetitions": len(subset),
            })
    for row in sorted(dt_rows, key=lambda item: item["dt"]):
        summary["time_step"].append({
            "dt": row["dt"], "lasso_error_theta": row["coefficient_error"],
            "ols_error_theta": row["ols_coefficient_error"],
        })
    for P in sorted({row["P"] for row in pair_rows}):
        subset = [row for row in pair_rows if row["P"] == P]
        summary["pair_count"].append({
            "P": P,
            "lasso_theta_to_full_sq": mean_ci([row["theta_to_full_sq"] for row in subset]),
            "ols_theta_to_full_sq": mean_ci([
                row["ols_theta_to_full_sq_recomputed"] for row in subset
            ]),
        })
    selected_rows = observation + stochastic + dt_rows
    summary["diagnostics"] = {
        "selected_models": len(selected_rows) + 1,
        "boundary_selections": int(
            sum(row["selected_at_grid_boundary"] for row in selected_rows)
            + bool(pair_selection["selected_at_grid_boundary"])
        ),
        "unconverged_path_candidates": int(
            sum(row["unconverged_candidates"] for row in selected_rows)
            + sum(not row["converged"] for row in pair_selection["path"])
        ),
        "pair_fits": len(pair_rows),
        "pair_nonconverged": int(sum(not row["converged"] for row in pair_rows)),
    }
    return summary


def verify_against_archives(spec, observation, stochastic, dt_rows, pair_rows, smoke):
    tolerance = 2e-9
    if spec["task"] == "qanr_redundant6_lasso_sweeps":
        return {
            "passed": True,
            "smoke": bool(smoke),
            "tolerance": tolerance,
            "archived_coefficient_comparison": "not_applicable_to_redundant_dictionary",
            "trajectory_protocol": "identical_seeds_and_physical_settings_to_matched_sweep",
            "finite_ols_models": int(sum(
                np.isfinite(float(row["ols_coefficient_error"]))
                for row in observation + stochastic + dt_rows
            )),
            "total_ols_models": len(observation) + len(stochastic) + len(dt_rows),
            "pair_models": len(pair_rows),
        }
    sources = spec["archived_ols_sources"]
    checks = {"smoke": bool(smoke), "tolerance": tolerance}

    archived = read_csv(ROOT / sources["observation_noise"])
    lookup = {
        (int(row["seed"]), float(row["sigma_obs"])): row
        for row in archived if row["method"] == "weak_form"
    }
    obs_differences = []
    for row in observation:
        key = (int(row["seed_index"]), float(row["sigma_obs"]))
        if key not in lookup:
            continue
        reference = lookup[key]
        obs_differences.extend([
            abs(row["ols_theta_1_recomputed"] - float(reference["theta_1"])),
            abs(row["ols_theta_2_recomputed"] - float(reference["theta_2"])),
        ])
    checks["observation_archived_models_compared"] = len(obs_differences) // 2
    checks["observation_max_abs_ols_difference"] = max(obs_differences, default=0.0)

    archived = read_csv(ROOT / sources["stochastic"])
    lookup = {
        (row["kind"], int(row["seed_index"]),
         float(row["sigma"]) if row["kind"] == "sigma" else int(row["particles"])): row
        for row in archived
    }
    stochastic_differences = []
    for row in stochastic:
        key_value = float(row["sigma"]) if row["kind"] == "sigma" else int(row["particles"])
        key = (row["kind"], int(row["seed_index"]), key_value)
        if key not in lookup:
            continue
        reference = lookup[key]
        stochastic_differences.extend([
            abs(row["ols_theta_1_recomputed"] - float(reference["theta_1"])),
            abs(row["ols_theta_2_recomputed"] - float(reference["theta_2"])),
        ])
    checks["stochastic_archived_models_compared"] = len(stochastic_differences) // 2
    checks["stochastic_max_abs_ols_difference"] = max(stochastic_differences, default=0.0)

    archived = read_csv(ROOT / sources["time_step"])
    lookup = {float(row["dt"]): row for row in archived}
    dt_differences = []
    for row in dt_rows:
        reference = lookup[float(row["dt"])]
        dt_differences.extend([
            abs(row["ols_theta_1_recomputed"] - float(reference["theta_1"])),
            abs(row["ols_theta_2_recomputed"] - float(reference["theta_2"])),
        ])
    checks["time_step_max_abs_ols_difference"] = max(dt_differences, default=0.0)

    archived = read_csv(ROOT / sources["pair_count"])
    lookup = {(int(row["P"]), int(row["pair_seed"])): row for row in archived}
    pair_differences = []
    for row in pair_rows:
        reference = lookup[(int(row["P"]), int(row["pair_seed"]))]
        pair_differences.append(abs(
            row["ols_theta_to_full_sq_recomputed"] - float(reference["theta_to_full_sq"])
        ))
    checks["pair_metric_max_abs_ols_difference"] = max(pair_differences, default=0.0)
    checks["passed"] = all(
        value <= tolerance for key, value in checks.items()
        if key.endswith("difference")
    )
    if not checks["passed"]:
        raise ValueError(f"archived OLS reproduction failed: {checks}")
    return checks


def source_snapshot(out, config_path):
    source_dir = Path(out) / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative] = digest(source)
    shutil.copy2(config_path, source_dir / "requested_protocol.json")
    hashes[str(Path(config_path).relative_to(ROOT))] = digest(config_path)
    return hashes


def run(config_path, out, workers=1, smoke=False):
    requested = read_json(config_path)
    validate_protocol(requested)
    spec = effective_protocol(requested, smoke)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "frozen_protocol.json", spec)
    hashes = source_snapshot(out, config_path)
    report = {
        "status": "running", "smoke_only": bool(smoke), "source_sha256": hashes,
        "frozen_protocol_sha256": digest(out / "frozen_protocol.json"),
        "started_utc": datetime.now(timezone.utc).isoformat(), "workers": workers,
    }
    write_json(out / "report.json", report)
    started = perf_counter()
    try:
        obs_payloads = [(index, spec) for index in range(spec["observation_noise"]["repetitions"])]
        obs_jobs = execute_jobs(out, "observation", obs_payloads, observation_worker, workers)
        observation = [row for result in obs_jobs for row in result["rows"]]

        stochastic_payloads = []
        for value in spec["process_noise"]["values"]:
            stochastic_payloads.extend(
                ("sigma", value, index, spec)
                for index in range(spec["process_noise"]["repetitions"])
            )
        for value in spec["particle_number"]["values"]:
            stochastic_payloads.extend(
                ("particles", value, index, spec)
                for index in range(spec["particle_number"]["repetitions"])
            )
        stochastic = execute_jobs(
            out, "stochastic", stochastic_payloads, stochastic_worker, workers
        )

        dt_payloads = [(value, spec) for value in spec["time_step"]["values"]]
        dt_rows = execute_jobs(out, "time_step", dt_payloads, time_step_worker, workers)

        pair_path = out / "jobs/pair/pair.json"
        if pair_path.exists():
            pair_result = read_json(pair_path)
        else:
            pair_result = run_pair_sweep(spec)
            write_json(pair_path, pair_result)
        pair_rows = pair_result["rows"]

        write_csv(out / "observation_lasso.csv", observation)
        write_csv(out / "stochastic_lasso.csv", stochastic)
        write_csv(out / "time_step_lasso.csv", dt_rows)
        write_csv(out / "pair_lasso.csv", pair_rows)
        write_json(out / "pair_full_selection.json", {
            "selection": pair_result["selection"], "theta_full": pair_result["theta_full"]
        })
        summary = summarize(
            observation, stochastic, dt_rows, pair_rows, pair_result["selection"]
        )
        write_json(out / "summary.json", summary)
        verification = verify_against_archives(
            requested, observation, stochastic, dt_rows, pair_rows, smoke
        )
        write_json(out / "verification.json", verification)
        report.update({
            "status": "smoke_completed" if smoke else "completed",
            "counts": {
                "observation_models": len(observation),
                "stochastic_models": len(stochastic),
                "time_step_models": len(dt_rows),
                "pair_models": len(pair_rows),
            },
            "summary_sha256": digest(out / "summary.json"),
            "verification_sha256": digest(out / "verification.json"),
        })
    except Exception as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/qanr_matched_lasso" / (
        ("smoke-" if args.smoke else "sweeps-") + timestamp
    )
    report = run(config_path, destination, workers=args.workers, smoke=args.smoke)
    print(json.dumps({"status": report["status"], "output": str(destination)}))


if __name__ == "__main__":
    main()
