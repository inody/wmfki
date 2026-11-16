"""Run the approved 10D Morse robustness sweeps in an isolated output tree.

Raw LASSO is selected only by an independent validation weak residual.  Raw
minimum-norm OLS uses the identical training system.  A third, latent test
trajectory is accessed only after coefficients have been frozen.
"""

from __future__ import annotations

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

from kernels import make_kernel_basis, make_kernel_evaluator, make_radial_basis
from pipeline import fit_kernel_from_particles
from scripts.run_1d_hat_lasso import select_lasso
from simulate import make_true_kernel

DEFAULT_CONFIG = ROOT / "configs/experiments/morse10d_sweeps.json"
SOURCE_FILES = (
    "configs/experiments/morse10d_sweeps.json",
    "scripts/run_morse10d_sweeps.py",
    "test/test_morse10d_sweeps.py",
    "scripts/run_1d_hat_lasso.py",
    "pipeline.py",
    "features.py",
    "kernels.py",
    "simulate.py",
    "regression.py",
    "tests.py",
    "time_tests.py",
    "utils.py",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(spec):
    if (spec.get("task") != "morse10d_robustness_sweeps"
            or spec.get("protocol_version") != 1 or spec.get("smoke_only") is not False):
        raise ValueError("invalid scientific Morse sweep protocol")
    if spec.get("model") != {
        "dimension": 10, "particles": 500, "steps": 2000, "dt": 0.01,
        "observation_stride": 5, "trajectories_per_split": 1, "sigma": 0.1,
        "initial_law": "normal", "initial_scale": 0.5, "solver": "euler",
        "interaction": {"mode": "morse_r", "a": 2.0, "b": 1.0,
                        "l1": 1.0, "l2": 2.0},
    }:
        raise ValueError("current Morse model or horizon changed")
    if spec.get("dictionary") != {
        "basis": "exp_r", "M": 8,
        "ell_grid": [0.6, 0.787, 1.032, 1.353, 1.774, 2.326, 3.05, 4.0],
    }:
        raise ValueError("current Morse dictionary changed")
    if spec.get("weak_tests") != {
        "spatial_type": "gaussian", "spatial_count": 25,
        "spatial_scale_factor": 1.5, "spatial_layout": "random",
        "spatial_quantiles": [0.01, 0.99], "temporal_type": "bump",
        "temporal_count": 8, "temporal_support_factor": 1.8,
        "normalize_rows": True,
    }:
        raise ValueError("current weak tests changed")
    if spec.get("sweeps") != {
        "process_sigma": [0.0, 0.025, 0.05, 0.1, 0.2],
        "particle_number": [100, 200, 350, 500],
        "sampled_pairs": [500, 1000, 3000, 5000, 10000],
    }:
        raise ValueError("approved sweep grid changed")
    if spec["pair_sampling"] != {
        "baseline_pairs": 5000, "mode": "resample_each_time",
        "with_replacement": True, "exclude_diagonal": True,
    }:
        raise ValueError("pair sampling protocol changed")
    required = {
        "method": "raw_lasso", "alpha_count": 60, "alpha_min_ratio": 1e-7,
        "max_iter": 200000, "tol": 1e-8, "fit_intercept": False,
        "standardize": False, "selection": "cyclic",
        "report_nonzero_threshold": 1e-8,
        "selection_metric": "independent_validation_weak_residual_mse",
        "truth_used_for_selection": False, "test_error_used_for_selection": False,
        "post_lasso_refit": False, "coefficient_thresholding": False,
    }
    if spec.get("estimator") != required:
        raise ValueError("raw LASSO protocol changed")
    if spec["evaluation"].get("scientific_repetitions") != 20:
        raise ValueError("scientific repetition count changed")
    if spec.get("pair_penalty_policy") != (
        "absolute_alpha_selected_at_Pmax_by_independent_validation_then_fixed_"
        "across_all_P_within_repetition"
    ):
        raise ValueError("P-sweep alpha policy changed")


def effective_protocol(spec, smoke=False):
    effective = json.loads(json.dumps(spec))
    effective["smoke_only"] = bool(smoke)
    if not smoke:
        return effective
    effective["model"].update({"particles": 24, "steps": 12, "observation_stride": 1})
    effective["weak_tests"].update({"spatial_count": 4, "temporal_count": 3})
    effective["pair_sampling"]["baseline_pairs"] = 60
    effective["sweeps"] = {
        "process_sigma": [0.0, 0.1],
        "particle_number": [12, 24],
        "sampled_pairs": [20, 60],
    }
    effective["estimator"].update({"alpha_count": 8, "max_iter": 20000})
    effective["evaluation"].update({
        "scientific_repetitions": 1, "profile_grid_size": 41,
        "latent_drift_time_count": 3, "latent_drift_target_count": 6,
    })
    return effective


def repetition_plan(spec):
    seeds = spec["seeds"]
    return [{
        "index": index,
        "training_seed": seeds["trajectory_start"] + seeds["repetition_stride"] * index,
        "validation_seed": seeds["trajectory_start"] + seeds["repetition_stride"] * index
        + seeds["validation_offset"],
        "test_seed": seeds["trajectory_start"] + seeds["repetition_stride"] * index
        + seeds["test_offset"],
        "training_pair_seed": seeds["pair_start"] + index,
        "validation_pair_seed": seeds["pair_start"] + index
        + seeds["pair_validation_offset"],
        "space_seed": seeds["space_start"] + index,
        "evaluation_seed": seeds["evaluation_start"] + index,
    } for index in range(spec["evaluation"]["scientific_repetitions"])]


def physical_conditions(spec):
    base_n, base_sigma = spec["model"]["particles"], spec["model"]["sigma"]
    values = [(base_n, float(value)) for value in spec["sweeps"]["process_sigma"]]
    values += [(int(value), float(base_sigma)) for value in spec["sweeps"]["particle_number"]]
    return list(dict.fromkeys(values))


def _trajectory_path(out, particles, sigma, split, index):
    sigma_key = format(float(sigma), ".12g").replace("-", "m").replace(".", "p")
    return Path(out) / "cache" / "trajectories" / f"N{particles}-sigma{sigma_key}" / split / f"rep-{index:03d}.npz"


def _radial_drift_fast(x, interaction):
    """Exact all-pair radial drift without materializing an N x N x d tensor."""
    x = np.asarray(x, dtype=float)
    squared_norm = np.einsum("nd,nd->n", x, x)
    squared_distance = (
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * (x @ x.T)
    )
    radius = np.sqrt(np.maximum(squared_distance, 0.0))
    _, _, _, radial_profile = make_true_kernel(interaction)
    radial = radial_profile(radius)
    return (x * radial.sum(axis=1, keepdims=True) - radial @ x) / x.shape[0]


def _morse_drift_fast(x, interaction):
    """Compatibility wrapper for the parity-tested Morse simulator."""
    if interaction.get("mode") != "morse_r":
        raise ValueError("Morse drift requires mode='morse_r'")
    return _radial_drift_fast(x, interaction)


def _simulate_truth(spec, particles, sigma, seed):
    model = spec["model"]
    rng = np.random.default_rng(seed)
    steps, dimension = int(model["steps"]), int(model["dimension"])
    dt = float(model["dt"])
    time = np.linspace(0.0, steps * dt, steps + 1)
    X = np.empty((1, steps + 1, particles, dimension), dtype=float)
    if model["initial_law"] == "normal":
        x = rng.normal(0.0, model["initial_scale"], size=(particles, dimension))
    elif model["initial_law"] == "normal_scale_mixture":
        scales = np.asarray(model["initial_scales"], dtype=float)
        if scales.ndim != 1 or scales.size < 2 or np.any(scales <= 0.0):
            raise ValueError("normal_scale_mixture requires positive initial_scales")
        particle_scales = rng.choice(scales, size=particles, replace=True)
        x = rng.normal(size=(particles, dimension)) * particle_scales[:, None]
    else:
        raise ValueError(f"unsupported pilot initial law: {model['initial_law']}")
    X[0, 0] = x
    if model["interaction"].get("mode") not in {"morse_r", "gaussian_shell"}:
        raise ValueError("fast simulator requires a supported radial interaction")
    for step in range(steps):
        drift = _radial_drift_fast(x, model["interaction"])
        increment = rng.normal(0.0, 1.0, size=x.shape)
        x = x + dt * drift + float(sigma) * np.sqrt(dt) * increment
        X[0, step + 1] = x
    return X, time


def load_or_simulate(out, spec, particles, sigma, split, plan):
    path = _trajectory_path(out, particles, sigma, split, plan["index"])
    if path.exists():
        with np.load(path) as saved:
            return saved["X"], saved["time"], True
    seed = plan[f"{split}_seed"]
    started = perf_counter()
    X, time = _simulate_truth(spec, particles, sigma, seed)
    elapsed = perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, X=X, time=time, simulation_seconds=np.asarray(elapsed)
        )
    temporary.replace(path)
    return X, time, False


def assembly_config(spec, particles, sigma, trajectory_seed, pair_seed, space_seed, pairs):
    model, weak = spec["model"], spec["weak_tests"]
    return {
        "simulation": {
            "N": int(particles), "T": int(model["steps"]), "dt": model["dt"],
            "R": 1, "sigma": float(sigma), "d": model["dimension"],
            "init": model["initial_law"], "init_scale": model["initial_scale"],
            "seed": int(trajectory_seed), "solver": model["solver"],
            "interaction": model["interaction"],
        },
        "space_tests": {
            "type": weak["spatial_type"], "L": weak["spatial_count"],
            "s_factor": weak["spatial_scale_factor"],
            "layout": weak["spatial_layout"], "seed": int(space_seed),
            "quantiles": weak["spatial_quantiles"],
        },
        "time_tests": {
            "type": weak["temporal_type"], "K": weak["temporal_count"],
            "s_factor": weak["temporal_support_factor"],
        },
        "pair_sampling": {
            "total_pairs": int(pairs), "with_replacement": True,
            "exclude_diagonal": True, "seed": int(pair_seed),
            "mode": spec["pair_sampling"]["mode"],
        },
        "kernel_basis": spec["dictionary"],
        "weak_form": {"normalize_rows": weak["normalize_rows"]},
        # The pipeline fit is discarded; this inexpensive ridge call only exposes A,b.
        "regression": {"method": "ridge", "standardize": False, "alpha": 1.0},
        "diagnostics": {"method": None},
        "output": {"save_plots": False},
    }


def assemble_system(spec, X, time, particles, sigma, trajectory_seed, pair_seed,
                    space_seed, pairs):
    config = assembly_config(
        spec, particles, sigma, trajectory_seed, pair_seed, space_seed, pairs
    )
    started = perf_counter()
    fit = fit_kernel_from_particles(
        config, X, time, return_system=True,
        pair_mode=spec["pair_sampling"]["mode"], pair_seed=pair_seed,
    )
    elapsed = perf_counter() - started
    A = np.asarray(fit["system"]["A_flat"], dtype=float)
    b = np.asarray(fit["system"]["b_flat"], dtype=float)
    return A, b, {
        "assembly_seconds": elapsed,
        "response_seconds": fit["diagnostics"]["timing"]["response_seconds"],
        "design_seconds": fit["diagnostics"]["timing"]["design_seconds"],
    }


def design_diagnostics(A):
    singular = np.linalg.svd(A, compute_uv=False)
    rank = int(np.linalg.matrix_rank(A))
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else None
    ata_condition = float(np.linalg.cond(A.T @ A))
    return {
        "design_rows": int(A.shape[0]), "design_columns": int(A.shape[1]),
        "design_rank": rank, "rank_deficient": bool(rank < A.shape[1]),
        "condition_A": condition if condition is not None and np.isfinite(condition) else None,
        "condition_AtA": ata_condition if np.isfinite(ata_condition) else None,
    }


def fit_fixed_alpha(A, b, alpha, settings, initial_theta=None):
    warm_start = initial_theta is not None
    model = Lasso(
        alpha=float(alpha), fit_intercept=False, warm_start=warm_start,
        max_iter=settings["max_iter"], tol=settings["tol"], selection="cyclic",
    )
    if warm_start:
        initial_theta = np.asarray(initial_theta, dtype=float)
        if initial_theta.shape != (A.shape[1],) or not np.all(np.isfinite(initial_theta)):
            raise ValueError("initial LASSO coefficients are incompatible or nonfinite")
        model.coef_ = initial_theta.copy()
        model.intercept_ = 0.0
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(A, b)
    elapsed = perf_counter() - started
    theta = np.asarray(model.coef_, dtype=float)
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    finite = bool(np.all(np.isfinite(theta)) and np.isfinite(model.dual_gap_))
    return theta, {
        "converged": bool(finite and warning_count == 0),
        "convergence_warnings": int(warning_count), "iterations": int(model.n_iter_),
        "dual_gap": float(model.dual_gap_) if finite else None,
        "fit_seconds": elapsed,
    }


def select_raw_lasso(A, b, Av, bv, settings):
    started = perf_counter()
    selection, _ = select_lasso(A, b, Av, bv, settings)
    elapsed = perf_counter() - started
    if selection["status"] != "selected":
        return None, selection, {
            "converged": False, "unconverged_candidates": len(selection["path"]),
            "selected_at_grid_boundary": False, "alpha": None,
            "iterations": None, "fit_seconds": elapsed,
        }
    selected = selection["selected"]
    return np.asarray(selection["theta"], dtype=float), selection, {
        "converged": True,
        "unconverged_candidates": int(sum(not row["converged"] for row in selection["path"])),
        "selected_at_grid_boundary": bool(selection["selected_at_grid_boundary"]),
        "alpha": float(selected["alpha_sk"]), "alpha_index": int(selected["index"]),
        "alpha_over_max": float(selected["alpha_over_max"]),
        "iterations": int(selected["iterations"]),
        "validation_weak_residual_mse": float(selected["validation_mse"]),
        "fit_seconds": elapsed,
    }


def latent_evaluation_system(spec, X, evaluation_seed):
    evaluation = spec["evaluation"]
    rng = np.random.default_rng(evaluation_seed)
    _, time_count, particles, dimension = X.shape
    time_indices = np.linspace(
        0, time_count - 1, min(evaluation["latent_drift_time_count"], time_count), dtype=int
    )
    target_count = min(evaluation["latent_drift_target_count"], particles)
    basis, _ = make_kernel_basis(spec["dictionary"])
    true_kernel, _, _, true_profile = make_true_kernel(spec["model"]["interaction"])
    rows, truth = [], []
    for time_index in time_indices:
        targets = rng.choice(particles, size=target_count, replace=False)
        z = X[0, time_index, targets, None, :] - X[0, time_index, None, :, :]
        basis_drift = basis(z).mean(axis=1)  # targets x atoms x dimension
        true_drift = true_kernel(z).mean(axis=1)
        rows.append(basis_drift.transpose(0, 2, 1).reshape(-1, spec["dictionary"]["M"]))
        truth.append(true_drift.reshape(-1))
    radius = np.linspace(*evaluation["profile_radius"], evaluation["profile_grid_size"])
    radial_basis, _ = make_radial_basis(spec["dictionary"])
    return np.vstack(rows), np.concatenate(truth), radial_basis(radius), true_profile(radius)


def relative_error(estimate, truth):
    denominator = float(np.sum(np.asarray(truth) ** 2))
    numerator = float(np.sum((np.asarray(estimate) - np.asarray(truth)) ** 2))
    if denominator <= 0 or not np.isfinite(numerator + denominator):
        return None
    return float(np.sqrt(numerator / denominator))


def evaluate_theta(theta, drift_A, drift_truth, profile_A, profile_truth):
    if theta is None or not np.all(np.isfinite(theta)):
        return {"latent_drift_relative_error": None, "profile_relative_error": None,
                "numerical_divergence": True}
    drift_error = relative_error(drift_A @ theta, drift_truth)
    profile_error = relative_error(profile_A @ theta, profile_truth)
    return {
        "latent_drift_relative_error": drift_error,
        "profile_relative_error": profile_error,
        "numerical_divergence": bool(drift_error is None or profile_error is None),
    }


def method_records(
    A, b, Av, bv, evaluation_system, settings, common, *,
    fixed_alpha=None, initial_theta=None,
):
    diagnostics = design_diagnostics(A)
    drift_A, drift_truth, profile_A, profile_truth = evaluation_system
    paths = {}
    if fixed_alpha is None:
        theta_lasso, selection, lasso_info = select_raw_lasso(A, b, Av, bv, settings)
        paths["lasso"] = selection
    else:
        theta_lasso, fixed = fit_fixed_alpha(
            A, b, fixed_alpha, settings, initial_theta=initial_theta
        )
        lasso_info = {
            **fixed, "unconverged_candidates": int(not fixed["converged"]),
            "selected_at_grid_boundary": False, "alpha": float(fixed_alpha),
            "alpha_index": None, "alpha_over_max": None,
            "validation_weak_residual_mse": float(np.mean((Av @ theta_lasso - bv) ** 2)),
        }
    started = perf_counter()
    theta_ols, _, ols_rank, singular = np.linalg.lstsq(A, b, rcond=None)
    ols_seconds = perf_counter() - started
    ols_finite = bool(np.all(np.isfinite(theta_ols)))
    records = []
    for method, theta, fit_info in (
        ("lasso", theta_lasso, lasso_info),
        ("ols", theta_ols, {
            "converged": ols_finite, "unconverged_candidates": 0,
            "selected_at_grid_boundary": False, "alpha": None,
            "alpha_index": None, "alpha_over_max": None,
            "iterations": None, "fit_seconds": ols_seconds,
            "validation_weak_residual_mse": float(np.mean((Av @ theta_ols - bv) ** 2))
            if ols_finite else None,
        }),
    ):
        metrics = evaluate_theta(theta, drift_A, drift_truth, profile_A, profile_truth)
        records.append({
            **common, "method": method,
            "coefficients": None if theta is None else [float(x) for x in theta],
            "nonzero_reported": None if theta is None else int(np.count_nonzero(
                np.abs(theta) > settings["report_nonzero_threshold"])),
            **diagnostics, **fit_info, **metrics,
            "ols_lstsq_rank": int(ols_rank) if method == "ols" else None,
            "ols_smallest_singular_value": float(singular[-1])
            if method == "ols" and singular.size else None,
        })
    return records, paths


def _tag_standard_records(records, spec, particles, sigma):
    tagged = []
    base_n, base_sigma = spec["model"]["particles"], spec["model"]["sigma"]
    if particles == base_n and sigma in spec["sweeps"]["process_sigma"]:
        tagged.extend([{**row, "sweep": "process_sigma", "sweep_value": sigma}
                       for row in records])
    if sigma == base_sigma and particles in spec["sweeps"]["particle_number"]:
        tagged.extend([{**row, "sweep": "particle_number", "sweep_value": particles}
                       for row in records])
    return tagged


def run_repetition(payload):
    out_string, spec, plan = payload
    out = Path(out_string)
    records, selections, trajectory_status = [], {}, []
    base_n, base_sigma = spec["model"]["particles"], spec["model"]["sigma"]
    baseline_data = None
    with threadpool_limits(limits=1):
        for particles, sigma in physical_conditions(spec):
            split_data = {}
            simulation_started = perf_counter()
            for split in ("training", "validation", "test"):
                X, time, cached = load_or_simulate(
                    out, spec, particles, sigma, split, plan
                )
                split_data[split] = (X, time)
                trajectory_status.append({
                    "particles": particles, "sigma": sigma, "split": split,
                    "cache_hit": cached,
                })
            simulation_seconds = perf_counter() - simulation_started
            observation_stride = int(spec["model"]["observation_stride"])
            observed = {
                split: (data[0][:, ::observation_stride], data[1][::observation_stride])
                for split, data in split_data.items()
            }
            train_A, train_b, train_timing = assemble_system(
                spec, *observed["training"], particles, sigma,
                plan["training_seed"], plan["training_pair_seed"],
                plan["space_seed"], spec["pair_sampling"]["baseline_pairs"],
            )
            validation_A, validation_b, validation_timing = assemble_system(
                spec, *observed["validation"], particles, sigma,
                plan["validation_seed"], plan["validation_pair_seed"],
                plan["space_seed"], spec["pair_sampling"]["baseline_pairs"],
            )
            evaluation_system = latent_evaluation_system(
                spec, split_data["test"][0], plan["evaluation_seed"]
            )
            fit_id = f"rep{plan['index']:03d}-N{particles}-sigma{sigma:g}-Pbaseline"
            common = {
                "repetition": plan["index"], "fit_id": fit_id,
                "particles": particles, "sigma": sigma,
                "pairs": spec["pair_sampling"]["baseline_pairs"],
                "training_seed": plan["training_seed"],
                "validation_seed": plan["validation_seed"], "test_seed": plan["test_seed"],
                "training_pair_seed": plan["training_pair_seed"],
                "validation_pair_seed": plan["validation_pair_seed"],
                "simulation_and_cache_seconds": simulation_seconds,
                "training_assembly_seconds": train_timing["assembly_seconds"],
                "validation_assembly_seconds": validation_timing["assembly_seconds"],
            }
            fitted, paths = method_records(
                train_A, train_b, validation_A, validation_b, evaluation_system,
                spec["estimator"], common,
            )
            records.extend(_tag_standard_records(fitted, spec, particles, sigma))
            selections[fit_id] = paths
            if particles == base_n and sigma == base_sigma:
                baseline_data = (observed, evaluation_system)

        if baseline_data is None:
            raise RuntimeError("baseline trajectory was not generated")
        split_data, evaluation_system = baseline_data
        pmax = max(spec["sweeps"]["sampled_pairs"])
        pmax_A, pmax_b, pmax_train_timing = assemble_system(
            spec, *split_data["training"], base_n, base_sigma,
            plan["training_seed"], plan["training_pair_seed"], plan["space_seed"], pmax,
        )
        pmax_Av, pmax_bv, pmax_validation_timing = assemble_system(
            spec, *split_data["validation"], base_n, base_sigma,
            plan["validation_seed"], plan["validation_pair_seed"], plan["space_seed"], pmax,
        )
        selected_theta, p_selection, selected_info = select_raw_lasso(
            pmax_A, pmax_b, pmax_Av, pmax_bv, spec["estimator"]
        )
        selections[f"rep{plan['index']:03d}-Pmax-selection"] = {"lasso": p_selection}
        alpha = selected_info["alpha"]
        for pairs in spec["sweeps"]["sampled_pairs"]:
            if pairs == pmax:
                A, b, train_timing = pmax_A, pmax_b, pmax_train_timing
            else:
                A, b, train_timing = assemble_system(
                    spec, *split_data["training"], base_n, base_sigma,
                    plan["training_seed"], plan["training_pair_seed"],
                    plan["space_seed"], pairs,
                )
            common = {
                "repetition": plan["index"],
                "fit_id": f"rep{plan['index']:03d}-P{pairs}",
                "sweep": "sampled_pairs", "sweep_value": pairs,
                "particles": base_n, "sigma": base_sigma, "pairs": pairs,
                "training_seed": plan["training_seed"],
                "validation_seed": plan["validation_seed"], "test_seed": plan["test_seed"],
                "training_pair_seed": plan["training_pair_seed"],
                "validation_pair_seed": plan["validation_pair_seed"],
                "simulation_and_cache_seconds": 0.0,
                "training_assembly_seconds": train_timing["assembly_seconds"],
                "validation_assembly_seconds": pmax_validation_timing["assembly_seconds"],
                "pmax_selected_alpha": alpha,
                "pair_design_relative_error": float(
                    np.linalg.norm(A - pmax_A) / max(np.linalg.norm(pmax_A), 1e-300)
                ),
            }
            if alpha is None:
                # Preserve a failed Pmax selection instead of choosing with truth/test data.
                failed = [{
                    **common, "method": "lasso", "coefficients": None,
                    "nonzero_reported": None, **design_diagnostics(A),
                    "converged": False, "unconverged_candidates": 1,
                    "selected_at_grid_boundary": False, "alpha": None,
                    "alpha_index": None, "alpha_over_max": None, "iterations": None,
                    "fit_seconds": 0.0, "validation_weak_residual_mse": None,
                    "latent_drift_relative_error": None, "profile_relative_error": None,
                    "numerical_divergence": True, "ols_lstsq_rank": None,
                    "ols_smallest_singular_value": None,
                }]
                # Still obtain the mandated OLS comparison on this system.
                both, _ = method_records(
                    A, b, pmax_Av, pmax_bv, evaluation_system,
                    spec["estimator"], common, fixed_alpha=1.0,
                )
                records.extend(failed + [row for row in both if row["method"] == "ols"])
            else:
                fitted, _ = method_records(
                    A, b, pmax_Av, pmax_bv, evaluation_system,
                    spec["estimator"], common, fixed_alpha=alpha,
                    initial_theta=selected_theta,
                )
                records.extend(fitted)
        # Completed jobs are self-contained; remove large latent arrays only after
        # all metrics have been written into the returned record. Failed jobs keep
        # their cache and can resume without repeating completed simulations.
        for particles, sigma in physical_conditions(spec):
            for split in ("training", "validation", "test"):
                _trajectory_path(out, particles, sigma, split, plan["index"]).unlink(
                    missing_ok=True
                )
    return {"index": plan["index"], "records": records,
            "selections": selections, "trajectory_cache": trajectory_status}


def finite_summary(values):
    array = np.asarray([value for value in values if value is not None], dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "count": len(values), "finite_count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "q25": float(np.quantile(finite, 0.25)) if finite.size else None,
        "q75": float(np.quantile(finite, 0.75)) if finite.size else None,
    }


def summarize(records):
    groups = []
    for sweep in ("process_sigma", "particle_number", "sampled_pairs"):
        values = sorted({row["sweep_value"] for row in records if row["sweep"] == sweep})
        for value in values:
            for method in ("lasso", "ols"):
                subset = [row for row in records if row["sweep"] == sweep
                          and row["sweep_value"] == value and row["method"] == method]
                groups.append({
                    "sweep": sweep, "value": value, "method": method,
                    "repetitions": len(subset),
                    "latent_drift_relative_error": finite_summary(
                        [row["latent_drift_relative_error"] for row in subset]),
                    "profile_relative_error": finite_summary(
                        [row["profile_relative_error"] for row in subset]),
                    "rank_deficient": int(sum(row["rank_deficient"] for row in subset)),
                    "nonconverged": int(sum(not row["converged"] for row in subset)),
                    "boundary_selections": int(sum(row["selected_at_grid_boundary"] for row in subset)),
                    "numerical_divergence": int(sum(row["numerical_divergence"] for row in subset)),
                })
    return {
        "groups": groups,
        "diagnostics": {
            "records": len(records),
            "unique_fits": len({row["fit_id"] for row in records}),
            "nonconverged": int(sum(not row["converged"] for row in records)),
            "boundary_selections": int(sum(row["selected_at_grid_boundary"] for row in records)),
            "rank_deficient": int(sum(row["rank_deficient"] for row in records)),
            "numerical_divergence": int(sum(row["numerical_divergence"] for row in records)),
        },
    }


def flatten_record(record):
    row = {key: value for key, value in record.items() if key != "coefficients"}
    coefficients = record.get("coefficients")
    for index in range(8):
        row[f"theta_{index + 1}"] = None if coefficients is None else coefficients[index]
    return row


def write_csv(path, records):
    rows = [flatten_record(record) for record in records]
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in known:
                fieldnames.append(key)
                known.add(key)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def snapshot_sources(out, config_path):
    source_dir = Path(out) / "source"
    hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative] = digest(source)
    config_relative = str(Path(config_path).resolve().relative_to(ROOT))
    hashes[config_relative] = digest(config_path)
    write_json(out / "source_manifest.json", {"sha256": hashes})
    return hashes


def _job_path(out, index):
    return Path(out) / "jobs" / f"repetition-{index:03d}.json"


def run(config_path, out, workers=1, smoke=False, resume=False):
    requested = read_json(config_path)
    validate_protocol(requested)
    spec = effective_protocol(requested, smoke)
    out = Path(out)
    plans = repetition_plan(spec)
    if out.exists():
        if not resume:
            raise FileExistsError(f"output exists; use --resume: {out}")
        if read_json(out / "frozen_protocol.json") != spec:
            raise ValueError("resume protocol does not match frozen protocol")
        frozen_hashes = read_json(out / "source_manifest.json")["sha256"]
        for relative, expected in frozen_hashes.items():
            if relative in SOURCE_FILES and digest(ROOT / relative) != expected:
                raise ValueError(f"source changed since run started: {relative}")
    else:
        out.mkdir(parents=True)
        write_json(out / "frozen_protocol.json", spec)
        snapshot_sources(out, config_path)
    manifest = {
        "task": spec["task"], "smoke_only": bool(smoke),
        "expected_repetitions": len(plans),
        "jobs": [{"index": plan["index"], "path": str(_job_path(out, plan["index"]).relative_to(out)),
                  "completed": _job_path(out, plan["index"]).exists()} for plan in plans],
        "trajectory_cache_policy": spec["data_policy"],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "manifest.json", manifest)
    report = {
        "status": "running", "smoke_only": bool(smoke), "workers": workers,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(out), "resumed": bool(resume),
    }
    write_json(out / "report.json", report)
    started = perf_counter()
    try:
        results = [None] * len(plans)
        missing = []
        for plan in plans:
            path = _job_path(out, plan["index"])
            if path.exists():
                results[plan["index"]] = read_json(path)
            else:
                missing.append(plan)
        payloads = [(str(out), spec, plan) for plan in missing]
        if workers == 1:
            generated, pool = map(run_repetition, payloads), None
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            generated = pool.map(run_repetition, payloads)
        try:
            for count, result in enumerate(generated, 1):
                write_json(_job_path(out, result["index"]), result)
                results[result["index"]] = result
                print(f"Morse repetition {count}/{len(missing)} new jobs", flush=True)
        finally:
            if pool is not None:
                pool.shutdown()
        records = [record for result in results for record in result["records"]]
        write_json(out / "records.json", records)
        write_csv(out / "records.csv", records)
        write_json(out / "summary.json", summarize(records))
        artifact_sha256 = {
            name: digest(out / name)
            for name in (
                "frozen_protocol.json", "source_manifest.json",
                "records.json", "records.csv", "summary.json",
            )
        }
        manifest["jobs"] = [{**job, "completed": True} for job in manifest["jobs"]]
        manifest.update({
            "status": "smoke_completed" if smoke else "completed",
            "record_count": len(records),
            "summary_sha256": artifact_sha256["summary.json"],
            "artifact_sha256": artifact_sha256,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json(out / "manifest.json", manifest)
        report.update({
            "status": manifest["status"], "record_count": len(records),
            "completed_repetitions": len(results),
            "summary_sha256": manifest["summary_sha256"],
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.resume and args.out_dir is None:
        parser.error("--resume requires --out-dir")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/morse10d_sweeps" / (
        ("smoke-" if args.smoke else "sweeps-") + timestamp
    )
    report = run(args.config.resolve(), destination, workers=args.workers,
                 smoke=args.smoke, resume=args.resume)
    print(json.dumps({"status": report["status"], "output": str(destination),
                      "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
