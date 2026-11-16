"""Run the regularized predator--prey weak-identification application.

Penalties for the prey and predator response equations are selected separately
on independent validation trajectories.  Truth and held-out errors are not
available to the selector.  Scientific and smoke outputs are explicitly
separated and source-hashed.
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

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predator_prey import (
    assemble_predator_prey_weak_systems,
    channel_kernels,
    make_predator_prey_bases,
    predator_prey_velocity,
    simulate_predator_prey,
)
from scripts.run_1d_hat_lasso import select_lasso
from tests import make_test_functions
from time_tests import make_time_test_functions


DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "predator_prey_application.json"
SOURCE_FILES = (
    "predator_prey.py",
    "features.py",
    "tests.py",
    "time_tests.py",
    "scripts/run_predator_prey_application.py",
    "scripts/run_1d_hat_lasso.py",
    "configs/experiments/predator_prey_application.json",
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


def validate_protocol(spec):
    if spec.get("task") != "predator_prey_weak_application" or spec.get("smoke_only") is not False:
        raise ValueError("invalid scientific predator--prey protocol")
    expected_model = {
        "dimension": 2, "prey_particles": 100, "a": 1.0, "b": 0.2,
        "c": 0.4, "p": 3.0, "delta": 0.05,
        "prey_initial_box": [-0.5, 0.5], "predator_initial_box": [-0.5, 0.5],
        "solver": "rk4",
    }
    if spec.get("model") != expected_model:
        raise ValueError("predator--prey model changed")
    dictionary = spec["dictionary"]
    if dictionary != {
        "prey_exponential_scales": [0.25, 0.5, 1.0, 2.0],
        "predator_inverse_powers": [1.0, 2.0, 3.0, 4.0, 5.0],
        "prey_prey_true": [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        "prey_predator_true": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        "predator_prey_true": [0.0, 0.0, 0.4, 0.0, 0.0],
    }:
        raise ValueError("predator--prey dictionary changed")
    observation = spec["observation"]
    expected_observation = {
        "training_trajectories": 4, "validation_trajectories": 4,
        "test_trajectories": 1, "horizon": 3.0, "dt": 0.01,
    }
    if {key: observation.get(key) for key in expected_observation} != expected_observation:
        raise ValueError("observation protocol changed")
    if float(observation.get("observation_noise", -1.0)) not in {0.001, 0.01}:
        raise ValueError("unapproved nominal observation noise")
    if spec["weak_tests"] != {
        "spatial_type": "gaussian",
        "grid_coordinates": [-1.5, -0.75, 0.0, 0.75, 1.5],
        "prey_width": 0.5, "predator_width": 0.7,
        "temporal_type": "bump", "temporal_count": 8,
        "temporal_support_factor": 1.8, "normalize_rows": True,
    }:
        raise ValueError("weak tests changed")
    required_estimator = {
        "method": "raw_lasso", "alpha_count": 60, "alpha_min_ratio": 1e-7,
        "max_iter": 200000, "tol": 1e-8, "fit_intercept": False,
        "standardize": False, "selection": "cyclic",
        "selection_metric": "independent_validation_weak_residual_mse",
        "truth_used_for_selection": False, "test_error_used_for_selection": False,
        "post_lasso_refit": False, "coefficient_thresholding": False,
        "report_nonzero_threshold": 1e-8,
    }
    if spec.get("estimator") != required_estimator:
        raise ValueError("estimator protocol changed")
    if spec["evaluation"].get("scientific_repetitions") != 20:
        raise ValueError("scientific repetition count changed")


def effective_protocol(spec, smoke):
    effective = json.loads(json.dumps(spec))
    effective["smoke_only"] = bool(smoke)
    if smoke:
        effective["model"]["prey_particles"] = 30
        effective["observation"].update({
            "training_trajectories": 1,
            "validation_trajectories": 1,
            "test_trajectories": 1,
            "horizon": 0.4,
            "dt": 0.02,
        })
        effective["evaluation"].update({
            "scientific_repetitions": 1,
            "display_particles": 40,
            "display_horizon": 0.6,
            "display_times": [0.0, 0.2, 0.4, 0.6],
        })
    return effective


def model_objects(spec):
    model = spec["model"]
    dictionary = spec["dictionary"]
    prey_basis, predator_basis, metadata = make_predator_prey_bases(
        model["delta"],
        dictionary["prey_exponential_scales"],
        dictionary["predator_inverse_powers"],
    )
    theta_xx = np.asarray(dictionary["prey_prey_true"], dtype=float)
    theta_xz = np.asarray(dictionary["prey_predator_true"], dtype=float)
    theta_zx = np.asarray(dictionary["predator_prey_true"], dtype=float)
    kernels = channel_kernels(
        theta_xx, theta_xz, theta_zx, prey_basis, predator_basis
    )
    return prey_basis, predator_basis, metadata, (theta_xx, theta_xz, theta_zx), kernels


def weak_objects(spec, time):
    weak = spec["weak_tests"]
    coordinates = np.asarray(weak["grid_coordinates"], dtype=float)
    centers = np.array(np.meshgrid(coordinates, coordinates)).reshape(2, -1).T
    prey_tests = make_test_functions({
        "centers": centers, "s": weak["prey_width"], "type": weak["spatial_type"]
    })
    predator_tests = make_test_functions({
        "centers": centers, "s": weak["predator_width"], "type": weak["spatial_type"]
    })
    time_tests = make_time_test_functions({"time_tests": {
        "K": weak["temporal_count"],
        "type": weak["temporal_type"],
        "s_factor": weak["temporal_support_factor"],
    }}, time)
    return prey_tests, predator_tests, time_tests


def simulate_split(spec, repetitions, seed, kernels, *, particle_count=None,
                   horizon=None, initial_prey=None, initial_predator=None):
    model = spec["model"]
    observation = spec["observation"]
    dt = float(observation["dt"])
    if horizon is None:
        horizon = observation["horizon"]
    steps = int(round(float(horizon) / dt))
    return simulate_predator_prey(
        particle_count=model["prey_particles"] if particle_count is None else particle_count,
        steps=steps,
        dt=dt,
        repetitions=repetitions,
        kernels=kernels,
        seed=seed,
        prey_box=model["prey_initial_box"],
        predator_box=model["predator_initial_box"],
        solver=model["solver"],
        initial_prey=initial_prey,
        initial_predator=initial_predator,
    )


def assemble(spec, prey, predator, time, prey_basis, predator_basis):
    prey_tests, predator_tests, time_tests = weak_objects(spec, time)
    return assemble_predator_prey_weak_systems(
        prey,
        predator,
        time,
        prey_tests=prey_tests,
        predator_tests=predator_tests,
        prey_basis=prey_basis,
        predator_basis=predator_basis,
        time_tests=time_tests,
        normalize_rows=spec["weak_tests"]["normalize_rows"],
    )


def selected_fit(training, validation, settings):
    selection, _ = select_lasso(
        training["A_flat"], training["b_flat"],
        validation["A_flat"], validation["b_flat"], settings,
    )
    if selection["status"] != "selected":
        raise RuntimeError("no converged LASSO candidate")
    theta = np.asarray(selection["theta"], dtype=float)
    return theta, selection


def relative_error(estimate, truth):
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    denominator = float(np.linalg.norm(truth))
    return float(np.linalg.norm(estimate - truth) / denominator)


def profile_curves(radius, theta_xx, theta_xz, theta_zx, prey_basis, predator_basis):
    displacement = np.column_stack([radius, np.zeros_like(radius)])
    xx, xz, zx = channel_kernels(
        theta_xx, theta_xz, theta_zx, prey_basis, predator_basis
    )
    return xx(displacement)[:, 0], xz(displacement)[:, 0], zx(displacement)[:, 0]


def drift_error(prey, predator, truth_kernels, estimated_kernels, stride):
    """Return the common species-balanced full-source relative drift error."""
    totals = {
        "prey_error": 0.0, "prey_truth": 0.0,
        "predator_error": 0.0, "predator_truth": 0.0,
    }
    for repetition in range(prey.shape[0]):
        for time_index in range(0, prey.shape[1], int(stride)):
            x = prey[repetition, time_index]
            z = predator[repetition, time_index, 0]
            true_x, true_z = predator_prey_velocity(x, z, truth_kernels)
            fitted_x, fitted_z = predator_prey_velocity(x, z, estimated_kernels)
            totals["prey_error"] += float(np.sum((fitted_x - true_x) ** 2))
            totals["prey_truth"] += float(np.sum(true_x ** 2))
            totals["predator_error"] += float(np.sum((fitted_z - true_z) ** 2))
            totals["predator_truth"] += float(np.sum(true_z ** 2))
    prey_ratio = totals["prey_error"] / totals["prey_truth"]
    predator_ratio = totals["predator_error"] / totals["predator_truth"]
    return {
        "species_balanced_relative": float(np.sqrt(0.5 * (prey_ratio + predator_ratio))),
        "prey_relative": float(np.sqrt(prey_ratio)),
        "predator_relative": float(np.sqrt(predator_ratio)),
        **totals,
    }


def run_repetition(payload):
    index, spec = payload
    seeds = spec["seeds"]
    stride = seeds["repetition_stride"]
    train_seed = seeds["training_start"] + stride * index
    validation_seed = train_seed + seeds["validation_offset"]
    test_seed = train_seed + seeds["test_offset"]
    prey_basis, predator_basis, _, truth, truth_kernels = model_objects(spec)
    observation = spec["observation"]

    with threadpool_limits(limits=1):
        train_x, train_z, time, train_diagnostics = simulate_split(
            spec, observation["training_trajectories"], train_seed, truth_kernels
        )
        validation_x, validation_z, validation_time, validation_diagnostics = simulate_split(
            spec, observation["validation_trajectories"], validation_seed, truth_kernels
        )
        if not all(train_diagnostics["finite"]) or not all(validation_diagnostics["finite"]):
            raise RuntimeError("true training or validation trajectory diverged")
        sigma_obs = float(observation["observation_noise"])
        training_rng = np.random.default_rng(train_seed + 1)
        validation_rng = np.random.default_rng(validation_seed + 1)
        observed_train_x = train_x + sigma_obs * training_rng.normal(size=train_x.shape)
        observed_train_z = train_z + sigma_obs * training_rng.normal(size=train_z.shape)
        observed_validation_x = validation_x + sigma_obs * validation_rng.normal(size=validation_x.shape)
        observed_validation_z = validation_z + sigma_obs * validation_rng.normal(size=validation_z.shape)
        training = assemble(
            spec, observed_train_x, observed_train_z, time, prey_basis, predator_basis
        )
        validation = assemble(
            spec,
            observed_validation_x,
            observed_validation_z,
            validation_time,
            prey_basis,
            predator_basis,
        )

        theta_prey, prey_selection = selected_fit(
            training["prey"], validation["prey"], spec["estimator"]
        )
        theta_zx, predator_selection = selected_fit(
            training["predator"], validation["predator"], spec["estimator"]
        )
        channel_count = truth[0].size
        theta_xx = theta_prey[:channel_count]
        theta_xz = theta_prey[channel_count:]
        estimated_kernels = channel_kernels(
            theta_xx, theta_xz, theta_zx, prey_basis, predator_basis
        )

        ols_prey, *_ = np.linalg.lstsq(
            training["prey"]["A_flat"], training["prey"]["b_flat"], rcond=None
        )
        ols_zx, *_ = np.linalg.lstsq(
            training["predator"]["A_flat"], training["predator"]["b_flat"], rcond=None
        )
        ols_xx = ols_prey[:channel_count]
        ols_xz = ols_prey[channel_count:]
        ols_kernels = channel_kernels(
            ols_xx, ols_xz, ols_zx, prey_basis, predator_basis
        )

        test_x, test_z, _, test_diagnostics = simulate_split(
            spec, observation["test_trajectories"], test_seed, truth_kernels
        )
        if not all(test_diagnostics["finite"]):
            raise RuntimeError("true test trajectory diverged")
        lasso_drift = drift_error(
            test_x, test_z, truth_kernels, estimated_kernels,
            spec["evaluation"]["drift_time_stride"],
        )
        ols_drift = drift_error(
            test_x, test_z, truth_kernels, ols_kernels,
            spec["evaluation"]["drift_time_stride"],
        )

    radius = np.linspace(
        spec["evaluation"]["profile_radius_min"],
        spec["evaluation"]["profile_radius_max"],
        spec["evaluation"]["profile_grid_size"],
    )
    true_profiles = profile_curves(radius, *truth, prey_basis, predator_basis)
    lasso_profiles = profile_curves(
        radius, theta_xx, theta_xz, theta_zx, prey_basis, predator_basis
    )
    ols_profiles = profile_curves(radius, ols_xx, ols_xz, ols_zx, prey_basis, predator_basis)
    prey_truth = np.concatenate([truth[0], truth[1]])

    record = {
        "index": index,
        "training_seed": train_seed,
        "validation_seed": validation_seed,
        "test_seed": test_seed,
        "theta_xx": theta_xx.tolist(),
        "theta_xz": theta_xz.tolist(),
        "theta_zx": theta_zx.tolist(),
        "ols_theta_xx": ols_xx.tolist(),
        "ols_theta_xz": ols_xz.tolist(),
        "ols_theta_zx": ols_zx.tolist(),
        "lasso_prey_coefficient_error": relative_error(theta_prey, prey_truth),
        "lasso_predator_coefficient_error": relative_error(theta_zx, truth[2]),
        "ols_prey_coefficient_error": relative_error(ols_prey, prey_truth),
        "ols_predator_coefficient_error": relative_error(ols_zx, truth[2]),
        "lasso_drift_error": lasso_drift["species_balanced_relative"],
        "lasso_prey_drift_error": lasso_drift["prey_relative"],
        "lasso_predator_drift_error": lasso_drift["predator_relative"],
        "ols_drift_error": ols_drift["species_balanced_relative"],
        "ols_prey_drift_error": ols_drift["prey_relative"],
        "ols_predator_drift_error": ols_drift["predator_relative"],
        "lasso_profile_errors": [
            relative_error(estimate, target)
            for estimate, target in zip(lasso_profiles, true_profiles)
        ],
        "ols_profile_errors": [
            relative_error(estimate, target)
            for estimate, target in zip(ols_profiles, true_profiles)
        ],
        "prey_design_rank": int(np.linalg.matrix_rank(training["prey"]["A_flat"])),
        "prey_design_columns": int(training["prey"]["A_flat"].shape[1]),
        "predator_design_rank": int(np.linalg.matrix_rank(training["predator"]["A_flat"])),
        "predator_design_columns": int(training["predator"]["A_flat"].shape[1]),
        "prey_selected_alpha": float(prey_selection["selected"]["alpha_sk"]),
        "predator_selected_alpha": float(predator_selection["selected"]["alpha_sk"]),
        "prey_selected_at_boundary": bool(prey_selection["selected_at_grid_boundary"]),
        "predator_selected_at_boundary": bool(predator_selection["selected_at_grid_boundary"]),
        "prey_nonconverged_candidates": int(sum(
            not row["converged"] for row in prey_selection["path"]
        )),
        "predator_nonconverged_candidates": int(sum(
            not row["converged"] for row in predator_selection["path"]
        )),
    }
    paths = {"prey": prey_selection, "predator": predator_selection}
    return record, paths


def finite_summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "q25": float(np.quantile(finite, 0.25)) if finite.size else None,
        "q75": float(np.quantile(finite, 0.75)) if finite.size else None,
    }


def summarize(records):
    scalar_names = (
        "lasso_prey_coefficient_error", "lasso_predator_coefficient_error",
        "ols_prey_coefficient_error", "ols_predator_coefficient_error",
        "lasso_drift_error", "lasso_prey_drift_error", "lasso_predator_drift_error",
        "ols_drift_error", "ols_prey_drift_error", "ols_predator_drift_error",
    )
    summary = {name: finite_summary([record[name] for record in records]) for name in scalar_names}
    channel_names = ("prey_prey", "prey_predator", "predator_prey")
    for method in ("lasso", "ols"):
        for channel_index, channel_name in enumerate(channel_names):
            summary[f"{method}_{channel_name}_profile_error"] = finite_summary([
                record[f"{method}_profile_errors"][channel_index] for record in records
            ])
    summary["diagnostics"] = {
        "prey_alpha_boundary_selections": int(sum(r["prey_selected_at_boundary"] for r in records)),
        "predator_alpha_boundary_selections": int(sum(r["predator_selected_at_boundary"] for r in records)),
        "prey_nonconverged_candidates": int(sum(r["prey_nonconverged_candidates"] for r in records)),
        "predator_nonconverged_candidates": int(sum(r["predator_nonconverged_candidates"] for r in records)),
        "prey_rank_deficient_fits": int(sum(r["prey_design_rank"] < r["prey_design_columns"] for r in records)),
        "predator_rank_deficient_fits": int(sum(r["predator_design_rank"] < r["predator_design_columns"] for r in records)),
    }
    return summary


def flatten_record(record):
    row = {key: value for key, value in record.items() if not isinstance(value, list)}
    for key in ("theta_xx", "theta_xz", "theta_zx", "ols_theta_xx", "ols_theta_xz", "ols_theta_zx"):
        for index, value in enumerate(record[key]):
            row[f"{key}_{index + 1}"] = value
    for method in ("lasso", "ols"):
        for index, value in enumerate(record[f"{method}_profile_errors"]):
            row[f"{method}_profile_error_{index + 1}"] = value
    return row


def write_records_csv(path, records):
    rows = [flatten_record(record) for record in records]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_display(spec, first_record, out_dir):
    prey_basis, predator_basis, _, truth, truth_kernels = model_objects(spec)
    fitted_kernels = channel_kernels(
        np.asarray(first_record["theta_xx"]),
        np.asarray(first_record["theta_xz"]),
        np.asarray(first_record["theta_zx"]),
        prey_basis,
        predator_basis,
    )
    evaluation = spec["evaluation"]
    seed = spec["seeds"]["training_start"] + spec["seeds"]["display_offset"]
    true_x, true_z, time, diagnostics = simulate_split(
        spec,
        1,
        seed,
        truth_kernels,
        particle_count=evaluation["display_particles"],
        horizon=evaluation["display_horizon"],
    )
    if not all(diagnostics["finite"]):
        raise RuntimeError("true display trajectory diverged")
    fitted_x, fitted_z, _, fitted_diagnostics = simulate_split(
        spec,
        1,
        seed + 1,
        fitted_kernels,
        particle_count=evaluation["display_particles"],
        horizon=evaluation["display_horizon"],
        initial_prey=true_x[:, 0],
        initial_predator=true_z[:, 0],
    )
    np.savez_compressed(
        out_dir / "display_trajectory.npz",
        time=time,
        true_prey=true_x,
        true_predator=true_z,
        fitted_prey=fitted_x,
        fitted_predator=fitted_z,
    )
    return {
        "fitted_finite": bool(all(fitted_diagnostics["finite"])),
        "fitted_failed_steps": fitted_diagnostics["failed_steps"],
        "trajectory_error": (
            float(np.sqrt(
                (np.sum((fitted_x - true_x) ** 2) + np.sum((fitted_z - true_z) ** 2))
                / (np.sum(true_x ** 2) + np.sum(true_z ** 2))
            )) if np.all(np.isfinite(fitted_x)) and np.all(np.isfinite(fitted_z)) else None
        ),
    }


def create_preview_figures(spec, records, out_dir):
    prey_basis, predator_basis, _, truth, _ = model_objects(spec)
    evaluation = spec["evaluation"]
    radius = np.linspace(
        evaluation["profile_radius_min"], evaluation["profile_radius_max"],
        evaluation["profile_grid_size"],
    )
    true_profiles = profile_curves(radius, *truth, prey_basis, predator_basis)
    channel_titles = ("prey--prey", "predator on prey", "prey on predator")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), constrained_layout=True)
    for channel_index, ax in enumerate(axes):
        curves = []
        for record in records:
            curves.append(profile_curves(
                radius,
                np.asarray(record["theta_xx"]),
                np.asarray(record["theta_xz"]),
                np.asarray(record["theta_zx"]),
                prey_basis,
                predator_basis,
            )[channel_index])
        curves = np.asarray(curves)
        for curve in curves:
            ax.plot(radius, curve, color="0.72", linewidth=0.7, alpha=0.55)
        ax.plot(radius, true_profiles[channel_index], color="black", linewidth=1.7, label="true")
        ax.plot(radius, np.median(curves, axis=0), color="tab:red", linewidth=1.5, label="median LASSO")
        ax.axhline(0.0, color="0.3", linewidth=0.6)
        ax.set_title(f"({chr(97 + channel_index)}) {channel_titles[channel_index]}")
        ax.set_xlabel("distance r")
        if channel_index == 0:
            ax.set_ylabel("directed force at displacement (r,0)")
            ax.legend(frameon=False)
    fig.savefig(out_dir / "predator_prey_profiles_preview.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "predator_prey_profiles_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    display = np.load(out_dir / "display_trajectory.npz")
    display_times = evaluation["display_times"]
    indices = [int(np.argmin(np.abs(display["time"] - value))) for value in display_times]
    fig, axes = plt.subplots(2, len(indices), figsize=(10.5, 5.1), constrained_layout=True)
    for column, (time_value, time_index) in enumerate(zip(display_times, indices)):
        for row, prefix in enumerate(("true", "fitted")):
            ax = axes[row, column]
            x = display[f"{prefix}_prey"][0, time_index]
            z = display[f"{prefix}_predator"][0, time_index, 0]
            ax.scatter(x[:, 0], x[:, 1], s=5, color="tab:blue", alpha=0.7, linewidths=0)
            ax.scatter(z[0], z[1], s=42, color="tab:red", marker="x", linewidths=1.6)
            ax.set_aspect("equal")
            ax.set_xlim(-1.65, 1.65)
            ax.set_ylim(-1.65, 1.65)
            ax.set_title(f"t={time_value:g}")
            if column == 0:
                ax.set_ylabel("true" if row == 0 else "learned")
            if row == 1:
                ax.set_xlabel("x1")
    fig.savefig(out_dir / "predator_prey_dynamics_preview.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "predator_prey_dynamics_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    requested = read_json(args.config)
    validate_protocol(requested)
    spec = effective_protocol(requested, args.smoke)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output is None:
        kind = "smoke" if args.smoke else "application"
        out_dir = ROOT / "outputs" / "validation" / "predator_prey" / f"{kind}-{timestamp}"
    else:
        out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "source_snapshot").mkdir()
    (out_dir / "lasso_paths").mkdir()

    source_hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        source_hashes[relative] = digest(source)
        target = out_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write_json(out_dir / "requested_protocol.json", requested)
    write_json(out_dir / "effective_protocol.json", spec)
    manifest = {
        "status": "running",
        "smoke_only": bool(args.smoke),
        "created_utc": timestamp,
        "source_sha256": source_hashes,
        "requested_protocol_sha256": digest(out_dir / "requested_protocol.json"),
        "effective_protocol_sha256": digest(out_dir / "effective_protocol.json"),
    }
    write_json(out_dir / "manifest.json", manifest)

    repetitions = spec["evaluation"]["scientific_repetitions"]
    payloads = [(index, spec) for index in range(repetitions)]
    if args.workers == 1:
        outputs = [run_repetition(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            outputs = list(executor.map(run_repetition, payloads))
    records = [item[0] for item in outputs]
    for record, paths in outputs:
        write_json(out_dir / "lasso_paths" / f"repetition-{record['index']:03d}.json", paths)
    write_json(out_dir / "records.json", records)
    write_records_csv(out_dir / "records.csv", records)
    summary = summarize(records)
    summary["display"] = create_display(spec, records[0], out_dir)
    write_json(out_dir / "summary.json", summary)
    create_preview_figures(spec, records, out_dir)

    artifact_hashes = {}
    for name in (
        "records.json", "records.csv", "summary.json", "display_trajectory.npz",
        "predator_prey_profiles_preview.pdf", "predator_prey_dynamics_preview.pdf",
    ):
        artifact_hashes[name] = digest(out_dir / name)
    manifest.update({
        "status": "smoke_complete" if args.smoke else "scientific_complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "repetitions": repetitions,
        "artifact_sha256": artifact_hashes,
    })
    write_json(out_dir / "manifest.json", manifest)
    print(out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
