"""Reproduce the R05 one-dimensional overcomplete-dictionary experiment.

The trajectory, spatial and temporal tests, and full-pair weak system are held
fixed while comparing inactive exponential scales. The raw-LASSO penalty is
selected only by relative radial-profile error; support size is recorded after tuning
but is never part of the tuning criterion.
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import make_kernel_evaluator, make_radial_basis
from pipeline import fit_kernel_from_particles
from scripts.run_sindy_comparison import (
    HORIZON,
    OBSERVATION_STEP,
    PARTICLES,
    PROCESS_NOISE,
    SPACE_TEST_WIDTH,
    TIME_TEST_SUPPORT,
    DETERMINISTIC_SPACE_CENTERS,
    reference_trajectory,
)
from scripts.run_weak_increment_validation import experiment_config
from simulate import simulate_with_kernel

EPSILON = 1e-8
DISPLAY_PARTICLE_COUNT = 30
TRUE_ACTIVE_COEFFICIENTS = np.array([1.0, -1.0])
DEFAULT_DICTIONARIES = {
    "quarter_octaves": (0.25, 0.5, 1.0, 2.0),
    "shifted_short": (0.2, 0.4, 0.8, 1.6),
    "shifted_long": (0.3, 0.6, 1.2, 2.4),
}


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile_quantities(theta, basis_params):
    radial_basis, _ = make_radial_basis(basis_params)
    radius = np.linspace(0.02, 3.0, 800)
    true_profile = 1.0 / (radius + EPSILON) - 1.0
    learned_profile = radial_basis(radius) @ theta
    error = float(
        np.linalg.norm(learned_profile - true_profile)
        / np.linalg.norm(true_profile)
    )
    return radius, true_profile, learned_profile, error


def make_overcomplete_config(seed, kappas):
    steps = int(round(HORIZON / OBSERVATION_STEP))
    config = experiment_config(
        PARTICLES,
        steps,
        OBSERVATION_STEP,
        PROCESS_NOISE,
        seed,
        DETERMINISTIC_SPACE_CENTERS,
        SPACE_TEST_WIDTH,
        TIME_TEST_SUPPORT,
        "bump",
    )
    config["kernel_basis"] = {
        "basis": "qanr_exp",
        "M": 2 + len(kappas),
        "ell_grid": list(kappas),
        "eps": EPSILON,
    }
    config["regression"].update(
        method="ridge",
        standardize=False,
        alpha=0.0,
    )
    return config


def fit_penalty_path(A, response, basis_params, alpha_count):
    sample_count = A.shape[0]
    alpha_max = float(np.max(np.abs(A.T @ response)) / sample_count)
    ratios = np.logspace(-0.1, -7.0, alpha_count)
    alphas = alpha_max * ratios
    model = Lasso(
        alpha=float(alphas[0]),
        fit_intercept=False,
        max_iter=5_000_000,
        tol=1e-10,
        selection="cyclic",
        warm_start=True,
    )
    rows = []
    for alpha, ratio in zip(alphas, ratios):
        model.alpha = float(alpha)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(A, response)
        theta = np.asarray(model.coef_, dtype=float).copy()
        _, _, _, error = profile_quantities(theta, basis_params)
        rows.append(
            {
                "alpha": float(alpha),
                "alpha_over_max": float(ratio),
                "profile_error": error,
                "nnz_1e8": int(np.sum(np.abs(theta) > 1e-8)),
                "convergence_warning": int(
                    any(issubclass(item.category, ConvergenceWarning) for item in caught)
                ),
                "iterations": int(model.n_iter_),
                "dual_gap": float(model.dual_gap_),
                **{
                    f"theta_{index + 1}": float(value)
                    for index, value in enumerate(theta)
                },
            }
        )
    converged = [row for row in rows if row["convergence_warning"] == 0]
    if not converged:
        raise RuntimeError("no converged LASSO fit in the penalty grid")
    return rows, min(converged, key=lambda row: row["profile_error"]), alpha_max


def plot_candidate(
    output,
    time,
    X,
    learned_trajectory,
    X0,
    radius,
    true_profile,
    learned_profile,
    labels,
    theta,
):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].plot(radius, true_profile, label="true", linewidth=2.0)
    axes[0].plot(radius, learned_profile, "--", label="learned", linewidth=2.0)
    axes[0].set(xlabel="distance r", ylabel="radial profile", title="Profile recovery")
    axes[0].legend()

    true_theta = np.r_[TRUE_ACTIVE_COEFFICIENTS, np.zeros(theta.size - 2)]
    positions = np.arange(theta.size)
    axes[1].bar(positions - 0.2, true_theta, width=0.4, label="true")
    axes[1].bar(positions + 0.2, theta, width=0.4, label="raw LASSO")
    axes[1].set_xticks(positions, labels, rotation=38, ha="right")
    axes[1].set(ylabel="coefficient", title="Dictionary coefficients")
    axes[1].legend()

    initial_order = np.argsort(X0[0, :, 0])
    particle_indices = initial_order[
        np.linspace(
            0,
            initial_order.size - 1,
            DISPLAY_PARTICLE_COUNT,
            dtype=int,
        )
    ]
    for plot_index, particle_index in enumerate(particle_indices):
        axes[2].plot(
            time,
            X[0, :, particle_index, 0],
            color="tab:blue",
            linewidth=0.45,
            alpha=0.40,
            label="true" if plot_index == 0 else None,
        )
        axes[2].plot(
            time,
            learned_trajectory[0, :, particle_index, 0],
            "--",
            color="tab:red",
            linewidth=0.45,
            alpha=0.52,
            label="learned" if plot_index == 0 else None,
        )
    axes[2].set(xlabel="time", ylabel="particle position", title="Coupled dynamics")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def evaluate_dictionary(
    name,
    kappas,
    X,
    time,
    X0,
    driving_noise,
    seed,
    alpha_count,
    out_dir,
):
    config = make_overcomplete_config(seed, kappas)
    started = perf_counter()
    system_fit = fit_kernel_from_particles(
        config,
        X,
        time,
        return_system=True,
        pair_mode="full",
    )
    assembly_seconds = perf_counter() - started
    A = system_fit["system"]["A_flat"]
    response = system_fit["system"]["b_flat"]
    path, best, alpha_max = fit_penalty_path(
        A, response, config["kernel_basis"], alpha_count
    )
    theta = np.array(
        [best[f"theta_{index + 1}"] for index in range(2 + len(kappas))]
    )
    true_theta = np.r_[TRUE_ACTIVE_COEFFICIENTS, np.zeros(len(kappas))]
    kernel = make_kernel_evaluator(theta, system_fit["system"]["kernel_basis"])
    learned_trajectory = simulate_with_kernel(
        kernel,
        X0,
        time,
        sigma=PROCESS_NOISE,
        noise=driving_noise,
    )
    difference = learned_trajectory - X
    dynamics = {
        "trajectory_error_rel": float(
            np.linalg.norm(difference) / np.linalg.norm(X)
        ),
        "trajectory_error_final": float(
            np.linalg.norm(difference[:, -1]) / np.linalg.norm(X[:, -1])
        ),
    }
    radius, true_profile, learned_profile, profile_error = profile_quantities(
        theta, config["kernel_basis"]
    )
    labels = system_fit["diagnostics"]["kernel_meta"]["labels"]
    plot_candidate(
        out_dir / f"{name}.png",
        time,
        X,
        learned_trajectory,
        X0,
        radius,
        true_profile,
        learned_profile,
        labels,
        theta,
    )
    summary = {
        "dictionary": name,
        "kappas": list(kappas),
        "alpha_sk": best["alpha"],
        "alpha_max": alpha_max,
        "alpha_over_max": best["alpha_over_max"],
        "profile_error": profile_error,
        "trajectory_error_rel": dynamics["trajectory_error_rel"],
        "trajectory_error_final": dynamics["trajectory_error_final"],
        "coefficient_error": float(np.linalg.norm(theta - true_theta)),
        "nnz_1e8": int(np.sum(np.abs(theta) > 1e-8)),
        "selected_indices_1based": [
            index + 1 for index, value in enumerate(theta) if abs(value) > 1e-8
        ],
        "max_inactive_abs": float(np.max(np.abs(theta[2:]))),
        "assembly_seconds": assembly_seconds,
        "iterations": best["iterations"],
        "dual_gap": best["dual_gap"],
        "labels": labels,
        "coefficients": theta.tolist(),
    }
    for row in path:
        row["dictionary"] = name
        row["kappas"] = ";".join(f"{value:g}" for value in kappas)
    profile_rows = [
        {
            "dictionary": name,
            "radius": float(r),
            "true_profile": float(true),
            "learned_profile": float(learned),
        }
        for r, true, learned in zip(radius, true_profile, learned_profile)
    ]
    initial_order = np.argsort(X0[0, :, 0])
    ranks = np.linspace(
        0,
        initial_order.size - 1,
        DISPLAY_PARTICLE_COUNT,
        dtype=int,
    )
    particle_indices = initial_order[ranks]
    trajectory_rows = []
    for rank, particle_index in zip(ranks, particle_indices):
        for time_index, time_value in enumerate(time):
            trajectory_rows.append(
                {
                    "dictionary": name,
                    "particle_index": int(particle_index),
                    "initial_rank_index": int(rank),
                    "time": float(time_value),
                    "reference_position": float(
                        X[0, time_index, particle_index, 0]
                    ),
                    "learned_position": float(
                        learned_trajectory[0, time_index, particle_index, 0]
                    ),
                }
            )
    return summary, path, profile_rows, trajectory_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/validation/one_dimensional_sparse_recovery",
    )
    parser.add_argument("--seed", type=int, default=40_000)
    parser.add_argument("--alpha-count", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    X, time, X0, driving_noise, _ = reference_trajectory(args.seed)
    summaries = []
    path_rows = []
    profile_rows = []
    trajectory_rows = []
    for name, kappas in DEFAULT_DICTIONARIES.items():
        summary, path, profiles, trajectories = evaluate_dictionary(
            name,
            kappas,
            X,
            time,
            X0,
            driving_noise,
            args.seed,
            args.alpha_count,
            out_dir,
        )
        summaries.append(summary)
        path_rows.extend(path)
        profile_rows.extend(profiles)
        trajectory_rows.extend(trajectories)
        print(json.dumps(summary, indent=2), flush=True)
    write_rows(out_dir / "penalty_sweep.csv", path_rows)
    write_rows(out_dir / "profile_curves.csv", profile_rows)
    write_rows(out_dir / "trajectory_curves.csv", trajectory_rows)
    (out_dir / "candidate_summary.json").write_text(json.dumps(summaries, indent=2))
    metadata = {
        "purpose": "R05 one-dimensional overcomplete-dictionary recovery experiment",
        "trajectory_seed": args.seed,
        "N": PARTICLES,
        "horizon": HORIZON,
        "dt": OBSERVATION_STEP,
        "sigma": PROCESS_NOISE,
        "sigma_obs": 0.0,
        "space_tests": {
            "type": "gaussian",
            "centers": DETERMINISTIC_SPACE_CENTERS.tolist(),
            "width": SPACE_TEST_WIDTH,
        },
        "time_tests": {"type": "bump", "support": TIME_TEST_SUPPORT},
        "displayed_trajectory_particles": {
            "count": DISPLAY_PARTICLE_COUNT,
            "selection": "equally spaced ranks of the initial positions",
        },
        "regression": {
            "method": "raw LASSO",
            "selection_objective": "minimum relative radial-profile error",
            "support_constraint": None,
            "manual_coefficient_thresholding": False,
            "alpha_count": args.alpha_count,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
