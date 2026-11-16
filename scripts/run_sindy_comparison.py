"""Compare weak-form and forward-difference regression under observation noise.

Both estimators see the same trajectory, the same observation-noise realization,
the same matched dictionary, and the same unpenalized least-squares solver.
Both responses and designs are built from the noisy positions ``Y``; neither
method receives noiseless design features, and both use all observed ordered
pairs rather than sampled pairs.  The smooth compactly supported test family is
deterministic and fixed before all trajectories, seeds, and observation-noise
levels.  The main
difference is how the time derivative enters the regression: the weak form
moves it onto a smooth temporal test function by integration by parts, while the
forward-difference baseline forms the forward Euler difference
``(Y(t_{n+1}) - Y(t_n)) / dt`` of the observed positions.

The R05 manuscript theory bounds the observation-induced response and design
perturbations.  This experiment illustrates the predicted absence of a
``dt**-2`` input-perturbation factor for the weak form, rather than directly
verifying a coefficient bound.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import compute_increment_regression_system
from kernels import make_kernel_evaluator
from pipeline import fit_kernel_from_particles
from scripts.run_weak_increment_validation import experiment_config, profile_error
from simulate import make_true_kernel, simulate_particles, simulate_with_kernel


PARTICLES = 100
HORIZON = 2.0
OBSERVATION_STEP = 0.005
PROCESS_NOISE = 0.1
TRUE_THETA = np.array([1.0, -1.0])
KERNEL_PARAMS = {"mode": "qanr", "eps": 1e-8, "r_max": 3.0}
SIGMA_OBS_VALUES = (0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
METHODS = ("weak_form", "sindy_euler")
DIVERGENCE_LIMIT = 1e3
BREAKDOWN_LEVEL = 0.1
SPACE_TEST_COUNT = 25
TIME_TEST_COUNT = 8
SPACE_TEST_MIN = 0.0
SPACE_TEST_MAX = 4.0
SPACE_TEST_WIDTH = (
    1.5 * (SPACE_TEST_MAX - SPACE_TEST_MIN) / (SPACE_TEST_COUNT - 1)
)
TIME_TEST_SUPPORT = 1.8 * HORIZON / (TIME_TEST_COUNT + 1)
DETERMINISTIC_SPACE_CENTERS = np.linspace(
    SPACE_TEST_MIN, SPACE_TEST_MAX, SPACE_TEST_COUNT
)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def initial_positions(seed, particles=PARTICLES, repetitions=1):
    X, *_ = simulate_particles(
        N=particles,
        T=0,
        dt=1.0,
        R=repetitions,
        sigma=0.0,
        d=1,
        init="gaussian_mixture",
        init_centers=[0.5, 2.0, 3.5],
        init_std=0.005,
        init_scale=1.0,
        kernel_params=KERNEL_PARAMS,
        seed=seed,
        solver="euler",
    )
    return X[:, 0]


def reference_trajectory(seed, *, particles=PARTICLES, horizon=HORIZON, step=OBSERVATION_STEP,
                         sigma=PROCESS_NOISE):
    """Generate one trajectory with explicitly held initial data and driving noise.

    The driving noise is returned so that the learned kernels can be re-simulated
    on the identical noise realization.
    """
    steps = int(round(horizon / step))
    t = np.linspace(0.0, horizon, steps + 1)
    X0 = initial_positions(seed, particles=particles)
    K_true, *_ = make_true_kernel(KERNEL_PARAMS)
    driving_noise = np.random.default_rng(seed + 1).normal(
        0.0, 1.0, size=(X0.shape[0], steps, particles, 1)
    )
    X = simulate_with_kernel(K_true, X0, t, sigma=sigma, noise=driving_noise)
    return X, t, X0, driving_noise, K_true


def minimum_pair_distance(X, stride=10):
    """Smallest distance between two distinct particles over the observed times."""
    smallest = np.inf
    for r in range(X.shape[0]):
        for n in range(0, X.shape[1], stride):
            x = X[r, n]
            gaps = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
            np.fill_diagonal(gaps, np.inf)
            smallest = min(smallest, float(gaps.min()))
    return smallest


def trajectory_errors(K_hat, X_reference, X0, t, driving_noise, sigma=PROCESS_NOISE):
    """Re-simulate with the learned kernel on the reference noise and compare."""
    X_hat = simulate_with_kernel(K_hat, X0, t, sigma=sigma, noise=driving_noise)
    diverged = (not np.all(np.isfinite(X_hat))) or float(np.max(np.abs(X_hat))) > DIVERGENCE_LIMIT
    if diverged:
        return {
            "trajectory_error_rel": float("nan"),
            "trajectory_error_rms": float("nan"),
            "trajectory_error_final": float("nan"),
            "diverged": True,
        }
    difference = X_hat - X_reference
    return {
        "trajectory_error_rel": float(
            np.linalg.norm(difference) / np.linalg.norm(X_reference)
        ),
        "trajectory_error_rms": float(np.sqrt(np.mean(difference ** 2))),
        "trajectory_error_final": float(
            np.linalg.norm(difference[:, -1]) / np.linalg.norm(X_reference[:, -1])
        ),
        "diverged": False,
    }


def sampled_drift_error(K_true, K_hat, X, seed, samples=200):
    """Relative error of the learned drift field on sampled target positions."""
    rng = np.random.default_rng(seed)
    R, T1, N, _ = X.shape
    reps = rng.integers(0, R, size=samples)
    times = rng.integers(0, T1, size=samples)
    targets = rng.integers(0, N, size=samples)
    true_norm = 0.0
    error_norm = 0.0
    for r, n, i in zip(reps, times, targets):
        z = X[r, n, i][None, :] - X[r, n]
        v_true = K_true(z).mean(axis=0)
        v_hat = K_hat(z).mean(axis=0)
        true_norm += float(v_true @ v_true)
        error_norm += float((v_hat - v_true) @ (v_hat - v_true))
    return float(np.sqrt(error_norm / true_norm)) if true_norm > 0.0 else float("nan")


def fit_weak_form(config, Y, t):
    started = perf_counter()
    result = fit_kernel_from_particles(config, Y, t, return_system=True, pair_mode="full")
    seconds = perf_counter() - started
    theta = np.asarray(result["theta_hat"], dtype=float)
    equations = int(result["system"]["A_flat"].shape[0])
    return (
        theta,
        result["system"]["kernel_basis"],
        result["diagnostics"].get("cond_ata"),
        seconds,
        equations,
    )


def fit_sindy_euler(config, Y, t, kernel_basis):
    started = perf_counter()
    A, y = compute_increment_regression_system(
        Y, t, kernel_basis, pair_chunk_size=config["weak_form"]["full_pair_chunk_size"]
    )
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    seconds = perf_counter() - started
    try:
        condition = float(np.linalg.cond(A.T @ A))
    except np.linalg.LinAlgError:
        condition = float("inf")
    return np.asarray(theta, dtype=float), condition, seconds, A.shape[0]


def run_experiment(
    out_dir,
    seeds,
    sigma_obs_values,
    step,
    sigma,
    time_test_type="bump",
):
    rows = []
    nominal_trajectory_rows = []
    nominal_particle_indices = []
    started = perf_counter()
    steps = int(round(HORIZON / step))

    for seed_index in range(seeds):
        trajectory_seed = 40_000 + 10 * seed_index
        X, t, X0, driving_noise, K_true = reference_trajectory(
            trajectory_seed, step=step, sigma=sigma
        )
        observation_noise = np.random.default_rng(trajectory_seed + 2).normal(
            0.0, 1.0, size=X.shape
        )

        for sigma_obs in sigma_obs_values:
            Y = X + sigma_obs * observation_noise
            config = experiment_config(
                PARTICLES,
                steps,
                step,
                sigma,
                trajectory_seed,
                DETERMINISTIC_SPACE_CENTERS,
                SPACE_TEST_WIDTH,
                TIME_TEST_SUPPORT,
                time_test_type,
            )

            theta_weak, kernel_basis, cond_weak, seconds_weak, rows_weak = fit_weak_form(
                config, Y, t
            )
            theta_sindy, cond_sindy, seconds_sindy, rows_sindy = fit_sindy_euler(
                config, Y, t, kernel_basis
            )
            gap = minimum_pair_distance(Y)

            if seed_index == 0 and sigma_obs == 0.0:
                nominal_kernel = make_kernel_evaluator(theta_weak, kernel_basis)
                nominal_learned = simulate_with_kernel(
                    nominal_kernel,
                    X0,
                    t,
                    sigma=sigma,
                    noise=driving_noise,
                )
                initial_order = np.argsort(X0[0, :, 0])
                ranks = np.linspace(0, initial_order.size - 1, 9, dtype=int)
                nominal_particle_indices = initial_order[ranks].tolist()
                for rank, particle_index in enumerate(nominal_particle_indices):
                    for time_index, time_value in enumerate(t):
                        nominal_trajectory_rows.append(
                            {
                                "trajectory_seed": trajectory_seed,
                                "particle_index": particle_index,
                                "initial_rank_index": int(ranks[rank]),
                                "time": float(time_value),
                                "reference_position": float(
                                    X[0, time_index, particle_index, 0]
                                ),
                                "learned_position": float(
                                    nominal_learned[0, time_index, particle_index, 0]
                                ),
                            }
                        )

            for method, theta, condition, seconds, equations in (
                ("weak_form", theta_weak, cond_weak, seconds_weak, rows_weak),
                ("sindy_euler", theta_sindy, cond_sindy, seconds_sindy, rows_sindy),
            ):
                K_hat = make_kernel_evaluator(theta, kernel_basis)
                record = {
                    "method": method,
                    "sigma_obs": sigma_obs,
                    "sigma": sigma,
                    "dt": step,
                    "seed": seed_index,
                    "trajectory_seed": trajectory_seed,
                    "theta_1": float(theta[0]),
                    "theta_2": float(theta[1]),
                    "coefficient_error": float(np.linalg.norm(theta - TRUE_THETA)),
                    "kernel_profile_error": profile_error(theta, config["kernel_basis"]),
                    "sampled_drift_error": sampled_drift_error(
                        K_true, K_hat, X, trajectory_seed + 3
                    ),
                    "equations": equations,
                    "cond_ata": condition,
                    "min_pair_distance": gap,
                    "fit_seconds": seconds,
                }
                record.update(
                    trajectory_errors(K_hat, X, X0, t, driving_noise, sigma=sigma)
                )
                rows.append(record)

            print(
                f"seed {seed_index:>3d}  sigma_obs={sigma_obs:<8g}  "
                f"weak: coef={rows[-2]['coefficient_error']:.3e} "
                f"traj={rows[-2]['trajectory_error_rel']:.3e}   "
                f"sindy: coef={rows[-1]['coefficient_error']:.3e} "
                f"traj={rows[-1]['trajectory_error_rel']:.3e}",
                flush=True,
            )

    out_dir = Path(out_dir)
    write_rows(out_dir / "sindy_vs_weak.csv", rows)
    if nominal_trajectory_rows:
        write_rows(out_dir / "nominal_trajectory.csv", nominal_trajectory_rows)
    summary = summarize(rows)
    write_rows(out_dir / "sindy_vs_weak_summary.csv", summary)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "N": PARTICLES,
                "horizon": HORIZON,
                "dt": step,
                "sigma": sigma,
                "sigma_obs_values": list(sigma_obs_values),
                "true_theta": TRUE_THETA.tolist(),
                "breakdown_levels": breakdown_levels(summary),
                "nominal_trajectory_panel": (
                    {
                        "trajectory_seed": 40_000,
                        "sigma_obs": 0.0,
                        "method": "weak_form",
                        "selection": (
                            "nine particles at equally spaced ranks of the initial "
                            "one-dimensional positions"
                        ),
                        "particle_indices": nominal_particle_indices,
                        "data_file": "nominal_trajectory.csv",
                    }
                    if nominal_trajectory_rows
                    else None
                ),
                "test_family": {
                    "selection": "deterministic and fixed before all trajectories",
                    "space_type": "gaussian",
                    "space_centers": DETERMINISTIC_SPACE_CENTERS.tolist(),
                    "space_width": SPACE_TEST_WIDTH,
                    "time_type": time_test_type,
                    "time_count": TIME_TEST_COUNT,
                    "time_support": TIME_TEST_SUPPORT,
                },
                "shared_randomness": (
                    "one trajectory per seed from explicit initial data and driving "
                    "noise; the observation-noise array is drawn once per seed and "
                    "scaled by each sigma_obs; the same deterministic test family "
                    "is used for every seed and noise level"
                ),
                "elapsed_seconds": perf_counter() - started,
            },
            indent=2,
        )
    )
    if any(float(row["sigma_obs"]) > 0.0 for row in rows):
        plot_results(rows, out_dir / "sindy_vs_weak.png")
    return rows


def _quantiles(values):
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(finite)),
        float(np.quantile(finite, 0.25)),
        float(np.quantile(finite, 0.75)),
    )


def summarize(rows):
    metrics = (
        "coefficient_error",
        "trajectory_error_rel",
        "trajectory_error_rms",
        "trajectory_error_final",
        "kernel_profile_error",
        "sampled_drift_error",
    )
    summary = []
    for method in METHODS:
        for sigma_obs in sorted({float(row["sigma_obs"]) for row in rows}):
            subset = [
                row
                for row in rows
                if row["method"] == method and float(row["sigma_obs"]) == sigma_obs
            ]
            if not subset:
                continue
            record = {
                "method": method,
                "sigma_obs": sigma_obs,
                "seeds": len(subset),
                "diverged": sum(1 for row in subset if str(row["diverged"]) in ("True", "true")),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in subset]
                median, low, high = _quantiles(values)
                record[f"{metric}_median"] = median
                record[f"{metric}_q25"] = low
                record[f"{metric}_q75"] = high
                if metric == "coefficient_error":
                    record["coefficient_error_rmse"] = float(
                        np.sqrt(np.mean(np.asarray(values) ** 2))
                    )
            record["fit_seconds_median"] = float(
                np.median([float(row["fit_seconds"]) for row in subset])
            )
            record["min_pair_distance"] = float(
                np.min([float(row["min_pair_distance"]) for row in subset])
            )
            summary.append(record)
    return summary


def breakdown_levels(summary, metrics=("coefficient_error", "trajectory_error_rel")):
    """Smallest swept sigma_obs whose median error exceeds BREAKDOWN_LEVEL."""
    levels = {}
    for method in METHODS:
        entries = sorted(
            (row for row in summary if row["method"] == method),
            key=lambda row: row["sigma_obs"],
        )
        for metric in metrics:
            crossing = None
            for row in entries:
                value = row[f"{metric}_median"]
                unstable = metric.startswith("trajectory_error") and row["diverged"] > 0
                if unstable or (np.isfinite(value) and value > BREAKDOWN_LEVEL):
                    crossing = row["sigma_obs"]
                    break
            levels[f"{method}/{metric}"] = crossing
    return levels


def _is_diverged(row):
    return str(row["diverged"]) in ("True", "true")


def _curve(rows, method, metric):
    """Median and interquartile band per noise level, with diverged levels dropped.

    A level at which any seed produced a divergent re-simulation is excluded
    rather than summarized over the surviving seeds, which would understate the
    error.  The dropped levels are returned so the caller can report them.
    """
    levels = sorted({float(row["sigma_obs"]) for row in rows})
    x, median, low, high, dropped = [], [], [], [], []
    for level in levels:
        subset = [
            row
            for row in rows
            if row["method"] == method and float(row["sigma_obs"]) == level
        ]
        if metric.startswith("trajectory_error") and any(_is_diverged(row) for row in subset):
            dropped.append(level)
            continue
        centre, q25, q75 = _quantiles([float(row[metric]) for row in subset])
        x.append(level)
        median.append(centre)
        low.append(q25)
        high.append(q75)
    return (
        np.asarray(x),
        np.asarray(median),
        np.asarray(low),
        np.asarray(high),
        dropped,
    )


def plot_results(rows, output):
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    styles = {
        "weak_form": ("tab:blue", "o", "weak form"),
        "sindy_euler": ("tab:orange", "s", "Euler difference"),
    }
    panels = (
        ("coefficient_error", r"median $\operatorname{Err}_{\theta}$"),
        ("trajectory_error_rel", r"median $\operatorname{Err}_{X}$"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (metric, label) in zip(axes, panels):
        notes = []
        for method, (color, marker, name) in styles.items():
            x, median, low, high, dropped = _curve(rows, method, metric)
            positive = x > 0.0
            ax.plot(x[positive], median[positive], marker=marker, color=color, label=name)
            ax.fill_between(x[positive], low[positive], high[positive], color=color, alpha=0.15)
            if np.any(~positive) and np.isfinite(median[~positive][0]):
                ax.axhline(
                    median[~positive][0],
                    color=color,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                )
            if dropped:
                notes.append(f"{name}: unstable for " r"$\sigma_{\rm obs}\geq$" f"{min(dropped):g}")
        ax.set(
            xscale="log",
            yscale="log",
            xlabel=r"observation noise $\sigma_{\rm obs}$",
            ylabel=label,
        )
        ax.set_title("; ".join(notes) if notes else "dotted: noise-free level", fontsize=10)
        ax.legend(loc="upper left")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare weak-form and Euler-difference kernel regression "
        "under observation noise."
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out-dir", default="outputs/validation/sindy_vs_weak")
    parser.add_argument(
        "--sigma-obs",
        default=",".join(f"{v:g}" for v in SIGMA_OBS_VALUES),
        help="comma-separated observation-noise levels",
    )
    parser.add_argument("--dt", type=float, default=OBSERVATION_STEP)
    parser.add_argument("--sigma", type=float, default=PROCESS_NOISE)
    parser.add_argument(
        "--time-test-type",
        choices=("gaussian", "bump"),
        default="bump",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate the summary and figure from the existing CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = read_rows(out_dir / "sindy_vs_weak.csv")
        write_rows(out_dir / "sindy_vs_weak_summary.csv", summarize(rows))
        if any(float(row["sigma_obs"]) > 0.0 for row in rows):
            plot_results(rows, out_dir / "sindy_vs_weak.png")
        return
    levels = [float(value) for value in args.sigma_obs.split(",") if value.strip()]
    run_experiment(
        out_dir,
        args.seeds,
        levels,
        args.dt,
        args.sigma,
        args.time_test_type,
    )


if __name__ == "__main__":
    main()
