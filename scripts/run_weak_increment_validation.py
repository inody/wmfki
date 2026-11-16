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

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

from features import compute_increment_regression_system
from kernels import make_radial_basis
from pipeline import fit_kernel_from_particles
from scripts.r05_numerical_settings import (
    SPACE_TEST_CENTERS,
    SPACE_TEST_WIDTH,
)
from scripts.run_messenger_qanr_sweep import qanr_config
from simulate import simulate_particles


DT_VALUES = [0.005, 0.01, 0.02, 0.05]
SIGMA_VALUES = [0.0, 0.05, 0.1, 0.2, 0.4]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def experiment_config(
    N,
    steps,
    dt,
    sigma,
    seed,
    centers,
    width,
    time_support,
    time_test_type="bump",
):
    config = qanr_config(
        N=N,
        R=1,
        sigma=sigma,
        alpha=0.0,
        powers=[1, 2],
        fit_pairs=N * (N - 1),
        seed=seed,
        dt=dt,
        T=steps,
        reg_method="ridge",
        pair_replacement=False,
        solver="euler",
        time_s_factor=1.8,
    )
    config["regression"].update(standardize=False, alpha=0.0)
    config["space_tests"].update(
        centers=np.asarray(centers).tolist(),
        s=float(width),
        L=len(centers),
    )
    config["time_tests"].update(
        support=float(time_support),
        type=time_test_type,
    )
    config["weak_form"].update(normalize_rows=True, full_pair_chunk_size=100_000)
    config["diagnostics"]["method"] = None
    return config


def fixed_test_parameters(X, L=25, K=8, total_time=2.0):
    del X
    if L != len(SPACE_TEST_CENTERS):
        raise ValueError("the R05 benchmark uses exactly 25 spatial tests")
    time_support = 1.8 * total_time / (K + 1)
    return SPACE_TEST_CENTERS.copy(), SPACE_TEST_WIDTH, time_support


def profile_error(theta, basis_params):
    radial_basis, _ = make_radial_basis(basis_params)
    r = np.linspace(0.02, 3.0, 800)
    force_basis = r[:, None] * radial_basis(r)
    theta_true = np.array([1.0, -1.0])
    true = force_basis @ theta_true
    estimated = force_basis @ theta
    return float(np.linalg.norm(estimated - true) / np.linalg.norm(true))


def simulate_fine(N, total_time, fine_dt, sigma, seed):
    steps = round(total_time / fine_dt)
    X, t, *_ = simulate_particles(
        N=N,
        T=steps,
        dt=fine_dt,
        R=1,
        sigma=sigma,
        d=1,
        init="gaussian_mixture",
        init_centers=[0.5, 2.0, 3.5],
        init_std=0.005,
        kernel_params={"mode": "qanr", "eps": 1e-8, "r_max": 3.0},
        seed=seed,
        solver="euler",
    )
    return X, t


def run_experiment(out_dir, seeds):
    N, total_time, fine_dt = 100, 2.0, min(DT_VALUES)
    theta_true = np.array([1.0, -1.0])
    rows = []
    started = perf_counter()

    for sigma in SIGMA_VALUES:
        for seed_index in range(seeds):
            trajectory_seed = 30_000 + 100 * int(round(100 * sigma)) + seed_index
            X_fine, t_fine = simulate_fine(
                N, total_time, fine_dt, sigma, trajectory_seed
            )
            centers, width, time_support = fixed_test_parameters(X_fine)

            for dt in DT_VALUES:
                stride = round(dt / fine_dt)
                X = X_fine[:, ::stride]
                t = t_fine[::stride]
                steps = t.size - 1
                config = experiment_config(
                    N,
                    steps,
                    dt,
                    sigma,
                    trajectory_seed,
                    centers,
                    width,
                    time_support,
                )

                weak = fit_kernel_from_particles(
                    config, X, t, return_system=True, pair_mode="full"
                )
                theta_weak = np.asarray(weak["theta_hat"])
                basis = weak["system"]["kernel_basis"]

                t0 = perf_counter()
                A_increment, b_increment = compute_increment_regression_system(
                    X,
                    t,
                    basis,
                    pair_chunk_size=config["weak_form"]["full_pair_chunk_size"],
                )
                theta_increment, *_ = np.linalg.lstsq(
                    A_increment, b_increment, rcond=None
                )
                increment_seconds = perf_counter() - t0

                for method, theta, design_seconds in (
                    (
                        "weak_form",
                        theta_weak,
                        weak["diagnostics"]["timing"]["design_seconds"],
                    ),
                    ("euler_increment", theta_increment, increment_seconds),
                ):
                    error = theta - theta_true
                    rows.append(
                        {
                            "method": method,
                            "sigma": sigma,
                            "dt": dt,
                            "seed": seed_index,
                            "trajectory_seed": trajectory_seed,
                            "theta_1": float(theta[0]),
                            "theta_2": float(theta[1]),
                            "coefficient_error_sq": float(error @ error),
                            "kernel_profile_error": profile_error(
                                theta, config["kernel_basis"]
                            ),
                            "fit_seconds": design_seconds,
                        }
                    )

    write_rows(Path(out_dir) / "weak_vs_increment.csv", rows)
    write_rows(Path(out_dir) / "weak_vs_increment_summary.csv", summarize(rows))
    metadata = {
        "seeds": seeds,
        "N": N,
        "total_time": total_time,
        "fine_dt": fine_dt,
        "trajectory_policy": (
            "one fine trajectory per sigma and seed, subsampled for all dt values"
        ),
        "test_family": {
            "selection": "deterministic and fixed before all trajectories",
            "space_centers": SPACE_TEST_CENTERS.tolist(),
            "space_width": SPACE_TEST_WIDTH,
            "time_type": "bump",
            "time_support": 1.8 * total_time / 9.0,
        },
        "elapsed_seconds": perf_counter() - started,
    }
    (Path(out_dir) / "metadata.json").write_text(json.dumps(metadata, indent=2))
    plot_results(rows, Path(out_dir) / "weak_vs_increment.png")
    return rows


def summarize(rows):
    summary = []
    for method in ("weak_form", "euler_increment"):
        for sigma in SIGMA_VALUES:
            for dt in DT_VALUES:
                subset = [
                    row
                    for row in rows
                    if row["method"] == method
                    and float(row["sigma"]) == sigma
                    and float(row["dt"]) == dt
                ]
                coefficient_errors = np.sqrt(
                    [float(row["coefficient_error_sq"]) for row in subset]
                )
                profile_errors = np.asarray(
                    [float(row["kernel_profile_error"]) for row in subset]
                )
                fit_times = np.asarray([float(row["fit_seconds"]) for row in subset])
                summary.append(
                    {
                        "method": method,
                        "sigma": sigma,
                        "dt": dt,
                        "seeds": len(subset),
                        "coefficient_rmse": float(
                            np.sqrt(
                                np.mean(
                                    [
                                        float(row["coefficient_error_sq"])
                                        for row in subset
                                    ]
                                )
                            )
                        ),
                        "coefficient_error_median": float(
                            np.median(coefficient_errors)
                        ),
                        "coefficient_error_q25": float(
                            np.quantile(coefficient_errors, 0.25)
                        ),
                        "coefficient_error_q75": float(
                            np.quantile(coefficient_errors, 0.75)
                        ),
                        "profile_error_median": float(np.median(profile_errors)),
                        "profile_error_q25": float(np.quantile(profile_errors, 0.25)),
                        "profile_error_q75": float(np.quantile(profile_errors, 0.75)),
                        "fit_seconds_median": float(np.median(fit_times)),
                    }
                )
    return summary


def aggregate(rows, filter_fn, x_key, y_key):
    selected = [row for row in rows if filter_fn(row)]
    xs = sorted({float(row[x_key]) for row in selected})
    means, lo, hi = [], [], []
    for x in xs:
        values = np.asarray(
            [float(row[y_key]) for row in selected if float(row[x_key]) == x]
        )
        values = np.sqrt(values) if y_key == "coefficient_error_sq" else values
        means.append(float(np.mean(values)))
        lo.append(float(np.quantile(values, 0.25)))
        hi.append(float(np.quantile(values, 0.75)))
    return np.asarray(xs), np.asarray(means), np.asarray(lo), np.asarray(hi)


def plot_curve(ax, rows, filter_fn, x_key, label, color, marker):
    x, mean, lo, hi = aggregate(
        rows, filter_fn, x_key=x_key, y_key="coefficient_error_sq"
    )
    ax.plot(x, mean, marker=marker, color=color, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15)


def plot_results(rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"weak_form": "tab:blue", "euler_increment": "tab:orange"}
    markers = {"weak_form": "o", "euler_increment": "s"}

    for method in colors:
        plot_curve(
            axes[0],
            rows,
            lambda row, method=method: row["method"] == method
            and float(row["sigma"]) == 0.1,
            "dt",
            method.replace("_", " "),
            colors[method],
            markers[method],
        )
        plot_curve(
            axes[1],
            rows,
            lambda row, method=method: row["method"] == method
            and float(row["dt"]) == min(DT_VALUES)
            and float(row["sigma"]) > 0.0,
            "sigma",
            method.replace("_", " "),
            colors[method],
            markers[method],
        )

    axes[0].set(xscale="log", yscale="log", xlabel=r"observation step $\Delta t$", ylabel="coefficient RMSE")
    axes[1].set(xscale="log", yscale="log", xlabel=r"process noise $\sigma$", ylabel="coefficient RMSE")
    axes[0].legend()
    axes[1].legend()

    for method in colors:
        selected = [
            row
            for row in rows
            if row["method"] == method and float(row["sigma"]) == 0.1
        ]
        x, mean, lo, hi = aggregate(
            selected, lambda row: True, "dt", "kernel_profile_error"
        )
        axes[2].plot(
            x,
            mean,
            marker=markers[method],
            color=colors[method],
            label=method.replace("_", " "),
        )
        axes[2].fill_between(x, lo, hi, color=colors[method], alpha=0.15)
    axes[2].set(xscale="log", yscale="log", xlabel=r"observation step $\Delta t$", ylabel="relative kernel-profile error")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare weak-form and Euler-increment kernel regression."
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--out-dir", default="outputs/validation/weak_vs_increment"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate summaries and plots from the existing CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = read_rows(Path(args.out_dir) / "weak_vs_increment.csv")
        write_rows(
            Path(args.out_dir) / "weak_vs_increment_summary.csv", summarize(rows)
        )
        plot_results(rows, Path(args.out_dir) / "weak_vs_increment.png")
    else:
        run_experiment(args.out_dir, args.seeds)


if __name__ == "__main__":
    main()
