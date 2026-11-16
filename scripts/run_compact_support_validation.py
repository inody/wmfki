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

from pipeline import fit_kernel_from_particles
from scripts.run_messenger_qanr_sweep import qanr_config
from simulate import simulate_particles


P_VALUES = [100, 300, 1_000, 3_000, 10_000, 30_000]


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


def compact_config(N, steps, dt, sigma, pairs, seed):
    centers = np.linspace(0.0, 4.0, 25)
    config = qanr_config(
        N=N,
        R=1,
        sigma=sigma,
        alpha=0.0,
        powers=[1, 2],
        fit_pairs=pairs,
        seed=seed,
        dt=dt,
        T=steps,
        reg_method="ridge",
        pair_replacement=True,
        solver="euler",
    )
    config["space_tests"].update(
        type="bump",
        layout="grid",
        L=centers.size,
        centers=centers.tolist(),
        s=0.65,
    )
    config["time_tests"].update(type="bump", K=8, support=0.6)
    config["regression"].update(standardize=False, alpha=0.0)
    config["weak_form"]["normalize_rows"] = True
    config["diagnostics"]["method"] = None
    return config


def simulate_fixed_trajectory(N, steps, dt, sigma):
    X, t, *_ = simulate_particles(
        N=N,
        T=steps,
        dt=dt,
        R=1,
        sigma=sigma,
        d=1,
        init="gaussian_mixture",
        init_centers=[0.5, 2.0, 3.5],
        init_std=0.005,
        kernel_params={"mode": "qanr", "eps": 1e-8, "r_max": 3.0},
        seed=50_000,
        solver="euler",
    )
    return X, t


def grouped(rows, key):
    output = {}
    for P in P_VALUES:
        values = np.asarray([float(row[key]) for row in rows if int(row["P"]) == P])
        output[P] = {
            "mean": float(np.mean(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
    return output


def slope(summary):
    values = [summary[P]["mean"] for P in P_VALUES]
    return float(np.polyfit(np.log(P_VALUES), np.log(values), 1)[0])


def run(out_dir, seeds):
    N, dt, total_time, sigma = 200, 0.01, 2.0, 0.1
    steps = round(total_time / dt)
    theta_true = np.array([1.0, -1.0])
    X, t = simulate_fixed_trajectory(N, steps, dt, sigma)

    full_config = compact_config(N, steps, dt, sigma, N * (N - 1), 0)
    full = fit_kernel_from_particles(
        full_config, X, t, pair_mode="full", return_system=True
    )
    A_full = full["system"]["A_flat"]
    theta_full = np.asarray(full["theta_hat"])
    full_error_sq = float(np.sum((theta_full - theta_true) ** 2))

    rows = []
    started = perf_counter()
    for P in P_VALUES:
        for seed in range(seeds):
            config = compact_config(N, steps, dt, sigma, P, seed)
            fit = fit_kernel_from_particles(
                config,
                X,
                t,
                pair_mode="resample_each_time",
                pair_seed=seed,
                return_system=True,
            )
            A_pair = fit["system"]["A_flat"]
            theta_pair = np.asarray(fit["theta_hat"])
            D = A_pair - A_full
            rows.append(
                {
                    "P": P,
                    "pair_seed": seed,
                    "design_error_sq": float(np.sum(D * D)),
                    "pair_residual_sq": float(np.sum((D @ theta_true) ** 2)),
                    "theta_to_full_sq": float(np.sum((theta_pair - theta_full) ** 2)),
                    "theta_to_true_sq": float(np.sum((theta_pair - theta_true) ** 2)),
                    "full_theta_to_true_sq": full_error_sq,
                    "theta_1": float(theta_pair[0]),
                    "theta_2": float(theta_pair[1]),
                }
            )

    out_dir = Path(out_dir)
    write_rows(out_dir / "compact_support_p_scaling.csv", rows)
    summaries = {
        metric: grouped(rows, metric)
        for metric in (
            "design_error_sq",
            "pair_residual_sq",
            "theta_to_full_sq",
            "theta_to_true_sq",
        )
    }
    report = {
        "seeds": seeds,
        "N": N,
        "dt": dt,
        "sigma": sigma,
        "spatial_centers": "25 equally spaced points on [0,4]",
        "spatial_bump_support_radius": 0.65,
        "temporal_bump_support_length": 0.6,
        "full_theta": theta_full.tolist(),
        "full_theta_error_sq": full_error_sq,
        "slopes": {metric: slope(summary) for metric, summary in summaries.items()},
        "elapsed_seconds": perf_counter() - started,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    plot(rows, summaries, full_error_sq, out_dir / "compact_support_validation.png")
    return report


def plot(rows, summaries, full_error_sq, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for metric, label, marker in (
        ("design_error_sq", r"$\|\widetilde A-A\|_F^2$", "o"),
        ("pair_residual_sq", r"$\|D\theta^*\|_2^2$", "s"),
        ("theta_to_full_sq", r"$\|\hat\theta_{S,P}-\hat\theta_S\|_2^2$", "^"),
    ):
        values = summaries[metric]
        axes[0].loglog(
            P_VALUES,
            [values[P]["mean"] for P in P_VALUES],
            marker + "-",
            label=label,
        )
    reference = summaries["design_error_sq"][1_000]["mean"]
    axes[0].loglog(
        P_VALUES,
        reference * 1_000 / np.asarray(P_VALUES),
        "k--",
        label=r"$P^{-1}$",
    )
    axes[0].set(xlabel=r"sampled pairs $P$", ylabel="mean squared error")
    axes[0].legend()

    theta_summary = summaries["theta_to_true_sq"]
    axes[1].loglog(
        P_VALUES,
        [theta_summary[P]["mean"] for P in P_VALUES],
        "o-",
        label="random-pair OLS",
    )
    axes[1].axhline(full_error_sq, color="tab:red", linestyle="--", label="full-pair OLS")
    axes[1].set(xlabel=r"sampled pairs $P$", ylabel=r"coefficient MSE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Validate pair sampling with fixed compactly supported tests."
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument(
        "--out-dir", default="outputs/validation/compact_support"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate the plot from the existing CSV",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = read_rows(out_dir / "compact_support_p_scaling.csv")
        summaries = {
            metric: grouped(rows, metric)
            for metric in (
                "design_error_sq",
                "pair_residual_sq",
                "theta_to_full_sq",
                "theta_to_true_sq",
            )
        }
        full_error_sq = float(rows[0]["full_theta_to_true_sq"])
        plot(rows, summaries, full_error_sq, out_dir / "compact_support_validation.png")
    else:
        run(out_dir, args.seeds)


if __name__ == "__main__":
    main()
