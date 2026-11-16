import argparse
import csv
import hashlib
import json
import os
import resource
import subprocess
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
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

from features import make_trapz_weights
from pipeline import fit_kernel_from_particles
from scripts.r05_numerical_settings import (
    SPACE_TEST_CENTERS,
    SPACE_TEST_WIDTH,
    apply_fixed_1d_tests,
)
from scripts.run_messenger_qanr_sweep import qanr_config
from simulate import simulate_particles


P_VALUES = [100, 300, 1_000, 3_000, 10_000, 30_000]
DT_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("P", "N", "repeat", "pair_seed"):
            if key in row:
                row[key] = int(row[key])
        if "dt" in row:
            row["dt"] = float(row["dt"])
    return rows


def matched_config(
    N,
    steps,
    dt,
    sigma,
    pairs,
    seed,
    time_test_type="bump",
):
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
        time_s_factor=1.8,
    )
    apply_fixed_1d_tests(
        config,
        time_test_type=time_test_type,
        horizon=steps * dt,
    )
    config["regression"]["standardize"] = False
    config["weak_form"]["normalize_rows"] = True
    config["diagnostics"]["method"] = None
    return config


def simulate_matched(N, steps, dt, sigma, seed):
    config = matched_config(N, steps, dt, sigma, 100, seed)
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
        kernel_params=config["simulation"]["interaction"],
        seed=seed,
        solver="euler",
    )
    return X, t


def run_p_scaling(out_dir, seeds, time_test_type="bump"):
    N, dt, total_time, sigma = 200, 0.01, 2.0, 0.1
    steps = round(total_time / dt)
    trajectory_seed = 20260
    X, t = simulate_matched(N, steps, dt, sigma, trajectory_seed)
    theta_true = np.array([1.0, -1.0])

    full_config = matched_config(
        N, steps, dt, sigma, N * (N - 1), 0, time_test_type
    )
    full = fit_kernel_from_particles(
        full_config, X, t, return_system=True, pair_mode="full"
    )
    A_full = full["system"]["A_flat"]
    theta_full = np.asarray(full["theta_hat"])
    full_error_sq = float(np.sum((theta_full - theta_true) ** 2))
    full_design_seconds = full["diagnostics"]["timing"]["design_seconds"]

    rows = []
    for P in P_VALUES:
        for pair_seed in range(seeds):
            config = matched_config(
                N, steps, dt, sigma, P, pair_seed, time_test_type
            )
            fit = fit_kernel_from_particles(
                config,
                X,
                t,
                return_system=True,
                pair_mode="resample_each_time",
                pair_seed=pair_seed,
            )
            A_pair = fit["system"]["A_flat"]
            D = A_pair - A_full
            theta_pair = np.asarray(fit["theta_hat"])
            rows.append(
                {
                    "P": P,
                    "pair_seed": pair_seed,
                    "design_error_sq": float(np.sum(D * D)),
                    "direction_error_sq": float(np.sum((D @ theta_true) ** 2)),
                    "theta_to_full_sq": float(np.sum((theta_pair - theta_full) ** 2)),
                    "theta_to_true_sq": float(np.sum((theta_pair - theta_true) ** 2)),
                    "full_theta_to_true_sq": full_error_sq,
                    "full_design_seconds": full_design_seconds,
                    "design_seconds": fit["diagnostics"]["timing"]["design_seconds"],
                }
            )
    path = Path(out_dir) / "p_scaling.csv"
    write_rows(path, rows)
    return rows


def fixed_space_config(config, time_test_type="bump"):
    return apply_fixed_1d_tests(
        config,
        time_test_type=time_test_type,
        horizon=2.0,
    )


def run_resampling_comparison(out_dir, seeds, time_test_type="bump"):
    N, fine_dt, total_time, sigma, P = 200, 0.005, 2.0, 0.1, 3_000
    fine_steps = round(total_time / fine_dt)
    X_fine, t_fine = simulate_matched(N, fine_steps, fine_dt, sigma, 20261)
    base = fixed_space_config(
        matched_config(
            N, fine_steps, fine_dt, sigma, P, 0, time_test_type
        ),
        time_test_type,
    )

    rows = []
    for dt in DT_VALUES:
        stride = round(dt / fine_dt)
        X = X_fine[:, ::stride]
        t = t_fine[::stride]
        steps = t.size - 1
        config = fixed_space_config(
            matched_config(N, steps, dt, sigma, P, 0, time_test_type),
            time_test_type,
        )
        config["space_tests"].update(
            centers=base["space_tests"]["centers"], s=base["space_tests"]["s"]
        )
        config["time_tests"]["support"] = base["time_tests"]["support"]
        full = fit_kernel_from_particles(
            config, X, t, return_system=True, pair_mode="full"
        )
        A_full = full["system"]["A_flat"]
        q_dt = float(np.sum(make_trapz_weights(t) ** 2))
        for mode in ("resample_each_time", "fixed_across_time"):
            for pair_seed in range(seeds):
                fit = fit_kernel_from_particles(
                    config,
                    X,
                    t,
                    return_system=True,
                    pair_mode=mode,
                    pair_seed=pair_seed,
                )
                D = fit["system"]["A_flat"] - A_full
                rows.append(
                    {
                        "dt": dt,
                        "P": P,
                        "mode": mode,
                        "pair_seed": pair_seed,
                        "Q_dt": q_dt,
                        "design_error_sq": float(np.sum(D * D)),
                        "design_seconds": fit["diagnostics"]["timing"]["design_seconds"],
                    }
                )
    write_rows(Path(out_dir) / "resampling_comparison.csv", rows)
    return rows


def synthetic_positions(N, steps, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.8, size=(N, 1))
    velocity = rng.normal(0.0, 0.1, size=(N, 1))
    t = np.linspace(0.0, 1.0, steps + 1)
    X = base[None, None, :, :] + t[None, :, None, None] * velocity[None, None]
    return X, t


def rss_megabytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def runtime_worker(N, mode, repeat):
    steps = 20
    X, t = synthetic_positions(N, steps, seed=repeat)
    P = 10 * N
    config = matched_config(N, steps, 1.0 / steps, 0.0, P, repeat)
    config["time_tests"].update(K=4, support=0.4)
    config["weak_form"]["full_pair_chunk_size"] = 100_000
    baseline_rss_mb = rss_megabytes()
    fit = fit_kernel_from_particles(
        config, X, t, pair_mode=mode, pair_seed=repeat
    )
    peak_rss_mb = rss_megabytes()
    payload = {
        "N": N,
        "mode": mode,
        "repeat": repeat,
        "P": N * (N - 1) if mode == "full" else P,
        "design_seconds": fit["diagnostics"]["timing"]["design_seconds"],
        "baseline_rss_mb": baseline_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "peak_rss_delta_mb": max(0.0, peak_rss_mb - baseline_rss_mb),
    }
    print(json.dumps(payload))


def run_runtime(out_dir, repeats, max_n):
    Ns = [N for N in [100, 200, 400, 800, 1600, 3200] if N <= max_n]
    rows = []
    script = Path(__file__).resolve()
    for N in Ns:
        for mode in ("full", "resample_each_time"):
            for repeat in range(repeats):
                command = [
                    sys.executable,
                    str(script),
                    "--runtime-worker",
                    "--worker-N",
                    str(N),
                    "--worker-mode",
                    mode,
                    "--worker-repeat",
                    str(repeat),
                ]
                completed = subprocess.run(
                    command, check=True, capture_output=True, text=True
                )
                rows.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    write_rows(Path(out_dir) / "runtime_scaling.csv", rows)
    return rows


def grouped(rows, x_key, y_key, group_key=None):
    values = {}
    for row in rows:
        key = (row[group_key], row[x_key]) if group_key else row[x_key]
        values.setdefault(key, []).append(float(row[y_key]))
    return {
        key: (np.mean(v), np.std(v), len(v))
        for key, v in values.items()
    }


def add_mean_with_ci(ax, x, summary, keys, style, label):
    means = np.asarray([summary[key][0] for key in keys])
    standard_errors = np.asarray(
        [summary[key][1] / np.sqrt(summary[key][2]) for key in keys]
    )
    ax.loglog(x, means, style, label=label)
    lower = np.maximum(means - 1.96 * standard_errors, np.finfo(float).tiny)
    upper = means + 1.96 * standard_errors
    ax.fill_between(x, lower, upper, alpha=0.16)


def loglog_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def summarize_results(p_rows, resampling_rows, runtime_rows):
    summary = {"p_scaling_slopes": {}, "dt_scaling_slopes": {}, "runtime_slopes": {}}
    for metric in (
        "design_error_sq",
        "direction_error_sq",
        "theta_to_full_sq",
        "theta_to_true_sq",
    ):
        values = grouped(p_rows, "P", metric)
        summary["p_scaling_slopes"][metric] = loglog_slope(
            P_VALUES, [values[P][0] for P in P_VALUES]
        )
    values = grouped(resampling_rows, "dt", "design_error_sq", "mode")
    for mode in ("resample_each_time", "fixed_across_time"):
        summary["dt_scaling_slopes"][mode] = loglog_slope(
            DT_VALUES, [values[(mode, dt)][0] for dt in DT_VALUES]
        )
    Ns = sorted({int(row["N"]) for row in runtime_rows})
    for mode in ("full", "resample_each_time"):
        medians = [
            np.median(
                [
                    float(row["design_seconds"])
                    for row in runtime_rows
                    if row["mode"] == mode and int(row["N"]) == N
                ]
            )
            for N in Ns
        ]
        summary["runtime_slopes"][mode] = loglog_slope(Ns, medians)
    return summary


def plot_results(out_dir, p_rows, resampling_rows, runtime_rows):
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))

    styles = [
        ("design_error_sq", r"$\|\widetilde A-A\|_F^2$"),
        ("direction_error_sq", r"$\|D\theta^*\|_2^2$"),
        ("theta_to_full_sq", r"$\|\hat\theta_{S,P}-\hat\theta_S\|_2^2$"),
    ]
    P = np.array(P_VALUES, dtype=float)
    for metric, label in styles:
        summary = grouped(p_rows, "P", metric)
        add_mean_with_ci(axes[0], P, summary, P_VALUES, "o-", label)
    reference = grouped(p_rows, "P", "design_error_sq")[P_VALUES[2]][0]
    axes[0].loglog(P, reference * P_VALUES[2] / P, "k--", label=r"$P^{-1}$")
    full_error = float(p_rows[0]["full_theta_to_true_sq"])
    axes[0].axhline(
        full_error,
        color="0.45",
        linestyle=":",
        label=r"full-pair $\|\hat\theta_S-\theta_S^*\|_2^2$",
    )
    axes[0].set(xlabel=r"sampled pairs $P$", ylabel="mean squared error")
    axes[0].legend()

    summary = grouped(resampling_rows, "dt", "design_error_sq", "mode")
    for mode, marker in [("resample_each_time", "o"), ("fixed_across_time", "s")]:
        keys = [(mode, dt) for dt in DT_VALUES]
        add_mean_with_ci(
            axes[1],
            np.asarray(DT_VALUES),
            summary,
            keys,
            marker + "-",
            mode.replace("_", " "),
        )
    ref = summary[("resample_each_time", DT_VALUES[2])][0]
    axes[1].loglog(
        DT_VALUES,
        ref * np.asarray(DT_VALUES) / DT_VALUES[2],
        "k--",
        label=r"$O(\Delta t)$",
    )
    axes[1].set(xlabel=r"observation step $\Delta t$", ylabel=r"$\mathbb{E}_{\mathrm{pair}}\|D\|_F^2$")
    axes[1].legend()

    summary = grouped(runtime_rows, "N", "design_seconds", "mode")
    Ns = sorted({int(row["N"]) for row in runtime_rows})
    for mode, marker in [("full", "o"), ("resample_each_time", "s")]:
        medians = []
        lower = []
        upper = []
        for N in Ns:
            vals = [float(r["design_seconds"]) for r in runtime_rows if r["mode"] == mode and int(r["N"]) == N]
            medians.append(np.median(vals))
            lower.append(np.quantile(vals, 0.25))
            upper.append(np.quantile(vals, 0.75))
        axes[2].loglog(Ns, medians, marker + "-", label=mode.replace("_", " "))
        axes[2].fill_between(Ns, lower, upper, alpha=0.16)
    axes[2].set(xlabel="particles $N$", ylabel="design assembly time [s]")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "random_pair_validation.png", dpi=220)
    plt.close(fig)

    p_summary = grouped(p_rows, "P", "theta_to_true_sq")
    time_summary = grouped(p_rows, "P", "design_seconds")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.loglog(
        [time_summary[P][0] for P in P_VALUES],
        [p_summary[P][0] for P in P_VALUES],
        "o-",
        label="random pair",
    )
    ax.scatter(
        [float(p_rows[0]["full_design_seconds"])],
        [full_error],
        marker="*",
        s=120,
        color="tab:red",
        label="full pair",
    )
    for P, x, y in zip(
        P_VALUES,
        [time_summary[P][0] for P in P_VALUES],
        [p_summary[P][0] for P in P_VALUES],
    ):
        ax.annotate(f"P={P:g}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set(
        xlabel="design assembly time [s]",
        ylabel=r"$\mathbb{E}_{\mathrm{pair}}\|\hat\theta_{S,P}-\theta_S^*\|_2^2$",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "random_pair_accuracy_runtime.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/experiments/random_pair_validation")
    parser.add_argument("--pair-seeds", type=int, default=100)
    parser.add_argument("--runtime-repeats", type=int, default=10)
    parser.add_argument("--runtime-max-n", type=int, default=1600)
    parser.add_argument("--runtime-worker", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--time-test-type",
        choices=("gaussian", "bump"),
        default="bump",
    )
    parser.add_argument("--worker-N", type=int, default=100)
    parser.add_argument("--worker-mode", default="full")
    parser.add_argument("--worker-repeat", type=int, default=0)
    args = parser.parse_args()

    if args.runtime_worker:
        runtime_worker(args.worker_N, args.worker_mode, args.worker_repeat)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    if args.plot_only:
        p_rows = read_rows(out_dir / "p_scaling.csv")
        resampling_rows = read_rows(out_dir / "resampling_comparison.csv")
        runtime_rows = read_rows(out_dir / "runtime_scaling.csv")
    elif args.runtime_only:
        runtime_rows = run_runtime(out_dir, args.runtime_repeats, args.runtime_max_n)
        summary = {"runtime_slopes": {}}
        Ns = sorted({int(row["N"]) for row in runtime_rows})
        if len(Ns) >= 2:
            for mode in ("full", "resample_each_time"):
                medians = [
                    np.median([
                        float(row["design_seconds"]) for row in runtime_rows
                        if row["mode"] == mode and int(row["N"]) == N
                    ]) for N in Ns
                ]
                summary["runtime_slopes"][mode] = loglog_slope(Ns, medians)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        runtime_sources = (
            "features.py", "pipeline.py", "simulate.py", "tests.py", "time_tests.py",
            "scripts/r05_numerical_settings.py", "scripts/run_random_pair_validation.py",
        )
        metadata = {
            "task": "standalone_runtime_scaling",
            "runtime_repeats": args.runtime_repeats,
            "runtime_max_n": args.runtime_max_n,
            "elapsed_seconds": perf_counter() - start,
            "time_test_type": args.time_test_type,
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in runtime_sources
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return
    else:
        p_rows = run_p_scaling(
            out_dir, args.pair_seeds, args.time_test_type
        )
        resampling_rows = run_resampling_comparison(
            out_dir, args.pair_seeds, args.time_test_type
        )
        runtime_rows = run_runtime(out_dir, args.runtime_repeats, args.runtime_max_n)
    plot_results(out_dir, p_rows, resampling_rows, runtime_rows)
    summary = summarize_results(p_rows, resampling_rows, runtime_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    metadata = {
        "pair_seeds": args.pair_seeds,
        "runtime_repeats": args.runtime_repeats,
        "runtime_max_n": args.runtime_max_n,
        "elapsed_seconds": perf_counter() - start,
        "trajectory_policy": "fixed within each batch-sampling experiment",
        "time_test_type": args.time_test_type,
        "space_test_selection": "deterministic and fixed before all trajectories",
        "space_centers": SPACE_TEST_CENTERS.tolist(),
        "space_width": SPACE_TEST_WIDTH,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
