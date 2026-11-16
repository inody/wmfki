import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
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
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
    }
)

from config import build_config
from kernels import make_radial_basis
from pipeline import fit_kernel_from_particles
from simulate import simulate_particles


CONFIGS = {
    "log2d": ROOT / "configs/experiments/log2d_matched.json",
    "morse10d": ROOT / "configs/experiments/morse_d10_manuscript.json",
}


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


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
        if "normalized_alpha_sk" in row:
            row["alpha_sk"] = row.pop("normalized_alpha_sk")
            row["legacy_unnormalized_alpha_sk"] = row.pop("reported_alpha_sk")
    return rows


def seed_config(name, seed, quick=False):
    config = build_config(read_json(CONFIGS[name]))
    config["simulation"]["seed"] = seed
    config["space_tests"]["seed"] = seed
    config["pair_sampling"]["seed"] = seed + 1_000_000
    config["diagnostics"]["method"] = None
    config["weak_form"]["normalize_rows"] = True

    if quick:
        if name == "log2d":
            config["simulation"].update(N=250, T=20)
            config["pair_sampling"]["total_pairs"] = 5_000
        else:
            config["simulation"].update(N=100, T=100)
            config["pair_sampling"]["total_pairs"] = 1_000
    return config


def simulate(config):
    sim = config["simulation"]
    return simulate_particles(
        N=sim["N"],
        T=sim["T"],
        dt=sim["dt"],
        R=sim["R"],
        sigma=sim["sigma"],
        d=sim["d"],
        init=sim["init"],
        init_scale=sim["init_scale"],
        init_centers=sim.get("init_centers"),
        init_std=sim.get("init_std"),
        kernel_params=sim["interaction"],
        seed=sim["seed"],
        solver=sim.get("solver", "euler"),
    )


def profile_values(name, config, theta, phi_true):
    if name == "log2d":
        r = np.linspace(0.02, 3.0, 500)
    else:
        r = np.linspace(0.05, 5.0, 400)
    radial_basis, _ = make_radial_basis(config["kernel_basis"])
    truth = phi_true(r)
    estimate = radial_basis(r) @ theta
    error = np.linalg.norm(estimate - truth) / np.linalg.norm(truth)
    return float(error)


def sampled_drift_error(K_true, K_hat, X, seed, times=12, targets=64, partners=128):
    rng = np.random.default_rng(seed)
    _, T1, N, _ = X.shape
    time_idx = np.linspace(0, T1 - 1, min(times, T1), dtype=int)
    sq_error = 0.0
    sq_truth = 0.0
    count = 0
    for n in time_idx:
        i_idx = rng.integers(0, N, size=min(targets, N))
        j_idx = rng.integers(0, N, size=(i_idx.size, partners))
        z = X[0, n, i_idx, None, :] - X[0, n, j_idx, :]
        true_drift = K_true(z).mean(axis=1)
        fitted_drift = K_hat(z).mean(axis=1)
        sq_error += float(np.sum((fitted_drift - true_drift) ** 2))
        sq_truth += float(np.sum(true_drift**2))
        count += i_idx.size
    return sq_error / max(count, 1), np.sqrt(sq_error / max(sq_truth, 1e-30))


def run_one(name, seed_index, quick=False):
    seed = 40_000 + seed_index
    config = seed_config(name, seed, quick=quick)
    started = perf_counter()
    X, t, K_true, _, meta = simulate(config)
    simulation_seconds = perf_counter() - started

    fit_started = perf_counter()
    fit = fit_kernel_from_particles(
        config,
        X,
        t,
        pair_mode="resample_each_time",
        pair_seed=config["pair_sampling"]["seed"],
        return_system=True,
    )
    fit_seconds = perf_counter() - fit_started
    theta = np.asarray(fit["theta_hat"])
    A = fit["system"]["A_flat"]
    singular_values = np.linalg.svd(A, compute_uv=False)
    cond_A = float(singular_values[0] / singular_values[-1])
    drift_mse, relative_drift_error = sampled_drift_error(
        K_true, fit["K_hat"], X, seed + 2_000_000
    )
    selected = np.abs(theta) > 1e-8

    row = {
        "experiment": name,
        "seed_index": seed_index,
        "trajectory_seed": seed,
        "pair_seed": config["pair_sampling"]["seed"],
        "N": config["simulation"]["N"],
        "T_steps": config["simulation"]["T"],
        "P": config["pair_sampling"]["total_pairs"],
        "alpha_sk": config["regression"]["alpha"],
        "legacy_unnormalized_alpha_sk": config["regression"].get(
            "legacy_unnormalized_alpha", ""
        ),
        "profile_error": profile_values(
            name, config, theta, meta["phi_true"]
        ),
        "sampled_drift_mse": drift_mse,
        "relative_sampled_drift_error": relative_drift_error,
        "nnz": int(selected.sum()),
        "cond_A": cond_A,
        "cond_AtA": float(np.linalg.cond(A.T @ A)),
        "simulation_seconds": simulation_seconds,
        "fit_seconds": fit_seconds,
        "total_seconds": perf_counter() - started,
    }
    if name == "log2d":
        theta_true = np.zeros(theta.size)
        theta_true[-1] = -1.0 / (2.0 * np.pi)
        row["coefficient_error"] = float(np.linalg.norm(theta - theta_true))
        row["true_atom_selected"] = int(selected[-1])
    else:
        row["coefficient_error"] = ""
        row["true_atom_selected"] = ""
    for index, value in enumerate(theta):
        row[f"theta_{index}"] = float(value)
        row[f"selected_{index}"] = int(selected[index])
    return row


def run_job(job):
    name, seed_index, quick = job
    return run_one(name, seed_index, quick=quick)


def summarize(rows):
    summaries = []
    for name in sorted({row["experiment"] for row in rows}):
        subset = [row for row in rows if row["experiment"] == name]
        summary = {"experiment": name, "seeds": len(subset)}
        for key in (
            "profile_error",
            "relative_sampled_drift_error",
            "nnz",
            "cond_A",
            "simulation_seconds",
            "fit_seconds",
            "total_seconds",
        ):
            values = np.asarray([float(row[key]) for row in subset])
            summary[f"{key}_median"] = float(np.median(values))
            summary[f"{key}_q25"] = float(np.quantile(values, 0.25))
            summary[f"{key}_q75"] = float(np.quantile(values, 0.75))
        atom_count = len([key for key in subset[0] if key.startswith("selected_")])
        for index in range(atom_count):
            summary[f"selection_frequency_{index}"] = float(
                np.mean([float(row[f"selected_{index}"]) for row in subset])
            )
        if name == "log2d":
            summary["true_atom_selection_frequency"] = float(
                np.mean([float(row["true_atom_selected"]) for row in subset])
            )
        summaries.append(summary)
    return summaries


def plot_results(rows, output):
    names = [name for name in ("log2d", "morse10d") if any(row["experiment"] == name for row in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    profile_data = [
        [float(row["profile_error"]) for row in rows if row["experiment"] == name]
        for name in names
    ]
    axes[0].boxplot(profile_data, tick_labels=names, showmeans=True)
    axes[0].set_ylabel("relative radial-profile error")

    for name_index, name in enumerate(names):
        subset = [row for row in rows if row["experiment"] == name]
        atom_count = len([key for key in subset[0] if key.startswith("selected_")])
        frequencies = [
            np.mean([float(row[f"selected_{index}"]) for row in subset])
            for index in range(atom_count)
        ]
        axis = axes[name_index + 1]
        x = np.arange(atom_count)
        color = "tab:blue" if name == "log2d" else "tab:orange"
        axis.bar(x, frequencies, color=color)
        if name == "morse10d":
            labels = [
                "0.600", "0.787", "1.032", "1.353",
                "1.774", "2.326", "3.050", "4.000",
            ]
            axis.set_xticks(x, labels, rotation=45, ha="right")
            axis.set_xlabel(r"dictionary scale $\ell$")
        else:
            axis.set_xlabel("dictionary atom index")
        axis.set(title=name, ylabel="selection frequency", ylim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-seed stability checks for the 2D and 10D examples."
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--experiments", default="log2d,morse10d", help="comma-separated names"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out-dir", default="outputs/validation/multiseed_stability"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate summaries and plots from the existing CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    names = [item.strip() for item in args.experiments.split(",") if item.strip()]
    unknown = sorted(set(names) - set(CONFIGS))
    if unknown:
        raise ValueError(f"Unknown experiments: {', '.join(unknown)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        rows = read_rows(out_dir / "multiseed_runs.csv")
        write_rows(out_dir / "multiseed_runs.csv", rows)
        write_rows(out_dir / "multiseed_summary.csv", summarize(rows))
        plot_results(rows, out_dir / "multiseed_stability.png")
        return

    jobs = [
        (name, seed_index, args.quick)
        for name in names
        for seed_index in range(args.seeds)
    ]
    if args.workers == 1:
        iterator = map(run_job, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        iterator = pool.map(run_job, jobs)
    rows = []
    try:
        for row in iterator:
            rows.append(row)
            write_rows(out_dir / "multiseed_runs.csv", rows)
            print(
                f"completed {row['experiment']} seed {row['seed_index']} "
                f"({len(rows)}/{len(jobs)})",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown()
    summaries = summarize(rows)
    write_rows(out_dir / "multiseed_summary.csv", summaries)
    plot_results(rows, out_dir / "multiseed_stability.png")
    metadata = {
        "quick": args.quick,
        "experiments": names,
        "seeds": args.seeds,
        "configs": {name: str(CONFIGS[name].relative_to(ROOT)) for name in names},
        "workers": args.workers,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
