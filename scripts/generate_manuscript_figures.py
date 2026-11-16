"""Generate the seven vector figures used by the legacy R05 manuscript.

This script is not part of the current L4DC figure pipeline. See
``scripts/regenerate_l4dc_figures.py`` for the active one-command entry point.

The validation panels are regenerated from archived CSV files. The 2D and 10D
tuning-run coefficients are cached after the deterministic seeded fits; pass
``--refit`` to recompute them from the experiment configurations.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from kernels import make_kernel_basis, make_kernel_evaluator, make_radial_basis
from pipeline import fit_kernel_from_particles
from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR,
    REFERENCE_COLOR,
    SECONDARY_COLOR,
    THREE_PANEL_SIZE,
    TRUE_COLOR,
    TWO_PANEL_SIZE,
    apply_manuscript_style,
    finish_axes,
    save_pdf,
)
from simulate import make_true_kernel, simulate_particles, simulate_with_kernel


FIGDIR = ROOT / "latex" / "R05" / "figures"
DATA = ROOT / "data" / "manuscript" / "R05"
CACHE = DATA / "recovery" / "coefficients.json"
P_VALUES = [100, 300, 1_000, 3_000, 10_000, 30_000]


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(rows, group_key, metric, secondary_key=None):
    output = {}
    for row in rows:
        key = row[group_key]
        try:
            key = int(key)
        except ValueError:
            key = float(key)
        if secondary_key is not None:
            key = (row[secondary_key], key)
        output.setdefault(key, []).append(float(row[metric]))
    return output


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return mean, mean - half, mean + half


def add_mean_ci(ax, x, summary, keys, label, color, marker):
    stats = [mean_ci(summary[key]) for key in keys]
    means = np.array([item[0] for item in stats])
    lower = np.array([item[1] for item in stats])
    upper = np.array([item[2] for item in stats])
    ax.plot(x, means, marker=marker, color=color, label=label)
    ax.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)


def one_dimensional_sparse_recovery():
    root = DATA / "basic_recovery"
    summary = json.loads((root / "summary.json").read_text())
    profile = read_csv(root / "profile_curves.csv")
    trajectories = read_csv(root / "trajectory_curves.csv")

    radius = np.asarray([float(row["radius"]) for row in profile])
    true_profile = np.asarray([float(row["true_profile"]) for row in profile])
    learned_profile = np.asarray([
        float(row["learned_profile"]) for row in profile
    ])
    theta = np.asarray(summary["coefficients"], dtype=float)
    theta_true = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    labels = [
        r"$1/(r+\varepsilon_K)$",
        r"$1$",
        r"$e^{-r/0.25}$",
        r"$e^{-r/0.5}$",
        r"$e^{-r}$",
        r"$e^{-r/2}$",
    ]

    fig, axes = plt.subplots(1, 3, figsize=THREE_PANEL_SIZE, constrained_layout=True)
    axes[0].plot(radius, true_profile, color=TRUE_COLOR, label=r"True $\phi(r)$")
    axes[0].plot(
        radius,
        learned_profile,
        "--",
        color=ESTIMATE_COLOR,
        label=r"LASSO $\hat\phi(r)$",
    )
    axes[0].set(
        xlabel=r"distance $r$",
        ylabel=r"radial profile $\phi(r)$",
        title=r"(a) Radial profile",
    )
    axes[0].legend(loc="best")

    positions = np.arange(theta.size)
    axes[1].bar(
        positions - 0.2,
        theta_true,
        0.38,
        color=TRUE_COLOR,
        label=r"True $\theta^\star$",
    )
    axes[1].bar(
        positions + 0.2,
        theta,
        0.38,
        color=ESTIMATE_COLOR,
        label=r"LASSO $\hat\theta$",
    )
    axes[1].axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8)
    axes[1].set_xticks(positions, labels, rotation=38, ha="right")
    axes[1].set(
        ylabel=r"coefficient $\theta_m$",
        title=r"(b) Dictionary coefficients",
    )
    axes[1].legend(loc="best")

    particle_indices = list(dict.fromkeys(
        row["particle_index"] for row in trajectories
    ))
    for particle_number, particle_index in enumerate(particle_indices):
        particle_rows = [
            row for row in trajectories
            if row["particle_index"] == particle_index
        ]
        time = np.asarray([float(row["time"]) for row in particle_rows])
        reference = np.asarray([
            float(row["reference_position"]) for row in particle_rows
        ])
        learned = np.asarray([
            float(row["learned_position"]) for row in particle_rows
        ])
        axes[2].plot(
            time,
            reference,
            color=TRUE_COLOR,
            linewidth=0.45,
            alpha=0.40,
            label="true" if particle_number == 0 else None,
        )
        axes[2].plot(
            time,
            learned,
            "--",
            color=ESTIMATE_COLOR,
            linewidth=0.45,
            alpha=0.52,
            label="learned" if particle_number == 0 else None,
        )
    axes[2].set(
        xlabel=r"time $t$",
        ylabel="particle position",
        title=r"(c) Coupled dynamics",
    )
    axes[2].legend(loc="best")
    for ax in axes:
        finish_axes(ax)
    save_pdf(fig, FIGDIR / "one_dimensional_sparse_recovery.pdf")


def random_pair_validation():
    root = DATA / "random_pair"
    p_rows = read_csv(root / "p_scaling.csv")
    runtime = read_csv(root / "runtime_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL_SIZE, constrained_layout=True)

    p = np.asarray(P_VALUES, dtype=float)
    coefficient_gap_sq = grouped(p_rows, "P", "theta_to_full_sq")
    add_mean_ci(
        axes[0],
        p,
        coefficient_gap_sq,
        P_VALUES,
        r"random pair vs. full pair",
        TRUE_COLOR,
        "o",
    )
    reference = mean_ci(coefficient_gap_sq[1_000])[0]
    axes[0].loglog(
        p,
        reference * 1_000 / p,
        "--",
        color=REFERENCE_COLOR,
        label=r"$P^{-1}$",
    )
    axes[0].set(
        xlabel=r"sampled pairs $P$",
        ylabel=r"sample mean of $\operatorname{Gap}_{\theta,P}^2$",
        title=r"(a) Accuracy versus sampled pairs",
    )
    axes[0].legend(loc="best")

    ns = sorted({int(row["N"]) for row in runtime})
    for mode, label, color, marker in (
        ("full", "full pair", TRUE_COLOR, "o"),
        ("resample_each_time", r"random pair, $P=10N$", SECONDARY_COLOR, "s"),
    ):
        values = [
            [
                float(row["design_seconds"])
                for row in runtime
                if row["mode"] == mode and int(row["N"]) == n
            ]
            for n in ns
        ]
        median = np.array([np.median(item) for item in values])
        lower = np.array([np.quantile(item, 0.25) for item in values])
        upper = np.array([np.quantile(item, 0.75) for item in values])
        axes[1].loglog(ns, median, marker=marker, color=color, label=label)
        axes[1].fill_between(
            ns, lower, upper, color=color, alpha=0.16, linewidth=0
        )
    axes[1].set(
        xlabel=r"particles $N$",
        ylabel="assembly time [s]",
        title=r"(b) Computational cost",
    )
    axes[1].legend(loc="upper left")
    for ax in axes:
        finish_axes(ax)
    save_pdf(fig, FIGDIR / "random_pair_validation.pdf")


def robustness_and_scaling():
    noise_rows = read_csv(DATA / "observation_noise" / "sindy_vs_weak.csv")
    euler = read_csv(DATA / "full_pair" / "dt_scaling.csv")
    stochastic = read_csv(DATA / "full_pair" / "stochastic_scaling.csv")

    euler_dt = np.asarray([float(row["dt"]) for row in euler])
    euler_l2 = np.asarray([float(row["theta_l2"]) for row in euler])
    euler_dt, euler_l2 = [
        np.asarray(values) for values in zip(*sorted(zip(euler_dt, euler_l2)))
    ]
    sigma = np.array(sorted({
        float(row["sigma"])
        for row in stochastic
        if row["kind"] == "sigma"
    }))
    sigma_rmse = np.array([
        np.sqrt(np.mean([
            float(row["theta_l2"]) ** 2
            for row in stochastic
            if row["kind"] == "sigma" and float(row["sigma"]) == value
        ]))
        for value in sigma
    ])
    particles = np.array(sorted({
        int(row["particles"])
        for row in stochastic
        if row["kind"] == "particles"
    }))
    particle_rmse = np.array([
        np.sqrt(np.mean([
            float(row["theta_l2"]) ** 2
            for row in stochastic
            if row["kind"] == "particles" and int(row["particles"]) == value
        ]))
        for value in particles
    ])

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), constrained_layout=True)
    axes = axes.ravel()
    noise_levels = np.asarray(sorted({
        float(row["sigma_obs"]) for row in noise_rows
    }))
    for method, label, color, marker in (
        ("weak_form", "weak form", TRUE_COLOR, "o"),
        ("sindy_euler", "Euler difference", ESTIMATE_COLOR, "s"),
    ):
        rmse = np.asarray([
            np.sqrt(np.mean([
                float(row["coefficient_error"]) ** 2
                for row in noise_rows
                if row["method"] == method
                and float(row["sigma_obs"]) == level
            ]))
            for level in noise_levels
        ])
        positive = noise_levels > 0.0
        axes[0].loglog(
            noise_levels[positive],
            rmse[positive],
            marker=marker,
            color=color,
            label=label,
        )
        axes[0].axhline(
            rmse[~positive][0], color=color, linestyle=":", linewidth=0.9
        )
    swept = sorted({
        float(row["sigma_obs"]) for row in noise_rows
        if float(row["sigma_obs"]) > 0.0
    })
    axes[0].set_xlim(min(swept) / 1.6, max(swept) * 1.6)
    axes[0].set(
        xlabel=r"observation noise $\sigma_{\rm obs}$",
        ylabel=r"$\operatorname{RMSE}_{\theta}$",
        title=r"(a) Observation-noise robustness",
    )
    axes[0].legend(loc="upper left")

    axes[1].loglog(sigma, sigma_rmse, "o-", color=TRUE_COLOR)
    reference = np.array([sigma.min(), sigma.max()])
    axes[1].loglog(
        reference,
        sigma_rmse[1] * reference / sigma[1],
        "--",
        color=REFERENCE_COLOR,
        label=r"$O(\sigma)$",
    )
    axes[1].set(
        xlabel=r"process noise $\sigma$",
        ylabel=r"$\operatorname{RMSE}_{\theta}$",
        title=r"(b) Process-noise scaling",
    )
    axes[1].legend(loc="upper left")

    reference = np.array([particles.min(), particles.max()], dtype=float)
    axes[2].loglog(particles, particle_rmse, "o-", color=TRUE_COLOR)
    axes[2].loglog(
        reference,
        particle_rmse[1] * (100.0 / reference) ** 0.5,
        "--",
        color=REFERENCE_COLOR,
        label=r"$O(N^{-1/2})$",
    )
    axes[2].set(
        xlabel=r"particles $N$",
        ylabel=r"$\operatorname{RMSE}_{\theta}$",
        title=r"(c) Particle-number scaling",
    )
    axes[2].set_xticks(particles, [str(value) for value in particles])
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].legend(loc="upper right")

    axes[3].loglog(euler_dt, euler_l2, "o-", color=TRUE_COLOR)
    reference = np.array([euler_dt.min(), euler_dt.max()])
    anchor = int(np.argmin(np.abs(euler_dt - 0.05)))
    axes[3].loglog(
        reference,
        euler_l2[anchor] * reference / euler_dt[anchor],
        "--",
        color=REFERENCE_COLOR,
        label=r"$O(\Delta t)$",
    )
    axes[3].set(
        xlabel=r"observation step $\Delta t$",
        ylabel=r"$\operatorname{Err}_{\theta}$",
        title=r"(d) Time-step scaling",
    )
    axes[3].legend(loc="upper left")
    for ax in axes:
        finish_axes(ax)
    save_pdf(fig, FIGDIR / "robustness_and_scaling.pdf")


def load_or_fit_coefficients(refit=False):
    if CACHE.exists() and not refit:
        return json.loads(CACHE.read_text())
    output = {}
    for name, config_path in (
        ("log2d", ROOT / "configs/experiments/log2d_matched.json"),
        ("morse10d", ROOT / "configs/experiments/morse_d10_manuscript.json"),
    ):
        config = json.loads(config_path.read_text())
        sim = config["simulation"]
        X, t, *_ = simulate_particles(
            N=sim["N"], T=sim["T"], dt=sim["dt"], R=sim["R"],
            sigma=sim["sigma"], d=sim["d"], init=sim["init"],
            init_scale=sim["init_scale"], seed=sim["seed"],
            kernel_params=sim["interaction"], solver=sim.get("solver", "euler"),
        )
        fit = fit_kernel_from_particles(
            config, X, t, pair_mode="resample_each_time",
            pair_seed=config["pair_sampling"]["seed"],
        )
        output[name] = [float(value) for value in fit["theta_hat"]]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(output, indent=2))
    return output


def recovery_figures(coefficients):
    delta = 1e-2
    theta_log = np.asarray(coefficients["log2d"])
    phi_log, _ = make_radial_basis({"basis": "messenger_log2d", "eps": delta, "powers": []})
    r = np.linspace(0.02, 3.0, 500)
    true = -1.0 / (2 * np.pi * (r**2 + delta**2))
    estimated = phi_log(r) @ theta_log
    labels = [str(index) for index in range(1, 9)]
    theta_true = np.zeros(8)
    theta_true[-1] = -1 / (2 * np.pi)
    _recovery_figure(
        r, true, estimated, theta_true, theta_log, labels,
        r"Dictionary atom index $m$", FIGDIR / "log2d_recovery.pdf", (-20, 2),
    )

    ells = np.array([0.600, 0.787, 1.032, 1.353, 1.774, 2.326, 3.050, 4.000])
    theta_morse = np.asarray(coefficients["morse10d"])
    phi_morse, _ = make_radial_basis({"basis": "exp_r", "ell_grid": ells.tolist(), "M": 8})
    r = np.linspace(0.05, 5.0, 400)
    true = r * (2 * np.exp(-r) - np.exp(-r / 2))
    estimated = phi_morse(r) @ theta_morse
    beta_true = np.array([2.0, -1.0])
    labels = [f"{value:.3f}" for value in ells]
    _recovery_figure(
        r, true, estimated, beta_true, theta_morse, labels,
        "interaction scale", FIGDIR / "morse_recovery.pdf", None,
        true_scales=np.array([1.0, 2.0]),
    )


def seeded_initial_condition_and_noise(simulation):
    """Reproduce the seeded Euler--Maruyama initial state and increments."""
    rng = np.random.default_rng(simulation["seed"])
    R = int(simulation["R"])
    N = int(simulation["N"])
    d = int(simulation["d"])
    scale = float(simulation["init_scale"])
    initial = np.empty((R, N, d), dtype=float)
    for run in range(R):
        if simulation["init"] == "normal":
            initial[run] = rng.normal(0.0, scale, size=(N, d))
        elif simulation["init"] == "uniform_ball":
            directions = rng.normal(0.0, 1.0, size=(N, d))
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
            radii = scale * rng.random(size=(N, 1)) ** (1.0 / d)
            initial[run] = directions * radii
        else:
            raise ValueError(f"Unsupported manuscript initial condition: {simulation['init']}")
    noise = rng.normal(
        0.0,
        1.0,
        size=(R, int(simulation["T"]), N, d),
    )
    return initial, noise


def coupled_dynamics(config_path, theta):
    config = json.loads(Path(config_path).read_text())
    simulation = config["simulation"]
    initial, noise = seeded_initial_condition_and_noise(simulation)
    time = np.linspace(
        0.0,
        float(simulation["dt"]) * int(simulation["T"]),
        int(simulation["T"]) + 1,
    )
    true_kernel, *_ = make_true_kernel(simulation["interaction"])
    basis, _ = make_kernel_basis(config["kernel_basis"])
    learned_kernel = make_kernel_evaluator(np.asarray(theta), basis)
    kwargs = {
        "X0": initial,
        "t": time,
        "sigma": float(simulation["sigma"]),
        "noise": noise,
    }
    true = simulate_with_kernel(true_kernel, **kwargs)
    learned = simulate_with_kernel(learned_kernel, **kwargs)
    return time, true, learned


def dynamics_figures(coefficients):
    log_config = ROOT / "configs/experiments/log2d_matched.json"
    time, true, learned = coupled_dynamics(log_config, coefficients["log2d"])
    _log2d_snapshots(time, true, learned)
    del true, learned

    morse_config = ROOT / "configs/experiments/morse_d10_manuscript.json"
    time, true, learned = coupled_dynamics(morse_config, coefficients["morse10d"])
    _morse_collective_statistics(time, true, learned)


def _log2d_snapshots(time, true, learned):
    indices = [0, (time.size - 1) // 2, time.size - 1]
    combined = np.concatenate([true[0, indices], learned[0, indices]], axis=0)
    limit = 1.05 * float(np.max(np.abs(combined)))
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.5), sharex=True, sharey=True)
    for row, (trajectory, color, label) in enumerate(
        ((true, TRUE_COLOR, "True kernel"), (learned, ESTIMATE_COLOR, "Learned kernel"))
    ):
        for column, index in enumerate(indices):
            ax = axes[row, column]
            points = trajectory[0, index]
            ax.scatter(points[:, 0], points[:, 1], s=2.2, alpha=0.38, color=color, linewidths=0)
            ax.set_aspect("equal")
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            if row == 0:
                ax.set_title(rf"$t={time[index]:g}$")
            if column == 0:
                ax.set_ylabel(label + "\n" + r"position $x_2$")
            if row == 1:
                ax.set_xlabel(r"position $x_1$")
            finish_axes(ax)
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.11, top=0.92, wspace=0.12, hspace=0.12)
    save_pdf(fig, FIGDIR / "log2d_coupled_dynamics.pdf")


def _morse_collective_statistics(time, true, learned):
    fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL_SIZE, constrained_layout=True)
    for trajectory, color, label in (
        (true, TRUE_COLOR, "True kernel"),
        (learned, ESTIMATE_COLOR, "Learned kernel"),
    ):
        radii = np.linalg.norm(trajectory[0], axis=-1)
        mean = np.mean(radii, axis=1)
        lower, upper = np.quantile(radii, [0.1, 0.9], axis=1)
        axes[0].plot(time, mean, color=color, label=label)
        axes[0].fill_between(time, lower, upper, color=color, alpha=0.16, linewidth=0)
    axes[0].set(
        xlabel=r"time $t$",
        ylabel=r"particle radius $\|X_i(t)\|_2$",
        title="(a) Mean radius and 10--90% band",
    )
    axes[0].legend(loc="best")

    true_distances = _pairwise_distances(true[0, -1])
    learned_distances = _pairwise_distances(learned[0, -1])
    upper = max(np.quantile(true_distances, 0.995), np.quantile(learned_distances, 0.995))
    bins = np.linspace(0.0, upper, 55)
    axes[1].hist(
        true_distances,
        bins=bins,
        density=True,
        histtype="stepfilled",
        linewidth=1.6,
        color=TRUE_COLOR,
        edgecolor=TRUE_COLOR,
        alpha=0.30,
        label="True kernel",
    )
    axes[1].hist(
        learned_distances,
        bins=bins,
        density=True,
        histtype="stepfilled",
        linewidth=1.6,
        color=ESTIMATE_COLOR,
        edgecolor=ESTIMATE_COLOR,
        alpha=0.30,
        label="Learned kernel",
    )
    axes[1].set(
        xlabel="pairwise distance at final time",
        ylabel="density",
        title=r"(b) Final pairwise-distance distribution",
    )
    axes[1].legend(loc="best")
    for ax in axes:
        finish_axes(ax)
    save_pdf(fig, FIGDIR / "morse_coupled_dynamics.pdf")


def _pairwise_distances(points):
    i, j = np.triu_indices(points.shape[0], k=1)
    return np.linalg.norm(points[i] - points[j], axis=1)


def _recovery_figure(
    r, true, estimated, theta_true, theta_hat, labels, xlabel, output, ylim,
    true_scales=None,
):
    fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL_SIZE, constrained_layout=True)
    axes[0].plot(r, true, color=TRUE_COLOR, label=r"True $\phi(r)$")
    axes[0].plot(r, estimated, "--", color=ESTIMATE_COLOR, label=r"LASSO $\hat\phi_P(r)$")
    axes[0].axhline(0, color=REFERENCE_COLOR, linestyle=":", linewidth=0.9)
    axes[0].set(xlabel=r"distance $r$", ylabel=r"radial profile $\phi(r)$", title=r"(a) Radial profile")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].legend(loc="best")

    if true_scales is not None:
        x = np.asarray(labels, dtype=float)
        axes[1].bar(
            true_scales, theta_true, 0.18, color=TRUE_COLOR,
            label=r"Generating $\beta^\star$",
        )
        axes[1].bar(
            x, theta_hat, 0.18, color=ESTIMATE_COLOR,
            label=r"LASSO $\hat\theta_P$",
        )
        coefficient_ylabel = "coefficient value"
        coefficient_title = r"(b) Coefficients by scale"
    else:
        x = np.arange(len(labels))
        axes[1].bar(x - 0.2, theta_true, 0.38, color=TRUE_COLOR, label=r"True $\theta^\star$")
        axes[1].bar(x + 0.2, theta_hat, 0.38, color=ESTIMATE_COLOR, label=r"LASSO $\hat\theta_P$")
        coefficient_ylabel = r"coefficient $\theta_m$"
        coefficient_title = r"(b) Dictionary coefficients"
    axes[1].axhline(0, color=REFERENCE_COLOR, linewidth=0.9)
    axes[1].set_xticks(x, labels, rotation=40, ha="right")
    axes[1].set(xlabel=xlabel, ylabel=coefficient_ylabel, title=coefficient_title)
    axes[1].legend(loc="best")
    for ax in axes:
        finish_axes(ax)
    save_pdf(fig, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit", action="store_true", help="recompute the 2D and 10D tuning-run coefficients")
    args = parser.parse_args()
    apply_manuscript_style()
    one_dimensional_sparse_recovery()
    robustness_and_scaling()
    random_pair_validation()
    coefficients = load_or_fit_coefficients(refit=args.refit)
    recovery_figures(coefficients)
    dynamics_figures(coefficients)
    print(f"Generated seven manuscript PDFs in {FIGDIR}")


if __name__ == "__main__":
    main()
