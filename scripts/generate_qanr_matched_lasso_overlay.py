"""Generate Figure 2 for the realizable six-atom LASSO/OLS study."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR,
    REFERENCE_COLOR,
    SECONDARY_COLOR,
    TERTIARY_COLOR,
    TRUE_COLOR,
    apply_manuscript_style,
    finish_axes,
)

DATA = ROOT / "data/manuscript/R05"
P_VALUES = [100, 300, 1_000, 3_000, 10_000, 30_000]
PANEL_BOX_ASPECT = 3 / 4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text())


def grouped(rows, group_key, metric):
    output = {}
    for row in rows:
        key = int(row[group_key])
        output.setdefault(key, []).append(float(row[metric]))
    return output


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return mean, mean - half, mean + half


def generate_p50(batch, destination, time_step_run=None, strong_lasso_noise_run=None):
    summary_ad = read_json(batch / "summary_a_to_d.json")
    summary_e = read_json(batch / "summary_e.json")["conditions"]
    summary_f = read_json(batch / "summary_f.json")["conditions"]
    requested = read_json(batch / "requested_protocol.json")
    ratio = int(requested["ratios"][0])
    time_summary = None
    time_manifest = None
    strong_lasso_summary = None
    strong_lasso_summary_path = None
    if strong_lasso_noise_run is not None:
        strong_lasso_noise_run = Path(strong_lasso_noise_run).resolve()
        strong_lasso_summary_path = strong_lasso_noise_run / "summary.json"
        strong_lasso_summary = read_json(strong_lasso_summary_path)
    if time_step_run is not None:
        time_step_run = Path(time_step_run).resolve()
        if time_step_run != batch:
            time_manifest = read_json(time_step_run / "manifest.json")
            if time_manifest.get("task") != "qanr_redundant6_time_step_stochastic_p50n_R100" \
                    or time_manifest.get("status") != "completed":
                raise ValueError("stochastic time-step run is not a completed production output")
            time_summary = read_json(time_step_run / "summary.json")
        else:
            time_step_run = None

    def panel_points(panel, method):
        if panel == "observation_noise" and method == "strong_fd_lasso":
            if strong_lasso_summary is None:
                raise ValueError("strong FD--LASSO noise run was not supplied")
            return sorted(({
                "value": row["observation_noise"],
                "coefficient": {
                    "rmse": row["coefficient_error_relative"]["rmse"],
                    "ci95": row["coefficient_error_relative"]["rmse_bootstrap_95"],
                },
            } for row in strong_lasso_summary["groups"]), key=lambda row: row["value"])
        if panel == "time_step" and time_summary is not None:
            return sorted(
                ({"value": row["observation_interval"], "coefficient": row["coefficient"]}
                 for row in time_summary if row["method"] == method),
                key=lambda row: row["value"],
            )
        source = "full" if method == "strong_fd_ols" else str(ratio)
        return sorted(
            (row for row in summary_ad if row["panel"] == panel
             and row["source"] == source and row["method"] == method),
            key=lambda row: row["value"],
        )

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.35), constrained_layout=True)
    axes = axes.ravel()
    method_style = {
        "lasso": ("weak LASSO", ESTIMATE_COLOR, "^", "--"),
        "ols": ("weak OLS", TRUE_COLOR, "o", "-"),
        "strong_fd_ols": ("strong FD--OLS", TERTIARY_COLOR, "s", "-."),
        "strong_fd_lasso": ("strong FD--LASSO", TERTIARY_COLOR, "s", "-."),
    }

    def add_rmse(axis, panel, xlabel, title, methods, *, zero=False):
        for method in methods:
            points = panel_points(panel, method)
            x = np.asarray([row["value"] for row in points], dtype=float)
            y = np.asarray([row["coefficient"]["rmse"] for row in points], dtype=float)
            ci = np.asarray([row["coefficient"]["ci95"] for row in points], dtype=float)
            label, color, marker, linestyle = method_style[method]
            if zero and np.any(x == 0):
                positive = x > 0
                axis.axhline(y[~positive][0], color=color, linestyle=":", linewidth=.9)
                x, y, ci = x[positive], y[positive], ci[positive]
            axis.loglog(x, y, marker=marker, linestyle=linestyle, color=color, label=label)
            axis.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=.10, linewidth=0)
        axis.set(xlabel=xlabel, ylabel="coefficient RMSE", title=title)
        axis.legend(loc="best", fontsize=6.8)

    strong_noise_method = "strong_fd_lasso" if strong_lasso_summary is not None else "strong_fd_ols"
    add_rmse(axes[0], "observation_noise", r"observation noise $\sigma_{\rm obs}$",
             "(a) Observation noise", ("lasso", "ols", strong_noise_method), zero=True)
    add_rmse(axes[1], "process_noise", r"process noise $\sigma$",
             "(b) Process noise", ("lasso", "ols"))
    add_rmse(axes[2], "particle_number", r"particles $N$",
             "(c) Particle number", ("lasso", "ols"))
    axes[2].plot([], [], "--", color=REFERENCE_COLOR, label=r"$O(N^{-1/2})$")
    xref = np.asarray(axes[2].get_xlim())
    lasso_particle = sorted(
        (row for row in summary_ad if row["panel"] == "particle_number"
         and row["source"] == str(ratio) and row["method"] == "lasso"),
        key=lambda row: row["value"],
    )
    anchor = lasso_particle[1]
    axes[2].loglog(xref, anchor["coefficient"]["rmse"] * (xref / anchor["value"]) ** -.5,
                   "--", color=REFERENCE_COLOR)
    axes[2].legend(loc="best", fontsize=6.8)

    add_rmse(axes[3], "time_step", r"observation interval $\Delta t$",
             "(d) Observation interval", ("lasso", "ols"))
    lasso_dt = panel_points("time_step", "lasso")
    if time_summary is None:
        xref = np.asarray([lasso_dt[0]["value"], lasso_dt[-1]["value"]], dtype=float)
        axes[3].loglog(xref, lasso_dt[1]["coefficient"]["rmse"]
                       * (xref / lasso_dt[1]["value"]) ** .5,
                       "--", color=REFERENCE_COLOR, label=r"$O(\Delta t^{1/2})$")
        axes[3].legend(loc="best", fontsize=6.8)

    for method in ("lasso", "ols"):
        points = sorted(summary_e, key=lambda row: row["P"])
        x = np.asarray([row["P"] for row in points], dtype=float)
        y = np.sqrt(np.asarray([row[method]["mean"] for row in points], dtype=float))
        ci = np.sqrt(np.asarray([row[method]["ci95"] for row in points], dtype=float))
        label, color, marker, linestyle = method_style[method]
        axes[4].loglog(x, y, marker=marker, linestyle=linestyle, color=color, label=label)
        axes[4].fill_between(x, np.maximum(ci[:, 0], np.finfo(float).tiny), ci[:, 1],
                             color=color, alpha=.10, linewidth=0)
    lasso_e = sorted(summary_e, key=lambda row: row["P"])
    xref = np.asarray([lasso_e[0]["P"], lasso_e[-1]["P"]], dtype=float)
    anchor = np.sqrt(lasso_e[2]["lasso"]["mean"])
    axes[4].loglog(xref, anchor * np.sqrt(lasso_e[2]["P"] / xref), "--",
                   color=REFERENCE_COLOR, label=r"$O(P^{-1/2})$")
    axes[4].set(xlabel=r"sampled pairs $P$", ylabel="normalized RMS\ncoefficient difference",
                title="(e) Effect of pair sampling")
    axes[4].legend(loc="best", fontsize=6.8)

    for mode, label, color, marker, linestyle in (
        ("full", "all ordered pairs", TRUE_COLOR, "o", "-"),
        ("resample_each_time", rf"sampled, $P={ratio}N$", SECONDARY_COLOR, "s", "--"),
    ):
        points = sorted((row for row in summary_f if row["mode"] == mode), key=lambda row: row["N"])
        x = np.asarray([row["N"] for row in points], dtype=float)
        y = 1e3 * np.asarray([row["median_seconds"] for row in points], dtype=float)
        lo = 1e3 * np.asarray([row["iqr_seconds"][0] for row in points], dtype=float)
        hi = 1e3 * np.asarray([row["iqr_seconds"][1] for row in points], dtype=float)
        axes[5].loglog(x, y, marker=marker, linestyle=linestyle, color=color, label=label)
        axes[5].fill_between(x, lo, hi, color=color, alpha=.16, linewidth=0)
    axes[5].set(xlabel=r"particles $N$", ylabel="feature time [ms]", title="(f) Feature computation")
    axes[5].legend(loc="upper left", fontsize=6.8)

    for axis in axes:
        axis.set_box_aspect(PANEL_BOX_ASPECT)
        finish_axes(axis)
    destination.mkdir(parents=True, exist_ok=False)
    pdf = destination / "validation_sweeps_lasso_ols_overlay.pdf"
    png = destination / "validation_sweeps_lasso_ols_overlay.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    metadata = {
        "batch": str(batch),
        "pair_policy": f"P={ratio}N in panels (a)--(d) and (f); panel (e) varies P",
        "panel_e_quantity": "root mean square of ||theta_P - theta_all||_2 / ||theta_true||_2",
        "panel_e_reference_rate": "P^(-1/2)",
        "summary_a_to_d_sha256": digest(batch / "summary_a_to_d.json"),
        "summary_e_sha256": digest(batch / "summary_e.json"),
        "summary_f_sha256": digest(batch / "summary_f.json"),
        "time_step_run": None if time_step_run is None else str(time_step_run),
        "time_step_manifest_sha256": (
            None if time_step_run is None else digest(time_step_run / "manifest.json")
        ),
        "time_step_summary_sha256": (
            None if time_step_run is None else digest(time_step_run / "summary.json")
        ),
        "strong_lasso_noise_run": (
            None if strong_lasso_noise_run is None else str(strong_lasso_noise_run)
        ),
        "strong_lasso_noise_summary_sha256": (
            None if strong_lasso_summary_path is None else digest(strong_lasso_summary_path)
        ),
        "pdf_sha256": digest(pdf),
        "png_sha256": digest(png),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pdf)


def generate(
    batch, destination, time_step_run=None, runtime_csv=None,
    strong_lasso_noise_run=None,
):
    batch, destination = Path(batch).resolve(), Path(destination).resolve()
    if (batch / "summary_a_to_d.json").is_file():
        generate_p50(
            batch, destination, time_step_run=time_step_run,
            strong_lasso_noise_run=strong_lasso_noise_run,
        )
        return
    time_step_run = None if time_step_run is None else Path(time_step_run).resolve()
    runtime_csv = (DATA / "random_pair/runtime_scaling.csv" if runtime_csv is None
                   else Path(runtime_csv).resolve())
    report = read_json(batch / "report.json")
    if report.get("status") != "completed" or report.get("smoke_only"):
        raise ValueError("a completed non-smoke matched-LASSO batch is required")
    summary = read_json(batch / "summary.json")
    protocol = read_json(batch / "frozen_protocol.json")
    truth_norm = float(np.linalg.norm(np.asarray(protocol["true_theta"], dtype=float)))
    if truth_norm <= 0:
        raise ValueError("relative coefficient error requires a nonzero truth")
    if time_step_run is not None:
        time_manifest = read_json(time_step_run / "manifest.json")
        if time_manifest.get("task") not in {
                "qanr_redundant6_time_step_R100",
                "qanr_redundant6_time_step_paired_R100",
        } or time_manifest.get("smoke") or time_manifest.get("records") != 600:
            raise ValueError("panel (d) requires the completed 100-repetition time-step run")
        time_summary = read_json(time_step_run / "summary.json")
    else:
        time_manifest = None
        time_summary = None

    runtime = read_csv(runtime_csv)

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.35), constrained_layout=True)
    axes = axes.ravel()

    def method_ordered_legend(axis, **kwargs):
        handles, labels = axis.get_legend_handles_labels()
        rank = {"weak LASSO": 0, "weak OLS": 1, "strong FD--OLS": 2}
        indices = sorted(range(len(labels)), key=lambda index: rank.get(labels[index], 3 + index))
        axis.legend([handles[index] for index in indices],
                    [labels[index] for index in indices], **kwargs)

    noise = summary["observation_noise"]
    noise_levels = np.asarray([row["sigma_obs"] for row in noise])
    for key, label, color, marker, linestyle in (
        ("lasso_rmse_theta", "weak LASSO", ESTIMATE_COLOR, "^", "--"),
        ("ols_rmse_theta", "weak OLS", TRUE_COLOR, "o", "-"),
        ("euler_rmse_theta", "strong FD--OLS", TERTIARY_COLOR, "s", "-."),
    ):
        values = np.asarray([row[key]["rmse"] for row in noise]) / truth_norm
        lower = np.asarray([row[key]["ci95"][0] for row in noise]) / truth_norm
        upper = np.asarray([row[key]["ci95"][1] for row in noise]) / truth_norm
        positive = noise_levels > 0
        axes[0].loglog(noise_levels[positive], values[positive], marker=marker,
                       linestyle=linestyle, color=color, label=label)
        axes[0].fill_between(noise_levels[positive], lower[positive], upper[positive],
                             color=color, alpha=.10, linewidth=0)
        axes[0].axhline(values[~positive][0], color=color, linestyle=":", linewidth=.9)
    swept = noise_levels[noise_levels > 0]
    axes[0].set_xlim(swept.min() / 1.6, swept.max() * 1.6)
    axes[0].set(xlabel=r"observation noise $\sigma_{\rm obs}$",
                ylabel="coefficient RMSE",
                title="(a) Observation noise")
    method_ordered_legend(axes[0], loc="upper left", fontsize=6.6)

    process = summary["process_noise"]
    sigma = np.asarray([row["value"] for row in process])
    ols_sigma = np.asarray([row["ols_rmse_theta"]["rmse"] for row in process]) / truth_norm
    lasso_sigma = np.asarray([row["lasso_rmse_theta"]["rmse"] for row in process]) / truth_norm
    for key, label, color, marker, linestyle in (
        ("lasso_rmse_theta", "weak LASSO", ESTIMATE_COLOR, "^", "--"),
        ("ols_rmse_theta", "weak OLS", TRUE_COLOR, "o", "-"),
    ):
        values = np.asarray([row[key]["rmse"] for row in process]) / truth_norm
        lower = np.asarray([row[key]["ci95"][0] for row in process]) / truth_norm
        upper = np.asarray([row[key]["ci95"][1] for row in process]) / truth_norm
        axes[1].loglog(sigma, values, marker=marker, linestyle=linestyle,
                       color=color, label=label)
        axes[1].fill_between(sigma, lower, upper, color=color, alpha=.10, linewidth=0)
    reference = np.asarray([sigma.min(), sigma.max()])
    axes[1].loglog(reference, ols_sigma[1] * reference / sigma[1], "--",
                   color=REFERENCE_COLOR, label=r"$O(\sigma)$")
    axes[1].set(xlabel=r"process noise $\sigma$", ylabel="coefficient RMSE",
                title="(b) Process noise")
    method_ordered_legend(axes[1], loc="upper left", fontsize=6.9)

    particle_rows = summary["particle_number"]
    particles = np.asarray([int(row["value"]) for row in particle_rows])
    ols_particles = np.asarray([row["ols_rmse_theta"]["rmse"] for row in particle_rows]) / truth_norm
    for key, label, color, marker, linestyle in (
        ("lasso_rmse_theta", "weak LASSO", ESTIMATE_COLOR, "^", "--"),
        ("ols_rmse_theta", "weak OLS", TRUE_COLOR, "o", "-"),
    ):
        values = np.asarray([row[key]["rmse"] for row in particle_rows]) / truth_norm
        lower = np.asarray([row[key]["ci95"][0] for row in particle_rows]) / truth_norm
        upper = np.asarray([row[key]["ci95"][1] for row in particle_rows]) / truth_norm
        axes[2].loglog(particles, values, marker=marker, linestyle=linestyle,
                       color=color, label=label)
        axes[2].fill_between(particles, lower, upper, color=color, alpha=.10, linewidth=0)
    reference = np.asarray([particles.min(), particles.max()], dtype=float)
    axes[2].loglog(reference, ols_particles[1] * (100 / reference) ** .5, "--",
                   color=REFERENCE_COLOR, label=r"$O(N^{-1/2})$")
    axes[2].set(xlabel=r"particles $N$", ylabel="coefficient RMSE",
                title="(c) Particle number")
    axes[2].set_xticks(particles, [str(value) for value in particles])
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    method_ordered_legend(axes[2], loc="upper right", fontsize=6.9)

    if time_summary is None:
        dt_rows = summary["time_step"]
        dt_values = np.asarray([row["dt"] for row in dt_rows])
        ols_dt = np.asarray([row["ols_error_theta"] for row in dt_rows]) / truth_norm
        lasso_dt_error = np.asarray([row["lasso_error_theta"] for row in dt_rows]) / truth_norm
        axes[3].loglog(dt_values, lasso_dt_error, "^--", color=ESTIMATE_COLOR, label="weak LASSO")
        axes[3].loglog(dt_values, ols_dt, "o-", color=TRUE_COLOR, label="weak OLS")
    else:
        dt_rows = time_summary["time_step"]
        dt_values = np.asarray([row["dt"] for row in dt_rows])
        ols_dt = np.asarray([row["ols_relative_rmse"]["rmse"] for row in dt_rows])
        lasso_dt_error = np.asarray([row["lasso_relative_rmse"]["rmse"] for row in dt_rows])
        for key, values, label, color, marker, style in (
            ("lasso_relative_rmse", lasso_dt_error, "weak LASSO", ESTIMATE_COLOR, "^", "--"),
            ("ols_relative_rmse", ols_dt, "weak OLS", TRUE_COLOR, "o", "-"),
        ):
            ci = np.asarray([row[key]["ci95"] for row in dt_rows])
            axes[3].loglog(dt_values, values, marker=marker, linestyle=style,
                           color=color, label=label)
            axes[3].fill_between(dt_values, ci[:, 0], ci[:, 1],
                                 color=color, alpha=.10, linewidth=0)
    reference = np.asarray([dt_values.min(), dt_values.max()])
    anchor = int(np.argmin(np.abs(dt_values - .05)))
    axes[3].loglog(reference, ols_dt[anchor] * reference / dt_values[anchor], "--",
                   color=REFERENCE_COLOR, label=r"$O(\Delta t)$")
    axes[3].set(xlabel=r"observation interval $\Delta t$", ylabel="coefficient RMSE",
                title="(d) Observation interval")
    method_ordered_legend(axes[3], loc="upper left", fontsize=6.9)

    pair = summary["pair_count"]
    p = np.asarray([row["P"] for row in pair], dtype=float)
    truth_norm_sq = truth_norm ** 2
    ols_means = np.sqrt(np.asarray([row["ols_theta_to_full_sq"]["mean"] for row in pair]) / truth_norm_sq)
    ols_lower = np.sqrt(np.asarray([row["ols_theta_to_full_sq"]["ci95"][0] for row in pair]) / truth_norm_sq)
    ols_upper = np.sqrt(np.asarray([row["ols_theta_to_full_sq"]["ci95"][1] for row in pair]) / truth_norm_sq)
    lasso_means = np.sqrt(np.asarray([row["lasso_theta_to_full_sq"]["mean"] for row in pair]) / truth_norm_sq)
    lasso_lower = np.sqrt(np.asarray([row["lasso_theta_to_full_sq"]["ci95"][0] for row in pair]) / truth_norm_sq)
    lasso_upper = np.sqrt(np.asarray([row["lasso_theta_to_full_sq"]["ci95"][1] for row in pair]) / truth_norm_sq)
    axes[4].loglog(p, lasso_means, "^--", color=ESTIMATE_COLOR, label="weak LASSO")
    axes[4].fill_between(p, np.maximum(lasso_lower, np.finfo(float).tiny), lasso_upper,
                         color=ESTIMATE_COLOR, alpha=.14, linewidth=0)
    axes[4].loglog(p, ols_means, "o-", color=TRUE_COLOR, label="weak OLS")
    axes[4].fill_between(p, np.maximum(ols_lower, np.finfo(float).tiny), ols_upper,
                         color=TRUE_COLOR, alpha=.14, linewidth=0)
    anchor = ols_means[2]
    axes[4].loglog(p, anchor * np.sqrt(1000 / p), "--", color=REFERENCE_COLOR,
                   label=r"$P^{-1/2}$")
    axes[4].set(xlabel=r"sampled pairs $P$", ylabel="normalized RMS\ncoefficient difference",
                title="(e) Effect of pair sampling")
    method_ordered_legend(axes[4], loc="best", fontsize=6.9)

    ns = sorted({int(row["N"]) for row in runtime})
    for mode, label, color, marker, linestyle in (
        ("full", "all ordered pairs", TRUE_COLOR, "o", "-"),
        ("resample_each_time", r"sampled, $P=10N$", SECONDARY_COLOR, "s", "--"),
    ):
        values = [[float(row["design_seconds"]) for row in runtime
                   if row["mode"] == mode and int(row["N"]) == n] for n in ns]
        median = np.asarray([np.median(item) for item in values])
        lower = np.asarray([np.quantile(item, .25) for item in values])
        upper = np.asarray([np.quantile(item, .75) for item in values])
        axes[5].loglog(ns, median, marker=marker, linestyle=linestyle,
                       color=color, label=label)
        axes[5].fill_between(ns, lower, upper, color=color, alpha=.16, linewidth=0)
    axes[5].set(xlabel=r"particles $N$", ylabel="feature time [s]", title="(f) Feature computation")
    axes[5].legend(loc="upper left", fontsize=6.9)

    for ax in axes:
        ax.set_box_aspect(PANEL_BOX_ASPECT)
        finish_axes(ax)
    destination.mkdir(parents=True, exist_ok=False)
    pdf = destination / "validation_sweeps_lasso_ols_overlay.pdf"
    png = destination / "validation_sweeps_lasso_ols_overlay.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    shutil.copy2(Path(__file__), destination / "generate_figure.py")
    metadata = {
        "matched_lasso_batch": str(batch),
        "matched_lasso_summary_sha256": digest(batch / "summary.json"),
        "matched_lasso_verification_sha256": digest(batch / "verification.json"),
        "expanded_trajectory_sources": {
            str(path.relative_to(batch)): digest(path)
            for path in (
                batch / "observation_lasso.csv",
                batch / "stochastic_lasso.csv",
                batch / "time_step_lasso.csv",
                batch / "pair_lasso.csv",
            )
        },
        "runtime_source": {
            "path": str(runtime_csv),
            "sha256": digest(runtime_csv),
        },
        "truth_coefficient_norm": truth_norm,
        "coefficient_metric": "||theta_hat-theta_star||_2/||theta_star||_2",
        "pair_metric": "RMS of ||theta_P-theta_full||_2/||theta_star||_2 across repetitions",
        "time_step_run": None if time_step_run is None else str(time_step_run),
        "time_step_manifest_sha256": (
            None if time_step_run is None else digest(time_step_run / "manifest.json")
        ),
        "time_step_records_sha256": (
            None if time_step_run is None else digest(time_step_run / "records.json")
        ),
        "trajectory_repetitions_in_panels_a_b_c_d": (
            {"a": 100, "b": 100, "c": 100, "d": 1}
            if time_step_run is None else {"a": 100, "b": 100, "c": 100, "d": 100}
        ),
        "trajectory_uncertainty": "95% delta-method intervals for relative RMSE",
        "panels_with_lasso_overlay": ["a", "b", "c", "d", "e"],
        "panel_f_data_unchanged": True,
        "panel_f_labels_updated": True,
        "pdf_sha256": digest(pdf), "png_sha256": digest(png),
        "manuscript_target": "latex/L4DC/figures/validation_sweeps.pdf",
        "manuscript_edited": True,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pdf)


def main():
    apply_manuscript_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--time-step-run", type=Path)
    parser.add_argument("--runtime-csv", type=Path)
    parser.add_argument("--strong-lasso-noise-run", type=Path)
    args = parser.parse_args()
    generate(
        args.batch,
        args.destination,
        time_step_run=args.time_step_run,
        runtime_csv=args.runtime_csv,
        strong_lasso_noise_run=args.strong_lasso_noise_run,
    )


if __name__ == "__main__":
    main()
