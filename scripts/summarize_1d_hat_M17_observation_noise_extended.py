"""Aggregate and plot 100 repetitions of the M=17 sweep over sigma_obs=0,...,1."""

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_1d_hat_experiment import write_json
from scripts.run_1d_hat_M17_observation_noise_extended import digest, job_plan, read_json
from scripts.summarize_1d_hat_M17_R100 import group_summary, quantitative_log_ticks

METRICS = ["force_error_weighted_rel", "trajectory_error_rel", "drift_error_rel", "trajectory_rmse"]
TITLES = {"force_error_weighted_rel": "Interaction-force relative error",
          "trajectory_error_rel": "Coupled trajectory relative error",
          "drift_error_rel": "Drift relative error", "trajectory_rmse": "Coupled trajectory RMSE"}


def source_models(total_batch, case_id):
    frozen, status = read_json(total_batch / "frozen_protocol.json"), read_json(total_batch / "batch_status.json")
    models, inputs = [], []
    for index in range(20):
        record = frozen["imported_existing"][f"{case_id}-{index}"]
        path = total_batch / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("source imported record changed")
        content = read_json(path)
        models.append(content["result"])
        inputs.append({"path": str(path.resolve()), "sha256": record["sha256"]})
    for index in range(80):
        record = status["results"][f"{case_id}-{index}"]
        path = total_batch / record["directory"] / "report.json"
        if digest(path) != record["report_sha256"]:
            raise ValueError("source new report changed")
        report = read_json(path)
        models.append(report.get("results", {}).get("hat_M17", {}))
        inputs.append({"path": str(path.resolve()), "sha256": record["report_sha256"]})
    return models, inputs


def plot_panel(ax, points, metric, letter=None):
    x = np.asarray([point["value"] for point in points], dtype=float)
    stats = [point["total100"]["metrics"][metric] for point in points]
    mean = np.asarray([np.nan if row["mean"] is None else row["mean"] for row in stats])
    lower = np.asarray([np.nan if row["mean_ci95"] is None else row["mean_ci95"][0] for row in stats])
    upper = np.asarray([np.nan if row["mean_ci95"] is None else row["mean_ci95"][1] for row in stats])
    ax.plot(x, mean, "o-", color="tab:orange", linewidth=1.8, markersize=5, label="M=17")
    ax.fill_between(x, lower, upper, color="tab:orange", alpha=.18, linewidth=0)
    ax.set_xscale("symlog", linthresh=.0003)
    if np.all(mean > 0) and np.all(lower[np.isfinite(lower)] > 0):
        ax.set_yscale("log")
        quantitative_log_ticks(ax, np.r_[mean, lower, upper])
    ax.set_xticks(x, [f"{value:g}" for value in x], rotation=55, ha="right")
    ax.tick_params(axis="x", labelsize=8)
    ax.minorticks_off()
    title = r"Observation noise $\sigma_{\rm obs}$"
    ax.set(xlabel=title, ylabel=TITLES[metric], title=(f"({letter}) " if letter else "") + title)
    ax.set_box_aspect(3 / 4)
    ax.grid(alpha=.18)


def make_plots(summary, destination):
    points = summary["observation_noise"]
    for metric in METRICS[:3]:
        fig, ax = plt.subplots(figsize=(5.2, 4.55), constrained_layout=True)
        plot_panel(ax, points, metric)
        ax.legend(loc="best")
        fig.supxlabel("Arithmetic means and pointwise percentile-bootstrap 95% CIs; 100 repetitions per value.", fontsize=8)
        fig.savefig(destination / f"m17_observation_noise_extended_{metric}.png", dpi=180)
        fig.savefig(destination / f"m17_observation_noise_extended_{metric}.pdf")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.75), constrained_layout=True)
    for ax, metric, letter in zip(axes, METRICS[:3], "abc"):
        plot_panel(ax, points, metric, letter)
    axes[0].legend(loc="best")
    fig.supxlabel("M=17; N=1000, T=2, dt=0.005, sigma=0.1. Arithmetic means and pointwise percentile-bootstrap 95% CIs; 100 repetitions.", fontsize=9)
    fig.savefig(destination / "m17_observation_noise_extended_all_metrics.png", dpi=180)
    fig.savefig(destination / "m17_observation_noise_extended_all_metrics.pdf")
    plt.close(fig)


def summarize(batch, destination):
    batch, destination = Path(batch).resolve(), Path(destination).resolve()
    frozen, status = read_json(batch / "frozen_protocol.json"), read_json(batch / "batch_status.json")
    spec, base = frozen["requested_protocol"], frozen["base"]
    if status["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("batch must finish before aggregation")
    if digest(batch / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for name, expected in frozen["source_sha256"].items():
        if digest(batch / "source_snapshot" / name) != expected:
            raise ValueError("source snapshot changed")
    planned = job_plan(spec, base)
    if planned != status["jobs"] or set(planned) != set(status["results"]):
        raise ValueError("added jobs are missing or changed")
    reports, added_inputs = {}, {}
    for key, record in status["results"].items():
        path = batch / record["directory"] / "report.json"
        if digest(path) != record["report_sha256"]:
            raise ValueError("added report changed")
        report = read_json(path)
        if report["job"] != planned[key] or report["status"] != record["status"]:
            raise ValueError("added report identity/status mismatch")
        for filename, expected in report["artifacts_sha256"].items():
            if digest(path.parent / filename) != expected:
                raise ValueError("added artifact changed")
        reports[key] = report
        added_inputs[key] = {"path": str(path.relative_to(batch)), "sha256": record["report_sha256"]}
    indices = np.random.default_rng(spec["bootstrap"]["seed"]).integers(0, 100, size=(spec["bootstrap"]["samples"], 100))
    total_batch = ROOT / spec["source_total100_batch"]
    source_frozen, source_status = read_json(total_batch / "frozen_protocol.json"), read_json(total_batch / "batch_status.json")
    if digest(total_batch / "frozen_protocol.json") != source_status["frozen_protocol_sha256"]:
        raise ValueError("source total100 frozen protocol changed")
    points, source_inputs = [], {}
    for value_index, value in enumerate(spec["full_values"]):
        case_id = "base" if value == 0 else f"sigma_obs-{value_index:02d}"
        if value in spec["existing_values"]:
            models, inputs = source_models(total_batch, case_id)
            source_inputs[case_id] = inputs
        else:
            models = [reports[f"{case_id}-{index}"].get("results", {}).get("hat_M17", {}) for index in range(100)]
        points.append({"value": value, "case": case_id, "source": "completed_M17_R100" if value in spec["existing_values"] else "added_extension",
                       "total100": group_summary(models, indices)})
    direction = {}
    for metric in METRICS[:3]:
        values = np.asarray([point["total100"]["metrics"][metric]["mean"] for point in points], dtype=float)
        direction[metric] = {"values": values.tolist(), "adjacent_non_decreasing": int(np.sum(np.diff(values) >= 0)),
                             "adjacent_total": len(values) - 1,
                             "fully_monotone": bool(np.all(np.diff(values) >= 0)),
                             "spearman": float(spearmanr(np.arange(len(values)), values).statistic)}
    summary = {"dictionary_size": 17, "repetitions": 100, "baseline": base["simulation"],
               "observation_noise": points, "increasing_direction": direction,
               "failures": [key for key, report in reports.items() if report["status"] != "completed"],
               "bootstrap": spec["bootstrap"], "bootstrap_indices_sha256": None,
               "inputs": {"source_low_values": source_inputs, "added_values": added_inputs},
               "frozen_protocol_sha256": status["frozen_protocol_sha256"]}
    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination / "bootstrap_indices.npy", indices)
    summary["bootstrap_indices_sha256"] = digest(destination / "bootstrap_indices.npy")
    write_json(destination / "summary.json", summary)
    shutil.copy2(Path(__file__), destination / "plot_script.py")
    make_plots(summary, destination)
    write_json(destination / "provenance.json", {"summary_sha256": digest(destination / "summary.json"),
        "plot_source_sha256": digest(Path(__file__)), "batch_directory": str(batch), "numerical_refitting_performed": True})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    summarize(args.batch, args.destination)
