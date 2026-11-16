"""Aggregate the 20 existing and 80 new M=17 repetitions and plot the sweeps for a single dictionary."""

import argparse
import json
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
from scripts.run_1d_hat_M17_R100 import digest, job_plan, read_json
from scripts.run_1d_hat_sweeps import cases
from scripts.summarize_1d_hat_repetitions import statistics

METRICS = ["force_error_weighted_rel", "trajectory_error_rel", "drift_error_rel", "trajectory_rmse"]
TITLES = {"force_error_weighted_rel": "Interaction-force relative error",
          "trajectory_error_rel": "Coupled trajectory relative error",
          "drift_error_rel": "Drift relative error", "trajectory_rmse": "Coupled trajectory RMSE"}
LABELS = {"sigma_obs": r"Observation noise $\sigma_{\rm obs}$", "sigma": r"Process noise $\sigma$",
          "N": r"Particles $N$", "dt": r"Time step $\Delta t$"}
DIRECTIONS = {"sigma_obs": 1, "sigma": 1, "N": -1, "dt": 1}


def extended_statistics(values, indices):
    result = statistics(values, indices)
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    result["minimum"] = float(finite.min()) if len(finite) else None
    result["maximum"] = float(finite.max()) if len(finite) else None
    return result


def diagnostics(models):
    return {
        "models": len(models),
        "missing_models": sum(not model for model in models),
        "unconverged_candidates": sum(sum(not row["converged"] for row in model.get("selection", {}).get("path", [])) for model in models),
        "candidate_solves": sum(len(model.get("selection", {}).get("path", [])) for model in models),
        "boundary_selections": sum(model.get("selection", {}).get("selected_at_grid_boundary", False) for model in models),
        "divergent_rollouts": sum(model.get("metrics", {}).get("trajectory_status") == "diverged" for model in models),
        "basis_domain_excursions": sum(model.get("metrics", {}).get("trajectory_exits_basis_interval", False) for model in models),
    }


def group_summary(models, indices):
    return {"metrics": {metric: extended_statistics([model.get("metrics", {}).get(metric) for model in models], indices)
                        for metric in METRICS}, "diagnostics": diagnostics(models)}


def expected_direction(summary, group):
    result = {}
    for metric in METRICS[:3]:
        total = matches = complete = 0
        correlations = []
        per_axis = {}
        for axis, direction in DIRECTIONS.items():
            values = np.asarray([point[group]["metrics"][metric]["mean"] for point in summary["accuracy"][axis]], dtype=float)
            finite = bool(np.all(np.isfinite(values)))
            differences = direction * np.diff(values) if finite else np.array([])
            good = int(np.sum(differences >= 0)) if finite else 0
            rho = float(direction * spearmanr(np.arange(len(values)), values).statistic) if finite else None
            per_axis[axis] = {"values": values.tolist(), "adjacent_expected_direction": good,
                              "adjacent_total": len(values) - 1, "fully_monotone": finite and good == len(values) - 1,
                              "aligned_spearman": rho}
            if finite:
                matches += good
                total += len(values) - 1
                complete += int(good == len(values) - 1)
                correlations.append(rho)
        result[metric] = {"per_axis": per_axis, "adjacent_expected_direction": matches,
                          "adjacent_total": total, "fully_monotone_sweeps": complete,
                          "aligned_spearman_mean": float(np.mean(correlations)) if correlations else None}
    return result


def quantitative_log_ticks(ax, values, count=4):
    """Always place numerically labeled ticks within the displayed range, including on logarithmic axes."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return
    lower, upper = float(values.min()), float(values.max())
    if np.isclose(lower, upper):
        lower, upper = lower / 1.1, upper * 1.1
    ticks = np.geomspace(lower, upper, count)
    labels = []
    for value in ticks:
        if value < 1e-3 or value >= 1e3:
            labels.append(f"{value:.1e}")
        elif value < .1:
            labels.append(f"{value:.3f}".rstrip("0").rstrip("."))
        else:
            labels.append(f"{value:.2f}".rstrip("0").rstrip("."))
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", which="major", labelleft=True)


def panel(ax, points, metric, group, axis, letter):
    x = np.asarray([point["value"] for point in points], dtype=float)
    stats = [point[group]["metrics"][metric] for point in points]
    mean = np.asarray([np.nan if row["mean"] is None else row["mean"] for row in stats])
    lower = np.asarray([np.nan if row["mean_ci95"] is None else row["mean_ci95"][0] for row in stats])
    upper = np.asarray([np.nan if row["mean_ci95"] is None else row["mean_ci95"][1] for row in stats])
    ax.plot(x, mean, "o-", color="tab:orange", linewidth=1.6, markersize=5, label="M=17")
    ax.fill_between(x, lower, upper, color="tab:orange", alpha=.18, linewidth=0)
    if np.any(x == 0):
        positive = x[x > 0]
        ax.set_xscale("symlog", linthresh=float(positive.min()))
    else:
        ax.set_xscale("log")
    if np.all(mean > 0) and np.all(lower[np.isfinite(lower)] > 0):
        ax.set_yscale("log")
        quantitative_log_ticks(ax, np.r_[mean, lower, upper])
    ax.set_xticks(x, [f"{value:g}" for value in x], rotation=25)
    ax.minorticks_off()
    ax.set(xlabel=LABELS[axis], ylabel=TITLES[metric], title=f"({letter}) {LABELS[axis]}")
    ax.set_box_aspect(3 / 4)
    ax.grid(alpha=.18)


def plot(summary, destination, group, suffix, repetitions):
    base = summary["baseline"]
    context = ("SMOKE ONLY" if summary["smoke_only"] else
               f"M=17; N={base['N']}, T={base['horizon']:g}, dt={base['dt']:g}, sigma={base['sigma']:g}, sigma_obs={base['sigma_obs']:g}; only the swept parameter varies.")
    for metric in METRICS[:3]:
        fig, axes = plt.subplots(2, 2, figsize=(9.8, 8.3), constrained_layout=True)
        for ax, axis, letter in zip(axes.ravel(), ("sigma_obs", "sigma", "N", "dt"), "abcd"):
            panel(ax, summary["accuracy"][axis], metric, group, axis, letter)
        axes[0, 0].legend(loc="best")
        fig.supxlabel(f"{context}\nArithmetic means and pointwise percentile-bootstrap 95% CIs; {repetitions} independent repetitions per condition.", fontsize=9)
        fig.savefig(destination / f"m17_{metric}_{suffix}.png", dpi=180)
        fig.savefig(destination / f"m17_{metric}_{suffix}.pdf")
        plt.close(fig)


def summarize(batch, destination):
    batch, destination = Path(batch).resolve(), Path(destination).resolve()
    frozen, status = read_json(batch / "frozen_protocol.json"), read_json(batch / "batch_status.json")
    if status["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("the predeclared new batch must finish before aggregation")
    if digest(batch / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for name, expected in frozen["source_sha256"].items():
        if digest(batch / "source_snapshot" / name) != expected:
            raise ValueError("source snapshot changed")
    for name, expected in frozen["source_provenance_sha256"].items():
        if digest(batch / "source_provenance" / name) != expected:
            raise ValueError("source provenance changed")
    expected_jobs = job_plan(frozen["protocol"], frozen["base"])
    if status["jobs"] != expected_jobs or set(status["results"]) != set(expected_jobs):
        raise ValueError("new jobs are missing or changed")
    new_reports = {}
    inputs = {"new": {}, "existing": frozen["imported_existing"]}
    for key, record in status["results"].items():
        path = batch / record["directory"] / "report.json"
        if digest(path) != record["report_sha256"]:
            raise ValueError("new report changed")
        report = read_json(path)
        if report["job"] != expected_jobs[key] or report["status"] != record["status"]:
            raise ValueError("new job identity/status mismatch")
        for filename, expected in report["artifacts_sha256"].items():
            if digest(path.parent / filename) != expected:
                raise ValueError("new artifact changed")
        new_reports[key] = report
        inputs["new"][key] = {"path": str(path.relative_to(batch)), "sha256": record["report_sha256"]}
    smoke = frozen["smoke_only"]
    existing_count = 0 if smoke else frozen["requested_protocol"]["existing_repetitions"]
    new_count = frozen["base"]["replication"]["count"]
    total_count = existing_count + new_count
    boot = frozen["requested_protocol"]["bootstrap"]
    rng_total = np.random.default_rng(boot["total_seed"]).integers(0, total_count, size=(boot["samples"], total_count))
    rng_new = np.random.default_rng(boot["new_only_seed"]).integers(0, new_count, size=(boot["samples"], new_count))
    rng_existing = (np.random.default_rng(boot["existing_only_seed"]).integers(0, existing_count, size=(boot["samples"], existing_count))
                    if existing_count else np.empty((boot["samples"], 0), dtype=int))
    summary = {"smoke_only": smoke, "dictionary_size": 17, "existing_repetitions": existing_count,
               "new_repetitions": new_count, "total_repetitions": total_count,
               "baseline": frozen["base"]["simulation"], "accuracy": {}, "inputs": inputs,
               "bootstrap": boot, "frozen_protocol_sha256": status["frozen_protocol_sha256"],
               "failures": [key for key, report in new_reports.items() if report["status"] != "completed"]}
    source_cases = cases(frozen["protocol"], frozen["base"])
    for case in source_cases:
        case_id = case["id"]
        existing_models = []
        for index in range(existing_count):
            record = frozen["imported_existing"][f"{case_id}-{index}"]
            path = batch / record["path"]
            if digest(path) != record["sha256"]:
                raise ValueError("imported existing record changed")
            imported = read_json(path)
            if imported["case"] != case_id or imported["index"] != index:
                raise ValueError("imported record identity mismatch")
            existing_models.append(imported["result"])
        new_models = [new_reports[f"{case_id}-{index}"].get("results", {}).get("hat_M17", {}) for index in range(new_count)]
        point = {"value": case["value"], "case": case_id,
                 "existing20": group_summary(existing_models, rng_existing) if existing_count else None,
                 "new80": group_summary(new_models, rng_new),
                 "total100": group_summary(existing_models + new_models, rng_total)}
        summary["accuracy"].setdefault(case["parameter"], []).append(point)
    summary["expected_direction"] = {
        "new80": expected_direction(summary, "new80"),
        "total100": expected_direction(summary, "total100"),
    }
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "summary.json", summary)
    np.savez(destination / "bootstrap_indices.npz", total=rng_total, new_only=rng_new, existing_only=rng_existing)
    shutil.copy2(Path(__file__), destination / "plot_script.py")
    write_json(destination / "provenance.json", {"summary_sha256": digest(destination / "summary.json"),
        "plot_source_sha256": digest(Path(__file__)), "batch_directory": str(batch), "numerical_refitting_performed": False})
    plot(summary, destination, "total100", "total100", total_count)
    plot(summary, destination, "new80", "new80", new_count)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    summarize(args.batch, args.destination)
