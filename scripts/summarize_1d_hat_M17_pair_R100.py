"""Aggregate the 20 existing and 80 new repetitions of the M=17 pair-count sweep."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_1d_hat_experiment import write_json
from scripts.run_1d_hat_M17_pair_R100 import digest, job_plan, read_json
from scripts.summarize_1d_hat_repetitions import statistics

METRICS = [
    "force_error_weighted_rel", "trajectory_error_rel", "drift_error_rel", "trajectory_rmse",
    "coefficient_difference_sq", "force_difference_sq_rel", "design_difference_sq",
]
TITLES = {
    "force_error_weighted_rel": "Interaction-force relative error",
    "trajectory_error_rel": "Coupled trajectory relative error",
    "drift_error_rel": "Drift relative error",
    "trajectory_rmse": "Coupled trajectory RMSE",
    "coefficient_difference_sq": "Squared coefficient difference",
    "force_difference_sq_rel": "Relative squared force difference",
    "design_difference_sq": "Squared design difference",
}


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
        "nonfinite_endpoints": sum(model.get("theta") is None for model in models),
        "unconverged_endpoints": sum(not model.get("endpoint", {}).get("converged", False) for model in models),
        "candidate_solves": sum(len(model.get("path", [])) for model in models),
        "unconverged_candidates": sum(sum(not row.get("converged", False) for row in model.get("path", []))
                                          for model in models),
        "divergent_rollouts": sum(model.get("metrics", {}).get("trajectory_status") == "diverged" for model in models),
        "basis_domain_excursions": sum(model.get("metrics", {}).get("trajectory_exits_basis_interval", False)
                                         for model in models),
    }


def group_summary(models, indices):
    return {
        "metrics": {metric: extended_statistics([model.get("metrics", {}).get(metric) for model in models], indices)
                    for metric in METRICS},
        "diagnostics": diagnostics(models),
    }


def quantitative_log_ticks(ax, values, count=4):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return
    lower, upper = float(values.min()), float(values.max())
    if np.isclose(lower, upper):
        lower, upper = lower / 1.1, upper * 1.1
    ticks = np.geomspace(lower, upper, count)
    labels = [f"{value:.1e}" if value < 1e-3 or value >= 1e3 else
              (f"{value:.3f}".rstrip("0").rstrip(".") if value < .1 else f"{value:.2f}".rstrip("0").rstrip("."))
              for value in ticks]
    ax.set_yticks(ticks, labels)


def plot(summary, destination, group, suffix, repetitions):
    for metric in ("force_error_weighted_rel", "force_difference_sq_rel"):
        x = np.asarray([point["value"] for point in summary["pairs"]], dtype=float)
        rows = [point[group]["metrics"][metric] for point in summary["pairs"]]
        mean = np.asarray([row["mean"] for row in rows], dtype=float)
        lower = np.asarray([row["mean_ci95"][0] for row in rows], dtype=float)
        upper = np.asarray([row["mean_ci95"][1] for row in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
        ax.plot(x, mean, "o-", color="tab:orange", linewidth=1.7)
        ax.fill_between(x, lower, upper, color="tab:orange", alpha=.18, linewidth=0)
        ax.set(xscale="log", yscale="log", xlabel=r"Sampled pairs $P$", ylabel=TITLES[metric])
        ax.set_xticks(x, [f"{value:g}" for value in x], rotation=25)
        quantitative_log_ticks(ax, np.r_[mean, lower, upper])
        ax.minorticks_off()
        ax.grid(alpha=.18)
        ax.set_box_aspect(3 / 4)
        fig.supxlabel(f"M=17; arithmetic means and pointwise bootstrap 95% CIs; {repetitions} repetitions per P.", fontsize=8)
        fig.savefig(destination / f"m17_pair_{metric}_{suffix}.png", dpi=180)
        fig.savefig(destination / f"m17_pair_{metric}_{suffix}.pdf")
        plt.close(fig)


def summarize(batch, destination):
    batch, destination = Path(batch).resolve(), Path(destination).resolve()
    frozen = read_json(batch / "frozen_protocol.json")
    status = read_json(batch / "batch_status.json")
    if status["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("pair batch must finish before aggregation")
    if digest(batch / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for name, expected in frozen["source_sha256"].items():
        if digest(batch / "source_snapshot" / name) != expected:
            raise ValueError("source snapshot changed")
    for name, expected in frozen["source_provenance_sha256"].items():
        if digest(batch / "source_provenance" / name) != expected:
            raise ValueError("source provenance changed")
    expected_jobs = job_plan(frozen["requested_protocol"], frozen["smoke_only"])
    if status["jobs"] != expected_jobs or set(status["results"]) != set(expected_jobs):
        raise ValueError("new pair jobs are missing or changed")
    reports = {}
    inputs = {"new": {}, "existing": frozen["imported_existing"]}
    for key, record in status["results"].items():
        path = batch / record["directory"] / "report.json"
        if digest(path) != record["report_sha256"]:
            raise ValueError("new pair report changed")
        report = read_json(path)
        if report["job"] != expected_jobs[key] or report["status"] != record["status"]:
            raise ValueError("new pair identity/status mismatch")
        for name, expected in report["artifacts_sha256"].items():
            if digest(path.parent / name) != expected:
                raise ValueError("new pair artifact changed")
        reports[key] = report
        inputs["new"][key] = {"path": str(path.relative_to(batch)), "sha256": record["report_sha256"]}
    smoke = frozen["smoke_only"]
    existing_count = 0 if smoke else frozen["requested_protocol"]["existing_repetitions"]
    new_count = frozen["protocol"]["new_repetitions"]
    total_count = existing_count + new_count
    boot = frozen["requested_protocol"]["bootstrap"]
    rng_total = np.random.default_rng(boot["total_seed"]).integers(0, total_count, size=(boot["samples"], total_count))
    rng_new = np.random.default_rng(boot["new_only_seed"]).integers(0, new_count, size=(boot["samples"], new_count))
    rng_existing = (np.random.default_rng(boot["existing_only_seed"]).integers(0, existing_count,
                     size=(boot["samples"], existing_count)) if existing_count else np.empty((boot["samples"], 0), dtype=int))
    summary = {
        "smoke_only": smoke, "dictionary_size": 17,
        "existing_repetitions": existing_count, "new_repetitions": new_count, "total_repetitions": total_count,
        "baseline": frozen["base"]["simulation"], "pairs": [], "inputs": inputs,
        "bootstrap": boot, "frozen_protocol_sha256": status["frozen_protocol_sha256"],
        "failures": [key for key, report in reports.items() if report["status"] != "completed"],
    }
    for pair_index, P in enumerate(frozen["protocol"]["pair_counts"]):
        existing_models = []
        for index in range(existing_count):
            record = frozen["imported_existing"][f"pairs-{pair_index}-{index}"]
            path = batch / record["path"]
            if digest(path) != record["sha256"]:
                raise ValueError("imported pair record changed")
            imported = read_json(path)
            if imported["pair_index"] != pair_index or imported["P"] != P or imported["index"] != index:
                raise ValueError("imported pair identity mismatch")
            existing_models.append(imported["result"])
        new_models = [reports[f"pairs-{pair_index}-{index}"].get("results", {}).get("hat_M17", {})
                      for index in range(new_count)]
        summary["pairs"].append({
            "value": P, "pair_index": pair_index,
            "existing20": group_summary(existing_models, rng_existing) if existing_count else None,
            "new80": group_summary(new_models, rng_new),
            "total100": group_summary(existing_models + new_models, rng_total),
        })
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "summary.json", summary)
    np.savez(destination / "bootstrap_indices.npz", total=rng_total, new_only=rng_new, existing_only=rng_existing)
    shutil.copy2(Path(__file__), destination / "plot_script.py")
    write_json(destination / "provenance.json", {
        "summary_sha256": digest(destination / "summary.json"),
        "plot_source_sha256": digest(Path(__file__)),
        "batch_directory": str(batch), "numerical_refitting_performed": False,
    })
    plot(summary, destination, "total100", "total100", total_count)
    plot(summary, destination, "new80", "new80", new_count)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    summarize(args.batch, args.destination)
