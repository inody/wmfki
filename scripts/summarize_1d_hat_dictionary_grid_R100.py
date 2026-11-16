"""Aggregate the 100-repetition comparison of ten QANR hat-dictionary sizes."""

import argparse
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
from scripts.run_1d_hat_dictionary_size_R100 import digest, read_json
from scripts.run_1d_hat_dictionary_grid_R100 import job_plan
from scripts.summarize_1d_hat_dictionary_size_R100 import diagnostics, extended_statistics, METRICS


def group_summary(models, indices):
    return {"metrics": {metric: extended_statistics([model.get("metrics", {}).get(metric) for model in models], indices)
                        for metric in METRICS}, "diagnostics": diagnostics(models)}


def plot(summary, destination):
    points = summary["dictionary_sizes"]
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    rng = np.random.default_rng(980099)
    for point in points:
        values = np.asarray(point["total100"]["metrics"]["force_error_weighted_rel"]["values"], dtype=float)
        ax.scatter(point["value"] * np.exp(rng.uniform(-.025, .025, len(values))), values,
                   s=7, alpha=.12, color="tab:gray", linewidths=0)
    sizes = np.asarray([point["value"] for point in points], dtype=float)
    rows = [point["total100"]["metrics"]["force_error_weighted_rel"] for point in points]
    means = np.asarray([row["mean"] for row in rows])
    lower = np.asarray([row["mean_ci95"][0] for row in rows])
    upper = np.asarray([row["mean_ci95"][1] for row in rows])
    ax.errorbar(sizes, means, yerr=np.vstack([means-lower, upper-means]), fmt="o-", color="tab:red",
                linewidth=1.6, capsize=2.5, label="mean and paired bootstrap 95% CI")
    ax.set(xscale="log", yscale="log", xlabel="dictionary size M", ylabel="held-out relative force error")
    ax.set_xticks(sizes, [str(int(x)) for x in sizes], rotation=0)
    ax.minorticks_off()
    ax.grid(alpha=.18)
    ax.legend(fontsize=8)
    fig.savefig(destination / "dictionary_grid_force_error.pdf")
    fig.savefig(destination / "dictionary_grid_force_error.png", dpi=180)
    plt.close(fig)


def summarize(batch, destination):
    batch, destination = Path(batch).resolve(), Path(destination).resolve()
    frozen = read_json(batch / "frozen_protocol.json")
    status = read_json(batch / "batch_status.json")
    if status["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("batch incomplete")
    if digest(batch / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for name, expected in frozen["source_sha256"].items():
        if digest(batch / "source_snapshot" / name) != expected:
            raise ValueError("source snapshot changed")
    for name, expected in frozen["source_provenance_sha256"].items():
        if digest(batch / "source_provenance" / name) != expected:
            raise ValueError("source provenance changed")
    jobs = job_plan(frozen["requested_protocol"], frozen["smoke_only"])
    if status["jobs"] != jobs or set(status["results"]) != set(jobs):
        raise ValueError("new jobs missing or changed")
    reports, inputs = {}, {"new": {}, "reused": frozen["imported_reused_models"]}
    for key, record in status["results"].items():
        path = batch / record["directory"] / "report.json"
        if digest(path) != record["report_sha256"]:
            raise ValueError("new report changed")
        report = read_json(path)
        if report["job"] != jobs[key] or report["status"] != record["status"]:
            raise ValueError("new report identity/status mismatch")
        for name, expected in report["artifacts_sha256"].items():
            if digest(path.parent / name) != expected:
                raise ValueError("new artifact changed")
        reports[key] = report
        inputs["new"][key] = {"path": str(path.relative_to(batch)), "sha256": record["report_sha256"]}
    count = frozen["base"]["replication"]["count"]
    seed = frozen["requested_protocol"]["bootstrap"]["seed"]
    samples = frozen["requested_protocol"]["bootstrap"]["samples"]
    indices = np.random.default_rng(seed).integers(0, count, size=(samples, count))
    reused = []
    if not frozen["smoke_only"] and frozen["imported_reused_models"]:
        for index in range(count):
            record = frozen["imported_reused_models"][f"run-{index}"]
            path = batch / record["path"]
            if digest(path) != record["sha256"]:
                raise ValueError("reused compact record changed")
            row = read_json(path)
            if row["global_index"] != index:
                raise ValueError("reused compact identity mismatch")
            if digest(Path(row["reference_directory"]) / "report.json") != row["reference_report_sha256"]:
                raise ValueError("trajectory reference changed")
            reused.append(row)
    sizes = frozen["requested_protocol"]["dictionary_sizes"] if not frozen["smoke_only"] else frozen["base"]["basis"]["sizes"]
    summary = {
        "smoke_only": frozen["smoke_only"], "total_repetitions": count,
        "baseline": frozen["base"]["simulation"], "dictionary_sizes": [], "inputs": inputs,
        "bootstrap": frozen["requested_protocol"]["bootstrap"],
        "frozen_protocol_sha256": status["frozen_protocol_sha256"],
        "failures": [key for key, report in reports.items() if report["status"] != "completed"],
        "representative": {"policy": frozen["requested_protocol"]["representative_policy"],
                           "source_report": (
                               str(Path(reused[0]["reference_directory"]) / "report.json")
                               if reused else str(Path(reports["run-0"]["reference_directory"]) / "report.json")
                           ),
                           "source_report_sha256": (
                               reused[0]["reference_report_sha256"]
                               if reused else reports["run-0"]["reference_report_sha256"]
                           ),
                           "dictionary": "hat_M17", "index": 0},
    }
    reused_sizes = set(frozen["requested_protocol"]["reused_sizes"])
    for M in sizes:
        name = f"hat_M{M}"
        models = ([row["results"][name] for row in reused] if M in reused_sizes else
                  [reports[f"run-{i}"].get("results", {}).get(name, {}) for i in range(count)])
        summary["dictionary_sizes"].append({"value": M, "source": "reused" if M in reused_sizes else "new",
                                             "total100": group_summary(models, indices)})
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "summary.json", summary)
    np.savez(destination / "bootstrap_indices.npz", total=indices)
    shutil.copy2(Path(__file__), destination / "plot_script.py")
    write_json(destination / "provenance.json", {"summary_sha256": digest(destination / "summary.json"),
        "plot_source_sha256": digest(Path(__file__)), "batch_directory": str(batch),
        "new_models_fitted": frozen["requested_protocol"]["new_models"] if not frozen["smoke_only"] else count})
    plot(summary, destination)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    summarize(args.batch, args.destination)
