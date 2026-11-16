"""Aggregate the OLS results for ten QANR hat-dictionary sizes over 100 repetitions."""

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
from scripts.run_1d_hat_dictionary_grid_OLS_R100 import job_plan
from scripts.summarize_1d_hat_dictionary_size_R100 import extended_statistics, METRICS

FIT_METRICS = ["train_mse", "validation_mse", "test_mse", "coefficient_l1", "coefficient_l2", "rank",
               "nonzero_reported"]


def diagnostics(models):
    return {
        "models": len(models),
        "missing_models": sum(not model for model in models),
        "nonfinite_fits": sum(model.get("fit", {}).get("status") != "fitted" for model in models),
        "rank_deficient": sum(not model.get("fit", {}).get("full_column_rank", False) for model in models),
        "divergent_rollouts": sum(model.get("metrics", {}).get("trajectory_status") == "diverged" for model in models),
        "basis_domain_excursions": sum(model.get("metrics", {}).get("trajectory_exits_basis_interval", False)
                                         for model in models),
    }


def group_summary(models, indices):
    return {
        "metrics": {metric: extended_statistics([model.get("metrics", {}).get(metric) for model in models], indices)
                    for metric in METRICS},
        "fit_metrics": {metric: extended_statistics([model.get("fit", {}).get(metric) for model in models], indices)
                        for metric in FIT_METRICS},
        "diagnostics": diagnostics(models),
    }


def plot(summary, destination):
    points = summary["dictionary_sizes"]
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    rng = np.random.default_rng(990099)
    for point in points:
        values = point["total100"]["metrics"]["force_error_weighted_rel"]["values"]
        finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
        if len(finite):
            ax.scatter(point["value"] * np.exp(rng.uniform(-.025, .025, len(finite))), finite,
                       s=7, alpha=.12, color="tab:gray", linewidths=0)
    sizes = np.asarray([point["value"] for point in points], dtype=float)
    rows = [point["total100"]["metrics"]["force_error_weighted_rel"] for point in points]
    means = np.asarray([row["mean"] if row["mean"] is not None else np.nan for row in rows])
    lower = np.asarray([row["mean_ci95"][0] if row["mean_ci95"] is not None else np.nan for row in rows])
    upper = np.asarray([row["mean_ci95"][1] if row["mean_ci95"] is not None else np.nan for row in rows])
    ax.errorbar(sizes, means, yerr=np.vstack([means-lower, upper-means]), fmt="o-", color="tab:red",
                linewidth=1.6, capsize=2.5, label="mean and paired bootstrap 95% CI")
    ax.set(xscale="log", yscale="log", xlabel="dictionary size M", ylabel="held-out relative force error")
    ax.set_xticks(sizes, [str(int(x)) for x in sizes])
    ax.minorticks_off()
    ax.grid(alpha=.18)
    ax.legend(fontsize=8)
    fig.savefig(destination / "dictionary_grid_ols_force_error.pdf")
    fig.savefig(destination / "dictionary_grid_ols_force_error.png", dpi=180)
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
        raise ValueError("jobs missing or changed")
    reports, inputs = {}, {"new": {}}
    for key, record in status["results"].items():
        path = batch / record["directory"] / "report.json"
        if digest(path) != record["report_sha256"]:
            raise ValueError("report changed")
        report = read_json(path)
        if report["job"] != jobs[key] or report["status"] != record["status"]:
            raise ValueError("report identity/status mismatch")
        for name, expected in report["artifacts_sha256"].items():
            if digest(path.parent / name) != expected:
                raise ValueError("artifact changed")
        reports[key] = report
        inputs["new"][key] = {"path": str(path.relative_to(batch)), "sha256": record["report_sha256"]}
    count = frozen["base"]["replication"]["count"]
    boot = frozen["requested_protocol"]["bootstrap"]
    indices = np.random.default_rng(boot["seed"]).integers(0, count, size=(boot["samples"], count))
    sizes = frozen["base"]["basis"]["sizes"]
    summary = {
        "estimator": frozen["requested_protocol"]["estimator"], "smoke_only": frozen["smoke_only"],
        "total_repetitions": count, "baseline": frozen["base"]["simulation"],
        "dictionary_sizes": [], "inputs": inputs, "bootstrap": boot,
        "frozen_protocol_sha256": status["frozen_protocol_sha256"],
        "failures": [key for key, report in reports.items() if report["status"] != "completed"],
        "representative": {"policy": frozen["requested_protocol"]["representative_policy"],
                           "source_report": None, "source_report_sha256": None,
                           "dictionary": None, "index": 0},
    }
    for M in sizes:
        name = f"hat_M{M}"
        models = [reports[f"run-{i}"].get("results", {}).get(name, {}) for i in range(count)]
        summary["dictionary_sizes"].append({"value": M, "source": "new", "total100": group_summary(models, indices)})
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "summary.json", summary)
    np.savez(destination / "bootstrap_indices.npz", total=indices)
    shutil.copy2(Path(__file__), destination / "plot_script.py")
    write_json(destination / "provenance.json", {"summary_sha256": digest(destination / "summary.json"),
        "plot_source_sha256": digest(Path(__file__)), "batch_directory": str(batch),
        "models_fitted": count * len(sizes)})
    plot(summary, destination)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    summarize(args.batch, args.destination)
