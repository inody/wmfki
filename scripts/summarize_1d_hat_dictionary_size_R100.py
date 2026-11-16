"""Aggregate the comparison of M=9,17,33,65 over the same 100 repetitions."""

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
from scripts.run_1d_hat_dictionary_size_R100 import digest, job_plan, read_json
from scripts.summarize_1d_hat_repetitions import statistics

METRICS = ["force_error_weighted_rel", "trajectory_error_rel", "drift_error_rel", "trajectory_rmse"]


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
        "unconverged_candidates": sum(sum(not row.get("converged", False) for row in model.get("selection", {}).get("path", []))
                                          for model in models),
        "candidate_solves": sum(len(model.get("selection", {}).get("path", [])) for model in models),
        "boundary_selections": sum(model.get("selection", {}).get("selected_at_grid_boundary", False) for model in models),
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


def plot(summary, destination):
    points = summary["dictionary_sizes"]
    fig, ax = plt.subplots(figsize=(5.3, 3.8), constrained_layout=True)
    rng = np.random.default_rng(970099)
    for point in points:
        M = point["value"]
        values = np.asarray(point["total100"]["metrics"]["force_error_weighted_rel"]["values"], dtype=float)
        jitter = np.exp(rng.uniform(-.035, .035, size=len(values)))
        ax.scatter(M * jitter, values, s=8, alpha=.16, color="tab:orange", linewidths=0)
    x = np.asarray([point["value"] for point in points], dtype=float)
    stats = [point["total100"]["metrics"]["force_error_weighted_rel"] for point in points]
    mean = np.asarray([row["mean"] for row in stats])
    lower = np.asarray([row["mean_ci95"][0] for row in stats])
    upper = np.asarray([row["mean_ci95"][1] for row in stats])
    ax.errorbar(x, mean, yerr=np.vstack([mean - lower, upper - mean]), fmt="o-", color="tab:red",
                linewidth=1.7, capsize=3, label="mean and bootstrap 95% CI")
    ax.set(xscale="log", yscale="log", xlabel="dictionary size M", ylabel="held-out relative force error")
    ax.set_xticks(x, [str(int(value)) for value in x])
    ax.minorticks_off()
    ax.grid(alpha=.18)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(destination / "dictionary_size_force_error.png", dpi=180)
    fig.savefig(destination / "dictionary_size_force_error.pdf")
    plt.close(fig)


def summarize(batch, destination):
    batch, destination = Path(batch).resolve(), Path(destination).resolve()
    frozen = read_json(batch / "frozen_protocol.json")
    status = read_json(batch / "batch_status.json")
    if status["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("batch must finish before aggregation")
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
    reports, inputs = {}, {"new": {}, "existing": frozen["imported_existing"], "new_M17": frozen["imported_new_M17"]}
    for key, record in status["results"].items():
        path = batch / record["directory"] / "report.json"
        if digest(path) != record["report_sha256"]:
            raise ValueError("new report changed")
        report = read_json(path)
        if report["job"] != jobs[key] or report["status"] != record["status"]:
            raise ValueError("new job identity/status mismatch")
        for name, expected in report["artifacts_sha256"].items():
            if digest(path.parent / name) != expected:
                raise ValueError("new artifact changed")
        reports[key] = report
        inputs["new"][key] = {"path": str(path.relative_to(batch)), "sha256": record["report_sha256"]}
    smoke = frozen["smoke_only"]
    old_count = 0 if smoke else frozen["requested_protocol"]["existing_repetitions"]
    new_count = frozen["base"]["replication"]["count"]
    total_count = old_count + new_count
    boot = frozen["requested_protocol"]["bootstrap"]
    rng_total = np.random.default_rng(boot["total_seed"]).integers(0, total_count, size=(boot["samples"], total_count))
    rng_new = np.random.default_rng(boot["new_only_seed"]).integers(0, new_count, size=(boot["samples"], new_count))
    rng_old = (np.random.default_rng(boot["existing_only_seed"]).integers(0, old_count, size=(boot["samples"], old_count))
               if old_count else np.empty((boot["samples"], 0), dtype=int))
    existing = []
    for index in range(old_count):
        record = frozen["imported_existing"][f"base-{index}"]
        path = batch / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("existing compact record changed")
        row = read_json(path)
        if row["index"] != index or digest(row["source_report"]) != row["source_report_sha256"]:
            raise ValueError("existing compact source mismatch")
        existing.append(row)
    new_m17 = []
    if not smoke:
        for index in range(new_count):
            record = frozen["imported_new_M17"][f"base-{index}"]
            path = batch / record["path"]
            if digest(path) != record["sha256"]:
                raise ValueError("new M17 compact record changed")
            row = read_json(path)
            if row["index"] != index or digest(row["source_report"]) != row["source_report_sha256"]:
                raise ValueError("new M17 compact source mismatch")
            new_m17.append(row["result"])
    sizes = frozen["requested_protocol"]["dictionary_sizes"] if not smoke else frozen["base"]["basis"]["sizes"]
    summary = {
        "smoke_only": smoke, "existing_repetitions": old_count, "new_repetitions": new_count,
        "total_repetitions": total_count, "baseline": frozen["base"]["simulation"],
        "dictionary_sizes": [], "inputs": inputs, "bootstrap": boot,
        "frozen_protocol_sha256": status["frozen_protocol_sha256"],
        "failures": [key for key, report in reports.items() if report["status"] != "completed"],
        "representative": {"policy": frozen["requested_protocol"]["representative_policy"],
                           "source_report": existing[0]["source_report"] if existing else None,
                           "source_report_sha256": existing[0]["source_report_sha256"] if existing else None,
                           "dictionary": "hat_M17", "index": 0},
    }
    for M in sizes:
        name = f"hat_M{M}"
        old_models = [row["results"][name] for row in existing]
        if M == 17:
            models_new = new_m17
        else:
            models_new = [reports[f"base-{index}"].get("results", {}).get(name, {}) for index in range(new_count)]
        summary["dictionary_sizes"].append({
            "value": M,
            "existing20": group_summary(old_models, rng_old) if old_count else None,
            "new80": group_summary(models_new, rng_new),
            "total100": group_summary(old_models + models_new, rng_total),
        })
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "summary.json", summary)
    np.savez(destination / "bootstrap_indices.npz", total=rng_total, new_only=rng_new, existing_only=rng_old)
    shutil.copy2(Path(__file__), destination / "plot_script.py")
    write_json(destination / "provenance.json", {"summary_sha256": digest(destination / "summary.json"),
        "plot_source_sha256": digest(Path(__file__)), "batch_directory": str(batch),
        "numerical_refitting_performed": True})
    plot(summary, destination)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    summarize(args.batch, args.destination)
