"""Aggregate all repetitions fixed in advance, without enforcing monotonicity or selecting favorable seeds."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_1d_hat_experiment import write_json

METRICS = ["force_error_weighted_rel", "trajectory_error_rel", "drift_error_rel", "trajectory_rmse"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
MARKERS = ["o", "s", "^", "D"]


def statistics(values, bootstrap_indices):
    values = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    finite = np.isfinite(values)
    result = {"planned_count": len(values), "available_count": int(finite.sum()),
              "failure_count": int((~finite).sum()),
              "values": [float(value) if np.isfinite(value) else None for value in values],
              "mean": None, "sample_sd": None, "median": None, "iqr": None, "mean_ci95": None}
    # Do not report a conditional mean that discards failures as the mean over all repetitions.
    if not np.all(finite):
        return result
    means = values[bootstrap_indices].mean(axis=1)
    result.update(mean=float(values.mean()), sample_sd=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                  median=float(np.median(values)), iqr=np.quantile(values, [0.25, 0.75]).tolist(),
                  mean_ci95=np.quantile(means, [0.025, 0.975]).tolist())
    return result


def monotonicity(values, bootstrap_indices):
    values = np.asarray(values, dtype=float)
    differences = np.diff(values, axis=1)
    complete = np.all(np.isfinite(values), axis=1)
    all_complete = bool(np.all(complete))
    return {
        "complete_repetitions": int(complete.sum()),
        "nonincreasing_repetitions": int(np.sum(np.all(differences[complete] <= 0, axis=1))),
        "mean_is_nonincreasing": bool(np.all(np.diff(values.mean(axis=0)) <= 0)) if all_complete else None,
        "adjacent_differences_next_minus_previous": [statistics(differences[:, i], bootstrap_indices)
                                                      for i in range(differences.shape[1])],
    }


def plot_summaries(destination, summary, sizes):
    names = [f"hat_M{size}" for size in sizes]
    titles = ["Interaction-force relative error", "Coupled trajectory relative error", "Drift relative error"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, metric, title in zip(axes, METRICS[:3], titles):
        stats = [summary["dictionaries"][name]["metrics"][metric] for name in names]
        raw = np.asarray([[np.nan if value is None else value for value in item["values"]] for item in stats]).T
        for row in raw:
            ax.plot(range(len(sizes)), row, color="0.7", alpha=0.45, linewidth=0.65)
        means = [np.nan if item["mean"] is None else item["mean"] for item in stats]
        ax.plot(range(len(sizes)), means, color="black", linewidth=1.2)
        for i, item in enumerate(stats):
            if item["mean"] is not None:
                lower, upper = item["mean_ci95"]
                ax.vlines(i, lower, upper, color=COLORS[i], linewidth=2)
                ax.plot([i - 0.06, i + 0.06], [lower, lower], color=COLORS[i])
                ax.plot([i - 0.06, i + 0.06], [upper, upper], color=COLORS[i])
                ax.plot(i, item["mean"], marker=MARKERS[i], color=COLORS[i], markersize=7)
            else:
                ax.text(i, 0.95, f"missing {item['failure_count']}", transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
        baseline = summary["dictionaries"]["baseline_six_atoms"]["metrics"][metric]
        if baseline["mean"] is not None:
            ax.axhline(baseline["mean"], color="tab:purple", linestyle="--", linewidth=1)
            ax.axhspan(*baseline["mean_ci95"], color="tab:purple", alpha=0.1)
        ax.set_xticks(range(len(sizes)), sizes)
        ax.set(xlabel="Dictionary size M", title=title)
        ax.set_ylim(bottom=0)
        ax.set_box_aspect(3 / 4)
        ax.grid(axis="y", alpha=0.2)
    handles = [Line2D([0], [0], color="0.7", label="Individual repetitions"),
               Line2D([0], [0], color="black", marker="o", label="Mean; bars: pointwise bootstrap 95% CI"),
               Line2D([0], [0], color="tab:purple", linestyle="--", label="6-atom mean; band: 95% CI")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9)
    N, R = summary["particle_count"], summary["planned_repetitions"]
    fig.suptitle(f"Nested hat dictionaries: N={N}, {R} fixed independent repetitions")
    fig.savefig(destination / "nested_dictionary_errors.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, key, title in zip(axes, ("nonzero_coefficients", "unconverged_candidates"),
                              ("Selected nonzero coefficients", "Unconverged candidates out of 60")):
        for i, name in enumerate(names):
            values = [row.get(key) for row in summary["runs"] if row["dictionary"] == name]
            valid = np.asarray([value for value in values if value is not None])
            ax.scatter(np.full(len(valid), i), valid, color=COLORS[i], marker=MARKERS[i], alpha=0.35)
            if len(valid) == summary["planned_repetitions"]:
                ax.plot(i, valid.mean(), marker="_", color="black", markersize=20, markeredgewidth=2)
        ax.set_xticks(range(len(sizes)), sizes)
        ax.set(xlabel="Dictionary size M", title=title)
        ax.set_ylim(bottom=0)
        ax.set_box_aspect(3 / 4)
    fig.savefig(destination / "sparsity_and_convergence.png", dpi=180)
    plt.close(fig)


def summarize(batch_dir, out_dir=None):
    batch_dir = Path(batch_dir).resolve()
    frozen_path, status_path = batch_dir / "frozen_protocol.json", batch_dir / "batch_status.json"
    frozen, status = json.loads(frozen_path.read_text()), json.loads(status_path.read_text())
    if status["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("the predeclared batch must finish before aggregation")
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != status["frozen_protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for name, digest in frozen["source_sha256"].items():
        if hashlib.sha256((batch_dir / "source_snapshot" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"source snapshot changed: {name}")
    count, sizes = frozen["effective_repetitions"], frozen["effective_sizes"]
    if len(status["runs"]) != count or sorted(run["index"] for run in status["runs"]) != list(range(count)):
        raise ValueError("missing or duplicate repetition records")
    settings = frozen["protocol"]["replication"]
    indices = np.random.default_rng(settings["bootstrap_seed"]).integers(0, count, size=(settings["bootstrap_samples"], count))
    names = [f"hat_M{size}" for size in sizes] + ["baseline_six_atoms"]
    rows, inputs = [], []
    for index in range(count):
        job = next(run for run in status["runs"] if run["index"] == index)
        report_path = batch_dir / f"run-{index + 1:02d}" / "fit" / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        if report is not None:
            digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            if job.get("report_sha256") is not None and digest != job["report_sha256"]:
                raise ValueError("fit report changed")
            inputs.append({"path": str(report_path), "sha256": digest})
            for split in ("train", "validation", "test"):
                if report["protocol"]["seeds"][split] != [frozen["protocol"]["seeds"][split][index]]:
                    raise ValueError("seed does not match the predeclared triplet")
            selection_path = report_path.parent / "frozen_selections.json"
            if report.get("frozen_selections_sha256") is not None:
                if hashlib.sha256(selection_path.read_bytes()).hexdigest() != report["frozen_selections_sha256"]:
                    raise ValueError("frozen model selection changed")
        for name in names:
            item = report.get("results", {}).get(name) if report else None
            row = {"index": index, "dictionary": name, "run_status": job["status"],
                   "fit_status": item["status"] if item else "missing", **{key: None for key in METRICS}}
            if item and item["status"] == "evaluated":
                if len(item["test_metrics"]) != 1:
                    raise ValueError("each repetition must evaluate exactly one independent trajectory")
                metrics = item["test_metrics"][0]
                row.update({key: metrics.get(key) for key in METRICS})
                selected = item["selection"]["selected"]
                row.update(nonzero_coefficients=selected["nonzero_reported"], alpha_sk=selected["alpha_sk"],
                           grid_boundary=item["selection"]["selected_at_grid_boundary"],
                           unconverged_candidates=sum(not entry["converged"] for entry in item["selection"]["path"]),
                           trajectory_status=metrics["trajectory_status"],
                           trajectory_exits_basis_interval=metrics.get("trajectory_exits_basis_interval"))
            rows.append(row)
    summary = {"status": "summarized_all_predeclared_repetitions", "smoke_only": frozen["smoke_only"],
               "particle_count": 8 if frozen["smoke_only"] else frozen["protocol"]["simulation"]["N"],
               "planned_repetitions": count, "runs": rows, "inputs": inputs, "dictionaries": {}, "monotonicity": {},
               "frozen_protocol_sha256": status["frozen_protocol_sha256"],
               "summary_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "aggregation": "Arithmetic mean of per-run relative errors; sample SD; paired percentile bootstrap pointwise 95% intervals.",
               "failure_policy": "If any metric is missing, its all-run mean and interval are null; failed runs are not silently discarded.",
               "bootstrap": {"samples": settings["bootstrap_samples"], "seed": settings["bootstrap_seed"]},
               "limitations": ["Intervals are pointwise, without a multiple-comparison correction.",
                               "Monotonicity is measured, never imposed by selection, smoothing, or early stopping.",
                               "The truth is not exactly represented in the hat dictionaries; no coefficient-recovery claim."]}
    for name in names:
        selected_rows = [row for row in rows if row["dictionary"] == name]
        summary["dictionaries"][name] = {
            "metrics": {key: statistics([row[key] for row in selected_rows], indices) for key in METRICS},
            "missing_models": sum(row["fit_status"] != "evaluated" for row in selected_rows),
            "boundary_selections": sum(row.get("grid_boundary", False) for row in selected_rows),
            "divergent_rollouts": sum(row.get("trajectory_status") == "diverged" for row in selected_rows),
        }
    for key in METRICS:
        values = np.asarray([[summary["dictionaries"][name]["metrics"][key]["values"][index]
                              for name in names[:-1]] for index in range(count)], dtype=float)
        summary["monotonicity"][key] = monotonicity(values, indices)
    destination = Path(out_dir) if out_dir else batch_dir / "summary"
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "summary.json", summary)
    plot_summaries(destination, summary, sizes)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    summarize(args.batch_dir, args.out_dir)
