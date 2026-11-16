"""Aggregate all repetitions fixed in advance and create a six-panel plot overlaying dictionary sizes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
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
from scripts.run_1d_hat_sweeps import cases, digest, job_plan, METRICS, read_json
from scripts.summarize_1d_hat_repetitions import statistics, COLORS, MARKERS

LABELS = {"sigma_obs": r"Observation noise $\sigma_{\rm obs}$", "sigma": r"Process noise $\sigma$",
          "N": r"Particles $N$", "dt": r"Time step $\Delta t$", "P": r"Sampled pairs $P$"}
TITLES = {"force_error_weighted_rel": "Interaction-force relative error", "trajectory_error_rel": "Coupled trajectory relative error",
          "drift_error_rel": "Drift relative error", "trajectory_rmse": "Coupled trajectory RMSE"}
PAIR_METRICS = ["coefficient_difference_sq", "force_difference_sq_rel", "design_difference_sq"]


def aggregate_models(reports, names, indices, pair=False):
    output = {}
    for name in names:
        rows = [report.get("results", {}).get(name, {}) for report in reports]
        metric_names = METRICS + PAIR_METRICS if pair else METRICS
        stats = {metric: statistics([row.get("metrics", {}).get(metric) for row in rows], indices) for metric in metric_names}
        output[name] = dict(metrics=stats,
            unconverged_candidates=sum(sum(not candidate["converged"] for candidate in row.get("selection", {}).get("path", [])) for row in rows),
            candidate_solves=sum(len(row.get("selection", {}).get("path", [])) for row in rows),
            boundary_selections=sum(row.get("selection", {}).get("selected_at_grid_boundary", False) for row in rows),
            unconverged_endpoints=sum("endpoint" in row and not row["endpoint"]["converged"] for row in rows),
            endpoint_count=sum("endpoint" in row for row in rows),
            pair_path_solves=sum(len(row.get("path", [])) for row in rows) if pair else 0,
            pair_path_unconverged=sum(sum(not c["converged"] for c in row.get("path", [])) for row in rows) if pair else 0,
            divergent_rollouts=sum(row.get("metrics", {}).get("trajectory_status") == "diverged" for row in rows),
            domain_excursions=sum(row.get("metrics", {}).get("trajectory_exits_basis_interval", False) for row in rows),
            missing_models=sum(not row for row in rows))
    return output


def yscale(ax, all_values):
    finite = np.asarray([v for v in all_values if v is not None and np.isfinite(v)], dtype=float)
    if finite.size and np.min(finite) <= 0:
        positive = finite[finite > 0]
        ax.set_yscale("symlog", linthresh=float(positive.min() / 10) if len(positive) else 1e-12)
        ax.text(.02, .98, "symlog (zero retained)", transform=ax.transAxes, va="top", fontsize=7)
    else:
        ax.set_yscale("log")


def error_panel(ax, points, metric, names, xlabel, title):
    x = np.asarray([point["value"] for point in points], dtype=float)
    all_values = []
    for i, name in enumerate(names):
        stats = [point["dictionaries"][name]["metrics"][metric] for point in points]
        means = np.array([np.nan if stat["mean"] is None else stat["mean"] for stat in stats])
        lower = np.array([np.nan if stat["mean_ci95"] is None else stat["mean_ci95"][0] for stat in stats])
        upper = np.array([np.nan if stat["mean_ci95"] is None else stat["mean_ci95"][1] for stat in stats])
        all_values.extend(means[np.isfinite(means)].tolist() + lower[np.isfinite(lower)].tolist())
        ax.plot(x, means, color=COLORS[i], linewidth=1.4)
        ax.fill_between(x, lower, upper, color=COLORS[i], alpha=.13, linewidth=0)
        for j, (value, mean, stat) in enumerate(zip(x, means, stats)):
            flagged = points[j]["dictionaries"][name]["unconverged_endpoints"] > 0
            if np.isfinite(mean):
                ax.plot(value, mean, marker=MARKERS[i], color=COLORS[i],
                        markerfacecolor="white" if flagged else COLORS[i], markersize=5)
            else:
                ax.plot(value, .025 + i * .045, marker="x", color=COLORS[i], transform=ax.get_xaxis_transform())
                ax.annotate(f"{stat['available_count']}/{stat['planned_count']}", (value, .025 + i * .045),
                            xycoords=ax.get_xaxis_transform(), xytext=(0, 4), textcoords="offset points",
                            color=COLORS[i], ha="center", fontsize=6)
    yscale(ax, all_values)
    if np.any(x == 0):
        positive = x[x > 0]
        ax.set_xscale("symlog", linthresh=float(positive.min()))
    else:
        ax.set_xscale("log")
    ax.set_xticks(x, [f"{value:g}" for value in x], rotation=25)
    ax.minorticks_off()
    ax.set(xlabel=xlabel, title=title)
    ax.set_box_aspect(3 / 4)
    ax.grid(alpha=.18)


def runtime_panel(ax, points, names):
    x = [point["value"] for point in points]
    for i, name in enumerate(names):
        for mode, style in (("full", "-"), ("sampled", "--")):
            stats = [point["dictionaries"][name][mode] for point in points]
            median = [stat["median"] if stat["median"] is not None else np.nan for stat in stats]
            lower = [stat["iqr"][0] if stat["iqr"] else np.nan for stat in stats]
            upper = [stat["iqr"][1] if stat["iqr"] else np.nan for stat in stats]
            ax.plot(x, median, style, marker=MARKERS[i], color=COLORS[i], markersize=4)
            ax.fill_between(x, lower, upper, color=COLORS[i], alpha=.08)
    ax.set(xlabel=r"Particles $N$", ylabel="Assembly time [s]", title="(f) Assembly cost", xscale="log", yscale="log")
    ax.set_xticks(x, [str(value) for value in x], rotation=25)
    ax.minorticks_off()
    ax.set_box_aspect(3 / 4)
    ax.grid(alpha=.18)
    ax.legend(handles=[Line2D([0], [0], color="0.3", label="Exact full (1D prefix sums)"),
                       Line2D([0], [0], color="0.3", linestyle="--", label=r"Sampled, $P=10N$")], fontsize=8)


def legend(fig, names):
    fig.legend(handles=[Line2D([0], [0], color=color, marker=marker, label=name.replace("hat_M", "M="))
                        for name, color, marker in zip(names, COLORS, MARKERS)],
               loc="outside upper center", ncol=len(names), fontsize=11)


def save_figure(fig, destination, stem):
    fig.savefig(destination / f"{stem}.png", dpi=180)
    fig.savefig(destination / f"{stem}.pdf")
    plt.close(fig)


def plot_summary(summary, destination):
    names = [f"hat_M{M}" for M in summary["sizes"]]
    count = summary["planned_repetitions"]
    base = summary["baseline"]
    context = ("SMOKE ONLY: plumbing check, not recovery evidence." if summary["smoke_only"] else
               f"Baseline: N={base['N']}, T={base['horizon']:g}, dt={base['dt']:g}, sigma={base['sigma']:g}, sigma_obs={base['sigma_obs']:g}; only the swept parameter varies.")
    for metric, stem in zip(METRICS[:3], ("force_sweeps", "trajectory_sweeps", "drift_sweeps")):
        fig, axes = plt.subplots(2, 3, figsize=(14.4, 9.3), constrained_layout=True)
        for ax, key, letter in zip(axes.ravel()[:4], ("sigma_obs", "sigma", "N", "dt"), "abcd"):
            error_panel(ax, summary["accuracy"][key], metric, names, LABELS[key], f"({letter}) {LABELS[key]}")
            ax.set_ylabel(TITLES[metric])
        error_panel(axes[1, 1], summary["pairs"], metric, names, LABELS["P"], "(e) Sampled-pair recovery")
        axes[1, 1].set_ylabel(TITLES[metric])
        runtime_panel(axes[1, 2], summary["runtime"], names)
        legend(fig, names)
        fig.supxlabel(context + f"\nErrors: means and pointwise bootstrap 95% CIs, {count} repetitions. Cost: median and IQR.\n"
                      "Open markers: some fixed-alpha pair fits unconverged. Crosses: incomplete metric (available/planned).", fontsize=9)
        save_figure(fig, destination, stem)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)
    for ax, metric, title in zip(axes, PAIR_METRICS,
        ("Squared coefficient difference", "Relative squared interaction-force difference", "Squared design difference")):
        error_panel(ax, summary["pairs"], metric, names, LABELS["P"], title)
        ax.set_ylabel("Sampled versus full pairs")
    legend(fig, names)
    fig.supxlabel(context + "\nFull-pair-selected absolute penalty held fixed. Finite-iteration outputs; open markers flag incomplete convergence.\n"
                  "Coefficient differences compare two fits in the SAME dictionary, not coefficients of the true kernel.", fontsize=9)
    save_figure(fig, destination, "pair_approximation")


def summarize(batch, destination):
    batch, destination = Path(batch).resolve(), Path(destination).resolve()
    status, frozen = read_json(batch / "batch_status.json"), read_json(batch / "frozen_protocol.json")
    if status["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("the predeclared batch must finish before aggregation")
    if digest(batch / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for name, expected in frozen["source_sha256"].items():
        if digest(batch / "source_snapshot" / name) != expected:
            raise ValueError("archived source changed")
    for name, expected_hash in frozen.get("baseline_provenance_sha256", {}).items():
        if digest(batch / "baseline_provenance" / name) != expected_hash:
            raise ValueError("baseline provenance changed")
    spec, base = frozen["protocol"], frozen["base"]
    expected = job_plan(spec, base)
    if status["jobs"] != expected or set(status["results"]) != set(expected):
        raise ValueError("missing or altered planned jobs")
    reports, inputs = {}, {}
    for key, record in status["results"].items():
        path = batch / record["directory"] / "report.json"
        if digest(path) != record["report_sha256"]:
            raise ValueError("report changed")
        report = read_json(path)
        if report["job"] != expected[key] or report["status"] != record["status"]:
            raise ValueError("job identity/status mismatch")
        if report["status"] == "completed":
            for filename, expected_hash in report["artifacts_sha256"].items():
                if digest(path.parent / filename) != expected_hash:
                    raise ValueError(f"artifact changed: {path.parent / filename}")
        reports[key] = report
        inputs[key] = dict(path=str(path.relative_to(batch)), sha256=record["report_sha256"])
    count = base["replication"]["count"]
    indices = np.random.default_rng(base["replication"]["bootstrap_seed"]).integers(
        0, count, size=(base["replication"]["bootstrap_samples"], count))
    names = [f"hat_M{M}" for M in base["basis"]["sizes"]]
    summary = dict(smoke_only=frozen["smoke_only"], planned_repetitions=count, sizes=base["basis"]["sizes"],
                   baseline=base["simulation"], accuracy={}, pairs=[], runtime=[], inputs=inputs,
                   frozen_protocol_sha256=status["frozen_protocol_sha256"],
                   bootstrap_seed=base["replication"]["bootstrap_seed"], bootstrap_samples=len(indices),
                   interval="pointwise percentile bootstrap of arithmetic per-run mean; indices paired across dictionaries and settings",
                   pair_estimand=spec["pair_nonconvergence_policy"], failures={key: r.get("error") for key, r in reports.items() if r["status"] != "completed"})
    for case in cases(spec, base):
        rows = [reports[f"{case['id']}-{i}"] for i in range(count)]
        point = dict(value=case["value"], case=case["id"], dictionaries=aggregate_models(rows, names, indices))
        summary["accuracy"].setdefault(case["parameter"], []).append(point)
    for pi, P in enumerate(spec["pair_counts"]):
        rows = [reports[f"pairs-{pi}-{i}"] for i in range(count)]
        summary["pairs"].append(dict(value=P, dictionaries=aggregate_models(rows, names, indices, pair=True)))
    timing_count = spec["runtime_repetitions"]
    timing_indices = np.random.default_rng(830000).integers(0, timing_count, size=(10000, timing_count))
    for ni, N in enumerate(spec["runtime_particles"]):
        point = dict(value=N, dictionaries={})
        for name in names:
            point["dictionaries"][name] = {}
            for mode in ("full", "sampled"):
                values = [next((row["seconds"] for row in reports[f"runtime-{ni}-{i}"].get("timings", [])
                                if row["name"] == name and row["mode"] == mode), None) for i in range(timing_count)]
                point["dictionaries"][name][mode] = statistics(values, timing_indices)
        summary["runtime"].append(point)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "summary.json", summary)
    np.save(destination / "bootstrap_indices.npy", indices)
    shutil.copy2(Path(__file__), destination / "plot_script.py")
    write_json(destination / "provenance.json", dict(summary_sha256=digest(destination / "summary.json"),
        plot_source_sha256=digest(Path(__file__)), batch_directory=str(batch), numerical_refitting_performed=False))
    plot_summary(summary, destination)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    summarize(args.batch, args.destination)
