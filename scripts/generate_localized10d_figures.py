"""Generate publication figures for the frozen realizable 10D scientific run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import make_radial_basis
from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR, TRUE_COLOR, apply_manuscript_style, finish_axes, save_pdf,
)
from scripts.run_localized10d_pilot import _simulate_coupled
from scripts.run_morse10d_sweeps import read_json, repetition_plan

BOOTSTRAP_SEED = 20261022
BOOTSTRAP_DRAWS = 10000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _baseline_lasso(records, spec):
    return next(
        row for row in records
        if row["repetition"] == 0 and row["sweep"] == "process_sigma"
        and float(row["sweep_value"]) == float(spec["model"]["sigma"])
        and row["method"] == "lasso"
    )


def _mean_ci(values, draws):
    values = np.asarray(values, dtype=float)
    sampled = values[draws].mean(axis=1)
    return float(values.mean()), np.quantile(sampled, [0.025, 0.975]).tolist()


def _sweep_statistics(records, spec):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result = {}
    slopes = {}
    repetitions = int(spec["evaluation"]["scientific_repetitions"])
    draws = rng.integers(0, repetitions, size=(BOOTSTRAP_DRAWS, repetitions))
    for sweep in ("process_sigma", "particle_number", "sampled_pairs"):
        result[sweep] = []
        for value in spec["sweeps"][sweep]:
            item = {"value": value}
            for method in ("lasso", "ols"):
                rows = sorted(
                    [row for row in records if row["sweep"] == sweep
                     and float(row["sweep_value"]) == float(value)
                     and row["method"] == method],
                    key=lambda row: row["repetition"],
                )
                if len(rows) != repetitions:
                    raise ValueError(f"incomplete {sweep}={value}, {method}")
                mean, ci = _mean_ci(
                    [row["coefficient_error_relative"] for row in rows], draws
                )
                item[method] = {"mean": mean, "ci95": ci}
            if sweep == "sampled_pairs":
                rows = sorted(
                    [row for row in records if row["sweep"] == sweep
                     and float(row["sweep_value"]) == float(value)
                     and row["method"] == "lasso"],
                    key=lambda row: row["repetition"],
                )
                design_key = (
                    "pair_design_relative_error"
                    if "pair_design_relative_error" in rows[0]
                    else "pair_design_relative_difference_from_P0"
                )
                mean, ci = _mean_ci([row[design_key] for row in rows], draws)
                item["sampled_design"] = {"mean": mean, "ci95": ci}
            result[sweep].append(item)
        for method in ("lasso", "ols"):
            positive = [item for item in result[sweep] if item["value"] > 0]
            slopes[f"{sweep}_{method}"] = float(np.polyfit(
                np.log([item["value"] for item in positive]),
                np.log([item[method]["mean"] for item in positive]), 1,
            )[0])
        if sweep == "sampled_pairs":
            positive = [item for item in result[sweep]
                        if item["sampled_design"]["mean"] > 0]
            slopes["sampled_pairs_design"] = float(np.polyfit(
                np.log([item["value"] for item in positive]),
                np.log([item["sampled_design"]["mean"] for item in positive]), 1,
            )[0])
    return result, slopes


def _errorbar(axis, items, method, color, marker, label):
    x = np.asarray([item["value"] for item in items], dtype=float)
    mean = np.asarray([item[method]["mean"] for item in items])
    ci = np.asarray([item[method]["ci95"] for item in items])
    axis.errorbar(x, mean, yerr=np.vstack((mean - ci[:, 0], ci[:, 1] - mean)),
                  color=color, marker=marker, capsize=2, label=label)
    return x, mean


def make_recovery(records, spec, destination):
    selected = _baseline_lasso(records, spec)
    theta = np.asarray(selected["coefficients"], dtype=float)
    target = np.asarray(spec["model"]["interaction"]["coefficients"], dtype=float)
    basis, meta = make_radial_basis(spec["dictionary"])
    radius = np.linspace(
        *spec["evaluation"]["profile_radius"],
        int(spec["evaluation"]["profile_grid_size"]),
    )
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), constrained_layout=True)
    radial_basis = basis(radius)
    axes[0].plot(radius, radius * (radial_basis @ target),
                 color=TRUE_COLOR, label="true")
    axes[0].plot(radius, radius * (radial_basis @ theta), color=ESTIMATE_COLOR,
                 linestyle="--", label="weak LASSO")
    axes[0].axhline(0, color="0.4", linewidth=0.7)
    axes[0].set(title="(a) Radial interaction", xlabel="pairwise distance $r$",
                ylabel=r"radial interaction $r\phi(r)$")
    axes[0].legend(loc="best")
    x = np.arange(target.size); width = 0.38
    axes[1].bar(x - width / 2, target, width, color=TRUE_COLOR, label="true")
    axes[1].bar(x + width / 2, theta, width, color=ESTIMATE_COLOR, label="weak LASSO")
    axes[1].set_xticks(x, [f"{center:g}" for center in meta["centers"]], rotation=30)
    axes[1].set(title="(b) Dictionary coefficients", xlabel="interaction radius", ylabel="coefficient")
    axes[1].legend(loc="best")
    for axis in axes: finish_axes(axis)
    save_pdf(fig, destination)
    return {"repetition": 0, "coefficients": theta.tolist(),
            "coefficient_error_relative": selected["coefficient_error_relative"],
            "latent_drift_relative_error": selected["latent_drift_relative_error"]}


def make_dynamics(records, spec, destination):
    theta = np.asarray(_baseline_lasso(records, spec)["coefficients"], dtype=float)
    plan = repetition_plan(spec)[0]
    truth, learned, trajectory_error = _simulate_coupled(spec, theta, plan["test_seed"])
    times = np.arange(truth.shape[0]) * spec["model"]["dt"]
    true_radial = np.linalg.norm(truth - truth.mean(axis=1, keepdims=True), axis=-1)
    learned_radial = np.linalg.norm(learned - learned.mean(axis=1, keepdims=True), axis=-1)
    tri = np.triu_indices(truth.shape[1], k=1)
    true_dist = np.linalg.norm(truth[-1, :, None, :] - truth[-1, None, :, :], axis=-1)[tri]
    learned_dist = np.linalg.norm(learned[-1, :, None, :] - learned[-1, None, :, :], axis=-1)[tri]
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), constrained_layout=True)
    for radial, color, style, label in (
        (true_radial, TRUE_COLOR, "-", "true"),
        (learned_radial, ESTIMATE_COLOR, "--", "weak LASSO"),
    ):
        axes[0].plot(times, radial.mean(axis=1), color=color, linestyle=style, label=label)
        axes[0].fill_between(times, *np.quantile(radial, [0.1, 0.9], axis=1),
                             color=color, alpha=0.12, linewidth=0)
    axes[0].set(title="(a) Coupled radial statistics", xlabel="time", ylabel="radius")
    axes[0].legend(loc="best")
    bounds = (min(true_dist.min(), learned_dist.min()), max(true_dist.max(), learned_dist.max()))
    bins = np.linspace(*bounds, 40)
    axes[1].hist(true_dist, bins=bins, density=True, histtype="step",
                 color=TRUE_COLOR, label="true")
    axes[1].hist(learned_dist, bins=bins, density=True, histtype="step",
                 color=ESTIMATE_COLOR, label="weak LASSO")
    axes[1].set(title="(b) Final pair distances", xlabel="pair distance", ylabel="density")
    axes[1].legend(loc="best")
    for axis in axes: finish_axes(axis)
    save_pdf(fig, destination)
    return {"repetition": 0, "trajectory_relative_error": trajectory_error}


def make_sweeps(statistics, destination):
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.15), constrained_layout=True)
    labels = (
        ("process_sigma", "(a) Process noise", r"process noise $\sigma$"),
        ("particle_number", "(b) Particle count", r"particles $N$"),
        ("sampled_pairs", "(c) Sampled pairs", r"sampled pairs $P$"),
    )
    for axis, (sweep, title, xlabel) in zip(axes, labels):
        items = statistics[sweep]
        x_lasso, mean_lasso = _errorbar(
            axis, items, "lasso", ESTIMATE_COLOR, "^", "weak LASSO"
        )
        _errorbar(axis, items, "ols", TRUE_COLOR, "o", "weak OLS")
        axis.set(title=title, xlabel=xlabel, yscale="log")
        if sweep in ("particle_number", "sampled_pairs"):
            reference = mean_lasso[0] * (x_lasso / x_lasso[0]) ** (-0.5)
            axis.plot(x_lasso, reference, color="0.35", linestyle=":", linewidth=1,
                      label=r"$-1/2$ reference")
        if sweep == "particle_number":
            axis.set_xticks(x_lasso, [str(int(value)) for value in x_lasso])
        if sweep == "sampled_pairs":
            axis.set_xscale("log")
        if sweep == "process_sigma": axis.set_xlim(left=0)
        if axis is axes[0]: axis.set_ylabel("coefficient error")
        axis.legend(loc="best", fontsize=7)
        finish_axes(axis)
    save_pdf(fig, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluation-config", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    destination = args.output or run_dir / "publication_figures"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("status") not in {"scientific_completed", "production_candidate_completed"} \
            or manifest.get("smoke") or manifest.get("smoke_only"):
        raise ValueError("figures require a completed non-smoke scientific run")
    spec = read_json(run_dir / "frozen_protocol.json")
    evaluation_override = None
    if args.evaluation_config is not None:
        evaluation_override = read_json(args.evaluation_config)
        if evaluation_override.get("task") != "localized10d_same_population_evaluation" \
                or evaluation_override.get("protocol_version") != 1:
            raise ValueError("invalid localized 10D evaluation override")
        configured_source = (ROOT / evaluation_override["source_run"]).resolve()
        if configured_source != run_dir:
            raise ValueError("evaluation override refers to a different scientific run")
        if int(evaluation_override["rollout_particles"]) != int(spec["model"]["particles"]):
            raise ValueError("same-population rollout must use the fitting particle count")
        spec = json.loads(json.dumps(spec))
        spec["evaluation"]["rollout_particles"] = int(evaluation_override["rollout_particles"])
        spec["evaluation"]["rollout_horizon"] = float(evaluation_override["rollout_horizon"])
    records = read_json(run_dir / "records.json")
    statistics, slopes = _sweep_statistics(records, spec)
    recovery = make_recovery(records, spec, destination / "localized10d_recovery.pdf")
    dynamics = make_dynamics(records, spec, destination / "localized10d_coupled_dynamics.pdf")
    make_sweeps(statistics, destination / "localized10d_robustness_sweeps.pdf")
    metadata = {
        "source_run": str(run_dir),
        "source_manifest_sha256": digest(run_dir / "source_manifest.json"),
        "source_run_manifest_sha256": digest(run_dir / "manifest.json"),
        "source_records_sha256": digest(run_dir / "records.json"),
        "generator_sha256": digest(Path(__file__)),
        "evaluation_override": evaluation_override,
        "evaluation_override_sha256": (
            digest(args.evaluation_config) if args.evaluation_config is not None else None
        ),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
                      "unit": "paired repetition", "interval": "percentile 95%"},
        "statistics": statistics, "log_log_slopes": slopes,
        "displayed_recovery": recovery, "displayed_dynamics": dynamics,
    }
    write_path = destination / "metadata.json"
    write_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    hashes = {path.name: digest(path) for path in destination.glob("*") if path.is_file()}
    (destination / "sha256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(destination), "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
