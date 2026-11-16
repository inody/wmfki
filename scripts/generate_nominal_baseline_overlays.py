"""Generate the nominal estimator overlays used by Figures 3, 4, 6, and 7."""

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
from predator_prey import (
    channel_kernels, make_predator_prey_bases, simulate_predator_prey,
)
from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, TRUE_COLOR,
    apply_manuscript_style, finish_axes, save_pdf,
)
from scripts.run_localized10d_pilot import _simulate_coupled
from scripts.run_morse10d_sweeps import repetition_plan
from scripts.run_predator_prey_application import model_objects

METHODS = (
    "weak_lasso",
    "weak_ols",
    "strong_forward_difference_lasso_validation_selected",
)
STYLES = {
    "weak_lasso": ("weak LASSO", ESTIMATE_COLOR, "--"),
    "weak_ols": ("weak OLS", SECONDARY_COLOR, ":"),
    "strong_forward_difference_lasso_validation_selected": (
        "strong FD--LASSO", TERTIARY_COLOR, "-."
    ),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method_rows(records, example, repetition=None):
    rows = [row for row in records if row["example"] == example and row["method"] in METHODS]
    if repetition is not None:
        rows = [row for row in rows if int(row["repetition"]) == int(repetition)]
    return rows


def split_predator(theta, prey_columns):
    theta = np.asarray(theta, dtype=float)
    return theta[:prey_columns], theta[prey_columns:2 * prey_columns], theta[2 * prey_columns:]


def predator_profiles(protocol, records, destination):
    model, dictionary = protocol["model"], protocol["dictionary"]
    prey_basis, predator_basis, _ = make_predator_prey_bases(
        model["delta"], dictionary["prey_exponential_scales"],
        dictionary["predator_inverse_powers"],
    )
    prey_columns = len(dictionary["prey_prey_true"])
    truth_channels = (
        np.asarray(dictionary["prey_prey_true"]),
        np.asarray(dictionary["prey_predator_true"]),
        np.asarray(dictionary["predator_prey_true"]),
    )
    radius = np.linspace(
        protocol["evaluation"]["profile_radius_min"],
        protocol["evaluation"]["profile_radius_max"],
        protocol["evaluation"]["profile_grid_size"],
    )
    displacement = np.column_stack([radius, np.zeros_like(radius)])
    prey_features = prey_basis(displacement)[:, :, 0]
    predator_features = predator_basis(displacement)[:, :, 0]
    def curves(channels):
        return (
            prey_features @ channels[0], prey_features @ channels[1],
            predator_features @ channels[2],
        )
    truth = curves(truth_channels)
    medians = {}
    for method in METHODS:
        method_curves = [
            curves(split_predator(row["coefficients"], prey_columns))
            for row in records if row["method"] == method
        ]
        medians[method] = np.median(np.asarray(method_curves), axis=0)
    titles = (r"(a) Prey--prey", r"(b) Predator on prey", r"(c) Prey on predator")
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), constrained_layout=True)
    for index, axis in enumerate(axes):
        axis.plot(radius, truth[index], color=TRUE_COLOR, linewidth=1.8, label="true")
        for method in METHODS:
            label, color, linestyle = STYLES[method]
            axis.plot(radius, medians[method][index], color=color,
                      linestyle=linestyle, linewidth=1.35, label=label)
        axis.axhline(0, color="0.4", linewidth=.65)
        axis.set(title=titles[index], xlabel=r"pairwise distance $r$")
        if index == 0:
            axis.set_ylabel(r"radial interaction $r\phi(r)$")
            axis.legend(loc="best", fontsize=6.6)
        finish_axes(axis)
    save_pdf(fig, destination)


def predator_dynamics(protocol, records, predator_run, destination):
    prey_basis, predator_basis, _, truth_channels, _ = model_objects(protocol)
    nominal = predator_run / "nominal_application"
    arrays = np.load(nominal / "display_trajectory.npz")
    report = read_json(nominal / "trajectory_report.json")
    nominal_records = read_json(nominal / "records.json")
    true_x, true_z, time = arrays["true_prey"], arrays["true_predator"], arrays["time"]
    trajectories = {
        "true": (true_x, true_z),
        "weak_lasso": (arrays["P50_lasso_prey"], arrays["P50_lasso_predator"]),
        "weak_ols": (arrays["P50_ols_prey"], arrays["P50_ols_predator"]),
    }
    metrics = {
        "weak_lasso": report["methods"]["P50"]["lasso"],
        "weak_ols": report["methods"]["P50"]["ols"],
    }
    prey_columns = len(truth_channels[0])
    method = "strong_forward_difference_lasso_validation_selected"
    row = next(item for item in records if item["method"] == method and item["repetition"] == 0)
    channels = split_predator(row["coefficients"], prey_columns)
    kernels = channel_kernels(*channels, prey_basis, predator_basis)
    prey, predator, fitted_time, diag = simulate_predator_prey(
        particle_count=true_x.shape[2], steps=time.size - 1,
        dt=float(report["native_dt"]), repetitions=1, kernels=kernels,
        seed=int(nominal_records[0]["seeds"]["test"]),
        solver=protocol["model"]["solver"],
        initial_prey=true_x[:, 0], initial_predator=true_z[:, 0],
    )
    if not np.array_equal(fitted_time, time):
        raise RuntimeError("predator display time grid changed")
    trajectories[method] = (prey, predator)
    finite = bool(all(diag["finite"])) and np.all(np.isfinite(prey)) and np.all(np.isfinite(predator))
    metrics[method] = {
        "finite": finite,
        "failed_steps": diag["failed_steps"],
        "trajectory_error": (
            float(np.sqrt(
                0.5 * (
                    np.sum((prey - true_x) ** 2) / np.sum(true_x ** 2)
                    + np.sum((predator - true_z) ** 2) / np.sum(true_z ** 2)
                )
            )) if finite else None
        ),
    }
    display_times = [0.0, 1.0, 2.0, 3.0]
    indices = [int(np.argmin(np.abs(time - value))) for value in display_times]

    def draw_snapshot(axis, key, time_index):
        prey_path, predator_path = trajectories[key]
        prey, predator = prey_path[0, time_index], predator_path[0, time_index, 0]
        finite = np.all(np.isfinite(prey)) and np.all(np.isfinite(predator))
        if finite:
            axis.scatter(prey[:, 0], prey[:, 1], s=2.8, color=TRUE_COLOR,
                         alpha=.68, linewidths=0)
            axis.scatter(predator[0], predator[1], s=24, color=ESTIMATE_COLOR,
                         marker="x", linewidths=1.2)
        else:
            axis.text(.5, .5, "diverged", transform=axis.transAxes,
                      ha="center", va="center")
        axis.set_aspect("equal")
        axis.set_xlim(-1.55, 1.55); axis.set_ylim(-1.55, 1.55)
        axis.set_xticks([-1, 0, 1]); axis.set_yticks([-1, 0, 1])
        finish_axes(axis)

    rows = ("true",) + METHODS
    row_labels = {
        "true": "true",
        **{key: STYLES[key][0] for key in METHODS},
    }
    apply_manuscript_style()
    fig, axes = plt.subplots(4, 4, figsize=(7.15, 6.3), constrained_layout=True)
    for row_index, key in enumerate(rows):
        for column, (time_value, time_index) in enumerate(zip(display_times, indices)):
            draw_snapshot(axes[row_index, column], key, time_index)
            if row_index == 0:
                axes[row_index, column].set_title(rf"$t={time_value:g}$")
            if column == 0:
                axes[row_index, column].set_ylabel(row_labels[key] + "\n" + r"$x_2$")
            if row_index == len(rows) - 1:
                axes[row_index, column].set_xlabel(r"$x_1$")
    save_pdf(fig, destination)
    return metrics


def localized_recovery(spec, records, destination):
    rows = method_rows(records, "localized_10d", repetition=0)
    coefficients = {row["method"]: np.asarray(row["coefficients"], dtype=float) for row in rows}
    target = np.asarray(spec["model"]["interaction"]["coefficients"], dtype=float)
    basis, meta = make_radial_basis(spec["dictionary"])
    radius = np.linspace(*spec["evaluation"]["profile_radius"],
                         int(spec["evaluation"]["profile_grid_size"]))
    radial_basis = basis(radius)
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), constrained_layout=True)
    axes[0].plot(radius, radius * (radial_basis @ target), color=TRUE_COLOR, label="true")
    for method in METHODS:
        label, color, linestyle = STYLES[method]
        axes[0].plot(radius, radius * (radial_basis @ coefficients[method]),
                     color=color, linestyle=linestyle, label=label)
    axes[0].axhline(0, color="0.4", linewidth=.7)
    axes[0].set(title="(a) Radial interaction", xlabel="pairwise distance $r$",
                ylabel=r"radial interaction $r\phi(r)$")
    axes[0].legend(loc="best", fontsize=6.8)
    x = np.arange(target.size); width = .19
    offsets = (-1.5, -.5, .5, 1.5)
    series = [("true", target, TRUE_COLOR)] + [
        (STYLES[method][0], coefficients[method], STYLES[method][1]) for method in METHODS
    ]
    for offset, (label, values, color) in zip(offsets, series):
        axes[1].bar(x + offset * width, values, width, color=color, label=label)
    axes[1].set_xticks(x, [f"{center:g}" for center in meta["centers"]], rotation=30)
    axes[1].set(title="(b) Dictionary coefficients", xlabel="interaction radius",
                ylabel="coefficient")
    axes[1].legend(loc="best", fontsize=6.5)
    for axis in axes: finish_axes(axis)
    save_pdf(fig, destination)


def localized_dynamics(spec, records, destination):
    rows = method_rows(records, "localized_10d", repetition=0)
    coefficients = {row["method"]: np.asarray(row["coefficients"], dtype=float) for row in rows}
    plan = repetition_plan(spec)[0]
    trajectories = {}
    metrics = {}
    truth = None
    for method in METHODS:
        true_path, learned, error = _simulate_coupled(spec, coefficients[method], plan["test_seed"])
        if truth is None:
            truth = true_path
        elif not np.array_equal(truth, true_path):
            raise RuntimeError("localized coupled truth changed across methods")
        trajectories[method] = learned
        metrics[method] = {"trajectory_relative_error": error}
    times = np.arange(truth.shape[0]) * spec["model"]["dt"]
    all_paths = {"true": truth, **trajectories}
    radial = {
        key: np.linalg.norm(path - path.mean(axis=1, keepdims=True), axis=-1)
        for key, path in all_paths.items()
    }
    tri = np.triu_indices(truth.shape[1], k=1)
    distances = {
        key: np.linalg.norm(path[-1, :, None, :] - path[-1, None, :, :], axis=-1)[tri]
        for key, path in all_paths.items()
    }
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), constrained_layout=True)
    axes[0].plot(times, radial["true"].mean(axis=1), color=TRUE_COLOR, label="true")
    axes[0].fill_between(times, *np.quantile(radial["true"], [.1, .9], axis=1),
                         color=TRUE_COLOR, alpha=.10, linewidth=0)
    for method in METHODS:
        label, color, linestyle = STYLES[method]
        axes[0].plot(times, radial[method].mean(axis=1), color=color,
                     linestyle=linestyle, label=label)
    axes[0].set(title="(a) Coupled radial statistics", xlabel="time", ylabel="radius")
    axes[0].legend(loc="best", fontsize=6.8)
    bounds = (min(values.min() for values in distances.values()),
              max(values.max() for values in distances.values()))
    bins = np.linspace(*bounds, 40)
    axes[1].hist(distances["true"], bins=bins, density=True, histtype="step",
                 color=TRUE_COLOR, label="true")
    for method in METHODS:
        label, color, _ = STYLES[method]
        axes[1].hist(distances[method], bins=bins, density=True, histtype="step",
                     color=color, label=label)
    axes[1].set(title="(b) Final pair distances", xlabel="pair distance", ylabel="density")
    axes[1].legend(loc="best", fontsize=6.8)
    for axis in axes: finish_axes(axis)
    save_pdf(fig, destination)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regularized-run", type=Path, required=True)
    parser.add_argument("--predator-run", type=Path, required=True)
    parser.add_argument("--localized-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regularized = args.regularized_run.resolve()
    predator_run = args.predator_run.resolve()
    localized_run = args.localized_run.resolve()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    records = read_json(regularized / "records.json")
    predator_protocol = read_json(predator_run / "effective_protocol.json")
    localized_spec = read_json(localized_run / "frozen_protocol.json")
    predator_rows = method_rows(records, "predator_prey")
    if len(predator_rows) != 60 or len(method_rows(records, "localized_10d")) != 60:
        raise ValueError("regularized nominal records are incomplete")
    predator_profiles(predator_protocol, predator_rows, output / "predator_prey_profiles.pdf")
    predator_metrics = predator_dynamics(
        predator_protocol, predator_rows, predator_run,
        output / "predator_prey_dynamics.pdf"
    )
    localized_recovery(localized_spec, records, output / "localized10d_recovery.pdf")
    localized_metrics = localized_dynamics(
        localized_spec, records, output / "localized10d_coupled_dynamics.pdf"
    )
    metadata = {
        "observation_noise": {"predator_prey": 0.01, "localized_10d": 0.01},
        "regularized_records_sha256": digest(regularized / "records.json"),
        "predator_protocol_sha256": digest(predator_run / "effective_protocol.json"),
        "localized_protocol_sha256": digest(localized_run / "frozen_protocol.json"),
        "predator_dynamics": predator_metrics,
        "localized_dynamics": localized_metrics,
    }
    for path in output.glob("*.pdf"):
        metadata.setdefault("figure_sha256", {})[path.name] = digest(path)
    json_default = lambda value: value.item() if isinstance(value, np.generic) else value
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    print(json.dumps({"output": str(output), "metadata": metadata},
                     indent=2, default=json_default))


if __name__ == "__main__":
    main()
