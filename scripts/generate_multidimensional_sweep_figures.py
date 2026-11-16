"""Generate separate 1x3 robustness figures for the 2D and 10D appendices."""

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
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR,
    REFERENCE_COLOR,
    TERTIARY_COLOR,
    TRUE_COLOR,
    apply_manuscript_style,
    finish_axes,
    save_pdf,
)


METHOD_STYLE = {
    "ols": {
        "label": "weak OLS", "color": TRUE_COLOR, "marker": "o", "linestyle": "-"
    },
    "lasso": {
        "label": "weak LASSO", "color": ESTIMATE_COLOR, "marker": "^", "linestyle": "--"
    },
    "strong_fd_ols": {
        "label": "strong FD--OLS", "color": TERTIARY_COLOR,
        "marker": "s", "linestyle": "-."
    },
    "strong_fd_lasso": {
        "label": "strong FD--LASSO", "color": TERTIARY_COLOR,
        "marker": "s", "linestyle": "-."
    },
}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_mean_interval(values, *, seed, draws=10000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("figure inputs must contain all predeclared finite repetitions")
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(values.mean()), float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def predator_groups(records, strong_records=None):
    order = ("observation_noise", "observation_dt", "prey_particles")
    result = {}
    for panel_index, sweep in enumerate(order):
        rows = [row for row in records if row["condition"]["sweep"] == sweep]
        values = sorted({float(row["condition"]["value"]) for row in rows})
        result[sweep] = {}
        for method in ("ols", "lasso"):
            series = []
            for value_index, value in enumerate(values):
                subset = []
                for row in rows:
                    if float(row["condition"]["value"]) != value:
                        continue
                    methods = row["methods"]
                    if "P50" in methods:
                        methods = methods["P50"]
                    subset.append(methods[method]["held_out_drift"]["species_balanced_relative"])
                mean, lower, upper = bootstrap_mean_interval(
                    subset, seed=91000 + 1000 * panel_index + value_index
                )
                series.append((value, mean, lower, upper, len(subset)))
            result[sweep][method] = series
    if strong_records is not None:
        strong_methods = {row.get("method") for row in strong_records}
        if strong_methods == {"strong_forward_difference_ols"}:
            strong_key = "strong_fd_ols"
        elif strong_methods == {"strong_forward_difference_lasso_validation_selected"}:
            strong_key = "strong_fd_lasso"
        else:
            raise ValueError(f"unexpected strong predator methods: {strong_methods}")
        series = []
        for value_index, value in enumerate(
            sorted({float(row["condition"]["value"]) for row in strong_records})
        ):
            subset = [
                row["held_out_drift"]["species_balanced_relative"]
                for row in strong_records
                if float(row["condition"]["value"]) == value
            ]
            mean, lower, upper = bootstrap_mean_interval(
                subset, seed=91000 + value_index
            )
            series.append((value, mean, lower, upper, len(subset)))
        result["observation_noise"][strong_key] = series
    return result


def morse_groups(records):
    order = ("process_sigma", "particle_number", "sampled_pairs")
    result = {}
    for panel_index, sweep in enumerate(order):
        rows = [row for row in records if row["sweep"] == sweep]
        values = sorted({float(row["sweep_value"]) for row in rows})
        result[sweep] = {}
        for method in ("ols", "lasso"):
            series = []
            for value_index, value in enumerate(values):
                subset = [
                    row["latent_drift_relative_error"] for row in rows
                    if row["method"] == method and float(row["sweep_value"]) == value
                ]
                mean, lower, upper = bootstrap_mean_interval(
                    subset, seed=92000 + 1000 * panel_index + value_index
                )
                series.append((value, mean, lower, upper, len(subset)))
            result[sweep][method] = series
    return result


def plot_row(groups, *, order, titles, xlabels, output, zero_symlog=False):
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), constrained_layout=True)
    for panel_index, (axis, sweep, title, xlabel) in enumerate(zip(axes, order, titles, xlabels)):
        all_positive = True
        for method in ("lasso", "ols", "strong_fd_lasso", "strong_fd_ols"):
            if method not in groups[sweep]:
                continue
            style = METHOD_STYLE[method]
            data = np.asarray(groups[sweep][method], dtype=float)
            x, mean, lower, upper = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            all_positive = all_positive and bool(np.all(mean > 0) and np.all(lower > 0))
            axis.plot(
                x, mean, color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], label=style["label"]
            )
            axis.fill_between(x, lower, upper, color=style["color"], alpha=0.16, linewidth=0)
        if all_positive:
            axis.set_yscale("log")
        if zero_symlog and panel_index == 0:
            positive = np.asarray([v[0] for v in groups[sweep]["lasso"] if v[0] > 0])
            axis.set_xscale("symlog", linthresh=float(positive.min() / 2))
            axis.set_xlim(left=0)
            shown = np.unique(np.concatenate(([0.0], positive[[0, -1]])))
            axis.xaxis.set_major_locator(FixedLocator(shown))
            axis.xaxis.set_major_formatter(FixedFormatter(
                ["0" if value == 0 else f"{value:g}" for value in shown]
            ))
            axis.xaxis.set_minor_locator(NullLocator())
            axis.xaxis.set_minor_formatter(NullFormatter())
        elif sweep not in {"prey_particles", "particle_number"}:
            axis.set_xscale("log")
            if sweep == "observation_dt":
                shown = np.asarray([0.005, 0.01, 0.04])
                axis.xaxis.set_major_locator(FixedLocator(shown))
                axis.xaxis.set_major_formatter(FixedFormatter(["0.005", "0.01", "0.04"]))
                axis.xaxis.set_minor_locator(NullLocator())
                axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        if panel_index == 0:
            axis.set_ylabel("drift error")
            axis.legend(loc="best")
        axis.grid(False)
        finish_axes(axis)
    save_pdf(fig, output)


def generate(predator_dir, morse_dir, output_dir, predator_strong_dir=None):
    predator_dir, morse_dir, output_dir = map(Path, (predator_dir, morse_dir, output_dir))
    predator_strong_dir = None if predator_strong_dir is None else Path(predator_strong_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predator_records = read_json(predator_dir / "records.json")
    predator_strong_records = (
        None if predator_strong_dir is None
        else read_json(predator_strong_dir / "records.json")
    )
    morse_records = read_json(morse_dir / "records.json")
    predator_output = output_dir / "predator_prey_robustness_sweeps.pdf"
    morse_output = output_dir / "morse10d_robustness_sweeps.pdf"
    plot_row(
        predator_groups(predator_records, predator_strong_records),
        order=("observation_noise", "observation_dt", "prey_particles"),
        titles=("(a) Observation noise", "(b) Observation interval", "(c) Prey count"),
        xlabels=(r"observation noise $\sigma_{\rm obs}$", r"observation interval $\Delta t$", r"prey particles $N$"),
        output=predator_output,
        zero_symlog=True,
    )
    plot_row(
        morse_groups(morse_records),
        order=("process_sigma", "particle_number", "sampled_pairs"),
        titles=("(a) Process noise", "(b) Particle count", "(c) Sampled pairs"),
        xlabels=(r"process noise $\sigma$", r"particles $N$", r"sampled pairs $P$"),
        output=morse_output,
        zero_symlog=True,
    )
    metadata = {
        "predator_output": str(predator_output),
        "morse_output": str(morse_output),
        "predator_records_sha256": digest(predator_dir / "records.json"),
        "predator_strong_records_sha256": (
            None if predator_strong_dir is None
            else digest(predator_strong_dir / "records.json")
        ),
        "morse_records_sha256": digest(morse_dir / "records.json"),
        "predator_figure_sha256": digest(predator_output),
        "morse_figure_sha256": digest(morse_output),
        "bootstrap_draws": 10000,
        "interval": "paired repetition percentile-bootstrap 95% interval for arithmetic mean",
        "missing_policy": "refuse figure generation if any predeclared value is missing or nonfinite",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predator-dir", type=Path, required=True)
    parser.add_argument("--morse-dir", type=Path, required=True)
    parser.add_argument("--predator-strong-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(
        args.predator_dir, args.morse_dir, args.output_dir,
        predator_strong_dir=args.predator_strong_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
