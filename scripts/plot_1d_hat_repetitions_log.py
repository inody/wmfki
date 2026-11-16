"""Also display the finalized repetition summary on a logarithmic vertical axis without changing any values."""

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
from scripts.summarize_1d_hat_repetitions import COLORS, MARKERS, METRICS


def plot(summary_path, destination):
    summary_path, destination = Path(summary_path), Path(destination)
    summary = json.loads(summary_path.read_text())
    names = [name for name in summary["dictionaries"] if name.startswith("hat_M")]
    sizes = [int(name.removeprefix("hat_M")) for name in names]
    titles = ["Interaction-force relative error", "Coupled trajectory relative error", "Drift relative error"]
    # Do not silently omit zero or missing values from a log plot; this view requires complete results with all values positive.
    for dictionary in summary["dictionaries"].values():
        for key in METRICS[:3]:
            stat = dictionary["metrics"][key]
            if stat["failure_count"] or any(value <= 0 for value in stat["values"]):
                raise ValueError("log view requires complete positive values; use the linear view instead")
    destination.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, key, title in zip(axes, METRICS[:3], titles):
        stats = [summary["dictionaries"][name]["metrics"][key] for name in names]
        for values in np.asarray([stat["values"] for stat in stats]).T:
            ax.plot(range(len(sizes)), values, color="0.7", alpha=0.45, linewidth=0.65)
        ax.plot(range(len(sizes)), [stat["mean"] for stat in stats], color="black", linewidth=1.2)
        for i, stat in enumerate(stats):
            lower, upper = stat["mean_ci95"]
            ax.vlines(i, lower, upper, color=COLORS[i], linewidth=2)
            ax.plot([i - 0.06, i + 0.06], [lower, lower], color=COLORS[i])
            ax.plot([i - 0.06, i + 0.06], [upper, upper], color=COLORS[i])
            ax.plot(i, stat["mean"], marker=MARKERS[i], color=COLORS[i], markersize=7)
        baseline = summary["dictionaries"]["baseline_six_atoms"]["metrics"][key]
        ax.axhline(baseline["mean"], color="tab:purple", linestyle="--")
        ax.axhspan(*baseline["mean_ci95"], color="tab:purple", alpha=0.1)
        ax.set_xticks(range(len(sizes)), sizes)
        ax.set(xlabel="Dictionary size M", title=title, yscale="log")
        ax.set_box_aspect(3 / 4)
        ax.grid(axis="y", alpha=0.2)
    handles = [Line2D([0], [0], color="0.7", label="Individual repetitions"),
               Line2D([0], [0], color="black", marker="o", label="Arithmetic mean; bars: pointwise bootstrap 95% CI"),
               Line2D([0], [0], color="tab:purple", linestyle="--", label="6-atom mean; band: 95% CI")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8.5)
    fig.suptitle(f"Nested hat spaces: N={summary['particle_count']}, {summary['planned_repetitions']} fixed independent repetitions")
    fig.savefig(destination / "nested_dictionary_errors_log.png", dpi=180)
    plt.close(fig)
    write_json(destination / "provenance.json", {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "operation": "Only the y-axis scale changes; all repetitions, means and intervals are unchanged.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    plot(args.summary, args.out_dir)
