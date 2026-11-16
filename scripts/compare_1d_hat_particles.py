"""Compare the saved 100- and 1000-particle experiments without retraining or configuration selection."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_1d_hat_experiment import write_json


def check_protocols(small, large):
    left, right = deepcopy(small), deepcopy(large)
    if left["simulation"].pop("N") != 100 or right["simulation"].pop("N") != 1000:
        raise ValueError("comparison requires N=100 and N=1000, in that order")
    if left != right:
        raise ValueError("protocols differ beyond the approved particle count")


def compare(small_path, large_path, out_dir):
    paths = [Path(small_path), Path(large_path)]
    reports = [json.loads(path.read_text()) for path in paths]
    check_protocols(reports[0]["protocol"], reports[1]["protocol"])
    manifests = []
    for path, report in zip(paths, reports):
        if report["smoke_only"] or report["status"] not in ("completed", "completed_with_fit_failures"):
            raise ValueError("completed scientific runs are required")
        manifest_path = Path(report["assembly_directory"]) / "manifest.json"
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != report["assembly_manifest_sha256"]:
            raise ValueError("assembly manifest hash mismatch")
        frozen = path.parent / "frozen_selections.json"
        if hashlib.sha256(frozen.read_bytes()).hexdigest() != report["frozen_selections_sha256"]:
            raise ValueError("frozen selection hash mismatch")
        manifests.append(json.loads(manifest_path.read_text()))
    if manifests[0]["test_functions"] != manifests[1]["test_functions"]:
        raise ValueError("test functions changed")
    names = list(reports[0]["results"])
    if set(names) != set(reports[1]["results"]):
        raise ValueError("candidate dictionaries changed")
    rows = []
    for report, manifest in zip(reports, manifests):
        for name in names:
            item = report["results"][name]
            design = manifest["splits"]["train"][0]["designs"][name]
            row = {"N": report["protocol"]["simulation"]["N"], "dictionary": name,
                   "status": item["status"], "columns": design["shape"][1],
                   "nonzero_columns": design["shape"][1] - len(design["zero_columns"]),
                   "train_rank": design["raw"]["rank"],
                   "assembly_seconds": sum(split[0]["designs"][name]["assembly_seconds"]
                                           for split in manifest["splits"].values())}
            if item["status"] == "evaluated":
                if len(item["test_metrics"]) != 1:
                    raise ValueError("this comparison reports one split, not an average over repetitions")
                row.update(item["test_metrics"][0])
                selected = item["selection"]["selected"]
                row.update({"nonzero_coefficients": selected["nonzero_reported"],
                            "alpha_sk": selected["alpha_sk"],
                            "alpha_over_max": selected["alpha_over_max"],
                            "validation_mse": selected["validation_mse"],
                            "test_weak_residual_mse": item["test_weak_residual_mse"],
                            "selected_at_grid_boundary": item["selection"]["selected_at_grid_boundary"],
                            "unconverged_candidates": sum(not candidate["converged"]
                                                          for candidate in item["selection"]["path"])})
            rows.append(row)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    summary = {"status": "compared_without_refitting", "rows": rows,
               "reports": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in paths],
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "limitations": ["One train/validation/test trajectory per N; no uncertainty estimate.",
                               "Same seeds do not imply nested particles or identical Brownian paths across N.",
                               "Alpha values are chosen separately using the unchanged validation rule.",
                               "No model is chosen from held-out performance in this comparison."]}
    write_json(out_dir / "comparison.json", summary)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    metrics = [("force_error_weighted_rel", "Distance-weighted force relative error"),
               ("drift_error_rel", "Drift relative error"),
               ("trajectory_rmse", "Common-noise trajectory RMSE"),
               ("nonzero_coefficients", "Nonzero coefficients")]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for N, color in ((100, "tab:blue"), (1000, "tab:orange")):
            values = [next(row for row in rows if row["N"] == N and row["dictionary"] == name).get(metric)
                      for name in names]
            values = [np.nan if value is None else value for value in values]
            ax.plot(range(len(names) - 1), values[:-1], "o-", label=f"N={N}", color=color)
            ax.plot(len(names) - 1, values[-1], "s", color=color)
        ax.axvline(len(names) - 1.5, color="0.8", linewidth=0.8)
        ax.set_xticks(range(len(names)), [name.replace("hat_M", "Hat ").replace("baseline_six_atoms", "6-atom") for name in names])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_box_aspect(3 / 4)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Particle count only: fixed model, dictionary, weak tests, and LASSO selection rule")
    fig.savefig(out_dir / "particle_comparison.png", dpi=180)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-report", type=Path, required=True)
    parser.add_argument("--large-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(args.small_report, args.large_report, args.out_dir)
    print(args.out_dir)
