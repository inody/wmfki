"""Run the production Generic 1D sampled-pair candidate at P=50N over R=100.

The runner reuses the current 100 predeclared frozen trajectory triplets and
held-out evaluation statistics. Training and validation weak systems use
independent sampled ordered pairs. Matched full-pair LASSO/OLS outputs are
read-only comparison inputs. Current figures and manuscript files are not edited.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import make_kernel_basis
from scripts.hat_sweep_numerics import evaluate_model
from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR, GRID_COLOR, REFERENCE_COLOR, TRUE_COLOR,
    apply_manuscript_style, finish_axes,
)
from scripts.prepare_1d_hat_experiment import array_hash, write_json
from scripts.run_1d_hat_sampled_pair_ratio_sweep_R5 import (
    candidates,
    compact_lasso_record,
    compact_ols_record,
    digest,
    job_plan,
    load_sources,
    read_json,
    validate_sources,
    verify_artifacts,
    worker,
)
from scripts.run_1d_hat_dictionary_size_R100 import load_data
from scripts.run_1d_hat_sweeps import load_statistics, save_statistics
from simulate import make_true_kernel

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat_sampled_p50n_R100.json"
SOURCES = [
    "features.py", "kernels.py", "simulate.py", "utils.py",
    "scripts/prepare_1d_hat_experiment.py", "scripts/run_1d_hat_lasso.py",
    "scripts/hat_sweep_numerics.py", "scripts/run_1d_hat_sweeps.py",
    "scripts/run_1d_hat_dictionary_size_R100.py",
    "scripts/run_1d_hat_sampled_pair_ratio_sweep_R5.py",
    "scripts/manuscript_plot_style.py",
    "scripts/run_1d_hat_sampled_p50n_R100.py",
    "configs/experiments/one_dimensional_hat_sampled_p50n_R100.json",
]
PANEL_BOX_ASPECT = 3 / 4


def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_indices(spec: dict) -> list[int]:
    block = spec["reference_indices"]
    return list(range(int(block["start"]), int(block["stop_exclusive"])))


def worker_spec(spec: dict) -> dict:
    result = dict(spec)
    result.update(
        source_lasso_frozen_protocol_sha256=spec["source_lasso_hashes"]["frozen_protocol.json"],
        source_ols_frozen_protocol_sha256=spec["source_ols_hashes"]["frozen_protocol.json"],
        reference_indices=reference_indices(spec),
        pair_ratios=[int(spec["pair_ratio"])],
        ratio_seed_stride=10,
    )
    return result


def validate_spec(spec: dict) -> None:
    if spec.get("task") != "one_dimensional_hat_sampled_p50n_R100" \
            or spec.get("protocol_version") != 1:
        raise ValueError("invalid production sampled-pair protocol")
    if spec.get("dictionary_sizes") != [9, 11, 14, 17, 22, 27, 33, 42, 52, 65]:
        raise ValueError("dictionary grid changed")
    if spec.get("repetitions") != 100 or reference_indices(spec) != list(range(100)):
        raise ValueError("production run requires all 100 predeclared repetitions")
    if spec.get("pair_ratio") != 50 or spec.get("pair_count") != 50000:
        raise ValueError("primary pair policy must be P=50N for N=1000")
    expected_sampling = {
        "ordered": True, "exclude_diagonal": True, "with_replacement": True,
        "resample_each_time": True, "shared_across_tests_and_atoms": True,
        "independent_training_validation": True,
    }
    if spec.get("pair_sampling") != expected_sampling:
        raise ValueError("pair sampling policy changed")
    if spec["training_pair_seed_offset"] == spec["validation_pair_seed_offset"]:
        raise ValueError("training and validation pair seeds must differ")
    if spec["bootstrap"] != {"samples": 10000, "seed": 985000,
                              "unit": "matched repetition"}:
        raise ValueError("bootstrap protocol changed")
    checks = spec["claim_checks"]
    if (checks["aggregate_lasso_force_rms_maximum_full_pair_multiplier"] != 1.25
            or checks["minimum_force_curve_spearman"] != 0.8):
        raise ValueError("claim thresholds changed")


def validate_locked_source_inputs(spec: dict, source: dict) -> None:
    for method in ("lasso", "ols"):
        batch = source[method]
        for relative, expected in spec[f"source_{method}_hashes"].items():
            actual = sha256(batch / relative)
            if actual != expected:
                raise ValueError(f"locked {method} input changed: {relative}")


def validate_current_artifact_audit(spec: dict) -> None:
    for relative, expected in spec["current_artifact_audit"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"current manuscript artifact changed before production run: {relative}")


def validate_locked_inputs(spec: dict, source: dict) -> None:
    validate_locked_source_inputs(spec, source)
    validate_current_artifact_audit(spec)


def load_reused_lasso_record(
    lasso_frozen: dict, key: str, lasso_directory: Path
) -> dict:
    """Load archived reused sizes, or an empty set for a fresh all-size run."""
    reused_entry = lasso_frozen.get("imported_reused_models", {}).get(key)
    if reused_entry is None:
        if lasso_frozen.get("source_mode") != "generated_in_this_run":
            raise KeyError(key)
        return {"results": {}, "source_model_records": []}
    reused_path = Path(lasso_directory) / reused_entry["path"]
    if sha256(reused_path) != reused_entry["sha256"]:
        raise ValueError("reused full-pair LASSO record changed")
    return read_json(reused_path)


def build_matched_sources_production(spec: dict, source: dict, destination: Path) -> tuple[list[dict], dict]:
    records, bundles = [], {}
    lasso_frozen, lasso_status = source["lasso_frozen"], source["lasso_status"]
    ols_status = source["ols_status"]
    statistics_directory = destination / "shared_evaluation_statistics"
    statistics_directory.mkdir()
    expected_names = {f"hat_M{size}" for size in spec["dictionary_sizes"]}
    for index in spec["reference_indices"]:
        key = f"run-{index}"
        reference_entry = lasso_frozen["references"][key]
        reference = Path(reference_entry["directory"])
        reference_report = verify_artifacts(reference, reference_entry["report_sha256"])

        new_entry = lasso_status["results"][key]
        new_directory = source["lasso"] / new_entry["directory"]
        new_report = verify_artifacts(new_directory, new_entry["report_sha256"])
        reused = load_reused_lasso_record(
            lasso_frozen, key, source["lasso"]
        )

        ols_entry = ols_status["results"][key]
        ols_directory = source["ols"] / ols_entry["directory"]
        ols_report = verify_artifacts(ols_directory, ols_entry["report_sha256"])
        lasso_results = {**reused["results"], **new_report["results"]}
        if set(lasso_results) != expected_names or set(ols_report["results"]) != expected_names:
            raise ValueError("matched full-pair records are incomplete")

        statistic_sources = []
        for directory, report in ((reference, reference_report), (new_directory, new_report)):
            path = directory / "evaluation_statistics.npz"
            if path.exists():
                expected = report["artifacts_sha256"]["evaluation_statistics.npz"]
                if sha256(path) != expected:
                    raise ValueError("held-out evaluation statistics changed")
                statistic_sources.append((path, expected))
        for model_source in reused.get("source_model_records", []):
            report_path = Path(model_source["path"])
            if report_path.name != "report.json" or not report_path.is_file():
                continue
            if sha256(report_path) != model_source["sha256"]:
                raise ValueError("reused model source report changed")
            source_report = read_json(report_path)
            path = report_path.parent / "evaluation_statistics.npz"
            if path.exists() and "evaluation_statistics.npz" in source_report.get("artifacts_sha256", {}):
                expected = source_report["artifacts_sha256"]["evaluation_statistics.npz"]
                if sha256(path) != expected:
                    raise ValueError("reused-size evaluation statistics changed")
                statistic_sources.append((path, expected))
        merged = {}
        for path, _ in statistic_sources:
            merged.update(load_statistics(path))
        if set(merged) != expected_names:
            raise ValueError(f"held-out statistics do not cover the dictionary grid at repetition {index}")
        merged_path = statistics_directory / f"run-{index + 1:03d}.npz"
        save_statistics(merged_path, merged)
        merged_hash = sha256(merged_path)

        for size in spec["dictionary_sizes"]:
            name = f"hat_M{size}"
            records.append({"mode": "full", "ratio": None, "repetition": index,
                            "M": size, "method": "weak_lasso",
                            **compact_lasso_record(lasso_results[name])})
            records.append({"mode": "full", "ratio": None, "repetition": index,
                            "M": size, "method": "weak_ols",
                            **compact_ols_record(ols_report["results"][name])})

        critical = {str(merged_path): merged_hash}
        for split in ("train", "validation", "test"):
            for filename in ("data.npz", "response.npy"):
                path = reference / split / filename
                if path.exists():
                    expected = reference_report["artifacts_sha256"][f"{split}/{filename}"]
                    if sha256(path) != expected:
                        raise ValueError("frozen split artifact changed")
                    critical[str(path)] = expected
        bundles[key] = {
            "reference_directory": str(reference),
            "reference_report_sha256": reference_entry["report_sha256"],
            "old_statistics": str(merged_path),
            "old_statistics_sha256": merged_hash,
            "new_statistics": str(merged_path),
            "new_statistics_sha256": merged_hash,
            "statistics_source_sha256": {str(path): expected for path, expected in statistic_sources},
            "critical_artifacts_sha256": critical,
        }
    write_json(destination / "matched_full_pair_records.json", {"records": records})
    write_json(destination / "frozen_reference_bundles.json", bundles)
    return records, bundles


def percentile_mean_ci(values: np.ndarray, bootstrap_indices: np.ndarray) -> list[float]:
    means = values[bootstrap_indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def aggregate(values: list[float], planned: int, bootstrap_indices: np.ndarray | None) -> dict:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    complete = len(finite) == planned
    if not len(finite):
        return {"planned_count": planned, "available_count": 0, "failure_count": planned,
                "values": [], "mean": None, "sample_sd": None, "median": None,
                "iqr": None, "rmse": None, "mean_ci95": None,
                "minimum": None, "maximum": None}
    return {
        "planned_count": planned, "available_count": int(len(finite)),
        "failure_count": int(planned - len(finite)), "values": finite.tolist(),
        "mean": float(np.mean(finite)) if complete else None,
        "sample_sd": float(np.std(finite, ddof=1)) if complete and planned > 1 else None,
        "median": float(np.median(finite)) if complete else None,
        "iqr": ([float(np.percentile(finite, 25)), float(np.percentile(finite, 75))]
                if complete else None),
        "rmse": float(np.sqrt(np.mean(finite ** 2))) if complete else None,
        "mean_ci95": (percentile_mean_ci(finite, bootstrap_indices)
                      if complete and bootstrap_indices is not None else None),
        "minimum": float(np.min(finite)) if complete else None,
        "maximum": float(np.max(finite)) if complete else None,
    }


def rankdata(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def sampled_records(destination: Path, status: dict, spec: dict) -> list[dict]:
    output = []
    for repetition in reference_indices(spec):
        key = f"ratio-{spec['pair_ratio']}-run-{repetition}"
        entry = status["results"][key]
        report_path = destination / entry["directory"] / "report.json"
        if sha256(report_path) != entry["report_sha256"]:
            raise ValueError(f"sampled report changed: {key}")
        report = read_json(report_path)
        if report["status"] != "completed":
            output.append({"mode": "sampled", "ratio": spec["pair_ratio"],
                           "repetition": repetition, "job_status": report["status"],
                           "job_error": report.get("error")})
            continue
        for size in spec["dictionary_sizes"]:
            name = f"hat_M{size}"
            design = report["results"][name]["design"]
            lasso = report["results"][name]["methods"]["weak_lasso"]
            selection = lasso["selection"]
            selected = selection.get("selected", {})
            output.append({
                "mode": "sampled", "ratio": spec["pair_ratio"],
                "repetition": repetition, "M": size, "method": "weak_lasso",
                "status": lasso["status"], "metrics": lasso.get("metrics", {}),
                "theta": lasso.get("theta"), "selection_status": selection.get("status"),
                "selected_alpha": selected.get("alpha_sk"),
                "selected_lambda": selected.get("lambda_summed_loss"),
                "selected_path_index": selected.get("index"),
                "selected_nonzero": selected.get("nonzero_reported"),
                "selected_iterations": selected.get("iterations"),
                "selected_converged": selected.get("converged"),
                "selected_at_grid_boundary": selection.get("selected_at_grid_boundary"),
                "path_nonconverged_count": sum(not row["converged"] for row in selection["path"]),
                "path_convergence_warning_count": sum(row["convergence_warnings"] for row in selection["path"]),
                "rank": design["rank"], "columns": design["columns"],
                "full_column_rank": design["full_column_rank"],
                "condition_number": design["condition_number"],
                "effective_condition_number": design["effective_condition_number"],
            })
            ols = report["results"][name]["methods"]["weak_ols"]
            fit = ols["fit"]
            output.append({
                "mode": "sampled", "ratio": spec["pair_ratio"],
                "repetition": repetition, "M": size, "method": "weak_ols",
                "status": ols["status"], "metrics": ols.get("metrics", {}),
                "theta": ols.get("theta"), "rank": fit["rank"],
                "columns": fit["columns"], "full_column_rank": fit["full_column_rank"],
                "condition_number": fit["condition_number"],
                "effective_condition_number": fit["effective_condition_number"],
                "train_mse": fit["train_mse"], "validation_mse": fit["validation_mse"],
            })
    return output


def select_rows(records: list[dict], mode: str, method: str, size: int | None = None) -> list[dict]:
    return [row for row in records if row.get("mode") == mode and row.get("method") == method
            and (size is None or row.get("M") == size)]


def values(rows: list[dict], metric: str | None = None, field: str | None = None) -> list[float]:
    if metric is not None:
        return [row.get("metrics", {}).get(metric) for row in rows]
    return [row.get(field) for row in rows]


def build_summary(destination: Path, status: dict, frozen: dict, spec: dict) -> tuple[dict, list[dict]]:
    full = read_json(destination / "matched_full_pair_records.json")["records"]
    sampled = sampled_records(destination, status, spec)
    records = full + sampled
    write_json(destination / "records.json", {"records": records})

    rng = np.random.default_rng(spec["bootstrap"]["seed"])
    bootstrap_indices = rng.integers(0, spec["repetitions"],
                                    size=(spec["bootstrap"]["samples"], spec["repetitions"]))
    summary = {
        "task": spec["task"], "status": "complete", "repetitions": spec["repetitions"],
        "pair_ratio": spec["pair_ratio"], "pair_count": spec["pair_count"],
        "dictionary_sizes": [], "global": {}, "diagnostics": {},
        "matched_full_pair_read_only": True,
    }
    table_rows = []
    for size in spec["dictionary_sizes"]:
        point = {"M": size, "sampled": {}, "full": {}}
        for mode in ("sampled", "full"):
            for method in ("weak_lasso", "weak_ols"):
                method_rows = select_rows(records, mode, method, size)
                result = {}
                for metric in spec["reporting"]["metrics"]:
                    result[metric] = aggregate(values(method_rows, metric=metric),
                                               spec["repetitions"], bootstrap_indices)
                for field in ("selected_lambda", "selected_alpha", "selected_path_index",
                              "selected_nonzero", "effective_condition_number"):
                    field_values = values(method_rows, field=field)
                    result[field] = aggregate(field_values, spec["repetitions"],
                                              bootstrap_indices if field == "selected_nonzero" else None)
                result["rank_deficient_count"] = sum(
                    row.get("full_column_rank") is False for row in method_rows)
                result["trajectory_divergence_count"] = sum(
                    row.get("metrics", {}).get("trajectory_status") == "diverged"
                    for row in method_rows)
                result["basis_exit_count"] = sum(
                    row.get("metrics", {}).get("trajectory_exits_basis_interval") is True
                    for row in method_rows)
                point[mode][method] = result
                for metric in spec["reporting"]["metrics"]:
                    stat = result[metric]
                    table_rows.append({
                        "mode": mode, "pair_ratio": spec["pair_ratio"] if mode == "sampled" else "full",
                        "method": method, "M": size, "metric": metric,
                        "planned_count": stat["planned_count"], "available_count": stat["available_count"],
                        "mean": stat["mean"], "sample_sd": stat["sample_sd"],
                        "median": stat["median"], "iqr_low": None if stat["iqr"] is None else stat["iqr"][0],
                        "iqr_high": None if stat["iqr"] is None else stat["iqr"][1],
                        "rmse": stat["rmse"],
                        "ci95_low": None if stat["mean_ci95"] is None else stat["mean_ci95"][0],
                        "ci95_high": None if stat["mean_ci95"] is None else stat["mean_ci95"][1],
                        "minimum": stat["minimum"], "maximum": stat["maximum"],
                    })
        summary["dictionary_sizes"].append(point)

    for mode in ("sampled", "full"):
        summary["global"][mode] = {}
        for method in ("weak_lasso", "weak_ols"):
            rows = select_rows(records, mode, method)
            summary["global"][mode][method] = {
                metric: aggregate(values(rows, metric=metric),
                                  spec["repetitions"] * len(spec["dictionary_sizes"]), None)
                for metric in spec["reporting"]["metrics"]
            }

    sampled_lasso = select_rows(records, "sampled", "weak_lasso")
    sampled_ols = select_rows(records, "sampled", "weak_ols")
    summary["diagnostics"] = {
        "planned_lasso_models": 1000,
        "selected_lasso_models": sum(row.get("selection_status") == "selected" for row in sampled_lasso),
        "selected_lasso_converged": sum(row.get("selected_converged") is True for row in sampled_lasso),
        "selected_lasso_grid_boundaries": sum(row.get("selected_at_grid_boundary") is True for row in sampled_lasso),
        "unselected_nonconverged_path_candidates": sum(row.get("path_nonconverged_count", 0) for row in sampled_lasso),
        "lasso_trajectory_divergences": sum(row.get("metrics", {}).get("trajectory_status") == "diverged" for row in sampled_lasso),
        "lasso_basis_exits": sum(row.get("metrics", {}).get("trajectory_exits_basis_interval") is True for row in sampled_lasso),
        "ols_rank_deficient_fits": sum(row.get("full_column_rank") is False for row in sampled_ols),
        "ols_trajectory_divergences": sum(row.get("metrics", {}).get("trajectory_status") == "diverged" for row in sampled_ols),
        "ols_basis_exits": sum(row.get("metrics", {}).get("trajectory_exits_basis_interval") is True for row in sampled_ols),
    }
    write_json(destination / "summary.json", summary)
    write_json(destination / "table_metrics.json", {"rows": table_rows})
    with (destination / "table_metrics.csv").open("w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(table_rows)
    return summary, records


def claim_report(summary: dict, spec: dict) -> dict:
    sizes = spec["dictionary_sizes"]
    sampled_force = [point["sampled"]["weak_lasso"]["force_error_weighted_rel"]["mean"]
                     for point in summary["dictionary_sizes"]]
    full_force = [point["full"]["weak_lasso"]["force_error_weighted_rel"]["mean"]
                  for point in summary["dictionary_sizes"]]
    sampled_ols = [point["sampled"]["weak_ols"]["force_error_weighted_rel"]["mean"]
                   for point in summary["dictionary_sizes"]]
    best_index = int(np.argmin(sampled_force))
    best_size = sizes[best_index]
    sampled_rms = summary["global"]["sampled"]["weak_lasso"]["force_error_weighted_rel"]["rmse"]
    full_rms = summary["global"]["full"]["weak_lasso"]["force_error_weighted_rel"]["rmse"]
    diagnostics = summary["diagnostics"]
    checks = {
        "aggregate_lasso_force_rms_within_1_25_of_full": (
            sampled_rms / full_rms <= spec["claim_checks"]["aggregate_lasso_force_rms_maximum_full_pair_multiplier"]),
        "force_curve_matches_full_pair_qualitatively": (
            spearman(sampled_force, full_force) >= spec["claim_checks"]["minimum_force_curve_spearman"]),
        "minimum_at_intermediate_dictionary_resolution": best_index not in (0, len(sizes) - 1),
        "coarse_to_intermediate_improvement": sampled_force[0] > sampled_force[best_index],
        "high_resolution_degradation": float(np.mean(sampled_force[-3:])) > sampled_force[best_index],
        "lasso_mean_force_below_ols_at_every_size": all(left < right for left, right in zip(sampled_force, sampled_ols)),
        "all_lasso_models_selected_and_converged": (
            diagnostics["selected_lasso_models"] == diagnostics["planned_lasso_models"]
            and diagnostics["selected_lasso_converged"] == diagnostics["planned_lasso_models"]),
        "no_lasso_rollout_divergence": diagnostics["lasso_trajectory_divergences"] == 0,
        "all_raw_ols_designs_rank_deficient": diagnostics["ols_rank_deficient_fits"] == 1000,
        "raw_ols_instability_retained": diagnostics["ols_trajectory_divergences"] > 0,
    }
    return {
        "status": "maintained" if all(checks.values()) else "not_fully_maintained",
        "checks": checks, "sampled_best_dictionary_size": best_size,
        "sampled_force_mean_curve": dict(zip(map(str, sizes), sampled_force)),
        "full_force_mean_curve": dict(zip(map(str, sizes), full_force)),
        "force_curve_spearman": spearman(sampled_force, full_force),
        "sampled_to_full_global_force_rms_ratio": sampled_rms / full_rms,
        "sampled_global_force_rms": sampled_rms, "full_global_force_rms": full_rms,
        "diagnostics": diagnostics,
        "statement": ("The Generic 1D qualitative claims are maintained under P=50N."
                      if all(checks.values()) else
                      "At least one prespecified Generic 1D claim check did not pass under P=50N."),
    }


def source_bundle_for_repetition(frozen: dict, repetition: int) -> dict:
    return frozen["reference_bundles"][f"run-{repetition}"]


def generate_figure1(destination: Path, summary: dict, records: list[dict], frozen: dict,
                     spec: dict, claims: dict) -> dict:
    apply_manuscript_style()
    size = int(claims["sampled_best_dictionary_size"])
    representative = next(row for row in records if row.get("mode") == "sampled"
                          and row.get("method") == "weak_lasso"
                          and row.get("repetition") == 0 and row.get("M") == size)
    bundle = source_bundle_for_repetition(frozen, 0)
    test = load_data(Path(bundle["reference_directory"]) / "test/data.npz")
    statistics = {**load_statistics(bundle["old_statistics"]),
                  **load_statistics(bundle["new_statistics"])}
    params = dict(candidates(frozen["effective_base"]))[f"hat_M{size}"]
    theta = np.asarray(representative["theta"], dtype=float)
    metrics, rollout = evaluate_model(test, params, theta,
                                      frozen["effective_base"]["simulation"]["sigma"],
                                      statistics[f"hat_M{size}"])
    if rollout is None:
        raise ValueError("representative sampled-pair rollout diverged")
    figure_dir = destination / "figure1_candidate"
    figure_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(figure_dir / "representative_rollout.npz", X_hat=rollout)

    basis, _ = make_kernel_basis(params)
    truth, *_ = make_true_kernel(frozen["effective_base"]["true_kernel"])
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), constrained_layout=True)
    z = np.linspace(-3.5, 3.5, 1201)
    vectors = basis(z[:, None])[..., 0]
    components = vectors * theta[None, :]
    learned = components.sum(axis=1)
    exact = truth(z[:, None])[:, 0]
    for index in range(theta.size):
        if abs(theta[index]) > frozen["effective_base"]["estimator"]["report_nonzero_threshold"]:
            axes[0].plot(z, components[:, index], color=REFERENCE_COLOR, linewidth=.45, alpha=.24)
    axes[0].plot(z, exact, color=TRUE_COLOR, linewidth=1.5, label=r"true $K^\star$")
    axes[0].plot(z, learned, "--", color=ESTIMATE_COLOR, linewidth=1.5,
                 label=rf"learned $K_{{{size}}}$")
    axes[0].axhline(0, color=GRID_COLOR, linewidth=.6)
    axes[0].set(xlabel=r"relative position $z$", ylabel=r"interaction $K(z)$",
                title="(a) Learned interaction")
    axes[0].legend(loc="best", fontsize=7.5)

    true_trajectory = test["X"][0, :, :, 0]
    fitted_trajectory = rollout[0, :, :, 0]
    time = test["t"]
    axes[1].plot(time, true_trajectory, color=TRUE_COLOR, linewidth=.18,
                 alpha=.055, rasterized=True)
    axes[1].plot(time, fitted_trajectory, "--", color=ESTIMATE_COLOR, linewidth=.18,
                 alpha=.07, rasterized=True)
    axes[1].set(xlabel=r"time $t$", ylabel="particle position",
                title=rf"(b) All trajectories ($N={true_trajectory.shape[1]}$)")
    axes[1].legend(handles=[
        Line2D([0], [0], color=TRUE_COLOR, linewidth=1.0, label="true"),
        Line2D([0], [0], color=ESTIMATE_COLOR, linestyle="--", linewidth=1.0, label="learned"),
    ], loc="best", fontsize=7.5)

    rng = np.random.default_rng(970050)
    sizes = np.asarray(spec["dictionary_sizes"], dtype=float)
    for point in summary["dictionary_sizes"]:
        M = point["M"]
        vals = np.asarray(point["sampled"]["weak_lasso"]["force_error_weighted_rel"]["values"])
        axes[2].scatter(M * np.exp(rng.uniform(-.035, .035, len(vals))), vals,
                        s=5.5, alpha=.12, color=REFERENCE_COLOR, linewidths=0)
    for method, fmt, color, label in (
        ("weak_lasso", "^--", ESTIMATE_COLOR, "weak LASSO mean (95% CI)"),
        ("weak_ols", "o-", TRUE_COLOR, "weak OLS mean (95% CI)"),
    ):
        stats = [point["sampled"][method]["force_error_weighted_rel"]
                 for point in summary["dictionary_sizes"]]
        means = np.asarray([row["mean"] for row in stats], dtype=float)
        lower = np.asarray([row["mean_ci95"][0] for row in stats], dtype=float)
        upper = np.asarray([row["mean_ci95"][1] for row in stats], dtype=float)
        axes[2].errorbar(sizes, means, yerr=np.vstack([means - lower, upper - means]),
                         fmt=fmt, color=color, linewidth=1.35, markersize=4,
                         capsize=2.5, label=label)
    axes[2].set(xscale="log", yscale="log", xlabel=r"dictionary size $M$",
                ylabel="relative force error", title=rf"(c) Dictionary size ($P=50N$)")
    axes[2].set_xticks(sizes, [str(int(value)) for value in sizes])
    axes[2].minorticks_off()
    axes[2].legend(loc="best", fontsize=7.0)
    for ax in axes:
        ax.set_box_aspect(PANEL_BOX_ASPECT)
        ax.grid(alpha=.16)
        finish_axes(ax)
    pdf = figure_dir / "one_dimensional_hat_sampled_p50n_figure1_candidate.pdf"
    png = figure_dir / "one_dimensional_hat_sampled_p50n_figure1_candidate.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    metadata = {
        "pair_ratio": 50, "pair_count": 50000, "repetitions": 100,
        "representative_M": size,
        "representative_policy": spec["representative_policy"],
        "representative_selected_after_aggregate_results": True,
        "representative_repetition": 0,
        "representative_metrics": metrics,
        "representative_theta": theta.tolist(),
        "representative_nonzero": representative["selected_nonzero"],
        "underlying_particle_count": int(true_trajectory.shape[1]),
        "displayed_particle_count": int(true_trajectory.shape[1]),
        "rollout_sha256": array_hash(rollout),
        "pdf_sha256": sha256(pdf), "png_sha256": sha256(png),
        "records_only_refit": False,
        "manuscript_edited": False, "current_figure_edited": False,
    }
    write_json(figure_dir / "metadata.json", metadata)
    shutil.copy2(Path(__file__), figure_dir / "generate_figure_snapshot.py")
    return metadata


def generate_comparison_figure(destination: Path, summary: dict, spec: dict) -> dict:
    apply_manuscript_style()
    sizes = np.asarray(spec["dictionary_sizes"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)
    panels = [
        (axes[0, 0], "weak_lasso", "force_error_weighted_rel", "LASSO force error"),
        (axes[0, 1], "weak_lasso", "trajectory_error_rel", "LASSO trajectory error"),
        (axes[1, 0], "weak_ols", "force_error_weighted_rel", "Raw OLS force error"),
    ]
    for ax, method, metric, title in panels:
        for mode, color, marker, label in (
            ("full", "black", "o", "matched full pairs"),
            ("sampled", ESTIMATE_COLOR, "^", "sampled P=50N"),
        ):
            stats = [point[mode][method][metric] for point in summary["dictionary_sizes"]]
            means = np.asarray([row["mean"] for row in stats], dtype=float)
            lower = np.asarray([row["mean_ci95"][0] for row in stats], dtype=float)
            upper = np.asarray([row["mean_ci95"][1] for row in stats], dtype=float)
            ax.errorbar(sizes, means, yerr=np.vstack([means - lower, upper - means]),
                        color=color, marker=marker, linestyle="--" if mode == "sampled" else "-",
                        linewidth=1.25, markersize=4, capsize=2.5, label=label)
        ax.set(xscale="log", yscale="log", xlabel="dictionary size M",
               ylabel="relative error", title=title)
        ax.set_xticks(sizes, [str(int(value)) for value in sizes])
        ax.minorticks_off(); ax.grid(alpha=.16); finish_axes(ax)
    ax = axes[1, 1]
    for mode, color, marker, label in (
        ("full", "black", "o", "matched full pairs"),
        ("sampled", ESTIMATE_COLOR, "^", "sampled P=50N"),
    ):
        medians = [point[mode]["weak_lasso"]["selected_lambda"]["median"]
                   for point in summary["dictionary_sizes"]]
        ax.plot(sizes, medians, color=color, marker=marker,
                linestyle="--" if mode == "sampled" else "-", label=label)
    ax.set(xscale="log", yscale="log", xlabel="dictionary size M",
           ylabel="selected lambda (median)", title="Validation-selected penalty")
    ax.set_xticks(sizes, [str(int(value)) for value in sizes])
    ax.minorticks_off(); ax.grid(alpha=.16); finish_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Generic 1D production candidate: sampled P=50N versus matched full pairs")
    figure_dir = destination / "comparison_figure"
    figure_dir.mkdir(parents=True, exist_ok=False)
    pdf = figure_dir / "sampled_p50n_vs_full_pair.pdf"
    png = figure_dir / "sampled_p50n_vs_full_pair.png"
    fig.savefig(pdf, dpi=300); fig.savefig(png, dpi=180); plt.close(fig)
    metadata = {"pdf_sha256": sha256(pdf), "png_sha256": sha256(png),
                "matched_full_pair_read_only": True}
    write_json(figure_dir / "metadata.json", metadata)
    return metadata


def verify_output(destination: Path, spec: dict, frozen: dict, status: dict,
                  summary: dict, claims: dict) -> dict:
    checks = []
    def add(name: str, passed: bool, details=None):
        checks.append({"name": name, "passed": bool(passed), "details": details})
    add("batch_completed", status["status"] == "completed", status["status"])
    add("all_100_jobs_completed", len(status["results"]) == 100
        and all(row["status"] == "completed" for row in status["results"].values()))
    reports_ok = True; pair_ok = True; seed_pairs = set()
    for key, entry in status["results"].items():
        path = destination / entry["directory"] / "report.json"
        reports_ok &= sha256(path) == entry["report_sha256"]
        report = read_json(path)
        pair_ok &= report["particles"] == 1000 and report["sampled_pairs"] == 50000
        seeds = (report["pair_seeds"]["training"], report["pair_seeds"]["validation"])
        pair_ok &= seeds[0] != seeds[1]
        seed_pairs.add(seeds)
    add("all_sampled_report_hashes_match", reports_ok)
    add("all_jobs_use_P_50N_and_independent_validation_pairs", pair_ok)
    add("all_100_pair_seed_pairs_unique", len(seed_pairs) == 100, len(seed_pairs))
    records = read_json(destination / "records.json")["records"]
    add("record_count_is_4000", len(records) == 4000, len(records))
    add("sampled_record_count_is_2000", sum(row.get("mode") == "sampled" for row in records) == 2000)
    add("full_comparison_record_count_is_2000", sum(row.get("mode") == "full" for row in records) == 2000)
    diagnostics = summary["diagnostics"]
    add("all_1000_lasso_models_selected", diagnostics["selected_lasso_models"] == 1000,
        diagnostics["selected_lasso_models"])
    add("all_selected_lasso_models_converged", diagnostics["selected_lasso_converged"] == 1000,
        diagnostics["selected_lasso_converged"])
    add("no_lasso_rollout_divergence", diagnostics["lasso_trajectory_divergences"] == 0,
        diagnostics["lasso_trajectory_divergences"])
    add("claim_report_maintained", claims["status"] == "maintained", claims["checks"])
    for relative, expected in spec["current_artifact_audit"].items():
        add(f"current_artifact_unchanged:{relative}", sha256(ROOT / relative) == expected)
    for method in ("lasso", "ols"):
        batch = ROOT / spec[f"source_{method}_batch"]
        for relative, expected in spec[f"source_{method}_hashes"].items():
            add(f"source_{method}_unchanged:{relative}", sha256(batch / relative) == expected)
    for name in ("summary.json", "records.json", "table_metrics.json", "table_metrics.csv",
                 "claim_report.json", "figure1_candidate/one_dimensional_hat_sampled_p50n_figure1_candidate.pdf",
                 "figure1_candidate/one_dimensional_hat_sampled_p50n_figure1_candidate.png",
                 "comparison_figure/sampled_p50n_vs_full_pair.pdf",
                 "comparison_figure/sampled_p50n_vs_full_pair.png"):
        add(f"artifact_exists:{name}", (destination / name).is_file())
    for name, expected in frozen["source_sha256"].items():
        add(f"source_snapshot_matches:{name}", sha256(ROOT / name) == expected
            and sha256(destination / "source_snapshot" / name) == expected)
    report = {
        "status": "passed" if all(row["passed"] for row in checks) else "failed",
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks), "checks": checks,
        "diagnostics": diagnostics,
        "artifact_sha256": {
            str(path.relative_to(destination)): sha256(path)
            for path in destination.rglob("*") if path.is_file()
            and path.name not in {"verification.json", "report.json"}
        },
    }
    verification_dir = destination / "verification"
    verification_dir.mkdir(exist_ok=True)
    write_json(verification_dir / "verification.json", report)
    return report


def initialize(spec: dict, source: dict, base: dict, destination: Path, workers: int) -> tuple[dict, dict]:
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {name: sha256(ROOT / name) for name in SOURCES}
    for name in SOURCES:
        target = destination / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    provenance = destination / "source_provenance"
    for method in ("lasso", "ols"):
        batch = source[method]
        for relative in spec[f"source_{method}_hashes"]:
            target = provenance / method / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(batch / relative, target)
    effective = worker_spec(spec)
    _, bundles = build_matched_sources_production(effective, source, destination)
    frozen = {
        "requested_protocol": spec, "worker_protocol": effective,
        "effective_base": base, "reference_bundles": bundles,
        "source_sha256": hashes,
        "source_provenance_sha256": {
            str(path.relative_to(provenance)): sha256(path)
            for path in provenance.rglob("*") if path.is_file()
        },
        "matched_full_pair_records_sha256": sha256(destination / "matched_full_pair_records.json"),
        "frozen_reference_bundles_sha256": sha256(destination / "frozen_reference_bundles.json"),
        "workers": workers, "python_version": sys.version,
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
    }
    write_json(destination / "frozen_protocol.json", frozen)
    status = {
        "status": "running", "stage": "sampled_p50n_R100_production_candidate",
        "frozen_protocol_sha256": sha256(destination / "frozen_protocol.json"),
        "jobs": job_plan(effective), "results": {},
    }
    write_json(destination / "batch_status.json", status)
    return frozen, status


def resume_state(spec: dict, destination: Path) -> tuple[dict, dict]:
    frozen = read_json(destination / "frozen_protocol.json")
    if frozen["requested_protocol"] != spec:
        raise ValueError("resume protocol mismatch")
    for name, expected in frozen["source_sha256"].items():
        if sha256(ROOT / name) != expected or sha256(destination / "source_snapshot" / name) != expected:
            raise ValueError(f"resume source mismatch: {name}")
    status = read_json(destination / "batch_status.json")
    if sha256(destination / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for key, entry in status["results"].items():
        if entry["status"] == "completed" and sha256(
                destination / entry["directory"] / "report.json") != entry["report_sha256"]:
            raise ValueError(f"completed report changed: {key}")
    status["status"] = "running"
    write_json(destination / "batch_status.json", status)
    return frozen, status


def reset_postprocessing_outputs(destination: Path) -> None:
    """Remove only derived outputs so a completed batch can be finalized again."""
    for name in ("figure1_candidate", "comparison_figure", "verification"):
        path = Path(destination) / name
        if path.exists():
            shutil.rmtree(path)


def run_batch(spec: dict, destination: Path, workers: int, resume: bool) -> dict:
    validate_spec(spec)
    source = load_sources(worker_spec(spec))
    validate_locked_inputs(spec, source)
    base = validate_sources(worker_spec(spec), source)
    destination = destination.resolve()
    if resume:
        frozen, status = resume_state(spec, destination)
    else:
        frozen, status = initialize(spec, source, base, destination, workers)
    pending = {key: job for key, job in status["jobs"].items()
               if status["results"].get(key, {}).get("status") != "completed"}
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {
            pool.submit(worker, destination, job, frozen["worker_protocol"],
                        frozen["effective_base"],
                        frozen["reference_bundles"][f"run-{job['repetition']}"]): key
            for key, job in pending.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            entry = future.result()
            status["results"][key] = entry
            write_json(destination / "batch_status.json", status)
            print(f"{len(status['results'])}/100: {key} {entry['status']} "
                  f"{entry['elapsed_seconds']:.1f}s", flush=True)
    status["status"] = ("completed" if len(status["results"]) == 100
                        and all(row["status"] == "completed" for row in status["results"].values())
                        else "completed_with_failures")
    status["elapsed_seconds_this_invocation"] = perf_counter() - started
    write_json(destination / "batch_status.json", status)

    reset_postprocessing_outputs(destination)
    summary, records = build_summary(destination, status, frozen, spec)
    claims = claim_report(summary, spec)
    write_json(destination / "claim_report.json", claims)
    figure1 = generate_figure1(destination, summary, records, frozen, spec, claims)
    comparison = generate_comparison_figure(destination, summary, spec)
    verification = verify_output(destination, spec, frozen, status, summary, claims)
    report = {
        "status": status["status"], "verification_status": verification["status"],
        "destination": str(destination), "jobs_completed": len(status["results"]),
        "jobs_planned": 100, "pair_ratio": 50, "pair_count": 50000,
        "claim_status": claims["status"], "claim_statement": claims["statement"],
        "representative_M": claims["sampled_best_dictionary_size"],
        "frozen_protocol_sha256": sha256(destination / "frozen_protocol.json"),
        "batch_status_sha256": sha256(destination / "batch_status.json"),
        "records_sha256": sha256(destination / "records.json"),
        "summary_sha256": sha256(destination / "summary.json"),
        "table_metrics_json_sha256": sha256(destination / "table_metrics.json"),
        "table_metrics_csv_sha256": sha256(destination / "table_metrics.csv"),
        "claim_report_sha256": sha256(destination / "claim_report.json"),
        "figure1_candidate": figure1,
        "comparison_figure": comparison,
        "verification_sha256": sha256(destination / "verification/verification.json"),
        "manuscript_edited": False, "current_figure_edited": False,
    }
    write_json(destination / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    if args.resume and args.out_dir is None:
        parser.error("--resume requires --out-dir")
    spec = read_json(args.config)
    validate_spec(spec)
    source = load_sources(worker_spec(spec))
    validate_locked_inputs(spec, source)
    base = validate_sources(worker_spec(spec), source)
    if not args.run:
        print(json.dumps({
            "status": "configuration_valid", "repetitions": spec["repetitions"],
            "dictionary_sizes": base["basis"]["sizes"], "pair_ratio": spec["pair_ratio"],
            "pair_count": spec["pair_count"], "jobs": 100,
        }, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / (
        "sampled-p50n-R100-" + stamp)
    report = run_batch(spec, destination, args.workers, args.resume)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
