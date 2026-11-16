"""Run a three-repetition matched localized-shell pilot in ten dimensions.

This pilot is isolated from manuscript scientific outputs.  It reuses the
validated train/validation/test and raw-LASSO machinery of the Morse sweep but
changes the interaction to a sparse, exactly realizable Gaussian-shell model.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import make_radial_basis
from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR, TRUE_COLOR, apply_manuscript_style, finish_axes, save_pdf,
)
from scripts.run_morse10d_sweeps import (
    _radial_drift_fast,
    physical_conditions,
    read_json,
    repetition_plan,
    run_repetition as _base_run_repetition,
    write_json,
)

DEFAULT_CONFIG = ROOT / "configs/experiments/localized10d_pilot.json"
SOURCE_FILES = (
    "configs/experiments/localized10d_pilot.json",
    "configs/experiments/localized10d_scientific.json",
    "scripts/run_localized10d_pilot.py",
    "scripts/run_morse10d_sweeps.py",
    "scripts/run_1d_hat_lasso.py",
    "test/test_localized10d_pilot.py",
    "features.py", "kernels.py", "pipeline.py", "regression.py", "simulate.py",
    "tests.py", "time_tests.py", "utils.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(spec):
    task = spec.get("task")
    if task not in {"localized10d_matched_pilot", "localized10d_matched_scientific"} \
            or spec.get("protocol_version") != 1:
        raise ValueError("invalid localized 10D protocol")
    expected_pilot_only = task == "localized10d_matched_pilot"
    if spec.get("pilot_only") is not expected_pilot_only:
        raise ValueError("localized pilot/scientific designation is inconsistent")
    model = spec.get("model", {})
    if (model.get("dimension"), model.get("particles"), model.get("steps"),
            model.get("dt"), model.get("observation_stride"), model.get("sigma")) != (
            10, 500, 500, 0.01, 2, 0.05):
        raise ValueError("pilot model, horizon, or baseline changed")
    if (model.get("initial_law"), model.get("initial_scales")) != (
        "normal_scale_mixture", [0.25, 0.4, 0.55, 0.7]
    ):
        raise ValueError("pilot excitation mixture changed")
    interaction = model.get("interaction", {})
    dictionary = spec.get("dictionary", {})
    if interaction.get("mode") != "gaussian_shell" or dictionary.get("basis") != "gaussian_shell":
        raise ValueError("pilot requires a Gaussian-shell truth and dictionary")
    if interaction.get("centers") != dictionary.get("centers") or interaction.get("h") != dictionary.get("h"):
        raise ValueError("truth and candidate shell grids must match")
    if interaction.get("coefficients") != [0.0, 1.5, -1.0, 0.0]:
        raise ValueError("pilot sparse target changed")
    if spec.get("sweeps") != {
        "process_sigma": [0.0, 0.05, 0.075, 0.1, 0.15],
        "particle_number": [100, 200, 350, 500],
        "sampled_pairs": [3000, 5000, 10000, 20000, 30000],
    }:
        raise ValueError("pilot sweep grid changed")
    if spec.get("pair_sampling") != {
        "baseline_pairs": 10000, "mode": "nested_resample_each_time",
        "with_replacement": True, "exclude_diagonal": True,
    }:
        raise ValueError("pilot nested pair-sampling protocol changed")
    expected_repetitions = 3 if expected_pilot_only else 20
    if spec["evaluation"].get("scientific_repetitions") != expected_repetitions:
        raise ValueError(f"localized run must use exactly {expected_repetitions} repetitions")
    estimator = spec["estimator"]
    forbidden = (
        estimator.get("truth_used_for_selection"), estimator.get("test_error_used_for_selection"),
        estimator.get("post_lasso_refit"), estimator.get("coefficient_thresholding"),
        estimator.get("standardize"), estimator.get("fit_intercept"),
    )
    if any(forbidden) or estimator.get("selection_metric") != "independent_validation_weak_residual_mse":
        raise ValueError("pilot estimator is not validation-selected raw LASSO")


def effective_protocol(spec, smoke=False):
    effective = json.loads(json.dumps(spec))
    if not smoke:
        return effective
    effective["model"].update({"particles": 30, "steps": 20, "observation_stride": 1})
    effective["weak_tests"].update({"spatial_count": 4, "temporal_count": 3})
    effective["pair_sampling"]["baseline_pairs"] = 60
    effective["sweeps"] = {
        "process_sigma": [0.0, 0.05], "particle_number": [15, 30],
        "sampled_pairs": [20, 60],
    }
    effective["estimator"].update({"alpha_count": 8, "max_iter": 20000})
    effective["evaluation"].update({
        "scientific_repetitions": 1, "profile_grid_size": 61,
        "latent_drift_time_count": 3, "latent_drift_target_count": 8,
        "rollout_particles": 20, "rollout_horizon": 0.2,
    })
    effective["pilot_only"] = True
    return effective


def _augment_record(record, target):
    theta = record.get("coefficients")
    if theta is None:
        absolute = relative = None
    else:
        theta = np.asarray(theta, dtype=float)
        absolute = float(np.linalg.norm(theta - target))
        relative = float(absolute / np.linalg.norm(target))
    return {**record, "coefficient_error_l2": absolute,
            "coefficient_error_relative": relative}


def run_localized_repetition(payload):
    out_string, spec, plan = payload
    result = _base_run_repetition(payload)
    target = np.asarray(spec["model"]["interaction"]["coefficients"], dtype=float)
    result["records"] = [_augment_record(row, target) for row in result["records"]]
    return result


def _finite_stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size), "finite_count": int(finite.size),
        "mean": None if finite.size != values.size or not finite.size else float(finite.mean()),
        "median": None if finite.size != values.size or not finite.size else float(np.median(finite)),
    }


def summarize(records):
    groups = []
    for sweep in ("process_sigma", "particle_number", "sampled_pairs"):
        values = sorted({float(row["sweep_value"]) for row in records if row["sweep"] == sweep})
        for value in values:
            for method in ("lasso", "ols"):
                rows = [row for row in records if row["sweep"] == sweep
                        and float(row["sweep_value"]) == value and row["method"] == method]
                conditions = np.asarray([row["condition_A"] for row in rows], dtype=float)
                groups.append({
                    "sweep": sweep, "value": value, "method": method,
                    "repetitions": len(rows),
                    "coefficient_error_relative": _finite_stats(
                        [row["coefficient_error_relative"] for row in rows]
                    ),
                    "latent_drift_relative_error": _finite_stats(
                        [row["latent_drift_relative_error"] for row in rows]
                    ),
                    "profile_relative_error": _finite_stats(
                        [row["profile_relative_error"] for row in rows]
                    ),
                    "pair_design_relative_error": _finite_stats(
                        [row.get("pair_design_relative_error") for row in rows]
                    ) if sweep == "sampled_pairs" else None,
                    "condition_A_geometric_mean": float(np.exp(np.mean(np.log(conditions)))),
                    "rank_deficient": int(sum(row["rank_deficient"] for row in rows)),
                    "nonconverged": int(sum(not row["converged"] for row in rows)),
                    "boundary_selections": int(sum(row["selected_at_grid_boundary"] for row in rows)),
                })
    return {"groups": groups, "record_count": len(records)}


def acceptance_report(spec, summary, jobs, dynamics):
    groups = summary["groups"]
    lasso = [row for row in groups if row["method"] == "lasso"]
    maximum_condition = max(row["condition_A_geometric_mean"] for row in groups)
    directions = {}
    for sweep, expected in spec["pilot_acceptance"]["required_lasso_endpoint_directions"].items():
        rows = sorted([row for row in lasso if row["sweep"] == sweep], key=lambda row: row["value"])
        direction_metric = (
            "pair_design_relative_error" if sweep == "sampled_pairs"
            else "coefficient_error_relative"
        )
        first = rows[0][direction_metric]["mean"]
        last = rows[-1][direction_metric]["mean"]
        passed = (last > first) if expected == "increase" else (last < first)
        positive = [
            row for row in rows
            if row["value"] > 0 and row[direction_metric]["mean"] > 0
        ]
        slope = float(np.polyfit(
            np.log([row["value"] for row in positive]),
            np.log([row[direction_metric]["mean"] for row in positive]), 1,
        )[0])
        slope_passed = slope > 0 if expected == "increase" else slope < 0
        means = [row[direction_metric]["mean"] for row in rows]
        tolerance = float(spec["pilot_acceptance"]["direction_relative_tolerance"])
        if expected == "increase":
            adjacent_passed = all(right >= (1.0 - tolerance) * left
                                  for left, right in zip(means, means[1:]))
        else:
            adjacent_passed = all(right <= (1.0 + tolerance) * left
                                  for left, right in zip(means, means[1:]))
        directions[sweep] = {
            "expected": expected, "metric": direction_metric,
            "first": first, "last": last,
            "log_log_slope": slope, "endpoint_passed": bool(passed),
            "slope_passed": bool(slope_passed),
            "adjacent_tolerance": tolerance,
            "adjacent_passed": bool(adjacent_passed),
            "passed": bool(passed and slope_passed and adjacent_passed),
        }
    all_records_finite = all(
        row[metric]["finite_count"] == row[metric]["count"]
        for row in groups
        for metric in ("coefficient_error_relative", "latent_drift_relative_error", "profile_relative_error")
    )
    no_rank_deficiency = not any(row["rank_deficient"] for row in groups)
    selections = [value["lasso"] for job in jobs for value in job["selections"].values()]
    all_selections_converged = all(item.get("status") == "selected" for item in selections)
    baseline = next(
        row for row in lasso
        if row["sweep"] == "process_sigma" and row["value"] == spec["model"]["sigma"]
    )
    accuracy = {
        "baseline_coefficient_error": {
            "value": baseline["coefficient_error_relative"]["mean"],
            "threshold": spec["pilot_acceptance"]["maximum_baseline_lasso_coefficient_error"],
        },
        "baseline_profile_error": {
            "value": baseline["profile_relative_error"]["mean"],
            "threshold": spec["pilot_acceptance"]["maximum_baseline_lasso_profile_error"],
        },
        "baseline_drift_error": {
            "value": baseline["latent_drift_relative_error"]["mean"],
            "threshold": spec["pilot_acceptance"]["maximum_baseline_lasso_drift_error"],
        },
        "first_repetition_trajectory_error": {
            "value": dynamics["first_repetition_coupled_trajectory_relative_error"],
            "threshold": spec["pilot_acceptance"]["maximum_first_repetition_trajectory_error"],
        },
    }
    for item in accuracy.values():
        item["passed"] = bool(item["value"] <= item["threshold"])
    criteria = {
        "condition_number": {
            "maximum_geometric_mean": maximum_condition,
            "threshold": spec["pilot_acceptance"]["maximum_geometric_mean_condition_A"],
            "passed": maximum_condition <= spec["pilot_acceptance"]["maximum_geometric_mean_condition_A"],
        },
        "directions": directions,
        "accuracy": accuracy,
        "all_reported_metrics_finite": all_records_finite,
        "no_rank_deficiency": no_rank_deficiency,
        "all_path_selections_have_converged_candidate": all_selections_converged,
        "fixed_pair_nonconverged_records": int(sum(
            row["nonconverged"] for row in lasso if row["sweep"] == "sampled_pairs"
        )),
    }
    passed = (
        criteria["condition_number"]["passed"]
        and all(item["passed"] for item in directions.values())
        and all(item["passed"] for item in accuracy.values())
        and all_records_finite and no_rank_deficiency and all_selections_converged
    )
    return {"passed": bool(passed), "criteria": criteria}


def _plot_sweeps(summary, path):
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.4), constrained_layout=True)
    labels = {
        "process_sigma": ("(a) Process noise", r"process noise $\sigma$"),
        "particle_number": ("(b) Particle count", r"particles $N$"),
        "sampled_pairs": ("(c) Sampled pairs", r"sampled pairs $P$"),
    }
    for axis, sweep in zip(axes, labels):
        if sweep == "sampled_pairs":
            rows = sorted(
                [row for row in summary["groups"] if row["sweep"] == sweep
                 and row["method"] == "lasso"], key=lambda row: row["value"]
            )
            axis.plot([row["value"] for row in rows],
                      [row["pair_design_relative_error"]["mean"] for row in rows],
                      color=ESTIMATE_COLOR, marker="^", label="sampled design")
            axis.set_xscale("log"); axis.set_ylabel("relative design error")
        else:
            for method, color, marker, name in (
                ("ols", TRUE_COLOR, "o", "OLS"),
                ("lasso", ESTIMATE_COLOR, "^", "LASSO"),
            ):
                rows = sorted(
                    [row for row in summary["groups"]
                     if row["sweep"] == sweep and row["method"] == method],
                    key=lambda row: row["value"],
                )
                axis.plot([row["value"] for row in rows],
                          [row["coefficient_error_relative"]["mean"] for row in rows],
                          color=color, marker=marker, label=name)
            axis.set_yscale("log")
            if axis is axes[0]: axis.set_ylabel("relative coefficient error")
        axis.set_title(labels[sweep][0]); axis.set_xlabel(labels[sweep][1])
        if sweep == "process_sigma": axis.set_xlim(left=0)
        axis.legend(loc="best")
        finish_axes(axis)
    save_pdf(fig, path)


def _simulate_coupled(spec, theta, seed):
    model = spec["model"]
    particles = min(int(spec["evaluation"]["rollout_particles"]), model["particles"])
    steps = min(int(round(spec["evaluation"]["rollout_horizon"] / model["dt"])), model["steps"])
    rng = np.random.default_rng(seed)
    if model["initial_law"] == "normal_scale_mixture":
        particle_scales = rng.choice(model["initial_scales"], size=particles, replace=True)
        true_x = rng.normal(size=(particles, model["dimension"])) * particle_scales[:, None]
    else:
        true_x = rng.normal(0.0, model["initial_scale"], size=(particles, model["dimension"]))
    learned_x = true_x.copy()
    true_path = np.empty((steps + 1, particles, model["dimension"])); true_path[0] = true_x
    learned_path = np.empty_like(true_path); learned_path[0] = learned_x
    learned_interaction = {
        **model["interaction"], "coefficients": np.asarray(theta, dtype=float).tolist()
    }
    for step in range(steps):
        increment = rng.normal(size=true_x.shape)
        true_x = true_x + model["dt"] * _radial_drift_fast(true_x, model["interaction"]) \
            + model["sigma"] * np.sqrt(model["dt"]) * increment
        learned_x = learned_x + model["dt"] * _radial_drift_fast(learned_x, learned_interaction) \
            + model["sigma"] * np.sqrt(model["dt"]) * increment
        true_path[step + 1] = true_x; learned_path[step + 1] = learned_x
    error = float(np.linalg.norm(learned_path - true_path) / np.linalg.norm(true_path))
    return true_path, learned_path, error


def _plot_profile_and_dynamics(spec, records, out):
    base_sigma = spec["model"]["sigma"]
    selected = next(row for row in records if row["repetition"] == 0
                    and row["sweep"] == "process_sigma"
                    and float(row["sweep_value"]) == base_sigma and row["method"] == "lasso")
    theta = np.asarray(selected["coefficients"], dtype=float)
    target = np.asarray(spec["model"]["interaction"]["coefficients"], dtype=float)
    basis, meta = make_radial_basis(spec["dictionary"])
    radius = np.linspace(*spec["evaluation"]["profile_radius"], 400)
    profile_true, profile_fit = basis(radius) @ target, basis(radius) @ theta
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.5), constrained_layout=True)
    axes[0].plot(radius, profile_true, color=TRUE_COLOR, label="true")
    axes[0].plot(radius, profile_fit, color=ESTIMATE_COLOR, linestyle="--", label="LASSO")
    axes[0].axhline(0, color="0.4", linewidth=0.7); axes[0].set_xlabel("distance $r$")
    axes[0].set_ylabel("radial profile"); axes[0].set_title("(a) Radial interaction"); axes[0].legend()
    x = np.arange(target.size); width = 0.38
    axes[1].bar(x - width / 2, target, width, color=TRUE_COLOR, label="true")
    axes[1].bar(x + width / 2, theta, width, color=ESTIMATE_COLOR, label="LASSO")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"{v:g}" for v in meta["centers"]], rotation=35)
    axes[1].set_xlabel("shell center"); axes[1].set_ylabel("coefficient")
    axes[1].set_title("(b) Matched coefficients"); axes[1].legend()
    for axis in axes: finish_axes(axis)
    save_pdf(fig, out / "localized10d_profile_coefficients.pdf")

    plan = repetition_plan(spec)[0]
    truth, learned, trajectory_error = _simulate_coupled(spec, theta, plan["test_seed"])
    times = np.arange(truth.shape[0]) * spec["model"]["dt"]
    radius_true = np.linalg.norm(truth - truth.mean(axis=1, keepdims=True), axis=-1).mean(axis=1)
    radius_fit = np.linalg.norm(learned - learned.mean(axis=1, keepdims=True), axis=-1).mean(axis=1)
    tri = np.triu_indices(truth.shape[1], k=1)
    dist_true = np.linalg.norm(truth[-1, :, None, :] - truth[-1, None, :, :], axis=-1)[tri]
    dist_fit = np.linalg.norm(learned[-1, :, None, :] - learned[-1, None, :, :], axis=-1)[tri]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.5), constrained_layout=True)
    axes[0].plot(times, radius_true, color=TRUE_COLOR, label="true")
    axes[0].plot(times, radius_fit, color=ESTIMATE_COLOR, linestyle="--", label="LASSO")
    axes[0].set_xlabel("time"); axes[0].set_ylabel("mean radius"); axes[0].set_title("(a) Coupled radius")
    axes[0].legend()
    bins = np.linspace(min(dist_true.min(), dist_fit.min()), max(dist_true.max(), dist_fit.max()), 35)
    axes[1].hist(dist_true, bins=bins, density=True, histtype="step", color=TRUE_COLOR, label="true")
    axes[1].hist(dist_fit, bins=bins, density=True, histtype="step", color=ESTIMATE_COLOR, label="LASSO")
    axes[1].set_xlabel("final pair distance"); axes[1].set_ylabel("density")
    axes[1].set_title("(b) Final distances"); axes[1].legend()
    for axis in axes: finish_axes(axis)
    save_pdf(fig, out / "localized10d_coupled_dynamics.pdf")
    return {"first_repetition_coupled_trajectory_relative_error": trajectory_error,
            "first_repetition_coefficients": theta.tolist()}


def write_csv(path, records):
    columns = sorted({key for row in records for key in row if not isinstance(row[key], (dict, list))})
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        writer.writerows([{key: row.get(key) for key in columns} for row in records])


def source_snapshot(out):
    hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        hashes[relative] = digest(source)
    write_json(out / "source_manifest.json", {"sha256": hashes})
    return hashes


def run(config_path, out, *, workers=1, smoke=False, resume=False):
    requested = read_json(config_path); validate_protocol(requested)
    spec = effective_protocol(requested, smoke=smoke)
    out = Path(out)
    if out.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {out}")
    if not out.exists():
        out.mkdir(parents=True)
        write_json(out / "frozen_protocol.json", spec)
        hashes = source_snapshot(out)
        write_json(out / "manifest.json", {
            "task": spec["task"], "status": "running", "smoke": smoke,
            "source_sha256": hashes, "jobs": [],
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
    else:
        manifest = read_json(out / "manifest.json")
        if manifest["source_sha256"] != {relative: digest(ROOT / relative) for relative in SOURCE_FILES}:
            raise ValueError("resume source hash mismatch")
        if read_json(out / "frozen_protocol.json") != spec:
            raise ValueError("resume protocol mismatch")
    plans = repetition_plan(spec)
    completed = {int(path.stem.split("-")[-1]) for path in (out / "jobs").glob("repetition-*.json")} \
        if (out / "jobs").exists() else set()
    pending = [plan for plan in plans if plan["index"] not in completed]
    (out / "jobs").mkdir(exist_ok=True)
    started = perf_counter()
    payloads = [(str(out), spec, plan) for plan in pending]
    if workers == 1:
        generated = map(run_localized_repetition, payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        generated = executor.map(run_localized_repetition, payloads)
    try:
        for result in generated:
            write_json(out / "jobs" / f"repetition-{result['index']:03d}.json", result)
            print(f"localized pilot repetition {result['index'] + 1}/{len(plans)}", flush=True)
    finally:
        if workers != 1: executor.shutdown(wait=True)
    jobs = [read_json(path) for path in sorted((out / "jobs").glob("repetition-*.json"))]
    records = [row for job in jobs for row in job["records"]]
    summary = summarize(records)
    write_json(out / "records.json", records); write_csv(out / "records.csv", records)
    write_json(out / "summary.json", summary)
    _plot_sweeps(summary, out / "localized10d_pilot_sweeps.pdf")
    dynamics = _plot_profile_and_dynamics(spec, records, out)
    write_json(out / "dynamics_summary.json", dynamics)
    acceptance = (
        {"passed": None, "applicable": False, "reason": "smoke run"}
        if smoke else acceptance_report(spec, summary, jobs, dynamics)
    )
    write_json(out / "acceptance.json", acceptance)
    artifacts = {name: digest(out / name) for name in (
        "frozen_protocol.json", "source_manifest.json", "records.json", "records.csv",
        "summary.json", "acceptance.json", "localized10d_pilot_sweeps.pdf",
        "localized10d_profile_coefficients.pdf", "localized10d_coupled_dynamics.pdf",
        "dynamics_summary.json",
    )}
    manifest = read_json(out / "manifest.json")
    manifest.update({
        "status": (
            "smoke_completed" if smoke else
            ("pilot_completed" if spec["pilot_only"] else "scientific_completed")
        ),
        "jobs": [job["index"] for job in jobs], "record_count": len(records),
        "acceptance_passed": acceptance["passed"], "artifact_sha256": artifacts,
        "elapsed_seconds_this_invocation": perf_counter() - started,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out_dir is None:
        label = "smoke" if args.smoke else "pilot"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out_dir = ROOT / "outputs" / "validation" / "localized10d_pilot" / f"{label}-{stamp}"
    result = run(args.config.resolve(), args.out_dir, workers=args.workers,
                 smoke=args.smoke, resume=args.resume)
    print(json.dumps({"output": str(args.out_dir), "status": result["status"],
                      "acceptance_passed": result["acceptance_passed"]}, indent=2))


if __name__ == "__main__":
    main()
