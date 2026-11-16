"""Run validation-selected strong FD-LASSO on the predator--prey noise sweep."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predator_prey import channel_kernels
from scripts.run_cross_example_nominal_comparison import (
    assemble_predator_prey_strong,
    relative_error,
)
from scripts.run_regularized_baseline_exploration import select_fd_lasso
from scripts.run_predator_prey_sweeps import (
    _integer_ratio,
    coefficient_and_profile_metrics,
    condition_plan,
    effective_protocol as effective_base_protocol,
    latent_for_condition,
    model_objects,
    observation_view,
    paired_inputs,
    read_json,
    species_balanced_drift_error,
    validate_protocol as validate_base_protocol,
    write_json,
)

DEFAULT_CONFIG = ROOT / "configs/experiments/predator_prey_strong_lasso_observation_noise_common_noise_001.json"
OUTPUT_ROOT = ROOT / "outputs/validation/predator_prey_strong_lasso_observation_noise"
METHOD = "strong_forward_difference_lasso_validation_selected"
SOURCE_FILES = (
    "configs/experiments/predator_prey_strong_lasso_observation_noise_common_noise_001.json",
    "configs/experiments/predator_prey_sweeps_common_noise_001.json",
    "scripts/run_predator_prey_strong_lasso_observation_noise.py",
    "scripts/run_regularized_baseline_exploration.py",
    "scripts/run_predator_prey_sweeps.py",
    "scripts/run_cross_example_nominal_comparison.py",
    "predator_prey.py", "features.py", "kernels.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(requested, base):
    if (requested.get("task"), requested.get("protocol_version")) != (
        "predator_prey_strong_lasso_observation_noise", 1
    ):
        raise ValueError("invalid predator--prey strong-LASSO protocol")
    if requested.get("study_status") != "manuscript_current":
        raise ValueError("strong FD-LASSO must use the current manuscript protocol")
    validate_base_protocol(base)
    for key in ("base_config", "weak_sweep_records"):
        if digest(ROOT / requested[key]) != requested[f"{key}_sha256"]:
            raise ValueError(f"pinned input changed: {key}")
    if requested["observation_noise_values"] != base["sweeps"]["observation_noise"]:
        raise ValueError("observation-noise grid changed")
    if requested["repetitions"] != base["evaluation"]["scientific_repetitions"]:
        raise ValueError("repetition count changed")
    strong = requested["strong_form"]
    if strong["prey_system_columns"] != 12 or strong["predator_system_columns"] != 5:
        raise ValueError("strong-form dictionary dimensions changed")
    if strong["penalty_policy"] != "select_prey_and_predator_systems_separately":
        raise ValueError("predator--prey penalty policy changed")
    estimator = strong["estimator"]
    if estimator["fit_intercept"] or estimator["standardize"]:
        raise ValueError("strong FD-LASSO must use raw columns and no intercept")


def effective_protocol(requested, base, smoke=False):
    spec = effective_base_protocol(base, smoke=smoke)
    spec["task"] = requested["task"]
    return spec


def observation_conditions(spec):
    return [condition for condition in condition_plan(spec)
            if condition["sweep"] == "observation_noise"]


def run_repetition(payload):
    repetition, spec, requested = payload
    started = perf_counter()
    prey_basis, predator_basis, _, truth, truth_kernels = model_objects(spec)
    truth_vector = np.concatenate(truth)
    training_inputs = paired_inputs(spec, repetition, "training")
    validation_inputs = paired_inputs(spec, repetition, "validation")
    test_inputs = paired_inputs(spec, repetition, "test")
    training_cache, validation_cache, test_cache = {}, {}, {}
    records = []
    estimator = requested["strong_form"]["estimator"]
    with threadpool_limits(limits=1):
        for condition in observation_conditions(spec):
            (training_prey, training_predator, _), latent_dt = latent_for_condition(
                spec, condition, training_inputs, truth_kernels, training_cache
            )
            observed_prey, observed_predator, observed_time = observation_view(
                training_prey, training_predator, latent_dt,
                training_inputs["unit_prey_noise"], training_inputs["unit_predator_noise"],
                spec["pairing"]["dt_common_fine"], condition["observation_dt"],
                condition["observation_noise"],
            )
            (validation_prey, validation_predator, _), validation_dt = latent_for_condition(
                spec, condition, validation_inputs, truth_kernels, validation_cache
            )
            observed_validation_prey, observed_validation_predator, validation_time = observation_view(
                validation_prey, validation_predator, validation_dt,
                validation_inputs["unit_prey_noise"], validation_inputs["unit_predator_noise"],
                spec["pairing"]["dt_common_fine"], condition["observation_dt"],
                condition["observation_noise"],
            )
            A_prey, b_prey, A_predator, b_predator = assemble_predator_prey_strong(
                observed_prey, observed_predator, observed_time, prey_basis, predator_basis
            )
            Av_prey, bv_prey, Av_predator, bv_predator = assemble_predator_prey_strong(
                observed_validation_prey, observed_validation_predator, validation_time,
                prey_basis, predator_basis,
            )
            theta_prey, prey_selection = select_fd_lasso(
                A_prey, b_prey, Av_prey, bv_prey, estimator
            )
            theta_predator, predator_selection = select_fd_lasso(
                A_predator, b_predator, Av_predator, bv_predator, estimator
            )
            channel_count = truth[0].size
            coefficients = (
                theta_prey[:channel_count], theta_prey[channel_count:], theta_predator
            )
            theta = np.concatenate(coefficients)
            estimated_kernels = channel_kernels(
                *coefficients, prey_basis, predator_basis
            )
            (test_prey, test_predator, _), test_dt = latent_for_condition(
                spec, condition, test_inputs, truth_kernels, test_cache
            )
            stride = _integer_ratio(spec["evaluation"]["drift_evaluation_interval"], test_dt)
            drift = species_balanced_drift_error(
                test_prey, test_predator, truth_kernels, estimated_kernels, stride
            )
            records.append({
                "repetition": repetition,
                "condition": condition,
                "method": METHOD,
                "training_seed": training_inputs["seed"],
                "validation_seed": validation_inputs["seed"],
                "test_seed": test_inputs["seed"],
                "coefficients": theta.tolist(),
                "coefficient_error_relative": relative_error(theta, truth_vector),
                "channels": coefficient_and_profile_metrics(
                    coefficients, truth, spec, prey_basis, predator_basis
                ),
                "held_out_drift": drift,
                "selection": {"prey": prey_selection, "predator": predator_selection},
                "status": "finite" if np.all(np.isfinite(theta)) else "nonfinite",
            })
    return {"repetition": repetition, "records": records,
            "elapsed_seconds": perf_counter() - started}


def summarize(records):
    groups = []
    for value in sorted({float(row["condition"]["value"]) for row in records}):
        rows = [row for row in records if float(row["condition"]["value"]) == value]
        drift = np.asarray([
            row["held_out_drift"]["species_balanced_relative"] for row in rows
        ], dtype=float)
        coefficient = np.asarray([row["coefficient_error_relative"] for row in rows])
        groups.append({
            "observation_noise": value,
            "repetitions": len(rows),
            "species_balanced_drift_error": {
                "mean": float(np.mean(drift)), "median": float(np.median(drift)),
                "q25": float(np.quantile(drift, 0.25)),
                "q75": float(np.quantile(drift, 0.75)),
            },
            "coefficient_error_relative": {
                "mean": float(np.mean(coefficient)),
                "rmse": float(np.sqrt(np.mean(coefficient ** 2))),
                "median": float(np.median(coefficient)),
            },
            "boundary_selected_systems": int(sum(
                selection["selected_at_grid_boundary"]
                for row in rows for selection in row["selection"].values()
            )),
            "unconverged_candidates": int(sum(
                selection["unconverged_candidates"]
                for row in rows for selection in row["selection"].values()
            )),
            "nonfinite": sum(row["status"] != "finite" for row in rows),
        })
    positive = [row for row in groups if row["observation_noise"] > 0]
    slope = float(np.polyfit(
        np.log([row["observation_noise"] for row in positive]),
        np.log([row["species_balanced_drift_error"]["mean"] for row in positive]), 1,
    )[0])
    return {"groups": groups, "observation_noise_drift_log_log_slope": slope}


def weak_pairing_check(records, requested):
    weak = read_json(ROOT / requested["weak_sweep_records"])
    weak_rows = [row for row in weak if row["condition"]["sweep"] == "observation_noise"]
    keys = {(row["repetition"], float(row["condition"]["value"])): row for row in weak_rows}
    checks = []
    for row in records:
        key = (row["repetition"], float(row["condition"]["value"]))
        source = keys[key]
        passed = (
            row["condition"] == source["condition"]
            and row["training_seed"] == source["seeds"]["training"]
            and row["validation_seed"] == source["seeds"]["validation"]
            and row["test_seed"] == source["seeds"]["test"]
        )
        checks.append({"repetition": key[0], "observation_noise": key[1],
                       "passed": passed})
    return {"count": len(checks), "passed_count": sum(row["passed"] for row in checks),
            "passed": all(row["passed"] for row in checks), "checks": checks}


def snapshot_sources(out):
    hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    return hashes


def run(config_path, out, *, workers=1, smoke=False):
    requested = read_json(config_path)
    base = read_json(ROOT / requested["base_config"])
    validate_protocol(requested, base)
    spec = effective_protocol(requested, base, smoke=smoke)
    out = Path(out).resolve()
    allowed = (ROOT / "outputs").resolve()
    if allowed not in out.parents:
        raise ValueError(f"output must be beneath {allowed}")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, out / "requested_protocol.json")
    write_json(out / "effective_protocol.json", spec)
    hashes = snapshot_sources(out)
    started = perf_counter()
    repetitions = int(spec["evaluation"]["scientific_repetitions"])
    jobs = [(index, spec, requested) for index in range(repetitions)]
    results = []
    if workers == 1:
        for job in jobs:
            result = run_repetition(job)
            results.append(result)
            print(f"completed repetition {result['repetition']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_repetition, job): job[0] for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"completed repetition {result['repetition']}", flush=True)
    records = [row for result in results for row in result["records"]]
    records.sort(key=lambda row: (float(row["condition"]["value"]), row["repetition"]))
    pairing = weak_pairing_check(records, requested)
    write_json(out / "records.json", records)
    write_json(out / "summary.json", summarize(records))
    write_json(out / "pairing_check.json", pairing)
    quality = {
        "records": len(records),
        "nonfinite": sum(row["status"] != "finite" for row in records),
        "boundary_selected_systems": sum(
            selection["selected_at_grid_boundary"]
            for row in records for selection in row["selection"].values()
        ),
        "unconverged_candidates": sum(
            selection["unconverged_candidates"]
            for row in records for selection in row["selection"].values()
        ),
        "pairing_passed": pairing["passed"],
    }
    write_json(out / "quality_report.json", quality)
    artifacts = {name: digest(out / name) for name in (
        "requested_protocol.json", "effective_protocol.json", "records.json",
        "summary.json", "pairing_check.json", "quality_report.json",
    )}
    manifest = {
        "task": requested["task"],
        "status": "smoke_complete" if smoke else "scientific_complete",
        "smoke": bool(smoke), "record_count": len(records),
        "pairing_passed": pairing["passed"], "source_sha256": hashes,
        "artifact_sha256": artifacts, "elapsed_seconds": perf_counter() - started,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = "smoke" if args.smoke else "scientific"
        args.out = OUTPUT_ROOT / f"{label}-{stamp}"
    manifest = run(args.config.resolve(), args.out, workers=args.workers, smoke=args.smoke)
    print(json.dumps({"output": str(args.out), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
