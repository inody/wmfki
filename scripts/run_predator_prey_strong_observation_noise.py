"""Add strong forward-difference OLS to the predator--prey noise sweep."""

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
    _ols,
    assemble_predator_prey_strong,
    relative_error,
)
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

DEFAULT_CONFIG = ROOT / "configs/experiments/predator_prey_strong_observation_noise.json"
SOURCE_FILES = (
    "configs/experiments/predator_prey_strong_observation_noise.json",
    "configs/experiments/predator_prey_sweeps.json",
    "scripts/run_predator_prey_strong_observation_noise.py",
    "scripts/verify_predator_prey_strong_observation_noise.py",
    "scripts/run_predator_prey_sweeps.py",
    "scripts/run_cross_example_nominal_comparison.py",
    "scripts/run_predator_prey_application.py",
    "test/test_predator_prey_strong_observation_noise.py",
    "predator_prey.py", "features.py", "tests.py", "time_tests.py", "kernels.py",
)
METHOD = "strong_forward_difference_ols"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(requested, base):
    if requested.get("task") != "predator_prey_strong_observation_noise" \
            or requested.get("protocol_version") != 1:
        raise ValueError("invalid predator--prey strong noise protocol")
    validate_base_protocol(base)
    if digest(ROOT / requested["base_config"]) != requested["base_config_sha256"]:
        raise ValueError("base predator--prey protocol hash changed")
    if digest(ROOT / requested["weak_sweep_records"]) \
            != requested["weak_sweep_records_sha256"]:
        raise ValueError("weak predator--prey sweep records changed")
    if requested.get("observation_noise_values") != base["sweeps"]["observation_noise"]:
        raise ValueError("observation-noise grid changed")
    if requested.get("repetitions") != base["evaluation"]["scientific_repetitions"]:
        raise ValueError("repetition count changed")
    if requested.get("training_trajectories") \
            != base["observation"]["training_trajectories"]:
        raise ValueError("training trajectory count changed")
    if requested.get("observation_step") != base["observation"]["dt"]:
        raise ValueError("observation step changed")
    strong = requested.get("strong_form", {})
    if strong.get("prey_system_columns") != 12 \
            or strong.get("predator_system_columns") != 5:
        raise ValueError("strong-form dictionary dimensions changed")
    if any(strong.get(key) for key in (
        "regularization", "intercept", "standardization", "coefficient_thresholding"
    )):
        raise ValueError("strong-form baseline must remain raw")


def effective_protocol(requested, base, smoke=False):
    spec = effective_base_protocol(base, smoke=smoke)
    spec["task"] = requested["task"]
    return spec


def observation_conditions(spec):
    return [
        condition for condition in condition_plan(spec)
        if condition["sweep"] == "observation_noise"
    ]


def run_repetition(payload):
    repetition, spec = payload
    started = perf_counter()
    prey_basis, predator_basis, _, truth, truth_kernels = model_objects(spec)
    truth_vector = np.concatenate(truth)
    training_inputs = paired_inputs(spec, repetition, "training")
    test_inputs = paired_inputs(spec, repetition, "test")
    training_cache, test_cache = {}, {}
    records = []
    with threadpool_limits(limits=1):
        for condition in observation_conditions(spec):
            (training_prey, training_predator, _), latent_dt = latent_for_condition(
                spec, condition, training_inputs, truth_kernels, training_cache
            )
            observed_prey, observed_predator, observed_time = observation_view(
                training_prey, training_predator, latent_dt,
                training_inputs["unit_prey_noise"],
                training_inputs["unit_predator_noise"],
                spec["pairing"]["dt_common_fine"],
                condition["observation_dt"], condition["observation_noise"],
            )
            A_prey, b_prey, A_predator, b_predator = assemble_predator_prey_strong(
                observed_prey, observed_predator, observed_time,
                prey_basis, predator_basis,
            )
            theta_prey, prey_diagnostics = _ols(A_prey, b_prey)
            theta_predator, predator_diagnostics = _ols(A_predator, b_predator)
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
            stride = _integer_ratio(
                spec["evaluation"]["drift_evaluation_interval"], test_dt
            )
            drift = species_balanced_drift_error(
                test_prey, test_predator, truth_kernels, estimated_kernels, stride
            )
            records.append({
                "repetition": repetition,
                "condition": condition,
                "method": METHOD,
                "training_seed": training_inputs["seed"],
                "test_seed": test_inputs["seed"],
                "coefficients": theta.tolist(),
                "coefficient_error_relative": relative_error(theta, truth_vector),
                "channels": coefficient_and_profile_metrics(
                    coefficients, truth, spec, prey_basis, predator_basis
                ),
                "held_out_drift": drift,
                "prey_system": prey_diagnostics,
                "predator_system": predator_diagnostics,
                "status": "finite" if prey_diagnostics["finite"]
                          and predator_diagnostics["finite"] else "nonfinite",
            })
    return {
        "repetition": repetition, "records": records,
        "elapsed_seconds": perf_counter() - started,
    }


def summarize(records):
    groups = []
    for value in sorted({float(row["condition"]["value"]) for row in records}):
        rows = [
            row for row in records if float(row["condition"]["value"]) == value
        ]
        drift = np.asarray([
            row["held_out_drift"]["species_balanced_relative"] for row in rows
        ], dtype=float)
        coefficient = np.asarray([
            row["coefficient_error_relative"] for row in rows
        ], dtype=float)
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
            "rank_deficient_prey": sum(
                row["prey_system"]["rank_deficient"] for row in rows
            ),
            "rank_deficient_predator": sum(
                row["predator_system"]["rank_deficient"] for row in rows
            ),
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
    weak_rows = [
        row for row in weak if row["condition"]["sweep"] == "observation_noise"
    ]
    keys = {
        (row["repetition"], float(row["condition"]["value"])): row
        for row in weak_rows
    }
    checks = []
    for row in records:
        key = (row["repetition"], float(row["condition"]["value"]))
        source = keys[key]
        passed = (
            row["condition"] == source["condition"]
            and row["training_seed"] == source["seeds"]["training"]
            and row["test_seed"] == source["seeds"]["test"]
        )
        checks.append({"repetition": key[0], "observation_noise": key[1],
                       "passed": passed})
    return {
        "count": len(checks), "passed_count": sum(item["passed"] for item in checks),
        "passed": all(item["passed"] for item in checks), "checks": checks,
    }


def snapshot_sources(out):
    hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    write_json(out / "source_manifest.json", {"sha256": hashes})
    return hashes


def run(config_path, out, *, workers=1, smoke=False):
    requested = read_json(config_path)
    base = read_json(ROOT / requested["base_config"])
    validate_protocol(requested, base)
    spec = effective_protocol(requested, base, smoke=smoke)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "requested_protocol.json", requested)
    write_json(out / "effective_protocol.json", spec)
    hashes = snapshot_sources(out)
    started = perf_counter()
    repetitions = int(spec["evaluation"]["scientific_repetitions"])
    results = []
    if workers == 1:
        for repetition in range(repetitions):
            results.append(run_repetition((repetition, spec)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_repetition, (repetition, spec)): repetition
                for repetition in range(repetitions)
            }
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda row: row["repetition"])
    records = [row for result in results for row in result["records"]]
    records.sort(key=lambda row: (
        float(row["condition"]["value"]), row["repetition"]
    ))
    pairing = weak_pairing_check(records, requested)
    write_json(out / "records.json", records)
    write_json(out / "summary.json", summarize(records))
    write_json(out / "pairing_check.json", pairing)
    quality = {
        "records": len(records),
        "nonfinite": sum(row["status"] != "finite" for row in records),
        "rank_deficient_prey": sum(
            row["prey_system"]["rank_deficient"] for row in records
        ),
        "rank_deficient_predator": sum(
            row["predator_system"]["rank_deficient"] for row in records
        ),
        "pairing_passed": pairing["passed"],
    }
    write_json(out / "quality_report.json", quality)
    artifacts = {
        name: digest(out / name) for name in (
            "requested_protocol.json", "effective_protocol.json", "source_manifest.json",
            "records.json", "summary.json", "pairing_check.json", "quality_report.json",
        )
    }
    manifest = {
        "task": requested["task"],
        "status": "smoke_complete" if smoke else "scientific_complete",
        "smoke": bool(smoke), "source_sha256": hashes,
        "record_count": len(records), "pairing_passed": pairing["passed"],
        "artifact_sha256": artifacts,
        "elapsed_seconds": perf_counter() - started,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = "smoke" if args.smoke else "scientific"
        args.out_dir = ROOT / "outputs/validation/predator_prey_strong_observation_noise" / f"{label}-{stamp}"
    result = run(args.config.resolve(), args.out_dir, workers=args.workers, smoke=args.smoke)
    print(json.dumps({"output": str(args.out_dir), "status": result["status"]}))


if __name__ == "__main__":
    main()
