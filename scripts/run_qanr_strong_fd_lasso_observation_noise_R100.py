"""Run validation-selected strong FD-LASSO over the Figure 2(a) noise grid."""

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

from features import compute_increment_regression_system
from scripts.run_qanr_matched_lasso_sweeps import apply_dictionary
from scripts.run_regularized_baseline_exploration import select_fd_lasso
from scripts.run_sindy_comparison import (
    DETERMINISTIC_SPACE_CENTERS,
    SPACE_TEST_WIDTH,
    TIME_TEST_SUPPORT,
    experiment_config as observation_config,
    reference_trajectory,
)

DEFAULT_CONFIG = ROOT / "configs/experiments/qanr_strong_fd_lasso_observation_noise_R100.json"
OUTPUT_ROOT = ROOT / "outputs/validation/qanr_strong_fd_lasso_observation_noise"
SOURCE_FILES = (
    "configs/experiments/qanr_strong_fd_lasso_observation_noise_R100.json",
    "scripts/run_qanr_strong_fd_lasso_observation_noise_R100.py",
    "scripts/run_regularized_baseline_exploration.py",
    "features.py", "kernels.py", "simulate.py",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(spec, base):
    if (spec.get("task"), spec.get("protocol_version")) != (
        "qanr_strong_fd_lasso_observation_noise_R100", 1
    ):
        raise ValueError("invalid QANR strong FD-LASSO protocol")
    if spec.get("study_status") != "manuscript_current":
        raise ValueError("QANR strong FD-LASSO must use the current manuscript protocol")
    for key in ("base_config", "production_config", "current_summary"):
        if digest(ROOT / spec[key]) != spec[f"{key}_sha256"]:
            raise ValueError(f"pinned input changed: {key}")
    block = base["observation_noise"]
    if spec["observation_noise_values"] != block["sigma_obs_values"]:
        raise ValueError("observation-noise grid changed")
    if spec["repetitions"] != block["repetitions"] or spec["repetitions"] != 100:
        raise ValueError("repetition count changed")
    estimator = spec["estimator"]
    if estimator["fit_intercept"] or estimator["standardize"]:
        raise ValueError("strong FD-LASSO must use raw columns and no intercept")


def relative_error(theta, truth):
    return float(np.linalg.norm(theta - truth) / np.linalg.norm(truth))


def run_repetition(payload):
    index, spec, base = payload
    block = base["observation_noise"]
    train_seed = int(block["training_seed_start"] + block["training_seed_stride"] * index)
    validation_seed = train_seed + 1000000
    started = perf_counter()
    records = []
    with threadpool_limits(limits=1):
        X, time, *_ = reference_trajectory(
            train_seed, particles=block["particles"], horizon=block["horizon"],
            step=block["dt"], sigma=block["sigma"],
        )
        Xv, tv, *_ = reference_trajectory(
            validation_seed, particles=block["particles"], horizon=block["horizon"],
            step=block["dt"], sigma=block["sigma"],
        )
        train_noise = np.random.default_rng(train_seed + 2).normal(size=X.shape)
        validation_noise = np.random.default_rng(validation_seed + 2).normal(size=Xv.shape)
        steps = int(round(block["horizon"] / block["dt"]))
        # This is the same observation configuration used by the verified nominal
        # reconstruction in the regularized-baseline experiment.
        config = apply_dictionary(observation_config(
            block["particles"], steps, block["dt"], block["sigma"], train_seed,
            DETERMINISTIC_SPACE_CENTERS, SPACE_TEST_WIDTH, TIME_TEST_SUPPORT, "bump",
        ), base)
        from kernels import make_kernel_basis
        basis, _ = make_kernel_basis(config["kernel_basis"])
        truth = np.asarray(spec.get("truth", [1.0, -1.0, 0.0, 0.0, 0.0, 0.0]), dtype=float)
        for sigma_obs in spec["observation_noise_values"]:
            observed = X + float(sigma_obs) * train_noise
            observed_v = Xv + float(sigma_obs) * validation_noise
            A, b = compute_increment_regression_system(observed, time, basis)
            Av, bv = compute_increment_regression_system(observed_v, tv, basis)
            theta, selection = select_fd_lasso(A, b, Av, bv, spec["estimator"])
            records.append({
                "repetition": index,
                "observation_noise": float(sigma_obs),
                "method": "strong_forward_difference_lasso_validation_selected",
                "training_seed": train_seed,
                "validation_seed": validation_seed,
                "coefficients": theta.tolist(),
                "coefficient_error_relative": relative_error(theta, truth),
                "selection": selection,
                "status": "finite" if np.all(np.isfinite(theta)) else "failed",
            })
    return {"repetition": index, "records": records,
            "elapsed_seconds": perf_counter() - started}


def bootstrap_rmse(values, draws, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = np.sqrt(np.mean(values[indices] ** 2, axis=1))
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def summarize(records, spec):
    groups = []
    for noise_index, sigma_obs in enumerate(spec["observation_noise_values"]):
        rows = sorted(
            [row for row in records if row["observation_noise"] == sigma_obs],
            key=lambda row: row["repetition"],
        )
        values = np.asarray([row["coefficient_error_relative"] for row in rows], dtype=float)
        groups.append({
            "observation_noise": float(sigma_obs),
            "method": "strong_forward_difference_lasso_validation_selected",
            "coefficient_error_relative": {
                "values": values.tolist(),
                "rmse": float(np.sqrt(np.mean(values ** 2))),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "rmse_bootstrap_95": bootstrap_rmse(
                    values, int(spec["bootstrap"]["draws"]),
                    int(spec["bootstrap"]["seed"]) + noise_index,
                ),
            },
            "failures": int(sum(row["status"] != "finite" for row in rows)),
            "boundary_selections": int(sum(
                row["selection"]["selected_at_grid_boundary"] for row in rows
            )),
            "unconverged_candidates": int(sum(
                row["selection"]["unconverged_candidates"] for row in rows
            )),
        })
    return {"groups": groups, "record_count": len(records)}


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
    spec = read_json(config_path)
    base = read_json(ROOT / spec["base_config"])
    validate_protocol(spec, base)
    effective = json.loads(json.dumps(spec))
    if smoke:
        effective["repetitions"] = 1
        effective["observation_noise_values"] = [0.0, 0.01]
        effective["estimator"]["alpha_count"] = 8
        effective["bootstrap"]["draws"] = 100
    out = Path(out).resolve()
    allowed = (ROOT / "outputs").resolve()
    if allowed not in out.parents:
        raise ValueError(f"output must be beneath {allowed}")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, out / "requested_protocol.json")
    write_json(out / "effective_protocol.json", effective)
    source_hashes = snapshot_sources(out)
    started = perf_counter()
    jobs = [(index, effective, base) for index in range(effective["repetitions"])]
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
    records.sort(key=lambda row: (row["observation_noise"], row["repetition"]))
    summary = summarize(records, effective)
    write_json(out / "records.json", records)
    write_json(out / "summary.json", summary)
    diagnostics = {
        "failures": sum(row["status"] != "finite" for row in records),
        "boundary_selections": sum(
            row["selection"]["selected_at_grid_boundary"] for row in records
        ),
        "unconverged_candidates": sum(
            row["selection"]["unconverged_candidates"] for row in records
        ),
    }
    write_json(out / "diagnostics.json", diagnostics)
    artifacts = {name: digest(out / name) for name in (
        "requested_protocol.json", "effective_protocol.json", "records.json",
        "summary.json", "diagnostics.json",
    )}
    manifest = {
        "task": spec["task"],
        "status": "smoke_complete" if smoke else "scientific_complete",
        "smoke": bool(smoke), "record_count": len(records),
        "source_sha256": source_hashes, "artifact_sha256": artifacts,
        "elapsed_seconds": perf_counter() - started,
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
    result = run(args.config.resolve(), args.out, workers=args.workers, smoke=args.smoke)
    print(json.dumps({"output": str(args.out), "status": result["status"]}))


if __name__ == "__main__":
    main()
