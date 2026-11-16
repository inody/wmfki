"""Pilot the 1D six-atom process-noise and time-step sweeps at sigma_obs=0.01.

This pilot does not alter any manuscript artifact. It keeps the frozen dictionary,
seeds, weak tests, raw LASSO path, validation rule, and raw minimum-norm OLS
solver. Observation errors are paired across process-noise levels and, for the
time-step sweep, are subsampled from one finest-grid Gaussian array at shared
physical times.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_full_pair_scaling import matched_config as full_matched_config
from scripts.run_qanr_matched_lasso_sweeps import (
    _fit_full,
    _selection_record,
    apply_dictionary,
    design_diagnostics,
    read_json,
    validate_protocol,
)
from simulate import simulate_particles

DEFAULT_CONFIG = ROOT / "configs/experiments/qanr_redundant6_common_noise_pilot.json"
SOURCE_FILES = (
    "configs/experiments/qanr_redundant6_common_noise_pilot.json",
    "configs/experiments/qanr_redundant6_lasso_sweeps_R100.json",
    "scripts/run_qanr_common_noise_pilot.py",
    "scripts/run_qanr_matched_lasso_sweeps.py",
    "scripts/run_1d_hat_lasso.py",
    "scripts/run_full_pair_scaling.py",
    "scripts/prepare_1d_hat_experiment.py",
    "scripts/r05_numerical_settings.py",
    "scripts/run_messenger_qanr_sweep.py",
    "bandwidth.py", "config.py", "features.py", "kernels.py", "metrics.py",
    "pipeline.py", "regression.py", "simulate.py", "tests.py",
    "time_tests.py", "utils.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_pilot(protocol, base):
    validate_protocol(base)
    if protocol.get("task") != "qanr_redundant6_common_noise_pilot" \
            or protocol.get("protocol_version") != 1:
        raise ValueError("invalid common-noise pilot protocol")
    if float(protocol.get("observation_noise", -1.0)) != 0.01:
        raise ValueError("pilot observation noise must remain sigma_obs=0.01")
    if int(protocol.get("repetitions", -1)) != 10:
        raise ValueError("pilot must retain ten predeclared repetitions")
    process = protocol.get("process_noise", {})
    if process != {
        "particles": 100, "horizon": 2.0, "dt": 0.001,
        "values": [0.01, 0.05, 0.1, 0.2, 0.4],
        "training_seed_start": 1234, "training_seed_stride": 1000,
        "validation_seed_offset": 1000000, "solver": "rk4",
        "stacked_trajectories": 1,
    }:
        raise ValueError("process-noise pilot protocol changed")
    time = protocol.get("time_step", {})
    expected_time = {
        "particles": 200, "horizon": 10.0,
        "values": [0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
        "process_noise": 0.0,
        "training_seed_start": 1234, "training_seed_stride": 1000,
        "validation_seed_offset": 1000000, "solver": "euler",
        "stacked_trajectories": 3,
        "noise_pairing": (
            "one_common_0.001_time_lattice_standard_normal_array_per_split_and_"
            "repetition_subsampled_at_shared_times"
        ),
    }
    if time != expected_time:
        raise ValueError("time-step pilot protocol changed")


def _stacked_trajectory_seed(master_seed, trajectory_index):
    """Derive a stable stream per trajectory, independent of the time grid."""
    sequence = np.random.SeedSequence(
        [int(master_seed), int(trajectory_index), 0x51414E52]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def simulate_positions(block, *, particles, horizon, dt, sigma, seed, solver, repetitions, base):
    steps = int(round(horizon / dt))
    config = full_matched_config(
        particles=particles, repetitions=repetitions, steps=steps, dt=dt,
        sigma=sigma, seed=seed, solver=solver, time_test_type="bump",
    )
    config = apply_dictionary(config, base)
    simulation_kwargs = {
        "N": particles, "T": steps, "dt": dt, "sigma": sigma, "d": 1,
        "init": "gaussian_mixture", "init_centers": [0.5, 2.0, 3.5],
        "init_std": 0.005, "init_scale": 1.0,
        "kernel_params": config["simulation"]["interaction"], "solver": solver,
    }
    if sigma == 0.0 and repetitions > 1:
        # simulate_particles consumes one (zero-scaled) Gaussian array per Euler
        # step. A single R>1 call therefore gives later trajectories different
        # initial states on different time grids. Separate, derived streams keep
        # all stacked initial states exactly paired across the time-step sweep.
        trajectories = []
        times = None
        for trajectory_index in range(repetitions):
            positions, candidate_times, *_ = simulate_particles(
                R=1,
                seed=_stacked_trajectory_seed(seed, trajectory_index),
                **simulation_kwargs,
            )
            if times is not None and not np.array_equal(times, candidate_times):
                raise RuntimeError("stacked trajectory time grids differ")
            trajectories.append(positions)
            times = candidate_times
        positions = np.concatenate(trajectories, axis=0)
    else:
        positions, times, *_ = simulate_particles(
            R=repetitions, seed=seed, **simulation_kwargs
        )
    return config, positions, times


def fit_record(A, b, Av, bv, fit, truth, prefix):
    _, theta, selected = _selection_record(A, b, Av, bv, prefix["estimator"])
    ols = np.asarray(fit["theta_hat"], dtype=float)
    truth_norm = float(np.linalg.norm(truth))
    return {
        "coefficients": theta.tolist(),
        "ols_coefficients": ols.tolist(),
        "coefficient_error_relative": float(np.linalg.norm(theta - truth) / truth_norm),
        "ols_coefficient_error_relative": float(np.linalg.norm(ols - truth) / truth_norm),
        "ols_max_abs_coefficient": float(np.max(np.abs(ols))),
        **design_diagnostics(A),
        **selected,
    }


def process_worker(payload):
    repetition, protocol, base = payload
    block = protocol["process_noise"]
    train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
    validation_seed = train_seed + block["validation_seed_offset"]
    noise_scale = float(protocol["observation_noise"])
    truth = np.asarray(base["true_theta"], dtype=float)
    rows = []
    with threadpool_limits(limits=1):
        for sigma in block["values"]:
            config, X, t = simulate_positions(
                block, particles=block["particles"], horizon=block["horizon"],
                dt=block["dt"], sigma=float(sigma), seed=train_seed,
                solver=block["solver"], repetitions=block["stacked_trajectories"],
                base=base,
            )
            validation_config, Xv, tv = simulate_positions(
                block, particles=block["particles"], horizon=block["horizon"],
                dt=block["dt"], sigma=float(sigma), seed=validation_seed,
                solver=block["solver"], repetitions=block["stacked_trajectories"],
                base=base,
            )
            train_noise = np.random.default_rng(train_seed + 2).normal(size=X.shape)
            validation_noise = np.random.default_rng(validation_seed + 2).normal(size=Xv.shape)
            fit, A, b = _fit_full(config, X + noise_scale * train_noise, t)
            _, Av, bv = _fit_full(
                validation_config, Xv + noise_scale * validation_noise, tv
            )
            rows.append({
                "sweep": "process_noise", "repetition": repetition,
                "value": float(sigma), "observation_noise": noise_scale,
                "training_seed": train_seed, "validation_seed": validation_seed,
                **fit_record(A, b, Av, bv, fit, truth, base),
            })
    return rows


def time_worker(payload):
    repetition, protocol, base = payload
    block = protocol["time_step"]
    train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
    validation_seed = train_seed + block["validation_seed_offset"]
    noise_scale = float(protocol["observation_noise"])
    truth = np.asarray(base["true_theta"], dtype=float)
    # The greatest common time lattice is 0.001 because 0.005 is not an
    # integer multiple of the smallest tested step, 0.002.
    finest = 0.001
    fine_steps = int(round(block["horizon"] / finest))
    shape = (block["stacked_trajectories"], fine_steps + 1, block["particles"], 1)
    train_noise_fine = np.random.default_rng(train_seed + 2).normal(size=shape)
    validation_noise_fine = np.random.default_rng(validation_seed + 2).normal(size=shape)
    rows = []
    with threadpool_limits(limits=1):
        for dt in block["values"]:
            config, X, t = simulate_positions(
                block, particles=block["particles"], horizon=block["horizon"],
                dt=float(dt), sigma=block["process_noise"], seed=train_seed,
                solver=block["solver"], repetitions=block["stacked_trajectories"],
                base=base,
            )
            validation_config, Xv, tv = simulate_positions(
                block, particles=block["particles"], horizon=block["horizon"],
                dt=float(dt), sigma=block["process_noise"], seed=validation_seed,
                solver=block["solver"], repetitions=block["stacked_trajectories"],
                base=base,
            )
            stride = int(round(float(dt) / finest))
            train_noise = train_noise_fine[:, ::stride]
            validation_noise = validation_noise_fine[:, ::stride]
            if train_noise.shape != X.shape or validation_noise.shape != Xv.shape:
                raise RuntimeError("subsampled observation-noise shape mismatch")
            fit, A, b = _fit_full(config, X + noise_scale * train_noise, t)
            _, Av, bv = _fit_full(
                validation_config, Xv + noise_scale * validation_noise, tv
            )
            rows.append({
                "sweep": "time_step", "repetition": repetition,
                "value": float(dt), "observation_noise": noise_scale,
                "training_seed": train_seed, "validation_seed": validation_seed,
                "stacked_trajectory_seed_policy": (
                    "SeedSequence([master_seed, trajectory_index, 0x51414E52])"
                ),
                **fit_record(A, b, Av, bv, fit, truth, base),
            })
    return rows


def rmse(values):
    values = np.asarray(values, dtype=float)
    estimate = float(np.sqrt(np.mean(values ** 2)))
    if values.size < 2 or estimate == 0.0:
        se = 0.0
    else:
        se = float(np.std(values ** 2, ddof=1) / np.sqrt(values.size) / (2 * estimate))
    return {"rmse": estimate, "standard_error": se, "count": int(values.size)}


def summarize(records):
    result = {"conditions": [], "diagnostics": {}}
    for sweep in ("process_noise", "time_step"):
        subset = [row for row in records if row["sweep"] == sweep]
        for value in sorted({row["value"] for row in subset}):
            rows = [row for row in subset if row["value"] == value]
            result["conditions"].append({
                "sweep": sweep, "value": value,
                "weak_lasso": rmse([row["coefficient_error_relative"] for row in rows]),
                "weak_ols": rmse([row["ols_coefficient_error_relative"] for row in rows]),
            })
    process = [row for row in result["conditions"] if row["sweep"] == "process_noise"]
    time = [row for row in result["conditions"] if row["sweep"] == "time_step"
            and row["value"] <= 0.02]
    result["diagnostics"] = {
        "records": len(records),
        "nonfinite_records": int(sum(
            not np.isfinite(row["coefficient_error_relative"])
            or not np.isfinite(row["ols_coefficient_error_relative"])
            for row in records
        )),
        "rank_deficient_ols": int(sum(row["ols_rank_deficient"] for row in records)),
        "boundary_selections": int(sum(row["selected_at_grid_boundary"] for row in records)),
        "unconverged_path_candidates": int(sum(row["unconverged_candidates"] for row in records)),
        "lasso_process_noise_slope": float(np.polyfit(
            np.log([row["value"] for row in process]),
            np.log([row["weak_lasso"]["rmse"] for row in process]), 1
        )[0]),
        "ols_process_noise_slope": float(np.polyfit(
            np.log([row["value"] for row in process]),
            np.log([row["weak_ols"]["rmse"] for row in process]), 1
        )[0]),
        "lasso_time_step_slope_dt_le_0_02": float(np.polyfit(
            np.log([row["value"] for row in time]),
            np.log([row["weak_lasso"]["rmse"] for row in time]), 1
        )[0]),
        "ols_time_step_slope_dt_le_0_02": float(np.polyfit(
            np.log([row["value"] for row in time]),
            np.log([row["weak_ols"]["rmse"] for row in time]), 1
        )[0]),
        "lasso_better_than_ols_every_condition": all(
            row["weak_lasso"]["rmse"] < row["weak_ols"]["rmse"]
            for row in result["conditions"]
        ),
    }
    return result


def validate_cached_rows(name, repetition, rows, protocol, base):
    block = protocol[name]
    expected_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
    expected_validation_seed = expected_seed + block["validation_seed_offset"]
    if not isinstance(rows, list) or [row.get("value") for row in rows] != block["values"]:
        raise ValueError(f"cached {name} repetition {repetition} has the wrong grid")
    for row in rows:
        if (
            row.get("sweep") != name
            or row.get("repetition") != repetition
            or row.get("training_seed") != expected_seed
            or row.get("validation_seed") != expected_validation_seed
            or row.get("observation_noise") != protocol["observation_noise"]
            or len(row.get("coefficients", [])) != len(base["true_theta"])
            or len(row.get("ols_coefficients", [])) != len(base["true_theta"])
        ):
            raise ValueError(f"cached {name} repetition {repetition} is incompatible")
    if name == "time_step" and any(
        row.get("stacked_trajectory_seed_policy")
        != "SeedSequence([master_seed, trajectory_index, 0x51414E52])"
        for row in rows
    ):
        raise ValueError(f"cached {name} repetition {repetition} lacks paired streams")


def execute_group(name, worker, protocol, base, out, workers, resume):
    jobs = out / "jobs" / name
    jobs.mkdir(parents=True, exist_ok=True)
    rows = []
    pending = []
    for repetition in range(protocol["repetitions"]):
        path = jobs / f"rep-{repetition:03d}.json"
        if resume and path.exists():
            cached = read_json(path)
            validate_cached_rows(name, repetition, cached, protocol, base)
            rows.extend(cached)
        else:
            pending.append(((repetition, protocol, base), path))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, payload): path for payload, path in pending}
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                write_json(futures[future], result)
                rows.extend(result)
                completed += 1
                print(f"{name}: {completed}/{len(pending)} repetitions", flush=True)
    return rows


def run(config_path, out, workers=1, resume=False):
    requested = read_json(config_path)
    base = read_json(ROOT / requested["base_protocol"])
    validate_pilot(requested, base)
    out = Path(out)
    if resume and (out / "requested_protocol.json").exists() \
            and read_json(out / "requested_protocol.json") != requested:
        raise ValueError("resume protocol differs from the saved requested protocol")
    out.mkdir(parents=True, exist_ok=resume)
    write_json(out / "requested_protocol.json", requested)
    source_dir = out / "source"
    hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    records = execute_group(
        "process_noise", process_worker, requested, base, out, workers, resume
    )
    records.extend(execute_group(
        "time_step", time_worker, requested, base, out, workers, resume
    ))
    records.sort(key=lambda row: (row["sweep"], row["repetition"], row["value"]))
    summary = summarize(records)
    write_json(out / "records.json", records)
    write_json(out / "summary.json", summary)
    write_json(out / "manifest.json", {
        "task": requested["task"], "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records), "source_sha256": hashes,
        "artifact_sha256": {
            "requested_protocol.json": digest(out / "requested_protocol.json"),
            "records.json": digest(out / "records.json"),
            "summary.json": digest(out / "summary.json"),
        },
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "outputs/validation/qanr_common_noise_pilot" / f"pilot-{stamp}"
    summary = run(args.config, args.output, workers=args.workers, resume=args.resume)
    print(json.dumps({"output": str(args.output), **summary["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
