"""Run the 100-repetition deterministic time-step sweep for Figure 2(d).

Each repetition stacks three independently initialized trajectories for training
and three independent trajectories for validation.  Seed indices are paired
across time steps.  The output retains every selected fit and convergence
diagnostic and reports relative coefficient RMSE with delta-method 95% intervals.
"""

from __future__ import annotations

import argparse
import csv
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

from scripts.run_full_pair_scaling import matched_config as full_matched_config
from scripts.run_qanr_matched_lasso_sweeps import (
    _fit_full,
    _selection_record,
    _simulate_full_system,
    apply_dictionary,
    design_diagnostics,
    read_json,
    validate_protocol,
)
from simulate import simulate_particles

DEFAULT_CONFIG = ROOT / "configs/experiments/qanr_redundant6_time_step_paired_R100.json"
LEGACY_CONFIG = ROOT / "configs/experiments/qanr_redundant6_time_step_R100.json"
PAIRED_TASK = "qanr_redundant6_time_step_paired_R100"
SEED_NAMESPACE = 0x51414E52
SEED_POLICY = {
    "name": "seedsequence_master_trajectory_namespace_v1",
    "namespace": SEED_NAMESPACE,
    "derivation": (
        "SeedSequence([split_master_seed, trajectory_index, namespace])"
        ".generate_state(1, uint64)"
    ),
    "paired_across_time_steps": True,
    "separate_training_validation_master_seeds": True,
}
SOURCE_FILES = (
    "scripts/run_qanr_time_step_repetitions.py",
    "scripts/verify_qanr_time_step_paired_R100.py",
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


def validate_time_protocol(protocol, base):
    validate_protocol(base)
    block = protocol.get("time_step", {})
    expected = {
        "particles": 200,
        "horizon": 10.0,
        "values": [0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
        "sigma": 0.0,
        "stacked_trajectories": 3,
        "repetitions": 100,
        "training_seed_start": 1234,
        "training_seed_stride": 1000,
        "validation_seed_offset": 1000000,
        "solver": "euler",
        "pair_mode": "full",
    }
    task = protocol.get("task")
    version = protocol.get("protocol_version")
    if task == "qanr_redundant6_time_step_R100" and version == 1:
        expected_block = expected
    elif task == PAIRED_TASK and version == 2:
        expected_block = {**expected, "trajectory_seed_policy": SEED_POLICY}
        if protocol.get("run_class") != "production":
            raise ValueError("paired time-step protocol is not marked production")
        if protocol.get("uncertainty", {}).get(
                "bitwise_paired_initial_states_across_time_steps") is not True:
            raise ValueError("paired time-step protocol lacks bitwise pairing requirement")
    else:
        raise ValueError("unknown time-step R100 protocol")
    if block != expected_block:
        raise ValueError("time-step R100 protocol changed")
    if protocol.get("true_theta") != base.get("true_theta"):
        raise ValueError("time-step truth differs from base six-atom protocol")
    if protocol.get("estimator") != base.get("estimator"):
        raise ValueError("time-step estimator differs from base six-atom protocol")


def effective_protocol(protocol, smoke=False):
    result = json.loads(json.dumps(protocol))
    if smoke:
        result["time_step"].update({
            "particles": 30, "horizon": 0.2, "values": [0.02, 0.01],
            "stacked_trajectories": (
                3 if protocol["task"] == PAIRED_TASK else 1
            ),
            "repetitions": 2,
        })
        result["estimator"].update({"alpha_count": 8, "max_iter": 20000})
        result["task"] = (
            f"{PAIRED_TASK}_smoke"
            if protocol["task"] == PAIRED_TASK
            else "qanr_redundant6_time_step_smoke"
        )
    return result


def array_digest(array):
    value = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": value.shape}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def trajectory_seed(master_seed, trajectory_index, policy=SEED_POLICY):
    sequence = np.random.SeedSequence([
        int(master_seed), int(trajectory_index), int(policy["namespace"])
    ])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def simulate_paired_positions(*, particles, repetitions, horizon, dt, sigma,
                              master_seed, solver, spec, policy):
    steps = int(round(horizon / dt))
    config = full_matched_config(
        particles=particles, repetitions=repetitions, steps=steps, dt=dt,
        sigma=sigma, seed=master_seed, solver=solver, time_test_type="bump",
    )
    config = apply_dictionary(config, spec)
    trajectories = []
    child_seeds = []
    initial_hashes = []
    times = None
    for trajectory_index in range(repetitions):
        child_seed = trajectory_seed(master_seed, trajectory_index, policy)
        positions, candidate_times, *_ = simulate_particles(
            N=particles, T=steps, dt=dt, R=1, sigma=sigma, d=1,
            init="gaussian_mixture", init_centers=[0.5, 2.0, 3.5],
            init_std=0.005, init_scale=1.0,
            kernel_params=config["simulation"]["interaction"],
            seed=child_seed, solver=solver,
        )
        if times is not None and not np.array_equal(times, candidate_times):
            raise RuntimeError("stacked trajectory time grids differ")
        trajectories.append(positions)
        child_seeds.append(child_seed)
        initial_hashes.append(array_digest(positions[0, 0]))
        times = candidate_times
    return (
        config, np.concatenate(trajectories, axis=0), times,
        child_seeds, initial_hashes,
    )


def _simulate_paired_full_system(**kwargs):
    config, positions, times, child_seeds, initial_hashes = simulate_paired_positions(
        **kwargs
    )
    fit, A, b = _fit_full(config, positions, times)
    return fit, A, b, child_seeds, initial_hashes


def run_one(payload):
    repetition, dt, protocol, base = payload
    block = protocol["time_step"]
    train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
    validation_seed = train_seed + block["validation_seed_offset"]
    truth = np.asarray(protocol["true_theta"], dtype=float)
    paired = protocol["task"] in {PAIRED_TASK, f"{PAIRED_TASK}_smoke"}
    with threadpool_limits(limits=1):
        if paired:
            common = {
                "particles": block["particles"],
                "repetitions": block["stacked_trajectories"],
                "horizon": block["horizon"], "dt": float(dt),
                "sigma": block["sigma"], "solver": block["solver"],
                "spec": base, "policy": block["trajectory_seed_policy"],
            }
            fit, A, b, train_children, train_hashes = _simulate_paired_full_system(
                master_seed=train_seed, **common
            )
            _, Av, bv, validation_children, validation_hashes = _simulate_paired_full_system(
                master_seed=validation_seed, **common
            )
        else:
            fit, A, b = _simulate_full_system(
                particles=block["particles"], repetitions=block["stacked_trajectories"],
                horizon=block["horizon"], dt=float(dt), sigma=block["sigma"],
                seed=train_seed, solver=block["solver"], spec=base,
            )
            _, Av, bv = _simulate_full_system(
                particles=block["particles"], repetitions=block["stacked_trajectories"],
                horizon=block["horizon"], dt=float(dt), sigma=block["sigma"],
                seed=validation_seed, solver=block["solver"], spec=base,
            )
            train_children = validation_children = None
            train_hashes = validation_hashes = None
        _, theta, selected = _selection_record(A, b, Av, bv, protocol["estimator"])
    ols_theta = np.asarray(fit["theta_hat"], dtype=float)
    truth_norm = float(np.linalg.norm(truth))
    record = {
        "repetition": int(repetition), "dt": float(dt),
        "training_seed": int(train_seed), "validation_seed": int(validation_seed),
        "coefficients": theta.tolist(), "ols_coefficients": ols_theta.tolist(),
        "coefficient_error_relative": float(np.linalg.norm(theta - truth) / truth_norm),
        "ols_coefficient_error_relative": float(np.linalg.norm(ols_theta - truth) / truth_norm),
        "ols_max_abs_coefficient": float(np.max(np.abs(ols_theta))),
        **design_diagnostics(A), **selected,
    }
    if paired:
        record.update({
            "trajectory_seed_policy": block["trajectory_seed_policy"]["name"],
            "training_trajectory_seeds": train_children,
            "validation_trajectory_seeds": validation_children,
            "training_initial_state_sha256": train_hashes,
            "validation_initial_state_sha256": validation_hashes,
        })
    return record


def rmse_ci(values):
    values = np.asarray(values, dtype=float)
    squared = values ** 2
    estimate = float(np.sqrt(np.mean(squared)))
    if values.size < 2 or estimate == 0.0:
        interval = [estimate, estimate]
    else:
        se = float(np.std(squared, ddof=1) / np.sqrt(values.size) / (2.0 * estimate))
        interval = [max(0.0, estimate - 1.96 * se), estimate + 1.96 * se]
    return {
        "rmse": estimate, "ci95": interval, "count": int(values.size),
        "ci_method": "delta_method_on_mean_squared_relative_error",
    }


def summarize(records):
    groups = []
    for dt in sorted({float(row["dt"]) for row in records}):
        subset = [row for row in records if float(row["dt"]) == dt]
        groups.append({
            "dt": dt,
            "lasso_relative_rmse": rmse_ci(
                [row["coefficient_error_relative"] for row in subset]
            ),
            "ols_relative_rmse": rmse_ci(
                [row["ols_coefficient_error_relative"] for row in subset]
            ),
            "repetitions": len(subset),
        })
    positive = [row for row in groups if row["dt"] <= 0.02]
    x = np.log([row["dt"] for row in positive])
    diagnostics = {
        "models": len(records),
        "boundary_selections": int(sum(row["selected_at_grid_boundary"] for row in records)),
        "unconverged_path_candidates": int(sum(row["unconverged_candidates"] for row in records)),
        "rank_deficient": int(sum(row["ols_rank_deficient"] for row in records)),
        "missing_or_nonfinite": int(sum(
            not np.isfinite(row["coefficient_error_relative"])
            or not np.isfinite(row["ols_coefficient_error_relative"])
            for row in records
        )),
        "lasso_log_log_slope_dt_le_0_02": float(np.polyfit(
            x, np.log([row["lasso_relative_rmse"]["rmse"] for row in positive]), 1
        )[0]),
        "ols_log_log_slope_dt_le_0_02": float(np.polyfit(
            x, np.log([row["ols_relative_rmse"]["rmse"] for row in positive]), 1
        )[0]),
    }
    return {"time_step": groups, "diagnostics": diagnostics}


def validate_cached_record(record, repetition, dt, protocol):
    block = protocol["time_step"]
    train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
    validation_seed = train_seed + block["validation_seed_offset"]
    if (
        record.get("repetition") != repetition
        or record.get("dt") != float(dt)
        or record.get("training_seed") != train_seed
        or record.get("validation_seed") != validation_seed
        or len(record.get("coefficients", [])) != len(protocol["true_theta"])
        or len(record.get("ols_coefficients", [])) != len(protocol["true_theta"])
    ):
        raise ValueError(f"cached record is incompatible: repetition={repetition}, dt={dt}")
    if protocol["task"] in {PAIRED_TASK, f"{PAIRED_TASK}_smoke"}:
        count = block["stacked_trajectories"]
        expected_train = [trajectory_seed(train_seed, index, block["trajectory_seed_policy"])
                          for index in range(count)]
        expected_validation = [
            trajectory_seed(validation_seed, index, block["trajectory_seed_policy"])
            for index in range(count)
        ]
        if (
            record.get("trajectory_seed_policy")
            != block["trajectory_seed_policy"]["name"]
            or record.get("training_trajectory_seeds") != expected_train
            or record.get("validation_trajectory_seeds") != expected_validation
            or len(record.get("training_initial_state_sha256", [])) != count
            or len(record.get("validation_initial_state_sha256", [])) != count
        ):
            raise ValueError(f"cached paired record is incompatible: repetition={repetition}, dt={dt}")


def snapshot_sources(out, config_path, base_path):
    source = Path(out) / "source"
    hashes = {}
    relatives = list(SOURCE_FILES)
    for path in (Path(config_path), Path(base_path)):
        relative = str(path.resolve().relative_to(ROOT))
        if relative not in relatives:
            relatives.append(relative)
    for relative in relatives:
        src = ROOT / relative
        dst = source / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[relative] = digest(src)
    return hashes


def run(config_path, out, workers=1, smoke=False, resume=False):
    requested = read_json(config_path)
    base_path = ROOT / requested["base_protocol"]
    base = read_json(base_path)
    validate_time_protocol(requested, base)
    protocol = effective_protocol(requested, smoke=smoke)
    if smoke:
        base = json.loads(json.dumps(base))
        base["estimator"] = protocol["estimator"]
    out = Path(out)
    previous_manifest = (
        read_json(out / "manifest.json")
        if resume and (out / "manifest.json").exists() else None
    )
    if resume and (out / "requested_protocol.json").exists() \
            and read_json(out / "requested_protocol.json") != requested:
        raise ValueError("resume protocol differs from the saved requested protocol")
    out.mkdir(parents=True, exist_ok=resume)
    jobs_dir = out / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    write_json(out / "requested_protocol.json", requested)
    write_json(out / "effective_protocol.json", protocol)
    source_hashes = snapshot_sources(out, config_path, base_path)
    invocation_started_utc = datetime.now(timezone.utc).isoformat()
    started_utc = (
        previous_manifest.get("started_utc", invocation_started_utc)
        if previous_manifest else invocation_started_utc
    )
    prior_elapsed = (
        float(previous_manifest.get("elapsed_seconds", 0.0))
        if previous_manifest else 0.0
    )
    started = perf_counter()
    payloads = [
        (repetition, dt, protocol, base)
        for repetition in range(protocol["time_step"]["repetitions"])
        for dt in protocol["time_step"]["values"]
    ]
    records = []
    missing = []
    for payload in payloads:
        repetition, dt, *_ = payload
        index = protocol["time_step"]["values"].index(dt)
        path = jobs_dir / f"rep-{repetition:03d}-dt-{index:02d}.json"
        if resume and path.exists():
            cached = read_json(path)
            validate_cached_record(cached, repetition, dt, protocol)
            records.append(cached)
        else:
            missing.append((payload, path))
    if missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, payload): path for payload, path in missing}
            completed = 0
            for future in as_completed(futures):
                record = future.result()
                write_json(futures[future], record)
                records.append(record)
                completed += 1
                if completed % 10 == 0 or completed == len(missing):
                    print(f"time-step jobs: {completed}/{len(missing)} new", flush=True)
    records.sort(key=lambda row: (row["repetition"], row["dt"]))
    summary = summarize(records)
    write_json(out / "records.json", records)
    write_json(out / "summary.json", summary)
    with (out / "records.csv").open("w", newline="") as handle:
        columns = [
            "repetition", "dt", "training_seed", "validation_seed",
            "coefficient_error_relative", "ols_coefficient_error_relative",
            "selected_at_grid_boundary", "unconverged_candidates", "ols_rank_deficient",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in records])
    artifacts = {
        name: digest(out / name)
        for name in ("requested_protocol.json", "effective_protocol.json", "records.json",
                     "records.csv", "summary.json")
    }
    job_hashes = {
        str(path.relative_to(out)): digest(path)
        for path in sorted(jobs_dir.glob("rep-*-dt-*.json"))
    }
    write_json(out / "manifest.json", {
        "task": protocol["task"], "smoke": bool(smoke),
        "started_utc": started_utc,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": prior_elapsed + perf_counter() - started,
        "workers": (
            int(previous_manifest.get("workers", workers))
            if previous_manifest else int(workers)
        ),
        "last_invocation": {
            "started_utc": invocation_started_utc,
            "workers": int(workers), "resume": bool(resume),
        },
        "records": len(records), "source_sha256": source_hashes,
        "artifact_sha256": artifacts, "job_sha256": job_hashes,
    })
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = "smoke" if args.smoke else "sweeps"
        output_family = (
            "qanr_redundant6_time_step_paired_R100"
            if read_json(args.config)["task"] == PAIRED_TASK
            else "qanr_redundant6_time_step_R100"
        )
        args.output = ROOT / "outputs/validation" / output_family / f"{label}-{stamp}"
    summary = run(args.config, args.output, workers=args.workers,
                  smoke=args.smoke, resume=args.resume)
    print(json.dumps({"output": str(args.output), "diagnostics": summary["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
