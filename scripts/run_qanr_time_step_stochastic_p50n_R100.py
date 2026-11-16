"""Run the stochastic fixed-step P=50N observation-interval study for Figure 2(d)."""

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

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR,
    SECONDARY_COLOR,
    apply_manuscript_style,
    finish_axes,
)
from scripts.run_qanr_matched_lasso_sweeps import read_json, validate_protocol
from scripts.run_qanr_sampled_ratio_screen import (
    PANEL_CODES,
    assemble_nested_prefix_designs,
    derived_seed,
    fit_sampled_models,
    matched_config,
    method_record,
    prepare_weak_system,
    sampled_drift_error,
    write_json,
)
from scripts.run_qanr_time_step_repetitions import simulate_paired_positions

DEFAULT_CONFIG = ROOT / "configs/experiments/qanr_redundant6_time_step_stochastic_p50n_R100.json"
SOURCE_FILES = (
    "configs/experiments/qanr_redundant6_time_step_stochastic_p50n_R100.json",
    "configs/experiments/qanr_redundant6_lasso_sweeps_R100.json",
    "scripts/run_qanr_time_step_stochastic_p50n_R100.py",
    "scripts/run_qanr_sampled_ratio_screen.py",
    "scripts/run_qanr_time_step_repetitions.py",
    "scripts/run_qanr_matched_lasso_sweeps.py",
    "scripts/run_full_pair_scaling.py",
    "simulate.py", "features.py", "pipeline.py", "regression.py",
    "kernels.py", "tests.py", "time_tests.py", "utils.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_spec(spec: dict, base: dict) -> None:
    validate_protocol(base)
    if spec.get("task") != "qanr_redundant6_time_step_stochastic_p50n_R100":
        raise ValueError("unexpected task")
    if spec.get("protocol_version") != 1 or spec.get("repetitions") != 100:
        raise ValueError("production protocol requires R=100")
    if spec.get("particles") != 200 or spec.get("horizon") != 10.0:
        raise ValueError("N or horizon changed")
    if spec.get("simulation_step") != 0.001:
        raise ValueError("fixed simulation step must be 0.001")
    expected_dt = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    if spec.get("observation_intervals") != expected_dt:
        raise ValueError("observation-interval grid changed")
    if spec.get("process_noise") != 0.1 or spec.get("observation_noise") != 0.0:
        raise ValueError("stochastic exact-position controls changed")
    if spec.get("solver") != "euler":
        raise ValueError("Euler--Maruyama solver required")
    if (spec.get("stacked_training_trajectories") != 3
            or spec.get("stacked_validation_trajectories") != 3):
        raise ValueError("three training and validation trajectories required")
    pair = spec["pair_sampling"]
    expected_pair = {
        "ratio": 50, "pairs": 10000, "ordered": True,
        "exclude_diagonal": True, "with_replacement": True,
        "resample_each_observed_time": True,
        "shared_across_tests_and_atoms": True,
        "independent_training_validation": True,
        "independent_across_observation_intervals": True,
    }
    if pair != expected_pair:
        raise ValueError("pair-sampling contract changed")
    if spec["estimator"] != base["estimator"]:
        raise ValueError("estimator differs from the frozen six-atom protocol")
    h = float(spec["simulation_step"])
    for dt in expected_dt:
        stride = dt / h
        if abs(stride - round(stride)) > 1e-12:
            raise ValueError(f"observation interval {dt} is not divisible by h")


def effective_spec(spec: dict, smoke: bool) -> dict:
    result = json.loads(json.dumps(spec))
    if smoke:
        result.update({
            "task": "qanr_redundant6_time_step_stochastic_p50n_smoke",
            "particles": 30,
            "horizon": 0.2,
            "repetitions": 2,
            "stacked_training_trajectories": 1,
            "stacked_validation_trajectories": 1,
        })
        result["observation_intervals"] = [0.02, 0.01, 0.005, 0.002]
        result["pair_sampling"]["pairs"] = 1500
        result["estimator"].update({"alpha_count": 8, "max_iter": 20000})
        result["slope_window"]["paired_bootstrap_resamples"] = 100
    return result


def snapshot_sources(out: Path, config_path: Path, base_path: Path) -> dict:
    files = list(SOURCE_FILES)
    for path in (config_path, base_path):
        relative = str(path.resolve().relative_to(ROOT))
        if relative not in files:
            files.append(relative)
    hashes = {}
    for relative in files:
        source = ROOT / relative
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    return hashes


def pair_seeds(spec: dict, repetition: int, condition_index: int,
               trajectory_seed: int) -> tuple[int, int]:
    train = derived_seed(
        spec["pair_seed_start"] + trajectory_seed,
        PANEL_CODES["time_step"], repetition, condition_index, 0,
    )
    validation = derived_seed(
        spec["pair_seed_start"] + spec["pair_validation_seed_offset"]
        + trajectory_seed,
        PANEL_CODES["time_step"], repetition, condition_index, 1,
    )
    if train == validation:
        raise RuntimeError("training and validation pair seeds coincide")
    return train, validation


def simulate_fine(spec: dict, base: dict, master_seed: int,
                  trajectories: int):
    return simulate_paired_positions(
        particles=spec["particles"], repetitions=trajectories,
        horizon=spec["horizon"], dt=spec["simulation_step"],
        sigma=spec["process_noise"], master_seed=master_seed,
        solver=spec["solver"], spec=base,
        policy=spec["trajectory_seed_policy"],
    )


def run_repetition(payload):
    repetition, spec, base = payload
    train_seed = spec["training_seed_start"] + spec["training_seed_stride"] * repetition
    validation_seed = train_seed + spec["validation_seed_offset"]
    truth = np.asarray(base["true_theta"], dtype=float)
    block = {
        "dt": spec["simulation_step"], "horizon": spec["horizon"],
        "solver": spec["solver"],
    }
    with threadpool_limits(limits=1):
        _, train_fine, fine_time, train_children, train_hashes = simulate_fine(
            spec, base, train_seed, spec["stacked_training_trajectories"]
        )
        _, validation_fine, validation_fine_time, validation_children, validation_hashes = simulate_fine(
            spec, base, validation_seed, spec["stacked_validation_trajectories"]
        )
        if not np.array_equal(fine_time, validation_fine_time):
            raise RuntimeError("training and validation fine grids differ")
        records, audits = [], []
        h = float(spec["simulation_step"])
        for condition_index, dt in enumerate(spec["observation_intervals"]):
            stride = int(round(float(dt) / h))
            train_positions = train_fine[:, ::stride]
            validation_positions = validation_fine[:, ::stride]
            observed_time = fine_time[::stride]
            expected_count = int(round(spec["horizon"] / float(dt))) + 1
            if observed_time.size != expected_count:
                raise RuntimeError("subsampled time grid has the wrong length")
            config = matched_config(
                block, spec["particles"], spec["process_noise"], train_seed,
                base, dt=float(dt),
                repetitions=spec["stacked_training_trajectories"],
            )
            train_pair_seed, validation_pair_seed = pair_seeds(
                spec, repetition, condition_index, train_seed
            )
            train_system = prepare_weak_system(
                config, train_positions, observed_time, train_pair_seed
            )
            validation_config = json.loads(json.dumps(config))
            validation_config["simulation"]["seed"] = validation_seed
            validation_system = prepare_weak_system(
                validation_config, validation_positions, observed_time,
                validation_pair_seed,
            )
            ratio = int(spec["pair_sampling"]["ratio"])
            train_designs, train_audit = assemble_nested_prefix_designs(
                train_positions, observed_time, train_system, [ratio], train_pair_seed
            )
            validation_designs, validation_audit = assemble_nested_prefix_designs(
                validation_positions, observed_time, validation_system, [ratio],
                validation_pair_seed,
            )
            fitted = fit_sampled_models(
                train_designs[ratio], np.asarray(train_system["b_flat"], dtype=float),
                validation_designs[ratio], np.asarray(validation_system["b_flat"], dtype=float),
                spec["estimator"], truth,
            )
            drift_seed = derived_seed(
                train_seed, PANEL_CODES["time_step"], repetition,
                condition_index, 991,
            )
            for method in ("lasso", "ols"):
                theta = fitted[method].get("theta")
                drift = None if theta is None else sampled_drift_error(
                    theta, validation_positions, validation_system["kernel_basis"],
                    truth, drift_seed, spec["drift_evaluation_samples"],
                )
                row = method_record(
                    "time_step", dt, repetition, ratio, spec["particles"],
                    method, fitted[method], pair_seed=train_pair_seed,
                    validation_pair_seed=validation_pair_seed,
                    drift_error=drift,
                )
                row.update({
                    "simulation_step": h,
                    "process_noise": spec["process_noise"],
                    "observation_noise": spec["observation_noise"],
                    "observation_stride": stride,
                    "shared_latent_path_across_observation_intervals": True,
                })
                records.append(row)
            audits.append({
                "observation_interval": float(dt),
                "observation_stride": stride,
                "training_pair_seed": train_pair_seed,
                "validation_pair_seed": validation_pair_seed,
                "training_sampling": train_audit,
                "validation_sampling": validation_audit,
            })
    return {
        "repetition": repetition,
        "training_master_seed": train_seed,
        "validation_master_seed": validation_seed,
        "training_trajectory_seeds": train_children,
        "validation_trajectory_seeds": validation_children,
        "training_initial_state_sha256": train_hashes,
        "validation_initial_state_sha256": validation_hashes,
        "records": records,
        "sampling_audits": audits,
    }


def rmse_summary(values) -> dict:
    values = np.asarray(values, dtype=float)
    squared = values ** 2
    estimate = float(np.sqrt(np.mean(squared)))
    if values.size < 2 or estimate == 0.0:
        se = 0.0
    else:
        se = float(np.std(squared, ddof=1) / np.sqrt(values.size) / (2 * estimate))
    return {
        "rmse": estimate,
        "standard_error": se,
        "ci95": [max(0.0, estimate - 1.96 * se), estimate + 1.96 * se],
        "count": int(values.size),
    }


def summarize(spec: dict, records: list[dict]) -> list[dict]:
    rows = []
    for dt in spec["observation_intervals"]:
        for method in ("lasso", "ols"):
            subset = [
                row for row in records
                if float(row["value"]) == float(dt) and row["method"] == method
            ]
            coefficient = [float(row["relative_coefficient_error"]) for row in subset]
            drift = [float(row["validation_drift_error"]) for row in subset]
            rows.append({
                "observation_interval": float(dt),
                "method": method,
                "coefficient": rmse_summary(coefficient),
                "validation_drift": rmse_summary(drift),
                "failed": int(sum(row["status"] != "selected" for row in subset)),
                "boundary_selections": int(sum(bool(row.get("selected_at_grid_boundary")) for row in subset)),
                "unconverged_path_candidates": int(sum(int(row.get("unconverged_candidates") or 0) for row in subset)),
            })
    return rows


def slope_report(spec: dict, records: list[dict]) -> dict:
    maximum = float(spec["slope_window"]["maximum_observation_interval"])
    values = sorted(float(dt) for dt in spec["observation_intervals"] if dt <= maximum)
    report = {"values": values}
    repetitions = int(spec["repetitions"])
    for method in ("lasso", "ols"):
        by_dt = {
            dt: np.asarray([
                float(row["relative_coefficient_error"])
                for row in sorted(records, key=lambda item: item["repetition"])
                if row["method"] == method and float(row["value"]) == dt
            ]) for dt in values
        }
        if any(array.size != repetitions for array in by_dt.values()):
            raise ValueError(f"incomplete slope input for {method}")
        rmse = np.asarray([np.sqrt(np.mean(by_dt[dt] ** 2)) for dt in values])
        point = float(np.polyfit(np.log(values), np.log(rmse), 1)[0])
        rng = np.random.default_rng(
            int(spec["slope_window"]["bootstrap_seed"]) + (0 if method == "lasso" else 1)
        )
        count = int(spec["slope_window"]["paired_bootstrap_resamples"])
        slopes = np.empty(count, dtype=float)
        for index in range(count):
            sampled = rng.integers(0, repetitions, size=repetitions)
            sampled_rmse = np.asarray([
                np.sqrt(np.mean(by_dt[dt][sampled] ** 2)) for dt in values
            ])
            slopes[index] = np.polyfit(np.log(values), np.log(sampled_rmse), 1)[0]
        report[method] = {
            "coefficient_rmse": rmse.tolist(),
            "coefficient_rmse_slope": point,
            "coefficient_mse_slope": 2.0 * point,
            "bootstrap_ci95": [float(x) for x in np.quantile(slopes, [0.025, 0.975])],
            "bootstrap_resamples": count,
        }
    return report


def verify(spec: dict, records: list[dict], results: list[dict]) -> dict:
    expected = int(spec["repetitions"]) * len(spec["observation_intervals"]) * 2
    issues = []
    if len(records) != expected:
        issues.append(f"expected {expected} records, found {len(records)}")
    expected_p = int(spec["pair_sampling"]["ratio"]) * int(spec["particles"])
    if any(row.get("P") != expected_p for row in records):
        issues.append("not every record uses P=50N")
    if any(row.get("pair_seed") == row.get("validation_pair_seed") for row in records):
        issues.append("training and validation pair seeds overlap")
    if any(float(row.get("simulation_step")) != float(spec["simulation_step"]) for row in records):
        issues.append("simulation step changed")
    if any(float(row.get("process_noise")) != float(spec["process_noise"]) for row in records):
        issues.append("process noise changed")
    if any(float(row.get("observation_noise")) != 0.0 for row in records):
        issues.append("observation noise is not zero")
    for result in results:
        if len(set(result["training_initial_state_sha256"])) != len(result["training_initial_state_sha256"]):
            issues.append(f"duplicate training initial state in repetition {result['repetition']}")
        if len(set(result["validation_initial_state_sha256"])) != len(result["validation_initial_state_sha256"]):
            issues.append(f"duplicate validation initial state in repetition {result['repetition']}")
    return {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "expected_records": expected,
        "actual_records": len(records),
        "all_records_use_P_equals_50N": not any(row.get("P") != expected_p for row in records),
        "independent_training_validation_pair_streams": not any(
            row.get("pair_seed") == row.get("validation_pair_seed") for row in records
        ),
        "fixed_simulation_step": spec["simulation_step"],
        "shared_latent_paths_across_observation_intervals": True,
    }


def plot(summary: list[dict], destination: Path) -> list[str]:
    apply_manuscript_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
    styles = {
        "lasso": (ESTIMATE_COLOR, "^", "--", "weak LASSO"),
        "ols": (SECONDARY_COLOR, "o", "-", "weak OLS"),
    }
    for method in ("lasso", "ols"):
        rows = sorted(
            (row for row in summary if row["method"] == method),
            key=lambda row: row["observation_interval"],
        )
        x = np.asarray([row["observation_interval"] for row in rows])
        y = np.asarray([row["coefficient"]["rmse"] for row in rows])
        lo = np.asarray([row["coefficient"]["ci95"][0] for row in rows])
        hi = np.asarray([row["coefficient"]["ci95"][1] for row in rows])
        color, marker, linestyle, label = styles[method]
        ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"observation interval $\Delta t$")
    ax.set_ylabel("relative coefficient RMSE")
    ax.legend(frameon=False)
    finish_axes(ax)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = destination / f"qanr_stochastic_time_step_p50n.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(str(path))
    plt.close(fig)
    return paths


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(encoded)


def run(config_path: Path, out: Path, workers: int, resume: bool, smoke: bool) -> None:
    requested = read_json(config_path)
    base_path = ROOT / requested["base_protocol"]
    base = read_json(base_path)
    validate_spec(requested, base)
    spec = effective_spec(requested, smoke)
    out = out.resolve()
    if resume and (out / "effective_protocol.json").exists():
        if read_json(out / "effective_protocol.json") != spec:
            raise ValueError("resume protocol differs from saved protocol")
    out.mkdir(parents=True, exist_ok=resume)
    write_json(out / "requested_protocol.json", requested)
    write_json(out / "effective_protocol.json", spec)
    source_hashes = snapshot_sources(out, config_path, base_path)
    started = perf_counter()
    manifest = {
        "task": spec["task"], "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "workers": workers, "source_sha256": source_hashes,
        "existing_production_outputs_modified": False,
    }
    write_json(out / "manifest.json", manifest)
    job_dir = out / "jobs"
    job_dir.mkdir(exist_ok=True)
    results, pending = [], []
    for repetition in range(int(spec["repetitions"])):
        path = job_dir / f"repetition-{repetition:03d}.json"
        if resume and path.exists():
            results.append(read_json(path))
        else:
            pending.append(((repetition, spec, base), path))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_repetition, payload): path for payload, path in pending}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                write_json(futures[future], result)
                results.append(result)
                if index % 2 == 0 or index == len(pending):
                    print(f"repetitions: {index}/{len(pending)} new", flush=True)
    results.sort(key=lambda item: item["repetition"])
    records = [row for result in results for row in result["records"]]
    records.sort(key=lambda row: (row["value"], row["method"], row["repetition"]))
    summary = summarize(spec, records)
    slopes = slope_report(spec, records)
    verification = verify(spec, records, results)
    if verification["status"] != "passed":
        raise ValueError("; ".join(verification["issues"]))
    figures = plot(summary, out / "figures")
    write_json(out / "records.json", records)
    write_csv(out / "records.csv", records)
    write_json(out / "summary.json", summary)
    write_json(out / "slope_report.json", slopes)
    write_json(out / "verification.json", verification)
    write_json(out / "trajectory_audit.json", [{
        key: result[key] for key in (
            "repetition", "training_master_seed", "validation_master_seed",
            "training_trajectory_seeds", "validation_trajectory_seeds",
            "training_initial_state_sha256", "validation_initial_state_sha256",
        )
    } for result in results])
    elapsed = perf_counter() - started
    manifest.update({
        "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed, "records": len(records),
        "figures": figures,
        "records_sha256": digest(out / "records.json"),
        "summary_sha256": digest(out / "summary.json"),
        "slope_report_sha256": digest(out / "slope_report.json"),
        "verification_sha256": digest(out / "verification.json"),
    })
    write_json(out / "manifest.json", manifest)
    report = [
        "# Stochastic fixed-step Figure 2(d) production candidate", "",
        f"- status: {verification['status']}",
        f"- repetitions: {spec['repetitions']}",
        f"- simulation step: {spec['simulation_step']}",
        f"- process noise: {spec['process_noise']}",
        f"- observation noise: {spec['observation_noise']}",
        f"- P/N: {spec['pair_sampling']['ratio']}",
        f"- weak LASSO RMSE slope: {slopes['lasso']['coefficient_rmse_slope']:.6g}",
        f"- weak LASSO slope CI95: {slopes['lasso']['bootstrap_ci95']}",
        f"- weak OLS RMSE slope: {slopes['ols']['coefficient_rmse_slope']:.6g}",
        f"- elapsed seconds: {elapsed:.3f}", "",
    ]
    (out / "report.md").write_text("\n".join(report))
    print(json.dumps({
        "status": "completed", "output": str(out),
        "lasso_rmse_slope": slopes["lasso"]["coefficient_rmse_slope"],
        "lasso_ci95": slopes["lasso"]["bootstrap_ci95"],
        "ols_rmse_slope": slopes["ols"]["coefficient_rmse_slope"],
        "elapsed_seconds": elapsed,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.out, args.workers, args.resume, args.smoke)


if __name__ == "__main__":
    main()
