"""Run the localized 10D robustness study at fixed observation noise.

The representative fit and every process-noise, particle-number, and pair-count
condition use independent Gaussian training and validation position errors at
the predeclared fixed level. LASSO is selected only by an independent
validation weak residual; latent test quantities are used only after fitting.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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

from scripts.run_localized10d_observation_noise import (
    assemble_with_fixed_tests,
    fixed_space_test_spec,
    observed_trajectory,
)
from scripts.run_localized10d_pilot import (
    _augment_record,
    _plot_profile_and_dynamics,
    _plot_sweeps,
    effective_protocol as effective_base_protocol,
    summarize,
    validate_protocol as validate_base_protocol,
    write_csv,
)
from scripts.run_morse10d_sweeps import (
    _trajectory_path,
    design_diagnostics,
    latent_evaluation_system,
    load_or_simulate,
    method_records,
    physical_conditions,
    read_json,
    repetition_plan,
    select_raw_lasso,
    write_json,
)

DEFAULT_CONFIG = ROOT / "configs/experiments/localized10d_noisy_robustness.json"
SOURCE_FILES = (
    "configs/experiments/localized10d_noisy_robustness.json",
    "configs/experiments/localized10d_scientific.json",
    "configs/experiments/localized10d_observation_noise_003.json",
    "configs/experiments/localized10d_observation_noise_001.json",
    "scripts/run_localized10d_noisy_robustness.py",
    "scripts/verify_localized10d_noisy_robustness.py",
    "scripts/run_localized10d_observation_noise.py",
    "scripts/run_localized10d_pilot.py",
    "scripts/run_morse10d_sweeps.py",
    "scripts/run_1d_hat_lasso.py",
    "test/test_localized10d_noisy_robustness.py",
    "features.py", "kernels.py", "pipeline.py", "regression.py", "simulate.py",
    "tests.py", "time_tests.py", "utils.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(requested, base, nominal):
    if requested.get("task") != "localized10d_fixed_noise_robustness" \
            or requested.get("protocol_version") != 1:
        raise ValueError("invalid fixed-noise localized 10D protocol")
    validate_base_protocol(base)
    if base.get("task") != "localized10d_matched_scientific":
        raise ValueError("fixed-noise robustness requires the scientific base protocol")
    for key in ("base_config", "nominal_config", "nominal_reference_records"):
        if not (ROOT / requested[key]).is_file():
            raise FileNotFoundError(ROOT / requested[key])
    if digest(ROOT / requested["base_config"]) != requested["base_config_sha256"]:
        raise ValueError("base protocol hash changed")
    if digest(ROOT / requested["nominal_config"]) != requested["nominal_config_sha256"]:
        raise ValueError("nominal protocol hash changed")
    if digest(ROOT / requested["nominal_reference_records"]) \
            != requested["nominal_reference_sha256"]:
        raise ValueError("nominal reference records changed")
    observation_noise = float(requested.get("observation_noise", -1.0))
    if observation_noise not in {0.01, 0.03}:
        raise ValueError("observation noise must use an approved fixed level")
    if (nominal.get("observation_noise_values") != [observation_noise]
            or float(nominal.get("nominal_observation_noise", -1.0)) != observation_noise):
        raise ValueError("nominal protocol and robustness observation noise differ")
    if int(requested.get("repetitions", -1)) != 20:
        raise ValueError("scientific repetition count must remain 20")
    if requested.get("observation_noise_seeds") != nominal.get("observation_noise_seeds"):
        raise ValueError("observation-noise seeds must match the nominal comparison")
    if requested.get("sweeps") != base.get("sweeps"):
        raise ValueError("robustness grids must match the frozen clean study")
    if requested.get("pair_penalty_policy") != base.get("pair_penalty_policy"):
        raise ValueError("pair-count penalty policy changed")
    if requested.get("spatial_test_construction") \
            != nominal.get("spatial_test_construction"):
        raise ValueError("spatial-test construction must match the nominal comparison")


def effective_protocol(requested, base, smoke=False):
    spec = effective_base_protocol(base, smoke=smoke)
    spec["task"] = requested["task"]
    spec["pilot_only"] = False
    spec["observation_noise"] = float(requested["observation_noise"])
    spec["observation_noise_seeds"] = requested["observation_noise_seeds"]
    spec["spatial_test_construction"] = requested["spatial_test_construction"]
    spec["pairing_policy"] = requested["pairing_policy"]
    spec["failure_policy"] = requested["failure_policy"]
    spec["baseline_parity_policy"] = requested["baseline_parity_policy"]
    spec["evaluation"]["rollout_particles"] = int(spec["model"]["particles"])
    if smoke:
        spec["evaluation"]["scientific_repetitions"] = 1
    return spec


def _noise_seed(spec, plan, split):
    seeds = spec["observation_noise_seeds"]
    return int(seeds[f"{split}_start"] + seeds["repetition_stride"] * plan["index"])


def _add_observation_noise(spec, latent, particles, plan, split):
    if split not in {"training", "validation"}:
        return np.asarray(latent, dtype=float), None
    seed = _noise_seed(spec, plan, split)
    latent = np.asarray(latent, dtype=float)
    max_particles = int(spec["model"]["particles"])
    full_shape = (latent.shape[0], latent.shape[1], max_particles, latent.shape[3])
    standard_normal = np.random.default_rng(seed).normal(size=full_shape)
    observed = observed_trajectory(
        latent, spec["observation_noise"], standard_normal[:, :, :particles, :]
    )
    return observed, seed


def _tag_standard_records(records, spec, particles, sigma):
    tagged = []
    base_n = int(spec["model"]["particles"])
    base_sigma = float(spec["model"]["sigma"])
    if particles == base_n and sigma in spec["sweeps"]["process_sigma"]:
        tagged.extend({**row, "sweep": "process_sigma", "sweep_value": sigma}
                      for row in records)
    if sigma == base_sigma and particles in spec["sweeps"]["particle_number"]:
        tagged.extend({**row, "sweep": "particle_number", "sweep_value": particles}
                      for row in records)
    return tagged


def _assemble(spec, trajectory, time, particles, sigma, plan, split, pairs, tests):
    return assemble_with_fixed_tests(
        spec, trajectory, time, particles, sigma,
        plan[f"{split}_seed"], plan[f"{split}_pair_seed"], plan["space_seed"],
        pairs, tests,
    )


def run_repetition(payload):
    out_string, spec, plan = payload
    out = Path(out_string)
    records, selections, trajectory_status = [], {}, []
    base_n = int(spec["model"]["particles"])
    base_sigma = float(spec["model"]["sigma"])
    target = np.asarray(spec["model"]["interaction"]["coefficients"], dtype=float)
    baseline_data = None

    with threadpool_limits(limits=1):
        for particles, sigma in physical_conditions(spec):
            split_data = {}
            simulation_started = perf_counter()
            for split in ("training", "validation", "test"):
                X, time, cached = load_or_simulate(out, spec, particles, sigma, split, plan)
                split_data[split] = (X, time)
                trajectory_status.append({
                    "particles": particles, "sigma": sigma, "split": split,
                    "cache_hit": cached,
                })
            simulation_seconds = perf_counter() - simulation_started
            stride = int(spec["model"]["observation_stride"])
            sampled = {
                split: (data[0][:, ::stride], data[1][::stride])
                for split, data in split_data.items()
            }
            observed, noise_seeds = {}, {}
            for split, (X, time) in sampled.items():
                observed_X, noise_seed = _add_observation_noise(
                    spec, X, particles, plan, split
                )
                observed[split] = (observed_X, time)
                noise_seeds[split] = noise_seed
            training_tests = fixed_space_test_spec(
                spec, observed["training"][0], plan["space_seed"]
            )
            validation_tests = fixed_space_test_spec(
                spec, observed["validation"][0], plan["space_seed"]
            )
            baseline_pairs = int(spec["pair_sampling"]["baseline_pairs"])
            train_A, train_b, train_timing = _assemble(
                spec, *observed["training"], particles, sigma, plan,
                "training", baseline_pairs, training_tests,
            )
            validation_A, validation_b, validation_timing = _assemble(
                spec, *observed["validation"], particles, sigma, plan,
                "validation", baseline_pairs, validation_tests,
            )
            evaluation_system = latent_evaluation_system(
                spec, split_data["test"][0], plan["evaluation_seed"]
            )
            fit_id = f"rep{plan['index']:03d}-N{particles}-sigma{sigma:g}-Pbaseline"
            common = {
                "repetition": plan["index"], "fit_id": fit_id,
                "particles": particles, "sigma": sigma,
                "observation_noise": float(spec["observation_noise"]),
                "pairs": baseline_pairs,
                "training_seed": plan["training_seed"],
                "validation_seed": plan["validation_seed"],
                "test_seed": plan["test_seed"],
                "training_observation_noise_seed": noise_seeds["training"],
                "validation_observation_noise_seed": noise_seeds["validation"],
                "training_pair_seed": plan["training_pair_seed"],
                "validation_pair_seed": plan["validation_pair_seed"],
                "simulation_and_cache_seconds": simulation_seconds,
                "training_assembly_seconds": train_timing["assembly_seconds"],
                "validation_assembly_seconds": validation_timing["assembly_seconds"],
            }
            fitted, paths = method_records(
                train_A, train_b, validation_A, validation_b, evaluation_system,
                spec["estimator"], common,
            )
            fitted = [_augment_record(row, target) for row in fitted]
            records.extend(_tag_standard_records(fitted, spec, particles, sigma))
            selections[fit_id] = paths
            if particles == base_n and sigma == base_sigma:
                baseline_data = (
                    observed, evaluation_system, training_tests, validation_tests
                )

        if baseline_data is None:
            raise RuntimeError("baseline observed trajectory was not generated")
        observed, evaluation_system, training_tests, validation_tests = baseline_data
        pmax = max(spec["sweeps"]["sampled_pairs"])
        pmax_A, pmax_b, pmax_train_timing = _assemble(
            spec, *observed["training"], base_n, base_sigma, plan,
            "training", pmax, training_tests,
        )
        pmax_Av, pmax_bv, pmax_validation_timing = _assemble(
            spec, *observed["validation"], base_n, base_sigma, plan,
            "validation", pmax, validation_tests,
        )
        selected_theta, p_selection, selected_info = select_raw_lasso(
            pmax_A, pmax_b, pmax_Av, pmax_bv, spec["estimator"]
        )
        selections[f"rep{plan['index']:03d}-Pmax-selection"] = {"lasso": p_selection}
        alpha = selected_info["alpha"]
        for pairs in spec["sweeps"]["sampled_pairs"]:
            if pairs == pmax:
                A, b, train_timing = pmax_A, pmax_b, pmax_train_timing
            else:
                A, b, train_timing = _assemble(
                    spec, *observed["training"], base_n, base_sigma, plan,
                    "training", pairs, training_tests,
                )
            common = {
                "repetition": plan["index"],
                "fit_id": f"rep{plan['index']:03d}-P{pairs}",
                "sweep": "sampled_pairs", "sweep_value": pairs,
                "particles": base_n, "sigma": base_sigma,
                "observation_noise": float(spec["observation_noise"]),
                "pairs": pairs,
                "training_seed": plan["training_seed"],
                "validation_seed": plan["validation_seed"],
                "test_seed": plan["test_seed"],
                "training_observation_noise_seed": _noise_seed(spec, plan, "training"),
                "validation_observation_noise_seed": _noise_seed(spec, plan, "validation"),
                "training_pair_seed": plan["training_pair_seed"],
                "validation_pair_seed": plan["validation_pair_seed"],
                "simulation_and_cache_seconds": 0.0,
                "training_assembly_seconds": train_timing["assembly_seconds"],
                "validation_assembly_seconds": pmax_validation_timing["assembly_seconds"],
                "pmax_selected_alpha": alpha,
                "pair_design_relative_error": float(
                    np.linalg.norm(A - pmax_A) / max(np.linalg.norm(pmax_A), 1e-300)
                ),
            }
            if alpha is None:
                failed = [{
                    **common, "method": "lasso", "coefficients": None,
                    "nonzero_reported": None, **design_diagnostics(A),
                    "converged": False, "unconverged_candidates": 1,
                    "selected_at_grid_boundary": False, "alpha": None,
                    "alpha_index": None, "alpha_over_max": None, "iterations": None,
                    "fit_seconds": 0.0, "validation_weak_residual_mse": None,
                    "latent_drift_relative_error": None,
                    "profile_relative_error": None, "numerical_divergence": True,
                    "ols_lstsq_rank": None, "ols_smallest_singular_value": None,
                }]
                both, _ = method_records(
                    A, b, pmax_Av, pmax_bv, evaluation_system,
                    spec["estimator"], common, fixed_alpha=1.0,
                )
                fitted = failed + [row for row in both if row["method"] == "ols"]
            else:
                fitted, _ = method_records(
                    A, b, pmax_Av, pmax_bv, evaluation_system,
                    spec["estimator"], common, fixed_alpha=alpha,
                    initial_theta=selected_theta,
                )
            records.extend(_augment_record(row, target) for row in fitted)

        for particles, sigma in physical_conditions(spec):
            for split in ("training", "validation", "test"):
                _trajectory_path(out, particles, sigma, split, plan["index"]).unlink(
                    missing_ok=True
                )
    return {
        "index": plan["index"], "records": records,
        "selections": selections, "trajectory_cache": trajectory_status,
    }


def baseline_parity(records, requested):
    reference = read_json(ROOT / requested["nominal_reference_records"])
    reference_methods = {"lasso": "weak_lasso", "ols": "weak_ols"}
    comparisons = []
    for method, reference_method in reference_methods.items():
        new_rows = sorted([
            row for row in records
            if row["sweep"] == "process_sigma"
            and float(row["sweep_value"]) == 0.05 and row["method"] == method
        ], key=lambda row: row["repetition"])
        old_rows = sorted([
            row for row in reference if row["method"] == reference_method
            and float(row["observation_noise"]) == float(requested["observation_noise"])
        ], key=lambda row: row["repetition"])
        if len(new_rows) != 20 or len(old_rows) != 20:
            raise ValueError("nominal parity comparison is incomplete")
        for new, old in zip(new_rows, old_rows):
            checks = {}
            for key in (
                "coefficients", "coefficient_error_relative",
                "latent_drift_relative_error", "profile_relative_error",
            ):
                checks[key] = bool(np.allclose(
                    np.asarray(new[key], dtype=float), np.asarray(old[key], dtype=float),
                    rtol=1e-12, atol=1e-13,
                ))
            if method == "lasso":
                checks["alpha"] = bool(np.isclose(
                    float(new["alpha"]), float(old["alpha"]), rtol=1e-12, atol=1e-15
                ))
            comparisons.append({
                "repetition": new["repetition"], "method": method,
                "checks": checks, "passed": all(checks.values()),
            })
    return {
        "count": len(comparisons),
        "passed_count": sum(item["passed"] for item in comparisons),
        "passed": all(item["passed"] for item in comparisons),
        "comparisons": comparisons,
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


def run(config_path, out, *, workers=1, smoke=False, resume=False):
    requested = read_json(config_path)
    base = read_json(ROOT / requested["base_config"])
    nominal = read_json(ROOT / requested["nominal_config"])
    validate_protocol(requested, base, nominal)
    spec = effective_protocol(requested, base, smoke=smoke)
    out = Path(out)
    if out.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {out}")
    if not out.exists():
        out.mkdir(parents=True)
        write_json(out / "requested_protocol.json", requested)
        write_json(out / "frozen_protocol.json", spec)
        hashes = snapshot_sources(out)
        write_json(out / "manifest.json", {
            "task": spec["task"], "status": "running", "smoke": bool(smoke),
            "source_sha256": hashes, "jobs": [],
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
    else:
        manifest = read_json(out / "manifest.json")
        current = {relative: digest(ROOT / relative) for relative in SOURCE_FILES}
        if manifest["source_sha256"] != current:
            raise ValueError("resume source hash mismatch")
        if read_json(out / "frozen_protocol.json") != spec:
            raise ValueError("resume protocol mismatch")

    plans = repetition_plan(spec)
    jobs_dir = out / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    completed = {
        int(path.stem.split("-")[-1]) for path in jobs_dir.glob("repetition-*.json")
    }
    pending = [plan for plan in plans if plan["index"] not in completed]
    started = perf_counter()
    payloads = [(str(out), spec, plan) for plan in pending]
    if workers == 1:
        generated, executor = map(run_repetition, payloads), None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        generated = executor.map(run_repetition, payloads)
    try:
        for result in generated:
            write_json(jobs_dir / f"repetition-{result['index']:03d}.json", result)
            print(f"fixed-noise repetition {result['index'] + 1}/{len(plans)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    jobs = [read_json(path) for path in sorted(jobs_dir.glob("repetition-*.json"))]
    records = [row for job in jobs for row in job["records"]]
    summary = summarize(records)
    parity = None if smoke else baseline_parity(records, requested)
    write_json(out / "records.json", records)
    write_csv(out / "records.csv", records)
    write_json(out / "summary.json", summary)
    write_json(out / "baseline_parity.json", parity)
    _plot_sweeps(summary, out / "localized10d_noisy_sweeps.pdf")
    dynamics = _plot_profile_and_dynamics(spec, records, out)
    write_json(out / "dynamics_summary.json", dynamics)
    diagnostics = {
        "record_count": len(records),
        "rank_deficient": sum(bool(row["rank_deficient"]) for row in records),
        "nonconverged_records": sum(not bool(row["converged"]) for row in records),
        "boundary_selections": sum(bool(row["selected_at_grid_boundary"]) for row in records),
        "numerical_divergence": sum(bool(row["numerical_divergence"]) for row in records),
        "missing_coefficients": sum(row["coefficients"] is None for row in records),
        "baseline_parity_passed": None if smoke else parity["passed"],
    }
    write_json(out / "quality_report.json", diagnostics)
    artifacts = {
        name: digest(out / name) for name in (
            "requested_protocol.json", "frozen_protocol.json", "source_manifest.json",
            "records.json", "records.csv", "summary.json", "baseline_parity.json",
            "quality_report.json", "localized10d_noisy_sweeps.pdf",
            "localized10d_profile_coefficients.pdf",
            "localized10d_coupled_dynamics.pdf", "dynamics_summary.json",
        )
    }
    manifest = read_json(out / "manifest.json")
    manifest.update({
        "status": "smoke_completed" if smoke else "scientific_completed",
        "jobs": [job["index"] for job in jobs], "record_count": len(records),
        "observation_noise": float(spec["observation_noise"]),
        "baseline_parity_passed": None if smoke else parity["passed"],
        "artifact_sha256": artifacts,
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
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.resume and args.out_dir is None:
        parser.error("--resume requires --out-dir")
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = "smoke" if args.smoke else "sweeps"
        args.out_dir = ROOT / "outputs/validation/localized10d_noisy_robustness" / f"{label}-{stamp}"
    result = run(
        args.config.resolve(), args.out_dir, workers=args.workers,
        smoke=args.smoke, resume=args.resume,
    )
    print(json.dumps({"status": result["status"], "output": str(args.out_dir)}))


if __name__ == "__main__":
    main()
