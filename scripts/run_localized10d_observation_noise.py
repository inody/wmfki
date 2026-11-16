"""Run a paired observation-noise sweep for the localized 10D benchmark.

The nominal observation noise (0.005) is fixed in the protocol before this
scientific run.  Within a repetition, all noise levels share latent trajectories,
standard-normal observation-error arrays, weak pair samples, and strong-form
target particles.  The three estimators are validation-selected raw weak LASSO,
raw weak OLS, and raw forward-difference OLS.
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
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import make_kernel_basis
from scripts.run_cross_example_nominal_comparison import (
    _ols,
    assemble_localized_strong,
    bootstrap_rmse,
)
from scripts.run_localized10d_pilot import validate_protocol as validate_base_protocol
from pipeline import fit_kernel_from_particles
from scripts.run_morse10d_sweeps import (
    _simulate_truth,
    assembly_config,
    evaluate_theta,
    latent_evaluation_system,
    method_records,
    repetition_plan,
)


DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "localized10d_observation_noise.json"
SOURCE_FILES = (
    "configs/experiments/localized10d_observation_noise.json",
    "configs/experiments/localized10d_observation_noise_003.json",
    "configs/experiments/localized10d_scientific.json",
    "scripts/run_localized10d_observation_noise.py",
    "scripts/verify_localized10d_observation_noise.py",
    "scripts/run_cross_example_nominal_comparison.py",
    "scripts/run_localized10d_pilot.py",
    "scripts/run_morse10d_sweeps.py",
    "scripts/run_1d_hat_lasso.py",
    "test/test_localized10d_observation_noise.py",
    "features.py", "kernels.py", "pipeline.py", "regression.py", "simulate.py",
    "tests.py", "time_tests.py", "utils.py",
)
METHODS = ("weak_lasso", "weak_ols", "strong_forward_difference_ols")
METHOD_LABELS = {
    "weak_lasso": "Weak LASSO",
    "weak_ols": "Weak OLS",
    "strong_forward_difference_ols": "Strong FD-OLS",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_protocol(sweep, base):
    task = sweep.get("task")
    if task not in {
        "localized10d_observation_noise_sweep",
        "localized10d_fixed_observation_noise_benchmark",
    } or sweep.get("protocol_version") != 1:
        raise ValueError("invalid localized-10D observation-noise protocol")
    validate_base_protocol(base)
    if digest(ROOT / sweep["base_config"]) != sweep["base_config_sha256"]:
        raise ValueError("base localized-10D protocol hash changed")
    clean_reference = sweep.get("clean_reference_records")
    clean_reference_sha256 = sweep.get("clean_reference_sha256")
    if clean_reference is None:
        if clean_reference_sha256 is not None:
            raise ValueError("clean localized-10D reference hash requires a source path")
    elif digest(ROOT / clean_reference) != clean_reference_sha256:
        raise ValueError("clean localized-10D reference records changed")
    if task == "localized10d_observation_noise_sweep":
        if sweep["observation_noise_values"] != [0.0, 0.001, 0.003, 0.005, 0.01, 0.03]:
            raise ValueError("observation-noise grid changed")
        if sweep["nominal_observation_noise"] != 0.005 or sweep["repetitions"] != 20:
            raise ValueError("predeclared nominal noise or repetition count changed")
        expected_selection = (
            "independent_validation_weak_residual_mse_separately_at_each_noise_level"
        )
    else:
        fixed_values = sweep["observation_noise_values"]
        if fixed_values not in ([0.01], [0.03]):
            raise ValueError("fixed observation-noise benchmark must use an approved level")
        if (sweep["nominal_observation_noise"] != fixed_values[0]
                or sweep["repetitions"] != 20):
            raise ValueError("fixed observation-noise benchmark changed")
        if sweep.get("spatial_test_construction") != (
            "data_dependent_from_each_observed_training_or_validation_array"
        ):
            raise ValueError("fixed benchmark must construct tests from observed positions")
        expected_selection = "independent_validation_weak_residual_mse"
    if sweep["weak_form"] != {
        "pairs": 10000,
        "lasso_selection": expected_selection,
        "truth_used_for_selection": False,
        "test_used_for_selection": False,
    }:
        raise ValueError("weak-form observation-noise protocol changed")
    strong = sweep["strong_form"]
    if strong["target_count"] != 64 or strong["target_seed_start"] != 14000000:
        raise ValueError("strong-form target protocol changed")
    if any(strong[key] for key in (
        "regularization", "intercept", "standardization", "coefficient_thresholding"
    )):
        raise ValueError("strong-form baseline must remain raw")
    if task == "localized10d_observation_noise_sweep":
        observation_step = base["model"]["dt"] * base["model"]["observation_stride"]
        balanced = base["model"]["sigma"] * np.sqrt(observation_step / 2.0)
        if not np.isclose(
            balanced, sweep["nominal_observation_noise"], rtol=0.0, atol=1e-15
        ):
            raise ValueError("nominal observation noise no longer balances increment scales")


def effective_protocol(sweep, base, smoke=False):
    sweep = json.loads(json.dumps(sweep))
    base = json.loads(json.dumps(base))
    if not smoke:
        return sweep, base
    if sweep["task"] == "localized10d_observation_noise_sweep":
        sweep["observation_noise_values"] = [0.0, sweep["nominal_observation_noise"]]
    sweep["repetitions"] = 1
    sweep["strong_form"]["target_count"] = 8
    sweep["weak_form"]["pairs"] = 60
    sweep["metric"]["bootstrap_draws"] = 100
    base["model"].update({"particles": 30, "steps": 20, "observation_stride": 1})
    base["weak_tests"].update({"spatial_count": 4, "temporal_count": 3})
    base["pair_sampling"]["baseline_pairs"] = 60
    base["evaluation"].update({
        "scientific_repetitions": 1, "profile_grid_size": 41,
        "latent_drift_time_count": 3, "latent_drift_target_count": 8,
    })
    base["estimator"].update({"alpha_count": 8, "max_iter": 20000})
    return sweep, base


def object_digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def observed_trajectory(latent, sigma_obs, standard_normal):
    latent = np.asarray(latent, dtype=float)
    standard_normal = np.asarray(standard_normal, dtype=float)
    if latent.shape != standard_normal.shape:
        raise ValueError("latent trajectory and observation-error array must match")
    return latent + float(sigma_obs) * standard_normal


def fixed_space_test_spec(spec, trajectory, seed):
    """Construct spatial tests once from the latent trajectory for paired reuse."""
    weak = spec["weak_tests"]
    trajectory = np.asarray(trajectory, dtype=float)
    dimension = trajectory.shape[-1]
    q_lo, q_hi = weak["spatial_quantiles"]
    lower, upper = np.quantile(trajectory.reshape(-1, dimension), [q_lo, q_hi], axis=0)
    count = int(weak["spatial_count"])
    rng = np.random.default_rng(seed)
    if count == 1:
        centers = ((lower + upper) / 2.0)[None, :]
    else:
        centers = rng.uniform(lower, upper, size=(count, dimension))
    box = np.maximum(upper - lower, 1e-12)
    delta = (float(np.prod(box)) / max(count, 1)) ** (1.0 / dimension)
    return {"centers": centers.tolist(), "s": float(weak["spatial_scale_factor"] * delta)}


def assemble_with_fixed_tests(spec, X, time, particles, sigma, trajectory_seed,
                              pair_seed, space_seed, pairs, fixed_tests):
    config = assembly_config(
        spec, particles, sigma, trajectory_seed, pair_seed, space_seed, pairs
    )
    config["space_tests"].update(fixed_tests)
    started = perf_counter()
    fit = fit_kernel_from_particles(
        config, X, time, return_system=True,
        pair_mode=spec["pair_sampling"]["mode"], pair_seed=pair_seed,
    )
    elapsed = perf_counter() - started
    return np.asarray(fit["system"]["A_flat"], dtype=float), np.asarray(
        fit["system"]["b_flat"], dtype=float
    ), {
        "assembly_seconds": elapsed,
        "response_seconds": fit["diagnostics"]["timing"]["response_seconds"],
        "design_seconds": fit["diagnostics"]["timing"]["design_seconds"],
    }


def _coefficient_error(theta, truth):
    if theta is None or not np.all(np.isfinite(theta)):
        return None
    return float(np.linalg.norm(np.asarray(theta) - truth) / np.linalg.norm(truth))


def run_repetition(payload):
    index, sweep, base = payload
    plan = repetition_plan(base)[index]
    particles = int(base["model"]["particles"])
    sigma = float(base["model"]["sigma"])
    observation_stride = int(base["model"]["observation_stride"])
    noise_seeds = sweep["observation_noise_seeds"]
    train_noise_seed = noise_seeds["training_start"] + noise_seeds["repetition_stride"] * index
    validation_noise_seed = noise_seeds["validation_start"] + noise_seeds["repetition_stride"] * index
    target_seed = sweep["strong_form"]["target_seed_start"] + index
    truth = np.asarray(base["model"]["interaction"]["coefficients"], dtype=float)
    started = perf_counter()
    records, selections = [], {}

    with threadpool_limits(limits=1):
        training, full_time = _simulate_truth(base, particles, sigma, plan["training_seed"])
        validation, _ = _simulate_truth(base, particles, sigma, plan["validation_seed"])
        test, _ = _simulate_truth(base, particles, sigma, plan["test_seed"])
        latent_training = training[:, ::observation_stride]
        latent_validation = validation[:, ::observation_stride]
        observed_time = full_time[::observation_stride]
        train_noise = np.random.default_rng(train_noise_seed).normal(size=latent_training.shape)
        validation_noise = np.random.default_rng(validation_noise_seed).normal(
            size=latent_validation.shape
        )
        target_rng = np.random.default_rng(target_seed)
        targets = np.sort(target_rng.choice(
            particles, size=int(sweep["strong_form"]["target_count"]), replace=False
        ))
        paired_latent_tests = sweep["task"] == "localized10d_observation_noise_sweep"
        training_tests = (
            fixed_space_test_spec(base, latent_training, plan["space_seed"])
            if paired_latent_tests else None
        )
        validation_tests = (
            fixed_space_test_spec(base, latent_validation, plan["space_seed"])
            if paired_latent_tests else None
        )
        kernel_basis, _ = make_kernel_basis(base["dictionary"])
        evaluation_system = latent_evaluation_system(base, test, plan["evaluation_seed"])

        for sigma_obs in sweep["observation_noise_values"]:
            observed_training = observed_trajectory(latent_training, sigma_obs, train_noise)
            observed_validation = observed_trajectory(
                latent_validation, sigma_obs, validation_noise
            )
            if not paired_latent_tests:
                training_tests = fixed_space_test_spec(
                    base, observed_training, plan["space_seed"]
                )
                validation_tests = fixed_space_test_spec(
                    base, observed_validation, plan["space_seed"]
                )
            train_A, train_b, train_timing = assemble_with_fixed_tests(
                base, observed_training, observed_time, particles, sigma,
                plan["training_seed"], plan["training_pair_seed"], plan["space_seed"],
                sweep["weak_form"]["pairs"], training_tests,
            )
            validation_A, validation_b, validation_timing = assemble_with_fixed_tests(
                base, observed_validation, observed_time, particles, sigma,
                plan["validation_seed"], plan["validation_pair_seed"], plan["space_seed"],
                sweep["weak_form"]["pairs"], validation_tests,
            )
            common = {
                "repetition": index, "observation_noise": float(sigma_obs),
                "particles": particles, "process_sigma": sigma,
                "observation_step": float(np.diff(observed_time)[0]),
                "training_seed": plan["training_seed"],
                "validation_seed": plan["validation_seed"], "test_seed": plan["test_seed"],
                "training_observation_noise_seed": train_noise_seed,
                "validation_observation_noise_seed": validation_noise_seed,
                "training_pair_seed": plan["training_pair_seed"],
                "validation_pair_seed": plan["validation_pair_seed"],
                "target_seed": target_seed,
                "training_space_test_sha256": object_digest(training_tests),
                "validation_space_test_sha256": object_digest(validation_tests),
                "pairs": int(sweep["weak_form"]["pairs"]),
                "training_assembly_seconds": train_timing["assembly_seconds"],
                "validation_assembly_seconds": validation_timing["assembly_seconds"],
            }
            weak_records, paths = method_records(
                train_A, train_b, validation_A, validation_b, evaluation_system,
                base["estimator"], common,
            )
            selections[f"sigma_obs={sigma_obs:g}"] = paths
            for row in weak_records:
                theta = row["coefficients"]
                records.append({
                    **row,
                    "method": "weak_lasso" if row["method"] == "lasso" else "weak_ols",
                    "coefficient_error_relative": _coefficient_error(theta, truth),
                    "status": "finite" if theta is not None and not row["numerical_divergence"]
                    else "failed",
                })

            strong_A, strong_b = assemble_localized_strong(
                observed_training, observed_time, targets, kernel_basis
            )
            strong_theta, strong_diagnostics = _ols(strong_A, strong_b)
            strong_metrics = evaluate_theta(strong_theta, *evaluation_system)
            records.append({
                **common, "method": "strong_forward_difference_ols",
                "coefficients": strong_theta.tolist(),
                "coefficient_error_relative": _coefficient_error(strong_theta, truth),
                "latent_drift_relative_error": strong_metrics["latent_drift_relative_error"],
                "profile_relative_error": strong_metrics["profile_relative_error"],
                "numerical_divergence": strong_metrics["numerical_divergence"],
                "target_count": int(targets.size), "system": strong_diagnostics,
                "status": "finite" if strong_diagnostics["finite"]
                          and not strong_metrics["numerical_divergence"] else "failed",
            })
    return {
        "index": index, "records": records, "selections": selections,
        "elapsed_seconds": perf_counter() - started,
    }


def _stats(values, draws, seed):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    finite = values[np.isfinite(values)]
    complete = finite.size == values.size and values.size > 0
    return {
        "count": int(values.size), "finite_count": int(finite.size),
        "complete": bool(complete),
        "mean": float(np.mean(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "q25": float(np.quantile(finite, 0.25)) if finite.size else None,
        "q75": float(np.quantile(finite, 0.75)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
        "rmse": float(np.sqrt(np.mean(finite ** 2))) if finite.size else None,
        "rmse_bootstrap_95": bootstrap_rmse(finite, draws, seed) if complete else None,
    }


def summarize(records, sweep):
    groups = []
    draws = int(sweep["metric"]["bootstrap_draws"])
    base_seed = int(sweep["metric"]["bootstrap_seed"])
    for noise_index, sigma_obs in enumerate(sweep["observation_noise_values"]):
        for method_index, method in enumerate(METHODS):
            rows = [row for row in records if row["method"] == method
                    and float(row["observation_noise"]) == float(sigma_obs)]
            group = {
                "observation_noise": float(sigma_obs), "method": method,
                "coefficient_error_relative": _stats(
                    [row["coefficient_error_relative"] for row in rows], draws,
                    base_seed + 100 * noise_index + method_index,
                ),
                "latent_drift_relative_error": _stats(
                    [row["latent_drift_relative_error"] for row in rows], draws,
                    base_seed + 1000 + 100 * noise_index + method_index,
                ),
                "profile_relative_error": _stats(
                    [row["profile_relative_error"] for row in rows], draws,
                    base_seed + 2000 + 100 * noise_index + method_index,
                ),
                "rank_deficient": int(sum(
                    row.get("rank_deficient", False)
                    or row.get("system", {}).get("rank_deficient", False) for row in rows
                )),
                "failed": int(sum(row["status"] != "finite" for row in rows)),
                "boundary_selections": int(sum(
                    bool(row.get("selected_at_grid_boundary", False)) for row in rows
                )),
                "nonconverged_path_candidates": int(sum(
                    int(row.get("unconverged_candidates", 0)) for row in rows
                )),
            }
            groups.append(group)
    nominal = [row for row in groups
               if row["observation_noise"] == sweep["nominal_observation_noise"]]
    winner = min(
        nominal, key=lambda row: row["coefficient_error_relative"]["rmse"]
    )["method"]
    return {
        "groups": groups, "record_count": len(records),
        "nominal_observation_noise": sweep["nominal_observation_noise"],
        "nominal_coefficient_rmse_winner": winner,
    }


def write_table(path, summary):
    groups = summary["groups"]
    lines = [
        "| $\\sigma_{\\rm obs}$ | Weak LASSO | Weak OLS | Strong FD-OLS |\n",
        "|---:|---:|---:|---:|\n",
    ]
    for sigma_obs in sorted({row["observation_noise"] for row in groups}):
        row_groups = {row["method"]: row for row in groups
                      if row["observation_noise"] == sigma_obs}
        values = {method: 100 * row_groups[method]["coefficient_error_relative"]["rmse"]
                  for method in METHODS}
        winner = min(values, key=values.get)
        cells = []
        for method in METHODS:
            value = values[method]
            text = f"{value:.3g}"
            cells.append(f"**{text}**" if method == winner else text)
        lines.append(f"| {sigma_obs:g} | " + " | ".join(cells) + " |\n")
    lines.append(
        "\nEntries are relative coefficient RMSE in percent over all predeclared repetitions. "
        "Bold marks the minimum at each observation-noise level.\n"
    )
    Path(path).write_text("".join(lines))


def source_snapshot(out):
    hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    return hashes


def run(config_path, out, *, workers=1, smoke=False, resume=False):
    requested = read_json(config_path)
    base = read_json(ROOT / requested["base_config"])
    validate_protocol(requested, base)
    sweep, effective_base = effective_protocol(requested, base, smoke=smoke)
    out = Path(out)
    if out.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "requested_protocol.json").exists():
        shutil.copy2(config_path, out / "requested_protocol.json")
        write_json(out / "effective_protocol.json", {
            "sweep": sweep, "base": effective_base, "smoke": bool(smoke)
        })
        source_hashes = source_snapshot(out)
        write_json(out / "run_state.json", {
            "source_sha256": source_hashes,
            "requested_protocol_sha256": digest(config_path),
        })
    else:
        state = read_json(out / "run_state.json")
        for relative, expected in state["source_sha256"].items():
            actual = digest(ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"source changed; refusing resume: {relative}")

    jobs_dir = out / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    repetition_count = int(sweep["repetitions"])
    completed = {}
    for index in range(repetition_count):
        path = jobs_dir / f"repetition-{index:03d}.json"
        if path.exists():
            completed[index] = read_json(path)
    missing = [index for index in range(repetition_count) if index not in completed]
    if missing:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = {
                executor.submit(run_repetition, (index, sweep, effective_base)): index
                for index in missing
            }
            for future in as_completed(futures):
                result = future.result()
                write_json(jobs_dir / f"repetition-{result['index']:03d}.json", result)
                completed[result["index"]] = result
                print(f"completed repetition {result['index']}", flush=True)

    jobs = [completed[index] for index in range(repetition_count)]
    records = [row for job in jobs for row in job["records"]]
    records.sort(key=lambda row: (
        row["observation_noise"], row["repetition"], METHODS.index(row["method"])
    ))
    summary = summarize(records, sweep)
    write_json(out / "records.json", records)
    write_json(out / "summary.json", summary)
    write_table(out / "observation_noise_table.md", summary)
    artifacts = {name: digest(out / name) for name in (
        "requested_protocol.json", "effective_protocol.json", "records.json",
        "summary.json", "observation_noise_table.md",
    )}
    manifest = {
        "task": requested["task"], "protocol_version": requested["protocol_version"],
        "status": "smoke_complete" if smoke else "scientific_complete",
        "smoke": bool(smoke), "created_utc": datetime.now(timezone.utc).isoformat(),
        "repetitions": repetition_count, "record_count": len(records),
        "nominal_observation_noise": requested["nominal_observation_noise"],
        "source_sha256": read_json(out / "run_state.json")["source_sha256"],
        "artifact_sha256": artifacts,
        "manuscript_modified": False,
    }
    write_json(out / "manifest.json", manifest)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        kind = "smoke" if args.smoke else "sweeps"
        args.out = ROOT / "outputs" / "validation" / "localized10d_observation_noise" / f"{kind}-{stamp}"
    out = run(args.config, args.out, workers=args.workers, smoke=args.smoke, resume=args.resume)
    print(out)


if __name__ == "__main__":
    main()
