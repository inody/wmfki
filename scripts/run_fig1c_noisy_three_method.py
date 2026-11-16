"""Run paired noisy Figure 1(c) comparisons for weak/strong LASSO and weak OLS."""

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

from features import compute_b_timeweak
from scripts.hat_sweep_numerics import assemble_sampled, errors_from_statistics
from scripts.run_1d_hat_lasso import select_lasso
from scripts.run_1d_hat_sampled_p50n_R100 import percentile_mean_ci
from scripts.run_1d_hat_sampled_pair_ratio_sweep_R5 import candidates
from scripts.run_1d_hat_sweeps import load_data, load_statistics, test_family
from scripts.run_fig1c_strong_fd_lasso import sufficient_statistics, fit_from_statistics

DEFAULT_CONFIG = ROOT / "configs/experiments/fig1c_noisy_three_method.json"
OUTPUT_ROOT = ROOT / "outputs/validation/fig1c_noisy_three_method"
SOURCE_FILES = (
    "configs/experiments/fig1c_noisy_three_method.json",
    "scripts/run_fig1c_noisy_three_method.py",
    "scripts/run_fig1c_strong_fd_lasso.py",
    "scripts/run_1d_hat_lasso.py",
    "scripts/hat_sweep_numerics.py",
    "features.py", "kernels.py",
)
METHODS = (
    "weak_lasso_validation_selected",
    "weak_ols",
    "strong_forward_difference_lasso_validation_selected",
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


def validate_protocol(spec):
    if (spec.get("task"), spec.get("protocol_version")) != ("fig1c_noisy_three_method", 1):
        raise ValueError("invalid noisy Figure 1(c) protocol")
    if spec.get("study_status") != "manuscript_current":
        raise ValueError("noisy Figure 1 comparison must use the current manuscript protocol")
    if float(spec.get("observation_noise")) != 0.01:
        raise ValueError("current Figure 1 protocol must use sigma_obs=0.01")
    if spec.get("dictionary_sizes") != [9, 11, 14, 17, 22, 27, 33, 42, 52, 65]:
        raise ValueError("dictionary-size grid changed")
    if int(spec.get("repetitions", 0)) != 100:
        raise ValueError("full protocol requires 100 repetitions")
    if (spec.get("pair_ratio"), spec.get("pair_count")) != (50, 50000):
        raise ValueError("weak pair protocol changed")
    if tuple(spec.get("methods", ())) != METHODS:
        raise ValueError("method list changed")
    estimator = spec["lasso_estimator"]
    if estimator["fit_intercept"] or estimator["standardize"]:
        raise ValueError("both LASSO estimators must use raw columns and no intercept")
    source = ROOT / spec["source_batch"]
    for relative, expected in spec["source_hashes"].items():
        if digest(source / relative) != expected:
            raise ValueError(f"frozen Figure 1 source changed: {relative}")


def pair_seeds(frozen, repetition):
    protocol = frozen["requested_protocol"]
    return (
        int(protocol["pair_seed_start"] + repetition * protocol["repetition_seed_stride"]
            + protocol["training_pair_seed_offset"]),
        int(protocol["pair_seed_start"] + repetition * protocol["repetition_seed_stride"]
            + protocol["validation_pair_seed_offset"]),
    )


def run_repetition(payload):
    repetition, spec, frozen = payload
    started = perf_counter()
    bundle = frozen["reference_bundles"][f"run-{repetition}"]
    reference = Path(bundle["reference_directory"])
    for path, expected in bundle["critical_artifacts_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"frozen repetition artifact changed: {path}")
    train = load_data(reference / "train/data.npz")
    validation = load_data(reference / "validation/data.npz")
    statistics = load_statistics(bundle["old_statistics"])
    statistics.update(load_statistics(bundle["new_statistics"]))
    sigma_obs = float(spec["observation_noise"])
    seeds = spec["observation_noise_seeds"]
    train_noise_seed = int(seeds["training_start"] + seeds["repetition_stride"] * repetition)
    validation_noise_seed = int(
        seeds["validation_start"] + seeds["repetition_stride"] * repetition
    )
    observed_train = train["X"] + sigma_obs * np.random.default_rng(
        train_noise_seed
    ).normal(size=train["X"].shape)
    observed_validation = validation["X"] + sigma_obs * np.random.default_rng(
        validation_noise_seed
    ).normal(size=validation["X"].shape)
    train_space, train_time = test_family(train["t"])
    validation_space, validation_time = test_family(validation["t"])
    train_pair_seed, validation_pair_seed = pair_seeds(frozen, repetition)
    base = json.loads(json.dumps(frozen["effective_base"]))
    base["estimator"].update(spec["lasso_estimator"])
    parameter_list = candidates(base)
    scale = np.sqrt(len(train_space["centers"]) * train_time["psi"].shape[0])
    train_b = compute_b_timeweak(
        observed_train, train["t"], base["simulation"]["sigma"], train_space, train_time
    ).reshape(-1) / scale
    validation_b = compute_b_timeweak(
        observed_validation, validation["t"], base["simulation"]["sigma"],
        validation_space, validation_time
    ).reshape(-1) / scale
    with threadpool_limits(limits=1):
        train_design = assemble_sampled(
            observed_train, train["t"], train_space, train_time, parameter_list,
            int(spec["pair_count"]), train_pair_seed,
        )
        validation_design = assemble_sampled(
            observed_validation, validation["t"], validation_space, validation_time,
            parameter_list, int(spec["pair_count"]), validation_pair_seed,
        )
        records = []
        for name, params in parameter_list:
            size = int(params["M"])
            A, Av = train_design[name], validation_design[name]
            selection, _ = select_lasso(A, train_b, Av, validation_b, spec["lasso_estimator"])
            if selection["status"] != "selected":
                raise RuntimeError(f"weak LASSO selection failed: {name}: {selection['status']}")
            weak_lasso = np.asarray(selection["theta"], dtype=float)
            weak_ols, _, weak_rank, weak_singular = np.linalg.lstsq(A, train_b, rcond=None)
            strong_training = sufficient_statistics(
                observed_train, train["t"], params["knots"]
            )
            strong_validation = sufficient_statistics(
                observed_validation, validation["t"], params["knots"]
            )
            strong_lasso, strong_selection = fit_from_statistics(
                strong_training, strong_validation, spec["lasso_estimator"]
            )
            for method, theta, details in (
                ("weak_lasso_validation_selected", weak_lasso, {
                    "selected_alpha": float(selection["selected"]["alpha_sk"]),
                    "selected_at_grid_boundary": bool(selection["selected_at_grid_boundary"]),
                    "selected_converged": bool(selection["selected"]["converged"]),
                    "unconverged_candidates": int(sum(
                        not row["converged"] for row in selection["path"]
                    )),
                }),
                ("weak_ols", weak_ols, {
                    "rank": int(weak_rank),
                    "columns": int(A.shape[1]),
                    "rank_deficient": bool(weak_rank < A.shape[1]),
                    "singular_values": weak_singular.tolist(),
                }),
                ("strong_forward_difference_lasso_validation_selected", strong_lasso,
                 strong_selection),
            ):
                metrics = errors_from_statistics(theta, statistics[name])
                records.append({
                    "repetition": repetition,
                    "M": size,
                    "method": method,
                    "observation_noise": sigma_obs,
                    "training_observation_noise_seed": train_noise_seed,
                    "validation_observation_noise_seed": validation_noise_seed,
                    "training_pair_seed": train_pair_seed,
                    "validation_pair_seed": validation_pair_seed,
                    "coefficients": np.asarray(theta, dtype=float).tolist(),
                    "drift_error_rel": float(metrics["drift_error_rel"]),
                    "force_error_weighted_rel": float(metrics["force_error_weighted_rel"]),
                    "diagnostics": details,
                    "status": "finite" if np.all(np.isfinite(theta)) else "failed",
                })
    return {"repetition": repetition, "records": records,
            "elapsed_seconds": perf_counter() - started}


def summarize(records, spec):
    groups = []
    for size_index, size in enumerate(spec["dictionary_sizes"]):
        for method_index, method in enumerate(METHODS):
            rows = sorted(
                [row for row in records if row["M"] == size and row["method"] == method],
                key=lambda row: row["repetition"],
            )
            values = np.asarray([row["drift_error_rel"] for row in rows], dtype=float)
            rng = np.random.default_rng(
                int(spec["bootstrap"]["seed"]) + 10 * size_index + method_index
            )
            indices = rng.integers(
                0, len(values), size=(int(spec["bootstrap"]["draws"]), len(values))
            )
            groups.append({
                "M": int(size), "method": method,
                "drift_error_rel": {
                    "values": values.tolist(),
                    "mean": float(np.mean(values)),
                    "rmse": float(np.sqrt(np.mean(values ** 2))),
                    "median": float(np.median(values)),
                    "mean_ci95": percentile_mean_ci(values, indices),
                },
                "force_error_weighted_rel": {
                    "mean": float(np.mean([row["force_error_weighted_rel"] for row in rows]))
                },
                "failures": int(sum(row["status"] != "finite" for row in rows)),
                "boundary_selections": int(sum(
                    bool(row["diagnostics"].get("selected_at_grid_boundary", False))
                    for row in rows
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
    validate_protocol(spec)
    frozen = read_json(ROOT / spec["source_batch"] / "frozen_protocol.json")
    effective = json.loads(json.dumps(spec))
    if smoke:
        effective["repetitions"] = 1
        effective["dictionary_sizes"] = [9, 65]
        effective["pair_count"] = 5000
        effective["lasso_estimator"]["alpha_count"] = 8
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
    jobs = [(index, effective, frozen) for index in range(effective["repetitions"])]
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
    records.sort(key=lambda row: (row["M"], METHODS.index(row["method"]), row["repetition"]))
    summary = summarize(records, effective)
    write_json(out / "records.json", records)
    write_json(out / "summary.json", summary)
    diagnostics = {
        "failures": sum(row["status"] != "finite" for row in records),
        "boundary_selections": sum(
            bool(row["diagnostics"].get("selected_at_grid_boundary", False))
            for row in records
        ),
        "weak_ols_rank_deficient": sum(
            row["method"] == "weak_ols"
            and row["diagnostics"].get("rank_deficient", False) for row in records
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
    manifest = run(args.config.resolve(), args.out, workers=args.workers, smoke=args.smoke)
    print(json.dumps({"output": str(args.out), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
