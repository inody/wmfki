"""Evaluate ten QANR hat-dictionary sizes by raw minimum-norm OLS over the same 100 repetitions."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.hat_sweep_numerics import evaluate_model, evaluation_statistics
from scripts.prepare_1d_hat_experiment import array_hash, basis_params, write_json
from scripts.run_1d_hat_dictionary_size_R100 import assemble_split, digest, read_json
from scripts.run_1d_hat_sweeps import save_statistics

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat_dictionary_grid_OLS_R100.json"
SOURCES = [
    "features.py", "kernels.py", "pipeline.py", "simulate.py",
    "scripts/prepare_1d_hat_experiment.py", "scripts/run_1d_hat_lasso.py",
    "scripts/hat_full_pair_assembly.py", "scripts/hat_sweep_numerics.py",
    "scripts/run_1d_hat_sweeps.py", "scripts/run_1d_hat_dictionary_size_R100.py",
    "scripts/run_1d_hat_dictionary_grid_R100.py",
    "scripts/run_1d_hat_dictionary_grid_OLS_R100.py",
    "scripts/summarize_1d_hat_dictionary_grid_OLS_R100.py",
    "scripts/verify_1d_hat_dictionary_grid_OLS_R100.py",
    "scripts/generate_1d_hat_practical_figure.py",
    "configs/experiments/one_dimensional_hat_dictionary_grid_R100.json",
    "configs/experiments/one_dimensional_hat_dictionary_grid_OLS_R100.json",
    "test/test_hat_dictionary_grid_OLS_R100.py",
]


def load_source(spec):
    batch = ROOT / spec["source_lasso_grid_batch"]
    return batch, read_json(batch / "frozen_protocol.json"), read_json(batch / "batch_status.json"), read_json(batch / "verification/verification.json")


def validate_protocol(spec, source_batch, source_frozen, source_status, source_verification):
    if spec["protocol_version"] != 1:
        raise ValueError("protocol version changed")
    if spec["dictionary_sizes"] != [9, 11, 14, 17, 22, 27, 33, 42, 52, 65]:
        raise ValueError("dictionary grid changed")
    if (spec["repetitions"], spec["jobs"], spec["models"]) != (100, 100, 1000):
        raise ValueError("planned counts changed")
    expected_estimator = {"name": "raw_minimum_norm_ols", "implementation": "numpy.linalg.lstsq",
                          "rcond": None, "column_standardization": False, "intercept": False,
                          "column_removal": False, "coefficient_thresholding": False, "post_fit_refit": False}
    if spec["estimator"] != expected_estimator:
        raise ValueError("OLS definition changed")
    if source_status["status"] != "completed" or source_verification["status"] != "passed":
        raise ValueError("LASSO grid source must be complete and verified")
    if digest(source_batch / "frozen_protocol.json") != source_status["frozen_protocol_sha256"]:
        raise ValueError("source frozen protocol changed")
    if digest(source_batch / "summary/summary.json") != source_verification["summary_sha256"]:
        raise ValueError("source summary changed")
    if source_frozen["requested_protocol"]["dictionary_sizes"] != spec["dictionary_sizes"]:
        raise ValueError("source dictionary grid differs")
    expected_references = 2 if source_frozen.get("smoke_only") else 100
    if len(source_frozen["references"]) != expected_references:
        raise ValueError("source trajectory references are incomplete")


def effective_base(source_frozen, spec, smoke=False):
    base = deepcopy(source_frozen["base"])
    base["basis"]["sizes"] = spec["dictionary_sizes"][:2] if smoke else spec["dictionary_sizes"]
    base["replication"]["count"] = 2 if smoke else spec["repetitions"]
    return base


def candidates(base):
    return [(f"hat_M{M}", basis_params(base, M)) for M in base["basis"]["sizes"]]


def job_plan(spec, smoke=False):
    count = 2 if smoke else spec["repetitions"]
    return {f"run-{index}": {"id": f"run-{index}", "global_index": index,
                              "relative_directory": f"run-{index + 1:03d}"}
            for index in range(count)}


def validate_reference(reference):
    directory = Path(reference["directory"])
    report_path = directory / "report.json"
    if digest(report_path) != reference["report_sha256"]:
        raise ValueError("trajectory reference report changed")
    report = read_json(report_path)
    for name, expected in report["artifacts_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("trajectory reference artifact changed")
    return directory


def fit_ols(A, b):
    started = perf_counter()
    theta, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    elapsed = perf_counter() - started
    finite = bool(np.all(np.isfinite(theta)) and np.all(np.isfinite(singular_values)))
    full_rank = int(rank) == A.shape[1]
    condition = (float(singular_values[0] / singular_values[-1])
                 if full_rank and len(singular_values) and singular_values[-1] > 0 else None)
    return theta, {
        "status": "fitted" if finite else "nonfinite",
        "solver": "numpy.linalg.lstsq", "rcond": None,
        "rank": int(rank), "columns": int(A.shape[1]), "full_column_rank": full_rank,
        "singular_values": singular_values.tolist(), "condition_number": condition,
        "reported_residual_sum_squares": residuals.tolist(),
        "train_mse": float(np.mean((A @ theta - b) ** 2)) if finite else None,
        "coefficient_l1": float(np.sum(np.abs(theta))) if finite else None,
        "coefficient_l2": float(np.linalg.norm(theta)) if finite else None,
        "nonzero_exact": int(np.count_nonzero(theta)) if finite else None,
        "nonzero_reported": int(np.count_nonzero(np.abs(theta) > 1e-8)) if finite else None,
        "fit_seconds": elapsed,
    }


def fit_job(directory, job, base, reference):
    train_A, train_b, _, train_manifest = assemble_split(directory / "train", reference, "train", base)
    fits = {}
    for name, params in candidates(base):
        theta, fit = fit_ols(train_A[name], train_b)
        fits[name] = {"basis": params, "theta": theta.tolist(), "fit": fit}
    # Freeze all ten train-only OLS vectors before validation and test arrays are read.
    write_json(directory / "frozen_ols_fits.json", fits)
    validation_A, validation_b, _, validation_manifest = assemble_split(
        directory / "validation", reference, "validation", base)
    for name in fits:
        theta = np.asarray(fits[name]["theta"], dtype=float)
        fits[name]["fit"]["validation_mse"] = (
            float(np.mean((validation_A[name] @ theta - validation_b) ** 2))
            if fits[name]["fit"]["status"] == "fitted" else None)
    test_A, test_b, test, test_manifest = assemble_split(directory / "test", reference, "test", base)
    statistics = evaluation_statistics(test["X"], test["t"], candidates(base), base["true_kernel"])
    save_statistics(directory / "evaluation_statistics.npz", statistics)
    results = {}
    for name, params in candidates(base):
        fit = fits[name]["fit"]
        theta = np.asarray(fits[name]["theta"], dtype=float)
        if fit["status"] != "fitted":
            results[name] = {"status": fit["status"], "basis": params, "theta": fits[name]["theta"],
                             "fit": fit, "metrics": {}}
            continue
        fit["test_mse"] = float(np.mean((test_A[name] @ theta - test_b) ** 2))
        metrics, rollout = evaluate_model(test, params, theta, base["simulation"]["sigma"], statistics[name])
        row = {"status": "evaluated", "basis": params, "theta": theta.tolist(), "fit": fit, "metrics": metrics}
        if rollout is not None:
            folder = directory / name
            folder.mkdir()
            np.savez_compressed(folder / "rollout.npz", X_hat=rollout)
            row["rollout_sha256"] = array_hash(rollout)
        results[name] = row
    return {
        "results": results,
        "splits": {"train": train_manifest, "validation": validation_manifest, "test": test_manifest},
        "frozen_ols_fits_sha256": digest(directory / "frozen_ols_fits.json"),
        "evaluation_statistics_sha256": digest(directory / "evaluation_statistics.npz"),
    }


def worker(batch, job, base, reference):
    batch = Path(batch)
    parent = batch / job["relative_directory"]
    directory, attempt = parent, 1
    while directory.exists():
        attempt += 1
        directory = parent.with_name(parent.name + f"-attempt-{attempt}")
    directory.mkdir(parents=True)
    started = perf_counter()
    result = {"job": job, "directory": str(directory.relative_to(batch)), "status": "running", "attempt": attempt}
    write_json(directory / "job_status.json", result)
    try:
        reference_directory = validate_reference(reference)
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
            content = fit_job(directory, job, base, reference_directory)
        result.update(content, status="completed", reference_directory=str(reference_directory),
                      reference_report_sha256=reference["report_sha256"])
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    result["elapsed_seconds"] = perf_counter() - started
    result["artifacts_sha256"] = {str(path.relative_to(directory)): digest(path)
                                  for path in directory.rglob("*") if path.is_file() and path.name != "job_status.json"}
    write_json(directory / "report.json", result)
    write_json(directory / "job_status.json", {key: result[key] for key in ("status", "directory", "elapsed_seconds", "attempt")})
    return {"status": result["status"], "directory": result["directory"],
            "report_sha256": digest(directory / "report.json"), "elapsed_seconds": result["elapsed_seconds"],
            "error": result.get("error")}


def run_batch(spec, destination, workers=6, smoke=False, resume=False):
    source_batch, source_frozen, source_status, source_verification = load_source(spec)
    validate_protocol(spec, source_batch, source_frozen, source_status, source_verification)
    destination = Path(destination).resolve()
    if resume:
        frozen = read_json(destination / "frozen_protocol.json")
        if frozen["requested_protocol"] != spec or frozen["smoke_only"] != smoke:
            raise ValueError("resume protocol mismatch")
        for name, expected in frozen["source_sha256"].items():
            if digest(ROOT / name) != expected or digest(destination / "source_snapshot" / name) != expected:
                raise ValueError("resume source mismatch")
        for name, expected in frozen["source_provenance_sha256"].items():
            if digest(destination / "source_provenance" / name) != expected:
                raise ValueError("source provenance changed")
        status = read_json(destination / "batch_status.json")
        if digest(destination / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
            raise ValueError("frozen protocol changed")
        for record in status["results"].values():
            if digest(destination / record["directory"] / "report.json") != record["report_sha256"]:
                raise ValueError("completed report changed")
    else:
        destination.mkdir(parents=True, exist_ok=False)
        provenance = destination / "source_provenance/lasso_grid_R100"
        provenance.mkdir(parents=True)
        for filename in ("frozen_protocol.json", "batch_status.json", "summary/summary.json", "verification/verification.json"):
            target = provenance / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_batch / filename, target)
        hashes = {name: digest(ROOT / name) for name in SOURCES}
        for name in SOURCES:
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        count = 2 if smoke else spec["repetitions"]
        references = {f"run-{i}": source_frozen["references"][f"run-{i}"] for i in range(count)}
        frozen = {
            "requested_protocol": spec, "base": effective_base(source_frozen, spec, smoke), "smoke_only": smoke,
            "source_sha256": hashes,
            "source_provenance_sha256": {str(path.relative_to(destination / "source_provenance")): digest(path)
                                           for path in (destination / "source_provenance").rglob("*") if path.is_file()},
            "references": references,
            "source_lasso_grid_frozen_protocol_sha256": source_status["frozen_protocol_sha256"],
            "source_lasso_grid_summary_sha256": source_verification["summary_sha256"],
            "workers": workers, "python_version": sys.version, "numpy_version": np.__version__,
        }
        write_json(destination / "frozen_protocol.json", frozen)
        status = {"status": "running", "stage": "ols_ten_dictionary_sizes", "smoke_only": smoke,
                  "frozen_protocol_sha256": digest(destination / "frozen_protocol.json"),
                  "jobs": job_plan(spec, smoke), "results": {}}
        write_json(destination / "batch_status.json", status)
    pending = {key: job for key, job in status["jobs"].items() if key not in status["results"]}
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(worker, destination, job, frozen["base"], frozen["references"][key]): key
                   for key, job in pending.items()}
        for future in as_completed(futures):
            key = futures[future]
            record = future.result()
            status["results"][key] = record
            write_json(destination / "batch_status.json", status)
            print(f"{len(status['results'])}/{len(status['jobs'])}: {key} {record['status']} {record['elapsed_seconds']:.1f}s", flush=True)
    status["status"] = ("completed" if all(record["status"] == "completed" for record in status["results"].values())
                        else "completed_with_failures")
    status["elapsed_seconds_this_invocation"] = perf_counter() - started
    write_json(destination / "batch_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or (args.resume and args.out_dir is None):
        parser.error("workers must be 1..8; resume requires --out-dir")
    spec = read_json(args.config)
    source = load_source(spec)
    validate_protocol(spec, *source)
    if not args.run and not args.smoke:
        print(json.dumps({"status": "configuration_valid", "dictionary_sizes": spec["dictionary_sizes"],
                          "jobs": spec["jobs"], "models": spec["models"], "estimator": spec["estimator"]}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / (
        ("dictionary-grid-OLS-R100-smoke-" if args.smoke else "dictionary-grid-OLS-R100-") + stamp)
    run_batch(spec, destination, args.workers, args.smoke, args.resume)
    from scripts.summarize_1d_hat_dictionary_grid_OLS_R100 import summarize
    summarize(destination, destination / "summary")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
