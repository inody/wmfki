"""Extend the M=17 pair-count sweep from 20 existing repetitions with 80 new repetitions."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_1d_hat_experiment import array_hash, write_json
from scripts.run_1d_hat_sweeps import (
    assemble_sampled,
    candidate_list,
    evaluate_model,
    fixed_penalty_path,
    force_energy,
    load_data,
    load_statistics,
    read_json,
    test_family,
)

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat_M17_pair_R100.json"
SOURCES = [
    "features.py", "kernels.py", "pipeline.py", "simulate.py",
    "scripts/prepare_1d_hat_experiment.py", "scripts/run_1d_hat_lasso.py",
    "scripts/run_1d_hat_repetitions.py", "scripts/hat_sweep_numerics.py",
    "scripts/run_1d_hat_sweeps.py", "scripts/run_1d_hat_M17_R100.py",
    "scripts/run_1d_hat_M17_pair_R100.py", "scripts/summarize_1d_hat_M17_pair_R100.py",
    "scripts/verify_1d_hat_M17_pair_R100.py",
    "configs/experiments/one_dimensional_hat_sweeps.json",
    "configs/experiments/one_dimensional_hat_M17_R100.json",
    "configs/experiments/one_dimensional_hat_M17_pair_R100.json",
    "test/test_hat_M17_pair_R100.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sources(spec):
    pair_batch = ROOT / spec["source_pair_batch"]
    reference_batch = ROOT / spec["source_reference_batch"]
    return (
        pair_batch,
        read_json(pair_batch / "frozen_protocol.json"),
        read_json(pair_batch / "batch_status.json"),
        reference_batch,
        read_json(reference_batch / "frozen_protocol.json"),
        read_json(reference_batch / "batch_status.json"),
    )


def comparable_base(base):
    result = {key: value for key, value in base.items() if key not in ("seeds", "replication")}
    result["basis"] = {**result["basis"], "sizes": [17]}
    return result


def validate_protocol(spec, pair_frozen, pair_status, reference_frozen, reference_status):
    if spec["protocol_version"] != 1 or spec["dictionary_size"] != 17:
        raise ValueError("this protocol is fixed to M=17")
    if (spec["existing_repetitions"], spec["new_repetitions"], spec["total_repetitions"]) != (20, 80, 100):
        raise ValueError("the declared extension is 20 existing plus 80 new repetitions")
    if spec["pair_counts"] != [100, 300, 1000, 3000, 10000, 30000]:
        raise ValueError("pair-count grid changed")
    if spec["new_jobs"] != 480 or spec["existing_records"] != 120:
        raise ValueError("declared job counts changed")
    if pair_status["status"] != "completed" or reference_status["status"] != "completed":
        raise ValueError("both source batches must be complete")
    pair_path = ROOT / spec["source_pair_batch"] / "frozen_protocol.json"
    reference_path = ROOT / spec["source_reference_batch"] / "frozen_protocol.json"
    if digest(pair_path) != pair_status["frozen_protocol_sha256"]:
        raise ValueError("source pair protocol hash mismatch")
    if digest(reference_path) != reference_status["frozen_protocol_sha256"]:
        raise ValueError("source reference protocol hash mismatch")
    if pair_frozen["protocol"]["pair_counts"] != spec["pair_counts"]:
        raise ValueError("source pair grid mismatch")
    if pair_frozen["base"]["replication"]["count"] != 20 or 17 not in pair_frozen["base"]["basis"]["sizes"]:
        raise ValueError("unexpected existing pair source")
    requested = reference_frozen["requested_protocol"]
    if (requested["existing_repetitions"], requested["new_repetitions"], requested["total_repetitions"]) != (20, 80, 100):
        raise ValueError("unexpected reference repetition source")
    if reference_frozen["base"]["replication"]["count"] != 80 or reference_frozen["base"]["basis"]["sizes"] != [17]:
        raise ValueError("reference source must contain the 80 new M=17 baselines")
    if comparable_base(pair_frozen["base"]) != comparable_base(reference_frozen["base"]):
        raise ValueError("physical, basis, test, or solver settings changed between source batches")
    if len([key for key in reference_status["jobs"] if key.startswith("base-")]) != 80:
        raise ValueError("reference source is missing new baseline jobs")


def job_plan(spec, smoke=False):
    repetitions = 2 if smoke else spec["new_repetitions"]
    pair_counts = spec["pair_counts"][:2] if smoke else spec["pair_counts"]
    jobs = {}
    for pair_index, P in enumerate(pair_counts):
        for local_index in range(repetitions):
            global_index = spec["existing_repetitions"] + local_index
            key = f"pairs-{pair_index}-{local_index}"
            jobs[key] = {
                "id": key,
                "pair_index": pair_index,
                "P": P,
                "local_index": local_index,
                "global_index": global_index,
                "reference_key": f"base-{local_index}",
                "relative_directory": f"new/P-{P}/run-{global_index + 1:03d}",
            }
    return jobs


def reference_directory(reference_batch, reference_status, job):
    record = reference_status["results"][job["reference_key"]]
    directory = reference_batch / record["directory"]
    report = read_json(directory / "report.json")
    if record["status"] != "completed" or digest(directory / "report.json") != record["report_sha256"]:
        raise ValueError("reference baseline report hash/status mismatch")
    if report["job"]["case"] != "base" or report["job"]["index"] != job["local_index"]:
        raise ValueError("reference baseline identity mismatch")
    if set(report["results"]) != {"hat_M17"} or report["results"]["hat_M17"]["status"] != "evaluated":
        raise ValueError("reference M=17 model is missing")
    for name, expected in report["artifacts_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("reference baseline artifact changed")
    return directory, record, report


def import_existing(destination, spec, pair_batch, pair_status):
    imported = {}
    for pair_index, P in enumerate(spec["pair_counts"]):
        for index in range(spec["existing_repetitions"]):
            key = f"pairs-{pair_index}-{index}"
            record = pair_status["results"][key]
            report_path = pair_batch / record["directory"] / "report.json"
            if record["status"] != "completed" or digest(report_path) != record["report_sha256"]:
                raise ValueError("existing pair report hash/status mismatch")
            report = read_json(report_path)
            model = report["results"]["hat_M17"]
            if report["P"] != P or model["basis"]["M"] != 17:
                raise ValueError("existing pair identity mismatch")
            target = destination / "existing" / f"P-{P}" / f"run-{index + 1:03d}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            compact = {
                "status": "imported", "pair_index": pair_index, "P": P, "index": index,
                "result": model, "pair_seed": report["pair_seed"],
                "source_report": str(report_path.resolve()),
                "source_report_sha256": record["report_sha256"],
                "source_frozen_protocol_sha256": pair_status["frozen_protocol_sha256"],
            }
            write_json(target, compact)
            imported[key] = {"path": str(target.relative_to(destination)), "sha256": digest(target)}
    if len(imported) != spec["existing_records"]:
        raise ValueError("existing pair record count mismatch")
    return imported


def pair_job_external(directory, job, base, reference):
    selections = read_json(reference / "frozen_selections.json")
    train = load_data(reference / "train/data.npz")
    response = np.load(reference / "train/response.npy")
    space, time = test_family(train["t"])
    pair_seed = 2_000_000 + 1000 * job["global_index"] + 10 * job["pair_index"]
    matrices = assemble_sampled(train["X"], train["t"], space, time, candidate_list(base), job["P"], pair_seed)
    endpoints = {}
    for name, params in candidate_list(base):
        folder = directory / name
        folder.mkdir()
        np.save(folder / "design.npy", matrices[name])
        endpoint, coefficients = fixed_penalty_path(matrices[name], response, selections[name], base["estimator"])
        np.save(folder / "path_coefficients.npy", coefficients)
        endpoints[name] = {"basis": params, **endpoint}
    write_json(directory / "frozen_endpoints.json", endpoints)
    test = load_data(reference / "test/data.npz")
    statistics = load_statistics(reference / "evaluation_statistics.npz")
    results = {}
    for name, params in candidate_list(base):
        endpoint = endpoints[name]
        row = {"endpoint": endpoint["endpoint"], "path": endpoint["path"], "basis": params,
               "theta": endpoint["theta"], "metrics": {}}
        if endpoint["theta"] is not None:
            theta = np.asarray(endpoint["theta"])
            full_theta = np.asarray(selections[name]["theta"])
            difference = theta - full_theta
            full_A = np.load(reference / "train" / f"{name}.npy")
            metrics, rollout = evaluate_model(test, params, theta, base["simulation"]["sigma"], statistics[name])
            denominator = force_energy(full_theta, statistics[name])
            metrics.update(
                coefficient_difference_sq=float(difference @ difference),
                design_difference_sq=float(np.sum((matrices[name] - full_A) ** 2)),
                force_difference_sq_rel=(force_energy(difference, statistics[name]) / denominator
                                         if denominator > 0 else None),
            )
            row.update(status="finite_solver_output", metrics=metrics)
            if rollout is not None:
                np.savez_compressed(directory / name / "rollout.npz", X_hat=rollout)
                row["rollout_sha256"] = array_hash(rollout)
        else:
            row["status"] = "nonfinite_solver_output"
        results[name] = row
    return {
        "P": job["P"], "pair_seed": pair_seed, "results": results,
        "reference_directory": str(reference.resolve()),
        "reference_selections_sha256": digest(reference / "frozen_selections.json"),
        "reference_statistics_sha256": digest(reference / "evaluation_statistics.npz"),
        "frozen_endpoints_sha256": digest(directory / "frozen_endpoints.json"),
        "estimand": "finite-iteration fixed-alpha solver output; nonconverged endpoints are flagged, not discarded",
    }


def worker(batch, job, base, reference_batch, reference_status):
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
        reference, record, report = reference_directory(Path(reference_batch), reference_status, job)
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
            content = pair_job_external(directory, job, base, reference)
        content.update(reference_report_sha256=record["report_sha256"], reference_job=job["reference_key"])
        result.update(content, status="completed")
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
    pair_batch, pair_frozen, pair_status, reference_batch, reference_frozen, reference_status = load_sources(spec)
    validate_protocol(spec, pair_frozen, pair_status, reference_frozen, reference_status)
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
        provenance = destination / "source_provenance"
        for label, source in (("existing20", pair_batch), ("new80_reference", reference_batch)):
            target = provenance / label
            target.mkdir(parents=True)
            for filename in ("frozen_protocol.json", "batch_status.json"):
                shutil.copy2(source / filename, target / filename)
            if (source / "verification/verification.json").exists():
                target.joinpath("verification").mkdir()
                shutil.copy2(source / "verification/verification.json", target / "verification/verification.json")
        hashes = {name: digest(ROOT / name) for name in SOURCES}
        for name in SOURCES:
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        imported = {} if smoke else import_existing(destination, spec, pair_batch, pair_status)
        frozen = {
            "requested_protocol": spec,
            "protocol": {"pair_counts": spec["pair_counts"][:2] if smoke else spec["pair_counts"],
                         "new_repetitions": 2 if smoke else spec["new_repetitions"]},
            "base": reference_frozen["base"], "smoke_only": smoke,
            "source_sha256": hashes,
            "source_provenance_sha256": {str(path.relative_to(provenance)): digest(path)
                                           for path in provenance.rglob("*") if path.is_file()},
            "imported_existing": imported,
            "source_pair_frozen_protocol_sha256": pair_status["frozen_protocol_sha256"],
            "source_reference_frozen_protocol_sha256": reference_status["frozen_protocol_sha256"],
            "workers": workers, "python_version": sys.version,
            "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        }
        write_json(destination / "frozen_protocol.json", frozen)
        status = {"status": "running", "stage": "new_pairs", "smoke_only": smoke,
                  "frozen_protocol_sha256": digest(destination / "frozen_protocol.json"),
                  "jobs": job_plan(spec, smoke), "results": {}}
        write_json(destination / "batch_status.json", status)
    pending = {key: job for key, job in status["jobs"].items() if key not in status["results"]}
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(worker, destination, job, frozen["base"], reference_batch, reference_status): key
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
    sources = load_sources(spec)
    validate_protocol(spec, sources[1], sources[2], sources[4], sources[5])
    if not args.run and not args.smoke:
        print(json.dumps({"status": "configuration_valid", "new_jobs": len(job_plan(spec)),
                          "existing_records": spec["existing_records"], "dictionary_size": 17}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / (
        ("M17-pair-R100-smoke-" if args.smoke else "M17-pair-R100-") + stamp)
    run_batch(spec, destination, args.workers, args.smoke, args.resume)
    from scripts.summarize_1d_hat_M17_pair_R100 import summarize
    summary = destination / "summary"
    if summary.exists():
        summary = destination / f"summary-{stamp}"
    summarize(destination, summary)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
