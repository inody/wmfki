"""Compare ten QANR hat-dictionary sizes over the same 100 repetitions."""

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
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_1d_hat_experiment import write_json
from scripts.run_1d_hat_dictionary_size_R100 import digest, fit_job, read_json

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat_dictionary_grid_R100.json"
SOURCES = [
    "features.py", "kernels.py", "pipeline.py", "simulate.py",
    "scripts/prepare_1d_hat_experiment.py", "scripts/run_1d_hat_lasso.py",
    "scripts/hat_full_pair_assembly.py", "scripts/hat_sweep_numerics.py",
    "scripts/run_1d_hat_sweeps.py", "scripts/run_1d_hat_M17_R100.py",
    "scripts/run_1d_hat_dictionary_size_R100.py",
    "scripts/run_1d_hat_dictionary_grid_R100.py",
    "scripts/summarize_1d_hat_dictionary_grid_R100.py",
    "scripts/verify_1d_hat_dictionary_grid_R100.py",
    "scripts/generate_1d_hat_practical_figure.py",
    "configs/experiments/one_dimensional_hat_sweeps.json",
    "configs/experiments/one_dimensional_hat_M17_R100.json",
    "configs/experiments/one_dimensional_hat_dictionary_size_R100.json",
    "configs/experiments/one_dimensional_hat_dictionary_grid_R100.json",
    "test/test_hat_dictionary_grid_R100.py",
]


def load_source(spec):
    batch = ROOT / spec["source_four_size_batch"]
    return batch, read_json(batch / "frozen_protocol.json"), read_json(batch / "batch_status.json"), read_json(batch / "verification/verification.json")


def validate_protocol(spec, source_batch, source_frozen, source_status, source_verification):
    if spec["protocol_version"] != 1:
        raise ValueError("protocol version changed")
    if spec["dictionary_sizes"] != [9, 11, 14, 17, 22, 27, 33, 42, 52, 65]:
        raise ValueError("ten-point dictionary grid changed")
    if spec["reused_sizes"] != [9, 17, 33, 65] or spec["newly_fitted_sizes"] != [11, 14, 22, 27, 42, 52]:
        raise ValueError("reused/new dictionary split changed")
    if (spec["repetitions"], spec["new_jobs"], spec["new_models"], spec["reused_models"]) != (100, 100, 600, 400):
        raise ValueError("planned counts changed")
    if source_status["status"] != "completed" or source_verification["status"] != "passed":
        raise ValueError("four-size source must be completed and verified")
    if digest(source_batch / "frozen_protocol.json") != source_status["frozen_protocol_sha256"]:
        raise ValueError("four-size frozen protocol changed")
    if digest(source_batch / "summary/summary.json") != source_verification["summary_sha256"]:
        raise ValueError("four-size summary changed")
    source_spec = source_frozen["requested_protocol"]
    if source_spec["dictionary_sizes"] != spec["reused_sizes"] or source_spec["total_repetitions"] != 100:
        raise ValueError("unexpected source dictionary sizes or repetition count")


def effective_base(source_frozen, spec, smoke=False):
    base = deepcopy(source_frozen["base"])
    base["basis"]["sizes"] = spec["newly_fitted_sizes"][:1] if smoke else spec["newly_fitted_sizes"]
    base["replication"]["count"] = 2 if smoke else spec["repetitions"]
    if smoke:
        base["estimator"] = {**base["estimator"], "alpha_count": 5, "max_iter": 2000}
    return base


def collect_source_records(source_batch, source_frozen, source_status, count=100):
    records = {}
    for global_index in range(count):
        if global_index < 20:
            compact_record = source_frozen["imported_existing"][f"base-{global_index}"]
            compact_path = source_batch / compact_record["path"]
            if digest(compact_path) != compact_record["sha256"]:
                raise ValueError("existing source compact record changed")
            compact = read_json(compact_path)
            report_path = Path(compact["source_report"])
            report_hash = compact["source_report_sha256"]
            results = compact["results"]
        else:
            local = global_index - 20
            new_record = source_status["results"][f"base-{local}"]
            new_report_path = source_batch / new_record["directory"] / "report.json"
            if digest(new_report_path) != new_record["report_sha256"]:
                raise ValueError("new three-size source report changed")
            new_report = read_json(new_report_path)
            m17_record = source_frozen["imported_new_M17"][f"base-{local}"]
            m17_path = source_batch / m17_record["path"]
            if digest(m17_path) != m17_record["sha256"]:
                raise ValueError("new M17 compact record changed")
            m17 = read_json(m17_path)
            results = dict(new_report["results"])
            results["hat_M17"] = m17["result"]
            report_path = Path(new_report["reference_directory"]) / "report.json"
            report_hash = new_report["reference_report_sha256"]
        if digest(report_path) != report_hash:
            raise ValueError("trajectory reference report changed")
        report = read_json(report_path)
        for name, expected in report["artifacts_sha256"].items():
            if digest(report_path.parent / name) != expected:
                raise ValueError("trajectory reference artifact changed")
        if set(results) != {"hat_M9", "hat_M17", "hat_M33", "hat_M65"}:
            raise ValueError("reused model missing")
        records[global_index] = {
            "results": results,
            "reference_directory": str(report_path.parent.resolve()),
            "reference_report_sha256": report_hash,
            "source_model_records": ([{"path": str(compact_path.resolve()), "sha256": digest(compact_path)}]
                                     if global_index < 20 else
                                     [{"path": str(new_report_path.resolve()), "sha256": digest(new_report_path)},
                                      {"path": str(m17_path.resolve()), "sha256": digest(m17_path)}]),
        }
    return records


def import_reused(destination, source_records):
    imported, references = {}, {}
    for index, record in source_records.items():
        target = destination / "reused_four_sizes" / f"run-{index + 1:03d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        content = {"global_index": index, "results": record["results"],
                   "source_model_records": record["source_model_records"],
                   "reference_directory": record["reference_directory"],
                   "reference_report_sha256": record["reference_report_sha256"]}
        write_json(target, content)
        imported[f"run-{index}"] = {"path": str(target.relative_to(destination)), "sha256": digest(target)}
        references[f"run-{index}"] = {"directory": record["reference_directory"],
                                      "report_sha256": record["reference_report_sha256"]}
    return imported, references


def job_plan(spec, smoke=False):
    count = 2 if smoke else spec["repetitions"]
    return {f"run-{index}": {"id": f"run-{index}", "global_index": index,
                              "relative_directory": f"new/run-{index + 1:03d}"}
            for index in range(count)}


def validate_reference(reference):
    directory = Path(reference["directory"])
    report_path = directory / "report.json"
    if digest(report_path) != reference["report_sha256"]:
        raise ValueError("reference report changed")
    report = read_json(report_path)
    for name, expected in report["artifacts_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("reference artifact changed")
    return directory


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
        provenance = destination / "source_provenance/four_size_R100"
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
        source_records = collect_source_records(source_batch, source_frozen, source_status, count)
        imported, references = ({}, {f"run-{i}": {"directory": source_records[i]["reference_directory"],
                                                    "report_sha256": source_records[i]["reference_report_sha256"]}
                                     for i in range(count)}) if smoke else import_reused(destination, source_records)
        frozen = {
            "requested_protocol": spec, "base": effective_base(source_frozen, spec, smoke), "smoke_only": smoke,
            "source_sha256": hashes,
            "source_provenance_sha256": {str(path.relative_to(destination / "source_provenance")): digest(path)
                                           for path in (destination / "source_provenance").rglob("*") if path.is_file()},
            "imported_reused_models": imported, "references": references,
            "source_four_size_frozen_protocol_sha256": source_status["frozen_protocol_sha256"],
            "source_four_size_summary_sha256": source_verification["summary_sha256"],
            "workers": workers, "python_version": sys.version,
            "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        }
        write_json(destination / "frozen_protocol.json", frozen)
        status = {"status": "running", "stage": "six_added_dictionary_sizes", "smoke_only": smoke,
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
                          "new_jobs": spec["new_jobs"], "new_models": spec["new_models"]}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / (
        ("dictionary-grid-R100-smoke-" if args.smoke else "dictionary-grid-R100-") + stamp)
    run_batch(spec, destination, args.workers, args.smoke, args.resume)
    from scripts.summarize_1d_hat_dictionary_grid_R100 import summarize
    summarize(destination, destination / "summary")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
