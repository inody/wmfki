"""Extend the four M=17 physical-parameter sweeps to 100 repetitions each."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
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
from scripts.prepare_1d_hat_experiment import write_json
from scripts.run_1d_hat_repetitions import SOURCES as BASE_SOURCES
from scripts.run_1d_hat_sweeps import accuracy_job, cases, digest, read_json

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat_M17_R100.json"
SOURCES = list(dict.fromkeys(BASE_SOURCES + [
    "scripts/hat_sweep_numerics.py", "scripts/run_1d_hat_sweeps.py",
    "scripts/summarize_1d_hat_sweeps.py", "scripts/verify_1d_hat_sweeps.py",
    "scripts/run_1d_hat_M17_R100.py", "scripts/summarize_1d_hat_M17_R100.py",
    "scripts/verify_1d_hat_M17_R100.py", "configs/experiments/one_dimensional_hat_sweeps.json",
    "configs/experiments/one_dimensional_hat_M17_R100.json", "test/test_hat_M17_R100.py",
]))


def seed_lists(count=80, start=8_000_000):
    return {
        "train": [start + 10 * k for k in range(count)],
        "validation": [start + 10 * k + 2 for k in range(count)],
        "test": [start + 10 * k + 4 for k in range(count)],
    }


def unique_cases(sweep_spec, base):
    result = {}
    for case in cases(sweep_spec, base):
        result.setdefault(case["id"], case)
    return result


def effective_base(source_frozen, spec, smoke=False):
    base = deepcopy(source_frozen["base"])
    count = 2 if smoke else spec["new_repetitions"]
    base["basis"]["sizes"] = [spec["dictionary_size"]]
    base["seeds"] = seed_lists(count, 8_000_000)
    base["replication"]["count"] = count
    if smoke:
        base["simulation"].update(N=12, horizon=.1, dt=.005)
        base["estimator"].update(alpha_count=5, max_iter=1000)
    return base


def effective_sweep(source_frozen, spec, smoke=False):
    sweep = deepcopy(source_frozen["protocol"])
    sweep["observation_seed_offset"] = spec["observation_seed_offset"]
    if smoke:
        sweep["sweeps"] = {"sigma_obs": [0., .01], "sigma": [.1, .2], "N": [8, 12], "dt": [.005, .01]}
    else:
        sweep["sweeps"] = deepcopy(spec["sweeps"])
    return sweep


def validate_protocol(spec, source_frozen, source_status):
    if spec["protocol_version"] != 1 or spec["dictionary_size"] != 17:
        raise ValueError("this protocol is fixed to M=17")
    if (spec["existing_repetitions"], spec["new_repetitions"], spec["total_repetitions"]) != (20, 80, 100):
        raise ValueError("the declared extension is 20 existing plus 80 new repetitions")
    if spec["unique_physical_conditions"] != 20 or spec["new_jobs"] != 1600 or spec["existing_records"] != 400:
        raise ValueError("the declared four-sweep job counts changed")
    if source_status["status"] != "completed" or len(source_status["results"]) != len(source_status["jobs"]):
        raise ValueError("source sweep must be complete")
    if digest(ROOT / spec["source_batch"] / "frozen_protocol.json") != source_status["frozen_protocol_sha256"]:
        raise ValueError("source frozen protocol hash mismatch")
    source_spec, source_base = source_frozen["protocol"], source_frozen["base"]
    if spec["sweeps"] != source_spec["sweeps"]:
        raise ValueError("the four grids must be identical to the completed sweep")
    if source_base["basis"]["sizes"] != [9, 17, 33, 65] or source_base["replication"]["count"] != 20:
        raise ValueError("unexpected source dictionary/repetition protocol")
    if spec["observation_seed_offset"] == source_spec["observation_seed_offset"]:
        raise ValueError("new observation noise needs a disjoint seed family")
    seeds = seed_lists()
    used = set()
    for values in seeds.values():
        for seed in values:
            if used.intersection({seed, seed + 1}):
                raise ValueError("new trajectory/Brownian seeds overlap")
            used.update({seed, seed + 1})
    old = {value + offset for values in source_base["seeds"].values() for value in values for offset in (0, 1)}
    observations = {spec["observation_seed_offset"] + value for values in seeds.values() for value in values}
    if used & old or used & observations or old & observations:
        raise ValueError("old, new and observation seed families must be disjoint")
    if len(unique_cases(source_spec, source_base)) != 20:
        raise ValueError("expected exactly 20 unique physical conditions")


def job_plan(sweep, base):
    count = base["replication"]["count"]
    result = {}
    for case_id, case in unique_cases(sweep, base).items():
        for index in range(count):
            key = f"{case_id}-{index}"
            relative = f"new/base/run-{index + 1:03d}" if case_id == "base" else f"new/accuracy/{case_id}/run-{index + 1:03d}"
            result[key] = dict(id=key, case=case_id, index=index, simulation=case["simulation"], relative_directory=relative)
    return result


def source_job_key(case_id, index):
    return f"base-{index}" if case_id == "base" else f"{case_id}-{index}"


def import_existing(destination, spec, source_frozen, source_status):
    source = ROOT / spec["source_batch"]
    imported = {}
    source_cases = unique_cases(source_frozen["protocol"], source_frozen["base"])
    for case_id, case in source_cases.items():
        for index in range(spec["existing_repetitions"]):
            key = source_job_key(case_id, index)
            record = source_status["results"][key]
            report_path = source / record["directory"] / "report.json"
            if record["status"] != "completed" or digest(report_path) != record["report_sha256"]:
                raise ValueError("source report failed its saved hash/status check")
            report = read_json(report_path)
            model = report["results"]["hat_M17"]
            if model["basis"]["M"] != 17 or model["status"] != "evaluated":
                raise ValueError("source M=17 model is missing")
            target = destination / "existing" / case_id / f"run-{index + 1:03d}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            compact = {
                "status": "imported", "case": case_id, "index": index,
                "simulation": case["simulation"], "result": model,
                "source_report": str(report_path.resolve()), "source_report_sha256": record["report_sha256"],
                "source_frozen_protocol_sha256": source_status["frozen_protocol_sha256"],
            }
            write_json(target, compact)
            imported[f"{case_id}-{index}"] = {"path": str(target.relative_to(destination)), "sha256": digest(target)}
    if len(imported) != spec["existing_records"]:
        raise ValueError("existing record count mismatch")
    return imported


def worker(batch, job, sweep, base, smoke, baseline_directory=None):
    batch = Path(batch)
    parent = batch / job["relative_directory"]
    directory, attempt = parent, 1
    while directory.exists():
        attempt += 1
        directory = parent.with_name(parent.name + f"-attempt-{attempt}")
    directory.mkdir(parents=True)
    started = perf_counter()
    result = {"job": job, "directory": str(directory.relative_to(batch)), "status": "running",
              "smoke_only": smoke, "attempt": attempt}
    write_json(directory / "job_status.json", result)
    try:
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
            content = accuracy_job(batch, directory, job, sweep, base,
                                   Path(baseline_directory) if baseline_directory else None)
        if set(content["results"]) != {"hat_M17"}:
            raise ValueError("only M=17 may be fitted")
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
    source = ROOT / spec["source_batch"]
    source_frozen, source_status = read_json(source / "frozen_protocol.json"), read_json(source / "batch_status.json")
    validate_protocol(spec, source_frozen, source_status)
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
        provenance.mkdir()
        for filename in ("frozen_protocol.json", "batch_status.json"):
            shutil.copy2(source / filename, provenance / filename)
        shutil.copytree(source / "verification", provenance / "verification")
        hashes = {name: digest(ROOT / name) for name in SOURCES}
        for name in SOURCES:
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        base = effective_base(source_frozen, spec, smoke)
        sweep = effective_sweep(source_frozen, spec, smoke)
        imported = {} if smoke else import_existing(destination, spec, source_frozen, source_status)
        frozen = {
            "requested_protocol": spec, "protocol": sweep, "base": base, "smoke_only": smoke,
            "source_sha256": hashes, "source_provenance_sha256": {
                str(path.relative_to(provenance)): digest(path) for path in provenance.rglob("*") if path.is_file()},
            "imported_existing": imported, "workers": workers, "python_version": sys.version,
            "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        }
        write_json(destination / "frozen_protocol.json", frozen)
        status = {"status": "running", "stage": "new_baseline", "smoke_only": smoke,
                  "frozen_protocol_sha256": digest(destination / "frozen_protocol.json"),
                  "jobs": job_plan(sweep, base), "results": {}}
        write_json(destination / "batch_status.json", status)
    base, sweep = frozen["base"], frozen["protocol"]
    started = perf_counter()
    for stage in ("base", "accuracy"):
        pending = {key: job for key, job in status["jobs"].items()
                   if ((job["case"] == "base") == (stage == "base")) and key not in status["results"]}
        status.update(status="running", stage="new_baseline" if stage == "base" else "new_accuracy")
        write_json(destination / "batch_status.json", status)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {}
            for key, job in pending.items():
                baseline = None if stage == "base" else destination / status["results"][f"base-{job['index']}"]["directory"]
                futures[pool.submit(worker, destination, job, sweep, base, smoke, baseline)] = key
            for future in as_completed(futures):
                key = futures[future]
                record = future.result()
                status["results"][key] = record
                write_json(destination / "batch_status.json", status)
                print(f"{len(status['results'])}/{len(status['jobs'])}: {key} {record['status']} {record['elapsed_seconds']:.1f}s", flush=True)
        if stage == "base" and any(status["results"][key]["status"] != "completed" for key in pending):
            status["status"] = "blocked_new_baseline"
            write_json(destination / "batch_status.json", status)
            raise RuntimeError("new baseline jobs failed; dependent jobs were not launched")
    status["status"] = "completed" if all(record["status"] == "completed" for record in status["results"].values()) else "completed_with_failures"
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
    source = ROOT / spec["source_batch"]
    frozen, status = read_json(source / "frozen_protocol.json"), read_json(source / "batch_status.json")
    validate_protocol(spec, frozen, status)
    base, sweep = effective_base(frozen, spec, args.smoke), effective_sweep(frozen, spec, args.smoke)
    if not args.run and not args.smoke:
        print(json.dumps({"status": "configuration_valid", "new_jobs": len(job_plan(sweep, base)),
                          "unique_conditions": len(unique_cases(sweep, base)), "dictionary_size": 17}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / (("M17-R100-smoke-" if args.smoke else "M17-R100-") + stamp)
    run_batch(spec, destination, args.workers, args.smoke, args.resume)
    from scripts.summarize_1d_hat_M17_R100 import summarize
    output = destination / "summary"
    if output.exists():
        output = destination / f"summary-{stamp}"
    summarize(destination, output)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
