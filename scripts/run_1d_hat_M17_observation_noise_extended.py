"""Extend the M=17 observation-noise sweep through sigma_obs=1 with 100 repetitions each."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
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
from scripts.run_1d_hat_M17_R100 import SOURCES as R100_SOURCES, digest, read_json
from scripts.run_1d_hat_sweeps import accuracy_job

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat_M17_observation_noise_extended.json"
SOURCES = list(dict.fromkeys(R100_SOURCES + [
    "scripts/run_1d_hat_M17_observation_noise_extended.py",
    "scripts/summarize_1d_hat_M17_observation_noise_extended.py",
    "scripts/verify_1d_hat_M17_observation_noise_extended.py",
    "configs/experiments/one_dimensional_hat_M17_observation_noise_extended.json",
    "test/test_hat_M17_observation_noise_extended.py",
]))


def source_state(spec):
    total = ROOT / spec["source_total100_batch"]
    total_frozen, total_status = read_json(total / "frozen_protocol.json"), read_json(total / "batch_status.json")
    old = ROOT / total_frozen["requested_protocol"]["source_batch"]
    old_frozen, old_status = read_json(old / "frozen_protocol.json"), read_json(old / "batch_status.json")
    return total, total_frozen, total_status, old, old_frozen, old_status


def validate(spec, state):
    total, total_frozen, total_status, old, old_frozen, old_status = state
    if spec["protocol_version"] != 1 or spec["dictionary_size"] != 17 or spec["repetitions"] != 100:
        raise ValueError("extension is fixed to M=17 and 100 repetitions")
    if spec["added_values"] != [.2, .4, .8, 1.] or spec["new_jobs"] != 400:
        raise ValueError("the four added observation-noise values changed")
    if spec["full_values"] != spec["existing_values"] + spec["added_values"]:
        raise ValueError("full observation-noise grid mismatch")
    if total_status["status"] != "completed" or len(total_status["results"]) != 1600:
        raise ValueError("M17 R100 source batch is incomplete")
    if old_status["status"] != "completed" or len(old_status["results"]) != 580:
        raise ValueError("original R20 source batch is incomplete")
    if digest(total / "frozen_protocol.json") != total_status["frozen_protocol_sha256"]:
        raise ValueError("M17 R100 frozen protocol changed")
    if digest(old / "frozen_protocol.json") != old_status["frozen_protocol_sha256"]:
        raise ValueError("R20 frozen protocol changed")
    if total_frozen["requested_protocol"]["sweeps"]["sigma_obs"] != spec["existing_values"]:
        raise ValueError("existing observation-noise grid mismatch")
    if old_frozen["base"]["replication"]["count"] != 20 or total_frozen["base"]["replication"]["count"] != 80:
        raise ValueError("expected immutable 20+80 source families")
    if old_frozen["protocol"]["observation_seed_offset"] != 1_000_000:
        raise ValueError("unexpected original observation seed offset")
    if total_frozen["protocol"]["observation_seed_offset"] != 10_000_000:
        raise ValueError("unexpected extension observation seed offset")


def combined_base(state):
    _, total_frozen, _, _, old_frozen, _ = state
    base = deepcopy(total_frozen["base"])
    base["basis"]["sizes"] = [17]
    base["replication"]["count"] = 100
    base["seeds"] = {split: old_frozen["base"]["seeds"][split] + total_frozen["base"]["seeds"][split]
                     for split in ("train", "validation", "test")}
    return base


def job_plan(spec, base):
    result = {}
    start = len(spec["existing_values"])
    for offset, value in enumerate(spec["added_values"]):
        case_id = f"sigma_obs-{start + offset:02d}"
        for index in range(spec["repetitions"]):
            key = f"{case_id}-{index}"
            result[key] = {
                "id": key, "case": case_id, "index": index, "value": value,
                "simulation": {**base["simulation"], "sigma_obs": value},
                "observation_seed_offset": 1_000_000 if index < 20 else 10_000_000,
                "relative_directory": f"added/{case_id}/run-{index + 1:03d}",
            }
    return result


def baseline_record(state, index):
    total, _, total_status, old, _, old_status = state
    if index < 20:
        record = old_status["results"][f"base-{index}"]
        root, local = old, index
    else:
        local = index - 20
        record = total_status["results"][f"base-{local}"]
        root = total
    report = root / record["directory"] / "report.json"
    if record["status"] != "completed" or digest(report) != record["report_sha256"]:
        raise ValueError("baseline source report failed hash/status check")
    content = read_json(report)
    if content["results"]["hat_M17"]["status"] != "evaluated":
        raise ValueError("baseline M17 model is unavailable")
    return root / record["directory"], report, record["report_sha256"], local


def create_baseline_links(batch, state):
    result = {}
    links = batch / "baseline_sources"
    links.mkdir()
    for index in range(100):
        source, report, report_hash, local = baseline_record(state, index)
        link = links / f"run-{index + 1:03d}"
        link.symlink_to(source.resolve(), target_is_directory=True)
        result[str(index)] = {"link": str(link.relative_to(batch)), "source_directory": str(source.resolve()),
                              "source_report": str(report.resolve()), "source_report_sha256": report_hash,
                              "source_local_index": local}
    return result


def worker(batch, job, sweep, base, baseline):
    batch, parent = Path(batch), Path(batch) / job["relative_directory"]
    directory, attempt = parent, 1
    while directory.exists():
        attempt += 1
        directory = parent.with_name(parent.name + f"-attempt-{attempt}")
    directory.mkdir(parents=True)
    started = perf_counter()
    result = {"job": job, "directory": str(directory.relative_to(batch)), "status": "running", "attempt": attempt}
    write_json(directory / "job_status.json", result)
    try:
        local_sweep = deepcopy(sweep)
        local_sweep["observation_seed_offset"] = job["observation_seed_offset"]
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
            content = accuracy_job(batch, directory, job, local_sweep, base, Path(baseline))
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


def run_batch(spec, destination, workers=6, resume=False):
    state = source_state(spec)
    validate(spec, state)
    destination = Path(destination).resolve()
    if resume:
        frozen = read_json(destination / "frozen_protocol.json")
        if frozen["requested_protocol"] != spec:
            raise ValueError("resume protocol mismatch")
        for name, expected in frozen["source_sha256"].items():
            if digest(ROOT / name) != expected or digest(destination / "source_snapshot" / name) != expected:
                raise ValueError("resume source mismatch")
        status = read_json(destination / "batch_status.json")
        if digest(destination / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
            raise ValueError("frozen protocol changed")
    else:
        destination.mkdir(parents=True, exist_ok=False)
        hashes = {name: digest(ROOT / name) for name in SOURCES}
        for name in SOURCES:
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        base = combined_base(state)
        sweep = deepcopy(state[1]["protocol"])
        baselines = create_baseline_links(destination, state)
        frozen = {"requested_protocol": spec, "base": base, "protocol": sweep,
                  "baseline_sources": baselines, "source_sha256": hashes, "workers": workers,
                  "source_frozen_protocol_sha256": {"total100": state[2]["frozen_protocol_sha256"],
                                                     "original20": state[5]["frozen_protocol_sha256"]},
                  "python_version": sys.version, "numpy_version": np.__version__, "sklearn_version": sklearn.__version__}
        write_json(destination / "frozen_protocol.json", frozen)
        status = {"status": "running", "stage": "added_observation_noise",
                  "frozen_protocol_sha256": digest(destination / "frozen_protocol.json"),
                  "jobs": job_plan(spec, base), "results": {}}
        write_json(destination / "batch_status.json", status)
    started = perf_counter()
    pending = {key: job for key, job in status["jobs"].items() if key not in status["results"]}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {}
        for key, job in pending.items():
            baseline = destination / frozen["baseline_sources"][str(job["index"])]["link"]
            futures[pool.submit(worker, destination, job, frozen["protocol"], frozen["base"], baseline)] = key
        for future in as_completed(futures):
            key = futures[future]
            record = future.result()
            status["results"][key] = record
            write_json(destination / "batch_status.json", status)
            print(f"{len(status['results'])}/{len(status['jobs'])}: {key} {record['status']} {record['elapsed_seconds']:.1f}s", flush=True)
    status["status"] = "completed" if all(row["status"] == "completed" for row in status["results"].values()) else "completed_with_failures"
    status["elapsed_seconds_this_invocation"] = perf_counter() - started
    write_json(destination / "batch_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    spec = read_json(args.config)
    state = source_state(spec)
    validate(spec, state)
    base = combined_base(state)
    if not args.run and args.resume is None:
        print({"status": "configuration_valid", "jobs": len(job_plan(spec, base)),
               "values": spec["added_values"], "repetitions": spec["repetitions"]})
        return
    destination = args.resume or args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / (
        "M17-observation-extended-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    status = run_batch(spec, destination, args.workers, args.resume is not None)
    from scripts.summarize_1d_hat_M17_observation_noise_extended import summarize
    summarize(destination, Path(destination) / "summary")
    print(destination)
    if status["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
