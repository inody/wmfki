"""Compute the complete 1D hat-dictionary grid without historical outputs.

This runner is the bootstrap entry point for a fresh-clone L4DC reproduction.
It preserves the 100 predeclared train/validation/test seed triplets used by the
published staged experiment, but fits all ten dictionary sizes directly rather
than importing four sizes from historical batches.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
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

from scripts.prepare_1d_hat_experiment import prepare, write_json
from scripts.run_1d_hat_dictionary_grid_R100 import job_plan
from scripts.run_1d_hat_dictionary_size_R100 import digest, fit_job, read_json

DEFAULT_CONFIG = (
    ROOT / "configs/experiments/one_dimensional_hat_dictionary_grid_from_scratch_R100.json"
)
SOURCES = [
    "features.py", "kernels.py", "pipeline.py", "simulate.py", "tests.py",
    "time_tests.py", "regression.py", "metrics.py", "config.py", "utils.py",
    "scripts/prepare_1d_hat_experiment.py", "scripts/run_1d_hat_lasso.py",
    "scripts/hat_full_pair_assembly.py", "scripts/hat_sweep_numerics.py",
    "scripts/run_1d_hat_sweeps.py", "scripts/run_1d_hat_dictionary_size_R100.py",
    "scripts/run_1d_hat_dictionary_grid_R100.py",
    "scripts/run_1d_hat_dictionary_grid_from_scratch.py",
    "scripts/summarize_1d_hat_dictionary_grid_R100.py",
    "scripts/verify_1d_hat_dictionary_grid_R100.py",
    "configs/experiments/one_dimensional_hat_dictionary_grid_from_scratch_R100.json",
    "configs/experiments/one_dimensional_hat_nested_20.json",
]


def validate_protocol(requested: dict, base: dict) -> None:
    if requested.get("task") != "hat_dictionary_grid_from_scratch_R100" \
            or requested.get("protocol_version") != 1:
        raise ValueError("invalid from-scratch hat-grid protocol")
    base_path = ROOT / requested["base_config"]
    if digest(base_path) != requested["base_config_sha256"]:
        raise ValueError("base hat protocol hash changed")
    sizes = [9, 11, 14, 17, 22, 27, 33, 42, 52, 65]
    if requested.get("dictionary_sizes") != sizes:
        raise ValueError("dictionary grid changed")
    if requested.get("reused_sizes") != [] \
            or requested.get("newly_fitted_sizes") != sizes:
        raise ValueError("from-scratch protocol must fit every dictionary")
    if (requested.get("repetitions"), requested.get("new_jobs"),
            requested.get("new_models"), requested.get("reused_models")) \
            != (100, 100, 1000, 0):
        raise ValueError("planned from-scratch counts changed")
    if base.get("protocol_version") != 1 or base["replication"]["count"] != 20:
        raise ValueError("unexpected base protocol")
    if base["basis"]["sizes"] != [9, 17, 33, 65]:
        raise ValueError("unexpected historical base dictionary")


def seed_triplet(base: dict, index: int) -> dict[str, int]:
    if not 0 <= index < 100:
        raise ValueError("repetition index outside 0..99")
    if index < 20:
        return {split: int(base["seeds"][split][index])
                for split in ("train", "validation", "test")}
    k = index - 20
    return {
        "train": 8_000_000 + 10 * k,
        "validation": 8_000_002 + 10 * k,
        "test": 8_000_004 + 10 * k,
    }


def effective_base(base: dict, requested: dict, smoke: bool) -> dict:
    result = deepcopy(base)
    result["basis"]["sizes"] = requested["dictionary_sizes"][:2] \
        if smoke else requested["dictionary_sizes"]
    result["replication"]["count"] = 2 if smoke else requested["repetitions"]
    if smoke:
        result["estimator"].update(alpha_count=5, max_iter=2000)
    return result


def preparation_spec(base: dict, seeds: dict[str, int]) -> dict:
    spec = deepcopy(base)
    spec.pop("replication", None)
    spec["basis"]["sizes"] = [9]
    spec["seeds"] = {split: [int(seed)] for split, seed in seeds.items()}
    return spec


def create_reference(directory: Path, base: dict, index: int, smoke: bool) -> dict:
    seeds = seed_triplet(base, index)
    assembly = directory / "assembly"
    prepare(preparation_spec(base, seeds), assembly, smoke=smoke, backend="cached_1d")
    manifest = read_json(assembly / "manifest.json")
    artifacts = {"assembly/manifest.json": digest(assembly / "manifest.json")}
    for split in ("train", "validation", "test"):
        effective_seed = int(manifest["effective_protocol"]["seeds"][split][0])
        source = assembly / split / str(effective_seed)
        target = directory / split
        target.mkdir(parents=True)
        for name in ("data.npz", "response.npy"):
            shutil.copy2(source / name, target / name)
            artifacts[f"{split}/{name}"] = digest(target / name)
    report = {
        "status": "completed",
        "global_index": index,
        "seeds": seeds,
        "smoke_only": smoke,
        "artifacts_sha256": artifacts,
    }
    write_json(directory / "report.json", report)
    return {"directory": str(directory.resolve()), "report_sha256": digest(directory / "report.json")}


def worker(batch_string: str, job: dict, base: dict, source_base: dict, smoke: bool) -> dict:
    batch = Path(batch_string)
    parent = batch / job["relative_directory"]
    directory, attempt = parent, 1
    while directory.exists():
        attempt += 1
        directory = parent.with_name(parent.name + f"-attempt-{attempt}")
    reference = batch / "references" / f"run-{job['global_index'] + 1:03d}"
    if reference.exists():
        shutil.rmtree(reference)
    directory.mkdir(parents=True, exist_ok=False)
    reference.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    result = {"job": job, "directory": str(directory.relative_to(batch)),
              "status": "running", "attempt": attempt}
    write_json(directory / "job_status.json", result)
    try:
        with (directory / "run.log").open("w") as log, redirect_stdout(log), \
                redirect_stderr(log), threadpool_limits(limits=1):
            reference_record = create_reference(
                reference, source_base, int(job["global_index"]), smoke
            )
            content = fit_job(directory, job, base, reference)
        result.update(
            content,
            status="completed",
            reference_directory=reference_record["directory"],
            reference_report_sha256=reference_record["report_sha256"],
        )
    except Exception as error:  # retain every failed repetition
        result.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    result["elapsed_seconds"] = perf_counter() - started
    result["artifacts_sha256"] = {
        str(path.relative_to(directory)): digest(path)
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"job_status.json", "report.json"}
    }
    write_json(directory / "report.json", result)
    write_json(directory / "job_status.json", {
        key: result[key] for key in ("status", "directory", "elapsed_seconds", "attempt")
    })
    return {
        "status": result["status"],
        "directory": result["directory"],
        "report_sha256": digest(directory / "report.json"),
        "elapsed_seconds": result["elapsed_seconds"],
        "error": result.get("error"),
    }


def run_batch(
    requested: dict, destination: Path, workers: int, smoke: bool, resume: bool = False
) -> dict:
    source_base = read_json(ROOT / requested["base_config"])
    validate_protocol(requested, source_base)
    destination = destination.resolve()
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    base = effective_base(source_base, requested, smoke)
    count = int(base["replication"]["count"])
    jobs = job_plan(requested, smoke)
    if resume:
        frozen = read_json(destination / "frozen_protocol.json")
        status = read_json(destination / "batch_status.json")
        if frozen["requested_protocol"] != requested or frozen["base"] != base \
                or frozen["smoke_only"] != smoke or status["jobs"] != jobs:
            raise ValueError("resume protocol mismatch")
        for name, expected in frozen["source_sha256"].items():
            if hashes[name] != expected or digest(destination / "source_snapshot" / name) != expected:
                raise ValueError("resume source mismatch")
        for key, record in status["results"].items():
            if record["status"] == "completed" and digest(
                destination / record["directory"] / "report.json"
            ) != record["report_sha256"]:
                raise ValueError(f"completed report changed: {key}")
        status["status"] = "running"
    else:
        destination.mkdir(parents=True, exist_ok=False)
        for name in SOURCES:
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        frozen = {
            "requested_protocol": requested,
            "base": base,
            "smoke_only": smoke,
            "source_sha256": hashes,
            "source_provenance_sha256": {},
            "imported_reused_models": {},
            "references": {},
            "source_mode": "generated_in_this_run",
            "workers": workers,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
        }
        write_json(destination / "frozen_protocol.json", frozen)
        status = {
            "status": "running",
            "stage": "all_dictionary_sizes_from_scratch",
            "smoke_only": smoke,
            "frozen_protocol_sha256": digest(destination / "frozen_protocol.json"),
            "jobs": jobs,
            "results": {},
        }
    write_json(destination / "batch_status.json", status)
    pending = {
        key: job for key, job in jobs.items()
        if status["results"].get(key, {}).get("status") != "completed"
    }
    started = perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(worker, str(destination), job, base, source_base, smoke): key
            for key, job in pending.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            record = future.result()
            status["results"][key] = record
            write_json(destination / "batch_status.json", status)
            print(
                f"{len(status['results'])}/{count}: {key} {record['status']} "
                f"{record['elapsed_seconds']:.1f}s",
                flush=True,
            )
    status["status"] = "completed" if all(
        record["status"] == "completed" for record in status["results"].values()
    ) else "completed_with_failures"
    status["elapsed_seconds_this_invocation"] = perf_counter() - started
    frozen["references"] = {}
    for key, record in status["results"].items():
        report = read_json(destination / record["directory"] / "report.json")
        if report["status"] == "completed":
            frozen["references"][key] = {
                "directory": report["reference_directory"],
                "report_sha256": report["reference_report_sha256"],
            }
    write_json(destination / "frozen_protocol.json", frozen)
    status["frozen_protocol_sha256"] = digest(destination / "frozen_protocol.json")
    write_json(destination / "batch_status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    requested = read_json(args.config)
    source_base = read_json(ROOT / requested["base_config"])
    validate_protocol(requested, source_base)
    if not args.run and not args.smoke:
        print(json.dumps({
            "status": "configuration_valid",
            "dictionary_sizes": requested["dictionary_sizes"],
            "repetitions": requested["repetitions"],
            "historical_outputs_required": False,
        }, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / (
        ("dictionary-grid-from-scratch-smoke-" if args.smoke
         else "dictionary-grid-from-scratch-R100-") + stamp
    )
    run_batch(requested, destination, args.workers, args.smoke, resume=args.resume)
    from scripts.summarize_1d_hat_dictionary_grid_R100 import summarize
    summary_directory = destination / "summary"
    if not (args.resume and summary_directory.is_dir()):
        summarize(destination, summary_directory)
    print(destination)


if __name__ == "__main__":
    main()
