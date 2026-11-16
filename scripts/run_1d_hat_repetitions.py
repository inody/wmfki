"""Run 20 independent repetitions for nested hat dictionaries, sharing trajectories within each repetition and retraining between repetitions."""

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

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_1d_hat_experiment import prepare, validate_spec, write_json
from scripts.run_1d_hat_lasso import run as fit

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat_nested_20.json"
SOURCES = [
    "features.py", "kernels.py", "simulate.py", "tests.py", "time_tests.py",
    "config.py", "pipeline.py", "regression.py", "metrics.py", "bandwidth.py", "utils.py",
    "scripts/run_weak_increment_validation.py", "scripts/r05_numerical_settings.py",
    "scripts/run_messenger_qanr_sweep.py", "pyproject.toml",
    "configs/experiments/one_dimensional_hat_N1000.json",
    "configs/experiments/one_dimensional_hat_nested_20.json",
    "scripts/run_sindy_comparison.py", "scripts/prepare_1d_hat_experiment.py",
    "scripts/hat_full_pair_assembly.py", "scripts/run_1d_hat_lasso.py",
    "scripts/run_1d_hat_repetitions.py", "scripts/summarize_1d_hat_repetitions.py",
]


def validate_repetitions(spec):
    validate_spec(spec)
    reference = json.loads((ROOT / "configs/experiments/one_dimensional_hat_N1000.json").read_text())
    for key in ("simulation", "true_kernel", "baseline", "estimator", "pair_mode", "pair_chunk_size", "selection_rule"):
        if spec[key] != reference[key]:
            raise ValueError(f"the replication must preserve {key}")
    expected_basis = {**reference["basis"], "sizes": [9, 17, 33, 65]}
    if spec["basis"] != expected_basis:
        raise ValueError("the approved nested sizes are 9,17,33,65")
    repetition = spec["replication"]
    if repetition["count"] != 20 or repetition["unit"] != "independent_train_validation_test_triplet":
        raise ValueError("exactly 20 independently fitted triplets are required")
    if any(len(seeds) != 20 for seeds in spec["seeds"].values()):
        raise ValueError("each split must declare 20 seeds before execution")
    if repetition["bootstrap_samples"] != 10000 or repetition["bootstrap_seed"] != 730000:
        raise ValueError("bootstrap settings must stay fixed")


def triplet_spec(spec, index):
    result = deepcopy(spec)
    result["seeds"] = {split: [seeds[index]] for split, seeds in spec["seeds"].items()}
    result.pop("replication")
    result["replication_context"] = {"index": index, "planned_count": spec["replication"]["count"]}
    return result


def run_one(index, spec, directory, smoke):
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    started = perf_counter()
    result = {"index": index, "directory": str(directory), "status": "running"}
    write_json(directory / "run_status.json", result)
    with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            with threadpool_limits(limits=1):
                prepare(triplet_spec(spec, index), directory / "assembly", smoke=smoke, backend="cached_1d")
                report = fit(directory / "assembly", directory / "fit", allow_smoke=smoke)
            result["status"] = report["status"]
            result["report_path"] = str(directory / "fit" / "report.json")
            result["report_sha256"] = hashlib.sha256((directory / "fit" / "report.json").read_bytes()).hexdigest()
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
    result["elapsed_seconds"] = perf_counter() - started
    write_json(directory / "run_status.json", result)
    return result


def run_batch(spec, out_dir, *, workers=4, smoke=False):
    validate_repetitions(spec)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must lie between 1 and 8")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    count = 2 if smoke else spec["replication"]["count"]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    for name in SOURCES:
        destination = out_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    frozen = {"protocol": spec, "smoke_only": smoke, "effective_repetitions": count,
              "effective_sizes": [4, 8] if smoke else spec["basis"]["sizes"],
              "assembly_backend": "cached_1d", "workers": workers, "source_sha256": source_hashes,
              "python_version": sys.version}
    write_json(out_dir / "frozen_protocol.json", frozen)
    status = {"status": "running", "smoke_only": smoke, "planned_repetitions": count,
              "frozen_protocol_sha256": hashlib.sha256((out_dir / "frozen_protocol.json").read_bytes()).hexdigest(),
              "runs": [{"index": index, "status": "pending"} for index in range(count)]}
    write_json(out_dir / "batch_status.json", status)
    started = perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(run_one, index, spec, out_dir / f"run-{index + 1:02d}", smoke): index
                       for index in range(count)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {"index": index, "status": "failed", "error": repr(error)}
                status["runs"][index] = result
                write_json(out_dir / "batch_status.json", status)
                print(f"{sum(run['status'] != 'pending' for run in status['runs'])}/{count}: run {index + 1} {result['status']}", flush=True)
        status["status"] = "completed" if all(run["status"] == "completed" for run in status["runs"]) else "completed_with_failures"
    except Exception as error:
        status["status"] = "failed"
        status["error"] = repr(error)
        raise
    finally:
        status["elapsed_seconds"] = perf_counter() - started
        write_json(out_dir / "batch_status.json", status)
    from scripts.summarize_1d_hat_repetitions import summarize
    summarize(out_dir)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    validate_repetitions(spec)
    if not args.run and not args.smoke:
        print(json.dumps({"status": "configuration_valid", "protocol": spec}, ensure_ascii=False, indent=2))
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = "nested-smoke" if args.smoke else "nested-N1000-R20"
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / f"{name}-{timestamp}"
    result = run_batch(spec, destination, workers=args.workers, smoke=args.smoke)
    print(json.dumps({"status": result["status"], "output": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
