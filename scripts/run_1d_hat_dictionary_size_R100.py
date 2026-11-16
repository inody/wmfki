"""Compare the baseline conditions for M=9,17,33,65 over the same 100 repetitions."""

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
from scripts.hat_sweep_numerics import assemble_full, evaluate_model, evaluation_statistics
from scripts.prepare_1d_hat_experiment import array_hash, basis_params, matrix_diagnostics, write_json
from scripts.run_1d_hat_lasso import select_lasso
from scripts.run_1d_hat_sweeps import load_data, read_json, save_statistics, test_family

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat_dictionary_size_R100.json"
SOURCES = [
    "features.py", "kernels.py", "pipeline.py", "simulate.py",
    "scripts/prepare_1d_hat_experiment.py", "scripts/run_1d_hat_lasso.py",
    "scripts/hat_full_pair_assembly.py", "scripts/hat_sweep_numerics.py",
    "scripts/run_1d_hat_sweeps.py", "scripts/run_1d_hat_M17_R100.py",
    "scripts/run_1d_hat_dictionary_size_R100.py",
    "scripts/summarize_1d_hat_dictionary_size_R100.py",
    "scripts/verify_1d_hat_dictionary_size_R100.py",
    "scripts/generate_1d_hat_practical_figure.py",
    "configs/experiments/one_dimensional_hat_sweeps.json",
    "configs/experiments/one_dimensional_hat_M17_R100.json",
    "configs/experiments/one_dimensional_hat_dictionary_size_R100.json",
    "test/test_hat_dictionary_size_R100.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sources(spec):
    existing = ROOT / spec["source_existing_batch"]
    reference = ROOT / spec["source_new_reference_batch"]
    return (
        existing, read_json(existing / "frozen_protocol.json"), read_json(existing / "batch_status.json"),
        reference, read_json(reference / "frozen_protocol.json"), read_json(reference / "batch_status.json"),
    )


def comparable_base(base):
    result = {key: value for key, value in base.items() if key not in ("seeds", "replication")}
    result["basis"] = {**result["basis"], "sizes": [9, 17, 33, 65]}
    return result


def validate_protocol(spec, existing_frozen, existing_status, reference_frozen, reference_status):
    if spec["protocol_version"] != 1 or spec["dictionary_sizes"] != [9, 17, 33, 65]:
        raise ValueError("dictionary-size grid changed")
    if spec["newly_fitted_sizes"] != [9, 33, 65]:
        raise ValueError("M=17 must be reused rather than refitted")
    if (spec["existing_repetitions"], spec["new_repetitions"], spec["total_repetitions"]) != (20, 80, 100):
        raise ValueError("the comparison is fixed to 20 existing plus 80 new repetitions")
    if (spec["new_jobs"], spec["new_models"], spec["existing_models"], spec["reused_new_M17_models"]) != (80, 240, 80, 80):
        raise ValueError("declared job/model counts changed")
    if existing_status["status"] != "completed" or reference_status["status"] != "completed":
        raise ValueError("both source batches must be complete")
    existing_path = ROOT / spec["source_existing_batch"] / "frozen_protocol.json"
    reference_path = ROOT / spec["source_new_reference_batch"] / "frozen_protocol.json"
    if digest(existing_path) != existing_status["frozen_protocol_sha256"]:
        raise ValueError("existing protocol hash mismatch")
    if digest(reference_path) != reference_status["frozen_protocol_sha256"]:
        raise ValueError("reference protocol hash mismatch")
    if existing_frozen["base"]["basis"]["sizes"] != spec["dictionary_sizes"] or existing_frozen["base"]["replication"]["count"] != 20:
        raise ValueError("unexpected existing dictionary source")
    if reference_frozen["base"]["basis"]["sizes"] != [17] or reference_frozen["base"]["replication"]["count"] != 80:
        raise ValueError("unexpected new M=17 source")
    if comparable_base(existing_frozen["base"]) != comparable_base(reference_frozen["base"]):
        raise ValueError("physical, test, basis-family, or solver settings differ")
    if len([key for key in reference_status["jobs"] if key.startswith("base-")]) != 80:
        raise ValueError("reference source is missing baseline jobs")


def effective_base(reference_frozen, spec, smoke=False):
    base = deepcopy(reference_frozen["base"])
    base["basis"]["sizes"] = spec["newly_fitted_sizes"][:1] if smoke else spec["newly_fitted_sizes"]
    base["replication"]["count"] = 2 if smoke else spec["new_repetitions"]
    if smoke:
        base["estimator"] = {**base["estimator"], "alpha_count": 5, "max_iter": 2000}
    return base


def candidates(base):
    return [(f"hat_M{M}", basis_params(base, M)) for M in base["basis"]["sizes"]]


def job_plan(spec, smoke=False):
    count = 2 if smoke else spec["new_repetitions"]
    return {f"base-{index}": {
        "id": f"base-{index}", "index": index, "reference_key": f"base-{index}",
        "relative_directory": f"new/run-{index + 21:03d}",
    } for index in range(count)}


def reference_directory(reference_batch, reference_status, job):
    record = reference_status["results"][job["reference_key"]]
    directory = reference_batch / record["directory"]
    report_path = directory / "report.json"
    if record["status"] != "completed" or digest(report_path) != record["report_sha256"]:
        raise ValueError("reference report hash/status mismatch")
    report = read_json(report_path)
    if report["job"]["case"] != "base" or report["job"]["index"] != job["index"]:
        raise ValueError("reference identity mismatch")
    if set(report["results"]) != {"hat_M17"} or report["results"]["hat_M17"]["status"] != "evaluated":
        raise ValueError("reference M=17 result missing")
    for name, expected in report["artifacts_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("reference artifact changed")
    return directory, record, report


def import_sources(destination, spec, existing_batch, existing_status, reference_batch, reference_status):
    imported_existing = {}
    for index in range(spec["existing_repetitions"]):
        key = f"base-{index}"
        record = existing_status["results"][key]
        report_path = existing_batch / record["directory"] / "report.json"
        if record["status"] != "completed" or digest(report_path) != record["report_sha256"]:
            raise ValueError("existing baseline report changed")
        report = read_json(report_path)
        if set(report["results"]) != {f"hat_M{M}" for M in spec["dictionary_sizes"]}:
            raise ValueError("existing dictionary results missing")
        target = destination / "existing" / f"run-{index + 1:03d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        compact = {"index": index, "results": report["results"], "source_report": str(report_path.resolve()),
                   "source_report_sha256": record["report_sha256"]}
        write_json(target, compact)
        imported_existing[key] = {"path": str(target.relative_to(destination)), "sha256": digest(target)}
    imported_m17 = {}
    for index in range(spec["new_repetitions"]):
        key = f"base-{index}"
        record = reference_status["results"][key]
        report_path = reference_batch / record["directory"] / "report.json"
        if record["status"] != "completed" or digest(report_path) != record["report_sha256"]:
            raise ValueError("new M=17 reference changed")
        report = read_json(report_path)
        target = destination / "reused_M17" / f"run-{index + 21:03d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        compact = {"index": index, "result": report["results"]["hat_M17"],
                   "source_report": str(report_path.resolve()), "source_report_sha256": record["report_sha256"]}
        write_json(target, compact)
        imported_m17[key] = {"path": str(target.relative_to(destination)), "sha256": digest(target)}
    return imported_existing, imported_m17


def assemble_split(directory, reference, split, base):
    source = reference / split / "data.npz"
    data = load_data(source)
    space, time = test_family(data["t"])
    designs = assemble_full(data["X"], data["t"], space, time, candidates(base))
    response_source = reference / split / "response.npy"
    response = np.load(response_source)
    directory.mkdir()
    np.save(directory / "response.npy", response)
    for name, A in designs.items():
        np.save(directory / f"{name}.npy", A)
    manifest = {
        "source_data": str(source.resolve()), "source_data_sha256": digest(source),
        "source_response": str(response_source.resolve()), "source_response_sha256": digest(response_source),
        "response_sha256": array_hash(response),
        "designs": {name: {"sha256": array_hash(A), **matrix_diagnostics(A)} for name, A in designs.items()},
    }
    return designs, response, data, manifest


def fit_job(directory, job, base, reference):
    splits, arrays = {}, {}
    for split in ("train", "validation"):
        A, b, _, manifest = assemble_split(directory / split, reference, split, base)
        arrays[split] = A, b
        splits[split] = manifest
    selections = {}
    for name, params in candidates(base):
        selection, path = select_lasso(arrays["train"][0][name], arrays["train"][1],
                                       arrays["validation"][0][name], arrays["validation"][1], base["estimator"])
        folder = directory / name
        folder.mkdir()
        np.save(folder / "path_coefficients.npy", path)
        selections[name] = {"basis": params, **selection}
    write_json(directory / "frozen_selections.json", selections)
    # Test data are accessed only after all dictionary selections are frozen.
    test_A, test_b, test, manifest = assemble_split(directory / "test", reference, "test", base)
    splits["test"] = manifest
    statistics = evaluation_statistics(test["X"], test["t"], candidates(base), base["true_kernel"])
    save_statistics(directory / "evaluation_statistics.npz", statistics)
    results = {}
    for name, params in candidates(base):
        selected = selections[name]
        if selected["status"] != "selected":
            results[name] = {"status": selected["status"], "selection": selected, "basis": params, "metrics": {}}
            continue
        metrics, rollout = evaluate_model(test, params, selected["theta"], base["simulation"]["sigma"], statistics[name])
        row = {"status": "evaluated", "selection": selected, "basis": params,
               "theta": selected["theta"], "metrics": metrics,
               "test_weak_residual_mse": float(np.mean((test_A[name] @ np.asarray(selected["theta"]) - test_b) ** 2))}
        if rollout is not None:
            np.savez_compressed(directory / name / "rollout.npz", X_hat=rollout)
            row["rollout_sha256"] = array_hash(rollout)
        results[name] = row
    return {"results": results, "splits": splits,
            "frozen_selections_sha256": digest(directory / "frozen_selections.json"),
            "evaluation_statistics_sha256": digest(directory / "evaluation_statistics.npz")}


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
        reference, record, _ = reference_directory(Path(reference_batch), reference_status, job)
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
            content = fit_job(directory, job, base, reference)
        result.update(content, status="completed", reference_directory=str(reference.resolve()),
                      reference_report_sha256=record["report_sha256"])
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
    existing_batch, existing_frozen, existing_status, reference_batch, reference_frozen, reference_status = load_sources(spec)
    validate_protocol(spec, existing_frozen, existing_status, reference_frozen, reference_status)
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
        for label, source in (("existing20", existing_batch), ("new80_M17", reference_batch)):
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
        imported_existing, imported_m17 = ({}, {}) if smoke else import_sources(
            destination, spec, existing_batch, existing_status, reference_batch, reference_status)
        base = effective_base(reference_frozen, spec, smoke)
        frozen = {
            "requested_protocol": spec, "base": base, "smoke_only": smoke,
            "source_sha256": hashes,
            "source_provenance_sha256": {str(path.relative_to(provenance)): digest(path)
                                           for path in provenance.rglob("*") if path.is_file()},
            "imported_existing": imported_existing, "imported_new_M17": imported_m17,
            "source_existing_frozen_protocol_sha256": existing_status["frozen_protocol_sha256"],
            "source_reference_frozen_protocol_sha256": reference_status["frozen_protocol_sha256"],
            "workers": workers, "python_version": sys.version,
            "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        }
        write_json(destination / "frozen_protocol.json", frozen)
        status = {"status": "running", "stage": "new_dictionary_sizes", "smoke_only": smoke,
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
                          "new_models": spec["new_models"], "dictionary_sizes": spec["dictionary_sizes"]}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / (
        ("dictionary-size-R100-smoke-" if args.smoke else "dictionary-size-R100-") + stamp)
    run_batch(spec, destination, args.workers, args.smoke, args.resume)
    from scripts.summarize_1d_hat_dictionary_size_R100 import summarize
    summary = destination / "summary"
    if summary.exists():
        summary = destination / f"summary-{stamp}"
    summarize(destination, summary)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
