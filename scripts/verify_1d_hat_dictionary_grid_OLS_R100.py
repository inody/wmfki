"""Verify the 100-repetition OLS results for ten QANR hat-dictionary sizes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import shutil
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_1d_hat_experiment import array_hash, write_json
from scripts.run_1d_hat_lasso import force_and_drift_errors, make_evaluation_kernel
from scripts.run_1d_hat_dictionary_size_R100 import digest, read_json
from scripts.run_1d_hat_dictionary_grid_OLS_R100 import fit_ols, job_plan
from scripts.run_1d_hat_sweeps import load_data
from scripts.summarize_1d_hat_dictionary_grid_OLS_R100 import diagnostics, extended_statistics, FIT_METRICS, METRICS
from simulate import make_true_kernel


def close(actual, expected, rtol=2e-9, atol=2e-10):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def direct_check(directory, key, truth_params):
    directory = Path(directory)
    report = read_json(directory / "report.json")
    if report["status"] != "completed":
        return {"job": key, "status": "skipped"}
    data = load_data(report["splits"]["test"]["source_data"])
    truth, *_ = make_true_kernel(truth_params)
    differences = {}
    with threadpool_limits(limits=1):
        for name, model in report["results"].items():
            if model["status"] != "evaluated":
                continue
            measured = force_and_drift_errors(data["X"], data["t"], truth,
                                              make_evaluation_kernel(model["basis"], model["theta"]))
            differences[name] = {}
            for metric in ("force_error_weighted_rel", "drift_error_rel"):
                close(measured[metric], model["metrics"][metric])
                differences[name][metric] = abs(measured[metric] - model["metrics"][metric])
    return {"job": key, "status": "passed", "differences": differences}


def verify(batch, destination, workers=3):
    batch, destination = Path(batch).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    frozen = read_json(batch / "frozen_protocol.json")
    status = read_json(batch / "batch_status.json")
    if status["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("batch incomplete")
    if digest(batch / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for name, expected in frozen["source_sha256"].items():
        if digest(batch / "source_snapshot" / name) != expected:
            raise ValueError("source snapshot changed")
    for name, expected in frozen["source_provenance_sha256"].items():
        if digest(batch / "source_provenance" / name) != expected:
            raise ValueError("source provenance changed")
    jobs = job_plan(frozen["requested_protocol"], frozen["smoke_only"])
    if status["jobs"] != jobs or set(status["results"]) != set(jobs):
        raise ValueError("jobs missing or changed")
    summary = read_json(batch / "summary/summary.json")
    with np.load(batch / "summary/bootstrap_indices.npz") as saved:
        indices = saved["total"]
    boot = frozen["requested_protocol"]["bootstrap"]
    count = frozen["base"]["replication"]["count"]
    expected_indices = np.random.default_rng(boot["seed"]).integers(0, count, size=(boot["samples"], count))
    np.testing.assert_array_equal(indices, expected_indices)
    result = {"status": "running", "jobs": 0, "models": 0, "finite_fits": 0, "rank_deficient": 0,
              "finite_rollouts": 0, "failed_jobs": [], "direct_checks": [],
              "frozen_protocol_sha256": status["frozen_protocol_sha256"],
              "verification_source_sha256": digest(Path(__file__))}
    reports = {}
    expected_names = {f"hat_M{M}" for M in frozen["base"]["basis"]["sizes"]}
    for key, record in status["results"].items():
        directory = batch / record["directory"]
        report_path = directory / "report.json"
        if digest(report_path) != record["report_sha256"]:
            raise ValueError("report hash mismatch")
        report = read_json(report_path)
        reports[key] = report
        if report["job"] != jobs[key] or report["status"] != record["status"]:
            raise ValueError("report identity/status mismatch")
        for name, expected in report["artifacts_sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError("artifact changed")
        result["jobs"] += 1
        if report["status"] != "completed":
            result["failed_jobs"].append(key)
            continue
        if digest(Path(report["reference_directory"]) / "report.json") != report["reference_report_sha256"]:
            raise ValueError("trajectory reference changed")
        if set(report["results"]) != expected_names:
            raise ValueError("model missing")
        frozen_fits = read_json(directory / "frozen_ols_fits.json")
        for split, manifest in report["splits"].items():
            if digest(manifest["source_data"]) != manifest["source_data_sha256"]:
                raise ValueError("source data changed")
            if digest(manifest["source_response"]) != manifest["source_response_sha256"]:
                raise ValueError("source response changed")
            if array_hash(np.load(directory / split / "response.npy")) != manifest["response_sha256"]:
                raise ValueError("response changed")
            for name, values in manifest["designs"].items():
                if array_hash(np.load(directory / split / f"{name}.npy")) != values["sha256"]:
                    raise ValueError("design changed")
        for name, model in report["results"].items():
            result["models"] += 1
            A = np.load(directory / "train" / f"{name}.npy")
            b = np.load(directory / "train/response.npy")
            with threadpool_limits(limits=1):
                theta, fit = fit_ols(A, b)
            np.testing.assert_array_equal(theta, model["theta"])
            np.testing.assert_array_equal(theta, frozen_fits[name]["theta"])
            for field in ("status", "rank", "columns", "full_column_rank", "rcond", "nonzero_exact", "nonzero_reported"):
                if fit[field] != model["fit"][field] or fit[field] != frozen_fits[name]["fit"][field]:
                    raise ValueError("OLS fit diagnostic mismatch")
            close(fit["singular_values"], model["fit"]["singular_values"], rtol=1e-12, atol=1e-30)
            close(fit["train_mse"], model["fit"]["train_mse"], rtol=1e-12, atol=1e-30)
            if model["fit"]["status"] != "fitted":
                continue
            result["finite_fits"] += 1
            if not model["fit"]["full_column_rank"]:
                result["rank_deficient"] += 1
            V = np.load(directory / "validation" / f"{name}.npy")
            c = np.load(directory / "validation/response.npy")
            T = np.load(directory / "test" / f"{name}.npy")
            d = np.load(directory / "test/response.npy")
            close(float(np.mean((V @ theta - c) ** 2)), model["fit"]["validation_mse"], rtol=1e-10, atol=1e-20)
            close(float(np.mean((T @ theta - d) ** 2)), model["fit"]["test_mse"], rtol=1e-10, atol=1e-20)
            if model["metrics"]["trajectory_status"] == "finite":
                with np.load(directory / name / "rollout.npz") as saved:
                    rollout = saved["X_hat"]
                if array_hash(rollout) != model["rollout_sha256"]:
                    raise ValueError("rollout changed")
                data = load_data(report["splits"]["test"]["source_data"])
                error = rollout - data["X"]
                close(np.sqrt(np.sum(error**2) / np.sum(data["X"]**2)), model["metrics"]["trajectory_error_rel"])
                close(np.sqrt(np.mean(error**2)), model["metrics"]["trajectory_rmse"])
                maximum = float(np.max(np.ptp(rollout[..., 0], axis=2)))
                close(maximum, model["metrics"]["trajectory_max_pair_distance"])
                result["finite_rollouts"] += 1
        if result["jobs"] % 10 == 0:
            print(result["jobs"], "OLS jobs verified", flush=True)
            write_json(destination / "verification.json", result)
    for point in summary["dictionary_sizes"]:
        name = f"hat_M{point['value']}"
        models = [reports[f"run-{i}"]["results"][name] for i in range(count)]
        if diagnostics(models) != point["total100"]["diagnostics"]:
            raise ValueError("diagnostic summary mismatch")
        for metric in METRICS:
            values = [model.get("metrics", {}).get(metric) for model in models]
            if extended_statistics(values, indices) != point["total100"]["metrics"][metric]:
                raise ValueError("metric summary mismatch")
        for metric in FIT_METRICS:
            values = [model.get("fit", {}).get(metric) for model in models]
            if extended_statistics(values, indices) != point["total100"]["fit_metrics"][metric]:
                raise ValueError("fit summary mismatch")
    targets = [key for key in ("run-0", f"run-{count-1}") if key in reports]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(direct_check, batch / status["results"][key]["directory"], key,
                               frozen["base"]["true_kernel"]) for key in targets]
        for future in as_completed(futures):
            result["direct_checks"].append(future.result())
    result.update(status="passed", summary_sha256=digest(batch / "summary/summary.json"),
                  direct_force_drift_conditions=len(result["direct_checks"]),
                  force_metrics_directly_recomputed_for_all_jobs=False)
    shutil.copy2(Path(__file__), destination / "verification_script.py")
    write_json(destination / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(verify(args.batch, args.out_dir or args.batch / "verification", args.workers))
