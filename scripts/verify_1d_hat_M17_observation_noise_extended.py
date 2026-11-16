"""Retrospectively verify the selections, trajectories, and aggregates for the extended observation-noise sweep."""

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
from scripts.run_1d_hat_M17_observation_noise_extended import digest, job_plan, read_json
from scripts.run_1d_hat_sweeps import load_data
from scripts.summarize_1d_hat_M17_R100 import group_summary
from scripts.summarize_1d_hat_M17_observation_noise_extended import METRICS, source_models
from simulate import make_true_kernel


def close(actual, expected, rtol=2e-9, atol=2e-10):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def direct_check(batch, directory, key, truth_params):
    batch, directory = Path(batch), Path(directory)
    report = read_json(directory / "report.json")
    if report["status"] != "completed" or report["results"]["hat_M17"]["status"] != "evaluated":
        return {"job": key, "status": "skipped"}
    data = load_data(batch / report["splits"]["test"]["data_file"])
    model = report["results"]["hat_M17"]
    truth, *_ = make_true_kernel(truth_params)
    with threadpool_limits(limits=1):
        measured = force_and_drift_errors(data["X"], data["t"], truth,
                                          make_evaluation_kernel(model["basis"], model["theta"]))
    differences = {}
    for metric in ("force_error_weighted_rel", "drift_error_rel"):
        close(measured[metric], model["metrics"][metric])
        differences[metric] = abs(measured[metric] - model["metrics"][metric])
    return {"job": key, "status": "passed", "differences": differences}


def verify(batch, destination, workers=4):
    batch, destination = Path(batch).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    frozen, status = read_json(batch / "frozen_protocol.json"), read_json(batch / "batch_status.json")
    if status["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("batch is incomplete")
    if digest(batch / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for name, expected in frozen["source_sha256"].items():
        if digest(batch / "source_snapshot" / name) != expected:
            raise ValueError("source snapshot changed")
    for record in frozen["baseline_sources"].values():
        if digest(record["source_report"]) != record["source_report_sha256"]:
            raise ValueError("baseline source report changed")
    planned = job_plan(frozen["requested_protocol"], frozen["base"])
    if planned != status["jobs"] or set(planned) != set(status["results"]):
        raise ValueError("planned jobs changed")
    result = {"status": "running", "jobs": 0, "selected_models": 0, "unselected_models": 0,
              "validation_scores": 0, "finite_rollouts": 0, "failed_jobs": [], "direct_checks": [],
              "frozen_protocol_sha256": status["frozen_protocol_sha256"], "verification_source_sha256": digest(Path(__file__))}
    reports = {}
    for key, record in status["results"].items():
        directory, path = batch / record["directory"], batch / record["directory"] / "report.json"
        if digest(path) != record["report_sha256"]:
            raise ValueError("report hash mismatch")
        report = read_json(path)
        reports[key] = report
        if report["job"] != planned[key] or report["status"] != record["status"]:
            raise ValueError("job identity/status mismatch")
        for filename, expected in report["artifacts_sha256"].items():
            if digest(directory / filename) != expected:
                raise ValueError("artifact hash mismatch")
        result["jobs"] += 1
        if report["status"] != "completed":
            result["failed_jobs"].append(key)
            continue
        model = report["results"]["hat_M17"]
        selection = model["selection"]
        A, b = np.load(directory / "train/hat_M17.npy"), np.load(directory / "train/response.npy")
        V, c = np.load(directory / "validation/hat_M17.npy"), np.load(directory / "validation/response.npy")
        path_coefficients = np.load(directory / "hat_M17/path_coefficients.npy")
        close(float(np.max(np.abs(A.T @ b)) / len(b)), selection["alpha_max"], rtol=1e-12, atol=1e-30)
        for row, theta in zip(selection["path"], path_coefficients):
            if row["validation_mse"] is not None:
                close(float(np.mean((V @ theta - c) ** 2)), row["validation_mse"], rtol=1e-10, atol=1e-20)
                result["validation_scores"] += 1
        eligible = [row for row in selection["path"] if row["converged"]]
        if selection["status"] == "selected":
            chosen = selection["selected"]
            if min(eligible, key=lambda row: row["validation_mse"])["index"] != chosen["index"]:
                raise ValueError("selected alpha is not validation minimizer")
            np.testing.assert_array_equal(model["theta"], path_coefficients[chosen["index"]])
            result["selected_models"] += 1
        else:
            if eligible:
                raise ValueError("unselected model has eligible candidate")
            result["unselected_models"] += 1
            continue
        if model["metrics"]["trajectory_status"] == "finite":
            with np.load(directory / "hat_M17/rollout.npz") as saved:
                rollout = saved["X_hat"]
            if array_hash(rollout) != model["rollout_sha256"]:
                raise ValueError("rollout hash mismatch")
            data = load_data(batch / report["splits"]["test"]["data_file"])
            error = rollout - data["X"]
            close(np.sqrt(np.sum(error ** 2) / np.sum(data["X"] ** 2)), model["metrics"]["trajectory_error_rel"])
            close(np.sqrt(np.mean(error ** 2)), model["metrics"]["trajectory_rmse"])
            result["finite_rollouts"] += 1
        if result["jobs"] % 100 == 0:
            print(result["jobs"], "jobs verified", flush=True)
            write_json(destination / "verification.json", result)
    summary = read_json(batch / "summary/summary.json")
    indices = np.load(batch / "summary/bootstrap_indices.npy")
    expected_indices = np.random.default_rng(frozen["requested_protocol"]["bootstrap"]["seed"]).integers(
        0, 100, size=(frozen["requested_protocol"]["bootstrap"]["samples"], 100))
    np.testing.assert_array_equal(indices, expected_indices)
    total_batch = ROOT / frozen["requested_protocol"]["source_total100_batch"]
    for point in summary["observation_noise"]:
        case_id = point["case"]
        if point["source"] == "completed_M17_R100":
            models, _ = source_models(total_batch, case_id)
        else:
            models = [reports[f"{case_id}-{index}"].get("results", {}).get("hat_M17", {}) for index in range(100)]
        reconstructed = group_summary(models, indices)
        if reconstructed != point["total100"]:
            raise ValueError("summary reconstruction mismatch")
    targets = [f"sigma_obs-{index:02d}-0" for index in range(7, 11)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(direct_check, batch, batch / status["results"][key]["directory"], key,
                               frozen["base"]["true_kernel"]) for key in targets]
        for future in as_completed(futures):
            result["direct_checks"].append(future.result())
    result.update(status="passed", summary_sha256=digest(batch / "summary/summary.json"),
                  direct_force_drift_conditions=len(result["direct_checks"]))
    shutil.copy2(Path(__file__), destination / "verification_script.py")
    write_json(destination / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(verify(args.batch, args.out_dir or args.batch / "verification", args.workers))
