"""Verify the 100-repetition comparison of ten QANR hat-dictionary sizes."""

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
from scripts.run_1d_hat_dictionary_grid_R100 import job_plan
from scripts.run_1d_hat_sweeps import load_data
from scripts.summarize_1d_hat_dictionary_size_R100 import diagnostics, extended_statistics, METRICS
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
            if model.get("theta") is None:
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
    expected_indices = np.random.default_rng(boot["seed"]).integers(
        0, frozen["base"]["replication"]["count"],
        size=(boot["samples"], frozen["base"]["replication"]["count"]))
    np.testing.assert_array_equal(indices, expected_indices)
    result = {"status": "running", "jobs": 0, "selected_models": 0, "validation_scores": 0,
              "finite_rollouts": 0, "reused_models": 0, "failed_jobs": [], "direct_checks": [],
              "frozen_protocol_sha256": status["frozen_protocol_sha256"],
              "verification_source_sha256": digest(Path(__file__))}
    reports = {}
    expected_names = {f"hat_M{M}" for M in frozen["base"]["basis"]["sizes"]}
    for key, record in status["results"].items():
        directory = batch / record["directory"]
        report_path = directory / "report.json"
        if digest(report_path) != record["report_sha256"]:
            raise ValueError("new report hash mismatch")
        report = read_json(report_path)
        reports[key] = report
        if report["job"] != jobs[key] or report["status"] != record["status"]:
            raise ValueError("new report identity/status mismatch")
        for name, expected in report["artifacts_sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError("new artifact changed")
        result["jobs"] += 1
        if report["status"] != "completed":
            result["failed_jobs"].append(key)
            continue
        if digest(Path(report["reference_directory"]) / "report.json") != report["reference_report_sha256"]:
            raise ValueError("trajectory reference changed")
        if set(report["results"]) != expected_names:
            raise ValueError("added dictionary result missing")
        for split, manifest in report["splits"].items():
            if digest(manifest["source_data"]) != manifest["source_data_sha256"]:
                raise ValueError("source data changed")
            if digest(manifest["source_response"]) != manifest["source_response_sha256"]:
                raise ValueError("source response changed")
            if array_hash(np.load(directory / split / "response.npy")) != manifest["response_sha256"]:
                raise ValueError("saved response changed")
            for name, values in manifest["designs"].items():
                if array_hash(np.load(directory / split / f"{name}.npy")) != values["sha256"]:
                    raise ValueError("saved design changed")
        selections = read_json(directory / "frozen_selections.json")
        for name, model in report["results"].items():
            selection = selections[name]
            chosen = selection["selected"]
            A = np.load(directory / "train" / f"{name}.npy")
            b = np.load(directory / "train/response.npy")
            V = np.load(directory / "validation" / f"{name}.npy")
            c = np.load(directory / "validation/response.npy")
            path = np.load(directory / name / "path_coefficients.npy")
            close(float(np.max(np.abs(A.T @ b)) / len(b)), selection["alpha_max"], rtol=1e-12, atol=1e-30)
            for row, theta in zip(selection["path"], path):
                if row["validation_mse"] is not None:
                    close(float(np.mean((V @ theta - c) ** 2)), row["validation_mse"], rtol=1e-10, atol=1e-20)
                    result["validation_scores"] += 1
            eligible = [row for row in selection["path"] if row["converged"]]
            if min(eligible, key=lambda row: row["validation_mse"])["index"] != chosen["index"]:
                raise ValueError("selection is not validation minimum")
            np.testing.assert_array_equal(model["theta"], path[chosen["index"]])
            result["selected_models"] += 1
            if model["metrics"]["trajectory_status"] == "finite":
                with np.load(directory / name / "rollout.npz") as saved:
                    rollout = saved["X_hat"]
                if array_hash(rollout) != model["rollout_sha256"]:
                    raise ValueError("rollout hash mismatch")
                data = load_data(manifest_data := report["splits"]["test"]["source_data"])
                error = rollout - data["X"]
                close(np.sqrt(np.sum(error**2) / np.sum(data["X"]**2)), model["metrics"]["trajectory_error_rel"])
                close(np.sqrt(np.mean(error**2)), model["metrics"]["trajectory_rmse"])
                maximum = float(np.max(np.ptp(rollout[..., 0], axis=2)))
                close(maximum, model["metrics"]["trajectory_max_pair_distance"])
                if (maximum > 4.) != model["metrics"]["trajectory_exits_basis_interval"]:
                    raise ValueError("basis-domain excursion mismatch")
                result["finite_rollouts"] += 1
        if result["jobs"] % 10 == 0:
            print(result["jobs"], "ten-grid jobs verified", flush=True)
            write_json(destination / "verification.json", result)
    reused = []
    for key, record in frozen["imported_reused_models"].items():
        path = batch / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("reused compact record changed")
        row = read_json(path)
        for source in row["source_model_records"]:
            if digest(source["path"]) != source["sha256"]:
                raise ValueError("reused model source changed")
        if digest(Path(row["reference_directory"]) / "report.json") != row["reference_report_sha256"]:
            raise ValueError("reused trajectory reference changed")
        reused.append(row)
        result["reused_models"] += len(row["results"])
    count = frozen["base"]["replication"]["count"]
    reused_sizes = set(frozen["requested_protocol"]["reused_sizes"])
    for point in summary["dictionary_sizes"]:
        name = f"hat_M{point['value']}"
        models = ([row["results"][name] for row in reused] if point["value"] in reused_sizes else
                  [reports[f"run-{i}"]["results"][name] for i in range(count)])
        if diagnostics(models) != point["total100"]["diagnostics"]:
            raise ValueError("diagnostic summary mismatch")
        for metric in METRICS:
            values = [model.get("metrics", {}).get(metric) for model in models]
            if extended_statistics(values, indices) != point["total100"]["metrics"][metric]:
                raise ValueError("metric summary mismatch")
    targets = [key for key in ("run-0", f"run-{count-1}") if key in reports]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(direct_check, batch / status["results"][key]["directory"], key,
                               frozen["base"]["true_kernel"]) for key in targets]
        for future in as_completed(futures):
            result["direct_checks"].append(future.result())
    representative = summary["representative"]
    if representative["source_report"] is not None and digest(representative["source_report"]) != representative["source_report_sha256"]:
        raise ValueError("fixed representative changed")
    result.update(status="passed", summary_sha256=digest(batch / "summary/summary.json"),
                  direct_force_drift_conditions=len(result["direct_checks"]),
                  force_metrics_directly_recomputed_for_all_new_jobs=False)
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
