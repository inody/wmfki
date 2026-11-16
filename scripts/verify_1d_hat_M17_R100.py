"""Recompute and verify the saved selections, trajectories, and aggregates for the 100-repetition M=17 sweeps."""

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
from scripts.run_1d_hat_M17_R100 import digest, job_plan, read_json
from scripts.run_1d_hat_sweeps import load_data
from scripts.summarize_1d_hat_M17_R100 import extended_statistics, METRICS
from simulate import make_true_kernel


def close(actual, expected, rtol=2e-9, atol=2e-10):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def direct_check(batch, directory, key, truth_params):
    batch, directory = Path(batch), Path(directory)
    report = read_json(directory / "report.json")
    if report["status"] != "completed":
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
    for name, expected in frozen["source_provenance_sha256"].items():
        if digest(batch / "source_provenance" / name) != expected:
            raise ValueError("source provenance changed")
    jobs = job_plan(frozen["protocol"], frozen["base"])
    if jobs != status["jobs"] or set(jobs) != set(status["results"]):
        raise ValueError("planned jobs changed")
    summary = read_json(batch / "summary/summary.json")
    indices = np.load(batch / "summary/bootstrap_indices.npz")
    boot = frozen["requested_protocol"]["bootstrap"]
    existing_count, new_count = summary["existing_repetitions"], summary["new_repetitions"]
    expected_total = np.random.default_rng(boot["total_seed"]).integers(0, existing_count + new_count,
                                                                       size=(boot["samples"], existing_count + new_count))
    expected_new = np.random.default_rng(boot["new_only_seed"]).integers(0, new_count,
                                                                         size=(boot["samples"], new_count))
    expected_old = np.random.default_rng(boot["existing_only_seed"]).integers(0, existing_count,
                                                                              size=(boot["samples"], existing_count))
    np.testing.assert_array_equal(indices["total"], expected_total)
    np.testing.assert_array_equal(indices["new_only"], expected_new)
    np.testing.assert_array_equal(indices["existing_only"], expected_old)
    result = {"status": "running", "jobs": 0, "selected_models": 0, "validation_scores": 0,
              "finite_rollouts": 0, "imported_records": 0, "failed_jobs": [], "direct_checks": [],
              "frozen_protocol_sha256": status["frozen_protocol_sha256"], "verification_source_sha256": digest(Path(__file__))}
    reports = {}
    for key, record in status["results"].items():
        directory = batch / record["directory"]
        path = directory / "report.json"
        if digest(path) != record["report_sha256"]:
            raise ValueError("new report hash mismatch")
        report = read_json(path)
        reports[key] = report
        if report["job"] != jobs[key] or report["status"] != record["status"]:
            raise ValueError("new job identity/status mismatch")
        for filename, expected in report["artifacts_sha256"].items():
            if digest(directory / filename) != expected:
                raise ValueError("new artifact hash mismatch")
        result["jobs"] += 1
        if report["status"] != "completed":
            result["failed_jobs"].append(key)
            continue
        if set(report["results"]) != {"hat_M17"}:
            raise ValueError("unexpected dictionary in new result")
        model = report["results"]["hat_M17"]
        selection = model["selection"]
        chosen = selection["selected"]
        A = np.load(directory / "train/hat_M17.npy")
        b = np.load(directory / "train/response.npy")
        V = np.load(directory / "validation/hat_M17.npy")
        c = np.load(directory / "validation/response.npy")
        path_coefficients = np.load(directory / "hat_M17/path_coefficients.npy")
        close(float(np.max(np.abs(A.T @ b)) / len(b)), selection["alpha_max"], rtol=1e-12, atol=1e-30)
        for row, theta in zip(selection["path"], path_coefficients):
            if row["validation_mse"] is not None:
                close(float(np.mean((V @ theta - c) ** 2)), row["validation_mse"], rtol=1e-10, atol=1e-20)
                result["validation_scores"] += 1
        eligible = [row for row in selection["path"] if row["converged"]]
        if min(eligible, key=lambda row: row["validation_mse"])["index"] != chosen["index"]:
            raise ValueError("selected alpha is not the validation minimizer")
        np.testing.assert_array_equal(model["theta"], path_coefficients[chosen["index"]])
        result["selected_models"] += 1
        if model["metrics"]["trajectory_status"] == "finite":
            with np.load(directory / "hat_M17/rollout.npz") as saved:
                rollout = saved["X_hat"]
            if array_hash(rollout) != model["rollout_sha256"]:
                raise ValueError("rollout hash mismatch")
            data = load_data(batch / report["splits"]["test"]["data_file"])
            error = rollout - data["X"]
            close(np.sqrt(np.sum(error ** 2) / np.sum(data["X"] ** 2)), model["metrics"]["trajectory_error_rel"])
            close(np.sqrt(np.mean(error ** 2)), model["metrics"]["trajectory_rmse"])
            maximum = float(np.max(np.ptp(rollout[..., 0], axis=2)))
            close(maximum, model["metrics"]["trajectory_max_pair_distance"])
            if (maximum > 4.) != model["metrics"]["trajectory_exits_basis_interval"]:
                raise ValueError("domain excursion mismatch")
            result["finite_rollouts"] += 1
        if result["jobs"] % 100 == 0:
            print(result["jobs"], "new jobs verified", flush=True)
            write_json(destination / "verification.json", result)
    imported = {}
    for key, record in frozen["imported_existing"].items():
        path = batch / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("imported compact record changed")
        row = read_json(path)
        if digest(row["source_report"]) != row["source_report_sha256"]:
            raise ValueError("external source report changed")
        imported[key] = row["result"]
        result["imported_records"] += 1
    # Reconstruct all aggregates for the existing 20, new 80, and combined 100 repetitions from the original individual values.
    for axis, points in summary["accuracy"].items():
        for point in points:
            case_id = point["case"]
            old = [imported[f"{case_id}-{i}"] for i in range(existing_count)]
            new = [reports[f"{case_id}-{i}"].get("results", {}).get("hat_M17", {}) for i in range(new_count)]
            for metric in METRICS:
                comparisons = [
                    (new, indices["new_only"], point["new80"]["metrics"][metric]),
                    (old + new, indices["total"], point["total100"]["metrics"][metric]),
                ]
                if existing_count:
                    comparisons.insert(0, (old, indices["existing_only"], point["existing20"]["metrics"][metric]))
                for models, bootstrap, recorded in comparisons:
                    values = [model.get("metrics", {}).get(metric) for model in models]
                    if extended_statistics(values, bootstrap) != recorded:
                        raise ValueError("summary statistic mismatch")
    spec = frozen["protocol"]["sweeps"]
    targets = ["base-0", f"sigma_obs-{len(spec['sigma_obs']) - 1:02d}-0",
               f"sigma-{len(spec['sigma']) - 1:02d}-0", "N-00-0", f"N-{len(spec['N']) - 2:02d}-0",
               "dt-00-0", f"dt-{len(spec['dt']) - 1:02d}-0"]
    targets = list(dict.fromkeys(key for key in targets if key in reports))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(direct_check, batch, batch / status["results"][key]["directory"], key,
                               frozen["base"]["true_kernel"]) for key in targets]
        for future in as_completed(futures):
            result["direct_checks"].append(future.result())
    result.update(status="passed", summary_sha256=digest(batch / "summary/summary.json"),
                  force_metrics_directly_recomputed_for_all_new_jobs=False,
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
