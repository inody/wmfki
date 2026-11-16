"""Verify the saved results and aggregates for the 100-repetition M=17 pair-count sweep."""

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
from scripts.run_1d_hat_M17_pair_R100 import digest, job_plan, read_json
from scripts.run_1d_hat_sweeps import force_energy, load_data, load_statistics
from scripts.summarize_1d_hat_M17_pair_R100 import extended_statistics, METRICS
from simulate import make_true_kernel


def close(actual, expected, rtol=2e-9, atol=2e-10):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def direct_check(directory, key, truth_params):
    directory = Path(directory)
    report = read_json(directory / "report.json")
    if report["status"] != "completed":
        return {"job": key, "status": "skipped"}
    reference = Path(report["reference_directory"])
    data = load_data(reference / "test/data.npz")
    model = report["results"]["hat_M17"]
    if model.get("theta") is None:
        return {"job": key, "status": "skipped_nonfinite"}
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
    frozen = read_json(batch / "frozen_protocol.json")
    status = read_json(batch / "batch_status.json")
    if status["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("pair batch is incomplete")
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
        raise ValueError("planned jobs changed or are missing")
    summary = read_json(batch / "summary/summary.json")
    indices = np.load(batch / "summary/bootstrap_indices.npz")
    boot = frozen["requested_protocol"]["bootstrap"]
    old_count, new_count = summary["existing_repetitions"], summary["new_repetitions"]
    expected_total = np.random.default_rng(boot["total_seed"]).integers(
        0, old_count + new_count, size=(boot["samples"], old_count + new_count))
    expected_new = np.random.default_rng(boot["new_only_seed"]).integers(
        0, new_count, size=(boot["samples"], new_count))
    expected_old = (np.random.default_rng(boot["existing_only_seed"]).integers(
        0, old_count, size=(boot["samples"], old_count)) if old_count else np.empty((boot["samples"], 0), dtype=int))
    np.testing.assert_array_equal(indices["total"], expected_total)
    np.testing.assert_array_equal(indices["new_only"], expected_new)
    np.testing.assert_array_equal(indices["existing_only"], expected_old)
    result = {
        "status": "running", "jobs": 0, "pair_models": 0, "candidate_solves": 0,
        "finite_rollouts": 0, "imported_records": 0, "failed_jobs": [], "direct_checks": [],
        "frozen_protocol_sha256": status["frozen_protocol_sha256"],
        "verification_source_sha256": digest(Path(__file__)),
    }
    reports = {}
    for key, record in status["results"].items():
        directory = batch / record["directory"]
        report_path = directory / "report.json"
        if digest(report_path) != record["report_sha256"]:
            raise ValueError("new report hash mismatch")
        report = read_json(report_path)
        reports[key] = report
        if report["job"] != jobs[key] or report["status"] != record["status"]:
            raise ValueError("new job identity/status mismatch")
        for name, expected in report["artifacts_sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError("new artifact hash mismatch")
        result["jobs"] += 1
        if report["status"] != "completed":
            result["failed_jobs"].append(key)
            continue
        expected_seed = 2_000_000 + 1000 * report["job"]["global_index"] + 10 * report["job"]["pair_index"]
        if report["pair_seed"] != expected_seed or report["P"] != report["job"]["P"]:
            raise ValueError("pair seed or P changed")
        reference = Path(report["reference_directory"])
        if digest(reference / "report.json") != report["reference_report_sha256"]:
            raise ValueError("reference report changed")
        if digest(reference / "frozen_selections.json") != report["reference_selections_sha256"]:
            raise ValueError("reference selections changed")
        if digest(reference / "evaluation_statistics.npz") != report["reference_statistics_sha256"]:
            raise ValueError("reference statistics changed")
        selections = read_json(reference / "frozen_selections.json")
        statistics = load_statistics(reference / "evaluation_statistics.npz")
        if set(report["results"]) != {"hat_M17"}:
            raise ValueError("unexpected dictionary")
        model = report["results"]["hat_M17"]
        path = np.load(directory / "hat_M17/path_coefficients.npy")
        endpoint = model["endpoint"]
        selection = selections["hat_M17"]
        if endpoint["index"] != selection["selected"]["index"]:
            raise ValueError("fixed alpha endpoint changed")
        expected_alphas = [row["alpha_sk"] for row in selection["path"][:len(path)]]
        if [row["alpha_sk"] for row in model["path"]] != expected_alphas:
            raise ValueError("absolute alpha path changed")
        result["candidate_solves"] += len(model["path"])
        if model.get("theta") is None:
            continue
        theta = np.asarray(model["theta"])
        np.testing.assert_array_equal(theta, path[-1])
        full_theta = np.asarray(selection["theta"])
        difference = theta - full_theta
        close(float(difference @ difference), model["metrics"]["coefficient_difference_sq"])
        sampled_A = np.load(directory / "hat_M17/design.npy")
        full_A = np.load(reference / "train/hat_M17.npy")
        close(float(np.sum((sampled_A - full_A) ** 2)), model["metrics"]["design_difference_sq"])
        denominator = force_energy(full_theta, statistics["hat_M17"])
        close(force_energy(difference, statistics["hat_M17"]) / denominator,
              model["metrics"]["force_difference_sq_rel"])
        result["pair_models"] += 1
        if model["metrics"]["trajectory_status"] == "finite":
            with np.load(directory / "hat_M17/rollout.npz") as saved:
                rollout = saved["X_hat"]
            if array_hash(rollout) != model["rollout_sha256"]:
                raise ValueError("rollout hash mismatch")
            data = load_data(reference / "test/data.npz")
            error = rollout - data["X"]
            close(np.sqrt(np.sum(error ** 2) / np.sum(data["X"] ** 2)), model["metrics"]["trajectory_error_rel"])
            close(np.sqrt(np.mean(error ** 2)), model["metrics"]["trajectory_rmse"])
            maximum = float(np.max(np.ptp(rollout[..., 0], axis=2)))
            close(maximum, model["metrics"]["trajectory_max_pair_distance"])
            if (maximum > 4.) != model["metrics"]["trajectory_exits_basis_interval"]:
                raise ValueError("domain excursion mismatch")
            result["finite_rollouts"] += 1
        if result["jobs"] % 50 == 0:
            print(result["jobs"], "new pair jobs verified", flush=True)
            write_json(destination / "verification.json", result)
    imported = {}
    for key, record in frozen["imported_existing"].items():
        path = batch / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("imported record changed")
        row = read_json(path)
        if digest(row["source_report"]) != row["source_report_sha256"]:
            raise ValueError("external existing report changed")
        imported[key] = row["result"]
        result["imported_records"] += 1
    for pair_index, point in enumerate(summary["pairs"]):
        old = [imported[f"pairs-{pair_index}-{index}"] for index in range(old_count)]
        new = [reports[f"pairs-{pair_index}-{index}"].get("results", {}).get("hat_M17", {})
               for index in range(new_count)]
        for metric in METRICS:
            comparisons = [
                (new, indices["new_only"], point["new80"]["metrics"][metric]),
                (old + new, indices["total"], point["total100"]["metrics"][metric]),
            ]
            if old_count:
                comparisons.insert(0, (old, indices["existing_only"], point["existing20"]["metrics"][metric]))
            for models, bootstrap, recorded in comparisons:
                values = [model.get("metrics", {}).get(metric) for model in models]
                if extended_statistics(values, bootstrap) != recorded:
                    raise ValueError("summary statistic mismatch")
    targets = [f"pairs-0-0", f"pairs-{len(summary['pairs']) - 1}-0"]
    targets = [key for key in targets if key in reports]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(direct_check, batch / status["results"][key]["directory"], key,
                               frozen["base"]["true_kernel"]) for key in targets]
        for future in as_completed(futures):
            result["direct_checks"].append(future.result())
    result.update(
        status="passed", summary_sha256=digest(batch / "summary/summary.json"),
        force_metrics_directly_recomputed_for_all_new_jobs=False,
        direct_force_drift_conditions=len(result["direct_checks"]),
    )
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
