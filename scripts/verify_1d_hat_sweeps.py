"""Independently verify the provenance, selection rules, errors, and aggregates of the saved sweeps without retraining."""

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
from features import make_trapz_weights
from scripts.prepare_1d_hat_experiment import array_hash, write_json
from scripts.run_1d_hat_lasso import force_and_drift_errors, make_evaluation_kernel
from scripts.run_1d_hat_sweeps import digest, job_plan, load_data, load_statistics, read_json, METRICS
from scripts.summarize_1d_hat_repetitions import statistics
from simulate import make_true_kernel


def close(actual, expected, rtol=2e-9, atol=2e-10):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def direct_check(batch, directory, key, frozen):
    batch, directory = Path(batch), Path(directory)
    report = read_json(directory / "report.json")
    if report["status"] != "completed":
        return dict(job=key, status="skipped_failed_job")
    pair = report["job"]["stage"] == "pairs"
    if pair:
        source = batch / report["reference_directory"] / "test/data.npz"
    elif report["job"]["stage"] == "base":
        source = directory / "test/data.npz"
    else:
        source = batch / report["splits"]["test"]["data_file"]
    data = load_data(source)
    truth, *_ = make_true_kernel(frozen["base"]["true_kernel"])
    differences = {}
    with threadpool_limits(limits=1):
        for name, model in report["results"].items():
            if model.get("theta") is None:
                continue
            measured = force_and_drift_errors(data["X"], data["t"], truth, make_evaluation_kernel(model["basis"], model["theta"]))
            differences[name] = {}
            for metric in ("force_error_weighted_rel", "drift_error_rel"):
                close(measured[metric], model["metrics"][metric])
                differences[name][metric] = abs(measured[metric] - model["metrics"][metric])
    return dict(job=key, status="passed", differences=differences)


def verify(batch, destination, workers=4):
    batch, destination = Path(batch).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    frozen, status = read_json(batch / "frozen_protocol.json"), read_json(batch / "batch_status.json")
    if status["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("batch is not complete")
    if digest(batch / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
        raise ValueError("protocol hash mismatch")
    expected = job_plan(frozen["protocol"], frozen["base"])
    if status["jobs"] != expected or set(status["results"]) != set(expected):
        raise ValueError("planned jobs changed or missing")
    for name, expected_hash in frozen["source_sha256"].items():
        if digest(batch / "source_snapshot" / name) != expected_hash:
            raise ValueError("source archive changed")
    for name, expected_hash in frozen.get("baseline_provenance_sha256", {}).items():
        if digest(batch / "baseline_provenance" / name) != expected_hash:
            raise ValueError("baseline provenance changed")
    for name in ("scripts/run_1d_hat_lasso.py", "simulate.py", "kernels.py", "features.py"):
        if digest(ROOT / name) != frozen["source_sha256"][name]:
            raise ValueError("direct evaluator differs from frozen source")
    summary = read_json(batch / "summary/summary.json")
    indices = np.load(batch / "summary/bootstrap_indices.npy")
    count = frozen["base"]["replication"]["count"]
    planned_indices = np.random.default_rng(frozen["base"]["replication"]["bootstrap_seed"]).integers(
        0, count, size=(frozen["base"]["replication"]["bootstrap_samples"], count))
    np.testing.assert_array_equal(indices, planned_indices)
    result = dict(status="running", smoke_only=frozen["smoke_only"], jobs=0, selected_models=0,
                  validation_scores=0, rollouts=0, pair_models=0, timing_measurements=0,
                  failed_jobs=[], direct_checks=[], source_sha256=digest(Path(__file__)),
                  frozen_protocol_sha256=status["frozen_protocol_sha256"])
    data_cache, checked_statistics = {}, set()
    reports = {}
    for key, record in status["results"].items():
        directory = batch / record["directory"]
        if digest(directory / "report.json") != record["report_sha256"]:
            raise ValueError("report hash mismatch")
        report = read_json(directory / "report.json")
        reports[key] = report
        if report["job"] != expected[key]:
            raise ValueError("job identity mismatch")
        for name, expected_hash in report["artifacts_sha256"].items():
            if digest(directory / name) != expected_hash:
                raise ValueError(f"artifact hash mismatch: {directory / name}")
        result["jobs"] += 1
        if report["status"] != "completed":
            result["failed_jobs"].append(key)
            continue
        stage = report["job"]["stage"]
        if stage == "runtime":
            result["timing_measurements"] += len(report["timings"])
            if report["numerical_threads"] != 1 or report["competing_sweep_workers"] != 0:
                raise ValueError("runtime concurrency mismatch")
            continue
        if stage == "pairs":
            reference = batch / report["reference_directory"]
            selections = read_json(reference / "frozen_selections.json")
            source, stats_path = reference / "test/data.npz", reference / "evaluation_statistics.npz"
        else:
            selections = read_json(directory / "frozen_selections.json")
            source = directory / "test/data.npz" if stage == "base" else batch / report["splits"]["test"]["data_file"]
            stats_path = directory / "evaluation_statistics.npz" if stage == "base" else batch / report["statistics_source"]
        if source not in data_cache:
            data_cache[source] = load_data(source)
        data = data_cache[source]
        stats = load_statistics(stats_path)
        if stats_path not in checked_statistics:
            weights = make_trapz_weights(data["t"])
            centered = data["X"][..., 0] - data["X"][..., 0].mean(axis=2, keepdims=True)
            expected_drift_energy = float(np.sum(centered ** 2 * weights[None, :, None]))
            expected_force_energy = 2 * data["X"].shape[2] * expected_drift_energy
            for values in stats.values():
                close(values["force_diag"].sum() + 2 * values["force_off"].sum(), expected_force_energy)
                close(values["drift_gram"].sum(), expected_drift_energy)
            checked_statistics.add(stats_path)
        for name, model in report["results"].items():
            if model.get("theta") is None:
                continue
            theta = np.asarray(model["theta"])
            path = np.load(directory / name / "path_coefficients.npy")
            if stage != "pairs":
                selection = selections[name]
                chosen = selection["selected"]
                A = np.load(directory / "train" / f"{name}.npy")
                b = np.load(directory / "train/response.npy")
                V = np.load(directory / "validation" / f"{name}.npy")
                c = np.load(directory / "validation/response.npy")
                alpha_max = float(np.max(np.abs(A.T @ b)) / len(b))
                close(alpha_max, selection["alpha_max"], rtol=1e-12, atol=1e-30)
                for row, coefficient in zip(selection["path"], path):
                    if row["validation_mse"] is not None:
                        score = float(np.mean((V @ coefficient - c) ** 2))
                        close(score, row["validation_mse"], rtol=1e-10, atol=1e-20)
                        result["validation_scores"] += 1
                eligible = [row for row in selection["path"] if row["converged"]]
                if min(eligible, key=lambda row: row["validation_mse"])["index"] != chosen["index"]:
                    raise ValueError("selection is not the converged validation minimizer")
                np.testing.assert_array_equal(theta, path[chosen["index"]])
                result["selected_models"] += 1
            else:
                endpoint = model["endpoint"]
                reference_selection = selections[name]
                if endpoint["index"] != reference_selection["selected"]["index"]:
                    raise ValueError("pair alpha endpoint changed")
                if [row["alpha_sk"] for row in model["path"]] != [row["alpha_sk"] for row in reference_selection["path"][:len(path)]]:
                    raise ValueError("pair absolute penalty path changed")
                np.testing.assert_array_equal(theta, path[-1])
                difference = theta - reference_selection["theta"]
                close(float(difference @ difference), model["metrics"]["coefficient_difference_sq"])
                result["pair_models"] += 1
            if model["metrics"]["trajectory_status"] == "finite":
                with np.load(directory / name / "rollout.npz") as saved:
                    rollout = saved["X_hat"]
                if array_hash(rollout) != model["rollout_sha256"]:
                    raise ValueError("rollout array hash mismatch")
                diff = rollout - data["X"]
                close(np.sqrt(np.sum(diff ** 2) / np.sum(data["X"] ** 2)), model["metrics"]["trajectory_error_rel"])
                close(np.sqrt(np.mean(diff ** 2)), model["metrics"]["trajectory_rmse"])
                maximum = float(np.max(np.ptp(rollout[..., 0], axis=2)))
                close(maximum, model["metrics"]["trajectory_max_pair_distance"])
                if (maximum > 4.) != model["metrics"]["trajectory_exits_basis_interval"]:
                    raise ValueError("domain excursion mismatch")
                result["rollouts"] += 1
        if result["jobs"] % 20 == 0:
            print(result["jobs"], "jobs verified", flush=True)
            write_json(destination / "verification.json", result)
    # Reconstruct the aggregates from the original jobs, not only from the saved per-repetition values.
    for axis, points in summary["accuracy"].items():
        for point in points:
            for name, model in point["dictionaries"].items():
                for metric, recorded in model["metrics"].items():
                    values = [reports[f"{point['case']}-{i}"].get("results", {}).get(name, {}).get("metrics", {}).get(metric) for i in range(count)]
                    if statistics(values, indices) != recorded:
                        raise ValueError("accuracy summary mismatch")
    for pi, point in enumerate(summary["pairs"]):
        for name, model in point["dictionaries"].items():
            for metric, recorded in model["metrics"].items():
                values = [reports[f"pairs-{pi}-{i}"].get("results", {}).get(name, {}).get("metrics", {}).get(metric) for i in range(count)]
                if statistics(values, indices) != recorded:
                    raise ValueError("pair summary mismatch")
    for ni, point in enumerate(summary["runtime"]):
        for name, modes in point["dictionaries"].items():
            for mode, recorded in modes.items():
                values = [next((row["seconds"] for row in reports[f"runtime-{ni}-{i}"].get("timings", []) if row["name"] == name and row["mode"] == mode), None)
                          for i in range(frozen["protocol"]["runtime_repetitions"])]
                if values != recorded["values"]:
                    raise ValueError("runtime values mismatch")
                if all(value is not None for value in values):
                    close(np.median(values), recorded["median"])
                    close(np.quantile(values, [.25, .75]), recorded["iqr"])
    spec = frozen["protocol"]
    targets = [f"sigma_obs-{len(spec['sweeps']['sigma_obs']) - 1:02d}-0", f"sigma-{len(spec['sweeps']['sigma']) - 1:02d}-0",
               "N-00-0", "dt-00-0", f"dt-{len(spec['sweeps']['dt']) - 1:02d}-0", "pairs-0-0", f"pairs-{len(spec['pair_counts']) - 1}-0"]
    targets = list(dict.fromkeys(key for key in targets if key in reports))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(direct_check, batch, batch / status["results"][key]["directory"], key, frozen) for key in targets]
        for future in as_completed(futures):
            result["direct_checks"].append(future.result())
    result.update(status="passed", evaluated_statistics_checked=len(checked_statistics),
                  force_metrics_directly_recomputed_for_all_jobs=False,
                  all_baseline_force_drift_rollouts_recomputed_at_import=not frozen["smoke_only"],
                  summary_sha256=digest(batch / "summary/summary.json"))
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
