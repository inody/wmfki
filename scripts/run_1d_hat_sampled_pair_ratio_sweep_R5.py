"""Explore the minimum sampled-pair ratio that reproduces the generic 1D full-pair result.

This R=5 preview reuses the first five frozen train/validation/test trajectory triplets,
their matched full-pair LASSO and raw OLS records, and their held-out evaluation
statistics.  Only the training and validation weak designs are reassembled from
independent sampled ordered pairs.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hat_sweep_numerics import assemble_sampled, evaluate_model
from scripts.prepare_1d_hat_experiment import basis_params, write_json
from scripts.run_1d_hat_dictionary_size_R100 import load_data
from scripts.run_1d_hat_lasso import select_lasso
from scripts.run_1d_hat_sweeps import load_statistics, test_family

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat_sampled_pair_ratio_sweep_R5.json"
SOURCES = [
    "features.py", "kernels.py", "simulate.py", "utils.py",
    "scripts/prepare_1d_hat_experiment.py", "scripts/run_1d_hat_lasso.py",
    "scripts/hat_sweep_numerics.py", "scripts/run_1d_hat_sweeps.py",
    "scripts/run_1d_hat_dictionary_size_R100.py",
    "scripts/run_1d_hat_sampled_pair_ratio_sweep_R5.py",
    "configs/experiments/one_dimensional_hat_sampled_pair_ratio_sweep_R5.json",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def candidates(base):
    return [(f"hat_M{size}", basis_params(base, size)) for size in base["basis"]["sizes"]]


def validate_spec(spec):
    if spec.get("task") != "one_dimensional_hat_sampled_pair_ratio_sweep_R5":
        raise ValueError("wrong task")
    if spec.get("protocol_version") != 1:
        raise ValueError("protocol version changed")
    if spec.get("dictionary_sizes") != [9, 11, 14, 17, 22, 27, 33, 42, 52, 65]:
        raise ValueError("dictionary grid changed")
    if spec.get("repetitions") != 5 or spec.get("reference_indices") != list(range(5)):
        raise ValueError("the preview must use the first five matched frozen repetitions")
    if spec.get("pair_ratios") != [20, 30, 50, 75, 100]:
        raise ValueError("pair-ratio grid changed")
    expected_sampling = {
        "ordered": True,
        "exclude_diagonal": True,
        "with_replacement": True,
        "resample_each_time": True,
        "shared_across_tests_and_atoms": True,
        "independent_training_validation": True,
        "independent_across_ratios": True,
    }
    if spec.get("pair_sampling") != expected_sampling:
        raise ValueError("pair-sampling policy changed")
    acceptance = spec.get("acceptance", {})
    if (acceptance.get("primary_metric") != "force_error_weighted_rel"
            or acceptance.get("maximum_sampled_to_full_rms_ratio") != 1.25
            or acceptance.get("dictionary_trend", {}).get("minimum_spearman_correlation") != 0.6):
        raise ValueError("preview acceptance rule changed")
    offsets = [spec["training_pair_seed_offset"], spec["validation_pair_seed_offset"]]
    if len(set(offsets)) != 2:
        raise ValueError("training and validation pair seeds must differ")


def load_sources(spec):
    lasso = ROOT / spec["source_lasso_batch"]
    ols = ROOT / spec["source_ols_batch"]
    return {
        "lasso": lasso,
        "lasso_frozen": read_json(lasso / "frozen_protocol.json"),
        "lasso_status": read_json(lasso / "batch_status.json"),
        "lasso_verification": read_json(lasso / "verification/verification.json"),
        "ols": ols,
        "ols_frozen": read_json(ols / "frozen_protocol.json"),
        "ols_status": read_json(ols / "batch_status.json"),
        "ols_verification": read_json(ols / "verification/verification.json"),
    }


def validate_sources(spec, source):
    for method in ("lasso", "ols"):
        batch = source[method]
        frozen_hash = digest(batch / "frozen_protocol.json")
        if frozen_hash != spec[f"source_{method}_frozen_protocol_sha256"]:
            raise ValueError(f"{method} source frozen protocol hash mismatch")
        if source[f"{method}_status"]["status"] != "completed":
            raise ValueError(f"{method} source batch is incomplete")
        if source[f"{method}_verification"]["status"] != "passed":
            raise ValueError(f"{method} source verification did not pass")
        if source[f"{method}_status"]["frozen_protocol_sha256"] != frozen_hash:
            raise ValueError(f"{method} source status points to another protocol")
    lf, of = source["lasso_frozen"], source["ols_frozen"]
    if lf["requested_protocol"]["dictionary_sizes"] != spec["dictionary_sizes"]:
        raise ValueError("LASSO source dictionary grid differs")
    if of["requested_protocol"]["dictionary_sizes"] != spec["dictionary_sizes"]:
        raise ValueError("OLS source dictionary grid differs")
    left, right = deepcopy(lf["base"]), deepcopy(of["base"])
    left["basis"]["sizes"] = spec["dictionary_sizes"]
    if left != right:
        raise ValueError("matched LASSO and OLS scientific protocols differ")
    for index in spec["reference_indices"]:
        key = f"run-{index}"
        if lf["references"][key] != of["references"][key]:
            raise ValueError("LASSO and OLS do not use the same frozen trajectory")
    base = left
    if (base["simulation"] != {"N": 1000, "horizon": 2.0, "dt": 0.005,
                               "sigma": 0.1, "sigma_obs": 0.0}
            or base["pair_mode"] != "full" or base["true_kernel"]["mode"] != "qanr"):
        raise ValueError("unexpected authoritative 1D protocol")
    return base


def verify_artifacts(directory, report_hash):
    directory = Path(directory)
    report_path = directory / "report.json"
    if digest(report_path) != report_hash:
        raise ValueError(f"reference report changed: {report_path}")
    report = read_json(report_path)
    for name, expected in report["artifacts_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"reference artifact changed: {directory / name}")
    return report


def compact_lasso_record(result):
    selection = result["selection"]
    chosen = selection.get("selected", {})
    return {
        "status": result["status"],
        "metrics": result.get("metrics", {}),
        "selection_status": selection.get("status"),
        "selected_alpha": chosen.get("alpha_sk"),
        "selected_lambda": chosen.get("lambda_summed_loss"),
        "selected_path_index": chosen.get("index"),
        "selected_at_grid_boundary": selection.get("selected_at_grid_boundary"),
        "selected_converged": chosen.get("converged"),
    }


def compact_ols_record(result):
    fit = result["fit"]
    singular = fit.get("singular_values", [])
    rank = fit.get("rank", 0)
    effective = (singular[0] / singular[rank - 1]
                 if rank and len(singular) >= rank and singular[rank - 1] > 0 else None)
    return {
        "status": result["status"],
        "metrics": result.get("metrics", {}),
        "rank": rank,
        "columns": fit.get("columns"),
        "full_column_rank": fit.get("full_column_rank"),
        "condition_number": fit.get("condition_number"),
        "effective_condition_number": effective,
    }


def build_matched_sources(spec, source, destination):
    records, bundles = [], {}
    lasso_frozen, lasso_status = source["lasso_frozen"], source["lasso_status"]
    ols_status = source["ols_status"]
    for index in spec["reference_indices"]:
        key = f"run-{index}"
        reference_entry = lasso_frozen["references"][key]
        reference = Path(reference_entry["directory"])
        reference_report = verify_artifacts(reference, reference_entry["report_sha256"])

        new_entry = lasso_status["results"][key]
        new_directory = source["lasso"] / new_entry["directory"]
        new_report = verify_artifacts(new_directory, new_entry["report_sha256"])
        reused_entry = lasso_frozen["imported_reused_models"][key]
        reused_path = source["lasso"] / reused_entry["path"]
        if digest(reused_path) != reused_entry["sha256"]:
            raise ValueError("reused full-pair LASSO record changed")
        reused = read_json(reused_path)

        ols_entry = ols_status["results"][key]
        ols_directory = source["ols"] / ols_entry["directory"]
        ols_report = verify_artifacts(ols_directory, ols_entry["report_sha256"])

        lasso_results = {**reused["results"], **new_report["results"]}
        if set(lasso_results) != {f"hat_M{size}" for size in spec["dictionary_sizes"]}:
            raise ValueError("matched full-pair LASSO record is incomplete")
        if set(ols_report["results"]) != set(lasso_results):
            raise ValueError("matched full-pair OLS record is incomplete")

        old_stats = reference / "evaluation_statistics.npz"
        new_stats = new_directory / "evaluation_statistics.npz"
        old_stats_hash = reference_report["artifacts_sha256"]["evaluation_statistics.npz"]
        new_stats_hash = new_report["artifacts_sha256"]["evaluation_statistics.npz"]
        if digest(old_stats) != old_stats_hash or digest(new_stats) != new_stats_hash:
            raise ValueError("held-out evaluation statistics changed")
        stats = {**load_statistics(old_stats), **load_statistics(new_stats)}
        if set(stats) != set(lasso_results):
            raise ValueError("held-out statistics do not cover the dictionary grid")

        for size in spec["dictionary_sizes"]:
            name = f"hat_M{size}"
            records.append({"mode": "full", "ratio": None, "repetition": index,
                            "M": size, "method": "weak_lasso",
                            **compact_lasso_record(lasso_results[name])})
            records.append({"mode": "full", "ratio": None, "repetition": index,
                            "M": size, "method": "weak_ols",
                            **compact_ols_record(ols_report["results"][name])})

        critical = {}
        for split in ("train", "validation", "test"):
            for filename in ("data.npz", "response.npy"):
                path = reference / split / filename
                if path.exists():
                    expected = reference_report["artifacts_sha256"][f"{split}/{filename}"]
                    if digest(path) != expected:
                        raise ValueError("frozen split artifact changed")
                    critical[str(path)] = expected
        critical[str(old_stats)] = old_stats_hash
        critical[str(new_stats)] = new_stats_hash
        bundles[key] = {
            "reference_directory": str(reference),
            "reference_report_sha256": reference_entry["report_sha256"],
            "old_statistics": str(old_stats),
            "old_statistics_sha256": old_stats_hash,
            "new_statistics": str(new_stats),
            "new_statistics_sha256": new_stats_hash,
            "critical_artifacts_sha256": critical,
        }
    write_json(destination / "matched_full_pair_records.json", {"records": records})
    write_json(destination / "frozen_reference_bundles.json", bundles)
    return records, bundles


def pair_seed(spec, repetition, ratio_index, split):
    offset = spec[f"{split}_pair_seed_offset"]
    return (spec["pair_seed_start"] + repetition * spec["repetition_seed_stride"]
            + ratio_index * spec["ratio_seed_stride"] + offset)


def job_plan(spec):
    jobs = {}
    for ratio_index, ratio in enumerate(spec["pair_ratios"]):
        for repetition in spec["reference_indices"]:
            key = f"ratio-{ratio}-run-{repetition}"
            jobs[key] = {
                "id": key, "ratio": ratio, "ratio_index": ratio_index,
                "repetition": repetition,
                "relative_directory": f"sampled/P-over-N-{ratio}/run-{repetition + 1:03d}",
            }
    return jobs


def design_diagnostics(singular_values, rank, columns):
    singular_values = np.asarray(singular_values, dtype=float)
    effective = (float(singular_values[0] / singular_values[rank - 1])
                 if rank and singular_values[rank - 1] > 0 else None)
    return {
        "rank": int(rank), "columns": int(columns), "full_column_rank": int(rank) == int(columns),
        "singular_values": singular_values.tolist(),
        "condition_number": (float(singular_values[0] / singular_values[-1])
                             if rank == columns and singular_values[-1] > 0 else None),
        "effective_condition_number": effective,
    }


def fit_job(directory, job, spec, base, bundle):
    for path, expected in bundle["critical_artifacts_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"frozen source artifact changed: {path}")
    reference = Path(bundle["reference_directory"])
    train = load_data(reference / "train/data.npz")
    validation = load_data(reference / "validation/data.npz")
    train_b = np.load(reference / "train/response.npy")
    validation_b = np.load(reference / "validation/response.npy")
    n_particles = int(train["X"].shape[2])
    pair_count = int(job["ratio"] * n_particles)
    train_space, train_time = test_family(train["t"])
    validation_space, validation_time = test_family(validation["t"])
    train_seed = pair_seed(spec, job["repetition"], job["ratio_index"], "training")
    validation_seed = pair_seed(spec, job["repetition"], job["ratio_index"], "validation")
    train_A = assemble_sampled(train["X"], train["t"], train_space, train_time,
                               candidates(base), pair_count, train_seed)
    validation_A = assemble_sampled(validation["X"], validation["t"], validation_space,
                                    validation_time, candidates(base), pair_count, validation_seed)
    if train_seed == validation_seed:
        raise ValueError("training and validation pair samples are not independent")

    statistics = {**load_statistics(bundle["old_statistics"]),
                  **load_statistics(bundle["new_statistics"])}
    test = load_data(reference / "test/data.npz")
    results = {}
    for name, params in candidates(base):
        selection, _ = select_lasso(train_A[name], train_b, validation_A[name], validation_b,
                                    base["estimator"])
        methods = {}
        if selection["status"] == "selected":
            theta = np.asarray(selection["theta"], dtype=float)
            try:
                metrics, _ = evaluate_model(test, params, theta, base["simulation"]["sigma"], statistics[name])
                status = "evaluated"
                error = None
            except Exception as caught:
                metrics, status = {}, "evaluation_failed"
                error = f"{type(caught).__name__}: {caught}"
            methods["weak_lasso"] = {
                "status": status, "selection": selection, "theta": theta.tolist(),
                "metrics": metrics, "evaluation_error": error,
            }
        else:
            methods["weak_lasso"] = {"status": selection["status"],
                                      "selection": selection, "metrics": {}}

        theta_ols, residuals, rank, singular_values = np.linalg.lstsq(
            train_A[name], train_b, rcond=None)
        diagnostic = design_diagnostics(singular_values, rank, train_A[name].shape[1])
        diagnostic.update(
            solver="numpy.linalg.lstsq", rcond=None,
            reported_residual_sum_squares=residuals.tolist(),
            train_mse=float(np.mean((train_A[name] @ theta_ols - train_b) ** 2)),
            validation_mse=float(np.mean((validation_A[name] @ theta_ols - validation_b) ** 2)),
        )
        if np.all(np.isfinite(theta_ols)):
            try:
                metrics_ols, _ = evaluate_model(
                    test, params, theta_ols, base["simulation"]["sigma"], statistics[name])
                ols_status, ols_error = "evaluated", None
            except Exception as caught:
                metrics_ols, ols_status = {}, "evaluation_failed"
                ols_error = f"{type(caught).__name__}: {caught}"
        else:
            metrics_ols, ols_status, ols_error = {}, "nonfinite_fit", None
        methods["weak_ols"] = {
            "status": ols_status, "theta": theta_ols.tolist(), "fit": diagnostic,
            "metrics": metrics_ols, "evaluation_error": ols_error,
        }
        results[name] = {"basis": params, "design": diagnostic, "methods": methods}
    return {
        "particles": n_particles, "pair_ratio": job["ratio"], "sampled_pairs": pair_count,
        "pair_seeds": {"training": train_seed, "validation": validation_seed},
        "pair_policy": spec["pair_sampling"], "results": results,
        "reference_directory": str(reference),
        "reference_report_sha256": bundle["reference_report_sha256"],
        "evaluation_statistics_sha256": {
            "existing_four_sizes": bundle["old_statistics_sha256"],
            "added_six_sizes": bundle["new_statistics_sha256"],
        },
    }


def worker(batch, job, spec, base, bundle):
    batch = Path(batch)
    parent = batch / job["relative_directory"]
    directory, attempt = parent, 1
    while directory.exists():
        attempt += 1
        directory = parent.with_name(parent.name + f"-attempt-{attempt}")
    directory.mkdir(parents=True)
    started = perf_counter()
    report = {"job": job, "directory": str(directory.relative_to(batch)),
              "status": "running", "attempt": attempt}
    write_json(directory / "status.json", report)
    try:
        with threadpool_limits(limits=1):
            report.update(fit_job(directory, job, spec, base, bundle), status="completed")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    report["elapsed_seconds"] = perf_counter() - started
    write_json(directory / "report.json", report)
    write_json(directory / "status.json", {
        "status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
        "attempt": attempt,
    })
    return {
        "status": report["status"], "directory": str(directory.relative_to(batch)),
        "report_sha256": digest(directory / "report.json"),
        "elapsed_seconds": report["elapsed_seconds"], "error": report.get("error"),
    }


def sampled_records(destination, status, spec):
    records = []
    for key, job in status["jobs"].items():
        if key not in status["results"]:
            continue
        entry = status["results"][key]
        report_path = Path(destination) / entry["directory"] / "report.json"
        if digest(report_path) != entry["report_sha256"]:
            raise ValueError("sampled report hash mismatch")
        report = read_json(report_path)
        if report["status"] != "completed":
            records.append({"mode": "sampled", "ratio": job["ratio"],
                            "repetition": job["repetition"], "job_status": report["status"],
                            "job_error": report.get("error")})
            continue
        for size in spec["dictionary_sizes"]:
            name = f"hat_M{size}"
            lasso = report["results"][name]["methods"]["weak_lasso"]
            selected = lasso.get("selection", {}).get("selected", {})
            records.append({
                "mode": "sampled", "ratio": job["ratio"], "repetition": job["repetition"],
                "M": size, "method": "weak_lasso", "status": lasso["status"],
                "metrics": lasso.get("metrics", {}),
                "selection_status": lasso.get("selection", {}).get("status"),
                "selected_alpha": selected.get("alpha_sk"),
                "selected_lambda": selected.get("lambda_summed_loss"),
                "selected_path_index": selected.get("index"),
                "selected_at_grid_boundary": lasso.get("selection", {}).get("selected_at_grid_boundary"),
                "selected_converged": selected.get("converged"),
                "rank": report["results"][name]["design"]["rank"],
                "columns": report["results"][name]["design"]["columns"],
                "full_column_rank": report["results"][name]["design"]["full_column_rank"],
                "condition_number": report["results"][name]["design"]["condition_number"],
                "effective_condition_number": report["results"][name]["design"]["effective_condition_number"],
            })
            ols = report["results"][name]["methods"]["weak_ols"]
            fit = ols["fit"]
            records.append({
                "mode": "sampled", "ratio": job["ratio"], "repetition": job["repetition"],
                "M": size, "method": "weak_ols", "status": ols["status"],
                "metrics": ols.get("metrics", {}),
                "rank": fit["rank"], "columns": fit["columns"],
                "full_column_rank": fit["full_column_rank"],
                "condition_number": fit["condition_number"],
                "effective_condition_number": fit["effective_condition_number"],
            })
    return records


def finite_values(records, *, mode, ratio, method, size=None, field=None, metric=None):
    values = []
    for row in records:
        if row.get("mode") != mode or row.get("ratio") != ratio or row.get("method") != method:
            continue
        if size is not None and row.get("M") != size:
            continue
        value = row.get(field) if field else row.get("metrics", {}).get(metric)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return values


def aggregate_values(values, planned):
    return {
        "planned_count": planned, "available_count": len(values),
        "rms": float(np.sqrt(np.mean(np.square(values)))) if len(values) == planned else None,
        "mean": float(np.mean(values)) if len(values) == planned else None,
        "median": float(np.median(values)) if values else None,
        "minimum": float(np.min(values)) if values else None,
        "maximum": float(np.max(values)) if values else None,
    }


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def spearman(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    a, b = rankdata(left), rankdata(right)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize(destination, status, spec):
    destination = Path(destination)
    full = read_json(destination / "matched_full_pair_records.json")["records"]
    sampled = sampled_records(destination, status, spec)
    records = full + sampled
    write_json(destination / "records.json", {"records": records})

    metrics = spec["reporting"]["metrics"]
    sizes = spec["dictionary_sizes"]
    summary = {"protocol": "R=5 preview", "per_dictionary": {}, "global": {},
               "acceptance": {}, "minimum_passing_ratio": None}
    modes = [("full", None)] + [("sampled", ratio) for ratio in spec["pair_ratios"]]
    for mode, ratio in modes:
        label = "full" if mode == "full" else f"ratio_{ratio}"
        summary["per_dictionary"][label] = {}
        summary["global"][label] = {}
        for method in ("weak_lasso", "weak_ols"):
            summary["per_dictionary"][label][method] = {}
            for size in sizes:
                row = {}
                for metric in metrics:
                    values = finite_values(records, mode=mode, ratio=ratio, method=method,
                                           size=size, metric=metric)
                    row[metric] = aggregate_values(values, spec["repetitions"])
                lambdas = finite_values(records, mode=mode, ratio=ratio, method=method,
                                        size=size, field="selected_lambda")
                conditions = finite_values(records, mode=mode, ratio=ratio, method=method,
                                           size=size, field="effective_condition_number")
                row["selected_lambda"] = aggregate_values(lambdas, spec["repetitions"])
                row["effective_condition_number"] = aggregate_values(conditions, spec["repetitions"])
                row["rank_deficient_count"] = sum(
                    1 for item in records
                    if item.get("mode") == mode and item.get("ratio") == ratio
                    and item.get("method") == method and item.get("M") == size
                    and item.get("full_column_rank") is False)
                summary["per_dictionary"][label][method][f"M{size}"] = row
            for metric in metrics:
                values = finite_values(records, mode=mode, ratio=ratio, method=method, metric=metric)
                summary["global"][label].setdefault(method, {})[metric] = aggregate_values(
                    values, spec["repetitions"] * len(sizes))

    full_curve = [summary["per_dictionary"]["full"]["weak_lasso"][f"M{size}"]
                  ["force_error_weighted_rel"]["rms"] for size in sizes]
    full_global = summary["global"]["full"]["weak_lasso"]["force_error_weighted_rel"]["rms"]
    near_multiplier = spec["acceptance"]["dictionary_trend"]["near_optimal_full_pair_multiplier"]
    near_sizes = [size for size, value in zip(sizes, full_curve)
                  if value <= near_multiplier * min(full_curve)]
    high_sizes = spec["acceptance"]["dictionary_trend"]["high_resolution_sizes"]
    passing = []
    for ratio in spec["pair_ratios"]:
        label = f"ratio_{ratio}"
        sampled_curve = [summary["per_dictionary"][label]["weak_lasso"][f"M{size}"]
                         ["force_error_weighted_rel"]["rms"] for size in sizes]
        sampled_global = summary["global"][label]["weak_lasso"]["force_error_weighted_rel"]["rms"]
        ols_global = summary["global"][label]["weak_ols"]["force_error_weighted_rel"]["rms"]
        primary_ratio = sampled_global / full_global if sampled_global is not None else None
        rho = spearman(full_curve, sampled_curve) if all(x is not None for x in sampled_curve) else None
        best_size = sizes[int(np.argmin(sampled_curve))] if all(x is not None for x in sampled_curve) else None
        high_mean = float(np.mean([sampled_curve[sizes.index(size)] for size in high_sizes]))
        near_mean = float(np.mean([sampled_curve[sizes.index(size)] for size in near_sizes]))
        failures = [row for row in sampled if row.get("ratio") == ratio
                    and row.get("method") == "weak_lasso"
                    and (row.get("selection_status") != "selected"
                         or row.get("status") != "evaluated"
                         or row.get("metrics", {}).get("trajectory_status") != "finite")]
        checks = {
            "primary_force_rms_within_1_25": primary_ratio is not None and primary_ratio <= 1.25,
            "lasso_force_advantage_over_ols": (sampled_global is not None and ols_global is not None
                                                and sampled_global < ols_global),
            "dictionary_resolution_trend_preserved": (
                rho is not None
                and rho >= spec["acceptance"]["dictionary_trend"]["minimum_spearman_correlation"]
                and best_size in near_sizes and high_mean > near_mean),
            "no_major_lasso_failure": len(failures) == 0,
        }
        summary["acceptance"][str(ratio)] = {
            "passes": all(checks.values()), "checks": checks,
            "sampled_to_full_primary_rms_ratio": primary_ratio,
            "sampled_lasso_primary_rms": sampled_global,
            "matched_full_lasso_primary_rms": full_global,
            "sampled_ols_primary_rms": ols_global,
            "force_curve_spearman": rho, "sampled_best_dictionary_size": best_size,
            "matched_full_near_optimal_sizes": near_sizes,
            "sampled_high_resolution_rms_mean": high_mean,
            "sampled_near_optimal_rms_mean": near_mean,
            "major_lasso_failure_count": len(failures),
        }
        if all(checks.values()):
            passing.append(ratio)
    summary["minimum_passing_ratio"] = min(passing) if passing else None
    summary["minimum_statement"] = (
        f"The minimum passing ratio on the prespecified tested grid is P/N={min(passing)}."
        if passing else "No ratio on the prespecified tested grid passed all preview criteria.")
    summary["caveat"] = ("This is an R=5 matched preview. The minimum is only over the tested grid "
                         "and is not a manuscript uncertainty estimate.")
    write_json(destination / "comparison_report.json", summary)
    generate_figure(destination, summary, spec)
    summary["figure_generation"] = {
        "records_only": True,
        "runner_sha256": digest(Path(__file__)),
    }
    summary["artifacts_sha256"] = {
        "records.json": digest(destination / "records.json"),
        "sampled_pair_ratio_comparison.png": digest(destination / "sampled_pair_ratio_comparison.png"),
        "sampled_pair_ratio_comparison.pdf": digest(destination / "sampled_pair_ratio_comparison.pdf"),
    }
    write_json(destination / "comparison_report.json", summary)
    return summary


def generate_figure(destination, summary, spec):
    sizes = spec["dictionary_sizes"]
    ratios = spec["pair_ratios"]
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(ratios)))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.4))

    def curve(label, method, metric, statistic="rms"):
        return [summary["per_dictionary"][label][method][f"M{size}"][metric][statistic]
                for size in sizes]

    panels = [
        (axes[0, 0], "weak_lasso", "force_error_weighted_rel", "LASSO force error (RMS)"),
        (axes[0, 1], "weak_lasso", "trajectory_error_rel", "LASSO trajectory error (RMS)"),
        (axes[0, 2], "weak_ols", "force_error_weighted_rel", "Raw OLS force error (RMS)"),
    ]
    for ax, method, metric, title in panels:
        ax.plot(sizes, curve("full", method, metric), "k--o", linewidth=2, label="full pairs")
        for ratio, color in zip(ratios, colors):
            ax.plot(sizes, curve(f"ratio_{ratio}", method, metric), "-o", color=color,
                    markersize=3.5, label=f"P/N={ratio}")
        ax.set(xlabel="dictionary size M", ylabel="held-out relative error", title=title, yscale="log")
        ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.plot(sizes, curve("full", "weak_lasso", "selected_lambda", "median"),
            "k--o", linewidth=2, label="full pairs")
    for ratio, color in zip(ratios, colors):
        ax.plot(sizes, curve(f"ratio_{ratio}", "weak_lasso", "selected_lambda", "median"),
                "-o", color=color, markersize=3.5, label=f"P/N={ratio}")
    ax.set(xlabel="dictionary size M", ylabel="selected lambda (median)",
           title="Validation-selected LASSO penalty", yscale="log")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.plot(sizes, curve("full", "weak_ols", "effective_condition_number", "median"),
            "k--o", linewidth=2, label="full pairs")
    for ratio, color in zip(ratios, colors):
        ax.plot(sizes, curve(f"ratio_{ratio}", "weak_ols", "effective_condition_number", "median"),
                "-o", color=color, markersize=3.5, label=f"P/N={ratio}")
    ax.set(xlabel="dictionary size M", ylabel="effective condition number (median)",
           title="Training weak-design conditioning", yscale="log")
    ax.grid(alpha=0.2)

    ax = axes[1, 2]
    relative = [summary["acceptance"][str(r)]["sampled_to_full_primary_rms_ratio"] for r in ratios]
    passed = [summary["acceptance"][str(r)]["passes"] for r in ratios]
    ax.axhline(1.25, color="tab:red", linestyle="--", label="1.25 threshold")
    ax.plot(ratios, relative, color="tab:blue", marker="o")
    for ratio, value, ok in zip(ratios, relative, passed):
        ax.scatter([ratio], [value], s=70, marker="o" if ok else "x",
                   color="tab:green" if ok else "tab:red", zorder=4)
    ax.set(xlabel="sampled pair ratio P/N", ylabel="LASSO force RMS / full-pair RMS",
           title="Preview acceptance on primary error")
    ax.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=6, frameon=False)
    minimum = summary["minimum_passing_ratio"]
    fig.suptitle(f"Generic 1D sampled-pair ratio sweep (R=5 preview; minimum tested pass: {minimum})")
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95), h_pad=1.8, w_pad=1.2)
    fig.savefig(Path(destination) / "sampled_pair_ratio_comparison.png", dpi=180)
    fig.savefig(Path(destination) / "sampled_pair_ratio_comparison.pdf")
    plt.close(fig)


def initialize(spec, source, base, destination, workers):
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    for name in SOURCES:
        target = destination / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    provenance = destination / "source_provenance"
    for label, batch in (("lasso_full_pair", source["lasso"]), ("ols_full_pair", source["ols"])):
        for filename in ("frozen_protocol.json", "batch_status.json", "verification/verification.json"):
            target = provenance / label / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(batch / filename, target)
    _, bundles = build_matched_sources(spec, source, destination)
    frozen = {
        "requested_protocol": spec, "effective_base": base,
        "source_sha256": hashes,
        "source_provenance_sha256": {
            str(path.relative_to(provenance)): digest(path)
            for path in provenance.rglob("*") if path.is_file()
        },
        "matched_full_pair_records_sha256": digest(destination / "matched_full_pair_records.json"),
        "frozen_reference_bundles_sha256": digest(destination / "frozen_reference_bundles.json"),
        "reference_bundles": bundles, "workers": workers,
        "python_version": sys.version, "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }
    write_json(destination / "frozen_protocol.json", frozen)
    status = {
        "status": "running", "stage": "sampled_pair_ratio_R5_preview",
        "frozen_protocol_sha256": digest(destination / "frozen_protocol.json"),
        "jobs": job_plan(spec), "results": {},
    }
    write_json(destination / "batch_status.json", status)
    return frozen, status


def resume_state(spec, destination):
    frozen = read_json(destination / "frozen_protocol.json")
    if frozen["requested_protocol"] != spec:
        raise ValueError("resume protocol mismatch")
    for name, expected in frozen["source_sha256"].items():
        if digest(ROOT / name) != expected or digest(destination / "source_snapshot" / name) != expected:
            raise ValueError("resume source changed")
    status = read_json(destination / "batch_status.json")
    if digest(destination / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
        raise ValueError("frozen preview protocol changed")
    for entry in status["results"].values():
        if digest(destination / entry["directory"] / "report.json") != entry["report_sha256"]:
            raise ValueError("completed sampled report changed")
    return frozen, status


def run_batch(spec, destination, workers=5, resume=False):
    validate_spec(spec)
    source = load_sources(spec)
    base = validate_sources(spec, source)
    destination = Path(destination).resolve()
    if resume:
        frozen, status = resume_state(spec, destination)
    else:
        frozen, status = initialize(spec, source, base, destination, workers)
    pending = {key: job for key, job in status["jobs"].items() if key not in status["results"]}
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {
            pool.submit(worker, destination, job, spec, frozen["effective_base"],
                        frozen["reference_bundles"][f"run-{job['repetition']}"]): key
            for key, job in pending.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            entry = future.result()
            status["results"][key] = entry
            write_json(destination / "batch_status.json", status)
            print(f"{len(status['results'])}/{len(status['jobs'])}: {key} {entry['status']} "
                  f"{entry['elapsed_seconds']:.1f}s", flush=True)
    status["status"] = ("completed" if len(status["results"]) == len(status["jobs"])
                        and all(x["status"] == "completed" for x in status["results"].values())
                        else "completed_with_failures")
    status["elapsed_seconds_this_invocation"] = perf_counter() - started
    write_json(destination / "batch_status.json", status)
    comparison = summarize(destination, status, spec)
    report = {
        "status": status["status"], "destination": str(destination),
        "jobs_completed": len(status["results"]), "jobs_planned": len(status["jobs"]),
        "minimum_passing_ratio": comparison["minimum_passing_ratio"],
        "minimum_statement": comparison["minimum_statement"],
        "caveat": comparison["caveat"],
        "frozen_protocol_sha256": digest(destination / "frozen_protocol.json"),
        "batch_status_sha256": digest(destination / "batch_status.json"),
        "comparison_report_sha256": digest(destination / "comparison_report.json"),
    }
    write_json(destination / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--figure-only", type=Path)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    spec = read_json(args.config)
    validate_spec(spec)
    if args.figure_only is not None:
        destination = args.figure_only.resolve()
        status = read_json(destination / "batch_status.json")
        summary = summarize(destination, status, spec)
        report_path = destination / "report.json"
        if report_path.exists():
            report = read_json(report_path)
            report.update(
                minimum_passing_ratio=summary["minimum_passing_ratio"],
                minimum_statement=summary["minimum_statement"],
                comparison_report_sha256=digest(destination / "comparison_report.json"),
                figure_regenerated_from_records=True,
                figure_generator_sha256=digest(Path(__file__)),
            )
            write_json(report_path, report)
        print(json.dumps({"destination": str(destination),
                          "minimum_passing_ratio": summary["minimum_passing_ratio"]}, indent=2))
        return
    source = load_sources(spec)
    base = validate_sources(spec, source)
    if not args.run:
        print(json.dumps({
            "status": "configuration_valid", "pair_ratios": spec["pair_ratios"],
            "repetitions": spec["repetitions"], "jobs": len(job_plan(spec)),
            "dictionary_sizes": base["basis"]["sizes"],
        }, indent=2))
        return
    if args.resume and args.out_dir is None:
        parser.error("--resume requires --out-dir")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / (
        "sampled-pair-ratio-R5-" + stamp)
    report = run_batch(spec, destination, args.workers, args.resume)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
