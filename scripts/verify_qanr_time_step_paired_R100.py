"""Independently verify the paired-initial-state QANR time-step R100 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_RUN = (
    ROOT / "outputs/validation/qanr_redundant6_time_step_R100"
    / "sweeps-20261022T200836Z"
)
EXPECTED_TASK = "qanr_redundant6_time_step_paired_R100"
EXPECTED_DTS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def array_digest(array):
    value = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": value.shape}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def trajectory_seed(master_seed, trajectory_index, namespace):
    sequence = np.random.SeedSequence([
        int(master_seed), int(trajectory_index), int(namespace)
    ])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def expected_initial_state(master_seed, trajectory_index, namespace, particles):
    seed = trajectory_seed(master_seed, trajectory_index, namespace)
    rng = np.random.default_rng(seed)
    centers = np.asarray([0.5, 2.0, 3.5], dtype=float)
    labels = rng.integers(0, centers.size, size=particles)
    state = centers[labels, None] + rng.normal(0.0, 0.005, size=(particles, 1))
    return seed, array_digest(state)


def close(actual, expected, tolerance=1e-12):
    if not np.isclose(float(actual), float(expected), rtol=tolerance, atol=tolerance):
        raise ValueError(f"numerical mismatch: {actual} != {expected}")


def rmse_ci(values):
    values = np.asarray(values, dtype=float)
    squared = values ** 2
    estimate = float(np.sqrt(np.mean(squared)))
    if values.size < 2 or estimate == 0.0:
        interval = [estimate, estimate]
    else:
        se = float(np.std(squared, ddof=1) / np.sqrt(values.size) / (2.0 * estimate))
        interval = [max(0.0, estimate - 1.96 * se), estimate + 1.96 * se]
    return estimate, interval


def grouped_summary(records):
    groups = []
    for dt in sorted(EXPECTED_DTS):
        rows = [row for row in records if float(row["dt"]) == dt]
        lasso, lasso_ci = rmse_ci([row["coefficient_error_relative"] for row in rows])
        ols, ols_ci = rmse_ci([row["ols_coefficient_error_relative"] for row in rows])
        groups.append({
            "dt": dt,
            "lasso_relative_rmse": lasso,
            "lasso_ci95": lasso_ci,
            "ols_relative_rmse": ols,
            "ols_ci95": ols_ci,
            "count": len(rows),
        })
    rate = [row for row in groups if row["dt"] <= 0.02]
    x = np.log([row["dt"] for row in rate])
    return groups, {
        "lasso": float(np.polyfit(
            x, np.log([row["lasso_relative_rmse"] for row in rate]), 1
        )[0]),
        "ols": float(np.polyfit(
            x, np.log([row["ols_relative_rmse"] for row in rate]), 1
        )[0]),
    }


def verify(batch, old_run=DEFAULT_OLD_RUN):
    batch = Path(batch).resolve()
    old_run = None if old_run is None else Path(old_run).resolve()
    manifest = read_json(batch / "manifest.json")
    requested = read_json(batch / "requested_protocol.json")
    effective = read_json(batch / "effective_protocol.json")
    summary = read_json(batch / "summary.json")
    records = read_json(batch / "records.json")

    if manifest.get("task") != EXPECTED_TASK or manifest.get("smoke"):
        raise ValueError("not a completed production paired R100 run")
    if requested != effective or requested.get("run_class") != "production":
        raise ValueError("effective production protocol differs from requested protocol")
    block = requested["time_step"]
    if (
        block["values"] != EXPECTED_DTS
        or block["particles"] != 200
        or block["horizon"] != 10.0
        or block["sigma"] != 0.0
        or block["stacked_trajectories"] != 3
        or block["repetitions"] != 100
        or requested["estimator"]["alpha_count"] != 60
        or requested["estimator"]["selection_metric"]
        != "independent_validation_weak_residual_mse"
    ):
        raise ValueError("production protocol invariants changed")

    for relative, expected in manifest["artifact_sha256"].items():
        if digest(batch / relative) != expected:
            raise ValueError(f"artifact hash mismatch: {relative}")
    for relative, expected in manifest["job_sha256"].items():
        if digest(batch / relative) != expected:
            raise ValueError(f"job hash mismatch: {relative}")
    current_source_matches = {}
    for relative, expected in manifest["source_sha256"].items():
        snapshot = batch / "source" / relative
        if digest(snapshot) != expected:
            raise ValueError(f"source snapshot hash mismatch: {relative}")
        current_source_matches[relative] = digest(ROOT / relative) == expected
    if not all(current_source_matches.values()):
        raise ValueError("current source differs from production snapshot")

    job_paths = sorted((batch / "jobs").glob("rep-*-dt-*.json"))
    if len(job_paths) != 600 or len(manifest["job_sha256"]) != 600:
        raise ValueError("expected 600 job artifacts")
    job_records = [read_json(path) for path in job_paths]
    job_records.sort(key=lambda row: (int(row["repetition"]), float(row["dt"])))
    records_sorted = sorted(
        records, key=lambda row: (int(row["repetition"]), float(row["dt"]))
    )
    if job_records != records_sorted or len(records) != 600 or manifest["records"] != 600:
        raise ValueError("job artifacts and aggregate records differ")

    truth = np.asarray(requested["true_theta"], dtype=float)
    truth_norm = float(np.linalg.norm(truth))
    namespace = int(block["trajectory_seed_policy"]["namespace"])
    alpha_count = int(requested["estimator"]["alpha_count"])
    boundary = 0
    unconverged = 0
    rank_deficient = 0
    nonfinite = 0
    all_initial_hashes = {"training": set(), "validation": set()}
    for repetition in range(100):
        rows = [row for row in records if int(row["repetition"]) == repetition]
        if sorted(float(row["dt"]) for row in rows) != sorted(EXPECTED_DTS):
            raise ValueError(f"time-step grid mismatch in repetition {repetition}")
        train_master = block["training_seed_start"] + block["training_seed_stride"] * repetition
        validation_master = train_master + block["validation_seed_offset"]
        expected_train = [
            expected_initial_state(train_master, index, namespace, block["particles"])
            for index in range(3)
        ]
        expected_validation = [
            expected_initial_state(validation_master, index, namespace, block["particles"])
            for index in range(3)
        ]
        expected_train_seeds = [entry[0] for entry in expected_train]
        expected_train_hashes = [entry[1] for entry in expected_train]
        expected_validation_seeds = [entry[0] for entry in expected_validation]
        expected_validation_hashes = [entry[1] for entry in expected_validation]
        if len(set(expected_train_hashes + expected_validation_hashes)) != 6:
            raise ValueError(f"non-distinct initial states in repetition {repetition}")
        all_initial_hashes["training"].update(expected_train_hashes)
        all_initial_hashes["validation"].update(expected_validation_hashes)
        for row in rows:
            if (
                row["training_seed"] != train_master
                or row["validation_seed"] != validation_master
                or row["training_trajectory_seeds"] != expected_train_seeds
                or row["validation_trajectory_seeds"] != expected_validation_seeds
                or row["training_initial_state_sha256"] != expected_train_hashes
                or row["validation_initial_state_sha256"] != expected_validation_hashes
                or row["trajectory_seed_policy"]
                != block["trajectory_seed_policy"]["name"]
            ):
                raise ValueError(f"seed or initial-state pairing mismatch: repetition {repetition}")
            theta = np.asarray(row["coefficients"], dtype=float)
            ols = np.asarray(row["ols_coefficients"], dtype=float)
            close(row["coefficient_error_relative"], np.linalg.norm(theta - truth) / truth_norm)
            close(row["ols_coefficient_error_relative"], np.linalg.norm(ols - truth) / truth_norm)
            values = np.r_[theta, ols, row["alpha_sk"], row["selected_validation_mse"]]
            nonfinite += int(not np.all(np.isfinite(values)))
            index = int(row["alpha_index"])
            if not 0 <= index < alpha_count:
                raise ValueError("selected alpha index outside the 60-point path")
            expected_boundary = index in (0, alpha_count - 1)
            if bool(row["selected_at_grid_boundary"]) != expected_boundary:
                raise ValueError("boundary flag mismatch")
            boundary += expected_boundary
            unconverged += int(row["unconverged_candidates"])
            rank_deficient += bool(row["ols_rank_deficient"])
    if nonfinite or len(all_initial_hashes["training"]) != 300 \
            or len(all_initial_hashes["validation"]) != 300 \
            or all_initial_hashes["training"] & all_initial_hashes["validation"]:
        raise ValueError("nonfinite fit or initial-state stream collision")

    recomputed, slopes = grouped_summary(records)
    for actual, expected in zip(summary["time_step"], recomputed):
        if actual["dt"] != expected["dt"] or actual["repetitions"] != 100:
            raise ValueError("summary grid/count mismatch")
        for key, simple_key in (
            ("lasso_relative_rmse", "lasso_relative_rmse"),
            ("ols_relative_rmse", "ols_relative_rmse"),
        ):
            close(actual[key]["rmse"], expected[simple_key])
            ci_key = "lasso_ci95" if key.startswith("lasso") else "ols_ci95"
            close(actual[key]["ci95"][0], expected[ci_key][0])
            close(actual[key]["ci95"][1], expected[ci_key][1])
            if actual[key]["count"] != 100:
                raise ValueError("summary RMSE count mismatch")
    diagnostics = summary["diagnostics"]
    close(diagnostics["lasso_log_log_slope_dt_le_0_02"], slopes["lasso"])
    close(diagnostics["ols_log_log_slope_dt_le_0_02"], slopes["ols"])
    if diagnostics != {
        "models": 600,
        "boundary_selections": boundary,
        "unconverged_path_candidates": unconverged,
        "rank_deficient": rank_deficient,
        "missing_or_nonfinite": 0,
        "lasso_log_log_slope_dt_le_0_02": diagnostics["lasso_log_log_slope_dt_le_0_02"],
        "ols_log_log_slope_dt_le_0_02": diagnostics["ols_log_log_slope_dt_le_0_02"],
    }:
        raise ValueError("diagnostic accounting mismatch")

    with (batch / "records.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 600:
        raise ValueError("records.csv count mismatch")
    for row in csv_rows:
        match = next(item for item in records if (
            int(item["repetition"]) == int(row["repetition"])
            and float(item["dt"]) == float(row["dt"])
        ))
        close(row["coefficient_error_relative"], match["coefficient_error_relative"])
        close(row["ols_coefficient_error_relative"], match["ols_coefficient_error_relative"])

    if old_run is None:
        result = {
            "status": "passed",
            "authoritative_candidate_run": str(batch),
            "old_run": None,
            "records_checked": 600,
            "job_artifacts_checked": 600,
            "artifact_hashes_checked": len(manifest["artifact_sha256"]),
            "source_hashes_checked": len(manifest["source_sha256"]),
            "all_current_sources_match_snapshot": True,
            "bitwise_initial_state_pairing": {
                "training_states": 300,
                "validation_states": 300,
                "all_six_time_steps_match_within_repetition_and_split": True,
                "training_validation_hash_overlap": 0,
            },
            "diagnostics": diagnostics,
            "new_slopes_dt_le_0_02": slopes,
            "historical_comparison_skipped": True,
            "manifest_sha256": digest(batch / "manifest.json"),
            "records_sha256": digest(batch / "records.json"),
            "summary_sha256": digest(batch / "summary.json"),
        }
        output = batch / "independent_verification.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    old_manifest = read_json(old_run / "manifest.json")
    old_requested = read_json(old_run / "requested_protocol.json")
    old_summary = read_json(old_run / "summary.json")
    old_records = read_json(old_run / "records.json")
    for relative, expected in old_manifest["artifact_sha256"].items():
        if digest(old_run / relative) != expected:
            raise ValueError(f"old artifact hash mismatch: {relative}")
    for relative, expected in old_manifest["source_sha256"].items():
        if digest(old_run / "source" / relative) != expected:
            raise ValueError(f"old source snapshot hash mismatch: {relative}")
    old_jobs = sorted((old_run / "jobs").glob("rep-*-dt-*.json"))
    old_job_records = [read_json(path) for path in old_jobs]
    old_job_records.sort(key=lambda row: (int(row["repetition"]), float(row["dt"])))
    if len(old_jobs) != 600 or old_job_records != sorted(
            old_records, key=lambda row: (int(row["repetition"]), float(row["dt"]))):
        raise ValueError("old job artifacts and aggregate records differ")
    for key, value in old_requested["time_step"].items():
        if block.get(key) != value:
            raise ValueError(f"old/new scientific setting differs: {key}")
    if (
        old_requested["base_protocol"] != requested["base_protocol"]
        or old_requested["true_theta"] != requested["true_theta"]
        or old_requested["estimator"] != requested["estimator"]
    ):
        raise ValueError("old/new base, truth, or estimator differs")
    old_recomputed, old_slopes = grouped_summary(old_records)
    for actual, expected in zip(old_summary["time_step"], old_recomputed):
        close(actual["lasso_relative_rmse"]["rmse"], expected["lasso_relative_rmse"])
        close(actual["ols_relative_rmse"]["rmse"], expected["ols_relative_rmse"])
    close(
        old_summary["diagnostics"]["lasso_log_log_slope_dt_le_0_02"],
        old_slopes["lasso"],
    )
    close(
        old_summary["diagnostics"]["ols_log_log_slope_dt_le_0_02"],
        old_slopes["ols"],
    )
    comparison = []
    for new, old in zip(recomputed, old_recomputed):
        if new["dt"] != old["dt"]:
            raise ValueError("old/new summary grids differ")
        comparison.append({
            "dt": new["dt"],
            "new_lasso_relative_rmse": new["lasso_relative_rmse"],
            "old_lasso_relative_rmse": old["lasso_relative_rmse"],
            "lasso_difference": (
                new["lasso_relative_rmse"] - old["lasso_relative_rmse"]
            ),
            "new_ols_relative_rmse": new["ols_relative_rmse"],
            "old_ols_relative_rmse": old["ols_relative_rmse"],
            "ols_difference": (
                new["ols_relative_rmse"] - old["ols_relative_rmse"]
            ),
        })
    result = {
        "status": "passed",
        "authoritative_candidate_run": str(batch),
        "old_run": str(old_run),
        "records_checked": 600,
        "job_artifacts_checked": 600,
        "artifact_hashes_checked": len(manifest["artifact_sha256"]),
        "source_hashes_checked": len(manifest["source_sha256"]),
        "all_current_sources_match_snapshot": True,
        "bitwise_initial_state_pairing": {
            "training_states": 300,
            "validation_states": 300,
            "all_six_time_steps_match_within_repetition_and_split": True,
            "training_validation_hash_overlap": 0,
        },
        "diagnostics": diagnostics,
        "new_slopes_dt_le_0_02": slopes,
        "old_slopes_dt_le_0_02": old_slopes,
        "old_artifact_hashes_checked": len(old_manifest["artifact_sha256"]),
        "old_source_hashes_checked": len(old_manifest["source_sha256"]),
        "old_job_artifacts_checked": 600,
        "slope_differences": {
            key: slopes[key] - old_slopes[key] for key in ("lasso", "ols")
        },
        "rmse_comparison": comparison,
        "manifest_sha256": digest(batch / "manifest.json"),
        "records_sha256": digest(batch / "records.json"),
        "summary_sha256": digest(batch / "summary.json"),
    }
    output = batch / "independent_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--old-run", type=Path, default=DEFAULT_OLD_RUN)
    parser.add_argument(
        "--skip-old-comparison", action="store_true",
        help="verify the fresh production run without requiring the historical run",
    )
    args = parser.parse_args()
    old_run = None if args.skip_old_comparison else args.old_run
    print(json.dumps(verify(args.batch, old_run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
