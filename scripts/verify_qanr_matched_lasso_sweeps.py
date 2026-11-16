"""Independently verify a completed matched-dictionary LASSO sweep batch."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value):
    return str(value).lower() == "true"


def rmse(values):
    x = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def rmse_ci(values):
    x = np.asarray(values, dtype=float)
    squared = x * x
    estimate = float(np.sqrt(np.mean(squared)))
    standard_error = float(np.std(squared, ddof=1) / np.sqrt(x.size) / (2.0 * estimate))
    return estimate, max(0.0, estimate - 1.96 * standard_error), estimate + 1.96 * standard_error


def close(actual, expected, tolerance=1e-12):
    if not np.isclose(float(actual), float(expected), rtol=tolerance, atol=tolerance):
        raise ValueError(f"summary mismatch: {actual} != {expected}")


def verify(batch):
    batch = Path(batch).resolve()
    report = read_json(batch / "report.json")
    protocol = read_json(batch / "frozen_protocol.json")
    summary = read_json(batch / "summary.json")
    archived_check = read_json(batch / "verification.json")
    if report.get("status") != "completed" or report.get("smoke_only"):
        raise ValueError("batch is not a completed scientific run")
    if protocol.get("smoke_only") or protocol["estimator"]["truth_used_for_selection"]:
        raise ValueError("invalid selection protocol")
    if digest(batch / "frozen_protocol.json") != report["frozen_protocol_sha256"]:
        raise ValueError("frozen protocol hash mismatch")
    for relative, expected in report["source_sha256"].items():
        snapshot = batch / "source" / relative
        if not snapshot.exists() and relative.startswith("configs/"):
            snapshot = batch / "source/requested_protocol.json"
        if digest(snapshot) != expected:
            raise ValueError(f"source snapshot mismatch: {relative}")

    observation = read_csv(batch / "observation_lasso.csv")
    stochastic = read_csv(batch / "stochastic_lasso.csv")
    dt_rows = read_csv(batch / "time_step_lasso.csv")
    pair_rows = read_csv(batch / "pair_lasso.csv")
    expected_counts = report["counts"]
    observed_counts = {
        "observation_models": len(observation), "stochastic_models": len(stochastic),
        "time_step_models": len(dt_rows), "pair_models": len(pair_rows),
    }
    protocol_counts = {
        "observation_models": (
            protocol["observation_noise"]["repetitions"]
            * len(protocol["observation_noise"]["sigma_obs_values"])
        ),
        "stochastic_models": (
            protocol["process_noise"]["repetitions"] * len(protocol["process_noise"]["values"])
            + protocol["particle_number"]["repetitions"]
            * len(protocol["particle_number"]["values"])
        ),
        "time_step_models": len(protocol["time_step"]["values"]),
        "pair_models": (
            protocol["pair_count"]["repetitions"] * len(protocol["pair_count"]["values"])
        ),
    }
    if observed_counts != expected_counts or observed_counts != protocol_counts:
        raise ValueError(f"count mismatch: {observed_counts}")

    selected = observation + stochastic + dt_rows
    for row in selected:
        values = [row["theta_1"], row["theta_2"], row["coefficient_error"],
                  row["ols_coefficient_error"], row["alpha_sk"],
                  row["selected_validation_mse"]]
        if not np.all(np.isfinite(np.asarray(values, dtype=float))):
            raise ValueError("nonfinite selected model")
        if not 0 <= int(row["alpha_index"]) < protocol["estimator"]["alpha_count"]:
            raise ValueError("alpha index outside frozen grid")
        expected_boundary = int(row["alpha_index"]) in (0, protocol["estimator"]["alpha_count"] - 1)
        if as_bool(row["selected_at_grid_boundary"]) != expected_boundary:
            raise ValueError("boundary-selection flag mismatch")

    for entry in summary["observation_noise"]:
        subset = [row for row in observation if float(row["sigma_obs"]) == entry["sigma_obs"]]
        if len(subset) != protocol["observation_noise"]["repetitions"]:
            raise ValueError("observation repetition count mismatch")
        for summary_key, row_key in (
            ("lasso_rmse_theta", "coefficient_error"),
            ("ols_rmse_theta", "ols_coefficient_error"),
            ("euler_rmse_theta", "euler_coefficient_error"),
        ):
            estimate, lower, upper = rmse_ci([row[row_key] for row in subset])
            close(entry[summary_key]["rmse"], estimate)
            close(entry[summary_key]["ci95"][0], lower)
            close(entry[summary_key]["ci95"][1], upper)
            if entry[summary_key]["count"] != len(subset):
                raise ValueError("RMSE count mismatch")
    for kind, key, block in (
        ("sigma", "process_noise", "process_noise"),
        ("particles", "particle_number", "particle_number"),
    ):
        for entry in summary[key]:
            subset = [row for row in stochastic
                      if row["kind"] == kind and float(row["value"]) == entry["value"]]
            expected = protocol[block]["repetitions"]
            if len(subset) != expected:
                raise ValueError(f"{kind} repetition count mismatch")
            for summary_key, row_key in (
                ("lasso_rmse_theta", "coefficient_error"),
                ("ols_rmse_theta", "ols_coefficient_error"),
            ):
                estimate, lower, upper = rmse_ci([row[row_key] for row in subset])
                close(entry[summary_key]["rmse"], estimate)
                close(entry[summary_key]["ci95"][0], lower)
                close(entry[summary_key]["ci95"][1], upper)
    for entry in summary["time_step"]:
        subset = [row for row in dt_rows if float(row["dt"]) == entry["dt"]]
        if len(subset) != 1:
            raise ValueError("time-step count mismatch")
        close(entry["lasso_error_theta"], subset[0]["coefficient_error"])

    pair_selection = read_json(batch / "pair_full_selection.json")["selection"]
    selected_alpha = float(pair_selection["selected"]["alpha_sk"])
    pair_nonconverged = 0
    for row in pair_rows:
        close(row["alpha_sk"], selected_alpha)
        if float(row["theta_to_full_sq"]) < 0:
            raise ValueError("negative squared pair discrepancy")
        pair_nonconverged += not as_bool(row["converged"])
    if pair_nonconverged != summary["diagnostics"]["pair_nonconverged"]:
        raise ValueError("pair convergence accounting mismatch")
    for entry in summary["pair_count"]:
        subset = [row for row in pair_rows if int(row["P"]) == entry["P"]]
        if len(subset) != 100:
            raise ValueError("pair repetition count mismatch")
        close(entry["lasso_theta_to_full_sq"]["mean"], np.mean([
            float(row["theta_to_full_sq"]) for row in subset
        ]))
        close(entry["ols_theta_to_full_sq"]["mean"], np.mean([
            float(row["ols_theta_to_full_sq_recomputed"]) for row in subset
        ]))

    if not archived_check.get("passed") or archived_check.get("smoke"):
        raise ValueError("archived OLS reproduction was not verified")
    for key, value in archived_check.items():
        if key.endswith("difference") and float(value) > archived_check["tolerance"]:
            raise ValueError("archived OLS difference exceeds tolerance")

    result = {
        "status": "passed",
        "batch": str(batch),
        "report_sha256": digest(batch / "report.json"),
        "summary_sha256": digest(batch / "summary.json"),
        "frozen_protocol_sha256": digest(batch / "frozen_protocol.json"),
        "counts": observed_counts,
        "selected_models_checked": len(selected) + 1,
        "pair_models_checked": len(pair_rows),
        "pair_absolute_alpha": selected_alpha,
        "pair_nonconverged": pair_nonconverged,
        "archived_ols_reproduction": archived_check,
    }
    output = batch / "independent_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.batch), indent=2))


if __name__ == "__main__":
    main()
