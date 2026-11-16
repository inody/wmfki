"""Evaluate the two error measures used by L4DC Table 1.

For each method, the table reports the coefficient RMSE and drift RMSE over
every repetition fixed
before inspection. The rendered table uses dimensionless point estimates;
delta-method standard errors remain in the JSON summary. Representative
trajectory errors remain diagnostic outputs and do not enter the table. No
estimator is refit or retuned.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import make_kernel_basis, make_kernel_evaluator
from predator_prey import channel_kernels
from scripts.run_localized10d_pilot import _simulate_coupled
from scripts.run_predator_prey_application import drift_error, model_objects, simulate_split
from scripts.run_sindy_comparison import reference_trajectory, trajectory_errors


DEFAULT_CONFIG = ROOT / "configs/experiments/cross_example_metric_table.json"
METHODS = ("weak_lasso", "weak_ols", "strong_forward_difference_ols")
REGULARIZED_METHODS = (
    "weak_lasso",
    "weak_ridge_validation_selected",
    "weak_ols",
    "strong_forward_difference_lasso_validation_selected",
    "strong_forward_difference_ridge_validation_selected",
    "strong_forward_difference_ols",
)
EXAMPLES = ("one_dimensional", "predator_prey", "localized_10d")
METHOD_LABELS = {
    "weak_lasso": "Weak LASSO",
    "weak_ols": "Weak OLS",
    "strong_forward_difference_ols": "Strong FD--OLS",
}
EXAMPLE_LABELS = {
    "one_dimensional": "1D Newtonian-repulsion",
    "predator_prey": "2D predator--prey",
    "localized_10d": "10D distance-localized",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verified_path(relative, expected):
    path = ROOT / relative
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"input hash mismatch: {relative}: {actual} != {expected}")
    return path


def rmse_with_standard_error(values):
    """Return RMSE and its delta-method standard error across repetitions."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None, None
    squared = values ** 2
    estimate = float(np.sqrt(np.mean(squared)))
    if values.size < 2:
        return estimate, None
    mean_squared_standard_error = float(
        np.std(squared, ddof=1) / np.sqrt(values.size)
    )
    if estimate == 0.0:
        standard_error = 0.0 if mean_squared_standard_error == 0.0 else None
    else:
        standard_error = mean_squared_standard_error / (2.0 * estimate)
    return estimate, standard_error


def one_dimensional_metrics(rows, table_spec, model_spec):
    block = table_spec["one_dimensional"]
    observation = model_spec["observation_noise"]
    repetition_zero = next(
        row for row in rows
        if row["example"] == "one_dimensional"
        and row["method"] == "weak_lasso"
        and int(row["repetition"]) == 0
    )
    evaluation_seed = int(repetition_zero["training_seed"]) + 2 * int(
        model_spec["validation_seed_offset"]
    )
    X, time, X0, driving_noise, _ = reference_trajectory(
        evaluation_seed,
        particles=int(observation["particles"]),
        horizon=float(observation["horizon"]),
        step=float(observation["dt"]),
        sigma=float(observation["sigma"]),
    )
    basis, _ = make_kernel_basis(model_spec["dictionary"])
    truth = np.asarray(model_spec["true_theta"], dtype=float)

    rng = np.random.default_rng(evaluation_seed + int(block["drift_sampling_seed_offset"]))
    count = int(block["drift_target_time_samples"])
    repetitions = rng.integers(0, X.shape[0], size=count)
    times = rng.integers(0, X.shape[1], size=count)
    targets = rng.integers(0, X.shape[2], size=count)
    design = np.empty((count, truth.size), dtype=float)
    for index, (rep, time_index, target) in enumerate(zip(repetitions, times, targets)):
        displacement = X[rep, time_index, target][None, :] - X[rep, time_index]
        design[index] = basis(displacement).mean(axis=0)[:, 0]
    true_drift = design @ truth
    denominator = float(np.sum(true_drift ** 2))
    if denominator <= 0.0:
        raise RuntimeError("one-dimensional held-out drift has zero norm")

    output = {}
    for method in METHODS:
        method_rows = [
            row for row in rows
            if row["example"] == "one_dimensional" and row["method"] == method
        ]
        drift_values = []
        for row in method_rows:
            theta = np.asarray(row["coefficients"], dtype=float)
            drift_values.append(float(np.sqrt(np.sum((design @ theta - true_drift) ** 2) / denominator)))
        first = next(row for row in method_rows if int(row["repetition"]) == 0)
        kernel = make_kernel_evaluator(np.asarray(first["coefficients"], dtype=float), basis)
        trajectory = trajectory_errors(
            kernel, X, X0, time, driving_noise, sigma=float(observation["sigma"])
        )
        output[method] = {
            "drift_values": drift_values,
            "trajectory_error_relative": (
                None if trajectory["diverged"] else trajectory["trajectory_error_rel"]
            ),
            "trajectory_status": "diverged" if trajectory["diverged"] else "finite",
        }
    return output, {
        "evaluation_seed": evaluation_seed,
        "drift_target_time_samples": count,
        "trajectory_repetition": 0,
    }


def predator_trajectory_metrics(rows, spec):
    prey_basis, predator_basis, _, truth, truth_kernels = model_objects(spec)
    evaluation = spec["evaluation"]
    seed = int(spec["seeds"]["training_start"]) + int(spec["seeds"]["display_offset"])
    particles = int(evaluation["display_particles"])
    horizon = float(evaluation["display_horizon"])
    true_x, true_z, _, diagnostics = simulate_split(
        spec, 1, seed, truth_kernels, particle_count=particles, horizon=horizon
    )
    if not all(diagnostics["finite"]):
        raise RuntimeError("predator--prey reference rollout diverged")
    channel_size = len(truth[0])
    output = {}
    for method in METHODS:
        first = next(
            row for row in rows
            if row["example"] == "predator_prey"
            and row["method"] == method
            and int(row["repetition"]) == 0
        )
        theta = np.asarray(first["coefficients"], dtype=float)
        estimated = channel_kernels(
            theta[:channel_size], theta[channel_size:2 * channel_size],
            theta[2 * channel_size:], prey_basis, predator_basis,
        )
        fitted_x, fitted_z, _, fitted_diagnostics = simulate_split(
            spec, 1, seed + 1, estimated, particle_count=particles, horizon=horizon,
            initial_prey=true_x[:, 0], initial_predator=true_z[:, 0],
        )
        finite = bool(
            all(fitted_diagnostics["finite"])
            and np.all(np.isfinite(fitted_x)) and np.all(np.isfinite(fitted_z))
        )
        error = None
        if finite:
            numerator = np.sum((fitted_x - true_x) ** 2) + np.sum((fitted_z - true_z) ** 2)
            denominator = np.sum(true_x ** 2) + np.sum(true_z ** 2)
            error = float(np.sqrt(numerator / denominator))
        representative_drift = drift_error(
            true_x, true_z, truth_kernels, estimated,
            spec["evaluation"]["drift_time_stride"],
        )
        output[method] = {
            "representative_drift_error_relative": representative_drift[
                "species_balanced_relative"
            ],
            "trajectory_error_relative": error,
            "trajectory_status": "finite" if finite else "diverged",
        }
    return output, {
        "evaluation_seed": seed,
        "particles": particles,
        "horizon": horizon,
        "trajectory_repetition": 0,
    }


def localized_trajectory_metrics(rows, spec):
    first_lasso = next(
        row for row in rows
        if row["example"] == "localized_10d"
        and row["method"] == "weak_lasso"
        and int(row["repetition"]) == 0
    )
    seed = int(first_lasso["test_seed"])
    output = {}
    for method in METHODS:
        first = next(
            row for row in rows
            if row["example"] == "localized_10d"
            and row["method"] == method
            and int(row["repetition"]) == 0
        )
        try:
            _, learned, error = _simulate_coupled(
                spec, np.asarray(first["coefficients"], dtype=float), seed
            )
            finite = bool(np.all(np.isfinite(learned)) and np.isfinite(error))
        except (FloatingPointError, OverflowError, ValueError):
            finite, error = False, None
        output[method] = {
            "trajectory_error_relative": float(error) if finite else None,
            "trajectory_status": "finite" if finite else "diverged",
        }
    return output, {"evaluation_seed": seed, "trajectory_repetition": 0}


def format_relative(value):
    """Format a dimensionless relative RMSE for the manuscript table."""
    if value is None:
        return "div."
    value = float(value)
    if value >= 1.0:
        return f"{value:.2f}"
    if value >= 0.1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def build_p50(table_spec, config_path, output):
    source_paths = {
        name: verified_path(item["path"], item["sha256"])
        for name, item in table_spec["source_artifacts"].items()
    }
    one_d = read_json(source_paths["one_dimensional_table"])
    predator_records = read_json(source_paths["predator_nominal_records"])
    localized_records = read_json(source_paths["localized_records"])
    legacy_records = read_json(source_paths["legacy_strong_records"])
    predator_protocol = read_json(source_paths["predator_protocol"])
    truth = np.concatenate([
        np.asarray(predator_protocol["dictionary"]["prey_prey_true"], dtype=float),
        np.asarray(predator_protocol["dictionary"]["prey_predator_true"], dtype=float),
        np.asarray(predator_protocol["dictionary"]["predator_prey_true"], dtype=float),
    ])

    groups = []
    one_d_groups = {row["method"]: row for row in one_d["groups"]}
    for method in METHODS:
        row = one_d_groups[method]
        groups.append({
            "example": "one_dimensional", "method": method,
            "repetitions": int(row["repetitions"]),
            "coefficient_rmse_relative": float(row["coefficient"]["rmse"]),
            "coefficient_rmse_standard_error_relative": float(row["coefficient"]["standard_error"]),
            "drift_rmse_relative": float(row["heldout_drift"]["rmse"]),
            "drift_rmse_standard_error_relative": float(row["heldout_drift"]["standard_error"]),
            "trajectory_error_relative": None, "trajectory_status": "not_in_table_artifact",
        })

    for method, source_name in (("weak_lasso", "lasso"), ("weak_ols", "ols")):
        coefficient_values, drift_values = [], []
        for record in predator_records:
            channels = record["methods"][source_name]["channels"]
            theta = np.concatenate([
                np.asarray(channels["prey_prey"]["coefficients"], dtype=float),
                np.asarray(channels["prey_predator"]["coefficients"], dtype=float),
                np.asarray(channels["predator_prey"]["coefficients"], dtype=float),
            ])
            coefficient_values.append(float(np.linalg.norm(theta - truth) / np.linalg.norm(truth)))
            drift_values.append(float(record["methods"][source_name]["held_out_drift"]["species_balanced_relative"]))
        coefficient, coefficient_se = rmse_with_standard_error(coefficient_values)
        drift, drift_se = rmse_with_standard_error(drift_values)
        groups.append({
            "example": "predator_prey", "method": method,
            "repetitions": len(coefficient_values),
            "coefficient_rmse_relative": coefficient,
            "coefficient_rmse_standard_error_relative": coefficient_se,
            "drift_rmse_relative": drift,
            "drift_rmse_standard_error_relative": drift_se,
            "trajectory_error_relative": None, "trajectory_status": "reported_separately",
        })

    for example in ("predator_prey", "localized_10d"):
        rows = [row for row in legacy_records if row["example"] == example
                and row["method"] == "strong_forward_difference_ols"]
        coefficient, coefficient_se = rmse_with_standard_error(
            [row["coefficient_error_relative"] for row in rows]
        )
        drift, drift_se = rmse_with_standard_error(
            [row["heldout_drift_error_relative"] for row in rows]
        )
        groups.append({
            "example": example, "method": "strong_forward_difference_ols",
            "repetitions": len(rows),
            "coefficient_rmse_relative": coefficient,
            "coefficient_rmse_standard_error_relative": coefficient_se,
            "drift_rmse_relative": drift,
            "drift_rmse_standard_error_relative": drift_se,
            "trajectory_error_relative": None, "trajectory_status": "unchanged_strong_baseline",
        })

    for method, source_name in (("weak_lasso", "lasso"), ("weak_ols", "ols")):
        rows = [row for row in localized_records
                if row["sweep"] == "process_sigma"
                and float(row["sweep_value"]) == 0.05
                and row["method"] == source_name]
        coefficient, coefficient_se = rmse_with_standard_error(
            [row["coefficient_error_relative"] for row in rows]
        )
        drift, drift_se = rmse_with_standard_error(
            [row["latent_drift_relative_error"] for row in rows]
        )
        groups.append({
            "example": "localized_10d", "method": method,
            "repetitions": len(rows),
            "coefficient_rmse_relative": coefficient,
            "coefficient_rmse_standard_error_relative": coefficient_se,
            "drift_rmse_relative": drift,
            "drift_rmse_standard_error_relative": drift_se,
            "trajectory_error_relative": None, "trajectory_status": "reported_separately",
        })

    summary = {
        "task": table_spec["task"], "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pair_policy": "P=50N for weak estimators; strong FD--OLS does not use the weak pair design",
        "common_observation_noise": 0.01,
        "source_artifacts": table_spec["source_artifacts"],
        "metrics": {
            "coefficient": "coefficient RMSE across all repetitions",
            "drift": "drift RMSE across all repetitions",
            "uncertainty": "delta-method standard errors stored in summary and omitted from rendered table",
        },
        "groups": groups,
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    write_json(output / "summary.json", summary)
    (output / "requested_protocol.json").write_text(Path(config_path).read_text())
    lookup = {(row["example"], row["method"]): row for row in groups}
    tex = [
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcccccc@{}}\n",
        "\\toprule\n",
        "System & \\multicolumn{2}{c}{Weak LASSO} & \\multicolumn{2}{c}{Weak OLS} & ",
        "\\multicolumn{2}{c}{Strong FD--OLS} \\\\\n",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n",
        " & coefficient & drift & coefficient & drift & coefficient & drift \\\\\n",
        "\\midrule\n",
    ]
    for example in EXAMPLES:
        cells = []
        for method in METHODS:
            row = lookup[(example, method)]
            cells.extend([format_relative(row["coefficient_rmse_relative"]),
                          format_relative(row["drift_rmse_relative"])])
        tex.append(EXAMPLE_LABELS[example] + " & " + " & ".join(cells) + " \\\\\n")
    tex.extend(["\\bottomrule\n", "\\end{tabular*}\n"])
    (output / "table_draft.tex").write_text("".join(tex))
    write_json(output / "artifact_sha256.json", {
        name: digest(output / name)
        for name in ("requested_protocol.json", "summary.json", "table_draft.tex")
    })
    return output


def format_three_significant_figures(value):
    """Format the manuscript point estimates with three significant figures."""
    value = float(value)
    if value == 0.0:
        return "0"
    decimals = 2 - int(np.floor(np.log10(abs(value))))
    if decimals > 0:
        return f"{value:.{decimals}f}"
    rounded = round(value, decimals)
    return f"{rounded:.0f}"


def build_regularized(table_spec, config_path, output):
    source = verified_path(
        table_spec["regularized_summary"],
        table_spec["regularized_summary_sha256"],
    )
    summary = read_json(source)
    lookup = {
        (row["example"], row["method"]): row
        for row in summary["groups"]
    }
    missing = [
        (example, method) for example in EXAMPLES for method in REGULARIZED_METHODS
        if (example, method) not in lookup
    ]
    if missing:
        raise RuntimeError(f"regularized table groups are incomplete: {missing}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    groups = []
    for example in EXAMPLES:
        for method in REGULARIZED_METHODS:
            row = lookup[(example, method)]
            groups.append({
                "example": example,
                "method": method,
                "repetitions": int(row["coefficient_error_relative"]["count"]),
                "coefficient_rmse_relative": float(
                    row["coefficient_error_relative"]["rmse"]
                ),
                "drift_rmse_relative": float(
                    row["heldout_drift_error_relative"]["rmse"]
                ),
            })
    rendered = {(row["example"], row["method"]): row for row in groups}
    row_labels = {
        "one_dimensional": "1D Newtonian",
        "predator_prey": "2D predator--prey",
        "localized_10d": "10D localized",
    }
    tex = [
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcccccc@{}}\n",
        "\\toprule\n",
        "& \\multicolumn{3}{c}{Weak form} & \\multicolumn{3}{c}{Strong FD} \\\\\n",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n",
        "System & LASSO & Ridge & OLS & LASSO & Ridge & OLS \\\\\n",
        "\\midrule\n",
    ]
    for example in EXAMPLES:
        cells = []
        for method in REGULARIZED_METHODS:
            row = rendered[(example, method)]
            value = (
                format_three_significant_figures(row["coefficient_rmse_relative"])
                + "/"
                + format_three_significant_figures(row["drift_rmse_relative"])
            )
            if method == "weak_lasso":
                value = "\\textbf{" + value + "}"
            cells.append(value)
        tex.append(row_labels[example] + " & " + " & ".join(cells) + " \\\\\n")
    tex.extend(["\\bottomrule\n", "\\end{tabular*}\n"])
    result = {
        "task": table_spec["task"],
        "protocol_version": 3,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "regularized_summary": table_spec["regularized_summary"],
        "regularized_summary_sha256": table_spec["regularized_summary_sha256"],
        "groups": groups,
    }
    write_json(output / "summary.json", result)
    (output / "requested_protocol.json").write_text(Path(config_path).read_text())
    (output / "table_draft.tex").write_text("".join(tex))
    write_json(output / "artifact_sha256.json", {
        name: digest(output / name)
        for name in ("requested_protocol.json", "summary.json", "table_draft.tex")
    })
    return output


def build(config_path, output):
    table_spec = read_json(config_path)
    if table_spec.get("task") == "cross_example_metric_table" \
            and table_spec.get("protocol_version") == 3:
        if tuple(table_spec.get("methods", [])) != REGULARIZED_METHODS:
            raise ValueError("regularized method order changed")
        return build_regularized(table_spec, config_path, output)
    if table_spec.get("task") == "cross_example_metric_table" \
            and table_spec.get("protocol_version") == 2:
        if tuple(table_spec.get("methods", [])) != METHODS:
            raise ValueError("method order changed")
        return build_p50(table_spec, config_path, output)
    if table_spec.get("task") != "cross_example_metric_table" \
            or table_spec.get("protocol_version") != 1:
        raise ValueError("invalid cross-example metric-table protocol")
    if tuple(table_spec.get("methods", [])) != METHODS:
        raise ValueError("method order changed")
    records_path = verified_path(
        table_spec["source_records"], table_spec["source_records_sha256"]
    )
    records = read_json(records_path)
    specs = {}
    for example in EXAMPLES:
        block = table_spec[example]
        specs[example] = read_json(verified_path(block["config"], block["config_sha256"]))

    one_d, one_d_protocol = one_dimensional_metrics(
        records, table_spec, specs["one_dimensional"]
    )
    predator, predator_protocol = predator_trajectory_metrics(
        records, specs["predator_prey"]
    )
    localized, localized_protocol = localized_trajectory_metrics(
        records, specs["localized_10d"]
    )

    summary = {
        "task": table_spec["task"],
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_records": table_spec["source_records"],
        "source_records_sha256": table_spec["source_records_sha256"],
        "metrics": {
            "coefficient": "coefficient RMSE across all repetitions",
            "drift": "drift RMSE across all repetitions",
            "trajectory": "diagnostic trajectory error for repetition zero",
            "uncertainty": "delta-method standard errors are stored in the JSON summary but omitted from the rendered table",
        },
        "evaluation_protocol": {
            "one_dimensional": one_d_protocol,
            "predator_prey": predator_protocol,
            "localized_10d": localized_protocol,
        },
        "groups": [],
    }
    for example in EXAMPLES:
        for method in METHODS:
            method_rows = [
                row for row in records
                if row["example"] == example and row["method"] == method
            ]
            coefficient, coefficient_standard_error = rmse_with_standard_error(
                [row["coefficient_error_relative"] for row in method_rows]
            )
            drift_values = (
                one_d[method]["drift_values"]
                if example == "one_dimensional"
                else [row["heldout_drift_error_relative"] for row in method_rows]
            )
            aggregate_drift, drift_standard_error = rmse_with_standard_error(drift_values)
            if example == "one_dimensional":
                representative_drift = one_d[method]["drift_values"][0]
                trajectory = one_d[method]["trajectory_error_relative"]
                trajectory_status = one_d[method]["trajectory_status"]
            elif example == "predator_prey":
                representative_drift = predator[method][
                    "representative_drift_error_relative"
                ]
                trajectory = predator[method]["trajectory_error_relative"]
                trajectory_status = predator[method]["trajectory_status"]
            else:
                first = next(row for row in method_rows if int(row["repetition"]) == 0)
                representative_drift = first["heldout_drift_error_relative"]
                trajectory = localized[method]["trajectory_error_relative"]
                trajectory_status = localized[method]["trajectory_status"]
            summary["groups"].append({
                "example": example,
                "method": method,
                "repetitions": len(method_rows),
                "coefficient_rmse_relative": coefficient,
                "coefficient_rmse_standard_error_relative": coefficient_standard_error,
                "drift_rmse_relative": aggregate_drift,
                "drift_rmse_standard_error_relative": drift_standard_error,
                "representative_drift_error_relative_diagnostic": representative_drift,
                "trajectory_error_relative": trajectory,
                "trajectory_status": trajectory_status,
            })

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    write_json(output / "summary.json", summary)
    (output / "requested_protocol.json").write_text(Path(config_path).read_text())

    groups = {(row["example"], row["method"]): row for row in summary["groups"]}
    tex = [
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcccccc@{}}\n",
        "\\toprule\n",
        "System & \\multicolumn{2}{c}{Weak LASSO} & \\multicolumn{2}{c}{Weak OLS} & "
        "\\multicolumn{2}{c}{Strong FD--OLS} \\\\\n",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n",
        " & coefficient & drift & coefficient & drift & coefficient & drift \\\\\n",
        "\\midrule\n",
    ]
    for example in EXAMPLES:
        cells = []
        for method in METHODS:
            row = groups[(example, method)]
            cells.extend([
                format_relative(row["coefficient_rmse_relative"]),
                format_relative(row["drift_rmse_relative"]),
            ])
        tex.append(EXAMPLE_LABELS[example] + " & " + " & ".join(cells) + " \\\\\n")
    tex.extend(["\\bottomrule\n", "\\end{tabular*}\n"])
    (output / "table_draft.tex").write_text("".join(tex))
    write_json(output / "artifact_sha256.json", {
        name: digest(output / name)
        for name in ("requested_protocol.json", "summary.json", "table_draft.tex")
    })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "outputs/validation/cross_example_metric_table" / stamp
    print(build(args.config, args.output))


if __name__ == "__main__":
    main()
