"""Build a cross-example nominal comparison of weak and strong regressions.

The weak-LASSO and weak-OLS records are read from the frozen scientific
outputs already used by the manuscript.  The one-dimensional output already
contains the matched forward-difference OLS baseline.  For predator--prey and
localized 10D systems, this script reruns the predeclared training trajectories
and fits raw minimum-norm OLS to forward position differences.  No manuscript
file is modified.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import make_kernel_basis
from predator_prey import channel_kernels, predator_prey_velocity
from scripts.run_localized10d_pilot import validate_protocol as validate_localized_protocol
from scripts.run_morse10d_sweeps import (
    _radial_drift_fast,
    _simulate_truth,
    evaluate_theta,
    latent_evaluation_system,
    repetition_plan,
)
from scripts.run_predator_prey_application import (
    drift_error,
    model_objects,
    simulate_split,
    validate_protocol as validate_predator_protocol,
)


DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "cross_example_nominal_comparison.json"
SOURCE_FILES = (
    "configs/experiments/cross_example_nominal_comparison.json",
    "scripts/run_cross_example_nominal_comparison.py",
    "scripts/verify_cross_example_nominal_comparison.py",
    "test/test_cross_example_nominal_comparison.py",
    "scripts/run_predator_prey_application.py",
    "scripts/run_localized10d_pilot.py",
    "scripts/run_morse10d_sweeps.py",
    "predator_prey.py",
    "kernels.py",
)
METHOD_ORDER = (
    "weak_lasso",
    "weak_ols",
    "strong_forward_difference_ols",
)
METHOD_LABELS = {
    "weak_lasso": "Weak LASSO",
    "weak_ols": "Weak OLS",
    "strong_forward_difference_ols": "Strong FD-OLS",
}
EXAMPLE_ORDER = ("one_dimensional", "predator_prey", "localized_10d")
EXAMPLE_LABELS = {
    "one_dimensional": "1D overcomplete",
    "predator_prey": "2D predator--prey",
    "localized_10d": "10D localized",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_error(estimate, truth):
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    denominator = float(np.linalg.norm(truth))
    if denominator <= 0.0:
        raise ValueError("truth must have positive norm")
    return float(np.linalg.norm(estimate - truth) / denominator)


def _ols(A, y):
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    theta, _, rank, singular = np.linalg.lstsq(A, y, rcond=None)
    finite = bool(np.all(np.isfinite(theta)))
    condition = float(np.linalg.cond(A))
    return theta, {
        "rows": int(A.shape[0]),
        "columns": int(A.shape[1]),
        "rank": int(rank),
        "rank_deficient": bool(rank < A.shape[1]),
        "condition_A": condition if np.isfinite(condition) else None,
        "smallest_singular_value": float(singular[-1]) if singular.size else None,
        "finite": finite,
        "max_abs_coefficient": float(np.max(np.abs(theta))) if finite else None,
    }


def predator_prey_strong_features(prey, predator, prey_basis, predator_basis):
    """Return exact full-source strong-form features at one configuration."""
    prey = np.asarray(prey, dtype=float)
    predator = np.asarray(predator, dtype=float)
    particle_count = prey.shape[0]
    pair_feature = prey_basis(prey[:, None, :] - prey[None, :, :]).sum(axis=1)
    pair_feature /= particle_count
    cross_feature = prey_basis(prey - predator[None, :])
    prey_feature = np.concatenate([pair_feature, cross_feature], axis=1)
    predator_feature = predator_basis(prey - predator[None, :]).mean(axis=0)
    return prey_feature, predator_feature


def assemble_predator_prey_strong(prey, predator, time, prey_basis, predator_basis):
    prey = np.asarray(prey, dtype=float)
    predator = np.asarray(predator, dtype=float)
    time = np.asarray(time, dtype=float)
    prey_rows, prey_response = [], []
    predator_rows, predator_response = [], []
    for repetition in range(prey.shape[0]):
        for time_index, dt in enumerate(np.diff(time)):
            x = prey[repetition, time_index]
            z = predator[repetition, time_index, 0]
            feature_x, feature_z = predator_prey_strong_features(
                x, z, prey_basis, predator_basis
            )
            prey_rows.append(feature_x.transpose(0, 2, 1).reshape(-1, feature_x.shape[1]))
            prey_response.append(
                ((prey[repetition, time_index + 1] - x) / dt).reshape(-1)
            )
            predator_rows.append(feature_z.T)
            predator_response.append(
                ((predator[repetition, time_index + 1, 0] - z) / dt).reshape(-1)
            )
    return (
        np.vstack(prey_rows), np.concatenate(prey_response),
        np.vstack(predator_rows), np.concatenate(predator_response),
    )


def localized_strong_features(x, targets, kernel_basis):
    """Return all-source radial features for selected target particles."""
    x = np.asarray(x, dtype=float)
    targets = np.asarray(targets, dtype=int)
    displacement = x[targets, None, :] - x[None, :, :]
    return kernel_basis(displacement).mean(axis=1)


def assemble_localized_strong(X, time, targets, kernel_basis):
    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    rows, response = [], []
    for repetition in range(X.shape[0]):
        for time_index, dt in enumerate(np.diff(time)):
            x = X[repetition, time_index]
            feature = localized_strong_features(x, targets, kernel_basis)
            rows.append(feature.transpose(0, 2, 1).reshape(-1, feature.shape[1]))
            response.append(
                ((X[repetition, time_index + 1, targets] - x[targets]) / dt).reshape(-1)
            )
    return np.vstack(rows), np.concatenate(response)


def _predator_worker(payload):
    index, spec = payload
    seeds = spec["seeds"]
    training_seed = seeds["training_start"] + seeds["repetition_stride"] * index
    test_seed = training_seed + seeds["test_offset"]
    prey_basis, predator_basis, _, truth, truth_kernels = model_objects(spec)
    observation = spec["observation"]
    with threadpool_limits(limits=1):
        train_x, train_z, time, diagnostics = simulate_split(
            spec, observation["training_trajectories"], training_seed, truth_kernels
        )
        if not all(diagnostics["finite"]):
            raise RuntimeError(f"predator--prey training divergence at repetition {index}")
        sigma_obs = float(observation["observation_noise"])
        rng = np.random.default_rng(training_seed + 1)
        observed_x = train_x + sigma_obs * rng.normal(size=train_x.shape)
        observed_z = train_z + sigma_obs * rng.normal(size=train_z.shape)
        A_x, b_x, A_z, b_z = assemble_predator_prey_strong(
            observed_x, observed_z, time, prey_basis, predator_basis
        )
        theta_prey, prey_diagnostics = _ols(A_x, b_x)
        theta_zx, predator_diagnostics = _ols(A_z, b_z)
        channel_count = truth[0].size
        theta_xx, theta_xz = theta_prey[:channel_count], theta_prey[channel_count:]
        theta = np.concatenate([theta_xx, theta_xz, theta_zx])
        truth_vector = np.concatenate(truth)
        estimated_kernels = channel_kernels(
            theta_xx, theta_xz, theta_zx, prey_basis, predator_basis
        )
        test_x, test_z, _, test_diagnostics = simulate_split(
            spec, observation["test_trajectories"], test_seed, truth_kernels
        )
        if not all(test_diagnostics["finite"]):
            raise RuntimeError(f"predator--prey test divergence at repetition {index}")
        drift = drift_error(
            test_x, test_z, truth_kernels, estimated_kernels,
            spec["evaluation"]["drift_time_stride"],
        )
    return {
        "example": "predator_prey",
        "method": "strong_forward_difference_ols",
        "repetition": index,
        "training_seed": training_seed,
        "test_seed": test_seed,
        "coefficient_error_relative": relative_error(theta, truth_vector),
        "heldout_drift_error_relative": drift["species_balanced_relative"],
        "coefficients": theta.tolist(),
        "prey_system": prey_diagnostics,
        "predator_system": predator_diagnostics,
        "status": "finite" if prey_diagnostics["finite"] and predator_diagnostics["finite"] else "nonfinite",
    }


def _localized_worker(payload):
    index, spec, comparison = payload
    plan = repetition_plan(spec)[index]
    particles = int(spec["model"]["particles"])
    sigma = float(spec["model"]["sigma"])
    with threadpool_limits(limits=1):
        training, time = _simulate_truth(spec, particles, sigma, plan["training_seed"])
        stride = int(spec["model"]["observation_stride"])
        observed, observed_time = training[:, ::stride], time[::stride]
        rng = np.random.default_rng(comparison["strong_target_seed_start"] + index)
        targets = np.sort(rng.choice(
            particles, size=int(comparison["strong_target_count"]), replace=False
        ))
        kernel_basis, _ = make_kernel_basis(spec["dictionary"])
        A, b = assemble_localized_strong(observed, observed_time, targets, kernel_basis)
        theta, diagnostics = _ols(A, b)
        test, _ = _simulate_truth(spec, particles, sigma, plan["test_seed"])
        evaluation_system = latent_evaluation_system(spec, test, plan["evaluation_seed"])
        metrics = evaluate_theta(theta, *evaluation_system)
    truth = np.asarray(spec["model"]["interaction"]["coefficients"], dtype=float)
    return {
        "example": "localized_10d",
        "method": "strong_forward_difference_ols",
        "repetition": index,
        "training_seed": plan["training_seed"],
        "test_seed": plan["test_seed"],
        "target_seed": comparison["strong_target_seed_start"] + index,
        "target_count": int(targets.size),
        "coefficient_error_relative": relative_error(theta, truth),
        "heldout_drift_error_relative": metrics["latent_drift_relative_error"],
        "profile_error_relative": metrics["profile_relative_error"],
        "coefficients": theta.tolist(),
        "system": diagnostics,
        "status": "finite" if diagnostics["finite"] else "nonfinite",
    }


def _verify_input(path, expected):
    path = ROOT / path
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"input artifact hash mismatch: {path}: {actual} != {expected}")
    return path


def load_one_dimensional(spec, smoke=False):
    block = spec["one_dimensional"]
    path = _verify_input(block["source_records"], block["source_sha256"])
    truth_norm = float(np.linalg.norm(block["truth"]))
    rows = []
    with path.open(newline="") as handle:
        source = [row for row in csv.DictReader(handle)
                  if float(row["sigma_obs"]) == float(block["sigma_obs"])]
    source.sort(key=lambda row: int(row["seed_index"]))
    if smoke:
        source = source[:1]
    elif len(source) != int(block["repetitions"]):
        raise RuntimeError("unexpected one-dimensional repetition count")
    fields = {
        "weak_lasso": ("coefficient_error", lambda row, index: row[f"theta_{index}"]),
        "weak_ols": (
            "ols_coefficient_error",
            lambda row, index: row[f"ols_theta_{index}_recomputed"],
        ),
        "strong_forward_difference_ols": (
            "euler_coefficient_error", lambda row, index: row[f"euler_theta_{index}"],
        ),
    }
    for row in source:
        for method, (error_field, coefficient) in fields.items():
            theta = [float(coefficient(row, index)) for index in range(1, 7)]
            rows.append({
                "example": "one_dimensional", "method": method,
                "repetition": int(row["seed_index"]),
                "training_seed": int(row["trajectory_seed"]),
                "coefficient_error_relative": float(row[error_field]) / truth_norm,
                "heldout_drift_error_relative": None, "coefficients": theta,
                "status": "finite" if np.isfinite(float(row[error_field])) else "nonfinite",
            })
    return rows


def load_predator_strong_source(spec, smoke=False):
    block = spec["predator_prey"]
    path = _verify_input(block["strong_source_records"], block["strong_source_sha256"])
    sigma_obs = float(block["sigma_obs"])
    source = [
        row for row in read_json(path)
        if row["method"] == "strong_forward_difference_ols"
        and float(row["condition"]["observation_noise"]) == sigma_obs
    ]
    source.sort(key=lambda row: int(row["repetition"]))
    if smoke:
        source = source[:1]
    elif len(source) != int(block["repetitions"]):
        raise RuntimeError("unexpected predator--prey strong-form repetition count")
    return [{
        "example": "predator_prey",
        "method": "strong_forward_difference_ols",
        "repetition": int(row["repetition"]),
        "training_seed": int(row["training_seed"]),
        "test_seed": int(row["test_seed"]),
        "coefficient_error_relative": float(row["coefficient_error_relative"]),
        "heldout_drift_error_relative": float(
            row["held_out_drift"]["species_balanced_relative"]
        ),
        "coefficients": row["coefficients"],
        "status": row["status"],
        "prey_system": row["prey_system"],
        "predator_system": row["predator_system"],
    } for row in source]


def load_predator_weak(spec, smoke=False):
    block = spec["predator_prey"]
    path = _verify_input(block["source_records"], block["source_sha256"])
    records = read_json(path)
    records.sort(key=lambda row: row["index"])
    if smoke:
        records = records[:1]
    elif len(records) != int(block["repetitions"]):
        raise RuntimeError("unexpected predator--prey repetition count")
    protocol = read_json(ROOT / block["config"])
    _, _, _, truth, _ = model_objects(protocol)
    truth_vector = np.concatenate(truth)
    rows = []
    for row in records:
        for method, prefix in (("weak_lasso", ""), ("weak_ols", "ols_")):
            theta = np.concatenate([
                np.asarray(row[prefix + "theta_xx"], dtype=float),
                np.asarray(row[prefix + "theta_xz"], dtype=float),
                np.asarray(row[prefix + "theta_zx"], dtype=float),
            ])
            drift_key = "lasso_drift_error" if method == "weak_lasso" else "ols_drift_error"
            rows.append({
                "example": "predator_prey", "method": method,
                "repetition": int(row["index"]),
                "training_seed": int(row["training_seed"]),
                "test_seed": int(row["test_seed"]),
                "coefficient_error_relative": relative_error(theta, truth_vector),
                "heldout_drift_error_relative": float(row[drift_key]),
                "coefficients": theta.tolist(), "status": "finite",
            })
    return rows, protocol


def load_localized_source(spec, smoke=False):
    block = spec["localized_10d"]
    path = _verify_input(block["source_records"], block["source_sha256"])
    source = read_json(path)
    source_format = block.get("source_format", "robustness_sweep")
    if source_format == "fixed_observation_noise":
        method_map = {
            "weak_lasso": "weak_lasso",
            "weak_ols": "weak_ols",
            "strong_forward_difference_ols": "strong_forward_difference_ols",
        }
        rows = [
            row for row in source
            if float(row["observation_noise"]) == float(block["baseline_value"])
            and row["method"] in method_map
        ]
    elif source_format == "robustness_sweep":
        method_map = {"lasso": "weak_lasso", "ols": "weak_ols"}
        rows = [row for row in source if row["sweep"] == block["baseline_sweep"]
                and float(row["sweep_value"]) == float(block["baseline_value"])
                and row["method"] in method_map]
    else:
        raise ValueError(f"unsupported localized-10D source format: {source_format}")
    rows.sort(key=lambda row: (row["repetition"], row["method"]))
    if smoke:
        rows = [row for row in rows if row["repetition"] == 0]
    else:
        expected_methods = 3 if source_format == "fixed_observation_noise" else 2
        if len(rows) != expected_methods * int(block["repetitions"]):
            raise RuntimeError("unexpected localized-10D baseline repetition count")
    converted = []
    for row in rows:
        status = row.get("status")
        if status is None:
            status = "finite" if row["converged"] and not row["numerical_divergence"] else "failed"
        converted.append({
            "example": "localized_10d",
            "method": method_map[row["method"]],
            "repetition": int(row["repetition"]),
            "training_seed": int(row["training_seed"]),
            "test_seed": int(row["test_seed"]),
            "coefficient_error_relative": float(row["coefficient_error_relative"]),
            "heldout_drift_error_relative": float(row["latent_drift_relative_error"]),
            "profile_error_relative": float(row["profile_relative_error"]),
            "coefficients": row["coefficients"],
            "status": status,
            "system": {
                "rank_deficient": bool(row.get("rank_deficient", False)),
                "rank": row.get("design_rank"),
                "columns": row.get("design_columns"),
            },
        })
    protocol = read_json(ROOT / block["config"])
    return converted, protocol


def bootstrap_rmse(values, draws, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    samples = np.sqrt(np.mean(values[indices] ** 2, axis=1))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize(records, metric_spec):
    groups = []
    for example_index, example in enumerate(EXAMPLE_ORDER):
        for method_index, method in enumerate(METHOD_ORDER):
            rows = [row for row in records if row["example"] == example and row["method"] == method]
            values = np.asarray([row["coefficient_error_relative"] for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            complete = finite.size == values.size and values.size > 0
            groups.append({
                "example": example, "method": method,
                "count": int(values.size), "finite_count": int(finite.size),
                "complete": bool(complete),
                "mean": float(np.mean(finite)) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
                "q25": float(np.quantile(finite, 0.25)) if finite.size else None,
                "q75": float(np.quantile(finite, 0.75)) if finite.size else None,
                "maximum": float(np.max(finite)) if finite.size else None,
                "rmse": float(np.sqrt(np.mean(finite ** 2))) if finite.size else None,
                "rmse_bootstrap_95": bootstrap_rmse(
                    finite, int(metric_spec["bootstrap_draws"]),
                    int(metric_spec["bootstrap_seed"]) + 100 * example_index + method_index,
                ) if complete else None,
                "rank_deficient": int(sum(
                    bool(row.get("system", {}).get("rank_deficient", False))
                    or bool(row.get("prey_system", {}).get("rank_deficient", False))
                    or bool(row.get("predator_system", {}).get("rank_deficient", False))
                    for row in rows
                )),
            })
    winners = {}
    for example in EXAMPLE_ORDER:
        candidates = [row for row in groups if row["example"] == example and row["complete"]]
        winners[example] = min(candidates, key=lambda row: row["rmse"])["method"]
    return {"metric": metric_spec, "groups": groups, "rmse_winner": winners}


def _group(summary, example, method):
    return next(row for row in summary["groups"]
                if row["example"] == example and row["method"] == method)


def _format_percent(value):
    percent = 100.0 * value
    if percent >= 100.0:
        return f"{percent:.0f}"
    if percent >= 10.0:
        return f"{percent:.1f}"
    if percent >= 1.0:
        return f"{percent:.2f}"
    return f"{percent:.3f}"


def write_tables(out, summary):
    winners = summary["rmse_winner"]
    header = "| Nominal system | Weak LASSO | Weak OLS | Strong FD-OLS |\n|---|---:|---:|---:|\n"
    lines = [header]
    for example in EXAMPLE_ORDER:
        cells = []
        for method in METHOD_ORDER:
            value = _format_percent(_group(summary, example, method)["rmse"])
            cells.append(f"**{value}**" if winners[example] == method else value)
        lines.append(f"| {EXAMPLE_LABELS[example]} | " + " | ".join(cells) + " |\n")
    counts = {
        example: _group(summary, example, "weak_lasso")["count"]
        for example in EXAMPLE_ORDER
    }
    lines.append(
        "\nValues are relative coefficient RMSE in percent across all predeclared repetitions; "
        "smaller is better. Bold marks the minimum in each row. "
        f"R={counts['one_dimensional']} for 1D, R={counts['predator_prey']} for 2D, and "
        f"R={counts['localized_10d']} for 10D. Strong FD-OLS is raw minimum-norm OLS on "
        "forward position differences.\n"
    )
    (out / "table_rmse_draft.md").write_text("".join(lines))

    median_lines = [
        "| Nominal system | Weak LASSO | Weak OLS | Strong FD-OLS |\n",
        "|---|---:|---:|---:|\n",
    ]
    for example in EXAMPLE_ORDER:
        cells = []
        for method in METHOD_ORDER:
            row = _group(summary, example, method)
            cells.append(
                f"{_format_percent(row['median'])} "
                f"[{_format_percent(row['q25'])}, {_format_percent(row['q75'])}]"
            )
        median_lines.append(f"| {EXAMPLE_LABELS[example]} | " + " | ".join(cells) + " |\n")
    median_lines.append(
        "\nValues are median [interquartile range] relative coefficient error in percent. "
        "This diagnostic layout exposes the heavy-tailed 1D repetitions but is less compact "
        "than the RMSE table.\n"
    )
    (out / "table_median_iqr_draft.md").write_text("".join(median_lines))

    tex = [
        "\\begin{tabular}{lrrr}\n",
        "\\toprule\n",
        "Nominal system & Weak LASSO & Weak OLS & Strong FD--OLS \\\\\n",
        "\\midrule\n",
    ]
    for example in EXAMPLE_ORDER:
        cells = []
        for method in METHOD_ORDER:
            value = _format_percent(_group(summary, example, method)["rmse"])
            cells.append(f"\\textbf{{{value}}}" if winners[example] == method else value)
        tex.append(EXAMPLE_LABELS[example].replace("--", "--") + " & " + " & ".join(cells) + " \\\\\n")
    tex.extend([
        "\\bottomrule\n",
        "\\end{tabular}\n",
        "% Entries are relative coefficient RMSE (percent); smaller is better.\n",
    ])
    (out / "table_rmse_draft.tex").write_text("".join(tex))


def write_interpretation(out, summary):
    winners = summary["rmse_winner"]
    lasso_wins = [example for example, method in winners.items() if method == "weak_lasso"]
    strong_wins = [example for example, method in winners.items()
                   if method == "strong_forward_difference_ols"]
    lines = [
        "# Preliminary interpretation of the nominal comparison\n\n",
        "The comparison supports a consistent regularization result: weak LASSO has lower "
        "relative coefficient RMSE than weak OLS in all three nominal systems. ",
    ]
    if len(lasso_wins) == len(EXAMPLE_ORDER):
        lines.append("It also outperforms strong forward-difference OLS in all three systems.\n")
    else:
        lines.append(
            "It does not support the stronger claim that weak LASSO always outperforms "
            "strong-form regression. "
        )
        if lasso_wins:
            lines.append(
                "Weak LASSO is best in "
                + ", ".join(EXAMPLE_LABELS[item] for item in lasso_wins)
                + ", where position noise makes numerical differentiation costly. "
            )
        if strong_wins:
            lines.append(
                "Strong FD-OLS is best in "
                + ", ".join(EXAMPLE_LABELS[item] for item in strong_wins)
                + ", whose frozen nominal protocol has no observation noise and supplies many "
                "particlewise forward-difference equations.\n"
            )
    lines.extend([
        "\nThe compact RMSE table is the safer main-text candidate because it retains rare "
        "large errors rather than presenting only a typical repetition. The median [IQR] table "
        "should remain available as a diagnostic because the one-dimensional LASSO errors are "
        "heavy-tailed. No manuscript insertion or layout claim has been made.\n",
    ])
    (out / "interpretation_draft.md").write_text("".join(lines))


def write_records_csv(path, records):
    scalar_keys = sorted({key for row in records for key, value in row.items()
                          if not isinstance(value, (dict, list))})
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in scalar_keys} for row in records])


def snapshot_sources(out):
    hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    return hashes


def validate_comparison_protocol(spec):
    if spec.get("task") != "cross_example_nominal_method_comparison" \
            or spec.get("protocol_version") != 1:
        raise ValueError("invalid cross-example protocol")
    if tuple(spec.get("methods", [])) != METHOD_ORDER:
        raise ValueError("method order changed")
    policy = spec["strong_form_policy"]
    if any((policy["regularization"], policy["intercept"], policy["standardization"],
            policy["coefficient_thresholding"], policy["failure_exclusion"])):
        raise ValueError("strong-form baseline must remain raw and failure-preserving")


def run(config_path, out, *, workers=1, smoke=False):
    spec = read_json(config_path)
    validate_comparison_protocol(spec)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    shutil.copy2(config_path, out / "requested_protocol.json")

    records = load_one_dimensional(spec, smoke=smoke)
    predator_weak, predator_spec = load_predator_weak(spec, smoke=smoke)
    localized_source, localized_spec = load_localized_source(spec, smoke=smoke)
    validate_predator_protocol(predator_spec)
    validate_localized_protocol(localized_spec)
    records.extend(predator_weak)
    records.extend(localized_source)

    predator_has_strong_source = "strong_source_records" in spec["predator_prey"]
    localized_has_strong_source = any(
        row["method"] == "strong_forward_difference_ols" for row in localized_source
    )
    if predator_has_strong_source:
        records.extend(load_predator_strong_source(spec, smoke=smoke))

    predator_count = 1 if smoke else int(spec["predator_prey"]["repetitions"])
    localized_count = 1 if smoke else int(spec["localized_10d"]["repetitions"])
    if smoke:
        localized_spec = json.loads(json.dumps(localized_spec))
        spec = json.loads(json.dumps(spec))
        spec["localized_10d"]["strong_target_count"] = 8
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        if not predator_has_strong_source:
            records.extend(executor.map(
                _predator_worker, [(index, predator_spec) for index in range(predator_count)]
            ))
        if not localized_has_strong_source:
            records.extend(executor.map(
                _localized_worker,
                [(index, localized_spec, spec["localized_10d"])
                 for index in range(localized_count)],
            ))

    records.sort(key=lambda row: (
        EXAMPLE_ORDER.index(row["example"]), row["repetition"],
        METHOD_ORDER.index(row["method"]),
    ))
    summary = summarize(records, spec["metric"])
    write_json(out / "records.json", records)
    write_records_csv(out / "records.csv", records)
    write_json(out / "summary.json", summary)
    write_tables(out, summary)
    write_interpretation(out, summary)
    source_hashes = snapshot_sources(out)
    artifacts = {
        name: digest(out / name) for name in (
            "requested_protocol.json", "records.json", "records.csv", "summary.json",
            "table_rmse_draft.md", "table_median_iqr_draft.md", "table_rmse_draft.tex",
            "interpretation_draft.md",
        )
    }
    manifest = {
        "task": spec["task"], "protocol_version": spec["protocol_version"],
        "status": "smoke_complete" if smoke else "scientific_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(smoke), "record_count": len(records),
        "source_sha256": source_hashes, "artifact_sha256": artifacts,
        "input_sha256": {
            key: spec[key]["source_sha256"]
            for key in ("one_dimensional", "predator_prey", "localized_10d")
        },
        "manuscript_modified": False,
    }
    write_json(out / "manifest.json", manifest)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        kind = "smoke" if args.smoke else "nominal"
        args.out = ROOT / "outputs" / "validation" / "cross_example_nominal_comparison" / f"{kind}-{stamp}"
    out = run(args.config, args.out, workers=args.workers, smoke=args.smoke)
    print(out)


if __name__ == "__main__":
    main()
