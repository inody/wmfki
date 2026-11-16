"""Smoke-preview Figure 2 with sampled weak systems in panels (a)--(d).

This runner is intentionally isolated from the frozen scientific batches and the
manuscript tree.  Training and validation use independent ordered-pair samples,
with replacement, redrawn at every observed time.  A single draw at a given
time is shared by every spatial test and every dictionary atom through
``features.compute_A_timeweak``.  Panels (a)--(d) use P=10N.  Panel (e) selects
one absolute LASSO penalty at nominal N=200, P0=2000 using an independent
sampled validation system and then holds it fixed for every plotted P.  Panel
(f) reads the existing P=10N timing data without modifying it.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from features import compute_A_timeweak
from pipeline import fit_kernel_from_particles
from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR,
    REFERENCE_COLOR,
    SECONDARY_COLOR,
    TERTIARY_COLOR,
    TRUE_COLOR,
    apply_manuscript_style,
    finish_axes,
)
from scripts.run_full_pair_scaling import matched_config as full_matched_config
from scripts.run_qanr_matched_lasso_sweeps import (
    _fit_full,
    _selection_record,
    apply_dictionary,
    design_diagnostics,
    fit_fixed_alpha,
    read_json,
    validate_protocol,
)
from scripts.run_qanr_time_step_repetitions import simulate_paired_positions
from scripts.run_sindy_comparison import (
    DETERMINISTIC_SPACE_CENTERS,
    SPACE_TEST_WIDTH,
    TIME_TEST_SUPPORT,
    experiment_config as observation_config,
    fit_sindy_euler,
    reference_trajectory,
)
from simulate import simulate_particles
from utils import make_pair_sampler

DEFAULT_CONFIG = ROOT / "configs/experiments/qanr_redundant6_sampled_p10n_smoke.json"
SOURCE_FILES = (
    "configs/experiments/qanr_redundant6_sampled_p10n_smoke.json",
    "configs/experiments/qanr_redundant6_lasso_sweeps_R100.json",
    "scripts/run_qanr_sampled_p10n_preview.py",
    "scripts/run_qanr_matched_lasso_sweeps.py",
    "scripts/run_qanr_time_step_repetitions.py",
    "scripts/run_1d_hat_lasso.py",
    "scripts/run_full_pair_scaling.py",
    "scripts/run_random_pair_validation.py",
    "scripts/run_sindy_comparison.py",
    "scripts/r05_numerical_settings.py",
    "features.py",
    "pipeline.py",
    "utils.py",
    "kernels.py",
    "regression.py",
    "simulate.py",
    "tests.py",
    "time_tests.py",
)
PANEL_BOX_ASPECT = 3 / 4
PAIR_NAMESPACE = 0x5031304E


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def derived_seed(master, *coordinates):
    sequence = np.random.SeedSequence([int(master), PAIR_NAMESPACE, *map(int, coordinates)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def validate_preview_protocol(protocol, base):
    validate_protocol(base)
    if protocol.get("task") != "qanr_redundant6_sampled_p10n_smoke_preview":
        raise ValueError("unexpected preview task")
    if protocol.get("protocol_version") != 1 or protocol.get("smoke_only") is not True:
        raise ValueError("preview must remain protocol-version 1 and smoke-only")
    if protocol.get("repetitions") not in (1, 2):
        raise ValueError("smoke preview repetitions must be one or two")
    sampling = protocol.get("pair_sampling", {})
    required_sampling = {
        "panels_a_to_d_policy": "P_equals_10N_for_training_and_independent_validation",
        "ratio": 10,
        "mode": "resample_each_time",
        "with_replacement": True,
        "exclude_diagonal": True,
        "time_independent": True,
        "common_indices_across_all_tests_and_atoms": True,
    }
    if sampling != required_sampling:
        raise ValueError("sampled-pair contract changed")
    pair = protocol.get("pair_count", {})
    if pair.get("particles") != 200 or pair.get("nominal_selection_pairs") != 2000:
        raise ValueError("panel (e) must select at nominal N=200, P0=2000")
    if pair.get("penalty_policy") != (
        "one_absolute_alpha_selected_at_nominal_P0_from_independent_sampled_"
        "validation_then_fixed_for_every_P_and_pair_repetition"
    ):
        raise ValueError("panel (e) penalty policy changed")
    if protocol.get("timing", {}).get("sampled_policy") != (
        "retain_existing_P_equals_10N_panel_f_data"
    ):
        raise ValueError("panel (f) timing policy changed")
    runtime = ROOT / protocol["timing"]["runtime_csv"]
    if not runtime.is_file():
        raise FileNotFoundError(runtime)


def audit_existing_implementation(protocol):
    matched_path = ROOT / "scripts/run_qanr_matched_lasso_sweeps.py"
    feature_path = ROOT / "features.py"
    pipeline_path = ROOT / "pipeline.py"
    figure_path = ROOT / "scripts/generate_qanr_matched_lasso_overlay.py"
    matched = matched_path.read_text()
    features = feature_path.read_text()
    pipeline = pipeline_path.read_text()
    figure = figure_path.read_text()
    required_evidence = (
        "fit, A, b = _fit_full(config, observed, t)",
        "fit, A, b = _simulate_full_system(",
        "full_fit, A, b = _fit_full(train_config, X, t)",
        "pair_mode=block[\"pair_mode\"], pair_seed=pair_seed",
        "i_idx, j_idx = pair_sampler(N)",
        "psi = kernel_basis(z)",
        "gphi = test_fns[\"grad_phi\"](xi)",
        "resample_each_call=pair_mode == \"resample_each_time\"",
        '("resample_each_time", r"sampled, $P=10N$"',
    )
    combined = matched + features + pipeline + figure
    missing = [item for item in required_evidence if item not in combined]
    if missing:
        raise RuntimeError(f"existing implementation audit evidence changed: {missing}")
    return {
        "status": "completed",
        "sources": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (matched_path, feature_path, pipeline_path, figure_path)
        },
        "findings": [
            {
                "scope": "Figure 2 panels (a)--(d) existing matched-LASSO runner",
                "finding": "training and validation weak designs are assembled with exact full ordered pairs",
                "implication": "the existing runner cannot implement the requested sampled P=10N preview without a separate path",
            },
            {
                "scope": "Figure 2 panel (e) existing matched-LASSO runner",
                "finding": "the absolute penalty is selected from full-pair training and validation systems",
                "implication": "this differs from the requested nominal N=200, P0=2000 sampled-validation selection policy",
            },
            {
                "scope": "sampled feature assembly",
                "finding": "resample_each_time draws with replacement once per trajectory/time and reuses those indices for all spatial tests and dictionary atoms",
                "implication": "the core pipeline already satisfies the requested sampled-index semantics",
            },
            {
                "scope": "Figure 2 panel (f)",
                "finding": "the existing timing series labels and computes the sampled method at P=10N",
                "implication": "the preview can retain the archived timing CSV unchanged",
            },
        ],
        "existing_files_modified_by_preview": False,
        "production_integration_note": (
            "Replacing the manuscript/current Figure 2 would require an explicitly approved "
            "change to the active figure configuration or generator; this smoke preview does not do so."
        ),
        "runtime_csv": protocol["timing"]["runtime_csv"],
    }


def audit_pair_implementation():
    """Exercise the exact sampler/features path used by the preview."""
    draws = []
    base_sampler = make_pair_sampler(
        40,
        rng=np.random.default_rng(123),
        with_replacement=True,
        exclude_diagonal=True,
        resample_each_call=True,
    )

    def sampler(n):
        i_idx, j_idx = base_sampler(n)
        draws.append((i_idx.copy(), j_idx.copy()))
        return i_idx, j_idx

    x = np.linspace(-1.0, 1.0, 6)[None, None, :, None]
    x = np.repeat(x, 4, axis=1)
    time = np.linspace(0.0, 0.3, 4)

    def phi(values):
        v = values[:, 0]
        return np.column_stack((np.ones_like(v), v))

    def grad_phi(values):
        result = np.zeros((values.shape[0], 2, 1))
        result[:, 1, 0] = 1.0
        return result

    def basis(z):
        v = z[:, 0]
        result = np.empty((z.shape[0], 3, 1))
        result[:, 0, 0] = v
        result[:, 1, 0] = v * np.abs(v)
        result[:, 2, 0] = np.exp(-v * v) * v
        return result

    tests = {"phi": phi, "grad_phi": grad_phi}
    temporal = {"psi": np.ones((2, time.size))}
    design = compute_A_timeweak(x, time, tests, basis, sampler, temporal)
    expected_calls = x.shape[0] * x.shape[1]
    if design.shape != (1, 2, 2, 3) or len(draws) != expected_calls:
        raise RuntimeError("pair sampler was not called exactly once per trajectory/time")
    if any(i.size != 40 or np.any(i == j) for i, j in draws):
        raise RuntimeError("sampled pairs violate size or off-diagonal contract")
    if all(np.array_equal(draws[0][0], item[0]) and np.array_equal(draws[0][1], item[1])
           for item in draws[1:]):
        raise RuntimeError("pair draws were unexpectedly fixed across time")
    independent = make_pair_sampler(
        40,
        rng=np.random.default_rng(456),
        with_replacement=True,
        exclude_diagonal=True,
        resample_each_call=True,
    )(6)
    if np.array_equal(draws[0][0], independent[0]) and np.array_equal(
            draws[0][1], independent[1]):
        raise RuntimeError("independent pair streams unexpectedly coincide")
    return {
        "status": "passed",
        "sampler_calls": len(draws),
        "expected_sampler_calls": expected_calls,
        "pairs_per_call": 40,
        "with_replacement": True,
        "off_diagonal": True,
        "resampled_each_time": True,
        "one_sampler_call_shared_by_all_tests_and_atoms": True,
        "features_sha256": digest(ROOT / "features.py"),
        "pipeline_sha256": digest(ROOT / "pipeline.py"),
        "utils_sha256": digest(ROOT / "utils.py"),
    }


def sampled_config(config, pairs, pair_seed, protocol):
    copied = json.loads(json.dumps(config))
    copied["pair_sampling"].update({
        "total_pairs": int(pairs),
        "with_replacement": protocol["pair_sampling"]["with_replacement"],
        "exclude_diagonal": protocol["pair_sampling"]["exclude_diagonal"],
        "seed": int(pair_seed),
        "mode": protocol["pair_sampling"]["mode"],
    })
    return copied


def fit_sampled(config, positions, time, pairs, pair_seed, protocol):
    config = sampled_config(config, pairs, pair_seed, protocol)
    fit = fit_kernel_from_particles(
        config,
        positions,
        time,
        return_system=True,
        pair_mode=protocol["pair_sampling"]["mode"],
        pair_seed=pair_seed,
    )
    return fit, np.asarray(fit["system"]["A_flat"]), np.asarray(fit["system"]["b_flat"])


def simulate_system(block, particles, sigma, seed, base, repetitions=1, dt=None):
    step = float(block["dt"] if dt is None else dt)
    steps = int(round(float(block["horizon"]) / step))
    config = full_matched_config(
        particles=int(particles),
        repetitions=int(repetitions),
        steps=steps,
        dt=step,
        sigma=float(sigma),
        seed=int(seed),
        solver=block["solver"],
        time_test_type="bump",
    )
    config = apply_dictionary(config, base)
    positions, time, *_ = simulate_particles(
        N=int(particles),
        T=steps,
        dt=step,
        R=int(repetitions),
        sigma=float(sigma),
        d=1,
        init="gaussian_mixture",
        init_centers=[0.5, 2.0, 3.5],
        init_std=0.005,
        init_scale=1.0,
        kernel_params=config["simulation"]["interaction"],
        seed=int(seed),
        solver=block["solver"],
    )
    return config, positions, time


def selected_record(fit, A, b, Av, bv, estimator, truth):
    _, theta, selected = _selection_record(A, b, Av, bv, estimator)
    ols = np.asarray(fit["theta_hat"], dtype=float)
    norm = float(np.linalg.norm(truth))
    return {
        "lasso_coefficients": theta.tolist(),
        "ols_coefficients": ols.tolist(),
        "lasso_relative_error": float(np.linalg.norm(theta - truth) / norm),
        "ols_relative_error": float(np.linalg.norm(ols - truth) / norm),
        **design_diagnostics(A),
        **selected,
    }


def panel_pair_seeds(protocol, trajectory_seed, panel_code, repetition, condition_index):
    master = protocol["pair_seed_offset"] + int(trajectory_seed)
    train = derived_seed(master, panel_code, repetition, condition_index, 0)
    validation = derived_seed(
        master + protocol["pair_validation_seed_offset"],
        panel_code,
        repetition,
        condition_index,
        1,
    )
    if train == validation:
        raise RuntimeError("training and validation pair seeds must differ")
    return train, validation


def run_observation(protocol, base):
    block = protocol["observation_noise"]
    truth = np.asarray(base["true_theta"], dtype=float)
    pairs = protocol["pair_sampling"]["ratio"] * block["particles"]
    records = []
    for repetition in range(protocol["repetitions"]):
        train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
        validation_seed = train_seed + protocol["validation_seed_offset"]
        X, time, *_ = reference_trajectory(train_seed, step=block["dt"], sigma=block["sigma"])
        Xv, tv, *_ = reference_trajectory(
            validation_seed, step=block["dt"], sigma=block["sigma"]
        )
        train_noise = np.random.default_rng(train_seed + 2).normal(size=X.shape)
        validation_noise = np.random.default_rng(validation_seed + 2).normal(size=Xv.shape)
        steps = int(round(block["horizon"] / block["dt"]))
        for condition_index, sigma_obs in enumerate(block["values"]):
            config = observation_config(
                block["particles"], steps, block["dt"], block["sigma"], train_seed,
                DETERMINISTIC_SPACE_CENTERS, SPACE_TEST_WIDTH, TIME_TEST_SUPPORT, "bump",
            )
            validation_config = observation_config(
                block["particles"], steps, block["dt"], block["sigma"], validation_seed,
                DETERMINISTIC_SPACE_CENTERS, SPACE_TEST_WIDTH, TIME_TEST_SUPPORT, "bump",
            )
            config, validation_config = apply_dictionary(config, base), apply_dictionary(
                validation_config, base
            )
            observed = X + float(sigma_obs) * train_noise
            observed_validation = Xv + float(sigma_obs) * validation_noise
            train_pair_seed, validation_pair_seed = panel_pair_seeds(
                protocol, train_seed, 1, repetition, condition_index
            )
            fit, A, b = fit_sampled(
                config, observed, time, pairs, train_pair_seed, protocol
            )
            _, Av, bv = fit_sampled(
                validation_config, observed_validation, tv, pairs,
                validation_pair_seed, protocol,
            )
            record = selected_record(fit, A, b, Av, bv, base["estimator"], truth)
            euler, _, _, _ = fit_sindy_euler(
                config, observed, time, fit["system"]["kernel_basis"]
            )
            records.append({
                "panel": "observation_noise", "repetition": repetition,
                "value": float(sigma_obs), "particles": block["particles"],
                "P": pairs, "training_seed": train_seed,
                "validation_seed": validation_seed,
                "training_pair_seed": train_pair_seed,
                "validation_pair_seed": validation_pair_seed,
                "euler_relative_error": float(
                    np.linalg.norm(np.asarray(euler) - truth) / np.linalg.norm(truth)
                ),
                **record,
            })
    return records


def run_stochastic(protocol, base, name, panel_code):
    block = protocol[name]
    truth = np.asarray(base["true_theta"], dtype=float)
    records = []
    for repetition in range(protocol["repetitions"]):
        train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
        validation_seed = train_seed + protocol["validation_seed_offset"]
        for condition_index, value in enumerate(block["values"]):
            if name == "process_noise":
                particles, sigma = block["particles"], float(value)
            else:
                particles, sigma = int(value), float(block["sigma"])
            pairs = protocol["pair_sampling"]["ratio"] * particles
            config, X, time = simulate_system(
                block, particles, sigma, train_seed, base
            )
            validation_config, Xv, tv = simulate_system(
                block, particles, sigma, validation_seed, base
            )
            train_pair_seed, validation_pair_seed = panel_pair_seeds(
                protocol, train_seed, panel_code, repetition, condition_index
            )
            fit, A, b = fit_sampled(
                config, X, time, pairs, train_pair_seed, protocol
            )
            _, Av, bv = fit_sampled(
                validation_config, Xv, tv, pairs, validation_pair_seed, protocol
            )
            records.append({
                "panel": name, "repetition": repetition, "value": float(value),
                "particles": particles, "P": pairs,
                "training_seed": train_seed, "validation_seed": validation_seed,
                "training_pair_seed": train_pair_seed,
                "validation_pair_seed": validation_pair_seed,
                **selected_record(fit, A, b, Av, bv, base["estimator"], truth),
            })
    return records


def run_time_step(protocol, base):
    block = protocol["time_step"]
    truth = np.asarray(base["true_theta"], dtype=float)
    pairs = protocol["pair_sampling"]["ratio"] * block["particles"]
    records = []
    for repetition in range(protocol["repetitions"]):
        train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
        validation_seed = train_seed + protocol["validation_seed_offset"]
        for condition_index, dt in enumerate(block["values"]):
            common = {
                "particles": block["particles"],
                "repetitions": block["stacked_trajectories"],
                "horizon": block["horizon"],
                "dt": float(dt),
                "sigma": block["sigma"],
                "solver": block["solver"],
                "spec": base,
                "policy": {
                    "namespace": 0x51414E52,
                    "name": "seedsequence_master_trajectory_namespace_v1",
                },
            }
            config, X, time, train_children, train_hashes = simulate_paired_positions(
                master_seed=train_seed, **common
            )
            validation_config, Xv, tv, validation_children, validation_hashes = (
                simulate_paired_positions(master_seed=validation_seed, **common)
            )
            train_pair_seed, validation_pair_seed = panel_pair_seeds(
                protocol, train_seed, 4, repetition, condition_index
            )
            fit, A, b = fit_sampled(
                config, X, time, pairs, train_pair_seed, protocol
            )
            _, Av, bv = fit_sampled(
                validation_config, Xv, tv, pairs, validation_pair_seed, protocol
            )
            records.append({
                "panel": "time_step", "repetition": repetition,
                "value": float(dt), "particles": block["particles"], "P": pairs,
                "training_seed": train_seed, "validation_seed": validation_seed,
                "training_pair_seed": train_pair_seed,
                "validation_pair_seed": validation_pair_seed,
                "training_trajectory_seeds": train_children,
                "validation_trajectory_seeds": validation_children,
                "training_initial_state_sha256": train_hashes,
                "validation_initial_state_sha256": validation_hashes,
                **selected_record(fit, A, b, Av, bv, base["estimator"], truth),
            })
    return records


def run_pair_count(protocol, base):
    block = protocol["pair_count"]
    particles = int(block["particles"])
    train_seed = int(block["training_seed"])
    validation_seed = train_seed + protocol["validation_seed_offset"]
    truth = np.asarray(base["true_theta"], dtype=float)
    sim_block = {**block, "solver": "euler"}
    config, X, time = simulate_system(
        sim_block, particles, block["sigma"], train_seed, base
    )
    validation_config, Xv, tv = simulate_system(
        sim_block, particles, block["sigma"], validation_seed, base
    )
    p0 = int(block["nominal_selection_pairs"])
    selection_train_pair_seed = derived_seed(
        protocol["pair_seed_offset"] + train_seed, 5, 0, p0, 0
    )
    selection_validation_pair_seed = derived_seed(
        protocol["pair_seed_offset"] + train_seed
        + protocol["pair_validation_seed_offset"], 5, 0, p0, 1
    )
    _, A0, b0 = fit_sampled(
        config, X, time, p0, selection_train_pair_seed, protocol
    )
    _, Av0, bv0 = fit_sampled(
        validation_config, Xv, tv, p0, selection_validation_pair_seed, protocol
    )
    selection, theta_p0, selected = _selection_record(
        A0, b0, Av0, bv0, base["estimator"]
    )
    alpha = selected["alpha_sk"]
    full_fit, A_full, b_full = _fit_full(config, X, time)
    theta_full_lasso, full_convergence = fit_fixed_alpha(
        A_full, b_full, alpha, base["estimator"]
    )
    theta_full_ols = np.asarray(full_fit["theta_hat"], dtype=float)
    rows = []
    for pair_repetition in range(block["pair_repetitions"]):
        for condition_index, pairs in enumerate(block["values"]):
            pair_seed = derived_seed(
                protocol["pair_seed_offset"] + train_seed,
                5, pair_repetition + 1, condition_index, pairs,
            )
            fit, A, b = fit_sampled(
                config, X, time, int(pairs), pair_seed, protocol
            )
            theta, convergence = fit_fixed_alpha(A, b, alpha, base["estimator"])
            ols = np.asarray(fit["theta_hat"], dtype=float)
            rows.append({
                "panel": "pair_count", "pair_repetition": pair_repetition,
                "value": int(pairs), "P": int(pairs), "particles": particles,
                "pair_seed": pair_seed, "alpha_sk": float(alpha),
                "lasso_relative_mse_to_full": float(
                    np.sum((theta - theta_full_lasso) ** 2) / np.sum(truth ** 2)
                ),
                "ols_relative_mse_to_full": float(
                    np.sum((ols - theta_full_ols) ** 2) / np.sum(truth ** 2)
                ),
                "lasso_coefficients": theta.tolist(),
                "ols_coefficients": ols.tolist(),
                **design_diagnostics(A),
                **convergence,
            })
    return rows, {
        "nominal_particles": particles,
        "nominal_selection_pairs": p0,
        "training_seed": train_seed,
        "validation_seed": validation_seed,
        "selection_training_pair_seed": selection_train_pair_seed,
        "selection_validation_pair_seed": selection_validation_pair_seed,
        "selected_absolute_alpha": float(alpha),
        "selection": selection,
        "theta_at_selection_P0": theta_p0.tolist(),
        "full_pair_lasso_reference": theta_full_lasso.tolist(),
        "full_pair_ols_reference": theta_full_ols.tolist(),
        "full_pair_lasso_convergence": full_convergence,
        "fixed_for_all_P_and_pair_repetitions": True,
    }


def rmse_ci(values):
    values = np.asarray(values, dtype=float)
    estimate = float(np.sqrt(np.mean(values ** 2)))
    if values.size < 2 or estimate == 0.0:
        interval = [estimate, estimate]
    else:
        se = float(np.std(values ** 2, ddof=1) / np.sqrt(values.size) / (2 * estimate))
        interval = [max(0.0, estimate - 1.96 * se), estimate + 1.96 * se]
    return {"rmse": estimate, "ci95": interval, "count": int(values.size)}


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        interval = [mean, mean]
    else:
        half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
        interval = [max(0.0, mean - half), mean + half]
    return {"mean": mean, "ci95": interval, "count": int(values.size)}


def summarize(records, pair_rows):
    summary = {name: [] for name in (
        "observation_noise", "process_noise", "particle_number", "time_step"
    )}
    for panel in summary:
        panel_rows = [row for row in records if row["panel"] == panel]
        for value in sorted({float(row["value"]) for row in panel_rows}):
            subset = [row for row in panel_rows if float(row["value"]) == value]
            entry = {
                "value": value,
                "particles": int(subset[0]["particles"]),
                "P": int(subset[0]["P"]),
                "lasso": rmse_ci([row["lasso_relative_error"] for row in subset]),
                "ols": rmse_ci([row["ols_relative_error"] for row in subset]),
            }
            if panel == "observation_noise":
                entry["euler"] = rmse_ci(
                    [row["euler_relative_error"] for row in subset]
                )
            summary[panel].append(entry)
    summary["pair_count"] = []
    for pairs in sorted({int(row["P"]) for row in pair_rows}):
        subset = [row for row in pair_rows if int(row["P"]) == pairs]
        summary["pair_count"].append({
            "P": pairs,
            "lasso": mean_ci([row["lasso_relative_mse_to_full"] for row in subset]),
            "ols": mean_ci([row["ols_relative_mse_to_full"] for row in subset]),
        })
    return summary


def verify_results(protocol, records, pair_rows, pair_selection, audit):
    issues = []
    ratio = protocol["pair_sampling"]["ratio"]
    for row in records:
        if int(row["P"]) != ratio * int(row["particles"]):
            issues.append(f"P != 10N: {row['panel']} value={row['value']}")
        if int(row["training_pair_seed"]) == int(row["validation_pair_seed"]):
            issues.append(f"shared train/validation pair seed: {row['panel']}")
        for key in ("lasso_relative_error", "ols_relative_error", "alpha_sk"):
            if not np.isfinite(float(row[key])):
                issues.append(f"nonfinite {key}: {row['panel']}")
    selected_alpha = float(pair_selection["selected_absolute_alpha"])
    for row in pair_rows:
        if not np.isclose(float(row["alpha_sk"]), selected_alpha, rtol=0.0, atol=0.0):
            issues.append("panel (e) alpha was not fixed exactly")
        if not np.isfinite(float(row["lasso_relative_mse_to_full"])):
            issues.append("panel (e) contains a nonfinite LASSO metric")
    if pair_selection["nominal_selection_pairs"] != 2000:
        issues.append("panel (e) nominal P0 is not 2000")
    if pair_selection["selection_training_pair_seed"] == pair_selection[
            "selection_validation_pair_seed"]:
        issues.append("panel (e) selection pair streams are not independent")
    if audit.get("status") != "passed":
        issues.append("pair implementation audit did not pass")
    expected = protocol["repetitions"] * sum(len(protocol[name]["values"]) for name in (
        "observation_noise", "process_noise", "particle_number", "time_step"
    ))
    if len(records) != expected:
        issues.append(f"panel (a)--(d) record count {len(records)} != {expected}")
    expected_pair = (
        protocol["pair_count"]["pair_repetitions"]
        * len(protocol["pair_count"]["values"])
    )
    if len(pair_rows) != expected_pair:
        issues.append(f"panel (e) record count {len(pair_rows)} != {expected_pair}")
    result = {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "panels_a_to_d_models": len(records),
        "panel_e_models": len(pair_rows),
        "all_panels_a_to_d_use_P_equals_10N": not any("P != 10N" in x for x in issues),
        "independent_training_validation_pair_streams": not any(
            "pair seed" in x or "pair streams" in x for x in issues
        ),
        "panel_e_fixed_absolute_alpha": not any("alpha" in x for x in issues),
        "pair_sampling_contract_audit": audit,
    }
    if issues:
        raise ValueError("; ".join(issues))
    return result


def add_rmse(axis, rows, key, label, color, marker, linestyle):
    x = np.asarray([row["value"] for row in rows], dtype=float)
    y = np.asarray([row[key]["rmse"] for row in rows])
    lower = np.asarray([row[key]["ci95"][0] for row in rows])
    upper = np.asarray([row[key]["ci95"][1] for row in rows])
    positive = x > 0
    axis.loglog(x[positive], y[positive], marker=marker, linestyle=linestyle,
                color=color, label=label)
    if np.any(upper[positive] > lower[positive]):
        axis.fill_between(x[positive], np.maximum(lower[positive], np.finfo(float).tiny),
                          upper[positive], color=color, alpha=.10, linewidth=0)
    if np.any(~positive):
        axis.axhline(y[~positive][0], color=color, linestyle=":", linewidth=.9)


def generate_preview(protocol, summary, runtime_csv, destination):
    runtime = read_csv(runtime_csv)
    apply_manuscript_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.35), constrained_layout=True)
    axes = axes.ravel()

    observation = summary["observation_noise"]
    for key, label, color, marker, style in (
        ("lasso", "weak LASSO", ESTIMATE_COLOR, "^", "--"),
        ("ols", "weak OLS", TRUE_COLOR, "o", "-"),
        ("euler", "strong FD-OLS", TERTIARY_COLOR, "s", "-."),
    ):
        add_rmse(axes[0], observation, key, label, color, marker, style)
    positive = [row["value"] for row in observation if row["value"] > 0]
    axes[0].set_xlim(min(positive) / 1.6, max(positive) * 1.6)
    axes[0].set(xlabel=r"observation noise $\sigma_{\rm obs}$",
                ylabel=r"relative $\operatorname{RMSE}_{\theta}$",
                title="(a) Observation noise")
    axes[0].legend(loc="upper left", fontsize=6.2)

    process = summary["process_noise"]
    add_rmse(axes[1], process, "lasso", "weak LASSO", ESTIMATE_COLOR, "^", "--")
    add_rmse(axes[1], process, "ols", "weak OLS", TRUE_COLOR, "o", "-")
    process_values = [row["value"] for row in process]
    axes[1].set_xticks(process_values, [f"{value:g}" for value in process_values])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set(xlabel=r"process noise $\sigma$",
                ylabel=r"relative $\operatorname{RMSE}_{\theta}$",
                title="(b) Process noise")
    axes[1].legend(loc="best", fontsize=6.5)

    particles = summary["particle_number"]
    add_rmse(axes[2], particles, "lasso", "weak LASSO", ESTIMATE_COLOR, "^", "--")
    add_rmse(axes[2], particles, "ols", "weak OLS", TRUE_COLOR, "o", "-")
    particle_values = [int(row["value"]) for row in particles]
    axes[2].set_xticks(particle_values, [str(value) for value in particle_values])
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].set(xlabel=r"particles $N$", ylabel=r"relative $\operatorname{RMSE}_{\theta}$",
                title="(c) Particle number")
    axes[2].legend(loc="best", fontsize=6.5)

    dt_rows = summary["time_step"]
    add_rmse(axes[3], dt_rows, "lasso", "weak LASSO", ESTIMATE_COLOR, "^", "--")
    add_rmse(axes[3], dt_rows, "ols", "weak OLS", TRUE_COLOR, "o", "-")
    dt_values = [row["value"] for row in dt_rows]
    axes[3].set_xticks(dt_values, [f"{value:g}" for value in dt_values])
    axes[3].xaxis.set_minor_formatter(NullFormatter())
    axes[3].set(xlabel=r"observation interval $\Delta t$",
                ylabel=r"relative $\operatorname{RMSE}_{\theta}$",
                title="(d) Observation interval")
    axes[3].legend(loc="best", fontsize=6.5)

    pair = summary["pair_count"]
    p = np.asarray([row["P"] for row in pair], dtype=float)
    pair_means = np.asarray([
        row[key]["mean"] for row in pair for key in ("lasso", "ols")
    ])
    display_floor = float(np.min(pair_means[pair_means > 0])) / 5.0
    for key, label, color, marker, style in (
        ("lasso", "weak LASSO", ESTIMATE_COLOR, "^", "--"),
        ("ols", "weak OLS", TRUE_COLOR, "o", "-"),
    ):
        mean = np.asarray([row[key]["mean"] for row in pair])
        lower = np.asarray([row[key]["ci95"][0] for row in pair])
        upper = np.asarray([row[key]["ci95"][1] for row in pair])
        axes[4].loglog(p, mean, marker=marker, linestyle=style, color=color, label=label)
        axes[4].fill_between(p, np.maximum(lower, display_floor), upper,
                             color=color, alpha=.12, linewidth=0)
    axes[4].set(xlabel=r"sampled pairs $P$",
                ylabel="relative mean squared\ncoefficient difference",
                title="(e) Effect of pair sampling")
    axes[4].legend(loc="best", fontsize=6.5)

    ns = sorted({int(row["N"]) for row in runtime})
    for mode, label, color, marker, style in (
        ("full", "all ordered pairs", TRUE_COLOR, "o", "-"),
        ("resample_each_time", r"sampled, $P=10N$", SECONDARY_COLOR, "s", "--"),
    ):
        values = [[float(row["design_seconds"]) for row in runtime
                   if row["mode"] == mode and int(row["N"]) == n] for n in ns]
        axes[5].loglog(ns, [np.median(item) for item in values], marker=marker,
                       linestyle=style, color=color, label=label)
    axes[5].set(xlabel=r"particles $N$", ylabel="feature time [s]",
                title="(f) Feature computation")
    axes[5].legend(loc="upper left", fontsize=6.5)

    for axis in axes:
        axis.set_box_aspect(PANEL_BOX_ASPECT)
        finish_axes(axis)
    fig.suptitle(protocol["preview"]["watermark"], fontsize=8, color=REFERENCE_COLOR)
    destination.mkdir(parents=True, exist_ok=True)
    stem = protocol["preview"]["filename_stem"]
    pdf, png = destination / f"{stem}.pdf", destination / f"{stem}.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    return pdf, png


def snapshot_sources(out, config_path, base_path):
    source_dir = Path(out) / "source"
    hashes = {}
    files = list(SOURCE_FILES)
    for path in (Path(config_path).resolve(), Path(base_path).resolve()):
        relative = str(path.relative_to(ROOT))
        if relative not in files:
            files.append(relative)
    for relative in files:
        source = ROOT / relative
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    return hashes


def run(config_path, out):
    protocol = read_json(config_path)
    base_path = ROOT / protocol["base_protocol"]
    base = read_json(base_path)
    validate_preview_protocol(protocol, base)
    existing_audit = audit_existing_implementation(protocol)
    audit = audit_pair_implementation()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "requested_protocol.json", protocol)
    write_json(out / "existing_implementation_audit.json", existing_audit)
    source_hashes = snapshot_sources(out, config_path, base_path)
    started = perf_counter()
    manifest = {
        "task": protocol["task"], "status": "running", "smoke_only": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes,
        "dirty_worktree_preserved": True,
        "existing_frozen_results_overwritten": False,
        "manuscript_or_current_figures_overwritten": False,
    }
    write_json(out / "manifest.json", manifest)
    with threadpool_limits(limits=1):
        records = run_observation(protocol, base)
        records.extend(run_stochastic(protocol, base, "process_noise", 2))
        records.extend(run_stochastic(protocol, base, "particle_number", 3))
        records.extend(run_time_step(protocol, base))
        pair_rows, pair_selection = run_pair_count(protocol, base)
    records.sort(key=lambda row: (row["panel"], row["repetition"], row["value"]))
    pair_rows.sort(key=lambda row: (row["pair_repetition"], row["P"]))
    summary = summarize(records, pair_rows)
    verification = verify_results(protocol, records, pair_rows, pair_selection, audit)
    write_json(out / "records_a_to_d.json", records)
    write_csv(out / "records_a_to_d.csv", records)
    write_json(out / "records_e.json", pair_rows)
    write_csv(out / "records_e.csv", pair_rows)
    write_json(out / "panel_e_selection.json", pair_selection)
    write_json(out / "summary.json", summary)
    write_json(out / "verification.json", verification)
    runtime_csv = ROOT / protocol["timing"]["runtime_csv"]
    preview_dir = out / "preview"
    pdf, png = generate_preview(protocol, summary, runtime_csv, preview_dir)
    manifest.update({
        "status": "smoke_completed",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "counts": {"panels_a_to_d": len(records), "panel_e": len(pair_rows)},
        "runtime_panel_f": {
            "path": str(runtime_csv.relative_to(ROOT)), "sha256": digest(runtime_csv),
            "data_unchanged": True, "sampled_policy": "P=10N",
        },
        "artifact_sha256": {
            str(path.relative_to(out)): digest(path)
            for path in (
                out / "requested_protocol.json",
                out / "existing_implementation_audit.json",
                out / "records_a_to_d.json",
                out / "records_a_to_d.csv", out / "records_e.json",
                out / "records_e.csv", out / "panel_e_selection.json",
                out / "summary.json", out / "verification.json", pdf, png,
            )
        },
        "preview_pdf": str(pdf), "preview_png": str(png),
    })
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = (
            ROOT / "outputs/validation/qanr_sampled_p10n_preview" / f"smoke-{stamp}"
        )
    manifest = run(args.config.resolve(), args.output)
    print(json.dumps({
        "status": manifest["status"],
        "output": str(Path(args.output).resolve()),
        "preview_png": manifest["preview_png"],
    }, indent=2))


if __name__ == "__main__":
    main()
