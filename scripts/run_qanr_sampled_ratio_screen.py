"""Screen sampled-pair ratios for the six-atom Figure 2(a--d) protocol.

The frozen/current manuscript artifacts are read-only inputs.  This exploratory
R=5 runner uses representative points from the current grids, independent
training/validation trajectories and pair streams, P=ratio*N, and LASSO penalty
selection on the sampled validation system at the same ratio.  Pair draws are
with replacement, independent across contributing observation times, nested
across ratios, and shared by all spatial tests and dictionary atoms.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import warnings

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from pipeline import fit_kernel_from_particles
from regression import fit_theta
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
    _selection_record,
    apply_dictionary,
    read_json,
    validate_protocol,
)
from scripts.run_qanr_time_step_repetitions import (
    SEED_POLICY,
    simulate_paired_positions,
    validate_time_protocol,
)
from scripts.run_sindy_comparison import (
    DETERMINISTIC_SPACE_CENTERS,
    SPACE_TEST_WIDTH,
    TIME_TEST_SUPPORT,
    experiment_config as observation_config,
    fit_sindy_euler,
    reference_trajectory,
)
from simulate import simulate_particles

DEFAULT_CONFIG = ROOT / "configs/experiments/qanr_redundant6_sampled_ratio_screen_R5.json"
PAIR_NAMESPACE = 0x5235504E
PANEL_CODES = {
    "observation_noise": 1,
    "process_noise": 2,
    "particle_number": 3,
    "time_step": 4,
}
SOURCE_FILES = (
    "configs/experiments/qanr_redundant6_sampled_ratio_screen_R5.json",
    "configs/experiments/qanr_redundant6_lasso_sweeps_R100.json",
    "configs/experiments/qanr_redundant6_time_step_paired_R100.json",
    "scripts/run_qanr_sampled_ratio_screen.py",
    "scripts/run_qanr_matched_lasso_sweeps.py",
    "scripts/run_qanr_time_step_repetitions.py",
    "scripts/run_1d_hat_lasso.py",
    "scripts/run_full_pair_scaling.py",
    "scripts/run_sindy_comparison.py",
    "scripts/r05_numerical_settings.py",
    "features.py", "pipeline.py", "utils.py", "kernels.py", "regression.py",
    "simulate.py", "tests.py", "time_tests.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    columns = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def derived_seed(master, *coordinates):
    sequence = np.random.SeedSequence([int(master), PAIR_NAMESPACE, *map(int, coordinates)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def validate_screen_protocol(protocol, base, time_protocol):
    validate_protocol(base)
    validate_time_protocol(time_protocol, base)
    if protocol.get("task") != "qanr_redundant6_sampled_ratio_screen_R5":
        raise ValueError("unexpected ratio-screen task")
    if protocol.get("protocol_version") != 1 or protocol.get("repetitions") != 5:
        raise ValueError("ratio screen must remain protocol-version 1 with R=5")
    if protocol.get("ratios") != [20, 30, 50, 75]:
        raise ValueError("ratio grid changed")
    expected_sampling = {
        "mode": "nested_prefix_resample_each_time",
        "with_replacement": True,
        "exclude_diagonal": True,
        "training_validation_independent": True,
        "time_independent": True,
        "common_indices_across_tests_and_atoms": True,
        "pairs_per_condition": "P=ratio*N",
        "penalty_selection": "independent_sampled_validation_at_the_same_ratio",
    }
    if protocol.get("sampling") != expected_sampling:
        raise ValueError("sampled-pair contract changed")
    current = {
        "observation_noise": base["observation_noise"]["sigma_obs_values"],
        "process_noise": base["process_noise"]["values"],
        "particle_number": base["particle_number"]["values"],
        "time_step": time_protocol["time_step"]["values"],
    }
    for panel, values in protocol["grids"].items():
        if panel in current and any(value not in current[panel] for value in values):
            raise ValueError(f"{panel} representative grid is outside the current grid")
    if protocol["acceptance"]["max_lasso_rmse_inflation_over_paired_full"] != 1.25:
        raise ValueError("acceptance inflation threshold changed")
    for key in ("matched_full_pair_batch", "matched_full_pair_time_step_batch"):
        if not (ROOT / protocol[key]).is_dir():
            raise FileNotFoundError(ROOT / protocol[key])


def snapshot_sources(out, config_path, base_path, time_path):
    hashes = {}
    files = list(SOURCE_FILES)
    for path in (config_path, base_path, time_path):
        relative = str(Path(path).resolve().relative_to(ROOT))
        if relative not in files:
            files.append(relative)
    for relative in files:
        source = ROOT / relative
        target = Path(out) / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    return hashes


def theta_columns(row, prefix):
    return np.asarray([float(row[f"{prefix}{index}"]) for index in range(1, 7)])


def load_full_lookup(protocol):
    batch = ROOT / protocol["matched_full_pair_batch"]
    observation = read_csv(batch / "observation_lasso.csv")
    stochastic = read_csv(batch / "stochastic_lasso.csv")
    time_records = read_json(
        ROOT / protocol["matched_full_pair_time_step_batch"] / "records.json"
    )
    return {
        "observation_noise": {
            (int(row["seed_index"]), float(row["sigma_obs"])): row
            for row in observation
        },
        "process_noise": {
            (int(row["seed_index"]), float(row["value"])): row
            for row in stochastic if row["kind"] == "sigma"
        },
        "particle_number": {
            (int(row["seed_index"]), int(float(row["value"]))): row
            for row in stochastic if row["kind"] == "particles"
        },
        "time_step": {
            (int(row["repetition"]), float(row["dt"])): row
            for row in time_records
        },
    }


def full_reference(lookup, panel, repetition, value, truth_norm):
    row = lookup[panel][(repetition, int(value) if panel == "particle_number" else float(value))]
    if panel == "time_step":
        return {
            "lasso": {
                "theta": np.asarray(row["coefficients"], dtype=float),
                "relative_error": float(row["coefficient_error_relative"]),
            },
            "ols": {
                "theta": np.asarray(row["ols_coefficients"], dtype=float),
                "relative_error": float(row["ols_coefficient_error_relative"]),
            },
            "diagnostics": {
                "lasso_unconverged_candidates": int(row["unconverged_candidates"]),
                "lasso_boundary": bool(row["selected_at_grid_boundary"]),
            },
        }
    result = {
        "lasso": {
            "theta": theta_columns(row, "theta_"),
            "relative_error": float(row["coefficient_error"]) / truth_norm,
        },
        "ols": {
            "theta": np.asarray([
                float(row[f"ols_theta_{index}_recomputed"]) for index in range(1, 7)
            ]),
            "relative_error": float(row["ols_coefficient_error"]) / truth_norm,
        },
        "diagnostics": {
            "lasso_unconverged_candidates": int(row["unconverged_candidates"]),
            "lasso_boundary": str(row["selected_at_grid_boundary"]).lower() == "true",
        },
    }
    if panel == "observation_noise":
        result["strong_fd_ols"] = {
            "theta": np.asarray([float(row[f"euler_theta_{index}"]) for index in range(1, 7)]),
            "relative_error": float(row["euler_coefficient_error"]) / truth_norm,
        }
    return result


def matched_config(block, particles, sigma, seed, base, *, dt=None, repetitions=1):
    step = float(block["dt"] if dt is None else dt)
    steps = int(round(float(block["horizon"]) / step))
    config = full_matched_config(
        particles=int(particles), repetitions=int(repetitions), steps=steps,
        dt=step, sigma=float(sigma), seed=int(seed), solver=block["solver"],
        time_test_type="bump",
    )
    return apply_dictionary(config, base)


def simulate_standard(block, particles, sigma, seed, base):
    config = matched_config(block, particles, sigma, seed, base)
    simulation = config["simulation"]
    positions, time, *_ = simulate_particles(
        N=int(particles), T=simulation["T"], dt=simulation["dt"], R=1,
        sigma=float(sigma), d=1, init="gaussian_mixture",
        init_centers=[0.5, 2.0, 3.5], init_std=0.005, init_scale=1.0,
        kernel_params=simulation["interaction"], seed=int(seed),
        solver=block["solver"],
    )
    return config, positions, time


def prepare_weak_system(config, positions, time, pair_seed):
    setup = json.loads(json.dumps(config))
    setup["pair_sampling"].update({
        "total_pairs": 1, "with_replacement": True,
        "exclude_diagonal": True, "seed": int(pair_seed),
        "mode": "resample_each_time",
    })
    setup["regression"].update(method="ridge", standardize=False, alpha=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_kernel_from_particles(
            setup, positions, time, return_system=True,
            pair_mode="resample_each_time", pair_seed=pair_seed,
        )
    return fit["system"]


def assemble_nested_prefix_designs(positions, time, system, ratios, pair_seed):
    """Assemble every P=ratio*N design from one nested draw per time."""
    positions = np.asarray(positions, dtype=float)
    R, T1, N, dimension = positions.shape
    if dimension != 1:
        raise ValueError("ratio screen is specialized to the 1D benchmark")
    ratios = sorted(map(int, ratios))
    budgets = {ratio: ratio * N for ratio in ratios}
    max_pairs = budgets[ratios[-1]]
    tests = system["test_fns"]
    basis_fn = system["kernel_basis"]
    temporal = np.asarray(system["time_tests"]["psi"], dtype=float)
    weights = np.asarray(system["weights"], dtype=float)
    row_scale = float(system["row_scale"])
    L = tests["phi"](positions[0, 0]).shape[1]
    K = temporal.shape[0]
    M = basis_fn(np.zeros((1, 1))).shape[1]
    designs = {ratio: np.zeros((R, L, K, M), dtype=float) for ratio in ratios}
    sampled_times = 0
    for repetition in range(R):
        for time_index in range(T1):
            weight_k = weights[time_index] * temporal[:, time_index]
            if np.all(weight_k == 0.0):
                continue
            sampled_times += 1
            rng = np.random.default_rng(np.random.SeedSequence([
                int(pair_seed), repetition, time_index, N, PAIR_NAMESPACE
            ]))
            flat = rng.integers(0, N * (N - 1), size=max_pairs)
            i_idx = flat // (N - 1)
            j_idx = flat % (N - 1)
            j_idx = j_idx + (j_idx >= i_idx)
            x = positions[repetition, time_index]
            cumulative = np.zeros((L, M), dtype=float)
            start = 0
            for ratio in ratios:
                stop = budgets[ratio]
                xi = x[i_idx[start:stop]]
                z = xi - x[j_idx[start:stop]]
                gradients = tests["grad_phi"](xi)[:, :, 0]
                basis = basis_fn(z)[:, :, 0]
                cumulative += gradients.T @ basis
                estimate = cumulative * (N - 1) / (N * stop)
                designs[ratio][repetition] += (
                    estimate[:, None, :] * weight_k[None, :, None]
                )
                start = stop
    flattened = {
        ratio: (design * row_scale).reshape(R * L * K, M)
        for ratio, design in designs.items()
    }
    return flattened, {
        "particles": N,
        "ratios": ratios,
        "budgets": budgets,
        "sampled_contributing_trajectory_times": sampled_times,
        "nested_prefixes": True,
        "resampled_each_contributing_time": True,
    }


def fit_sampled_models(A, b, Av, bv, estimator, truth):
    result = {}
    try:
        _, theta, selected = _selection_record(A, b, Av, bv, estimator)
        result["lasso"] = {
            "status": "selected",
            "theta": theta,
            "relative_error": float(np.linalg.norm(theta - truth) / np.linalg.norm(truth)),
            **selected,
        }
    except Exception as error:
        result["lasso"] = {
            "status": "failed", "error": f"{type(error).__name__}: {error}",
            "theta": None, "relative_error": None,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta_ols, _ = fit_theta(
            A, b, method="ridge", alpha=0.0, standardize=False
        )
    result["ols"] = {
        "status": "selected" if np.all(np.isfinite(theta_ols)) else "failed",
        "theta": np.asarray(theta_ols, dtype=float),
        "relative_error": float(
            np.linalg.norm(np.asarray(theta_ols) - truth) / np.linalg.norm(truth)
        ),
    }
    return result


def sampled_drift_error(theta, latent_positions, basis_fn, truth, seed, samples):
    theta = np.asarray(theta, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if not np.all(np.isfinite(theta)):
        return None
    rng = np.random.default_rng(seed)
    R, T1, N, _ = latent_positions.shape
    reps = rng.integers(0, R, size=samples)
    times = rng.integers(0, T1, size=samples)
    targets = rng.integers(0, N, size=samples)
    numerator = 0.0
    denominator = 0.0
    for repetition, time_index, target in zip(reps, times, targets):
        x = latent_positions[repetition, time_index]
        z = x[target][None, :] - x
        feature = basis_fn(z).mean(axis=0)[:, 0]
        true_value = float(feature @ truth)
        estimate = float(feature @ theta)
        numerator += (estimate - true_value) ** 2
        denominator += true_value ** 2
    return float(np.sqrt(numerator / denominator)) if denominator > 0 else None


def pair_seeds(protocol, panel, repetition, condition_index, trajectory_seed):
    train = derived_seed(
        protocol["pair_seed_start"] + trajectory_seed,
        PANEL_CODES[panel], repetition, condition_index, 0,
    )
    validation = derived_seed(
        protocol["pair_seed_start"] + protocol["pair_validation_seed_offset"]
        + trajectory_seed,
        PANEL_CODES[panel], repetition, condition_index, 1,
    )
    if train == validation:
        raise RuntimeError("training and validation pair seeds coincide")
    return train, validation


def method_record(panel, value, repetition, ratio, particles, method, entry,
                  pair_seed=None, validation_pair_seed=None, drift_error=None):
    theta = entry.get("theta")
    return {
        "panel": panel,
        "value": float(value),
        "repetition": int(repetition),
        "ratio": None if ratio == "full" else int(ratio),
        "ratio_label": str(ratio),
        "P": None if ratio == "full" else int(ratio) * int(particles),
        "particles": int(particles),
        "method": method,
        "status": entry.get("status", "selected"),
        "relative_coefficient_error": entry.get("relative_error"),
        "validation_drift_error": drift_error,
        "coefficients": None if theta is None else [float(x) for x in theta],
        "pair_seed": pair_seed,
        "validation_pair_seed": validation_pair_seed,
        "alpha_sk": entry.get("alpha_sk"),
        "alpha_index": entry.get("alpha_index"),
        "selected_at_grid_boundary": entry.get("selected_at_grid_boundary"),
        "unconverged_candidates": entry.get("unconverged_candidates", 0),
        "selected_validation_mse": entry.get("selected_validation_mse"),
        "error": entry.get("error"),
    }


def run_condition(protocol, base, lookup, panel, value, repetition, config,
                  train_positions, train_time, validation_positions,
                  validation_time, latent_validation_positions, particles,
                  trajectory_seed, strong_theta=None):
    truth = np.asarray(base["true_theta"], dtype=float)
    truth_norm = float(np.linalg.norm(truth))
    full = full_reference(lookup, panel, repetition, value, truth_norm)
    train_pair_seed, validation_pair_seed = pair_seeds(
        protocol, panel, repetition, protocol["grids"][panel].index(value),
        trajectory_seed,
    )
    train_system = prepare_weak_system(
        config, train_positions, train_time, train_pair_seed
    )
    validation_config = json.loads(json.dumps(config))
    validation_config["simulation"]["seed"] += protocol["validation_seed_offset"]
    validation_system = prepare_weak_system(
        validation_config, validation_positions, validation_time, validation_pair_seed
    )
    train_designs, train_sampling = assemble_nested_prefix_designs(
        train_positions, train_time, train_system, protocol["ratios"], train_pair_seed
    )
    validation_designs, validation_sampling = assemble_nested_prefix_designs(
        validation_positions, validation_time, validation_system,
        protocol["ratios"], validation_pair_seed,
    )
    b = np.asarray(train_system["b_flat"], dtype=float)
    bv = np.asarray(validation_system["b_flat"], dtype=float)
    nominal = np.isclose(float(value), float(protocol["nominal_conditions"][panel]))
    drift_seed = derived_seed(trajectory_seed, PANEL_CODES[panel], repetition, 777)
    records = []
    for method in ("lasso", "ols"):
        drift = None
        if nominal:
            drift = sampled_drift_error(
                full[method]["theta"], latent_validation_positions,
                validation_system["kernel_basis"], truth, drift_seed,
                protocol["nominal_conditions"]["drift_evaluation_samples"],
            )
        records.append(method_record(
            panel, value, repetition, "full", particles, method,
            full[method], drift_error=drift,
        ))
    if panel == "observation_noise":
        strong = full["strong_fd_ols"]
        drift = None
        if nominal:
            drift = sampled_drift_error(
                strong["theta"], latent_validation_positions,
                validation_system["kernel_basis"], truth, drift_seed,
                protocol["nominal_conditions"]["drift_evaluation_samples"],
            )
        records.append(method_record(
            panel, value, repetition, "full", particles, "strong_fd_ols",
            strong, drift_error=drift,
        ))
    for ratio in protocol["ratios"]:
        fitted = fit_sampled_models(
            train_designs[ratio], b, validation_designs[ratio], bv,
            base["estimator"], truth,
        )
        for method in ("lasso", "ols"):
            drift = None
            if nominal and fitted[method].get("theta") is not None:
                drift = sampled_drift_error(
                    fitted[method]["theta"], latent_validation_positions,
                    validation_system["kernel_basis"], truth, drift_seed,
                    protocol["nominal_conditions"]["drift_evaluation_samples"],
                )
            records.append(method_record(
                panel, value, repetition, ratio, particles, method,
                fitted[method], pair_seed=train_pair_seed,
                validation_pair_seed=validation_pair_seed, drift_error=drift,
            ))
    audit = {
        "panel": panel, "value": float(value), "repetition": repetition,
        "training_pair_seed": train_pair_seed,
        "validation_pair_seed": validation_pair_seed,
        "training_sampling": train_sampling,
        "validation_sampling": validation_sampling,
    }
    return records, audit


def run_repetition(payload):
    repetition, protocol, base, time_protocol = payload
    lookup = load_full_lookup(protocol)
    records, audits = [], []
    validation_offset = protocol["validation_seed_offset"]
    with threadpool_limits(limits=1):
        # Observation-noise trajectories and standardized errors are paired across levels.
        block = base["observation_noise"]
        train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
        validation_seed = train_seed + validation_offset
        X, time, *_ = reference_trajectory(train_seed, step=block["dt"], sigma=block["sigma"])
        Xv, tv, *_ = reference_trajectory(
            validation_seed, step=block["dt"], sigma=block["sigma"]
        )
        train_noise = np.random.default_rng(train_seed + 2).normal(size=X.shape)
        validation_noise = np.random.default_rng(validation_seed + 2).normal(size=Xv.shape)
        steps = int(round(block["horizon"] / block["dt"]))
        for value in protocol["grids"]["observation_noise"]:
            config = observation_config(
                block["particles"], steps, block["dt"], block["sigma"], train_seed,
                DETERMINISTIC_SPACE_CENTERS, SPACE_TEST_WIDTH, TIME_TEST_SUPPORT, "bump",
            )
            config = apply_dictionary(config, base)
            observed = X + float(value) * train_noise
            observed_validation = Xv + float(value) * validation_noise
            condition_records, audit = run_condition(
                protocol, base, lookup, "observation_noise", value, repetition,
                config, observed, time, observed_validation, tv, Xv,
                block["particles"], train_seed,
            )
            records.extend(condition_records)
            audits.append(audit)

        # Process-noise and particle-number conditions share cached trajectories when equal.
        trajectory_cache = {}
        for panel in ("process_noise", "particle_number"):
            block = base[panel]
            train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
            validation_seed = train_seed + validation_offset
            for value in protocol["grids"][panel]:
                if panel == "process_noise":
                    particles, sigma = block["particles"], float(value)
                else:
                    particles, sigma = int(value), float(block["sigma"])
                key_train = (particles, sigma, block["dt"], block["horizon"], train_seed)
                key_validation = (
                    particles, sigma, block["dt"], block["horizon"], validation_seed
                )
                if key_train not in trajectory_cache:
                    trajectory_cache[key_train] = simulate_standard(
                        block, particles, sigma, train_seed, base
                    )
                if key_validation not in trajectory_cache:
                    trajectory_cache[key_validation] = simulate_standard(
                        block, particles, sigma, validation_seed, base
                    )
                config, positions, candidate_time = trajectory_cache[key_train]
                _, validation_positions, validation_time = trajectory_cache[key_validation]
                condition_records, audit = run_condition(
                    protocol, base, lookup, panel, value, repetition, config,
                    positions, candidate_time, validation_positions, validation_time,
                    validation_positions, particles, train_seed,
                )
                records.extend(condition_records)
                audits.append(audit)

        # Paired-initial-state time-step systems use the current Figure 2(d) seed policy.
        block = time_protocol["time_step"]
        train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
        validation_seed = train_seed + block["validation_seed_offset"]
        for value in protocol["grids"]["time_step"]:
            common = {
                "particles": block["particles"],
                "repetitions": block["stacked_trajectories"],
                "horizon": block["horizon"], "dt": float(value),
                "sigma": block["sigma"], "solver": block["solver"],
                "spec": base, "policy": block["trajectory_seed_policy"],
            }
            config, positions, candidate_time, *_ = simulate_paired_positions(
                master_seed=train_seed, **common
            )
            validation_config, validation_positions, validation_time, *_ = (
                simulate_paired_positions(master_seed=validation_seed, **common)
            )
            condition_records, audit = run_condition(
                protocol, base, lookup, "time_step", value, repetition, config,
                positions, candidate_time, validation_positions, validation_time,
                validation_positions, block["particles"], train_seed,
            )
            records.extend(condition_records)
            audits.append(audit)
    return {"repetition": repetition, "records": records, "sampling_audits": audits}


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def grouped_summary(records):
    groups = []
    keys = sorted({
        (row["panel"], float(row["value"]), row["ratio_label"], row["method"])
        for row in records
    })
    for panel, value, ratio_label, method in keys:
        subset = [row for row in records if row["panel"] == panel
                  and float(row["value"]) == value
                  and row["ratio_label"] == ratio_label and row["method"] == method]
        finite = [float(row["relative_coefficient_error"]) for row in subset
                  if row["relative_coefficient_error"] is not None
                  and np.isfinite(float(row["relative_coefficient_error"]))]
        drift = [float(row["validation_drift_error"]) for row in subset
                 if row["validation_drift_error"] is not None
                 and np.isfinite(float(row["validation_drift_error"]))]
        groups.append({
            "panel": panel, "value": value, "ratio_label": ratio_label,
            "ratio": None if ratio_label == "full" else int(ratio_label),
            "method": method, "count": len(subset), "finite_count": len(finite),
            "relative_coefficient_rmse": None if not finite else rmse(finite),
            "validation_drift_rmse": None if not drift else rmse(drift),
            "failed": int(sum(row["status"] != "selected" for row in subset)),
            "unconverged_path_candidates": int(sum(
                int(row.get("unconverged_candidates") or 0) for row in subset
            )),
            "boundary_selections": int(sum(
                bool(row.get("selected_at_grid_boundary")) for row in subset
            )),
        })
    return groups


def group_lookup(groups):
    return {
        (row["panel"], float(row["value"]), row["ratio_label"], row["method"]): row
        for row in groups
    }


def slope(rows):
    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    positive = (x > 0) & (y > 0) & np.isfinite(y)
    if np.count_nonzero(positive) < 2:
        return None
    return float(np.polyfit(np.log(x[positive]), np.log(y[positive]), 1)[0])


def screen_diagnostics(protocol, records, groups, r100):
    lookup = group_lookup(groups)
    ratios = [str(value) for value in protocol["ratios"]]
    panels = list(PANEL_CODES)
    slopes = []
    endpoints = []
    rankings = []
    inflation = []
    for ratio_label in ["full", *ratios]:
        for panel in panels:
            values = sorted(protocol["grids"][panel])
            for method in ("lasso", "ols"):
                points = [(value, lookup[(panel, float(value), ratio_label, method)][
                    "relative_coefficient_rmse"]) for value in values]
                slopes.append({
                    "ratio_label": ratio_label, "panel": panel, "method": method,
                    "log_log_slope": slope(points), "values": values,
                })
            first = lookup[(panel, float(values[0]), ratio_label, "lasso")][
                "relative_coefficient_rmse"]
            last = lookup[(panel, float(values[-1]), ratio_label, "lasso")][
                "relative_coefficient_rmse"]
            endpoints.append({
                "ratio_label": ratio_label, "panel": panel,
                "first_value": values[0], "last_value": values[-1],
                "first_rmse": first, "last_rmse": last,
                "difference": last - first,
                "direction": int(np.sign(last - first)),
            })
            for value in values:
                methods = {
                    method: lookup[(panel, float(value), ratio_label, method)][
                        "relative_coefficient_rmse"]
                    for method in ("lasso", "ols")
                }
                if panel == "observation_noise":
                    methods["strong_fd_ols"] = lookup[
                        (panel, float(value), "full", "strong_fd_ols")
                    ]["relative_coefficient_rmse"]
                ordering = [item[0] for item in sorted(methods.items(), key=lambda item: item[1])]
                rankings.append({
                    "ratio_label": ratio_label, "panel": panel, "value": value,
                    "ordering_best_to_worst": ordering,
                    "lasso_rank": ordering.index("lasso") + 1,
                })
    for ratio_label in ratios:
        for panel in panels:
            for value in protocol["grids"][panel]:
                sampled = lookup[(panel, float(value), ratio_label, "lasso")][
                    "relative_coefficient_rmse"]
                full = lookup[(panel, float(value), "full", "lasso")][
                    "relative_coefficient_rmse"]
                inflation.append({
                    "ratio": int(ratio_label), "panel": panel, "value": value,
                    "sampled_rmse": sampled, "paired_full_rmse": full,
                    "inflation": sampled / full,
                })
    current_endpoint = {}
    current_endpoint_difference = {}
    for panel in panels:
        values = sorted(protocol["grids"][panel])
        current = {float(row["value"]): row for row in r100[panel]}
        difference = current[float(values[-1])]["lasso"] - current[float(values[0])]["lasso"]
        current_endpoint[panel] = int(np.sign(difference))
        current_endpoint_difference[panel] = float(difference)
    acceptance = []
    threshold = protocol["acceptance"]["max_lasso_rmse_inflation_over_paired_full"]
    for ratio in protocol["ratios"]:
        label = str(ratio)
        ratio_inflation = [row for row in inflation if row["ratio"] == ratio]
        endpoint_ok = all(
            row["direction"] == current_endpoint[row["panel"]]
            for row in endpoints if row["ratio_label"] == label
        )
        lasso_advantage = all(
            row["lasso_rank"] == 1 for row in rankings if row["ratio_label"] == label
        )
        ratio_records = [row for row in records if row["ratio_label"] == label]
        major_nonconvergence = any(
            row["status"] != "selected"
            or row["relative_coefficient_error"] is None
            or not np.isfinite(float(row["relative_coefficient_error"]))
            for row in ratio_records
        )
        max_inflation = max(row["inflation"] for row in ratio_inflation)
        acceptance.append({
            "ratio": ratio, "max_lasso_rmse_inflation": max_inflation,
            "inflation_within_1_25": bool(max_inflation <= threshold),
            "endpoint_direction_preserved": bool(endpoint_ok),
            "lasso_best_at_every_displayed_condition": bool(lasso_advantage),
            "major_nonconvergence": bool(major_nonconvergence),
            "passed": bool(
                max_inflation <= threshold and endpoint_ok
                and lasso_advantage and not major_nonconvergence
            ),
        })
    passing = [row["ratio"] for row in acceptance if row["passed"]]
    return {
        "slopes": slopes, "endpoints": endpoints,
        "current_r100_endpoint_direction": current_endpoint,
        "current_r100_endpoint_difference": current_endpoint_difference,
        "rankings": rankings, "inflation": inflation, "acceptance": acceptance,
        "minimum_passing_ratio": min(passing) if passing else None,
        "scope_limitations": protocol["grids"]["limitations"],
    }


def build_conclusion(diagnostics):
    acceptance = diagnostics["acceptance"]
    best = min(acceptance, key=lambda row: row["max_lasso_rmse_inflation"])
    drivers = {}
    for row in acceptance:
        subset = [item for item in diagnostics["inflation"] if item["ratio"] == row["ratio"]]
        drivers[str(row["ratio"])] = max(subset, key=lambda item: item["inflation"])
    paired_full_direction = {
        row["panel"]: row["direction"]
        for row in diagnostics["endpoints"] if row["ratio_label"] == "full"
    }
    current_direction = diagnostics["current_r100_endpoint_direction"]
    return {
        "minimum_passing_ratio": diagnostics["minimum_passing_ratio"],
        "tested_ratios": [row["ratio"] for row in acceptance],
        "tested_ratio_passed": {str(row["ratio"]): row["passed"] for row in acceptance},
        "best_tested_ratio_by_max_inflation": best["ratio"],
        "best_tested_max_inflation": best["max_lasso_rmse_inflation"],
        "worst_condition_by_ratio": drivers,
        "all_tested_ratios_preserve_lasso_ranking": all(
            row["lasso_best_at_every_displayed_condition"] for row in acceptance
        ),
        "all_tested_ratios_avoid_major_nonconvergence": all(
            not row["major_nonconvergence"] for row in acceptance
        ),
        "all_tested_ratios_fail_1_25_inflation_gate": all(
            not row["inflation_within_1_25"] for row in acceptance
        ),
        "all_tested_ratios_fail_current_endpoint_gate": all(
            not row["endpoint_direction_preserved"] for row in acceptance
        ),
        "paired_full_R5_endpoint_direction": paired_full_direction,
        "current_R100_endpoint_direction": current_direction,
        "paired_full_R5_particle_direction_matches_current": (
            paired_full_direction["particle_number"]
            == current_direction["particle_number"]
        ),
        "interpretation": (
            "No tested ratio is provisionally acceptable on this R=5 representative grid. "
            "The time-step sweep drives the largest paired-full inflation, while the R=5 "
            "particle-number endpoint direction is opposite to the current R=100 claim even "
            "for the paired full-pair subset. This is an exploratory rejection, not a "
            "production threshold estimate beyond P/N=75."
        ),
    }


def current_r100_reference(protocol):
    main = read_json(ROOT / protocol["matched_full_pair_batch"] / "summary.json")
    time = read_json(
        ROOT / protocol["matched_full_pair_time_step_batch"] / "summary.json"
    )
    result = {panel: [] for panel in PANEL_CODES}
    for row in main["observation_noise"]:
        result["observation_noise"].append({
            "value": row["sigma_obs"],
            "lasso": row["lasso_rmse_theta"]["rmse"] / np.sqrt(2),
            "ols": row["ols_rmse_theta"]["rmse"] / np.sqrt(2),
            "strong_fd_ols": row["euler_rmse_theta"]["rmse"] / np.sqrt(2),
        })
    for panel in ("process_noise", "particle_number"):
        for row in main[panel]:
            result[panel].append({
                "value": row["value"],
                "lasso": row["lasso_rmse_theta"]["rmse"] / np.sqrt(2),
                "ols": row["ols_rmse_theta"]["rmse"] / np.sqrt(2),
            })
    for row in time["time_step"]:
        result["time_step"].append({
            "value": row["dt"],
            "lasso": row["lasso_relative_rmse"]["rmse"],
            "ols": row["ols_relative_rmse"]["rmse"],
        })
    return result


def plot_sweeps(protocol, groups, r100, destination):
    lookup = group_lookup(groups)
    apply_manuscript_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)
    axes = axes.ravel()
    colors = {20: "#7f7f7f", 30: SECONDARY_COLOR, 50: ESTIMATE_COLOR, 75: "#7b3294"}
    titles = {
        "observation_noise": "(a) Observation noise",
        "process_noise": "(b) Process noise",
        "particle_number": "(c) Particle number",
        "time_step": "(d) Observation interval",
    }
    xlabels = {
        "observation_noise": r"observation noise $\sigma_{\rm obs}$",
        "process_noise": r"process noise $\sigma$",
        "particle_number": r"particles $N$",
        "time_step": r"observation interval $\Delta t$",
    }
    for axis, panel in zip(axes, PANEL_CODES):
        values = np.asarray(sorted(protocol["grids"][panel]), dtype=float)
        positive = values > 0
        for ratio in protocol["ratios"]:
            for method, marker, style in (("lasso", "o", "-"), ("ols", "x", ":")):
                y = np.asarray([
                    lookup[(panel, float(value), str(ratio), method)][
                        "relative_coefficient_rmse"] for value in values
                ])
                label = f"{method}, P={ratio}N"
                axis.loglog(values[positive], y[positive], marker=marker,
                            linestyle=style, color=colors[ratio], linewidth=1.1,
                            markersize=4, label=label)
                if np.any(~positive):
                    axis.scatter(values[~positive] + values[positive].min() / 3,
                                 y[~positive], marker=marker, color=colors[ratio], s=18)
        for method, color, style in (("lasso", "black", "-"), ("ols", "black", ":")):
            y = np.asarray([
                lookup[(panel, float(value), "full", method)]["relative_coefficient_rmse"]
                for value in values
            ])
            axis.loglog(values[positive], y[positive], color=color, linestyle=style,
                        linewidth=2.0, label=f"paired full {method}")
        current = {float(row["value"]): row for row in r100[panel]}
        r100_values = [value for value in values if float(value) in current and value > 0]
        axis.loglog(r100_values, [current[float(value)]["lasso"] for value in r100_values],
                    "--", color=REFERENCE_COLOR, linewidth=1.2, label="current R=100 LASSO")
        if panel == "observation_noise":
            axis.loglog(r100_values,
                        [current[float(value)]["strong_fd_ols"] for value in r100_values],
                        "s-.", color=TERTIARY_COLOR, linewidth=1.0,
                        label="current strong FD-OLS")
        axis.set(title=titles[panel], xlabel=xlabels[panel],
                 ylabel=r"relative coefficient RMSE")
        axis.set_xticks(values[positive], [f"{value:g}" for value in values[positive]])
        axis.xaxis.set_minor_formatter(NullFormatter())
        finish_axes(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.055),
               ncol=4, fontsize=7)
    fig.suptitle(protocol["output"]["watermark"], fontsize=9, color=REFERENCE_COLOR)
    destination.mkdir(parents=True, exist_ok=True)
    stem = protocol["output"]["figure_stem"] + "_sweeps"
    pdf, png = destination / f"{stem}.pdf", destination / f"{stem}.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    return [pdf, png]


def plot_diagnostics(protocol, diagnostics, groups, destination):
    apply_manuscript_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
    ratios = protocol["ratios"]
    colors = dict(zip(PANEL_CODES, (ESTIMATE_COLOR, TRUE_COLOR, SECONDARY_COLOR, TERTIARY_COLOR)))

    for panel in PANEL_CODES:
        values = []
        for ratio in ratios:
            subset = [row["inflation"] for row in diagnostics["inflation"]
                      if row["ratio"] == ratio and row["panel"] == panel]
            values.append(max(subset))
        axes[0, 0].plot(ratios, values, "o-", color=colors[panel],
                        label=panel.replace("_", " "))
    axes[0, 0].axhline(1.25, color=REFERENCE_COLOR, linestyle="--", label="1.25 threshold")
    axes[0, 0].set(xlabel=r"pair ratio $P/N$", ylabel="max LASSO RMSE inflation",
                   title="(a) Paired-full degradation")
    axes[0, 0].legend(fontsize=7)

    for panel in PANEL_CODES:
        sampled = [next(row["log_log_slope"] for row in diagnostics["slopes"]
                        if row["ratio_label"] == str(ratio) and row["panel"] == panel
                        and row["method"] == "lasso") for ratio in ratios]
        full = next(row["log_log_slope"] for row in diagnostics["slopes"]
                    if row["ratio_label"] == "full" and row["panel"] == panel
                    and row["method"] == "lasso")
        axes[0, 1].plot(ratios, sampled, "o-", color=colors[panel],
                        label=panel.replace("_", " "))
        axes[0, 1].axhline(full, color=colors[panel], linestyle=":", linewidth=.9)
    axes[0, 1].axhline(0, color="0.4", linewidth=.8)
    axes[0, 1].set(xlabel=r"pair ratio $P/N$", ylabel="representative log--log slope",
                   title="(b) LASSO slopes; dotted = paired full")

    current_endpoint = diagnostics["current_r100_endpoint_difference"]
    for panel in PANEL_CODES:
        normalized = []
        for ratio in ratios:
            difference = next(row["difference"] for row in diagnostics["endpoints"]
                              if row["ratio_label"] == str(ratio) and row["panel"] == panel)
            normalized.append(difference / max(abs(current_endpoint[panel]), 1e-300))
        axes[1, 0].plot(ratios, normalized, "o-", color=colors[panel],
                        label=panel.replace("_", " "))
    axes[1, 0].axhline(0, color="0.4", linewidth=.8)
    axes[1, 0].set(xlabel=r"pair ratio $P/N$",
                   ylabel="endpoint change / |current R=100 change|",
                   title="(c) Endpoint direction")

    acceptance = diagnostics["acceptance"]
    metrics = {
        "inflation": [row["inflation_within_1_25"] for row in acceptance],
        "endpoint": [row["endpoint_direction_preserved"] for row in acceptance],
        "LASSO rank": [row["lasso_best_at_every_displayed_condition"] for row in acceptance],
        "convergence": [not row["major_nonconvergence"] for row in acceptance],
    }
    matrix = np.asarray(list(metrics.values()), dtype=float)
    axes[1, 1].imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axes[1, 1].set_xticks(range(len(ratios)), ratios)
    axes[1, 1].set_yticks(range(len(metrics)), list(metrics))
    axes[1, 1].set(xlabel=r"pair ratio $P/N$", title="(d) Provisional checks")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1, 1].text(column, row, "pass" if matrix[row, column] else "fail",
                            ha="center", va="center", fontsize=8)

    for axis in axes.ravel()[:3]:
        finish_axes(axis)
    fig.suptitle(protocol["output"]["watermark"], fontsize=9, color=REFERENCE_COLOR)
    stem = protocol["output"]["figure_stem"] + "_diagnostics"
    pdf, png = destination / f"{stem}.pdf", destination / f"{stem}.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    return [pdf, png]


def plot_nominal_and_ranking(protocol, diagnostics, groups, destination):
    lookup = group_lookup(groups)
    apply_manuscript_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3), constrained_layout=True)
    ratios = protocol["ratios"]
    colors = dict(zip(PANEL_CODES, (ESTIMATE_COLOR, TRUE_COLOR, SECONDARY_COLOR, TERTIARY_COLOR)))
    for axis, metric, title in (
        (axes[0], "relative_coefficient_rmse", "(a) Nominal coefficient error"),
        (axes[1], "validation_drift_rmse", "(b) Nominal validation-drift error"),
    ):
        for panel in PANEL_CODES:
            nominal = float(protocol["nominal_conditions"][panel])
            full = lookup[(panel, nominal, "full", "lasso")][metric]
            sampled = [lookup[(panel, nominal, str(ratio), "lasso")][metric]
                       for ratio in ratios]
            axis.semilogy(ratios, sampled, "o-", color=colors[panel],
                          label=panel.replace("_", " "))
            axis.axhline(full, color=colors[panel], linestyle=":", linewidth=.9)
        axis.set(xlabel=r"pair ratio $P/N$", ylabel="relative RMSE", title=title)
        finish_axes(axis)
    axes[0].legend(fontsize=6.8)

    ranking_rows = [row for row in diagnostics["rankings"] if row["ratio_label"] != "full"]
    labels = sorted({f"{row['panel']}:{row['value']:g}" for row in ranking_rows})
    matrix = np.zeros((len(labels), len(ratios)))
    for i, label in enumerate(labels):
        panel, value = label.split(":")
        for j, ratio in enumerate(ratios):
            matrix[i, j] = next(row["lasso_rank"] for row in ranking_rows
                                if row["panel"] == panel and row["value"] == float(value)
                                and row["ratio_label"] == str(ratio))
    axes[2].imshow(matrix, vmin=1, vmax=3, cmap="RdYlGn_r", aspect="auto")
    axes[2].set_xticks(range(len(ratios)), ratios)
    axes[2].set_yticks(range(len(labels)), [label.replace("_", " ") for label in labels],
                       fontsize=6.2)
    axes[2].set(xlabel=r"pair ratio $P/N$", title="(c) LASSO rank (1 = best)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[2].text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7)
    fig.suptitle(protocol["output"]["watermark"], fontsize=9, color=REFERENCE_COLOR)
    stem = protocol["output"]["figure_stem"] + "_nominal_ranking"
    pdf, png = destination / f"{stem}.pdf", destination / f"{stem}.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    return [pdf, png]


def verify_outputs(protocol, records, sampling_audits, diagnostics):
    issues = []
    expected_conditions = sum(len(protocol["grids"][panel]) for panel in PANEL_CODES)
    expected_records_per_repetition = sum(
        (3 if panel == "observation_noise" else 2)
        + 2 * len(protocol["ratios"])
        for panel in PANEL_CODES for _ in protocol["grids"][panel]
    )
    if len(records) != protocol["repetitions"] * expected_records_per_repetition:
        issues.append("record count mismatch")
    for audit in sampling_audits:
        if audit["training_pair_seed"] == audit["validation_pair_seed"]:
            issues.append("training/validation pair seed collision")
        for split in ("training_sampling", "validation_sampling"):
            sampling = audit[split]
            if sampling["ratios"] != protocol["ratios"] or not sampling["nested_prefixes"]:
                issues.append("sampling ratio/prefix contract mismatch")
            if any(sampling["budgets"][str(ratio)] != ratio * sampling["particles"]
                   if isinstance(next(iter(sampling["budgets"])), str)
                   else sampling["budgets"][ratio] != ratio * sampling["particles"]
                   for ratio in protocol["ratios"]):
                issues.append("P != ratio*N")
    sampled = [row for row in records if row["ratio_label"] != "full"]
    if any(row["pair_seed"] == row["validation_pair_seed"] for row in sampled):
        issues.append("sampled record reuses train/validation pair stream")
    if any(row["status"] != "selected" for row in records):
        issues.append("one or more selected models failed")
    if diagnostics["minimum_passing_ratio"] is not None \
            and diagnostics["minimum_passing_ratio"] not in protocol["ratios"]:
        issues.append("invalid minimum passing ratio")
    result = {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "records": len(records),
        "sampling_condition_audits": len(sampling_audits),
        "expected_conditions_per_repetition": expected_conditions,
        "training_validation_pairs_independent": not any("pair" in issue for issue in issues),
        "all_P_equal_ratio_times_N": not any("P !=" in issue for issue in issues),
        "minimum_passing_ratio": diagnostics["minimum_passing_ratio"],
    }
    if issues:
        raise ValueError("; ".join(issues))
    return result


def run(config_path, out, workers=1, resume=False):
    protocol = read_json(config_path)
    base_path = ROOT / protocol["base_protocol"]
    time_path = ROOT / protocol["time_step_protocol"]
    base = read_json(base_path)
    time_protocol = read_json(time_path)
    validate_screen_protocol(protocol, base, time_protocol)
    out = Path(out).resolve()
    if resume and (out / "requested_protocol.json").exists() \
            and read_json(out / "requested_protocol.json") != protocol:
        raise ValueError("resume protocol differs from saved protocol")
    out.mkdir(parents=True, exist_ok=resume)
    jobs = out / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    write_json(out / "requested_protocol.json", protocol)
    source_hashes = snapshot_sources(out, config_path, base_path, time_path)
    started = perf_counter()
    manifest = {
        "task": protocol["task"], "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "workers": workers, "source_sha256": source_hashes,
        "current_frozen_manuscript_modified": False,
    }
    write_json(out / "manifest.json", manifest)
    results, pending = [], []
    for repetition in range(protocol["repetitions"]):
        path = jobs / f"repetition-{repetition:02d}.json"
        if resume and path.exists():
            results.append(read_json(path))
        else:
            pending.append(((repetition, protocol, base, time_protocol), path))
    if pending:
        if workers == 1:
            for index, (payload, path) in enumerate(pending, 1):
                result = run_repetition(payload)
                write_json(path, result)
                results.append(result)
                print(f"repetitions: {index}/{len(pending)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_repetition, payload): path
                           for payload, path in pending}
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    write_json(futures[future], result)
                    results.append(result)
                    completed += 1
                    print(f"repetitions: {completed}/{len(pending)}", flush=True)
    results.sort(key=lambda item: item["repetition"])
    records = [row for result in results for row in result["records"]]
    audits = [row for result in results for row in result["sampling_audits"]]
    records.sort(key=lambda row: (
        row["panel"], row["value"], row["ratio_label"], row["method"], row["repetition"]
    ))
    groups = grouped_summary(records)
    r100 = current_r100_reference(protocol)
    diagnostics = screen_diagnostics(protocol, records, groups, r100)
    verification = verify_outputs(protocol, records, audits, diagnostics)
    conclusion = build_conclusion(diagnostics)
    write_json(out / "records.json", records)
    write_csv(out / "records.csv", records)
    write_json(out / "sampling_audits.json", audits)
    write_json(out / "summary.json", groups)
    write_json(out / "screening_diagnostics.json", diagnostics)
    write_json(out / "current_r100_reference.json", r100)
    write_json(out / "verification.json", verification)
    write_json(out / "conclusion.json", conclusion)
    figure_dir = out / "figures"
    figures = []
    figures.extend(plot_sweeps(protocol, groups, r100, figure_dir))
    figures.extend(plot_diagnostics(protocol, diagnostics, groups, figure_dir))
    figures.extend(plot_nominal_and_ranking(protocol, diagnostics, groups, figure_dir))
    artifacts = [
        out / "requested_protocol.json", out / "records.json", out / "records.csv",
        out / "sampling_audits.json", out / "summary.json",
        out / "screening_diagnostics.json", out / "current_r100_reference.json",
        out / "verification.json", out / "conclusion.json", *figures,
    ]
    manifest.update({
        "status": "completed", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "counts": {"records": len(records), "sampling_audits": len(audits)},
        "minimum_passing_ratio": diagnostics["minimum_passing_ratio"],
        "scope_limitations": protocol["grids"]["limitations"],
        "artifact_sha256": {str(path.relative_to(out)): digest(path) for path in artifacts},
    })
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "outputs/validation/qanr_sampled_ratio_screen" / f"R5-{stamp}"
    manifest = run(
        args.config.resolve(), args.output, workers=args.workers, resume=args.resume
    )
    print(json.dumps({
        "status": manifest["status"], "output": str(Path(args.output).resolve()),
        "minimum_passing_ratio": manifest["minimum_passing_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
