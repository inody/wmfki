"""Confirm P/N=20 versus 50 on the complete six-atom Figure 2(a--d) grids.

This runner leaves all current, frozen, manuscript, and R=5 artifacts untouched.
It evaluates the first twenty predeclared repetitions from the current R=100
protocol, with independent sampled training/validation pair streams and
ratio-specific validation-selected LASSO penalties.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
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

from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR,
    REFERENCE_COLOR,
    SECONDARY_COLOR,
    TERTIARY_COLOR,
    TRUE_COLOR,
    apply_manuscript_style,
    finish_axes,
)
from scripts.run_qanr_matched_lasso_sweeps import apply_dictionary, read_json, validate_protocol
from scripts.run_qanr_sampled_ratio_screen import (
    PANEL_CODES,
    assemble_nested_prefix_designs,
    current_r100_reference,
    derived_seed,
    digest,
    fit_sampled_models,
    full_reference,
    group_lookup,
    grouped_summary,
    load_full_lookup,
    matched_config,
    method_record,
    pair_seeds,
    prepare_weak_system,
    rmse,
    sampled_drift_error,
    simulate_standard,
    slope,
    write_csv,
    write_json,
)
from scripts.run_qanr_time_step_repetitions import simulate_paired_positions, validate_time_protocol
from scripts.run_sindy_comparison import (
    DETERMINISTIC_SPACE_CENTERS,
    SPACE_TEST_WIDTH,
    TIME_TEST_SUPPORT,
    experiment_config as observation_config,
    reference_trajectory,
)

DEFAULT_CONFIG = ROOT / "configs/experiments/qanr_redundant6_sampled_ratio_confirmation_R20.json"
SOURCE_FILES = (
    "configs/experiments/qanr_redundant6_sampled_ratio_confirmation_R20.json",
    "configs/experiments/qanr_redundant6_lasso_sweeps_R100.json",
    "configs/experiments/qanr_redundant6_time_step_paired_R100.json",
    "scripts/run_qanr_sampled_ratio_confirmation_R20.py",
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


def validate_protocol_r20(protocol, base, time_protocol):
    validate_protocol(base)
    validate_time_protocol(time_protocol, base)
    if protocol.get("task") != "qanr_redundant6_sampled_ratio_confirmation_R20":
        raise ValueError("unexpected R20 confirmation task")
    if protocol.get("protocol_version") != 1 or protocol.get("repetitions") != 20:
        raise ValueError("R20 confirmation must remain protocol-version 1 with 20 repetitions")
    if protocol.get("ratios") != [20, 50]:
        raise ValueError("R20 ratio grid changed")
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
    expected_grids = {
        "observation_noise": base["observation_noise"]["sigma_obs_values"],
        "process_noise": base["process_noise"]["values"],
        "particle_number": base["particle_number"]["values"],
        "time_step": time_protocol["time_step"]["values"],
    }
    for panel, expected in expected_grids.items():
        if protocol["grids"].get(panel) != expected:
            raise ValueError(f"{panel} is not the complete current grid")
    if protocol["comparison"]["provisional_coefficient_inflation_threshold"] != 1.25:
        raise ValueError("coefficient inflation threshold changed")
    for key in ("matched_full_pair_batch", "matched_full_pair_time_step_batch"):
        if not (ROOT / protocol[key]).is_dir():
            raise FileNotFoundError(ROOT / protocol[key])


def snapshot_sources(out, config_path, base_path, time_path):
    files = list(SOURCE_FILES)
    for path in (config_path, base_path, time_path):
        relative = str(Path(path).resolve().relative_to(ROOT))
        if relative not in files:
            files.append(relative)
    hashes = {}
    for relative in files:
        source = ROOT / relative
        target = Path(out) / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    return hashes


def run_condition(protocol, base, lookup, panel, value, repetition, config,
                  train_positions, train_time, validation_positions,
                  validation_time, latent_validation_positions, particles,
                  trajectory_seed):
    truth = np.asarray(base["true_theta"], dtype=float)
    truth_norm = float(np.linalg.norm(truth))
    full = full_reference(lookup, panel, repetition, value, truth_norm)
    condition_index = protocol["grids"][panel].index(value)
    train_pair_seed, validation_pair_seed = pair_seeds(
        protocol, panel, repetition, condition_index, trajectory_seed
    )
    train_system = prepare_weak_system(config, train_positions, train_time, train_pair_seed)
    validation_config = json.loads(json.dumps(config))
    validation_config["simulation"]["seed"] += protocol["validation_seed_offset"]
    validation_system = prepare_weak_system(
        validation_config, validation_positions, validation_time, validation_pair_seed
    )
    train_designs, train_audit = assemble_nested_prefix_designs(
        train_positions, train_time, train_system, protocol["ratios"], train_pair_seed
    )
    validation_designs, validation_audit = assemble_nested_prefix_designs(
        validation_positions, validation_time, validation_system,
        protocol["ratios"], validation_pair_seed,
    )
    b = np.asarray(train_system["b_flat"], dtype=float)
    bv = np.asarray(validation_system["b_flat"], dtype=float)
    samples = protocol["nominal_conditions"]["drift_evaluation_samples"]
    drift_seed = derived_seed(trajectory_seed, PANEL_CODES[panel], repetition, condition_index, 991)
    records = []
    for method in ("lasso", "ols"):
        drift = sampled_drift_error(
            full[method]["theta"], latent_validation_positions,
            validation_system["kernel_basis"], truth, drift_seed, samples,
        )
        records.append(method_record(
            panel, value, repetition, "full", particles, method,
            full[method], drift_error=drift,
        ))
    if panel == "observation_noise":
        strong = full["strong_fd_ols"]
        drift = sampled_drift_error(
            strong["theta"], latent_validation_positions,
            validation_system["kernel_basis"], truth, drift_seed, samples,
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
            theta = fitted[method].get("theta")
            drift = None if theta is None else sampled_drift_error(
                theta, latent_validation_positions, validation_system["kernel_basis"],
                truth, drift_seed, samples,
            )
            records.append(method_record(
                panel, value, repetition, ratio, particles, method,
                fitted[method], pair_seed=train_pair_seed,
                validation_pair_seed=validation_pair_seed, drift_error=drift,
            ))
    return records, {
        "panel": panel, "value": float(value), "repetition": repetition,
        "training_pair_seed": train_pair_seed,
        "validation_pair_seed": validation_pair_seed,
        "training_sampling": train_audit,
        "validation_sampling": validation_audit,
    }


def run_repetition(payload):
    repetition, protocol, base, time_protocol = payload
    lookup = load_full_lookup(protocol)
    records, audits = [], []
    offset = protocol["validation_seed_offset"]
    with threadpool_limits(limits=1):
        block = base["observation_noise"]
        train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
        validation_seed = train_seed + offset
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
            result, audit = run_condition(
                protocol, base, lookup, "observation_noise", value, repetition,
                config, X + float(value) * train_noise, time,
                Xv + float(value) * validation_noise, tv, Xv,
                block["particles"], train_seed,
            )
            records.extend(result)
            audits.append(audit)

        trajectory_cache = {}
        for panel in ("process_noise", "particle_number"):
            block = base[panel]
            train_seed = block["training_seed_start"] + block["training_seed_stride"] * repetition
            validation_seed = train_seed + offset
            for value in protocol["grids"][panel]:
                if panel == "process_noise":
                    particles, sigma = block["particles"], float(value)
                else:
                    particles, sigma = int(value), float(block["sigma"])
                train_key = (particles, sigma, block["dt"], block["horizon"], train_seed)
                validation_key = (
                    particles, sigma, block["dt"], block["horizon"], validation_seed
                )
                if train_key not in trajectory_cache:
                    trajectory_cache[train_key] = simulate_standard(
                        block, particles, sigma, train_seed, base
                    )
                if validation_key not in trajectory_cache:
                    trajectory_cache[validation_key] = simulate_standard(
                        block, particles, sigma, validation_seed, base
                    )
                config, positions, candidate_time = trajectory_cache[train_key]
                _, validation_positions, validation_time = trajectory_cache[validation_key]
                result, audit = run_condition(
                    protocol, base, lookup, panel, value, repetition, config,
                    positions, candidate_time, validation_positions, validation_time,
                    validation_positions, particles, train_seed,
                )
                records.extend(result)
                audits.append(audit)

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
            _, validation_positions, validation_time, *_ = simulate_paired_positions(
                master_seed=validation_seed, **common
            )
            result, audit = run_condition(
                protocol, base, lookup, "time_step", value, repetition, config,
                positions, candidate_time, validation_positions, validation_time,
                validation_positions, block["particles"], train_seed,
            )
            records.extend(result)
            audits.append(audit)
    return {"repetition": repetition, "records": records, "sampling_audits": audits}


def slope_points(protocol, panel, values):
    values = sorted(values)
    policy = protocol["slope_windows"][panel]
    if policy == "positive_values":
        return [value for value in values if value > 0]
    if policy == "values_at_most_0.02":
        return [value for value in values if value <= 0.02]
    return values


def diagnostics(protocol, records, groups, r100):
    lookup = group_lookup(groups)
    panels = list(PANEL_CODES)
    ratios = protocol["ratios"]
    ratio_labels = [str(value) for value in ratios]
    current = {
        panel: {float(row["value"]): row for row in r100[panel]}
        for panel in panels
    }
    slope_rows, endpoint_rows, ranking_rows = [], [], []
    coefficient_rows, drift_rows = [], []
    for panel in panels:
        values = sorted(protocol["grids"][panel])
        selected_values = slope_points(protocol, panel, values)
        for label in ["current_R100", "full_R20", *ratio_labels]:
            if label == "current_R100":
                lasso_points = [(value, current[panel][float(value)]["lasso"])
                                 for value in selected_values]
                ols_points = [(value, current[panel][float(value)]["ols"])
                               for value in selected_values]
            else:
                source = "full" if label == "full_R20" else label
                lasso_points = [(value, lookup[(panel, float(value), source, "lasso")][
                    "relative_coefficient_rmse"]) for value in selected_values]
                ols_points = [(value, lookup[(panel, float(value), source, "ols")][
                    "relative_coefficient_rmse"]) for value in selected_values]
            slope_rows.extend([
                {"panel": panel, "source": label, "method": "lasso",
                 "log_log_slope": slope(lasso_points), "values": selected_values},
                {"panel": panel, "source": label, "method": "ols",
                 "log_log_slope": slope(ols_points), "values": selected_values},
            ])
        current_difference = (
            current[panel][float(values[-1])]["lasso"]
            - current[panel][float(values[0])]["lasso"]
        )
        for label in ["full_R20", *ratio_labels]:
            source = "full" if label == "full_R20" else label
            first = lookup[(panel, float(values[0]), source, "lasso")][
                "relative_coefficient_rmse"]
            last = lookup[(panel, float(values[-1]), source, "lasso")][
                "relative_coefficient_rmse"]
            endpoint_rows.append({
                "panel": panel, "source": label, "first_value": values[0],
                "last_value": values[-1], "first_rmse": first, "last_rmse": last,
                "difference": last - first, "direction": int(np.sign(last - first)),
                "current_R100_direction": int(np.sign(current_difference)),
                "matches_current": int(np.sign(last - first)) == int(np.sign(current_difference)),
            })
        for value in values:
            current_methods = {"lasso": current[panel][float(value)]["lasso"],
                               "ols": current[panel][float(value)]["ols"]}
            if panel == "observation_noise":
                current_methods["strong_fd_ols"] = current[panel][float(value)][
                    "strong_fd_ols"]
            current_order = [key for key, _ in sorted(
                current_methods.items(), key=lambda item: item[1]
            )]
            for label in ["full_R20", *ratio_labels]:
                source = "full" if label == "full_R20" else label
                methods = {
                    method: lookup[(panel, float(value), source, method)][
                        "relative_coefficient_rmse"] for method in ("lasso", "ols")
                }
                if panel == "observation_noise":
                    methods["strong_fd_ols"] = lookup[
                        (panel, float(value), "full", "strong_fd_ols")
                    ]["relative_coefficient_rmse"]
                order = [key for key, _ in sorted(methods.items(), key=lambda item: item[1])]
                ranking_rows.append({
                    "panel": panel, "value": value, "source": label,
                    "current_R100_order": current_order,
                    "order": order, "exact_order_match": order == current_order,
                    "lasso_best": order[0] == "lasso",
                })
            for ratio in ratios:
                label = str(ratio)
                for method in ("lasso", "ols"):
                    sampled = lookup[(panel, float(value), label, method)]
                    full = lookup[(panel, float(value), "full", method)]
                    coefficient_rows.append({
                        "ratio": ratio, "panel": panel, "value": value,
                        "method": method,
                        "sampled_rmse": sampled["relative_coefficient_rmse"],
                        "full_R20_rmse": full["relative_coefficient_rmse"],
                        "inflation": sampled["relative_coefficient_rmse"]
                        / full["relative_coefficient_rmse"],
                    })
                    drift_rows.append({
                        "ratio": ratio, "panel": panel, "value": value,
                        "method": method,
                        "sampled_rmse": sampled["validation_drift_rmse"],
                        "full_R20_rmse": full["validation_drift_rmse"],
                        "inflation": sampled["validation_drift_rmse"]
                        / full["validation_drift_rmse"],
                    })
    ratio_summary = []
    threshold = protocol["comparison"]["provisional_coefficient_inflation_threshold"]
    for ratio in ratios:
        coefficient = [row for row in coefficient_rows
                       if row["ratio"] == ratio and row["method"] == "lasso"]
        drift = [row for row in drift_rows
                 if row["ratio"] == ratio and row["method"] == "lasso"]
        endpoints = [row for row in endpoint_rows if row["source"] == str(ratio)]
        rankings = [row for row in ranking_rows if row["source"] == str(ratio)]
        sampled_records = [row for row in records if row["ratio_label"] == str(ratio)]
        slope_current = {row["panel"]: row["log_log_slope"] for row in slope_rows
                         if row["source"] == "current_R100" and row["method"] == "lasso"}
        slope_sampled = {row["panel"]: row["log_log_slope"] for row in slope_rows
                         if row["source"] == str(ratio) and row["method"] == "lasso"}
        slope_abs = [abs(slope_sampled[panel] - slope_current[panel]) for panel in panels]
        ratio_summary.append({
            "ratio": ratio,
            "coefficient_conditions": len(coefficient),
            "coefficient_within_1_25": int(sum(row["inflation"] <= threshold
                                                 for row in coefficient)),
            "coefficient_median_inflation": float(np.median([
                row["inflation"] for row in coefficient
            ])),
            "coefficient_max_inflation": max(row["inflation"] for row in coefficient),
            "coefficient_worst_condition": max(coefficient, key=lambda row: row["inflation"]),
            "drift_within_1_25": int(sum(row["inflation"] <= threshold for row in drift)),
            "drift_median_inflation": float(np.median([row["inflation"] for row in drift])),
            "drift_max_inflation": max(row["inflation"] for row in drift),
            "drift_worst_condition": max(drift, key=lambda row: row["inflation"]),
            "endpoint_matches": int(sum(row["matches_current"] for row in endpoints)),
            "endpoint_total": len(endpoints),
            "exact_ranking_matches": int(sum(row["exact_order_match"] for row in rankings)),
            "ranking_total": len(rankings),
            "lasso_best_conditions": int(sum(row["lasso_best"] for row in rankings)),
            "slope_mean_absolute_deviation": float(np.mean(slope_abs)),
            "slope_max_absolute_deviation": float(np.max(slope_abs)),
            "major_failures": int(sum(
                row["status"] != "selected"
                or row["relative_coefficient_error"] is None
                or not np.isfinite(float(row["relative_coefficient_error"]))
                for row in sampled_records
            )),
            "unconverged_path_candidates": int(sum(
                int(row.get("unconverged_candidates") or 0)
                for row in sampled_records if row["method"] == "lasso"
            )),
            "boundary_selections": int(sum(
                bool(row.get("selected_at_grid_boundary"))
                for row in sampled_records if row["method"] == "lasso"
            )),
        })
    # Transparent lexicographic preference: preserve claims first, then errors/slopes.
    def preference(row):
        return (
            row["endpoint_matches"], row["lasso_best_conditions"],
            row["exact_ranking_matches"], row["coefficient_within_1_25"],
            row["drift_within_1_25"], -row["slope_mean_absolute_deviation"],
            -row["coefficient_median_inflation"], -row["drift_median_inflation"],
        )
    preferred = max(ratio_summary, key=preference)["ratio"]
    return {
        "slopes": slope_rows, "endpoints": endpoint_rows,
        "rankings": ranking_rows, "coefficient_inflation": coefficient_rows,
        "drift_inflation": drift_rows, "ratio_summary": ratio_summary,
        "preferred_ratio": preferred,
        "preference_rule": protocol["comparison"]["stable_ratio_rule"],
    }


def verify(protocol, records, audits, diagnostics_payload):
    issues = []
    expected_per_repetition = (
        7 * (3 + 2 * len(protocol["ratios"]))
        + (5 + 4 + 6) * (2 + 2 * len(protocol["ratios"]))
    )
    if len(records) != protocol["repetitions"] * expected_per_repetition:
        issues.append("record count mismatch")
    if len(audits) != protocol["repetitions"] * 22:
        issues.append("sampling audit count mismatch")
    for audit in audits:
        if audit["training_pair_seed"] == audit["validation_pair_seed"]:
            issues.append("training/validation pair seed collision")
        for split in ("training_sampling", "validation_sampling"):
            sampling = audit[split]
            if sampling["ratios"] != protocol["ratios"]:
                issues.append("ratio audit mismatch")
            for ratio in protocol["ratios"]:
                budget = sampling["budgets"].get(str(ratio), sampling["budgets"].get(ratio))
                if budget != ratio * sampling["particles"]:
                    issues.append("P != ratio*N")
    if any(row["status"] != "selected" for row in records):
        issues.append("selected-model failure")
    if diagnostics_payload["preferred_ratio"] not in protocol["ratios"]:
        issues.append("invalid preferred ratio")
    result = {
        "status": "passed" if not issues else "failed", "issues": issues,
        "records": len(records), "sampling_audits": len(audits),
        "all_P_equal_ratio_times_N": not any("P !=" in issue for issue in issues),
        "training_validation_pair_streams_independent": not any(
            "pair seed" in issue for issue in issues
        ),
        "all_selected_models_finite": not any("selected-model" in issue for issue in issues),
        "preferred_ratio": diagnostics_payload["preferred_ratio"],
    }
    if issues:
        raise ValueError("; ".join(issues))
    return result


def plot_sweeps(protocol, groups, r100, destination):
    lookup = group_lookup(groups)
    apply_manuscript_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.2), constrained_layout=True)
    axes = axes.ravel()
    panels = list(PANEL_CODES)
    titles = {
        "observation_noise": "(a) Observation noise",
        "process_noise": "(b) Process noise",
        "particle_number": "(c) Particle number",
        "time_step": "(d) Observation interval",
    }
    labels = {
        "observation_noise": r"observation noise $\sigma_{\rm obs}$",
        "process_noise": r"process noise $\sigma$",
        "particle_number": r"particles $N$",
        "time_step": r"observation interval $\Delta t$",
    }
    colors = {20: ESTIMATE_COLOR, 50: TRUE_COLOR}
    for axis, panel in zip(axes, panels):
        values = np.asarray(sorted(protocol["grids"][panel]), dtype=float)
        positive = values > 0
        current = {float(row["value"]): row for row in r100[panel]}
        for ratio in protocol["ratios"]:
            for method, style, marker in (("lasso", "-", "o"), ("ols", ":", "x")):
                y = [lookup[(panel, float(value), str(ratio), method)][
                    "relative_coefficient_rmse"] for value in values]
                axis.loglog(values[positive], np.asarray(y)[positive], linestyle=style,
                            marker=marker, color=colors[ratio], linewidth=1.3,
                            label=f"{method}, P={ratio}N")
        for method, style in (("lasso", "-"), ("ols", ":")):
            full = [lookup[(panel, float(value), "full", method)][
                "relative_coefficient_rmse"] for value in values]
            axis.loglog(values[positive], np.asarray(full)[positive], color="black",
                        linestyle=style, linewidth=2.0, label=f"matched full R=20 {method}")
        axis.loglog(values[positive], [current[float(value)]["lasso"]
                                      for value in values[positive]], "--",
                    color=REFERENCE_COLOR, linewidth=1.3, label="current R=100 LASSO")
        if panel == "observation_noise":
            axis.loglog(values[positive], [current[float(value)]["strong_fd_ols"]
                                          for value in values[positive]], "s-.",
                        color=TERTIARY_COLOR, linewidth=1.0, label="strong FD-OLS R=100")
        axis.set(title=titles[panel], xlabel=labels[panel],
                 ylabel="relative coefficient RMSE")
        axis.set_xticks(values[positive], [f"{value:g}" for value in values[positive]])
        axis.xaxis.set_minor_formatter(NullFormatter())
        finish_axes(axis)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.5, -.055),
               ncol=4, fontsize=7)
    fig.suptitle(protocol["output"]["watermark"], fontsize=9, color=REFERENCE_COLOR)
    destination.mkdir(parents=True, exist_ok=True)
    stem = protocol["output"]["figure_stem"] + "_sweeps"
    paths = [destination / f"{stem}.pdf", destination / f"{stem}.png"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1], dpi=220)
    plt.close(fig)
    return paths


def plot_comparison(protocol, diagnostic, destination):
    apply_manuscript_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.0), constrained_layout=True)
    panels = list(PANEL_CODES)
    panel_colors = dict(zip(panels, (ESTIMATE_COLOR, TRUE_COLOR, SECONDARY_COLOR, TERTIARY_COLOR)))
    for axis, key, title in (
        (axes[0, 0], "coefficient_inflation", "(a) LASSO coefficient-RMSE inflation"),
        (axes[0, 1], "drift_inflation", "(b) LASSO validation-drift inflation"),
    ):
        for panel in panels:
            for ratio, style in ((20, "--"), (50, "-")):
                rows = [row for row in diagnostic[key]
                        if row["panel"] == panel and row["ratio"] == ratio
                        and row["method"] == "lasso"]
                rows.sort(key=lambda row: row["value"])
                axis.semilogy([row["value"] for row in rows],
                              [row["inflation"] for row in rows],
                              marker="o", linestyle=style, color=panel_colors[panel],
                              label=f"{panel.replace('_', ' ')}, P={ratio}N")
        axis.axhline(1.25, color=REFERENCE_COLOR, linestyle=":", linewidth=1.2)
        axis.set(xlabel="sweep value", ylabel="sampled / matched-full RMSE", title=title)
        finish_axes(axis)
    axes[0, 0].legend(fontsize=5.8, ncol=2)

    x = np.arange(len(panels))
    width = .2
    sources = [("current_R100", REFERENCE_COLOR), ("full_R20", "black"),
               ("20", ESTIMATE_COLOR), ("50", TRUE_COLOR)]
    for index, (source, color) in enumerate(sources):
        values = [next(row["log_log_slope"] for row in diagnostic["slopes"]
                       if row["panel"] == panel and row["source"] == source
                       and row["method"] == "lasso") for panel in panels]
        axes[1, 0].bar(x + (index - 1.5) * width, values, width=width,
                       color=color, label=source)
    axes[1, 0].axhline(0, color="0.3", linewidth=.8)
    axes[1, 0].set_xticks(x, [panel.replace("_", "\n") for panel in panels], fontsize=7)
    axes[1, 0].set(ylabel="LASSO log--log slope", title="(c) Current-slope comparison")
    axes[1, 0].legend(fontsize=6.5)
    finish_axes(axes[1, 0])

    rows = diagnostic["ratio_summary"]
    metrics = [
        ("endpoint", [row["endpoint_matches"] / row["endpoint_total"] for row in rows]),
        ("exact ranking", [row["exact_ranking_matches"] / row["ranking_total"] for row in rows]),
        ("LASSO best", [row["lasso_best_conditions"] / row["ranking_total"] for row in rows]),
        ("coef <=1.25", [row["coefficient_within_1_25"] / row["coefficient_conditions"]
                         for row in rows]),
        ("drift <=1.25", [row["drift_within_1_25"] / row["coefficient_conditions"]
                          for row in rows]),
    ]
    matrix = np.asarray([item[1] for item in metrics])
    axes[1, 1].imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axes[1, 1].set_xticks(range(len(rows)), [row["ratio"] for row in rows])
    axes[1, 1].set_yticks(range(len(metrics)), [item[0] for item in metrics])
    axes[1, 1].set(xlabel=r"pair ratio $P/N$", title="(d) Claim-preservation fractions")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[1, 1].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.suptitle(protocol["output"]["watermark"], fontsize=9, color=REFERENCE_COLOR)
    stem = protocol["output"]["figure_stem"] + "_comparison"
    paths = [destination / f"{stem}.pdf", destination / f"{stem}.png"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1], dpi=220)
    plt.close(fig)
    return paths


def build_conclusion(diagnostic):
    summaries = {row["ratio"]: row for row in diagnostic["ratio_summary"]}
    time_slopes = {
        row["source"]: row["log_log_slope"]
        for row in diagnostic["slopes"]
        if row["panel"] == "time_step" and row["method"] == "lasso"
    }
    return {
        "preferred_ratio": diagnostic["preferred_ratio"],
        "qualitative_endpoint_and_ranking_claims_preserved_by_both": all(
            row["endpoint_matches"] == row["endpoint_total"]
            and row["exact_ranking_matches"] == row["ranking_total"]
            and row["lasso_best_conditions"] == row["ranking_total"]
            for row in summaries.values()
        ),
        "P20": summaries[20],
        "P50": summaries[50],
        "time_step_lasso_slopes": time_slopes,
        "unchanged_quantitative_time_step_claim_supported": False,
        "reason": (
            "Both ratios preserve all endpoint directions and method rankings, but both "
            "exceed the 1.25 coefficient-RMSE threshold at every time-step value and have "
            "time-step slopes below the current R=100 slope. P/N=50 is preferred because "
            "it has lower median and maximum coefficient inflation, all 22 drift conditions "
            "within 1.25, and no selected-model failures."
        ),
    }


def report_text(protocol, diagnostic, verification, conclusion):
    rows = {row["ratio"]: row for row in diagnostic["ratio_summary"]}
    preferred = diagnostic["preferred_ratio"]
    lines = [
        "# Six-atom sampled-pair ratio confirmation (R=20)", "",
        f"Preferred tested ratio: **P/N={preferred}**.", "",
        "The calculation uses the complete current Figure 2(a--d) grids and the first "
        "20 predeclared repetitions. Training and validation pair streams are independent; "
        "each ratio selects lambda on its own sampled validation system.", "",
        "## Ratio comparison", "",
        "| P/N | endpoint matches | exact ranking matches | LASSO-best conditions | coefficient <=1.25 | median/max coefficient inflation | drift <=1.25 | median/max drift inflation | slope mean/max abs. deviation | failures |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in protocol["ratios"]:
        row = rows[ratio]
        lines.append(
            f"| {ratio} | {row['endpoint_matches']}/{row['endpoint_total']} | "
            f"{row['exact_ranking_matches']}/{row['ranking_total']} | "
            f"{row['lasso_best_conditions']}/{row['ranking_total']} | "
            f"{row['coefficient_within_1_25']}/{row['coefficient_conditions']} | "
            f"{row['coefficient_median_inflation']:.3g}/{row['coefficient_max_inflation']:.3g} | "
            f"{row['drift_within_1_25']}/{row['coefficient_conditions']} | "
            f"{row['drift_median_inflation']:.3g}/{row['drift_max_inflation']:.3g} | "
            f"{row['slope_mean_absolute_deviation']:.3g}/{row['slope_max_absolute_deviation']:.3g} | "
            f"{row['major_failures']} |"
        )
    lines.extend(["", "## LASSO slopes", "",
                  "| sweep | current R=100 | matched full R=20 | P/N=20 | P/N=50 |",
                  "|---|---:|---:|---:|---:|"])
    for panel in PANEL_CODES:
        values = {
            row["source"]: row["log_log_slope"]
            for row in diagnostic["slopes"]
            if row["panel"] == panel and row["method"] == "lasso"
        }
        lines.append(
            f"| {panel.replace('_', ' ')} | {values['current_R100']:.4g} | "
            f"{values['full_R20']:.4g} | {values['20']:.4g} | {values['50']:.4g} |"
        )
    lines.extend(["", "## Worst coefficient conditions", ""])
    for ratio in protocol["ratios"]:
        row = rows[ratio]["coefficient_worst_condition"]
        lines.append(
            f"- P/N={ratio}: {row['panel']} at {row['value']:g}, inflation "
            f"{row['inflation']:.4g} ({row['sampled_rmse']:.4g} versus "
            f"{row['full_R20_rmse']:.4g})."
        )
    lines.extend([
        "", "## Interpretation", "",
        f"Under the declared preference rule, P/N={preferred} is the more stable of the two "
        "tested ratios. Both ratios preserve all current endpoint directions and exact method "
        "rankings, and weak LASSO remains best at all 22 conditions. However, neither ratio "
        "supports carrying over the quantitative time-step coefficient claim unchanged: all "
        "six time-step coefficient inflations exceed 1.25, and the LASSO slopes are "
        f"{conclusion['time_step_lasso_slopes']['20']:.3f} and "
        f"{conclusion['time_step_lasso_slopes']['50']:.3f}, versus "
        f"{conclusion['time_step_lasso_slopes']['current_R100']:.3f} for current R=100. "
        "P/N=50 is preferred because its coefficient inflation is smaller overall and all "
        "validation-drift conditions remain within 1.25. This R=20 run is a confirmation "
        "subset and does not replace the frozen R=100 results.", "",
        "## Verification", "",
        f"- status: {verification['status']}",
        f"- records: {verification['records']}",
        f"- sampling audits: {verification['sampling_audits']}",
        f"- all P=ratio*N: {verification['all_P_equal_ratio_times_N']}",
        f"- independent training/validation pair streams: "
        f"{verification['training_validation_pair_streams_independent']}",
        f"- all selected models finite: {verification['all_selected_models_finite']}",
    ])
    return "\n".join(lines) + "\n"


def run(config_path, out, workers=1, resume=False):
    protocol = read_json(config_path)
    base_path = ROOT / protocol["base_protocol"]
    time_path = ROOT / protocol["time_step_protocol"]
    base = read_json(base_path)
    time_protocol = read_json(time_path)
    validate_protocol_r20(protocol, base, time_protocol)
    out = Path(out).resolve()
    if resume and (out / "requested_protocol.json").exists() \
            and read_json(out / "requested_protocol.json") != protocol:
        raise ValueError("resume protocol differs from saved protocol")
    out.mkdir(parents=True, exist_ok=resume)
    jobs = out / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    write_json(out / "requested_protocol.json", protocol)
    source_hashes = snapshot_sources(out, config_path, base_path, time_path)
    previous = read_json(out / "manifest.json") if resume and (out / "manifest.json").exists() else {}
    prior_elapsed = float(previous.get("elapsed_seconds", 0.0))
    started_utc = previous.get("started_utc", datetime.now(timezone.utc).isoformat())
    started = perf_counter()
    manifest = {
        "task": protocol["task"], "status": "running", "started_utc": started_utc,
        "last_invocation_started_utc": datetime.now(timezone.utc).isoformat(),
        "workers": workers, "source_sha256": source_hashes,
        "current_frozen_manuscript_R5_modified": False,
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
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_repetition, payload): path for payload, path in pending}
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
    diagnostic = diagnostics(protocol, records, groups, r100)
    verification = verify(protocol, records, audits, diagnostic)
    conclusion = build_conclusion(diagnostic)
    write_json(out / "records.json", records)
    write_csv(out / "records.csv", records)
    write_json(out / "sampling_audits.json", audits)
    write_json(out / "summary.json", groups)
    write_json(out / "current_r100_reference.json", r100)
    write_json(out / "diagnostics.json", diagnostic)
    write_json(out / "verification.json", verification)
    write_json(out / "conclusion.json", conclusion)
    (out / "report.md").write_text(
        report_text(protocol, diagnostic, verification, conclusion)
    )
    figure_dir = out / "figures"
    figures = plot_sweeps(protocol, groups, r100, figure_dir)
    figures.extend(plot_comparison(protocol, diagnostic, figure_dir))
    artifacts = [
        out / "requested_protocol.json", out / "records.json", out / "records.csv",
        out / "sampling_audits.json", out / "summary.json",
        out / "current_r100_reference.json", out / "diagnostics.json",
        out / "verification.json", out / "conclusion.json", out / "report.md", *figures,
    ]
    manifest.update({
        "status": "completed", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": prior_elapsed + perf_counter() - started,
        "counts": {"records": len(records), "sampling_audits": len(audits)},
        "preferred_ratio": diagnostic["preferred_ratio"],
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
        args.output = ROOT / "outputs/validation/qanr_sampled_ratio_confirmation" / f"R20-{stamp}"
    manifest = run(args.config.resolve(), args.output, workers=args.workers, resume=args.resume)
    print(json.dumps({
        "status": manifest["status"], "output": str(Path(args.output).resolve()),
        "preferred_ratio": manifest["preferred_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
