"""Matched R=5 search for a sampled-pair replacement of the 2D prey--prey block.

The frozen scientific predator--prey protocol, trajectories, observation noise,
and split seeds are retained.  Only the quadratic prey--prey weak-design block
is approximated.  Ratios P/N are coupled by nested ordered off-diagonal pair
prefixes; training and validation use independent pair streams.  Cross-species
blocks and held-out drift evaluation remain exact.

All artifacts are written to a new output directory.  Existing scientific
results, manuscript files, and current figures are never modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import compute_b_timeweak, make_trapz_weights
from predator_prey import channel_kernels, make_predator_prey_bases
from scripts.run_1d_hat_lasso import select_lasso
from scripts.run_predator_prey_sweeps import (
    CHANNEL_NAMES,
    _integer_ratio,
    coefficient_and_profile_metrics,
    condition_plan,
    condition_signature,
    effective_protocol as effective_base_protocol,
    latent_for_condition,
    model_objects,
    observation_view,
    paired_inputs,
    read_json,
    residual_metrics,
    selection_diagnostics,
    species_balanced_drift_error,
    split_seed,
    system_diagnostics,
    validate_protocol as validate_base_protocol,
    weak_objects,
    write_json,
)


DEFAULT_CONFIG = ROOT / "configs/experiments/predator_prey_sampled_pair_ratio_study_R5.json"
SOURCE_FILES = (
    "configs/experiments/predator_prey_sampled_pair_ratio_study_R5.json",
    "configs/experiments/predator_prey_sweeps_common_noise_001.json",
    "scripts/run_predator_prey_sampled_pair_ratio_study.py",
    "scripts/run_predator_prey_sweeps.py",
    "scripts/run_1d_hat_lasso.py",
    "predator_prey.py",
    "features.py",
    "tests.py",
    "time_tests.py",
    "pyproject.toml",
)
SWEEPS = ("observation_noise", "observation_dt", "prey_particles")
METHODS = ("lasso", "ols")
NOMINAL_CONDITION_ID = "observation_noise-04"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_protocol(requested):
    if requested.get("task") != "predator_prey_sampled_pair_ratio_study":
        raise ValueError("wrong sampled-pair ratio-study task")
    if requested.get("protocol_version") != 1:
        raise ValueError("unsupported ratio-study protocol version")
    if requested.get("study_status") != "exploratory_matched_R5":
        raise ValueError("ratio study must remain exploratory matched R=5")
    if requested.get("study") != {
        "repetitions": 5,
        "repetition_indices": [0, 1, 2, 3, 4],
        "pair_ratios": [20, 30, 50, 75],
        "run_all_sweep_conditions": True,
        "matched_trajectory_policy": "reuse_the_frozen_protocol_seeds_and_trajectory_generation_for_each_repetition_condition_and_split",
    }:
        raise ValueError("ratio-study dimensions changed")
    if requested.get("pair_sampling") != {
        "quadratic_block": "prey_prey_only",
        "ordered": True,
        "exclude_diagonal": True,
        "with_replacement": True,
        "time_policy": "independent_at_each_observation_time",
        "ratio_coupling": "nested_prefixes_of_one_max_ratio_draw_at_each_trajectory_time",
        "condition_coupling": "same_split_seed_and_physical_time_share_pair_prefixes_when_particle_count_matches",
        "index_sharing": "one_index_prefix_shared_by_all_spatial_tests_temporal_tests_and_dictionary_atoms_at_a_trajectory_time",
        "cross_species_policy": "prey_predator_and_predator_prey_sums_exact",
        "training_validation_policy": "independent_pair_samples",
        "lambda_selection": "sampled_validation_weak_residual_mse",
        "base_seed": 932000,
        "training_offset": 0,
        "validation_offset": 1000000,
        "repetition_stride": 1013,
    }:
        raise ValueError("pair-sampling protocol changed")
    if requested.get("provisional_gate") != {
        "nominal_lasso_drift_rmse_ratio_max": 1.25,
        "nominal_major_profile_rmse_ratio_max": 1.25,
        "major_profile_channels": ["prey_prey", "prey_predator", "predator_prey"],
        "sweep_direction_policy": "sampled_lasso_endpoint_mean_difference_must_have_the_same_sign_as_matched_full_pair",
        "lasso_advantage_policy": "sampled_lasso_mean_held_out_drift_must_be_below_sampled_ols_at_both_endpoints_of_each_sweep",
        "selected_fit_policy": "no_missing_or_nonfinite_selected_lasso_fit; path_warnings_are_reported_but_not_by_themselves_disqualifying",
    }:
        raise ValueError("provisional acceptance gate changed")
    if requested.get("output_policy") != {
        "exclusive_new_directory": True,
        "modify_existing_results": False,
        "modify_manuscript": False,
        "modify_current_figures": False,
    }:
        raise ValueError("output protection policy changed")


def audit_reference(requested):
    base_path = ROOT / requested["base_config"]
    reference = ROOT / requested["reference_frozen_run"]
    paths = {
        "base_config": (base_path, requested["base_config_sha256"]),
        "manifest": (reference / "manifest.json", requested["reference_manifest_sha256"]),
        "records": (reference / "records.json", requested["reference_records_sha256"]),
        "requested_protocol": (
            reference / "requested_protocol.json",
            requested["reference_requested_protocol_sha256"],
        ),
    }
    for name, (path, expected) in paths.items():
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"frozen {name} changed or is unavailable: {path}")
    base = read_json(base_path)
    frozen_requested = read_json(paths["requested_protocol"][0])
    manifest = read_json(paths["manifest"][0])
    validate_base_protocol(base)
    if base != frozen_requested:
        raise ValueError("base config differs from frozen requested protocol")
    if manifest.get("status") != "scientific_complete" or manifest.get("smoke_only"):
        raise ValueError("reference is not a completed scientific run")
    if manifest.get("requested_protocol_sha256") != payload_digest(base):
        raise ValueError("frozen requested-protocol payload hash mismatch")
    if manifest.get("artifact_sha256", {}).get("records.json") != digest(paths["records"][0]):
        raise ValueError("reference records disagree with their manifest")
    source_checks = {}
    for relative, expected in manifest["source_sha256"].items():
        actual = digest(ROOT / relative) if (ROOT / relative).is_file() else None
        source_checks[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": actual == expected,
        }
        if actual != expected:
            raise ValueError(f"current core source differs from frozen run: {relative}")
    return base, {
        "status": "passed",
        "reference_run": requested["reference_frozen_run"],
        "reference_status": manifest["status"],
        "reference_repetitions": manifest["repetitions"],
        "base_config_sha256": digest(base_path),
        "reference_manifest_sha256": digest(paths["manifest"][0]),
        "reference_records_sha256": digest(paths["records"][0]),
        "reference_requested_protocol_sha256": digest(paths["requested_protocol"][0]),
        "source_checks": source_checks,
    }


def effective_protocol(requested, base):
    validate_protocol(requested)
    spec = effective_base_protocol(base, smoke=False)
    spec["source_base_task"] = spec["task"]
    spec["task"] = requested["task"]
    spec["protocol_version"] = requested["protocol_version"]
    spec["study_status"] = requested["study_status"]
    spec["evaluation"]["scientific_repetitions"] = requested["study"]["repetitions"]
    spec["study"] = deepcopy(requested["study"])
    spec["pair_sampling"] = deepcopy(requested["pair_sampling"])
    spec["provisional_gate"] = deepcopy(requested["provisional_gate"])
    spec["output_policy"] = deepcopy(requested["output_policy"])
    return spec


def pair_seed(spec, repetition, split):
    pair = spec["pair_sampling"]
    if split not in {"training", "validation"}:
        raise ValueError("pair sampling is defined only for training and validation")
    return int(
        pair["base_seed"]
        + pair[f"{split}_offset"]
        + pair["repetition_stride"] * int(repetition)
    )


def draw_pairs(particle_count, pair_count, entropy):
    particle_count, pair_count = int(particle_count), int(pair_count)
    if particle_count < 2 or pair_count < 1:
        raise ValueError("invalid particle or pair count")
    rng = np.random.default_rng(np.random.SeedSequence([int(value) for value in entropy]))
    flat = rng.integers(
        0, particle_count * (particle_count - 1), size=pair_count, dtype=np.int64
    )
    first = flat // (particle_count - 1)
    second = flat % (particle_count - 1)
    second = second + (second >= first)
    return first, second


def _update_hash(hasher, first, second):
    hasher.update(np.asarray(first, dtype="<i8").tobytes())
    hasher.update(np.asarray(second, dtype="<i8").tobytes())


def assemble_multi_ratio_weak_systems(
    prey,
    predator,
    time,
    *,
    prey_tests,
    predator_tests,
    prey_basis,
    predator_basis,
    time_tests,
    ratios,
    seed,
    common_fine_dt,
    normalize_rows=True,
):
    """Assemble nested sampled prey--prey systems and one exact predator system."""
    prey = np.asarray(prey, dtype=float)
    predator = np.asarray(predator, dtype=float)
    time = np.asarray(time, dtype=float)
    ratios = tuple(sorted({int(value) for value in ratios}))
    if prey.ndim != 4 or prey.shape[-1] != 2:
        raise ValueError("prey must have shape (R,T+1,N,2)")
    if predator.shape != (prey.shape[0], prey.shape[1], 1, 2):
        raise ValueError("predator must have shape (R,T+1,1,2)")
    if time.shape != (prey.shape[1],) or not ratios or ratios[0] < 1:
        raise ValueError("invalid time grid or pair ratios")
    if not np.all(np.isfinite(prey)) or not np.all(np.isfinite(predator)):
        raise ValueError("weak assembly requires finite trajectories")

    weights = make_trapz_weights(time)
    response_prey = compute_b_timeweak(prey, time, 0.0, prey_tests, time_tests, weights)
    response_predator = compute_b_timeweak(
        predator, time, 0.0, predator_tests, time_tests, weights
    )
    repetitions, time_count, particle_count, _ = prey.shape
    temporal_count = time_tests["psi"].shape[0]
    prey_test_count = prey_tests["centers"].shape[0]
    predator_test_count = predator_tests["centers"].shape[0]
    prey_columns = prey_basis(np.zeros((1, 2))).shape[-2]
    predator_columns = predator_basis(np.zeros((1, 2))).shape[-2]
    prey_designs = {
        ratio: np.zeros(
            (repetitions, prey_test_count, temporal_count, 2 * prey_columns),
            dtype=float,
        )
        for ratio in ratios
    }
    predator_design = np.zeros(
        (repetitions, predator_test_count, temporal_count, predator_columns),
        dtype=float,
    )
    hashers = {ratio: hashlib.sha256() for ratio in ratios}
    draw_count = 0
    max_ratio = max(ratios)
    max_pairs = max_ratio * particle_count

    for trajectory_index in range(repetitions):
        for time_index, time_value in enumerate(time):
            fine_tick = int(round(float(time_value) / float(common_fine_dt)))
            if not np.isclose(
                fine_tick * float(common_fine_dt), time_value, rtol=0.0, atol=1e-12
            ):
                raise ValueError("observation time is off the common fine grid")
            first, second = draw_pairs(
                particle_count,
                max_pairs,
                (int(seed), trajectory_index, fine_tick, particle_count),
            )
            for ratio in ratios:
                stop = ratio * particle_count
                _update_hash(hashers[ratio], first[:stop], second[:stop])
            draw_count += 1
            temporal_weight = weights[time_index] * time_tests["psi"][:, time_index]
            if np.all(temporal_weight == 0.0):
                continue

            x = prey[trajectory_index, time_index]
            z = predator[trajectory_index, time_index, 0]
            sampled_basis = prey_basis(x[first] - x[second])
            sampled_gradients = prey_tests["grad_phi"](x[first])
            gradients_x = prey_tests["grad_phi"](x)
            cross_basis = prey_basis(x - z[None, :])
            prey_predator_feature = np.einsum(
                "ild,imd->lm", gradients_x, cross_basis
            ) / particle_count
            for ratio in ratios:
                pair_count = ratio * particle_count
                prey_prey_feature = np.einsum(
                    "pld,pmd->lm",
                    sampled_gradients[:pair_count],
                    sampled_basis[:pair_count],
                ) * (particle_count - 1) / (particle_count * pair_count)
                joined = np.concatenate(
                    [prey_prey_feature, prey_predator_feature], axis=1
                )
                prey_designs[ratio][trajectory_index] += (
                    joined[:, None, :] * temporal_weight[None, :, None]
                )

            gradient_z = predator_tests["grad_phi"](z[None, :])[0]
            mean_predator_feature = predator_basis(x - z[None, :]).mean(axis=0)
            predator_feature = np.einsum(
                "ld,md->lm", gradient_z, mean_predator_feature
            )
            predator_design[trajectory_index] += (
                predator_feature[:, None, :] * temporal_weight[None, :, None]
            )

    if normalize_rows:
        prey_scale = 1.0 / np.sqrt(prey_test_count * temporal_count)
        predator_scale = 1.0 / np.sqrt(predator_test_count * temporal_count)
        for ratio in ratios:
            prey_designs[ratio] *= prey_scale
        response_prey *= prey_scale
        predator_design *= predator_scale
        response_predator *= predator_scale
    else:
        prey_scale = predator_scale = 1.0

    prey_systems = {}
    sampling = {}
    for ratio in ratios:
        design = prey_designs[ratio]
        pair_count = ratio * particle_count
        prey_systems[ratio] = {
            "A": design,
            "b": response_prey,
            "A_flat": design.reshape(-1, design.shape[-1]),
            "b_flat": response_prey.reshape(-1),
            "row_scale": prey_scale,
        }
        sampling[ratio] = {
            "ratio": ratio,
            "particle_count": particle_count,
            "pair_count_per_observation_time": pair_count,
            "trajectory_count": repetitions,
            "observation_time_count": time_count,
            "draw_count": draw_count,
            "total_sampled_pairs": draw_count * pair_count,
            "indices_sha256": hashers[ratio].hexdigest(),
            "normalization": "(N-1)/(N*P)",
            "ordered": True,
            "exclude_diagonal": True,
            "with_replacement": True,
            "nested_within_max_ratio": True,
            "seed": int(seed),
        }
    return {
        "prey": prey_systems,
        "predator": {
            "A": predator_design,
            "b": response_predator,
            "A_flat": predator_design.reshape(-1, predator_design.shape[-1]),
            "b_flat": response_predator.reshape(-1),
            "row_scale": predator_scale,
        },
        "pair_sampling": sampling,
        "weights": weights,
    }


def assemble(spec, prey, predator, time, prey_basis, predator_basis, seed):
    prey_tests, predator_tests, temporal = weak_objects(spec, time)
    return assemble_multi_ratio_weak_systems(
        prey,
        predator,
        time,
        prey_tests=prey_tests,
        predator_tests=predator_tests,
        prey_basis=prey_basis,
        predator_basis=predator_basis,
        time_tests=temporal,
        ratios=spec["study"]["pair_ratios"],
        seed=seed,
        common_fine_dt=spec["pairing"]["dt_common_fine"],
        normalize_rows=spec["weak_tests"]["normalize_rows"],
    )


def _select(A, b, validation_A, validation_b, settings):
    selection, _ = select_lasso(A, b, validation_A, validation_b, settings)
    if selection["status"] != "selected":
        return None, selection
    return np.asarray(selection["theta"], dtype=float), selection


def _ols(system):
    theta, _, rank, singular = np.linalg.lstsq(
        system["A_flat"], system["b_flat"], rcond=None
    )
    finite = bool(np.all(np.isfinite(theta)) and np.all(np.isfinite(singular)))
    return theta, {
        "rank": int(rank),
        "singular_values": singular.tolist(),
        "finite": finite,
    }


def fit_condition(
    spec,
    condition,
    repetition,
    train_inputs,
    validation_inputs,
    latent_caches,
    prey_basis,
    predator_basis,
    truth_kernels,
):
    observed = []
    for inputs, cache in zip((train_inputs, validation_inputs), latent_caches):
        (prey, predator, _), latent_dt = latent_for_condition(
            spec, condition, inputs, truth_kernels, cache
        )
        observed.append(
            observation_view(
                prey,
                predator,
                latent_dt,
                inputs["unit_prey_noise"],
                inputs["unit_predator_noise"],
                spec["pairing"]["dt_common_fine"],
                condition["observation_dt"],
                condition["observation_noise"],
            )
        )
    training_seed = pair_seed(spec, repetition, "training")
    validation_seed = pair_seed(spec, repetition, "validation")
    training = assemble(
        spec, *observed[0], prey_basis, predator_basis, training_seed
    )
    validation = assemble(
        spec, *observed[1], prey_basis, predator_basis, validation_seed
    )
    ratios = spec["study"]["pair_ratios"]
    for ratio in ratios:
        if (
            training["pair_sampling"][ratio]["indices_sha256"]
            == validation["pair_sampling"][ratio]["indices_sha256"]
        ):
            raise RuntimeError("training and validation pair samples coincide")

    predator_theta, predator_selection = _select(
        training["predator"]["A_flat"],
        training["predator"]["b_flat"],
        validation["predator"]["A_flat"],
        validation["predator"]["b_flat"],
        spec["estimator"],
    )
    predator_ols, predator_ols_info = _ols(training["predator"])
    channel_count = prey_basis(np.zeros((1, 2))).shape[-2]
    output = {}
    paths = {"predator": predator_selection, "prey": {}}

    for ratio in ratios:
        train_prey = training["prey"][ratio]
        validation_prey = validation["prey"][ratio]
        prey_theta, prey_selection = _select(
            train_prey["A_flat"],
            train_prey["b_flat"],
            validation_prey["A_flat"],
            validation_prey["b_flat"],
            spec["estimator"],
        )
        prey_ols, prey_ols_info = _ols(train_prey)
        paths["prey"][str(ratio)] = prey_selection
        lasso_available = prey_theta is not None and predator_theta is not None
        lasso_coefficients = (
            None
            if not lasso_available
            else (
                prey_theta[:channel_count],
                prey_theta[channel_count:],
                predator_theta,
            )
        )
        ols_available = prey_ols_info["finite"] and predator_ols_info["finite"]
        ols_coefficients = (
            None
            if not ols_available
            else (
                prey_ols[:channel_count],
                prey_ols[channel_count:],
                predator_ols,
            )
        )
        methods = {
            "lasso": {
                "fit_status": "selected" if lasso_available else "fit_failed",
                "prey_selection_status": prey_selection["status"],
                "predator_selection_status": predator_selection["status"],
                "selection": {
                    "prey": (
                        selection_diagnostics(prey_selection)
                        if prey_selection["status"] == "selected"
                        else {"status": prey_selection["status"]}
                    ),
                    "predator": (
                        selection_diagnostics(predator_selection)
                        if predator_selection["status"] == "selected"
                        else {"status": predator_selection["status"]}
                    ),
                },
            },
            "ols": {
                "fit_status": "finite" if ols_available else "nonfinite",
                "prey": prey_ols_info,
                "predator": predator_ols_info,
            },
        }
        if prey_theta is not None:
            methods["lasso"]["prey"] = residual_metrics(
                prey_theta, train_prey, validation_prey
            )
        if predator_theta is not None:
            methods["lasso"]["predator"] = residual_metrics(
                predator_theta, training["predator"], validation["predator"]
            )
        methods["ols"]["prey"].update(
            residual_metrics(prey_ols, train_prey, validation_prey)
        )
        methods["ols"]["predator"].update(
            residual_metrics(
                predator_ols, training["predator"], validation["predator"]
            )
        )
        output[ratio] = {
            "arrays": {"lasso": lasso_coefficients, "ols": ols_coefficients},
            "methods": methods,
            "systems": system_diagnostics(
                {"prey": train_prey, "predator": training["predator"]},
                {"prey": validation_prey, "predator": validation["predator"]},
            ),
            "pair_sampling": {
                "training": deepcopy(training["pair_sampling"][ratio]),
                "validation": deepcopy(validation["pair_sampling"][ratio]),
                "independent_training_validation": True,
            },
        }
    return output, paths


def run_repetition(payload):
    repetition, spec = payload
    started = perf_counter()
    prey_basis, predator_basis, _, truth, truth_kernels = model_objects(spec)
    train_inputs = paired_inputs(spec, repetition, "training")
    validation_inputs = paired_inputs(spec, repetition, "validation")
    conditions = condition_plan(spec)
    fit_cache = {}
    labelled = {}
    paths = {}
    with threadpool_limits(limits=1):
        train_cache, validation_cache = {}, {}
        for condition in conditions:
            signature = condition_signature(condition)
            if signature not in fit_cache:
                fit_cache[signature] = fit_condition(
                    spec,
                    condition,
                    repetition,
                    train_inputs,
                    validation_inputs,
                    (train_cache, validation_cache),
                    prey_basis,
                    predator_basis,
                    truth_kernels,
                )
            labelled[condition["id"]] = fit_cache[signature][0]
            paths[condition["id"]] = fit_cache[signature][1]

        del train_inputs, validation_inputs, train_cache, validation_cache
        test_inputs = paired_inputs(spec, repetition, "test")
        test_cache = {}
        evaluation_cache = {}
        records = []
        for condition in conditions:
            signature = condition_signature(condition)
            (test_prey, test_predator, _), native_dt = latent_for_condition(
                spec, condition, test_inputs, truth_kernels, test_cache
            )
            stride = _integer_ratio(
                spec["evaluation"]["drift_evaluation_interval"], native_dt
            )
            for ratio in spec["study"]["pair_ratios"]:
                fitted = labelled[condition["id"]][ratio]
                methods = deepcopy(fitted["methods"])
                for method in METHODS:
                    coefficients = fitted["arrays"][method]
                    if coefficients is None:
                        methods[method]["evaluation_status"] = "not_available"
                        methods[method]["channels"] = None
                        methods[method]["held_out_drift"] = None
                        continue
                    key = (signature, ratio, method)
                    if key not in evaluation_cache:
                        estimated_kernels = channel_kernels(
                            *coefficients, prey_basis, predator_basis
                        )
                        evaluation_cache[key] = {
                            "channels": coefficient_and_profile_metrics(
                                coefficients,
                                truth,
                                spec,
                                prey_basis,
                                predator_basis,
                            ),
                            "held_out_drift": species_balanced_drift_error(
                                test_prey,
                                test_predator,
                                truth_kernels,
                                estimated_kernels,
                                stride,
                            ),
                        }
                    methods[method].update(deepcopy(evaluation_cache[key]))
                    methods[method]["evaluation_status"] = "finite"
                records.append(
                    {
                        "repetition": repetition,
                        "pair_ratio": ratio,
                        "condition": deepcopy(condition),
                        "seeds": {
                            name: split_seed(spec, repetition, name)
                            for name in ("training", "validation", "test")
                        },
                        "pair_sampling": deepcopy(fitted["pair_sampling"]),
                        "systems": deepcopy(fitted["systems"]),
                        "methods": methods,
                    }
                )
    return {
        "repetition": repetition,
        "records": records,
        "lasso_paths": paths,
        "elapsed_seconds": perf_counter() - started,
    }


def _finite(values):
    array = np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=float)
    return array[np.isfinite(array)]


def rmse(values):
    array = _finite(values)
    return float(np.sqrt(np.mean(array ** 2))) if array.size else None


def mean(values):
    array = _finite(values)
    return float(np.mean(array)) if array.size else None


def _reference_subset(requested):
    records = read_json(ROOT / requested["reference_frozen_run"] / "records.json")
    wanted = set(requested["study"]["repetition_indices"])
    subset = [row for row in records if int(row["repetition"]) in wanted]
    expected = len(wanted) * len(condition_plan(read_json(ROOT / requested["base_config"])))
    if len(subset) != expected:
        raise ValueError("reference R=5 subset is incomplete")
    return subset


def validate_matching(sampled, reference, ratios):
    reference_map = {
        (int(row["repetition"]), row["condition"]["id"]): row for row in reference
    }
    checks = []
    for row in sampled:
        key = (int(row["repetition"]), row["condition"]["id"])
        source = reference_map[key]
        passed = row["condition"] == source["condition"] and row["seeds"] == source["seeds"]
        checks.append(passed)
    expected = len(reference) * len(ratios)
    if len(sampled) != expected or not all(checks):
        raise ValueError("sampled records are not matched to the full-pair reference")
    return {
        "status": "passed",
        "sampled_record_count": len(sampled),
        "reference_record_count": len(reference),
        "expected_sampled_record_count": expected,
        "all_conditions_and_split_seeds_matched": True,
    }


def _method_values(records, condition_id, method, path):
    selected = [row for row in records if row["condition"]["id"] == condition_id]
    values = []
    for row in selected:
        item = row["methods"][method]
        for key in path:
            if item is None:
                break
            item = item[key]
        if item is not None:
            values.append(item)
    return values


def _metric(records, condition_id, method, channel=None, metric="species_balanced_relative"):
    if channel is None:
        return _method_values(records, condition_id, method, ("held_out_drift", metric))
    return _method_values(records, condition_id, method, ("channels", channel, metric))


def _coefficient_norm(record, method):
    channels = record["methods"][method].get("channels")
    if channels is None:
        return None
    vector = np.concatenate(
        [np.asarray(channels[name]["coefficients"], dtype=float) for name in CHANNEL_NAMES]
    )
    return float(np.linalg.norm(vector))


def build_comparison_report(sampled, reference, requested):
    ratios = requested["study"]["pair_ratios"]
    gate = requested["provisional_gate"]
    reference_by_sweep = {
        sweep: [row for row in reference if row["condition"]["sweep"] == sweep]
        for sweep in SWEEPS
    }
    reference_nominal = [
        row for row in reference if row["condition"]["id"] == NOMINAL_CONDITION_ID
    ]
    full_nominal = {
        "lasso_drift_rmse": rmse(
            row["methods"]["lasso"]["held_out_drift"]["species_balanced_relative"]
            for row in reference_nominal
        ),
        "channels": {},
    }
    for channel in CHANNEL_NAMES:
        full_nominal["channels"][channel] = {
            "profile_error_rmse": rmse(
                row["methods"]["lasso"]["channels"][channel]["profile_error_relative"]
                for row in reference_nominal
            ),
            "coefficient_error_rmse": rmse(
                row["methods"]["lasso"]["channels"][channel]["coefficient_error_relative"]
                for row in reference_nominal
            ),
        }

    full_directions = {}
    for sweep, rows in reference_by_sweep.items():
        values = sorted({float(row["condition"]["value"]) for row in rows})
        endpoint_means = {}
        for method in METHODS:
            endpoint_means[method] = [
                mean(
                    row["methods"][method]["held_out_drift"]["species_balanced_relative"]
                    for row in rows
                    if float(row["condition"]["value"]) == value
                )
                for value in (values[0], values[-1])
            ]
        difference = endpoint_means["lasso"][1] - endpoint_means["lasso"][0]
        full_directions[sweep] = {
            "values": [values[0], values[-1]],
            "lasso_endpoint_means": endpoint_means["lasso"],
            "ols_endpoint_means": endpoint_means["ols"],
            "lasso_endpoint_difference": difference,
            "direction_sign": int(np.sign(difference)),
            "lasso_advantage_at_endpoints": [
                endpoint_means["lasso"][index] < endpoint_means["ols"][index]
                for index in (0, 1)
            ],
        }

    ratio_reports = {}
    for ratio in ratios:
        rows = [row for row in sampled if int(row["pair_ratio"]) == ratio]
        nominal = [row for row in rows if row["condition"]["id"] == NOMINAL_CONDITION_ID]
        drift_value = rmse(
            row["methods"]["lasso"]["held_out_drift"]["species_balanced_relative"]
            for row in nominal
            if row["methods"]["lasso"]["held_out_drift"] is not None
        )
        nominal_report = {
            "lasso_drift_rmse": drift_value,
            "lasso_drift_rmse_over_full": (
                drift_value / full_nominal["lasso_drift_rmse"]
                if drift_value is not None
                else None
            ),
            "channels": {},
        }
        for channel in CHANNEL_NAMES:
            profile_value = rmse(
                row["methods"]["lasso"]["channels"][channel]["profile_error_relative"]
                for row in nominal
                if row["methods"]["lasso"]["channels"] is not None
            )
            coefficient_value = rmse(
                row["methods"]["lasso"]["channels"][channel]["coefficient_error_relative"]
                for row in nominal
                if row["methods"]["lasso"]["channels"] is not None
            )
            nominal_report["channels"][channel] = {
                "profile_error_rmse": profile_value,
                "profile_error_rmse_over_full": (
                    profile_value / full_nominal["channels"][channel]["profile_error_rmse"]
                    if profile_value is not None
                    else None
                ),
                "coefficient_error_rmse": coefficient_value,
                "coefficient_error_rmse_over_full": (
                    coefficient_value
                    / full_nominal["channels"][channel]["coefficient_error_rmse"]
                    if coefficient_value is not None
                    else None
                ),
            }

        directions = {}
        for sweep in SWEEPS:
            sweep_rows = [row for row in rows if row["condition"]["sweep"] == sweep]
            values = sorted({float(row["condition"]["value"]) for row in sweep_rows})
            endpoints = {}
            for method in METHODS:
                endpoints[method] = [
                    mean(
                        row["methods"][method]["held_out_drift"][
                            "species_balanced_relative"
                        ]
                        for row in sweep_rows
                        if float(row["condition"]["value"]) == value
                        and row["methods"][method]["held_out_drift"] is not None
                    )
                    for value in (values[0], values[-1])
                ]
            difference = endpoints["lasso"][1] - endpoints["lasso"][0]
            sign = int(np.sign(difference))
            advantage = [
                endpoints["lasso"][index] < endpoints["ols"][index]
                for index in (0, 1)
            ]
            directions[sweep] = {
                "values": [values[0], values[-1]],
                "lasso_endpoint_means": endpoints["lasso"],
                "ols_endpoint_means": endpoints["ols"],
                "lasso_endpoint_difference": difference,
                "direction_sign": sign,
                "full_pair_direction_sign": full_directions[sweep]["direction_sign"],
                "direction_preserved": sign == full_directions[sweep]["direction_sign"],
                "lasso_advantage_at_endpoints": advantage,
                "lasso_advantage_preserved": all(advantage),
            }

        fit_failures = sum(
            row["methods"]["lasso"]["fit_status"] != "selected"
            or row["methods"]["lasso"]["evaluation_status"] != "finite"
            for row in rows
        )
        selected_boundary = 0
        path_nonconverged = 0
        for row in rows:
            selection = row["methods"]["lasso"].get("selection", {})
            for species in ("prey", "predator"):
                item = selection.get(species, {})
                selected_boundary += int(item.get("selected_at_grid_boundary", False))
                path_nonconverged += int(item.get("nonconverged_candidates", 0))
        ols_nonfinite = sum(
            row["methods"]["ols"]["evaluation_status"] != "finite" for row in rows
        )
        ols_rank_deficient_prey = sum(
            int(row["methods"]["ols"]["prey"]["rank"]) < 12 for row in rows
        )
        ols_rank_deficient_predator = sum(
            int(row["methods"]["ols"]["predator"]["rank"]) < 5 for row in rows
        )
        drift_gate = (
            nominal_report["lasso_drift_rmse_over_full"] is not None
            and nominal_report["lasso_drift_rmse_over_full"]
            <= gate["nominal_lasso_drift_rmse_ratio_max"]
        )
        profile_gates = {
            channel: (
                nominal_report["channels"][channel]["profile_error_rmse_over_full"]
                is not None
                and nominal_report["channels"][channel]["profile_error_rmse_over_full"]
                <= gate["nominal_major_profile_rmse_ratio_max"]
            )
            for channel in gate["major_profile_channels"]
        }
        direction_gate = all(item["direction_preserved"] for item in directions.values())
        advantage_gate = all(
            item["lasso_advantage_preserved"] for item in directions.values()
        )
        fit_gate = fit_failures == 0
        passed = (
            drift_gate
            and all(profile_gates.values())
            and direction_gate
            and advantage_gate
            and fit_gate
        )
        ratio_reports[str(ratio)] = {
            "pair_ratio": ratio,
            "nominal": nominal_report,
            "sweep_endpoints": directions,
            "selected_fit_diagnostics": {
                "failures": fit_failures,
                "selected_at_grid_boundary_records": selected_boundary,
                "path_nonconverged_candidates_total": path_nonconverged,
            },
            "ols_stability": {
                "nonfinite_records": ols_nonfinite,
                "rank_deficient_prey_records": ols_rank_deficient_prey,
                "rank_deficient_predator_records": ols_rank_deficient_predator,
            },
            "gate": {
                "nominal_drift": drift_gate,
                "nominal_profiles": profile_gates,
                "all_sweep_directions": direction_gate,
                "all_endpoint_lasso_advantages": advantage_gate,
                "selected_fits": fit_gate,
                "passed": passed,
            },
        }

    passing = [ratio for ratio in ratios if ratio_reports[str(ratio)]["gate"]["passed"]]
    return {
        "task": requested["task"],
        "study_status": requested["study_status"],
        "repetitions": requested["study"]["repetition_indices"],
        "pair_ratios": ratios,
        "reference": {
            "run": requested["reference_frozen_run"],
            "records_sha256": requested["reference_records_sha256"],
            "full_pair_nominal": full_nominal,
            "full_pair_sweep_endpoints": full_directions,
        },
        "gate_definition": deepcopy(gate),
        "ratios": ratio_reports,
        "minimum_passing_ratio": min(passing) if passing else None,
        "passing_ratios": passing,
        "interpretation": (
            "provisional matched-R5 gate only; a larger confirmatory run is required"
        ),
    }


def _series(records, ratio, sweep, method):
    rows = [
        row for row in records
        if (ratio is None or int(row.get("pair_ratio", -1)) == ratio)
        and row["condition"]["sweep"] == sweep
    ]
    values = sorted({float(row["condition"]["value"]) for row in rows})
    means = []
    for value in values:
        means.append(
            mean(
                row["methods"][method]["held_out_drift"]["species_balanced_relative"]
                for row in rows
                if float(row["condition"]["value"]) == value
                and row["methods"][method]["held_out_drift"] is not None
            )
        )
    return np.asarray(values), np.asarray(means)


def _save_figure(fig, stem):
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return (png, pdf)


def make_figures(sampled, reference, report, destination, requested):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

    figures = destination / "figures"
    figures.mkdir()
    ratios = report["pair_ratios"]
    colors = dict(zip(ratios, plt.cm.viridis(np.linspace(0.15, 0.85, len(ratios)))))
    generated = []

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    metrics = {
        "nominal drift": [
            report["ratios"][str(r)]["nominal"]["lasso_drift_rmse_over_full"]
            for r in ratios
        ],
        "prey--prey profile": [
            report["ratios"][str(r)]["nominal"]["channels"]["prey_prey"][
                "profile_error_rmse_over_full"
            ]
            for r in ratios
        ],
        "predator on prey profile": [
            report["ratios"][str(r)]["nominal"]["channels"]["prey_predator"][
                "profile_error_rmse_over_full"
            ]
            for r in ratios
        ],
        "prey on predator profile": [
            report["ratios"][str(r)]["nominal"]["channels"]["predator_prey"][
                "profile_error_rmse_over_full"
            ]
            for r in ratios
        ],
    }
    for label, values in metrics.items():
        axes[0].plot(ratios, values, marker="o", label=label)
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle=":", label="full pair")
    axes[0].axhline(1.25, color="#D55E00", linewidth=1.0, linestyle="--", label="1.25 gate")
    axes[0].set(
        xlabel=r"pair ratio $P/N$",
        ylabel="RMS error relative to matched full pair",
        yscale="log",
        title="(a) Nominal accuracy gate",
    )
    axes[0].legend(frameon=False, fontsize=8)
    criteria = [
        "drift",
        "profile xx",
        "profile xz",
        "profile zx",
        "noise direction",
        "dt direction",
        "N direction",
        "noise advantage",
        "dt advantage",
        "N advantage",
        "selected fits",
        "overall",
    ]
    matrix = []
    for ratio in ratios:
        item = report["ratios"][str(ratio)]
        matrix.append([
            item["gate"]["nominal_drift"],
            item["gate"]["nominal_profiles"]["prey_prey"],
            item["gate"]["nominal_profiles"]["prey_predator"],
            item["gate"]["nominal_profiles"]["predator_prey"],
            item["sweep_endpoints"]["observation_noise"]["direction_preserved"],
            item["sweep_endpoints"]["observation_dt"]["direction_preserved"],
            item["sweep_endpoints"]["prey_particles"]["direction_preserved"],
            item["sweep_endpoints"]["observation_noise"]["lasso_advantage_preserved"],
            item["sweep_endpoints"]["observation_dt"]["lasso_advantage_preserved"],
            item["sweep_endpoints"]["prey_particles"]["lasso_advantage_preserved"],
            item["gate"]["selected_fits"],
            item["gate"]["passed"],
        ])
    axes[1].imshow(
        np.asarray(matrix, dtype=int).T,
        aspect="auto",
        cmap=ListedColormap(["#CC6677", "#228833"]),
        vmin=0,
        vmax=1,
    )
    axes[1].set_xticks(range(len(ratios)), [str(r) for r in ratios])
    axes[1].set_yticks(range(len(criteria)), criteria)
    axes[1].set(xlabel=r"pair ratio $P/N$", title="(b) Provisional gate summary")
    for row_index in range(len(criteria)):
        for column_index in range(len(ratios)):
            axes[1].text(
                column_index,
                row_index,
                "pass" if matrix[column_index][row_index] else "fail",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )
    generated.extend(_save_figure(fig, figures / "ratio_gate_summary"))
    plt.close(fig)

    titles = {
        "observation_noise": "Observation noise",
        "observation_dt": "Observation interval",
        "prey_particles": "Prey count",
    }
    xlabels = {
        "observation_noise": r"observation noise $\sigma_{\rm obs}$",
        "observation_dt": r"observation interval $\Delta t$",
        "prey_particles": r"prey particles $N$",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    for column, sweep in enumerate(SWEEPS):
        for row_index, method in enumerate(METHODS):
            axis = axes[row_index, column]
            x, y = _series(reference, None, sweep, method)
            axis.plot(x, y, color="black", linewidth=2.0, marker="s", label="full pair")
            for ratio in ratios:
                sx, sy = _series(sampled, ratio, sweep, method)
                axis.plot(
                    sx,
                    sy,
                    color=colors[ratio],
                    linewidth=1.15,
                    marker="o",
                    markersize=3.5,
                    label=f"P/N={ratio}",
                )
            axis.set_yscale("log")
            if sweep == "observation_noise":
                positive = x[x > 0]
                axis.set_xscale("symlog", linthresh=float(positive.min() / 2.0))
                axis.set_xlim(left=0.0)
            elif sweep == "observation_dt":
                axis.set_xscale("log")
                axis.xaxis.set_major_locator(FixedLocator(x))
                axis.xaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in x]))
                axis.xaxis.set_minor_locator(NullLocator())
                axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_title(f"{titles[sweep]} — {method.upper()}")
            axis.set_xlabel(xlabels[sweep])
            if column == 0:
                axis.set_ylabel("relative held-out drift error")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if row_index == 0 and column == 0:
                axis.legend(frameon=False, fontsize=7)
    generated.extend(_save_figure(fig, figures / "matched_sweep_heldout_errors"))
    plt.close(fig)

    prey_basis, predator_basis, _ = make_predator_prey_bases(
        0.05, [0.25, 0.5, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    radius = np.linspace(0.0, 2.0, 500)
    displacement = np.column_stack([radius, np.zeros_like(radius)])
    features = (
        prey_basis(displacement)[:, :, 0],
        prey_basis(displacement)[:, :, 0],
        predator_basis(displacement)[:, :, 0],
    )
    truth_coefficients = (
        np.asarray([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.4, 0.0, 0.0]),
    )
    nominal_reference = [
        row for row in reference if row["condition"]["id"] == NOMINAL_CONDITION_ID
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        truth_curve = features[index] @ truth_coefficients[index]
        full_curves = np.asarray([
            features[index]
            @ np.asarray(row["methods"]["lasso"]["channels"][channel]["coefficients"])
            for row in nominal_reference
        ])
        axis.plot(radius, truth_curve, color="black", linewidth=1.8, label="true")
        axis.plot(
            radius,
            np.median(full_curves, axis=0),
            color="#777777",
            linewidth=1.7,
            linestyle="--",
            label="full-pair median",
        )
        for ratio in ratios:
            rows = [
                row for row in sampled
                if int(row["pair_ratio"]) == ratio
                and row["condition"]["id"] == NOMINAL_CONDITION_ID
            ]
            curves = np.asarray([
                features[index]
                @ np.asarray(row["methods"]["lasso"]["channels"][channel]["coefficients"])
                for row in rows
                if row["methods"]["lasso"]["channels"] is not None
            ])
            axis.plot(
                radius,
                np.median(curves, axis=0),
                color=colors[ratio],
                linewidth=1.1,
                label=f"P/N={ratio}",
            )
        axis.axhline(0.0, color="#666666", linewidth=0.6)
        axis.set_title(("Prey--prey", "Predator on prey", "Prey on predator")[index])
        axis.set_xlabel(r"distance $r$")
        if index == 0:
            axis.set_ylabel("directed force")
            axis.legend(frameon=False, fontsize=7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    generated.extend(_save_figure(fig, figures / "matched_nominal_lasso_profiles"))
    plt.close(fig)

    condition_ids = [
        condition["id"]
        for condition in condition_plan(read_json(ROOT / requested["base_config"]))
    ]
    drift_matrix = np.zeros((len(ratios), len(condition_ids)))
    norm_matrix = np.zeros_like(drift_matrix)
    for column, condition_id in enumerate(condition_ids):
        full_rows = [row for row in reference if row["condition"]["id"] == condition_id]
        full_drift = rmse(
            row["methods"]["ols"]["held_out_drift"]["species_balanced_relative"]
            for row in full_rows
        )
        full_norm = rmse(
            value for value in (_coefficient_norm(row, "ols") for row in full_rows)
            if value is not None
        )
        for row_index, ratio in enumerate(ratios):
            sample_rows = [
                row for row in sampled
                if int(row["pair_ratio"]) == ratio
                and row["condition"]["id"] == condition_id
            ]
            sampled_drift = rmse(
                row["methods"]["ols"]["held_out_drift"]["species_balanced_relative"]
                for row in sample_rows
                if row["methods"]["ols"]["held_out_drift"] is not None
            )
            sampled_norm = rmse(
                value for value in (_coefficient_norm(row, "ols") for row in sample_rows)
                if value is not None
            )
            drift_matrix[row_index, column] = math.log10(sampled_drift / full_drift)
            norm_matrix[row_index, column] = math.log10(sampled_norm / full_norm)
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 5.2), constrained_layout=True)
    for axis, matrix, title in (
        (axes[0], drift_matrix, "OLS held-out drift RMSE: log10(sampled/full)"),
        (axes[1], norm_matrix, "OLS coefficient-norm RMSE: log10(sampled/full)"),
    ):
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axis.set_yticks(range(len(ratios)), [str(r) for r in ratios])
        axis.set_xticks(range(len(condition_ids)), condition_ids, rotation=45, ha="right", fontsize=7)
        axis.set_ylabel(r"pair ratio $P/N$")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.8)
    generated.extend(_save_figure(fig, figures / "ols_stability_relative_to_full"))
    plt.close(fig)
    return generated


def render_markdown(report):
    lines = [
        "# Predator--prey sampled-pair ratio study (matched R=5)",
        "",
        "This is an exploratory matched R=5 screen, not a replacement scientific run.",
        "",
        "## Provisional gate",
        "",
        "| P/N | drift/full | xx profile/full | xz profile/full | zx profile/full | directions | endpoint LASSO advantage | fit failures | pass |",
        "|---:|---:|---:|---:|---:|:---:|:---:|---:|:---:|",
    ]
    for ratio in report["pair_ratios"]:
        item = report["ratios"][str(ratio)]
        nominal = item["nominal"]
        directions = sum(
            row["direction_preserved"] for row in item["sweep_endpoints"].values()
        )
        advantages = sum(
            row["lasso_advantage_preserved"] for row in item["sweep_endpoints"].values()
        )
        lines.append(
            "| {ratio} | {drift:.3g} | {xx:.3g} | {xz:.3g} | {zx:.3g} | {directions}/3 | {advantages}/3 | {failures} | {passed} |".format(
                ratio=ratio,
                drift=nominal["lasso_drift_rmse_over_full"],
                xx=nominal["channels"]["prey_prey"]["profile_error_rmse_over_full"],
                xz=nominal["channels"]["prey_predator"]["profile_error_rmse_over_full"],
                zx=nominal["channels"]["predator_prey"]["profile_error_rmse_over_full"],
                directions=directions,
                advantages=advantages,
                failures=item["selected_fit_diagnostics"]["failures"],
                passed="PASS" if item["gate"]["passed"] else "FAIL",
            )
        )
    lines.extend([
        "",
        "## Minimum passing ratio",
        "",
        str(report["minimum_passing_ratio"]),
        "",
        "## Interpretation limits",
        "",
        "- Only repetitions 0--4 of the frozen protocol are used.",
        "- Pair ratios share nested pair prefixes, while training and validation streams are independent.",
        "- Cross-species blocks and held-out drift evaluation are exact.",
        "- A passing value is provisional and requires a larger confirmatory run before manuscript use.",
    ])
    return "\n".join(lines) + "\n"


def source_hashes():
    return {relative: digest(ROOT / relative) for relative in SOURCE_FILES}


def snapshot_sources(destination):
    hashes = source_hashes()
    for relative in SOURCE_FILES:
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    write_json(destination / "source_manifest.json", {"sha256": hashes})
    return hashes


def run_study(requested, destination, *, workers=1):
    validate_protocol(requested)
    base, audit = audit_reference(requested)
    spec = effective_protocol(requested, base)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "jobs").mkdir()
    (destination / "lasso_paths").mkdir()
    write_json(destination / "requested_protocol.json", requested)
    write_json(destination / "effective_protocol.json", spec)
    write_json(destination / "reference_audit.json", audit)
    hashes = snapshot_sources(destination)
    reference = _reference_subset(requested)
    write_json(destination / "matched_full_pair_reference_R5.json", reference)
    started = perf_counter()
    repetitions = requested["study"]["repetition_indices"]
    manifest = {
        "task": requested["task"],
        "status": "running",
        "study_status": requested["study_status"],
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "workers": int(workers),
        "repetitions": repetitions,
        "pair_ratios": requested["study"]["pair_ratios"],
        "condition_count": len(condition_plan(spec)),
        "completed_repetitions": [],
        "source_sha256": hashes,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "reference_audit_status": audit["status"],
    }
    write_json(destination / "manifest.json", manifest)
    results = []
    try:
        if workers == 1:
            for repetition in repetitions:
                result = run_repetition((repetition, spec))
                results.append(result)
                write_json(destination / "jobs" / f"repetition-{repetition:03d}.json", result)
                write_json(
                    destination / "lasso_paths" / f"repetition-{repetition:03d}.json",
                    result["lasso_paths"],
                )
                manifest["completed_repetitions"].append(repetition)
                manifest["updated_utc"] = utc_now()
                write_json(destination / "manifest.json", manifest)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(run_repetition, (repetition, spec)): repetition
                    for repetition in repetitions
                }
                for future in as_completed(futures):
                    result = future.result()
                    repetition = int(result["repetition"])
                    results.append(result)
                    write_json(destination / "jobs" / f"repetition-{repetition:03d}.json", result)
                    write_json(
                        destination / "lasso_paths" / f"repetition-{repetition:03d}.json",
                        result["lasso_paths"],
                    )
                    manifest["completed_repetitions"] = sorted(
                        manifest["completed_repetitions"] + [repetition]
                    )
                    manifest["updated_utc"] = utc_now()
                    write_json(destination / "manifest.json", manifest)
        results.sort(key=lambda row: int(row["repetition"]))
        records = [record for result in results for record in result["records"]]
        records.sort(
            key=lambda row: (
                int(row["pair_ratio"]),
                int(row["repetition"]),
                SWEEPS.index(row["condition"]["sweep"]),
                int(row["condition"]["index"]),
            )
        )
        matching = validate_matching(records, reference, requested["study"]["pair_ratios"])
        report = build_comparison_report(records, reference, requested)
        write_json(destination / "records.json", records)
        write_json(destination / "matching_report.json", matching)
        write_json(destination / "comparison_report.json", report)
        (destination / "comparison_report.md").write_text(render_markdown(report))
        figure_files = make_figures(
            records, reference, report, destination, requested
        )
        figure_manifest = {
            path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in figure_files
        }
        write_json(destination / "figure_manifest.json", figure_manifest)
        artifacts = {
            name: digest(destination / name)
            for name in (
                "requested_protocol.json",
                "effective_protocol.json",
                "reference_audit.json",
                "source_manifest.json",
                "matched_full_pair_reference_R5.json",
                "records.json",
                "matching_report.json",
                "comparison_report.json",
                "comparison_report.md",
                "figure_manifest.json",
            )
        }
        manifest.update(
            {
                "status": "exploratory_complete",
                "updated_utc": utc_now(),
                "completed_utc": utc_now(),
                "completed_repetitions": repetitions,
                "elapsed_seconds": perf_counter() - started,
                "repetition_elapsed_seconds": [row["elapsed_seconds"] for row in results],
                "matching_status": matching["status"],
                "minimum_passing_ratio": report["minimum_passing_ratio"],
                "passing_ratios": report["passing_ratios"],
                "artifact_sha256": artifacts,
            }
        )
        write_json(destination / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(
            {
                "status": "interrupted",
                "updated_utc": utc_now(),
                "elapsed_seconds": perf_counter() - started,
                "last_error": f"{type(error).__name__}: {error}",
            }
        )
        write_json(destination / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 5:
        parser.error("--workers must be between 1 and 5")
    requested = read_json(args.config)
    if args.output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = (
            ROOT
            / "outputs/validation/predator_prey_sampled_pair_ratio_study"
            / f"matched-R5-{timestamp}"
        )
    manifest = run_study(requested, args.output, workers=args.workers)
    print(Path(args.output).resolve())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "minimum_passing_ratio": manifest["minimum_passing_ratio"],
                "passing_ratios": manifest["passing_ratios"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
