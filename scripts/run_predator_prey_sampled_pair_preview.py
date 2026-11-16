"""Run a smoke-only sampled-pair preview of the 2D predator--prey sweeps.

Only the quadratic prey--prey weak-design block is sampled.  At every
observation time, P=10N ordered off-diagonal prey pairs are drawn with
replacement and one draw is shared by all spatial tests, temporal tests, and
dictionary atoms.  The prey--predator and predator--prey sums remain exact.
Training and validation use independent pair streams, and LASSO penalties are
selected using the sampled validation systems.

This preview writes only to a newly created output directory.  It does not
modify the frozen sweep, manuscript, or current figures.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
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
from scripts.run_1d_hat_lasso import select_lasso
from scripts.run_predator_prey_sweeps import (
    CHANNEL_NAMES,
    _integer_ratio,
    coefficient_and_profile_metrics,
    condition_plan,
    condition_signature,
    effective_protocol as effective_base_protocol,
    finite_summary,
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
from predator_prey import channel_kernels, make_predator_prey_bases


DEFAULT_CONFIG = ROOT / "configs/experiments/predator_prey_sampled_pair_preview.json"
SOURCE_FILES = (
    "configs/experiments/predator_prey_sampled_pair_preview.json",
    "configs/experiments/predator_prey_sweeps_common_noise_001.json",
    "scripts/run_predator_prey_sampled_pair_preview.py",
    "scripts/run_predator_prey_sweeps.py",
    "scripts/run_1d_hat_lasso.py",
    "predator_prey.py",
    "features.py",
    "tests.py",
    "time_tests.py",
    "pyproject.toml",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_preview_protocol(requested):
    if requested.get("task") != "predator_prey_sampled_pair_preview":
        raise ValueError("wrong sampled-pair preview task")
    if requested.get("protocol_version") != 1 or requested.get("smoke_only") is not True:
        raise ValueError("the sampled-pair preview must remain smoke-only protocol version 1")
    if requested.get("preview") != {
        "repetitions": 2,
        "training_trajectories": 1,
        "validation_trajectories": 1,
        "test_trajectories": 1,
        "horizon": 0.4,
        "run_all_sweep_conditions": True,
    }:
        raise ValueError("preview dimensions changed")
    if requested.get("pair_sampling") != {
        "quadratic_block": "prey_prey_only",
        "pairs_per_prey": 10,
        "ordered": True,
        "exclude_diagonal": True,
        "with_replacement": True,
        "time_policy": "independent_at_each_observation_time",
        "index_sharing": "one_index_set_shared_by_all_spatial_tests_temporal_tests_and_dictionary_atoms_at_a_trajectory_time",
        "cross_species_policy": "prey_predator_and_predator_prey_sums_exact",
        "training_validation_policy": "independent_pair_samples",
        "lambda_selection": "sampled_validation_weak_residual_mse",
        "base_seed": 731000,
        "training_offset": 0,
        "validation_offset": 1000000,
        "repetition_stride": 1009,
    }:
        raise ValueError("sampled-pair protocol changed")
    if requested.get("output_policy") != {
        "exclusive_new_directory": True,
        "modify_manuscript": False,
        "modify_current_figures": False,
        "modify_frozen_results": False,
    }:
        raise ValueError("output protection policy changed")


def audit_frozen_reference(requested):
    """Verify the immutable base config, frozen result, and current core sources."""
    base_path = ROOT / requested["base_config"]
    reference = ROOT / requested["reference_frozen_run"]
    manifest_path = reference / "manifest.json"
    requested_path = reference / "requested_protocol.json"
    records_path = reference / "records.json"
    expected = {
        base_path: requested["base_config_sha256"],
        manifest_path: requested["reference_manifest_sha256"],
        requested_path: requested["reference_requested_protocol_sha256"],
        records_path: requested["reference_records_sha256"],
    }
    for path, expected_hash in expected.items():
        if not path.is_file() or digest(path) != expected_hash:
            raise ValueError(f"frozen reference changed or is unavailable: {path}")

    base = read_json(base_path)
    validate_base_protocol(base)
    frozen_requested = read_json(requested_path)
    manifest = read_json(manifest_path)
    if frozen_requested != base:
        raise ValueError("base config differs from the frozen requested protocol")
    if manifest.get("status") != "scientific_complete" or manifest.get("smoke_only"):
        raise ValueError("reference sweep is not a completed scientific run")
    if manifest.get("requested_protocol_sha256") != payload_digest(base):
        raise ValueError("frozen requested-protocol payload hash mismatch")
    if manifest.get("artifact_sha256", {}).get("records.json") != digest(records_path):
        raise ValueError("frozen records hash is inconsistent with its manifest")

    source_checks = {}
    for relative, expected_hash in manifest.get("source_sha256", {}).items():
        current = ROOT / relative
        actual = digest(current) if current.is_file() else None
        source_checks[relative] = {
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
            "matched": actual == expected_hash,
        }
        if actual != expected_hash:
            raise ValueError(f"core source differs from frozen sweep: {relative}")
    return base, {
        "status": "passed",
        "base_config": str(base_path.relative_to(ROOT)),
        "base_config_sha256": digest(base_path),
        "reference_frozen_run": str(reference.relative_to(ROOT)),
        "reference_manifest_sha256": digest(manifest_path),
        "reference_requested_protocol_sha256": digest(requested_path),
        "reference_records_sha256": digest(records_path),
        "reference_status": manifest["status"],
        "reference_repetitions": manifest["repetitions"],
        "current_core_sources_match_reference": True,
        "source_checks": source_checks,
    }


def effective_protocol(requested, base):
    validate_preview_protocol(requested)
    spec = effective_base_protocol(base, smoke=True)
    preview = requested["preview"]
    spec["source_base_task"] = spec["task"]
    spec["task"] = requested["task"]
    spec["protocol_version"] = requested["protocol_version"]
    spec["smoke_only"] = True
    spec["evaluation"]["scientific_repetitions"] = preview["repetitions"]
    for name in (
        "training_trajectories",
        "validation_trajectories",
        "test_trajectories",
        "horizon",
    ):
        spec["observation"][name] = preview[name]
    spec["preview"] = deepcopy(preview)
    spec["pair_sampling"] = deepcopy(requested["pair_sampling"])
    spec["output_policy"] = deepcopy(requested["output_policy"])
    return spec


def pair_seed(spec, repetition, split):
    pair = spec["pair_sampling"]
    if split not in {"training", "validation"}:
        raise ValueError("pair samples are defined only for training and validation")
    return int(
        pair["base_seed"]
        + pair[f"{split}_offset"]
        + pair["repetition_stride"] * int(repetition)
    )


def draw_ordered_off_diagonal_pairs(particle_count, pair_count, entropy):
    """Draw ordered off-diagonal pair ranks directly, with replacement."""
    particle_count = int(particle_count)
    pair_count = int(pair_count)
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


def make_time_independent_pair_provider(seed, common_fine_dt):
    """Return stateless time-indexed draws that pair matching sweep conditions."""
    seed = int(seed)
    common_fine_dt = float(common_fine_dt)
    if common_fine_dt <= 0.0:
        raise ValueError("common_fine_dt must be positive")

    def provider(particle_count, pair_count, trajectory_index, time_index, time_value):
        fine_tick = int(round(float(time_value) / common_fine_dt))
        if not np.isclose(
            fine_tick * common_fine_dt, time_value, rtol=0.0, atol=1e-12
        ):
            raise ValueError("observation time is off the common fine grid")
        return draw_ordered_off_diagonal_pairs(
            particle_count,
            pair_count,
            (seed, int(trajectory_index), fine_tick, int(particle_count)),
        )

    return provider


def _update_pair_hash(hasher, first, second):
    hasher.update(np.asarray(first, dtype="<i8").tobytes())
    hasher.update(np.asarray(second, dtype="<i8").tobytes())


def assemble_sampled_predator_prey_weak_systems(
    prey,
    predator,
    time,
    *,
    prey_tests,
    predator_tests,
    prey_basis,
    predator_basis,
    time_tests,
    pair_count,
    pair_index_provider,
    normalize_rows=True,
):
    """Assemble sampled prey--prey and exact cross-species weak systems."""
    prey = np.asarray(prey, dtype=float)
    predator = np.asarray(predator, dtype=float)
    time = np.asarray(time, dtype=float)
    pair_count = int(pair_count)
    if prey.ndim != 4 or prey.shape[-1] != 2:
        raise ValueError("prey must have shape (R,T+1,N,2)")
    if predator.shape != (prey.shape[0], prey.shape[1], 1, 2):
        raise ValueError("predator must have shape (R,T+1,1,2)")
    if time.shape != (prey.shape[1],):
        raise ValueError("time has the wrong shape")
    if pair_count < 1 or not callable(pair_index_provider):
        raise ValueError("positive pair_count and pair_index_provider are required")
    if not np.all(np.isfinite(prey)) or not np.all(np.isfinite(predator)):
        raise ValueError("weak assembly requires finite trajectories")

    weights = make_trapz_weights(time)
    response_prey = compute_b_timeweak(
        prey, time, 0.0, prey_tests, time_tests, weights
    )
    response_predator = compute_b_timeweak(
        predator, time, 0.0, predator_tests, time_tests, weights
    )

    repetitions, time_count, particle_count, _ = prey.shape
    temporal_count = time_tests["psi"].shape[0]
    prey_test_count = prey_tests["centers"].shape[0]
    predator_test_count = predator_tests["centers"].shape[0]
    prey_column_count = prey_basis(np.zeros((1, 2))).shape[-2]
    predator_column_count = predator_basis(np.zeros((1, 2))).shape[-2]
    prey_design = np.zeros(
        (repetitions, prey_test_count, temporal_count, 2 * prey_column_count),
        dtype=float,
    )
    predator_design = np.zeros(
        (repetitions, predator_test_count, temporal_count, predator_column_count),
        dtype=float,
    )
    pair_hasher = hashlib.sha256()
    draw_count = 0

    for trajectory_index in range(repetitions):
        for time_index in range(time_count):
            x = prey[trajectory_index, time_index]
            z = predator[trajectory_index, time_index, 0]
            first, second = pair_index_provider(
                particle_count,
                pair_count,
                trajectory_index,
                time_index,
                time[time_index],
            )
            first = np.asarray(first, dtype=np.int64)
            second = np.asarray(second, dtype=np.int64)
            if first.shape != (pair_count,) or second.shape != (pair_count,):
                raise ValueError("pair provider returned the wrong number of pairs")
            if (
                np.any(first < 0)
                or np.any(first >= particle_count)
                or np.any(second < 0)
                or np.any(second >= particle_count)
                or np.any(first == second)
            ):
                raise ValueError("pair provider must return valid off-diagonal pairs")
            _update_pair_hash(pair_hasher, first, second)
            draw_count += 1

            temporal_weight = weights[time_index] * time_tests["psi"][:, time_index]
            if np.all(temporal_weight == 0.0):
                continue

            sampled_basis = prey_basis(x[first] - x[second])
            sampled_gradients = prey_tests["grad_phi"](x[first])
            prey_prey_feature = np.einsum(
                "pld,pmd->lm", sampled_gradients, sampled_basis
            ) * (particle_count - 1) / (particle_count * pair_count)

            gradients_x = prey_tests["grad_phi"](x)
            cross_basis = prey_basis(x - z[None, :])
            prey_predator_feature = np.einsum(
                "ild,imd->lm", gradients_x, cross_basis
            ) / particle_count
            joined = np.concatenate(
                [prey_prey_feature, prey_predator_feature], axis=1
            )
            prey_design[trajectory_index] += (
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
        prey_design *= prey_scale
        response_prey *= prey_scale
        predator_design *= predator_scale
        response_predator *= predator_scale
    else:
        prey_scale = predator_scale = 1.0

    return {
        "prey": {
            "A": prey_design,
            "b": response_prey,
            "A_flat": prey_design.reshape(-1, prey_design.shape[-1]),
            "b_flat": response_prey.reshape(-1),
            "row_scale": prey_scale,
        },
        "predator": {
            "A": predator_design,
            "b": response_predator,
            "A_flat": predator_design.reshape(-1, predator_design.shape[-1]),
            "b_flat": response_predator.reshape(-1),
            "row_scale": predator_scale,
        },
        "weights": weights,
        "pair_sampling": {
            "particle_count": particle_count,
            "pair_count_per_observation_time": pair_count,
            "pairs_per_prey": pair_count / particle_count,
            "trajectory_count": repetitions,
            "observation_time_count": time_count,
            "draw_count": draw_count,
            "total_sampled_pairs": draw_count * pair_count,
            "indices_sha256": pair_hasher.hexdigest(),
            "normalization": "(N-1)/(N*P)",
            "ordered": True,
            "exclude_diagonal": True,
            "with_replacement": True,
            "cross_species_sums": "exact",
        },
    }


def assemble(spec, prey, predator, time, prey_basis, predator_basis, seed):
    prey_tests, predator_tests, temporal = weak_objects(spec, time)
    particle_count = prey.shape[2]
    pair_count = int(spec["pair_sampling"]["pairs_per_prey"] * particle_count)
    provider = make_time_independent_pair_provider(
        seed, spec["pairing"]["dt_common_fine"]
    )
    systems = assemble_sampled_predator_prey_weak_systems(
        prey,
        predator,
        time,
        prey_tests=prey_tests,
        predator_tests=predator_tests,
        prey_basis=prey_basis,
        predator_basis=predator_basis,
        time_tests=temporal,
        pair_count=pair_count,
        pair_index_provider=provider,
        normalize_rows=spec["weak_tests"]["normalize_rows"],
    )
    systems["pair_sampling"]["seed"] = int(seed)
    return systems


def fit_condition(
    spec,
    condition,
    experiment_repetition,
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
    training_seed = pair_seed(spec, experiment_repetition, "training")
    validation_seed = pair_seed(spec, experiment_repetition, "validation")
    training = assemble(
        spec, *observed[0], prey_basis, predator_basis, training_seed
    )
    validation = assemble(
        spec, *observed[1], prey_basis, predator_basis, validation_seed
    )
    if training_seed == validation_seed:
        raise RuntimeError("training and validation pair seeds must differ")
    if (
        training["pair_sampling"]["indices_sha256"]
        == validation["pair_sampling"]["indices_sha256"]
    ):
        raise RuntimeError("training and validation pair samples unexpectedly coincide")

    selections = {}
    paths = {}
    for species in ("prey", "predator"):
        selection, _ = select_lasso(
            training[species]["A_flat"],
            training[species]["b_flat"],
            validation[species]["A_flat"],
            validation[species]["b_flat"],
            spec["estimator"],
        )
        if selection["status"] != "selected":
            raise RuntimeError(f"no converged {species} LASSO candidate")
        selections[species] = selection
        paths[species] = selection

    lasso_prey = np.asarray(selections["prey"]["theta"], dtype=float)
    lasso_predator = np.asarray(selections["predator"]["theta"], dtype=float)
    ols_prey, _, ols_prey_rank, ols_prey_singular = np.linalg.lstsq(
        training["prey"]["A_flat"], training["prey"]["b_flat"], rcond=None
    )
    ols_predator, _, ols_predator_rank, ols_predator_singular = np.linalg.lstsq(
        training["predator"]["A_flat"], training["predator"]["b_flat"], rcond=None
    )
    channel_count = prey_basis(np.zeros((1, 2))).shape[-2]
    arrays = {
        "lasso": (
            lasso_prey[:channel_count],
            lasso_prey[channel_count:],
            lasso_predator,
        ),
        "ols": (
            ols_prey[:channel_count],
            ols_prey[channel_count:],
            ols_predator,
        ),
    }
    methods = {
        "lasso": {
            "prey": residual_metrics(
                lasso_prey, training["prey"], validation["prey"]
            ),
            "predator": residual_metrics(
                lasso_predator, training["predator"], validation["predator"]
            ),
            "selection": {
                name: selection_diagnostics(selections[name])
                for name in ("prey", "predator")
            },
            "selection_input": {
                "prey": "sampled_training_and_independent_sampled_validation",
                "predator": "exact_cross_species_training_and_validation",
            },
        },
        "ols": {
            "prey": {
                **residual_metrics(ols_prey, training["prey"], validation["prey"]),
                "rank": int(ols_prey_rank),
                "singular_values": ols_prey_singular.tolist(),
            },
            "predator": {
                **residual_metrics(
                    ols_predator, training["predator"], validation["predator"]
                ),
                "rank": int(ols_predator_rank),
                "singular_values": ols_predator_singular.tolist(),
            },
        },
    }
    return {
        "arrays": arrays,
        "methods": methods,
        "systems": system_diagnostics(training, validation),
        "pair_sampling": {
            "training": deepcopy(training["pair_sampling"]),
            "validation": deepcopy(validation["pair_sampling"]),
            "independent_training_validation": True,
        },
        "paths": paths,
    }


def run_repetition(payload):
    repetition, spec = payload
    started = perf_counter()
    prey_basis, predator_basis, _, truth, truth_kernels = model_objects(spec)
    train_inputs = paired_inputs(spec, repetition, "training")
    validation_inputs = paired_inputs(spec, repetition, "validation")
    conditions = condition_plan(spec)
    fit_cache = {}
    labelled_fits = {}
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
            labelled_fits[condition["id"]] = fit_cache[signature]
            paths[condition["id"]] = fit_cache[signature]["paths"]

        # All sampled-validation selections are frozen before held-out generation.
        del train_inputs, validation_inputs, train_cache, validation_cache
        test_inputs = paired_inputs(spec, repetition, "test")
        test_cache = {}
        evaluation_cache = {}
        records = []
        for condition in conditions:
            signature = condition_signature(condition)
            fitted = labelled_fits[condition["id"]]
            (test_prey, test_predator, _), native_dt = latent_for_condition(
                spec, condition, test_inputs, truth_kernels, test_cache
            )
            stride = _integer_ratio(
                spec["evaluation"]["drift_evaluation_interval"], native_dt
            )
            methods = deepcopy(fitted["methods"])
            for method, coefficients in fitted["arrays"].items():
                evaluation_key = (signature, method)
                if evaluation_key not in evaluation_cache:
                    estimated_kernels = channel_kernels(
                        *coefficients, prey_basis, predator_basis
                    )
                    evaluation_cache[evaluation_key] = {
                        "channels": coefficient_and_profile_metrics(
                            coefficients, truth, spec, prey_basis, predator_basis
                        ),
                        "held_out_drift": species_balanced_drift_error(
                            test_prey,
                            test_predator,
                            truth_kernels,
                            estimated_kernels,
                            stride,
                        ),
                    }
                methods[method].update(deepcopy(evaluation_cache[evaluation_key]))
            records.append(
                {
                    "repetition": repetition,
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


def summarize(records, conditions):
    summary = {"conditions": {}}
    for condition in conditions:
        rows = [row for row in records if row["condition"]["id"] == condition["id"]]
        item = {"condition": condition, "methods": {}}
        for method in ("lasso", "ols"):
            metrics = {
                name: finite_summary(
                    [
                        row["methods"][method]["held_out_drift"][name]
                        for row in rows
                    ]
                )
                for name in (
                    "species_balanced_relative",
                    "prey_relative",
                    "predator_relative",
                )
            }
            metrics["channels"] = {
                channel: {
                    metric: finite_summary(
                        [
                            row["methods"][method]["channels"][channel][metric]
                            for row in rows
                        ]
                    )
                    for metric in (
                        "coefficient_error_l2",
                        "coefficient_error_relative",
                        "profile_error_relative",
                    )
                }
                for channel in CHANNEL_NAMES
            }
            if method == "lasso":
                metrics["convergence"] = {
                    species: {
                        "boundary_selections": int(
                            sum(
                                row["methods"][method]["selection"][species][
                                    "selected_at_grid_boundary"
                                ]
                                for row in rows
                            )
                        ),
                        "nonconverged_candidates": int(
                            sum(
                                row["methods"][method]["selection"][species][
                                    "nonconverged_candidates"
                                ]
                                for row in rows
                            )
                        ),
                        "convergence_warnings": int(
                            sum(
                                row["methods"][method]["selection"][species][
                                    "convergence_warnings"
                                ]
                                for row in rows
                            )
                        ),
                    }
                    for species in ("prey", "predator")
                }
            item["methods"][method] = metrics
        summary["conditions"][condition["id"]] = item
    return summary


def pair_sampling_summary(records):
    rows = []
    for record in records:
        training = record["pair_sampling"]["training"]
        validation = record["pair_sampling"]["validation"]
        rows.append(
            {
                "repetition": record["repetition"],
                "condition_id": record["condition"]["id"],
                "condition_signature": list(condition_signature(record["condition"])),
                "particle_count": training["particle_count"],
                "pair_count_per_observation_time": training[
                    "pair_count_per_observation_time"
                ],
                "training_seed": training["seed"],
                "validation_seed": validation["seed"],
                "training_indices_sha256": training["indices_sha256"],
                "validation_indices_sha256": validation["indices_sha256"],
                "independent_training_validation": (
                    training["seed"] != validation["seed"]
                    and training["indices_sha256"] != validation["indices_sha256"]
                ),
                "cross_species_sums": training["cross_species_sums"],
            }
        )
    return {
        "status": "passed"
        if all(row["independent_training_validation"] for row in rows)
        else "failed",
        "record_count": len(rows),
        "all_training_validation_samples_independent": all(
            row["independent_training_validation"] for row in rows
        ),
        "all_cross_species_sums_exact": all(
            row["cross_species_sums"] == "exact" for row in rows
        ),
        "rows": rows,
    }


PREVIEW_SWEEPS = ("observation_noise", "observation_dt", "prey_particles")
PREVIEW_METHOD_STYLE = {
    "lasso": {
        "label": "weak LASSO",
        "color": "#D55E00",
        "marker": "^",
        "linestyle": "--",
    },
    "ols": {
        "label": "weak OLS",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
}


def preview_sweep_groups(records):
    """Extract held-out errors without rerunning fitting or evaluation."""
    if not isinstance(records, list) or not records:
        raise ValueError("records.json must contain a nonempty list")
    groups = {}
    for sweep in PREVIEW_SWEEPS:
        sweep_rows = [row for row in records if row["condition"]["sweep"] == sweep]
        if not sweep_rows:
            raise ValueError(f"records contain no {sweep} sweep")
        values = sorted({float(row["condition"]["value"]) for row in sweep_rows})
        groups[sweep] = {}
        for method in ("lasso", "ols"):
            series = []
            for value in values:
                selected = [
                    row for row in sweep_rows
                    if float(row["condition"]["value"]) == value
                ]
                selected.sort(key=lambda row: int(row["repetition"]))
                repetitions = [int(row["repetition"]) for row in selected]
                errors = np.asarray(
                    [
                        row["methods"][method]["held_out_drift"][
                            "species_balanced_relative"
                        ]
                        for row in selected
                    ],
                    dtype=float,
                )
                if (
                    errors.size < 1
                    or not np.all(np.isfinite(errors))
                    or len(repetitions) != len(set(repetitions))
                ):
                    raise ValueError("preview records are missing finite unique repetitions")
                series.append(
                    {
                        "value": value,
                        "repetitions": repetitions,
                        "errors": errors.tolist(),
                        "mean": float(np.mean(errors)),
                        "minimum": float(np.min(errors)),
                        "maximum": float(np.max(errors)),
                    }
                )
            groups[sweep][method] = series
    return groups


def nominal_preview_records(records):
    """Return one copy per repetition of the saved nominal fitted coefficients."""
    selected = [
        row for row in records
        if row["condition"]["sweep"] == "observation_noise"
        and int(row["condition"]["prey_particles"]) == 100
        and np.isclose(float(row["condition"]["observation_dt"]), 0.01)
        and np.isclose(float(row["condition"]["observation_noise"]), 0.01)
    ]
    selected.sort(key=lambda row: int(row["repetition"]))
    if not selected:
        return []
    repetitions = [int(row["repetition"]) for row in selected]
    if len(repetitions) != len(set(repetitions)):
        raise ValueError("nominal preview records contain duplicate repetitions")
    return selected


def _save_preview_figure(fig, stem):
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def _plot_sweep_preview(groups, output_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

    titles = {
        "observation_noise": "(a) Observation noise",
        "observation_dt": "(b) Observation interval",
        "prey_particles": "(c) Prey count",
    }
    labels = {
        "observation_noise": r"observation noise $\sigma_{\rm obs}$",
        "observation_dt": r"observation interval $\Delta t$",
        "prey_particles": r"prey particles $N$",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), constrained_layout=True)
    for panel_index, (axis, sweep) in enumerate(zip(axes, PREVIEW_SWEEPS)):
        for method in ("lasso", "ols"):
            style = PREVIEW_METHOD_STYLE[method]
            series = groups[sweep][method]
            x = np.asarray([row["value"] for row in series], dtype=float)
            means = np.asarray([row["mean"] for row in series], dtype=float)
            repetitions = sorted({rep for row in series for rep in row["repetitions"]})
            for repetition in repetitions:
                individual = []
                for row in series:
                    position = row["repetitions"].index(repetition)
                    individual.append(row["errors"][position])
                axis.plot(
                    x,
                    individual,
                    color=style["color"],
                    linewidth=0.65,
                    alpha=0.25,
                )
            axis.plot(
                x,
                means,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.35,
                markersize=4.5,
                label=style["label"],
            )
        axis.set_yscale("log")
        if sweep == "observation_noise":
            positive = np.asarray(
                [row["value"] for row in groups[sweep]["lasso"] if row["value"] > 0]
            )
            axis.set_xscale("symlog", linthresh=float(positive.min() / 2.0))
            axis.set_xlim(left=0.0)
            shown = np.unique(np.concatenate(([0.0], positive[[0, -1]])))
            axis.xaxis.set_major_locator(FixedLocator(shown))
            axis.xaxis.set_major_formatter(
                FixedFormatter(["0" if value == 0 else f"{value:g}" for value in shown])
            )
            axis.xaxis.set_minor_locator(NullLocator())
            axis.xaxis.set_minor_formatter(NullFormatter())
        elif sweep == "observation_dt":
            axis.set_xscale("log")
            shown = np.asarray([row["value"] for row in groups[sweep]["lasso"]])
            axis.xaxis.set_major_locator(FixedLocator(shown))
            axis.xaxis.set_major_formatter(FixedFormatter([f"{value:g}" for value in shown]))
            axis.xaxis.set_minor_locator(NullLocator())
            axis.xaxis.set_minor_formatter(NullFormatter())
        else:
            shown = np.asarray([row["value"] for row in groups[sweep]["lasso"]])
            axis.xaxis.set_major_locator(FixedLocator(shown))
            axis.xaxis.set_major_formatter(FixedFormatter([f"{value:g}" for value in shown]))
        axis.set_title(titles[sweep])
        axis.set_xlabel(labels[sweep])
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if panel_index == 0:
            axis.set_ylabel("relative held-out drift error")
            axis.legend(loc="best", frameon=False)
    count = len({rep for row in groups["observation_noise"]["lasso"] for rep in row["repetitions"]})
    fig.suptitle(f"Sampled-pair $P=10N$ smoke preview ($R={count}$)", fontsize=11)
    fig.text(
        0.5,
        -0.015,
        "Thin lines: individual repetitions; markers: arithmetic mean. Preview only; no uncertainty interval.",
        ha="center",
        fontsize=8,
    )
    files = _save_preview_figure(
        fig, output_directory / "predator_prey_sampled_pair_sweeps_preview"
    )
    plt.close(fig)
    return files


def _plot_nominal_profile_preview(records, output_directory):
    if not records:
        return ()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prey_basis, predator_basis, _ = make_predator_prey_bases(
        0.05, [0.25, 0.5, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    truth = (
        np.asarray([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.4, 0.0, 0.0]),
    )
    radius = np.linspace(0.0, 2.0, 500)
    displacement = np.column_stack([radius, np.zeros_like(radius)])
    features = (
        prey_basis(displacement)[:, :, 0],
        prey_basis(displacement)[:, :, 0],
        predator_basis(displacement)[:, :, 0],
    )
    channel_names = ("prey_prey", "prey_predator", "predator_prey")
    titles = ("(a) Prey--prey", "(b) Predator on prey", "(c) Prey on predator")
    truth_curves = [basis @ theta for basis, theta in zip(features, truth)]
    curves = []
    for row in records:
        curves.append(
            [
                basis
                @ np.asarray(row["methods"]["lasso"]["channels"][channel]["coefficients"])
                for basis, channel in zip(features, channel_names)
            ]
        )
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), constrained_layout=True)
    for channel_index, axis in enumerate(axes):
        channel_curves = np.asarray([row[channel_index] for row in curves])
        for repetition, curve in zip(
            [int(row["repetition"]) for row in records], channel_curves
        ):
            axis.plot(
                radius,
                curve,
                color="#999999",
                linewidth=0.9,
                alpha=0.65,
                label=f"LASSO repetition {repetition}" if channel_index == 0 else None,
            )
        axis.plot(
            radius,
            truth_curves[channel_index],
            color="#000000",
            linewidth=1.6,
            label="true" if channel_index == 0 else None,
        )
        axis.plot(
            radius,
            np.median(channel_curves, axis=0),
            color=PREVIEW_METHOD_STYLE["lasso"]["color"],
            linestyle="--",
            linewidth=1.5,
            label="median LASSO" if channel_index == 0 else None,
        )
        axis.axhline(0.0, color="#666666", linewidth=0.6)
        axis.set_title(titles[channel_index])
        axis.set_xlabel(r"distance $r$")
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if channel_index == 0:
            axis.set_ylabel("directed force")
            axis.legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("Nominal sampled-pair smoke profile from saved LASSO coefficients", fontsize=11)
    files = _save_preview_figure(
        fig, output_directory / "predator_prey_sampled_pair_nominal_profiles_preview"
    )
    plt.close(fig)
    return files


def generate_preview_figures(run_directory, output_directory=None):
    """Generate preview-only figures from an existing records.json.

    No trajectory simulation, weak-system assembly, regression, model selection,
    held-out evaluation, bootstrap, or metric recomputation is performed.
    """
    run_directory = Path(run_directory).resolve()
    records_path = run_directory / "records.json"
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    records = read_json(records_path)
    groups = preview_sweep_groups(records)
    nominal = nominal_preview_records(records)
    output_directory = (
        run_directory / "preview"
        if output_directory is None
        else Path(output_directory).resolve()
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    sweep_files = _plot_sweep_preview(groups, output_directory)
    profile_files = _plot_nominal_profile_preview(nominal, output_directory)
    trajectory_candidates = [
        run_directory / "display_trajectory.npz",
        run_directory / "trajectory.npz",
    ]
    saved_trajectory = next((path for path in trajectory_candidates if path.is_file()), None)
    generated = [*sweep_files, *profile_files]
    metadata = {
        "task": "predator_prey_sampled_pair_preview_figures",
        "source_records": str(records_path),
        "source_records_sha256": digest(records_path),
        "record_count": len(records),
        "repetitions": sorted({int(row["repetition"]) for row in records}),
        "figure_data_policy": "records.json only; no experiment, fit, selection, held-out evaluation, bootstrap, or metric recomputation",
        "sweep_statistic": "arithmetic mean with individual repetition lines; no uncertainty interval for R=2 smoke preview",
        "nominal_profile": (
            "reconstructed only for plotting from saved nominal LASSO coefficient vectors and the fixed audited dictionary"
            if profile_files
            else "not generated because no saved nominal coefficient records were present"
        ),
        "trajectory_generated": False,
        "trajectory_note": (
            "a trajectory artifact exists but was intentionally not read because this mode is records-only"
            if saved_trajectory is not None
            else "no saved trajectory artifact is present in the smoke output"
        ),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": digest(Path(__file__)),
        "files": {
            path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in generated
        },
        "limitations": [
            "smoke preview with two repetitions and a 0.4 time horizon",
            "not a scientific uncertainty estimate or manuscript-ready result",
            "profile curves are a visualization of saved coefficients, not a rerun",
        ],
    }
    write_json(output_directory / "metadata.json", metadata)
    return metadata


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


def run_preview(requested, destination, *, workers=1):
    validate_preview_protocol(requested)
    base, frozen_audit = audit_frozen_reference(requested)
    spec = effective_protocol(requested, base)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "jobs").mkdir()
    (destination / "lasso_paths").mkdir()
    write_json(destination / "requested_protocol.json", requested)
    write_json(destination / "effective_protocol.json", spec)
    write_json(destination / "frozen_reference_audit.json", frozen_audit)
    hashes = snapshot_sources(destination)
    started = perf_counter()
    repetitions = int(spec["evaluation"]["scientific_repetitions"])
    manifest = {
        "task": requested["task"],
        "status": "running",
        "smoke_only": True,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "workers": int(workers),
        "repetitions": repetitions,
        "condition_count": len(condition_plan(spec)),
        "completed_repetitions": [],
        "source_sha256": hashes,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "frozen_reference_audit_status": frozen_audit["status"],
    }
    write_json(destination / "manifest.json", manifest)
    results = []
    try:
        if workers == 1:
            for repetition in range(repetitions):
                result = run_repetition((repetition, spec))
                results.append(result)
                write_json(
                    destination / "jobs" / f"repetition-{repetition:03d}.json",
                    result,
                )
                write_json(
                    destination
                    / "lasso_paths"
                    / f"repetition-{repetition:03d}.json",
                    result["lasso_paths"],
                )
                manifest["completed_repetitions"].append(repetition)
                manifest["updated_utc"] = utc_now()
                write_json(destination / "manifest.json", manifest)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(run_repetition, (repetition, spec)): repetition
                    for repetition in range(repetitions)
                }
                for future in as_completed(futures):
                    result = future.result()
                    repetition = int(result["repetition"])
                    results.append(result)
                    write_json(
                        destination / "jobs" / f"repetition-{repetition:03d}.json",
                        result,
                    )
                    write_json(
                        destination
                        / "lasso_paths"
                        / f"repetition-{repetition:03d}.json",
                        result["lasso_paths"],
                    )
                    manifest["completed_repetitions"] = sorted(
                        manifest["completed_repetitions"] + [repetition]
                    )
                    manifest["updated_utc"] = utc_now()
                    write_json(destination / "manifest.json", manifest)

        results.sort(key=lambda row: row["repetition"])
        records = [record for result in results for record in result["records"]]
        conditions = condition_plan(spec)
        pair_report = pair_sampling_summary(records)
        if pair_report["status"] != "passed":
            raise RuntimeError("pair-sampling independence or exact-cross-species check failed")
        write_json(destination / "records.json", records)
        write_json(destination / "summary.json", summarize(records, conditions))
        write_json(destination / "pair_sampling_report.json", pair_report)
        artifacts = {
            name: digest(destination / name)
            for name in (
                "requested_protocol.json",
                "effective_protocol.json",
                "frozen_reference_audit.json",
                "source_manifest.json",
                "records.json",
                "summary.json",
                "pair_sampling_report.json",
            )
        }
        manifest.update(
            {
                "status": "smoke_complete",
                "updated_utc": utc_now(),
                "completed_utc": utc_now(),
                "completed_repetitions": list(range(repetitions)),
                "elapsed_seconds": perf_counter() - started,
                "repetition_elapsed_seconds": [
                    result["elapsed_seconds"] for result in results
                ],
                "pair_sampling_report_status": pair_report["status"],
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
    parser.add_argument(
        "--figures-from-output",
        type=Path,
        help="generate figures from an existing records.json without rerunning",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        help="new figure directory; defaults to RUN/preview",
    )
    args = parser.parse_args()
    if args.figures_from_output is not None:
        if args.output is not None:
            parser.error("--figures-from-output cannot be combined with --output")
        metadata = generate_preview_figures(
            args.figures_from_output, args.figure_output
        )
        print(
            json.dumps(
                {
                    "status": "preview_figures_complete",
                    "output": str(
                        (
                            args.figures_from_output.resolve() / "preview"
                            if args.figure_output is None
                            else args.figure_output.resolve()
                        )
                    ),
                    "files": sorted(metadata["files"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.figure_output is not None:
        parser.error("--figure-output requires --figures-from-output")
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    requested = read_json(args.config)
    if args.output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = (
            ROOT
            / "outputs/validation/predator_prey_sampled_pair_preview"
            / f"smoke-{timestamp}"
        )
    manifest = run_preview(requested, args.output, workers=args.workers)
    print(Path(args.output).resolve())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "completed_repetitions": manifest["completed_repetitions"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
