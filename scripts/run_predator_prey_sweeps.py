"""Run paired two-dimensional predator--prey robustness sweeps.

The scientific protocol varies observation noise, observation spacing, and prey
count one at a time.  LASSO penalties for the prey and predator equations are
selected separately on independent validation trajectories.  OLS uses exactly
the same raw weak systems.  Held-out latent trajectories are touched only after
all fits for a repetition have been frozen.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predator_prey import (
    assemble_predator_prey_weak_systems,
    channel_kernels,
    make_predator_prey_bases,
    predator_prey_velocity,
    simulate_predator_prey,
)
from scripts.run_1d_hat_lasso import select_lasso
from tests import make_test_functions
from time_tests import make_time_test_functions


DEFAULT_CONFIG = ROOT / "configs/experiments/predator_prey_sweeps.json"
SOURCE_FILES = (
    "predator_prey.py",
    "features.py",
    "tests.py",
    "time_tests.py",
    "scripts/run_1d_hat_lasso.py",
    "scripts/run_predator_prey_sweeps.py",
    "configs/experiments/predator_prey_sweeps.json",
    "pyproject.toml",
)
SPLIT_OFFSETS = {"training": 0, "validation": 10000, "test": 20000}
CHANNEL_NAMES = ("prey_prey", "prey_predator", "predator_prey")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    """Atomically write strict JSON so interrupted jobs remain resumable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_protocol(spec):
    """Reject drift from the approved scientific protocol."""
    if spec.get("task") != "predator_prey_robustness_sweeps":
        raise ValueError("wrong predator--prey sweep task")
    if spec.get("smoke_only") is not False:
        raise ValueError("the requested protocol must be scientific, not smoke")
    expected_model = {
        "dimension": 2, "prey_particles": 100, "a": 1.0, "b": 0.2,
        "c": 0.4, "p": 3.0, "delta": 0.05,
        "prey_initial_box": [-0.5, 0.5],
        "predator_initial_box": [-0.5, 0.5], "solver": "rk4",
    }
    if spec.get("model") != expected_model:
        raise ValueError("predator--prey model changed")
    observation = spec.get("observation", {})
    expected_observation = {
        "training_trajectories": 4, "validation_trajectories": 4,
        "test_trajectories": 1, "horizon": 3.0, "dt": 0.01,
    }
    if {key: observation.get(key) for key in expected_observation} != expected_observation:
        raise ValueError("observation split or time grid changed")
    if float(observation.get("observation_noise", -1.0)) not in {0.001, 0.01}:
        raise ValueError("unapproved baseline observation noise")
    if spec.get("sweeps") != {
        "observation_noise": [0.0, 3e-4, 1e-3, 3e-3, 1e-2],
        "observation_dt": [0.04, 0.02, 0.01, 0.005],
        "prey_particles": [25, 50, 100, 200],
    }:
        raise ValueError("approved sweep grid changed")
    if spec.get("dictionary") != {
        "prey_exponential_scales": [0.25, 0.5, 1.0, 2.0],
        "predator_inverse_powers": [1.0, 2.0, 3.0, 4.0, 5.0],
        "prey_prey_true": [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        "prey_predator_true": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        "predator_prey_true": [0.0, 0.0, 0.4, 0.0, 0.0],
    }:
        raise ValueError("predator--prey dictionary changed")
    if spec.get("weak_tests") != {
        "spatial_type": "gaussian",
        "grid_coordinates": [-1.5, -0.75, 0.0, 0.75, 1.5],
        "prey_width": 0.5, "predator_width": 0.7,
        "temporal_type": "bump", "temporal_count": 8,
        "temporal_support_factor": 1.8, "normalize_rows": True,
    }:
        raise ValueError("weak-test protocol changed")
    if spec.get("estimator") != {
        "method": "raw_lasso", "alpha_count": 60,
        "alpha_min_ratio": 1e-7, "max_iter": 200000, "tol": 1e-8,
        "fit_intercept": False, "standardize": False,
        "selection": "cyclic",
        "selection_metric": "independent_validation_weak_residual_mse",
        "truth_used_for_selection": False,
        "test_error_used_for_selection": False,
        "post_lasso_refit": False, "coefficient_thresholding": False,
        "report_nonzero_threshold": 1e-8,
    }:
        raise ValueError("estimator protocol changed")
    if spec.get("evaluation") != {
        "scientific_repetitions": 20, "profile_radius_min": 0.0,
        "profile_radius_max": 2.0, "profile_grid_size": 500,
        "drift_evaluation_interval": 0.05,
        "species_balance": "equal_relative_energy_weight",
    }:
        raise ValueError("evaluation protocol changed")
    pairing = spec.get("pairing", {})
    if pairing != {
        "dt_common_fine": 0.005,
        "dt_trajectory_policy": "simulate_once_at_common_fine_dt_and_subsample",
        "observation_noise_policy": "shared_latent_trajectory_and_shared_unit_noise",
        "particle_policy": "shared_seed_with_nested_initial_prey_and_observation_noise",
        "training_start": 42000, "validation_offset": 10000,
        "test_offset": 20000, "repetition_stride": 97,
    }:
        raise ValueError("pairing or seed protocol changed")
    if spec.get("smoke") != {
        "repetitions": 1, "training_trajectories": 1,
        "validation_trajectories": 1, "test_trajectories": 1,
        "horizon": 0.4,
    }:
        raise ValueError("smoke protocol changed")


def effective_protocol(requested, smoke=False):
    validate_protocol(requested)
    spec = deepcopy(requested)
    spec["smoke_only"] = bool(smoke)
    if smoke:
        settings = spec["smoke"]
        spec["evaluation"]["scientific_repetitions"] = settings["repetitions"]
        for name in ("training_trajectories", "validation_trajectories", "test_trajectories", "horizon"):
            spec["observation"][name] = settings[name]
    return spec


def condition_plan(spec):
    """Return all labelled points; the three labelled baselines share a fit."""
    baseline = spec["observation"]
    base_n = spec["model"]["prey_particles"]
    definitions = (
        ("observation_noise", spec["sweeps"]["observation_noise"]),
        ("observation_dt", spec["sweeps"]["observation_dt"]),
        ("prey_particles", spec["sweeps"]["prey_particles"]),
    )
    points = []
    for sweep, values in definitions:
        for index, value in enumerate(values):
            condition = {
                "id": f"{sweep}-{index:02d}", "sweep": sweep,
                "index": index, "value": value,
                "prey_particles": base_n, "observation_dt": baseline["dt"],
                "observation_noise": baseline["observation_noise"],
            }
            condition[sweep] = value
            points.append(condition)
    return points


def condition_signature(condition):
    return (
        int(condition["prey_particles"]),
        float(condition["observation_dt"]),
        float(condition["observation_noise"]),
    )


def _integer_ratio(coarse, fine):
    ratio = int(round(float(coarse) / float(fine)))
    if ratio < 1 or not np.isclose(ratio * fine, coarse, rtol=0.0, atol=1e-12):
        raise ValueError(f"{coarse} is not an integer multiple of {fine}")
    return ratio


def observation_view(prey, predator, latent_dt, unit_prey_noise, unit_predator_noise,
                     noise_dt, observation_dt, sigma):
    """Subsample latent paths and one shared fine-grid unit-noise realization."""
    latent_step = _integer_ratio(observation_dt, latent_dt)
    noise_step = _integer_ratio(observation_dt, noise_dt)
    observed_prey = prey[:, ::latent_step]
    observed_predator = predator[:, ::latent_step]
    prey_noise = unit_prey_noise[:prey.shape[0], ::noise_step, :prey.shape[2]]
    predator_noise = unit_predator_noise[:predator.shape[0], ::noise_step]
    if prey_noise.shape != observed_prey.shape or predator_noise.shape != observed_predator.shape:
        raise ValueError("paired noise and latent trajectory grids do not match")
    time = np.arange(observed_prey.shape[1], dtype=float) * float(observation_dt)
    return (
        observed_prey + float(sigma) * prey_noise,
        observed_predator + float(sigma) * predator_noise,
        time,
    )


def split_seed(spec, repetition, split):
    pairing = spec["pairing"]
    offsets = {
        "training": 0,
        "validation": pairing["validation_offset"],
        "test": pairing["test_offset"],
    }
    return int(pairing["training_start"] + pairing["repetition_stride"] * repetition + offsets[split])


def paired_inputs(spec, repetition, split):
    """Generate nested initial states and fine-grid noise without RNG-shape coupling."""
    count = int(spec["observation"][f"{split}_trajectories"])
    max_particles = max(spec["sweeps"]["prey_particles"])
    fine_dt = float(spec["pairing"]["dt_common_fine"])
    steps = int(round(float(spec["observation"]["horizon"]) / fine_dt))
    seed = split_seed(spec, repetition, split)
    streams = np.random.SeedSequence(seed).spawn(4)
    prey_rng, predator_rng, prey_noise_rng, predator_noise_rng = map(np.random.default_rng, streams)
    prey_box = spec["model"]["prey_initial_box"]
    predator_box = spec["model"]["predator_initial_box"]
    return {
        "seed": seed,
        "initial_prey": prey_rng.uniform(*prey_box, size=(count, max_particles, 2)),
        "initial_predator": predator_rng.uniform(*predator_box, size=(count, 1, 2)),
        "unit_prey_noise": prey_noise_rng.normal(size=(count, steps + 1, max_particles, 2)),
        "unit_predator_noise": predator_noise_rng.normal(size=(count, steps + 1, 1, 2)),
    }


def model_objects(spec):
    model, dictionary = spec["model"], spec["dictionary"]
    prey_basis, predator_basis, metadata = make_predator_prey_bases(
        model["delta"], dictionary["prey_exponential_scales"],
        dictionary["predator_inverse_powers"],
    )
    truth = tuple(np.asarray(dictionary[name], dtype=float) for name in (
        "prey_prey_true", "prey_predator_true", "predator_prey_true"
    ))
    return prey_basis, predator_basis, metadata, truth, channel_kernels(
        *truth, prey_basis, predator_basis
    )


def weak_objects(spec, time):
    weak = spec["weak_tests"]
    coordinates = np.asarray(weak["grid_coordinates"], dtype=float)
    centers = np.array(np.meshgrid(coordinates, coordinates)).reshape(2, -1).T
    prey_tests = make_test_functions({
        "centers": centers, "s": weak["prey_width"], "type": weak["spatial_type"]
    })
    predator_tests = make_test_functions({
        "centers": centers, "s": weak["predator_width"], "type": weak["spatial_type"]
    })
    temporal = make_time_test_functions({"time_tests": {
        "K": weak["temporal_count"], "type": weak["temporal_type"],
        "s_factor": weak["temporal_support_factor"],
    }}, time)
    return prey_tests, predator_tests, temporal


def assemble(spec, prey, predator, time, prey_basis, predator_basis):
    prey_tests, predator_tests, temporal = weak_objects(spec, time)
    return assemble_predator_prey_weak_systems(
        prey, predator, time, prey_tests=prey_tests,
        predator_tests=predator_tests, prey_basis=prey_basis,
        predator_basis=predator_basis, time_tests=temporal,
        normalize_rows=spec["weak_tests"]["normalize_rows"],
    )


def simulate_latent(spec, inputs, particle_count, integration_dt, kernels):
    count = inputs["initial_prey"].shape[0]
    steps = int(round(float(spec["observation"]["horizon"]) / integration_dt))
    prey, predator, time, diagnostics = simulate_predator_prey(
        particle_count=int(particle_count), steps=steps, dt=float(integration_dt),
        repetitions=count, kernels=kernels, seed=inputs["seed"],
        prey_box=spec["model"]["prey_initial_box"],
        predator_box=spec["model"]["predator_initial_box"],
        solver=spec["model"]["solver"],
        initial_prey=inputs["initial_prey"][:, :particle_count],
        initial_predator=inputs["initial_predator"],
    )
    if not all(diagnostics["finite"]):
        raise RuntimeError("true predator--prey trajectory diverged")
    return prey, predator, time


def latent_for_condition(spec, condition, inputs, kernels, cache):
    particles = int(condition["prey_particles"])
    base_particles = int(spec["model"]["prey_particles"])
    integration_dt = (
        float(spec["pairing"]["dt_common_fine"])
        if particles == base_particles else float(spec["observation"]["dt"])
    )
    key = (particles, integration_dt)
    if key not in cache:
        cache[key] = simulate_latent(spec, inputs, particles, integration_dt, kernels)
    return cache[key], integration_dt


def _safe_condition_number(matrix):
    value = float(np.linalg.cond(matrix))
    return value if np.isfinite(value) else None


def system_diagnostics(training, validation):
    result = {}
    for species in ("prey", "predator"):
        train_A, validation_A = training[species]["A_flat"], validation[species]["A_flat"]
        result[species] = {
            "training_rows": int(train_A.shape[0]),
            "validation_rows": int(validation_A.shape[0]),
            "columns": int(train_A.shape[1]),
            "training_rank": int(np.linalg.matrix_rank(train_A)),
            "training_condition_number": _safe_condition_number(train_A),
        }
    return result


def residual_metrics(theta, training, validation):
    return {
        "training_weak_residual_mse": float(np.mean((training["A_flat"] @ theta - training["b_flat"]) ** 2)),
        "validation_weak_residual_mse": float(np.mean((validation["A_flat"] @ theta - validation["b_flat"]) ** 2)),
    }


def selection_diagnostics(selection):
    selected = selection["selected"]
    return {
        "alpha": float(selected["alpha_sk"]),
        "alpha_over_max": float(selected["alpha_over_max"]),
        "selected_index": int(selected["index"]),
        "selected_at_grid_boundary": bool(selection["selected_at_grid_boundary"]),
        "selected_iterations": int(selected["iterations"]),
        "selected_dual_gap": float(selected["dual_gap"]),
        "selected_nonzero_exact": int(selected["nonzero_exact"]),
        "selected_nonzero_reported": int(selected["nonzero_reported"]),
        "nonconverged_candidates": int(sum(not row["converged"] for row in selection["path"])),
        "convergence_warnings": int(sum(row["convergence_warnings"] for row in selection["path"])),
        "candidate_count": len(selection["path"]),
    }


def profile_curves(radius, coefficients, prey_basis, predator_basis):
    displacement = np.column_stack([radius, np.zeros_like(radius)])
    kernels = channel_kernels(*coefficients, prey_basis, predator_basis)
    return tuple(kernel(displacement)[:, 0] for kernel in kernels)


def relative_l2(estimate, truth):
    denominator = float(np.sum(np.asarray(truth) ** 2))
    if denominator <= 0.0:
        return None
    return float(np.sqrt(np.sum((np.asarray(estimate) - truth) ** 2) / denominator))


def coefficient_and_profile_metrics(coefficients, truth, spec, prey_basis, predator_basis):
    radius = np.linspace(
        spec["evaluation"]["profile_radius_min"],
        spec["evaluation"]["profile_radius_max"],
        spec["evaluation"]["profile_grid_size"],
    )
    estimated_profiles = profile_curves(radius, coefficients, prey_basis, predator_basis)
    true_profiles = profile_curves(radius, truth, prey_basis, predator_basis)
    channels = {}
    for name, estimate, target, estimated_profile, true_profile in zip(
            CHANNEL_NAMES, coefficients, truth, estimated_profiles, true_profiles):
        channels[name] = {
            "coefficients": np.asarray(estimate).tolist(),
            "coefficient_error_l2": float(np.linalg.norm(np.asarray(estimate) - target)),
            "coefficient_error_relative": relative_l2(estimate, target),
            "profile_error_relative": relative_l2(estimated_profile, true_profile),
        }
    return channels


def species_balanced_drift_error(prey, predator, truth_kernels, estimated_kernels, stride=1):
    """Give prey and predator equal weight after species-wise energy normalization."""
    totals = {"prey_error": 0.0, "prey_truth": 0.0,
              "predator_error": 0.0, "predator_truth": 0.0}
    for repetition in range(prey.shape[0]):
        for time_index in range(0, prey.shape[1], int(stride)):
            x, z = prey[repetition, time_index], predator[repetition, time_index, 0]
            true_x, true_z = predator_prey_velocity(x, z, truth_kernels)
            fitted_x, fitted_z = predator_prey_velocity(x, z, estimated_kernels)
            totals["prey_error"] += float(np.sum((fitted_x - true_x) ** 2))
            totals["prey_truth"] += float(np.sum(true_x ** 2))
            totals["predator_error"] += float(np.sum((fitted_z - true_z) ** 2))
            totals["predator_truth"] += float(np.sum(true_z ** 2))
    prey_ratio = totals["prey_error"] / totals["prey_truth"]
    predator_ratio = totals["predator_error"] / totals["predator_truth"]
    return {
        "species_balanced_relative": float(np.sqrt(0.5 * (prey_ratio + predator_ratio))),
        "prey_relative": float(np.sqrt(prey_ratio)),
        "predator_relative": float(np.sqrt(predator_ratio)),
        **totals,
    }


def fit_condition(spec, condition, train_inputs, validation_inputs, latent_caches,
                  prey_basis, predator_basis, truth_kernels):
    observed = []
    for inputs, cache in zip((train_inputs, validation_inputs), latent_caches):
        (prey, predator, _), latent_dt = latent_for_condition(
            spec, condition, inputs, truth_kernels, cache
        )
        observed.append(observation_view(
            prey, predator, latent_dt, inputs["unit_prey_noise"],
            inputs["unit_predator_noise"], spec["pairing"]["dt_common_fine"],
            condition["observation_dt"], condition["observation_noise"],
        ))
    training = assemble(spec, *observed[0], prey_basis, predator_basis)
    validation = assemble(spec, *observed[1], prey_basis, predator_basis)
    fits, paths = {}, {}
    selections = {}
    for species in ("prey", "predator"):
        selection, _ = select_lasso(
            training[species]["A_flat"], training[species]["b_flat"],
            validation[species]["A_flat"], validation[species]["b_flat"],
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
        "lasso": (lasso_prey[:channel_count], lasso_prey[channel_count:], lasso_predator),
        "ols": (ols_prey[:channel_count], ols_prey[channel_count:], ols_predator),
    }
    fits["lasso"] = {
        "prey": residual_metrics(lasso_prey, training["prey"], validation["prey"]),
        "predator": residual_metrics(lasso_predator, training["predator"], validation["predator"]),
        "selection": {name: selection_diagnostics(selections[name]) for name in ("prey", "predator")},
    }
    fits["ols"] = {
        "prey": {**residual_metrics(ols_prey, training["prey"], validation["prey"]),
                 "rank": int(ols_prey_rank), "singular_values": ols_prey_singular.tolist()},
        "predator": {**residual_metrics(ols_predator, training["predator"], validation["predator"]),
                     "rank": int(ols_predator_rank), "singular_values": ols_predator_singular.tolist()},
    }
    return {
        "arrays": arrays, "methods": fits,
        "systems": system_diagnostics(training, validation), "paths": paths,
    }


def run_repetition(payload):
    repetition, spec = payload
    started = perf_counter()
    prey_basis, predator_basis, _, truth, truth_kernels = model_objects(spec)
    train_inputs = paired_inputs(spec, repetition, "training")
    validation_inputs = paired_inputs(spec, repetition, "validation")
    conditions = condition_plan(spec)
    fit_cache, labelled_fits, paths = {}, {}, {}
    with threadpool_limits(limits=1):
        train_cache, validation_cache = {}, {}
        for condition in conditions:
            signature = condition_signature(condition)
            if signature not in fit_cache:
                fit_cache[signature] = fit_condition(
                    spec, condition, train_inputs, validation_inputs,
                    (train_cache, validation_cache), prey_basis, predator_basis,
                    truth_kernels,
                )
            labelled_fits[condition["id"]] = fit_cache[signature]
            paths[condition["id"]] = fit_cache[signature]["paths"]
        # Selection is complete and coefficients are frozen before test generation.
        del train_inputs, validation_inputs, train_cache, validation_cache
        test_inputs = paired_inputs(spec, repetition, "test")
        test_cache, evaluation_cache = {}, {}
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
                            test_prey, test_predator, truth_kernels,
                            estimated_kernels, stride,
                        ),
                    }
                methods[method].update(deepcopy(evaluation_cache[evaluation_key]))
            records.append({
                "repetition": repetition, "condition": deepcopy(condition),
                "seeds": {name: split_seed(spec, repetition, name)
                          for name in ("training", "validation", "test")},
                "systems": deepcopy(fitted["systems"]), "methods": methods,
            })
    return {
        "repetition": repetition, "records": records, "lasso_paths": paths,
        "elapsed_seconds": perf_counter() - started,
    }


def finite_summary(values):
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "count": len(values), "finite_count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "q25": float(np.quantile(finite, 0.25)) if finite.size else None,
        "q75": float(np.quantile(finite, 0.75)) if finite.size else None,
    }


def summarize(records, conditions):
    summary = {"conditions": {}}
    for condition in conditions:
        rows = [row for row in records if row["condition"]["id"] == condition["id"]]
        item = {"condition": condition, "methods": {}}
        for method in ("lasso", "ols"):
            metrics = {
                name: finite_summary([row["methods"][method]["held_out_drift"][name] for row in rows])
                for name in ("species_balanced_relative", "prey_relative", "predator_relative")
            }
            metrics["channels"] = {
                channel: {
                    metric: finite_summary([
                        row["methods"][method]["channels"][channel][metric] for row in rows
                    ])
                    for metric in ("coefficient_error_l2", "coefficient_error_relative", "profile_error_relative")
                }
                for channel in CHANNEL_NAMES
            }
            if method == "lasso":
                metrics["convergence"] = {
                    species: {
                        "boundary_selections": int(sum(
                            row["methods"][method]["selection"][species]["selected_at_grid_boundary"]
                            for row in rows
                        )),
                        "nonconverged_candidates": int(sum(
                            row["methods"][method]["selection"][species]["nonconverged_candidates"]
                            for row in rows
                        )),
                        "convergence_warnings": int(sum(
                            row["methods"][method]["selection"][species]["convergence_warnings"]
                            for row in rows
                        )),
                    } for species in ("prey", "predator")
                }
            item["methods"][method] = metrics
        summary["conditions"][condition["id"]] = item
    return summary


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def _new_batch(requested, spec, destination, workers):
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "jobs").mkdir()
    (destination / "lasso_paths").mkdir()
    write_json(destination / "requested_protocol.json", requested)
    write_json(destination / "effective_protocol.json", spec)
    manifest = {
        "status": "running", "smoke_only": bool(spec["smoke_only"]),
        "created_utc": utc_now(), "updated_utc": utc_now(),
        "requested_protocol_sha256": payload_digest(requested),
        "effective_protocol_sha256": payload_digest(spec),
        "source_sha256": source_hashes(), "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__, "workers": workers,
        "repetitions": spec["evaluation"]["scientific_repetitions"],
        "condition_count": len(condition_plan(spec)), "completed_repetitions": [],
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def _resume_batch(requested, destination, workers):
    manifest = read_json(destination / "manifest.json")
    spec = read_json(destination / "effective_protocol.json")
    if payload_digest(requested) != manifest["requested_protocol_sha256"]:
        raise ValueError("resume requested protocol mismatch")
    if payload_digest(spec) != manifest["effective_protocol_sha256"]:
        raise ValueError("resume effective protocol is corrupt or changed")
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("resume source hash mismatch")
    expected_smoke = bool(spec["smoke_only"])
    if expected_smoke not in (True, False):
        raise ValueError("invalid saved smoke mode")
    manifest.update({"status": "running", "updated_utc": utc_now(), "workers": workers})
    write_json(destination / "manifest.json", manifest)
    return spec, manifest


def run_batch(requested, destination, *, workers=1, smoke=False, resume=False):
    """Run or resume a batch, checkpointing every completed repetition."""
    validate_protocol(requested)
    destination = Path(destination).resolve()
    started = perf_counter()
    if resume:
        spec, manifest = _resume_batch(requested, destination, workers)
    else:
        spec = effective_protocol(requested, smoke)
        manifest = _new_batch(requested, spec, destination, workers)
    repetitions = int(spec["evaluation"]["scientific_repetitions"])
    completed = {
        int(path.stem.split("-")[-1]) for path in (destination / "jobs").glob("repetition-*.json")
    }
    pending = [index for index in range(repetitions) if index not in completed]

    def save_result(result):
        index = int(result["repetition"])
        write_json(destination / "jobs" / f"repetition-{index:03d}.json", result)
        write_json(destination / "lasso_paths" / f"repetition-{index:03d}.json", result["lasso_paths"])
        completed.add(index)
        manifest.update({
            "status": "running", "updated_utc": utc_now(),
            "completed_repetitions": sorted(completed),
        })
        write_json(destination / "manifest.json", manifest)

    try:
        if workers == 1:
            for index in pending:
                save_result(run_repetition((index, spec)))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(run_repetition, (index, spec)): index for index in pending
                }
                for future in as_completed(futures):
                    save_result(future.result())
        jobs = [
            read_json(destination / "jobs" / f"repetition-{index:03d}.json")
            for index in range(repetitions)
        ]
        records = [record for job in jobs for record in job["records"]]
        write_json(destination / "records.json", records)
        write_json(destination / "summary.json", summarize(records, condition_plan(spec)))
        artifacts = {
            name: digest(destination / name)
            for name in ("requested_protocol.json", "effective_protocol.json", "records.json", "summary.json")
        }
        manifest.update({
            "status": "smoke_complete" if spec["smoke_only"] else "scientific_complete",
            "updated_utc": utc_now(), "completed_utc": utc_now(),
            "completed_repetitions": list(range(repetitions)),
            "elapsed_seconds_this_invocation": perf_counter() - started,
            "repetition_elapsed_seconds": [job["elapsed_seconds"] for job in jobs],
            "artifact_sha256": artifacts,
        })
        write_json(destination / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update({
            "status": "interrupted", "updated_utc": utc_now(),
            "completed_repetitions": sorted(completed),
            "last_error": f"{type(error).__name__}: {error}",
            "elapsed_seconds_this_invocation": perf_counter() - started,
        })
        write_json(destination / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    if args.resume is not None and (args.output is not None or args.smoke):
        parser.error("--resume cannot be combined with --output or --smoke")
    requested = read_json(args.config)
    if args.resume is not None:
        destination, resume = args.resume, True
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        mode = "smoke" if args.smoke else "scientific"
        destination = args.output or (
            ROOT / "outputs/validation/predator_prey_sweeps" / f"{mode}-{timestamp}"
        )
        resume = False
    manifest = run_batch(
        requested, destination, workers=args.workers, smoke=args.smoke, resume=resume
    )
    print(Path(destination).resolve())
    print(json.dumps({
        "status": manifest["status"],
        "completed_repetitions": manifest["completed_repetitions"],
        "elapsed_seconds_this_invocation": manifest["elapsed_seconds_this_invocation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
