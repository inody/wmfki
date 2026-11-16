"""Compare hat dictionaries against the 1000-particle baseline, with all six Fig. 2 items fixed in advance."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from features import compute_b_timeweak, make_trapz_weights
from scripts.prepare_1d_hat_experiment import array_hash, basis_params, matrix_diagnostics, test_family, write_json
from scripts.run_1d_hat_repetitions import SOURCES as BASE_SOURCES, validate_repetitions
from scripts.run_1d_hat_lasso import select_lasso
from scripts.run_sindy_comparison import initial_positions, reference_trajectory
from scripts.hat_sweep_numerics import (assemble_full, assemble_sampled, evaluate_model,
    evaluation_statistics, force_energy)

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat_sweeps.json"
SOURCES = list(dict.fromkeys(BASE_SOURCES + [
    "scripts/hat_sweep_numerics.py", "scripts/run_1d_hat_sweeps.py",
    "scripts/summarize_1d_hat_sweeps.py", "scripts/verify_1d_hat_sweeps.py",
    "configs/experiments/one_dimensional_hat_sweeps.json",
    "test/test_hat_sweep_numerics.py", "test/test_hat_sweeps.py",
]))
METRICS = ["force_error_weighted_rel", "trajectory_error_rel", "drift_error_rel", "trajectory_rmse"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_protocol(spec, base):
    validate_repetitions(base)
    if spec["protocol_version"] != 1 or set(spec["sweeps"]) != {"N", "dt", "sigma", "sigma_obs"}:
        raise ValueError("all four accuracy sweeps are required")
    for key, values in spec["sweeps"].items():
        if not values or values != sorted(set(values)) or not all(np.isfinite(v) for v in values):
            raise ValueError("grid must be finite and strictly increasing")
        if base["simulation"][key] not in values:
            raise ValueError("each grid must include the common baseline")
        if key == "N" and any(type(v) is not int or v < 2 for v in values):
            raise ValueError("invalid particle counts")
        if key in ("sigma", "sigma_obs") and any(v < 0 for v in values):
            raise ValueError("negative noise")
        if key == "dt" and any(v <= 0 or not np.isclose(base["simulation"]["horizon"] / v,
                                                       round(base["simulation"]["horizon"] / v)) for v in values):
            raise ValueError("steps must divide the fixed horizon")
    for key in ("pair_counts", "runtime_particles"):
        values = spec[key]
        if values != sorted(set(values)) or not values or any(type(v) is not int or v < 2 for v in values):
            raise ValueError("invalid positive integer grid")
    if spec["runtime_repetitions"] != 10 or spec["runtime_pair_ratio"] != 10:
        raise ValueError("runtime uses 10 repeats and P=10N")
    if any(spec[k] != v for k, v in dict(observation_seed_offset=1000000, pair_seed_base=2000000,
                                       runtime_seed_base=3000000, runtime_order_seed=4000000).items()):
        raise ValueError("seed families must stay disjoint and predeclared")


def effective_protocol(spec, base, smoke):
    spec, base = deepcopy(spec), deepcopy(base)
    if smoke:
        base["simulation"].update(N=12, horizon=.1, dt=.005)
        base["basis"]["sizes"] = [9, 17, 33, 65]
        base["replication"]["count"] = 2
        # Check only the numerical-method wiring; this is distinct from the scientific selection range.
        base["estimator"].update(alpha_count=5, max_iter=1000)
        spec["sweeps"] = {"sigma_obs": [0., .01], "sigma": [.1, .2], "N": [8, 12], "dt": [.005, .01]}
        spec["pair_counts"] = [20, 40]
        spec["runtime_particles"] = [8, 12]
        spec["runtime_repetitions"] = 2
    return spec, base


def cases(spec, base):
    result = []
    for parameter, values in spec["sweeps"].items():
        for index, value in enumerate(values):
            simulation = {**base["simulation"], parameter: value}
            baseline = simulation == base["simulation"]
            result.append(dict(id="base" if baseline else f"{parameter}-{index:02d}",
                               parameter=parameter, value=value, simulation=simulation))
    return result


def candidate_list(base):
    return [(f"hat_M{M}", basis_params(base, M)) for M in base["basis"]["sizes"]]


def save_statistics(path, statistics):
    np.savez(path, **{name + "__" + key: value for name, stat in statistics.items() for key, value in stat.items()})


def load_statistics(path):
    result = {}
    with np.load(path) as data:
        for key in data.files:
            name, metric = key.split("__")
            result.setdefault(name, {})[metric] = data[key]
    return result


def load_data(path):
    with np.load(path) as data:
        return {key: data[key] for key in ("X", "t", "X0", "driving_noise")}


def coverage(Y, t, radius=4.):
    N = Y.shape[2]
    total = Y.shape[0] * N * (N - 1) * make_trapz_weights(t).sum()
    minimum, maximum, outside = np.inf, 0., 0.
    for path in Y:
        for x, weight in zip(path, make_trapz_weights(t)):
            ordered = np.sort(x[:, 0])
            minimum = min(minimum, float(np.min(np.diff(ordered))))
            maximum = max(maximum, float(ordered[-1] - ordered[0]))
            if ordered[-1] - ordered[0] > radius:
                # Double the one-sided count starting from the smaller particle.
                # Near the boundary, correct searchsorted roundoff by comparing the difference directly.
                indices = np.searchsorted(ordered, ordered + radius, side="right")
                for i in range(N):
                    j = indices[i]
                    while j < N and ordered[j] - ordered[i] <= radius:
                        j += 1
                    while j > i + 1 and ordered[j - 1] - ordered[i] > radius:
                        j -= 1
                    indices[i] = j
                outside += weight * 2 * int(np.sum(N - indices))
    return dict(minimum=minimum, maximum=maximum, outside_basis_interval_fraction=outside / total)


def make_split(batch, destination, base_directory, split, index, simulation, spec, base):
    destination.mkdir()
    seed = base["seeds"][split][index]
    if base_directory is not None and all(simulation[k] == base["simulation"][k] for k in ("N", "dt", "sigma", "horizon")):
        source = base_directory / split / "data.npz"
        data = load_data(source)
    else:
        X, t, X0, noise, _ = reference_trajectory(seed, particles=simulation["N"], horizon=simulation["horizon"],
                                                 step=simulation["dt"], sigma=simulation["sigma"])
        data = dict(X=X, t=t, X0=X0, driving_noise=noise)
        source = destination / "data.npz"
        np.savez_compressed(source, **data)
    obs_seed = spec["observation_seed_offset"] + seed
    Y = data["X"]
    if simulation["sigma_obs"]:
        Y = Y + simulation["sigma_obs"] * np.random.default_rng(obs_seed).normal(size=Y.shape)
        np.save(destination / "Y.npy", Y)
    space, time = test_family(data["t"])
    started = perf_counter()
    designs = assemble_full(Y, data["t"], space, time, candidate_list(base))
    assembly_seconds = perf_counter() - started
    b = compute_b_timeweak(Y, data["t"], simulation["sigma"], space, time).reshape(-1) / np.sqrt(200)
    np.save(destination / "response.npy", b)
    for name, A in designs.items():
        np.save(destination / f"{name}.npy", A)
    manifest = dict(seed=seed, brownian_seed=seed + 1, observation_seed=obs_seed,
                    data_file=str(source.relative_to(batch)), data_file_sha256=digest(source),
                    data_array_sha256={key: array_hash(value) for key, value in data.items()},
                    Y_sha256=array_hash(Y), response_sha256=array_hash(b),
                    designs={name: dict(sha256=array_hash(A), **matrix_diagnostics(A)) for name, A in designs.items()},
                    distances=coverage(Y, data["t"]), assembly_seconds_all_dictionaries=assembly_seconds)
    write_json(destination / "manifest.json", manifest)
    return designs, b, data, manifest


def evaluate_frozen(directory, data, selections, simulation, statistics):
    results = {}
    for name, selected in selections.items():
        folder = directory / name
        folder.mkdir(exist_ok=True)
        if selected["status"] != "selected":
            results[name] = dict(status=selected["status"], selection=selected, metrics={})
            continue
        metrics, rollout = evaluate_model(data, selected["basis"], selected["theta"], simulation["sigma"], statistics[name])
        row = dict(status="evaluated", selection=selected, basis=selected["basis"], theta=selected["theta"], metrics=metrics)
        if rollout is not None:
            np.savez_compressed(folder / "rollout.npz", X_hat=rollout)
            row["rollout_sha256"] = array_hash(rollout)
        results[name] = row
    return results


def accuracy_job(batch, directory, job, spec, base, base_directory):
    index, simulation = job["index"], job["simulation"]
    splits = {}
    arrays = {}
    for split in ("train", "validation"):
        A, b, _, manifest = make_split(batch, directory / split, base_directory, split, index, simulation, spec, base)
        arrays[split] = A, b
        splits[split] = manifest
    selections = {}
    for name, params in candidate_list(base):
        selection, path = select_lasso(arrays["train"][0][name], arrays["train"][1],
                                      arrays["validation"][0][name], arrays["validation"][1], base["estimator"])
        folder = directory / name
        folder.mkdir()
        np.save(folder / "path_coefficients.npy", path)
        selections[name] = dict(basis=params, **selection)
    write_json(directory / "frozen_selections.json", selections)
    # Only after this point, load this case's evaluation trajectory and evaluation quadratic forms.
    Atest, btest, test, manifest = make_split(batch, directory / "test", base_directory, "test", index, simulation, spec, base)
    splits["test"] = manifest
    if base_directory is not None and all(simulation[k] == base["simulation"][k] for k in ("N", "dt", "sigma", "horizon")):
        statistics = load_statistics(base_directory / "evaluation_statistics.npz")
        statistics_source = str((base_directory / "evaluation_statistics.npz").relative_to(batch))
    else:
        statistics = evaluation_statistics(test["X"], test["t"], candidate_list(base), base["true_kernel"])
        save_statistics(directory / "evaluation_statistics.npz", statistics)
        statistics_source = str((directory / "evaluation_statistics.npz").relative_to(batch))
    results = evaluate_frozen(directory, test, selections, simulation, statistics)
    for name, row in results.items():
        if row["status"] == "evaluated":
            row["test_weak_residual_mse"] = float(np.mean((Atest[name] @ np.asarray(row["theta"]) - btest) ** 2))
    return dict(simulation=simulation, results=results, splits=splits, statistics_source=statistics_source,
                frozen_selections_sha256=digest(directory / "frozen_selections.json"))


def import_baseline(batch, directory, job, spec, base):
    archive = ROOT / spec["base_results"]
    old = archive / f"run-{job['index'] + 1:02d}"
    old_status = read_json(archive / "batch_status.json")
    old_frozen = read_json(archive / "frozen_protocol.json")
    if digest(archive / "frozen_protocol.json") != old_status["frozen_protocol_sha256"] or old_frozen["protocol"] != base:
        raise ValueError("baseline frozen protocol mismatch")
    record = next(row for row in old_status["runs"] if row["index"] == job["index"])
    if record["status"] != "completed" or digest(old / "fit/report.json") != record["report_sha256"]:
        raise ValueError("baseline report hash/status mismatch")
    report = read_json(old / "fit/report.json")
    original = read_json(old / "assembly/manifest.json")
    original_selections = read_json(old / "fit/frozen_selections.json")
    for key in ("simulation", "basis", "true_kernel", "estimator"):
        if report["protocol"][key] != base[key]:
            raise ValueError(f"baseline {key} mismatch")
    if digest(old / "assembly/manifest.json") != report["assembly_manifest_sha256"]:
        raise ValueError("baseline assembly manifest hash mismatch")
    if digest(old / "fit/frozen_selections.json") != report["frozen_selections_sha256"]:
        raise ValueError("baseline selections hash mismatch")
    # Verify historical artifacts against their contemporaneous source snapshots; do not relabel them with the current source.
    snapshot = ROOT / spec["base_results"] / "source_snapshot"
    for source_map in (old_frozen["source_sha256"], original["source_sha256"], report["source_sha256"]):
        for name, expected in source_map.items():
            if digest(snapshot / name) != expected:
                raise ValueError("baseline archived source hash mismatch")
    selections, results, splits = {}, {}, {}
    for name, params in candidate_list(base):
        saved = report["results"][name]
        if (saved["basis"] != params or saved["status"] != "evaluated"
                or saved["selection"] != original_selections[name]
                or saved["theta"] != original_selections[name]["theta"]):
            raise ValueError("invalid baseline model")
        selections[name] = saved["selection"]
        folder = directory / name
        folder.mkdir()
        shutil.copy2(old / "fit" / name / "path_coefficients.npy", folder / "path_coefficients.npy")
    write_json(directory / "frozen_selections.json", selections)
    for split in ("train", "validation", "test"):
        seed = base["seeds"][split][job["index"]]
        saved = original["splits"][split][0]
        if saved["seed"] != seed:
            raise ValueError("baseline seed mismatch")
        source, target = old / "assembly" / split / str(seed), directory / split
        target.mkdir()
        with np.load(source / "data.npz") as data:
            if array_hash(data["Y"]) != saved["observation_sha256"] or array_hash(data["t"]) != saved["time_sha256"]:
                raise ValueError("baseline observation/time hash mismatch")
            np.testing.assert_array_equal(data["X"], data["Y"])
            np.testing.assert_array_equal(data["X0"], data["X"][:, 0])
            np.testing.assert_array_equal(data["X0"], initial_positions(seed, particles=base["simulation"]["N"]))
            expected_noise = np.random.default_rng(seed + 1).normal(size=data["driving_noise"].shape)
            np.testing.assert_array_equal(data["driving_noise"], expected_noise)
        for filename in ["data.npz", "response.npy"] + [name + ".npy" for name, _ in candidate_list(base)]:
            shutil.copy2(source / filename, target / filename)
        if array_hash(np.load(target / "response.npy")) != saved["response_sha256"]:
            raise ValueError("baseline response hash mismatch")
        for name, _ in candidate_list(base):
            if array_hash(np.load(target / f"{name}.npy")) != saved["designs"][name]["sha256"]:
                raise ValueError("baseline design hash mismatch")
        splits[split] = saved
    test = load_data(directory / "test/data.npz")
    stats = evaluation_statistics(test["X"], test["t"], candidate_list(base), base["true_kernel"])
    save_statistics(directory / "evaluation_statistics.npz", stats)
    parity = {}
    for name, params in candidate_list(base):
        saved = report["results"][name]
        metrics, rollout = evaluate_model(test, params, saved["theta"], base["simulation"]["sigma"], stats[name])
        old_metrics = saved["test_metrics"][0]
        differences = {key: abs(metrics[key] - old_metrics[key]) for key in METRICS}
        if max(differences.values()) > 2e-7:
            raise ValueError("baseline evaluation parity failed")
        with np.load(old / "fit" / name / "rollout_0.npz") as data:
            original_rollout = data["X_hat"]
        np.testing.assert_allclose(rollout, original_rollout, atol=2e-7, rtol=1e-7)
        shutil.copy2(old / "fit" / name / "rollout_0.npz", directory / name / "rollout.npz")
        parity[name] = dict(metric_differences=differences,
                            rollout_max_absolute_difference=float(np.max(np.abs(rollout - original_rollout))),
                            rollout_backend=metrics["rollout_backend"],
                            rollout_stiffness_indicator=metrics["rollout_stiffness_indicator"])
        results[name] = dict(status="evaluated", selection=selections[name], theta=saved["theta"], basis=params,
                             metrics=old_metrics, test_weak_residual_mse=saved["test_weak_residual_mse"],
                             rollout_sha256=array_hash(original_rollout))
    shutil.copy2(old / "fit/report.json", directory / "imported_report.json")
    shutil.copy2(old / "assembly/manifest.json", directory / "imported_manifest.json")
    shutil.copy2(old / "fit/frozen_selections.json", directory / "imported_frozen_selections.json")
    return dict(simulation=base["simulation"], results=results, splits=splits, parity=parity,
                baseline_source=str(old), baseline_report_sha256=digest(old / "fit/report.json"),
                frozen_selections_sha256=digest(directory / "frozen_selections.json"))


def fixed_penalty_path(A, b, reference_selection, settings):
    stop = reference_selection["selected"]["index"]
    model = Lasso(fit_intercept=False, warm_start=True, max_iter=settings["max_iter"],
                  tol=settings["tol"], selection="cyclic")
    rows, coefficients = [], []
    for reference in reference_selection["path"][:stop + 1]:
        model.alpha = reference["alpha_sk"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(A, b)
        finite = bool(np.all(np.isfinite(model.coef_)) and np.isfinite(model.dual_gap_))
        warning_count = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        rows.append(dict(index=reference["index"], alpha_sk=model.alpha, iterations=int(model.n_iter_),
                         dual_gap=float(model.dual_gap_) if finite else None,
                         finite=finite, converged=finite and warning_count == 0, convergence_warnings=warning_count))
        coefficients.append(model.coef_.copy())
    return dict(path=rows, endpoint=rows[-1], theta=coefficients[-1].tolist() if rows[-1]["finite"] else None), np.asarray(coefficients)


def pair_job(batch, directory, job, spec, base, base_directory):
    # Do not reselect regularization for each P; use the full-pair path already frozen by independent validation.
    reference = read_json(base_directory / "frozen_selections.json")
    train = load_data(base_directory / "train/data.npz")
    b = np.load(base_directory / "train/response.npy")
    space, time = test_family(train["t"])
    pair_seed = spec["pair_seed_base"] + 1000 * job["index"] + 10 * job["pair_index"]
    matrices = assemble_sampled(train["X"], train["t"], space, time, candidate_list(base), job["P"], pair_seed)
    endpoints = {}
    for name, params in candidate_list(base):
        folder = directory / name
        folder.mkdir()
        np.save(folder / "design.npy", matrices[name])
        endpoint, coefficients = fixed_penalty_path(matrices[name], b, reference[name], base["estimator"])
        np.save(folder / "path_coefficients.npy", coefficients)
        endpoints[name] = dict(basis=params, **endpoint)
    write_json(directory / "frozen_endpoints.json", endpoints)
    test = load_data(base_directory / "test/data.npz")
    stats = load_statistics(base_directory / "evaluation_statistics.npz")
    results = {}
    for name, params in candidate_list(base):
        endpoint = endpoints[name]
        row = dict(endpoint=endpoint["endpoint"], path=endpoint["path"], basis=params, theta=endpoint["theta"], metrics={})
        if endpoint["theta"] is not None:
            theta = np.asarray(endpoint["theta"])
            full_theta = np.asarray(reference[name]["theta"])
            difference = theta - full_theta
            full_A = np.load(base_directory / "train" / f"{name}.npy")
            metrics, rollout = evaluate_model(test, params, theta, base["simulation"]["sigma"], stats[name])
            denominator = force_energy(full_theta, stats[name])
            metrics.update(coefficient_difference_sq=float(difference @ difference),
                           design_difference_sq=float(np.sum((matrices[name] - full_A) ** 2)),
                           force_difference_sq_rel=force_energy(difference, stats[name]) / denominator if denominator > 0 else None)
            row.update(status="finite_solver_output", metrics=metrics)
            if rollout is not None:
                np.savez_compressed(directory / name / "rollout.npz", X_hat=rollout)
                row["rollout_sha256"] = array_hash(rollout)
        else:
            row["status"] = "nonfinite_solver_output"
        results[name] = row
    return dict(P=job["P"], pair_seed=pair_seed, results=results,
                reference_directory=str(base_directory.relative_to(batch)),
                reference_selections_sha256=digest(base_directory / "frozen_selections.json"),
                frozen_endpoints_sha256=digest(directory / "frozen_endpoints.json"),
                estimand="finite-iteration fixed-alpha solver output; nonconverged endpoints are flagged, not discarded")


def runtime_job(batch, directory, job, spec, base):
    N, repeat = job["N"], job["index"]
    seed = spec["runtime_seed_base"] + 1000 * job["particle_index"] + 10 * repeat
    X, t, _, _, _ = reference_trajectory(seed, particles=N, horizon=base["simulation"]["horizon"],
                                         step=base["simulation"]["dt"], sigma=base["simulation"]["sigma"])
    space, time = test_family(t)
    schedule = [(candidate, mode) for candidate in candidate_list(base) for mode in ("full", "sampled")]
    rng = np.random.default_rng(spec["runtime_order_seed"] + 1000 * job["particle_index"] + repeat)
    rng.shuffle(schedule)
    timings = []
    for candidate, mode in schedule:
        # Use identical timer boundaries for both methods, excluding data generation, file I/O, response construction, and regression.
        started = perf_counter()
        if mode == "full":
            arrays = assemble_full(X, t, space, time, [candidate])
        else:
            arrays = assemble_sampled(X, t, space, time, [candidate], spec["runtime_pair_ratio"] * N, seed + 2)
        elapsed = perf_counter() - started
        A = arrays[candidate[0]]
        if not np.all(np.isfinite(A)):
            raise ValueError("nonfinite runtime matrix")
        timings.append(dict(name=candidate[0], mode=mode, seconds=elapsed, matrix_sha256=array_hash(A)))
    return dict(N=N, repeat=repeat, seed=seed, X_sha256=array_hash(X), timings=timings,
                numerical_threads=1, competing_sweep_workers=0)


def worker(batch, job, spec, base, smoke, base_directory=None):
    batch = Path(batch)
    parent = batch / job["relative_directory"]
    directory = parent
    attempt = 1
    while directory.exists():
        attempt += 1
        directory = parent.with_name(parent.name + f"-attempt-{attempt}")
    directory.mkdir(parents=True)
    started = perf_counter()
    result = dict(job=job, directory=str(directory.relative_to(batch)), status="running", smoke_only=smoke, attempt=attempt)
    write_json(directory / "job_status.json", result)
    try:
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
            if job["stage"] == "base" and not smoke:
                content = import_baseline(batch, directory, job, spec, base)
            elif job["stage"] in ("base", "accuracy"):
                content = accuracy_job(batch, directory, job, spec, base, Path(base_directory) if base_directory else None)
            elif job["stage"] == "pairs":
                content = pair_job(batch, directory, job, spec, base, Path(base_directory))
            else:
                content = runtime_job(batch, directory, job, spec, base)
        result.update(content)
        result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    result["elapsed_seconds"] = perf_counter() - started
    result["artifacts_sha256"] = {str(path.relative_to(directory)): digest(path)
                                   for path in directory.rglob("*") if path.is_file() and path.name != "job_status.json"}
    write_json(directory / "report.json", result)
    write_json(directory / "job_status.json", {key: result[key] for key in ("status", "directory", "elapsed_seconds", "attempt")})
    return dict(status=result["status"], directory=result["directory"], report_sha256=digest(directory / "report.json"),
                elapsed_seconds=result["elapsed_seconds"], error=result.get("error"))


def job_plan(spec, base):
    count = base["replication"]["count"]
    jobs = {}
    def add(key, **values):
        jobs[key] = dict(id=key, **values)
    for index in range(count):
        add(f"base-{index}", stage="base", index=index, simulation=base["simulation"], relative_directory=f"base/run-{index + 1:02d}")
    for case in cases(spec, base):
        if case["id"] == "base":
            continue
        for index in range(count):
            add(f"{case['id']}-{index}", stage="accuracy", index=index, case=case["id"], simulation=case["simulation"],
                relative_directory=f"accuracy/{case['id']}/run-{index + 1:02d}")
    for pi, P in enumerate(spec["pair_counts"]):
        for index in range(count):
            add(f"pairs-{pi}-{index}", stage="pairs", index=index, P=P, pair_index=pi,
                relative_directory=f"pairs/P-{P}/run-{index + 1:02d}")
    for ni, N in enumerate(spec["runtime_particles"]):
        for index in range(spec["runtime_repetitions"]):
            add(f"runtime-{ni}-{index}", stage="runtime", index=index, N=N, particle_index=ni,
                relative_directory=f"runtime/N-{N}/run-{index + 1:02d}")
    return jobs


def run_batch(spec, base, destination, workers=6, smoke=False, resume=False):
    validate_protocol(spec, base)
    destination = Path(destination).resolve()
    if resume:
        frozen = read_json(destination / "frozen_protocol.json")
        if frozen["requested_protocol"] != spec or frozen["requested_base"] != base or frozen["smoke_only"] != smoke:
            raise ValueError("resume protocol mismatch")
        for name, expected in frozen["source_sha256"].items():
            if digest(ROOT / name) != expected or digest(destination / "source_snapshot" / name) != expected:
                raise ValueError("resume source mismatch; do not overwrite the old experiment")
        status = read_json(destination / "batch_status.json")
        if status["frozen_protocol_sha256"] != digest(destination / "frozen_protocol.json"):
            raise ValueError("frozen protocol hash mismatch")
        for name, expected in frozen.get("baseline_provenance_sha256", {}).items():
            if digest(destination / "baseline_provenance" / name) != expected:
                raise ValueError("baseline provenance changed")
        for result in status["results"].values():
            if digest(destination / result["directory"] / "report.json") != result["report_sha256"]:
                raise ValueError("saved job report hash mismatch")
    else:
        if not smoke:
            validation = read_json(ROOT / spec["backend_validation"])
            if validation["status"] != "passed":
                raise ValueError("backend parity validation is required")
            for name, expected in validation["source_sha256"].items():
                if digest(ROOT / name) != expected:
                    raise ValueError("backend changed after parity validation")
        destination.mkdir(parents=True, exist_ok=False)
        hashes = {name: digest(ROOT / name) for name in SOURCES}
        if not smoke:
            archive = ROOT / spec["base_results"]
            provenance = destination / "baseline_provenance"
            provenance.mkdir()
            for filename in ("frozen_protocol.json", "batch_status.json", "verification.json"):
                shutil.copy2(archive / filename, provenance / filename)
            for filename in ("parity.json", "verification_script.py"):
                shutil.copy2((ROOT / spec["backend_validation"]).parent / filename, provenance / ("backend_" + filename))
            shutil.copytree(archive / "source_snapshot", provenance / "source_snapshot")
            if spec.get("supersedes_attempt"):
                previous = ROOT / spec["supersedes_attempt"]
                previous_status = read_json(previous / "batch_status.json")
                previous_protocol = read_json(previous / "frozen_protocol.json")
                if (previous_status["status"] != "blocked_baseline_import" or
                        any(previous_status["jobs"][key]["stage"] != "base" for key in previous_status["results"])):
                    raise ValueError("superseded attempt ran beyond baseline verification")
                if previous_protocol["requested_base"] != base:
                    raise ValueError("base settings changed after failed arithmetic check")
                for key in ("sweeps", "pair_counts", "runtime_particles", "runtime_repetitions", "runtime_pair_ratio",
                            "observation_seed_offset", "pair_seed_base", "pair_penalty_policy", "pair_nonconvergence_policy"):
                    if previous_protocol["requested_protocol"][key] != spec[key]:
                        raise ValueError("scientific settings changed after failed arithmetic check")
                retired = provenance / "superseded_attempt"
                retired.mkdir()
                for filename in ("frozen_protocol.json", "batch_status.json"):
                    shutil.copy2(previous / filename, retired / filename)
                shutil.copytree(previous / "source_snapshot", retired / "source_snapshot")
                for key, record in previous_status["results"].items():
                    if record["status"] == "failed":
                        shutil.copy2(previous / record["directory"] / "report.json", retired / f"{key}-failure.json")
        for name in SOURCES:
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        effective_spec, effective_base = effective_protocol(spec, base, smoke)
        baseline_hashes = {} if smoke else {str(p.relative_to(provenance)): digest(p)
                                             for p in provenance.rglob("*") if p.is_file()}
        frozen = dict(requested_protocol=spec, requested_base=base, protocol=effective_spec, base=effective_base,
                      smoke_only=smoke, source_sha256=hashes, baseline_provenance_sha256=baseline_hashes,
                      workers=workers, python_version=sys.version,
                      numpy_version=np.__version__, sklearn_version=sklearn.__version__)
        write_json(destination / "frozen_protocol.json", frozen)
        status = dict(status="running", smoke_only=smoke, frozen_protocol_sha256=digest(destination / "frozen_protocol.json"),
                      jobs=job_plan(effective_spec, effective_base), results={})
        write_json(destination / "batch_status.json", status)
    spec, base = frozen["protocol"], frozen["base"]
    started = perf_counter()
    for stage in ("base", "accuracy", "pairs", "runtime"):
        pending = {key: job for key, job in status["jobs"].items() if job["stage"] == stage and key not in status["results"]}
        status.update(status="running", stage=stage)
        write_json(destination / "batch_status.json", status)
        stage_workers = 1 if stage == "runtime" else workers
        # Run runtime jobs sequentially in one separate process after all other production-batch computations finish.
        with ProcessPoolExecutor(max_workers=stage_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {}
            for key, job in pending.items():
                parent = None if stage in ("base", "runtime") else destination / status["results"][f"base-{job['index']}"]["directory"]
                futures[pool.submit(worker, destination, job, spec, base, smoke, parent)] = key
            for future in as_completed(futures):
                key = futures[future]
                # Stop on process failures that the worker cannot catch; resuming still does not overwrite existing output.
                result = future.result()
                status["results"][key] = result
                write_json(destination / "batch_status.json", status)
                print(f"{len(status['results'])}/{len(status['jobs'])}: {key} {result['status']} {result['elapsed_seconds']:.1f}s", flush=True)
        if stage == "base" and any(status["results"][key]["status"] != "completed"
                                    for key, job in status["jobs"].items() if job["stage"] == "base"):
            status["status"] = "blocked_baseline_import"
            write_json(destination / "batch_status.json", status)
            raise RuntimeError("baseline integrity/parity failed; dependent work was not launched")
    status["status"] = "completed" if all(r["status"] == "completed" for r in status["results"].values()) else "completed_with_failures"
    status["elapsed_seconds_this_invocation"] = perf_counter() - started
    write_json(destination / "batch_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or (args.resume and args.out_dir is None):
        parser.error("workers must be 1..8; resume needs an output directory")
    spec = read_json(args.config)
    base = read_json(ROOT / spec["base_config"])
    validate_protocol(spec, base)
    if not args.run and not args.smoke:
        jobs = job_plan(spec, base)
        print(json.dumps(dict(status="configuration_valid", jobs=len(jobs),
                              stages={stage: sum(j["stage"] == stage for j in jobs.values())
                                      for stage in ("base", "accuracy", "pairs", "runtime")}), indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / (("sweeps-smoke-" if args.smoke else "sweeps-R20-") + stamp)
    run_batch(spec, base, destination, args.workers, args.smoke, args.resume)
    from scripts.summarize_1d_hat_sweeps import summarize
    summary_directory = destination / "summary"
    if summary_directory.exists():
        summary_directory = destination / f"summary-{stamp}"
    summarize(destination, summary_directory)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
