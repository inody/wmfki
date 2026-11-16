"""Compute a validation-selected strong FD-LASSO curve for Figure 1(c).

The run reuses the frozen Generic 1D trajectory triplets and held-out quadratic
statistics.  Strong-form features use every target, every observation interval,
and the exact all-particle source average.  A sorted-prefix implementation
computes the one-dimensional hat features without materializing all pairs.
Existing outputs and manuscript files are not modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hat_sweep_numerics import errors_from_statistics
from scripts.run_1d_hat_lasso import select_lasso
from scripts.run_1d_hat_sampled_p50n_R100 import percentile_mean_ci
from scripts.run_1d_hat_sampled_pair_ratio_sweep_R5 import candidates
from scripts.run_1d_hat_sweeps import load_data, load_statistics

DEFAULT_CONFIG = ROOT / "configs/experiments/fig1c_strong_fd_lasso.json"
OUTPUT_ROOT = ROOT / "outputs/validation/fig1c_strong_fd_lasso"
SOURCE_FILES = (
    "configs/experiments/fig1c_strong_fd_lasso.json",
    "scripts/run_fig1c_strong_fd_lasso.py",
    "scripts/generate_1d_hat_practical_figure.py",
    "scripts/run_1d_hat_lasso.py",
    "scripts/hat_sweep_numerics.py",
    "kernels.py",
)


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


def validate_protocol(spec):
    if (spec.get("task"), spec.get("protocol_version")) != ("fig1c_strong_fd_lasso", 1):
        raise ValueError("invalid Figure 1(c) strong-LASSO protocol")
    if spec.get("study_status") != "candidate_non_manuscript":
        raise ValueError("Figure 1(c) strong-LASSO run must remain a candidate")
    if spec.get("dictionary_sizes") != [9, 11, 14, 17, 22, 27, 33, 42, 52, 65]:
        raise ValueError("dictionary-size grid changed")
    if int(spec.get("repetitions", 0)) != 100:
        raise ValueError("full protocol requires 100 repetitions")
    estimator = spec["estimator"]
    if estimator["fit_intercept"] or estimator["standardize"]:
        raise ValueError("strong FD-LASSO must use raw columns and no intercept")
    source = ROOT / spec["source_batch"]
    for relative, expected in spec["source_hashes"].items():
        if digest(source / relative) != expected:
            raise ValueError(f"frozen Figure 1 source changed: {relative}")
    expected_features = {
        "source_average": "exact_all_ordered_off_diagonal_pairs",
        "target_particles": "all",
        "time_intervals": "all",
        "implementation": "exact_1d_sorted_prefix_moments_for_piecewise_linear_hat_basis",
    }
    if spec["strong_features"] != expected_features:
        raise ValueError("strong feature protocol changed")


def _range_moments(xs, prefix1, prefix2, lower, upper, lower_side, upper_side):
    lo = np.searchsorted(xs, lower, side=lower_side)
    hi = np.searchsorted(xs, upper, side=upper_side)
    count = hi - lo
    total1 = prefix1[hi] - prefix1[lo]
    total2 = prefix2[hi] - prefix2[lo]
    return count.astype(float), total1, total2


def exact_hat_strong_features_1d(x, knots):
    """Exact N^{-1} sum_j Phi_m(x_i-x_j) in O(NM) time."""
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if x.ndim != 1 or knots.ndim != 1 or knots[0] != 0.0:
        raise ValueError("invalid 1D positions or hat knots")
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    prefix1 = np.concatenate(([0.0], np.cumsum(xs)))
    prefix2 = np.concatenate(([0.0], np.cumsum(xs * xs)))
    feature = np.zeros((xs.size, knots.size), dtype=float)
    xi = xs
    xi2 = xi * xi
    for k, (a, b) in enumerate(zip(knots[:-1], knots[1:])):
        # Distances are assigned to [a,b), matching the continuous nodal basis.
        cl, sl, sql = _range_moments(
            xs, prefix1, prefix2, xi - b, xi - a, "right", "right"
        )
        cr, sr, sqr = _range_moments(
            xs, prefix1, prefix2, xi + a, xi + b, "left", "left"
        )
        first = (cl + cr) * xi - sl - sr
        second = (cl * xi2 - 2.0 * xi * sl + sql) - (
            sqr - 2.0 * xi * sr + cr * xi2
        )
        width = b - a
        feature[:, k] += (b * first - second) / width
        feature[:, k + 1] += (second - a * first) / width

    radius = knots[-1]
    cl, sl, _ = _range_moments(
        xs, prefix1, prefix2, np.full_like(xi, -np.inf), xi - radius,
        "left", "right",
    )
    cr, sr, _ = _range_moments(
        xs, prefix1, prefix2, xi + radius, np.full_like(xi, np.inf),
        "left", "right",
    )
    feature[:, -1] += (cl + cr) * xi - sl - sr
    feature /= xs.size
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return feature[inverse]


def sufficient_statistics(X, time, knots):
    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    if X.ndim != 4 or X.shape[0] != 1 or X.shape[-1] != 1:
        raise ValueError("expected one 1D trajectory")
    columns = len(knots)
    gram = np.zeros((columns, columns), dtype=float)
    score = np.zeros(columns, dtype=float)
    response_square = 0.0
    rows = 0
    for n, dt in enumerate(np.diff(time)):
        A = exact_hat_strong_features_1d(X[0, n, :, 0], knots)
        b = (X[0, n + 1, :, 0] - X[0, n, :, 0]) / dt
        gram += A.T @ A
        score += A.T @ b
        response_square += float(b @ b)
        rows += b.size
    return {
        "gram": gram / rows,
        "score": score / rows,
        "response_square": response_square / rows,
        "rows": rows,
    }


def compressed_system(statistics):
    gram = 0.5 * (statistics["gram"] + statistics["gram"].T)
    score = statistics["score"]
    columns = gram.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = max(float(np.max(eigenvalues)), 1.0) * np.finfo(float).eps * columns * 10.0
    eigenvalues = np.where(eigenvalues > tolerance, eigenvalues, 0.0)
    A = np.sqrt(columns) * (np.sqrt(eigenvalues)[:, None] * eigenvectors.T)
    b, *_ = np.linalg.lstsq(A.T, columns * score, rcond=None)
    return A, b


def mse_from_statistics(theta, statistics):
    theta = np.asarray(theta, dtype=float)
    return float(
        theta @ statistics["gram"] @ theta
        - 2.0 * theta @ statistics["score"]
        + statistics["response_square"]
    )


def fit_from_statistics(training, validation, estimator):
    A, b = compressed_system(training)
    Av, bv = compressed_system(validation)
    selection, path = select_lasso(A, b, Av, bv, estimator)
    if selection["status"] != "selected":
        raise RuntimeError(f"strong FD-LASSO selection failed: {selection['status']}")
    theta = np.asarray(selection["theta"], dtype=float)
    selected = selection["selected"]
    return theta, {
        "selected_alpha": float(selected["alpha_sk"]),
        "selected_alpha_over_max": float(selected["alpha_over_max"]),
        "selected_index": int(selected["index"]),
        "selected_at_grid_boundary": bool(selection["selected_at_grid_boundary"]),
        "selected_converged": bool(selected["converged"]),
        "selected_iterations": int(selected["iterations"]),
        "selected_dual_gap": float(selected["dual_gap"]),
        "selected_nonzero": int(selected["nonzero_reported"]),
        "training_mse": mse_from_statistics(theta, training),
        "validation_mse": mse_from_statistics(theta, validation),
        "unconverged_candidates": int(sum(not row["converged"] for row in selection["path"])),
        "candidate_count": len(path),
        "training_rows": int(training["rows"]),
        "validation_rows": int(validation["rows"]),
    }


def run_repetition(payload):
    repetition, spec, frozen = payload
    started = perf_counter()
    bundle = frozen["reference_bundles"][f"run-{repetition}"]
    reference = Path(bundle["reference_directory"])
    train_path = reference / "train/data.npz"
    validation_path = reference / "validation/data.npz"
    for path in (train_path, validation_path, Path(bundle["old_statistics"])):
        expected = bundle["critical_artifacts_sha256"].get(str(path))
        if expected is None:
            expected = bundle.get("old_statistics_sha256") if path == Path(bundle["old_statistics"]) else None
        if expected is not None and digest(path) != expected:
            raise ValueError(f"frozen repetition artifact changed: {path}")
    train = load_data(train_path)
    validation = load_data(validation_path)
    evaluation_statistics = load_statistics(bundle["old_statistics"])
    parameters = dict(candidates(frozen["effective_base"]))
    records = []
    with threadpool_limits(limits=1):
        for size in spec["dictionary_sizes"]:
            name = f"hat_M{size}"
            params = parameters[name]
            training = sufficient_statistics(train["X"], train["t"], params["knots"])
            validation_stats = sufficient_statistics(
                validation["X"], validation["t"], params["knots"]
            )
            theta, selection = fit_from_statistics(
                training, validation_stats, spec["estimator"]
            )
            metrics = errors_from_statistics(theta, evaluation_statistics[name])
            records.append({
                "repetition": repetition,
                "M": int(size),
                "method": "strong_forward_difference_lasso_validation_selected",
                "coefficients": theta.tolist(),
                "drift_error_rel": float(metrics["drift_error_rel"]),
                "force_error_weighted_rel": float(metrics["force_error_weighted_rel"]),
                "selection": selection,
                "status": "finite" if np.all(np.isfinite(theta)) else "failed",
            })
    return {
        "repetition": repetition,
        "records": records,
        "elapsed_seconds": perf_counter() - started,
    }


def summarize(records, spec):
    groups = []
    for size_index, size in enumerate(spec["dictionary_sizes"]):
        rows = sorted(
            [row for row in records if row["M"] == size],
            key=lambda row: row["repetition"],
        )
        values = np.asarray([row["drift_error_rel"] for row in rows], dtype=float)
        rng = np.random.default_rng(int(spec["bootstrap"]["seed"]) + size_index)
        indices = rng.integers(0, len(values), size=(int(spec["bootstrap"]["draws"]), len(values)))
        groups.append({
            "M": int(size),
            "drift_error_rel": {
                "values": values.tolist(),
                "mean": float(np.mean(values)),
                "rmse": float(np.sqrt(np.mean(values ** 2))),
                "median": float(np.median(values)),
                "mean_ci95": percentile_mean_ci(values, indices),
            },
            "force_error_weighted_rel": {
                "mean": float(np.mean([row["force_error_weighted_rel"] for row in rows]))
            },
            "failures": int(sum(row["status"] != "finite" for row in rows)),
            "boundary_selections": int(sum(
                row["selection"]["selected_at_grid_boundary"] for row in rows
            )),
            "unconverged_candidates": int(sum(
                row["selection"]["unconverged_candidates"] for row in rows
            )),
        })
    return {"groups": groups, "record_count": len(records)}


def snapshot_sources(out):
    hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(source)
    return hashes


def run(config_path, out, *, workers=1, smoke=False):
    spec = read_json(config_path)
    validate_protocol(spec)
    source = ROOT / spec["source_batch"]
    frozen = read_json(source / "frozen_protocol.json")
    effective = json.loads(json.dumps(spec))
    if smoke:
        effective["repetitions"] = 1
        effective["dictionary_sizes"] = [9, 65]
        effective["estimator"]["alpha_count"] = 8
        effective["bootstrap"]["draws"] = 100
    out = Path(out).resolve()
    allowed = OUTPUT_ROOT.resolve()
    if out != allowed and allowed not in out.parents:
        raise ValueError(f"output must be beneath {allowed}")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, out / "requested_protocol.json")
    write_json(out / "effective_protocol.json", effective)
    source_hashes = snapshot_sources(out)
    started = perf_counter()
    jobs = [(index, effective, frozen) for index in range(effective["repetitions"])]
    results = []
    if workers == 1:
        for job in jobs:
            result = run_repetition(job)
            results.append(result)
            print(f"completed repetition {result['repetition']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_repetition, job): job[0] for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"completed repetition {result['repetition']}", flush=True)
    records = [row for result in results for row in result["records"]]
    records.sort(key=lambda row: (row["M"], row["repetition"]))
    summary = summarize(records, effective)
    write_json(out / "records.json", records)
    write_json(out / "summary.json", summary)
    diagnostics = {
        "failures": sum(row["status"] != "finite" for row in records),
        "boundary_selections": sum(
            row["selection"]["selected_at_grid_boundary"] for row in records
        ),
        "unconverged_candidates": sum(
            row["selection"]["unconverged_candidates"] for row in records
        ),
    }
    write_json(out / "diagnostics.json", diagnostics)
    artifacts = {
        name: digest(out / name) for name in (
            "requested_protocol.json", "effective_protocol.json", "records.json",
            "summary.json", "diagnostics.json",
        )
    }
    manifest = {
        "task": spec["task"],
        "status": "smoke_complete" if smoke else "scientific_complete",
        "smoke": bool(smoke),
        "record_count": len(records),
        "source_sha256": source_hashes,
        "artifact_sha256": artifacts,
        "elapsed_seconds": perf_counter() - started,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = "smoke" if args.smoke else "scientific"
        args.out = OUTPUT_ROOT / f"{label}-{stamp}"
    manifest = run(args.config.resolve(), args.out, workers=args.workers, smoke=args.smoke)
    print(json.dumps({"output": str(args.out), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
