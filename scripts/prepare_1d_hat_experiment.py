"""Prepare the hat-basis experiment for the existing one-dimensional example without training or regularization selection."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import compute_A_full_timeweak, compute_b_timeweak, make_trapz_weights
from kernels import make_kernel_basis
from scripts.run_sindy_comparison import (
    DETERMINISTIC_SPACE_CENTERS,
    HORIZON,
    KERNEL_PARAMS,
    OBSERVATION_STEP,
    PARTICLES,
    PROCESS_NOISE,
    SPACE_TEST_WIDTH,
    TIME_TEST_COUNT,
    reference_trajectory,
)
from tests import make_test_functions
from time_tests import make_time_test_functions

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat.json"


def validate_spec(spec):
    """Fix the first-stage conditions and prevent implicit model changes or duplicate seeds."""
    if spec["protocol_version"] != 1:
        raise ValueError("unsupported protocol version")
    particles = spec["simulation"]["N"]
    if type(particles) is not int or particles not in (PARTICLES, 1000):
        raise ValueError("the approved particle comparison uses N=100 or N=1000")
    expected = {
        "N": particles, "horizon": HORIZON, "dt": OBSERVATION_STEP,
        "sigma": PROCESS_NOISE, "sigma_obs": 0.0,
    }
    if spec["simulation"] != expected or spec["true_kernel"] != KERNEL_PARAMS:
        raise ValueError("the first-stage config must preserve the existing 1D benchmark")
    if spec["pair_mode"] != "full":
        raise ValueError("the first stage uses exact full pairs")
    chunk = spec["pair_chunk_size"]
    if type(chunk) is not int or chunk < 1:
        raise ValueError("pair_chunk_size must be a positive integer")
    basis = spec["basis"]
    if basis["type"] != "hat" or basis["knot_spacing"] != "uniform":
        raise ValueError("the first stage uses uniform radial hat functions")
    if basis["extrapolation"] != "constant":
        raise ValueError("endpoint extension must be explicit and continuous")
    if not np.isfinite(basis["radius_max"]) or basis["radius_max"] <= 0:
        raise ValueError("radius_max must be positive and finite")
    sizes = basis["sizes"]
    if not sizes or any(type(m) is not int or m < 2 for m in sizes):
        raise ValueError("basis sizes must be integers >= 2")
    if sizes != sorted(set(sizes)):
        raise ValueError("basis sizes must be strictly increasing")
    if set(spec["seeds"]) != {"train", "validation", "test"}:
        raise ValueError("train, validation, and test splits are required")
    used = set()
    for seeds in spec["seeds"].values():
        if not seeds:
            raise ValueError("each split needs a seed")
        for seed in seeds:
            if type(seed) is not int or seed < 0 or used.intersection({seed, seed + 1}):
                raise ValueError("initial-state and Brownian seeds must be disjoint")
            used.update({seed, seed + 1})
    expected_baseline = {
        "basis": "qanr_exp", "M": 6,
        "ell_grid": [0.25, 0.5, 1.0, 2.0], "eps": KERNEL_PARAMS["eps"],
    }
    if spec["baseline"] != expected_baseline:
        raise ValueError("preserve the existing six-atom dictionary for comparison")
    if spec["estimator"] is not None:
        settings = spec["estimator"]
        if (settings["method"] != "lasso" or settings["standardize"] is not False
                or settings["fit_intercept"] is not False or settings["selection"] != "cyclic"):
            raise ValueError("the approved pilot uses raw cyclic LASSO without an intercept")
        if (type(settings["alpha_count"]) is not int or settings["alpha_count"] < 2
                or type(settings["max_iter"]) is not int or settings["max_iter"] < 1):
            raise ValueError("alpha_count and max_iter must be positive integers")
        if not 0 < settings["alpha_min_ratio"] < 1:
            raise ValueError("alpha_min_ratio must lie in (0, 1)")
        for key in ("tol", "report_nonzero_threshold"):
            if not np.isfinite(settings[key]) or settings[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")


def basis_params(spec, size):
    return {
        "basis": "hat", "M": size,
        "knots": np.linspace(0.0, spec["basis"]["radius_max"], size).tolist(),
        "extrapolation": spec["basis"]["extrapolation"],
    }


def test_family(t):
    space = make_test_functions({
        "type": "gaussian", "centers": DETERMINISTIC_SPACE_CENTERS,
        "s": SPACE_TEST_WIDTH,
    })
    time = make_time_test_functions({"time_tests": {
        "type": "bump", "K": TIME_TEST_COUNT,
        "support": 1.8 * float(t[-1] - t[0]) / (TIME_TEST_COUNT + 1),
    }}, t)
    return space, time


def assemble_design(Y, t, space, time, params, chunk_size):
    basis, _ = make_kernel_basis(params)
    A = compute_A_full_timeweak(
        Y, t, space, basis, time, weights=make_trapz_weights(t),
        pair_chunk_size=chunk_size,
    )
    scale = 1.0 / np.sqrt(A.shape[1] * A.shape[2])
    return (A * scale).reshape(-1, params["M"])


def matrix_diagnostics(A):
    """Use the normalized matrix only for diagnostics; do not alter the saved regression matrix."""
    if A.ndim != 2 or not np.all(np.isfinite(A)):
        raise ValueError("design must be a finite matrix")
    norms = np.linalg.norm(A, axis=0)
    normalized = np.divide(A, norms, out=np.zeros_like(A), where=norms > 0)

    def spectrum(B):
        singular = np.linalg.svd(B, compute_uv=False)
        tolerance = np.finfo(B.dtype).eps * max(B.shape) * singular[0]
        rank = int(np.sum(singular > tolerance))
        full_rank = rank == B.shape[1]
        return {
            "singular_values": singular.tolist(), "rank": rank,
            "rank_tolerance": float(tolerance), "full_column_rank": full_rank,
            "condition_number": float(singular[0] / singular[-1]) if full_rank else None,
        }

    return {
        "shape": list(A.shape), "column_norms": norms.tolist(),
        "zero_columns": np.flatnonzero(norms == 0).tolist(),
        "raw": spectrum(A), "column_normalized_diagnostic_only": spectrum(normalized),
    }


def distance_diagnostics(Y, t, radius_max, sizes):
    """Distance distribution based on time integrals over off-diagonal pairs, without inspecting the training truth."""
    edges = np.linspace(0.0, radius_max, 81)
    histogram = np.zeros(edges.size - 1)
    tail = 0.0
    first_cell = {str(m): 0.0 for m in sizes}
    minimum, maximum = np.inf, 0.0
    weights = make_trapz_weights(t)
    particles = Y.shape[2]
    mask = ~np.eye(particles, dtype=bool)
    for path in Y:
        for x, weight in zip(path, weights):
            radii = np.linalg.norm(x[:, None] - x[None, :], axis=-1)[mask]
            histogram += weight * np.histogram(radii, bins=edges)[0]
            tail += weight * np.count_nonzero(radii > radius_max)
            for m in sizes:
                first_cell[str(m)] += weight * np.count_nonzero(radii < radius_max / (m - 1))
            minimum = min(minimum, float(radii.min()))
            maximum = max(maximum, float(radii.max()))
    total = float(Y.shape[0] * particles * (particles - 1) * weights.sum())
    return {
        "edges": edges.tolist(), "bin_probability": (histogram / total).tolist(),
        "outside_basis_interval_fraction": tail / total,
        "first_hat_cell_fraction": {m: count / total for m, count in first_cell.items()},
        "minimum": minimum, "maximum": maximum,
        "measure": "trapezoidal-time-weighted ordered off-diagonal pair distances",
    }


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare(spec, out_dir, *, smoke=False, backend="reference"):
    validate_spec(spec)
    if backend not in ("reference", "cached_1d"):
        raise ValueError("unknown full-pair assembly backend")
    effective = deepcopy(spec)
    if smoke:
        # Small-scale wiring and shape check; do not use it for scientific performance evaluation.
        effective["simulation"].update(N=8, horizon=0.1)
        effective["basis"]["sizes"] = [4, 8]
        effective["seeds"] = {split: seeds[:1] for split, seeds in spec["seeds"].items()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    sources = [
        "kernels.py", "features.py", "simulate.py", "tests.py", "time_tests.py",
        "scripts/run_sindy_comparison.py", "scripts/prepare_1d_hat_experiment.py",
    ]
    if backend == "cached_1d":
        sources.append("scripts/hat_full_pair_assembly.py")
    manifest = {
        "status": "running", "smoke_only": smoke, "fitting_performed": False,
        "assembly_backend": backend,
        "assembly_timing": ("per-candidate accumulation, excluding shared geometry" if backend == "cached_1d"
                            else "per-candidate total assembly"),
        "requested_protocol": spec, "effective_protocol": effective,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "numpy_version": np.__version__, "splits": {},
        "test_functions": {
            "space_centers": DETERMINISTIC_SPACE_CENTERS.tolist(),
            "space_width": SPACE_TEST_WIDTH, "time_count": TIME_TEST_COUNT,
            "time_type": "bump", "time_support_factor": 1.8,
            "row_scale": 1.0 / np.sqrt(len(DETERMINISTIC_SPACE_CENTERS) * TIME_TEST_COUNT),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    started = perf_counter()
    sim = effective["simulation"]
    sizes = effective["basis"]["sizes"]
    try:
        for split, seeds in effective["seeds"].items():
            manifest["splits"][split] = []
            for seed in seeds:
                path = out_dir / split / str(seed)
                path.mkdir(parents=True)
                X, t, X0, noise, _ = reference_trajectory(
                    seed, particles=sim["N"], horizon=sim["horizon"],
                    step=sim["dt"], sigma=sim["sigma"],
                )
                if not np.all(np.isfinite(X)):
                    raise ValueError("nonfinite reference trajectory")
                Y = X.copy()  # Fix sigma_obs at 0 in the first stage.
                np.savez_compressed(path / "data.npz", X=X, Y=Y, t=t, X0=X0, driving_noise=noise)
                space, time = test_family(t)
                scale = manifest["test_functions"]["row_scale"]
                response = (compute_b_timeweak(Y, t, sim["sigma"], space, time) * scale).reshape(-1)
                if not np.all(np.isfinite(response)):
                    raise ValueError("nonfinite weak response")
                np.save(path / "response.npy", response)
                entry = {
                    "seed": seed, "brownian_seed": seed + 1,
                    "observation_sha256": array_hash(Y), "time_sha256": array_hash(t),
                    "response_sha256": array_hash(response),
                    "distances": distance_diagnostics(Y, t, effective["basis"]["radius_max"], sizes),
                    "designs": {},
                }
                candidates = [(f"hat_M{m}", basis_params(effective, m)) for m in sizes]
                candidates.append(("baseline_six_atoms", effective["baseline"]))
                cached, cached_seconds = {}, {}
                if backend == "cached_1d":
                    from scripts.hat_full_pair_assembly import assemble_1d_designs
                    cached, cached_seconds = assemble_1d_designs(Y, t, space, time, candidates)
                for name, params in candidates:
                    assembly_start = perf_counter()
                    A = cached[name] if backend == "cached_1d" else assemble_design(
                        Y, t, space, time, params, effective["pair_chunk_size"])
                    assembly_seconds = cached_seconds[name] if backend == "cached_1d" else perf_counter() - assembly_start
                    np.save(path / f"{name}.npy", A)
                    entry["designs"][name] = {
                        "basis": params, "assembly_seconds": assembly_seconds,
                        "sha256": array_hash(A), **matrix_diagnostics(A),
                    }
                write_json(path / "diagnostics.json", entry)
                manifest["splits"][split].append(entry)
                write_json(out_dir / "manifest.json", manifest)
        manifest["status"] = "smoke_passed" if smoke else "assembled_not_fitted"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] = perf_counter() - started
        write_json(out_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate the configuration only (default)")
    mode.add_argument("--smoke", action="store_true", help="check assembly under reduced settings; do not train")
    mode.add_argument("--assemble", action="store_true", help="save matrices for the full configuration; do not train")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    validate_spec(spec)
    if not args.smoke and not args.assemble:
        print(json.dumps({
            "status": "configuration_valid", "fitting_performed": False,
            "protocol": spec, "weak_rows_per_trajectory": len(DETERMINISTIC_SPACE_CENTERS) * TIME_TEST_COUNT,
            "estimator_status": spec["estimator_status"],
            "note": "This script only assembles matrices; fitting is a separate frozen-protocol step.",
        }, ensure_ascii=False, indent=2))
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    label = "smoke" if args.smoke else "assembly"
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / f"{label}-{timestamp}"
    manifest = prepare(spec, destination, smoke=args.smoke)
    print(json.dumps({"status": manifest["status"], "out_dir": str(destination),
                      "fitting_performed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
