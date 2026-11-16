"""Isolate weak-form quadrature error on one deterministic reference trajectory.

The experiment uses the one-dimensional matched two-atom benchmark from the
manuscript.  A fine RK4 trajectory is generated once for each reference step.
All observation grids are nested subsamples of that trajectory, and the spatial
test functions are fixed from the finest trajectory before any weak systems are
assembled.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import make_trapz_weights
from kernels import make_kernel_basis
from regression import fit_theta
from simulate import make_true_kernel, simulate_particles
from scripts.r05_numerical_settings import SPACE_TEST_CENTERS, SPACE_TEST_WIDTH
from tests import make_test_functions
from time_tests import make_time_test_functions


DEFAULT_DT_VALUES = (0.1, 0.05, 0.025, 0.0125, 0.00625)
TRUE_THETA = np.array([1.0, -1.0])


def _initial_positions(*, seed, repetitions, particles):
    X0, _, _, _, _ = simulate_particles(
        N=particles,
        T=0,
        dt=1.0,
        R=repetitions,
        sigma=0.0,
        d=1,
        init="gaussian_mixture",
        init_centers=[0.5, 2.0, 3.5],
        init_std=0.005,
        init_scale=1.0,
        kernel_params={"mode": "qanr", "eps": 1e-8, "r_max": 3.0},
        seed=seed,
        solver="rk4",
    )
    return X0[:, 0]


def _batched_qanr_drift(x, kernel):
    z = x[:, :, None, :] - x[:, None, :, :]
    return kernel(z).mean(axis=2)


def simulate_reference_rk4(*, X0, horizon, step, kernel):
    steps_float = horizon / step
    steps = int(round(steps_float))
    if not np.isclose(steps, steps_float, rtol=0.0, atol=1e-10):
        raise ValueError("horizon must be an integer multiple of the reference step")

    time = np.linspace(0.0, horizon, steps + 1)
    trajectory = np.empty((X0.shape[0], steps + 1, X0.shape[1], 1), dtype=float)
    trajectory[:, 0] = X0
    x = X0.copy()
    for n in range(steps):
        k1 = _batched_qanr_drift(x, kernel)
        k2 = _batched_qanr_drift(x + 0.5 * step * k1, kernel)
        k3 = _batched_qanr_drift(x + 0.5 * step * k2, kernel)
        k4 = _batched_qanr_drift(x + step * k3, kernel)
        x = x + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[:, n + 1] = x
    return trajectory, time


def fixed_spatial_tests():
    return make_test_functions(
        {
            "centers": SPACE_TEST_CENTERS[:, None],
            "s": SPACE_TEST_WIDTH,
            "type": "gaussian",
        }
    )


def instantaneous_weak_features(
    trajectory,
    test_functions,
    kernel_basis,
    *,
    chunk_size=16,
):
    """Evaluate the time integrands before applying quadrature weights.

    The full ordered-pair feature is evaluated as

        N^{-2} sum_i grad(phi)(X_i) dot sum_j Phi(X_i-X_j),

    which is algebraically identical to the ordered double sum and avoids
    repeating the spatial-test evaluation for every pair.
    """

    repetitions, time_count, particles, _ = trajectory.shape
    test_count = test_functions["centers"].shape[0]
    atom_count = kernel_basis(np.zeros((1, 1))).shape[1]
    mean_phi = np.empty((repetitions, time_count, test_count), dtype=float)
    pair_features = np.empty(
        (repetitions, time_count, test_count, atom_count), dtype=float
    )

    for repetition in range(repetitions):
        for start in range(0, time_count, chunk_size):
            stop = min(start + chunk_size, time_count)
            x = trajectory[repetition, start:stop]
            phi = test_functions["phi"](x)
            grad_phi = test_functions["grad_phi"](x)
            mean_phi[repetition, start:stop] = phi.mean(axis=1)

            z = x[:, :, None, :] - x[:, None, :, :]
            basis_sum = kernel_basis(z).sum(axis=2)
            pair_features[repetition, start:stop] = (
                np.einsum("bnld,bnmd->blm", grad_phi, basis_sum)
                / (particles * particles)
            )
    return mean_phi, pair_features


def integrate_weak_system(
    mean_phi,
    pair_features,
    time,
    *,
    time_test_count=8,
    time_test_type="bump",
):
    config = {
        "time_tests": {
            "type": time_test_type,
            "K": time_test_count,
            "s_factor": 1.8,
            "sigma_div": 6,
        }
    }
    time_tests = make_time_test_functions(config, time)
    weights = make_trapz_weights(time)
    weighted_dpsi = time_tests["dpsi"] * weights[None, :]
    weighted_psi = time_tests["psi"] * weights[None, :]

    response = -np.einsum("kt,rtl->rlk", weighted_dpsi, mean_phi)
    design = np.einsum("kt,rtlm->rlkm", weighted_psi, pair_features)
    row_scale = 1.0 / np.sqrt(mean_phi.shape[2] * time_test_count)
    return design * row_scale, response * row_scale


def fit_ols(design, response):
    flat_design = design.reshape(-1, design.shape[-1])
    flat_response = response.reshape(-1)
    theta, _ = fit_theta(
        flat_design,
        flat_response,
        method="ridge",
        alpha=0.0,
        standardize=True,
    )
    return np.asarray(theta, dtype=float)


def _subsample_indices(time, observation_step):
    reference_step = float(time[1] - time[0])
    stride_float = observation_step / reference_step
    stride = int(round(stride_float))
    if not np.isclose(stride, stride_float, rtol=0.0, atol=1e-9):
        raise ValueError("observation steps must be nested in the reference grid")
    indices = np.arange(0, time.size, stride, dtype=int)
    if indices[-1] != time.size - 1:
        raise ValueError("observation grid must include the final reference time")
    return indices


def evaluate_reference(
    trajectory,
    time,
    test_functions,
    kernel_basis,
    observation_steps,
    *,
    reference_divisor,
    chunk_size,
    time_test_type,
):
    mean_phi, pair_features = instantaneous_weak_features(
        trajectory,
        test_functions,
        kernel_basis,
        chunk_size=chunk_size,
    )
    design_ref, response_ref = integrate_weak_system(
        mean_phi, pair_features, time, time_test_type=time_test_type
    )
    theta_ref = fit_ols(design_ref, response_ref)

    rows = []
    for observation_step in observation_steps:
        indices = _subsample_indices(time, observation_step)
        design, response = integrate_weak_system(
            mean_phi[:, indices],
            pair_features[:, indices],
            time[indices],
            time_test_type=time_test_type,
        )
        theta = fit_ols(design, response)
        rows.append(
            {
                "reference_divisor": reference_divisor,
                "reference_step": float(time[1] - time[0]),
                "dt": observation_step,
                "theta_1": theta[0],
                "theta_2": theta[1],
                "theta_error": float(np.linalg.norm(theta - TRUE_THETA)),
                "theta_to_reference": float(np.linalg.norm(theta - theta_ref)),
                "design_error": float(np.linalg.norm(design - design_ref)),
                "response_error": float(np.linalg.norm(response - response_ref)),
            }
        )
    reference = {
        "reference_divisor": reference_divisor,
        "reference_step": float(time[1] - time[0]),
        "theta_1": float(theta_ref[0]),
        "theta_2": float(theta_ref[1]),
        "theta_error": float(np.linalg.norm(theta_ref - TRUE_THETA)),
        "design": design_ref,
        "response": response_ref,
    }
    return rows, reference


def loglog_slope(rows, metric):
    dt = np.asarray([row["dt"] for row in rows], dtype=float)
    values = np.asarray([row[metric] for row in rows], dtype=float)
    valid = values > 0.0
    return float(np.polyfit(np.log(dt[valid]), np.log(values[valid]), 1)[0])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    output_dir = ROOT / args.output_dir
    observation_steps = tuple(float(value) for value in args.observation_steps)
    divisors = sorted({int(value) for value in args.reference_divisors})
    finest_divisor = max(divisors)
    base_step = max(observation_steps)

    X0 = _initial_positions(
        seed=args.seed,
        repetitions=args.repetitions,
        particles=args.particles,
    )
    kernel, _, _, _ = make_true_kernel(
        {"mode": "qanr", "eps": 1e-8, "r_max": 3.0}
    )
    kernel_basis, _ = make_kernel_basis(
        {"M": 2, "basis": "grad_abs_power", "powers": [1, 2], "eps": 1e-8}
    )

    finest_step = base_step / finest_divisor
    print(f"Generating finest RK4 trajectory with h={finest_step:.10g}")
    finest_trajectory, finest_time = simulate_reference_rk4(
        X0=X0,
        horizon=args.horizon,
        step=finest_step,
        kernel=kernel,
    )
    test_functions = fixed_spatial_tests()

    all_rows = []
    references = {}
    final_states = {}
    for divisor in sorted(divisors, reverse=True):
        step = base_step / divisor
        if divisor == finest_divisor:
            trajectory, time = finest_trajectory, finest_time
        else:
            print(f"Generating independent RK4 check trajectory with h={step:.10g}")
            trajectory, time = simulate_reference_rk4(
                X0=X0,
                horizon=args.horizon,
                step=step,
                kernel=kernel,
            )
        print(f"Evaluating weak systems for reference divisor {divisor}")
        rows, reference = evaluate_reference(
            trajectory,
            time,
            test_functions,
            kernel_basis,
            observation_steps,
            reference_divisor=divisor,
            chunk_size=args.chunk_size,
            time_test_type=args.time_test_type,
        )
        all_rows.extend(rows)
        references[divisor] = reference
        final_states[divisor] = trajectory[:, -1].copy()

    finest_rows = [
        row for row in all_rows if row["reference_divisor"] == finest_divisor
    ]
    coarser_divisor = min(divisors)
    summary = {
        "seed": args.seed,
        "particles": args.particles,
        "repetitions": args.repetitions,
        "horizon": args.horizon,
        "observation_steps": observation_steps,
        "time_test_type": args.time_test_type,
        "space_test_selection": "deterministic and fixed before all trajectories",
        "reference_divisors": divisors,
        "fixed_spatial_centers": test_functions["centers"][:, 0].tolist(),
        "spatial_width": float(test_functions["s"]),
        "slopes_finest_reference": {
            metric: loglog_slope(finest_rows, metric)
            for metric in (
                "theta_error",
                "theta_to_reference",
                "design_error",
                "response_error",
            )
        },
        "reference_estimates": {
            str(divisor): {
                "step": references[divisor]["reference_step"],
                "theta": [
                    references[divisor]["theta_1"],
                    references[divisor]["theta_2"],
                ],
                "theta_error": references[divisor]["theta_error"],
            }
            for divisor in divisors
        },
        "reference_stability": {
            "trajectory_final_l2": float(
                np.linalg.norm(final_states[finest_divisor] - final_states[coarser_divisor])
            ),
            "design_l2": float(
                np.linalg.norm(
                    references[finest_divisor]["design"]
                    - references[coarser_divisor]["design"]
                )
            ),
            "response_l2": float(
                np.linalg.norm(
                    references[finest_divisor]["response"]
                    - references[coarser_divisor]["response"]
                )
            ),
        },
    }

    write_csv(output_dir / "results.csv", all_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_dir / 'results.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/validation/fixed_trajectory_quadrature",
    )
    parser.add_argument("--particles", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reference-divisors", nargs="+", type=int, default=[256, 512])
    parser.add_argument(
        "--observation-steps",
        nargs="+",
        type=float,
        default=list(DEFAULT_DT_VALUES),
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--time-test-type",
        choices=("gaussian", "bump"),
        default="bump",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
