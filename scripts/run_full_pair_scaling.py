"""Reproduce the R05 one-dimensional full-pair scaling experiments.

All runs use the deterministic 25-center spatial test family on [0, 4].  The
``--time-test-type`` switch permits a like-for-like Gaussian/bump comparison.
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import fit_kernel_from_particles
from scripts.r05_numerical_settings import (
    SPACE_TEST_CENTERS,
    SPACE_TEST_WIDTH,
    apply_fixed_1d_tests,
)
from scripts.run_messenger_qanr_sweep import qanr_config
from simulate import simulate_particles

TRUE_THETA = np.array([1.0, -1.0])
DT_VALUES = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002)
SIGMA_VALUES = (0.01, 0.05, 0.1, 0.2, 0.4)
PARTICLE_VALUES = (50, 100, 200, 400)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matched_config(*, particles, repetitions, steps, dt, sigma, seed,
                   solver, time_test_type):
    config = qanr_config(
        N=particles,
        R=repetitions,
        sigma=sigma,
        alpha=0.0,
        powers=[1, 2],
        fit_pairs=particles * (particles - 1),
        seed=seed,
        dt=dt,
        T=steps,
        reg_method="ridge",
        pair_replacement=False,
        solver=solver,
        time_s_factor=1.8,
    )
    apply_fixed_1d_tests(
        config,
        time_test_type=time_test_type,
        horizon=steps * dt,
    )
    config["regression"].update(standardize=False, alpha=0.0)
    config["diagnostics"]["method"] = None
    config["weak_form"]["normalize_rows"] = True
    return config


def simulate_and_fit(*, particles, repetitions, steps, dt, sigma, seed,
                     solver, time_test_type):
    config = matched_config(
        particles=particles,
        repetitions=repetitions,
        steps=steps,
        dt=dt,
        sigma=sigma,
        seed=seed,
        solver=solver,
        time_test_type=time_test_type,
    )
    X, t, *_ = simulate_particles(
        N=particles,
        T=steps,
        dt=dt,
        R=repetitions,
        sigma=sigma,
        d=1,
        init="gaussian_mixture",
        init_centers=[0.5, 2.0, 3.5],
        init_std=0.005,
        init_scale=1.0,
        kernel_params=config["simulation"]["interaction"],
        seed=seed,
        solver=solver,
    )
    fit = fit_kernel_from_particles(config, X, t, pair_mode="full")
    theta = np.asarray(fit["theta_hat"], dtype=float)
    return theta, fit["diagnostics"]["timing"]


def grid_matched_rows(time_test_type):
    rows = []
    for dt in DT_VALUES:
        started = perf_counter()
        theta, timing = simulate_and_fit(
            particles=200,
            repetitions=3,
            steps=round(10.0 / dt),
            dt=dt,
            sigma=0.0,
            seed=1234,
            solver="euler",
            time_test_type=time_test_type,
        )
        rows.append({
            "dt": dt,
            "theta_1": theta[0],
            "theta_2": theta[1],
            "theta_l2": float(np.linalg.norm(theta - TRUE_THETA)),
            "response_seconds": timing["response_seconds"],
            "design_seconds": timing["design_seconds"],
            "elapsed_seconds": perf_counter() - started,
        })
        print(f"dt={dt:g}: error={rows[-1]['theta_l2']:.4e}", flush=True)
    return rows


def stochastic_job(job):
    kind, value, seed_index, time_test_type = job
    if kind == "sigma":
        particles, sigma, seeds = 100, float(value), 30
    else:
        particles, sigma, seeds = int(value), 0.1, 20
    seed = 1234 + seed_index * 1000
    started = perf_counter()
    theta, timing = simulate_and_fit(
        particles=particles,
        repetitions=1,
        steps=2000,
        dt=0.001,
        sigma=sigma,
        seed=seed,
        solver="rk4",
        time_test_type=time_test_type,
    )
    return {
        "kind": kind,
        "value": value,
        "particles": particles,
        "sigma": sigma,
        "seed_index": seed_index,
        "trajectory_seed": seed,
        "theta_1": theta[0],
        "theta_2": theta[1],
        "theta_l2": float(np.linalg.norm(theta - TRUE_THETA)),
        "response_seconds": timing["response_seconds"],
        "design_seconds": timing["design_seconds"],
        "elapsed_seconds": perf_counter() - started,
    }


def stochastic_rows(time_test_type, workers):
    jobs = []
    for sigma in SIGMA_VALUES:
        jobs.extend(("sigma", sigma, index, time_test_type) for index in range(30))
    for particles in PARTICLE_VALUES:
        jobs.extend(("particles", particles, index, time_test_type) for index in range(20))
    if workers == 1:
        iterator = map(stochastic_job, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(stochastic_job, jobs)
    rows = []
    try:
        for index, row in enumerate(iterator, 1):
            rows.append(row)
            if index % 10 == 0 or index == len(jobs):
                print(f"stochastic runs: {index}/{len(jobs)}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    return rows


def loglog_slope(x, y):
    return float(np.polyfit(np.log(np.asarray(x)), np.log(np.asarray(y)), 1)[0])


def summarize(dt_rows, stochastic):
    sigma_rmse = []
    for sigma in SIGMA_VALUES:
        values = [r["theta_l2"] for r in stochastic if r["kind"] == "sigma" and r["sigma"] == sigma]
        sigma_rmse.append(float(np.sqrt(np.mean(np.square(values)))))
    particle_rmse = []
    for particles in PARTICLE_VALUES:
        values = [r["theta_l2"] for r in stochastic if r["kind"] == "particles" and r["particles"] == particles]
        particle_rmse.append(float(np.sqrt(np.mean(np.square(values)))))
    return {
        "dt_slope": loglog_slope([r["dt"] for r in dt_rows], [r["theta_l2"] for r in dt_rows]),
        "sigma_values": SIGMA_VALUES,
        "sigma_rmse": sigma_rmse,
        "sigma_slope": loglog_slope(SIGMA_VALUES, sigma_rmse),
        "particle_values": PARTICLE_VALUES,
        "particle_rmse": particle_rmse,
        "particle_slope": loglog_slope(PARTICLE_VALUES, particle_rmse),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--time-test-type", choices=("gaussian", "bump"), default="bump")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-dt", action="store_true")
    parser.add_argument("--skip-stochastic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    dt_rows = [] if args.skip_dt else grid_matched_rows(args.time_test_type)
    stochastic = [] if args.skip_stochastic else stochastic_rows(args.time_test_type, args.workers)
    if dt_rows:
        write_rows(out / "dt_scaling.csv", dt_rows)
    if stochastic:
        write_rows(out / "stochastic_scaling.csv", stochastic)
    summary = summarize(dt_rows, stochastic) if dt_rows and stochastic else {}
    metadata = {
        "time_test_type": args.time_test_type,
        "space_test_selection": "deterministic and fixed before all trajectories",
        "space_centers": SPACE_TEST_CENTERS.tolist(),
        "space_width": SPACE_TEST_WIDTH,
        "workers": args.workers,
        "elapsed_seconds": perf_counter() - started,
        "summary": summary,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
