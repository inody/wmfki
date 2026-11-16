import argparse
import csv
import json
import os
import sys
from itertools import product

import matplotlib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bandwidth import sample_pair_distances
from config import build_config
from kernels import make_radial_basis
from metrics import weighted_kernel_l2
from pipeline import fit_kernel_from_particles
from simulate import simulate_particles, simulate_with_kernel


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_list(text, cast):
    if isinstance(text, (list, tuple)):
        return [cast(x) for x in text]
    return [cast(x) for x in text.split(",") if x.strip()]


def qanr_config(N, R, sigma, alpha, powers, fit_pairs, seed, dt=0.01, T=100, reg_method="lasso", pair_replacement=True, solver="euler", time_s_factor=2.0):
    return build_config(
        {
            "simulation": {
                "N": N,
                "T": T,
                "dt": dt,
                "R": R,
                "sigma": sigma,
                "d": 1,
                "init": "gaussian_mixture",
                "init_centers": [0.5, 2.0, 3.5],
                "init_std": 0.005,
                "init_scale": 1.0,
                "seed": seed,
                "interaction": {"mode": "qanr", "eps": 1e-8, "r_max": 3.0},
                "solver": solver,
            },
            "space_tests": {
                "L": 25,
                "type": "gaussian",
                "layout": "grid",
                "quantiles": (0.01, 0.99),
                "s_factor": 1.5,
                "seed": seed,
            },
            "time_tests": {"type": "gaussian", "K": 8, "s_factor": time_s_factor},
            "pair_sampling": {
                "total_pairs": fit_pairs,
                "with_replacement": pair_replacement,
                "exclude_diagonal": True,
                "seed": seed,
            },
            "kernel_basis": {
                "M": len(powers),
                "basis": "grad_abs_power",
                "powers": powers,
                "eps": 1e-8,
            },
            "regression": {
                "method": reg_method,
                "standardize": True,
                "alpha": alpha,
                "max_iter": 50000,
                "tol": 1e-6,
            },
            "diagnostics": {"method": None, "seed": seed},
            "output": {"save_plots": False},
        }
    )


def true_theta_for_powers(powers):
    powers = np.asarray(powers, dtype=float)
    theta = np.zeros_like(powers, dtype=float)
    idx1 = np.where(np.isclose(powers, 1.0, rtol=0.0, atol=1e-12))[0]
    idx2 = np.where(np.isclose(powers, 2.0, rtol=0.0, atol=1e-12))[0]
    if idx1.size == 1:
        theta[idx1[0]] = 1.0
    if idx2.size == 1:
        theta[idx2[0]] = -1.0
    return theta


def find_first_crossing(r, y):
    for i in range(len(r) - 1):
        y0, y1 = y[i], y[i + 1]
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if y0 == 0:
            return float(r[i])
        if y0 * y1 < 0:
            return float(r[i] - y0 * (r[i + 1] - r[i]) / (y1 - y0))
    return None


def support_metrics(theta_hat, theta_true, threshold):
    est = np.abs(theta_hat) > threshold
    true = np.abs(theta_true) > 0
    tp = int(np.sum(est & true))
    fp = int(np.sum(est & ~true))
    fn = int(np.sum(~est & true))
    denom = tp + fp + fn
    tpr = float(tp / denom) if denom else 1.0
    return tp, fp, fn, tpr, int(np.sum(est))


def evaluate_fit(cfg, X, theta_hat, phi_true_fn, r_samples, support_threshold):
    powers = cfg["kernel_basis"]["powers"]
    theta_true = true_theta_for_powers(powers)
    phi_basis, _ = make_radial_basis(cfg["kernel_basis"])

    r_max = float(np.quantile(r_samples, 0.99))
    r_grid = np.linspace(max(1e-6, float(np.quantile(r_samples, 0.01))), r_max, 800)
    phi_true = phi_true_fn(r_grid)
    phi_hat = phi_basis(r_grid) @ theta_hat
    force_true = r_grid * phi_true
    force_hat = r_grid * phi_hat

    hist, edges = np.histogram(r_samples, bins=200, range=(0.0, r_max), density=True)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    phi_true_c = phi_true_fn(r_centers)
    phi_hat_c = phi_basis(r_centers) @ theta_hat
    force_l2w, force_l2w_rel = weighted_kernel_l2(
        r_centers * phi_true_c,
        r_centers * phi_hat_c,
        r_centers,
        hist,
        dr,
    )

    tp, fp, fn, tpr, nnz = support_metrics(theta_hat, theta_true, support_threshold)
    theta_l2 = float(np.linalg.norm(theta_hat - theta_true))
    return {
        "theta_true": theta_true,
        "r_grid": r_grid,
        "phi_true": phi_true,
        "phi_hat": phi_hat,
        "force_true": force_true,
        "force_hat": force_hat,
        "r_cross_hat": find_first_crossing(r_grid, force_hat),
        "force_l2w": force_l2w,
        "force_l2w_rel": force_l2w_rel,
        "theta_l2": theta_l2,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tpr": tpr,
        "nnz": nnz,
    }


def plot_best(out_dir, tag, cfg, X, t, K_true, K_hat, eval_data, theta_hat):
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)
    powers = np.asarray(cfg["kernel_basis"]["powers"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eval_data["r_grid"], eval_data["force_true"], label="true", linewidth=2)
    ax.plot(eval_data["r_grid"], eval_data["force_hat"], "--", label="learned", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(1.0, color="tab:blue", alpha=0.3)
    if eval_data["r_cross_hat"] is not None:
        ax.axvline(eval_data["r_cross_hat"], color="tab:orange", alpha=0.3, linestyle="--")
    ax.set_xlabel("r")
    ax.set_ylabel("force magnitude for z=r")
    ax.set_title(tag)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{tag}_force.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    theta_true = eval_data["theta_true"]
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.vlines(powers, 0.0, theta_true, color="tab:blue", alpha=0.65, label="theta_true")
    ax.scatter(powers, theta_true, color="tab:blue", s=22)
    ax.vlines(powers, 0.0, theta_hat, color="tab:orange", alpha=0.8)
    ax.scatter(powers, theta_hat, color="tab:orange", s=24, label="theta_hat")
    ax.set_xlabel("dictionary power")
    ax.set_ylabel("coefficient theta")
    ax.set_title(tag)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{tag}_theta.png"), dpi=160)
    plt.close(fig)

    R_sim, T1, N_sim, d_sim = X.shape
    T_sim = T1 - 1
    noise = np.random.default_rng(2026).normal(0.0, 1.0, size=(1, T_sim, N_sim, d_sim))
    X0 = X[:1, 0, :, :]
    X_true_sim = simulate_with_kernel(K_true, X0, t, sigma=cfg["simulation"]["sigma"], noise=noise)
    X_hat_sim = simulate_with_kernel(K_hat, X0, t, sigma=cfg["simulation"]["sigma"], noise=noise)

    idx = np.linspace(0, N_sim - 1, min(80, N_sim), dtype=int)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, i in enumerate(idx):
        ax.plot(t, X_true_sim[0, :, i, 0], color="tab:blue", alpha=0.35, linewidth=0.8, label="true" if k == 0 else None)
        ax.plot(t, X_hat_sim[0, :, i, 0], color="tab:orange", alpha=0.35, linewidth=0.8, linestyle="--", label="learned" if k == 0 else None)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(tag)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{tag}_trajectory.png"), dpi=160)
    plt.close(fig)


def run(args):
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "configs"))

    Ns = parse_list(args.Ns, int)
    Rs = parse_list(args.Rs, int)
    sigmas = parse_list(args.sigmas, float)
    alphas = parse_list(args.alphas, float)
    powers = parse_list(args.powers, float)
    dts = parse_list(getattr(args, "dts", [0.01]), float)
    T_total = getattr(args, "T_total", None)

    manifest = {
        "Ns": Ns,
        "Rs": Rs,
        "sigmas": sigmas,
        "alphas": alphas,
        "powers": powers,
        "dts": dts,
        "T_total": T_total,
        "fit_pairs": args.fit_pairs,
        "distance_pairs": args.distance_pairs,
        "support_threshold": args.support_threshold,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    reg_method = getattr(args, "reg_method", "lasso")
    pair_replacement = getattr(args, "pair_replacement", True)
    solver = getattr(args, "solver", "euler")
    time_s_factor = getattr(args, "time_s_factor", 2.0)
    multi_dt = len(dts) > 1
    rows = []
    best_by_setting = {}
    for N, R, sigma, alpha, dt in product(Ns, Rs, sigmas, alphas, dts):
        T = round(T_total / dt) if T_total is not None else 100
        tag = f"N{N}_R{R}_sigma{sigma:g}_alpha{alpha:.0e}".replace("+", "")
        if multi_dt:
            tag += f"_dt{dt:g}"
        print(f"\n=== {tag} ===")
        cfg = qanr_config(N, R, sigma, alpha, powers, args.fit_pairs, args.seed, dt=dt, T=T, reg_method=reg_method, pair_replacement=pair_replacement, solver=solver, time_s_factor=time_s_factor)
        with open(os.path.join(out_dir, "configs", f"{tag}.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        X, t, K_true, _, meta = simulate_particles(
            N=N,
            T=cfg["simulation"]["T"],
            dt=cfg["simulation"]["dt"],
            R=R,
            sigma=sigma,
            d=1,
            init="gaussian_mixture",
            init_centers=[0.5, 2.0, 3.5],
            init_std=0.005,
            init_scale=1.0,
            kernel_params=cfg["simulation"]["interaction"],
            seed=args.seed,
            solver=cfg["simulation"].get("solver", "euler"),
        )
        fit = fit_kernel_from_particles(cfg, X, t)
        theta_hat = np.asarray(fit["theta_hat"], dtype=float)
        r_samples = sample_pair_distances(
            X,
            num_pairs=args.distance_pairs,
            num_times=48,
            seed=args.seed,
            exclude_self=True,
        )
        ev = evaluate_fit(cfg, X, theta_hat, meta["phi_true"], r_samples, args.support_threshold)
        row = {
            "tag": tag,
            "N": N,
            "R": R,
            "sigma": sigma,
            "alpha": alpha,
            "dt": dt,
            "T": T,
            "nnz": ev["nnz"],
            "tpr": ev["tpr"],
            "tp": ev["tp"],
            "fp": ev["fp"],
            "fn": ev["fn"],
            "theta_l2": ev["theta_l2"],
            "force_l2w_rel": ev["force_l2w_rel"],
            "r_cross_hat": ev["r_cross_hat"],
            "cond_ata": fit["diagnostics"].get("cond_ata"),
        }
        rows.append(row)
        print(
            "  "
            f"nnz={row['nnz']} tpr={row['tpr']:.3f} "
            f"theta_l2={row['theta_l2']:.3e} force_rel={row['force_l2w_rel']:.3e} "
            f"r_cross={row['r_cross_hat']}"
        )

        key = (N, R, sigma, dt)
        score = ev["force_l2w_rel"] + 0.05 * max(0, ev["nnz"] - 2)
        if key not in best_by_setting or score < best_by_setting[key]["score"]:
            best_by_setting[key] = {
                "score": score,
                "tag": tag,
                "cfg": cfg,
                "X": X,
                "t": t,
                "K_true": K_true,
                "K_hat": fit["K_hat"],
                "eval": ev,
                "theta_hat": theta_hat,
                "row": row,
            }

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_rows = []
    for key, item in best_by_setting.items():
        best_rows.append(item["row"])
        plot_best(
            out_dir,
            item["tag"],
            item["cfg"],
            item["X"],
            item["t"],
            item["K_true"],
            item["K_hat"],
            item["eval"],
            item["theta_hat"],
        )

    with open(os.path.join(out_dir, "best_by_setting.json"), "w") as f:
        json.dump(best_rows, f, indent=2)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {os.path.join(out_dir, 'best_by_setting.json')}")
    print(f"Saved representative plots in {os.path.join(out_dir, 'plots')}")


def run_sweep(cfg):
    sc = cfg.get("sweep", {})

    class _Args:
        pass

    args = _Args()
    args.out_dir = sc.get("out_dir", "outputs/sweep")
    args.fit_pairs = int(sc.get("fit_pairs", 10000))
    args.distance_pairs = int(sc.get("distance_pairs", 50000))
    args.support_threshold = float(sc.get("support_threshold", 1e-3))
    args.seed = int(sc.get("seed", 1234))
    args.Ns = sc.get("Ns", [500])
    args.Rs = sc.get("Rs", [5])
    args.sigmas = sc.get("sigmas", [0.1])
    args.alphas = sc.get("alphas", [1e-4])
    args.powers = sc.get("powers", [1, 2, 3, 4, 5, 6, 7])
    args.dts = sc.get("dts", [0.01])
    args.T_total = sc.get("T_total", None)
    args.reg_method = sc.get("reg_method", "lasso")
    args.pair_replacement = sc.get("pair_replacement", True)
    args.solver = sc.get("solver", "euler")
    args.time_s_factor = sc.get("time_s_factor", 2.0)

    run(args)


def main():
    parser = argparse.ArgumentParser(description="Messenger-style QANR particle-only weak-form sweep.")
    parser.add_argument("--out-dir", default="outputs/messenger_qanr/quick")
    parser.add_argument("--Ns", default="500,1000")
    parser.add_argument("--Rs", default="1,4")
    parser.add_argument("--sigmas", default="0,0.1,0.4472135955")
    parser.add_argument("--alphas", default="1e-6,1e-5,1e-4")
    parser.add_argument("--powers", default="1,2,3,4,5,6,7")
    parser.add_argument("--dts", default="0.01")
    parser.add_argument("--T-total", type=float, default=None, dest="T_total")
    parser.add_argument("--fit-pairs", type=int, default=10000)
    parser.add_argument("--distance-pairs", type=int, default=50000)
    parser.add_argument("--support-threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
