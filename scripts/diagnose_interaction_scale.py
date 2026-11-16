import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bandwidth import sample_pair_distances
from config import build_config, merge_config
from kernels import make_radial_basis
from pipeline import fit_kernel_from_particles
from simulate import simulate_particles, simulate_with_kernel


def _load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def _sync_ell_grid(cfg, meta):
    kb = cfg["kernel_basis"]
    if kb.get("ell_grid") is not None:
        return

    km = meta["kernel_meta"]
    l1 = km.get("l1", 0.5)
    l2 = km.get("l2", 1.5)
    ell_min = 0.2 * min(l1, l2)
    ell_max = 5.0 * max(l1, l2)
    M = int(kb["M"])
    base_grid = np.logspace(np.log10(ell_min), np.log10(ell_max), M)
    ells = np.unique(np.sort(np.concatenate([base_grid, [l1, l2]])))
    if ells.size > M + 5:
        idx = np.linspace(0, ells.size - 1, M).round().astype(int)
        ells = ells[idx]
    kb["ell_grid"] = ells.tolist()
    kb["M"] = len(ells)


def _true_crossing(interaction):
    mode = interaction.get("mode")
    if mode == "qanr":
        return 1.0
    if mode not in ("morse", "morse_r"):
        return None

    a = float(interaction["a"])
    b = float(interaction["b"])
    l1 = float(interaction["l1"])
    l2 = float(interaction["l2"])
    denom = (1.0 / l1) - (1.0 / l2)
    if a <= 0.0 or b <= 0.0 or denom == 0.0:
        return None
    r = np.log(a / b) / denom
    if not np.isfinite(r) or r <= 0.0:
        return None
    return float(r)


def _find_crossings(r, y):
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    roots = []
    for n in range(r.size - 1):
        y0 = y[n]
        y1 = y[n + 1]
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if y0 == 0.0:
            roots.append(float(r[n]))
        elif y0 * y1 < 0.0:
            root = r[n] - y0 * (r[n + 1] - r[n]) / (y1 - y0)
            roots.append(float(root))
    return roots


def _first_observed_crossing(roots, r_min, r_max):
    for root in roots:
        if r_min <= root <= r_max:
            return root
    return None


def _fmt(x, precision=4):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    return f"{float(x):.{precision}g}"


def _dictionary_axis(kernel_meta, theta_size):
    if "ells" in kernel_meta:
        return np.asarray(kernel_meta["ells"], dtype=float), "dictionary scale ell", None, True
    if "powers" in kernel_meta:
        return np.asarray(kernel_meta["powers"], dtype=float), "dictionary power", None, False
    if "labels" in kernel_meta:
        labels = list(kernel_meta["labels"])
        return np.arange(len(labels), dtype=float), "dictionary atom", labels, False
    return np.arange(theta_size, dtype=float), "dictionary index", None, False


def _print_quantiles(name, values):
    parts = [f"{k}={_fmt(v)}" for k, v in values.items()]
    print(f"{name}: " + ", ".join(parts))


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _plot_diagnostics(
    prefix,
    r_samples,
    qs,
    r_grid,
    phi_true,
    phi_hat,
    roots_true,
    roots_hat,
    theta_hat,
    theta_true_basis,
    kernel_meta,
    X,
    t,
    K_true,
    K_hat,
    sigma,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_parent(f"{prefix}_dummy")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(r_samples, bins=80, density=True, alpha=0.75, color="tab:gray")
    for label, color in [("q10", "tab:blue"), ("q50", "tab:green"), ("q90", "tab:purple")]:
        ax.axvline(qs[label], color=color, linestyle="--", linewidth=1.2, label=label)
    for root in roots_true:
        ax.axvline(root, color="tab:red", linewidth=1.6, label="true crossing")
    ax.set_xlabel("pair distance r")
    ax.set_ylabel("density")
    ax.set_title("Observed Pair-Distance Distribution")
    ax.legend()
    fig.tight_layout()
    path = f"{prefix}_distance_hist.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r_grid, phi_true, label="phi_true", color="tab:blue", linewidth=2)
    ax.plot(r_grid, phi_hat, label="phi_hat", color="tab:orange", linestyle="--", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    for root in roots_true:
        ax.axvline(root, color="tab:blue", alpha=0.35, linewidth=1.2)
    for root in roots_hat:
        ax.axvline(root, color="tab:orange", alpha=0.35, linestyle="--", linewidth=1.2)
    ax.set_xlabel("r")
    ax.set_ylabel("phi(r)")
    ax.set_title("Interaction Shape")
    ax.legend()
    fig.tight_layout()
    path = f"{prefix}_phi.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r_grid, r_grid * phi_true, label="r phi_true", color="tab:blue", linewidth=2)
    ax.plot(
        r_grid,
        r_grid * phi_hat,
        label="r phi_hat",
        color="tab:orange",
        linestyle="--",
        linewidth=2,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    for root in roots_true:
        ax.axvline(root, color="tab:blue", alpha=0.35, linewidth=1.2)
    for root in roots_hat:
        ax.axvline(root, color="tab:orange", alpha=0.35, linestyle="--", linewidth=1.2)
    ax.set_xlabel("r")
    ax.set_ylabel("force magnitude for z=r")
    ax.set_title("Radial Force Shape")
    ax.legend()
    fig.tight_layout()
    path = f"{prefix}_force.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")

    x, xlabel, xtick_labels, use_logx = _dictionary_axis(kernel_meta, theta_hat.size)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, color="black", linewidth=0.8)
    if theta_true_basis is not None:
        ax.vlines(x, 0.0, theta_true_basis, color="tab:blue", alpha=0.65, label="theta_true")
        ax.scatter(x, theta_true_basis, color="tab:blue", s=22)
    ax.vlines(x, 0.0, theta_hat, color="tab:orange", alpha=0.8)
    ax.scatter(x, theta_hat, color="tab:orange", s=24, label="theta_hat")
    if use_logx:
        ax.set_xscale("log")
    if xtick_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=30, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("coefficient theta")
    ax.set_title("Recovered Dictionary Coefficients")
    ax.legend()
    fig.tight_layout()
    path = f"{prefix}_theta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")

    R_sim, T1, N_sim, d_sim = X.shape
    T_sim = T1 - 1
    noise = np.random.default_rng(2026).normal(0.0, 1.0, size=(1, T_sim, N_sim, d_sim))
    X0 = X[:1, 0, :, :]
    X_true_sim = simulate_with_kernel(K_true, X0, t, sigma=sigma, noise=noise)
    X_hat_sim = simulate_with_kernel(K_hat, X0, t, sigma=sigma, noise=noise)

    idx = np.linspace(0, N_sim - 1, min(80, N_sim), dtype=int)
    fig, ax = plt.subplots(figsize=(7, 4))
    comp = 0
    for n, i in enumerate(idx):
        ax.plot(
            t,
            X_true_sim[0, :, i, comp],
            color="tab:blue",
            alpha=0.35,
            linewidth=0.8,
            label="true" if n == 0 else None,
        )
    for n, i in enumerate(idx):
        ax.plot(
            t,
            X_hat_sim[0, :, i, comp],
            color="tab:orange",
            alpha=0.35,
            linewidth=0.8,
            linestyle="--",
            label="learned" if n == 0 else None,
        )
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title("Trajectory Replay With Shared Noise")
    ax.legend()
    fig.tight_layout()
    path = f"{prefix}_trajectory.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")


def _true_theta_on_current_basis(cfg, interaction, theta_from_sim):
    if theta_from_sim is not None:
        theta = np.asarray(theta_from_sim, dtype=float)
        if theta.shape == (int(cfg["kernel_basis"]["M"]),):
            return theta

    mode = interaction.get("mode")
    basis = cfg["kernel_basis"].get("basis")
    if (mode, basis) not in (("morse", "exp"), ("morse_r", "exp_r")):
        if (mode, basis) == ("log_attraction", "messenger_log2d"):
            theta = np.zeros(8, dtype=float)
            theta[7] = float(interaction.get("strength", -1.0 / (2.0 * np.pi)))
            return theta
        if (mode, basis) == ("log_attraction", "grad_abs_power"):
            powers = np.asarray(cfg["kernel_basis"].get("powers", []), dtype=float)
            idx = np.where(np.isclose(powers, 0.0, rtol=0.0, atol=1e-12))[0]
            if idx.size != 1:
                return None
            theta = np.zeros_like(powers, dtype=float)
            theta[idx[0]] = float(interaction.get("strength", -1.0 / (2.0 * np.pi)))
            return theta
        if (mode, basis) != ("qanr", "grad_abs_power"):
            return None
        powers = np.asarray(cfg["kernel_basis"].get("powers", []), dtype=float)
        if powers.size == 0:
            return None
        idx1 = np.where(np.isclose(powers, 1.0, rtol=0.0, atol=1e-12))[0]
        idx2 = np.where(np.isclose(powers, 2.0, rtol=0.0, atol=1e-12))[0]
        if idx1.size != 1 or idx2.size != 1:
            return None
        theta = np.zeros_like(powers, dtype=float)
        theta[idx1[0]] = 1.0
        theta[idx2[0]] = -1.0
        return theta
    if cfg["kernel_basis"].get("ell_grid") is None:
        return None

    ells = np.asarray(cfg["kernel_basis"]["ell_grid"], dtype=float)
    l1 = float(interaction["l1"])
    l2 = float(interaction["l2"])
    idx1 = np.where(np.isclose(ells, l1, rtol=0.0, atol=1e-12))[0]
    idx2 = np.where(np.isclose(ells, l2, rtol=0.0, atol=1e-12))[0]
    if idx1.size != 1 or idx2.size != 1:
        return None

    theta = np.zeros_like(ells, dtype=float)
    theta[idx1[0]] = float(interaction["a"])
    theta[idx2[0]] = -float(interaction["b"])
    return theta


def run_diagnostic(cfg):
    dcfg = cfg.get("diagnose", {})
    num_pairs = int(dcfg.get("num_pairs", 100000))
    num_times = int(dcfg.get("num_times", 64))
    grid_size = int(dcfg.get("grid_size", 1000))
    fit_pairs = dcfg.get("fit_pairs")
    plot_prefix = dcfg.get("plot_prefix")

    if fit_pairs is not None:
        cfg["pair_sampling"]["total_pairs"] = int(fit_pairs)

    sim_cfg = cfg["simulation"]
    interaction = sim_cfg["interaction"]

    print(f"Interaction: {interaction}")

    X, t, K_true, theta_true, meta = simulate_particles(
        N=sim_cfg["N"],
        T=sim_cfg["T"],
        dt=sim_cfg["dt"],
        R=sim_cfg["R"],
        sigma=sim_cfg["sigma"],
        d=sim_cfg.get("d", 1),
        init=sim_cfg["init"],
        init_scale=sim_cfg["init_scale"],
        init_centers=sim_cfg.get("init_centers"),
        init_std=sim_cfg.get("init_std"),
        kernel_params=interaction,
        seed=sim_cfg["seed"],
        external_drift_fn=sim_cfg.get("external_drift_fn"),
    )

    _sync_ell_grid(cfg, meta)
    cfg["diagnostics"]["true_kernel"] = {"phi_true_fn": meta["phi_true"]}
    theta_true_basis = _true_theta_on_current_basis(cfg, interaction, theta_true)

    r_samples = sample_pair_distances(
        X,
        num_pairs=num_pairs,
        num_times=num_times,
        seed=cfg["diagnostics"].get("seed", 1234),
        time_mode=cfg.get("bandwidth", {}).get("time_mode", "uniform"),
        exclude_self=True,
    )
    if r_samples.size == 0:
        raise RuntimeError("No pair distances were sampled.")

    qs = {
        "q01": np.quantile(r_samples, 0.01),
        "q05": np.quantile(r_samples, 0.05),
        "q10": np.quantile(r_samples, 0.10),
        "q25": np.quantile(r_samples, 0.25),
        "q50": np.quantile(r_samples, 0.50),
        "q75": np.quantile(r_samples, 0.75),
        "q90": np.quantile(r_samples, 0.90),
        "q95": np.quantile(r_samples, 0.95),
        "q99": np.quantile(r_samples, 0.99),
    }

    print("\nDistance distribution from observed particles")
    _print_quantiles("r", qs)

    print("\nFitting kernel with current weak-form pipeline...")
    fit = fit_kernel_from_particles(cfg, X, t)
    diag = fit["diagnostics"]
    test_meta = diag["test_meta"]
    s = float(test_meta["s"])
    centers = np.asarray(test_meta["centers"], dtype=float).reshape(-1)
    if centers.size > 1:
        center_delta = float(np.median(np.diff(np.sort(centers))))
    else:
        center_delta = None

    r_cross_true = _true_crossing(interaction)
    print("\nWeak-form spatial scale")
    print(f"test width s = {_fmt(s)}")
    print(f"median test-center spacing = {_fmt(center_delta)}")
    if r_cross_true is not None:
        print(f"true sign-crossing r_cross = {_fmt(r_cross_true)}")
        print(f"s / r_cross = {_fmt(s / r_cross_true)}")
    if "l1" in interaction:
        print(f"s / l1 = {_fmt(s / float(interaction['l1']))}")
    if "l2" in interaction:
        print(f"s / l2 = {_fmt(s / float(interaction['l2']))}")

    r_min = max(1e-8, float(qs["q01"]))
    r_max = float(qs["q99"])
    if r_cross_true is not None:
        r_max = max(r_max, 1.5 * r_cross_true)
    r_grid = np.linspace(1e-8, r_max, int(grid_size))

    phi_true = meta["phi_true"](r_grid)
    phi_basis, _ = make_radial_basis(
        {
            "M": cfg["kernel_basis"]["M"],
            "basis": cfg["kernel_basis"]["basis"],
            "ell_grid": cfg["kernel_basis"].get("ell_grid"),
            "powers": cfg["kernel_basis"].get("powers"),
            "ell_min": cfg["kernel_basis"].get("ell_min"),
            "ell_max": cfg["kernel_basis"].get("ell_max"),
            "p_min": cfg["kernel_basis"].get("p_min"),
            "p_max": cfg["kernel_basis"].get("p_max"),
            "eps": cfg["kernel_basis"].get("eps", 1e-3),
        }
    )
    theta_hat = np.asarray(fit["theta_hat"], dtype=float)
    phi_hat = phi_basis(r_grid) @ theta_hat

    roots_true = _find_crossings(r_grid, phi_true)
    roots_hat = _find_crossings(r_grid, phi_hat)
    root_true_obs = _first_observed_crossing(roots_true, r_min, r_max)
    root_hat_obs = _first_observed_crossing(roots_hat, r_min, r_max)

    print("\nSign crossing diagnostics on observed distance range")
    print(f"observed range used = [{_fmt(r_min)}, {_fmt(r_max)}]")
    print(f"true roots in grid = {[_fmt(x) for x in roots_true]}")
    print(f"estimated roots in grid = {[_fmt(x) for x in roots_hat]}")
    print(f"first true root in observed range = {_fmt(root_true_obs)}")
    print(f"first estimated root in observed range = {_fmt(root_hat_obs)}")

    print("\nphi(r) at representative observed distances")
    print("r        phi_true      phi_hat")
    for label in ("q05", "q10", "q25", "q50", "q75", "q90", "q95"):
        rv = float(qs[label])
        tv = float(meta["phi_true"](np.asarray([rv]))[0])
        hv = float((phi_basis(np.asarray([rv])) @ theta_hat).reshape(-1)[0])
        print(f"{label:<4} {_fmt(rv, 5):>10} {_fmt(tv, 5):>12} {_fmt(hv, 5):>12}")

    print("\nFit diagnostics")
    print(f"kernel basis = {cfg['kernel_basis']['basis']}")
    print(f"nnz(theta_hat) = {int(np.sum(np.abs(theta_hat) > 1e-8))}")
    print(f"cond(A^T A) = {_fmt(diag.get('cond_ata'), 5)}")
    print(f"kernel_L2w_rel = {_fmt(diag.get('kernel_L2w_rel'), 5)}")
    if theta_true_basis is not None and theta_true_basis.shape == theta_hat.shape:
        print(f"theta_l2 = {_fmt(np.linalg.norm(theta_true_basis - theta_hat), 5)}")

    _, _, theta_labels, _ = _dictionary_axis(diag["kernel_meta"], theta_hat.size)
    print("\nEstimated coefficients")
    print("atom                         theta_true    theta_hat")
    for j, value in enumerate(theta_hat):
        label = theta_labels[j] if theta_labels is not None and j < len(theta_labels) else str(j)
        true_value = None
        if theta_true_basis is not None and theta_true_basis.shape == theta_hat.shape:
            true_value = float(theta_true_basis[j])
        true_text = "-" if true_value is None else _fmt(true_value, 5)
        print(f"{label[:28]:28s} {true_text:>12s} {_fmt(value, 5):>12s}")

    if plot_prefix:
        print("\nSaving diagnostic plots")
        _plot_diagnostics(
            plot_prefix,
            r_samples,
            qs,
            r_grid,
            phi_true,
            phi_hat,
            roots_true,
            roots_hat,
            theta_hat,
            theta_true_basis,
            diag["kernel_meta"],
            X,
            t,
            K_true,
            fit["K_hat"],
            meta["sigma"],
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Diagnose whether observed particle scales can resolve attraction-repulsion."
    )
    parser.add_argument("--input", "-i", default="configs/config.json", help="Config JSON path")
    args = parser.parse_args()

    cfg = build_config(merge_config({}, _load_config(args.input)))
    try:
        run_diagnostic(cfg)
    except Exception as exc:
        print(f"diagnostic failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
