"""
Reproduce all figures in latex/R04/main.tex and slidev/slides.md.

This legacy script is not part of the current L4DC figure pipeline. See
``scripts/regenerate_l4dc_figures.py`` for the active one-command entry point.

Usage (from project root):
    uv run python scripts/generate_paper_figures.py

Outputs
-------
latex/R04/figures/*.png   paper-quality figures
slidev/public/*.png       same files copied for the slide deck

Prerequisites
-------------
- uv (dependency manager)
- LaTeX distribution with lualatex (for text.usetex=True)
  macOS:  brew install --cask mactex
  Linux:  apt-get install texlive-full

Expected runtime: ~5 min (dominated by d=10 Morse simulation).
"""

import sys, os, csv, shutil, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIGDIR   = os.path.join(ROOT, "latex", "R04", "figures")
SLIDEDIR = os.path.join(ROOT, "slidev", "public")
OUTDIR   = os.path.join(ROOT, "outputs")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex":    True,
    "font.family":    "serif",
    "font.size":      11,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# Paired subfigures must share identical (W, H) so LaTeX renders them at equal height.
W_FULL = 9.0    # full-width (1.0\linewidth)
W_SUB  = 4.5    # subfigure pair (0.48\linewidth each)
H_STD  = 3.5    # height for ALL subfigure plots
H_LOG  = 5.5    # log2d 2×3 snapshot grid

C_TRUE = "#2166ac"   # blue – true kernel / Euler
C_HAT  = "#d6604d"   # red  – estimated / RK4
C_REF  = "#888888"   # grey – reference slope

EPS_Q = 1e-8   # QANR regularisation
EPS_L = 1e-2   # Log2D regularisation

from simulate import simulate_particles, simulate_with_kernel
from kernels  import make_radial_basis
from pipeline import fit_kernel_from_particles


def savefig(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    shutil.copy(path, os.path.join(SLIDEDIR, name))
    plt.close(fig)
    print(f"  {name}")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# §4.1  QANR  (d=1, power-law dict)
# ─────────────────────────────────────────────────────────────────────────────
print("§4.1 QANR")

POWERS = [1, 2, 3, 4, 5, 6, 7]
kp_q   = {"mode": "qanr", "eps": EPS_Q}

X_q, t_q, _, _, _ = simulate_particles(
    N=500, T=600, dt=0.01, R=1, sigma=0.1, d=1,
    init="gaussian_mixture", init_centers=np.array([0.5, 2.0, 3.5]),
    init_std=5e-3, seed=42, kernel_params=kp_q,
)

cfg_q = {
    "task": "fit",
    "simulation": {
        "N": 500, "T": 600, "dt": 0.01, "R": 1, "sigma": 0.1, "d": 1,
        "init": "gaussian_mixture", "init_scale": 5e-3,
        "init_centers": [0.5, 2.0, 3.5], "seed": 42,
        "solver": "euler", "interaction": kp_q,
    },
    "time_tests":    {"type": "gaussian", "K": 8, "s_factor": 1.8, "sigma_div": 6},
    "pair_sampling": {"total_pairs": 50000, "with_replacement": True, "seed": 42},
    "space_tests":   {"type": "gaussian", "L": 25, "s_factor": 1.5,
                      "layout": "random", "seed": 42, "quantiles": [0.01, 0.99]},
    "kernel_basis":  {"basis": "grad_abs_power", "powers": POWERS},
    "regression":    {"method": "lasso", "standardize": True, "alpha": 7.071067811865476e-6},
    "output":        {"prefix": "/tmp/qanr_paperfig", "save_plots": False},
}
res_q = fit_kernel_from_particles(cfg_q, X_q, t_q)
th_q  = np.array(res_q["theta_hat"])
print(f"  nnz={int(np.sum(np.abs(th_q)>1e-8))}")

def phi_m_q(r, p):
    if np.isclose(p, 1.0):  return 1.0 / (r + EPS_Q)
    elif np.isclose(p, 2.0): return np.ones_like(r)
    else:                    return r ** (p - 2.0)

def K_hat_q(z):
    r  = np.linalg.norm(z, axis=-1, keepdims=True)
    rs = r.squeeze(-1)
    return z * sum(th_q[m] * phi_m_q(rs, POWERS[m]) for m in range(len(POWERS)))[..., None]

noise_q = np.random.default_rng(43).standard_normal(size=(1, 600, 500, 1))
X_qhat  = simulate_with_kernel(K_hat_q, X_q[0:1, 0], t_q, sigma=0.1, noise=noise_q)

# Trajectory
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, H_STD), sharey=True)
for i in range(0, 500, 10):
    axes[0].plot(t_q, X_q[0, :, i, 0],    lw=0.3, alpha=0.35, color=C_TRUE)
    axes[1].plot(t_q, X_qhat[0, :, i, 0], lw=0.3, alpha=0.35, color=C_HAT)
for ax, (color, lbl) in zip(axes, [
    (C_TRUE, r"True $K^*$"),
    (C_HAT,  r"Estimated $\hat{K}$"),
]):
    ax.set_xlabel(r"$t$")
    ax.legend(handles=[plt.Line2D([],[],color=color,lw=1.5,label=lbl)],
              loc="upper right", framealpha=0.7)
axes[0].set_ylabel(r"$X_i(t)$")
fig.tight_layout()
savefig(fig, "qanr_traj.png")

# Kernel profile
r_q          = np.linspace(0.05, 6.0, 500)
phi_true_q_v = 1.0 / (r_q + EPS_Q) - 1.0
phi_hat_q_v  = sum(th_q[m] * phi_m_q(r_q, POWERS[m]) for m in range(len(POWERS)))

fig, ax = plt.subplots(figsize=(W_SUB, H_STD))
ax.plot(r_q, phi_true_q_v, color=C_TRUE, lw=2, label=r"True $\phi(r)$")
ax.plot(r_q, phi_hat_q_v,  color=C_HAT,  lw=2, ls="--", label=r"LASSO $\hat\phi(r)$")
ax.axhline(0, color=C_REF, lw=0.8, ls=":")
ax.set_xlim(0.05, 6); ax.set_ylim(-2, 15)
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$\phi(r)$")
ax.legend(); fig.tight_layout()
savefig(fig, "qanr_phi.png")

# Dict coefficients
th_true_q = np.array([1.0, -1.0, 0, 0, 0, 0, 0])
LABELS_Q  = [r"$r^{-1}$", r"$1$", r"$r$", r"$r^2$", r"$r^3$", r"$r^4$", r"$r^5$"]
x_pos     = np.arange(len(POWERS))

fig, ax = plt.subplots(figsize=(W_SUB, H_STD))
ax.bar(x_pos-0.2, th_true_q, width=0.35, color=C_TRUE, alpha=0.75,
       label=r"True $\theta^*$", edgecolor="k", lw=0.5)
ax.bar(x_pos+0.2, th_q,       width=0.35, color=C_HAT,  alpha=0.75,
       label=r"LASSO $\hat\theta$", edgecolor="k", lw=0.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(LABELS_Q)
ax.set_xlabel(r"Dictionary atom $\phi_m(r)$")
ax.set_ylabel(r"Coefficient $\theta_m$")
ax.legend(); fig.tight_layout()
savefig(fig, "qanr_theta.png")


# ─────────────────────────────────────────────────────────────────────────────
# §4.1  Theory verification  (reads pre-computed CSV + hardcoded multi-seed data)
# ─────────────────────────────────────────────────────────────────────────────
print("§4.1 Theory verification")

euler_rows = read_csv(os.path.join(OUTDIR, "qanr/theory_dt_euler_fixed/results.csv"))
rk4_rows   = read_csv(os.path.join(OUTDIR, "qanr/theory_dt_sigma6/results.csv"))
euler_dt = np.array([float(r["dt"])       for r in euler_rows])
euler_l2 = np.array([float(r["theta_l2"]) for r in euler_rows])
rk4_dt   = np.array([float(r["dt"])       for r in rk4_rows])
rk4_l2   = np.array([float(r["theta_l2"]) for r in rk4_rows])
euler_dt, euler_l2 = [np.array(v) for v in zip(*sorted(zip(euler_dt, euler_l2)))]
rk4_dt,   rk4_l2  = [np.array(v) for v in zip(*sorted(zip(rk4_dt,   rk4_l2)))]

# Multi-seed sweep data (hardcoded from prior sweep experiments)
sigma_vals = np.array([0.01, 0.05, 0.10, 0.20, 0.40])
El2sq      = np.array([8.46e-3, 4.00e-2, 8.24e-2, 2.17e-1, 7.70e-1])**2

N_vals  = np.array([50, 100, 200, 400])
El2sq_N = np.array([1.491e-01, 8.182e-02, 6.355e-02, 4.619e-02])**2

# theory_dt.png  (0.48\linewidth, same size as theory_sigma / theory_N)
fig, ax = plt.subplots(figsize=(W_SUB, H_STD))
ax.loglog(euler_dt, euler_l2, "o-", color=C_TRUE, lw=1.8, ms=5,
          label=r"Euler--Maruyama")
ax.loglog(rk4_dt,   rk4_l2,   "s-", color=C_HAT,  lw=1.8, ms=5,
          label=r"RK4 ($\sigma=0$)")
dt_ref = np.array([euler_dt.min(), euler_dt.max()])
c_ref  = euler_l2[-1] * 3.0   # 3× offset so line is visible above the data
ax.loglog(dt_ref, c_ref * (dt_ref / euler_dt[-1]), "--", color=C_REF, lw=1.4,
          label=r"$O(\Delta t)$")
ax.set_xlabel(r"$\Delta t$"); ax.set_ylabel(r"$\|\hat\theta - \theta^*\|_2$")
ax.legend(); fig.tight_layout()
savefig(fig, "theory_dt.png")

# theory_sigma.png  (0.48\linewidth subfigure)
fig, ax = plt.subplots(figsize=(W_SUB, H_STD))
ax.loglog(sigma_vals, np.sqrt(El2sq), "o-", color=C_TRUE, lw=1.8, ms=5)
sig_ref = np.array([sigma_vals[0], sigma_vals[-1]])
c2      = np.sqrt(El2sq[1])
ax.loglog(sig_ref, c2 * (sigma_vals[1] / sig_ref)**(-1), "--", color=C_REF, lw=1.2,
          label=r"$O(\sigma)$")
ax.set_xlabel(r"$\sigma$"); ax.set_ylabel(r"$\sqrt{E[\|\hat\theta-\theta^*\|^2]}$")
ax.legend(); fig.tight_layout()
savefig(fig, "theory_sigma.png")

# theory_N.png  (0.48\linewidth subfigure)
fig, ax = plt.subplots(figsize=(W_SUB, H_STD))
ax.loglog(N_vals, np.sqrt(El2sq_N), "o-", color=C_TRUE, lw=1.8, ms=5)
N_ref = np.array([N_vals[0], N_vals[-1]], dtype=float)
c3    = np.sqrt(El2sq_N[1])
ax.loglog(N_ref, c3 * (100.0 / N_ref)**0.5, "--", color=C_REF, lw=1.2,
          label=r"$O(N^{-1/2})$")
ax.set_xlabel(r"$N$"); ax.set_ylabel(r"$\sqrt{E[\|\hat\theta-\theta^*\|^2]}$")
ax.legend(); fig.tight_layout()
savefig(fig, "theory_N.png")


# ─────────────────────────────────────────────────────────────────────────────
# §4.2  Log-attraction / Keller-Segel  (d=2, Messenger-Bortz dict)
# ─────────────────────────────────────────────────────────────────────────────
print("§4.2 Log-attraction (d=2)")

cfg_l = read_json(os.path.join(ROOT, "configs", "experiments", "log2d_matched.json"))
sim_l = cfg_l["simulation"]
kp_l = sim_l["interaction"]
X_l, t_l, _, _, _ = simulate_particles(
    N=sim_l["N"], T=sim_l["T"], dt=sim_l["dt"], R=sim_l["R"],
    sigma=sim_l["sigma"], d=sim_l["d"], init=sim_l["init"],
    init_scale=sim_l["init_scale"], seed=sim_l["seed"], kernel_params=kp_l,
)
res_l = fit_kernel_from_particles(
    cfg_l, X_l, t_l, pair_mode="resample_each_time",
    pair_seed=cfg_l["pair_sampling"]["seed"],
)
th_l_full = np.asarray(res_l["theta_hat"])
print(f"  theta={np.array2string(th_l_full, precision=4)}")
phi_fn_l, _ = make_radial_basis({"basis": "messenger_log2d", "eps": EPS_L, "powers": []})
K_hat_l = res_l["K_hat"]

noise_l = np.random.default_rng(43).standard_normal(
    size=(1, sim_l["T"], sim_l["N"], sim_l["d"])
)
X_lhat = simulate_with_kernel(
    K_hat_l, X_l[0:1, 0], t_l, sigma=sim_l["sigma"], noise=noise_l
)

# Trajectory snapshots 2×3
times_idx = [0, 40, 80]
time_labs  = [r"$t=0$", r"$t=4$", r"$t=8$"]
fig, axes = plt.subplots(2, 3, figsize=(W_FULL, H_LOG))
for col, (ti, tl) in enumerate(zip(times_idx, time_labs)):
    for row, (Xr, color, ylbl) in enumerate([
        (X_l,    C_TRUE, r"True $K^*$"     + "\n" + r"$x_2$"),
        (X_lhat, C_HAT,  r"LASSO $\hat{K}$" + "\n" + r"$x_2$"),
    ]):
        ax   = axes[row, col]
        snap = Xr[0, ti]
        lim  = max(np.abs(X_l[0]).max() * 1.15, 0.5)
        ax.scatter(snap[:, 0], snap[:, 1], s=1.5, alpha=0.3, color=color, rasterized=True)
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        if row == 0: ax.set_title(tl)
        if col == 0: ax.set_ylabel(ylbl)
        if row == 1: ax.set_xlabel(r"$x_1$")
fig.tight_layout()
savefig(fig, "log2d_traj.png")

# Kernel profile
r_l       = np.linspace(0.02, 3.0, 500)
phi_true_l = -1.0 / (2 * np.pi * (r_l**2 + EPS_L**2))
phi_hat_l  = phi_fn_l(r_l) @ th_l_full

fig, ax = plt.subplots(figsize=(W_SUB, H_STD))
ax.plot(r_l, phi_true_l, color=C_TRUE, lw=2, label=r"True $\phi(r)$")
ax.plot(r_l, phi_hat_l,  color=C_HAT,  lw=2, ls="--", label=r"LASSO $\hat\phi(r)$")
ax.axhline(0, color=C_REF, lw=0.8, ls=":")
ax.set_xlim(0.02, 3); ax.set_ylim(-20, 2)
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$\phi(r)$")
ax.legend(); fig.tight_layout()
savefig(fig, "log2d_phi.png")

# Dict coefficients (explicit labels — no [...]_δ shorthand)
LABELS_L = [
    r"$r^2$", r"$r^3$", r"$r^4$", r"$r^5$", r"$r^6$",
    r"$(r^2{+}\delta^2)^{-3/4}$",
    r"$\frac{\log(r^2{+}\delta^2)}{2\sqrt{r^2{+}\delta^2}}$",
    r"$\frac{1}{r^2+\delta^2}$",
]
th_true_l = np.zeros(8); th_true_l[7] = -1 / (2 * np.pi)
x8 = np.arange(8)

fig, ax = plt.subplots(figsize=(W_SUB, H_STD))
ax.bar(x8-0.2, th_true_l, width=0.35, color=C_TRUE, alpha=0.75,
       label=r"True $\theta^*$", edgecolor="k", lw=0.5)
ax.bar(x8+0.2, th_l_full, width=0.35, color=C_HAT,  alpha=0.75,
       label=r"LASSO $\hat\theta$", edgecolor="k", lw=0.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x8)
ax.set_xticklabels(LABELS_L, rotation=45, ha="right")
ax.set_xlabel(r"Dictionary atom $\phi_m(r)$")
ax.set_ylabel(r"Coefficient $\theta_m$")
ax.legend(); fig.tight_layout()
savefig(fig, "log2d_theta.png")


# ─────────────────────────────────────────────────────────────────────────────
# §4.3  Morse potential  (d=10, exp_r dict)
#
# The trajectory and pair-sampling seeds are fixed by the manuscript config.
# ─────────────────────────────────────────────────────────────────────────────
print("§4.3 Morse (d=10)")

cfg_m = read_json(os.path.join(ROOT, "configs", "experiments", "morse_d10_manuscript.json"))
sim_m = cfg_m["simulation"]
kp_m = sim_m["interaction"]
ells = np.asarray(cfg_m["kernel_basis"]["ell_grid"], dtype=float)
X_m, t_m, _, _, _ = simulate_particles(
    N=sim_m["N"], T=sim_m["T"], dt=sim_m["dt"], R=sim_m["R"],
    sigma=sim_m["sigma"], d=sim_m["d"], init=sim_m["init"],
    init_scale=sim_m["init_scale"], seed=sim_m["seed"], kernel_params=kp_m,
)
res_m = fit_kernel_from_particles(
    cfg_m, X_m, t_m, pair_mode="resample_each_time",
    pair_seed=cfg_m["pair_sampling"]["seed"],
)
th_m = np.asarray(res_m["theta_hat"])
nnz_m = int(np.sum(np.abs(th_m) > 1e-8))
print(f"  theta={np.array2string(th_m, precision=4)}")

phi_fn_m, _ = make_radial_basis({"basis": "exp_r", "ell_grid": list(ells), "M": len(ells)})
r_m       = np.linspace(0.05, 5.0, 400)
# phi_true: phi(r) = r*(2*exp(-r/l1) - exp(-r/l2))  [from morse_r definition]
phi_true_m = r_m * (2.0 * np.exp(-r_m / 1.0) - 1.0 * np.exp(-r_m / 2.0))
phi_hat_m  = phi_fn_m(r_m) @ th_m

# Kernel profile
fig, ax = plt.subplots(figsize=(W_SUB, H_STD))
ax.plot(r_m, phi_true_m, color=C_TRUE, lw=2, label=r"True $\phi(r)$")
ax.plot(r_m, phi_hat_m,  color=C_HAT,  lw=2, ls="--", label=r"LASSO $\hat\phi(r)$")
ax.axhline(0, color=C_REF, lw=0.8, ls=":")
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$\phi(r)$")
ax.legend(); fig.tight_layout()
savefig(fig, "morse_phi.png")

# Dict coefficients (paired bar chart, same style as qanr_theta / log2d_theta)
# True theta*: ell=1.0 (+2) and ell=2.0 (-1) mapped to nearest grid atoms
#   ell=1.0 → index 2 (ell=1.032), ell=2.0 → index 4 (ell=1.774)
LABELS_M = [f"${e:.3f}$" for e in ells]
x8m = np.arange(len(ells))
th_true_m = np.zeros(8)
th_true_m[2] = +2.0   # nearest to ell=1.0
th_true_m[4] = -1.0   # nearest to ell=2.0

fig, ax = plt.subplots(figsize=(W_SUB, H_STD))
ax.bar(x8m - 0.2, th_true_m, width=0.35, color=C_TRUE, alpha=0.75,
       label=r"True $\theta^*$", edgecolor="k", lw=0.5)
ax.bar(x8m + 0.2, th_m,      width=0.35, color=C_HAT,  alpha=0.75,
       label=r"LASSO $\hat\theta$", edgecolor="k", lw=0.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x8m); ax.set_xticklabels(LABELS_M, rotation=45, ha="right")
ax.set_xlabel(r"Scale $\ell_m$")
ax.set_ylabel(r"Coefficient $\theta_m$")
ax.legend(); fig.tight_layout()
savefig(fig, "morse_theta.png")

# Simulate with estimated kernel
K_hat_m = res_m["K_hat"]
noise_m = np.random.default_rng(43).standard_normal(
    size=(1, sim_m["T"], sim_m["N"], sim_m["d"])
)
X_mhat = simulate_with_kernel(
    K_hat_m, X_m[0:1, 0], t_m, sigma=sim_m["sigma"], noise=noise_m
)

# Trajectory norms
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, H_STD), sharey=True)
for ax, Xr, color, title in [
    (axes[0], X_m,    C_TRUE, r"True $K^*$"),
    (axes[1], X_mhat, C_HAT, r"LASSO $\hat{K}$"),
]:
    norms = np.linalg.norm(Xr[0], axis=-1)
    mn = norms.mean(1); sn = norms.std(1)
    ax.plot(t_m, mn, color=color, lw=2, label=r"mean $\|X_i\|$")
    ax.fill_between(t_m, mn - sn, mn + sn, alpha=0.15, color=color)
    for i in range(0, 30, 3):
        ax.plot(t_m, norms[:, i], lw=0.3, alpha=0.2, color=color)
    ax.set_xlabel(r"$t$")
    ax.legend(handles=[
        plt.Line2D([], [], color=color, lw=1.8, label=title),
    ], fontsize=9)
axes[0].set_ylabel(r"$\|X_i(t)\|$")
fig.tight_layout()
savefig(fig, "morse_traj.png")

print("\nAll figures saved.")
print(f"  Paper: {FIGDIR}")
print(f"  Slides: {SLIDEDIR}")
