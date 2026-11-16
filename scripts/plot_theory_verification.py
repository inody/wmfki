"""
Generate a 3-panel theory verification figure for the paper.

Panels:
  (a) dt-sweep: Euler O(dt) vs RK4 convergence
  (b) sigma-sweep: E[l2^2] vs sigma (O(sigma^2) slope)
  (c) N-sweep:    E[l2^2] vs N     (O(1/N) slope)

Run from project root:
    uv run python scripts/plot_theory_verification.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "outputs", "qanr")
os.makedirs(OUT, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C1 = "#2166ac"   # blue – Euler / main data
C2 = "#d6604d"   # red  – RK4
CREF = "#888888"  # grey – reference slope

# ── Panel (a): dt-sweep ────────────────────────────────────────────────────────
import csv

def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader]

euler_rows = read_csv(os.path.join(ROOT, "outputs/qanr/theory_dt_euler_fixed/results.csv"))
rk4_rows   = read_csv(os.path.join(ROOT, "outputs/qanr/theory_dt_sigma6/results.csv"))

euler_dt  = np.array([float(r["dt"])       for r in euler_rows])
euler_l2  = np.array([float(r["theta_l2"]) for r in euler_rows])
rk4_dt    = np.array([float(r["dt"])       for r in rk4_rows])
rk4_l2    = np.array([float(r["theta_l2"]) for r in rk4_rows])

# Sort by dt
idx = np.argsort(euler_dt); euler_dt = euler_dt[idx]; euler_l2 = euler_l2[idx]
idx = np.argsort(rk4_dt);   rk4_dt   = rk4_dt[idx];   rk4_l2   = rk4_l2[idx]

# ── Panel (b): sigma-sweep (multi-seed data stored as constants) ───────────────
sigma_vals = np.array([0.01, 0.05, 0.10, 0.20, 0.40])
El2sq      = np.array([71.608, 64.040, 67.847, 117.638, 370.817]) * (sigma_vals**2) / 100.0
# C_eff = E[l2^2]*N/sigma^2, so E[l2^2] = C_eff * sigma^2 / N, N=100
# Restored: E[l2^2] = [8.46e-3^2, 4.00e-2^2, 8.24e-2^2, 2.17e-1^2, 7.70e-1^2]
El2sq = np.array([8.46e-3, 4.00e-2, 8.24e-2, 2.17e-1, 7.70e-1])**2

# ── Panel (c): N-sweep (multi-seed data) ───────────────────────────────────────
N_vals    = np.array([50,  100,   200,   400])
El2sq_N   = np.array([1.491e-01, 8.182e-02, 6.355e-02, 4.619e-02])**2

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plt.subplots_adjust(wspace=0.38)

# ── (a) dt-sweep ───────────────────────────────────────────────────────────────
ax = axes[0]
ax.loglog(euler_dt, euler_l2, "o-", color=C1, label="Euler–Maruyama", linewidth=1.5, markersize=5)
ax.loglog(rk4_dt,   rk4_l2,   "s-", color=C2, label="RK4 (σ=0)",     linewidth=1.5, markersize=5)

# O(dt^1) reference anchored at dt=0.05
dt_ref = np.array([euler_dt.min(), euler_dt.max()])
c1 = euler_l2[np.argmin(np.abs(euler_dt - 0.05))]
ax.loglog(dt_ref, c1 * (0.05 / dt_ref)**(-1), "--", color=CREF, alpha=0.7,
          linewidth=1.2, label=r"$O(\Delta t)$")

ax.set_xlabel(r"$\Delta t$", fontsize=12)
ax.set_ylabel(r"$\|\hat\theta - \theta^*\|_2$", fontsize=12)
ax.set_title(r"(a) $\Delta t$-convergence", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.tick_params(labelsize=9)

# ── (b) sigma-sweep ────────────────────────────────────────────────────────────
ax = axes[1]
ax.loglog(sigma_vals, np.sqrt(El2sq), "o-", color=C1, linewidth=1.5, markersize=5,
          label=r"$\sqrt{E[\|\hat\theta-\theta^*\|^2]}$")

# O(sigma) reference anchored at sigma=0.05
c2 = np.sqrt(El2sq[1])
sig_ref = np.array([sigma_vals[0], sigma_vals[2]])
ax.loglog(sig_ref, c2 * (sigma_vals[1] / sig_ref)**(-1), "--", color=CREF, alpha=0.7,
          linewidth=1.2, label=r"$O(\sigma)$")

ax.set_xlabel(r"$\sigma$", fontsize=12)
ax.set_ylabel(r"$\sqrt{E[\|\hat\theta - \theta^*\|^2]}$", fontsize=12)
ax.set_title(r"(b) $\sigma$-scaling  ($N=100$)", fontsize=11)
ax.legend(fontsize=8)
ax.tick_params(labelsize=9)

# ── (c) N-sweep ────────────────────────────────────────────────────────────────
ax = axes[2]
ax.loglog(N_vals, np.sqrt(El2sq_N), "o-", color=C1, linewidth=1.5, markersize=5,
          label=r"$\sqrt{E[\|\hat\theta-\theta^*\|^2]}$")

# O(1/sqrt(N)) reference anchored at N=100
c3 = np.sqrt(El2sq_N[1])
N_ref = np.array([N_vals[0], N_vals[-1]], dtype=float)
ax.loglog(N_ref, c3 * (100.0 / N_ref)**0.5, "--", color=CREF, alpha=0.7,
          linewidth=1.2, label=r"$O(N^{-1/2})$")

ax.set_xlabel(r"$N$", fontsize=12)
ax.set_ylabel(r"$\sqrt{E[\|\hat\theta - \theta^*\|^2]}$", fontsize=12)
ax.set_title(r"(c) $N$-scaling  ($\sigma=0.1$)", fontsize=11)
ax.legend(fontsize=8)
ax.tick_params(labelsize=9)

fig.suptitle(
    "Theory verification: QANR problem, " +
    r"$\|\hat\theta - \theta^*\|^2 \lesssim C\,(\sigma^2/N + \Delta t^{2p})$",
    fontsize=11, y=1.02
)

out_path = os.path.join(OUT, "theory_all_sweeps_v2.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# Also copy to latex figures
latex_fig = os.path.join(ROOT, "latex", "R02", "figures", "theory_verification.png")
import shutil
shutil.copy(out_path, latex_fig)
print(f"Copied to: {latex_fig}")
