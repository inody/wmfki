import json
import os
import sys

import matplotlib
import numpy as np

from config import build_config, merge_config
from kernels import make_radial_basis
from metrics import drift_mse
from pipeline import fit_kernel_from_particles
from simulate import simulate_particles, simulate_with_kernel

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def run_fit(cfg):
    sim_cfg = cfg["simulation"]

    print("-" * 40)
    print("PARAMETERS:")
    print(
        f"  Simulation: N={sim_cfg['N']}, T={sim_cfg['T']}, dt={sim_cfg['dt']}, R={sim_cfg['R']}, d={sim_cfg.get('d', 1)}"
    )
    print(f"  Physics:    sigma={sim_cfg['sigma']}, Interaction={sim_cfg['interaction']['mode']}")
    print(f"  Regression: {cfg['regression']['method'].upper()}, alpha={cfg['regression']['alpha']}")
    print(f"  Basis:      M={cfg['kernel_basis']['M']}, {cfg['kernel_basis']['basis']}")
    print("-" * 40)

    print("Simulating particles...")
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
        kernel_params=sim_cfg["interaction"],
        seed=sim_cfg["seed"],
        external_drift_fn=sim_cfg.get("external_drift_fn"),
        solver=sim_cfg.get("solver", "euler"),
    )

    if cfg["kernel_basis"]["ell_grid"] is None:
        l1 = meta["kernel_meta"].get("l1", 0.5)
        l2 = meta["kernel_meta"].get("l2", 1.5)
        ell_min = 0.2 * min(l1, l2)
        ell_max = 5.0 * max(l1, l2)
        M_dict = cfg["kernel_basis"]["M"]
        base_grid = np.logspace(np.log10(ell_min), np.log10(ell_max), M_dict)
        ells = np.unique(np.sort(np.concatenate([base_grid, [l1, l2]])))
        if ells.size > M_dict + 5:
            idx = np.linspace(0, ells.size - 1, M_dict).round().astype(int)
            ells = ells[idx]
        cfg["kernel_basis"]["ell_grid"] = ells.tolist()
        cfg["kernel_basis"]["M"] = len(ells)

    cfg["diagnostics"]["true_kernel"] = {"phi_true_fn": meta["phi_true"]}

    alpha = cfg["regression"]["alpha"]
    method = str(cfg["regression"].get("method", "lasso")).upper()
    print(f"Running {method} with alpha={alpha}...")

    fit_lasso = fit_kernel_from_particles(cfg, X, t)

    d_mse = drift_mse(K_true, fit_lasso["K_hat"], X)
    l2w_rel = fit_lasso["diagnostics"].get("kernel_L2w_rel", 1.0)
    nnz = int(np.sum(np.abs(fit_lasso["theta_hat"]) > 1e-8))

    theta_err = {"l2": None, "l1": None, "linf": None}
    theta_true_for_err = None
    if theta_true is not None:
        theta_true_for_err = np.asarray(theta_true, dtype=float)
    else:
        if (
            sim_cfg.get("interaction", {}).get("mode") == "morse_r"
            and cfg["kernel_basis"].get("basis") == "exp_r"
            and cfg["kernel_basis"].get("ell_grid") is not None
        ):
            ells = np.asarray(cfg["kernel_basis"]["ell_grid"], dtype=float)
            l1 = float(sim_cfg["interaction"].get("l1", np.nan))
            l2 = float(sim_cfg["interaction"].get("l2", np.nan))
            a = float(sim_cfg["interaction"].get("a", np.nan))
            b = float(sim_cfg["interaction"].get("b", np.nan))
            idx1 = np.where(np.isclose(ells, l1, rtol=0.0, atol=1e-12))[0]
            idx2 = np.where(np.isclose(ells, l2, rtol=0.0, atol=1e-12))[0]
            if idx1.size == 1 and idx2.size == 1:
                theta_true_for_err = np.zeros_like(fit_lasso["theta_hat"], dtype=float)
                theta_true_for_err[idx1[0]] = a
                theta_true_for_err[idx2[0]] = -b

    if theta_true_for_err is not None:
        tt = np.asarray(theta_true_for_err, dtype=float)
        th = np.asarray(fit_lasso["theta_hat"], dtype=float)
        if tt.shape == th.shape:
            diff = tt - th
            theta_err = {
                "l2": float(np.linalg.norm(diff, ord=2)),
                "l1": float(np.linalg.norm(diff, ord=1)),
                "linf": float(np.linalg.norm(diff, ord=np.inf)),
            }

    print(
        f"Result: Alpha={alpha:.1e}, Terms={nnz}, RelError={l2w_rel:.1%}, DriftMSE={d_mse:.3e}"
    )
    if theta_err["l2"] is not None:
        print(
            "ThetaErr:"
            f" l2={theta_err['l2']:.3e},"
            f" l1={theta_err['l1']:.3e},"
            f" linf={theta_err['linf']:.3e}"
        )

    out_cfg = cfg.get("output", {})
    save_plots = out_cfg.get("save_plots", False)
    prefix = out_cfg.get("prefix", "output")

    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    def _to_jsonable(obj):
        if callable(obj):
            name = getattr(obj, "__name__", "callable")
            mod = getattr(obj, "__module__", None)
            return f"<callable:{mod}.{name}>" if mod else f"<callable:{name}>"
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return f"<non-serializable:{type(obj).__name__}>"

    log_payload = {
        "config": cfg,
        "metrics": {
            "alpha": alpha,
            "rel_error": float(l2w_rel),
            "drift_mse": float(d_mse),
            "nnz": int(nnz),
            "theta_err": theta_err,
        },
        "theta_hat": fit_lasso.get("theta_hat"),
        "diagnostics": fit_lasso.get("diagnostics", {}),
        "model_info": fit_lasso.get("model_info", {}),
        "timestamp": None,
    }
    log_path = f"{prefix}_log.json"
    with open(log_path, "w") as f:
        json.dump(_to_jsonable(log_payload), f, indent=2)
    print(f"Saved {log_path}")

    for name, fit in [("lasso", fit_lasso)]:
        km = fit["diagnostics"]["kernel_meta"]
        if "ells" in km:
            r_max_plot = np.max(km["ells"]) * 2.5
        else:
            r_max_plot = 5.0

        r_grid = np.linspace(0.0, r_max_plot, 200)
        phi_true = meta["phi_true"](r_grid)

        kb_plot = cfg["kernel_basis"]
        phi_fn, _ = make_radial_basis(
            {
                "M": kb_plot["M"],
                "basis": kb_plot["basis"],
                "ell_grid": kb_plot.get("ell_grid"),
                "powers": kb_plot.get("powers"),
                "p_min": kb_plot.get("p_min"),
                "p_max": kb_plot.get("p_max"),
                "eps": kb_plot.get("eps", 1e-3),
            }
        )
        phi_hat = phi_fn(r_grid) @ fit["theta_hat"]

        d_mse = drift_mse(K_true, fit["K_hat"], X)
        l2w_rel = fit["diagnostics"].get("kernel_L2w_rel", None)
        nnz = int(np.sum(np.abs(fit["theta_hat"]) > 1e-8))

        print(f"Final {name}: drift_mse={d_mse:.4e}  RelError={l2w_rel:.1%}  Terms={nnz}")

        out_cfg = cfg.get("output", {})
        save_plots = out_cfg.get("save_plots", False)
        prefix = out_cfg.get("prefix", "output")

        pdir = os.path.dirname(prefix)
        if pdir:
            os.makedirs(pdir, exist_ok=True)

        if save_plots:
            try:
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                plt.figure(figsize=(6, 4))
                plt.plot(r_grid, phi_true, label="phi_true")
                plt.plot(r_grid, phi_hat, label=f"phi_hat ({name})", linestyle="--")
                plt.xlabel("r")
                plt.ylabel("phi(r)")
                plt.title(f"{name}: RelError={l2w_rel:.1%}, Terms={nnz}")
                plt.legend()
                plt.tight_layout()
                filename = f"{prefix}_{name}_phi.png"
                plt.savefig(filename, dpi=150)
                print(f"Saved {filename}")
                plt.close()

                theta_true_vis = None
                if theta_true is not None:
                    tt = np.asarray(theta_true, dtype=float)
                    if tt.shape[0] == fit["theta_hat"].shape[0]:
                        theta_true_vis = tt
                else:
                    if (
                        sim_cfg.get("interaction", {}).get("mode") == "morse_r"
                        and kb_plot.get("basis") == "exp_r"
                        and kb_plot.get("ell_grid") is not None
                    ):
                        ells = np.asarray(kb_plot["ell_grid"], dtype=float)
                        l1 = float(sim_cfg["interaction"].get("l1", np.nan))
                        l2 = float(sim_cfg["interaction"].get("l2", np.nan))
                        a = float(sim_cfg["interaction"].get("a", np.nan))
                        b = float(sim_cfg["interaction"].get("b", np.nan))
                        idx1 = np.where(np.isclose(ells, l1, rtol=0.0, atol=1e-12))[0]
                        idx2 = np.where(np.isclose(ells, l2, rtol=0.0, atol=1e-12))[0]
                        if idx1.size == 1 and idx2.size == 1:
                            theta_true_vis = np.zeros_like(fit["theta_hat"])
                            theta_true_vis[idx1[0]] = a
                            theta_true_vis[idx2[0]] = -b
                        else:
                            print(
                                "Note: true theta not shown (ell_grid does not include l1/l2 exactly)."
                            )

                km = fit["diagnostics"]["kernel_meta"]
                if "ells" in km:
                    x = np.asarray(km["ells"], dtype=float)
                    x_label = "ell"
                    use_logx = True
                elif "powers" in km:
                    x = np.asarray(km["powers"], dtype=float)
                    x_label = "power"
                    use_logx = False
                else:
                    x = np.arange(len(fit["theta_hat"]))
                    x_label = "index"
                    use_logx = False

                plt.figure(figsize=(7, 4))
                plt.hlines(0.0, x.min(), x.max(), colors="k", linewidth=0.5)
                if theta_true_vis is not None:
                    plt.vlines(x, 0.0, theta_true_vis, color="tab:blue", alpha=0.7, label="theta_true")
                    plt.scatter(x, theta_true_vis, color="tab:blue", s=20)
                plt.vlines(x, 0.0, fit["theta_hat"], color="tab:orange", alpha=0.7, label="theta_hat")
                plt.scatter(x, fit["theta_hat"], color="tab:orange", s=20)
                if use_logx:
                    plt.xscale("log")
                plt.xlabel(x_label)
                plt.ylabel("theta")
                plt.title(f"Theta Comparison ({name})")
                plt.legend()
                plt.tight_layout()
                filename_theta = f"{prefix}_{name}_theta.png"
                plt.savefig(filename_theta, dpi=150)
                print(f"Saved {filename_theta}")
                plt.close()

                R_sim, T1, N_sim, _ = X.shape
                T_sim = T1 - 1
                d_sim = X.shape[-1]
                noise = np.random.default_rng(2026).normal(0.0, 1.0, size=(1, T_sim, N_sim, d_sim))
                X0 = X[:1, 0, :, :]
                X_true_sim = simulate_with_kernel(
                    K_true, X0, t, sigma=meta["sigma"], noise=noise
                )
                X_hat_sim = simulate_with_kernel(fit["K_hat"], X0, t, sigma=meta["sigma"], noise=noise)

                r0 = 0
                n_plot = min(100, N_sim)
                idx = np.linspace(0, N_sim - 1, n_plot, dtype=int)
                tt = t

                plt.figure(figsize=(7, 4))
                comp = 0
                for i in idx:
                    plt.plot(
                        tt,
                        X_true_sim[r0, :, i, comp],
                        color="tab:blue",
                        alpha=0.5,
                        label="True" if i == idx[0] else None,
                    )
                for i in idx:
                    plt.plot(
                        tt,
                        X_hat_sim[r0, :, i, comp],
                        color="tab:orange",
                        alpha=0.5,
                        linestyle="--",
                        label="Est" if i == idx[0] else None,
                    )
                plt.xlabel("t")
                plt.ylabel("x(t) component 0")
                plt.title(f"Trajectory Comparison ({name})")
                plt.legend()
                plt.tight_layout()
                filename_traj = f"{prefix}_{name}_traj.png"
                plt.savefig(filename_traj, dpi=150)
                print(f"Saved {filename_traj}")
                plt.close()
            except Exception as e:
                print(f"Plot skipped: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run weak-form kernel identification.")
    parser.add_argument("--input", "-i", dest="config_path", default="configs/config.json")
    args = parser.parse_args()

    base_cfg = {}
    print(f"Loading config from {args.config_path}...")
    try:
        with open(args.config_path, "r") as f:
            json_cfg = json.load(f)
        base_cfg = merge_config(base_cfg, json_cfg)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    cfg = build_config(base_cfg)
    task = cfg.get("task", "fit")

    if task == "fit":
        run_fit(cfg)
    elif task == "diagnose":
        from diagnose_interaction_scale import run_diagnostic
        run_diagnostic(cfg)
    elif task == "sweep":
        from run_messenger_qanr_sweep import run_sweep
        run_sweep(cfg)
    else:
        print(f"Unknown task: {task!r}. Valid tasks: fit, diagnose, sweep", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
