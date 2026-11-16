from time import perf_counter

import numpy as np

from config import default_config, merge_config
from features import (
    compute_A_timeweak,
    compute_A_full_timeweak,
    compute_b_timeweak,
    make_trapz_weights,
)
from kernels import make_kernel_basis, make_kernel_evaluator, make_radial_basis
from metrics import estimate_distance_weights, weighted_kernel_l2
from regression import fit_theta
from tests import make_test_functions
from time_tests import make_time_test_functions
from utils import make_nested_pair_sampler, make_pair_sampler
from bandwidth import sample_pair_distances, choose_space_bandwidth


def fit_kernel_from_particles(
    config,
    X,
    t,
    *,
    return_system=False,
    pair_mode=None,
    pair_seed=None,
):
    cfg = merge_config(default_config(), config or {})

    X = np.asarray(X)
    t = np.asarray(t)
    d_cfg = int(cfg.get("simulation", {}).get("d", 1))
    if X.ndim != 4 or X.shape[-1] != d_cfg:
        raise ValueError("X must have shape (R, T+1, N, d) matching simulation.d")

    if pair_seed is None:
        pair_seed = int(cfg["pair_sampling"]["seed"])
    rng = np.random.default_rng(pair_seed)

    # Test functions
    # Space Test functions
    scfg = cfg["space_tests"]
    L = int(scfg["L"])
    q_lo, q_hi = scfg["quantiles"]
    x_all = X.reshape(-1, d_cfg)
    a, b = np.quantile(x_all, [q_lo, q_hi], axis=0)
    
    test_fns = None
    stype = scfg.get("type", "gaussian")
    layout = scfg.get("layout", "grid")
    if d_cfg > 1 and layout == "grid":
        layout = "random"
    space_rng = np.random.default_rng(scfg.get("seed", cfg["pair_sampling"]["seed"]))

    if stype in ("gaussian", "bump"):
        if L <= 0:
            raise ValueError("L must be > 0 for spatial tests")

        fixed_centers = scfg.get("centers")
        if fixed_centers is not None:
            centers = np.asarray(fixed_centers, dtype=float)
            if centers.ndim == 1:
                centers = centers[:, None]
            if centers.shape != (L, d_cfg):
                raise ValueError("space_tests.centers must have shape (L, d)")
            if L > 1:
                delta = float(np.mean(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
            else:
                delta = float(scfg.get("s", 1.0))
        elif d_cfg == 1 and layout == "grid":
            if L == 1:
                centers = np.array([(a[0] + b[0]) / 2.0], dtype=float)
                delta = (b[0] - a[0]) if b[0] > a[0] else 1.0
            else:
                centers = np.linspace(a[0], b[0], L)
                delta = (b[0] - a[0]) / (L - 1) if b[0] > a[0] else 1.0
        else:
            # random centers inside quantile box for d >= 1
            if L == 1:
                centers = ((a + b) / 2.0)[None, :]
            else:
                centers = space_rng.uniform(a, b, size=(L, d_cfg))
            # heuristic delta: average spacing via volume-per-center
            box = np.maximum(b - a, 1e-12)
            volume = float(np.prod(box))
            delta = (volume / max(L, 1)) ** (1.0 / d_cfg)

        # Optional: auto bandwidth for space tests (minimal)
        bw_cfg = cfg.get("bandwidth", {})
        if bw_cfg.get("auto", False):
            r_samples = sample_pair_distances(
                X,
                num_pairs=bw_cfg.get("num_pairs", 20000),
                num_times=bw_cfg.get("num_times", 32),
                seed=bw_cfg.get("seed", cfg["pair_sampling"]["seed"]),
                time_mode=bw_cfg.get("time_mode", "uniform"),
                exclude_self=True,
            )
            bw = choose_space_bandwidth(
                r_samples,
                delta=delta,
                quantile=bw_cfg.get("quantile", 0.5),
                c=bw_cfg.get("c", 1.0),
                clip=bw_cfg.get("clip", (1e-3, 1e3)),
            )
            if bw is not None:
                scfg["s_factor"] = float(bw["s_factor"])
                scfg["s"] = float(bw["s_abs"])
                scfg["_bandwidth_stats"] = {
                    "r_q": float(bw["r_q"]),
                    "s_abs": float(bw["s_abs"]),
                    "s_factor": float(bw["s_factor"]),
                }
        s = float(scfg.get("s", scfg["s_factor"] * delta))
        test_fns = make_test_functions({"centers": centers, "s": s, "type": stype})
    
    else:
        raise ValueError(f"Unknown space test type: {stype}")

    if test_fns is None:
         raise ValueError("Failed to create test functions.")


    # Kernel basis
    # r_max estimation is removed; we rely on explicit ell_grid.
    kb_params = cfg["kernel_basis"].copy()
    kb_params["eps"] = cfg["kernel_basis"].get("eps", 1e-3)
    
    basis_type = kb_params["basis"]
    kernel_basis, kernel_meta = make_kernel_basis(kb_params)

    # Time test functions
    # Always use time_weakform logic
    time_tests = make_time_test_functions(cfg, t)  # K functions
    
    # Compute b and A (weak form)
    trapezoid_weights = make_trapz_weights(t)
    sigma = cfg.get("sigma", cfg.get("simulation", {}).get("sigma"))
    t0 = perf_counter()
    b = compute_b_timeweak(X, t, sigma, test_fns, time_tests, trapezoid_weights)
    response_seconds = perf_counter() - t0

    ps_cfg = cfg["pair_sampling"]
    total_pairs = int(ps_cfg["total_pairs"])
    if pair_mode is None:
        pair_mode = ps_cfg.get("mode")
    if pair_mode is None:
        is_exact_count = total_pairs == X.shape[2] * (X.shape[2] - 1)
        if (
            is_exact_count
            and not ps_cfg.get("with_replacement", True)
            and ps_cfg.get("exclude_diagonal", True)
        ):
            pair_mode = "full"
        else:
            pair_mode = "resample_each_time"

    t0 = perf_counter()
    if pair_mode == "full":
        A = compute_A_full_timeweak(
            X,
            t,
            test_fns,
            kernel_basis,
            time_tests,
            weights=trapezoid_weights,
            pair_chunk_size=cfg["weak_form"].get("full_pair_chunk_size", 100000),
        )
    elif pair_mode in (
        "resample_each_time", "fixed_across_time", "nested_resample_each_time"
    ):
        if pair_mode == "nested_resample_each_time":
            if not ps_cfg.get("with_replacement", True):
                raise ValueError("nested pair mode requires sampling with replacement")
            pair_sampler = make_nested_pair_sampler(
                total_pairs, seed=pair_seed,
                exclude_diagonal=ps_cfg.get("exclude_diagonal", True),
            )
        else:
            pair_sampler = make_pair_sampler(
                total_pairs,
                rng=rng,
                with_replacement=ps_cfg.get("with_replacement", True),
                exclude_diagonal=ps_cfg.get("exclude_diagonal", True),
                resample_each_call=pair_mode == "resample_each_time",
            )
        A = compute_A_timeweak(
            X,
            t,
            test_fns,
            kernel_basis,
            pair_sampler,
            time_tests,
            weights=trapezoid_weights,
        )
    else:
        raise ValueError(f"Unknown pair mode: {pair_mode}")
    design_seconds = perf_counter() - t0

    # External drift correction (known)
    f = cfg.get("external_drift_fn", None)
    if f is not None:
        # subtract <grad_phi · f, psi> from b
        # shape b: (R, L, K)
        R, L, K = b.shape
        psi_time = time_tests["psi"]  # (K, T+1)
        weights = trapezoid_weights
        
        corr = np.zeros_like(b)
        for r in range(R):
            for n in range(t.size):
                if weights[n] == 0: continue
                x = X[r, n]
                gphi = test_fns["grad_phi"](x)  # (N, L, d)
                fx = np.asarray(f(x))
                if fx.ndim == 1:
                    fx = fx[:, None]
                mean = np.einsum("nld,nd->nl", gphi, fx).mean(axis=0)
                corr[r, :, :] += mean[:, None] * (weights[n] * psi_time[:, n])[None, :]
        b = b - corr

    row_scale = 1.0
    if cfg.get("weak_form", {}).get("normalize_rows", True):
        row_scale = 1.0 / np.sqrt(test_fns["phi"](X[0, 0]).shape[1] * time_tests["psi"].shape[0])
        A = A * row_scale
        b = b * row_scale

    # Flatten for regression
    R, L, K, M = A.shape
    S = R * L * K
    A_flat = A.reshape(S, M)
    y_flat = b.reshape(S)

    t0 = perf_counter()
    theta_hat, info = fit_theta(
        A_flat,
        y_flat,
        method=cfg["regression"]["method"],
        alpha=cfg["regression"]["alpha"],
        standardize=cfg["regression"]["standardize"],
        max_iter=cfg["regression"].get("max_iter", 10000),
        tol=cfg["regression"].get("tol", 1e-4),
        selection=cfg["regression"].get("selection", "cyclic"),
        threshold=cfg["regression"].get("threshold", 0.05),
    )
    regression_seconds = perf_counter() - t0

    # Diagnostics
    try:
        ata = A_flat.T @ A_flat
        cond = np.linalg.cond(ata)
    except Exception:
        cond = np.inf

    diagnostics = {
        "cond_ata": cond,
        "alpha": info.get("alpha"),
        "kernel_meta": kernel_meta,
        "test_meta": {
            "centers": centers,
            "s": s,
            "quantiles": (q_lo, q_hi),
            "layout": layout,
        },
        "time_test_meta": None if time_tests is None else time_tests.get("meta"),
        "pair_mode": pair_mode,
        "pair_seed": pair_seed,
        "row_scale": row_scale,
        "timing": {
            "response_seconds": response_seconds,
            "design_seconds": design_seconds,
            "regression_seconds": regression_seconds,
        },
    }
    bw_stats = scfg.get("_bandwidth_stats")
    if bw_stats is not None:
        diagnostics["bandwidth"] = bw_stats

    # Optional weighted kernel L2 (requires true theta and params)
    dcfg = cfg.get("diagnostics", {})
    if dcfg.get("method") == "l2w" and dcfg.get("true_kernel"):
        tk = dcfg["true_kernel"]
        phi_hat_fn, _ = make_radial_basis(
            {
                "M": M,
                "basis": basis_type,
                "ell_grid": cfg["kernel_basis"].get("ell_grid"),
                "powers": cfg["kernel_basis"].get("powers"),
                "ell_min": cfg["kernel_basis"].get("ell_min"),
                "ell_max": cfg["kernel_basis"].get("ell_max"),
                "p_min": cfg["kernel_basis"].get("p_min"),
                "p_max": cfg["kernel_basis"].get("p_max"),
                "eps": cfg["kernel_basis"].get("eps", 1e-3),
                "knots": cfg["kernel_basis"].get("knots"),
                "extrapolation": cfg["kernel_basis"].get("extrapolation"),
            }
        )
        rng_w = np.random.default_rng(dcfg.get("seed", 1234))
        r_centers, w, dr, edges = estimate_distance_weights(
            X,
            total_pairs=dcfg.get("total_pairs", 50_000),
            bins=dcfg.get("bins", 200),
            rng=rng_w,
            exclude_diagonal=True,
        )
        if "phi_true_fn" in tk:
            phi_true = tk["phi_true_fn"](r_centers)
        else:
            theta_true = np.asarray(tk["theta_true"], dtype=float)
            phi_true_fn, _ = make_radial_basis(tk["kernel_basis"])
            phi_true = phi_true_fn(r_centers) @ theta_true
        phi_hat = phi_hat_fn(r_centers) @ theta_hat
        l2w, l2w_rel = weighted_kernel_l2(phi_true, phi_hat, r_centers, w, dr)
        diagnostics["kernel_L2w"] = l2w
        diagnostics["kernel_L2w_rel"] = l2w_rel
        diagnostics["w_hist"] = w
        diagnostics["r_bins"] = edges

    results = {
        "theta_hat": theta_hat,
        "K_hat": make_kernel_evaluator(theta_hat, kernel_basis),
        "diagnostics": diagnostics,
        "model_info": info,
    }
    if return_system:
        results["system"] = {
            "A": A,
            "b": b,
            "A_flat": A_flat,
            "b_flat": y_flat,
            "weights": trapezoid_weights,
            "test_fns": test_fns,
            "time_tests": time_tests,
            "kernel_basis": kernel_basis,
            "row_scale": row_scale,
            "pair_mode": pair_mode,
        }
    return results
