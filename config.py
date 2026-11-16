def default_config():
    return {
        "task": "fit",
        "pair_sampling": {
            "total_pairs": 50000,
            "with_replacement": True,
            "exclude_diagonal": True,
            "seed": 1234,
        },
        "space_tests": {
            "L": 25,
            "type": "gaussian",
            "layout": "grid",
            "quantiles": (0.01, 0.99),
            "s_factor": 1.5,
            "seed": 1234,
        },
        "time_tests": {
            "K": 8,
            "type": "gaussian",
            "s_factor": 2.0,
            "grid_mode": "global",
        },
        "weak_form": {
            "normalize_rows": True,
            "full_pair_chunk_size": 100000,
        },
        "bandwidth": {
            "auto": False,
            "num_pairs": 20000,
            "num_times": 32,
            "time_mode": "uniform",
            "quantile": 0.5,
            "c": 1.0,
            "clip": (1e-3, 1e3),
        },
        "kernel_basis": {
            "M": 32,
            "basis": "exp_r",
            "ell_grid": None,
        },
        "regression": {
            "method": "lasso",
            "standardize": True,
            "alpha": 0.002,
        },
        "simulation": {
            "N": 500,
            "T": 200,
            "dt": 0.01,
            "R": 5,
            "sigma": 0.1,
            "d": 1,
            "init": "uniform",
            "init_scale": 1.0,
            "seed": 1234,
            "interaction": {
                "mode": "morse_r",
                "a": 1.0,
                "b": 0.8,
                "l1": 0.5,
                "l2": 1.5,
            },
            "external_drift_fn": None,
        },
        "diagnostics": {
            "method": "l2w",
            "total_pairs": 50000,
            "bins": 200,
            "seed": 1234,
        },
        "output": {
            "prefix": "output",
            "save_plots": True,
        },
        "diagnose": {
            "num_pairs": 100000,
            "num_times": 64,
            "grid_size": 1000,
            "fit_pairs": None,
            "plot_prefix": None,
        },
        "sweep": {
            "out_dir": "outputs/sweep",
            "Ns": [500],
            "Rs": [5],
            "sigmas": [0.1],
            "alphas": [1e-4],
            "powers": [1, 2, 3, 4, 5, 6, 7],
            "fit_pairs": 10000,
            "distance_pairs": 50000,
            "support_threshold": 1e-3,
            "seed": 1234,
        },
    }




def merge_config(base, override):
    """
    Recursively merge override into base without mutating inputs.
    """
    if override is None:
        return base
    out = base.copy()
    for k, v in override.items():
        if k in out and isinstance(v, dict) and isinstance(out[k], dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out


def build_config(overrides=None):
    return merge_config(default_config(), overrides or {})
