import numpy as np


def make_true_kernel(params):
    """
    True odd kernel: K(z) = z * sum_m theta_m * exp(-( |z| - mu_m )^2 / (2 h^2))
    """
    mode = params.get("mode", "morse")
    r_max = float(params.get("r_max", 2.0))

    if mode == "morse_r":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.8))
        l1 = float(params.get("l1", 0.5))
        l2 = float(params.get("l2", 1.5))

        def phi_true(r):
            r = np.asarray(r)
            # Alignment with dictionary: phi(r) includes 'r' factor
            # Original Morse: a*exp(-r/l1) - ...
            # Modified:       r * (a*exp(-r/l1) - ...)
            term1 = a * np.exp(-r / l1)
            term2 = b * np.exp(-r / l2)
            return r * (term1 - term2)

        meta = {"r_max": r_max, "mode": mode, "a": a, "b": b, "l1": l1, "l2": l2}

    elif mode == "gaussian_shell":
        centers = np.asarray(params.get("centers"), dtype=float)
        coefficients = np.asarray(params.get("coefficients"), dtype=float)
        h = float(params.get("h"))
        if (
            centers.ndim != 1
            or coefficients.shape != centers.shape
            or centers.size < 1
            or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(coefficients))
            or np.any(centers <= 0.0)
            or np.any(np.diff(centers) <= 0.0)
            or not np.isfinite(h)
            or h <= 0.0
        ):
            raise ValueError("invalid gaussian_shell interaction")

        def phi_true(r):
            r = np.asarray(r, dtype=float)
            atoms = np.exp(-0.5 * ((r[..., None] - centers) / h) ** 2)
            return atoms @ coefficients

        meta = {
            "r_max": r_max, "mode": mode, "centers": centers.copy(),
            "coefficients": coefficients.copy(), "theta": coefficients.copy(), "h": h,
        }

    elif mode == "morse":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.8))
        l1 = float(params.get("l1", 0.5))
        l2 = float(params.get("l2", 1.5))

        def phi_true(r):
            r = np.asarray(r)
            term1 = a * np.exp(-r / l1)
            term2 = b * np.exp(-r / l2)
            return term1 - term2

        meta = {"r_max": r_max, "mode": mode, "a": a, "b": b, "l1": l1, "l2": l2}

    elif mode == "invpower":
        c1 = float(params.get("c1", 1.0))
        c2 = float(params.get("c2", 1.0))
        p1 = float(params.get("p1", 2.0))
        p2 = float(params.get("p2", 4.0))
        eps = float(params.get("eps", 1e-3))
        rc = float(params.get("rc", r_max))

        def phi_true(r):
            r = np.asarray(r)
            base = c1 / (r + eps) ** p1 - c2 / (r + eps) ** p2
            return base * (r <= rc)

        meta = {
            "r_max": r_max,
            "mode": mode,
            "c1": c1,
            "c2": c2,
            "p1": p1,
            "p2": p2,
            "eps": eps,
            "rc": rc,
        }

    elif mode == "poly_cutoff":
        betas = np.asarray(params.get("betas", [1.0, -0.5]), dtype=float)
        rc = float(params.get("rc", r_max))

        def phi_true(r):
            r = np.asarray(r)
            powers = np.vstack([r**q for q in range(len(betas))]).T
            val = powers @ betas
            return val * (r <= rc)

        meta = {"r_max": r_max, "mode": mode, "betas": betas, "rc": rc}

    elif mode == "qanr":
        eps = float(params.get("eps", 1e-12))

        def phi_true(r):
            r = np.asarray(r)
            return 1.0 / (r + eps) - 1.0

        meta = {
            "r_max": r_max,
            "mode": mode,
            "eps": eps,
            "theta": np.asarray([1.0, -1.0], dtype=float),
            "powers": np.asarray([1.0, 2.0], dtype=float),
        }

    elif mode == "log_attraction":
        eps = float(params.get("eps", 1e-2))
        strength = float(params.get("strength", -1.0 / (2.0 * np.pi)))

        def phi_true(r):
            r = np.asarray(r)
            return strength / (r * r + eps * eps)

        meta = {
            "r_max": r_max,
            "mode": mode,
            "eps": eps,
            "strength": strength,
            "theta": np.asarray([strength], dtype=float),
            "powers": np.asarray([0.0], dtype=float),
        }

    else:
        raise ValueError(f"Unknown true kernel mode: {mode}")

    def K(z):
        z = np.asarray(z)
        if z.ndim == 0:
            z = z[None]
        if z.ndim == 1:
            z = z[:, None]
        r = np.linalg.norm(z, axis=-1)
        kappa = phi_true(r)
        return z * kappa[..., None]

    theta_out = meta.get("theta", None)
    return K, theta_out, meta, phi_true


def simulate_particles(
    N=200,
    T=200,
    dt=0.01,
    R=5,
    sigma=0.2,
    init="uniform",
    init_scale=1.0,
    init_centers=None,
    init_std=None,
    kernel_params=None,
    external_drift_fn=None,
    seed=1234,
    d=1,
    solver="euler",
):
    """
    Euler-Maruyama simulation for interacting particles in R^d.
    Returns:
        X: (R, T+1, N, d)
        t: (T+1,)
        K_true: callable
        theta_true: (M,)
        meta: dict
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, dt * T, T + 1)

    kernel_params = kernel_params or {}
    K_true, theta_true, kernel_meta, phi_true = make_true_kernel(kernel_params)

    d = int(d)
    if d < 1:
        raise ValueError("d must be >= 1")
    X = np.zeros((R, T + 1, N, d), dtype=float)

    for r in range(R):
        if init == "uniform":
            x = rng.uniform(-init_scale, init_scale, size=(N, d))
        elif init == "normal":
            x = rng.normal(0.0, init_scale, size=(N, d))
        elif init == "gaussian_mixture":
            centers = np.asarray(
                init_centers if init_centers is not None else [-0.75, 0.0, 0.75],
                dtype=float,
            )
            std = float(init_std if init_std is not None else init_scale)
            if std <= 0.0:
                raise ValueError("init_std must be positive for gaussian_mixture.")
            labels = rng.integers(0, centers.size, size=N)
            if d == 1:
                x = centers[labels, None] + rng.normal(0.0, std, size=(N, d))
            else:
                x = np.zeros((N, d), dtype=float)
                x[:, 0] = centers[labels]
                x += rng.normal(0.0, std, size=(N, d))
        elif init == "uniform_ball":
            radius = float(init_scale)
            if d == 1:
                x = rng.uniform(-radius, radius, size=(N, d))
            else:
                v = rng.normal(0.0, 1.0, size=(N, d))
                norm = np.linalg.norm(v, axis=1, keepdims=True)
                norm = np.maximum(norm, 1e-12)
                direction = v / norm
                radii = radius * rng.random(size=(N, 1)) ** (1.0 / d)
                x = direction * radii
        else:
            raise ValueError("Unknown init method.")
        X[r, 0, :, :] = x

        for n in range(T):
            if solver == "rk4" and sigma == 0.0:
                def _drift(xc):
                    z = xc[:, None, :] - xc[None, :, :]
                    d_ = K_true(z).mean(axis=1)
                    if external_drift_fn is not None:
                        d_ = d_ + external_drift_fn(xc)
                    return d_
                k1 = _drift(x)
                k2 = _drift(x + 0.5 * dt * k1)
                k3 = _drift(x + 0.5 * dt * k2)
                k4 = _drift(x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                z = x[:, None, :] - x[None, :, :]
                drift = K_true(z).mean(axis=1)
                if external_drift_fn is not None:
                    drift = drift + external_drift_fn(x)
                noise = rng.normal(0.0, 1.0, size=(N, d))
                x = x + drift * dt + sigma * np.sqrt(dt) * noise
            X[r, n + 1, :, :] = x

    meta = {
        "kernel_meta": kernel_meta,
        "sigma": sigma,
        "dt": dt,
        "T": T,
        "R": R,
        "N": N,
        "d": d,
        "phi_true": phi_true,
    }
    return X, t, K_true, theta_true, meta


def simulate_with_kernel(
    K,
    X0,
    t,
    sigma=0.0,
    external_drift_fn=None,
    noise=None,
):
    """
    Simulate using a provided kernel K and initial positions X0.
    X0: (R, N, d) initial positions
    t: (T+1,)
    noise: optional standard normal noise, shape (R, T, N, d)
    Returns X: (R, T+1, N, d)
    """
    t = np.asarray(t)
    X0 = np.asarray(X0)
    if X0.ndim != 3:
        raise ValueError("X0 must have shape (R, N, d)")
    R, N, d = X0.shape
    T = t.shape[0] - 1
    if T < 1:
        raise ValueError("t must have at least 2 points")

    if noise is None:
        rng = np.random.default_rng()
        noise = rng.normal(0.0, 1.0, size=(R, T, N, d))
    else:
        noise = np.asarray(noise)
        if noise.shape != (R, T, N, d):
            raise ValueError("noise must have shape (R, T, N, d)")

    X = np.zeros((R, T + 1, N, d), dtype=float)
    X[:, 0, :, :] = X0
    for r in range(R):
        x = X0[r].copy()
        for n in range(T):
            dt = t[n + 1] - t[n]
            z = x[:, None, :] - x[None, :, :]
            drift = K(z).mean(axis=1)
            if external_drift_fn is not None:
                drift = drift + external_drift_fn(x)
            x = x + drift * dt + sigma * np.sqrt(dt) * noise[r, n]
            X[r, n + 1, :, :] = x
    return X
