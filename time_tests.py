import numpy as np


def make_time_test_functions(config, t):
    """
    Build time test functions psi_k(t) and dpsi_k(t).
    Returns dict with:
        psi: (K, T+1)
        dpsi: (K, T+1)
        meta: centers, support, type
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be 1D array with at least 2 points")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")

    tcfg = config.get("time_tests", {})
    K = int(tcfg.get("K", config.get("K_time", 10)))
    if K < 1:
        raise ValueError("time test count K must be >= 1")
    test_type = tcfg.get("type", config.get("time_test_type", "hat"))
    s_factor = float(tcfg.get("s_factor", 2.0))

    t0, tT = float(t[0]), float(t[-1])
    Tspan = tT - t0
    if Tspan <= 0:
        raise ValueError("t must span positive duration")

    # Always use global mode: K points uniformly inside (t0, tT)
    # centers = t0 + i * step, i=1..K.  step = Tspan / (K+1).
    step = Tspan / (K + 1)
    support = float(tcfg.get("support", step * s_factor))
    
    if support <= 0:
        raise ValueError("time support must be positive")

    centers = np.linspace(t0, tT, K + 2)[1:-1]

    if test_type not in ("hat", "gaussian", "bump"):
        raise ValueError(f"Unsupported time test type: {test_type}")

    psi = np.zeros((K, t.size), dtype=float)
    dpsi = np.zeros((K, t.size), dtype=float)

    h = support / 2.0
    for k, c in enumerate(centers):
        if test_type == "hat":
            dist = t - c
            inside = np.abs(dist) <= h
            psi_k = np.zeros_like(t)
            psi_k[inside] = 1.0 - np.abs(dist[inside]) / h
            psi[k, :] = psi_k

            dpsi_k = np.zeros_like(t)
            left = (dist < 0) & inside
            right = (dist > 0) & inside
            dpsi_k[left] = 1.0 / h
            dpsi_k[right] = -1.0 / h
            dpsi[k, :] = dpsi_k

        elif test_type == "gaussian":
            # smooth bump: exp( - (t-c)^2 / (2*sigma^2) ), clipped outside support.
            # sigma_div controls how many sigma fit within the half-support h.
            # sigma_div=3 → clip at 3σ, boundary value exp(-4.5)≈0.011 (introduces
            # a jump discontinuity that biases the IBP identity by ~1.6%).
            # sigma_div=6 → clip at 6σ, boundary value exp(-18)≈1.5e-8 (negligible).
            sigma_div = float(tcfg.get("sigma_div", 6))
            sigma_t = h / sigma_div
            dist = t - c
            inside = np.abs(dist) <= h

            val = np.exp(-0.5 * (dist / sigma_t) ** 2)
            dval = -val * (dist / sigma_t ** 2)

            psi[k, inside] = val[inside]
            dpsi[k, inside] = dval[inside]

        elif test_type == "bump":
            dist = t - c
            u = (dist / h) ** 2
            inside = u < 1.0
            v = 1.0 - u[inside]
            val = np.exp(1.0 - 1.0 / v)
            first = -val / (v * v)
            psi[k, inside] = val
            dpsi[k, inside] = first * (2.0 * dist[inside] / (h * h))

    return {"psi": psi, "dpsi": dpsi, "meta": {"centers": centers, "support": support, "type": test_type}}
