import numpy as np


def make_test_functions(params):
    """
    Isotropic spatial test functions in d dimensions.

    Supported types are ``gaussian`` and the compactly supported C-infinity
    ``bump`` function. The parameter ``s`` is the Gaussian width or bump
    support radius.
    Returns a dict of callables.
    """
    centers = np.asarray(params["centers"], dtype=float)
    s = float(params["s"])
    test_type = str(params.get("type", "gaussian")).lower()
    if s <= 0:
        raise ValueError("s must be positive.")
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2:
        raise ValueError("centers must have shape (L, d)")

    if test_type not in ("gaussian", "bump"):
        raise ValueError(f"Unsupported spatial test type: {test_type}")

    def displacement(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim == 2:
            return x[:, None, :] - centers[None, :, :]
        return x[..., None, :] - centers

    def bump_values(dx):
        u = np.sum((dx / s) ** 2, axis=-1)
        inside = u < 1.0
        value = np.zeros_like(u, dtype=float)
        first = np.zeros_like(u, dtype=float)
        second = np.zeros_like(u, dtype=float)
        v = 1.0 - u[inside]
        value[inside] = np.exp(1.0 - 1.0 / v)
        first[inside] = -value[inside] / (v * v)
        second[inside] = value[inside] * (v ** -4 - 2.0 * v ** -3)
        return u, value, first, second

    def phi(x):
        dx = displacement(x)
        if test_type == "gaussian":
            r2 = np.sum((dx / s) ** 2, axis=-1)
            return np.exp(-0.5 * r2)
        _, value, _, _ = bump_values(dx)
        return value

    def grad_phi(x):
        dx = displacement(x)
        if test_type == "gaussian":
            r2 = np.sum((dx / s) ** 2, axis=-1)
            ph = np.exp(-0.5 * r2)
            return -(dx / (s ** 2)) * ph[..., None]
        _, _, first, _ = bump_values(dx)
        return first[..., None] * (2.0 * dx / (s * s))

    def lap_phi(x):
        dx = displacement(x)
        r2 = np.sum((dx / s) ** 2, axis=-1)
        d = centers.shape[1]
        if test_type == "gaussian":
            ph = np.exp(-0.5 * r2)
            return (r2 - d) * (ph / (s ** 2))
        _, _, first, second = bump_values(dx)
        return (4.0 * r2 * second + 2.0 * d * first) / (s * s)

    return {
        "phi": phi,
        "grad_phi": grad_phi,
        "lap_phi": lap_phi,
        "centers": centers,
        "s": s,
        "type": test_type,
    }
