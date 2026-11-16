import numpy as np


def make_radial_basis(params):
    """
    Radial basis functions phi_m(r) on r in [0, r_max].
    Returns:
        phi_fn(r) -> (..., M)
        meta dict with centers and width
    """
    # r_max logic removed
    basis = params.get("basis", "rbf")
    h = params.get("h", None)
    centers = params.get("centers", None)
    eps = float(params.get("eps", 1e-3))
    ell_min = params.get("ell_min", None)
    ell_max = params.get("ell_max", None)
    ell_grid = params.get("ell_grid", None)
    p_min = params.get("p_min", None)
    p_max = params.get("p_max", None)
    powers_param = params.get("powers", None)

    if powers_param is not None:
        M = len(powers_param)
    else:
        M = int(params["M"])

    if basis == "exp":
        if ell_grid is not None:
            ells = np.asarray(ell_grid, dtype=float)
        else:
            raise ValueError("ell_grid must be provided for exp basis.")

        def phi_fn(r):
            r = np.asarray(r)
            return np.exp(-r[..., None] / ells)

        meta = {"ells": ells, "basis": basis}

    elif basis == "exp_r":
        if ell_grid is not None:
            ells = np.asarray(ell_grid, dtype=float)
        else:
            raise ValueError("ell_grid must be provided for exp_r basis.")

        def phi_fn(r):
            r = np.asarray(r)
            return (r[..., None]) * np.exp(-r[..., None] / ells)

        meta = {"ells": ells, "basis": basis}

    elif basis == "gaussian_shell":
        shell_centers = np.asarray(centers, dtype=float)
        shell_width = float(h)
        if (
            shell_centers.ndim != 1
            or shell_centers.size != M
            or not np.all(np.isfinite(shell_centers))
            or np.any(shell_centers <= 0.0)
            or np.any(np.diff(shell_centers) <= 0.0)
            or not np.isfinite(shell_width)
            or shell_width <= 0.0
        ):
            raise ValueError(
                "gaussian_shell requires M increasing positive centers and h > 0"
            )

        def phi_fn(r):
            r = np.asarray(r, dtype=float)
            return np.exp(
                -0.5 * ((r[..., None] - shell_centers) / shell_width) ** 2
            )

        meta = {
            "centers": shell_centers.copy(), "h": shell_width, "basis": basis
        }

    elif basis == "invpower":
        if p_min is None or p_max is None:
            p_min, p_max = 1.0, float(M)
        powers = np.linspace(p_min, p_max, M)

        def phi_fn(r):
            r = np.asarray(r)
            return (r[..., None] + eps) ** (-powers)

        meta = {"powers": powers, "eps": eps, "basis": basis}

    elif basis == "grad_abs_power":
        if powers_param is not None:
            powers = np.asarray(powers_param, dtype=float)
        else:
            if p_min is None or p_max is None:
                p_min, p_max = 1.0, float(M)
            powers = np.linspace(p_min, p_max, M)

        def phi_fn(r):
            r = np.asarray(r)
            out = np.empty(r.shape + (powers.size,), dtype=float)
            for j, p in enumerate(powers):
                if np.isclose(p, 0.0):
                    out[..., j] = 1.0 / (r * r + eps * eps)
                elif np.isclose(p, 1.0):
                    out[..., j] = 1.0 / (r + eps)
                else:
                    out[..., j] = r ** (p - 2.0)
            return out

        meta = {"powers": powers, "eps": eps, "basis": basis}

    elif basis == "qanr_exp":
        if ell_grid is None:
            raise ValueError("ell_grid must be provided for qanr_exp basis.")
        ells = np.asarray(ell_grid, dtype=float)
        if ells.ndim != 1 or ells.size < 1 or np.any(ells <= 0.0):
            raise ValueError("ell_grid must contain positive one-dimensional scales.")
        if M != ells.size + 2:
            raise ValueError("qanr_exp requires M = 2 + len(ell_grid).")
        labels = ["1/(r+eps)", "1"] + [f"exp(-r/{ell:g})" for ell in ells]

        def phi_fn(r):
            r = np.asarray(r)
            out = np.empty(r.shape + (ells.size + 2,), dtype=float)
            out[..., 0] = 1.0 / (r + eps)
            out[..., 1] = 1.0
            out[..., 2:] = np.exp(-r[..., None] / ells)
            return out

        meta = {
            "ells": ells,
            "labels": labels,
            "eps": eps,
            "basis": basis,
        }

    elif basis == "hat":
        # Nodal linear basis on a prescribed radial interval. Constant endpoint
        # extension avoids a discontinuity in z * phi(|z|) at the outer boundary.
        knots = np.asarray(params["knots"], dtype=float).copy()
        if (
            knots.ndim != 1
            or knots.size < 2
            or not np.all(np.isfinite(knots))
            or knots[0] != 0.0
            or np.any(np.diff(knots) <= 0.0)
        ):
            raise ValueError("hat knots must be finite, strictly increasing, and start at zero")
        if params["M"] != knots.size:
            raise ValueError("hat M must equal the number of knots")
        if params.get("extrapolation") != "constant":
            raise ValueError("hat requires explicit extrapolation='constant'")
        M = knots.size

        def phi_fn(r):
            r = np.asarray(r, dtype=float)
            if np.any(~np.isfinite(r)) or np.any(r < 0.0):
                raise ValueError("hat radii must be finite and nonnegative")
            flat = np.minimum(r.reshape(-1), knots[-1])
            left = np.clip(np.searchsorted(knots, flat, side="right") - 1, 0, M - 2)
            fraction = (flat - knots[left]) / (knots[left + 1] - knots[left])
            out = np.zeros((flat.size, M), dtype=float)
            rows = np.arange(flat.size)
            out[rows, left] = 1.0 - fraction
            out[rows, left + 1] = fraction
            return out.reshape(r.shape + (M,))

        meta = {"basis": basis, "knots": knots.copy(), "extrapolation": "constant"}

    elif basis == "messenger_log2d":
        labels = [
            "r^2",
            "r^3",
            "r^4",
            "r^5",
            "r^6",
            "[r^(1/2)]_delta",
            "[r(log r - 1)]_delta",
            "1/(r^2+delta^2)",
        ]
        powers = np.asarray([2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)

        def phi_fn(r):
            r = np.asarray(r, dtype=float)
            out = np.empty(r.shape + (len(labels),), dtype=float)
            for j, p in enumerate(powers):
                out[..., j] = r ** (p - 2.0)
            r_delta = np.sqrt(r * r + eps * eps)
            out[..., 5] = r_delta ** (-1.5)
            out[..., 6] = np.log(r_delta) / r_delta
            out[..., 7] = 1.0 / (r * r + eps * eps)
            return out

        meta = {"labels": labels, "eps": eps, "basis": basis}

    else:
        raise ValueError(f"Unknown radial basis: {basis}")

    return phi_fn, meta


def make_kernel_basis(params):
    """
    Odd force-field kernel basis:
        K_theta(z) = sum_m theta_m * (z * phi_m(|z|))
    Returns:
        basis_fn(z) -> (..., M, d) for vector z (d=1 uses (..., 1))
        meta dict with centers and width
    """
    phi_fn, meta = make_radial_basis(params)

    def basis_fn(z):
        z = np.asarray(z)
        if z.ndim == 0:
            z = z[None]
        if z.ndim == 1:
            z = z[:, None]
        if z.ndim < 2:
            raise ValueError("z must be at least 1D after reshape")
        r = np.linalg.norm(z, axis=-1)
        phi = phi_fn(r)[..., None]  # (..., M, 1)
        return z[..., None, :] * phi  # (..., M, d)

    return basis_fn, meta


def make_kernel_evaluator(theta, basis_fn):
    theta = np.asarray(theta, dtype=float)

    def K_hat(z):
        psi = basis_fn(z)
        if psi.ndim < 2:
            raise ValueError("basis_fn must return at least 2D array")
        # (..., M, d) -> (..., d)
        return np.tensordot(psi, theta, axes=([-2], [0]))

    return K_hat
