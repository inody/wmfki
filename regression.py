import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler


def fit_theta(
    A,
    y,
    method="ridge",
    alpha=1e-3,
    standardize=True,
    **kwargs,
):
    """
    Fit theta for y ≈ A theta.
    A: (S, M), y: (S,)
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if A.ndim != 2:
        raise ValueError("A must be 2D (S, M)")
    if y.shape[0] != A.shape[0]:
        raise ValueError("y and A row counts do not match")

    scaler = None
    if standardize:
        scaler = StandardScaler(with_mean=False, with_std=True)
        A_fit = scaler.fit_transform(A)
    else:
        A_fit = A

    method = method.lower()
    model = None

    if method == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(A_fit, y)
    elif method == "lasso":
        max_iter = int(kwargs.get("max_iter", 10000))
        tol = float(kwargs.get("tol", 1e-4))
        selection = kwargs.get("selection", "cyclic")
        model = Lasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=max_iter,
            tol=tol,
            selection=selection,
        )
        model.fit(A_fit, y)
    elif method == "stlsq":
        thresh = float(kwargs.get("threshold", 0.05))
        max_iter = int(kwargs.get("max_iter", 10))
        coef = np.zeros(A_fit.shape[1])
        active = np.ones_like(coef, dtype=bool)
        for _ in range(max_iter):
            if not np.any(active):
                break
            # OLS on active set
            coef_active, *_ = np.linalg.lstsq(A_fit[:, active], y, rcond=None)
            coef[:] = 0.0
            coef[active] = coef_active
            new_active = np.abs(coef) >= thresh
            if np.all(new_active == active):
                break
            active = new_active
        theta_hat = coef
        if scaler is not None:
            theta_hat = theta_hat / scaler.scale_
        info = {
            "model": None,
            "scaler": scaler,
            "alpha": alpha,
            "standardize": standardize,
            "method": method,
            "nnz": int(np.sum(np.abs(theta_hat) > 1e-8)),
        }
        return theta_hat, info

    else:
        raise ValueError(f"Unknown method: {method}")

    theta_hat = model.coef_
    if scaler is not None:
        theta_hat = theta_hat / scaler.scale_
    info = {
        "model": model,
        "scaler": scaler,
        "alpha": getattr(model, "alpha_", alpha),
        "standardize": standardize,
        "method": method,
    }
    return theta_hat, info
