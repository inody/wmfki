import numpy as np


def make_trapz_weights(t):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be 1D array with at least 2 points")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("t must be strictly increasing")
    w = np.zeros_like(t)
    w[0] = dt[0] / 2.0
    w[-1] = dt[-1] / 2.0
    if t.size > 2:
        w[1:-1] = 0.5 * (t[2:] - t[:-2])
    return w


def compute_b_timeweak(X, t, sigma, test_fns, time_tests, weights=None):
    """
    Compute time-weak response b_{r,l,k} via integration by parts in time.
    Returns:
        b: (R, L, K)
    """
    X = np.asarray(X)
    t = np.asarray(t)
    sigma = float(sigma)
    if X.ndim != 4 or X.shape[-1] < 1:
        raise ValueError("X must have shape (R, T+1, N, d)")
    if t.ndim != 1 or t.size != X.shape[1]:
        raise ValueError("t must have shape (T+1,)")

    psi = np.asarray(time_tests["psi"], dtype=float)
    dpsi = np.asarray(time_tests["dpsi"], dtype=float)
    if psi.shape != dpsi.shape:
        raise ValueError("psi and dpsi must have same shape (K, T+1)")
    if psi.shape[1] != t.size:
        raise ValueError("psi length must match t")
    K = psi.shape[0]
    if weights is None:
        weights = make_trapz_weights(t)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != t.shape:
        raise ValueError("weights must have shape (T+1,)")

    R, T1, N, _ = X.shape
    L = test_fns["phi"](X[0, 0]).shape[1]

    b = np.zeros((R, L, K), dtype=float)
    weighted_dpsi = dpsi * weights[None, :]
    weighted_psi = psi * weights[None, :]

    for r in range(R):
        mu_phi = np.zeros((T1, L), dtype=float)
        mu_lap = np.zeros((T1, L), dtype=float)
        for n in range(T1):
            x = X[r, n]
            mu_phi[n, :] = test_fns["phi"](x).mean(axis=0)
            mu_lap[n, :] = test_fns["lap_phi"](x).mean(axis=0)

        term1 = -(weighted_dpsi @ mu_phi)  # (K, L)
        term2 = -(0.5 * sigma ** 2) * (weighted_psi @ mu_lap)  # (K, L)
        b[r, :, :] = (term1 + term2).T

    return b


def compute_A_timeweak(X, t, test_fns, kernel_basis, pair_sampler, time_tests, weights=None):
    """
    Compute time-weak design tensor A_{r,l,k,m}.
    Returns:
        A: (R, L, K, M)
    """
    X = np.asarray(X)
    t = np.asarray(t)
    if X.ndim != 4 or X.shape[-1] < 1:
        raise ValueError("X must have shape (R, T+1, N, d)")
    if t.ndim != 1 or t.size != X.shape[1]:
        raise ValueError("t must have shape (T+1,)")

    psi_time = np.asarray(time_tests["psi"], dtype=float)
    if psi_time.shape[1] != t.size:
        raise ValueError("psi length must match t")
    K = psi_time.shape[0]
    if weights is None:
        weights = make_trapz_weights(t)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != t.shape:
        raise ValueError("weights must have shape (T+1,)")

    R, T1, N, _ = X.shape
    M = kernel_basis(np.zeros((1, X.shape[-1]))).shape[1]
    L = test_fns["phi"](X[0, 0]).shape[1]

    A = np.zeros((R, L, K, M), dtype=float)

    for r in range(R):
        for n in range(T1):
            weight_k = weights[n] * psi_time[:, n]  # (K,)
            if np.all(weight_k == 0.0):
                continue
            x = X[r, n]
            i_idx, j_idx = pair_sampler(N)
            xi = x[i_idx]
            xj = x[j_idx]
            z = xi - xj
            psi = kernel_basis(z)  # (P, M, d)
            gphi = test_fns["grad_phi"](xi)  # (P, L, d)
            # Unbiased estimator of (1/N^2) sum_{i!=j}: scale by (N-1)/(N*P)
            G = np.einsum("pld,pmd->lm", gphi, psi) * (N - 1) / (N * psi.shape[0])  # (L, M)

            A[r, :, :, :] += G[:, None, :] * weight_k[None, :, None]

    return A


def compute_A_full_timeweak(
    X,
    t,
    test_fns,
    kernel_basis,
    time_tests,
    weights=None,
    pair_chunk_size=100_000,
):
    """Compute the exact ordered-pair weak design without materializing N^2 pairs."""
    X = np.asarray(X)
    t = np.asarray(t)
    if X.ndim != 4 or X.shape[-1] < 1:
        raise ValueError("X must have shape (R, T+1, N, d)")
    if t.ndim != 1 or t.size != X.shape[1]:
        raise ValueError("t must have shape (T+1,)")

    psi_time = np.asarray(time_tests["psi"], dtype=float)
    if psi_time.shape[1] != t.size:
        raise ValueError("psi length must match t")
    if weights is None:
        weights = make_trapz_weights(t)
    weights = np.asarray(weights, dtype=float)

    R, T1, N, _ = X.shape
    K = psi_time.shape[0]
    M = kernel_basis(np.zeros((1, X.shape[-1]))).shape[1]
    L = test_fns["phi"](X[0, 0]).shape[1]
    A = np.zeros((R, L, K, M), dtype=float)
    total_pairs = N * (N - 1)
    pair_chunk_size = max(1, int(pair_chunk_size))

    for r in range(R):
        for n in range(T1):
            weight_k = weights[n] * psi_time[:, n]
            if np.all(weight_k == 0.0):
                continue
            x = X[r, n]
            G_sum = np.zeros((L, M), dtype=float)
            for start in range(0, total_pairs, pair_chunk_size):
                stop = min(start + pair_chunk_size, total_pairs)
                flat = np.arange(start, stop, dtype=np.int64)
                i_idx = flat // (N - 1)
                j_idx = flat % (N - 1)
                j_idx = j_idx + (j_idx >= i_idx)
                z = x[i_idx] - x[j_idx]
                basis = kernel_basis(z)
                gphi = test_fns["grad_phi"](x[i_idx])
                G_sum += np.einsum("pld,pmd->lm", gphi, basis)
            G = G_sum / (N * N)
            A[r, :, :, :] += G[:, None, :] * weight_k[None, :, None]

    return A


def compute_increment_regression_system(
    X,
    t,
    kernel_basis,
    external_drift_fn=None,
    pair_chunk_size=100_000,
):
    """Build Euler-increment regression rows using exact ordered-pair features.

    Each row represents one particle coordinate over one observation interval:
    ``(X(t_{n+1}) - X(t_n)) / dt = A theta + noise``.
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    if X.ndim != 4 or X.shape[-1] < 1:
        raise ValueError("X must have shape (R, T+1, N, d)")
    if t.ndim != 1 or t.size != X.shape[1]:
        raise ValueError("t must have shape (T+1,)")
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("t must be strictly increasing")

    R, T1, N, d = X.shape
    M = kernel_basis(np.zeros((1, d))).shape[1]
    rows_per_time = N * d
    A = np.empty((R * (T1 - 1) * rows_per_time, M), dtype=float)
    y = np.empty(R * (T1 - 1) * rows_per_time, dtype=float)
    total_pairs = N * (N - 1)
    pair_chunk_size = max(1, int(pair_chunk_size))

    row = 0
    for r in range(R):
        for n in range(T1 - 1):
            x = X[r, n]
            feature_sum = np.zeros((N, M, d), dtype=float)
            for start in range(0, total_pairs, pair_chunk_size):
                stop = min(start + pair_chunk_size, total_pairs)
                flat = np.arange(start, stop, dtype=np.int64)
                i_idx = flat // (N - 1)
                j_idx = flat % (N - 1)
                j_idx = j_idx + (j_idx >= i_idx)
                basis = kernel_basis(x[i_idx] - x[j_idx])
                np.add.at(feature_sum, i_idx, basis)

            feature = feature_sum / N
            response = (X[r, n + 1] - x) / dt[n]
            if external_drift_fn is not None:
                response = response - np.asarray(external_drift_fn(x), dtype=float)

            stop_row = row + rows_per_time
            A[row:stop_row] = feature.transpose(0, 2, 1).reshape(rows_per_time, M)
            y[row:stop_row] = response.reshape(rows_per_time)
            row = stop_row

    return A, y
