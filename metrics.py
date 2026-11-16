import numpy as np

from kernels import make_radial_basis


def estimate_distance_weights(
    X,
    r_max=None,
    total_pairs=50_000,
    bins=200,
    rng=None,
    exclude_diagonal=True,
    max_total_samples=2_000_000,
):
    """
    Estimate distance density w(r) on [0, r_max] via histogram.
    """
    X = np.asarray(X)
    if X.ndim != 4 or X.shape[-1] < 1:
        raise ValueError("X must have shape (R, T+1, N, d)")
    if rng is None:
        rng = np.random.default_rng()

    R, T1, N, _ = X.shape
    total_needed = R * T1 * total_pairs
    if total_needed > max_total_samples:
        ratio = max_total_samples / total_needed
        select_prob = min(1.0, ratio)
    else:
        select_prob = 1.0

    samples = []
    for r in range(R):
        for n in range(T1):
            if rng.random() > select_prob:
                continue
            x = X[r, n]
            i_idx = rng.integers(0, N, size=total_pairs)
            j_idx = rng.integers(0, N, size=total_pairs)
            if exclude_diagonal:
                mask = i_idx == j_idx
                while np.any(mask):
                    j_idx[mask] = rng.integers(0, N, size=np.sum(mask))
                    mask = i_idx == j_idx
            d = np.linalg.norm(x[i_idx] - x[j_idx], axis=-1)
            samples.append(d)
    d_all = np.concatenate(samples, axis=0)
    if r_max is None:
        r_max = float(np.max(d_all))
    hist, edges = np.histogram(d_all, bins=bins, range=(0.0, r_max), density=True)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    w = hist
    return r_centers, w, dr, edges


def weighted_kernel_l2(phi_true, phi_hat, r_centers, w, dr):
    """
    phi_true/phi_hat: scalar functions evaluated on r_centers (shape (B,))
    """
    phi_true = np.asarray(phi_true)
    phi_hat = np.asarray(phi_hat)
    diff = phi_hat - phi_true
    num = np.sum((diff**2) * w) * dr
    den = np.sum((phi_true**2) * w) * dr
    l2w = np.sqrt(num)
    l2w_rel = l2w / (np.sqrt(den) + 1e-12)
    return float(l2w), float(l2w_rel)


def drift_mse(K_true, K_hat, X, exclude_diagonal=True):
    """
    Compute MSE of drift field on particle locations:
        b(x_i) = (1/N) sum_j K(x_i - x_j)
    X: (R, T+1, N, d)
    """
    X = np.asarray(X)
    if X.ndim != 4 or X.shape[-1] < 1:
        raise ValueError("X must have shape (R, T+1, N, d)")
    R, T1, N, _ = X.shape
    sq_err = 0.0
    count = 0
    for r in range(R):
        for n in range(T1):
            x = X[r, n]
            z = x[:, None, :] - x[None, :, :]
            if exclude_diagonal:
                mask = ~np.eye(N, dtype=bool)
                z_use = z[mask].reshape(N, N - 1, -1)
            else:
                z_use = z
            z_flat = z_use.reshape(-1, z_use.shape[-1])
            k_true = K_true(z_flat).reshape(z_use.shape)
            k_hat = K_hat(z_flat).reshape(z_use.shape)
            b_true = k_true.mean(axis=1)
            b_hat = k_hat.mean(axis=1)
            sq_err += np.sum(np.sum((b_true - b_hat) ** 2, axis=-1))
            count += N
    return float(sq_err / max(count, 1))
