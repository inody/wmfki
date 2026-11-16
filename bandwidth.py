import numpy as np


def sample_pair_distances(
    X,
    num_pairs=20000,
    num_times=32,
    seed=0,
    time_mode="uniform",
    exclude_self=True,
):
    """
    Sample pairwise distances r = ||X_i(t) - X_j(t)||.
    X: (R, T+1, N, d)
    Returns: 1D np.ndarray of length num_pairs (or fewer if N<2).
    """
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError("X must have shape (R, T+1, N, d)")
    R, T1, N, d = X.shape
    if N < 2:
        return np.array([], dtype=float)

    rng = np.random.default_rng(seed)

    num_times = int(num_times)
    if num_times < 1:
        num_times = 1
    if num_times > T1:
        num_times = T1

    if time_mode == "late":
        start = int(0.3 * T1)
        time_indices = rng.integers(start, T1, size=num_times)
    else:
        time_indices = rng.integers(0, T1, size=num_times)

    per_time = int(np.ceil(num_pairs / num_times))
    dists = []

    for t_idx in time_indices:
        m = per_time
        r_idx = rng.integers(0, R, size=m)
        i_idx = rng.integers(0, N, size=m)
        j_idx = rng.integers(0, N, size=m)
        if exclude_self:
            mask = i_idx == j_idx
            tries = 0
            while np.any(mask) and tries < 5:
                j_idx[mask] = rng.integers(0, N, size=int(mask.sum()))
                mask = i_idx == j_idx
                tries += 1

        xi = X[r_idx, t_idx, i_idx, :]
        xj = X[r_idx, t_idx, j_idx, :]
        diff = xi - xj
        dist = np.linalg.norm(diff, axis=-1)
        dists.append(dist)

    if not dists:
        return np.array([], dtype=float)

    r_samples = np.concatenate(dists, axis=0)
    if r_samples.size > num_pairs:
        r_samples = r_samples[:num_pairs]
    return r_samples


def choose_space_bandwidth(
    r_samples,
    delta,
    quantile=0.5,
    c=1.0,
    clip=(1e-3, 1e3),
):
    """
    Compute absolute width s_abs and corresponding s_factor.
    s_abs = c * quantile(r)
    s_factor = s_abs / delta
    """
    if r_samples is None or r_samples.size == 0:
        return None
    q = float(quantile)
    q = min(max(q, 0.0), 1.0)
    r_q = float(np.quantile(r_samples, q))
    s_abs = c * r_q
    lo, hi = clip
    s_abs = float(np.clip(s_abs, lo, hi))
    if delta <= 0:
        delta = 1.0
    s_factor = s_abs / float(delta)
    return {
        "r_q": r_q,
        "s_abs": s_abs,
        "s_factor": s_factor,
    }
