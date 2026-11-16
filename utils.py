import numpy as np


def make_pair_sampler(
    P,
    rng=None,
    with_replacement=True,
    exclude_diagonal=False,
    resample_each_call=True,
):
    P = int(P)
    if P < 1:
        raise ValueError("P must be >= 1")
    if rng is None:
        rng = np.random.default_rng()

    cached = {}

    def draw(N):
        if with_replacement:
            i_idx = rng.integers(0, N, size=P)
            j_idx = rng.integers(0, N, size=P)
            if exclude_diagonal:
                mask = i_idx == j_idx
                # resample until no diagonal collisions
                while np.any(mask):
                    j_idx[mask] = rng.integers(0, N, size=np.sum(mask))
                    mask = i_idx == j_idx
        else:
            if exclude_diagonal:
                total = N * (N - 1)
                if P > total:
                    raise ValueError("P exceeds N*(N-1) for without_replacement.")
                flat = rng.choice(total, size=P, replace=False)
                i_idx = flat // (N - 1)
                j_idx = flat % (N - 1)
                j_idx = j_idx + (j_idx >= i_idx)
            else:
                total = N * N
                if P > total:
                    raise ValueError("P exceeds N*N for without_replacement.")
                flat = rng.choice(total, size=P, replace=False)
                i_idx = flat // N
                j_idx = flat % N
        return i_idx, j_idx

    def sampler(N):
        if resample_each_call:
            return draw(N)
        if N not in cached:
            cached[N] = draw(N)
        return cached[N]

    return sampler


def make_nested_pair_sampler(P, seed, *, exclude_diagonal=True):
    """Resample at each call while nesting pair prefixes across different P.

    A fresh generator is derived from ``(seed, call index, N)``.  Thus two
    samplers with the same seed and different pair budgets return identical
    prefixes at every corresponding time call.  Ordered off-diagonal pairs are
    sampled directly, avoiding budget-dependent rejection draws.
    """
    P = int(P)
    if P < 1:
        raise ValueError("P must be >= 1")
    if not exclude_diagonal:
        raise ValueError("nested sampler currently requires exclude_diagonal=True")
    call_index = 0

    def sampler(N):
        nonlocal call_index
        N = int(N)
        if N < 2:
            raise ValueError("N must be >= 2 for off-diagonal pairs")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), call_index, N]))
        call_index += 1
        flat = rng.integers(0, N * (N - 1), size=P)
        i_idx = flat // (N - 1)
        j_idx = flat % (N - 1)
        j_idx = j_idx + (j_idx >= i_idx)
        return i_idx, j_idx

    return sampler

