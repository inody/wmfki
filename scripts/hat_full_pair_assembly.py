"""Exact full-pair assembly for the one-dimensional hat experiment, aggregating sums by particle first."""

from time import perf_counter

import numpy as np

from features import make_trapz_weights
from kernels import make_radial_basis


def assemble_1d_designs(Y, t, space, time, candidates):
    """Compute N^-2 sum_i grad(phi)(x_i) sum_j K_m(x_i-x_j).

    At most two hat functions are nonzero at each distance, so aggregate with
    bincount. Self-interaction is exactly zero because z=0. Do not sample pairs
    or subsample distances. The returned times are aggregation times for each
    dictionary and exclude shared distance and gradient computations.
    """
    Y, t = np.asarray(Y), np.asarray(t)
    if Y.ndim != 4 or Y.shape[-1] != 1 or t.shape != (Y.shape[1],):
        raise ValueError("cached assembly requires (R,T,N,1) and matching times")
    if not np.all(np.isfinite(Y)) or Y.shape[2] < 2:
        raise ValueError("positions must be finite, with N >= 2")
    if len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    for _, params in candidates:
        make_radial_basis(params)  # Run the existing basis, knot, and extension checks.
        if params["basis"] not in ("hat", "qanr_exp"):
            raise ValueError("unsupported cached basis")
        if params["basis"] == "qanr_exp" and params["M"] != 2 + len(params["ell_grid"]):
            raise ValueError("baseline size does not match its profiles")
    psi = np.asarray(time["psi"])
    if psi.ndim != 2 or psi.shape[1] != len(t):
        raise ValueError("temporal tests do not match the time grid")
    weights = make_trapz_weights(t)
    R, _, N, _ = Y.shape
    L, K = space["phi"](Y[0, 0]).shape[1], psi.shape[0]
    arrays = {name: np.zeros((R, L, K, params["M"])) for name, params in candidates}
    seconds = {name: 0.0 for name, _ in candidates}
    offsets = {name: (np.arange(N) * params["M"])[:, None] for name, params in candidates}
    for repetition, path in enumerate(Y):
        for n, x in enumerate(path):
            temporal_weight = weights[n] * psi[:, n]
            if not np.any(temporal_weight):
                continue
            z = x[:, 0, None] - x[None, :, 0]
            radius = np.abs(z)
            gradient = space["grad_phi"](x)[:, :, 0]
            for name, params in candidates:
                started = perf_counter()
                M = params["M"]
                if params["basis"] == "hat":
                    knots = np.asarray(params["knots"])
                    left = np.clip(np.searchsorted(knots, radius, side="right") - 1, 0, M - 2)
                    fraction = np.clip((radius - knots[left]) / (knots[left + 1] - knots[left]), 0, 1)
                    indices = left + offsets[name]
                    sums = np.bincount(indices.ravel(), weights=(z * (1 - fraction)).ravel(), minlength=N * M)
                    sums += np.bincount((indices + 1).ravel(), weights=(z * fraction).ravel(), minlength=N * M)
                    sums = sums.reshape(N, M)
                else:
                    sums = np.empty((N, M))
                    sums[:, 0] = (z / (radius + params["eps"])).sum(axis=1)
                    sums[:, 1] = z.sum(axis=1)
                    for j, ell in enumerate(params["ell_grid"], start=2):
                        sums[:, j] = (z * np.exp(-radius / ell)).sum(axis=1)
                G = (gradient.T @ sums) / (N * N)
                arrays[name][repetition] += G[:, None, :] * temporal_weight[None, :, None]
                seconds[name] += perf_counter() - started
    scale = 1.0 / np.sqrt(L * K)
    return {name: (A * scale).reshape(-1, A.shape[-1]) for name, A in arrays.items()}, seconds
