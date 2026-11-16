"""Exact sum assembly and evaluation for the one-dimensional hat sweeps, without changing the model or normalization."""

import numpy as np

from features import make_trapz_weights
from kernels import make_radial_basis
from scripts.run_1d_hat_lasso import make_evaluation_kernel, relative_error
from simulate import make_true_kernel, simulate_with_kernel
from utils import make_pair_sampler


def hat_particle_sums(x, knots):
    """Compute sum_j (x_i-x_j)b_m(|x_i-x_j|) using interval moments of orders 0, 1, and 2."""
    x, knots = np.asarray(x, dtype=float).reshape(-1), np.asarray(knots, dtype=float)
    # The sum is translation-invariant, so center the positions to reduce cancellation in second moments.
    x = x - (x.min() + x.max()) / 2
    ordered = np.sort(x)
    first = np.r_[0., np.cumsum(ordered)]
    second = np.r_[0., np.cumsum(ordered * ordered)]
    left = np.searchsorted(ordered, x[:, None] - knots, side="right")
    right = np.searchsorted(ordered, x[:, None] + knots, side="left")

    def moments(lower, upper):
        count = upper - lower
        s1, s2 = first[upper] - first[lower], second[upper] - second[lower]
        z1 = x[:, None] * count - s1
        z2 = x[:, None] ** 2 * count - 2 * x[:, None] * s1 + s2
        return z1, z2

    positive1, positive2 = moments(left[:, 1:], left[:, :-1])
    negative1, negative2 = moments(right[:, :-1], right[:, 1:])
    z1, zabs = positive1 + negative1, positive2 - negative2
    widths = np.diff(knots)
    result = np.zeros((len(x), len(knots)))
    result[:, :-1] += (knots[1:] * z1 - zabs) / widths
    result[:, 1:] += (zabs - knots[:-1] * z1) / widths
    # For r >= knots[-1], the last radial hat equals 1 (constant extension).
    lower_tail = x * left[:, -1] - first[left[:, -1]]
    upper_tail = x * (len(x) - right[:, -1]) - (first[-1] - first[right[:, -1]])
    result[:, -1] += lower_tail + upper_tail
    return result


def validate_candidates(candidates):
    if not candidates or len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("nonempty, uniquely named candidates are required")
    for _, params in candidates:
        make_radial_basis(params)
        if params["basis"] != "hat":
            raise ValueError("this sweep backend supports only radial hats")


def assemble_full(Y, t, space, time, candidates):
    validate_candidates(candidates)
    R, T, N, d = Y.shape
    if d != 1 or N < 2 or len(t) != T:
        raise ValueError("full assembly needs (R,T,N,1)")
    L, K = space["phi"](Y[0, 0]).shape[1], len(time["psi"])
    arrays = {name: np.zeros((R, L, K, params["M"])) for name, params in candidates}
    for repetition, path in enumerate(Y):
        for n, weight in enumerate(make_trapz_weights(t)):
            temporal_weight = weight * time["psi"][:, n]
            if not np.any(temporal_weight):
                continue
            gradient = space["grad_phi"](path[n])[:, :, 0]
            for name, params in candidates:
                sums = hat_particle_sums(path[n, :, 0], params["knots"])
                G = gradient.T @ sums / N ** 2
                arrays[name][repetition] += G[:, None, :] * temporal_weight[None, :, None]
    return {name: (A / np.sqrt(L * K)).reshape(-1, A.shape[-1]) for name, A in arrays.items()}


def assemble_sampled(Y, t, space, time, candidates, P, seed=None, sampler=None):
    validate_candidates(candidates)
    R, T, N, d = Y.shape
    if d != 1 or N < 2 or len(t) != T or P < 1:
        raise ValueError("invalid sampled assembly inputs")
    if sampler is None:
        sampler = make_pair_sampler(P, np.random.default_rng(seed), with_replacement=True,
                                    exclude_diagonal=True, resample_each_call=True)
    L, K = space["phi"](Y[0, 0]).shape[1], len(time["psi"])
    arrays = {name: np.zeros((R, L, K, params["M"])) for name, params in candidates}
    for repetition, path in enumerate(Y):
        for n, weight in enumerate(make_trapz_weights(t)):
            temporal_weight = weight * time["psi"][:, n]
            if not np.any(temporal_weight):
                continue
            i, j = sampler(N)
            if len(i) != P or len(j) != P or np.any(i == j):
                raise ValueError("sampler must return P off-diagonal ordered pairs")
            z = path[n, i, 0] - path[n, j, 0]
            radius = np.abs(z)
            gradient = space["grad_phi"](path[n])[:, :, 0]
            for name, params in candidates:
                knots = np.asarray(params["knots"])
                M = params["M"]
                left = np.clip(np.searchsorted(knots, radius, side="right") - 1, 0, M - 2)
                fraction = np.clip((radius - knots[left]) / (knots[left + 1] - knots[left]), 0, 1)
                ids = i * M + left
                sums = np.bincount(ids, weights=z * (1 - fraction), minlength=N * M)
                sums += np.bincount(ids + 1, weights=z * fraction, minlength=N * M)
                G = gradient.T @ sums.reshape(N, M) * (N - 1) / (N * P)
                arrays[name][repetition] += G[:, None, :] * temporal_weight[None, :, None]
    return {name: (A / np.sqrt(L * K)).reshape(-1, A.shape[-1]) for name, A in arrays.items()}


def simulate_hat(params, theta, X0, t, noise, sigma):
    theta = np.asarray(theta)
    R, N, d = X0.shape
    if d != 1 or noise.shape != (R, len(t) - 1, N, 1):
        raise ValueError("invalid rollout shapes")
    output = np.zeros((R, len(t), N, 1))
    output[:, 0] = X0
    for repetition in range(R):
        x = X0[repetition, :, 0].copy()
        for n, step in enumerate(np.diff(t)):
            if not np.all(np.isfinite(x)) or np.ptp(x) > 2e3:
                raise FloatingPointError("rollout left the numerical safety range")
            drift = hat_particle_sums(x, params["knots"]) @ theta / N
            x = x + step * drift + sigma * np.sqrt(step) * noise[repetition, n, :, 0]
            output[repetition, n + 1, :, 0] = x
    if not np.all(np.isfinite(output)) or np.max(np.abs(output)) > 1e3:
        raise FloatingPointError("nonfinite or divergent rollout")
    return output


def evaluation_statistics(X, t, candidates, true_params):
    """Quadratic forms on the true evaluation trajectory, using all pairs and times and never exposing them to training."""
    validate_candidates(candidates)
    true_kernel, _, _, _ = make_true_kernel(true_params)
    N = X.shape[2]
    output = {}
    for name, params in candidates:
        M = params["M"]
        output[name] = dict(force_diag=np.zeros(M), force_off=np.zeros(M - 1), force_cross=np.zeros(M),
                            force_norm=0., drift_gram=np.zeros((M, M)), drift_cross=np.zeros(M), drift_norm=0.)
    for path in X:
        for x, weight in zip(path, make_trapz_weights(t)):
            z = x[:, None, 0] - x[None, :, 0]
            radius = np.abs(z).ravel()
            truth = true_kernel(z[..., None])[..., 0]
            radial_truth = true_kernel(radius[:, None])[:, 0]
            true_drift = truth.mean(axis=1)
            for name, params in candidates:
                M, knots = params["M"], np.asarray(params["knots"])
                left = np.clip(np.searchsorted(knots, radius, side="right") - 1, 0, M - 2)
                fraction = np.clip((radius - knots[left]) / (knots[left + 1] - knots[left]), 0, 1)
                low, high = radius * (1 - fraction), radius * fraction
                stat = output[name]
                stat["force_diag"] += weight * (np.bincount(left, weights=low ** 2, minlength=M)
                                                   + np.bincount(left + 1, weights=high ** 2, minlength=M))
                stat["force_off"] += weight * np.bincount(left, weights=low * high, minlength=M - 1)
                stat["force_cross"] += weight * (np.bincount(left, weights=low * radial_truth, minlength=M)
                                                    + np.bincount(left + 1, weights=high * radial_truth, minlength=M))
                stat["force_norm"] += weight * float(np.sum(truth ** 2))
                drift = hat_particle_sums(x[:, 0], knots) / N
                stat["drift_gram"] += weight * drift.T @ drift
                stat["drift_cross"] += weight * drift.T @ true_drift
                stat["drift_norm"] += weight * float(np.sum(true_drift ** 2))
    return output


def force_energy(theta, stat):
    theta = np.asarray(theta)
    return float(np.dot(theta ** 2, stat["force_diag"]) + 2 * np.dot(theta[:-1] * theta[1:], stat["force_off"]))


def errors_from_statistics(theta, stat):
    theta = np.asarray(theta)
    f = force_energy(theta, stat) - 2 * theta @ stat["force_cross"] + stat["force_norm"]
    d = theta @ stat["drift_gram"] @ theta - 2 * theta @ stat["drift_cross"] + stat["drift_norm"]
    # Allow only roundoff in the quadratic forms; a substantially negative value indicates an inconsistency.
    for numerator, denominator in ((f, stat["force_norm"]), (d, stat["drift_norm"])):
        if numerator < -1e-10 * max(1., denominator):
            raise ValueError("negative squared error beyond roundoff")
    return {"force_error_weighted_rel": relative_error(max(float(f), 0.), stat["force_norm"]),
            "drift_error_rel": relative_error(max(float(d), 0.), stat["drift_norm"])}


def rollout_backend(params, theta, t):
    """Arithmetic implementation determined only by the coefficients and step size; it does not inspect the truth or evaluation error.

    When 2*dt*Lip(K)>=1, a stiff Euler update may amplify roundoff, so use the
    existing full-pair implementation from the first step. Do not change the
    model or step size. This condition is a conservative implementation choice,
    not a theorem guaranteeing Euler stability.
    """
    theta, knots = np.asarray(theta), np.asarray(params["knots"])
    slopes = np.diff(theta) / np.diff(knots)
    lipschitz = max(float(np.max(np.abs(theta[:-1] + knots[:-1] * slopes))),
                    float(np.max(np.abs(theta[1:] + knots[1:] * slopes))), float(abs(theta[-1])))
    indicator = float(2 * np.max(np.diff(t)) * lipschitz)
    return ("reference_pairwise" if indicator >= 1 else "prefix_moments"), indicator


def evaluate_model(data, params, theta, sigma, stat):
    metrics = errors_from_statistics(theta, stat)
    backend, indicator = rollout_backend(params, theta, data["t"])
    metrics.update(rollout_backend=backend, rollout_stiffness_indicator=indicator)
    try:
        with np.errstate(over="raise", invalid="raise"):
            if backend == "prefix_moments":
                rollout = simulate_hat(params, theta, data["X0"], data["t"], data["driving_noise"], sigma)
            else:
                kernel = make_evaluation_kernel(params, theta)
                def guarded(z):
                    if np.any(~np.isfinite(z)) or np.max(np.abs(z)) > 2e3:
                        raise FloatingPointError("rollout left the numerical safety range")
                    return kernel(z)
                rollout = simulate_with_kernel(guarded, data["X0"], data["t"], sigma=sigma, noise=data["driving_noise"])
                if not np.all(np.isfinite(rollout)) or np.max(np.abs(rollout)) > 1e3:
                    raise FloatingPointError("nonfinite or divergent rollout")
        error = rollout - data["X"]
        maximum = float(np.max(np.ptp(rollout[..., 0], axis=2)))
        metrics.update(trajectory_status="finite", trajectory_error_rel=relative_error(
            float(np.sum(error ** 2)), float(np.sum(data["X"] ** 2))),
            trajectory_rmse=float(np.sqrt(np.mean(error ** 2))), trajectory_max_pair_distance=maximum,
            trajectory_exits_basis_interval=maximum > params["knots"][-1])
    except FloatingPointError as error:
        rollout = None
        metrics.update(trajectory_status="diverged", trajectory_error_rel=None, trajectory_rmse=None,
                       trajectory_failure=str(error))
    return metrics, rollout
