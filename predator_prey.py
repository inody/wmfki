"""Regularized multi-species predator--prey simulation and weak assembly.

The model is a smooth version of the first-order predator--swarm system of
Chen and Kolokolnikov (2014).  Prey positions are denoted by ``x_i`` and the
single predator position by ``z``::

    dx_i/dt = N^{-1} sum_{j != i} K_xx(x_i-x_j) + K_xz(x_i-z),
    dz/dt   = N^{-1} sum_j K_zx(x_j-z).

Separate dictionaries are used for the three directed interaction channels.
"""

from __future__ import annotations

import numpy as np

from features import compute_b_timeweak, make_trapz_weights


def softened_inverse_power_profile(r, power, delta):
    """Return ``(r^2 + delta^2)^(-power/2)``."""
    r = np.asarray(r, dtype=float)
    power = np.asarray(power, dtype=float)
    delta = float(delta)
    if np.any(power <= 0.0) or delta <= 0.0:
        raise ValueError("power and delta must be positive")
    return (r * r + delta * delta) ** (-0.5 * power)


def make_predator_prey_bases(delta, exponential_scales, predator_powers):
    """Construct the three vector-valued channel dictionaries.

    The prey--prey and prey--predator dictionaries contain a softened inverse
    square profile, a constant profile, and exponential profiles.  The
    predator--prey dictionary contains softened inverse powers.
    """
    delta = float(delta)
    scales = np.asarray(exponential_scales, dtype=float)
    powers = np.asarray(predator_powers, dtype=float)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if scales.ndim != 1 or scales.size < 1 or np.any(scales <= 0.0):
        raise ValueError("exponential_scales must be a nonempty positive vector")
    if powers.ndim != 1 or powers.size < 1 or np.any(powers <= 0.0):
        raise ValueError("predator_powers must be a nonempty positive vector")

    def prey_basis(displacement):
        displacement = np.asarray(displacement, dtype=float)
        if displacement.ndim < 1 or displacement.shape[-1] < 1:
            raise ValueError("displacement must have a nonempty final coordinate axis")
        radius = np.linalg.norm(displacement, axis=-1)
        profiles = np.empty(radius.shape + (2 + scales.size,), dtype=float)
        profiles[..., 0] = softened_inverse_power_profile(radius, 2.0, delta)
        profiles[..., 1] = 1.0
        profiles[..., 2:] = np.exp(-radius[..., None] / scales)
        return displacement[..., None, :] * profiles[..., :, None]

    def predator_basis(source_minus_predator):
        source_minus_predator = np.asarray(source_minus_predator, dtype=float)
        if source_minus_predator.ndim < 1 or source_minus_predator.shape[-1] < 1:
            raise ValueError("displacement must have a nonempty final coordinate axis")
        radius = np.linalg.norm(source_minus_predator, axis=-1)
        profiles = softened_inverse_power_profile(
            radius[..., None], powers, delta
        )
        return source_minus_predator[..., None, :] * profiles[..., :, None]

    metadata = {
        "delta": delta,
        "prey_labels": ["softened inverse square", "constant"]
        + [f"exp(-r/{scale:g})" for scale in scales],
        "predator_labels": [f"softened inverse power {power:g}" for power in powers],
        "exponential_scales": scales.copy(),
        "predator_powers": powers.copy(),
    }
    return prey_basis, predator_basis, metadata


def channel_kernels(theta_prey_prey, theta_prey_predator, theta_predator_prey,
                    prey_basis, predator_basis):
    """Create evaluators for the three directed interaction channels."""
    theta_prey_prey = np.asarray(theta_prey_prey, dtype=float)
    theta_prey_predator = np.asarray(theta_prey_predator, dtype=float)
    theta_predator_prey = np.asarray(theta_predator_prey, dtype=float)

    def contract(basis, theta):
        return np.tensordot(basis, theta, axes=([-2], [0]))

    return (
        lambda displacement: contract(prey_basis(displacement), theta_prey_prey),
        lambda displacement: contract(prey_basis(displacement), theta_prey_predator),
        lambda source_minus_predator: contract(
            predator_basis(source_minus_predator), theta_predator_prey
        ),
    )


def predator_prey_velocity(prey, predator, kernels):
    """Evaluate prey and predator velocities at one configuration."""
    prey = np.asarray(prey, dtype=float)
    predator = np.asarray(predator, dtype=float)
    if prey.ndim != 2 or prey.shape[1] != 2 or predator.shape != (2,):
        raise ValueError("prey and predator must have shapes (N,2) and (2,)")
    prey_prey, prey_predator, predator_prey = kernels
    particle_count = prey.shape[0]
    prey_displacements = prey[:, None, :] - prey[None, :, :]
    prey_velocity = prey_prey(prey_displacements).sum(axis=1) / particle_count
    prey_velocity += prey_predator(prey - predator[None, :])
    predator_velocity = predator_prey(prey - predator[None, :]).mean(axis=0)
    return prey_velocity, predator_velocity


_rhs = predator_prey_velocity


def simulate_predator_prey(
    *,
    particle_count,
    steps,
    dt,
    repetitions,
    kernels,
    seed,
    prey_box=(-0.5, 0.5),
    predator_box=(-0.5, 0.5),
    solver="rk4",
    initial_prey=None,
    initial_predator=None,
    divergence_threshold=1e6,
):
    """Simulate the regularized predator--prey system.

    Returns ``(prey, predator, time, diagnostics)`` with shapes
    ``(R,T+1,N,2)`` and ``(R,T+1,1,2)``.  A non-finite or excessively large
    learned rollout is retained as ``NaN`` from its first failed step onward.
    """
    particle_count = int(particle_count)
    steps = int(steps)
    repetitions = int(repetitions)
    dt = float(dt)
    if particle_count < 2 or steps < 1 or repetitions < 1 or dt <= 0.0:
        raise ValueError("invalid simulation dimensions")
    if solver not in {"euler", "rk4"}:
        raise ValueError("solver must be 'euler' or 'rk4'")

    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, steps * dt, steps + 1)
    prey_out = np.full((repetitions, steps + 1, particle_count, 2), np.nan)
    predator_out = np.full((repetitions, steps + 1, 1, 2), np.nan)

    if initial_prey is not None:
        initial_prey = np.asarray(initial_prey, dtype=float)
        if initial_prey.shape != (repetitions, particle_count, 2):
            raise ValueError("initial_prey has the wrong shape")
    if initial_predator is not None:
        initial_predator = np.asarray(initial_predator, dtype=float)
        if initial_predator.shape == (repetitions, 2):
            initial_predator = initial_predator[:, None, :]
        if initial_predator.shape != (repetitions, 1, 2):
            raise ValueError("initial_predator has the wrong shape")

    failed_steps = []
    for repetition in range(repetitions):
        prey = (
            initial_prey[repetition].copy()
            if initial_prey is not None
            else rng.uniform(prey_box[0], prey_box[1], size=(particle_count, 2))
        )
        predator = (
            initial_predator[repetition, 0].copy()
            if initial_predator is not None
            else rng.uniform(predator_box[0], predator_box[1], size=2)
        )
        prey_out[repetition, 0] = prey
        predator_out[repetition, 0, 0] = predator
        failed_step = None

        for step in range(steps):
            if solver == "euler":
                prey_velocity, predator_velocity = _rhs(prey, predator, kernels)
                prey_next = prey + dt * prey_velocity
                predator_next = predator + dt * predator_velocity
            else:
                k1_x, k1_z = _rhs(prey, predator, kernels)
                k2_x, k2_z = _rhs(
                    prey + 0.5 * dt * k1_x,
                    predator + 0.5 * dt * k1_z,
                    kernels,
                )
                k3_x, k3_z = _rhs(
                    prey + 0.5 * dt * k2_x,
                    predator + 0.5 * dt * k2_z,
                    kernels,
                )
                k4_x, k4_z = _rhs(
                    prey + dt * k3_x,
                    predator + dt * k3_z,
                    kernels,
                )
                prey_next = prey + (dt / 6.0) * (
                    k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x
                )
                predator_next = predator + (dt / 6.0) * (
                    k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z
                )

            maximum = max(
                float(np.max(np.abs(prey_next))),
                float(np.max(np.abs(predator_next))),
            )
            if not np.isfinite(maximum) or maximum > divergence_threshold:
                failed_step = step + 1
                break
            prey, predator = prey_next, predator_next
            prey_out[repetition, step + 1] = prey
            predator_out[repetition, step + 1, 0] = predator
        failed_steps.append(failed_step)

    diagnostics = {
        "failed_steps": failed_steps,
        "finite": [step is None for step in failed_steps],
    }
    return prey_out, predator_out, time, diagnostics


def assemble_predator_prey_weak_systems(
    prey,
    predator,
    time,
    *,
    prey_tests,
    predator_tests,
    prey_basis,
    predator_basis,
    time_tests,
    normalize_rows=True,
):
    """Assemble weak systems for prey and predator response equations.

    The prey system concatenates prey--prey and prey--predator columns.  The
    predator system contains the predator--prey columns.  Ordered prey pairs
    are evaluated exactly, while the diagonal contributes zero by oddness.
    """
    prey = np.asarray(prey, dtype=float)
    predator = np.asarray(predator, dtype=float)
    time = np.asarray(time, dtype=float)
    if prey.ndim != 4 or prey.shape[-1] != 2:
        raise ValueError("prey must have shape (R,T+1,N,2)")
    if predator.shape != (prey.shape[0], prey.shape[1], 1, 2):
        raise ValueError("predator must have shape (R,T+1,1,2)")
    if time.shape != (prey.shape[1],):
        raise ValueError("time has the wrong shape")
    if not np.all(np.isfinite(prey)) or not np.all(np.isfinite(predator)):
        raise ValueError("weak assembly requires finite trajectories")

    weights = make_trapz_weights(time)
    response_prey = compute_b_timeweak(
        prey, time, 0.0, prey_tests, time_tests, weights
    )
    response_predator = compute_b_timeweak(
        predator, time, 0.0, predator_tests, time_tests, weights
    )

    repetitions, time_count, particle_count, _ = prey.shape
    temporal_count = time_tests["psi"].shape[0]
    prey_test_count = prey_tests["centers"].shape[0]
    predator_test_count = predator_tests["centers"].shape[0]
    prey_column_count = prey_basis(np.zeros((1, 2))).shape[-2]
    predator_column_count = predator_basis(np.zeros((1, 2))).shape[-2]

    prey_design = np.zeros(
        (repetitions, prey_test_count, temporal_count, 2 * prey_column_count),
        dtype=float,
    )
    predator_design = np.zeros(
        (repetitions, predator_test_count, temporal_count, predator_column_count),
        dtype=float,
    )

    for repetition in range(repetitions):
        for time_index in range(time_count):
            temporal_weight = weights[time_index] * time_tests["psi"][:, time_index]
            if np.all(temporal_weight == 0.0):
                continue
            x = prey[repetition, time_index]
            z = predator[repetition, time_index, 0]

            gradients_x = prey_tests["grad_phi"](x)
            pair_basis = prey_basis(x[:, None, :] - x[None, :, :])
            mean_pair_feature = pair_basis.sum(axis=1) / particle_count
            prey_prey_feature = np.einsum(
                "ild,imd->lm", gradients_x, mean_pair_feature
            ) / particle_count
            cross_basis = prey_basis(x - z[None, :])
            prey_predator_feature = np.einsum(
                "ild,imd->lm", gradients_x, cross_basis
            ) / particle_count
            joined = np.concatenate(
                [prey_prey_feature, prey_predator_feature], axis=1
            )
            prey_design[repetition] += (
                joined[:, None, :] * temporal_weight[None, :, None]
            )

            gradient_z = predator_tests["grad_phi"](z[None, :])[0]
            mean_predator_feature = predator_basis(x - z[None, :]).mean(axis=0)
            zx_feature = np.einsum(
                "ld,md->lm", gradient_z, mean_predator_feature
            )
            predator_design[repetition] += (
                zx_feature[:, None, :] * temporal_weight[None, :, None]
            )

    if normalize_rows:
        prey_scale = 1.0 / np.sqrt(prey_test_count * temporal_count)
        predator_scale = 1.0 / np.sqrt(predator_test_count * temporal_count)
        prey_design *= prey_scale
        response_prey *= prey_scale
        predator_design *= predator_scale
        response_predator *= predator_scale
    else:
        prey_scale = predator_scale = 1.0

    return {
        "prey": {
            "A": prey_design,
            "b": response_prey,
            "A_flat": prey_design.reshape(-1, prey_design.shape[-1]),
            "b_flat": response_prey.reshape(-1),
            "row_scale": prey_scale,
        },
        "predator": {
            "A": predator_design,
            "b": response_predator,
            "A_flat": predator_design.reshape(-1, predator_design.shape[-1]),
            "b_flat": response_predator.reshape(-1),
            "row_scale": predator_scale,
        },
        "weights": weights,
    }
