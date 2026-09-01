"""Conservative macro-micro compression for kinetic Vlasov states."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp


Array = jax.Array


def velocity_trapezoid_weights(velocity: Array) -> Array:
    velocity = jnp.asarray(velocity)
    if velocity.ndim != 1 or velocity.shape[0] < 2:
        raise ValueError("velocity must be a one-dimensional grid")
    spacing = velocity[1:] - velocity[:-1]
    weights = jnp.zeros_like(velocity)
    weights = weights.at[:-1].add(0.5 * spacing)
    weights = weights.at[1:].add(0.5 * spacing)
    return weights


def raw_velocity_moments(distribution: Array, velocity: Array) -> Array:
    """Return the first three raw moments at every spatial point."""
    distribution = jnp.asarray(distribution)
    velocity = jnp.asarray(velocity, dtype=distribution.dtype)
    weights = velocity_trapezoid_weights(velocity).astype(distribution.dtype)
    powers = jnp.stack((jnp.ones_like(velocity), velocity, velocity * velocity))
    return jnp.einsum("lv,vx,v->lx", powers, distribution, weights)


def local_maxwellian_from_moments(
    moments: Array,
    velocity: Array,
    *,
    temperature_floor: float = 1e-8,
) -> Array:
    """Build a density-normalized local Maxwellian from M0:M2."""
    moments = jnp.asarray(moments)
    velocity = jnp.asarray(velocity, dtype=moments.dtype)
    density = jnp.maximum(moments[0], 1e-30)
    flow = moments[1] / density
    temperature = jnp.maximum(
        moments[2] / density - flow * flow,
        jnp.asarray(temperature_floor, dtype=moments.dtype),
    )
    centered = velocity[:, None] - flow[None]
    raw = jnp.exp(-0.5 * centered * centered / temperature[None])
    weights = velocity_trapezoid_weights(velocity).astype(moments.dtype)
    normalization = jnp.einsum("vx,v->x", raw, weights)
    return raw * (density / jnp.maximum(normalization, 1e-30))[None]


def _fixed_moment_basis(velocity: Array, dtype) -> Tuple[Array, Array]:
    velocity = jnp.asarray(velocity, dtype=dtype)
    gaussian = jnp.exp(-0.5 * velocity * velocity)
    weights = velocity_trapezoid_weights(velocity).astype(dtype)
    gaussian = gaussian / jnp.maximum(jnp.sum(gaussian * weights), 1e-30)
    basis = jnp.stack((gaussian, velocity * gaussian, velocity * velocity * gaussian))
    powers = jnp.stack((jnp.ones_like(velocity), velocity, velocity * velocity))
    gram = jnp.einsum("lv,jv,v->lj", powers, basis, weights)
    return basis, gram


def restore_first_three_moments(
    approximation: Array,
    target_moments: Array,
    velocity: Array,
) -> Array:
    """Apply the minimum three-basis correction matching M0:M2 exactly."""
    approximation = jnp.asarray(approximation)
    basis, gram = _fixed_moment_basis(velocity, approximation.dtype)
    residual = target_moments - raw_velocity_moments(approximation, velocity)
    coefficients = jnp.linalg.solve(gram, residual)
    return approximation + jnp.einsum("jv,jx->vx", basis, coefficients)


def randomized_low_rank_approximation(
    matrix: Array,
    rank: int,
    key: Array,
    *,
    oversample: int = 8,
    power_iterations: int = 1,
) -> Array:
    """Approximate a matrix using a deterministic-seeded randomized SVD."""
    matrix = jnp.asarray(matrix)
    maximum_rank = min(int(matrix.shape[0]), int(matrix.shape[1]))
    requested_rank = int(rank)
    if requested_rank <= 0 or requested_rank > maximum_rank:
        raise ValueError(f"rank must be in [1, {maximum_rank}]")
    sketch_rank = min(maximum_rank, requested_rank + max(int(oversample), 0))
    omega = jax.random.normal(
        key,
        (matrix.shape[1], sketch_rank),
        dtype=matrix.dtype,
    )
    sample = matrix @ omega
    for _ in range(max(int(power_iterations), 0)):
        sample = matrix @ (matrix.T @ sample)
    basis, _ = jnp.linalg.qr(sample, mode="reduced")
    compressed = basis.T @ matrix
    left, singular, right = jnp.linalg.svd(compressed, full_matrices=False)
    retained_left = basis @ left[:, :requested_rank]
    return (retained_left * singular[:requested_rank]) @ right[:requested_rank]


def positive_moment_projection(
    approximation: Array,
    target_moments: Array,
    velocity: Array,
    *,
    iterations: int = 8,
) -> Tuple[Array, Array]:
    """Project onto positivity and the first-three-moment constraints."""
    approximation = jnp.asarray(approximation)
    target_moments = jnp.asarray(target_moments, dtype=approximation.dtype)
    original_negative = approximation < 0.0
    projected = approximation
    for _ in range(max(int(iterations), 0)):
        projected = restore_first_three_moments(
            jnp.maximum(projected, 0.0),
            target_moments,
            velocity,
        )
    projected = jnp.maximum(projected, 0.0)
    density = raw_velocity_moments(projected, velocity)[0]
    projected = projected * (
        target_moments[0] / jnp.maximum(density, 1e-30)
    )[None]
    return projected, original_negative


def compress_macro_micro_state(
    distribution: Array,
    velocity: Array,
    rank: int,
    key: Array,
    *,
    oversample: int = 8,
    power_iterations: int = 1,
    temperature_floor: float = 1e-8,
) -> Tuple[Array, Dict[str, Array]]:
    """Compress the kinetic residual while preserving M0:M2 and positivity."""
    distribution = jnp.asarray(distribution)
    velocity = jnp.asarray(velocity, dtype=distribution.dtype)
    target_moments = raw_velocity_moments(distribution, velocity)
    macro = local_maxwellian_from_moments(
        target_moments,
        velocity,
        temperature_floor=temperature_floor,
    )
    micro = distribution - macro
    compressed_micro = randomized_low_rank_approximation(
        micro,
        rank,
        key,
        oversample=oversample,
        power_iterations=power_iterations,
    )
    restored = restore_first_three_moments(
        macro + compressed_micro,
        target_moments,
        velocity,
    )
    positive, negative_mask = positive_moment_projection(
        restored,
        target_moments,
        velocity,
    )
    final_moments = raw_velocity_moments(positive, velocity)
    moment_error = final_moments - target_moments
    scale = jnp.maximum(
        jnp.max(jnp.abs(target_moments), axis=1, keepdims=True),
        1e-12,
    )
    diagnostics = {
        "maximum_relative_moment_error": jnp.max(
            jnp.abs(moment_error) / scale
        ),
        "maximum_relative_density_error": jnp.max(
            jnp.abs(moment_error[0]) / scale[0]
        ),
        "maximum_absolute_momentum_error": jnp.max(jnp.abs(moment_error[1])),
        "maximum_relative_second_moment_error": jnp.max(
            jnp.abs(moment_error[2]) / scale[2]
        ),
        "minimum_distribution": jnp.min(positive),
        "negative_preprojection_fraction": jnp.mean(negative_mask),
        "relative_phase_space_compression_error": jnp.linalg.norm(
            positive - distribution
        )
        / jnp.maximum(jnp.linalg.norm(distribution - macro), 1e-30),
    }
    return positive, diagnostics


def positive_density_preserving_projection(values: Array, velocity: Array) -> Array:
    """Remove interpolation undershoots while preserving local density."""
    values = jnp.asarray(values)
    velocity = jnp.asarray(velocity, dtype=values.dtype)
    target_density = jnp.trapezoid(values, x=velocity, axis=-2)
    positive = jnp.maximum(values, 1e-30)
    positive_density = jnp.trapezoid(positive, x=velocity, axis=-2)
    return positive * (
        target_density / jnp.maximum(positive_density, 1e-30)
    )[..., None, :]
