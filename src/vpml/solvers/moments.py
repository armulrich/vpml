"""Conservative three-moment equations, periodic operators, and admissibility.

The state contains centered raw moments (M0 - 1, M1, M2 - 1), not the
primitive-field dataset schema. Heat-flux gradients are supplied by callers."""

from __future__ import annotations

from typing import Tuple

from .._jax_precision import enable_x64

enable_x64()

import jax
import jax.numpy as jnp

Array = jax.Array
DEFAULT_DENSITY_FLOOR = 1e-4
DEFAULT_PRESSURE_FLOOR = 1e-4


def primitive_fields(
    state: Array,
    k_arr: Array,
    *,
    poisson_sign: float = 1.0,
    density_floor: float = DEFAULT_DENSITY_FLOOR,
) -> Array:
    """Return ``(rho - 1, u, pressure - 1, E)`` from centered moments."""
    state = jnp.asarray(state)
    density_perturbation, momentum, second_perturbation = (
        state[:, 0],
        state[:, 1],
        state[:, 2],
    )
    rho = 1.0 + density_perturbation
    safe_rho = jnp.maximum(rho, jnp.asarray(density_floor, rho.dtype))
    velocity = momentum / safe_rho
    pressure_perturbation = second_perturbation - momentum * velocity
    field = electric_field_from_density(
        density_perturbation, k_arr, poisson_sign=poisson_sign
    )
    return jnp.stack(
        (density_perturbation, velocity, pressure_perturbation, field), axis=1
    )


def electric_field_from_density(
    density_perturbation: Array,
    k_arr: Array,
    *,
    poisson_sign: float = 1.0,
) -> Array:
    """Solve Poisson's equation from the centered density field."""
    density_perturbation = jnp.asarray(density_perturbation)
    k_arr = jnp.asarray(k_arr, dtype=density_perturbation.dtype)
    density_hat = jnp.fft.rfft(
        density_perturbation
        - jnp.mean(density_perturbation, axis=-1, keepdims=True),
        axis=-1,
    )
    field_hat = jnp.zeros_like(density_hat)
    field_hat = field_hat.at[:, 1:].set(
        (float(poisson_sign) * 1j) * density_hat[:, 1:] / k_arr[None, 1:]
    )
    return jnp.fft.irfft(
        field_hat, n=density_perturbation.shape[-1], axis=-1
    ).astype(density_perturbation.dtype)


def spectral_derivative(values: Array, k_arr: Array) -> Array:
    values = jnp.asarray(values)
    k_arr = jnp.asarray(k_arr, dtype=values.dtype)
    values_hat = jnp.fft.rfft(values, axis=-1)
    derivative_hat = 1j * k_arr * values_hat
    return jnp.fft.irfft(derivative_hat, n=values.shape[-1], axis=-1).astype(values.dtype)


def _dealias(values: Array) -> Array:
    values = jnp.asarray(values)
    coefficients = jnp.fft.rfft(values, axis=-1)
    cutoff = int(values.shape[-1]) // 3
    mask = jnp.arange(coefficients.shape[-1]) <= cutoff
    return jnp.fft.irfft(
        coefficients * mask,
        n=values.shape[-1],
        axis=-1,
    ).astype(values.dtype)


def _batched_low_moment_flux_terms(
    momentum: Array,
    second_perturbation: Array,
    raw_third: Array,
    force_momentum: Array,
    force_second: Array,
    k_arr: Array,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Evaluate low-moment derivatives and dealiased forces in one FFT batch."""
    values = jnp.stack(
        (momentum, second_perturbation, raw_third, force_momentum, force_second),
        axis=1,
    )
    coefficients = jnp.fft.rfft(values, axis=-1)
    real_dtype = values.dtype
    k_values = jnp.asarray(k_arr, dtype=real_dtype)
    cutoff = int(values.shape[-1]) // 3
    mask = (jnp.arange(coefficients.shape[-1]) <= cutoff).astype(real_dtype)
    derivative_multiplier = 1j * k_values
    transformed = jnp.stack(
        (
            derivative_multiplier * coefficients[:, 0],
            derivative_multiplier * coefficients[:, 1],
            derivative_multiplier * coefficients[:, 2] * mask,
            coefficients[:, 3] * mask,
            coefficients[:, 4] * mask,
        ),
        axis=1,
    )
    physical = jnp.fft.irfft(
        transformed, n=values.shape[-1], axis=-1
    ).astype(real_dtype)
    return tuple(physical[:, index] for index in range(5))


def low_moment_rhs(
    state: Array,
    heat_flux_gradient: Array,
    k_arr: Array,
    *,
    poisson_sign: float = 1.0,
    density_floor: float = DEFAULT_DENSITY_FLOOR,
    pressure_floor: float = DEFAULT_PRESSURE_FLOOR,
) -> Array:
    """Conservative three-moment equations with central heat-flux closure."""
    state = jnp.asarray(state)
    density_perturbation, momentum, second_perturbation = (
        state[:, 0],
        state[:, 1],
        state[:, 2],
    )
    rho = 1.0 + density_perturbation
    safe_rho = jnp.maximum(rho, jnp.asarray(density_floor, rho.dtype))
    velocity = momentum / safe_rho
    pressure_perturbation = second_perturbation - momentum * velocity
    pressure = jnp.maximum(
        1.0 + pressure_perturbation, jnp.asarray(pressure_floor, state.dtype)
    )
    field = electric_field_from_density(
        density_perturbation, k_arr, poisson_sign=poisson_sign
    )
    raw_third = rho * velocity**3 + 3.0 * velocity * pressure
    (
        momentum_derivative,
        second_derivative,
        third_derivative,
        force_momentum,
        force_second,
    ) = _batched_low_moment_flux_terms(
        momentum,
        second_perturbation,
        raw_third,
        rho * field,
        momentum * field,
        k_arr,
    )
    return jnp.stack(
        (
            -momentum_derivative,
            -second_derivative - force_momentum,
            -third_derivative - heat_flux_gradient - 2.0 * force_second,
        ),
        axis=1,
    )


def low_moment_rk4_step(
    state: Array,
    heat_flux_gradient: Array,
    k_arr: Array,
    dt: float,
    *,
    poisson_sign: float = 1.0,
    density_floor: float = DEFAULT_DENSITY_FLOOR,
    pressure_floor: float = DEFAULT_PRESSURE_FLOOR,
) -> Array:
    """Advance the fluid moments while holding the closure over one small step."""
    rhs = lambda value: low_moment_rhs(
        value,
        heat_flux_gradient,
        k_arr,
        poisson_sign=poisson_sign,
        density_floor=density_floor,
        pressure_floor=pressure_floor,
    )
    dt_value = jnp.asarray(dt, dtype=state.dtype)
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt_value * k1)
    k3 = rhs(state + 0.5 * dt_value * k2)
    k4 = rhs(state + dt_value * k3)
    updated = state + (dt_value / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Standard 2/3 pseudo-spectral filtering prevents unresolved products from
    # feeding the highest retained modes back into the fluid state.
    updated = _dealias(updated)
    return limit_low_moment_state(
        updated,
        reference_state=state,
        density_floor=density_floor,
        pressure_floor=pressure_floor,
    )


def limit_low_moment_state(
    updated: Array,
    *,
    reference_state: Array | None = None,
    density_floor: float = DEFAULT_DENSITY_FLOOR,
    pressure_floor: float = DEFAULT_PRESSURE_FLOOR,
) -> Array:
    """Enforce density/pressure admissibility while preserving density mass."""
    updated = jnp.asarray(updated)
    reference = updated if reference_state is None else jnp.asarray(reference_state)
    density_mean = 1.0 + jnp.mean(reference[:, 0], axis=-1, keepdims=True)
    floor = jnp.asarray(density_floor, updated.dtype)
    updated_density = 1.0 + updated[:, 0]
    density_excess = jnp.maximum(updated_density - floor, 0.0)
    mean_excess = jnp.mean(density_excess, axis=-1, keepdims=True)
    target_excess = jnp.maximum(density_mean - floor, 0.0)
    rescale_density = mean_excess > jnp.finfo(updated.dtype).eps
    # Guard the division itself: an inactive 0/0 can still poison reverse AD.
    safe_mean_excess = jnp.where(rescale_density, mean_excess, 1.0)
    corrected_density = jnp.where(
        rescale_density,
        floor + density_excess * target_excess / safe_mean_excess,
        jnp.broadcast_to(density_mean, density_excess.shape),
    )
    density_perturbation = jnp.where(
        jnp.min(updated_density, axis=-1, keepdims=True) < floor,
        corrected_density - 1.0,
        updated[:, 0],
    )
    momentum = updated[:, 1]
    density = 1.0 + density_perturbation
    kinetic = momentum * momentum / density
    raw_pressure = 1.0 + updated[:, 2] - kinetic
    pressure_floor_value = jnp.asarray(pressure_floor, updated.dtype)
    rounding_margin = (
        8.0
        * jnp.finfo(updated.dtype).eps
        * jnp.maximum(1.0, jnp.abs(kinetic))
    )
    corrected_second_perturbation = (
        pressure_floor_value + kinetic + rounding_margin - 1.0
    )
    second_perturbation = jnp.where(
        raw_pressure < pressure_floor_value,
        corrected_second_perturbation,
        updated[:, 2],
    )
    return jnp.stack((density_perturbation, momentum, second_perturbation), axis=1)
