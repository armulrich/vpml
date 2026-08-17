"""Low-moment Vlasov--Poisson solver with a causal spectral heat-flux closure."""

from __future__ import annotations

import math
from typing import Dict, Tuple

from .jax_runtime import bootstrap_jax_runtime

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp


Array = jax.Array

DEFAULT_DENSITY_FLOOR = 1e-4
DEFAULT_PRESSURE_FLOOR = 1e-4
DEFAULT_NORMALIZED_HEAT_FLUX_BOUND = 8.0


def primitive_fields(
    state: Array,
    k_arr: Array,
    *,
    poisson_sign: float = 1.0,
    density_floor: float = DEFAULT_DENSITY_FLOOR,
) -> Array:
    """Return ``(rho - 1, u, pressure - 1, E)`` from conservative moments."""
    state = jnp.asarray(state)
    rho, momentum, second = state[:, 0], state[:, 1], state[:, 2]
    safe_rho = jnp.maximum(rho, jnp.asarray(density_floor, rho.dtype))
    velocity = momentum / safe_rho
    pressure = second - momentum * velocity
    field = electric_field_from_density(rho, k_arr, poisson_sign=poisson_sign)
    return jnp.stack((rho - 1.0, velocity, pressure - 1.0, field), axis=1)


def electric_field_from_density(
    density: Array,
    k_arr: Array,
    *,
    poisson_sign: float = 1.0,
) -> Array:
    density = jnp.asarray(density)
    k_arr = jnp.asarray(k_arr, dtype=density.dtype)
    density_hat = jnp.fft.rfft(density - jnp.mean(density, axis=-1, keepdims=True), axis=-1)
    field_hat = jnp.zeros_like(density_hat)
    field_hat = field_hat.at[:, 1:].set(
        (float(poisson_sign) * 1j) * density_hat[:, 1:] / k_arr[None, 1:]
    )
    return jnp.fft.irfft(field_hat, n=density.shape[-1], axis=-1).astype(density.dtype)


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


def _complex_weights(real: Array, imag: Array, dtype) -> Array:
    return jnp.asarray(real, dtype=dtype) + 1j * jnp.asarray(imag, dtype=dtype)


def spectral_channel_operator(
    values: Array,
    local_weight: Array,
    spectral_real: Array,
    spectral_imag: Array,
) -> Array:
    """Periodic translation-equivariant channel operator."""
    values = jnp.asarray(values)
    local = jnp.einsum("bcn,oc->bon", values, local_weight)
    values_hat = jnp.fft.rfft(values, axis=-1)
    retained = min(int(spectral_real.shape[0]), int(values_hat.shape[-1]))
    weights = _complex_weights(
        spectral_real[:retained], spectral_imag[:retained], values_hat.dtype
    )
    low_hat = jnp.einsum(
        "bcm,moc->bom", values_hat[:, :, :retained], weights
    )
    output_hat = jnp.zeros(
        (values.shape[0], local_weight.shape[0], values_hat.shape[-1]),
        dtype=values_hat.dtype,
    ).at[:, :, :retained].set(low_hat)
    nonlocal_part = jnp.fft.irfft(output_hat, n=values.shape[-1], axis=-1)
    return (local + nonlocal_part).astype(values.dtype)


def init_spectral_memory_params(
    key: Array,
    *,
    width: int,
    spectral_modes: int,
    input_channels: int = 4,
    dtype=jnp.float32,
) -> Dict[str, Array]:
    """Initialize a compact gated spectral recurrent closure."""
    width = int(width)
    modes = int(spectral_modes)
    if width <= 0 or modes <= 0:
        raise ValueError("width and spectral_modes must be positive")
    keys = jax.random.split(key, 7)
    joined = width + int(input_channels)
    local_scale = math.sqrt(2.0 / float(joined + 2 * width))
    spectral_scale = 0.15 / math.sqrt(float(joined))
    return {
        "recurrent_local": (
            local_scale * jax.random.normal(keys[0], (2 * width, joined), dtype=dtype)
        ),
        "recurrent_spectral_real": (
            spectral_scale
            * jax.random.normal(keys[1], (modes, 2 * width, joined), dtype=dtype)
        ),
        "recurrent_spectral_imag": (
            spectral_scale
            * jax.random.normal(keys[2], (modes, 2 * width, joined), dtype=dtype)
        ),
        "output_local": jnp.zeros((1, width), dtype=dtype),
        "output_spectral_real": jnp.zeros((modes, 1, width), dtype=dtype),
        "output_spectral_imag": jnp.zeros((modes, 1, width), dtype=dtype),
        "amplitude_gain": jnp.zeros((input_channels,), dtype=dtype),
    }


def spectral_memory_closure_step(
    params: Dict[str, Array],
    state: Array,
    hidden: Array,
    amplitude: Array,
    k_arr: Array,
    *,
    input_scale: Array,
    heat_flux_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    previous_heat_flux_gradient: Array | None = None,
    dynamic_amplitude_scaling: bool = False,
    poisson_sign: float = 1.0,
    normalized_heat_flux_bound: float = DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
) -> Tuple[Array, Array]:
    """Advance closure memory and return the central heat-flux divergence."""
    fields = primitive_fields(state, k_arr, poisson_sign=poisson_sign)
    real_dtype = fields.dtype
    if dynamic_amplitude_scaling:
        density_amplitude = jnp.sqrt(jnp.mean(jnp.square(fields[:, 0]), axis=-1))
        nonzero = density_amplitude > 0.0
        physical_amplitude = density_amplitude
        normalized = jnp.where(
            nonzero[:, None, None],
            fields / jnp.where(nonzero, density_amplitude, 1.0)[:, None, None],
            0.0,
        )
        log_amplitude = jnp.where(nonzero, jnp.log(density_amplitude), 0.0)
    else:
        physical_amplitude = jnp.asarray(amplitude, dtype=real_dtype)
        scale = jnp.asarray(input_scale, dtype=real_dtype)[None, :, None]
        normalized = fields / scale
        log_amplitude = jnp.log(jnp.maximum(physical_amplitude, 1e-8))
    amplitude_feature = (
        (log_amplitude - jnp.asarray(amplitude_center, dtype=real_dtype))
        / jnp.asarray(max(float(amplitude_scale), 1e-8), dtype=real_dtype)
    )
    gain_exponent = (
        amplitude_feature[:, None]
        * jnp.tanh(jnp.asarray(params["amplitude_gain"], dtype=real_dtype))[None, :]
    )
    gain = jnp.exp(jnp.clip(gain_exponent, -6.0, 6.0))
    if previous_heat_flux_gradient is not None:
        closure_history = jnp.asarray(previous_heat_flux_gradient, dtype=real_dtype)
        if dynamic_amplitude_scaling:
            closure_history = jnp.where(
                nonzero[:, None],
                closure_history
                / jnp.where(nonzero, density_amplitude, 1.0)[:, None],
                0.0,
            )
        else:
            closure_history = closure_history / jnp.asarray(
                heat_flux_scale, dtype=real_dtype
            )
        normalized = jnp.concatenate((normalized, closure_history[:, None]), axis=1)
    normalized = normalized * gain[:, :, None]
    joined = jnp.concatenate((normalized, hidden), axis=1)
    recurrent = spectral_channel_operator(
        joined,
        params["recurrent_local"],
        params["recurrent_spectral_real"],
        params["recurrent_spectral_imag"],
    )
    gate_logits, candidate_logits = jnp.split(recurrent, 2, axis=1)
    gate = jax.nn.sigmoid(gate_logits)
    candidate = jnp.tanh(candidate_logits)
    hidden_new = gate * hidden + (1.0 - gate) * candidate
    raw_heat_flux_normalized = spectral_channel_operator(
        hidden_new,
        params["output_local"],
        params["output_spectral_real"],
        params["output_spectral_imag"],
    )[:, 0]
    bound = jnp.asarray(normalized_heat_flux_bound, dtype=real_dtype)
    heat_flux_normalized = bound * jnp.tanh(raw_heat_flux_normalized / bound)
    heat_flux = (
        physical_amplitude[:, None]
        * jnp.asarray(heat_flux_scale, dtype=real_dtype)
        * heat_flux_normalized
    )
    heat_flux_gradient = spectral_derivative(heat_flux, k_arr)
    return hidden_new, heat_flux_gradient


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
    rho, momentum, second = state[:, 0], state[:, 1], state[:, 2]
    safe_rho = jnp.maximum(rho, jnp.asarray(density_floor, rho.dtype))
    velocity = momentum / safe_rho
    pressure = jnp.maximum(
        second - momentum * velocity,
        jnp.asarray(pressure_floor, state.dtype),
    )
    field = electric_field_from_density(rho, k_arr, poisson_sign=poisson_sign)
    raw_third_resolved = _dealias(rho * velocity**3 + 3.0 * velocity * pressure)
    force_momentum = _dealias(rho * field)
    force_second = _dealias(momentum * field)
    return jnp.stack(
        (
            -spectral_derivative(momentum, k_arr),
            -spectral_derivative(second, k_arr) - force_momentum,
            -spectral_derivative(raw_third_resolved, k_arr)
            - heat_flux_gradient
            - 2.0 * force_second,
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
    density_mean = jnp.mean(state[:, 0], axis=-1, keepdims=True)
    floor = jnp.asarray(density_floor, state.dtype)
    density_excess = jnp.maximum(updated[:, 0] - floor, 0.0)
    mean_excess = jnp.mean(density_excess, axis=-1, keepdims=True)
    target_excess = jnp.maximum(density_mean - floor, 0.0)
    corrected_density = jnp.where(
        mean_excess > jnp.finfo(state.dtype).eps,
        floor + density_excess * target_excess / mean_excess,
        jnp.broadcast_to(density_mean, density_excess.shape),
    )
    density = jnp.where(
        jnp.min(updated[:, 0], axis=-1, keepdims=True) < floor,
        corrected_density,
        updated[:, 0],
    )
    momentum = updated[:, 1]
    kinetic = momentum * momentum / density
    raw_pressure = updated[:, 2] - kinetic
    pressure_floor_value = jnp.asarray(pressure_floor, state.dtype)
    rounding_margin = (
        8.0
        * jnp.finfo(state.dtype).eps
        * jnp.maximum(1.0, jnp.abs(kinetic))
    )
    corrected_second = pressure_floor_value + kinetic + rounding_margin
    second = jnp.where(
        raw_pressure < pressure_floor_value,
        corrected_second,
        updated[:, 2],
    )
    return jnp.stack((density, momentum, second), axis=1)


def warm_spectral_memory(
    params: Dict[str, Array],
    history: Array,
    amplitude: Array,
    k_arr: Array,
    *,
    width: int,
    input_scale: Array,
    heat_flux_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    closure_history_input: bool = False,
    dynamic_amplitude_scaling: bool = False,
    return_closure_history: bool = False,
    poisson_sign: float = 1.0,
    normalized_heat_flux_bound: float = DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
) -> Array:
    """Initialize causal memory from teacher states preceding a rollout anchor."""
    history = jnp.asarray(history)
    hidden0 = jnp.zeros(
        (history.shape[0], int(width), history.shape[-1]), dtype=history.dtype
    )

    previous0 = jnp.zeros((history.shape[0], history.shape[-1]), dtype=history.dtype)

    def body(carry, state):
        hidden, previous = carry
        hidden_new, gradient = spectral_memory_closure_step(
            params,
            state,
            hidden,
            amplitude,
            k_arr,
            input_scale=input_scale,
            heat_flux_scale=heat_flux_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            previous_heat_flux_gradient=(previous if closure_history_input else None),
            dynamic_amplitude_scaling=dynamic_amplitude_scaling,
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
        )
        return (hidden_new, gradient), None

    (hidden, previous), _ = jax.lax.scan(
        body, (hidden0, previous0), jnp.swapaxes(history, 0, 1)
    )
    return (hidden, previous) if return_closure_history else hidden


def rollout_low_moment_closure(
    params: Dict[str, Array],
    initial_state: Array,
    hidden: Array,
    amplitude: Array,
    k_arr: Array,
    *,
    horizon: int,
    dt: float,
    input_scale: Array,
    heat_flux_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    previous_heat_flux_gradient: Array | None = None,
    closure_history_input: bool = False,
    dynamic_amplitude_scaling: bool = False,
    return_closure_history: bool = False,
    poisson_sign: float = 1.0,
    normalized_heat_flux_bound: float = DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    density_floor: float = DEFAULT_DENSITY_FLOOR,
    pressure_floor: float = DEFAULT_PRESSURE_FLOOR,
) -> Tuple[Array, Array]:
    """Autonomously roll out the closed low-moment model."""
    def body(carry, _):
        state, memory, previous = carry
        memory_new, heat_flux_gradient = spectral_memory_closure_step(
            params,
            state,
            memory,
            amplitude,
            k_arr,
            input_scale=input_scale,
            heat_flux_scale=heat_flux_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            previous_heat_flux_gradient=(previous if closure_history_input else None),
            dynamic_amplitude_scaling=dynamic_amplitude_scaling,
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
        )
        state_new = low_moment_rk4_step(
            state,
            heat_flux_gradient,
            k_arr,
            dt,
            poisson_sign=poisson_sign,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        )
        return (state_new, memory_new, heat_flux_gradient), state_new

    checkpointed_body = jax.checkpoint(body)
    if previous_heat_flux_gradient is None:
        previous_heat_flux_gradient = jnp.zeros(
            (initial_state.shape[0], initial_state.shape[-1]), dtype=initial_state.dtype
        )
    (final_state, final_hidden, final_gradient), states = jax.lax.scan(
        checkpointed_body,
        (initial_state, hidden, previous_heat_flux_gradient),
        xs=None,
        length=int(horizon),
    )
    del final_state
    result = (jnp.swapaxes(states, 0, 1), final_hidden)
    return result + (final_gradient,) if return_closure_history else result
