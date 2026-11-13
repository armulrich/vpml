"""Learned low-moment heat-flux closures and their latent-state equations.

These functions evaluate a closure. Fluid integration and history advancement
are handled separately by solvers.moment_rollout."""

from __future__ import annotations

from typing import Dict, Tuple

from .._jax_precision import enable_x64

enable_x64()

import jax
import jax.numpy as jnp

from .moment_features import (
    DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    DYNAMIC_INPUT_SCALING,
    FIXED_INPUT_SCALING,
    _window_fno_features,
)

from .spectral_layers import (
    spectral_channel_operator,
)

from ..solvers.moments import (
    primitive_fields,
)

Array = jax.Array


def explicit_window_closure_step(
    params: Dict[str, Array],
    state: Array,
    history: Array,
    amplitude: Array,
    k_arr: Array,
    *,
    input_scale: Array,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    previous_heat_flux_gradient: Array | None = None,
    heat_flux_gradient_history: Array | None = None,
    poisson_sign: float = 1.0,
    normalized_heat_flux_bound: float = DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    encoded_history: Array | None = None,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
) -> Array:
    """Predict the closure from the current state and an explicit state window."""
    state = jnp.asarray(state)
    features, current_amplitude = _window_fno_features(
        params,
        state,
        history,
        amplitude,
        k_arr,
        input_scale=input_scale,
        heat_flux_gradient_scale=heat_flux_gradient_scale,
        amplitude_center=amplitude_center,
        amplitude_scale=amplitude_scale,
        previous_heat_flux_gradient=previous_heat_flux_gradient,
        heat_flux_gradient_history=heat_flux_gradient_history,
        poisson_sign=poisson_sign,
        encoded_history=encoded_history,
        input_scaling=input_scaling,
        dynamic_amplitude_floor=dynamic_amplitude_floor,
    )
    raw = spectral_channel_operator(
        features,
        params["output_local"],
        params["output_spectral_real"],
        params["output_spectral_imag"],
    )[:, 0]
    bound = jnp.asarray(normalized_heat_flux_bound, dtype=state.dtype)
    normalized_gradient = bound * jnp.tanh(raw / bound)
    if input_scaling == DYNAMIC_INPUT_SCALING:
        output_scale = current_amplitude
    else:
        output_scale = jnp.full(
            (state.shape[0],), heat_flux_gradient_scale, dtype=state.dtype
        )
    gradient = output_scale[:, None] * normalized_gradient
    if allow_uniform_heating:
        return gradient
    return gradient - jnp.mean(gradient, axis=-1, keepdims=True)


def initialize_burles_latent_state(
    params: Dict[str, Array],
    state: Array,
    history: Array,
    amplitude: Array,
    k_arr: Array,
    *,
    input_scale: Array,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    previous_heat_flux_gradient: Array | None = None,
    heat_flux_gradient_history: Array | None = None,
    poisson_sign: float = 1.0,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
) -> Array:
    """Encode causal history into the compact learned memory at a reset."""
    latent_dim = int(params["compact_latent_decay_raw"].shape[0])
    if latent_dim == 0:
        return jnp.zeros((state.shape[0], 0, state.shape[-1]), dtype=state.dtype)
    features, _ = _window_fno_features(
        params,
        state,
        history,
        amplitude,
        k_arr,
        input_scale=input_scale,
        heat_flux_gradient_scale=heat_flux_gradient_scale,
        amplitude_center=amplitude_center,
        amplitude_scale=amplitude_scale,
        previous_heat_flux_gradient=previous_heat_flux_gradient,
        heat_flux_gradient_history=heat_flux_gradient_history,
        poisson_sign=poisson_sign,
        input_scaling=input_scaling,
        dynamic_amplitude_floor=dynamic_amplitude_floor,
    )
    raw = spectral_channel_operator(
        features,
        params["compact_latent_init_local"],
        params["compact_latent_init_spectral_real"],
        params["compact_latent_init_spectral_imag"],
    )
    return jnp.tanh(raw)


def burles_latent_closure_rhs(
    params: Dict[str, Array],
    state: Array,
    compact_latent: Array,
    history: Array,
    amplitude: Array,
    k_arr: Array,
    *,
    input_scale: Array,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    previous_heat_flux_gradient: Array,
    heat_flux_gradient_history: Array,
    poisson_sign: float = 1.0,
    normalized_heat_flux_bound: float = DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
) -> Tuple[Array, Array]:
    """Return the next effective closure and compact-memory time derivative."""
    features, current_amplitude = _window_fno_features(
        params,
        state,
        history,
        amplitude,
        k_arr,
        input_scale=input_scale,
        heat_flux_gradient_scale=heat_flux_gradient_scale,
        amplitude_center=amplitude_center,
        amplitude_scale=amplitude_scale,
        previous_heat_flux_gradient=previous_heat_flux_gradient,
        heat_flux_gradient_history=heat_flux_gradient_history,
        poisson_sign=poisson_sign,
        input_scaling=input_scaling,
        dynamic_amplitude_floor=dynamic_amplitude_floor,
        compact_latent=compact_latent,
    )
    raw_gradient = spectral_channel_operator(
        features,
        params["output_local"],
        params["output_spectral_real"],
        params["output_spectral_imag"],
    )[:, 0]
    bound = jnp.asarray(normalized_heat_flux_bound, dtype=state.dtype)
    normalized_gradient = bound * jnp.tanh(raw_gradient / bound)
    output_scale = (
        current_amplitude
        if input_scaling == DYNAMIC_INPUT_SCALING
        else jnp.full(
            (state.shape[0],), heat_flux_gradient_scale, dtype=state.dtype
        )
    )
    gradient = output_scale[:, None] * normalized_gradient
    if not allow_uniform_heating:
        gradient = gradient - jnp.mean(gradient, axis=-1, keepdims=True)
    latent_dim = int(compact_latent.shape[1])
    if latent_dim == 0:
        latent_rhs = jnp.zeros_like(compact_latent)
    else:
        forcing = spectral_channel_operator(
            features,
            params["compact_latent_rhs_local"],
            params["compact_latent_rhs_spectral_real"],
            params["compact_latent_rhs_spectral_imag"],
        )
        decay = jax.nn.softplus(params["compact_latent_decay_raw"])[None, :, None]
        latent_rhs = jnp.tanh(forcing) - decay * compact_latent
    return gradient, latent_rhs


def spectral_memory_closure_step(
    params: Dict[str, Array],
    state: Array,
    hidden: Array,
    amplitude: Array,
    k_arr: Array,
    *,
    input_scale: Array,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    previous_heat_flux_gradient: Array | None = None,
    poisson_sign: float = 1.0,
    normalized_heat_flux_bound: float = DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
) -> Tuple[Array, Array]:
    """Advance closure memory and return the central heat-flux divergence."""
    fields = primitive_fields(state, k_arr, poisson_sign=poisson_sign)
    real_dtype = fields.dtype
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
        closure_history = closure_history / jnp.asarray(
            heat_flux_gradient_scale, dtype=real_dtype
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
    raw_gradient_normalized = spectral_channel_operator(
        hidden_new,
        params["output_local"],
        params["output_spectral_real"],
        params["output_spectral_imag"],
    )[:, 0]
    bound = jnp.asarray(normalized_heat_flux_bound, dtype=real_dtype)
    gradient_normalized = bound * jnp.tanh(raw_gradient_normalized / bound)
    heat_flux_gradient = (
        jnp.asarray(heat_flux_gradient_scale, dtype=real_dtype)
        * gradient_normalized
    )
    heat_flux_gradient = heat_flux_gradient - jnp.mean(
        heat_flux_gradient, axis=-1, keepdims=True
    )
    return hidden_new, heat_flux_gradient
