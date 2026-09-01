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
DEFAULT_NORMALIZED_HEAT_FLUX_BOUND = 128.0
FIXED_INPUT_SCALING = "fixed_training_scale"
DYNAMIC_INPUT_SCALING = "current_density_rms_arcsinh"
INPUT_SCALING_KINDS = frozenset((FIXED_INPUT_SCALING, DYNAMIC_INPUT_SCALING))


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


def init_explicit_window_params(
    key: Array,
    *,
    width: int,
    spectral_modes: int,
    memory_steps: int,
    input_channels: int = 4,
    dtype=jnp.float32,
) -> Dict[str, Array]:
    """Initialize a causal finite-window spectral closure.

    Unlike the recurrent backend, this model does not compress the observed
    history into a latent state.  Every retained lag has an explicit learned
    coefficient in the Fourier-domain delay kernel.
    """
    width = int(width)
    modes = int(spectral_modes)
    lags = int(memory_steps)
    channels = int(input_channels)
    if min(width, modes, lags, channels) <= 0:
        raise ValueError("width, spectral_modes, memory_steps, and channels must be positive")
    keys = jax.random.split(key, 10)
    history_scale = 0.08 / math.sqrt(float(lags * channels))
    current_scale = 0.12 / math.sqrt(float(channels))
    mixer_scale = math.sqrt(2.0 / float(2 * width))
    return {
        "history_local": history_scale
        * jax.random.normal(keys[0], (width, lags, channels), dtype=dtype),
        "history_spectral_real": history_scale
        * jax.random.normal(keys[1], (modes, width, lags, channels), dtype=dtype),
        "history_spectral_imag": history_scale
        * jax.random.normal(keys[2], (modes, width, lags, channels), dtype=dtype),
        "current_local": current_scale
        * jax.random.normal(keys[3], (width, channels), dtype=dtype),
        "current_spectral_real": current_scale
        * jax.random.normal(keys[4], (modes, width, channels), dtype=dtype),
        "current_spectral_imag": current_scale
        * jax.random.normal(keys[5], (modes, width, channels), dtype=dtype),
        "mixer_local": mixer_scale
        * jax.random.normal(keys[6], (width, width), dtype=dtype),
        "mixer_spectral_real": (0.08 / math.sqrt(float(width)))
        * jax.random.normal(keys[7], (modes, width, width), dtype=dtype),
        "mixer_spectral_imag": (0.08 / math.sqrt(float(width)))
        * jax.random.normal(keys[8], (modes, width, width), dtype=dtype),
        "output_local": jnp.zeros((1, width), dtype=dtype),
        "output_spectral_real": jnp.zeros((modes, 1, width), dtype=dtype),
        "output_spectral_imag": jnp.zeros((modes, 1, width), dtype=dtype),
        "amplitude_gain": jnp.zeros((channels,), dtype=dtype),
    }


def init_window_fno_params(
    key: Array,
    *,
    width: int,
    spectral_modes: int,
    memory_steps: int,
    depth: int = 4,
    input_channels: int = 5,
    dtype=jnp.float32,
) -> Dict[str, Array]:
    """Initialize a deep Fourier operator over an explicit causal window."""
    width = int(width)
    modes = int(spectral_modes)
    lags = int(memory_steps)
    depth = int(depth)
    channels = int(input_channels)
    if min(width, modes, lags, depth, channels) <= 0:
        raise ValueError(
            "width, spectral_modes, memory_steps, depth, and input_channels "
            "must be positive"
        )
    keys = iter(jax.random.split(key, 10 + 3 * depth))
    history_scale = 0.08 / math.sqrt(float(lags * channels))
    current_scale = 0.12 / math.sqrt(float(channels))
    params = {
        "history_local": history_scale
        * jax.random.normal(next(keys), (width, lags, channels), dtype=dtype),
        "history_spectral_real": history_scale
        * jax.random.normal(next(keys), (modes, width, lags, channels), dtype=dtype),
        "history_spectral_imag": history_scale
        * jax.random.normal(next(keys), (modes, width, lags, channels), dtype=dtype),
        "current_local": current_scale
        * jax.random.normal(next(keys), (width, channels), dtype=dtype),
        "current_spectral_real": current_scale
        * jax.random.normal(next(keys), (modes, width, channels), dtype=dtype),
        "current_spectral_imag": current_scale
        * jax.random.normal(next(keys), (modes, width, channels), dtype=dtype),
        "output_local": jnp.zeros((1, width), dtype=dtype),
        "output_spectral_real": jnp.zeros((modes, 1, width), dtype=dtype),
        "output_spectral_imag": jnp.zeros((modes, 1, width), dtype=dtype),
        "amplitude_gain": jnp.zeros((channels,), dtype=dtype),
    }
    local_scale = math.sqrt(1.0 / float(width))
    spectral_scale = 0.08 / math.sqrt(float(width))
    for block in range(depth):
        params[f"fno_block_{block}_local"] = local_scale * jax.random.normal(
            next(keys), (width, width), dtype=dtype
        )
        params[f"fno_block_{block}_spectral_real"] = (
            spectral_scale
            * jax.random.normal(next(keys), (modes, width, width), dtype=dtype)
        )
        params[f"fno_block_{block}_spectral_imag"] = (
            spectral_scale
            * jax.random.normal(next(keys), (modes, width, width), dtype=dtype)
        )
    return params


def init_causal_spacetime_operator_params(
    key: Array,
    *,
    width: int,
    spectral_modes: int,
    memory_steps: int,
    depth: int = 4,
    temporal_kernel_size: int = 5,
    input_channels: int = 5,
    dtype=jnp.float32,
) -> Dict[str, Array]:
    """Initialize a causal nonlinear operator over the complete history window."""
    width = int(width)
    modes = int(spectral_modes)
    lags = int(memory_steps)
    depth = int(depth)
    temporal_kernel_size = int(temporal_kernel_size)
    channels = int(input_channels)
    if min(width, modes, lags, depth, temporal_kernel_size, channels) <= 0:
        raise ValueError(
            "width, spectral_modes, memory_steps, depth, temporal_kernel_size, "
            "and input_channels must be positive"
        )
    keys = iter(jax.random.split(key, 10 + 4 * depth))
    lift_scale = 0.12 / math.sqrt(float(channels))
    current_scale = 0.12 / math.sqrt(float(channels))
    mixer_scale = math.sqrt(1.0 / float(width))
    params = {
        "history_lift_local": lift_scale
        * jax.random.normal(next(keys), (width, channels), dtype=dtype),
        "history_lift_spectral_real": lift_scale
        * jax.random.normal(next(keys), (modes, width, channels), dtype=dtype),
        "history_lift_spectral_imag": lift_scale
        * jax.random.normal(next(keys), (modes, width, channels), dtype=dtype),
        "current_local": current_scale
        * jax.random.normal(next(keys), (width, channels), dtype=dtype),
        "current_spectral_real": current_scale
        * jax.random.normal(next(keys), (modes, width, channels), dtype=dtype),
        "current_spectral_imag": current_scale
        * jax.random.normal(next(keys), (modes, width, channels), dtype=dtype),
        "mixer_local": mixer_scale
        * jax.random.normal(next(keys), (width, width), dtype=dtype),
        "mixer_spectral_real": (0.06 / math.sqrt(float(width)))
        * jax.random.normal(next(keys), (modes, width, width), dtype=dtype),
        "mixer_spectral_imag": (0.06 / math.sqrt(float(width)))
        * jax.random.normal(next(keys), (modes, width, width), dtype=dtype),
        "output_local": jnp.zeros((1, width), dtype=dtype),
        "output_spectral_real": jnp.zeros((modes, 1, width), dtype=dtype),
        "output_spectral_imag": jnp.zeros((modes, 1, width), dtype=dtype),
        "amplitude_gain": jnp.zeros((channels,), dtype=dtype),
    }
    block_scale = 0.08 / math.sqrt(float(width))
    temporal_scale = math.sqrt(2.0 / float(temporal_kernel_size * width))
    for block in range(depth):
        params[f"block_{block}_spatial_local"] = block_scale * jax.random.normal(
            next(keys), (width, width), dtype=dtype
        )
        params[f"block_{block}_spatial_real"] = block_scale * jax.random.normal(
            next(keys), (modes, width, width), dtype=dtype
        )
        params[f"block_{block}_spatial_imag"] = block_scale * jax.random.normal(
            next(keys), (modes, width, width), dtype=dtype
        )
        params[f"block_{block}_temporal"] = temporal_scale * jax.random.normal(
            next(keys), (temporal_kernel_size, width, width), dtype=dtype
        )
    return params


def _causal_temporal_convolution(
    values: Array,
    kernel: Array,
    *,
    dilation: int,
) -> Array:
    """Apply one causal temporal convolution independently at every x point."""
    values = jnp.asarray(values)
    batch, lags, width, nx = values.shape
    temporal = jnp.transpose(values, (0, 3, 1, 2)).reshape(batch * nx, lags, width)
    left_padding = int(dilation) * (int(kernel.shape[0]) - 1)
    convolved = jax.lax.conv_general_dilated(
        temporal,
        jnp.asarray(kernel, dtype=values.dtype),
        window_strides=(1,),
        padding=((left_padding, 0),),
        rhs_dilation=(int(dilation),),
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    return jnp.transpose(
        convolved.reshape(batch, nx, lags, width), (0, 2, 3, 1)
    )


def _normalized_closure_fields(
    params: Dict[str, Array],
    state: Array,
    amplitude: Array,
    k_arr: Array,
    *,
    input_scale: Array,
    amplitude_center: float,
    amplitude_scale: float,
    previous_heat_flux_gradient: Array | None,
    heat_flux_gradient_scale: float,
    poisson_sign: float,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
) -> Array:
    fields = primitive_fields(state, k_arr, poisson_sign=poisson_sign)
    real_dtype = fields.dtype
    if input_scaling not in INPUT_SCALING_KINDS:
        raise ValueError(f"Unsupported low-moment input scaling: {input_scaling}")
    if input_scaling == DYNAMIC_INPUT_SCALING:
        physical_amplitude = jnp.sqrt(jnp.mean(jnp.square(fields[:, 0]), axis=-1))
        physical_amplitude = jnp.maximum(
            physical_amplitude,
            jnp.asarray(dynamic_amplitude_floor, dtype=real_dtype),
        )
        normalized = jnp.arcsinh(fields / physical_amplitude[:, None, None])
    else:
        physical_amplitude = jnp.asarray(amplitude, dtype=real_dtype)
        normalized = fields / jnp.asarray(input_scale, dtype=real_dtype)[None, :, None]
    if previous_heat_flux_gradient is not None:
        previous = jnp.asarray(previous_heat_flux_gradient, dtype=real_dtype)
        if input_scaling == DYNAMIC_INPUT_SCALING:
            previous = jnp.arcsinh(previous / physical_amplitude[:, None])
        else:
            previous = previous / jnp.asarray(heat_flux_gradient_scale, dtype=real_dtype)
        normalized = jnp.concatenate((normalized, previous[:, None]), axis=1)
    log_amplitude = jnp.log(jnp.maximum(physical_amplitude, 1e-8))
    amplitude_feature = (
        (log_amplitude - jnp.asarray(amplitude_center, dtype=real_dtype))
        / jnp.asarray(max(float(amplitude_scale), 1e-8), dtype=real_dtype)
    )
    gain_exponent = amplitude_feature[:, None] * jnp.tanh(
        jnp.asarray(params["amplitude_gain"], dtype=real_dtype)
    )[None, :]
    gain = jnp.exp(jnp.clip(gain_exponent, -6.0, 6.0))
    return normalized * gain[:, :, None]


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
    history = jnp.asarray(history)
    batch, lag_count, _, nx = history.shape
    current = _normalized_closure_fields(
        params,
        state,
        amplitude,
        k_arr,
        input_scale=input_scale,
        amplitude_center=amplitude_center,
        amplitude_scale=amplitude_scale,
        previous_heat_flux_gradient=previous_heat_flux_gradient,
        heat_flux_gradient_scale=heat_flux_gradient_scale,
        poisson_sign=poisson_sign,
        input_scaling=input_scaling,
        dynamic_amplitude_floor=dynamic_amplitude_floor,
    )
    current_amplitude = None
    if input_scaling == DYNAMIC_INPUT_SCALING:
        current_fields = primitive_fields(state, k_arr, poisson_sign=poisson_sign)
        current_amplitude = jnp.maximum(
            jnp.sqrt(jnp.mean(jnp.square(current_fields[:, 0]), axis=-1)),
            jnp.asarray(dynamic_amplitude_floor, dtype=state.dtype),
        )
    if encoded_history is None:
        encoded_history = encode_explicit_window_history(
            params,
            history,
            k_arr,
            input_scale=input_scale,
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            heat_flux_gradient_history=heat_flux_gradient_history,
            poisson_sign=poisson_sign,
            input_scaling=input_scaling,
            normalization_amplitude=current_amplitude,
            dynamic_amplitude_floor=dynamic_amplitude_floor,
        )
    local = encoded_history + jnp.einsum(
        "bcn,wc->bwn", current, params["current_local"]
    )
    current_hat = jnp.fft.rfft(current, axis=-1)
    retained = min(
        int(params["current_spectral_real"].shape[0]), int(current_hat.shape[-1])
    )
    current_weights = _complex_weights(
        params["current_spectral_real"][:retained],
        params["current_spectral_imag"][:retained],
        current_hat.dtype,
    )
    encoded_hat = jnp.einsum(
        "bcm,mwc->bwm", current_hat[..., :retained], current_weights
    )
    full_hat = jnp.zeros(
        (batch, params["current_local"].shape[0], current_hat.shape[-1]),
        dtype=current_hat.dtype,
    ).at[..., :retained].set(encoded_hat)
    encoded = local + jnp.fft.irfft(full_hat, n=nx, axis=-1)
    encoded = jax.nn.silu(encoded)
    if "fno_block_0_local" in params:
        features = encoded
        block = 0
        residual_scale = jnp.asarray(0.5, dtype=state.dtype)
        while f"fno_block_{block}_local" in params:
            update = spectral_channel_operator(
                features,
                params[f"fno_block_{block}_local"],
                params[f"fno_block_{block}_spectral_real"],
                params[f"fno_block_{block}_spectral_imag"],
            )
            features = features + residual_scale * jax.nn.silu(update)
            block += 1
    else:
        mixed = spectral_channel_operator(
            encoded,
            params["mixer_local"],
            params["mixer_spectral_real"],
            params["mixer_spectral_imag"],
        )
        features = encoded + jax.nn.silu(mixed)
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


def encode_explicit_window_history(
    params: Dict[str, Array],
    history: Array,
    k_arr: Array,
    *,
    input_scale: Array,
    heat_flux_gradient_scale: float,
    heat_flux_gradient_history: Array | None = None,
    poisson_sign: float = 1.0,
    input_scaling: str = FIXED_INPUT_SCALING,
    normalization_amplitude: Array | None = None,
    dynamic_amplitude_floor: float = 1e-6,
) -> Array:
    """Encode a sampled history once; reuse it until the next sampled update."""
    history = jnp.asarray(history)
    batch, lag_count, _, nx = history.shape
    history_flat = history.reshape(batch * lag_count, history.shape[2], nx)
    fields = primitive_fields(
        history_flat, k_arr, poisson_sign=poisson_sign
    ).reshape(batch, lag_count, 4, nx)
    if input_scaling not in INPUT_SCALING_KINDS:
        raise ValueError(f"Unsupported low-moment input scaling: {input_scaling}")
    if input_scaling == DYNAMIC_INPUT_SCALING:
        if normalization_amplitude is None:
            normalization_amplitude = jnp.sqrt(
                jnp.mean(jnp.square(fields[:, -1, 0]), axis=-1)
            )
        normalization_amplitude = jnp.maximum(
            jnp.asarray(normalization_amplitude, dtype=history.dtype),
            jnp.asarray(dynamic_amplitude_floor, dtype=history.dtype),
        )
        fields = jnp.arcsinh(
            fields / normalization_amplitude[:, None, None, None]
        )
    else:
        fields = fields / jnp.asarray(input_scale, dtype=history.dtype)[
            None, None, :, None
        ]
    spacetime = "history_lift_local" in params
    history_weights = (
        params["history_lift_local"] if spacetime else params["history_local"]
    )
    expected_channels = int(history_weights.shape[-1])
    if expected_channels == 5:
        if heat_flux_gradient_history is None:
            heat_flux_gradient_history = jnp.zeros(
                (batch, lag_count, nx), dtype=history.dtype
            )
        closure_history = jnp.asarray(
            heat_flux_gradient_history, dtype=history.dtype
        )
        if closure_history.shape != (batch, lag_count, nx):
            raise ValueError(
                "heat-flux-gradient history must match the state-history batch, "
                "lag, and spatial dimensions"
            )
        if input_scaling == DYNAMIC_INPUT_SCALING:
            closure_history = jnp.arcsinh(
                closure_history / normalization_amplitude[:, None, None]
            )
        else:
            closure_history = closure_history / jnp.asarray(
                heat_flux_gradient_scale, dtype=history.dtype
            )
        fields = jnp.concatenate((fields, closure_history[:, :, None]), axis=2)
    elif expected_channels != 4:
        raise ValueError(
            f"explicit-window encoder expects 4 or 5 input channels, got {expected_channels}"
        )
    if spacetime:
        local = jnp.einsum(
            "blcn,wc->blwn", fields, params["history_lift_local"]
        )
        fields_hat = jnp.fft.rfft(fields, axis=-1)
        retained = min(
            int(params["history_lift_spectral_real"].shape[0]),
            int(fields_hat.shape[-1]),
        )
        weights = _complex_weights(
            params["history_lift_spectral_real"][:retained],
            params["history_lift_spectral_imag"][:retained],
            fields_hat.dtype,
        )
        lifted_hat = jnp.einsum(
            "blcm,mwc->blwm", fields_hat[..., :retained], weights
        )
        full_hat = jnp.zeros(
            (
                batch,
                lag_count,
                params["history_lift_local"].shape[0],
                fields_hat.shape[-1],
            ),
            dtype=fields_hat.dtype,
        ).at[..., :retained].set(lifted_hat)
        encoded = jax.nn.silu(
            local + jnp.fft.irfft(full_hat, n=nx, axis=-1)
        )
        block = 0
        residual_scale = jnp.asarray(0.5, dtype=history.dtype)
        while f"block_{block}_temporal" in params:
            flattened = encoded.reshape(batch * lag_count, encoded.shape[2], nx)
            spatial = spectral_channel_operator(
                flattened,
                params[f"block_{block}_spatial_local"],
                params[f"block_{block}_spatial_real"],
                params[f"block_{block}_spatial_imag"],
            ).reshape(encoded.shape)
            temporal = _causal_temporal_convolution(
                encoded,
                params[f"block_{block}_temporal"],
                dilation=2**block,
            )
            encoded = encoded + residual_scale * jax.nn.silu(spatial + temporal)
            block += 1
        return encoded[:, -1]

    local = jnp.einsum("blcn,wlc->bwn", fields, params["history_local"])
    fields_hat = jnp.fft.rfft(fields, axis=-1)
    retained = min(
        int(params["history_spectral_real"].shape[0]), int(fields_hat.shape[-1])
    )
    weights = _complex_weights(
        params["history_spectral_real"][:retained],
        params["history_spectral_imag"][:retained],
        fields_hat.dtype,
    )
    encoded_hat = jnp.einsum(
        "blcm,mwlc->bwm", fields_hat[..., :retained], weights
    )
    full_hat = jnp.zeros(
        (batch, params["history_local"].shape[0], fields_hat.shape[-1]),
        dtype=fields_hat.dtype,
    ).at[..., :retained].set(encoded_hat)
    return local + jnp.fft.irfft(full_hat, n=nx, axis=-1)


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
    density_mean = 1.0 + jnp.mean(state[:, 0], axis=-1, keepdims=True)
    floor = jnp.asarray(density_floor, state.dtype)
    updated_density = 1.0 + updated[:, 0]
    density_excess = jnp.maximum(updated_density - floor, 0.0)
    mean_excess = jnp.mean(density_excess, axis=-1, keepdims=True)
    target_excess = jnp.maximum(density_mean - floor, 0.0)
    corrected_density = jnp.where(
        mean_excess > jnp.finfo(state.dtype).eps,
        floor + density_excess * target_excess / mean_excess,
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
    pressure_floor_value = jnp.asarray(pressure_floor, state.dtype)
    rounding_margin = (
        8.0
        * jnp.finfo(state.dtype).eps
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


def warm_spectral_memory(
    params: Dict[str, Array],
    history: Array,
    amplitude: Array,
    k_arr: Array,
    *,
    width: int,
    input_scale: Array,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    closure_history_input: bool = False,
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
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            previous_heat_flux_gradient=(previous if closure_history_input else None),
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
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    previous_heat_flux_gradient: Array | None = None,
    closure_history_input: bool = False,
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
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            previous_heat_flux_gradient=(previous if closure_history_input else None),
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


def rollout_explicit_window_closure(
    params: Dict[str, Array],
    initial_state: Array,
    history: Array,
    history_counter: Array,
    amplitude: Array,
    k_arr: Array,
    *,
    horizon: int,
    dt: float,
    memory_stride: int,
    input_scale: Array,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    previous_heat_flux_gradient: Array | None = None,
    closure_history_input: bool = False,
    encoded_history: Array | None = None,
    poisson_sign: float = 1.0,
    normalized_heat_flux_bound: float = DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    density_floor: float = DEFAULT_DENSITY_FLOOR,
    pressure_floor: float = DEFAULT_PRESSURE_FLOOR,
    heat_flux_gradient_history: Array | None = None,
    scan_unroll: int = 1,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
) -> Tuple[Array, Tuple[Array, Array, Array, Array, Array]]:
    """Roll out while retaining a sampled window of model-produced states."""
    stride = int(memory_stride)
    if stride <= 0:
        raise ValueError("memory_stride must be positive")
    unroll = int(scan_unroll)
    if unroll <= 0:
        raise ValueError("scan_unroll must be positive")
    if previous_heat_flux_gradient is None:
        previous_heat_flux_gradient = jnp.zeros(
            (initial_state.shape[0], initial_state.shape[-1]), dtype=initial_state.dtype
        )
    if heat_flux_gradient_history is None:
        heat_flux_gradient_history = jnp.zeros(
            (history.shape[0], history.shape[1], history.shape[-1]),
            dtype=history.dtype,
        )
    history_counter = jnp.asarray(history_counter, dtype=jnp.int32)
    if encoded_history is None:
        encoded_history = encode_explicit_window_history(
            params,
            history,
            k_arr,
            input_scale=input_scale,
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            heat_flux_gradient_history=(
                heat_flux_gradient_history if closure_history_input else None
            ),
            poisson_sign=poisson_sign,
            input_scaling=input_scaling,
            dynamic_amplitude_floor=dynamic_amplitude_floor,
        )

    def body(carry, _):
        state, window, closure_window, encoded, counter, previous = carry
        gradient = explicit_window_closure_step(
            params,
            state,
            window,
            amplitude,
            k_arr,
            input_scale=input_scale,
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            previous_heat_flux_gradient=(previous if closure_history_input else None),
            heat_flux_gradient_history=(
                closure_window if closure_history_input else None
            ),
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
            encoded_history=(
                None if input_scaling == DYNAMIC_INPUT_SCALING else encoded
            ),
            input_scaling=input_scaling,
            dynamic_amplitude_floor=dynamic_amplitude_floor,
            allow_uniform_heating=allow_uniform_heating,
        )
        state_new = low_moment_rk4_step(
            state,
            gradient,
            k_arr,
            dt,
            poisson_sign=poisson_sign,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        )
        next_counter = counter + 1
        should_sample = next_counter >= stride
        def sample_window(_):
            sampled_window = jnp.concatenate(
                (window[:, 1:], state_new[:, None]), axis=1
            )
            sampled_closure_window = jnp.concatenate(
                (closure_window[:, 1:], gradient[:, None]), axis=1
            )
            sampled_encoded = encode_explicit_window_history(
                params,
                sampled_window,
                k_arr,
                input_scale=input_scale,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                heat_flux_gradient_history=(
                    sampled_closure_window if closure_history_input else None
                    ),
                    poisson_sign=poisson_sign,
                    input_scaling=input_scaling,
                    dynamic_amplitude_floor=dynamic_amplitude_floor,
                )
            return (
                sampled_window,
                sampled_closure_window,
                sampled_encoded,
                jnp.asarray(0, dtype=jnp.int32),
            )

        def retain_window(_):
            return window, closure_window, encoded, next_counter

        window_new, closure_window_new, encoded_new, counter_new = jax.lax.cond(
            should_sample, sample_window, retain_window, operand=None
        )
        return (
            state_new,
            window_new,
            closure_window_new,
            encoded_new,
            counter_new,
            gradient,
        ), state_new

    (
        final_state,
        final_window,
        final_closure_window,
        final_encoded,
        final_counter,
        final_gradient,
    ), states = jax.lax.scan(
        jax.checkpoint(body),
        (
            initial_state,
            history,
            heat_flux_gradient_history,
            encoded_history,
            history_counter,
            previous_heat_flux_gradient,
        ),
        xs=None,
        length=int(horizon),
        unroll=unroll,
    )
    del final_state
    return jnp.swapaxes(states, 0, 1), (
        final_window,
        final_closure_window,
        final_encoded,
        final_counter,
        final_gradient,
    )
