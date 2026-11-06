"""History encoders and input normalization for low-moment closure models."""

from __future__ import annotations

from typing import Dict, Tuple

from .._jax_precision import enable_x64

enable_x64()

import jax
import jax.numpy as jnp

from .spectral_layers import (
    _causal_temporal_convolution,
    _complex_weights,
    _rms_with_floor,
    spectral_channel_operator,
)

from ..solvers.moments import (
    primitive_fields,
)

Array = jax.Array
DEFAULT_NORMALIZED_HEAT_FLUX_BOUND = 128.0
FIXED_INPUT_SCALING = "fixed_training_scale"
DYNAMIC_INPUT_SCALING = "current_density_rms_arcsinh"
INPUT_SCALING_KINDS = frozenset((FIXED_INPUT_SCALING, DYNAMIC_INPUT_SCALING))


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
        physical_amplitude = _rms_with_floor(fields[:, 0], dynamic_amplitude_floor)
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


def _window_fno_features(
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
    compact_latent: Array | None = None,
) -> Tuple[Array, Array]:
    """Return FNO features and the physical amplitude used for output scaling."""
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
    current_amplitude = jnp.asarray(amplitude, dtype=state.dtype)
    if input_scaling == DYNAMIC_INPUT_SCALING:
        current_fields = primitive_fields(state, k_arr, poisson_sign=poisson_sign)
        current_amplitude = _rms_with_floor(
            current_fields[:, 0], dynamic_amplitude_floor
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
    if "amplitude_local" in params:
        log_amplitude = jnp.log(jnp.maximum(current_amplitude, 1e-8))
        normalized_log_amplitude = (
            log_amplitude - jnp.asarray(amplitude_center, dtype=state.dtype)
        ) / jnp.asarray(max(float(amplitude_scale), 1e-8), dtype=state.dtype)
        encoded = encoded + (
            normalized_log_amplitude[:, None, None]
            * params["amplitude_local"][:, 0][None, :, None]
        )
    if compact_latent is not None and "compact_latent_local" in params:
        compact_latent = jnp.asarray(compact_latent, dtype=state.dtype)
        encoded = encoded + spectral_channel_operator(
            compact_latent,
            params["compact_latent_local"],
            params["compact_latent_spectral_real"],
            params["compact_latent_spectral_imag"],
        )
    activation = jax.nn.gelu if "activation_gelu" in params else jax.nn.silu
    encoded = activation(encoded)
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
            features = features + residual_scale * activation(update)
            block += 1
    else:
        mixed = spectral_channel_operator(
            encoded,
            params["mixer_local"],
            params["mixer_spectral_real"],
            params["mixer_spectral_imag"],
        )
        features = encoded + activation(mixed)
    return features, current_amplitude


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
            normalization_amplitude = _rms_with_floor(
                fields[:, -1, 0], dynamic_amplitude_floor
            )
        else:
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
