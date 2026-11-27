"""Spatial spectral layers, causal temporal convolution, and RMS scaling."""

from __future__ import annotations

from .._jax_precision import enable_x64

enable_x64()

import jax
import jax.numpy as jnp

Array = jax.Array


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


def _rms_with_floor(values: Array, floor: float) -> Array:
    """Floor RMS without differentiating sqrt at zero or changing floor rounding."""
    values = jnp.asarray(values)
    mean_square = jnp.mean(jnp.square(values), axis=-1)
    floor_value = jnp.asarray(floor, dtype=values.dtype)
    # Keep the original rounded comparison: sqrt(max(mean_square, floor**2))
    # can change the forward value or threshold when floor**2 rounds.
    # This sqrt only selects a branch; the differentiated sqrt is guarded below.
    below_floor = jnp.sqrt(jax.lax.stop_gradient(mean_square)) < floor_value
    zero = mean_square == 0.0
    safe_mean_square = jnp.where(below_floor | zero, 1.0, mean_square)
    rms = jnp.sqrt(safe_mean_square)
    # A zero floor stays zero, with the zero subgradient of RMS at the origin.
    rms = jnp.where(zero, 0.0, rms)
    # Retain maximum's original tie convention at an exactly attained floor.
    return jnp.maximum(jnp.where(below_floor, floor_value, rms), floor_value)
