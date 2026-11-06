"""Parameter initializers for spectral-memory, window, and latent closures."""

from __future__ import annotations

from typing import Dict
import math

from .._jax_precision import enable_x64

enable_x64()

import jax
import jax.numpy as jnp

Array = jax.Array


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
    output_scale = 1.0e-4 / math.sqrt(float(width))
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
        # Keep the initial closure close to the unclosed fluid model while
        # ensuring every trainable neural component starts from the recorded
        # random seed.
        "output_local": output_scale
        * jax.random.normal(next(keys), (1, width), dtype=dtype),
        "output_spectral_real": output_scale
        * jax.random.normal(next(keys), (modes, 1, width), dtype=dtype),
        "output_spectral_imag": output_scale
        * jax.random.normal(next(keys), (modes, 1, width), dtype=dtype),
        "amplitude_gain": 1.0e-5
        * jax.random.normal(next(keys), (channels,), dtype=dtype),
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


def init_burles_latent_fno_params(
    key: Array,
    *,
    width: int,
    spectral_modes: int,
    memory_steps: int,
    depth: int = 8,
    latent_dim: int = 6,
    input_channels: int = 5,
    dtype=jnp.float32,
) -> Dict[str, Array]:
    """Initialize a Burles-style history FNO with compact learned memory.

    ``latent_dim=0`` is the matched history-only control.  Positive dimensions
    add a learned state-space realization of unresolved memory; no Hermite
    basis, projected kinetic propagator, or prescribed latent dynamics enter.
    """
    latent_dim = int(latent_dim)
    if latent_dim < 0:
        raise ValueError("latent_dim cannot be negative")
    params = init_window_fno_params(
        key,
        width=width,
        spectral_modes=spectral_modes,
        memory_steps=memory_steps,
        depth=depth,
        input_channels=input_channels,
        dtype=dtype,
    )
    # Preserve old SiLU checkpoints while matching the GELU nonlinearity in
    # newly initialized Burles-family FNOs.
    params["activation_gelu"] = jnp.zeros((0,), dtype=dtype)
    width = int(width)
    modes = int(spectral_modes)
    extra = iter(jax.random.split(jax.random.fold_in(key, 0xB171E5), 14))
    # Burles et al. append log(A_scale) as a spatially constant channel.  Keep
    # it explicit rather than relying only on multiplicative normalization.
    params["amplitude_local"] = (0.12 / math.sqrt(float(width))) * jax.random.normal(
        next(extra), (width, 1), dtype=dtype
    )
    latent_lift_scale = 0.08 / math.sqrt(float(max(latent_dim, 1)))
    params["compact_latent_local"] = latent_lift_scale * jax.random.normal(
        next(extra), (width, latent_dim), dtype=dtype
    )
    params["compact_latent_spectral_real"] = latent_lift_scale * jax.random.normal(
        next(extra), (modes, width, latent_dim), dtype=dtype
    )
    params["compact_latent_spectral_imag"] = latent_lift_scale * jax.random.normal(
        next(extra), (modes, width, latent_dim), dtype=dtype
    )
    state_scale = 0.04 / math.sqrt(float(width))
    params["compact_latent_init_local"] = state_scale * jax.random.normal(
        next(extra), (latent_dim, width), dtype=dtype
    )
    params["compact_latent_init_spectral_real"] = state_scale * jax.random.normal(
        next(extra), (modes, latent_dim, width), dtype=dtype
    )
    params["compact_latent_init_spectral_imag"] = state_scale * jax.random.normal(
        next(extra), (modes, latent_dim, width), dtype=dtype
    )
    params["compact_latent_rhs_local"] = state_scale * jax.random.normal(
        next(extra), (latent_dim, width), dtype=dtype
    )
    params["compact_latent_rhs_spectral_real"] = state_scale * jax.random.normal(
        next(extra), (modes, latent_dim, width), dtype=dtype
    )
    params["compact_latent_rhs_spectral_imag"] = state_scale * jax.random.normal(
        next(extra), (modes, latent_dim, width), dtype=dtype
    )
    # A stable random diagonal decay leaves all compact-memory dynamics
    # trainable while avoiding an unstable random ODE at epoch zero.
    target_decay = jnp.asarray(0.25, dtype=dtype)
    inverse_softplus = jnp.log(jnp.expm1(target_decay))
    params["compact_latent_decay_raw"] = inverse_softplus + 0.02 * jax.random.normal(
        next(extra), (latent_dim,), dtype=dtype
    )
    # Unlike the earlier near-zero pilot head, this scale permits gradients to
    # reach the deep FNO from the first update while remaining bounded.
    output_scale = 0.04 / math.sqrt(float(width))
    params["output_local"] = output_scale * jax.random.normal(
        next(extra), (1, width), dtype=dtype
    )
    params["output_spectral_real"] = output_scale * jax.random.normal(
        next(extra), (modes, 1, width), dtype=dtype
    )
    params["output_spectral_imag"] = output_scale * jax.random.normal(
        next(extra), (modes, 1, width), dtype=dtype
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
