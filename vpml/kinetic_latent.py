"""Compact kinetic latent-state dynamics for low-moment closures."""

from __future__ import annotations

import math
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from vpml.low_moment import (
    electric_field_from_density,
    low_moment_rhs,
    low_moment_rk4_step,
    spectral_derivative,
)


Array = jax.Array


def _periodic_convolution(values: Array, kernel: Array, bias: Array) -> Array:
    values = jnp.asarray(values)
    kernel = jnp.asarray(kernel, dtype=values.dtype)
    radius = int(kernel.shape[-1]) // 2
    if int(kernel.shape[-1]) % 2 != 1:
        raise ValueError("periodic convolution kernels must have odd width")
    if radius:
        values = jnp.concatenate(
            (values[..., -radius:], values, values[..., :radius]),
            axis=-1,
        )
    result = jax.lax.conv_general_dilated(
        values,
        kernel,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
    )
    return result + jnp.asarray(bias, dtype=result.dtype)[None, :, None]


def init_kinetic_latent_dynamics(
    key: Array,
    *,
    resolved_channels: int,
    latent_rank: int,
    width: int,
    depth: int,
    kernel_size: int = 5,
    output_channels: int | None = None,
    output_init_scale: float = 1e-2,
    dtype=jnp.float32,
) -> Dict[str, Array]:
    """Initialize a periodic residual CNN for one latent time increment."""
    if output_channels is None:
        output_channels = latent_rank
    if min(
        resolved_channels,
        latent_rank,
        width,
        depth,
        kernel_size,
        output_channels,
    ) <= 0:
        raise ValueError("all model dimensions must be positive")
    if kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd")
    keys = iter(jax.random.split(key, depth + 2))

    def kernel(k, output_channels: int, input_channels: int, scale: float = 1.0):
        std = scale * math.sqrt(2.0 / float(input_channels * kernel_size))
        return std * jax.random.normal(
            k,
            (output_channels, input_channels, kernel_size),
            dtype=dtype,
        )

    input_channels = int(resolved_channels) + int(latent_rank)
    params: Dict[str, Array] = {
        "lift_kernel": kernel(next(keys), width, input_channels),
        "lift_bias": jnp.zeros((width,), dtype=dtype),
    }
    for block in range(depth):
        params[f"block_{block}_kernel"] = kernel(next(keys), width, width)
        params[f"block_{block}_bias"] = jnp.zeros((width,), dtype=dtype)
    if float(output_init_scale) < 0.0:
        raise ValueError("output_init_scale must be nonnegative")
    params["output_kernel"] = kernel(
        next(keys),
        output_channels,
        width,
        scale=float(output_init_scale),
    )
    params["output_bias"] = jnp.zeros((output_channels,), dtype=dtype)
    return params


def init_state_conditioned_latent_operator(
    key: Array,
    *,
    resolved_channels: int,
    latent_rank: int,
    width: int,
    depth: int,
    spectral_modes: int,
    operator_rank: int,
    kernel_size: int = 5,
    output_projection_init_scale: float = 0.0,
    conditioner_output_channels: int | None = None,
    conditioner_output_init_scale: float = 1e-2,
    closure_aligned_output: bool = False,
    latent_delay_input: bool = False,
    dtype=jnp.float32,
) -> Dict[str, Array]:
    """Initialize a low-rank, state-conditioned spectral propagator update."""
    if min(spectral_modes, operator_rank) <= 0:
        raise ValueError("spectral_modes and operator_rank must be positive")
    if float(output_projection_init_scale) < 0.0:
        raise ValueError("output_projection_init_scale must be nonnegative")
    conditioner_key, projection_key = jax.random.split(key)
    params = init_kinetic_latent_dynamics(
        conditioner_key,
        resolved_channels=(
            int(resolved_channels) + int(latent_rank)
            if latent_delay_input
            else int(resolved_channels)
        ),
        latent_rank=latent_rank,
        width=width,
        depth=depth,
        kernel_size=kernel_size,
        output_channels=(
            int(operator_rank)
            if conditioner_output_channels is None
            else int(conditioner_output_channels)
        ),
        output_init_scale=float(conditioner_output_init_scale),
        dtype=dtype,
    )
    input_channels = int(resolved_channels) + int(latent_rank)
    if latent_delay_input:
        input_channels += int(latent_rank)
    (
        projection_real_key,
        projection_imag_key,
        output_real_key,
        output_imag_key,
    ) = jax.random.split(projection_key, 4)
    projection_scale = math.sqrt(1.0 / float(input_channels))
    shape_v = (spectral_modes, input_channels, operator_rank)
    shape_u = (spectral_modes, operator_rank, latent_rank)
    params["operator_v_real"] = projection_scale * jax.random.normal(
        projection_real_key,
        shape_v,
        dtype=dtype,
    )
    params["operator_v_imag"] = projection_scale * jax.random.normal(
        projection_imag_key,
        shape_v,
        dtype=dtype,
    )
    output_scale = float(output_projection_init_scale) / math.sqrt(
        float(operator_rank)
    )
    params["operator_u_real"] = output_scale * jax.random.normal(
        output_real_key,
        shape_u,
        dtype=dtype,
    )
    params["operator_u_imag"] = output_scale * jax.random.normal(
        output_imag_key,
        shape_u,
        dtype=dtype,
    )
    if closure_aligned_output:
        closure_real_key, closure_imag_key = jax.random.split(output_imag_key)
        closure_shape = (spectral_modes, operator_rank)
        params["operator_u_closure_real"] = output_scale * jax.random.normal(
            closure_real_key,
            closure_shape,
            dtype=dtype,
        )
        params["operator_u_closure_imag"] = output_scale * jax.random.normal(
            closure_imag_key,
            closure_shape,
            dtype=dtype,
        )
    return params


def bounded_normalized_latent_correction(
    correction: Array,
    correction_bounds: Array,
) -> Array:
    """Apply a smooth channelwise trust bound to a normalized correction."""
    correction = jnp.asarray(correction)
    bounds = jnp.asarray(correction_bounds, dtype=correction.dtype)
    if bounds.ndim != 1 or int(bounds.shape[0]) != int(correction.shape[-2]):
        raise ValueError("correction_bounds must match the latent channels")
    bounds = bounds.reshape((1,) * (correction.ndim - 2) + (-1, 1))
    return bounds * jnp.tanh(correction / bounds)


def bounded_normalized_closure_correction(
    correction: Array,
    bound: float | Array,
) -> Array:
    """Apply a smooth scalar trust bound to a normalized C3 update."""
    correction = jnp.asarray(correction)
    bound = jnp.asarray(bound, dtype=correction.dtype)
    return bound * jnp.tanh(correction / bound)


def stabilize_linear_latent_propagator(
    propagator: np.ndarray,
    *,
    resolved_channels: int,
    maximum_spectral_radius: float,
) -> tuple[np.ndarray, list[dict]]:
    """Scale unstable modal latent blocks without changing resolved forcing."""
    if not 0.0 < maximum_spectral_radius <= 1.0:
        raise ValueError("maximum_spectral_radius must lie in (0, 1]")
    stabilized = np.asarray(propagator).copy()
    diagnostics = []
    for mode in range(stabilized.shape[0]):
        latent_block = stabilized[mode, resolved_channels:, :]
        spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(latent_block)))
        )
        factor = min(1.0, maximum_spectral_radius / spectral_radius)
        if factor < 1.0:
            stabilized[mode, resolved_channels:, :] *= factor
            diagnostics.append(
                {
                    "mode": mode,
                    "original_spectral_radius": spectral_radius,
                    "scale_factor": factor,
                }
            )
    return stabilized, diagnostics


def stabilize_coupled_linear_propagator(
    propagator: np.ndarray,
    *,
    maximum_spectral_radius: float,
) -> tuple[np.ndarray, list[dict]]:
    """Project every complete modal state transition into the stable disk."""
    if not 0.0 < maximum_spectral_radius <= 1.0:
        raise ValueError("maximum_spectral_radius must lie in (0, 1]")
    stabilized = np.asarray(propagator, dtype=np.complex128).copy()
    diagnostics = []
    for mode in range(stabilized.shape[0]):
        eigenvalues, eigenvectors = np.linalg.eig(stabilized[mode])
        magnitudes = np.abs(eigenvalues)
        adjusted = magnitudes > maximum_spectral_radius
        if np.any(adjusted):
            projected = eigenvalues.copy()
            projected[adjusted] *= maximum_spectral_radius / magnitudes[adjusted]
            stabilized[mode] = (
                eigenvectors @ np.diag(projected) @ np.linalg.inv(eigenvectors)
            )
        final_radius = float(
            np.max(np.abs(np.linalg.eigvals(stabilized[mode])))
        )
        diagnostics.append(
            {
                "mode": mode,
                "original_spectral_radius": float(np.max(magnitudes)),
                "final_spectral_radius": final_radius,
                "adjusted_eigenvalues": int(np.sum(adjusted)),
            }
        )
    return stabilized.astype(np.asarray(propagator).dtype), diagnostics


def kinetic_latent_dynamics_step(
    params: Dict[str, Array],
    resolved_state: Array,
    latent_state: Array,
    *,
    depth: int,
) -> Array:
    """Advance one normalized latent step conditioned on normalized moments."""
    return latent_state + kinetic_latent_dynamics_correction(
        params,
        resolved_state,
        latent_state,
        depth=depth,
    )


def kinetic_latent_dynamics_correction(
    params: Dict[str, Array],
    resolved_state: Array,
    latent_state: Array,
    *,
    depth: int,
    zero_bias: bool = False,
) -> Array:
    """Return the normalized nonlinear latent increment."""
    inputs = jnp.concatenate((resolved_state, latent_state), axis=1)
    hidden = jax.nn.gelu(
        _periodic_convolution(
            inputs,
            params["lift_kernel"],
            (
                jnp.zeros_like(params["lift_bias"])
                if zero_bias
                else params["lift_bias"]
            ),
        )
    )
    for block in range(int(depth)):
        update = jax.nn.gelu(
            _periodic_convolution(
                hidden,
                params[f"block_{block}_kernel"],
                (
                    jnp.zeros_like(params[f"block_{block}_bias"])
                    if zero_bias
                    else params[f"block_{block}_bias"]
                ),
            )
        )
        hidden = (hidden + update) / math.sqrt(2.0)
    return _periodic_convolution(
        hidden,
        params["output_kernel"],
        (
            jnp.zeros_like(params["output_bias"])
            if zero_bias
            else params["output_bias"]
        ),
    )


def equilibrium_preserving_latent_cnn_correction(
    params: Dict[str, Array],
    normalized_resolved: Array,
    normalized_latent: Array,
    *,
    depth: int,
    normalized_latent_delta: Array | None = None,
) -> Array:
    """Return the direct CNN residual after removing its affine response."""
    conditioned_resolved = normalized_resolved
    if normalized_latent_delta is not None:
        conditioned_resolved = jnp.concatenate(
            (conditioned_resolved, normalized_latent_delta), axis=1
        )
    zero_resolved = jnp.zeros_like(conditioned_resolved)
    zero_latent = jnp.zeros_like(normalized_latent)

    def network(resolved, latent):
        return kinetic_latent_dynamics_correction(
            params,
            resolved,
            latent,
            depth=depth,
            zero_bias=False,
        )

    base = network(zero_resolved, zero_latent)
    _, linear = jax.jvp(
        network,
        (zero_resolved, zero_latent),
        (conditioned_resolved, normalized_latent),
    )
    return network(conditioned_resolved, normalized_latent) - base - linear


def apply_linear_latent_propagator(
    propagator: Array,
    resolved_state: Array,
    latent_state: Array,
) -> Array:
    """Apply a mode-resolved train-only linear latent propagator."""
    resolved_state = jnp.asarray(resolved_state)
    latent_state = jnp.asarray(latent_state, dtype=resolved_state.dtype)
    inputs = jnp.concatenate((resolved_state, latent_state), axis=1)
    inputs_hat = jnp.fft.rfft(inputs, axis=-1, norm="forward")
    updated_hat = jnp.einsum(
        "bck,kcr->brk",
        inputs_hat,
        jnp.asarray(propagator, dtype=inputs_hat.dtype),
    )
    return jnp.fft.irfft(
        updated_hat,
        n=inputs.shape[-1],
        axis=-1,
        norm="forward",
    )


def apply_coupled_linear_propagator(
    propagator: Array,
    resolved_state: Array,
    latent_state: Array,
    *,
    resolved_scale: Array,
    latent_scale: Array,
) -> Array:
    """Apply a stable modal transition to the complete normalized state."""
    resolved_state = jnp.asarray(resolved_state)
    latent_state = jnp.asarray(latent_state, dtype=resolved_state.dtype)
    scale = jnp.concatenate(
        (
            jnp.asarray(resolved_scale, dtype=resolved_state.dtype),
            jnp.asarray(latent_scale, dtype=resolved_state.dtype),
        )
    )
    normalized = jnp.concatenate((resolved_state, latent_state), axis=1)
    normalized = normalized / scale[None, :, None]
    normalized_hat = jnp.fft.rfft(normalized, axis=-1, norm="forward")
    updated_hat = jnp.einsum(
        "bck,kcd->bdk",
        normalized_hat,
        jnp.asarray(propagator, dtype=normalized_hat.dtype),
    )
    updated = jnp.fft.irfft(
        updated_hat,
        n=normalized.shape[-1],
        axis=-1,
        norm="forward",
    )
    return updated * scale[None, :, None]


def state_conditioned_latent_operator_features(
    params: Dict[str, Array],
    normalized_resolved: Array,
    normalized_latent: Array,
    *,
    depth: int,
    equilibrium_preserving: bool = False,
    normalized_latent_delta: Array | None = None,
) -> Array:
    """Return the conditioned low-rank spectral features before decoding."""
    conditioned_resolved = normalized_resolved
    if normalized_latent_delta is not None:
        conditioned_resolved = jnp.concatenate(
            (conditioned_resolved, normalized_latent_delta), axis=1
        )
    inputs = jnp.concatenate((conditioned_resolved, normalized_latent), axis=1)
    inputs_hat = jnp.fft.rfft(inputs, axis=-1, norm="forward")
    spectral_modes = min(
        int(inputs_hat.shape[-1]),
        int(params["operator_v_real"].shape[0]),
    )
    operator_v = jax.lax.complex(
        params["operator_v_real"][:spectral_modes],
        params["operator_v_imag"][:spectral_modes],
    )
    projected_hat = jnp.einsum(
        "bck,kcj->bjk",
        inputs_hat[:, :, :spectral_modes],
        operator_v,
    )
    projected = jnp.fft.irfft(
        projected_hat,
        n=inputs.shape[-1],
        axis=-1,
        norm="forward",
    )
    modulation = jnp.tanh(
        kinetic_latent_dynamics_correction(
            params,
            conditioned_resolved,
            normalized_latent,
            depth=depth,
            zero_bias=equilibrium_preserving,
        )
    )
    modulated_hat = jnp.fft.rfft(
        modulation * projected,
        axis=-1,
        norm="forward",
    )
    return modulated_hat[:, :, :spectral_modes]


def state_conditioned_latent_operator_correction(
    params: Dict[str, Array],
    normalized_resolved: Array,
    normalized_latent: Array,
    *,
    depth: int,
    equilibrium_preserving: bool = False,
    normalized_latent_delta: Array | None = None,
) -> Array:
    """Apply a translation-equivariant low-rank multiplicative correction."""
    conditioned_resolved = normalized_resolved
    if normalized_latent_delta is not None:
        conditioned_resolved = jnp.concatenate(
            (conditioned_resolved, normalized_latent_delta), axis=1
        )
    inputs = jnp.concatenate((conditioned_resolved, normalized_latent), axis=1)
    inputs_hat = jnp.fft.rfft(inputs, axis=-1, norm="forward")
    spectral_modes = min(
        int(inputs_hat.shape[-1]),
        int(params["operator_v_real"].shape[0]),
    )
    modulated_hat = state_conditioned_latent_operator_features(
        params,
        normalized_resolved,
        normalized_latent,
        depth=depth,
        equilibrium_preserving=equilibrium_preserving,
        normalized_latent_delta=normalized_latent_delta,
    )
    operator_u = jax.lax.complex(
        params["operator_u_real"][:spectral_modes],
        params["operator_u_imag"][:spectral_modes],
    )
    correction_hat = jnp.einsum(
        "bjk,kjr->brk",
        modulated_hat[:, :, :spectral_modes],
        operator_u,
    )
    full_correction_hat = jnp.zeros(
        (
            inputs.shape[0],
            normalized_latent.shape[1],
            inputs_hat.shape[-1],
        ),
        dtype=correction_hat.dtype,
    ).at[:, :, :spectral_modes].set(correction_hat)
    return jnp.fft.irfft(
        full_correction_hat,
        n=inputs.shape[-1],
        axis=-1,
        norm="forward",
    )


def state_conditioned_closure_correction(
    params: Dict[str, Array],
    normalized_resolved: Array,
    normalized_latent: Array,
    *,
    depth: int,
    equilibrium_preserving: bool = False,
    normalized_latent_delta: Array | None = None,
) -> Array:
    """Decode the nonlinear update in the observable C3 direction."""
    conditioned_resolved = normalized_resolved
    if normalized_latent_delta is not None:
        conditioned_resolved = jnp.concatenate(
            (conditioned_resolved, normalized_latent_delta), axis=1
        )
    inputs = jnp.concatenate((conditioned_resolved, normalized_latent), axis=1)
    inputs_hat = jnp.fft.rfft(inputs, axis=-1, norm="forward")
    spectral_modes = min(
        int(inputs_hat.shape[-1]),
        int(params["operator_u_closure_real"].shape[0]),
    )
    modulated_hat = state_conditioned_latent_operator_features(
        params,
        normalized_resolved,
        normalized_latent,
        depth=depth,
        equilibrium_preserving=equilibrium_preserving,
        normalized_latent_delta=normalized_latent_delta,
    )
    closure_u = jax.lax.complex(
        params["operator_u_closure_real"][:spectral_modes],
        params["operator_u_closure_imag"][:spectral_modes],
    )
    correction_hat = jnp.einsum(
        "bjk,kj->bk",
        modulated_hat[:, :, :spectral_modes],
        closure_u,
    )
    full_correction_hat = jnp.zeros_like(inputs_hat[:, 0]).at[
        :, :spectral_modes
    ].set(correction_hat)
    return jnp.fft.irfft(
        full_correction_hat,
        n=inputs.shape[-1],
        axis=-1,
        norm="forward",
    )


def gated_spectral_closure_correction(
    params: Dict[str, Array],
    normalized_resolved: Array,
    normalized_latent: Array,
    *,
    gate_scale: float | Array,
    gate_power: int = 2,
) -> Array:
    """Apply a mode-wise latent readout that vanishes at linear order."""
    inputs = jnp.concatenate((normalized_resolved, normalized_latent), axis=1)
    latent_amplitude = jnp.sqrt(
        jnp.mean(jnp.square(normalized_latent), axis=(1, 2))
    )
    gate_scale = jnp.asarray(gate_scale, dtype=inputs.dtype)
    gate = jnp.power(latent_amplitude, int(gate_power)) / (
        jnp.power(latent_amplitude, int(gate_power))
        + jnp.power(gate_scale, int(gate_power))
    )
    gated_inputs = gate[:, None, None] * inputs
    inputs_hat = jnp.fft.rfft(gated_inputs, axis=-1, norm="forward")
    spectral_modes = min(
        int(inputs_hat.shape[-1]),
        int(params["operator_u_gated_closure_real"].shape[0]),
    )
    readout = jax.lax.complex(
        params["operator_u_gated_closure_real"][:spectral_modes],
        params["operator_u_gated_closure_imag"][:spectral_modes],
    )
    correction_hat = jnp.einsum(
        "bck,kc->bk",
        inputs_hat[:, :, :spectral_modes],
        readout,
    )
    full_correction_hat = jnp.zeros_like(inputs_hat[:, 0]).at[
        :, :spectral_modes
    ].set(correction_hat)
    return jnp.fft.irfft(
        full_correction_hat,
        n=inputs.shape[-1],
        axis=-1,
        norm="forward",
    )


def gated_spectral_latent_correction(
    params: Dict[str, Array],
    normalized_resolved: Array,
    normalized_latent: Array,
    *,
    gate_scale: float | Array,
    gate_power: int = 2,
) -> Array:
    """Apply a mode-wise update to every kinetic latent coordinate."""
    inputs = jnp.concatenate((normalized_resolved, normalized_latent), axis=1)
    latent_amplitude = jnp.sqrt(
        jnp.mean(jnp.square(normalized_latent), axis=(1, 2))
    )
    gate_scale = jnp.asarray(gate_scale, dtype=inputs.dtype)
    gate = jnp.power(latent_amplitude, int(gate_power)) / (
        jnp.power(latent_amplitude, int(gate_power))
        + jnp.power(gate_scale, int(gate_power))
    )
    inputs_hat = jnp.fft.rfft(
        gate[:, None, None] * inputs,
        axis=-1,
        norm="forward",
    )
    spectral_modes = min(
        int(inputs_hat.shape[-1]),
        int(params["operator_u_gated_latent_real"].shape[0]),
    )
    readout = jax.lax.complex(
        params["operator_u_gated_latent_real"][:spectral_modes],
        params["operator_u_gated_latent_imag"][:spectral_modes],
    )
    correction_hat = jnp.einsum(
        "bck,kcr->brk",
        inputs_hat[:, :, :spectral_modes],
        readout,
    )
    full_correction_hat = jnp.zeros(
        (
            inputs.shape[0],
            normalized_latent.shape[1],
            inputs_hat.shape[-1],
        ),
        dtype=correction_hat.dtype,
    ).at[:, :, :spectral_modes].set(correction_hat)
    return jnp.fft.irfft(
        full_correction_hat,
        n=inputs.shape[-1],
        axis=-1,
        norm="forward",
    )


def closure_orthogonal_latent_correction(
    correction: Array,
    basis: Array,
) -> Array:
    """Remove the component of a latent update visible through C3."""
    correction = jnp.asarray(correction)
    readout = jnp.asarray(basis, dtype=correction.dtype)[0]
    denominator = jnp.sum(jnp.square(readout))
    visible = jnp.einsum("r,...rx->...x", readout, correction)
    return correction - jnp.einsum(
        "r,...x->...rx", readout / denominator, visible
    )


def closure_aligned_latent_correction(
    closure_correction: Array,
    basis: Array,
) -> Array:
    """Lift a C3 update into the minimum-norm latent direction."""
    closure_correction = jnp.asarray(closure_correction)
    readout = jnp.asarray(basis, dtype=closure_correction.dtype)[0]
    denominator = jnp.sum(jnp.square(readout))
    return jnp.einsum(
        "r,...x->...rx", readout / denominator, closure_correction
    )


def state_conditioned_kinetic_latent_step(
    params: Dict[str, Array],
    resolved_state: Array,
    latent_state: Array,
    propagator: Array,
    *,
    resolved_center: Array,
    resolved_scale: Array,
    latent_center: Array,
    latent_scale: Array,
    depth: int,
    dynamics_model: str = "additive_residual",
    normalized_correction_bound: float | None = None,
    normalized_state_bound: float | None = None,
) -> Array:
    """Advance fixed linear phase mixing plus a state-conditioned update."""
    resolved_center = jnp.asarray(resolved_center, dtype=resolved_state.dtype)
    resolved_scale = jnp.asarray(resolved_scale, dtype=resolved_state.dtype)
    latent_center = jnp.asarray(latent_center, dtype=latent_state.dtype)
    latent_scale = jnp.asarray(latent_scale, dtype=latent_state.dtype)
    linear = apply_linear_latent_propagator(
        propagator,
        resolved_state,
        latent_state,
    )
    normalized_resolved = (
        resolved_state - resolved_center[None, :, None]
    ) / resolved_scale[None, :, None]
    normalized_latent = (
        latent_state - latent_center[None, :, None]
    ) / latent_scale[None, :, None]
    if dynamics_model == "additive_residual":
        correction = kinetic_latent_dynamics_correction(
            params,
            normalized_resolved,
            normalized_latent,
            depth=depth,
        )
    elif dynamics_model == "multiplicative_operator":
        correction = state_conditioned_latent_operator_correction(
            params,
            normalized_resolved,
            normalized_latent,
            depth=depth,
        )
    else:
        raise ValueError(f"Unknown latent dynamics model: {dynamics_model}")
    if normalized_correction_bound is not None:
        bound = jnp.asarray(normalized_correction_bound, dtype=correction.dtype)
        correction = bound * jnp.tanh(correction / bound)
    density_amplitude = jnp.sqrt(
        jnp.mean(jnp.square(resolved_state[:, 0]), axis=-1)
    )
    nonlinear_gate = jnp.tanh(
        jnp.square(density_amplitude / resolved_scale[0])
    )
    updated = linear + (
        nonlinear_gate[:, None, None]
        * correction
        * latent_scale[None, :, None]
    )
    if normalized_state_bound is not None:
        state_bound = jnp.asarray(normalized_state_bound, dtype=updated.dtype)
        normalized_updated = (
            updated - latent_center[None, :, None]
        ) / latent_scale[None, :, None]
        updated = latent_center[None, :, None] + latent_scale[None, :, None] * jnp.clip(
            normalized_updated, -state_bound, state_bound
        )
    return updated.astype(latent_state.dtype)


def rollout_state_conditioned_kinetic_latent_dynamics(
    params: Dict[str, Array],
    resolved_history: Array,
    initial_latent: Array,
    propagator: Array,
    *,
    resolved_center: Array,
    resolved_scale: Array,
    latent_center: Array,
    latent_scale: Array,
    depth: int,
    dynamics_model: str = "additive_residual",
) -> Array:
    """Roll out the structured latent dynamics with supplied resolved fields."""
    def body(latent, resolved):
        updated = state_conditioned_kinetic_latent_step(
            params,
            resolved,
            latent,
            propagator,
            resolved_center=resolved_center,
            resolved_scale=resolved_scale,
            latent_center=latent_center,
            latent_scale=latent_scale,
            depth=depth,
            dynamics_model=dynamics_model,
        )
        return updated, updated

    _, trajectory = jax.lax.scan(
        body,
        initial_latent,
        jnp.swapaxes(resolved_history, 0, 1),
    )
    return jnp.swapaxes(trajectory, 0, 1)


def rollout_kinetic_latent_dynamics(
    params: Dict[str, Array],
    resolved_history: Array,
    initial_latent: Array,
    *,
    depth: int,
) -> Array:
    """Roll out latent state while conditioning on a supplied resolved history."""
    resolved_history = jnp.asarray(resolved_history)

    def body(latent, resolved):
        updated = kinetic_latent_dynamics_step(
            params,
            resolved,
            latent,
            depth=depth,
        )
        return updated, updated

    _, trajectory = jax.lax.scan(
        body,
        initial_latent,
        jnp.swapaxes(resolved_history, 0, 1),
    )
    return jnp.swapaxes(trajectory, 0, 1)


def reconstruct_first_unresolved_field(latent_state: Array, basis: Array) -> Array:
    """Read out the first unresolved Hermite field from POD coordinates."""
    latent_state = jnp.asarray(latent_state)
    basis = jnp.asarray(basis, dtype=latent_state.dtype)
    if basis.ndim != 2 or basis.shape[1] != latent_state.shape[-2]:
        raise ValueError("basis rank must match latent channels")
    return jnp.einsum("r,...rx->...x", basis[0], latent_state)


def resolved_hermite_to_low_moment_state(resolved_state: Array) -> Array:
    """Convert physical C0:C2 fields to the conservative three-moment state."""
    resolved_state = jnp.asarray(resolved_state)
    if resolved_state.shape[-2] != 3:
        raise ValueError("resolved_state must contain C0, C1, and C2")
    density = resolved_state[..., 0, :]
    momentum = resolved_state[..., 1, :]
    second = density + math.sqrt(2.0) * resolved_state[..., 2, :]
    return jnp.stack((density, momentum, second), axis=-2)


def low_moment_state_to_resolved_hermite(state: Array) -> Array:
    """Convert the conservative three-moment state to physical C0:C2 fields."""
    state = jnp.asarray(state)
    if state.shape[-2] != 3:
        raise ValueError("state must contain density, momentum, and second moment")
    density = state[..., 0, :]
    momentum = state[..., 1, :]
    second_hermite = (state[..., 2, :] - density) / math.sqrt(2.0)
    return jnp.stack((density, momentum, second_hermite), axis=-2)


def latent_heat_flux_gradient(
    state: Array,
    latent_state: Array,
    basis: Array,
    k_arr: Array,
) -> Array:
    """Decode C3 and return the spatial derivative of central heat flux."""
    state = jnp.asarray(state)
    latent_state = jnp.asarray(latent_state, dtype=state.dtype)
    c3 = reconstruct_first_unresolved_field(latent_state, basis)
    density = 1.0 + state[:, 0]
    momentum = state[:, 1]
    velocity = momentum / density
    pressure = 1.0 + state[:, 2] - momentum * velocity
    raw_third = math.sqrt(6.0) * c3 + 3.0 * momentum
    heat_flux = raw_third - 3.0 * velocity * pressure - density * velocity**3
    heat_flux_hat = jnp.fft.rfft(heat_flux, axis=-1)
    gradient_hat = 1j * jnp.asarray(k_arr, dtype=heat_flux_hat.real.dtype) * heat_flux_hat
    gradient = jnp.fft.irfft(gradient_hat, n=state.shape[-1], axis=-1)
    return gradient - jnp.mean(gradient, axis=-1, keepdims=True)


def _dealias_state(values: Array) -> Array:
    coefficients = jnp.fft.rfft(values, axis=-1)
    cutoff = int(values.shape[-1]) // 3
    mask = jnp.arange(coefficients.shape[-1]) <= cutoff
    return jnp.fft.irfft(
        coefficients * mask,
        n=values.shape[-1],
        axis=-1,
    ).astype(values.dtype)


def _linear_low_moment_step(
    state: Array,
    first_unresolved: Array,
    k_arr: Array,
    dt: float,
    *,
    poisson_sign: float,
) -> Array:
    """Advance the equilibrium linearization of the three-moment equations."""
    heat_flux_gradient = math.sqrt(6.0) * spectral_derivative(
        first_unresolved, k_arr
    )

    def rhs(value):
        density = value[:, 0]
        momentum = value[:, 1]
        second = value[:, 2]
        field = electric_field_from_density(
            density, k_arr, poisson_sign=poisson_sign
        )
        return jnp.stack(
            (
                -spectral_derivative(momentum, k_arr),
                -spectral_derivative(second, k_arr) - field,
                -3.0 * spectral_derivative(momentum, k_arr)
                - heat_flux_gradient,
            ),
            axis=1,
        )

    dt_value = jnp.asarray(dt, dtype=state.dtype)
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt_value * k1)
    k3 = rhs(state + 0.5 * dt_value * k2)
    k4 = rhs(state + dt_value * k3)
    return _dealias_state(
        state + (dt_value / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    )


def projected_hermite_latent_rhs(
    state: Array,
    latent_state: Array,
    basis: Array,
    k_arr: Array,
    *,
    tail_damping: float,
    tail_power: float,
    poisson_sign: float = 1.0,
) -> Array:
    """Project nonlinear Vlasov Hermite-tail dynamics into the latent basis."""
    state = jnp.asarray(state)
    latent_state = jnp.asarray(latent_state, dtype=state.dtype)
    basis = jnp.asarray(basis, dtype=state.dtype)
    unresolved = jnp.einsum("nr,brx->bnx", basis, latent_state)
    c2 = low_moment_state_to_resolved_hermite(state)[:, 2]
    lower = jnp.concatenate((c2[:, None], unresolved[:, :-1]), axis=1)
    upper = jnp.concatenate(
        (unresolved[:, 1:], jnp.zeros_like(unresolved[:, :1])), axis=1
    )
    orders = jnp.arange(
        3, 3 + basis.shape[0], dtype=state.dtype
    )
    streaming_flux = (
        jnp.sqrt(orders)[None, :, None] * lower
        + jnp.sqrt(orders + 1.0)[None, :, None] * upper
    )
    maximum_order = jnp.asarray(float(2 + basis.shape[0]), dtype=state.dtype)
    damping = jnp.asarray(tail_damping, dtype=state.dtype) * jnp.power(
        orders / maximum_order,
        jnp.asarray(tail_power, dtype=state.dtype),
    )
    field = electric_field_from_density(
        state[:, 0], k_arr, poisson_sign=poisson_sign
    )
    acceleration = -jnp.sqrt(orders)[None, :, None] * _dealias_state(
        field[:, None, :] * lower
    )
    unresolved_rhs = (
        -spectral_derivative(streaming_flux, k_arr)
        + acceleration
        - damping[None, :, None] * unresolved
    )
    return jnp.einsum("nr,bnx->brx", basis, unresolved_rhs)


def projected_hermite_electric_kick_rhs(
    state: Array,
    latent_state: Array,
    basis: Array,
    k_arr: Array,
    *,
    poisson_sign: float = 1.0,
) -> tuple[Array, Array]:
    """Return only the quadratic electric-force part of Vlasov dynamics."""
    state = jnp.asarray(state)
    latent_state = jnp.asarray(latent_state, dtype=state.dtype)
    basis = jnp.asarray(basis, dtype=state.dtype)
    unresolved = jnp.einsum("nr,brx->bnx", basis, latent_state)
    c2 = low_moment_state_to_resolved_hermite(state)[:, 2]
    lower = jnp.concatenate((c2[:, None], unresolved[:, :-1]), axis=1)
    orders = jnp.arange(3, 3 + basis.shape[0], dtype=state.dtype)
    field = electric_field_from_density(
        state[:, 0], k_arr, poisson_sign=poisson_sign
    )
    latent_rhs = jnp.einsum(
        "nr,bnx->brx",
        basis,
        -jnp.sqrt(orders)[None, :, None]
        * _dealias_state(field[:, None, :] * lower),
    )
    fluid_rhs = jnp.stack(
        (
            jnp.zeros_like(field),
            -_dealias_state(state[:, 0] * field),
            -2.0 * _dealias_state(state[:, 1] * field),
        ),
        axis=1,
    )
    return fluid_rhs, latent_rhs


def advance_projected_hermite_electric_kick(
    state: Array,
    latent_state: Array,
    basis: Array,
    k_arr: Array,
    *,
    fine_steps: int,
    fine_dt: float,
    kick_scale: float = 1.0,
    poisson_sign: float = 1.0,
) -> tuple[Array, Array]:
    """Integrate the nonlinear electric kick while density remains fixed."""
    if int(fine_steps) <= 0 or float(fine_dt) <= 0.0:
        raise ValueError("fine_steps and fine_dt must be positive")
    dt_value = jnp.asarray(fine_dt, dtype=state.dtype)
    scale = jnp.asarray(kick_scale, dtype=state.dtype)

    def rhs(fluid, latent):
        fluid_rhs, latent_rhs = projected_hermite_electric_kick_rhs(
            fluid,
            latent,
            basis,
            k_arr,
            poisson_sign=poisson_sign,
        )
        return scale * fluid_rhs, scale * latent_rhs

    def substep(_, carry):
        fluid, latent = carry
        k1_fluid, k1_latent = rhs(fluid, latent)
        k2_fluid, k2_latent = rhs(
            fluid + 0.5 * dt_value * k1_fluid,
            latent + 0.5 * dt_value * k1_latent,
        )
        k3_fluid, k3_latent = rhs(
            fluid + 0.5 * dt_value * k2_fluid,
            latent + 0.5 * dt_value * k2_latent,
        )
        k4_fluid, k4_latent = rhs(
            fluid + dt_value * k3_fluid,
            latent + dt_value * k3_latent,
        )
        updated_fluid = fluid + (dt_value / 6.0) * (
            k1_fluid + 2.0 * k2_fluid + 2.0 * k3_fluid + k4_fluid
        )
        updated_latent = latent + (dt_value / 6.0) * (
            k1_latent + 2.0 * k2_latent + 2.0 * k3_latent + k4_latent
        )
        return updated_fluid, updated_latent

    return jax.lax.fori_loop(
        0,
        int(fine_steps),
        substep,
        (state, latent_state),
    )


def advance_coupled_semilinear_strang(
    state: Array,
    latent_state: Array,
    latent_correction: Array,
    half_propagator: Array,
    basis: Array,
    k_arr: Array,
    *,
    resolved_scale: Array,
    latent_scale: Array,
    fine_steps: int,
    fine_dt: float,
    kick_scale: float = 1.0,
    correction_location: str = "midpoint",
    poisson_sign: float = 1.0,
) -> tuple[Array, Array]:
    """Compose fitted linear phase mixing and the exact electric kick."""
    if correction_location not in {"midpoint", "endpoint"}:
        raise ValueError(
            f"Unsupported semilinear correction location: {correction_location}"
        )

    def linear_half_step(fluid, latent):
        updated = apply_coupled_linear_propagator(
            half_propagator,
            low_moment_state_to_resolved_hermite(fluid),
            latent,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
        )
        return resolved_hermite_to_low_moment_state(updated[:, :3]), updated[:, 3:]

    half_state, half_latent = linear_half_step(state, latent_state)
    kicked_state, kicked_latent = advance_projected_hermite_electric_kick(
        half_state,
        half_latent,
        basis,
        k_arr,
        fine_steps=fine_steps,
        fine_dt=fine_dt,
        kick_scale=kick_scale,
        poisson_sign=poisson_sign,
    )
    correction = jnp.asarray(latent_correction, dtype=kicked_latent.dtype)
    if correction_location == "midpoint":
        kicked_latent = kicked_latent + correction
    updated_state, updated_latent = linear_half_step(kicked_state, kicked_latent)
    if correction_location == "endpoint":
        updated_latent = updated_latent + correction
    return updated_state, updated_latent


def advance_coupled_projected_hermite(
    state: Array,
    latent_state: Array,
    latent_correction: Array,
    basis: Array,
    k_arr: Array,
    *,
    fine_steps: int,
    fine_dt: float,
    tail_damping: float,
    tail_power: float,
    poisson_sign: float = 1.0,
) -> tuple[Array, Array]:
    """Advance one cadence with a physical Hermite baseline and latent source."""
    cadence = jnp.asarray(float(fine_steps) * float(fine_dt), dtype=state.dtype)
    correction_rate = jnp.asarray(latent_correction, dtype=state.dtype) / cadence

    def rhs(fluid, latent):
        latent_rhs = projected_hermite_latent_rhs(
            fluid,
            latent,
            basis,
            k_arr,
            tail_damping=tail_damping,
            tail_power=tail_power,
            poisson_sign=poisson_sign,
        ) + correction_rate
        fluid_rhs = low_moment_rhs(
            fluid,
            latent_heat_flux_gradient(fluid, latent, basis, k_arr),
            k_arr,
            poisson_sign=poisson_sign,
        )
        return fluid_rhs, latent_rhs

    dt_value = jnp.asarray(fine_dt, dtype=state.dtype)

    def substep(_, carry):
        fluid, latent = carry
        k1_fluid, k1_latent = rhs(fluid, latent)
        k2_fluid, k2_latent = rhs(
            fluid + 0.5 * dt_value * k1_fluid,
            latent + 0.5 * dt_value * k1_latent,
        )
        k3_fluid, k3_latent = rhs(
            fluid + 0.5 * dt_value * k2_fluid,
            latent + 0.5 * dt_value * k2_latent,
        )
        k4_fluid, k4_latent = rhs(
            fluid + dt_value * k3_fluid,
            latent + dt_value * k3_latent,
        )
        updated_fluid = _dealias_state(
            fluid
            + (dt_value / 6.0)
            * (k1_fluid + 2.0 * k2_fluid + 2.0 * k3_fluid + k4_fluid)
        )
        updated_latent = _dealias_state(
            latent
            + (dt_value / 6.0)
            * (k1_latent + 2.0 * k2_latent + 2.0 * k3_latent + k4_latent)
        )
        return updated_fluid, updated_latent

    return jax.lax.fori_loop(
        0,
        int(fine_steps),
        substep,
        (state, latent_state),
    )


def coupled_low_moment_latent_step(
    params: Dict[str, Array],
    state: Array,
    latent_state: Array,
    propagator: Array,
    basis: Array,
    k_arr: Array,
    *,
    previous_latent_state: Array | None = None,
    resolved_center: Array,
    resolved_scale: Array,
    latent_center: Array,
    latent_scale: Array,
    depth: int,
    correction_bounds: Array | None = None,
    closure_residual_scale: float | Array = 1.0,
    closure_correction_bound: float | Array = 40.0,
    closure_aligned_output: bool = False,
    closure_only_correction: bool = False,
    closure_readout_mode: str = "multiplicative",
    closure_gate_scale: float | Array = 0.1,
    closure_gate_power: int = 2,
    latent_readout_mode: str = "multiplicative",
    latent_gate_scale: float | Array = 0.1,
    latent_gate_power: int = 2,
    linear_baseline: str = "fitted_propagator",
    hermite_tail_damping: float = 10.0,
    hermite_tail_power: float = 6.0,
    semilinear_kick_scale: float = 1.0,
    semilinear_correction_location: str = "midpoint",
    fine_steps: int,
    fine_dt: float,
    dynamics_model: str = "multiplicative_operator",
    poisson_sign: float = 1.0,
) -> tuple[Array, Array]:
    """Advance a stable linear baseline plus coupled nonlinear residuals."""
    if int(fine_steps) <= 0 or float(fine_dt) <= 0.0:
        raise ValueError("fine_steps and fine_dt must be positive")
    if dynamics_model != "multiplicative_operator":
        raise ValueError("coupled dynamics require multiplicative_operator")
    if linear_baseline not in {
        "fitted_propagator",
        "projected_hermite",
        "semilinear_strang",
    }:
        raise ValueError(f"Unsupported linear_baseline: {linear_baseline}")
    if closure_readout_mode not in {
        "multiplicative",
        "gated_linear",
        "gated_linear_residual",
    }:
        raise ValueError(f"Unsupported closure_readout_mode: {closure_readout_mode}")
    if latent_readout_mode not in {
        "equilibrium_cnn",
        "multiplicative",
        "gated_linear",
        "gated_linear_residual",
    }:
        raise ValueError(f"Unsupported latent_readout_mode: {latent_readout_mode}")
    resolved = low_moment_state_to_resolved_hermite(state)
    linear_state = apply_coupled_linear_propagator(
        propagator,
        resolved,
        latent_state,
        resolved_scale=resolved_scale,
        latent_scale=latent_scale,
    )
    linear_latent = linear_state[:, 3:]
    normalized_resolved = resolved / jnp.asarray(
        resolved_scale, dtype=resolved.dtype
    )[None, :, None]
    normalized_latent = latent_state / jnp.asarray(
        latent_scale, dtype=latent_state.dtype
    )[None, :, None]
    normalized_latent_delta = None
    if previous_latent_state is not None:
        normalized_latent_delta = (
            latent_state
            - jnp.asarray(previous_latent_state, dtype=latent_state.dtype)
        ) / jnp.asarray(latent_scale, dtype=latent_state.dtype)[None, :, None]
    if closure_only_correction:
        if not closure_aligned_output:
            raise ValueError(
                "closure_only_correction requires closure_aligned_output"
            )
        correction = jnp.zeros_like(linear_latent)
    else:
        if latent_readout_mode == "equilibrium_cnn":
            correction = equilibrium_preserving_latent_cnn_correction(
                params,
                normalized_resolved,
                normalized_latent,
                depth=depth,
                normalized_latent_delta=normalized_latent_delta,
            )
        elif latent_readout_mode == "multiplicative":
            correction = state_conditioned_latent_operator_correction(
                params,
                normalized_resolved,
                normalized_latent,
                depth=depth,
                equilibrium_preserving=True,
                normalized_latent_delta=normalized_latent_delta,
            )
        else:
            correction = gated_spectral_latent_correction(
                params,
                normalized_resolved,
                normalized_latent,
                gate_scale=latent_gate_scale,
                gate_power=latent_gate_power,
            )
            if latent_readout_mode == "gated_linear_residual":
                correction = correction + state_conditioned_latent_operator_correction(
                    params,
                    normalized_resolved,
                    normalized_latent,
                    depth=depth,
                    equilibrium_preserving=True,
                )
        if correction_bounds is None:
            correction_bounds = jnp.ones(
                (latent_state.shape[1],), dtype=correction.dtype
            )
        correction = bounded_normalized_latent_correction(
            correction,
            correction_bounds,
        ) * jnp.asarray(
            latent_scale, dtype=correction.dtype
        )[None, :, None]
    if closure_aligned_output:
        correction = closure_orthogonal_latent_correction(correction, basis)
        if closure_readout_mode == "multiplicative":
            raw_closure_correction = state_conditioned_closure_correction(
                params,
                normalized_resolved,
                normalized_latent,
                depth=depth,
                equilibrium_preserving=True,
                normalized_latent_delta=normalized_latent_delta,
            )
        else:
            raw_closure_correction = gated_spectral_closure_correction(
                params,
                normalized_resolved,
                normalized_latent,
                gate_scale=closure_gate_scale,
                gate_power=closure_gate_power,
            )
            if closure_readout_mode == "gated_linear_residual":
                raw_closure_correction = (
                    raw_closure_correction
                    + state_conditioned_closure_correction(
                        params,
                        normalized_resolved,
                        normalized_latent,
                        depth=depth,
                        equilibrium_preserving=True,
                    )
                )
        closure_correction = bounded_normalized_closure_correction(
            raw_closure_correction,
            closure_correction_bound,
        ) * jnp.asarray(closure_residual_scale, dtype=correction.dtype)
        correction = correction + closure_aligned_latent_correction(
            closure_correction,
            basis,
        )
    if linear_baseline == "projected_hermite":
        return advance_coupled_projected_hermite(
            state,
            latent_state,
            correction,
            basis,
            k_arr,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            tail_damping=hermite_tail_damping,
            tail_power=hermite_tail_power,
            poisson_sign=poisson_sign,
        )
    if linear_baseline == "semilinear_strang":
        return advance_coupled_semilinear_strang(
            state,
            latent_state,
            correction,
            propagator,
            basis,
            k_arr,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            kick_scale=semilinear_kick_scale,
            correction_location=semilinear_correction_location,
            poisson_sign=poisson_sign,
        )
    updated_latent = linear_latent + correction
    linear_fluid_state = resolved_hermite_to_low_moment_state(linear_state[:, :3])

    def fluid_substep(index, carry):
        nonlinear_fluid, linear_fluid = carry
        fraction = (
            jnp.asarray(index, dtype=state.dtype) + jnp.asarray(0.5, state.dtype)
        ) / jnp.asarray(float(fine_steps), dtype=state.dtype)
        nonlinear_latent = (
            (1.0 - fraction) * latent_state + fraction * updated_latent
        )
        baseline_latent = (
            (1.0 - fraction) * latent_state + fraction * linear_latent
        )
        heat_flux_gradient = latent_heat_flux_gradient(
            nonlinear_fluid, nonlinear_latent, basis, k_arr
        )
        nonlinear_fluid = low_moment_rk4_step(
            nonlinear_fluid,
            heat_flux_gradient,
            k_arr,
            fine_dt,
            poisson_sign=poisson_sign,
        )
        first_unresolved = reconstruct_first_unresolved_field(
            baseline_latent, basis
        )
        linear_fluid = _linear_low_moment_step(
            linear_fluid,
            first_unresolved,
            k_arr,
            fine_dt,
            poisson_sign=poisson_sign,
        )
        return nonlinear_fluid, linear_fluid

    nonlinear_fluid, baseline_fluid = jax.lax.fori_loop(
        0,
        int(fine_steps),
        fluid_substep,
        (state, state),
    )
    updated_state = linear_fluid_state + nonlinear_fluid - baseline_fluid
    return updated_state, updated_latent


def rollout_coupled_low_moment_latent(
    params: Dict[str, Array],
    initial_state: Array,
    initial_latent: Array,
    propagator: Array,
    basis: Array,
    k_arr: Array,
    *,
    steps: int,
    initial_previous_latent: Array | None = None,
    latent_delay_input: bool = False,
    resolved_center: Array,
    resolved_scale: Array,
    latent_center: Array,
    latent_scale: Array,
    depth: int,
    correction_bounds: Array | None = None,
    closure_residual_scale: float | Array = 1.0,
    closure_correction_bound: float | Array = 40.0,
    closure_aligned_output: bool = False,
    closure_only_correction: bool = False,
    closure_readout_mode: str = "multiplicative",
    closure_gate_scale: float | Array = 0.1,
    closure_gate_power: int = 2,
    latent_readout_mode: str = "multiplicative",
    latent_gate_scale: float | Array = 0.1,
    latent_gate_power: int = 2,
    linear_baseline: str = "fitted_propagator",
    hermite_tail_damping: float = 10.0,
    hermite_tail_power: float = 6.0,
    semilinear_kick_scale: float = 1.0,
    semilinear_correction_location: str = "midpoint",
    fine_steps: int,
    fine_dt: float,
    dynamics_model: str = "multiplicative_operator",
    poisson_sign: float = 1.0,
) -> tuple[Array, Array]:
    """Roll out the fully autonomous coupled state for a fixed horizon."""
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    if initial_previous_latent is None:
        initial_previous_latent = initial_latent

    def body(carry, _):
        updated_state, updated_latent = coupled_low_moment_latent_step(
            params,
            carry[0],
            carry[1],
            propagator,
            basis,
            k_arr,
            previous_latent_state=(carry[2] if latent_delay_input else None),
            resolved_center=resolved_center,
            resolved_scale=resolved_scale,
            latent_center=latent_center,
            latent_scale=latent_scale,
            correction_bounds=correction_bounds,
            closure_residual_scale=closure_residual_scale,
            closure_correction_bound=closure_correction_bound,
            closure_aligned_output=closure_aligned_output,
            closure_only_correction=closure_only_correction,
            closure_readout_mode=closure_readout_mode,
            closure_gate_scale=closure_gate_scale,
            closure_gate_power=closure_gate_power,
            latent_readout_mode=latent_readout_mode,
            latent_gate_scale=latent_gate_scale,
            latent_gate_power=latent_gate_power,
            linear_baseline=linear_baseline,
            hermite_tail_damping=hermite_tail_damping,
            hermite_tail_power=hermite_tail_power,
            semilinear_kick_scale=semilinear_kick_scale,
            semilinear_correction_location=semilinear_correction_location,
            depth=depth,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            dynamics_model=dynamics_model,
            poisson_sign=poisson_sign,
        )
        return (updated_state, updated_latent, carry[1]), (
            updated_state,
            updated_latent,
        )

    (_, _, _), (state_history, latent_history) = jax.lax.scan(
        jax.checkpoint(body),
        (initial_state, initial_latent, initial_previous_latent),
        xs=None,
        length=int(steps),
    )
    return jnp.swapaxes(state_history, 0, 1), jnp.swapaxes(latent_history, 0, 1)
