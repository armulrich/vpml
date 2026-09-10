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
    limit_low_moment_state,
    low_moment_rk4_step,
    spectral_derivative,
)


Array = jax.Array

# Eight causal features for each of modes 1--4, plus six relative-phase pairs.
# Each pair contributes raw sine/cosine and versions conditioned on nonlinear
# frequency history and signed-transfer contrast. These expose modal geometry
# without introducing an absolute-time or single-turnaround clock.
BOUNCE_FEATURE_COUNT = 68


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
    conditioner_experts: int = 0,
    dtype=jnp.float32,
) -> Dict[str, Array]:
    """Initialize a low-rank, state-conditioned spectral propagator update."""
    if min(spectral_modes, operator_rank) <= 0:
        raise ValueError("spectral_modes and operator_rank must be positive")
    if float(output_projection_init_scale) < 0.0:
        raise ValueError("output_projection_init_scale must be nonnegative")
    conditioner_key, projection_key, expert_key = jax.random.split(key, 3)
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
    if int(conditioner_experts) < 0:
        raise ValueError("conditioner_experts must be nonnegative")
    if int(conditioner_experts) > 0:
        output_channels = int(
            operator_rank
            if conditioner_output_channels is None
            else conditioner_output_channels
        )
        params["expert_output_kernel"] = jnp.zeros(
            (
                int(conditioner_experts),
                output_channels,
                int(width),
                int(kernel_size),
            ),
            dtype=dtype,
        )
        params["expert_output_bias"] = jnp.zeros(
            (int(conditioner_experts), output_channels), dtype=dtype
        )
        # A separate zero-initialized phase response lets an expert distinguish
        # electric-energy damping from regrowth at the same state amplitude.
        # Keeping this branch separate from the amplitude experts preserves
        # existing checkpoints exactly until it is explicitly trained.
        params["expert_phase_output_kernel"] = jnp.zeros(
            (
                int(conditioner_experts),
                output_channels,
                int(width),
                int(kernel_size),
            ),
            dtype=dtype,
        )
        params["expert_phase_output_bias"] = jnp.zeros(
            (int(conditioner_experts), output_channels), dtype=dtype
        )
        params["expert_cyclic_output_kernel"] = jnp.zeros(
            (
                int(conditioner_experts),
                BOUNCE_FEATURE_COUNT,
                output_channels,
                int(width),
                int(kernel_size),
            ),
            dtype=dtype,
        )
        params["expert_cyclic_output_bias"] = jnp.zeros(
            (int(conditioner_experts), BOUNCE_FEATURE_COUNT, output_channels), dtype=dtype
        )
        params["expert_phase_gain"] = jnp.zeros(
            (int(conditioner_experts), output_channels), dtype=dtype
        )
        params["expert_bounce_gate_gain"] = jnp.zeros(
            (int(conditioner_experts), BOUNCE_FEATURE_COUNT), dtype=dtype
        )
        params["expert_cyclic_gate_hidden_kernel"] = (
            0.1
            / math.sqrt(float(BOUNCE_FEATURE_COUNT))
            * jax.random.normal(
                expert_key, (BOUNCE_FEATURE_COUNT, 16), dtype=dtype
            )
        )
        params["expert_cyclic_gate_hidden_bias"] = jnp.zeros((16,), dtype=dtype)
        params["expert_cyclic_gate_output_kernel"] = jnp.zeros(
            (int(conditioner_experts), 16), dtype=dtype
        )
        # Translation-equivariant mode-1 closure actuation. The two real
        # components multiply the instantaneous complex density mode.
        params["expert_cyclic_low_mode_closure_gain"] = jnp.zeros(
            (BOUNCE_FEATURE_COUNT, 2), dtype=dtype
        )
        params["expert_cyclic_low_mode_pressure_gain"] = jnp.zeros(
            (BOUNCE_FEATURE_COUNT, 2), dtype=dtype
        )
        params["expert_cyclic_modal_pressure_gain"] = jnp.zeros(
            (BOUNCE_FEATURE_COUNT, 3, 2), dtype=dtype
        )
        params["expert_cyclic_low_mode_momentum_gain"] = jnp.zeros(
            (BOUNCE_FEATURE_COUNT, 2), dtype=dtype
        )
        params["expert_gate_kernel"] = (
            0.1
            / math.sqrt(float(width))
            * jax.random.normal(
                expert_key,
                (int(conditioner_experts), int(width)),
                dtype=dtype,
            )
        )
        params["expert_gate_bias"] = jnp.zeros(
            (int(conditioner_experts),), dtype=dtype
        )
        # Optional abstention route.  Its expert output is identically zero;
        # the negative initialization preserves legacy expert mixtures until
        # null routing is explicitly trained.
        params["expert_null_gate_bias"] = jnp.asarray((-20.0,), dtype=dtype)
        # Give the experts distinct, state-observable responsibilities from the
        # first update.  A small learned gate alone starts almost uniformly and
        # therefore leaves all expert output gradients effectively identical.
        # These centers partition the log10 RMS of the normalized state from
        # late-time near-linear states through strongly nonlinear states.
        default_centers = jnp.asarray((-3.0, -1.3, -0.7, 0.25), dtype=dtype)
        if int(conditioner_experts) == 1:
            centers = jnp.asarray((0.0,), dtype=dtype)
        else:
            centers = jnp.interp(
                jnp.linspace(0.0, 3.0, int(conditioner_experts), dtype=dtype),
                jnp.arange(4, dtype=dtype),
                default_centers,
            )
        params["expert_gate_log_amplitude_centers"] = centers
        params["expert_gate_log_amplitude_width"] = jnp.asarray(0.30, dtype=dtype)
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


def _history_frequency_activation(exposure, deadband, *, dtype):
    """Compact causal activation from running maximum modal frequency."""
    exposure = jnp.asarray(exposure, dtype=dtype)
    mode_count = exposure.shape[1] // 2
    history_frequency = jnp.max(exposure[:, mode_count:], axis=1)
    deadband = jnp.asarray(deadband, dtype=dtype)
    coordinate = jnp.clip(
        (history_frequency - deadband[0])
        / jnp.maximum(deadband[1] - deadband[0], 1.0e-8),
        0.0,
        1.0,
    )
    return coordinate * coordinate * (3.0 - 2.0 * coordinate)


def kinetic_latent_dynamics_correction(
    params: Dict[str, Array],
    resolved_state: Array,
    latent_state: Array,
    *,
    depth: int,
    zero_bias: bool = False,
    bounce_exposure: Array | None = None,
    specialist_activation: Array | None = None,
    specialist_high_mix: Array | None = None,
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
    correction = _periodic_convolution(
        hidden,
        params["output_kernel"],
        (
            jnp.zeros_like(params["output_bias"])
            if zero_bias
            else params["output_bias"]
        ),
    )
    if "expert_output_kernel" in params:
        pooled_hidden = jnp.mean(hidden, axis=-1)
        gate_logits = jnp.einsum(
            "bw,ew->be", pooled_hidden, params["expert_gate_kernel"]
        ) + params["expert_gate_bias"][None, :]
        if bounce_exposure is not None and "expert_bounce_gate_gain" in params:
            gate_logits = gate_logits + jnp.einsum(
                "bf,ef->be",
                jnp.asarray(bounce_exposure, dtype=gate_logits.dtype)[
                    :, : params["expert_bounce_gate_gain"].shape[1]
                ],
                params["expert_bounce_gate_gain"],
            )
        if (
            bounce_exposure is not None
            and "expert_cyclic_gate_hidden_kernel" in params
            and "expert_cyclic_gate_output_kernel" in params
        ):
            cyclic_input = jnp.asarray(bounce_exposure, dtype=gate_logits.dtype)[
                :, : params["expert_cyclic_gate_hidden_kernel"].shape[0]
            ]
            cyclic_hidden = jnp.tanh(
                cyclic_input @ params["expert_cyclic_gate_hidden_kernel"]
                + params["expert_cyclic_gate_hidden_bias"][None, :]
            )
            gate_logits = gate_logits + jnp.einsum(
                "bh,eh->be",
                cyclic_hidden,
                params["expert_cyclic_gate_output_kernel"],
            )
        if "expert_gate_log_amplitude_centers" in params:
            # Work with mean-square amplitude rather than RMS so the gate is
            # smooth at equilibrium.  The floor only defines the limiting log
            # amplitude and has zero first derivative there.
            mean_square = jnp.mean(jnp.square(inputs), axis=(1, 2))
            log_amplitude = 0.5 * jnp.log10(mean_square + 1.0e-8)
            centers = params["expert_gate_log_amplitude_centers"]
            width = params["expert_gate_log_amplitude_width"]
            gate_logits = gate_logits - 0.5 * jnp.square(
                (log_amplitude[:, None] - centers[None, :]) / width
            )
        if "expert_null_gate_bias" in params:
            null_logits = jnp.broadcast_to(
                params["expert_null_gate_bias"][None, :],
                (gate_logits.shape[0], 1),
            )
            gates = jax.nn.softmax(
                jnp.concatenate((null_logits, gate_logits), axis=1), axis=1
            )[:, 1:]
        else:
            gates = jax.nn.softmax(gate_logits, axis=1)
        specialist_gates = None
        if (
            specialist_activation is not None
            and "specialist_router_bounce_gate_gain" in params
        ):
            specialist_features = jnp.asarray(
                bounce_exposure, dtype=gate_logits.dtype
            )
            if (
                params["specialist_router_bounce_gate_gain"].shape[1]
                > specialist_features.shape[1]
            ):
                band_count = 8
                band_width = int(math.ceil(latent_state.shape[1] / band_count))
                padded = jnp.pad(
                    latent_state,
                    ((0, 0), (0, band_count * band_width - latent_state.shape[1]), (0, 0)),
                ).reshape(latent_state.shape[0], band_count, band_width, -1)
                latent_delta = resolved_state[:, -latent_state.shape[1] :]
                delta_padded = jnp.pad(
                    latent_delta,
                    ((0, 0), (0, band_count * band_width - latent_delta.shape[1]), (0, 0)),
                ).reshape(latent_state.shape[0], band_count, band_width, -1)
                specialist_features = jnp.concatenate(
                    (
                        specialist_features,
                        jnp.tanh(jnp.mean(jnp.square(padded), axis=(2, 3))),
                        jnp.tanh(jnp.mean(padded * delta_padded, axis=(2, 3))),
                    ),
                    axis=1,
                )
            specialist_logits = jnp.einsum(
                "bf,ef->be",
                specialist_features[
                    :, : params["specialist_router_bounce_gate_gain"].shape[1]
                ],
                params["specialist_router_bounce_gate_gain"],
            ) + params["specialist_router_gate_bias"][None, :]
            if (
                "specialist_cyclic_gate_hidden_kernel" in params
                and "specialist_cyclic_gate_output_kernel" in params
            ):
                specialist_input = specialist_features[
                    :, : params["specialist_cyclic_gate_hidden_kernel"].shape[0]
                ]
                specialist_hidden = jnp.tanh(
                    specialist_input @ params["specialist_cyclic_gate_hidden_kernel"]
                    + params["specialist_cyclic_gate_hidden_bias"][None, :]
                )
                specialist_logits = specialist_logits + jnp.einsum(
                    "bh,eh->be",
                    specialist_hidden,
                    params["specialist_cyclic_gate_output_kernel"],
                )
            specialist_null = jnp.broadcast_to(
                params["specialist_router_null_gate_bias"][None, :],
                (specialist_logits.shape[0], 1),
            )
            specialist_gates = jax.nn.softmax(
                jnp.concatenate((specialist_null, specialist_logits), axis=1),
                axis=1,
            )[:, 1:]
            if (
                specialist_high_mix is not None
                and "specialist_high_router_bounce_gate_gain" in params
            ):
                high_logits = jnp.einsum(
                    "bf,ef->be",
                    jnp.asarray(bounce_exposure, dtype=gate_logits.dtype)[
                        :, : params["specialist_high_router_bounce_gate_gain"].shape[1]
                    ],
                    params["specialist_high_router_bounce_gate_gain"],
                ) + params["specialist_high_router_gate_bias"][None, :]
                high_null = jnp.broadcast_to(
                    params["specialist_high_router_null_gate_bias"][None, :],
                    (high_logits.shape[0], 1),
                )
                high_gates = jax.nn.softmax(
                    jnp.concatenate((high_null, high_logits), axis=1), axis=1
                )[:, 1:]
                specialist_gates = specialist_gates + specialist_high_mix[:, None] * (
                    high_gates - specialist_gates
                )
            if "specialist_cyclic_output_kernel" not in params:
                gates = gates + specialist_activation[:, None] * (
                    specialist_gates - gates
                )
        elif specialist_activation is not None:
            # A causal train-calibrated deadband can also protect a learned
            # base expert router directly.  Below the weak/strong gap this is
            # exactly the null path; above it the fitted cyclic router is
            # unchanged.
            gates = gates * specialist_activation[:, None]
        phase = None
        if (
            "expert_phase_gain" in params
            or "expert_phase_output_kernel" in params
        ):
            # Continuity and Poisson imply d<E^2>/dt = 2<E j> (up to the
            # fixed Poisson sign).  Its normalized correlation is a bounded,
            # instantaneous phase coordinate that separates damping and
            # regrowth without requiring future information.  Positive input
            # scales do not change its sign.
            density = resolved_state[:, 0]
            momentum = resolved_state[:, 1]
            density_hat = jnp.fft.rfft(
                density - jnp.mean(density, axis=-1, keepdims=True), axis=-1
            )
            mode_numbers = jnp.arange(density_hat.shape[-1], dtype=density.dtype)
            field_hat = jnp.zeros_like(density_hat)
            field_hat = field_hat.at[:, 1:].set(
                1j * density_hat[:, 1:] / mode_numbers[None, 1:]
            )
            field = jnp.fft.irfft(field_hat, n=density.shape[-1], axis=-1)
            centered_momentum = momentum - jnp.mean(
                momentum, axis=-1, keepdims=True
            )
            numerator = jnp.mean(field * centered_momentum, axis=-1)
            denominator = jnp.sqrt(
                jnp.mean(jnp.square(field), axis=-1)
                * jnp.mean(jnp.square(centered_momentum), axis=-1)
                + jnp.asarray(1.0e-12, dtype=density.dtype)
            )
            phase = numerator / denominator
        expert_corrections = jax.vmap(
            lambda kernel, bias: _periodic_convolution(hidden, kernel, bias),
            in_axes=(0, 0),
            out_axes=1,
        )(
            params["expert_output_kernel"],
            params["expert_output_bias"],
        )
        if "expert_phase_gain" in params:
            expert_corrections = expert_corrections * (
                1.0
                + phase[:, None, None, None]
                * params["expert_phase_gain"][None, :, :, None]
            )
        correction = correction + jnp.einsum(
            "be,beox->box", gates, expert_corrections
        )
        if "expert_phase_output_kernel" in params:
            phase_corrections = jax.vmap(
                lambda kernel, bias: _periodic_convolution(hidden, kernel, bias),
                in_axes=(0, 0),
                out_axes=1,
            )(
                params["expert_phase_output_kernel"],
                params["expert_phase_output_bias"],
            )
            correction = correction + jnp.einsum(
                "b,be,beox->box", phase, gates, phase_corrections
            )
        if (
            bounce_exposure is not None
            and "expert_cyclic_output_kernel" in params
        ):
            cyclic_features = jnp.asarray(bounce_exposure, dtype=correction.dtype)[
                :, : params["expert_cyclic_output_kernel"].shape[1]
            ]
            cyclic_gates = (
                gates if specialist_activation is None
                else gates * specialist_activation[:, None]
            )
            if "cyclic_contract_before_convolution" in params:
                # Convolution is linear in its kernel and bias. Contract the
                # spatially constant causal weights first, avoiding one full
                # convolution per expert/feature pair. Opt-in preserves legacy
                # floating-point operation order for historical checkpoints.
                kernels = jnp.einsum(
                    "be,bf,efowk->bowk", cyclic_gates, cyclic_features,
                    params["expert_cyclic_output_kernel"],
                )
                biases = jnp.einsum(
                    "be,bf,efo->bo", cyclic_gates, cyclic_features,
                    params["expert_cyclic_output_bias"],
                )
                correction = correction + jax.vmap(
                    lambda values, kernel, bias: _periodic_convolution(
                        values[None], kernel, bias
                    )[0]
                )(hidden, kernels, biases)
            else:
                cyclic_corrections = jax.vmap(
                    lambda expert_kernel, expert_bias: jax.vmap(
                        lambda kernel, bias: _periodic_convolution(hidden, kernel, bias),
                        in_axes=(0, 0), out_axes=1,
                    )(expert_kernel, expert_bias),
                    in_axes=(0, 0), out_axes=1,
                )(params["expert_cyclic_output_kernel"], params["expert_cyclic_output_bias"])
                correction = correction + jnp.einsum(
                    "bf,be,befcx->bcx", cyclic_features, cyclic_gates, cyclic_corrections
                )
        if (
            bounce_exposure is not None
            and specialist_gates is not None
            and "specialist_cyclic_output_kernel" in params
        ):
            specialist_corrections = jax.vmap(
                lambda expert_kernel, expert_bias: jax.vmap(
                    lambda kernel, bias: _periodic_convolution(hidden, kernel, bias),
                    in_axes=(0, 0),
                    out_axes=1,
                )(expert_kernel, expert_bias),
                in_axes=(0, 0),
                out_axes=1,
            )(
                params["specialist_cyclic_output_kernel"],
                params["specialist_cyclic_output_bias"],
            )
            specialist_direction = jnp.ones_like(specialist_activation)
            if "specialist_nonpositive_transfer_only" in params:
                specialist_direction = jnp.where(
                    phase <= 0.0,
                    jnp.ones_like(specialist_activation),
                    jnp.zeros_like(specialist_activation),
                )
            correction = correction + jnp.einsum(
                "bf,be,befcx->bcx",
                jnp.asarray(bounce_exposure, dtype=correction.dtype)[
                    :, : params["specialist_cyclic_output_kernel"].shape[1]
                ],
                specialist_gates
                * (specialist_activation * specialist_direction)[:, None],
                specialist_corrections,
            )
        if (
            bounce_exposure is not None
            and specialist_activation is not None
            and "addon_router_bounce_gate_gain" in params
            and "addon_cyclic_output_kernel" in params
        ):
            addon_features = jnp.asarray(bounce_exposure, dtype=correction.dtype)
            if params["addon_router_bounce_gate_gain"].shape[1] > addon_features.shape[1]:
                band_count = 8
                band_width = int(math.ceil(latent_state.shape[1] / band_count))
                padded = jnp.pad(
                    latent_state,
                    ((0, 0), (0, band_count * band_width - latent_state.shape[1]), (0, 0)),
                ).reshape(latent_state.shape[0], band_count, band_width, -1)
                latent_delta = resolved_state[:, -latent_state.shape[1] :]
                delta_padded = jnp.pad(
                    latent_delta,
                    ((0, 0), (0, band_count * band_width - latent_delta.shape[1]), (0, 0)),
                ).reshape(latent_state.shape[0], band_count, band_width, -1)
                band_energy = jnp.tanh(jnp.mean(jnp.square(padded), axis=(2, 3)))
                band_transfer = jnp.tanh(jnp.mean(padded * delta_padded, axis=(2, 3)))
                addon_features = jnp.concatenate(
                    (addon_features, band_energy, band_transfer), axis=1
                )
            addon_logits = jnp.einsum(
                "bf,ef->be",
                addon_features[:, : params["addon_router_bounce_gate_gain"].shape[1]],
                params["addon_router_bounce_gate_gain"],
            ) + params["addon_router_gate_bias"][None, :]
            if "addon_cyclic_gate_hidden_kernel" in params:
                addon_hidden = jnp.tanh(
                    addon_features[:, : params["addon_cyclic_gate_hidden_kernel"].shape[0]]
                    @ params["addon_cyclic_gate_hidden_kernel"]
                    + params["addon_cyclic_gate_hidden_bias"][None, :]
                )
                addon_logits = addon_logits + jnp.einsum(
                    "bh,eh->be", addon_hidden, params["addon_cyclic_gate_output_kernel"]
                )
            addon_null = jnp.broadcast_to(
                params["addon_router_null_gate_bias"][None, :],
                (addon_logits.shape[0], 1),
            )
            addon_gates = jax.nn.softmax(
                jnp.concatenate((addon_null, addon_logits), axis=1), axis=1
            )[:, 1:]
            if (
                "addon_low_amplitude_only" in params
                and specialist_high_mix is not None
            ):
                addon_gates = addon_gates * (1.0 - specialist_high_mix)[:, None]
            if "addon_positive_transfer_only" in params:
                addon_gates = addon_gates * (phase > 0.0)[:, None]
            if "addon_nonpositive_transfer_only" in params:
                addon_gates = addon_gates * (phase <= 0.0)[:, None]
            if "addon_transfer_direction" in params:
                direction = params["addon_transfer_direction"][None, :]
                direction_mask = jnp.where(
                    direction > 0.0,
                    (phase > 0.0)[:, None],
                    jnp.where(
                        direction < 0.0,
                        (phase <= 0.0)[:, None],
                        jnp.ones_like(addon_gates, dtype=bool),
                    ),
                )
                addon_gates = addon_gates * direction_mask.astype(addon_gates.dtype)
            addon_corrections = jax.vmap(
                lambda expert_kernel, expert_bias: jax.vmap(
                    lambda kernel, bias: _periodic_convolution(hidden, kernel, bias),
                    in_axes=(0, 0),
                    out_axes=1,
                )(expert_kernel, expert_bias),
                in_axes=(0, 0),
                out_axes=1,
            )(
                params["addon_cyclic_output_kernel"],
                params["addon_cyclic_output_bias"],
            )
            correction = correction + jnp.einsum(
                "bf,be,befcx->bcx",
                addon_features[:, : params["addon_cyclic_output_kernel"].shape[1]],
                addon_gates * specialist_activation[:, None],
                addon_corrections,
            )
    return correction


def equilibrium_preserving_latent_cnn_correction(
    params: Dict[str, Array],
    normalized_resolved: Array,
    normalized_latent: Array,
    *,
    depth: int,
    normalized_latent_delta: Array | None = None,
    input_compression_scale: float = 0.0,
    bounce_exposure: Array | None = None,
    specialist_activation: Array | None = None,
    specialist_high_mix: Array | None = None,
) -> Array:
    """Return the direct CNN residual after removing its affine response.

    A positive ``input_compression_scale`` applies a smooth signed-asinh map
    inside the nonlinear conditioner.  The subtraction below is still taken
    with respect to the original inputs, so the complete correction has zero
    value and Jacobian at equilibrium.
    """
    if float(input_compression_scale) < 0.0:
        raise ValueError("input_compression_scale must be nonnegative")
    conditioned_resolved = normalized_resolved
    if normalized_latent_delta is not None:
        conditioned_resolved = jnp.concatenate(
            (conditioned_resolved, normalized_latent_delta), axis=1
        )
    zero_resolved = jnp.zeros_like(conditioned_resolved)
    zero_latent = jnp.zeros_like(normalized_latent)

    def network(resolved, latent):
        if float(input_compression_scale) > 0.0:
            scale = jnp.asarray(input_compression_scale, dtype=resolved.dtype)
            resolved = scale * jnp.arcsinh(resolved / scale)
            latent = scale * jnp.arcsinh(latent / scale)
        return kinetic_latent_dynamics_correction(
            params,
            resolved,
            latent,
            depth=depth,
            bounce_exposure=bounce_exposure,
            specialist_activation=specialist_activation,
            specialist_high_mix=specialist_high_mix,
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
    physical_tail_activation: Array | None = None,
    physical_tail_core_rows: int | None = None,
    core_damping_relaxation: Array | None = None,
    physical_identity_basis: bool = False,
) -> Array:
    """Project nonlinear Vlasov Hermite-tail dynamics into the latent basis.

    The physical-only benchmark may explicitly declare an identity basis to
    elide redundant dense projections. Callers must supply an actual identity.
    """
    state = jnp.asarray(state)
    latent_state = jnp.asarray(latent_state, dtype=state.dtype)
    basis = jnp.asarray(basis, dtype=state.dtype)
    if physical_identity_basis and basis.shape[0] != basis.shape[1]:
        raise ValueError("physical_identity_basis requires a square identity basis")
    c2 = low_moment_state_to_resolved_hermite(state)[:, 2]
    orders = jnp.arange(
        3, 3 + basis.shape[0], dtype=state.dtype
    )
    maximum_order = jnp.asarray(float(2 + basis.shape[0]), dtype=state.dtype)
    damping_denominator = maximum_order
    if physical_tail_activation is not None and physical_tail_core_rows is not None:
        core_maximum_order = jnp.asarray(
            float(2 + int(physical_tail_core_rows)), dtype=state.dtype
        )
        damping_denominator = jnp.where(
            orders <= core_maximum_order,
            core_maximum_order,
            maximum_order,
        )
    elif core_damping_relaxation is not None:
        relaxation = jnp.asarray(core_damping_relaxation, dtype=state.dtype)
        damping_denominator = maximum_order * (1.0 + relaxation[:, None])
    damping = jnp.asarray(tail_damping, dtype=state.dtype) * jnp.power(
        orders / damping_denominator,
        jnp.asarray(tail_power, dtype=state.dtype),
    )
    field = electric_field_from_density(
        state[:, 0], k_arr, poisson_sign=poisson_sign
    )
    if int(basis.shape[0]) - int(basis.shape[1]) > 2:
        # Evaluate the Galerkin system directly in latent coordinates.  All
        # operations are linear in the Hermite-order axis, and dealiasing acts
        # only in x, so projection commutes exactly without reconstructing the
        # full physical tail at every RHS evaluation.
        lower_matrix = jnp.zeros_like(basis).at[1:].set(basis[:-1])
        upper_matrix = jnp.zeros_like(basis).at[:-1].set(basis[1:])
        streaming_matrix = basis.T @ (
            jnp.sqrt(orders)[:, None] * lower_matrix
            + jnp.sqrt(orders + 1.0)[:, None] * upper_matrix
        )
        if physical_tail_activation is not None and physical_tail_core_rows is not None:
            activation = jnp.asarray(physical_tail_activation, dtype=state.dtype)
            boundary = int(physical_tail_core_rows) - 1
            boundary_matrix = jnp.sqrt(orders[boundary] + 1.0) * jnp.outer(
                basis[boundary], basis[boundary + 1]
            )
            streaming = jnp.einsum("rs,bsx->brx", streaming_matrix, latent_state)
            streaming = streaming + (activation - 1.0)[:, None, None] * jnp.einsum(
                "rs,bsx->brx", boundary_matrix, latent_state
            )
        else:
            streaming = jnp.einsum(
                "rs,bsx->brx", streaming_matrix, latent_state
            )
        c2_coupling = jnp.sqrt(orders[0]) * basis[0]
        streaming = streaming + c2[:, None, :] * c2_coupling[None, :, None]
        acceleration_matrix = basis.T @ (
            jnp.sqrt(orders)[:, None] * lower_matrix
        )
        acceleration_lower = jnp.einsum(
            "rs,bsx->brx", acceleration_matrix, latent_state
        ) + c2[:, None, :] * c2_coupling[None, :, None]
        acceleration = -_dealias_state(field[:, None, :] * acceleration_lower)
        if damping.ndim == 2:
            damping_matrix = jnp.einsum(
                "nr,bn,ns->brs", basis, damping, basis
            )
            damped = jnp.einsum("brs,bsx->brx", damping_matrix, latent_state)
        else:
            damping_matrix = basis.T @ (damping[:, None] * basis)
            damped = jnp.einsum(
                "rs,bsx->brx", damping_matrix, latent_state
            )
        return -spectral_derivative(streaming, k_arr) + acceleration - damped

    unresolved = (latent_state if physical_identity_basis else
                  jnp.einsum("nr,brx->bnx", basis, latent_state))
    lower = jnp.concatenate((c2[:, None], unresolved[:, :-1]), axis=1)
    upper = jnp.concatenate(
        (unresolved[:, 1:], jnp.zeros_like(unresolved[:, :1])), axis=1
    )
    if physical_tail_activation is not None and physical_tail_core_rows is not None:
        activation = jnp.asarray(physical_tail_activation, dtype=state.dtype)
        upper = upper.at[:, int(physical_tail_core_rows) - 1].multiply(
            activation[:, None]
        )
    streaming_flux = (
        jnp.sqrt(orders)[None, :, None] * lower
        + jnp.sqrt(orders + 1.0)[None, :, None] * upper
    )
    acceleration = -jnp.sqrt(orders)[None, :, None] * _dealias_state(
        field[:, None, :] * lower
    )
    unresolved_rhs = (
        -spectral_derivative(streaming_flux, k_arr)
        + acceleration
        - (damping[:, :, None] if damping.ndim == 2 else damping[None, :, None])
        * unresolved
    )
    return unresolved_rhs if physical_identity_basis else jnp.einsum("nr,bnx->brx", basis, unresolved_rhs)


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
    physical_tail_activation: Array | None = None,
    physical_tail_core_rows: int | None = None,
    core_damping_relaxation: Array | None = None,
    physical_identity_basis: bool = False,
) -> tuple[Array, Array]:
    """Advance one cadence with a physical Hermite baseline and latent source."""
    cadence = jnp.asarray(float(fine_steps) * float(fine_dt), dtype=state.dtype)
    correction_rate = jnp.asarray(latent_correction, dtype=state.dtype) / cadence
    # Hermite streaming speeds grow like sqrt(m).  Preserve the requested
    # macro cadence while subcycling long physical tails so extending the
    # cutoff does not silently violate the explicit RK4 stability interval.
    maximum_order = int(basis.shape[0]) + 2
    # The existing Nx=64 solver is empirically stable through M=128 at the
    # native 0.01 step.  Scale only beyond that verified range so M<=128 keeps
    # the frozen discrete map exactly.
    hermite_substeps = max(1, int(math.ceil(math.sqrt(maximum_order / 128.0))))

    def rhs(fluid, latent):
        latent_rhs = projected_hermite_latent_rhs(
            fluid,
            latent,
            basis,
            k_arr,
            tail_damping=tail_damping,
            tail_power=tail_power,
            poisson_sign=poisson_sign,
            physical_tail_activation=physical_tail_activation,
            physical_tail_core_rows=physical_tail_core_rows,
            core_damping_relaxation=core_damping_relaxation,
            physical_identity_basis=physical_identity_basis,
        ) + correction_rate
        fluid_rhs = low_moment_rhs(
            fluid,
            latent_heat_flux_gradient(fluid, latent, basis, k_arr),
            k_arr,
            poisson_sign=poisson_sign,
        )
        return fluid_rhs, latent_rhs

    dt_value = jnp.asarray(
        float(fine_dt) / float(hermite_substeps), dtype=state.dtype
    )

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
        updated_fluid = limit_low_moment_state(
            _dealias_state(
                fluid
                + (dt_value / 6.0)
                * (k1_fluid + 2.0 * k2_fluid + 2.0 * k3_fluid + k4_fluid)
            ),
            reference_state=fluid,
        )
        updated_latent = _dealias_state(
            latent
            + (dt_value / 6.0)
            * (k1_latent + 2.0 * k2_latent + 2.0 * k3_latent + k4_latent)
        )
        return updated_fluid, updated_latent

    return jax.lax.fori_loop(
        0,
        int(fine_steps) * hermite_substeps,
        substep,
        (state, latent_state),
    )


def _bounce_phase_features(state: Array, exposure: Array, k_arr: Array) -> Array:
    """Causal cyclic phase, frequency, and signed energy-transfer features."""
    density_hat = jnp.fft.rfft(state[:, 0], axis=-1, norm="forward")
    momentum_hat = jnp.fft.rfft(state[:, 1], axis=-1, norm="forward")
    mode_count = min(4, density_hat.shape[1] - 1)
    modes = jnp.arange(1, mode_count + 1)
    field_mode = 1j * density_hat[:, 1 : mode_count + 1] / k_arr[modes][None, :]
    frequency = jnp.sqrt(
        jnp.maximum(2.0 * k_arr[modes][None, :] * jnp.abs(field_mode), 0.0)
    )
    transfer = jnp.real(
        field_mode * jnp.conj(momentum_hat[:, 1 : mode_count + 1])
    )
    transfer_scale = (
        jnp.abs(field_mode)
        * jnp.abs(momentum_hat[:, 1 : mode_count + 1])
        + 1.0e-12
    )
    if exposure.ndim == 2 and exposure.shape[1] >= 2 * mode_count:
        phase = exposure[:, :mode_count]
        maximum_frequency = exposure[:, mode_count : 2 * mode_count]
    else:
        mode_count = 1
        phase = (exposure[:, 0] if exposure.ndim == 2 else exposure)[:, None]
        maximum_frequency = (
            exposure[:, 1] if exposure.ndim == 2 else jnp.zeros_like(phase[:, 0])
        )[:, None]
        frequency = frequency[:, :1]
        transfer = transfer[:, :1]
        transfer_scale = transfer_scale[:, :1]
    cosine_phase = jnp.cos(phase) - 1.0
    signed_transfer = transfer / transfer_scale
    history_amplitude = jnp.tanh(maximum_frequency)
    features = jnp.stack(
        (
            jnp.sin(phase),
            cosine_phase,
            jnp.tanh(frequency),
            signed_transfer,
            history_amplitude * cosine_phase,
            history_amplitude * signed_transfer,
            jnp.power(history_amplitude, 3) * cosine_phase,
            jnp.power(history_amplitude, 3) * signed_transfer,
        ),
        axis=2,
    )
    flattened = features.reshape(features.shape[0], -1)
    if mode_count < 2:
        return flattened.astype(state.dtype)
    pairwise = []
    conditioned_pairwise = []
    for left in range(mode_count):
        for right in range(left + 1, mode_count):
            relative_phase = phase[:, left] - phase[:, right]
            sine_relative = jnp.sin(relative_phase)
            cosine_relative = jnp.cos(relative_phase) - 1.0
            pairwise.extend((sine_relative, cosine_relative))
            history_pair = jnp.sqrt(
                jnp.maximum(history_amplitude[:, left] * history_amplitude[:, right], 0.0)
            )
            transfer_contrast = 0.5 * (
                signed_transfer[:, left] - signed_transfer[:, right]
            )
            conditioned_pairwise.extend(
                (
                    history_pair * sine_relative,
                    history_pair * cosine_relative,
                    transfer_contrast * sine_relative,
                    transfer_contrast * cosine_relative,
                )
            )
    return jnp.concatenate(
        (
            flattened,
            jnp.stack(pairwise, axis=1),
            jnp.stack(conditioned_pairwise, axis=1),
        ),
        axis=1,
    ).astype(state.dtype)


def coupled_low_moment_latent_step(
    params: Dict[str, Array],
    state: Array,
    latent_state: Array,
    propagator: Array,
    basis: Array,
    k_arr: Array,
    *,
    previous_latent_state: Array | None = None,
    bounce_exposure: Array | None = None,
    resolved_center: Array,
    resolved_scale: Array,
    latent_center: Array,
    latent_scale: Array,
    depth: int,
    latent_input_scale: Array | None = None,
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
    latent_state_bound: float = 0.0,
    equilibrium_input_compression_scale: float = 0.0,
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
    if float(latent_state_bound) < 0.0:
        raise ValueError("latent_state_bound must be nonnegative")

    def bound_latent(value):
        if float(latent_state_bound) == 0.0:
            return value
        scale = jnp.asarray(latent_scale, dtype=value.dtype)[None, :, None]
        return jnp.clip(
            value / scale,
            -float(latent_state_bound),
            float(latent_state_bound),
        ) * scale
    resolved = low_moment_state_to_resolved_hermite(state)
    if linear_baseline == "projected_hermite":
        # The fitted propagator is not part of this baseline.  Skipping it is
        # also essential for split learned/physical tails, where the learned
        # operator may cover only a prefix of the physical latent state.
        linear_latent = latent_state
    else:
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
    # Input conditioning is independent of physical output units. Very small
    # training variances need not amplify numerical tail noise into a large
    # low-moment response. None preserves legacy checkpoint behavior exactly.
    feature_scale = jnp.asarray(
        latent_scale if latent_input_scale is None else latent_input_scale,
        dtype=latent_state.dtype,
    )
    normalized_latent = latent_state / feature_scale[None, :, None]
    normalized_latent_delta = None
    if previous_latent_state is not None:
        normalized_latent_delta = (
            latent_state
            - jnp.asarray(previous_latent_state, dtype=latent_state.dtype)
        ) / feature_scale[None, :, None]
    # The equilibrium CNN consumes lift inputs directly; spectral operator
    # tensors are unused by that readout and need not exist in its checkpoint.
    input_kernel = (
        params["lift_kernel"]
        if latent_readout_mode == "equilibrium_cnn"
        else params["operator_v_real"]
    )
    learned_input_channels = int(input_kernel.shape[1])
    learned_latent_rank = (
        (learned_input_channels - int(normalized_resolved.shape[1])) // 2
        if normalized_latent_delta is not None
        else learned_input_channels - int(normalized_resolved.shape[1])
    )
    learned_latent_rank = min(int(normalized_latent.shape[1]), learned_latent_rank)
    learned_normalized_latent = normalized_latent[:, :learned_latent_rank]
    learned_normalized_latent_delta = (
        normalized_latent_delta[:, :learned_latent_rank]
        if normalized_latent_delta is not None
        else None
    )
    if closure_only_correction:
        if not closure_aligned_output:
            raise ValueError(
                "closure_only_correction requires closure_aligned_output"
            )
        correction = jnp.zeros_like(linear_latent)
    else:
        if latent_readout_mode == "equilibrium_cnn":
            specialist_activation = (
                _history_frequency_activation(
                    bounce_exposure,
                    params["expert_history_frequency_deadband"],
                    dtype=state.dtype,
                )
                if (
                    bounce_exposure is not None
                    and "expert_history_frequency_deadband" in params
                )
                else (
                    jnp.ones((state.shape[0],), dtype=state.dtype)
                    if "specialist_unrestricted_activation" in params
                    else None
                )
            )
            specialist_high_mix = None
            if (
                bounce_exposure is not None
                and "specialist_router_amplitude_split" in params
            ):
                exposure = jnp.asarray(bounce_exposure, dtype=state.dtype)
                mode_count = exposure.shape[1] // 2
                maximum = jnp.max(exposure[:, mode_count:], axis=1)
                split = jnp.asarray(
                    params["specialist_router_amplitude_split"], dtype=state.dtype
                )
                specialist_high_mix = jax.nn.sigmoid(
                    (maximum - split[0]) / jnp.maximum(split[1], 1.0e-6)
                )
            correction = equilibrium_preserving_latent_cnn_correction(
                params,
                normalized_resolved,
                learned_normalized_latent,
                depth=depth,
                normalized_latent_delta=learned_normalized_latent_delta,
                input_compression_scale=equilibrium_input_compression_scale,
                bounce_exposure=(
                    _bounce_phase_features(state, bounce_exposure, k_arr)
                    if bounce_exposure is not None
                    else None
                ),
                specialist_activation=specialist_activation,
                specialist_high_mix=specialist_high_mix,
            )
        elif latent_readout_mode == "multiplicative":
            correction = state_conditioned_latent_operator_correction(
                params,
                normalized_resolved,
                learned_normalized_latent,
                depth=depth,
                equilibrium_preserving=True,
                normalized_latent_delta=learned_normalized_latent_delta,
            )
        else:
            correction = gated_spectral_latent_correction(
                params,
                normalized_resolved,
                learned_normalized_latent,
                gate_scale=latent_gate_scale,
                gate_power=latent_gate_power,
            )
            if latent_readout_mode == "gated_linear_residual":
                correction = correction + state_conditioned_latent_operator_correction(
                    params,
                    normalized_resolved,
                    learned_normalized_latent,
                    depth=depth,
                    equilibrium_preserving=True,
                )
        if correction_bounds is None:
            correction_bounds = jnp.ones(
                (latent_state.shape[1],), dtype=correction.dtype
            )
        correction = bounded_normalized_latent_correction(
            correction,
            jnp.asarray(correction_bounds)[:learned_latent_rank],
        ) * jnp.asarray(
            latent_scale, dtype=correction.dtype
        )[None, :learned_latent_rank, None]
        if learned_latent_rank < int(latent_state.shape[1]):
            correction = jnp.pad(
                correction,
                ((0, 0), (0, int(latent_state.shape[1]) - learned_latent_rank), (0, 0)),
            )
    cyclic_low_mode_pressure = None
    cyclic_low_mode_momentum = None
    if (
        bounce_exposure is not None
        and "expert_cyclic_low_mode_closure_gain" in params
    ):
        cyclic_features = _bounce_phase_features(state, bounce_exposure, k_arr)
        gain_parts = jnp.einsum(
            "bf,fc->bc",
            cyclic_features[
                :, : params["expert_cyclic_low_mode_closure_gain"].shape[0]
            ],
            params["expert_cyclic_low_mode_closure_gain"],
        )
        density_hat = jnp.fft.rfft(state[:, 0], axis=-1, norm="forward")
        closure_hat = jnp.zeros_like(density_hat)
        complex_gain = gain_parts[:, 0] + 1j * gain_parts[:, 1]
        closure_hat = closure_hat.at[:, 1].set(complex_gain * density_hat[:, 1])
        low_mode_closure = jnp.fft.irfft(
            closure_hat, n=state.shape[-1], axis=-1, norm="forward"
        ).astype(correction.dtype)
        if "expert_history_frequency_deadband" in params:
            activation = _history_frequency_activation(
                bounce_exposure,
                params["expert_history_frequency_deadband"],
                dtype=correction.dtype,
            )
            low_mode_closure = low_mode_closure * activation[:, None]
        correction = correction + closure_aligned_latent_correction(
            low_mode_closure, basis
        )
    if (
        bounce_exposure is not None
        and "expert_cyclic_low_mode_pressure_gain" in params
    ):
        cyclic_features = _bounce_phase_features(state, bounce_exposure, k_arr)
        gain_parts = jnp.einsum(
            "bf,fc->bc",
            cyclic_features[
                :, : params["expert_cyclic_low_mode_pressure_gain"].shape[0]
            ],
            params["expert_cyclic_low_mode_pressure_gain"],
        )
        density_hat = jnp.fft.rfft(state[:, 0], axis=-1, norm="forward")
        pressure_hat = jnp.zeros_like(density_hat)
        complex_gain = gain_parts[:, 0] + 1j * gain_parts[:, 1]
        pressure_hat = pressure_hat.at[:, 1].set(complex_gain * density_hat[:, 1])
        cyclic_low_mode_pressure = jnp.fft.irfft(
            pressure_hat, n=state.shape[-1], axis=-1, norm="forward"
        ).astype(state.dtype)
    if (
        bounce_exposure is not None
        and "specialist_cyclic_low_mode_pressure_gain" in params
        and "specialist_router_bounce_gate_gain" in params
    ):
        cyclic_features = _bounce_phase_features(state, bounce_exposure, k_arr)
        specialist_logits = jnp.einsum(
            "bf,ef->be",
            cyclic_features[:, : params["specialist_router_bounce_gate_gain"].shape[1]],
            params["specialist_router_bounce_gate_gain"],
        ) + params["specialist_router_gate_bias"][None, :]
        if (
            "specialist_cyclic_gate_hidden_kernel" in params
            and "specialist_cyclic_gate_output_kernel" in params
        ):
            hidden_gate = jnp.tanh(
                cyclic_features[:, : params["specialist_cyclic_gate_hidden_kernel"].shape[0]]
                @ params["specialist_cyclic_gate_hidden_kernel"]
                + params["specialist_cyclic_gate_hidden_bias"][None, :]
            )
            specialist_logits = specialist_logits + jnp.einsum(
                "bh,eh->be", hidden_gate, params["specialist_cyclic_gate_output_kernel"]
            )
        null_logits = jnp.broadcast_to(
            params["specialist_router_null_gate_bias"][None, :],
            (specialist_logits.shape[0], 1),
        )
        specialist_gates = jax.nn.softmax(
            jnp.concatenate((null_logits, specialist_logits), axis=1), axis=1
        )[:, 1:]
        if (
            "specialist_router_positive_prototypes" in params
            and "specialist_router_null_prototypes" in params
        ):
            prototype_features = (
                cyclic_features[:, : params["specialist_router_feature_center"].shape[0]]
                - params["specialist_router_feature_center"][None, :]
            ) / params["specialist_router_feature_scale"][None, :]
            positive_distance = jnp.sqrt(
                jnp.maximum(
                    jnp.min(
                        jnp.sum(
                            jnp.square(
                                prototype_features[:, None, :]
                                - params["specialist_router_positive_prototypes"][None, :, :]
                            ),
                            axis=-1,
                        ),
                        axis=1,
                    ),
                    jnp.finfo(state.dtype).tiny,
                )
            )
            null_distance = jnp.sqrt(
                jnp.maximum(
                    jnp.min(
                        jnp.sum(
                            jnp.square(
                                prototype_features[:, None, :]
                                - params["specialist_router_null_prototypes"][None, :, :]
                            ),
                            axis=-1,
                        ),
                        axis=1,
                    ),
                    jnp.finfo(state.dtype).tiny,
                )
            )
            prototype_gate = jax.nn.sigmoid(
                (
                    null_distance
                    - positive_distance
                    - params["specialist_router_prototype_threshold"][0]
                )
                / params["specialist_router_prototype_temperature"][0]
            )
            specialist_gates = jnp.zeros_like(specialist_gates).at[:, 1].set(
                prototype_gate
            )
        if (
            "specialist_positive_transfer_pressure_router" in params
            or "specialist_negative_transfer_pressure_router" in params
        ):
            density = state[:, 0]
            momentum = state[:, 1]
            density_hat_for_gate = jnp.fft.rfft(
                density - jnp.mean(density, axis=-1, keepdims=True), axis=-1
            )
            mode_numbers = jnp.arange(
                density_hat_for_gate.shape[-1], dtype=density.dtype
            )
            field_hat_for_gate = jnp.zeros_like(density_hat_for_gate)
            field_hat_for_gate = field_hat_for_gate.at[:, 1:].set(
                1j * density_hat_for_gate[:, 1:] / mode_numbers[None, 1:]
            )
            field_for_gate = jnp.fft.irfft(
                field_hat_for_gate, n=density.shape[-1], axis=-1
            )
            total_transfer = jnp.mean(
                field_for_gate
                * (momentum - jnp.mean(momentum, axis=-1, keepdims=True)),
                axis=-1,
            )
            route_positive = "specialist_positive_transfer_pressure_router" in params
            directed = jnp.where(
                total_transfer > 0.0 if route_positive else total_transfer < 0.0,
                jnp.ones_like(total_transfer),
                jnp.zeros_like(total_transfer),
            )
            specialist_gates = jnp.zeros_like(specialist_gates).at[:, 1].set(
                directed
            )
        if "specialist_prototype_nonpositive_transfer_only" in params:
            density = state[:, 0]
            momentum = state[:, 1]
            density_hat_for_gate = jnp.fft.rfft(
                density - jnp.mean(density, axis=-1, keepdims=True), axis=-1
            )
            mode_numbers = jnp.arange(
                density_hat_for_gate.shape[-1], dtype=density.dtype
            )
            field_hat_for_gate = jnp.zeros_like(density_hat_for_gate)
            field_hat_for_gate = field_hat_for_gate.at[:, 1:].set(
                1j * density_hat_for_gate[:, 1:] / mode_numbers[None, 1:]
            )
            field_for_gate = jnp.fft.irfft(
                field_hat_for_gate, n=density.shape[-1], axis=-1
            )
            total_transfer = jnp.mean(
                field_for_gate
                * (momentum - jnp.mean(momentum, axis=-1, keepdims=True)),
                axis=-1,
            )
            specialist_gates = specialist_gates * (total_transfer <= 0.0)[:, None]
        gain_parts = jnp.einsum(
            "bf,efc,be->bc",
            cyclic_features[:, : params["specialist_cyclic_low_mode_pressure_gain"].shape[1]],
            params["specialist_cyclic_low_mode_pressure_gain"],
            specialist_gates,
        )
        if "expert_history_frequency_deadband" in params:
            activation = _history_frequency_activation(
                bounce_exposure,
                params["expert_history_frequency_deadband"],
                dtype=state.dtype,
            )
            gain_parts = gain_parts * activation[:, None]
        density_hat = jnp.fft.rfft(state[:, 0], axis=-1, norm="forward")
        pressure_hat = jnp.zeros_like(density_hat)
        complex_gain = gain_parts[:, 0] + 1j * gain_parts[:, 1]
        pressure_hat = pressure_hat.at[:, 1].set(complex_gain * density_hat[:, 1])
        specialist_pressure = jnp.fft.irfft(
            pressure_hat, n=state.shape[-1], axis=-1, norm="forward"
        ).astype(state.dtype)
        cyclic_low_mode_pressure = (
            specialist_pressure
            if cyclic_low_mode_pressure is None
            else cyclic_low_mode_pressure + specialist_pressure
        )
    if (
        bounce_exposure is not None
        and "expert_cyclic_modal_pressure_gain" in params
    ):
        cyclic_features = _bounce_phase_features(state, bounce_exposure, k_arr)
        gain_parts = jnp.einsum(
            "bf,fmc->bmc",
            cyclic_features[
                :, : params["expert_cyclic_modal_pressure_gain"].shape[0]
            ],
            params["expert_cyclic_modal_pressure_gain"],
        )
        density_hat = jnp.fft.rfft(state[:, 0], axis=-1, norm="forward")
        pressure_hat = jnp.zeros_like(density_hat)
        # Modes 2--4 only. Mode 1 has its own independently tested actuator;
        # keeping it out makes this a clean test of the missing modal signal.
        mode_count = min(3, int(pressure_hat.shape[-1]) - 2)
        complex_gain = gain_parts[..., 0] + 1j * gain_parts[..., 1]
        pressure_hat = pressure_hat.at[:, 2 : mode_count + 2].set(
            complex_gain[:, :mode_count] * density_hat[:, 2 : mode_count + 2]
        )
        modal_pressure = jnp.fft.irfft(
            pressure_hat, n=state.shape[-1], axis=-1, norm="forward"
        ).astype(state.dtype)
        cyclic_low_mode_pressure = (
            modal_pressure
            if cyclic_low_mode_pressure is None
            else cyclic_low_mode_pressure + modal_pressure
        )
    if (
        bounce_exposure is not None
        and "expert_cyclic_low_mode_momentum_gain" in params
    ):
        cyclic_features = _bounce_phase_features(state, bounce_exposure, k_arr)
        gain_parts = jnp.einsum(
            "bf,fc->bc",
            cyclic_features[
                :, : params["expert_cyclic_low_mode_momentum_gain"].shape[0]
            ],
            params["expert_cyclic_low_mode_momentum_gain"],
        )
        density_hat = jnp.fft.rfft(state[:, 0], axis=-1, norm="forward")
        momentum_hat = jnp.zeros_like(density_hat)
        complex_gain = gain_parts[:, 0] + 1j * gain_parts[:, 1]
        momentum_hat = momentum_hat.at[:, 1].set(complex_gain * density_hat[:, 1])
        cyclic_low_mode_momentum = jnp.fft.irfft(
            momentum_hat, n=state.shape[-1], axis=-1, norm="forward"
        ).astype(state.dtype)
    if closure_aligned_output:
        correction = closure_orthogonal_latent_correction(correction, basis)
        if closure_readout_mode == "multiplicative":
            raw_closure_correction = state_conditioned_closure_correction(
                params,
                normalized_resolved,
                learned_normalized_latent,
                depth=depth,
                equilibrium_preserving=True,
                normalized_latent_delta=learned_normalized_latent_delta,
            )
        else:
            raw_closure_correction = gated_spectral_closure_correction(
                params,
                normalized_resolved,
                learned_normalized_latent,
                gate_scale=closure_gate_scale,
                gate_power=closure_gate_power,
            )
            if closure_readout_mode == "gated_linear_residual":
                raw_closure_correction = (
                    raw_closure_correction
                    + state_conditioned_closure_correction(
                        params,
                        normalized_resolved,
                        learned_normalized_latent,
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
    physical_tail_activation = None
    physical_tail_core_rows = None
    core_damping_relaxation = None
    late_envelope_damping = None
    if (
        bounce_exposure is not None
        and "late_envelope_router_kernel" in params
        and "late_envelope_damping_rate" in params
    ):
        cyclic_features = _bounce_phase_features(state, bounce_exposure, k_arr)
        late_probability = jax.nn.sigmoid(
            cyclic_features[:, : params["late_envelope_router_kernel"].shape[0]]
            @ params["late_envelope_router_kernel"]
            + params["late_envelope_router_bias"][0]
        )
        late_probability = 1.0 - late_probability
        if "expert_history_frequency_deadband" in params:
            late_probability = late_probability * _history_frequency_activation(
                bounce_exposure,
                params["expert_history_frequency_deadband"],
                dtype=state.dtype,
            )
        if "late_envelope_nonpositive_transfer_only" in params:
            density_hat = jnp.fft.rfft(state[:, 0], axis=-1, norm="forward")
            momentum_hat = jnp.fft.rfft(state[:, 1], axis=-1, norm="forward")
            mode_count = min(4, density_hat.shape[1] - 1)
            modes = jnp.arange(1, mode_count + 1)
            field_hat = (
                1j
                * density_hat[:, 1 : mode_count + 1]
                / k_arr[modes][None, :]
            )
            transfer = jnp.sum(
                jnp.real(
                    field_hat * jnp.conj(momentum_hat[:, 1 : mode_count + 1])
                ),
                axis=1,
            )
            late_probability = late_probability * (transfer <= 0.0).astype(
                state.dtype
            )
        late_envelope_damping = late_probability * jnp.maximum(
            jnp.asarray(params["late_envelope_damping_rate"], dtype=state.dtype)[0],
            0.0,
        )
    if (
        bounce_exposure is not None
        and "specialist_core_damping_router_kernel" in params
    ):
        cyclic_features = _bounce_phase_features(state, bounce_exposure, k_arr)
        router_input = cyclic_features[
            :, : params["specialist_core_damping_router_kernel"].shape[0]
        ]
        probability = jax.nn.sigmoid(
            router_input @ params["specialist_core_damping_router_kernel"]
            + params["specialist_core_damping_router_bias"][0]
        )
        if "specialist_core_damping_router_complement" in params:
            probability = 1.0 - probability
        if "specialist_core_damping_router_deadband" in params:
            deadband = jnp.asarray(
                params["specialist_core_damping_router_deadband"], dtype=state.dtype
            )
            coordinate = jnp.clip(
                (probability - deadband[0])
                / jnp.maximum(deadband[1] - deadband[0], 1.0e-8),
                0.0,
                1.0,
            )
            probability = coordinate * coordinate * (3.0 - 2.0 * coordinate)
        scale = jnp.asarray(
            params.get(
                "specialist_core_damping_relaxation_scale",
                jnp.ones((1,), dtype=state.dtype),
            ),
            dtype=state.dtype,
        )[0]
        core_damping_relaxation = probability * jnp.maximum(scale, -0.9)
    appended_physical_rows = int(basis.shape[1]) - learned_latent_rank
    if appended_physical_rows > 0:
        # The frozen rank-60 POD spans the 62 physical rows C3:C64.  Additional
        # columns may be a compressed basis over many more physical rows, so
        # their column count cannot be used to locate the physical boundary.
        physical_tail_core_rows = min(
            int(basis.shape[0]), learned_latent_rank + 2
        )
        if (
            bounce_exposure is not None
            and "specialist_router_bounce_gate_gain" in params
            and "specialist_cyclic_gate_hidden_kernel" in params
            and "specialist_cyclic_gate_output_kernel" in params
        ):
            cyclic_features = _bounce_phase_features(state, bounce_exposure, k_arr)
            cyclic_input = cyclic_features[
                :, : params["specialist_cyclic_gate_hidden_kernel"].shape[0]
            ]
            hidden = jnp.tanh(
                cyclic_input @ params["specialist_cyclic_gate_hidden_kernel"]
                + params["specialist_cyclic_gate_hidden_bias"][None, :]
            )
            logits = (
                jnp.einsum(
                    "bf,ef->be",
                    cyclic_features[
                        :, : params["specialist_router_bounce_gate_gain"].shape[1]
                    ],
                    params["specialist_router_bounce_gate_gain"],
                )
                + params["specialist_router_gate_bias"][None, :]
                + jnp.einsum(
                    "bh,eh->be", hidden, params["specialist_cyclic_gate_output_kernel"]
                )
            )
            null_logits = jnp.broadcast_to(
                params["specialist_router_null_gate_bias"][None, :],
                (logits.shape[0], 1),
            )
            physical_tail_activation = jnp.sum(
                jax.nn.softmax(jnp.concatenate((null_logits, logits), axis=1), axis=1)[
                    :, 1:
                ],
                axis=1,
            )
            if "specialist_physical_tail_gate_deadband" in params:
                deadband = jnp.asarray(
                    params["specialist_physical_tail_gate_deadband"], dtype=state.dtype
                )
                coordinate = jnp.clip(
                    (physical_tail_activation - deadband[0])
                    / jnp.maximum(deadband[1] - deadband[0], 1.0e-8),
                    0.0,
                    1.0,
                )
                physical_tail_activation = coordinate * coordinate * (
                    3.0 - 2.0 * coordinate
                )
            if "specialist_physical_tail_gate_scale" in params:
                physical_tail_activation = physical_tail_activation * jnp.clip(
                    jnp.asarray(
                        params["specialist_physical_tail_gate_scale"], dtype=state.dtype
                    )[0],
                    0.0,
                    1.0,
                )
            if "specialist_physical_tail_router_kernel" in params:
                router_input = cyclic_features[
                    :, : params["specialist_physical_tail_router_kernel"].shape[0]
                ]
                router_logit = (
                    router_input @ params["specialist_physical_tail_router_kernel"]
                    + params["specialist_physical_tail_router_bias"][0]
                )
                # Zero initialization is the identity modulation, while the
                # bounded factor can learn both suppression and amplification.
                physical_tail_activation = physical_tail_activation * (
                    2.0 * jax.nn.sigmoid(router_logit)
                )
            if (
                "specialist_physical_tail_nonpositive_transfer_only" in params
                or "specialist_physical_tail_positive_transfer_only" in params
            ):
                density_hat = jnp.fft.rfft(state[:, 0], axis=-1, norm="forward")
                momentum_hat = jnp.fft.rfft(state[:, 1], axis=-1, norm="forward")
                mode_count = min(4, density_hat.shape[1] - 1)
                modes = jnp.arange(1, mode_count + 1)
                field_hat = (
                    1j * density_hat[:, 1 : mode_count + 1] / k_arr[modes][None, :]
                )
                total_transfer = jnp.sum(
                    jnp.real(field_hat * jnp.conj(momentum_hat[:, 1 : mode_count + 1])),
                    axis=1,
                )
                keep = (
                    total_transfer > 0.0
                    if "specialist_physical_tail_positive_transfer_only" in params
                    else total_transfer <= 0.0
                )
                physical_tail_activation = physical_tail_activation * keep.astype(
                    state.dtype
                )
            physical_tail_activation = jnp.clip(
                physical_tail_activation, 0.0, 1.0
            )
        else:
            physical_tail_activation = jnp.zeros(
                (state.shape[0],), dtype=state.dtype
            )
    if linear_baseline == "projected_hermite":
        updated_state, updated_latent = advance_coupled_projected_hermite(
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
            physical_tail_activation=physical_tail_activation,
            physical_tail_core_rows=physical_tail_core_rows,
            core_damping_relaxation=core_damping_relaxation,
        )
        if late_envelope_damping is not None:
            updated_field = electric_field_from_density(
                updated_state[:, 0], k_arr, poisson_sign=poisson_sign
            )
            cadence = jnp.asarray(float(fine_steps) * float(fine_dt), dtype=state.dtype)
            updated_state = updated_state.at[:, 1].add(
                cadence * late_envelope_damping[:, None] * updated_field
            )
        if cyclic_low_mode_pressure is not None or cyclic_low_mode_momentum is not None:
            reference_state = updated_state
            if cyclic_low_mode_momentum is not None:
                updated_state = updated_state.at[:, 1].add(cyclic_low_mode_momentum)
            if cyclic_low_mode_pressure is not None:
                updated_state = updated_state.at[:, 2].add(cyclic_low_mode_pressure)
            updated_state = limit_low_moment_state(
                updated_state,
                reference_state=reference_state,
            )
        return updated_state, bound_latent(updated_latent)
    if linear_baseline == "semilinear_strang":
        updated_state, updated_latent = advance_coupled_semilinear_strang(
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
        return updated_state, bound_latent(updated_latent)
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
    return updated_state, bound_latent(updated_latent)


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
    initial_bounce_exposure: Array | None = None,
    latent_delay_input: bool = False,
    resolved_center: Array,
    resolved_scale: Array,
    latent_center: Array,
    latent_scale: Array,
    depth: int,
    latent_input_scale: Array | None = None,
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
    latent_state_bound: float = 0.0,
    equilibrium_input_compression_scale: float = 0.0,
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
    bounce_mode_count = min(4, k_arr.shape[0] - 1)
    def exposure_increment(state):
        density_hat = jnp.fft.rfft(state[:, 0], axis=-1, norm="forward")
        modes = jnp.arange(1, bounce_mode_count + 1)
        field_mode = (
            1j
            * density_hat[:, 1 : bounce_mode_count + 1]
            / k_arr[modes][None, :]
        )
        increment = float(fine_steps) * float(fine_dt) * jnp.sqrt(
            jnp.maximum(
                2.0 * k_arr[modes][None, :] * jnp.abs(field_mode), 0.0
            )
        )
        return increment.astype(state.dtype)

    if initial_bounce_exposure is None:
        initial_frequency = exposure_increment(initial_state) / (
            float(fine_steps) * float(fine_dt)
        )
        initial_bounce_exposure = jnp.concatenate(
            (jnp.zeros_like(initial_frequency), initial_frequency), axis=1
        )
    else:
        initial_bounce_exposure = jnp.asarray(
            initial_bounce_exposure, dtype=initial_state.dtype
        )
        if initial_bounce_exposure.shape != (
            initial_state.shape[0],
            2 * bounce_mode_count,
        ):
            raise ValueError("initial_bounce_exposure has incompatible shape")

    def body(carry, _):
        updated_state, updated_latent = coupled_low_moment_latent_step(
            params,
            carry[0],
            carry[1],
            propagator,
            basis,
            k_arr,
            previous_latent_state=(carry[2] if latent_delay_input else None),
            bounce_exposure=carry[3],
            resolved_center=resolved_center,
            resolved_scale=resolved_scale,
            latent_center=latent_center,
            latent_scale=latent_scale,
            latent_input_scale=latent_input_scale,
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
            latent_state_bound=latent_state_bound,
            equilibrium_input_compression_scale=equilibrium_input_compression_scale,
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
        increment = exposure_increment(carry[0])
        updated_exposure = jnp.concatenate(
            (
                carry[3][:, :bounce_mode_count] + increment,
                carry[3][:, bounce_mode_count:],
            ),
            axis=1,
        )
        return (updated_state, updated_latent, carry[1], updated_exposure), (
            updated_state,
            updated_latent,
        )

    (_, _, _, _), (state_history, latent_history) = jax.lax.scan(
        jax.checkpoint(body),
        (initial_state, initial_latent, initial_previous_latent, initial_bounce_exposure),
        xs=None,
        length=int(steps),
    )
    return jnp.swapaxes(state_history, 0, 1), jnp.swapaxes(latent_history, 0, 1)
