"""Solver-embedded low-moment objectives and causal-memory rollout adapters."""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from vpml.models.moment_features import FIXED_INPUT_SCALING
from vpml.models.moment_closures import explicit_window_closure_step, spectral_memory_closure_step
from vpml.solvers.moments import primitive_fields
from vpml.solvers.moment_rollout import (
    rollout_burles_latent_closure,
    rollout_explicit_window_closure,
    rollout_low_moment_closure,
    warm_spectral_memory,
)
from vpml.training.low_moment_configuration import (
    REGIMES,
    _uses_compact_latent,
    _uses_window_memory,
)


def _rollout_window_memory(
    params,
    initial_state,
    history,
    history_counter,
    amplitude,
    k_arr,
    *,
    memory_backend: str,
    compact_latent=None,
    **kwargs,
):
    """Dispatch window rollouts without changing legacy backend semantics."""
    if _uses_compact_latent(memory_backend):
        kwargs.pop("encoded_history", None)
        kwargs.pop("closure_history_input", None)
        return rollout_burles_latent_closure(
            params,
            initial_state,
            history,
            history_counter,
            amplitude,
            k_arr,
            compact_latent=compact_latent,
            **kwargs,
        )
    return rollout_explicit_window_closure(
        params,
        initial_state,
        history,
        history_counter,
        amplitude,
        k_arr,
        **kwargs,
    )


def _unpack_window_memory(memory, memory_backend: str):
    if _uses_compact_latent(memory_backend):
        history, closure, encoded, counter, gradient, latent = memory
        return history, closure, encoded, counter, gradient, latent
    history, closure, encoded, counter, gradient = memory
    return history, closure, encoded, counter, gradient, None


def _block_relative_sample_loss(
    predicted_fields,
    target_fields,
    regime_index,
    target_indices,
    *,
    block_steps: int,
    block_count: int,
    floor_rms,
    trajectory_target_norm=None,
):
    """Relative primitive-field error in fixed physical-time blocks."""
    # Late-time linear Landau signals are far below the squared float32 range.
    # Accumulate this diagnostic/training normalization in float64 so the
    # measured teacher-grid convergence floor remains effective.
    predicted_fields = jnp.asarray(predicted_fields, dtype=jnp.float64)
    target_fields = jnp.asarray(target_fields, dtype=jnp.float64)
    floor_rms = jnp.asarray(floor_rms, dtype=jnp.float64)
    if trajectory_target_norm is not None:
        trajectory_target_norm = jnp.asarray(
            trajectory_target_norm, dtype=jnp.float64
        )
    squared_error = jnp.sum(jnp.square(predicted_fields - target_fields), axis=-1)
    block_ids = jnp.minimum(
        jnp.maximum((target_indices - 1) // int(block_steps), 0),
        int(block_count) - 1,
    )
    membership = jax.nn.one_hot(block_ids, int(block_count), dtype=squared_error.dtype)
    numerator = jnp.einsum("btq,btc->bqc", membership, squared_error)
    sample_counts = jnp.sum(membership, axis=1)
    supplied_target_norm = trajectory_target_norm is not None
    if not supplied_target_norm:
        target_energy = jnp.sum(jnp.square(target_fields), axis=-1)
        trajectory_target_norm = jnp.einsum(
            "btq,btc->bqc", membership, target_energy
        )
    channel_floor = floor_rms[regime_index]
    floor_energy = (
        sample_counts[:, :, None]
        * predicted_fields.shape[-1]
        * jnp.square(channel_floor[:, None, :])
    )
    denominator = (
        trajectory_target_norm
        if supplied_target_norm
        else jnp.maximum(trajectory_target_norm, floor_energy)
    )
    valid = sample_counts > 0
    safe_denominator = jnp.where(valid[:, :, None], denominator, 1.0)
    ratios = numerator / safe_denominator
    divisor = (
        int(block_count)
        if supplied_target_norm
        else jnp.maximum(jnp.sum(valid, axis=1), 1)
    )
    return jnp.sum(jnp.where(valid[:, :, None], ratios, 0.0), axis=(1, 2)) / (
        divisor * predicted_fields.shape[2]
    )


@jax.custom_jvp
def _sqrt_with_zero_subgradient(value):
    """Square root for an L2 objective, with zero subgradient at exact agreement.

    Positive values are unchanged. This adds no amplitude floor or sample weight.
    """
    return jnp.sqrt(value)


@_sqrt_with_zero_subgradient.defjvp
def _sqrt_with_zero_subgradient_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    root = _sqrt_with_zero_subgradient(value)
    denominator = 2 * jnp.sqrt(jnp.where(value == 0, 1, value))
    return root, jnp.where(value == 0, 0, tangent / denominator)


def make_loss_function(
    *,
    regimes: Sequence[str] = REGIMES,
    k_arr: np.ndarray,
    width: int,
    horizon: int,
    dt: float,
    input_scale: np.ndarray,
    regime_scales: np.ndarray,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    poisson_sign: float,
    normalized_heat_flux_bound: float,
    density_floor: float,
    pressure_floor: float,
    relative_trajectory_loss: bool = False,
    global_relative_trajectory_loss: bool = False,
    global_relative_loss_norm: str = "squared_l2",
    global_relative_normalization: str = "window_rms",
    closure_history_input: bool = False,
    relative_time_block_steps: int = 0,
    relative_time_block_count: int = 0,
    convergence_floor_rms: Optional[np.ndarray] = None,
    memory_backend: str = "latent_recurrent",
    memory_stride: int = 1,
    scan_unroll: int = 1,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
    autonomous_burnin_steps: int = 0,
    detach_autonomous_burnin: bool = True,
    log_energy_weight: float = 0.0,
    log_growth_weight: float = 0.0,
):
    if global_relative_loss_norm not in ("squared_l2", "l2"):
        raise ValueError("global_relative_loss_norm must be squared_l2 or l2")
    if global_relative_loss_norm == "l2" and not global_relative_trajectory_loss:
        raise ValueError("The l2 option requires global_relative_trajectory_loss")
    if global_relative_normalization not in ("window_rms", "initial_rms"):
        raise ValueError("Unknown global_relative_normalization")
    if global_relative_normalization == "initial_rms" and not global_relative_trajectory_loss:
        raise ValueError("Initial RMS requires global_relative_trajectory_loss")
    default_k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    input_scale_jax = jnp.asarray(input_scale, dtype=jnp.float32)
    regime_scales_jax = jnp.asarray(regime_scales, dtype=jnp.float32)
    convergence_floor_jax = (
        None
        if convergence_floor_rms is None
        else jnp.asarray(convergence_floor_rms, dtype=jnp.float32)
    )

    def loss(params, batch):
        k_jax = batch.get("k_arr", default_k_jax)
        if _uses_window_memory(memory_backend):
            initial_state = batch["initial"]
            history = batch["memory"]
            history_counter = jnp.asarray(0, dtype=jnp.int32)
            closure_history = (
                batch["heat_flux_gradient_history"]
                if closure_history_input
                else None
            )
            previous_gradient = (
                closure_history[:, -1]
                if closure_history is not None
                else None
            )
            encoded_history = None
            compact_latent = None
            if int(autonomous_burnin_steps) > 0:
                burnin_states, burnin_memory = _rollout_window_memory(
                    params,
                    initial_state,
                    history,
                    history_counter,
                    batch["amplitude"],
                    k_jax,
                    memory_backend=memory_backend,
                    horizon=int(autonomous_burnin_steps),
                    dt=dt,
                    memory_stride=memory_stride,
                    input_scale=input_scale_jax,
                    heat_flux_gradient_scale=heat_flux_gradient_scale,
                    amplitude_center=amplitude_center,
                    amplitude_scale=amplitude_scale,
                    closure_history_input=closure_history_input,
                    poisson_sign=poisson_sign,
                    normalized_heat_flux_bound=normalized_heat_flux_bound,
                    density_floor=density_floor,
                    pressure_floor=pressure_floor,
                    scan_unroll=scan_unroll,
                    input_scaling=input_scaling,
                    dynamic_amplitude_floor=dynamic_amplitude_floor,
                    allow_uniform_heating=allow_uniform_heating,
                )
                initial_state = burnin_states[:, -1]
                if detach_autonomous_burnin:
                    initial_state = jax.lax.stop_gradient(initial_state)
                    burnin_memory = jax.tree_util.tree_map(jax.lax.stop_gradient, burnin_memory)
                (
                    history,
                    closure_history,
                    encoded_history,
                    history_counter,
                    previous_gradient,
                    compact_latent,
                ) = _unpack_window_memory(
                    burnin_memory,
                    memory_backend,
                )
            rollout_result = _rollout_window_memory(
                params,
                initial_state,
                history,
                history_counter,
                batch["amplitude"],
                k_jax,
                memory_backend=memory_backend,
                compact_latent=compact_latent,
                horizon=horizon,
                dt=dt,
                memory_stride=memory_stride,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                previous_heat_flux_gradient=previous_gradient,
                closure_history_input=closure_history_input,
                encoded_history=encoded_history,
                heat_flux_gradient_history=closure_history,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
                density_floor=density_floor,
                pressure_floor=pressure_floor,
                scan_unroll=scan_unroll,
                input_scaling=input_scaling,
                dynamic_amplitude_floor=dynamic_amplitude_floor,
                allow_uniform_heating=allow_uniform_heating,
            )
            predicted = rollout_result[0]
        else:
            if int(autonomous_burnin_steps) > 0:
                raise ValueError("autonomous history burn-in requires window memory")
            warm_result = warm_spectral_memory(
                params,
                batch["memory"],
                batch["amplitude"],
                k_jax,
                width=width,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                closure_history_input=closure_history_input,
                return_closure_history=closure_history_input,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
            )
            if closure_history_input:
                hidden, previous_gradient = warm_result
            else:
                hidden = warm_result
                previous_gradient = None
            rollout_result = rollout_low_moment_closure(
                params,
                batch["initial"],
                hidden,
                batch["amplitude"],
                k_jax,
                horizon=horizon,
                dt=dt,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                previous_heat_flux_gradient=previous_gradient,
                closure_history_input=closure_history_input,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
                density_floor=density_floor,
                pressure_floor=pressure_floor,
            )
            predicted = rollout_result[0]
        batch_count, time_count = predicted.shape[:2]
        predicted_fields = primitive_fields(
            predicted.reshape(batch_count * time_count, 3, predicted.shape[-1]),
            k_jax,
            poisson_sign=poisson_sign,
        ).reshape(batch_count, time_count, 4, predicted.shape[-1])
        target_fields = primitive_fields(
            batch["targets"].reshape(
                batch_count * time_count, 3, batch["targets"].shape[-1]
            ),
            k_jax,
            poisson_sign=poisson_sign,
        ).reshape(batch_count, time_count, 4, batch["targets"].shape[-1])
        if relative_trajectory_loss:
            if convergence_floor_jax is not None:
                target_indices = batch["start_index"][:, None] + jnp.arange(
                    1, time_count + 1, dtype=jnp.int32
                )[None, :]
                sample_loss = _block_relative_sample_loss(
                    predicted_fields,
                    target_fields,
                    batch["regime_index"],
                    target_indices,
                    block_steps=relative_time_block_steps,
                    block_count=relative_time_block_count,
                    floor_rms=convergence_floor_jax,
                )
            else:
                numerator = jnp.sum(
                    jnp.square(predicted_fields - target_fields), axis=(1, 3)
                )
                denominator = jnp.sum(jnp.square(target_fields), axis=(1, 3))
                sample_loss = jnp.where(
                    jnp.all(denominator > 0.0, axis=1),
                    jnp.mean(numerator / denominator, axis=1),
                    jnp.nan,
                )
        elif global_relative_trajectory_loss:
            numerator = jnp.sum(
                jnp.square(predicted_fields - target_fields), axis=(1, 2, 3)
            )
            if global_relative_normalization == "initial_rms":
                # One scale from the complete IC at t=0, independent of the
                # anchor or its target window. It is used only in the loss.
                scale = batch["initial_reference_rms"]
                denominator = jnp.square(scale) * np.prod(target_fields.shape[1:])
            else:
                denominator = jnp.sum(jnp.square(target_fields), axis=(1, 2, 3))
            sample_loss = jnp.where(
                denominator > 0.0,
                numerator / denominator,
                jnp.nan,
            )
            if global_relative_loss_norm == "l2":
                sample_loss = _sqrt_with_zero_subgradient(sample_loss)
        else:
            scales = regime_scales_jax[batch["regime_index"]][:, None, :, None]
            normalized_error = (predicted_fields - target_fields) / scales
            sample_loss = jnp.mean(jnp.square(normalized_error), axis=(1, 2, 3))
        if float(log_energy_weight) or float(log_growth_weight):
            field_index = 3
            predicted_field = jnp.asarray(
                predicted_fields[:, :, field_index], dtype=jnp.float64
            )
            target_field = jnp.asarray(
                target_fields[:, :, field_index], dtype=jnp.float64
            )
            energy_floor = jnp.asarray(1.0e-60, dtype=jnp.float64)
            predicted_energy = jnp.maximum(
                jnp.mean(jnp.square(predicted_field), axis=-1),
                energy_floor,
            )
            target_energy = jnp.maximum(
                jnp.mean(jnp.square(target_field), axis=-1),
                energy_floor,
            )
            predicted_log = jnp.log(predicted_energy)
            target_log = jnp.log(target_energy)
            energy_loss = jnp.mean(jnp.square(predicted_log - target_log), axis=1)
            growth_loss = jnp.mean(
                jnp.square(
                    jnp.diff(predicted_log, axis=1)
                    - jnp.diff(target_log, axis=1)
                ),
                axis=1,
            )
            sample_loss = (
                sample_loss
                + jnp.asarray(log_energy_weight, dtype=sample_loss.dtype) * energy_loss
                + jnp.asarray(log_growth_weight, dtype=sample_loss.dtype) * growth_loss
            )
        regime_loss = jnp.stack(
            [
                jnp.sum(
                    jnp.where(batch["regime_index"] == index, sample_loss, 0.0)
                )
                / jnp.maximum(jnp.sum(batch["regime_index"] == index), 1)
                for index in (REGIMES.index(regime) for regime in regimes)
            ]
        )
        return jnp.mean(sample_loss), regime_loss

    return loss


def make_continuous_chunk_loss_function(
    *,
    regimes: Sequence[str] = REGIMES,
    k_arr: np.ndarray,
    width: int,
    horizon: int,
    warm_memory: bool,
    dt: float,
    input_scale: np.ndarray,
    regime_scales: np.ndarray,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    poisson_sign: float,
    normalized_heat_flux_bound: float,
    density_floor: float,
    pressure_floor: float,
    relative_trajectory_loss: bool = False,
    closure_history_input: bool = False,
    relative_time_block_steps: int = 0,
    relative_time_block_count: int = 0,
    convergence_floor_rms: Optional[np.ndarray] = None,
    memory_backend: str = "latent_recurrent",
    memory_stride: int = 1,
    scan_unroll: int = 1,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
):
    """Return one truncated-gradient chunk of a continuous autonomous rollout."""
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    input_scale_jax = jnp.asarray(input_scale, dtype=jnp.float32)
    regime_scales_jax = jnp.asarray(regime_scales, dtype=jnp.float32)
    convergence_floor_jax = (
        None
        if convergence_floor_rms is None
        else jnp.asarray(convergence_floor_rms, dtype=jnp.float32)
    )

    def loss(
        params,
        initial_state,
        memory_or_hidden,
        target_fields,
        amplitude,
        regime_index,
        trajectory_target_norm=None,
        target_indices=None,
    ):
        if target_fields.shape[-2] == 3:
            target_shape = target_fields.shape
            target_fields = primitive_fields(
                target_fields.reshape(
                    target_shape[0] * target_shape[1], 3, target_shape[-1]
                ),
                k_jax,
                poisson_sign=poisson_sign,
            ).reshape(target_shape[0], target_shape[1], 4, target_shape[-1])
        if _uses_window_memory(memory_backend):
            if warm_memory:
                history = memory_or_hidden
                closure_history = jnp.zeros(
                    (history.shape[0], history.shape[1], history.shape[-1]),
                    dtype=history.dtype,
                )
                encoded_history = None
                history_counter = jnp.asarray(0, dtype=jnp.int32)
                previous_gradient = jnp.zeros(
                    (initial_state.shape[0], initial_state.shape[-1]),
                    dtype=initial_state.dtype,
                )
                compact_latent = None
            else:
                (
                    history,
                    closure_history,
                    encoded_history,
                    history_counter,
                    previous_gradient,
                    compact_latent,
                ) = _unpack_window_memory(memory_or_hidden, memory_backend)
            rollout_result = _rollout_window_memory(
                params,
                initial_state,
                history,
                history_counter,
                amplitude,
                k_jax,
                memory_backend=memory_backend,
                compact_latent=compact_latent,
                horizon=horizon,
                dt=dt,
                memory_stride=memory_stride,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                previous_heat_flux_gradient=previous_gradient,
                closure_history_input=closure_history_input,
                encoded_history=encoded_history,
                heat_flux_gradient_history=closure_history,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
                density_floor=density_floor,
                pressure_floor=pressure_floor,
                scan_unroll=scan_unroll,
                input_scaling=input_scaling,
                dynamic_amplitude_floor=dynamic_amplitude_floor,
                allow_uniform_heating=allow_uniform_heating,
            )
            predicted, final_memory = rollout_result
        elif warm_memory:
            warmed = warm_spectral_memory(
                params,
                memory_or_hidden,
                amplitude,
                k_jax,
                width=width,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                closure_history_input=closure_history_input,
                return_closure_history=closure_history_input,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
            )
            if closure_history_input:
                hidden, previous_gradient = warmed
            else:
                hidden = warmed
                previous_gradient = None
        elif closure_history_input:
            hidden, previous_gradient = memory_or_hidden
        else:
            hidden = memory_or_hidden
            previous_gradient = None
        if not _uses_window_memory(memory_backend):
            rollout_result = rollout_low_moment_closure(
                params,
                initial_state,
                hidden,
                amplitude,
                k_jax,
                horizon=horizon,
                dt=dt,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                previous_heat_flux_gradient=previous_gradient,
                closure_history_input=closure_history_input,
                return_closure_history=closure_history_input,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
                density_floor=density_floor,
                pressure_floor=pressure_floor,
            )
            if closure_history_input:
                predicted, final_hidden, final_gradient = rollout_result
            else:
                predicted, final_hidden = rollout_result
                final_gradient = None
            final_memory = (
                (final_hidden, final_gradient)
                if closure_history_input
                else final_hidden
            )
        batch_count, time_count = predicted.shape[:2]
        predicted_fields = primitive_fields(
            predicted.reshape(batch_count * time_count, 3, predicted.shape[-1]),
            k_jax,
            poisson_sign=poisson_sign,
        ).reshape(batch_count, time_count, 4, predicted.shape[-1])
        if relative_trajectory_loss:
            if convergence_floor_jax is not None:
                sample_loss = _block_relative_sample_loss(
                    predicted_fields,
                    target_fields,
                    regime_index,
                    target_indices,
                    block_steps=relative_time_block_steps,
                    block_count=relative_time_block_count,
                    floor_rms=convergence_floor_jax,
                    trajectory_target_norm=trajectory_target_norm,
                )
            else:
                numerator = jnp.sum(
                    jnp.square(predicted_fields - target_fields), axis=(1, 3)
                )
                if trajectory_target_norm is None:
                    trajectory_target_norm = jnp.sum(
                        jnp.square(target_fields), axis=(1, 3)
                    )
                sample_loss = jnp.where(
                    jnp.all(trajectory_target_norm > 0.0, axis=1),
                    jnp.mean(numerator / trajectory_target_norm, axis=1),
                    jnp.nan,
                )
        else:
            scales = regime_scales_jax[regime_index][:, None, :, None]
            sample_loss = jnp.mean(
                jnp.square((predicted_fields - target_fields) / scales), axis=(1, 2, 3)
            )
        regime_loss = jnp.stack(
            [
                jnp.sum(jnp.where(regime_index == index, sample_loss, 0.0))
                / jnp.maximum(jnp.sum(regime_index == index), 1)
                for index in (REGIMES.index(regime) for regime in regimes)
            ]
        )
        return jnp.mean(sample_loss), (
            regime_loss,
            predicted[:, -1],
            final_memory,
        )

    return loss


def make_supervised_heat_flux_loss(
    *,
    regimes: Sequence[str] = REGIMES,
    k_arr: np.ndarray,
    width: int,
    input_scale: np.ndarray,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    poisson_sign: float,
    normalized_heat_flux_bound: float,
    memory_backend: str = "latent_recurrent",
    closure_history_input: bool = False,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
):
    """Fit the causal closure map using a fixed RMS for each complete IC."""
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    input_scale_jax = jnp.asarray(input_scale, dtype=jnp.float32)

    def loss(params, batch):
        if _uses_window_memory(memory_backend):
            closure_history = (
                batch["heat_flux_gradient_history"]
                if closure_history_input
                else None
            )
            previous_gradient = (
                batch["heat_flux_gradient_history"][:, -1]
                if closure_history_input
                else None
            )
            predicted_gradient = explicit_window_closure_step(
                params,
                batch["initial"],
                batch["memory"],
                batch["amplitude"],
                k_jax,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                previous_heat_flux_gradient=previous_gradient,
                heat_flux_gradient_history=closure_history,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
                input_scaling=input_scaling,
                dynamic_amplitude_floor=dynamic_amplitude_floor,
                allow_uniform_heating=allow_uniform_heating,
            )
        else:
            hidden = warm_spectral_memory(
                params,
                batch["memory"],
                batch["amplitude"],
                k_jax,
                width=width,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
            )
            _, predicted_gradient = spectral_memory_closure_step(
                params,
                batch["initial"],
                hidden,
                batch["amplitude"],
                k_jax,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
            )
        squared_error = jnp.mean(
            jnp.square(predicted_gradient - batch["heat_flux_gradient_target"]),
            axis=-1,
        )
        sample_loss = squared_error / jnp.square(
            batch["heat_flux_gradient_case_scale"]
        )
        regime_loss = jnp.stack(
            [
                jnp.sum(jnp.where(batch["regime_index"] == index, sample_loss, 0.0))
                / jnp.maximum(jnp.sum(batch["regime_index"] == index), 1)
                for index in (REGIMES.index(regime) for regime in regimes)
            ]
        )
        return jnp.mean(regime_loss), regime_loss

    return loss
