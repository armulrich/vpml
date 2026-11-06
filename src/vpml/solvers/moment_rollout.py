"""Coupled fluid and learned-closure integration with resumable history state.

The moment equations live in solvers.moments; model evaluation and history
encoding live in models.moment_closures and models.moment_features."""

from __future__ import annotations

from typing import Dict, Tuple

from .._jax_precision import enable_x64

enable_x64()

import jax
import jax.numpy as jnp

from ..models.moment_closures import (
    burles_latent_closure_rhs,
    explicit_window_closure_step,
    initialize_burles_latent_state,
    spectral_memory_closure_step,
)

from ..models.moment_features import (
    DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    DYNAMIC_INPUT_SCALING,
    FIXED_INPUT_SCALING,
    encode_explicit_window_history,
)

from .moments import (
    DEFAULT_DENSITY_FLOOR,
    DEFAULT_PRESSURE_FLOOR,
    _dealias,
    limit_low_moment_state,
    low_moment_rhs,
    low_moment_rk4_step,
)

Array = jax.Array


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


def rollout_burles_latent_closure(
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
    compact_latent: Array | None = None,
    previous_heat_flux_gradient: Array | None = None,
    heat_flux_gradient_history: Array | None = None,
    poisson_sign: float = 1.0,
    normalized_heat_flux_bound: float = DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    density_floor: float = DEFAULT_DENSITY_FLOOR,
    pressure_floor: float = DEFAULT_PRESSURE_FLOOR,
    scan_unroll: int = 1,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
    latent_bound: float = 8.0,
) -> Tuple[Array, Tuple[Array, Array, Array, Array, Array, Array]]:
    """Advance a Burles-compatible closure and compact learned memory.

    The accepted history is sampled at ``memory_stride``.  The first SSPRK3
    derivative uses the closure carried by the accepted state; the network is
    then evaluated on each provisional stage to supply the next derivative.
    """
    stride = int(memory_stride)
    unroll = int(scan_unroll)
    if stride <= 0 or unroll <= 0:
        raise ValueError("memory_stride and scan_unroll must be positive")
    state = jnp.asarray(initial_state)
    history = jnp.asarray(history, dtype=state.dtype)
    if previous_heat_flux_gradient is None:
        previous_heat_flux_gradient = jnp.zeros(
            (state.shape[0], state.shape[-1]), dtype=state.dtype
        )
    if heat_flux_gradient_history is None:
        heat_flux_gradient_history = jnp.zeros(
            (history.shape[0], history.shape[1], history.shape[-1]),
            dtype=state.dtype,
        )
    if compact_latent is None:
        compact_latent = initialize_burles_latent_state(
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
    dt_value = jnp.asarray(dt, dtype=state.dtype)
    latent_bound_value = jnp.asarray(latent_bound, dtype=state.dtype)

    def bounded_latent(value):
        return latent_bound_value * jnp.tanh(value / latent_bound_value)

    def physical_stage(reference, value):
        value = _dealias(value)
        return limit_low_moment_state(
            value,
            reference_state=reference,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        )

    def closure_and_memory_rhs(fluid, latent, previous, window, closure_window):
        return burles_latent_closure_rhs(
            params,
            fluid,
            latent,
            window,
            amplitude,
            k_arr,
            input_scale=input_scale,
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            previous_heat_flux_gradient=previous,
            heat_flux_gradient_history=closure_window,
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
            input_scaling=input_scaling,
            dynamic_amplitude_floor=dynamic_amplitude_floor,
            allow_uniform_heating=allow_uniform_heating,
        )

    def body(carry, _):
        fluid0, latent0, window, closure_window, counter, closure0 = carry
        closure1, latent_rhs0 = closure_and_memory_rhs(
            fluid0, latent0, closure0, window, closure_window
        )
        fluid1 = physical_stage(
            fluid0,
            fluid0
            + dt_value
            * low_moment_rhs(
                fluid0,
                closure0,
                k_arr,
                poisson_sign=poisson_sign,
                density_floor=density_floor,
                pressure_floor=pressure_floor,
            ),
        )
        latent1 = bounded_latent(latent0 + dt_value * latent_rhs0)

        closure2, latent_rhs1 = closure_and_memory_rhs(
            fluid1, latent1, closure1, window, closure_window
        )
        fluid2_candidate = fluid1 + dt_value * low_moment_rhs(
            fluid1,
            closure1,
            k_arr,
            poisson_sign=poisson_sign,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        )
        fluid2 = physical_stage(fluid0, 0.75 * fluid0 + 0.25 * fluid2_candidate)
        latent2 = bounded_latent(
            0.75 * latent0 + 0.25 * (latent1 + dt_value * latent_rhs1)
        )

        closure3, latent_rhs2 = closure_and_memory_rhs(
            fluid2, latent2, closure2, window, closure_window
        )
        fluid3_candidate = fluid2 + dt_value * low_moment_rhs(
            fluid2,
            closure2,
            k_arr,
            poisson_sign=poisson_sign,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        )
        fluid3 = physical_stage(
            fluid0, (1.0 / 3.0) * fluid0 + (2.0 / 3.0) * fluid3_candidate
        )
        latent3 = bounded_latent(
            (1.0 / 3.0) * latent0
            + (2.0 / 3.0) * (latent2 + dt_value * latent_rhs2)
        )
        closure_final, _ = closure_and_memory_rhs(
            fluid3, latent3, closure3, window, closure_window
        )

        next_counter = counter + 1
        should_sample = next_counter >= stride

        def sample_window(_):
            sampled_window = jnp.concatenate(
                (window[:, 1:], fluid3[:, None]), axis=1
            )
            sampled_closure_window = jnp.concatenate(
                (closure_window[:, 1:], closure_final[:, None]), axis=1
            )
            return (
                sampled_window,
                sampled_closure_window,
                jnp.asarray(0, dtype=jnp.int32),
            )

        def retain_window(_):
            return window, closure_window, next_counter

        window_new, closure_window_new, counter_new = jax.lax.cond(
            should_sample, sample_window, retain_window, operand=None
        )
        return (
            fluid3,
            latent3,
            window_new,
            closure_window_new,
            counter_new,
            closure_final,
        ), fluid3

    (
        _,
        final_latent,
        final_window,
        final_closure_window,
        final_counter,
        final_gradient,
    ), states = jax.lax.scan(
        jax.checkpoint(body),
        (
            state,
            jnp.asarray(compact_latent, dtype=state.dtype),
            history,
            jnp.asarray(heat_flux_gradient_history, dtype=state.dtype),
            jnp.asarray(history_counter, dtype=jnp.int32),
            jnp.asarray(previous_heat_flux_gradient, dtype=state.dtype),
        ),
        xs=None,
        length=int(horizon),
        unroll=unroll,
    )
    final_encoded = encode_explicit_window_history(
        params,
        final_window,
        k_arr,
        input_scale=input_scale,
        heat_flux_gradient_scale=heat_flux_gradient_scale,
        heat_flux_gradient_history=final_closure_window,
        poisson_sign=poisson_sign,
        input_scaling=input_scaling,
        dynamic_amplitude_floor=dynamic_amplitude_floor,
    )
    return jnp.swapaxes(states, 0, 1), (
        final_window,
        final_closure_window,
        final_encoded,
        final_counter,
        final_gradient,
        final_latent,
    )
