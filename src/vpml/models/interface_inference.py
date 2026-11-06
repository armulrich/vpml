"""Fourier-Hermite features and learned interface-flux evaluation."""

from __future__ import annotations

from typing import Optional, Tuple

from .._jax_precision import enable_x64

enable_x64()

import jax.numpy as jnp

from .interface_closure import (
    LearnedInterfaceClosure,
)

Array = jnp.ndarray


def _hermitian_rfft_weights(nk: int, *, dtype: object = jnp.float64) -> Array:
    if int(nk) <= 0:
        raise ValueError("nk must be positive")
    weights = jnp.ones((int(nk),), dtype=dtype)
    if int(nk) > 2:
        weights = weights.at[1:-1].set(2.0)
    return weights


def learned_closure_global_indicators(
    a_hat: Array,
    k_arr: Array,
    *,
    n_low: int,
) -> Tuple[Array, Array]:
    """
    Return the low-cost global state summaries used by the learned interface closure:

        E = sum_{k != 0} |E_k|^2
        H_low = sum_k sum_{n=0}^{n_low} |C_{n,k}|^2

    The sums are evaluated on the stored rFFT half-spectrum with Hermitian weights so
    they remain comparable to the full-spectrum quantities for real-valued states.
    """
    a_hat = jnp.asarray(a_hat)
    real_dtype = jnp.real(a_hat).dtype
    complex_dtype = a_hat.dtype
    k_arr = jnp.asarray(k_arr, dtype=real_dtype)
    if a_hat.ndim != 2:
        raise ValueError(f"a_hat must have shape (Nv, Nk), got {a_hat.shape}")
    if k_arr.ndim != 1 or k_arr.shape[0] != a_hat.shape[1]:
        raise ValueError(f"k_arr must have shape ({a_hat.shape[1]},), got {k_arr.shape}")
    if int(n_low) < 0:
        raise ValueError("n_low must be nonnegative")

    weights = _hermitian_rfft_weights(a_hat.shape[1], dtype=real_dtype)

    e_hat = jnp.zeros((a_hat.shape[1],), dtype=complex_dtype)
    if a_hat.shape[1] > 1:
        e_hat = e_hat.at[1:].set(1j * a_hat[0, 1:] / k_arr[1:])
    field_activity = jnp.sum(weights[1:] * (jnp.abs(e_hat[1:]) ** 2))

    n_hi = min(int(n_low), int(a_hat.shape[0]) - 1)
    low_energy = jnp.sum(weights[None, :] * (jnp.abs(a_hat[: n_hi + 1, :]) ** 2))
    return field_activity.astype(real_dtype), low_energy.astype(real_dtype)


def learned_closure_raw_base_features(
    a_hat: Array,
    k_arr: Array,
    Nv: int,
    learned: LearnedInterfaceClosure,
) -> Array:
    """Build the unstandardized current-state feature block for each nonzero rFFT mode."""
    Nv = int(Nv)
    Nm = int(learned.Nm)
    if Nm > Nv:
        raise ValueError(f"Learned closure Nm={Nm} exceeds Nv={Nv}")

    a_hat = jnp.asarray(a_hat)
    real_dtype = jnp.real(a_hat).dtype
    k_arr = jnp.asarray(k_arr, dtype=real_dtype)
    if k_arr.shape[0] <= 1:
        return jnp.zeros((0, learned.raw_base_dim), dtype=real_dtype)

    z = jnp.swapaxes(a_hat[Nv - Nm : Nv, 1:], 0, 1)
    feature_cols = [
        jnp.real(z),
        jnp.imag(z),
        jnp.abs(k_arr[1:])[:, None],
        jnp.full((z.shape[0], 1), float(Nv), dtype=real_dtype),
    ]
    if learned.include_global_indicators:
        field_activity, low_energy = learned_closure_global_indicators(
            a_hat,
            k_arr,
            n_low=int(learned.n_low),
        )
        globals_mat = jnp.broadcast_to(
            jnp.array([field_activity, low_energy], dtype=real_dtype)[None, :],
            (z.shape[0], 2),
        )
        feature_cols.append(globals_mat)
    return jnp.concatenate(feature_cols, axis=1)


def learned_closure_raw_features(
    a_hat: Array,
    k_arr: Array,
    Nv: int,
    learned: LearnedInterfaceClosure,
    *,
    a_hat_prev: Optional[Array] = None,
) -> Array:
    """Build the unstandardized learned-closure features for each nonzero rFFT mode."""
    current = learned_closure_raw_base_features(a_hat, k_arr, Nv, learned)
    if learned.context_mode == "none":
        return current
    if learned.context_mode == "lag1_delta":
        prev_state = a_hat if a_hat_prev is None else a_hat_prev
        previous = learned_closure_raw_base_features(prev_state, k_arr, Nv, learned)
        return jnp.concatenate([current, previous, current - previous], axis=1)
    raise ValueError(f"Unsupported context_mode={learned.context_mode!r}")


def scale_learned_closure_raw_features(
    raw_features: Array,
    learned: LearnedInterfaceClosure,
) -> Array:
    """Apply the k/Nv scaling used before global feature standardization."""
    raw_features = jnp.asarray(raw_features)
    real_dtype = raw_features.dtype
    k_scale = jnp.asarray(float(learned.k_scale), dtype=real_dtype)
    nv_scale = jnp.asarray(float(learned.nv_scale), dtype=real_dtype)
    base_dim = int(learned.raw_base_dim)
    k_col = 2 * int(learned.Nm)

    def scale_base_features(features: Array) -> Array:
        prefix = jnp.ones((k_col,), dtype=real_dtype)
        scaled_cols = jnp.stack((jnp.reciprocal(k_scale), jnp.reciprocal(nv_scale)))
        suffix = jnp.ones((base_dim - k_col - 2,), dtype=real_dtype)
        scale = jnp.concatenate((prefix, scaled_cols, suffix), axis=0)
        return features * scale[None, :]

    if learned.context_mode == "none":
        return scale_base_features(raw_features)
    if learned.context_mode == "lag1_delta":
        current = raw_features[:, :base_dim]
        previous = raw_features[:, base_dim : 2 * base_dim]
        current = scale_base_features(current)
        previous = scale_base_features(previous)
        delta = current - previous
        return jnp.concatenate([current, previous, delta], axis=1)
    raise ValueError(f"Unsupported context_mode={learned.context_mode!r}")


def learned_interface_q_hat(
    a_hat: Array,
    k_arr: Array,
    Nv: int,
    learned: LearnedInterfaceClosure,
    *,
    a_hat_prev: Optional[Array] = None,
) -> Array:
    """
    Return the learned interface term q_k on the rFFT half-spectrum.

    In a real-valued simulation, the negative-k values are the conjugates of the
    stored positive modes, so only the nonnegative rFFT entries need to be modeled
    explicitly here.
    """
    Nv = int(Nv)
    Nm = int(learned.Nm)
    if Nm > Nv:
        raise ValueError(f"Learned closure Nm={Nm} exceeds Nv={Nv}")

    a_hat = jnp.asarray(a_hat)
    real_dtype = jnp.real(a_hat).dtype
    complex_dtype = a_hat.dtype
    k_arr = jnp.asarray(k_arr, dtype=real_dtype)
    q_hat = jnp.zeros((k_arr.shape[0],), dtype=complex_dtype)
    if k_arr.shape[0] <= 1:
        return q_hat

    raw_features = learned_closure_raw_features(
        a_hat,
        k_arr,
        Nv,
        learned,
        a_hat_prev=a_hat_prev,
    )
    raw_features = scale_learned_closure_raw_features(raw_features, learned)
    pred = learned.predict_q_components(raw_features)
    q_nonzero = (pred[:, 0] + 1j * pred[:, 1]).astype(complex_dtype)
    return q_hat.at[1:].set(q_nonzero)


def _standardized_learned_closure_features(
    a_hat: Array,
    k_arr: Array,
    Nv: int,
    learned: LearnedInterfaceClosure,
    *,
    a_hat_prev: Optional[Array] = None,
) -> Array:
    """Return standardized nonzero-k interface-closure features."""
    raw_features = learned_closure_raw_features(
        a_hat,
        k_arr,
        int(Nv),
        learned,
        a_hat_prev=a_hat_prev,
    )
    scaled = scale_learned_closure_raw_features(raw_features, learned)
    return learned.standardized_inputs(scaled)


def learned_boundary_flux_hat(
    a_hat: Array,
    k_arr: Array,
    Nv: int,
    vth: float,
    learned: LearnedInterfaceClosure,
    *,
    a_hat_prev: Optional[Array] = None,
) -> Array:
    """Return the learned Hermite-boundary interface contribution, nonzero only on the last row."""
    del vth  # q_k is learned directly in physical units.
    a_hat = jnp.asarray(a_hat)
    q_hat = learned_interface_q_hat(a_hat, k_arr, Nv, learned, a_hat_prev=a_hat_prev)
    B_hat = jnp.zeros_like(a_hat)
    return B_hat.at[int(Nv) - 1].set(q_hat)
