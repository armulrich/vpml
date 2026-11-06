"""Solver-embedded interface-flux rollout objectives."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from vpml.core import (
    Array,
    FourierHermiteIMEX,
    LearnedInterfaceClosure,
    learned_boundary_flux_hat,
    learned_interface_q_hat,
)
from vpml.training.interface_configuration import (
    EXACT_ROLLOUT_PRECISION_FLOAT64,
    REGIME_LINEAR,
    exact_rollout_precision_dtypes,
)
from vpml.training.interface_state import (
    build_learned_interface_closure,
    cast_learned_closure_for_rollout,
)


def _linear_explicit_n_hat_for_state(
    a_hat: Array,
    *,
    integ: FourierHermiteIMEX,
    poisson_sign: float,
) -> Array:
    complex_dtype = getattr(integ, "complex_dtype", jnp.complex128)
    e_hat = integ.E_hat_from_rho_hat(
        jnp.asarray(a_hat, dtype=complex_dtype)[0],
        poisson_sign=float(poisson_sign),
    ).astype(complex_dtype)
    n_hat = jnp.zeros_like(a_hat, dtype=complex_dtype)
    if int(integ.Nv) > 1:
        n_hat = n_hat.at[1].set(-e_hat)
    return integ.apply_mask_hat(n_hat)


def _nonlinear_explicit_n_hat_for_state(
    a_hat: Array,
    *,
    integ: FourierHermiteIMEX,
    poisson_sign: float,
) -> Array:
    real_dtype = getattr(integ, "real_dtype", jnp.float64)
    complex_dtype = getattr(integ, "complex_dtype", jnp.complex128)
    a_hat = jnp.asarray(a_hat, dtype=complex_dtype)
    m_eq = jnp.zeros((int(integ.Nv),), dtype=real_dtype).at[0].set(1.0)
    a_phys = jnp.fft.irfft(a_hat, n=int(integ.Nx), axis=1).astype(real_dtype)
    e_phys = integ.E_phys_from_a_hat(a_hat, poisson_sign=float(poisson_sign))
    n_phys = jnp.zeros_like(a_phys)
    n_phys = n_phys.at[1:].set(
        -(integ.sqrt_n[1:, None] / float(integ.vth))
        * e_phys[None, :]
        * (a_phys[:-1] + m_eq[:-1, None])
    )
    return integ.apply_mask_hat(jnp.fft.rfft(n_phys, axis=1).astype(complex_dtype))


def rollout_anchor_closure_flux_from_anchor_stencil(
    anchor_stencil: Array,
    *,
    learned: LearnedInterfaceClosure,
    integ: FourierHermiteIMEX,
    rollout_horizon: int,
    explicit_n_hat_fn,
    selected_k_index: Optional[Array] = None,
) -> Array:
    """Return q(C_h) for h=0..H-1 from a compact forward CNAB2 anchor stencil.

    The stencil stores (current, previous, previous-previous) retained states.
    """
    stencil = jnp.asarray(anchor_stencil)
    if int(stencil.shape[0]) != 3:
        raise ValueError(f"anchor_stencil must have shape (3, Nv, Nk), got {stencil.shape}")
    selected_k = None if selected_k_index is None else jnp.asarray(selected_k_index, dtype=jnp.int32)
    current_state = stencil[0]
    prev_state = stencil[1]
    prev_prev_state = stencil[2]
    n_prev = explicit_n_hat_fn(prev_state, integ=integ)
    b_prev = learned_boundary_flux_hat(
        prev_state,
        integ.k_arr,
        integ.Nv,
        integ.vth,
        learned,
        a_hat_prev=prev_prev_state,
    )

    def step(carry, _):
        state, state_prev, n_prev_step, b_prev_step = carry
        n_hat = explicit_n_hat_fn(state, integ=integ)
        q_hat = learned_interface_q_hat(
            state,
            integ.k_arr,
            integ.Nv,
            learned,
            a_hat_prev=state_prev,
        )
        b_hat = jnp.zeros_like(state).at[int(integ.Nv) - 1].set(q_hat)
        state_new = integ.step_cnab2(
            state,
            n_hat,
            n_prev_step,
            extra_hat=b_hat,
            extra_hat_prev=b_prev_step,
        )
        q_out = q_hat if selected_k is None else q_hat[selected_k]
        return (state_new, state, n_hat, b_hat), q_out

    init = (current_state, prev_state, n_prev, b_prev)
    (_, _, _, _), q_hist = jax.lax.scan(step, init, xs=None, length=int(rollout_horizon))
    return q_hist


def interface_flux_rollout_loss_for_anchor_batch(
    anchor_stencils: Array,
    ref_q_windows: Array,
    k_indices: Array,
    *,
    learned: LearnedInterfaceClosure,
    forward_integ: FourierHermiteIMEX,
    rollout_horizon: int,
    explicit_n_hat_fn,
    rollout_precision: str = EXACT_ROLLOUT_PRECISION_FLOAT64,
    loss_target_std: Optional[Array] = None,
) -> Array:
    """H-step all-k interface-flux loss over full-history anchors."""
    real_dtype, complex_dtype = exact_rollout_precision_dtypes(str(rollout_precision))
    anchor_stencils = jnp.asarray(anchor_stencils, dtype=complex_dtype)
    ref_q_windows = jnp.asarray(ref_q_windows, dtype=complex_dtype)
    if int(ref_q_windows.ndim) != 3:
        raise ValueError(
            "Canonical interface-flux training requires q windows for every Fourier mode"
        )
    if int(k_indices.ndim) != 1:
        raise ValueError("k_indices must be a one-dimensional compatibility field")
    learned_rollout = cast_learned_closure_for_rollout(
        learned,
        precision=str(rollout_precision),
    )
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for interface-flux rollout")

    if horizon == 1:
        pred_selected = jax.vmap(
            lambda anchor_stencil: learned_interface_q_hat(
                anchor_stencil[0],
                forward_integ.k_arr,
                forward_integ.Nv,
                learned_rollout,
                a_hat_prev=anchor_stencil[1],
            )[None, :]
        )(anchor_stencils)
    else:
        pred_selected = jax.vmap(
            lambda anchor_stencil: rollout_anchor_closure_flux_from_anchor_stencil(
                anchor_stencil,
                learned=learned_rollout,
                integ=forward_integ,
                rollout_horizon=horizon,
                explicit_n_hat_fn=explicit_n_hat_fn,
                selected_k_index=None,
            )
        )(anchor_stencils)
    pred_components = jnp.stack(
        [jnp.real(pred_selected[:, :, 1:]), jnp.imag(pred_selected[:, :, 1:])],
        axis=-1,
    )
    ref_components = jnp.stack(
        [jnp.real(ref_q_windows[:, :, 1:]), jnp.imag(ref_q_windows[:, :, 1:])],
        axis=-1,
    )
    std_source = (
        learned_rollout.target_std
        if loss_target_std is None
        else loss_target_std
    )
    std = jnp.maximum(jnp.asarray(std_source, dtype=real_dtype), 1e-12)
    return jnp.mean(
        ((pred_components - ref_components) / std[None, None, None, :]) ** 2
    )


def make_interface_flux_rollout_batch_loss(
    *,
    regime_weights: Dict[str, float],
    Nm: int,
    k_scale: float,
    nv_scale: float,
    stats: Dict[str, np.ndarray],
    hidden_width: int,
    res_blocks: int,
    Nv_targets: Sequence[int],
    train_regimes: Sequence[str],
    teacher_backend: str,
    teacher_Lx: float,
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    teacher_proj_Nv: Optional[int],
    projection_quadrature_Nv: Optional[int],
    n_low: int,
    context_mode: str,
    rollout_horizon: int,
    poisson_sign: float,
    rollout_dealias_23: bool,
    rollout_precision: str = EXACT_ROLLOUT_PRECISION_FLOAT64,
    regime_q_loss_stds: Optional[Dict[str, float]] = None,
    equilibrium_centered: bool = True,
    complex_normalization_mode: str = "phase_isotropic",
    translation_augmented: bool = True,
    rollout_Nx: Optional[int] = None,
) -> Tuple[object, Sequence[str]]:
    active_regimes = tuple(regime for regime in train_regimes if regime in regime_weights)
    weights = np.asarray([float(regime_weights[regime]) for regime in active_regimes], dtype=np.float64)
    weights = weights / np.sum(weights)
    weight_arr = jnp.asarray(weights, dtype=jnp.float64)
    target_nvs = tuple(int(v) for v in Nv_targets)
    retained_nx = int(teacher_Nx) if rollout_Nx is None else int(rollout_Nx)
    if retained_nx <= 1 or retained_nx > int(teacher_Nx):
        raise ValueError(
            f"rollout_Nx={retained_nx} must be greater than one and no larger "
            f"than teacher_Nx={int(teacher_Nx)}"
        )
    real_dtype, complex_dtype = exact_rollout_precision_dtypes(str(rollout_precision))
    q_loss_stds = {
        regime: jnp.full(
            (2,),
            float(regime_q_loss_stds[regime]),
            dtype=real_dtype,
        )
        for regime in active_regimes
        if regime_q_loss_stds is not None
    }
    if regime_q_loss_stds is not None:
        missing = tuple(regime for regime in active_regimes if regime not in q_loss_stds)
        if missing:
            raise ValueError(f"Missing interface-flux rollout regime loss scales for {missing!r}")
    linear_integrators = {
        int(target_nv): FourierHermiteIMEX(
            Nx=retained_nx,
            Nv=int(target_nv),
            Lx=float(teacher_Lx),
            dt=float(teacher_dt),
            vth=1.0,
            dealias_23=bool(rollout_dealias_23),
            closure=None,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
        )
        for target_nv in target_nvs
    }
    nonlinear_integrators = {
        int(target_nv): FourierHermiteIMEX(
            Nx=retained_nx,
            Nv=int(target_nv),
            Lx=float(teacher_Lx),
            dt=float(teacher_dt),
            vth=1.0,
            dealias_23=bool(rollout_dealias_23),
            closure=None,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
        )
        for target_nv in target_nvs
    }

    def make_loss_fn_for_target(target_nv: int):
        linear_forward = linear_integrators[int(target_nv)]
        nonlinear_forward = nonlinear_integrators[int(target_nv)]

        def linear_explicit(a_hat: Array, *, integ: FourierHermiteIMEX) -> Array:
            return _linear_explicit_n_hat_for_state(a_hat, integ=integ, poisson_sign=float(poisson_sign))

        def nonlinear_explicit(a_hat: Array, *, integ: FourierHermiteIMEX) -> Array:
            return _nonlinear_explicit_n_hat_for_state(a_hat, integ=integ, poisson_sign=float(poisson_sign))

        def loss_fn_for_target(
            params: Dict[str, Array],
            regime_batches: Dict[str, Dict[str, Array]],
        ) -> Tuple[Array, Dict[str, Array]]:
            learned = build_learned_interface_closure(
                params=params,
                Nm=Nm,
                k_scale=k_scale,
                nv_scale=nv_scale,
                stats=stats,
                hidden_width=hidden_width,
                res_blocks=res_blocks,
                equilibrium_centered=bool(equilibrium_centered),
                complex_normalization_mode=str(complex_normalization_mode),
                translation_augmented=bool(translation_augmented),
                Nv_targets=Nv_targets,
                train_regimes=train_regimes,
                teacher_backend=teacher_backend,
                teacher_Lx=teacher_Lx,
                teacher_Nx=teacher_Nx,
                teacher_Nv=teacher_Nv,
                teacher_vmin=teacher_vmin,
                teacher_vmax=teacher_vmax,
                teacher_dt=teacher_dt,
                teacher_proj_Nv=teacher_proj_Nv,
                projection_quadrature_Nv=projection_quadrature_Nv,
                n_low=n_low,
                rollout_horizon=rollout_horizon,
                rollout_Nx=retained_nx,
            )
            total_q = jnp.asarray(0.0, dtype=jnp.float64)
            for weight, regime in zip(weight_arr, active_regimes):
                batch = regime_batches[regime]
                if regime == REGIME_LINEAR:
                    q_loss = interface_flux_rollout_loss_for_anchor_batch(
                        batch["anchor_stencils"],
                        batch["ref_q_windows"],
                        batch["k_indices"],
                        learned=learned,
                        forward_integ=linear_forward,
                        rollout_horizon=rollout_horizon,
                        explicit_n_hat_fn=linear_explicit,
                        rollout_precision=str(rollout_precision),
                        loss_target_std=q_loss_stds.get(regime),
                    )
                else:
                    q_loss = interface_flux_rollout_loss_for_anchor_batch(
                        batch["anchor_stencils"],
                        batch["ref_q_windows"],
                        batch["k_indices"],
                        learned=learned,
                        forward_integ=nonlinear_forward,
                        rollout_horizon=rollout_horizon,
                        explicit_n_hat_fn=nonlinear_explicit,
                        rollout_precision=str(rollout_precision),
                        loss_target_std=q_loss_stds.get(regime),
                    )
                total_q = total_q + weight * q_loss
            return total_q, {
                "q": total_q,
            }

        return loss_fn_for_target

    target_loss_fns = {
        int(target_nv): make_loss_fn_for_target(int(target_nv))
        for target_nv in target_nvs
    }
    default_target_nv = int(target_nvs[0])

    def loss_fn(
        params: Dict[str, Array],
        regime_batches: Dict[str, Dict[str, Array]],
    ) -> Tuple[Array, Dict[str, Array]]:
        return target_loss_fns[default_target_nv](params, regime_batches)

    loss_fn.target_nvs = target_nvs  # type: ignore[attr-defined]
    loss_fn.target_loss_fns = target_loss_fns  # type: ignore[attr-defined]
    loss_fn.rollout_precision = str(rollout_precision)  # type: ignore[attr-defined]
    loss_fn.translation_augmented = bool(translation_augmented)  # type: ignore[attr-defined]
    loss_fn.regime_q_loss_stds = (
        {}
        if regime_q_loss_stds is None
        else {str(key): float(value) for key, value in regime_q_loss_stds.items()}
    )  # type: ignore[attr-defined]
    loss_fn.teacher_Lx = float(teacher_Lx)  # type: ignore[attr-defined]
    loss_fn.rollout_Nx = retained_nx  # type: ignore[attr-defined]
    return loss_fn, active_regimes
