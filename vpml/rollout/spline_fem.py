"""Spline-grid online rollout backend for coarse physical-space closure tests.

This module intentionally sits beside the Fourier-Hermite online rollout path.
It uses the existing cubic-spline physical-grid Vlasov-Poisson stepper as the
coarse propagator and adds a learned physical-space residual to the low-grid
state update. The goal is to test the same online-rollout idea without a
Fourier-Hermite boundary flux.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

from vpml.jax_runtime import bootstrap_jax_runtime

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp

from vpml.core import Array
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    _physical_grid_ops,
    advect_v_cubic,
    advect_x_cubic,
    compute_electric_field_from_distribution,
    cubic_bspline_interp_constant,
    cubic_bspline_interp_periodic,
    cubic_bspline_prefilter_constant,
    cubic_bspline_prefilter_periodic,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


SPLINE_RESIDUAL_INPUT_DIM = 7


def maxwellian_on_grid(v: Array) -> Array:
    v = jnp.asarray(v, dtype=jnp.float64)
    return (jnp.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi)).astype(jnp.float64)


def restrict_state_to_grid(
    f_state: Array,
    src: PhysicalGridVlasovPoissonConfig,
    dst: PhysicalGridVlasovPoissonConfig,
    *,
    src_ops: Dict[str, Array] | None = None,
) -> Array:
    """Cubic-spline restrict/interpolate one physical-grid state onto another."""
    if not math.isclose(float(src.Lx), float(dst.Lx)):
        raise ValueError("restrict_state_to_grid currently requires matching Lx")
    src_ops = _physical_grid_ops(src) if src_ops is None else src_ops
    f_state = jnp.asarray(f_state, dtype=jnp.float64)

    v_coords_1d = (dst.v - float(src.vmin)) / float(src.dv)
    v_coords = jnp.broadcast_to(v_coords_1d[:, None], (int(dst.Nv), int(src.Nx)))
    v_coeffs = cubic_bspline_prefilter_constant(
        f_state,
        src_ops["v_prefilter_sub"],
        src_ops["v_prefilter_diag"],
        src_ops["v_prefilter_sup"],
    )
    f_on_dst_v = cubic_bspline_interp_constant(v_coeffs, v_coords, cval=0.0)

    x_coords_1d = jnp.mod(dst.x / float(src.dx), float(src.Nx))
    x_coords = jnp.broadcast_to(x_coords_1d[None, :], (int(dst.Nv), int(dst.Nx)))
    x_coeffs = cubic_bspline_prefilter_periodic(f_on_dst_v, src_ops["periodic_den"])
    return cubic_bspline_interp_periodic(x_coeffs, x_coords).astype(jnp.float64)


def restrict_history_to_grid(
    f_history: Array,
    src: PhysicalGridVlasovPoissonConfig,
    dst: PhysicalGridVlasovPoissonConfig,
) -> Array:
    """Restrict a physical-grid history from the HR grid to a coarse grid."""
    src_ops = _physical_grid_ops(src)
    return jax.vmap(lambda f: restrict_state_to_grid(f, src, dst, src_ops=src_ops))(
        jnp.asarray(f_history, dtype=jnp.float64)
    )


def init_spline_residual_params(
    key: Array,
    *,
    input_dim: int = SPLINE_RESIDUAL_INPUT_DIM,
    hidden_width: int = 64,
    res_blocks: int = 2,
) -> Dict[str, Array | Sequence[Dict[str, Array]]]:
    """Initialize a small pointwise residual MLP for the spline-grid backend."""
    keys = jax.random.split(key, 3 + 2 * int(res_blocks))
    params: Dict[str, Array | Sequence[Dict[str, Array]]] = {
        "W0": jax.random.normal(keys[0], (int(input_dim), int(hidden_width)), dtype=jnp.float64)
        / math.sqrt(float(input_dim)),
        "b0": jnp.zeros((int(hidden_width),), dtype=jnp.float64),
    }
    blocks = []
    for i in range(int(res_blocks)):
        blocks.append(
            {
                "W1": jax.random.normal(
                    keys[1 + 2 * i],
                    (int(hidden_width), int(hidden_width)),
                    dtype=jnp.float64,
                )
                / math.sqrt(float(hidden_width)),
                "b1": jnp.zeros((int(hidden_width),), dtype=jnp.float64),
                "W2": jax.random.normal(
                    keys[2 + 2 * i],
                    (int(hidden_width), int(hidden_width)),
                    dtype=jnp.float64,
                )
                / math.sqrt(float(hidden_width)),
                "b2": jnp.zeros((int(hidden_width),), dtype=jnp.float64),
            }
        )
    params["blocks"] = tuple(blocks)
    params["Wout"] = jnp.zeros((int(hidden_width), 1), dtype=jnp.float64)
    params["bout"] = jnp.zeros((1,), dtype=jnp.float64)
    return params


def _silu(x: Array) -> Array:
    return x * jax.nn.sigmoid(x)


def _residual_mlp_apply(params: Dict[str, object], features: Array) -> Array:
    h = _silu(features @ jnp.asarray(params["W0"]) + jnp.asarray(params["b0"]))
    for block in params["blocks"]:
        h = h + (
            _silu(h @ jnp.asarray(block["W1"]) + jnp.asarray(block["b1"]))
            @ jnp.asarray(block["W2"])
            + jnp.asarray(block["b2"])
        )
    return (h @ jnp.asarray(params["Wout"]) + jnp.asarray(params["bout"]))[:, 0]


def spline_residual_features(
    f_state: Array,
    config: PhysicalGridVlasovPoissonConfig,
    *,
    ops: Dict[str, Array] | None = None,
) -> Array:
    """Pointwise features for a physical-space residual closure."""
    ops = _physical_grid_ops(config) if ops is None else ops
    f_state = jnp.asarray(f_state, dtype=jnp.float64)
    eq = maxwellian_on_grid(ops["v"])[:, None]
    delta_f = f_state - eq
    v_scale = max(abs(float(config.vmin)), abs(float(config.vmax)), 1.0)
    v_norm = ops["v"][:, None] / float(v_scale)
    theta = 2.0 * math.pi * ops["x"][None, :] / float(config.Lx)
    e_phys = compute_electric_field_from_distribution(f_state, config, ops=ops)[None, :]
    rho = jnp.trapezoid(f_state, x=ops["v"], axis=0)
    rho_p = (rho - jnp.mean(rho))[None, :]
    features = jnp.stack(
        [
            delta_f,
            f_state,
            jnp.broadcast_to(v_norm, f_state.shape),
            jnp.broadcast_to(jnp.sin(theta), f_state.shape),
            jnp.broadcast_to(jnp.cos(theta), f_state.shape),
            jnp.broadcast_to(e_phys, f_state.shape),
            jnp.broadcast_to(rho_p, f_state.shape),
        ],
        axis=-1,
    )
    return features.reshape((-1, SPLINE_RESIDUAL_INPUT_DIM)).astype(jnp.float64)


def spline_residual(
    params: Dict[str, object],
    f_state: Array,
    config: PhysicalGridVlasovPoissonConfig,
    *,
    ops: Dict[str, Array] | None = None,
) -> Array:
    """Evaluate a mass-neutral learned coarse residual on the coarse grid."""
    features = spline_residual_features(f_state, config, ops=ops)
    raw = _residual_mlp_apply(params, features).reshape((int(config.Nv), int(config.Nx)))
    # Avoid a learned residual that changes total particle number at leading order.
    return (raw - jnp.mean(raw)).astype(jnp.float64)


def spline_fem_base_step(
    f_state: Array,
    config: PhysicalGridVlasovPoissonConfig,
    *,
    ops: Dict[str, Array] | None = None,
) -> Array:
    """One coarse cubic-spline semi-Lagrangian Vlasov-Poisson step."""
    return spline_fem_base_step_dt(f_state, config, float(config.dt), ops=ops)


def spline_fem_base_step_dt(
    f_state: Array,
    config: PhysicalGridVlasovPoissonConfig,
    dt: float,
    *,
    ops: Dict[str, Array] | None = None,
) -> Array:
    """One cubic-spline semi-Lagrangian step with an explicit signed time step."""
    ops = _physical_grid_ops(config) if ops is None else ops
    f_half = advect_x_cubic(f_state, config, ops, 0.5 * float(dt))
    e_mid = compute_electric_field_from_distribution(f_half, config, ops=ops)
    f_vel = advect_v_cubic(f_half, config, ops, e_mid, float(dt))
    return advect_x_cubic(f_vel, config, ops, 0.5 * float(dt)).astype(jnp.float64)


def spline_fem_step_with_residual(
    f_state: Array,
    params: Dict[str, object],
    config: PhysicalGridVlasovPoissonConfig,
    *,
    ops: Dict[str, Array] | None = None,
) -> Array:
    """One coarse spline-grid step plus a learned coarse residual."""
    ops = _physical_grid_ops(config) if ops is None else ops
    base_next = spline_fem_base_step(f_state, config, ops=ops)
    correction = spline_residual(params, f_state, config, ops=ops)
    return (base_next + float(config.dt) * correction).astype(jnp.float64)


def spline_fem_step_with_residual_dt(
    f_state: Array,
    params: Dict[str, object],
    config: PhysicalGridVlasovPoissonConfig,
    dt: float,
    *,
    ops: Dict[str, Array] | None = None,
) -> Array:
    """One signed coarse step with the residual inserted using the signed dt."""
    ops = _physical_grid_ops(config) if ops is None else ops
    base_next = spline_fem_base_step_dt(f_state, config, float(dt), ops=ops)
    correction = spline_residual(params, f_state, config, ops=ops)
    return (base_next + float(dt) * correction).astype(jnp.float64)


def physical_l2_norm_sq(
    f_state: Array,
    config: PhysicalGridVlasovPoissonConfig,
    *,
    ops: Dict[str, Array] | None = None,
) -> Array:
    ops = _physical_grid_ops(config) if ops is None else ops
    f_state = jnp.asarray(f_state, dtype=jnp.float64)
    v_int = jnp.trapezoid(f_state * f_state, x=ops["v"], axis=0)
    return (float(config.dx) * jnp.sum(v_int)).astype(jnp.float64)


def select_rollout_anchor_indices(
    *,
    history_length: int,
    rollout_horizon: int,
    rollout_anchor_samples: int,
) -> Array:
    max_anchor = int(history_length) - int(rollout_horizon) - 1
    if max_anchor < 0:
        raise ValueError(
            f"history_length={history_length} is too short for rollout_horizon={rollout_horizon}"
        )
    if int(rollout_anchor_samples) <= 0 or int(rollout_anchor_samples) >= max_anchor + 1:
        return jnp.arange(0, max_anchor + 1, dtype=jnp.int32)
    return jnp.rint(
        jnp.linspace(0.0, float(max_anchor), int(rollout_anchor_samples))
    ).astype(jnp.int32)


def spline_fem_rollout_loss(
    params: Dict[str, object],
    ref_history: Array,
    config: PhysicalGridVlasovPoissonConfig,
    *,
    rollout_horizon: int,
    rollout_anchor_samples: int,
) -> Array:
    """Relative online rollout loss in the low-grid perturbation norm."""
    ops = _physical_grid_ops(config)
    ref_history = jnp.asarray(ref_history, dtype=jnp.float64)
    equilibrium = maxwellian_on_grid(ops["v"])[:, None]
    horizon = int(rollout_horizon)
    anchors = select_rollout_anchor_indices(
        history_length=int(ref_history.shape[0]),
        rollout_horizon=horizon,
        rollout_anchor_samples=int(rollout_anchor_samples),
    )

    def one_anchor(anchor_idx: Array) -> Array:
        initial = ref_history[anchor_idx]

        def step(carry: Array, _unused: Array) -> tuple[Array, Array]:
            nxt = spline_fem_step_with_residual(carry, params, config, ops=ops)
            return nxt, nxt

        _, pred_hist = jax.lax.scan(step, initial, xs=None, length=horizon)
        offsets = jnp.arange(1, horizon + 1, dtype=jnp.int32)
        ref = jnp.take(ref_history, anchor_idx + offsets, axis=0)

        def rel_one(pred: Array, target: Array) -> Array:
            num = physical_l2_norm_sq(pred - target, config, ops=ops)
            den = physical_l2_norm_sq(target - equilibrium, config, ops=ops)
            return num / (den + 1.0e-30)

        return jnp.mean(jax.vmap(rel_one)(pred_hist, ref))

    return jnp.mean(jax.vmap(one_anchor)(anchors)).astype(jnp.float64)


def spline_fem_teacher_lifted_rollout_loss(
    params: Dict[str, object],
    low_anchors: Array,
    teacher_targets_fwd: Array,
    teacher_targets_bwd: Array,
    low_config: PhysicalGridVlasovPoissonConfig,
    teacher_config: PhysicalGridVlasovPoissonConfig,
    *,
    backward_weight: float = 1.0,
) -> Array:
    """Relative online loss after lifting LR rollouts back to the fixed HR grid.

    ``low_anchors`` has shape ``(B, M_v, N_x^lo)``. The target arrays have shape
    ``(B, H, N_v^HR, N_x^HR)``. The model is still evolved entirely on the low
    grid; only the predicted states used inside the loss are interpolated back
    to the teacher grid.
    """
    low_ops = _physical_grid_ops(low_config)
    teacher_ops = _physical_grid_ops(teacher_config)
    low_anchors = jnp.asarray(low_anchors, dtype=jnp.float64)
    teacher_targets_fwd = jnp.asarray(teacher_targets_fwd, dtype=jnp.float64)
    teacher_targets_bwd = jnp.asarray(teacher_targets_bwd, dtype=jnp.float64)
    equilibrium_hr = maxwellian_on_grid(teacher_ops["v"])[:, None]
    horizon = int(teacher_targets_fwd.shape[1])

    def rel_hr_error(low_pred: Array, target_hr: Array) -> Array:
        pred_hr = restrict_state_to_grid(
            low_pred,
            low_config,
            teacher_config,
            src_ops=low_ops,
        )
        num = physical_l2_norm_sq(pred_hr - target_hr, teacher_config, ops=teacher_ops)
        den = physical_l2_norm_sq(target_hr - equilibrium_hr, teacher_config, ops=teacher_ops)
        return num / (den + 1.0e-30)

    def one_window(anchor: Array, targets_fwd: Array, targets_bwd: Array) -> Array:
        def step_fwd(carry: Array, _unused: Array) -> tuple[Array, Array]:
            nxt = spline_fem_step_with_residual_dt(
                carry,
                params,
                low_config,
                float(low_config.dt),
                ops=low_ops,
            )
            return nxt, nxt

        def step_bwd(carry: Array, _unused: Array) -> tuple[Array, Array]:
            nxt = spline_fem_step_with_residual_dt(
                carry,
                params,
                low_config,
                -float(low_config.dt),
                ops=low_ops,
            )
            return nxt, nxt

        _, pred_fwd = jax.lax.scan(step_fwd, anchor, xs=None, length=horizon)
        _, pred_bwd = jax.lax.scan(step_bwd, anchor, xs=None, length=horizon)
        loss_fwd = jnp.mean(jax.vmap(rel_hr_error)(pred_fwd, targets_fwd))
        loss_bwd = jnp.mean(jax.vmap(rel_hr_error)(pred_bwd, targets_bwd))
        weight = max(float(backward_weight), 0.0)
        return (loss_fwd + weight * loss_bwd) / (1.0 + weight)

    return jnp.mean(jax.vmap(one_window)(low_anchors, teacher_targets_fwd, teacher_targets_bwd)).astype(
        jnp.float64
    )


def spline_fem_lr_teacher_rollout_loss(
    params: Dict[str, object],
    low_anchors: Array,
    low_targets_fwd: Array,
    low_targets_bwd: Array,
    low_config: PhysicalGridVlasovPoissonConfig,
    *,
    backward_weight: float = 1.0,
) -> Array:
    """H-step closure-improvement loss against the restricted teacher.

    The rollout is evolved entirely on the low grid, and each prediction is
    compared to ``R_M F^{HR}`` on that same low grid. Lifting to the HR grid is
    left to evaluation/plotting. The denominator is the corresponding
    no-correction coarse-solver defect, so a no-correction model scores about
    one and useful corrections must beat the baseline coarse solver.
    """
    low_ops = _physical_grid_ops(low_config)
    low_anchors = jnp.asarray(low_anchors, dtype=jnp.float64)
    low_targets_fwd = jnp.asarray(low_targets_fwd, dtype=jnp.float64)
    low_targets_bwd = jnp.asarray(low_targets_bwd, dtype=jnp.float64)
    horizon = int(low_targets_fwd.shape[1])

    def improvement_error(pred: Array, baseline: Array, target: Array) -> Array:
        num = physical_l2_norm_sq(pred - target, low_config, ops=low_ops)
        base = physical_l2_norm_sq(baseline - target, low_config, ops=low_ops)
        scale = physical_l2_norm_sq(target, low_config, ops=low_ops)
        return num / (base + 1.0e-12 * scale + 1.0e-30)

    def one_window(anchor: Array, targets_fwd: Array, targets_bwd: Array) -> Array:
        def step_fwd(carry: Array, _unused: Array) -> tuple[Array, Array]:
            nxt = spline_fem_step_with_residual_dt(
                carry,
                params,
                low_config,
                float(low_config.dt),
                ops=low_ops,
            )
            return nxt, nxt

        def base_step_fwd(carry: Array, _unused: Array) -> tuple[Array, Array]:
            nxt = spline_fem_base_step_dt(
                carry,
                low_config,
                float(low_config.dt),
                ops=low_ops,
            )
            return nxt, nxt

        def step_bwd(carry: Array, _unused: Array) -> tuple[Array, Array]:
            nxt = spline_fem_step_with_residual_dt(
                carry,
                params,
                low_config,
                -float(low_config.dt),
                ops=low_ops,
            )
            return nxt, nxt

        def base_step_bwd(carry: Array, _unused: Array) -> tuple[Array, Array]:
            nxt = spline_fem_base_step_dt(
                carry,
                low_config,
                -float(low_config.dt),
                ops=low_ops,
            )
            return nxt, nxt

        _, pred_fwd = jax.lax.scan(step_fwd, anchor, xs=None, length=horizon)
        _, pred_bwd = jax.lax.scan(step_bwd, anchor, xs=None, length=horizon)
        _, base_fwd = jax.lax.scan(base_step_fwd, anchor, xs=None, length=horizon)
        _, base_bwd = jax.lax.scan(base_step_bwd, anchor, xs=None, length=horizon)
        loss_fwd = jnp.mean(jax.vmap(improvement_error)(pred_fwd, base_fwd, targets_fwd))
        loss_bwd = jnp.mean(jax.vmap(improvement_error)(pred_bwd, base_bwd, targets_bwd))
        weight = max(float(backward_weight), 0.0)
        return (loss_fwd + weight * loss_bwd) / (1.0 + weight)

    return jnp.mean(jax.vmap(one_window)(low_anchors, low_targets_fwd, low_targets_bwd)).astype(
        jnp.float64
    )


def spline_fem_lr_teacher_defect_loss(
    params: Dict[str, object],
    low_anchors: Array,
    low_targets_fwd: Array,
    low_targets_bwd: Array,
    low_config: PhysicalGridVlasovPoissonConfig,
    *,
    backward_weight: float = 1.0,
) -> Array:
    """Direct coarse-defect loss against the restricted HR teacher.

    For each teacher state in the stored window, this loss compares the learned
    signed ``dt * residual`` correction with the exact one-step LR defect

        R_M F_{m +/- 1}^{HR} - Phi_{+/- dt}^M(R_M F_m^{HR}).

    This is the spline-grid analogue of supervising the Hermite interface target:
    the model is trained on the missing coarse update itself, not only on the
    downstream H-step state error.
    """
    low_ops = _physical_grid_ops(low_config)
    low_anchors = jnp.asarray(low_anchors, dtype=jnp.float64)
    low_targets_fwd = jnp.asarray(low_targets_fwd, dtype=jnp.float64)
    low_targets_bwd = jnp.asarray(low_targets_bwd, dtype=jnp.float64)
    horizon = int(low_targets_fwd.shape[1])

    def defect_error(state: Array, target_next: Array, dt: float) -> Array:
        base_next = spline_fem_base_step_dt(state, low_config, float(dt), ops=low_ops)
        exact_defect = target_next - base_next
        predicted_defect = float(dt) * spline_residual(params, state, low_config, ops=low_ops)
        num = physical_l2_norm_sq(predicted_defect - exact_defect, low_config, ops=low_ops)
        den = physical_l2_norm_sq(exact_defect, low_config, ops=low_ops)
        scale = physical_l2_norm_sq(target_next, low_config, ops=low_ops)
        return num / (den + 1.0e-12 * scale + 1.0e-30)

    def one_window(anchor: Array, targets_fwd: Array, targets_bwd: Array) -> Array:
        if horizon > 1:
            fwd_states = jnp.concatenate([anchor[None, :, :], targets_fwd[:-1]], axis=0)
            bwd_states = jnp.concatenate([anchor[None, :, :], targets_bwd[:-1]], axis=0)
        else:
            fwd_states = anchor[None, :, :]
            bwd_states = anchor[None, :, :]

        loss_fwd = jnp.mean(
            jax.vmap(lambda state, target: defect_error(state, target, float(low_config.dt)))(
                fwd_states,
                targets_fwd,
            )
        )
        loss_bwd = jnp.mean(
            jax.vmap(lambda state, target: defect_error(state, target, -float(low_config.dt)))(
                bwd_states,
                targets_bwd,
            )
        )
        weight = max(float(backward_weight), 0.0)
        return (loss_fwd + weight * loss_bwd) / (1.0 + weight)

    return jnp.mean(jax.vmap(one_window)(low_anchors, low_targets_fwd, low_targets_bwd)).astype(
        jnp.float64
    )
