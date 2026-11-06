"""The maintained Adam updates and learning-rate schedule.

The low-moment and interface optimizers intentionally retain their distinct
clipping, decay, dtype, and return-value contracts."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from vpml.core import Array


def _cosine_learning_rate(
    step: int, total_steps: int, initial_rate: float, final_rate: float
) -> float:
    if total_steps <= 1:
        return float(final_rate)
    fraction = min(max(float(step) / float(total_steps - 1), 0.0), 1.0)
    weight = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return float(final_rate + (initial_rate - final_rate) * weight)


def _adam_step(
    params,
    grads,
    state,
    learning_rate: float,
    grad_clip: float,
    weight_decay: float = 0.0,
    update_norm_cap: float = 0.0,
):
    norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(grads))
        + 1e-30
    )
    scale = (
        jnp.asarray(1.0, dtype=norm.dtype)
        if float(grad_clip) <= 0.0
        else jnp.minimum(1.0, jnp.asarray(grad_clip, norm.dtype) / norm)
    )
    grads = jax.tree_util.tree_map(lambda value: value * scale, grads)
    step = state["step"] + 1
    m = jax.tree_util.tree_map(
        lambda old, grad: 0.9 * old + 0.1 * grad, state["m"], grads
    )
    v = jax.tree_util.tree_map(
        lambda old, grad: 0.999 * old + 0.001 * grad * grad, state["v"], grads
    )
    bias_m = 1.0 - 0.9**step
    bias_v = 1.0 - 0.999**step
    updates = jax.tree_util.tree_map(
        lambda value, m_value, v_value: learning_rate
        * (
            (m_value / bias_m) / (jnp.sqrt(v_value / bias_v) + 1e-8)
            + jnp.asarray(weight_decay, dtype=value.dtype) * value
        ),
        params,
        m,
        v,
    )
    update_norm = jnp.sqrt(
        sum(
            jnp.sum(jnp.square(leaf))
            for leaf in jax.tree_util.tree_leaves(updates)
        )
        + 1e-30
    )
    if float(update_norm_cap) > 0.0:
        update_scale = jnp.minimum(
            1.0, jnp.asarray(update_norm_cap, update_norm.dtype) / update_norm
        )
        updates = jax.tree_util.tree_map(
            lambda value: value * update_scale, updates
        )
    else:
        update_scale = jnp.asarray(1.0, dtype=update_norm.dtype)
    params = jax.tree_util.tree_map(lambda value, update: value - update, params, updates)
    return (
        params,
        {"step": step, "m": m, "v": v},
        norm,
        update_norm,
        update_scale,
    )


def adam_init(params: Dict[str, Array]) -> Dict[str, object]:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": jnp.array(0, dtype=jnp.int32), "m": zeros, "v": zeros}


# Both trainers used the same initialization; retain the old private spelling.
_adam_init = adam_init


def adam_step(
    params: Dict[str, Array],
    grads: Dict[str, Array],
    state: Dict[str, object],
    lr: float,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    grad_clip: Optional[float] = None,
) -> Tuple[Dict[str, Array], Dict[str, object]]:
    lr = jnp.asarray(lr, dtype=jnp.float64)
    if grad_clip is not None:
        sq_norm = sum(jnp.sum(jnp.abs(g) ** 2) for g in jax.tree_util.tree_leaves(grads))
        norm = jnp.sqrt(jnp.maximum(sq_norm, jnp.asarray(1e-30, dtype=jnp.float64)))
        clip = jnp.asarray(float(grad_clip), dtype=jnp.float64)
        scale = jnp.minimum(jnp.asarray(1.0, dtype=jnp.float64), clip / norm)
        grads = jax.tree_util.tree_map(lambda g: scale * g, grads)

    step = state["step"] + jnp.array(1, dtype=jnp.int32)
    m = jax.tree_util.tree_map(
        lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * g_i,
        state["m"],
        grads,
    )
    v = jax.tree_util.tree_map(
        lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * (jnp.abs(g_i) ** 2),
        state["v"],
        grads,
    )
    bias1 = 1.0 - beta1 ** step
    bias2 = 1.0 - beta2 ** step
    params = jax.tree_util.tree_map(
        lambda p_i, m_i, v_i: p_i - lr * (m_i / bias1) / (jnp.sqrt(v_i / bias2) + eps),
        params,
        m,
        v,
    )
    return params, {"step": step, "m": m, "v": v}


def _tree_all_finite(tree) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True, dtype=jnp.bool_)
    checks = [jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves]
    return jnp.all(jnp.stack(checks))
