"""
Train a shared learned interface closure for Landau-family runs using a selectable teacher.

The grid-cubic-spline teacher is a full Vlasov-Poisson semi-Lagrangian solve on a fine
(x, v) grid with JAX cubic spline interpolation. Teacher snapshots are projected onto
the Fourier-Hermite basis, and the learned target is

    q_k^* = -i k v_th sqrt(Nv) C_{Nv,k}^{HR}.

The higher-order-Hermite teacher uses direct Hermite-space rollouts to produce the same
coefficient histories without a projection step. The current trainer preserves the q-only
path and also supports pure online-rollout and online-hybrid training modes.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from vpml.jax_runtime import bootstrap_jax_runtime, print_jax_runtime_summary

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))

from vpml.nonlinear_landau import (
    NonlinearLandauParams,
    run_nonlinear_landau_rollout_raw,
)
from vpml.core import (
    Array,
    FourierHermiteIMEX,
    GRID_CUBIC_SPLINE_TEACHER_BACKEND,
    HIGHER_ORDER_HERMITE_TEACHER_BACKEND,
    LearnedInterfaceClosure,
    e_hat_history_from_a_hat_history,
    init_interface_closure_params,
    irfft_x,
    learned_boundary_flux_hat,
    learned_interface_q_hat,
    load_learned_interface_closure_npz,
    normalize_teacher_backend_name,
    rfft_x,
    scale_learned_closure_raw_features,
    save_learned_interface_closure_npz,
)
from vpml.linear_landau import LinearLandauConfig, linear_explicit_N_hat, run_linear_landau_cnab2_raw
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    compute_electric_field_from_distribution,
    cubic_bspline_interp_constant,
    cubic_bspline_prefilter_constant,
    extract_interface_supervised_pairs_from_coeff_history,
    gaussian_pdf,
    hermite_dual_basis_scaled,
    normalize_density_on_grid,
    project_distribution_snapshot_to_fourier_hermite,
    run_semilagrangian_vlasov_poisson,
)
from vpml.visualization.training import save_training_loss_plot, save_training_loss_q_diagnostic_plot

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass

REGIME_LINEAR = "linear_landau"
REGIME_WEAK = "nonlinear_landau_weak"
REGIME_STRONG = "nonlinear_landau_strong"
ALL_REGIMES = (REGIME_LINEAR, REGIME_WEAK, REGIME_STRONG)
CACHE_FORMAT = "landau_interface_dataset_teacher_v6"
ONLINE_REFERENCE_CACHE_FORMAT = "landau_interface_online_reference_v4"
ONLINE_HYBRID_LOSS_DEFINITION = "q_trajectory_field_distribution_v1"
ONLINE_TRAINING_MODE = "online_rollout"
OFFLINE_TRAINING_MODE = "offline_rollout"
ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1 = "field_distribution_v1"
ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR = "fourier_hermite_bidir"
ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_BIDIR = "fourier_hermite_closure_bidir"
ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_DETACHED_BIDIR = "fourier_hermite_closure_detached_bidir"
ONLINE_LOSS_BACKEND_FOURIER_HERMITE_ROLLOUT_QLOSS = "fourier_hermite_rollout_qloss"
ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_ACTION_BIDIR = "fourier_hermite_closure_action_bidir"
ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BOUNDARY_STEP_BIDIR = "fourier_hermite_boundary_step_bidir"
ONLINE_LOSS_BACKEND_FOURIER_HERMITE_POSTERIOR_BIDIR = "fourier_hermite_posterior_bidir"
ONLINE_LOSS_BACKEND_FOURIER_HERMITE_PROJECTED_XV_BIDIR = "fourier_hermite_projected_xv_bidir"
ROLLOUT_ANCHOR_INDICES_KEY = "__rollout_anchor_indices"
ONLINE_ROLLOUT_DIRECTION_BIDIR = "bidir"
ONLINE_ROLLOUT_DIRECTION_FORWARD = "forward"
ALL_ONLINE_ROLLOUT_DIRECTIONS = (
    ONLINE_ROLLOUT_DIRECTION_BIDIR,
    ONLINE_ROLLOUT_DIRECTION_FORWARD,
)
PROJECTED_XV_METRIC_PHYSICAL_L2 = "physical_l2"
PROJECTED_XV_METRIC_GRAM_RIESZ = "gram_riesz"
ALL_PROJECTED_XV_METRICS = (
    PROJECTED_XV_METRIC_PHYSICAL_L2,
    PROJECTED_XV_METRIC_GRAM_RIESZ,
)
ALL_ONLINE_LOSS_BACKENDS = (
    ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1,
    ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR,
    ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_BIDIR,
    ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_DETACHED_BIDIR,
    ONLINE_LOSS_BACKEND_FOURIER_HERMITE_ROLLOUT_QLOSS,
    ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_ACTION_BIDIR,
    ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BOUNDARY_STEP_BIDIR,
    ONLINE_LOSS_BACKEND_FOURIER_HERMITE_POSTERIOR_BIDIR,
    ONLINE_LOSS_BACKEND_FOURIER_HERMITE_PROJECTED_XV_BIDIR,
)
ALL_TEACHER_BACKENDS = (
    GRID_CUBIC_SPLINE_TEACHER_BACKEND,
    HIGHER_ORDER_HERMITE_TEACHER_BACKEND,
)


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def parse_str_tuple(text: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def online_reference_coeff_key(target_nv: int) -> str:
    return f"a_hat_ref_nv{int(target_nv)}"


def online_reference_anchor_coeff_key(target_nv: int) -> str:
    return f"a_hat_anchor_nv{int(target_nv)}"


def online_reference_anchor_index_key(target_nv: int) -> str:
    return f"anchor_index_nv{int(target_nv)}"


def online_reference_q_key(target_nv: int) -> str:
    return f"q_hat_ref_nv{int(target_nv)}"


def online_loss_backend_uses_projected_coefficients(backend: str) -> bool:
    return str(backend) in {
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_DETACHED_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_ROLLOUT_QLOSS,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_ACTION_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BOUNDARY_STEP_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_POSTERIOR_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_PROJECTED_XV_BIDIR,
    }


def online_loss_backend_uses_closure_q(backend: str) -> bool:
    return str(backend) in {
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_DETACHED_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_ROLLOUT_QLOSS,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_ACTION_BIDIR,
    }


def online_loss_backend_uses_rollout_qloss(backend: str) -> bool:
    return str(backend) == ONLINE_LOSS_BACKEND_FOURIER_HERMITE_ROLLOUT_QLOSS


def online_loss_backend_uses_action_q(backend: str) -> bool:
    return str(backend) == ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_ACTION_BIDIR


def online_loss_backend_uses_boundary_step(backend: str) -> bool:
    return str(backend) == ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BOUNDARY_STEP_BIDIR


def online_loss_backend_uses_posterior_rollout(backend: str) -> bool:
    return str(backend) == ONLINE_LOSS_BACKEND_FOURIER_HERMITE_POSTERIOR_BIDIR


def online_loss_backend_uses_projected_xv(backend: str) -> bool:
    return str(backend) == ONLINE_LOSS_BACKEND_FOURIER_HERMITE_PROJECTED_XV_BIDIR


def online_loss_backend_has_reference_q_targets(backend: str) -> bool:
    return online_loss_backend_uses_closure_q(str(backend)) or online_loss_backend_uses_projected_xv(str(backend))


def online_reference_num_cases(payload: Dict[str, Array]) -> int:
    if "E_hat_ref" in payload:
        return int(np.asarray(payload["E_hat_ref"]).shape[0])
    for key, value in payload.items():
        if str(key).startswith("a_hat_ref_nv"):
            return int(np.asarray(value).shape[0])
    for key, value in payload.items():
        if str(key).startswith("a_hat_anchor_nv"):
            return int(np.asarray(value).shape[0])
    for key, value in payload.items():
        if str(key).startswith("q_hat_ref_nv"):
            return int(np.asarray(value).shape[0])
    return 0


def build_dataset_cache_metadata(
    *,
    regimes: Sequence[str],
    teacher_backend: str,
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_L: float,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    teacher_proj_Nv: Optional[int],
    linear_T: float,
    linear_eps: float,
    linear_modes: Sequence[float],
    linear_num_samples: int,
    linear_seed: int,
    linear_poisson_sign: float,
    linear_history_stride: int,
    nonlinear_T: float,
    nonlinear_k0: float,
    nonlinear_poisson_sign: float,
    nonlinear_history_stride: int,
    weak_eps: Sequence[float],
    strong_eps: Sequence[float],
    Nv_targets: Sequence[int],
    Nm: int,
    val_fraction: float,
    n_low: int,
    context_mode: str = "none",
    projection_mode: str = "shared_max",
    teacher_proj_Nv_targets: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    payload = {
        "dataset_format": np.array([CACHE_FORMAT], dtype=np.str_),
        "regimes": np.asarray(tuple(regimes), dtype=np.str_),
        "n_low": np.array([int(n_low)], dtype=np.int32),
        "Nm": np.array([int(Nm)], dtype=np.int32),
        "Nv_targets": np.asarray(tuple(int(v) for v in Nv_targets), dtype=np.int32),
        "teacher_backend": np.array([str(teacher_backend)], dtype=np.str_),
        "context_mode": np.array([str(context_mode)], dtype=np.str_),
        "projection_mode": np.array([str(projection_mode)], dtype=np.str_),
        "teacher_Nx": np.array([int(teacher_Nx)], dtype=np.int32),
        "teacher_Nv": np.array([int(teacher_Nv)], dtype=np.int32),
        "teacher_L": np.array([float(teacher_L)], dtype=np.float64),
        "teacher_vmin": np.array([float(teacher_vmin)], dtype=np.float64),
        "teacher_vmax": np.array([float(teacher_vmax)], dtype=np.float64),
        "teacher_dt": np.array([float(teacher_dt)], dtype=np.float64),
        "linear_T": np.array([float(linear_T)], dtype=np.float64),
        "linear_eps": np.array([float(linear_eps)], dtype=np.float64),
        "linear_modes": np.asarray(tuple(float(v) for v in linear_modes), dtype=np.float64),
        "linear_num_samples": np.array([int(linear_num_samples)], dtype=np.int32),
        "linear_seed": np.array([int(linear_seed)], dtype=np.int32),
        "linear_poisson_sign": np.array([float(linear_poisson_sign)], dtype=np.float64),
        "linear_history_stride": np.array([int(linear_history_stride)], dtype=np.int32),
        "nonlinear_T": np.array([float(nonlinear_T)], dtype=np.float64),
        "nonlinear_k0": np.array([float(nonlinear_k0)], dtype=np.float64),
        "nonlinear_poisson_sign": np.array([float(nonlinear_poisson_sign)], dtype=np.float64),
        "nonlinear_history_stride": np.array([int(nonlinear_history_stride)], dtype=np.int32),
        "weak_eps": np.asarray(tuple(float(v) for v in weak_eps), dtype=np.float64),
        "strong_eps": np.asarray(tuple(float(v) for v in strong_eps), dtype=np.float64),
        "val_fraction": np.array([float(val_fraction)], dtype=np.float64),
    }
    if teacher_proj_Nv is not None:
        payload["teacher_proj_Nv"] = np.array([int(teacher_proj_Nv)], dtype=np.int32)
    if teacher_proj_Nv_targets is not None:
        payload["teacher_proj_Nv_targets"] = np.asarray(
            tuple(int(v) for v in teacher_proj_Nv_targets),
            dtype=np.int32,
        )
    return payload


def build_online_reference_cache_metadata(
    *,
    regimes: Sequence[str],
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_L: float,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    linear_T: float,
    linear_eps: float,
    linear_modes: Sequence[float],
    linear_num_samples: int,
    linear_seed: int,
    linear_poisson_sign: float,
    nonlinear_T: float,
    nonlinear_k0: float,
    nonlinear_poisson_sign: float,
    weak_eps: Sequence[float],
    strong_eps: Sequence[float],
    val_fraction: float,
    online_v_probes: int,
    online_loss_backend: str,
    Nv_targets: Optional[Sequence[int]] = None,
    rollout_horizon: int = 0,
    rollout_anchor_pool_size: int = 0,
) -> Dict[str, np.ndarray]:
    payload = {
        "dataset_format": np.array([ONLINE_REFERENCE_CACHE_FORMAT], dtype=np.str_),
        "regimes": np.asarray(tuple(regimes), dtype=np.str_),
        "teacher_backend": np.array([GRID_CUBIC_SPLINE_TEACHER_BACKEND], dtype=np.str_),
        "teacher_Nx": np.array([int(teacher_Nx)], dtype=np.int32),
        "teacher_Nv": np.array([int(teacher_Nv)], dtype=np.int32),
        "teacher_L": np.array([float(teacher_L)], dtype=np.float64),
        "teacher_vmin": np.array([float(teacher_vmin)], dtype=np.float64),
        "teacher_vmax": np.array([float(teacher_vmax)], dtype=np.float64),
        "teacher_dt": np.array([float(teacher_dt)], dtype=np.float64),
        "linear_T": np.array([float(linear_T)], dtype=np.float64),
        "linear_eps": np.array([float(linear_eps)], dtype=np.float64),
        "linear_modes": np.asarray(tuple(float(v) for v in linear_modes), dtype=np.float64),
        "linear_num_samples": np.array([int(linear_num_samples)], dtype=np.int32),
        "linear_seed": np.array([int(linear_seed)], dtype=np.int32),
        "linear_poisson_sign": np.array([float(linear_poisson_sign)], dtype=np.float64),
        "nonlinear_T": np.array([float(nonlinear_T)], dtype=np.float64),
        "nonlinear_k0": np.array([float(nonlinear_k0)], dtype=np.float64),
        "nonlinear_poisson_sign": np.array([float(nonlinear_poisson_sign)], dtype=np.float64),
        "weak_eps": np.asarray(tuple(float(v) for v in weak_eps), dtype=np.float64),
        "strong_eps": np.asarray(tuple(float(v) for v in strong_eps), dtype=np.float64),
        "val_fraction": np.array([float(val_fraction)], dtype=np.float64),
        "online_v_probes": np.array([int(online_v_probes)], dtype=np.int32),
        "online_loss_backend": np.array([str(online_loss_backend)], dtype=np.str_),
        "rollout_horizon": np.array([int(rollout_horizon)], dtype=np.int32),
        "rollout_anchor_pool_size": np.array([int(rollout_anchor_pool_size)], dtype=np.int32),
    }
    if Nv_targets is not None:
        payload["Nv_targets"] = np.asarray(tuple(int(v) for v in Nv_targets), dtype=np.int32)
    return payload


def adam_init(params: Dict[str, Array]) -> Dict[str, object]:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": jnp.array(0, dtype=jnp.int32), "m": zeros, "v": zeros}


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


def sample_initial_condition(
    rng: np.random.Generator,
    x: np.ndarray,
    modes: Sequence[float],
    eps: float,
) -> np.ndarray:
    amplitudes = rng.uniform(0.5, 1.5, size=len(modes))
    phases = rng.uniform(0.0, 2.0 * math.pi, size=len(modes))
    a0 = np.zeros_like(x)
    for amp, phase, mode in zip(amplitudes, phases, modes):
        a0 = a0 + amp * np.cos(float(mode) * x + phase)
    return (float(eps) / max(len(modes), 1)) * a0


def split_history_train_val(history: np.ndarray, val_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    history = np.asarray(history)
    if history.shape[0] <= 1:
        return history, history
    n_val = max(1, int(round(history.shape[0] * float(val_fraction))))
    n_val = min(n_val, history.shape[0] - 1)
    return history[:-n_val], history[-n_val:]


def append_pairs(
    accum: Dict[str, Dict[str, list]],
    regime: str,
    split: str,
    pairs_by_nv: Dict[int, Dict[str, np.ndarray]],
) -> None:
    for payload in pairs_by_nv.values():
        accum[regime][f"{split}_inputs_base"].append(payload["inputs_base"])
        accum[regime][f"{split}_targets"].append(payload["targets"])


def finalize_regime_arrays(accum: Dict[str, Dict[str, list]]) -> Dict[str, Dict[str, np.ndarray]]:
    dataset: Dict[str, Dict[str, np.ndarray]] = {}
    for regime, payload in accum.items():
        if not payload["train_inputs_base"]:
            continue
        dataset[regime] = {
            "train_inputs_base": np.concatenate(payload["train_inputs_base"], axis=0).astype(np.float64),
            "train_targets": np.concatenate(payload["train_targets"], axis=0).astype(np.float64),
            "val_inputs_base": np.concatenate(payload["val_inputs_base"], axis=0).astype(np.float64),
            "val_targets": np.concatenate(payload["val_targets"], axis=0).astype(np.float64),
        }
    return dataset


def maxwellian_equilibrium(v: Array) -> Array:
    return normalize_density_on_grid(gaussian_pdf(v, mean=0.0, sigma=1.0), v)


def _projected_history_projector(
    v: Array,
    projection_order: int,
    *,
    equilibrium: Array,
    vth: float = 1.0,
):
    dual_basis = hermite_dual_basis_scaled(int(projection_order), v, vth=vth)

    def projector(f_state: Array) -> Array:
        return project_distribution_snapshot_to_fourier_hermite(
            f_state,
            v,
            int(projection_order),
            vth=vth,
            equilibrium=equilibrium,
            dual_basis=dual_basis,
        )

    return projector


def _multi_projected_history_projector(
    v: Array,
    projection_orders: Sequence[int],
    *,
    equilibrium: Array,
    vth: float = 1.0,
):
    orders = tuple(int(order) for order in projection_orders)
    dual_bases = tuple(
        hermite_dual_basis_scaled(int(order), v, vth=vth)
        for order in orders
    )

    def projector(f_state: Array) -> Array:
        pieces = []
        for order, dual_basis in zip(orders, dual_bases):
            pieces.append(
                project_distribution_snapshot_to_fourier_hermite(
                    f_state,
                    v,
                    int(order),
                    vth=vth,
                    equilibrium=equilibrium,
                    dual_basis=dual_basis,
                )
            )
        return jnp.concatenate(tuple(pieces), axis=0)

    return projector


def _run_landau_teacher_projected_history(
    config: PhysicalGridVlasovPoissonConfig,
    perturbation_x: Array,
    *,
    projection_order: int,
    history_stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    v = config.v
    equilibrium = maxwellian_equilibrium(v)
    f0 = equilibrium[:, None] * (1.0 + jnp.asarray(perturbation_x, dtype=jnp.float64)[None, :])
    raw = run_semilagrangian_vlasov_poisson(
        config,
        f0,
        history_stride=history_stride,
        return_state_history=True,
        history_projector=_projected_history_projector(
            v,
            int(projection_order),
            equilibrium=equilibrium,
            vth=1.0,
        ),
    )
    return (
        np.asarray(raw["state_history"], dtype=np.complex128),
        np.asarray(raw["k_arr"], dtype=np.float64),
    )


def _run_landau_teacher_projected_histories(
    config: PhysicalGridVlasovPoissonConfig,
    perturbation_x: Array,
    *,
    projection_orders: Sequence[int],
    history_stride: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    orders = tuple(sorted(int(order) for order in projection_orders))
    if not orders:
        raise ValueError("projection_orders must be nonempty")
    histories: Dict[int, np.ndarray] = {}
    k_arr = None
    for order in orders:
        coeff_hist, order_k_arr = _run_landau_teacher_projected_history(
            config,
            perturbation_x,
            projection_order=int(order),
            history_stride=history_stride,
        )
        histories[int(order)] = coeff_hist
        if k_arr is None:
            k_arr = order_k_arr
        elif not np.array_equal(order_k_arr, k_arr):
            raise ValueError("Projected teacher histories returned inconsistent Fourier grids")
    assert k_arr is not None
    return histories, np.asarray(k_arr, dtype=np.float64)


def _run_linear_landau_higher_order_history(
    config: LinearLandauConfig,
    perturbation_x: Array,
    *,
    history_stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = run_linear_landau_cnab2_raw(
        config,
        return_state_history=True,
        perturbation_x=perturbation_x,
    )
    a_hat_hist = np.asarray(raw["a_hat_hist"], dtype=np.complex128)
    nsteps = a_hat_hist.shape[0] - 1
    stride = max(int(history_stride), 1)
    hist_steps = np.arange(0, nsteps + 1, stride, dtype=np.int32)
    if hist_steps[-1] != nsteps:
        hist_steps = np.concatenate([hist_steps, np.array([nsteps], dtype=np.int32)])
    integ = FourierHermiteIMEX(
        Nx=int(config.Nx),
        Nv=int(config.Nv),
        Lx=float(config.L),
        dt=float(config.dt),
        vth=1.0,
        dealias_23=False,
        closure=None,
    )
    return a_hat_hist[hist_steps], np.asarray(integ.k_arr, dtype=np.float64)


def _run_nonlinear_landau_higher_order_history(
    params: NonlinearLandauParams,
    *,
    history_stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = run_nonlinear_landau_rollout_raw(
        params,
        "truncation",
        return_state_history=True,
        history_stride=history_stride,
    )
    return (
        np.asarray(raw["a_hat_hist"], dtype=np.complex128),
        np.asarray(raw["k_arr"], dtype=np.float64),
    )


def _append_coeff_history_to_accum(
    accum: Dict[str, Dict[str, object]],
    regime_name: str,
    coeff_hist: np.ndarray,
    *,
    k_arr: np.ndarray,
    Nv_targets: Sequence[int],
    Nm: int,
    val_fraction: float,
    n_low: int,
    context_mode: str,
) -> None:
    train_hist, val_hist = split_history_train_val(coeff_hist, val_fraction)
    for split, hist in (("train", train_hist), ("val", val_hist)):
        append_pairs(
            accum,
            regime_name,
            split,
            extract_interface_supervised_pairs_from_coeff_history(
                hist,
                Nv_targets=Nv_targets,
                Nm=Nm,
                k_arr=k_arr,
                vth=1.0,
                include_global_indicators=True,
                n_low=int(n_low),
                context_mode=context_mode,
            ),
        )


def build_linear_landau_regime(
    *,
    teacher_backend: str,
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_L: float,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    teacher_proj_Nv: Optional[int],
    Nv_targets: Sequence[int],
    Nm: int,
    T: float,
    eps: float,
    modes: Sequence[float],
    num_samples: int,
    seed: int,
    poisson_sign: float,
    history_stride: int,
    val_fraction: float,
    n_low: int,
    context_mode: str,
    per_target_projection_orders: bool = False,
) -> Dict[str, np.ndarray]:
    teacher_backend = normalize_teacher_backend_name(teacher_backend)
    rng = np.random.default_rng(seed)
    if teacher_backend == GRID_CUBIC_SPLINE_TEACHER_BACKEND:
        config = PhysicalGridVlasovPoissonConfig(
            Nx=int(teacher_Nx),
            Nv=int(teacher_Nv),
            Lx=float(teacher_L),
            vmin=float(teacher_vmin),
            vmax=float(teacher_vmax),
            dt=float(teacher_dt),
            T=float(T),
            poisson_sign=float(poisson_sign),
            snapshot_times=(),
        )
        x = np.asarray(config.x, dtype=np.float64)
    elif teacher_backend == HIGHER_ORDER_HERMITE_TEACHER_BACKEND:
        config = LinearLandauConfig(
            Nv=int(teacher_Nv),
            Nx=int(teacher_Nx),
            L=float(teacher_L),
            dt=float(teacher_dt),
            T=float(T),
            eps=float(eps),
            modes=tuple(float(v) for v in modes),
            poisson_sign=float(poisson_sign),
        )
        integ = FourierHermiteIMEX(
            Nx=int(config.Nx),
            Nv=int(config.Nv),
            Lx=float(config.L),
            dt=float(config.dt),
            vth=1.0,
            dealias_23=False,
            closure=None,
        )
        x = np.asarray(integ.x, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported teacher_backend={teacher_backend!r}")
    accum = {
        REGIME_LINEAR: {
            "train_inputs_base": [],
            "train_targets": [],
            "val_inputs_base": [],
            "val_targets": [],
        }
    }

    for _ in range(int(num_samples)):
        perturb = sample_initial_condition(rng, x, modes, eps)
        if teacher_backend == GRID_CUBIC_SPLINE_TEACHER_BACKEND and bool(per_target_projection_orders):
            assert teacher_proj_Nv is not None
            projection_orders = tuple(sorted({int(Nv) + 1 for Nv in Nv_targets}))
            coeff_histories, k_arr = _run_landau_teacher_projected_histories(
                config,
                perturb,
                projection_orders=projection_orders,
                history_stride=history_stride,
            )
            for Nv in Nv_targets:
                coeff_hist = coeff_histories[int(Nv) + 1]
                _append_coeff_history_to_accum(
                    accum,
                    REGIME_LINEAR,
                    coeff_hist,
                    k_arr=k_arr,
                    Nv_targets=(int(Nv),),
                    Nm=Nm,
                    val_fraction=val_fraction,
                    n_low=n_low,
                    context_mode=context_mode,
                )
            continue

        if teacher_backend == GRID_CUBIC_SPLINE_TEACHER_BACKEND:
            assert teacher_proj_Nv is not None
            coeff_hist, k_arr = _run_landau_teacher_projected_history(
                config,
                perturb,
                projection_order=int(teacher_proj_Nv),
                history_stride=history_stride,
            )
        else:
            coeff_hist, k_arr = _run_linear_landau_higher_order_history(
                config,
                perturb,
                history_stride=history_stride,
            )

        _append_coeff_history_to_accum(
            accum,
            REGIME_LINEAR,
            coeff_hist,
            k_arr=k_arr,
            Nv_targets=Nv_targets,
            Nm=Nm,
            val_fraction=val_fraction,
            n_low=n_low,
            context_mode=context_mode,
        )
    return finalize_regime_arrays(accum)[REGIME_LINEAR]


def build_nonlinear_landau_regime(
    regime_name: str,
    eps_values: Sequence[float],
    *,
    teacher_backend: str,
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_L: float,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    teacher_proj_Nv: Optional[int],
    Nv_targets: Sequence[int],
    Nm: int,
    T: float,
    k0: float,
    poisson_sign: float,
    history_stride: int,
    val_fraction: float,
    n_low: int,
    context_mode: str,
    per_target_projection_orders: bool = False,
) -> Dict[str, np.ndarray]:
    teacher_backend = normalize_teacher_backend_name(teacher_backend)
    if teacher_backend == GRID_CUBIC_SPLINE_TEACHER_BACKEND:
        config = PhysicalGridVlasovPoissonConfig(
            Nx=int(teacher_Nx),
            Nv=int(teacher_Nv),
            Lx=float(teacher_L),
            vmin=float(teacher_vmin),
            vmax=float(teacher_vmax),
            dt=float(teacher_dt),
            T=float(T),
            poisson_sign=float(poisson_sign),
            snapshot_times=(),
        )
        perturb_template = np.cos(float(k0) * np.asarray(config.x, dtype=np.float64))
    elif teacher_backend == HIGHER_ORDER_HERMITE_TEACHER_BACKEND:
        config = NonlinearLandauParams(
            Nx=int(teacher_Nx),
            Nv=int(teacher_Nv),
            L=float(teacher_L),
            dt=float(teacher_dt),
            T=float(T),
            k0=float(k0),
            dealias_23=False,
            poisson_sign=float(poisson_sign),
            snapshot_times=(),
        )
    else:
        raise ValueError(f"Unsupported teacher_backend={teacher_backend!r}")
    accum = {
        regime_name: {
            "train_inputs_base": [],
            "train_targets": [],
            "val_inputs_base": [],
            "val_targets": [],
        }
    }

    for eps in eps_values:
        if teacher_backend == GRID_CUBIC_SPLINE_TEACHER_BACKEND and bool(per_target_projection_orders):
            assert teacher_proj_Nv is not None
            projection_orders = tuple(sorted({int(Nv) + 1 for Nv in Nv_targets}))
            coeff_histories, k_arr = _run_landau_teacher_projected_histories(
                config,
                float(eps) * perturb_template,
                projection_orders=projection_orders,
                history_stride=history_stride,
            )
            for Nv in Nv_targets:
                coeff_hist = coeff_histories[int(Nv) + 1]
                _append_coeff_history_to_accum(
                    accum,
                    regime_name,
                    coeff_hist,
                    k_arr=k_arr,
                    Nv_targets=(int(Nv),),
                    Nm=Nm,
                    val_fraction=val_fraction,
                    n_low=n_low,
                    context_mode=context_mode,
                )
            continue

        if teacher_backend == GRID_CUBIC_SPLINE_TEACHER_BACKEND:
            assert teacher_proj_Nv is not None
            coeff_hist, k_arr = _run_landau_teacher_projected_history(
                config,
                float(eps) * perturb_template,
                projection_order=int(teacher_proj_Nv),
                history_stride=history_stride,
            )
        else:
            coeff_hist, k_arr = _run_nonlinear_landau_higher_order_history(
                NonlinearLandauParams(
                    Nx=int(config.Nx),
                    Nv=int(config.Nv),
                    L=float(config.L),
                    dt=float(config.dt),
                    T=float(config.T),
                    eps=float(eps),
                    k0=float(config.k0),
                    vth=float(config.vth),
                    dealias_23=bool(config.dealias_23),
                    poisson_sign=float(config.poisson_sign),
                    snapshot_times=tuple(config.snapshot_times),
                    v_range=tuple(config.v_range),
                    Nv_plot=int(config.Nv_plot),
                    vmin=float(config.vmin),
                    vmax=float(config.vmax),
                ),
                history_stride=history_stride,
            )

        _append_coeff_history_to_accum(
            accum,
            regime_name,
            coeff_hist,
            k_arr=k_arr,
            Nv_targets=Nv_targets,
            Nm=Nm,
            val_fraction=val_fraction,
            n_low=n_low,
            context_mode=context_mode,
        )
    return finalize_regime_arrays(accum)[regime_name]


def _cache_value_mismatch(actual: np.ndarray, expected: np.ndarray) -> bool:
    if actual.shape != expected.shape:
        return True
    if actual.dtype.kind in {"U", "S", "O"} or expected.dtype.kind in {"U", "S", "O"}:
        return not np.array_equal(np.asarray(actual, dtype=np.str_), np.asarray(expected, dtype=np.str_))
    return not np.array_equal(actual, expected)


def load_dataset_cache(
    path: Path,
    *,
    expected_metadata: Dict[str, np.ndarray],
    allow_nv_superset: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    with np.load(path) as data:
        cached_projection_mode = (
            str(np.asarray(data["projection_mode"], dtype=np.str_).reshape(-1)[0])
            if "projection_mode" in data.files and data["projection_mode"].size
            else "shared_max"
        )
        for key, expected in expected_metadata.items():
            if key not in data.files:
                raise ValueError(f"Dataset cache {path} is missing metadata field '{key}'.")
            actual = np.asarray(data[key])
            if key == "Nv_targets" and bool(allow_nv_superset):
                actual_nv = tuple(int(v) for v in np.asarray(actual, dtype=np.int32).reshape(-1))
                expected_nv = tuple(int(v) for v in np.asarray(expected, dtype=np.int32).reshape(-1))
                if not set(expected_nv).issubset(set(actual_nv)):
                    raise ValueError(
                        f"Dataset cache {path} Nv_targets={actual_nv} do not cover requested Nv-targets={expected_nv}."
                    )
                continue
            if key == "teacher_proj_Nv" and bool(allow_nv_superset) and cached_projection_mode == "per_target":
                if "teacher_proj_Nv_targets" not in data.files:
                    raise ValueError(
                        f"Dataset cache {path} uses per-target projection mode but is missing 'teacher_proj_Nv_targets'."
                    )
                actual_proj_targets = tuple(
                    int(v) for v in np.asarray(data["teacher_proj_Nv_targets"], dtype=np.int32).reshape(-1)
                )
                expected_nv = tuple(int(v) for v in np.asarray(expected_metadata["Nv_targets"], dtype=np.int32).reshape(-1))
                required_proj = tuple(int(v) + 1 for v in expected_nv)
                if not set(required_proj).issubset(set(actual_proj_targets)):
                    raise ValueError(
                        f"Dataset cache {path} per-target projection orders={actual_proj_targets} "
                        f"do not cover requested orders={required_proj}."
                    )
                continue
            if key == "teacher_proj_Nv_targets" and bool(allow_nv_superset) and cached_projection_mode == "per_target":
                actual_proj_targets = tuple(int(v) for v in np.asarray(actual, dtype=np.int32).reshape(-1))
                expected_proj_targets = tuple(int(v) for v in np.asarray(expected, dtype=np.int32).reshape(-1))
                if not set(expected_proj_targets).issubset(set(actual_proj_targets)):
                    raise ValueError(
                        f"Dataset cache {path} per-target projection orders={actual_proj_targets} "
                        f"do not cover requested projection orders={expected_proj_targets}."
                    )
                continue
            if _cache_value_mismatch(actual, np.asarray(expected)):
                raise ValueError(
                    f"Dataset cache {path} metadata mismatch for '{key}'. "
                    "Rebuilding with the current teacher configuration is required."
                )
        regimes = tuple(str(v) for v in np.asarray(data["regimes"], dtype=np.str_).tolist())
        dataset: Dict[str, Dict[str, np.ndarray]] = {}
        for regime in regimes:
            dataset[regime] = {
                "train_inputs_base": np.asarray(data[f"{regime}_train_inputs_base"], dtype=np.float64),
                "train_targets": np.asarray(data[f"{regime}_train_targets"], dtype=np.float64),
                "val_inputs_base": np.asarray(data[f"{regime}_val_inputs_base"], dtype=np.float64),
                "val_targets": np.asarray(data[f"{regime}_val_targets"], dtype=np.float64),
            }
        return dataset


def load_online_reference_cache(
    path: Path,
    *,
    expected_metadata: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], np.ndarray]:
    with np.load(path) as data:
        for key, expected in expected_metadata.items():
            if key not in data.files:
                raise ValueError(f"Online reference cache {path} is missing metadata field '{key}'.")
            actual = np.asarray(data[key])
            if _cache_value_mismatch(actual, np.asarray(expected)):
                raise ValueError(
                    f"Online reference cache {path} metadata mismatch for '{key}'. "
                    "Rebuilding with the current teacher configuration is required."
                )
        if "v_probe" not in data.files:
            raise ValueError(f"Online reference cache {path} is missing 'v_probe'.")
        regimes = tuple(str(v) for v in np.asarray(data["regimes"], dtype=np.str_).tolist())
        dataset: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for regime in regimes:
            dataset[regime] = {}
            for split in ("train", "val"):
                prefix = f"{regime}_{split}_"
                payload = {
                    name[len(prefix):]: np.asarray(data[name])
                    for name in data.files
                    if name.startswith(prefix)
                }
                dataset[regime][split] = payload
        return dataset, np.asarray(data["v_probe"], dtype=np.float64)


def save_dataset_cache(
    path: Path,
    dataset: Dict[str, Dict[str, np.ndarray]],
    *,
    metadata: Dict[str, np.ndarray],
) -> None:
    payload: Dict[str, np.ndarray] = dict(metadata)
    for regime, arrays in dataset.items():
        payload[f"{regime}_train_inputs_base"] = np.asarray(arrays["train_inputs_base"], dtype=np.float64)
        payload[f"{regime}_train_targets"] = np.asarray(arrays["train_targets"], dtype=np.float64)
        payload[f"{regime}_val_inputs_base"] = np.asarray(arrays["val_inputs_base"], dtype=np.float64)
        payload[f"{regime}_val_targets"] = np.asarray(arrays["val_targets"], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def save_online_reference_cache(
    path: Path,
    dataset: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    *,
    v_probe: Array,
    metadata: Dict[str, np.ndarray],
) -> None:
    payload: Dict[str, np.ndarray] = dict(metadata)
    payload["v_probe"] = np.asarray(v_probe, dtype=np.float64)
    for regime, splits in dataset.items():
        for split in ("train", "val"):
            for key, value in splits.get(split, {}).items():
                payload[f"{regime}_{split}_{key}"] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def select_nv_targets_from_dataset(
    dataset: Dict[str, Dict[str, np.ndarray]],
    *,
    Nv_targets: Sequence[int],
    Nm: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    nv_targets = np.asarray(tuple(int(v) for v in Nv_targets), dtype=np.int64)
    nv_col = 2 * int(Nm) + 1
    subset: Dict[str, Dict[str, np.ndarray]] = {}
    for regime, arrays in dataset.items():
        train_inputs = np.asarray(arrays["train_inputs_base"], dtype=np.float64)
        val_inputs = np.asarray(arrays["val_inputs_base"], dtype=np.float64)
        train_nv = np.rint(train_inputs[:, nv_col]).astype(np.int64)
        val_nv = np.rint(val_inputs[:, nv_col]).astype(np.int64)
        train_mask = np.isin(train_nv, nv_targets)
        val_mask = np.isin(val_nv, nv_targets)
        if not np.any(train_mask):
            raise ValueError(
                f"Requested Nv-targets={tuple(int(v) for v in nv_targets)} do not exist in cached train split for regime '{regime}'."
            )
        subset[regime] = {
            "train_inputs_base": train_inputs[train_mask].astype(np.float64),
            "train_targets": np.asarray(arrays["train_targets"], dtype=np.float64)[train_mask].astype(np.float64),
            "val_inputs_base": val_inputs[val_mask].astype(np.float64),
            "val_targets": np.asarray(arrays["val_targets"], dtype=np.float64)[val_mask].astype(np.float64),
        }
    return subset


def build_mixed_landau_dataset(
    *,
    dataset_cache: Optional[Path],
    regimes: Sequence[str],
    teacher_backend: str,
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_L: float,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    teacher_proj_Nv: Optional[int],
    linear_T: float,
    linear_eps: float,
    linear_modes: Sequence[float],
    linear_num_samples: int,
    linear_seed: int,
    linear_poisson_sign: float,
    linear_history_stride: int,
    nonlinear_T: float,
    nonlinear_k0: float,
    nonlinear_poisson_sign: float,
    nonlinear_history_stride: int,
    weak_eps: Sequence[float],
    strong_eps: Sequence[float],
    Nv_targets: Sequence[int],
    Nm: int,
    val_fraction: float,
    n_low: int,
    context_mode: str = "none",
    allow_cached_nv_superset: bool = False,
    per_target_projection_orders: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    teacher_backend = normalize_teacher_backend_name(teacher_backend)
    teacher_proj_Nv_targets = (
        tuple(int(v) + 1 for v in Nv_targets)
        if teacher_backend == GRID_CUBIC_SPLINE_TEACHER_BACKEND and bool(per_target_projection_orders)
        else None
    )
    cache_metadata = build_dataset_cache_metadata(
        regimes=regimes,
        teacher_backend=teacher_backend,
        teacher_Nx=teacher_Nx,
        teacher_Nv=teacher_Nv,
        teacher_L=teacher_L,
        teacher_vmin=teacher_vmin,
        teacher_vmax=teacher_vmax,
        teacher_dt=teacher_dt,
        teacher_proj_Nv=teacher_proj_Nv,
        linear_T=linear_T,
        linear_eps=linear_eps,
        linear_modes=linear_modes,
        linear_num_samples=linear_num_samples,
        linear_seed=linear_seed,
        linear_poisson_sign=linear_poisson_sign,
        linear_history_stride=linear_history_stride,
        nonlinear_T=nonlinear_T,
        nonlinear_k0=nonlinear_k0,
        nonlinear_poisson_sign=nonlinear_poisson_sign,
        nonlinear_history_stride=nonlinear_history_stride,
        weak_eps=weak_eps,
        strong_eps=strong_eps,
        Nv_targets=Nv_targets,
        Nm=Nm,
        val_fraction=val_fraction,
        n_low=n_low,
        context_mode=context_mode,
        projection_mode=(
            "per_target"
            if teacher_backend == GRID_CUBIC_SPLINE_TEACHER_BACKEND and bool(per_target_projection_orders)
            else ("shared_max" if teacher_backend == GRID_CUBIC_SPLINE_TEACHER_BACKEND else "none")
        ),
        teacher_proj_Nv_targets=teacher_proj_Nv_targets,
    )
    if dataset_cache is not None and dataset_cache.exists():
        try:
            cached = load_dataset_cache(
                dataset_cache,
                expected_metadata=cache_metadata,
                allow_nv_superset=allow_cached_nv_superset,
            )
            selected = {regime: cached[regime] for regime in regimes}
            if bool(allow_cached_nv_superset):
                selected = select_nv_targets_from_dataset(selected, Nv_targets=Nv_targets, Nm=Nm)
            return selected
        except ValueError as exc:
            print(f"[data] ignoring incompatible dataset cache {dataset_cache}: {exc}")

    dataset: Dict[str, Dict[str, np.ndarray]] = {}
    active = tuple(regimes)
    if REGIME_LINEAR in active:
        dataset[REGIME_LINEAR] = build_linear_landau_regime(
            teacher_backend=teacher_backend,
            teacher_Nx=teacher_Nx,
            teacher_Nv=teacher_Nv,
            teacher_L=teacher_L,
            teacher_vmin=teacher_vmin,
            teacher_vmax=teacher_vmax,
            teacher_dt=teacher_dt,
            teacher_proj_Nv=teacher_proj_Nv,
            Nv_targets=Nv_targets,
            Nm=Nm,
            T=linear_T,
            eps=linear_eps,
            modes=linear_modes,
            num_samples=linear_num_samples,
            seed=linear_seed,
            poisson_sign=linear_poisson_sign,
            history_stride=linear_history_stride,
            val_fraction=val_fraction,
            n_low=int(n_low),
            context_mode=context_mode,
            per_target_projection_orders=bool(per_target_projection_orders),
        )
    if REGIME_WEAK in active:
        dataset[REGIME_WEAK] = build_nonlinear_landau_regime(
            REGIME_WEAK,
            weak_eps,
            teacher_backend=teacher_backend,
            teacher_Nx=teacher_Nx,
            teacher_Nv=teacher_Nv,
            teacher_L=teacher_L,
            teacher_vmin=teacher_vmin,
            teacher_vmax=teacher_vmax,
            teacher_dt=teacher_dt,
            teacher_proj_Nv=teacher_proj_Nv,
            Nv_targets=Nv_targets,
            Nm=Nm,
            T=nonlinear_T,
            k0=nonlinear_k0,
            poisson_sign=nonlinear_poisson_sign,
            history_stride=nonlinear_history_stride,
            val_fraction=val_fraction,
            n_low=int(n_low),
            context_mode=context_mode,
            per_target_projection_orders=bool(per_target_projection_orders),
        )
    if REGIME_STRONG in active:
        dataset[REGIME_STRONG] = build_nonlinear_landau_regime(
            REGIME_STRONG,
            strong_eps,
            teacher_backend=teacher_backend,
            teacher_Nx=teacher_Nx,
            teacher_Nv=teacher_Nv,
            teacher_L=teacher_L,
            teacher_vmin=teacher_vmin,
            teacher_vmax=teacher_vmax,
            teacher_dt=teacher_dt,
            teacher_proj_Nv=teacher_proj_Nv,
            Nv_targets=Nv_targets,
            Nm=Nm,
            T=nonlinear_T,
            k0=nonlinear_k0,
            poisson_sign=nonlinear_poisson_sign,
            history_stride=nonlinear_history_stride,
            val_fraction=val_fraction,
            n_low=int(n_low),
            context_mode=context_mode,
            per_target_projection_orders=bool(per_target_projection_orders),
        )

    if dataset_cache is not None:
        save_dataset_cache(dataset_cache, dataset, metadata=cache_metadata)
    return dataset


def build_model_inputs(
    inputs_base: np.ndarray,
    *,
    Nm: int,
    k_scale: float,
    nv_scale: float,
    context_mode: str,
    include_global_indicators: bool = True,
) -> np.ndarray:
    inputs = np.asarray(inputs_base, dtype=np.float64).copy()
    k_col = 2 * int(Nm)
    nv_col = k_col + 1
    base_dim = 2 * int(Nm) + (4 if bool(include_global_indicators) else 2)
    if context_mode == "none":
        inputs[:, k_col] = inputs[:, k_col] / float(k_scale)
        inputs[:, nv_col] = inputs[:, nv_col] / float(nv_scale)
        return inputs
    if context_mode == "lag1_delta":
        current = inputs[:, :base_dim]
        previous = inputs[:, base_dim : 2 * base_dim]
        current[:, k_col] = current[:, k_col] / float(k_scale)
        current[:, nv_col] = current[:, nv_col] / float(nv_scale)
        previous[:, k_col] = previous[:, k_col] / float(k_scale)
        previous[:, nv_col] = previous[:, nv_col] / float(nv_scale)
        delta = current - previous
        return np.concatenate([current, previous, delta], axis=1)
    raise ValueError(f"Unsupported context_mode={context_mode!r}")


def safe_feature_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=np.float64)
    return np.where(std > 1e-12, std, 1.0)


def choose_k_scale(dataset: Dict[str, Dict[str, np.ndarray]], *, Nm: int) -> float:
    k_col = 2 * int(Nm)
    return max(float(np.max(arrays["train_inputs_base"][:, k_col])) for arrays in dataset.values())


def choose_nv_scale(dataset: Dict[str, Dict[str, np.ndarray]], *, Nm: int) -> float:
    nv_col = 2 * int(Nm) + 1
    return max(float(np.max(arrays["train_inputs_base"][:, nv_col])) for arrays in dataset.values())


def prepare_training_dataset(
    dataset_base: Dict[str, Dict[str, np.ndarray]],
    *,
    Nm: int,
    k_scale: float,
    nv_scale: float,
    context_mode: str,
) -> Tuple[Dict[str, Dict[str, Array]], Dict[str, np.ndarray]]:
    scaled_dataset: Dict[str, Dict[str, np.ndarray]] = {}
    train_inputs_all = []
    train_targets_all = []
    for regime, arrays in dataset_base.items():
        train_inputs = build_model_inputs(
            arrays["train_inputs_base"],
            Nm=Nm,
            k_scale=k_scale,
            nv_scale=nv_scale,
            context_mode=context_mode,
        )
        val_inputs = build_model_inputs(
            arrays["val_inputs_base"],
            Nm=Nm,
            k_scale=k_scale,
            nv_scale=nv_scale,
            context_mode=context_mode,
        )
        scaled_dataset[regime] = {
            "train_inputs": train_inputs,
            "train_targets": np.asarray(arrays["train_targets"], dtype=np.float64),
            "val_inputs": val_inputs,
            "val_targets": np.asarray(arrays["val_targets"], dtype=np.float64),
        }
        train_inputs_all.append(train_inputs)
        train_targets_all.append(np.asarray(arrays["train_targets"], dtype=np.float64))

    input_mean = np.mean(np.concatenate(train_inputs_all, axis=0), axis=0)
    input_std = safe_feature_std(np.std(np.concatenate(train_inputs_all, axis=0), axis=0))
    target_mean = np.mean(np.concatenate(train_targets_all, axis=0), axis=0)
    target_std = safe_feature_std(np.std(np.concatenate(train_targets_all, axis=0), axis=0))

    prepared: Dict[str, Dict[str, Array]] = {}
    target_std_safe = target_std[None, :]
    target_mean_row = target_mean[None, :]
    for regime, arrays in scaled_dataset.items():
        prepared[regime] = {
            "train_inputs": jnp.asarray(arrays["train_inputs"], dtype=jnp.float64),
            "train_targets": jnp.asarray(arrays["train_targets"], dtype=jnp.float64),
            "train_targets_std": jnp.asarray((arrays["train_targets"] - target_mean_row) / target_std_safe, dtype=jnp.float64),
            "val_inputs": jnp.asarray(arrays["val_inputs"], dtype=jnp.float64),
            "val_targets": jnp.asarray(arrays["val_targets"], dtype=jnp.float64),
            "val_targets_std": jnp.asarray((arrays["val_targets"] - target_mean_row) / target_std_safe, dtype=jnp.float64),
        }

    stats = {
        "input_mean": np.asarray(input_mean, dtype=np.float64),
        "input_std": np.asarray(input_std, dtype=np.float64),
        "target_mean": np.asarray(target_mean, dtype=np.float64),
        "target_std": np.asarray(target_std, dtype=np.float64),
    }
    return prepared, stats


def summarize_dataset(prepared: Dict[str, Dict[str, Array]]) -> Dict[str, int]:
    return {regime: int(arrays["train_inputs"].shape[0]) for regime, arrays in prepared.items()}


def build_identity_training_stats(
    *,
    Nm: int,
    context_mode: str,
    include_global_indicators: bool = True,
) -> Dict[str, np.ndarray]:
    base_dim = 2 * int(Nm) + (4 if bool(include_global_indicators) else 2)
    input_dim = base_dim if str(context_mode) == "none" else 3 * base_dim
    return {
        "input_mean": np.zeros((input_dim,), dtype=np.float64),
        "input_std": np.ones((input_dim,), dtype=np.float64),
        "target_mean": np.zeros((2,), dtype=np.float64),
        "target_std": np.ones((2,), dtype=np.float64),
    }


def _accumulate_feature_moments(
    total: Optional[np.ndarray],
    total_sq: Optional[np.ndarray],
    count: int,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D feature block, got shape {values.shape}")
    if values.shape[0] == 0:
        if total is None or total_sq is None:
            return (
                np.zeros((values.shape[1],), dtype=np.float64),
                np.zeros((values.shape[1],), dtype=np.float64),
                int(count),
            )
        return total, total_sq, int(count)
    block_sum = np.sum(values, axis=0, dtype=np.float64)
    block_sum_sq = np.sum(values * values, axis=0, dtype=np.float64)
    if total is None or total_sq is None:
        total = np.zeros_like(block_sum, dtype=np.float64)
        total_sq = np.zeros_like(block_sum_sq, dtype=np.float64)
    return total + block_sum, total_sq + block_sum_sq, int(count) + int(values.shape[0])


def _online_extended_coeff_history_for_q_stats(
    a_hat_hist: np.ndarray,
    q_hat_hist: Optional[np.ndarray],
    *,
    target_nv: int,
    k_arr: np.ndarray,
) -> np.ndarray:
    """Append the omitted Hermite coefficient reconstructed from q, or zero if q is unavailable."""
    target_nv = int(target_nv)
    a_hat_hist = np.asarray(a_hat_hist, dtype=np.complex128)
    q_hat_arr = None if q_hat_hist is None else np.asarray(q_hat_hist, dtype=np.complex128)
    k_arr = np.asarray(k_arr, dtype=np.float64)
    if a_hat_hist.ndim != 3:
        raise ValueError(f"a_hat_hist must have shape (T, Nv, Nk), got {a_hat_hist.shape}")
    if q_hat_arr is not None and q_hat_arr.shape != (a_hat_hist.shape[0], a_hat_hist.shape[2]):
        raise ValueError(
            f"q_hat_hist must have shape {(a_hat_hist.shape[0], a_hat_hist.shape[2])}, "
            f"got {q_hat_arr.shape}"
        )
    if int(a_hat_hist.shape[1]) != target_nv:
        raise ValueError(
            f"Expected retained history for Nv={target_nv}, got Nv={a_hat_hist.shape[1]}"
        )
    extended = np.zeros(
        (a_hat_hist.shape[0], target_nv + 1, a_hat_hist.shape[2]),
        dtype=np.complex128,
    )
    extended[:, :target_nv, :] = a_hat_hist
    if q_hat_arr is not None and a_hat_hist.shape[2] > 1:
        denom = -1j * k_arr[1:] * math.sqrt(float(target_nv))
        extended[:, target_nv, 1:] = q_hat_arr[:, 1:] / denom[None, :]
    return extended


def build_online_q_training_stats_from_reference(
    online_dataset: Dict[str, Dict[str, Dict[str, Array]]],
    *,
    active_regimes: Sequence[str],
    Nv_targets: Sequence[int],
    Nm: int,
    k_arr: np.ndarray,
    k_scale: float,
    nv_scale: float,
    n_low: int,
    context_mode: str,
    require_q_targets: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute q-feature normalization for online projected-coefficient training.

    The pure online rollout losses do not always optimize a direct q loss, but
    the learned object inserted in the PDE is still q_k.  When q targets are
    available, this builds the same feature/target statistics used by the
    offline q-loss branch.  Otherwise it still standardizes the closure inputs
    from retained reference histories and leaves the output scale at the zero
    baseline.
    """
    input_sum: Optional[np.ndarray] = None
    input_sum_sq: Optional[np.ndarray] = None
    target_sum: Optional[np.ndarray] = None
    target_sum_sq: Optional[np.ndarray] = None
    input_count = 0
    target_count = 0

    for regime in active_regimes:
        group = online_dataset.get(regime, {}).get("train", {})
        if not group:
            continue
        for target_nv in Nv_targets:
            coeff_key = online_reference_coeff_key(int(target_nv))
            anchor_coeff_key = online_reference_anchor_coeff_key(int(target_nv))
            anchor_index_key = online_reference_anchor_index_key(int(target_nv))
            q_key = online_reference_q_key(int(target_nv))
            use_anchor_stencils = coeff_key not in group and anchor_coeff_key in group
            if coeff_key not in group and not use_anchor_stencils:
                continue
            if bool(require_q_targets) and q_key not in group:
                continue
            coeff_cases = np.asarray(
                group[anchor_coeff_key] if use_anchor_stencils else group[coeff_key],
                dtype=np.complex128,
            )
            q_cases = (
                None
                if q_key not in group
                else np.asarray(group[q_key], dtype=np.complex128)
            )
            anchor_index_cases = (
                None
                if not use_anchor_stencils
                else np.asarray(group[anchor_index_key], dtype=np.int32)
            )
            if q_cases is not None and coeff_cases.shape[0] != q_cases.shape[0]:
                raise ValueError(
                    f"{regime} Nv={target_nv} coeff/q case-count mismatch: "
                    f"{coeff_cases.shape[0]} vs {q_cases.shape[0]}"
                )
            for case_idx in range(int(coeff_cases.shape[0])):
                if use_anchor_stencils:
                    if coeff_cases[case_idx].ndim != 4 or coeff_cases[case_idx].shape[1] != 3:
                        raise ValueError(
                            f"{anchor_coeff_key} must have shape (cases, anchors, 3, Nv, Nk), "
                            f"got {coeff_cases.shape}"
                        )
                    assert anchor_index_cases is not None
                    anchor_indices = np.asarray(anchor_index_cases[case_idx], dtype=np.int32)
                    coeff_hist_for_stats = coeff_cases[case_idx][:, 0, :, :]
                    q_hist_for_stats = (
                        None
                        if q_cases is None
                        else np.asarray(q_cases[case_idx][anchor_indices], dtype=np.complex128)
                    )
                else:
                    coeff_hist_for_stats = coeff_cases[case_idx]
                    q_hist_for_stats = None if q_cases is None else q_cases[case_idx]
                extended_hist = _online_extended_coeff_history_for_q_stats(
                    coeff_hist_for_stats,
                    q_hist_for_stats,
                    target_nv=int(target_nv),
                    k_arr=k_arr,
                )
                pairs = extract_interface_supervised_pairs_from_coeff_history(
                    extended_hist,
                    Nv_targets=(int(target_nv),),
                    Nm=int(Nm),
                    k_arr=np.asarray(k_arr, dtype=np.float64),
                    vth=1.0,
                    include_global_indicators=True,
                    n_low=int(n_low),
                    context_mode=str(context_mode),
                )[int(target_nv)]
                inputs = build_model_inputs(
                    pairs["inputs_base"],
                    Nm=int(Nm),
                    k_scale=float(k_scale),
                    nv_scale=float(nv_scale),
                    context_mode=str(context_mode),
                    include_global_indicators=True,
                )
                targets = np.asarray(pairs["targets"], dtype=np.float64)
                input_sum, input_sum_sq, input_count = _accumulate_feature_moments(
                    input_sum,
                    input_sum_sq,
                    input_count,
                    inputs,
                )
                if use_anchor_stencils and q_cases is not None:
                    q_full = np.asarray(q_cases[case_idx], dtype=np.complex128)
                    targets = np.stack(
                        [np.real(q_full[:, 1:]), np.imag(q_full[:, 1:])],
                        axis=-1,
                    ).reshape(-1, 2)
                target_sum, target_sum_sq, target_count = _accumulate_feature_moments(
                    target_sum,
                    target_sum_sq,
                    target_count,
                    targets,
                )

    if input_sum is None or input_sum_sq is None or target_sum is None or target_sum_sq is None:
        raise ValueError("No online q reference samples were available for normalization")
    if input_count <= 0 or target_count <= 0:
        raise ValueError("Online q reference normalization received zero samples")

    input_mean = input_sum / float(input_count)
    input_var = np.maximum(input_sum_sq / float(input_count) - input_mean * input_mean, 0.0)
    target_mean = target_sum / float(target_count)
    target_var = np.maximum(target_sum_sq / float(target_count) - target_mean * target_mean, 0.0)
    return {
        "input_mean": np.asarray(input_mean, dtype=np.float64),
        "input_std": safe_feature_std(np.sqrt(input_var)),
        "target_mean": np.asarray(target_mean, dtype=np.float64),
        "target_std": safe_feature_std(np.sqrt(target_var)),
    }


def init_online_rollout_params(
    key: Array,
    *,
    input_dim: int,
    hidden_width: int,
    res_blocks: int,
    target_mean: Optional[Array] = None,
    target_std: Optional[Array] = None,
) -> Dict[str, Array]:
    """Initialize online training near the truncation baseline.

    Long solver-in-the-loop rollouts are numerically fragile if the closure
    starts from a random nonzero boundary flux. Keep the hidden stack random so
    gradients can flow immediately, but zero the output heads so the initial
    rollout matches the stable zero-closure baseline.
    """
    params = init_interface_closure_params(
        key,
        input_dim=int(input_dim),
        hidden_width=int(hidden_width),
        res_blocks=int(res_blocks),
    )
    for name in ("W_lin", "b_lin", "W_out", "b_out"):
        params[name] = jnp.zeros_like(params[name])
    if target_mean is not None and target_std is not None:
        mean = jnp.asarray(target_mean, dtype=jnp.float64)
        std = jnp.maximum(jnp.asarray(target_std, dtype=jnp.float64), 1e-12)
        params["b_lin"] = (-(mean / std)).astype(jnp.float64)
    return params


def jax_hermite_basis_phi_scaled(N: int, v: Array, vth: float = 1.0) -> Array:
    if int(N) < 0:
        raise ValueError("N must be nonnegative")
    if float(vth) <= 0.0:
        raise ValueError("vth must be positive")
    v = jnp.asarray(v, dtype=jnp.float64)
    if int(N) == 0:
        return jnp.zeros((0, v.size), dtype=jnp.float64)

    xi = v / float(vth)
    w = jnp.exp(-0.5 * xi ** 2) / (math.sqrt(2.0 * math.pi) * float(vth))
    h = jnp.zeros((int(N), v.size), dtype=jnp.float64).at[0].set(1.0)
    if int(N) > 1:
        h = h.at[1].set(xi)

    def body(i: int, arr: Array) -> Array:
        i_f = jnp.asarray(i, dtype=jnp.float64)
        next_row = (xi / jnp.sqrt(i_f + 1.0)) * arr[i] - jnp.sqrt(i_f / (i_f + 1.0)) * arr[i - 1]
        return arr.at[i + 1].set(next_row)

    if int(N) > 2:
        h = jax.lax.fori_loop(1, int(N) - 1, body, h)
    return (w[None, :] * h).astype(jnp.float64)


def reconstruct_delta_f_from_a_hat_history(
    a_hat_hist: Array,
    *,
    Nx: int,
    v_probe: Array,
    vth: float = 1.0,
) -> Array:
    a_hat_hist = jnp.asarray(a_hat_hist, dtype=jnp.complex128)
    a_phys_hist = jax.vmap(lambda a_hat: irfft_x(a_hat, int(Nx)))(a_hat_hist)
    phi = jax_hermite_basis_phi_scaled(int(a_phys_hist.shape[1]), v_probe, vth=vth)
    return jnp.einsum("tnx,nv->tvx", a_phys_hist, phi).astype(jnp.float64)


def build_projected_xv_metric_matrix(
    *,
    Nv: int,
    v_grid: Array,
    vth: float = 1.0,
    metric: str = PROJECTED_XV_METRIC_PHYSICAL_L2,
) -> Optional[Array]:
    """Build the Hermite-space metric used by the projected-xv rollout loss.

    ``physical_l2`` keeps the legacy decoded-space quadrature. ``gram_riesz`` uses
    the finite-grid projection/reconstruction cross-Gram matrix to build the
    Riesz metric of the biorthogonalized reconstruction basis. This keeps the
    rollout loss in physical space while making it consistent with the discrete
    Hermite projection that produced the reference coefficients.
    """
    metric_name = str(metric)
    if metric_name == PROJECTED_XV_METRIC_PHYSICAL_L2:
        return None
    if metric_name != PROJECTED_XV_METRIC_GRAM_RIESZ:
        raise ValueError(
            f"projected_xv_metric must be one of {ALL_PROJECTED_XV_METRICS!r}, got {metric!r}"
        )
    v_np = np.asarray(v_grid, dtype=np.float64)
    if v_np.ndim != 1 or v_np.size < 2:
        raise ValueError("projected-xv gram_riesz metric requires a one-dimensional v grid")
    phi = np.asarray(
        jax_hermite_basis_phi_scaled(int(Nv), jnp.asarray(v_np, dtype=jnp.float64), vth=float(vth)),
        dtype=np.float64,
    )
    dual = np.asarray(
        hermite_dual_basis_scaled(int(Nv), jnp.asarray(v_np, dtype=jnp.float64), vth=float(vth)),
        dtype=np.float64,
    )
    mass = np.trapezoid(phi[:, None, :] * phi[None, :, :], x=v_np, axis=2)
    mass = 0.5 * (mass + mass.T)
    cross_gram = np.trapezoid(dual[:, None, :] * phi[None, :, :], x=v_np, axis=2)
    if not np.all(np.isfinite(cross_gram)):
        raise ValueError("projected-xv Hermite cross-Gram matrix is nonfinite")
    # Numerical pseudo-inverse of the finite-grid projection/reconstruction map.
    # This is a rank/conditioning cutoff, not a hand-selected Hermite-mode taper.
    biorthogonal_map = np.linalg.pinv(cross_gram, rcond=1e-6)
    metric_matrix = biorthogonal_map.T @ mass @ biorthogonal_map
    metric_matrix = 0.5 * (metric_matrix + metric_matrix.T)
    return jnp.asarray(metric_matrix, dtype=jnp.float64)


def coefficient_metric_history_norm(
    a_hat_hist: Array,
    *,
    Nx: int,
    Lx: float,
    hermite_metric: Array,
) -> Array:
    a_hat_hist = jnp.asarray(a_hat_hist, dtype=jnp.complex128)
    a_phys_hist = jax.vmap(lambda a_hat: irfft_x(a_hat, int(Nx)))(a_hat_hist)
    metric = jnp.asarray(hermite_metric, dtype=jnp.float64)
    density_tx = jnp.einsum("tnx,nm,tmx->tx", a_phys_hist, metric, a_phys_hist)
    dx = jnp.asarray(float(Lx) / float(Nx), dtype=jnp.float64)
    return jnp.maximum(dx * jnp.sum(density_tx, axis=1), 0.0)


def resample_distribution_history_to_probe_grid(
    f_hist: Array,
    config: PhysicalGridVlasovPoissonConfig,
    *,
    v_probe: Array,
) -> Array:
    f_hist = jnp.asarray(f_hist, dtype=jnp.float64)
    v_probe = jnp.asarray(v_probe, dtype=jnp.float64)
    Nv = int(config.Nv)
    Nx = int(config.Nx)
    coords_1d = (v_probe - float(config.vmin)) / float(config.dv)
    coords = jnp.broadcast_to(coords_1d[:, None], (int(v_probe.shape[0]), Nx))
    sub = jnp.full((Nv - 1,), 1.0, dtype=jnp.float64)
    diag = jnp.full((Nv,), 4.0, dtype=jnp.float64)
    sup = jnp.full((Nv - 1,), 1.0, dtype=jnp.float64)

    def sample_one(f_state: Array) -> Array:
        coeffs = cubic_bspline_prefilter_constant(f_state, sub, diag, sup)
        return cubic_bspline_interp_constant(coeffs, coords, cval=0.0)

    return jax.vmap(sample_one)(f_hist).astype(jnp.float64)


def _split_episode_payloads(
    payloads: Sequence[Dict[str, Array]],
    *,
    val_fraction: float,
) -> Tuple[Sequence[Dict[str, Array]], Sequence[Dict[str, Array]]]:
    if len(payloads) <= 1:
        return payloads, payloads
    n_val = max(1, int(round(len(payloads) * float(val_fraction))))
    n_val = min(n_val, len(payloads) - 1)
    return payloads[:-n_val], payloads[-n_val:]


def _stack_episode_payloads(payloads: Sequence[Dict[str, Array]]) -> Dict[str, Array]:
    if not payloads:
        return {}
    keys = tuple(payloads[0].keys())
    out: Dict[str, Array] = {}
    for key in keys:
        out[key] = jnp.stack([jnp.asarray(payload[key]) for payload in payloads], axis=0)
    return out


def build_physical_reference_episode(
    config: PhysicalGridVlasovPoissonConfig,
    perturbation_x: Array,
    *,
    v_probe: Array,
) -> Dict[str, Array]:
    equilibrium = maxwellian_equilibrium(config.v)
    perturb = jnp.asarray(perturbation_x, dtype=jnp.float64)
    f0 = equilibrium[:, None] * (1.0 + perturb[None, :])
    raw = run_semilagrangian_vlasov_poisson(
        config,
        f0,
        history_stride=1,
        return_state_history=True,
    )
    f_hist = jnp.asarray(raw["state_history"], dtype=jnp.float64)
    e_hat_hist = jax.vmap(
        lambda f_state: jnp.fft.rfft(
            compute_electric_field_from_distribution(f_state, config)
        ).astype(jnp.complex128)
    )(f_hist)
    sampled_hist = resample_distribution_history_to_probe_grid(f_hist, config, v_probe=v_probe)
    eq_probe = maxwellian_equilibrium(jnp.asarray(v_probe, dtype=jnp.float64))
    return {
        "times": jnp.asarray(raw["state_history_times"], dtype=jnp.float64),
        "E_hat_ref": e_hat_hist,
        "delta_f_ref": sampled_hist - eq_probe[None, :, None],
    }


def build_projected_reference_episode(
    config: PhysicalGridVlasovPoissonConfig,
    perturbation_x: Array,
    *,
    projection_orders: Sequence[int],
    stored_projection_orders: Optional[Sequence[int]] = None,
    closure_q_targets: Sequence[int] = (),
    compact_rollout_qloss: bool = False,
    rollout_horizon: int = 0,
    rollout_anchor_pool_size: int = 0,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_FORWARD,
) -> Dict[str, Array]:
    coeff_histories, k_arr = _run_landau_teacher_projected_histories(
        config,
        perturbation_x,
        projection_orders=projection_orders,
        history_stride=1,
    )
    stored_orders = (
        tuple(int(order) for order in projection_orders)
        if stored_projection_orders is None
        else tuple(int(order) for order in stored_projection_orders)
    )
    payload: Dict[str, Array] = {}
    if bool(compact_rollout_qloss):
        if str(rollout_direction) != ONLINE_ROLLOUT_DIRECTION_FORWARD:
            raise ValueError("compact rollout q-loss reference caches currently require forward rollouts")
        if int(rollout_horizon) <= 0:
            raise ValueError("compact rollout q-loss reference caches require rollout_horizon > 0")
        if int(rollout_anchor_pool_size) <= 0:
            raise ValueError("compact rollout q-loss reference caches require rollout_anchor_pool_size > 0")
        for target_nv in closure_q_targets:
            target_i = int(target_nv)
            if target_i not in coeff_histories:
                raise ValueError(f"compact rollout q-loss requires retained coefficients for Nv={target_i}")
            retained_hist = jnp.asarray(coeff_histories[target_i], dtype=jnp.complex128)
            anchor_indices = _select_rollout_anchor_indices(
                history_length=int(retained_hist.shape[0]),
                rollout_horizon=int(rollout_horizon),
                rollout_anchor_samples=int(rollout_anchor_pool_size),
                rollout_direction=ONLINE_ROLLOUT_DIRECTION_FORWARD,
            )
            prev_indices = jnp.maximum(anchor_indices - 1, 0)
            prev_prev_indices = jnp.maximum(anchor_indices - 2, 0)
            anchor_stencils = jnp.stack(
                (
                    jnp.take(retained_hist, anchor_indices, axis=0),
                    jnp.take(retained_hist, prev_indices, axis=0),
                    jnp.take(retained_hist, prev_prev_indices, axis=0),
                ),
                axis=1,
            )
            payload[online_reference_anchor_coeff_key(target_i)] = anchor_stencils
            payload[online_reference_anchor_index_key(target_i)] = anchor_indices
    else:
        for order in stored_orders:
            payload[online_reference_coeff_key(int(order))] = jnp.asarray(
                coeff_histories[int(order)],
                dtype=jnp.complex128,
            )
    k_arr_j = jnp.asarray(k_arr, dtype=jnp.float64)
    for target_nv in closure_q_targets:
        target_i = int(target_nv)
        projection_order = target_i + 1
        if projection_order not in coeff_histories:
            raise ValueError(
                f"closure q target Nv={target_i} requires projected coefficient order {projection_order}"
            )
        coeff_hist = jnp.asarray(coeff_histories[projection_order], dtype=jnp.complex128)
        payload[online_reference_q_key(target_i)] = (
            -1j
            * k_arr_j[None, :]
            * math.sqrt(float(target_i))
            * coeff_hist[:, target_i, :]
        ).astype(jnp.complex128)
    return payload


def build_online_reference_dataset(
    *,
    dataset_cache: Optional[Path],
    regimes: Sequence[str],
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_L: float,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    linear_T: float,
    linear_eps: float,
    linear_modes: Sequence[float],
    linear_num_samples: int,
    linear_seed: int,
    linear_poisson_sign: float,
    nonlinear_T: float,
    nonlinear_k0: float,
    nonlinear_poisson_sign: float,
    weak_eps: Sequence[float],
    strong_eps: Sequence[float],
    val_fraction: float,
    online_v_probes: int,
    online_loss_backend: str,
    Nv_targets: Sequence[int],
    rollout_horizon: int,
    rollout_anchor_samples: int,
    rollout_anchor_pool_size: int,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
) -> Tuple[Dict[str, Dict[str, Dict[str, Array]]], Array]:
    effective_online_v_probes = (
        int(online_v_probes)
        if str(online_loss_backend) == ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1
        else 0
    )
    cache_metadata = build_online_reference_cache_metadata(
        regimes=regimes,
        teacher_Nx=teacher_Nx,
        teacher_Nv=teacher_Nv,
        teacher_L=teacher_L,
        teacher_vmin=teacher_vmin,
        teacher_vmax=teacher_vmax,
        teacher_dt=teacher_dt,
        linear_T=linear_T,
        linear_eps=linear_eps,
        linear_modes=linear_modes,
        linear_num_samples=linear_num_samples,
        linear_seed=linear_seed,
        linear_poisson_sign=linear_poisson_sign,
        nonlinear_T=nonlinear_T,
        nonlinear_k0=nonlinear_k0,
        nonlinear_poisson_sign=nonlinear_poisson_sign,
        weak_eps=weak_eps,
        strong_eps=strong_eps,
        val_fraction=val_fraction,
        online_v_probes=effective_online_v_probes,
        online_loss_backend=online_loss_backend,
        Nv_targets=(
            tuple(int(v) for v in Nv_targets)
            if online_loss_backend_uses_projected_coefficients(str(online_loss_backend))
            else None
        ),
        rollout_horizon=rollout_horizon,
        rollout_anchor_pool_size=(
            int(rollout_anchor_pool_size)
            if online_loss_backend_uses_rollout_qloss(str(online_loss_backend))
            and str(rollout_direction) == ONLINE_ROLLOUT_DIRECTION_FORWARD
            else int(rollout_anchor_samples)
        ),
    )
    if dataset_cache is not None and dataset_cache.exists():
        try:
            cached_dataset, cached_v_probe = load_online_reference_cache(
                dataset_cache,
                expected_metadata=cache_metadata,
            )
            selected = {regime: cached_dataset[regime] for regime in regimes}
            return selected, jnp.asarray(cached_v_probe, dtype=jnp.float64)
        except ValueError as exc:
            print(f"[data] ignoring incompatible online reference cache {dataset_cache}: {exc}")

    if str(online_loss_backend) == ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1:
        v_probe = jnp.linspace(float(teacher_vmin), float(teacher_vmax), effective_online_v_probes, dtype=jnp.float64)
    else:
        v_probe = jnp.zeros((0,), dtype=jnp.float64)
    dataset: Dict[str, Dict[str, Dict[str, Array]]] = {}

    if online_loss_backend_uses_projected_coefficients(str(online_loss_backend)):
        target_nvs = tuple(sorted({int(v) for v in Nv_targets}))
        if not target_nvs:
            raise ValueError(f"{online_loss_backend} requires at least one target Nv")
        if int(rollout_horizon) <= 0:
            raise ValueError(f"{online_loss_backend} requires rollout_horizon > 0")
        closure_q_targets = (
            target_nvs
            if online_loss_backend_has_reference_q_targets(str(online_loss_backend))
            else ()
        )
        compact_rollout_qloss = (
            online_loss_backend_uses_rollout_qloss(str(online_loss_backend))
            and str(rollout_direction) == ONLINE_ROLLOUT_DIRECTION_FORWARD
        )
        effective_anchor_pool_size = (
            int(rollout_anchor_pool_size)
            if compact_rollout_qloss
            else int(rollout_anchor_samples)
        )
        projection_orders = tuple(sorted(
            set(target_nvs).union({int(v) + 1 for v in closure_q_targets})
        ))

        if REGIME_LINEAR in regimes:
            config = PhysicalGridVlasovPoissonConfig(
                Nx=int(teacher_Nx),
                Nv=int(teacher_Nv),
                Lx=float(teacher_L),
                vmin=float(teacher_vmin),
                vmax=float(teacher_vmax),
                dt=float(teacher_dt),
                T=float(linear_T),
                poisson_sign=float(linear_poisson_sign),
                snapshot_times=(),
            )
            rng = np.random.default_rng(int(linear_seed))
            x = np.asarray(config.x, dtype=np.float64)
            payloads: List[Dict[str, Array]] = []
            for _ in range(int(linear_num_samples)):
                perturb = sample_initial_condition(rng, x, linear_modes, linear_eps)
                payload = build_projected_reference_episode(
                    config,
                    perturb,
                    projection_orders=projection_orders,
                    stored_projection_orders=() if compact_rollout_qloss else target_nvs,
                    closure_q_targets=closure_q_targets,
                    compact_rollout_qloss=compact_rollout_qloss,
                    rollout_horizon=int(rollout_horizon),
                    rollout_anchor_pool_size=int(effective_anchor_pool_size),
                    rollout_direction=str(rollout_direction),
                )
                payloads.append(payload)
            train_payloads, val_payloads = _split_episode_payloads(payloads, val_fraction=val_fraction)
            dataset[REGIME_LINEAR] = {
                "train": _stack_episode_payloads(train_payloads),
                "val": _stack_episode_payloads(val_payloads),
            }

        nonlinear_config = PhysicalGridVlasovPoissonConfig(
            Nx=int(teacher_Nx),
            Nv=int(teacher_Nv),
            Lx=float(teacher_L),
            vmin=float(teacher_vmin),
            vmax=float(teacher_vmax),
            dt=float(teacher_dt),
            T=float(nonlinear_T),
            poisson_sign=float(nonlinear_poisson_sign),
            snapshot_times=(),
        )
        perturb_template = np.cos(float(nonlinear_k0) * np.asarray(nonlinear_config.x, dtype=np.float64))

        for regime_name, eps_values in ((REGIME_WEAK, weak_eps), (REGIME_STRONG, strong_eps)):
            if regime_name not in regimes:
                continue
            payloads = []
            for eps in eps_values:
                payloads.append(
                    build_projected_reference_episode(
                        nonlinear_config,
                        float(eps) * perturb_template,
                        projection_orders=projection_orders,
                        stored_projection_orders=() if compact_rollout_qloss else target_nvs,
                        closure_q_targets=closure_q_targets,
                        compact_rollout_qloss=compact_rollout_qloss,
                        rollout_horizon=int(rollout_horizon),
                        rollout_anchor_pool_size=int(effective_anchor_pool_size),
                        rollout_direction=str(rollout_direction),
                    )
                )
            train_payloads, val_payloads = _split_episode_payloads(payloads, val_fraction=val_fraction)
            dataset[regime_name] = {
                "train": _stack_episode_payloads(train_payloads),
                "val": _stack_episode_payloads(val_payloads),
            }

        if dataset_cache is not None:
            save_online_reference_cache(
                dataset_cache,
                dataset,
                v_probe=v_probe,
                metadata=cache_metadata,
            )
        return dataset, v_probe

    if REGIME_LINEAR in regimes:
        config = PhysicalGridVlasovPoissonConfig(
            Nx=int(teacher_Nx),
            Nv=int(teacher_Nv),
            Lx=float(teacher_L),
            vmin=float(teacher_vmin),
            vmax=float(teacher_vmax),
            dt=float(teacher_dt),
            T=float(linear_T),
            poisson_sign=float(linear_poisson_sign),
            snapshot_times=(),
        )
        rng = np.random.default_rng(int(linear_seed))
        x = np.asarray(config.x, dtype=np.float64)
        payloads: List[Dict[str, Array]] = []
        for _ in range(int(linear_num_samples)):
            perturb = sample_initial_condition(rng, x, linear_modes, linear_eps)
            payload = build_physical_reference_episode(config, perturb, v_probe=v_probe)
            payload["perturbation_x"] = jnp.asarray(perturb, dtype=jnp.float64)
            payloads.append(payload)
        train_payloads, val_payloads = _split_episode_payloads(payloads, val_fraction=val_fraction)
        dataset[REGIME_LINEAR] = {
            "train": _stack_episode_payloads(train_payloads),
            "val": _stack_episode_payloads(val_payloads),
        }

    nonlinear_config = PhysicalGridVlasovPoissonConfig(
        Nx=int(teacher_Nx),
        Nv=int(teacher_Nv),
        Lx=float(teacher_L),
        vmin=float(teacher_vmin),
        vmax=float(teacher_vmax),
        dt=float(teacher_dt),
        T=float(nonlinear_T),
        poisson_sign=float(nonlinear_poisson_sign),
        snapshot_times=(),
    )
    perturb_template = np.cos(float(nonlinear_k0) * np.asarray(nonlinear_config.x, dtype=np.float64))

    for regime_name, eps_values in ((REGIME_WEAK, weak_eps), (REGIME_STRONG, strong_eps)):
        if regime_name not in regimes:
            continue
        payloads = []
        for eps in eps_values:
            payload = build_physical_reference_episode(
                nonlinear_config,
                float(eps) * perturb_template,
                v_probe=v_probe,
            )
            payload["eps"] = jnp.asarray(float(eps), dtype=jnp.float64)
            payloads.append(payload)
        train_payloads, val_payloads = _split_episode_payloads(payloads, val_fraction=val_fraction)
        dataset[regime_name] = {
            "train": _stack_episode_payloads(train_payloads),
            "val": _stack_episode_payloads(val_payloads),
        }

    if dataset_cache is not None:
        save_online_reference_cache(
            dataset_cache,
            dataset,
            v_probe=v_probe,
            metadata=cache_metadata,
        )
    return dataset, v_probe


def run_linear_landau_online_history(
    learned: LearnedInterfaceClosure,
    *,
    config: LinearLandauConfig,
    perturbation_x: Array,
) -> Array:
    integ = FourierHermiteIMEX(
        Nx=int(config.Nx),
        Nv=int(config.Nv),
        Lx=float(config.L),
        dt=float(config.dt),
        vth=1.0,
        dealias_23=False,
        closure=None,
    )
    m_eq = jnp.zeros((int(config.Nv),), dtype=jnp.float64).at[0].set(1.0)
    a_phys0 = jnp.zeros((int(config.Nv), int(config.Nx)), dtype=jnp.float64).at[0].set(
        jnp.asarray(perturbation_x, dtype=jnp.float64)
    )
    a_hat0 = integ.apply_mask_hat(rfft_x(a_phys0))
    n0 = linear_explicit_N_hat(
        a_hat0,
        integ,
        m_eq,
        poisson_sign=float(config.poisson_sign),
        dissipation=None,
    )
    b0 = learned_boundary_flux_hat(a_hat0, integ.k_arr, integ.Nv, integ.vth, learned)
    nsteps = int(round(float(config.T) / float(config.dt)))

    def step(carry, _):
        a_hat, n_prev, b_prev = carry
        n_hat = linear_explicit_N_hat(
            a_hat,
            integ,
            m_eq,
            poisson_sign=float(config.poisson_sign),
            dissipation=None,
        )
        b_hat = learned_boundary_flux_hat(a_hat, integ.k_arr, integ.Nv, integ.vth, learned)
        a_new = integ.step_cnab2(
            a_hat,
            n_hat,
            n_prev,
            extra_hat=b_hat,
            extra_hat_prev=b_prev,
        )
        return (a_new, n_hat, b_hat), a_new

    step = jax.checkpoint(step)
    (_, _, _), states = jax.lax.scan(step, (a_hat0, n0, b0), xs=None, length=nsteps)
    return jnp.concatenate([a_hat0[None, :, :], states], axis=0)


def run_nonlinear_landau_online_history(
    learned: LearnedInterfaceClosure,
    *,
    Nx: int,
    Nv: int,
    L: float,
    dt: float,
    T: float,
    eps: Array,
    k0: float,
    dealias_23: bool,
    poisson_sign: float,
    vth: float = 1.0,
) -> Array:
    integ = FourierHermiteIMEX(
        Nx=int(Nx),
        Nv=int(Nv),
        Lx=float(L),
        dt=float(dt),
        vth=float(vth),
        dealias_23=bool(dealias_23),
        closure=None,
    )
    m_eq = jnp.zeros((int(Nv),), dtype=jnp.float64).at[0].set(1.0)
    a_phys0 = jnp.zeros((int(Nv), int(Nx)), dtype=jnp.float64)
    a_phys0 = a_phys0.at[0].set(jnp.asarray(eps, dtype=jnp.float64) * jnp.cos(float(k0) * integ.x))
    a_hat0 = integ.apply_mask_hat(rfft_x(a_phys0))

    def explicit_n_hat(a_hat: Array) -> Array:
        a_phys = irfft_x(a_hat, int(Nx))
        e_phys = integ.E_phys_from_a_hat(a_hat, poisson_sign=float(poisson_sign))
        n_phys = jnp.zeros_like(a_phys)
        n_phys = n_phys.at[1:].set(
            -(integ.sqrt_n[1:, None] / float(vth))
            * e_phys[None, :]
            * (a_phys[:-1] + m_eq[:-1, None])
        )
        return integ.apply_mask_hat(rfft_x(n_phys))

    n0 = explicit_n_hat(a_hat0)
    b0 = learned_boundary_flux_hat(a_hat0, integ.k_arr, integ.Nv, integ.vth, learned)
    nsteps = int(round(float(T) / float(dt)))

    def step(carry, _):
        a_hat, n_prev, b_prev = carry
        n_hat = explicit_n_hat(a_hat)
        b_hat = learned_boundary_flux_hat(a_hat, integ.k_arr, integ.Nv, integ.vth, learned)
        a_new = integ.step_cnab2(
            a_hat,
            n_hat,
            n_prev,
            extra_hat=b_hat,
            extra_hat_prev=b_prev,
        )
        return (a_new, n_hat, b_hat), a_new

    step = jax.checkpoint(step)
    (_, _, _), states = jax.lax.scan(step, (a_hat0, n0, b0), xs=None, length=nsteps)
    return jnp.concatenate([a_hat0[None, :, :], states], axis=0)


def build_learned_interface_closure(
    *,
    params: Dict[str, Array],
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
    n_low: int,
    training_mode: str = OFFLINE_TRAINING_MODE,
    train_objective: str = "q_only",
    context_mode: str = "none",
    rollout_horizon: int = 0,
    rollout_anchor_samples: int = 0,
    tail_start_fraction: float = 2.0 / 3.0,
    loss_backend: Optional[str] = None,
    lambda_q: float = 1.0,
    lambda_E: float = 0.0,
    lambda_dist: float = 0.0,
    lambda_tail: float = 0.0,
    lambda_neg: float = 0.0,
    lambda_reg: float = 0.0,
    online_v_probes: int = 0,
    stability_loss_definition: Optional[str] = None,
) -> LearnedInterfaceClosure:
    return LearnedInterfaceClosure(
        params=params,
        Nm=Nm,
        k_scale=k_scale,
        nv_scale=nv_scale,
        input_mean=jnp.asarray(stats["input_mean"], dtype=jnp.float64),
        input_std=jnp.asarray(stats["input_std"], dtype=jnp.float64),
        target_mean=jnp.asarray(stats["target_mean"], dtype=jnp.float64),
        target_std=jnp.asarray(stats["target_std"], dtype=jnp.float64),
        hidden_width=int(hidden_width),
        res_blocks=int(res_blocks),
        Nv_targets=tuple(int(v) for v in Nv_targets),
        train_regimes=tuple(str(v) for v in train_regimes),
        teacher_backend=str(normalize_teacher_backend_name(teacher_backend)),
        teacher_Lx=float(teacher_Lx),
        teacher_Nx=int(teacher_Nx),
        teacher_Nv=int(teacher_Nv),
        teacher_vmin=float(teacher_vmin),
        teacher_vmax=float(teacher_vmax),
        teacher_dt=float(teacher_dt),
        teacher_proj_Nv=None if teacher_proj_Nv is None else int(teacher_proj_Nv),
        include_global_indicators=True,
        n_low=int(n_low),
        training_mode=str(training_mode),
        train_objective=str(train_objective),
        context_mode=str(context_mode),
        context_lags=1 if str(context_mode) == "lag1_delta" else 0,
        base_input_dim=2 * int(Nm) + 4,
        rollout_horizon=int(rollout_horizon),
        rollout_anchor_samples=int(rollout_anchor_samples),
        tail_start_fraction=float(tail_start_fraction),
        loss_backend=None if loss_backend is None else str(loss_backend),
        lambda_q=float(lambda_q),
        lambda_E=float(lambda_E),
        lambda_dist=float(lambda_dist),
        lambda_tail=float(lambda_tail),
        lambda_neg=float(lambda_neg),
        lambda_reg=float(lambda_reg),
        online_v_probes=int(online_v_probes),
        stability_loss_definition=(
            None
            if stability_loss_definition is None
            else str(stability_loss_definition)
        ),
    )


def make_regime_balanced_loss(
    prepared: Dict[str, Dict[str, Array]],
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
    n_low: int,
    context_mode: str,
):
    active_regimes = tuple(regime for regime in train_regimes if regime in prepared)
    weights = np.asarray([float(regime_weights[regime]) for regime in active_regimes], dtype=np.float64)
    weights = weights / np.sum(weights)

    def loss_fn(params: Dict[str, Array]) -> Array:
        learned = build_learned_interface_closure(
            params=params,
            Nm=Nm,
            k_scale=k_scale,
            nv_scale=nv_scale,
            stats=stats,
            hidden_width=hidden_width,
            res_blocks=res_blocks,
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
            n_low=n_low,
            train_objective="q_only",
            context_mode=context_mode,
        )
        losses = []
        for weight, regime in zip(weights, active_regimes):
            pred_std = learned.predict_standardized_components(prepared[regime]["train_inputs"])
            target_std = prepared[regime]["train_targets_std"]
            losses.append(float(weight) * jnp.mean((pred_std - target_std) ** 2))
        return jnp.sum(jnp.stack(losses))

    return loss_fn


def train_with_loss(
    params: Dict[str, Array],
    loss_fn,
    *,
    epochs: int,
    learning_rate: float,
    grad_clip: Optional[float],
    log_every: int,
) -> Tuple[Dict[str, Array], np.ndarray]:
    if int(epochs) <= 0:
        return params, np.zeros((0,), dtype=np.float64)

    state = adam_init(params)
    history = np.zeros((int(epochs),), dtype=np.float64)

    @jax.jit
    def train_step(
        current_params: Dict[str, Array],
        current_state: Dict[str, object],
    ) -> Tuple[Dict[str, Array], Dict[str, object], Array]:
        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        next_params, next_state = adam_step(
            current_params,
            grads,
            current_state,
            learning_rate,
            grad_clip=grad_clip,
        )
        return next_params, next_state, loss

    for epoch in range(int(epochs)):
        params, state, loss = train_step(params, state)
        history[epoch] = float(loss)
        if epoch == 0 or (epoch + 1) % max(int(log_every), 1) == 0 or epoch + 1 == int(epochs):
            print(f"[train] epoch {epoch + 1:04d}/{int(epochs):04d} loss={float(loss):.6e}")
    return params, history


def make_regime_balanced_batch_loss(
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
    n_low: int,
    context_mode: str,
):
    active_regimes = tuple(regime for regime in train_regimes if regime in regime_weights)
    weights = np.asarray([float(regime_weights[regime]) for regime in active_regimes], dtype=np.float64)
    weights = weights / np.sum(weights)

    def loss_fn(
        params: Dict[str, Array],
        batch_inputs: Dict[str, Array],
        batch_targets_std: Dict[str, Array],
    ) -> Array:
        learned = build_learned_interface_closure(
            params=params,
            Nm=Nm,
            k_scale=k_scale,
            nv_scale=nv_scale,
            stats=stats,
            hidden_width=hidden_width,
            res_blocks=res_blocks,
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
            n_low=n_low,
            train_objective="q_only",
            context_mode=context_mode,
        )
        losses = []
        for weight, regime in zip(weights, active_regimes):
            pred_std = learned.predict_standardized_components(batch_inputs[regime])
            target_std = batch_targets_std[regime]
            losses.append(float(weight) * jnp.mean((pred_std - target_std) ** 2))
        return jnp.sum(jnp.stack(losses))

    return loss_fn, active_regimes


def train_with_minibatch_loss(
    params: Dict[str, Array],
    prepared: Dict[str, Dict[str, Array]],
    batch_loss_fn,
    *,
    active_regimes: Sequence[str],
    epochs: int,
    learning_rate: float,
    grad_clip: Optional[float],
    log_every: int,
    batch_size: int,
    steps_per_epoch: int,
    seed: int,
) -> Tuple[Dict[str, Array], np.ndarray]:
    if int(epochs) <= 0:
        return params, np.zeros((0,), dtype=np.float64)
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive for minibatch training")
    if int(steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be positive for minibatch training")

    train_sizes = {
        regime: int(prepared[regime]["train_inputs"].shape[0])
        for regime in active_regimes
    }
    state = adam_init(params)
    history = np.zeros((int(epochs),), dtype=np.float64)

    @jax.jit
    def train_step(
        current_params: Dict[str, Array],
        current_state: Dict[str, object],
        batch_inputs: Dict[str, Array],
        batch_targets_std: Dict[str, Array],
    ) -> Tuple[Dict[str, Array], Dict[str, object], Array]:
        loss, grads = jax.value_and_grad(batch_loss_fn)(current_params, batch_inputs, batch_targets_std)
        next_params, next_state = adam_step(
            current_params,
            grads,
            current_state,
            learning_rate,
            grad_clip=grad_clip,
        )
        return next_params, next_state, loss

    rng = np.random.default_rng(int(seed))

    for epoch in range(int(epochs)):
        running_loss = jnp.asarray(0.0, dtype=jnp.float64)
        for _ in range(int(steps_per_epoch)):
            batch_inputs = {}
            batch_targets_std = {}
            for regime in active_regimes:
                idx = rng.integers(0, train_sizes[regime], size=int(batch_size), endpoint=False)
                batch_inputs[regime] = prepared[regime]["train_inputs"][idx]
                batch_targets_std[regime] = prepared[regime]["train_targets_std"][idx]
            params, state, loss = train_step(params, state, batch_inputs, batch_targets_std)
            running_loss = running_loss + loss
        history[epoch] = float(running_loss / float(steps_per_epoch))
        if epoch == 0 or (epoch + 1) % max(int(log_every), 1) == 0 or epoch + 1 == int(epochs):
            print(f"[train] epoch {epoch + 1:04d}/{int(epochs):04d} loss={history[epoch]:.6e}")
    return params, history


def l2_regularization(params: Dict[str, Array]) -> Array:
    return jnp.sum(
        jnp.stack([jnp.sum(jnp.abs(value) ** 2) for value in jax.tree_util.tree_leaves(params)])
    )


def tail_mode_weights(Nv: int, tail_start_fraction: float) -> np.ndarray:
    start = min(int(math.ceil(float(tail_start_fraction) * float(Nv))), int(Nv) - 1)
    count = int(Nv) - start
    if count <= 0:
        start = int(Nv) - 1
        count = 1
    ramp = np.linspace(0.0, 1.0, count, dtype=np.float64) ** 2
    if count == 1:
        ramp[...] = 1.0
    return ramp


def _rollout_k_weights(nk: int) -> Array:
    weights = jnp.ones((int(nk),), dtype=jnp.float64)
    if int(nk) > 2:
        weights = weights.at[1:-1].set(2.0)
    return weights


def _rollout_step_weights(nt: int) -> Array:
    if int(nt) <= 1:
        return jnp.ones((int(nt),), dtype=jnp.float64)
    ramp = jnp.linspace(0.0, 1.0, int(nt), dtype=jnp.float64)
    return 1.0 + 3.0 * (ramp ** 2)


def _rollout_n_weights(nv: int) -> Array:
    nv_i = int(nv)
    weights = jnp.ones((nv_i,), dtype=jnp.float64)
    if nv_i <= 0:
        return weights
    weights = weights.at[0].set(8.0)
    if nv_i > 1:
        tail_start = min(int(math.ceil((2.0 / 3.0) * float(nv_i))), nv_i - 1)
        count = nv_i - tail_start
        if count > 0:
            tail_ramp = 1.0 + 7.0 * (jnp.linspace(0.0, 1.0, count, dtype=jnp.float64) ** 2)
            weights = weights.at[tail_start:].set(jnp.maximum(weights[tail_start:], tail_ramp))
    return weights


def online_trajectory_loss_terms(
    a_hat_hist: Array,
    *,
    k_arr: Array,
    ref_E_hat: Array,
    ref_delta_f: Array,
    Nx: int,
    v_probe: Array,
    eq_probe: Array,
    tail_start_fraction: float,
    poisson_sign: float,
) -> Tuple[Array, Array, Array, Array]:
    pred_E_hat = e_hat_history_from_a_hat_history(
        jnp.asarray(a_hat_hist, dtype=jnp.complex128),
        jnp.asarray(k_arr, dtype=jnp.float64),
        poisson_sign=float(poisson_sign),
    )
    pred_delta_f = reconstruct_delta_f_from_a_hat_history(
        a_hat_hist,
        Nx=int(Nx),
        v_probe=v_probe,
        vth=1.0,
    )
    k_weights = _rollout_k_weights(int(pred_E_hat.shape[1]))
    field_num = jnp.sum(k_weights[None, 1:] * (jnp.abs(pred_E_hat[:, 1:] - ref_E_hat[:, 1:]) ** 2))
    field_den = jnp.sum(k_weights[None, 1:] * (jnp.abs(ref_E_hat[:, 1:]) ** 2)) + 1e-30
    field_loss = field_num / field_den

    ref_delta_f = jnp.asarray(ref_delta_f, dtype=jnp.float64)
    dist_num = jnp.mean((pred_delta_f - ref_delta_f) ** 2)
    dist_den = jnp.mean(ref_delta_f ** 2) + 1e-30
    dist_loss = dist_num / dist_den

    Nv = int(a_hat_hist.shape[1])
    tail_start = min(int(math.ceil(float(tail_start_fraction) * float(Nv))), int(Nv) - 1)
    tail_weights = jnp.asarray(tail_mode_weights(Nv, tail_start_fraction), dtype=jnp.float64)
    tail_energy = jnp.mean(
        tail_weights[None, :, None] * k_weights[None, None, :] * (jnp.abs(a_hat_hist[:, tail_start:, :]) ** 2)
    )
    tail_loss = tail_energy / dist_den

    full_f = eq_probe[None, :, None] + pred_delta_f
    neg_num = jnp.mean(jax.nn.relu(-full_f) ** 2)
    neg_den = jnp.mean(eq_probe ** 2) + 1e-30
    neg_loss = neg_num / neg_den
    return field_loss, dist_loss, tail_loss, neg_loss


def online_full_state_loss_terms(
    pred_a_hat_hist: Array,
    ref_a_hat_hist: Array,
    *,
    k_mask: Optional[Array] = None,
) -> Tuple[Array, Array]:
    pred_a_hat_hist = jnp.asarray(pred_a_hat_hist, dtype=jnp.complex128)
    ref_a_hat_hist = jnp.asarray(ref_a_hat_hist, dtype=jnp.complex128)
    time_weights = _rollout_step_weights(int(ref_a_hat_hist.shape[0]))
    n_weights = _rollout_n_weights(int(ref_a_hat_hist.shape[1]))
    k_weights = _rollout_k_weights(int(ref_a_hat_hist.shape[2]))
    if k_mask is not None:
        k_weights = k_weights * jnp.asarray(k_mask, dtype=jnp.float64)
    weights = (
        time_weights[:, None, None]
        * n_weights[None, :, None]
        * k_weights[None, None, :]
    )
    num = jnp.sum(weights * (jnp.abs(pred_a_hat_hist - ref_a_hat_hist) ** 2))
    den = jnp.sum(weights * (jnp.abs(ref_a_hat_hist) ** 2))
    return num, den


def online_projected_xv_loss_terms(
    pred_a_hat_hist: Array,
    ref_a_hat_hist: Array,
    *,
    Nx: int,
    Lx: float,
    v_grid: Array,
    vth: float = 1.0,
    tail_window: int = 0,
    k_mask: Optional[Array] = None,
    hermite_metric: Optional[Array] = None,
) -> Array:
    """Relative projected physical-space L2 loss induced by the reconstruction basis.

    The common equilibrium f0 cancels in the difference, so reconstructing the
    perturbation delta-f is enough for the physical-space distribution error.  When
    ``tail_window`` is set, both numerator and denominator are restricted to the
    same closure-adjacent Hermite window so the loss measures error relative to
    the scale of the tail the closure is supposed to control.
    """
    pred_a_hat_hist = jnp.asarray(pred_a_hat_hist, dtype=jnp.complex128)
    ref_a_hat_hist = jnp.asarray(ref_a_hat_hist, dtype=jnp.complex128)
    diff_a_hat = pred_a_hat_hist - ref_a_hat_hist
    ref_scale_a_hat = ref_a_hat_hist
    if k_mask is not None:
        mask = jnp.asarray(k_mask, dtype=jnp.float64)[None, None, :]
        diff_a_hat = diff_a_hat * mask
        ref_scale_a_hat = ref_scale_a_hat * mask
    if int(tail_window) > 0 and int(tail_window) < int(diff_a_hat.shape[1]):
        tail_start = int(diff_a_hat.shape[1]) - int(tail_window)
        n_mask = (jnp.arange(int(diff_a_hat.shape[1])) >= tail_start).astype(jnp.float64)
        n_mask = n_mask[None, :, None]
        diff_a_hat = diff_a_hat * n_mask
        ref_scale_a_hat = ref_scale_a_hat * n_mask
    if hermite_metric is not None:
        num_t = coefficient_metric_history_norm(
            diff_a_hat,
            Nx=int(Nx),
            Lx=float(Lx),
            hermite_metric=hermite_metric,
        )
        den_t = coefficient_metric_history_norm(
            ref_scale_a_hat,
            Nx=int(Nx),
            Lx=float(Lx),
            hermite_metric=hermite_metric,
        )
        return jnp.sum(num_t / (den_t + 1e-30))
    diff_delta_f = reconstruct_delta_f_from_a_hat_history(
        diff_a_hat,
        Nx=int(Nx),
        v_probe=v_grid,
        vth=float(vth),
    )
    ref_delta_f = reconstruct_delta_f_from_a_hat_history(
        ref_scale_a_hat,
        Nx=int(Nx),
        v_probe=v_grid,
        vth=float(vth),
    )
    v_grid = jnp.asarray(v_grid, dtype=jnp.float64)
    loss_v = jnp.trapezoid(diff_delta_f ** 2, x=v_grid, axis=1)
    ref_v = jnp.trapezoid(ref_delta_f ** 2, x=v_grid, axis=1)
    dx = jnp.asarray(float(Lx) / float(Nx), dtype=jnp.float64)
    num_t = dx * jnp.sum(loss_v, axis=1)
    den_t = dx * jnp.sum(ref_v, axis=1)
    return jnp.sum(num_t / (den_t + 1e-30))


def online_field_hat_loss_terms(
    pred_a_hat_hist: Array,
    ref_a_hat_hist: Array,
    *,
    k_arr: Array,
    poisson_sign: float,
    k_mask: Optional[Array] = None,
) -> Tuple[Array, Array]:
    pred_e_hat_hist = e_hat_history_from_a_hat_history(
        jnp.asarray(pred_a_hat_hist, dtype=jnp.complex128),
        jnp.asarray(k_arr, dtype=jnp.float64),
        poisson_sign=float(poisson_sign),
    )
    ref_e_hat_hist = e_hat_history_from_a_hat_history(
        jnp.asarray(ref_a_hat_hist, dtype=jnp.complex128),
        jnp.asarray(k_arr, dtype=jnp.float64),
        poisson_sign=float(poisson_sign),
    )
    time_weights = _rollout_step_weights(int(ref_e_hat_hist.shape[0]))
    k_weights = _rollout_k_weights(int(ref_e_hat_hist.shape[1]))
    if k_mask is not None:
        k_weights = k_weights * jnp.asarray(k_mask, dtype=jnp.float64)
    weights = time_weights[:, None] * k_weights[None, :]
    num = jnp.sum(weights * (jnp.abs(pred_e_hat_hist - ref_e_hat_hist) ** 2))
    den = jnp.sum(weights * (jnp.abs(ref_e_hat_hist) ** 2))
    return num, den


def online_closure_flux_loss_terms(
    pred_q_hat_hist: Array,
    ref_q_hat_hist: Array,
    *,
    k_mask: Optional[Array] = None,
) -> Tuple[Array, Array]:
    pred_q_hat_hist = jnp.asarray(pred_q_hat_hist, dtype=jnp.complex128)
    ref_q_hat_hist = jnp.asarray(ref_q_hat_hist, dtype=jnp.complex128)
    time_weights = _rollout_step_weights(int(ref_q_hat_hist.shape[0]))
    k_weights = _rollout_k_weights(int(ref_q_hat_hist.shape[1]))
    if k_mask is not None:
        k_weights = k_weights * jnp.asarray(k_mask, dtype=jnp.float64)
    weights = time_weights[:, None] * k_weights[None, :]
    num = jnp.sum(weights * (jnp.abs(pred_q_hat_hist - ref_q_hat_hist) ** 2))
    den = jnp.sum(weights * (jnp.abs(ref_q_hat_hist) ** 2))
    return num, den


def online_standardized_q_loss_terms(
    pred_q_hat_hist: Array,
    ref_q_hat_hist: Array,
    *,
    target_std: Array,
) -> Array:
    """Offline q-loss geometry evaluated on a q history."""
    pred_q_hat_hist = jnp.asarray(pred_q_hat_hist, dtype=jnp.complex128)
    ref_q_hat_hist = jnp.asarray(ref_q_hat_hist, dtype=jnp.complex128)
    pred_components = jnp.stack(
        [jnp.real(pred_q_hat_hist[:, 1:]), jnp.imag(pred_q_hat_hist[:, 1:])],
        axis=-1,
    )
    ref_components = jnp.stack(
        [jnp.real(ref_q_hat_hist[:, 1:]), jnp.imag(ref_q_hat_hist[:, 1:])],
        axis=-1,
    )
    std = jnp.maximum(jnp.asarray(target_std, dtype=jnp.float64), 1e-12)
    diff_std = (pred_components - ref_components) / std[None, None, :]
    return jnp.mean(diff_std**2)


def online_direct_q_relative_mse_for_history(
    ref_a_hat_hist: Array,
    ref_q_hat_hist: Array,
    *,
    learned: LearnedInterfaceClosure,
    k_arr: Array,
    Nv: int,
    k_mask: Optional[Array] = None,
) -> Array:
    """Diagnostic only: direct teacher-q error on reference states.

    This is not added to the online objective. It checks whether a rollout loss
    that is decreasing is also learning the actual boundary flux q_k.
    """
    ref_a_hat_hist = jnp.asarray(ref_a_hat_hist, dtype=jnp.complex128)
    ref_q_hat_hist = jnp.asarray(ref_q_hat_hist, dtype=jnp.complex128)
    if int(ref_a_hat_hist.shape[0]) > 1:
        states = ref_a_hat_hist[1:]
        prev_states = ref_a_hat_hist[:-1]
        targets = ref_q_hat_hist[1:]
    else:
        states = ref_a_hat_hist
        prev_states = ref_a_hat_hist
        targets = ref_q_hat_hist

    preds = jax.vmap(
        lambda state, prev_state: learned_interface_q_hat(
            state,
            k_arr,
            int(Nv),
            learned,
            a_hat_prev=prev_state,
        )
    )(states, prev_states)
    diff = preds[:, 1:] - targets[:, 1:]
    target = targets[:, 1:]
    if k_mask is not None:
        weights = jnp.asarray(k_mask, dtype=jnp.float64)[1:]
        num = jnp.sum(weights[None, :] * (jnp.abs(diff) ** 2))
        den = jnp.sum(weights[None, :] * (jnp.abs(target) ** 2))
    else:
        num = jnp.sum(jnp.abs(diff) ** 2)
        den = jnp.sum(jnp.abs(target) ** 2)
    return num / (den + 1e-30)


def online_rollout_q_relative_mse_for_history(
    ref_hist: Array,
    ref_q_hist: Array,
    *,
    learned: LearnedInterfaceClosure,
    forward_integ: FourierHermiteIMEX,
    backward_integ: FourierHermiteIMEX,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    explicit_n_hat_fn,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
    rollout_anchor_indices: Optional[Array] = None,
) -> Array:
    """Diagnostic only: q error on the same rollout windows as the state loss."""
    ref_hist = jnp.asarray(ref_hist, dtype=jnp.complex128)
    ref_q_hist = jnp.asarray(ref_q_hist, dtype=jnp.complex128)
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for rollout q diagnostic")
    direction_mode = str(rollout_direction)
    if direction_mode not in ALL_ONLINE_ROLLOUT_DIRECTIONS:
        raise ValueError(
            f"rollout_direction must be one of {ALL_ONLINE_ROLLOUT_DIRECTIONS!r}, "
            f"got {rollout_direction!r}"
        )
    anchor_indices = _resolve_rollout_anchor_indices(
        rollout_anchor_indices,
        history_length=int(ref_hist.shape[0]),
        rollout_horizon=horizon,
        rollout_anchor_samples=int(rollout_anchor_samples),
        rollout_direction=direction_mode,
    )
    offsets = jnp.arange(1, horizon + 1, dtype=jnp.int32)

    def anchor_step(carry, anchor_idx):
        pred_forward = rollout_closure_flux_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=forward_integ,
            rollout_horizon=horizon,
            direction=+1,
            explicit_n_hat_fn=explicit_n_hat_fn,
            detach_rollout_state_for_q=False,
        )
        ref_forward = jnp.take(ref_q_hist, anchor_idx + offsets, axis=0)
        num_forward, den_forward = online_closure_flux_loss_terms(
            pred_forward,
            ref_forward,
            k_mask=forward_integ.mask,
        )

        if direction_mode == ONLINE_ROLLOUT_DIRECTION_FORWARD:
            return (carry[0] + num_forward, carry[1] + den_forward), None

        pred_backward = rollout_closure_flux_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=backward_integ,
            rollout_horizon=horizon,
            direction=-1,
            explicit_n_hat_fn=explicit_n_hat_fn,
            detach_rollout_state_for_q=False,
        )
        ref_backward = jnp.take(ref_q_hist, anchor_idx - offsets, axis=0)
        num_backward, den_backward = online_closure_flux_loss_terms(
            pred_backward,
            ref_backward,
            k_mask=backward_integ.mask,
        )
        return (
            carry[0] + num_forward + num_backward,
            carry[1] + den_forward + den_backward,
        ), None

    (num_total, den_total), _ = jax.lax.scan(
        anchor_step,
        (
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        ),
        anchor_indices,
    )
    return num_total / (den_total + 1e-30)


def _closure_action_response(integ: FourierHermiteIMEX) -> Array:
    """Linear CNAB2 response of the next state to a unit current-step closure q."""
    basis = jnp.zeros((int(integ.Nv), int(integ.Nk)), dtype=jnp.complex128)
    basis = basis.at[int(integ.Nv) - 1].set(1.0 + 0.0j)
    rhs = float(integ.dt) * 1.5 * basis
    response = integ.implicit_solve(integ.apply_mask_hat(rhs))
    return integ.apply_mask_hat(response)


def _closure_action_q_target_from_state_response(
    *,
    base_next: Array,
    ref_next: Array,
    response: Array,
) -> Array:
    """Infer the current closure q that best corrects the next retained state."""
    response = jnp.asarray(response, dtype=jnp.complex128)
    residual = (
        jnp.asarray(ref_next, dtype=jnp.complex128)
        - jnp.asarray(base_next, dtype=jnp.complex128)
    )
    n_weights = _rollout_n_weights(int(response.shape[0]))[:, None]
    numerator = jnp.sum(n_weights * jnp.conj(response) * residual, axis=0)
    denom = jnp.sum(n_weights * (jnp.abs(response) ** 2), axis=0)
    q_target = jnp.where(
        denom > 1e-30,
        numerator / (denom + 1e-30),
        jnp.zeros_like(numerator),
    )
    if q_target.shape[0] > 0:
        q_target = q_target.at[0].set(0.0 + 0.0j)
    return jax.lax.stop_gradient(q_target.astype(jnp.complex128))


def _safe_history_state(ref_hist: Array, idx: Array) -> Array:
    idx = jnp.asarray(idx, dtype=jnp.int32)
    idx_clip = jnp.clip(idx, 0, int(ref_hist.shape[0]) - 1)
    return ref_hist[idx_clip]


def _linear_explicit_n_hat_for_state(
    a_hat: Array,
    *,
    integ: FourierHermiteIMEX,
    poisson_sign: float,
) -> Array:
    m_eq = jnp.zeros((int(integ.Nv),), dtype=jnp.float64).at[0].set(1.0)
    return linear_explicit_N_hat(
        a_hat,
        integ,
        m_eq,
        poisson_sign=float(poisson_sign),
        dissipation=None,
    )


def _nonlinear_explicit_n_hat_for_state(
    a_hat: Array,
    *,
    integ: FourierHermiteIMEX,
    poisson_sign: float,
) -> Array:
    m_eq = jnp.zeros((int(integ.Nv),), dtype=jnp.float64).at[0].set(1.0)
    a_phys = irfft_x(a_hat, int(integ.Nx))
    e_phys = integ.E_phys_from_a_hat(a_hat, poisson_sign=float(poisson_sign))
    n_phys = jnp.zeros_like(a_phys)
    n_phys = n_phys.at[1:].set(
        -(integ.sqrt_n[1:, None] / float(integ.vth))
        * e_phys[None, :]
        * (a_phys[:-1] + m_eq[:-1, None])
    )
    return integ.apply_mask_hat(rfft_x(n_phys))


def rollout_from_anchor_state(
    ref_hist: Array,
    *,
    anchor_idx: Array,
    learned: LearnedInterfaceClosure,
    integ: FourierHermiteIMEX,
    rollout_horizon: int,
    direction: int,
    explicit_n_hat_fn,
) -> Array:
    direction_i = int(direction)
    if direction_i not in {-1, 1}:
        raise ValueError(f"direction must be +/-1, got {direction!r}")
    anchor_idx = jnp.asarray(anchor_idx, dtype=jnp.int32)
    current_state = _safe_history_state(ref_hist, anchor_idx)
    prev_state = _safe_history_state(ref_hist, anchor_idx - direction_i)
    prev_prev_state = _safe_history_state(ref_hist, anchor_idx - 2 * direction_i)
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
        b_hat = learned_boundary_flux_hat(
            state,
            integ.k_arr,
            integ.Nv,
            integ.vth,
            learned,
            a_hat_prev=state_prev,
        )
        state_new = integ.step_cnab2(
            state,
            n_hat,
            n_prev_step,
            extra_hat=b_hat,
            extra_hat_prev=b_prev_step,
        )
        return (state_new, state, n_hat, b_hat), state_new

    init = (current_state, prev_state, n_prev, b_prev)
    (_, _, _, _), states = jax.lax.scan(step, init, xs=None, length=int(rollout_horizon))
    return states


def rollout_closure_flux_from_anchor_state(
    ref_hist: Array,
    *,
    anchor_idx: Array,
    learned: LearnedInterfaceClosure,
    integ: FourierHermiteIMEX,
    rollout_horizon: int,
    direction: int,
    explicit_n_hat_fn,
    detach_rollout_state_for_q: bool = False,
) -> Array:
    direction_i = int(direction)
    if direction_i not in {-1, 1}:
        raise ValueError(f"direction must be +/-1, got {direction!r}")
    anchor_idx = jnp.asarray(anchor_idx, dtype=jnp.int32)
    current_state = _safe_history_state(ref_hist, anchor_idx)
    prev_state = _safe_history_state(ref_hist, anchor_idx - direction_i)
    prev_prev_state = _safe_history_state(ref_hist, anchor_idx - 2 * direction_i)
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
        q_state = jax.lax.stop_gradient(state) if bool(detach_rollout_state_for_q) else state
        q_state_prev = (
            jax.lax.stop_gradient(state_prev)
            if bool(detach_rollout_state_for_q)
            else state_prev
        )
        q_hat = learned_interface_q_hat(
            q_state,
            integ.k_arr,
            integ.Nv,
            learned,
            a_hat_prev=q_state_prev,
        )
        b_hat = jnp.zeros_like(state, dtype=jnp.complex128).at[int(integ.Nv) - 1].set(q_hat)
        state_new = integ.step_cnab2(
            state,
            n_hat,
            n_prev_step,
            extra_hat=b_hat,
            extra_hat_prev=b_prev_step,
        )
        q_state_new = (
            jax.lax.stop_gradient(state_new)
            if bool(detach_rollout_state_for_q)
            else state_new
        )
        q_state_new_prev = (
            jax.lax.stop_gradient(state)
            if bool(detach_rollout_state_for_q)
            else state
        )
        q_new = learned_interface_q_hat(
            q_state_new,
            integ.k_arr,
            integ.Nv,
            learned,
            a_hat_prev=q_state_new_prev,
        )
        if bool(detach_rollout_state_for_q):
            return (
                jax.lax.stop_gradient(state_new),
                jax.lax.stop_gradient(state),
                jax.lax.stop_gradient(n_hat),
                jax.lax.stop_gradient(b_hat),
            ), q_new
        return (state_new, state, n_hat, b_hat), q_new

    init = (current_state, prev_state, n_prev, b_prev)
    (_, _, _, _), q_hist = jax.lax.scan(step, init, xs=None, length=int(rollout_horizon))
    return q_hist


def rollout_anchor_closure_flux_from_anchor_state(
    ref_hist: Array,
    *,
    anchor_idx: Array,
    learned: LearnedInterfaceClosure,
    integ: FourierHermiteIMEX,
    rollout_horizon: int,
    direction: int,
    explicit_n_hat_fn,
) -> Array:
    """Return q(C_h) for h=0..H-1, then step C_h -> C_{h+1}."""
    direction_i = int(direction)
    if direction_i not in {-1, 1}:
        raise ValueError(f"direction must be +/-1, got {direction!r}")
    anchor_idx = jnp.asarray(anchor_idx, dtype=jnp.int32)
    current_state = _safe_history_state(ref_hist, anchor_idx)
    prev_state = _safe_history_state(ref_hist, anchor_idx - direction_i)
    prev_prev_state = _safe_history_state(ref_hist, anchor_idx - 2 * direction_i)
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
        b_hat = jnp.zeros_like(state, dtype=jnp.complex128).at[int(integ.Nv) - 1].set(q_hat)
        state_new = integ.step_cnab2(
            state,
            n_hat,
            n_prev_step,
            extra_hat=b_hat,
            extra_hat_prev=b_prev_step,
        )
        return (state_new, state, n_hat, b_hat), q_hat

    init = (current_state, prev_state, n_prev, b_prev)
    (_, _, _, _), q_hist = jax.lax.scan(step, init, xs=None, length=int(rollout_horizon))
    return q_hist


def rollout_anchor_closure_flux_from_anchor_stencil(
    anchor_stencil: Array,
    *,
    learned: LearnedInterfaceClosure,
    integ: FourierHermiteIMEX,
    rollout_horizon: int,
    explicit_n_hat_fn,
) -> Array:
    """Return q(C_h) for h=0..H-1 from a compact forward CNAB2 anchor stencil.

    The stencil stores (current, previous, previous-previous) retained states.
    """
    stencil = jnp.asarray(anchor_stencil, dtype=jnp.complex128)
    if int(stencil.shape[0]) != 3:
        raise ValueError(f"anchor_stencil must have shape (3, Nv, Nk), got {stencil.shape}")
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
        b_hat = jnp.zeros_like(state, dtype=jnp.complex128).at[int(integ.Nv) - 1].set(q_hat)
        state_new = integ.step_cnab2(
            state,
            n_hat,
            n_prev_step,
            extra_hat=b_hat,
            extra_hat_prev=b_prev_step,
        )
        return (state_new, state, n_hat, b_hat), q_hat

    init = (current_state, prev_state, n_prev, b_prev)
    (_, _, _, _), q_hist = jax.lax.scan(step, init, xs=None, length=int(rollout_horizon))
    return q_hist


def rollout_closure_action_from_anchor_state(
    ref_hist: Array,
    *,
    anchor_idx: Array,
    learned: LearnedInterfaceClosure,
    integ: FourierHermiteIMEX,
    rollout_horizon: int,
    direction: int,
    explicit_n_hat_fn,
) -> Tuple[Array, Array]:
    direction_i = int(direction)
    if direction_i not in {-1, 1}:
        raise ValueError(f"direction must be +/-1, got {direction!r}")
    anchor_idx = jnp.asarray(anchor_idx, dtype=jnp.int32)
    current_state = _safe_history_state(ref_hist, anchor_idx)
    prev_state = _safe_history_state(ref_hist, anchor_idx - direction_i)
    prev_prev_state = _safe_history_state(ref_hist, anchor_idx - 2 * direction_i)
    n_prev = explicit_n_hat_fn(prev_state, integ=integ)
    b_prev = learned_boundary_flux_hat(
        prev_state,
        integ.k_arr,
        integ.Nv,
        integ.vth,
        learned,
        a_hat_prev=prev_prev_state,
    )
    response = _closure_action_response(integ)
    offsets = direction_i * jnp.arange(1, int(rollout_horizon) + 1, dtype=jnp.int32)
    ref_next_hist = jnp.take(ref_hist, anchor_idx + offsets, axis=0)

    def step(carry, ref_next):
        state, state_prev, n_prev_step, b_prev_step = carry
        state_sg = jax.lax.stop_gradient(state)
        state_prev_sg = jax.lax.stop_gradient(state_prev)
        n_prev_step_sg = jax.lax.stop_gradient(n_prev_step)
        b_prev_step_sg = jax.lax.stop_gradient(b_prev_step)
        n_hat = explicit_n_hat_fn(state_sg, integ=integ)

        zero_current = jnp.zeros_like(state_sg, dtype=jnp.complex128)
        base_next = integ.step_cnab2(
            state_sg,
            n_hat,
            n_prev_step_sg,
            extra_hat=zero_current,
            extra_hat_prev=b_prev_step_sg,
        )
        q_target = _closure_action_q_target_from_state_response(
            base_next=base_next,
            ref_next=ref_next,
            response=response,
        )
        q_pred = learned_interface_q_hat(
            state_sg,
            integ.k_arr,
            integ.Nv,
            learned,
            a_hat_prev=state_prev_sg,
        )
        b_hat = jnp.zeros_like(state_sg, dtype=jnp.complex128).at[int(integ.Nv) - 1].set(q_pred)
        state_new = integ.step_cnab2(
            state_sg,
            n_hat,
            n_prev_step_sg,
            extra_hat=b_hat,
            extra_hat_prev=b_prev_step_sg,
        )
        return (
            jax.lax.stop_gradient(state_new),
            state_sg,
            jax.lax.stop_gradient(n_hat),
            jax.lax.stop_gradient(b_hat),
        ), (q_pred, q_target)

    init = (
        jax.lax.stop_gradient(current_state),
        jax.lax.stop_gradient(prev_state),
        jax.lax.stop_gradient(n_prev),
        jax.lax.stop_gradient(b_prev),
    )
    (_, _, _, _), (q_pred_hist, q_target_hist) = jax.lax.scan(
        step,
        init,
        ref_next_hist,
        length=int(rollout_horizon),
    )
    return q_pred_hist, q_target_hist


def rollout_boundary_step_from_anchor_state(
    ref_hist: Array,
    *,
    anchor_idx: Array,
    learned: LearnedInterfaceClosure,
    integ: FourierHermiteIMEX,
    rollout_horizon: int,
    direction: int,
    explicit_n_hat_fn,
) -> Tuple[Array, Array]:
    direction_i = int(direction)
    if direction_i not in {-1, 1}:
        raise ValueError(f"direction must be +/-1, got {direction!r}")
    anchor_idx = jnp.asarray(anchor_idx, dtype=jnp.int32)
    current_state = _safe_history_state(ref_hist, anchor_idx)
    prev_state = _safe_history_state(ref_hist, anchor_idx - direction_i)
    prev_prev_state = _safe_history_state(ref_hist, anchor_idx - 2 * direction_i)
    n_prev = explicit_n_hat_fn(prev_state, integ=integ)
    b_prev = learned_boundary_flux_hat(
        prev_state,
        integ.k_arr,
        integ.Nv,
        integ.vth,
        learned,
        a_hat_prev=prev_prev_state,
    )
    offsets = direction_i * jnp.arange(1, int(rollout_horizon) + 1, dtype=jnp.int32)
    ref_next_hist = jnp.take(ref_hist, anchor_idx + offsets, axis=0)

    def step(carry, ref_next):
        state, state_prev, n_prev_step, b_prev_step = carry
        state_sg = jax.lax.stop_gradient(state)
        state_prev_sg = jax.lax.stop_gradient(state_prev)
        n_prev_step_sg = jax.lax.stop_gradient(n_prev_step)
        b_prev_step_sg = jax.lax.stop_gradient(b_prev_step)
        n_hat = explicit_n_hat_fn(state_sg, integ=integ)
        b_hat = learned_boundary_flux_hat(
            state_sg,
            integ.k_arr,
            integ.Nv,
            integ.vth,
            learned,
            a_hat_prev=state_prev_sg,
        )
        state_new = integ.step_cnab2(
            state_sg,
            n_hat,
            n_prev_step_sg,
            extra_hat=b_hat,
            extra_hat_prev=b_prev_step_sg,
        )
        return (
            jax.lax.stop_gradient(state_new),
            state_sg,
            jax.lax.stop_gradient(n_hat),
            jax.lax.stop_gradient(b_hat),
        ), (state_new[int(integ.Nv) - 1], ref_next[int(integ.Nv) - 1])

    init = (
        jax.lax.stop_gradient(current_state),
        jax.lax.stop_gradient(prev_state),
        jax.lax.stop_gradient(n_prev),
        jax.lax.stop_gradient(b_prev),
    )
    (_, _, _, _), (pred_boundary_hist, ref_boundary_hist) = jax.lax.scan(
        step,
        init,
        ref_next_hist,
        length=int(rollout_horizon),
    )
    return pred_boundary_hist, ref_boundary_hist


def _rollout_anchor_bounds(
    *,
    history_length: int,
    rollout_horizon: int,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
) -> Tuple[int, int, int]:
    direction_mode = str(rollout_direction)
    if direction_mode not in ALL_ONLINE_ROLLOUT_DIRECTIONS:
        raise ValueError(
            f"rollout_direction must be one of {ALL_ONLINE_ROLLOUT_DIRECTIONS!r}, "
            f"got {rollout_direction!r}"
        )
    if direction_mode == ONLINE_ROLLOUT_DIRECTION_FORWARD:
        start = 0
        stop = int(history_length) - int(rollout_horizon)
    else:
        start = int(rollout_horizon)
        stop = int(history_length) - int(rollout_horizon)
    num_valid = int(stop) - int(start)
    if num_valid <= 0:
        raise ValueError(
            f"Reference history length={int(history_length)} is too short for "
            f"rollout_horizon={int(rollout_horizon)} and rollout_direction={direction_mode!r}"
        )
    return int(start), int(stop), int(num_valid)


def _select_rollout_anchor_indices(
    *,
    history_length: int,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
) -> Array:
    start, stop, num_valid = _rollout_anchor_bounds(
        history_length=int(history_length),
        rollout_horizon=int(rollout_horizon),
        rollout_direction=str(rollout_direction),
    )
    if int(rollout_anchor_samples) <= 0 or int(rollout_anchor_samples) >= num_valid:
        return jnp.arange(
            int(start),
            int(stop),
            dtype=jnp.int32,
        )
    if int(rollout_anchor_samples) == 1:
        return jnp.asarray(
            [int(start) + ((num_valid - 1) // 2)],
            dtype=jnp.int32,
        )
    sampled_positions = np.rint(
        np.linspace(0, num_valid - 1, num=int(rollout_anchor_samples), dtype=np.float64)
    ).astype(np.int32)
    sampled_positions = np.unique(sampled_positions)
    return jnp.asarray(int(start) + sampled_positions, dtype=jnp.int32)


def _sample_rollout_anchor_indices(
    *,
    history_length: int,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    rollout_direction: str,
    rng: np.random.Generator,
) -> Array:
    start, stop, num_valid = _rollout_anchor_bounds(
        history_length=int(history_length),
        rollout_horizon=int(rollout_horizon),
        rollout_direction=str(rollout_direction),
    )
    if int(rollout_anchor_samples) <= 0 or int(rollout_anchor_samples) >= num_valid:
        anchors = np.arange(int(start), int(stop), dtype=np.int32)
    else:
        anchors = rng.choice(
            np.arange(int(start), int(stop), dtype=np.int32),
            size=int(rollout_anchor_samples),
            replace=False,
        )
        anchors = np.sort(anchors.astype(np.int32))
    return jnp.asarray(anchors, dtype=jnp.int32)


def _select_rollout_anchor_pool_indices(
    *,
    anchor_pool_size: int,
    rollout_anchor_samples: int,
) -> Array:
    pool_size = int(anchor_pool_size)
    if pool_size <= 0:
        raise ValueError("anchor_pool_size must be positive")
    if int(rollout_anchor_samples) <= 0 or int(rollout_anchor_samples) >= pool_size:
        return jnp.arange(pool_size, dtype=jnp.int32)
    if int(rollout_anchor_samples) == 1:
        return jnp.asarray([(pool_size - 1) // 2], dtype=jnp.int32)
    sampled_positions = np.rint(
        np.linspace(0, pool_size - 1, num=int(rollout_anchor_samples), dtype=np.float64)
    ).astype(np.int32)
    sampled_positions = np.unique(sampled_positions)
    return jnp.asarray(sampled_positions, dtype=jnp.int32)


def _sample_rollout_anchor_pool_indices(
    *,
    anchor_pool_size: int,
    rollout_anchor_samples: int,
    rng: np.random.Generator,
) -> Array:
    pool_size = int(anchor_pool_size)
    if pool_size <= 0:
        raise ValueError("anchor_pool_size must be positive")
    if int(rollout_anchor_samples) <= 0 or int(rollout_anchor_samples) >= pool_size:
        anchors = np.arange(pool_size, dtype=np.int32)
    else:
        anchors = rng.choice(
            np.arange(pool_size, dtype=np.int32),
            size=int(rollout_anchor_samples),
            replace=False,
        )
        anchors = np.sort(anchors.astype(np.int32))
    return jnp.asarray(anchors, dtype=jnp.int32)


def _resolve_rollout_anchor_pool_indices(
    rollout_anchor_indices: Optional[Array],
    *,
    anchor_pool_size: int,
    rollout_anchor_samples: int,
) -> Array:
    if rollout_anchor_indices is not None:
        return jnp.asarray(rollout_anchor_indices, dtype=jnp.int32)
    return _select_rollout_anchor_pool_indices(
        anchor_pool_size=int(anchor_pool_size),
        rollout_anchor_samples=int(rollout_anchor_samples),
    )


def _resolve_rollout_anchor_indices(
    rollout_anchor_indices: Optional[Array],
    *,
    history_length: int,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
) -> Array:
    if rollout_anchor_indices is not None:
        return jnp.asarray(rollout_anchor_indices, dtype=jnp.int32)
    return _select_rollout_anchor_indices(
        history_length=int(history_length),
        rollout_horizon=int(rollout_horizon),
        rollout_anchor_samples=int(rollout_anchor_samples),
        rollout_direction=str(rollout_direction),
    )


def online_fourier_hermite_bidir_loss_for_history(
    ref_hist: Array,
    *,
    learned: LearnedInterfaceClosure,
    forward_integ: FourierHermiteIMEX,
    backward_integ: FourierHermiteIMEX,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    explicit_n_hat_fn,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
    rollout_anchor_indices: Optional[Array] = None,
) -> Array:
    ref_hist = jnp.asarray(ref_hist, dtype=jnp.complex128)
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for fourier_hermite_bidir")
    direction_mode = str(rollout_direction)
    if direction_mode not in ALL_ONLINE_ROLLOUT_DIRECTIONS:
        raise ValueError(
            f"rollout_direction must be one of {ALL_ONLINE_ROLLOUT_DIRECTIONS!r}, "
            f"got {rollout_direction!r}"
        )
    anchor_indices = _resolve_rollout_anchor_indices(
        rollout_anchor_indices,
        history_length=int(ref_hist.shape[0]),
        rollout_horizon=horizon,
        rollout_anchor_samples=int(rollout_anchor_samples),
        rollout_direction=direction_mode,
    )
    offsets = jnp.arange(1, horizon + 1, dtype=jnp.int32)

    def anchor_step(carry, anchor_idx):
        pred_forward = rollout_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=forward_integ,
            rollout_horizon=horizon,
            direction=+1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        ref_forward = jnp.take(ref_hist, anchor_idx + offsets, axis=0)
        num_forward, den_forward = online_full_state_loss_terms(
            pred_forward,
            ref_forward,
            k_mask=forward_integ.mask,
        )

        if direction_mode == ONLINE_ROLLOUT_DIRECTION_FORWARD:
            return (carry[0] + num_forward, carry[1] + den_forward), None

        pred_backward = rollout_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=backward_integ,
            rollout_horizon=horizon,
            direction=-1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        ref_backward = jnp.take(ref_hist, anchor_idx - offsets, axis=0)
        num_backward, den_backward = online_full_state_loss_terms(
            pred_backward,
            ref_backward,
            k_mask=backward_integ.mask,
        )

        return (
            carry[0] + num_forward + num_backward,
            carry[1] + den_forward + den_backward,
        ), None

    (num_total, den_total), _ = jax.lax.scan(
        anchor_step,
        (
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        ),
        anchor_indices,
    )
    return num_total / (den_total + 1e-30)


def online_fourier_hermite_posterior_bidir_components_for_history(
    ref_hist: Array,
    *,
    learned: LearnedInterfaceClosure,
    forward_integ: FourierHermiteIMEX,
    backward_integ: FourierHermiteIMEX,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    explicit_n_hat_fn,
    poisson_sign: float,
    state_weight: float,
    field_weight: float,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
    rollout_anchor_indices: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
    ref_hist = jnp.asarray(ref_hist, dtype=jnp.complex128)
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for fourier_hermite_posterior_bidir")
    direction_mode = str(rollout_direction)
    if direction_mode not in ALL_ONLINE_ROLLOUT_DIRECTIONS:
        raise ValueError(
            f"rollout_direction must be one of {ALL_ONLINE_ROLLOUT_DIRECTIONS!r}, "
            f"got {rollout_direction!r}"
        )
    anchor_indices = _resolve_rollout_anchor_indices(
        rollout_anchor_indices,
        history_length=int(ref_hist.shape[0]),
        rollout_horizon=horizon,
        rollout_anchor_samples=int(rollout_anchor_samples),
        rollout_direction=direction_mode,
    )
    offsets = jnp.arange(1, horizon + 1, dtype=jnp.int32)

    def anchor_step(carry, anchor_idx):
        pred_forward = rollout_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=forward_integ,
            rollout_horizon=horizon,
            direction=+1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        ref_forward = jnp.take(ref_hist, anchor_idx + offsets, axis=0)
        state_num_forward, state_den_forward = online_full_state_loss_terms(
            pred_forward,
            ref_forward,
            k_mask=forward_integ.mask,
        )
        field_num_forward, field_den_forward = online_field_hat_loss_terms(
            pred_forward,
            ref_forward,
            k_arr=forward_integ.k_arr,
            poisson_sign=float(poisson_sign),
            k_mask=forward_integ.mask,
        )

        if direction_mode == ONLINE_ROLLOUT_DIRECTION_FORWARD:
            return (
                carry[0] + state_num_forward,
                carry[1] + state_den_forward,
                carry[2] + field_num_forward,
                carry[3] + field_den_forward,
            ), None

        pred_backward = rollout_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=backward_integ,
            rollout_horizon=horizon,
            direction=-1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        ref_backward = jnp.take(ref_hist, anchor_idx - offsets, axis=0)
        state_num_backward, state_den_backward = online_full_state_loss_terms(
            pred_backward,
            ref_backward,
            k_mask=backward_integ.mask,
        )
        field_num_backward, field_den_backward = online_field_hat_loss_terms(
            pred_backward,
            ref_backward,
            k_arr=backward_integ.k_arr,
            poisson_sign=float(poisson_sign),
            k_mask=backward_integ.mask,
        )

        return (
            carry[0] + state_num_forward + state_num_backward,
            carry[1] + state_den_forward + state_den_backward,
            carry[2] + field_num_forward + field_num_backward,
            carry[3] + field_den_forward + field_den_backward,
        ), None

    (state_num, state_den, field_num, field_den), _ = jax.lax.scan(
        anchor_step,
        (
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        ),
        anchor_indices,
    )
    state_loss = state_num / (state_den + 1e-30)
    field_loss = field_num / (field_den + 1e-30)
    total_loss = (
        jnp.asarray(float(state_weight), dtype=jnp.float64) * state_loss
        + jnp.asarray(float(field_weight), dtype=jnp.float64) * field_loss
    )
    return total_loss, state_loss, field_loss


def online_fourier_hermite_projected_xv_bidir_loss_for_history(
    ref_hist: Array,
    *,
    learned: LearnedInterfaceClosure,
    forward_integ: FourierHermiteIMEX,
    backward_integ: FourierHermiteIMEX,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    explicit_n_hat_fn,
    v_grid: Array,
    projected_xv_tail_window: int = 0,
    projected_xv_hermite_metric: Optional[Array] = None,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
    rollout_anchor_indices: Optional[Array] = None,
) -> Array:
    ref_hist = jnp.asarray(ref_hist, dtype=jnp.complex128)
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for fourier_hermite_projected_xv_bidir")
    direction_mode = str(rollout_direction)
    if direction_mode not in ALL_ONLINE_ROLLOUT_DIRECTIONS:
        raise ValueError(
            f"rollout_direction must be one of {ALL_ONLINE_ROLLOUT_DIRECTIONS!r}, "
            f"got {rollout_direction!r}"
        )
    anchor_indices = _resolve_rollout_anchor_indices(
        rollout_anchor_indices,
        history_length=int(ref_hist.shape[0]),
        rollout_horizon=horizon,
        rollout_anchor_samples=int(rollout_anchor_samples),
        rollout_direction=direction_mode,
    )
    offsets = jnp.arange(1, horizon + 1, dtype=jnp.int32)

    def anchor_step(carry, anchor_idx):
        pred_forward = rollout_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=forward_integ,
            rollout_horizon=horizon,
            direction=+1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        ref_forward = jnp.take(ref_hist, anchor_idx + offsets, axis=0)
        loss_forward = online_projected_xv_loss_terms(
            pred_forward,
            ref_forward,
            Nx=int(forward_integ.Nx),
            Lx=float(forward_integ.Lx),
            v_grid=v_grid,
            tail_window=int(projected_xv_tail_window),
            k_mask=forward_integ.mask,
            hermite_metric=projected_xv_hermite_metric,
        )

        if direction_mode == ONLINE_ROLLOUT_DIRECTION_FORWARD:
            return carry + loss_forward, None

        pred_backward = rollout_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=backward_integ,
            rollout_horizon=horizon,
            direction=-1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        ref_backward = jnp.take(ref_hist, anchor_idx - offsets, axis=0)
        loss_backward = online_projected_xv_loss_terms(
            pred_backward,
            ref_backward,
            Nx=int(backward_integ.Nx),
            Lx=float(backward_integ.Lx),
            v_grid=v_grid,
            tail_window=int(projected_xv_tail_window),
            k_mask=backward_integ.mask,
            hermite_metric=projected_xv_hermite_metric,
        )

        return carry + loss_forward + loss_backward, None

    loss_total, _ = jax.lax.scan(
        anchor_step,
        jnp.asarray(0.0, dtype=jnp.float64),
        anchor_indices,
    )
    direction_count = 1 if direction_mode == ONLINE_ROLLOUT_DIRECTION_FORWARD else 2
    sample_count = jnp.asarray(
        int(direction_count) * int(horizon) * int(anchor_indices.shape[0]),
        dtype=jnp.float64,
    )
    return loss_total / sample_count


def online_fourier_hermite_closure_bidir_loss_for_history(
    ref_hist: Array,
    ref_q_hist: Array,
    *,
    learned: LearnedInterfaceClosure,
    forward_integ: FourierHermiteIMEX,
    backward_integ: FourierHermiteIMEX,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    explicit_n_hat_fn,
    detach_rollout_state_for_q: bool = False,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
    rollout_anchor_indices: Optional[Array] = None,
) -> Array:
    ref_hist = jnp.asarray(ref_hist, dtype=jnp.complex128)
    ref_q_hist = jnp.asarray(ref_q_hist, dtype=jnp.complex128)
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for fourier_hermite_closure_bidir")
    direction_mode = str(rollout_direction)
    if direction_mode not in ALL_ONLINE_ROLLOUT_DIRECTIONS:
        raise ValueError(
            f"rollout_direction must be one of {ALL_ONLINE_ROLLOUT_DIRECTIONS!r}, "
            f"got {rollout_direction!r}"
        )
    anchor_indices = _resolve_rollout_anchor_indices(
        rollout_anchor_indices,
        history_length=int(ref_hist.shape[0]),
        rollout_horizon=horizon,
        rollout_anchor_samples=int(rollout_anchor_samples),
        rollout_direction=direction_mode,
    )
    offsets = jnp.arange(1, horizon + 1, dtype=jnp.int32)

    def anchor_step(carry, anchor_idx):
        pred_forward = rollout_closure_flux_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=forward_integ,
            rollout_horizon=horizon,
            direction=+1,
            explicit_n_hat_fn=explicit_n_hat_fn,
            detach_rollout_state_for_q=bool(detach_rollout_state_for_q),
        )
        ref_forward = jnp.take(ref_q_hist, anchor_idx + offsets, axis=0)
        num_forward, den_forward = online_closure_flux_loss_terms(
            pred_forward,
            ref_forward,
            k_mask=forward_integ.mask,
        )

        if direction_mode == ONLINE_ROLLOUT_DIRECTION_FORWARD:
            return (
                carry[0] + num_forward,
                carry[1] + den_forward,
            ), None

        pred_backward = rollout_closure_flux_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=backward_integ,
            rollout_horizon=horizon,
            direction=-1,
            explicit_n_hat_fn=explicit_n_hat_fn,
            detach_rollout_state_for_q=bool(detach_rollout_state_for_q),
        )
        ref_backward = jnp.take(ref_q_hist, anchor_idx - offsets, axis=0)
        num_backward, den_backward = online_closure_flux_loss_terms(
            pred_backward,
            ref_backward,
            k_mask=backward_integ.mask,
        )

        return (
            carry[0] + num_forward + num_backward,
            carry[1] + den_forward + den_backward,
        ), None

    (num_total, den_total), _ = jax.lax.scan(
        anchor_step,
        (
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        ),
        anchor_indices,
    )
    return num_total / (den_total + 1e-30)


def online_fourier_hermite_rollout_qloss_for_history(
    ref_hist: Array,
    ref_q_hist: Array,
    *,
    learned: LearnedInterfaceClosure,
    forward_integ: FourierHermiteIMEX,
    backward_integ: FourierHermiteIMEX,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    explicit_n_hat_fn,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
    rollout_anchor_indices: Optional[Array] = None,
) -> Array:
    ref_hist = jnp.asarray(ref_hist, dtype=jnp.complex128)
    ref_q_hist = jnp.asarray(ref_q_hist, dtype=jnp.complex128)
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for fourier_hermite_rollout_qloss")
    direction_mode = str(rollout_direction)
    if direction_mode not in ALL_ONLINE_ROLLOUT_DIRECTIONS:
        raise ValueError(
            f"rollout_direction must be one of {ALL_ONLINE_ROLLOUT_DIRECTIONS!r}, "
            f"got {rollout_direction!r}"
        )
    anchor_indices = _resolve_rollout_anchor_indices(
        rollout_anchor_indices,
        history_length=int(ref_hist.shape[0]),
        rollout_horizon=horizon,
        rollout_anchor_samples=int(rollout_anchor_samples),
        rollout_direction=direction_mode,
    )
    offsets = jnp.arange(0, horizon, dtype=jnp.int32)

    def anchor_step(total, anchor_idx):
        pred_forward = rollout_anchor_closure_flux_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=forward_integ,
            rollout_horizon=horizon,
            direction=+1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        ref_forward = jnp.take(ref_q_hist, anchor_idx + offsets, axis=0)
        loss_forward = online_standardized_q_loss_terms(
            pred_forward,
            ref_forward,
            target_std=learned.target_std,
        )

        if direction_mode == ONLINE_ROLLOUT_DIRECTION_FORWARD:
            return total + loss_forward, None

        pred_backward = rollout_anchor_closure_flux_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=backward_integ,
            rollout_horizon=horizon,
            direction=-1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        ref_backward = jnp.take(ref_q_hist, anchor_idx - offsets, axis=0)
        loss_backward = online_standardized_q_loss_terms(
            pred_backward,
            ref_backward,
            target_std=learned.target_std,
        )
        return total + loss_forward + loss_backward, None

    loss_total, _ = jax.lax.scan(
        anchor_step,
        jnp.asarray(0.0, dtype=jnp.float64),
        anchor_indices,
    )
    direction_count = 1 if direction_mode == ONLINE_ROLLOUT_DIRECTION_FORWARD else 2
    sample_count = jnp.asarray(
        int(direction_count) * int(anchor_indices.shape[0]),
        dtype=jnp.float64,
    )
    return loss_total / sample_count


def online_fourier_hermite_rollout_qloss_for_anchor_stencils(
    anchor_stencils: Array,
    anchor_time_indices: Array,
    ref_q_hist: Array,
    *,
    learned: LearnedInterfaceClosure,
    forward_integ: FourierHermiteIMEX,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    explicit_n_hat_fn,
    rollout_anchor_indices: Optional[Array] = None,
) -> Array:
    """H-step online q-loss from compact forward anchor stencils."""
    anchor_stencils = jnp.asarray(anchor_stencils, dtype=jnp.complex128)
    anchor_time_indices = jnp.asarray(anchor_time_indices, dtype=jnp.int32)
    ref_q_hist = jnp.asarray(ref_q_hist, dtype=jnp.complex128)
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for compact rollout q-loss")
    anchor_indices = _resolve_rollout_anchor_pool_indices(
        rollout_anchor_indices,
        anchor_pool_size=int(anchor_stencils.shape[0]),
        rollout_anchor_samples=int(rollout_anchor_samples),
    )
    offsets = jnp.arange(0, horizon, dtype=jnp.int32)

    def anchor_step(total, pool_idx):
        pred_forward = rollout_anchor_closure_flux_from_anchor_stencil(
            anchor_stencils[pool_idx],
            learned=learned,
            integ=forward_integ,
            rollout_horizon=horizon,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        ref_forward = jnp.take(ref_q_hist, anchor_time_indices[pool_idx] + offsets, axis=0)
        loss_forward = online_standardized_q_loss_terms(
            pred_forward,
            ref_forward,
            target_std=learned.target_std,
        )
        return total + loss_forward, None

    loss_total, _ = jax.lax.scan(
        anchor_step,
        jnp.asarray(0.0, dtype=jnp.float64),
        anchor_indices,
    )
    sample_count = jnp.asarray(int(anchor_indices.shape[0]), dtype=jnp.float64)
    return loss_total / sample_count


def online_fourier_hermite_closure_action_bidir_loss_for_history(
    ref_hist: Array,
    *,
    learned: LearnedInterfaceClosure,
    forward_integ: FourierHermiteIMEX,
    backward_integ: FourierHermiteIMEX,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    explicit_n_hat_fn,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
    rollout_anchor_indices: Optional[Array] = None,
) -> Array:
    ref_hist = jnp.asarray(ref_hist, dtype=jnp.complex128)
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for fourier_hermite_closure_action_bidir")
    direction_mode = str(rollout_direction)
    if direction_mode not in ALL_ONLINE_ROLLOUT_DIRECTIONS:
        raise ValueError(
            f"rollout_direction must be one of {ALL_ONLINE_ROLLOUT_DIRECTIONS!r}, "
            f"got {rollout_direction!r}"
        )
    anchor_indices = _resolve_rollout_anchor_indices(
        rollout_anchor_indices,
        history_length=int(ref_hist.shape[0]),
        rollout_horizon=horizon,
        rollout_anchor_samples=int(rollout_anchor_samples),
        rollout_direction=direction_mode,
    )

    def anchor_step(carry, anchor_idx):
        pred_forward, target_forward = rollout_closure_action_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=forward_integ,
            rollout_horizon=horizon,
            direction=+1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        num_forward, den_forward = online_closure_flux_loss_terms(
            pred_forward,
            target_forward,
            k_mask=forward_integ.mask,
        )

        if direction_mode == ONLINE_ROLLOUT_DIRECTION_FORWARD:
            return (
                carry[0] + num_forward,
                carry[1] + den_forward,
            ), None

        pred_backward, target_backward = rollout_closure_action_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=backward_integ,
            rollout_horizon=horizon,
            direction=-1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        num_backward, den_backward = online_closure_flux_loss_terms(
            pred_backward,
            target_backward,
            k_mask=backward_integ.mask,
        )

        return (
            carry[0] + num_forward + num_backward,
            carry[1] + den_forward + den_backward,
        ), None

    (num_total, den_total), _ = jax.lax.scan(
        anchor_step,
        (
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        ),
        anchor_indices,
    )
    return num_total / (den_total + 1e-30)


def online_fourier_hermite_boundary_step_bidir_loss_for_history(
    ref_hist: Array,
    *,
    learned: LearnedInterfaceClosure,
    forward_integ: FourierHermiteIMEX,
    backward_integ: FourierHermiteIMEX,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    explicit_n_hat_fn,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
    rollout_anchor_indices: Optional[Array] = None,
) -> Array:
    ref_hist = jnp.asarray(ref_hist, dtype=jnp.complex128)
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for fourier_hermite_boundary_step_bidir")
    direction_mode = str(rollout_direction)
    if direction_mode not in ALL_ONLINE_ROLLOUT_DIRECTIONS:
        raise ValueError(
            f"rollout_direction must be one of {ALL_ONLINE_ROLLOUT_DIRECTIONS!r}, "
            f"got {rollout_direction!r}"
        )
    anchor_indices = _resolve_rollout_anchor_indices(
        rollout_anchor_indices,
        history_length=int(ref_hist.shape[0]),
        rollout_horizon=horizon,
        rollout_anchor_samples=int(rollout_anchor_samples),
        rollout_direction=direction_mode,
    )

    def anchor_step(carry, anchor_idx):
        pred_forward, ref_forward = rollout_boundary_step_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=forward_integ,
            rollout_horizon=horizon,
            direction=+1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        num_forward, den_forward = online_closure_flux_loss_terms(
            pred_forward,
            ref_forward,
            k_mask=forward_integ.mask,
        )

        if direction_mode == ONLINE_ROLLOUT_DIRECTION_FORWARD:
            return (
                carry[0] + num_forward,
                carry[1] + den_forward,
            ), None

        pred_backward, ref_backward = rollout_boundary_step_from_anchor_state(
            ref_hist,
            anchor_idx=anchor_idx,
            learned=learned,
            integ=backward_integ,
            rollout_horizon=horizon,
            direction=-1,
            explicit_n_hat_fn=explicit_n_hat_fn,
        )
        num_backward, den_backward = online_closure_flux_loss_terms(
            pred_backward,
            ref_backward,
            k_mask=backward_integ.mask,
        )

        return (
            carry[0] + num_forward + num_backward,
            carry[1] + den_forward + den_backward,
        ), None

    (num_total, den_total), _ = jax.lax.scan(
        anchor_step,
        (
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        ),
        anchor_indices,
    )
    return num_total / (den_total + 1e-30)


def make_online_hybrid_batch_loss(
    *,
    prepared: Dict[str, Dict[str, Array]],
    online_dataset: Dict[str, Dict[str, Dict[str, Array]]],
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
    teacher_proj_Nv: int,
    n_low: int,
    context_mode: str,
    tail_start_fraction: float,
    loss_backend: str,
    lambda_q: float,
    lambda_E: float,
    lambda_dist: float,
    lambda_tail: float,
    lambda_neg: float,
    lambda_reg: float,
    online_v_probes: int,
    nonlinear_T: float,
    nonlinear_k0: float,
    poisson_sign: float,
    rollout_dealias_23: bool,
) -> Tuple[object, Sequence[str]]:
    active_regimes = tuple(
        regime
        for regime in train_regimes
        if regime in prepared
        and regime in online_dataset
        and int(prepared[regime]["train_inputs"].shape[0]) > 0
        and bool(online_dataset[regime].get("train"))
        and int(online_dataset[regime]["train"]["E_hat_ref"].shape[0]) > 0
    )
    weights = np.asarray([float(regime_weights[regime]) for regime in active_regimes], dtype=np.float64)
    weights = weights / np.sum(weights)
    weight_arr = jnp.asarray(weights, dtype=jnp.float64)
    target_nvs = tuple(int(v) for v in Nv_targets)
    target_nv_max = max(target_nvs)
    v_probe = jnp.linspace(float(teacher_vmin), float(teacher_vmax), int(online_v_probes), dtype=jnp.float64)
    eq_probe = maxwellian_equilibrium(v_probe)
    k_arr = FourierHermiteIMEX(
        Nx=int(teacher_Nx),
        Nv=int(target_nv_max),
        Lx=float(teacher_Lx),
        dt=float(teacher_dt),
        vth=1.0,
        dealias_23=bool(rollout_dealias_23),
        closure=None,
    ).k_arr
    linear_T = (
        float(online_dataset[REGIME_LINEAR]["train"]["times"].shape[1] - 1) * float(teacher_dt)
        if REGIME_LINEAR in online_dataset and online_dataset[REGIME_LINEAR].get("train")
        else float(nonlinear_T)
    )
    linear_configs = {
        int(target_nv): LinearLandauConfig(
            method="learned",
            Nv=int(target_nv),
            Nx=int(teacher_Nx),
            L=float(teacher_Lx),
            dt=float(teacher_dt),
            T=linear_T,
            poisson_sign=float(poisson_sign),
        )
        for target_nv in target_nvs
    }
    lambda_q_arr = jnp.asarray(float(lambda_q), dtype=jnp.float64)
    lambda_E_arr = jnp.asarray(float(lambda_E), dtype=jnp.float64)
    lambda_dist_arr = jnp.asarray(float(lambda_dist), dtype=jnp.float64)
    lambda_tail_arr = jnp.asarray(float(lambda_tail), dtype=jnp.float64)
    lambda_neg_arr = jnp.asarray(float(lambda_neg), dtype=jnp.float64)
    lambda_reg_arr = jnp.asarray(float(lambda_reg), dtype=jnp.float64)

    def make_loss_fn_for_target(target_nv: int):
        linear_config = linear_configs[int(target_nv)]

        def loss_fn_for_target(
            params: Dict[str, Array],
            q_batches: Dict[str, Dict[str, Array]],
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
                n_low=n_low,
                training_mode=ONLINE_TRAINING_MODE,
                train_objective="trajectory_q_hybrid",
                context_mode=context_mode,
                rollout_horizon=0,
                tail_start_fraction=tail_start_fraction,
                loss_backend=loss_backend,
                lambda_q=lambda_q,
                lambda_E=lambda_E,
                lambda_dist=lambda_dist,
                lambda_tail=lambda_tail,
                lambda_neg=lambda_neg,
                lambda_reg=lambda_reg,
                online_v_probes=online_v_probes,
                stability_loss_definition=ONLINE_HYBRID_LOSS_DEFINITION,
            )
            total_q = jnp.asarray(0.0, dtype=jnp.float64)
            total_field = jnp.asarray(0.0, dtype=jnp.float64)
            total_dist = jnp.asarray(0.0, dtype=jnp.float64)
            total_tail = jnp.asarray(0.0, dtype=jnp.float64)
            total_neg = jnp.asarray(0.0, dtype=jnp.float64)

            for weight, regime in zip(weight_arr, active_regimes):
                q_batch = q_batches[regime]
                pred_std = learned.predict_standardized_components(q_batch["inputs"])
                regime_q = jnp.mean((pred_std - q_batch["targets_std"]) ** 2)

                batch = regime_batches[regime]
                if regime == REGIME_LINEAR:
                    field_terms, dist_terms, tail_terms, neg_terms = jax.vmap(
                        lambda perturbation_x, ref_e_hat, ref_delta_f: online_trajectory_loss_terms(
                            run_linear_landau_online_history(
                                learned,
                                config=linear_config,
                                perturbation_x=perturbation_x,
                            ),
                            k_arr=k_arr,
                            ref_E_hat=ref_e_hat,
                            ref_delta_f=ref_delta_f,
                            Nx=int(teacher_Nx),
                            v_probe=v_probe,
                            eq_probe=eq_probe,
                            tail_start_fraction=tail_start_fraction,
                            poisson_sign=float(poisson_sign),
                        )
                    )(
                        batch["perturbation_x"],
                        batch["E_hat_ref"],
                        batch["delta_f_ref"],
                    )
                else:
                    field_terms, dist_terms, tail_terms, neg_terms = jax.vmap(
                        lambda eps, ref_e_hat, ref_delta_f: online_trajectory_loss_terms(
                            run_nonlinear_landau_online_history(
                                learned,
                                Nx=int(teacher_Nx),
                                Nv=int(target_nv),
                                L=float(teacher_Lx),
                                dt=float(teacher_dt),
                                T=float(nonlinear_T),
                                eps=eps,
                                k0=float(nonlinear_k0),
                                dealias_23=bool(rollout_dealias_23),
                                poisson_sign=float(poisson_sign),
                            ),
                            k_arr=k_arr,
                            ref_E_hat=ref_e_hat,
                            ref_delta_f=ref_delta_f,
                            Nx=int(teacher_Nx),
                            v_probe=v_probe,
                            eq_probe=eq_probe,
                            tail_start_fraction=tail_start_fraction,
                            poisson_sign=float(poisson_sign),
                        )
                    )(
                        batch["eps"],
                        batch["E_hat_ref"],
                        batch["delta_f_ref"],
                    )
                total_q = total_q + weight * (lambda_q_arr * regime_q)
                total_field = total_field + weight * (lambda_E_arr * jnp.mean(field_terms))
                total_dist = total_dist + weight * (lambda_dist_arr * jnp.mean(dist_terms))
                total_tail = total_tail + weight * (lambda_tail_arr * jnp.mean(tail_terms))
                total_neg = total_neg + weight * (lambda_neg_arr * jnp.mean(neg_terms))

            reg_term = lambda_reg_arr * l2_regularization(params)
            total_loss = total_q + total_field + total_dist + total_tail + total_neg + reg_term
            return total_loss, {
                "q": total_q,
                "state": jnp.asarray(0.0, dtype=jnp.float64),
                "field": total_field,
                "dist": total_dist,
                "tail": total_tail,
                "neg": total_neg,
                "reg": reg_term,
            }

        return loss_fn_for_target

    target_loss_fns = {
        int(target_nv): make_loss_fn_for_target(int(target_nv))
        for target_nv in target_nvs
    }
    default_target_nv = int(target_nvs[0])

    def loss_fn(
        params: Dict[str, Array],
        q_batches: Dict[str, Dict[str, Array]],
        regime_batches: Dict[str, Dict[str, Array]],
    ) -> Tuple[Array, Dict[str, Array]]:
        return target_loss_fns[default_target_nv](params, q_batches, regime_batches)

    loss_fn.target_nvs = target_nvs  # type: ignore[attr-defined]
    loss_fn.target_loss_fns = target_loss_fns  # type: ignore[attr-defined]
    return loss_fn, active_regimes


def make_online_fourier_hermite_bidir_batch_loss(
    *,
    online_dataset: Dict[str, Dict[str, Dict[str, Array]]],
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
    n_low: int,
    context_mode: str,
    rollout_horizon: int,
    rollout_anchor_samples: int,
    loss_backend: str,
    poisson_sign: float,
    rollout_dealias_23: bool,
    posterior_state_weight: float = 0.25,
    posterior_field_weight: float = 1.0,
    projected_xv_tail_window: int = 0,
    projected_xv_metric: str = PROJECTED_XV_METRIC_PHYSICAL_L2,
    rollout_direction: str = ONLINE_ROLLOUT_DIRECTION_BIDIR,
) -> Tuple[object, Sequence[str]]:
    target_nvs = tuple(int(v) for v in Nv_targets)
    if not target_nvs:
        raise ValueError(f"{loss_backend} requires at least one target Nv")
    direction_mode = str(rollout_direction)
    if direction_mode not in ALL_ONLINE_ROLLOUT_DIRECTIONS:
        raise ValueError(
            f"rollout_direction must be one of {ALL_ONLINE_ROLLOUT_DIRECTIONS!r}, "
            f"got {rollout_direction!r}"
        )
    if str(loss_backend) not in {
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_DETACHED_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_ROLLOUT_QLOSS,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_ACTION_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BOUNDARY_STEP_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_POSTERIOR_BIDIR,
        ONLINE_LOSS_BACKEND_FOURIER_HERMITE_PROJECTED_XV_BIDIR,
    }:
        raise ValueError(f"Unsupported Fourier-Hermite online backend {loss_backend!r}")
    active_regimes = tuple(
        regime
        for regime in train_regimes
        if regime in online_dataset
        and bool(online_dataset[regime].get("train"))
        and online_reference_num_cases(online_dataset[regime]["train"]) > 0
    )
    weights = np.asarray([float(regime_weights[regime]) for regime in active_regimes], dtype=np.float64)
    weights = weights / np.sum(weights)
    weight_arr = jnp.asarray(weights, dtype=jnp.float64)
    projected_xv_v_grid = jnp.linspace(
        float(teacher_vmin),
        float(teacher_vmax),
        int(teacher_Nv),
        dtype=jnp.float64,
    )
    projected_xv_metric_name = str(projected_xv_metric)
    if projected_xv_metric_name not in ALL_PROJECTED_XV_METRICS:
        raise ValueError(
            f"projected_xv_metric must be one of {ALL_PROJECTED_XV_METRICS!r}, "
            f"got {projected_xv_metric!r}"
        )
    projected_xv_metric_mats = {
        int(target_nv): build_projected_xv_metric_matrix(
            Nv=int(target_nv),
            v_grid=projected_xv_v_grid,
            metric=projected_xv_metric_name,
        )
        for target_nv in target_nvs
    }

    linear_integrators = {
        int(target_nv): (
            FourierHermiteIMEX(
                Nx=int(teacher_Nx),
                Nv=int(target_nv),
                Lx=float(teacher_Lx),
                dt=float(teacher_dt),
                vth=1.0,
                dealias_23=False,
                closure=None,
            ),
            FourierHermiteIMEX(
                Nx=int(teacher_Nx),
                Nv=int(target_nv),
                Lx=float(teacher_Lx),
                dt=-float(teacher_dt),
                vth=1.0,
                dealias_23=False,
                closure=None,
            ),
        )
        for target_nv in target_nvs
    }
    nonlinear_integrators = {
        int(target_nv): (
            FourierHermiteIMEX(
                Nx=int(teacher_Nx),
                Nv=int(target_nv),
                Lx=float(teacher_Lx),
                dt=float(teacher_dt),
                vth=1.0,
                dealias_23=bool(rollout_dealias_23),
                closure=None,
            ),
            FourierHermiteIMEX(
                Nx=int(teacher_Nx),
                Nv=int(target_nv),
                Lx=float(teacher_Lx),
                dt=-float(teacher_dt),
                vth=1.0,
                dealias_23=bool(rollout_dealias_23),
                closure=None,
            ),
        )
        for target_nv in target_nvs
    }

    def mean_history_loss(
        ref_batch: Array,
        *,
        learned: LearnedInterfaceClosure,
        forward_integ: FourierHermiteIMEX,
        backward_integ: FourierHermiteIMEX,
        explicit_n_hat_fn,
        rollout_anchor_indices: Optional[Array] = None,
    ) -> Array:
        ref_batch = jnp.asarray(ref_batch, dtype=jnp.complex128)

        def case_step(total, ref_hist):
            loss = online_fourier_hermite_bidir_loss_for_history(
                ref_hist,
                learned=learned,
                forward_integ=forward_integ,
                backward_integ=backward_integ,
                rollout_horizon=rollout_horizon,
                rollout_anchor_samples=rollout_anchor_samples,
                explicit_n_hat_fn=explicit_n_hat_fn,
                rollout_direction=direction_mode,
                rollout_anchor_indices=rollout_anchor_indices,
            )
            return total + loss, None

        total, _ = jax.lax.scan(
            case_step,
            jnp.asarray(0.0, dtype=jnp.float64),
            ref_batch,
        )
        return total / jnp.asarray(ref_batch.shape[0], dtype=jnp.float64)

    def mean_posterior_history_loss(
        ref_batch: Array,
        *,
        learned: LearnedInterfaceClosure,
        forward_integ: FourierHermiteIMEX,
        backward_integ: FourierHermiteIMEX,
        explicit_n_hat_fn,
        rollout_anchor_indices: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array]:
        ref_batch = jnp.asarray(ref_batch, dtype=jnp.complex128)

        def case_step(carry, ref_hist):
            loss, state_loss, field_loss = online_fourier_hermite_posterior_bidir_components_for_history(
                ref_hist,
                learned=learned,
                forward_integ=forward_integ,
                backward_integ=backward_integ,
                rollout_horizon=rollout_horizon,
                rollout_anchor_samples=rollout_anchor_samples,
                explicit_n_hat_fn=explicit_n_hat_fn,
                poisson_sign=float(poisson_sign),
                state_weight=float(posterior_state_weight),
                field_weight=float(posterior_field_weight),
                rollout_direction=direction_mode,
                rollout_anchor_indices=rollout_anchor_indices,
            )
            return (
                carry[0] + loss,
                carry[1] + state_loss,
                carry[2] + field_loss,
            ), None

        (loss_total, state_total, field_total), _ = jax.lax.scan(
            case_step,
            (
                jnp.asarray(0.0, dtype=jnp.float64),
                jnp.asarray(0.0, dtype=jnp.float64),
                jnp.asarray(0.0, dtype=jnp.float64),
            ),
            ref_batch,
        )
        scale = jnp.asarray(1.0 / float(ref_batch.shape[0]), dtype=jnp.float64)
        return loss_total * scale, state_total * scale, field_total * scale

    def mean_projected_xv_history_loss(
        ref_batch: Array,
        *,
        learned: LearnedInterfaceClosure,
        forward_integ: FourierHermiteIMEX,
        backward_integ: FourierHermiteIMEX,
        explicit_n_hat_fn,
        rollout_anchor_indices: Optional[Array] = None,
    ) -> Array:
        ref_batch = jnp.asarray(ref_batch, dtype=jnp.complex128)

        def case_step(total, ref_hist):
            loss = online_fourier_hermite_projected_xv_bidir_loss_for_history(
                ref_hist,
                learned=learned,
                forward_integ=forward_integ,
                backward_integ=backward_integ,
                rollout_horizon=rollout_horizon,
                rollout_anchor_samples=rollout_anchor_samples,
                explicit_n_hat_fn=explicit_n_hat_fn,
                v_grid=projected_xv_v_grid,
                projected_xv_tail_window=int(projected_xv_tail_window),
                projected_xv_hermite_metric=projected_xv_metric_mats[int(forward_integ.Nv)],
                rollout_direction=direction_mode,
                rollout_anchor_indices=rollout_anchor_indices,
            )
            return total + loss, None

        total, _ = jax.lax.scan(
            case_step,
            jnp.asarray(0.0, dtype=jnp.float64),
            ref_batch,
        )
        return total / jnp.asarray(ref_batch.shape[0], dtype=jnp.float64)

    def mean_rollout_q_diagnostic(
        ref_batch: Array,
        ref_q_batch: Array,
        *,
        learned: LearnedInterfaceClosure,
        forward_integ: FourierHermiteIMEX,
        backward_integ: FourierHermiteIMEX,
        explicit_n_hat_fn,
        rollout_anchor_indices: Optional[Array] = None,
    ) -> Array:
        ref_batch = jnp.asarray(ref_batch, dtype=jnp.complex128)
        ref_q_batch = jnp.asarray(ref_q_batch, dtype=jnp.complex128)

        def case_step(total, inputs):
            ref_hist, ref_q_hist = inputs
            q_rel_mse = online_rollout_q_relative_mse_for_history(
                ref_hist,
                ref_q_hist,
                learned=learned,
                forward_integ=forward_integ,
                backward_integ=backward_integ,
                rollout_horizon=rollout_horizon,
                rollout_anchor_samples=rollout_anchor_samples,
                explicit_n_hat_fn=explicit_n_hat_fn,
                rollout_direction=direction_mode,
                rollout_anchor_indices=rollout_anchor_indices,
            )
            return total + q_rel_mse, None

        total, _ = jax.lax.scan(
            case_step,
            jnp.asarray(0.0, dtype=jnp.float64),
            (ref_batch, ref_q_batch),
        )
        return total / jnp.asarray(ref_batch.shape[0], dtype=jnp.float64)

    def mean_closure_history_loss(
        ref_batch: Array,
        ref_q_batch: Array,
        *,
        learned: LearnedInterfaceClosure,
        forward_integ: FourierHermiteIMEX,
        backward_integ: FourierHermiteIMEX,
        explicit_n_hat_fn,
        rollout_anchor_indices: Optional[Array] = None,
    ) -> Array:
        ref_batch = jnp.asarray(ref_batch, dtype=jnp.complex128)
        ref_q_batch = jnp.asarray(ref_q_batch, dtype=jnp.complex128)
        detach_rollout_state_for_q = (
            str(loss_backend) == ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_DETACHED_BIDIR
        )

        def case_step(total, inputs):
            ref_hist, ref_q_hist = inputs
            if online_loss_backend_uses_action_q(str(loss_backend)):
                loss = online_fourier_hermite_closure_action_bidir_loss_for_history(
                    ref_hist,
                    learned=learned,
                    forward_integ=forward_integ,
                    backward_integ=backward_integ,
                    rollout_horizon=rollout_horizon,
                    rollout_anchor_samples=rollout_anchor_samples,
                    explicit_n_hat_fn=explicit_n_hat_fn,
                    rollout_direction=direction_mode,
                    rollout_anchor_indices=rollout_anchor_indices,
                )
            elif online_loss_backend_uses_rollout_qloss(str(loss_backend)):
                loss = online_fourier_hermite_rollout_qloss_for_history(
                    ref_hist,
                    ref_q_hist,
                    learned=learned,
                    forward_integ=forward_integ,
                    backward_integ=backward_integ,
                    rollout_horizon=rollout_horizon,
                    rollout_anchor_samples=rollout_anchor_samples,
                    explicit_n_hat_fn=explicit_n_hat_fn,
                    rollout_direction=direction_mode,
                    rollout_anchor_indices=rollout_anchor_indices,
                )
            else:
                loss = online_fourier_hermite_closure_bidir_loss_for_history(
                    ref_hist,
                    ref_q_hist,
                    learned=learned,
                    forward_integ=forward_integ,
                    backward_integ=backward_integ,
                    rollout_horizon=rollout_horizon,
                    rollout_anchor_samples=rollout_anchor_samples,
                    explicit_n_hat_fn=explicit_n_hat_fn,
                    detach_rollout_state_for_q=detach_rollout_state_for_q,
                    rollout_direction=direction_mode,
                    rollout_anchor_indices=rollout_anchor_indices,
                )
            return total + loss, None

        total, _ = jax.lax.scan(
            case_step,
            jnp.asarray(0.0, dtype=jnp.float64),
            (ref_batch, ref_q_batch),
        )
        return total / jnp.asarray(ref_batch.shape[0], dtype=jnp.float64)

    def mean_compact_rollout_q_loss(
        ref_anchor_batch: Array,
        ref_anchor_index_batch: Array,
        ref_q_batch: Array,
        *,
        learned: LearnedInterfaceClosure,
        forward_integ: FourierHermiteIMEX,
        explicit_n_hat_fn,
        rollout_anchor_indices: Optional[Array] = None,
    ) -> Array:
        ref_anchor_batch = jnp.asarray(ref_anchor_batch, dtype=jnp.complex128)
        ref_anchor_index_batch = jnp.asarray(ref_anchor_index_batch, dtype=jnp.int32)
        ref_q_batch = jnp.asarray(ref_q_batch, dtype=jnp.complex128)

        def case_step(total, inputs):
            anchor_stencils, anchor_indices, ref_q_hist = inputs
            loss = online_fourier_hermite_rollout_qloss_for_anchor_stencils(
                anchor_stencils,
                anchor_indices,
                ref_q_hist,
                learned=learned,
                forward_integ=forward_integ,
                rollout_horizon=rollout_horizon,
                rollout_anchor_samples=rollout_anchor_samples,
                explicit_n_hat_fn=explicit_n_hat_fn,
                rollout_anchor_indices=rollout_anchor_indices,
            )
            return total + loss, None

        total, _ = jax.lax.scan(
            case_step,
            jnp.asarray(0.0, dtype=jnp.float64),
            (ref_anchor_batch, ref_anchor_index_batch, ref_q_batch),
        )
        return total / jnp.asarray(ref_anchor_batch.shape[0], dtype=jnp.float64)

    def mean_boundary_history_loss(
        ref_batch: Array,
        *,
        learned: LearnedInterfaceClosure,
        forward_integ: FourierHermiteIMEX,
        backward_integ: FourierHermiteIMEX,
        explicit_n_hat_fn,
        rollout_anchor_indices: Optional[Array] = None,
    ) -> Array:
        ref_batch = jnp.asarray(ref_batch, dtype=jnp.complex128)

        def case_step(total, ref_hist):
            loss = online_fourier_hermite_boundary_step_bidir_loss_for_history(
                ref_hist,
                learned=learned,
                forward_integ=forward_integ,
                backward_integ=backward_integ,
                rollout_horizon=rollout_horizon,
                rollout_anchor_samples=rollout_anchor_samples,
                explicit_n_hat_fn=explicit_n_hat_fn,
                rollout_direction=direction_mode,
                rollout_anchor_indices=rollout_anchor_indices,
            )
            return total + loss, None

        total, _ = jax.lax.scan(
            case_step,
            jnp.asarray(0.0, dtype=jnp.float64),
            ref_batch,
        )
        return total / jnp.asarray(ref_batch.shape[0], dtype=jnp.float64)

    def make_loss_fn_for_target(target_nv: int):
        coeff_key = online_reference_coeff_key(int(target_nv))
        anchor_coeff_key = online_reference_anchor_coeff_key(int(target_nv))
        anchor_index_key = online_reference_anchor_index_key(int(target_nv))
        q_key = online_reference_q_key(int(target_nv))
        linear_forward, linear_backward = linear_integrators[int(target_nv)]
        nonlinear_forward, nonlinear_backward = nonlinear_integrators[int(target_nv)]

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
                Nv_targets=Nv_targets,
                train_regimes=train_regimes,
                teacher_backend=teacher_backend,
                teacher_Lx=teacher_Lx,
                teacher_Nx=teacher_Nx,
                teacher_Nv=teacher_Nv,
                teacher_vmin=teacher_vmin,
                teacher_vmax=teacher_vmax,
                teacher_dt=teacher_dt,
                teacher_proj_Nv=None,
                n_low=n_low,
                training_mode=ONLINE_TRAINING_MODE,
                train_objective="trajectory",
                context_mode=context_mode,
                rollout_horizon=rollout_horizon,
                rollout_anchor_samples=rollout_anchor_samples,
                loss_backend=str(loss_backend),
                lambda_q=0.0,
                lambda_E=0.0,
                lambda_dist=0.0,
                lambda_tail=0.0,
                lambda_neg=0.0,
                lambda_reg=0.0,
                online_v_probes=0,
            )
            total_q = jnp.asarray(0.0, dtype=jnp.float64)
            total_state = jnp.asarray(0.0, dtype=jnp.float64)
            total_field = jnp.asarray(0.0, dtype=jnp.float64)
            total_q_diag = jnp.asarray(0.0, dtype=jnp.float64)

            for weight, regime in zip(weight_arr, active_regimes):
                batch = regime_batches[regime]
                rollout_anchor_indices = (
                    jnp.asarray(batch[ROLLOUT_ANCHOR_INDICES_KEY], dtype=jnp.int32)
                    if ROLLOUT_ANCHOR_INDICES_KEY in batch
                    else None
                )
                use_anchor_stencils = (
                    online_loss_backend_uses_rollout_qloss(str(loss_backend))
                    and anchor_coeff_key in batch
                )
                ref_batch = jnp.asarray(
                    batch[anchor_coeff_key] if use_anchor_stencils else batch[coeff_key],
                    dtype=jnp.complex128,
                )
                ref_anchor_index_batch = (
                    jnp.asarray(batch[anchor_index_key], dtype=jnp.int32)
                    if use_anchor_stencils
                    else None
                )
                ref_q_batch = (
                    jnp.asarray(batch[q_key], dtype=jnp.complex128)
                    if q_key in batch
                    else None
                )
                if regime == REGIME_LINEAR:
                    if ref_q_batch is not None and not use_anchor_stencils:
                        total_q_diag = total_q_diag + weight * mean_rollout_q_diagnostic(
                            ref_batch,
                            ref_q_batch,
                            learned=learned,
                            forward_integ=linear_forward,
                            backward_integ=linear_backward,
                            explicit_n_hat_fn=linear_explicit,
                            rollout_anchor_indices=rollout_anchor_indices,
                        )
                    if online_loss_backend_uses_projected_xv(str(loss_backend)):
                        state_terms = jax.vmap(
                            lambda ref_hist: online_fourier_hermite_projected_xv_bidir_loss_for_history(
                                ref_hist,
                                learned=learned,
                                forward_integ=linear_forward,
                                backward_integ=linear_backward,
                                rollout_horizon=rollout_horizon,
                                rollout_anchor_samples=rollout_anchor_samples,
                                explicit_n_hat_fn=linear_explicit,
                                v_grid=projected_xv_v_grid,
                                projected_xv_tail_window=int(projected_xv_tail_window),
                                projected_xv_hermite_metric=projected_xv_metric_mats[int(linear_forward.Nv)],
                                rollout_direction=direction_mode,
                                rollout_anchor_indices=rollout_anchor_indices,
                            )
                        )(ref_batch)
                        total_state = total_state + weight * jnp.mean(state_terms)
                        continue
                    if online_loss_backend_uses_posterior_rollout(str(loss_backend)):
                        posterior_terms, state_terms, field_terms = jax.vmap(
                            lambda ref_hist: online_fourier_hermite_posterior_bidir_components_for_history(
                                ref_hist,
                                learned=learned,
                                forward_integ=linear_forward,
                                backward_integ=linear_backward,
                                rollout_horizon=rollout_horizon,
                                rollout_anchor_samples=rollout_anchor_samples,
                                explicit_n_hat_fn=linear_explicit,
                                poisson_sign=float(poisson_sign),
                                state_weight=float(posterior_state_weight),
                                field_weight=float(posterior_field_weight),
                                rollout_direction=direction_mode,
                                rollout_anchor_indices=rollout_anchor_indices,
                            )
                        )(ref_batch)
                        del posterior_terms
                        total_state = total_state + weight * jnp.mean(state_terms)
                        total_field = total_field + weight * jnp.mean(field_terms)
                        continue
                    if online_loss_backend_uses_boundary_step(str(loss_backend)):
                        state_terms = jax.vmap(
                            lambda ref_hist: online_fourier_hermite_boundary_step_bidir_loss_for_history(
                                ref_hist,
                                learned=learned,
                                forward_integ=linear_forward,
                                backward_integ=linear_backward,
                                rollout_horizon=rollout_horizon,
                                rollout_anchor_samples=rollout_anchor_samples,
                                explicit_n_hat_fn=linear_explicit,
                                rollout_direction=direction_mode,
                                rollout_anchor_indices=rollout_anchor_indices,
                            )
                        )(ref_batch)
                        total_state = total_state + weight * jnp.mean(state_terms)
                        continue
                    if online_loss_backend_uses_closure_q(str(loss_backend)):
                        assert ref_q_batch is not None
                        if online_loss_backend_uses_action_q(str(loss_backend)):
                            q_terms = jax.vmap(
                                lambda ref_hist: online_fourier_hermite_closure_action_bidir_loss_for_history(
                                    ref_hist,
                                    learned=learned,
                                    forward_integ=linear_forward,
                                    backward_integ=linear_backward,
                                    rollout_horizon=rollout_horizon,
                                    rollout_anchor_samples=rollout_anchor_samples,
                                    explicit_n_hat_fn=linear_explicit,
                                    rollout_direction=direction_mode,
                                    rollout_anchor_indices=rollout_anchor_indices,
                                )
                            )(ref_batch)
                        elif online_loss_backend_uses_rollout_qloss(str(loss_backend)):
                            if use_anchor_stencils:
                                assert ref_anchor_index_batch is not None
                                q_terms = jax.vmap(
                                    lambda anchor_stencils, anchor_indices, ref_q_hist: online_fourier_hermite_rollout_qloss_for_anchor_stencils(
                                        anchor_stencils,
                                        anchor_indices,
                                        ref_q_hist,
                                        learned=learned,
                                        forward_integ=linear_forward,
                                        rollout_horizon=rollout_horizon,
                                        rollout_anchor_samples=rollout_anchor_samples,
                                        explicit_n_hat_fn=linear_explicit,
                                        rollout_anchor_indices=rollout_anchor_indices,
                                    )
                                )(ref_batch, ref_anchor_index_batch, ref_q_batch)
                            else:
                                q_terms = jax.vmap(
                                    lambda ref_hist, ref_q_hist: online_fourier_hermite_rollout_qloss_for_history(
                                        ref_hist,
                                        ref_q_hist,
                                        learned=learned,
                                        forward_integ=linear_forward,
                                        backward_integ=linear_backward,
                                        rollout_horizon=rollout_horizon,
                                        rollout_anchor_samples=rollout_anchor_samples,
                                        explicit_n_hat_fn=linear_explicit,
                                        rollout_direction=direction_mode,
                                        rollout_anchor_indices=rollout_anchor_indices,
                                    )
                                )(ref_batch, ref_q_batch)
                        else:
                            q_terms = jax.vmap(
                                lambda ref_hist, ref_q_hist: online_fourier_hermite_closure_bidir_loss_for_history(
                                    ref_hist,
                                    ref_q_hist,
                                    learned=learned,
                                    forward_integ=linear_forward,
                                    backward_integ=linear_backward,
                                    rollout_horizon=rollout_horizon,
                                    rollout_anchor_samples=rollout_anchor_samples,
                                    explicit_n_hat_fn=linear_explicit,
                                    detach_rollout_state_for_q=(
                                        str(loss_backend)
                                        == ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_DETACHED_BIDIR
                                    ),
                                    rollout_direction=direction_mode,
                                    rollout_anchor_indices=rollout_anchor_indices,
                                )
                            )(ref_batch, ref_q_batch)
                        total_q = total_q + weight * jnp.mean(q_terms)
                        if use_anchor_stencils:
                            total_q_diag = total_q_diag + weight * jnp.mean(q_terms)
                        continue
                    state_terms = jax.vmap(
                        lambda ref_hist: online_fourier_hermite_bidir_loss_for_history(
                            ref_hist,
                            learned=learned,
                            forward_integ=linear_forward,
                            backward_integ=linear_backward,
                            rollout_horizon=rollout_horizon,
                            rollout_anchor_samples=rollout_anchor_samples,
                            explicit_n_hat_fn=linear_explicit,
                            rollout_direction=direction_mode,
                            rollout_anchor_indices=rollout_anchor_indices,
                        )
                    )(ref_batch)
                else:
                    if ref_q_batch is not None and not use_anchor_stencils:
                        total_q_diag = total_q_diag + weight * mean_rollout_q_diagnostic(
                            ref_batch,
                            ref_q_batch,
                            learned=learned,
                            forward_integ=nonlinear_forward,
                            backward_integ=nonlinear_backward,
                            explicit_n_hat_fn=nonlinear_explicit,
                            rollout_anchor_indices=rollout_anchor_indices,
                        )
                    if online_loss_backend_uses_projected_xv(str(loss_backend)):
                        state_terms = jax.vmap(
                            lambda ref_hist: online_fourier_hermite_projected_xv_bidir_loss_for_history(
                                ref_hist,
                                learned=learned,
                                forward_integ=nonlinear_forward,
                                backward_integ=nonlinear_backward,
                                rollout_horizon=rollout_horizon,
                                rollout_anchor_samples=rollout_anchor_samples,
                                explicit_n_hat_fn=nonlinear_explicit,
                                v_grid=projected_xv_v_grid,
                                projected_xv_tail_window=int(projected_xv_tail_window),
                                projected_xv_hermite_metric=projected_xv_metric_mats[int(nonlinear_forward.Nv)],
                                rollout_direction=direction_mode,
                                rollout_anchor_indices=rollout_anchor_indices,
                            )
                        )(ref_batch)
                        total_state = total_state + weight * jnp.mean(state_terms)
                        continue
                    if online_loss_backend_uses_posterior_rollout(str(loss_backend)):
                        posterior_terms, state_terms, field_terms = jax.vmap(
                            lambda ref_hist: online_fourier_hermite_posterior_bidir_components_for_history(
                                ref_hist,
                                learned=learned,
                                forward_integ=nonlinear_forward,
                                backward_integ=nonlinear_backward,
                                rollout_horizon=rollout_horizon,
                                rollout_anchor_samples=rollout_anchor_samples,
                                explicit_n_hat_fn=nonlinear_explicit,
                                poisson_sign=float(poisson_sign),
                                state_weight=float(posterior_state_weight),
                                field_weight=float(posterior_field_weight),
                                rollout_direction=direction_mode,
                                rollout_anchor_indices=rollout_anchor_indices,
                            )
                        )(ref_batch)
                        del posterior_terms
                        total_state = total_state + weight * jnp.mean(state_terms)
                        total_field = total_field + weight * jnp.mean(field_terms)
                        continue
                    if online_loss_backend_uses_boundary_step(str(loss_backend)):
                        state_terms = jax.vmap(
                            lambda ref_hist: online_fourier_hermite_boundary_step_bidir_loss_for_history(
                                ref_hist,
                                learned=learned,
                                forward_integ=nonlinear_forward,
                                backward_integ=nonlinear_backward,
                                rollout_horizon=rollout_horizon,
                                rollout_anchor_samples=rollout_anchor_samples,
                                explicit_n_hat_fn=nonlinear_explicit,
                                rollout_direction=direction_mode,
                                rollout_anchor_indices=rollout_anchor_indices,
                            )
                        )(ref_batch)
                        total_state = total_state + weight * jnp.mean(state_terms)
                        continue
                    if online_loss_backend_uses_closure_q(str(loss_backend)):
                        assert ref_q_batch is not None
                        if online_loss_backend_uses_action_q(str(loss_backend)):
                            q_terms = jax.vmap(
                                lambda ref_hist: online_fourier_hermite_closure_action_bidir_loss_for_history(
                                    ref_hist,
                                    learned=learned,
                                    forward_integ=nonlinear_forward,
                                    backward_integ=nonlinear_backward,
                                    rollout_horizon=rollout_horizon,
                                    rollout_anchor_samples=rollout_anchor_samples,
                                    explicit_n_hat_fn=nonlinear_explicit,
                                    rollout_direction=direction_mode,
                                    rollout_anchor_indices=rollout_anchor_indices,
                                )
                            )(ref_batch)
                        elif online_loss_backend_uses_rollout_qloss(str(loss_backend)):
                            if use_anchor_stencils:
                                assert ref_anchor_index_batch is not None
                                q_terms = jax.vmap(
                                    lambda anchor_stencils, anchor_indices, ref_q_hist: online_fourier_hermite_rollout_qloss_for_anchor_stencils(
                                        anchor_stencils,
                                        anchor_indices,
                                        ref_q_hist,
                                        learned=learned,
                                        forward_integ=nonlinear_forward,
                                        rollout_horizon=rollout_horizon,
                                        rollout_anchor_samples=rollout_anchor_samples,
                                        explicit_n_hat_fn=nonlinear_explicit,
                                        rollout_anchor_indices=rollout_anchor_indices,
                                    )
                                )(ref_batch, ref_anchor_index_batch, ref_q_batch)
                            else:
                                q_terms = jax.vmap(
                                    lambda ref_hist, ref_q_hist: online_fourier_hermite_rollout_qloss_for_history(
                                        ref_hist,
                                        ref_q_hist,
                                        learned=learned,
                                        forward_integ=nonlinear_forward,
                                        backward_integ=nonlinear_backward,
                                        rollout_horizon=rollout_horizon,
                                        rollout_anchor_samples=rollout_anchor_samples,
                                        explicit_n_hat_fn=nonlinear_explicit,
                                        rollout_direction=direction_mode,
                                        rollout_anchor_indices=rollout_anchor_indices,
                                    )
                                )(ref_batch, ref_q_batch)
                        else:
                            q_terms = jax.vmap(
                                lambda ref_hist, ref_q_hist: online_fourier_hermite_closure_bidir_loss_for_history(
                                    ref_hist,
                                    ref_q_hist,
                                    learned=learned,
                                    forward_integ=nonlinear_forward,
                                    backward_integ=nonlinear_backward,
                                    rollout_horizon=rollout_horizon,
                                    rollout_anchor_samples=rollout_anchor_samples,
                                    explicit_n_hat_fn=nonlinear_explicit,
                                    detach_rollout_state_for_q=(
                                        str(loss_backend)
                                        == ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_DETACHED_BIDIR
                                    ),
                                    rollout_direction=direction_mode,
                                    rollout_anchor_indices=rollout_anchor_indices,
                                )
                            )(ref_batch, ref_q_batch)
                        total_q = total_q + weight * jnp.mean(q_terms)
                        if use_anchor_stencils:
                            total_q_diag = total_q_diag + weight * jnp.mean(q_terms)
                        continue
                    state_terms = jax.vmap(
                        lambda ref_hist: online_fourier_hermite_bidir_loss_for_history(
                            ref_hist,
                            learned=learned,
                            forward_integ=nonlinear_forward,
                            backward_integ=nonlinear_backward,
                            rollout_horizon=rollout_horizon,
                            rollout_anchor_samples=rollout_anchor_samples,
                            explicit_n_hat_fn=nonlinear_explicit,
                            rollout_direction=direction_mode,
                            rollout_anchor_indices=rollout_anchor_indices,
                        )
                    )(ref_batch)
                total_state = total_state + weight * jnp.mean(state_terms)

            zero = jnp.asarray(0.0, dtype=jnp.float64)
            if online_loss_backend_uses_posterior_rollout(str(loss_backend)):
                total_loss = (
                    jnp.asarray(float(posterior_state_weight), dtype=jnp.float64) * total_state
                    + jnp.asarray(float(posterior_field_weight), dtype=jnp.float64) * total_field
                )
            else:
                total_loss = total_q + total_state
            return total_loss, {
                "q": total_q,
                "state": total_state,
                "field": total_field,
                "dist": zero,
                "tail": zero,
                "neg": zero,
                "reg": zero,
                "q_diag": total_q_diag,
            }

        return loss_fn_for_target

    def make_exact_loss_fn_for_target(target_nv: int):
        coeff_key = online_reference_coeff_key(int(target_nv))
        anchor_coeff_key = online_reference_anchor_coeff_key(int(target_nv))
        anchor_index_key = online_reference_anchor_index_key(int(target_nv))
        q_key = online_reference_q_key(int(target_nv))
        linear_forward, linear_backward = linear_integrators[int(target_nv)]
        nonlinear_forward, nonlinear_backward = nonlinear_integrators[int(target_nv)]

        def linear_explicit(a_hat: Array, *, integ: FourierHermiteIMEX) -> Array:
            return _linear_explicit_n_hat_for_state(a_hat, integ=integ, poisson_sign=float(poisson_sign))

        def nonlinear_explicit(a_hat: Array, *, integ: FourierHermiteIMEX) -> Array:
            return _nonlinear_explicit_n_hat_for_state(a_hat, integ=integ, poisson_sign=float(poisson_sign))

        def exact_loss_fn_for_target(
            params: Dict[str, Array],
        ) -> Tuple[Array, Dict[str, Array]]:
            learned = build_learned_interface_closure(
                params=params,
                Nm=Nm,
                k_scale=k_scale,
                nv_scale=nv_scale,
                stats=stats,
                hidden_width=hidden_width,
                res_blocks=res_blocks,
                Nv_targets=Nv_targets,
                train_regimes=train_regimes,
                teacher_backend=teacher_backend,
                teacher_Lx=teacher_Lx,
                teacher_Nx=teacher_Nx,
                teacher_Nv=teacher_Nv,
                teacher_vmin=teacher_vmin,
                teacher_vmax=teacher_vmax,
                teacher_dt=teacher_dt,
                teacher_proj_Nv=None,
                n_low=n_low,
                training_mode=ONLINE_TRAINING_MODE,
                train_objective="trajectory",
                context_mode=context_mode,
                rollout_horizon=rollout_horizon,
                rollout_anchor_samples=rollout_anchor_samples,
                loss_backend=str(loss_backend),
                lambda_q=0.0,
                lambda_E=0.0,
                lambda_dist=0.0,
                lambda_tail=0.0,
                lambda_neg=0.0,
                lambda_reg=0.0,
                online_v_probes=0,
            )
            total_q = jnp.asarray(0.0, dtype=jnp.float64)
            total_state = jnp.asarray(0.0, dtype=jnp.float64)
            total_field = jnp.asarray(0.0, dtype=jnp.float64)
            total_q_diag = jnp.asarray(0.0, dtype=jnp.float64)

            for weight, regime in zip(weight_arr, active_regimes):
                train_group = online_dataset[regime]["train"]
                use_anchor_stencils = (
                    online_loss_backend_uses_rollout_qloss(str(loss_backend))
                    and anchor_coeff_key in train_group
                )
                ref_batch = jnp.asarray(
                    train_group[anchor_coeff_key] if use_anchor_stencils else train_group[coeff_key],
                    dtype=jnp.complex128,
                )
                ref_anchor_index_batch = (
                    jnp.asarray(train_group[anchor_index_key], dtype=jnp.int32)
                    if use_anchor_stencils
                    else None
                )
                ref_q_batch = (
                    jnp.asarray(train_group[q_key], dtype=jnp.complex128)
                    if q_key in train_group
                    else None
                )
                if regime == REGIME_LINEAR:
                    if ref_q_batch is not None and not use_anchor_stencils:
                        total_q_diag = total_q_diag + weight * mean_rollout_q_diagnostic(
                            ref_batch,
                            ref_q_batch,
                            learned=learned,
                            forward_integ=linear_forward,
                            backward_integ=linear_backward,
                            explicit_n_hat_fn=linear_explicit,
                        )
                    if online_loss_backend_uses_projected_xv(str(loss_backend)):
                        regime_state = mean_projected_xv_history_loss(
                            ref_batch,
                            learned=learned,
                            forward_integ=linear_forward,
                            backward_integ=linear_backward,
                            explicit_n_hat_fn=linear_explicit,
                        )
                        total_state = total_state + weight * regime_state
                        continue
                    if online_loss_backend_uses_posterior_rollout(str(loss_backend)):
                        _, regime_state, regime_field = mean_posterior_history_loss(
                            ref_batch,
                            learned=learned,
                            forward_integ=linear_forward,
                            backward_integ=linear_backward,
                            explicit_n_hat_fn=linear_explicit,
                        )
                        total_state = total_state + weight * regime_state
                        total_field = total_field + weight * regime_field
                        continue
                    if online_loss_backend_uses_boundary_step(str(loss_backend)):
                        regime_state = mean_boundary_history_loss(
                            ref_batch,
                            learned=learned,
                            forward_integ=linear_forward,
                            backward_integ=linear_backward,
                            explicit_n_hat_fn=linear_explicit,
                        )
                        total_state = total_state + weight * regime_state
                        continue
                    if online_loss_backend_uses_closure_q(str(loss_backend)):
                        assert ref_q_batch is not None
                        if use_anchor_stencils:
                            assert ref_anchor_index_batch is not None
                            regime_q = mean_compact_rollout_q_loss(
                                ref_batch,
                                ref_anchor_index_batch,
                                ref_q_batch,
                                learned=learned,
                                forward_integ=linear_forward,
                                explicit_n_hat_fn=linear_explicit,
                            )
                            total_q_diag = total_q_diag + weight * regime_q
                        else:
                            regime_q = mean_closure_history_loss(
                                ref_batch,
                                ref_q_batch,
                                learned=learned,
                                forward_integ=linear_forward,
                                backward_integ=linear_backward,
                                explicit_n_hat_fn=linear_explicit,
                            )
                        total_q = total_q + weight * regime_q
                        continue
                    regime_state = mean_history_loss(
                        ref_batch,
                        learned=learned,
                        forward_integ=linear_forward,
                        backward_integ=linear_backward,
                        explicit_n_hat_fn=linear_explicit,
                    )
                else:
                    if ref_q_batch is not None and not use_anchor_stencils:
                        total_q_diag = total_q_diag + weight * mean_rollout_q_diagnostic(
                            ref_batch,
                            ref_q_batch,
                            learned=learned,
                            forward_integ=nonlinear_forward,
                            backward_integ=nonlinear_backward,
                            explicit_n_hat_fn=nonlinear_explicit,
                        )
                    if online_loss_backend_uses_projected_xv(str(loss_backend)):
                        regime_state = mean_projected_xv_history_loss(
                            ref_batch,
                            learned=learned,
                            forward_integ=nonlinear_forward,
                            backward_integ=nonlinear_backward,
                            explicit_n_hat_fn=nonlinear_explicit,
                        )
                        total_state = total_state + weight * regime_state
                        continue
                    if online_loss_backend_uses_posterior_rollout(str(loss_backend)):
                        _, regime_state, regime_field = mean_posterior_history_loss(
                            ref_batch,
                            learned=learned,
                            forward_integ=nonlinear_forward,
                            backward_integ=nonlinear_backward,
                            explicit_n_hat_fn=nonlinear_explicit,
                        )
                        total_state = total_state + weight * regime_state
                        total_field = total_field + weight * regime_field
                        continue
                    if online_loss_backend_uses_boundary_step(str(loss_backend)):
                        regime_state = mean_boundary_history_loss(
                            ref_batch,
                            learned=learned,
                            forward_integ=nonlinear_forward,
                            backward_integ=nonlinear_backward,
                            explicit_n_hat_fn=nonlinear_explicit,
                        )
                        total_state = total_state + weight * regime_state
                        continue
                    if online_loss_backend_uses_closure_q(str(loss_backend)):
                        assert ref_q_batch is not None
                        if use_anchor_stencils:
                            assert ref_anchor_index_batch is not None
                            regime_q = mean_compact_rollout_q_loss(
                                ref_batch,
                                ref_anchor_index_batch,
                                ref_q_batch,
                                learned=learned,
                                forward_integ=nonlinear_forward,
                                explicit_n_hat_fn=nonlinear_explicit,
                            )
                            total_q_diag = total_q_diag + weight * regime_q
                        else:
                            regime_q = mean_closure_history_loss(
                                ref_batch,
                                ref_q_batch,
                                learned=learned,
                                forward_integ=nonlinear_forward,
                                backward_integ=nonlinear_backward,
                                explicit_n_hat_fn=nonlinear_explicit,
                            )
                        total_q = total_q + weight * regime_q
                        continue
                    regime_state = mean_history_loss(
                        ref_batch,
                        learned=learned,
                        forward_integ=nonlinear_forward,
                        backward_integ=nonlinear_backward,
                        explicit_n_hat_fn=nonlinear_explicit,
                    )
                total_state = total_state + weight * regime_state

            zero = jnp.asarray(0.0, dtype=jnp.float64)
            if online_loss_backend_uses_posterior_rollout(str(loss_backend)):
                total_loss = (
                    jnp.asarray(float(posterior_state_weight), dtype=jnp.float64) * total_state
                    + jnp.asarray(float(posterior_field_weight), dtype=jnp.float64) * total_field
                )
            else:
                total_loss = total_q + total_state
            return total_loss, {
                "q": total_q,
                "state": total_state,
                "field": total_field,
                "dist": zero,
                "tail": zero,
                "neg": zero,
                "reg": zero,
                "q_diag": total_q_diag,
            }

        return exact_loss_fn_for_target

    target_loss_fns = {
        int(target_nv): make_loss_fn_for_target(int(target_nv))
        for target_nv in target_nvs
    }
    exact_target_loss_fns = {
        int(target_nv): make_exact_loss_fn_for_target(int(target_nv))
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
    loss_fn.rollout_horizon = int(rollout_horizon)  # type: ignore[attr-defined]
    loss_fn.rollout_anchor_samples = int(rollout_anchor_samples)  # type: ignore[attr-defined]
    loss_fn.rollout_direction = direction_mode  # type: ignore[attr-defined]
    loss_fn.randomize_rollout_anchors = online_loss_backend_uses_projected_coefficients(str(loss_backend))  # type: ignore[attr-defined]
    if len(target_nvs) == 1:
        loss_fn.exact_loss_fn = exact_target_loss_fns[default_target_nv]  # type: ignore[attr-defined]
    else:
        def exact_loss_fn(
            params: Dict[str, Array],
        ) -> Tuple[Array, Dict[str, Array]]:
            total = jnp.asarray(0.0, dtype=jnp.float64)
            aux_sum: Optional[Dict[str, Array]] = None
            for target_nv in target_nvs:
                loss, aux = exact_target_loss_fns[int(target_nv)](params)
                total = total + loss
                if aux_sum is None:
                    aux_sum = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in aux.items()}
                else:
                    aux_sum = {
                        key: aux_sum[key] + jnp.asarray(aux[key], dtype=jnp.float64)
                        for key in aux_sum
                    }
            assert aux_sum is not None
            scale = jnp.asarray(1.0 / float(len(target_nvs)), dtype=jnp.float64)
            return total * scale, {key: value * scale for key, value in aux_sum.items()}

        loss_fn.exact_loss_fn = exact_loss_fn  # type: ignore[attr-defined]
    return loss_fn, active_regimes


def make_online_trajectory_batch_loss(
    *,
    online_dataset: Dict[str, Dict[str, Dict[str, Array]]],
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
    n_low: int,
    context_mode: str,
    tail_start_fraction: float,
    loss_backend: str,
    lambda_E: float,
    lambda_dist: float,
    lambda_tail: float,
    lambda_neg: float,
    lambda_reg: float,
    online_v_probes: int,
    nonlinear_T: float,
    nonlinear_k0: float,
    poisson_sign: float,
    rollout_dealias_23: bool,
) -> Tuple[object, Sequence[str]]:
    active_regimes = tuple(
        regime
        for regime in train_regimes
        if regime in online_dataset
        and bool(online_dataset[regime].get("train"))
        and int(online_dataset[regime]["train"]["E_hat_ref"].shape[0]) > 0
    )
    weights = np.asarray([float(regime_weights[regime]) for regime in active_regimes], dtype=np.float64)
    weights = weights / np.sum(weights)
    weight_arr = jnp.asarray(weights, dtype=jnp.float64)
    target_nvs = tuple(int(v) for v in Nv_targets)
    target_nv_max = max(target_nvs)
    v_probe = jnp.linspace(float(teacher_vmin), float(teacher_vmax), int(online_v_probes), dtype=jnp.float64)
    eq_probe = maxwellian_equilibrium(v_probe)
    k_arr = FourierHermiteIMEX(
        Nx=int(teacher_Nx),
        Nv=int(target_nv_max),
        Lx=float(teacher_Lx),
        dt=float(teacher_dt),
        vth=1.0,
        dealias_23=bool(rollout_dealias_23),
        closure=None,
    ).k_arr
    linear_T = (
        float(online_dataset[REGIME_LINEAR]["train"]["times"].shape[1] - 1) * float(teacher_dt)
        if REGIME_LINEAR in online_dataset and online_dataset[REGIME_LINEAR].get("train")
        else float(nonlinear_T)
    )
    linear_configs = {
        int(target_nv): LinearLandauConfig(
            method="learned",
            Nv=int(target_nv),
            Nx=int(teacher_Nx),
            L=float(teacher_Lx),
            dt=float(teacher_dt),
            T=linear_T,
            poisson_sign=float(poisson_sign),
        )
        for target_nv in target_nvs
    }
    lambda_E_arr = jnp.asarray(float(lambda_E), dtype=jnp.float64)
    lambda_dist_arr = jnp.asarray(float(lambda_dist), dtype=jnp.float64)
    lambda_tail_arr = jnp.asarray(float(lambda_tail), dtype=jnp.float64)
    lambda_neg_arr = jnp.asarray(float(lambda_neg), dtype=jnp.float64)
    lambda_reg_arr = jnp.asarray(float(lambda_reg), dtype=jnp.float64)

    def make_loss_fn_for_target(target_nv: int):
        linear_config = linear_configs[int(target_nv)]

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
                Nv_targets=Nv_targets,
                train_regimes=train_regimes,
                teacher_backend=teacher_backend,
                teacher_Lx=teacher_Lx,
                teacher_Nx=teacher_Nx,
                teacher_Nv=teacher_Nv,
                teacher_vmin=teacher_vmin,
                teacher_vmax=teacher_vmax,
                teacher_dt=teacher_dt,
                teacher_proj_Nv=None,
                n_low=n_low,
                training_mode=ONLINE_TRAINING_MODE,
                train_objective="trajectory",
                context_mode=context_mode,
                rollout_horizon=0,
                tail_start_fraction=tail_start_fraction,
                loss_backend=loss_backend,
                lambda_q=0.0,
                lambda_E=lambda_E,
                lambda_dist=lambda_dist,
                lambda_tail=lambda_tail,
                lambda_neg=lambda_neg,
                lambda_reg=lambda_reg,
                online_v_probes=online_v_probes,
            )
            total_field = jnp.asarray(0.0, dtype=jnp.float64)
            total_dist = jnp.asarray(0.0, dtype=jnp.float64)
            total_tail = jnp.asarray(0.0, dtype=jnp.float64)
            total_neg = jnp.asarray(0.0, dtype=jnp.float64)

            for weight, regime in zip(weight_arr, active_regimes):
                batch = regime_batches[regime]
                if regime == REGIME_LINEAR:
                    field_terms, dist_terms, tail_terms, neg_terms = jax.vmap(
                        lambda perturbation_x, ref_e_hat, ref_delta_f: online_trajectory_loss_terms(
                            run_linear_landau_online_history(
                                learned,
                                config=linear_config,
                                perturbation_x=perturbation_x,
                            ),
                            k_arr=k_arr,
                            ref_E_hat=ref_e_hat,
                            ref_delta_f=ref_delta_f,
                            Nx=int(teacher_Nx),
                            v_probe=v_probe,
                            eq_probe=eq_probe,
                            tail_start_fraction=tail_start_fraction,
                            poisson_sign=float(poisson_sign),
                        )
                    )(
                        batch["perturbation_x"],
                        batch["E_hat_ref"],
                        batch["delta_f_ref"],
                    )
                else:
                    field_terms, dist_terms, tail_terms, neg_terms = jax.vmap(
                        lambda eps, ref_e_hat, ref_delta_f: online_trajectory_loss_terms(
                            run_nonlinear_landau_online_history(
                                learned,
                                Nx=int(teacher_Nx),
                                Nv=int(target_nv),
                                L=float(teacher_Lx),
                                dt=float(teacher_dt),
                                T=float(nonlinear_T),
                                eps=eps,
                                k0=float(nonlinear_k0),
                                dealias_23=bool(rollout_dealias_23),
                                poisson_sign=float(poisson_sign),
                            ),
                            k_arr=k_arr,
                            ref_E_hat=ref_e_hat,
                            ref_delta_f=ref_delta_f,
                            Nx=int(teacher_Nx),
                            v_probe=v_probe,
                            eq_probe=eq_probe,
                            tail_start_fraction=tail_start_fraction,
                            poisson_sign=float(poisson_sign),
                        )
                    )(
                        batch["eps"],
                        batch["E_hat_ref"],
                        batch["delta_f_ref"],
                    )
                total_field = total_field + weight * (lambda_E_arr * jnp.mean(field_terms))
                total_dist = total_dist + weight * (lambda_dist_arr * jnp.mean(dist_terms))
                total_tail = total_tail + weight * (lambda_tail_arr * jnp.mean(tail_terms))
                total_neg = total_neg + weight * (lambda_neg_arr * jnp.mean(neg_terms))

            reg_term = lambda_reg_arr * l2_regularization(params)
            total_loss = total_field + total_dist + total_tail + total_neg + reg_term
            return total_loss, {
                "q": jnp.asarray(0.0, dtype=jnp.float64),
                "state": jnp.asarray(0.0, dtype=jnp.float64),
                "field": total_field,
                "dist": total_dist,
                "tail": total_tail,
                "neg": total_neg,
                "reg": reg_term,
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
    return loss_fn, active_regimes


def _format_train_loss_log(
    *,
    epoch: int,
    epochs: int,
    history: Dict[str, np.ndarray],
    components: Sequence[str],
) -> str:
    parts = [
        f"[train] epoch {int(epoch) + 1:04d}/{int(epochs):04d}",
        f"loss={history['total'][int(epoch)]:.6e}",
    ]
    for key in components:
        if key == "total":
            continue
        if key in history:
            parts.append(f"{key}={history[key][int(epoch)]:.6e}")
    return " ".join(parts)


def online_training_log_components(
    *,
    train_objective: str,
    online_loss_backend: str,
) -> Tuple[str, ...]:
    if str(train_objective) == "trajectory":
        if online_loss_backend_uses_closure_q(str(online_loss_backend)):
            return ("q",)
        if str(online_loss_backend) in {
            ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR,
            ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BOUNDARY_STEP_BIDIR,
        }:
            return ("state",)
        if str(online_loss_backend) == ONLINE_LOSS_BACKEND_FOURIER_HERMITE_PROJECTED_XV_BIDIR:
            return ("state", "q_diag")
        if online_loss_backend_uses_posterior_rollout(str(online_loss_backend)):
            return ("state", "field")
        return ("field", "dist", "tail", "neg", "reg")
    if str(train_objective) == "trajectory_q_hybrid":
        return ("q", "field", "dist", "tail", "neg", "reg")
    return ()


def _online_group_history_length(group: Dict[str, Array]) -> int:
    for key, value in group.items():
        if str(key).startswith("a_hat_ref_nv"):
            arr = np.asarray(value)
            if arr.ndim < 2:
                raise ValueError(f"Expected online history array for {key!r}, got shape {arr.shape}")
            return int(arr.shape[1])
    raise ValueError("Online rollout batch is missing an a_hat_ref_nv* history array")


def _online_group_anchor_pool_size(group: Dict[str, Array]) -> int:
    for key, value in group.items():
        if str(key).startswith("a_hat_anchor_nv"):
            arr = np.asarray(value)
            if arr.ndim < 2:
                raise ValueError(f"Expected online anchor array for {key!r}, got shape {arr.shape}")
            return int(arr.shape[1])
    raise ValueError("Online rollout batch is missing an a_hat_anchor_nv* anchor array")


def _online_group_uses_anchor_pool(group: Dict[str, Array]) -> bool:
    return any(str(key).startswith("a_hat_anchor_nv") for key in group)


def train_with_online_trajectory_minibatch_loss(
    params: Dict[str, Array],
    online_dataset: Dict[str, Dict[str, Dict[str, Array]]],
    batch_loss_fn,
    *,
    active_regimes: Sequence[str],
    epochs: int,
    learning_rate: float,
    grad_clip: Optional[float],
    log_every: int,
    online_case_batch_size: int,
    steps_per_epoch: int,
    seed: int,
    log_components: Sequence[str] = (),
) -> Tuple[Dict[str, Array], Dict[str, np.ndarray]]:
    if int(online_case_batch_size) <= 0:
        raise ValueError("online_case_batch_size must be positive for online rollout training")
    if int(steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be positive for online rollout training")

    train_sizes = {
        regime: online_reference_num_cases(online_dataset[regime]["train"])
        for regime in active_regimes
    }
    state = adam_init(params)
    history = {
        key: np.zeros((int(epochs),), dtype=np.float64)
        for key in ("total", "q", "state", "field", "dist", "tail", "neg", "reg", "q_diag")
    }

    def make_train_step(target_batch_loss_fn):
        @jax.jit
        def train_step(
            current_params: Dict[str, Array],
            current_state: Dict[str, object],
            regime_batches: Dict[str, Dict[str, Array]],
        ) -> Tuple[Dict[str, Array], Dict[str, object], Dict[str, Array], Array]:
            (loss, aux), grads = jax.value_and_grad(target_batch_loss_fn, has_aux=True)(current_params, regime_batches)
            aux = dict(aux)
            aux["total"] = loss
            all_finite = _tree_all_finite(aux) & _tree_all_finite(grads)

            def apply_update(_: None) -> Tuple[Dict[str, Array], Dict[str, object]]:
                return adam_step(
                    current_params,
                    grads,
                    current_state,
                    learning_rate,
                    grad_clip=grad_clip,
                )

            def keep_state(_: None) -> Tuple[Dict[str, Array], Dict[str, object]]:
                return current_params, current_state

            next_params, next_state = jax.lax.cond(all_finite, apply_update, keep_state, operand=None)
            return next_params, next_state, aux, all_finite

        return train_step

    target_nvs = tuple(int(v) for v in getattr(batch_loss_fn, "target_nvs", ()))
    target_loss_fns = getattr(batch_loss_fn, "target_loss_fns", None)
    if target_nvs and isinstance(target_loss_fns, dict):
        train_steps = {
            int(target_nv): make_train_step(target_loss_fns[int(target_nv)])
            for target_nv in target_nvs
        }
    else:
        train_steps = {0: make_train_step(batch_loss_fn)}

    randomize_rollout_anchors = bool(getattr(batch_loss_fn, "randomize_rollout_anchors", False))
    rollout_horizon_attr = int(getattr(batch_loss_fn, "rollout_horizon", 0))
    rollout_anchor_samples_attr = int(getattr(batch_loss_fn, "rollout_anchor_samples", 0))
    rollout_direction_attr = str(
        getattr(batch_loss_fn, "rollout_direction", ONLINE_ROLLOUT_DIRECTION_BIDIR)
    )
    train_history_lengths = (
        {
            regime: _online_group_history_length(online_dataset[regime]["train"])
            for regime in active_regimes
            if not _online_group_uses_anchor_pool(online_dataset[regime]["train"])
        }
        if randomize_rollout_anchors
        else {}
    )
    train_anchor_pool_sizes = (
        {
            regime: _online_group_anchor_pool_size(online_dataset[regime]["train"])
            for regime in active_regimes
            if _online_group_uses_anchor_pool(online_dataset[regime]["train"])
        }
        if randomize_rollout_anchors
        else {}
    )
    rng = np.random.default_rng(int(seed))
    for epoch in range(int(epochs)):
        running = {
            key: jnp.asarray(0.0, dtype=jnp.float64)
            for key in ("total", "q", "state", "field", "dist", "tail", "neg", "reg", "q_diag")
        }
        for step_idx in range(int(steps_per_epoch)):
            regime_batches: Dict[str, Dict[str, Array]] = {}
            for regime in active_regimes:
                group = online_dataset[regime]["train"]
                size = train_sizes[regime]
                batch_n = int(min(online_case_batch_size, size))
                idx = rng.integers(0, size, size=batch_n, endpoint=False)
                regime_batches[regime] = {key: value[idx] for key, value in group.items()}
                if randomize_rollout_anchors:
                    if regime in train_anchor_pool_sizes:
                        regime_batches[regime][ROLLOUT_ANCHOR_INDICES_KEY] = _sample_rollout_anchor_pool_indices(
                            anchor_pool_size=train_anchor_pool_sizes[regime],
                            rollout_anchor_samples=rollout_anchor_samples_attr,
                            rng=rng,
                        )
                    else:
                        regime_batches[regime][ROLLOUT_ANCHOR_INDICES_KEY] = _sample_rollout_anchor_indices(
                            history_length=train_history_lengths[regime],
                            rollout_horizon=rollout_horizon_attr,
                            rollout_anchor_samples=rollout_anchor_samples_attr,
                            rollout_direction=rollout_direction_attr,
                            rng=rng,
                        )
            if target_nvs:
                target_nv = int(target_nvs[int(rng.integers(0, len(target_nvs)))])
                params, state, aux, all_finite = train_steps[target_nv](params, state, regime_batches)
            else:
                params, state, aux, all_finite = train_steps[0](params, state, regime_batches)
            if not bool(all_finite):
                raise FloatingPointError(
                    "online rollout produced non-finite loss/gradients at "
                    f"epoch {epoch + 1}, step {step_idx + 1}; "
                    "reduce TRAIN_LR, TRAIN_LAMBDA_TAIL, TRAIN_GRAD_CLIP, "
                    "or TRAIN_STEPS_PER_EPOCH."
                )
            for key in running:
                if key in aux:
                    running[key] = running[key] + aux[key]
        for key in history:
            history[key][epoch] = float(running[key] / float(steps_per_epoch))
        if epoch == 0 or (epoch + 1) % max(int(log_every), 1) == 0 or epoch + 1 == int(epochs):
            print(_format_train_loss_log(epoch=epoch, epochs=epochs, history=history, components=log_components))
    return params, history


def train_with_exact_monotone_online_loss(
    params: Dict[str, Array],
    exact_loss_fn,
    *,
    epochs: int,
    learning_rate: float,
    grad_clip: Optional[float],
    log_every: int,
    steps_per_epoch: int,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 8,
    min_learning_rate: float = 1e-8,
) -> Tuple[Dict[str, Array], Dict[str, np.ndarray]]:
    if int(epochs) <= 0:
        return params, {
            key: np.zeros((0,), dtype=np.float64)
            for key in ("total", "state", "field", "dist", "tail", "neg", "reg")
        }
    if int(steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be positive for exact online rollout training")

    state = adam_init(params)
    history = {
        key: np.zeros((int(epochs),), dtype=np.float64)
        for key in ("total", "state", "field", "dist", "tail", "neg", "reg")
    }

    @jax.jit
    def evaluate_loss(
        current_params: Dict[str, Array],
    ) -> Tuple[Array, Dict[str, Array]]:
        loss, aux = exact_loss_fn(current_params)
        aux = dict(aux)
        aux["total"] = loss
        return loss, aux

    @jax.jit
    def loss_and_grad(
        current_params: Dict[str, Array],
    ) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
        (loss, aux), grads = jax.value_and_grad(exact_loss_fn, has_aux=True)(current_params)
        aux = dict(aux)
        aux["total"] = loss
        return loss, aux, grads

    @jax.jit
    def propose_step(
        current_params: Dict[str, Array],
        current_state: Dict[str, object],
        grads: Dict[str, Array],
        step_lr: Array,
    ) -> Tuple[Dict[str, Array], Dict[str, object]]:
        return adam_step(
            current_params,
            grads,
            current_state,
            step_lr,
            grad_clip=grad_clip,
        )

    current_loss, current_aux = evaluate_loss(params)
    if not bool(_tree_all_finite(current_aux)):
        raise FloatingPointError("exact online rollout loss is non-finite before the first optimization step")

    for epoch in range(int(epochs)):
        for step_idx in range(int(steps_per_epoch)):
            loss_before, aux_before, grads = loss_and_grad(params)
            if not bool(_tree_all_finite(aux_before) & _tree_all_finite(grads)):
                raise FloatingPointError(
                    "exact online rollout produced non-finite loss/gradients at "
                    f"epoch {epoch + 1}, step {step_idx + 1}"
                )
            accepted = False
            step_lr = float(learning_rate)
            loss_before_f = float(loss_before)
            tolerance = 1e-12 * max(1.0, abs(loss_before_f))
            for _ in range(int(max_backtracks) + 1):
                proposal_params, proposal_state = propose_step(
                    params,
                    state,
                    grads,
                    jnp.asarray(step_lr, dtype=jnp.float64),
                )
                proposal_loss, proposal_aux = evaluate_loss(proposal_params)
                if bool(_tree_all_finite(proposal_aux)):
                    proposal_loss_f = float(proposal_loss)
                    if proposal_loss_f <= loss_before_f + tolerance:
                        params = proposal_params
                        state = proposal_state
                        current_loss = proposal_loss
                        current_aux = proposal_aux
                        accepted = True
                        break
                step_lr *= float(backtrack_factor)
                if step_lr < float(min_learning_rate):
                    break
            if not accepted:
                current_loss = loss_before
                current_aux = aux_before
                break
        for key in history:
            history[key][epoch] = float(current_aux[key])
        if epoch == 0 or (epoch + 1) % max(int(log_every), 1) == 0 or epoch + 1 == int(epochs):
            print(
                f"[train] epoch {epoch + 1:04d}/{int(epochs):04d} "
                f"loss={history['total'][epoch]:.6e} "
                f"state={history['state'][epoch]:.6e} "
                f"field={history['field'][epoch]:.6e} "
                f"dist={history['dist'][epoch]:.6e} "
                f"tail={history['tail'][epoch]:.6e} "
                f"neg={history['neg'][epoch]:.6e} "
                f"reg={history['reg'][epoch]:.6e}"
            )
    return params, history


def train_with_online_hybrid_minibatch_loss(
    params: Dict[str, Array],
    prepared: Dict[str, Dict[str, Array]],
    online_dataset: Dict[str, Dict[str, Dict[str, Array]]],
    batch_loss_fn,
    *,
    active_regimes: Sequence[str],
    epochs: int,
    learning_rate: float,
    grad_clip: Optional[float],
    log_every: int,
    batch_size: int,
    online_case_batch_size: int,
    steps_per_epoch: int,
    seed: int,
    log_components: Sequence[str] = (),
) -> Tuple[Dict[str, Array], Dict[str, np.ndarray]]:
    if int(online_case_batch_size) <= 0:
        raise ValueError("online_case_batch_size must be positive for online hybrid training")
    if int(steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be positive for online hybrid training")

    q_train_sizes = {
        regime: int(prepared[regime]["train_inputs"].shape[0])
        for regime in active_regimes
    }
    online_train_sizes = {
        regime: int(online_dataset[regime]["train"]["E_hat_ref"].shape[0])
        for regime in active_regimes
    }
    state = adam_init(params)
    history = {
        key: np.zeros((int(epochs),), dtype=np.float64)
        for key in ("total", "q", "state", "field", "dist", "tail", "neg", "reg")
    }

    def make_train_step(target_batch_loss_fn):
        @jax.jit
        def train_step(
            current_params: Dict[str, Array],
            current_state: Dict[str, object],
            q_batches: Dict[str, Dict[str, Array]],
            regime_batches: Dict[str, Dict[str, Array]],
        ) -> Tuple[Dict[str, Array], Dict[str, object], Dict[str, Array], Array]:
            (loss, aux), grads = jax.value_and_grad(target_batch_loss_fn, has_aux=True)(
                current_params,
                q_batches,
                regime_batches,
            )
            aux = dict(aux)
            aux["total"] = loss
            all_finite = _tree_all_finite(aux) & _tree_all_finite(grads)

            def apply_update(_: None) -> Tuple[Dict[str, Array], Dict[str, object]]:
                return adam_step(
                    current_params,
                    grads,
                    current_state,
                    learning_rate,
                    grad_clip=grad_clip,
                )

            def keep_state(_: None) -> Tuple[Dict[str, Array], Dict[str, object]]:
                return current_params, current_state

            next_params, next_state = jax.lax.cond(all_finite, apply_update, keep_state, operand=None)
            return next_params, next_state, aux, all_finite

        return train_step

    target_nvs = tuple(int(v) for v in getattr(batch_loss_fn, "target_nvs", ()))
    target_loss_fns = getattr(batch_loss_fn, "target_loss_fns", None)
    if target_nvs and isinstance(target_loss_fns, dict):
        train_steps = {
            int(target_nv): make_train_step(target_loss_fns[int(target_nv)])
            for target_nv in target_nvs
        }
    else:
        train_steps = {0: make_train_step(batch_loss_fn)}

    rng = np.random.default_rng(int(seed))
    use_full_q_batch = int(batch_size) <= 0
    for epoch in range(int(epochs)):
        running = {
            key: jnp.asarray(0.0, dtype=jnp.float64)
            for key in ("total", "q", "state", "field", "dist", "tail", "neg", "reg")
        }
        for step_idx in range(int(steps_per_epoch)):
            q_batches: Dict[str, Dict[str, Array]] = {}
            regime_batches: Dict[str, Dict[str, Array]] = {}
            for regime in active_regimes:
                q_size = q_train_sizes[regime]
                if use_full_q_batch:
                    idx_q = np.arange(q_size, dtype=np.int64)
                else:
                    batch_n = int(min(batch_size, q_size))
                    idx_q = rng.integers(0, q_size, size=batch_n, endpoint=False)
                q_batches[regime] = {
                    "inputs": prepared[regime]["train_inputs"][idx_q],
                    "targets_std": prepared[regime]["train_targets_std"][idx_q],
                }

                online_group = online_dataset[regime]["train"]
                online_size = online_train_sizes[regime]
                batch_n = int(min(online_case_batch_size, online_size))
                idx_online = rng.integers(0, online_size, size=batch_n, endpoint=False)
                regime_batches[regime] = {key: value[idx_online] for key, value in online_group.items()}
            if target_nvs:
                target_nv = int(target_nvs[int(rng.integers(0, len(target_nvs)))])
                params, state, aux, all_finite = train_steps[target_nv](params, state, q_batches, regime_batches)
            else:
                params, state, aux, all_finite = train_steps[0](params, state, q_batches, regime_batches)
            if not bool(all_finite):
                raise FloatingPointError(
                    "online hybrid rollout produced non-finite loss/gradients at "
                    f"epoch {epoch + 1}, step {step_idx + 1}; "
                    "reduce TRAIN_LR, TRAIN_LAMBDA_Q, TRAIN_LAMBDA_TAIL, "
                    "TRAIN_GRAD_CLIP, or TRAIN_STEPS_PER_EPOCH."
                )
            for key in running:
                running[key] = running[key] + aux[key]
        for key in history:
            history[key][epoch] = float(running[key] / float(steps_per_epoch))
        if epoch == 0 or (epoch + 1) % max(int(log_every), 1) == 0 or epoch + 1 == int(epochs):
            print(_format_train_loss_log(epoch=epoch, epochs=epochs, history=history, components=log_components))
    return params, history


def evaluate_regime_metrics(
    learned: LearnedInterfaceClosure,
    prepared: Dict[str, Dict[str, Array]],
) -> Dict[str, np.ndarray]:
    metrics: Dict[str, np.ndarray] = {}
    for regime, arrays in prepared.items():
        pred = np.asarray(learned.predict_q_components(arrays["val_inputs"]), dtype=np.float64)
        target = np.asarray(arrays["val_targets"], dtype=np.float64)
        if target.shape[0] == 0:
            mse = float("nan")
            rel_l2 = float("nan")
        else:
            mse = float(np.mean((pred - target) ** 2))
            denom = max(float(np.linalg.norm(target)), 1e-30)
            rel_l2 = float(np.linalg.norm(pred - target) / denom)
        metrics[f"val_q_mse_{regime}"] = np.array([mse], dtype=np.float64)
        metrics[f"val_q_rel_l2_{regime}"] = np.array([rel_l2], dtype=np.float64)
        metrics[f"val_num_samples_{regime}"] = np.array([target.shape[0]], dtype=np.int32)
    return metrics


def _load_init_checkpoint_for_online_trajectory(
    init_checkpoint: Path,
    *,
    Nm: int,
    hidden_width: int,
    res_blocks: int,
    Nv_targets: Sequence[int],
    context_mode: str,
) -> Tuple[Dict[str, Array], Dict[str, np.ndarray], float, float]:
    learned = load_learned_interface_closure_npz(init_checkpoint)
    expected_targets = tuple(int(v) for v in Nv_targets)
    actual_targets = tuple(int(v) for v in learned.Nv_targets)
    if actual_targets != expected_targets:
        raise ValueError(
            f"--init-checkpoint Nv_targets={actual_targets} does not match requested Nv-targets={expected_targets}"
        )
    if int(learned.Nm) != int(Nm):
        raise ValueError(f"--init-checkpoint Nm={int(learned.Nm)} does not match requested Nm={int(Nm)}")
    if int(learned.hidden_width) != int(hidden_width):
        raise ValueError(
            f"--init-checkpoint hidden_width={int(learned.hidden_width)} does not match requested hidden_width={int(hidden_width)}"
        )
    if int(learned.res_blocks) != int(res_blocks):
        raise ValueError(
            f"--init-checkpoint res_blocks={int(learned.res_blocks)} does not match requested res_blocks={int(res_blocks)}"
        )
    if str(learned.context_mode) != str(context_mode):
        raise ValueError(
            f"--init-checkpoint context_mode={learned.context_mode!r} does not match requested context_mode={context_mode!r}"
        )
    params = {
        key: jnp.asarray(value, dtype=jnp.float64)
        for key, value in learned.params.items()
    }
    stats = {
        "input_mean": np.asarray(learned.input_mean, dtype=np.float64),
        "input_std": np.asarray(learned.input_std, dtype=np.float64),
        "target_mean": np.asarray(learned.target_mean, dtype=np.float64),
        "target_std": np.asarray(learned.target_std, dtype=np.float64),
    }
    return params, stats, float(learned.k_scale), float(learned.nv_scale)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a shared learned interface closure from a selectable Landau teacher")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional learned-closure checkpoint used to initialize online trajectory training.",
    )
    parser.add_argument("--dataset-cache", type=Path, default=None)
    parser.add_argument("--loss-plot", type=Path, default=None)
    parser.add_argument("--build-dataset-only", action="store_true")
    parser.add_argument("--allow-dataset-cache-nv-superset", action="store_true")
    parser.add_argument("--per-target-projection-orders", action="store_true")
    parser.add_argument("--Nv-targets", type=str, default="6,8,10,12,20,40,80,160,300")
    parser.add_argument("--Nm", type=int, default=6)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--k-scale", type=float, default=None)
    parser.add_argument("--nv-scale", type=float, default=None)
    parser.add_argument("--n-low", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--training-mode", type=str, default=OFFLINE_TRAINING_MODE, choices=(OFFLINE_TRAINING_MODE, ONLINE_TRAINING_MODE))
    parser.add_argument("--train-objective", type=str, default="q_only", choices=("q_only", "trajectory", "trajectory_q_hybrid"))
    parser.add_argument("--context-mode", type=str, default="none", choices=("none", "lag1_delta"))
    parser.add_argument("--tail-start-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--lambda-q", type=float, default=1.0)
    parser.add_argument("--lambda-E", type=float, default=0.5)
    parser.add_argument("--lambda-dist", type=float, default=1.0)
    parser.add_argument("--lambda-tail", type=float, default=0.05)
    parser.add_argument("--lambda-neg", type=float, default=0.05)
    parser.add_argument("--lambda-reg", type=float, default=1e-6)
    parser.add_argument("--rollout-horizon", type=int, default=0)
    parser.add_argument("--rollout-anchor-samples", type=int, default=0)
    parser.add_argument("--rollout-anchor-pool-size", type=int, default=0)
    parser.add_argument("--rollout-direction", type=str, default=ONLINE_ROLLOUT_DIRECTION_BIDIR, choices=ALL_ONLINE_ROLLOUT_DIRECTIONS)
    parser.add_argument("--rollout-dealias-23", action="store_true")
    parser.add_argument("--online-loss-backend", type=str, default=ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1)
    parser.add_argument("--projected-xv-tail-window", type=int, default=0)
    parser.add_argument("--projected-xv-metric", type=str, default=PROJECTED_XV_METRIC_PHYSICAL_L2, choices=ALL_PROJECTED_XV_METRICS)
    parser.add_argument("--posterior-state-weight", type=float, default=0.25)
    parser.add_argument("--posterior-field-weight", type=float, default=1.0)
    parser.add_argument("--online-v-probes", type=int, default=64)
    parser.add_argument("--online-case-batch-size", type=int, default=1)
    parser.add_argument("--online-reference-cache", type=Path, default=None)
    parser.add_argument("--regimes", type=str, default="linear_landau,nonlinear_landau_weak,nonlinear_landau_strong")
    parser.add_argument("--weight-linear", type=float, default=1.0)
    parser.add_argument("--weight-weak", type=float, default=1.0)
    parser.add_argument("--weight-strong", type=float, default=1.0)

    parser.add_argument("--teacher-backend", type=str, default=GRID_CUBIC_SPLINE_TEACHER_BACKEND, choices=ALL_TEACHER_BACKENDS)
    parser.add_argument("--teacher-Nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", type=int, default=512)
    parser.add_argument("--teacher-L", type=float, default=4.0 * math.pi)
    parser.add_argument("--teacher-vmin", type=float, default=-8.0)
    parser.add_argument("--teacher-vmax", type=float, default=8.0)
    parser.add_argument("--teacher-dt", type=float, default=1e-2)
    parser.add_argument("--teacher-poisson-sign", type=float, default=1.0)
    parser.add_argument("--teacher-proj-Nv", type=int, default=None)

    parser.add_argument("--linear-T", type=float, default=20.0)
    parser.add_argument("--linear-eps", type=float, default=1e-2)
    parser.add_argument("--linear-modes", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--linear-num-samples", type=int, default=8)
    parser.add_argument("--linear-seed", type=int, default=0)
    parser.add_argument("--linear-history-stride", type=int, default=2)

    parser.add_argument("--nonlinear-T", type=float, default=20.0)
    parser.add_argument("--nonlinear-k0", type=float, default=0.5)
    parser.add_argument("--nonlinear-history-stride", type=int, default=20)
    parser.add_argument("--weak-eps", type=str, default="0.05,0.1")
    parser.add_argument("--strong-eps", type=str, default="0.25,0.5")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    print_jax_runtime_summary(jax, context="training")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    training_mode = str(args.training_mode)
    if args.checkpoint is None and not bool(args.build_dataset_only):
        raise ValueError("--checkpoint is required unless --build-dataset-only is set")
    if training_mode == OFFLINE_TRAINING_MODE and bool(args.build_dataset_only) and args.dataset_cache is None:
        raise ValueError("--build-dataset-only requires --dataset-cache so the generated dataset can be reused")

    Nv_targets = parse_int_tuple(args.Nv_targets)
    if not Nv_targets:
        raise ValueError("At least one target Nv must be provided")
    if any(int(Nv) < int(args.Nm) for Nv in Nv_targets):
        raise ValueError(
            f"Invalid training configuration: Nm={int(args.Nm)} requires every target Nv to satisfy Nv >= Nm. "
            f"Received Nv-targets={Nv_targets}."
        )
    linear_modes = parse_float_tuple(args.linear_modes)
    weak_eps = parse_float_tuple(args.weak_eps)
    strong_eps = parse_float_tuple(args.strong_eps)
    regimes = tuple(regime for regime in parse_str_tuple(args.regimes) if regime in ALL_REGIMES)
    if not regimes:
        raise ValueError("At least one valid training regime must be selected")

    teacher_backend = normalize_teacher_backend_name(args.teacher_backend)
    online_loss_backend = str(args.online_loss_backend)
    teacher_proj_Nv: Optional[int] = None
    online_reference_cache = args.dataset_cache
    if training_mode == ONLINE_TRAINING_MODE:
        if args.init_checkpoint is not None and args.train_objective != "trajectory":
            raise ValueError("--init-checkpoint is only supported for online_rollout trajectory training")
        if args.init_checkpoint is not None and bool(args.build_dataset_only):
            raise ValueError("--init-checkpoint is not used with --build-dataset-only")
        if args.init_checkpoint is not None and not args.init_checkpoint.exists():
            raise ValueError(f"--init-checkpoint does not exist: {args.init_checkpoint}")
        if bool(args.allow_dataset_cache_nv_superset):
            raise ValueError("online_rollout does not support --allow-dataset-cache-nv-superset")
        if bool(args.per_target_projection_orders):
            raise ValueError("online_rollout does not support --per-target-projection-orders")
        if teacher_backend != GRID_CUBIC_SPLINE_TEACHER_BACKEND:
            raise ValueError("online_rollout only supports teacher_backend=grid_cubic_spline")
        if args.train_objective == "trajectory":
            if bool(args.build_dataset_only) and args.dataset_cache is None:
                raise ValueError("online_rollout --build-dataset-only requires --dataset-cache")
            if args.teacher_proj_Nv is not None:
                raise ValueError("online_rollout trajectory does not use --teacher-proj-Nv")
            if args.online_reference_cache is not None:
                raise ValueError("online_rollout trajectory does not use --online-reference-cache")
            if online_loss_backend_uses_projected_coefficients(online_loss_backend):
                if int(args.rollout_horizon) <= 0:
                    raise ValueError(f"{online_loss_backend} requires --rollout-horizon > 0")
        elif args.train_objective == "trajectory_q_hybrid":
            online_reference_cache = args.online_reference_cache
            if online_reference_cache is None:
                raise ValueError("online_rollout trajectory_q_hybrid requires --online-reference-cache")
            teacher_proj_Nv = int(args.teacher_proj_Nv) if args.teacher_proj_Nv is not None else max(Nv_targets) + 1
            if teacher_proj_Nv <= max(Nv_targets):
                raise ValueError("teacher-proj-Nv must exceed every target Nv")
            if float(args.lambda_q) <= 0.0 and not bool(args.build_dataset_only):
                raise ValueError("trajectory_q_hybrid requires --lambda-q > 0")
            if online_loss_backend != ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1:
                raise ValueError("trajectory_q_hybrid only supports online_loss_backend=field_distribution_v1")
            if int(args.batch_size) <= 0 and not bool(args.build_dataset_only):
                raise ValueError(
                    "trajectory_q_hybrid requires --batch-size > 0 so the offline q-loss component "
                    "does not compile the full q dataset into the JAX train step"
                )
        else:
            raise ValueError(
                "online_rollout requires --train-objective trajectory or trajectory_q_hybrid"
            )
        if online_loss_backend not in ALL_ONLINE_LOSS_BACKENDS:
            raise ValueError(
                f"Unsupported online loss backend {args.online_loss_backend!r}; "
                f"expected one of {ALL_ONLINE_LOSS_BACKENDS!r}"
            )
        if (
            online_loss_backend == ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1
            and int(args.online_v_probes) <= 0
        ):
            raise ValueError("online_rollout requires --online-v-probes > 0")
        if online_loss_backend_uses_projected_coefficients(online_loss_backend):
            if int(args.online_v_probes) != 0:
                raise ValueError(f"{online_loss_backend} requires --online-v-probes 0")
            if int(args.rollout_anchor_samples) < 0:
                raise ValueError(f"{online_loss_backend} requires --rollout-anchor-samples >= 0")
            if online_loss_backend_uses_rollout_qloss(online_loss_backend) and args.rollout_direction == ONLINE_ROLLOUT_DIRECTION_FORWARD:
                if int(args.rollout_anchor_pool_size) <= 0:
                    raise ValueError(f"{online_loss_backend} forward compact cache requires --rollout-anchor-pool-size > 0")
            if online_loss_backend_uses_posterior_rollout(online_loss_backend):
                if float(args.posterior_state_weight) < 0.0 or float(args.posterior_field_weight) < 0.0:
                    raise ValueError("posterior rollout weights must be nonnegative")
                if float(args.posterior_state_weight) + float(args.posterior_field_weight) <= 0.0:
                    raise ValueError("at least one posterior rollout weight must be positive")
            if (
                not bool(args.build_dataset_only)
                and any(
                float(value) != 0.0
                for value in (
                    args.lambda_E,
                    args.lambda_dist,
                    args.lambda_tail,
                    args.lambda_neg,
                    args.lambda_reg,
                )
                )
            ):
                raise ValueError(f"{online_loss_backend} requires lambda_E=lambda_dist=lambda_tail=lambda_neg=lambda_reg=0")
        if int(args.online_case_batch_size) <= 0 and not bool(args.build_dataset_only):
            raise ValueError("online_rollout requires --online-case-batch-size > 0")
        if (
            online_loss_backend == ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1
            and float(args.lambda_E) <= 0.0
            and float(args.lambda_dist) <= 0.0
            and not bool(args.build_dataset_only)
        ):
            raise ValueError("online_rollout requires lambda_E > 0 or lambda_dist > 0")
    else:
        if args.init_checkpoint is not None:
            raise ValueError("--init-checkpoint is only supported for online_rollout trajectory training")
        if args.train_objective in {"trajectory", "trajectory_q_hybrid"}:
            raise ValueError(f"{args.train_objective} objective is only supported with --training-mode online_rollout")
        if teacher_backend == GRID_CUBIC_SPLINE_TEACHER_BACKEND:
            teacher_proj_Nv = int(args.teacher_proj_Nv) if args.teacher_proj_Nv is not None else max(Nv_targets) + 1
            if teacher_proj_Nv <= max(Nv_targets):
                raise ValueError("teacher-proj-Nv must exceed every target Nv")
        elif teacher_backend == HIGHER_ORDER_HERMITE_TEACHER_BACKEND:
            if bool(args.per_target_projection_orders):
                raise ValueError("higher_order_hermite does not support --per-target-projection-orders")
            if args.teacher_proj_Nv is not None:
                raise ValueError("higher_order_hermite does not use --teacher-proj-Nv")
            if int(args.teacher_Nv) <= max(Nv_targets):
                raise ValueError("higher_order_hermite requires teacher-Nv to exceed every target Nv")
        else:
            raise ValueError(f"Unsupported teacher backend: {teacher_backend!r}")

    regime_weights = {
        REGIME_LINEAR: float(args.weight_linear),
        REGIME_WEAK: float(args.weight_weak),
        REGIME_STRONG: float(args.weight_strong),
    }
    val_metrics: Dict[str, np.ndarray] = {}
    online_component_history: Optional[Dict[str, np.ndarray]] = None

    if training_mode == OFFLINE_TRAINING_MODE:
        dataset_base = build_mixed_landau_dataset(
            dataset_cache=args.dataset_cache,
            regimes=regimes,
            teacher_backend=teacher_backend,
            teacher_Nx=args.teacher_Nx,
            teacher_Nv=args.teacher_Nv,
            teacher_L=args.teacher_L,
            teacher_vmin=args.teacher_vmin,
            teacher_vmax=args.teacher_vmax,
            teacher_dt=args.teacher_dt,
            teacher_proj_Nv=teacher_proj_Nv,
            linear_T=args.linear_T,
            linear_eps=args.linear_eps,
            linear_modes=linear_modes,
            linear_num_samples=args.linear_num_samples,
            linear_seed=args.linear_seed,
            linear_poisson_sign=args.teacher_poisson_sign,
            linear_history_stride=args.linear_history_stride,
            nonlinear_T=args.nonlinear_T,
            nonlinear_k0=args.nonlinear_k0,
            nonlinear_poisson_sign=args.teacher_poisson_sign,
            nonlinear_history_stride=args.nonlinear_history_stride,
            weak_eps=weak_eps,
            strong_eps=strong_eps,
            Nv_targets=Nv_targets,
            Nm=args.Nm,
            val_fraction=args.val_fraction,
            n_low=args.n_low,
            context_mode=args.context_mode,
            allow_cached_nv_superset=bool(args.allow_dataset_cache_nv_superset),
            per_target_projection_orders=bool(args.per_target_projection_orders) if teacher_backend == GRID_CUBIC_SPLINE_TEACHER_BACKEND else False,
        )
        if bool(args.build_dataset_only):
            cache_msg = f"Saved shared dataset cache to {args.dataset_cache}" if args.dataset_cache is not None else "Built dataset in memory"
            print(cache_msg)
            for regime, arrays in dataset_base.items():
                print(f"[data] {regime}: {arrays['train_inputs_base'].shape[0]} training samples cached")
            return

        k_scale = float(args.k_scale) if args.k_scale is not None else choose_k_scale(dataset_base, Nm=args.Nm)
        nv_scale = float(args.nv_scale) if args.nv_scale is not None else choose_nv_scale(dataset_base, Nm=args.Nm)
        prepared, stats = prepare_training_dataset(
            dataset_base,
            Nm=args.Nm,
            k_scale=k_scale,
            nv_scale=nv_scale,
            context_mode=args.context_mode,
        )
        for regime, count in summarize_dataset(prepared).items():
            print(f"[data] {regime}: {count} training samples")

        input_dim = int(stats["input_mean"].shape[0])
        params = init_interface_closure_params(
            jax.random.PRNGKey(args.seed),
            input_dim=input_dim,
            hidden_width=int(args.hidden_width),
            res_blocks=int(args.res_blocks),
        )
        if int(args.batch_size) > 0:
            batch_loss_fn, active_regimes = make_regime_balanced_batch_loss(
                regime_weights=regime_weights,
                Nm=args.Nm,
                k_scale=k_scale,
                nv_scale=nv_scale,
                stats=stats,
                hidden_width=args.hidden_width,
                res_blocks=args.res_blocks,
                Nv_targets=Nv_targets,
                train_regimes=regimes,
                teacher_backend=teacher_backend,
                teacher_Lx=args.teacher_L,
                teacher_Nx=args.teacher_Nx,
                teacher_Nv=args.teacher_Nv,
                teacher_vmin=args.teacher_vmin,
                teacher_vmax=args.teacher_vmax,
                teacher_dt=args.teacher_dt,
                teacher_proj_Nv=teacher_proj_Nv,
                n_low=args.n_low,
                context_mode=args.context_mode,
            )
            train_sizes = [int(prepared[regime]["train_inputs"].shape[0]) for regime in active_regimes]
            steps_per_epoch = int(args.steps_per_epoch)
            if steps_per_epoch <= 0:
                steps_per_epoch = max(1, math.ceil(max(train_sizes) / float(args.batch_size)))
            params, loss_history = train_with_minibatch_loss(
                params,
                prepared,
                batch_loss_fn,
                active_regimes=active_regimes,
                epochs=args.epochs,
                learning_rate=args.lr,
                grad_clip=args.grad_clip,
                log_every=args.log_every,
                batch_size=args.batch_size,
                steps_per_epoch=steps_per_epoch,
                seed=args.seed,
            )
        else:
            loss_fn = make_regime_balanced_loss(
                prepared,
                regime_weights=regime_weights,
                Nm=args.Nm,
                k_scale=k_scale,
                nv_scale=nv_scale,
                stats=stats,
                hidden_width=args.hidden_width,
                res_blocks=args.res_blocks,
                Nv_targets=Nv_targets,
                train_regimes=regimes,
                teacher_backend=teacher_backend,
                teacher_Lx=args.teacher_L,
                teacher_Nx=args.teacher_Nx,
                teacher_Nv=args.teacher_Nv,
                teacher_vmin=args.teacher_vmin,
                teacher_vmax=args.teacher_vmax,
                teacher_dt=args.teacher_dt,
                teacher_proj_Nv=teacher_proj_Nv,
                n_low=args.n_low,
                context_mode=args.context_mode,
            )
            params, loss_history = train_with_loss(
                params,
                loss_fn,
                epochs=args.epochs,
                learning_rate=args.lr,
                grad_clip=args.grad_clip,
                log_every=args.log_every,
            )

        learned = build_learned_interface_closure(
            params=params,
            Nm=args.Nm,
            k_scale=k_scale,
            nv_scale=nv_scale,
            stats=stats,
            hidden_width=args.hidden_width,
            res_blocks=args.res_blocks,
            Nv_targets=Nv_targets,
            train_regimes=regimes,
            teacher_backend=teacher_backend,
            teacher_Lx=args.teacher_L,
            teacher_Nx=args.teacher_Nx,
            teacher_Nv=args.teacher_Nv,
            teacher_vmin=args.teacher_vmin,
            teacher_vmax=args.teacher_vmax,
            teacher_dt=args.teacher_dt,
            teacher_proj_Nv=teacher_proj_Nv,
            n_low=args.n_low,
            training_mode=OFFLINE_TRAINING_MODE,
            train_objective=args.train_objective,
            context_mode=args.context_mode,
            rollout_horizon=0,
            tail_start_fraction=args.tail_start_fraction,
            lambda_q=args.lambda_q,
            lambda_E=0.0,
            lambda_tail=0.0,
            lambda_reg=0.0,
            stability_loss_definition=None,
        )
        val_metrics = evaluate_regime_metrics(learned, prepared)
    else:
        online_dataset, _ = build_online_reference_dataset(
            dataset_cache=online_reference_cache,
            regimes=regimes,
            teacher_Nx=args.teacher_Nx,
            teacher_Nv=args.teacher_Nv,
            teacher_L=args.teacher_L,
            teacher_vmin=args.teacher_vmin,
            teacher_vmax=args.teacher_vmax,
            teacher_dt=args.teacher_dt,
            linear_T=args.linear_T,
            linear_eps=args.linear_eps,
            linear_modes=linear_modes,
            linear_num_samples=args.linear_num_samples,
            linear_seed=args.linear_seed,
            linear_poisson_sign=args.teacher_poisson_sign,
            nonlinear_T=args.nonlinear_T,
            nonlinear_k0=args.nonlinear_k0,
            nonlinear_poisson_sign=args.teacher_poisson_sign,
            weak_eps=weak_eps,
            strong_eps=strong_eps,
            val_fraction=args.val_fraction,
            online_v_probes=args.online_v_probes,
            online_loss_backend=online_loss_backend,
            Nv_targets=Nv_targets,
            rollout_horizon=args.rollout_horizon,
            rollout_anchor_samples=args.rollout_anchor_samples,
            rollout_anchor_pool_size=args.rollout_anchor_pool_size,
            rollout_direction=args.rollout_direction,
        )
        for regime in regimes:
            if regime not in online_dataset:
                continue
            train_count = online_reference_num_cases(online_dataset[regime]["train"]) if online_dataset[regime].get("train") else 0
            val_count = online_reference_num_cases(online_dataset[regime]["val"]) if online_dataset[regime].get("val") else 0
            print(f"[data] {regime}: train={train_count} episodes val={val_count} episodes")

        if args.train_objective == "trajectory_q_hybrid":
            dataset_base = build_mixed_landau_dataset(
                dataset_cache=args.dataset_cache,
                regimes=regimes,
                teacher_backend=teacher_backend,
                teacher_Nx=args.teacher_Nx,
                teacher_Nv=args.teacher_Nv,
                teacher_L=args.teacher_L,
                teacher_vmin=args.teacher_vmin,
                teacher_vmax=args.teacher_vmax,
                teacher_dt=args.teacher_dt,
                teacher_proj_Nv=teacher_proj_Nv,
                linear_T=args.linear_T,
                linear_eps=args.linear_eps,
                linear_modes=linear_modes,
                linear_num_samples=args.linear_num_samples,
                linear_seed=args.linear_seed,
                linear_poisson_sign=args.teacher_poisson_sign,
                linear_history_stride=args.linear_history_stride,
                nonlinear_T=args.nonlinear_T,
                nonlinear_k0=args.nonlinear_k0,
                nonlinear_poisson_sign=args.teacher_poisson_sign,
                nonlinear_history_stride=args.nonlinear_history_stride,
                weak_eps=weak_eps,
                strong_eps=strong_eps,
                Nv_targets=Nv_targets,
                Nm=args.Nm,
                val_fraction=args.val_fraction,
                n_low=args.n_low,
                context_mode=args.context_mode,
                allow_cached_nv_superset=False,
                per_target_projection_orders=False,
            )
            if bool(args.build_dataset_only):
                if args.dataset_cache is not None:
                    print(f"Saved hybrid q dataset cache to {args.dataset_cache}")
                print(f"Prepared hybrid online reference dataset cache at {online_reference_cache}")
                return

            k_scale = float(args.k_scale) if args.k_scale is not None else choose_k_scale(dataset_base, Nm=args.Nm)
            nv_scale = float(args.nv_scale) if args.nv_scale is not None else choose_nv_scale(dataset_base, Nm=args.Nm)
            prepared, stats = prepare_training_dataset(
                dataset_base,
                Nm=args.Nm,
                k_scale=k_scale,
                nv_scale=nv_scale,
                context_mode=args.context_mode,
            )
            for regime, count in summarize_dataset(prepared).items():
                print(f"[data] {regime}: {count} q samples")

            params = init_online_rollout_params(
                jax.random.PRNGKey(args.seed),
                input_dim=int(stats["input_mean"].shape[0]),
                hidden_width=int(args.hidden_width),
                res_blocks=int(args.res_blocks),
                target_mean=stats["target_mean"],
                target_std=stats["target_std"],
            )
            batch_loss_fn, active_regimes = make_online_hybrid_batch_loss(
                prepared=prepared,
                online_dataset=online_dataset,
                regime_weights=regime_weights,
                Nm=args.Nm,
                k_scale=k_scale,
                nv_scale=nv_scale,
                stats=stats,
                hidden_width=args.hidden_width,
                res_blocks=args.res_blocks,
                Nv_targets=Nv_targets,
                train_regimes=regimes,
                teacher_backend=teacher_backend,
                teacher_Lx=args.teacher_L,
                teacher_Nx=args.teacher_Nx,
                teacher_Nv=args.teacher_Nv,
                teacher_vmin=args.teacher_vmin,
                teacher_vmax=args.teacher_vmax,
                teacher_dt=args.teacher_dt,
                teacher_proj_Nv=int(teacher_proj_Nv),
                n_low=args.n_low,
                context_mode=args.context_mode,
                tail_start_fraction=args.tail_start_fraction,
                loss_backend=args.online_loss_backend,
                lambda_q=args.lambda_q,
                lambda_E=args.lambda_E,
                lambda_dist=args.lambda_dist,
                lambda_tail=args.lambda_tail,
                lambda_neg=args.lambda_neg,
                lambda_reg=args.lambda_reg,
                online_v_probes=args.online_v_probes,
                nonlinear_T=args.nonlinear_T,
                nonlinear_k0=args.nonlinear_k0,
                poisson_sign=args.teacher_poisson_sign,
                rollout_dealias_23=bool(args.rollout_dealias_23),
            )
            train_sizes = [online_reference_num_cases(online_dataset[regime]["train"]) for regime in active_regimes]
            steps_per_epoch = int(args.steps_per_epoch)
            if steps_per_epoch <= 0:
                steps_per_epoch = max(1, math.ceil(max(train_sizes) / float(args.online_case_batch_size)))
            params, online_component_history = train_with_online_hybrid_minibatch_loss(
                params,
                prepared,
                online_dataset,
                batch_loss_fn,
                active_regimes=active_regimes,
                epochs=args.epochs,
                learning_rate=args.lr,
                grad_clip=args.grad_clip,
                log_every=args.log_every,
                batch_size=args.batch_size,
                online_case_batch_size=args.online_case_batch_size,
                steps_per_epoch=steps_per_epoch,
                seed=args.seed,
                log_components=online_training_log_components(
                    train_objective=args.train_objective,
                    online_loss_backend=online_loss_backend,
                ),
            )
            loss_history = online_component_history["total"]
            val_metrics = evaluate_regime_metrics(
                build_learned_interface_closure(
                    params=params,
                    Nm=args.Nm,
                    k_scale=k_scale,
                    nv_scale=nv_scale,
                    stats=stats,
                    hidden_width=args.hidden_width,
                    res_blocks=args.res_blocks,
                    Nv_targets=Nv_targets,
                    train_regimes=regimes,
                    teacher_backend=teacher_backend,
                    teacher_Lx=args.teacher_L,
                    teacher_Nx=args.teacher_Nx,
                    teacher_Nv=args.teacher_Nv,
                    teacher_vmin=args.teacher_vmin,
                    teacher_vmax=args.teacher_vmax,
                    teacher_dt=args.teacher_dt,
                    teacher_proj_Nv=teacher_proj_Nv,
                    n_low=args.n_low,
                    training_mode=ONLINE_TRAINING_MODE,
                    train_objective="trajectory_q_hybrid",
                    context_mode=args.context_mode,
                    rollout_horizon=0,
                    tail_start_fraction=args.tail_start_fraction,
                    loss_backend=args.online_loss_backend,
                    lambda_q=args.lambda_q,
                    lambda_E=args.lambda_E,
                    lambda_dist=args.lambda_dist,
                    lambda_tail=args.lambda_tail,
                    lambda_neg=args.lambda_neg,
                    lambda_reg=args.lambda_reg,
                    online_v_probes=args.online_v_probes,
                    stability_loss_definition=ONLINE_HYBRID_LOSS_DEFINITION,
                ),
                prepared,
            )
        else:
            if bool(args.build_dataset_only):
                print(f"Prepared online reference dataset cache at {online_reference_cache}")
                return

            target_nv_max = max(int(v) for v in Nv_targets)
            integ = FourierHermiteIMEX(
                Nx=int(args.teacher_Nx),
                Nv=int(target_nv_max),
                Lx=float(args.teacher_L),
                dt=float(args.teacher_dt),
                vth=1.0,
                dealias_23=bool(args.rollout_dealias_23),
                closure=None,
            )
            k_scale = float(args.k_scale) if args.k_scale is not None else float(jnp.max(jnp.asarray(integ.k_arr[1:], dtype=jnp.float64)))
            nv_scale = float(args.nv_scale) if args.nv_scale is not None else float(target_nv_max)
            if args.init_checkpoint is not None:
                params, stats, k_scale, nv_scale = _load_init_checkpoint_for_online_trajectory(
                    args.init_checkpoint,
                    Nm=args.Nm,
                    hidden_width=args.hidden_width,
                    res_blocks=args.res_blocks,
                    Nv_targets=Nv_targets,
                    context_mode=args.context_mode,
                )
                if args.k_scale is not None and not np.isclose(float(args.k_scale), float(k_scale)):
                    raise ValueError("--k-scale must match --init-checkpoint k_scale when warm-starting online training")
                if args.nv_scale is not None and not np.isclose(float(args.nv_scale), float(nv_scale)):
                    raise ValueError("--nv-scale must match --init-checkpoint nv_scale when warm-starting online training")
                print(f"[train] initialized online trajectory parameters from {args.init_checkpoint}")
            else:
                if online_loss_backend_uses_projected_coefficients(str(online_loss_backend)):
                    stats = build_online_q_training_stats_from_reference(
                        online_dataset,
                        active_regimes=regimes,
                        Nv_targets=Nv_targets,
                        Nm=args.Nm,
                        k_arr=np.asarray(integ.k_arr, dtype=np.float64),
                        k_scale=k_scale,
                        nv_scale=nv_scale,
                        n_low=args.n_low,
                        context_mode=args.context_mode,
                        require_q_targets=online_loss_backend_has_reference_q_targets(str(online_loss_backend)),
                    )
                    print(
                        "[data] online closure normalization: "
                        f"input_std_max={float(np.max(stats['input_std'])):.3e} "
                        f"target_std={np.asarray(stats['target_std'], dtype=np.float64)}"
                    )
                else:
                    stats = build_identity_training_stats(Nm=args.Nm, context_mode=args.context_mode)
                params = init_online_rollout_params(
                    jax.random.PRNGKey(args.seed),
                    input_dim=int(stats["input_mean"].shape[0]),
                    hidden_width=int(args.hidden_width),
                    res_blocks=int(args.res_blocks),
                    target_mean=stats["target_mean"],
                    target_std=stats["target_std"],
                )
            if online_loss_backend_uses_projected_coefficients(online_loss_backend):
                batch_loss_fn, active_regimes = make_online_fourier_hermite_bidir_batch_loss(
                    online_dataset=online_dataset,
                    regime_weights=regime_weights,
                    Nm=args.Nm,
                    k_scale=k_scale,
                    nv_scale=nv_scale,
                    stats=stats,
                    hidden_width=args.hidden_width,
                    res_blocks=args.res_blocks,
                    Nv_targets=Nv_targets,
                    train_regimes=regimes,
                    teacher_backend=teacher_backend,
                    teacher_Lx=args.teacher_L,
                    teacher_Nx=args.teacher_Nx,
                    teacher_Nv=args.teacher_Nv,
                    teacher_vmin=args.teacher_vmin,
                    teacher_vmax=args.teacher_vmax,
                    teacher_dt=args.teacher_dt,
                    n_low=args.n_low,
                    context_mode=args.context_mode,
                    rollout_horizon=args.rollout_horizon,
                    rollout_anchor_samples=args.rollout_anchor_samples,
                    projected_xv_tail_window=args.projected_xv_tail_window,
                    projected_xv_metric=args.projected_xv_metric,
                    rollout_direction=args.rollout_direction,
                    loss_backend=online_loss_backend,
                    poisson_sign=args.teacher_poisson_sign,
                    rollout_dealias_23=bool(args.rollout_dealias_23),
                    posterior_state_weight=args.posterior_state_weight,
                    posterior_field_weight=args.posterior_field_weight,
                )
            else:
                batch_loss_fn, active_regimes = make_online_trajectory_batch_loss(
                    online_dataset=online_dataset,
                    regime_weights=regime_weights,
                    Nm=args.Nm,
                    k_scale=k_scale,
                    nv_scale=nv_scale,
                    stats=stats,
                    hidden_width=args.hidden_width,
                    res_blocks=args.res_blocks,
                    Nv_targets=Nv_targets,
                    train_regimes=regimes,
                    teacher_backend=teacher_backend,
                    teacher_Lx=args.teacher_L,
                    teacher_Nx=args.teacher_Nx,
                    teacher_Nv=args.teacher_Nv,
                    teacher_vmin=args.teacher_vmin,
                    teacher_vmax=args.teacher_vmax,
                    teacher_dt=args.teacher_dt,
                    n_low=args.n_low,
                    context_mode=args.context_mode,
                    tail_start_fraction=args.tail_start_fraction,
                    loss_backend=args.online_loss_backend,
                    lambda_E=args.lambda_E,
                    lambda_dist=args.lambda_dist,
                    lambda_tail=args.lambda_tail,
                    lambda_neg=args.lambda_neg,
                    lambda_reg=args.lambda_reg,
                    online_v_probes=args.online_v_probes,
                    nonlinear_T=args.nonlinear_T,
                    nonlinear_k0=args.nonlinear_k0,
                    poisson_sign=args.teacher_poisson_sign,
                    rollout_dealias_23=bool(args.rollout_dealias_23),
                )
            train_sizes = [online_reference_num_cases(online_dataset[regime]["train"]) for regime in active_regimes]
            steps_per_epoch = int(args.steps_per_epoch)
            if steps_per_epoch <= 0:
                steps_per_epoch = max(1, math.ceil(max(train_sizes) / float(args.online_case_batch_size)))
            params, online_component_history = train_with_online_trajectory_minibatch_loss(
                params,
                online_dataset,
                batch_loss_fn,
                active_regimes=active_regimes,
                epochs=args.epochs,
                learning_rate=args.lr,
                grad_clip=args.grad_clip,
                log_every=args.log_every,
                online_case_batch_size=args.online_case_batch_size,
                steps_per_epoch=steps_per_epoch,
                seed=args.seed,
                log_components=online_training_log_components(
                    train_objective=args.train_objective,
                    online_loss_backend=online_loss_backend,
                ),
            )
            loss_history = online_component_history["total"]

        for regime in regimes:
            if regime in online_dataset and online_dataset[regime].get("val"):
                val_metrics[f"val_num_cases_{regime}"] = np.array(
                    [online_reference_num_cases(online_dataset[regime]["val"])],
                    dtype=np.int32,
                )

        learned = build_learned_interface_closure(
            params=params,
            Nm=args.Nm,
            k_scale=k_scale,
            nv_scale=nv_scale,
            stats=stats,
            hidden_width=args.hidden_width,
            res_blocks=args.res_blocks,
            Nv_targets=Nv_targets,
            train_regimes=regimes,
            teacher_backend=teacher_backend,
            teacher_Lx=args.teacher_L,
            teacher_Nx=args.teacher_Nx,
            teacher_Nv=args.teacher_Nv,
            teacher_vmin=args.teacher_vmin,
            teacher_vmax=args.teacher_vmax,
            teacher_dt=args.teacher_dt,
            teacher_proj_Nv=teacher_proj_Nv,
            n_low=args.n_low,
            training_mode=ONLINE_TRAINING_MODE,
            train_objective=args.train_objective,
            context_mode=args.context_mode,
            rollout_horizon=(
                int(args.rollout_horizon)
                if (
                    args.train_objective == "trajectory"
                    and online_loss_backend_uses_projected_coefficients(online_loss_backend)
                )
                else 0
            ),
            rollout_anchor_samples=(
                int(args.rollout_anchor_samples)
                if (
                    args.train_objective == "trajectory"
                    and online_loss_backend_uses_projected_coefficients(online_loss_backend)
                )
                else 0
            ),
            tail_start_fraction=args.tail_start_fraction,
            loss_backend=args.online_loss_backend,
            lambda_q=args.lambda_q if args.train_objective == "trajectory_q_hybrid" else 0.0,
            lambda_E=(
                args.lambda_E
                if online_loss_backend == ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1
                else 0.0
            ),
            lambda_dist=(
                args.lambda_dist
                if online_loss_backend == ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1
                else 0.0
            ),
            lambda_tail=(
                args.lambda_tail
                if online_loss_backend == ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1
                else 0.0
            ),
            lambda_neg=(
                args.lambda_neg
                if online_loss_backend == ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1
                else 0.0
            ),
            lambda_reg=(
                args.lambda_reg
                if online_loss_backend == ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1
                else 0.0
            ),
            online_v_probes=(
                args.online_v_probes
                if online_loss_backend == ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1
                else 0
            ),
            stability_loss_definition=(
                ONLINE_HYBRID_LOSS_DEFINITION if args.train_objective == "trajectory_q_hybrid" else None
            ),
        )

    assert args.checkpoint is not None
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_learned_interface_closure_npz(args.checkpoint, learned)

    metrics_path = args.checkpoint.with_suffix(".metrics.npz")
    used_lambda_q = args.lambda_q if args.train_objective in {"q_only", "trajectory_q_hybrid"} else 0.0
    field_backend_active = (
        training_mode == ONLINE_TRAINING_MODE
        and online_loss_backend == ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1
    )
    used_lambda_E = (
        args.lambda_E
        if field_backend_active and args.train_objective in {"trajectory", "trajectory_q_hybrid"}
        else 0.0
    )
    used_lambda_dist = args.lambda_dist if field_backend_active else 0.0
    used_lambda_tail = (
        args.lambda_tail
        if field_backend_active and args.train_objective in {"trajectory", "trajectory_q_hybrid"}
        else 0.0
    )
    used_lambda_neg = args.lambda_neg if field_backend_active else 0.0
    used_lambda_reg = (
        args.lambda_reg
        if field_backend_active and args.train_objective in {"trajectory", "trajectory_q_hybrid"}
        else 0.0
    )
    metrics_payload: Dict[str, np.ndarray] = {
        "train_loss": np.asarray(loss_history, dtype=np.float64),
        "Nm": np.array([args.Nm], dtype=np.int32),
        "hidden_width": np.array([args.hidden_width], dtype=np.int32),
        "res_blocks": np.array([args.res_blocks], dtype=np.int32),
        "k_scale": np.array([k_scale], dtype=np.float64),
        "nv_scale": np.array([nv_scale], dtype=np.float64),
        "Nv_targets": np.asarray(Nv_targets, dtype=np.int32),
        "regimes": np.asarray(regimes, dtype=np.str_),
        "weight_linear": np.array([args.weight_linear], dtype=np.float64),
        "weight_weak": np.array([args.weight_weak], dtype=np.float64),
        "weight_strong": np.array([args.weight_strong], dtype=np.float64),
        "input_mean": np.asarray(stats["input_mean"], dtype=np.float64),
        "input_std": np.asarray(stats["input_std"], dtype=np.float64),
        "target_mean": np.asarray(stats["target_mean"], dtype=np.float64),
        "target_std": np.asarray(stats["target_std"], dtype=np.float64),
        "teacher_backend": np.array([str(teacher_backend)], dtype=np.str_),
        "teacher_Lx": np.array([args.teacher_L], dtype=np.float64),
        "teacher_Nx": np.array([args.teacher_Nx], dtype=np.int32),
        "teacher_Nv": np.array([args.teacher_Nv], dtype=np.int32),
        "teacher_vmin": np.array([args.teacher_vmin], dtype=np.float64),
        "teacher_vmax": np.array([args.teacher_vmax], dtype=np.float64),
        "teacher_dt": np.array([args.teacher_dt], dtype=np.float64),
        "n_low": np.array([args.n_low], dtype=np.int32),
        "training_mode": np.array([training_mode], dtype=np.str_),
        "train_objective": np.array([args.train_objective], dtype=np.str_),
        "context_mode": np.array([args.context_mode], dtype=np.str_),
        "rollout_horizon": np.array(
            [
                int(args.rollout_horizon)
                if (
                    training_mode == ONLINE_TRAINING_MODE
                    and online_loss_backend_uses_projected_coefficients(online_loss_backend)
                    and args.train_objective == "trajectory"
                )
                else 0
            ],
            dtype=np.int32,
        ),
        "rollout_anchor_samples": np.array(
            [
                int(args.rollout_anchor_samples)
                if (
                    training_mode == ONLINE_TRAINING_MODE
                    and online_loss_backend_uses_projected_coefficients(online_loss_backend)
                    and args.train_objective == "trajectory"
                )
                else 0
            ],
            dtype=np.int32,
        ),
        "rollout_direction": np.array(
            [
                str(args.rollout_direction)
                if (
                    training_mode == ONLINE_TRAINING_MODE
                    and online_loss_backend_uses_projected_coefficients(online_loss_backend)
                    and args.train_objective == "trajectory"
                )
                else ""
            ],
            dtype=np.str_,
        ),
        "tail_start_fraction": np.array([args.tail_start_fraction], dtype=np.float64),
        "projected_xv_tail_window": np.array([args.projected_xv_tail_window], dtype=np.int32),
        "projected_xv_metric": np.array([args.projected_xv_metric], dtype=np.str_),
        "loss_backend": np.array(
            [] if training_mode == OFFLINE_TRAINING_MODE else [args.online_loss_backend],
            dtype=np.str_,
        ),
        "lambda_q": np.array([used_lambda_q], dtype=np.float64),
        "lambda_E": np.array([used_lambda_E], dtype=np.float64),
        "lambda_dist": np.array([used_lambda_dist], dtype=np.float64),
        "lambda_tail": np.array([used_lambda_tail], dtype=np.float64),
        "lambda_neg": np.array([used_lambda_neg], dtype=np.float64),
        "lambda_reg": np.array([used_lambda_reg], dtype=np.float64),
        "online_v_probes": np.array(
            [
                args.online_v_probes
                if field_backend_active
                else 0
            ],
            dtype=np.int32,
        ),
    }
    if teacher_proj_Nv is not None:
        metrics_payload["teacher_proj_Nv"] = np.array([teacher_proj_Nv], dtype=np.int32)
    if training_mode == ONLINE_TRAINING_MODE:
        assert online_component_history is not None
        logged_components = online_training_log_components(
            train_objective=args.train_objective,
            online_loss_backend=online_loss_backend,
        )
        for component in logged_components:
            if component in online_component_history:
                metrics_payload[f"train_loss_{component}"] = np.asarray(
                    online_component_history[component],
                    dtype=np.float64,
                )
                if component == "q_diag":
                    metrics_payload["train_q_rel_mse_diag"] = np.asarray(
                        online_component_history[component],
                        dtype=np.float64,
                    )
        if args.train_objective == "trajectory_q_hybrid":
            metrics_payload["stability_loss_definition"] = np.array([ONLINE_HYBRID_LOSS_DEFINITION], dtype=np.str_)
    metrics_payload.update(val_metrics)
    np.savez(metrics_path, **metrics_payload)

    loss_plot_path = args.loss_plot if args.loss_plot is not None else args.checkpoint.with_suffix(".loss.png")
    save_training_loss_plot(
        np.asarray(loss_history, dtype=np.float64),
        loss_plot_path,
        val_metrics=val_metrics,
        train_objective=args.train_objective,
        loss_backend=(args.online_loss_backend if training_mode == ONLINE_TRAINING_MODE else None),
    )
    q_diag_plot_path: Optional[Path] = None
    if (
        training_mode == ONLINE_TRAINING_MODE
        and online_component_history is not None
        and "q_diag" in online_component_history
    ):
        q_diag_history = np.asarray(online_component_history["q_diag"], dtype=np.float64)
        if q_diag_history.size and np.any(np.isfinite(q_diag_history)) and np.nanmax(q_diag_history) > 0.0:
            q_diag_plot_path = Path(loss_plot_path).with_suffix(".qdiag.png")
            save_training_loss_q_diagnostic_plot(
                np.asarray(loss_history, dtype=np.float64),
                q_diag_history,
                q_diag_plot_path,
                loss_backend=args.online_loss_backend,
            )

    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved loss plot to {loss_plot_path}")
    if q_diag_plot_path is not None:
        print(f"Saved q diagnostic plot to {q_diag_plot_path}")
    for key in sorted(val_metrics):
        print(f"{key}: {float(np.asarray(val_metrics[key]).reshape(-1)[0]):.6e}")


if __name__ == "__main__":
    main()
