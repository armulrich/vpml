"""
Train a shared learned interface closure for Landau-family runs using a selectable teacher.

The grid-cubic-spline teacher is a full Vlasov-Poisson semi-Lagrangian solve on a fine
(x, v) grid with JAX cubic spline interpolation. Teacher snapshots are projected onto
the Fourier-Hermite basis, and the learned target is

    q_k^* = -i k v_th sqrt(Nv) C_{Nv,k}^{HR}.

The higher-order-Hermite teacher uses direct Hermite-space rollouts to produce the same
coefficient histories without a projection step. The current trainer preserves the q-only
path and also supports a stability-aware mode that augments the supervised loss with
short-horizon field and excess-tail penalties.
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
    e_hat_from_rho_hat,
    e_hat_history_from_a_hat_history,
    init_interface_closure_params,
    irfft_x,
    learned_boundary_flux_hat,
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
    extract_interface_rollout_windows_from_coeff_history,
    gaussian_pdf,
    hermite_dual_basis_scaled,
    normalize_density_on_grid,
    project_distribution_snapshot_to_fourier_hermite,
    run_semilagrangian_vlasov_poisson,
)
from vpml.visualization.training import save_training_loss_plot

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass

REGIME_LINEAR = "linear_landau"
REGIME_WEAK = "nonlinear_landau_weak"
REGIME_STRONG = "nonlinear_landau_strong"
ALL_REGIMES = (REGIME_LINEAR, REGIME_WEAK, REGIME_STRONG)
CACHE_FORMAT = "landau_interface_dataset_teacher_v6"
STABILITY_LOSS_DEFINITION = "window_hybrid_v1"
ONLINE_TRAINING_MODE = "online_rollout"
OFFLINE_TRAINING_MODE = "offline_rollout"
ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1 = "field_distribution_v1"
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
    rollout_horizon: int = 0,
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
        "rollout_horizon": np.array([int(rollout_horizon)], dtype=np.int32),
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


def append_rollout_windows(
    accum: Dict[str, Dict[str, object]],
    regime: str,
    split: str,
    windows_by_nv: Dict[int, Dict[str, np.ndarray]],
) -> None:
    bucket = accum[regime][f"{split}_rollout"]
    assert isinstance(bucket, dict)
    for Nv, payload in windows_by_nv.items():
        target = bucket.setdefault(
            int(Nv),
            {
                "prev_state": [],
                "curr_state": [],
                "future_state": [],
                "future_E_hat": [],
            },
        )
        for key in ("prev_state", "curr_state", "future_state", "future_E_hat"):
            target[key].append(np.asarray(payload[key]))


def finalize_regime_arrays(accum: Dict[str, Dict[str, list]]) -> Dict[str, Dict[str, np.ndarray]]:
    dataset: Dict[str, Dict[str, np.ndarray]] = {}
    for regime, payload in accum.items():
        if not payload["train_inputs_base"]:
            continue
        train_rollout = {}
        val_rollout = {}
        train_rollout_bucket = payload["train_rollout"]
        val_rollout_bucket = payload["val_rollout"]
        assert isinstance(train_rollout_bucket, dict)
        assert isinstance(val_rollout_bucket, dict)
        for Nv, arrays in train_rollout_bucket.items():
            train_rollout[int(Nv)] = {
                key: np.concatenate(arrays[key], axis=0).astype(np.complex128)
                if arrays[key]
                else np.zeros((0,), dtype=np.complex128)
                for key in ("prev_state", "curr_state", "future_state", "future_E_hat")
            }
        for Nv, arrays in val_rollout_bucket.items():
            val_rollout[int(Nv)] = {
                key: np.concatenate(arrays[key], axis=0).astype(np.complex128)
                if arrays[key]
                else np.zeros((0,), dtype=np.complex128)
                for key in ("prev_state", "curr_state", "future_state", "future_E_hat")
            }
        dataset[regime] = {
            "train_inputs_base": np.concatenate(payload["train_inputs_base"], axis=0).astype(np.float64),
            "train_targets": np.concatenate(payload["train_targets"], axis=0).astype(np.float64),
            "val_inputs_base": np.concatenate(payload["val_inputs_base"], axis=0).astype(np.float64),
            "val_targets": np.concatenate(payload["val_targets"], axis=0).astype(np.float64),
            "train_rollout": train_rollout,
            "val_rollout": val_rollout,
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
    rollout_horizon: int,
) -> None:
    train_hist, val_hist = split_history_train_val(coeff_hist, val_fraction)
    use_rollout_windows = int(rollout_horizon) > 0
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
        if use_rollout_windows:
            append_rollout_windows(
                accum,
                regime_name,
                split,
                extract_interface_rollout_windows_from_coeff_history(
                    hist,
                    Nv_targets=Nv_targets,
                    Nm=Nm,
                    k_arr=k_arr,
                    vth=1.0,
                    rollout_horizon=rollout_horizon,
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
    rollout_horizon: int,
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
    use_rollout_windows = int(rollout_horizon) > 0

    accum = {
        REGIME_LINEAR: {
            "train_inputs_base": [],
            "train_targets": [],
            "val_inputs_base": [],
            "val_targets": [],
            "train_rollout": {},
            "val_rollout": {},
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
                    rollout_horizon=rollout_horizon,
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
            rollout_horizon=rollout_horizon,
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
    rollout_horizon: int,
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
            "train_rollout": {},
            "val_rollout": {},
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
                    rollout_horizon=rollout_horizon,
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
            rollout_horizon=rollout_horizon,
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
            train_rollout: Dict[int, Dict[str, np.ndarray]] = {}
            val_rollout: Dict[int, Dict[str, np.ndarray]] = {}
            rollout_prefix = f"{regime}_rollout_nv"
            nv_suffixes = []
            for name in data.files:
                if not name.startswith(rollout_prefix):
                    continue
                remainder = name[len(rollout_prefix):]
                nv_text = remainder.split("_", 1)[0]
                if nv_text:
                    nv_suffixes.append(int(nv_text))
            nv_suffixes = sorted(set(nv_suffixes))
            for Nv in nv_suffixes:
                train_rollout[Nv] = {
                    "prev_state": np.asarray(data[f"{regime}_rollout_nv{Nv}_train_prev_state"], dtype=np.complex128),
                    "curr_state": np.asarray(data[f"{regime}_rollout_nv{Nv}_train_curr_state"], dtype=np.complex128),
                    "future_state": np.asarray(data[f"{regime}_rollout_nv{Nv}_train_future_state"], dtype=np.complex128),
                    "future_E_hat": np.asarray(data[f"{regime}_rollout_nv{Nv}_train_future_E_hat"], dtype=np.complex128),
                }
                val_rollout[Nv] = {
                    "prev_state": np.asarray(data[f"{regime}_rollout_nv{Nv}_val_prev_state"], dtype=np.complex128),
                    "curr_state": np.asarray(data[f"{regime}_rollout_nv{Nv}_val_curr_state"], dtype=np.complex128),
                    "future_state": np.asarray(data[f"{regime}_rollout_nv{Nv}_val_future_state"], dtype=np.complex128),
                    "future_E_hat": np.asarray(data[f"{regime}_rollout_nv{Nv}_val_future_E_hat"], dtype=np.complex128),
                }
            dataset[regime] = {
                "train_inputs_base": np.asarray(data[f"{regime}_train_inputs_base"], dtype=np.float64),
                "train_targets": np.asarray(data[f"{regime}_train_targets"], dtype=np.float64),
                "val_inputs_base": np.asarray(data[f"{regime}_val_inputs_base"], dtype=np.float64),
                "val_targets": np.asarray(data[f"{regime}_val_targets"], dtype=np.float64),
                "train_rollout": train_rollout,
                "val_rollout": val_rollout,
            }
        return dataset


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
        for split in ("train", "val"):
            rollout = arrays.get(f"{split}_rollout", {})
            for Nv, window_payload in rollout.items():
                prefix = f"{regime}_rollout_nv{int(Nv)}_{split}"
                payload[f"{prefix}_prev_state"] = np.asarray(window_payload["prev_state"], dtype=np.complex128)
                payload[f"{prefix}_curr_state"] = np.asarray(window_payload["curr_state"], dtype=np.complex128)
                payload[f"{prefix}_future_state"] = np.asarray(window_payload["future_state"], dtype=np.complex128)
                payload[f"{prefix}_future_E_hat"] = np.asarray(window_payload["future_E_hat"], dtype=np.complex128)
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
            "train_rollout": {
                int(Nv): {
                    key: np.asarray(payload[key])
                    for key in ("prev_state", "curr_state", "future_state", "future_E_hat")
                }
                for Nv, payload in arrays.get("train_rollout", {}).items()
                if int(Nv) in set(int(v) for v in nv_targets.tolist())
            },
            "val_rollout": {
                int(Nv): {
                    key: np.asarray(payload[key])
                    for key in ("prev_state", "curr_state", "future_state", "future_E_hat")
                }
                for Nv, payload in arrays.get("val_rollout", {}).items()
                if int(Nv) in set(int(v) for v in nv_targets.tolist())
            },
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
    rollout_horizon: int = 0,
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
        rollout_horizon=rollout_horizon,
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
            rollout_horizon=rollout_horizon,
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
            rollout_horizon=rollout_horizon,
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
            rollout_horizon=rollout_horizon,
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
            "train_rollout": arrays.get("train_rollout", {}),
            "val_rollout": arrays.get("val_rollout", {}),
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
            "train_rollout": {
                int(Nv): {
                    "prev_state": jnp.asarray(payload["prev_state"], dtype=jnp.complex128),
                    "curr_state": jnp.asarray(payload["curr_state"], dtype=jnp.complex128),
                    "future_state": jnp.asarray(payload["future_state"], dtype=jnp.complex128),
                    "future_E_hat": jnp.asarray(payload["future_E_hat"], dtype=jnp.complex128),
                }
                for Nv, payload in arrays.get("train_rollout", {}).items()
            },
            "val_rollout": {
                int(Nv): {
                    "prev_state": jnp.asarray(payload["prev_state"], dtype=jnp.complex128),
                    "curr_state": jnp.asarray(payload["curr_state"], dtype=jnp.complex128),
                    "future_state": jnp.asarray(payload["future_state"], dtype=jnp.complex128),
                    "future_E_hat": jnp.asarray(payload["future_E_hat"], dtype=jnp.complex128),
                }
                for Nv, payload in arrays.get("val_rollout", {}).items()
            },
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


def init_online_rollout_params(
    key: Array,
    *,
    input_dim: int,
    hidden_width: int,
    res_blocks: int,
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


def build_online_reference_dataset(
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
) -> Tuple[Dict[str, Dict[str, Dict[str, Array]]], Array]:
    v_probe = jnp.linspace(float(teacher_vmin), float(teacher_vmax), int(online_v_probes), dtype=jnp.float64)
    dataset: Dict[str, Dict[str, Dict[str, Array]]] = {}

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


def build_rollout_weights(H: int) -> np.ndarray:
    if int(H) <= 0:
        raise ValueError("rollout_horizon must be positive")
    return np.asarray([1.0 / (2 ** idx) for idx in range(int(H))], dtype=np.float64)


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


def rollout_window_teacher_scales(
    future_state: Array,
    future_E_hat: Array,
    *,
    rollout_weights: Array,
    tail_start_fraction: float,
) -> Tuple[Array, Array]:
    future_state = jnp.asarray(future_state, dtype=jnp.complex128)
    future_E_hat = jnp.asarray(future_E_hat, dtype=jnp.complex128)
    H = int(future_state.shape[0])
    Nv = int(future_state.shape[1])
    k_weights = _rollout_k_weights(int(future_state.shape[2]))
    tail_start = min(int(math.ceil(float(tail_start_fraction) * float(Nv))), int(Nv) - 1)
    tail_weights = jnp.asarray(tail_mode_weights(Nv, tail_start_fraction), dtype=jnp.float64)
    weights = jnp.asarray(rollout_weights, dtype=jnp.float64)[:H]

    field_energy_by_step = jnp.sum(
        k_weights[None, 1:] * (jnp.abs(future_E_hat[:, 1:]) ** 2),
        axis=1,
    )
    field_true_total = jnp.sum(weights * field_energy_by_step)

    true_tail = jnp.abs(future_state[:, tail_start:, :]) ** 2
    alpha = tail_weights[:, None] * k_weights[None, :]
    tail_energy_by_step = jnp.sum(alpha[None, :, :] * (true_tail ** 2), axis=(1, 2))
    tail_true_total = jnp.sum(weights * tail_energy_by_step)
    return field_true_total, tail_true_total


def rollout_window_hybrid_loss_from_sequences(
    pred_future_state: Array,
    pred_future_E_hat: Array,
    future_state: Array,
    future_E_hat: Array,
    *,
    rollout_weights: Array,
    tail_start_fraction: float,
    field_ref_scale: Array,
    tail_ref_scale: Array,
) -> Tuple[Array, Array]:
    pred_future_state = jnp.asarray(pred_future_state, dtype=jnp.complex128)
    pred_future_E_hat = jnp.asarray(pred_future_E_hat, dtype=jnp.complex128)
    future_state = jnp.asarray(future_state, dtype=jnp.complex128)
    future_E_hat = jnp.asarray(future_E_hat, dtype=jnp.complex128)

    H = int(future_state.shape[0])
    Nv = int(future_state.shape[1])
    k_weights = _rollout_k_weights(int(future_state.shape[2]))
    tail_start = min(int(math.ceil(float(tail_start_fraction) * float(Nv))), int(Nv) - 1)
    tail_weights = jnp.asarray(tail_mode_weights(Nv, tail_start_fraction), dtype=jnp.float64)
    weights = jnp.asarray(rollout_weights, dtype=jnp.float64)[:H]

    field_num_by_step = jnp.sum(
        k_weights[None, 1:] * (jnp.abs(pred_future_E_hat[:, 1:] - future_E_hat[:, 1:]) ** 2),
        axis=1,
    )
    field_num_total = jnp.sum(weights * field_num_by_step)
    field_true_total, tail_true_total = rollout_window_teacher_scales(
        future_state,
        future_E_hat,
        rollout_weights=weights,
        tail_start_fraction=tail_start_fraction,
    )

    pred_tail = jnp.abs(pred_future_state[:, tail_start:, :]) ** 2
    true_tail = jnp.abs(future_state[:, tail_start:, :]) ** 2
    excess = jnp.maximum(pred_tail - true_tail, 0.0)
    alpha = tail_weights[:, None] * k_weights[None, :]
    tail_num_by_step = jnp.sum(alpha[None, :, :] * (excess ** 2), axis=(1, 2))
    tail_num_total = jnp.sum(weights * tail_num_by_step)

    field_loss = field_num_total / (
        field_true_total + jnp.asarray(field_ref_scale, dtype=jnp.float64) + 1e-30
    )
    tail_loss = tail_num_total / (
        tail_true_total + jnp.asarray(tail_ref_scale, dtype=jnp.float64) + 1e-30
    )
    return field_loss, tail_loss


def build_stability_rollout_reference_scales(
    prepared: Dict[str, Dict[str, Array]],
    *,
    active_regimes: Sequence[str],
    regime_rollout_nvs: Dict[str, Sequence[int]],
    rollout_weights: Array,
    tail_start_fraction: float,
) -> Dict[str, Dict[int, Dict[str, Array]]]:
    scales: Dict[str, Dict[int, Dict[str, Array]]] = {}
    for regime in active_regimes:
        scales[regime] = {}
        for Nv in regime_rollout_nvs[regime]:
            payload = prepared[regime]["train_rollout"][int(Nv)]
            future_state = payload["future_state"]
            future_E_hat = payload["future_E_hat"]
            if int(future_state.shape[0]) == 0:
                field_ref = jnp.asarray(0.0, dtype=jnp.float64)
                tail_ref = jnp.asarray(0.0, dtype=jnp.float64)
            else:
                field_scales, tail_scales = jax.vmap(
                    lambda states, e_hat: rollout_window_teacher_scales(
                        states,
                        e_hat,
                        rollout_weights=rollout_weights,
                        tail_start_fraction=tail_start_fraction,
                    )
                )(future_state, future_E_hat)
                field_ref = jnp.mean(field_scales)
                tail_ref = jnp.mean(tail_scales)
            scales[regime][int(Nv)] = {
                "field": jnp.asarray(field_ref, dtype=jnp.float64),
                "tail": jnp.asarray(tail_ref, dtype=jnp.float64),
            }
    return scales


def landau_explicit_n_hat(
    a_hat: Array,
    *,
    integ: FourierHermiteIMEX,
    m_eq: Array,
    poisson_sign: float,
) -> Array:
    a_phys = jnp.fft.irfft(a_hat, n=int(integ.Nx), axis=1).astype(jnp.float64)
    e_phys = integ.E_phys_from_a_hat(a_hat, poisson_sign=float(poisson_sign))
    n_phys = jnp.zeros_like(a_phys)
    n_phys = n_phys.at[1:].set(
        -(integ.sqrt_n[1:, None] / float(integ.vth))
        * e_phys[None, :]
        * (a_phys[:-1] + m_eq[:-1, None])
    )
    return integ.apply_mask_hat(jnp.fft.rfft(n_phys, axis=1).astype(jnp.complex128))


def rollout_window_loss_terms(
    learned: LearnedInterfaceClosure,
    *,
    prev_state: Array,
    curr_state: Array,
    future_state: Array,
    future_E_hat: Array,
    integ: FourierHermiteIMEX,
    m_eq: Array,
    poisson_sign: float,
    rollout_weights: Array,
    tail_start_fraction: float,
    field_ref_scale: Array,
    tail_ref_scale: Array,
) -> Tuple[Array, Array]:
    H = int(future_state.shape[0])

    n_prev = landau_explicit_n_hat(prev_state, integ=integ, m_eq=m_eq, poisson_sign=poisson_sign)
    n_curr = landau_explicit_n_hat(curr_state, integ=integ, m_eq=m_eq, poisson_sign=poisson_sign)
    b_prev = learned_boundary_flux_hat(
        prev_state,
        integ.k_arr,
        integ.Nv,
        integ.vth,
        learned,
        a_hat_prev=prev_state,
    )
    b_curr = learned_boundary_flux_hat(
        curr_state,
        integ.k_arr,
        integ.Nv,
        integ.vth,
        learned,
        a_hat_prev=prev_state,
    )

    def body(carry, _step_idx):
        a_prev, a_curr, n_prev, n_curr, b_prev, b_curr = carry
        a_next = integ.step_cnab2(
            a_curr,
            n_curr,
            n_prev,
            extra_hat=b_curr,
            extra_hat_prev=b_prev,
        )
        e_pred = e_hat_from_rho_hat(a_next[0], integ.k_arr, poisson_sign=float(poisson_sign))

        n_next = landau_explicit_n_hat(a_next, integ=integ, m_eq=m_eq, poisson_sign=poisson_sign)
        b_next = learned_boundary_flux_hat(
            a_next,
            integ.k_arr,
            integ.Nv,
            integ.vth,
            learned,
            a_hat_prev=a_curr,
        )
        next_carry = (a_curr, a_next, n_curr, n_next, b_curr, b_next)
        return next_carry, (a_next, e_pred)

    _, (pred_future_state, pred_future_E_hat) = jax.lax.scan(
        body,
        (prev_state, curr_state, n_prev, n_curr, b_prev, b_curr),
        jnp.arange(H, dtype=jnp.int32),
    )
    return rollout_window_hybrid_loss_from_sequences(
        pred_future_state,
        pred_future_E_hat,
        future_state,
        future_E_hat,
        rollout_weights=rollout_weights,
        tail_start_fraction=tail_start_fraction,
        field_ref_scale=field_ref_scale,
        tail_ref_scale=tail_ref_scale,
    )


def make_stability_aware_batch_loss(
    *,
    prepared: Dict[str, Dict[str, Array]],
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
    rollout_horizon: int,
    tail_start_fraction: float,
    lambda_q: float,
    lambda_E: float,
    lambda_tail: float,
    lambda_reg: float,
    rollout_dealias_23: bool,
    poisson_sign: float,
):
    active_regimes = tuple(regime for regime in train_regimes if regime in prepared and regime in regime_weights)
    weights = np.asarray([float(regime_weights[regime]) for regime in active_regimes], dtype=np.float64)
    weights = weights / np.sum(weights)
    rollout_weights = jnp.asarray(build_rollout_weights(int(rollout_horizon)), dtype=jnp.float64)
    regime_rollout_nvs = {
        regime: tuple(sorted(int(v) for v in prepared[regime]["train_rollout"].keys()))
        for regime in active_regimes
    }
    rollout_reference_scales = build_stability_rollout_reference_scales(
        prepared,
        active_regimes=active_regimes,
        regime_rollout_nvs=regime_rollout_nvs,
        rollout_weights=rollout_weights,
        tail_start_fraction=tail_start_fraction,
    )
    integrators = {
        (regime, Nv): FourierHermiteIMEX(
            Nx=int(teacher_Nx),
            Nv=int(Nv),
            Lx=float(teacher_Lx),
            dt=float(teacher_dt),
            vth=1.0,
            dealias_23=bool(rollout_dealias_23),
            closure=None,
        )
        for regime in active_regimes
        for Nv in regime_rollout_nvs[regime]
    }
    m_eq = {
        int(Nv): jnp.zeros((int(Nv),), dtype=jnp.float64).at[0].set(1.0)
        for regime in active_regimes
        for Nv in regime_rollout_nvs[regime]
    }
    weight_arr = jnp.asarray(weights, dtype=jnp.float64)
    lambda_q_arr = jnp.asarray(float(lambda_q), dtype=jnp.float64)
    lambda_E_arr = jnp.asarray(float(lambda_E), dtype=jnp.float64)
    lambda_tail_arr = jnp.asarray(float(lambda_tail), dtype=jnp.float64)
    lambda_reg_arr = jnp.asarray(float(lambda_reg), dtype=jnp.float64)

    def loss_fn(
        params: Dict[str, Array],
        batch_inputs: Dict[str, Array],
        batch_targets_std: Dict[str, Array],
        rollout_batches: Dict[str, Dict[int, Dict[str, Array]]],
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
            train_objective="stability_aware",
            context_mode=context_mode,
            rollout_horizon=rollout_horizon,
            tail_start_fraction=tail_start_fraction,
            lambda_q=lambda_q,
            lambda_E=lambda_E,
            lambda_tail=lambda_tail,
            lambda_reg=lambda_reg,
            stability_loss_definition=STABILITY_LOSS_DEFINITION,
        )
        total_q = jnp.asarray(0.0, dtype=jnp.float64)
        total_field = jnp.asarray(0.0, dtype=jnp.float64)
        total_tail = jnp.asarray(0.0, dtype=jnp.float64)
        for weight, regime in zip(weight_arr, active_regimes):
            pred_std = learned.predict_standardized_components(batch_inputs[regime])
            target_std = batch_targets_std[regime]
            regime_q = jnp.mean((pred_std - target_std) ** 2)

            regime_field = jnp.asarray(0.0, dtype=jnp.float64)
            regime_tail = jnp.asarray(0.0, dtype=jnp.float64)
            rollout_groups = rollout_batches[regime]
            if rollout_groups:
                field_terms = []
                tail_terms = []
                for Nv in regime_rollout_nvs[regime]:
                    batch = rollout_groups[int(Nv)]
                    if batch["prev_state"].shape[0] == 0:
                        continue
                    field_term, tail_term = jax.vmap(
                        lambda prev_state, curr_state, future_state, future_E_hat: rollout_window_loss_terms(
                            learned,
                            prev_state=prev_state,
                            curr_state=curr_state,
                            future_state=future_state,
                            future_E_hat=future_E_hat,
                            integ=integrators[(regime, int(Nv))],
                            m_eq=m_eq[int(Nv)],
                            poisson_sign=poisson_sign,
                            rollout_weights=rollout_weights,
                            tail_start_fraction=tail_start_fraction,
                            field_ref_scale=rollout_reference_scales[regime][int(Nv)]["field"],
                            tail_ref_scale=rollout_reference_scales[regime][int(Nv)]["tail"],
                        )
                    )(
                        batch["prev_state"],
                        batch["curr_state"],
                        batch["future_state"],
                        batch["future_E_hat"],
                    )
                    field_terms.append(jnp.mean(field_term))
                    tail_terms.append(jnp.mean(tail_term))
                if field_terms:
                    regime_field = jnp.mean(jnp.stack(field_terms))
                    regime_tail = jnp.mean(jnp.stack(tail_terms))
            total_q = total_q + weight * (lambda_q_arr * regime_q)
            total_field = total_field + weight * (lambda_E_arr * regime_field)
            total_tail = total_tail + weight * (lambda_tail_arr * regime_tail)
        reg_term = lambda_reg_arr * l2_regularization(params)
        total_loss = total_q + total_field + total_tail + reg_term
        return total_loss, {
            "q": total_q,
            "field": total_field,
            "tail": total_tail,
            "reg": reg_term,
        }

    return loss_fn, active_regimes, regime_rollout_nvs


def train_with_stability_aware_minibatch_loss(
    params: Dict[str, Array],
    prepared: Dict[str, Dict[str, Array]],
    batch_loss_fn,
    *,
    active_regimes: Sequence[str],
    regime_rollout_nvs: Dict[str, Sequence[int]],
    epochs: int,
    learning_rate: float,
    grad_clip: Optional[float],
    log_every: int,
    batch_size: int,
    rollout_batch_size: int,
    steps_per_epoch: int,
    seed: int,
) -> Tuple[Dict[str, Array], Dict[str, np.ndarray]]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive for stability-aware training")
    if int(rollout_batch_size) <= 0:
        raise ValueError("rollout_batch_size must be positive for stability-aware training")
    if int(steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be positive for stability-aware training")

    q_train_sizes = {
        regime: int(prepared[regime]["train_inputs"].shape[0])
        for regime in active_regimes
    }
    rollout_train_sizes = {
        regime: {
            int(Nv): int(prepared[regime]["train_rollout"][int(Nv)]["prev_state"].shape[0])
            for Nv in regime_rollout_nvs[regime]
        }
        for regime in active_regimes
    }
    state = adam_init(params)
    history = {
        key: np.zeros((int(epochs),), dtype=np.float64)
        for key in ("total", "q", "field", "tail", "reg")
    }

    @jax.jit
    def train_step(
        current_params: Dict[str, Array],
        current_state: Dict[str, object],
        batch_inputs: Dict[str, Array],
        batch_targets_std: Dict[str, Array],
        rollout_batches: Dict[str, Dict[int, Dict[str, Array]]],
    ) -> Tuple[Dict[str, Array], Dict[str, object], Dict[str, Array]]:
        (loss, aux), grads = jax.value_and_grad(batch_loss_fn, has_aux=True)(
            current_params,
            batch_inputs,
            batch_targets_std,
            rollout_batches,
        )
        next_params, next_state = adam_step(
            current_params,
            grads,
            current_state,
            learning_rate,
            grad_clip=grad_clip,
        )
        aux = dict(aux)
        aux["total"] = loss
        return next_params, next_state, aux

    rng = np.random.default_rng(int(seed))
    for epoch in range(int(epochs)):
        running = {
            key: jnp.asarray(0.0, dtype=jnp.float64)
            for key in ("total", "q", "field", "tail", "reg")
        }
        for _ in range(int(steps_per_epoch)):
            batch_inputs = {}
            batch_targets_std = {}
            rollout_batches: Dict[str, Dict[int, Dict[str, Array]]] = {}
            for regime in active_regimes:
                idx = rng.integers(0, q_train_sizes[regime], size=int(batch_size), endpoint=False)
                batch_inputs[regime] = prepared[regime]["train_inputs"][idx]
                batch_targets_std[regime] = prepared[regime]["train_targets_std"][idx]
                rollout_batches[regime] = {}
                for Nv in regime_rollout_nvs[regime]:
                    group = prepared[regime]["train_rollout"][int(Nv)]
                    size = rollout_train_sizes[regime][int(Nv)]
                    if size == 0:
                        rollout_batches[regime][int(Nv)] = {
                            "prev_state": group["prev_state"],
                            "curr_state": group["curr_state"],
                            "future_state": group["future_state"],
                            "future_E_hat": group["future_E_hat"],
                        }
                        continue
                    idx_roll = rng.integers(0, size, size=int(min(rollout_batch_size, size)), endpoint=False)
                    rollout_batches[regime][int(Nv)] = {
                        "prev_state": group["prev_state"][idx_roll],
                        "curr_state": group["curr_state"][idx_roll],
                        "future_state": group["future_state"][idx_roll],
                        "future_E_hat": group["future_E_hat"][idx_roll],
                    }
            params, state, aux = train_step(params, state, batch_inputs, batch_targets_std, rollout_batches)
            for key in running:
                running[key] = running[key] + aux[key]
        for key, values in history.items():
            values[epoch] = float(running[key] / float(steps_per_epoch))
        if epoch == 0 or (epoch + 1) % max(int(log_every), 1) == 0 or epoch + 1 == int(epochs):
            print(
                f"[train] epoch {epoch + 1:04d}/{int(epochs):04d} "
                f"loss={history['total'][epoch]:.6e} "
                f"q={history['q'][epoch]:.6e} "
                f"field={history['field'][epoch]:.6e} "
                f"tail={history['tail'][epoch]:.6e} "
                f"reg={history['reg'][epoch]:.6e}"
            )
    return params, history


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
    target_nv = int(Nv_targets[0])
    v_probe = jnp.linspace(float(teacher_vmin), float(teacher_vmax), int(online_v_probes), dtype=jnp.float64)
    eq_probe = maxwellian_equilibrium(v_probe)
    k_arr = FourierHermiteIMEX(
        Nx=int(teacher_Nx),
        Nv=int(target_nv),
        Lx=float(teacher_Lx),
        dt=float(teacher_dt),
        vth=1.0,
        dealias_23=bool(rollout_dealias_23),
        closure=None,
    ).k_arr
    linear_config = LinearLandauConfig(
        method="learned",
        Nv=int(target_nv),
        Nx=int(teacher_Nx),
        L=float(teacher_Lx),
        dt=float(teacher_dt),
        T=float(online_dataset[REGIME_LINEAR]["train"]["times"].shape[1] - 1) * float(teacher_dt)
        if REGIME_LINEAR in online_dataset and online_dataset[REGIME_LINEAR].get("train")
        else float(nonlinear_T),
        poisson_sign=float(poisson_sign),
    )
    lambda_E_arr = jnp.asarray(float(lambda_E), dtype=jnp.float64)
    lambda_dist_arr = jnp.asarray(float(lambda_dist), dtype=jnp.float64)
    lambda_tail_arr = jnp.asarray(float(lambda_tail), dtype=jnp.float64)
    lambda_neg_arr = jnp.asarray(float(lambda_neg), dtype=jnp.float64)
    lambda_reg_arr = jnp.asarray(float(lambda_reg), dtype=jnp.float64)

    def loss_fn(
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
            "field": total_field,
            "dist": total_dist,
            "tail": total_tail,
            "neg": total_neg,
            "reg": reg_term,
        }

    return loss_fn, active_regimes


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
) -> Tuple[Dict[str, Array], Dict[str, np.ndarray]]:
    if int(online_case_batch_size) <= 0:
        raise ValueError("online_case_batch_size must be positive for online rollout training")
    if int(steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be positive for online rollout training")

    train_sizes = {
        regime: int(online_dataset[regime]["train"]["E_hat_ref"].shape[0])
        for regime in active_regimes
    }
    state = adam_init(params)
    history = {
        key: np.zeros((int(epochs),), dtype=np.float64)
        for key in ("total", "field", "dist", "tail", "neg", "reg")
    }

    @jax.jit
    def train_step(
        current_params: Dict[str, Array],
        current_state: Dict[str, object],
        regime_batches: Dict[str, Dict[str, Array]],
    ) -> Tuple[Dict[str, Array], Dict[str, object], Dict[str, Array]]:
        (loss, aux), grads = jax.value_and_grad(batch_loss_fn, has_aux=True)(current_params, regime_batches)
        next_params, next_state = adam_step(
            current_params,
            grads,
            current_state,
            learning_rate,
            grad_clip=grad_clip,
        )
        aux = dict(aux)
        aux["total"] = loss
        return next_params, next_state, aux

    rng = np.random.default_rng(int(seed))
    for epoch in range(int(epochs)):
        running = {
            key: jnp.asarray(0.0, dtype=jnp.float64)
            for key in ("total", "field", "dist", "tail", "neg", "reg")
        }
        for _ in range(int(steps_per_epoch)):
            regime_batches: Dict[str, Dict[str, Array]] = {}
            for regime in active_regimes:
                group = online_dataset[regime]["train"]
                size = train_sizes[regime]
                batch_n = int(min(online_case_batch_size, size))
                idx = rng.integers(0, size, size=batch_n, endpoint=False)
                regime_batches[regime] = {key: value[idx] for key, value in group.items()}
            params, state, aux = train_step(params, state, regime_batches)
            for key in running:
                running[key] = running[key] + aux[key]
        for key in history:
            history[key][epoch] = float(running[key] / float(steps_per_epoch))
        if epoch == 0 or (epoch + 1) % max(int(log_every), 1) == 0 or epoch + 1 == int(epochs):
            print(
                f"[train] epoch {epoch + 1:04d}/{int(epochs):04d} "
                f"loss={history['total'][epoch]:.6e} "
                f"field={history['field'][epoch]:.6e} "
                f"dist={history['dist'][epoch]:.6e} "
                f"tail={history['tail'][epoch]:.6e} "
                f"neg={history['neg'][epoch]:.6e} "
                f"reg={history['reg'][epoch]:.6e}"
            )
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a shared learned interface closure from a selectable Landau teacher")
    parser.add_argument("--checkpoint", type=Path, default=None)
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
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--k-scale", type=float, default=None)
    parser.add_argument("--nv-scale", type=float, default=None)
    parser.add_argument("--n-low", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--training-mode", type=str, default=OFFLINE_TRAINING_MODE, choices=(OFFLINE_TRAINING_MODE, ONLINE_TRAINING_MODE))
    parser.add_argument("--train-objective", type=str, default="q_only", choices=("q_only", "stability_aware", "trajectory"))
    parser.add_argument("--context-mode", type=str, default="none", choices=("none", "lag1_delta"))
    parser.add_argument("--rollout-horizon", type=int, default=2)
    parser.add_argument("--tail-start-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--lambda-q", type=float, default=1.0)
    parser.add_argument("--lambda-E", type=float, default=0.5)
    parser.add_argument("--lambda-dist", type=float, default=1.0)
    parser.add_argument("--lambda-tail", type=float, default=0.05)
    parser.add_argument("--lambda-neg", type=float, default=0.05)
    parser.add_argument("--lambda-reg", type=float, default=1e-6)
    parser.add_argument("--rollout-dealias-23", action="store_true")
    parser.add_argument("--online-loss-backend", type=str, default=ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1)
    parser.add_argument("--online-v-probes", type=int, default=64)
    parser.add_argument("--online-case-batch-size", type=int, default=1)
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
    teacher_proj_Nv: Optional[int] = None
    if training_mode == ONLINE_TRAINING_MODE:
        if bool(args.build_dataset_only):
            raise ValueError("online_rollout does not support --build-dataset-only")
        if args.dataset_cache is not None:
            raise ValueError("online_rollout does not support --dataset-cache")
        if bool(args.allow_dataset_cache_nv_superset):
            raise ValueError("online_rollout does not support --allow-dataset-cache-nv-superset")
        if bool(args.per_target_projection_orders):
            raise ValueError("online_rollout does not support --per-target-projection-orders")
        if teacher_backend != GRID_CUBIC_SPLINE_TEACHER_BACKEND:
            raise ValueError("online_rollout only supports teacher_backend=grid_cubic_spline")
        if len(Nv_targets) != 1:
            raise ValueError("online_rollout requires exactly one target Nv")
        if args.teacher_proj_Nv is not None:
            raise ValueError("online_rollout does not use --teacher-proj-Nv")
        if args.train_objective != "trajectory":
            raise ValueError("online_rollout requires --train-objective trajectory")
        if str(args.online_loss_backend) != ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1:
            raise ValueError(
                f"Unsupported online loss backend {args.online_loss_backend!r}; "
                f"expected {ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1!r}"
            )
        if int(args.online_v_probes) <= 0:
            raise ValueError("online_rollout requires --online-v-probes > 0")
        if int(args.online_case_batch_size) <= 0:
            raise ValueError("online_rollout requires --online-case-batch-size > 0")
        if float(args.lambda_E) <= 0.0 and float(args.lambda_dist) <= 0.0:
            raise ValueError("online_rollout requires lambda_E > 0 or lambda_dist > 0")
    else:
        if args.train_objective == "trajectory":
            raise ValueError("trajectory objective is only supported with --training-mode online_rollout")
        if args.train_objective == "stability_aware" and int(args.batch_size) <= 0:
            raise ValueError("stability_aware training requires --batch-size > 0")
        if args.train_objective == "stability_aware" and (
            float(args.lambda_E) <= 0.0 and float(args.lambda_tail) <= 0.0
        ):
            raise ValueError("stability_aware training requires lambda_E > 0 or lambda_tail > 0")
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
    stability_component_history: Optional[Dict[str, np.ndarray]] = None
    online_component_history: Optional[Dict[str, np.ndarray]] = None

    if training_mode == OFFLINE_TRAINING_MODE:
        dataset_rollout_horizon = int(args.rollout_horizon) if args.train_objective == "stability_aware" else 0
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
            rollout_horizon=dataset_rollout_horizon,
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
        if args.train_objective == "stability_aware":
            batch_loss_fn, active_regimes, regime_rollout_nvs = make_stability_aware_batch_loss(
                prepared=prepared,
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
                rollout_horizon=args.rollout_horizon,
                tail_start_fraction=args.tail_start_fraction,
                lambda_q=args.lambda_q,
                lambda_E=args.lambda_E,
                lambda_tail=args.lambda_tail,
                lambda_reg=args.lambda_reg,
                rollout_dealias_23=bool(args.rollout_dealias_23),
                poisson_sign=args.teacher_poisson_sign,
            )
            q_train_sizes = [int(prepared[regime]["train_inputs"].shape[0]) for regime in active_regimes]
            steps_per_epoch = int(args.steps_per_epoch)
            if steps_per_epoch <= 0:
                steps_per_epoch = max(1, math.ceil(max(q_train_sizes) / float(args.batch_size)))
            params, stability_component_history = train_with_stability_aware_minibatch_loss(
                params,
                prepared,
                batch_loss_fn,
                active_regimes=active_regimes,
                regime_rollout_nvs=regime_rollout_nvs,
                epochs=args.epochs,
                learning_rate=args.lr,
                grad_clip=args.grad_clip,
                log_every=args.log_every,
                batch_size=args.batch_size,
                rollout_batch_size=args.rollout_batch_size,
                steps_per_epoch=steps_per_epoch,
                seed=args.seed,
            )
            loss_history = stability_component_history["total"]
        elif int(args.batch_size) > 0:
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
            rollout_horizon=args.rollout_horizon if args.train_objective == "stability_aware" else 0,
            tail_start_fraction=args.tail_start_fraction,
            lambda_q=args.lambda_q,
            lambda_E=args.lambda_E if args.train_objective == "stability_aware" else 0.0,
            lambda_tail=args.lambda_tail if args.train_objective == "stability_aware" else 0.0,
            lambda_reg=args.lambda_reg if args.train_objective == "stability_aware" else 0.0,
            stability_loss_definition=(
                STABILITY_LOSS_DEFINITION if args.train_objective == "stability_aware" else None
            ),
        )
        val_metrics = evaluate_regime_metrics(learned, prepared)
    else:
        online_dataset, _ = build_online_reference_dataset(
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
        )
        for regime in regimes:
            if regime not in online_dataset:
                continue
            train_count = int(online_dataset[regime]["train"]["E_hat_ref"].shape[0]) if online_dataset[regime].get("train") else 0
            val_count = int(online_dataset[regime]["val"]["E_hat_ref"].shape[0]) if online_dataset[regime].get("val") else 0
            print(f"[data] {regime}: train={train_count} episodes val={val_count} episodes")

        integ = FourierHermiteIMEX(
            Nx=int(args.teacher_Nx),
            Nv=int(Nv_targets[0]),
            Lx=float(args.teacher_L),
            dt=float(args.teacher_dt),
            vth=1.0,
            dealias_23=bool(args.rollout_dealias_23),
            closure=None,
        )
        k_scale = float(args.k_scale) if args.k_scale is not None else float(jnp.max(jnp.asarray(integ.k_arr[1:], dtype=jnp.float64)))
        nv_scale = float(args.nv_scale) if args.nv_scale is not None else float(Nv_targets[0])
        stats = build_identity_training_stats(Nm=args.Nm, context_mode=args.context_mode)
        params = init_online_rollout_params(
            jax.random.PRNGKey(args.seed),
            input_dim=int(stats["input_mean"].shape[0]),
            hidden_width=int(args.hidden_width),
            res_blocks=int(args.res_blocks),
        )
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
        train_sizes = [int(online_dataset[regime]["train"]["E_hat_ref"].shape[0]) for regime in active_regimes]
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
        )
        loss_history = online_component_history["total"]
        for regime in regimes:
            if regime in online_dataset and online_dataset[regime].get("val"):
                val_metrics[f"val_num_cases_{regime}"] = np.array(
                    [int(online_dataset[regime]["val"]["E_hat_ref"].shape[0])],
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
            teacher_proj_Nv=None,
            n_low=args.n_low,
            training_mode=ONLINE_TRAINING_MODE,
            train_objective="trajectory",
            context_mode=args.context_mode,
            rollout_horizon=0,
            tail_start_fraction=args.tail_start_fraction,
            loss_backend=args.online_loss_backend,
            lambda_q=0.0,
            lambda_E=args.lambda_E,
            lambda_dist=args.lambda_dist,
            lambda_tail=args.lambda_tail,
            lambda_neg=args.lambda_neg,
            lambda_reg=args.lambda_reg,
            online_v_probes=args.online_v_probes,
        )

    assert args.checkpoint is not None
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_learned_interface_closure_npz(args.checkpoint, learned)

    metrics_path = args.checkpoint.with_suffix(".metrics.npz")
    used_lambda_E = args.lambda_E if args.train_objective in {"stability_aware", "trajectory"} else 0.0
    used_lambda_dist = args.lambda_dist if training_mode == ONLINE_TRAINING_MODE else 0.0
    used_lambda_tail = args.lambda_tail if args.train_objective in {"stability_aware", "trajectory"} else 0.0
    used_lambda_neg = args.lambda_neg if training_mode == ONLINE_TRAINING_MODE else 0.0
    used_lambda_reg = args.lambda_reg if args.train_objective in {"stability_aware", "trajectory"} else 0.0
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
        "rollout_horizon": np.array([args.rollout_horizon], dtype=np.int32),
        "tail_start_fraction": np.array([args.tail_start_fraction], dtype=np.float64),
        "loss_backend": np.array(
            [] if training_mode == OFFLINE_TRAINING_MODE else [args.online_loss_backend],
            dtype=np.str_,
        ),
        "lambda_q": np.array([args.lambda_q if training_mode == OFFLINE_TRAINING_MODE else 0.0], dtype=np.float64),
        "lambda_E": np.array([used_lambda_E], dtype=np.float64),
        "lambda_dist": np.array([used_lambda_dist], dtype=np.float64),
        "lambda_tail": np.array([used_lambda_tail], dtype=np.float64),
        "lambda_neg": np.array([used_lambda_neg], dtype=np.float64),
        "lambda_reg": np.array([used_lambda_reg], dtype=np.float64),
        "online_v_probes": np.array([args.online_v_probes if training_mode == ONLINE_TRAINING_MODE else 0], dtype=np.int32),
    }
    if teacher_proj_Nv is not None:
        metrics_payload["teacher_proj_Nv"] = np.array([teacher_proj_Nv], dtype=np.int32)
    if args.train_objective == "stability_aware":
        assert stability_component_history is not None
        metrics_payload["train_loss_q"] = np.asarray(stability_component_history["q"], dtype=np.float64)
        metrics_payload["train_loss_field"] = np.asarray(stability_component_history["field"], dtype=np.float64)
        metrics_payload["train_loss_tail"] = np.asarray(stability_component_history["tail"], dtype=np.float64)
        metrics_payload["train_loss_reg"] = np.asarray(stability_component_history["reg"], dtype=np.float64)
        metrics_payload["stability_loss_definition"] = np.array([STABILITY_LOSS_DEFINITION], dtype=np.str_)
    if training_mode == ONLINE_TRAINING_MODE:
        assert online_component_history is not None
        metrics_payload["train_loss_field"] = np.asarray(online_component_history["field"], dtype=np.float64)
        metrics_payload["train_loss_dist"] = np.asarray(online_component_history["dist"], dtype=np.float64)
        metrics_payload["train_loss_tail"] = np.asarray(online_component_history["tail"], dtype=np.float64)
        metrics_payload["train_loss_neg"] = np.asarray(online_component_history["neg"], dtype=np.float64)
        metrics_payload["train_loss_reg"] = np.asarray(online_component_history["reg"], dtype=np.float64)
    metrics_payload.update(val_metrics)
    np.savez(metrics_path, **metrics_payload)

    loss_plot_path = args.loss_plot if args.loss_plot is not None else args.checkpoint.with_suffix(".loss.png")
    save_training_loss_plot(
        np.asarray(loss_history, dtype=np.float64),
        loss_plot_path,
        val_metrics=val_metrics,
        train_objective=args.train_objective,
    )

    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved loss plot to {loss_plot_path}")
    for key in sorted(val_metrics):
        print(f"{key}: {float(np.asarray(val_metrics[key]).reshape(-1)[0]):.6e}")


if __name__ == "__main__":
    main()
