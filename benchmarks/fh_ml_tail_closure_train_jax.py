"""
Train a shared learned interface closure for Landau-family runs using a physical-grid teacher.

The teacher is a full Vlasov-Poisson semi-Lagrangian solve on a fine (x, v) grid with
JAX cubic spline interpolation. Teacher snapshots are projected onto the Fourier-Hermite
basis, and the learned target is

    q_k^* = -i k v_th sqrt(Nv) C_{Nv,k}^{HR}.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from vpml.jax_runtime import bootstrap_jax_runtime, print_jax_runtime_summary

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))

from vpml.core import (
    Array,
    LearnedInterfaceClosure,
    init_interface_closure_params,
    save_learned_interface_closure_npz,
)
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    extract_interface_supervised_pairs_from_coeff_history,
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
CACHE_FORMAT = "landau_interface_dataset_physical_teacher_v4"


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def parse_str_tuple(text: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def build_dataset_cache_metadata(
    *,
    regimes: Sequence[str],
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_L: float,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    teacher_proj_Nv: int,
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
    projection_mode: str = "shared_max",
    teacher_proj_Nv_targets: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    payload = {
        "dataset_format": np.array([CACHE_FORMAT], dtype=np.str_),
        "regimes": np.asarray(tuple(regimes), dtype=np.str_),
        "n_low": np.array([int(n_low)], dtype=np.int32),
        "Nm": np.array([int(Nm)], dtype=np.int32),
        "Nv_targets": np.asarray(tuple(int(v) for v in Nv_targets), dtype=np.int32),
        "projection_mode": np.array([str(projection_mode)], dtype=np.str_),
        "teacher_Nx": np.array([int(teacher_Nx)], dtype=np.int32),
        "teacher_Nv": np.array([int(teacher_Nv)], dtype=np.int32),
        "teacher_L": np.array([float(teacher_L)], dtype=np.float64),
        "teacher_vmin": np.array([float(teacher_vmin)], dtype=np.float64),
        "teacher_vmax": np.array([float(teacher_vmax)], dtype=np.float64),
        "teacher_dt": np.array([float(teacher_dt)], dtype=np.float64),
        "teacher_proj_Nv": np.array([int(teacher_proj_Nv)], dtype=np.int32),
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

    v = config.v
    equilibrium = maxwellian_equilibrium(v)
    f0 = equilibrium[:, None] * (1.0 + jnp.asarray(perturbation_x, dtype=jnp.float64)[None, :])
    raw = run_semilagrangian_vlasov_poisson(
        config,
        f0,
        history_stride=history_stride,
        return_state_history=True,
        history_projector=_multi_projected_history_projector(
            v,
            orders,
            equilibrium=equilibrium,
            vth=1.0,
        ),
    )
    stacked = np.asarray(raw["state_history"], dtype=np.complex128)
    histories: Dict[int, np.ndarray] = {}
    start = 0
    for order in orders:
        stop = start + int(order)
        histories[int(order)] = stacked[:, start:stop, :]
        start = stop
    return histories, np.asarray(raw["k_arr"], dtype=np.float64)


def build_linear_landau_regime(
    *,
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_L: float,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    teacher_proj_Nv: int,
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
    per_target_projection_orders: bool = False,
) -> Dict[str, np.ndarray]:
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
    rng = np.random.default_rng(seed)
    x = np.asarray(config.x, dtype=np.float64)

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
        if bool(per_target_projection_orders):
            projection_orders = tuple(sorted({int(Nv) + 1 for Nv in Nv_targets}))
            coeff_histories, k_arr = _run_landau_teacher_projected_histories(
                config,
                perturb,
                projection_orders=projection_orders,
                history_stride=history_stride,
            )
            for Nv in Nv_targets:
                coeff_hist = coeff_histories[int(Nv) + 1]
                train_hist, val_hist = split_history_train_val(coeff_hist, val_fraction)
                append_pairs(
                    accum,
                    REGIME_LINEAR,
                    "train",
                    extract_interface_supervised_pairs_from_coeff_history(
                        train_hist,
                        Nv_targets=(int(Nv),),
                        Nm=Nm,
                        k_arr=k_arr,
                        vth=1.0,
                        include_global_indicators=True,
                        n_low=int(n_low),
                    ),
                )
                append_pairs(
                    accum,
                    REGIME_LINEAR,
                    "val",
                    extract_interface_supervised_pairs_from_coeff_history(
                        val_hist,
                        Nv_targets=(int(Nv),),
                        Nm=Nm,
                        k_arr=k_arr,
                        vth=1.0,
                        include_global_indicators=True,
                        n_low=int(n_low),
                    ),
                )
        else:
            coeff_hist, k_arr = _run_landau_teacher_projected_history(
                config,
                perturb,
                projection_order=int(teacher_proj_Nv),
                history_stride=history_stride,
            )
            train_hist, val_hist = split_history_train_val(coeff_hist, val_fraction)
            append_pairs(
                accum,
                REGIME_LINEAR,
                "train",
                extract_interface_supervised_pairs_from_coeff_history(
                    train_hist,
                    Nv_targets=Nv_targets,
                    Nm=Nm,
                    k_arr=k_arr,
                    vth=1.0,
                    include_global_indicators=True,
                    n_low=int(n_low),
                ),
            )
            append_pairs(
                accum,
                REGIME_LINEAR,
                "val",
                extract_interface_supervised_pairs_from_coeff_history(
                    val_hist,
                    Nv_targets=Nv_targets,
                    Nm=Nm,
                    k_arr=k_arr,
                    vth=1.0,
                    include_global_indicators=True,
                    n_low=int(n_low),
                ),
            )
    return finalize_regime_arrays(accum)[REGIME_LINEAR]


def build_nonlinear_landau_regime(
    regime_name: str,
    eps_values: Sequence[float],
    *,
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_L: float,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    teacher_proj_Nv: int,
    Nv_targets: Sequence[int],
    Nm: int,
    T: float,
    k0: float,
    poisson_sign: float,
    history_stride: int,
    val_fraction: float,
    n_low: int,
    per_target_projection_orders: bool = False,
) -> Dict[str, np.ndarray]:
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
    accum = {
        regime_name: {
            "train_inputs_base": [],
            "train_targets": [],
            "val_inputs_base": [],
            "val_targets": [],
        }
    }

    for eps in eps_values:
        if bool(per_target_projection_orders):
            projection_orders = tuple(sorted({int(Nv) + 1 for Nv in Nv_targets}))
            coeff_histories, k_arr = _run_landau_teacher_projected_histories(
                config,
                float(eps) * perturb_template,
                projection_orders=projection_orders,
                history_stride=history_stride,
            )
            for Nv in Nv_targets:
                coeff_hist = coeff_histories[int(Nv) + 1]
                train_hist, val_hist = split_history_train_val(coeff_hist, val_fraction)
                append_pairs(
                    accum,
                    regime_name,
                    "train",
                    extract_interface_supervised_pairs_from_coeff_history(
                        train_hist,
                        Nv_targets=(int(Nv),),
                        Nm=Nm,
                        k_arr=k_arr,
                        vth=1.0,
                        include_global_indicators=True,
                        n_low=int(n_low),
                    ),
                )
                append_pairs(
                    accum,
                    regime_name,
                    "val",
                    extract_interface_supervised_pairs_from_coeff_history(
                        val_hist,
                        Nv_targets=(int(Nv),),
                        Nm=Nm,
                        k_arr=k_arr,
                        vth=1.0,
                        include_global_indicators=True,
                        n_low=int(n_low),
                    ),
                )
        else:
            coeff_hist, k_arr = _run_landau_teacher_projected_history(
                config,
                float(eps) * perturb_template,
                projection_order=int(teacher_proj_Nv),
                history_stride=history_stride,
            )
            train_hist, val_hist = split_history_train_val(coeff_hist, val_fraction)
            append_pairs(
                accum,
                regime_name,
                "train",
                extract_interface_supervised_pairs_from_coeff_history(
                    train_hist,
                    Nv_targets=Nv_targets,
                    Nm=Nm,
                    k_arr=k_arr,
                    vth=1.0,
                    include_global_indicators=True,
                    n_low=int(n_low),
                ),
            )
            append_pairs(
                accum,
                regime_name,
                "val",
                extract_interface_supervised_pairs_from_coeff_history(
                    val_hist,
                    Nv_targets=Nv_targets,
                    Nm=Nm,
                    k_arr=k_arr,
                    vth=1.0,
                    include_global_indicators=True,
                    n_low=int(n_low),
                ),
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
                    "Rebuilding with the current physical-grid teacher configuration is required."
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
        if not np.any(val_mask):
            raise ValueError(
                f"Requested Nv-targets={tuple(int(v) for v in nv_targets)} do not exist in cached val split for regime '{regime}'."
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
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_L: float,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    teacher_proj_Nv: int,
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
    allow_cached_nv_superset: bool = False,
    per_target_projection_orders: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    teacher_proj_Nv_targets = (
        tuple(int(v) + 1 for v in Nv_targets)
        if bool(per_target_projection_orders)
        else None
    )
    cache_metadata = build_dataset_cache_metadata(
        regimes=regimes,
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
        projection_mode="per_target" if bool(per_target_projection_orders) else "shared_max",
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
            per_target_projection_orders=bool(per_target_projection_orders),
        )
    if REGIME_WEAK in active:
        dataset[REGIME_WEAK] = build_nonlinear_landau_regime(
            REGIME_WEAK,
            weak_eps,
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
            per_target_projection_orders=bool(per_target_projection_orders),
        )
    if REGIME_STRONG in active:
        dataset[REGIME_STRONG] = build_nonlinear_landau_regime(
            REGIME_STRONG,
            strong_eps,
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
            per_target_projection_orders=bool(per_target_projection_orders),
        )

    if dataset_cache is not None:
        save_dataset_cache(dataset_cache, dataset, metadata=cache_metadata)
    return dataset


def build_model_inputs(inputs_base: np.ndarray, *, Nm: int, k_scale: float, nv_scale: float) -> np.ndarray:
    inputs = np.asarray(inputs_base, dtype=np.float64).copy()
    k_col = 2 * int(Nm)
    nv_col = k_col + 1
    inputs[:, k_col] = inputs[:, k_col] / float(k_scale)
    inputs[:, nv_col] = inputs[:, nv_col] / float(nv_scale)
    return inputs


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
) -> Tuple[Dict[str, Dict[str, Array]], Dict[str, np.ndarray]]:
    scaled_dataset: Dict[str, Dict[str, np.ndarray]] = {}
    train_inputs_all = []
    train_targets_all = []
    for regime, arrays in dataset_base.items():
        train_inputs = build_model_inputs(arrays["train_inputs_base"], Nm=Nm, k_scale=k_scale, nv_scale=nv_scale)
        val_inputs = build_model_inputs(arrays["val_inputs_base"], Nm=Nm, k_scale=k_scale, nv_scale=nv_scale)
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
    teacher_proj_Nv: int,
    n_low: int,
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
        teacher_backend=str(teacher_backend),
        teacher_Lx=float(teacher_Lx),
        teacher_Nx=int(teacher_Nx),
        teacher_Nv=int(teacher_Nv),
        teacher_vmin=float(teacher_vmin),
        teacher_vmax=float(teacher_vmax),
        teacher_dt=float(teacher_dt),
        teacher_proj_Nv=int(teacher_proj_Nv),
        include_global_indicators=True,
        n_low=int(n_low),
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
    teacher_proj_Nv: int,
    n_low: int,
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
    teacher_proj_Nv: int,
    n_low: int,
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


def evaluate_regime_metrics(
    learned: LearnedInterfaceClosure,
    prepared: Dict[str, Dict[str, Array]],
) -> Dict[str, np.ndarray]:
    metrics: Dict[str, np.ndarray] = {}
    for regime, arrays in prepared.items():
        pred = np.asarray(learned.predict_q_components(arrays["val_inputs"]), dtype=np.float64)
        target = np.asarray(arrays["val_targets"], dtype=np.float64)
        mse = float(np.mean((pred - target) ** 2))
        denom = max(float(np.linalg.norm(target)), 1e-30)
        rel_l2 = float(np.linalg.norm(pred - target) / denom)
        metrics[f"val_q_mse_{regime}"] = np.array([mse], dtype=np.float64)
        metrics[f"val_q_rel_l2_{regime}"] = np.array([rel_l2], dtype=np.float64)
        metrics[f"val_num_samples_{regime}"] = np.array([target.shape[0]], dtype=np.int32)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a shared learned interface closure from a physical-grid Landau teacher")
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
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--k-scale", type=float, default=None)
    parser.add_argument("--nv-scale", type=float, default=None)
    parser.add_argument("--n-low", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--regimes", type=str, default="linear_landau,nonlinear_landau_weak,nonlinear_landau_strong")
    parser.add_argument("--weight-linear", type=float, default=1.0)
    parser.add_argument("--weight-weak", type=float, default=1.0)
    parser.add_argument("--weight-strong", type=float, default=1.0)

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
    if args.checkpoint is None and not bool(args.build_dataset_only):
        raise ValueError("--checkpoint is required unless --build-dataset-only is set")
    if bool(args.build_dataset_only) and args.dataset_cache is None:
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

    teacher_proj_Nv = int(args.teacher_proj_Nv) if args.teacher_proj_Nv is not None else max(Nv_targets) + 1
    if teacher_proj_Nv <= max(Nv_targets):
        raise ValueError("teacher-proj-Nv must exceed every target Nv")

    dataset_base = build_mixed_landau_dataset(
        dataset_cache=args.dataset_cache,
        regimes=regimes,
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
        allow_cached_nv_superset=bool(args.allow_dataset_cache_nv_superset),
        per_target_projection_orders=bool(args.per_target_projection_orders),
    )
    if bool(args.build_dataset_only):
        cache_msg = f"Saved shared dataset cache to {args.dataset_cache}" if args.dataset_cache is not None else "Built dataset in memory"
        print(cache_msg)
        for regime, arrays in dataset_base.items():
            print(f"[data] {regime}: {arrays['train_inputs_base'].shape[0]} training samples cached")
        return

    k_scale = float(args.k_scale) if args.k_scale is not None else choose_k_scale(dataset_base, Nm=args.Nm)
    nv_scale = float(args.nv_scale) if args.nv_scale is not None else choose_nv_scale(dataset_base, Nm=args.Nm)

    prepared, stats = prepare_training_dataset(dataset_base, Nm=args.Nm, k_scale=k_scale, nv_scale=nv_scale)
    for regime, count in summarize_dataset(prepared).items():
        print(f"[data] {regime}: {count} training samples")

    params = init_interface_closure_params(
        jax.random.PRNGKey(args.seed),
        input_dim=2 * int(args.Nm) + 4,
        hidden_width=int(args.hidden_width),
        res_blocks=int(args.res_blocks),
    )
    regime_weights = {
        REGIME_LINEAR: float(args.weight_linear),
        REGIME_WEAK: float(args.weight_weak),
        REGIME_STRONG: float(args.weight_strong),
    }
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
            teacher_backend="physical_grid_cubic_v1",
            teacher_Lx=args.teacher_L,
            teacher_Nx=args.teacher_Nx,
            teacher_Nv=args.teacher_Nv,
            teacher_vmin=args.teacher_vmin,
            teacher_vmax=args.teacher_vmax,
            teacher_dt=args.teacher_dt,
            teacher_proj_Nv=teacher_proj_Nv,
            n_low=args.n_low,
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
            teacher_backend="physical_grid_cubic_v1",
            teacher_Lx=args.teacher_L,
            teacher_Nx=args.teacher_Nx,
            teacher_Nv=args.teacher_Nv,
            teacher_vmin=args.teacher_vmin,
            teacher_vmax=args.teacher_vmax,
            teacher_dt=args.teacher_dt,
            teacher_proj_Nv=teacher_proj_Nv,
            n_low=args.n_low,
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
        teacher_backend="physical_grid_cubic_v1",
        teacher_Lx=args.teacher_L,
        teacher_Nx=args.teacher_Nx,
        teacher_Nv=args.teacher_Nv,
        teacher_vmin=args.teacher_vmin,
        teacher_vmax=args.teacher_vmax,
        teacher_dt=args.teacher_dt,
        teacher_proj_Nv=teacher_proj_Nv,
        n_low=args.n_low,
    )
    val_metrics = evaluate_regime_metrics(learned, prepared)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_learned_interface_closure_npz(args.checkpoint, learned)

    metrics_path = args.checkpoint.with_suffix(".metrics.npz")
    metrics_payload: Dict[str, np.ndarray] = {
        "train_loss": loss_history,
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
        "teacher_backend": np.array(["physical_grid_cubic_v1"], dtype=np.str_),
        "teacher_Lx": np.array([args.teacher_L], dtype=np.float64),
        "teacher_Nx": np.array([args.teacher_Nx], dtype=np.int32),
        "teacher_Nv": np.array([args.teacher_Nv], dtype=np.int32),
        "teacher_vmin": np.array([args.teacher_vmin], dtype=np.float64),
        "teacher_vmax": np.array([args.teacher_vmax], dtype=np.float64),
        "teacher_dt": np.array([args.teacher_dt], dtype=np.float64),
        "teacher_proj_Nv": np.array([teacher_proj_Nv], dtype=np.int32),
        "n_low": np.array([args.n_low], dtype=np.int32),
    }
    metrics_payload.update(val_metrics)
    np.savez(metrics_path, **metrics_payload)

    loss_plot_path = args.loss_plot if args.loss_plot is not None else args.checkpoint.with_suffix(".loss.png")
    save_training_loss_plot(loss_history, loss_plot_path, val_metrics=val_metrics)

    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved loss plot to {loss_plot_path}")
    for key in sorted(val_metrics):
        print(f"{key}: {float(np.asarray(val_metrics[key]).reshape(-1)[0]):.6e}")


if __name__ == "__main__":
    main()
