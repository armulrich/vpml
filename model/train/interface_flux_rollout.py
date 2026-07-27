"""Canonical solver-embedded interface-flux rollout trainer."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import os
from pathlib import Path
import time
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
    LearnedInterfaceClosure,
    e_hat_history_from_a_hat_history,
    init_interface_closure_params,
    learned_boundary_flux_hat,
    learned_interface_q_hat,
    load_learned_interface_closure_npz,
    normalize_teacher_backend_name,
    save_learned_interface_closure_npz,
)
from vpml.linear_landau import LinearLandauConfig, linear_explicit_N_hat, run_linear_landau_cnab2_raw
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    build_cubic_spline_hermite_projection_matrix,
    compute_electric_field_from_distribution,
    extract_interface_supervised_pairs_from_coeff_history,
    gaussian_pdf,
    normalize_density_on_grid,
    project_distribution_snapshot_with_hermite_matrix,
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
INTERFACE_FLUX_ROLLOUT_CACHE_FORMAT = "landau_interface_flux_rollout_reference"
INTERFACE_FLUX_ROLLOUT_TRAINING_MODE = "solver_embedded_interface_flux_rollout"
INTERFACE_FLUX_ROLLOUT_OBJECTIVE = "interface_flux_rollout"
INTERFACE_FLUX_ROLLOUT_LOSS_BACKEND = "regime_balanced_all_k_interface_flux"
INTERFACE_FLUX_PROJECTION_SCHEME = "cubic_spline_uniform_trapezoid"
EXACT_ROLLOUT_PRECISION_FLOAT64 = "float64"
EXACT_ROLLOUT_PRECISION_FLOAT32 = "float32"
ALL_EXACT_ROLLOUT_PRECISIONS = (
    EXACT_ROLLOUT_PRECISION_FLOAT64,
    EXACT_ROLLOUT_PRECISION_FLOAT32,
)
EXACT_TARGET_SAMPLING_CYCLE = "cycle"


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def interface_flux_rollout_coeff_key(projection_order: int) -> str:
    return f"a_hat_ref_order{int(projection_order)}"


def build_interface_flux_rollout_cache_metadata(
    *,
    regimes: Sequence[str],
    teacher_Nx: int,
    teacher_Nv: int,
    projection_quadrature_Nv: int,
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
    Nv_targets: Sequence[int],
    max_projection_order: int,
) -> Dict[str, np.ndarray]:
    return {
        "dataset_format": np.array([INTERFACE_FLUX_ROLLOUT_CACHE_FORMAT], dtype=np.str_),
        "regimes": np.asarray(tuple(regimes), dtype=np.str_),
        "teacher_backend": np.array([GRID_CUBIC_SPLINE_TEACHER_BACKEND], dtype=np.str_),
        "teacher_Nx": np.array([int(teacher_Nx)], dtype=np.int32),
        "teacher_Nv": np.array([int(teacher_Nv)], dtype=np.int32),
        "projection_quadrature_Nv": np.array(
            [int(projection_quadrature_Nv)], dtype=np.int32
        ),
        "projection_quadrature_scheme": np.array(
            [INTERFACE_FLUX_PROJECTION_SCHEME], dtype=np.str_
        ),
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
        "Nv_targets": np.asarray(tuple(int(v) for v in Nv_targets), dtype=np.int32),
        "max_projection_order": np.array([int(max_projection_order)], dtype=np.int32),
    }


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


def append_pairs(
    accum: Dict[str, Dict[str, list]],
    regime: str,
    split: str,
    pairs_by_nv: Dict[int, Dict[str, np.ndarray]],
) -> None:
    for payload in pairs_by_nv.values():
        accum[regime][f"{split}_inputs_base"].append(payload["inputs_base"])
        accum[regime][f"{split}_targets"].append(payload["targets"])


def maxwellian_equilibrium(v: Array) -> Array:
    return normalize_density_on_grid(gaussian_pdf(v, mean=0.0, sigma=1.0), v)


def _projected_history_projector(
    v: Array,
    projection_order: int,
    *,
    projection_quadrature_Nv: int,
    equilibrium: Array,
    projection_matrix: Optional[Array] = None,
    vth: float = 1.0,
):
    matrix = (
        build_cubic_spline_hermite_projection_matrix(
            v,
            int(projection_order),
            int(projection_quadrature_Nv),
            vth=float(vth),
        )
        if projection_matrix is None
        else jnp.asarray(projection_matrix, dtype=jnp.float64)
    )

    def projector(f_state: Array) -> Array:
        return project_distribution_snapshot_with_hermite_matrix(
            f_state,
            matrix,
            equilibrium=equilibrium,
        )

    return projector


def _run_landau_teacher_projected_history(
    config: PhysicalGridVlasovPoissonConfig,
    perturbation_x: Array,
    *,
    projection_order: int,
    projection_quadrature_Nv: int,
    history_stride: int,
    equilibrium: Optional[Array] = None,
    projection_matrix: Optional[Array] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    v = config.v
    equilibrium = (
        maxwellian_equilibrium(v)
        if equilibrium is None
        else jnp.asarray(equilibrium, dtype=jnp.float64)
    )
    f0 = equilibrium[:, None] * (1.0 + jnp.asarray(perturbation_x, dtype=jnp.float64)[None, :])
    raw = run_semilagrangian_vlasov_poisson(
        config,
        f0,
        history_stride=history_stride,
        return_state_history=True,
        history_projector=_projected_history_projector(
            v,
            int(projection_order),
            projection_quadrature_Nv=int(projection_quadrature_Nv),
            equilibrium=equilibrium,
            projection_matrix=projection_matrix,
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
    projection_quadrature_Nv: int,
    history_stride: int,
    equilibrium: Optional[Array] = None,
    projection_matrices: Optional[Dict[int, Array]] = None,
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
            projection_quadrature_Nv=int(projection_quadrature_Nv),
            history_stride=history_stride,
            equilibrium=equilibrium,
            projection_matrix=(
                None
                if projection_matrices is None
                else projection_matrices.get(int(order))
            ),
        )
        histories[int(order)] = coeff_hist
        if k_arr is None:
            k_arr = order_k_arr
        elif not np.array_equal(order_k_arr, k_arr):
            raise ValueError("Projected teacher histories returned inconsistent Fourier grids")
    assert k_arr is not None
    return histories, np.asarray(k_arr, dtype=np.float64)


def _cache_value_mismatch(actual: np.ndarray, expected: np.ndarray) -> bool:
    if actual.shape != expected.shape:
        return True
    if actual.dtype.kind in {"U", "S", "O"} or expected.dtype.kind in {"U", "S", "O"}:
        return not np.array_equal(np.asarray(actual, dtype=np.str_), np.asarray(expected, dtype=np.str_))
    return not np.array_equal(actual, expected)


def load_interface_flux_rollout_reference_cache(
    path: Path,
    *,
    expected_metadata: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    with np.load(path) as data:
        for key, expected in expected_metadata.items():
            if key not in data.files:
                raise ValueError(f"Exact q-rollout cache {path} is missing metadata field '{key}'.")
            actual = np.asarray(data[key])
            if _cache_value_mismatch(actual, np.asarray(expected)):
                raise ValueError(
                    f"Exact q-rollout cache {path} metadata mismatch for '{key}'. "
                    "Rebuilding with the current teacher configuration is required."
                )
        regimes = tuple(str(v) for v in np.asarray(data["regimes"], dtype=np.str_).tolist())
        max_projection_order = int(np.asarray(data["max_projection_order"], dtype=np.int32).reshape(-1)[0])
        coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
        dataset: Dict[str, Dict[str, np.ndarray]] = {}
        for regime in regimes:
            field = f"{regime}_{coeff_key}"
            if field not in data.files:
                raise ValueError(f"Exact q-rollout cache {path} is missing '{field}'.")
            dataset[regime] = {
                coeff_key: np.asarray(data[field], dtype=np.complex128),
            }
        return dataset


def save_interface_flux_rollout_reference_cache(
    path: Path,
    dataset: Dict[str, Dict[str, np.ndarray]],
    *,
    metadata: Dict[str, np.ndarray],
) -> None:
    payload: Dict[str, np.ndarray] = dict(metadata)
    for regime, arrays in dataset.items():
        for key, value in arrays.items():
            payload[f"{regime}_{key}"] = np.asarray(value, dtype=np.complex128)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _run_interface_flux_rollout_projected_history(
    config: PhysicalGridVlasovPoissonConfig,
    perturbation_x: np.ndarray,
    *,
    max_projection_order: int,
    projection_quadrature_Nv: int,
    equilibrium: Optional[Array] = None,
    projection_matrix: Optional[Array] = None,
) -> np.ndarray:
    coeff_histories, _ = _run_landau_teacher_projected_histories(
        config,
        perturbation_x,
        projection_orders=(int(max_projection_order),),
        projection_quadrature_Nv=int(projection_quadrature_Nv),
        history_stride=1,
        equilibrium=equilibrium,
        projection_matrices=(
            None
            if projection_matrix is None
            else {int(max_projection_order): projection_matrix}
        ),
    )
    return np.asarray(coeff_histories[int(max_projection_order)], dtype=np.complex128)


def build_interface_flux_rollout_reference_dataset(
    *,
    dataset_cache: Optional[Path],
    regimes: Sequence[str],
    teacher_Nx: int,
    teacher_Nv: int,
    projection_quadrature_Nv: int,
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
    Nv_targets: Sequence[int],
    min_projection_order: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], int]:
    max_projection_order = max(max(int(v) for v in Nv_targets) + 1, int(min_projection_order or 0))
    cache_metadata = build_interface_flux_rollout_cache_metadata(
        regimes=regimes,
        teacher_Nx=teacher_Nx,
        teacher_Nv=teacher_Nv,
        projection_quadrature_Nv=projection_quadrature_Nv,
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
        Nv_targets=Nv_targets,
        max_projection_order=max_projection_order,
    )
    if dataset_cache is not None and dataset_cache.exists():
        try:
            return (
                load_interface_flux_rollout_reference_cache(
                    dataset_cache,
                    expected_metadata=cache_metadata,
                ),
                max_projection_order,
            )
        except ValueError as exc:
            print(f"[cache] ignoring interface-flux rollout cache {dataset_cache}: {exc}")

    coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
    dataset: Dict[str, Dict[str, np.ndarray]] = {}
    active = tuple(regimes)
    projection_v = jnp.linspace(
        float(teacher_vmin),
        float(teacher_vmax),
        int(teacher_Nv),
        dtype=jnp.float64,
    )
    projection_equilibrium = maxwellian_equilibrium(projection_v)
    projection_matrix = build_cubic_spline_hermite_projection_matrix(
        projection_v,
        int(max_projection_order),
        int(projection_quadrature_Nv),
        vth=1.0,
    )

    if REGIME_LINEAR in active:
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
        cases = []
        for _ in range(int(linear_num_samples)):
            perturb = sample_initial_condition(
                rng,
                x,
                modes=linear_modes,
                eps=float(linear_eps),
            )
            cases.append(
                _run_interface_flux_rollout_projected_history(
                    config,
                    perturb,
                    max_projection_order=max_projection_order,
                    projection_quadrature_Nv=int(projection_quadrature_Nv),
                    equilibrium=projection_equilibrium,
                    projection_matrix=projection_matrix,
                )
            )
        dataset[REGIME_LINEAR] = {coeff_key: np.stack(cases, axis=0)}

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
    for regime, eps_values in (
        (REGIME_WEAK, weak_eps),
        (REGIME_STRONG, strong_eps),
    ):
        if regime not in active:
            continue
        cases = []
        for eps in eps_values:
            cases.append(
                _run_interface_flux_rollout_projected_history(
                    nonlinear_config,
                    float(eps) * perturb_template,
                    max_projection_order=max_projection_order,
                    projection_quadrature_Nv=int(projection_quadrature_Nv),
                    equilibrium=projection_equilibrium,
                    projection_matrix=projection_matrix,
                )
            )
        dataset[regime] = {coeff_key: np.stack(cases, axis=0)}

    if dataset_cache is not None:
        save_interface_flux_rollout_reference_cache(dataset_cache, dataset, metadata=cache_metadata)
    return dataset, max_projection_order


def build_interface_flux_rollout_qpair_dataset(
    exact_dataset: Dict[str, Dict[str, np.ndarray]],
    *,
    max_projection_order: int,
    Nv_targets: Sequence[int],
    Nm: int,
    k_arr: np.ndarray,
    val_fraction: float,
    linear_history_stride: int,
    nonlinear_history_stride: int,
    rollout_horizon: int,
    n_low: int,
    context_mode: str,
    store_training_pairs: bool = True,
    k_scale: Optional[float] = None,
    nv_scale: Optional[float] = None,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
    accum = {
        regime: {
            "train_inputs_base": [],
            "train_targets": [],
            "val_inputs_base": [],
            "val_targets": [],
            "train_case_indices": [],
            "train_time_indices": [],
            "train_k_indices": [],
            "train_target_nvs": [],
            "train_anchor_case_indices": [],
            "train_anchor_time_indices": [],
            "train_anchor_target_nvs": [],
            "val_case_indices": [],
            "val_time_indices": [],
            "val_k_indices": [],
            "val_target_nvs": [],
            "val_anchor_case_indices": [],
            "val_anchor_time_indices": [],
            "val_anchor_target_nvs": [],
        }
        for regime in exact_dataset
    }
    input_sum: Optional[np.ndarray] = None
    input_sum_sq: Optional[np.ndarray] = None
    target_sum: Optional[np.ndarray] = None
    target_sum_sq: Optional[np.ndarray] = None
    input_count = 0
    target_count = 0

    if not bool(store_training_pairs) and (k_scale is None or nv_scale is None):
        raise ValueError("k_scale and nv_scale are required when store_training_pairs=False")

    def sampled_split_indices(history_length: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
        nsteps = int(history_length) - 1
        sampled = np.arange(0, nsteps + 1, max(int(stride), 1), dtype=np.int32)
        if sampled.size == 0 or int(sampled[-1]) != nsteps:
            sampled = np.concatenate([sampled, np.array([nsteps], dtype=np.int32)])
        if sampled.shape[0] <= 1:
            return sampled, sampled
        n_val = max(1, int(round(int(sampled.shape[0]) * float(val_fraction))))
        n_val = min(n_val, int(sampled.shape[0]) - 1)
        return sampled[:-n_val], sampled[-n_val:]

    def append_index_rows(
        *,
        regime: str,
        split: str,
        case_idx: int,
        time_indices: np.ndarray,
        k_count: int,
        include_flattened_k_rows: bool,
    ) -> None:
        if str(context_mode) == "lag1_delta":
            time_indices = np.asarray(time_indices, dtype=np.int32)[1:]
        else:
            time_indices = np.asarray(time_indices, dtype=np.int32)
        if int(time_indices.shape[0]) == 0:
            return
        for target_nv in Nv_targets:
            anchor_rows = int(time_indices.shape[0])
            accum[regime][f"{split}_anchor_case_indices"].append(
                np.full(anchor_rows, int(case_idx), dtype=np.int32)
            )
            accum[regime][f"{split}_anchor_time_indices"].append(time_indices.astype(np.int32))
            accum[regime][f"{split}_anchor_target_nvs"].append(
                np.full(anchor_rows, int(target_nv), dtype=np.int32)
            )
            if bool(include_flattened_k_rows):
                k_indices_one_time = np.arange(1, int(k_count), dtype=np.int32)
                if int(k_indices_one_time.shape[0]) == 0:
                    continue
                case_rows = np.full(
                    int(time_indices.shape[0]) * int(k_indices_one_time.shape[0]),
                    int(case_idx),
                    dtype=np.int32,
                )
                time_rows_base = np.repeat(
                    time_indices, int(k_indices_one_time.shape[0])
                ).astype(np.int32)
                k_rows_base = np.tile(k_indices_one_time, int(time_indices.shape[0])).astype(
                    np.int32
                )
                rows = int(time_rows_base.shape[0])
                accum[regime][f"{split}_case_indices"].append(case_rows)
                accum[regime][f"{split}_time_indices"].append(time_rows_base)
                accum[regime][f"{split}_k_indices"].append(k_rows_base)
                accum[regime][f"{split}_target_nvs"].append(
                    np.full(rows, int(target_nv), dtype=np.int32)
                )

    def accumulate_training_stats(pairs_by_nv: Dict[int, Dict[str, np.ndarray]]) -> None:
        nonlocal input_sum, input_sum_sq, target_sum, target_sum_sq, input_count, target_count
        assert k_scale is not None
        assert nv_scale is not None
        for payload in pairs_by_nv.values():
            inputs = build_model_inputs(
                payload["inputs_base"],
                Nm=int(Nm),
                k_scale=float(k_scale),
                nv_scale=float(nv_scale),
                context_mode=str(context_mode),
                include_global_indicators=True,
            )
            targets = np.asarray(payload["targets"], dtype=np.float64)
            input_sum, input_sum_sq, input_count = _accumulate_feature_moments(
                input_sum,
                input_sum_sq,
                input_count,
                inputs,
            )
            target_sum, target_sum_sq, target_count = _accumulate_feature_moments(
                target_sum,
                target_sum_sq,
                target_count,
                targets,
            )

    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for interface-flux rollout q-pairs")
    for regime, group in exact_dataset.items():
        cases = np.asarray(group[coeff_key], dtype=np.complex128)
        stride = int(linear_history_stride) if regime == REGIME_LINEAR else int(nonlinear_history_stride)
        stride = max(stride, 1)
        for case_idx, case_hist in enumerate(cases):
            train_times, val_times = sampled_split_indices(int(case_hist.shape[0]), stride)
            if int(train_times.shape[0]) > 0:
                train_limit = int(train_times[-1])
                train_times = train_times[(train_times + horizon - 1) <= train_limit]
            for split, original_times in (("train", train_times), ("val", val_times)):
                if int(original_times.shape[0]) == 0:
                    continue
                hist = case_hist[np.asarray(original_times, dtype=np.int32)]
                if int(hist.shape[0]) == 0:
                    continue
                pairs = extract_interface_supervised_pairs_from_coeff_history(
                    hist,
                    Nv_targets=Nv_targets,
                    Nm=Nm,
                    k_arr=k_arr,
                    vth=1.0,
                    include_global_indicators=True,
                    n_low=int(n_low),
                    context_mode=context_mode,
                )
                if bool(store_training_pairs) or split == "val":
                    append_pairs(
                        accum,
                        regime,
                        split,
                        pairs,
                    )
                elif split == "train":
                    accumulate_training_stats(pairs)
                append_index_rows(
                    regime=regime,
                    split=split,
                    case_idx=int(case_idx),
                    time_indices=np.asarray(original_times, dtype=np.int32),
                    k_count=int(case_hist.shape[-1]),
                    include_flattened_k_rows=bool(store_training_pairs) or split == "val",
                )
    raw_base_dim = 2 * int(Nm) + 4
    input_dim = raw_base_dim if str(context_mode) == "none" else 3 * raw_base_dim
    dataset: Dict[str, Dict[str, np.ndarray]] = {}
    for regime, payload in accum.items():
        has_any_rows = any(payload[f"{split}_case_indices"] for split in ("train", "val"))
        if not has_any_rows:
            continue

        def concat_or_empty(key: str, shape_tail: Tuple[int, ...], dtype) -> np.ndarray:
            values = payload[key]
            if values:
                return np.concatenate(values, axis=0).astype(dtype)
            return np.zeros((0, *shape_tail), dtype=dtype)

        dataset[regime] = {
            "train_inputs_base": concat_or_empty("train_inputs_base", (input_dim,), np.float64),
            "train_targets": concat_or_empty("train_targets", (2,), np.float64),
            "val_inputs_base": concat_or_empty("val_inputs_base", (input_dim,), np.float64),
            "val_targets": concat_or_empty("val_targets", (2,), np.float64),
        }
    for regime, arrays in dataset.items():
        for split in ("train", "val"):
            for suffix in ("case_indices", "time_indices", "k_indices", "target_nvs"):
                key = f"{split}_{suffix}"
                arrays[key] = (
                    np.concatenate(accum[regime][key], axis=0).astype(np.int32)
                    if accum[regime][key]
                    else np.zeros((0,), dtype=np.int32)
                )
            for suffix in ("anchor_case_indices", "anchor_time_indices", "anchor_target_nvs"):
                key = f"{split}_{suffix}"
                arrays[key] = (
                    np.concatenate(accum[regime][key], axis=0).astype(np.int32)
                    if accum[regime][key]
                    else np.zeros((0,), dtype=np.int32)
                )
    computed_stats: Optional[Dict[str, np.ndarray]] = None
    if not bool(store_training_pairs):
        if (
            input_sum is None
            or input_sum_sq is None
            or target_sum is None
            or target_sum_sq is None
            or input_count <= 0
            or target_count <= 0
        ):
            raise ValueError("Exact q-rollout could not compute training normalization stats")
        input_mean = input_sum / float(input_count)
        input_var = np.maximum(input_sum_sq / float(input_count) - input_mean * input_mean, 0.0)
        target_mean = target_sum / float(target_count)
        target_var = np.maximum(target_sum_sq / float(target_count) - target_mean * target_mean, 0.0)
        computed_stats = {
            "input_mean": np.asarray(input_mean, dtype=np.float64),
            "input_std": safe_feature_std(np.sqrt(input_var)),
            "target_mean": np.asarray(target_mean, dtype=np.float64),
            "target_std": safe_feature_std(np.sqrt(target_var)),
        }
    return dataset, computed_stats


def interface_flux_rollout_regime_loss_stds(
    exact_dataset: Dict[str, Dict[str, np.ndarray]],
    qpair_dataset: Dict[str, Dict[str, np.ndarray]],
    *,
    max_projection_order: int,
    target_nvs: Sequence[int],
    k_arr: np.ndarray,
    rollout_horizon: int,
    chunk_size: int = 64,
) -> Dict[str, float]:
    """Compute fixed phase-isotropic q-loss scales for each training regime."""
    coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
    k_values = np.asarray(k_arr, dtype=np.float64)
    if int(k_values.shape[0]) <= 1:
        raise ValueError("Exact q-rollout regime scaling requires positive Fourier modes")
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("Exact q-rollout regime scaling requires rollout_horizon > 0")
    chunk_n = max(int(chunk_size), 1)
    offsets = np.arange(horizon, dtype=np.int32)
    positive_k_sq = k_values[1:] ** 2
    scales: Dict[str, float] = {}

    for regime, arrays in qpair_dataset.items():
        if regime not in exact_dataset:
            continue
        histories = np.asarray(exact_dataset[regime][coeff_key])
        anchor_cases = np.asarray(arrays["train_anchor_case_indices"], dtype=np.int32)
        anchor_times = np.asarray(arrays["train_anchor_time_indices"], dtype=np.int32)
        anchor_targets = np.asarray(arrays["train_anchor_target_nvs"], dtype=np.int32)
        sum_abs_sq = 0.0
        value_count = 0

        for target_nv in target_nvs:
            selected = np.flatnonzero(anchor_targets == int(target_nv)).astype(np.int32)
            for start in range(0, int(selected.shape[0]), chunk_n):
                chunk = selected[start : start + chunk_n]
                if int(chunk.shape[0]) == 0:
                    continue
                case_idx = anchor_cases[chunk]
                time_idx = anchor_times[chunk]
                window_times = time_idx[:, None] + offsets[None, :]
                if int(np.max(window_times)) >= int(histories.shape[1]):
                    raise ValueError(
                        f"Exact q-rollout regime scale window exceeds history for {regime}"
                    )
                coeff = histories[
                    case_idx[:, None],
                    window_times,
                    int(target_nv),
                    1:,
                ]
                q_abs_sq = (
                    float(target_nv)
                    * positive_k_sq[None, None, :]
                    * np.abs(coeff) ** 2
                )
                sum_abs_sq += float(np.sum(q_abs_sq, dtype=np.float64))
                value_count += int(q_abs_sq.size)

        if value_count <= 0:
            raise ValueError(f"Exact q-rollout regime '{regime}' has no q targets for scaling")
        component_variance = 0.5 * sum_abs_sq / float(value_count)
        component_std = math.sqrt(max(component_variance, 0.0))
        if not math.isfinite(component_std) or component_std <= 1e-12:
            raise ValueError(
                f"Exact q-rollout regime '{regime}' has invalid q-loss scale {component_std}"
            )
        scales[str(regime)] = float(component_std)
    return scales


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


def phase_isotropic_complex_training_stats(
    stats: Dict[str, np.ndarray],
    *,
    Nm: int,
    context_mode: str,
) -> Dict[str, np.ndarray]:
    """Return phase-isotropic normalization for complex closure features and q."""
    input_mean = np.asarray(stats["input_mean"], dtype=np.float64).copy()
    input_std = np.asarray(stats["input_std"], dtype=np.float64).copy()
    target_mean = np.asarray(stats["target_mean"], dtype=np.float64).copy()
    target_std = np.asarray(stats["target_std"], dtype=np.float64).copy()
    base_dim = 2 * int(Nm) + 4
    if str(context_mode) == "none":
        block_offsets = (0,)
    elif str(context_mode) == "lag1_delta":
        block_offsets = (0, base_dim, 2 * base_dim)
    else:
        raise ValueError(f"Unsupported context_mode={context_mode!r}")

    for offset in block_offsets:
        for mode_idx in range(int(Nm)):
            real_idx = int(offset + mode_idx)
            imag_idx = int(offset + int(Nm) + mode_idx)
            second_moment = 0.5 * (
                input_std[real_idx] ** 2
                + input_mean[real_idx] ** 2
                + input_std[imag_idx] ** 2
                + input_mean[imag_idx] ** 2
            )
            shared_std = float(safe_feature_std(np.array([math.sqrt(second_moment)]))[0])
            input_mean[[real_idx, imag_idx]] = 0.0
            input_std[[real_idx, imag_idx]] = shared_std

    target_second_moment = 0.5 * float(
        np.sum(target_std * target_std + target_mean * target_mean)
    )
    shared_target_std = float(
        safe_feature_std(np.array([math.sqrt(target_second_moment)]))[0]
    )
    target_mean[:] = 0.0
    target_std[:] = shared_target_std
    return {
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def exact_rollout_precision_dtypes(precision: str) -> Tuple[object, object]:
    mode = str(precision)
    if mode == EXACT_ROLLOUT_PRECISION_FLOAT64:
        return jnp.float64, jnp.complex128
    if mode == EXACT_ROLLOUT_PRECISION_FLOAT32:
        return jnp.float32, jnp.complex64
    raise ValueError(
        f"exact_rollout_precision must be one of {ALL_EXACT_ROLLOUT_PRECISIONS!r}, "
        f"got {precision!r}"
    )


def exact_rollout_numpy_complex_dtype(complex_dtype: object) -> np.dtype:
    dtype = np.dtype(complex_dtype)
    if dtype == np.dtype(np.complex64):
        return np.dtype(np.complex64)
    if dtype == np.dtype(np.complex128):
        return np.dtype(np.complex128)
    raise ValueError(f"unsupported exact rollout complex dtype {complex_dtype!r}")


def exact_rollout_numpy_real_dtype(complex_dtype: object) -> np.dtype:
    complex_np_dtype = exact_rollout_numpy_complex_dtype(complex_dtype)
    return np.dtype(np.float32 if complex_np_dtype == np.dtype(np.complex64) else np.float64)


def cast_learned_closure_for_rollout(
    learned: LearnedInterfaceClosure,
    *,
    precision: str,
) -> LearnedInterfaceClosure:
    real_dtype, _ = exact_rollout_precision_dtypes(str(precision))
    if str(precision) == EXACT_ROLLOUT_PRECISION_FLOAT64:
        return learned
    params = jax.tree_util.tree_map(lambda value: jnp.asarray(value, dtype=real_dtype), learned.params)
    return replace(
        learned,
        params=params,
        input_mean=jnp.asarray(learned.input_mean, dtype=real_dtype),
        input_std=jnp.asarray(learned.input_std, dtype=real_dtype),
        target_mean=jnp.asarray(learned.target_mean, dtype=real_dtype),
        target_std=jnp.asarray(learned.target_std, dtype=real_dtype),
    )


def default_exact_k_scale(k_arr: np.ndarray) -> float:
    k_nonzero = np.asarray(k_arr, dtype=np.float64)[1:]
    if int(k_nonzero.shape[0]) == 0:
        return 1.0
    return max(float(np.max(np.abs(k_nonzero))), 1.0)


def prepare_validation_dataset_from_stats(
    dataset_base: Dict[str, Dict[str, np.ndarray]],
    *,
    Nm: int,
    k_scale: float,
    nv_scale: float,
    context_mode: str,
    stats: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, Array]]:
    prepared: Dict[str, Dict[str, Array]] = {}
    target_std_safe = np.maximum(np.asarray(stats["target_std"], dtype=np.float64), 1e-12)[None, :]
    target_mean_row = np.asarray(stats["target_mean"], dtype=np.float64)[None, :]
    input_dim = int(np.asarray(stats["input_mean"]).shape[0])
    for regime, arrays in dataset_base.items():
        val_inputs_base = np.asarray(arrays.get("val_inputs_base", np.zeros((0, input_dim), dtype=np.float64)), dtype=np.float64)
        val_targets = np.asarray(arrays.get("val_targets", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)
        if val_inputs_base.size:
            val_inputs = build_model_inputs(
                val_inputs_base,
                Nm=Nm,
                k_scale=k_scale,
                nv_scale=nv_scale,
                context_mode=context_mode,
            )
        else:
            val_inputs = np.zeros((0, input_dim), dtype=np.float64)
        prepared[regime] = {
            "train_inputs": jnp.zeros((0, input_dim), dtype=jnp.float64),
            "train_targets": jnp.zeros((0, 2), dtype=jnp.float64),
            "train_targets_std": jnp.zeros((0, 2), dtype=jnp.float64),
            "val_inputs": jnp.asarray(val_inputs, dtype=jnp.float64),
            "val_targets": jnp.asarray(val_targets, dtype=jnp.float64),
            "val_targets_std": jnp.asarray((val_targets - target_mean_row) / target_std_safe, dtype=jnp.float64),
        }
    return prepared


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


def build_learned_interface_closure(
    *,
    params: Dict[str, Array],
    Nm: int,
    k_scale: float,
    nv_scale: float,
    stats: Dict[str, np.ndarray],
    hidden_width: int,
    res_blocks: int,
    equilibrium_centered: bool = False,
    complex_normalization_mode: str = "componentwise",
    translation_augmented: bool = False,
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
    rollout_horizon: int,
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
        equilibrium_centered=bool(equilibrium_centered),
        complex_normalization_mode=str(complex_normalization_mode),
        translation_augmented=bool(translation_augmented),
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
        projection_quadrature_Nv=(
            None
            if projection_quadrature_Nv is None
            else int(projection_quadrature_Nv)
        ),
        include_global_indicators=True,
        n_low=int(n_low),
        training_mode=INTERFACE_FLUX_ROLLOUT_TRAINING_MODE,
        train_objective=INTERFACE_FLUX_ROLLOUT_OBJECTIVE,
        context_mode="none",
        context_lags=0,
        base_input_dim=2 * int(Nm) + 4,
        rollout_horizon=int(rollout_horizon),
        loss_backend=INTERFACE_FLUX_ROLLOUT_LOSS_BACKEND,
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


def prepare_interface_flux_rollout_sampling_state(
    reference_dataset: Dict[str, Dict[str, np.ndarray]],
    interface_flux_dataset: Dict[str, Dict[str, np.ndarray]],
    *,
    max_projection_order: int,
    target_nvs: Sequence[int],
    history_dtype: object = np.complex128,
) -> Dict[str, Dict[str, object]]:
    coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
    history_np_dtype = exact_rollout_numpy_complex_dtype(history_dtype)
    sampling_state: Dict[str, Dict[str, object]] = {}
    for regime, arrays in interface_flux_dataset.items():
        train_target_nvs = np.asarray(arrays["train_target_nvs"], dtype=np.int32)
        target_pools = {
            int(target_nv): np.flatnonzero(train_target_nvs == int(target_nv)).astype(np.int32)
            for target_nv in target_nvs
        }
        train_anchor_target_nvs = np.asarray(arrays["train_anchor_target_nvs"], dtype=np.int32)
        anchor_target_pools = {
            int(target_nv): np.flatnonzero(train_anchor_target_nvs == int(target_nv)).astype(
                np.int32
            )
            for target_nv in target_nvs
        }
        state: Dict[str, object] = {
            "histories": np.asarray(
                reference_dataset[regime][coeff_key],
                dtype=history_np_dtype,
            ),
            "train_case_indices": np.asarray(arrays["train_case_indices"], dtype=np.int32),
            "train_time_indices": np.asarray(arrays["train_time_indices"], dtype=np.int32),
            "train_k_indices": np.asarray(arrays["train_k_indices"], dtype=np.int32),
            "target_pools": target_pools,
            "train_anchor_case_indices": np.asarray(
                arrays["train_anchor_case_indices"], dtype=np.int32
            ),
            "train_anchor_time_indices": np.asarray(
                arrays["train_anchor_time_indices"], dtype=np.int32
            ),
            "anchor_target_pools": anchor_target_pools,
        }
        sampling_state[regime] = state
    return sampling_state


def select_interface_flux_rollout_regime_indices(
    sampling_state: Dict[str, Dict[str, object]],
    *,
    regime: str,
    target_nv: int,
    batch_size: int,
    rng: np.random.Generator,
    all_k_loss: bool = False,
) -> np.ndarray:
    regime_state = sampling_state[regime]
    target_pools = (
        regime_state["anchor_target_pools"] if bool(all_k_loss) else regime_state["target_pools"]
    )
    target_pool = target_pools[int(target_nv)]
    if int(target_pool.shape[0]) <= 0:
        raise ValueError(
            f"Exact q-rollout regime '{regime}' has no valid q-pairs for Nv={int(target_nv)}"
        )
    batch_n = int(min(int(batch_size), int(target_pool.shape[0])))
    return np.asarray(
        target_pool[rng.integers(0, int(target_pool.shape[0]), size=batch_n, endpoint=False)],
        dtype=np.int32,
    )


def translate_interface_flux_rollout_anchor_batch(
    anchor_stencils: np.ndarray,
    ref_q_windows: np.ndarray,
    *,
    k_arr: np.ndarray,
    shifts: np.ndarray,
    k_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Translate complete exact-q anchors using the repository rFFT convention."""
    stencils = np.asarray(anchor_stencils)
    q_windows = np.asarray(ref_q_windows)
    k_values = np.asarray(k_arr, dtype=np.float64)
    shift_values = np.asarray(shifts, dtype=np.float64)
    if stencils.ndim != 4:
        raise ValueError("anchor_stencils must have shape (B,S,Nv,Nk)")
    if shift_values.shape != (int(stencils.shape[0]),):
        raise ValueError("shifts must have shape (B,) matching anchor_stencils")
    if k_values.shape != (int(stencils.shape[-1]),):
        raise ValueError("k_arr must have shape (Nk,) matching anchor_stencils")

    phases = np.exp(-1j * shift_values[:, None] * k_values[None, :])
    translated_stencils = stencils * phases[:, None, None, :]
    if q_windows.ndim == 3:
        if tuple(q_windows.shape[::2]) != (int(stencils.shape[0]), int(stencils.shape[-1])):
            raise ValueError("all-k ref_q_windows must have shape (B,H,Nk)")
        translated_q = q_windows * phases[:, None, :]
    elif q_windows.ndim == 2:
        if k_indices is None:
            raise ValueError("selected-k ref_q_windows require k_indices")
        selected_k = np.asarray(k_indices, dtype=np.int32)
        if selected_k.shape != (int(stencils.shape[0]),):
            raise ValueError("k_indices must have shape (B,)")
        translated_q = q_windows * phases[np.arange(int(stencils.shape[0])), selected_k][:, None]
    else:
        raise ValueError("ref_q_windows must have shape (B,H) or (B,H,Nk)")
    return translated_stencils, translated_q


def sample_interface_flux_rollout_regime_batch(
    sampling_state: Dict[str, Dict[str, object]],
    *,
    regime: str,
    target_nv: int,
    rollout_horizon: int,
    batch_size: int,
    k_arr: np.ndarray,
    rng: np.random.Generator,
    complex_dtype: object = jnp.complex128,
    all_k_loss: bool = False,
    selected_indices: Optional[np.ndarray] = None,
    translation_augmentation: bool = False,
    domain_length: Optional[float] = None,
) -> Dict[str, Array]:
    regime_state = sampling_state[regime]
    histories = regime_state["histories"]
    complex_np_dtype = exact_rollout_numpy_complex_dtype(complex_dtype)
    real_np_dtype = exact_rollout_numpy_real_dtype(complex_dtype)
    real_jax_dtype = jnp.float32 if real_np_dtype == np.dtype(np.float32) else jnp.float64
    k_arr_np = np.asarray(k_arr, dtype=real_np_dtype)
    if selected_indices is None:
        selected = select_interface_flux_rollout_regime_indices(
            sampling_state,
            regime=regime,
            target_nv=int(target_nv),
            batch_size=int(batch_size),
            rng=rng,
            all_k_loss=bool(all_k_loss),
        )
    else:
        selected = np.asarray(selected_indices, dtype=np.int32)
        if selected.ndim != 1 or int(selected.shape[0]) <= 0:
            raise ValueError("selected_indices must be a nonempty 1D array")
    batch_n = int(selected.shape[0])
    if bool(all_k_loss):
        case_idx = regime_state["train_anchor_case_indices"][selected]
        time_idx = regime_state["train_anchor_time_indices"][selected]
        k_indices = np.zeros((batch_n,), dtype=np.int32)
    else:
        case_idx = regime_state["train_case_indices"][selected]
        time_idx = regime_state["train_time_indices"][selected]
        k_indices = regime_state["train_k_indices"][selected]
    stencil_times = np.stack(
        (
            time_idx,
            np.maximum(time_idx - 1, 0),
            np.maximum(time_idx - 2, 0),
        ),
        axis=1,
    )
    target_nv_i = int(target_nv)
    stencils = histories[case_idx[:, None], stencil_times, :target_nv_i, :]
    nk = int(histories.shape[-1])
    if nk <= 1:
        raise ValueError("Exact q-rollout requires at least one nonzero Fourier mode")
    offsets = np.arange(int(rollout_horizon), dtype=np.int32)
    window_times = time_idx[:, None] + offsets[None, :]
    if bool(all_k_loss):
        q_coeff = histories[case_idx[:, None], window_times, target_nv_i, :]
        q_windows = (
            -1j
            * k_arr_np[None, None, :]
            * math.sqrt(float(target_nv_i))
            * q_coeff
        )
    else:
        q_coeff = histories[case_idx[:, None], window_times, target_nv_i, k_indices[:, None]]
        q_windows = (
            -1j
            * k_arr_np[k_indices][:, None]
            * math.sqrt(float(target_nv_i))
            * q_coeff
        )
    if bool(translation_augmentation):
        if domain_length is None or float(domain_length) <= 0.0:
            raise ValueError("translation augmentation requires a positive domain_length")
        shifts = rng.uniform(0.0, float(domain_length), size=batch_n)
        stencils, q_windows = translate_interface_flux_rollout_anchor_batch(
            stencils,
            q_windows,
            k_arr=k_arr_np,
            shifts=shifts,
            k_indices=k_indices,
        )
    batch = {
        "anchor_stencils": jnp.asarray(stencils, dtype=complex_dtype),
        "ref_q_windows": jnp.asarray(q_windows, dtype=complex_dtype),
        "k_indices": jnp.asarray(k_indices, dtype=jnp.int32),
    }
    return batch


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
) -> Tuple[object, Sequence[str]]:
    active_regimes = tuple(regime for regime in train_regimes if regime in regime_weights)
    weights = np.asarray([float(regime_weights[regime]) for regime in active_regimes], dtype=np.float64)
    weights = weights / np.sum(weights)
    weight_arr = jnp.asarray(weights, dtype=jnp.float64)
    target_nvs = tuple(int(v) for v in Nv_targets)
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
            Nx=int(teacher_Nx),
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
            Nx=int(teacher_Nx),
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


def interface_flux_cutoff_for_step(
    target_nvs: Sequence[int],
    global_step: int,
) -> int:
    targets = tuple(int(value) for value in target_nvs)
    if not targets:
        raise ValueError("target_nvs must not be empty")
    if int(global_step) < 0:
        raise ValueError("global_step must be nonnegative")
    return targets[int(global_step) % len(targets)]


def train_with_interface_flux_rollout_minibatch_loss(
    params: Dict[str, Array],
    reference_dataset: Dict[str, Dict[str, np.ndarray]],
    interface_flux_dataset: Dict[str, Dict[str, np.ndarray]],
    batch_loss_fn,
    *,
    max_projection_order: int,
    active_regimes: Sequence[str],
    k_arr: np.ndarray,
    epochs: int,
    learning_rate: float,
    grad_clip: Optional[float],
    log_every: int,
    batch_size: int,
    steps_per_epoch: int,
    rollout_horizon: int,
    seed: int,
    log_components: Sequence[str] = (),
    profile_trace_dir: Optional[Path] = None,
    profile_train_steps: int = 0,
    profile_skip_steps: int = 1,
) -> Tuple[Dict[str, Array], Dict[str, np.ndarray]]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive for interface-flux rollout training")
    if int(steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be positive for interface-flux rollout training")

    optimizer_state = adam_init(params)
    history = {
        key: np.zeros((int(epochs),), dtype=np.float64)
        for key in ("total", "q")
    }

    def make_train_step(target_batch_loss_fn):
        @jax.jit
        def train_step(
            current_params: Dict[str, Array],
            current_state: Dict[str, object],
            regime_batches: Dict[str, Dict[str, Array]],
        ) -> Tuple[Dict[str, Array], Dict[str, object], Dict[str, Array], Array]:
            (loss, aux), grads = jax.value_and_grad(
                target_batch_loss_fn,
                has_aux=True,
            )(current_params, regime_batches)
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

            next_params, next_state = jax.lax.cond(
                all_finite,
                apply_update,
                keep_state,
                operand=None,
            )
            return next_params, next_state, aux, all_finite

        return train_step

    target_nvs = tuple(int(value) for value in getattr(batch_loss_fn, "target_nvs", ()))
    target_loss_fns = getattr(batch_loss_fn, "target_loss_fns", None)
    if not target_nvs or not isinstance(target_loss_fns, dict):
        raise ValueError("Canonical interface-flux loss must provide per-cutoff loss functions")
    train_steps = {
        target_nv: make_train_step(target_loss_fns[target_nv])
        for target_nv in target_nvs
    }

    rng = np.random.default_rng(int(seed))
    rollout_precision = str(
        getattr(batch_loss_fn, "rollout_precision", EXACT_ROLLOUT_PRECISION_FLOAT64)
    )
    _, batch_complex_dtype = exact_rollout_precision_dtypes(rollout_precision)
    history_complex_dtype = exact_rollout_numpy_complex_dtype(batch_complex_dtype)
    translation_augmented = bool(
        getattr(batch_loss_fn, "translation_augmented", False)
    )
    translation_domain_length = getattr(batch_loss_fn, "teacher_Lx", None)
    sampling_state = prepare_interface_flux_rollout_sampling_state(
        reference_dataset,
        interface_flux_dataset,
        max_projection_order=int(max_projection_order),
        target_nvs=target_nvs,
        history_dtype=history_complex_dtype,
    )

    profile_steps = max(int(profile_train_steps), 0)
    profile_skip = max(int(profile_skip_steps), 0)
    profile_dir = (
        None
        if profile_trace_dir is None or profile_steps <= 0
        else Path(profile_trace_dir)
    )
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[profile] interface-flux rollout tracing {profile_steps} train step(s) "
            f"after {profile_skip} warmup step(s) to {profile_dir}"
        )
    profiled_seconds: List[float] = []

    for epoch in range(int(epochs)):
        running = {
            key: jnp.asarray(0.0, dtype=jnp.float64)
            for key in history
        }
        for step_idx in range(int(steps_per_epoch)):
            global_step = int(epoch) * int(steps_per_epoch) + int(step_idx)
            target_nv = interface_flux_cutoff_for_step(target_nvs, global_step)
            selected_by_regime = {
                regime: select_interface_flux_rollout_regime_indices(
                    sampling_state,
                    regime=regime,
                    target_nv=target_nv,
                    batch_size=int(batch_size),
                    rng=rng,
                    all_k_loss=True,
                )
                for regime in active_regimes
            }
            regime_batches = {
                regime: sample_interface_flux_rollout_regime_batch(
                    sampling_state,
                    regime=regime,
                    target_nv=target_nv,
                    rollout_horizon=int(rollout_horizon),
                    batch_size=int(batch_size),
                    k_arr=np.asarray(k_arr, dtype=np.float64),
                    rng=rng,
                    complex_dtype=batch_complex_dtype,
                    all_k_loss=True,
                    selected_indices=selected_by_regime[regime],
                    translation_augmentation=translation_augmented,
                    domain_length=(
                        float(translation_domain_length)
                        if translation_augmented
                        and translation_domain_length is not None
                        else None
                    ),
                )
                for regime in active_regimes
            }

            should_profile = (
                profile_dir is not None
                and profile_skip <= global_step < profile_skip + profile_steps
            )
            started = time.perf_counter() if should_profile else 0.0
            if should_profile:
                with jax.profiler.trace(str(profile_dir)):
                    params, optimizer_state, aux, all_finite = train_steps[target_nv](
                        params,
                        optimizer_state,
                        regime_batches,
                    )
                    jax.block_until_ready(aux["total"])
                profiled_seconds.append(time.perf_counter() - started)
            else:
                params, optimizer_state, aux, all_finite = train_steps[target_nv](
                    params,
                    optimizer_state,
                    regime_batches,
                )
            if not bool(all_finite):
                raise FloatingPointError(
                    "interface-flux rollout produced non-finite loss/gradients at "
                    f"epoch {epoch + 1}, step {step_idx + 1}; "
                    "reduce TRAIN_LR, TRAIN_BATCH_SIZE, TRAIN_ROLLOUT_HORIZON, "
                    "or TRAIN_GRAD_CLIP."
                )
            for key in running:
                running[key] = running[key] + aux[key]

        for key in history:
            history[key][epoch] = float(running[key] / float(steps_per_epoch))
        if (
            epoch == 0
            or (epoch + 1) % max(int(log_every), 1) == 0
            or epoch + 1 == int(epochs)
        ):
            print(
                _format_train_loss_log(
                    epoch=epoch,
                    epochs=epochs,
                    history=history,
                    components=log_components,
                )
            )

    if profiled_seconds:
        mean_seconds = float(np.mean(np.asarray(profiled_seconds, dtype=np.float64)))
        print(
            f"[profile] interface-flux rollout traced {len(profiled_seconds)} step(s); "
            f"mean profiled wall time={mean_seconds:.3f}s/step"
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


def _load_init_checkpoint_for_interface_closure(
    init_checkpoint: Path,
    *,
    Nm: int,
    hidden_width: int,
    res_blocks: int,
    Nv_targets: Sequence[int],
    context_mode: str,
    equilibrium_centered: Optional[bool] = None,
    complex_normalization_mode: Optional[str] = None,
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
    if (
        equilibrium_centered is not None
        and bool(learned.equilibrium_centered) != bool(equilibrium_centered)
    ):
        raise ValueError(
            "--init-checkpoint equilibrium_centered metadata does not match the requested closure behavior"
        )
    if (
        complex_normalization_mode is not None
        and str(learned.complex_normalization_mode) != str(complex_normalization_mode)
    ):
        raise ValueError(
            "--init-checkpoint complex_normalization_mode metadata does not match the requested normalization"
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


CANONICAL_NV_TARGETS = (6, 7, 12, 20, 36, 64)
CANONICAL_REGIMES = (REGIME_LINEAR, REGIME_WEAK, REGIME_STRONG)
CANONICAL_WEAK_EPS = (0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18)
CANONICAL_STRONG_EPS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65)
CANONICAL_METADATA_SCHEMA_VERSION = 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the canonical solver-embedded interface-flux closure."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-cache", type=Path, default=None)
    parser.add_argument("--loss-plot", type=Path, default=None)
    parser.add_argument("--build-dataset-only", action="store_true")
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument(
        "--precision",
        choices=ALL_EXACT_ROLLOUT_PRECISIONS,
        default=EXACT_ROLLOUT_PRECISION_FLOAT32,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps-per-epoch", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T-final", type=float, default=60.0)
    parser.add_argument("--Nm", type=int, default=6)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--n-low", type=int, default=2)
    parser.add_argument("--k-scale", type=float, default=None)
    parser.add_argument("--nv-scale", type=float, default=None)
    parser.add_argument("--teacher-Nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", type=int, default=512)
    parser.add_argument("--projection-quadrature-Nv", type=int, default=None)
    parser.add_argument("--teacher-L", type=float, default=4.0 * math.pi)
    parser.add_argument("--teacher-vmin", type=float, default=-8.0)
    parser.add_argument("--teacher-vmax", type=float, default=8.0)
    parser.add_argument("--teacher-dt", type=float, default=1e-2)
    parser.add_argument("--teacher-poisson-sign", type=float, default=1.0)
    parser.add_argument("--linear-eps", type=float, default=1e-2)
    parser.add_argument("--linear-modes", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--linear-num-samples", type=int, default=8)
    parser.add_argument("--linear-seed", type=int, default=0)
    parser.add_argument("--history-stride", type=int, default=20)
    parser.add_argument("--nonlinear-k0", type=float, default=0.5)
    parser.add_argument("--profile-trace-dir", type=Path, default=None)
    parser.add_argument("--profile-train-steps", type=int, default=0)
    parser.add_argument("--profile-skip-steps", type=int, default=1)
    return parser


def _canonical_metrics_payload(
    *,
    args: argparse.Namespace,
    loss_history: np.ndarray,
    component_history: Dict[str, np.ndarray],
    stats: Dict[str, np.ndarray],
    k_scale: float,
    nv_scale: float,
    teacher_projection_order: int,
    projection_quadrature_Nv: int,
    regime_loss_stds: Dict[str, float],
    val_metrics: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {
        "metadata_schema_version": np.array(
            [CANONICAL_METADATA_SCHEMA_VERSION], dtype=np.int32
        ),
        "training_mode": np.array(
            [INTERFACE_FLUX_ROLLOUT_TRAINING_MODE], dtype=np.str_
        ),
        "train_objective": np.array(
            [INTERFACE_FLUX_ROLLOUT_OBJECTIVE], dtype=np.str_
        ),
        "loss_backend": np.array(
            [INTERFACE_FLUX_ROLLOUT_LOSS_BACKEND], dtype=np.str_
        ),
        "train_loss": np.asarray(loss_history, dtype=np.float64),
        "train_loss_interface_flux": np.asarray(
            component_history["q"], dtype=np.float64
        ),
        "Nm": np.array([args.Nm], dtype=np.int32),
        "hidden_width": np.array([args.hidden_width], dtype=np.int32),
        "res_blocks": np.array([args.res_blocks], dtype=np.int32),
        "k_scale": np.array([k_scale], dtype=np.float64),
        "nv_scale": np.array([nv_scale], dtype=np.float64),
        "Nv_targets": np.asarray(CANONICAL_NV_TARGETS, dtype=np.int32),
        "regimes": np.asarray(CANONICAL_REGIMES, dtype=np.str_),
        "regime_weights": np.full((len(CANONICAL_REGIMES),), 1.0 / 3.0),
        "input_mean": np.asarray(stats["input_mean"], dtype=np.float64),
        "input_std": np.asarray(stats["input_std"], dtype=np.float64),
        "target_mean": np.asarray(stats["target_mean"], dtype=np.float64),
        "target_std": np.asarray(stats["target_std"], dtype=np.float64),
        "teacher_backend": np.array(
            [GRID_CUBIC_SPLINE_TEACHER_BACKEND], dtype=np.str_
        ),
        "teacher_Lx": np.array([args.teacher_L], dtype=np.float64),
        "teacher_Nx": np.array([args.teacher_Nx], dtype=np.int32),
        "teacher_Nv": np.array([args.teacher_Nv], dtype=np.int32),
        "teacher_vmin": np.array([args.teacher_vmin], dtype=np.float64),
        "teacher_vmax": np.array([args.teacher_vmax], dtype=np.float64),
        "teacher_dt": np.array([args.teacher_dt], dtype=np.float64),
        "teacher_proj_Nv": np.array(
            [teacher_projection_order], dtype=np.int32
        ),
        "projection_quadrature_Nv": np.array(
            [int(projection_quadrature_Nv)], dtype=np.int32
        ),
        "projection_quadrature_scheme": np.array(
            [INTERFACE_FLUX_PROJECTION_SCHEME], dtype=np.str_
        ),
        "T_final": np.array([args.T_final], dtype=np.float64),
        "n_low": np.array([args.n_low], dtype=np.int32),
        "context_mode": np.array(["none"], dtype=np.str_),
        "rollout_horizon": np.array([args.rollout_horizon], dtype=np.int32),
        "precision": np.array([args.precision], dtype=np.str_),
        "target_sampling": np.array([EXACT_TARGET_SAMPLING_CYCLE], dtype=np.str_),
        "all_positive_k": np.array([True], dtype=np.bool_),
        "regime_balanced": np.array([True], dtype=np.bool_),
        "equilibrium_centered": np.array([True], dtype=np.bool_),
        "complex_normalization_mode": np.array(
            ["phase_isotropic"], dtype=np.str_
        ),
        "translation_augmented": np.array([True], dtype=np.bool_),
        "interface_flux_regime_loss_regimes": np.asarray(
            CANONICAL_REGIMES, dtype=np.str_
        ),
        "interface_flux_regime_loss_stds": np.asarray(
            [regime_loss_stds[regime] for regime in CANONICAL_REGIMES],
            dtype=np.float64,
        ),
    }
    payload.update(val_metrics)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    print_jax_runtime_summary(jax, context="interface-flux training")
    args = build_arg_parser().parse_args(argv)
    if int(args.rollout_horizon) <= 0:
        raise ValueError("--rollout-horizon must be positive")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.steps_per_epoch) <= 0:
        raise ValueError("--steps-per-epoch must be positive")
    if float(args.T_final) <= 0.0:
        raise ValueError("--T-final must be positive")
    projection_quadrature_Nv = (
        int(args.teacher_Nv)
        if args.projection_quadrature_Nv is None
        else int(args.projection_quadrature_Nv)
    )
    if projection_quadrature_Nv <= 3:
        raise ValueError("--projection-quadrature-Nv must exceed three")
    if any(target_nv < int(args.Nm) for target_nv in CANONICAL_NV_TARGETS):
        raise ValueError(
            f"Nm={args.Nm} exceeds the smallest canonical cutoff "
            f"{min(CANONICAL_NV_TARGETS)}"
        )

    linear_modes = parse_float_tuple(args.linear_modes)
    reference, max_projection_order = build_interface_flux_rollout_reference_dataset(
        dataset_cache=args.dataset_cache,
        regimes=CANONICAL_REGIMES,
        teacher_Nx=args.teacher_Nx,
        teacher_Nv=args.teacher_Nv,
        projection_quadrature_Nv=projection_quadrature_Nv,
        teacher_L=args.teacher_L,
        teacher_vmin=args.teacher_vmin,
        teacher_vmax=args.teacher_vmax,
        teacher_dt=args.teacher_dt,
        linear_T=args.T_final,
        linear_eps=args.linear_eps,
        linear_modes=linear_modes,
        linear_num_samples=args.linear_num_samples,
        linear_seed=args.linear_seed,
        linear_poisson_sign=args.teacher_poisson_sign,
        nonlinear_T=args.T_final,
        nonlinear_k0=args.nonlinear_k0,
        nonlinear_poisson_sign=args.teacher_poisson_sign,
        weak_eps=CANONICAL_WEAK_EPS,
        strong_eps=CANONICAL_STRONG_EPS,
        Nv_targets=CANONICAL_NV_TARGETS,
        min_projection_order=None,
    )
    coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
    print(
        "[data] physical teacher and projection grids: "
        f"teacher_Nv={int(args.teacher_Nv)} "
        f"projection_quadrature_Nv={projection_quadrature_Nv}"
    )
    for regime in CANONICAL_REGIMES:
        cases = np.asarray(reference[regime][coeff_key])
        print(
            f"[data] interface-flux rollout {regime}: "
            f"cases={cases.shape[0]} history={cases.shape[1]} "
            f"order={cases.shape[2]}"
        )
    if args.build_dataset_only:
        if args.dataset_cache is None:
            raise ValueError("--build-dataset-only requires --dataset-cache")
        print(f"Saved interface-flux reference cache to {args.dataset_cache}")
        return

    k_arr = np.asarray(
        FourierHermiteIMEX(
            Nx=int(args.teacher_Nx),
            Nv=max(CANONICAL_NV_TARGETS),
            Lx=float(args.teacher_L),
            dt=float(args.teacher_dt),
            vth=1.0,
            dealias_23=False,
            closure=None,
        ).k_arr,
        dtype=np.float64,
    )
    init_params: Optional[Dict[str, Array]] = None
    init_stats: Optional[Dict[str, np.ndarray]] = None
    init_k_scale: Optional[float] = None
    init_nv_scale: Optional[float] = None
    if args.init_checkpoint is not None:
        init_params, init_stats, init_k_scale, init_nv_scale = (
            _load_init_checkpoint_for_interface_closure(
                args.init_checkpoint,
                Nm=args.Nm,
                hidden_width=args.hidden_width,
                res_blocks=args.res_blocks,
                Nv_targets=CANONICAL_NV_TARGETS,
                context_mode="none",
                equilibrium_centered=True,
                complex_normalization_mode="phase_isotropic",
            )
        )
        print(
            "[train] initialized interface-flux parameters from "
            f"{args.init_checkpoint}"
        )
    k_scale = (
        float(args.k_scale)
        if args.k_scale is not None
        else float(init_k_scale)
        if init_k_scale is not None
        else default_exact_k_scale(k_arr)
    )
    nv_scale = (
        float(args.nv_scale)
        if args.nv_scale is not None
        else float(init_nv_scale)
        if init_nv_scale is not None
        else float(max(CANONICAL_NV_TARGETS))
    )
    dataset_base, precomputed_stats = build_interface_flux_rollout_qpair_dataset(
        reference,
        max_projection_order=max_projection_order,
        Nv_targets=CANONICAL_NV_TARGETS,
        Nm=args.Nm,
        k_arr=k_arr,
        val_fraction=0.0,
        linear_history_stride=args.history_stride,
        nonlinear_history_stride=args.history_stride,
        rollout_horizon=args.rollout_horizon,
        n_low=args.n_low,
        context_mode="none",
        store_training_pairs=False,
        k_scale=k_scale,
        nv_scale=nv_scale,
    )
    if init_stats is not None:
        stats = init_stats
    else:
        if precomputed_stats is None:
            raise RuntimeError("interface-flux training statistics are unavailable")
        stats = phase_isotropic_complex_training_stats(
            precomputed_stats,
            Nm=args.Nm,
            context_mode="none",
        )
    prepared = prepare_validation_dataset_from_stats(
        dataset_base,
        Nm=args.Nm,
        k_scale=k_scale,
        nv_scale=nv_scale,
        context_mode="none",
        stats=stats,
    )
    for regime in CANONICAL_REGIMES:
        anchors = int(dataset_base[regime]["train_anchor_time_indices"].shape[0])
        print(f"[data] {regime}: {anchors} interface-flux rollout anchors")

    regime_loss_stds = interface_flux_rollout_regime_loss_stds(
        reference,
        dataset_base,
        max_projection_order=max_projection_order,
        target_nvs=CANONICAL_NV_TARGETS,
        k_arr=k_arr,
        rollout_horizon=args.rollout_horizon,
    )
    print(
        "[data] fixed regime-balanced interface-flux scales: "
        + " ".join(
            f"{regime}={regime_loss_stds[regime]:.6e}"
            for regime in CANONICAL_REGIMES
        )
    )
    print(
        "[data] canonical constraints: all_positive_k=1 "
        "equilibrium_centered=1 complex_normalization=phase_isotropic "
        "translation_augmentation=1 "
        f"cutoff_cycle={','.join(str(value) for value in CANONICAL_NV_TARGETS)}"
    )

    params = (
        init_params
        if init_params is not None
        else init_interface_closure_params(
            jax.random.PRNGKey(args.seed),
            input_dim=int(stats["input_mean"].shape[0]),
            hidden_width=int(args.hidden_width),
            res_blocks=int(args.res_blocks),
        )
    )
    loss_fn, active_regimes = make_interface_flux_rollout_batch_loss(
        regime_weights={regime: 1.0 for regime in CANONICAL_REGIMES},
        Nm=args.Nm,
        k_scale=k_scale,
        nv_scale=nv_scale,
        stats=stats,
        hidden_width=args.hidden_width,
        res_blocks=args.res_blocks,
        Nv_targets=CANONICAL_NV_TARGETS,
        train_regimes=CANONICAL_REGIMES,
        teacher_backend=GRID_CUBIC_SPLINE_TEACHER_BACKEND,
        teacher_Lx=args.teacher_L,
        teacher_Nx=args.teacher_Nx,
        teacher_Nv=args.teacher_Nv,
        teacher_vmin=args.teacher_vmin,
        teacher_vmax=args.teacher_vmax,
        teacher_dt=args.teacher_dt,
        teacher_proj_Nv=max_projection_order,
        projection_quadrature_Nv=projection_quadrature_Nv,
        n_low=args.n_low,
        context_mode="none",
        rollout_horizon=args.rollout_horizon,
        poisson_sign=args.teacher_poisson_sign,
        rollout_dealias_23=False,
        rollout_precision=args.precision,
        regime_q_loss_stds=regime_loss_stds,
        equilibrium_centered=True,
        complex_normalization_mode="phase_isotropic",
        translation_augmented=True,
    )
    params, component_history = train_with_interface_flux_rollout_minibatch_loss(
        params,
        reference,
        dataset_base,
        loss_fn,
        max_projection_order=max_projection_order,
        active_regimes=active_regimes,
        k_arr=k_arr,
        epochs=args.epochs,
        learning_rate=args.lr,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        rollout_horizon=args.rollout_horizon,
        seed=args.seed,
        log_components=("q",),
        profile_trace_dir=args.profile_trace_dir,
        profile_train_steps=args.profile_train_steps,
        profile_skip_steps=args.profile_skip_steps,
    )
    learned = build_learned_interface_closure(
        params=params,
        Nm=args.Nm,
        k_scale=k_scale,
        nv_scale=nv_scale,
        stats=stats,
        hidden_width=args.hidden_width,
        res_blocks=args.res_blocks,
        equilibrium_centered=True,
        complex_normalization_mode="phase_isotropic",
        translation_augmented=True,
        Nv_targets=CANONICAL_NV_TARGETS,
        train_regimes=CANONICAL_REGIMES,
        teacher_backend=GRID_CUBIC_SPLINE_TEACHER_BACKEND,
        teacher_Lx=args.teacher_L,
        teacher_Nx=args.teacher_Nx,
        teacher_Nv=args.teacher_Nv,
        teacher_vmin=args.teacher_vmin,
        teacher_vmax=args.teacher_vmax,
        teacher_dt=args.teacher_dt,
        teacher_proj_Nv=max_projection_order,
        projection_quadrature_Nv=projection_quadrature_Nv,
        n_low=args.n_low,
        rollout_horizon=args.rollout_horizon,
    )
    val_metrics = evaluate_regime_metrics(learned, prepared)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_learned_interface_closure_npz(args.checkpoint, learned)
    metrics_path = args.checkpoint.with_suffix(".metrics.npz")
    loss_history = np.asarray(component_history["total"], dtype=np.float64)
    metrics_payload = _canonical_metrics_payload(
        args=args,
        loss_history=loss_history,
        component_history=component_history,
        stats=stats,
        k_scale=k_scale,
        nv_scale=nv_scale,
        teacher_projection_order=max_projection_order,
        projection_quadrature_Nv=projection_quadrature_Nv,
        regime_loss_stds=regime_loss_stds,
        val_metrics=val_metrics,
    )
    np.savez(metrics_path, **metrics_payload)
    loss_plot_path = (
        args.loss_plot
        if args.loss_plot is not None
        else args.checkpoint.with_suffix(".loss.png")
    )
    save_training_loss_plot(
        loss_history,
        loss_plot_path,
        loss_metadata=metrics_payload,
        val_metrics=val_metrics,
    )
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved loss plot to {loss_plot_path}")
    for key in sorted(val_metrics):
        print(f"{key}: {float(np.asarray(val_metrics[key]).reshape(-1)[0]):.6e}")


if __name__ == "__main__":
    main()
