"""Held-out interface-flux metrics and canonical result payloads."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from vpml.datasets.interface_flux import (
    IC_SPLIT_HELDOUT,
    IC_SPLIT_TRAIN,
    grouped_history_gather,
)
from vpml.models.interface_closure import (
    GRID_CUBIC_SPLINE_TEACHER_BACKEND,
    LearnedInterfaceClosure,
)
from vpml.solvers.physical_grid import extract_interface_supervised_pairs_from_coeff_history
from vpml.training.interface_configuration import (
    CANONICAL_METADATA_SCHEMA_VERSION,
    CANONICAL_NV_TARGETS,
    CANONICAL_REGIMES,
    EXACT_TARGET_SAMPLING_CYCLE,
    INTERFACE_FLUX_PROJECTION_SCHEME,
    INTERFACE_FLUX_ROLLOUT_LOSS_BACKEND,
    INTERFACE_FLUX_ROLLOUT_OBJECTIVE,
    INTERFACE_FLUX_ROLLOUT_TRAINING_MODE,
    interface_flux_rollout_coeff_key,
)
from vpml.training.interface_data import build_model_inputs
from vpml.training.interface_reference import (
    _reference_case_histories,
    restrict_rfft_coefficients_to_grid,
)


def evaluate_heldout_interface_flux_metrics(
    learned: LearnedInterfaceClosure,
    reference_dataset: Dict[str, Dict[str, object]],
    interface_flux_dataset: Dict[str, Dict[str, np.ndarray]],
    *,
    max_projection_order: int,
    target_nvs: Sequence[int],
    Nm: int,
    k_arr: np.ndarray,
    k_scale: float,
    nv_scale: float,
    n_low: int,
    chunk_size: int = 64,
    source_Nx: Optional[int] = None,
    rollout_Nx: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Evaluate direct interface flux only on complete held-out trajectories."""
    coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
    retained_k_count = int(np.asarray(k_arr).shape[0])
    if retained_k_count <= 1:
        raise ValueError("Held-out interface-flux evaluation requires positive Fourier modes")
    if (source_Nx is None) != (rollout_Nx is None):
        raise ValueError("source_Nx and rollout_Nx must be provided together")
    metrics: Dict[str, np.ndarray] = {}
    chunk_n = max(int(chunk_size), 1)
    for regime, arrays in interface_flux_dataset.items():
        histories = _reference_case_histories(reference_dataset[regime], coeff_key)
        anchor_cases = np.asarray(arrays["val_anchor_case_indices"], dtype=np.int32)
        anchor_times = np.asarray(arrays["val_anchor_time_indices"], dtype=np.int32)
        anchor_targets = np.asarray(arrays["val_anchor_target_nvs"], dtype=np.int32)
        squared_error = 0.0
        target_squared = 0.0
        scalar_count = 0
        sample_count = 0
        for target_nv in target_nvs:
            selected = np.flatnonzero(anchor_targets == int(target_nv)).astype(np.int32)
            for start in range(0, int(selected.shape[0]), chunk_n):
                chunk = selected[start : start + chunk_n]
                if int(chunk.shape[0]) == 0:
                    continue
                states = grouped_history_gather(
                    histories,
                    anchor_cases[chunk],
                    anchor_times[chunk],
                    hermite_slice=slice(None),
                    fourier_slice=slice(0, retained_k_count),
                )
                if source_Nx is not None and rollout_Nx is not None:
                    states = restrict_rfft_coefficients_to_grid(
                        states,
                        source_Nx=int(source_Nx),
                        target_Nx=int(rollout_Nx),
                    )
                pairs = extract_interface_supervised_pairs_from_coeff_history(
                    np.asarray(states, dtype=np.complex128),
                    Nv_targets=(int(target_nv),),
                    Nm=int(Nm),
                    k_arr=np.asarray(k_arr, dtype=np.float64),
                    vth=1.0,
                    include_global_indicators=True,
                    n_low=int(n_low),
                    context_mode="none",
                )[int(target_nv)]
                model_inputs = build_model_inputs(
                    pairs["inputs_base"],
                    Nm=int(Nm),
                    k_scale=float(k_scale),
                    nv_scale=float(nv_scale),
                    context_mode="none",
                )
                target = np.asarray(pairs["targets"], dtype=np.float64)
                prediction = np.asarray(
                    learned.predict_q_components(jnp.asarray(model_inputs)),
                    dtype=np.float64,
                )
                residual = prediction - target
                squared_error += float(np.sum(residual * residual, dtype=np.float64))
                target_squared += float(np.sum(target * target, dtype=np.float64))
                scalar_count += int(target.size)
                sample_count += int(target.shape[0])
        mse = (
            float(squared_error / float(scalar_count))
            if scalar_count > 0
            else float("nan")
        )
        rel_l2 = (
            float(math.sqrt(squared_error / max(target_squared, 1e-300)))
            if scalar_count > 0
            else float("nan")
        )
        metrics[f"val_q_mse_{regime}"] = np.array([mse], dtype=np.float64)
        metrics[f"val_q_rel_l2_{regime}"] = np.array([rel_l2], dtype=np.float64)
        metrics[f"val_num_samples_{regime}"] = np.array(
            [sample_count], dtype=np.int64
        )
    return metrics


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
    ic_manifest: Dict[str, object],
    reference_cache_dir: Path,
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
        "rollout_Nx": np.array([args.rollout_Nx], dtype=np.int32),
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
        "ic_manifest_sha256": np.array(
            [str(ic_manifest["sha256"])], dtype=np.str_
        ),
        "ic_manifest_path": np.array([str(args.ic_manifest)], dtype=np.str_),
        "reference_cache_dir": np.array(
            [str(reference_cache_dir)], dtype=np.str_
        ),
        "training_ic_count": np.array(
            [
                sum(
                    str(case["split"]) == IC_SPLIT_TRAIN
                    for case in ic_manifest["cases"]
                )
            ],
            dtype=np.int32,
        ),
        "heldout_ic_count": np.array(
            [
                sum(
                    str(case["split"]) == IC_SPLIT_HELDOUT
                    for case in ic_manifest["cases"]
                )
            ],
            dtype=np.int32,
        ),
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
