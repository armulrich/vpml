"""Train a causal spectral-memory closure for a three-moment fluid solver."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple

from vpml.jax_runtime import bootstrap_jax_runtime, print_jax_runtime_summary

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model.train.interface_flux_data import (
    IC_SPLIT_HELDOUT,
    IC_SPLIT_TRAIN,
    case_shard_paths,
    load_ic_manifest,
    load_sharded_reference,
    sha256_json,
)
from vpml.low_moment import (
    DEFAULT_DENSITY_FLOOR,
    DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    DEFAULT_PRESSURE_FLOOR,
    electric_field_from_density,
    init_spectral_memory_params,
    primitive_fields,
    rollout_low_moment_closure,
    spectral_memory_closure_step,
    warm_spectral_memory,
)
from vpml.metrics import (
    EarlyElectricFieldGrowthMetric,
    EarlyGrowthConfig,
    FieldErrorConfig,
    SelfGeneratedFieldErrorMetric,
)


REGIMES = (
    "linear_landau",
    "nonlinear_landau_weak",
    "nonlinear_landau_strong",
)
TRAINING_MODE = "solver_embedded_low_moment_spectral_memory"
OBJECTIVE = "low_moment_trajectory"
MODEL_BACKEND = "causal_translation_equivariant_spectral_memory"
CHECKPOINT_SCHEMA = 2


def _restrict_rfft(values: np.ndarray, source_nx: int, target_nx: int) -> np.ndarray:
    values = np.asarray(values)
    source_nx = int(source_nx)
    target_nx = int(target_nx)
    if not 1 < target_nx <= source_nx:
        raise ValueError("Require 1 < target_nx <= source_nx")
    target_nk = target_nx // 2 + 1
    restricted = np.asarray(values[..., :target_nk]).copy()
    if source_nx != target_nx:
        restricted *= float(target_nx) / float(source_nx)
        if target_nx % 2 == 0:
            restricted[..., -1] = 2.0 * np.real(restricted[..., -1])
    return restricted


def low_hermite_coefficients_to_conservative(
    coefficients: np.ndarray,
    *,
    source_nx: int,
    target_nx: int,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Convert C0:C2 into centered density, momentum, and second moment."""
    requested_dtype = np.dtype(dtype)
    complex_dtype = np.complex128 if requested_dtype == np.float64 else np.complex64
    coeff = np.asarray(coefficients, dtype=complex_dtype)
    if coeff.shape[-2] < 3:
        raise ValueError("At least C0, C1, and C2 are required")
    coeff = _restrict_rfft(coeff[..., :3, :], source_nx, target_nx)
    physical = np.fft.irfft(coeff, n=int(target_nx), axis=-1)
    density_perturbation = physical[..., 0, :]
    momentum = physical[..., 1, :]
    second_perturbation = density_perturbation + math.sqrt(2.0) * physical[..., 2, :]
    return np.stack(
        (density_perturbation, momentum, second_perturbation), axis=-2
    ).astype(requested_dtype)


def _primitive_numpy(state: np.ndarray, domain_length: float) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    density_perturbation, momentum, second_perturbation = (
        state[..., 0, :],
        state[..., 1, :],
        state[..., 2, :],
    )
    rho = 1.0 + density_perturbation
    safe_rho = np.where(np.abs(rho) > 1e-8, rho, 1e-8)
    velocity = momentum / safe_rho
    pressure_perturbation = second_perturbation - momentum * velocity
    nx = int(state.shape[-1])
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(nx, d=float(domain_length) / nx)
    rho_hat = np.fft.rfft(
        density_perturbation
        - np.mean(density_perturbation, axis=-1, keepdims=True),
        axis=-1,
    )
    field_hat = np.zeros_like(rho_hat)
    field_hat[..., 1:] = 1j * rho_hat[..., 1:] / k_arr[1:]
    field = np.fft.irfft(field_hat, n=nx, axis=-1)
    return np.stack(
        (density_perturbation, velocity, pressure_perturbation, field), axis=-2
    )


def _central_heat_flux_numpy(
    coefficients: np.ndarray,
    state: np.ndarray,
    *,
    source_nx: int,
    target_nx: int,
) -> np.ndarray:
    coeff = _restrict_rfft(np.asarray(coefficients)[..., :4, :], source_nx, target_nx)
    physical = np.fft.irfft(coeff, n=int(target_nx), axis=-1)
    rho = 1.0 + state[..., 0, :]
    momentum = state[..., 1, :]
    safe_rho = np.where(np.abs(rho) > 1e-8, rho, 1e-8)
    velocity = momentum / safe_rho
    pressure = 1.0 + state[..., 2, :] - momentum * velocity
    raw_third = math.sqrt(6.0) * physical[..., 3, :] + 3.0 * momentum
    return raw_third - 3.0 * velocity * pressure - rho * velocity**3


def _load_reference_cache(cache_dir: Path):
    cache_dir = Path(cache_dir).resolve()
    with (cache_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    manifest = load_ic_manifest(cache_dir / "ic_manifest.json")
    configuration = dict(metadata["configuration"])
    projection_order = int(configuration["max_projection_order"])
    if projection_order < 4:
        raise ValueError("The reference cache must contain at least C0:C3")
    coefficient_key = f"a_hat_ref_order{projection_order}"
    grouped = load_sharded_reference(cache_dir, manifest, coeff_key=coefficient_key)
    return grouped, manifest, metadata, coefficient_key


def _case_amplitudes(manifest: Mapping[str, object]) -> Dict[str, float]:
    return {str(case["case_id"]): float(case["epsilon"]) for case in manifest["cases"]}


def _build_anchor_index(
    grouped: Mapping[str, Mapping[str, object]],
    coefficient_key: str,
    *,
    horizon: int,
    history_stride: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for regime in REGIMES:
        group = grouped[regime]
        histories = tuple(group[coefficient_key])
        splits = np.asarray(group["case_splits"], dtype=np.str_)
        payload: Dict[str, np.ndarray] = {}
        for split_name, split_value in (("train", IC_SPLIT_TRAIN), ("val", IC_SPLIT_HELDOUT)):
            case_rows = []
            time_rows = []
            for case_index, (history, split) in enumerate(zip(histories, splits)):
                if str(split) != split_value:
                    continue
                times = np.arange(
                    0,
                    int(history.shape[0]) - int(horizon),
                    max(int(history_stride), 1),
                    dtype=np.int32,
                )
                case_rows.append(np.full(times.shape, case_index, dtype=np.int32))
                time_rows.append(times)
            payload[f"{split_name}_cases"] = (
                np.concatenate(case_rows) if case_rows else np.zeros((0,), dtype=np.int32)
            )
            payload[f"{split_name}_times"] = (
                np.concatenate(time_rows) if time_rows else np.zeros((0,), dtype=np.int32)
            )
        result[regime] = payload
    return result


def _gather_coefficients(
    histories: Sequence[np.ndarray],
    case_indices: np.ndarray,
    time_indices: np.ndarray,
    *,
    coefficient_count: int,
) -> np.ndarray:
    cases, times = np.broadcast_arrays(
        np.asarray(case_indices, dtype=np.int32), np.asarray(time_indices, dtype=np.int32)
    )
    sample = np.asarray(histories[0][0, :coefficient_count])
    output = np.empty(cases.shape + sample.shape, dtype=sample.dtype)
    flat_cases = cases.reshape(-1)
    flat_times = times.reshape(-1)
    flat_output = output.reshape((flat_cases.size,) + sample.shape)
    for case_index in np.unique(flat_cases):
        positions = np.flatnonzero(flat_cases == int(case_index))
        ordered = positions[np.argsort(flat_times[positions], kind="stable")]
        flat_output[ordered] = histories[int(case_index)][
            flat_times[ordered], :coefficient_count, :
        ]
    return output


def _statistics_path(cache_dir: Path, rollout_nx: int, stats_stride: int) -> Path:
    key = sha256_json(
        {
            "kind": "low_moment_spectral_memory_stats_v3_centered_gradient",
            "rollout_nx": int(rollout_nx),
            "stats_stride": int(stats_stride),
        }
    )[:20]
    return Path(cache_dir) / "derived" / f"{key}.npz"


def _compute_training_statistics(
    grouped: Mapping[str, Mapping[str, object]],
    manifest: Mapping[str, object],
    coefficient_key: str,
    *,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    stats_stride: int,
) -> Dict[str, np.ndarray]:
    amplitudes = _case_amplitudes(manifest)
    global_sum = np.zeros((4,), dtype=np.float64)
    global_count = 0
    regime_sum = {regime: np.zeros((4,), dtype=np.float64) for regime in REGIMES}
    regime_count = {regime: 0 for regime in REGIMES}
    heat_flux_gradient_sum_global = 0.0
    heat_flux_gradient_count_global = 0
    heat_flux_gradient_max_abs = 0.0
    heat_flux_gradient_sum = {regime: 0.0 for regime in REGIMES}
    heat_flux_gradient_count = {regime: 0 for regime in REGIMES}
    train_log_amplitudes = []
    for regime in REGIMES:
        group = grouped[regime]
        histories = tuple(group[coefficient_key])
        case_ids = np.asarray(group["case_ids"], dtype=np.str_)
        splits = np.asarray(group["case_splits"], dtype=np.str_)
        for history, case_id, split in zip(histories, case_ids, splits):
            if str(split) != IC_SPLIT_TRAIN:
                continue
            train_log_amplitudes.append(math.log(amplitudes[str(case_id)]))
            indices = np.arange(0, int(history.shape[0]), max(int(stats_stride), 1))
            for start in range(0, int(indices.size), 64):
                rows = indices[start : start + 64]
                coeff = np.asarray(history[rows, :4, :])
                state = low_hermite_coefficients_to_conservative(
                    coeff, source_nx=source_nx, target_nx=rollout_nx, dtype=np.float64
                )
                fields = _primitive_numpy(state, domain_length)
                squares = np.sum(fields * fields, axis=(0, 2), dtype=np.float64)
                count = int(fields.shape[0] * fields.shape[-1])
                global_sum += squares
                global_count += count
                regime_sum[regime] += squares
                regime_count[regime] += count
                heat_flux = _central_heat_flux_numpy(
                    coeff,
                    state,
                    source_nx=source_nx,
                    target_nx=rollout_nx,
                )
                k_arr = 2.0 * math.pi * np.fft.rfftfreq(
                    int(rollout_nx), d=float(domain_length) / float(rollout_nx)
                )
                heat_flux_gradient = np.fft.irfft(
                    1j * k_arr * np.fft.rfft(heat_flux, axis=-1),
                    n=int(rollout_nx),
                    axis=-1,
                )
                heat_flux_gradient_sum[regime] += float(
                    np.sum(heat_flux_gradient * heat_flux_gradient, dtype=np.float64)
                )
                heat_flux_gradient_count[regime] += int(heat_flux_gradient.size)
                heat_flux_gradient_sum_global += float(
                    np.sum(heat_flux_gradient * heat_flux_gradient, dtype=np.float64)
                )
                heat_flux_gradient_count_global += int(heat_flux_gradient.size)
                heat_flux_gradient_max_abs = max(
                    heat_flux_gradient_max_abs,
                    float(np.max(np.abs(heat_flux_gradient))),
                )
    input_scale = np.sqrt(global_sum / max(global_count, 1))
    regime_scales = np.stack(
        [np.sqrt(regime_sum[regime] / max(regime_count[regime], 1)) for regime in REGIMES]
    )
    input_scale = np.maximum(input_scale, np.array([1e-5, 1e-5, 1e-5, 1e-5]))
    regime_scales = np.maximum(regime_scales, 1e-6)
    log_amplitudes = np.asarray(train_log_amplitudes, dtype=np.float64)
    return {
        "input_scale": input_scale,
        "regime_scales": regime_scales,
        "heat_flux_gradient_scale": np.array(
            [
                math.sqrt(
                    heat_flux_gradient_sum_global
                    / max(heat_flux_gradient_count_global, 1)
                )
            ],
            dtype=np.float64,
        ),
        "heat_flux_gradient_regime_scales": np.asarray(
            [
                max(
                    math.sqrt(
                        heat_flux_gradient_sum[regime]
                        / max(heat_flux_gradient_count[regime], 1)
                    ),
                    1e-6,
                )
                for regime in REGIMES
            ],
            dtype=np.float64,
        ),
        "heat_flux_gradient_max_abs": np.array(
            [heat_flux_gradient_max_abs], dtype=np.float64
        ),
        "amplitude_center": np.array([np.mean(log_amplitudes)], dtype=np.float64),
        "amplitude_scale": np.array(
            [max(float(np.std(log_amplitudes)), 1e-8)], dtype=np.float64
        ),
    }


def load_or_compute_training_statistics(
    cache_dir: Path,
    grouped,
    manifest,
    coefficient_key: str,
    *,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    stats_stride: int,
) -> Dict[str, np.ndarray]:
    path = _statistics_path(cache_dir, rollout_nx, stats_stride)
    if path.exists():
        print(f"[data] reusing low-moment statistics from {path}")
        with np.load(path) as payload:
            return {key: np.asarray(payload[key]) for key in payload.files}
    stats = _compute_training_statistics(
        grouped,
        manifest,
        coefficient_key,
        source_nx=source_nx,
        rollout_nx=rollout_nx,
        domain_length=domain_length,
        stats_stride=stats_stride,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)
    print(f"[data] saved low-moment statistics to {path}")
    return stats


def sample_batch(
    rng: np.random.Generator,
    grouped,
    anchors,
    manifest,
    coefficient_key: str,
    *,
    split: str,
    batch_size_per_regime: int,
    horizon: int,
    memory_steps: int,
    memory_stride: int,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    translation_augmentation: bool,
    explicit_selection: Optional[
        Mapping[str, Tuple[np.ndarray, np.ndarray]]
    ] = None,
    time_blocks: int = 6,
) -> Dict[str, np.ndarray]:
    amplitudes_by_id = _case_amplitudes(manifest)
    memories = []
    initials = []
    targets = []
    amplitudes = []
    regime_indices = []
    heat_flux_gradient_targets = []
    for regime_index, regime in enumerate(REGIMES):
        group = grouped[regime]
        histories = tuple(group[coefficient_key])
        case_ids = np.asarray(group["case_ids"], dtype=np.str_)
        case_pool = anchors[regime][f"{split}_cases"]
        time_pool = anchors[regime][f"{split}_times"]
        if int(case_pool.size) == 0:
            raise ValueError(f"No {split} anchors for {regime}")
        if explicit_selection is not None:
            cases, times = explicit_selection[regime]
            cases = np.asarray(cases, dtype=np.int32)
            times = np.asarray(times, dtype=np.int32)
        else:
            unique_cases = np.unique(case_pool)
            cases = rng.choice(
                unique_cases,
                size=int(batch_size_per_regime),
                replace=int(batch_size_per_regime) > int(unique_cases.size),
            ).astype(np.int32)
            block_count = max(int(time_blocks), 1)
            block_offset = int(rng.integers(0, block_count))
            block_ids = (block_offset + np.arange(cases.size)) % block_count
            selected_times = []
            for case_index, block_index in zip(cases, block_ids):
                available = time_pool[case_pool == int(case_index)]
                edges = np.linspace(0, available.size, block_count + 1, dtype=np.int32)
                left, right = int(edges[block_index]), int(edges[block_index + 1])
                if right <= left:
                    left, right = 0, int(available.size)
                selected_times.append(available[int(rng.integers(left, right))])
            times = np.asarray(selected_times, dtype=np.int32)
        memory_offsets = int(memory_stride) * np.arange(
            int(memory_steps), 0, -1, dtype=np.int32
        )
        memory_times = np.maximum(times[:, None] - memory_offsets[None, :], 0)
        target_times = times[:, None] + np.arange(1, int(horizon) + 1, dtype=np.int32)[None]
        memory_coeff = _gather_coefficients(
            histories, cases[:, None], memory_times, coefficient_count=3
        )
        initial_coeff = _gather_coefficients(
            histories, cases, times, coefficient_count=4
        )
        target_coeff = _gather_coefficients(
            histories, cases[:, None], target_times, coefficient_count=3
        )
        memories.append(
            low_hermite_coefficients_to_conservative(
                memory_coeff, source_nx=source_nx, target_nx=rollout_nx
            )
        )
        initials.append(
            low_hermite_coefficients_to_conservative(
                initial_coeff, source_nx=source_nx, target_nx=rollout_nx
            )
        )
        targets.append(
            low_hermite_coefficients_to_conservative(
                target_coeff, source_nx=source_nx, target_nx=rollout_nx
            )
        )
        current_coeff = initial_coeff
        current_state = low_hermite_coefficients_to_conservative(
            current_coeff,
            source_nx=source_nx,
            target_nx=rollout_nx,
            dtype=np.float64,
        )
        central_heat_flux = _central_heat_flux_numpy(
            current_coeff,
            current_state,
            source_nx=source_nx,
            target_nx=rollout_nx,
        )
        k_arr = 2.0 * math.pi * np.fft.rfftfreq(
            int(rollout_nx), d=float(domain_length) / float(rollout_nx)
        )
        heat_flux_gradient_targets.append(
            np.fft.irfft(
                1j * k_arr * np.fft.rfft(central_heat_flux, axis=-1),
                n=int(rollout_nx),
                axis=-1,
            )
        )
        amplitudes.append(
            np.asarray([amplitudes_by_id[str(case_ids[index])] for index in cases])
        )
        regime_indices.append(np.full((len(cases),), regime_index, dtype=np.int32))
    batch = {
        "memory": np.concatenate(memories, axis=0).astype(np.float32),
        "initial": np.concatenate(initials, axis=0).astype(np.float32),
        "targets": np.concatenate(targets, axis=0).astype(np.float32),
        "amplitude": np.concatenate(amplitudes, axis=0).astype(np.float32),
        "regime_index": np.concatenate(regime_indices, axis=0),
        "heat_flux_gradient_target": np.concatenate(
            heat_flux_gradient_targets, axis=0
        ).astype(np.float32),
    }
    if translation_augmentation:
        shifts = rng.integers(0, int(rollout_nx), size=batch["initial"].shape[0])
        for row, shift in enumerate(shifts):
            batch["memory"][row] = np.roll(batch["memory"][row], int(shift), axis=-1)
            batch["initial"][row] = np.roll(batch["initial"][row], int(shift), axis=-1)
            batch["targets"][row] = np.roll(batch["targets"][row], int(shift), axis=-1)
            batch["heat_flux_gradient_target"][row] = np.roll(
                batch["heat_flux_gradient_target"][row], int(shift), axis=-1
            )
    return batch


def build_complete_trajectory_case_batches(
    rng: np.random.Generator,
    grouped,
    *,
    split: str,
    batch_size_per_regime: int,
    shuffle: bool,
) -> Sequence[Dict[str, np.ndarray]]:
    """Return balanced case batches that cover every IC in a split once."""
    split_value = IC_SPLIT_TRAIN if split == "train" else IC_SPLIT_HELDOUT
    rows_by_regime = {}
    for regime in REGIMES:
        rows = np.flatnonzero(
            np.asarray(grouped[regime]["case_splits"], dtype=np.str_) == split_value
        ).astype(np.int32)
        if rows.size == 0:
            raise ValueError(f"No {split} trajectories for {regime}")
        rows_by_regime[regime] = rng.permutation(rows) if shuffle else rows
    counts = {int(rows.size) for rows in rows_by_regime.values()}
    if len(counts) != 1:
        raise ValueError(
            f"Complete-trajectory training requires equal per-regime IC counts, got {counts}"
        )
    batch_size = max(int(batch_size_per_regime), 1)
    case_count = counts.pop()
    return tuple(
        {
            regime: rows_by_regime[regime][start : start + batch_size]
            for regime in REGIMES
        }
        for start in range(0, case_count, batch_size)
    )


def sample_complete_trajectory_batch(
    rng: np.random.Generator,
    grouped,
    manifest,
    coefficient_key: str,
    case_rows: Mapping[str, np.ndarray],
    *,
    memory_steps: int,
    memory_stride: int,
    source_nx: int,
    rollout_nx: int,
    translation_augmentation: bool,
) -> Dict[str, np.ndarray]:
    """Load complete, balanced trajectories initialized only at ``t=0``."""
    del memory_stride  # Every pre-initial memory sample is clamped to t=0.
    amplitudes_by_id = _case_amplitudes(manifest)
    memories = []
    initials = []
    targets = []
    amplitudes = []
    regime_indices = []
    trajectory_steps = None
    for regime_index, regime in enumerate(REGIMES):
        group = grouped[regime]
        histories = tuple(group[coefficient_key])
        case_ids = np.asarray(group["case_ids"], dtype=np.str_)
        cases = np.asarray(case_rows[regime], dtype=np.int32)
        available_steps = min(int(histories[index].shape[0]) - 1 for index in cases)
        trajectory_steps = (
            available_steps
            if trajectory_steps is None
            else min(int(trajectory_steps), available_steps)
        )
        memory_times = np.zeros((cases.size, int(memory_steps)), dtype=np.int32)
        initial_times = np.zeros((cases.size,), dtype=np.int32)
        target_times = np.broadcast_to(
            np.arange(1, available_steps + 1, dtype=np.int32)[None, :],
            (cases.size, available_steps),
        )
        memory_coeff = _gather_coefficients(
            histories, cases[:, None], memory_times, coefficient_count=3
        )
        initial_coeff = _gather_coefficients(
            histories, cases, initial_times, coefficient_count=4
        )
        target_coeff = _gather_coefficients(
            histories, cases[:, None], target_times, coefficient_count=3
        )
        memories.append(
            low_hermite_coefficients_to_conservative(
                memory_coeff, source_nx=source_nx, target_nx=rollout_nx
            )
        )
        initials.append(
            low_hermite_coefficients_to_conservative(
                initial_coeff, source_nx=source_nx, target_nx=rollout_nx
            )
        )
        targets.append(
            low_hermite_coefficients_to_conservative(
                target_coeff, source_nx=source_nx, target_nx=rollout_nx
            )
        )
        amplitudes.append(
            np.asarray([amplitudes_by_id[str(case_ids[index])] for index in cases])
        )
        regime_indices.append(np.full((cases.size,), regime_index, dtype=np.int32))
    if trajectory_steps is None:
        raise ValueError("No complete trajectories selected")
    batch = {
        "memory": np.concatenate(memories, axis=0).astype(np.float32),
        "initial": np.concatenate(initials, axis=0).astype(np.float32),
        "targets": np.concatenate(
            [value[:, :trajectory_steps] for value in targets], axis=0
        ).astype(np.float32),
        "amplitude": np.concatenate(amplitudes, axis=0).astype(np.float32),
        "regime_index": np.concatenate(regime_indices, axis=0),
    }
    if translation_augmentation:
        shifts = rng.integers(0, int(rollout_nx), size=batch["initial"].shape[0])
        for row, shift in enumerate(shifts):
            batch["memory"][row] = np.roll(
                batch["memory"][row], int(shift), axis=-1
            )
            batch["initial"][row] = np.roll(
                batch["initial"][row], int(shift), axis=-1
            )
            batch["targets"][row] = np.roll(
                batch["targets"][row], int(shift), axis=-1
            )
    return batch


def build_diagnostic_panel(
    rng: np.random.Generator,
    grouped,
    anchors,
    manifest,
    coefficient_key: str,
    *,
    split: str,
    start_times: Sequence[float],
    cases_per_regime: Optional[int],
    horizon: int,
    dt: float,
    memory_steps: int,
    memory_stride: int,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
) -> Sequence[Dict[str, np.ndarray]]:
    amplitudes = _case_amplitudes(manifest)
    selections = {}
    for regime in REGIMES:
        group = grouped[regime]
        split_value = IC_SPLIT_TRAIN if split == "train" else IC_SPLIT_HELDOUT
        rows = np.flatnonzero(np.asarray(group["case_splits"]) == split_value)
        rows = np.asarray(
            sorted(rows, key=lambda row: amplitudes[str(group["case_ids"][row])]),
            dtype=np.int32,
        )
        if cases_per_regime is not None and int(cases_per_regime) < int(rows.size):
            positions = np.rint(
                np.linspace(0, rows.size - 1, int(cases_per_regime))
            ).astype(np.int32)
            rows = rows[positions]
        selections[regime] = rows
    panel = []
    for start_time in start_times:
        time_index = int(round(float(start_time) / float(dt)))
        explicit = {}
        for regime in REGIMES:
            rows = selections[regime]
            histories = tuple(grouped[regime][coefficient_key])
            if any(time_index + int(horizon) >= int(histories[row].shape[0]) for row in rows):
                raise ValueError(
                    f"Diagnostic start t={start_time:g} exceeds the valid H={horizon} window"
                )
            explicit[regime] = (
                rows,
                np.full(rows.shape, time_index, dtype=np.int32),
            )
        panel.append(
            sample_batch(
                rng,
                grouped,
                anchors,
                manifest,
                coefficient_key,
                split=split,
                batch_size_per_regime=len(selections[REGIMES[0]]),
                horizon=horizon,
                memory_steps=memory_steps,
                memory_stride=memory_stride,
                source_nx=source_nx,
                rollout_nx=rollout_nx,
                domain_length=domain_length,
                translation_augmentation=False,
                explicit_selection=explicit,
            )
        )
    return panel


def _adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": jnp.array(0, dtype=jnp.int32), "m": zeros, "v": zeros}


def _adam_step(params, grads, state, learning_rate: float, grad_clip: float):
    norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(grads))
        + 1e-30
    )
    scale = jnp.minimum(1.0, jnp.asarray(grad_clip, norm.dtype) / norm)
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
    params = jax.tree_util.tree_map(
        lambda value, m_value, v_value: value
        - learning_rate * (m_value / bias_m) / (jnp.sqrt(v_value / bias_v) + 1e-8),
        params,
        m,
        v,
    )
    return params, {"step": step, "m": m, "v": v}, norm


def make_loss_function(
    *,
    k_arr: np.ndarray,
    width: int,
    horizon: int,
    dt: float,
    input_scale: np.ndarray,
    regime_scales: np.ndarray,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    poisson_sign: float,
    normalized_heat_flux_bound: float,
    density_floor: float,
    pressure_floor: float,
    relative_trajectory_loss: bool = False,
    closure_history_input: bool = False,
):
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    input_scale_jax = jnp.asarray(input_scale, dtype=jnp.float32)
    regime_scales_jax = jnp.asarray(regime_scales, dtype=jnp.float32)

    def loss(params, batch):
        warm_result = warm_spectral_memory(
            params,
            batch["memory"],
            batch["amplitude"],
            k_jax,
            width=width,
            input_scale=input_scale_jax,
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            closure_history_input=closure_history_input,
            return_closure_history=closure_history_input,
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
        )
        if closure_history_input:
            hidden, previous_gradient = warm_result
        else:
            hidden = warm_result
            previous_gradient = None
        rollout_result = rollout_low_moment_closure(
            params,
            batch["initial"],
            hidden,
            batch["amplitude"],
            k_jax,
            horizon=horizon,
            dt=dt,
            input_scale=input_scale_jax,
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            previous_heat_flux_gradient=previous_gradient,
            closure_history_input=closure_history_input,
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        )
        predicted = rollout_result[0]
        batch_count, time_count = predicted.shape[:2]
        predicted_fields = primitive_fields(
            predicted.reshape(batch_count * time_count, 3, predicted.shape[-1]),
            k_jax,
            poisson_sign=poisson_sign,
        ).reshape(batch_count, time_count, 4, predicted.shape[-1])
        target_fields = primitive_fields(
            batch["targets"].reshape(
                batch_count * time_count, 3, batch["targets"].shape[-1]
            ),
            k_jax,
            poisson_sign=poisson_sign,
        ).reshape(batch_count, time_count, 4, batch["targets"].shape[-1])
        if relative_trajectory_loss:
            numerator = jnp.sum(
                jnp.square(predicted_fields - target_fields), axis=(1, 3)
            )
            denominator = jnp.sum(jnp.square(target_fields), axis=(1, 3))
            sample_loss = jnp.where(
                jnp.all(denominator > 0.0, axis=1),
                jnp.mean(numerator / denominator, axis=1),
                jnp.nan,
            )
        else:
            scales = regime_scales_jax[batch["regime_index"]][:, None, :, None]
            normalized_error = (predicted_fields - target_fields) / scales
            sample_loss = jnp.mean(jnp.square(normalized_error), axis=(1, 2, 3))
        regime_loss = jnp.stack(
            [
                jnp.sum(
                    jnp.where(batch["regime_index"] == index, sample_loss, 0.0)
                )
                / jnp.maximum(jnp.sum(batch["regime_index"] == index), 1)
                for index in range(len(REGIMES))
            ]
        )
        return jnp.mean(sample_loss), regime_loss

    return loss


def make_continuous_chunk_loss_function(
    *,
    k_arr: np.ndarray,
    width: int,
    horizon: int,
    warm_memory: bool,
    dt: float,
    input_scale: np.ndarray,
    regime_scales: np.ndarray,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    poisson_sign: float,
    normalized_heat_flux_bound: float,
    density_floor: float,
    pressure_floor: float,
    relative_trajectory_loss: bool = False,
    closure_history_input: bool = False,
):
    """Return one truncated-gradient chunk of a continuous autonomous rollout."""
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    input_scale_jax = jnp.asarray(input_scale, dtype=jnp.float32)
    regime_scales_jax = jnp.asarray(regime_scales, dtype=jnp.float32)

    def loss(
        params,
        initial_state,
        memory_or_hidden,
        targets,
        amplitude,
        regime_index,
        trajectory_target_norm=None,
    ):
        if warm_memory:
            warmed = warm_spectral_memory(
                params,
                memory_or_hidden,
                amplitude,
                k_jax,
                width=width,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                closure_history_input=closure_history_input,
                return_closure_history=closure_history_input,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
            )
            if closure_history_input:
                hidden, previous_gradient = warmed
            else:
                hidden = warmed
                previous_gradient = None
        elif closure_history_input:
            hidden, previous_gradient = memory_or_hidden
        else:
            hidden = memory_or_hidden
            previous_gradient = None
        rollout_result = rollout_low_moment_closure(
            params,
            initial_state,
            hidden,
            amplitude,
            k_jax,
            horizon=horizon,
            dt=dt,
            input_scale=input_scale_jax,
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            previous_heat_flux_gradient=previous_gradient,
            closure_history_input=closure_history_input,
            return_closure_history=closure_history_input,
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        )
        if closure_history_input:
            predicted, final_hidden, final_gradient = rollout_result
        else:
            predicted, final_hidden = rollout_result
            final_gradient = None
        batch_count, time_count = predicted.shape[:2]
        predicted_fields = primitive_fields(
            predicted.reshape(batch_count * time_count, 3, predicted.shape[-1]),
            k_jax,
            poisson_sign=poisson_sign,
        ).reshape(batch_count, time_count, 4, predicted.shape[-1])
        target_fields = primitive_fields(
            targets.reshape(batch_count * time_count, 3, targets.shape[-1]),
            k_jax,
            poisson_sign=poisson_sign,
        ).reshape(batch_count, time_count, 4, targets.shape[-1])
        if relative_trajectory_loss:
            numerator = jnp.sum(
                jnp.square(predicted_fields - target_fields), axis=(1, 3)
            )
            if trajectory_target_norm is None:
                trajectory_target_norm = jnp.sum(
                    jnp.square(target_fields), axis=(1, 3)
                )
            sample_loss = jnp.where(
                jnp.all(trajectory_target_norm > 0.0, axis=1),
                jnp.mean(numerator / trajectory_target_norm, axis=1),
                jnp.nan,
            )
        else:
            scales = regime_scales_jax[regime_index][:, None, :, None]
            sample_loss = jnp.mean(
                jnp.square((predicted_fields - target_fields) / scales), axis=(1, 2, 3)
            )
        regime_loss = jnp.stack(
            [
                jnp.sum(jnp.where(regime_index == index, sample_loss, 0.0))
                / jnp.maximum(jnp.sum(regime_index == index), 1)
                for index in range(len(REGIMES))
            ]
        )
        final_memory = (
            (final_hidden, final_gradient) if closure_history_input else final_hidden
        )
        return jnp.mean(sample_loss), (
            regime_loss,
            predicted[:, -1],
            final_memory,
        )

    return loss


def make_supervised_heat_flux_loss(
    *,
    k_arr: np.ndarray,
    width: int,
    input_scale: np.ndarray,
    heat_flux_gradient_scale: float,
    heat_flux_gradient_regime_scales: np.ndarray,
    amplitude_center: float,
    amplitude_scale: float,
    poisson_sign: float,
    normalized_heat_flux_bound: float,
):
    """Warm up the causal closure on the physical low-moment heat flux."""
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    input_scale_jax = jnp.asarray(input_scale, dtype=jnp.float32)
    gradient_scales = jnp.asarray(
        heat_flux_gradient_regime_scales, dtype=jnp.float32
    )

    def loss(params, batch):
        hidden = warm_spectral_memory(
            params,
            batch["memory"],
            batch["amplitude"],
            k_jax,
            width=width,
            input_scale=input_scale_jax,
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
        )
        _, predicted_gradient = spectral_memory_closure_step(
            params,
            batch["initial"],
            hidden,
            batch["amplitude"],
            k_jax,
            input_scale=input_scale_jax,
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
        )
        scale = (
            batch["amplitude"]
            * gradient_scales[batch["regime_index"]]
        )[:, None]
        return jnp.mean(
            jnp.square(
                (predicted_gradient - batch["heat_flux_gradient_target"])
                / jnp.maximum(scale, 1e-8)
            )
        )

    return loss


def _save_checkpoint(path: Path, params, metadata: Mapping[str, object], stats):
    payload = {f"param_{key}": np.asarray(value) for key, value in params.items()}
    payload.update({f"stat_{key}": np.asarray(value) for key, value in stats.items()})
    payload["metadata_json"] = np.array([json.dumps(dict(metadata), sort_keys=True)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _load_checkpoint(path: Path):
    with np.load(Path(path), allow_pickle=False) as payload:
        params = {
            key.removeprefix("param_"): jnp.asarray(payload[key])
            for key in payload.files
            if key.startswith("param_")
        }
        stats = {
            key.removeprefix("stat_"): np.asarray(payload[key])
            for key in payload.files
            if key.startswith("stat_")
        }
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).reshape(-1)[0]))
    if not params or not stats:
        raise ValueError(f"Incomplete low-moment checkpoint: {path}")
    return params, metadata, stats


def _plot_losses(
    train_loss,
    train_ema_loss,
    train_eval_epochs,
    train_eval_loss,
    val_epochs,
    val_loss,
    path: Path,
    metadata: Optional[Mapping[str, object]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    ax.semilogy(
        np.arange(1, len(train_loss) + 1),
        train_loss,
        color="#8b93a1",
        alpha=0.55,
        label="stochastic batch mean",
    )
    if len(train_ema_loss):
        ax.semilogy(
            np.arange(1, len(train_ema_loss) + 1),
            train_ema_loss,
            color="#526071",
            linewidth=1.8,
            label="batch-loss EMA",
        )
    if len(train_eval_loss):
        ax.semilogy(
            train_eval_epochs,
            train_eval_loss,
            color="#172033",
            marker="o",
            label="fixed training panel",
        )
    if len(val_loss):
        ax.semilogy(
            val_epochs,
            val_loss,
            color="#c44e52",
            marker="o",
            label="all held-out ICs at fixed times",
        )
    if metadata:
        warmup = int(metadata.get("supervised_heat_flux_warmup_epochs", 0))
        stages = [(warmup, "heat-flux warm-up")] if warmup else []
        stages.extend(
            (int(epochs), f"H={int(horizon)}")
            for horizon, epochs in metadata.get("horizon_curriculum", [])
        )
        left = 0
        for index, (epochs, label) in enumerate(stages):
            right = left + epochs
            if left:
                ax.axvline(left + 0.5, color="#8b93a1", linewidth=0.8, alpha=0.6)
            ax.text(
                0.5 * (left + right) + 0.5,
                0.02,
                label,
                rotation=90 if epochs <= 5 else 0,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#5f6877",
                transform=ax.get_xaxis_transform(),
            )
            left = right
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized trajectory loss")
    ax.set_title("Fixed-horizon low-moment trajectory training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_horizon_curriculum(
    text: str,
    *,
    final_horizon: int,
    total_epochs: int,
) -> Tuple[Tuple[int, int], ...]:
    if not str(text).strip():
        return ((int(final_horizon), int(total_epochs)),)
    stages = []
    for item in str(text).split(","):
        horizon_text, separator, epochs_text = item.strip().partition(":")
        if not separator:
            raise ValueError("Horizon curriculum entries must have H:epochs form")
        horizon = int(horizon_text)
        epochs = int(epochs_text)
        if horizon <= 0 or epochs <= 0:
            raise ValueError("Horizon curriculum values must be positive")
        stages.append((horizon, epochs))
    if stages[-1][0] != int(final_horizon):
        raise ValueError("The final curriculum horizon must equal --rollout-horizon")
    if sum(epochs for _, epochs in stages) != int(total_epochs):
        raise ValueError("Horizon curriculum epoch counts must sum to --epochs")
    if any(right < left for (left, _), (right, _) in zip(stages, stages[1:])):
        raise ValueError("Horizon curriculum must be nondecreasing")
    return tuple(stages)


def _electric_field_energy(field_hat: np.ndarray, *, nx: int, dx: float) -> np.ndarray:
    """Compute energy in float64 so large finite float32 fields remain diagnosable."""
    field = np.fft.irfft(
        np.asarray(field_hat, dtype=np.complex128), n=int(nx), axis=-1
    )
    with np.errstate(over="ignore", invalid="ignore"):
        return 0.5 * float(dx) * np.sum(field * field, axis=-1, dtype=np.float64)


def _json_float(value: float) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def _evaluate_heldout(
    params,
    grouped,
    manifest,
    coefficient_key: str,
    cache_dir: Path,
    outdir: Path,
    *,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    dt: float,
    width: int,
    memory_steps: int,
    input_scale: np.ndarray,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    poisson_sign: float,
    chunk_steps: int,
    normalized_heat_flux_bound: float,
    density_floor: float,
    pressure_floor: float,
    closure_history_input: bool = False,
) -> None:
    amplitudes_by_id = _case_amplitudes(manifest)
    initial_states = []
    case_records = []
    for regime in REGIMES:
        group = grouped[regime]
        histories = tuple(group[coefficient_key])
        for case_index, (case_id, split) in enumerate(
            zip(group["case_ids"], group["case_splits"])
        ):
            if str(split) != IC_SPLIT_HELDOUT:
                continue
            state = low_hermite_coefficients_to_conservative(
                np.asarray(histories[case_index][0, :3, :]),
                source_nx=source_nx,
                target_nx=rollout_nx,
            )
            initial_states.append(state)
            case_records.append((regime, str(case_id), case_index, histories[case_index]))
    state = jnp.asarray(np.stack(initial_states), dtype=jnp.float32)
    amplitude = jnp.asarray(
        [amplitudes_by_id[record[1]] for record in case_records], dtype=jnp.float32
    )
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(
        int(rollout_nx), d=float(domain_length) / float(rollout_nx)
    )
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    # Training clamps unavailable pre-t=0 history to the initial state. Apply
    # the same causal initialization during autonomous held-out evaluation.
    initial_history = jnp.repeat(state[:, None, :, :], int(memory_steps), axis=1)
    hidden, previous_gradient = warm_spectral_memory(
        params,
        initial_history,
        amplitude,
        k_jax,
        width=width,
        input_scale=jnp.asarray(input_scale, dtype=jnp.float32),
        heat_flux_gradient_scale=heat_flux_gradient_scale,
        amplitude_center=amplitude_center,
        amplitude_scale=amplitude_scale,
        closure_history_input=closure_history_input,
        return_closure_history=True,
        poisson_sign=poisson_sign,
        normalized_heat_flux_bound=normalized_heat_flux_bound,
    )
    field_hat_chunks = [
        np.fft.rfft(
            np.asarray(
                electric_field_from_density(state[:, 0], k_jax, poisson_sign=poisson_sign)
            ),
            axis=-1,
        )
    ]
    case_count = len(case_records)
    minimum_density = np.full((case_count,), np.inf, dtype=np.float64)
    minimum_pressure = np.full((case_count,), np.inf, dtype=np.float64)
    density_limiter_hits = np.zeros((case_count,), dtype=np.int64)
    pressure_limiter_hits = np.zeros((case_count,), dtype=np.int64)
    state_point_count = 0

    def run_chunk(current_state, current_hidden, current_gradient, length: int):
        return rollout_low_moment_closure(
            params,
            current_state,
            current_hidden,
            amplitude,
            k_jax,
            horizon=length,
            dt=dt,
            input_scale=jnp.asarray(input_scale, dtype=jnp.float32),
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            previous_heat_flux_gradient=current_gradient,
            closure_history_input=closure_history_input,
            return_closure_history=True,
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        )

    total_steps = int(round(float(manifest.get("T_final", 0.0)) / float(dt)))
    if total_steps <= 0:
        total_steps = int(case_records[0][3].shape[0]) - 1
    completed = 0
    compiled = {}
    while completed < total_steps:
        length = min(int(chunk_steps), total_steps - completed)
        if length not in compiled:
            compiled[length] = jax.jit(
                lambda s, h, g, n=length: run_chunk(s, h, g, n)
            )
        states, hidden, previous_gradient = compiled[length](
            state, hidden, previous_gradient
        )
        state = states[:, -1]
        states_numpy = np.asarray(states, dtype=np.float64)
        density = 1.0 + states_numpy[:, :, 0]
        momentum = states_numpy[:, :, 1]
        pressure = 1.0 + states_numpy[:, :, 2] - momentum * momentum / density
        minimum_density = np.minimum(
            minimum_density, np.min(density, axis=(1, 2))
        )
        minimum_pressure = np.minimum(
            minimum_pressure, np.min(pressure, axis=(1, 2))
        )
        density_limiter_hits += np.sum(
            density <= 1.01 * float(density_floor), axis=(1, 2)
        )
        pressure_limiter_hits += np.sum(
            pressure <= 1.02 * float(pressure_floor), axis=(1, 2)
        )
        state_point_count += int(length * rollout_nx)
        flat = states.reshape(-1, 3, rollout_nx)
        fields = electric_field_from_density(flat[:, 0], k_jax, poisson_sign=poisson_sign)
        fields = np.asarray(fields).reshape(states.shape[0], length, rollout_nx)
        field_hat_chunks.append(np.fft.rfft(fields, axis=-1).transpose(1, 0, 2))
        completed += length
        print(f"[eval] autonomous held-out rollout: {completed}/{total_steps}")
    learned_hat = np.concatenate(
        [field_hat_chunks[0][None, ...], *field_hat_chunks[1:]], axis=0
    ).transpose(1, 0, 2)
    times = np.arange(total_steps + 1, dtype=np.float64) * float(dt)
    dx = float(domain_length) / float(rollout_nx)
    learned_energy = _electric_field_energy(
        learned_hat, nx=rollout_nx, dx=dx
    )
    growth_metric = EarlyElectricFieldGrowthMetric(
        EarlyGrowthConfig(sample_selector="local_maxima")
    )
    summaries = []
    metric1_traces = []
    eval_root = outdir / "heldout_cases"
    eval_root.mkdir(parents=True, exist_ok=True)
    for row, (regime, case_id, _, _) in enumerate(case_records):
        _, _, snapshot_path = case_shard_paths(cache_dir, case_id)
        with np.load(snapshot_path) as reference:
            hr_times = np.asarray(reference["E_hat_hist_times"], dtype=np.float64)
            hr_hat_full = np.asarray(reference["E_hat_hist"], dtype=np.complex128)
            hr_energy = np.asarray(reference["energy"], dtype=np.float64)
            hr_k = np.asarray(reference["k_arr"], dtype=np.float64)
        restricted_hr_hat = _restrict_rfft(hr_hat_full, source_nx, rollout_nx)
        finite = np.all(np.isfinite(learned_hat[row]), axis=-1) & np.isfinite(
            learned_energy[row]
        )
        finite_count = int(np.argmax(~finite)) if not np.all(finite) else int(finite.size)
        finite_count = max(finite_count, 2)
        reference_energy_max = max(float(np.nanmax(hr_energy)), 1e-30)
        blowup_limit = 1e6 * reference_energy_max
        blowup = np.flatnonzero(learned_energy[row, :finite_count] > blowup_limit)
        failure_index = int(blowup[0]) if blowup.size else finite_count
        bounded_to_final = failure_index == int(times.size)
        failure_time = None if bounded_to_final else float(times[failure_index])
        growth = growth_metric.compare(
            times[:finite_count],
            learned_energy[row, :finite_count],
            hr_times,
            hr_energy,
        )
        field_metric = SelfGeneratedFieldErrorMetric(
            FieldErrorConfig(final_time=times[finite_count - 1])
        )
        field_error = field_metric.evaluate_fourier(
            times[:finite_count],
            np.asarray(learned_hat[row, :finite_count], dtype=np.complex128),
            k_arr,
            hr_times,
            restricted_hr_hat,
            k_arr,
        )
        case_dir = eval_root / case_id
        case_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
        ax.semilogy(hr_times, hr_energy, color="#2463eb", label="kinetic teacher")
        ax.semilogy(times, learned_energy[row], color="#6f3cc3", label="low-moment closure")
        if failure_time is not None:
            ax.axvline(failure_time, color="#c44e52", linestyle="--", label="instability threshold")
        ax.set_xlabel("t")
        ax.set_ylabel("Electric-field energy")
        ax.set_title(f"{case_id} ({regime})")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(case_dir / "metric1_energy.png", dpi=200)
        plt.close(fig)
        learned_complex = np.asarray(learned_hat[row], dtype=np.complex128)
        with np.errstate(over="ignore", invalid="ignore"):
            relative = np.sqrt(
                np.sum(np.abs(learned_complex - restricted_hr_hat) ** 2, axis=-1)
                / np.maximum(
                    np.sum(np.abs(restricted_hr_hat) ** 2, axis=-1), 1e-30
                )
            )
        fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
        ax.semilogy(times, relative, color="#c44e52")
        ax.set_xlabel("t")
        ax.set_ylabel("Relative electric-field error")
        if failure_time is not None:
            ax.axvline(failure_time, color="#c44e52", linestyle="--")
        metric_text = _json_float(field_error.epsilon_E)
        metric_label = "non-finite" if metric_text is None else f"{metric_text:.3e}"
        interval_label = "full" if finite_count == times.size else f"t<={times[finite_count - 1]:.2f}"
        ax.set_title(
            f"{case_id}: integrated $\\varepsilon_E={metric_label}$ ({interval_label})"
        )
        ax.grid(alpha=0.25)
        fig.savefig(case_dir / "metric2_field_error.png", dpi=200)
        plt.close(fig)
        summary = {
            "case_id": case_id,
            "regime": regime,
            "bounded_to_final_time": bounded_to_final,
            "diagnostic_failure_time": failure_time,
            "epsilon_grow": _json_float(growth.epsilon_grow),
            "gamma_hr": _json_float(growth.gamma_grow_hr),
            "gamma_theta": _json_float(growth.gamma_grow_theta),
            "epsilon_E": _json_float(field_error.epsilon_E),
            "epsilon_E_final_time": float(times[finite_count - 1]),
            "maximum_energy_ratio": _json_float(
                np.nanmax(learned_energy[row, :finite_count]) / reference_energy_max
            ),
            "stability_energy_multiplier": 1e6,
            "minimum_density": _json_float(minimum_density[row]),
            "minimum_pressure": _json_float(minimum_pressure[row]),
            "density_limiter_fraction": float(
                density_limiter_hits[row] / max(state_point_count, 1)
            ),
            "pressure_limiter_fraction": float(
                pressure_limiter_hits[row] / max(state_point_count, 1)
            ),
        }
        with (case_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        summaries.append(summary)
        metric1_traces.append(
            (case_id, hr_times, hr_energy, learned_energy[row], failure_time)
        )
        status = "bounded" if bounded_to_final else f"failed@t={failure_time:.2f}"
        print(f"[eval] {case_id}: {status} epsilon_E={metric_label}")
    fig, axes = plt.subplots(
        len(metric1_traces),
        1,
        figsize=(10.0, 2.0 * len(metric1_traces)),
        sharex=True,
        constrained_layout=True,
    )
    for index, (case_id, hr_times, hr_energy, model_energy, failure_time) in enumerate(
        metric1_traces
    ):
        axis = axes[index]
        axis.semilogy(hr_times, hr_energy, color="#2463eb", label="kinetic teacher")
        axis.semilogy(times, model_energy, color="#6f3cc3", label="low-moment closure")
        if failure_time is not None:
            axis.axvline(failure_time, color="#c44e52", linestyle="--")
        axis.set_ylabel(case_id, rotation=0, ha="right", va="center", fontsize=8)
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("t")
    fig.suptitle("Held-out electric-field energy")
    fig.savefig(eval_root / "heldout_metric1_summary.png", dpi=180)
    plt.close(fig)
    aggregate = {
        "bounded_cases": sum(case["bounded_to_final_time"] for case in summaries),
        "case_count": len(summaries),
        "cases": summaries,
    }
    with (eval_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--rollout-Nx", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=1024)
    parser.add_argument(
        "--training-schedule",
        choices=("random_windows", "continuous_trajectories"),
        default="random_windows",
        help="Use teacher-reset windows or complete autonomous IC trajectories",
    )
    parser.add_argument(
        "--horizon-curriculum",
        type=str,
        default="",
        help="Comma-separated H:epochs stages ending at rollout-horizon",
    )
    parser.add_argument("--memory-steps", type=int, default=50)
    parser.add_argument("--memory-stride", type=int, default=10)
    parser.add_argument("--history-stride", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-regime batch size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--supervised-warmup-epochs", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=30)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--spectral-modes", type=int, default=16)
    parser.add_argument("--closure-history-input", action="store_true")
    parser.add_argument("--relative-trajectory-loss", action="store_true")
    parser.add_argument("--stats-stride", type=int, default=20)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--training-diagnostic-cases-per-regime", type=int, default=4)
    parser.add_argument(
        "--diagnostic-start-times",
        type=str,
        default="0,20,40,60,80,100",
    )
    parser.add_argument("--loss-ema-decay", type=float, default=0.95)
    parser.add_argument(
        "--nonfinite-trajectory-penalty",
        type=float,
        default=1e3,
        help=(
            "Fixed normalized loss assigned to an unstable autonomous suffix; "
            "finite-prefix gradients are retained"
        ),
    )
    parser.add_argument(
        "--normalized-heat-flux-bound",
        type=float,
        default=DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    )
    parser.add_argument("--density-floor", type=float, default=DEFAULT_DENSITY_FLOOR)
    parser.add_argument("--pressure-floor", type=float, default=DEFAULT_PRESSURE_FLOOR)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize trainable parameters from a compatible low-moment checkpoint",
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        default=None,
        help="Skip training and regenerate held-out diagnostics from this checkpoint",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--evaluation-chunk-steps", type=int, default=500)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    print_jax_runtime_summary(jax, context="low-moment closure training")
    args = build_arg_parser().parse_args(argv)
    if args.rollout_horizon <= 0 or args.memory_steps <= 0:
        raise ValueError("rollout horizon and memory steps must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    if args.training_schedule == "continuous_trajectories":
        if args.horizon_curriculum.strip():
            raise ValueError("continuous trajectories require a fixed rollout horizon")
        if int(args.supervised_warmup_epochs) != 0:
            raise ValueError("continuous trajectories do not use supervised warm-up")
        if int(args.gradient_accumulation_steps) != 1:
            raise ValueError(
                "continuous trajectories already accumulate every chunk; set "
                "gradient-accumulation-steps=1"
            )
    elif any((args.closure_history_input, args.relative_trajectory_loss)):
        raise ValueError("the scaling/history ablation requires continuous trajectories")
    if not 0.0 <= args.loss_ema_decay < 1.0:
        raise ValueError("loss EMA decay must be in [0, 1)")
    if not math.isfinite(args.nonfinite_trajectory_penalty) or (
        args.nonfinite_trajectory_penalty <= 0.0
    ):
        raise ValueError("nonfinite trajectory penalty must be finite and positive")
    if min(args.normalized_heat_flux_bound, args.density_floor, args.pressure_floor) <= 0:
        raise ValueError("heat-flux bound and admissibility floors must be positive")
    diagnostic_start_times = tuple(
        float(value) for value in args.diagnostic_start_times.split(",") if value.strip()
    )
    if not diagnostic_start_times:
        raise ValueError("at least one diagnostic start time is required")
    curriculum = ()
    if args.evaluate_checkpoint is None:
        if not 0 <= int(args.supervised_warmup_epochs) < int(args.epochs):
            raise ValueError("supervised warm-up epochs must be in [0, epochs)")
        curriculum = parse_horizon_curriculum(
            args.horizon_curriculum,
            final_horizon=args.rollout_horizon,
            total_epochs=args.epochs - args.supervised_warmup_epochs,
        )
    grouped, manifest, cache_metadata, coefficient_key = _load_reference_cache(
        args.reference_cache
    )
    configuration = dict(cache_metadata["configuration"])
    source_nx = int(configuration["teacher_Nx"])
    domain_length = float(configuration["teacher_L"])
    dt = float(configuration["teacher_dt"])
    poisson_sign = float(configuration["teacher_poisson_sign"])
    if int(args.rollout_Nx) > source_nx:
        raise ValueError("rollout-Nx cannot exceed the cached teacher Nx")
    outdir = args.outdir.resolve()
    if args.evaluate_checkpoint is not None:
        params, checkpoint_metadata, checkpoint_stats = _load_checkpoint(
            args.evaluate_checkpoint
        )
        if int(checkpoint_metadata.get("schema_version", 0)) != CHECKPOINT_SCHEMA:
            raise ValueError(
                "Checkpoint uses the obsolete full-state/heat-flux parameterization"
            )
        if str(checkpoint_metadata.get("manifest_sha256")) != str(manifest["sha256"]):
            raise ValueError("Checkpoint and reference-cache IC manifests do not match")
        checkpoint_source_nx = int(checkpoint_metadata["source_Nx"])
        if checkpoint_source_nx != source_nx:
            raise ValueError("Checkpoint and reference-cache source Nx do not match")
        rollout_nx = int(checkpoint_metadata["rollout_Nx"])
        outdir.mkdir(parents=True, exist_ok=True)
        print(
            f"[eval] regenerating held-out diagnostics from {args.evaluate_checkpoint}"
        )
        _evaluate_heldout(
            params,
            grouped,
            manifest,
            coefficient_key,
            Path(args.reference_cache),
            outdir,
            source_nx=source_nx,
            rollout_nx=rollout_nx,
            domain_length=domain_length,
            dt=dt,
            width=int(checkpoint_metadata["width"]),
            memory_steps=int(checkpoint_metadata["memory_steps"]),
            input_scale=checkpoint_stats["input_scale"],
            heat_flux_gradient_scale=float(checkpoint_stats["heat_flux_gradient_scale"][0]),
            amplitude_center=float(checkpoint_stats["amplitude_center"][0]),
            amplitude_scale=float(checkpoint_stats["amplitude_scale"][0]),
            poisson_sign=poisson_sign,
            chunk_steps=args.evaluation_chunk_steps,
            normalized_heat_flux_bound=float(
                checkpoint_metadata.get(
                    "normalized_heat_flux_bound", DEFAULT_NORMALIZED_HEAT_FLUX_BOUND
                )
            ),
            density_floor=float(
                checkpoint_metadata.get("density_floor", DEFAULT_DENSITY_FLOOR)
            ),
            pressure_floor=float(
                checkpoint_metadata.get("pressure_floor", DEFAULT_PRESSURE_FLOOR)
            ),
            closure_history_input=bool(
                checkpoint_metadata.get("closure_history_input", False)
            ),
        )
        metrics_path = outdir / "training_metrics.npz"
        if metrics_path.exists():
            with np.load(metrics_path, allow_pickle=False) as metrics:
                _plot_losses(
                    metrics["train_loss"],
                    metrics["train_ema_loss"]
                    if "train_ema_loss" in metrics.files
                    else np.asarray([]),
                    metrics["train_eval_epochs"]
                    if "train_eval_epochs" in metrics.files
                    else np.asarray([]),
                    metrics["train_eval_loss"]
                    if "train_eval_loss" in metrics.files
                    else np.asarray([]),
                    metrics["val_epochs"],
                    metrics["val_loss"],
                    outdir / "training_loss.png",
                    checkpoint_metadata,
                )
        return
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    anchors = _build_anchor_index(
        grouped,
        coefficient_key,
        horizon=args.rollout_horizon,
        history_stride=args.history_stride,
    )
    for regime in REGIMES:
        print(
            f"[data] {regime}: train_anchors={anchors[regime]['train_times'].size} "
            f"heldout_anchors={anchors[regime]['val_times'].size}"
        )
    stats = load_or_compute_training_statistics(
        args.reference_cache,
        grouped,
        manifest,
        coefficient_key,
        source_nx=source_nx,
        rollout_nx=args.rollout_Nx,
        domain_length=domain_length,
        stats_stride=args.stats_stride,
    )
    print(
        "[data] low-moment input scales: "
        + ", ".join(f"{value:.4e}" for value in stats["input_scale"])
    )
    required_output_bound = float(stats["heat_flux_gradient_max_abs"][0]) / float(
        stats["heat_flux_gradient_scale"][0]
    )
    if float(args.normalized_heat_flux_bound) <= required_output_bound:
        raise ValueError(
            "normalized heat-flux-gradient bound cannot represent the training data: "
            f"configured={args.normalized_heat_flux_bound:.6g} "
            f"required>{required_output_bound:.6g}"
        )
    print(
        "[data] heat-flux-gradient scale/bound: "
        f"scale={float(stats['heat_flux_gradient_scale'][0]):.6e} "
        f"required_normalized_max={required_output_bound:.3f} "
        f"configured_bound={args.normalized_heat_flux_bound:.3f}"
    )
    horizon_description = (
        f"fixed_H={args.rollout_horizon}"
        if len(curriculum) == 1
        else "curriculum=" + ",".join(f"{h}:{e}" for h, e in curriculum)
    )
    print(
        f"[model] width={args.width} modes={args.spectral_modes} "
        f"memory_span={args.memory_steps * args.memory_stride * dt:.3f} "
        f"rollout_span={args.rollout_horizon * dt:.3f} "
        f"warmup={args.supervised_warmup_epochs} {horizon_description}"
    )
    nx = int(args.rollout_Nx)
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(nx, d=domain_length / nx)
    params = init_spectral_memory_params(
        jax.random.PRNGKey(args.seed),
        width=args.width,
        spectral_modes=args.spectral_modes,
        input_channels=5 if args.closure_history_input else 4,
        dtype=jnp.float32,
    )
    if args.init_checkpoint is not None:
        initialized_params, initialized_metadata, _ = _load_checkpoint(
            args.init_checkpoint
        )
        expected = {
            "schema_version": CHECKPOINT_SCHEMA,
            "state_representation": "centered_conservative_v1",
            "closure_output": "zero_mean_heat_flux_gradient",
            "manifest_sha256": str(manifest["sha256"]),
            "source_Nx": source_nx,
            "rollout_Nx": int(args.rollout_Nx),
            "width": int(args.width),
            "spectral_modes": int(args.spectral_modes),
        }
        mismatches = {
            key: (initialized_metadata.get(key), value)
            for key, value in expected.items()
            if initialized_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Incompatible initialization checkpoint: {mismatches}")
        params = initialized_params
        print(f"[train] initialized closure from {args.init_checkpoint}")
    optimizer = _adam_init(params)
    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(args.seed + 1)
    train_history = []
    train_ema_history = []
    train_eval_history = []
    train_eval_regime_history = []
    train_eval_epochs = []
    val_history = []
    val_regime_history = []
    val_epochs = []
    autonomous_val_history = []
    autonomous_val_regime_history = []
    autonomous_val_epochs = []
    best_val = math.inf
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "training_mode": TRAINING_MODE,
        "objective": OBJECTIVE,
        "model_backend": MODEL_BACKEND,
        "reference_cache": str(Path(args.reference_cache).resolve()),
        "manifest_sha256": str(manifest["sha256"]),
        "source_Nx": source_nx,
        "rollout_Nx": args.rollout_Nx,
        "dt": dt,
        "rollout_horizon": args.rollout_horizon,
        "training_schedule": args.training_schedule,
        "continuous_trajectory_final_time": (
            float(configuration["T_final"])
            if args.training_schedule == "continuous_trajectories"
            else None
        ),
        "horizon_curriculum": (
            [] if len(curriculum) == 1 else [list(stage) for stage in curriculum]
        ),
        "supervised_heat_flux_warmup_epochs": args.supervised_warmup_epochs,
        "memory_steps": args.memory_steps,
        "memory_stride": args.memory_stride,
        "width": args.width,
        "spectral_modes": args.spectral_modes,
        "batch_size_per_regime": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "nonfinite_trajectory_penalty": args.nonfinite_trajectory_penalty,
        "init_checkpoint": (
            None
            if args.init_checkpoint is None
            else str(args.init_checkpoint.resolve())
        ),
        "diagnostic_start_times": list(diagnostic_start_times),
        "training_diagnostic_cases_per_regime": (
            args.training_diagnostic_cases_per_regime
        ),
        "regimes": list(REGIMES),
        "normalized_heat_flux_bound": args.normalized_heat_flux_bound,
        "density_floor": args.density_floor,
        "pressure_floor": args.pressure_floor,
        "state_representation": "centered_conservative_v1",
        "closure_output": "zero_mean_heat_flux_gradient",
        "closure_history_input": args.closure_history_input,
        "relative_trajectory_loss": args.relative_trajectory_loss,
    }
    started = time.perf_counter()
    global_epoch = 0
    if int(args.supervised_warmup_epochs) > 0:
        warmup_loss = make_supervised_heat_flux_loss(
            k_arr=k_arr,
            width=args.width,
            input_scale=stats["input_scale"],
            heat_flux_gradient_scale=float(stats["heat_flux_gradient_scale"][0]),
            heat_flux_gradient_regime_scales=stats[
                "heat_flux_gradient_regime_scales"
            ],
            amplitude_center=float(stats["amplitude_center"][0]),
            amplitude_scale=float(stats["amplitude_scale"][0]),
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=args.normalized_heat_flux_bound,
        )
        warmup_value_and_grad = jax.jit(jax.value_and_grad(warmup_loss))
        warmup_validation_batch = sample_batch(
            validation_rng,
            grouped,
            anchors,
            manifest,
            coefficient_key,
            split="val",
            batch_size_per_regime=args.training_diagnostic_cases_per_regime,
            horizon=1,
            memory_steps=args.memory_steps,
            memory_stride=args.memory_stride,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
            translation_augmentation=False,
        )
        warmup_validation_batch_jax = {
            key: jnp.asarray(value) for key, value in warmup_validation_batch.items()
        }
        warmup_evaluate = jax.jit(warmup_loss)
        print(
            f"[train] supervised central-heat-flux warm-up "
            f"epochs={args.supervised_warmup_epochs}"
        )
        for _ in range(int(args.supervised_warmup_epochs)):
            global_epoch += 1
            epoch_losses = []
            for _ in range(int(args.steps_per_epoch)):
                batch = sample_batch(
                    rng,
                    grouped,
                    anchors,
                    manifest,
                    coefficient_key,
                    split="train",
                    batch_size_per_regime=args.batch_size,
                    horizon=1,
                    memory_steps=args.memory_steps,
                    memory_stride=args.memory_stride,
                    source_nx=source_nx,
                    rollout_nx=args.rollout_Nx,
                    domain_length=domain_length,
                    translation_augmentation=True,
                )
                batch_jax = {key: jnp.asarray(value) for key, value in batch.items()}
                loss_value, grads = warmup_value_and_grad(params, batch_jax)
                params, optimizer, grad_norm = _adam_step(
                    params,
                    grads,
                    optimizer,
                    args.learning_rate,
                    args.grad_clip,
                )
                epoch_losses.append(float(loss_value))
            mean_loss = float(np.mean(epoch_losses))
            if not math.isfinite(mean_loss):
                raise FloatingPointError(
                    f"Non-finite heat-flux warm-up loss at epoch {global_epoch}"
                )
            train_history.append(mean_loss)
            val_text = ""
            if (
                global_epoch == 1
                or global_epoch % int(args.validation_every) == 0
                or global_epoch == int(args.supervised_warmup_epochs)
            ):
                val_value = float(
                    warmup_evaluate(params, warmup_validation_batch_jax)
                )
                val_history.append(val_value)
                val_epochs.append(global_epoch)
                val_text = f" heldout={val_value:.6e}"
            elapsed = time.perf_counter() - started
            print(
                f"[train] epoch {global_epoch:04d}/{args.epochs:04d} "
                f"warmup_q={mean_loss:.6e}{val_text} "
                f"grad={float(grad_norm):.3e} elapsed={elapsed / 60.0:.1f}m"
            )
    for stage_horizon, stage_epochs in curriculum:
        loss_fn = make_loss_function(
            k_arr=k_arr,
            width=args.width,
            horizon=stage_horizon,
            dt=dt,
            input_scale=stats["input_scale"],
            regime_scales=stats["regime_scales"],
            heat_flux_gradient_scale=float(stats["heat_flux_gradient_scale"][0]),
            amplitude_center=float(stats["amplitude_center"][0]),
            amplitude_scale=float(stats["amplitude_scale"][0]),
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=args.normalized_heat_flux_bound,
            density_floor=args.density_floor,
            pressure_floor=args.pressure_floor,
            relative_trajectory_loss=args.relative_trajectory_loss,
            closure_history_input=args.closure_history_input,
        )
        value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        evaluate_loss = jax.jit(loss_fn)
        validation_panel = build_diagnostic_panel(
            validation_rng,
            grouped,
            anchors,
            manifest,
            coefficient_key,
            split="val",
            start_times=diagnostic_start_times,
            cases_per_regime=None,
            horizon=stage_horizon,
            dt=dt,
            memory_steps=args.memory_steps,
            memory_stride=args.memory_stride,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
        )
        training_diagnostic_panel = build_diagnostic_panel(
            np.random.default_rng(args.seed + 2),
            grouped,
            anchors,
            manifest,
            coefficient_key,
            split="train",
            start_times=diagnostic_start_times,
            cases_per_regime=args.training_diagnostic_cases_per_regime,
            horizon=stage_horizon,
            dt=dt,
            memory_steps=args.memory_steps,
            memory_stride=args.memory_stride,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
        )
        validation_panel_jax = tuple(
            {key: jnp.asarray(value) for key, value in batch.items()}
            for batch in validation_panel
        )
        training_diagnostic_panel_jax = tuple(
            {key: jnp.asarray(value) for key, value in batch.items()}
            for batch in training_diagnostic_panel
        )

        def evaluate_panel(panel):
            values = []
            regime_values = []
            for panel_batch in panel:
                value, per_regime = evaluate_loss(params, panel_batch)
                values.append(float(value))
                regime_values.append(np.asarray(per_regime, dtype=np.float64))
            return float(np.mean(values)), np.mean(regime_values, axis=0)

        continuous_grad_fns = {}
        continuous_eval_fns = {}

        def get_continuous_chunk_function(chunk_length, warm_memory, *, gradients):
            key = (int(chunk_length), bool(warm_memory))
            cache = continuous_grad_fns if gradients else continuous_eval_fns
            if key not in cache:
                chunk_loss = make_continuous_chunk_loss_function(
                    k_arr=k_arr,
                    width=args.width,
                    horizon=chunk_length,
                    warm_memory=warm_memory,
                    dt=dt,
                    input_scale=stats["input_scale"],
                    regime_scales=stats["regime_scales"],
                    heat_flux_gradient_scale=float(stats["heat_flux_gradient_scale"][0]),
                    amplitude_center=float(stats["amplitude_center"][0]),
                    amplitude_scale=float(stats["amplitude_scale"][0]),
                    poisson_sign=poisson_sign,
                    normalized_heat_flux_bound=args.normalized_heat_flux_bound,
                    density_floor=args.density_floor,
                    pressure_floor=args.pressure_floor,
                    relative_trajectory_loss=args.relative_trajectory_loss,
                    closure_history_input=args.closure_history_input,
                )
                cache[key] = jax.jit(
                    jax.value_and_grad(chunk_loss, has_aux=True)
                    if gradients
                    else chunk_loss
                )
            return cache[key]

        def run_complete_trajectory_batch(batch, *, gradients):
            state = jnp.asarray(batch["initial"])
            memory_or_hidden = jnp.asarray(batch["memory"])
            targets = jnp.asarray(batch["targets"])
            amplitude = jnp.asarray(batch["amplitude"])
            regime_index = jnp.asarray(batch["regime_index"])
            total_steps = int(targets.shape[1])
            if args.relative_trajectory_loss:
                target_shape = targets.shape
                target_fields = primitive_fields(
                    targets.reshape(
                        target_shape[0] * target_shape[1], 3, target_shape[-1]
                    ),
                    jnp.asarray(k_arr, dtype=jnp.float32),
                    poisson_sign=poisson_sign,
                ).reshape(target_shape[0], target_shape[1], 4, target_shape[-1])
                trajectory_target_norm = jnp.sum(
                    jnp.square(target_fields), axis=(1, 3)
                )
                if np.any(np.asarray(trajectory_target_norm) <= 0.0):
                    raise ValueError(
                        "Relative trajectory loss excludes zero-norm target trajectories"
                    )
            else:
                trajectory_target_norm = jnp.ones(
                    (targets.shape[0],), dtype=targets.dtype
                )
            accumulated_grads = None
            accumulated_loss = 0.0
            accumulated_regime = np.zeros((len(REGIMES),), dtype=np.float64)
            completed = 0
            first_chunk = True
            failure_step = None
            while completed < total_steps:
                chunk_length = min(int(stage_horizon), total_steps - completed)
                target_chunk = targets[:, completed : completed + chunk_length]
                function = get_continuous_chunk_function(
                    chunk_length, first_chunk, gradients=gradients
                )
                function_args = (
                    params,
                    state,
                    memory_or_hidden,
                    target_chunk,
                    amplitude,
                    regime_index,
                    trajectory_target_norm,
                )
                if gradients:
                    (
                        (loss_value, (regime_value, final_state, final_hidden)),
                        chunk_grads,
                    ) = function(*function_args)
                else:
                    loss_value, (regime_value, final_state, final_hidden) = function(
                        *function_args
                    )
                    chunk_grads = None
                loss_float = float(loss_value)
                if not math.isfinite(loss_float):
                    failure_step = completed
                    remaining_weight = float(total_steps - completed) / float(
                        total_steps
                    )
                    accumulated_loss += (
                        remaining_weight * args.nonfinite_trajectory_penalty
                    )
                    accumulated_regime += (
                        remaining_weight * args.nonfinite_trajectory_penalty
                    )
                    break
                weight = (
                    1.0
                    if args.relative_trajectory_loss
                    else float(chunk_length) / float(total_steps)
                )
                accumulated_loss += weight * loss_float
                accumulated_regime += weight * np.asarray(
                    regime_value, dtype=np.float64
                )
                if gradients:
                    weighted = jax.tree_util.tree_map(
                        lambda value: weight * value, chunk_grads
                    )
                    accumulated_grads = (
                        weighted
                        if accumulated_grads is None
                        else jax.tree_util.tree_map(
                            lambda left, right: left + right,
                            accumulated_grads,
                            weighted,
                        )
                    )
                state = jax.lax.stop_gradient(final_state)
                memory_or_hidden = jax.lax.stop_gradient(final_hidden)
                completed += chunk_length
                first_chunk = False
            return (
                accumulated_loss,
                accumulated_regime,
                accumulated_grads,
                failure_step,
            )

        def evaluate_complete_split(split):
            case_batches = build_complete_trajectory_case_batches(
                np.random.default_rng(args.seed + (3 if split == "train" else 4)),
                grouped,
                split=split,
                batch_size_per_regime=args.batch_size,
                shuffle=False,
            )
            values = []
            regime_values = []
            weights = []
            failed_batches = 0
            earliest_failure_step = None
            for case_rows in case_batches:
                batch = sample_complete_trajectory_batch(
                    validation_rng,
                    grouped,
                    manifest,
                    coefficient_key,
                    case_rows,
                    memory_steps=args.memory_steps,
                    memory_stride=args.memory_stride,
                    source_nx=source_nx,
                    rollout_nx=args.rollout_Nx,
                    translation_augmentation=False,
                )
                value, per_regime, _, failure_step = run_complete_trajectory_batch(
                    batch, gradients=False
                )
                if failure_step is not None:
                    failed_batches += 1
                    earliest_failure_step = (
                        failure_step
                        if earliest_failure_step is None
                        else min(earliest_failure_step, failure_step)
                    )
                values.append(value)
                regime_values.append(per_regime)
                weights.append(int(case_rows[REGIMES[0]].size))
            return (
                float(np.average(values, weights=weights)),
                np.average(np.asarray(regime_values), axis=0, weights=weights),
                failed_batches,
                earliest_failure_step,
            )

        initial_train_eval, initial_train_regime = evaluate_panel(
            training_diagnostic_panel_jax
        )
        initial_val_eval, initial_val_regime = evaluate_panel(validation_panel_jax)
        metadata["initial_training_panel_loss"] = initial_train_eval
        metadata["initial_heldout_panel_loss"] = initial_val_eval
        metadata["initial_training_panel_regime_loss"] = initial_train_regime.tolist()
        metadata["initial_heldout_panel_regime_loss"] = initial_val_regime.tolist()
        stage_label = "fixed horizon" if len(curriculum) == 1 else "curriculum stage"
        print(
            f"[train] {stage_label} H={stage_horizon} "
            f"epochs={stage_epochs} span={stage_horizon * dt:.3f}"
        )
        print(
            f"[train] initial-model baseline train_panel={initial_train_eval:.6e} "
            f"heldout_panel={initial_val_eval:.6e}"
        )
        initial_autonomous_val = None
        if args.training_schedule == "continuous_trajectories":
            (
                initial_autonomous_val,
                initial_autonomous_regime,
                initial_autonomous_failures,
                initial_autonomous_failure_step,
            ) = evaluate_complete_split("val")
            metadata["initial_heldout_autonomous_loss"] = initial_autonomous_val
            metadata["initial_heldout_autonomous_regime_loss"] = (
                initial_autonomous_regime.tolist()
            )
            metadata["initial_heldout_autonomous_failed_batches"] = (
                initial_autonomous_failures
            )
            print(
                "[train] continuous autonomous schedule: one complete pass over "
                "every training IC per epoch; optimizer updates occur only after "
                "complete T trajectories"
            )
            print(
                f"[train] initial heldout_autonomous={initial_autonomous_val:.6e} "
                f"failed_batches={initial_autonomous_failures} "
                f"earliest_failure_t="
                f"{None if initial_autonomous_failure_step is None else initial_autonomous_failure_step * dt} "
                "regime=("
                + ",".join(f"{value:.3e}" for value in initial_autonomous_regime)
                + ")"
            )
        loss_ema = None
        for _ in range(stage_epochs):
            global_epoch += 1
            epoch_losses = []
            epoch_regime_losses = []
            epoch_grad_norms = []
            epoch_failed_batches = 0
            epoch_earliest_failure_step = None
            if args.training_schedule == "continuous_trajectories":
                training_steps = build_complete_trajectory_case_batches(
                    rng,
                    grouped,
                    split="train",
                    batch_size_per_regime=args.batch_size,
                    shuffle=True,
                )
            else:
                training_steps = range(int(args.steps_per_epoch))
            for step_selection in training_steps:
                if args.training_schedule == "continuous_trajectories":
                    batch = sample_complete_trajectory_batch(
                        rng,
                        grouped,
                        manifest,
                        coefficient_key,
                        step_selection,
                        memory_steps=args.memory_steps,
                        memory_stride=args.memory_stride,
                        source_nx=source_nx,
                        rollout_nx=args.rollout_Nx,
                        translation_augmentation=True,
                    )
                    (
                        optimizer_step_loss,
                        optimizer_step_regime,
                        grads,
                        failure_step,
                    ) = (
                        run_complete_trajectory_batch(batch, gradients=True)
                    )
                    if failure_step is not None:
                        epoch_failed_batches += 1
                        epoch_earliest_failure_step = (
                            failure_step
                            if epoch_earliest_failure_step is None
                            else min(epoch_earliest_failure_step, failure_step)
                        )
                    if grads is None:
                        epoch_losses.append(optimizer_step_loss)
                        epoch_regime_losses.append(optimizer_step_regime)
                        continue
                    params, optimizer, grad_norm = _adam_step(
                        params,
                        grads,
                        optimizer,
                        args.learning_rate,
                        args.grad_clip,
                    )
                    epoch_losses.append(optimizer_step_loss)
                    epoch_regime_losses.append(optimizer_step_regime)
                    loss_ema = (
                        optimizer_step_loss
                        if loss_ema is None
                        else args.loss_ema_decay * loss_ema
                        + (1.0 - args.loss_ema_decay) * optimizer_step_loss
                    )
                    epoch_grad_norms.append(float(grad_norm))
                    continue
                accumulated_grads = None
                optimizer_step_losses = []
                for _ in range(int(args.gradient_accumulation_steps)):
                    batch = sample_batch(
                        rng,
                        grouped,
                        anchors,
                        manifest,
                        coefficient_key,
                        split="train",
                        batch_size_per_regime=args.batch_size,
                        horizon=stage_horizon,
                        memory_steps=args.memory_steps,
                        memory_stride=args.memory_stride,
                        source_nx=source_nx,
                        rollout_nx=args.rollout_Nx,
                        domain_length=domain_length,
                        translation_augmentation=True,
                    )
                    batch_jax = {
                        key: jnp.asarray(value) for key, value in batch.items()
                    }
                    (loss_value, regime_value), grads = value_and_grad(
                        params, batch_jax
                    )
                    loss_float = float(loss_value)
                    if not math.isfinite(loss_float):
                        raise FloatingPointError(
                            f"Non-finite loss at epoch {global_epoch}, H={stage_horizon}"
                        )
                    optimizer_step_losses.append(loss_float)
                    epoch_losses.append(loss_float)
                    epoch_regime_losses.append(
                        np.asarray(regime_value, dtype=np.float64)
                    )
                    accumulated_grads = (
                        grads
                        if accumulated_grads is None
                        else jax.tree_util.tree_map(
                            lambda left, right: left + right,
                            accumulated_grads,
                            grads,
                        )
                    )
                grads = jax.tree_util.tree_map(
                    lambda value: value / float(args.gradient_accumulation_steps),
                    accumulated_grads,
                )
                params, optimizer, grad_norm = _adam_step(
                    params,
                    grads,
                    optimizer,
                    args.learning_rate,
                    args.grad_clip,
                )
                optimizer_step_loss = float(np.mean(optimizer_step_losses))
                loss_ema = (
                    optimizer_step_loss
                    if loss_ema is None
                    else args.loss_ema_decay * loss_ema
                    + (1.0 - args.loss_ema_decay) * optimizer_step_loss
                )
                epoch_grad_norms.append(float(grad_norm))
            mean_loss = float(np.mean(epoch_losses))
            mean_regime_loss = np.mean(epoch_regime_losses, axis=0)
            train_history.append(mean_loss)
            train_ema_history.append(float(loss_ema))
            val_text = ""
            should_validate = (
                global_epoch == 1
                or global_epoch % int(args.validation_every) == 0
                or global_epoch == int(args.epochs)
            )
            if should_validate:
                train_eval_value, train_eval_regime = evaluate_panel(
                    training_diagnostic_panel_jax
                )
                val_value, val_regime = evaluate_panel(validation_panel_jax)
                train_eval_history.append(train_eval_value)
                train_eval_regime_history.append(train_eval_regime)
                train_eval_epochs.append(global_epoch)
                val_history.append(val_value)
                val_regime_history.append(val_regime)
                val_epochs.append(global_epoch)
                val_text = (
                    f" train_panel={train_eval_value:.6e} "
                    f"heldout_panel={val_value:.6e} "
                    f"panel_ratio=({train_eval_value / max(initial_train_eval, 1e-30):.3f},"
                    f"{val_value / max(initial_val_eval, 1e-30):.3f})"
                )
                checkpoint_metric = val_value
                if args.training_schedule == "continuous_trajectories":
                    (
                        autonomous_value,
                        autonomous_regime,
                        autonomous_failures,
                        autonomous_failure_step,
                    ) = evaluate_complete_split("val")
                    autonomous_val_history.append(autonomous_value)
                    autonomous_val_regime_history.append(autonomous_regime)
                    autonomous_val_epochs.append(global_epoch)
                    val_text += (
                        f" heldout_autonomous={autonomous_value:.6e} "
                        f"autonomous_ratio={autonomous_value / max(initial_autonomous_val, 1e-30):.3f}"
                        f" autonomous_failed_batches={autonomous_failures}"
                    )
                    checkpoint_metric = autonomous_value
                if (
                    stage_horizon == args.rollout_horizon
                    and checkpoint_metric < best_val
                ):
                    best_val = checkpoint_metric
                    _save_checkpoint(
                        outdir / "best_low_moment_closure.npz", params, metadata, stats
                    )
            elapsed = time.perf_counter() - started
            print(
                f"[train] epoch {global_epoch:04d}/{args.epochs:04d} "
                f"H={stage_horizon} batch={mean_loss:.6e} "
                f"ema={float(loss_ema):.6e}{val_text} "
                f"regime=({','.join(f'{value:.3e}' for value in mean_regime_loss)}) "
                + (
                    f"failed_batches={epoch_failed_batches} "
                    f"earliest_failure_t="
                    f"{None if epoch_earliest_failure_step is None else epoch_earliest_failure_step * dt} "
                    if args.training_schedule == "continuous_trajectories"
                    else ""
                )
                +
                f"grad_mean={np.mean(epoch_grad_norms):.3e} "
                f"grad_max={np.max(epoch_grad_norms):.3e} "
                f"elapsed={elapsed / 60.0:.1f}m"
            )
            if should_validate:
                print(
                    "[validation] panel_regime train=("
                    + ",".join(f"{value:.3e}" for value in train_eval_regime)
                    + ") heldout=("
                    + ",".join(f"{value:.3e}" for value in val_regime)
                    + ")"
                )
                if args.training_schedule == "continuous_trajectories":
                    print(
                        "[validation] heldout_autonomous_regime=("
                        + ",".join(f"{value:.3e}" for value in autonomous_regime)
                        + ")"
                    )
    checkpoint = outdir / "low_moment_closure.npz"
    _save_checkpoint(checkpoint, params, metadata, stats)
    np.savez(
        outdir / "training_metrics.npz",
        train_loss=np.asarray(train_history),
        train_ema_loss=np.asarray(train_ema_history),
        train_eval_epochs=np.asarray(train_eval_epochs),
        train_eval_loss=np.asarray(train_eval_history),
        train_eval_regime_loss=np.asarray(train_eval_regime_history),
        val_epochs=np.asarray(val_epochs),
        val_loss=np.asarray(val_history),
        val_regime_loss=np.asarray(val_regime_history),
        autonomous_val_epochs=np.asarray(autonomous_val_epochs),
        autonomous_val_loss=np.asarray(autonomous_val_history),
        autonomous_val_regime_loss=np.asarray(autonomous_val_regime_history),
        metadata_json=np.array([json.dumps(metadata, sort_keys=True)]),
    )
    _plot_losses(
        train_history,
        train_ema_history,
        train_eval_epochs,
        train_eval_history,
        val_epochs,
        val_history,
        outdir / "training_loss.png",
        metadata,
    )
    print(f"Saved low-moment closure to {checkpoint}")
    if not args.skip_evaluation:
        _evaluate_heldout(
            params,
            grouped,
            manifest,
            coefficient_key,
            Path(args.reference_cache),
            outdir,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
            dt=dt,
            width=args.width,
            memory_steps=args.memory_steps,
            input_scale=stats["input_scale"],
            heat_flux_gradient_scale=float(stats["heat_flux_gradient_scale"][0]),
            amplitude_center=float(stats["amplitude_center"][0]),
            amplitude_scale=float(stats["amplitude_scale"][0]),
            poisson_sign=poisson_sign,
            chunk_steps=args.evaluation_chunk_steps,
            normalized_heat_flux_bound=args.normalized_heat_flux_bound,
            density_floor=args.density_floor,
            pressure_floor=args.pressure_floor,
            closure_history_input=args.closure_history_input,
        )


if __name__ == "__main__":
    main()
