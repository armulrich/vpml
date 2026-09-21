"""Train a causal spectral-memory closure for a three-moment fluid solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
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
    DYNAMIC_INPUT_SCALING,
    FIXED_INPUT_SCALING,
    INPUT_SCALING_KINDS,
    electric_field_from_density,
    encode_explicit_window_history,
    explicit_window_closure_step,
    init_burles_latent_fno_params,
    init_causal_spacetime_operator_params,
    init_explicit_window_params,
    init_spectral_memory_params,
    init_window_fno_params,
    primitive_fields,
    rollout_low_moment_closure,
    rollout_burles_latent_closure,
    rollout_explicit_window_closure,
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
WINDOW_MEMORY_BACKENDS = frozenset(
    (
        "explicit_window",
        "causal_spacetime_operator",
        "window_fno",
        "burles_latent_fno",
    )
)


def parse_training_regimes(value: str) -> Tuple[str, ...]:
    if value == "all3":
        return REGIMES
    if value == "linear_landau":
        return ("linear_landau",)
    raise ValueError(f"Unsupported training regimes: {value!r}")


def _checkpoint_regimes(metadata: Mapping[str, object]) -> Tuple[str, ...]:
    """Old checkpoints predate selection and therefore expose all three regimes."""
    regimes = tuple(metadata.get("regimes", REGIMES))
    if regimes not in (REGIMES, ("linear_landau",)):
        raise ValueError(f"Unsupported checkpoint regimes: {regimes}")
    if "training_regimes" in metadata and parse_training_regimes(
        str(metadata["training_regimes"])
    ) != regimes:
        raise ValueError("Checkpoint training_regimes and regimes disagree")
    return regimes


def _validate_regime_resume_configuration(saved, current) -> None:
    if _checkpoint_regimes(saved) != _checkpoint_regimes(current):
        raise ValueError("Resume configuration mismatch: training regimes changed")
    # Missing fields in legacy all3 runs mean the original all3 statistics.
    for key, default in (
        ("normalization_policy", "fixed_all3_training_statistics"),
        ("normalization_regimes", list(REGIMES)),
        ("normalization_checkpoint", None),
        ("normalization_checkpoint_sha256", None),
    ):
        if saved.get(key, default) != current.get(key, default):
            raise ValueError(f"Resume configuration mismatch: {key}")
    for key in ("stats_stride", "normalization_statistics_sha256"):
        if key in saved and saved[key] != current.get(key):
            raise ValueError(f"Resume configuration mismatch: {key}")


def _uses_window_memory(memory_backend: str) -> bool:
    return memory_backend in WINDOW_MEMORY_BACKENDS


def _uses_compact_latent(memory_backend: str) -> bool:
    return memory_backend == "burles_latent_fno"


def _rollout_window_memory(
    params,
    initial_state,
    history,
    history_counter,
    amplitude,
    k_arr,
    *,
    memory_backend: str,
    compact_latent=None,
    **kwargs,
):
    """Dispatch window rollouts without changing legacy backend semantics."""
    if _uses_compact_latent(memory_backend):
        kwargs.pop("encoded_history", None)
        kwargs.pop("closure_history_input", None)
        return rollout_burles_latent_closure(
            params,
            initial_state,
            history,
            history_counter,
            amplitude,
            k_arr,
            compact_latent=compact_latent,
            **kwargs,
        )
    return rollout_explicit_window_closure(
        params,
        initial_state,
        history,
        history_counter,
        amplitude,
        k_arr,
        **kwargs,
    )


def _unpack_window_memory(memory, memory_backend: str):
    if _uses_compact_latent(memory_backend):
        history, closure, encoded, counter, gradient, latent = memory
        return history, closure, encoded, counter, gradient, latent
    history, closure, encoded, counter, gradient = memory
    return history, closure, encoded, counter, gradient, None


def _load_convergence_floor(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        floor_rms = np.asarray(payload["floor_rms"], dtype=np.float64)
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).reshape(-1)[0]))
    if floor_rms.shape != (len(REGIMES), 4):
        raise ValueError(
            "Convergence floor must have shape (3 regimes, 4 primitive channels), "
            f"got {floor_rms.shape}"
        )
    if not np.all(np.isfinite(floor_rms)) or np.any(floor_rms <= 0.0):
        raise ValueError("Convergence floor RMS values must be finite and positive")
    return floor_rms, metadata


def _block_relative_sample_loss(
    predicted_fields,
    target_fields,
    regime_index,
    target_indices,
    *,
    block_steps: int,
    block_count: int,
    floor_rms,
    trajectory_target_norm=None,
):
    """Relative primitive-field error in fixed physical-time blocks."""
    # Late-time linear Landau signals are far below the squared float32 range.
    # Accumulate this diagnostic/training normalization in float64 so the
    # measured teacher-grid convergence floor remains effective.
    predicted_fields = jnp.asarray(predicted_fields, dtype=jnp.float64)
    target_fields = jnp.asarray(target_fields, dtype=jnp.float64)
    floor_rms = jnp.asarray(floor_rms, dtype=jnp.float64)
    if trajectory_target_norm is not None:
        trajectory_target_norm = jnp.asarray(
            trajectory_target_norm, dtype=jnp.float64
        )
    squared_error = jnp.sum(jnp.square(predicted_fields - target_fields), axis=-1)
    block_ids = jnp.minimum(
        jnp.maximum((target_indices - 1) // int(block_steps), 0),
        int(block_count) - 1,
    )
    membership = jax.nn.one_hot(block_ids, int(block_count), dtype=squared_error.dtype)
    numerator = jnp.einsum("btq,btc->bqc", membership, squared_error)
    sample_counts = jnp.sum(membership, axis=1)
    supplied_target_norm = trajectory_target_norm is not None
    if not supplied_target_norm:
        target_energy = jnp.sum(jnp.square(target_fields), axis=-1)
        trajectory_target_norm = jnp.einsum(
            "btq,btc->bqc", membership, target_energy
        )
    channel_floor = floor_rms[regime_index]
    floor_energy = (
        sample_counts[:, :, None]
        * predicted_fields.shape[-1]
        * jnp.square(channel_floor[:, None, :])
    )
    denominator = (
        trajectory_target_norm
        if supplied_target_norm
        else jnp.maximum(trajectory_target_norm, floor_energy)
    )
    valid = sample_counts > 0
    safe_denominator = jnp.where(valid[:, :, None], denominator, 1.0)
    ratios = numerator / safe_denominator
    divisor = (
        int(block_count)
        if supplied_target_norm
        else jnp.maximum(jnp.sum(valid, axis=1), 1)
    )
    return jnp.sum(jnp.where(valid[:, :, None], ratios, 0.0), axis=(1, 2)) / (
        divisor * predicted_fields.shape[2]
    )


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


def _primitive_numpy(
    state: np.ndarray,
    domain_length: float,
    *,
    poisson_sign: float = 1.0,
) -> np.ndarray:
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
    field_hat[..., 1:] = float(poisson_sign) * 1j * rho_hat[..., 1:] / k_arr[1:]
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
    regimes: Sequence[str] = REGIMES,
    horizon: int,
    history_stride: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for regime in regimes:
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


def parse_train_case_limits(value: str) -> Dict[str, int]:
    limits: Dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("training case limits must use regime=count entries")
        regime, raw_count = (part.strip() for part in item.split("=", 1))
        if regime not in REGIMES:
            raise ValueError(f"Unknown training-case-limit regime: {regime}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError("training case limits must be positive")
        limits[regime] = count
    return limits


def limit_training_anchors(
    anchors: Mapping[str, Mapping[str, np.ndarray]],
    case_limits: Mapping[str, int],
    *,
    regimes: Sequence[str] = REGIMES,
    seed: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    limited: Dict[str, Dict[str, np.ndarray]] = {}
    selected_cases: Dict[str, np.ndarray] = {}
    for regime in regimes:
        regime_index = REGIMES.index(regime)
        payload = {
            key: np.asarray(value).copy()
            for key, value in anchors[regime].items()
        }
        available = np.unique(payload["train_cases"])
        count = min(int(case_limits.get(regime, available.size)), int(available.size))
        permutation = np.random.default_rng(
            int(seed) + 1009 * (regime_index + 1)
        ).permutation(available)
        selected = np.sort(permutation[:count]).astype(np.int32)
        mask = np.isin(payload["train_cases"], selected)
        payload["train_cases"] = payload["train_cases"][mask]
        payload["train_times"] = payload["train_times"][mask]
        limited[regime] = payload
        selected_cases[regime] = selected
    return limited, selected_cases


def _gather_coefficients(
    histories: Sequence[np.ndarray],
    case_indices: np.ndarray,
    time_indices: np.ndarray,
    *,
    coefficient_count: int,
) -> np.ndarray:
    cases, times = np.broadcast_arrays(
        np.asarray(case_indices, dtype=np.int32),
        np.asarray(time_indices, dtype=np.float64),
    )
    sample = np.asarray(histories[0][0, :coefficient_count])
    output = np.empty(cases.shape + sample.shape, dtype=sample.dtype)
    flat_cases = cases.reshape(-1)
    flat_times = times.reshape(-1)
    flat_output = output.reshape((flat_cases.size,) + sample.shape)
    for case_index in np.unique(flat_cases):
        positions = np.flatnonzero(flat_cases == int(case_index))
        ordered = positions[np.argsort(flat_times[positions], kind="stable")]
        selected_times = flat_times[ordered]
        left = np.floor(selected_times + 1e-10).astype(np.int64)
        fraction = selected_times - left
        right = np.minimum(left + 1, int(histories[int(case_index)].shape[0]) - 1)
        left_values = histories[int(case_index)][left, :coefficient_count, :]
        if np.all(np.abs(fraction) < 1e-12):
            flat_output[ordered] = left_values
        else:
            right_values = histories[int(case_index)][right, :coefficient_count, :]
            interpolated = left_values + fraction[:, None, None] * (
                right_values - left_values
            )
            flat_output[ordered] = np.asarray(interpolated, dtype=sample.dtype)
    return output


def _statistics_path(cache_dir: Path, rollout_nx: int, stats_stride: int) -> Path:
    key = sha256_json(
        {
            "kind": "low_moment_spectral_memory_stats_v4_case_gradient_rms",
            "rollout_nx": int(rollout_nx),
            "stats_stride": int(stats_stride),
        }
    )[:20]
    return Path(cache_dir) / "derived" / f"{key}.npz"


def _primitive_target_cache_root(
    cache_dir: Path,
    manifest: Mapping[str, object],
    *,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    poisson_sign: float,
) -> Path:
    key = sha256_json(
        {
            "kind": "low_moment_primitive_targets_v1",
            "manifest_sha256": str(manifest["sha256"]),
            "source_nx": int(source_nx),
            "rollout_nx": int(rollout_nx),
            "domain_length": float(domain_length),
            "poisson_sign": float(poisson_sign),
            "dtype": "float32",
        }
    )[:20]
    return Path(cache_dir) / "derived" / f"low_moment_primitive_targets_{key}"


def _load_or_build_primitive_target_cache(
    cache_dir: Path,
    grouped: Mapping[str, Mapping[str, object]],
    manifest: Mapping[str, object],
    coefficient_key: str,
    *,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    poisson_sign: float,
    chunk_steps: int = 128,
) -> Tuple[Dict[str, Tuple[np.ndarray, ...]], Dict[str, np.ndarray], Path]:
    """Build reusable primitive-field targets and full-trajectory channel norms."""
    root = _primitive_target_cache_root(
        cache_dir,
        manifest,
        source_nx=source_nx,
        rollout_nx=rollout_nx,
        domain_length=domain_length,
        poisson_sign=poisson_sign,
    )
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    chunk_steps = max(int(chunk_steps), 1)
    metadata_path = root / "metadata.json"
    stored_case_metadata: Mapping[str, object] = {}
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                stored_metadata = json.load(handle)
            if (
                int(stored_metadata.get("schema_version", 0)) == 1
                and str(stored_metadata.get("manifest_sha256"))
                == str(manifest["sha256"])
                and int(stored_metadata.get("source_nx", -1)) == int(source_nx)
                and int(stored_metadata.get("rollout_nx", -1)) == int(rollout_nx)
            ):
                stored_case_metadata = dict(stored_metadata.get("cases", {}))
        except (OSError, ValueError, TypeError):
            stored_case_metadata = {}
    target_histories: Dict[str, Tuple[np.ndarray, ...]] = {}
    trajectory_norms: Dict[str, np.ndarray] = {}
    case_metadata: Dict[str, object] = {}
    for regime in REGIMES:
        group = grouped[regime]
        histories = tuple(group[coefficient_key])
        case_ids = np.asarray(group["case_ids"], dtype=np.str_)
        cached_cases = []
        for history, case_id_value in zip(histories, case_ids):
            case_id = str(case_id_value)
            path = cases_dir / f"{case_id}.npy"
            expected_shape = (int(history.shape[0]), 4, int(rollout_nx))
            valid = False
            if path.exists():
                try:
                    candidate = np.load(path, mmap_mode="r", allow_pickle=False)
                    valid = candidate.shape == expected_shape and candidate.dtype == np.float32
                    del candidate
                except (OSError, ValueError):
                    valid = False
            if not valid:
                temporary = path.with_name(f".{path.name}.tmp")
                if temporary.exists():
                    temporary.unlink()
                output = np.lib.format.open_memmap(
                    temporary,
                    mode="w+",
                    dtype=np.float32,
                    shape=expected_shape,
                )
                for start in range(0, expected_shape[0], chunk_steps):
                    stop = min(start + chunk_steps, expected_shape[0])
                    coefficients = np.asarray(history[start:stop, :3, :])
                    state = low_hermite_coefficients_to_conservative(
                        coefficients,
                        source_nx=source_nx,
                        target_nx=rollout_nx,
                        dtype=np.float32,
                    )
                    output[start:stop] = _primitive_numpy(
                        state,
                        domain_length,
                        poisson_sign=poisson_sign,
                    ).astype(np.float32)
                output.flush()
                del output
                os.replace(temporary, path)
                print(f"[data] cached primitive targets for {case_id}")
            cached = np.load(path, mmap_mode="r", allow_pickle=False)
            stored_case = stored_case_metadata.get(case_id, {})
            stored_norm = (
                stored_case.get("trajectory_target_norm")
                if isinstance(stored_case, Mapping)
                else None
            )
            norm = np.asarray(stored_norm, dtype=np.float64) if stored_norm is not None else None
            if norm is None or norm.shape != (4,):
                norm = np.sum(
                    np.square(np.asarray(cached[1:], dtype=np.float64)),
                    axis=(0, 2),
                    dtype=np.float64,
                )
            if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
                raise ValueError(f"Primitive target cache has invalid norm for {case_id}")
            cached_cases.append(cached)
            trajectory_norms[case_id] = norm
            case_metadata[case_id] = {
                "shape": list(expected_shape),
                "dtype": "float32",
                "trajectory_target_norm": norm.tolist(),
            }
        target_histories[regime] = tuple(cached_cases)
    metadata = {
        "schema_version": 1,
        "manifest_sha256": str(manifest["sha256"]),
        "source_nx": int(source_nx),
        "rollout_nx": int(rollout_nx),
        "domain_length": float(domain_length),
        "poisson_sign": float(poisson_sign),
        "cases": case_metadata,
    }
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_metadata, metadata_path)
    return target_histories, trajectory_norms, root


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
    heat_flux_gradient_case_ids = []
    heat_flux_gradient_case_scales = []
    train_log_amplitudes = []
    for regime in REGIMES:
        group = grouped[regime]
        histories = tuple(group[coefficient_key])
        case_ids = np.asarray(group["case_ids"], dtype=np.str_)
        splits = np.asarray(group["case_splits"], dtype=np.str_)
        for history, case_id, split in zip(histories, case_ids, splits):
            is_training_case = str(split) == IC_SPLIT_TRAIN
            if is_training_case:
                train_log_amplitudes.append(math.log(amplitudes[str(case_id)]))
            case_gradient_sum = 0.0
            case_gradient_count = 0
            indices = np.arange(0, int(history.shape[0]), max(int(stats_stride), 1))
            for start in range(0, int(indices.size), 64):
                rows = indices[start : start + 64]
                coeff = np.asarray(history[rows, :4, :])
                state = low_hermite_coefficients_to_conservative(
                    coeff, source_nx=source_nx, target_nx=rollout_nx, dtype=np.float64
                )
                fields = _primitive_numpy(state, domain_length)
                if is_training_case:
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
                gradient_square_sum = float(
                    np.sum(heat_flux_gradient * heat_flux_gradient, dtype=np.float64)
                )
                case_gradient_sum += float(
                    gradient_square_sum
                )
                case_gradient_count += int(heat_flux_gradient.size)
                if is_training_case:
                    heat_flux_gradient_sum[regime] += gradient_square_sum
                    heat_flux_gradient_count[regime] += int(heat_flux_gradient.size)
                    heat_flux_gradient_sum_global += gradient_square_sum
                    heat_flux_gradient_count_global += int(heat_flux_gradient.size)
                    heat_flux_gradient_max_abs = max(
                        heat_flux_gradient_max_abs,
                        float(np.max(np.abs(heat_flux_gradient))),
                    )
            heat_flux_gradient_case_ids.append(str(case_id))
            heat_flux_gradient_case_scales.append(
                max(math.sqrt(case_gradient_sum / max(case_gradient_count, 1)), 1e-8)
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
        "heat_flux_gradient_case_ids": np.asarray(
            heat_flux_gradient_case_ids, dtype=np.str_
        ),
        "heat_flux_gradient_case_scales": np.asarray(
            heat_flux_gradient_case_scales, dtype=np.float64
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


def _load_normalization_statistics(
    path: Path,
    *,
    manifest: Mapping[str, object],
    source_nx: int,
    rollout_nx: int,
) -> Tuple[Dict[str, np.ndarray], str]:
    """Read stats only; canonical regime axes are independent of active regimes."""
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as payload:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).reshape(-1)[0]))
            stats = {
                key.removeprefix("stat_"): np.asarray(payload[key])
                for key in payload.files if key.startswith("stat_")
            }
    expected = {
        "schema_version": CHECKPOINT_SCHEMA,
        "manifest_sha256": str(manifest["sha256"]),
        "source_Nx": int(source_nx),
        "rollout_Nx": int(rollout_nx),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Normalization checkpoint incompatible {key}: {metadata.get(key)!r}")
    if metadata.get("normalization_policy", "fixed_all3_training_statistics") != (
        "fixed_all3_training_statistics"
    ) or tuple(metadata.get("normalization_regimes", metadata.get("regimes", REGIMES))) != REGIMES:
        raise ValueError("Normalization checkpoint must contain fixed all3 statistics")
    shapes = {
        "input_scale": (4,),
        "regime_scales": (len(REGIMES), 4),
        "heat_flux_gradient_scale": (1,),
        "heat_flux_gradient_regime_scales": (len(REGIMES),),
        "heat_flux_gradient_max_abs": (1,),
        "amplitude_center": (1,),
        "amplitude_scale": (1,),
    }
    for key, shape in shapes.items():
        if key not in stats or stats[key].shape != shape:
            raise ValueError(f"Normalization statistic {key} must have shape {shape}")
        value = stats[key]
        if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
            raise ValueError(f"Normalization statistic {key} must be real numeric")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Normalization statistic {key} must be finite")
        if key.endswith("scale") or key.endswith("scales"):
            if np.any(value <= 0):
                raise ValueError(f"Normalization statistic {key} must be positive")
    if stats["heat_flux_gradient_max_abs"][0] < 0:
        raise ValueError("Normalization heat_flux_gradient_max_abs must be nonnegative")
    ids = stats.get("heat_flux_gradient_case_ids", np.asarray([]))
    scales = stats.get("heat_flux_gradient_case_scales", np.asarray([]))
    expected_ids = {str(case["case_id"]) for case in manifest["cases"]}
    if (
        ids.ndim != 1 or ids.dtype.kind not in "US"
        or len(set(ids.tolist())) != ids.size
        or set(ids.tolist()) != expected_ids
        or scales.shape != ids.shape
        or not np.issubdtype(scales.dtype, np.number)
        or np.iscomplexobj(scales)
        or not np.all(np.isfinite(scales)) or np.any(scales <= 0)
    ):
        raise ValueError("Normalization case IDs/scales must cover the manifest with positive RMS")
    return stats, digest


def sample_batch(
    rng: np.random.Generator,
    grouped,
    anchors,
    manifest,
    coefficient_key: str,
    *,
    regimes: Sequence[str] = REGIMES,
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
    heat_flux_gradient_case_scale_by_id: Optional[Mapping[str, float]] = None,
    include_heat_flux_gradient_history: bool = False,
    autonomous_burnin_steps: int = 0,
    reference_steps_per_solver_step: float = 1.0,
) -> Dict[str, np.ndarray]:
    amplitudes_by_id = _case_amplitudes(manifest)
    memories = []
    initials = []
    targets = []
    amplitudes = []
    regime_indices = []
    heat_flux_gradient_targets = []
    heat_flux_gradient_histories = []
    heat_flux_gradient_case_scales = []
    start_indices = []
    for regime in regimes:
        regime_index = REGIMES.index(regime)
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
        burnin_steps = int(autonomous_burnin_steps)
        reference_ratio = float(reference_steps_per_solver_step)
        if reference_ratio <= 0.0:
            raise ValueError("reference_steps_per_solver_step must be positive")
        if burnin_steps:
            # A random training reset supplies only the three physical moments.
            # The closure history is then generated by the model itself before
            # any scored target is observed.
            memory_times = np.broadcast_to(
                times[:, None], (times.size, int(memory_steps))
            )
        else:
            memory_offsets = float(memory_stride) * reference_ratio * np.arange(
                int(memory_steps), 0, -1, dtype=np.float64
            )
            raw_memory_times = times[:, None] - memory_offsets[None, :]
            padded_memory = raw_memory_times < 0
            memory_times = np.maximum(raw_memory_times, 0)
        target_times = times[:, None] + reference_ratio * (
            burnin_steps
            + np.arange(1, int(horizon) + 1, dtype=np.float64)[None]
        )
        memory_coefficient_count = 4 if include_heat_flux_gradient_history else 3
        memory_coeff = _gather_coefficients(
            histories,
            cases[:, None],
            memory_times,
            coefficient_count=memory_coefficient_count,
        )
        initial_coeff = _gather_coefficients(
            histories, cases, times, coefficient_count=4
        )
        target_coeff = _gather_coefficients(
            histories, cases[:, None], target_times, coefficient_count=3
        )
        memory_state = low_hermite_coefficients_to_conservative(
            memory_coeff, source_nx=source_nx, target_nx=rollout_nx
        )
        if burnin_steps:
            # A random reset exposes only its current physical state. Generate
            # the causal history from an equilibrium-padded buffer.
            memory_state[...] = 0.0
        else:
            # Before t=0 the causal history is the centered equilibrium state,
            # rather than a fictitious copy of the perturbed initial condition.
            memory_state[padded_memory] = 0.0
        memories.append(memory_state)
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
        if include_heat_flux_gradient_history:
            memory_state = low_hermite_coefficients_to_conservative(
                memory_coeff,
                source_nx=source_nx,
                target_nx=rollout_nx,
                dtype=np.float64,
            )
            memory_heat_flux = _central_heat_flux_numpy(
                memory_coeff,
                memory_state,
                source_nx=source_nx,
                target_nx=rollout_nx,
            )
            memory_gradient = np.fft.irfft(
                1j * k_arr * np.fft.rfft(memory_heat_flux, axis=-1),
                n=int(rollout_nx),
                axis=-1,
            )
            if not burnin_steps:
                memory_gradient[padded_memory] = 0.0
            else:
                memory_gradient[...] = 0.0
            heat_flux_gradient_histories.append(memory_gradient)
        if heat_flux_gradient_case_scale_by_id is not None:
            heat_flux_gradient_case_scales.append(
                np.asarray(
                    [
                        heat_flux_gradient_case_scale_by_id[str(case_ids[index])]
                        for index in cases
                    ],
                    dtype=np.float64,
                )
            )
        amplitudes.append(
            np.asarray([amplitudes_by_id[str(case_ids[index])] for index in cases])
        )
        regime_indices.append(np.full((len(cases),), regime_index, dtype=np.int32))
        start_indices.append(
            np.rint(times / reference_ratio).astype(np.int32) + burnin_steps
        )
    batch = {
        "memory": np.concatenate(memories, axis=0).astype(np.float32),
        "initial": np.concatenate(initials, axis=0).astype(np.float32),
        "targets": np.concatenate(targets, axis=0).astype(np.float32),
        "amplitude": np.concatenate(amplitudes, axis=0).astype(np.float32),
        "regime_index": np.concatenate(regime_indices, axis=0),
        "start_index": np.concatenate(start_indices, axis=0).astype(np.int32),
        "heat_flux_gradient_target": np.concatenate(
            heat_flux_gradient_targets, axis=0
        ).astype(np.float32),
    }
    if include_heat_flux_gradient_history:
        batch["heat_flux_gradient_history"] = np.concatenate(
            heat_flux_gradient_histories, axis=0
        ).astype(np.float32)
    if heat_flux_gradient_case_scale_by_id is not None:
        batch["heat_flux_gradient_case_scale"] = np.concatenate(
            heat_flux_gradient_case_scales, axis=0
        ).astype(np.float32)
    if translation_augmentation:
        shifts = rng.integers(0, int(rollout_nx), size=batch["initial"].shape[0])
        for row, shift in enumerate(shifts):
            batch["memory"][row] = np.roll(batch["memory"][row], int(shift), axis=-1)
            batch["initial"][row] = np.roll(batch["initial"][row], int(shift), axis=-1)
            batch["targets"][row] = np.roll(batch["targets"][row], int(shift), axis=-1)
            batch["heat_flux_gradient_target"][row] = np.roll(
                batch["heat_flux_gradient_target"][row], int(shift), axis=-1
            )
            if include_heat_flux_gradient_history:
                batch["heat_flux_gradient_history"][row] = np.roll(
                    batch["heat_flux_gradient_history"][row], int(shift), axis=-1
                )
    return batch


def build_complete_trajectory_case_batches(
    rng: np.random.Generator,
    grouped,
    *,
    regimes: Sequence[str] = REGIMES,
    split: str,
    batch_size_per_regime: int,
    shuffle: bool,
) -> Sequence[Dict[str, np.ndarray]]:
    """Return balanced case batches that cover every IC in a split once."""
    split_value = IC_SPLIT_TRAIN if split == "train" else IC_SPLIT_HELDOUT
    rows_by_regime = {}
    for regime in regimes:
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
            for regime in regimes
        }
        for start in range(0, case_count, batch_size)
    )


def build_balanced_random_window_epoch(
    rng: np.random.Generator,
    anchors,
    *,
    regimes: Sequence[str] = REGIMES,
    split: str,
    batch_size_per_regime: int,
    time_blocks: int = 6,
    deterministic_epoch: Optional[int] = None,
) -> Sequence[Mapping[str, Tuple[np.ndarray, np.ndarray]]]:
    """Cover every available IC exactly once with balanced random windows."""
    batch_size = int(batch_size_per_regime)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    cases_by_regime = {}
    for regime in regimes:
        case_pool = np.asarray(anchors[regime][f"{split}_cases"], dtype=np.int32)
        unique = np.unique(case_pool)
        cases_by_regime[regime] = (
            unique if deterministic_epoch is not None else rng.permutation(unique)
        )
    counts = {len(value) for value in cases_by_regime.values()}
    if len(counts) != 1:
        raise ValueError(f"Balanced random windows require equal regime counts: {counts}")
    count = counts.pop()
    selections = []
    for start in range(0, count, batch_size):
        selection = {}
        for regime in regimes:
            cases = cases_by_regime[regime][start : start + batch_size]
            case_pool = np.asarray(anchors[regime][f"{split}_cases"], dtype=np.int32)
            time_pool = np.asarray(anchors[regime][f"{split}_times"], dtype=np.int32)
            times = []
            for offset, case_index in enumerate(cases):
                available = time_pool[case_pool == int(case_index)]
                if deterministic_epoch is not None:
                    position = (
                        int(deterministic_epoch) - 1 + 131 * int(case_index)
                    ) % int(available.size)
                    times.append(available[position])
                    continue
                block_count = max(int(time_blocks), 1)
                block_index = (start // batch_size + offset) % block_count
                edges = np.linspace(0, available.size, block_count + 1, dtype=np.int32)
                left, right = int(edges[block_index]), int(edges[block_index + 1])
                if right <= left:
                    left, right = 0, int(available.size)
                times.append(available[int(rng.integers(left, right))])
            selection[regime] = (cases, np.asarray(times, dtype=np.int32))
        selections.append(selection)
    return tuple(selections)


def _full_anchor_steps_per_epoch(anchors, regimes, batch_size, accumulation_steps):
    """Count complete selected-regime updates without consuming the training RNG."""
    counts = {int(anchors[regime]["train_cases"].size) for regime in regimes}
    if len(counts) != 1 or min(counts) <= 0:
        raise ValueError(f"Full-anchor sweep requires equal nonempty regime counts: {counts}")
    count = counts.pop()
    samples_per_update = int(batch_size) * int(accumulation_steps)
    if samples_per_update <= 0 or count % samples_per_update:
        raise ValueError(
            "Full-anchor count must divide by batch size * gradient accumulation; "
            f"got {count} and {samples_per_update}"
        )
    return count // samples_per_update


def build_balanced_full_anchor_epoch(
    rng: np.random.Generator,
    anchors,
    *,
    regimes: Sequence[str] = REGIMES,
    split: str,
    batch_size_per_regime: int,
) -> Sequence[Mapping[str, Tuple[np.ndarray, np.ndarray]]]:
    """Shuffle and consume every anchor exactly once in balanced minibatches."""
    batch_size = int(batch_size_per_regime)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    shuffled = {}
    counts = set()
    for regime in regimes:
        cases = np.asarray(anchors[regime][f"{split}_cases"], dtype=np.int32)
        times = np.asarray(anchors[regime][f"{split}_times"], dtype=np.int32)
        if cases.size != times.size or cases.size == 0:
            raise ValueError(f"Invalid {split} anchor table for {regime}")
        order = rng.permutation(cases.size)
        shuffled[regime] = (cases[order], times[order])
        counts.add(int(cases.size))
    if len(counts) != 1:
        raise ValueError(f"Balanced full sweep requires equal regime counts: {counts}")
    count = counts.pop()
    if count % batch_size:
        raise ValueError(
            "Full-anchor sweep requires the per-regime anchor count to be divisible "
            f"by batch size; got {count} and {batch_size}"
        )
    return tuple(
        {
            regime: (
                shuffled[regime][0][start : start + batch_size],
                shuffled[regime][1][start : start + batch_size],
            )
            for regime in regimes
        }
        for start in range(0, count, batch_size)
    )


def sample_complete_trajectory_batch(
    rng: np.random.Generator,
    grouped,
    manifest,
    coefficient_key: str,
    case_rows: Mapping[str, np.ndarray],
    *,
    regimes: Sequence[str] = REGIMES,
    memory_steps: int,
    memory_stride: int,
    source_nx: int,
    rollout_nx: int,
    translation_augmentation: bool,
    primitive_target_histories: Optional[
        Mapping[str, Sequence[np.ndarray]]
    ] = None,
    trajectory_target_norm_by_id: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Load complete, balanced trajectories initialized only at ``t=0``."""
    del memory_stride  # Every pre-initial memory sample is clamped to t=0.
    amplitudes_by_id = _case_amplitudes(manifest)
    memories = []
    initials = []
    targets = []
    target_fields = []
    target_norms = []
    amplitudes = []
    regime_indices = []
    trajectory_steps = None
    for regime in regimes:
        regime_index = REGIMES.index(regime)
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
        initial_times = np.zeros((cases.size,), dtype=np.int32)
        initial_coeff = _gather_coefficients(
            histories, cases, initial_times, coefficient_count=4
        )
        memories.append(
            np.zeros(
                (cases.size, int(memory_steps), 3, int(rollout_nx)),
                dtype=np.float64,
            )
        )
        initials.append(
            low_hermite_coefficients_to_conservative(
                initial_coeff, source_nx=source_nx, target_nx=rollout_nx
            )
        )
        if primitive_target_histories is None:
            target_times = np.broadcast_to(
                np.arange(1, available_steps + 1, dtype=np.int32)[None, :],
                (cases.size, available_steps),
            )
            target_coeff = _gather_coefficients(
                histories, cases[:, None], target_times, coefficient_count=3
            )
            targets.append(
                low_hermite_coefficients_to_conservative(
                    target_coeff, source_nx=source_nx, target_nx=rollout_nx
                )
            )
        else:
            cached_histories = primitive_target_histories[regime]
            target_fields.append(
                np.stack(
                    [
                        np.asarray(cached_histories[int(index)][1 : available_steps + 1])
                        for index in cases
                    ],
                    axis=0,
                )
            )
            if trajectory_target_norm_by_id is None:
                raise ValueError("Cached primitive targets require trajectory norms")
            target_norms.append(
                np.stack(
                    [
                        np.asarray(
                            trajectory_target_norm_by_id[str(case_ids[index])],
                            dtype=np.float64,
                        )
                        for index in cases
                    ],
                    axis=0,
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
        "amplitude": np.concatenate(amplitudes, axis=0).astype(np.float32),
        "regime_index": np.concatenate(regime_indices, axis=0),
    }
    if primitive_target_histories is None:
        batch["targets"] = np.concatenate(
            [value[:, :trajectory_steps] for value in targets], axis=0
        ).astype(np.float32)
    else:
        batch["target_fields"] = np.concatenate(
            [value[:, :trajectory_steps] for value in target_fields], axis=0
        ).astype(np.float32)
        batch["trajectory_target_norm"] = np.concatenate(target_norms, axis=0)
    if translation_augmentation:
        shifts = rng.integers(0, int(rollout_nx), size=batch["initial"].shape[0])
        for row, shift in enumerate(shifts):
            batch["memory"][row] = np.roll(
                batch["memory"][row], int(shift), axis=-1
            )
            batch["initial"][row] = np.roll(
                batch["initial"][row], int(shift), axis=-1
            )
            target_key = "target_fields" if "target_fields" in batch else "targets"
            batch[target_key][row] = np.roll(
                batch[target_key][row], int(shift), axis=-1
            )
    return batch


def build_diagnostic_panel(
    rng: np.random.Generator,
    grouped,
    anchors,
    manifest,
    coefficient_key: str,
    *,
    regimes: Sequence[str] = REGIMES,
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
    heat_flux_gradient_case_scale_by_id: Optional[Mapping[str, float]] = None,
    include_heat_flux_gradient_history: bool = False,
    autonomous_burnin_steps: int = 0,
    reference_dt: Optional[float] = None,
) -> Sequence[Dict[str, np.ndarray]]:
    amplitudes = _case_amplitudes(manifest)
    selections = {}
    for regime in regimes:
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
    reference_dt_value = float(dt if reference_dt is None else reference_dt)
    reference_ratio = float(dt) / reference_dt_value
    for start_time in start_times:
        time_index = int(round(float(start_time) / reference_dt_value))
        explicit = {}
        for regime in regimes:
            rows = selections[regime]
            histories = tuple(grouped[regime][coefficient_key])
            required_reference_steps = int(
                math.ceil(
                    (int(horizon) + int(autonomous_burnin_steps))
                    * reference_ratio
                )
            )
            if any(
                time_index + required_reference_steps >= int(histories[row].shape[0])
                for row in rows
            ):
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
                regimes=regimes,
                split=split,
                batch_size_per_regime=len(selections[regimes[0]]),
                horizon=horizon,
                memory_steps=memory_steps,
                memory_stride=memory_stride,
                source_nx=source_nx,
                rollout_nx=rollout_nx,
                domain_length=domain_length,
                translation_augmentation=False,
                explicit_selection=explicit,
                heat_flux_gradient_case_scale_by_id=(
                    heat_flux_gradient_case_scale_by_id
                ),
                include_heat_flux_gradient_history=(
                    include_heat_flux_gradient_history
                ),
                autonomous_burnin_steps=autonomous_burnin_steps,
                reference_steps_per_solver_step=reference_ratio,
            )
        )
    return panel


def _adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": jnp.array(0, dtype=jnp.int32), "m": zeros, "v": zeros}


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


def make_loss_function(
    *,
    regimes: Sequence[str] = REGIMES,
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
    global_relative_trajectory_loss: bool = False,
    closure_history_input: bool = False,
    relative_time_block_steps: int = 0,
    relative_time_block_count: int = 0,
    convergence_floor_rms: Optional[np.ndarray] = None,
    memory_backend: str = "latent_recurrent",
    memory_stride: int = 1,
    scan_unroll: int = 1,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
    autonomous_burnin_steps: int = 0,
    log_energy_weight: float = 0.0,
    log_growth_weight: float = 0.0,
):
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    input_scale_jax = jnp.asarray(input_scale, dtype=jnp.float32)
    regime_scales_jax = jnp.asarray(regime_scales, dtype=jnp.float32)
    convergence_floor_jax = (
        None
        if convergence_floor_rms is None
        else jnp.asarray(convergence_floor_rms, dtype=jnp.float32)
    )

    def loss(params, batch):
        if _uses_window_memory(memory_backend):
            initial_state = batch["initial"]
            history = batch["memory"]
            history_counter = jnp.asarray(0, dtype=jnp.int32)
            closure_history = (
                batch["heat_flux_gradient_history"]
                if closure_history_input
                else None
            )
            previous_gradient = (
                closure_history[:, -1]
                if closure_history is not None
                else None
            )
            encoded_history = None
            compact_latent = None
            if int(autonomous_burnin_steps) > 0:
                burnin_states, burnin_memory = _rollout_window_memory(
                    params,
                    initial_state,
                    history,
                    history_counter,
                    batch["amplitude"],
                    k_jax,
                    memory_backend=memory_backend,
                    horizon=int(autonomous_burnin_steps),
                    dt=dt,
                    memory_stride=memory_stride,
                    input_scale=input_scale_jax,
                    heat_flux_gradient_scale=heat_flux_gradient_scale,
                    amplitude_center=amplitude_center,
                    amplitude_scale=amplitude_scale,
                    closure_history_input=closure_history_input,
                    poisson_sign=poisson_sign,
                    normalized_heat_flux_bound=normalized_heat_flux_bound,
                    density_floor=density_floor,
                    pressure_floor=pressure_floor,
                    scan_unroll=scan_unroll,
                    input_scaling=input_scaling,
                    dynamic_amplitude_floor=dynamic_amplitude_floor,
                    allow_uniform_heating=allow_uniform_heating,
                )
                initial_state = jax.lax.stop_gradient(burnin_states[:, -1])
                (
                    history,
                    closure_history,
                    encoded_history,
                    history_counter,
                    previous_gradient,
                    compact_latent,
                ) = _unpack_window_memory(
                    jax.tree_util.tree_map(jax.lax.stop_gradient, burnin_memory),
                    memory_backend,
                )
            rollout_result = _rollout_window_memory(
                params,
                initial_state,
                history,
                history_counter,
                batch["amplitude"],
                k_jax,
                memory_backend=memory_backend,
                compact_latent=compact_latent,
                horizon=horizon,
                dt=dt,
                memory_stride=memory_stride,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                previous_heat_flux_gradient=previous_gradient,
                closure_history_input=closure_history_input,
                encoded_history=encoded_history,
                heat_flux_gradient_history=closure_history,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
                density_floor=density_floor,
                pressure_floor=pressure_floor,
                scan_unroll=scan_unroll,
                input_scaling=input_scaling,
                dynamic_amplitude_floor=dynamic_amplitude_floor,
                allow_uniform_heating=allow_uniform_heating,
            )
            predicted = rollout_result[0]
        else:
            if int(autonomous_burnin_steps) > 0:
                raise ValueError("autonomous history burn-in requires window memory")
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
            if convergence_floor_jax is not None:
                target_indices = batch["start_index"][:, None] + jnp.arange(
                    1, time_count + 1, dtype=jnp.int32
                )[None, :]
                sample_loss = _block_relative_sample_loss(
                    predicted_fields,
                    target_fields,
                    batch["regime_index"],
                    target_indices,
                    block_steps=relative_time_block_steps,
                    block_count=relative_time_block_count,
                    floor_rms=convergence_floor_jax,
                )
            else:
                numerator = jnp.sum(
                    jnp.square(predicted_fields - target_fields), axis=(1, 3)
                )
                denominator = jnp.sum(jnp.square(target_fields), axis=(1, 3))
                sample_loss = jnp.where(
                    jnp.all(denominator > 0.0, axis=1),
                    jnp.mean(numerator / denominator, axis=1),
                    jnp.nan,
                )
        elif global_relative_trajectory_loss:
            numerator = jnp.sum(
                jnp.square(predicted_fields - target_fields), axis=(1, 2, 3)
            )
            denominator = jnp.sum(jnp.square(target_fields), axis=(1, 2, 3))
            sample_loss = jnp.where(
                denominator > 0.0,
                numerator / denominator,
                jnp.nan,
            )
        else:
            scales = regime_scales_jax[batch["regime_index"]][:, None, :, None]
            normalized_error = (predicted_fields - target_fields) / scales
            sample_loss = jnp.mean(jnp.square(normalized_error), axis=(1, 2, 3))
        if float(log_energy_weight) or float(log_growth_weight):
            field_index = 3
            predicted_field = jnp.asarray(
                predicted_fields[:, :, field_index], dtype=jnp.float64
            )
            target_field = jnp.asarray(
                target_fields[:, :, field_index], dtype=jnp.float64
            )
            energy_floor = jnp.asarray(1.0e-60, dtype=jnp.float64)
            predicted_energy = jnp.maximum(
                jnp.mean(jnp.square(predicted_field), axis=-1),
                energy_floor,
            )
            target_energy = jnp.maximum(
                jnp.mean(jnp.square(target_field), axis=-1),
                energy_floor,
            )
            predicted_log = jnp.log(predicted_energy)
            target_log = jnp.log(target_energy)
            energy_loss = jnp.mean(jnp.square(predicted_log - target_log), axis=1)
            growth_loss = jnp.mean(
                jnp.square(
                    jnp.diff(predicted_log, axis=1)
                    - jnp.diff(target_log, axis=1)
                ),
                axis=1,
            )
            sample_loss = (
                sample_loss
                + jnp.asarray(log_energy_weight, dtype=sample_loss.dtype) * energy_loss
                + jnp.asarray(log_growth_weight, dtype=sample_loss.dtype) * growth_loss
            )
        regime_loss = jnp.stack(
            [
                jnp.sum(
                    jnp.where(batch["regime_index"] == index, sample_loss, 0.0)
                )
                / jnp.maximum(jnp.sum(batch["regime_index"] == index), 1)
                for index in (REGIMES.index(regime) for regime in regimes)
            ]
        )
        return jnp.mean(sample_loss), regime_loss

    return loss


def make_continuous_chunk_loss_function(
    *,
    regimes: Sequence[str] = REGIMES,
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
    relative_time_block_steps: int = 0,
    relative_time_block_count: int = 0,
    convergence_floor_rms: Optional[np.ndarray] = None,
    memory_backend: str = "latent_recurrent",
    memory_stride: int = 1,
    scan_unroll: int = 1,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
):
    """Return one truncated-gradient chunk of a continuous autonomous rollout."""
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    input_scale_jax = jnp.asarray(input_scale, dtype=jnp.float32)
    regime_scales_jax = jnp.asarray(regime_scales, dtype=jnp.float32)
    convergence_floor_jax = (
        None
        if convergence_floor_rms is None
        else jnp.asarray(convergence_floor_rms, dtype=jnp.float32)
    )

    def loss(
        params,
        initial_state,
        memory_or_hidden,
        target_fields,
        amplitude,
        regime_index,
        trajectory_target_norm=None,
        target_indices=None,
    ):
        if target_fields.shape[-2] == 3:
            target_shape = target_fields.shape
            target_fields = primitive_fields(
                target_fields.reshape(
                    target_shape[0] * target_shape[1], 3, target_shape[-1]
                ),
                k_jax,
                poisson_sign=poisson_sign,
            ).reshape(target_shape[0], target_shape[1], 4, target_shape[-1])
        if _uses_window_memory(memory_backend):
            if warm_memory:
                history = memory_or_hidden
                closure_history = jnp.zeros(
                    (history.shape[0], history.shape[1], history.shape[-1]),
                    dtype=history.dtype,
                )
                encoded_history = None
                history_counter = jnp.asarray(0, dtype=jnp.int32)
                previous_gradient = jnp.zeros(
                    (initial_state.shape[0], initial_state.shape[-1]),
                    dtype=initial_state.dtype,
                )
                compact_latent = None
            else:
                (
                    history,
                    closure_history,
                    encoded_history,
                    history_counter,
                    previous_gradient,
                    compact_latent,
                ) = _unpack_window_memory(memory_or_hidden, memory_backend)
            rollout_result = _rollout_window_memory(
                params,
                initial_state,
                history,
                history_counter,
                amplitude,
                k_jax,
                memory_backend=memory_backend,
                compact_latent=compact_latent,
                horizon=horizon,
                dt=dt,
                memory_stride=memory_stride,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                previous_heat_flux_gradient=previous_gradient,
                closure_history_input=closure_history_input,
                encoded_history=encoded_history,
                heat_flux_gradient_history=closure_history,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
                density_floor=density_floor,
                pressure_floor=pressure_floor,
                scan_unroll=scan_unroll,
                input_scaling=input_scaling,
                dynamic_amplitude_floor=dynamic_amplitude_floor,
                allow_uniform_heating=allow_uniform_heating,
            )
            predicted, final_memory = rollout_result
        elif warm_memory:
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
        if not _uses_window_memory(memory_backend):
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
            final_memory = (
                (final_hidden, final_gradient)
                if closure_history_input
                else final_hidden
            )
        batch_count, time_count = predicted.shape[:2]
        predicted_fields = primitive_fields(
            predicted.reshape(batch_count * time_count, 3, predicted.shape[-1]),
            k_jax,
            poisson_sign=poisson_sign,
        ).reshape(batch_count, time_count, 4, predicted.shape[-1])
        if relative_trajectory_loss:
            if convergence_floor_jax is not None:
                sample_loss = _block_relative_sample_loss(
                    predicted_fields,
                    target_fields,
                    regime_index,
                    target_indices,
                    block_steps=relative_time_block_steps,
                    block_count=relative_time_block_count,
                    floor_rms=convergence_floor_jax,
                    trajectory_target_norm=trajectory_target_norm,
                )
            else:
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
                for index in (REGIMES.index(regime) for regime in regimes)
            ]
        )
        return jnp.mean(sample_loss), (
            regime_loss,
            predicted[:, -1],
            final_memory,
        )

    return loss


def make_supervised_heat_flux_loss(
    *,
    regimes: Sequence[str] = REGIMES,
    k_arr: np.ndarray,
    width: int,
    input_scale: np.ndarray,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    poisson_sign: float,
    normalized_heat_flux_bound: float,
    memory_backend: str = "latent_recurrent",
    closure_history_input: bool = False,
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
):
    """Fit the causal closure map using a fixed RMS for each complete IC."""
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    input_scale_jax = jnp.asarray(input_scale, dtype=jnp.float32)

    def loss(params, batch):
        if _uses_window_memory(memory_backend):
            closure_history = (
                batch["heat_flux_gradient_history"]
                if closure_history_input
                else None
            )
            previous_gradient = (
                batch["heat_flux_gradient_history"][:, -1]
                if closure_history_input
                else None
            )
            predicted_gradient = explicit_window_closure_step(
                params,
                batch["initial"],
                batch["memory"],
                batch["amplitude"],
                k_jax,
                input_scale=input_scale_jax,
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                previous_heat_flux_gradient=previous_gradient,
                heat_flux_gradient_history=closure_history,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
                input_scaling=input_scaling,
                dynamic_amplitude_floor=dynamic_amplitude_floor,
                allow_uniform_heating=allow_uniform_heating,
            )
        else:
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
        squared_error = jnp.mean(
            jnp.square(predicted_gradient - batch["heat_flux_gradient_target"]),
            axis=-1,
        )
        sample_loss = squared_error / jnp.square(
            batch["heat_flux_gradient_case_scale"]
        )
        regime_loss = jnp.stack(
            [
                jnp.sum(jnp.where(batch["regime_index"] == index, sample_loss, 0.0))
                / jnp.maximum(jnp.sum(batch["regime_index"] == index), 1)
                for index in (REGIMES.index(regime) for regime in regimes)
            ]
        )
        return jnp.mean(regime_loss), regime_loss

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


def _save_training_state(
    path: Path,
    *,
    params,
    optimizer,
    rng: np.random.Generator,
    global_epoch: int,
    loss_ema: Optional[float],
    best_val: float,
    histories: Mapping[str, object],
    in_epoch: Optional[Mapping[str, object]] = None,
) -> None:
    payload = {f"param_{key}": np.asarray(value) for key, value in params.items()}
    payload.update(
        {f"optimizer_m_{key}": np.asarray(value) for key, value in optimizer["m"].items()}
    )
    payload.update(
        {f"optimizer_v_{key}": np.asarray(value) for key, value in optimizer["v"].items()}
    )
    payload["optimizer_step"] = np.asarray(optimizer["step"])
    payload["global_epoch"] = np.asarray(global_epoch, dtype=np.int64)
    payload["loss_ema"] = np.asarray(
        np.nan if loss_ema is None else loss_ema, dtype=np.float64
    )
    payload["best_val"] = np.asarray(best_val, dtype=np.float64)
    payload["rng_state_json"] = np.asarray(
        [json.dumps(rng.bit_generator.state, sort_keys=True)]
    )
    for key, value in histories.items():
        payload[f"history_{key}"] = np.asarray(value)
    if in_epoch is not None:
        for key, value in in_epoch.items():
            if key == "rng_state":
                payload["in_epoch_rng_state_json"] = np.asarray(
                    [json.dumps(value, sort_keys=True)]
                )
            else:
                payload[f"in_epoch_{key}"] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, path)


def _load_training_state(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        params = {
            key.removeprefix("param_"): jnp.asarray(payload[key])
            for key in payload.files
            if key.startswith("param_")
        }
        optimizer = {
            "step": jnp.asarray(payload["optimizer_step"]),
            "m": {
                key.removeprefix("optimizer_m_"): jnp.asarray(payload[key])
                for key in payload.files
                if key.startswith("optimizer_m_")
            },
            "v": {
                key.removeprefix("optimizer_v_"): jnp.asarray(payload[key])
                for key in payload.files
                if key.startswith("optimizer_v_")
            },
        }
        histories = {
            key.removeprefix("history_"): np.asarray(payload[key])
            for key in payload.files
            if key.startswith("history_")
        }
        state = {
            "global_epoch": int(np.asarray(payload["global_epoch"])),
            "loss_ema": float(np.asarray(payload["loss_ema"])),
            "best_val": float(np.asarray(payload["best_val"])),
            "rng_state": json.loads(str(np.asarray(payload["rng_state_json"]).reshape(-1)[0])),
        }
        if "in_epoch_number" in payload.files:
            state["in_epoch"] = {
                "number": int(np.asarray(payload["in_epoch_number"])),
                "completed_steps": int(
                    np.asarray(payload["in_epoch_completed_steps"])
                ),
                "rng_state": json.loads(
                    str(np.asarray(payload["in_epoch_rng_state_json"]).reshape(-1)[0])
                ),
                "losses": np.asarray(payload["in_epoch_losses"]),
                "regime_losses": np.asarray(payload["in_epoch_regime_losses"]),
                "grad_norms": np.asarray(payload["in_epoch_grad_norms"]),
                "update_norms": np.asarray(payload["in_epoch_update_norms"]),
                "update_scales": np.asarray(payload["in_epoch_update_scales"]),
            }
    if set(optimizer["m"]) != set(params) or set(optimizer["v"]) != set(params):
        raise ValueError(f"Incomplete optimizer state: {path}")
    return params, optimizer, histories, state


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
        label=(
            "changing-anchor batch mean"
            if metadata and metadata.get("deterministic_anchor_cycle", False)
            else "stochastic batch mean"
        ),
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
        if metadata.get("supervised_only", False):
            supervised_epochs = int(metadata.get("supervised_heat_flux_epochs", 0))
            stages = [(supervised_epochs, "supervised closure capacity")]
        else:
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


def _interpolate_time_series(
    source_times: np.ndarray, source_values: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    values = np.asarray(source_values)
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, source_times.size - 1)
    left = np.maximum(right - 1, 0)
    denominator = source_times[right] - source_times[left]
    weight = np.divide(
        target_times - source_times[left],
        denominator,
        out=np.zeros_like(target_times),
        where=denominator > 0.0,
    )
    reshape = (target_times.size,) + (1,) * (values.ndim - 1)
    return values[left] + weight.reshape(reshape) * (values[right] - values[left])


def _energy_peak_angular_frequency(times, energy, *, final_time: float = 40.0):
    """Estimate early oscillation frequency from successive energy maxima."""
    times = np.asarray(times, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    indices = np.flatnonzero((times <= float(final_time)) & np.isfinite(energy))
    if indices.size < 3:
        return None
    values = energy[indices]
    maxima = indices[1:-1][
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    ]
    if maxima.size < 3:
        return None
    periods = np.diff(times[maxima])
    periods = periods[periods > 0.0]
    if periods.size < 2:
        return None
    return float(2.0 * math.pi / np.median(periods))


def _evaluate_heldout(
    params,
    grouped,
    manifest,
    coefficient_key: str,
    cache_dir: Path,
    outdir: Path,
    *,
    regimes: Sequence[str] = REGIMES,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    dt: float,
    reference_dt: float,
    width: int,
    memory_steps: int,
    memory_stride: int,
    scan_unroll: int,
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
    memory_backend: str = "latent_recurrent",
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
) -> None:
    amplitudes_by_id = _case_amplitudes(manifest)
    initial_states = []
    case_records = []
    for regime in regimes:
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
    if not initial_states:
        raise ValueError(f"No held-out cases for selected regimes: {regimes}")
    state = jnp.asarray(np.stack(initial_states), dtype=jnp.float32)
    amplitude = jnp.asarray(
        [amplitudes_by_id[record[1]] for record in case_records], dtype=jnp.float32
    )
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(
        int(rollout_nx), d=float(domain_length) / float(rollout_nx)
    )
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    # No trajectory exists before t=0. Use the centered equilibrium state as
    # causal padding instead of pretending that the initial perturbation has
    # already persisted for the full memory span.
    initial_history = jnp.zeros(
        (state.shape[0], int(memory_steps), state.shape[1], state.shape[-1]),
        dtype=state.dtype,
    )
    previous_gradient = jnp.zeros((state.shape[0], state.shape[-1]), dtype=state.dtype)
    if _uses_window_memory(memory_backend):
        initial_closure_history = jnp.zeros(
            (state.shape[0], int(memory_steps), state.shape[-1]), dtype=state.dtype
        )
        initial_encoded_history = encode_explicit_window_history(
            params,
            initial_history,
            k_jax,
            input_scale=jnp.asarray(input_scale, dtype=jnp.float32),
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            heat_flux_gradient_history=(
                initial_closure_history if closure_history_input else None
            ),
            poisson_sign=poisson_sign,
            input_scaling=input_scaling,
            dynamic_amplitude_floor=dynamic_amplitude_floor,
        )
        model_memory = (
            initial_history,
            initial_closure_history,
            initial_encoded_history,
            jnp.asarray(0, dtype=jnp.int32),
            previous_gradient,
        )
        if _uses_compact_latent(memory_backend):
            model_memory = model_memory + (None,)
    else:
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
        model_memory = (hidden, previous_gradient)
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

    def run_chunk(current_state, current_memory, length: int):
        if _uses_window_memory(memory_backend):
            window, closure_window, encoded, counter, gradient, compact_latent = (
                _unpack_window_memory(current_memory, memory_backend)
            )
            return _rollout_window_memory(
                params,
                current_state,
                window,
                counter,
                amplitude,
                k_jax,
                memory_backend=memory_backend,
                compact_latent=compact_latent,
                horizon=length,
                dt=dt,
                memory_stride=memory_stride,
                input_scale=jnp.asarray(input_scale, dtype=jnp.float32),
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                previous_heat_flux_gradient=gradient,
                closure_history_input=closure_history_input,
                encoded_history=encoded,
                heat_flux_gradient_history=closure_window,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
                density_floor=density_floor,
                pressure_floor=pressure_floor,
                scan_unroll=scan_unroll,
                input_scaling=input_scaling,
                dynamic_amplitude_floor=dynamic_amplitude_floor,
                allow_uniform_heating=allow_uniform_heating,
            )
        current_hidden, current_gradient = current_memory
        states, final_hidden, final_gradient = rollout_low_moment_closure(
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
        return states, (final_hidden, final_gradient)

    reference_steps = int(case_records[0][3].shape[0]) - 1
    final_time = reference_steps * float(reference_dt)
    total_steps = int(round(final_time / float(dt)))
    if total_steps <= 0 or not math.isclose(
        total_steps * float(dt), final_time, abs_tol=1e-9
    ):
        raise ValueError("solver dt must divide the cached reference duration")
    completed = 0
    compiled = {}
    while completed < total_steps:
        length = min(int(chunk_steps), total_steps - completed)
        if length not in compiled:
            compiled[length] = jax.jit(
                lambda s, m, n=length: run_chunk(s, m, n)
            )
        states, model_memory = compiled[length](state, model_memory)
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
        np.savez_compressed(
            case_dir / "trajectory.npz",
            times=times,
            model_energy=learned_energy[row],
            teacher_times=hr_times,
            teacher_energy=hr_energy,
            model_E_hat=learned_hat[row, :, :5],
            teacher_E_hat=restricted_hr_hat[:, :5],
        )
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
        matched_reference_hat = _interpolate_time_series(
            hr_times, restricted_hr_hat, times
        )
        with np.errstate(over="ignore", invalid="ignore"):
            relative = np.sqrt(
                np.sum(
                    np.abs(learned_complex - matched_reference_hat) ** 2,
                    axis=-1,
                )
                / np.maximum(
                    np.sum(np.abs(matched_reference_hat) ** 2, axis=-1), 1e-30
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
        model_frequency = _energy_peak_angular_frequency(
            times[:finite_count], learned_energy[row, :finite_count]
        )
        teacher_frequency = _energy_peak_angular_frequency(hr_times, hr_energy)
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
            "energy_peak_angular_frequency_model": model_frequency,
            "energy_peak_angular_frequency_teacher": teacher_frequency,
            "energy_peak_frequency_relative_error": (
                None
                if model_frequency is None or teacher_frequency in (None, 0.0)
                else abs(model_frequency - teacher_frequency)
                / abs(teacher_frequency)
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
    axes = np.atleast_1d(axes)
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
        "regimes": list(regimes),
        "canonical_regimes": list(REGIMES),
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
    parser.add_argument(
        "--training-regimes",
        choices=("all3", "linear_landau"),
        default="all3",
        help=(
            "Regimes exposed to training and diagnostics; normalization remains fixed "
            "to all3 training statistics. Checkpoint evaluation uses saved regimes."
        ),
    )
    parser.add_argument("--rollout-Nx", type=int, default=256)
    parser.add_argument(
        "--solver-dt",
        type=float,
        default=None,
        help="Low-moment solver timestep; defaults to the reference-cache timestep",
    )
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
    parser.add_argument(
        "--scan-unroll",
        type=int,
        default=1,
        help="Compile this many rollout steps per scan iteration without changing dt",
    )
    parser.add_argument(
        "--memory-backend",
        choices=(
            "latent_recurrent",
            "explicit_window",
            "causal_spacetime_operator",
            "window_fno",
            "burles_latent_fno",
        ),
        default="latent_recurrent",
        help="Select recurrent, fixed-window, causal space-time, or window-FNO memory",
    )
    parser.add_argument("--spacetime-depth", type=int, default=4)
    parser.add_argument("--temporal-kernel-size", type=int, default=5)
    parser.add_argument("--fno-depth", type=int, default=4)
    parser.add_argument(
        "--latent-memory-dim",
        type=int,
        default=6,
        help="Compact learned memory channels; zero gives the matched Burles history-only control",
    )
    parser.add_argument("--history-stride", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-regime batch size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--planned-epochs",
        type=int,
        default=None,
        help="Length of the fixed learning-rate schedule, including later resumes",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        default=None,
        help="Resume model, optimizer, RNG, counters, and metrics from a run directory",
    )
    parser.add_argument("--supervised-warmup-epochs", type=int, default=0)
    parser.add_argument(
        "--supervised-only",
        action="store_true",
        help="Fit the exact heat-flux-gradient map without solver rollout",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=30)
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=0,
        help=(
            "Persist exact within-epoch state this often; currently supported "
            "for full-anchor sweeps without translation augmentation"
        ),
    )
    parser.add_argument(
        "--training-passes-per-epoch",
        type=int,
        default=1,
        help="Balanced random-window passes over every training IC per epoch",
    )
    parser.add_argument(
        "--deterministic-anchor-cycle",
        action="store_true",
        help="Use one deterministic no-replacement temporal anchor per training IC and epoch",
    )
    parser.add_argument(
        "--full-anchor-sweep",
        action="store_true",
        help="Shuffle and consume every valid training anchor exactly once per epoch",
    )
    parser.add_argument(
        "--translation-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly translate each spatial training window",
    )
    parser.add_argument(
        "--train-case-limits",
        type=str,
        default="",
        help=(
            "Diagnostic-only comma-separated regime=count limits. Selection is "
            "deterministic and nested for a fixed seed; held-out cases and training "
            "statistics remain unchanged."
        ),
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument(
        "--data-parallel-devices",
        type=int,
        default=1,
        help="Number of equal balanced JAX devices used inside each minibatch",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--update-norm-cap",
        type=float,
        default=0.0,
        help="Optional global L2 cap on each accepted AdamW parameter update",
    )
    parser.add_argument(
        "--require-first-update-descent",
        action="store_true",
        help="Abort unless the first accepted update lowers its exact accumulated batch",
    )
    parser.add_argument(
        "--report-every-update-descent",
        action="store_true",
        help="Re-evaluate each exact accumulated batch after its update and log before/after loss",
    )
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--spectral-modes", type=int, default=16)
    parser.add_argument("--closure-history-input", action="store_true")
    parser.add_argument(
        "--autonomous-history-burnin-steps",
        type=int,
        default=0,
        help="Generate closure history autonomously before each scored random window",
    )
    parser.add_argument(
        "--input-scaling",
        choices=tuple(sorted(INPUT_SCALING_KINDS)),
        default=FIXED_INPUT_SCALING,
        help="Closure input/output scaling geometry",
    )
    parser.add_argument(
        "--dynamic-amplitude-floor",
        type=float,
        default=1e-6,
        help="Numerical floor used only by current-density-RMS scaling",
    )
    parser.add_argument(
        "--allow-uniform-heating",
        action="store_true",
        help=(
            "Allow the effective closure to include a spatially uniform source; "
            "the default remains a zero-mean periodic heat-flux gradient"
        ),
    )
    parser.add_argument("--relative-trajectory-loss", action="store_true")
    parser.add_argument(
        "--global-relative-trajectory-loss",
        action="store_true",
        help="Use one relative L2 ratio over all fields, times, and points",
    )
    parser.add_argument("--log-energy-weight", type=float, default=0.0)
    parser.add_argument("--log-growth-weight", type=float, default=0.0)
    parser.add_argument(
        "--relative-time-block",
        type=float,
        default=0.0,
        help="Fixed physical-time block duration for relative trajectory loss",
    )
    parser.add_argument(
        "--convergence-floor-file",
        type=Path,
        default=None,
        help="NPZ containing regime/channel RMS teacher-grid disagreement",
    )
    parser.add_argument("--stats-stride", type=int, default=20)
    parser.add_argument(
        "--normalization-checkpoint",
        type=Path,
        default=None,
        help="Load only fixed all3 statistics from a compatible checkpoint, never weights",
    )
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
    training_regimes = parse_training_regimes(args.training_regimes)
    planned_epochs = int(args.epochs if args.planned_epochs is None else args.planned_epochs)
    if planned_epochs < int(args.epochs) or planned_epochs <= 0:
        raise ValueError("planned-epochs must be positive and at least epochs")
    train_case_limits = parse_train_case_limits(args.train_case_limits)
    if set(train_case_limits) - set(training_regimes):
        raise ValueError("training-case limits must refer to selected training regimes")
    if args.rollout_horizon <= 0 or args.memory_steps <= 0 or args.scan_unroll <= 0:
        raise ValueError("rollout horizon, memory steps, and scan unroll must be positive")
    if args.solver_dt is not None and args.solver_dt <= 0.0:
        raise ValueError("solver-dt must be positive")
    if args.autonomous_history_burnin_steps < 0:
        raise ValueError("autonomous-history-burnin-steps cannot be negative")
    if args.autonomous_history_burnin_steps and (
        args.training_schedule != "random_windows"
        or not _uses_window_memory(args.memory_backend)
    ):
        raise ValueError("autonomous history burn-in requires random-window window-memory training")
    if min(args.learning_rate, args.final_learning_rate) <= 0.0:
        raise ValueError("learning rates must be positive")
    if args.final_learning_rate > args.learning_rate:
        raise ValueError("final-learning-rate cannot exceed learning-rate")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay cannot be negative")
    if args.grad_clip < 0.0:
        raise ValueError("grad-clip cannot be negative; zero disables clipping")
    if args.relative_trajectory_loss and args.global_relative_trajectory_loss:
        raise ValueError("choose only one relative trajectory loss geometry")
    if args.full_anchor_sweep and args.deterministic_anchor_cycle:
        raise ValueError("full-anchor-sweep and deterministic-anchor-cycle are exclusive")
    if args.full_anchor_sweep and args.training_schedule != "random_windows":
        raise ValueError("full-anchor-sweep requires random-window training")
    if args.full_anchor_sweep and int(args.training_passes_per_epoch) != 1:
        raise ValueError("full-anchor-sweep requires one training pass per epoch")
    if args.checkpoint_every_updates < 0:
        raise ValueError("checkpoint-every-updates cannot be negative")
    if args.checkpoint_every_updates and (
        not args.full_anchor_sweep
        or args.training_schedule != "random_windows"
        or args.translation_augmentation
    ):
        raise ValueError(
            "within-epoch checkpoints require a full-anchor random-window sweep "
            "without translation augmentation"
        )
    if min(args.log_energy_weight, args.log_growth_weight) < 0.0:
        raise ValueError("energy and growth weights cannot be negative")
    if min(args.gradient_accumulation_steps, args.training_passes_per_epoch) <= 0:
        raise ValueError("gradient accumulation and training passes must be positive")
    if args.data_parallel_devices <= 0:
        raise ValueError("data-parallel-devices must be positive")
    if args.data_parallel_devices > len(jax.devices()):
        raise ValueError(
            f"Requested {args.data_parallel_devices} data-parallel devices, "
            f"but JAX exposes {len(jax.devices())}"
        )
    if args.update_norm_cap < 0.0:
        raise ValueError("update-norm-cap cannot be negative")
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
    if (
        _uses_window_memory(args.memory_backend)
        and args.supervised_warmup_epochs
        and not args.supervised_only
    ):
        raise ValueError("window-memory training does not use supervised warm-up")
    if args.memory_backend == "causal_spacetime_operator":
        if not args.closure_history_input:
            raise ValueError(
                "causal-spacetime-operator requires closure-history-input"
            )
        if min(args.spacetime_depth, args.temporal_kernel_size) <= 0:
            raise ValueError("space-time depth and temporal kernel size must be positive")
    if args.memory_backend in ("window_fno", "burles_latent_fno"):
        if args.fno_depth <= 0:
            raise ValueError("window-fno depth must be positive")
    if args.memory_backend == "burles_latent_fno":
        if not args.closure_history_input:
            raise ValueError("burles-latent-fno requires closure-history-input")
        if args.latent_memory_dim < 0:
            raise ValueError("compact latent dimension cannot be negative")
    if (
        args.input_scaling == DYNAMIC_INPUT_SCALING or args.allow_uniform_heating
    ) and not _uses_window_memory(args.memory_backend):
        raise ValueError(
            "dynamic scaling and uniform heating currently require a window-memory backend"
        )
    if args.dynamic_amplitude_floor <= 0.0:
        raise ValueError("dynamic-amplitude-floor must be positive")
    if args.supervised_only:
        if args.training_schedule != "random_windows":
            raise ValueError("supervised-only training requires random_windows")
        if args.relative_trajectory_loss:
            raise ValueError("supervised-only training does not use trajectory loss")
        if args.horizon_curriculum.strip():
            raise ValueError("supervised-only training does not use a horizon curriculum")
        if int(args.supervised_warmup_epochs) != 0:
            raise ValueError("supervised-only training does not use warm-up epochs")
    if (args.relative_time_block > 0.0) != (args.convergence_floor_file is not None):
        raise ValueError(
            "relative-time-block and convergence-floor-file must be enabled together"
        )
    if args.relative_time_block > 0.0 and not args.relative_trajectory_loss:
        raise ValueError("fixed time blocks require relative-trajectory-loss")
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
        if args.supervised_only:
            curriculum = ()
        else:
            if not 0 <= int(args.supervised_warmup_epochs) < int(args.epochs):
                raise ValueError("supervised warm-up epochs must be in [0, epochs)")
            curriculum = parse_horizon_curriculum(
                args.horizon_curriculum,
                final_horizon=args.rollout_horizon,
                total_epochs=planned_epochs - args.supervised_warmup_epochs,
            )
    grouped, manifest, cache_metadata, coefficient_key = _load_reference_cache(
        args.reference_cache
    )
    configuration = dict(cache_metadata["configuration"])
    source_nx = int(configuration["teacher_Nx"])
    domain_length = float(configuration["teacher_L"])
    reference_dt = float(configuration["teacher_dt"])
    dt = reference_dt if args.solver_dt is None else float(args.solver_dt)
    reference_steps_per_solver_step = dt / reference_dt
    poisson_sign = float(configuration["teacher_poisson_sign"])
    relative_time_block_steps = 0
    relative_time_block_count = 0
    convergence_floor_rms = None
    convergence_floor_metadata = None
    if args.convergence_floor_file is not None:
        relative_time_block_steps = int(round(args.relative_time_block / dt))
        if relative_time_block_steps <= 0 or not math.isclose(
            relative_time_block_steps * dt, args.relative_time_block, abs_tol=1e-10
        ):
            raise ValueError("relative-time-block must be an integer multiple of dt")
        total_steps = int(round(float(configuration["T_final"]) / dt))
        relative_time_block_count = int(
            math.ceil(total_steps / relative_time_block_steps)
        )
        convergence_floor_rms, convergence_floor_metadata = _load_convergence_floor(
            args.convergence_floor_file
        )
        expected_floor_metadata = {
            "regimes": list(REGIMES),
            "channels": [
                "density_perturbation",
                "velocity",
                "pressure_perturbation",
                "electric_field",
            ],
            "rollout_Nx": int(args.rollout_Nx),
        }
        floor_mismatches = {
            key: (convergence_floor_metadata.get(key), expected)
            for key, expected in expected_floor_metadata.items()
            if convergence_floor_metadata.get(key) != expected
        }
        if floor_mismatches:
            raise ValueError(
                f"Convergence floor metadata does not match training: {floor_mismatches}"
            )
        print(
            f"[data] block-relative trajectory loss: block={args.relative_time_block:g} "
            f"blocks={relative_time_block_count} "
            f"convergence_floor={args.convergence_floor_file}"
        )
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
            regimes=_checkpoint_regimes(checkpoint_metadata),
            source_nx=source_nx,
            rollout_nx=rollout_nx,
            domain_length=domain_length,
            dt=float(checkpoint_metadata.get("dt", reference_dt)),
            reference_dt=reference_dt,
            width=int(checkpoint_metadata["width"]),
            memory_steps=int(checkpoint_metadata["memory_steps"]),
            memory_stride=int(checkpoint_metadata["memory_stride"]),
            scan_unroll=int(checkpoint_metadata.get("scan_unroll", 1)),
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
            memory_backend=str(
                checkpoint_metadata.get("memory_backend", "latent_recurrent")
            ),
            input_scaling=str(
                checkpoint_metadata.get("input_scaling", FIXED_INPUT_SCALING)
            ),
            dynamic_amplitude_floor=float(
                checkpoint_metadata.get("dynamic_amplitude_floor", 1e-6)
            ),
            allow_uniform_heating=bool(
                checkpoint_metadata.get("allow_uniform_heating", False)
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
    if args.resume_run is not None and args.resume_run.resolve() != outdir:
        raise ValueError("resume-run must name the same directory as outdir")
    if args.resume_run is None and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    anchors = _build_anchor_index(
        grouped,
        coefficient_key,
        regimes=training_regimes,
        horizon=int(
            math.ceil(
                (
                    1
                    if args.supervised_only
                    else args.rollout_horizon + args.autonomous_history_burnin_steps
                )
                * reference_steps_per_solver_step
            )
        ),
        history_stride=args.history_stride,
    )
    selected_training_cases: Dict[str, np.ndarray] = {
        regime: np.unique(anchors[regime]["train_cases"]) for regime in training_regimes
    }
    if train_case_limits:
        anchors, selected_training_cases = limit_training_anchors(
            anchors,
            train_case_limits,
            regimes=training_regimes,
            seed=args.seed,
        )
    for regime in training_regimes:
        selected_case_ids = np.asarray(grouped[regime]["case_ids"], dtype=np.str_)[
            selected_training_cases[regime]
        ]
        print(
            f"[data] {regime}: train_anchors={anchors[regime]['train_times'].size} "
            f"heldout_anchors={anchors[regime]['val_times'].size} "
            f"selected_train_cases={','.join(selected_case_ids.tolist())}"
        )
    if args.full_anchor_sweep and training_regimes != REGIMES:
        args.steps_per_epoch = _full_anchor_steps_per_epoch(
            anchors, training_regimes, args.batch_size, args.gradient_accumulation_steps
        )
        print(f"[data] selected full sweep: {args.steps_per_epoch} optimizer updates/epoch")
    # Preserve the original all3 normalization, independent of training exposure.
    # Matched history-only/latent runs share these exact cached statistics.
    normalization_checkpoint_sha256 = None
    if args.normalization_checkpoint is not None:
        stats, normalization_checkpoint_sha256 = _load_normalization_statistics(
            args.normalization_checkpoint,
            manifest=manifest,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
        )
        print(f"[data] fixed normalization from {args.normalization_checkpoint}")
    else:
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
    heat_flux_gradient_case_scale_by_id = dict(
        zip(
            np.asarray(stats["heat_flux_gradient_case_ids"], dtype=np.str_),
            np.asarray(stats["heat_flux_gradient_case_scales"], dtype=np.float64),
        )
    )
    primitive_target_histories = None
    trajectory_target_norm_by_id = None
    primitive_target_cache_root = None
    if args.training_schedule == "continuous_trajectories":
        (
            primitive_target_histories,
            trajectory_target_norm_by_id,
            primitive_target_cache_root,
        ) = _load_or_build_primitive_target_cache(
            args.reference_cache,
            grouped,
            manifest,
            coefficient_key,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
            poisson_sign=poisson_sign,
        )
        print(f"[data] primitive target cache: {primitive_target_cache_root}")
    if args.supervised_only:
        horizon_description = "supervised_q_only"
    elif len(curriculum) == 1:
        horizon_description = f"fixed_H={args.rollout_horizon}"
    else:
        horizon_description = "curriculum=" + ",".join(
            f"{h}:{e}" for h, e in curriculum
        )
    spacetime_description = (
        f" depth={args.spacetime_depth} temporal_kernel={args.temporal_kernel_size}"
        if args.memory_backend == "causal_spacetime_operator"
        else ""
    )
    if args.memory_backend in ("window_fno", "burles_latent_fno"):
        spacetime_description = f" depth={args.fno_depth}"
    if args.memory_backend == "burles_latent_fno":
        spacetime_description += f" latent={args.latent_memory_dim}"
    print(
        f"[model] backend={args.memory_backend} width={args.width} "
        f"modes={args.spectral_modes}{spacetime_description} "
        f"memory_span={args.memory_steps * args.memory_stride * dt:.3f} "
        f"rollout_span={args.rollout_horizon * dt:.3f} "
        f"scan_unroll={args.scan_unroll} "
        f"warmup={args.supervised_warmup_epochs} {horizon_description}"
    )
    nx = int(args.rollout_Nx)
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(nx, d=domain_length / nx)
    if args.memory_backend == "causal_spacetime_operator":
        params = init_causal_spacetime_operator_params(
            jax.random.PRNGKey(args.seed),
            width=args.width,
            spectral_modes=args.spectral_modes,
            memory_steps=args.memory_steps,
            depth=args.spacetime_depth,
            temporal_kernel_size=args.temporal_kernel_size,
            input_channels=5,
            dtype=jnp.float32,
        )
    elif args.memory_backend == "burles_latent_fno":
        params = init_burles_latent_fno_params(
            jax.random.PRNGKey(args.seed),
            width=args.width,
            spectral_modes=args.spectral_modes,
            memory_steps=args.memory_steps,
            depth=args.fno_depth,
            latent_dim=args.latent_memory_dim,
            input_channels=5,
            dtype=jnp.float32,
        )
    elif args.memory_backend == "window_fno":
        params = init_window_fno_params(
            jax.random.PRNGKey(args.seed),
            width=args.width,
            spectral_modes=args.spectral_modes,
            memory_steps=args.memory_steps,
            depth=args.fno_depth,
            input_channels=5 if args.closure_history_input else 4,
            dtype=jnp.float32,
        )
    elif args.memory_backend == "explicit_window":
        params = init_explicit_window_params(
            jax.random.PRNGKey(args.seed),
            width=args.width,
            spectral_modes=args.spectral_modes,
            memory_steps=args.memory_steps,
            input_channels=5 if args.closure_history_input else 4,
            dtype=jnp.float32,
        )
    else:
        params = init_spectral_memory_params(
            jax.random.PRNGKey(args.seed),
            width=args.width,
            spectral_modes=args.spectral_modes,
            input_channels=5 if args.closure_history_input else 4,
            dtype=jnp.float32,
        )
    parameter_count = sum(
        int(value.size) for value in jax.tree_util.tree_leaves(params)
    )
    print(f"[model] trainable_parameters={parameter_count}")
    if args.init_checkpoint is not None:
        initialized_params, initialized_metadata, _ = _load_checkpoint(
            args.init_checkpoint
        )
        expected = {
            "schema_version": CHECKPOINT_SCHEMA,
            "state_representation": "centered_conservative_v1",
            "closure_output": (
                "effective_gradient_with_uniform_heating"
                if args.allow_uniform_heating
                else "zero_mean_heat_flux_gradient"
            ),
            "manifest_sha256": str(manifest["sha256"]),
            "source_Nx": source_nx,
            "rollout_Nx": int(args.rollout_Nx),
            "width": int(args.width),
            "spectral_modes": int(args.spectral_modes),
            "memory_backend": str(args.memory_backend),
        }
        if args.memory_backend == "burles_latent_fno":
            expected["latent_memory_dim"] = int(args.latent_memory_dim)
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
    train_update_norm_history = []
    train_update_scale_min_history = []
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
        "objective": (
            "supervised_heat_flux_gradient_capacity"
            if args.supervised_only
            else OBJECTIVE
        ),
        "model_backend": MODEL_BACKEND,
        "reference_cache": str(Path(args.reference_cache).resolve()),
        "manifest_sha256": str(manifest["sha256"]),
        "source_Nx": source_nx,
        "rollout_Nx": args.rollout_Nx,
        "dt": dt,
        "reference_dt": reference_dt,
        "reference_steps_per_solver_step": reference_steps_per_solver_step,
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
        "supervised_heat_flux_epochs": (
            args.epochs if args.supervised_only else args.supervised_warmup_epochs
        ),
        "memory_steps": args.memory_steps,
        "memory_stride": args.memory_stride,
        "scan_unroll": args.scan_unroll,
        "memory_backend": args.memory_backend,
        "primitive_target_cache": (
            str(primitive_target_cache_root)
            if primitive_target_cache_root is not None
            else None
        ),
        "fno_depth": (
            int(args.fno_depth)
            if args.memory_backend in ("window_fno", "burles_latent_fno")
            else None
        ),
        "latent_memory_dim": (
            int(args.latent_memory_dim)
            if args.memory_backend == "burles_latent_fno"
            else None
        ),
        "closure_integration": (
            "ssprk3_carried_then_provisional"
            if args.memory_backend == "burles_latent_fno"
            else "legacy_backend"
        ),
        "spacetime_depth": (
            args.spacetime_depth
            if args.memory_backend == "causal_spacetime_operator"
            else None
        ),
        "temporal_kernel_size": (
            args.temporal_kernel_size
            if args.memory_backend == "causal_spacetime_operator"
            else None
        ),
        "input_scaling": args.input_scaling,
        "dynamic_amplitude_floor": args.dynamic_amplitude_floor,
        "supervised_only": args.supervised_only,
        "width": args.width,
        "spectral_modes": args.spectral_modes,
        "batch_size_per_regime": args.batch_size,
        "train_case_limits": dict(train_case_limits),
        "selected_training_case_ids": {
            regime: np.asarray(grouped[regime]["case_ids"], dtype=np.str_)[
                selected_training_cases[regime]
            ].tolist()
            for regime in training_regimes
        },
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "data_parallel_devices": args.data_parallel_devices,
        "training_passes_per_epoch": args.training_passes_per_epoch,
        "steps_per_epoch": args.steps_per_epoch,
        "steps_per_epoch_source": (
            "selected_full_anchor_count"
            if args.full_anchor_sweep and training_regimes != REGIMES
            else "configured"
        ),
        "deterministic_anchor_cycle": args.deterministic_anchor_cycle,
        "full_anchor_sweep": args.full_anchor_sweep,
        "translation_augmentation": args.translation_augmentation,
        "planned_epochs": planned_epochs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "final_learning_rate": args.final_learning_rate,
        "weight_decay": args.weight_decay,
        "update_norm_cap": args.update_norm_cap,
        "require_first_update_descent": args.require_first_update_descent,
        "report_every_update_descent": args.report_every_update_descent,
        "grad_clip": args.grad_clip,
        "optimizer": "adamw_cosine",
        "autonomous_history_burnin_steps": args.autonomous_history_burnin_steps,
        "autonomous_history_burnin_detached": bool(args.autonomous_history_burnin_steps),
        "compact_latent_initializer_gradient_disconnected": bool(
            args.memory_backend == "burles_latent_fno"
            and args.latent_memory_dim > 0
            and args.autonomous_history_burnin_steps > 0
        ),
        "training_history_source": (
            "model_generated_from_low_moment_reset"
            if args.autonomous_history_burnin_steps
            else "teacher_low_moment_history_at_random_reset"
        ),
        "teacher_heat_flux_history_input": bool(
            args.closure_history_input
            and not args.autonomous_history_burnin_steps
        ),
        "log_energy_weight": args.log_energy_weight,
        "log_growth_weight": args.log_growth_weight,
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
        "regimes": list(training_regimes),
        "training_regimes": args.training_regimes,
        "canonical_regimes": list(REGIMES),
        "regime_indices": [REGIMES.index(regime) for regime in training_regimes],
        "normalization_checkpoint": (
            str(args.normalization_checkpoint.resolve())
            if args.normalization_checkpoint is not None else None
        ),
        "normalization_checkpoint_sha256": normalization_checkpoint_sha256,
        "normalization_policy": "fixed_all3_training_statistics",
        "normalization_regimes": list(REGIMES),
        "stats_stride": args.stats_stride,
        "normalization_statistics_sha256": sha256_json(
            {key: np.asarray(value).tolist() for key, value in stats.items()}
        ),
        "normalized_heat_flux_bound": args.normalized_heat_flux_bound,
        "density_floor": args.density_floor,
        "pressure_floor": args.pressure_floor,
        "state_representation": "centered_conservative_v1",
        "closure_output": (
            "effective_gradient_with_uniform_heating"
            if args.allow_uniform_heating
            else "zero_mean_heat_flux_gradient"
        ),
        "allow_uniform_heating": args.allow_uniform_heating,
        "closure_history_input": args.closure_history_input,
        "relative_trajectory_loss": args.relative_trajectory_loss,
        "global_relative_trajectory_loss": args.global_relative_trajectory_loss,
        "relative_time_block": args.relative_time_block,
        "convergence_floor_file": (
            None
            if args.convergence_floor_file is None
            else str(args.convergence_floor_file.resolve())
        ),
        "convergence_floor_metadata": convergence_floor_metadata,
    }
    if metadata["compact_latent_initializer_gradient_disconnected"]:
        print(
            "[train] known limitation: detached autonomous burn-in disconnects "
            "compact_latent_init_* from the scored rollout gradient; preserved "
            "for the controlled comparison"
        )
    immutable_resume_keys = (
        "manifest_sha256",
        "rollout_Nx",
        "dt",
        "reference_dt",
        "rollout_horizon",
        "horizon_curriculum",
        "memory_steps",
        "memory_stride",
        "memory_backend",
        "fno_depth",
        "latent_memory_dim",
        "input_scaling",
        "width",
        "spectral_modes",
        "batch_size_per_regime",
        "train_case_limits",
        "selected_training_case_ids",
        "gradient_accumulation_steps",
        "data_parallel_devices",
        "training_passes_per_epoch",
        "steps_per_epoch",
        "deterministic_anchor_cycle",
        "full_anchor_sweep",
        "translation_augmentation",
        "planned_epochs",
        "learning_rate",
        "final_learning_rate",
        "weight_decay",
        "grad_clip",
        "report_every_update_descent",
        "update_norm_cap",
        "autonomous_history_burnin_steps",
        "log_energy_weight",
        "log_growth_weight",
        "global_relative_trajectory_loss",
    )
    configuration_path = outdir / "run_configuration.json"
    resumed_histories = None
    resumed_state = None
    if args.resume_run is not None:
        if not configuration_path.is_file():
            raise FileNotFoundError(f"Missing resume configuration: {configuration_path}")
        saved_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        _validate_regime_resume_configuration(saved_configuration, metadata)
        mismatches = {
            key: (saved_configuration.get(key), metadata.get(key))
            for key in immutable_resume_keys
            if saved_configuration.get(key) != metadata.get(key)
        }
        if mismatches:
            raise ValueError(f"Resume configuration mismatch: {mismatches}")
        params, optimizer, resumed_histories, resumed_state = _load_training_state(
            outdir / "training_state.npz"
        )
        rng.bit_generator.state = resumed_state["rng_state"]
        print(
            f"[train] resumed exact state at epoch={resumed_state['global_epoch']} "
            f"optimizer_step={int(np.asarray(optimizer['step']))}"
        )
        if "in_epoch" in resumed_state:
            print(
                "[train] found resumable partial epoch "
                f"{resumed_state['in_epoch']['number']} after "
                f"{resumed_state['in_epoch']['completed_steps']} updates"
            )
    else:
        configuration_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    started = time.perf_counter()
    global_epoch = 0 if resumed_state is None else int(resumed_state["global_epoch"])
    if global_epoch >= int(args.epochs):
        raise ValueError(
            f"Resume epoch {global_epoch} has already reached requested epoch {args.epochs}"
        )
    if resumed_histories is not None:
        train_history = resumed_histories["train_loss"].tolist()
        train_ema_history = resumed_histories["train_ema_loss"].tolist()
        train_update_norm_history = resumed_histories.get(
            "train_update_norm", np.asarray([], dtype=np.float64)
        ).tolist()
        train_update_scale_min_history = resumed_histories.get(
            "train_update_scale_min", np.asarray([], dtype=np.float64)
        ).tolist()
        train_eval_epochs = resumed_histories["train_eval_epochs"].astype(int).tolist()
        train_eval_history = resumed_histories["train_eval_loss"].tolist()
        train_eval_regime_history = resumed_histories[
            "train_eval_regime_loss"
        ].tolist()
        val_epochs = resumed_histories["val_epochs"].astype(int).tolist()
        val_history = resumed_histories["val_loss"].tolist()
        val_regime_history = resumed_histories["val_regime_loss"].tolist()
        autonomous_val_epochs = resumed_histories[
            "autonomous_val_epochs"
        ].astype(int).tolist()
        autonomous_val_history = resumed_histories["autonomous_val_loss"].tolist()
        autonomous_val_regime_history = resumed_histories[
            "autonomous_val_regime_loss"
        ].tolist()
        best_val = float(resumed_state["best_val"])
    supervised_epochs = (
        int(args.epochs) if args.supervised_only else int(args.supervised_warmup_epochs)
    )
    if supervised_epochs > 0:
        warmup_loss = make_supervised_heat_flux_loss(
            regimes=training_regimes,
            k_arr=k_arr,
            width=args.width,
            input_scale=stats["input_scale"],
            heat_flux_gradient_scale=float(stats["heat_flux_gradient_scale"][0]),
            amplitude_center=float(stats["amplitude_center"][0]),
            amplitude_scale=float(stats["amplitude_scale"][0]),
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=args.normalized_heat_flux_bound,
            memory_backend=args.memory_backend,
            closure_history_input=args.closure_history_input,
            input_scaling=args.input_scaling,
            dynamic_amplitude_floor=args.dynamic_amplitude_floor,
            allow_uniform_heating=args.allow_uniform_heating,
        )
        warmup_value_and_grad = jax.jit(
            jax.value_and_grad(warmup_loss, has_aux=True)
        )
        warmup_validation_panel = build_diagnostic_panel(
            validation_rng,
            grouped,
            anchors,
            manifest,
            coefficient_key,
            regimes=training_regimes,
            split="val",
            start_times=diagnostic_start_times,
            cases_per_regime=None,
            horizon=1,
            dt=dt,
            memory_steps=args.memory_steps,
            memory_stride=args.memory_stride,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
            heat_flux_gradient_case_scale_by_id=(
                heat_flux_gradient_case_scale_by_id
            ),
            include_heat_flux_gradient_history=args.closure_history_input,
            reference_dt=reference_dt,
        )
        warmup_validation_panel_jax = tuple(
            {key: jnp.asarray(value) for key, value in batch.items()}
            for batch in warmup_validation_panel
        )
        warmup_evaluate = jax.jit(warmup_loss)
        phase_name = "capacity probe" if args.supervised_only else "warm-up"
        print(
            f"[train] supervised central-heat-flux {phase_name} "
            f"epochs={supervised_epochs} normalization=fixed_per_IC_RMS"
        )
        for _ in range(supervised_epochs):
            global_epoch += 1
            epoch_losses = []
            epoch_regime_losses = []
            for _ in range(int(args.steps_per_epoch)):
                batch = sample_batch(
                    rng,
                    grouped,
                    anchors,
                    manifest,
                    coefficient_key,
                    regimes=training_regimes,
                    split="train",
                    batch_size_per_regime=args.batch_size,
                    horizon=1,
                    memory_steps=args.memory_steps,
                    memory_stride=args.memory_stride,
                    source_nx=source_nx,
                    rollout_nx=args.rollout_Nx,
                    domain_length=domain_length,
                    translation_augmentation=True,
                    heat_flux_gradient_case_scale_by_id=(
                        heat_flux_gradient_case_scale_by_id
                    ),
                    include_heat_flux_gradient_history=args.closure_history_input,
                    reference_steps_per_solver_step=(
                        reference_steps_per_solver_step
                    ),
                )
                batch_jax = {key: jnp.asarray(value) for key, value in batch.items()}
                (loss_value, regime_value), grads = warmup_value_and_grad(
                    params, batch_jax
                )
                params, optimizer, grad_norm, _, _ = _adam_step(
                    params,
                    grads,
                    optimizer,
                    args.learning_rate,
                    args.grad_clip,
                    update_norm_cap=args.update_norm_cap,
                )
                epoch_losses.append(float(loss_value))
                epoch_regime_losses.append(
                    np.asarray(regime_value, dtype=np.float64)
                )
            mean_loss = float(np.mean(epoch_losses))
            mean_regime_loss = np.mean(epoch_regime_losses, axis=0)
            if not math.isfinite(mean_loss):
                raise FloatingPointError(
                    f"Non-finite heat-flux warm-up loss at epoch {global_epoch}"
                )
            train_history.append(mean_loss)
            val_text = ""
            if (
                global_epoch == 1
                or global_epoch % int(args.validation_every) == 0
                or global_epoch == supervised_epochs
            ):
                panel_values = []
                panel_regime_values = []
                for panel_batch in warmup_validation_panel_jax:
                    val_result = warmup_evaluate(params, panel_batch)
                    panel_values.append(float(val_result[0]))
                    panel_regime_values.append(
                        np.asarray(val_result[1], dtype=np.float64)
                    )
                val_value = float(np.mean(panel_values))
                val_regime = np.mean(panel_regime_values, axis=0)
                val_regime_history.append(val_regime)
                if val_value < best_val:
                    best_val = val_value
                    _save_checkpoint(
                        outdir / "best_low_moment_closure.npz",
                        params,
                        metadata,
                        stats,
                    )
                val_history.append(val_value)
                val_epochs.append(global_epoch)
                val_text = (
                    f" heldout={val_value:.6e} heldout_regime=("
                    + ",".join(f"{value:.3e}" for value in val_regime)
                    + ")"
                )
            elapsed = time.perf_counter() - started
            print(
                f"[train] epoch {global_epoch:04d}/{args.epochs:04d} "
                f"supervised_q={mean_loss:.6e}{val_text} regime=("
                + ",".join(f"{value:.3e}" for value in mean_regime_loss)
                + ") "
                f"grad={float(grad_norm):.3e} elapsed={elapsed / 60.0:.1f}m"
            )
    curriculum_epoch_offset = int(args.supervised_warmup_epochs)
    for stage_horizon, stage_epochs in curriculum:
        stage_end_epoch = curriculum_epoch_offset + int(stage_epochs)
        if global_epoch >= stage_end_epoch:
            curriculum_epoch_offset = stage_end_epoch
            continue
        loss_fn = make_loss_function(
            regimes=training_regimes,
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
            global_relative_trajectory_loss=args.global_relative_trajectory_loss,
            closure_history_input=args.closure_history_input,
            relative_time_block_steps=relative_time_block_steps,
            relative_time_block_count=relative_time_block_count,
            convergence_floor_rms=convergence_floor_rms,
            memory_backend=args.memory_backend,
            memory_stride=args.memory_stride,
            scan_unroll=args.scan_unroll,
            input_scaling=args.input_scaling,
            dynamic_amplitude_floor=args.dynamic_amplitude_floor,
            allow_uniform_heating=args.allow_uniform_heating,
            autonomous_burnin_steps=args.autonomous_history_burnin_steps,
            log_energy_weight=args.log_energy_weight,
            log_growth_weight=args.log_growth_weight,
        )
        value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        parallel_value_and_grad = None
        if int(args.data_parallel_devices) > 1:
            def parallel_loss_and_grad(parallel_params, parallel_batch):
                (value, regime_value), gradients = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(parallel_params, parallel_batch)
                value = jax.lax.pmean(value, axis_name="data")
                regime_value = jax.lax.pmean(regime_value, axis_name="data")
                gradients = jax.tree_util.tree_map(
                    lambda leaf: jax.lax.pmean(leaf, axis_name="data"), gradients
                )
                return (value, regime_value), gradients

            parallel_value_and_grad = jax.pmap(
                parallel_loss_and_grad,
                axis_name="data",
                in_axes=(None, 0),
                devices=jax.devices()[: int(args.data_parallel_devices)],
            )
        evaluate_loss = jax.jit(loss_fn)
        validation_panel = build_diagnostic_panel(
            validation_rng,
            grouped,
            anchors,
            manifest,
            coefficient_key,
            regimes=training_regimes,
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
            autonomous_burnin_steps=args.autonomous_history_burnin_steps,
            include_heat_flux_gradient_history=args.closure_history_input,
            reference_dt=reference_dt,
        )
        training_diagnostic_panel = build_diagnostic_panel(
            np.random.default_rng(args.seed + 2),
            grouped,
            anchors,
            manifest,
            coefficient_key,
            regimes=training_regimes,
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
            autonomous_burnin_steps=args.autonomous_history_burnin_steps,
            include_heat_flux_gradient_history=args.closure_history_input,
            reference_dt=reference_dt,
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
                    regimes=training_regimes,
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
                    relative_time_block_steps=relative_time_block_steps,
                    relative_time_block_count=relative_time_block_count,
                    convergence_floor_rms=convergence_floor_rms,
                    memory_backend=args.memory_backend,
                    memory_stride=args.memory_stride,
                    scan_unroll=args.scan_unroll,
                    input_scaling=args.input_scaling,
                    dynamic_amplitude_floor=args.dynamic_amplitude_floor,
                    allow_uniform_heating=args.allow_uniform_heating,
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
            amplitude = jnp.asarray(batch["amplitude"])
            regime_index = jnp.asarray(batch["regime_index"])
            if "target_fields" in batch:
                target_fields = jnp.asarray(batch["target_fields"])
            else:
                target_states = jnp.asarray(batch["targets"])
                target_shape = target_states.shape
                target_fields = primitive_fields(
                    target_states.reshape(
                        target_shape[0] * target_shape[1], 3, target_shape[-1]
                    ),
                    jnp.asarray(k_arr, dtype=jnp.float32),
                    poisson_sign=poisson_sign,
                ).reshape(target_shape[0], target_shape[1], 4, target_shape[-1])
            total_steps = int(target_fields.shape[1])
            if args.relative_trajectory_loss:
                target_shape = target_fields.shape
                if (
                    "trajectory_target_norm" in batch
                    and convergence_floor_rms is None
                ):
                    trajectory_target_norm = jnp.asarray(
                        batch["trajectory_target_norm"], dtype=target_fields.dtype
                    )
                elif convergence_floor_rms is not None:
                    full_indices = jnp.broadcast_to(
                        jnp.arange(1, target_shape[1] + 1, dtype=jnp.int32)[None, :],
                        (target_shape[0], target_shape[1]),
                    )
                    membership = jax.nn.one_hot(
                        jnp.minimum(
                            (full_indices - 1) // relative_time_block_steps,
                            relative_time_block_count - 1,
                        ),
                        relative_time_block_count,
                        dtype=target_fields.dtype,
                    )
                    target_energy = jnp.sum(jnp.square(target_fields), axis=-1)
                    trajectory_target_norm = jnp.einsum(
                        "btq,btc->bqc", membership, target_energy
                    )
                    full_counts = jnp.sum(membership, axis=1)
                    floor_by_regime = jnp.asarray(
                        convergence_floor_rms, dtype=target_fields.dtype
                    )[regime_index]
                    trajectory_target_norm = jnp.maximum(
                        trajectory_target_norm,
                        full_counts[:, :, None]
                        * target_shape[-1]
                        * jnp.square(floor_by_regime[:, None, :]),
                    )
                else:
                    trajectory_target_norm = jnp.sum(
                        jnp.square(target_fields), axis=(1, 3)
                    )
                if convergence_floor_rms is None and np.any(
                    np.asarray(trajectory_target_norm) <= 0.0
                ):
                    raise ValueError(
                        "Relative trajectory loss excludes zero-norm target trajectories"
                    )
            else:
                trajectory_target_norm = jnp.ones(
                    (target_fields.shape[0],), dtype=target_fields.dtype
                )
            accumulated_grads = None
            accumulated_loss = 0.0
            accumulated_regime = np.zeros((len(training_regimes),), dtype=np.float64)
            completed = 0
            first_chunk = True
            failure_step = None
            while completed < total_steps:
                chunk_length = min(int(stage_horizon), total_steps - completed)
                target_chunk = target_fields[:, completed : completed + chunk_length]
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
                    jnp.broadcast_to(
                        jnp.arange(
                            completed + 1,
                            completed + chunk_length + 1,
                            dtype=jnp.int32,
                        )[None, :],
                        (target_fields.shape[0], chunk_length),
                    ),
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
                regimes=training_regimes,
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
                    regimes=training_regimes,
                    memory_steps=args.memory_steps,
                    memory_stride=args.memory_stride,
                    source_nx=source_nx,
                    rollout_nx=args.rollout_Nx,
                    translation_augmentation=False,
                    primitive_target_histories=primitive_target_histories,
                    trajectory_target_norm_by_id=trajectory_target_norm_by_id,
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
                weights.append(int(case_rows[training_regimes[0]].size))
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
        loss_ema = (
            None
            if resumed_state is None or not math.isfinite(resumed_state["loss_ema"])
            else float(resumed_state["loss_ema"])
        )
        if args.resume_run is None and global_epoch == 0:
            _save_checkpoint(
                outdir / "epoch000_low_moment_closure.npz", params, metadata, stats
            )
            _save_training_state(
                outdir / "training_state.npz",
                params=params,
                optimizer=optimizer,
                rng=rng,
                global_epoch=0,
                loss_ema=loss_ema,
                best_val=best_val,
                histories={
                    "train_loss": train_history,
                    "train_ema_loss": train_ema_history,
                    "train_update_norm": train_update_norm_history,
                    "train_update_scale_min": train_update_scale_min_history,
                    "train_eval_epochs": train_eval_epochs,
                    "train_eval_loss": train_eval_history,
                    "train_eval_regime_loss": train_eval_regime_history,
                    "val_epochs": val_epochs,
                    "val_loss": val_history,
                    "val_regime_loss": val_regime_history,
                    "autonomous_val_epochs": autonomous_val_epochs,
                    "autonomous_val_loss": autonomous_val_history,
                    "autonomous_val_regime_loss": autonomous_val_regime_history,
                },
            )
        while global_epoch < min(stage_end_epoch, int(args.epochs)):
            global_epoch += 1
            partial_epoch = (
                None if resumed_state is None else resumed_state.get("in_epoch")
            )
            if partial_epoch is not None:
                if int(partial_epoch["number"]) != global_epoch:
                    raise ValueError(
                        "Partial-epoch state does not follow the last completed epoch"
                    )
                completed_epoch_steps = int(partial_epoch["completed_steps"])
                if not 0 < completed_epoch_steps < int(args.steps_per_epoch):
                    raise ValueError("Invalid partial-epoch completed step count")
                rng.bit_generator.state = partial_epoch["rng_state"]
                epoch_losses = partial_epoch["losses"].astype(float).tolist()
                epoch_regime_losses = (
                    partial_epoch["regime_losses"].astype(float).tolist()
                )
                epoch_grad_norms = partial_epoch["grad_norms"].astype(float).tolist()
                epoch_update_norms = (
                    partial_epoch["update_norms"].astype(float).tolist()
                )
                epoch_update_scales = (
                    partial_epoch["update_scales"].astype(float).tolist()
                )
                print(
                    f"[train] resuming epoch {global_epoch} at update "
                    f"{completed_epoch_steps}/{args.steps_per_epoch}"
                )
            else:
                completed_epoch_steps = 0
                epoch_losses = []
                epoch_regime_losses = []
                epoch_grad_norms = []
                epoch_update_norms = []
                epoch_update_scales = []
            epoch_failed_batches = 0
            epoch_earliest_failure_step = None
            learning_rate = _cosine_learning_rate(
                int(np.asarray(optimizer["step"])),
                planned_epochs * int(args.steps_per_epoch),
                args.learning_rate,
                args.final_learning_rate,
            )
            if args.training_schedule == "continuous_trajectories":
                training_steps = build_complete_trajectory_case_batches(
                    rng,
                    grouped,
                    regimes=training_regimes,
                    split="train",
                    batch_size_per_regime=args.batch_size,
                    shuffle=True,
                )
            else:
                epoch_rng_state = json.loads(
                    json.dumps(rng.bit_generator.state, sort_keys=True)
                )
                training_steps = range(
                    completed_epoch_steps, int(args.steps_per_epoch)
                )
                if args.full_anchor_sweep:
                    epoch_window_selections = build_balanced_full_anchor_epoch(
                        rng,
                        anchors,
                        regimes=training_regimes,
                        split="train",
                        batch_size_per_regime=args.batch_size,
                    )
                else:
                    epoch_window_selections = tuple(
                        selection
                        for _ in range(int(args.training_passes_per_epoch))
                        for selection in build_balanced_random_window_epoch(
                            rng,
                            anchors,
                            regimes=training_regimes,
                            split="train",
                            batch_size_per_regime=args.batch_size,
                            deterministic_epoch=(
                                global_epoch if args.deterministic_anchor_cycle else None
                            ),
                        )
                    )
                expected_batches = int(args.steps_per_epoch) * int(
                    args.gradient_accumulation_steps
                )
                if len(epoch_window_selections) != expected_batches:
                    raise ValueError(
                        "steps-per-epoch * gradient-accumulation-steps must cover "
                        f"the configured balanced IC passes: expected {len(epoch_window_selections)}, "
                        f"configured {expected_batches}"
                    )
                epoch_window_iterator = iter(epoch_window_selections)
                for _ in range(
                    completed_epoch_steps * int(args.gradient_accumulation_steps)
                ):
                    next(epoch_window_iterator)
            for epoch_step, step_selection in enumerate(
                training_steps, start=completed_epoch_steps
            ):
                if args.training_schedule == "continuous_trajectories":
                    batch = sample_complete_trajectory_batch(
                        rng,
                        grouped,
                        manifest,
                        coefficient_key,
                        step_selection,
                        regimes=training_regimes,
                        memory_steps=args.memory_steps,
                        memory_stride=args.memory_stride,
                        source_nx=source_nx,
                        rollout_nx=args.rollout_Nx,
                        translation_augmentation=args.translation_augmentation,
                        primitive_target_histories=primitive_target_histories,
                        trajectory_target_norm_by_id=trajectory_target_norm_by_id,
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
                    (
                        params,
                        optimizer,
                        grad_norm,
                        update_norm,
                        update_scale,
                    ) = _adam_step(
                        params,
                        grads,
                        optimizer,
                        args.learning_rate,
                        args.grad_clip,
                        update_norm_cap=args.update_norm_cap,
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
                    epoch_update_norms.append(float(update_norm))
                    epoch_update_scales.append(float(update_scale))
                    continue
                accumulated_grads = None
                optimizer_step_losses = []
                optimizer_step_batches = []
                for _ in range(int(args.gradient_accumulation_steps)):
                    window_selection = next(epoch_window_iterator)
                    batch = sample_batch(
                        rng,
                        grouped,
                        anchors,
                        manifest,
                        coefficient_key,
                        regimes=training_regimes,
                        split="train",
                        batch_size_per_regime=args.batch_size,
                        horizon=stage_horizon,
                        memory_steps=args.memory_steps,
                        memory_stride=args.memory_stride,
                        source_nx=source_nx,
                        rollout_nx=args.rollout_Nx,
                        domain_length=domain_length,
                        translation_augmentation=args.translation_augmentation,
                        explicit_selection=window_selection,
                        autonomous_burnin_steps=args.autonomous_history_burnin_steps,
                        include_heat_flux_gradient_history=args.closure_history_input,
                        reference_steps_per_solver_step=(
                            reference_steps_per_solver_step
                        ),
                    )
                    if parallel_value_and_grad is None:
                        batch_jax = {
                            key: jnp.asarray(value) for key, value in batch.items()
                        }
                    else:
                        device_count = int(args.data_parallel_devices)
                        regime_indices = np.asarray(batch["regime_index"])
                        per_device_indices = [[] for _ in range(device_count)]
                        for regime_index in (REGIMES.index(regime) for regime in training_regimes):
                            rows = np.flatnonzero(regime_indices == regime_index)
                            if rows.size % device_count:
                                raise ValueError(
                                    "Each regime's minibatch must divide evenly across "
                                    "data-parallel devices"
                                )
                            for device_index, chunk in enumerate(
                                np.split(rows, device_count)
                            ):
                                per_device_indices[device_index].extend(chunk.tolist())
                        ordered_indices = np.asarray(
                            [row for rows in per_device_indices for row in rows],
                            dtype=np.int32,
                        )
                        batch_jax = {
                            key: jnp.asarray(value[ordered_indices]).reshape(
                                device_count,
                                len(ordered_indices) // device_count,
                                *value.shape[1:],
                            )
                            for key, value in batch.items()
                        }
                    optimizer_step_batches.append(batch_jax)
                    if parallel_value_and_grad is None:
                        (loss_value, regime_value), grads = value_and_grad(
                            params, batch_jax
                        )
                    else:
                        (replicated_loss, replicated_regime), replicated_grads = (
                            parallel_value_and_grad(params, batch_jax)
                        )
                        loss_value = replicated_loss[0]
                        regime_value = replicated_regime[0]
                        grads = jax.tree_util.tree_map(
                            lambda leaf: leaf[0], replicated_grads
                        )
                    loss_float = float(loss_value)
                    if not math.isfinite(loss_float):
                        selection_text = {
                            regime: {
                                "case_rows": np.asarray(value[0]).tolist(),
                                "reset_times": (
                                    np.asarray(value[1], dtype=np.float64)
                                    * reference_dt
                                ).tolist(),
                            }
                            for regime, value in window_selection.items()
                        }
                        raise FloatingPointError(
                            f"Non-finite loss at epoch {global_epoch}, H={stage_horizon}, "
                            f"regime_loss={np.asarray(regime_value).tolist()}, "
                            f"selection={selection_text}"
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
                learning_rate = _cosine_learning_rate(
                    int(np.asarray(optimizer["step"])),
                    planned_epochs * int(args.steps_per_epoch),
                    args.learning_rate,
                    args.final_learning_rate,
                )
                (
                    params,
                    optimizer,
                    grad_norm,
                    update_norm,
                    update_scale,
                ) = _adam_step(
                    params,
                    grads,
                    optimizer,
                    learning_rate,
                    args.grad_clip,
                    args.weight_decay,
                    args.update_norm_cap,
                )
                should_report_update = args.report_every_update_descent or (
                    args.require_first_update_descent
                    and int(np.asarray(optimizer["step"])) == 1
                )
                if should_report_update:
                    post_update_loss = float(
                        np.mean(
                            [
                                float(evaluate_loss(params, selected_batch)[0])
                                for selected_batch in optimizer_step_batches
                            ]
                        )
                    )
                    pre_update_loss = float(np.mean(optimizer_step_losses))
                    print(
                        "[update-check] exact-batch "
                        f"step={int(np.asarray(optimizer['step']))} "
                        f"before={pre_update_loss:.6e} after={post_update_loss:.6e} "
                        f"ratio={post_update_loss / max(pre_update_loss, 1e-30):.6f}"
                    )
                    if (
                        args.require_first_update_descent
                        and int(np.asarray(optimizer["step"])) == 1
                        and (
                            not math.isfinite(post_update_loss)
                            or post_update_loss >= pre_update_loss
                        )
                    ):
                        raise FloatingPointError(
                            "First accepted update did not lower its exact accumulated batch"
                        )
                optimizer_step_loss = float(np.mean(optimizer_step_losses))
                loss_ema = (
                    optimizer_step_loss
                    if loss_ema is None
                    else args.loss_ema_decay * loss_ema
                    + (1.0 - args.loss_ema_decay) * optimizer_step_loss
                )
                epoch_grad_norms.append(float(grad_norm))
                epoch_update_norms.append(float(update_norm))
                epoch_update_scales.append(float(update_scale))
                completed_steps = epoch_step + 1
                if (
                    args.checkpoint_every_updates
                    and completed_steps < int(args.steps_per_epoch)
                    and completed_steps % int(args.checkpoint_every_updates) == 0
                ):
                    resume_rng = np.random.default_rng()
                    resume_rng.bit_generator.state = epoch_rng_state
                    _save_training_state(
                        outdir / "training_state.npz",
                        params=params,
                        optimizer=optimizer,
                        rng=resume_rng,
                        global_epoch=global_epoch - 1,
                        loss_ema=loss_ema,
                        best_val=best_val,
                        histories={
                            "train_loss": train_history,
                            "train_ema_loss": train_ema_history,
                            "train_update_norm": train_update_norm_history,
                            "train_update_scale_min": train_update_scale_min_history,
                            "train_eval_epochs": train_eval_epochs,
                            "train_eval_loss": train_eval_history,
                            "train_eval_regime_loss": train_eval_regime_history,
                            "val_epochs": val_epochs,
                            "val_loss": val_history,
                            "val_regime_loss": val_regime_history,
                            "autonomous_val_epochs": autonomous_val_epochs,
                            "autonomous_val_loss": autonomous_val_history,
                            "autonomous_val_regime_loss": autonomous_val_regime_history,
                        },
                        in_epoch={
                            "number": global_epoch,
                            "completed_steps": completed_steps,
                            "rng_state": epoch_rng_state,
                            "losses": epoch_losses,
                            "regime_losses": epoch_regime_losses,
                            "grad_norms": epoch_grad_norms,
                            "update_norms": epoch_update_norms,
                            "update_scales": epoch_update_scales,
                        },
                    )
                    os.sync()
                    print(
                        f"[checkpoint] epoch {global_epoch} partial update "
                        f"{completed_steps}/{args.steps_per_epoch}"
                    )
            resumed_state = None
            mean_loss = float(np.mean(epoch_losses))
            mean_regime_loss = np.mean(epoch_regime_losses, axis=0)
            train_history.append(mean_loss)
            train_ema_history.append(float(loss_ema))
            train_update_norm_history.append(float(np.mean(epoch_update_norms)))
            train_update_scale_min_history.append(float(np.min(epoch_update_scales)))
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
                f"update_mean={np.mean(epoch_update_norms):.3e} "
                f"update_scale_min={np.min(epoch_update_scales):.3e} "
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
            completed_metadata = dict(metadata)
            completed_metadata["completed_epochs"] = global_epoch
            completed_metadata["optimizer_steps"] = int(np.asarray(optimizer["step"]))
            completed_metadata["current_learning_rate"] = float(learning_rate)
            _save_checkpoint(
                outdir / f"epoch{global_epoch:03d}_low_moment_closure.npz",
                params,
                completed_metadata,
                stats,
            )
            _save_training_state(
                outdir / "training_state.npz",
                params=params,
                optimizer=optimizer,
                rng=rng,
                global_epoch=global_epoch,
                loss_ema=loss_ema,
                best_val=best_val,
                histories={
                    "train_loss": train_history,
                    "train_ema_loss": train_ema_history,
                    "train_update_norm": train_update_norm_history,
                    "train_update_scale_min": train_update_scale_min_history,
                    "train_eval_epochs": train_eval_epochs,
                    "train_eval_loss": train_eval_history,
                    "train_eval_regime_loss": train_eval_regime_history,
                    "val_epochs": val_epochs,
                    "val_loss": val_history,
                    "val_regime_loss": val_regime_history,
                    "autonomous_val_epochs": autonomous_val_epochs,
                    "autonomous_val_loss": autonomous_val_history,
                    "autonomous_val_regime_loss": autonomous_val_regime_history,
                },
            )
        curriculum_epoch_offset = stage_end_epoch
    checkpoint = outdir / "low_moment_closure.npz"
    final_epoch_checkpoint = outdir / f"epoch{global_epoch:03d}_low_moment_closure.npz"
    if final_epoch_checkpoint.is_file():
        # The epoch checkpoint already materialized every device array.  Copy
        # that durable artifact instead of triggering a second large JAX host
        # transfer after the final accepted update.
        shutil.copy2(final_epoch_checkpoint, checkpoint)
    else:
        _save_checkpoint(checkpoint, params, metadata, stats)
    np.savez(
        outdir / "training_metrics.npz",
        train_loss=np.asarray(train_history),
        train_ema_loss=np.asarray(train_ema_history),
        train_update_norm=np.asarray(train_update_norm_history),
        train_update_scale_min=np.asarray(train_update_scale_min_history),
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
            regimes=training_regimes,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
            dt=dt,
            reference_dt=reference_dt,
            width=args.width,
            memory_steps=args.memory_steps,
            memory_stride=args.memory_stride,
            scan_unroll=args.scan_unroll,
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
            memory_backend=args.memory_backend,
            input_scaling=args.input_scaling,
            dynamic_amplitude_floor=args.dynamic_amplitude_floor,
            allow_uniform_heating=args.allow_uniform_heating,
        )


if __name__ == "__main__":
    main()
