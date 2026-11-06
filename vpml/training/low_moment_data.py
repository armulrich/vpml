"""Reference conversion, primitive-target caches, and training normalization."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from vpml.datasets.interface_flux import (
    IC_SPLIT_TRAIN,
    load_ic_manifest,
    load_sharded_reference,
    sha256_json,
)
from vpml.fourier import restrict_rfft_numpy
from vpml.training.low_moment_configuration import CHECKPOINT_SCHEMA, REGIMES


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


def _restrict_rfft(values: np.ndarray, source_nx: int, target_nx: int) -> np.ndarray:
    """Restrict coefficients while retaining the legacy independent-copy contract."""
    return restrict_rfft_numpy(
        values, source_nx=source_nx, target_nx=target_nx
    ).copy()


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
