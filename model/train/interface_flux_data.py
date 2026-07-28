"""Manifest and sharded-cache helpers for interface-flux training data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


IC_MANIFEST_FORMAT = "landau_interface_flux_ic_manifest_v1"
REFERENCE_CACHE_FORMAT = "landau_interface_flux_sharded_reference_v1"
IC_GENERATOR_VERSION = "multimode_stratified_v1"
IC_SPLIT_TRAIN = "train"
IC_SPLIT_HELDOUT = "heldout"
IC_SPLITS = (IC_SPLIT_TRAIN, IC_SPLIT_HELDOUT)
DEFAULT_AMPLITUDE_RANGES = {
    "linear_landau": (0.005, 0.02),
    "nonlinear_landau_weak": (0.02, 0.20),
    "nonlinear_landau_strong": (0.20, 0.65),
}


def sample_initial_condition(
    rng: np.random.Generator,
    x: np.ndarray,
    modes: Sequence[float],
    eps: float,
) -> np.ndarray:
    """Sample the retained legacy random multimode perturbation."""
    amplitudes = rng.uniform(0.5, 1.5, size=len(modes))
    phases = rng.uniform(0.0, 2.0 * math.pi, size=len(modes))
    perturbation = np.zeros_like(x)
    for amplitude, phase, mode in zip(amplitudes, phases, modes):
        perturbation += amplitude * np.cos(float(mode) * x + phase)
    return (float(eps) / max(len(modes), 1)) * perturbation


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_bytes))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def evaluate_manifest_case(case: Mapping[str, object], x: np.ndarray) -> np.ndarray:
    """Evaluate one immutable multimode perturbation on a requested spatial grid."""
    x_arr = np.asarray(x, dtype=np.float64)
    modes = np.asarray(case["modes"], dtype=np.float64)
    weights = np.asarray(case["mode_weights"], dtype=np.float64)
    phases = np.asarray(case["relative_phases"], dtype=np.float64)
    if not (modes.shape == weights.shape == phases.shape) or modes.ndim != 1:
        raise ValueError(f"Malformed IC mode arrays for case {case.get('case_id')!r}")
    shape = np.sum(
        weights[:, None] * np.cos(modes[:, None] * x_arr[None, :] + phases[:, None]),
        axis=0,
    )
    return (
        float(case["epsilon"])
        * float(case["shape_normalization"])
        * shape
    ).astype(np.float64)


def _heldout_indices(
    rng: np.random.Generator,
    *,
    cases_per_regime: int,
    heldout_per_regime: int,
) -> set[int]:
    if int(heldout_per_regime) <= 0 or int(heldout_per_regime) >= int(cases_per_regime):
        raise ValueError("heldout_per_regime must be between zero and cases_per_regime")
    bins = np.array_split(np.arange(int(cases_per_regime), dtype=np.int32), int(heldout_per_regime))
    return {
        int(group[int(rng.integers(0, len(group)))])
        for group in bins
    }


def build_ic_manifest(
    *,
    cases_per_regime: int = 20,
    heldout_per_regime: int = 4,
    generation_seed: int = 1729,
    split_seed: int = 2718,
    modes: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    domain_length: float = 4.0 * math.pi,
    normalization_grid_points: int = 16384,
    amplitude_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Dict[str, object]:
    """Build a deterministic, stratified, whole-trajectory IC manifest."""
    regimes = tuple(DEFAULT_AMPLITUDE_RANGES)
    ranges = dict(DEFAULT_AMPLITUDE_RANGES if amplitude_ranges is None else amplitude_ranges)
    if set(ranges) != set(regimes):
        raise ValueError(f"amplitude_ranges must define exactly {regimes!r}")
    modes_tuple = tuple(float(value) for value in modes)
    if not modes_tuple or any(value <= 0.0 for value in modes_tuple):
        raise ValueError("modes must contain positive Fourier wavenumbers")
    if int(cases_per_regime) <= 1:
        raise ValueError("cases_per_regime must exceed one")
    if int(normalization_grid_points) < 1024:
        raise ValueError("normalization_grid_points must be at least 1024")

    generation_sequences = np.random.SeedSequence(int(generation_seed)).spawn(len(regimes))
    split_sequences = np.random.SeedSequence(int(split_seed)).spawn(len(regimes))
    x_norm = np.linspace(
        0.0,
        float(domain_length),
        int(normalization_grid_points),
        endpoint=False,
        dtype=np.float64,
    )
    cases = []
    for regime_index, regime in enumerate(regimes):
        rng = np.random.default_rng(generation_sequences[regime_index])
        split_rng = np.random.default_rng(split_sequences[regime_index])
        heldout = _heldout_indices(
            split_rng,
            cases_per_regime=int(cases_per_regime),
            heldout_per_regime=int(heldout_per_regime),
        )
        eps_min, eps_max = (float(value) for value in ranges[regime])
        if not 0.0 < eps_min < eps_max < 1.0:
            raise ValueError(f"Invalid amplitude range for {regime}: {(eps_min, eps_max)!r}")
        for case_index in range(int(cases_per_regime)):
            stratum_position = (case_index + float(rng.uniform())) / float(cases_per_regime)
            epsilon = eps_min + (eps_max - eps_min) * stratum_position
            weights = rng.normal(size=len(modes_tuple))
            weights /= max(float(np.linalg.norm(weights)), 1e-30)
            phases = rng.uniform(0.0, 2.0 * math.pi, size=len(modes_tuple))
            translation = phases[0] / modes_tuple[0]
            phases = np.mod(
                phases - np.asarray(modes_tuple, dtype=np.float64) * translation,
                2.0 * math.pi,
            )
            phases[0] = 0.0
            raw_shape = np.sum(
                weights[:, None]
                * np.cos(
                    np.asarray(modes_tuple)[:, None] * x_norm[None, :]
                    + phases[:, None]
                ),
                axis=0,
            )
            shape_normalization = 1.0 / max(float(np.max(np.abs(raw_shape))), 1e-30)
            cases.append(
                {
                    "case_id": f"{regime}_ic{case_index:02d}",
                    "regime": regime,
                    "epsilon": float(epsilon),
                    "modes": list(modes_tuple),
                    "mode_weights": [float(value) for value in weights],
                    "relative_phases": [float(value) for value in phases],
                    "shape_normalization": float(shape_normalization),
                    "split": (
                        IC_SPLIT_HELDOUT if case_index in heldout else IC_SPLIT_TRAIN
                    ),
                }
            )

    manifest: Dict[str, object] = {
        "format": IC_MANIFEST_FORMAT,
        "generator_version": IC_GENERATOR_VERSION,
        "generation_seed": int(generation_seed),
        "split_seed": int(split_seed),
        "cases_per_regime": int(cases_per_regime),
        "heldout_per_regime": int(heldout_per_regime),
        "domain_length": float(domain_length),
        "normalization_grid_points": int(normalization_grid_points),
        "amplitude_ranges": {
            regime: [float(value) for value in ranges[regime]]
            for regime in regimes
        },
        "modes": list(modes_tuple),
        "cases": cases,
    }
    manifest["sha256"] = sha256_json(manifest)
    validate_ic_manifest(manifest)
    return manifest


def validate_ic_manifest(manifest: Mapping[str, object]) -> None:
    if str(manifest.get("format")) != IC_MANIFEST_FORMAT:
        raise ValueError(f"Unsupported IC manifest format {manifest.get('format')!r}")
    stored_hash = str(manifest.get("sha256", ""))
    without_hash = dict(manifest)
    without_hash.pop("sha256", None)
    if not stored_hash or stored_hash != sha256_json(without_hash):
        raise ValueError("IC manifest SHA256 does not match its contents")
    cases = list(manifest.get("cases", ()))
    expected_per_regime = int(manifest["cases_per_regime"])
    expected_heldout = int(manifest["heldout_per_regime"])
    case_ids = [str(case["case_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("IC manifest case IDs must be unique")
    for regime in DEFAULT_AMPLITUDE_RANGES:
        selected = [case for case in cases if str(case["regime"]) == regime]
        if len(selected) != expected_per_regime:
            raise ValueError(
                f"IC manifest has {len(selected)} {regime} cases, expected {expected_per_regime}"
            )
        heldout = [case for case in selected if str(case["split"]) == IC_SPLIT_HELDOUT]
        if len(heldout) != expected_heldout:
            raise ValueError(
                f"IC manifest has {len(heldout)} held-out {regime} cases, "
                f"expected {expected_heldout}"
            )
        if any(str(case["split"]) not in IC_SPLITS for case in selected):
            raise ValueError(f"IC manifest has an invalid split in regime {regime}")


def load_ic_manifest(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_ic_manifest(manifest)
    return manifest


def load_or_create_ic_manifest(path: Path, **build_kwargs: object) -> Dict[str, object]:
    path = Path(path)
    expected = build_ic_manifest(**build_kwargs)
    if path.exists():
        actual = load_ic_manifest(path)
        if str(actual["sha256"]) != str(expected["sha256"]):
            raise ValueError(
                f"Existing IC manifest {path} does not match the requested generator settings"
            )
        return actual
    _write_json_atomic(path, expected)
    return expected


def reference_cache_directory(root: Path, configuration: Mapping[str, object]) -> Path:
    return Path(root) / sha256_json(dict(configuration))[:20]


def initialize_reference_cache(
    cache_dir: Path,
    *,
    configuration: Mapping[str, object],
    manifest: Mapping[str, object],
) -> Dict[str, object]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cases").mkdir(exist_ok=True)
    (cache_dir / "snapshots").mkdir(exist_ok=True)
    metadata = {
        "format": REFERENCE_CACHE_FORMAT,
        "configuration": dict(configuration),
        "configuration_sha256": sha256_json(dict(configuration)),
        "manifest_sha256": str(manifest["sha256"]),
    }
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            actual = json.load(handle)
        if actual != metadata:
            raise ValueError(f"Reference-cache metadata mismatch at {cache_dir}")
    else:
        _write_json_atomic(metadata_path, metadata)
    manifest_path = cache_dir / "ic_manifest.json"
    if manifest_path.exists():
        if str(load_ic_manifest(manifest_path)["sha256"]) != str(manifest["sha256"]):
            raise ValueError(f"Reference-cache manifest mismatch at {cache_dir}")
    else:
        _write_json_atomic(manifest_path, dict(manifest))
    return metadata


def case_shard_paths(cache_dir: Path, case_id: str) -> Tuple[Path, Path, Path]:
    case_id = str(case_id)
    return (
        Path(cache_dir) / "cases" / f"{case_id}.npy",
        Path(cache_dir) / "cases" / f"{case_id}.json",
        Path(cache_dir) / "snapshots" / f"{case_id}.npz",
    )


def case_shard_is_complete(
    cache_dir: Path,
    case_id: str,
    *,
    expected_shape: Sequence[int],
    expected_dtype: np.dtype = np.dtype(np.complex64),
) -> bool:
    data_path, marker_path, snapshot_path = case_shard_paths(cache_dir, case_id)
    if not data_path.exists() or not marker_path.exists() or not snapshot_path.exists():
        return False
    try:
        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
        array = np.load(data_path, mmap_mode="r")
        if tuple(array.shape) != tuple(int(value) for value in expected_shape):
            return False
        if np.dtype(array.dtype) != np.dtype(expected_dtype):
            return False
        if marker.get("shape") != list(int(value) for value in expected_shape):
            return False
        if str(marker.get("dtype")) != np.dtype(expected_dtype).name:
            return False
        with np.load(snapshot_path) as snapshots:
            if not {
                "times",
                "energy",
                "E_hat_hist_times",
                "E_hat_hist",
                "k_arr",
                "perturbation_x",
                "snapshot_times",
                "snapshot_f",
            }.issubset(snapshots.files):
                return False
        return str(marker.get("sha256")) == sha256_file(data_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def write_case_shard(
    cache_dir: Path,
    case: Mapping[str, object],
    history: np.ndarray,
    *,
    history_times: np.ndarray,
    snapshots: Optional[Mapping[str, np.ndarray]] = None,
) -> Path:
    """Write one case atomically without changing coefficient values."""
    data_path, marker_path, snapshot_path = case_shard_paths(
        cache_dir, str(case["case_id"])
    )
    history_arr = np.asarray(history)
    if history_arr.ndim != 3:
        raise ValueError(f"Projected history must be 3D, got {history_arr.shape}")
    temporary = data_path.with_name(f".{data_path.name}.tmp")
    mapped = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.complex64,
        shape=history_arr.shape,
    )
    for start in range(0, int(history_arr.shape[0]), 128):
        stop = min(start + 128, int(history_arr.shape[0]))
        mapped[start:stop] = history_arr[start:stop]
    mapped.flush()
    del mapped
    temporary.replace(data_path)
    digest = sha256_file(data_path)
    marker = {
        "case_id": str(case["case_id"]),
        "regime": str(case["regime"]),
        "split": str(case["split"]),
        "shape": list(int(value) for value in history_arr.shape),
        "dtype": np.dtype(np.complex64).name,
        "sha256": digest,
        "history_times_start": float(np.asarray(history_times)[0]),
        "history_times_stop": float(np.asarray(history_times)[-1]),
        "history_count": int(np.asarray(history_times).shape[0]),
    }
    _write_json_atomic(marker_path, marker)
    if snapshots is not None:
        snapshot_tmp = snapshot_path.with_name(f".{snapshot_path.name}.tmp")
        with snapshot_tmp.open("wb") as handle:
            np.savez(handle, **{key: np.asarray(value) for key, value in snapshots.items()})
            handle.flush()
            os.fsync(handle.fileno())
        snapshot_tmp.replace(snapshot_path)
    return data_path


def load_sharded_reference(
    cache_dir: Path,
    manifest: Mapping[str, object],
    *,
    coeff_key: str,
) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for case in manifest["cases"]:
        data_path, marker_path, _ = case_shard_paths(
            cache_dir, str(case["case_id"])
        )
        if not data_path.exists() or not marker_path.exists():
            raise FileNotFoundError(f"Incomplete reference shard for {case['case_id']}")
        history = np.load(data_path, mmap_mode="r")
        regime = str(case["regime"])
        group = grouped.setdefault(
            regime,
            {
                coeff_key: [],
                "case_ids": [],
                "case_splits": [],
            },
        )
        group[coeff_key].append(history)
        group["case_ids"].append(str(case["case_id"]))
        group["case_splits"].append(str(case["split"]))
    for group in grouped.values():
        group[coeff_key] = tuple(group[coeff_key])
        group["case_ids"] = np.asarray(group["case_ids"], dtype=np.str_)
        group["case_splits"] = np.asarray(group["case_splits"], dtype=np.str_)
    return grouped


@dataclass
class StreamingMoments:
    """Chan-Golub-LeVeque block-merge moments in float64."""

    count: int = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None

    def update(self, values: np.ndarray) -> None:
        block = np.asarray(values, dtype=np.float64)
        if block.ndim != 2:
            raise ValueError(f"Expected a two-dimensional feature block, got {block.shape}")
        if int(block.shape[0]) == 0:
            return
        block_count = int(block.shape[0])
        block_mean = np.mean(block, axis=0, dtype=np.float64)
        centered = block - block_mean
        block_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
        if self.count == 0:
            self.count = block_count
            self.mean = block_mean
            self.m2 = block_m2
            return
        assert self.mean is not None and self.m2 is not None
        total = self.count + block_count
        delta = block_mean - self.mean
        self.mean = self.mean + delta * (float(block_count) / float(total))
        self.m2 = (
            self.m2
            + block_m2
            + delta * delta * (float(self.count * block_count) / float(total))
        )
        self.count = total

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count <= 0 or self.mean is None or self.m2 is None:
            raise ValueError("Cannot finalize empty streaming moments")
        variance = np.maximum(self.m2 / float(self.count), 0.0)
        return self.mean.copy(), np.sqrt(variance)


def grouped_history_gather(
    histories: Sequence[np.ndarray] | np.ndarray,
    case_indices: np.ndarray,
    time_indices: np.ndarray,
    *,
    hermite_slice: object,
    fourier_slice: object,
) -> np.ndarray:
    """Gather memmap rows by case while restoring the original sampled order."""
    cases = np.asarray(case_indices, dtype=np.int32)
    times = np.asarray(time_indices, dtype=np.int32)
    try:
        cases, times = np.broadcast_arrays(cases, times)
    except ValueError as exc:
        raise ValueError(
            "case_indices and time_indices must have broadcast-compatible shapes"
        ) from exc
    if isinstance(histories, np.ndarray) and histories.ndim == 4:
        return np.asarray(histories[(cases, times, hermite_slice, fourier_slice)])
    sequence = tuple(histories)
    if not sequence:
        raise ValueError("histories must be nonempty")
    if cases.ndim == 2 and np.all(cases == cases[:, :1]):
        sample = np.asarray(
            sequence[0][(int(times[0, 0]), hermite_slice, fourier_slice)]
        )
        output = np.empty(cases.shape + sample.shape, dtype=sample.dtype)
        row_cases = cases[:, 0]
        for case_index in np.unique(row_cases):
            positions = np.flatnonzero(row_cases == int(case_index))
            ordered = positions[
                np.argsort(np.min(times[positions], axis=1), kind="stable")
            ]
            output[ordered] = sequence[int(case_index)][
                times[ordered], hermite_slice, fourier_slice
            ]
        return output
    sample = np.asarray(
        sequence[0][(int(times.reshape(-1)[0]), hermite_slice, fourier_slice)]
    )
    output = np.empty(cases.shape + sample.shape, dtype=sample.dtype)
    flat_cases = cases.reshape(-1)
    flat_times = times.reshape(-1)
    flat_output = output.reshape((flat_cases.size,) + sample.shape)
    for case_index in np.unique(flat_cases):
        positions = np.flatnonzero(flat_cases == int(case_index))
        ordered = positions[np.argsort(flat_times[positions], kind="stable")]
        flat_output[ordered] = sequence[int(case_index)][
            flat_times[ordered], hermite_slice, fourier_slice
        ]
    return output
