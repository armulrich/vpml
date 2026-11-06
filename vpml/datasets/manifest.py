"""Immutable scientific manifests for Gaussian reference datasets.

A manifest stores the actual Fourier coefficients, not just the RNG seeds.
All strength variants of an initial shape have the same data split. Hardware,
paths, job IDs and source provenance belong in run records, not this manifest.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from numbers import Real
import os
from pathlib import Path
import tempfile

import numpy as np

from .initial_conditions import (
    _finite_real, _integer, draw_coefficients, validate_amplitudes,
)

SCHEMA_VERSION = 1
OUTPUT_SCHEMA = "gaussian-fields-v1"
_SPLITS = ("train", "validation", "test")
_MANIFEST_KEYS = {"schema_version", "output_schema", "numerics", "amplitudes", "cases"}
_CASE_KEYS = {"case_id", "shape_id", "split", "k0", "mode_count", "decay", "seed", "strength", "coefficients"}


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _integral_ratio(numerator: float, denominator: float, name: str) -> int:
    ratio = numerator / denominator
    if not np.isfinite(ratio) or not np.isclose(ratio, round(ratio), atol=1e-9, rtol=0):
        raise ValueError(f"{name} must be an integer")
    return int(round(ratio))


@dataclass(frozen=True)
class Numerics:
    """Numerical protocol; output values use linear interpolation in time.

    save_dt need not be an integer multiple of dt. The final time must be a
    multiple of both intervals, so the initial and final samples are explicit.
    """

    nx: int = 1024
    nv: int = 8192
    dt: float = 0.01
    final_time: float = 120.0
    vmin: float = -8.0
    vmax: float = 8.0
    output_nx: int = 128
    save_dt: float = 0.025
    block_steps: int = 100
    storage_dtype: str = "float32"

    def __post_init__(self) -> None:
        for name in ("nx", "nv", "output_nx"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=8))
        object.__setattr__(self, "block_steps", _integer(self.block_steps, "block_steps"))
        if self.nx % 2 or self.output_nx % 2 or self.output_nx > self.nx:
            raise ValueError("nx and output_nx must be even, with output_nx <= nx")
        for name in ("dt", "final_time", "save_dt"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name, positive=True))
        for name in ("vmin", "vmax"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.vmin >= self.vmax:
            raise ValueError("vmin must be less than vmax")
        if self.save_dt < self.dt or self.save_dt > self.final_time:
            raise ValueError("save_dt must lie between dt and final_time")
        _integral_ratio(self.final_time, self.dt, "final_time/dt")
        _integral_ratio(self.final_time, self.save_dt, "final_time/save_dt")
        if self.storage_dtype not in ("float32", "float64"):
            raise ValueError("storage_dtype must be float32 or float64")

    @property
    def nsteps(self) -> int:
        return _integral_ratio(self.final_time, self.dt, "final_time/dt")

    @property
    def nsamples(self) -> int:
        return _integral_ratio(self.final_time, self.save_dt, "final_time/save_dt") + 1

    def to_dict(self) -> dict:
        return asdict(self)


def _require_keys(value: Mapping, expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly {', '.join(sorted(expected))}")


def _numerics(value: Numerics | Mapping | None) -> Numerics:
    if value is None:
        return Numerics()
    if isinstance(value, Numerics):
        return value
    _require_keys(value, {field.name for field in fields(Numerics)}, "numerics")
    return Numerics(**value)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _shape_id(case: Mapping, amplitudes: Mapping) -> str:
    content = {name: case[name] for name in ("k0", "mode_count", "decay", "seed", "coefficients")}
    content["amplitudes"] = dict(amplitudes)
    return "shape_" + hashlib.sha256(_canonical(content)).hexdigest()[:24]


def _case_id(shape_id: str, strength: float) -> str:
    return "case_" + hashlib.sha256(_canonical({"shape_id": shape_id, "strength": strength})).hexdigest()[:24]


def _choices(values: Sequence, name: str, convert) -> list:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a nonempty sequence")
    result = [convert(value, name) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def build_manifest(
    *,
    k0_values: Sequence[float] = (0.3, 0.4, 0.5),
    mode_counts: Sequence[int] = (3, 5),
    decays: Sequence[float] = (1.2, 2.0),
    draws_per_combination: int = 20,
    strengths: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 1.0),
    base_seed: int = 28001,
    numerics: Numerics | Mapping | None = None,
    amplitudes: Mapping[str, float] | None = None,
    split_counts: Sequence[int] | None = None,
) -> dict:
    """Build a deterministic manifest; small argument sets support local tests.

    Default: 240 shapes and 1,200 cases. Each parameter combination contributes
    16 training, two validation and two test shapes, giving 192/24/24 shapes.
    For other draw counts, each held-out split receives floor(draws/10) shapes.
    Explicit split_counts=(train, validation, test) must sum to the draw count.
    """
    numerical = _numerics(numerics)
    amplitudes = validate_amplitudes(amplitudes)
    k0_values = _choices(k0_values, "k0_values", lambda v, n: _finite_real(v, n, positive=True))
    mode_counts = _choices(mode_counts, "mode_counts", _integer)
    decays = _choices(decays, "decays", lambda v, n: _finite_real(v, n, positive=True))
    strengths = _choices(strengths, "strengths", _finite_real)
    draws = _integer(draws_per_combination, "draws_per_combination")
    base_seed = _integer(base_seed, "base_seed", minimum=0)
    if split_counts is None:
        held_out = draws // 10
        split_counts = (draws - 2 * held_out, held_out, held_out)
    if (not isinstance(split_counts, Sequence) or isinstance(split_counts, (str, bytes))
            or len(split_counts) != 3):
        raise ValueError("split_counts must contain train, validation and test counts")
    split_counts = tuple(_integer(count, "split count", minimum=0) for count in split_counts)
    if sum(split_counts) != draws:
        raise ValueError("split_counts must sum to draws_per_combination")
    labels = [label for label, count in zip(_SPLITS, split_counts) for _ in range(count)]
    cases = []
    shape_index = 0
    for k0 in k0_values:
        for modes in mode_counts:
            if 2 * modes >= numerical.nx:
                raise ValueError("Every initial Fourier mode must lie below the spatial Nyquist frequency")
            for decay in decays:
                for split in labels:
                    seed = base_seed + shape_index
                    shape_index += 1
                    base = dict(k0=k0, mode_count=modes, decay=decay, seed=seed,
                                coefficients=draw_coefficients(seed, modes, decay).tolist())
                    shape_id = _shape_id(base, amplitudes)
                    for strength in strengths:
                        cases.append(dict(base, shape_id=shape_id, split=split, strength=strength,
                                          case_id=_case_id(shape_id, strength)))
    manifest = dict(schema_version=SCHEMA_VERSION, output_schema=OUTPUT_SCHEMA,
                    numerics=numerical.to_dict(), amplitudes=amplitudes, cases=cases)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping) -> None:
    """Reject incompatible, incomplete or ambiguous manifests without mutation."""
    _require_keys(manifest, _MANIFEST_KEYS, "manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema_version")
    if manifest["output_schema"] != OUTPUT_SCHEMA:
        raise ValueError("Unsupported output_schema")
    numerical = _numerics(manifest["numerics"])
    amplitudes = validate_amplitudes(manifest["amplitudes"])
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    identifiers = set()
    shapes: dict[str, tuple[str, set[float]]] = {}
    for case in cases:
        _require_keys(case, _CASE_KEYS, "case")
        _finite_real(case["k0"], "k0", positive=True)
        modes = _integer(case["mode_count"], "mode_count")
        _finite_real(case["decay"], "decay", positive=True)
        _integer(case["seed"], "seed", minimum=0)
        strength = _finite_real(case["strength"], "strength")
        if 2 * modes >= numerical.nx:
            raise ValueError("Initial Fourier modes must lie below the spatial Nyquist frequency")
        if case["split"] not in _SPLITS:
            raise ValueError("split must be train, validation or test")
        raw_coefficients = case["coefficients"]
        if (not isinstance(raw_coefficients, list) or len(raw_coefficients) != 3
                or any(not isinstance(field, list) or len(field) != 2 for field in raw_coefficients)
                or any(not isinstance(part, list) or len(part) != modes
                       for field in raw_coefficients for part in field)):
            raise ValueError("coefficients must be nested JSON lists with shape (3, 2, mode_count)")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               for field in raw_coefficients for part in field for value in part):
            raise ValueError("Every Fourier coefficient must be a real number")
        try:
            coefficients = np.asarray(raw_coefficients, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("coefficients must be a numeric array") from exc
        if coefficients.shape != (3, 2, modes) or not np.isfinite(coefficients).all():
            raise ValueError("coefficients must have finite shape (3, 2, mode_count)")
        if not np.allclose(0.5 * np.sum(coefficients ** 2, axis=(1, 2)), 1, rtol=1e-12, atol=1e-12):
            raise ValueError("Each Fourier shape must have unit spatial RMS")
        shape_id = _shape_id(case, amplitudes)
        if case["shape_id"] != shape_id or case["case_id"] != _case_id(shape_id, strength):
            raise ValueError("Case or shape identity does not match its parameters and coefficients")
        if case["case_id"] in identifiers:
            raise ValueError("Duplicate case_id")
        identifiers.add(case["case_id"])
        if shape_id not in shapes:
            shapes[shape_id] = (case["split"], set())
        split, seen_strengths = shapes[shape_id]
        if case["split"] != split:
            raise ValueError("All strengths of a shape must share one split")
        if strength in seen_strengths:
            raise ValueError("Duplicate strength for an initial shape")
        seen_strengths.add(strength)
    if len({tuple(sorted(values)) for _, values in shapes.values()}) != 1:
        raise ValueError("Every shape must have the same strength values")
    # Ensure all metadata can be recorded as strict JSON (without NaN or objects).
    try:
        _canonical(manifest)
    except (ValueError, TypeError) as exc:
        raise ValueError("Manifest must contain only finite JSON values") from exc


def write_manifest(path: str | Path, manifest: Mapping) -> Path:
    """Publish a validated manifest atomically; never replace an existing path."""
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(manifest) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Linking is atomic and exclusive, unlike os.replace, which overwrites.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)
    return path


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _invalid_constant(value: str):
    raise ValueError(f"Nonfinite JSON value: {value}")


def read_manifest(path: str | Path) -> dict:
    """Load strict JSON and validate before any worker can consume it."""
    manifest = json.loads(Path(path).read_text(), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    validate_manifest(manifest)
    return manifest
