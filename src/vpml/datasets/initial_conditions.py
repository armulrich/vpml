"""Smooth Gaussian initial conditions, independent of solver and compute backend.

Coefficients are ordered as density, velocity, temperature; their second axis
contains cosine then sine coefficients. Spatial coordinates are x/L in [0, 1),
so the fundamental wavenumber determines L rather than the normalized shapes.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

DEFAULT_AMPLITUDES = {"density": 0.18, "velocity": 0.25, "temperature": 0.10}


def _integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_real(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if value < 0 or (positive and value == 0):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {relation}")
    return float(value)


def validate_amplitudes(amplitudes: Mapping[str, float] | None = None) -> dict[str, float]:
    """Return an independent copy of the three finite, nonnegative amplitudes."""
    values = DEFAULT_AMPLITUDES if amplitudes is None else amplitudes
    if not isinstance(values, Mapping) or set(values) != set(DEFAULT_AMPLITUDES):
        raise ValueError("amplitudes must contain exactly density, velocity, temperature")
    return {name: _finite_real(values[name], f"amplitude {name}") for name in DEFAULT_AMPLITUDES}


def draw_coefficients(seed: int, modes: int, decay: float) -> np.ndarray:
    """Draw independent Fourier shapes and normalize each to unit spatial RMS.

    Normal coefficients are weighted by j**(-decay) before normalization.
    The returned float64 array has shape (3, 2, modes).
    """
    seed = _integer(seed, "seed", minimum=0)
    modes = _integer(modes, "modes")
    decay = _finite_real(decay, "decay", positive=True)
    rng = np.random.default_rng(seed)
    coefficients = rng.standard_normal((3, 2, modes)) * np.arange(1, modes + 1)[None, None, :] ** (-decay)
    norms = np.sqrt(0.5 * np.sum(coefficients ** 2, axis=(1, 2)))
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Fourier coefficients have an invalid RMS")
    coefficients /= norms[:, None, None]
    return coefficients


def fields_from_coefficients(
    coefficients: np.ndarray,
    nx: int,
    strength: float,
    amplitudes: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate positive density and temperature and zero-total-momentum velocity.

    Density and temperature each have spatial mean one. The velocity is centered
    with the density weight, so mean(n*u)=0. No clipping or implicit resampling
    is applied. Unrepresentable exponential profiles are rejected.
    """
    nx = _integer(nx, "nx", minimum=2)
    strength = _finite_real(strength, "strength")
    amplitudes = validate_amplitudes(amplitudes)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 3 or coefficients.shape[:2] != (3, 2) or coefficients.shape[2] < 1:
        raise ValueError("coefficients must have shape (3, 2, modes), with modes >= 1")
    if not np.isfinite(coefficients).all():
        raise ValueError("coefficients must be finite")
    modes = coefficients.shape[2]
    if 2 * modes >= nx:
        raise ValueError("nx must resolve every Fourier mode below its Nyquist frequency")
    theta = 2 * np.pi * np.arange(nx) / nx
    phase = np.arange(1, modes + 1)[:, None] * theta[None, :]
    shapes = coefficients[:, 0] @ np.cos(phase) + coefficients[:, 1] @ np.sin(phase)
    gn, gu, gt = shapes
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            density = np.exp(strength * amplitudes["density"] * gn)
            density /= density.mean()
            temperature = np.exp(strength * amplitudes["temperature"] * gt)
            temperature /= temperature.mean()
            velocity = strength * amplitudes["velocity"] * (gu - np.mean(density * gu) / density.mean())
        except FloatingPointError as exc:
            raise ValueError("Initial fields cannot be represented in float64") from exc
    if (not all(np.isfinite(field).all() for field in (density, velocity, temperature))
            or np.any(density <= 0) or np.any(temperature <= 0)):
        raise ValueError("Initial fields must be finite with positive density and temperature")
    return density, velocity, temperature


def gaussian_distribution(
    n: np.ndarray,
    u: np.ndarray,
    temperature: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Evaluate f0(v,x) from fields; finite-velocity truncation is not renormalized."""
    n, u, temperature, v = (np.asarray(a, dtype=np.float64) for a in (n, u, temperature, v))
    if n.ndim != 1 or len(n) == 0 or u.shape != n.shape or temperature.shape != n.shape:
        raise ValueError("n, u and temperature must be equally sized nonempty 1D fields")
    if v.ndim != 1 or len(v) < 2 or not np.all(np.diff(v) > 0):
        raise ValueError("v must be a strictly increasing 1D grid with at least two points")
    if not all(np.isfinite(a).all() for a in (n, u, temperature, v)):
        raise ValueError("Fields and velocity grid must be finite")
    if np.any(n <= 0) or np.any(temperature <= 0):
        raise ValueError("Density and temperature must be positive")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            result = n[None, :] / np.sqrt(2 * np.pi * temperature[None, :]) * np.exp(
                -(v[:, None] - u[None, :]) ** 2 / (2 * temperature[None, :])
            )
        except FloatingPointError as exc:
            raise ValueError("Gaussian distribution cannot be represented in float64") from exc
    if not np.isfinite(result).all():
        raise ValueError("Gaussian distribution must be finite")
    return result
