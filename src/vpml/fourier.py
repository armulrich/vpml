"""Shared restriction of unnormalized real-FFT coefficients.

NumPy and JAX callers use the same scaling and Nyquist convention. Restriction
acts on the last array axis; all preceding axes are batch or field axes.
"""
from __future__ import annotations

import numpy as np


def _restrict_coefficients(values, source_nx: int, target_nx: int, namespace):
    source_nx = int(source_nx)
    target_nx = int(target_nx)
    if not 1 < target_nx <= source_nx:
        raise ValueError("Require 1 < target_nx <= source_nx")
    target_nk = target_nx // 2 + 1
    if values.ndim == 0 or values.shape[-1] < target_nk:
        raise ValueError(f"Coefficient width cannot represent target_nx={target_nx}")
    coefficients = values[..., :target_nk]
    if source_nx == target_nx:
        return coefficients
    coefficients = coefficients * (float(target_nx) / float(source_nx))
    if target_nx % 2 == 0:
        # The target Nyquist bin represents both source frequencies +k and -k.
        nyquist = 2.0 * namespace.real(coefficients[..., -1:])
        coefficients = namespace.concatenate((coefficients[..., :-1], nyquist), axis=-1)
    return coefficients


def restrict_rfft_numpy(coefficients, *, source_nx: int, target_nx: int) -> np.ndarray:
    """Restrict NumPy rFFT coefficients, preserving source normalization.

    An equal-grid call returns a view cropped to the required coefficient
    width. Callers requiring independent storage can explicitly copy it.
    """
    return _restrict_coefficients(np.asarray(coefficients), source_nx, target_nx, np)


def restrict_rfft_jax(coefficients, *, source_nx: int, target_nx: int):
    """JAX equivalent; grid sizes are static when called from a JIT kernel."""
    import jax.numpy as jnp

    return _restrict_coefficients(jnp.asarray(coefficients), source_nx, target_nx, jnp)
