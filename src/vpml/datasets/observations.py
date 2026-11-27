"""Moment definitions and Fourier restriction for Gaussian reference data.

The kinetic state has axes ``(velocity, position)``. Saved observations have
axes ``(field, position)`` and histories prepend a time axis. All quantities
use the nondimensional units of :mod:`vpml.physical_grid`.

Primitive and central moments are computed on the kinetic spatial grid before
restriction. In particular, filtering raw moments and then dividing them does
not reproduce the saved primitive fields. Raw moments are stored separately,
without the equilibrium subtraction used by the older ``states.npy`` format.
"""

from __future__ import annotations

from vpml._jax_precision import enable_x64

enable_x64()

import jax.numpy as jnp

from vpml.fourier import restrict_rfft_jax

from .schema import FIELD_NAMES, RAW_MOMENT_NAMES, OUTPUT_SCHEMA, observation_schema


def restrict_fourier(values, output_nx: int):
    """Low-pass restrict real periodic values along their last axis.

    For an even output grid its Nyquist coefficient combines the positive and
    negative source frequencies. Merely cropping an rFFT would halve a cosine
    exactly at that output Nyquist frequency.
    """
    values = jnp.asarray(values, dtype=jnp.float64)
    source_nx = values.shape[-1]
    if isinstance(output_nx, bool) or not isinstance(output_nx, int):
        raise ValueError("output_nx must be an integer")
    if output_nx < 2 or output_nx % 2 or output_nx > source_nx:
        raise ValueError("output_nx must be even and between 2 and the input size")
    if output_nx == source_nx:
        return values
    coefficients = restrict_rfft_jax(
        jnp.fft.rfft(values, axis=-1), source_nx=source_nx, target_nx=output_nx,
    )
    return jnp.fft.irfft(coefficients, n=output_nx, axis=-1)


def raw_to_primitives(raw_moments):
    """Convert full ``(M0,M1,M2)`` to ``(n,u,p)`` on the same grid.

    ``M_j = integral v**j f dv`` and ``p = M2 - M1**2/M0``. Density is not
    clipped: a nonpositive density is a numerical failure for the caller to report.
    """
    raw = jnp.asarray(raw_moments, dtype=jnp.float64)
    if raw.ndim < 2 or raw.shape[-2] != 3:
        raise ValueError("raw moments must have shape (..., 3, nx)")
    density, momentum, second = (raw[..., i, :] for i in range(3))
    velocity = momentum / density
    pressure = second - momentum * velocity
    return jnp.stack((density, velocity, pressure), axis=-2)


def moment_fields(distribution, velocity_grid, domain_length: float):
    """Return primitive/central fields and full raw moments on the fine grid.

    ``p = integral (v-u)**2 f dv`` and ``Q = integral (v-u)**3 f dv``;
    there is no extra factor of one half in either definition. ``dQ_dx`` uses
    the Fourier derivative in physical spatial coordinates.
    """
    f = jnp.asarray(distribution, dtype=jnp.float64)
    v = jnp.asarray(velocity_grid, dtype=jnp.float64)
    if f.ndim != 2 or v.ndim != 1 or f.shape[0] != v.shape[0]:
        raise ValueError("distribution must have shape (len(velocity_grid), nx)")
    if domain_length <= 0:
        raise ValueError("domain_length must be positive")
    density = jnp.trapezoid(f, x=v, axis=0)
    momentum = jnp.trapezoid(v[:, None] * f, x=v, axis=0)
    second = jnp.trapezoid(v[:, None] ** 2 * f, x=v, axis=0)
    flow = momentum / density
    central_velocity = v[:, None] - flow[None, :]
    pressure = jnp.trapezoid(central_velocity ** 2 * f, x=v, axis=0)
    heat_flux = jnp.trapezoid(central_velocity ** 3 * f, x=v, axis=0)
    nx = f.shape[1]
    wave_numbers = 2 * jnp.pi * jnp.fft.rfftfreq(nx, d=domain_length / nx)
    derivative = jnp.fft.irfft(1j * wave_numbers * jnp.fft.rfft(heat_flux), n=nx)
    fields = jnp.stack((density, flow, pressure, heat_flux, derivative))
    raw = jnp.stack((density, momentum, second))
    return fields, raw
