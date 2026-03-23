"""Unit tests for nonlinear simulation (two-stream, bump-on-tail)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.fh_nonlinear_sim_jax import (
    BumpOnTailParams,
    TwoStreamParams,
    equilibrium_coeffs_two_stream,
    simulate_bump_on_tail,
    simulate_two_stream,
)


# =============================================================================
# Equilibrium coefficients
# =============================================================================


def test_equilibrium_coeffs_two_stream_shape() -> None:
    """Two-stream equilibrium has shape (Nv,) and even modes only."""
    m = np.array(equilibrium_coeffs_two_stream(10, 2.4))
    assert m.shape == (10,)
    assert m[0] > 0
    assert m[1] == 0
    assert np.all(m[1::2] == 0)  # odd modes are zero
    assert np.all(m[::2] >= 0)


def test_equilibrium_coeffs_two_stream_normalization() -> None:
    """Two-stream m_0 gives unit density."""
    m = np.array(equilibrium_coeffs_two_stream(20, 2.4))
    # m_0 is the n=0 coeff; for normalized basis, integral of phi_0 is 1, so rho = m_0
    assert np.isclose(m[0], 1.0, rtol=1e-5)


# =============================================================================
# Two-stream simulation
# =============================================================================


def test_simulate_two_stream_returns_valid_shapes() -> None:
    """simulate_two_stream returns correct shapes."""
    p = TwoStreamParams()
    x, v, snaps, times, energy = simulate_two_stream(p)
    assert x.shape == (p.Nx,)
    assert v.shape == (p.Nv,)
    assert len(snaps) == len(p.snapshot_times)
    assert times.shape[0] == int(p.T / p.dt) + 1
    assert energy.shape == times.shape
    assert np.all(energy >= 0)
    for t, arr in snaps.items():
        assert arr.shape == (p.Nv, p.Nx)


def test_simulate_two_stream_energy_dynamics() -> None:
    """Electric energy should evolve for two-stream instability ."""
    p = TwoStreamParams()
    _, _, _, times, energy = simulate_two_stream(p)
    # Energy should change over time (not static)
    assert np.std(energy) > 1e-10
    assert np.all(energy >= 0)


# =============================================================================
# Bump-on-tail simulation
# =============================================================================


def test_simulate_bump_on_tail_system_a_returns_valid_shapes() -> None:
    """simulate_bump_on_tail(system='A') returns correct shapes (paper params)."""
    p = BumpOnTailParams()
    x, m, snaps, times, energy = simulate_bump_on_tail(p, system="A")
    assert x.shape == (p.Nx,)
    assert m.shape == (p.Nv,)
    assert len(snaps) == len(p.snapshot_times)
    assert energy.shape == times.shape
    assert np.all(energy >= 0)


