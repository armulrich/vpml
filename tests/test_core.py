"""Unit tests for vpml.core."""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from vpml.core import (
    FourierHermiteIMEX,
    HermiteExponentialFilter,
    HouLiFilter,
    HyperCollisions,
    NonlocalClosure,
    hermite_damping_term,
    irfft_x,
    rfft_x,
    tridiag_solve,
)


# =============================================================================
# FFT 
# =============================================================================


def test_rfft_irfft_roundtrip() -> None:
    """rfft_x and irfft_x should be inverses."""
    Nv, Nx = 4, 16 # course grid
    a_phys = jnp.ones((Nv, Nx), dtype=jnp.float64) * 0.5
    a_hat = rfft_x(a_phys)
    a_back = irfft_x(a_hat, Nx)
    np.testing.assert_allclose(np.array(a_phys), np.array(a_back), rtol=1e-10)


def test_rfft_irfft_preserves_shape() -> None:
    """Preserves (Nv, Nk) -> (Nv, Nx)."""
    Nv, Nx = 8, 32 # course grid
    a_phys = jnp.arange(Nv * Nx, dtype=jnp.float64).reshape(Nv, Nx)
    a_hat = rfft_x(a_phys)
    assert a_hat.shape == (Nv, Nx // 2 + 1)
    a_back = irfft_x(a_hat, Nx)
    assert a_back.shape == (Nv, Nx)


# =============================================================================
# Tridiagonal solver
# =============================================================================


def test_tridiag_solve_simple() -> None:
    """Solve a 3x3 tridiagonal system A x = b."""
    # [[4, 1, 0], [1, 3, 1], [0, 1, 2]] x = [1, 2, 1]
    sub = jnp.array([1.0, 1.0], dtype=jnp.complex128)
    diag = jnp.array([4.0, 3.0, 2.0], dtype=jnp.complex128)
    sup = jnp.array([1.0, 1.0], dtype=jnp.complex128)
    rhs = jnp.array([1.0, 2.0, 1.0], dtype=jnp.complex128)
    x = tridiag_solve(sub, diag, sup, rhs)
    Ax = jnp.array([
        4 * x[0] + x[1],
        x[0] + 3 * x[1] + x[2],
        x[1] + 2 * x[2],
    ])
    np.testing.assert_allclose(np.array(Ax), np.array(rhs), rtol=1e-10)


# =============================================================================
# Damping / filter models
# =============================================================================



def test_hou_li_filter_damping_rates() -> None:
    """HouLiFilter.damping_rates are non-negative."""
    filt = HouLiFilter(chi_over_dt=1.0, Nv=12, p=36)
    gamma = np.array(filt.damping_rates())
    assert gamma.shape == (12,)
    assert np.all(gamma >= 0)
    assert gamma[0] == 0
    assert gamma[-1] > 0


def test_hermite_exponential_filter_factors() -> None:
    """HermiteExponentialFilter factors are in (0, 1] and decreasing."""
    filt = HermiteExponentialFilter(alpha=8.0, Nv=20, p=8)
    fac = np.array(filt.factors())
    assert fac.shape == (20,)
    assert np.all(fac > 0) # right range
    assert np.all(fac <= 1) 
    assert np.all(np.diff(fac) <= 0)  # monotonically decreasing



# =============================================================================
# FourierHermiteIMEX
# =============================================================================


def test_fourier_hermite_imex_invalid_params() -> None:
    """FourierHermiteIMEX rejects invalid Nx, Nv, vth."""
    with pytest.raises(ValueError):
        FourierHermiteIMEX(Nx=0, Nv=8, Lx=2 * np.pi, dt=0.1)
    with pytest.raises(ValueError):
        FourierHermiteIMEX(Nx=32, Nv=-1, Lx=2 * np.pi, dt=0.1)
    with pytest.raises(ValueError):
        FourierHermiteIMEX(Nx=32, Nv=8, Lx=2 * np.pi, dt=0.1, vth=0)


def test_E_phys_from_a_hat_zero_density() -> None:
    """E should be zero when density (a0) is zero."""
    integ = FourierHermiteIMEX(Nx=32, Nv=6, Lx=2 * np.pi, dt=0.1)
    a_hat = jnp.zeros((6, 17), dtype=jnp.complex128)
    a_hat = a_hat.at[0, 0].set(1.0)  # constant mode -> zero gradient
    E = integ.E_phys_from_a_hat(a_hat)
    np.testing.assert_allclose(np.array(E), 0.0, atol=1e-12)


def test_E_phys_from_a_hat_cosine_density() -> None:
    """E from rho = cos(kx) should be -sin(kx)/k (for poisson_sign=+1)."""
    Nx = 64
    integ = FourierHermiteIMEX(Nx=Nx, Nv=4, Lx=2 * np.pi, dt=0.1)
    # a0 = cos(x) -> rho = cos(x), k=1
    x = np.array(integ.x)
    rho = np.cos(x)
    rho_hat = np.fft.rfft(rho).astype(np.complex128)
    a_hat = jnp.zeros((4, Nx // 2 + 1), dtype=jnp.complex128)
    a_hat = a_hat.at[0].set(jnp.array(rho_hat))
    E = np.array(integ.E_phys_from_a_hat(a_hat))
    # i k E_k = rho_k  -> E_k = -i rho_k/k  -> E(x) = -sin(x) for rho=cos(x)
    E_expected = -np.sin(x)
    np.testing.assert_allclose(E, E_expected, rtol=1e-5, atol=1e-5)


def test_streaming_hat_preserves_constant() -> None:
    """Streaming of a constant in x (only n=0) should give zero."""
    integ = FourierHermiteIMEX(Nx=32, Nv=6, Lx=2 * np.pi, dt=0.1)
    a_hat = jnp.zeros((6, 17), dtype=jnp.complex128)
    a_hat = a_hat.at[0, 0].set(1.0)  # constant density
    La = integ.streaming_hat(a_hat)
    np.testing.assert_allclose(np.array(La), 0.0, atol=1e-12)


def test_electric_energy_nonnegative() -> None:
    """Electric energy should be non-negative."""
    integ = FourierHermiteIMEX(Nx=32, Nv=6, Lx=2 * np.pi, dt=0.1)
    a_hat = jnp.zeros((6, 17), dtype=jnp.complex128)
    a_hat = a_hat.at[0, 1].set(0.1 + 0j)  # one Fourier mode
    en = integ.electric_energy(a_hat)
    assert float(en) >= 0


# =============================================================================
# NonlocalClosure
# =============================================================================


