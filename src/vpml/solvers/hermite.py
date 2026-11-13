"""Fourier-Hermite operators and the CNAB2 integrator for periodic 1D1V systems.

States have shape (Hermite mode, nonnegative Fourier mode). The zeroth
Hermite coefficient is density; equilibrium subtraction is the caller's
responsibility. Poisson uses E_k = poisson_sign * i * rho_k / k for k != 0
and E_0 = 0. Analytical closure parameters live in models.hermite_closures;
learned models and checkpoint I/O do not belong to this solver module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import numpy as np

from .._jax_precision import enable_x64

enable_x64()

import jax
import jax.numpy as jnp

from ..models.hermite_closures import (
    NonlocalClosure,
)

from .implicit import (
    tridiag_solve,
)

Array = jnp.ndarray


def e_hat_from_rho_hat(
    rho_hat: Array,
    k_arr: Array,
    *,
    poisson_sign: float = +1.0,
) -> Array:
    """Compute electric-field Fourier coefficients from density coefficients."""
    rho_hat = jnp.asarray(rho_hat, dtype=jnp.complex128)
    k_arr = jnp.asarray(k_arr, dtype=jnp.float64)
    if rho_hat.ndim != 1:
        raise ValueError(f"rho_hat must have shape (Nk,), got {rho_hat.shape}")
    if k_arr.ndim != 1 or k_arr.shape[0] != rho_hat.shape[0]:
        raise ValueError(f"k_arr must have shape ({rho_hat.shape[0]},), got {k_arr.shape}")

    E_hat = jnp.zeros_like(rho_hat, dtype=jnp.complex128)
    return E_hat.at[1:].set((float(poisson_sign) * 1j) * rho_hat[1:] / k_arr[1:])


def e_hat_history_from_a_hat_history(
    a_hat_hist: Array,
    k_arr: Array,
    *,
    poisson_sign: float = +1.0,
) -> Array:
    """Convert an ``a_hat`` history of shape ``(Nt, Nv, Nk)`` into ``E_hat(t,k)``."""
    a_hat_hist = jnp.asarray(a_hat_hist, dtype=jnp.complex128)
    if a_hat_hist.ndim != 3:
        raise ValueError(f"a_hat_hist must have shape (Nt, Nv, Nk), got {a_hat_hist.shape}")
    return jax.vmap(
        lambda a_hat: e_hat_from_rho_hat(a_hat[0], k_arr, poisson_sign=poisson_sign)
    )(a_hat_hist)


def rfft_x(u_phys: Array) -> Array:
    """Real FFT over x axis=1, returning complex128."""
    return jnp.fft.rfft(u_phys, axis=1).astype(jnp.complex128)


def irfft_x(u_hat: Array, Nx: int) -> Array:
    """Inverse real FFT over x axis=1, returning float64."""
    return jnp.fft.irfft(u_hat, n=Nx, axis=1).astype(jnp.float64)


def hermite_basis_phi_scaled(N: int, v: np.ndarray, vth: float = 1.0) -> np.ndarray:
    """
    Build the scaled AW Hermite basis φ_n^{(vth)}(v) on a velocity grid.

    The recurrence matches the visualization helper used by the nonlinear JAX demos:

        φ_n^{(vth)}(v) = (1 / vth) φ_n(v / vth)

    where φ_0 is the unit-mass Gaussian.
    """
    if N < 0:
        raise ValueError("N must be nonnegative")
    if vth <= 0.0:
        raise ValueError("vth must be positive")

    v = np.asarray(v, dtype=float)
    if N == 0:
        return np.zeros((0, v.size), dtype=float)

    xi = v / float(vth)
    w = np.exp(-0.5 * xi ** 2) / (math.sqrt(2.0 * math.pi) * float(vth))

    h = np.zeros((N, v.size), dtype=float)
    h[0] = 1.0
    if N > 1:
        h[1] = xi
    for n in range(1, N - 1):
        h[n + 1] = (xi / math.sqrt(n + 1)) * h[n] - math.sqrt(n / (n + 1)) * h[n - 1]
    return w * h


def hermite_basis_phi(N: int, v: np.ndarray) -> np.ndarray:
    """Build the unscaled AW Hermite basis φ_n(v) on a velocity grid."""
    return hermite_basis_phi_scaled(N, v, vth=1.0)


def _mask_23(Nx: int) -> Array:
    """
    2/3 dealias mask for an rFFT array of length Nk=Nx//2+1.

    We keep Fourier indices j <= floor(Nx/3), zero out j > floor(Nx/3).
    """
    Nk = Nx // 2 + 1
    cutoff = Nx // 3
    mask = jnp.ones((Nk,), dtype=bool)
    mask = mask.at[cutoff + 1 :].set(False)
    return mask


@dataclass
class FourierHermiteIMEX:
    """
    IMEX CNAB2 integrator for systems of the form:

        d/dt a_hat = L(a_hat) + N(a_hat, t)

    where:
      - a_hat has shape (Nv, Nk) with Nk = Nx//2+1 (rFFT in x).
      - L is the linear streaming operator, tridiagonal in Hermite index.
      - N is an explicit term (usually electric-field acceleration + optional damping).

    The CNAB2 update is:

        (I - dt/2 L) a^{n+1} = (I + dt/2 L) a^n + dt*( 3/2 N^n - 1/2 N^{n-1} )

    This class provides:
      - precomputed per-k tridiagonal matrices for the implicit solve
      - batched Thomas solve via jax.vmap(tridiag_solve)
      - utilities for Poisson solve E from density a0

    Parameters
    ----------
    Nx : int
        Number of x grid points.
    Nv : int
        Number of Hermite modes.
    Lx : float
        Domain length in x.
    dt : float
        Time step size.
    vth : float
        Hermite thermal scaling. vth=1 recovers the unscaled basis.
        Streaming operator scales by vth and acceleration term scales by 1/vth.
    dealias_23 : bool
        If True, apply 2/3 dealiasing mask to *all* Fourier-space arrays passed
        through `apply_mask_hat`.
    closure : Optional[NonlocalClosure]
        If provided with Nm=1, modifies the last Hermite mode via the closure substitution.
        (Time-stepping supports Nm=1 only.)
    """
    Nx: int
    Nv: int
    Lx: float
    dt: float
    vth: float = 1.0
    dealias_23: bool = True
    closure: Optional[NonlocalClosure] = None
    real_dtype: object = jnp.float64
    complex_dtype: object = jnp.complex128

    def __post_init__(self):
        if self.Nx <= 0 or self.Nv <= 0:
            raise ValueError("Nx and Nv must be positive")
        if self.vth <= 0:
            raise ValueError("vth must be positive")
        self.real_dtype = jnp.dtype(self.real_dtype)
        self.complex_dtype = jnp.dtype(self.complex_dtype)

        dx = float(self.Lx) / float(self.Nx)
        self.dx = dx

        self.x = jnp.linspace(0.0, float(self.Lx), int(self.Nx), endpoint=False)
        self.k_arr = (2.0 * jnp.pi * jnp.fft.rfftfreq(int(self.Nx), d=dx)).astype(self.real_dtype)
        self.Nk = int(self.Nx // 2 + 1)

        self.sqrt_n = jnp.sqrt(jnp.arange(int(self.Nv), dtype=self.real_dtype))
        self.sqrt_np1 = jnp.sqrt(jnp.arange(1, int(self.Nv) + 1, dtype=self.real_dtype))

        # 2/3 dealias mask (rFFT length Nk)
        self.mask = _mask_23(self.Nx) if self.dealias_23 else None

        # --- Precompute tridiagonal coefficients for the implicit CN solve ---
        dt_half = 0.5 * float(self.dt)
        s = jnp.sqrt(jnp.arange(1, int(self.Nv), dtype=self.real_dtype))              # sqrt(1..Nv-1)
        val = (1j * dt_half * self.k_arr * float(self.vth)).astype(self.complex_dtype)  # (Nk,)

        # sub/sup: (Nk, Nv-1)
        self.sub = (val[:, None] * s[None, :]).astype(self.complex_dtype)
        self.sup = self.sub.copy()
        self.diag = jnp.ones((self.Nk, int(self.Nv)), dtype=self.complex_dtype)

        # Optional Nm=1 closure -> diagonal modification on last Hermite mode
        self._closure_Ldiag_last = None
        if self.closure is not None:
            Nm = int(self.closure.Nm)
            if Nm != 1:
                raise ValueError("Time-stepping integrator supports NonlocalClosure only for Nm=1.")
            mu = jnp.asarray(self.closure.mu_tail[0], dtype=self.real_dtype)
            signk = jnp.sign(self.k_arr)
            # L_diag_last = k*vth*sqrt(Nv)*mu*sign(k)
            Ldiag_last = (self.k_arr * float(self.vth) * jnp.sqrt(float(self.Nv)) * mu * signk)
            self._closure_Ldiag_last = Ldiag_last.astype(self.complex_dtype)
            self.diag = self.diag.at[:, int(self.Nv) - 1].set(1.0 - dt_half * self._closure_Ldiag_last)

        # Batched solver (vmap over k)
        self._solve_batched = jax.vmap(tridiag_solve, in_axes=(0, 0, 0, 0), out_axes=0)

    # -------------------------
    # Fourier-space utilities
    # -------------------------

    def apply_mask_hat(self, a_hat: Array) -> Array:
        """Apply the 2/3 dealias mask (if enabled) to an array shaped (..., Nk)."""
        if self.mask is None:
            return a_hat
        # broadcast mask onto leading axes
        return jnp.where(self.mask, a_hat, 0.0)

    # -------------------------
    # Linear streaming operator
    # -------------------------

    def streaming_hat(self, a_hat: Array) -> Array:
        """
        Streaming operator in Fourier-x / Hermite-v space:

            (L a)_n = -i k vth ( sqrt(n) a_{n-1} + sqrt(n+1) a_{n+1} )

        plus optional Nm=1 nonlocal closure contribution on the last mode.
        """
        a_down = jnp.concatenate([jnp.zeros_like(a_hat[:1]), a_hat[:-1]], axis=0)  # a_{n-1}
        a_up = jnp.concatenate([a_hat[1:], jnp.zeros_like(a_hat[:1])], axis=0)     # a_{n+1}

        out = -1j * (self.k_arr * float(self.vth))[None, :] * (
            self.sqrt_n[:, None] * a_down + self.sqrt_np1[:, None] * a_up
        )

        if self._closure_Ldiag_last is not None:
            out = out.at[int(self.Nv) - 1].add(self._closure_Ldiag_last * a_hat[int(self.Nv) - 1])

        return out

    # -------------------------
    # Poisson solver utilities
    # -------------------------

    def E_hat_from_rho_hat(self, rho_hat: Array, poisson_sign: float = +1.0) -> Array:
        """
        Compute E_hat from rho_hat using
            E_k = poisson_sign * i * rho_k / k,   k!=0
            E_0 = 0,
        so that i k E_k = -poisson_sign * rho_k.
        """
        E_hat = jnp.zeros_like(rho_hat)
        E_hat = E_hat.at[1:].set((poisson_sign * 1j) * rho_hat[1:] / self.k_arr[1:])
        return E_hat

    def E_phys_from_a_hat(self, a_hat: Array, poisson_sign: float = +1.0) -> Array:
        """Compute electric field E(x) in physical space from a_hat[0,k]."""
        rho_hat = a_hat[0]
        E_hat = self.E_hat_from_rho_hat(rho_hat, poisson_sign=poisson_sign)
        return jnp.fft.irfft(E_hat, n=int(self.Nx)).astype(self.real_dtype)

    def electric_energy(self, a_hat: Array, poisson_sign: float = +1.0) -> Array:
        """Return 0.5 * ∫ E^2 dx (discrete trapezoid == dx * sum)."""
        E = self.E_phys_from_a_hat(a_hat, poisson_sign=poisson_sign)
        return 0.5 * float(self.dx) * jnp.sum(E * E)

    # -------------------------
    # IMEX CNAB2 step
    # -------------------------

    def implicit_solve(self, rhs_hat: Array) -> Array:
        """
        Solve (I - dt/2 L) a_new = rhs_hat for all Fourier modes k (batched).
        """
        # vmap solver expects (Nk, Nv), so transpose:
        rhs_b = jnp.swapaxes(rhs_hat, 0, 1)  # (Nk, Nv)
        sol_b = self._solve_batched(self.sub, self.diag, self.sup, rhs_b)  # (Nk, Nv)
        return jnp.swapaxes(sol_b, 0, 1)  # (Nv, Nk)

    def step_cnab2(
        self,
        a_hat: Array,
        N_hat: Array,
        N_hat_prev: Array,
        *,
        extra_hat: Optional[Array] = None,
        extra_hat_prev: Optional[Array] = None,
        hermite_filter: Optional[Array] = None,
    ) -> Array:
        """
        One CNAB2 step given current state a_hat and explicit terms N_hat, N_hat_prev.

        Parameters
        ----------
        a_hat : (Nv, Nk) complex
            Current state.
        N_hat : (Nv, Nk) complex
            Explicit term at the current time.
        N_hat_prev : (Nv, Nk) complex
            Explicit term at the previous time.
        extra_hat : Optional[(Nv, Nk)] complex
            Additional explicit term, typically a learned Hermite-boundary flux.
        extra_hat_prev : Optional[(Nv, Nk)] complex
            Previous-step value of `extra_hat`.
        hermite_filter : Optional[(Nv,)]
            If provided, multiplies the updated a_hat by this per-mode factor.

        Returns
        -------
        a_hat_new : (Nv, Nk) complex
        """
        dt = float(self.dt)
        dt_half = 0.5 * dt

        if extra_hat is None:
            extra_hat = jnp.zeros_like(a_hat)
        if extra_hat_prev is None:
            extra_hat_prev = jnp.zeros_like(a_hat)

        La = self.streaming_hat(a_hat)
        rhs = a_hat + dt_half * La + dt * (
            1.5 * (N_hat + extra_hat) - 0.5 * (N_hat_prev + extra_hat_prev)
        )
        rhs = self.apply_mask_hat(rhs)

        a_new = self.implicit_solve(rhs)
        a_new = self.apply_mask_hat(a_new)

        if hermite_filter is not None:
            a_new = hermite_filter[:, None] * a_new

        a_new = self.apply_mask_hat(a_new)
        return a_new


def hermite_damping_term(a_phys: Array, damping_rates: Array) -> Array:
    """
    Apply a Hermite-diagonal damping term in physical space:

        d/dt a_n += -gamma_n a_n

    This returns N_phys = -gamma[:,None] * a_phys.

    Using physical space here is convenient because a_phys is typically already available
    when forming nonlinear products E(x)*something(x).
    """
    return -(damping_rates[:, None] * a_phys)
