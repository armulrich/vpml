"""
Core Fourier–Hermite operators + IMEX CNAB2 integrator in **JAX** for 1D1V Vlasov-type
models with periodic x and Hermite expansion in v.

This module is intentionally "physics-agnostic": it provides
  - Fourier grid utilities (rFFT in x)
  - Hermite ladder streaming operator (tridiagonal in Hermite index)
  - Poisson solver for E from density a0
  - Optional spectral dealiasing in x (2/3 rule)
  - Optional Hermite damping models:
        * Hyper-collisions (Palisso et al. 2024, arXiv:2412.07073)
        * Hou–Li filter interpreted as damping
  - Optional Hermite exponential filter (spectral viscosity) as a per-step multiplier
  - Optional Nm=1 Hammett–Perkins nonlocal closure (time-stepping support)

The nonlinear physics (equilibrium coefficients, acceleration term, external control fields, etc.)
live in separate scripts that import this module.

Notes on conventions
--------------------
We evolve *Hermite coefficients of the perturbation* a_n(x,t) such that

    f(x,v,t) = sum_n a_n(x,t) φ_n(v)

where φ_n are normalized probabilists' Hermite functions with Gaussian weight.
Density is ρ(x,t) = a_0(x,t).

Poisson in 1D periodic domain is handled in Fourier space:
    i k E_k = poisson_sign * ρ_k   (k != 0),  E_0 = 0
with poisson_sign = +1 matching the convention used in the provided numpy two-stream/bump-on-tail script.
Some papers use the opposite sign; you can just flip it by passing poisson_sign=-1.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp


# Prefer float64/complex128 if the user enabled it (recommended for eigenvalues/roots).
try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass

Array = jnp.ndarray


# =============================================================================
# Hermite-space dissipation / closure models
# =============================================================================

@dataclass(frozen=True)
class HyperCollisions:
    """
    Hyper-collisional damping in Hermite index n (Palisso et al. Eq. 19).

        dC_n/dt += -gamma_n * C_n

    with
        gamma_n = nu * n!/(n-2α+1)! * (Nv-2α)!/(Nv-1)!    for n >= 2α-1
                = 0                                     otherwise

    Parameters
    ----------
    alpha : int
        Hyper-collision order α.
    nu : float
        Collision frequency ν.
    Nv : int
        Number of Hermite modes (0..Nv-1).
    """
    alpha: int
    nu: float
    Nv: int

    def damping_rates(self) -> Array:
        n = jnp.arange(self.Nv, dtype=jnp.float64)
        alpha = int(self.alpha)
        Nv = int(self.Nv)

        cutoff = 2 * alpha - 1
        mask = n >= cutoff

        # term1 = n!/(n-2α+1)!  (use gammaln for stability)
        # n-2α+1 == n - (2α-1)
        term1 = jnp.exp(
            jsp.special.gammaln(n + 1.0)
            - jsp.special.gammaln(n - float(2 * alpha - 1) + 1.0)
        )

        # term2 = (Nv-2α)!/(Nv-1)!
        term2 = jnp.exp(
            jsp.special.gammaln(float(Nv - 2 * alpha) + 1.0)
            - jsp.special.gammaln(float(Nv))
        )

        gamma = float(self.nu) * term1 * term2
        return jnp.where(mask, gamma, 0.0)


@dataclass(frozen=True)
class HouLiFilter:
    """
    Hou–Li exponential filter interpreted as an equivalent damping rate (Palisso et al. Eq. 20).

        C_n <- C_n * exp( -χ (n/(Nv-1))^p )

    If applied once per time step, this is equivalent to a damping term

        dC_n/dt += -(χ/Δt) (n/(Nv-1))^p C_n

    We store chi_over_dt = χ/Δt.

    Parameters
    ----------
    chi_over_dt : float
        χ/Δt in the paper's notation.
    Nv : int
        Number of Hermite modes.
    p : int
        Filter exponent (paper uses p=36 in most plots).
    """
    chi_over_dt: float
    Nv: int
    p: int = 36

    def damping_rates(self) -> Array:
        n = jnp.arange(self.Nv, dtype=jnp.float64)
        frac = n / float(self.Nv - 1)
        return float(self.chi_over_dt) * frac ** float(self.p)

    def per_step_factors(self, dt: float) -> Array:
        """
        Multiplicative factors to apply once per time step of size dt:
            exp( -dt * (χ/Δt) (n/(Nv-1))^p )
        """
        n = jnp.arange(self.Nv, dtype=jnp.float64)
        frac = n / float(self.Nv - 1)
        return jnp.exp(-float(dt) * float(self.chi_over_dt) * frac ** float(self.p))


@dataclass(frozen=True)
class HermiteExponentialFilter:
    """
    Simple exponential "spectral viscosity" filter used frequently in nonlinear Hermite truncations:

        C_n <- C_n * exp( -alpha * (n/(Nv-1))^p )

    This matches the style used in the provided numpy two-stream/bump-on-tail script.
    """
    alpha: float
    Nv: int
    p: int = 8

    def factors(self) -> Array:
        n = jnp.arange(self.Nv, dtype=jnp.float64)
        frac = n / float(self.Nv - 1)
        return jnp.exp(-float(self.alpha) * frac ** float(self.p))


@dataclass(frozen=True)
class NonlocalClosure:
    """
    Hammett–Perkins / nonlocal closure "tail" coefficients.

    The paper's Eq. (21) is typically written as a closure for the ghost coefficient C_{Nv}:

        C_{Nv} = sum_{j=0..Nm-1} i * mu_{Nv-Nm+j} * sign(k) * C_{Nv-Nm+j}

    This module stores the tail mu coefficients as an array mu_tail of shape (Nm,).

    IMPORTANT
    ---------
    For *time-stepping* with a tridiagonal implicit solve, only Nm=1 is supported
    (it remains tridiagonal after substitution).
    For response-function / eigenvalue benchmarks, Nm>1 is also useful and is handled
    in the benchmark script (dense linear algebra).
    """
    mu_tail: Array  # shape (Nm,)

    @property
    def Nm(self) -> int:
        return int(self.mu_tail.shape[0])


# =============================================================================
# Core numerical building blocks
# =============================================================================

def rfft_x(u_phys: Array) -> Array:
    """Real FFT over x axis=1, returning complex128."""
    return jnp.fft.rfft(u_phys, axis=1).astype(jnp.complex128)


def irfft_x(u_hat: Array, Nx: int) -> Array:
    """Inverse real FFT over x axis=1, returning float64."""
    return jnp.fft.irfft(u_hat, n=Nx, axis=1).astype(jnp.float64)
# Backwards-compatible aliases (older drafts used leading underscores)
_rfft_x = rfft_x
_irfft_x = irfft_x



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


def tridiag_solve(sub: Array, diag: Array, sup: Array, rhs: Array) -> Array:
    """
    Thomas algorithm for a single tridiagonal system in JAX.

    Parameters
    ----------
    sub : (N-1,) complex
        Sub-diagonal entries A[i, i-1] for i=1..N-1.
    diag : (N,) complex
        Diagonal entries A[i,i].
    sup : (N-1,) complex
        Super-diagonal entries A[i, i+1] for i=0..N-2.
    rhs : (N,) complex
        Right-hand side.

    Returns
    -------
    x : (N,) complex
        Solution of Ax=rhs.
    """
    N = rhs.shape[0]
    z = jnp.array(0.0 + 0.0j, dtype=rhs.dtype)

    # Pad to length N for scan-friendly indexing.
    sub_p = jnp.concatenate([jnp.array([z], dtype=rhs.dtype), sub])      # (N,)
    sup_p = jnp.concatenate([sup, jnp.array([z], dtype=rhs.dtype)])      # (N,)

    # Forward sweep
    c0 = sup_p[0] / diag[0]
    d0 = rhs[0] / diag[0]

    def fwd(carry, i):
        c_prev, d_prev = carry
        den = diag[i] - sub_p[i] * c_prev
        c = sup_p[i] / den
        d = (rhs[i] - sub_p[i] * d_prev) / den
        return (c, d), (c, d)

    (_, _), hist = jax.lax.scan(fwd, (c0, d0), jnp.arange(1, N))
    c_full = jnp.concatenate([jnp.array([c0], dtype=rhs.dtype), hist[0]])
    d_full = jnp.concatenate([jnp.array([d0], dtype=rhs.dtype), hist[1]])

    # Back substitution
    xN = d_full[-1]

    def bwd(x_next, i):
        x_i = d_full[i] - c_full[i] * x_next
        return x_i, x_i

    _, x_rev = jax.lax.scan(bwd, xN, jnp.arange(N - 2, -1, -1))
    x0_to_N2 = x_rev[::-1]
    return jnp.concatenate([x0_to_N2, jnp.array([xN], dtype=rhs.dtype)])


# =============================================================================
# Fourier–Hermite IMEX CNAB2 integrator
# =============================================================================

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

    def __post_init__(self):
        if self.Nx <= 0 or self.Nv <= 0:
            raise ValueError("Nx and Nv must be positive")
        if self.vth <= 0:
            raise ValueError("vth must be positive")

        dx = float(self.Lx) / float(self.Nx)
        self.dx = dx

        self.x = jnp.linspace(0.0, float(self.Lx), int(self.Nx), endpoint=False)
        self.k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(int(self.Nx), d=dx).astype(jnp.float64)
        self.Nk = int(self.Nx // 2 + 1)

        self.sqrt_n = jnp.sqrt(jnp.arange(int(self.Nv), dtype=jnp.float64))
        self.sqrt_np1 = jnp.sqrt(jnp.arange(1, int(self.Nv) + 1, dtype=jnp.float64))

        # 2/3 dealias mask (rFFT length Nk)
        self.mask = _mask_23(self.Nx) if self.dealias_23 else None

        # --- Precompute tridiagonal coefficients for the implicit CN solve ---
        dt_half = 0.5 * float(self.dt)
        s = jnp.sqrt(jnp.arange(1, int(self.Nv), dtype=jnp.float64))              # sqrt(1..Nv-1)
        val = (1j * dt_half * self.k_arr * float(self.vth)).astype(jnp.complex128)  # (Nk,)

        # sub/sup: (Nk, Nv-1)
        self.sub = (val[:, None] * s[None, :]).astype(jnp.complex128)
        self.sup = self.sub.copy()
        self.diag = jnp.ones((self.Nk, int(self.Nv)), dtype=jnp.complex128)

        # Optional Nm=1 closure -> diagonal modification on last Hermite mode
        self._closure_Ldiag_last = None
        if self.closure is not None:
            Nm = int(self.closure.Nm)
            if Nm != 1:
                raise ValueError("Time-stepping integrator supports NonlocalClosure only for Nm=1.")
            mu = jnp.asarray(self.closure.mu_tail[0], dtype=jnp.float64)
            signk = jnp.sign(self.k_arr)
            # L_diag_last = k*vth*sqrt(Nv)*mu*sign(k)
            Ldiag_last = (self.k_arr * float(self.vth) * jnp.sqrt(float(self.Nv)) * mu * signk)
            self._closure_Ldiag_last = Ldiag_last.astype(jnp.complex128)
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
        Compute E_hat from rho_hat using:
            i k E_k = poisson_sign * rho_k,   k!=0
            E_0 = 0
        """
        E_hat = jnp.zeros_like(rho_hat)
        E_hat = E_hat.at[1:].set((poisson_sign * 1j) * rho_hat[1:] / self.k_arr[1:])
        return E_hat

    def E_phys_from_a_hat(self, a_hat: Array, poisson_sign: float = +1.0) -> Array:
        """Compute electric field E(x) in physical space from a_hat[0,k]."""
        rho_hat = a_hat[0]
        E_hat = self.E_hat_from_rho_hat(rho_hat, poisson_sign=poisson_sign)
        return jnp.fft.irfft(E_hat, n=int(self.Nx)).astype(jnp.float64)

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
        hermite_filter : Optional[(Nv,)]
            If provided, multiplies the updated a_hat by this per-mode factor.

        Returns
        -------
        a_hat_new : (Nv, Nk) complex
        """
        dt = float(self.dt)
        dt_half = 0.5 * dt

        La = self.streaming_hat(a_hat)
        rhs = a_hat + dt_half * La + dt * (1.5 * N_hat - 0.5 * N_hat_prev)
        rhs = self.apply_mask_hat(rhs)

        a_new = self.implicit_solve(rhs)
        a_new = self.apply_mask_hat(a_new)

        if hermite_filter is not None:
            a_new = hermite_filter[:, None] * a_new

        a_new = self.apply_mask_hat(a_new)
        return a_new


# =============================================================================
# Small helpers for explicit-term construction (shared by sims/benchmarks)
# =============================================================================

def hermite_damping_term(a_phys: Array, damping_rates: Array) -> Array:
    """
    Apply a Hermite-diagonal damping term in physical space:

        d/dt a_n += -gamma_n a_n

    This returns N_phys = -gamma[:,None] * a_phys.

    Using physical space here is convenient because a_phys is typically already available
    when forming nonlinear products E(x)*something(x).
    """
    return -(damping_rates[:, None] * a_phys)
