"""Analytical closures and dissipation models for Hermite truncations."""

from __future__ import annotations

from dataclasses import dataclass

from .._jax_precision import enable_x64

enable_x64()

import jax.numpy as jnp
import jax.scipy as jsp

Array = jnp.ndarray


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

    The paper's Eq. (21) is typically written as a closure for the unresolved tail coefficient C_{Nv}:

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
