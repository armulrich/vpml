"""
Benchmark suite in **JAX** to reproduce the *key linear tests* from:

  O. Palisso et al., "Effects of Artificial Collisions, Filtering, and Nonlocal Closure Approaches
  on Hermite-based Vlasov-Poisson Simulations", arXiv:2412.07073 (v3).

This script is designed to run side-by-side with the nonlinear solver script
fh_nonlinear_sim_jax.py, sharing the same core numerics via fh_core_jax.py.

What this benchmark script covers
---------------------------------
- Fig. 2-style damping profiles for:
    * Hypercollisions (Eq. 19)
    * Hou–Li filter interpreted as damping (Eq. 20, typically p=36)
- Fig. 3-style response-function approximation tests:
    * collisionless truncation
    * hypercollisions (tuned ν for each Nv)
    * nonlocal closure (Nm=1 and Nm=3)
    * filtering (tuned χ/Δt for each Nv)
- Fig. 4/5-style eigenvalue (discrete Landau root) scans using the approximate response function:
    * nonlocal closure parameter μ_{Nv-1} sweep
    * hypercollision ν sweep
    * filtering χ/Δt sweep
- Optional: linear Landau damping time simulation using the shared IMEX integrator (Section 4.1 style)

Everything numerical is in JAX. Plotting uses matplotlib (CPU).

Practical note
--------------
For eigenvalue/root-finding and response-function accuracy, enable 64-bit:
    export JAX_ENABLE_X64=True
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

# Matplotlib cache directory fix for sandboxed environments (optional)
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from fh_core_jax import (
    Array,
    FourierHermiteIMEX,
    HyperCollisions,
    HouLiFilter,
    NonlocalClosure,
    HermiteExponentialFilter,
    rfft_x,
    irfft_x,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


# =============================================================================
# Plasma dispersion function and response function
# =============================================================================

def plasma_dispersion_Z(xi: Array) -> Array:
    """
    Fried–Conte plasma dispersion function Z(ξ).

    We use the exact relation:
        Z(ξ) = i*sqrt(pi) * w(ξ),
    where w is the Faddeeva function (wofz).

    Note: JAX (in this environment) does not provide `jax.scipy.special.wofz`, and
    `jax.scipy.special.erfc` is real-only, so we fall back to SciPy’s `wofz`.
    """
    try:
        wofz = jsp.special.wofz  # type: ignore[attr-defined]
        w = wofz(xi.astype(jnp.complex128))
    except Exception:
        try:
            from scipy.special import wofz as sp_wofz  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "plasma_dispersion_Z requires SciPy (`pip install scipy`) in this environment "
                "because complex erfc/wofz are not available in jax.scipy."
            ) from exc
        w = jnp.asarray(sp_wofz(np.asarray(xi, dtype=np.complex128)))
    return 1j * jnp.sqrt(jnp.pi) * w


def response_function_R(xi: Array) -> Array:
    """Collisionless response function R(ξ) = 1 + ξ Z(ξ)."""
    return 1.0 + xi * plasma_dispersion_Z(xi)


def solve_landau_root_xi(
    k: float,
    xi0: Optional[complex] = None,
    maxiter: int = 60,
    tol: float = 1e-13,
) -> complex:
    """
    Solve the collisionless dispersion relation used in the paper:
        k^2 = - R(ξ)   <=>  F(ξ) = R(ξ) + k^2 = 0
    with complex Newton + simple backtracking.

    Returns
    -------
    xi : complex
        ξ = ω / sqrt(2|k|)

    Notes
    -----
    Uses the identity:
        Z'(ξ) = -2 (1 + ξ Z(ξ)) = -2 R(ξ)
    so:
        dR/dξ = Z(ξ) + ξ Z'(ξ) = Z(ξ) - 2 ξ R(ξ)
    """
    k2 = float(k) * float(k)

    if xi0 is None:
        # Heuristic start: warm plasma frequency with modest damping.
        wr = math.sqrt(1.0 + 3.0 * k2)
        xi = complex(wr / math.sqrt(2.0 * abs(k)), -0.5)
    else:
        xi = complex(xi0)

    for _ in range(maxiter):
        xi_j = jnp.array(xi, dtype=jnp.complex128)
        Z = plasma_dispersion_Z(xi_j)
        R = 1.0 + xi_j * Z
        dR = Z - (2.0 * xi_j * R)

        F = complex(jax.device_get(R)) + k2
        if abs(F) < tol:
            return xi
        Fp = complex(jax.device_get(dR))
        if Fp == 0.0:
            break

        step = F / Fp

        # Backtracking line search
        lam = 1.0
        xi_new = xi - step
        F_new = complex(jax.device_get(response_function_R(jnp.array(xi_new, jnp.complex128)))) + k2
        while abs(F_new) > abs(F) and lam > 1e-3:
            lam *= 0.5
            xi_new = xi - lam * step
            F_new = complex(jax.device_get(response_function_R(jnp.array(xi_new, jnp.complex128)))) + k2

        xi = xi_new
        if abs(lam * step) < tol:
            return xi

    return xi


def landau_omega(k: float, xi: complex) -> complex:
    """ω = ξ * sqrt(2|k|)."""
    return xi * math.sqrt(2.0 * abs(k))


def landau_gamma(k: float, xi: complex) -> float:
    """Damping rate γ = Im(ω) (typically negative)."""
    return float(np.imag(landau_omega(k, xi)))


# =============================================================================
# AW Hermite response function approximation (paper Eq. 17)
# =============================================================================

def advection_matrix_Abarbar(Nv: int) -> Array:
    """
    Real symmetric tridiagonal matrix with off-diagonals sqrt(n), n=1..Nv-1.
    Paper Eq. (16) up to scaling conventions.
    """
    s = jnp.sqrt(jnp.arange(1, Nv, dtype=jnp.float64))
    A = jnp.zeros((Nv, Nv), dtype=jnp.complex128)
    A = A.at[jnp.arange(Nv - 1), jnp.arange(1, Nv)].set(s)
    A = A.at[jnp.arange(1, Nv), jnp.arange(Nv - 1)].set(s)
    return A


def modify_A_for_method(
    A: Array,
    *,
    k: float,
    dissipation: Optional[Union[HyperCollisions, HouLiFilter]] = None,
    closure: Optional[NonlocalClosure] = None,
) -> Array:
    """
    Apply paper-style modifications:
      - dissipation: add i η_n to diagonal, with η_n = -gamma_n/k
      - nonlocal closure: modify last row by substituting ghost C_{Nv}
    """
    Nv = A.shape[0]
    A2 = A

    # Dissipation -> diagonal term i eta
    if dissipation is not None:
        gamma = dissipation.damping_rates()
        eta = -gamma / float(k)
        A2 = A2 + 1j * jnp.diag(eta.astype(jnp.float64))

    # Nonlocal closure -> last row modification (can make A dense in last row)
    if closure is not None:
        Nm = closure.Nm
        mu = closure.mu_tail.astype(jnp.float64)  # length Nm, corresponds to mu_{Nv-Nm..Nv-1}
        signk = float(np.sign(k)) if k != 0.0 else 0.0

        row = A2[Nv - 1].copy()
        cols = jnp.arange(Nv - Nm, Nv)
        # Add sqrt(Nv) * i * mu * sign(k) into those columns
        row = row.at[cols].add(jnp.sqrt(float(Nv)) * (1j * mu * signk))
        A2 = A2.at[Nv - 1].set(row)

    return A2


def response_function_aw_Nv(
    xi: Array,
    *,
    Nv: int,
    k: float = 1.0,
    dissipation: Optional[Union[HyperCollisions, HouLiFilter]] = None,
    closure: Optional[NonlocalClosure] = None,
) -> Array:
    """
    Hermite response approximation (paper Eq. 17):

        R_aw_Nv(ξ) = -c * e0^T ( ξ I - c A )^{-1} e1
        c = sign(k) / sqrt(2)

    where A is the modified advection matrix.
    """
    if k == 0.0:
        raise ValueError("k must be nonzero")
    c = float(np.sign(k)) / math.sqrt(2.0)

    A = advection_matrix_Abarbar(int(Nv))
    A = modify_A_for_method(A, k=k, dissipation=dissipation, closure=closure)

    I = jnp.eye(int(Nv), dtype=jnp.complex128)
    M = (xi.astype(jnp.complex128) * I) - (c * A)

    e1 = jnp.zeros((int(Nv),), dtype=jnp.complex128).at[1].set(1.0 + 0j)
    x = jnp.linalg.solve(M, e1)
    return -(c) * x[0]


def response_function_aw_Nv_and_deriv(
    xi: Array,
    *,
    Nv: int,
    k: float = 1.0,
    dissipation: Optional[Union[HyperCollisions, HouLiFilter]] = None,
    closure: Optional[NonlocalClosure] = None,
) -> Tuple[Array, Array]:
    """
    Return (R_aw_Nv(ξ), d/dξ R_aw_Nv(ξ)).

    If M(ξ) = ξI - cA, then:
        R = -c e0^T M^{-1} e1
        dR/dξ =  c e0^T M^{-2} e1
    """
    if k == 0.0:
        raise ValueError("k must be nonzero")
    c = float(np.sign(k)) / math.sqrt(2.0)

    A = advection_matrix_Abarbar(int(Nv))
    A = modify_A_for_method(A, k=k, dissipation=dissipation, closure=closure)

    I = jnp.eye(int(Nv), dtype=jnp.complex128)
    M = (xi.astype(jnp.complex128) * I) - (c * A)

    e1 = jnp.zeros((int(Nv),), dtype=jnp.complex128).at[1].set(1.0 + 0j)
    x = jnp.linalg.solve(M, e1)
    y = jnp.linalg.solve(M, x)

    Rv = -(c) * x[0]
    dRv = (c) * y[0]
    return Rv, dRv


def discrete_root_from_response(
    *,
    k: float,
    Nv: int,
    dissipation: Optional[Union[HyperCollisions, HouLiFilter]] = None,
    closure: Optional[NonlocalClosure] = None,
    xi0: Optional[complex] = None,
    maxiter: int = 60,
    tol: float = 1e-13,
) -> complex:
    """
    Compute the *discrete* Landau root ξ* of the modified finite Hermite system by solving:
        k^2 + R_aw_Nv(ξ) = 0
    via complex Newton using R_aw_Nv and its derivative.
    """
    k2 = float(k) * float(k)

    xi = complex(solve_landau_root_xi(k) if xi0 is None else xi0)

    for _ in range(maxiter):
        Rv_j, dRv_j = response_function_aw_Nv_and_deriv(
            jnp.array(xi, dtype=jnp.complex128),
            Nv=Nv, k=k, dissipation=dissipation, closure=closure,
        )
        Rv = complex(jax.device_get(Rv_j))
        dRv = complex(jax.device_get(dRv_j))

        F = Rv + k2
        if abs(F) < tol:
            return xi
        if dRv == 0.0:
            break

        step = F / dRv

        # Backtracking
        lam = 1.0
        xi_new = xi - step
        Rv_new = complex(jax.device_get(response_function_aw_Nv(jnp.array(xi_new, jnp.complex128),
                                                              Nv=Nv, k=k,
                                                              dissipation=dissipation, closure=closure)))
        F_new = Rv_new + k2
        while abs(F_new) > abs(F) and lam > 1e-3:
            lam *= 0.5
            xi_new = xi - lam * step
            Rv_new = complex(jax.device_get(response_function_aw_Nv(jnp.array(xi_new, jnp.complex128),
                                                                  Nv=Nv, k=k,
                                                                  dissipation=dissipation, closure=closure)))
            F_new = Rv_new + k2

        xi = xi_new
        if abs(lam * step) < tol:
            return xi

    return xi


# =============================================================================
# Benchmark framework
# =============================================================================

@dataclass
class BenchmarkResult:
    name: str
    payload: Dict[str, np.ndarray]

    def save_npz(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.payload)


class Benchmark:
    name: str

    def run(self) -> BenchmarkResult:
        raise NotImplementedError

    def plot(self, result: BenchmarkResult, outdir: Union[str, Path]) -> None:
        raise NotImplementedError


# =============================================================================
# Fig. 2: damping profiles
# =============================================================================

@dataclass
class Fig2DampingProfiles(Benchmark):
    name: str = "fig2_damping_profiles"
    p: int = 36

    def run(self) -> BenchmarkResult:
        nu = 1.0
        chi_over_dt = 1.0

        # Nv=20, alpha=1..8
        Nv1 = 20
        n1 = np.arange(Nv1)
        hyper1 = {}
        for a in range(1, 9):
            hyper1[a] = np.array(HyperCollisions(alpha=a, nu=nu, Nv=Nv1).damping_rates())
        filt1 = np.array(HouLiFilter(chi_over_dt=chi_over_dt, Nv=Nv1, p=self.p).damping_rates())

        # Nv=1000, alpha list
        Nv2 = 1000
        n2 = np.arange(Nv2)
        alphas2 = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        hyper2 = {a: np.array(HyperCollisions(alpha=a, nu=nu, Nv=Nv2).damping_rates()) for a in alphas2}
        filt2 = np.array(HouLiFilter(chi_over_dt=chi_over_dt, Nv=Nv2, p=self.p).damping_rates())

        payload: Dict[str, np.ndarray] = {
            "n_Nv20": n1,
            "filt_Nv20": filt1,
            "n_Nv1000": n2,
            "filt_Nv1000": filt2,
        }
        for a, g in hyper1.items():
            payload[f"hyper_Nv20_alpha{a}"] = g
        for a, g in hyper2.items():
            payload[f"hyper_Nv1000_alpha{a}"] = g

        return BenchmarkResult(self.name, payload)

    def plot(self, result: BenchmarkResult, outdir: Union[str, Path]) -> None:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        # (a) Nv=20
        ax = axes[0]
        n = result.payload["n_Nv20"]
        keys = sorted([k for k in result.payload if k.startswith("hyper_Nv20_alpha")],
                      key=lambda s: int(s.split("alpha")[1]))
        for key in keys:
            a = int(key.split("alpha")[1])
            ax.semilogy(n, result.payload[key], label=fr"$\alpha={a}$")
        ax.semilogy(n, result.payload["filt_Nv20"], "k--", lw=2, label=fr"$p={self.p}$")
        ax.set_title(r"(a) Damping rate $N_v=20$")
        ax.set_xlabel("nth Hermite moment")
        ax.set_ylabel("damping rate")
        ax.set_ylim(1e-7, 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)

        # (b) Nv=1000
        ax = axes[1]
        n = result.payload["n_Nv1000"]
        keys = sorted([k for k in result.payload if k.startswith("hyper_Nv1000_alpha")],
                      key=lambda s: int(s.split("alpha")[1]))
        for key in keys:
            a = int(key.split("alpha")[1])
            ax.semilogy(n, result.payload[key], label=fr"$\alpha={a}$")
        ax.semilogy(n, result.payload["filt_Nv1000"], "k--", lw=2, label=fr"$p={self.p}$")
        ax.set_title(r"(b) Damping rate $N_v=10^3$")
        ax.set_xlabel("nth Hermite moment")
        ax.set_ylim(1e-16, 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)

        fig.savefig(outdir / "fig2_damping_profiles.png", dpi=200)
        plt.close(fig)


# =============================================================================
# Fig. 3: response-function approximation benchmarks
# =============================================================================

@dataclass
class Fig3ResponseFunction(Benchmark):
    """
    Matches the logic of paper Fig. 3:
      (a) L2 error vs Nv for different methods (Table 1 parameters)
      (b) absolute error curves vs xi at Nv=12
    """
    name: str = "fig3_response_function"
    k: float = 1.0
    xi_min: float = 1e-2
    xi_max: float = 1e2
    n_xi: int = 50_000  # paper uses 1e5

    p: int = 36

    # Table 1 values from the prompt script (Nv in {4,6,8,10,12})
    table1_collisions_nu: Dict[int, Dict[int, float]] = None
    table1_nonlocal_nm1_mu: Dict[int, float] = None
    table1_nonlocal_nm3_mu: Dict[int, Tuple[float, float, float]] = None
    table1_filter_chi_over_dt: Dict[int, float] = None

    def __post_init__(self):
        if self.table1_collisions_nu is None:
            self.table1_collisions_nu = {
                4:  {1: 1.69, 2: 1.88},
                6:  {1: 1.74, 2: 9.69, 3: 2.35},
                8:  {1: 1.72, 2: 17.65, 3: 4.81, 4: 2.74},
                10: {1: 1.68, 2: 6.46, 3: 13.31, 4: 11.19},
                12: {1: 1.64, 2: 6.63, 3: 19.83, 4: 24.51},
            }
        if self.table1_nonlocal_nm1_mu is None:
            self.table1_nonlocal_nm1_mu = {4: -0.93, 6: -0.96, 8: -0.97, 10: -0.97, 12: -0.98}
        if self.table1_nonlocal_nm3_mu is None:
            self.table1_nonlocal_nm3_mu = {
                4:  (-1.31, 0.0, -0.30),
                6:  (-1.37, 0.0, -0.37),
                8:  (-1.40, 0.0, -0.40),
                10: (-1.42, 0.0, -0.42),
                12: (-1.44, 0.0, -0.44),
            }
        if self.table1_filter_chi_over_dt is None:
            self.table1_filter_chi_over_dt = {4: 1.88, 6: 2.35, 8: 2.75, 10: 3.13, 12: 3.51}

    def run(self) -> BenchmarkResult:
        xi = np.logspace(np.log10(self.xi_min), np.log10(self.xi_max), self.n_xi)
        xi_j = jnp.array(xi, dtype=jnp.float64)

        R_true = np.array(response_function_R(xi_j))  # complex

        def L2_rel_err(R_approx: np.ndarray) -> float:
            diff = R_approx - R_true
            num = np.sqrt(np.mean(np.abs(diff) ** 2))
            den = np.sqrt(np.mean(np.abs(R_true) ** 2))
            return float(num / den)

        Nv_list = [4, 6, 8, 10, 12]
        errs: Dict[str, list] = {
            "truncation": [],
            "hyper_a1": [], "hyper_a2": [], "hyper_a3": [], "hyper_a4": [],
            "nonlocal_nm1": [],
            "nonlocal_nm3": [],
            "filter": [],
        }

        # vmap over xi for speed
        for Nv in Nv_list:
            # truncation
            R_tr = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k))(xi_j))
            errs["truncation"].append(L2_rel_err(R_tr))

            # hypercollisions
            for a in [1, 2, 3, 4]:
                if a in self.table1_collisions_nu[Nv]:
                    nu = self.table1_collisions_nu[Nv][a]
                    diss = HyperCollisions(alpha=a, nu=nu, Nv=Nv)
                    R_h = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, dissipation=diss))(xi_j))
                    errs[f"hyper_a{a}"].append(L2_rel_err(R_h))
                else:
                    errs[f"hyper_a{a}"].append(np.nan)

            # nonlocal Nm=1
            mu1 = self.table1_nonlocal_nm1_mu[Nv]
            clo1 = NonlocalClosure(mu_tail=jnp.array([mu1], dtype=jnp.float64))
            R_nl1 = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, closure=clo1))(xi_j))
            errs["nonlocal_nm1"].append(L2_rel_err(R_nl1))

            # nonlocal Nm=3
            muNv1, muNv2, muNv3 = self.table1_nonlocal_nm3_mu[Nv]
            clo3 = NonlocalClosure(mu_tail=jnp.array([muNv3, muNv2, muNv1], dtype=jnp.float64))
            R_nl3 = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, closure=clo3))(xi_j))
            errs["nonlocal_nm3"].append(L2_rel_err(R_nl3))

            # filtering
            chi = self.table1_filter_chi_over_dt[Nv]
            filt = HouLiFilter(chi_over_dt=chi, Nv=Nv, p=self.p)
            R_f = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, dissipation=filt))(xi_j))
            errs["filter"].append(L2_rel_err(R_f))

        # Error curves at Nv=12
        Nv = 12
        abs_err_curves: Dict[str, np.ndarray] = {}

        R_tr = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k))(xi_j))
        abs_err_curves["truncation"] = np.abs(R_tr - R_true)

        for a in [1, 2, 3, 4]:
            nu = self.table1_collisions_nu[Nv][a]
            diss = HyperCollisions(alpha=a, nu=nu, Nv=Nv)
            R_h = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, dissipation=diss))(xi_j))
            abs_err_curves[f"hyper_a{a}"] = np.abs(R_h - R_true)

        mu1 = self.table1_nonlocal_nm1_mu[Nv]
        clo1 = NonlocalClosure(mu_tail=jnp.array([mu1], dtype=jnp.float64))
        R_nl1 = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, closure=clo1))(xi_j))
        abs_err_curves["nonlocal_nm1"] = np.abs(R_nl1 - R_true)

        muNv1, muNv2, muNv3 = self.table1_nonlocal_nm3_mu[Nv]
        clo3 = NonlocalClosure(mu_tail=jnp.array([muNv3, muNv2, muNv1], dtype=jnp.float64))
        R_nl3 = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, closure=clo3))(xi_j))
        abs_err_curves["nonlocal_nm3"] = np.abs(R_nl3 - R_true)

        chi = self.table1_filter_chi_over_dt[Nv]
        filt = HouLiFilter(chi_over_dt=chi, Nv=Nv, p=self.p)
        R_f = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, dissipation=filt))(xi_j))
        abs_err_curves["filter"] = np.abs(R_f - R_true)

        payload: Dict[str, np.ndarray] = {
            "Nv_list": np.array(Nv_list),
            "xi": xi,
            "R_true_real": np.real(R_true),
            "R_true_imag": np.imag(R_true),
        }
        for kname, arr in errs.items():
            payload[f"err_{kname}"] = np.array(arr, dtype=float)
        for kname, arr in abs_err_curves.items():
            payload[f"abs_err_{kname}"] = np.array(arr, dtype=float)

        return BenchmarkResult(self.name, payload)

    def plot(self, result: BenchmarkResult, outdir: Union[str, Path]) -> None:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        Nv_list = result.payload["Nv_list"]

        # (a) convergence
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.semilogy(Nv_list, result.payload["err_truncation"], "o-", label="truncation")
        for a in [1, 2, 3, 4]:
            ax.semilogy(Nv_list, result.payload[f"err_hyper_a{a}"], "o-", label=fr"hyper $\alpha={a}$")
        ax.semilogy(Nv_list, result.payload["err_nonlocal_nm1"], "o-", label=r"nonlocal $N_m=1$")
        ax.semilogy(Nv_list, result.payload["err_nonlocal_nm3"], "o-", label=r"nonlocal $N_m=3$")
        ax.semilogy(Nv_list, result.payload["err_filter"], "o-", label="filter")
        ax.set_xlabel(r"$N_v$")
        ax.set_ylabel(r"$\|R_{N_v}^{aw}-R\|_2/\|R\|_2$")
        ax.set_title("Fig. 3(a) — response-function convergence (k=1)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.savefig(outdir / "fig3a_response_function_convergence.png", dpi=200)
        plt.close(fig)

        # (b) absolute error at Nv=12
        xi = result.payload["xi"]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for key in [k for k in result.payload if k.startswith("abs_err_")]:
            label = key.replace("abs_err_", "")
            ax.loglog(xi, result.payload[key], label=label)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$|R_{12}^{aw}(\xi)-R(\xi)|$")
        ax.set_title("Fig. 3(b) — absolute response-function error (Nv=12, k=1)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(outdir / "fig3b_response_function_abs_error_Nv12.png", dpi=200)
        plt.close(fig)


# =============================================================================
# Fig. 4/5: eigenvalue scans via discrete Landau root
# =============================================================================

@dataclass
class Fig4EigenvalueScan(Benchmark):
    name: str = "fig4_eigenvalue_scan"
    Nv: int = 20
    k_list: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    p: int = 36

    mu_range: Tuple[float, float] = (-3.0, 0.0)
    n_mu: int = 250

    nu_range: Tuple[float, float] = (0.0, 100.0)
    n_nu: int = 250

    chi_over_dt_range: Tuple[float, float] = (0.0, 25.0)
    n_chi: int = 250

    def run(self) -> BenchmarkResult:
        Nv = int(self.Nv)

        # True collisionless gammas
        xi_true = {k: solve_landau_root_xi(float(k)) for k in self.k_list}
        gamma_true = {k: landau_gamma(float(k), xi_true[k]) for k in self.k_list}

        # (a) nonlocal mu sweep (Nm=1)
        mu_vals = np.linspace(self.mu_range[0], self.mu_range[1], self.n_mu)
        err_nonlocal = {k: np.zeros_like(mu_vals) for k in self.k_list}
        for k in self.k_list:
            g_true = gamma_true[k]
            xi_guess = xi_true[k]
            for i, mu in enumerate(mu_vals):
                clo = NonlocalClosure(mu_tail=jnp.array([mu], dtype=jnp.float64))
                xi_star = discrete_root_from_response(k=float(k), Nv=Nv, closure=clo, xi0=xi_guess)
                xi_guess = xi_star
                g_star = landau_gamma(float(k), xi_star)
                err_nonlocal[k][i] = (g_true - g_star) / g_true

        # (b-e) hypercollision nu sweeps
        nu_vals = np.linspace(self.nu_range[0], self.nu_range[1], self.n_nu)
        err_hyper = {}  # (alpha,k)->curve
        for a in [1, 2, 3, 4]:
            for k in self.k_list:
                g_true = gamma_true[k]
                curve = np.zeros_like(nu_vals)
                xi_guess = xi_true[k]
                for i, nu in enumerate(nu_vals):
                    diss = HyperCollisions(alpha=a, nu=float(nu), Nv=Nv)
                    xi_star = discrete_root_from_response(k=float(k), Nv=Nv, dissipation=diss, xi0=xi_guess)
                    xi_guess = xi_star
                    g_star = landau_gamma(float(k), xi_star)
                    curve[i] = (g_true - g_star) / g_true
                err_hyper[(a, k)] = curve

        # (f) filter chi sweep
        chi_vals = np.linspace(self.chi_over_dt_range[0], self.chi_over_dt_range[1], self.n_chi)
        err_filter = {k: np.zeros_like(chi_vals) for k in self.k_list}
        for k in self.k_list:
            g_true = gamma_true[k]
            xi_guess = xi_true[k]
            for i, chi in enumerate(chi_vals):
                filt = HouLiFilter(chi_over_dt=float(chi), Nv=Nv, p=self.p)
                xi_star = discrete_root_from_response(k=float(k), Nv=Nv, dissipation=filt, xi0=xi_guess)
                xi_guess = xi_star
                g_star = landau_gamma(float(k), xi_star)
                err_filter[k][i] = (g_true - g_star) / g_true

        payload: Dict[str, np.ndarray] = {
            "Nv": np.array([Nv]),
            "k_list": np.array(self.k_list),
            "mu_vals": mu_vals,
            "nu_vals": nu_vals,
            "chi_vals": chi_vals,
        }
        for k in self.k_list:
            payload[f"err_nonlocal_k{k}"] = err_nonlocal[k]
            payload[f"err_filter_k{k}"] = err_filter[k]
            payload[f"gamma_true_k{k}"] = np.array([gamma_true[k]])
        for (a, k), curve in err_hyper.items():
            payload[f"err_hyper_a{a}_k{k}"] = curve

        return BenchmarkResult(self.name, payload)

    def plot(self, result: BenchmarkResult, outdir: Union[str, Path]) -> None:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        k_list = result.payload["k_list"].tolist()
        mu_vals = result.payload["mu_vals"]
        nu_vals = result.payload["nu_vals"]
        chi_vals = result.payload["chi_vals"]
        Nv = int(result.payload["Nv"][0])

        # (a) nonlocal closure
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for k in k_list:
            ax.plot(mu_vals, result.payload[f"err_nonlocal_k{k}"], label=fr"$k={k}$")
        ax.axhline(0.0, color="k", lw=1, alpha=0.5)
        ax.set_xlabel(r"$\mu_{N_v-1}$")
        ax.set_ylabel(r"$(\gamma-\gamma^*)/\gamma$")
        ax.set_title(fr"Nonlocal closure (Nm=1), $N_v={Nv}$")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(outdir / f"fig4a_nonlocal_scan_Nv{Nv}.png", dpi=200)
        plt.close(fig)

        # (b-e) hypercollisions
        for a in [1, 2, 3, 4]:
            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
            for k in k_list:
                ax.plot(nu_vals, result.payload[f"err_hyper_a{a}_k{k}"], label=fr"$k={k}$")
            ax.axhline(0.0, color="k", lw=1, alpha=0.5)
            ax.set_xlabel(r"$\nu$")
            ax.set_ylabel(r"$(\gamma-\gamma^*)/\gamma$")
            ax.set_title(fr"Hypercollisions $\alpha={a}$, $N_v={Nv}$")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.savefig(outdir / f"fig4_hyper_a{a}_scan_Nv{Nv}.png", dpi=200)
            plt.close(fig)

        # (f) filter
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for k in k_list:
            ax.plot(chi_vals, result.payload[f"err_filter_k{k}"], label=fr"$k={k}$")
        ax.axhline(0.0, color="k", lw=1, alpha=0.5)
        ax.set_xlabel(r"$\chi/\Delta t$")
        ax.set_ylabel(r"$(\gamma-\gamma^*)/\gamma$")
        ax.set_title(fr"Hou–Li filter (p={self.p}), $N_v={Nv}$")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(outdir / f"fig4f_filter_scan_Nv{Nv}.png", dpi=200)
        plt.close(fig)


# =============================================================================
# Optional: linear Landau damping time simulation using the shared IMEX integrator
# =============================================================================

@dataclass
class LinearLandauTimeBenchmark(Benchmark):
    """
    A minimal Section 4.1-style time simulation in the AW Hermite basis using the shared IMEX integrator.

    Setup
    -----
      - Domain length L = 4π so k are half-integers.
      - Excite k=0.5 and k=1.5 with eps=1e-2.
      - Start from perturbation: a0(x,0)=eps*(cos(0.5x)+cos(1.5x)), other modes 0.

    Method options
    --------------
    - "truncation": no dissipation/closure
    - "hyper": hypercollisions (alpha, nu)
    - "filter": Hou–Li filter (chi_over_dt, p)
    - "nonlocal": Nm=1 closure (mu)

    Important
    ---------
    This benchmark is for comparing *time-domain* damping curves, not eigenvalue scans.
    """
    name: str = "linear_landau_time"
    method: str = "truncation"
    alpha: int = 2
    nu: float = 16.76
    chi_over_dt: float = 6.94
    mu: float = -1.01
    p: int = 36

    Nv: int = 20
    Nx: int = 10
    L: float = 4 * math.pi
    dt: float = 1e-2
    T: float = 40.0
    eps: float = 1e-2
    modes: Tuple[float, ...] = (0.5, 1.5)

    poisson_sign: float = -1.0  # often used in Landau damping conventions

    def run(self) -> BenchmarkResult:
        dissipation = None
        herm_filter = None
        closure = None

        if self.method == "hyper":
            dissipation = HyperCollisions(alpha=self.alpha, nu=self.nu, Nv=self.Nv)
        elif self.method == "filter":
            dissipation = HouLiFilter(chi_over_dt=self.chi_over_dt, Nv=self.Nv, p=self.p)
        elif self.method == "nonlocal":
            closure = NonlocalClosure(mu_tail=jnp.array([self.mu], dtype=jnp.float64))
        elif self.method == "truncation":
            pass
        else:
            raise ValueError("Unknown method")

        integ = FourierHermiteIMEX(Nx=self.Nx, Nv=self.Nv, Lx=self.L, dt=self.dt,
                                   vth=1.0, dealias_23=False, closure=closure)

        # Maxwellian equilibrium in this normalized basis: m0=1, others 0.
        m = jnp.zeros((self.Nv,), dtype=jnp.float64).at[0].set(1.0)

        # Initial condition: only a0 excited
        a0 = jnp.zeros((self.Nx,), dtype=jnp.float64)
        for k in self.modes:
            a0 = a0 + jnp.cos(float(k) * integ.x)
        a0 = float(self.eps) * a0

        a_phys0 = jnp.zeros((self.Nv, self.Nx), dtype=jnp.float64).at[0].set(a0)
        a_hat0 = rfft_x(a_phys0)

        # Optional Hou–Li per-step factor (if we use filter as explicit multiplication instead of damping in N)
        # For the paper's filter-as-damping interpretation, we keep it in dissipation (diagonal in N).
        if isinstance(dissipation, HouLiFilter):
            herm_filter = None

        # Explicit term for *linear* Landau damping:
        #   N_n = -sqrt(n) E * m_{n-1}   (drop E * a_{n-1} term as O(eps^2))
        def explicit_N_hat(a_hat: Array) -> Array:
            a_phys = irfft_x(a_hat, self.Nx)
            E = integ.E_phys_from_a_hat(a_hat, poisson_sign=self.poisson_sign)

            N_phys = jnp.zeros_like(a_phys)
            N_phys = N_phys.at[1:].set(-integ.sqrt_n[1:, None] * E[None, :] * m[:-1, None])

            if dissipation is not None:
                gamma = dissipation.damping_rates().astype(jnp.float64)
                N_phys = N_phys - gamma[:, None] * a_phys

            return integ.apply_mask_hat(rfft_x(N_phys))

        N0 = explicit_N_hat(a_hat0)
        E0 = integ.E_phys_from_a_hat(a_hat0, poisson_sign=self.poisson_sign)
        Ehat0 = jnp.fft.rfft(E0)

        nsteps = int(round(float(self.T) / float(self.dt)))
        times = np.linspace(0.0, nsteps * float(self.dt), nsteps + 1)

        def step(carry, i):
            a_hat, N_prev = carry
            N_hat = explicit_N_hat(a_hat)
            a_new = integ.step_cnab2(a_hat, N_hat, N_prev, hermite_filter=herm_filter)
            E = integ.E_phys_from_a_hat(a_new, poisson_sign=self.poisson_sign)
            Ehat = jnp.fft.rfft(E)
            return (a_new, N_hat), Ehat

        (_, _), Ehat_hist = jax.lax.scan(step, (a_hat0, N0), jnp.arange(1, nsteps + 1, dtype=jnp.int32))
        Ehat_hist = jnp.concatenate([Ehat0[None, :], Ehat_hist], axis=0)

        # Extract |E_k| for k=0.5 and 1.5 (indices 1 and 3 for Nx=10, L=4π)
        Ehat_np = np.array(Ehat_hist)
        E05 = np.abs(Ehat_np[:, 1])
        E15 = np.abs(Ehat_np[:, 3])

        # Analytic damping rates (collisionless)
        xi05 = solve_landau_root_xi(0.5)
        xi15 = solve_landau_root_xi(1.5)
        g05 = landau_gamma(0.5, xi05)
        g15 = landau_gamma(1.5, xi15)

        payload = {
            "times": times,
            "E_abs_k0p5": E05,
            "E_abs_k1p5": E15,
            "gamma_k0p5": np.array([g05]),
            "gamma_k1p5": np.array([g15]),
        }
        return BenchmarkResult(self.name + "_" + self.method, payload)

    def plot(self, result: BenchmarkResult, outdir: Union[str, Path]) -> None:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

        t = result.payload["times"]
        E05 = result.payload["E_abs_k0p5"]
        E15 = result.payload["E_abs_k1p5"]
        g05 = float(result.payload["gamma_k0p5"][0])
        g15 = float(result.payload["gamma_k1p5"][0])

        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.semilogy(t, E05, label=r"$|E_{k=0.5}(t)|$")
        ax.semilogy(t, E15, label=r"$|E_{k=1.5}(t)|$")

        ax.semilogy(t, E05[0] * np.exp(g05 * t), "k--", lw=1.5, label=r"analytic $k=0.5$")
        ax.semilogy(t, E15[0] * np.exp(g15 * t), "k:", lw=1.5, label=r"analytic $k=1.5$")

        ax.set_xlabel("t")
        ax.set_ylabel(r"$|E_k(t)|$")
        ax.set_title(f"Linear Landau damping — method={self.method}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(outdir / f"linear_landau_{self.method}.png", dpi=200)
        plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="JAX benchmarks for arXiv:2412.07073")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p2 = sub.add_parser("fig2", help="Damping profiles (Fig. 2 style)")
    p2.add_argument("--outdir", type=str, default="out_bench")

    p3 = sub.add_parser("fig3", help="Response-function benchmarks (Fig. 3 style)")
    p3.add_argument("--outdir", type=str, default="out_bench")
    p3.add_argument("--n_xi", type=int, default=50_000)

    p4 = sub.add_parser("fig4", help="Eigenvalue scan via discrete root (Fig. 4/5 style)")
    p4.add_argument("--outdir", type=str, default="out_bench")
    p4.add_argument("--Nv", type=int, default=20)

    pl = sub.add_parser("linear_landau", help="Linear Landau damping time simulation (optional)")
    pl.add_argument("--outdir", type=str, default="out_bench")
    pl.add_argument("--method", choices=["truncation", "hyper", "filter", "nonlocal"], default="truncation")
    pl.add_argument("--alpha", type=int, default=2)
    pl.add_argument("--nu", type=float, default=16.76)
    pl.add_argument("--chi_over_dt", type=float, default=6.94)
    pl.add_argument("--mu", type=float, default=-1.01)
    pl.add_argument("--T", type=float, default=40.0)

    args = parser.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "fig2":
        b = Fig2DampingProfiles()
        res = b.run()
        res.save_npz(outdir / "fig2_damping_profiles.npz")
        b.plot(res, outdir)

    elif args.cmd == "fig3":
        b = Fig3ResponseFunction(n_xi=args.n_xi)
        res = b.run()
        res.save_npz(outdir / "fig3_response_function.npz")
        b.plot(res, outdir)

    elif args.cmd == "fig4":
        b = Fig4EigenvalueScan(Nv=args.Nv)
        res = b.run()
        res.save_npz(outdir / f"fig4_eigen_scan_Nv{args.Nv}.npz")
        b.plot(res, outdir)

    elif args.cmd == "linear_landau":
        b = LinearLandauTimeBenchmark(method=args.method, alpha=args.alpha, nu=args.nu,
                                      chi_over_dt=args.chi_over_dt, mu=args.mu, T=args.T)
        res = b.run()
        res.save_npz(outdir / f"linear_landau_{args.method}.npz")
        b.plot(res, outdir)


if __name__ == "__main__":
    main()
