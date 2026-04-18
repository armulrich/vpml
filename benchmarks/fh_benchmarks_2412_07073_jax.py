"""
Benchmark suite in **JAX** to reproduce the *key linear tests* from:

  O. Palisso et al., "Effects of Artificial Collisions, Filtering, and Nonlocal Closure Approaches
  on Hermite-based Vlasov-Poisson Simulations", arXiv:2412.07073 (v3).

This script is designed to run side-by-side with the nonlinear solver script
``benchmarks/fh_nonlinear_sim_jax.py``, sharing the same core numerics via ``vpml.core``.

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
- Fig. 4/5-style eigenvalue scans using the discrete linearized Vlasov–Poisson matrix:
    * nonlocal closure parameter μ_{Nv-1} sweep
    * hypercollision ν sweep
    * filtering χ/Δt sweep
- Optional: linear Landau damping time simulation using implicit midpoint/JFNK (Section 4.1 style)
- Fig. 10-style nonlinear Landau damping phase-space snapshots for the classical closures

Everything numerical is in JAX. Plotting uses matplotlib (CPU).

Practical note
--------------
For eigenvalue/root-finding and response-function accuracy, enable 64-bit:
    export JAX_ENABLE_X64=True
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from vpml.jax_runtime import bootstrap_jax_runtime, print_jax_runtime_summary

bootstrap_jax_runtime()

try:
    import scipy.linalg as la
except Exception:  # pragma: no cover
    la = None

# Matplotlib cache directory fix for sandboxed environments (optional)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))
# Silence noisy XLA/JAX C++ warnings on macOS benchmark runs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp

from vpml.core import (
    Array,
    HermiteExponentialFilter,
    HouLiFilter,
    HyperCollisions,
    LearnedInterfaceClosure,
    NonlocalClosure,
    hermite_basis_phi,
    load_learned_interface_closure_npz,
)
from vpml.linear_landau import (
    LinearLandauConfig,
    landau_gamma_mag,
    response_function_R,
    run_linear_landau_rollout,
    solve_landau_root_xi,
)
from vpml.nonlinear_landau import (
    NonlinearLandauParams,
    _time_key,
    run_nonlinear_landau_rollout_raw,
)
from vpml.visualization.benchmarks import (
    save_fig2_damping_profiles,
    save_fig3_response_function,
    save_fig4_eigenvalue_scan,
    save_fig10_learned_comparison_phase_space,
    save_fig10_nonlinear_landau_phase_space,
    save_linear_landau_comparison as save_linear_landau_comparison_figure,
    save_linear_landau_time,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


def build_linear_Q(
    Nv: int,
    k: float,
    gamma: Optional[np.ndarray] = None,
    mu_nm1: Optional[float] = None,
    mu_tail: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Discrete linearized Vlasov–Poisson matrix Q used in the paper's Fig. 4/5 benchmark.

    The least-damped discrete mode is extracted from the eigenvalue of Q with the
    smallest positive real part.
    """
    if k == 0.0:
        raise ValueError("k must be nonzero")
    if Nv < 2:
        raise ValueError("Nv must be at least 2")

    Q = np.zeros((Nv, Nv), dtype=np.complex128)
    gamma_vec = np.zeros(Nv, dtype=float) if gamma is None else np.asarray(gamma, dtype=float)
    if gamma_vec.shape != (Nv,):
        raise ValueError(f"gamma must have shape ({Nv},), got {gamma_vec.shape}")

    np.fill_diagonal(Q, gamma_vec)

    # Streaming + Poisson coupling in the Hermite basis.
    Q[0, 1] = 1j * k
    Q[1, 0] = 1j * k * (1.0 + 1.0 / (k * k))

    for n in range(1, Nv - 1):
        if n >= 2:
            Q[n, n - 1] = 1j * k * math.sqrt(n)
        Q[n, n + 1] = 1j * k * math.sqrt(n + 1)

    Q[Nv - 1, Nv - 2] = 1j * k * math.sqrt(Nv - 1)

    if mu_nm1 is not None and mu_tail is not None:
        raise ValueError("Pass only one of mu_nm1 or mu_tail")

    if mu_tail is None and mu_nm1 is not None:
        mu_tail = np.array([mu_nm1], dtype=np.complex128)

    if mu_tail is not None:
        mu_tail = np.asarray(mu_tail, dtype=np.complex128)
        Nm = int(mu_tail.shape[0])
        if Nm > Nv:
            raise ValueError(f"mu_tail length {Nm} exceeds Nv={Nv}")
        cols = np.arange(Nv - Nm, Nv)
        Q[Nv - 1, cols] += -abs(k) * math.sqrt(Nv) * mu_tail

    return Q


def least_damped_gamma_star(
    Nv: int,
    k: float,
    gamma: Optional[np.ndarray] = None,
    mu_nm1: Optional[float] = None,
    mu_tail: Optional[np.ndarray] = None,
    tol: float = 1e-12,
) -> float:
    """Return the smallest positive real part among the discrete eigenvalues of Q."""
    Q = build_linear_Q(Nv, k, gamma=gamma, mu_nm1=mu_nm1, mu_tail=mu_tail)
    vals = la.eigvals(Q) if la is not None else np.linalg.eigvals(Q)
    re = np.real(vals)
    pos = re[re > tol]
    return 0.0 if pos.size == 0 else float(np.min(pos))


def select_paper_optimal_parameter(
    x: np.ndarray,
    rel_err: np.ndarray,
    near_zero_tol: float = 1e-2,
) -> float:
    """
    Heuristic for the dashed vertical lines in Fig. 4.

    We use the first parameter value whose relative error is within `near_zero_tol`
    of zero; if that never happens, fall back to the minimum-|error| point.
    """
    abs_err = np.abs(rel_err)
    near_zero = np.flatnonzero(abs_err <= near_zero_tol)
    if near_zero.size:
        return float(x[int(near_zero[0])])
    return float(x[int(np.argmin(abs_err))])


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
      - nonlocal closure: modify last row by substituting the unresolved tail coefficient C_{Nv}
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
        mu = jnp.asarray(closure.mu_tail, dtype=jnp.complex128)
        Nm = int(closure.Nm)
        if Nm > Nv:
            raise ValueError(f"Tail length {Nm} exceeds Nv={Nv}")
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
    A = modify_A_for_method(
        A,
        k=k,
        dissipation=dissipation,
        closure=closure,
    )

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
    A = modify_A_for_method(
        A,
        k=k,
        dissipation=dissipation,
        closure=closure,
    )

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
            Nv=Nv,
            k=k,
            dissipation=dissipation,
            closure=closure,
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
                                                              dissipation=dissipation,
                                                              closure=closure)))
        F_new = Rv_new + k2
        while abs(F_new) > abs(F) and lam > 1e-3:
            lam *= 0.5
            xi_new = xi - lam * step
            Rv_new = complex(jax.device_get(response_function_aw_Nv(jnp.array(xi_new, jnp.complex128),
                                                                  Nv=Nv, k=k,
                                                                  dissipation=dissipation,
                                                                  closure=closure)))
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
        save_fig2_damping_profiles(result.payload, p=self.p, outdir=outdir)


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
    n_xi: int = 100_000
    n_xi_plot: int = 4_001
    learned_checkpoint: Optional[str] = None
    plot_floor: float = 1e-8
    plot_ymax: float = 5e-1

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
        xi_plot = np.logspace(np.log10(self.xi_min), np.log10(self.xi_max), self.n_xi_plot)
        xi_j = jnp.array(xi, dtype=jnp.float64)
        xi_plot_j = jnp.array(xi_plot, dtype=jnp.float64)
        if self.learned_checkpoint is not None:
            raise ValueError(
                "The learned interface closure is state-dependent and is not supported in fig3 "
                "response-function benchmarks. Remove --learned-checkpoint and use time-domain rollouts instead."
            )

        R_true = np.array(response_function_R(xi_j))  # complex
        R_true_plot = np.array(response_function_R(xi_plot_j))  # complex

        def L2_err(R_approx: np.ndarray) -> float:
            diff = R_approx - R_true
            return float(np.sqrt(np.sum(np.abs(diff) ** 2)))

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
            errs["truncation"].append(L2_err(R_tr))

            # hypercollisions
            for a in [1, 2, 3, 4]:
                if a in self.table1_collisions_nu[Nv]:
                    nu = self.table1_collisions_nu[Nv][a]
                    diss = HyperCollisions(alpha=a, nu=nu, Nv=Nv)
                    R_h = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, dissipation=diss))(xi_j))
                    errs[f"hyper_a{a}"].append(L2_err(R_h))
                else:
                    errs[f"hyper_a{a}"].append(np.nan)

            # nonlocal Nm=1
            mu1 = self.table1_nonlocal_nm1_mu[Nv]
            clo1 = NonlocalClosure(mu_tail=jnp.array([mu1], dtype=jnp.float64))
            R_nl1 = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, closure=clo1))(xi_j))
            errs["nonlocal_nm1"].append(L2_err(R_nl1))

            # nonlocal Nm=3
            muNv1, muNv2, muNv3 = self.table1_nonlocal_nm3_mu[Nv]
            clo3 = NonlocalClosure(mu_tail=jnp.array([muNv3, muNv2, muNv1], dtype=jnp.float64))
            R_nl3 = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, closure=clo3))(xi_j))
            errs["nonlocal_nm3"].append(L2_err(R_nl3))

            # filtering
            chi = self.table1_filter_chi_over_dt[Nv]
            filt = HouLiFilter(chi_over_dt=chi, Nv=Nv, p=self.p)
            R_f = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, dissipation=filt))(xi_j))
            errs["filter"].append(L2_err(R_f))

        # Error curves at Nv=12
        Nv = 12
        abs_err_curves: Dict[str, np.ndarray] = {}

        R_tr = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k))(xi_plot_j))
        abs_err_curves["truncation"] = np.abs(R_tr - R_true_plot)

        for a in [1, 2, 3, 4]:
            nu = self.table1_collisions_nu[Nv][a]
            diss = HyperCollisions(alpha=a, nu=nu, Nv=Nv)
            R_h = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, dissipation=diss))(xi_plot_j))
            abs_err_curves[f"hyper_a{a}"] = np.abs(R_h - R_true_plot)

        mu1 = self.table1_nonlocal_nm1_mu[Nv]
        clo1 = NonlocalClosure(mu_tail=jnp.array([mu1], dtype=jnp.float64))
        R_nl1 = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, closure=clo1))(xi_plot_j))
        abs_err_curves["nonlocal_nm1"] = np.abs(R_nl1 - R_true_plot)

        muNv1, muNv2, muNv3 = self.table1_nonlocal_nm3_mu[Nv]
        clo3 = NonlocalClosure(mu_tail=jnp.array([muNv3, muNv2, muNv1], dtype=jnp.float64))
        R_nl3 = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, closure=clo3))(xi_plot_j))
        abs_err_curves["nonlocal_nm3"] = np.abs(R_nl3 - R_true_plot)

        chi = self.table1_filter_chi_over_dt[Nv]
        filt = HouLiFilter(chi_over_dt=chi, Nv=Nv, p=self.p)
        R_f = np.array(jax.vmap(lambda z: response_function_aw_Nv(z, Nv=Nv, k=self.k, dissipation=filt))(xi_plot_j))
        abs_err_curves["filter"] = np.abs(R_f - R_true_plot)

        payload: Dict[str, np.ndarray] = {
            "Nv_list": np.array(Nv_list),
            "xi": xi,
            "xi_plot": xi_plot,
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
        save_fig3_response_function(
            result.payload,
            Nv_list=np.asarray(result.payload["Nv_list"], dtype=int),
            xi_plot=np.asarray(result.payload["xi_plot"], dtype=float),
            xi_min=self.xi_min,
            xi_max=self.xi_max,
            plot_floor=self.plot_floor,
            plot_ymax=self.plot_ymax,
            has_learned="err_learned" in result.payload,
            outdir=outdir,
        )


# =============================================================================
# Fig. 4/5: eigenvalue scans via the discrete linearized VP matrix
# =============================================================================

@dataclass
class Fig4EigenvalueScan(Benchmark):
    name: str = "fig4_eigenvalue_scan"
    Nv: int = 20
    k_list: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    learned_checkpoint: Optional[str] = None
    p: int = 36

    mu_range: Tuple[float, float] = (-3.0, 0.0)
    n_mu: int = 601
    hyper_nu_ranges: Dict[int, Tuple[float, float]] = field(default_factory=lambda: {
        1: (0.0, 10.0),
        2: (0.0, 25.0),
        3: (0.0, 25.0),
        4: (0.0, 20.0),
    })
    n_nu: int = 1001
    chi_over_dt_range: Tuple[float, float] = (0.0, 15.0)
    n_chi: int = 601

    def run(self) -> BenchmarkResult:
        Nv = int(self.Nv)
        k_list = tuple(float(k) for k in self.k_list)
        if self.learned_checkpoint is not None:
            raise ValueError(
                "The learned interface closure is state-dependent and is not supported in fig4 "
                "eigenvalue benchmarks. Remove --learned-checkpoint and use time-domain rollouts instead."
            )

        def rel_err(g_true: float, g_star: float) -> float:
            return 0.0 if abs(g_true) < 1e-14 else (g_true - g_star) / g_true

        # True collisionless Landau damping magnitudes.
        xi_true = {k: solve_landau_root_xi(k) for k in k_list}
        gamma_true = {k: landau_gamma_mag(k, xi_true[k]) for k in k_list}

        # (a) nonlocal mu sweep (Nm=1)
        mu_vals = np.linspace(self.mu_range[0], self.mu_range[1], self.n_mu)
        err_nonlocal = {k: np.zeros_like(mu_vals) for k in k_list}
        for k in k_list:
            g_true = gamma_true[k]
            for i, mu in enumerate(mu_vals):
                g_star = least_damped_gamma_star(Nv=Nv, k=k, mu_nm1=float(mu))
                err_nonlocal[k][i] = rel_err(g_true, g_star)
        mu_opt = float(np.median([
            select_paper_optimal_parameter(mu_vals, err_nonlocal[k]) for k in k_list
        ]))

        # (b-e) hypercollision nu sweeps
        err_hyper = {}  # (alpha,k)->curve
        nu_axes = {}
        opt_hyper = {}
        for a in [1, 2, 3, 4]:
            nu_min, nu_max = self.hyper_nu_ranges[a]
            nu_vals = np.linspace(nu_min, nu_max, self.n_nu)
            nu_axes[a] = nu_vals
            for k in k_list:
                g_true = gamma_true[k]
                curve = np.zeros_like(nu_vals)
                for i, nu in enumerate(nu_vals):
                    diss = HyperCollisions(alpha=a, nu=float(nu), Nv=Nv)
                    gamma = np.asarray(diss.damping_rates(), dtype=float)
                    g_star = least_damped_gamma_star(Nv=Nv, k=k, gamma=gamma)
                    curve[i] = rel_err(g_true, g_star)
                err_hyper[(a, k)] = curve
                opt_hyper[(a, k)] = select_paper_optimal_parameter(nu_vals, curve)

        # (f) filter chi sweep
        chi_vals = np.linspace(self.chi_over_dt_range[0], self.chi_over_dt_range[1], self.n_chi)
        err_filter = {k: np.zeros_like(chi_vals) for k in k_list}
        opt_filter = {}
        for k in k_list:
            g_true = gamma_true[k]
            for i, chi in enumerate(chi_vals):
                filt = HouLiFilter(chi_over_dt=float(chi), Nv=Nv, p=self.p)
                gamma = np.asarray(filt.damping_rates(), dtype=float)
                g_star = least_damped_gamma_star(Nv=Nv, k=k, gamma=gamma)
                err_filter[k][i] = rel_err(g_true, g_star)
            opt_filter[k] = select_paper_optimal_parameter(chi_vals, err_filter[k])

        payload: Dict[str, np.ndarray] = {
            "Nv": np.array([Nv]),
            "k_list": np.array(k_list, dtype=float),
            "mu_vals": mu_vals,
            "chi_vals": chi_vals,
            "mu_opt": np.array([mu_opt], dtype=float),
        }
        for k in k_list:
            payload[f"err_nonlocal_k{k}"] = err_nonlocal[k]
            payload[f"err_filter_k{k}"] = err_filter[k]
            payload[f"gamma_true_k{k}"] = np.array([gamma_true[k]])
            payload[f"opt_filter_k{k}"] = np.array([opt_filter[k]], dtype=float)
        for a, nu_vals in nu_axes.items():
            payload[f"nu_vals_a{a}"] = nu_vals
        for (a, k), curve in err_hyper.items():
            payload[f"err_hyper_a{a}_k{k}"] = curve
            payload[f"opt_hyper_a{a}_k{k}"] = np.array([opt_hyper[(a, k)]], dtype=float)

        return BenchmarkResult(self.name, payload)

    def plot(self, result: BenchmarkResult, outdir: Union[str, Path]) -> None:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        save_fig4_eigenvalue_scan(
            result.payload,
            Nv=int(result.payload["Nv"][0]),
            k_list=tuple(float(k) for k in result.payload["k_list"].tolist()),
            outdir=outdir,
        )


# =============================================================================
# Fig. 10: nonlinear Landau damping phase-space snapshots
# =============================================================================

def simulate_nonlinear_landau_method(
    params: NonlinearLandauParams,
    method: str,
    *,
    alpha: Optional[int] = None,
    nu: Optional[float] = None,
    chi_over_dt: Optional[float] = None,
    mu: Optional[float] = None,
    learned_closure: Optional[LearnedInterfaceClosure] = None,
) -> Dict[str, np.ndarray]:
    """
    Nonlinear Landau damping in the shared Fourier-Hermite perturbation formulation.
    """
    raw = run_nonlinear_landau_rollout_raw(
        params,
        method,
        alpha=alpha,
        nu=nu,
        chi_over_dt=chi_over_dt,
        mu=mu,
        learned_closure=learned_closure,
        return_state_history=False,
    )

    x = np.asarray(raw["x"], dtype=float)
    v = np.linspace(params.v_range[0], params.v_range[1], int(params.Nv_plot))
    phi = hermite_basis_phi(int(params.Nv), v)
    snaps_phys = np.asarray(raw["snapshot_a_phys"], dtype=float)
    m_eq = np.asarray(raw["m_eq"], dtype=float)

    out: Dict[str, np.ndarray] = {
        "x": x,
        "v": v,
        "times": np.asarray(params.snapshot_times, dtype=float),
    }
    for idx, t in enumerate(params.snapshot_times):
        full_f = (snaps_phys[idx] + m_eq[:, None]).T @ phi
        out[f"f_{_time_key(float(t))}"] = full_f.T.astype(float)
    return out


@dataclass
class Fig10NonlinearLandauPhaseSpace(Benchmark):
    """
    Paper-style Figure 10 layout without the unavailable reference-solution panel.
    """
    name: str = "fig10_nonlinear_landau_phase_space"
    params: NonlinearLandauParams = field(default_factory=NonlinearLandauParams)
    hyper_nu_a1: float = 0.55
    hyper_nu_a2: float = 1.312
    hyper_nu_a3: float = 2.013
    filter_chi_over_dt: float = 12.228
    nonlocal_mu_nm1: float = -1.017234

    def run(self) -> BenchmarkResult:
        params = self.params
        method_specs = {
            "truncation": dict(method="truncation"),
            "nonlocal_nm1": dict(method="nonlocal", mu=self.nonlocal_mu_nm1),
            "hyper_a1": dict(method="hyper", alpha=1, nu=self.hyper_nu_a1),
            "hyper_a2": dict(method="hyper", alpha=2, nu=self.hyper_nu_a2),
            "hyper_a3": dict(method="hyper", alpha=3, nu=self.hyper_nu_a3),
            "filter": dict(method="filter", chi_over_dt=self.filter_chi_over_dt),
        }

        payload: Dict[str, np.ndarray] = {
            "x": np.empty((0,), dtype=float),
            "v": np.empty((0,), dtype=float),
            "times": np.asarray(params.snapshot_times, dtype=float),
        }
        for name, spec in method_specs.items():
            data = simulate_nonlinear_landau_method(params, **spec)
            if payload["x"].size == 0:
                payload["x"] = data["x"]
                payload["v"] = data["v"]
            for key, value in data.items():
                if key in {"x", "v", "times"}:
                    continue
                payload[f"{name}_{key}"] = value
        return BenchmarkResult(self.name, payload)

    def plot(self, result: BenchmarkResult, outdir: Union[str, Path]) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        save_fig10_nonlinear_landau_phase_space(
            result.payload,
            times=tuple(float(t) for t in np.asarray(result.payload["times"], dtype=float)),
            vmin=float(self.params.vmin),
            vmax=float(self.params.vmax),
            time_key_fn=_time_key,
            outdir=outdir,
        )


@dataclass
class Fig10LearnedComparisonPhaseSpace(Benchmark):
    """
    Learned-vs-baseline nonlinear Landau phase-space comparison with four panels.
    """
    name: str = "fig10_learned_comparison_phase_space"
    params: NonlinearLandauParams = field(default_factory=NonlinearLandauParams)
    nonlocal_mu_nm1: float = -1.017234
    learned_checkpoint: Optional[str] = None

    def run(self) -> BenchmarkResult:
        if self.learned_checkpoint is None:
            raise ValueError("learned_checkpoint must be provided for fig10_learned_comparison")

        params = self.params
        learned_closure = load_learned_interface_closure_npz(self.learned_checkpoint)
        method_specs = {
            "nonlocal_nm1": dict(method="nonlocal", mu=self.nonlocal_mu_nm1),
            "learned": dict(method="learned", learned_closure=learned_closure),
        }

        payload: Dict[str, np.ndarray] = {
            "x": np.empty((0,), dtype=float),
            "v": np.empty((0,), dtype=float),
            "times": np.asarray(params.snapshot_times, dtype=float),
        }
        for name, spec in method_specs.items():
            data = simulate_nonlinear_landau_method(params, **spec)
            if payload["x"].size == 0:
                payload["x"] = data["x"]
                payload["v"] = data["v"]
            for key, value in data.items():
                if key in {"x", "v", "times"}:
                    continue
                payload[f"{name}_{key}"] = value
        return BenchmarkResult(self.name, payload)

    def plot(self, result: BenchmarkResult, outdir: Union[str, Path]) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        save_fig10_learned_comparison_phase_space(
            result.payload,
            times=tuple(float(t) for t in np.asarray(result.payload["times"], dtype=float)),
            vmin=float(self.params.vmin),
            vmax=float(self.params.vmax),
            time_key_fn=_time_key,
            outdir=outdir,
            baseline_name="nonlocal_nm1",
            baseline_title=r"Nonlocal closure ($N_m=1$)",
            learned_title="Learned interface closure",
        )

# =============================================================================
# Optional: linear Landau damping time simulation using the shared IMEX integrator
# =============================================================================

@dataclass
class LinearLandauTimeBenchmark(Benchmark):
    """
    A Section 4.1-style time simulation in the AW Hermite basis using implicit midpoint
    with a Jacobian-free Newton-Krylov solve.

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
    - "learned": learned interface closure from a checkpoint

    Important
    ---------
    This benchmark is for comparing *time-domain* damping curves, not eigenvalue scans.
    Classical methods use the paper-style implicit-midpoint/JFNK path. The learned
    interface closure is advanced with the shared CNAB2 solver so the closure enters
    through the same explicit interface-term insertion used in the nonlinear runs.
    """
    name: str = "linear_landau_time"
    method: str = "truncation"
    alpha: int = 2
    nu: float = 16.76
    chi_over_dt: float = 7.56
    mu: float = -1.01
    p: int = 36
    learned_checkpoint: Optional[str] = None

    Nv: int = 20
    Nx: int = 10
    L: float = 4 * math.pi
    dt: float = 1e-2
    T: float = 40.0
    eps: float = 1e-2
    modes: Tuple[float, ...] = (0.5, 1.5)
    newton_tol: float = 1e-8
    gmres_tol: float = 1e-8
    max_newton_iter: int = 10
    gmres_restart: Optional[int] = None
    gmres_maxiter: Optional[int] = None

    poisson_sign: float = +1.0

    def run(self) -> BenchmarkResult:
        learned_closure = None
        if self.method == "learned":
            if self.learned_checkpoint is None:
                raise ValueError("learned_checkpoint must be provided when method='learned'")
            learned_closure = load_learned_interface_closure_npz(self.learned_checkpoint)
        config = LinearLandauConfig(
            method=self.method,
            alpha=self.alpha,
            nu=self.nu,
            chi_over_dt=self.chi_over_dt,
            mu=self.mu,
            p=self.p,
            Nv=self.Nv,
            Nx=self.Nx,
            L=self.L,
            dt=self.dt,
            T=self.T,
            eps=self.eps,
            modes=tuple(float(k) for k in self.modes),
            poisson_sign=self.poisson_sign,
            newton_tol=self.newton_tol,
            gmres_tol=self.gmres_tol,
            max_newton_iter=self.max_newton_iter,
            gmres_restart=self.gmres_restart,
            gmres_maxiter=self.gmres_maxiter,
        )
        payload = run_linear_landau_rollout(
            config,
            learned_closure=learned_closure,
            solver_backend="cnab2" if self.method == "learned" else "jfnk",
        )
        return BenchmarkResult(self.name + "_" + self.method, payload)

    def plot(self, result: BenchmarkResult, outdir: Union[str, Path]) -> None:
        save_linear_landau_time(result.payload, method=self.method, outdir=outdir)


def plot_linear_landau_comparison(
    results: Dict[str, BenchmarkResult],
    outdir: Union[str, Path],
) -> None:
    save_linear_landau_comparison_figure(results, outdir)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    print_jax_runtime_summary(jax, context="benchmarks")
    parser = argparse.ArgumentParser(description="JAX benchmarks for arXiv:2412.07073")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p2 = sub.add_parser("fig2", help="Damping profiles (Fig. 2 style)")
    p2.add_argument("--outdir", type=str, default="out_bench")

    p3 = sub.add_parser("fig3", help="Response-function benchmarks (Fig. 3 style)")
    p3.add_argument("--outdir", type=str, default="out_bench")
    p3.add_argument("--n_xi", type=int, default=100_000)
    p3.add_argument("--learned-checkpoint", type=str, default=None)

    p4 = sub.add_parser("fig4", help="Eigenvalue scan via discrete VP eigenvalues (Fig. 4/5 style)")
    p4.add_argument("--outdir", type=str, default="out_bench")
    p4.add_argument("--Nv", type=int, default=20)
    p4.add_argument("--learned-checkpoint", type=str, default=None)

    p10 = sub.add_parser("fig10", help="Classical nonlinear Landau phase-space snapshots (Fig. 10 style, no learned panel)")
    p10.add_argument("--outdir", type=str, default="out_bench")
    p10.add_argument("--Nx", type=int, default=200)
    p10.add_argument("--Nv", type=int, default=300)
    p10.add_argument("--dt", type=float, default=1e-2)
    p10.add_argument("--T", type=float, default=40.0)

    p10l = sub.add_parser("fig10_learned_comparison", help="Four-panel nonlinear Landau comparison: nonlocal vs learned")
    p10l.add_argument("--outdir", type=str, default="out_bench")
    p10l.add_argument("--Nx", type=int, default=200)
    p10l.add_argument("--Nv", type=int, default=300)
    p10l.add_argument("--dt", type=float, default=1e-2)
    p10l.add_argument("--T", type=float, default=40.0)
    p10l.add_argument("--learned-checkpoint", type=str, required=True)

    pl = sub.add_parser("linear_landau", help="Linear Landau damping time simulation via implicit midpoint/JFNK")
    pl.add_argument("--outdir", type=str, default="out_bench")
    pl.add_argument("--method", choices=["truncation", "hyper", "filter", "nonlocal", "learned"], default="truncation")
    pl.add_argument("--alpha", type=int, default=2)
    pl.add_argument("--nu", type=float, default=16.76)
    pl.add_argument("--chi_over_dt", type=float, default=7.56)
    pl.add_argument("--mu", type=float, default=-1.01)
    pl.add_argument("--T", type=float, default=40.0)
    pl.add_argument("--learned-checkpoint", type=str, default=None)

    args = parser.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "fig2":
        b = Fig2DampingProfiles()
        res = b.run()
        res.save_npz(outdir / "fig2_damping_profiles.npz")
        b.plot(res, outdir)

    elif args.cmd == "fig3":
        b = Fig3ResponseFunction(n_xi=args.n_xi, learned_checkpoint=args.learned_checkpoint)
        res = b.run()
        res.save_npz(outdir / "fig3_response_function.npz")
        b.plot(res, outdir)

    elif args.cmd == "fig4":
        b = Fig4EigenvalueScan(Nv=args.Nv, learned_checkpoint=args.learned_checkpoint)
        res = b.run()
        res.save_npz(outdir / f"fig4_eigen_scan_Nv{args.Nv}.npz")
        b.plot(res, outdir)

    elif args.cmd == "fig10":
        p = NonlinearLandauParams(Nx=args.Nx, Nv=args.Nv, dt=args.dt, T=args.T)
        b = Fig10NonlinearLandauPhaseSpace(params=p)
        res = b.run()
        res.save_npz(outdir / "fig10_nonlinear_landau_phase_space.npz")
        b.plot(res, outdir)

    elif args.cmd == "fig10_learned_comparison":
        p = NonlinearLandauParams(Nx=args.Nx, Nv=args.Nv, dt=args.dt, T=args.T)
        b = Fig10LearnedComparisonPhaseSpace(params=p, learned_checkpoint=args.learned_checkpoint)
        res = b.run()
        res.save_npz(outdir / "fig10_learned_vs_nonlocal_phase_space.npz")
        b.plot(res, outdir)

    elif args.cmd == "linear_landau":
        b = LinearLandauTimeBenchmark(method=args.method, alpha=args.alpha, nu=args.nu,
                                      chi_over_dt=args.chi_over_dt, mu=args.mu, T=args.T,
                                      learned_checkpoint=args.learned_checkpoint)
        res = b.run()
        res.save_npz(outdir / f"linear_landau_{args.method}.npz")
        b.plot(res, outdir)
        if args.learned_checkpoint is not None:
            comparison_results: Dict[str, BenchmarkResult] = {args.method: res}
            for method in ["truncation", "hyper", "filter", "nonlocal", "learned"]:
                if method in comparison_results:
                    continue
                bench = LinearLandauTimeBenchmark(
                    method=method,
                    alpha=args.alpha,
                    nu=args.nu,
                    chi_over_dt=args.chi_over_dt,
                    mu=args.mu,
                    T=args.T,
                    learned_checkpoint=args.learned_checkpoint,
                )
                comparison_results[method] = bench.run()
            plot_linear_landau_comparison(comparison_results, outdir)


if __name__ == "__main__":
    main()
