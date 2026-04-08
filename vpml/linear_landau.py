"""Shared linear-Landau rollout utilities for benchmarks and training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .jax_runtime import bootstrap_jax_runtime

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np

from .core import (
    Array,
    FourierHermiteIMEX,
    HouLiFilter,
    HyperCollisions,
    LearnedInterfaceClosure,
    NonlocalClosure,
    e_hat_history_from_a_hat_history,
    implicit_midpoint_jfnk_step,
    irfft_x,
    learned_boundary_flux_hat,
    rfft_x,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


@dataclass(frozen=True)
class LinearLandauConfig:
    """Configuration for the paper-style linear Landau damping benchmark."""

    method: str = "truncation"
    alpha: int = 2
    nu: float = 16.76
    chi_over_dt: float = 7.56
    mu: float = -1.01
    p: int = 36
    Nv: int = 20
    Nx: int = 10
    L: float = 4.0 * math.pi
    dt: float = 1e-2
    T: float = 40.0
    eps: float = 1e-2
    modes: Tuple[float, ...] = (0.5, 1.5)
    poisson_sign: float = +1.0
    newton_tol: float = 1e-8
    gmres_tol: float = 1e-8
    max_newton_iter: int = 10
    gmres_restart: Optional[int] = None
    gmres_maxiter: Optional[int] = None


def _flatten_hat_real(a_hat: Array) -> Array:
    arr = jnp.asarray(a_hat, dtype=jnp.complex128)
    return jnp.concatenate(
        [jnp.real(arr).reshape(-1), jnp.imag(arr).reshape(-1)],
        axis=0,
    )


def _unflatten_hat_real(vec: Array, shape: Tuple[int, int]) -> Array:
    vec = jnp.asarray(vec, dtype=jnp.float64)
    size = int(np.prod(shape))
    re = vec[:size].reshape(shape)
    im = vec[size:].reshape(shape)
    return (re + 1j * im).astype(jnp.complex128)


def _resolve_method_components(
    config: LinearLandauConfig,
    learned_closure: Optional[LearnedInterfaceClosure],
) -> Tuple[Optional[HyperCollisions | HouLiFilter], Optional[NonlocalClosure]]:
    if config.method == "hyper":
        return HyperCollisions(alpha=config.alpha, nu=config.nu, Nv=config.Nv), None
    if config.method == "filter":
        return HouLiFilter(chi_over_dt=config.chi_over_dt, Nv=config.Nv, p=config.p), None
    if config.method == "nonlocal":
        closure = NonlocalClosure(mu_tail=jnp.array([config.mu], dtype=jnp.float64))
        return None, closure
    if config.method == "learned":
        if learned_closure is None:
            raise ValueError("learned_closure must be provided when method='learned'")
        return None, None
    if config.method == "truncation":
        return None, None
    raise ValueError(f"Unknown linear Landau method: {config.method}")


def build_linear_landau_problem(
    config: LinearLandauConfig,
    *,
    learned_closure: Optional[LearnedInterfaceClosure] = None,
) -> Dict[str, object]:
    """Construct the common linear-Landau benchmark state and RHS."""
    dissipation, closure = _resolve_method_components(config, learned_closure)
    integ = FourierHermiteIMEX(
        Nx=config.Nx,
        Nv=config.Nv,
        Lx=config.L,
        dt=config.dt,
        vth=1.0,
        dealias_23=False,
        closure=closure,
    )

    m_eq = jnp.zeros((config.Nv,), dtype=jnp.float64).at[0].set(1.0)
    a0 = jnp.zeros((config.Nx,), dtype=jnp.float64)
    for k in config.modes:
        a0 = a0 + jnp.cos(float(k) * integ.x)
    a0 = float(config.eps) * a0

    a_phys0 = jnp.zeros((config.Nv, config.Nx), dtype=jnp.float64).at[0].set(a0)
    a_hat0 = rfft_x(a_phys0)

    def explicit_N_hat(a_hat: Array) -> Array:
        a_phys = irfft_x(a_hat, config.Nx)
        E = integ.E_phys_from_a_hat(a_hat, poisson_sign=config.poisson_sign)

        N_phys = jnp.zeros_like(a_phys)
        N_phys = N_phys.at[1:].set(-integ.sqrt_n[1:, None] * E[None, :] * m_eq[:-1, None])

        if dissipation is not None:
            gamma = dissipation.damping_rates().astype(jnp.float64)
            N_phys = N_phys - gamma[:, None] * a_phys

        return integ.apply_mask_hat(rfft_x(N_phys))

    def full_rhs_hat(a_hat: Array) -> Array:
        rhs = integ.streaming_hat(a_hat) + explicit_N_hat(a_hat)
        if learned_closure is not None:
            rhs = rhs + learned_boundary_flux_hat(
                a_hat,
                integ.k_arr,
                integ.Nv,
                integ.vth,
                learned_closure,
            )
        return rhs

    return {
        "integrator": integ,
        "a_hat0": jnp.asarray(a_hat0, dtype=jnp.complex128),
        "full_rhs_hat": jax.jit(full_rhs_hat),
        "nsteps": int(round(float(config.T) / float(config.dt))),
    }


def _linear_midpoint_step_matrix(
    full_rhs_hat,
    shape: Tuple[int, int],
    dt: float,
) -> Array:
    zero = jnp.zeros(shape, dtype=jnp.complex128)

    def rhs_real(vec: Array) -> Array:
        return _flatten_hat_real(full_rhs_hat(_unflatten_hat_real(vec, shape)))

    A = jax.jacfwd(rhs_real)(_flatten_hat_real(zero))
    I = jnp.eye(A.shape[0], dtype=jnp.float64)
    lhs = I - 0.5 * float(dt) * A
    rhs = I + 0.5 * float(dt) * A
    return jnp.linalg.solve(lhs, rhs)


def linear_explicit_N_hat(
    a_hat: Array,
    integ: FourierHermiteIMEX,
    m_eq: Array,
    *,
    poisson_sign: float,
    dissipation: Optional[HyperCollisions | HouLiFilter] = None,
) -> Array:
    """Explicit source term for the linear perturbation hierarchy."""
    a_phys = irfft_x(a_hat, integ.Nx)
    E = integ.E_phys_from_a_hat(a_hat, poisson_sign=poisson_sign)
    N_phys = jnp.zeros_like(a_phys)
    N_phys = N_phys.at[1:].set(-integ.sqrt_n[1:, None] * E[None, :] * m_eq[:-1, None])

    if dissipation is not None:
        gamma = dissipation.damping_rates().astype(jnp.float64)
        N_phys = N_phys - gamma[:, None] * a_phys

    return integ.apply_mask_hat(rfft_x(N_phys))


def run_linear_landau_cnab2_raw(
    config: LinearLandauConfig,
    *,
    learned_closure: Optional[LearnedInterfaceClosure] = None,
    return_state_history: bool = False,
) -> Dict[str, np.ndarray | Array]:
    """Advance the linear Landau problem with the shared IMEX CNAB2 solver."""
    dissipation, closure = _resolve_method_components(config, learned_closure)
    integ = FourierHermiteIMEX(
        Nx=config.Nx,
        Nv=config.Nv,
        Lx=config.L,
        dt=config.dt,
        vth=1.0,
        dealias_23=False,
        closure=closure,
    )

    m_eq = jnp.zeros((config.Nv,), dtype=jnp.float64).at[0].set(1.0)
    a0 = jnp.zeros((config.Nx,), dtype=jnp.float64)
    for k in config.modes:
        a0 = a0 + jnp.cos(float(k) * integ.x)
    a0 = float(config.eps) * a0
    a_phys0 = jnp.zeros((config.Nv, config.Nx), dtype=jnp.float64).at[0].set(a0)
    a_hat0 = integ.apply_mask_hat(rfft_x(a_phys0))

    N0 = linear_explicit_N_hat(
        a_hat0,
        integ,
        m_eq,
        poisson_sign=config.poisson_sign,
        dissipation=dissipation,
    )
    B0 = (
        jnp.zeros_like(a_hat0)
        if learned_closure is None
        else learned_boundary_flux_hat(a_hat0, integ.k_arr, integ.Nv, integ.vth, learned_closure)
    )
    nsteps = int(round(float(config.T) / float(config.dt)))

    def step(carry, _):
        a_hat, N_prev, B_prev = carry
        N_hat = linear_explicit_N_hat(
            a_hat,
            integ,
            m_eq,
            poisson_sign=config.poisson_sign,
            dissipation=dissipation,
        )
        B_hat = (
            jnp.zeros_like(a_hat)
            if learned_closure is None
            else learned_boundary_flux_hat(a_hat, integ.k_arr, integ.Nv, integ.vth, learned_closure)
        )
        a_new = integ.step_cnab2(
            a_hat,
            N_hat,
            N_prev,
            extra_hat=B_hat,
            extra_hat_prev=B_prev,
        )
        return (a_new, N_hat, B_hat), a_new

    (_, _, _), states = jax.lax.scan(step, (a_hat0, N0, B0), xs=None, length=nsteps)
    a_hat_hist = jnp.concatenate([a_hat0[None, :, :], states], axis=0)
    Ehat_hist = _extract_ehat_history(a_hat_hist, integ, poisson_sign=config.poisson_sign)
    raw: Dict[str, np.ndarray | Array] = {"E_hat_hist": Ehat_hist}
    if return_state_history:
        raw["a_hat_hist"] = a_hat_hist
    return raw


def _extract_ehat_history(
    a_hat_hist: Array,
    integ: FourierHermiteIMEX,
    *,
    poisson_sign: float,
) -> Array:
    return e_hat_history_from_a_hat_history(
        a_hat_hist,
        integ.k_arr,
        poisson_sign=poisson_sign,
    )


def _landau_mode_index(integ: FourierHermiteIMEX, mode: float) -> int:
    idx = int(np.argmin(np.abs(np.asarray(integ.k_arr) - float(mode))))
    if not np.isclose(float(np.asarray(integ.k_arr)[idx]), float(mode), atol=1e-12):
        raise ValueError(f"Mode {mode} is not represented on the configured Fourier grid")
    return idx


def linear_landau_payload(
    config: LinearLandauConfig,
    Ehat_hist: Array,
    *,
    newton_hist: Optional[np.ndarray] = None,
    gmres_hist: Optional[np.ndarray] = None,
    residual_hist: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Build benchmark-style observables from a rollout history."""
    from benchmarks.fh_benchmarks_2412_07073_jax import landau_gamma, solve_landau_root_xi

    integ = FourierHermiteIMEX(
        Nx=config.Nx,
        Nv=config.Nv,
        Lx=config.L,
        dt=config.dt,
        vth=1.0,
        dealias_23=False,
        closure=None,
    )
    k05_idx = _landau_mode_index(integ, 0.5)
    k15_idx = _landau_mode_index(integ, 1.5)

    nframes = int(np.asarray(Ehat_hist).shape[0])
    times = np.linspace(0.0, int(nframes - 1) * float(config.dt), nframes)
    xi05 = solve_landau_root_xi(0.5)
    xi15 = solve_landau_root_xi(1.5)
    payload = {
        "times": times,
        "E_abs_k0p5": np.abs(np.asarray(Ehat_hist[:, k05_idx])),
        "E_abs_k1p5": np.abs(np.asarray(Ehat_hist[:, k15_idx])),
        "gamma_k0p5": np.array([landau_gamma(0.5, xi05)]),
        "gamma_k1p5": np.array([landau_gamma(1.5, xi15)]),
    }
    if newton_hist is not None:
        payload["newton_iters"] = np.asarray(newton_hist, dtype=float)
    if gmres_hist is not None:
        payload["gmres_iters"] = np.asarray(gmres_hist, dtype=float)
    if residual_hist is not None:
        payload["solver_residual_norm"] = np.asarray(residual_hist, dtype=float)
    return payload


def run_linear_landau_rollout_raw(
    config: LinearLandauConfig,
    *,
    learned_closure: Optional[LearnedInterfaceClosure] = None,
    solver_backend: str = "jfnk",
    return_state_history: bool = False,
) -> Dict[str, np.ndarray | Array]:
    """
    Advance the shared linear-Landau problem with a chosen implicit-midpoint backend.

    `solver_backend="jfnk"` matches the benchmark path.
    `solver_backend="linear_direct"` exploits linearity to build the exact midpoint step
    matrix in a JAX-differentiable way for training.
    """
    if config.method == "learned":
        if solver_backend != "cnab2":
            raise ValueError(
                "The learned interface closure is state-dependent and must be advanced with "
                "solver_backend='cnab2'."
            )
        return run_linear_landau_cnab2_raw(
            config,
            learned_closure=learned_closure,
            return_state_history=return_state_history,
        )

    problem = build_linear_landau_problem(config, learned_closure=learned_closure)
    integ = problem["integrator"]
    a_hat0 = problem["a_hat0"]
    full_rhs_hat = problem["full_rhs_hat"]
    nsteps = int(problem["nsteps"])

    if solver_backend == "jfnk":
        Ehat0 = np.asarray(
            integ.E_hat_from_rho_hat(a_hat0[0], poisson_sign=config.poisson_sign)
        )
        a_curr = jnp.asarray(a_hat0, dtype=jnp.complex128)
        Ehat_hist = [Ehat0]
        state_hist = [np.asarray(a_curr)] if return_state_history else None
        newton_hist = np.zeros(nsteps, dtype=float)
        gmres_hist = np.zeros(nsteps, dtype=float)
        residual_hist = np.zeros(nsteps, dtype=float)

        for step_idx in range(nsteps):
            predictor = a_curr + float(config.dt) * full_rhs_hat(a_curr)
            a_next, solver_info = implicit_midpoint_jfnk_step(
                a_curr,
                full_rhs_hat,
                config.dt,
                initial_guess=predictor,
                newton_tol=config.newton_tol,
                gmres_tol=config.gmres_tol,
                max_newton_iter=config.max_newton_iter,
                gmres_restart=config.gmres_restart,
                gmres_maxiter=config.gmres_maxiter,
            )
            Ehat = np.asarray(
                integ.E_hat_from_rho_hat(a_next[0], poisson_sign=config.poisson_sign)
            )
            Ehat_hist.append(Ehat)
            newton_hist[step_idx] = solver_info["newton_iters"]
            gmres_hist[step_idx] = solver_info["gmres_iters"]
            residual_hist[step_idx] = solver_info["residual_norm"]
            a_curr = a_next
            if state_hist is not None:
                state_hist.append(np.asarray(a_curr))

        raw: Dict[str, np.ndarray | Array] = {
            "E_hat_hist": jnp.asarray(np.asarray(Ehat_hist), dtype=jnp.complex128),
            "newton_iters": np.asarray(newton_hist, dtype=float),
            "gmres_iters": np.asarray(gmres_hist, dtype=float),
            "solver_residual_norm": np.asarray(residual_hist, dtype=float),
        }
        if state_hist is not None:
            raw["a_hat_hist"] = jnp.asarray(np.asarray(state_hist), dtype=jnp.complex128)
        return raw

    if solver_backend != "linear_direct":
        raise ValueError(f"Unknown linear Landau solver backend: {solver_backend}")

    shape = tuple(int(v) for v in a_hat0.shape)
    step_matrix = _linear_midpoint_step_matrix(full_rhs_hat, shape, config.dt)
    flat0 = _flatten_hat_real(a_hat0)

    def step(flat_state, _):
        flat_next = step_matrix @ flat_state
        return flat_next, flat_next

    _, flat_hist = jax.lax.scan(step, flat0, xs=None, length=nsteps)
    flat_hist = jnp.concatenate([flat0[None, :], flat_hist], axis=0)
    a_hat_hist = jax.vmap(lambda v: _unflatten_hat_real(v, shape))(flat_hist)
    Ehat_hist = _extract_ehat_history(a_hat_hist, integ, poisson_sign=config.poisson_sign)
    raw = {"E_hat_hist": Ehat_hist}
    if return_state_history:
        raw["a_hat_hist"] = a_hat_hist
    return raw


def run_linear_landau_rollout(
    config: LinearLandauConfig,
    *,
    learned_closure: Optional[LearnedInterfaceClosure] = None,
    solver_backend: str = "jfnk",
    return_state_history: bool = False,
) -> Dict[str, np.ndarray | Array]:
    """Return benchmark-style observables for a linear-Landau rollout."""
    raw = run_linear_landau_rollout_raw(
        config,
        learned_closure=learned_closure,
        solver_backend=solver_backend,
        return_state_history=return_state_history,
    )
    payload = linear_landau_payload(
        config,
        raw["E_hat_hist"],
        newton_hist=raw.get("newton_iters"),
        gmres_hist=raw.get("gmres_iters"),
        residual_hist=raw.get("solver_residual_norm"),
    )
    payload["E_hat_hist"] = raw["E_hat_hist"]
    if return_state_history and "a_hat_hist" in raw:
        payload["a_hat_hist"] = raw["a_hat_hist"]
    return payload
