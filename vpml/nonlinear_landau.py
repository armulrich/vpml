"""Shared nonlinear-Landau rollout utilities for benchmarks and model evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

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
    hermite_damping_term,
    irfft_x,
    learned_boundary_flux_hat,
    rfft_x,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


@dataclass(frozen=True)
class NonlinearLandauParams:
    Nx: int = 200
    Nv: int = 300
    L: float = 4.0 * math.pi
    dt: float = 1e-2
    T: float = 40.0
    eps: float = 0.5
    k0: float = 0.5
    vth: float = 1.0
    dealias_23: bool = False
    poisson_sign: float = +1.0
    snapshot_times: Tuple[float, ...] = (20.0, 40.0)
    v_range: Tuple[float, float] = (-4.0, 4.0)
    Nv_plot: int = 1000
    vmin: float = 0.0
    vmax: float = 0.5


def _snapshot_indices(snapshot_times: Sequence[float], dt: float) -> np.ndarray:
    return np.asarray([int(round(float(t) / float(dt))) for t in snapshot_times], dtype=np.int32)


def _time_key(t: float) -> str:
    t_str = f"{float(t):g}".replace("-", "m").replace(".", "p")
    return f"t{t_str}"


def run_nonlinear_landau_rollout_raw(
    params: NonlinearLandauParams,
    method: str,
    *,
    alpha: Optional[int] = None,
    nu: Optional[float] = None,
    chi_over_dt: Optional[float] = None,
    mu: Optional[float] = None,
    learned_closure: Optional[LearnedInterfaceClosure] = None,
    return_state_history: bool = False,
    history_stride: int = 1,
) -> Dict[str, np.ndarray | Array]:
    """Advance nonlinear Landau damping with the shared Fourier-Hermite CNAB2 solver."""
    closure = None
    damping = None
    if method == "hyper":
        if alpha is None or nu is None:
            raise ValueError("hyper method requires alpha and nu")
        damping = HyperCollisions(alpha=int(alpha), nu=float(nu), Nv=int(params.Nv))
    elif method == "filter":
        if chi_over_dt is None:
            raise ValueError("filter method requires chi_over_dt")
        damping = HouLiFilter(chi_over_dt=float(chi_over_dt), Nv=int(params.Nv), p=36)
    elif method == "nonlocal":
        if mu is None:
            raise ValueError("nonlocal method requires mu")
        closure = NonlocalClosure(mu_tail=jnp.array([float(mu)], dtype=jnp.float64))
    elif method == "learned":
        if learned_closure is None:
            raise ValueError("learned method requires learned_closure")
    elif method != "truncation":
        raise ValueError(f"Unknown nonlinear Landau method: {method}")

    history_stride = max(int(history_stride), 1)

    integ = FourierHermiteIMEX(
        Nx=int(params.Nx),
        Nv=int(params.Nv),
        Lx=float(params.L),
        dt=float(params.dt),
        vth=float(params.vth),
        dealias_23=bool(params.dealias_23),
        closure=closure,
    )
    m_eq = jnp.zeros((int(params.Nv),), dtype=jnp.float64).at[0].set(1.0)
    a_phys0 = jnp.zeros((int(params.Nv), int(params.Nx)), dtype=jnp.float64)
    a_phys0 = a_phys0.at[0].set(float(params.eps) * jnp.cos(float(params.k0) * integ.x))
    a_hat0 = integ.apply_mask_hat(rfft_x(a_phys0))

    damping_rates = None
    if damping is not None:
        damping_rates = damping.damping_rates().astype(jnp.float64)

    def explicit_n_hat(a_hat: Array) -> Array:
        a_phys = irfft_x(a_hat, int(params.Nx))
        E = integ.E_phys_from_a_hat(a_hat, poisson_sign=float(params.poisson_sign))
        N_phys = jnp.zeros_like(a_phys)
        N_phys = N_phys.at[1:].set(
            -(integ.sqrt_n[1:, None] / float(params.vth))
            * E[None, :]
            * (a_phys[:-1] + m_eq[:-1, None])
        )
        if damping_rates is not None:
            N_phys = N_phys + hermite_damping_term(a_phys, damping_rates)
        return integ.apply_mask_hat(rfft_x(N_phys))

    nsteps = int(round(float(params.T) / float(params.dt)))
    snap_steps = _snapshot_indices(params.snapshot_times, params.dt)
    snaps0 = jnp.zeros((len(snap_steps), int(params.Nv), int(params.Nx)), dtype=jnp.float64)
    if 0 in snap_steps:
        snap0_idx = int(np.where(snap_steps == 0)[0][0])
        snaps0 = snaps0.at[snap0_idx].set(irfft_x(a_hat0, int(params.Nx)))

    hist_steps = np.arange(0, nsteps + 1, history_stride, dtype=np.int32)
    if hist_steps[-1] != nsteps:
        hist_steps = np.concatenate([hist_steps, np.asarray([nsteps], dtype=np.int32)])
    hist0 = None
    if return_state_history:
        hist0 = jnp.zeros((len(hist_steps), int(params.Nv), int(integ.Nk)), dtype=jnp.complex128)
        hist0 = hist0.at[0].set(a_hat0)

    n0 = explicit_n_hat(a_hat0)
    b0 = (
        learned_boundary_flux_hat(a_hat0, integ.k_arr, integ.Nv, integ.vth, learned_closure)
        if learned_closure is not None
        else jnp.zeros_like(a_hat0)
    )

    def maybe_store(snaps: Array, step_i: Array, a_hat_new: Array) -> Array:
        a_phys_new = irfft_x(a_hat_new, int(params.Nx))
        for j, snap_step in enumerate(snap_steps):
            snaps = jax.lax.cond(
                step_i == int(snap_step),
                lambda s, arr=a_phys_new, idx=j: s.at[idx].set(arr),
                lambda s: s,
                snaps,
            )
        return snaps

    def maybe_store_history(history: Array, step_i: Array, a_hat_new: Array) -> Array:
        if not return_state_history:
            return history
        history = jax.lax.cond(
            (step_i % history_stride) == 0,
            lambda h: h.at[step_i // history_stride].set(a_hat_new),
            lambda h: h,
            history,
        )
        if int(hist_steps[-1]) != nsteps:
            history = jax.lax.cond(
                step_i == int(nsteps),
                lambda h: h.at[len(hist_steps) - 1].set(a_hat_new),
                lambda h: h,
                history,
            )
        return history

    def step(carry, i):
        a_hat, n_prev, b_prev, snaps, history = carry
        n_hat = explicit_n_hat(a_hat)
        b_hat = (
            jnp.zeros_like(a_hat)
            if learned_closure is None
            else learned_boundary_flux_hat(a_hat, integ.k_arr, integ.Nv, integ.vth, learned_closure)
        )
        a_new = integ.step_cnab2(
            a_hat,
            n_hat,
            n_prev,
            extra_hat=b_hat,
            extra_hat_prev=b_prev,
        )
        snaps = maybe_store(snaps, i, a_new)
        history = maybe_store_history(history, i, a_new)
        return (a_new, n_hat, b_hat, snaps, history), 0.0

    (a_last, n_last, b_last, snaps_out, hist_out), _ = jax.lax.scan(
        step,
        (a_hat0, n0, b0, snaps0, hist0),
        jnp.arange(1, nsteps + 1, dtype=jnp.int32),
    )
    del a_last, n_last, b_last

    raw: Dict[str, np.ndarray | Array] = {
        "x": np.asarray(integ.x),
        "snapshot_times": np.asarray(params.snapshot_times, dtype=float),
        "snapshot_a_phys": np.asarray(snaps_out),
        "m_eq": np.asarray(m_eq),
        "k_arr": np.asarray(integ.k_arr, dtype=float),
    }
    if return_state_history and hist_out is not None:
        raw["a_hat_hist"] = hist_out
        raw["a_hat_hist_times"] = hist_steps.astype(float) * float(params.dt)
    return raw


__all__ = [
    "NonlinearLandauParams",
    "run_nonlinear_landau_rollout_raw",
]
