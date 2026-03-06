"""
Nonlinear 1D1V Vlasov–Poisson simulations in **JAX** using the core Fourier–Hermite IMEX
integrator in ``vpml.core``.

This script ports the provided numpy "two-stream + bump-on-tail" solver to JAX and keeps
the same mathematical formulation (perturbation form, Hermite truncation, CNAB2, optional
dealiasing, optional exponential Hermite filter, and (for bump-on-tail) the System C
pole-elimination control field).

What this script is for
-----------------------
- Run nonlinear two-stream instability (System A, H=0).
- Run nonlinear bump-on-tail instability (System A, H=0) and System C (controlled).

"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Matplotlib cache directory fix for sandboxed environments (optional)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from vpml.core import (
    Array,
    FourierHermiteIMEX,
    HermiteExponentialFilter,
    rfft_x,
    irfft_x,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


# =============================================================================
# Hermite / equilibrium helpers (JAX)
# =============================================================================

def equilibrium_coeffs_two_stream(N: int, vbar: float) -> Array:
    """
    Two-stream equilibrium:
        μ(v) = 0.5 N(vbar,1) + 0.5 N(-vbar,1)

    In the normalized Hermite basis φ_n(v) = w(v) He_n(v)/sqrt(n!),
    the coefficients are:
        m_n = vbar^n / sqrt(n!)  for even n, else 0.
    """
    n = jnp.arange(N, dtype=jnp.float64)
    log_fact = jsp.special.gammaln(n + 1.0)
    coeff = jnp.exp(n * jnp.log(float(vbar)) - 0.5 * log_fact)
    even = (n % 2) == 0
    return jnp.where(even, coeff, 0.0).astype(jnp.float64)


def hermite_coeffs_gaussian(N: int, mean: float, sigma: float) -> Array:
    r"""
    Expand a normalized Gaussian W_{mean,sigma}(v) in the normalized Hermite basis:

        W_{m,σ}(v) = (1/(sqrt(2π) σ)) exp(-(v-m)^2/(2σ^2))
                  = Σ c_n φ_n(v)

    with coefficients computed via the recurrence (see the provided numpy script):

        H_0 = 1
        H_1 = m
        H_{n+1} = m H_n + n (σ^2 - 1) H_{n-1}

        c_n = H_n / sqrt(n!)
    """
    if N <= 0:
        return jnp.zeros((0,), dtype=jnp.float64)

    mean = float(mean)
    sigma = float(sigma)
    s2m1 = sigma * sigma - 1.0

    H = jnp.zeros((N,), dtype=jnp.float64)
    H = H.at[0].set(1.0)
    if N > 1:
        H = H.at[1].set(mean)

    def body(i, H_):
        # i runs 1..N-2, sets H[i+1]
        val = mean * H_[i] + (i.astype(H_.dtype) * s2m1) * H_[i - 1]
        return H_.at[i + 1].set(val)

    if N > 2:
        H = jax.lax.fori_loop(1, N - 1, body, H)

    n = jnp.arange(N, dtype=jnp.float64)
    c = H * jnp.exp(-0.5 * jsp.special.gammaln(n + 1.0))
    return c


def equilibrium_coeffs_bump_on_tail(N: int, vbar1: float, vbar2: float, vth: float = 1.0) -> Array:
    r"""
    Bump-on-tail equilibrium:
        μ(v) = 0.9 * W_{v̄1, 1}(v) + 0.1 * W_{v̄2, 1/2}(v)

    expanded in the *scaled* Hermite basis φ_n^{(vth)}.

    Scaling rule:
      Expand W_{m,σ}(v) in φ^{(vth)}(v) by expanding W_{m/vth, σ/vth}(ξ) in standard φ(ξ).
    """
    if vth <= 0:
        raise ValueError("vth must be positive")
    w1, w2 = 0.9, 0.1
    c1 = hermite_coeffs_gaussian(N, vbar1 / vth, 1.0 / vth)
    c2 = hermite_coeffs_gaussian(N, vbar2 / vth, 0.5 / vth)
    return (w1 * c1 + w2 * c2).astype(jnp.float64)


def initial_perturbation_coeffs_bump_on_tail(N: int, eps: float, vbar2: float, vth: float = 1.0) -> Array:
    r"""
    Initial perturbation (paper Section 5.2):
        f(0,x,v) = (ε/10) W_{v̄2, 1/2}(v) cos(β x)
    in scaled basis φ^{(vth)}.
    """
    if vth <= 0:
        raise ValueError("vth must be positive")
    return (0.1 * float(eps)) * hermite_coeffs_gaussian(N, vbar2 / vth, 0.5 / vth)


def gaussian_vhat(eta: Array, mean: float, sigma: float, amplitude: float = 1.0) -> Array:
    r"""
    Continuous Fourier transform in v:
        \hat{W}(η) = ∫ W(v) e^{-i η v} dv
    for a normalized Gaussian W_{mean,sigma}:
        \hat{W}(η) = exp(-i η mean - 0.5 sigma^2 η^2)
    """
    return float(amplitude) * jnp.exp(-1j * eta * float(mean) - 0.5 * (float(sigma) ** 2) * (eta ** 2))


# =============================================================================
# Parameter sets
# =============================================================================

@dataclass
class TwoStreamParams:
    beta: float = 0.2
    vbar: float = 2.4
    eps: float = 0.005

    Nx: int = 128
    Nv: int = 80

    dt: float = 0.1
    T: float = 80.0

    dealias_23: bool = True

    # Hermite spectral-viscosity filter (multiplicative per step)
    hermite_filter: bool = True
    filter_alpha: float = 8.0
    filter_p: int = 8

    poisson_sign: float = +1.0

    snapshot_times: Tuple[float, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80)


@dataclass
class BumpOnTailParams:
    beta: float = 0.1
    vbar1: float = -3.0
    vbar2: float = 4.5
    eps: float = 0.05

    # System C pole-elimination parameter (h(z)=α z in the paper)
    alpha: float = -0.01

    # Scaled Hermite basis parameter
    vth: float = 3.0

    Nx: int = 128
    Nv: int = 200

    dt: float = 0.1
    T: float = 60.0

    dealias_23: bool = True

    hermite_filter: bool = True
    filter_alpha: float = 12.0
    filter_p: int = 8

    poisson_sign: float = +1.0

    snapshot_times: Tuple[float, ...] = (0, 15, 25, 35, 40, 45, 50, 55, 60)


# =============================================================================
# Simulation kernels (JAX)
# =============================================================================

def _make_snapshot_indices(snapshot_times: Tuple[float, ...], dt: float) -> np.ndarray:
    """Map snapshot times to integer step indices (round to nearest)."""
    idx = [int(round(t / float(dt))) for t in snapshot_times]
    return np.array(idx, dtype=np.int32)


def simulate_two_stream(p: TwoStreamParams):
    """
    Nonlinear two-stream instability (System A, H=0), matching the numpy script.

    Returns
    -------
    x : np.ndarray (Nx,)
    m : np.ndarray (Nv,) equilibrium Hermite coefficients
    snaps : Dict[float, np.ndarray] mapping snapshot time -> a_phys (Nv,Nx)
    times : np.ndarray (nt,)
    energy : np.ndarray (nt,)
    """
    Lx = 2.0 * math.pi / float(p.beta)
    integrator = FourierHermiteIMEX(
        Nx=p.Nx, Nv=p.Nv, Lx=Lx, dt=p.dt, vth=1.0, dealias_23=p.dealias_23, closure=None
    )

    # Equilibrium μ coefficients
    m = equilibrium_coeffs_two_stream(p.Nv, p.vbar)

    # Initial perturbation: a_n(x,0) = eps*cos(beta x)*m_n
    a0 = float(p.eps) * jnp.cos(float(p.beta) * integrator.x)
    a_phys0 = (m[:, None] * a0[None, :]).astype(jnp.float64)
    a_hat0 = rfft_x(a_phys0)

    # Per-step Hermite filter factors
    herm_filt = None
    if p.hermite_filter:
        herm_filt = HermiteExponentialFilter(alpha=p.filter_alpha, Nv=p.Nv, p=p.filter_p).factors()

    # Snapshot bookkeeping
    snap_times = tuple(t for t in p.snapshot_times)
    snap_steps = _make_snapshot_indices(snap_times, p.dt)
    S = len(snap_steps)

    snaps0 = jnp.zeros((S, p.Nv, p.Nx), dtype=jnp.float64)

    # Write t=0 snapshot if requested
    def _write_init(snaps):
        return snaps.at[int(np.where(snap_steps == 0)[0][0])].set(irfft_x(a_hat0, p.Nx))
    if 0 in snap_steps:
        snaps0 = _write_init(snaps0)

    # Explicit term N_hat(a_hat,t): nonlinear acceleration with equilibrium source
    def explicit_N_hat(a_hat: Array, t: float) -> Array:
        a_phys = irfft_x(a_hat, p.Nx)
        E = integrator.E_phys_from_a_hat(a_hat, poisson_sign=p.poisson_sign)

        Nv = p.Nv
        N_phys = jnp.zeros((Nv, p.Nx), dtype=jnp.float64)
        # N_n = -sqrt(n) E (a_{n-1} + m_{n-1}), n>=1
        N_phys = N_phys.at[1:].set(-integrator.sqrt_n[1:, None] * E[None, :] * (a_phys[:-1] + m[:-1, None]))
        N_hat = rfft_x(N_phys)
        return integrator.apply_mask_hat(N_hat)

    # Initial explicit term and energy
    N0 = explicit_N_hat(a_hat0, t=0.0)
    energy0 = integrator.electric_energy(a_hat0, poisson_sign=p.poisson_sign)

    nsteps = int(round(float(p.T) / float(p.dt)))
    times = np.linspace(0.0, nsteps * float(p.dt), nsteps + 1)

    def maybe_update_snaps(snaps: Array, step_i: Array, a_hat_new: Array) -> Array:
        # step_i is an int scalar (JAX)
        def update_one(snaps_, j):
            cond = step_i == int(snap_steps[j])

            def do_write(s):
                return s.at[j].set(irfft_x(a_hat_new, p.Nx))

            return jax.lax.cond(cond, do_write, lambda s: s, snaps_)

        for j in range(S):
            snaps = update_one(snaps, j)
        return snaps

    def step(carry, i):
        a_hat, N_prev, snaps = carry
        t_cur = (i - 1) * float(p.dt)

        N_hat = explicit_N_hat(a_hat, t_cur)
        a_new = integrator.step_cnab2(a_hat, N_hat, N_prev, hermite_filter=herm_filt)

        # Diagnostics at t=i*dt
        en = integrator.electric_energy(a_new, poisson_sign=p.poisson_sign)
        snaps = maybe_update_snaps(snaps, i, a_new)

        return (a_new, N_hat, snaps), en

    (a_last, N_last, snaps_out), energy_hist = jax.lax.scan(
        step, (a_hat0, N0, snaps0), jnp.arange(1, nsteps + 1, dtype=jnp.int32)
    )

    energy = jnp.concatenate([jnp.array([energy0], dtype=jnp.float64), energy_hist], axis=0)

    # Convert outputs to numpy
    x_np = np.array(integrator.x)
    m_np = np.array(m)
    energy_np = np.array(energy)

    snaps_np = np.array(snaps_out)
    snaps: Dict[float, np.ndarray] = {t: snaps_np[j] for j, t in enumerate(snap_times)}

    return x_np, m_np, snaps, times, energy_np


def simulate_bump_on_tail(p: BumpOnTailParams, system: str = "A"):
    """
    Nonlinear bump-on-tail instability (System A or System C), matching the numpy script.

    System C uses the pole-elimination control field H(x,t) from the paper.
    """
    system = system.upper()
    if system not in {"A", "C"}:
        raise ValueError("system must be 'A' or 'C'")

    Lx = 2.0 * math.pi / float(p.beta)
    integrator = FourierHermiteIMEX(
        Nx=p.Nx, Nv=p.Nv, Lx=Lx, dt=p.dt, vth=p.vth, dealias_23=p.dealias_23, closure=None
    )

    # Equilibrium μ in scaled Hermite basis
    m = equilibrium_coeffs_bump_on_tail(p.Nv, p.vbar1, p.vbar2, vth=p.vth)

    # Initial perturbation: cos(beta x) * g(v)
    g = initial_perturbation_coeffs_bump_on_tail(p.Nv, p.eps, p.vbar2, vth=p.vth)
    a_phys0 = (g[:, None] * jnp.cos(float(p.beta) * integrator.x)[None, :]).astype(jnp.float64)
    a_hat0 = rfft_x(a_phys0)

    # Per-step Hermite filter factors
    herm_filt = None
    if p.hermite_filter:
        herm_filt = HermiteExponentialFilter(alpha=p.filter_alpha, Nv=p.Nv, p=p.filter_p).factors()

    # Precomputations for System C external field H(x,t)
    if system == "C":
        X_hat = jnp.fft.rfft(jnp.cos(float(p.beta) * integrator.x)).astype(jnp.complex128)  # numpy/JAX scaling

        # perturbation Gaussian in v: (eps/10) W_{v2, 1/2}
        amp_g = 0.1 * float(p.eps)
        mean_g, sig_g = float(p.vbar2), 0.5

        # equilibrium Gaussian mixture
        w1, w2 = 0.9, 0.1
        mu_mean1, mu_sig1 = float(p.vbar1), 1.0
        mu_mean2, mu_sig2 = float(p.vbar2), 0.5

        def H_phys(t: float) -> Array:
            # Eq. (5.1)-style field (without the δ(t) impulse)
            eta = integrator.k_arr * t

            S_hat = X_hat * gaussian_vhat(eta, mean_g, sig_g, amplitude=amp_g)

            muhat = (gaussian_vhat(eta, mu_mean1, mu_sig1, amplitude=w1)
                     + gaussian_vhat(eta, mu_mean2, mu_sig2, amplitude=w2))
            U_hat_raw = float(p.Nx) * (t * muhat)  # match rfft scaling

            H_hat = jnp.zeros((integrator.Nk,), dtype=jnp.complex128)
            H_hat = H_hat.at[1:].set((S_hat[1:] + float(p.alpha) * U_hat_raw[1:]) / (1j * integrator.k_arr[1:]))
            return jnp.fft.irfft(H_hat, n=int(p.Nx)).astype(jnp.float64)
    else:
        def H_phys(t: float) -> Array:
            return jnp.zeros((p.Nx,), dtype=jnp.float64)

    # Snapshot bookkeeping
    snap_times = tuple(t for t in p.snapshot_times)
    snap_steps = _make_snapshot_indices(snap_times, p.dt)
    S = len(snap_steps)

    snaps0 = jnp.zeros((S, p.Nv, p.Nx), dtype=jnp.float64)
    if 0 in snap_steps:
        idx0 = int(np.where(snap_steps == 0)[0][0])
        snaps0 = snaps0.at[idx0].set(irfft_x(a_hat0, p.Nx))

    def explicit_N_hat(a_hat: Array, t: float) -> Array:
        a_phys = irfft_x(a_hat, p.Nx)
        E = integrator.E_phys_from_a_hat(a_hat, poisson_sign=p.poisson_sign)
        H = H_phys(t)
        Etot = E + H

        Nv = p.Nv
        N_phys = jnp.zeros((Nv, p.Nx), dtype=jnp.float64)
        # N_n = -(sqrt(n)/vth) Etot (a_{n-1} + m_{n-1}), n>=1
        N_phys = N_phys.at[1:].set(-(integrator.sqrt_n[1:, None] / float(p.vth)) * Etot[None, :]
                                   * (a_phys[:-1] + m[:-1, None]))
        N_hat = rfft_x(N_phys)
        return integrator.apply_mask_hat(N_hat)

    N0 = explicit_N_hat(a_hat0, t=0.0)
    energy0 = integrator.electric_energy(a_hat0, poisson_sign=p.poisson_sign)

    nsteps = int(round(float(p.T) / float(p.dt)))
    times = np.linspace(0.0, nsteps * float(p.dt), nsteps + 1)

    def maybe_update_snaps(snaps: Array, step_i: Array, a_hat_new: Array) -> Array:
        def update_one(snaps_, j):
            cond = step_i == int(snap_steps[j])

            def do_write(s):
                return s.at[j].set(irfft_x(a_hat_new, p.Nx))

            return jax.lax.cond(cond, do_write, lambda s: s, snaps_)

        for j in range(S):
            snaps = update_one(snaps, j)
        return snaps

    def step(carry, i):
        a_hat, N_prev, snaps = carry
        t_cur = (i - 1) * float(p.dt)

        N_hat = explicit_N_hat(a_hat, t_cur)
        a_new = integrator.step_cnab2(a_hat, N_hat, N_prev, hermite_filter=herm_filt)

        en = integrator.electric_energy(a_new, poisson_sign=p.poisson_sign)
        snaps = maybe_update_snaps(snaps, i, a_new)
        return (a_new, N_hat, snaps), en

    (a_last, N_last, snaps_out), energy_hist = jax.lax.scan(
        step, (a_hat0, N0, snaps0), jnp.arange(1, nsteps + 1, dtype=jnp.int32)
    )
    energy = jnp.concatenate([jnp.array([energy0], dtype=jnp.float64), energy_hist], axis=0)

    x_np = np.array(integrator.x)
    m_np = np.array(m)
    energy_np = np.array(energy)

    snaps_np = np.array(snaps_out)
    snaps: Dict[float, np.ndarray] = {t: snaps_np[j] for j, t in enumerate(snap_times)}
    return x_np, m_np, snaps, times, energy_np


# =============================================================================
# Plotting (optional convenience)
# =============================================================================

def hermite_basis_phi(N: int, v: np.ndarray) -> np.ndarray:
    """
    Build φ_n(v)=w(v) He_n(v)/sqrt(n!) on a v-grid (for visualization).
    Uses a stable normalized recurrence (same as the numpy script).
    """
    v = np.asarray(v)
    Nv = v.size
    w = np.exp(-0.5 * v ** 2) / math.sqrt(2.0 * math.pi)

    h = np.zeros((N, Nv), dtype=float)
    h[0] = 1.0
    if N > 1:
        h[1] = v
    for n in range(1, N - 1):
        h[n + 1] = (v / math.sqrt(n + 1)) * h[n] - math.sqrt(n / (n + 1)) * h[n - 1]
    return w * h


def hermite_basis_phi_scaled(N: int, v: np.ndarray, vth: float) -> np.ndarray:
    """Scaled basis φ^{(vth)} (for visualization)."""
    v = np.asarray(v)
    xi = v / float(vth)
    w = np.exp(-0.5 * xi ** 2) / (math.sqrt(2.0 * math.pi) * float(vth))

    h = np.zeros((N, v.size), dtype=float)
    h[0] = 1.0
    if N > 1:
        h[1] = xi
    for n in range(1, N - 1):
        h[n + 1] = (xi / math.sqrt(n + 1)) * h[n] - math.sqrt(n / (n + 1)) * h[n - 1]
    return w * h


def plot_snapshot_panel(
    x: np.ndarray,
    m: np.ndarray,
    snaps: Dict[float, np.ndarray],
    *,
    v_range: Tuple[float, float] = (-6.0, 6.0),
    Nv_plot: int = 256,
    vmin: float = 0.0,
    vmax: float = 0.22,
    title: str = "",
    savepath: Optional[str] = None,
):
    times = sorted(snaps.keys())
    if len(times) != 9:
        raise ValueError("Expected 9 snapshot times for a 3x3 panel.")

    v = np.linspace(v_range[0], v_range[1], Nv_plot)
    phi = hermite_basis_phi(len(m), v)  # (Nv, Nv_plot)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(title)

    for ax, t in zip(axes.flat, times):
        a_phys = snaps[t]  # (Nv, Nx)
        F = (a_phys + m[:, None]).T @ phi  # (Nx, Nv_plot)
        im = ax.pcolormesh(x, v, F.T, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"t={t:g}")
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if savepath is not None:
        fig.savefig(savepath, dpi=200)
        plt.close(fig)
    return fig


def plot_bump_snapshot_panel(
    x: np.ndarray,
    m: np.ndarray,
    snaps: Dict[float, np.ndarray],
    *,
    vth: float,
    v_range: Tuple[float, float] = (-9.0, 9.0),
    Nv_plot: int = 256,
    vmin: float = 0.0,
    vmax: float = 0.35,
    title: str = "",
    savepath: Optional[str] = None,
):
    times = sorted(snaps.keys())
    if len(times) != 9:
        raise ValueError("Expected 9 snapshot times for a 3x3 panel.")

    v = np.linspace(v_range[0], v_range[1], Nv_plot)
    phi = hermite_basis_phi_scaled(len(m), v, vth=vth)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(title)

    for ax, t in zip(axes.flat, times):
        a_phys = snaps[t]
        F = (a_phys + m[:, None]).T @ phi
        im = ax.pcolormesh(x, v, F.T, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"t={t:g}")
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if savepath is not None:
        fig.savefig(savepath, dpi=200)
        plt.close(fig)
    return fig


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Nonlinear 1D1V Fourier–Hermite simulations (JAX).")
    sub = parser.add_subparsers(dest="case", required=True)

    p_ts = sub.add_parser("two_stream", help="Two-stream instability (System A)")
    p_ts.add_argument("--outdir", type=str, default="out_nonlinear")
    p_ts.add_argument("--Nx", type=int, default=128)
    p_ts.add_argument("--Nv", type=int, default=80)
    p_ts.add_argument("--dt", type=float, default=0.1)
    p_ts.add_argument("--T", type=float, default=80.0)

    p_bot = sub.add_parser("bump_on_tail", help="Bump-on-tail (System A or C)")
    p_bot.add_argument("--system", choices=["A", "C", "AC"], default="AC")
    p_bot.add_argument("--outdir", type=str, default="out_nonlinear")
    p_bot.add_argument("--Nx", type=int, default=128)
    p_bot.add_argument("--Nv", type=int, default=200)
    p_bot.add_argument("--dt", type=float, default=0.1)
    p_bot.add_argument("--T", type=float, default=60.0)
    p_bot.add_argument("--vth", type=float, default=3.0)

    args = parser.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.case == "two_stream":
        p = TwoStreamParams(Nx=args.Nx, Nv=args.Nv, dt=args.dt, T=args.T)
        x, m, snaps, times, energy = simulate_two_stream(p)

        plot_snapshot_panel(
            x, m, snaps,
            title="Two-stream (System A, H=0) — JAX Fourier–Hermite",
            savepath=str(outdir / "two_stream_snapshots.png"),
        )
        np.savez(outdir / "two_stream_energy.npz", times=times, energy=energy)
        print(f"Wrote {outdir / 'two_stream_snapshots.png'}")
        print(f"Wrote {outdir / 'two_stream_energy.npz'}")

    elif args.case == "bump_on_tail":
        p = BumpOnTailParams(Nx=args.Nx, Nv=args.Nv, dt=args.dt, T=args.T, vth=args.vth)

        if args.system in ("A", "AC"):
            x, m, snapsA, times, energyA = simulate_bump_on_tail(p, system="A")
            plot_bump_snapshot_panel(
                x, m, snapsA, vth=p.vth,
                title="Bump-on-tail (System A, H=0) — JAX Fourier–Hermite",
                savepath=str(outdir / "bump_on_tail_systemA_snapshots.png"),
            )
            np.savez(outdir / "bump_on_tail_systemA_energy.npz", times=times, energy=energyA)
            print(f"Wrote {outdir / 'bump_on_tail_systemA_snapshots.png'}")

        if args.system in ("C", "AC"):
            x, m, snapsC, times, energyC = simulate_bump_on_tail(p, system="C")
            plot_bump_snapshot_panel(
                x, m, snapsC, vth=p.vth,
                title=f"Bump-on-tail (System C, alpha={p.alpha}) — JAX Fourier–Hermite",
                savepath=str(outdir / "bump_on_tail_systemC_snapshots.png"),
            )
            np.savez(outdir / "bump_on_tail_systemC_energy.npz", times=times, energy=energyC)
            print(f"Wrote {outdir / 'bump_on_tail_systemC_snapshots.png'}")

        if args.system == "AC":
            # energy comparison plot
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            ax.plot(times, energyA, label="System A")
            ax.plot(times, energyC, label="System C")
            ax.set_xlabel("t")
            ax.set_ylabel(r"$\frac{1}{2}\int E^2\,dx$")
            ax.set_title("Bump-on-tail electric energy")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.savefig(outdir / "bump_on_tail_energy_comparison.png", dpi=200)
            plt.close(fig)
            print(f"Wrote {outdir / 'bump_on_tail_energy_comparison.png'}")


if __name__ == "__main__":
    main()
