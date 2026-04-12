"""
Nonlinear 1D1V Vlasov-Poisson simulations on a physical (x, v) grid.

Both the two-stream and bump-on-tail paths use the shared JAX semi-Lagrangian cubic-spline
solver from ``vpml.physical_grid``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from vpml.jax_runtime import bootstrap_jax_runtime, print_jax_runtime_summary

bootstrap_jax_runtime()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp

from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    gaussian_pdf,
    gaussian_vhat,
    normalize_density_on_grid,
    run_semilagrangian_vlasov_poisson,
)
from vpml.visualization.nonlinear import (
    plot_snapshot_panel as _plot_snapshot_panel,
    save_bump_on_tail_energy_comparison,
)
from vpml.visualization.common import save_figure

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


@dataclass
class TwoStreamParams:
    beta: float = 0.2
    vbar: float = 2.4
    eps: float = 0.005
    Nx: int = 128
    Nv: int = 128
    dt: float = 0.1
    T: float = 80.0
    vmin: float = -6.0
    vmax: float = 6.0
    poisson_sign: float = +1.0
    snapshot_times: Tuple[float, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80)


@dataclass
class BumpOnTailParams:
    beta: float = 0.1
    vbar1: float = -3.0
    vbar2: float = 4.5
    eps: float = 0.05
    alpha: float = -0.01
    Nx: int = 128
    Nv: int = 256
    dt: float = 0.1
    T: float = 60.0
    vmin: float = -12.0
    vmax: float = 12.0
    poisson_sign: float = +1.0
    snapshot_times: Tuple[float, ...] = (0, 15, 25, 35, 40, 45, 50, 55, 60)


def simulate_two_stream(p: TwoStreamParams):
    config = PhysicalGridVlasovPoissonConfig(
        Nx=int(p.Nx),
        Nv=int(p.Nv),
        Lx=2.0 * math.pi / float(p.beta),
        vmin=float(p.vmin),
        vmax=float(p.vmax),
        dt=float(p.dt),
        T=float(p.T),
        poisson_sign=float(p.poisson_sign),
        snapshot_times=tuple(float(t) for t in p.snapshot_times),
    )
    v = config.v
    mu = 0.5 * gaussian_pdf(v, float(p.vbar), 1.0) + 0.5 * gaussian_pdf(v, -float(p.vbar), 1.0)
    mu = normalize_density_on_grid(mu, v)
    f0 = mu[:, None] * (1.0 + float(p.eps) * jnp.cos(float(p.beta) * config.x)[None, :])
    raw = run_semilagrangian_vlasov_poisson(config, f0)
    snaps_np = np.asarray(raw["snapshot_f"], dtype=float)
    snaps: Dict[float, np.ndarray] = {
        float(t): snaps_np[j] for j, t in enumerate(np.asarray(raw["snapshot_times"], dtype=float))
    }
    return (
        np.asarray(raw["x"], dtype=float),
        np.asarray(raw["v"], dtype=float),
        snaps,
        np.asarray(raw["times"], dtype=float),
        np.asarray(raw["energy"], dtype=float),
    )


def simulate_bump_on_tail(p: BumpOnTailParams, system: str = "A"):
    system = system.upper()
    if system not in {"A", "C"}:
        raise ValueError("system must be 'A' or 'C'")

    config = PhysicalGridVlasovPoissonConfig(
        Nx=int(p.Nx),
        Nv=int(p.Nv),
        Lx=2.0 * math.pi / float(p.beta),
        vmin=float(p.vmin),
        vmax=float(p.vmax),
        dt=float(p.dt),
        T=float(p.T),
        poisson_sign=float(p.poisson_sign),
        snapshot_times=tuple(float(t) for t in p.snapshot_times),
    )
    v = config.v
    equilibrium = 0.9 * gaussian_pdf(v, float(p.vbar1), 1.0) + 0.1 * gaussian_pdf(v, float(p.vbar2), 0.5)
    equilibrium = normalize_density_on_grid(equilibrium, v)
    perturb = 0.1 * float(p.eps) * gaussian_pdf(v, float(p.vbar2), 0.5)[:, None] * jnp.cos(float(p.beta) * config.x)[None, :]
    f0 = equilibrium[:, None] + perturb

    if system == "C":
        X_hat = jnp.fft.rfft(jnp.cos(float(p.beta) * config.x)).astype(jnp.complex128)
        amp_g = 0.1 * float(p.eps)
        mean_g = float(p.vbar2)
        sig_g = 0.5
        w1, w2 = 0.9, 0.1
        mu_mean1, mu_sig1 = float(p.vbar1), 1.0
        mu_mean2, mu_sig2 = float(p.vbar2), 0.5

        def external_field_fn(t, x, k_arr):
            del x
            t = jnp.asarray(t, dtype=jnp.float64)
            eta = jnp.asarray(k_arr, dtype=jnp.float64) * t
            S_hat = X_hat * gaussian_vhat(eta, mean_g, sig_g, amplitude=amp_g)
            muhat = (
                gaussian_vhat(eta, mu_mean1, mu_sig1, amplitude=w1)
                + gaussian_vhat(eta, mu_mean2, mu_sig2, amplitude=w2)
            )
            U_hat_raw = float(config.Nx) * (t * muhat)
            H_hat = jnp.zeros((int(config.Nk),), dtype=jnp.complex128)
            H_hat = H_hat.at[1:].set((S_hat[1:] + float(p.alpha) * U_hat_raw[1:]) / (1j * k_arr[1:]))
            return jnp.fft.irfft(H_hat, n=int(config.Nx)).astype(jnp.float64)

    else:
        external_field_fn = None

    raw = run_semilagrangian_vlasov_poisson(config, f0, external_field_fn=external_field_fn)
    snaps_np = np.asarray(raw["snapshot_f"], dtype=float)
    snaps: Dict[float, np.ndarray] = {
        float(t): snaps_np[j] for j, t in enumerate(np.asarray(raw["snapshot_times"], dtype=float))
    }
    return (
        np.asarray(raw["x"], dtype=float),
        np.asarray(raw["v"], dtype=float),
        snaps,
        np.asarray(raw["times"], dtype=float),
        np.asarray(raw["energy"], dtype=float),
    )


def plot_snapshot_panel(
    x: np.ndarray,
    v: np.ndarray,
    snaps: Dict[float, np.ndarray],
    *,
    vmin: float = 0.0,
    vmax: float | None = None,
    paper_view: bool = False,
    title: str = "",
    savepath: str | None = None,
):
    fig = _plot_snapshot_panel(
        x,
        v,
        snaps,
        vmin=vmin,
        vmax=vmax,
        paper_view=paper_view,
        title=title,
    )
    if savepath is not None:
        save_figure(fig, savepath, dpi=200)
    return fig


def main():
    import argparse

    print_jax_runtime_summary(jax, context="nonlinear")
    parser = argparse.ArgumentParser(description="Nonlinear 1D1V physical-grid Vlasov-Poisson simulations (JAX).")
    sub = parser.add_subparsers(dest="case", required=True)

    p_ts = sub.add_parser("two_stream", help="Two-stream instability (System A)")
    p_ts.add_argument("--outdir", type=str, default="out_nonlinear")
    p_ts.add_argument("--Nx", type=int, default=128)
    p_ts.add_argument("--Nv", type=int, default=128)
    p_ts.add_argument("--dt", type=float, default=0.1)
    p_ts.add_argument("--T", type=float, default=80.0)
    p_ts.add_argument("--raw-view", action="store_true")

    p_bot = sub.add_parser("bump_on_tail", help="Bump-on-tail (System A or C)")
    p_bot.add_argument("--system", choices=["A", "C", "AC"], default="AC")
    p_bot.add_argument("--outdir", type=str, default="out_nonlinear")
    p_bot.add_argument("--Nx", type=int, default=128)
    p_bot.add_argument("--Nv", type=int, default=256)
    p_bot.add_argument("--dt", type=float, default=0.1)
    p_bot.add_argument("--T", type=float, default=60.0)
    p_bot.add_argument("--vmin", type=float, default=-12.0)
    p_bot.add_argument("--vmax", type=float, default=12.0)

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.case == "two_stream":
        p = TwoStreamParams(Nx=args.Nx, Nv=args.Nv, dt=args.dt, T=args.T)
        x, v, snaps, times, energy = simulate_two_stream(p)
        plot_snapshot_panel(
            x,
            v,
            snaps,
            paper_view=not args.raw_view,
            title="Two-stream (System A, H=0) — JAX semi-Lagrangian cubic spline",
            savepath=str(outdir / "two_stream_snapshots.png"),
        )
        np.savez(outdir / "two_stream_energy.npz", times=times, energy=energy)
        print(f"Wrote {outdir / 'two_stream_snapshots.png'}")
        print(f"Wrote {outdir / 'two_stream_energy.npz'}")

    elif args.case == "bump_on_tail":
        p = BumpOnTailParams(Nx=args.Nx, Nv=args.Nv, dt=args.dt, T=args.T, vmin=args.vmin, vmax=args.vmax)

        if args.system in ("A", "AC"):
            x, v, snapsA, times, energyA = simulate_bump_on_tail(p, system="A")
            plot_snapshot_panel(
                x,
                v,
                snapsA,
                title="Bump-on-tail (System A, H=0) — JAX semi-Lagrangian cubic spline",
                savepath=str(outdir / "bump_on_tail_systemA_snapshots.png"),
            )
            np.savez(outdir / "bump_on_tail_systemA_energy.npz", times=times, energy=energyA)
            print(f"Wrote {outdir / 'bump_on_tail_systemA_snapshots.png'}")

        if args.system in ("C", "AC"):
            x, v, snapsC, times, energyC = simulate_bump_on_tail(p, system="C")
            plot_snapshot_panel(
                x,
                v,
                snapsC,
                title=f"Bump-on-tail (System C, alpha={p.alpha}) — JAX semi-Lagrangian cubic spline",
                savepath=str(outdir / "bump_on_tail_systemC_snapshots.png"),
            )
            np.savez(outdir / "bump_on_tail_systemC_energy.npz", times=times, energy=energyC)
            print(f"Wrote {outdir / 'bump_on_tail_systemC_snapshots.png'}")

        if args.system == "AC":
            save_bump_on_tail_energy_comparison(
                times,
                energyA,
                energyC,
                outdir / "bump_on_tail_energy_comparison.png",
            )
            print(f"Wrote {outdir / 'bump_on_tail_energy_comparison.png'}")


if __name__ == "__main__":
    main()
