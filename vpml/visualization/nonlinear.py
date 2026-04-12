"""Nonlinear physical-grid visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .common import save_figure


def plot_snapshot_panel(
    x: np.ndarray,
    v: np.ndarray,
    snaps: Dict[float, np.ndarray],
    *,
    vmin: float = 0.0,
    vmax: float | None = None,
    paper_view: bool = False,
    title: str = "",
) -> plt.Figure:
    """Build the standard 3x3 phase-space snapshot panel."""
    times = sorted(snaps.keys())
    if len(times) != 9:
        raise ValueError("Expected 9 snapshot times for a 3x3 panel.")

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(title)

    for ax, t in zip(axes.flat, times):
        field = snaps[t][::-1, :] if paper_view else snaps[t]
        im = ax.pcolormesh(x, v, field, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"t={t:g}")
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def save_snapshot_panel(
    x: np.ndarray,
    v: np.ndarray,
    snaps: Dict[float, np.ndarray],
    output_path: str | Path,
    *,
    vmin: float = 0.0,
    vmax: float | None = None,
    paper_view: bool = False,
    title: str = "",
) -> Path:
    """Save the standard 3x3 phase-space snapshot panel."""
    fig = plot_snapshot_panel(
        x,
        v,
        snaps,
        vmin=vmin,
        vmax=vmax,
        paper_view=paper_view,
        title=title,
    )
    return save_figure(fig, output_path, dpi=200)


def plot_bump_on_tail_energy_comparison(
    times: np.ndarray,
    energy_a: np.ndarray,
    energy_c: np.ndarray,
) -> plt.Figure:
    """Build the bump-on-tail electric-energy comparison figure."""
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(times, energy_a, label="System A")
    ax.plot(times, energy_c, label="System C")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\frac{1}{2}\int E^2\,dx$")
    ax.set_title("Bump-on-tail electric energy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def save_bump_on_tail_energy_comparison(
    times: np.ndarray,
    energy_a: np.ndarray,
    energy_c: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save the bump-on-tail electric-energy comparison figure."""
    fig = plot_bump_on_tail_energy_comparison(times, energy_a, energy_c)
    return save_figure(fig, output_path, dpi=200)
