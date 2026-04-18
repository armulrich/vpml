"""Training visualizations for learned interface-closure runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .common import save_figure


def _training_loss_ylabel(train_objective: str) -> str:
    if str(train_objective) == "trajectory":
        return (
            r"$\mathcal{L}_{\mathrm{traj}}(\theta)="
            r"\lambda_E\mathcal{L}_E+"
            r"\lambda_{\mathrm{dist}}\mathcal{L}_{\delta f}+"
            r"\lambda_{\mathrm{tail}}\mathcal{L}_{\mathrm{tail}}+"
            r"\lambda_{\mathrm{neg}}\mathcal{L}_{\mathrm{neg}}+"
            r"\lambda_{\mathrm{reg}}\|\theta\|_2^2$"
        )
    if str(train_objective) == "stability_aware":
        return (
            r"$\mathcal{L}_{\mathrm{stab}}(\theta)="
            r"\lambda_q\mathcal{L}_q+"
            r"\lambda_E\mathcal{L}_{E,\mathrm{window}}^{\mathrm{hyb}}+"
            r"\lambda_{\mathrm{tail}}\mathcal{L}_{\mathrm{tail},\mathrm{window}}^{\mathrm{hyb}}+"
            r"\lambda_{\mathrm{reg}}\|\theta\|_2^2$"
        )
    return r"$\mathcal{L}(\theta)=\mathbb{E}_{\mathrm{regime}}\mathbb{E}_{t,k>0}\left[\left|q_k^\theta-q_k^\star\right|^2\right]$"


def plot_training_loss(
    loss_history: np.ndarray,
    *,
    val_metrics: Optional[Dict[str, np.ndarray]] = None,
    train_objective: str = "q_only",
) -> plt.Figure:
    """Build the shared interface-closure training-loss figure."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    epochs = np.arange(1, int(len(loss_history)) + 1, dtype=int)
    if len(loss_history) > 0:
        ax.semilogy(epochs, np.maximum(np.asarray(loss_history, dtype=np.float64), 1e-30), lw=2.0, color="#111827")
    else:
        ax.plot([], [])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(_training_loss_ylabel(train_objective))
    ax.set_title("Shared Interface-Closure Training Loss")
    ax.grid(True, alpha=0.3)
    if val_metrics:
        lines = []
        for key in sorted(val_metrics):
            if key.startswith("val_q_mse_"):
                regime = key.removeprefix("val_q_mse_")
                lines.append(f"{regime}: {float(np.asarray(val_metrics[key]).reshape(-1)[0]):.3e}")
        if lines:
            ax.text(
                0.98,
                0.98,
                "Validation MSE\n" + "\n".join(lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
            )
    return fig


def save_training_loss_plot(
    loss_history: np.ndarray,
    output_path: str | Path,
    *,
    val_metrics: Optional[Dict[str, np.ndarray]] = None,
    train_objective: str = "q_only",
) -> Path:
    """Save the shared interface-closure training-loss figure."""
    fig = plot_training_loss(loss_history, val_metrics=val_metrics, train_objective=train_objective)
    return save_figure(fig, output_path, dpi=220)
