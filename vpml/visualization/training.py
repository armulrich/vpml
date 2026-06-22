"""Training visualizations for learned interface-closure runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .common import save_figure


def _training_loss_ylabel(train_objective: str, loss_backend: str | None = None) -> str:
    if str(train_objective) == "trajectory":
        if str(loss_backend) == "fourier_hermite_bidir":
            return (
                r"$\mathcal{L}_{\mathrm{FH-state}}(\theta)="
                r"\frac{\sum_{s,m,n,k}w_{m,n,k}"
                r"\left|\hat a_{n,k}^{\theta}(t_s\pm m\Delta t)-"
                r"\hat a_{n,k}^{\star}(t_s\pm m\Delta t)\right|^2}"
                r"{\sum_{s,m,n,k}w_{m,n,k}"
                r"\left|\hat a_{n,k}^{\star}(t_s\pm m\Delta t)\right|^2}$"
            )
        if str(loss_backend) == "fourier_hermite_closure_bidir":
            return (
                r"$\mathcal{L}_{\mathrm{FH-q}}(\theta)="
                r"\frac{\sum_{s,m,k}w_{m,k}"
                r"\left|q_k^\theta(\tilde a(t_s\pm m\Delta t))-"
                r"q_{k}^{\star}(t_s\pm m\Delta t)\right|^2}"
                r"{\sum_{s,m,k}w_{m,k}"
                r"\left|q_{k}^{\star}(t_s\pm m\Delta t)\right|^2}$"
            )
        if str(loss_backend) == "fourier_hermite_projected_xv_bidir":
            return (
                r"$\mathcal{L}_{xv}^{N_v}(\theta)="
                r"\frac{1}{2H|\mathcal{S}|}\sum_{s,m,\pm}"
                r"\frac{\int\!\int"
                r"\left|f_\theta^{N_v}(x,v,t_s\pm m\Delta t)-"
                r"f_{\mathrm{HR}}^{N_v}(x,v,t_s\pm m\Delta t)\right|^2\,dv\,dx}"
                r"{\int\!\int\left|f_{\mathrm{HR}}^{N_v}(x,v,t_s\pm m\Delta t)\right|^2\,dv\,dx}$"
            )
        return (
            r"$\mathcal{L}_{\mathrm{traj}}(\theta)="
            r"\lambda_E\mathcal{L}_E+"
            r"\lambda_{\mathrm{dist}}\mathcal{L}_{\delta f}+"
            r"\lambda_{\mathrm{tail}}\mathcal{L}_{\mathrm{tail}}+"
            r"\lambda_{\mathrm{neg}}\mathcal{L}_{\mathrm{neg}}+"
            r"\lambda_{\mathrm{reg}}\|\theta\|_2^2$"
        )
    if str(train_objective) == "trajectory_q_hybrid":
        return (
            r"$\mathcal{L}_{\mathrm{traj+q}}(\theta)="
            r"\lambda_q\mathcal{L}_q+"
            r"\lambda_E\mathcal{L}_E+"
            r"\lambda_{\mathrm{dist}}\mathcal{L}_{\delta f}+"
            r"\lambda_{\mathrm{tail}}\mathcal{L}_{\mathrm{tail}}+"
            r"\lambda_{\mathrm{neg}}\mathcal{L}_{\mathrm{neg}}+"
            r"\lambda_{\mathrm{reg}}\|\theta\|_2^2$"
        )
    if str(train_objective) == "q_rollout":
        return (
            r"$\mathcal{L}_{q}^{H}(\theta)="
            r"\frac{1}{H|\mathcal{S}|}\sum_{s,h,k>0}"
            r"\left|S_q(q_k^\theta(C_h^\theta))-S_q(q_k^\star(t_s+h\Delta t))\right|^2$"
        )
    return r"$\mathcal{L}(\theta)=\mathbb{E}_{\mathrm{regime}}\mathbb{E}_{t,k>0}\left[\left|q_k^\theta-q_k^\star\right|^2\right]$"


def plot_training_loss(
    loss_history: np.ndarray,
    *,
    val_metrics: Optional[Dict[str, np.ndarray]] = None,
    train_objective: str = "q_only",
    loss_backend: str | None = None,
) -> plt.Figure:
    """Build the shared interface-closure training-loss figure."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    epochs = np.arange(1, int(len(loss_history)) + 1, dtype=int)
    if len(loss_history) > 0:
        ax.semilogy(epochs, np.maximum(np.asarray(loss_history, dtype=np.float64), 1e-30), lw=2.0, color="#111827")
    else:
        ax.plot([], [])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(_training_loss_ylabel(train_objective, loss_backend=loss_backend))
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
    loss_backend: str | None = None,
) -> Path:
    """Save the shared interface-closure training-loss figure."""
    fig = plot_training_loss(
        loss_history,
        val_metrics=val_metrics,
        train_objective=train_objective,
        loss_backend=loss_backend,
    )
    return save_figure(fig, output_path, dpi=220)


def plot_training_loss_q_diagnostic(
    loss_history: np.ndarray,
    q_diag_history: np.ndarray,
    *,
    loss_backend: str | None = None,
) -> plt.Figure:
    """Plot the optimized online loss beside the rollout-window q diagnostic."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    epochs = np.arange(1, int(len(loss_history)) + 1, dtype=int)
    loss = np.maximum(np.asarray(loss_history, dtype=np.float64), 1e-30)
    q_diag = np.maximum(np.asarray(q_diag_history, dtype=np.float64), 1e-30)
    if len(loss) > 0:
        ax.semilogy(epochs, loss, lw=2.0, color="#111827", label=r"optimized $\mathcal{L}_{xv}$")
    if len(q_diag) > 0:
        ax.semilogy(epochs, q_diag, lw=2.0, color="#b45309", label=r"rollout-window $q$ relative MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training diagnostic")
    title = "Online Loss vs Rollout-Window Closure Diagnostic"
    if loss_backend:
        title += f" ({loss_backend})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return fig


def save_training_loss_q_diagnostic_plot(
    loss_history: np.ndarray,
    q_diag_history: np.ndarray,
    output_path: str | Path,
    *,
    loss_backend: str | None = None,
) -> Path:
    """Save the online loss/q-diagnostic figure."""
    fig = plot_training_loss_q_diagnostic(
        loss_history,
        q_diag_history,
        loss_backend=loss_backend,
    )
    return save_figure(fig, output_path, dpi=220)
