"""Training visualizations for learned interface-closure runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .common import save_figure


CANONICAL_TRAINING_MODE = "solver_embedded_interface_flux_rollout"
CANONICAL_OBJECTIVE = "interface_flux_rollout"
CANONICAL_LOSS_BACKEND = "regime_balanced_all_k_interface_flux"
LEGACY_INTERFACE_FLUX_TRIPLE = (
    "exact_q_rollout",
    "q_rollout",
    "exact_fourier_hermite_q_rollout",
)


def _metadata_scalar(
    metadata: Mapping[str, np.ndarray],
    key: str,
    default: str = "",
) -> str:
    if key not in metadata:
        return default
    values = np.asarray(metadata[key]).reshape(-1)
    return str(values[0]) if values.size else default


def normalize_interface_flux_loss_metadata(
    metadata: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Normalize canonical and retained exact-q metrics for plotting."""
    normalized = {str(key): np.asarray(value) for key, value in metadata.items()}
    triple = (
        _metadata_scalar(normalized, "training_mode"),
        _metadata_scalar(normalized, "train_objective"),
        _metadata_scalar(normalized, "loss_backend"),
    )
    if triple == LEGACY_INTERFACE_FLUX_TRIPLE:
        normalized["training_mode"] = np.array(
            [CANONICAL_TRAINING_MODE], dtype=np.str_
        )
        normalized["train_objective"] = np.array(
            [CANONICAL_OBJECTIVE], dtype=np.str_
        )
        normalized["loss_backend"] = np.array(
            [CANONICAL_LOSS_BACKEND], dtype=np.str_
        )
        if "exact_q_regime_loss_regimes" in normalized:
            normalized["interface_flux_regime_loss_regimes"] = np.asarray(
                normalized["exact_q_regime_loss_regimes"], dtype=np.str_
            )
        if "exact_q_regime_loss_stds" in normalized:
            normalized["interface_flux_regime_loss_stds"] = np.asarray(
                normalized["exact_q_regime_loss_stds"], dtype=np.float64
            )
    elif triple != (
        CANONICAL_TRAINING_MODE,
        CANONICAL_OBJECTIVE,
        CANONICAL_LOSS_BACKEND,
    ):
        raise ValueError(
            "Loss metrics are not from the canonical interface-flux trainer or "
            f"a retained solver-embedded exact-q run: {triple!r}"
        )
    return normalized


def _short_regime_name(regime: str) -> str:
    return {
        "linear_landau": "lin",
        "nonlinear_landau_weak": "weak",
        "nonlinear_landau_strong": "strong",
    }.get(str(regime), str(regime))


def interface_flux_loss_ylabel(
    regimes: Sequence[str],
    regime_stds: Sequence[float],
) -> str:
    regimes_tuple = tuple(_short_regime_name(value) for value in regimes)
    stds = tuple(float(value) for value in regime_stds)
    if not regimes_tuple or len(regimes_tuple) != len(stds):
        raise ValueError("Interface-flux loss metadata requires matching regimes and scales")
    regime_text = ",".join(rf"\mathrm{{{value}}}" for value in regimes_tuple)
    sigma_text = r",\ ".join(
        rf"\sigma_{{\mathrm{{{regime}}}}}={std:.5g}"
        for regime, std in zip(regimes_tuple, stds)
    )
    return (
        r"$\mathcal{L}_{\mathrm{IF}}^{H}(\theta)="
        rf"\sum_{{r\in\{{{regime_text}\}}}}"
        r"\frac{w_r}{2BH|\mathcal{K}_{+}|}"
        r"\sum_{i=1}^{B}\sum_{h=0}^{H-1}\sum_{k\in\mathcal{K}_{+}}"
        r"\frac{|q_{r,i,h,k}^{\theta}-q_{r,i,h,k}^{\star}|^2}{\sigma_r^2}$"
        "\n"
        r"$w_r=\frac{1}{3},\quad "
        r"|{\Delta q}|^2=(\Delta\operatorname{Re}q)^2+"
        r"(\Delta\operatorname{Im}q)^2,\quad "
        + sigma_text
        + "$"
    )


def _loss_metadata_label(metadata: Mapping[str, np.ndarray]) -> str:
    normalized = normalize_interface_flux_loss_metadata(metadata)
    regimes = np.asarray(
        normalized.get(
            "interface_flux_regime_loss_regimes",
            normalized.get("regimes", np.array([], dtype=np.str_)),
        ),
        dtype=np.str_,
    ).reshape(-1)
    stds = np.asarray(
        normalized.get(
            "interface_flux_regime_loss_stds",
            np.array([], dtype=np.float64),
        ),
        dtype=np.float64,
    ).reshape(-1)
    if regimes.size != stds.size or regimes.size == 0:
        raise ValueError(
            "Interface-flux loss metrics do not contain complete regime scale metadata"
        )
    return interface_flux_loss_ylabel(regimes, stds)


def plot_training_loss(
    loss_history: np.ndarray,
    *,
    loss_metadata: Mapping[str, np.ndarray],
    val_metrics: Optional[Dict[str, np.ndarray]] = None,
) -> plt.Figure:
    """Build the canonical interface-flux training-loss figure."""
    normalized = normalize_interface_flux_loss_metadata(loss_metadata)
    fig, ax = plt.subplots(figsize=(12.0, 6.4), constrained_layout=True)
    epochs = np.arange(1, int(len(loss_history)) + 1, dtype=int)
    if len(loss_history) > 0:
        ax.semilogy(epochs, np.maximum(np.asarray(loss_history, dtype=np.float64), 1e-30), lw=2.0, color="#111827")
    else:
        ax.plot([], [])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(_loss_metadata_label(normalized), fontsize=8)
    ax.set_title("Solver-Embedded Interface-Flux Training Loss")
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
    loss_metadata: Mapping[str, np.ndarray],
    val_metrics: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """Save the canonical interface-flux training-loss figure."""
    fig = plot_training_loss(
        loss_history,
        loss_metadata=loss_metadata,
        val_metrics=val_metrics,
    )
    return save_figure(fig, output_path, dpi=220)


def load_training_metrics(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(Path(path)) as payload:
        metrics = {key: np.asarray(payload[key]) for key in payload.files}
    return normalize_interface_flux_loss_metadata(metrics)


def save_training_loss_plot_from_metrics(
    metrics_path: str | Path,
    output_path: str | Path,
) -> Path:
    metrics = load_training_metrics(metrics_path)
    if "train_loss" not in metrics:
        raise ValueError(f"Training metrics at {metrics_path} do not contain train_loss")
    val_metrics = {
        key: value
        for key, value in metrics.items()
        if key.startswith("val_")
    }
    return save_training_loss_plot(
        np.asarray(metrics["train_loss"], dtype=np.float64),
        output_path,
        loss_metadata=metrics,
        val_metrics=val_metrics,
    )


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
        label = (
            r"optimized $\mathcal{L}_{\mathrm{chain}}$"
            if str(loss_backend) == "exact_fourier_hermite_q_rollout_chain_only"
            else
            r"optimized $\mathcal{L}_{\mathrm{dyn}}+\lambda_{\mathrm{chain}}\mathcal{L}_{\mathrm{chain}}$"
            if str(loss_backend) == "exact_fourier_hermite_q_rollout_tail_chain"
            else r"optimized $\mathcal{L}_{xv}$"
        )
        ax.semilogy(epochs, loss, lw=2.0, color="#111827", label=label)
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
