"""Plot raw epoch training loss with held-out validation loss only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--title",
        default="Latent-6 low-moment trajectory training",
    )
    args = parser.parse_args()

    with np.load(args.metrics, allow_pickle=False) as payload:
        train_loss = np.asarray(payload["train_loss"], dtype=np.float64)
        val_epochs = np.asarray(payload["val_epochs"], dtype=np.int64)
        val_loss = np.asarray(payload["val_loss"], dtype=np.float64)

    train_epochs = np.arange(1, train_loss.size + 1, dtype=np.int64)
    if train_loss.size == 0 or val_loss.size == 0:
        raise ValueError("Training and held-out loss histories must be nonempty")
    if val_epochs.size != val_loss.size:
        raise ValueError("Held-out epochs and losses must have equal lengths")
    if not np.all(np.isfinite(train_loss)) or not np.all(np.isfinite(val_loss)):
        raise ValueError("Loss histories must be finite")

    fig, ax = plt.subplots(figsize=(12.0, 6.4), constrained_layout=True)
    ax.semilogy(
        train_epochs,
        train_loss,
        color="#111827",
        linewidth=3.0,
        label="training batch mean",
        zorder=3,
    )
    ax.semilogy(
        val_epochs,
        val_loss,
        color="#c74e52",
        linewidth=3.0,
        label="held-out IC validation",
        zorder=4,
    )

    ax.set_xlim(0.0, float(max(train_epochs[-1], val_epochs[-1])))
    ax.set_xticks(np.arange(0, int(ax.get_xlim()[1]) + 1, 2, dtype=int))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.set_ylim(1.5e-2, 3.0e-1)
    y_ticks = np.geomspace(2.0e-2, 3.0e-1, 9, dtype=np.float64)
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.3f}".rstrip("0").rstrip("."))
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        r"$\mathcal{L}_{\mathrm{traj}}^{H}(\theta)="
        r"\frac{1}{B}\sum_{i=1}^{B}"
        r"\frac{\|\widehat{\mathbf{y}}^{\theta}_{i,1:H}-"
        r"\mathbf{y}^{\star}_{i,1:H}\|_2^2}"
        r"{\|\mathbf{y}^{\star}_{i,1:H}\|_2^2},\quad"
        r"\mathbf{y}=(n,u,p,E)$",
        fontsize=10,
    )
    ax.set_title(args.title)
    ax.grid(True, which="major", color="#b8b8b8", linewidth=0.9, alpha=0.42)
    ax.grid(False, which="minor")
    ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.0)
    ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.legend(frameon=True, facecolor="white", edgecolor="#b0b0b0")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
