"""Shared figure-saving helpers for vpml visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    *,
    dpi: int,
    close: bool = True,
) -> Path:
    """Save a matplotlib figure to disk and optionally close it."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi))
    if close:
        plt.close(fig)
    return output_path
