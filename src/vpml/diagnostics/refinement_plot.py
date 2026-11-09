"""Plots for explicitly paired Gaussian refinement calculations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vpml.datasets.schema import DIAGNOSTIC_NAMES, FIELD_NAMES


_COLORS = {"baseline": "#555555", "finer_x": "#2166ac",
           "finer_v": "#d87520", "half_dt": "#8050a6"}
_LABELS = {"baseline": "Baseline", "finer_x": "Finer spatial grid",
           "finer_v": "Finer velocity grid", "half_dt": "Smaller timestep"}


def _shared_history(base: dict, other: dict):
    """Select a common interval without interpolating either history."""
    ta = np.asarray(base["arrays"]["times"])
    tb = np.asarray(other["arrays"]["times"])
    end = min(ta[-1], tb[-1])
    ia = np.flatnonzero(ta <= end + 1e-12)
    ib = np.flatnonzero(tb <= end + 1e-12)
    if len(ia) < 2 or len(ia) != len(ib) or not np.allclose(
            ta[ia], tb[ib], atol=1e-12, rtol=0):
        raise ValueError("Plot requires at least two exact shared observation times")
    return ta[ia], ia, ib


def plot_comparison(variants: list[dict], comparisons: list[dict],
                    output_path: Path) -> Path:
    """Save a four-panel PNG/PDF; captions and interpretation remain external.

    The closure-error curves use each comparison history's space-time RMS as
    their denominator, consistent with the scalar refinement metrics. A zero
    reference norm has no relative-error curve and is omitted explicitly.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("Refinement figure must have a .png or .pdf suffix")
    lookup = {case["variant"]: case for case in variants}
    if not variants or len(lookup) != len(variants):
        raise ValueError("Plot requires unique, nonempty variant names")
    columns = {name: i for i, name in enumerate(DIAGNOSTIC_NAMES)}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    min_axis, negative_axis, energy_axis, error_axis = axes.flat
    for case in variants:
        name, arrays = case["variant"], case["arrays"]
        times = np.asarray(arrays["diagnostic_times"])
        diagnostics = np.asarray(arrays["diagnostics"])
        style = dict(color=_COLORS.get(name), label=_LABELS.get(name, name),
                     linewidth=1.8, alpha=.95)
        min_axis.plot(times, diagnostics[:, columns["min_f"]], **style)
        negative_axis.plot(times, np.maximum(
            diagnostics[:, columns["negative_mass_fraction"]], 1e-20), **style)
        energy_axis.plot(times, diagnostics[:, columns["electric_energy"]], **style)
    min_axis.set_yscale("symlog", linthresh=1e-8)
    min_axis.axhline(-1e-8, color="#a64036", linestyle="--", linewidth=1,
                    label="Rejection threshold")
    min_axis.set_ylabel(r"Minimum $f$")
    min_axis.legend(fontsize=9, loc="best")
    negative_axis.set_yscale("log")
    negative_axis.axhline(1e-8, color="#a64036", linestyle="--", linewidth=1)
    negative_axis.set_ylabel(r"Negative mass fraction (display floor $10^{-20}$)")
    energy_axis.set_ylabel(r"Electric energy $W_E$")
    flux_index = FIELD_NAMES.index("dQ_dx")
    omitted = []
    for comparison in comparisons:
        base = lookup[comparison["base_variant"]]
        other = lookup[comparison["other_variant"]]
        times, ia, ib = _shared_history(base, other)
        a = np.asarray(base["arrays"]["fields"])[ia, flux_index]
        b = np.asarray(other["arrays"]["fields"])[ib, flux_index]
        if a.shape != b.shape:
            raise ValueError("Plot requires the same spatial observation grid")
        reference_rms = np.sqrt(np.trapezoid(np.mean(b ** 2, axis=-1), times)
                                / (times[-1] - times[0]))
        name = other["variant"]
        if reference_rms == 0:
            omitted.append(_LABELS.get(name, name))
            continue
        error_axis.plot(times, np.sqrt(np.mean((a - b) ** 2, axis=-1)) / reference_rms,
                        color=_COLORS.get(name), label=_LABELS.get(name, name), linewidth=1.8)
    error_axis.set_ylabel(r"RMS difference in $\partial_xQ$" + "\n/ comparison history RMS")
    if error_axis.lines:
        error_axis.legend(fontsize=9, loc="best")
    if omitted:
        error_axis.text(.02, .97, "Zero reference norm: " + ", ".join(omitted),
                        transform=error_axis.transAxes, va="top", fontsize=9)
    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.set_xlabel("Simulation time")
        ax.grid(alpha=.2)
        ax.text(.98, .97, f"({label})", transform=ax.transAxes,
                ha="right", va="top", fontsize=12,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path
