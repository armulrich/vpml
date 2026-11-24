"""Plot electric-energy curves from a saved Gaussian rollout evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from vpml.io.serialization import read_json


def energy_figure(directory: Path) -> Figure:
    """Overlay the kinetic reference and both autonomous forecasts for each strength."""
    report = read_json(directory / "report.json")
    cases = sorted(report["cases"], key=lambda case: case["strength"])
    if not cases:
        raise ValueError("The evaluation has no cases to plot")
    curves = (
        ("reference", "Kinetic reference", "#246BB5", "-"),
        ("initial", "Untrained model", "#666B73", "--"),
        ("trained", f"Model at update {report['checkpoint_step']}", "#813EA4", "-"),
    )
    fig = Figure(figsize=(max(5.4, 3.05 * len(cases)), 3.7), layout="constrained")
    axes = fig.subplots(1, len(cases), sharey=True, squeeze=False)[0]
    min_positive, maximum = np.inf, 0.
    for ax, case in zip(axes, cases, strict=True):
        stops = []
        ax.set_yscale("log", nonpositive="mask")
        ax.set_autoscaley_on(False)
        with np.load(directory / f"{case['case_id']}.npz", allow_pickle=False) as data:
            times = data["times"]
            if times.ndim != 1 or len(times) < 2 or not np.all(np.diff(times) > 0):
                raise ValueError(f"Invalid time grid for {case['case_id']}")
            for key, label, color, style in curves:
                energy = data[f"{key}_energy"]
                if energy.ndim != 1 or len(energy) > len(times) or not np.isfinite(energy).all():
                    raise ValueError(f"Invalid {key} energy series for {case['case_id']}")
                if key == "reference" and len(energy) != len(times):
                    raise ValueError(f"Incomplete kinetic reference for {case['case_id']}")
                positive = energy[energy > 0]
                if positive.size:
                    min_positive = min(min_positive, positive.min())
                    maximum = max(maximum, positive.max())
                ax.plot(times[:len(energy)], np.ma.masked_less_equal(energy, 0), color=color,
                        ls=style, lw=1.25, label=label)
                if len(energy) < len(times):
                    failed_step = case[key].get("first_nonfinite_step")
                    if failed_step is None:
                        stops.append(f"{label} incomplete")
                    else:
                        stops.append(f"{label} nonfinite at $t={times[failed_step]:g}$")
                    if len(energy) and energy[-1] > 0:
                        ax.plot(times[len(energy) - 1], energy[-1], marker="x", color=color, ms=6)
            ax.set_xlim(times[0], times[-1])
            ax.set_xticks(np.linspace(times[0], times[-1], 4))
        if stops:
            ax.text(.02, .98, "\n".join(stops), transform=ax.transAxes, va="top", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
        ax.set_title(rf"$\lambda = {case['strength']:g}$")
        ax.set_xlabel(r"$\omega_p t$", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", which="major", color="#DFE2E7", lw=.65)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$W_E(t)=\frac{1}{2}\int_0^L E(x,t)^2\,\mathrm{d}x$", fontsize=12)
    # Log-axis padding and outlying ticks can overflow near a diverging forecast.
    # Enclose every observed positive value without asking for an unrepresentable tick.
    if maximum and (np.log10(maximum) > 280 or np.log10(min_positive) < -280):
        lo, hi = np.log10([min_positive, maximum])
        exponents = np.arange(np.ceil(lo), np.floor(hi) + 1, max(1, np.ceil((hi - lo) / 7)))
        axes[0].set_yticks(np.power(10., exponents))
        axes[0].yaxis.set_minor_locator(NullLocator())
        axes[0].set_ylim(min_positive / 2 if lo > -300 else min_positive,
                         maximum * 2 if hi < 307 else maximum)
    else:
        axes[0].autoscale(enable=True, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False,
               fontsize=9, handlelength=2.5)
    return fig


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory with report.json and the saved curves")
    parser.add_argument("--output", type=Path, help="Figure path (default: DIRECTORY/electric-energy.png)")
    args = parser.parse_args(argv)
    path = args.output or args.directory / "electric-energy.png"
    energy_figure(args.directory).savefig(path, dpi=180)


if __name__ == "__main__":
    main()
