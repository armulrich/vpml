"""Benchmark plotting helpers for the Fourier-Hermite benchmark suite."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np

from .common import save_figure


def _spectral_with_bad(color: str = "#d1d5db"):
    cmap = plt.colormaps["Spectral_r"].copy()
    cmap.set_bad(color=color)
    return cmap


def _phase_field_masked(field: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_invalid(np.asarray(field, dtype=float))


def save_fig2_damping_profiles(
    payload: Dict[str, np.ndarray],
    *,
    p: int,
    outdir: str | Path,
) -> Path:
    """Save the Fig. 2 damping-profile comparison."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax = axes[0]
    n = payload["n_Nv20"]
    keys = sorted([k for k in payload if k.startswith("hyper_Nv20_alpha")], key=lambda s: int(s.split("alpha")[1]))
    for key in keys:
        alpha = int(key.split("alpha")[1])
        ax.semilogy(n, payload[key], label=fr"$\alpha={alpha}$")
    ax.semilogy(n, payload["filt_Nv20"], "k--", lw=2, label=fr"$p={p}$")
    ax.set_title(r"(a) Damping rate $N_v=20$")
    ax.set_xlabel("nth Hermite moment")
    ax.set_ylabel("damping rate")
    ax.set_ylim(1e-7, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    ax = axes[1]
    n = payload["n_Nv1000"]
    keys = sorted([k for k in payload if k.startswith("hyper_Nv1000_alpha")], key=lambda s: int(s.split("alpha")[1]))
    for key in keys:
        alpha = int(key.split("alpha")[1])
        ax.semilogy(n, payload[key], label=fr"$\alpha={alpha}$")
    ax.semilogy(n, payload["filt_Nv1000"], "k--", lw=2, label=fr"$p={p}$")
    ax.set_title(r"(b) Damping rate $N_v=10^3$")
    ax.set_xlabel("nth Hermite moment")
    ax.set_ylim(1e-16, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    return save_figure(fig, outdir / "fig2_damping_profiles.png", dpi=200)


def save_fig3_response_function(
    payload: Dict[str, np.ndarray],
    *,
    Nv_list: Sequence[int],
    xi_plot: np.ndarray,
    xi_min: float,
    xi_max: float,
    plot_floor: float,
    plot_ymax: float,
    has_learned: bool,
    outdir: str | Path,
) -> Dict[str, Path]:
    """Save the Fig. 3 response-function figures."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paper_methods = ["hyper_a1", "hyper_a2", "hyper_a3", "hyper_a4", "nonlocal_nm1", "nonlocal_nm3", "filter"]
    labels = {
        "hyper_a1": r"hyper $\alpha=1$",
        "hyper_a2": r"hyper $\alpha=2$",
        "hyper_a3": r"hyper $\alpha=3$",
        "hyper_a4": r"hyper $\alpha=4$",
        "nonlocal_nm1": r"nonlocal $N_m=1$",
        "nonlocal_nm3": r"nonlocal $N_m=3$",
        "filter": "filter",
        "learned": "learned",
    }
    styles = {
        "hyper_a1": dict(color="#4E79A7", ls="-", lw=2.0, marker="o", ms=4),
        "hyper_a2": dict(color="#F28E2B", ls="-", lw=2.0, marker="o", ms=4),
        "hyper_a3": dict(color="#59A14F", ls="-", lw=2.0, marker="o", ms=4),
        "hyper_a4": dict(color="#B07AA1", ls="-", lw=2.0, marker="o", ms=4),
        "nonlocal_nm1": dict(color="#EDC948", ls="--", lw=2.2, marker="o", ms=4),
        "nonlocal_nm3": dict(color="#FF9DA7", ls="--", lw=2.2, marker="o", ms=4),
        "filter": dict(color="#E15759", ls=":", lw=2.8, marker=None),
        "learned": dict(color="#BCBD22", ls="-", lw=2.6, marker="o", ms=4),
    }

    def masked_curve(name: str) -> np.ndarray:
        curve = np.asarray(payload[f"abs_err_{name}"], dtype=float).copy()
        curve[curve < plot_floor] = np.nan
        return curve

    outputs: Dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name in paper_methods:
        ax.semilogy(Nv_list, payload[f"err_{name}"], label=labels[name], **styles[name])
    ax.set_xlabel(r"$N_v$")
    ax.set_ylabel(r"$\|R_{N_v}^{aw}-R\|_2$")
    ax.set_title("Fig. 3(a) — response-function convergence (k=1)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(Nv_list)
    ax.legend(ncol=2, fontsize=8)
    outputs["fig3a"] = save_figure(fig, outdir / "fig3a_response_function_convergence.png", dpi=200)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name in paper_methods:
        ax.loglog(xi_plot, masked_curve(name), label=labels[name], **styles[name])
    ax.set_xlabel(r"$\xi := \omega/(\sqrt{2}|k|)$")
    ax.set_ylabel(r"$|R_{12}^{aw}(\xi)-R(\xi)|$")
    ax.set_title("Fig. 3(b) — absolute response-function error (Nv=12, k=1)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xi_min, xi_max)
    ax.set_ylim(plot_floor, plot_ymax)
    ax.legend(fontsize=8)
    outputs["fig3b"] = save_figure(fig, outdir / "fig3b_response_function_abs_error_Nv12.png", dpi=200)

    if has_learned:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
        axa, axb = axes

        for name in paper_methods:
            axa.semilogy(Nv_list, payload[f"err_{name}"], label=labels[name], **styles[name])
        axa.semilogy(Nv_list, payload["err_learned"], label=labels["learned"], **styles["learned"])
        axa.set_xlabel(r"$N_v$")
        axa.set_ylabel(r"$\|R_{N_v}^{aw}-R\|_2$")
        axa.set_title("Comparison — response-function convergence")
        axa.grid(True, alpha=0.3)
        axa.set_xticks(Nv_list)
        axa.legend(ncol=2, fontsize=8)

        for name in paper_methods:
            axb.loglog(xi_plot, masked_curve(name), label=labels[name], **styles[name])
        axb.loglog(xi_plot, masked_curve("learned"), label=labels["learned"], **styles["learned"])
        axb.set_xlabel(r"$\xi := \omega/(\sqrt{2}|k|)$")
        axb.set_ylabel(r"$|R_{12}^{aw}(\xi)-R(\xi)|$")
        axb.set_title("Comparison — absolute response-function error")
        axb.grid(True, alpha=0.3)
        axb.set_xlim(xi_min, xi_max)
        axb.set_ylim(plot_floor, plot_ymax)
        axb.legend(fontsize=8)
        outputs["fig3_comparison"] = save_figure(fig, outdir / "fig3_comparison.png", dpi=200)

    return outputs


def save_fig4_eigenvalue_scan(
    payload: Dict[str, np.ndarray],
    *,
    Nv: int,
    k_list: Sequence[float],
    outdir: str | Path,
) -> Dict[str, Path]:
    """Save the Fig. 4 eigenvalue-scan figures."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mu_vals = payload["mu_vals"]
    chi_vals = payload["chi_vals"]
    y_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    y_label = r"$\left(\gamma-\gamma^\ast\right)/\gamma$"
    colors = {
        0.5: "#1f77b4",
        1.0: "#2ca02c",
        1.5: "#d62728",
        2.0: "#ff7f0e",
    }
    titles = {
        "a": r"(a) Nonlocal closure with $N_m = 1$" + "\nSmith [38]",
        "b": r"(b) Artificial collisions $\alpha = 1$" + "\nLenard and Bernstein [29]",
        "c": r"(c) Artificial collisions $\alpha = 2$" + "\nCamporeale et al. [23]",
        "d": r"(d) Artificial collisions $\alpha = 3$",
        "e": r"(e) Artificial collisions $\alpha = 4$",
        "f": r"(f) Hou and Li [31] filter",
    }

    def style_axis(ax: plt.Axes, *, xlabel: str, xlim: Tuple[float, float], xticks: Sequence[float]) -> None:
        ax.axhline(0.0, color="0.45", lw=0.9, ls="--")
        ax.set_xlim(*xlim)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xticks(list(xticks))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:g}" for tick in y_ticks])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(y_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def draw_curves(ax: plt.Axes, x: np.ndarray, key_template: str, opt_template: str) -> None:
        for k in k_list:
            color = colors[float(k)]
            ax.plot(x, payload[key_template.format(k=k)], color=color, lw=2.0, label=fr"$k={k:g}$")
            ax.axvline(float(payload[opt_template.format(k=k)][0]), color=color, lw=1.2, ls="--", alpha=0.55)

    outputs: Dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for k in k_list:
        ax.plot(mu_vals, payload[f"err_nonlocal_k{k}"], color=colors[float(k)], lw=2.0, label=fr"$k={k:g}$")
    ax.axvline(float(payload["mu_opt"][0]), color=colors[2.0], lw=1.2, ls="--", alpha=0.55)
    style_axis(ax, xlabel=r"$\mu_{N_v-1}$", xlim=(-3.0, 0.0), xticks=[-3, -2, -1, 0])
    ax.set_title(titles["a"], fontsize=12, pad=12)
    ax.legend(loc="lower left")
    outputs["fig4a"] = save_figure(fig, outdir / f"fig4a_nonlocal_scan_Nv{Nv}.png", dpi=200)

    for alpha, xlim, xticks, panel in [
        (1, (0.0, 10.0), [0, 5, 10], "b"),
        (2, (0.0, 25.0), [5, 10, 15, 20, 25], "c"),
        (3, (0.0, 25.0), [5, 10, 15, 20, 25], "d"),
        (4, (0.0, 20.0), [5, 10, 15, 20], "e"),
    ]:
        nu_vals = payload[f"nu_vals_a{alpha}"]
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        draw_curves(ax, nu_vals, f"err_hyper_a{alpha}_k{{k}}", f"opt_hyper_a{alpha}_k{{k}}")
        style_axis(ax, xlabel=r"$\nu$", xlim=xlim, xticks=xticks)
        ax.set_title(titles[panel], fontsize=12, pad=12)
        outputs[f"fig4_hyper_a{alpha}"] = save_figure(fig, outdir / f"fig4_hyper_a{alpha}_scan_Nv{Nv}.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    draw_curves(ax, chi_vals, "err_filter_k{k}", "opt_filter_k{k}")
    style_axis(ax, xlabel=r"$\chi/\Delta t$", xlim=(0.0, 15.0), xticks=[5, 10, 15])
    ax.set_title(titles["f"], fontsize=12, pad=12)
    outputs["fig4f"] = save_figure(fig, outdir / f"fig4f_filter_scan_Nv{Nv}.png", dpi=200)

    fig, axes = plt.subplots(2, 3, figsize=(14, 10), constrained_layout=True)
    axa, axb, axc, axd, axe, axf = axes.ravel()

    for k in k_list:
        axa.plot(mu_vals, payload[f"err_nonlocal_k{k}"], color=colors[float(k)], lw=2.0, label=fr"$k={k:g}$")
    axa.axvline(float(payload["mu_opt"][0]), color=colors[2.0], lw=1.2, ls="--", alpha=0.55)
    style_axis(axa, xlabel=r"$\mu_{N_v-1}$", xlim=(-3.0, 0.0), xticks=[-3, -2, -1, 0])
    axa.set_title(titles["a"], fontsize=12, pad=12)
    axa.legend(loc="lower left")

    for ax, alpha, xlim, xticks, panel in [
        (axb, 1, (0.0, 10.0), [0, 5, 10], "b"),
        (axc, 2, (0.0, 25.0), [5, 10, 15, 20, 25], "c"),
        (axd, 3, (0.0, 25.0), [5, 10, 15, 20, 25], "d"),
        (axe, 4, (0.0, 20.0), [5, 10, 15, 20], "e"),
    ]:
        nu_vals = payload[f"nu_vals_a{alpha}"]
        draw_curves(ax, nu_vals, f"err_hyper_a{alpha}_k{{k}}", f"opt_hyper_a{alpha}_k{{k}}")
        style_axis(ax, xlabel=r"$\nu$", xlim=xlim, xticks=xticks)
        ax.set_title(titles[panel], fontsize=12, pad=12)

    draw_curves(axf, chi_vals, "err_filter_k{k}", "opt_filter_k{k}")
    style_axis(axf, xlabel=r"$\chi/\Delta t$", xlim=(0.0, 15.0), xticks=[5, 10, 15])
    axf.set_title(titles["f"], fontsize=12, pad=12)

    outputs["fig4_paper_style"] = save_figure(fig, outdir / f"fig4_paper_style_Nv{Nv}.png", dpi=220)
    return outputs


def save_fig10_nonlinear_landau_phase_space(
    payload: Dict[str, np.ndarray],
    *,
    times: Sequence[float],
    vmin: float,
    vmax: float,
    time_key_fn: Callable[[float], str],
    outdir: str | Path,
) -> Path:
    """Save the Fig. 10 nonlinear Landau phase-space panel."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(payload["x"], dtype=float)
    v = np.asarray(payload["v"], dtype=float)
    times = tuple(float(t) for t in np.asarray(times, dtype=float))

    panel_specs = [
        ("truncation", "(a) Collisionless with\nclosure by truncation"),
        ("nonlocal_nm1", r"(b) Nonlocal closure" + "\n" + r"with $N_m = 1$"),
        ("hyper_a1", r"(c) Artificial collisions $\alpha = 1$" + "\n" + "Lenard and Bernstein"),
        ("hyper_a2", r"(d) Artificial collisions $\alpha = 2$" + "\n" + "Camporeale et al."),
        ("hyper_a3", r"(e) Artificial collisions $\alpha = 3$"),
        ("filter", "(f) Hou and Li filter"),
    ]

    n_panels = len(panel_specs)
    ncols = 3
    nrows = int(math.ceil(n_panels / ncols))
    fig = plt.figure(figsize=(14, 4.8 * nrows), constrained_layout=True)
    outer = fig.add_gridspec(nrows, ncols, wspace=0.25, hspace=0.2)

    xticks = [0.0, 2.0 * math.pi, 4.0 * math.pi]
    xticklabels = ["0", r"$2\pi$", r"$4\pi$"]

    for idx, (name, title) in enumerate(panel_specs):
        row = idx // 3
        col = idx % 3
        sub = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=outer[row, col],
            width_ratios=[24, 1.2],
            hspace=0.18,
            wspace=0.08,
        )
        for ti, t in enumerate(times):
            ax = fig.add_subplot(sub[ti, 0])
            cax = fig.add_subplot(sub[ti, 1])
            field = np.asarray(payload[f"{name}_f_{time_key_fn(float(t))}"], dtype=float)
            im = ax.pcolormesh(
                x,
                v,
                field,
                shading="auto",
                vmin=float(vmin),
                vmax=float(vmax),
                cmap="Spectral_r",
            )
            if ti == 0:
                ax.set_title(title, fontsize=11, pad=6)
            if ti == 1:
                ax.set_xlabel("x")
                ax.set_xticks(xticks, xticklabels)
            else:
                ax.set_xticks(xticks, [])
            ax.set_ylabel("v")
            ax.set_yticks([-4, -2, 0, 2, 4])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.set_ylabel(rf"$f(x,v,t={t:g})$", rotation=90)

    for idx in range(len(panel_specs), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(outer[row, col])
        ax.axis("off")

    return save_figure(fig, outdir / "fig10_nonlinear_landau_phase_space.png", dpi=220)


def save_fig10_learned_comparison_phase_space(
    payload: Dict[str, np.ndarray],
    *,
    times: Sequence[float],
    vmin: float,
    vmax: float,
    time_key_fn: Callable[[float], str],
    outdir: str | Path,
    baseline_name: str = "nonlocal_nm1",
    baseline_title: str = r"Nonlocal closure ($N_m=1$)",
    learned_title: str = "Learned interface closure",
) -> Path:
    """Save a 2x2 phase-space comparison between a baseline closure and the learned closure."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(payload["x"], dtype=float)
    v = np.asarray(payload["v"], dtype=float)
    times = tuple(float(t) for t in np.asarray(times, dtype=float))
    if len(times) != 2:
        raise ValueError("learned comparison expects exactly two snapshot times")

    panel_specs = [
        (baseline_name, baseline_title),
        ("learned", learned_title),
    ]
    for name, _ in panel_specs:
        for t in times:
            key = f"{name}_f_{time_key_fn(float(t))}"
            if key not in payload:
                raise ValueError(f"Missing payload key for learned comparison: {key}")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    xticks = [0.0, 2.0 * math.pi, 4.0 * math.pi]
    xticklabels = ["0", r"$2\pi$", r"$4\pi$"]
    cmap = _spectral_with_bad()

    for row, (name, row_title) in enumerate(panel_specs):
        for col, t in enumerate(times):
            ax = axes[row, col]
            field_raw = np.asarray(payload[f"{name}_f_{time_key_fn(float(t))}"], dtype=float)
            field = _phase_field_masked(field_raw)
            im = ax.pcolormesh(
                x,
                v,
                field,
                shading="auto",
                vmin=float(vmin),
                vmax=float(vmax),
                cmap=cmap,
            )
            if not np.isfinite(field_raw).any():
                ax.set_facecolor("#d1d5db")
                ax.text(
                    0.5,
                    0.5,
                    "nonfinite snapshot",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#334155",
                    bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
                )
            if row == 0:
                ax.set_title(rf"$t={t:g}$", fontsize=11, pad=6)
            if col == 0:
                ax.set_ylabel("v")
                ax.text(
                    -0.18,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                )
            else:
                ax.set_ylabel("")
            ax.set_xlabel("x")
            ax.set_xticks(xticks, xticklabels)
            ax.set_yticks([-4, -2, 0, 2, 4])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.set_ylabel(rf"$f(x,v,t={t:g})$", rotation=90)

    return save_figure(fig, outdir / "fig10_learned_vs_nonlocal_phase_space.png", dpi=220)


def save_fig10_learned_comparison_nv_sweep_phase_space(
    payload: Dict[str, np.ndarray],
    *,
    nv_list: Sequence[int],
    times: Sequence[float],
    row_labels: Sequence[str],
    vmin: float,
    vmax: float,
    time_key_fn: Callable[[float], str],
    outdir: str | Path,
) -> Path:
    """Save a nonlinear Landau phase-space sweep across deployment Nv values."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(payload["x"], dtype=float)
    v = np.asarray(payload["v"], dtype=float)
    times = tuple(float(t) for t in np.asarray(times, dtype=float))
    nv_list = tuple(int(vv) for vv in nv_list)
    if len(times) != 2:
        raise ValueError("nonlinear Landau Nv sweep expects exactly two snapshot times")
    if len(row_labels) != len(nv_list):
        raise ValueError("row_labels length must match nv_list length")

    fig = plt.figure(figsize=(15.0, 2.6 * len(nv_list) + 1.0), constrained_layout=True)
    grid = fig.add_gridspec(len(nv_list), 4, wspace=0.08, hspace=0.18)
    xticks = [0.0, 2.0 * math.pi, 4.0 * math.pi]
    xticklabels = ["0", r"$2\pi$", r"$4\pi$"]
    column_titles = [
        rf"Nonlocal, $t={times[0]:g}$",
        rf"Nonlocal, $t={times[1]:g}$",
        rf"Learned, $t={times[0]:g}$",
        rf"Learned, $t={times[1]:g}$",
    ]
    cmap = _spectral_with_bad()

    mesh_ref = None
    all_axes = []
    for row, (Nv, row_label) in enumerate(zip(nv_list, row_labels)):
        for col, (method_name, t) in enumerate(
            [
                ("nonlocal", times[0]),
                ("nonlocal", times[1]),
                ("learned", times[0]),
                ("learned", times[1]),
            ]
        ):
            ax = fig.add_subplot(grid[row, col])
            all_axes.append(ax)
            key = f"nv{int(Nv)}_{method_name}_f_{time_key_fn(float(t))}"
            if key not in payload:
                raise ValueError(f"Missing sweep phase-space payload key: {key}")
            field_raw = np.asarray(payload[key], dtype=float)
            field = _phase_field_masked(field_raw)
            mesh_ref = ax.pcolormesh(
                x,
                v,
                field,
                shading="auto",
                vmin=float(vmin),
                vmax=float(vmax),
                cmap=cmap,
            )
            if not np.isfinite(field_raw).any():
                ax.set_facecolor("#d1d5db")
                ax.text(
                    0.5,
                    0.5,
                    "nonfinite snapshot",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color="#334155",
                    bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
                )
            if row == 0:
                ax.set_title(column_titles[col], fontsize=10.5, pad=6)
            ax.set_xlabel("x")
            ax.set_xticks(xticks, xticklabels)
            ax.set_yticks([-4, -2, 0, 2, 4])
            if col == 0:
                ax.set_ylabel("v")
                ax.text(
                    -0.34,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10.5,
                )
            else:
                ax.set_ylabel("")

    if mesh_ref is not None:
        fig.colorbar(mesh_ref, ax=all_axes, fraction=0.018, pad=0.01)
    return save_figure(fig, outdir / "fig10_learned_vs_nonlocal_nv_sweep_phase_space.png", dpi=220)


def save_linear_landau_time(
    payload: Dict[str, np.ndarray],
    *,
    method: str,
    outdir: str | Path,
) -> Path:
    """Save the single-method linear Landau damping time-series figure."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t = payload["times"]
    E05 = payload["E_abs_k0p5"]
    E15 = payload["E_abs_k1p5"]
    g05 = float(payload["gamma_k0p5"][0])
    g15 = float(payload["gamma_k1p5"][0])

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.semilogy(t, E05, label=r"$|E_{k=0.5}(t)|$")
    ax.semilogy(t, E15, label=r"$|E_{k=1.5}(t)|$")
    ax.semilogy(t, E05[0] * np.exp(g05 * t), "k--", lw=1.5, label=r"analytic $k=0.5$")
    ax.semilogy(t, E15[0] * np.exp(g15 * t), "k:", lw=1.5, label=r"analytic $k=1.5$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$|E_k(t)|$")
    backend_label = "CNAB2" if method == "learned" else "implicit midpoint/JFNK"
    ax.set_title(f"Linear Landau damping — {backend_label}, method={method}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, outdir / f"linear_landau_{method}.png", dpi=200)


def save_linear_landau_comparison(
    results: Dict[str, Dict[str, np.ndarray] | object],
    outdir: str | Path,
) -> Path:
    """Save the multi-method linear Landau damping comparison figure."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    method_order = ["truncation", "hyper", "filter", "nonlocal", "learned"]
    method_labels = {
        "truncation": "truncation",
        "hyper": "hyper",
        "filter": "filter",
        "nonlocal": "nonlocal",
        "learned": "learned",
    }
    method_styles = {
        "truncation": dict(color="#4b5563", ls="-", lw=1.8),
        "hyper": dict(color="#2563eb", ls="-", lw=1.8),
        "filter": dict(color="#059669", ls="-", lw=1.8),
        "nonlocal": dict(color="#d97706", ls="-", lw=1.8),
        "learned": dict(color="#111827", ls="-", lw=2.4),
    }

    def payload_for(method: str):
        item = results[method]
        return item.payload if hasattr(item, "payload") else item

    first_method = method_order[0] if method_order[0] in results else next(iter(results))
    first = payload_for(first_method)
    t = first["times"]
    g05 = float(first["gamma_k0p5"][0])
    g15 = float(first["gamma_k1p5"][0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, data_key, gamma, title in [
        (axes[0], "E_abs_k0p5", g05, r"Comparison — $k=0.5$"),
        (axes[1], "E_abs_k1p5", g15, r"Comparison — $k=1.5$"),
    ]:
        for method in method_order:
            if method not in results:
                continue
            ax.semilogy(
                t,
                payload_for(method)[data_key],
                label=method_labels[method],
                **method_styles[method],
            )
        ref_curve = first[data_key][0] * np.exp(gamma * t)
        ax.semilogy(t, ref_curve, "k--", lw=1.4, label="analytic")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$|E_k(t)|$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    return save_figure(fig, outdir / "linear_landau_comparison.png", dpi=200)
