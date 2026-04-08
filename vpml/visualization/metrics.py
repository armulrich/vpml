"""Visualization helpers for a posteriori metric evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from vpml.metrics import FourierFieldComparison, GrowthComparisonResult


def _safe_log10(values: np.ndarray, floor: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), np.maximum(arr, float(floor)), np.nan)
    return np.log10(arr)


def _safe_log10_masked(values: np.ndarray, floor: float) -> np.ma.MaskedArray:
    return np.ma.masked_invalid(_safe_log10(values, floor))


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("values must be nonempty")
    if values.size == 1:
        delta = 0.5
        return np.array([values[0] - delta, values[0] + delta], dtype=np.float64)
    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[first], mids, [last]]).astype(np.float64)


def _plot_semilogy_finite(ax: plt.Axes, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    if np.count_nonzero(mask) >= 2:
        ax.semilogy(x[mask], y[mask], **kwargs)


def _plot_semilogy_capped(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    y_cap: float,
    annotate_divergence: bool = False,
    **kwargs,
) -> None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return
    x_f = x[mask]
    y_f = y[mask]
    over = y_f > float(y_cap)
    if np.any(over):
        first = int(np.argmax(over))
        y_plot = np.minimum(y_f[: first + 1], float(y_cap))
        ax.semilogy(x_f[: first + 1], y_plot, **kwargs)
        if annotate_divergence:
            color = kwargs.get("color", "#111827")
            ax.axvline(float(x_f[first]), color=color, lw=1.0, ls=":", alpha=0.65)
    else:
        ax.semilogy(x_f, y_f, **kwargs)


def _format_metric_value(value: float) -> str:
    return "inf" if not np.isfinite(value) else f"{float(value):.3e}"


def _finite_log_limits(arrays: Sequence[np.ndarray], floor: float) -> tuple[float, float]:
    finite_chunks = []
    for arr in arrays:
        log_arr = _safe_log10(arr, floor)
        finite = log_arr[np.isfinite(log_arr)]
        if finite.size:
            finite_chunks.append(finite)
    if not finite_chunks:
        default = np.log10(float(floor))
        return float(default), float(default + 1.0)
    return (
        float(min(np.min(chunk) for chunk in finite_chunks)),
        float(max(np.max(chunk) for chunk in finite_chunks)),
    )


def _set_semilogy_finite_ylim(
    ax: plt.Axes,
    arrays: Sequence[np.ndarray],
    *,
    floor: float = 1e-14,
    max_decades: float = 18.0,
) -> None:
    positive_chunks = []
    for arr in arrays:
        values = np.asarray(arr, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values) & (values > 0.0)]
        if finite.size:
            positive_chunks.append(finite)
    if not positive_chunks:
        return

    y_min = float(max(min(np.min(chunk) for chunk in positive_chunks), floor))
    y_max = float(max(np.max(chunk) for chunk in positive_chunks))
    if y_max <= y_min:
        y_max = y_min * 10.0

    max_ratio = 10.0 ** float(max_decades)
    ratio = y_max / y_min if y_min > 0.0 else np.inf
    if np.isfinite(ratio) and ratio > max_ratio:
        y_max = y_min * max_ratio
    elif not np.isfinite(ratio):
        y_max = y_min * max_ratio

    ax.set_ylim(y_min, y_max)


def _reference_semilogy_cap(reference: np.ndarray, *, floor: float = 1e-14, cap_factor: float = 1e6) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    finite = ref[np.isfinite(ref) & (ref > 0.0)]
    if finite.size == 0:
        return float(floor * cap_factor)
    return float(max(np.max(finite) * float(cap_factor), floor * cap_factor))


def _magma_with_bad(color: str = "#d1d5db"):
    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad(color=color)
    return cmap


def _first_nonfinite_time(times: np.ndarray, values: np.ndarray) -> Optional[float]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    arr = np.asarray(values)
    if arr.ndim == 1:
        bad = ~np.isfinite(arr)
    else:
        bad = np.any(~np.isfinite(arr), axis=tuple(range(1, arr.ndim)))
    idx = np.where(bad)[0]
    if idx.size == 0:
        return None
    return float(times[int(idx[0])])


@dataclass(frozen=True)
class GrowthSweepCase:
    """One nonlinear Metric 1 sweep case for a specific deployment Nv."""

    Nv: int
    times_theta: np.ndarray
    energy_theta: np.ndarray
    comparison: GrowthComparisonResult
    in_training_targets: bool = False
    beyond_training_range: bool = False


@dataclass(frozen=True)
class FieldSweepCase:
    """One nonlinear Metric 2 sweep case for a specific deployment Nv."""

    Nv: int
    comparison: FourierFieldComparison
    epsilon_E: float
    in_training_targets: bool = False
    beyond_training_range: bool = False


def plot_growth_metric(
    times_hr: np.ndarray,
    energy_hr: np.ndarray,
    times_theta: np.ndarray,
    energy_theta: np.ndarray,
    comparison: GrowthComparisonResult,
    *,
    title: str,
) -> plt.Figure:
    """Plot Metric 1 energy traces and fitted early-time lines."""

    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    _plot_semilogy_finite(ax, times_hr, energy_hr, label=r"HR $\mathcal{E}_E(t)$", color="#1d4ed8", lw=2.0)
    _plot_semilogy_finite(ax, times_theta, energy_theta, label=r"$\theta$ $\mathcal{E}_E(t)$", color="#b91c1c", lw=2.0)
    ax.axvspan(comparison.t_a, comparison.t_b, color="#d1d5db", alpha=0.25, label="fit window")

    fit_t_hr = np.linspace(comparison.fit_hr.t_a, comparison.fit_hr.t_b, 200, dtype=np.float64)
    fit_t_theta = np.linspace(comparison.fit_theta.t_a, comparison.fit_theta.t_b, 200, dtype=np.float64)
    fit_y_hr = np.exp(comparison.fit_hr.intercept + 2.0 * comparison.fit_hr.gamma_grow * fit_t_hr)
    fit_y_theta = np.exp(comparison.fit_theta.intercept + 2.0 * comparison.fit_theta.gamma_grow * fit_t_theta)
    _plot_semilogy_finite(ax, fit_t_hr, fit_y_hr, color="#1e3a8a", lw=1.8, ls="--", label=r"HR fit")
    _plot_semilogy_finite(ax, fit_t_theta, fit_y_theta, color="#7f1d1d", lw=1.8, ls="--", label=r"$\theta$ fit")
    _set_semilogy_finite_ylim(ax, [energy_hr, energy_theta, fit_y_hr, fit_y_theta])

    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathcal{E}_E(t)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    ax.text(
        0.99,
        0.99,
        "\n".join(
            [
                rf"$\gamma_{{grow}}^{{HR}}={comparison.gamma_grow_hr:.4e}$",
                rf"$\gamma_{{grow}}^\theta={comparison.gamma_grow_theta:.4e}$",
                rf"$\varepsilon_{{grow}}={comparison.epsilon_grow:.4e}$",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.95},
    )
    return fig


def plot_field_metric(
    comparison: FourierFieldComparison,
    *,
    title: str,
    log_floor: float = 1e-14,
) -> plt.Figure:
    """Plot Metric 2 as HR/theta/log-difference time-by-mode panels."""

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    time_edges = _centers_to_edges(comparison.times)
    k_edges = _centers_to_edges(comparison.selected_k)
    cmap = _magma_with_bad()
    panels = [
        (_safe_log10_masked(np.abs(comparison.E_hat_hr), log_floor).T, r"$\log_{10}|\hat E_k^{HR}(t)|$", axes[0]),
        (_safe_log10_masked(np.abs(comparison.E_hat_theta), log_floor).T, r"$\log_{10}|\hat E_k^\theta(t)|$", axes[1]),
        (_safe_log10_masked(np.abs(comparison.E_hat_theta - comparison.E_hat_hr), log_floor).T, r"$\log_{10}|\hat E_k^\theta(t)-\hat E_k^{HR}(t)|$", axes[2]),
    ]
    for data, panel_title, ax in panels:
        mesh = ax.pcolormesh(time_edges, k_edges, data, shading="auto", cmap=cmap)
        ax.set_title(panel_title)
        ax.set_xlabel("t")
        ax.set_ylabel("k")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    return fig


def plot_metric_summary(
    times_hr: np.ndarray,
    energy_hr: np.ndarray,
    times_theta: np.ndarray,
    energy_theta: np.ndarray,
    growth: GrowthComparisonResult,
    field: FourierFieldComparison,
    *,
    title: str,
    epsilon_E: Optional[float] = None,
    log_floor: float = 1e-14,
) -> plt.Figure:
    """Combine Metric 1 and Metric 2 visuals into a single figure."""

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])

    ax_top = fig.add_subplot(grid[0, :])
    _plot_semilogy_finite(ax_top, times_hr, energy_hr, label=r"HR $\mathcal{E}_E(t)$", color="#1d4ed8", lw=2.0)
    _plot_semilogy_finite(ax_top, times_theta, energy_theta, label=r"$\theta$ $\mathcal{E}_E(t)$", color="#b91c1c", lw=2.0)
    ax_top.axvspan(growth.t_a, growth.t_b, color="#d1d5db", alpha=0.25, label="fit window")
    fit_t_hr = np.linspace(growth.fit_hr.t_a, growth.fit_hr.t_b, 200, dtype=np.float64)
    fit_t_theta = np.linspace(growth.fit_theta.t_a, growth.fit_theta.t_b, 200, dtype=np.float64)
    _plot_semilogy_finite(
        ax_top,
        fit_t_hr,
        np.exp(growth.fit_hr.intercept + 2.0 * growth.fit_hr.gamma_grow * fit_t_hr),
        color="#1e3a8a",
        lw=1.8,
        ls="--",
    )
    _plot_semilogy_finite(
        ax_top,
        fit_t_theta,
        np.exp(growth.fit_theta.intercept + 2.0 * growth.fit_theta.gamma_grow * fit_t_theta),
        color="#7f1d1d",
        lw=1.8,
        ls="--",
    )
    _set_semilogy_finite_ylim(
        ax_top,
        [
            energy_hr,
            energy_theta,
            np.exp(growth.fit_hr.intercept + 2.0 * growth.fit_hr.gamma_grow * fit_t_hr),
            np.exp(growth.fit_theta.intercept + 2.0 * growth.fit_theta.gamma_grow * fit_t_theta),
        ],
    )
    ax_top.set_xlabel("t")
    ax_top.set_ylabel(r"$\mathcal{E}_E(t)$")
    ax_top.set_title(title)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=9, ncol=2)
    text_lines = [
        rf"$\gamma_{{grow}}^{{HR}}={growth.gamma_grow_hr:.4e}$",
        rf"$\gamma_{{grow}}^\theta={growth.gamma_grow_theta:.4e}$",
        rf"$\varepsilon_{{grow}}={growth.epsilon_grow:.4e}$",
    ]
    if epsilon_E is not None:
        text_lines.append(rf"$\varepsilon_E={float(epsilon_E):.4e}$")
    ax_top.text(
        0.99,
        0.99,
        "\n".join(text_lines),
        transform=ax_top.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.95},
    )

    time_edges = _centers_to_edges(field.times)
    k_edges = _centers_to_edges(field.selected_k)
    cmap = _magma_with_bad()
    heatmaps = [
        (_safe_log10_masked(np.abs(field.E_hat_hr), log_floor).T, r"$\log_{10}|\hat E_k^{HR}(t)|$"),
        (_safe_log10_masked(np.abs(field.E_hat_theta), log_floor).T, r"$\log_{10}|\hat E_k^\theta(t)|$"),
        (_safe_log10_masked(np.abs(field.E_hat_theta - field.E_hat_hr), log_floor).T, r"$\log_{10}|\hat E_k^\theta(t)-\hat E_k^{HR}(t)|$"),
    ]
    for idx, (data, panel_title) in enumerate(heatmaps):
        ax = fig.add_subplot(grid[1, idx])
        mesh = ax.pcolormesh(time_edges, k_edges, data, shading="auto", cmap=cmap)
        ax.set_title(panel_title)
        ax.set_xlabel("t")
        ax.set_ylabel("k")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

    return fig


def _nv_status_suffix(*, in_training_targets: bool, beyond_training_range: bool) -> str:
    if beyond_training_range:
        return " (out of train range)"
    if in_training_targets:
        return ""
    return " (unseen)"


def plot_growth_metric_sweep(
    times_hr: np.ndarray,
    energy_hr: np.ndarray,
    cases: Sequence[GrowthSweepCase],
    *,
    title: str,
) -> plt.Figure:
    """Plot Metric 1 across a sweep of deployment Nv values on one figure."""

    if not cases:
        raise ValueError("cases must be nonempty")

    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    _plot_semilogy_finite(ax, times_hr, energy_hr, color="#1d4ed8", lw=2.6, label=r"HR $\mathcal{E}_E(t)$")
    y_cap = _reference_semilogy_cap(energy_hr)

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(cases)))
    summary_lines = []
    fit_window = (cases[0].comparison.t_a, cases[0].comparison.t_b)
    ax.axvspan(fit_window[0], fit_window[1], color="#d1d5db", alpha=0.22, label="fit window")

    for color, case in zip(colors, cases):
        label = rf"$N_v={int(case.Nv)}$"
        if case.beyond_training_range:
            label += " (out)"
        elif not case.in_training_targets:
            label += " (unseen)"
        _plot_semilogy_capped(
            ax,
            np.asarray(case.times_theta, dtype=np.float64),
            np.asarray(case.energy_theta, dtype=np.float64),
            y_cap=y_cap,
            annotate_divergence=True,
            color=color,
            lw=2.0,
            label=label,
        )
        fit_t = np.linspace(case.comparison.fit_theta.t_a, case.comparison.fit_theta.t_b, 200, dtype=np.float64)
        fit_y = np.exp(
            case.comparison.fit_theta.intercept + 2.0 * case.comparison.fit_theta.gamma_grow * fit_t
        )
        _plot_semilogy_capped(ax, fit_t, fit_y, y_cap=y_cap, color=color, lw=1.4, ls="--", alpha=0.95)
        summary_lines.append(
            rf"$N_v={int(case.Nv)}$: $\varepsilon_{{grow}}={_format_metric_value(case.comparison.epsilon_grow)}$"
        )
    finite_hr = np.asarray(energy_hr, dtype=np.float64)
    y_min = np.nanmin(finite_hr[np.isfinite(finite_hr) & (finite_hr > 0.0)]) if np.any(np.isfinite(finite_hr) & (finite_hr > 0.0)) else 1e-14
    ax.set_ylim(max(float(y_min), 1e-14), float(y_cap))

    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathcal{E}_E(t)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    ax.text(
        0.99,
        0.99,
        "\n".join(
            [
                rf"$\gamma_{{grow}}^{{HR}}={cases[0].comparison.gamma_grow_hr:.4e}$",
                *summary_lines,
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.95},
    )
    return fig


def plot_field_metric_sweep(
    cases: Sequence[FieldSweepCase],
    *,
    title: str,
    log_floor: float = 1e-14,
) -> plt.Figure:
    """Plot Metric 2 across a sweep of deployment Nv values."""

    if not cases:
        raise ValueError("cases must be nonempty")

    fig = plt.figure(figsize=(12.8, 2.9 * len(cases) + 3.2), constrained_layout=True)
    grid = fig.add_gridspec(
        len(cases) + 1,
        3,
        height_ratios=[1.1] + [1.0] * len(cases),
        width_ratios=[1.0, 1.0, 0.045],
    )

    ax_top = fig.add_subplot(grid[0, :2])
    nv_vals = np.asarray([int(case.Nv) for case in cases], dtype=np.float64)
    eps_vals = np.asarray([float(case.epsilon_E) for case in cases], dtype=np.float64)
    markers = [
        "^" if case.beyond_training_range else ("o" if case.in_training_targets else "s")
        for case in cases
    ]
    finite_mask = np.isfinite(eps_vals)
    finite_eps = eps_vals[finite_mask]
    if finite_eps.size:
        y_cap = float(np.max(finite_eps) * 1.2 if np.max(finite_eps) > 0.0 else 1.0)
    else:
        y_cap = 1.0
    for nv, eps, marker in zip(nv_vals, eps_vals, markers):
        if np.isfinite(eps):
            ax_top.plot([nv], [eps], marker=marker, color="#b91c1c", ms=7)
        else:
            ax_top.plot([nv], [y_cap], marker=marker, color="#991b1b", ms=7)
            ax_top.text(nv, y_cap, " inf", fontsize=8, va="bottom", ha="left", color="#991b1b")
    if np.count_nonzero(finite_mask) >= 2:
        ax_top.plot(nv_vals[finite_mask], eps_vals[finite_mask], color="#b91c1c", lw=1.8)
    ax_top.set_ylim(0.0, y_cap * 1.15)
    ax_top.set_xscale("log", base=2)
    ax_top.set_xticks(nv_vals, [str(int(v)) for v in nv_vals])
    ax_top.set_xlabel(r"$N_v$")
    ax_top.set_ylabel(r"$\varepsilon_E$")
    ax_top.set_title(title)
    ax_top.grid(True, alpha=0.3)

    for row, case in enumerate(cases, start=1):
        comp = case.comparison
        time_edges = _centers_to_edges(comp.times)
        k_edges = _centers_to_edges(comp.selected_k)
        hr_ax = fig.add_subplot(grid[row, 0])
        theta_ax = fig.add_subplot(grid[row, 1])
        cax = fig.add_subplot(grid[row, 2])
        cmap = _magma_with_bad()

        hr_log = _safe_log10_masked(np.abs(comp.E_hat_hr), log_floor).T
        theta_log = _safe_log10_masked(np.abs(comp.E_hat_theta), log_floor).T
        row_vmin, row_vmax = _finite_log_limits([np.abs(comp.E_hat_hr)], log_floor)
        if row_vmax <= row_vmin:
            row_vmax = row_vmin + 1.0

        mesh_ref = hr_ax.pcolormesh(
            time_edges,
            k_edges,
            hr_log,
            shading="auto",
            cmap=cmap,
            vmin=row_vmin,
            vmax=row_vmax,
        )
        theta_ax.pcolormesh(
            time_edges,
            k_edges,
            theta_log,
            shading="auto",
            cmap=cmap,
            vmin=row_vmin,
            vmax=row_vmax,
        )

        if row == 1:
            hr_ax.set_title(r"$\log_{10}|\hat E_k^{HR}(t)|$")
            theta_ax.set_title(r"$\log_{10}|\hat E_k^\theta(t)|$")
        hr_ax.set_xlabel("t")
        theta_ax.set_xlabel("t")
        hr_ax.set_ylabel("k")
        theta_ax.set_ylabel("k")
        nonfinite_theta_t = _first_nonfinite_time(comp.times, comp.E_hat_theta)
        if nonfinite_theta_t is not None:
            theta_ax.axvline(nonfinite_theta_t, color="#475569", lw=1.1, ls="--", alpha=0.8)
            theta_ax.text(
                0.99,
                0.98,
                rf"nonfinite at $t \approx {nonfinite_theta_t:.2f}$",
                transform=theta_ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#334155",
                bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
            )
        row_label = rf"$N_v={int(case.Nv)}$" + _nv_status_suffix(
            in_training_targets=case.in_training_targets,
            beyond_training_range=case.beyond_training_range,
        )
        hr_ax.text(
            -0.42,
            0.5,
            row_label + "\n" + rf"$\varepsilon_E={_format_metric_value(case.epsilon_E)}$",
            transform=hr_ax.transAxes,
            ha="left",
            va="center",
            fontsize=9,
        )
        fig.colorbar(mesh_ref, cax=cax)
    return fig
