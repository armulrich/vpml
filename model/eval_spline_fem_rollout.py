"""Evaluate the current spline-grid online residual rollout sweep.

This evaluator intentionally uses the existing semi-Lagrangian coarse step plus
the learned signed ``dt * residual`` correction. It does not use the RK45
experiment.
For each low velocity grid, it compares the fixed HR teacher, the low-grid solver
without correction lifted back to the teacher grid, and the learned-correction
rollout lifted back to the teacher grid.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))

from vpml.jax_runtime import bootstrap_jax_runtime

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from vpml.metrics import (
    EarlyElectricFieldGrowthMetric,
    EarlyGrowthConfig,
    FieldErrorConfig,
    SelfGeneratedFieldErrorMetric,
)
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    _physical_grid_ops,
    compute_electric_field_from_distribution,
    electric_energy_from_field,
    run_semilagrangian_vlasov_poisson,
)
from vpml.rollout.spline_fem import (
    maxwellian_on_grid,
    restrict_state_to_grid,
    spline_fem_base_step,
    spline_fem_step_with_residual,
)
from vpml.visualization.common import save_figure

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _json_scalar(value: float) -> float | str:
    value = float(value)
    if np.isfinite(value):
        return value
    if value > 0.0:
        return "inf"
    if value < 0.0:
        return "-inf"
    return "nan"


def _spectral_with_bad(color: str = "#d1d5db"):
    cmap = plt.colormaps["Spectral_r"].copy()
    cmap.set_bad(color=color)
    return cmap


def _magma_with_bad(color: str = "#d1d5db"):
    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad(color=color)
    return cmap


def _phase_field_masked(field: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_invalid(np.asarray(field, dtype=np.float64))


def _safe_log10_masked(values: np.ndarray, floor: float) -> np.ma.MaskedArray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), np.maximum(arr, float(floor)), np.nan)
    return np.ma.masked_invalid(np.log10(arr))


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


def _finite_log_limits(arrays: Sequence[np.ndarray], floor: float) -> Tuple[float, float]:
    finite_chunks = []
    for arr in arrays:
        log_arr = np.asarray(_safe_log10_masked(arr, floor).filled(np.nan), dtype=np.float64)
        finite = log_arr[np.isfinite(log_arr)]
        if finite.size:
            finite_chunks.append(finite)
    if not finite_chunks:
        return math.log10(float(floor)), 0.0
    values = np.concatenate(finite_chunks)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if math.isclose(lo, hi):
        hi = lo + 1.0
    return lo, hi


def _checkpoint_for_vgrid(checkpoint_dir: Path, vgrid: int) -> Path:
    return checkpoint_dir / f"vgrid{int(vgrid)}" / "spline_fem_residual.npz"


def load_spline_residual_checkpoint(path: Path) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    with np.load(path) as data:
        res_blocks = int(np.asarray(data["res_blocks"]).reshape(-1)[0])
        params: Dict[str, object] = {
            "W0": jnp.asarray(data["W0"], dtype=jnp.float64),
            "b0": jnp.asarray(data["b0"], dtype=jnp.float64),
            "Wout": jnp.asarray(data["Wout"], dtype=jnp.float64),
            "bout": jnp.asarray(data["bout"], dtype=jnp.float64),
        }
        blocks = []
        for i in range(res_blocks):
            blocks.append(
                {
                    "W1": jnp.asarray(data[f"block{i}_W1"], dtype=jnp.float64),
                    "b1": jnp.asarray(data[f"block{i}_b1"], dtype=jnp.float64),
                    "W2": jnp.asarray(data[f"block{i}_W2"], dtype=jnp.float64),
                    "b2": jnp.asarray(data[f"block{i}_b2"], dtype=jnp.float64),
                }
            )
        params["blocks"] = tuple(blocks)
        metadata = {key: np.asarray(data[key]) for key in data.files if key not in {
            "W0",
            "b0",
            "Wout",
            "bout",
            *(f"block{i}_{name}" for i in range(res_blocks) for name in ("W1", "b1", "W2", "b2")),
        }}
    return params, metadata


def _ehat_from_state(f_state: jnp.ndarray, config: PhysicalGridVlasovPoissonConfig) -> jnp.ndarray:
    e_phys = compute_electric_field_from_distribution(f_state, config)
    return (jnp.fft.rfft(e_phys) / float(config.Nx)).astype(jnp.complex128)


def _make_nonlinear_initial_state(
    config: PhysicalGridVlasovPoissonConfig,
    *,
    eps: float,
    k0: float,
) -> jnp.ndarray:
    equilibrium = maxwellian_on_grid(config.v)
    perturb = 1.0 + float(eps) * jnp.cos(float(k0) * config.x)
    return (equilibrium[:, None] * perturb[None, :]).astype(jnp.float64)


def run_teacher_reference(
    config: PhysicalGridVlasovPoissonConfig,
    initial_state: jnp.ndarray,
) -> Dict[str, np.ndarray]:
    raw = run_semilagrangian_vlasov_poisson(
        config,
        initial_state,
        history_stride=1,
        return_state_history=True,
        history_projector=lambda state: _ehat_from_state(state, config),
    )
    return {
        "times": np.asarray(raw["times"], dtype=np.float64),
        "energy": np.asarray(raw["energy"], dtype=np.float64),
        "E_hat_hist": np.asarray(raw["state_history"], dtype=np.complex128),
        "k_arr": np.asarray(raw["k_arr"], dtype=np.float64),
        "snapshot_times": np.asarray(raw["snapshot_times"], dtype=np.float64),
        "snapshot_f": np.asarray(raw["snapshot_f"], dtype=np.float64),
        "x": np.asarray(raw["x"], dtype=np.float64),
        "v": np.asarray(raw["v"], dtype=np.float64),
    }


def run_low_spline_rollout(
    config: PhysicalGridVlasovPoissonConfig,
    initial_state: jnp.ndarray,
    *,
    snapshot_times: Sequence[float],
    params: Optional[Dict[str, object]] = None,
) -> Dict[str, np.ndarray]:
    ops = _physical_grid_ops(config)
    nsteps = int(config.nsteps)
    snap_steps = np.asarray(
        [int(round(float(t) / float(config.dt))) for t in snapshot_times],
        dtype=np.int32,
    )
    if np.any(snap_steps < 0) or np.any(snap_steps > nsteps):
        raise ValueError("snapshot_times must lie inside the rollout interval")

    initial_state = jnp.asarray(initial_state, dtype=jnp.float64)
    snaps0 = jnp.zeros((len(snap_steps), int(config.Nv), int(config.Nx)), dtype=jnp.float64)
    if 0 in snap_steps:
        zero_idx = int(np.where(snap_steps == 0)[0][0])
        snaps0 = snaps0.at[zero_idx].set(initial_state)

    e0 = compute_electric_field_from_distribution(initial_state, config, ops=ops)
    energy0 = electric_energy_from_field(e0, config)
    ehat0 = (jnp.fft.rfft(e0) / float(config.Nx)).astype(jnp.complex128)

    def maybe_store_snapshot(snaps: jnp.ndarray, step_i: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        for j, snap_step in enumerate(snap_steps):
            snaps = jax.lax.cond(
                step_i == int(snap_step),
                lambda arr, idx=j, value=state: arr.at[idx].set(value),
                lambda arr: arr,
                snaps,
            )
        return snaps

    def step(carry: Tuple[jnp.ndarray, jnp.ndarray], step_i: jnp.ndarray):
        state, snaps = carry
        if params is None:
            next_state = spline_fem_base_step(state, config, ops=ops)
        else:
            next_state = spline_fem_step_with_residual(state, params, config, ops=ops)
        e_phys = compute_electric_field_from_distribution(next_state, config, ops=ops)
        energy = electric_energy_from_field(e_phys, config)
        ehat = (jnp.fft.rfft(e_phys) / float(config.Nx)).astype(jnp.complex128)
        snaps = maybe_store_snapshot(snaps, step_i, next_state)
        return (next_state, snaps), (energy, ehat)

    (_, snaps_out), (energy_tail, ehat_tail) = jax.lax.scan(
        step,
        (initial_state, snaps0),
        jnp.arange(1, nsteps + 1, dtype=jnp.int32),
    )
    return {
        "times": np.linspace(0.0, nsteps * float(config.dt), nsteps + 1),
        "energy": np.asarray(jnp.concatenate([jnp.array([energy0]), energy_tail], axis=0), dtype=np.float64),
        "E_hat_hist": np.asarray(jnp.concatenate([ehat0[None, :], ehat_tail], axis=0), dtype=np.complex128),
        "k_arr": np.asarray(ops["k_arr"], dtype=np.float64),
        "snapshot_times": np.asarray(tuple(float(t) for t in snapshot_times), dtype=np.float64),
        "snapshot_f": np.asarray(snaps_out, dtype=np.float64),
        "x": np.asarray(ops["x"], dtype=np.float64),
        "v": np.asarray(ops["v"], dtype=np.float64),
    }


def evaluate_metric_pair(
    payload: Dict[str, np.ndarray],
    teacher: Dict[str, np.ndarray],
    *,
    growth_metric: EarlyElectricFieldGrowthMetric,
    field_metric: SelfGeneratedFieldErrorMetric,
) -> Dict[str, object]:
    growth = growth_metric.compare(
        payload["times"],
        payload["energy"],
        teacher["times"],
        teacher["energy"],
    )
    field = field_metric.evaluate_fourier(
        payload["times"],
        payload["E_hat_hist"],
        payload["k_arr"],
        teacher["times"],
        teacher["E_hat_hist"],
        teacher["k_arr"],
    )
    return {
        "epsilon_grow": float(growth.epsilon_grow),
        "gamma_grow": float(growth.gamma_grow_theta),
        "gamma_grow_hr": float(growth.gamma_grow_hr),
        "fit_t_a": float(growth.t_a),
        "fit_t_b": float(growth.t_b),
        "epsilon_E": float(field.epsilon_E),
        "field_T": float(field.T),
        "num_common_modes": int(field.num_modes),
    }


def plot_metric1(
    teacher: Dict[str, np.ndarray],
    cases: Sequence[Dict[str, object]],
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.semilogy(teacher["times"], np.maximum(teacher["energy"], 1.0e-30), color="black", lw=2.0, label="HR teacher")
    cmap = plt.get_cmap("viridis", max(len(cases), 1))
    for idx, case in enumerate(cases):
        color = cmap(idx)
        vgrid = int(case["vgrid"])
        baseline = case["baseline"]
        learned = case["learned"]
        ax.semilogy(
            baseline["times"],
            np.maximum(baseline["energy"], 1.0e-30),
            color=color,
            ls="--",
            alpha=0.75,
            label=f"Mv={vgrid} no correction",
        )
        ax.semilogy(
            learned["times"],
            np.maximum(learned["energy"], 1.0e-30),
            color=color,
            lw=1.8,
            label=f"Mv={vgrid} learned",
        )
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathcal{E}_E(t)$")
    ax.set_title("Spline-grid Metric 1: teacher, no correction, learned correction")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return save_figure(fig, output_path, dpi=220)


def plot_metric2(cases: Sequence[Dict[str, object]], output_path: Path, *, log_floor: float = 1.0e-14) -> Path:
    """Plot Metric 2 with the same spectral context as the Hermite sweeps.

    Unlike the Hermite Nv-sweep, this spline-grid evaluation has two low-grid
    rollouts per row: the raw coarse solver and the learned-correction solver.
    The plot therefore shows HR, no-correction, and learned spectra side by side
    instead of only HR/theta.
    """

    if not cases:
        raise ValueError("cases must be nonempty")

    fig = plt.figure(figsize=(14.2, 2.9 * len(cases) + 3.2), constrained_layout=True)
    grid = fig.add_gridspec(
        len(cases) + 1,
        4,
        height_ratios=[1.1] + [1.0] * len(cases),
        width_ratios=[1.0, 1.0, 1.0, 0.045],
    )

    ax_top = fig.add_subplot(grid[0, :3])
    m_vals = np.asarray([int(case["vgrid"]) for case in cases], dtype=np.float64)
    baseline_vals = np.asarray([float(case["baseline_metrics"]["epsilon_E"]) for case in cases], dtype=np.float64)
    learned_vals = np.asarray([float(case["learned_metrics"]["epsilon_E"]) for case in cases], dtype=np.float64)
    ax_top.plot(m_vals, baseline_vals, marker="o", color="#64748b", lw=1.9, label="no correction")
    ax_top.plot(m_vals, learned_vals, marker="o", color="#b91c1c", lw=1.9, label="learned correction")
    ax_top.set_xscale("log", base=2)
    ax_top.set_yscale("log")
    ax_top.set_xticks(m_vals, [str(int(v)) for v in m_vals])
    ax_top.set_xlabel(r"low velocity grid $M_v$")
    ax_top.set_ylabel(r"$\varepsilon_E(T)$")
    ax_top.set_title("Spline-grid Metric 2: self-generated field error")
    ax_top.grid(True, which="both", alpha=0.3)
    ax_top.legend(frameon=False, fontsize=9)

    for row, case in enumerate(cases, start=1):
        baseline_comp = case["baseline_field_comparison"]
        learned_comp = case["learned_field_comparison"]
        time_edges = _centers_to_edges(baseline_comp.times)
        k_edges = _centers_to_edges(baseline_comp.selected_k)
        cmap = _magma_with_bad()
        row_vmin, row_vmax = _finite_log_limits([np.abs(baseline_comp.E_hat_hr)], log_floor)

        panels = [
            (_safe_log10_masked(np.abs(baseline_comp.E_hat_hr), log_floor).T, r"$\log_{10}|\hat E_k^{HR}(t)|$"),
            (_safe_log10_masked(np.abs(baseline_comp.E_hat_theta), log_floor).T, r"$\log_{10}|\hat E_k(t)|$ no correction"),
            (_safe_log10_masked(np.abs(learned_comp.E_hat_theta), log_floor).T, r"$\log_{10}|\hat E_k(t)|$ learned"),
        ]
        mesh_ref = None
        for col, (data, title) in enumerate(panels):
            ax = fig.add_subplot(grid[row, col])
            mesh_ref = ax.pcolormesh(
                time_edges,
                k_edges,
                data,
                shading="auto",
                cmap=cmap,
                vmin=row_vmin,
                vmax=row_vmax,
            )
            if row == 1:
                ax.set_title(title, fontsize=10)
            ax.set_xlabel("t")
            ax.set_ylabel("k")
            if col == 0:
                ax.text(
                    -0.42,
                    0.5,
                    (
                        rf"$M_v={int(case['vgrid'])}$"
                        "\n"
                        rf"no corr. $\varepsilon_E={baseline_vals[row - 1]:.3e}$"
                        "\n"
                        rf"learned $\varepsilon_E={learned_vals[row - 1]:.3e}$"
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=8.5,
                )
        cax = fig.add_subplot(grid[row, 3])
        if mesh_ref is not None:
            fig.colorbar(mesh_ref, cax=cax)

    return save_figure(fig, output_path, dpi=220)


def plot_fig10_teacher_baseline_learned(
    cases: Sequence[Dict[str, object]],
    *,
    times: Sequence[float],
    output_path: Path,
    phase_vmin: Optional[float],
    phase_vmax: Optional[float],
    plot_vmin: float,
    plot_vmax: float,
) -> Path:
    if len(times) != 2:
        raise ValueError("Fig. 10 spline comparison expects exactly two snapshot times")
    nrows = len(cases)
    ncols = 6
    fig = plt.figure(figsize=(18.5, 2.6 * nrows + 1.0), constrained_layout=True)
    grid = fig.add_gridspec(nrows, ncols, wspace=0.08, hspace=0.18)
    headers = (
        f"High Resolution (HR)\nt={times[0]:g}",
        f"High Resolution (HR)\nt={times[1]:g}",
        f"Low Resolution (LR)\nt={times[0]:g}",
        f"Low Resolution (LR)\nt={times[1]:g}",
        f"Low Resolution +\nLearned Correction\nt={times[0]:g}",
        f"Low Resolution +\nLearned Correction\nt={times[1]:g}",
    )
    cmap = _spectral_with_bad()
    xticks = [0.0, 2.0 * math.pi, 4.0 * math.pi]
    xticklabels = ["0", r"$2\pi$", r"$4\pi$"]
    mesh_ref = None
    all_axes = []

    for row, case in enumerate(cases):
        teacher_snaps = np.asarray(case["teacher"]["snapshot_f"], dtype=np.float64)
        baseline_snaps = np.asarray(case["baseline_lifted"]["snapshot_f"], dtype=np.float64)
        learned_snaps = np.asarray(case["learned_lifted"]["snapshot_f"], dtype=np.float64)
        panels = (
            teacher_snaps[0],
            teacher_snaps[1],
            baseline_snaps[0],
            baseline_snaps[1],
            learned_snaps[0],
            learned_snaps[1],
        )
        row_vmin = float(phase_vmin) if phase_vmin is not None else 0.0
        row_vmax = float(phase_vmax) if phase_vmax is not None else 0.5
        if math.isclose(row_vmin, row_vmax):
            row_vmax = row_vmin + 1.0e-12
        x = np.asarray(case["teacher"]["x"], dtype=np.float64)
        v = np.asarray(case["teacher"]["v"], dtype=np.float64)
        for col, panel in enumerate(panels):
            ax = fig.add_subplot(grid[row, col])
            all_axes.append(ax)
            field_raw = np.asarray(panel, dtype=np.float64)
            field = _phase_field_masked(field_raw)
            mesh_ref = ax.pcolormesh(
                x,
                v,
                field,
                shading="auto",
                vmin=row_vmin,
                vmax=row_vmax,
                cmap=cmap,
                rasterized=True,
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
                ax.set_title(headers[col], fontsize=10.5, pad=6)
            ax.set_xlabel("x")
            ax.set_xticks(xticks, xticklabels)
            ax.set_yticks([-4, -2, 0, 2, 4])
            if col == 0:
                ax.set_ylabel("v")
                ax.text(
                    -0.34,
                    0.5,
                    rf"$M_v={int(case['vgrid'])}$",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10.5,
                )
            else:
                ax.set_ylabel("")
            ax.set_ylim(float(plot_vmin), float(plot_vmax))

    if mesh_ref is not None:
        fig.colorbar(mesh_ref, ax=all_axes, fraction=0.018, pad=0.01)

    fig.suptitle("Spline-grid Fig. 10: fixed HR teacher, lifted low grid, lifted learned correction", fontsize=12)
    return save_figure(fig, output_path, dpi=220)


def snapshot_payload(
    snapshots: np.ndarray,
    config: PhysicalGridVlasovPoissonConfig,
    snapshot_times: Sequence[float],
) -> Dict[str, np.ndarray]:
    ops = _physical_grid_ops(config)
    return {
        "snapshot_times": np.asarray(tuple(float(t) for t in snapshot_times), dtype=np.float64),
        "snapshot_f": np.asarray(snapshots, dtype=np.float64),
        "x": np.asarray(ops["x"], dtype=np.float64),
        "v": np.asarray(ops["v"], dtype=np.float64),
    }


def lifted_snapshot_payload(
    low_snapshots: np.ndarray,
    low_config: PhysicalGridVlasovPoissonConfig,
    teacher_config: PhysicalGridVlasovPoissonConfig,
    snapshot_times: Sequence[float],
) -> Dict[str, np.ndarray]:
    low_ops = _physical_grid_ops(low_config)
    lifted = jax.vmap(
        lambda state: restrict_state_to_grid(
            state,
            low_config,
            teacher_config,
            src_ops=low_ops,
        )
    )(jnp.asarray(low_snapshots, dtype=jnp.float64))
    teacher_ops = _physical_grid_ops(teacher_config)
    return {
        "snapshot_times": np.asarray(tuple(float(t) for t in snapshot_times), dtype=np.float64),
        "snapshot_f": np.asarray(lifted, dtype=np.float64),
        "x": np.asarray(teacher_ops["x"], dtype=np.float64),
        "v": np.asarray(teacher_ops["v"], dtype=np.float64),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    vgrid_list = parse_int_tuple(args.vgrid_list)
    snapshot_times = parse_float_tuple(args.snapshot_times)
    if len(snapshot_times) != 2:
        raise ValueError("--snapshot-times must contain exactly two values for the Fig. 10 layout")

    teacher_config = PhysicalGridVlasovPoissonConfig(
        Nx=int(args.teacher_Nx),
        Nv=int(args.teacher_Nv),
        Lx=float(args.teacher_L),
        vmin=float(args.teacher_vmin),
        vmax=float(args.teacher_vmax),
        dt=float(args.dt),
        T=float(args.T),
        poisson_sign=float(args.poisson_sign),
        snapshot_times=snapshot_times,
    )
    initial_teacher = _make_nonlinear_initial_state(
        teacher_config,
        eps=float(args.eps),
        k0=float(args.k0),
    )
    print(
        "[spline-fem-eval] running fixed HR teacher "
        f"Nx={teacher_config.Nx}, Nv={teacher_config.Nv}, T={teacher_config.T}"
    )
    teacher = run_teacher_reference(teacher_config, initial_teacher)

    growth_window = None
    if args.growth_time_window:
        growth_window = tuple(float(v) for v in parse_float_tuple(args.growth_time_window))
        if len(growth_window) != 2:
            raise ValueError("--growth-time-window must be 't0,t1'")
    growth_metric = EarlyElectricFieldGrowthMetric(
        EarlyGrowthConfig(time_window=growth_window)
    )
    field_metric = SelfGeneratedFieldErrorMetric(
        FieldErrorConfig(
            final_time=float(args.T),
            num_low_modes=None if args.field_num_low_modes is None else int(args.field_num_low_modes),
            k_max=None if args.field_k_max is None else float(args.field_k_max),
        )
    )

    cases: List[Dict[str, object]] = []
    for vgrid in vgrid_list:
        checkpoint = _checkpoint_for_vgrid(Path(args.checkpoint_dir), int(vgrid))
        if not checkpoint.exists():
            if bool(args.skip_missing):
                print(f"[spline-fem-eval] skipping missing checkpoint {checkpoint}")
                continue
            raise FileNotFoundError(checkpoint)
        params, checkpoint_meta = load_spline_residual_checkpoint(checkpoint)
        low_config = PhysicalGridVlasovPoissonConfig(
            Nx=int(args.low_Nx),
            Nv=int(vgrid),
            Lx=float(args.teacher_L),
            vmin=float(args.teacher_vmin),
            vmax=float(args.teacher_vmax),
            dt=float(args.dt),
            T=float(args.T),
            poisson_sign=float(args.poisson_sign),
            snapshot_times=snapshot_times,
        )
        initial_low = restrict_state_to_grid(initial_teacher, teacher_config, low_config)
        teacher_row = snapshot_payload(
            teacher["snapshot_f"],
            teacher_config,
            snapshot_times,
        )
        print(f"[spline-fem-eval] rolling low grid Mv={int(vgrid)} without correction")
        baseline = run_low_spline_rollout(
            low_config,
            initial_low,
            snapshot_times=snapshot_times,
            params=None,
        )
        print(f"[spline-fem-eval] rolling low grid Mv={int(vgrid)} with learned correction")
        learned = run_low_spline_rollout(
            low_config,
            initial_low,
            snapshot_times=snapshot_times,
            params=params,
        )
        baseline_lifted = lifted_snapshot_payload(
            baseline["snapshot_f"],
            low_config,
            teacher_config,
            snapshot_times,
        )
        learned_lifted = lifted_snapshot_payload(
            learned["snapshot_f"],
            low_config,
            teacher_config,
            snapshot_times,
        )
        baseline_metrics = evaluate_metric_pair(
            baseline,
            teacher,
            growth_metric=growth_metric,
            field_metric=field_metric,
        )
        learned_metrics = evaluate_metric_pair(
            learned,
            teacher,
            growth_metric=growth_metric,
            field_metric=field_metric,
        )
        baseline_field_comparison = field_metric.prepare_fourier_comparison(
            baseline["times"],
            baseline["E_hat_hist"],
            baseline["k_arr"],
            teacher["times"],
            teacher["E_hat_hist"],
            teacher["k_arr"],
        )
        learned_field_comparison = field_metric.prepare_fourier_comparison(
            learned["times"],
            learned["E_hat_hist"],
            learned["k_arr"],
            teacher["times"],
            teacher["E_hat_hist"],
            teacher["k_arr"],
        )
        case_path = outdir / f"spline_fem_eval_vgrid{int(vgrid)}.npz"
        np.savez(
            case_path,
            teacher_times=teacher["times"],
            teacher_energy=teacher["energy"],
            teacher_E_hat_hist=teacher["E_hat_hist"],
            teacher_k_arr=teacher["k_arr"],
            teacher_snapshot_f=teacher_row["snapshot_f"],
            baseline_times=baseline["times"],
            baseline_energy=baseline["energy"],
            baseline_E_hat_hist=baseline["E_hat_hist"],
            baseline_k_arr=baseline["k_arr"],
            baseline_snapshot_f=baseline["snapshot_f"],
            baseline_lifted_snapshot_f=baseline_lifted["snapshot_f"],
            learned_times=learned["times"],
            learned_energy=learned["energy"],
            learned_E_hat_hist=learned["E_hat_hist"],
            learned_k_arr=learned["k_arr"],
            learned_snapshot_f=learned["snapshot_f"],
            learned_lifted_snapshot_f=learned_lifted["snapshot_f"],
            x=teacher_row["x"],
            v=teacher_row["v"],
            snapshot_times=np.asarray(snapshot_times, dtype=np.float64),
            baseline_epsilon_grow=np.array([baseline_metrics["epsilon_grow"]], dtype=np.float64),
            baseline_epsilon_E=np.array([baseline_metrics["epsilon_E"]], dtype=np.float64),
            learned_epsilon_grow=np.array([learned_metrics["epsilon_grow"]], dtype=np.float64),
            learned_epsilon_E=np.array([learned_metrics["epsilon_E"]], dtype=np.float64),
        )
        cases.append(
            {
                "vgrid": int(vgrid),
                "checkpoint": str(checkpoint),
                "case_npz": str(case_path),
                "checkpoint_meta": checkpoint_meta,
                "teacher": teacher_row,
                "baseline": baseline,
                "baseline_lifted": baseline_lifted,
                "learned": learned,
                "learned_lifted": learned_lifted,
                "baseline_metrics": baseline_metrics,
                "learned_metrics": learned_metrics,
                "baseline_field_comparison": baseline_field_comparison,
                "learned_field_comparison": learned_field_comparison,
            }
        )
        print(
            f"[spline-fem-eval] Mv={int(vgrid)} "
            f"Metric1 no-corr={baseline_metrics['epsilon_grow']:.4e} "
            f"learned={learned_metrics['epsilon_grow']:.4e}; "
            f"Metric2 no-corr={baseline_metrics['epsilon_E']:.4e} "
            f"learned={learned_metrics['epsilon_E']:.4e}"
        )

    if not cases:
        raise ValueError("No spline FEM evaluation cases were available")

    metric1_png = plot_metric1(
        teacher,
        cases,
        outdir / "spline_fem_metric1_teacher_baseline_learned.png",
    )
    metric2_png = plot_metric2(
        cases,
        outdir / "spline_fem_metric2_baseline_vs_learned.png",
    )
    fig10_png = plot_fig10_teacher_baseline_learned(
        cases,
        times=snapshot_times,
        output_path=outdir / "spline_fem_fig10_teacher_baseline_learned.png",
        phase_vmin=args.phase_vmin,
        phase_vmax=args.phase_vmax,
        plot_vmin=float(args.plot_vmin),
        plot_vmax=float(args.plot_vmax),
    )
    summary = {
        "outdir": str(outdir),
        "vgrid_list": [int(case["vgrid"]) for case in cases],
        "method": "semi_lagrangian_plus_dt_residual",
        "teacher": {
            "Nx": int(args.teacher_Nx),
            "Nv": int(args.teacher_Nv),
            "dt": float(args.dt),
            "T": float(args.T),
            "eps": float(args.eps),
            "k0": float(args.k0),
            "snapshot_times": list(float(t) for t in snapshot_times),
        },
        "low_grid": {
            "Nx": int(args.low_Nx),
            "vgrid_list": [int(case["vgrid"]) for case in cases],
        },
        "cases": [
            {
                "vgrid": int(case["vgrid"]),
                "checkpoint": str(case["checkpoint"]),
                "case_npz": str(case["case_npz"]),
                "baseline": {
                    key: _json_scalar(value) if isinstance(value, float) else value
                    for key, value in case["baseline_metrics"].items()
                },
                "learned": {
                    key: _json_scalar(value) if isinstance(value, float) else value
                    for key, value in case["learned_metrics"].items()
                },
            }
            for case in cases
        ],
        "artifacts": {
            "metric1_png": str(metric1_png),
            "metric2_png": str(metric2_png),
            "fig10_png": str(fig10_png),
        },
    }
    summary_path = outdir / "spline_fem_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[spline-fem-eval] saved Metric 1 plot to {metric1_png}")
    print(f"[spline-fem-eval] saved Metric 2 plot to {metric2_png}")
    print(f"[spline-fem-eval] saved Fig. 10 comparison to {fig10_png}")
    print(f"[spline-fem-eval] saved summary to {summary_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate spline/FEM online residual rollouts against a fixed HR teacher"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--vgrid-list", type=str, default="32,64,128,256")
    parser.add_argument("--low-Nx", type=int, default=200)
    parser.add_argument("--teacher-Nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", type=int, default=512)
    parser.add_argument("--teacher-L", type=float, default=4.0 * math.pi)
    parser.add_argument("--teacher-vmin", type=float, default=-8.0)
    parser.add_argument("--teacher-vmax", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--T", type=float, default=40.0)
    parser.add_argument("--poisson-sign", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--k0", type=float, default=0.5)
    parser.add_argument("--snapshot-times", type=str, default="20.0,40.0")
    parser.add_argument("--growth-time-window", type=str, default="")
    parser.add_argument("--field-k-max", type=float, default=None)
    parser.add_argument("--field-num-low-modes", type=int, default=None)
    parser.add_argument("--phase-vmin", type=float, default=0.0)
    parser.add_argument("--phase-vmax", type=float, default=0.5)
    parser.add_argument("--plot-vmin", type=float, default=-4.0)
    parser.add_argument("--plot-vmax", type=float, default=4.0)
    parser.add_argument("--skip-missing", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
