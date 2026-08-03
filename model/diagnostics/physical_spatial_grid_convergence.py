"""Semi-Lagrangian spatial-grid convergence diagnostic for Landau teachers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from vpml.jax_runtime import bootstrap_jax_runtime, print_jax_runtime_summary

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))

from model.diagnostics.physical_velocity_grid_convergence import (
    _case_perturbations,
    _display_case,
    _energy_block_changes,
    _parse_float_tuple,
    _parse_int_tuple,
)
from model.diagnostics.projection_quadrature_convergence import (
    _load_teacher_snapshot_artifact,
)
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    gaussian_pdf,
    normalize_density_on_grid,
    run_semilagrangian_vlasov_poisson,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


def _distribution_successive_x_change(
    coarse_snapshots: np.ndarray,
    refined_snapshots: np.ndarray,
    *,
    equilibrium: np.ndarray,
    row_chunk: int = 256,
) -> Tuple[float, np.ndarray]:
    """Compare periodic phase-space snapshots after spectral x resampling."""
    coarse = np.asarray(coarse_snapshots, dtype=np.float64)
    refined = np.asarray(refined_snapshots, dtype=np.float64)
    equilibrium = np.asarray(equilibrium, dtype=np.float64)
    if coarse.ndim != 3 or refined.ndim != 3:
        raise ValueError("snapshots must have shape (time, Nv, Nx)")
    if coarse.shape[:2] != refined.shape[:2]:
        raise ValueError("coarse and refined snapshots must share time and Nv")
    if equilibrium.shape != (coarse.shape[1],):
        raise ValueError("equilibrium must have shape (Nv,)")

    coarse_delta = coarse - equilibrium[None, :, None]
    refined_delta = refined - equilibrium[None, :, None]
    target_nx = int(refined.shape[-1])
    difference_sq = np.zeros((coarse.shape[0],), dtype=np.float64)
    reference_sq = np.zeros((coarse.shape[0],), dtype=np.float64)

    for start in range(0, int(coarse.shape[1]), int(row_chunk)):
        stop = min(start + int(row_chunk), int(coarse.shape[1]))
        coarse_on_refined = resample(
            coarse_delta[:, start:stop, :],
            target_nx,
            axis=-1,
        )
        difference = coarse_on_refined - refined_delta[:, start:stop, :]
        difference_sq += np.sum(difference * difference, axis=(1, 2))
        reference_block = refined_delta[:, start:stop, :]
        reference_sq += np.sum(reference_block * reference_block, axis=(1, 2))

    difference_norms = np.sqrt(difference_sq)
    reference_norms = np.sqrt(reference_sq)
    tiny = np.finfo(np.float64).tiny
    max_normalized_change = float(np.max(difference_norms)) / max(
        float(np.max(reference_norms)),
        tiny,
    )
    return max_normalized_change, difference_norms / np.maximum(
        reference_norms,
        tiny,
    )


def _save_energy_plot(
    *,
    energy_by_case: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    figure_path: Path,
) -> None:
    case_names = tuple(energy_by_case)
    spatial_grids = sorted(
        {int(grid) for by_grid in energy_by_case.values() for grid in by_grid}
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(spatial_grids)))
    fig, axes = plt.subplots(
        len(case_names),
        1,
        figsize=(9.5, 2.65 * len(case_names)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for axis, case_name in zip(axes, case_names):
        label, _ = _display_case(case_name)
        for color, spatial_nx in zip(colors, spatial_grids):
            times, energy = energy_by_case[case_name][spatial_nx]
            axis.semilogy(
                times,
                np.maximum(energy, np.finfo(np.float64).tiny),
                color=color,
                linewidth=1.35,
                label=rf"$N_x={spatial_nx:,}$",
            )
        axis.set_title(label, loc="left", fontsize=10)
        axis.set_ylabel(r"$\mathcal{E}(t)$")
        axis.grid(True, which="both", alpha=0.22)
    axes[-1].set_xlabel(r"Time $t$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(spatial_grids),
        frameon=False,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.91, hspace=0.34)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_plot(
    *,
    summary_by_case: Dict[str, Dict[str, Dict[str, object]]],
    figure_path: Path,
) -> None:
    refined_grids = tuple(
        sorted(
            {
                int(grid)
                for by_refined_grid in summary_by_case.values()
                for grid in by_refined_grid
            }
        )
    )
    x_positions = np.arange(len(refined_grids), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.35), sharey=True)
    metric_keys = (
        "global_energy_refinement_change",
        "max_snapshot_distribution_refinement_change",
    )
    titles = (
        r"Electric-field energy trajectory $\mathcal{E}(t)$",
        r"Phase-space perturbation $f-f_{\rm eq}$",
    )
    for case_name, by_refined_grid in summary_by_case.items():
        label, color = _display_case(case_name)
        for axis, metric_key in zip(axes, metric_keys):
            values = [
                float(by_refined_grid[str(grid)][metric_key])
                for grid in refined_grids
            ]
            axis.semilogy(
                x_positions,
                np.maximum(np.asarray(values), 1e-16),
                marker="o",
                color=color,
                linewidth=1.6,
                label=label,
            )
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel(r"Refined solver spatial-grid points $N_x$")
        axis.set_xticks(x_positions)
        axis.set_xticklabels([f"{grid:,}" for grid in refined_grids])
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Relative successive-grid change")
    axes[1].tick_params(axis="y", labelleft=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.79, wspace=0.20)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refine only the semi-Lagrangian solver's periodic spatial grid "
            "for otherwise identical Landau teachers."
        )
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--spatial-Nx-list", type=str, default="128,256,512")
    parser.add_argument("--target-Nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", type=int, default=8192)
    parser.add_argument(
        "--reused-teacher-snapshots",
        type=Path,
        default=None,
        help="Optional snapshot artifact providing one Nx refinement level.",
    )
    parser.add_argument("--teacher-L", type=float, default=4.0 * math.pi)
    parser.add_argument("--teacher-vmin", type=float, default=-8.0)
    parser.add_argument("--teacher-vmax", type=float, default=8.0)
    parser.add_argument("--teacher-dt", type=float, default=0.01)
    parser.add_argument("--T-final", type=float, default=120.0)
    parser.add_argument(
        "--snapshot-times",
        type=str,
        default="0,20,40,60,80,100,120",
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--linear-eps", type=float, default=0.01)
    parser.add_argument("--linear-modes", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--linear-seed", type=int, default=0)
    parser.add_argument("--weak-eps", type=float, default=0.1)
    parser.add_argument("--strong-eps", type=float, default=0.5)
    parser.add_argument("--nonlinear-k0", type=float, default=0.5)
    parser.add_argument("--poisson-sign", type=float, default=1.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    print_jax_runtime_summary(jax, context="semi-Lagrangian spatial-grid diagnostic")
    args = _build_arg_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostic directory: {outdir}"
        )
    outdir.mkdir(parents=True)

    spatial_grids = tuple(sorted(set(_parse_int_tuple(args.spatial_Nx_list))))
    snapshot_times = _parse_float_tuple(args.snapshot_times)
    linear_modes = _parse_float_tuple(args.linear_modes)
    target_nx = int(args.target_Nx)
    if len(spatial_grids) < 2:
        raise ValueError("--spatial-Nx-list must contain at least two grids")
    if target_nx not in spatial_grids or not any(nx > target_nx for nx in spatial_grids):
        raise ValueError("--target-Nx must have a finer comparison grid")
    if float(args.relative_tolerance) <= 0.0:
        raise ValueError("--relative-tolerance must be positive")

    reused_config = None
    reused_snapshots: Dict[str, np.ndarray] = {}
    reused_energy: Dict[str, np.ndarray] = {}
    reused_path = None
    if args.reused_teacher_snapshots is not None:
        reused_path = args.reused_teacher_snapshots.resolve()
        reused_config, reused_snapshots, reused_energy = (
            _load_teacher_snapshot_artifact(reused_path)
        )
        expected = {
            "Nv": int(args.teacher_Nv),
            "Lx": float(args.teacher_L),
            "vmin": float(args.teacher_vmin),
            "vmax": float(args.teacher_vmax),
            "dt": float(args.teacher_dt),
            "T": float(args.T_final),
            "poisson_sign": float(args.poisson_sign),
            "snapshot_times": tuple(snapshot_times),
        }
        actual = {
            "Nv": int(reused_config.Nv),
            "Lx": float(reused_config.Lx),
            "vmin": float(reused_config.vmin),
            "vmax": float(reused_config.vmax),
            "dt": float(reused_config.dt),
            "T": float(reused_config.T),
            "poisson_sign": float(reused_config.poisson_sign),
            "snapshot_times": tuple(reused_config.snapshot_times),
        }
        if actual != expected:
            raise ValueError(
                "Reused teacher configuration does not match the requested "
                f"diagnostic: actual={actual}, expected={expected}"
            )
        if int(reused_config.Nx) not in spatial_grids:
            raise ValueError("reused teacher Nx must appear in --spatial-Nx-list")
        print(
            "[diagnostic] reusing physical teacher snapshots from "
            f"{reused_path} (Nx={reused_config.Nx}, Nv={reused_config.Nv})"
        )

    baseline_config = PhysicalGridVlasovPoissonConfig(
        Nx=spatial_grids[0],
        Nv=int(args.teacher_Nv),
        Lx=float(args.teacher_L),
        vmin=float(args.teacher_vmin),
        vmax=float(args.teacher_vmax),
        dt=float(args.teacher_dt),
        T=float(args.T_final),
        poisson_sign=float(args.poisson_sign),
        snapshot_times=tuple(snapshot_times),
    )
    case_names = tuple(
        _case_perturbations(
            baseline_config,
            linear_eps=float(args.linear_eps),
            linear_modes=linear_modes,
            linear_seed=int(args.linear_seed),
            weak_eps=float(args.weak_eps),
            strong_eps=float(args.strong_eps),
            nonlinear_k0=float(args.nonlinear_k0),
        )
    )
    if reused_snapshots and set(reused_snapshots) != set(case_names):
        raise ValueError("reused artifact contains different representative cases")

    equilibrium = np.asarray(
        normalize_density_on_grid(
            gaussian_pdf(baseline_config.v, mean=0.0, sigma=1.0),
            baseline_config.v,
        ),
        dtype=np.float64,
    )
    block_edges = tuple(
        sorted(
            {
                0.0,
                min(60.0, float(args.T_final)),
                min(80.0, float(args.T_final)),
                min(100.0, float(args.T_final)),
                float(args.T_final),
            }
        )
    )
    records = []
    summary_by_case: Dict[str, Dict[str, Dict[str, object]]] = {}
    energy_by_case: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

    print(
        "[diagnostic] spatial Nx="
        + ",".join(str(value) for value in spatial_grids)
        + f" fixed Nv={int(args.teacher_Nv)} T={float(args.T_final):g}"
    )
    for case_name in case_names:
        snapshots_by_grid: Dict[int, np.ndarray] = {}
        energy_by_case[case_name] = {}
        if reused_config is not None:
            reused_nx = int(reused_config.Nx)
            snapshots_by_grid[reused_nx] = reused_snapshots[case_name]
            energy_by_case[case_name][reused_nx] = (
                reused_energy[f"{case_name}_times"],
                reused_energy[f"{case_name}_energy"],
            )

        for spatial_nx in spatial_grids:
            if reused_config is not None and spatial_nx == int(reused_config.Nx):
                continue
            config = PhysicalGridVlasovPoissonConfig(
                Nx=int(spatial_nx),
                Nv=int(args.teacher_Nv),
                Lx=float(args.teacher_L),
                vmin=float(args.teacher_vmin),
                vmax=float(args.teacher_vmax),
                dt=float(args.teacher_dt),
                T=float(args.T_final),
                poisson_sign=float(args.poisson_sign),
                snapshot_times=tuple(snapshot_times),
            )
            perturbation = _case_perturbations(
                config,
                linear_eps=float(args.linear_eps),
                linear_modes=linear_modes,
                linear_seed=int(args.linear_seed),
                weak_eps=float(args.weak_eps),
                strong_eps=float(args.strong_eps),
                nonlinear_k0=float(args.nonlinear_k0),
            )[case_name]
            print(f"[diagnostic] running Nx={spatial_nx}: {case_name}")
            f0 = equilibrium[:, None] * (
                1.0 + jnp.asarray(perturbation, dtype=jnp.float64)[None, :]
            )
            raw = run_semilagrangian_vlasov_poisson(config, f0)
            snapshots_by_grid[int(spatial_nx)] = np.asarray(
                raw["snapshot_f"],
                dtype=np.float64,
            )
            energy_by_case[case_name][int(spatial_nx)] = (
                np.asarray(raw["times"], dtype=np.float64),
                np.asarray(raw["energy"], dtype=np.float64),
            )

        summary_by_case[case_name] = {}
        for coarse_nx, refined_nx in zip(spatial_grids[:-1], spatial_grids[1:]):
            distribution_change, snapshot_changes = (
                _distribution_successive_x_change(
                    snapshots_by_grid[coarse_nx],
                    snapshots_by_grid[refined_nx],
                    equilibrium=equilibrium,
                )
            )
            coarse_times, coarse_energy = energy_by_case[case_name][coarse_nx]
            refined_times, refined_energy = energy_by_case[case_name][refined_nx]
            np.testing.assert_allclose(
                coarse_times,
                refined_times,
                rtol=0.0,
                atol=1e-13,
            )
            energy_changes = _energy_block_changes(
                coarse_energy,
                refined_energy,
                times=refined_times,
                block_edges=block_edges,
            )
            passes = bool(
                max(
                    float(energy_changes["global_energy_refinement_change"]),
                    distribution_change,
                )
                < float(args.relative_tolerance)
            )
            row: Dict[str, object] = {
                "case": case_name,
                "coarse_spatial_Nx": int(coarse_nx),
                "refined_spatial_Nx": int(refined_nx),
                **energy_changes,
                "max_snapshot_distribution_refinement_change": distribution_change,
                "passes_tolerance": passes,
            }
            records.append(row)
            summary_by_case[case_name][str(refined_nx)] = {
                key: value for key, value in row.items() if key != "case"
            }
            summary_by_case[case_name][str(refined_nx)][
                "snapshot_distribution_relative_changes"
            ] = [float(value) for value in snapshot_changes]

    target_refined_nx = min(nx for nx in spatial_grids if nx > target_nx)
    target_passes = all(
        bool(by_grid[str(target_refined_nx)]["passes_tolerance"])
        for by_grid in summary_by_case.values()
    )
    csv_path = outdir / "physical_spatial_grid_convergence.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    energy_path = outdir / "physical_spatial_grid_energy.png"
    _save_energy_plot(energy_by_case=energy_by_case, figure_path=energy_path)
    convergence_path = outdir / "physical_spatial_grid_convergence.png"
    _save_convergence_plot(
        summary_by_case=summary_by_case,
        figure_path=convergence_path,
    )
    payload = {
        "diagnostic": "physical_spatial_grid_self_convergence",
        "comparison": (
            "Only the periodic solver spatial grid Nx changes. Nv, dt, T, "
            "velocity domain, solver, and initial-condition parameters remain "
            "fixed. Coarse phase-space snapshots are Fourier resampled in x "
            "before direct comparison."
        ),
        "metric_definitions": {
            "global_energy_refinement_change": (
                "||energy_coarse-energy_refined||_L2(0,T) "
                "/ ||energy_refined||_L2(0,T)"
            ),
            "max_snapshot_distribution_refinement_change": (
                "max_t ||delta_f_coarse_to_refined-delta_f_refined||_L2(x,v) "
                "/ max_t ||delta_f_refined||_L2(x,v)"
            ),
        },
        "teacher": {
            "spatial_Nx": list(spatial_grids),
            "physical_Nv": int(args.teacher_Nv),
            "L": float(args.teacher_L),
            "vmin": float(args.teacher_vmin),
            "vmax": float(args.teacher_vmax),
            "dt": float(args.teacher_dt),
            "T_final": float(args.T_final),
        },
        "snapshot_times": list(snapshot_times),
        "relative_tolerance": float(args.relative_tolerance),
        "successive_refinement_summary": summary_by_case,
        "reused_teacher_snapshot_artifact": (
            str(reused_path) if reused_path is not None else None
        ),
        "recommendation": {
            "target_spatial_Nx": target_nx,
            "comparison_spatial_Nx": target_refined_nx,
            "target_passes_tolerance_for_all_cases": target_passes,
            "qualification": (
                f"Nx={target_nx} passes the successive-change gate against "
                f"Nx={target_refined_nx} for every representative case."
                if target_passes
                else f"Nx={target_nx} does not pass the successive-change gate "
                f"against Nx={target_refined_nx} for every representative case."
            ),
        },
    }
    json_path = outdir / "physical_spatial_grid_convergence.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Saved spatial-grid convergence JSON to {json_path}")
    print(f"Saved spatial-grid convergence CSV to {csv_path}")
    print(f"Saved spatial-grid energy figure to {energy_path}")
    print(f"Saved spatial-grid convergence figure to {convergence_path}")
    print(
        f"[diagnostic] Nx={target_nx} successive-change gate="
        f"{int(target_passes)} against Nx={target_refined_nx}"
    )


if __name__ == "__main__":
    main()
