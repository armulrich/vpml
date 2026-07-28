"""Physical velocity-grid convergence diagnostic for Landau teachers."""

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))

from model.train.interface_flux_rollout import sample_initial_condition
from model.diagnostics.projection_quadrature_convergence import (
    _load_teacher_snapshot_artifact,
)
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    cubic_bspline_interp_constant,
    cubic_bspline_prefilter_constant,
    gaussian_pdf,
    normalize_density_on_grid,
    run_semilagrangian_vlasov_poisson,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


def _parse_int_tuple(text: str) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_float_tuple(text: str) -> Tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    difference_norm = float(np.linalg.norm(value - reference))
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm <= np.finfo(np.float64).tiny:
        return float("nan")
    return difference_norm / reference_norm


def _json_float(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def _case_perturbations(
    config: PhysicalGridVlasovPoissonConfig,
    *,
    linear_eps: float,
    linear_modes: Sequence[float],
    linear_seed: int,
    weak_eps: float,
    strong_eps: float,
    nonlinear_k0: float,
) -> Dict[str, np.ndarray]:
    x = np.asarray(config.x, dtype=np.float64)
    linear = sample_initial_condition(
        np.random.default_rng(int(linear_seed)),
        x,
        modes=linear_modes,
        eps=float(linear_eps),
    )
    mode = np.cos(float(nonlinear_k0) * x)
    return {
        "linear_sample00": np.asarray(linear, dtype=np.float64),
        f"weak_eps{str(float(weak_eps)).replace('.', 'p')}": float(weak_eps) * mode,
        f"strong_eps{str(float(strong_eps)).replace('.', 'p')}": (
            float(strong_eps) * mode
        ),
    }


def _resample_velocity_snapshots(
    snapshots: np.ndarray,
    *,
    source_v: np.ndarray,
    target_v: np.ndarray,
) -> np.ndarray:
    snapshots = np.asarray(snapshots, dtype=np.float64)
    source_v = np.asarray(source_v, dtype=np.float64)
    target_v = np.asarray(target_v, dtype=np.float64)
    if snapshots.ndim != 3:
        raise ValueError(
            f"snapshots must have shape (time, Nv, Nx), got {snapshots.shape}"
        )
    if int(snapshots.shape[1]) != int(source_v.size):
        raise ValueError("snapshot velocity dimension must match source_v")
    if source_v.ndim != 1 or target_v.ndim != 1:
        raise ValueError("source_v and target_v must be one-dimensional")
    source_dv = np.diff(source_v)
    if not np.all(source_dv > 0.0):
        raise ValueError("source_v must be strictly increasing")
    if not np.allclose(source_dv, source_dv[0], rtol=1e-12, atol=1e-14):
        raise ValueError("source_v must be uniformly spaced")
    if target_v[0] < source_v[0] or target_v[-1] > source_v[-1]:
        raise ValueError("target_v must lie within the source velocity interval")

    time_count, source_nv, nx = snapshots.shape
    flattened = np.transpose(snapshots, (1, 0, 2)).reshape(
        source_nv,
        time_count * nx,
    )
    target_coords = np.clip(
        (target_v - source_v[0]) / source_dv[0],
        0.0,
        float(source_nv - 1),
    )
    coordinates = np.broadcast_to(
        target_coords[:, None],
        (int(target_v.size), int(flattened.shape[1])),
    )
    sub = jnp.full((source_nv - 1,), 1.0, dtype=jnp.float64)
    diagonal = jnp.full((source_nv,), 4.0, dtype=jnp.float64)
    sup = jnp.full((source_nv - 1,), 1.0, dtype=jnp.float64)

    @jax.jit
    def resample(values: jax.Array) -> jax.Array:
        coefficients = cubic_bspline_prefilter_constant(
            values,
            sub,
            diagonal,
            sup,
        )
        return cubic_bspline_interp_constant(
            coefficients,
            jnp.asarray(coordinates, dtype=jnp.float64),
            cval=0.0,
        )

    resampled = np.asarray(
        resample(jnp.asarray(flattened, dtype=jnp.float64)),
        dtype=np.float64,
    )
    return np.transpose(
        resampled.reshape(int(target_v.size), time_count, nx),
        (1, 0, 2),
    )


def _distribution_successive_change(
    coarse_snapshots: np.ndarray,
    refined_snapshots: np.ndarray,
    *,
    coarse_equilibrium: np.ndarray,
    refined_equilibrium: np.ndarray,
    coarse_v: np.ndarray,
    refined_v: np.ndarray,
) -> Tuple[float, np.ndarray]:
    coarse_delta = np.asarray(coarse_snapshots, dtype=np.float64) - np.asarray(
        coarse_equilibrium,
        dtype=np.float64,
    )[None, :, None]
    refined_delta = np.asarray(refined_snapshots, dtype=np.float64) - np.asarray(
        refined_equilibrium,
        dtype=np.float64,
    )[None, :, None]
    coarse_on_refined = _resample_velocity_snapshots(
        coarse_delta,
        source_v=coarse_v,
        target_v=refined_v,
    )
    if coarse_on_refined.shape != refined_delta.shape:
        raise ValueError(
            "resampled coarse snapshots and refined snapshots must have equal shape"
        )

    difference_norms = np.linalg.norm(
        coarse_on_refined - refined_delta,
        axis=(1, 2),
    )
    refined_norms = np.linalg.norm(refined_delta, axis=(1, 2))
    denominator = max(float(np.max(refined_norms)), np.finfo(np.float64).tiny)
    max_normalized_change = float(np.max(difference_norms)) / denominator
    snapshot_relative_changes = difference_norms / np.maximum(
        refined_norms,
        np.finfo(np.float64).tiny,
    )
    return max_normalized_change, snapshot_relative_changes


def _energy_block_changes(
    coarse: np.ndarray,
    refined: np.ndarray,
    *,
    times: np.ndarray,
    block_edges: Sequence[float],
) -> Dict[str, Optional[float]]:
    values: Dict[str, Optional[float]] = {
        "global_energy_refinement_change": _json_float(
            _relative_l2(coarse, refined)
        )
    }
    for start, stop in zip(block_edges[:-1], block_edges[1:]):
        include_stop = math.isclose(float(stop), float(block_edges[-1]))
        mask = (times >= float(start) - 1e-12) & (
            times <= float(stop) + 1e-12
            if include_stop
            else times < float(stop) - 1e-12
        )
        key = f"energy_refinement_change_t{float(start):g}_to_{float(stop):g}"
        values[key] = _json_float(_relative_l2(coarse[mask], refined[mask]))
    return values


def _display_case(case_name: str) -> Tuple[str, str]:
    if case_name.startswith("linear"):
        return "Linear", "#1f4e79"
    if case_name.startswith("weak"):
        return r"Weak nonlinear ($\epsilon=0.1$)", "#2a9d8f"
    if case_name.startswith("strong"):
        return r"Strong nonlinear ($\epsilon=0.5$)", "#c44e52"
    return case_name, "#555555"


def _save_energy_plot(
    *,
    energy_by_case: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    figure_path: Path,
) -> None:
    case_names = tuple(energy_by_case)
    fig, axes = plt.subplots(
        len(case_names),
        1,
        figsize=(9.5, 2.65 * len(case_names)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    physical_grids = sorted(
        {
            int(grid)
            for by_grid in energy_by_case.values()
            for grid in by_grid
        }
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(physical_grids)))
    for axis, case_name in zip(axes, case_names):
        label, _ = _display_case(case_name)
        for color, physical_nv in zip(colors, physical_grids):
            times, energy = energy_by_case[case_name][int(physical_nv)]
            axis.semilogy(
                times,
                np.maximum(energy, np.finfo(np.float64).tiny),
                color=color,
                linewidth=1.35,
                label=rf"$N_v={physical_nv:,}$",
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
        ncol=len(physical_grids),
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
    all_refined_grids = tuple(
        sorted(
            {
                int(grid)
                for by_refined_grid in summary_by_case.values()
                for grid in by_refined_grid
            }
        )
    )
    x_positions = np.arange(len(all_refined_grids), dtype=np.float64)
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
                for grid in all_refined_grids
            ]
            axis.semilogy(
                x_positions,
                np.maximum(np.asarray(values, dtype=np.float64), 1e-16),
                marker="o",
                color=color,
                linewidth=1.6,
                label=label,
            )
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel(r"Refined physical velocity points $N_v$")
        axis.set_xticks(x_positions)
        axis.set_xticklabels([f"{grid:,}" for grid in all_refined_grids])
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


def _save_teacher_artifact(
    *,
    artifact_path: Path,
    config: PhysicalGridVlasovPoissonConfig,
    raw_by_case: Dict[str, Dict[str, np.ndarray]],
) -> None:
    payload: Dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "teacher_Nx": np.asarray(int(config.Nx), dtype=np.int64),
        "teacher_Nv": np.asarray(int(config.Nv), dtype=np.int64),
        "teacher_L": np.asarray(float(config.Lx), dtype=np.float64),
        "teacher_vmin": np.asarray(float(config.vmin), dtype=np.float64),
        "teacher_vmax": np.asarray(float(config.vmax), dtype=np.float64),
        "teacher_dt": np.asarray(float(config.dt), dtype=np.float64),
        "T_final": np.asarray(float(config.T), dtype=np.float64),
        "poisson_sign": np.asarray(float(config.poisson_sign), dtype=np.float64),
        "snapshot_times": np.asarray(config.snapshot_times, dtype=np.float64),
        "source_v": np.asarray(config.v, dtype=np.float64),
        "k_arr": np.asarray(config.k_arr, dtype=np.float64),
        "case_count": np.asarray(len(raw_by_case), dtype=np.int64),
    }
    for case_idx, (case_name, raw) in enumerate(raw_by_case.items()):
        prefix = f"case_{case_idx:03d}"
        payload[f"{prefix}_name"] = np.asarray(case_name)
        payload[f"{prefix}_snapshot_f"] = np.asarray(
            raw["snapshot_f"],
            dtype=np.float64,
        )
        payload[f"{prefix}_times"] = np.asarray(raw["times"], dtype=np.float64)
        payload[f"{prefix}_energy"] = np.asarray(raw["energy"], dtype=np.float64)
    np.savez(artifact_path, **payload)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refine the physical velocity grid of otherwise identical Landau "
            "teachers and compare electric-field energy plus the direct "
            "phase-space distribution."
        )
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--teacher-Nx", type=int, default=256)
    parser.add_argument(
        "--physical-Nv-list",
        type=str,
        default="512,1024,2048,4096",
    )
    parser.add_argument(
        "--snapshot-artifact-Nv",
        type=int,
        default=2048,
        help=(
            "Physical grid whose snapshots are saved for the independent "
            "projection-quadrature diagnostic."
        ),
    )
    parser.add_argument(
        "--coarse-teacher-snapshots",
        type=Path,
        default=None,
        help=(
            "Optional snapshot artifact to reuse as the coarsest physical grid. "
            "Only the finer grids in --physical-Nv-list are simulated."
        ),
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
    print_jax_runtime_summary(jax, context="physical velocity-grid diagnostic")
    args = _build_arg_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostic directory: {outdir}"
        )
    outdir.mkdir(parents=True)

    requested_physical_grids = tuple(
        sorted(set(_parse_int_tuple(args.physical_Nv_list)))
    )
    snapshot_times = _parse_float_tuple(args.snapshot_times)
    linear_modes = _parse_float_tuple(args.linear_modes)
    snapshot_artifact_nv = int(args.snapshot_artifact_Nv)
    if not requested_physical_grids:
        raise ValueError("--physical-Nv-list must contain at least one grid")
    if any(value <= 3 for value in requested_physical_grids):
        raise ValueError("every physical Nv must exceed three")
    if not snapshot_times:
        raise ValueError("--snapshot-times must not be empty")
    if max(snapshot_times) > float(args.T_final) + 1e-12:
        raise ValueError("snapshot times must not exceed --T-final")
    if float(args.relative_tolerance) <= 0.0:
        raise ValueError("--relative-tolerance must be positive")

    snapshots_by_case: Dict[str, Dict[int, np.ndarray]] = {}
    energy_by_case: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    equilibrium_by_grid: Dict[int, np.ndarray] = {}
    config_by_grid: Dict[int, PhysicalGridVlasovPoissonConfig] = {}
    artifact_raw_by_case: Dict[str, Dict[str, np.ndarray]] = {}
    reused_physical_nv: Optional[int] = None

    if args.coarse_teacher_snapshots is not None:
        coarse_artifact = args.coarse_teacher_snapshots.resolve()
        if not coarse_artifact.is_file():
            raise FileNotFoundError(
                f"Coarse teacher snapshot artifact does not exist: {coarse_artifact}"
            )
        coarse_config, coarse_snapshots, coarse_energy = (
            _load_teacher_snapshot_artifact(coarse_artifact)
        )
        reused_physical_nv = int(coarse_config.Nv)
        expected_config = {
            "Nx": int(args.teacher_Nx),
            "Lx": float(args.teacher_L),
            "vmin": float(args.teacher_vmin),
            "vmax": float(args.teacher_vmax),
            "dt": float(args.teacher_dt),
            "T": float(args.T_final),
            "poisson_sign": float(args.poisson_sign),
        }
        actual_config = {
            "Nx": int(coarse_config.Nx),
            "Lx": float(coarse_config.Lx),
            "vmin": float(coarse_config.vmin),
            "vmax": float(coarse_config.vmax),
            "dt": float(coarse_config.dt),
            "T": float(coarse_config.T),
            "poisson_sign": float(coarse_config.poisson_sign),
        }
        if actual_config != expected_config:
            raise ValueError(
                "Reused coarse teacher configuration does not match the "
                f"requested diagnostic: actual={actual_config}, "
                f"expected={expected_config}"
            )
        np.testing.assert_allclose(
            np.asarray(coarse_config.snapshot_times, dtype=np.float64),
            np.asarray(snapshot_times, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )
        if any(
            int(value) <= reused_physical_nv
            for value in requested_physical_grids
        ):
            raise ValueError(
                "Every simulated physical grid must be finer than the reused "
                f"coarse grid Nv={reused_physical_nv}"
            )
        config_by_grid[reused_physical_nv] = coarse_config
        equilibrium_by_grid[reused_physical_nv] = np.asarray(
            normalize_density_on_grid(
                gaussian_pdf(coarse_config.v, mean=0.0, sigma=1.0),
                coarse_config.v,
            ),
            dtype=np.float64,
        )
        for case_name, snapshots in coarse_snapshots.items():
            snapshots_by_case.setdefault(case_name, {})[
                reused_physical_nv
            ] = np.asarray(snapshots, dtype=np.float64)
            energy_by_case.setdefault(case_name, {})[
                reused_physical_nv
            ] = (
                np.asarray(
                    coarse_energy[f"{case_name}_times"],
                    dtype=np.float64,
                ),
                np.asarray(
                    coarse_energy[f"{case_name}_energy"],
                    dtype=np.float64,
                ),
            )
        print(
            "[diagnostic] reusing coarse physical teacher snapshots from "
            f"{coarse_artifact} (Nv={reused_physical_nv})"
        )

    physical_grids = tuple(
        sorted(
            set(requested_physical_grids)
            | ({reused_physical_nv} if reused_physical_nv is not None else set())
        )
    )
    if len(physical_grids) < 2:
        raise ValueError(
            "At least two physical grids are required, including any reused grid"
        )
    if snapshot_artifact_nv not in physical_grids:
        raise ValueError("--snapshot-artifact-Nv must appear in the physical grids")

    print(
        "[diagnostic] physical velocity grids: "
        + ",".join(str(value) for value in physical_grids)
        + f" dt={float(args.teacher_dt):g} T={float(args.T_final):g}"
    )
    print(
        "[diagnostic] direct distribution comparison: "
        "coarse cubic-B-spline state resampled to each refined physical grid"
    )

    for physical_nv in physical_grids:
        if reused_physical_nv is not None and int(physical_nv) == reused_physical_nv:
            continue
        config = PhysicalGridVlasovPoissonConfig(
            Nx=int(args.teacher_Nx),
            Nv=int(physical_nv),
            Lx=float(args.teacher_L),
            vmin=float(args.teacher_vmin),
            vmax=float(args.teacher_vmax),
            dt=float(args.teacher_dt),
            T=float(args.T_final),
            poisson_sign=float(args.poisson_sign),
            snapshot_times=tuple(snapshot_times),
        )
        config_by_grid[int(physical_nv)] = config
        equilibrium = np.asarray(
            normalize_density_on_grid(
                gaussian_pdf(config.v, mean=0.0, sigma=1.0),
                config.v,
            ),
            dtype=np.float64,
        )
        equilibrium_by_grid[int(physical_nv)] = equilibrium
        perturbations = _case_perturbations(
            config,
            linear_eps=float(args.linear_eps),
            linear_modes=linear_modes,
            linear_seed=int(args.linear_seed),
            weak_eps=float(args.weak_eps),
            strong_eps=float(args.strong_eps),
            nonlinear_k0=float(args.nonlinear_k0),
        )
        if snapshots_by_case and set(perturbations) != set(snapshots_by_case):
            raise ValueError(
                "Reused and generated teacher artifacts have different cases: "
                f"reused={sorted(snapshots_by_case)}, "
                f"generated={sorted(perturbations)}"
            )
        for case_name, perturbation in perturbations.items():
            print(
                f"[diagnostic] running physical teacher Nv={physical_nv}: "
                f"{case_name}"
            )
            f0 = equilibrium[:, None] * (
                1.0 + jnp.asarray(perturbation, dtype=jnp.float64)[None, :]
            )
            raw_untyped = run_semilagrangian_vlasov_poisson(config, f0)
            raw = {
                key: np.asarray(value)
                for key, value in raw_untyped.items()
                if key in {"snapshot_f", "times", "energy"}
            }
            snapshots_by_case.setdefault(case_name, {})[int(physical_nv)] = (
                np.asarray(raw["snapshot_f"], dtype=np.float64)
            )
            energy_by_case.setdefault(case_name, {})[int(physical_nv)] = (
                np.asarray(raw["times"], dtype=np.float64),
                np.asarray(raw["energy"], dtype=np.float64),
            )
            if int(physical_nv) == snapshot_artifact_nv:
                artifact_raw_by_case[case_name] = raw

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
    for case_name, by_grid in snapshots_by_case.items():
        summary_by_case[case_name] = {}
        for coarse_nv, refined_nv in zip(physical_grids[:-1], physical_grids[1:]):
            coarse_config = config_by_grid[int(coarse_nv)]
            refined_config = config_by_grid[int(refined_nv)]
            distribution_change, snapshot_distribution_changes = (
                _distribution_successive_change(
                    by_grid[int(coarse_nv)],
                    by_grid[int(refined_nv)],
                    coarse_equilibrium=equilibrium_by_grid[int(coarse_nv)],
                    refined_equilibrium=equilibrium_by_grid[int(refined_nv)],
                    coarse_v=np.asarray(coarse_config.v, dtype=np.float64),
                    refined_v=np.asarray(refined_config.v, dtype=np.float64),
                )
            )
            coarse_times, coarse_energy = energy_by_case[case_name][int(coarse_nv)]
            refined_times, refined_energy = energy_by_case[case_name][int(refined_nv)]
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
            global_energy_change = float(
                energy_changes["global_energy_refinement_change"]
            )
            passes = bool(
                max(global_energy_change, distribution_change)
                < float(args.relative_tolerance)
            )
            row: Dict[str, object] = {
                "case": case_name,
                "coarse_physical_Nv": int(coarse_nv),
                "refined_physical_Nv": int(refined_nv),
                **energy_changes,
                "max_snapshot_distribution_refinement_change": (
                    distribution_change
                ),
                "passes_tolerance": passes,
            }
            records.append(row)
            summary_by_case[case_name][str(int(refined_nv))] = {
                key: value for key, value in row.items() if key != "case"
            }
            summary_by_case[case_name][str(int(refined_nv))][
                "snapshot_distribution_relative_changes"
            ] = [
                float(value) for value in snapshot_distribution_changes
            ]

    finest_nv = int(physical_grids[-1])
    finest_pair_passes = all(
        bool(by_grid[str(finest_nv)]["passes_tolerance"])
        for by_grid in summary_by_case.values()
    )
    teacher_artifact_path = (
        outdir / f"physical_teacher_nv{snapshot_artifact_nv}_snapshots.npz"
    )
    _save_teacher_artifact(
        artifact_path=teacher_artifact_path,
        config=config_by_grid[snapshot_artifact_nv],
        raw_by_case=artifact_raw_by_case,
    )

    csv_path = outdir / "physical_velocity_grid_convergence.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    energy_path = outdir / "physical_velocity_grid_energy.png"
    _save_energy_plot(energy_by_case=energy_by_case, figure_path=energy_path)
    convergence_path = outdir / "physical_velocity_grid_convergence.png"
    _save_convergence_plot(
        summary_by_case=summary_by_case,
        figure_path=convergence_path,
    )

    payload = {
        "diagnostic": "physical_velocity_grid_self_convergence",
        "physical_discretization": (
            "The semi-Lagrangian solver evolves nodal f(x,v,t) on each physical "
            "velocity grid and uses cubic B-splines for interpolation."
        ),
        "comparison": (
            "Electric-field energy is compared over the complete trajectory. "
            "For direct phase-space comparison, each coarse perturbation "
            "f-f_eq is cubic-B-spline resampled to the next physical grid before "
            "the L2_xv change is measured. No Hermite projection is used."
        ),
        "metric_definitions": {
            "global_energy_refinement_change": (
                "||E_coarse-E_refined||_L2(0,T) / ||E_refined||_L2(0,T)"
            ),
            "max_snapshot_distribution_refinement_change": (
                "max_t ||delta_f_coarse_to_refined-delta_f_refined||_L2(x,v) "
                "/ max_t ||delta_f_refined||_L2(x,v)"
            ),
        },
        "teacher": {
            "Nx": int(args.teacher_Nx),
            "physical_Nv": list(int(value) for value in physical_grids),
            "L": float(args.teacher_L),
            "vmin": float(args.teacher_vmin),
            "vmax": float(args.teacher_vmax),
            "dt": float(args.teacher_dt),
            "T_final": float(args.T_final),
        },
        "snapshot_times": list(float(value) for value in snapshot_times),
        "relative_tolerance": float(args.relative_tolerance),
        "successive_refinement_summary": summary_by_case,
        "recommendation": {
            "finest_physical_Nv_tested": finest_nv,
            "finest_pair_passes_tolerance_for_all_cases": finest_pair_passes,
            "successive_change_gate_physical_Nv": (
                finest_nv if finest_pair_passes else None
            ),
            "physical_Nv_for_followup": finest_nv,
            "qualification": (
                "The finest successive-grid comparison passes the requested gate."
                if finest_pair_passes
                else "Finest tested grid only; the requested tolerance was not "
                "established for every representative case."
            ),
        },
        "projection_source_physical_Nv": snapshot_artifact_nv,
        "projection_source_teacher_snapshot_artifact": str(
            teacher_artifact_path
        ),
    }
    json_path = outdir / "physical_velocity_grid_convergence.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Saved physical-grid convergence JSON to {json_path}")
    print(f"Saved physical-grid convergence CSV to {csv_path}")
    print(f"Saved physical-grid energy figure to {energy_path}")
    print(f"Saved physical-grid convergence figure to {convergence_path}")
    print(
        "Saved reusable projection-source teacher snapshots to "
        f"{teacher_artifact_path}"
    )
    print(
        "[diagnostic] physical Nv recommendation: "
        f"followup={finest_nv} successive_change_gate="
        f"{int(finest_pair_passes)}"
    )


if __name__ == "__main__":
    main()
