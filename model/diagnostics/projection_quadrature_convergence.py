"""Convergence diagnostic for fixed-teacher Fourier-Hermite projection quadrature."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from vpml.jax_runtime import bootstrap_jax_runtime, print_jax_runtime_summary

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))

from model.train.interface_flux_rollout import (
    CANONICAL_NV_TARGETS,
    INTERFACE_FLUX_PROJECTION_SCHEME,
    sample_initial_condition,
)
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    build_cubic_spline_hermite_projection_matrix,
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


def _relative_l2(value: np.ndarray, reference: np.ndarray) -> Tuple[float, float, float]:
    difference_norm = float(np.linalg.norm(value - reference))
    reference_norm = float(np.linalg.norm(reference))
    relative = (
        difference_norm / reference_norm
        if reference_norm > np.finfo(np.float64).tiny
        else float("nan")
    )
    return relative, difference_norm, reference_norm


def _snapshot_digest(snapshot_f: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(snapshot_f, dtype=np.float64)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _project_snapshots(
    snapshot_f: np.ndarray,
    equilibrium: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    perturbation = np.asarray(snapshot_f, dtype=np.float64) - equilibrium[None, :, None]
    time_count, source_nv, nx = perturbation.shape
    flattened = np.transpose(perturbation, (1, 0, 2)).reshape(source_nv, time_count * nx)
    moments = projection_matrix @ flattened
    moments = moments.reshape(int(projection_matrix.shape[0]), time_count, nx)
    moments = np.transpose(moments, (1, 0, 2))
    return np.fft.rfft(moments, axis=2).astype(np.complex128)


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
        f"strong_eps{str(float(strong_eps)).replace('.', 'p')}": float(strong_eps) * mode,
    }


def _load_teacher_snapshot_artifact(
    artifact_path: Path,
) -> Tuple[
    PhysicalGridVlasovPoissonConfig,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    with np.load(artifact_path, allow_pickle=False) as payload:
        schema_version = int(np.asarray(payload["schema_version"]).item())
        if schema_version != 1:
            raise ValueError(
                f"Unsupported teacher snapshot artifact schema: {schema_version}"
            )
        snapshot_times = tuple(
            float(value) for value in np.asarray(payload["snapshot_times"])
        )
        config = PhysicalGridVlasovPoissonConfig(
            Nx=int(np.asarray(payload["teacher_Nx"]).item()),
            Nv=int(np.asarray(payload["teacher_Nv"]).item()),
            Lx=float(np.asarray(payload["teacher_L"]).item()),
            vmin=float(np.asarray(payload["teacher_vmin"]).item()),
            vmax=float(np.asarray(payload["teacher_vmax"]).item()),
            dt=float(np.asarray(payload["teacher_dt"]).item()),
            T=float(np.asarray(payload["T_final"]).item()),
            poisson_sign=float(np.asarray(payload["poisson_sign"]).item()),
            snapshot_times=snapshot_times,
        )
        case_count = int(np.asarray(payload["case_count"]).item())
        snapshots_by_case: Dict[str, np.ndarray] = {}
        energy_payload: Dict[str, np.ndarray] = {}
        for case_idx in range(case_count):
            prefix = f"case_{case_idx:03d}"
            case_name = str(np.asarray(payload[f"{prefix}_name"]).item())
            snapshots_by_case[case_name] = np.asarray(
                payload[f"{prefix}_snapshot_f"],
                dtype=np.float64,
            )
            energy_payload[f"{case_name}_times"] = np.asarray(
                payload[f"{prefix}_times"],
                dtype=np.float64,
            )
            energy_payload[f"{case_name}_energy"] = np.asarray(
                payload[f"{prefix}_energy"],
                dtype=np.float64,
            )
        source_v = np.asarray(payload["source_v"], dtype=np.float64)
        k_arr = np.asarray(payload["k_arr"], dtype=np.float64)

    np.testing.assert_allclose(
        source_v,
        np.asarray(config.v, dtype=np.float64),
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        k_arr,
        np.asarray(config.k_arr, dtype=np.float64),
        rtol=0.0,
        atol=1e-13,
    )
    return config, snapshots_by_case, energy_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hold one physical-grid teacher trajectory fixed while refining only "
            "the velocity quadrature applied to its cubic-spline representation."
        )
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--teacher-snapshots",
        type=Path,
        default=None,
        help=(
            "Optional snapshot artifact from physical_velocity_grid_convergence. "
            "When provided, the physical teacher is reused instead of rerun."
        ),
    )
    parser.add_argument("--teacher-Nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", type=int, default=512)
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
    parser.add_argument(
        "--projection-quadrature-Nv-list",
        type=str,
        default="512,1024,2048,4096,8192,16384",
    )
    parser.add_argument("--reference-projection-Nv", type=int, default=16384)
    parser.add_argument("--projection-order", type=int, default=65)
    parser.add_argument("--cutoffs", type=str, default="6,7,12,20,36,64")
    parser.add_argument("--linear-eps", type=float, default=0.01)
    parser.add_argument("--linear-modes", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--linear-seed", type=int, default=0)
    parser.add_argument("--weak-eps", type=float, default=0.1)
    parser.add_argument("--strong-eps", type=float, default=0.5)
    parser.add_argument("--nonlinear-k0", type=float, default=0.5)
    parser.add_argument("--poisson-sign", type=float, default=1.0)
    return parser


def _json_float(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def _max_finite(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else float("nan")


def _save_convergence_plot(
    *,
    refinement_summary_by_case: Dict[str, Dict[str, Dict[str, float]]],
    figure_path: Path,
) -> None:
    display = {
        "linear_sample00": ("Linear", "#1f4e79"),
        "weak_eps0p1": (r"Weak nonlinear ($\epsilon=0.1$)", "#2a9d8f"),
        "strong_eps0p5": (r"Strong nonlinear ($\epsilon=0.5$)", "#c44e52"),
    }
    all_refined_grids = tuple(
        sorted(
            {
                int(grid)
                for by_refined_grid in refinement_summary_by_case.values()
                for grid in by_refined_grid
            }
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for case_name, by_refined_grid in refinement_summary_by_case.items():
        refined_grids = tuple(sorted(int(value) for value in by_refined_grid))
        c_values = [
            by_refined_grid[str(int(grid))][
                "global_C0_through_N_refinement_change"
            ]
            for grid in refined_grids
        ]
        q_values = [
            by_refined_grid[str(int(grid))]["global_qN_refinement_change"]
            for grid in refined_grids
        ]
        label, color = display.get(case_name, (case_name, None))
        axes[0].loglog(
            refined_grids,
            np.maximum(np.asarray(c_values, dtype=np.float64), 1e-16),
            marker="o",
            label=label,
            color=color,
        )
        axes[1].loglog(
            refined_grids,
            np.maximum(np.asarray(q_values, dtype=np.float64), 1e-16),
            marker="o",
            label=label,
            color=color,
        )
    for axis, title in zip(
        axes,
        (
            r"Resolved coefficients $C_{0:N}$",
            r"Interface flux $q_N$",
        ),
    ):
        axis.set_title(title)
        axis.set_xlabel(r"Projection quadrature points $M$")
        axis.set_xticks(all_refined_grids)
        axis.set_xticklabels([f"{grid:,}" for grid in all_refined_grids])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"Relative refinement change $\delta_M^C$")
    axes[1].set_ylabel(r"Relative refinement change $\delta_M^q$")
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
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.82, wspace=0.18)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    print_jax_runtime_summary(jax, context="projection quadrature diagnostic")
    args = _build_arg_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic directory: {outdir}")
    outdir.mkdir(parents=True)

    snapshot_times = _parse_float_tuple(args.snapshot_times)
    quadrature_grids = _parse_int_tuple(args.projection_quadrature_Nv_list)
    cutoffs = _parse_int_tuple(args.cutoffs)
    linear_modes = _parse_float_tuple(args.linear_modes)
    if not snapshot_times:
        raise ValueError("--snapshot-times must not be empty")
    if not quadrature_grids:
        raise ValueError("--projection-quadrature-Nv-list must not be empty")
    if int(args.reference_projection_Nv) not in quadrature_grids:
        raise ValueError("--reference-projection-Nv must appear in the quadrature grid list")
    if max(snapshot_times) > float(args.T_final) + 1e-12:
        raise ValueError("snapshot times must not exceed --T-final")
    if max(cutoffs) >= int(args.projection_order):
        raise ValueError("--projection-order must exceed every requested cutoff")
    if tuple(cutoffs) != tuple(CANONICAL_NV_TARGETS):
        print(
            "[diagnostic] warning: requested cutoffs differ from the canonical cycle "
            f"{CANONICAL_NV_TARGETS}"
        )

    snapshots_by_case: Optional[Dict[str, np.ndarray]] = None
    reused_energy_payload: Optional[Dict[str, np.ndarray]] = None
    if args.teacher_snapshots is not None:
        teacher_artifact = args.teacher_snapshots.resolve()
        if not teacher_artifact.is_file():
            raise FileNotFoundError(
                f"Teacher snapshot artifact does not exist: {teacher_artifact}"
            )
        config, snapshots_by_case, reused_energy_payload = (
            _load_teacher_snapshot_artifact(teacher_artifact)
        )
        if not np.allclose(
            np.asarray(snapshot_times, dtype=np.float64),
            np.asarray(config.snapshot_times, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "--snapshot-times must match the reused teacher snapshot artifact"
            )
        print(f"[diagnostic] reusing physical teacher snapshots from {teacher_artifact}")
    else:
        config = PhysicalGridVlasovPoissonConfig(
            Nx=int(args.teacher_Nx),
            Nv=int(args.teacher_Nv),
            Lx=float(args.teacher_L),
            vmin=float(args.teacher_vmin),
            vmax=float(args.teacher_vmax),
            dt=float(args.teacher_dt),
            T=float(args.T_final),
            poisson_sign=float(args.poisson_sign),
            snapshot_times=tuple(snapshot_times),
        )
    source_v = np.asarray(config.v, dtype=np.float64)
    equilibrium = np.asarray(
        normalize_density_on_grid(
            gaussian_pdf(config.v, mean=0.0, sigma=1.0),
            config.v,
        ),
        dtype=np.float64,
    )
    k_arr = np.asarray(config.k_arr, dtype=np.float64)
    perturbations = (
        _case_perturbations(
            config,
            linear_eps=float(args.linear_eps),
            linear_modes=linear_modes,
            linear_seed=int(args.linear_seed),
            weak_eps=float(args.weak_eps),
            strong_eps=float(args.strong_eps),
            nonlinear_k0=float(args.nonlinear_k0),
        )
        if snapshots_by_case is None
        else None
    )

    print(
        "[diagnostic] fixed physical teacher grid: "
        f"Nx={int(config.Nx)} Nv={int(config.Nv)} dt={float(config.dt):g} "
        f"T={float(config.T):g}"
    )
    print(
        "[diagnostic] projection quadrature grids: "
        + ",".join(str(value) for value in quadrature_grids)
        + f" finest_tested={int(args.reference_projection_Nv)}"
    )

    projection_matrices: Dict[int, np.ndarray] = {}
    for quadrature_nv in quadrature_grids:
        print(f"[diagnostic] building projection operator Nv={int(quadrature_nv)}")
        projection_matrices[int(quadrature_nv)] = np.asarray(
            build_cubic_spline_hermite_projection_matrix(
                config.v,
                int(args.projection_order),
                int(quadrature_nv),
                vth=1.0,
            ),
            dtype=np.float64,
        )

    projected: Dict[str, Dict[int, np.ndarray]] = {}
    snapshot_hashes: Dict[str, str] = {}
    energy_payload: Dict[str, np.ndarray] = (
        dict(reused_energy_payload) if reused_energy_payload is not None else {}
    )
    case_names = (
        tuple(snapshots_by_case)
        if snapshots_by_case is not None
        else tuple(perturbations or {})
    )
    for case_name in case_names:
        if snapshots_by_case is not None:
            print(f"[diagnostic] projecting reused teacher trajectory: {case_name}")
            snapshots = np.asarray(snapshots_by_case[case_name], dtype=np.float64)
        else:
            print(f"[diagnostic] running fixed teacher trajectory: {case_name}")
            perturbation = (perturbations or {})[case_name]
            f0 = equilibrium[:, None] * (
                1.0 + jnp.asarray(perturbation, dtype=jnp.float64)[None, :]
            )
            raw = run_semilagrangian_vlasov_poisson(config, f0)
            snapshots = np.asarray(raw["snapshot_f"], dtype=np.float64)
            energy_payload[f"{case_name}_times"] = np.asarray(
                raw["times"],
                dtype=np.float64,
            )
            energy_payload[f"{case_name}_energy"] = np.asarray(
                raw["energy"],
                dtype=np.float64,
            )
        snapshot_hashes[case_name] = _snapshot_digest(snapshots)
        projected[case_name] = {
            int(quadrature_nv): _project_snapshots(
                snapshots,
                equilibrium,
                projection_matrices[int(quadrature_nv)],
            )
            for quadrature_nv in quadrature_grids
        }

    np.savez_compressed(outdir / "fixed_teacher_energy.npz", **energy_payload)

    records = []
    summary_by_case: Dict[str, Dict[str, Dict[str, Optional[float] | bool]]] = {}
    reference_nv = int(args.reference_projection_Nv)
    for case_name, by_grid in projected.items():
        reference = by_grid[reference_nv]
        summary_by_case[case_name] = {}
        for quadrature_nv in quadrature_grids:
            current = by_grid[int(quadrature_nv)]
            c_relative_values = []
            q_relative_values = []
            c_absolute_values = []
            c_reference_values = []
            q_absolute_values = []
            q_reference_values = []
            for time_idx, snapshot_time in enumerate(snapshot_times):
                for cutoff in cutoffs:
                    c_rel, c_abs, c_ref_norm = _relative_l2(
                        current[time_idx, : int(cutoff) + 1, :],
                        reference[time_idx, : int(cutoff) + 1, :],
                    )
                    q_current = (
                        -1j
                        * k_arr[1:]
                        * math.sqrt(float(cutoff))
                        * current[time_idx, int(cutoff), 1:]
                    )
                    q_reference = (
                        -1j
                        * k_arr[1:]
                        * math.sqrt(float(cutoff))
                        * reference[time_idx, int(cutoff), 1:]
                    )
                    q_rel, q_abs, q_ref_norm = _relative_l2(q_current, q_reference)
                    c_relative_values.append(c_rel)
                    q_relative_values.append(q_rel)
                    c_absolute_values.append(c_abs)
                    c_reference_values.append(c_ref_norm)
                    q_absolute_values.append(q_abs)
                    q_reference_values.append(q_ref_norm)
                    records.append(
                        {
                            "case": case_name,
                            "teacher_Nv": int(config.Nv),
                            "projection_quadrature_Nv": int(quadrature_nv),
                            "reference_projection_Nv": reference_nv,
                            "time": float(snapshot_time),
                            "cutoff": int(cutoff),
                            "C0_through_N_relative_l2": c_rel,
                            "C0_through_N_absolute_l2": c_abs,
                            "C0_through_N_reference_l2": c_ref_norm,
                            "qN_relative_l2": q_rel,
                            "qN_absolute_l2": q_abs,
                            "qN_reference_l2": q_ref_norm,
                        }
                    )
            max_c = _max_finite(c_relative_values)
            max_q = _max_finite(q_relative_values)
            global_c = math.sqrt(
                sum(value * value for value in c_absolute_values)
                / max(sum(value * value for value in c_reference_values), np.finfo(np.float64).tiny)
            )
            global_q = math.sqrt(
                sum(value * value for value in q_absolute_values)
                / max(sum(value * value for value in q_reference_values), np.finfo(np.float64).tiny)
            )
            peak_scaled_c = max(c_absolute_values) / max(
                max(c_reference_values),
                np.finfo(np.float64).tiny,
            )
            peak_scaled_q = max(q_absolute_values) / max(
                max(q_reference_values),
                np.finfo(np.float64).tiny,
            )
            summary_by_case[case_name][str(int(quadrature_nv))] = {
                "max_C0_through_N_relative_l2": _json_float(max_c),
                "max_qN_relative_l2": _json_float(max_q),
                "global_C0_through_N_relative_l2": _json_float(global_c),
                "global_qN_relative_l2": _json_float(global_q),
                "max_C0_through_N_absolute_over_peak_reference": _json_float(
                    peak_scaled_c
                ),
                "max_qN_absolute_over_peak_reference": _json_float(peak_scaled_q),
                "passes_one_percent": bool(
                    max(global_c, global_q, peak_scaled_c, peak_scaled_q) < 0.01
                ),
            }

    refinement_records = []
    refinement_summary_by_case: Dict[
        str, Dict[str, Dict[str, Optional[float] | bool | int]]
    ] = {}
    ordered_grids = tuple(sorted(int(value) for value in quadrature_grids))
    for case_name, by_grid in projected.items():
        refinement_summary_by_case[case_name] = {}
        for coarse_nv, refined_nv in zip(ordered_grids[:-1], ordered_grids[1:]):
            coarse = by_grid[coarse_nv]
            refined = by_grid[refined_nv]
            c_difference_squared = 0.0
            c_refined_squared = 0.0
            q_difference_squared = 0.0
            q_refined_squared = 0.0
            for time_idx in range(len(snapshot_times)):
                for cutoff in cutoffs:
                    c_coarse = coarse[time_idx, : int(cutoff) + 1, :]
                    c_refined = refined[time_idx, : int(cutoff) + 1, :]
                    c_difference_squared += float(
                        np.linalg.norm(c_refined - c_coarse) ** 2
                    )
                    c_refined_squared += float(np.linalg.norm(c_refined) ** 2)
                    q_coarse = (
                        -1j
                        * k_arr[1:]
                        * math.sqrt(float(cutoff))
                        * coarse[time_idx, int(cutoff), 1:]
                    )
                    q_refined = (
                        -1j
                        * k_arr[1:]
                        * math.sqrt(float(cutoff))
                        * refined[time_idx, int(cutoff), 1:]
                    )
                    q_difference_squared += float(
                        np.linalg.norm(q_refined - q_coarse) ** 2
                    )
                    q_refined_squared += float(np.linalg.norm(q_refined) ** 2)
            c_change = math.sqrt(
                c_difference_squared
                / max(c_refined_squared, np.finfo(np.float64).tiny)
            )
            q_change = math.sqrt(
                q_difference_squared
                / max(q_refined_squared, np.finfo(np.float64).tiny)
            )
            row = {
                "case": case_name,
                "teacher_Nv": int(config.Nv),
                "coarse_projection_Nv": coarse_nv,
                "refined_projection_Nv": refined_nv,
                "refinement_ratio": float(refined_nv) / float(coarse_nv),
                "global_C0_through_N_refinement_change": c_change,
                "global_qN_refinement_change": q_change,
                "passes_one_percent_change": bool(max(c_change, q_change) < 0.01),
            }
            refinement_records.append(row)
            refinement_summary_by_case[case_name][str(refined_nv)] = {
                key: value for key, value in row.items() if key != "case"
            }

    csv_path = outdir / "projection_quadrature_convergence.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    refinement_csv_path = outdir / "projection_quadrature_refinement_summary.csv"
    with refinement_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(refinement_records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(refinement_records)

    payload = {
        "finest_grid_comparison": (
            f"Fixed physical-grid teacher trajectory at Nv={int(config.Nv)}; "
            f"Nv={reference_nv} is the finest projection quadrature tested, "
            "not an exact solution"
        ),
        "same_physical_trajectory_for_every_projection_grid": True,
        "projection_scheme": INTERFACE_FLUX_PROJECTION_SCHEME,
        "teacher": {
            "Nx": int(config.Nx),
            "Nv": int(config.Nv),
            "L": float(config.Lx),
            "vmin": float(config.vmin),
            "vmax": float(config.vmax),
            "dt": float(config.dt),
            "T_final": float(config.T),
        },
        "reused_teacher_snapshot_artifact": (
            str(args.teacher_snapshots.resolve())
            if args.teacher_snapshots is not None
            else None
        ),
        "projection_order": int(args.projection_order),
        "cutoffs": list(int(value) for value in cutoffs),
        "snapshot_times": list(float(value) for value in snapshot_times),
        "projection_quadrature_Nv": list(int(value) for value in quadrature_grids),
        "reference_projection_Nv": reference_nv,
        "snapshot_sha256": snapshot_hashes,
        "error_summary": {
            "raw_per_snapshot_relative_errors": (
                "Retained in CSV and max fields; may be ill-conditioned when a reference "
                "coefficient or interface flux is numerically zero."
            ),
            "headline_metric": (
                "Successive-refinement relative change compares M/2 with M and is a "
                "self-convergence diagnostic, not a discretization-error estimate."
            ),
            "one_percent_gate": (
                "The headline one-percent gate uses global relative L2 changes under "
                "successive projection-grid doubling for both C and q."
            ),
        },
        "successive_refinement_summary": refinement_summary_by_case,
        "finest_grid_comparison_summary": summary_by_case,
    }
    json_path = outdir / "projection_quadrature_convergence.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    figure_path = outdir / "projection_quadrature_convergence.png"
    _save_convergence_plot(
        refinement_summary_by_case=refinement_summary_by_case,
        figure_path=figure_path,
    )

    print(f"Saved projection convergence JSON to {json_path}")
    print(f"Saved projection convergence CSV to {csv_path}")
    print(f"Saved projection refinement summary to {refinement_csv_path}")
    print(f"Saved projection convergence plot to {figure_path}")
    selected_refined_nv = (
        4096 if 4096 in ordered_grids[1:] else int(ordered_grids[1])
    )
    selected_coarse_nv = ordered_grids[
        ordered_grids.index(selected_refined_nv) - 1
    ]
    for case_name in projected:
        selected = refinement_summary_by_case[case_name].get(
            str(selected_refined_nv)
        )
        if selected is not None:
            print(
                f"[diagnostic] {case_name} projection "
                f"{selected_coarse_nv}->{selected_refined_nv}: "
                f"global_C_change="
                f"{selected['global_C0_through_N_refinement_change']:.6e} "
                f"global_q_change={selected['global_qN_refinement_change']:.6e} "
                f"passes_1pct_change="
                f"{int(bool(selected['passes_one_percent_change']))}"
            )


if __name__ == "__main__":
    main()
