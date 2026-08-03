"""Joint spatial/velocity convergence diagnostic for Landau teachers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
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

from model.diagnostics.physical_spatial_grid_convergence import (
    _distribution_successive_x_change,
)
from model.diagnostics.physical_velocity_grid_convergence import (
    _case_perturbations,
    _display_case,
    _distribution_successive_change,
    _energy_block_changes,
    _parse_float_tuple,
    _resample_velocity_snapshots,
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


def _atomic_savez(path: Path, **payload: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _pair_directory(root: Path, nx: int, nv: int) -> Path:
    return root / f"Nx{int(nx)}_Nv{int(nv)}"


def _simulation_metadata(
    *,
    nx: int,
    nv: int,
    args: argparse.Namespace,
    snapshot_times: Sequence[float],
) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "Nx": int(nx),
        "Nv": int(nv),
        "L": float(args.teacher_L),
        "vmin": float(args.teacher_vmin),
        "vmax": float(args.teacher_vmax),
        "dt": float(args.teacher_dt),
        "T_final": float(args.T_final),
        "poisson_sign": float(args.poisson_sign),
        "snapshot_times": [float(value) for value in snapshot_times],
        "linear_eps": float(args.linear_eps),
        "linear_modes": [
            float(value) for value in _parse_float_tuple(args.linear_modes)
        ],
        "linear_seed": int(args.linear_seed),
        "weak_eps": float(args.weak_eps),
        "strong_eps": float(args.strong_eps),
        "nonlinear_k0": float(args.nonlinear_k0),
    }


def _parse_case_names(text: str, available: Sequence[str]) -> Tuple[str, ...]:
    requested = tuple(part.strip() for part in text.split(",") if part.strip())
    if not requested or requested == ("all",):
        return tuple(available)
    unknown = tuple(name for name in requested if name not in available)
    if unknown:
        raise ValueError(
            f"Unknown case(s) {unknown}; choose from {tuple(available)} or all"
        )
    return requested


def _run_case_segmented(
    *,
    config: PhysicalGridVlasovPoissonConfig,
    f0: np.ndarray | jax.Array,
    case_name: str,
    cases_dir: Path,
    snapshot_times: Sequence[float],
    checkpoint_interval_time: float,
) -> Dict[str, np.ndarray]:
    total_steps = int(config.nsteps)
    checkpoint_steps = int(round(float(checkpoint_interval_time) / float(config.dt)))
    if checkpoint_steps <= 0:
        checkpoint_steps = total_steps
    if not math.isclose(
        checkpoint_steps * float(config.dt),
        float(checkpoint_interval_time),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) and checkpoint_steps != total_steps:
        raise ValueError("checkpoint interval must be an integer number of time steps")

    progress_dir = cases_dir / f".{case_name}.progress"
    progress_dir.mkdir(exist_ok=True)
    checkpoint_path = progress_dir / "checkpoint.npz"
    if checkpoint_path.exists():
        with np.load(checkpoint_path) as checkpoint:
            completed_steps = int(checkpoint["completed_steps"])
            state = np.asarray(checkpoint["final_state"], dtype=np.float64)
            energy = np.asarray(checkpoint["energy"], dtype=np.float64)
        if state.shape != (int(config.Nv), int(config.Nx)):
            raise ValueError(f"Checkpoint state shape mismatch: {checkpoint_path}")
        if energy.shape != (completed_steps + 1,):
            raise ValueError(f"Checkpoint energy shape mismatch: {checkpoint_path}")
        print(
            f"[joint-grid] resuming {case_name} at "
            f"t={completed_steps * float(config.dt):g}",
            flush=True,
        )
    else:
        completed_steps = 0
        state = np.asarray(f0, dtype=np.float64)
        energy = np.empty((0,), dtype=np.float64)

    snapshot_steps = np.asarray(
        [int(round(float(value) / float(config.dt))) for value in snapshot_times],
        dtype=np.int64,
    )
    if np.any(snapshot_steps < 0) or np.any(snapshot_steps > total_steps):
        raise ValueError("snapshot times must lie inside the simulation interval")

    while completed_steps < total_steps:
        segment_start = completed_steps
        segment_stop = min(total_steps, segment_start + checkpoint_steps)
        local_snapshot_indices = tuple(
            idx
            for idx, step in enumerate(snapshot_steps)
            if segment_start <= int(step) <= segment_stop
            and not (progress_dir / f"snapshot_{idx:03d}.npy").exists()
        )
        local_snapshot_times = tuple(
            (int(snapshot_steps[idx]) - segment_start) * float(config.dt)
            for idx in local_snapshot_indices
        )
        segment_config = PhysicalGridVlasovPoissonConfig(
            Nx=int(config.Nx),
            Nv=int(config.Nv),
            Lx=float(config.Lx),
            vmin=float(config.vmin),
            vmax=float(config.vmax),
            dt=float(config.dt),
            T=(segment_stop - segment_start) * float(config.dt),
            poisson_sign=float(config.poisson_sign),
            snapshot_times=local_snapshot_times,
        )
        print(
            f"[joint-grid] {case_name}: "
            f"t={segment_start * float(config.dt):g}:"
            f"{segment_stop * float(config.dt):g}",
            flush=True,
        )
        raw = run_semilagrangian_vlasov_poisson(
            segment_config,
            jnp.asarray(state, dtype=jnp.float64),
            return_final_state=True,
        )
        segment_energy = np.asarray(raw["energy"], dtype=np.float64)
        energy = (
            segment_energy
            if segment_start == 0
            else np.concatenate([energy, segment_energy[1:]])
        )
        for local_idx, snapshot_idx in enumerate(local_snapshot_indices):
            _atomic_save_npy(
                progress_dir / f"snapshot_{snapshot_idx:03d}.npy",
                np.asarray(raw["snapshot_f"][local_idx], dtype=np.float64),
            )
        state = np.asarray(raw["final_state"], dtype=np.float64)
        completed_steps = segment_stop
        _atomic_savez(
            checkpoint_path,
            completed_steps=np.asarray(completed_steps, dtype=np.int64),
            final_state=state,
            energy=energy,
        )

    missing_snapshots = tuple(
        idx
        for idx in range(len(snapshot_times))
        if not (progress_dir / f"snapshot_{idx:03d}.npy").exists()
    )
    if missing_snapshots:
        raise RuntimeError(f"Missing completed snapshots: {missing_snapshots}")
    snapshots = np.stack(
        [
            np.load(progress_dir / f"snapshot_{idx:03d}.npy")
            for idx in range(len(snapshot_times))
        ],
        axis=0,
    )
    return {
        "snapshot_f": np.asarray(snapshots, dtype=np.float64),
        "times": np.linspace(
            0.0,
            total_steps * float(config.dt),
            total_steps + 1,
            dtype=np.float64,
        ),
        "energy": np.asarray(energy, dtype=np.float64),
    }


def _simulate_pair(args: argparse.Namespace) -> None:
    print_jax_runtime_summary(jax, context="joint physical-grid simulation")
    pair_dir = args.outdir.resolve()
    pair_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = pair_dir / "cases"
    cases_dir.mkdir(exist_ok=True)
    snapshot_times = _parse_float_tuple(args.snapshot_times)
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
    metadata = _simulation_metadata(
        nx=config.Nx,
        nv=config.Nv,
        args=args,
        snapshot_times=snapshot_times,
    )
    metadata_path = pair_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing != metadata:
            raise ValueError(
                f"Existing pair metadata does not match request: {pair_dir}"
            )
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    equilibrium = np.asarray(
        normalize_density_on_grid(
            gaussian_pdf(config.v, mean=0.0, sigma=1.0),
            config.v,
        ),
        dtype=np.float64,
    )
    perturbations = _case_perturbations(
        config,
        linear_eps=float(args.linear_eps),
        linear_modes=_parse_float_tuple(args.linear_modes),
        linear_seed=int(args.linear_seed),
        weak_eps=float(args.weak_eps),
        strong_eps=float(args.strong_eps),
        nonlinear_k0=float(args.nonlinear_k0),
    )
    selected_cases = _parse_case_names(args.cases, tuple(perturbations))
    for case_name in selected_cases:
        perturbation = perturbations[case_name]
        case_path = cases_dir / f"{case_name}.npz"
        if case_path.exists():
            print(f"[joint-grid] reusing completed case {case_path}", flush=True)
            continue
        print(
            f"[joint-grid] running Nx={config.Nx} Nv={config.Nv}: {case_name}",
            flush=True,
        )
        f0 = equilibrium[:, None] * (
            1.0 + jnp.asarray(perturbation, dtype=jnp.float64)[None, :]
        )
        raw = _run_case_segmented(
            config=config,
            f0=f0,
            case_name=case_name,
            cases_dir=cases_dir,
            snapshot_times=snapshot_times,
            checkpoint_interval_time=float(args.checkpoint_interval_time),
        )
        _atomic_savez(
            case_path,
            snapshot_f=np.asarray(raw["snapshot_f"], dtype=np.float64),
            times=np.asarray(raw["times"], dtype=np.float64),
            energy=np.asarray(raw["energy"], dtype=np.float64),
        )
        shutil.rmtree(cases_dir / f".{case_name}.progress")
        print(f"[joint-grid] saved {case_path}", flush=True)
    expected_paths = tuple(cases_dir / f"{name}.npz" for name in perturbations)
    complete_path = pair_dir / "COMPLETE"
    if all(path.is_file() for path in expected_paths):
        complete_path.write_text("complete\n")
    elif complete_path.exists():
        complete_path.unlink()


def _load_pair(
    root: Path,
    *,
    nx: int,
    nv: int,
    case_name: str,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    pair_dir = _pair_directory(root, nx, nv)
    metadata = json.loads((pair_dir / "metadata.json").read_text())
    if int(metadata["Nx"]) != int(nx) or int(metadata["Nv"]) != int(nv):
        raise ValueError(f"Pair metadata mismatch: {pair_dir}")
    with np.load(pair_dir / "cases" / f"{case_name}.npz") as payload:
        arrays = {
            "snapshot_f": np.asarray(payload["snapshot_f"], dtype=np.float64),
            "times": np.asarray(payload["times"], dtype=np.float64),
            "energy": np.asarray(payload["energy"], dtype=np.float64),
        }
    return metadata, arrays


def _project_snapshots(
    snapshots: np.ndarray,
    *,
    source_v: np.ndarray,
    projection_matrix: np.ndarray,
    equilibrium: np.ndarray,
    fourier_modes: int,
) -> np.ndarray:
    perturbation = np.asarray(snapshots, dtype=np.float64) - np.asarray(
        equilibrium,
        dtype=np.float64,
    )[None, :, None]
    time_count, _, nx = perturbation.shape
    projected = np.empty(
        (
            time_count,
            int(projection_matrix.shape[0]),
            min(int(fourier_modes) + 1, nx // 2 + 1),
        ),
        dtype=np.complex128,
    )
    for time_idx in range(time_count):
        moments = projection_matrix @ perturbation[time_idx]
        projected[time_idx] = (
            np.fft.rfft(moments, axis=1)[:, : projected.shape[2]]
            / float(nx)
        )
    return projected


def _projected_change(
    coarse: np.ndarray,
    refined: np.ndarray,
    *,
    cutoffs: Sequence[int],
    domain_length: float,
) -> Tuple[float, float]:
    common_modes = min(int(coarse.shape[2]), int(refined.shape[2]))
    k_arr = (
        2.0
        * math.pi
        * np.arange(common_modes, dtype=np.float64)
        / float(domain_length)
    )
    c_difference_sq = 0.0
    c_reference_sq = 0.0
    q_difference_sq = 0.0
    q_reference_sq = 0.0
    for cutoff in cutoffs:
        c_coarse = coarse[:, : int(cutoff) + 1, :common_modes]
        c_refined = refined[:, : int(cutoff) + 1, :common_modes]
        c_difference_sq += float(np.linalg.norm(c_coarse - c_refined) ** 2)
        c_reference_sq += float(np.linalg.norm(c_refined) ** 2)
        q_coarse = (
            -1j
            * k_arr[1:]
            * math.sqrt(float(cutoff))
            * coarse[:, int(cutoff), 1:common_modes]
        )
        q_refined = (
            -1j
            * k_arr[1:]
            * math.sqrt(float(cutoff))
            * refined[:, int(cutoff), 1:common_modes]
        )
        q_difference_sq += float(np.linalg.norm(q_coarse - q_refined) ** 2)
        q_reference_sq += float(np.linalg.norm(q_refined) ** 2)
    tiny = np.finfo(np.float64).tiny
    return (
        math.sqrt(c_difference_sq / max(c_reference_sq, tiny)),
        math.sqrt(q_difference_sq / max(q_reference_sq, tiny)),
    )


def _joint_distribution_change(
    coarse_snapshots: np.ndarray,
    refined_snapshots: np.ndarray,
    *,
    coarse_equilibrium: np.ndarray,
    refined_equilibrium: np.ndarray,
    coarse_v: np.ndarray,
    refined_v: np.ndarray,
) -> float:
    coarse_delta = np.asarray(coarse_snapshots, dtype=np.float64) - np.asarray(
        coarse_equilibrium,
        dtype=np.float64,
    )[None, :, None]
    refined_delta = np.asarray(refined_snapshots, dtype=np.float64) - np.asarray(
        refined_equilibrium,
        dtype=np.float64,
    )[None, :, None]
    coarse_on_refined_v = _resample_velocity_snapshots(
        coarse_delta,
        source_v=coarse_v,
        target_v=refined_v,
    )
    change, _ = _distribution_successive_x_change(
        coarse_on_refined_v,
        refined_delta,
        equilibrium=np.zeros((refined_delta.shape[1],), dtype=np.float64),
    )
    return change


def _max_energy_change(changes: Dict[str, Optional[float]]) -> float:
    finite = [
        float(value)
        for value in changes.values()
        if value is not None and math.isfinite(float(value))
    ]
    return max(finite) if finite else float("inf")


def _comparison_metrics(
    *,
    coarse: Dict[str, np.ndarray],
    refined: Dict[str, np.ndarray],
    coarse_config: PhysicalGridVlasovPoissonConfig,
    refined_config: PhysicalGridVlasovPoissonConfig,
    coarse_projected: np.ndarray,
    refined_projected: np.ndarray,
    cutoffs: Sequence[int],
    block_edges: Sequence[float],
    comparison_kind: str,
    tolerance: float,
) -> Dict[str, object]:
    np.testing.assert_allclose(
        coarse["times"],
        refined["times"],
        rtol=0.0,
        atol=1e-13,
    )
    energy_changes = _energy_block_changes(
        coarse["energy"],
        refined["energy"],
        times=refined["times"],
        block_edges=block_edges,
    )
    coarse_equilibrium = np.asarray(
        normalize_density_on_grid(
            gaussian_pdf(coarse_config.v, mean=0.0, sigma=1.0),
            coarse_config.v,
        ),
        dtype=np.float64,
    )
    refined_equilibrium = np.asarray(
        normalize_density_on_grid(
            gaussian_pdf(refined_config.v, mean=0.0, sigma=1.0),
            refined_config.v,
        ),
        dtype=np.float64,
    )
    if comparison_kind == "x":
        distribution_change, _ = _distribution_successive_x_change(
            coarse["snapshot_f"],
            refined["snapshot_f"],
            equilibrium=coarse_equilibrium,
        )
    elif comparison_kind == "v":
        distribution_change, _ = _distribution_successive_change(
            coarse["snapshot_f"],
            refined["snapshot_f"],
            coarse_equilibrium=coarse_equilibrium,
            refined_equilibrium=refined_equilibrium,
            coarse_v=np.asarray(coarse_config.v, dtype=np.float64),
            refined_v=np.asarray(refined_config.v, dtype=np.float64),
        )
    elif comparison_kind == "joint":
        distribution_change = _joint_distribution_change(
            coarse["snapshot_f"],
            refined["snapshot_f"],
            coarse_equilibrium=coarse_equilibrium,
            refined_equilibrium=refined_equilibrium,
            coarse_v=np.asarray(coarse_config.v, dtype=np.float64),
            refined_v=np.asarray(refined_config.v, dtype=np.float64),
        )
    else:
        raise ValueError(f"Unsupported comparison kind: {comparison_kind}")
    c_change, q_change = _projected_change(
        coarse_projected,
        refined_projected,
        cutoffs=cutoffs,
        domain_length=float(refined_config.Lx),
    )
    max_energy = _max_energy_change(energy_changes)
    training_relevant_max = max(max_energy, c_change, q_change)
    full_phase_space_max = max(training_relevant_max, distribution_change)
    return {
        **energy_changes,
        "max_energy_block_refinement_change": max_energy,
        "projected_C0_through_N_refinement_change": c_change,
        "projected_qN_refinement_change": q_change,
        "phase_space_distribution_refinement_change": distribution_change,
        "training_relevant_max_change": training_relevant_max,
        "full_phase_space_max_change": full_phase_space_max,
        "passes_training_relevant_one_percent": bool(
            training_relevant_max < tolerance
        ),
        "passes_full_phase_space_one_percent": bool(
            full_phase_space_max < tolerance
        ),
    }


def _config_from_metadata(metadata: Dict[str, object]) -> PhysicalGridVlasovPoissonConfig:
    return PhysicalGridVlasovPoissonConfig(
        Nx=int(metadata["Nx"]),
        Nv=int(metadata["Nv"]),
        Lx=float(metadata["L"]),
        vmin=float(metadata["vmin"]),
        vmax=float(metadata["vmax"]),
        dt=float(metadata["dt"]),
        T=float(metadata["T_final"]),
        poisson_sign=float(metadata["poisson_sign"]),
        snapshot_times=tuple(float(v) for v in metadata["snapshot_times"]),
    )


def _save_summary_plot(records: Sequence[Dict[str, object]], path: Path) -> None:
    cases = tuple(dict.fromkeys(str(record["case"]) for record in records))
    directions = tuple(
        dict.fromkeys(str(record["comparison"]) for record in records)
    )
    colors = {
        "energy": "#1f4e79",
        "C": "#2a9d8f",
        "q": "#e9c46a",
        "f": "#c44e52",
    }
    fig, axes = plt.subplots(
        len(cases),
        1,
        figsize=(10.5, 2.8 * len(cases)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    width = 0.18
    x = np.arange(len(directions), dtype=np.float64)
    metric_specs = (
        ("energy", "max_energy_block_refinement_change"),
        ("C", "projected_C0_through_N_refinement_change"),
        ("q", "projected_qN_refinement_change"),
        ("f", "phase_space_distribution_refinement_change"),
    )
    representative_records = {
        str(record["comparison"]): record for record in records
    }

    def comparison_label(direction: str) -> str:
        record = representative_records[direction]
        coarse_nx = int(record["coarse_Nx"])
        coarse_nv = int(record["coarse_Nv"])
        refined_nx = int(record["refined_Nx"])
        refined_nv = int(record["refined_Nv"])
        if coarse_nv == refined_nv:
            return (
                rf"$N_x$: {coarse_nx:,}$\rightarrow${refined_nx:,}"
                + "\n"
                + rf"$N_v={coarse_nv:,}$"
            )
        if coarse_nx == refined_nx:
            return (
                rf"$N_v$: {coarse_nv:,}$\rightarrow${refined_nv:,}"
                + "\n"
                + rf"$N_x={coarse_nx:,}$"
            )
        return (
            rf"$(N_x,N_v)$: ({coarse_nx:,},{coarse_nv:,})"
            + "\n"
            + rf"$\rightarrow$ ({refined_nx:,},{refined_nv:,})"
        )

    for axis, case in zip(axes, cases):
        case_records = {
            str(record["comparison"]): record
            for record in records
            if str(record["case"]) == case
        }
        for metric_idx, (label, key) in enumerate(metric_specs):
            values = [float(case_records[direction][key]) for direction in directions]
            axis.bar(
                x + (metric_idx - 1.5) * width,
                np.maximum(values, 1e-16),
                width=width,
                label=label,
                color=colors[label],
            )
        display, _ = _display_case(case)
        axis.set_yscale("log")
        axis.set_title(display, loc="left", fontsize=10)
        axis.set_ylabel("Relative successive-grid change")
        axis.grid(True, axis="y", which="both", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [comparison_label(direction) for direction in directions]
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.91, hspace=0.38)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _analyze(args: argparse.Namespace) -> None:
    print_jax_runtime_summary(jax, context="joint physical-grid analysis")
    root = args.corner_root.resolve()
    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis directory: {outdir}")
    outdir.mkdir(parents=True)
    base_nx = int(args.base_Nx)
    base_nv = int(args.base_Nv)
    refined_nx = int(args.refined_Nx)
    refined_nv = int(args.refined_Nv)
    pairs = {
        "A_base": (base_nx, base_nv),
        "B_refine_x": (refined_nx, base_nv),
        "C_refine_v": (base_nx, refined_nv),
    }
    comparison_specs = [
        ("x_at_base_v", "A_base", "B_refine_x", "x"),
        ("v_at_base_x", "A_base", "C_refine_v", "v"),
    ]
    if bool(args.include_diagonal):
        pairs["D_refine_both"] = (refined_nx, refined_nv)
        comparison_specs.extend(
            [
                ("diagonal", "A_base", "D_refine_both", "joint"),
                ("x_at_refined_v", "C_refine_v", "D_refine_both", "x"),
                ("v_at_refined_x", "B_refine_x", "D_refine_both", "v"),
            ]
        )
    cutoffs = tuple(int(value) for value in _parse_int_tuple(args.cutoffs))
    projection_order = max(cutoffs) + 1
    block_edges = _parse_float_tuple(args.time_block_edges)
    tolerance = float(args.relative_tolerance)
    all_case_names = ("linear_sample00", "weak_eps0p1", "strong_eps0p5")
    case_names = _parse_case_names(args.cases, all_case_names)

    metadata_by_pair = {
        name: json.loads(
            (
                _pair_directory(root, nx, nv) / "metadata.json"
            ).read_text()
        )
        for name, (nx, nv) in pairs.items()
    }
    common_keys = (
        "L",
        "vmin",
        "vmax",
        "dt",
        "T_final",
        "poisson_sign",
        "snapshot_times",
        "linear_eps",
        "linear_modes",
        "linear_seed",
        "weak_eps",
        "strong_eps",
        "nonlinear_k0",
    )
    baseline_metadata = metadata_by_pair["A_base"]
    for name, metadata in metadata_by_pair.items():
        if any(metadata[key] != baseline_metadata[key] for key in common_keys):
            raise ValueError(f"Physical-grid pair {name} changes non-grid inputs")

    projection_matrices: Dict[int, np.ndarray] = {}
    for nv in sorted({int(nv) for _, nv in pairs.values()}):
        config = next(
            _config_from_metadata(metadata_by_pair[name])
            for name, (_, pair_nv) in pairs.items()
            if int(pair_nv) == nv
        )
        projection_matrices[nv] = np.asarray(
            build_cubic_spline_hermite_projection_matrix(
                config.v,
                projection_order,
                int(args.projection_quadrature_Nv),
                vth=1.0,
            ),
            dtype=np.float64,
        )

    records = []
    summary: Dict[str, Dict[str, Dict[str, object]]] = {}
    for case_name in case_names:
        arrays_by_pair: Dict[str, Dict[str, np.ndarray]] = {}
        projected_by_pair: Dict[str, np.ndarray] = {}
        configs_by_pair = {
            name: _config_from_metadata(metadata)
            for name, metadata in metadata_by_pair.items()
        }
        for name, (nx, nv) in pairs.items():
            _, arrays = _load_pair(root, nx=nx, nv=nv, case_name=case_name)
            arrays_by_pair[name] = arrays
            equilibrium = np.asarray(
                normalize_density_on_grid(
                    gaussian_pdf(configs_by_pair[name].v, mean=0.0, sigma=1.0),
                    configs_by_pair[name].v,
                ),
                dtype=np.float64,
            )
            projected_by_pair[name] = _project_snapshots(
                arrays["snapshot_f"],
                source_v=np.asarray(configs_by_pair[name].v, dtype=np.float64),
                projection_matrix=projection_matrices[nv],
                equilibrium=equilibrium,
                fourier_modes=int(args.fourier_modes),
            )
        summary[case_name] = {}
        for comparison, coarse_name, refined_name, kind in comparison_specs:
            metrics = _comparison_metrics(
                coarse=arrays_by_pair[coarse_name],
                refined=arrays_by_pair[refined_name],
                coarse_config=configs_by_pair[coarse_name],
                refined_config=configs_by_pair[refined_name],
                coarse_projected=projected_by_pair[coarse_name],
                refined_projected=projected_by_pair[refined_name],
                cutoffs=cutoffs,
                block_edges=block_edges,
                comparison_kind=kind,
                tolerance=tolerance,
            )
            row = {
                "case": case_name,
                "comparison": comparison,
                "coarse_Nx": int(configs_by_pair[coarse_name].Nx),
                "coarse_Nv": int(configs_by_pair[coarse_name].Nv),
                "refined_Nx": int(configs_by_pair[refined_name].Nx),
                "refined_Nv": int(configs_by_pair[refined_name].Nv),
                **metrics,
            }
            records.append(row)
            summary[case_name][comparison] = row

    base_comparisons = tuple(
        comparison
        for comparison, _, _, _ in comparison_specs
        if comparison in {"x_at_base_v", "v_at_base_x", "diagonal"}
    )
    base_training_passes = all(
        bool(summary[case][comparison]["passes_training_relevant_one_percent"])
        for case in case_names
        for comparison in base_comparisons
    )
    base_full_passes = all(
        bool(summary[case][comparison]["passes_full_phase_space_one_percent"])
        for case in case_names
        for comparison in base_comparisons
    )
    csv_path = outdir / "physical_grid_2d_convergence.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "diagnostic": "physical_grid_2d_self_convergence",
        "corners": {
            name: {"Nx": nx, "Nv": nv} for name, (nx, nv) in pairs.items()
        },
        "projection_quadrature_Nv": int(args.projection_quadrature_Nv),
        "projection_order": projection_order,
        "fourier_modes": int(args.fourier_modes),
        "cutoffs": list(cutoffs),
        "time_block_edges": list(block_edges),
        "relative_tolerance": tolerance,
        "comparison_summary": summary,
        "recommendation": {
            "base_pair": {"Nx": base_nx, "Nv": base_nv},
            "base_passes_training_relevant_one_percent": base_training_passes,
            "base_passes_full_phase_space_one_percent": base_full_passes,
            "qualification": (
                "Base pair passes all requested training-relevant "
                "one-percent gates."
                if base_training_passes
                else "Base pair fails at least one requested "
                "training-relevant one-percent gate."
            ),
        },
    }
    json_path = outdir / "physical_grid_2d_convergence.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    figure_path = outdir / "physical_grid_2d_convergence.png"
    _save_summary_plot(records, figure_path)
    print(f"Saved joint-grid convergence CSV to {csv_path}")
    print(f"Saved joint-grid convergence JSON to {json_path}")
    print(f"Saved joint-grid convergence figure to {figure_path}")
    print(
        "[joint-grid] base pair "
        f"Nx={base_nx} Nv={base_nv}: training_gate="
        f"{int(base_training_passes)} full_phase_space_gate="
        f"{int(base_full_passes)}"
    )


def _parse_int_tuple(text: str) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--outdir", type=Path, required=True)
    simulate.add_argument("--teacher-Nx", type=int, required=True)
    simulate.add_argument("--teacher-Nv", type=int, required=True)
    simulate.add_argument("--teacher-L", type=float, default=4.0 * math.pi)
    simulate.add_argument("--teacher-vmin", type=float, default=-8.0)
    simulate.add_argument("--teacher-vmax", type=float, default=8.0)
    simulate.add_argument("--teacher-dt", type=float, default=0.01)
    simulate.add_argument("--T-final", type=float, default=120.0)
    simulate.add_argument(
        "--snapshot-times",
        type=str,
        default="0,20,40,60,80,100,120",
    )
    simulate.add_argument("--linear-eps", type=float, default=0.01)
    simulate.add_argument("--linear-modes", type=str, default="0.5,1.0,1.5,2.0")
    simulate.add_argument("--linear-seed", type=int, default=0)
    simulate.add_argument("--weak-eps", type=float, default=0.1)
    simulate.add_argument("--strong-eps", type=float, default=0.5)
    simulate.add_argument("--nonlinear-k0", type=float, default=0.5)
    simulate.add_argument("--poisson-sign", type=float, default=1.0)
    simulate.add_argument(
        "--cases",
        type=str,
        default="all",
        help="Comma-separated case IDs or all.",
    )
    simulate.add_argument(
        "--checkpoint-interval-time",
        type=float,
        default=10.0,
        help="Physical-time interval between resumable checkpoints.",
    )

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--corner-root", type=Path, required=True)
    analyze.add_argument("--outdir", type=Path, required=True)
    analyze.add_argument("--base-Nx", type=int, default=512)
    analyze.add_argument("--base-Nv", type=int, default=8192)
    analyze.add_argument("--refined-Nx", type=int, default=1024)
    analyze.add_argument("--refined-Nv", type=int, default=16384)
    analyze.add_argument("--projection-quadrature-Nv", type=int, default=4096)
    analyze.add_argument("--fourier-modes", type=int, default=6)
    analyze.add_argument("--cutoffs", type=str, default="6,7,12,20,36,64")
    analyze.add_argument(
        "--time-block-edges",
        type=str,
        default="0,60,80,100,120",
    )
    analyze.add_argument("--relative-tolerance", type=float, default=0.01)
    analyze.add_argument(
        "--cases",
        type=str,
        default="all",
        help="Comma-separated case IDs or all.",
    )
    analyze.add_argument(
        "--include-diagonal",
        action="store_true",
        help="Also require the jointly refined fourth corner.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.command == "simulate":
        _simulate_pair(args)
    elif args.command == "analyze":
        _analyze(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
