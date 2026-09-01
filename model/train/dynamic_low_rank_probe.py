"""Probe high-velocity-resolution macro-micro rank truncation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model.train.interface_flux_data import load_ic_manifest
from vpml.dynamic_low_rank import (
    compress_macro_micro_state,
    positive_density_preserving_projection,
)
from vpml.jax_runtime import print_jax_runtime_summary
from vpml.metrics.field_error import FieldErrorConfig, SelfGeneratedFieldErrorMetric
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    build_physical_grid_ops,
    compute_electric_field_from_distribution,
    semilagrangian_vlasov_poisson_step,
)


def _electric_energy(field: np.ndarray, dx: float) -> np.ndarray:
    return 0.5 * float(dx) * np.sum(np.square(field), axis=-1, dtype=np.float64)


def _project_snapshot(
    snapshot: np.ndarray,
    source_v: np.ndarray,
    config: PhysicalGridVlasovPoissonConfig,
) -> jax.Array:
    source = np.asarray(snapshot, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != source_v.size:
        raise ValueError("snapshot must have shape (source_Nv, source_Nx)")
    if source.shape[1] % int(config.Nx) != 0:
        raise ValueError("source spatial grid must be divisible by probe Nx")
    restricted = source[:, :: source.shape[1] // int(config.Nx)]
    target_v = np.asarray(config.v)
    projected = np.empty((int(config.Nv), int(config.Nx)), dtype=np.float64)
    for spatial_index in range(int(config.Nx)):
        projected[:, spatial_index] = np.interp(
            target_v,
            source_v,
            restricted[:, spatial_index],
        )
    return positive_density_preserving_projection(jnp.asarray(projected), config.v)


def _teacher_field_window(
    payload: np.lib.npyio.NpzFile,
    *,
    start_time: float,
    horizon: int,
    cadence_steps: int,
    nx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(payload["E_hat_hist_times"], dtype=np.float64)
    matches = np.flatnonzero(np.isclose(times, start_time, rtol=0.0, atol=1e-12))
    if matches.size != 1:
        raise ValueError(f"Teacher field history has no unique t={start_time:g}")
    indices = int(matches[0]) + int(cadence_steps) * np.arange(int(horizon) + 1)
    if indices[-1] >= times.size:
        raise ValueError("Requested teacher field window exceeds the cached history")
    source_hat = np.asarray(payload["E_hat_hist"][indices], dtype=np.complex128)
    source_nx = 2 * (source_hat.shape[-1] - 1)
    if source_nx % int(nx) != 0:
        raise ValueError("teacher Nx must be divisible by probe Nx")
    source_field = np.fft.irfft(source_hat, n=source_nx, axis=-1)
    return times[indices], source_field[:, :: source_nx // int(nx)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--case-ids",
        default="linear_landau_ic09,nonlinear_landau_strong_ic12",
    )
    parser.add_argument("--start-times", default="100,60")
    parser.add_argument("--ranks", default="16,32,48")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--cadence-steps", type=int, default=10)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--high-nv", type=int, default=1024)
    parser.add_argument("--coarse-nv", type=int, default=512)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--rank-projection-schedule",
        choices=("repeated", "initial_only", "both"),
        default="repeated",
        help="Apply rank projection after every macro step, only initially, or both.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print_jax_runtime_summary(jax, context="high-resolution macro-micro probe")
    case_ids = [value.strip() for value in args.case_ids.split(",") if value.strip()]
    start_times = [float(value) for value in args.start_times.split(",") if value.strip()]
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    if len(case_ids) != len(start_times):
        raise ValueError("case-ids and start-times must contain the same number of entries")
    if not ranks or min(ranks) <= 0 or args.horizon <= 0 or args.cadence_steps <= 0:
        raise ValueError("ranks, horizon, and cadence-steps must be positive")
    if args.high_nv <= args.coarse_nv:
        raise ValueError("high-nv must exceed coarse-nv")
    outdir = Path(args.outdir)
    if outdir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {outdir}")
    outdir.mkdir(parents=True)

    metadata = json.loads((args.reference_cache / "metadata.json").read_text())
    teacher = metadata["configuration"]
    manifest = load_ic_manifest(args.reference_cache / "ic_manifest.json")
    cases = {str(case["case_id"]): case for case in manifest["cases"]}
    dt = float(teacher["teacher_dt"])
    macro_dt = dt * int(args.cadence_steps)

    def make_config(nv: int) -> PhysicalGridVlasovPoissonConfig:
        return PhysicalGridVlasovPoissonConfig(
            Nx=int(args.nx),
            Nv=int(nv),
            Lx=float(teacher["teacher_L"]),
            vmin=float(teacher["teacher_vmin"]),
            vmax=float(teacher["teacher_vmax"]),
            dt=dt,
            T=float(args.horizon) * macro_dt,
            poisson_sign=float(teacher["teacher_poisson_sign"]),
        )

    high_config = make_config(args.high_nv)
    coarse_config = make_config(args.coarse_nv)
    high_ops = build_physical_grid_ops(high_config)
    coarse_ops = build_physical_grid_ops(coarse_config)

    def advance_macro(config, ops, state):
        def body(_, current):
            updated, _ = semilagrangian_vlasov_poisson_step(config, current, ops=ops)
            return updated

        return jax.lax.fori_loop(0, int(args.cadence_steps), body, state)

    def compile_baseline(config, ops):
        @jax.jit
        def rollout(initial):
            def body(state, _):
                updated = advance_macro(config, ops, state)
                return updated, updated

            _, states = jax.lax.scan(body, initial, xs=None, length=int(args.horizon))
            return states

        return rollout

    high_rollout = compile_baseline(high_config, high_ops)
    coarse_rollout = compile_baseline(coarse_config, coarse_ops)
    rank_rollouts = {}
    for rank in ranks if args.rank_projection_schedule in ("repeated", "both") else ():
        def rollout(initial, *, retained_rank=rank):
            def body(carry, _):
                state, key = carry
                advanced = advance_macro(high_config, high_ops, state)
                advanced = positive_density_preserving_projection(
                    advanced,
                    high_config.v,
                )
                key, compression_key = jax.random.split(key)
                compressed, diagnostics = compress_macro_micro_state(
                    advanced,
                    high_config.v,
                    retained_rank,
                    compression_key,
                    oversample=int(args.oversample),
                    power_iterations=int(args.power_iterations),
                )
                return (compressed, key), (compressed, diagnostics)

            (_, _), outputs = jax.lax.scan(
                body,
                (initial, jax.random.PRNGKey(int(args.seed) + retained_rank)),
                xs=None,
                length=int(args.horizon),
            )
            return outputs

        rank_rollouts[rank] = jax.jit(rollout)

    records = []
    traces = []
    for case_id, start_time in zip(case_ids, start_times):
        if case_id not in cases:
            raise ValueError(f"Unknown probe case: {case_id}")
        snapshot_path = args.reference_cache / "snapshots" / f"{case_id}.npz"
        with np.load(snapshot_path, mmap_mode="r") as payload:
            snapshot_times = np.asarray(payload["snapshot_times"], dtype=np.float64)
            matches = np.flatnonzero(
                np.isclose(snapshot_times, start_time, rtol=0.0, atol=1e-12)
            )
            if matches.size != 1:
                raise ValueError(
                    f"{case_id} requires a cached snapshot at t={start_time:g}; "
                    f"available={snapshot_times.tolist()}"
                )
            snapshot = np.asarray(payload["snapshot_f"][int(matches[0])], dtype=np.float64)
            source_v = np.asarray(payload["v"], dtype=np.float64)
            times, teacher_field = _teacher_field_window(
                payload,
                start_time=start_time,
                horizon=int(args.horizon),
                cadence_steps=int(args.cadence_steps),
                nx=int(args.nx),
            )

        initial_high = _project_snapshot(snapshot, source_v, high_config)
        initial_coarse = _project_snapshot(snapshot, source_v, coarse_config)
        high_states = np.asarray(high_rollout(initial_high))
        coarse_states = np.asarray(coarse_rollout(initial_coarse))
        trajectories = {
            "coarse_nv512": np.concatenate((np.asarray(initial_coarse)[None], coarse_states)),
            "full_nv1024": np.concatenate((np.asarray(initial_high)[None], high_states)),
        }
        diagnostics_by_rank: Dict[str, Dict[str, float]] = {}
        for rank, run_rank in rank_rollouts.items():
            states, diagnostics = run_rank(initial_high)
            trajectories[f"rank_{rank}"] = np.concatenate(
                (np.asarray(initial_high)[None], np.asarray(states))
            )
            diagnostics_by_rank[str(rank)] = {
                key: (
                    float(np.min(np.asarray(value)))
                    if key == "minimum_distribution"
                    else float(np.max(np.asarray(value)))
                )
                for key, value in diagnostics.items()
            }
        if args.rank_projection_schedule in ("initial_only", "both"):
            for rank in ranks:
                compression_key = jax.random.PRNGKey(int(args.seed) + rank)
                compressed_initial, diagnostics = compress_macro_micro_state(
                    initial_high,
                    high_config.v,
                    rank,
                    compression_key,
                    oversample=int(args.oversample),
                    power_iterations=int(args.power_iterations),
                )
                states = high_rollout(compressed_initial)
                label = f"initial_rank_{rank}"
                trajectories[label] = np.concatenate(
                    (
                        np.asarray(compressed_initial)[None],
                        np.asarray(states),
                    )
                )
                diagnostics_by_rank[label] = {
                    key: float(np.asarray(value))
                    for key, value in diagnostics.items()
                }

        field_rows = {}
        for label, trajectory in trajectories.items():
            config = coarse_config if label == "coarse_nv512" else high_config
            ops = coarse_ops if label == "coarse_nv512" else high_ops
            field_rows[label] = np.asarray(
                jax.jit(
                    jax.vmap(
                        lambda state: compute_electric_field_from_distribution(
                            state,
                            config,
                            ops=ops,
                        )
                    )
                )(jnp.asarray(trajectory))
            )

        metric = SelfGeneratedFieldErrorMetric(
            FieldErrorConfig(final_time=float(times[-1]))
        )
        k_arr = np.asarray(high_config.k_arr)
        case_metrics = {}
        for label, field in field_rows.items():
            epsilon = metric.evaluate_fourier(
                times,
                np.fft.rfft(field, axis=-1),
                k_arr,
                times,
                np.fft.rfft(teacher_field, axis=-1),
                k_arr,
            ).epsilon_E
            case_metrics[label] = {"epsilon_E": float(epsilon)}
        record = {
            "case_id": case_id,
            "regime": str(cases[case_id]["regime"]),
            "start_time": start_time,
            "end_time": float(times[-1]),
            "metrics": case_metrics,
            "compression_diagnostics": diagnostics_by_rank,
        }
        records.append(record)
        traces.append(
            (
                case_id,
                times,
                _electric_energy(teacher_field, high_config.dx),
                {
                    label: _electric_energy(field, high_config.dx)
                    for label, field in field_rows.items()
                },
            )
        )
        print(json.dumps(record, sort_keys=True), flush=True)

    report = {
        "method": "high_nv_macro_micro_rank_projection_probe",
        "warning": (
            "This validates conservative truncation at high velocity resolution; "
            "it is not yet a factor-only projector-splitting DLR integrator."
        ),
        "configuration": {
            "case_ids": case_ids,
            "start_times": start_times,
            "ranks": ranks,
            "horizon": int(args.horizon),
            "macro_dt": macro_dt,
            "nx": int(args.nx),
            "high_nv": int(args.high_nv),
            "coarse_nv": int(args.coarse_nv),
            "rank_projection_schedule": str(args.rank_projection_schedule),
        },
        "cases": records,
    }
    (outdir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(
        len(traces),
        1,
        figsize=(10.0, 3.2 * len(traces)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = {
        "coarse_nv512": "#6b7280",
        "full_nv1024": "#111827",
        "rank_16": "#16a34a",
        "rank_32": "#7c3aed",
        "rank_48": "#dc2626",
    }
    for axis, (case_id, times, target_energy, candidate_energy) in zip(axes, traces):
        axis.semilogy(times, np.maximum(target_energy, 1e-30), color="#2563eb", label="teacher")
        for label, energy in candidate_energy.items():
            axis.semilogy(
                times,
                np.maximum(energy, 1e-30),
                label=label,
                color=colors.get(label),
            )
        axis.set_title(case_id)
        axis.set_ylabel("E-field energy")
        axis.grid(alpha=0.22)
    axes[0].legend(ncol=min(len(ranks) + 3, 6), fontsize=8)
    axes[-1].set_xlabel("t")
    fig.savefig(outdir / "metric1_probe.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
