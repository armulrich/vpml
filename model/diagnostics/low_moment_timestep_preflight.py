"""Compare low-moment SSPRK3 timesteps with an exact kinetic closure history.

This is a bounded numerical preflight.  It never trains a model and writes to
a new diagnostics directory without modifying reference caches or prior runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from model.train.interface_flux_data import load_ic_manifest
from model.train.low_moment_closure import (
    _central_heat_flux_numpy,
    _primitive_numpy,
    low_hermite_coefficients_to_conservative,
)
from vpml.low_moment import _dealias, limit_low_moment_state, low_moment_rhs


REGIMES = (
    "linear_landau",
    "nonlinear_landau_weak",
    "nonlinear_landau_strong",
)


def _interpolate_coefficients(
    history: np.ndarray,
    times: np.ndarray,
    *,
    reference_dt: float,
) -> np.ndarray:
    indices = np.asarray(times, dtype=np.float64) / float(reference_dt)
    left = np.floor(indices + 1e-10).astype(np.int64)
    fraction = indices - left
    right = np.minimum(left + 1, int(history.shape[0]) - 1)
    if np.any(left < 0) or np.any(right >= int(history.shape[0])):
        raise ValueError("Requested time lies outside the reference trajectory")
    left_values = np.asarray(history[left, :4], dtype=np.complex128)
    right_values = np.asarray(history[right, :4], dtype=np.complex128)
    return left_values + fraction[:, None, None] * (right_values - left_values)


def _teacher_state_and_closure(
    history: np.ndarray,
    times: np.ndarray,
    *,
    reference_dt: float,
    source_nx: int,
    target_nx: int,
    domain_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = _interpolate_coefficients(
        history, times, reference_dt=reference_dt
    )
    state = low_hermite_coefficients_to_conservative(
        coefficients,
        source_nx=source_nx,
        target_nx=target_nx,
        dtype=np.float64,
    )
    heat_flux = _central_heat_flux_numpy(
        coefficients,
        state,
        source_nx=source_nx,
        target_nx=target_nx,
    )
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(
        int(target_nx), d=float(domain_length) / int(target_nx)
    )
    gradient = np.fft.irfft(
        1j * k_arr * np.fft.rfft(heat_flux, axis=-1),
        n=int(target_nx),
        axis=-1,
    )
    return state, gradient


def _make_oracle_step(k_arr: np.ndarray):
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)

    def physical_stage(reference, value):
        return limit_low_moment_state(
            _dealias(value),
            reference_state=reference,
            density_floor=1e-4,
            pressure_floor=1e-4,
        )

    @jax.jit
    def step(state, closure_at_start, closure_at_end, closure_at_half, dt):
        dt_value = jnp.asarray(dt, dtype=state.dtype)
        stage1 = physical_stage(
            state,
            state
            + dt_value
            * low_moment_rhs(
                state,
                closure_at_start,
                k_jax,
                density_floor=1e-4,
                pressure_floor=1e-4,
            ),
        )
        stage2_candidate = stage1 + dt_value * low_moment_rhs(
            stage1,
            closure_at_end,
            k_jax,
            density_floor=1e-4,
            pressure_floor=1e-4,
        )
        stage2 = physical_stage(
            state, 0.75 * state + 0.25 * stage2_candidate
        )
        stage3_candidate = stage2 + dt_value * low_moment_rhs(
            stage2,
            closure_at_half,
            k_jax,
            density_floor=1e-4,
            pressure_floor=1e-4,
        )
        return physical_stage(
            state,
            (1.0 / 3.0) * state + (2.0 / 3.0) * stage3_candidate,
        )

    return step


def _field_energy(fields: np.ndarray, domain_length: float) -> np.ndarray:
    field = np.asarray(fields)[..., 3, :]
    return 0.5 * (float(domain_length) / int(field.shape[-1])) * np.sum(
        field * field, axis=-1, dtype=np.float64
    )


def _relative_l2(predicted: np.ndarray, target: np.ndarray) -> float:
    numerator = np.sum(np.square(predicted - target), dtype=np.float64)
    denominator = np.sum(np.square(target), dtype=np.float64)
    return float(math.sqrt(numerator / max(denominator, 1e-60)))


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _run_case(
    history: np.ndarray,
    *,
    start_time: float,
    duration: float,
    dt: float,
    reference_dt: float,
    source_nx: int,
    target_nx: int,
    domain_length: float,
    oracle_step,
) -> Mapping[str, object]:
    step_count = int(round(float(duration) / float(dt)))
    if not math.isclose(step_count * float(dt), float(duration), abs_tol=1e-10):
        raise ValueError("duration must be an integer multiple of dt")
    starts = float(start_time) + float(dt) * np.arange(step_count)
    ends = starts + float(dt)
    halves = starts + 0.5 * float(dt)
    all_times = np.concatenate(([float(start_time)], ends))
    teacher_states, _ = _teacher_state_and_closure(
        history,
        all_times,
        reference_dt=reference_dt,
        source_nx=source_nx,
        target_nx=target_nx,
        domain_length=domain_length,
    )
    _, closure_start = _teacher_state_and_closure(
        history,
        starts,
        reference_dt=reference_dt,
        source_nx=source_nx,
        target_nx=target_nx,
        domain_length=domain_length,
    )
    _, closure_end = _teacher_state_and_closure(
        history,
        ends,
        reference_dt=reference_dt,
        source_nx=source_nx,
        target_nx=target_nx,
        domain_length=domain_length,
    )
    _, closure_half = _teacher_state_and_closure(
        history,
        halves,
        reference_dt=reference_dt,
        source_nx=source_nx,
        target_nx=target_nx,
        domain_length=domain_length,
    )
    state = jnp.asarray(teacher_states[0:1], dtype=jnp.float32)
    predicted = [np.asarray(state[0], dtype=np.float64)]
    for index in range(step_count):
        state = oracle_step(
            state,
            jnp.asarray(closure_start[index : index + 1], dtype=jnp.float32),
            jnp.asarray(closure_end[index : index + 1], dtype=jnp.float32),
            jnp.asarray(closure_half[index : index + 1], dtype=jnp.float32),
            jnp.asarray(dt, dtype=jnp.float32),
        )
        predicted.append(np.asarray(state[0], dtype=np.float64))
    predicted_states = np.stack(predicted)
    predicted_fields = _primitive_numpy(predicted_states, domain_length)
    teacher_fields = _primitive_numpy(teacher_states, domain_length)
    predicted_energy = _field_energy(predicted_fields, domain_length)
    teacher_energy = _field_energy(teacher_fields, domain_length)
    rho = 1.0 + predicted_states[:, 0]
    momentum = predicted_states[:, 1]
    pressure = 1.0 + predicted_states[:, 2] - momentum * momentum / np.maximum(
        rho, 1e-8
    )
    finite = bool(np.all(np.isfinite(predicted_states)))
    field_relative_l2 = _relative_l2(
            predicted_fields[:, 3], teacher_fields[:, 3]
        )
    primitive_relative_l2 = _relative_l2(predicted_fields, teacher_fields)
    log_energy_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.log(np.maximum(predicted_energy, 1e-60))
                        - np.log(np.maximum(teacher_energy, 1e-60))
                    )
                )
            )
        )
    result = {
        "finite": finite,
        "field_relative_l2": _finite_or_none(field_relative_l2),
        "primitive_relative_l2": _finite_or_none(primitive_relative_l2),
        "log_energy_rmse": _finite_or_none(log_energy_rmse),
        "minimum_density": _finite_or_none(np.nanmin(rho)),
        "minimum_pressure": _finite_or_none(np.nanmin(pressure)),
    }
    return {
        "metrics": result,
        "times": all_times,
        "fields": predicted_fields,
        "energy": predicted_energy,
    }


def _aggregate(rows: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    material = list(rows)
    keys = ("field_relative_l2", "primitive_relative_l2", "log_energy_rmse")
    values = {
        key: [float(row[key]) for row in material if row[key] is not None]
        for key in keys
    }
    densities = [
        float(row["minimum_density"])
        for row in material
        if row["minimum_density"] is not None
    ]
    pressures = [
        float(row["minimum_pressure"])
        for row in material
        if row["minimum_pressure"] is not None
    ]
    return {
        "case_windows": len(material),
        "finite_case_windows": sum(bool(row["finite"]) for row in material),
        **{
            f"mean_{key}": (float(np.mean(values[key])) if values[key] else None)
            for key in keys
        },
        **{
            f"max_{key}": (float(np.max(values[key])) if values[key] else None)
            for key in keys
        },
        "minimum_density": float(np.min(densities)) if densities else None,
        "minimum_pressure": float(np.min(pressures)) if pressures else None,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--rollout-Nx", type=int, default=128)
    parser.add_argument("--timesteps", type=str, default="0.025,0.05,0.1")
    parser.add_argument("--start-times", type=str, default="0,40,80")
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.outdir}")
    args.outdir.mkdir(parents=True)

    metadata = json.loads((args.reference_cache / "metadata.json").read_text())
    configuration = metadata["configuration"]
    manifest = load_ic_manifest(args.reference_cache / "ic_manifest.json")
    source_nx = int(configuration["teacher_Nx"])
    reference_dt = float(configuration["teacher_dt"])
    domain_length = float(configuration["teacher_L"])
    timesteps = tuple(float(value) for value in args.timesteps.split(","))
    start_times = tuple(float(value) for value in args.start_times.split(","))
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(
        int(args.rollout_Nx), d=domain_length / int(args.rollout_Nx)
    )
    oracle_step = _make_oracle_step(k_arr)
    selected = [case for case in manifest["cases"] if case["split"] == "heldout"]
    selected.sort(key=lambda case: (REGIMES.index(case["regime"]), case["case_id"]))
    trajectories: Dict[tuple[str, float, float], Mapping[str, object]] = {}
    records = []
    for case in selected:
        case_id = str(case["case_id"])
        history = np.load(
            args.reference_cache / "cases" / f"{case_id}.npy", mmap_mode="r"
        )
        for start_time in start_times:
            for dt in timesteps:
                run = _run_case(
                    history,
                    start_time=start_time,
                    duration=args.duration,
                    dt=dt,
                    reference_dt=reference_dt,
                    source_nx=source_nx,
                    target_nx=args.rollout_Nx,
                    domain_length=domain_length,
                    oracle_step=oracle_step,
                )
                trajectories[(case_id, start_time, dt)] = run
                records.append(
                    {
                        "case_id": case_id,
                        "regime": str(case["regime"]),
                        "start_time": start_time,
                        "dt": dt,
                        **run["metrics"],
                    }
                )

    reference_dt_choice = min(timesteps)
    pairwise = []
    for case in selected:
        case_id = str(case["case_id"])
        for start_time in start_times:
            fine = trajectories[(case_id, start_time, reference_dt_choice)]
            for dt in timesteps:
                if dt == reference_dt_choice:
                    continue
                coarse = trajectories[(case_id, start_time, dt)]
                ratio = int(round(dt / reference_dt_choice))
                if not math.isclose(
                    ratio * reference_dt_choice, dt, abs_tol=1e-10
                ):
                    continue
                fine_fields = np.asarray(fine["fields"])[::ratio]
                fine_energy = np.asarray(fine["energy"])[::ratio]
                coarse_fields = np.asarray(coarse["fields"])
                coarse_energy = np.asarray(coarse["energy"])
                pair_finite = bool(
                    np.all(np.isfinite(coarse_fields))
                    and np.all(np.isfinite(fine_fields))
                    and np.all(np.isfinite(coarse_energy))
                    and np.all(np.isfinite(fine_energy))
                )
                pair_field_error = _relative_l2(
                    coarse_fields[:, 3], fine_fields[:, 3]
                )
                pair_energy_error = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                np.log(np.maximum(coarse_energy, 1e-60))
                                - np.log(np.maximum(fine_energy, 1e-60))
                            )
                        )
                    )
                )
                pairwise.append(
                    {
                        "case_id": case_id,
                        "regime": str(case["regime"]),
                        "start_time": start_time,
                        "dt": dt,
                        "reference_dt": reference_dt_choice,
                        "finite": pair_finite,
                        "field_relative_l2": _finite_or_none(pair_field_error),
                        "log_energy_rmse": _finite_or_none(pair_energy_error),
                    }
                )

    aggregate = {}
    for dt in timesteps:
        by_dt = [row for row in records if row["dt"] == dt]
        aggregate[str(dt)] = {
            "all": _aggregate(by_dt),
            "by_regime": {
                regime: _aggregate(
                    row for row in by_dt if row["regime"] == regime
                )
                for regime in REGIMES
            },
        }
    pairwise_aggregate = {}
    for dt in timesteps:
        rows = [row for row in pairwise if row["dt"] == dt]
        if rows:
            finite_rows = [
                row
                for row in rows
                if row["field_relative_l2"] is not None
                and row["log_energy_rmse"] is not None
            ]
            pairwise_aggregate[str(dt)] = {
                "case_windows": len(rows),
                "finite_case_windows": len(finite_rows),
                "mean_field_relative_l2": (
                    float(np.mean([row["field_relative_l2"] for row in finite_rows]))
                    if finite_rows
                    else None
                ),
                "max_field_relative_l2": (
                    float(np.max([row["field_relative_l2"] for row in finite_rows]))
                    if finite_rows
                    else None
                ),
                "mean_log_energy_rmse": (
                    float(np.mean([row["log_energy_rmse"] for row in finite_rows]))
                    if finite_rows
                    else None
                ),
                "max_log_energy_rmse": (
                    float(np.max([row["log_energy_rmse"] for row in finite_rows]))
                    if finite_rows
                    else None
                ),
            }

    payload = {
        "diagnostic": "exact_closure_low_moment_timestep_preflight",
        "reference_cache": str(args.reference_cache.resolve()),
        "manifest_sha256": str(manifest["sha256"]),
        "reference_dt": reference_dt,
        "rollout_Nx": int(args.rollout_Nx),
        "timesteps": list(timesteps),
        "start_times": list(start_times),
        "duration": float(args.duration),
        "heldout_case_count": len(selected),
        "records": records,
        "aggregate": aggregate,
        "pairwise_against_smallest_dt": pairwise_aggregate,
    }
    (args.outdir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"aggregate": aggregate, "pairwise": pairwise_aggregate}, indent=2))


if __name__ == "__main__":
    main()
