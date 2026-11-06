"""Held-out low-moment trajectory metrics and diagnostic figures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model.train.interface_flux_data import IC_SPLIT_HELDOUT, case_shard_paths
from vpml.low_moment import (
    DEFAULT_DENSITY_FLOOR,
    DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    DEFAULT_PRESSURE_FLOOR,
    FIXED_INPUT_SCALING,
    electric_field_from_density,
    encode_explicit_window_history,
    rollout_low_moment_closure,
    warm_spectral_memory,
)
from vpml.metrics import (
    EarlyElectricFieldGrowthMetric,
    EarlyGrowthConfig,
    FieldErrorConfig,
    SelfGeneratedFieldErrorMetric,
)
from vpml.training.low_moment_configuration import (
    CHECKPOINT_SCHEMA,
    REGIMES,
    _checkpoint_regimes,
    _uses_compact_latent,
    _uses_window_memory,
)
from vpml.training.low_moment_data import (
    _case_amplitudes,
    _restrict_rfft,
    low_hermite_coefficients_to_conservative,
)
from vpml.training.low_moment_losses import _rollout_window_memory, _unpack_window_memory
from vpml.training.low_moment_state import _load_checkpoint


def _plot_losses(
    train_loss,
    train_ema_loss,
    train_eval_epochs,
    train_eval_loss,
    val_epochs,
    val_loss,
    path: Path,
    metadata: Optional[Mapping[str, object]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    ax.semilogy(
        np.arange(1, len(train_loss) + 1),
        train_loss,
        color="#8b93a1",
        alpha=0.55,
        label=(
            "changing-anchor batch mean"
            if metadata and metadata.get("deterministic_anchor_cycle", False)
            else "stochastic batch mean"
        ),
    )
    if len(train_ema_loss):
        ax.semilogy(
            np.arange(1, len(train_ema_loss) + 1),
            train_ema_loss,
            color="#526071",
            linewidth=1.8,
            label="batch-loss EMA",
        )
    if len(train_eval_loss):
        ax.semilogy(
            train_eval_epochs,
            train_eval_loss,
            color="#172033",
            marker="o",
            label="fixed training panel",
        )
    if len(val_loss):
        ax.semilogy(
            val_epochs,
            val_loss,
            color="#c44e52",
            marker="o",
            label="all held-out ICs at fixed times",
        )
    if metadata:
        warmup = int(metadata.get("supervised_heat_flux_warmup_epochs", 0))
        if metadata.get("supervised_only", False):
            supervised_epochs = int(metadata.get("supervised_heat_flux_epochs", 0))
            stages = [(supervised_epochs, "supervised closure capacity")]
        else:
            stages = [(warmup, "heat-flux warm-up")] if warmup else []
        stages.extend(
            (int(epochs), f"H={int(horizon)}")
            for horizon, epochs in metadata.get("horizon_curriculum", [])
        )
        left = 0
        for index, (epochs, label) in enumerate(stages):
            right = left + epochs
            if left:
                ax.axvline(left + 0.5, color="#8b93a1", linewidth=0.8, alpha=0.6)
            ax.text(
                0.5 * (left + right) + 0.5,
                0.02,
                label,
                rotation=90 if epochs <= 5 else 0,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#5f6877",
                transform=ax.get_xaxis_transform(),
            )
            left = right
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized trajectory loss")
    ax.set_title("Fixed-horizon low-moment trajectory training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _electric_field_energy(field_hat: np.ndarray, *, nx: int, dx: float) -> np.ndarray:
    """Compute energy in float64 so large finite float32 fields remain diagnosable."""
    field = np.fft.irfft(
        np.asarray(field_hat, dtype=np.complex128), n=int(nx), axis=-1
    )
    with np.errstate(over="ignore", invalid="ignore"):
        return 0.5 * float(dx) * np.sum(field * field, axis=-1, dtype=np.float64)


def _json_float(value: float) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def _interpolate_time_series(
    source_times: np.ndarray, source_values: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    values = np.asarray(source_values)
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, source_times.size - 1)
    left = np.maximum(right - 1, 0)
    denominator = source_times[right] - source_times[left]
    weight = np.divide(
        target_times - source_times[left],
        denominator,
        out=np.zeros_like(target_times),
        where=denominator > 0.0,
    )
    reshape = (target_times.size,) + (1,) * (values.ndim - 1)
    return values[left] + weight.reshape(reshape) * (values[right] - values[left])


def _energy_peak_angular_frequency(times, energy, *, final_time: float = 40.0):
    """Estimate early oscillation frequency from successive energy maxima."""
    times = np.asarray(times, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    indices = np.flatnonzero((times <= float(final_time)) & np.isfinite(energy))
    if indices.size < 3:
        return None
    values = energy[indices]
    maxima = indices[1:-1][
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    ]
    if maxima.size < 3:
        return None
    periods = np.diff(times[maxima])
    periods = periods[periods > 0.0]
    if periods.size < 2:
        return None
    return float(2.0 * math.pi / np.median(periods))


def _evaluate_heldout(
    params,
    grouped,
    manifest,
    coefficient_key: str,
    cache_dir: Path,
    outdir: Path,
    *,
    regimes: Sequence[str] = REGIMES,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    dt: float,
    reference_dt: float,
    width: int,
    memory_steps: int,
    memory_stride: int,
    scan_unroll: int,
    input_scale: np.ndarray,
    heat_flux_gradient_scale: float,
    amplitude_center: float,
    amplitude_scale: float,
    poisson_sign: float,
    chunk_steps: int,
    normalized_heat_flux_bound: float,
    density_floor: float,
    pressure_floor: float,
    closure_history_input: bool = False,
    memory_backend: str = "latent_recurrent",
    input_scaling: str = FIXED_INPUT_SCALING,
    dynamic_amplitude_floor: float = 1e-6,
    allow_uniform_heating: bool = False,
) -> None:
    amplitudes_by_id = _case_amplitudes(manifest)
    initial_states = []
    case_records = []
    for regime in regimes:
        group = grouped[regime]
        histories = tuple(group[coefficient_key])
        for case_index, (case_id, split) in enumerate(
            zip(group["case_ids"], group["case_splits"])
        ):
            if str(split) != IC_SPLIT_HELDOUT:
                continue
            state = low_hermite_coefficients_to_conservative(
                np.asarray(histories[case_index][0, :3, :]),
                source_nx=source_nx,
                target_nx=rollout_nx,
            )
            initial_states.append(state)
            case_records.append((regime, str(case_id), case_index, histories[case_index]))
    if not initial_states:
        raise ValueError(f"No held-out cases for selected regimes: {regimes}")
    state = jnp.asarray(np.stack(initial_states), dtype=jnp.float32)
    amplitude = jnp.asarray(
        [amplitudes_by_id[record[1]] for record in case_records], dtype=jnp.float32
    )
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(
        int(rollout_nx), d=float(domain_length) / float(rollout_nx)
    )
    k_jax = jnp.asarray(k_arr, dtype=jnp.float32)
    # No trajectory exists before t=0. Use the centered equilibrium state as
    # causal padding instead of pretending that the initial perturbation has
    # already persisted for the full memory span.
    initial_history = jnp.zeros(
        (state.shape[0], int(memory_steps), state.shape[1], state.shape[-1]),
        dtype=state.dtype,
    )
    previous_gradient = jnp.zeros((state.shape[0], state.shape[-1]), dtype=state.dtype)
    if _uses_window_memory(memory_backend):
        initial_closure_history = jnp.zeros(
            (state.shape[0], int(memory_steps), state.shape[-1]), dtype=state.dtype
        )
        initial_encoded_history = encode_explicit_window_history(
            params,
            initial_history,
            k_jax,
            input_scale=jnp.asarray(input_scale, dtype=jnp.float32),
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            heat_flux_gradient_history=(
                initial_closure_history if closure_history_input else None
            ),
            poisson_sign=poisson_sign,
            input_scaling=input_scaling,
            dynamic_amplitude_floor=dynamic_amplitude_floor,
        )
        model_memory = (
            initial_history,
            initial_closure_history,
            initial_encoded_history,
            jnp.asarray(0, dtype=jnp.int32),
            previous_gradient,
        )
        if _uses_compact_latent(memory_backend):
            model_memory = model_memory + (None,)
    else:
        hidden, previous_gradient = warm_spectral_memory(
            params,
            initial_history,
            amplitude,
            k_jax,
            width=width,
            input_scale=jnp.asarray(input_scale, dtype=jnp.float32),
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            closure_history_input=closure_history_input,
            return_closure_history=True,
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
        )
        model_memory = (hidden, previous_gradient)
    field_hat_chunks = [
        np.fft.rfft(
            np.asarray(
                electric_field_from_density(state[:, 0], k_jax, poisson_sign=poisson_sign)
            ),
            axis=-1,
        )
    ]
    case_count = len(case_records)
    minimum_density = np.full((case_count,), np.inf, dtype=np.float64)
    minimum_pressure = np.full((case_count,), np.inf, dtype=np.float64)
    density_limiter_hits = np.zeros((case_count,), dtype=np.int64)
    pressure_limiter_hits = np.zeros((case_count,), dtype=np.int64)
    state_point_count = 0

    def run_chunk(current_state, current_memory, length: int):
        if _uses_window_memory(memory_backend):
            window, closure_window, encoded, counter, gradient, compact_latent = (
                _unpack_window_memory(current_memory, memory_backend)
            )
            return _rollout_window_memory(
                params,
                current_state,
                window,
                counter,
                amplitude,
                k_jax,
                memory_backend=memory_backend,
                compact_latent=compact_latent,
                horizon=length,
                dt=dt,
                memory_stride=memory_stride,
                input_scale=jnp.asarray(input_scale, dtype=jnp.float32),
                heat_flux_gradient_scale=heat_flux_gradient_scale,
                amplitude_center=amplitude_center,
                amplitude_scale=amplitude_scale,
                previous_heat_flux_gradient=gradient,
                closure_history_input=closure_history_input,
                encoded_history=encoded,
                heat_flux_gradient_history=closure_window,
                poisson_sign=poisson_sign,
                normalized_heat_flux_bound=normalized_heat_flux_bound,
                density_floor=density_floor,
                pressure_floor=pressure_floor,
                scan_unroll=scan_unroll,
                input_scaling=input_scaling,
                dynamic_amplitude_floor=dynamic_amplitude_floor,
                allow_uniform_heating=allow_uniform_heating,
            )
        current_hidden, current_gradient = current_memory
        states, final_hidden, final_gradient = rollout_low_moment_closure(
            params,
            current_state,
            current_hidden,
            amplitude,
            k_jax,
            horizon=length,
            dt=dt,
            input_scale=jnp.asarray(input_scale, dtype=jnp.float32),
            heat_flux_gradient_scale=heat_flux_gradient_scale,
            amplitude_center=amplitude_center,
            amplitude_scale=amplitude_scale,
            previous_heat_flux_gradient=current_gradient,
            closure_history_input=closure_history_input,
            return_closure_history=True,
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=normalized_heat_flux_bound,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        )
        return states, (final_hidden, final_gradient)

    reference_steps = int(case_records[0][3].shape[0]) - 1
    final_time = reference_steps * float(reference_dt)
    total_steps = int(round(final_time / float(dt)))
    if total_steps <= 0 or not math.isclose(
        total_steps * float(dt), final_time, abs_tol=1e-9
    ):
        raise ValueError("solver dt must divide the cached reference duration")
    completed = 0
    compiled = {}
    while completed < total_steps:
        length = min(int(chunk_steps), total_steps - completed)
        if length not in compiled:
            compiled[length] = jax.jit(
                lambda s, m, n=length: run_chunk(s, m, n)
            )
        states, model_memory = compiled[length](state, model_memory)
        state = states[:, -1]
        states_numpy = np.asarray(states, dtype=np.float64)
        density = 1.0 + states_numpy[:, :, 0]
        momentum = states_numpy[:, :, 1]
        pressure = 1.0 + states_numpy[:, :, 2] - momentum * momentum / density
        minimum_density = np.minimum(
            minimum_density, np.min(density, axis=(1, 2))
        )
        minimum_pressure = np.minimum(
            minimum_pressure, np.min(pressure, axis=(1, 2))
        )
        density_limiter_hits += np.sum(
            density <= 1.01 * float(density_floor), axis=(1, 2)
        )
        pressure_limiter_hits += np.sum(
            pressure <= 1.02 * float(pressure_floor), axis=(1, 2)
        )
        state_point_count += int(length * rollout_nx)
        flat = states.reshape(-1, 3, rollout_nx)
        fields = electric_field_from_density(flat[:, 0], k_jax, poisson_sign=poisson_sign)
        fields = np.asarray(fields).reshape(states.shape[0], length, rollout_nx)
        field_hat_chunks.append(np.fft.rfft(fields, axis=-1).transpose(1, 0, 2))
        completed += length
        print(f"[eval] autonomous held-out rollout: {completed}/{total_steps}")
    learned_hat = np.concatenate(
        [field_hat_chunks[0][None, ...], *field_hat_chunks[1:]], axis=0
    ).transpose(1, 0, 2)
    times = np.arange(total_steps + 1, dtype=np.float64) * float(dt)
    dx = float(domain_length) / float(rollout_nx)
    learned_energy = _electric_field_energy(
        learned_hat, nx=rollout_nx, dx=dx
    )
    growth_metric = EarlyElectricFieldGrowthMetric(
        EarlyGrowthConfig(sample_selector="local_maxima")
    )
    summaries = []
    metric1_traces = []
    eval_root = outdir / "heldout_cases"
    eval_root.mkdir(parents=True, exist_ok=True)
    for row, (regime, case_id, _, _) in enumerate(case_records):
        _, _, snapshot_path = case_shard_paths(cache_dir, case_id)
        with np.load(snapshot_path) as reference:
            hr_times = np.asarray(reference["E_hat_hist_times"], dtype=np.float64)
            hr_hat_full = np.asarray(reference["E_hat_hist"], dtype=np.complex128)
            hr_energy = np.asarray(reference["energy"], dtype=np.float64)
            hr_k = np.asarray(reference["k_arr"], dtype=np.float64)
        restricted_hr_hat = _restrict_rfft(hr_hat_full, source_nx, rollout_nx)
        finite = np.all(np.isfinite(learned_hat[row]), axis=-1) & np.isfinite(
            learned_energy[row]
        )
        finite_count = int(np.argmax(~finite)) if not np.all(finite) else int(finite.size)
        finite_count = max(finite_count, 2)
        reference_energy_max = max(float(np.nanmax(hr_energy)), 1e-30)
        blowup_limit = 1e6 * reference_energy_max
        blowup = np.flatnonzero(learned_energy[row, :finite_count] > blowup_limit)
        failure_index = int(blowup[0]) if blowup.size else finite_count
        bounded_to_final = failure_index == int(times.size)
        failure_time = None if bounded_to_final else float(times[failure_index])
        growth = growth_metric.compare(
            times[:finite_count],
            learned_energy[row, :finite_count],
            hr_times,
            hr_energy,
        )
        field_metric = SelfGeneratedFieldErrorMetric(
            FieldErrorConfig(final_time=times[finite_count - 1])
        )
        field_error = field_metric.evaluate_fourier(
            times[:finite_count],
            np.asarray(learned_hat[row, :finite_count], dtype=np.complex128),
            k_arr,
            hr_times,
            restricted_hr_hat,
            k_arr,
        )
        case_dir = eval_root / case_id
        case_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            case_dir / "trajectory.npz",
            times=times,
            model_energy=learned_energy[row],
            teacher_times=hr_times,
            teacher_energy=hr_energy,
            model_E_hat=learned_hat[row, :, :5],
            teacher_E_hat=restricted_hr_hat[:, :5],
        )
        fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
        ax.semilogy(hr_times, hr_energy, color="#2463eb", label="kinetic teacher")
        ax.semilogy(times, learned_energy[row], color="#6f3cc3", label="low-moment closure")
        if failure_time is not None:
            ax.axvline(failure_time, color="#c44e52", linestyle="--", label="instability threshold")
        ax.set_xlabel("t")
        ax.set_ylabel("Electric-field energy")
        ax.set_title(f"{case_id} ({regime})")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(case_dir / "metric1_energy.png", dpi=200)
        plt.close(fig)
        learned_complex = np.asarray(learned_hat[row], dtype=np.complex128)
        matched_reference_hat = _interpolate_time_series(
            hr_times, restricted_hr_hat, times
        )
        with np.errstate(over="ignore", invalid="ignore"):
            relative = np.sqrt(
                np.sum(
                    np.abs(learned_complex - matched_reference_hat) ** 2,
                    axis=-1,
                )
                / np.maximum(
                    np.sum(np.abs(matched_reference_hat) ** 2, axis=-1), 1e-30
                )
            )
        fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
        ax.semilogy(times, relative, color="#c44e52")
        ax.set_xlabel("t")
        ax.set_ylabel("Relative electric-field error")
        if failure_time is not None:
            ax.axvline(failure_time, color="#c44e52", linestyle="--")
        metric_text = _json_float(field_error.epsilon_E)
        metric_label = "non-finite" if metric_text is None else f"{metric_text:.3e}"
        interval_label = "full" if finite_count == times.size else f"t<={times[finite_count - 1]:.2f}"
        ax.set_title(
            f"{case_id}: integrated $\\varepsilon_E={metric_label}$ ({interval_label})"
        )
        ax.grid(alpha=0.25)
        fig.savefig(case_dir / "metric2_field_error.png", dpi=200)
        plt.close(fig)
        model_frequency = _energy_peak_angular_frequency(
            times[:finite_count], learned_energy[row, :finite_count]
        )
        teacher_frequency = _energy_peak_angular_frequency(hr_times, hr_energy)
        summary = {
            "case_id": case_id,
            "regime": regime,
            "bounded_to_final_time": bounded_to_final,
            "diagnostic_failure_time": failure_time,
            "epsilon_grow": _json_float(growth.epsilon_grow),
            "gamma_hr": _json_float(growth.gamma_grow_hr),
            "gamma_theta": _json_float(growth.gamma_grow_theta),
            "epsilon_E": _json_float(field_error.epsilon_E),
            "epsilon_E_final_time": float(times[finite_count - 1]),
            "maximum_energy_ratio": _json_float(
                np.nanmax(learned_energy[row, :finite_count]) / reference_energy_max
            ),
            "stability_energy_multiplier": 1e6,
            "minimum_density": _json_float(minimum_density[row]),
            "minimum_pressure": _json_float(minimum_pressure[row]),
            "density_limiter_fraction": float(
                density_limiter_hits[row] / max(state_point_count, 1)
            ),
            "pressure_limiter_fraction": float(
                pressure_limiter_hits[row] / max(state_point_count, 1)
            ),
            "energy_peak_angular_frequency_model": model_frequency,
            "energy_peak_angular_frequency_teacher": teacher_frequency,
            "energy_peak_frequency_relative_error": (
                None
                if model_frequency is None or teacher_frequency in (None, 0.0)
                else abs(model_frequency - teacher_frequency)
                / abs(teacher_frequency)
            ),
        }
        with (case_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        summaries.append(summary)
        metric1_traces.append(
            (case_id, hr_times, hr_energy, learned_energy[row], failure_time)
        )
        status = "bounded" if bounded_to_final else f"failed@t={failure_time:.2f}"
        print(f"[eval] {case_id}: {status} epsilon_E={metric_label}")
    fig, axes = plt.subplots(
        len(metric1_traces),
        1,
        figsize=(10.0, 2.0 * len(metric1_traces)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for index, (case_id, hr_times, hr_energy, model_energy, failure_time) in enumerate(
        metric1_traces
    ):
        axis = axes[index]
        axis.semilogy(hr_times, hr_energy, color="#2463eb", label="kinetic teacher")
        axis.semilogy(times, model_energy, color="#6f3cc3", label="low-moment closure")
        if failure_time is not None:
            axis.axvline(failure_time, color="#c44e52", linestyle="--")
        axis.set_ylabel(case_id, rotation=0, ha="right", va="center", fontsize=8)
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("t")
    fig.suptitle("Held-out electric-field energy")
    fig.savefig(eval_root / "heldout_metric1_summary.png", dpi=180)
    plt.close(fig)
    aggregate = {
        "regimes": list(regimes),
        "canonical_regimes": list(REGIMES),
        "bounded_cases": sum(case["bounded_to_final_time"] for case in summaries),
        "case_count": len(summaries),
        "cases": summaries,
    }
    with (eval_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def evaluate_saved_checkpoint(
    args, *, grouped, manifest, coefficient_key, source_nx,
    domain_length, reference_dt, poisson_sign, outdir,
):
    """Regenerate held-out diagnostics using the checkpoint's saved configuration."""
    params, checkpoint_metadata, checkpoint_stats = _load_checkpoint(
        args.evaluate_checkpoint
    )
    if int(checkpoint_metadata.get("schema_version", 0)) != CHECKPOINT_SCHEMA:
        raise ValueError(
            "Checkpoint uses the obsolete full-state/heat-flux parameterization"
        )
    if str(checkpoint_metadata.get("manifest_sha256")) != str(manifest["sha256"]):
        raise ValueError("Checkpoint and reference-cache IC manifests do not match")
    checkpoint_source_nx = int(checkpoint_metadata["source_Nx"])
    if checkpoint_source_nx != source_nx:
        raise ValueError("Checkpoint and reference-cache source Nx do not match")
    rollout_nx = int(checkpoint_metadata["rollout_Nx"])
    outdir.mkdir(parents=True, exist_ok=True)
    print(
        f"[eval] regenerating held-out diagnostics from {args.evaluate_checkpoint}"
    )
    _evaluate_heldout(
        params,
        grouped,
        manifest,
        coefficient_key,
        Path(args.reference_cache),
        outdir,
        regimes=_checkpoint_regimes(checkpoint_metadata),
        source_nx=source_nx,
        rollout_nx=rollout_nx,
        domain_length=domain_length,
        dt=float(checkpoint_metadata.get("dt", reference_dt)),
        reference_dt=reference_dt,
        width=int(checkpoint_metadata["width"]),
        memory_steps=int(checkpoint_metadata["memory_steps"]),
        memory_stride=int(checkpoint_metadata["memory_stride"]),
        scan_unroll=int(checkpoint_metadata.get("scan_unroll", 1)),
        input_scale=checkpoint_stats["input_scale"],
        heat_flux_gradient_scale=float(checkpoint_stats["heat_flux_gradient_scale"][0]),
        amplitude_center=float(checkpoint_stats["amplitude_center"][0]),
        amplitude_scale=float(checkpoint_stats["amplitude_scale"][0]),
        poisson_sign=poisson_sign,
        chunk_steps=args.evaluation_chunk_steps,
        normalized_heat_flux_bound=float(
            checkpoint_metadata.get(
                "normalized_heat_flux_bound", DEFAULT_NORMALIZED_HEAT_FLUX_BOUND
            )
        ),
        density_floor=float(
            checkpoint_metadata.get("density_floor", DEFAULT_DENSITY_FLOOR)
        ),
        pressure_floor=float(
            checkpoint_metadata.get("pressure_floor", DEFAULT_PRESSURE_FLOOR)
        ),
        closure_history_input=bool(
            checkpoint_metadata.get("closure_history_input", False)
        ),
        memory_backend=str(
            checkpoint_metadata.get("memory_backend", "latent_recurrent")
        ),
        input_scaling=str(
            checkpoint_metadata.get("input_scaling", FIXED_INPUT_SCALING)
        ),
        dynamic_amplitude_floor=float(
            checkpoint_metadata.get("dynamic_amplitude_floor", 1e-6)
        ),
        allow_uniform_heating=bool(
            checkpoint_metadata.get("allow_uniform_heating", False)
        ),
    )
    metrics_path = outdir / "training_metrics.npz"
    if metrics_path.exists():
        with np.load(metrics_path, allow_pickle=False) as metrics:
            _plot_losses(
                metrics["train_loss"],
                metrics["train_ema_loss"]
                if "train_ema_loss" in metrics.files
                else np.asarray([]),
                metrics["train_eval_epochs"]
                if "train_eval_epochs" in metrics.files
                else np.asarray([]),
                metrics["train_eval_loss"]
                if "train_eval_loss" in metrics.files
                else np.asarray([]),
                metrics["val_epochs"],
                metrics["val_loss"],
                outdir / "training_loss.png",
                checkpoint_metadata,
            )
    return
