"""Diagnose long-horizon behavior of the Burles-style compact latent closure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping

from vpml.jax_runtime import bootstrap_jax_runtime

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter1d

from model.train.interface_flux_data import IC_SPLIT_HELDOUT, load_ic_manifest
from model.train.low_moment_closure import (
    REGIMES,
    _electric_field_energy,
    _load_checkpoint,
    _restrict_rfft,
    low_hermite_coefficients_to_conservative,
)
from vpml.low_moment import (
    DYNAMIC_INPUT_SCALING,
    _dealias,
    burles_latent_closure_rhs,
    electric_field_from_density,
    encode_explicit_window_history,
    initialize_burles_latent_state,
    limit_low_moment_state,
    low_moment_rhs,
    rollout_burles_latent_closure,
)
from vpml.metrics import FieldErrorConfig, SelfGeneratedFieldErrorMetric


def _json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_cases(cache_dir: Path, metadata: Mapping[str, object]):
    manifest = load_ic_manifest(cache_dir / "ic_manifest.json")
    source_nx = int(metadata["source_Nx"])
    rollout_nx = int(metadata["rollout_Nx"])
    states = []
    amplitudes = []
    records = []
    for regime in REGIMES:
        for case in manifest["cases"]:
            if case["regime"] != regime or case["split"] != IC_SPLIT_HELDOUT:
                continue
            case_id = str(case["case_id"])
            coefficients = np.load(
                cache_dir / "cases" / f"{case_id}.npy", mmap_mode="r"
            )
            states.append(
                low_hermite_coefficients_to_conservative(
                    coefficients[0, :3],
                    source_nx=source_nx,
                    target_nx=rollout_nx,
                )
            )
            amplitudes.append(float(case["epsilon"]))
            records.append({"case_id": case_id, "regime": regime})
    return (
        jnp.asarray(np.stack(states), dtype=jnp.float32),
        jnp.asarray(amplitudes, dtype=jnp.float32),
        records,
        manifest,
    )


def _configuration(cache_dir: Path, checkpoint_metadata: Mapping[str, object]):
    cache_metadata = _json(cache_dir / "metadata.json")
    configuration = cache_metadata["configuration"]
    nx = int(checkpoint_metadata["rollout_Nx"])
    domain_length = float(configuration["teacher_L"])
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(nx, d=domain_length / nx)
    return configuration, domain_length, jnp.asarray(k_arr, dtype=jnp.float32)


def _initial_memory(params, state, amplitude, k_arr, settings):
    steps = int(settings["memory_steps"])
    history = jnp.zeros(
        (state.shape[0], steps, state.shape[1], state.shape[-1]), dtype=state.dtype
    )
    closure_history = jnp.zeros(
        (state.shape[0], steps, state.shape[-1]), dtype=state.dtype
    )
    previous = jnp.zeros((state.shape[0], state.shape[-1]), dtype=state.dtype)
    encoded = encode_explicit_window_history(
        params,
        history,
        k_arr,
        input_scale=settings["input_scale"],
        heat_flux_gradient_scale=settings["heat_flux_gradient_scale"],
        heat_flux_gradient_history=closure_history,
        poisson_sign=settings["poisson_sign"],
        input_scaling=settings["input_scaling"],
        dynamic_amplitude_floor=settings["dynamic_amplitude_floor"],
    )
    return (
        history,
        closure_history,
        encoded,
        jnp.asarray(0, dtype=jnp.int32),
        previous,
        None,
    )


def _standard_rollout(params, state, memory, amplitude, k_arr, horizon, settings):
    history, closure_history, _, counter, previous, latent = memory
    return rollout_burles_latent_closure(
        params,
        state,
        history,
        counter,
        amplitude,
        k_arr,
        horizon=int(horizon),
        dt=settings["dt"],
        memory_stride=settings["memory_stride"],
        input_scale=settings["input_scale"],
        heat_flux_gradient_scale=settings["heat_flux_gradient_scale"],
        amplitude_center=settings["amplitude_center"],
        amplitude_scale=settings["amplitude_scale"],
        compact_latent=latent,
        previous_heat_flux_gradient=previous,
        heat_flux_gradient_history=closure_history,
        poisson_sign=settings["poisson_sign"],
        normalized_heat_flux_bound=settings["normalized_heat_flux_bound"],
        density_floor=settings["density_floor"],
        pressure_floor=settings["pressure_floor"],
        scan_unroll=settings["scan_unroll"],
        input_scaling=settings["input_scaling"],
        dynamic_amplitude_floor=settings["dynamic_amplitude_floor"],
        allow_uniform_heating=settings["allow_uniform_heating"],
    )


def _telemetry_rollout(
    params,
    initial_state,
    memory,
    amplitude,
    k_arr,
    horizon,
    settings,
    *,
    reset_latent_each_step: bool = False,
):
    """Mirror the production SSPRK3 rollout while returning latent/closure traces."""
    history, closure_history, _, counter, previous, compact_latent = memory
    state = jnp.asarray(initial_state)
    if compact_latent is None:
        compact_latent = initialize_burles_latent_state(
            params,
            state,
            history,
            amplitude,
            k_arr,
            input_scale=settings["input_scale"],
            heat_flux_gradient_scale=settings["heat_flux_gradient_scale"],
            amplitude_center=settings["amplitude_center"],
            amplitude_scale=settings["amplitude_scale"],
            previous_heat_flux_gradient=previous,
            heat_flux_gradient_history=closure_history,
            poisson_sign=settings["poisson_sign"],
            input_scaling=settings["input_scaling"],
            dynamic_amplitude_floor=settings["dynamic_amplitude_floor"],
        )
    dt_value = jnp.asarray(settings["dt"], dtype=state.dtype)
    latent_bound = jnp.asarray(8.0, dtype=state.dtype)

    def bounded_latent(value):
        return latent_bound * jnp.tanh(value / latent_bound)

    def physical_stage(reference, value):
        return limit_low_moment_state(
            _dealias(value),
            reference_state=reference,
            density_floor=settings["density_floor"],
            pressure_floor=settings["pressure_floor"],
        )

    def closure_rhs(fluid, latent, carried, window, closure_window):
        return burles_latent_closure_rhs(
            params,
            fluid,
            latent,
            window,
            amplitude,
            k_arr,
            input_scale=settings["input_scale"],
            heat_flux_gradient_scale=settings["heat_flux_gradient_scale"],
            amplitude_center=settings["amplitude_center"],
            amplitude_scale=settings["amplitude_scale"],
            previous_heat_flux_gradient=carried,
            heat_flux_gradient_history=closure_window,
            poisson_sign=settings["poisson_sign"],
            normalized_heat_flux_bound=settings["normalized_heat_flux_bound"],
            input_scaling=settings["input_scaling"],
            dynamic_amplitude_floor=settings["dynamic_amplitude_floor"],
            allow_uniform_heating=settings["allow_uniform_heating"],
        )

    def body(carry, _):
        fluid0, latent0, window, closure_window, history_counter, closure0 = carry
        if reset_latent_each_step:
            latent0 = jnp.zeros_like(latent0)
        closure1, latent_rhs0 = closure_rhs(
            fluid0, latent0, closure0, window, closure_window
        )
        fluid1 = physical_stage(
            fluid0,
            fluid0
            + dt_value
            * low_moment_rhs(
                fluid0,
                closure0,
                k_arr,
                poisson_sign=settings["poisson_sign"],
                density_floor=settings["density_floor"],
                pressure_floor=settings["pressure_floor"],
            ),
        )
        latent1 = bounded_latent(latent0 + dt_value * latent_rhs0)

        closure2, latent_rhs1 = closure_rhs(
            fluid1, latent1, closure1, window, closure_window
        )
        fluid2_candidate = fluid1 + dt_value * low_moment_rhs(
            fluid1,
            closure1,
            k_arr,
            poisson_sign=settings["poisson_sign"],
            density_floor=settings["density_floor"],
            pressure_floor=settings["pressure_floor"],
        )
        fluid2 = physical_stage(fluid0, 0.75 * fluid0 + 0.25 * fluid2_candidate)
        latent2 = bounded_latent(
            0.75 * latent0 + 0.25 * (latent1 + dt_value * latent_rhs1)
        )

        closure3, latent_rhs2 = closure_rhs(
            fluid2, latent2, closure2, window, closure_window
        )
        fluid3_candidate = fluid2 + dt_value * low_moment_rhs(
            fluid2,
            closure2,
            k_arr,
            poisson_sign=settings["poisson_sign"],
            density_floor=settings["density_floor"],
            pressure_floor=settings["pressure_floor"],
        )
        fluid3 = physical_stage(
            fluid0, (1.0 / 3.0) * fluid0 + (2.0 / 3.0) * fluid3_candidate
        )
        latent3 = bounded_latent(
            (1.0 / 3.0) * latent0
            + (2.0 / 3.0) * (latent2 + dt_value * latent_rhs2)
        )
        closure_final, _ = closure_rhs(
            fluid3, latent3, closure3, window, closure_window
        )
        next_counter = history_counter + 1
        should_sample = next_counter >= int(settings["memory_stride"])

        def sample(_):
            return (
                jnp.concatenate((window[:, 1:], fluid3[:, None]), axis=1),
                jnp.concatenate(
                    (closure_window[:, 1:], closure_final[:, None]), axis=1
                ),
                jnp.asarray(0, dtype=jnp.int32),
            )

        def retain(_):
            return window, closure_window, next_counter

        next_window, next_closure_window, next_counter = jax.lax.cond(
            should_sample, sample, retain, operand=None
        )
        next_carry = (
            fluid3,
            latent3,
            next_window,
            next_closure_window,
            next_counter,
            closure_final,
        )
        return next_carry, (fluid3, latent3, closure_final)

    final, traces = jax.lax.scan(
        jax.checkpoint(body),
        (state, compact_latent, history, closure_history, counter, previous),
        xs=None,
        length=int(horizon),
        unroll=int(settings["scan_unroll"]),
    )
    states, latents, closures = traces
    final_fluid, final_latent, final_window, final_closure, final_counter, final_grad = final
    final_encoded = encode_explicit_window_history(
        params,
        final_window,
        k_arr,
        input_scale=settings["input_scale"],
        heat_flux_gradient_scale=settings["heat_flux_gradient_scale"],
        heat_flux_gradient_history=final_closure,
        poisson_sign=settings["poisson_sign"],
        input_scaling=settings["input_scaling"],
        dynamic_amplitude_floor=settings["dynamic_amplitude_floor"],
    )
    del final_fluid
    final_memory = (
        final_window,
        final_closure,
        final_encoded,
        final_counter,
        final_grad,
        final_latent,
    )
    return (
        jnp.swapaxes(states, 0, 1),
        jnp.swapaxes(latents, 0, 1),
        jnp.swapaxes(closures, 0, 1),
        final_memory,
    )


def _field_data(initial_state, states, k_arr, nx, domain_length):
    complete = np.concatenate((np.asarray(initial_state)[:, None], np.asarray(states)), axis=1)
    flat = complete.reshape(-1, 3, nx)
    fields = np.asarray(electric_field_from_density(jnp.asarray(flat[:, 0]), k_arr))
    fields = fields.reshape(complete.shape[0], complete.shape[1], nx)
    hats = np.fft.rfft(fields, axis=-1)
    energy = _electric_field_energy(hats, nx=nx, dx=domain_length / nx)
    return complete, fields, hats, energy


def _teacher(cache_dir: Path, case_id: str, source_nx: int, target_nx: int):
    with np.load(cache_dir / "snapshots" / f"{case_id}.npz") as payload:
        times = np.asarray(payload["E_hat_hist_times"], dtype=np.float64)
        hats = _restrict_rfft(
            np.asarray(payload["E_hat_hist"], dtype=np.complex128),
            source_nx,
            target_nx,
        )
        energy = np.asarray(payload["energy"], dtype=np.float64)
    return times, hats, energy


def _prefix_metrics(times, hats, energy, teacher_values, k_arr, horizons):
    teacher_times, teacher_hats, teacher_energy = teacher_values
    rows = []
    for horizon in horizons:
        keep = times <= float(horizon) + 1e-12
        metric = SelfGeneratedFieldErrorMetric(FieldErrorConfig(final_time=float(horizon)))
        epsilon_e = metric.evaluate_fourier(
            times[keep], hats[keep], k_arr, teacher_times, teacher_hats, k_arr
        ).epsilon_E
        teacher_log = np.interp(
            times[keep], teacher_times, np.log10(np.maximum(teacher_energy, 1e-30))
        )
        model_log = np.log10(np.maximum(energy[keep], 1e-30))
        rows.append(
            {
                "horizon": float(horizon),
                "epsilon_E": float(epsilon_e),
                "log_energy_rmse_decades": float(
                    np.sqrt(np.mean(np.square(model_log - teacher_log)))
                ),
            }
        )
    return rows


def _sustained_onset(mask: np.ndarray, samples: int):
    if samples <= 1:
        hits = np.flatnonzero(mask)
        return None if not hits.size else int(hits[0])
    run = np.convolve(mask.astype(np.int32), np.ones(samples, dtype=np.int32), mode="valid")
    hits = np.flatnonzero(run >= samples)
    return None if not hits.size else int(hits[0])


def _drift_onsets(times, model_energy, teacher_times, teacher_energy):
    dt = float(np.median(np.diff(times)))
    teacher_dt = float(np.median(np.diff(teacher_times)))
    envelope_span = 2.0
    model_log = np.log10(np.maximum(model_energy, 1e-30))
    teacher_log = np.log10(np.maximum(teacher_energy, 1e-30))
    model_envelope = maximum_filter1d(
        model_log, size=max(3, int(round(envelope_span / dt))), mode="nearest"
    )
    teacher_envelope = maximum_filter1d(
        teacher_log,
        size=max(3, int(round(envelope_span / teacher_dt))),
        mode="nearest",
    )
    difference = model_envelope - np.interp(times, teacher_times, teacher_envelope)
    eligible = times >= 5.0
    result = {"maximum_upward_envelope_error_decades": float(np.max(difference[eligible]))}
    for threshold in (0.5, 1.0):
        index = _sustained_onset(
            eligible & (difference > threshold), max(1, int(round(2.0 / dt)))
        )
        result[f"upward_{threshold:g}_decade_sustained_onset"] = (
            None if index is None else float(times[index])
        )
    return result


def _plot_latent_norms(times, latents, records, outpath):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=True)
    for row, (axis, record) in enumerate(zip(axes.flat, records)):
        channel_rms = np.sqrt(np.mean(np.square(latents[row]), axis=-1))
        total = np.sqrt(np.mean(np.square(latents[row]), axis=(1, 2)))
        for channel in range(channel_rms.shape[1]):
            axis.plot(times[1:], channel_rms[:, channel], lw=0.8, alpha=0.65)
        axis.plot(times[1:], total, color="black", lw=2.0, label="total RMS")
        axis.set_title(record["case_id"], fontsize=9)
        axis.grid(alpha=0.25)
        if row == 0:
            axis.legend(fontsize=8)
    fig.supxlabel("t")
    fig.supylabel("latent spatial RMS")
    fig.suptitle("Persistent latent-state norms — E20", fontsize=18)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _plot_latent_spectra(times, latents, records, outpath):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=True, sharey=True)
    images = []
    spectra = []
    for row in range(latents.shape[0]):
        hat = np.fft.rfft(latents[row], axis=-1)
        power = np.mean(np.square(np.abs(hat)), axis=1)
        spectra.append(np.log10(np.maximum(power, 1e-20)))
    low = float(np.percentile(np.concatenate([value.ravel() for value in spectra]), 2.0))
    high = float(np.percentile(np.concatenate([value.ravel() for value in spectra]), 98.0))
    for axis, record, spectrum in zip(axes.flat, records, spectra):
        image = axis.imshow(
            spectrum.T,
            origin="lower",
            aspect="auto",
            extent=(times[1], times[-1], 0, spectrum.shape[1] - 1),
            cmap="magma",
            vmin=low,
            vmax=high,
        )
        images.append(image)
        axis.set_title(record["case_id"], fontsize=9)
    fig.supxlabel("t")
    fig.supylabel("spatial Fourier mode")
    fig.suptitle("Latent-state spectral power — E20", fontsize=18)
    fig.colorbar(images[-1], ax=axes.ravel().tolist(), label="log10 mean channel power")
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.07, top=0.92, wspace=0.12, hspace=0.22)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _plot_transfer(times, states, fields, closures, records, outpath):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=True)
    dt = float(np.median(np.diff(times)))
    for row, (axis, record) in enumerate(zip(axes.flat, records)):
        momentum = states[row, 1:, 1]
        field_current = np.mean(fields[row, 1:] * momentum, axis=-1)
        closure_source = -0.5 * np.mean(closures[row], axis=-1)
        cumulative = np.cumsum(closure_source) * dt
        scale = max(
            float(np.max(np.abs(field_current))),
            float(np.max(np.abs(closure_source))),
            1e-20,
        )
        axis.plot(times[1:], field_current / scale, color="#3366cc", lw=1.0, label="<E,j>")
        axis.plot(times[1:], closure_source / scale, color="#cc3333", lw=1.0, label="closure source")
        cumulative_scale = max(float(np.max(np.abs(cumulative))), 1e-20)
        axis.plot(
            times[1:], cumulative / cumulative_scale,
            color="#222222", lw=1.4, alpha=0.8, label="cumulative closure source"
        )
        axis.axhline(0.0, color="0.7", lw=0.6)
        axis.set_title(record["case_id"], fontsize=9)
        axis.grid(alpha=0.25)
        if row == 0:
            axis.legend(fontsize=7, ncol=3)
    fig.supxlabel("t")
    fig.supylabel("per-case normalized signed transfer")
    fig.suptitle("Field-current and learned-closure energy transfer — E20", fontsize=18)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _plot_ablation_energy(times, energies, teachers, records, outpath):
    colors = {"E20": "#7047d7", "z_reset": "#e67e22", "closure_off": "#222222"}
    fig, axes = plt.subplots(len(records), 1, figsize=(12, 2.3 * len(records)), sharex=True)
    for row, (axis, record) in enumerate(zip(axes, records)):
        teacher_times, _, teacher_energy = teachers[row]
        axis.semilogy(teacher_times, teacher_energy, color="#2468ff", lw=1.4, label="kinetic teacher")
        for name in ("E20", "z_reset", "closure_off"):
            axis.semilogy(times, energies[name][row], color=colors[name], lw=1.15, label=name)
        axis.set_ylabel(record["case_id"], rotation=0, ha="right", va="center", fontsize=8)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("t")
    fig.suptitle("E20 autonomous ablations", fontsize=17)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _plot_horizons(rows, outpath):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for variant, values in grouped.items():
        horizons = sorted({float(value["horizon"]) for value in values})
        mean_e = [np.mean([v["epsilon_E"] for v in values if v["horizon"] == h]) for h in horizons]
        mean_log = [np.mean([v["log_energy_rmse_decades"] for v in values if v["horizon"] == h]) for h in horizons]
        axes[0].plot(horizons, mean_e, marker="o", label=variant)
        axes[1].plot(horizons, mean_log, marker="o", label=variant)
    axes[0].set_ylabel("mean epsilon_E")
    axes[1].set_ylabel("mean log-energy RMSE [decades]")
    for axis in axes:
        axis.set_xlabel("autonomous horizon")
        axis.set_xticks((10, 20, 40, 120))
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Fixed-horizon autonomous errors")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-e15", type=Path, required=True)
    parser.add_argument("--checkpoint-e20", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--stored-e20-evaluation", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=False)
    params20, metadata20, stats20 = _load_checkpoint(args.checkpoint_e20)
    params15, metadata15, _ = _load_checkpoint(args.checkpoint_e15)
    if metadata15["manifest_sha256"] != metadata20["manifest_sha256"]:
        raise ValueError("E15 and E20 manifests differ")
    state, amplitude, records, manifest = _load_cases(args.reference_cache, metadata20)
    configuration, domain_length, k_arr = _configuration(args.reference_cache, metadata20)
    nx = int(metadata20["rollout_Nx"])
    dt = float(metadata20["dt"])
    total_steps = int(round(float(configuration["T_final"]) / dt))
    times = np.arange(total_steps + 1, dtype=np.float64) * dt
    settings = {
        "dt": dt,
        "memory_steps": int(metadata20["memory_steps"]),
        "memory_stride": int(metadata20["memory_stride"]),
        "scan_unroll": int(metadata20.get("scan_unroll", 1)),
        "input_scale": jnp.asarray(stats20["input_scale"], dtype=jnp.float32),
        "heat_flux_gradient_scale": float(stats20["heat_flux_gradient_scale"][0]),
        "amplitude_center": float(stats20["amplitude_center"][0]),
        "amplitude_scale": float(stats20["amplitude_scale"][0]),
        "poisson_sign": float(configuration["teacher_poisson_sign"]),
        "normalized_heat_flux_bound": float(metadata20["normalized_heat_flux_bound"]),
        "density_floor": float(metadata20["density_floor"]),
        "pressure_floor": float(metadata20["pressure_floor"]),
        "input_scaling": str(metadata20["input_scaling"]),
        "dynamic_amplitude_floor": float(metadata20["dynamic_amplitude_floor"]),
        "allow_uniform_heating": bool(metadata20["allow_uniform_heating"]),
    }
    if settings["input_scaling"] != DYNAMIC_INPUT_SCALING:
        raise ValueError("This diagnostic is scoped to the matched dynamic-scaling run")

    initial20 = _initial_memory(params20, state, amplitude, k_arr, settings)
    print("[diagnostic] compiling E20 uninterrupted telemetry rollout", flush=True)
    telemetry = jax.jit(
        lambda: _telemetry_rollout(
            params20, state, initial20, amplitude, k_arr, total_steps, settings
        )
    )()
    states20, latents20, closures20, _ = jax.tree_util.tree_map(
        lambda value: np.asarray(value), telemetry
    )
    complete20, fields20, hats20, energy20 = _field_data(
        state, states20, k_arr, nx, domain_length
    )

    print("[diagnostic] checking against stored chunked rollout", flush=True)
    parity_rows = []
    for row, record in enumerate(records):
        with np.load(
            args.stored_e20_evaluation
            / "heldout_cases"
            / record["case_id"]
            / "trajectory.npz"
        ) as stored:
            stored_hat = np.asarray(stored["model_E_hat"])
            stored_energy = np.asarray(stored["model_energy"])
        modes = stored_hat.shape[-1]
        absolute = float(np.max(np.abs(hats20[row, :, :modes] - stored_hat)))
        energy_absolute = float(np.max(np.abs(energy20[row] - stored_energy)))
        energy_relative = float(
            np.max(np.abs(energy20[row] - stored_energy) / np.maximum(stored_energy, 1e-30))
        )
        parity_rows.append(
            {
                **record,
                "max_abs_first_saved_E_hat_modes": absolute,
                "max_abs_energy": energy_absolute,
                "max_relative_energy": energy_relative,
            }
        )

    print("[diagnostic] running E15 baseline", flush=True)
    initial15 = _initial_memory(params15, state, amplitude, k_arr, settings)
    states15, _ = jax.jit(
        lambda: _standard_rollout(
            params15, state, initial15, amplitude, k_arr, total_steps, settings
        )
    )()
    _, _, hats15, energy15 = _field_data(state, states15, k_arr, nx, domain_length)

    print("[diagnostic] running no-persistent-z ablation", flush=True)
    reset_states, _, _, _ = jax.jit(
        lambda: _telemetry_rollout(
            params20,
            state,
            initial20,
            amplitude,
            k_arr,
            total_steps,
            settings,
            reset_latent_each_step=True,
        )
    )()
    _, _, reset_hats, reset_energy = _field_data(
        state, reset_states, k_arr, nx, domain_length
    )

    print("[diagnostic] running learned-closure-off ablation", flush=True)
    zero_params = dict(params20)
    for key in ("output_local", "output_spectral_real", "output_spectral_imag"):
        zero_params[key] = jnp.zeros_like(zero_params[key])
    off_memory = _initial_memory(zero_params, state, amplitude, k_arr, settings)
    off_states, _ = jax.jit(
        lambda: _standard_rollout(
            zero_params, state, off_memory, amplitude, k_arr, total_steps, settings
        )
    )()
    _, _, off_hats, off_energy = _field_data(state, off_states, k_arr, nx, domain_length)

    teachers = [
        _teacher(args.reference_cache, record["case_id"], int(metadata20["source_Nx"]), nx)
        for record in records
    ]
    variants = {
        "E15": (hats15, energy15),
        "E20": (hats20, energy20),
        "z_reset": (reset_hats, reset_energy),
        "closure_off": (off_hats, off_energy),
    }
    horizon_rows = []
    horizons = (10.0, 20.0, 40.0, 120.0)
    k_numpy = np.asarray(k_arr, dtype=np.float64)
    for variant, (hats, energies) in variants.items():
        for row, record in enumerate(records):
            for metric_row in _prefix_metrics(
                times, hats[row], energies[row], teachers[row], k_numpy, horizons
            ):
                horizon_rows.append({"variant": variant, **record, **metric_row})

    onset_rows = []
    for variant in ("E15", "E20", "z_reset", "closure_off"):
        energies = variants[variant][1]
        for row, record in enumerate(records):
            if record["regime"] != "linear_landau" and record["case_id"] != "nonlinear_landau_strong_ic12":
                continue
            teacher_times, _, teacher_energy = teachers[row]
            onset_rows.append(
                {
                    "variant": variant,
                    **record,
                    **_drift_onsets(times, energies[row], teacher_times, teacher_energy),
                }
            )

    with (args.outdir / "chunk_parity.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "comparison": "uninterrupted telemetry scan versus stored chunk_steps=250 evaluation",
                "rows": parity_rows,
                "maximums": {
                    key: max(row[key] for row in parity_rows)
                    for key in (
                        "max_abs_first_saved_E_hat_modes",
                        "max_abs_energy",
                        "max_relative_energy",
                    )
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    for filename, rows in (
        ("fixed_horizon_metrics.csv", horizon_rows),
        ("drift_onsets.csv", onset_rows),
    ):
        with (args.outdir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    closure_source = -0.5 * np.mean(closures20, axis=-1)
    latent_rms = np.sqrt(np.mean(np.square(latents20), axis=-1))
    np.savez_compressed(
        args.outdir / "e20_latent_closure_telemetry.npz",
        times=times,
        case_ids=np.asarray([record["case_id"] for record in records]),
        latent_channel_rms=latent_rms,
        latent_spectral_power=np.stack(
            [
                np.mean(np.square(np.abs(np.fft.rfft(latents20[row], axis=-1))), axis=1)
                for row in range(len(records))
            ]
        ),
        closure_mean_energy_source=closure_source,
        closure_rms=np.sqrt(np.mean(np.square(closures20), axis=-1)),
    )
    _plot_latent_norms(times, latents20, records, args.outdir / "latent_norms.png")
    _plot_latent_spectra(times, latents20, records, args.outdir / "latent_spectra.png")
    _plot_transfer(
        times, complete20, fields20, closures20, records, args.outdir / "closure_energy_transfer.png"
    )
    _plot_ablation_energy(
        times,
        {"E20": energy20, "z_reset": reset_energy, "closure_off": off_energy},
        teachers,
        records,
        args.outdir / "e20_ablation_energy_trajectories.png",
    )
    _plot_horizons(horizon_rows, args.outdir / "fixed_horizon_errors.png")

    aggregate = {}
    for variant in variants:
        aggregate[variant] = {}
        for horizon in horizons:
            selected = [
                row
                for row in horizon_rows
                if row["variant"] == variant and row["horizon"] == horizon
            ]
            aggregate[variant][f"T{horizon:g}"] = {
                "mean_epsilon_E": float(np.mean([row["epsilon_E"] for row in selected])),
                "mean_log_energy_rmse_decades": float(
                    np.mean([row["log_energy_rmse_decades"] for row in selected])
                ),
                "max_epsilon_E": float(np.max([row["epsilon_E"] for row in selected])),
            }
    summary = {
        "checkpoint_e15": str(args.checkpoint_e15.resolve()),
        "checkpoint_e20": str(args.checkpoint_e20.resolve()),
        "reference_cache": str(args.reference_cache.resolve()),
        "manifest_sha256": manifest["sha256"],
        "chunk_parity_maximums": _json(args.outdir / "chunk_parity.json")["maximums"],
        "fixed_horizon_aggregate": aggregate,
        "drift_onsets": onset_rows,
        "ablation_definitions": {
            "z_reset": "set all six latent channels to zero at the start of every accepted solver step",
            "closure_off": "set learned closure output head to zero; retain identical physical SSPRK3 solver",
        },
    }
    with (args.outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
