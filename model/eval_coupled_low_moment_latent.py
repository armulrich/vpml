"""Evaluate a saved coupled low-moment latent model without retraining."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model.train.coupled_low_moment_latent import (
    _energy_regrowth_diagnostics,
    _load_complete_case,
    _make_functions,
    _matrix_square_root_propagator,
)
from model.train.interface_flux_data import case_shard_paths, load_ic_manifest
from model.train.kinetic_latent_dynamics_probe import REGIMES, _selected_cases
from model.train.low_moment_closure import (
    _electric_field_energy,
    _json_float,
    _restrict_rfft,
)
from vpml.jax_runtime import print_jax_runtime_summary
from vpml.kinetic_latent import (
    init_state_conditioned_latent_operator,
    resolved_hermite_to_low_moment_state,
)
from vpml.low_moment import electric_field_from_density, primitive_fields
from vpml.metrics import (
    EarlyElectricFieldGrowthMetric,
    EarlyGrowthConfig,
    FieldErrorConfig,
    SelfGeneratedFieldErrorMetric,
)
from vpml.metrics.trajectory import envelope_diagnostics, latent_saturation_diagnostics, matched_cadence_growth
from model.train.coupled_run_support import file_digest


def _assert_outputs_absent(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing evaluation outputs: {joined}")


def _load_saved_model(run_dir: Path, checkpoint_name="best_coupled_low_moment_latent.npz"):
    report = json.loads((run_dir / "report.json").read_text())
    config = dict(report["configuration"])
    checkpoint_path = run_dir / checkpoint_name
    operator_modes = int(config["operator_modes"])
    with np.load(checkpoint_path, allow_pickle=False) as payload:
        array_keys = [
            "basis",
            "linear_propagator",
            "resolved_center",
            "resolved_scale",
            "latent_center",
            "latent_scale",
            "k_arr",
        ]
        nonparameter_keys = set(array_keys) | {
            "latent_input_scale",
            "correction_bounds",
            "latent_residual_scale",
            "closure_residual_scale",
            "closure_correction_bound",
        }
        params = {
            key: jnp.asarray(payload[key])
            for key in payload.files
            if key not in nonparameter_keys
        }
        arrays = {
            key: np.asarray(payload[key])
            for key in array_keys
        }
        if "latent_input_scale" in payload.files:
            arrays["latent_input_scale"] = np.asarray(payload["latent_input_scale"])
        arrays["correction_bounds"] = (
            np.asarray(payload["correction_bounds"])
            if "correction_bounds" in payload.files
            else np.ones_like(arrays["latent_scale"])
        )
        arrays["latent_residual_scale"] = (
            np.asarray(payload["latent_residual_scale"])
            if "latent_residual_scale" in payload.files
            else np.ones_like(arrays["latent_scale"])
        )
    return report, config, params, arrays


def _case_plot_paths(root: Path, case_ids: list[str]) -> list[Path]:
    paths = [
        root / "heldout_metric1_summary.png",
        root / "heldout_metric2_summary.png",
        root / "summary.json",
    ]
    for case_id in case_ids:
        case_dir = root / case_id
        paths.extend(
            (
                case_dir / "metric1_energy.png",
                case_dir / "metric2_field_error.png",
                case_dir / "summary.json",
            )
        )
    return paths


def _turnaround_diagnostics(model_energy, teacher_energy, *, cadence: float):
    """Measure damping-minimum to later-peak growth and final retention."""
    teacher_energy = np.asarray(teacher_energy, dtype=np.float64)
    model_energy = np.asarray(model_energy, dtype=np.float64)
    first = min(teacher_energy.size - 2, max(1, int(round(5.0 / cadence))))
    local_minimum = np.zeros(teacher_energy.shape, dtype=bool)
    local_minimum[1:-1] = (
        (teacher_energy[1:-1] <= teacher_energy[:-2])
        & (teacher_energy[1:-1] < teacher_energy[2:])
    )
    local_minimum[:first] = False
    future_maximum = np.maximum.accumulate(teacher_energy[::-1])[::-1]
    tiny = np.finfo(np.float64).tiny
    score = np.full(teacher_energy.shape, -np.inf, dtype=np.float64)
    score[local_minimum] = np.log(np.maximum(future_maximum[local_minimum], tiny)) - np.log(
        np.maximum(teacher_energy[local_minimum], tiny)
    )
    minimum_index = int(np.argmax(score))
    peak_index = minimum_index + int(np.argmax(teacher_energy[minimum_index:]))

    def factor(values):
        return float(values[peak_index] / max(values[minimum_index], tiny))

    def retention(values):
        return float(values[-1] / max(values[peak_index], tiny))

    return {
        "turnaround_start_time": float(minimum_index * cadence),
        "turnaround_peak_time": float(peak_index * cadence),
        "teacher_turnaround_factor": factor(teacher_energy),
        "model_turnaround_factor": factor(model_energy),
        "teacher_post_peak_retention": retention(teacher_energy),
        "model_post_peak_retention": retention(model_energy),
    }


def _full_transition_diagnostics(model_energy, teacher_energy, *, cadence: float):
    """Score every significant envelope damping-to-growth transition."""
    teacher = np.asarray(teacher_energy, dtype=np.float64)
    model = np.asarray(model_energy, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    # A five-time-unit Hann average suppresses carrier-scale plasma oscillations
    # while retaining nonlinear envelope changes.
    width = max(5, int(round(5.0 / cadence)) | 1)
    kernel = np.hanning(width)
    kernel /= np.sum(kernel)
    log_envelope = np.convolve(np.log(np.maximum(teacher, tiny)), kernel, mode="same")
    minima = np.flatnonzero(
        (log_envelope[1:-1] <= log_envelope[:-2])
        & (log_envelope[1:-1] < log_envelope[2:])
    ) + 1
    maxima = np.flatnonzero(
        (log_envelope[1:-1] >= log_envelope[:-2])
        & (log_envelope[1:-1] > log_envelope[2:])
    ) + 1
    transitions = []
    floor = 1.0e-8 * max(float(np.max(teacher)), tiny)
    for start in minima:
        later = maxima[maxima > start]
        if later.size == 0:
            continue
        end = int(later[0])
        teacher_factor = float(teacher[end] / max(teacher[start], tiny))
        duration = float((end - start) * cadence)
        if teacher_factor < 2.0 or teacher[end] < floor or duration < 2.0:
            continue
        transitions.append(
            {
                "start_time": float(start * cadence),
                "peak_time": float(end * cadence),
                "duration": duration,
                "teacher_factor": teacher_factor,
                "model_factor": float(model[end] / max(model[start], tiny)),
                "late_recurrence_candidate": bool(start >= 0.8 * (teacher.size - 1)),
            }
        )
    return {
        "significant_transition_count": len(transitions),
        "significant_transitions": transitions,
    }


def evaluate(
    run_dir: Path,
    *,
    selected_case_ids: tuple[str, ...] = (),
    eval_subdir: str = "heldout_cases",
    reference_cache_override: Path | None = None,
    projected_cache_override: Path | None = None,
    rollout_steps_override: int | None = None,
    checkpoint_name: str = "best_coupled_low_moment_latent.npz",
    float64: bool = False,
) -> None:
    if float64:
        jax.config.update("jax_enable_x64", True)
    report, config, params, arrays = _load_saved_model(run_dir, checkpoint_name)
    if float64:
        def promote(value):
            value = np.asarray(value)
            if np.iscomplexobj(value):
                return value.astype(np.complex128)
            if np.issubdtype(value.dtype, np.floating):
                return value.astype(np.float64)
            return value
        params = {key: jnp.asarray(promote(value)) for key, value in params.items()}
        arrays = {key: promote(value) for key, value in arrays.items()}
    checkpoint_digest = file_digest(run_dir / checkpoint_name)
    model_sources = (Path(__file__), Path("vpml/kinetic_latent.py"),
                     Path("vpml/low_moment.py"), Path("vpml/metrics/early_growth.py"),
                     Path("vpml/metrics/field_error.py"), Path("vpml/metrics/trajectory.py"),
                     Path("model/train/coupled_low_moment_latent.py"))
    source_digests = {str(p.resolve()): file_digest(p) for p in model_sources}
    reference_cache = (
        reference_cache_override
        if reference_cache_override is not None
        else Path(config["reference_cache"])
    )
    projected_cache = (
        projected_cache_override
        if projected_cache_override is not None
        else Path(config["projected_cache"])
    )
    if rollout_steps_override is not None:
        config["rollout_steps"] = int(rollout_steps_override)
    metadata = json.loads((reference_cache / "metadata.json").read_text())
    teacher = metadata["configuration"]
    manifest = load_ic_manifest(reference_cache / "ic_manifest.json")
    cases = [dict(case) for case in manifest["cases"]]
    if selected_case_ids:
        requested = set(selected_case_ids)
        heldout = [case for case in cases if str(case["case_id"]) in requested]
        missing = requested - {str(case["case_id"]) for case in heldout}
        if missing:
            raise ValueError(f"Unknown selected case IDs: {sorted(missing)}")
    else:
        heldout = [
            case
            for regime in REGIMES
            for case in _selected_cases(cases, regime, "heldout")
        ]
    case_ids = [str(case["case_id"]) for case in heldout]
    eval_root = run_dir / eval_subdir
    _assert_outputs_absent(_case_plot_paths(eval_root, case_ids))

    expected_samples = int(config["rollout_steps"]) + 1
    target = np.stack(
        [
            _load_complete_case(
                projected_cache,
                case,
                latent_rank=int(config["latent_rank"]),
                expected_samples=expected_samples,
                dtype=np.float64 if float64 else np.float32,
            )
            for case in heldout
        ]
    )
    model_propagator = (
        _matrix_square_root_propagator(arrays["linear_propagator"])
        if str(config.get("linear_baseline", "fitted_propagator"))
        == "semilinear_strang"
        else arrays["linear_propagator"]
    )
    _, rollout_function, _ = _make_functions(
        propagator=model_propagator,
        basis=arrays["basis"],
        k_arr=arrays["k_arr"],
        resolved_center=arrays["resolved_center"],
        resolved_scale=arrays["resolved_scale"],
        latent_center=arrays["latent_center"],
        latent_scale=arrays["latent_scale"],
        latent_input_scale=arrays.get("latent_input_scale"),
        depth=int(config["depth"]),
        fine_steps=int(config["fine_steps"]),
        fine_dt=float(teacher["teacher_dt"]),
        horizon=int(config["rollout_steps"]),
        poisson_sign=float(teacher["teacher_poisson_sign"]),
        gradient_chunk_steps=int(config["gradient_chunk_steps"]),
        correction_bounds=arrays["correction_bounds"],
        latent_residual_scale=arrays.get(
            "latent_residual_scale", np.ones_like(arrays["latent_scale"])
        ),
        latent_residual_weight=float(config.get("latent_residual_weight", 0.0)),
        latent_state_residual_weight=float(
            config.get("latent_state_residual_weight", 0.0)
        ),
        closure_residual_scale=float(config.get("closure_residual_scale", 1.0)),
        closure_residual_weight=float(config.get("closure_residual_weight", 0.0)),
        closure_correction_bound=float(config.get("closure_correction_bound", 40.0)),
        closure_aligned_output=bool(config.get("closure_aligned_output", False)),
        closure_only_correction=bool(config.get("closure_only_correction", False)),
        closure_readout_mode=str(config.get("closure_readout_mode", "multiplicative")),
        closure_gate_scale=float(config.get("closure_gate_scale", 0.1)),
        closure_gate_power=int(config.get("closure_gate_power", 2)),
        latent_readout_mode=str(config.get("latent_readout_mode", "multiplicative")),
        latent_gate_scale=float(config.get("latent_gate_scale", 0.1)),
        latent_gate_power=int(config.get("latent_gate_power", 2)),
        latent_state_bound=float(config.get("latent_state_bound", 0.0)),
        equilibrium_input_compression_scale=float(
            config.get("equilibrium_input_compression_scale", 0.0)
        ),
        autonomous_latent_weight=float(config.get("autonomous_latent_weight", 0.0)),
        electric_spectrum_weight=float(config.get("electric_spectrum_weight", 0.0)),
        electric_log_energy_weight=float(config.get("electric_log_energy_weight", 0.0)),
        electric_log_energy_floor_ratio=float(
            config.get("electric_log_energy_floor_ratio", 1e-8)
        ),
        electric_chunk_log_growth_weight=float(
            config.get("electric_chunk_log_growth_weight", 0.0)
        ),
        electric_sliding_log_growth_weight=float(
            config.get("electric_sliding_log_growth_weight", 0.0)
        ),
        electric_sliding_log_growth_direction=str(
            config.get("electric_sliding_log_growth_direction", "balanced")
        ),
        electric_growth_window_steps=int(
            config.get("electric_growth_window_steps", 100)
        ),
        electric_time_relative_weight=float(
            config.get("electric_time_relative_weight", 0.0)
        ),
        electric_transfer_weight=float(config.get("electric_transfer_weight", 0.0)),
        electric_time_relative_floor_ratio=float(
            config.get("electric_time_relative_floor_ratio", 1e-4)
        ),
        post_minimum_log_energy_weight=float(
            config.get("post_minimum_log_energy_weight", 0.0)
        ),
        post_minimum_field_weight=float(
            config.get("post_minimum_field_weight", 0.0)
        ),
        turnaround_field_weight=float(config.get("turnaround_field_weight", 0.0)),
        post_peak_log_energy_weight=float(
            config.get("post_peak_log_energy_weight", 0.0)
        ),
        peak_log_energy_weight=float(config.get("peak_log_energy_weight", 0.0)),
        turnaround_log_ratio_weight=float(
            config.get("turnaround_log_ratio_weight", 0.0)
        ),
        post_peak_retention_weight=float(
            config.get("post_peak_retention_weight", 0.0)
        ),
        post_peak_retention_floor=float(
            config.get("post_peak_retention_floor", 0.0)
        ),
        post_minimum_energy_modes=int(config.get("post_minimum_energy_modes", 4)),
        teacher_residual_stride=int(config.get("teacher_residual_stride", 50)),
        teacher_rollout_steps=int(config.get("teacher_rollout_steps", 10)),
        linear_baseline=str(config.get("linear_baseline", "fitted_propagator")),
        hermite_tail_damping=float(config.get("hermite_tail_damping", 10.0)),
        hermite_tail_power=float(config.get("hermite_tail_power", 6.0)),
        semilinear_kick_scale=float(config.get("semilinear_kick_scale", 1.0)),
        semilinear_correction_location=str(
            config.get("semilinear_correction_location", "midpoint")
        ),
        latent_delay_input=bool(config.get("latent_delay_input", False)),
    )
    predicted_resolved, predicted_latent = jax.jit(rollout_function)(
        params,
        jnp.asarray(target[:, 0, :3]),
        jnp.asarray(target[:, 0, 3:]),
    )
    predicted_resolved = np.asarray(predicted_resolved, dtype=np.float64)
    predicted_latent = np.asarray(predicted_latent, dtype=np.float64)
    if not (
        np.all(np.isfinite(predicted_resolved))
        and np.all(np.isfinite(predicted_latent))
    ):
        raise FloatingPointError("Saved model produced a non-finite held-out rollout")

    initial_density = target[:, 0, 0].astype(np.float64 if float64 else np.float32)
    density_history = np.concatenate(
        (initial_density[:, None], predicted_resolved[:, :, 0]), axis=1
    )
    field = np.asarray(
        electric_field_from_density(
            jnp.asarray(density_history.reshape(-1, int(config["nx"]))),
            jnp.asarray(arrays["k_arr"], dtype=jnp.float64 if float64 else jnp.float32),
            poisson_sign=float(teacher["teacher_poisson_sign"]),
        )
    ).reshape(len(heldout), expected_samples, int(config["nx"]))
    learned_hat = np.fft.rfft(field, axis=-1)
    times = np.arange(expected_samples, dtype=np.float64) * float(config["cadence"])
    dx = float(teacher["teacher_L"]) / int(config["nx"])
    learned_energy = _electric_field_energy(
        learned_hat, nx=int(config["nx"]), dx=dx
    )

    growth_metric = EarlyElectricFieldGrowthMetric(
        EarlyGrowthConfig(sample_selector="local_maxima")
    )
    summaries = []
    metric1_traces = []
    metric2_traces = []
    eval_root.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(heldout):
        case_id = str(case["case_id"])
        regime = str(case["regime"])
        _, _, snapshot_path = case_shard_paths(reference_cache, case_id)
        with np.load(snapshot_path, allow_pickle=False) as reference:
            hr_times = np.asarray(reference["E_hat_hist_times"], dtype=np.float64)
            hr_hat = np.asarray(reference["E_hat_hist"], dtype=np.complex128)
            hr_energy = np.asarray(reference["energy"], dtype=np.float64)
            hr_k = np.asarray(reference["k_arr"], dtype=np.float64)
        restricted_hr_hat = _restrict_rfft(
            hr_hat, int(teacher["teacher_Nx"]), int(config["nx"])
        )
        growth = growth_metric.compare(
            times,
            learned_energy[index],
            hr_times,
            hr_energy,
        )
        field_metric = SelfGeneratedFieldErrorMetric(
            FieldErrorConfig(final_time=float(times[-1]))
        )
        field_error = field_metric.evaluate_fourier(
            times,
            learned_hat[index],
            arrays["k_arr"],
            hr_times,
            restricted_hr_hat,
            arrays["k_arr"],
        )
        comparison = field_metric.prepare_fourier_comparison(
            times,
            learned_hat[index],
            arrays["k_arr"],
            hr_times,
            restricted_hr_hat,
            arrays["k_arr"],
        )
        relative_field = np.sqrt(
            np.sum(np.abs(comparison.E_hat_theta - comparison.E_hat_hr) ** 2, axis=-1)
            / np.maximum(np.sum(np.abs(comparison.E_hat_hr) ** 2, axis=-1), 1e-30)
        )
        state = predicted_resolved[index]
        density = 1.0 + state[:, 0]
        momentum = state[:, 1]
        second = state[:, 0] + math.sqrt(2.0) * state[:, 2]
        pressure = 1.0 + second - momentum * momentum / density
        normalized_latent = predicted_latent[index] / arrays["latent_scale"][None, :, None]
        target_state = resolved_hermite_to_low_moment_state(
            jnp.asarray(target[index, :, :3])
        )
        comparison_model_field = field[index]
        comparison_teacher_field = np.asarray(
            primitive_fields(
                target_state,
                jnp.asarray(arrays["k_arr"]),
                poisson_sign=float(teacher["teacher_poisson_sign"]),
            )[:, 3]
        )
        regrowth = _energy_regrowth_diagnostics(
            comparison_model_field[1:],
            comparison_teacher_field[1:],
            cadence=float(config["cadence"]),
        )
        comparison_model_energy = np.mean(
            np.square(comparison_model_field), axis=-1
        )
        comparison_teacher_energy = np.mean(
            np.square(comparison_teacher_field), axis=-1
        )
        end_index = min(
            comparison_model_energy.size - 1,
            int(round(regrowth["regrowth_end_time"] / float(config["cadence"]))),
        )
        model_post_rebound_retention = float(
            comparison_model_energy[-1]
            / max(comparison_model_energy[end_index], np.finfo(np.float64).tiny)
        )
        teacher_post_rebound_retention = float(
            comparison_teacher_energy[-1]
            / max(comparison_teacher_energy[end_index], np.finfo(np.float64).tiny)
        )
        turnaround = _turnaround_diagnostics(
            comparison_model_energy,
            comparison_teacher_energy,
            cadence=float(config["cadence"]),
        )
        transitions = _full_transition_diagnostics(
            comparison_model_energy,
            comparison_teacher_energy,
            cadence=float(config["cadence"]),
        )
        summary = {
            "case_id": case_id,
            "regime": regime,
            "bounded_to_final_time": True,
            "epsilon_grow": _json_float(growth.epsilon_grow),
            "gamma_hr": _json_float(growth.gamma_grow_hr),
            "gamma_theta": _json_float(growth.gamma_grow_theta),
            **matched_cadence_growth(times, growth.gamma_grow_theta, hr_times, hr_energy,
                                     fit_window=(growth.t_a, growth.t_b)),
            "epsilon_E": _json_float(field_error.epsilon_E),
            "epsilon_E_final_time": float(field_error.T),
            "minimum_density": _json_float(np.min(density)),
            "minimum_pressure": _json_float(np.min(pressure)),
            "maximum_normalized_latent": _json_float(
                np.max(np.abs(normalized_latent))
            ),
            "latent_excursion_fraction_above_96": float(
                np.mean(np.abs(normalized_latent) >= 96.0)
            ),
            "latent_saturation": latent_saturation_diagnostics(
                normalized_latent,
                bound=float(config.get("latent_state_bound", 0.0)),
                cadence=float(config["cadence"]),
                initial_time=float(config["cadence"]),
            ),
            "full_envelope": envelope_diagnostics(
                comparison_model_energy, comparison_teacher_energy,
                cadence=float(config["cadence"]),
            ),
            **regrowth,
            "model_post_rebound_retention": model_post_rebound_retention,
            "teacher_post_rebound_retention": teacher_post_rebound_retention,
            **turnaround,
            **transitions,
        }
        case_dir = eval_root / case_id
        case_dir.mkdir(exist_ok=True)

        fig, axis = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
        axis.semilogy(hr_times, hr_energy, color="#2463eb", label="kinetic teacher")
        axis.semilogy(times, learned_energy[index], color="#6f3cc3", label="coupled latent model")
        axis.set_xlabel("t")
        axis.set_ylabel("Electric-field energy")
        axis.set_title(f"{case_id} ({regime})")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.savefig(case_dir / "metric1_energy.png", dpi=200)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
        axis.semilogy(comparison.times, relative_field, color="#c44e52")
        axis.set_xlabel("t")
        axis.set_ylabel("Relative electric-field error")
        axis.set_title(f"{case_id}: integrated $\\varepsilon_E={field_error.epsilon_E:.3e}$")
        axis.grid(alpha=0.25)
        fig.savefig(case_dir / "metric2_field_error.png", dpi=200)
        plt.close(fig)
        (case_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        summaries.append(summary)
        metric1_traces.append((case_id, hr_times, hr_energy, learned_energy[index]))
        metric2_traces.append((case_id, comparison.times, relative_field))
        print(
            f"[eval] {case_id}: epsilon_grow={summary['epsilon_grow']:.3e} "
            f"epsilon_E={summary['epsilon_E']:.3e}",
            flush=True,
        )

    fig, axes = plt.subplots(
        len(metric1_traces),
        1,
        figsize=(10.0, 2.0 * len(metric1_traces)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for index, (case_id, hr_times, hr_energy, model_energy) in enumerate(metric1_traces):
        axes[index].semilogy(hr_times, hr_energy, color="#2463eb", label="kinetic teacher")
        axes[index].semilogy(times, model_energy, color="#6f3cc3", label="coupled latent model")
        axes[index].set_ylabel(case_id, rotation=0, ha="right", va="center", fontsize=8)
        axes[index].grid(alpha=0.2)
        if index == 0:
            axes[index].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("t")
    fig.suptitle("Held-out electric-field energy (Metric 1)")
    fig.savefig(eval_root / "heldout_metric1_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(metric2_traces),
        1,
        figsize=(10.0, 2.0 * len(metric2_traces)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for index, (case_id, comparison_times, relative_field) in enumerate(metric2_traces):
        axes[index].semilogy(comparison_times, relative_field, color="#c44e52")
        axes[index].set_ylabel(case_id, rotation=0, ha="right", va="center", fontsize=8)
        axes[index].grid(alpha=0.2)
    axes[-1].set_xlabel("t")
    fig.suptitle("Held-out relative electric-field error (Metric 2)")
    fig.savefig(eval_root / "heldout_metric2_summary.png", dpi=180)
    plt.close(fig)

    aggregate = {
        "source_checkpoint": str(
            (run_dir / checkpoint_name).resolve()
        ),
        "source_checkpoint_sha256": checkpoint_digest,
        "evaluation_precision": "float64" if float64 else "legacy",
        "source_sha256": source_digests,
        "reference_manifest_sha256": file_digest(reference_cache / "ic_manifest.json"),
        "projected_metadata_sha256": file_digest(projected_cache / "metadata.json"),
        "metric_protocol": {
            "metric1": "early_growth_local_maxima_native_reference_times",
            "matched_metric1": "reference_energy_interpolated_to_model_times_native_fit_window",
            "metric2": "time_integrated_relative_field_l2",
            "full_envelope": "projected_reference_trailing_mean_2.5_5_10_floor1e-8",
        },
        "case_count": len(summaries),
        "bounded_cases": sum(row["bounded_to_final_time"] for row in summaries),
        "cases": summaries,
    }
    (eval_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--eval-subdir", default="heldout_cases")
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--projected-cache", type=Path)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--checkpoint-name", default="best_coupled_low_moment_latent.npz")
    parser.add_argument("--float64", action="store_true", help="Use float64 parameters, state and field arithmetic.")
    args = parser.parse_args()
    print_jax_runtime_summary(jax, context="coupled low-moment latent evaluation")
    selected_case_ids = tuple(value for value in args.case_ids.split(",") if value)
    evaluate(
        args.run_dir,
        selected_case_ids=selected_case_ids,
        eval_subdir=args.eval_subdir,
        reference_cache_override=args.reference_cache,
        projected_cache_override=args.projected_cache,
        rollout_steps_override=args.rollout_steps,
        checkpoint_name=args.checkpoint_name,
        float64=args.float64,
    )


if __name__ == "__main__":
    main()
