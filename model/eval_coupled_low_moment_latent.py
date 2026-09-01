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
from vpml.kinetic_latent import init_state_conditioned_latent_operator
from vpml.low_moment import electric_field_from_density
from vpml.metrics import (
    EarlyElectricFieldGrowthMetric,
    EarlyGrowthConfig,
    FieldErrorConfig,
    SelfGeneratedFieldErrorMetric,
)


def _assert_outputs_absent(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing evaluation outputs: {joined}")


def _load_saved_model(run_dir: Path):
    report = json.loads((run_dir / "report.json").read_text())
    config = dict(report["configuration"])
    checkpoint_path = run_dir / "best_coupled_low_moment_latent.npz"
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


def evaluate(run_dir: Path) -> None:
    report, config, params, arrays = _load_saved_model(run_dir)
    reference_cache = Path(config["reference_cache"])
    projected_cache = Path(config["projected_cache"])
    metadata = json.loads((reference_cache / "metadata.json").read_text())
    teacher = metadata["configuration"]
    manifest = load_ic_manifest(reference_cache / "ic_manifest.json")
    cases = [dict(case) for case in manifest["cases"]]
    heldout = [
        case
        for regime in REGIMES
        for case in _selected_cases(cases, regime, "heldout")
    ]
    case_ids = [str(case["case_id"]) for case in heldout]
    eval_root = run_dir / "heldout_cases"
    _assert_outputs_absent(_case_plot_paths(eval_root, case_ids))

    expected_samples = int(config["rollout_steps"]) + 1
    target = np.stack(
        [
            _load_complete_case(
                projected_cache,
                case,
                latent_rank=int(config["latent_rank"]),
                expected_samples=expected_samples,
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
        autonomous_latent_weight=float(config.get("autonomous_latent_weight", 0.0)),
        electric_spectrum_weight=float(config.get("electric_spectrum_weight", 0.0)),
        electric_log_energy_weight=float(config.get("electric_log_energy_weight", 0.0)),
        electric_log_energy_floor_ratio=float(
            config.get("electric_log_energy_floor_ratio", 1e-8)
        ),
        electric_chunk_log_growth_weight=float(
            config.get("electric_chunk_log_growth_weight", 0.0)
        ),
        electric_growth_window_steps=int(
            config.get("electric_growth_window_steps", 100)
        ),
        electric_time_relative_weight=float(
            config.get("electric_time_relative_weight", 0.0)
        ),
        electric_time_relative_floor_ratio=float(
            config.get("electric_time_relative_floor_ratio", 1e-4)
        ),
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

    initial_density = target[:, 0, 0].astype(np.float32)
    density_history = np.concatenate(
        (initial_density[:, None], predicted_resolved[:, :, 0]), axis=1
    )
    field = np.asarray(
        electric_field_from_density(
            jnp.asarray(density_history.reshape(-1, int(config["nx"]))),
            jnp.asarray(arrays["k_arr"], dtype=jnp.float32),
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
        summary = {
            "case_id": case_id,
            "regime": regime,
            "bounded_to_final_time": True,
            "epsilon_grow": _json_float(growth.epsilon_grow),
            "gamma_hr": _json_float(growth.gamma_grow_hr),
            "gamma_theta": _json_float(growth.gamma_grow_theta),
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
            (run_dir / "best_coupled_low_moment_latent.npz").resolve()
        ),
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
    args = parser.parse_args()
    print_jax_runtime_summary(jax, context="coupled low-moment latent evaluation")
    evaluate(args.run_dir)


if __name__ == "__main__":
    main()
