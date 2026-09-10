"""Probe gradient alignment of low-mode electric-energy transfer supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from model.eval_coupled_low_moment_latent import _load_saved_model
from model.train.coupled_low_moment_latent import (
    _load_complete_case,
    _make_functions,
    _turnaround_indices,
)
from model.train.interface_flux_data import load_ic_manifest
from vpml.kinetic_latent import resolved_hermite_to_low_moment_state
from vpml.low_moment import primitive_fields


def _tree_dot(left, right):
    return sum(jnp.vdot(a, b).real for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)))


def _tree_norm(tree):
    return jnp.sqrt(jnp.maximum(_tree_dot(tree, tree), 0.0))


def _gradient_cosine(left, right):
    return _tree_dot(left, right) / jnp.maximum(
        _tree_norm(left) * _tree_norm(right), jnp.finfo(jnp.float32).tiny
    )


def _tail_output_gradient(tree, start: int):
    masked = {key: jnp.zeros_like(value) for key, value in tree.items()}
    if "output_kernel" in tree:
        masked["output_kernel"] = masked["output_kernel"].at[start:].set(
            tree["output_kernel"][start:]
        )
    if "output_bias" in tree:
        masked["output_bias"] = masked["output_bias"].at[start:].set(
            tree["output_bias"][start:]
        )
    return masked


def _output_row_alignment(reference, candidate):
    """Return per-output-row dot products and cosines for two gradients."""
    left_kernel = reference["output_kernel"].reshape(reference["output_kernel"].shape[0], -1)
    right_kernel = candidate["output_kernel"].reshape(candidate["output_kernel"].shape[0], -1)
    left = jnp.concatenate((left_kernel, reference["output_bias"][:, None]), axis=1)
    right = jnp.concatenate((right_kernel, candidate["output_bias"][:, None]), axis=1)
    dot = jnp.sum(left * right, axis=1)
    denominator = jnp.sqrt(jnp.sum(jnp.square(left), axis=1) * jnp.sum(jnp.square(right), axis=1))
    cosine = dot / jnp.maximum(denominator, jnp.finfo(left.dtype).tiny)
    return {
        "dot": np.asarray(dot).tolist(),
        "cosine": np.asarray(cosine).tolist(),
        "reference_norm": np.asarray(jnp.linalg.norm(left, axis=1)).tolist(),
        "candidate_norm": np.asarray(jnp.linalg.norm(right, axis=1)).tolist(),
    }


def _low_mode_transfer_loss(
    params, trajectory, rollout, k_arr, *, poisson_sign: float, maximum_mode: int
):
    predicted_resolved, _ = rollout(params, trajectory[:, 0, :3], trajectory[:, 0, 3:])
    predicted_resolved = jnp.concatenate(
        (trajectory[:, :1, :3], predicted_resolved), axis=1
    )

    def transfer(resolved):
        shape = resolved.shape
        state = resolved_hermite_to_low_moment_state(
            resolved.reshape(-1, 3, shape[-1])
        )
        fields = primitive_fields(state, k_arr, poisson_sign=poisson_sign)
        electric = fields[:, 3].reshape(shape[0], shape[1], shape[-1])
        momentum = state[:, 1].reshape(shape[0], shape[1], shape[-1])
        electric_hat = jnp.fft.rfft(electric, axis=-1, norm="forward")
        momentum_hat = jnp.fft.rfft(momentum, axis=-1, norm="forward")
        modal = 2.0 * jnp.real(
            electric_hat[..., 1 : maximum_mode + 1]
            * jnp.conj(momentum_hat[..., 1 : maximum_mode + 1])
        )
        energy = jnp.mean(jnp.square(electric), axis=-1)
        return modal, energy

    predicted_transfer, _ = transfer(predicted_resolved)
    target_transfer, target_energy = transfer(trajectory[:, :, :3])
    positive_growth = target_energy[:, 1:] > target_energy[:, :-1]
    scale = jnp.maximum(
        jnp.mean(jnp.square(target_transfer), axis=(1, 2), keepdims=True),
        jnp.finfo(target_transfer.dtype).tiny,
    )
    time_weights = positive_growth.astype(target_transfer.dtype)
    time_weights = time_weights / jnp.maximum(jnp.sum(time_weights, axis=1, keepdims=True), 1.0)
    error = jnp.mean(
        jnp.square(predicted_transfer[:, 1:] - target_transfer[:, 1:]) / scale,
        axis=-1,
    )
    return jnp.mean(jnp.sum(time_weights * error, axis=1))


def _low_mode_regrowth_field_loss(
    params, trajectory, rollout, k_arr, *, poisson_sign: float, maximum_mode: int
):
    predicted_resolved, _ = rollout(params, trajectory[:, 0, :3], trajectory[:, 0, 3:])
    predicted_resolved = jnp.concatenate((trajectory[:, :1, :3], predicted_resolved), axis=1)

    def field_and_energy(resolved):
        shape = resolved.shape
        state = resolved_hermite_to_low_moment_state(resolved.reshape(-1, 3, shape[-1]))
        field = primitive_fields(state, k_arr, poisson_sign=poisson_sign)[:, 3]
        field = field.reshape(shape[0], shape[1], shape[-1])
        modes = jnp.fft.rfft(field, axis=-1, norm="forward")[..., 1 : maximum_mode + 1]
        return modes, jnp.sum(jnp.square(jnp.abs(modes)), axis=-1)

    predicted_modes, _ = field_and_energy(predicted_resolved)
    target_modes, target_energy = field_and_energy(trajectory[:, :, :3])
    minimum_index, _ = _turnaround_indices(target_energy)
    time_index = jnp.arange(target_energy.shape[1])[None, :]
    mask = (time_index >= minimum_index[:, None]).astype(target_energy.dtype)
    scale = jnp.maximum(
        jnp.mean(jnp.square(jnp.abs(target_modes)), axis=(1, 2), keepdims=True),
        jnp.finfo(target_energy.dtype).tiny,
    )
    error = jnp.mean(jnp.square(jnp.abs(predicted_modes - target_modes)) / scale, axis=-1)
    mask = mask / jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    return jnp.mean(jnp.sum(mask * error, axis=1))


def _low_mode_post_minimum_log_energy_loss(
    params, trajectory, rollout, k_arr, *, poisson_sign: float, maximum_mode: int
):
    predicted_resolved, _ = rollout(params, trajectory[:, 0, :3], trajectory[:, 0, 3:])
    predicted_resolved = jnp.concatenate((trajectory[:, :1, :3], predicted_resolved), axis=1)

    def energy(resolved):
        shape = resolved.shape
        state = resolved_hermite_to_low_moment_state(resolved.reshape(-1, 3, shape[-1]))
        field = primitive_fields(state, k_arr, poisson_sign=poisson_sign)[:, 3]
        field = field.reshape(shape[0], shape[1], shape[-1])
        modes = jnp.fft.rfft(field, axis=-1, norm="forward")[..., 1 : maximum_mode + 1]
        return jnp.sum(jnp.square(jnp.abs(modes)), axis=-1)

    predicted_energy = energy(predicted_resolved)
    target_energy = energy(trajectory[:, :, :3])
    minimum_index, _ = _turnaround_indices(target_energy)
    time_index = jnp.arange(target_energy.shape[1])[None, :]
    mask = (time_index >= minimum_index[:, None]).astype(target_energy.dtype)
    floor = jnp.maximum(
        1e-6 * jnp.max(target_energy, axis=1, keepdims=True),
        jnp.finfo(target_energy.dtype).tiny,
    )
    error = jnp.square(
        jnp.log(predicted_energy + floor) - jnp.log(target_energy + floor)
    )
    mask = mask / jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    return jnp.mean(jnp.sum(mask * error, axis=1))


def _low_mode_post_peak_log_energy_loss(
    params, trajectory, rollout, k_arr, *, poisson_sign: float, maximum_mode: int
):
    predicted_resolved, _ = rollout(params, trajectory[:, 0, :3], trajectory[:, 0, 3:])
    predicted_resolved = jnp.concatenate((trajectory[:, :1, :3], predicted_resolved), axis=1)

    def energy(resolved):
        shape = resolved.shape
        state = resolved_hermite_to_low_moment_state(resolved.reshape(-1, 3, shape[-1]))
        field = primitive_fields(state, k_arr, poisson_sign=poisson_sign)[:, 3]
        field = field.reshape(shape[0], shape[1], shape[-1])
        modes = jnp.fft.rfft(field, axis=-1, norm="forward")[..., 1 : maximum_mode + 1]
        return jnp.sum(jnp.square(jnp.abs(modes)), axis=-1)

    predicted_energy = energy(predicted_resolved)
    target_energy = energy(trajectory[:, :, :3])
    time_index = jnp.arange(target_energy.shape[1])[None, :]
    minimum_index, peak_index = _turnaround_indices(target_energy)
    mask = (time_index >= peak_index[:, None]).astype(target_energy.dtype)
    floor = jnp.maximum(
        1e-6 * jnp.max(target_energy, axis=1, keepdims=True),
        jnp.finfo(target_energy.dtype).tiny,
    )
    error = jnp.square(
        jnp.log(predicted_energy + floor) - jnp.log(target_energy + floor)
    )
    mask = mask / jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    return jnp.mean(jnp.sum(mask * error, axis=1))


def _low_mode_turnaround_log_factor_loss(
    params, trajectory, rollout, k_arr, *, poisson_sign: float, maximum_mode: int
):
    """Match the teacher minimum-to-peak low-mode energy amplification."""
    predicted_resolved, _ = rollout(params, trajectory[:, 0, :3], trajectory[:, 0, 3:])
    predicted_resolved = jnp.concatenate((trajectory[:, :1, :3], predicted_resolved), axis=1)

    def energy(resolved):
        shape = resolved.shape
        state = resolved_hermite_to_low_moment_state(resolved.reshape(-1, 3, shape[-1]))
        field = primitive_fields(state, k_arr, poisson_sign=poisson_sign)[:, 3]
        field = field.reshape(shape[0], shape[1], shape[-1])
        modes = jnp.fft.rfft(field, axis=-1, norm="forward")[..., 1 : maximum_mode + 1]
        return jnp.sum(jnp.square(jnp.abs(modes)), axis=-1)

    predicted_energy = energy(predicted_resolved)
    target_energy = energy(trajectory[:, :, :3])
    minimum_index, peak_index = _turnaround_indices(target_energy)
    case_index = jnp.arange(target_energy.shape[0])
    floor = jnp.maximum(
        1e-6 * jnp.max(target_energy, axis=1),
        jnp.finfo(target_energy.dtype).tiny,
    )

    def log_factor(values):
        return jnp.log(values[case_index, peak_index] + floor) - jnp.log(
            values[case_index, minimum_index] + floor
        )

    return jnp.mean(jnp.square(log_factor(predicted_energy) - log_factor(target_energy)))


def _low_mode_peak_log_energy_loss(
    params, trajectory, rollout, k_arr, *, poisson_sign: float, maximum_mode: int
):
    """Match absolute low-mode energy at the teacher turnaround peak."""
    predicted_resolved, _ = rollout(params, trajectory[:, 0, :3], trajectory[:, 0, 3:])
    predicted_resolved = jnp.concatenate((trajectory[:, :1, :3], predicted_resolved), axis=1)

    def energy(resolved):
        shape = resolved.shape
        state = resolved_hermite_to_low_moment_state(resolved.reshape(-1, 3, shape[-1]))
        field = primitive_fields(state, k_arr, poisson_sign=poisson_sign)[:, 3]
        field = field.reshape(shape[0], shape[1], shape[-1])
        modes = jnp.fft.rfft(field, axis=-1, norm="forward")[..., 1 : maximum_mode + 1]
        return jnp.sum(jnp.square(jnp.abs(modes)), axis=-1)

    predicted_energy = energy(predicted_resolved)
    target_energy = energy(trajectory[:, :, :3])
    _, peak_index = _turnaround_indices(target_energy)
    case_index = jnp.arange(target_energy.shape[0])
    floor = jnp.maximum(
        1e-6 * jnp.max(target_energy, axis=1),
        jnp.finfo(target_energy.dtype).tiny,
    )
    return jnp.mean(
        jnp.square(
            jnp.log(predicted_energy[case_index, peak_index] + floor)
            - jnp.log(target_energy[case_index, peak_index] + floor)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--maximum-mode", type=int, default=4)
    parser.add_argument("--case-limit", type=int, default=2)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--latent-state-bound", type=float, default=0.0)
    parser.add_argument("--tail-output-from", type=int, default=0)
    parser.add_argument("--row-alignment-only", action="store_true")
    parser.add_argument(
        "--regime",
        choices=("nonlinear_landau_weak", "nonlinear_landau_strong"),
        default="nonlinear_landau_strong",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report, config, params, arrays = _load_saved_model(args.run_dir)
    reference_cache = Path(config["reference_cache"])
    projected_cache = Path(config["projected_cache"])
    teacher = json.loads((reference_cache / "metadata.json").read_text())["configuration"]
    cases = load_ic_manifest(reference_cache / "ic_manifest.json")["cases"]
    selected = [
        dict(case)
        for case in cases
        if case["split"] == "train" and case["regime"] == args.regime
    ]
    selected = selected[
        int(args.case_offset) : int(args.case_offset) + int(args.case_limit)
    ]
    if not selected:
        raise ValueError("case selection is empty")
    expected_samples = int(config["rollout_steps"]) + 1
    trajectory = jnp.asarray(
        np.stack(
            [
                _load_complete_case(
                    projected_cache,
                    case,
                    latent_rank=int(config["latent_rank"]),
                    expected_samples=expected_samples,
                )
                for case in selected
            ]
        )
    )
    loss, rollout, components = _make_functions(
        propagator=arrays["linear_propagator"], basis=arrays["basis"], k_arr=arrays["k_arr"],
        resolved_center=arrays["resolved_center"], resolved_scale=arrays["resolved_scale"],
        latent_center=arrays["latent_center"], latent_scale=arrays["latent_scale"],
        depth=int(config["depth"]), fine_steps=int(config["fine_steps"]),
        fine_dt=float(teacher["teacher_dt"]), horizon=int(config["rollout_steps"]),
        poisson_sign=float(teacher["teacher_poisson_sign"]),
        gradient_chunk_steps=int(config["gradient_chunk_steps"]),
        correction_bounds=arrays["correction_bounds"],
        latent_residual_scale=arrays["latent_residual_scale"],
        latent_residual_weight=float(config["latent_residual_weight"]),
        latent_state_residual_weight=float(config["latent_state_residual_weight"]),
        closure_residual_scale=float(config["closure_residual_scale"]),
        closure_residual_weight=float(config["closure_residual_weight"]),
        closure_correction_bound=float(config["closure_correction_bound"]),
        closure_aligned_output=bool(config["closure_aligned_output"]),
        closure_only_correction=bool(config["closure_only_correction"]),
        closure_readout_mode=str(config["closure_readout_mode"]),
        closure_gate_scale=float(config["closure_gate_scale"]),
        closure_gate_power=int(config["closure_gate_power"]),
        latent_readout_mode=str(config["latent_readout_mode"]),
        latent_gate_scale=float(config["latent_gate_scale"]),
        latent_gate_power=int(config["latent_gate_power"]),
        latent_state_bound=float(args.latent_state_bound),
        equilibrium_input_compression_scale=float(config.get("equilibrium_input_compression_scale", 0.0)),
        autonomous_latent_weight=float(config["autonomous_latent_weight"]),
        electric_spectrum_weight=float(config["electric_spectrum_weight"]),
        electric_log_energy_weight=float(config["electric_log_energy_weight"]),
        electric_log_energy_floor_ratio=float(config["electric_log_energy_floor_ratio"]),
        electric_chunk_log_growth_weight=float(config["electric_chunk_log_growth_weight"]),
        electric_growth_window_steps=int(config["electric_growth_window_steps"]),
        electric_time_relative_weight=float(config["electric_time_relative_weight"]),
        electric_time_relative_floor_ratio=float(config["electric_time_relative_floor_ratio"]),
        teacher_residual_stride=int(config["teacher_residual_stride"]),
        teacher_rollout_steps=int(config["teacher_rollout_steps"]),
        linear_baseline=str(config["linear_baseline"]),
        hermite_tail_damping=float(config["hermite_tail_damping"]),
        hermite_tail_power=float(config["hermite_tail_power"]),
        semilinear_kick_scale=float(config["semilinear_kick_scale"]),
        semilinear_correction_location=str(config["semilinear_correction_location"]),
        latent_delay_input=bool(config["latent_delay_input"]),
    )
    del loss, report
    physical_value, physical_grad = jax.value_and_grad(lambda p: components(p, trajectory)[1])(params)
    if args.row_alignment_only:
        post_peak_value, post_peak_grad = jax.value_and_grad(
            lambda p: _low_mode_post_peak_log_energy_loss(
                p, trajectory, rollout, jnp.asarray(arrays["k_arr"]),
                poisson_sign=float(teacher["teacher_poisson_sign"]),
                maximum_mode=args.maximum_mode,
            )
        )(params)
        peak_value, peak_grad = jax.value_and_grad(
            lambda p: _low_mode_peak_log_energy_loss(
                p, trajectory, rollout, jnp.asarray(arrays["k_arr"]),
                poisson_sign=float(teacher["teacher_poisson_sign"]),
                maximum_mode=args.maximum_mode,
            )
        )(params)
        result = {
            "case_ids": [case["case_id"] for case in selected],
            "maximum_mode": args.maximum_mode,
            "case_offset": int(args.case_offset),
            "regime": args.regime,
            "physical_loss": float(physical_value),
            "post_peak_log_energy_loss": float(post_peak_value),
            "peak_log_energy_loss": float(peak_value),
            "post_peak_output_row_alignment": _output_row_alignment(
                physical_grad, post_peak_grad
            ),
            "peak_output_row_alignment": _output_row_alignment(
                physical_grad, peak_grad
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return
    transfer_value, transfer_grad = jax.value_and_grad(
        lambda p: _low_mode_transfer_loss(
            p, trajectory, rollout, jnp.asarray(arrays["k_arr"]),
            poisson_sign=float(teacher["teacher_poisson_sign"]), maximum_mode=args.maximum_mode,
        )
    )(params)
    field_value, field_grad = jax.value_and_grad(
        lambda p: _low_mode_regrowth_field_loss(
            p, trajectory, rollout, jnp.asarray(arrays["k_arr"]),
            poisson_sign=float(teacher["teacher_poisson_sign"]), maximum_mode=args.maximum_mode,
        )
    )(params)
    energy_value, energy_grad = jax.value_and_grad(
        lambda p: _low_mode_post_minimum_log_energy_loss(
            p, trajectory, rollout, jnp.asarray(arrays["k_arr"]),
            poisson_sign=float(teacher["teacher_poisson_sign"]), maximum_mode=args.maximum_mode,
        )
    )(params)
    post_peak_value, post_peak_grad = jax.value_and_grad(
        lambda p: _low_mode_post_peak_log_energy_loss(
            p, trajectory, rollout, jnp.asarray(arrays["k_arr"]),
            poisson_sign=float(teacher["teacher_poisson_sign"]), maximum_mode=args.maximum_mode,
        )
    )(params)
    factor_value, factor_grad = jax.value_and_grad(
        lambda p: _low_mode_turnaround_log_factor_loss(
            p, trajectory, rollout, jnp.asarray(arrays["k_arr"]),
            poisson_sign=float(teacher["teacher_poisson_sign"]), maximum_mode=args.maximum_mode,
        )
    )(params)
    peak_value, peak_grad = jax.value_and_grad(
        lambda p: _low_mode_peak_log_energy_loss(
            p, trajectory, rollout, jnp.asarray(arrays["k_arr"]),
            poisson_sign=float(teacher["teacher_poisson_sign"]), maximum_mode=args.maximum_mode,
        )
    )(params)
    if int(args.tail_output_from) > 0:
        physical_grad = _tail_output_gradient(physical_grad, int(args.tail_output_from))
        transfer_grad = _tail_output_gradient(transfer_grad, int(args.tail_output_from))
        field_grad = _tail_output_gradient(field_grad, int(args.tail_output_from))
        energy_grad = _tail_output_gradient(energy_grad, int(args.tail_output_from))
        post_peak_grad = _tail_output_gradient(post_peak_grad, int(args.tail_output_from))
        factor_grad = _tail_output_gradient(factor_grad, int(args.tail_output_from))
        peak_grad = _tail_output_gradient(peak_grad, int(args.tail_output_from))
    result = {
        "case_ids": [case["case_id"] for case in selected],
        "maximum_mode": args.maximum_mode,
        "tail_output_from": int(args.tail_output_from),
        "regime": args.regime,
        "physical_loss": float(physical_value),
        "transfer_loss": float(transfer_value),
        "physical_gradient_norm": float(_tree_norm(physical_grad)),
        "transfer_gradient_norm": float(_tree_norm(transfer_grad)),
        "gradient_cosine": float(_gradient_cosine(physical_grad, transfer_grad)),
        "regrowth_field_loss": float(field_value),
        "regrowth_field_gradient_norm": float(_tree_norm(field_grad)),
        "regrowth_field_gradient_cosine": float(_gradient_cosine(physical_grad, field_grad)),
        "post_minimum_log_energy_loss": float(energy_value),
        "post_minimum_log_energy_gradient_norm": float(_tree_norm(energy_grad)),
        "post_minimum_log_energy_gradient_cosine": float(_gradient_cosine(physical_grad, energy_grad)),
        "post_peak_log_energy_loss": float(post_peak_value),
        "post_peak_log_energy_gradient_norm": float(_tree_norm(post_peak_grad)),
        "post_peak_log_energy_gradient_cosine": float(_gradient_cosine(physical_grad, post_peak_grad)),
        "turnaround_log_factor_loss": float(factor_value),
        "turnaround_log_factor_gradient_norm": float(_tree_norm(factor_grad)),
        "turnaround_log_factor_gradient_cosine": float(_gradient_cosine(physical_grad, factor_grad)),
        "peak_log_energy_loss": float(peak_value),
        "peak_log_energy_gradient_norm": float(_tree_norm(peak_grad)),
        "peak_log_energy_gradient_cosine": float(_gradient_cosine(physical_grad, peak_grad)),
        "post_peak_output_row_alignment": _output_row_alignment(
            physical_grad, post_peak_grad
        ),
        "peak_output_row_alignment": _output_row_alignment(physical_grad, peak_grad),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
