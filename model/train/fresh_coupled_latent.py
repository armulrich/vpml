"""Train one compact neural closure from random parameters on an IC family.

The physical low-moment/latent step is shared with the existing model. No
specialist, teacher-fitted neural readout, rebound clock, or checkpoint is loaded.
An absolute-epoch horizon schedule is identical for E10 and E200.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from model.train.coupled_low_moment_latent import _atomic_savez, _training_state_payload, _load_training_state
from model.train.coupled_run_support import RunRecord, atomic_json, make_model_functions
from model.train.kinetic_latent_dynamics_probe import _adam_init, _adam_step
from vpml.kinetic_latent import init_state_conditioned_latent_operator, resolved_hermite_to_low_moment_state
from vpml.low_moment import primitive_fields


def curriculum_horizon(epoch: int, full_horizon: int):
    """Fixed prefix: requested total epoch count never changes this schedule."""
    if epoch < 1:
        raise ValueError("epoch must be positive")
    limits = ((2, 50), (4, 100), (6, 200), (8, 400))
    return min(full_horizon, next((h for last, h in limits if epoch <= last), full_horizon))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-initial-scale", type=float, default=1e-6)
    parser.add_argument("--latent-state-bound", type=float, default=0.0)
    parser.add_argument("--hermite-tail-damping", type=float, default=10.0)
    parser.add_argument("--envelope-weight", type=float, default=0.1)
    parser.add_argument("--envelope-change-weight", type=float, default=0.1)
    parser.add_argument("--latent-weight", type=float, default=0.1)
    parser.add_argument("--energy-floor-ratio", type=float, default=1e-8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-run", type=Path, help="Continue an interrupted fresh run in a new output directory")
    parser.add_argument("--validation-every", type=int, default=2)
    parser.add_argument("--parameter-step-limit", type=float, default=1e-4)
    parser.add_argument("--backtracks", type=int, default=6)
    parser.add_argument("--latent-input-floor-quantile", type=float, default=0.5)
    return parser


def replay_training_rng(seed, case_counts, steps_per_epoch, full_horizon, epoch, completed_step):
    """Recover the current permutation and RNG without repeating any gradients."""
    if epoch < 1 or not 0 <= completed_step <= steps_per_epoch:
        raise ValueError("invalid saved training cursor")
    rng = np.random.default_rng(seed)
    for current_epoch in range(1, epoch + 1):
        orders = [rng.permutation(count) for count in case_counts]
        horizon = curriculum_horizon(current_epoch, full_horizon)
        steps = steps_per_epoch if current_epoch < epoch else completed_step
        for step in range(steps):
            if step % 2 and horizon != full_horizon:
                for _ in case_counts:
                    rng.integers(full_horizon - horizon + 1)
    return rng, orders


def local_window_mean(values, width):
    """Sum each window locally so an early peak cannot erase a late signal."""
    kernel = jnp.ones((1, 1, width), dtype=values.dtype) / width
    return jax.lax.conv_general_dilated(values[:, None, :], kernel, (1,), "VALID",
        dimension_numbers=("NCH", "OIH", "NCH"))[:, 0, :]


def event_agnostic_loss(predicted_fields, target_fields, predicted_latent,
                        target_latent, *, cadence, envelope_weight,
                        change_weight, latent_weight, floor_ratio,
                        full_energy_scale=None):
    """Per-case physical, latent and all-window envelope loss, without events."""
    # Shape (case,time,channel,x); per-case normalization prevents the largest
    # amplitude from setting the entire objective's scale.
    denominator = jnp.sum(target_fields ** 2, axis=(1, 3))
    channel_errors = (jnp.sum((predicted_fields - target_fields) ** 2, axis=(1, 3))
                      / jnp.maximum(denominator, 1e-20))
    latent = jnp.mean(jnp.sum((predicted_latent - target_latent) ** 2, axis=(1, 2, 3))
                     / jnp.maximum(jnp.sum(target_latent ** 2, axis=(1, 2, 3)), 1e-20))
    energy = jnp.mean(predicted_fields[:, :, 3] ** 2, axis=-1)
    reference = jnp.mean(target_fields[:, :, 3] ** 2, axis=-1)
    scale = (jnp.max(reference, axis=1, keepdims=True) if full_energy_scale is None
             else jnp.asarray(full_energy_scale).reshape(-1, 1))
    floor = jnp.maximum(floor_ratio * scale, 1e-30)
    field_error = jnp.mean((predicted_fields[:, :, 3] - target_fields[:, :, 3]) ** 2, axis=-1)
    levels, changes, local_field_errors = [], [], []
    for duration in (2.5, 5.0, 10.0):
        width = max(1, int(round(duration / cadence)))
        if 2 * width > energy.shape[1]:
            continue
        def log_average(values):
            average = local_window_mean(values, width)
            return jnp.log(jnp.maximum(average, 0) + floor)
        predicted, target = log_average(energy), log_average(reference)
        local_field_errors.append(jnp.mean(local_window_mean(field_error, width)
            / (local_window_mean(reference, width) + floor)))
        levels.append(jnp.mean((predicted - target) ** 2))
        changes.append(jnp.mean((predicted[:, width:] - predicted[:, :-width]
                                 - target[:, width:] + target[:, :-width]) ** 2))
    if not levels:
        raise ValueError("training horizon is too short for envelope windows")
    level = jnp.mean(jnp.stack(levels))
    change = jnp.mean(jnp.stack(changes))
    # Give the electric-field channel the same one-of-four weight as before,
    # but normalize within every window so early energy cannot hide late phase
    # errors. Moment channels keep their original trajectory normalization.
    physical = (jnp.sum(jnp.mean(channel_errors[:, :3], axis=0))
                + jnp.mean(jnp.stack(local_field_errors))) / 4
    total = physical + latent_weight * latent + envelope_weight * level + change_weight * change
    return total, jnp.stack((physical, latent, level, change))


def fresh_parameters(config):
    return init_state_conditioned_latent_operator(
        jax.random.PRNGKey(config["seed"]), resolved_channels=3,
        latent_rank=config["latent_rank"], width=config["width"], depth=config["depth"],
        spectral_modes=config["operator_modes"], operator_rank=config["operator_rank"],
        kernel_size=config["kernel_size"],
        output_projection_init_scale=config["operator_output_init_scale"],
        conditioner_output_channels=config["latent_rank"],
        conditioner_output_init_scale=config["operator_output_init_scale"],
        latent_delay_input=True, conditioner_experts=0,
    )


def backtracked_update(params, proposal, evaluate, *, initial_loss, initial_guard,
                       step_limit, backtracks):
    """Bound an optimizer proposal and verify it on current and guard batches.

    The guard contains training cases, not held-out examples or target events.
    Rejected proposals never replace the parameters or optimizer state.
    """
    delta = jax.tree_util.tree_map(lambda new, old: new - old, proposal, params)
    norm = float(jnp.sqrt(sum(jnp.sum(value ** 2) for value in delta.values())))
    scale = min(1.0, step_limit / max(norm, 1e-30))
    attempts = []
    for attempt in range(backtracks + 1):
        candidate = jax.tree_util.tree_map(lambda old, change: old + scale * change, params, delta)
        current, guard = evaluate(candidate)
        finite = bool(np.isfinite(current) and np.isfinite(guard))
        passed = (finite and current <= initial_loss * 1.001 + 1e-10
                  and guard <= initial_guard * 1.05 + 1e-8)
        attempts.append(dict(scale=scale, loss=current if np.isfinite(current) else None,
                             guard=guard if np.isfinite(guard) else None, passed=passed))
        if passed:
            return candidate, dict(accepted=True, scale=scale, delta_norm=scale * norm,
                                   attempts=attempts)
        scale *= 0.5
    return params, dict(accepted=False, scale=0.0, delta_norm=0.0, attempts=attempts)


def evaluate_guard_first(candidate, current_value, guard_value, initial_guard):
    """Skip a costly batch evaluation when the guard already requires rejection.

    Both functions are deterministic. A skipped loss is recorded as null by
    backtracked_update; acceptance and step scales are unchanged.
    """
    guard = float(guard_value(candidate))
    if not np.isfinite(guard) or guard > initial_guard * 1.05 + 1e-8:
        return float("nan"), guard
    return float(current_value(candidate)), guard


def descent_safe_adam_step(params, gradient, optimizer, learning_rate, grad_clip=1.):
    """Reset stale first moments only when Adam proposes an uphill direction.

    Backtracking on the current batch needs a descent direction. Keep second
    moments and the optimizer clock; the existing guard still checks the step.
    """
    proposal, updated, norm = _adam_step(params, gradient, optimizer, learning_rate, grad_clip)
    def directional_derivative(candidate):
        return float(sum(jnp.sum(gradient[key].astype(jnp.float64) *
            (candidate[key]-params[key]).astype(jnp.float64)) for key in params))
    before = directional_derivative(proposal)
    reset = before >= 0.
    if reset:
        reset_state = {**optimizer, "m": jax.tree_util.tree_map(jnp.zeros_like, optimizer["m"])}
        proposal, updated, norm = _adam_step(params, gradient, reset_state, learning_rate, grad_clip)
    after = directional_derivative(proposal)
    if not np.isfinite(after) or after > 0.:
        raise FloatingPointError("Adam safeguard failed to produce a non-uphill direction")
    return proposal, updated, norm, dict(reset_first_moment=reset,
        directional_derivative_before=before, directional_derivative_after=after)


def main():
    args = build_parser().parse_args()
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.learning_rate <= 0 or args.validation_every < 1:
        raise ValueError("epochs, steps and learning rate must be positive")
    if not 0 < args.energy_floor_ratio < 1 or min(args.envelope_weight,
            args.envelope_change_weight, args.latent_weight, args.output_initial_scale) < 0:
        raise ValueError("invalid loss weight, initialization scale, or floor")
    if args.parameter_step_limit <= 0 or args.backtracks < 0:
        raise ValueError("step limit must be positive and backtracks nonnegative")
    if not 0 <= args.latent_input_floor_quantile <= 1:
        raise ValueError("input floor quantile must lie in [0,1]")
    if args.resume_run:
        if args.preflight_only:
            raise ValueError("resume and preflight-only are separate operations")
        origin = json.loads((args.resume_run / "run.json").read_text())
        try:
            os.kill(int(origin["pid"]), 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("source process still exists; refuse a concurrent continuation")
    sources = (Path(__file__), Path("model/train/coupled_run_support.py"),
               Path("model/train/coupled_low_moment_latent.py"),
               Path("vpml/kinetic_latent.py"), Path("vpml/low_moment.py"),
               args.projected_cache / "basis.npy", args.projected_cache / "statistics.npz",
               args.projected_cache / "metadata.json", args.reference_cache / "metadata.json",
               args.projected_cache / "coupled_nonlinear_correction_bounds_hermiteVlasovDamp1.000e+01_power6.000_tol1.000e-03.npz",
               args.reference_cache / "ic_manifest.json")
    sources += (Path("model/train/kinetic_latent_dynamics_probe.py"),
                Path("model/train/run_fresh_coupled_latent.sh"))
    if args.resume_run:
        sources += tuple(args.resume_run / name for name in
            ("training_state.npz", "report.json", "development.json", "best_coupled_low_moment_latent.npz"))
    record = RunRecord(args.outdir, {k: str(v) if isinstance(v, Path) else v
                                   for k, v in vars(args).items()}, sources=sources)
    try:
        metadata = json.loads((args.projected_cache / "metadata.json").read_text())
        teacher = json.loads((args.reference_cache / "metadata.json").read_text())["configuration"]
        manifest = json.loads((args.reference_cache / "ic_manifest.json").read_text())
        full_horizon = int(metadata["time_samples"]) - 1
        rank, nx = int(metadata["rank"]), int(metadata["nx"])
        if rank > 60:
            raise ValueError("fresh recipe is limited to at most 60 evolved latent channels")
        config = dict(reference_cache=str(args.reference_cache), projected_cache=str(args.projected_cache),
                      latent_rank=rank, nx=nx, basis_modes=int(metadata["basis_modes"]),
                      cadence=float(metadata["cadence"]), fine_steps=int(round(float(metadata["cadence"])
                          / float(teacher["teacher_dt"]))), rollout_steps=full_horizon,
                      gradient_chunk_steps=full_horizon, depth=args.depth, width=args.width,
                      kernel_size=5, operator_rank=16, operator_modes=nx // 2 + 1,
                      operator_output_init_scale=args.output_initial_scale, seed=args.seed,
                      latent_delay_input=True, latent_readout_mode="equilibrium_cnn",
                      equilibrium_input_compression_scale=4.0, conditioner_experts=0,
                      linear_baseline="projected_hermite", hermite_tail_damping=args.hermite_tail_damping,
                      hermite_tail_power=6.0, latent_state_bound=args.latent_state_bound,
                      initialization="random", init_checkpoint=None,
                      envelope_weight=args.envelope_weight, envelope_change_weight=args.envelope_change_weight,
                      latent_weight=args.latent_weight, energy_floor_ratio=args.energy_floor_ratio)
        config["objective_protocol"] = "local_field_and_envelope_v2"
        config["optimizer_protocol"] = "adam_descent_safeguard_v1"
        config.update(parameter_step_limit=args.parameter_step_limit, backtracks=args.backtracks,
                      learning_rate=args.learning_rate, steps_per_epoch=args.steps_per_epoch)
        arrays = dict(np.load(args.projected_cache / "statistics.npz", allow_pickle=False))
        input_floor = float(np.quantile(arrays["latent_scale"], args.latent_input_floor_quantile))
        arrays["latent_input_scale"] = np.maximum(arrays["latent_scale"], input_floor)
        config.update(latent_input_floor_quantile=args.latent_input_floor_quantile,
                      latent_input_floor=input_floor)
        record.stage("input_conditioning", input_floor=input_floor,
            original_scale_ratio=float(np.max(arrays["latent_scale"]) / np.min(arrays["latent_scale"])),
            feature_scale_ratio=float(np.max(arrays["latent_input_scale"]) / np.min(arrays["latent_input_scale"])))
        arrays["basis"] = np.load(args.projected_cache / "basis.npy")
        arrays["k_arr"] = (2 * np.pi * np.fft.rfftfreq(nx, d=float(teacher["teacher_L"]) / nx)).astype(np.float32)
        # This unused compatibility array is not fitted. Projected-Hermite
        # dynamics ignores the learned linear-propagator checkpoint field.
        arrays["linear_propagator"] = np.broadcast_to(np.eye(rank + 3, dtype=np.complex64),
                                                       (nx // 2 + 1, rank + 3, rank + 3)).copy()
        with np.load(args.projected_cache / "coupled_nonlinear_correction_bounds_hermiteVlasovDamp1.000e+01_power6.000_tol1.000e-03.npz") as saved:
            arrays["correction_bounds"] = saved["bounds"]
        if args.hermite_tail_damping != 10:
            raise ValueError("this calibration belongs to damping 10; do not silently reuse it")
        train = [case for case in manifest["cases"] if case["split"] == "train"]
        regimes = ("linear_landau", "nonlinear_landau_weak", "nonlinear_landau_strong")
        by_regime = [[case for case in train if case["regime"] == regime] for regime in regimes]
        if args.steps_per_epoch > min(map(len, by_regime)):
            raise ValueError("steps per epoch must not exceed distinct training cases per regime")
        trajectories = {case["case_id"]: np.load(args.projected_cache / "cases" / (case["case_id"] + ".npy"),
                                                mmap_mode="r") for case in train}
        from model.train.coupled_run_support import field_from_resolved
        energy_scales = {}
        for case_id, values in trajectories.items():
            if values.shape != (full_horizon + 1, rank + 3, nx):
                raise ValueError(f"incompatible training case {case_id}: {values.shape}")
            field = field_from_resolved(values[:, :3], arrays["k_arr"],
                                        poisson_sign=teacher["teacher_poisson_sign"])
            energy_scales[case_id] = float(np.max(np.mean(field ** 2, axis=-1)))
        params = fresh_parameters(config)
        optimizer = _adam_init(params)
        history = []
        def save(name):
            _atomic_savez(args.outdir / name, {**arrays, **{k: np.asarray(v) for k, v in params.items()}})
        save("initial_coupled_low_moment_latent.npz")
        atomic_json(args.outdir / "report.json", {"configuration": config, "history": history})
        atomic_json(args.outdir / "training_cases.json", {"cases": train})
        record.stage("random_initialization_saved", seed=args.seed, state_channels=rank + 3,
                     parameter_count=sum(value.size for value in params.values()))
        rng = np.random.default_rng(args.seed)
        start_epoch, start_step, resume_orders = 1, 0, None
        if args.resume_run:
            previous_report = json.loads((args.resume_run / "report.json").read_text())
            previous_config = {k: v for k, v in previous_report["configuration"].items()
                               if k != "continuation_from"}
            if previous_config != config:
                raise ValueError("continuation configuration differs from the original recipe")
            params, restored, completed_epoch = _load_training_state(
                args.resume_run / "training_state.npz", params, {"physical": optimizer})
            optimizer = restored["physical"]
            with np.load(args.resume_run / "training_state.npz") as saved:
                start_epoch, start_step = completed_epoch + 1, int(saved["completed_step"])
                expected_rng = json.loads(str(saved["rng_state_json"]))
            rng, resume_orders = replay_training_rng(args.seed, list(map(len, by_regime)),
                args.steps_per_epoch, full_horizon, start_epoch, start_step)
            if rng.bit_generator.state != expected_rng:
                raise ValueError("replayed sampling cursor differs from the saved RNG state")
            history = [row for row in previous_report["history"] if
                       (row["epoch"], row["step"]) < (start_epoch, start_step)]
            if len(history) != int(optimizer["step"]):
                raise ValueError("saved history and optimizer update count differ")
            config["continuation_from"] = str(args.resume_run)
            save("last_coupled_low_moment_latent.npz")
            (args.outdir / "training_state.npz").write_bytes((args.resume_run / "training_state.npz").read_bytes())
            atomic_json(args.outdir / "report.json", {"configuration": config, "history": history})
            record.stage("resume_state_restored", epoch=start_epoch, next_step=start_step,
                         accepted_updates=int(optimizer["step"]), rng_replay_equal=True)
        functions = {}
        rejected_updates = 0
        validation_history = []
        best_score = float("inf")
        if args.resume_run:
            validation_history = json.loads((args.resume_run / "development.json").read_text())["history"]
            best_score = min(row["score"] for row in validation_history if row["finite"])
            (args.outdir / "best_coupled_low_moment_latent.npz").write_bytes(
                (args.resume_run / "best_coupled_low_moment_latent.npz").read_bytes())
            atomic_json(args.outdir / "development.json", {"history": validation_history})
        def physical_fields(values):
            flat = values.reshape(-1, 3, nx)
            physical = primitive_fields(resolved_hermite_to_low_moment_state(flat),
                jnp.asarray(arrays["k_arr"]), poisson_sign=float(teacher["teacher_poisson_sign"]))
            return physical.reshape(values.shape[0], values.shape[1], 4, nx)
        if not args.preflight_only:
            development = [case for case in manifest["cases"] if case["split"] == "heldout"
                           and case["case_id"] != "nonlinear_landau_weak_ic16"]
            development_target = jnp.asarray(np.stack([np.load(args.projected_cache / "cases" /
                (case["case_id"] + ".npy")) for case in development]))
            _, development_rollout, _ = make_model_functions(config, arrays, teacher)
            development_rollout = jax.jit(development_rollout)
            def validate(epoch):
                nonlocal best_score
                record.stage("development_start", epoch=epoch, horizon=full_horizon,
                             case_ids=[c["case_id"] for c in development])
                predicted, latent = development_rollout(params, development_target[:, 0, :3],
                                                        development_target[:, 0, 3:])
                predicted = jnp.concatenate((development_target[:, :1, :3], predicted), axis=1)
                latent = jnp.concatenate((development_target[:, :1, 3:], latent), axis=1)
                value, components = event_agnostic_loss(physical_fields(predicted),
                    physical_fields(development_target[:, :, :3]), latent, development_target[:, :, 3:],
                    cadence=config["cadence"], envelope_weight=args.envelope_weight,
                    change_weight=args.envelope_change_weight, latent_weight=args.latent_weight,
                    floor_ratio=args.energy_floor_ratio)
                score = float(value)
                finite = bool(np.isfinite(score) and np.all(np.isfinite(np.asarray(latent))))
                row = dict(epoch=epoch, score=score if finite else None, finite=finite,
                           components=np.asarray(components).tolist() if finite else None,
                           scope="legacy development; not independent generalization")
                if finite and score < best_score:
                    best_score = score
                    save("best_coupled_low_moment_latent.npz")
                    row["best_updated"] = True
                validation_history.append(row)
                atomic_json(args.outdir / "development.json", {"history": validation_history})
                record.stage("development_complete", **row)
                return finite
            if not args.resume_run and not validate(0):
                raise FloatingPointError("random initialization has nonfinite full development rollout")
        for epoch in range(start_epoch, args.epochs + 1):
            horizon = curriculum_horizon(epoch, full_horizon)
            if horizon not in functions:
                record.stage("compile_start", epoch=epoch, horizon=horizon)
                _, rollout, _ = make_model_functions(config, arrays, teacher, horizon=horizon,
                                                      gradient_chunk_steps=horizon)
                def loss(model, target, energy_scale):
                    predicted, latent = rollout(model, target[:, 0, :3], target[:, 0, 3:])
                    predicted = jnp.concatenate((target[:, :1, :3], predicted), axis=1)
                    latent = jnp.concatenate((target[:, :1, 3:], latent), axis=1)
                    return event_agnostic_loss(physical_fields(predicted), physical_fields(target[:, :, :3]), latent,
                        target[:, :, 3:], cadence=config["cadence"], envelope_weight=args.envelope_weight,
                        change_weight=args.envelope_change_weight, latent_weight=args.latent_weight,
                        floor_ratio=args.energy_floor_ratio, full_energy_scale=energy_scale)
                gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))
                sample = jnp.asarray(np.stack([trajectories[cases[0]["case_id"]][:horizon + 1]
                                               for cases in by_regime]))
                sample_scales = jnp.ones((3,), dtype=sample.dtype)
                functions[horizon] = (
                    gradient.lower(params, sample, sample_scales).compile(),
                    jax.jit(lambda p, t, s: loss(p, t, s)[0]).lower(params, sample, sample_scales).compile(),
                )
                record.stage("compile_complete", epoch=epoch, horizon=horizon)
            guard_cases = [max(cases, key=lambda c: c["epsilon"])["case_id"] for cases in by_regime]
            guard_start = (full_horizon - horizon) // 2
            guard_batch = jnp.asarray(np.stack([trajectories[c][guard_start:guard_start + horizon + 1]
                                                for c in guard_cases]))
            guard_scales = jnp.asarray([energy_scales[c] for c in guard_cases], dtype=guard_batch.dtype)
            orders = (resume_orders if epoch == start_epoch and resume_orders is not None
                      else [rng.permutation(len(cases)) for cases in by_regime])
            for step in range(start_step if epoch == start_epoch else 0, args.steps_per_epoch):
                case_ids = [cases[order[step]]["case_id"] for cases, order in zip(by_regime, orders)]
                # Alternate initial-value trajectories and unbiased teacher-start
                # windows until the full horizon is reached. Neither sampler
                # selects peaks, growth signs, nor case-specific transitions.
                starts = [0 if step % 2 == 0 or horizon == full_horizon else
                          int(rng.integers(full_horizon - horizon + 1)) for _ in case_ids]
                batch = jnp.asarray(np.stack([trajectories[c][start:start + horizon + 1]
                                              for c, start in zip(case_ids, starts)]))
                record.stage("gradient_start", epoch=epoch, step=step, horizon=horizon,
                             case_ids=case_ids, starts=starts)
                batch_scales = jnp.asarray([energy_scales[c] for c in case_ids], dtype=batch.dtype)
                gradient_function, value_function = functions[horizon]
                (value, components), gradient = gradient_function(params, batch, batch_scales)
                scalar = float(value)
                finite = np.isfinite(scalar) and all(np.all(np.isfinite(np.asarray(g))) for g in gradient.values())
                if not finite:
                    raise FloatingPointError(f"nonfinite gradient at epoch {epoch} step {step}")
                if args.preflight_only:
                    record.stage("preflight_gradient_passed", loss=scalar,
                                 components=np.asarray(components).tolist(), starts=starts)
                    if step >= min(1, args.steps_per_epoch - 1):
                        record.finish("preflight_passed", checked_batches=step + 1)
                        return
                    continue
                proposal, proposed_optimizer, grad_norm, direction = descent_safe_adam_step(
                    params, gradient, optimizer, args.learning_rate, 1.0)
                initial_guard = float(value_function(params, guard_batch, guard_scales))
                if not np.isfinite(initial_guard):
                    raise FloatingPointError("current parameters fail the fixed training guard")
                def evaluate_proposal(candidate):
                    return evaluate_guard_first(candidate,
                        lambda p: value_function(p, batch, batch_scales),
                        lambda p: value_function(p, guard_batch, guard_scales), initial_guard)
                params, update = backtracked_update(params, proposal, evaluate_proposal,
                    initial_loss=scalar, initial_guard=initial_guard,
                    step_limit=args.parameter_step_limit, backtracks=args.backtracks)
                if update["accepted"]:
                    optimizer = proposed_optimizer
                    rejected_updates = 0
                else:
                    rejected_updates += 1
                    record.stage("update_rejected", epoch=epoch, step=step, update=update,
                                 loss=scalar, gradient_norm=float(grad_norm), optimizer_direction=direction)
                    if rejected_updates >= 3:
                        record.finish("no_acceptable_update", epoch=epoch, step=step)
                        return
                    continue
                params = jax.tree_util.tree_map(lambda v: v.block_until_ready(), params)
                if not all(np.all(np.isfinite(np.asarray(p))) for p in params.values()):
                    raise FloatingPointError("nonfinite Adam update")
                row = dict(epoch=epoch, step=step, horizon=horizon, loss=scalar,
                           components=np.asarray(components).tolist(), gradient_norm=float(grad_norm),
                           case_ids=case_ids, starts=starts, update=update, optimizer_direction=direction)
                history.append(row)
                save("last_coupled_low_moment_latent.npz")
                _atomic_savez(args.outdir / "training_state.npz", {
                    **_training_state_payload(params, {"physical": optimizer}),
                    "completed_epoch": np.asarray(epoch - 1), "completed_step": np.asarray(step + 1),
                    "rng_state_json": np.asarray(json.dumps(rng.bit_generator.state)),
                })
                atomic_json(args.outdir / "report.json", {"configuration": config, "history": history})
                record.stage("update_saved", **row)
            save(f"epoch{epoch:03d}_coupled_low_moment_latent.npz")
            if epoch % args.validation_every == 0 or epoch == args.epochs:
                if not validate(epoch):
                    record.finish("nonfinite_development", epoch=epoch)
                    return
        record.finish("training_complete_unvalidated")
    except BaseException as error:
        record.finish("failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
