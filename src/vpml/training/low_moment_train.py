"""Train a causal spectral-memory closure for a three-moment fluid solver."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import (
    Dict,
    Optional,
    Sequence,
)

from vpml.jax_runtime import bootstrap_jax_runtime, print_jax_runtime_summary

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np

from vpml.datasets.interface_flux import sha256_json
from vpml.low_moment import primitive_fields


# Compatibility exports; reusable implementations live in vpml.training.
from vpml.training.low_moment_configuration import (
    build_arg_parser,
    validate_training_options,
    REGIMES,
    TRAINING_MODE,
    OBJECTIVE,
    MODEL_BACKEND,
    CHECKPOINT_SCHEMA,
    WINDOW_MEMORY_BACKENDS,
    parse_training_regimes,
    _checkpoint_regimes,
    _validate_regime_resume_configuration,
    _uses_window_memory,
    _uses_compact_latent,
    parse_train_case_limits,
    parse_horizon_curriculum,
)
from vpml.training.low_moment_data import (
    _load_convergence_floor,
    _restrict_rfft,
    low_hermite_coefficients_to_conservative,
    _primitive_numpy,
    _central_heat_flux_numpy,
    _load_reference_cache,
    _case_amplitudes,
    _gather_coefficients,
    _statistics_path,
    _primitive_target_cache_root,
    _load_or_build_primitive_target_cache,
    _compute_training_statistics,
    load_or_compute_training_statistics,
    _load_normalization_statistics,
)
from vpml.training.low_moment_sampling import (
    _build_anchor_index,
    limit_training_anchors,
    sample_batch,
    build_complete_trajectory_case_batches,
    build_balanced_random_window_epoch,
    _full_anchor_steps_per_epoch,
    build_balanced_full_anchor_epoch,
    sample_complete_trajectory_batch,
    build_diagnostic_panel,
)
from vpml.training.optimization import (
    _adam_init,
    _cosine_learning_rate,
    _adam_step,
)
from vpml.training.low_moment_losses import (
    _rollout_window_memory,
    _unpack_window_memory,
    _block_relative_sample_loss,
    _sqrt_with_zero_subgradient,
    _sqrt_with_zero_subgradient_jvp,
    make_loss_function,
    make_continuous_chunk_loss_function,
    make_supervised_heat_flux_loss,
)
from vpml.training.low_moment_state import (
    initialize_model_parameters,
    _save_checkpoint,
    _load_checkpoint,
    _save_training_state,
    _load_training_state,
)
from vpml.training.low_moment_evaluation import (
    evaluate_saved_checkpoint,
    _plot_losses,
    _electric_field_energy,
    _json_float,
    _interpolate_time_series,
    _energy_peak_angular_frequency,
    _evaluate_heldout,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    print_jax_runtime_summary(jax, context="low-moment closure training")
    args = build_arg_parser().parse_args(argv)
    (
        training_regimes,
        planned_epochs,
        train_case_limits,
        diagnostic_start_times,
        curriculum,
    ) = validate_training_options(args)
    grouped, manifest, cache_metadata, coefficient_key = _load_reference_cache(
        args.reference_cache
    )
    configuration = dict(cache_metadata["configuration"])
    source_nx = int(configuration["teacher_Nx"])
    domain_length = float(configuration["teacher_L"])
    reference_dt = float(configuration["teacher_dt"])
    dt = reference_dt if args.solver_dt is None else float(args.solver_dt)
    reference_steps_per_solver_step = dt / reference_dt
    poisson_sign = float(configuration["teacher_poisson_sign"])
    relative_time_block_steps = 0
    relative_time_block_count = 0
    convergence_floor_rms = None
    convergence_floor_metadata = None
    if args.convergence_floor_file is not None:
        relative_time_block_steps = int(round(args.relative_time_block / dt))
        if relative_time_block_steps <= 0 or not math.isclose(
            relative_time_block_steps * dt, args.relative_time_block, abs_tol=1e-10
        ):
            raise ValueError("relative-time-block must be an integer multiple of dt")
        total_steps = int(round(float(configuration["T_final"]) / dt))
        relative_time_block_count = int(
            math.ceil(total_steps / relative_time_block_steps)
        )
        convergence_floor_rms, convergence_floor_metadata = _load_convergence_floor(
            args.convergence_floor_file
        )
        expected_floor_metadata = {
            "regimes": list(REGIMES),
            "channels": [
                "density_perturbation",
                "velocity",
                "pressure_perturbation",
                "electric_field",
            ],
            "rollout_Nx": int(args.rollout_Nx),
        }
        floor_mismatches = {
            key: (convergence_floor_metadata.get(key), expected)
            for key, expected in expected_floor_metadata.items()
            if convergence_floor_metadata.get(key) != expected
        }
        if floor_mismatches:
            raise ValueError(
                f"Convergence floor metadata does not match training: {floor_mismatches}"
            )
        print(
            f"[data] block-relative trajectory loss: block={args.relative_time_block:g} "
            f"blocks={relative_time_block_count} "
            f"convergence_floor={args.convergence_floor_file}"
        )
    if int(args.rollout_Nx) > source_nx:
        raise ValueError("rollout-Nx cannot exceed the cached teacher Nx")
    outdir = args.outdir.resolve()
    if args.evaluate_checkpoint is not None:
        evaluate_saved_checkpoint(
            args, grouped=grouped, manifest=manifest, coefficient_key=coefficient_key,
            source_nx=source_nx, domain_length=domain_length, reference_dt=reference_dt,
            poisson_sign=poisson_sign, outdir=outdir,
        )
        return
    if args.resume_run is not None and args.resume_run.resolve() != outdir:
        raise ValueError("resume-run must name the same directory as outdir")
    if args.resume_run is None and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    anchors = _build_anchor_index(
        grouped,
        coefficient_key,
        regimes=training_regimes,
        horizon=int(
            math.ceil(
                (
                    1
                    if args.supervised_only
                    else args.rollout_horizon + args.autonomous_history_burnin_steps
                )
                * reference_steps_per_solver_step
            )
        ),
        history_stride=args.history_stride,
    )
    selected_training_cases: Dict[str, np.ndarray] = {
        regime: np.unique(anchors[regime]["train_cases"]) for regime in training_regimes
    }
    if train_case_limits:
        anchors, selected_training_cases = limit_training_anchors(
            anchors,
            train_case_limits,
            regimes=training_regimes,
            seed=args.seed,
        )
    for regime in training_regimes:
        selected_case_ids = np.asarray(grouped[regime]["case_ids"], dtype=np.str_)[
            selected_training_cases[regime]
        ]
        print(
            f"[data] {regime}: train_anchors={anchors[regime]['train_times'].size} "
            f"heldout_anchors={anchors[regime]['val_times'].size} "
            f"selected_train_cases={','.join(selected_case_ids.tolist())}"
        )
    if args.full_anchor_sweep and training_regimes != REGIMES:
        args.steps_per_epoch = _full_anchor_steps_per_epoch(
            anchors, training_regimes, args.batch_size, args.gradient_accumulation_steps
        )
        print(f"[data] selected full sweep: {args.steps_per_epoch} optimizer updates/epoch")
    # Preserve the original all3 normalization, independent of training exposure.
    # Matched history-only/latent runs share these exact cached statistics.
    normalization_checkpoint_sha256 = None
    if args.normalization_checkpoint is not None:
        stats, normalization_checkpoint_sha256 = _load_normalization_statistics(
            args.normalization_checkpoint,
            manifest=manifest,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
        )
        print(f"[data] fixed normalization from {args.normalization_checkpoint}")
    else:
        stats = load_or_compute_training_statistics(
            args.reference_cache,
            grouped,
            manifest,
            coefficient_key,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
            stats_stride=args.stats_stride,
        )
    print(
        "[data] low-moment input scales: "
        + ", ".join(f"{value:.4e}" for value in stats["input_scale"])
    )
    required_output_bound = float(stats["heat_flux_gradient_max_abs"][0]) / float(
        stats["heat_flux_gradient_scale"][0]
    )
    if float(args.normalized_heat_flux_bound) <= required_output_bound:
        raise ValueError(
            "normalized heat-flux-gradient bound cannot represent the training data: "
            f"configured={args.normalized_heat_flux_bound:.6g} "
            f"required>{required_output_bound:.6g}"
        )
    print(
        "[data] heat-flux-gradient scale/bound: "
        f"scale={float(stats['heat_flux_gradient_scale'][0]):.6e} "
        f"required_normalized_max={required_output_bound:.3f} "
        f"configured_bound={args.normalized_heat_flux_bound:.3f}"
    )
    heat_flux_gradient_case_scale_by_id = dict(
        zip(
            np.asarray(stats["heat_flux_gradient_case_ids"], dtype=np.str_),
            np.asarray(stats["heat_flux_gradient_case_scales"], dtype=np.float64),
        )
    )
    primitive_target_histories = None
    trajectory_target_norm_by_id = None
    primitive_target_cache_root = None
    if args.training_schedule == "continuous_trajectories":
        (
            primitive_target_histories,
            trajectory_target_norm_by_id,
            primitive_target_cache_root,
        ) = _load_or_build_primitive_target_cache(
            args.reference_cache,
            grouped,
            manifest,
            coefficient_key,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
            poisson_sign=poisson_sign,
        )
        print(f"[data] primitive target cache: {primitive_target_cache_root}")
    if args.supervised_only:
        horizon_description = "supervised_q_only"
    elif len(curriculum) == 1:
        horizon_description = f"fixed_H={args.rollout_horizon}"
    else:
        horizon_description = "curriculum=" + ",".join(
            f"{h}:{e}" for h, e in curriculum
        )
    spacetime_description = (
        f" depth={args.spacetime_depth} temporal_kernel={args.temporal_kernel_size}"
        if args.memory_backend == "causal_spacetime_operator"
        else ""
    )
    if args.memory_backend in ("window_fno", "burles_latent_fno"):
        spacetime_description = f" depth={args.fno_depth}"
    if args.memory_backend == "burles_latent_fno":
        spacetime_description += f" latent={args.latent_memory_dim}"
    print(
        f"[model] backend={args.memory_backend} width={args.width} "
        f"modes={args.spectral_modes}{spacetime_description} "
        f"memory_span={args.memory_steps * args.memory_stride * dt:.3f} "
        f"rollout_span={args.rollout_horizon * dt:.3f} "
        f"scan_unroll={args.scan_unroll} "
        f"warmup={args.supervised_warmup_epochs} {horizon_description}"
    )
    nx = int(args.rollout_Nx)
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(nx, d=domain_length / nx)
    params = initialize_model_parameters(args)
    parameter_count = sum(
        int(value.size) for value in jax.tree_util.tree_leaves(params)
    )
    print(f"[model] trainable_parameters={parameter_count}")
    if args.init_checkpoint is not None:
        initialized_params, initialized_metadata, _ = _load_checkpoint(
            args.init_checkpoint
        )
        expected = {
            "schema_version": CHECKPOINT_SCHEMA,
            "state_representation": "centered_conservative_v1",
            "closure_output": (
                "effective_gradient_with_uniform_heating"
                if args.allow_uniform_heating
                else "zero_mean_heat_flux_gradient"
            ),
            "manifest_sha256": str(manifest["sha256"]),
            "source_Nx": source_nx,
            "rollout_Nx": int(args.rollout_Nx),
            "width": int(args.width),
            "spectral_modes": int(args.spectral_modes),
            "memory_backend": str(args.memory_backend),
        }
        if args.memory_backend == "burles_latent_fno":
            expected["latent_memory_dim"] = int(args.latent_memory_dim)
        mismatches = {
            key: (initialized_metadata.get(key), value)
            for key, value in expected.items()
            if initialized_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Incompatible initialization checkpoint: {mismatches}")
        params = initialized_params
        print(f"[train] initialized closure from {args.init_checkpoint}")
    optimizer = _adam_init(params)
    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(args.seed + 1)
    train_history = []
    train_ema_history = []
    train_update_norm_history = []
    train_update_scale_min_history = []
    train_eval_history = []
    train_eval_regime_history = []
    train_eval_epochs = []
    val_history = []
    val_regime_history = []
    val_epochs = []
    autonomous_val_history = []
    autonomous_val_regime_history = []
    autonomous_val_epochs = []
    best_val = math.inf
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "training_mode": TRAINING_MODE,
        "objective": (
            "supervised_heat_flux_gradient_capacity"
            if args.supervised_only
            else OBJECTIVE
        ),
        "model_backend": MODEL_BACKEND,
        "reference_cache": str(Path(args.reference_cache).resolve()),
        "manifest_sha256": str(manifest["sha256"]),
        "source_Nx": source_nx,
        "rollout_Nx": args.rollout_Nx,
        "dt": dt,
        "reference_dt": reference_dt,
        "reference_steps_per_solver_step": reference_steps_per_solver_step,
        "rollout_horizon": args.rollout_horizon,
        "training_schedule": args.training_schedule,
        "continuous_trajectory_final_time": (
            float(configuration["T_final"])
            if args.training_schedule == "continuous_trajectories"
            else None
        ),
        "horizon_curriculum": (
            [] if len(curriculum) == 1 else [list(stage) for stage in curriculum]
        ),
        "supervised_heat_flux_warmup_epochs": args.supervised_warmup_epochs,
        "supervised_heat_flux_epochs": (
            args.epochs if args.supervised_only else args.supervised_warmup_epochs
        ),
        "memory_steps": args.memory_steps,
        "memory_stride": args.memory_stride,
        "scan_unroll": args.scan_unroll,
        "memory_backend": args.memory_backend,
        "primitive_target_cache": (
            str(primitive_target_cache_root)
            if primitive_target_cache_root is not None
            else None
        ),
        "fno_depth": (
            int(args.fno_depth)
            if args.memory_backend in ("window_fno", "burles_latent_fno")
            else None
        ),
        "latent_memory_dim": (
            int(args.latent_memory_dim)
            if args.memory_backend == "burles_latent_fno"
            else None
        ),
        "closure_integration": (
            "ssprk3_carried_then_provisional"
            if args.memory_backend == "burles_latent_fno"
            else "legacy_backend"
        ),
        "spacetime_depth": (
            args.spacetime_depth
            if args.memory_backend == "causal_spacetime_operator"
            else None
        ),
        "temporal_kernel_size": (
            args.temporal_kernel_size
            if args.memory_backend == "causal_spacetime_operator"
            else None
        ),
        "input_scaling": args.input_scaling,
        "dynamic_amplitude_floor": args.dynamic_amplitude_floor,
        "supervised_only": args.supervised_only,
        "width": args.width,
        "spectral_modes": args.spectral_modes,
        "batch_size_per_regime": args.batch_size,
        "train_case_limits": dict(train_case_limits),
        "selected_training_case_ids": {
            regime: np.asarray(grouped[regime]["case_ids"], dtype=np.str_)[
                selected_training_cases[regime]
            ].tolist()
            for regime in training_regimes
        },
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "data_parallel_devices": args.data_parallel_devices,
        "training_passes_per_epoch": args.training_passes_per_epoch,
        "steps_per_epoch": args.steps_per_epoch,
        "steps_per_epoch_source": (
            "selected_full_anchor_count"
            if args.full_anchor_sweep and training_regimes != REGIMES
            else "configured"
        ),
        "deterministic_anchor_cycle": args.deterministic_anchor_cycle,
        "full_anchor_sweep": args.full_anchor_sweep,
        "translation_augmentation": args.translation_augmentation,
        "planned_epochs": planned_epochs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "final_learning_rate": args.final_learning_rate,
        "weight_decay": args.weight_decay,
        "update_norm_cap": args.update_norm_cap,
        "require_first_update_descent": args.require_first_update_descent,
        "report_every_update_descent": args.report_every_update_descent,
        "grad_clip": args.grad_clip,
        "optimizer": "adamw_cosine",
        "autonomous_history_burnin_steps": args.autonomous_history_burnin_steps,
        "autonomous_history_burnin_detached": bool(args.autonomous_history_burnin_steps),
        "compact_latent_initializer_gradient_disconnected": bool(
            args.memory_backend == "burles_latent_fno"
            and args.latent_memory_dim > 0
            and args.autonomous_history_burnin_steps > 0
        ),
        "training_history_source": (
            "model_generated_from_low_moment_reset"
            if args.autonomous_history_burnin_steps
            else "teacher_low_moment_history_at_random_reset"
        ),
        "teacher_heat_flux_history_input": bool(
            args.closure_history_input
            and not args.autonomous_history_burnin_steps
        ),
        "log_energy_weight": args.log_energy_weight,
        "log_growth_weight": args.log_growth_weight,
        "nonfinite_trajectory_penalty": args.nonfinite_trajectory_penalty,
        "init_checkpoint": (
            None
            if args.init_checkpoint is None
            else str(args.init_checkpoint.resolve())
        ),
        "diagnostic_start_times": list(diagnostic_start_times),
        "training_diagnostic_cases_per_regime": (
            args.training_diagnostic_cases_per_regime
        ),
        "regimes": list(training_regimes),
        "training_regimes": args.training_regimes,
        "canonical_regimes": list(REGIMES),
        "regime_indices": [REGIMES.index(regime) for regime in training_regimes],
        "normalization_checkpoint": (
            str(args.normalization_checkpoint.resolve())
            if args.normalization_checkpoint is not None else None
        ),
        "normalization_checkpoint_sha256": normalization_checkpoint_sha256,
        "normalization_policy": "fixed_all3_training_statistics",
        "normalization_regimes": list(REGIMES),
        "stats_stride": args.stats_stride,
        "normalization_statistics_sha256": sha256_json(
            {key: np.asarray(value).tolist() for key, value in stats.items()}
        ),
        "normalized_heat_flux_bound": args.normalized_heat_flux_bound,
        "density_floor": args.density_floor,
        "pressure_floor": args.pressure_floor,
        "state_representation": "centered_conservative_v1",
        "closure_output": (
            "effective_gradient_with_uniform_heating"
            if args.allow_uniform_heating
            else "zero_mean_heat_flux_gradient"
        ),
        "allow_uniform_heating": args.allow_uniform_heating,
        "closure_history_input": args.closure_history_input,
        "relative_trajectory_loss": args.relative_trajectory_loss,
        "global_relative_trajectory_loss": args.global_relative_trajectory_loss,
        "relative_time_block": args.relative_time_block,
        "convergence_floor_file": (
            None
            if args.convergence_floor_file is None
            else str(args.convergence_floor_file.resolve())
        ),
        "convergence_floor_metadata": convergence_floor_metadata,
    }
    if metadata["compact_latent_initializer_gradient_disconnected"]:
        print(
            "[train] known limitation: detached autonomous burn-in disconnects "
            "compact_latent_init_* from the scored rollout gradient; preserved "
            "for the controlled comparison"
        )
    immutable_resume_keys = (
        "manifest_sha256",
        "rollout_Nx",
        "dt",
        "reference_dt",
        "rollout_horizon",
        "horizon_curriculum",
        "memory_steps",
        "memory_stride",
        "memory_backend",
        "fno_depth",
        "latent_memory_dim",
        "input_scaling",
        "width",
        "spectral_modes",
        "batch_size_per_regime",
        "train_case_limits",
        "selected_training_case_ids",
        "gradient_accumulation_steps",
        "data_parallel_devices",
        "training_passes_per_epoch",
        "steps_per_epoch",
        "deterministic_anchor_cycle",
        "full_anchor_sweep",
        "translation_augmentation",
        "planned_epochs",
        "learning_rate",
        "final_learning_rate",
        "weight_decay",
        "grad_clip",
        "report_every_update_descent",
        "update_norm_cap",
        "autonomous_history_burnin_steps",
        "log_energy_weight",
        "log_growth_weight",
        "global_relative_trajectory_loss",
    )
    configuration_path = outdir / "run_configuration.json"
    resumed_histories = None
    resumed_state = None
    if args.resume_run is not None:
        if not configuration_path.is_file():
            raise FileNotFoundError(f"Missing resume configuration: {configuration_path}")
        saved_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        _validate_regime_resume_configuration(saved_configuration, metadata)
        mismatches = {
            key: (saved_configuration.get(key), metadata.get(key))
            for key in immutable_resume_keys
            if saved_configuration.get(key) != metadata.get(key)
        }
        if mismatches:
            raise ValueError(f"Resume configuration mismatch: {mismatches}")
        params, optimizer, resumed_histories, resumed_state = _load_training_state(
            outdir / "training_state.npz"
        )
        rng.bit_generator.state = resumed_state["rng_state"]
        print(
            f"[train] resumed exact state at epoch={resumed_state['global_epoch']} "
            f"optimizer_step={int(np.asarray(optimizer['step']))}"
        )
        if "in_epoch" in resumed_state:
            print(
                "[train] found resumable partial epoch "
                f"{resumed_state['in_epoch']['number']} after "
                f"{resumed_state['in_epoch']['completed_steps']} updates"
            )
    else:
        configuration_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    started = time.perf_counter()
    global_epoch = 0 if resumed_state is None else int(resumed_state["global_epoch"])
    if global_epoch >= int(args.epochs):
        raise ValueError(
            f"Resume epoch {global_epoch} has already reached requested epoch {args.epochs}"
        )
    if resumed_histories is not None:
        train_history = resumed_histories["train_loss"].tolist()
        train_ema_history = resumed_histories["train_ema_loss"].tolist()
        train_update_norm_history = resumed_histories.get(
            "train_update_norm", np.asarray([], dtype=np.float64)
        ).tolist()
        train_update_scale_min_history = resumed_histories.get(
            "train_update_scale_min", np.asarray([], dtype=np.float64)
        ).tolist()
        train_eval_epochs = resumed_histories["train_eval_epochs"].astype(int).tolist()
        train_eval_history = resumed_histories["train_eval_loss"].tolist()
        train_eval_regime_history = resumed_histories[
            "train_eval_regime_loss"
        ].tolist()
        val_epochs = resumed_histories["val_epochs"].astype(int).tolist()
        val_history = resumed_histories["val_loss"].tolist()
        val_regime_history = resumed_histories["val_regime_loss"].tolist()
        autonomous_val_epochs = resumed_histories[
            "autonomous_val_epochs"
        ].astype(int).tolist()
        autonomous_val_history = resumed_histories["autonomous_val_loss"].tolist()
        autonomous_val_regime_history = resumed_histories[
            "autonomous_val_regime_loss"
        ].tolist()
        best_val = float(resumed_state["best_val"])
    supervised_epochs = (
        int(args.epochs) if args.supervised_only else int(args.supervised_warmup_epochs)
    )
    if supervised_epochs > 0:
        warmup_loss = make_supervised_heat_flux_loss(
            regimes=training_regimes,
            k_arr=k_arr,
            width=args.width,
            input_scale=stats["input_scale"],
            heat_flux_gradient_scale=float(stats["heat_flux_gradient_scale"][0]),
            amplitude_center=float(stats["amplitude_center"][0]),
            amplitude_scale=float(stats["amplitude_scale"][0]),
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=args.normalized_heat_flux_bound,
            memory_backend=args.memory_backend,
            closure_history_input=args.closure_history_input,
            input_scaling=args.input_scaling,
            dynamic_amplitude_floor=args.dynamic_amplitude_floor,
            allow_uniform_heating=args.allow_uniform_heating,
        )
        warmup_value_and_grad = jax.jit(
            jax.value_and_grad(warmup_loss, has_aux=True)
        )
        warmup_validation_panel = build_diagnostic_panel(
            validation_rng,
            grouped,
            anchors,
            manifest,
            coefficient_key,
            regimes=training_regimes,
            split="val",
            start_times=diagnostic_start_times,
            cases_per_regime=None,
            horizon=1,
            dt=dt,
            memory_steps=args.memory_steps,
            memory_stride=args.memory_stride,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
            heat_flux_gradient_case_scale_by_id=(
                heat_flux_gradient_case_scale_by_id
            ),
            include_heat_flux_gradient_history=args.closure_history_input,
            reference_dt=reference_dt,
        )
        warmup_validation_panel_jax = tuple(
            {key: jnp.asarray(value) for key, value in batch.items()}
            for batch in warmup_validation_panel
        )
        warmup_evaluate = jax.jit(warmup_loss)
        phase_name = "capacity probe" if args.supervised_only else "warm-up"
        print(
            f"[train] supervised central-heat-flux {phase_name} "
            f"epochs={supervised_epochs} normalization=fixed_per_IC_RMS"
        )
        for _ in range(supervised_epochs):
            global_epoch += 1
            epoch_losses = []
            epoch_regime_losses = []
            for _ in range(int(args.steps_per_epoch)):
                batch = sample_batch(
                    rng,
                    grouped,
                    anchors,
                    manifest,
                    coefficient_key,
                    regimes=training_regimes,
                    split="train",
                    batch_size_per_regime=args.batch_size,
                    horizon=1,
                    memory_steps=args.memory_steps,
                    memory_stride=args.memory_stride,
                    source_nx=source_nx,
                    rollout_nx=args.rollout_Nx,
                    domain_length=domain_length,
                    translation_augmentation=True,
                    heat_flux_gradient_case_scale_by_id=(
                        heat_flux_gradient_case_scale_by_id
                    ),
                    include_heat_flux_gradient_history=args.closure_history_input,
                    reference_steps_per_solver_step=(
                        reference_steps_per_solver_step
                    ),
                )
                batch_jax = {key: jnp.asarray(value) for key, value in batch.items()}
                (loss_value, regime_value), grads = warmup_value_and_grad(
                    params, batch_jax
                )
                params, optimizer, grad_norm, _, _ = _adam_step(
                    params,
                    grads,
                    optimizer,
                    args.learning_rate,
                    args.grad_clip,
                    update_norm_cap=args.update_norm_cap,
                )
                epoch_losses.append(float(loss_value))
                epoch_regime_losses.append(
                    np.asarray(regime_value, dtype=np.float64)
                )
            mean_loss = float(np.mean(epoch_losses))
            mean_regime_loss = np.mean(epoch_regime_losses, axis=0)
            if not math.isfinite(mean_loss):
                raise FloatingPointError(
                    f"Non-finite heat-flux warm-up loss at epoch {global_epoch}"
                )
            train_history.append(mean_loss)
            val_text = ""
            if (
                global_epoch == 1
                or global_epoch % int(args.validation_every) == 0
                or global_epoch == supervised_epochs
            ):
                panel_values = []
                panel_regime_values = []
                for panel_batch in warmup_validation_panel_jax:
                    val_result = warmup_evaluate(params, panel_batch)
                    panel_values.append(float(val_result[0]))
                    panel_regime_values.append(
                        np.asarray(val_result[1], dtype=np.float64)
                    )
                val_value = float(np.mean(panel_values))
                val_regime = np.mean(panel_regime_values, axis=0)
                val_regime_history.append(val_regime)
                if val_value < best_val:
                    best_val = val_value
                    _save_checkpoint(
                        outdir / "best_low_moment_closure.npz",
                        params,
                        metadata,
                        stats,
                    )
                val_history.append(val_value)
                val_epochs.append(global_epoch)
                val_text = (
                    f" heldout={val_value:.6e} heldout_regime=("
                    + ",".join(f"{value:.3e}" for value in val_regime)
                    + ")"
                )
            elapsed = time.perf_counter() - started
            print(
                f"[train] epoch {global_epoch:04d}/{args.epochs:04d} "
                f"supervised_q={mean_loss:.6e}{val_text} regime=("
                + ",".join(f"{value:.3e}" for value in mean_regime_loss)
                + ") "
                f"grad={float(grad_norm):.3e} elapsed={elapsed / 60.0:.1f}m"
            )
    curriculum_epoch_offset = int(args.supervised_warmup_epochs)
    for stage_horizon, stage_epochs in curriculum:
        stage_end_epoch = curriculum_epoch_offset + int(stage_epochs)
        if global_epoch >= stage_end_epoch:
            curriculum_epoch_offset = stage_end_epoch
            continue
        loss_fn = make_loss_function(
            regimes=training_regimes,
            k_arr=k_arr,
            width=args.width,
            horizon=stage_horizon,
            dt=dt,
            input_scale=stats["input_scale"],
            regime_scales=stats["regime_scales"],
            heat_flux_gradient_scale=float(stats["heat_flux_gradient_scale"][0]),
            amplitude_center=float(stats["amplitude_center"][0]),
            amplitude_scale=float(stats["amplitude_scale"][0]),
            poisson_sign=poisson_sign,
            normalized_heat_flux_bound=args.normalized_heat_flux_bound,
            density_floor=args.density_floor,
            pressure_floor=args.pressure_floor,
            relative_trajectory_loss=args.relative_trajectory_loss,
            global_relative_trajectory_loss=args.global_relative_trajectory_loss,
            closure_history_input=args.closure_history_input,
            relative_time_block_steps=relative_time_block_steps,
            relative_time_block_count=relative_time_block_count,
            convergence_floor_rms=convergence_floor_rms,
            memory_backend=args.memory_backend,
            memory_stride=args.memory_stride,
            scan_unroll=args.scan_unroll,
            input_scaling=args.input_scaling,
            dynamic_amplitude_floor=args.dynamic_amplitude_floor,
            allow_uniform_heating=args.allow_uniform_heating,
            autonomous_burnin_steps=args.autonomous_history_burnin_steps,
            log_energy_weight=args.log_energy_weight,
            log_growth_weight=args.log_growth_weight,
        )
        value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        parallel_value_and_grad = None
        if int(args.data_parallel_devices) > 1:
            def parallel_loss_and_grad(parallel_params, parallel_batch):
                (value, regime_value), gradients = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(parallel_params, parallel_batch)
                value = jax.lax.pmean(value, axis_name="data")
                regime_value = jax.lax.pmean(regime_value, axis_name="data")
                gradients = jax.tree_util.tree_map(
                    lambda leaf: jax.lax.pmean(leaf, axis_name="data"), gradients
                )
                return (value, regime_value), gradients

            parallel_value_and_grad = jax.pmap(
                parallel_loss_and_grad,
                axis_name="data",
                in_axes=(None, 0),
                devices=jax.devices()[: int(args.data_parallel_devices)],
            )
        evaluate_loss = jax.jit(loss_fn)
        validation_panel = build_diagnostic_panel(
            validation_rng,
            grouped,
            anchors,
            manifest,
            coefficient_key,
            regimes=training_regimes,
            split="val",
            start_times=diagnostic_start_times,
            cases_per_regime=None,
            horizon=stage_horizon,
            dt=dt,
            memory_steps=args.memory_steps,
            memory_stride=args.memory_stride,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
            autonomous_burnin_steps=args.autonomous_history_burnin_steps,
            include_heat_flux_gradient_history=args.closure_history_input,
            reference_dt=reference_dt,
        )
        training_diagnostic_panel = build_diagnostic_panel(
            np.random.default_rng(args.seed + 2),
            grouped,
            anchors,
            manifest,
            coefficient_key,
            regimes=training_regimes,
            split="train",
            start_times=diagnostic_start_times,
            cases_per_regime=args.training_diagnostic_cases_per_regime,
            horizon=stage_horizon,
            dt=dt,
            memory_steps=args.memory_steps,
            memory_stride=args.memory_stride,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
            autonomous_burnin_steps=args.autonomous_history_burnin_steps,
            include_heat_flux_gradient_history=args.closure_history_input,
            reference_dt=reference_dt,
        )
        validation_panel_jax = tuple(
            {key: jnp.asarray(value) for key, value in batch.items()}
            for batch in validation_panel
        )
        training_diagnostic_panel_jax = tuple(
            {key: jnp.asarray(value) for key, value in batch.items()}
            for batch in training_diagnostic_panel
        )

        def evaluate_panel(panel):
            values = []
            regime_values = []
            for panel_batch in panel:
                value, per_regime = evaluate_loss(params, panel_batch)
                values.append(float(value))
                regime_values.append(np.asarray(per_regime, dtype=np.float64))
            return float(np.mean(values)), np.mean(regime_values, axis=0)

        continuous_grad_fns = {}
        continuous_eval_fns = {}

        def get_continuous_chunk_function(chunk_length, warm_memory, *, gradients):
            key = (int(chunk_length), bool(warm_memory))
            cache = continuous_grad_fns if gradients else continuous_eval_fns
            if key not in cache:
                chunk_loss = make_continuous_chunk_loss_function(
                    regimes=training_regimes,
                    k_arr=k_arr,
                    width=args.width,
                    horizon=chunk_length,
                    warm_memory=warm_memory,
                    dt=dt,
                    input_scale=stats["input_scale"],
                    regime_scales=stats["regime_scales"],
                    heat_flux_gradient_scale=float(stats["heat_flux_gradient_scale"][0]),
                    amplitude_center=float(stats["amplitude_center"][0]),
                    amplitude_scale=float(stats["amplitude_scale"][0]),
                    poisson_sign=poisson_sign,
                    normalized_heat_flux_bound=args.normalized_heat_flux_bound,
                    density_floor=args.density_floor,
                    pressure_floor=args.pressure_floor,
                    relative_trajectory_loss=args.relative_trajectory_loss,
                    closure_history_input=args.closure_history_input,
                    relative_time_block_steps=relative_time_block_steps,
                    relative_time_block_count=relative_time_block_count,
                    convergence_floor_rms=convergence_floor_rms,
                    memory_backend=args.memory_backend,
                    memory_stride=args.memory_stride,
                    scan_unroll=args.scan_unroll,
                    input_scaling=args.input_scaling,
                    dynamic_amplitude_floor=args.dynamic_amplitude_floor,
                    allow_uniform_heating=args.allow_uniform_heating,
                )
                cache[key] = jax.jit(
                    jax.value_and_grad(chunk_loss, has_aux=True)
                    if gradients
                    else chunk_loss
                )
            return cache[key]

        def run_complete_trajectory_batch(batch, *, gradients):
            state = jnp.asarray(batch["initial"])
            memory_or_hidden = jnp.asarray(batch["memory"])
            amplitude = jnp.asarray(batch["amplitude"])
            regime_index = jnp.asarray(batch["regime_index"])
            if "target_fields" in batch:
                target_fields = jnp.asarray(batch["target_fields"])
            else:
                target_states = jnp.asarray(batch["targets"])
                target_shape = target_states.shape
                target_fields = primitive_fields(
                    target_states.reshape(
                        target_shape[0] * target_shape[1], 3, target_shape[-1]
                    ),
                    jnp.asarray(k_arr, dtype=jnp.float32),
                    poisson_sign=poisson_sign,
                ).reshape(target_shape[0], target_shape[1], 4, target_shape[-1])
            total_steps = int(target_fields.shape[1])
            if args.relative_trajectory_loss:
                target_shape = target_fields.shape
                if (
                    "trajectory_target_norm" in batch
                    and convergence_floor_rms is None
                ):
                    trajectory_target_norm = jnp.asarray(
                        batch["trajectory_target_norm"], dtype=target_fields.dtype
                    )
                elif convergence_floor_rms is not None:
                    full_indices = jnp.broadcast_to(
                        jnp.arange(1, target_shape[1] + 1, dtype=jnp.int32)[None, :],
                        (target_shape[0], target_shape[1]),
                    )
                    membership = jax.nn.one_hot(
                        jnp.minimum(
                            (full_indices - 1) // relative_time_block_steps,
                            relative_time_block_count - 1,
                        ),
                        relative_time_block_count,
                        dtype=target_fields.dtype,
                    )
                    target_energy = jnp.sum(jnp.square(target_fields), axis=-1)
                    trajectory_target_norm = jnp.einsum(
                        "btq,btc->bqc", membership, target_energy
                    )
                    full_counts = jnp.sum(membership, axis=1)
                    floor_by_regime = jnp.asarray(
                        convergence_floor_rms, dtype=target_fields.dtype
                    )[regime_index]
                    trajectory_target_norm = jnp.maximum(
                        trajectory_target_norm,
                        full_counts[:, :, None]
                        * target_shape[-1]
                        * jnp.square(floor_by_regime[:, None, :]),
                    )
                else:
                    trajectory_target_norm = jnp.sum(
                        jnp.square(target_fields), axis=(1, 3)
                    )
                if convergence_floor_rms is None and np.any(
                    np.asarray(trajectory_target_norm) <= 0.0
                ):
                    raise ValueError(
                        "Relative trajectory loss excludes zero-norm target trajectories"
                    )
            else:
                trajectory_target_norm = jnp.ones(
                    (target_fields.shape[0],), dtype=target_fields.dtype
                )
            accumulated_grads = None
            accumulated_loss = 0.0
            accumulated_regime = np.zeros((len(training_regimes),), dtype=np.float64)
            completed = 0
            first_chunk = True
            failure_step = None
            while completed < total_steps:
                chunk_length = min(int(stage_horizon), total_steps - completed)
                target_chunk = target_fields[:, completed : completed + chunk_length]
                function = get_continuous_chunk_function(
                    chunk_length, first_chunk, gradients=gradients
                )
                function_args = (
                    params,
                    state,
                    memory_or_hidden,
                    target_chunk,
                    amplitude,
                    regime_index,
                    trajectory_target_norm,
                    jnp.broadcast_to(
                        jnp.arange(
                            completed + 1,
                            completed + chunk_length + 1,
                            dtype=jnp.int32,
                        )[None, :],
                        (target_fields.shape[0], chunk_length),
                    ),
                )
                if gradients:
                    (
                        (loss_value, (regime_value, final_state, final_hidden)),
                        chunk_grads,
                    ) = function(*function_args)
                else:
                    loss_value, (regime_value, final_state, final_hidden) = function(
                        *function_args
                    )
                    chunk_grads = None
                loss_float = float(loss_value)
                if not math.isfinite(loss_float):
                    failure_step = completed
                    remaining_weight = float(total_steps - completed) / float(
                        total_steps
                    )
                    accumulated_loss += (
                        remaining_weight * args.nonfinite_trajectory_penalty
                    )
                    accumulated_regime += (
                        remaining_weight * args.nonfinite_trajectory_penalty
                    )
                    break
                weight = (
                    1.0
                    if args.relative_trajectory_loss
                    else float(chunk_length) / float(total_steps)
                )
                accumulated_loss += weight * loss_float
                accumulated_regime += weight * np.asarray(
                    regime_value, dtype=np.float64
                )
                if gradients:
                    weighted = jax.tree_util.tree_map(
                        lambda value: weight * value, chunk_grads
                    )
                    accumulated_grads = (
                        weighted
                        if accumulated_grads is None
                        else jax.tree_util.tree_map(
                            lambda left, right: left + right,
                            accumulated_grads,
                            weighted,
                        )
                    )
                state = jax.lax.stop_gradient(final_state)
                memory_or_hidden = jax.lax.stop_gradient(final_hidden)
                completed += chunk_length
                first_chunk = False
            return (
                accumulated_loss,
                accumulated_regime,
                accumulated_grads,
                failure_step,
            )

        def evaluate_complete_split(split):
            case_batches = build_complete_trajectory_case_batches(
                np.random.default_rng(args.seed + (3 if split == "train" else 4)),
                grouped,
                regimes=training_regimes,
                split=split,
                batch_size_per_regime=args.batch_size,
                shuffle=False,
            )
            values = []
            regime_values = []
            weights = []
            failed_batches = 0
            earliest_failure_step = None
            for case_rows in case_batches:
                batch = sample_complete_trajectory_batch(
                    validation_rng,
                    grouped,
                    manifest,
                    coefficient_key,
                    case_rows,
                    regimes=training_regimes,
                    memory_steps=args.memory_steps,
                    memory_stride=args.memory_stride,
                    source_nx=source_nx,
                    rollout_nx=args.rollout_Nx,
                    translation_augmentation=False,
                    primitive_target_histories=primitive_target_histories,
                    trajectory_target_norm_by_id=trajectory_target_norm_by_id,
                )
                value, per_regime, _, failure_step = run_complete_trajectory_batch(
                    batch, gradients=False
                )
                if failure_step is not None:
                    failed_batches += 1
                    earliest_failure_step = (
                        failure_step
                        if earliest_failure_step is None
                        else min(earliest_failure_step, failure_step)
                    )
                values.append(value)
                regime_values.append(per_regime)
                weights.append(int(case_rows[training_regimes[0]].size))
            return (
                float(np.average(values, weights=weights)),
                np.average(np.asarray(regime_values), axis=0, weights=weights),
                failed_batches,
                earliest_failure_step,
            )

        initial_train_eval, initial_train_regime = evaluate_panel(
            training_diagnostic_panel_jax
        )
        initial_val_eval, initial_val_regime = evaluate_panel(validation_panel_jax)
        metadata["initial_training_panel_loss"] = initial_train_eval
        metadata["initial_heldout_panel_loss"] = initial_val_eval
        metadata["initial_training_panel_regime_loss"] = initial_train_regime.tolist()
        metadata["initial_heldout_panel_regime_loss"] = initial_val_regime.tolist()
        stage_label = "fixed horizon" if len(curriculum) == 1 else "curriculum stage"
        print(
            f"[train] {stage_label} H={stage_horizon} "
            f"epochs={stage_epochs} span={stage_horizon * dt:.3f}"
        )
        print(
            f"[train] initial-model baseline train_panel={initial_train_eval:.6e} "
            f"heldout_panel={initial_val_eval:.6e}"
        )
        initial_autonomous_val = None
        if args.training_schedule == "continuous_trajectories":
            (
                initial_autonomous_val,
                initial_autonomous_regime,
                initial_autonomous_failures,
                initial_autonomous_failure_step,
            ) = evaluate_complete_split("val")
            metadata["initial_heldout_autonomous_loss"] = initial_autonomous_val
            metadata["initial_heldout_autonomous_regime_loss"] = (
                initial_autonomous_regime.tolist()
            )
            metadata["initial_heldout_autonomous_failed_batches"] = (
                initial_autonomous_failures
            )
            print(
                "[train] continuous autonomous schedule: one complete pass over "
                "every training IC per epoch; optimizer updates occur only after "
                "complete T trajectories"
            )
            print(
                f"[train] initial heldout_autonomous={initial_autonomous_val:.6e} "
                f"failed_batches={initial_autonomous_failures} "
                f"earliest_failure_t="
                f"{None if initial_autonomous_failure_step is None else initial_autonomous_failure_step * dt} "
                "regime=("
                + ",".join(f"{value:.3e}" for value in initial_autonomous_regime)
                + ")"
            )
        loss_ema = (
            None
            if resumed_state is None or not math.isfinite(resumed_state["loss_ema"])
            else float(resumed_state["loss_ema"])
        )
        if args.resume_run is None and global_epoch == 0:
            _save_checkpoint(
                outdir / "epoch000_low_moment_closure.npz", params, metadata, stats
            )
            _save_training_state(
                outdir / "training_state.npz",
                params=params,
                optimizer=optimizer,
                rng=rng,
                global_epoch=0,
                loss_ema=loss_ema,
                best_val=best_val,
                histories={
                    "train_loss": train_history,
                    "train_ema_loss": train_ema_history,
                    "train_update_norm": train_update_norm_history,
                    "train_update_scale_min": train_update_scale_min_history,
                    "train_eval_epochs": train_eval_epochs,
                    "train_eval_loss": train_eval_history,
                    "train_eval_regime_loss": train_eval_regime_history,
                    "val_epochs": val_epochs,
                    "val_loss": val_history,
                    "val_regime_loss": val_regime_history,
                    "autonomous_val_epochs": autonomous_val_epochs,
                    "autonomous_val_loss": autonomous_val_history,
                    "autonomous_val_regime_loss": autonomous_val_regime_history,
                },
            )
        while global_epoch < min(stage_end_epoch, int(args.epochs)):
            global_epoch += 1
            partial_epoch = (
                None if resumed_state is None else resumed_state.get("in_epoch")
            )
            if partial_epoch is not None:
                if int(partial_epoch["number"]) != global_epoch:
                    raise ValueError(
                        "Partial-epoch state does not follow the last completed epoch"
                    )
                completed_epoch_steps = int(partial_epoch["completed_steps"])
                if not 0 < completed_epoch_steps < int(args.steps_per_epoch):
                    raise ValueError("Invalid partial-epoch completed step count")
                rng.bit_generator.state = partial_epoch["rng_state"]
                epoch_losses = partial_epoch["losses"].astype(float).tolist()
                epoch_regime_losses = (
                    partial_epoch["regime_losses"].astype(float).tolist()
                )
                epoch_grad_norms = partial_epoch["grad_norms"].astype(float).tolist()
                epoch_update_norms = (
                    partial_epoch["update_norms"].astype(float).tolist()
                )
                epoch_update_scales = (
                    partial_epoch["update_scales"].astype(float).tolist()
                )
                print(
                    f"[train] resuming epoch {global_epoch} at update "
                    f"{completed_epoch_steps}/{args.steps_per_epoch}"
                )
            else:
                completed_epoch_steps = 0
                epoch_losses = []
                epoch_regime_losses = []
                epoch_grad_norms = []
                epoch_update_norms = []
                epoch_update_scales = []
            epoch_failed_batches = 0
            epoch_earliest_failure_step = None
            learning_rate = _cosine_learning_rate(
                int(np.asarray(optimizer["step"])),
                planned_epochs * int(args.steps_per_epoch),
                args.learning_rate,
                args.final_learning_rate,
            )
            if args.training_schedule == "continuous_trajectories":
                training_steps = build_complete_trajectory_case_batches(
                    rng,
                    grouped,
                    regimes=training_regimes,
                    split="train",
                    batch_size_per_regime=args.batch_size,
                    shuffle=True,
                )
            else:
                epoch_rng_state = json.loads(
                    json.dumps(rng.bit_generator.state, sort_keys=True)
                )
                training_steps = range(
                    completed_epoch_steps, int(args.steps_per_epoch)
                )
                if args.full_anchor_sweep:
                    epoch_window_selections = build_balanced_full_anchor_epoch(
                        rng,
                        anchors,
                        regimes=training_regimes,
                        split="train",
                        batch_size_per_regime=args.batch_size,
                    )
                else:
                    epoch_window_selections = tuple(
                        selection
                        for _ in range(int(args.training_passes_per_epoch))
                        for selection in build_balanced_random_window_epoch(
                            rng,
                            anchors,
                            regimes=training_regimes,
                            split="train",
                            batch_size_per_regime=args.batch_size,
                            deterministic_epoch=(
                                global_epoch if args.deterministic_anchor_cycle else None
                            ),
                        )
                    )
                expected_batches = int(args.steps_per_epoch) * int(
                    args.gradient_accumulation_steps
                )
                if len(epoch_window_selections) != expected_batches:
                    raise ValueError(
                        "steps-per-epoch * gradient-accumulation-steps must cover "
                        f"the configured balanced IC passes: expected {len(epoch_window_selections)}, "
                        f"configured {expected_batches}"
                    )
                epoch_window_iterator = iter(epoch_window_selections)
                for _ in range(
                    completed_epoch_steps * int(args.gradient_accumulation_steps)
                ):
                    next(epoch_window_iterator)
            for epoch_step, step_selection in enumerate(
                training_steps, start=completed_epoch_steps
            ):
                if args.training_schedule == "continuous_trajectories":
                    batch = sample_complete_trajectory_batch(
                        rng,
                        grouped,
                        manifest,
                        coefficient_key,
                        step_selection,
                        regimes=training_regimes,
                        memory_steps=args.memory_steps,
                        memory_stride=args.memory_stride,
                        source_nx=source_nx,
                        rollout_nx=args.rollout_Nx,
                        translation_augmentation=args.translation_augmentation,
                        primitive_target_histories=primitive_target_histories,
                        trajectory_target_norm_by_id=trajectory_target_norm_by_id,
                    )
                    (
                        optimizer_step_loss,
                        optimizer_step_regime,
                        grads,
                        failure_step,
                    ) = (
                        run_complete_trajectory_batch(batch, gradients=True)
                    )
                    if failure_step is not None:
                        epoch_failed_batches += 1
                        epoch_earliest_failure_step = (
                            failure_step
                            if epoch_earliest_failure_step is None
                            else min(epoch_earliest_failure_step, failure_step)
                        )
                    if grads is None:
                        epoch_losses.append(optimizer_step_loss)
                        epoch_regime_losses.append(optimizer_step_regime)
                        continue
                    (
                        params,
                        optimizer,
                        grad_norm,
                        update_norm,
                        update_scale,
                    ) = _adam_step(
                        params,
                        grads,
                        optimizer,
                        args.learning_rate,
                        args.grad_clip,
                        update_norm_cap=args.update_norm_cap,
                    )
                    epoch_losses.append(optimizer_step_loss)
                    epoch_regime_losses.append(optimizer_step_regime)
                    loss_ema = (
                        optimizer_step_loss
                        if loss_ema is None
                        else args.loss_ema_decay * loss_ema
                        + (1.0 - args.loss_ema_decay) * optimizer_step_loss
                    )
                    epoch_grad_norms.append(float(grad_norm))
                    epoch_update_norms.append(float(update_norm))
                    epoch_update_scales.append(float(update_scale))
                    continue
                accumulated_grads = None
                optimizer_step_losses = []
                optimizer_step_batches = []
                for _ in range(int(args.gradient_accumulation_steps)):
                    window_selection = next(epoch_window_iterator)
                    batch = sample_batch(
                        rng,
                        grouped,
                        anchors,
                        manifest,
                        coefficient_key,
                        regimes=training_regimes,
                        split="train",
                        batch_size_per_regime=args.batch_size,
                        horizon=stage_horizon,
                        memory_steps=args.memory_steps,
                        memory_stride=args.memory_stride,
                        source_nx=source_nx,
                        rollout_nx=args.rollout_Nx,
                        domain_length=domain_length,
                        translation_augmentation=args.translation_augmentation,
                        explicit_selection=window_selection,
                        autonomous_burnin_steps=args.autonomous_history_burnin_steps,
                        include_heat_flux_gradient_history=args.closure_history_input,
                        reference_steps_per_solver_step=(
                            reference_steps_per_solver_step
                        ),
                    )
                    if parallel_value_and_grad is None:
                        batch_jax = {
                            key: jnp.asarray(value) for key, value in batch.items()
                        }
                    else:
                        device_count = int(args.data_parallel_devices)
                        regime_indices = np.asarray(batch["regime_index"])
                        per_device_indices = [[] for _ in range(device_count)]
                        for regime_index in (REGIMES.index(regime) for regime in training_regimes):
                            rows = np.flatnonzero(regime_indices == regime_index)
                            if rows.size % device_count:
                                raise ValueError(
                                    "Each regime's minibatch must divide evenly across "
                                    "data-parallel devices"
                                )
                            for device_index, chunk in enumerate(
                                np.split(rows, device_count)
                            ):
                                per_device_indices[device_index].extend(chunk.tolist())
                        ordered_indices = np.asarray(
                            [row for rows in per_device_indices for row in rows],
                            dtype=np.int32,
                        )
                        batch_jax = {
                            key: jnp.asarray(value[ordered_indices]).reshape(
                                device_count,
                                len(ordered_indices) // device_count,
                                *value.shape[1:],
                            )
                            for key, value in batch.items()
                        }
                    optimizer_step_batches.append(batch_jax)
                    if parallel_value_and_grad is None:
                        (loss_value, regime_value), grads = value_and_grad(
                            params, batch_jax
                        )
                    else:
                        (replicated_loss, replicated_regime), replicated_grads = (
                            parallel_value_and_grad(params, batch_jax)
                        )
                        loss_value = replicated_loss[0]
                        regime_value = replicated_regime[0]
                        grads = jax.tree_util.tree_map(
                            lambda leaf: leaf[0], replicated_grads
                        )
                    loss_float = float(loss_value)
                    if not math.isfinite(loss_float):
                        selection_text = {
                            regime: {
                                "case_rows": np.asarray(value[0]).tolist(),
                                "reset_times": (
                                    np.asarray(value[1], dtype=np.float64)
                                    * reference_dt
                                ).tolist(),
                            }
                            for regime, value in window_selection.items()
                        }
                        raise FloatingPointError(
                            f"Non-finite loss at epoch {global_epoch}, H={stage_horizon}, "
                            f"regime_loss={np.asarray(regime_value).tolist()}, "
                            f"selection={selection_text}"
                        )
                    optimizer_step_losses.append(loss_float)
                    epoch_losses.append(loss_float)
                    epoch_regime_losses.append(
                        np.asarray(regime_value, dtype=np.float64)
                    )
                    accumulated_grads = (
                        grads
                        if accumulated_grads is None
                        else jax.tree_util.tree_map(
                            lambda left, right: left + right,
                            accumulated_grads,
                            grads,
                        )
                    )
                grads = jax.tree_util.tree_map(
                    lambda value: value / float(args.gradient_accumulation_steps),
                    accumulated_grads,
                )
                learning_rate = _cosine_learning_rate(
                    int(np.asarray(optimizer["step"])),
                    planned_epochs * int(args.steps_per_epoch),
                    args.learning_rate,
                    args.final_learning_rate,
                )
                (
                    params,
                    optimizer,
                    grad_norm,
                    update_norm,
                    update_scale,
                ) = _adam_step(
                    params,
                    grads,
                    optimizer,
                    learning_rate,
                    args.grad_clip,
                    args.weight_decay,
                    args.update_norm_cap,
                )
                should_report_update = args.report_every_update_descent or (
                    args.require_first_update_descent
                    and int(np.asarray(optimizer["step"])) == 1
                )
                if should_report_update:
                    post_update_loss = float(
                        np.mean(
                            [
                                float(evaluate_loss(params, selected_batch)[0])
                                for selected_batch in optimizer_step_batches
                            ]
                        )
                    )
                    pre_update_loss = float(np.mean(optimizer_step_losses))
                    print(
                        "[update-check] exact-batch "
                        f"step={int(np.asarray(optimizer['step']))} "
                        f"before={pre_update_loss:.6e} after={post_update_loss:.6e} "
                        f"ratio={post_update_loss / max(pre_update_loss, 1e-30):.6f}"
                    )
                    if (
                        args.require_first_update_descent
                        and int(np.asarray(optimizer["step"])) == 1
                        and (
                            not math.isfinite(post_update_loss)
                            or post_update_loss >= pre_update_loss
                        )
                    ):
                        raise FloatingPointError(
                            "First accepted update did not lower its exact accumulated batch"
                        )
                optimizer_step_loss = float(np.mean(optimizer_step_losses))
                loss_ema = (
                    optimizer_step_loss
                    if loss_ema is None
                    else args.loss_ema_decay * loss_ema
                    + (1.0 - args.loss_ema_decay) * optimizer_step_loss
                )
                epoch_grad_norms.append(float(grad_norm))
                epoch_update_norms.append(float(update_norm))
                epoch_update_scales.append(float(update_scale))
                completed_steps = epoch_step + 1
                if (
                    args.checkpoint_every_updates
                    and completed_steps < int(args.steps_per_epoch)
                    and completed_steps % int(args.checkpoint_every_updates) == 0
                ):
                    resume_rng = np.random.default_rng()
                    resume_rng.bit_generator.state = epoch_rng_state
                    _save_training_state(
                        outdir / "training_state.npz",
                        params=params,
                        optimizer=optimizer,
                        rng=resume_rng,
                        global_epoch=global_epoch - 1,
                        loss_ema=loss_ema,
                        best_val=best_val,
                        histories={
                            "train_loss": train_history,
                            "train_ema_loss": train_ema_history,
                            "train_update_norm": train_update_norm_history,
                            "train_update_scale_min": train_update_scale_min_history,
                            "train_eval_epochs": train_eval_epochs,
                            "train_eval_loss": train_eval_history,
                            "train_eval_regime_loss": train_eval_regime_history,
                            "val_epochs": val_epochs,
                            "val_loss": val_history,
                            "val_regime_loss": val_regime_history,
                            "autonomous_val_epochs": autonomous_val_epochs,
                            "autonomous_val_loss": autonomous_val_history,
                            "autonomous_val_regime_loss": autonomous_val_regime_history,
                        },
                        in_epoch={
                            "number": global_epoch,
                            "completed_steps": completed_steps,
                            "rng_state": epoch_rng_state,
                            "losses": epoch_losses,
                            "regime_losses": epoch_regime_losses,
                            "grad_norms": epoch_grad_norms,
                            "update_norms": epoch_update_norms,
                            "update_scales": epoch_update_scales,
                        },
                    )
                    os.sync()
                    print(
                        f"[checkpoint] epoch {global_epoch} partial update "
                        f"{completed_steps}/{args.steps_per_epoch}"
                    )
            resumed_state = None
            mean_loss = float(np.mean(epoch_losses))
            mean_regime_loss = np.mean(epoch_regime_losses, axis=0)
            train_history.append(mean_loss)
            train_ema_history.append(float(loss_ema))
            train_update_norm_history.append(float(np.mean(epoch_update_norms)))
            train_update_scale_min_history.append(float(np.min(epoch_update_scales)))
            val_text = ""
            should_validate = (
                global_epoch == 1
                or global_epoch % int(args.validation_every) == 0
                or global_epoch == int(args.epochs)
            )
            if should_validate:
                train_eval_value, train_eval_regime = evaluate_panel(
                    training_diagnostic_panel_jax
                )
                val_value, val_regime = evaluate_panel(validation_panel_jax)
                train_eval_history.append(train_eval_value)
                train_eval_regime_history.append(train_eval_regime)
                train_eval_epochs.append(global_epoch)
                val_history.append(val_value)
                val_regime_history.append(val_regime)
                val_epochs.append(global_epoch)
                val_text = (
                    f" train_panel={train_eval_value:.6e} "
                    f"heldout_panel={val_value:.6e} "
                    f"panel_ratio=({train_eval_value / max(initial_train_eval, 1e-30):.3f},"
                    f"{val_value / max(initial_val_eval, 1e-30):.3f})"
                )
                checkpoint_metric = val_value
                if args.training_schedule == "continuous_trajectories":
                    (
                        autonomous_value,
                        autonomous_regime,
                        autonomous_failures,
                        autonomous_failure_step,
                    ) = evaluate_complete_split("val")
                    autonomous_val_history.append(autonomous_value)
                    autonomous_val_regime_history.append(autonomous_regime)
                    autonomous_val_epochs.append(global_epoch)
                    val_text += (
                        f" heldout_autonomous={autonomous_value:.6e} "
                        f"autonomous_ratio={autonomous_value / max(initial_autonomous_val, 1e-30):.3f}"
                        f" autonomous_failed_batches={autonomous_failures}"
                    )
                    checkpoint_metric = autonomous_value
                if (
                    stage_horizon == args.rollout_horizon
                    and checkpoint_metric < best_val
                ):
                    best_val = checkpoint_metric
                    _save_checkpoint(
                        outdir / "best_low_moment_closure.npz", params, metadata, stats
                    )
            elapsed = time.perf_counter() - started
            print(
                f"[train] epoch {global_epoch:04d}/{args.epochs:04d} "
                f"H={stage_horizon} batch={mean_loss:.6e} "
                f"ema={float(loss_ema):.6e}{val_text} "
                f"regime=({','.join(f'{value:.3e}' for value in mean_regime_loss)}) "
                + (
                    f"failed_batches={epoch_failed_batches} "
                    f"earliest_failure_t="
                    f"{None if epoch_earliest_failure_step is None else epoch_earliest_failure_step * dt} "
                    if args.training_schedule == "continuous_trajectories"
                    else ""
                )
                +
                f"grad_mean={np.mean(epoch_grad_norms):.3e} "
                f"grad_max={np.max(epoch_grad_norms):.3e} "
                f"update_mean={np.mean(epoch_update_norms):.3e} "
                f"update_scale_min={np.min(epoch_update_scales):.3e} "
                f"elapsed={elapsed / 60.0:.1f}m"
            )
            if should_validate:
                print(
                    "[validation] panel_regime train=("
                    + ",".join(f"{value:.3e}" for value in train_eval_regime)
                    + ") heldout=("
                    + ",".join(f"{value:.3e}" for value in val_regime)
                    + ")"
                )
                if args.training_schedule == "continuous_trajectories":
                    print(
                        "[validation] heldout_autonomous_regime=("
                        + ",".join(f"{value:.3e}" for value in autonomous_regime)
                        + ")"
                    )
            completed_metadata = dict(metadata)
            completed_metadata["completed_epochs"] = global_epoch
            completed_metadata["optimizer_steps"] = int(np.asarray(optimizer["step"]))
            completed_metadata["current_learning_rate"] = float(learning_rate)
            _save_checkpoint(
                outdir / f"epoch{global_epoch:03d}_low_moment_closure.npz",
                params,
                completed_metadata,
                stats,
            )
            _save_training_state(
                outdir / "training_state.npz",
                params=params,
                optimizer=optimizer,
                rng=rng,
                global_epoch=global_epoch,
                loss_ema=loss_ema,
                best_val=best_val,
                histories={
                    "train_loss": train_history,
                    "train_ema_loss": train_ema_history,
                    "train_update_norm": train_update_norm_history,
                    "train_update_scale_min": train_update_scale_min_history,
                    "train_eval_epochs": train_eval_epochs,
                    "train_eval_loss": train_eval_history,
                    "train_eval_regime_loss": train_eval_regime_history,
                    "val_epochs": val_epochs,
                    "val_loss": val_history,
                    "val_regime_loss": val_regime_history,
                    "autonomous_val_epochs": autonomous_val_epochs,
                    "autonomous_val_loss": autonomous_val_history,
                    "autonomous_val_regime_loss": autonomous_val_regime_history,
                },
            )
        curriculum_epoch_offset = stage_end_epoch
    checkpoint = outdir / "low_moment_closure.npz"
    final_epoch_checkpoint = outdir / f"epoch{global_epoch:03d}_low_moment_closure.npz"
    if final_epoch_checkpoint.is_file():
        # The epoch checkpoint already materialized every device array.  Copy
        # that durable artifact instead of triggering a second large JAX host
        # transfer after the final accepted update.
        shutil.copy2(final_epoch_checkpoint, checkpoint)
    else:
        _save_checkpoint(checkpoint, params, metadata, stats)
    np.savez(
        outdir / "training_metrics.npz",
        train_loss=np.asarray(train_history),
        train_ema_loss=np.asarray(train_ema_history),
        train_update_norm=np.asarray(train_update_norm_history),
        train_update_scale_min=np.asarray(train_update_scale_min_history),
        train_eval_epochs=np.asarray(train_eval_epochs),
        train_eval_loss=np.asarray(train_eval_history),
        train_eval_regime_loss=np.asarray(train_eval_regime_history),
        val_epochs=np.asarray(val_epochs),
        val_loss=np.asarray(val_history),
        val_regime_loss=np.asarray(val_regime_history),
        autonomous_val_epochs=np.asarray(autonomous_val_epochs),
        autonomous_val_loss=np.asarray(autonomous_val_history),
        autonomous_val_regime_loss=np.asarray(autonomous_val_regime_history),
        metadata_json=np.array([json.dumps(metadata, sort_keys=True)]),
    )
    _plot_losses(
        train_history,
        train_ema_history,
        train_eval_epochs,
        train_eval_history,
        val_epochs,
        val_history,
        outdir / "training_loss.png",
        metadata,
    )
    print(f"Saved low-moment closure to {checkpoint}")
    if not args.skip_evaluation:
        _evaluate_heldout(
            params,
            grouped,
            manifest,
            coefficient_key,
            Path(args.reference_cache),
            outdir,
            regimes=training_regimes,
            source_nx=source_nx,
            rollout_nx=args.rollout_Nx,
            domain_length=domain_length,
            dt=dt,
            reference_dt=reference_dt,
            width=args.width,
            memory_steps=args.memory_steps,
            memory_stride=args.memory_stride,
            scan_unroll=args.scan_unroll,
            input_scale=stats["input_scale"],
            heat_flux_gradient_scale=float(stats["heat_flux_gradient_scale"][0]),
            amplitude_center=float(stats["amplitude_center"][0]),
            amplitude_scale=float(stats["amplitude_scale"][0]),
            poisson_sign=poisson_sign,
            chunk_steps=args.evaluation_chunk_steps,
            normalized_heat_flux_bound=args.normalized_heat_flux_bound,
            density_floor=args.density_floor,
            pressure_floor=args.pressure_floor,
            closure_history_input=args.closure_history_input,
            memory_backend=args.memory_backend,
            input_scaling=args.input_scaling,
            dynamic_amplitude_floor=args.dynamic_amplitude_floor,
            allow_uniform_heating=args.allow_uniform_heating,
        )


if __name__ == "__main__":
    main()
