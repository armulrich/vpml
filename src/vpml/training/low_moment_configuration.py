"""Training regimes, checkpoint compatibility, and horizon schedules."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Mapping, Tuple

import jax

from vpml.low_moment import (
    DEFAULT_DENSITY_FLOOR,
    DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    DEFAULT_PRESSURE_FLOOR,
    DYNAMIC_INPUT_SCALING,
    FIXED_INPUT_SCALING,
    INPUT_SCALING_KINDS,
)


REGIMES = (
    "linear_landau",
    "nonlinear_landau_weak",
    "nonlinear_landau_strong",
)


TRAINING_MODE = "solver_embedded_low_moment_spectral_memory"


OBJECTIVE = "low_moment_trajectory"


MODEL_BACKEND = "causal_translation_equivariant_spectral_memory"


CHECKPOINT_SCHEMA = 2


WINDOW_MEMORY_BACKENDS = frozenset(
    (
        "explicit_window",
        "causal_spacetime_operator",
        "window_fno",
        "burles_latent_fno",
    )
)


def parse_training_regimes(value: str) -> Tuple[str, ...]:
    if value == "all3":
        return REGIMES
    if value == "linear_landau":
        return ("linear_landau",)
    raise ValueError(f"Unsupported training regimes: {value!r}")


def _checkpoint_regimes(metadata: Mapping[str, object]) -> Tuple[str, ...]:
    """Old checkpoints predate selection and therefore expose all three regimes."""
    regimes = tuple(metadata.get("regimes", REGIMES))
    if regimes not in (REGIMES, ("linear_landau",)):
        raise ValueError(f"Unsupported checkpoint regimes: {regimes}")
    if "training_regimes" in metadata and parse_training_regimes(
        str(metadata["training_regimes"])
    ) != regimes:
        raise ValueError("Checkpoint training_regimes and regimes disagree")
    return regimes


def _validate_regime_resume_configuration(saved, current) -> None:
    if _checkpoint_regimes(saved) != _checkpoint_regimes(current):
        raise ValueError("Resume configuration mismatch: training regimes changed")
    # Missing fields in legacy all3 runs mean the original all3 statistics.
    for key, default in (
        ("normalization_policy", "fixed_all3_training_statistics"),
        ("normalization_regimes", list(REGIMES)),
        ("normalization_checkpoint", None),
        ("normalization_checkpoint_sha256", None),
    ):
        if saved.get(key, default) != current.get(key, default):
            raise ValueError(f"Resume configuration mismatch: {key}")
    for key in ("stats_stride", "normalization_statistics_sha256"):
        if key in saved and saved[key] != current.get(key):
            raise ValueError(f"Resume configuration mismatch: {key}")


def _uses_window_memory(memory_backend: str) -> bool:
    return memory_backend in WINDOW_MEMORY_BACKENDS


def _uses_compact_latent(memory_backend: str) -> bool:
    return memory_backend == "burles_latent_fno"


def parse_train_case_limits(value: str) -> Dict[str, int]:
    limits: Dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("training case limits must use regime=count entries")
        regime, raw_count = (part.strip() for part in item.split("=", 1))
        if regime not in REGIMES:
            raise ValueError(f"Unknown training-case-limit regime: {regime}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError("training case limits must be positive")
        limits[regime] = count
    return limits


def parse_horizon_curriculum(
    text: str,
    *,
    final_horizon: int,
    total_epochs: int,
) -> Tuple[Tuple[int, int], ...]:
    if not str(text).strip():
        return ((int(final_horizon), int(total_epochs)),)
    stages = []
    for item in str(text).split(","):
        horizon_text, separator, epochs_text = item.strip().partition(":")
        if not separator:
            raise ValueError("Horizon curriculum entries must have H:epochs form")
        horizon = int(horizon_text)
        epochs = int(epochs_text)
        if horizon <= 0 or epochs <= 0:
            raise ValueError("Horizon curriculum values must be positive")
        stages.append((horizon, epochs))
    if stages[-1][0] != int(final_horizon):
        raise ValueError("The final curriculum horizon must equal --rollout-horizon")
    if sum(epochs for _, epochs in stages) != int(total_epochs):
        raise ValueError("Horizon curriculum epoch counts must sum to --epochs")
    if any(right < left for (left, _), (right, _) in zip(stages, stages[1:])):
        raise ValueError("Horizon curriculum must be nondecreasing")
    return tuple(stages)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--training-regimes",
        choices=("all3", "linear_landau"),
        default="all3",
        help=(
            "Regimes exposed to training and diagnostics; normalization remains fixed "
            "to all3 training statistics. Checkpoint evaluation uses saved regimes."
        ),
    )
    parser.add_argument("--rollout-Nx", type=int, default=256)
    parser.add_argument(
        "--solver-dt",
        type=float,
        default=None,
        help="Low-moment solver timestep; defaults to the reference-cache timestep",
    )
    parser.add_argument("--rollout-horizon", type=int, default=1024)
    parser.add_argument(
        "--training-schedule",
        choices=("random_windows", "continuous_trajectories"),
        default="random_windows",
        help="Use teacher-reset windows or complete autonomous IC trajectories",
    )
    parser.add_argument(
        "--horizon-curriculum",
        type=str,
        default="",
        help="Comma-separated H:epochs stages ending at rollout-horizon",
    )
    parser.add_argument("--memory-steps", type=int, default=50)
    parser.add_argument("--memory-stride", type=int, default=10)
    parser.add_argument(
        "--scan-unroll",
        type=int,
        default=1,
        help="Compile this many rollout steps per scan iteration without changing dt",
    )
    parser.add_argument(
        "--memory-backend",
        choices=(
            "latent_recurrent",
            "explicit_window",
            "causal_spacetime_operator",
            "window_fno",
            "burles_latent_fno",
        ),
        default="latent_recurrent",
        help="Select recurrent, fixed-window, causal space-time, or window-FNO memory",
    )
    parser.add_argument("--spacetime-depth", type=int, default=4)
    parser.add_argument("--temporal-kernel-size", type=int, default=5)
    parser.add_argument("--fno-depth", type=int, default=4)
    parser.add_argument(
        "--latent-memory-dim",
        type=int,
        default=6,
        help="Compact learned memory channels; zero gives the matched Burles history-only control",
    )
    parser.add_argument("--history-stride", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-regime batch size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--planned-epochs",
        type=int,
        default=None,
        help="Length of the fixed learning-rate schedule, including later resumes",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        default=None,
        help="Resume model, optimizer, RNG, counters, and metrics from a run directory",
    )
    parser.add_argument("--supervised-warmup-epochs", type=int, default=0)
    parser.add_argument(
        "--supervised-only",
        action="store_true",
        help="Fit the exact heat-flux-gradient map without solver rollout",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=30)
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=0,
        help=(
            "Persist exact within-epoch state this often; currently supported "
            "for full-anchor sweeps without translation augmentation"
        ),
    )
    parser.add_argument(
        "--training-passes-per-epoch",
        type=int,
        default=1,
        help="Balanced random-window passes over every training IC per epoch",
    )
    parser.add_argument(
        "--deterministic-anchor-cycle",
        action="store_true",
        help="Use one deterministic no-replacement temporal anchor per training IC and epoch",
    )
    parser.add_argument(
        "--full-anchor-sweep",
        action="store_true",
        help="Shuffle and consume every valid training anchor exactly once per epoch",
    )
    parser.add_argument(
        "--translation-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly translate each spatial training window",
    )
    parser.add_argument(
        "--train-case-limits",
        type=str,
        default="",
        help=(
            "Diagnostic-only comma-separated regime=count limits. Selection is "
            "deterministic and nested for a fixed seed; held-out cases and training "
            "statistics remain unchanged."
        ),
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument(
        "--data-parallel-devices",
        type=int,
        default=1,
        help="Number of equal balanced JAX devices used inside each minibatch",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--update-norm-cap",
        type=float,
        default=0.0,
        help="Optional global L2 cap on each accepted AdamW parameter update",
    )
    parser.add_argument(
        "--require-first-update-descent",
        action="store_true",
        help="Abort unless the first accepted update lowers its exact accumulated batch",
    )
    parser.add_argument(
        "--report-every-update-descent",
        action="store_true",
        help="Re-evaluate each exact accumulated batch after its update and log before/after loss",
    )
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--spectral-modes", type=int, default=16)
    parser.add_argument("--closure-history-input", action="store_true")
    parser.add_argument(
        "--autonomous-history-burnin-steps",
        type=int,
        default=0,
        help="Generate closure history autonomously before each scored random window",
    )
    parser.add_argument(
        "--input-scaling",
        choices=tuple(sorted(INPUT_SCALING_KINDS)),
        default=FIXED_INPUT_SCALING,
        help="Closure input/output scaling geometry",
    )
    parser.add_argument(
        "--dynamic-amplitude-floor",
        type=float,
        default=1e-6,
        help="Numerical floor used only by current-density-RMS scaling",
    )
    parser.add_argument(
        "--allow-uniform-heating",
        action="store_true",
        help=(
            "Allow the effective closure to include a spatially uniform source; "
            "the default remains a zero-mean periodic heat-flux gradient"
        ),
    )
    parser.add_argument("--relative-trajectory-loss", action="store_true")
    parser.add_argument(
        "--global-relative-trajectory-loss",
        action="store_true",
        help="Use one relative L2 ratio over all fields, times, and points",
    )
    parser.add_argument("--log-energy-weight", type=float, default=0.0)
    parser.add_argument("--log-growth-weight", type=float, default=0.0)
    parser.add_argument(
        "--relative-time-block",
        type=float,
        default=0.0,
        help="Fixed physical-time block duration for relative trajectory loss",
    )
    parser.add_argument(
        "--convergence-floor-file",
        type=Path,
        default=None,
        help="NPZ containing regime/channel RMS teacher-grid disagreement",
    )
    parser.add_argument("--stats-stride", type=int, default=20)
    parser.add_argument(
        "--normalization-checkpoint",
        type=Path,
        default=None,
        help="Load only fixed all3 statistics from a compatible checkpoint, never weights",
    )
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--training-diagnostic-cases-per-regime", type=int, default=4)
    parser.add_argument(
        "--diagnostic-start-times",
        type=str,
        default="0,20,40,60,80,100",
    )
    parser.add_argument("--loss-ema-decay", type=float, default=0.95)
    parser.add_argument(
        "--nonfinite-trajectory-penalty",
        type=float,
        default=1e3,
        help=(
            "Fixed normalized loss assigned to an unstable autonomous suffix; "
            "finite-prefix gradients are retained"
        ),
    )
    parser.add_argument(
        "--normalized-heat-flux-bound",
        type=float,
        default=DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    )
    parser.add_argument("--density-floor", type=float, default=DEFAULT_DENSITY_FLOOR)
    parser.add_argument("--pressure-floor", type=float, default=DEFAULT_PRESSURE_FLOOR)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize trainable parameters from a compatible low-moment checkpoint",
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        default=None,
        help="Skip training and regenerate held-out diagnostics from this checkpoint",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--evaluation-chunk-steps", type=int, default=500)
    return parser


def validate_training_options(args):
    """Validate schedule choices and derive options before loading reference data."""
    training_regimes = parse_training_regimes(args.training_regimes)
    planned_epochs = int(args.epochs if args.planned_epochs is None else args.planned_epochs)
    if planned_epochs < int(args.epochs) or planned_epochs <= 0:
        raise ValueError("planned-epochs must be positive and at least epochs")
    train_case_limits = parse_train_case_limits(args.train_case_limits)
    if set(train_case_limits) - set(training_regimes):
        raise ValueError("training-case limits must refer to selected training regimes")
    if args.rollout_horizon <= 0 or args.memory_steps <= 0 or args.scan_unroll <= 0:
        raise ValueError("rollout horizon, memory steps, and scan unroll must be positive")
    if args.solver_dt is not None and args.solver_dt <= 0.0:
        raise ValueError("solver-dt must be positive")
    if args.autonomous_history_burnin_steps < 0:
        raise ValueError("autonomous-history-burnin-steps cannot be negative")
    if args.autonomous_history_burnin_steps and (
        args.training_schedule != "random_windows"
        or not _uses_window_memory(args.memory_backend)
    ):
        raise ValueError("autonomous history burn-in requires random-window window-memory training")
    if min(args.learning_rate, args.final_learning_rate) <= 0.0:
        raise ValueError("learning rates must be positive")
    if args.final_learning_rate > args.learning_rate:
        raise ValueError("final-learning-rate cannot exceed learning-rate")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay cannot be negative")
    if args.grad_clip < 0.0:
        raise ValueError("grad-clip cannot be negative; zero disables clipping")
    if args.relative_trajectory_loss and args.global_relative_trajectory_loss:
        raise ValueError("choose only one relative trajectory loss geometry")
    if args.full_anchor_sweep and args.deterministic_anchor_cycle:
        raise ValueError("full-anchor-sweep and deterministic-anchor-cycle are exclusive")
    if args.full_anchor_sweep and args.training_schedule != "random_windows":
        raise ValueError("full-anchor-sweep requires random-window training")
    if args.full_anchor_sweep and int(args.training_passes_per_epoch) != 1:
        raise ValueError("full-anchor-sweep requires one training pass per epoch")
    if args.checkpoint_every_updates < 0:
        raise ValueError("checkpoint-every-updates cannot be negative")
    if args.checkpoint_every_updates and (
        not args.full_anchor_sweep
        or args.training_schedule != "random_windows"
        or args.translation_augmentation
    ):
        raise ValueError(
            "within-epoch checkpoints require a full-anchor random-window sweep "
            "without translation augmentation"
        )
    if min(args.log_energy_weight, args.log_growth_weight) < 0.0:
        raise ValueError("energy and growth weights cannot be negative")
    if min(args.gradient_accumulation_steps, args.training_passes_per_epoch) <= 0:
        raise ValueError("gradient accumulation and training passes must be positive")
    if args.data_parallel_devices <= 0:
        raise ValueError("data-parallel-devices must be positive")
    if args.data_parallel_devices > len(jax.devices()):
        raise ValueError(
            f"Requested {args.data_parallel_devices} data-parallel devices, "
            f"but JAX exposes {len(jax.devices())}"
        )
    if args.update_norm_cap < 0.0:
        raise ValueError("update-norm-cap cannot be negative")
    if args.training_schedule == "continuous_trajectories":
        if args.horizon_curriculum.strip():
            raise ValueError("continuous trajectories require a fixed rollout horizon")
        if int(args.supervised_warmup_epochs) != 0:
            raise ValueError("continuous trajectories do not use supervised warm-up")
        if int(args.gradient_accumulation_steps) != 1:
            raise ValueError(
                "continuous trajectories already accumulate every chunk; set "
                "gradient-accumulation-steps=1"
            )
    if (
        _uses_window_memory(args.memory_backend)
        and args.supervised_warmup_epochs
        and not args.supervised_only
    ):
        raise ValueError("window-memory training does not use supervised warm-up")
    if args.memory_backend == "causal_spacetime_operator":
        if not args.closure_history_input:
            raise ValueError(
                "causal-spacetime-operator requires closure-history-input"
            )
        if min(args.spacetime_depth, args.temporal_kernel_size) <= 0:
            raise ValueError("space-time depth and temporal kernel size must be positive")
    if args.memory_backend in ("window_fno", "burles_latent_fno"):
        if args.fno_depth <= 0:
            raise ValueError("window-fno depth must be positive")
    if args.memory_backend == "burles_latent_fno":
        if not args.closure_history_input:
            raise ValueError("burles-latent-fno requires closure-history-input")
        if args.latent_memory_dim < 0:
            raise ValueError("compact latent dimension cannot be negative")
    if (
        args.input_scaling == DYNAMIC_INPUT_SCALING or args.allow_uniform_heating
    ) and not _uses_window_memory(args.memory_backend):
        raise ValueError(
            "dynamic scaling and uniform heating currently require a window-memory backend"
        )
    if args.dynamic_amplitude_floor <= 0.0:
        raise ValueError("dynamic-amplitude-floor must be positive")
    if args.supervised_only:
        if args.training_schedule != "random_windows":
            raise ValueError("supervised-only training requires random_windows")
        if args.relative_trajectory_loss:
            raise ValueError("supervised-only training does not use trajectory loss")
        if args.horizon_curriculum.strip():
            raise ValueError("supervised-only training does not use a horizon curriculum")
        if int(args.supervised_warmup_epochs) != 0:
            raise ValueError("supervised-only training does not use warm-up epochs")
    if (args.relative_time_block > 0.0) != (args.convergence_floor_file is not None):
        raise ValueError(
            "relative-time-block and convergence-floor-file must be enabled together"
        )
    if args.relative_time_block > 0.0 and not args.relative_trajectory_loss:
        raise ValueError("fixed time blocks require relative-trajectory-loss")
    if not 0.0 <= args.loss_ema_decay < 1.0:
        raise ValueError("loss EMA decay must be in [0, 1)")
    if not math.isfinite(args.nonfinite_trajectory_penalty) or (
        args.nonfinite_trajectory_penalty <= 0.0
    ):
        raise ValueError("nonfinite trajectory penalty must be finite and positive")
    if min(args.normalized_heat_flux_bound, args.density_floor, args.pressure_floor) <= 0:
        raise ValueError("heat-flux bound and admissibility floors must be positive")
    diagnostic_start_times = tuple(
        float(value) for value in args.diagnostic_start_times.split(",") if value.strip()
    )
    if not diagnostic_start_times:
        raise ValueError("at least one diagnostic start time is required")
    curriculum = ()
    if args.evaluate_checkpoint is None:
        if args.supervised_only:
            curriculum = ()
        else:
            if not 0 <= int(args.supervised_warmup_epochs) < int(args.epochs):
                raise ValueError("supervised warm-up epochs must be in [0, epochs)")
            curriculum = parse_horizon_curriculum(
                args.horizon_curriculum,
                final_horizon=args.rollout_horizon,
                total_epochs=planned_epochs - args.supervised_warmup_epochs,
            )
    return training_regimes, planned_epochs, train_case_limits, diagnostic_start_times, curriculum
