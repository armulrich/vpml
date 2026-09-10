"""Train an autonomous low-moment solver with a compact kinetic latent state."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import scipy.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.train.interface_flux_data import load_ic_manifest
from model.train.kinetic_latent_dynamics_probe import (
    REGIMES,
    _adam_init,
    _adam_step,
    _load_or_build_projected_cache,
    _selected_cases,
    _tree_all_finite,
    _tree_l2_norm,
)
from vpml.jax_runtime import print_jax_runtime_summary
from vpml.kinetic_latent import (
    advance_coupled_projected_hermite,
    advance_coupled_semilinear_strang,
    apply_coupled_linear_propagator,
    bounded_normalized_closure_correction,
    bounded_normalized_latent_correction,
    closure_aligned_latent_correction,
    closure_orthogonal_latent_correction,
    coupled_low_moment_latent_step,
    equilibrium_preserving_latent_cnn_correction,
    gated_spectral_closure_correction,
    gated_spectral_latent_correction,
    init_state_conditioned_latent_operator,
    low_moment_state_to_resolved_hermite,
    resolved_hermite_to_low_moment_state,
    rollout_coupled_low_moment_latent,
    stabilize_coupled_linear_propagator,
    state_conditioned_closure_correction,
    state_conditioned_latent_operator_features,
    state_conditioned_latent_operator_correction,
)
from vpml.low_moment import primitive_fields


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--supplemental-strong-manifest", type=Path)
    parser.add_argument("--supplemental-strong-case-ids", type=str, default="")
    parser.add_argument(
        "--supplemental-strong-case-prefix",
        type=str,
        default="discovery_",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--latent-rank", type=int, default=32)
    parser.add_argument("--basis-modes", type=int, default=64)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--cadence", type=float, default=0.1)
    parser.add_argument("--fine-steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--operator-rank", type=int, default=16)
    parser.add_argument("--operator-modes", type=int, default=0)
    parser.add_argument("--operator-output-init-scale", type=float, default=1e-3)
    parser.add_argument("--conditioner-experts", type=int, default=0)
    parser.add_argument("--latent-delay-input", action="store_true")
    parser.add_argument("--fit-multiplicative-output-readout", action="store_true")
    parser.add_argument("--multiplicative-output-ridge", type=float, default=1e-4)
    parser.add_argument(
        "--multiplicative-output-initial-scale", type=float, default=1.0
    )
    parser.add_argument("--latent-residual-weight", type=float, default=1.0)
    parser.add_argument("--latent-state-residual-weight", type=float, default=1.0)
    parser.add_argument("--low-mode-latent-residual-weight", type=float, default=0.0)
    parser.add_argument("--closure-residual-weight", type=float, default=0.0)
    parser.add_argument("--closure-aligned-output", action="store_true")
    parser.add_argument("--closure-correction-bound", type=float, default=40.0)
    parser.add_argument("--closure-only-correction", action="store_true")
    parser.add_argument(
        "--closure-readout-mode",
        choices=("multiplicative", "gated_linear", "gated_linear_residual"),
        default="multiplicative",
    )
    parser.add_argument("--closure-gate-scale", type=float, default=0.1)
    parser.add_argument("--closure-gate-power", type=int, default=2)
    parser.add_argument("--closure-readout-ridge", type=float, default=1e-4)
    parser.add_argument("--closure-readout-initial-scale", type=float, default=1.0)
    parser.add_argument(
        "--closure-readout-trajectory-exponent", type=float, default=0.75
    )
    parser.add_argument(
        "--latent-readout-mode",
        choices=(
            "equilibrium_cnn",
            "multiplicative",
            "gated_linear",
            "gated_linear_residual",
        ),
        default="multiplicative",
    )
    parser.add_argument("--latent-gate-scale", type=float, default=0.1)
    parser.add_argument("--latent-gate-power", type=int, default=2)
    parser.add_argument("--equilibrium-input-compression-scale", type=float, default=0.0)
    parser.add_argument("--latent-readout-ridge", type=float, default=1e-4)
    parser.add_argument("--latent-readout-initial-scale", type=float, default=1.0)
    parser.add_argument(
        "--latent-readout-trajectory-exponent", type=float, default=0.75
    )
    parser.add_argument("--autonomous-latent-weight", type=float, default=0.0)
    parser.add_argument("--electric-spectrum-weight", type=float, default=0.0)
    parser.add_argument("--electric-log-energy-weight", type=float, default=0.0)
    parser.add_argument("--electric-log-energy-floor-ratio", type=float, default=1e-8)
    parser.add_argument("--electric-chunk-log-growth-weight", type=float, default=0.0)
    parser.add_argument("--electric-sliding-log-growth-weight", type=float, default=0.0)
    parser.add_argument(
        "--electric-sliding-log-growth-direction",
        choices=(
            "balanced",
            "balanced_episodes",
            "growth",
            "growth_episodes",
            "damping",
        ),
        default="balanced",
    )
    parser.add_argument("--electric-growth-window-steps", type=int, default=100)
    parser.add_argument("--electric-time-relative-weight", type=float, default=0.0)
    parser.add_argument("--electric-transfer-weight", type=float, default=0.0)
    parser.add_argument(
        "--electric-time-relative-floor-ratio", type=float, default=1e-4
    )
    parser.add_argument("--post-minimum-log-energy-weight", type=float, default=0.0)
    parser.add_argument("--post-minimum-field-weight", type=float, default=0.0)
    parser.add_argument("--turnaround-field-weight", type=float, default=0.0)
    parser.add_argument("--post-peak-log-energy-weight", type=float, default=0.0)
    parser.add_argument("--peak-log-energy-weight", type=float, default=0.0)
    parser.add_argument("--turnaround-log-ratio-weight", type=float, default=0.0)
    parser.add_argument("--post-peak-retention-weight", type=float, default=0.0)
    parser.add_argument("--post-peak-retention-floor", type=float, default=0.0)
    parser.add_argument("--linear-regime-weight", type=float, default=1.0)
    parser.add_argument("--weak-regime-weight", type=float, default=1.0)
    parser.add_argument("--strong-regime-weight", type=float, default=1.0)
    parser.add_argument("--post-minimum-energy-modes", type=int, default=4)
    parser.add_argument("--latent-gradient-ratio", type=float, default=1.0)
    parser.add_argument("--teacher-internal-update-ratio", type=float, default=10.0)
    parser.add_argument("--teacher-residual-stride", type=int, default=50)
    parser.add_argument("--teacher-rollout-steps", type=int, default=10)
    parser.add_argument(
        "--training-objective",
        choices=("joint", "teacher_residual_probe"),
        default="joint",
    )
    parser.add_argument(
        "--correction-energy-tolerance", type=float, default=1e-3
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--teacher-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--update-combination",
        choices=("conflict_safe", "direct_sum"),
        default="conflict_safe",
    )
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--normalize-accumulated-gradients", action="store_true")
    parser.add_argument(
        "--gradient-aggregation",
        choices=("mean", "coordinate_median"),
        default="mean",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--parameter-update-clip", type=float, default=0.05)
    parser.add_argument("--trust-region-backtracks", type=int, default=0)
    parser.add_argument("--trust-region-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--trust-region-objective",
        choices=("total", "physical"),
        default="total",
    )
    parser.add_argument("--trust-region-batches", type=int, default=1)
    parser.add_argument("--linear-ridge", type=float, default=1e-4)
    parser.add_argument("--linear-max-spectral-radius", type=float, default=1.0)
    parser.add_argument(
        "--linear-baseline",
        choices=("fitted_propagator", "projected_hermite", "semilinear_strang"),
        default="fitted_propagator",
    )
    parser.add_argument("--hermite-tail-damping", type=float, default=10.0)
    parser.add_argument("--hermite-tail-power", type=float, default=6.0)
    parser.add_argument("--semilinear-kick-scale", type=float, default=1.0)
    parser.add_argument(
        "--semilinear-correction-location",
        choices=("midpoint", "endpoint"),
        default="midpoint",
    )
    parser.add_argument("--latent-excursion-limit", type=float, default=1000.0)
    parser.add_argument("--latent-state-bound", type=float, default=0.0)
    parser.add_argument("--train-tail-output-from", type=int, default=0)
    parser.add_argument("--train-output-rows", type=str, default="")
    parser.add_argument("--train-expert-output-only", action="store_true")
    parser.add_argument("--train-expert-phase-output-only", action="store_true")
    parser.add_argument("--train-expert-gate-only", action="store_true")
    parser.add_argument("--train-expert-cyclic-gate-only", action="store_true")
    parser.add_argument("--train-expert-cyclic-output-only", action="store_true")
    parser.add_argument("--train-cyclic-low-mode-closure-only", action="store_true")
    parser.add_argument("--train-cyclic-low-mode-pressure-only", action="store_true")
    parser.add_argument("--train-cyclic-modal-pressure-only", action="store_true")
    parser.add_argument("--train-cyclic-low-mode-momentum-only", action="store_true")
    parser.add_argument("--train-cyclic-feature-indices", type=str, default="")
    parser.add_argument("--train-expert-indices", type=str, default="")
    parser.add_argument("--weak-train-case-ids", type=str, default="")
    parser.add_argument("--strong-train-case-ids", type=str, default="")
    parser.add_argument("--linear-nonregression-limit", type=float, default=0.0)
    parser.add_argument("--gradient-chunk-steps", type=int, default=300)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--loss-ema-decay", type=float, default=0.95)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "Load model parameters from a checkpoint while starting a fresh "
            "optimizer and epoch count. Mutually exclusive with resume state."
        ),
    )
    parser.add_argument("--resume-training-state", type=Path)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def _load_complete_case(
    projected_cache: Path,
    case: dict,
    *,
    latent_rank: int,
    expected_samples: int,
    dtype=np.float32,
) -> np.ndarray:
    values = np.load(
        projected_cache / "cases" / f"{case['case_id']}.npy",
        mmap_mode="r",
    )
    if int(values.shape[0]) != int(expected_samples):
        raise ValueError(
            f"{case['case_id']} has {values.shape[0]} samples; "
            f"expected the complete {expected_samples}-sample trajectory"
        )
    return np.asarray(values[:, : 3 + int(latent_rank)], dtype=dtype)


def _balanced_complete_batch(
    projected_cache: Path,
    train_by_regime: dict[str, list[dict]],
    orders: dict[str, np.ndarray],
    step: int,
    *,
    latent_rank: int,
    expected_samples: int,
) -> tuple[np.ndarray, list[dict]]:
    rows = []
    selected_cases = []
    for regime in REGIMES:
        cases = train_by_regime[regime]
        index = int(orders[regime][step % len(cases)])
        selected_cases.append(dict(cases[index]))
        rows.append(
            _load_complete_case(
                projected_cache,
                cases[index],
                latent_rank=latent_rank,
                expected_samples=expected_samples,
            )
        )
    return np.stack(rows), selected_cases


def _fit_or_load_coupled_linear_propagator(
    projected_cache: Path,
    cases: list[dict],
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    latent_rank: int,
    nx: int,
    ridge: float,
    maximum_spectral_radius: float,
) -> tuple[np.ndarray, list[dict]]:
    """Fit a train-only full-state modal transition in scale-only coordinates."""
    path = projected_cache / (
        "coupled_linear_propagator_scale_only"
        f"_ridge{ridge:.3e}_radius{maximum_spectral_radius:.6f}.npy"
    )
    if path.exists():
        propagator = np.load(path)
        _, diagnostics = stabilize_coupled_linear_propagator(
            propagator,
            maximum_spectral_radius=maximum_spectral_radius,
        )
        print(f"[data] reusing stable coupled linear propagator {path}", flush=True)
        return propagator, diagnostics

    modes = int(nx) // 2 + 1
    channels = 3 + int(latent_rank)
    scale = np.concatenate(
        (
            np.asarray(resolved_scale, dtype=np.float64),
            np.asarray(latent_scale, dtype=np.float64),
        )
    )
    gram = np.zeros((modes, channels, channels), dtype=np.complex128)
    cross = np.zeros_like(gram)
    for case in _selected_cases(cases, "linear_landau", "train"):
        values = np.load(
            projected_cache / "cases" / f"{case['case_id']}.npy",
            mmap_mode="r",
        )[:, :channels]
        normalized = np.asarray(values, dtype=np.float64) / scale[None, :, None]
        coefficients = np.fft.rfft(normalized, axis=-1, norm="forward")
        by_mode = coefficients.transpose(0, 2, 1)
        inputs = by_mode[:-1]
        targets = by_mode[1:]
        gram += np.einsum(
            "tkc,tkd->kcd", inputs.conj(), inputs, optimize=True
        )
        cross += np.einsum(
            "tkc,tkd->kcd", inputs.conj(), targets, optimize=True
        )

    propagator = np.empty_like(cross)
    identity = np.eye(channels)
    for mode in range(modes):
        regularization = (
            float(ridge) * float(np.trace(gram[mode]).real) / float(channels)
        )
        regularization = max(regularization, np.finfo(np.float64).tiny)
        propagator[mode] = np.linalg.solve(
            gram[mode] + regularization * identity,
            cross[mode],
        )
    propagator, diagnostics = stabilize_coupled_linear_propagator(
        propagator.astype(np.complex64),
        maximum_spectral_radius=maximum_spectral_radius,
    )
    np.save(path, propagator)
    adjusted = sum(row["adjusted_eigenvalues"] for row in diagnostics)
    maximum_before = max(row["original_spectral_radius"] for row in diagnostics)
    maximum_after = max(row["final_spectral_radius"] for row in diagnostics)
    print(
        f"[data] saved stable coupled linear propagator {path} "
        f"adjusted_eigenvalues={adjusted} "
        f"maximum_radius={maximum_before:.9f}->{maximum_after:.9f}",
        flush=True,
    )
    return propagator, diagnostics


def _matrix_square_root_propagator(propagator: np.ndarray) -> np.ndarray:
    """Return mode-wise half steps whose composition recovers the propagator."""
    propagator = np.asarray(propagator)
    half = np.stack([scipy.linalg.sqrtm(mode) for mode in propagator]).astype(
        np.complex64
    )
    residuals = np.asarray(
        [
            np.linalg.norm(half[index] @ half[index] - propagator[index])
            / max(np.linalg.norm(propagator[index]), np.finfo(np.float64).tiny)
            for index in range(propagator.shape[0])
        ]
    )
    if not np.all(np.isfinite(half)) or float(np.max(residuals)) > 5e-5:
        raise FloatingPointError(
            "Unable to construct an accurate finite semilinear half propagator"
        )
    return half


def _teacher_normalized_latent_residual_blocks(
    projected_cache: Path,
    cases: list[dict],
    propagator: np.ndarray,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
):
    """Yield adjacent-step train-only residuals required beyond the linear map."""
    scale = np.concatenate(
        (
            np.asarray(resolved_scale, dtype=np.float64),
            np.asarray(latent_scale, dtype=np.float64),
        )
    )
    propagator = np.asarray(propagator, dtype=np.complex128)
    for case in cases:
        if str(case["split"]) != "train":
            continue
        values = np.asarray(
            np.load(
                projected_cache / "cases" / f"{case['case_id']}.npy",
                mmap_mode="r",
            ),
            dtype=np.float64,
        )
        normalized = values / scale[None, :, None]
        coefficients = np.fft.rfft(normalized, axis=-1, norm="forward")
        predicted = np.einsum(
            "tck,kcd->tdk",
            coefficients[:-1],
            propagator,
            optimize=True,
        )
        residual_hat = coefficients[1:, 3:] - predicted[:, 3:]
        yield np.fft.irfft(
            residual_hat,
            n=values.shape[-1],
            axis=-1,
            norm="forward",
        )


def _smooth_correction_distortion(
    residual_blocks,
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return channelwise residual and smooth-limiter distortion energies."""
    bounds = np.asarray(bounds, dtype=np.float64)
    total = np.zeros_like(bounds)
    distortion = np.zeros_like(bounds)
    for residual in residual_blocks():
        residual = np.asarray(residual, dtype=np.float64)
        limited = bounds[None, :, None] * np.tanh(
            residual / bounds[None, :, None]
        )
        total += np.sum(np.square(residual), axis=(0, 2), dtype=np.float64)
        distortion += np.sum(
            np.square(residual - limited), axis=(0, 2), dtype=np.float64
        )
    return total, distortion


def _calibrate_smooth_correction_bounds(
    residual_blocks,
    *,
    channels: int,
    energy_tolerance: float,
    bisection_steps: int = 12,
) -> tuple[np.ndarray, dict]:
    """Calibrate global channelwise bounds from train-only residual energy."""
    if not 0.0 < float(energy_tolerance) < 1.0:
        raise ValueError("energy_tolerance must lie in (0, 1)")
    maximum = np.zeros((int(channels),), dtype=np.float64)
    total = np.zeros_like(maximum)
    for residual in residual_blocks():
        residual = np.asarray(residual, dtype=np.float64)
        if residual.ndim != 3 or residual.shape[1] != int(channels):
            raise ValueError("residual blocks must have shape (time, channels, x)")
        maximum = np.maximum(maximum, np.max(np.abs(residual), axis=(0, 2)))
        total += np.sum(np.square(residual), axis=(0, 2), dtype=np.float64)

    active = total > np.finfo(np.float64).tiny
    lower = np.ones_like(maximum)
    upper = np.maximum(lower, maximum)
    for _ in range(12):
        _, distortion = _smooth_correction_distortion(residual_blocks, upper)
        ratio = np.divide(
            distortion,
            total,
            out=np.zeros_like(distortion),
            where=active,
        )
        failing = active & (ratio > float(energy_tolerance))
        if not np.any(failing):
            break
        upper[failing] *= 2.0
    else:
        raise RuntimeError("Unable to calibrate finite correction bounds")

    for _ in range(int(bisection_steps)):
        middle = np.sqrt(lower * upper)
        _, distortion = _smooth_correction_distortion(residual_blocks, middle)
        ratio = np.divide(
            distortion,
            total,
            out=np.zeros_like(distortion),
            where=active,
        )
        failing = active & (ratio > float(energy_tolerance))
        lower = np.where(failing, middle, lower)
        upper = np.where(failing, upper, middle)

    bounds = np.where(active, upper, 1.0)
    total, distortion = _smooth_correction_distortion(residual_blocks, bounds)
    ratio = np.divide(
        distortion,
        total,
        out=np.zeros_like(distortion),
        where=active,
    )
    diagnostics = {
        "maximum_required_correction": maximum,
        "residual_energy": total,
        "distortion_energy": distortion,
        "channel_distortion_fraction": ratio,
        "overall_distortion_fraction": float(
            np.sum(distortion) / np.sum(total)
        ),
    }
    return bounds.astype(np.float32), diagnostics


def _calibrate_conservative_correction_bounds(
    residual_blocks,
    *,
    channels: int,
    energy_tolerance: float,
) -> tuple[np.ndarray, dict]:
    """Calibrate broad smooth bounds in two passes over an expensive baseline."""
    total = np.zeros((int(channels),), dtype=np.float64)
    maximum = np.zeros_like(total)
    for residual in residual_blocks():
        values = np.asarray(residual, dtype=np.float64)
        total += np.sum(np.square(values), axis=(0, 2), dtype=np.float64)
        maximum = np.maximum(maximum, np.max(np.abs(values), axis=(0, 2)))
    active = total > 0.0
    bounds = np.where(active, np.maximum(1.0, 5.0 * maximum), 1.0)
    for _ in range(4):
        _, distortion = _smooth_correction_distortion(residual_blocks, bounds)
        ratio = np.divide(
            distortion, total, out=np.zeros_like(distortion), where=active
        )
        if np.all((~active) | (ratio <= float(energy_tolerance))):
            break
        bounds = np.where(ratio > float(energy_tolerance), 2.0 * bounds, bounds)
    else:
        raise RuntimeError("Unable to calibrate conservative correction bounds")
    diagnostics = {
        "maximum_required_correction": maximum,
        "residual_energy": total,
        "distortion_energy": distortion,
        "channel_distortion_fraction": ratio,
        "overall_distortion_fraction": float(np.sum(distortion) / np.sum(total)),
    }
    return bounds.astype(np.float32), diagnostics


def _fit_or_load_correction_bounds(
    projected_cache: Path,
    cases: list[dict],
    propagator: np.ndarray,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    linear_ridge: float,
    maximum_spectral_radius: float,
    energy_tolerance: float,
    linear_baseline: str = "fitted_propagator",
    basis: np.ndarray | None = None,
    k_arr: np.ndarray | None = None,
    fine_steps: int = 10,
    fine_dt: float = 0.01,
    hermite_tail_damping: float = 10.0,
    hermite_tail_power: float = 6.0,
    semilinear_kick_scale: float = 1.0,
    semilinear_correction_location: str = "midpoint",
    poisson_sign: float = 1.0,
) -> tuple[np.ndarray, dict]:
    if linear_baseline == "fitted_propagator":
        stem = (
            "coupled_nonlinear_correction_bounds"
            f"_ridge{linear_ridge:.3e}"
            f"_radius{maximum_spectral_radius:.6f}"
        )
    elif linear_baseline == "projected_hermite":
        stem = (
            "coupled_nonlinear_correction_bounds"
            f"_hermiteVlasovDamp{hermite_tail_damping:.3e}"
            f"_power{hermite_tail_power:.3f}"
        )
    else:
        stem = (
            "coupled_nonlinear_correction_bounds"
            f"_semilinearStrangKick{semilinear_kick_scale:.6f}"
            f"_{semilinear_correction_location}"
            f"_ridge{linear_ridge:.3e}"
            f"_radius{maximum_spectral_radius:.6f}"
        )
    path = projected_cache / stem
    path = path.with_name(path.name + f"_tol{energy_tolerance:.3e}.npz")
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            bounds = np.asarray(payload["bounds"], dtype=np.float32)
            diagnostics = {
                key: np.asarray(payload[key])
                for key in (
                    "maximum_required_correction",
                    "residual_energy",
                    "distortion_energy",
                    "channel_distortion_fraction",
                )
            }
            diagnostics["overall_distortion_fraction"] = float(
                payload["overall_distortion_fraction"]
            )
        print(f"[data] reusing calibrated correction bounds {path}", flush=True)
        return bounds, diagnostics

    def residual_blocks():
        if linear_baseline == "projected_hermite":
            if basis is None or k_arr is None:
                raise ValueError(
                    "projected-Hermite correction bounds require basis and k_arr"
                )
            return _projected_hermite_normalized_latent_residual_blocks(
                projected_cache,
                cases,
                basis,
                k_arr,
                latent_scale=latent_scale,
                fine_steps=fine_steps,
                fine_dt=fine_dt,
                hermite_tail_damping=hermite_tail_damping,
                hermite_tail_power=hermite_tail_power,
                poisson_sign=poisson_sign,
            )
        if linear_baseline == "semilinear_strang":
            if basis is None or k_arr is None:
                raise ValueError(
                    "semilinear correction bounds require basis and k_arr"
                )
            return _semilinear_normalized_latent_residual_blocks(
                projected_cache,
                cases,
                propagator,
                basis,
                k_arr,
                resolved_scale=resolved_scale,
                latent_scale=latent_scale,
                fine_steps=fine_steps,
                fine_dt=fine_dt,
                kick_scale=semilinear_kick_scale,
                poisson_sign=poisson_sign,
                midpoint=semilinear_correction_location == "midpoint",
            )
        return _teacher_normalized_latent_residual_blocks(
            projected_cache,
            cases,
            propagator,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
        )

    calibration = (
        _calibrate_conservative_correction_bounds
        if linear_baseline == "projected_hermite"
        else _calibrate_smooth_correction_bounds
    )
    bounds, diagnostics = calibration(
        residual_blocks,
        channels=int(np.asarray(latent_scale).size),
        energy_tolerance=float(energy_tolerance),
    )
    np.savez_compressed(path, bounds=bounds, **diagnostics)
    print(f"[data] saved calibrated correction bounds {path}", flush=True)
    return bounds, diagnostics


def _fit_or_load_latent_residual_scale(
    projected_cache: Path,
    cases: list[dict],
    propagator: np.ndarray,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    linear_ridge: float,
    maximum_spectral_radius: float,
    linear_baseline: str = "fitted_propagator",
    basis: np.ndarray | None = None,
    k_arr: np.ndarray | None = None,
    fine_steps: int = 10,
    fine_dt: float = 0.01,
    hermite_tail_damping: float = 10.0,
    hermite_tail_power: float = 6.0,
    semilinear_kick_scale: float = 1.0,
    poisson_sign: float = 1.0,
) -> np.ndarray:
    """Return train-only channel RMS scales for the nonlinear latent residual."""
    if linear_baseline == "fitted_propagator":
        stem = (
            "coupled_nonlinear_residual_rms"
            f"_ridge{linear_ridge:.3e}"
            f"_radius{maximum_spectral_radius:.6f}.npy"
        )
    elif linear_baseline == "projected_hermite":
        stem = (
            "coupled_nonlinear_residual_rms"
            f"_hermiteVlasovDamp{hermite_tail_damping:.3e}"
            f"_power{hermite_tail_power:.3f}.npy"
        )
    else:
        stem = (
            "coupled_nonlinear_residual_rms"
            f"_semilinearStrangKick{semilinear_kick_scale:.6f}"
            f"_ridge{linear_ridge:.3e}"
            f"_radius{maximum_spectral_radius:.6f}.npy"
        )
    path = projected_cache / stem
    if path.exists():
        scale = np.asarray(np.load(path), dtype=np.float32)
        if (
            scale.shape != np.asarray(latent_scale).shape
            or np.any(~np.isfinite(scale))
            or np.any(scale <= 0.0)
        ):
            raise ValueError(f"Invalid cached nonlinear residual scale: {path}")
        print(f"[data] reusing train-only nonlinear residual scale {path}", flush=True)
        return scale

    total = np.zeros_like(np.asarray(latent_scale), dtype=np.float64)
    count = 0
    if linear_baseline == "projected_hermite":
        if basis is None or k_arr is None:
            raise ValueError(
                "projected-Hermite residual scale requires basis and k_arr"
            )
        residual_blocks = _projected_hermite_normalized_latent_residual_blocks(
            projected_cache,
            cases,
            basis,
            k_arr,
            latent_scale=latent_scale,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            hermite_tail_damping=hermite_tail_damping,
            hermite_tail_power=hermite_tail_power,
            poisson_sign=poisson_sign,
        )
    elif linear_baseline == "semilinear_strang":
        if basis is None or k_arr is None:
            raise ValueError(
                "semilinear residual scale requires basis and k_arr"
            )
        residual_blocks = _semilinear_normalized_latent_residual_blocks(
            projected_cache,
            cases,
            propagator,
            basis,
            k_arr,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            kick_scale=semilinear_kick_scale,
            poisson_sign=poisson_sign,
            midpoint=False,
        )
    else:
        residual_blocks = _teacher_normalized_latent_residual_blocks(
            projected_cache,
            cases,
            propagator,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
        )
    for residual in residual_blocks:
        residual = np.asarray(residual, dtype=np.float64)
        total += np.sum(np.square(residual), axis=(0, 2), dtype=np.float64)
        count += int(residual.shape[0] * residual.shape[2])
    if count <= 0:
        raise ValueError("No train-only latent residual samples were available")
    scale = np.sqrt(total / float(count))
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("Train-only nonlinear residual scale is not finite and positive")
    scale = scale.astype(np.float32)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, scale)
    os.replace(temporary, path)
    print(f"[data] saved train-only nonlinear residual scale {path}", flush=True)
    return scale


def _projected_hermite_baseline_latent_rows(
    values: np.ndarray,
    basis: np.ndarray,
    k_arr: np.ndarray,
    *,
    fine_steps: int,
    fine_dt: float,
    hermite_tail_damping: float,
    hermite_tail_power: float,
    poisson_sign: float,
) -> np.ndarray:
    """Advance teacher states once with the fixed projected-Hermite baseline."""
    baseline_step = jax.jit(
        lambda resolved, latent: advance_coupled_projected_hermite(
            resolved_hermite_to_low_moment_state(resolved),
            latent,
            jnp.zeros_like(latent),
            jnp.asarray(basis),
            jnp.asarray(k_arr),
            fine_steps=int(fine_steps),
            fine_dt=float(fine_dt),
            tail_damping=float(hermite_tail_damping),
            tail_power=float(hermite_tail_power),
            poisson_sign=float(poisson_sign),
        )[1]
    )
    rows = []
    for start in range(0, values.shape[0] - 1, 256):
        stop = min(start + 256, values.shape[0] - 1)
        rows.append(
            np.asarray(
                baseline_step(
                    jnp.asarray(values[start:stop, :3], dtype=jnp.float32),
                    jnp.asarray(values[start:stop, 3:], dtype=jnp.float32),
                ),
                dtype=np.float64,
            )
        )
    return np.concatenate(rows, axis=0)


def _projected_hermite_normalized_latent_residual_blocks(
    projected_cache: Path,
    cases: list[dict],
    basis: np.ndarray,
    k_arr: np.ndarray,
    *,
    latent_scale: np.ndarray,
    fine_steps: int,
    fine_dt: float,
    hermite_tail_damping: float,
    hermite_tail_power: float,
    poisson_sign: float,
):
    """Yield train-only one-step residuals of the nonlinear physical baseline."""
    latent_scale = np.asarray(latent_scale, dtype=np.float64)
    cache_dir = projected_cache / (
        "projected_hermite_vlasov_residuals"
        f"_damp{hermite_tail_damping:.3e}"
        f"_power{hermite_tail_power:.3f}"
    )
    cache_dir.mkdir(exist_ok=True)
    for case in cases:
        if str(case["split"]) != "train":
            continue
        residual_path = cache_dir / f"{case['case_id']}.npy"
        if residual_path.exists():
            yield np.load(residual_path, mmap_mode="r")
            continue
        values = np.asarray(
            np.load(
                projected_cache / "cases" / f"{case['case_id']}.npy",
                mmap_mode="r",
            )[:, : 3 + latent_scale.size],
            dtype=np.float64,
        )
        baseline_latent = _projected_hermite_baseline_latent_rows(
            values,
            basis,
            k_arr,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            hermite_tail_damping=hermite_tail_damping,
            hermite_tail_power=hermite_tail_power,
            poisson_sign=poisson_sign,
        )
        residual = (
            (values[1:, 3:] - baseline_latent)
            / latent_scale[None, :, None]
        ).astype(np.float32)
        temporary = residual_path.with_name(f".{residual_path.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, residual)
        os.replace(temporary, residual_path)
        print(
            f"[data] cached projected-Vlasov residual {case['case_id']}",
            flush=True,
        )
        yield np.load(residual_path, mmap_mode="r")


def _make_semilinear_strang_baseline_step(
    propagator: np.ndarray,
    basis: np.ndarray,
    k_arr: np.ndarray,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    fine_steps: int,
    fine_dt: float,
    kick_scale: float,
    poisson_sign: float,
):
    """Compile one reusable fixed semilinear Strang baseline step."""
    half_propagator = _matrix_square_root_propagator(propagator)
    return jax.jit(
        lambda resolved, latent: advance_coupled_semilinear_strang(
            resolved_hermite_to_low_moment_state(resolved),
            latent,
            jnp.zeros_like(latent),
            jnp.asarray(half_propagator),
            jnp.asarray(basis),
            jnp.asarray(k_arr),
            resolved_scale=jnp.asarray(resolved_scale),
            latent_scale=jnp.asarray(latent_scale),
            fine_steps=int(fine_steps),
            fine_dt=float(fine_dt),
            kick_scale=float(kick_scale),
            poisson_sign=float(poisson_sign),
        )
    )


def _semilinear_strang_baseline_rows(
    values: np.ndarray,
    propagator: np.ndarray,
    basis: np.ndarray,
    k_arr: np.ndarray,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    fine_steps: int,
    fine_dt: float,
    kick_scale: float,
    poisson_sign: float,
    baseline_step=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance teacher states once with the fixed semilinear Strang map."""
    if baseline_step is None:
        baseline_step = _make_semilinear_strang_baseline_step(
            propagator,
            basis,
            k_arr,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            kick_scale=kick_scale,
            poisson_sign=poisson_sign,
        )
    resolved_rows = []
    latent_rows = []
    for start in range(0, values.shape[0] - 1, 256):
        stop = min(start + 256, values.shape[0] - 1)
        state, latent = baseline_step(
            jnp.asarray(values[start:stop, :3], dtype=jnp.float32),
            jnp.asarray(values[start:stop, 3:], dtype=jnp.float32),
        )
        resolved_rows.append(
            np.asarray(low_moment_state_to_resolved_hermite(state), dtype=np.float64)
        )
        latent_rows.append(np.asarray(latent, dtype=np.float64))
    return np.concatenate(resolved_rows), np.concatenate(latent_rows)


def _semilinear_midpoint_correction(
    target_next: np.ndarray,
    baseline_resolved: np.ndarray,
    baseline_latent: np.ndarray,
    half_propagator: np.ndarray,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
) -> np.ndarray:
    """Infer the minimum-norm midpoint latent correction from final defects."""
    scale = np.concatenate((resolved_scale, latent_scale)).astype(np.float64)
    baseline = np.concatenate((baseline_resolved, baseline_latent), axis=1)
    normalized_defect = (target_next - baseline) / scale[None, :, None]
    defect_hat = np.fft.rfft(normalized_defect, axis=-1, norm="forward")
    latent_to_output = np.asarray(half_propagator, dtype=np.complex128)[:, 3:, :]
    inverse = np.stack(
        [np.linalg.pinv(matrix, rcond=1e-8) for matrix in latent_to_output]
    )
    correction_hat = np.einsum(
        "tdk,kdr->trk", defect_hat, inverse, optimize=True
    )
    return np.fft.irfft(
        correction_hat,
        n=target_next.shape[-1],
        axis=-1,
        norm="forward",
    ).real


def _semilinear_endpoint_latent_correction(
    normalized_midpoint_correction: np.ndarray,
    half_propagator: np.ndarray,
) -> np.ndarray:
    """Map a normalized midpoint latent source through the second half-step."""
    correction_hat = np.fft.rfft(
        normalized_midpoint_correction, axis=-1, norm="forward"
    )
    endpoint_hat = np.einsum(
        "tck,kcd->tdk",
        correction_hat,
        np.asarray(half_propagator)[:, 3:, 3:],
        optimize=True,
    )
    return np.fft.irfft(
        endpoint_hat,
        n=normalized_midpoint_correction.shape[-1],
        axis=-1,
        norm="forward",
    ).real


def _semilinear_normalized_latent_residual_blocks(
    projected_cache: Path,
    cases: list[dict],
    propagator: np.ndarray,
    basis: np.ndarray,
    k_arr: np.ndarray,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    fine_steps: int,
    fine_dt: float,
    kick_scale: float,
    poisson_sign: float,
    midpoint: bool,
):
    """Yield cached train-only semilinear midpoint or final latent defects."""
    kind = "midpoint" if midpoint else "final"
    cache_dir = projected_cache / (
        "semilinear_strang_residuals"
        f"_kick{kick_scale:.6f}_{kind}"
    )
    cache_dir.mkdir(exist_ok=True)
    half_propagator = _matrix_square_root_propagator(propagator)
    baseline_step = _make_semilinear_strang_baseline_step(
        propagator,
        basis,
        k_arr,
        resolved_scale=resolved_scale,
        latent_scale=latent_scale,
        fine_steps=fine_steps,
        fine_dt=fine_dt,
        kick_scale=kick_scale,
        poisson_sign=poisson_sign,
    )
    for case in cases:
        if str(case["split"]) != "train":
            continue
        residual_path = cache_dir / f"{case['case_id']}.npy"
        if residual_path.exists():
            yield np.load(residual_path, mmap_mode="r")
            continue
        values = np.asarray(
            np.load(
                projected_cache / "cases" / f"{case['case_id']}.npy",
                mmap_mode="r",
            )[:, : 3 + latent_scale.size],
            dtype=np.float64,
        )
        baseline_resolved, baseline_latent = _semilinear_strang_baseline_rows(
            values,
            propagator,
            basis,
            k_arr,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            kick_scale=kick_scale,
            poisson_sign=poisson_sign,
            baseline_step=baseline_step,
        )
        if midpoint:
            residual = _semilinear_midpoint_correction(
                values[1:],
                baseline_resolved,
                baseline_latent,
                half_propagator,
                resolved_scale=resolved_scale,
                latent_scale=latent_scale,
            )
        else:
            residual = (
                values[1:, 3:] - baseline_latent
            ) / latent_scale[None, :, None]
        residual = np.asarray(residual, dtype=np.float32)
        temporary = residual_path.with_name(f".{residual_path.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, residual)
        os.replace(temporary, residual_path)
        print(
            f"[data] cached semilinear {kind} residual {case['case_id']}",
            flush=True,
        )
        yield np.load(residual_path, mmap_mode="r")


def _fit_or_load_closure_residual_scale(
    projected_cache: Path,
    cases: list[dict],
    propagator: np.ndarray,
    basis: np.ndarray,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    linear_ridge: float,
    maximum_spectral_radius: float,
    linear_baseline: str = "fitted_propagator",
    k_arr: np.ndarray | None = None,
    fine_steps: int = 10,
    fine_dt: float = 0.01,
    hermite_tail_damping: float = 10.0,
    hermite_tail_power: float = 6.0,
    semilinear_kick_scale: float = 1.0,
    poisson_sign: float = 1.0,
) -> float:
    """Return the train-only RMS of the one-step C3 residual."""
    if linear_baseline == "fitted_propagator":
        stem = (
            "coupled_closure_residual_rms"
            f"_ridge{linear_ridge:.3e}"
            f"_radius{maximum_spectral_radius:.6f}.npy"
        )
    elif linear_baseline == "projected_hermite":
        stem = (
            "coupled_closure_residual_rms"
            f"_hermiteVlasovDamp{hermite_tail_damping:.3e}"
            f"_power{hermite_tail_power:.3f}.npy"
        )
    else:
        stem = (
            "coupled_closure_residual_rms"
            f"_semilinearStrangKick{semilinear_kick_scale:.6f}"
            f"_ridge{linear_ridge:.3e}"
            f"_radius{maximum_spectral_radius:.6f}.npy"
        )
    path = projected_cache / stem
    if path.exists():
        scale = float(np.asarray(np.load(path)))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid cached closure residual scale: {path}")
        print(f"[data] reusing train-only closure residual scale {path}", flush=True)
        return scale

    basis_row = np.asarray(basis, dtype=np.float64)[0]
    latent_scale = np.asarray(latent_scale, dtype=np.float64)
    total = 0.0
    count = 0
    for case in cases:
        if str(case["split"]) != "train":
            continue
        values = np.asarray(
            np.load(
                projected_cache / "cases" / f"{case['case_id']}.npy",
                mmap_mode="r",
            )[:, : 3 + latent_scale.size],
            dtype=np.float64,
        )
        if linear_baseline == "projected_hermite":
            if k_arr is None:
                raise ValueError("projected-Hermite closure scale requires k_arr")
            baseline_latent = _projected_hermite_baseline_latent_rows(
                values,
                basis,
                k_arr,
                fine_steps=fine_steps,
                fine_dt=fine_dt,
                hermite_tail_damping=hermite_tail_damping,
                hermite_tail_power=hermite_tail_power,
                poisson_sign=poisson_sign,
            )
            physical_residual = values[1:, 3:] - baseline_latent
        elif linear_baseline == "semilinear_strang":
            if k_arr is None:
                raise ValueError("semilinear closure scale requires k_arr")
            _, baseline_latent = _semilinear_strang_baseline_rows(
                values,
                propagator,
                basis,
                k_arr,
                resolved_scale=resolved_scale,
                latent_scale=latent_scale,
                fine_steps=fine_steps,
                fine_dt=fine_dt,
                kick_scale=semilinear_kick_scale,
                poisson_sign=poisson_sign,
            )
            physical_residual = values[1:, 3:] - baseline_latent
        else:
            scale = np.concatenate((resolved_scale, latent_scale))
            coefficients = np.fft.rfft(
                values / scale[None, :, None], axis=-1, norm="forward"
            )
            linear_hat = np.einsum(
                "tck,kcd->tdk", coefficients[:-1], propagator, optimize=True
            )
            physical_residual = np.fft.irfft(
                coefficients[1:, 3:] - linear_hat[:, 3:],
                n=values.shape[-1],
                axis=-1,
                norm="forward",
            ) * latent_scale[None, :, None]
        closure_residual = np.einsum(
            "r,trx->tx", basis_row, physical_residual, optimize=True
        )
        total += float(np.sum(np.square(closure_residual), dtype=np.float64))
        count += int(closure_residual.size)
    if count <= 0:
        raise ValueError("No train-only closure residual samples were available")
    scale = math.sqrt(total / float(count))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Train-only closure residual scale is not finite and positive")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(scale, dtype=np.float32))
    os.replace(temporary, path)
    print(f"[data] saved train-only closure residual scale {path}", flush=True)
    return float(scale)


def _fit_or_load_gated_closure_readout(
    projected_cache: Path,
    cases: list[dict],
    propagator: np.ndarray,
    basis: np.ndarray,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    closure_residual_scale: float,
    linear_ridge: float,
    maximum_spectral_radius: float,
    gate_scale: float,
    gate_power: int,
    readout_ridge: float,
    trajectory_exponent: float,
    linear_baseline: str = "fitted_propagator",
    k_arr: np.ndarray | None = None,
    fine_steps: int = 10,
    fine_dt: float = 0.01,
    hermite_tail_damping: float = 10.0,
    hermite_tail_power: float = 6.0,
    poisson_sign: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Fit one shared train-only map from gated latent state to missing C3."""
    path = projected_cache / (
        "coupled_gated_closure_readout"
        f"_linearRidge{linear_ridge:.3e}"
        f"_radius{maximum_spectral_radius:.6f}"
        f"_gate{gate_scale:.3e}"
        f"_power{int(gate_power)}"
        f"_ridge{readout_ridge:.3e}"
        f"_alpha{trajectory_exponent:.3f}.npz"
        if linear_baseline == "fitted_propagator"
        else (
            "coupled_gated_closure_readout"
            f"_gate{gate_scale:.3e}"
            f"_power{int(gate_power)}"
            f"_ridge{readout_ridge:.3e}"
            f"_alpha{trajectory_exponent:.3f}"
            f"_hermiteVlasovDamp{hermite_tail_damping:.3e}"
            f"_power{hermite_tail_power:.3f}.npz"
        )
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            weights = np.asarray(payload["weights"], dtype=np.complex64)
            diagnostics = {
                "train_relative_mse": float(payload["train_relative_mse"]),
                "weight_rms": float(payload["weight_rms"]),
                "weight_maximum": float(payload["weight_maximum"]),
            }
        print(f"[data] reusing train-only gated closure readout {path}", flush=True)
        return weights, diagnostics

    if not np.isfinite(float(gate_scale)) or float(gate_scale) <= 0.0:
        raise ValueError("closure gate scale must be finite and positive")
    if not np.isfinite(float(readout_ridge)) or float(readout_ridge) < 0.0:
        raise ValueError("closure readout ridge must be finite and nonnegative")
    if not 0.0 <= float(trajectory_exponent) <= 1.0:
        raise ValueError("closure trajectory exponent must lie in [0, 1]")

    resolved_scale = np.asarray(resolved_scale, dtype=np.float64)
    latent_scale = np.asarray(latent_scale, dtype=np.float64)
    scale = np.concatenate((resolved_scale, latent_scale))
    basis_row = np.asarray(basis, dtype=np.float64)[0]
    propagator = np.asarray(propagator, dtype=np.complex128)
    channels = int(scale.size)
    modes = int(propagator.shape[0])
    gram = np.zeros((modes, channels, channels), dtype=np.complex128)
    cross = np.zeros((modes, channels), dtype=np.complex128)
    cached_rows = []

    for case in cases:
        if str(case["split"]) != "train":
            continue
        values = np.asarray(
            np.load(
                projected_cache / "cases" / f"{case['case_id']}.npy",
                mmap_mode="r",
            )[:, :channels],
            dtype=np.float64,
        )
        normalized = values / scale[None, :, None]
        if linear_baseline == "projected_hermite":
            if k_arr is None:
                raise ValueError("projected-Hermite closure fit requires k_arr")
            baseline_latent = _projected_hermite_baseline_latent_rows(
                values,
                basis,
                k_arr,
                fine_steps=fine_steps,
                fine_dt=fine_dt,
                hermite_tail_damping=hermite_tail_damping,
                hermite_tail_power=hermite_tail_power,
                poisson_sign=poisson_sign,
            )
            physical_latent_residual = values[1:, 3:] - baseline_latent
        else:
            coefficients = np.fft.rfft(normalized, axis=-1, norm="forward")
            linear_hat = np.einsum(
                "tck,kcd->tdk",
                coefficients[:-1],
                propagator,
                optimize=True,
            )
            physical_latent_residual = np.fft.irfft(
                coefficients[1:, 3:] - linear_hat[:, 3:],
                n=values.shape[-1],
                axis=-1,
                norm="forward",
            ) * latent_scale[None, :, None]
        target = np.einsum(
            "r,trx->tx",
            basis_row,
            physical_latent_residual,
            optimize=True,
        ) / float(closure_residual_scale)
        latent_amplitude = np.sqrt(
            np.mean(np.square(normalized[:-1, 3:]), axis=(1, 2))
        )
        gate = np.power(latent_amplitude, int(gate_power)) / (
            np.power(latent_amplitude, int(gate_power))
            + float(gate_scale) ** int(gate_power)
        )
        features = gate[:, None, None] * normalized[:-1]
        features_hat = np.fft.rfft(features, axis=-1, norm="forward")
        target_hat = np.fft.rfft(target, axis=-1, norm="forward")
        target_energy = float(np.mean(np.square(target), dtype=np.float64))
        trajectory_weight = max(
            target_energy, np.finfo(np.float64).tiny
        ) ** (-float(trajectory_exponent))
        gram += trajectory_weight * np.einsum(
            "tck,tdk->kcd",
            features_hat.conj(),
            features_hat,
            optimize=True,
        )
        cross += trajectory_weight * np.einsum(
            "tck,tk->kc",
            features_hat.conj(),
            target_hat,
            optimize=True,
        )
        cached_rows.append((features_hat, target_hat, trajectory_weight))

    if not cached_rows:
        raise ValueError("No train-only trajectories were available for closure fit")
    weights = np.zeros((modes, channels), dtype=np.complex128)
    identity = np.eye(channels, dtype=np.complex128)
    for mode in range(modes):
        regularization = (
            float(readout_ridge)
            * float(np.trace(gram[mode]).real)
            / float(channels)
        )
        regularization = max(regularization, np.finfo(np.float64).tiny)
        weights[mode] = np.linalg.solve(
            gram[mode] + regularization * identity,
            cross[mode],
        )
    weights[0] = weights[0].real
    if (modes - 1) * 2 == int(values.shape[-1]):
        weights[-1] = weights[-1].real

    error = 0.0
    total = 0.0
    for features_hat, target_hat, trajectory_weight in cached_rows:
        predicted_hat = np.einsum(
            "tck,kc->tk", features_hat, weights, optimize=True
        )
        error += trajectory_weight * float(
            np.sum(np.square(np.abs(predicted_hat - target_hat)), dtype=np.float64)
        )
        total += trajectory_weight * float(
            np.sum(np.square(np.abs(target_hat)), dtype=np.float64)
        )
    diagnostics = {
        "train_relative_mse": error / total,
        "weight_rms": float(np.sqrt(np.mean(np.square(np.abs(weights))))),
        "weight_maximum": float(np.max(np.abs(weights))),
    }
    weights = weights.astype(np.complex64)
    np.savez_compressed(path, weights=weights, **diagnostics)
    print(f"[data] saved train-only gated closure readout {path}", flush=True)
    return weights, diagnostics


def _fit_or_load_gated_latent_readout(
    projected_cache: Path,
    cases: list[dict],
    propagator: np.ndarray,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    linear_ridge: float,
    maximum_spectral_radius: float,
    gate_scale: float,
    gate_power: int,
    readout_ridge: float,
    trajectory_exponent: float,
    linear_baseline: str = "fitted_propagator",
    basis: np.ndarray | None = None,
    k_arr: np.ndarray | None = None,
    fine_steps: int = 10,
    fine_dt: float = 0.01,
    hermite_tail_damping: float = 10.0,
    hermite_tail_power: float = 6.0,
    poisson_sign: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Fit a shared train-only map to the complete nonlinear latent residual."""
    path = projected_cache / (
        "coupled_gated_latent_readout"
        f"_linearRidge{linear_ridge:.3e}"
        f"_radius{maximum_spectral_radius:.6f}"
        f"_gate{gate_scale:.3e}"
        f"_power{int(gate_power)}"
        f"_ridge{readout_ridge:.3e}"
        f"_alpha{trajectory_exponent:.3f}"
        + (
            f"_hermiteVlasovDamp{hermite_tail_damping:.3e}"
            f"_power{hermite_tail_power:.3f}"
            if linear_baseline == "projected_hermite"
            else ""
        )
        + ".npz"
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            weights = np.asarray(payload["weights"], dtype=np.complex64)
            diagnostics = {
                "train_relative_mse": float(payload["train_relative_mse"]),
                "weight_rms": float(payload["weight_rms"]),
                "weight_maximum": float(payload["weight_maximum"]),
            }
        print(f"[data] reusing train-only gated latent readout {path}", flush=True)
        return weights, diagnostics

    resolved_scale = np.asarray(resolved_scale, dtype=np.float64)
    latent_scale = np.asarray(latent_scale, dtype=np.float64)
    scale = np.concatenate((resolved_scale, latent_scale))
    propagator = np.asarray(propagator, dtype=np.complex128)
    channels = int(scale.size)
    latent_rank = int(latent_scale.size)
    modes = int(propagator.shape[0])
    gram = np.zeros((modes, channels, channels), dtype=np.complex128)
    cross = np.zeros((modes, channels, latent_rank), dtype=np.complex128)
    cached_rows = []
    for case in cases:
        if str(case["split"]) != "train":
            continue
        values = np.asarray(
            np.load(
                projected_cache / "cases" / f"{case['case_id']}.npy",
                mmap_mode="r",
            )[:, :channels],
            dtype=np.float64,
        )
        normalized = values / scale[None, :, None]
        if linear_baseline == "projected_hermite":
            if basis is None or k_arr is None:
                raise ValueError(
                    "projected_hermite readout fitting requires basis and k_arr"
                )
            baseline_latent = _projected_hermite_baseline_latent_rows(
                values,
                basis,
                k_arr,
                fine_steps=fine_steps,
                fine_dt=fine_dt,
                hermite_tail_damping=hermite_tail_damping,
                hermite_tail_power=hermite_tail_power,
                poisson_sign=poisson_sign,
            )
            target = (
                values[1:, 3:] - baseline_latent
            ) / latent_scale[None, :, None]
        else:
            coefficients = np.fft.rfft(normalized, axis=-1, norm="forward")
            linear_hat = np.einsum(
                "tck,kcd->tdk", coefficients[:-1], propagator, optimize=True
            )
            target = np.fft.irfft(
                coefficients[1:, 3:] - linear_hat[:, 3:],
                n=values.shape[-1],
                axis=-1,
                norm="forward",
            )
        latent_amplitude = np.sqrt(
            np.mean(np.square(normalized[:-1, 3:]), axis=(1, 2))
        )
        gate = np.power(latent_amplitude, int(gate_power)) / (
            np.power(latent_amplitude, int(gate_power))
            + float(gate_scale) ** int(gate_power)
        )
        features_hat = np.fft.rfft(
            gate[:, None, None] * normalized[:-1],
            axis=-1,
            norm="forward",
        )
        target_hat = np.fft.rfft(target, axis=-1, norm="forward")
        target_energy = float(np.mean(np.square(target), dtype=np.float64))
        trajectory_weight = max(
            target_energy, np.finfo(np.float64).tiny
        ) ** (-float(trajectory_exponent))
        gram += trajectory_weight * np.einsum(
            "tck,tdk->kcd",
            features_hat.conj(),
            features_hat,
            optimize=True,
        )
        cross += trajectory_weight * np.einsum(
            "tck,trk->kcr",
            features_hat.conj(),
            target_hat,
            optimize=True,
        )
        cached_rows.append((features_hat, target_hat, trajectory_weight))
    if not cached_rows:
        raise ValueError("No train-only trajectories were available for latent fit")

    weights = np.zeros_like(cross)
    identity = np.eye(channels, dtype=np.complex128)
    for mode in range(modes):
        regularization = (
            float(readout_ridge)
            * float(np.trace(gram[mode]).real)
            / float(channels)
        )
        regularization = max(regularization, np.finfo(np.float64).tiny)
        weights[mode] = np.linalg.solve(
            gram[mode] + regularization * identity,
            cross[mode],
        )
    weights[0] = weights[0].real
    if (modes - 1) * 2 == int(values.shape[-1]):
        weights[-1] = weights[-1].real

    error = 0.0
    total = 0.0
    for features_hat, target_hat, trajectory_weight in cached_rows:
        predicted_hat = np.einsum(
            "tck,kcr->trk", features_hat, weights, optimize=True
        )
        error += trajectory_weight * float(
            np.sum(np.square(np.abs(predicted_hat - target_hat)), dtype=np.float64)
        )
        total += trajectory_weight * float(
            np.sum(np.square(np.abs(target_hat)), dtype=np.float64)
        )
    diagnostics = {
        "train_relative_mse": error / total,
        "weight_rms": float(np.sqrt(np.mean(np.square(np.abs(weights))))),
        "weight_maximum": float(np.max(np.abs(weights))),
    }
    weights = weights.astype(np.complex64)
    np.savez_compressed(path, weights=weights, **diagnostics)
    print(f"[data] saved train-only gated latent readout {path}", flush=True)
    return weights, diagnostics


def _fit_multiplicative_endpoint_readout(
    params,
    projected_cache: Path,
    cases: list[dict],
    propagator: np.ndarray,
    basis: np.ndarray,
    k_arr: np.ndarray,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    correction_bounds: np.ndarray,
    depth: int,
    fine_steps: int,
    fine_dt: float,
    kick_scale: float,
    poisson_sign: float,
    linear_baseline: str,
    hermite_tail_damping: float,
    hermite_tail_power: float,
    latent_delay_input: bool,
    ridge: float,
):
    """Fit the existing nonlinear feature decoder to endpoint residuals."""
    modes, operator_rank, latent_rank = np.asarray(
        params["operator_u_real"]
    ).shape
    gram = np.zeros((modes, operator_rank, operator_rank), dtype=np.complex128)
    cross = np.zeros((modes, operator_rank, latent_rank), dtype=np.complex128)
    target_energy = 0.0
    regime_terms = {
        regime: {
            "gram": np.zeros_like(gram),
            "cross": np.zeros_like(cross),
            "target_energy": 0.0,
        }
        for regime in REGIMES
    }
    if latent_delay_input:
        feature_function = jax.jit(
            lambda resolved, latent, latent_delta: (
                state_conditioned_latent_operator_features(
                    params,
                    resolved,
                    latent,
                    depth=depth,
                    equilibrium_preserving=True,
                    normalized_latent_delta=latent_delta,
                )
            )
        )
    else:
        feature_function = jax.jit(
            lambda resolved, latent: state_conditioned_latent_operator_features(
                params,
                resolved,
                latent,
                depth=depth,
                equilibrium_preserving=True,
            )
        )
    train_cases = [case for case in cases if str(case["split"]) == "train"]
    if linear_baseline == "projected_hermite":
        residuals = _projected_hermite_normalized_latent_residual_blocks(
            projected_cache,
            train_cases,
            basis,
            k_arr,
            latent_scale=latent_scale,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            hermite_tail_damping=hermite_tail_damping,
            hermite_tail_power=hermite_tail_power,
            poisson_sign=poisson_sign,
        )
    else:
        residuals = _semilinear_normalized_latent_residual_blocks(
            projected_cache,
            train_cases,
            propagator,
            basis,
            k_arr,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            kick_scale=kick_scale,
            poisson_sign=poisson_sign,
            midpoint=False,
        )
    bounds = np.asarray(correction_bounds, dtype=np.float64)[None, :, None]
    for case, endpoint_target in zip(train_cases, residuals):
        values = np.asarray(
            np.load(
                projected_cache / "cases" / f"{case['case_id']}.npy",
                mmap_mode="r",
            ),
            dtype=np.float32,
        )
        normalized_resolved = jnp.asarray(
            values[:-1, :3] / resolved_scale[None, :, None]
        )
        normalized_latent = jnp.asarray(
            values[:-1, 3:] / latent_scale[None, :, None]
        )
        if latent_delay_input:
            previous_latent = np.concatenate(
                (values[:1, 3:], values[:-2, 3:]), axis=0
            )
            normalized_latent_delta = jnp.asarray(
                (values[:-1, 3:] - previous_latent)
                / latent_scale[None, :, None]
            )
            features = np.asarray(
                feature_function(
                    normalized_resolved,
                    normalized_latent,
                    normalized_latent_delta,
                )
            )
        else:
            features = np.asarray(
                feature_function(normalized_resolved, normalized_latent)
            )
        endpoint_target = np.asarray(endpoint_target, dtype=np.float64)
        raw_target = bounds * np.arctanh(
            np.clip(endpoint_target / bounds, -0.999, 0.999)
        )
        target_hat = np.fft.rfft(raw_target, axis=-1, norm="forward")[
            :, :, :modes
        ]
        case_gram = np.einsum(
            "tjk,tlk->kjl", features.conj(), features, optimize=True
        )
        case_cross = np.einsum(
            "tjk,trk->kjr", features.conj(), target_hat, optimize=True
        )
        case_energy = float(np.sum(np.square(np.abs(target_hat))))
        gram += case_gram
        cross += case_cross
        target_energy += case_energy
        terms = regime_terms[str(case["regime"])]
        terms["gram"] += case_gram
        terms["cross"] += case_cross
        terms["target_energy"] += case_energy

    weights = np.zeros_like(cross)
    identity = np.eye(operator_rank, dtype=np.complex128)
    for mode in range(modes):
        regularization = (
            float(ridge)
            * float(np.trace(gram[mode]).real)
            / float(operator_rank)
        )
        regularization = max(regularization, np.finfo(np.float64).tiny)
        weights[mode] = np.linalg.solve(
            gram[mode] + regularization * identity,
            cross[mode],
        )
    weights[0] = weights[0].real
    if (modes - 1) * 2 == int(values.shape[-1]):
        weights[-1] = weights[-1].real

    def relative_mse(terms):
        error = float(terms["target_energy"])
        for mode in range(modes):
            weight = weights[mode]
            error += float(
                np.trace(weight.conj().T @ terms["gram"][mode] @ weight).real
                - 2.0
                * np.trace(weight.conj().T @ terms["cross"][mode]).real
            )
        return max(error, 0.0) / max(
            float(terms["target_energy"]), np.finfo(np.float64).tiny
        )

    diagnostics = {
        "train_relative_mse": relative_mse(
            {"gram": gram, "cross": cross, "target_energy": target_energy}
        ),
        "train_regime_relative_mse": {
            regime: relative_mse(terms) for regime, terms in regime_terms.items()
        },
        "weight_rms": float(np.sqrt(np.mean(np.square(np.abs(weights))))),
        "weight_maximum": float(np.max(np.abs(weights))),
    }
    updated = dict(params)
    updated["operator_u_real"] = jnp.asarray(
        weights.real, dtype=params["operator_u_real"].dtype
    )
    updated["operator_u_imag"] = jnp.asarray(
        weights.imag, dtype=params["operator_u_imag"].dtype
    )
    return updated, diagnostics


def _electric_spectral_amplitude_terms(predicted_field, target_field):
    """Return mode-amplitude error terms while leaving phase to the field loss."""
    predicted_hat = jnp.fft.rfft(predicted_field, axis=-1, norm="forward")[..., 1:]
    target_hat = jnp.fft.rfft(target_field, axis=-1, norm="forward")[..., 1:]
    target_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.abs(target_hat)), axis=-1))
    floor = jnp.maximum(
        1e-2 * target_rms,
        jnp.asarray(jnp.finfo(target_hat.real.dtype).tiny, target_hat.real.dtype),
    )[..., None]
    predicted_amplitude = jnp.sqrt(
        jnp.square(jnp.abs(predicted_hat)) + jnp.square(floor)
    )
    target_amplitude = jnp.sqrt(
        jnp.square(jnp.abs(target_hat)) + jnp.square(floor)
    )
    return (
        jnp.sum(jnp.square(predicted_amplitude - target_amplitude), axis=-1),
        jnp.sum(jnp.square(target_amplitude), axis=-1),
    )


def _electric_chunk_log_growth_error(
    initial_energy,
    final_energy,
    target_initial_energy,
    target_final_energy,
    energy_floor,
):
    """Penalize the wrong electric-energy trend over a complete gradient chunk."""
    predicted_growth = jnp.log(final_energy + energy_floor) - jnp.log(
        initial_energy + energy_floor
    )
    target_growth = jnp.log(target_final_energy + energy_floor) - jnp.log(
        target_initial_energy + energy_floor
    )
    return jnp.square(predicted_growth - target_growth)


def _equal_contiguous_episode_weights(active):
    """Give every contiguous true episode equal total mass per case."""
    active = jnp.asarray(active, dtype=bool)
    starts = active & jnp.concatenate(
        (jnp.ones_like(active[:, :1]), ~active[:, :-1]), axis=1
    )
    episode_index = jnp.cumsum(starts.astype(jnp.int32), axis=1) - 1
    memberships = jax.nn.one_hot(
        episode_index, active.shape[1], dtype=jnp.float32
    ) * active[:, :, None]
    lengths = jnp.sum(memberships, axis=1)
    exists = lengths > 0.0
    episode_count = jnp.maximum(jnp.sum(exists, axis=1, keepdims=True), 1.0)
    episode_mass = exists.astype(jnp.float32) / episode_count
    return jnp.sum(
        memberships
        * episode_mass[:, None, :]
        / jnp.maximum(lengths[:, None, :], 1.0),
        axis=2,
    ).astype(jnp.float32)


def _turnaround_indices(target_energy, *, minimum_start_index: int = 50):
    """Select the local damping minimum with the strongest later regrowth."""
    target_energy = jnp.asarray(target_energy)
    times = jnp.arange(target_energy.shape[1])[None, :]
    local_minimum = jnp.concatenate(
        (
            jnp.zeros_like(target_energy[:, :1], dtype=bool),
            (target_energy[:, 1:-1] <= target_energy[:, :-2])
            & (target_energy[:, 1:-1] < target_energy[:, 2:]),
            jnp.zeros_like(target_energy[:, :1], dtype=bool),
        ),
        axis=1,
    ) & (times >= int(minimum_start_index))
    future_maximum = jax.lax.associative_scan(
        jnp.maximum, target_energy[:, ::-1], axis=1
    )[:, ::-1]
    score = jnp.where(
        local_minimum,
        jnp.log(jnp.maximum(future_maximum, jnp.finfo(target_energy.dtype).tiny))
        - jnp.log(jnp.maximum(target_energy, jnp.finfo(target_energy.dtype).tiny)),
        -jnp.inf,
    )
    minimum_index = jnp.argmax(score, axis=1)
    peak_index = jnp.argmax(
        jnp.where(times >= minimum_index[:, None], target_energy, -jnp.inf), axis=1
    )
    return minimum_index, peak_index


def _physical_trajectory_terms(
    predicted_state,
    target_resolved,
    k_arr,
    *,
    poisson_sign: float,
    electric_energy_floor=None,
    electric_time_relative_floor=None,
):
    """Return per-sample, per-channel physical trajectory loss terms."""
    target_state = resolved_hermite_to_low_moment_state(target_resolved)
    predicted_fields = primitive_fields(
        predicted_state,
        k_arr,
        poisson_sign=poisson_sign,
    )
    target_fields = primitive_fields(
        target_state,
        k_arr,
        poisson_sign=poisson_sign,
    )
    spectral_numerator, spectral_denominator = _electric_spectral_amplitude_terms(
        predicted_fields[:, 3], target_fields[:, 3]
    )
    predicted_energy = jnp.mean(jnp.square(predicted_fields[:, 3]), axis=-1)
    target_energy = jnp.mean(jnp.square(target_fields[:, 3]), axis=-1)
    if electric_energy_floor is None:
        electric_energy_floor = jnp.maximum(
            1e-8 * target_energy,
            jnp.asarray(jnp.finfo(target_energy.dtype).tiny, target_energy.dtype),
        )
    electric_energy_floor = jnp.asarray(
        electric_energy_floor, dtype=target_energy.dtype
    )
    if electric_time_relative_floor is None:
        electric_time_relative_floor = electric_energy_floor
    electric_time_relative_floor = jnp.asarray(
        electric_time_relative_floor, dtype=target_energy.dtype
    )
    log_energy_error = jnp.square(
        jnp.log(predicted_energy + electric_energy_floor)
        - jnp.log(target_energy + electric_energy_floor)
    )
    electric_time_relative_error = jnp.mean(
        jnp.square(predicted_fields[:, 3] - target_fields[:, 3]), axis=-1
    ) / (target_energy + electric_time_relative_floor)
    return (
        jnp.sum(jnp.square(predicted_fields - target_fields), axis=-1),
        jnp.sum(jnp.square(target_fields), axis=-1),
        spectral_numerator,
        spectral_denominator,
        log_energy_error,
        electric_time_relative_error,
    )


def _physical_sample_loss(numerator, denominator):
    valid = jnp.all(denominator > 0.0, axis=-1)
    return jnp.where(
        valid,
        jnp.mean(numerator / denominator, axis=-1),
        jnp.nan,
    )


def _sgd_step(params, grads, state, learning_rate: float, grad_clip: float):
    """Take one globally clipped SGD step while preserving checkpoint shape."""
    grad_norm = _tree_l2_norm(grads)
    tiny = jnp.finfo(grad_norm.dtype).tiny
    scale = jnp.minimum(
        1.0,
        jnp.asarray(grad_clip, dtype=grad_norm.dtype)
        / jnp.maximum(grad_norm, tiny),
    )
    updated = jax.tree_util.tree_map(
        lambda parameter, gradient: parameter
        - jnp.asarray(learning_rate, dtype=parameter.dtype)
        * scale.astype(parameter.dtype)
        * gradient,
        params,
        grads,
    )
    return updated, state, grad_norm


def _aggregate_gradients(gradients, *, normalize: bool, method: str = "mean"):
    """Aggregate sampled gradients with optional per-batch normalization."""
    if not gradients:
        raise ValueError("gradients must be nonempty")
    if normalize:
        normalized = []
        for gradient in gradients:
            norm = _tree_l2_norm(gradient)
            tiny = jnp.finfo(norm.dtype).tiny
            normalized.append(
                jax.tree_util.tree_map(
                    lambda value: value / jnp.maximum(norm, tiny),
                    gradient,
                )
            )
        gradients = normalized
    if method == "mean":
        count = float(len(gradients))
        return jax.tree_util.tree_map(
            lambda *values: sum(values) / count,
            *gradients,
        )
    if method == "coordinate_median":
        return jax.tree_util.tree_map(
            lambda *values: jnp.median(jnp.stack(values, axis=0), axis=0),
            *gradients,
        )
    raise ValueError(f"Unsupported gradient aggregation method: {method}")


def _mean_gradients(gradients, *, normalize: bool):
    """Backward-compatible mean-gradient helper used by focused tests."""
    return _aggregate_gradients(gradients, normalize=normalize, method="mean")


def _gradient_group_norms(grads) -> tuple[jax.Array, jax.Array, jax.Array]:
    output = {
        key: value for key, value in grads.items() if key.startswith("operator_u_")
    }
    projection = {
        key: value for key, value in grads.items() if key.startswith("operator_v_")
    }
    conditioner = {
        key: value
        for key, value in grads.items()
        if not key.startswith("operator_u_") and not key.startswith("operator_v_")
    }
    return (
        _tree_l2_norm(output),
        _tree_l2_norm(projection),
        _tree_l2_norm(conditioner),
    )


def _clip_parameter_update(params, updated, maximum_norm: float):
    """Clip the actual Adam parameter displacement, not only its raw gradient."""
    delta = jax.tree_util.tree_map(lambda new, old: new - old, updated, params)
    norm = _tree_l2_norm(delta)
    maximum = jnp.asarray(maximum_norm, dtype=norm.dtype)
    factor = jnp.minimum(1.0, maximum / jnp.maximum(norm, jnp.finfo(norm.dtype).tiny))
    clipped = jax.tree_util.tree_map(
        lambda old, change: old + factor * change, params, delta
    )
    return clipped, norm, norm * factor


def _tree_inner_product(left, right):
    products = [
        jnp.real(jnp.vdot(left_value, right_value))
        for left_value, right_value in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
        )
    ]
    return sum(products, start=jnp.asarray(0.0, dtype=products[0].dtype))


def _mask_tail_output_gradients(
    gradients,
    start: int,
    rows=(),
    *,
    expert_output_only: bool = False,
    expert_phase_output_only: bool = False,
    expert_gate_only: bool = False,
    expert_cyclic_gate_only: bool = False,
    expert_cyclic_output_only: bool = False,
    cyclic_low_mode_closure_only: bool = False,
    cyclic_low_mode_pressure_only: bool = False,
    cyclic_modal_pressure_only: bool = False,
    cyclic_low_mode_momentum_only: bool = False,
    cyclic_feature_indices=(),
    expert_indices=(),
):
    """Freeze the learned core and update selected output channels or a tail."""
    selected = tuple(int(row) for row in rows)
    if (
        expert_output_only
        or expert_phase_output_only
        or expert_gate_only
        or expert_cyclic_gate_only
        or expert_cyclic_output_only
        or cyclic_low_mode_closure_only
        or cyclic_low_mode_pressure_only
        or cyclic_modal_pressure_only
        or cyclic_low_mode_momentum_only
    ):
        selected_experts = tuple(int(index) for index in expert_indices)
        if cyclic_low_mode_momentum_only:
            trainable_expert_keys = {"expert_cyclic_low_mode_momentum_gain"}
        elif cyclic_modal_pressure_only:
            trainable_expert_keys = {"expert_cyclic_modal_pressure_gain"}
        elif cyclic_low_mode_pressure_only:
            trainable_expert_keys = {
                "expert_cyclic_low_mode_pressure_gain",
                "expert_cyclic_modal_pressure_gain",
            }
        elif cyclic_low_mode_closure_only:
            trainable_expert_keys = {"expert_cyclic_low_mode_closure_gain"}
        elif expert_cyclic_output_only:
            cyclic_output_prefix = (
                "specialist_cyclic_output"
                if "specialist_cyclic_output_kernel" in gradients
                else "expert_cyclic_output"
            )
            trainable_expert_keys = {
                f"{cyclic_output_prefix}_kernel",
                f"{cyclic_output_prefix}_bias",
            }
        elif expert_cyclic_gate_only:
            # The causal cyclic router is affine, b_e + w_e^T phi.  Keep the
            # generic state-dependent gate frozen, but allow its intercepts so
            # training can suppress a mismatched specialist at neutral phase.
            trainable_expert_keys = {
                "expert_gate_bias",
                "expert_null_gate_bias",
                "expert_bounce_gate_gain",
            }
        elif expert_gate_only:
            trainable_expert_keys = {
                "expert_gate_kernel",
                "expert_gate_bias",
                "expert_null_gate_bias",
                "expert_bounce_gate_gain",
            }
        elif expert_phase_output_only:
            trainable_expert_keys = {
                "expert_phase_output_kernel",
                "expert_phase_output_bias",
            }
        else:
            trainable_expert_keys = {
                "expert_output_kernel",
                "expert_output_bias",
                "expert_phase_output_kernel",
                "expert_phase_output_bias",
                "expert_phase_gain",
                "expert_gate_kernel",
                "expert_gate_bias",
                "expert_null_gate_bias",
                "expert_bounce_gate_gain",
            }
        masked = {
            key: value if key in trainable_expert_keys else jnp.zeros_like(value)
            for key, value in gradients.items()
        }
        if (cyclic_low_mode_closure_only or cyclic_low_mode_pressure_only or cyclic_modal_pressure_only or cyclic_low_mode_momentum_only) and cyclic_feature_indices:
            feature_indices = jnp.asarray(
                tuple(int(index) for index in cyclic_feature_indices)
            )
            gain_keys = (
                ("expert_cyclic_low_mode_momentum_gain",)
                if cyclic_low_mode_momentum_only
                else (
                    "expert_cyclic_low_mode_pressure_gain",
                    "expert_cyclic_modal_pressure_gain",
                )
                if cyclic_low_mode_pressure_only
                else ("expert_cyclic_modal_pressure_gain",)
                if cyclic_modal_pressure_only
                else ("expert_cyclic_low_mode_closure_gain",)
            )
            for gain_key in gain_keys:
                if gain_key not in masked:
                    continue
                selected_value = jnp.zeros_like(masked[gain_key])
                masked[gain_key] = selected_value.at[feature_indices].set(
                    masked[gain_key][feature_indices]
                )
        if selected_experts:
            indices = jnp.asarray(selected_experts)
            selected_keys = (
                (
                    "expert_gate_kernel",
                    "expert_gate_bias",
                    "expert_bounce_gate_gain",
                )
                if expert_gate_only
                else (
                    (
                        f"{cyclic_output_prefix}_kernel",
                        f"{cyclic_output_prefix}_bias",
                    )
                    if expert_cyclic_output_only
                    else (
                    (
                        "expert_gate_bias",
                        "expert_bounce_gate_gain",
                    )
                    if expert_cyclic_gate_only
                    else (
                    ("expert_phase_output_kernel", "expert_phase_output_bias")
                    if expert_phase_output_only
                    else (
                    "expert_output_kernel",
                    "expert_output_bias",
                    "expert_phase_output_kernel",
                    "expert_phase_output_bias",
                    "expert_phase_gain",
                    "expert_bounce_gate_gain",
                    )
                    )
                    )
                )
            )
            for key in selected_keys:
                if key not in masked:
                    continue
                selected_value = jnp.zeros_like(masked[key])
                masked[key] = selected_value.at[indices].set(masked[key][indices])
            if expert_cyclic_gate_only and cyclic_feature_indices:
                feature_indices = jnp.asarray(
                    tuple(int(index) for index in cyclic_feature_indices)
                )
                key = "expert_bounce_gate_gain"
                selected_value = jnp.zeros_like(masked[key])
                selected_value = selected_value.at[
                    indices[:, None], feature_indices[None, :]
                ].set(masked[key][indices[:, None], feature_indices[None, :]])
                masked[key] = selected_value
            if expert_cyclic_output_only and cyclic_feature_indices:
                feature_indices = jnp.asarray(
                    tuple(int(index) for index in cyclic_feature_indices)
                )
                for key in (
                    f"{cyclic_output_prefix}_kernel",
                    f"{cyclic_output_prefix}_bias",
                ):
                    if key not in masked:
                        continue
                    selected_value = jnp.zeros_like(masked[key])
                    selected_value = selected_value.at[
                        indices[:, None], feature_indices[None, :]
                    ].set(masked[key][indices[:, None], feature_indices[None, :]])
                    masked[key] = selected_value
            if expert_phase_output_only and selected:
                output_indices = jnp.asarray(selected)
                for key in (
                    "expert_phase_output_kernel",
                    "expert_phase_output_bias",
                ):
                    if key not in masked:
                        continue
                    selected_value = jnp.zeros_like(masked[key])
                    selected_value = selected_value.at[
                        indices[:, None], output_indices[None, :]
                    ].set(masked[key][indices[:, None], output_indices[None, :]])
                    masked[key] = selected_value
            for key in ("expert_gate_kernel", "expert_gate_bias"):
                if key in masked:
                    if expert_gate_only or expert_cyclic_gate_only:
                        selected_value = jnp.zeros_like(masked[key])
                        masked[key] = selected_value.at[indices].set(
                            masked[key][indices]
                        )
                    else:
                        masked[key] = jnp.zeros_like(masked[key])
        return masked
    if int(start) <= 0 and not selected:
        return gradients
    masked = {key: jnp.zeros_like(value) for key, value in gradients.items()}
    for key in ("output_kernel", "output_bias"):
        if key in gradients:
            if selected:
                indices = jnp.asarray(selected)
                masked[key] = masked[key].at[indices].set(gradients[key][indices])
            else:
                masked[key] = masked[key].at[int(start) :].set(
                    gradients[key][int(start) :]
                )
    return masked


def _combine_conflict_safe_gradients(
    physical_grads,
    auxiliary_grads,
    *,
    auxiliary_weight: float,
    maximum_auxiliary_ratio: float,
):
    """Add a weighted auxiliary direction without opposing physical descent."""
    physical_norm = _tree_l2_norm(physical_grads)
    auxiliary_norm = _tree_l2_norm(auxiliary_grads)
    weighted_auxiliary = jax.tree_util.tree_map(
        lambda value: float(auxiliary_weight) * value,
        auxiliary_grads,
    )
    physical_squared = jnp.square(physical_norm)
    weighted_auxiliary_norm = _tree_l2_norm(weighted_auxiliary)
    inner_product = _tree_inner_product(physical_grads, weighted_auxiliary)
    tiny = jnp.finfo(physical_norm.dtype).tiny
    cosine = inner_product / jnp.maximum(
        physical_norm * weighted_auxiliary_norm,
        tiny,
    )
    conflicting = inner_product < 0.0
    projection_coefficient = jnp.where(
        conflicting & (physical_squared > 0.0),
        inner_product / jnp.maximum(physical_squared, tiny),
        0.0,
    )
    projected_auxiliary = jax.tree_util.tree_map(
        lambda auxiliary, physical: auxiliary - projection_coefficient * physical,
        weighted_auxiliary,
        physical_grads,
    )
    projected_norm = _tree_l2_norm(projected_auxiliary)
    maximum_norm = (
        jnp.asarray(maximum_auxiliary_ratio, dtype=physical_norm.dtype)
        * physical_norm
    )
    contribution_scale = jnp.where(
        projected_norm > 0.0,
        jnp.minimum(1.0, maximum_norm / jnp.maximum(projected_norm, tiny)),
        0.0,
    )
    combined = jax.tree_util.tree_map(
        lambda physical, auxiliary: physical + contribution_scale * auxiliary,
        physical_grads,
        projected_auxiliary,
    )
    contribution_norm = contribution_scale * projected_norm
    physical_alignment = _tree_inner_product(physical_grads, combined) / jnp.maximum(
        physical_squared,
        tiny,
    )
    return (
        combined,
        physical_norm,
        auxiliary_norm,
        cosine,
        contribution_norm,
        conflicting,
        physical_alignment,
    )


def _combine_gradients(
    physical_grads,
    auxiliary_grads,
    *,
    auxiliary_weight: float,
    maximum_auxiliary_ratio: float,
    mode: str,
):
    """Combine physical and auxiliary gradients before a shared optimizer step."""
    if mode == "conflict_safe":
        return _combine_conflict_safe_gradients(
            physical_grads,
            auxiliary_grads,
            auxiliary_weight=auxiliary_weight,
            maximum_auxiliary_ratio=maximum_auxiliary_ratio,
        )
    if mode != "direct_sum":
        raise ValueError(f"Unknown update-combination mode: {mode}")

    physical_norm = _tree_l2_norm(physical_grads)
    auxiliary_norm = _tree_l2_norm(auxiliary_grads)
    weighted_auxiliary = jax.tree_util.tree_map(
        lambda value: float(auxiliary_weight) * value,
        auxiliary_grads,
    )
    weighted_auxiliary_norm = _tree_l2_norm(weighted_auxiliary)
    inner_product = _tree_inner_product(physical_grads, weighted_auxiliary)
    tiny = jnp.finfo(physical_norm.dtype).tiny
    cosine = inner_product / jnp.maximum(
        physical_norm * weighted_auxiliary_norm,
        tiny,
    )
    combined = jax.tree_util.tree_map(
        lambda physical, auxiliary: physical + auxiliary,
        physical_grads,
        weighted_auxiliary,
    )
    physical_alignment = _tree_inner_product(physical_grads, combined) / jnp.maximum(
        jnp.square(physical_norm),
        tiny,
    )
    return (
        combined,
        physical_norm,
        auxiliary_norm,
        cosine,
        weighted_auxiliary_norm,
        inner_product < 0.0,
        physical_alignment,
    )


def _combine_conflict_safe_updates(
    physical_update,
    teacher_update,
    *,
    maximum_teacher_ratio: float,
):
    """Combine independently conditioned Adam steps without opposing physics."""
    physical_norm = _tree_l2_norm(physical_update)
    teacher_norm = _tree_l2_norm(teacher_update)
    physical_squared = jnp.square(physical_norm)
    inner_product = _tree_inner_product(physical_update, teacher_update)
    tiny = jnp.finfo(physical_norm.dtype).tiny
    cosine = inner_product / jnp.maximum(physical_norm * teacher_norm, tiny)
    conflicting = inner_product < 0.0
    projection_coefficient = jnp.where(
        conflicting & (physical_squared > 0.0),
        inner_product / jnp.maximum(physical_squared, tiny),
        0.0,
    )
    projected_teacher = jax.tree_util.tree_map(
        lambda teacher, physical: teacher - projection_coefficient * physical,
        teacher_update,
        physical_update,
    )
    projected_norm = _tree_l2_norm(projected_teacher)
    maximum_norm = (
        jnp.asarray(maximum_teacher_ratio, dtype=physical_norm.dtype)
        * physical_norm
    )
    contribution_scale = jnp.where(
        projected_norm > 0.0,
        jnp.minimum(1.0, maximum_norm / jnp.maximum(projected_norm, tiny)),
        0.0,
    )
    combined = jax.tree_util.tree_map(
        lambda physical, teacher: physical + contribution_scale * teacher,
        physical_update,
        projected_teacher,
    )
    contribution_norm = contribution_scale * projected_norm
    physical_alignment = _tree_inner_product(
        physical_update, combined
    ) / jnp.maximum(physical_squared, tiny)
    return (
        combined,
        physical_norm,
        teacher_norm,
        cosine,
        contribution_norm,
        conflicting,
        physical_alignment,
    )


def _combine_groupwise_conflict_safe_updates(
    physical_update,
    teacher_update,
    *,
    output_ratio: float,
    internal_ratio: float,
):
    """Use a tight decoder trust region and a wider internal trust region."""
    output_keys = tuple(
        key for key in physical_update if key.startswith("operator_u_")
    )
    internal_keys = tuple(key for key in physical_update if key not in output_keys)

    def select(tree, keys):
        return {key: tree[key] for key in keys}

    output_result = _combine_conflict_safe_updates(
        select(physical_update, output_keys),
        select(teacher_update, output_keys),
        maximum_teacher_ratio=output_ratio,
    )
    internal_result = _combine_conflict_safe_updates(
        select(physical_update, internal_keys),
        select(teacher_update, internal_keys),
        maximum_teacher_ratio=internal_ratio,
    )
    combined = {**output_result[0], **internal_result[0]}
    contribution = jax.tree_util.tree_map(
        lambda total, physical: total - physical,
        combined,
        physical_update,
    )
    physical_norm = _tree_l2_norm(physical_update)
    teacher_norm = _tree_l2_norm(teacher_update)
    physical_squared = jnp.square(physical_norm)
    tiny = jnp.finfo(physical_norm.dtype).tiny
    cosine = _tree_inner_product(
        physical_update, teacher_update
    ) / jnp.maximum(physical_norm * teacher_norm, tiny)
    physical_alignment = _tree_inner_product(
        physical_update, combined
    ) / jnp.maximum(physical_squared, tiny)
    return (
        combined,
        physical_norm,
        teacher_norm,
        cosine,
        _tree_l2_norm(contribution),
        output_result[5] | internal_result[5],
        physical_alignment,
    )


def _combine_optimizer_updates(
    physical_update,
    teacher_update,
    *,
    mode: str,
    output_ratio: float,
    internal_ratio: float,
):
    """Combine independently preconditioned updates for the selected probe."""
    if mode == "conflict_safe":
        return _combine_groupwise_conflict_safe_updates(
            physical_update,
            teacher_update,
            output_ratio=output_ratio,
            internal_ratio=internal_ratio,
        )
    if mode != "direct_sum":
        raise ValueError(f"Unknown update-combination mode: {mode}")
    combined = jax.tree_util.tree_map(
        lambda physical, teacher: physical + teacher,
        physical_update,
        teacher_update,
    )
    physical_norm = _tree_l2_norm(physical_update)
    teacher_norm = _tree_l2_norm(teacher_update)
    inner_product = _tree_inner_product(physical_update, teacher_update)
    tiny = jnp.finfo(physical_norm.dtype).tiny
    cosine = inner_product / jnp.maximum(physical_norm * teacher_norm, tiny)
    physical_alignment = _tree_inner_product(
        physical_update, combined
    ) / jnp.maximum(jnp.square(physical_norm), tiny)
    return (
        combined,
        physical_norm,
        teacher_norm,
        cosine,
        teacher_norm,
        inner_product < 0.0,
        physical_alignment,
    )


def _scale_auxiliary_update(candidate, params, weight: float):
    """Convert an optimizer candidate into its weighted auxiliary displacement."""
    return jax.tree_util.tree_map(
        lambda new, old: float(weight) * (new - old), candidate, params
    )


def _make_functions(
    *,
    propagator,
    basis,
    k_arr,
    resolved_center,
    resolved_scale,
    latent_center,
    latent_scale,
    depth: int,
    fine_steps: int,
    fine_dt: float,
    horizon: int,
    poisson_sign: float,
    gradient_chunk_steps: int,
    latent_input_scale=None,
    correction_bounds=None,
    latent_residual_scale=None,
    latent_residual_weight: float = 0.0,
    latent_state_residual_weight: float = 1.0,
    low_mode_latent_residual_weight: float = 0.0,
    closure_residual_scale: float = 1.0,
    closure_residual_weight: float = 0.0,
    closure_correction_bound: float = 40.0,
    closure_aligned_output: bool = False,
    closure_only_correction: bool = False,
    closure_readout_mode: str = "multiplicative",
    closure_gate_scale: float = 0.1,
    closure_gate_power: int = 2,
    latent_readout_mode: str = "multiplicative",
    latent_gate_scale: float = 0.1,
    latent_gate_power: int = 2,
    latent_state_bound: float = 0.0,
    equilibrium_input_compression_scale: float = 0.0,
    autonomous_latent_weight: float = 0.0,
    electric_spectrum_weight: float = 0.0,
    electric_log_energy_weight: float = 0.0,
    electric_log_energy_floor_ratio: float = 1e-8,
    electric_chunk_log_growth_weight: float = 0.0,
    electric_sliding_log_growth_weight: float = 0.0,
    electric_sliding_log_growth_direction: str = "balanced",
    electric_growth_window_steps: int = 100,
    electric_time_relative_weight: float = 0.0,
    electric_transfer_weight: float = 0.0,
    electric_time_relative_floor_ratio: float = 1e-4,
    post_minimum_log_energy_weight: float = 0.0,
    post_minimum_field_weight: float = 0.0,
    turnaround_field_weight: float = 0.0,
    post_peak_log_energy_weight: float = 0.0,
    peak_log_energy_weight: float = 0.0,
    turnaround_log_ratio_weight: float = 0.0,
    post_peak_retention_weight: float = 0.0,
    post_peak_retention_floor: float = 0.0,
    post_minimum_energy_modes: int = 4,
    teacher_residual_stride: int = 1,
    teacher_rollout_steps: int = 1,
    teacher_residual_causal_history: bool = False,
    linear_baseline: str = "fitted_propagator",
    hermite_tail_damping: float = 10.0,
    hermite_tail_power: float = 6.0,
    semilinear_kick_scale: float = 1.0,
    semilinear_correction_location: str = "midpoint",
    latent_delay_input: bool = False,
    return_teacher_residual_function: bool = False,
    case_loss_weights=None,
):
    propagator = jnp.asarray(propagator)
    basis = jnp.asarray(basis)
    k_arr = jnp.asarray(k_arr)
    resolved_center = jnp.asarray(resolved_center)
    resolved_scale = jnp.asarray(resolved_scale)
    latent_center = jnp.asarray(latent_center)
    latent_scale = jnp.asarray(latent_scale)
    if case_loss_weights is not None:
        case_loss_weights = jnp.asarray(case_loss_weights)
        if case_loss_weights.shape != (3,):
            raise ValueError("case_loss_weights must contain linear, weak, strong")
        if bool(jnp.any(case_loss_weights < 0.0)) or not bool(
            jnp.any(case_loss_weights > 0.0)
        ):
            raise ValueError("case_loss_weights must be nonnegative and nonzero")

    def weighted_case_mean(values):
        if case_loss_weights is None:
            return jnp.mean(values)
        return jnp.sum(values * case_loss_weights) / jnp.sum(case_loss_weights)
    if correction_bounds is None:
        correction_bounds = jnp.ones_like(latent_scale)
    correction_bounds_array = np.asarray(correction_bounds)
    if (
        correction_bounds_array.ndim != 1
        or correction_bounds_array.shape != np.asarray(latent_scale).shape
        or not np.all(np.isfinite(correction_bounds_array))
        or np.any(correction_bounds_array <= 0.0)
    ):
        raise ValueError(
            "correction_bounds must match latent_scale and be finite and positive"
        )
    correction_bounds = jnp.asarray(correction_bounds, dtype=latent_scale.dtype)
    if latent_residual_scale is None:
        latent_residual_scale = jnp.ones_like(latent_scale)
    latent_residual_scale_array = np.asarray(latent_residual_scale)
    if (
        latent_residual_scale_array.shape != np.asarray(latent_scale).shape
        or np.any(~np.isfinite(latent_residual_scale_array))
        or np.any(latent_residual_scale_array <= 0.0)
    ):
        raise ValueError(
            "latent_residual_scale must match latent_scale and be finite and positive"
        )
    if float(latent_residual_weight) < 0.0:
        raise ValueError("latent_residual_weight must be nonnegative")
    if float(latent_state_residual_weight) < 0.0:
        raise ValueError("latent_state_residual_weight must be nonnegative")
    if not np.isfinite(float(closure_residual_scale)) or float(
        closure_residual_scale
    ) <= 0.0:
        raise ValueError("closure_residual_scale must be finite and positive")
    if float(closure_residual_weight) < 0.0:
        raise ValueError("closure_residual_weight must be nonnegative")
    if not np.isfinite(float(closure_correction_bound)) or float(
        closure_correction_bound
    ) <= 0.0:
        raise ValueError("closure_correction_bound must be finite and positive")
    if closure_only_correction and not closure_aligned_output:
        raise ValueError(
            "closure_only_correction requires closure_aligned_output"
        )
    if float(autonomous_latent_weight) < 0.0:
        raise ValueError("autonomous_latent_weight must be nonnegative")
    if float(electric_spectrum_weight) < 0.0:
        raise ValueError("electric_spectrum_weight must be nonnegative")
    if float(electric_log_energy_weight) < 0.0:
        raise ValueError("electric_log_energy_weight must be nonnegative")
    if float(electric_chunk_log_growth_weight) < 0.0:
        raise ValueError("electric_chunk_log_growth_weight must be nonnegative")
    if float(electric_sliding_log_growth_weight) < 0.0:
        raise ValueError("electric_sliding_log_growth_weight must be nonnegative")
    if electric_sliding_log_growth_direction not in {
        "balanced",
        "balanced_episodes",
        "growth",
        "growth_episodes",
        "damping",
    }:
        raise ValueError(
            "electric_sliding_log_growth_direction must be balanced, "
            "balanced_episodes, growth, growth_episodes, or damping"
        )
    if int(electric_growth_window_steps) <= 0:
        raise ValueError("electric_growth_window_steps must be positive")
    if (
        float(electric_sliding_log_growth_weight) > 0.0
        and 2 * int(electric_growth_window_steps) > int(horizon) + 1
    ):
        raise ValueError(
            "twice electric_growth_window_steps cannot exceed horizon + 1 when "
            "the smoothed sliding log-growth loss is enabled"
        )
    if (
        float(electric_chunk_log_growth_weight) > 0.0
        and int(gradient_chunk_steps) % int(electric_growth_window_steps) != 0
    ):
        raise ValueError(
            "gradient_chunk_steps must be divisible by electric_growth_window_steps"
        )
    growth_window_steps = (
        int(electric_growth_window_steps)
        if float(electric_chunk_log_growth_weight) > 0.0
        else int(gradient_chunk_steps)
    )
    if not 0.0 < float(electric_log_energy_floor_ratio) < 1.0:
        raise ValueError("electric_log_energy_floor_ratio must lie in (0, 1)")
    if float(electric_time_relative_weight) < 0.0:
        raise ValueError("electric_time_relative_weight must be nonnegative")
    if float(electric_transfer_weight) < 0.0:
        raise ValueError("electric_transfer_weight must be nonnegative")
    if float(low_mode_latent_residual_weight) < 0.0:
        raise ValueError("low_mode_latent_residual_weight must be nonnegative")
    if not 0.0 < float(electric_time_relative_floor_ratio) < 1.0:
        raise ValueError("electric_time_relative_floor_ratio must lie in (0, 1)")
    if float(post_minimum_log_energy_weight) < 0.0:
        raise ValueError("post_minimum_log_energy_weight must be nonnegative")
    if float(post_peak_log_energy_weight) < 0.0:
        raise ValueError("post_peak_log_energy_weight must be nonnegative")
    if float(peak_log_energy_weight) < 0.0:
        raise ValueError("peak_log_energy_weight must be nonnegative")
    if float(turnaround_log_ratio_weight) < 0.0:
        raise ValueError("turnaround_log_ratio_weight must be nonnegative")
    if float(post_peak_retention_weight) < 0.0:
        raise ValueError("post_peak_retention_weight must be nonnegative")
    if not 0.0 <= float(post_peak_retention_floor) <= 1.0:
        raise ValueError("post_peak_retention_floor must lie in [0, 1]")
    if float(post_minimum_field_weight) < 0.0:
        raise ValueError("post_minimum_field_weight must be nonnegative")
    if float(turnaround_field_weight) < 0.0:
        raise ValueError("turnaround_field_weight must be nonnegative")
    if not 1 <= int(post_minimum_energy_modes) < int(k_arr.shape[0]):
        raise ValueError("post_minimum_energy_modes must select nonzero retained modes")
    if int(teacher_residual_stride) <= 0:
        raise ValueError("teacher_residual_stride must be positive")
    if int(teacher_residual_stride) > int(horizon):
        raise ValueError("teacher_residual_stride may not exceed the horizon")
    if int(teacher_rollout_steps) <= 0:
        raise ValueError("teacher_rollout_steps must be positive")
    if int(teacher_rollout_steps) > int(horizon):
        raise ValueError("teacher_rollout_steps may not exceed the horizon")
    if float(equilibrium_input_compression_scale) < 0.0:
        raise ValueError(
            "equilibrium_input_compression_scale must be nonnegative"
        )
    latent_residual_scale = jnp.asarray(
        latent_residual_scale, dtype=latent_scale.dtype
    )
    closure_residual_scale = jnp.asarray(
        closure_residual_scale, dtype=latent_scale.dtype
    )

    def macro_step(params, state, latent, previous_latent, bounce_exposure=None):
        return coupled_low_moment_latent_step(
            params,
            state,
            latent,
            propagator,
            basis,
            k_arr,
            previous_latent_state=(previous_latent if latent_delay_input else None),
            bounce_exposure=bounce_exposure,
            resolved_center=resolved_center,
            resolved_scale=resolved_scale,
            latent_center=latent_center,
            latent_scale=latent_scale,
            latent_input_scale=latent_input_scale,
            depth=depth,
            correction_bounds=correction_bounds,
            closure_residual_scale=closure_residual_scale,
            closure_correction_bound=closure_correction_bound,
            closure_aligned_output=closure_aligned_output,
            closure_only_correction=closure_only_correction,
            closure_readout_mode=closure_readout_mode,
            closure_gate_scale=closure_gate_scale,
            closure_gate_power=closure_gate_power,
            latent_readout_mode=latent_readout_mode,
            latent_gate_scale=latent_gate_scale,
            latent_gate_power=latent_gate_power,
            latent_state_bound=latent_state_bound,
            equilibrium_input_compression_scale=equilibrium_input_compression_scale,
            linear_baseline=linear_baseline,
            hermite_tail_damping=hermite_tail_damping,
            hermite_tail_power=hermite_tail_power,
            semilinear_kick_scale=semilinear_kick_scale,
            semilinear_correction_location=semilinear_correction_location,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            dynamics_model="multiplicative_operator",
            poisson_sign=poisson_sign,
        )

    def teacher_residual_loss_function(params, trajectory, *, return_errors=False,
                                       return_channel_errors=False):
        if return_channel_errors and linear_baseline not in {"projected_hermite", "semilinear_strang"}:
            raise ValueError("Channel attribution requires a physical macro step")
        rollout_steps = int(teacher_rollout_steps)
        starts = list(
            range(
                0,
                int(horizon) - rollout_steps + 1,
                int(teacher_residual_stride),
            )
        )
        final_start = int(horizon) - rollout_steps
        if starts[-1] != final_start:
            starts.append(final_start)
        batch_size = trajectory.shape[0]
        chunks = len(starts)
        current_resolved = jnp.stack(
            [trajectory[:, start : start + rollout_steps, :3] for start in starts],
            axis=1,
        ).reshape(batch_size * chunks, rollout_steps, 3, trajectory.shape[-1])
        target_latent = jnp.stack(
            [
                trajectory[:, start + 1 : start + rollout_steps + 1, 3:]
                for start in starts
            ],
            axis=1,
        ).reshape(
            batch_size * chunks,
            rollout_steps,
            trajectory.shape[2] - 3,
            trajectory.shape[-1],
        )
        initial_latent = jnp.stack(
            [trajectory[:, start, 3:] for start in starts], axis=1
        ).reshape(
            batch_size * chunks,
            trajectory.shape[2] - 3,
            trajectory.shape[-1],
        )
        initial_previous_latent = jnp.stack(
            [trajectory[:, max(start - 1, 0), 3:] for start in starts], axis=1
        ).reshape(
            batch_size * chunks,
            trajectory.shape[2] - 3,
            trajectory.shape[-1],
        )

        residual_history = None
        if teacher_residual_causal_history:
            if linear_baseline not in {"projected_hermite", "semilinear_strang"}:
                raise ValueError("Causal residual history requires a physical macro step")
            from model.train.teacher_prefix_history import teacher_prefix_history
            shape = trajectory.shape
            states = resolved_hermite_to_low_moment_state(
                trajectory[:, :, :3].reshape(-1, 3, shape[-1])
            ).reshape(shape[0], shape[1], 3, shape[-1])
            history = teacher_prefix_history(states, k_arr, float(fine_steps) * float(fine_dt))
            residual_history = jnp.stack(
                [history[:, start:start + rollout_steps] for start in starts], axis=1
            ).reshape(batch_size * chunks, rollout_steps, history.shape[-1])

        def teacher_latent_step(carry, inputs):
            latent, previous_latent = carry
            resolved, target, history = inputs
            if linear_baseline in {"projected_hermite", "semilinear_strang"}:
                _, updated = macro_step(
                    params,
                    resolved_hermite_to_low_moment_state(resolved),
                    latent,
                    previous_latent,
                    history,
                )
                normalized_error = (updated - target) / (
                    latent_scale[None, :, None]
                    * latent_residual_scale[None, :, None]
                )
                if return_channel_errors:
                    return (updated, latent), jnp.mean(jnp.square(normalized_error), axis=2)
                latent_error = jnp.mean(
                    jnp.square(normalized_error), axis=(1, 2)
                )
                low_mode_error_hat = jnp.fft.rfft(
                    normalized_error, axis=-1, norm="forward"
                )[..., 1 : int(post_minimum_energy_modes) + 1]
                low_mode_latent_error = jnp.mean(
                    jnp.square(jnp.abs(low_mode_error_hat)), axis=(1, 2)
                )
                closure_error = jnp.einsum(
                    "r,brx->bx", basis[0], updated - target
                ) / closure_residual_scale
                closure_error = jnp.mean(jnp.square(closure_error), axis=1)
                return (updated, latent), (
                    float(latent_state_residual_weight) * latent_error
                    + float(low_mode_latent_residual_weight)
                    * low_mode_latent_error
                    + float(closure_residual_weight) * closure_error
                )
            linear = apply_coupled_linear_propagator(
                propagator,
                resolved,
                latent,
                resolved_scale=resolved_scale,
                latent_scale=latent_scale,
            )
            if closure_only_correction:
                predicted_correction = jnp.zeros_like(linear[:, 3:])
            else:
                if latent_readout_mode == "multiplicative":
                    predicted_correction = state_conditioned_latent_operator_correction(
                        params,
                        resolved / resolved_scale[None, :, None],
                        latent / latent_scale[None, :, None],
                        depth=depth,
                        equilibrium_preserving=True,
                        normalized_latent_delta=(
                            (latent - previous_latent)
                            / latent_scale[None, :, None]
                            if latent_delay_input
                            else None
                        ),
                    )
                else:
                    predicted_correction = gated_spectral_latent_correction(
                        params,
                        resolved / resolved_scale[None, :, None],
                        latent / latent_scale[None, :, None],
                        gate_scale=latent_gate_scale,
                        gate_power=latent_gate_power,
                    )
                    if latent_readout_mode == "gated_linear_residual":
                        predicted_correction = (
                            predicted_correction
                            + state_conditioned_latent_operator_correction(
                                params,
                                resolved / resolved_scale[None, :, None],
                                latent / latent_scale[None, :, None],
                                depth=depth,
                                equilibrium_preserving=True,
                            )
                        )
                predicted_correction = bounded_normalized_latent_correction(
                    predicted_correction,
                    correction_bounds,
                ) * latent_scale[None, :, None]
            if closure_aligned_output:
                predicted_correction = closure_orthogonal_latent_correction(
                    predicted_correction,
                    basis,
                )
                if closure_readout_mode == "multiplicative":
                    raw_predicted_closure = state_conditioned_closure_correction(
                        params,
                        resolved / resolved_scale[None, :, None],
                        latent / latent_scale[None, :, None],
                        depth=depth,
                        equilibrium_preserving=True,
                        normalized_latent_delta=(
                            (latent - previous_latent)
                            / latent_scale[None, :, None]
                            if latent_delay_input
                            else None
                        ),
                    )
                else:
                    raw_predicted_closure = gated_spectral_closure_correction(
                        params,
                        resolved / resolved_scale[None, :, None],
                        latent / latent_scale[None, :, None],
                        gate_scale=closure_gate_scale,
                        gate_power=closure_gate_power,
                    )
                    if closure_readout_mode == "gated_linear_residual":
                        raw_predicted_closure = (
                            raw_predicted_closure
                            + state_conditioned_closure_correction(
                                params,
                                resolved / resolved_scale[None, :, None],
                                latent / latent_scale[None, :, None],
                                depth=depth,
                                equilibrium_preserving=True,
                            )
                        )
                predicted_closure = bounded_normalized_closure_correction(
                    raw_predicted_closure,
                    closure_correction_bound,
                ) * closure_residual_scale
                predicted_correction = (
                    predicted_correction
                    + closure_aligned_latent_correction(
                        predicted_closure,
                        basis,
                    )
                )
            updated = linear[:, 3:] + predicted_correction
            updated = updated.astype(latent.dtype)
            normalized_error = (updated - target) / (
                latent_scale[None, :, None]
                * latent_residual_scale[None, :, None]
            )
            latent_error = jnp.mean(jnp.square(normalized_error), axis=(1, 2))
            low_mode_error_hat = jnp.fft.rfft(
                normalized_error, axis=-1, norm="forward"
            )[..., 1 : int(post_minimum_energy_modes) + 1]
            low_mode_latent_error = jnp.mean(
                jnp.square(jnp.abs(low_mode_error_hat)), axis=(1, 2)
            )
            closure_error = jnp.einsum(
                "r,brx->bx", basis[0], updated - target
            ) / closure_residual_scale
            closure_error = jnp.mean(jnp.square(closure_error), axis=1)
            return (updated, latent), (
                float(latent_state_residual_weight) * latent_error
                + float(low_mode_latent_residual_weight) * low_mode_latent_error
                + float(closure_residual_weight) * closure_error
            )

        _, errors = jax.lax.scan(
            teacher_latent_step,
            (initial_latent, initial_previous_latent),
            (
                jnp.swapaxes(current_resolved, 0, 1),
                jnp.swapaxes(target_latent, 0, 1),
                (jnp.swapaxes(residual_history, 0, 1)
                 if residual_history is not None else None),
            ),
        )
        if return_channel_errors:
            return errors.reshape(rollout_steps, batch_size, chunks, -1).transpose(1, 2, 0, 3)
        if return_errors:
            return errors.reshape(rollout_steps, batch_size, chunks).transpose(1, 2, 0)
        return jnp.mean(errors)

    def loss_components_function(params, trajectory):
        initial_resolved = trajectory[:, 0, :3]
        initial_latent = trajectory[:, 0, 3:]
        initial_state = resolved_hermite_to_low_moment_state(initial_resolved)
        complete_targets = jnp.swapaxes(trajectory, 0, 1)
        targets = complete_targets[1:]
        target_shape = complete_targets.shape
        target_states = resolved_hermite_to_low_moment_state(
            complete_targets[:, :, :3].reshape(-1, 3, target_shape[-1])
        )
        target_fields = primitive_fields(
            target_states,
            k_arr,
            poisson_sign=poisson_sign,
        ).reshape(target_shape[0], target_shape[1], 4, target_shape[-1])
        complete_target_energy = jnp.mean(
            jnp.square(target_fields[:, :, 3]), axis=-1
        )
        target_low_mode_hat = jnp.fft.rfft(
            target_fields[:, :, 3], axis=-1, norm="forward"
        )[..., 1 : int(post_minimum_energy_modes) + 1]
        target_momentum = target_states[:, 1].reshape(
            target_shape[0], target_shape[1], target_shape[-1]
        )
        target_momentum_hat = jnp.fft.rfft(
            target_momentum, axis=-1, norm="forward"
        )[..., 1 : int(post_minimum_energy_modes) + 1]
        complete_target_transfer = 2.0 * jnp.real(
            target_low_mode_hat * jnp.conj(target_momentum_hat)
        )
        complete_target_low_mode_energy = jnp.sum(
            jnp.square(jnp.abs(target_low_mode_hat)), axis=-1
        ).T
        target_energy = complete_target_energy[1:]
        energy_floor = jnp.maximum(
            float(electric_log_energy_floor_ratio)
            * jnp.max(complete_target_energy, axis=0),
            jnp.asarray(jnp.finfo(target_energy.dtype).tiny, target_energy.dtype),
        )
        time_relative_floor = jnp.maximum(
            float(electric_time_relative_floor_ratio)
            * jnp.max(target_energy, axis=0),
            jnp.asarray(jnp.finfo(target_energy.dtype).tiny, target_energy.dtype),
        )
        target_window_initial_energy = complete_target_energy[
            :-growth_window_steps:growth_window_steps
        ].reshape(
            horizon // gradient_chunk_steps,
            gradient_chunk_steps // growth_window_steps,
            trajectory.shape[0],
        )
        target_window_final_energy = complete_target_energy[
            growth_window_steps::growth_window_steps
        ].reshape(
            horizon // gradient_chunk_steps,
            gradient_chunk_steps // growth_window_steps,
            trajectory.shape[0],
        )
        targets = targets.reshape(
            horizon // gradient_chunk_steps,
            gradient_chunk_steps,
            *targets.shape[1:],
        )

        def step_body(carry, target):
            (
                state,
                latent,
                previous_latent,
                physical_num,
                physical_den,
                spectrum_num,
                spectrum_den,
                log_energy_error,
                time_relative_error,
                latent_error,
                bounce_exposure,
            ) = carry
            old_latent = latent
            density_hat = jnp.fft.rfft(state[:, 0], axis=-1, norm="forward")
            bounce_mode_count = min(4, k_arr.shape[0] - 1)
            bounce_modes = jnp.arange(1, bounce_mode_count + 1)
            field_mode = (
                1j
                * density_hat[:, 1 : bounce_mode_count + 1]
                / k_arr[bounce_modes][None, :]
            )
            bounce_increment = float(fine_steps) * float(fine_dt) * jnp.sqrt(
                jnp.maximum(
                    2.0 * k_arr[bounce_modes][None, :] * jnp.abs(field_mode),
                    0.0,
                )
            ).astype(bounce_exposure.dtype)
            state, latent = macro_step(
                params, state, latent, previous_latent, bounce_exposure
            )
            (
                numerator,
                denominator,
                spectral_numerator,
                spectral_denominator,
                step_log_energy_error,
                step_time_relative_error,
            ) = (
                _physical_trajectory_terms(
                    state,
                    target[:, :3],
                    k_arr,
                    poisson_sign=poisson_sign,
                    electric_energy_floor=energy_floor,
                    electric_time_relative_floor=time_relative_floor,
                )
            )
            predicted_field = primitive_fields(
                state,
                k_arr,
                poisson_sign=poisson_sign,
            )[:, 3]
            predicted_energy = jnp.mean(jnp.square(predicted_field), axis=-1)
            predicted_hat = jnp.fft.rfft(
                predicted_field, axis=-1, norm="forward"
            )[..., 1 : int(post_minimum_energy_modes) + 1]
            predicted_low_mode_energy = jnp.sum(
                jnp.square(jnp.abs(predicted_hat)), axis=-1
            )
            predicted_momentum_hat = jnp.fft.rfft(
                state[:, 1], axis=-1, norm="forward"
            )[..., 1 : int(post_minimum_energy_modes) + 1]
            predicted_transfer = 2.0 * jnp.real(
                predicted_hat * jnp.conj(predicted_momentum_hat)
            )
            return (
                state,
                latent,
                old_latent,
                physical_num + numerator,
                physical_den + denominator,
                spectrum_num + spectral_numerator,
                spectrum_den + spectral_denominator,
                log_energy_error + step_log_energy_error,
                time_relative_error + step_time_relative_error,
                latent_error
                + jnp.mean(
                    jnp.square(
                        (latent - target[:, 3:])
                        / jnp.asarray(latent_scale, dtype=latent.dtype)[None, :, None]
                    ),
                    axis=(1, 2),
                ),
                jnp.concatenate(
                    (
                        bounce_exposure[:, :bounce_mode_count] + bounce_increment,
                        jnp.maximum(
                            bounce_exposure[:, bounce_mode_count:],
                            bounce_increment
                            / (float(fine_steps) * float(fine_dt)),
                        ),
                    ),
                    axis=1,
                ),
            ), (
                predicted_energy,
                predicted_low_mode_energy,
                predicted_hat,
                predicted_transfer,
            )

        def chunk_body(carry, chunk_inputs):
            (
                chunk_targets,
                target_initial_energies,
                target_final_energies,
            ) = chunk_inputs
            state, latent, previous_latent, bounce_exposure = carry
            state = jax.lax.stop_gradient(state)
            latent = jax.lax.stop_gradient(latent)
            previous_latent = jax.lax.stop_gradient(previous_latent)
            bounce_exposure = jax.lax.stop_gradient(bounce_exposure)
            initial_field = primitive_fields(
                state,
                k_arr,
                poisson_sign=poisson_sign,
            )[:, 3]
            initial_energy = jnp.mean(jnp.square(initial_field), axis=-1)
            zeros = jnp.zeros((trajectory.shape[0], 4), dtype=trajectory.dtype)
            spectral_zeros = jnp.zeros((trajectory.shape[0],), dtype=trajectory.dtype)
            log_energy_zeros = jnp.zeros(
                (trajectory.shape[0],), dtype=trajectory.dtype
            )
            time_relative_zeros = jnp.zeros(
                (trajectory.shape[0],), dtype=trajectory.dtype
            )
            latent_zeros = jnp.zeros((trajectory.shape[0],), dtype=trajectory.dtype)
            final, (
                predicted_energies,
                predicted_low_mode_energies,
                predicted_low_mode_fields,
                predicted_transfers,
            ) = jax.lax.scan(
                jax.checkpoint(step_body),
                (
                    state,
                    latent,
                    previous_latent,
                    zeros,
                    zeros,
                    spectral_zeros,
                    spectral_zeros,
                    log_energy_zeros,
                    time_relative_zeros,
                    latent_zeros,
                    bounce_exposure,
                ),
                chunk_targets,
            )
            predicted_final_energies = predicted_energies[
                growth_window_steps - 1 :: growth_window_steps
            ]
            predicted_initial_energies = jnp.concatenate(
                (initial_energy[None], predicted_final_energies[:-1]),
                axis=0,
            )
            chunk_log_growth_error = _electric_chunk_log_growth_error(
                predicted_initial_energies,
                predicted_final_energies,
                target_initial_energies,
                target_final_energies,
                energy_floor,
            )
            return (final[0], final[1], final[2], final[10]), (
                final[3],
                final[4],
                final[5],
                final[6],
                final[7],
                final[8],
                final[9],
                chunk_log_growth_error,
                predicted_low_mode_energies,
                predicted_low_mode_fields,
                predicted_transfers,
            )

        (_, _, _, _), (
            chunk_numerators,
            chunk_denominators,
            chunk_spectral_numerators,
            chunk_spectral_denominators,
            chunk_log_energy_errors,
            chunk_time_relative_errors,
            chunk_latent_errors,
            chunk_log_growth_errors,
            chunk_low_mode_energies,
            chunk_low_mode_fields,
            chunk_transfers,
        ) = jax.lax.scan(
            chunk_body,
            (
                initial_state,
                initial_latent,
                initial_latent,
                jnp.zeros(
                    (trajectory.shape[0], 2 * min(4, k_arr.shape[0] - 1)),
                    dtype=trajectory.dtype,
                ),
            ),
            (
                targets,
                target_window_initial_energy,
                target_window_final_energy,
            ),
        )
        numerator = jnp.sum(chunk_numerators, axis=0)
        denominator = jnp.sum(chunk_denominators, axis=0)
        spectral_numerator = jnp.sum(chunk_spectral_numerators, axis=0)
        spectral_denominator = jnp.sum(chunk_spectral_denominators, axis=0)
        log_energy_loss = weighted_case_mean(
            jnp.sum(chunk_log_energy_errors, axis=0) / float(horizon)
        )
        time_relative_electric_loss = weighted_case_mean(
            jnp.sum(chunk_time_relative_errors, axis=0) / float(horizon)
        )
        autonomous_latent_loss = weighted_case_mean(
            jnp.sum(chunk_latent_errors, axis=0) / float(horizon)
        )
        chunk_log_growth_loss = weighted_case_mean(chunk_log_growth_errors)
        predicted_low_mode_energy = jnp.swapaxes(
            chunk_low_mode_energies.reshape(horizon, trajectory.shape[0]), 0, 1
        )
        complete_predicted_low_mode_energy = jnp.concatenate(
            (complete_target_low_mode_energy[:, :1], predicted_low_mode_energy),
            axis=1,
        )
        low_mode_floor = jnp.maximum(
            1e-6 * jnp.max(complete_target_low_mode_energy, axis=1, keepdims=True),
            jnp.finfo(trajectory.dtype).tiny,
        )
        sliding_window = int(electric_growth_window_steps)
        # A trailing average is causal and removes carrier-scale plasma
        # oscillations before measuring sustained damping or growth.  The
        # following lag is measured on that envelope, so every eligible
        # window contributes without selecting a particular turnaround.
        def trailing_mean(values):
            prefix = jnp.concatenate(
                (jnp.zeros_like(values[:, :1]), jnp.cumsum(values, axis=1)),
                axis=1,
            )
            return (
                prefix[:, sliding_window:] - prefix[:, :-sliding_window]
            ) / float(sliding_window)

        predicted_envelope = trailing_mean(complete_predicted_low_mode_energy)
        target_envelope = trailing_mean(complete_target_low_mode_energy)
        predicted_sliding_growth = (
            jnp.log(predicted_envelope[:, sliding_window:] + low_mode_floor)
            - jnp.log(predicted_envelope[:, :-sliding_window] + low_mode_floor)
        )
        target_sliding_growth = (
            jnp.log(target_envelope[:, sliding_window:] + low_mode_floor)
            - jnp.log(target_envelope[:, :-sliding_window] + low_mode_floor)
        )
        positive_growth_weight = jnp.maximum(target_sliding_growth, 0.0)
        negative_growth_weight = jnp.maximum(-target_sliding_growth, 0.0)
        positive_growth_weight = positive_growth_weight / jnp.maximum(
            jnp.sum(positive_growth_weight, axis=1, keepdims=True),
            jnp.finfo(trajectory.dtype).tiny,
        )
        negative_growth_weight = negative_growth_weight / jnp.maximum(
            jnp.sum(negative_growth_weight, axis=1, keepdims=True),
            jnp.finfo(trajectory.dtype).tiny,
        )
        if electric_sliding_log_growth_direction == "balanced_episodes":
            sliding_growth_weight = 0.5 * (
                _equal_contiguous_episode_weights(target_sliding_growth > 0.0)
                + _equal_contiguous_episode_weights(target_sliding_growth < 0.0)
            ).astype(trajectory.dtype)
        elif electric_sliding_log_growth_direction == "growth":
            sliding_growth_weight = positive_growth_weight
        elif electric_sliding_log_growth_direction == "growth_episodes":
            sliding_growth_weight = _equal_contiguous_episode_weights(
                target_sliding_growth > 0.0
            ).astype(trajectory.dtype)
        elif electric_sliding_log_growth_direction == "damping":
            sliding_growth_weight = negative_growth_weight
        else:
            sliding_growth_weight = 0.5 * (
                positive_growth_weight + negative_growth_weight
            )
        sliding_log_growth_loss = weighted_case_mean(
            jnp.sum(
                sliding_growth_weight
                * jnp.square(predicted_sliding_growth - target_sliding_growth),
                axis=1,
            )
        )
        predicted_low_mode_field = jnp.swapaxes(
            chunk_low_mode_fields.reshape(
                horizon, trajectory.shape[0], int(post_minimum_energy_modes)
            ),
            0,
            1,
        )
        predicted_transfer = jnp.swapaxes(
            chunk_transfers.reshape(
                horizon, trajectory.shape[0], int(post_minimum_energy_modes)
            ),
            0,
            1,
        )
        target_transfer = jnp.swapaxes(complete_target_transfer[1:], 0, 1)
        transfer_scale = jnp.maximum(
            jnp.mean(jnp.square(target_transfer), axis=1, keepdims=True),
            jnp.finfo(trajectory.dtype).tiny,
        )
        positive_transfer_weight = jnp.sum(
            jnp.maximum(target_transfer, 0.0), axis=-1
        )
        negative_transfer_weight = jnp.sum(
            jnp.maximum(-target_transfer, 0.0), axis=-1
        )
        positive_transfer_weight = positive_transfer_weight / jnp.maximum(
            jnp.sum(positive_transfer_weight, axis=1, keepdims=True),
            jnp.finfo(trajectory.dtype).tiny,
        )
        negative_transfer_weight = negative_transfer_weight / jnp.maximum(
            jnp.sum(negative_transfer_weight, axis=1, keepdims=True),
            jnp.finfo(trajectory.dtype).tiny,
        )
        # Damping often dominates the raw transfer integral.  Equal directional
        # mass prevents any number of later growth phases from being erased by
        # a long initial damping interval.
        transfer_time_weight = 0.5 * (
            positive_transfer_weight + negative_transfer_weight
        )
        electric_transfer_loss = weighted_case_mean(
            jnp.sum(
                transfer_time_weight
                * jnp.mean(
                    jnp.square(predicted_transfer - target_transfer)
                    / transfer_scale,
                    axis=-1,
                ),
                axis=1,
            )
        )
        target_low_mode_field = jnp.swapaxes(target_low_mode_hat[1:], 0, 1)
        target_low_mode_energy = complete_target_low_mode_energy[:, 1:]
        minimum_index, peak_index = _turnaround_indices(
            complete_target_low_mode_energy
        )
        post_minimum_mask = (
            jnp.arange(1, horizon + 1)[None, :] >= minimum_index[:, None]
        ).astype(trajectory.dtype)
        post_minimum_mask = post_minimum_mask / jnp.maximum(
            jnp.sum(post_minimum_mask, axis=1, keepdims=True), 1.0
        )
        post_minimum_field_loss = weighted_case_mean(
            jnp.sum(
                post_minimum_mask
                * (
                    jnp.sum(
                        jnp.square(
                            jnp.abs(
                                predicted_low_mode_field - target_low_mode_field
                            )
                        ),
                        axis=-1,
                    )
                    / (target_low_mode_energy + low_mode_floor)
                ),
                axis=1,
            )
        )
        turnaround_mask = (
            (jnp.arange(1, horizon + 1)[None, :] >= minimum_index[:, None])
            & (jnp.arange(1, horizon + 1)[None, :] <= peak_index[:, None])
        ).astype(trajectory.dtype)
        turnaround_mask = turnaround_mask / jnp.maximum(
            jnp.sum(turnaround_mask, axis=1, keepdims=True), 1.0
        )
        turnaround_field_loss = weighted_case_mean(
            jnp.sum(
                turnaround_mask
                * (
                    jnp.sum(
                        jnp.square(
                            jnp.abs(predicted_low_mode_field - target_low_mode_field)
                        ),
                        axis=-1,
                    )
                    / (target_low_mode_energy + low_mode_floor)
                ),
                axis=1,
            )
        )
        post_minimum_log_energy_loss = weighted_case_mean(
            jnp.sum(
                post_minimum_mask
                * jnp.square(
                    jnp.log(predicted_low_mode_energy + low_mode_floor)
                    - jnp.log(target_low_mode_energy + low_mode_floor)
                ),
                axis=1,
            )
        )
        post_peak_mask = (
            jnp.arange(1, horizon + 1)[None, :] >= peak_index[:, None]
        ).astype(trajectory.dtype)
        post_peak_mask = post_peak_mask / jnp.maximum(
            jnp.sum(post_peak_mask, axis=1, keepdims=True), 1.0
        )
        post_peak_log_energy_loss = weighted_case_mean(
            jnp.sum(
                post_peak_mask
                * jnp.square(
                    jnp.log(predicted_low_mode_energy + low_mode_floor)
                    - jnp.log(target_low_mode_energy + low_mode_floor)
                ),
                axis=1,
            )
        )
        case_index = jnp.arange(trajectory.shape[0])
        predicted_peak_index = jnp.maximum(peak_index - 1, 0)
        predicted_minimum_index = jnp.maximum(minimum_index - 1, 0)
        predicted_peak_energy = predicted_low_mode_energy[
            case_index, predicted_peak_index
        ]
        predicted_minimum_energy = predicted_low_mode_energy[
            case_index, predicted_minimum_index
        ]
        target_peak_energy = target_low_mode_energy[
            case_index, predicted_peak_index
        ]
        target_minimum_energy = target_low_mode_energy[
            case_index, predicted_minimum_index
        ]
        peak_log_energy_loss = weighted_case_mean(
            jnp.square(
                jnp.log(
                    predicted_low_mode_energy[case_index, predicted_peak_index]
                    + low_mode_floor[:, 0]
                )
                - jnp.log(
                    target_low_mode_energy[case_index, predicted_peak_index]
                    + low_mode_floor[:, 0]
                )
            )
        )
        turnaround_log_ratio_loss = weighted_case_mean(
            jnp.square(
                jnp.log(
                    (predicted_peak_energy + low_mode_floor[:, 0])
                    / (predicted_minimum_energy + low_mode_floor[:, 0])
                )
                - jnp.log(
                    (target_peak_energy + low_mode_floor[:, 0])
                    / (target_minimum_energy + low_mode_floor[:, 0])
                )
            )
        )
        post_peak_retention_loss = weighted_case_mean(
            jnp.square(
                jnp.log(
                    (predicted_low_mode_energy[:, -1] + low_mode_floor[:, 0])
                    / (predicted_peak_energy + low_mode_floor[:, 0])
                )
                - jnp.log(
                    jnp.maximum(
                        (target_low_mode_energy[:, -1] + low_mode_floor[:, 0])
                        / (target_peak_energy + low_mode_floor[:, 0]),
                        float(post_peak_retention_floor),
                    )
                )
            )
        )
        physical_loss = weighted_case_mean(
            _physical_sample_loss(numerator, denominator)
        )
        spectral_loss = weighted_case_mean(
            jnp.where(
                spectral_denominator > 0.0,
                spectral_numerator / spectral_denominator,
                0.0,
            )
        )
        physical_loss = physical_loss + float(electric_spectrum_weight) * spectral_loss
        physical_loss = physical_loss + (
            float(electric_log_energy_weight) * log_energy_loss
        )
        physical_loss = physical_loss + (
            float(electric_chunk_log_growth_weight) * chunk_log_growth_loss
        )
        physical_loss = physical_loss + (
            float(electric_sliding_log_growth_weight) * sliding_log_growth_loss
        )
        physical_loss = physical_loss + (
            float(electric_time_relative_weight) * time_relative_electric_loss
        )
        physical_loss = physical_loss + (
            float(electric_transfer_weight) * electric_transfer_loss
        )
        physical_loss = physical_loss + (
            float(post_minimum_log_energy_weight) * post_minimum_log_energy_loss
        )
        physical_loss = physical_loss + (
            float(post_minimum_field_weight) * post_minimum_field_loss
        )
        physical_loss = physical_loss + (
            float(turnaround_field_weight) * turnaround_field_loss
        )
        physical_loss = physical_loss + (
            float(post_peak_log_energy_weight) * post_peak_log_energy_loss
        )
        physical_loss = physical_loss + (
            float(peak_log_energy_weight) * peak_log_energy_loss
        )
        physical_loss = physical_loss + (
            float(turnaround_log_ratio_weight) * turnaround_log_ratio_loss
        )
        physical_loss = physical_loss + (
            float(post_peak_retention_weight) * post_peak_retention_loss
        )
        teacher_residual_loss = teacher_residual_loss_function(params, trajectory)
        teacher_residual_loss = teacher_residual_loss + (
            float(autonomous_latent_weight) * autonomous_latent_loss
        )
        total_loss = (
            physical_loss
            + float(latent_residual_weight) * teacher_residual_loss
        )
        return total_loss, physical_loss, teacher_residual_loss

    def loss_function(params, trajectory):
        return loss_components_function(params, trajectory)[0]

    def rollout_function(params, initial_resolved, initial_latent):
        initial_state = resolved_hermite_to_low_moment_state(initial_resolved)
        states, latents = rollout_coupled_low_moment_latent(
            params,
            initial_state,
            initial_latent,
            propagator,
            basis,
            k_arr,
            steps=horizon,
            latent_delay_input=latent_delay_input,
            resolved_center=resolved_center,
            resolved_scale=resolved_scale,
            latent_center=latent_center,
            latent_scale=latent_scale,
            latent_input_scale=latent_input_scale,
            depth=depth,
            correction_bounds=correction_bounds,
            closure_residual_scale=closure_residual_scale,
            closure_correction_bound=closure_correction_bound,
            closure_aligned_output=closure_aligned_output,
            closure_only_correction=closure_only_correction,
            closure_readout_mode=closure_readout_mode,
            closure_gate_scale=closure_gate_scale,
            closure_gate_power=closure_gate_power,
            latent_readout_mode=latent_readout_mode,
            latent_gate_scale=latent_gate_scale,
            latent_gate_power=latent_gate_power,
            latent_state_bound=latent_state_bound,
            equilibrium_input_compression_scale=equilibrium_input_compression_scale,
            linear_baseline=linear_baseline,
            hermite_tail_damping=hermite_tail_damping,
            hermite_tail_power=hermite_tail_power,
            semilinear_kick_scale=semilinear_kick_scale,
            semilinear_correction_location=semilinear_correction_location,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            dynamics_model="multiplicative_operator",
            poisson_sign=poisson_sign,
        )
        return low_moment_state_to_resolved_hermite(states), latents

    functions = (loss_function, rollout_function, loss_components_function)
    if return_teacher_residual_function:
        return (*functions, teacher_residual_loss_function)
    return functions


def _relative_error(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denominator_value = float(np.sum(np.square(denominator), dtype=np.float64))
    if not denominator_value > 0.0:
        raise ValueError("Complete target trajectory has zero norm")
    return float(np.sum(np.square(numerator), dtype=np.float64) / denominator_value)


def _energy_regrowth_diagnostics(
    predicted_field: np.ndarray,
    target_field: np.ndarray,
    *,
    cadence: float,
) -> dict[str, float]:
    """Compare model and teacher energy envelopes on the teacher turnaround interval."""
    predicted_energy = np.mean(np.square(predicted_field), axis=-1)
    target_energy = np.mean(np.square(target_field), axis=-1)
    maxima = np.flatnonzero(
        (target_energy[1:-1] >= target_energy[:-2])
        & (target_energy[1:-1] > target_energy[2:])
    ) + 1
    times = (np.arange(target_energy.size, dtype=np.float64) + 1.0) * float(
        cadence
    )
    maxima = maxima[times[maxima] >= 5.0]
    best_factor = 1.0
    start_index = int(maxima[0]) if maxima.size else 0
    end_index = start_index
    for offset, candidate in enumerate(maxima[:-1]):
        future = maxima[offset + 1 :]
        selected = int(future[np.argmax(target_energy[future])])
        factor = float(
            target_energy[selected]
            / max(target_energy[int(candidate)], np.finfo(np.float64).tiny)
        )
        if factor > best_factor:
            best_factor = factor
            start_index = int(candidate)
            end_index = selected
    radius = max(1, int(round(1.0 / float(cadence))))

    def local_envelope(values: np.ndarray, index: int) -> float:
        lower = max(0, int(index) - radius)
        upper = min(values.size, int(index) + radius + 1)
        return float(np.max(values[lower:upper]))

    predicted_factor = local_envelope(
        predicted_energy, end_index
    ) / max(
        local_envelope(predicted_energy, start_index),
        np.finfo(np.float64).tiny,
    )
    return {
        "teacher_regrowth_factor": float(best_factor),
        "predicted_regrowth_factor": float(predicted_factor),
        "regrowth_log_error": float(
            abs(
                math.log(max(predicted_factor, np.finfo(np.float64).tiny))
                - math.log(max(best_factor, np.finfo(np.float64).tiny))
            )
        ),
        "regrowth_start_time": float(times[start_index]),
        "regrowth_end_time": float(times[end_index]),
    }


def _evaluate(
    params,
    rollout,
    projected_cache: Path,
    cases: list[dict],
    *,
    latent_rank: int,
    expected_samples: int,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    k_arr: np.ndarray,
    poisson_sign: float,
    correction_function=None,
    closure_function=None,
    correction_bounds: np.ndarray | None = None,
    propagator: np.ndarray | None = None,
    latent_residual_scale: np.ndarray | None = None,
    basis: np.ndarray | None = None,
    closure_residual_scale: float = 1.0,
    cadence: float = 0.1,
    linear_baseline: str = "fitted_propagator",
    fine_steps: int = 10,
    fine_dt: float = 0.01,
    hermite_tail_damping: float = 10.0,
    hermite_tail_power: float = 6.0,
    semilinear_kick_scale: float = 1.0,
    semilinear_correction_location: str = "midpoint",
    latent_delay_input: bool = False,
) -> dict:
    semilinear_baseline_step = None
    if linear_baseline == "semilinear_strang":
        if propagator is None or basis is None:
            raise ValueError(
                "semilinear evaluation requires propagator and basis"
            )
        semilinear_baseline_step = _make_semilinear_strang_baseline_step(
            propagator,
            basis,
            k_arr,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
            fine_steps=fine_steps,
            fine_dt=fine_dt,
            kick_scale=semilinear_kick_scale,
            poisson_sign=poisson_sign,
        )
    by_regime = {}
    for regime in REGIMES:
        rows = []
        for case in _selected_cases(cases, regime, "heldout"):
            target = _load_complete_case(
                projected_cache,
                case,
                latent_rank=latent_rank,
                expected_samples=expected_samples,
            )
            predicted_resolved, predicted_latent = rollout(
                params,
                jnp.asarray(target[None, 0, :3]),
                jnp.asarray(target[None, 0, 3:]),
            )
            predicted_resolved = np.asarray(predicted_resolved[0])
            predicted_latent = np.asarray(predicted_latent[0])
            target_resolved = target[1:, :3]
            target_latent = target[1:, 3:]
            resolved_error = _relative_error(
                (predicted_resolved - target_resolved)
                / resolved_scale[None, :, None],
                target_resolved / resolved_scale[None, :, None],
            )
            latent_error = _relative_error(
                (predicted_latent - target_latent) / latent_scale[None, :, None],
                target_latent / latent_scale[None, :, None],
            )
            predicted_fields = np.asarray(
                primitive_fields(
                    resolved_hermite_to_low_moment_state(
                        jnp.asarray(predicted_resolved)
                    ),
                    jnp.asarray(k_arr),
                    poisson_sign=poisson_sign,
                ),
                dtype=np.float64,
            )
            target_fields = np.asarray(
                primitive_fields(
                    resolved_hermite_to_low_moment_state(jnp.asarray(target_resolved)),
                    jnp.asarray(k_arr),
                    poisson_sign=poisson_sign,
                ),
                dtype=np.float64,
            )
            channel_errors = [
                _relative_error(
                    predicted_fields[:, channel] - target_fields[:, channel],
                    target_fields[:, channel],
                )
                for channel in range(4)
            ]
            physical_error = float(np.mean(channel_errors))
            field_error = float(channel_errors[3])
            predicted_electric_hat = np.fft.rfft(
                predicted_fields[:, 3], axis=-1, norm="forward"
            )[:, 1:]
            target_electric_hat = np.fft.rfft(
                target_fields[:, 3], axis=-1, norm="forward"
            )[:, 1:]
            spectral_amplitude_error = _relative_error(
                np.abs(predicted_electric_hat) - np.abs(target_electric_hat),
                np.abs(target_electric_hat),
            )
            regrowth = _energy_regrowth_diagnostics(
                predicted_fields[:, 3],
                target_fields[:, 3],
                cadence=float(cadence),
            )
            finite = bool(
                np.all(np.isfinite(predicted_resolved))
                and np.all(np.isfinite(predicted_latent))
            )
            normalized_latent = (
                predicted_latent / latent_scale[None, :, None]
            )
            row = {
                    "case_id": str(case["case_id"]),
                    "resolved_relative_mse": resolved_error,
                    "latent_relative_mse": latent_error,
                    "physical_relative_loss": physical_error,
                    "physical_channel_relative_mse": {
                        name: float(value)
                        for name, value in zip(
                            ("density", "velocity", "pressure", "electric_field"),
                            channel_errors,
                        )
                    },
                    "electric_field_relative_mse": field_error,
                    "electric_spectral_amplitude_relative_mse": spectral_amplitude_error,
                    **regrowth,
                    "maximum_normalized_latent": float(
                        np.max(np.abs(normalized_latent))
                    ),
                    "finite": finite,
                }
            if correction_function is not None:
                if correction_bounds is None or propagator is None:
                    raise ValueError(
                        "correction bounds and propagator are required for correction diagnostics"
                    )
                resolved_inputs = np.concatenate(
                    (target[None, 0, :3], predicted_resolved[:-1]), axis=0
                )
                latent_inputs = np.concatenate(
                    (target[None, 0, 3:], predicted_latent[:-1]), axis=0
                )
                previous_latent_inputs = np.concatenate(
                    (target[None, 0, 3:], latent_inputs[:-1]),
                    axis=0,
                )
                normalized_latent_delta = (
                    (latent_inputs - previous_latent_inputs)
                    / latent_scale[None, :, None]
                    if latent_delay_input
                    else np.zeros_like(latent_inputs)
                )
                raw_correction = np.asarray(
                    correction_function(
                        params,
                        jnp.asarray(
                            resolved_inputs / resolved_scale[None, :, None]
                        ),
                        jnp.asarray(
                            latent_inputs / latent_scale[None, :, None]
                        ),
                        jnp.asarray(normalized_latent_delta),
                    )
                )
                absolute_correction = np.abs(raw_correction)
                bound_ratio = absolute_correction / np.asarray(
                    correction_bounds, dtype=np.float64
                )[None, :, None]
                row.update(
                    {
                        "correction_normalized_rms": float(
                            np.sqrt(np.mean(np.square(raw_correction)))
                        ),
                        "correction_old_bound_exceedance_fraction": float(
                            np.mean(absolute_correction > 1.0)
                        ),
                        "correction_bound_usage_p99": float(
                            np.quantile(bound_ratio, 0.99)
                        ),
                        "correction_bound_usage_maximum": float(
                            np.max(bound_ratio)
                        ),
                    }
                )
                teacher_current = target[:-1]
                teacher_next = target[1:]
                teacher_previous_latent = np.concatenate(
                    (target[None, 0, 3:], target[:-2, 3:]), axis=0
                )
                teacher_normalized_latent_delta = (
                    (teacher_current[:, 3:] - teacher_previous_latent)
                    / latent_scale[None, :, None]
                    if latent_delay_input
                    else np.zeros_like(teacher_current[:, 3:])
                )
                target_closure_latent = None
                if linear_baseline == "projected_hermite":
                    if basis is None:
                        raise ValueError(
                            "basis is required for projected-Hermite diagnostics"
                        )
                    baseline_latent = _projected_hermite_baseline_latent_rows(
                        target,
                        basis,
                        k_arr,
                        fine_steps=fine_steps,
                        fine_dt=fine_dt,
                        hermite_tail_damping=hermite_tail_damping,
                        hermite_tail_power=hermite_tail_power,
                        poisson_sign=poisson_sign,
                    )
                    teacher_target_correction = (
                        teacher_next[:, 3:] - baseline_latent
                    ) / latent_scale[None, :, None]
                    target_closure_latent = teacher_next[:, 3:] - baseline_latent
                elif linear_baseline == "semilinear_strang":
                    if basis is None:
                        raise ValueError(
                            "basis is required for semilinear diagnostics"
                        )
                    baseline_resolved, baseline_latent = (
                        _semilinear_strang_baseline_rows(
                            target,
                            propagator,
                            basis,
                            k_arr,
                            resolved_scale=resolved_scale,
                            latent_scale=latent_scale,
                            fine_steps=fine_steps,
                            fine_dt=fine_dt,
                            kick_scale=semilinear_kick_scale,
                            poisson_sign=poisson_sign,
                            baseline_step=semilinear_baseline_step,
                        )
                    )
                    endpoint_target_correction = (
                        teacher_next[:, 3:] - baseline_latent
                    ) / latent_scale[None, :, None]
                    if semilinear_correction_location == "midpoint":
                        teacher_target_correction = _semilinear_midpoint_correction(
                            teacher_next,
                            baseline_resolved,
                            baseline_latent,
                            _matrix_square_root_propagator(propagator),
                            resolved_scale=resolved_scale,
                            latent_scale=latent_scale,
                        )
                    else:
                        teacher_target_correction = endpoint_target_correction
                    target_closure_latent = (
                        teacher_target_correction
                        * latent_scale[None, :, None]
                    )
                else:
                    teacher_linear = np.asarray(
                        apply_coupled_linear_propagator(
                            jnp.asarray(propagator),
                            jnp.asarray(teacher_current[:, :3]),
                            jnp.asarray(teacher_current[:, 3:]),
                            resolved_scale=jnp.asarray(resolved_scale),
                            latent_scale=jnp.asarray(latent_scale),
                        )
                    )
                    baseline_latent = teacher_linear[:, 3:]
                    teacher_target_correction = (
                        teacher_next[:, 3:] - baseline_latent
                    ) / latent_scale[None, :, None]
                    target_closure_latent = teacher_next[:, 3:] - baseline_latent
                teacher_raw_correction = np.asarray(
                    correction_function(
                        params,
                        jnp.asarray(
                            teacher_current[:, :3]
                            / resolved_scale[None, :, None]
                        ),
                        jnp.asarray(
                            teacher_current[:, 3:]
                            / latent_scale[None, :, None]
                        ),
                        jnp.asarray(teacher_normalized_latent_delta),
                    )
                )
                teacher_predicted_correction = np.asarray(correction_bounds)[
                    None, :, None
                ] * np.tanh(
                    teacher_raw_correction
                    / np.asarray(correction_bounds)[None, :, None]
                )
                row["teacher_target_correction_rms"] = float(
                    np.sqrt(np.mean(np.square(teacher_target_correction)))
                )
                row["teacher_input_predicted_correction_rms"] = float(
                    np.sqrt(np.mean(np.square(teacher_predicted_correction)))
                )
                if latent_residual_scale is not None:
                    residual_target = teacher_target_correction
                    residual_prediction = teacher_predicted_correction
                    if (
                        linear_baseline == "semilinear_strang"
                        and semilinear_correction_location == "midpoint"
                    ):
                        residual_target = endpoint_target_correction
                        residual_prediction = _semilinear_endpoint_latent_correction(
                            teacher_predicted_correction,
                            _matrix_square_root_propagator(propagator),
                        )
                    normalized_error = (
                        residual_prediction - residual_target
                    ) / np.asarray(latent_residual_scale)[None, :, None]
                    row["teacher_residual_normalized_mse"] = float(
                        np.mean(np.square(normalized_error))
                    )
                if closure_function is not None:
                    if basis is None:
                        raise ValueError("basis is required for closure diagnostics")
                    predicted_closure = np.asarray(
                        closure_function(
                            params,
                            jnp.asarray(
                                teacher_current[:, :3]
                                / resolved_scale[None, :, None]
                            ),
                            jnp.asarray(
                                teacher_current[:, 3:]
                                / latent_scale[None, :, None]
                            ),
                            jnp.asarray(teacher_normalized_latent_delta),
                        )
                    ) * float(closure_residual_scale)
                    target_closure = np.einsum(
                        "r,trx->tx",
                        np.asarray(basis)[0],
                        target_closure_latent,
                    )
                    row["teacher_target_closure_rms"] = float(
                        np.sqrt(np.mean(np.square(target_closure)))
                    )
                    row["teacher_input_predicted_closure_rms"] = float(
                        np.sqrt(np.mean(np.square(predicted_closure)))
                    )
                    row["teacher_closure_relative_mse"] = _relative_error(
                        predicted_closure - target_closure,
                        target_closure,
                    )
            rows.append(row)
        regrowth_rows = [
            row for row in rows if row["teacher_regrowth_factor"] > 1.01
        ]
        if regrowth_rows:
            teacher_regrowth_factor = float(
                np.exp(
                    np.mean(
                        [
                            math.log(row["teacher_regrowth_factor"])
                            for row in regrowth_rows
                        ]
                    )
                )
            )
            predicted_regrowth_factor = float(
                np.exp(
                    np.mean(
                        [
                            math.log(
                                max(
                                    row["predicted_regrowth_factor"],
                                    np.finfo(np.float64).tiny,
                                )
                            )
                            for row in regrowth_rows
                        ]
                    )
                )
            )
            regrowth_log_error = float(
                np.mean([row["regrowth_log_error"] for row in regrowth_rows])
            )
        else:
            teacher_regrowth_factor = 1.0
            predicted_regrowth_factor = 1.0
            regrowth_log_error = 0.0
        by_regime[regime] = {
            "cases": rows,
            "resolved_relative_mse": float(
                np.mean([row["resolved_relative_mse"] for row in rows])
            ),
            "latent_relative_mse": float(
                np.mean([row["latent_relative_mse"] for row in rows])
            ),
            "physical_relative_loss": float(
                np.mean([row["physical_relative_loss"] for row in rows])
            ),
            "electric_field_relative_mse": float(
                np.mean([row["electric_field_relative_mse"] for row in rows])
            ),
            "electric_spectral_amplitude_relative_mse": float(
                np.mean(
                    [
                        row["electric_spectral_amplitude_relative_mse"]
                        for row in rows
                    ]
                )
            ),
            "regrowth_case_count": len(regrowth_rows),
            "teacher_regrowth_factor": teacher_regrowth_factor,
            "predicted_regrowth_factor": predicted_regrowth_factor,
            "regrowth_log_error": regrowth_log_error,
            "maximum_normalized_latent": float(
                np.max([row["maximum_normalized_latent"] for row in rows])
            ),
            "finite": all(row["finite"] for row in rows),
        }
        if correction_function is not None:
            by_regime[regime].update(
                {
                    "correction_normalized_rms": float(
                        np.mean(
                            [row["correction_normalized_rms"] for row in rows]
                        )
                    ),
                    "correction_old_bound_exceedance_fraction": float(
                        np.mean(
                            [
                                row["correction_old_bound_exceedance_fraction"]
                                for row in rows
                            ]
                        )
                    ),
                    "correction_bound_usage_p99": float(
                        np.mean(
                            [row["correction_bound_usage_p99"] for row in rows]
                        )
                    ),
                    "correction_bound_usage_maximum": float(
                        np.max(
                            [
                                row["correction_bound_usage_maximum"]
                                for row in rows
                            ]
                        )
                    ),
                    "teacher_target_correction_rms": float(
                        np.mean([row["teacher_target_correction_rms"] for row in rows])
                    ),
                    "teacher_input_predicted_correction_rms": float(
                        np.mean(
                            [
                                row["teacher_input_predicted_correction_rms"]
                                for row in rows
                            ]
                        )
                    ),
                }
            )
            if closure_function is not None:
                by_regime[regime].update(
                    {
                        "teacher_target_closure_rms": float(
                            np.mean(
                                [row["teacher_target_closure_rms"] for row in rows]
                            )
                        ),
                        "teacher_input_predicted_closure_rms": float(
                            np.mean(
                                [
                                    row["teacher_input_predicted_closure_rms"]
                                    for row in rows
                                ]
                            )
                        ),
                        "teacher_closure_relative_mse": float(
                            np.mean(
                                [row["teacher_closure_relative_mse"] for row in rows]
                            )
                        ),
                    }
                )
            if latent_residual_scale is not None:
                by_regime[regime]["teacher_residual_normalized_mse"] = float(
                    np.mean(
                        [row["teacher_residual_normalized_mse"] for row in rows]
                    )
                )
    return by_regime


def _atomic_savez(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def _training_state_payload(params, optimizers) -> dict[str, object]:
    payload = {
        f"params__{key}": np.asarray(value) for key, value in params.items()
    }
    for objective, optimizer in optimizers.items():
        payload.update(
            {
                f"adam_{objective}_m__{key}": np.asarray(value)
                for key, value in optimizer["m"].items()
            }
        )
        payload.update(
            {
                f"adam_{objective}_v__{key}": np.asarray(value)
                for key, value in optimizer["v"].items()
            }
        )
        payload[f"adam_{objective}_step"] = np.asarray(optimizer["step"])
    return payload


def _load_training_state(path: Path, params, optimizers):
    """Restore model and Adam state while validating the current architecture."""
    with np.load(path, allow_pickle=False) as saved:
        restored_params = {}
        for key, template in params.items():
            saved_key = f"params__{key}"
            if saved_key not in saved:
                raise ValueError(f"Resume state is missing {saved_key}")
            value = np.asarray(saved[saved_key])
            if value.shape != tuple(template.shape):
                raise ValueError(
                    f"Resume parameter {key} has shape {value.shape}; "
                    f"expected {tuple(template.shape)}"
                )
            restored_params[key] = jnp.asarray(value, dtype=template.dtype)

        restored_optimizers = {}
        for objective, template in optimizers.items():
            moments = {}
            variances = {}
            for key, value_template in template["m"].items():
                moment_key = f"adam_{objective}_m__{key}"
                variance_key = f"adam_{objective}_v__{key}"
                if moment_key not in saved or variance_key not in saved:
                    raise ValueError(
                        f"Resume state is missing Adam values for {objective}:{key}"
                    )
                moment = np.asarray(saved[moment_key])
                variance = np.asarray(saved[variance_key])
                expected = tuple(value_template.shape)
                if moment.shape != expected or variance.shape != expected:
                    raise ValueError(
                        f"Resume Adam values for {objective}:{key} do not match "
                        f"the expected shape {expected}"
                    )
                moments[key] = jnp.asarray(moment, dtype=value_template.dtype)
                variances[key] = jnp.asarray(variance, dtype=value_template.dtype)
            step_key = f"adam_{objective}_step"
            if step_key not in saved:
                raise ValueError(f"Resume state is missing {step_key}")
            restored_optimizers[objective] = {
                "m": moments,
                "v": variances,
                "step": jnp.asarray(saved[step_key], dtype=template["step"].dtype),
            }
        if "completed_epoch" not in saved:
            raise ValueError("Resume state is missing completed_epoch")
        completed_epoch = int(np.asarray(saved["completed_epoch"]))
    return restored_params, restored_optimizers, completed_epoch


def _first_nonfinite_step(resolved: np.ndarray, latent: np.ndarray) -> int | None:
    finite = np.all(np.isfinite(resolved), axis=tuple(range(1, resolved.ndim)))
    finite &= np.all(np.isfinite(latent), axis=tuple(range(1, latent.ndim)))
    indices = np.flatnonzero(~finite)
    return None if indices.size == 0 else int(indices[0]) + 1


def _finite_maximum(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return None if finite.size == 0 else float(np.max(np.abs(finite)))


def _failure_diagnostics(
    params,
    batch: np.ndarray,
    batch_cases: list[dict],
    rollout,
    correction_function,
    *,
    resolved_scale: np.ndarray,
    latent_scale: np.ndarray,
    correction_bounds: np.ndarray,
) -> list[dict]:
    diagnostics = []
    for trajectory, case in zip(batch, batch_cases):
        try:
            predicted_resolved, predicted_latent = rollout(
                params,
                jnp.asarray(trajectory[None, 0, :3]),
                jnp.asarray(trajectory[None, 0, 3:]),
            )
            predicted_resolved = np.asarray(predicted_resolved[0])
            predicted_latent = np.asarray(predicted_latent[0])
            first_nonfinite = _first_nonfinite_step(
                predicted_resolved, predicted_latent
            )
            resolved_inputs = np.concatenate(
                (trajectory[None, 0, :3], predicted_resolved[:-1]), axis=0
            )
            latent_inputs = np.concatenate(
                (trajectory[None, 0, 3:], predicted_latent[:-1]), axis=0
            )
            raw_correction = np.asarray(
                correction_function(
                    params,
                    jnp.asarray(resolved_inputs / resolved_scale[None, :, None]),
                    jnp.asarray(latent_inputs / latent_scale[None, :, None]),
                )
            )
            bounded = np.asarray(correction_bounds)[None, :, None] * np.tanh(
                raw_correction / np.asarray(correction_bounds)[None, :, None]
            )
            diagnostics.append(
                {
                    "case_id": str(case["case_id"]),
                    "regime": str(case["regime"]),
                    "first_nonfinite_step": first_nonfinite,
                    "maximum_resolved_state": _finite_maximum(predicted_resolved),
                    "maximum_normalized_latent": _finite_maximum(
                        predicted_latent / latent_scale[None, :, None]
                    ),
                    "maximum_raw_normalized_correction": _finite_maximum(
                        raw_correction
                    ),
                    "maximum_bounded_normalized_correction": _finite_maximum(bounded),
                }
            )
        except Exception as error:  # Preserve the original training failure.
            diagnostics.append(
                {
                    "case_id": str(case["case_id"]),
                    "regime": str(case["regime"]),
                    "diagnostic_error": repr(error),
                }
            )
    return diagnostics


def _write_training_artifacts(report: dict, outdir: Path) -> None:
    history = list(report["history"])
    epochs = np.asarray([row["epoch"] for row in history], dtype=np.int32)
    train_loss = np.asarray([row["loss"] for row in history], dtype=np.float64)
    train_physical = np.asarray(
        [row.get("physical_loss", row["loss"]) for row in history],
        dtype=np.float64,
    )
    train_latent = np.asarray(
        [
            row.get(
                "teacher_residual_loss",
                row.get(
                    "autonomous_latent_loss",
                    row.get("latent_residual_loss", np.nan),
                ),
            )
            for row in history
        ],
        dtype=np.float64,
    )
    train_ema = np.asarray([row["loss_ema"] for row in history], dtype=np.float64)
    validation_rows = [row for row in history if "heldout" in row]
    validation_epochs = np.asarray(
        [row["epoch"] for row in validation_rows], dtype=np.int32
    )
    physical_by_regime = np.asarray(
        [
            [row["heldout"][regime]["physical_relative_loss"] for regime in REGIMES]
            for row in validation_rows
        ],
        dtype=np.float64,
    )
    field_by_regime = np.asarray(
        [
            [
                row["heldout"][regime]["electric_field_relative_mse"]
                for regime in REGIMES
            ]
            for row in validation_rows
        ],
        dtype=np.float64,
    )
    spectrum_by_regime = np.asarray(
        [
            [
                row["heldout"][regime][
                    "electric_spectral_amplitude_relative_mse"
                ]
                for regime in REGIMES
            ]
            for row in validation_rows
        ],
        dtype=np.float64,
    )
    latent_by_regime = np.asarray(
        [
            [row["heldout"][regime]["latent_relative_mse"] for regime in REGIMES]
            for row in validation_rows
        ],
        dtype=np.float64,
    )
    checkpoint_score = np.asarray(
        [row["checkpoint_score"] for row in validation_rows], dtype=np.float64
    )
    np.savez(
        outdir / "training_metrics.npz",
        epochs=epochs,
        train_objective=train_loss,
        train_physical_loss=train_physical,
        train_teacher_residual_loss=train_latent,
        train_ema_loss=train_ema,
        validation_epochs=validation_epochs,
        heldout_physical_loss_by_regime=physical_by_regime,
        heldout_electric_field_mse_by_regime=field_by_regime,
        heldout_electric_spectral_amplitude_mse_by_regime=spectrum_by_regime,
        heldout_latent_mse_by_regime=latent_by_regime,
        checkpoint_score=checkpoint_score,
    )

    fig, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    axis.semilogy(
        epochs,
        train_loss,
        color="#8b93a1",
        alpha=0.55,
        label="loss monitor",
    )
    axis.semilogy(
        epochs,
        train_physical,
        color="#4477aa",
        linewidth=1.4,
        label="full-T physical",
    )
    latent_weight = float(report.get("configuration", {}).get("latent_residual_weight", 0.0))
    if latent_weight > 0.0 and np.any(np.isfinite(train_latent)):
        axis.semilogy(
            epochs,
            latent_weight * train_latent,
            color="#228833",
            linewidth=1.2,
            label="weighted teacher increment",
        )
    axis.semilogy(epochs, train_ema, color="#526071", linewidth=1.8, label="batch-loss EMA")
    axis.semilogy(
        validation_epochs,
        np.mean(physical_by_regime, axis=1),
        color="#c44e52",
        marker="o",
        label="held-out physical mean",
    )
    axis.semilogy(
        validation_epochs,
        checkpoint_score,
        color="#172033",
        marker="o",
        label="held-out worst regime",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Relative physical trajectory loss")
    axis.set_title("Complete-T coupled low-moment latent training")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(outdir / "training_loss.png", dpi=200)
    plt.close(fig)

    labels = ("linear", "weak nonlinear", "strong nonlinear")
    colors = ("#4477aa", "#cc6677", "#228833")
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.0), constrained_layout=True)
    panels = (
        (physical_by_regime, "Physical trajectory loss"),
        (field_by_regime, "Electric-field relative MSE"),
        (spectrum_by_regime, "Electric-spectrum amplitude MSE"),
        (latent_by_regime, "Latent MSE (diagnostic only)"),
    )
    for axis, (values, title) in zip(axes, panels):
        for index, (label, color) in enumerate(zip(labels, colors)):
            axis.semilogy(
                validation_epochs,
                values[:, index],
                color=color,
                marker="o",
                label=label,
            )
        axis.set_xlabel("Epoch")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Relative error")
    axes[-1].legend()
    fig.savefig(outdir / "training_diagnostics.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    print_jax_runtime_summary(jax, context="coupled low-moment latent training")
    if not 0.0 <= float(args.loss_ema_decay) < 1.0:
        raise ValueError("loss_ema_decay must lie in [0, 1)")
    if float(args.operator_output_init_scale) < 0.0:
        raise ValueError("operator_output_init_scale must be nonnegative")
    if float(args.multiplicative_output_ridge) < 0.0:
        raise ValueError("multiplicative_output_ridge must be nonnegative")
    if not 0.0 < float(args.multiplicative_output_initial_scale) <= 1.0:
        raise ValueError("multiplicative_output_initial_scale must lie in (0, 1]")
    if args.fit_multiplicative_output_readout and not (
        (
            args.linear_baseline == "projected_hermite"
            or (
                args.linear_baseline == "semilinear_strang"
                and args.semilinear_correction_location == "endpoint"
            )
        )
        and args.latent_readout_mode == "multiplicative"
        and not args.closure_only_correction
    ):
        raise ValueError(
            "multiplicative output fitting requires an endpoint projected-Hermite "
            "or semilinear multiplicative latent correction"
        )
    if args.latent_delay_input and (
        args.latent_readout_mode not in {"equilibrium_cnn", "multiplicative"}
        or args.closure_readout_mode != "multiplicative"
    ):
        raise ValueError(
            "latent delay input requires multiplicative readouts"
        )
    if float(args.latent_residual_weight) < 0.0:
        raise ValueError("latent_residual_weight must be nonnegative")
    if float(args.latent_state_residual_weight) < 0.0:
        raise ValueError("latent_state_residual_weight must be nonnegative")
    if float(args.low_mode_latent_residual_weight) < 0.0:
        raise ValueError("low_mode_latent_residual_weight must be nonnegative")
    if not np.isfinite(float(args.closure_correction_bound)) or float(
        args.closure_correction_bound
    ) <= 0.0:
        raise ValueError("closure_correction_bound must be finite and positive")
    if args.closure_only_correction and not args.closure_aligned_output:
        raise ValueError(
            "closure_only_correction requires closure_aligned_output"
        )
    if not np.isfinite(float(args.closure_gate_scale)) or float(
        args.closure_gate_scale
    ) <= 0.0:
        raise ValueError("closure_gate_scale must be finite and positive")
    if int(args.closure_gate_power) < 2 or int(args.closure_gate_power) % 2:
        raise ValueError("closure_gate_power must be a positive even integer")
    if not np.isfinite(float(args.closure_readout_ridge)) or float(
        args.closure_readout_ridge
    ) < 0.0:
        raise ValueError("closure_readout_ridge must be finite and nonnegative")
    if not np.isfinite(float(args.closure_readout_initial_scale)) or not (
        0.0 < float(args.closure_readout_initial_scale) <= 1.0
    ):
        raise ValueError("closure_readout_initial_scale must lie in (0, 1]")
    if not 0.0 <= float(args.closure_readout_trajectory_exponent) <= 1.0:
        raise ValueError("closure_readout_trajectory_exponent must lie in [0, 1]")
    if args.closure_readout_mode != "multiplicative" and not (
        args.closure_aligned_output and args.closure_only_correction
    ):
        raise ValueError(
            "gated closure readout requires closure-aligned, closure-only output"
        )
    if not np.isfinite(float(args.latent_gate_scale)) or float(
        args.latent_gate_scale
    ) <= 0.0:
        raise ValueError("latent_gate_scale must be finite and positive")
    if int(args.latent_gate_power) < 2 or int(args.latent_gate_power) % 2:
        raise ValueError("latent_gate_power must be a positive even integer")
    if float(args.equilibrium_input_compression_scale) < 0.0:
        raise ValueError(
            "equilibrium_input_compression_scale must be nonnegative"
        )
    if not np.isfinite(float(args.latent_readout_ridge)) or float(
        args.latent_readout_ridge
    ) < 0.0:
        raise ValueError("latent_readout_ridge must be finite and nonnegative")
    if not np.isfinite(float(args.latent_readout_initial_scale)) or not (
        0.0 < float(args.latent_readout_initial_scale) <= 1.0
    ):
        raise ValueError("latent_readout_initial_scale must lie in (0, 1]")
    if not 0.0 <= float(args.latent_readout_trajectory_exponent) <= 1.0:
        raise ValueError("latent_readout_trajectory_exponent must lie in [0, 1]")
    if args.latent_readout_mode != "multiplicative" and args.closure_only_correction:
        raise ValueError("gated latent readout is incompatible with closure-only output")
    if float(args.autonomous_latent_weight) < 0.0:
        raise ValueError("autonomous_latent_weight must be nonnegative")
    if float(args.electric_spectrum_weight) < 0.0:
        raise ValueError("electric_spectrum_weight must be nonnegative")
    if float(args.electric_log_energy_weight) < 0.0:
        raise ValueError("electric_log_energy_weight must be nonnegative")
    if float(args.electric_chunk_log_growth_weight) < 0.0:
        raise ValueError("electric_chunk_log_growth_weight must be nonnegative")
    if float(args.electric_sliding_log_growth_weight) < 0.0:
        raise ValueError("electric_sliding_log_growth_weight must be nonnegative")
    if int(args.electric_growth_window_steps) <= 0:
        raise ValueError("electric_growth_window_steps must be positive")
    if not 0.0 < float(args.electric_log_energy_floor_ratio) < 1.0:
        raise ValueError("electric_log_energy_floor_ratio must lie in (0, 1)")
    if float(args.electric_time_relative_weight) < 0.0:
        raise ValueError("electric_time_relative_weight must be nonnegative")
    if float(args.electric_transfer_weight) < 0.0:
        raise ValueError("electric_transfer_weight must be nonnegative")
    if float(args.post_minimum_log_energy_weight) < 0.0:
        raise ValueError("post_minimum_log_energy_weight must be nonnegative")
    if float(args.post_minimum_field_weight) < 0.0:
        raise ValueError("post_minimum_field_weight must be nonnegative")
    if float(args.turnaround_field_weight) < 0.0:
        raise ValueError("turnaround_field_weight must be nonnegative")
    if float(args.post_peak_log_energy_weight) < 0.0:
        raise ValueError("post_peak_log_energy_weight must be nonnegative")
    if float(args.peak_log_energy_weight) < 0.0:
        raise ValueError("peak_log_energy_weight must be nonnegative")
    if float(args.turnaround_log_ratio_weight) < 0.0:
        raise ValueError("turnaround_log_ratio_weight must be nonnegative")
    if float(args.post_peak_retention_weight) < 0.0:
        raise ValueError("post_peak_retention_weight must be nonnegative")
    if not 0.0 <= float(args.post_peak_retention_floor) <= 1.0:
        raise ValueError("post_peak_retention_floor must lie in [0, 1]")
    regime_weights = (
        float(args.linear_regime_weight),
        float(args.weak_regime_weight),
        float(args.strong_regime_weight),
    )
    if any(weight < 0.0 for weight in regime_weights) or not any(
        weight > 0.0 for weight in regime_weights
    ):
        raise ValueError("regime weights must be nonnegative and not all zero")
    if not 1 <= int(args.post_minimum_energy_modes) <= int(args.nx) // 2:
        raise ValueError("post_minimum_energy_modes must select retained nonzero modes")
    if not 0.0 < float(args.electric_time_relative_floor_ratio) < 1.0:
        raise ValueError("electric_time_relative_floor_ratio must lie in (0, 1)")
    if float(args.latent_gradient_ratio) < 0.0:
        raise ValueError("latent_gradient_ratio must be nonnegative")
    if (
        float(args.latent_residual_weight) > 0.0
        and float(args.latent_state_residual_weight) == 0.0
        and float(args.low_mode_latent_residual_weight) == 0.0
        and float(args.closure_residual_weight) == 0.0
        and float(args.autonomous_latent_weight) == 0.0
    ):
        raise ValueError(
            "latent_residual_weight is positive but every latent auxiliary "
            "component is disabled"
        )
    if float(args.teacher_internal_update_ratio) < 0.0:
        raise ValueError("teacher_internal_update_ratio must be nonnegative")
    if float(args.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not np.isfinite(float(args.latent_excursion_limit)) or float(
        args.latent_excursion_limit
    ) <= 0.0:
        raise ValueError("latent_excursion_limit must be finite and positive")
    if not np.isfinite(float(args.latent_state_bound)) or float(
        args.latent_state_bound
    ) < 0.0:
        raise ValueError("latent_state_bound must be finite and nonnegative")
    if not 0 <= int(args.train_tail_output_from) < int(args.latent_rank):
        raise ValueError("train_tail_output_from must lie in [0, latent_rank)")
    if int(args.conditioner_experts) < 0:
        raise ValueError("conditioner_experts must be nonnegative")
    if args.train_expert_output_only and int(args.conditioner_experts) <= 0:
        raise ValueError("train_expert_output_only requires conditioner_experts")
    if args.train_expert_phase_output_only and int(args.conditioner_experts) <= 0:
        raise ValueError(
            "train_expert_phase_output_only requires conditioner_experts"
        )
    if args.train_expert_gate_only and int(args.conditioner_experts) <= 0:
        raise ValueError("train_expert_gate_only requires conditioner_experts")
    if args.train_expert_cyclic_gate_only and int(args.conditioner_experts) <= 0:
        raise ValueError(
            "train_expert_cyclic_gate_only requires conditioner_experts"
        )
    if args.train_expert_cyclic_output_only and int(args.conditioner_experts) <= 0:
        raise ValueError(
            "train_expert_cyclic_output_only requires conditioner_experts"
        )
    if args.train_cyclic_low_mode_closure_only and int(args.conditioner_experts) <= 0:
        raise ValueError(
            "train_cyclic_low_mode_closure_only requires conditioner_experts"
        )
    if args.train_cyclic_low_mode_pressure_only and int(args.conditioner_experts) <= 0:
        raise ValueError(
            "train_cyclic_low_mode_pressure_only requires conditioner_experts"
        )
    if args.train_cyclic_modal_pressure_only and int(args.conditioner_experts) <= 0:
        raise ValueError(
            "train_cyclic_modal_pressure_only requires conditioner_experts"
        )
    if args.train_cyclic_low_mode_momentum_only and int(args.conditioner_experts) <= 0:
        raise ValueError(
            "train_cyclic_low_mode_momentum_only requires conditioner_experts"
        )
    if sum(
        bool(value)
        for value in (
            args.train_expert_output_only,
            args.train_expert_phase_output_only,
            args.train_expert_gate_only,
            args.train_expert_cyclic_gate_only,
            args.train_expert_cyclic_output_only,
            args.train_cyclic_low_mode_closure_only,
            args.train_cyclic_low_mode_pressure_only,
            args.train_cyclic_modal_pressure_only,
            args.train_cyclic_low_mode_momentum_only,
        )
    ) > 1:
        raise ValueError(
            "expert output, phase-output, gate-only, cyclic-gate-only, and "
            "cyclic-output-only, cyclic-low-mode-closure-only, and "
            "cyclic-low-mode-pressure-only modes are exclusive"
        )
    train_output_rows = tuple(
        int(value)
        for value in str(args.train_output_rows).split(",")
        if value.strip()
    )
    train_cyclic_feature_indices = tuple(
        int(value)
        for value in str(args.train_cyclic_feature_indices).split(",")
        if value.strip()
    )
    if len(set(train_cyclic_feature_indices)) != len(
        train_cyclic_feature_indices
    ) or any(index < 0 or index >= 68 for index in train_cyclic_feature_indices):
        raise ValueError(
            "train_cyclic_feature_indices must be unique indices in [0, 68)"
        )
    if train_cyclic_feature_indices and not (
        args.train_expert_cyclic_gate_only
        or args.train_expert_cyclic_output_only
        or args.train_cyclic_low_mode_closure_only
        or args.train_cyclic_low_mode_pressure_only
        or args.train_cyclic_modal_pressure_only
        or args.train_cyclic_low_mode_momentum_only
    ):
        raise ValueError(
            "train_cyclic_feature_indices requires a cyclic gate/output mode"
        )
    if len(set(train_output_rows)) != len(train_output_rows) or any(
        row < 0 or row >= int(args.latent_rank) for row in train_output_rows
    ):
        raise ValueError("train_output_rows must be unique rows in [0, latent_rank)")
    train_expert_indices = tuple(
        int(value)
        for value in str(args.train_expert_indices).split(",")
        if value.strip()
    )
    if len(set(train_expert_indices)) != len(train_expert_indices) or any(
        index < 0 or index >= int(args.conditioner_experts)
        for index in train_expert_indices
    ):
        raise ValueError(
            "train_expert_indices must be unique indices in [0, conditioner_experts)"
        )
    if not np.isfinite(float(args.linear_nonregression_limit)) or float(
        args.linear_nonregression_limit
    ) < 0.0:
        raise ValueError("linear_nonregression_limit must be finite and nonnegative")
    if float(args.teacher_learning_rate) <= 0.0:
        raise ValueError("teacher_learning_rate must be positive")
    if int(args.teacher_residual_stride) <= 0:
        raise ValueError("teacher_residual_stride must be positive")
    if int(args.teacher_rollout_steps) <= 0:
        raise ValueError("teacher_rollout_steps must be positive")
    if int(args.gradient_accumulation_steps) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(args.steps_per_epoch) % int(args.gradient_accumulation_steps) != 0:
        raise ValueError(
            "steps_per_epoch must be divisible by gradient_accumulation_steps"
        )
    if float(args.parameter_update_clip) <= 0.0:
        raise ValueError("parameter_update_clip must be positive")
    if float(args.hermite_tail_damping) < 0.0:
        raise ValueError("hermite_tail_damping must be nonnegative")
    if float(args.hermite_tail_power) <= 0.0:
        raise ValueError("hermite_tail_power must be positive")
    if not 0.0 <= float(args.semilinear_kick_scale) <= 1.0:
        raise ValueError("semilinear_kick_scale must lie in [0, 1]")
    if int(args.trust_region_backtracks) < 0:
        raise ValueError("trust_region_backtracks must be nonnegative")
    if float(args.trust_region_tolerance) < 0.0:
        raise ValueError("trust_region_tolerance must be nonnegative")
    if int(args.trust_region_batches) <= 0:
        raise ValueError("trust_region_batches must be positive")
    if not 0.0 < float(args.correction_energy_tolerance) < 1.0:
        raise ValueError("correction_energy_tolerance must lie in (0, 1)")
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.outdir}")
    args.outdir.mkdir(parents=True)
    metadata = json.loads((args.reference_cache / "metadata.json").read_text())
    configuration = metadata["configuration"]
    manifest = load_ic_manifest(args.reference_cache / "ic_manifest.json")
    cases = [dict(case) for case in manifest["cases"]]
    supplemental_case_ids = tuple(
        value.strip()
        for value in str(args.supplemental_strong_case_ids).split(",")
        if value.strip()
    )
    if supplemental_case_ids:
        if args.supplemental_strong_manifest is None:
            raise ValueError(
                "supplemental_strong_manifest is required when supplemental "
                "strong case IDs are selected"
            )
        supplemental = load_ic_manifest(args.supplemental_strong_manifest)
        by_id = {
            str(case["case_id"]): dict(case)
            for case in supplemental["cases"]
            if str(case["regime"]) == "nonlinear_landau_strong"
            and str(case["split"]) == "train"
        }
        if (
            len(set(supplemental_case_ids)) != len(supplemental_case_ids)
            or not set(supplemental_case_ids).issubset(by_id)
        ):
            raise ValueError(
                "supplemental strong case IDs must uniquely select train-only "
                "strong cases"
            )
        prefix = str(args.supplemental_strong_case_prefix)
        existing_ids = {str(case["case_id"]) for case in cases}
        for case_id in supplemental_case_ids:
            case = by_id[case_id]
            renamed = f"{prefix}{case_id}"
            if renamed in existing_ids:
                raise ValueError(f"supplemental case ID collision: {renamed}")
            case["case_id"] = renamed
            cases.append(case)
            existing_ids.add(renamed)
    basis, resolved_center, resolved_scale, latent_center, latent_scale = (
        _load_or_build_projected_cache(
            args.reference_cache,
            args.projected_cache,
            cases,
            configuration,
            args,
        )
    )
    cache_metadata = json.loads((args.projected_cache / "metadata.json").read_text())
    expected_samples = int(cache_metadata["time_samples"])
    horizon = expected_samples - 1
    if (
        float(args.electric_sliding_log_growth_weight) > 0.0
        and 2 * int(args.electric_growth_window_steps) > horizon + 1
    ):
        raise ValueError(
            "twice electric_growth_window_steps cannot exceed the projected-cache "
            "horizon + 1 when electric_sliding_log_growth_weight is enabled"
        )
    expected_horizon = int(round(float(configuration["T_final"]) / args.cadence))
    if horizon != expected_horizon:
        raise ValueError(
            f"Projected cache covers H={horizon}; complete T={configuration['T_final']} "
            f"requires H={expected_horizon}"
        )
    teacher_dt = float(configuration["teacher_dt"])
    if not np.isclose(int(args.fine_steps) * teacher_dt, float(args.cadence)):
        raise ValueError("fine_steps * teacher_dt must equal latent cadence")
    if int(args.nx) != int(cache_metadata["nx"]):
        raise ValueError("nx must match the projected latent cache")
    domain_length = float(configuration["teacher_L"])
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(
        int(args.nx), d=domain_length / int(args.nx)
    ).astype(np.float32)
    poisson_sign = float(configuration["teacher_poisson_sign"])
    if int(args.latent_rank) != int(cache_metadata["rank"]):
        raise ValueError("latent rank must match the projected latent cache")
    if int(args.gradient_chunk_steps) <= 0:
        raise ValueError("gradient_chunk_steps must be positive")
    if horizon % int(args.gradient_chunk_steps) != 0:
        raise ValueError(
            f"complete horizon H={horizon} must be divisible by "
            f"gradient_chunk_steps={args.gradient_chunk_steps}"
        )
    train_by_regime = {
        regime: _selected_cases(cases, regime, "train") for regime in REGIMES
    }
    weak_train_case_ids = tuple(
        value.strip()
        for value in str(args.weak_train_case_ids).split(",")
        if value.strip()
    )
    if weak_train_case_ids:
        weak_regime = "nonlinear_landau_weak"
        available = {case["case_id"] for case in train_by_regime[weak_regime]}
        if len(set(weak_train_case_ids)) != len(weak_train_case_ids) or not set(
            weak_train_case_ids
        ).issubset(available):
            raise ValueError(
                "weak_train_case_ids must uniquely select train-only weak cases"
            )
        selected_ids = set(weak_train_case_ids)
        train_by_regime[weak_regime] = [
            case
            for case in train_by_regime[weak_regime]
            if case["case_id"] in selected_ids
        ]
    strong_train_case_ids = tuple(
        value.strip()
        for value in str(args.strong_train_case_ids).split(",")
        if value.strip()
    )
    if strong_train_case_ids:
        strong_regime = "nonlinear_landau_strong"
        available = {case["case_id"] for case in train_by_regime[strong_regime]}
        if len(set(strong_train_case_ids)) != len(strong_train_case_ids) or not set(
            strong_train_case_ids
        ).issubset(available):
            raise ValueError(
                "strong_train_case_ids must uniquely select train-only strong cases"
            )
        selected_ids = set(strong_train_case_ids)
        train_by_regime[strong_regime] = [
            case
            for case in train_by_regime[strong_regime]
            if case["case_id"] in selected_ids
        ]
    minimum_cases = min(len(rows) for rows in train_by_regime.values())
    if int(args.steps_per_epoch) > minimum_cases:
        raise ValueError(
            "steps_per_epoch may not exceed the number of independent training "
            f"cases per regime ({minimum_cases})"
        )
    propagator, propagator_diagnostics = _fit_or_load_coupled_linear_propagator(
        args.projected_cache,
        cases,
        resolved_scale=resolved_scale,
        latent_scale=latent_scale,
        latent_rank=int(args.latent_rank),
        nx=int(args.nx),
        ridge=float(args.linear_ridge),
        maximum_spectral_radius=float(args.linear_max_spectral_radius),
    )
    model_propagator = (
        _matrix_square_root_propagator(propagator)
        if args.linear_baseline == "semilinear_strang"
        else propagator
    )
    correction_bounds, correction_bound_diagnostics = (
        _fit_or_load_correction_bounds(
            args.projected_cache,
            cases,
            propagator,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
            linear_ridge=float(args.linear_ridge),
            maximum_spectral_radius=float(args.linear_max_spectral_radius),
            energy_tolerance=float(args.correction_energy_tolerance),
            linear_baseline=str(args.linear_baseline),
            basis=basis,
            k_arr=k_arr,
            fine_steps=int(args.fine_steps),
            fine_dt=teacher_dt,
            hermite_tail_damping=float(args.hermite_tail_damping),
            hermite_tail_power=float(args.hermite_tail_power),
            semilinear_kick_scale=float(args.semilinear_kick_scale),
            semilinear_correction_location=str(
                args.semilinear_correction_location
            ),
            poisson_sign=poisson_sign,
        )
    )
    maximum_bound_distortion = float(
        np.max(correction_bound_diagnostics["channel_distortion_fraction"])
    )
    print(
        "[data] nonlinear correction bounds: "
        f"minimum={np.min(correction_bounds):.6e} "
        f"median={np.median(correction_bounds):.6e} "
        f"maximum={np.max(correction_bounds):.6e} "
        f"overall_distortion="
        f"{correction_bound_diagnostics['overall_distortion_fraction']:.6e} "
        f"maximum_channel_distortion={maximum_bound_distortion:.6e}",
        flush=True,
    )
    if maximum_bound_distortion > float(args.correction_energy_tolerance) * 1.01:
        raise FloatingPointError(
            "Calibrated correction bounds exceed the distortion tolerance"
        )
    latent_residual_scale = _fit_or_load_latent_residual_scale(
        args.projected_cache,
        cases,
        propagator,
        resolved_scale=resolved_scale,
        latent_scale=latent_scale,
        linear_ridge=float(args.linear_ridge),
        maximum_spectral_radius=float(args.linear_max_spectral_radius),
        linear_baseline=str(args.linear_baseline),
        basis=basis,
        k_arr=k_arr,
        fine_steps=int(args.fine_steps),
        fine_dt=teacher_dt,
        hermite_tail_damping=float(args.hermite_tail_damping),
        hermite_tail_power=float(args.hermite_tail_power),
        semilinear_kick_scale=float(args.semilinear_kick_scale),
        poisson_sign=poisson_sign,
    )
    print(
        "[data] nonlinear residual RMS scale: "
        f"minimum={np.min(latent_residual_scale):.6e} "
        f"median={np.median(latent_residual_scale):.6e} "
        f"maximum={np.max(latent_residual_scale):.6e}",
        flush=True,
    )
    closure_residual_scale = _fit_or_load_closure_residual_scale(
        args.projected_cache,
        cases,
        propagator,
        basis,
        resolved_scale=resolved_scale,
        latent_scale=latent_scale,
        linear_ridge=float(args.linear_ridge),
        maximum_spectral_radius=float(args.linear_max_spectral_radius),
        linear_baseline=str(args.linear_baseline),
        k_arr=k_arr,
        fine_steps=int(args.fine_steps),
        fine_dt=teacher_dt,
        hermite_tail_damping=float(args.hermite_tail_damping),
        hermite_tail_power=float(args.hermite_tail_power),
        semilinear_kick_scale=float(args.semilinear_kick_scale),
        poisson_sign=poisson_sign,
    )
    print(
        "[data] closure-visible C3 residual RMS scale: "
        f"{closure_residual_scale:.6e}",
        flush=True,
    )
    operator_modes = int(args.operator_modes) or (int(args.nx) // 2 + 1)
    params = init_state_conditioned_latent_operator(
        jax.random.PRNGKey(int(args.seed)),
        resolved_channels=3,
        latent_rank=int(args.latent_rank),
        width=int(args.width),
        depth=int(args.depth),
        spectral_modes=operator_modes,
        operator_rank=int(args.operator_rank),
        kernel_size=int(args.kernel_size),
        output_projection_init_scale=float(args.operator_output_init_scale),
        conditioner_output_channels=(
            int(args.latent_rank)
            if args.latent_readout_mode == "equilibrium_cnn"
            else None
        ),
        conditioner_output_init_scale=(
            float(args.operator_output_init_scale)
            if args.latent_readout_mode == "equilibrium_cnn"
            else 1e-2
        ),
        closure_aligned_output=bool(args.closure_aligned_output),
        latent_delay_input=bool(args.latent_delay_input),
        conditioner_experts=int(args.conditioner_experts),
    )
    multiplicative_readout_diagnostics = None
    if args.fit_multiplicative_output_readout and args.resume_training_state is None:
        params, multiplicative_readout_diagnostics = (
            _fit_multiplicative_endpoint_readout(
                params,
                args.projected_cache,
                cases,
                propagator,
                basis,
                k_arr,
                resolved_scale=resolved_scale,
                latent_scale=latent_scale,
                correction_bounds=correction_bounds,
                depth=int(args.depth),
                fine_steps=int(args.fine_steps),
                fine_dt=teacher_dt,
                kick_scale=float(args.semilinear_kick_scale),
                poisson_sign=poisson_sign,
                linear_baseline=str(args.linear_baseline),
                hermite_tail_damping=float(args.hermite_tail_damping),
                hermite_tail_power=float(args.hermite_tail_power),
                latent_delay_input=bool(args.latent_delay_input),
                ridge=float(args.multiplicative_output_ridge),
            )
        )
        readout_scale = float(args.multiplicative_output_initial_scale)
        params["operator_u_real"] = readout_scale * params["operator_u_real"]
        params["operator_u_imag"] = readout_scale * params["operator_u_imag"]
        print(
            "[data] multiplicative endpoint readout: "
            f"train_relative_mse="
            f"{multiplicative_readout_diagnostics['train_relative_mse']:.6e} "
            f"regime={multiplicative_readout_diagnostics['train_regime_relative_mse']} "
            f"weight_rms={multiplicative_readout_diagnostics['weight_rms']:.6e} "
            f"weight_maximum="
            f"{multiplicative_readout_diagnostics['weight_maximum']:.6e} "
            f"initial_scale={readout_scale:.6e}",
            flush=True,
        )
    closure_readout_diagnostics = None
    if args.closure_readout_mode != "multiplicative":
        readout, closure_readout_diagnostics = _fit_or_load_gated_closure_readout(
            args.projected_cache,
            cases,
            propagator,
            basis,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
            closure_residual_scale=closure_residual_scale,
            linear_ridge=float(args.linear_ridge),
            maximum_spectral_radius=float(args.linear_max_spectral_radius),
            gate_scale=float(args.closure_gate_scale),
            gate_power=int(args.closure_gate_power),
            readout_ridge=float(args.closure_readout_ridge),
            trajectory_exponent=float(args.closure_readout_trajectory_exponent),
            linear_baseline=str(args.linear_baseline),
            k_arr=k_arr,
            fine_steps=int(args.fine_steps),
            fine_dt=teacher_dt,
            hermite_tail_damping=float(args.hermite_tail_damping),
            hermite_tail_power=float(args.hermite_tail_power),
            poisson_sign=poisson_sign,
        )
        readout = float(args.closure_readout_initial_scale) * readout
        params["operator_u_gated_closure_real"] = jnp.asarray(readout.real)
        params["operator_u_gated_closure_imag"] = jnp.asarray(readout.imag)
        print(
            "[data] gated closure readout: "
            f"train_relative_mse={closure_readout_diagnostics['train_relative_mse']:.6e} "
            f"weight_rms={closure_readout_diagnostics['weight_rms']:.6e} "
            f"weight_maximum={closure_readout_diagnostics['weight_maximum']:.6e}",
            flush=True,
        )
    latent_readout_diagnostics = None
    if args.latent_readout_mode in {"gated_linear", "gated_linear_residual"}:
        readout, latent_readout_diagnostics = _fit_or_load_gated_latent_readout(
            args.projected_cache,
            cases,
            propagator,
            resolved_scale=resolved_scale,
            latent_scale=latent_scale,
            linear_ridge=float(args.linear_ridge),
            maximum_spectral_radius=float(args.linear_max_spectral_radius),
            gate_scale=float(args.latent_gate_scale),
            gate_power=int(args.latent_gate_power),
            readout_ridge=float(args.latent_readout_ridge),
            trajectory_exponent=float(args.latent_readout_trajectory_exponent),
            linear_baseline=str(args.linear_baseline),
            basis=basis,
            k_arr=k_arr,
            fine_steps=int(args.fine_steps),
            fine_dt=teacher_dt,
            hermite_tail_damping=float(args.hermite_tail_damping),
            hermite_tail_power=float(args.hermite_tail_power),
            poisson_sign=poisson_sign,
        )
        readout = float(args.latent_readout_initial_scale) * readout
        params["operator_u_gated_latent_real"] = jnp.asarray(readout.real)
        params["operator_u_gated_latent_imag"] = jnp.asarray(readout.imag)
        print(
            "[data] gated latent readout: "
            f"train_relative_mse={latent_readout_diagnostics['train_relative_mse']:.6e} "
            f"weight_rms={latent_readout_diagnostics['weight_rms']:.6e} "
            f"weight_maximum={latent_readout_diagnostics['weight_maximum']:.6e}",
            flush=True,
        )
    optimizers = {
        "physical": _adam_init(params),
        "teacher": _adam_init(params),
    }
    completed_epoch = 0
    resumed_history = []
    if args.init_checkpoint is not None and args.resume_training_state is not None:
        raise ValueError(
            "--init-checkpoint and --resume-training-state cannot be combined"
        )
    if args.init_checkpoint is not None:
        with np.load(args.init_checkpoint, allow_pickle=False) as checkpoint:
            missing = [
                key
                for key in params
                if key not in checkpoint and not key.startswith("expert_")
            ]
            if missing:
                raise ValueError(
                    "Initialization checkpoint is missing model parameters: "
                    + ", ".join(missing)
                )
            incompatible = [
                key
                for key, value in params.items()
                if key in checkpoint and checkpoint[key].shape != value.shape
            ]
            if incompatible:
                details = ", ".join(
                    f"{key}: checkpoint{checkpoint[key].shape} != model{params[key].shape}"
                    for key in incompatible
                )
                raise ValueError(
                    "Initialization checkpoint is incompatible with the requested "
                    f"model configuration ({details})"
                )
            params = {
                key: (
                    jnp.asarray(checkpoint[key], dtype=value.dtype)
                    if key in checkpoint
                    else value
                )
                for key, value in params.items()
            }
            # Presence of a null logit changes the softmax denominator even at
            # its zero initializer.  Legacy expert checkpoints without this
            # optional leaf must retain their exact routing semantics.
            if "expert_null_gate_bias" not in checkpoint:
                params.pop("expert_null_gate_bias", None)
            for key in checkpoint.files:
                if key.startswith("specialist_") or key == (
                    "expert_history_frequency_deadband"
                ):
                    params[key] = jnp.asarray(checkpoint[key])
        # Optional checkpoint-defined router state changes the parameter tree,
        # so its fresh optimizer state must be built after those leaves load.
        optimizers = {
            "physical": _adam_init(params),
            "teacher": _adam_init(params),
        }
        print(
            f"[train] initialized parameters from {args.init_checkpoint}; "
            "optimizer and epoch count are fresh",
            flush=True,
        )
    if args.resume_training_state is not None:
        params, optimizers, completed_epoch = _load_training_state(
            args.resume_training_state, params, optimizers
        )
        source_report = args.resume_training_state.with_name("report.json")
        if source_report.exists():
            resumed_history = json.loads(source_report.read_text()).get("history", [])
        if int(args.epochs) <= completed_epoch:
            raise ValueError(
                f"epochs={args.epochs} must exceed resumed epoch {completed_epoch}"
            )
        print(
            f"[train] resumed epoch {completed_epoch} and optimizer state from "
            f"{args.resume_training_state}",
            flush=True,
        )
    (
        loss_function,
        rollout_function,
        loss_components_function,
        teacher_residual_loss_function,
    ) = _make_functions(
        propagator=model_propagator,
        basis=basis,
        k_arr=k_arr,
        resolved_center=resolved_center,
        resolved_scale=resolved_scale,
        latent_center=latent_center,
        latent_scale=latent_scale,
        depth=int(args.depth),
        fine_steps=int(args.fine_steps),
        fine_dt=teacher_dt,
        horizon=horizon,
        poisson_sign=poisson_sign,
        gradient_chunk_steps=int(args.gradient_chunk_steps),
        correction_bounds=correction_bounds,
        latent_residual_scale=latent_residual_scale,
        latent_residual_weight=float(args.latent_residual_weight),
        latent_state_residual_weight=float(args.latent_state_residual_weight),
        low_mode_latent_residual_weight=float(
            args.low_mode_latent_residual_weight
        ),
        closure_residual_scale=closure_residual_scale,
        closure_residual_weight=float(args.closure_residual_weight),
        closure_correction_bound=float(args.closure_correction_bound),
        closure_aligned_output=bool(args.closure_aligned_output),
        closure_only_correction=bool(args.closure_only_correction),
        closure_readout_mode=str(args.closure_readout_mode),
        closure_gate_scale=float(args.closure_gate_scale),
        closure_gate_power=int(args.closure_gate_power),
        latent_readout_mode=str(args.latent_readout_mode),
        latent_gate_scale=float(args.latent_gate_scale),
        latent_gate_power=int(args.latent_gate_power),
        latent_state_bound=float(args.latent_state_bound),
        equilibrium_input_compression_scale=float(
            args.equilibrium_input_compression_scale
        ),
        autonomous_latent_weight=float(args.autonomous_latent_weight),
        electric_spectrum_weight=float(args.electric_spectrum_weight),
        electric_log_energy_weight=float(args.electric_log_energy_weight),
        electric_log_energy_floor_ratio=float(args.electric_log_energy_floor_ratio),
        electric_chunk_log_growth_weight=float(
            args.electric_chunk_log_growth_weight
        ),
        electric_sliding_log_growth_weight=float(
            args.electric_sliding_log_growth_weight
        ),
        electric_sliding_log_growth_direction=str(
            args.electric_sliding_log_growth_direction
        ),
        electric_growth_window_steps=int(args.electric_growth_window_steps),
        electric_time_relative_weight=float(args.electric_time_relative_weight),
        electric_transfer_weight=float(args.electric_transfer_weight),
        electric_time_relative_floor_ratio=float(
            args.electric_time_relative_floor_ratio
        ),
        post_minimum_log_energy_weight=float(args.post_minimum_log_energy_weight),
        post_minimum_field_weight=float(args.post_minimum_field_weight),
        turnaround_field_weight=float(args.turnaround_field_weight),
        post_peak_log_energy_weight=float(args.post_peak_log_energy_weight),
        peak_log_energy_weight=float(args.peak_log_energy_weight),
        turnaround_log_ratio_weight=float(args.turnaround_log_ratio_weight),
        post_peak_retention_weight=float(args.post_peak_retention_weight),
        post_peak_retention_floor=float(args.post_peak_retention_floor),
        post_minimum_energy_modes=int(args.post_minimum_energy_modes),
        teacher_residual_stride=int(args.teacher_residual_stride),
        teacher_rollout_steps=int(args.teacher_rollout_steps),
        linear_baseline=str(args.linear_baseline),
        hermite_tail_damping=float(args.hermite_tail_damping),
        hermite_tail_power=float(args.hermite_tail_power),
        semilinear_kick_scale=float(args.semilinear_kick_scale),
        semilinear_correction_location=str(args.semilinear_correction_location),
        latent_delay_input=bool(args.latent_delay_input),
        return_teacher_residual_function=True,
        case_loss_weights=jnp.asarray(
            (
                float(args.linear_regime_weight),
                float(args.weak_regime_weight),
                float(args.strong_regime_weight),
            )
        ),
    )
    rollout = jax.jit(rollout_function)
    @jax.jit
    def correction_diagnostic(
        model_params,
        normalized_resolved,
        normalized_latent,
        normalized_latent_delta,
    ):
        if args.closure_only_correction:
            return jnp.zeros_like(normalized_latent)
        if args.latent_readout_mode == "equilibrium_cnn":
            return equilibrium_preserving_latent_cnn_correction(
                model_params,
                normalized_resolved,
                normalized_latent,
                depth=int(args.depth),
                normalized_latent_delta=(
                    normalized_latent_delta if args.latent_delay_input else None
                ),
                input_compression_scale=float(
                    args.equilibrium_input_compression_scale
                ),
            )
        if args.latent_readout_mode == "multiplicative":
            return state_conditioned_latent_operator_correction(
                model_params,
                normalized_resolved,
                normalized_latent,
                depth=int(args.depth),
                equilibrium_preserving=True,
                normalized_latent_delta=(
                    normalized_latent_delta if args.latent_delay_input else None
                ),
            )
        correction = gated_spectral_latent_correction(
            model_params,
            normalized_resolved,
            normalized_latent,
            gate_scale=float(args.latent_gate_scale),
            gate_power=int(args.latent_gate_power),
        )
        if args.latent_readout_mode == "gated_linear_residual":
            correction = correction + state_conditioned_latent_operator_correction(
                model_params,
                normalized_resolved,
                normalized_latent,
                depth=int(args.depth),
                equilibrium_preserving=True,
            )
        return correction

    closure_diagnostic = None
    if args.closure_aligned_output:
        @jax.jit
        def closure_diagnostic(
            model_params,
            normalized_resolved,
            normalized_latent,
            normalized_latent_delta,
        ):
            if args.closure_readout_mode == "multiplicative":
                raw_correction = state_conditioned_closure_correction(
                    model_params,
                    normalized_resolved,
                    normalized_latent,
                    depth=int(args.depth),
                    equilibrium_preserving=True,
                    normalized_latent_delta=(
                        normalized_latent_delta
                        if args.latent_delay_input
                        else None
                    ),
                )
            else:
                raw_correction = gated_spectral_closure_correction(
                    model_params,
                    normalized_resolved,
                    normalized_latent,
                    gate_scale=float(args.closure_gate_scale),
                    gate_power=int(args.closure_gate_power),
                )
                if args.closure_readout_mode == "gated_linear_residual":
                    raw_correction = raw_correction + state_conditioned_closure_correction(
                        model_params,
                        normalized_resolved,
                        normalized_latent,
                        depth=int(args.depth),
                        equilibrium_preserving=True,
                    )
            return bounded_normalized_closure_correction(
                raw_correction,
                float(args.closure_correction_bound),
            )

    if args.training_objective == "teacher_residual_probe":

        @jax.jit
        def loss_and_gradients(model_params, trajectory):
            teacher_residual_loss, teacher_residual_grads = jax.value_and_grad(
                teacher_residual_loss_function
            )(model_params, trajectory)
            physical_grads = jax.tree_util.tree_map(
                jnp.zeros_like, teacher_residual_grads
            )
            finite = jnp.isfinite(teacher_residual_loss) & _tree_all_finite(
                teacher_residual_grads
            )
            return (
                teacher_residual_loss,
                jnp.zeros_like(teacher_residual_loss),
                teacher_residual_loss,
                physical_grads,
                teacher_residual_grads,
                finite,
            )

    else:

        @jax.jit
        def loss_and_gradients(model_params, trajectory):
            components, pullback = jax.vjp(
                lambda candidate: loss_components_function(candidate, trajectory),
                model_params,
            )
            loss_monitor, physical_loss, teacher_residual_loss = components
            zeros = jnp.zeros_like(loss_monitor)
            ones = jnp.ones_like(loss_monitor)
            physical_grads = pullback((zeros, ones, zeros))[0]
            teacher_residual_grads = pullback((zeros, zeros, ones))[0]
            finite = (
                jnp.isfinite(loss_monitor)
                & _tree_all_finite(physical_grads)
                & _tree_all_finite(teacher_residual_grads)
            )
            return (
                loss_monitor,
                physical_loss,
                teacher_residual_loss,
                physical_grads,
                teacher_residual_grads,
                finite,
            )

    @jax.jit
    def apply_gradients(
        model_params,
        optimizer_states,
        physical_grads,
        teacher_residual_grads,
    ):
        (
            grads,
            physical_grad_norm,
            teacher_residual_grad_norm,
            gradient_cosine,
            auxiliary_contribution_norm,
            gradient_conflict,
            physical_gradient_alignment,
        ) = _combine_gradients(
            physical_grads,
            teacher_residual_grads,
            auxiliary_weight=float(args.latent_residual_weight),
            maximum_auxiliary_ratio=float(args.latent_gradient_ratio),
            mode=args.update_combination,
        )
        finite = (
            _tree_all_finite(physical_grads)
            & _tree_all_finite(teacher_residual_grads)
            & _tree_all_finite(grads)
            & jnp.isfinite(gradient_cosine)
            & jnp.isfinite(physical_gradient_alignment)
        )
        group_norms = _gradient_group_norms(grads)
        safe_grads = jax.tree_util.tree_map(
            lambda value: jnp.where(finite, value, jnp.zeros_like(value)),
            grads,
        )
        safe_physical_grads = jax.tree_util.tree_map(
            lambda value: jnp.where(finite, value, jnp.zeros_like(value)),
            physical_grads,
        )
        safe_teacher_grads = jax.tree_util.tree_map(
            lambda value: jnp.where(finite, value, jnp.zeros_like(value)),
            teacher_residual_grads,
        )
        optimizer_step = _adam_step if args.optimizer == "adam" else _sgd_step
        if args.training_objective == "joint":
            # Preserve the physical/teacher gradient geometry through Adam.
            # Combining separately preconditioned Adam displacements can reverse
            # the positive raw-gradient alignment established above.
            updated, joint_optimizer, _ = optimizer_step(
                model_params,
                safe_grads,
                optimizer_states["physical"],
                float(args.learning_rate),
                float(args.grad_clip),
            )
            physical_update = jax.tree_util.tree_map(
                lambda new, old: new - old, updated, model_params
            )
            physical_update_norm = _tree_l2_norm(physical_update)
            teacher_update_norm = jnp.asarray(0.0, dtype=physical_update_norm.dtype)
            update_cosine = jnp.asarray(0.0, dtype=physical_update_norm.dtype)
            teacher_update_contribution_norm = jnp.asarray(
                0.0, dtype=physical_update_norm.dtype
            )
            update_conflict = jnp.asarray(False)
            physical_update_alignment = jnp.asarray(
                1.0, dtype=physical_update_norm.dtype
            )
            updated_optimizers = {
                "physical": joint_optimizer,
                "teacher": optimizer_states["teacher"],
            }
        else:
            teacher_candidate, teacher_optimizer, _ = optimizer_step(
                model_params,
                safe_teacher_grads,
                optimizer_states["teacher"],
                float(args.teacher_learning_rate),
                float(args.grad_clip),
            )
            teacher_update = _scale_auxiliary_update(
                teacher_candidate,
                model_params,
                float(args.latent_residual_weight),
            )
            updated = jax.tree_util.tree_map(
                lambda old, change: old + change,
                model_params,
                teacher_update,
            )
            teacher_update_norm = _tree_l2_norm(teacher_update)
            physical_update_norm = jnp.asarray(0.0, dtype=teacher_update_norm.dtype)
            update_cosine = jnp.asarray(0.0, dtype=teacher_update_norm.dtype)
            teacher_update_contribution_norm = teacher_update_norm
            update_conflict = jnp.asarray(False)
            physical_update_alignment = jnp.asarray(
                0.0, dtype=teacher_update_norm.dtype
            )
            updated_optimizers = {
                "physical": optimizer_states["physical"],
                "teacher": teacher_optimizer,
            }
        grad_norm = _tree_l2_norm(grads)
        updated, update_norm, clipped_update_norm = _clip_parameter_update(
            model_params,
            updated,
            float(args.parameter_update_clip),
        )
        update_finite = _tree_all_finite(updated)
        finite = finite & update_finite
        updated = jax.tree_util.tree_map(
            lambda new, old: new.astype(old.dtype), updated, model_params
        )
        updated = jax.tree_util.tree_map(
            lambda new, old: jnp.where(finite, new, old), updated, model_params
        )
        updated_optimizers = jax.tree_util.tree_map(
            lambda new, old: jnp.where(finite, new, old),
            updated_optimizers,
            optimizer_states,
        )
        return (
            updated,
            updated_optimizers,
            grad_norm,
            group_norms,
            update_norm,
            clipped_update_norm,
            physical_grad_norm,
            teacher_residual_grad_norm,
            gradient_cosine,
            auxiliary_contribution_norm,
            gradient_conflict,
            physical_gradient_alignment,
            physical_update_norm,
            teacher_update_norm,
            update_cosine,
            teacher_update_contribution_norm,
            update_conflict,
            physical_update_alignment,
            finite,
        )

    rng = np.random.default_rng(int(args.seed))
    initial_orders = {
        regime: rng.permutation(len(train_by_regime[regime])) for regime in REGIMES
    }
    preflight, _ = _balanced_complete_batch(
        args.projected_cache,
        train_by_regime,
        initial_orders,
        0,
        latent_rank=int(args.latent_rank),
        expected_samples=expected_samples,
    )
    print(
        f"[model] autonomous_state=3+{args.latent_rank} nx={args.nx} "
        f"parameters={sum(value.size for value in jax.tree_util.tree_leaves(params))} "
        f"H={horizon} span={horizon * args.cadence:.3f} "
        f"gradient_chunk={args.gradient_chunk_steps} fine_steps={args.fine_steps} "
        f"linear_baseline={args.linear_baseline} "
        f"semilinear_kick_scale={args.semilinear_kick_scale:.3f} "
        f"semilinear_correction_location={args.semilinear_correction_location} "
        f"latent_delay_input={int(args.latent_delay_input)} "
        f"hermite_tail=({args.hermite_tail_damping:.3e},"
        f"{args.hermite_tail_power:.3f}) "
        f"gradient_accumulation={args.gradient_accumulation_steps} "
        f"teacher_residual_weight={args.latent_residual_weight:.3e} "
        f"latent_state_residual_weight={args.latent_state_residual_weight:.3e} "
        f"low_mode_latent_residual_weight="
        f"{args.low_mode_latent_residual_weight:.3e} "
        f"closure_residual_weight={args.closure_residual_weight:.3e} "
        f"closure_correction_bound={args.closure_correction_bound:.3e} "
        f"closure_aligned_output={int(args.closure_aligned_output)} "
        f"closure_only_correction={int(args.closure_only_correction)} "
        f"autonomous_latent_weight={args.autonomous_latent_weight:.3e} "
        f"electric_spectrum_weight={args.electric_spectrum_weight:.3e} "
        f"electric_log_energy_weight={args.electric_log_energy_weight:.3e} "
        f"electric_log_energy_floor_ratio={args.electric_log_energy_floor_ratio:.3e} "
        f"electric_chunk_log_growth_weight="
        f"{args.electric_chunk_log_growth_weight:.3e} "
        f"electric_sliding_log_growth_weight="
        f"{args.electric_sliding_log_growth_weight:.3e} "
        f"electric_sliding_log_growth_direction="
        f"{args.electric_sliding_log_growth_direction} "
        f"electric_growth_window_steps={args.electric_growth_window_steps} "
        f"electric_time_relative_weight={args.electric_time_relative_weight:.3e} "
        f"electric_transfer_weight={args.electric_transfer_weight:.3e} "
        f"post_minimum_log_energy_weight={args.post_minimum_log_energy_weight:.3e} "
        f"post_minimum_field_weight={args.post_minimum_field_weight:.3e} "
        f"turnaround_field_weight={args.turnaround_field_weight:.3e} "
        f"post_peak_log_energy_weight={args.post_peak_log_energy_weight:.3e} "
        f"peak_log_energy_weight={args.peak_log_energy_weight:.3e} "
        f"post_minimum_energy_modes={args.post_minimum_energy_modes:d} "
        f"electric_time_relative_floor_ratio="
        f"{args.electric_time_relative_floor_ratio:.3e} "
        f"teacher_latent_stride={args.teacher_residual_stride} "
        f"teacher_latent_rollout_steps={args.teacher_rollout_steps} "
        f"physical_lr={args.learning_rate:.3e} "
        f"teacher_lr={args.teacher_learning_rate:.3e} "
        f"update_combination={args.update_combination} "
        f"teacher_update_ratios=(output={args.latent_gradient_ratio:.3e},"
        f"internal={args.teacher_internal_update_ratio:.3e}) "
        f"update_clip={args.parameter_update_clip:.3e} "
        f"trust_region=(backtracks={args.trust_region_backtracks},"
        f"tolerance={args.trust_region_tolerance:.3e})",
        flush=True,
    )
    maximum_radius = max(
        row["final_spectral_radius"] for row in propagator_diagnostics
    )
    print(
        f"[preflight] coupled_linear_maximum_spectral_radius={maximum_radius:.9f}",
        flush=True,
    )
    if maximum_radius > float(args.linear_max_spectral_radius) + 2e-5:
        raise FloatingPointError("Coupled linear propagator violates stability bound")
    zero_correction_keys = (
        {"output_kernel", "output_bias"}
        if args.latent_readout_mode == "equilibrium_cnn"
        else set()
    )
    zero_correction_params = {
        key: (
            jnp.zeros_like(value)
            if key.startswith("operator_u_") or key in zero_correction_keys
            else value
        )
        for key, value in params.items()
    }
    zero_correction_evaluation = _evaluate(
        zero_correction_params,
        rollout,
        args.projected_cache,
        cases,
        latent_rank=int(args.latent_rank),
        expected_samples=expected_samples,
        resolved_scale=resolved_scale,
        latent_scale=latent_scale,
        k_arr=k_arr,
        poisson_sign=poisson_sign,
        correction_function=correction_diagnostic,
        closure_function=closure_diagnostic,
        correction_bounds=correction_bounds,
        propagator=propagator,
        latent_residual_scale=latent_residual_scale,
        basis=basis,
        closure_residual_scale=closure_residual_scale,
        cadence=float(args.cadence),
        linear_baseline=str(args.linear_baseline),
        fine_steps=int(args.fine_steps),
        fine_dt=teacher_dt,
        hermite_tail_damping=float(args.hermite_tail_damping),
        hermite_tail_power=float(args.hermite_tail_power),
        semilinear_kick_scale=float(args.semilinear_kick_scale),
        semilinear_correction_location=str(args.semilinear_correction_location),
        latent_delay_input=bool(args.latent_delay_input),
    )
    initial_evaluation = _evaluate(
        params,
        rollout,
        args.projected_cache,
        cases,
        latent_rank=int(args.latent_rank),
        expected_samples=expected_samples,
        resolved_scale=resolved_scale,
        latent_scale=latent_scale,
        k_arr=k_arr,
        poisson_sign=poisson_sign,
        correction_function=correction_diagnostic,
        closure_function=closure_diagnostic,
        correction_bounds=correction_bounds,
        propagator=propagator,
        latent_residual_scale=latent_residual_scale,
        basis=basis,
        closure_residual_scale=closure_residual_scale,
        cadence=float(args.cadence),
        linear_baseline=str(args.linear_baseline),
        fine_steps=int(args.fine_steps),
        fine_dt=teacher_dt,
        hermite_tail_damping=float(args.hermite_tail_damping),
        hermite_tail_power=float(args.hermite_tail_power),
        semilinear_kick_scale=float(args.semilinear_kick_scale),
        semilinear_correction_location=str(args.semilinear_correction_location),
        latent_delay_input=bool(args.latent_delay_input),
    )
    print(
        "[preflight] zero_correction_heldout="
        + ",".join(
            f"{regime.split('_')[-1]}:(phys={values['physical_relative_loss']:.3e},"
            f"E={values['electric_field_relative_mse']:.3e},"
            f"regrow={values['predicted_regrowth_factor']:.2e}/"
            f"{values['teacher_regrowth_factor']:.2e},"
            f"zmax={values['maximum_normalized_latent']:.3e})"
            for regime, values in zero_correction_evaluation.items()
        ),
        flush=True,
    )
    print(
        "[preflight] initialized_correction_heldout="
        + ",".join(
            f"{regime.split('_')[-1]}:(phys={values['physical_relative_loss']:.3e},"
            f"E={values['electric_field_relative_mse']:.3e},"
            f"regrow={values['predicted_regrowth_factor']:.2e}/"
            f"{values['teacher_regrowth_factor']:.2e},"
            f"zmax={values['maximum_normalized_latent']:.3e})"
            for regime, values in initial_evaluation.items()
        ),
        flush=True,
    )
    if not all(values["finite"] for values in initial_evaluation.values()):
        raise FloatingPointError("Non-finite initialized held-out rollout")
    initial_latent_maximum = max(
        values["maximum_normalized_latent"]
        for values in initial_evaluation.values()
    )
    excursion_tolerance = max(
        1.0e-6,
        1.0e-6 * float(args.latent_excursion_limit),
    )
    if initial_latent_maximum > (
        float(args.latent_excursion_limit) + excursion_tolerance
    ):
        raise FloatingPointError(
            "Initialized model exceeds the normalized latent excursion limit: "
            f"{initial_latent_maximum:.6e} > {args.latent_excursion_limit:.6e}"
        )
    zero_correction_linear_loss = zero_correction_evaluation["linear_landau"][
        "physical_relative_loss"
    ]
    linear_nonregression_limit = max(
        1.1 * float(zero_correction_linear_loss),
        float(args.linear_nonregression_limit),
    )
    linear_initialized = initial_evaluation["linear_landau"][
        "physical_relative_loss"
    ]
    if linear_initialized > linear_nonregression_limit:
        raise FloatingPointError(
            "Nonlinear initialization regresses the linear held-out baseline"
        )
    preflight_array = jnp.asarray(preflight)
    preflight_components, preflight_pullback = jax.vjp(
        lambda candidate: loss_components_function(candidate, preflight_array),
        params,
    )
    preflight_loss, preflight_physical, preflight_teacher_residual = (
        preflight_components
    )
    preflight_zeros = jnp.zeros_like(preflight_loss)
    preflight_ones = jnp.ones_like(preflight_loss)
    preflight_physical_grads = preflight_pullback(
        (preflight_zeros, preflight_ones, preflight_zeros)
    )[0]
    preflight_teacher_residual_grads = preflight_pullback(
        (preflight_zeros, preflight_zeros, preflight_ones)
    )[0]
    (
        preflight_grads,
        _,
        _,
        preflight_cosine,
        preflight_auxiliary_norm,
        preflight_conflict,
        preflight_alignment,
    ) = _combine_gradients(
        preflight_physical_grads,
        preflight_teacher_residual_grads,
        auxiliary_weight=float(args.latent_residual_weight),
        maximum_auxiliary_ratio=float(args.latent_gradient_ratio),
        mode=args.update_combination,
    )
    preflight_group_norms = _gradient_group_norms(preflight_grads)
    preflight_finite = bool(
        jnp.isfinite(preflight_loss) & _tree_all_finite(preflight_grads)
    )
    print(
        f"[preflight] full_trajectory_loss={float(preflight_loss):.6e} "
        f"physical={float(preflight_physical):.6e} "
        f"teacher_residual={float(preflight_teacher_residual):.6e} "
        f"gradient_norm={float(_tree_l2_norm(preflight_grads)):.6e} "
        f"auxiliary=(cos={float(preflight_cosine):+.3f},"
        f"conflict={int(preflight_conflict)},"
        f"norm={float(preflight_auxiliary_norm):.3e},"
        f"physical_alignment={float(preflight_alignment):.6f}) "
        f"gradient_groups=(U={float(preflight_group_norms[0]):.3e},"
        f"V={float(preflight_group_norms[1]):.3e},"
        f"conditioner={float(preflight_group_norms[2]):.3e}) "
        f"finite={int(preflight_finite)}",
        flush=True,
    )
    if not preflight_finite:
        raise FloatingPointError("Non-finite complete-trajectory preflight")
    if args.latent_readout_mode == "equilibrium_cnn":
        missing_required_gradient = float(preflight_group_norms[2]) <= 0.0
    elif (
        args.closure_readout_mode == "gated_linear"
        or args.latent_readout_mode == "gated_linear"
    ):
        missing_required_gradient = float(preflight_group_norms[0]) <= 0.0
    else:
        missing_required_gradient = any(
            float(value) <= 0.0 for value in preflight_group_norms
        )
    if missing_required_gradient:
        raise FloatingPointError(
            "Nonlinear initialization leaves a parameter group without gradients"
        )

    trust_guard_arrays = [preflight]
    for guard_step in range(1, int(args.trust_region_batches)):
        guard_batch, _ = _balanced_complete_batch(
            args.projected_cache,
            train_by_regime,
            initial_orders,
            guard_step,
            latent_rank=int(args.latent_rank),
            expected_samples=expected_samples,
        )
        trust_guard_arrays.append(guard_batch)
    # Keep balanced batches separate. Each carries one linear, weak, and strong
    # case, matching the three-entry regime weight vector used by the loss.
    trust_guard_batches = tuple(jnp.asarray(batch) for batch in trust_guard_arrays)

    @jax.jit
    def trust_guard_objective(model_params):
        objective_index = 1 if args.trust_region_objective == "physical" else 0
        values = tuple(
            loss_components_function(model_params, batch)[objective_index]
            for batch in trust_guard_batches
        )
        return jnp.mean(jnp.stack(values))

    trust_guard_loss = float(trust_guard_objective(params))

    configuration_report = {
        **vars(args),
        "reference_cache": str(args.reference_cache),
        "projected_cache": str(args.projected_cache),
        "outdir": str(args.outdir),
        "rollout_steps": horizon,
        "rollout_span": horizon * float(args.cadence),
        "coupled_linear_maximum_spectral_radius": maximum_radius,
        "correction_bound_minimum": float(np.min(correction_bounds)),
        "correction_bound_median": float(np.median(correction_bounds)),
        "correction_bound_maximum": float(np.max(correction_bounds)),
        "correction_bound_overall_distortion": float(
            correction_bound_diagnostics["overall_distortion_fraction"]
        ),
        "correction_bound_maximum_channel_distortion": maximum_bound_distortion,
        "latent_residual_scale_minimum": float(np.min(latent_residual_scale)),
        "latent_residual_scale_median": float(np.median(latent_residual_scale)),
        "latent_residual_scale_maximum": float(np.max(latent_residual_scale)),
        "closure_residual_scale": float(closure_residual_scale),
        "closure_correction_bound": float(args.closure_correction_bound),
        "linear_nonregression_limit": linear_nonregression_limit,
    }

    def model_checkpoint_payload(model_params):
        checkpoint = {
            key: np.asarray(value) for key, value in model_params.items()
        }
        checkpoint.update(
            {
                "basis": np.asarray(basis),
                "linear_propagator": np.asarray(propagator),
                "resolved_center": np.asarray(resolved_center),
                "resolved_scale": np.asarray(resolved_scale),
                "latent_center": np.asarray(latent_center),
                "latent_scale": np.asarray(latent_scale),
                "latent_residual_scale": np.asarray(latent_residual_scale),
                "closure_residual_scale": np.asarray(closure_residual_scale),
                "closure_correction_bound": np.asarray(
                    args.closure_correction_bound
                ),
                "correction_bounds": np.asarray(correction_bounds),
                "k_arr": np.asarray(k_arr),
            }
        )
        return checkpoint

    history = list(resumed_history)
    best_score = min(
        (
            float(row["checkpoint_score"])
            for row in history
            if "checkpoint_score" in row
        ),
        default=float("inf"),
    )
    best_params = params
    if args.resume_training_state is not None:
        source_best_checkpoint = args.resume_training_state.with_name(
            "best_coupled_low_moment_latent.npz"
        )
        destination_best_checkpoint = (
            args.outdir / "best_coupled_low_moment_latent.npz"
        )
        if source_best_checkpoint.exists() and not destination_best_checkpoint.exists():
            shutil.copy2(source_best_checkpoint, destination_best_checkpoint)
    loss_ema = None
    started = time.perf_counter()
    for epoch in range(completed_epoch + 1, int(args.epochs) + 1):
        orders = {
            regime: rng.permutation(len(train_by_regime[regime]))
            for regime in REGIMES
        }
        losses = []
        physical_losses = []
        teacher_residual_losses = []
        gradients = []
        group_gradients = []
        update_norms = []
        clipped_update_norms = []
        physical_gradient_norms = []
        teacher_residual_gradient_norms = []
        gradient_cosines = []
        auxiliary_contribution_norms = []
        gradient_conflicts = []
        physical_gradient_alignments = []
        physical_update_norms = []
        teacher_update_norms = []
        update_cosines = []
        teacher_update_contribution_norms = []
        update_conflicts = []
        physical_update_alignments = []
        trust_region_scales = []
        trust_region_backtrack_counts = []
        pending_physical_gradients = []
        pending_teacher_gradients = []
        for step in range(int(args.steps_per_epoch)):
            batch, batch_cases = _balanced_complete_batch(
                args.projected_cache,
                train_by_regime,
                orders,
                step,
                latent_rank=int(args.latent_rank),
                expected_samples=expected_samples,
            )
            (
                loss,
                physical_loss,
                teacher_residual_loss,
                physical_grads,
                teacher_residual_grads,
                gradient_finite,
            ) = loss_and_gradients(params, jnp.asarray(batch))
            if not bool(gradient_finite):
                raise FloatingPointError(
                    f"Non-finite loss or gradient at epoch {epoch}, step {step + 1}"
                )
            losses.append(float(loss))
            physical_losses.append(float(physical_loss))
            teacher_residual_losses.append(float(teacher_residual_loss))
            loss_ema = (
                float(loss)
                if loss_ema is None
                else float(args.loss_ema_decay) * loss_ema
                + (1.0 - float(args.loss_ema_decay)) * float(loss)
            )
            pending_physical_gradients.append(
                _mask_tail_output_gradients(
                    physical_grads,
                    int(args.train_tail_output_from),
                    train_output_rows,
                    expert_output_only=bool(args.train_expert_output_only),
                    expert_phase_output_only=bool(
                        args.train_expert_phase_output_only
                    ),
                    expert_gate_only=bool(args.train_expert_gate_only),
                    expert_cyclic_gate_only=bool(
                        args.train_expert_cyclic_gate_only
                    ),
                    expert_cyclic_output_only=bool(
                        args.train_expert_cyclic_output_only
                    ),
                    cyclic_low_mode_closure_only=bool(
                        args.train_cyclic_low_mode_closure_only
                    ),
                    cyclic_low_mode_pressure_only=bool(
                        args.train_cyclic_low_mode_pressure_only
                    ),
                    cyclic_modal_pressure_only=bool(
                        args.train_cyclic_modal_pressure_only
                    ),
                    cyclic_low_mode_momentum_only=bool(
                        args.train_cyclic_low_mode_momentum_only
                    ),
                    cyclic_feature_indices=train_cyclic_feature_indices,
                    expert_indices=train_expert_indices,
                )
            )
            pending_teacher_gradients.append(
                _mask_tail_output_gradients(
                    teacher_residual_grads,
                    int(args.train_tail_output_from),
                    train_output_rows,
                    expert_output_only=bool(args.train_expert_output_only),
                    expert_phase_output_only=bool(
                        args.train_expert_phase_output_only
                    ),
                    expert_gate_only=bool(args.train_expert_gate_only),
                    expert_cyclic_gate_only=bool(
                        args.train_expert_cyclic_gate_only
                    ),
                    expert_cyclic_output_only=bool(
                        args.train_expert_cyclic_output_only
                    ),
                    cyclic_low_mode_closure_only=bool(
                        args.train_cyclic_low_mode_closure_only
                    ),
                    cyclic_low_mode_pressure_only=bool(
                        args.train_cyclic_low_mode_pressure_only
                    ),
                    cyclic_modal_pressure_only=bool(
                        args.train_cyclic_modal_pressure_only
                    ),
                    cyclic_low_mode_momentum_only=bool(
                        args.train_cyclic_low_mode_momentum_only
                    ),
                    cyclic_feature_indices=train_cyclic_feature_indices,
                    expert_indices=train_expert_indices,
                )
            )
            if (step + 1) % int(args.gradient_accumulation_steps) != 0:
                continue
            physical_grads = _aggregate_gradients(
                pending_physical_gradients,
                normalize=bool(args.normalize_accumulated_gradients),
                method=str(args.gradient_aggregation),
            )
            teacher_residual_grads = _aggregate_gradients(
                pending_teacher_gradients,
                normalize=bool(args.normalize_accumulated_gradients),
                method=str(args.gradient_aggregation),
            )
            pending_physical_gradients.clear()
            pending_teacher_gradients.clear()
            previous_params = params
            (
                params,
                optimizers,
                grad_norm,
                group_norms,
                update_norm,
                clipped_update_norm,
                physical_grad_norm,
                teacher_residual_grad_norm,
                gradient_cosine,
                auxiliary_contribution_norm,
                gradient_conflict,
                physical_gradient_alignment,
                physical_update_norm,
                teacher_update_norm,
                update_cosine,
                teacher_update_contribution_norm,
                update_conflict,
                physical_update_alignment,
                finite,
            ) = apply_gradients(
                params,
                optimizers,
                physical_grads,
                teacher_residual_grads,
            )
            trust_scale = 1.0
            trust_backtracks = 0
            if bool(finite) and int(args.trust_region_backtracks) > 0:
                proposed_params = params
                allowed_guard_loss = trust_guard_loss * (
                    1.0 + float(args.trust_region_tolerance)
                )
                accepted = False
                for backtrack in range(int(args.trust_region_backtracks) + 1):
                    candidate_guard_loss = float(trust_guard_objective(params))
                    if (
                        np.isfinite(candidate_guard_loss)
                        and candidate_guard_loss <= allowed_guard_loss
                    ):
                        trust_guard_loss = candidate_guard_loss
                        trust_backtracks = backtrack
                        accepted = True
                        break
                    trust_scale *= 0.5
                    params = jax.tree_util.tree_map(
                        lambda old, proposed: old
                        + trust_scale * (proposed - old),
                        previous_params,
                        proposed_params,
                    )
                if not accepted:
                    params = previous_params
                    trust_scale = 0.0
                    trust_backtracks = int(args.trust_region_backtracks) + 1
            trust_region_scales.append(float(trust_scale))
            trust_region_backtrack_counts.append(int(trust_backtracks))
            if not bool(finite):
                failure = {
                    "epoch": epoch,
                    "step": step + 1,
                    "loss": float(loss),
                    "physical_loss": float(physical_loss),
                    "teacher_residual_loss": float(teacher_residual_loss),
                    "gradient_norm": float(grad_norm),
                    "parameter_update_norm": float(update_norm),
                    "physical_gradient_norm": float(physical_grad_norm),
                    "teacher_residual_gradient_norm": float(
                        teacher_residual_grad_norm
                    ),
                    "gradient_cosine": float(gradient_cosine),
                    "auxiliary_contribution_norm": float(
                        auxiliary_contribution_norm
                    ),
                    "gradient_conflict": bool(gradient_conflict),
                    "physical_gradient_alignment": float(
                        physical_gradient_alignment
                    ),
                    "physical_update_norm": float(physical_update_norm),
                    "teacher_update_norm": float(teacher_update_norm),
                    "update_cosine": float(update_cosine),
                    "teacher_update_contribution_norm": float(
                        teacher_update_contribution_norm
                    ),
                    "update_conflict": bool(update_conflict),
                    "physical_update_alignment": float(
                        physical_update_alignment
                    ),
                    "cases": _failure_diagnostics(
                        params,
                        batch,
                        batch_cases,
                        rollout,
                        correction_diagnostic,
                        resolved_scale=resolved_scale,
                        latent_scale=latent_scale,
                        correction_bounds=correction_bounds,
                    ),
                }
                _atomic_savez(
                    args.outdir / "latest_coupled_low_moment_latent.npz",
                    model_checkpoint_payload(params),
                )
                state_payload = _training_state_payload(params, optimizers)
                state_payload.update(
                    {
                        "completed_epoch": np.asarray(epoch - 1),
                        "failed_epoch": np.asarray(epoch),
                        "failed_step": np.asarray(step + 1),
                    }
                )
                _atomic_savez(
                    args.outdir / "latest_training_state.npz", state_payload
                )
                _atomic_write_json(
                    args.outdir / "failure_diagnostics.json", failure
                )
                partial_report = {
                    "configuration": configuration_report,
                    "history": history,
                    "best_checkpoint_score": best_score,
                    "failure": failure,
                }
                _atomic_write_json(args.outdir / "report.json", partial_report)
                if history:
                    _write_training_artifacts(partial_report, args.outdir)
                raise FloatingPointError(
                    f"Non-finite loss or gradient at epoch {epoch}, step {step + 1}"
                )
            gradients.append(float(grad_norm))
            group_gradients.append(tuple(float(value) for value in group_norms))
            update_norms.append(float(update_norm))
            clipped_update_norms.append(float(clipped_update_norm))
            physical_gradient_norms.append(float(physical_grad_norm))
            teacher_residual_gradient_norms.append(
                float(teacher_residual_grad_norm)
            )
            gradient_cosines.append(float(gradient_cosine))
            auxiliary_contribution_norms.append(
                float(auxiliary_contribution_norm)
            )
            gradient_conflicts.append(float(gradient_conflict))
            physical_gradient_alignments.append(
                float(physical_gradient_alignment)
            )
            physical_update_norms.append(float(physical_update_norm))
            teacher_update_norms.append(float(teacher_update_norm))
            update_cosines.append(float(update_cosine))
            teacher_update_contribution_norms.append(
                float(teacher_update_contribution_norm)
            )
            update_conflicts.append(float(update_conflict))
            physical_update_alignments.append(float(physical_update_alignment))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "physical_loss": float(np.mean(physical_losses)),
            "teacher_residual_loss": float(
                np.mean(teacher_residual_losses)
            ),
            "loss_ema": float(loss_ema),
            "gradient_mean": float(np.mean(gradients)),
            "parameter_update_mean": float(np.mean(update_norms)),
            "clipped_parameter_update_mean": float(
                np.mean(clipped_update_norms)
            ),
            "physical_gradient_mean": float(np.mean(physical_gradient_norms)),
            "teacher_residual_gradient_mean": float(
                np.mean(teacher_residual_gradient_norms)
            ),
            "gradient_cosine_mean": float(np.mean(gradient_cosines)),
            "gradient_cosine_minimum": float(np.min(gradient_cosines)),
            "gradient_conflict_fraction": float(np.mean(gradient_conflicts)),
            "auxiliary_contribution_mean": float(
                np.mean(auxiliary_contribution_norms)
            ),
            "physical_gradient_alignment_minimum": float(
                np.min(physical_gradient_alignments)
            ),
            "physical_update_mean": float(np.mean(physical_update_norms)),
            "teacher_update_mean": float(np.mean(teacher_update_norms)),
            "update_cosine_mean": float(np.mean(update_cosines)),
            "update_conflict_fraction": float(np.mean(update_conflicts)),
            "teacher_update_contribution_mean": float(
                np.mean(teacher_update_contribution_norms)
            ),
            "physical_update_alignment_minimum": float(
                np.min(physical_update_alignments)
            ),
            "trust_region_scale_mean": float(np.mean(trust_region_scales)),
            "trust_region_scale_minimum": float(np.min(trust_region_scales)),
            "trust_region_backtracks_maximum": int(
                np.max(trust_region_backtrack_counts)
            ),
            "trust_region_guard_loss": float(trust_guard_loss),
            "gradient_group_mean": {
                name: float(np.mean([values[index] for values in group_gradients]))
                for index, name in enumerate(("operator_u", "operator_v", "conditioner"))
            },
        }
        message = (
            f"[train] epoch {epoch:03d}/{args.epochs:03d} "
            f"monitor={row['loss']:.6e} physical={row['physical_loss']:.6e} "
            f"teacher_residual={row['teacher_residual_loss']:.6e} "
            f"ema={row['loss_ema']:.6e} "
            f"grad={row['gradient_mean']:.3e} "
            f"grad_balance=(phys={row['physical_gradient_mean']:.2e},"
            f"teacher={row['teacher_residual_gradient_mean']:.2e},"
            f"aux={row['auxiliary_contribution_mean']:.2e},"
            f"cos={row['gradient_cosine_mean']:+.2f}/"
            f"{row['gradient_cosine_minimum']:+.2f},"
            f"conflict={row['gradient_conflict_fraction']:.0%},"
            f"align_min={row['physical_gradient_alignment_minimum']:.3f}) "
            f"update={row['clipped_parameter_update_mean']:.3e}/"
            f"{row['parameter_update_mean']:.3e} "
            f"dual_update=(phys={row['physical_update_mean']:.2e},"
            f"teacher={row['teacher_update_mean']:.2e},"
            f"used={row['teacher_update_contribution_mean']:.2e},"
            f"cos={row['update_cosine_mean']:+.2f},"
            f"conflict={row['update_conflict_fraction']:.0%},"
            f"align_min={row['physical_update_alignment_minimum']:.3f}) "
            f"trust=(scale={row['trust_region_scale_mean']:.2e}/"
            f"{row['trust_region_scale_minimum']:.2e},"
            f"backtracks={row['trust_region_backtracks_maximum']},"
            f"guard={row['trust_region_guard_loss']:.3e}) "
            f"grad_groups=(U={row['gradient_group_mean']['operator_u']:.2e},"
            f"V={row['gradient_group_mean']['operator_v']:.2e},"
            f"C={row['gradient_group_mean']['conditioner']:.2e})"
        )
        validated = epoch == 1 or epoch % int(args.validation_every) == 0
        if validated:
            evaluation = _evaluate(
                params,
                rollout,
                args.projected_cache,
                cases,
                latent_rank=int(args.latent_rank),
                expected_samples=expected_samples,
                resolved_scale=resolved_scale,
                latent_scale=latent_scale,
                k_arr=k_arr,
                poisson_sign=poisson_sign,
                correction_function=correction_diagnostic,
                closure_function=closure_diagnostic,
                correction_bounds=correction_bounds,
                propagator=propagator,
                latent_residual_scale=latent_residual_scale,
                basis=basis,
                closure_residual_scale=closure_residual_scale,
                cadence=float(args.cadence),
                linear_baseline=str(args.linear_baseline),
                fine_steps=int(args.fine_steps),
                fine_dt=teacher_dt,
                hermite_tail_damping=float(args.hermite_tail_damping),
                hermite_tail_power=float(args.hermite_tail_power),
                semilinear_kick_scale=float(args.semilinear_kick_scale),
                semilinear_correction_location=str(
                    args.semilinear_correction_location
                ),
                latent_delay_input=bool(args.latent_delay_input),
            )
            row["heldout"] = evaluation
            row["heldout_finite"] = all(
                values["finite"] for values in evaluation.values()
            )
            row["heldout_latent_excursion_passed"] = all(
                values["maximum_normalized_latent"]
                <= float(args.latent_excursion_limit) + excursion_tolerance
                for values in evaluation.values()
            )
            if not row["heldout_latent_excursion_passed"]:
                raise FloatingPointError(
                    "Held-out validation exceeded the normalized latent excursion limit"
                )
            score = max(
                values["physical_relative_loss"]
                + float(args.electric_spectrum_weight)
                * values["electric_spectral_amplitude_relative_mse"]
                for values in evaluation.values()
            )
            row["checkpoint_score"] = score
            row["linear_nonregression_passed"] = bool(
                evaluation["linear_landau"]["physical_relative_loss"]
                <= linear_nonregression_limit
            )
            if (
                np.isfinite(score)
                and row["heldout_finite"]
                and row["linear_nonregression_passed"]
                and score < best_score
            ):
                best_score = score
                best_params = params
                row["best_updated"] = True
            heldout_messages = []
            for regime, values in evaluation.items():
                closure_message = ""
                if "teacher_input_predicted_closure_rms" in values:
                    closure_message = (
                        f"closure={values['teacher_input_predicted_closure_rms']:.2e}/"
                        f"{values['teacher_target_closure_rms']:.2e},"
                        f"closure_rel={values['teacher_closure_relative_mse']:.2e},"
                    )
                heldout_messages.append(
                    f"{regime.split('_')[-1]}:(phys={values['physical_relative_loss']:.3e},"
                    f"E={values['electric_field_relative_mse']:.3e},"
                    f"Espec={values['electric_spectral_amplitude_relative_mse']:.3e},"
                    f"z_diag={values['latent_relative_mse']:.3e},"
                    f"zmax={values['maximum_normalized_latent']:.3e},"
                    f"corr_rms={values['correction_normalized_rms']:.2e},"
                    f"teacher_corr={values['teacher_input_predicted_correction_rms']:.2e}/"
                    f"{values['teacher_target_correction_rms']:.2e},"
                    f"{closure_message}"
                    f"teacher_resid={values['teacher_residual_normalized_mse']:.2e},"
                    f"regrow={values['predicted_regrowth_factor']:.2e}/"
                    f"{values['teacher_regrowth_factor']:.2e}"
                    f"[n={values['regrowth_case_count']}],"
                    f"old_gt1={values['correction_old_bound_exceedance_fraction']:.2%},"
                    f"bound_p99={values['correction_bound_usage_p99']:.2e})"
                )
            message += " heldout=" + ",".join(heldout_messages)
            message += (
                f" linear_guard={int(row['linear_nonregression_passed'])}"
            )
        row["elapsed_minutes"] = (time.perf_counter() - started) / 60.0
        history.append(row)
        print(message + f" elapsed={row['elapsed_minutes']:.1f}m", flush=True)

        if validated:
            if row.get("best_updated"):
                _atomic_savez(
                    args.outdir / "best_coupled_low_moment_latent.npz",
                    model_checkpoint_payload(best_params),
                )
            _atomic_savez(
                args.outdir / "latest_coupled_low_moment_latent.npz",
                model_checkpoint_payload(params),
            )
            state_payload = _training_state_payload(params, optimizers)
            state_payload["completed_epoch"] = np.asarray(epoch)
            _atomic_savez(
                args.outdir / "latest_training_state.npz", state_payload
            )
            partial_report = {
                "configuration": configuration_report,
                "history": history,
                "best_checkpoint_score": best_score,
            }
            _atomic_write_json(args.outdir / "report.json", partial_report)
            _write_training_artifacts(partial_report, args.outdir)
            if not row["heldout_finite"]:
                raise FloatingPointError(
                    f"Non-finite held-out rollout at epoch {epoch}"
                )

    report = {
        "configuration": configuration_report,
        "history": history,
        "best_checkpoint_score": best_score,
    }
    _atomic_write_json(args.outdir / "report.json", report)
    _write_training_artifacts(report, args.outdir)
    print(f"Saved coupled low-moment latent model to {args.outdir}", flush=True)
    best_checkpoint_path = args.outdir / "best_coupled_low_moment_latent.npz"
    if not args.skip_evaluation and best_checkpoint_path.exists():
        from model.eval_coupled_low_moment_latent import evaluate

        evaluate(args.outdir)
    elif not args.skip_evaluation:
        print(
            "Skipping final evaluation because no checkpoint passed the acceptance gate.",
            flush=True,
        )


if __name__ == "__main__":
    main()
