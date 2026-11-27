"""Canonical interface-flux regimes, precision choices, and schema identifiers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import numpy as np


REGIME_LINEAR = "linear_landau"


REGIME_WEAK = "nonlinear_landau_weak"


REGIME_STRONG = "nonlinear_landau_strong"


ALL_REGIMES = (REGIME_LINEAR, REGIME_WEAK, REGIME_STRONG)


INTERFACE_FLUX_ROLLOUT_TRAINING_MODE = "solver_embedded_interface_flux_rollout"


INTERFACE_FLUX_ROLLOUT_OBJECTIVE = "interface_flux_rollout"


INTERFACE_FLUX_ROLLOUT_LOSS_BACKEND = "regime_balanced_all_k_interface_flux"


INTERFACE_FLUX_PROJECTION_SCHEME = "cubic_spline_uniform_trapezoid"


EXACT_ROLLOUT_PRECISION_FLOAT64 = "float64"


EXACT_ROLLOUT_PRECISION_FLOAT32 = "float32"


ALL_EXACT_ROLLOUT_PRECISIONS = (
    EXACT_ROLLOUT_PRECISION_FLOAT64,
    EXACT_ROLLOUT_PRECISION_FLOAT32,
)


EXACT_TARGET_SAMPLING_CYCLE = "cycle"


CANONICAL_NV_TARGETS = (6, 7, 12, 20, 36, 64)


CANONICAL_REGIMES = (REGIME_LINEAR, REGIME_WEAK, REGIME_STRONG)


CANONICAL_METADATA_SCHEMA_VERSION = 4


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def interface_flux_rollout_coeff_key(projection_order: int) -> str:
    return f"a_hat_ref_order{int(projection_order)}"


def exact_rollout_precision_dtypes(precision: str) -> Tuple[object, object]:
    mode = str(precision)
    if mode == EXACT_ROLLOUT_PRECISION_FLOAT64:
        return jnp.float64, jnp.complex128
    if mode == EXACT_ROLLOUT_PRECISION_FLOAT32:
        return jnp.float32, jnp.complex64
    raise ValueError(
        f"exact_rollout_precision must be one of {ALL_EXACT_ROLLOUT_PRECISIONS!r}, "
        f"got {precision!r}"
    )


def exact_rollout_numpy_complex_dtype(complex_dtype: object) -> np.dtype:
    dtype = np.dtype(complex_dtype)
    if dtype == np.dtype(np.complex64):
        return np.dtype(np.complex64)
    if dtype == np.dtype(np.complex128):
        return np.dtype(np.complex128)
    raise ValueError(f"unsupported exact rollout complex dtype {complex_dtype!r}")


def exact_rollout_numpy_real_dtype(complex_dtype: object) -> np.dtype:
    complex_np_dtype = exact_rollout_numpy_complex_dtype(complex_dtype)
    return np.dtype(np.float32 if complex_np_dtype == np.dtype(np.complex64) else np.float64)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the canonical solver-embedded interface-flux closure."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--reference-cache-dir", type=Path, required=True)
    parser.add_argument("--ic-manifest", type=Path, required=True)
    parser.add_argument("--loss-plot", type=Path, default=None)
    parser.add_argument("--build-dataset-only", action="store_true")
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument(
        "--precision",
        choices=ALL_EXACT_ROLLOUT_PRECISIONS,
        default=EXACT_ROLLOUT_PRECISION_FLOAT32,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps-per-epoch", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T-final", type=float, default=120.0)
    parser.add_argument("--Nm", type=int, default=6)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--n-low", type=int, default=2)
    parser.add_argument("--k-scale", type=float, default=None)
    parser.add_argument("--nv-scale", type=float, default=None)
    parser.add_argument("--teacher-Nx", type=int, default=256)
    parser.add_argument("--rollout-Nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", type=int, default=8192)
    parser.add_argument("--projection-quadrature-Nv", type=int, default=4096)
    parser.add_argument("--teacher-L", type=float, default=4.0 * math.pi)
    parser.add_argument("--teacher-vmin", type=float, default=-8.0)
    parser.add_argument("--teacher-vmax", type=float, default=8.0)
    parser.add_argument("--teacher-dt", type=float, default=1e-2)
    parser.add_argument("--teacher-poisson-sign", type=float, default=1.0)
    parser.add_argument("--ic-cases-per-regime", type=int, default=20)
    parser.add_argument("--ic-heldout-per-regime", type=int, default=4)
    parser.add_argument("--ic-generation-seed", type=int, default=1729)
    parser.add_argument("--ic-split-seed", type=int, default=2718)
    parser.add_argument("--ic-modes", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--history-stride", type=int, default=20)
    parser.add_argument(
        "--cache-snapshot-times",
        type=str,
        default="20.0,40.0,60.0,80.0,100.0,120.0",
    )
    parser.add_argument("--profile-trace-dir", type=Path, default=None)
    parser.add_argument("--profile-train-steps", type=int, default=0)
    parser.add_argument("--profile-skip-steps", type=int, default=1)
    return parser


def validate_training_options(args):
    """Validate the requested rollout and teacher grids before reference work."""
    if int(args.rollout_horizon) <= 0:
        raise ValueError("--rollout-horizon must be positive")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.steps_per_epoch) <= 0:
        raise ValueError("--steps-per-epoch must be positive")
    if float(args.T_final) <= 0.0:
        raise ValueError("--T-final must be positive")
    if int(args.rollout_Nx) <= 1 or int(args.rollout_Nx) > int(args.teacher_Nx):
        raise ValueError(
            f"--rollout-Nx={int(args.rollout_Nx)} must be greater than one and "
            f"no larger than --teacher-Nx={int(args.teacher_Nx)}"
        )
    projection_quadrature_Nv = (
        int(args.teacher_Nv)
        if args.projection_quadrature_Nv is None
        else int(args.projection_quadrature_Nv)
    )
    if projection_quadrature_Nv <= 3:
        raise ValueError("--projection-quadrature-Nv must exceed three")
    if any(target_nv < int(args.Nm) for target_nv in CANONICAL_NV_TARGETS):
        raise ValueError(
            f"Nm={args.Nm} exceeds the smallest canonical cutoff "
            f"{min(CANONICAL_NV_TARGETS)}"
        )
    return projection_quadrature_Nv
