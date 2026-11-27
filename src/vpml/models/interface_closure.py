"""Learned Hermite-interface model, network parameters, and metadata identifiers.

Checkpoint I/O and trajectory feature construction live in interface_checkpoint
and interface_inference, respectively."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .._jax_precision import enable_x64

enable_x64()

import jax
import jax.numpy as jnp

Array = jnp.ndarray
LEGACY_GRID_CUBIC_TEACHER_BACKEND = "physical_grid_cubic_v1"
GRID_CUBIC_SPLINE_TEACHER_BACKEND = "grid_cubic_spline"
HIGHER_ORDER_HERMITE_TEACHER_BACKEND = "higher_order_hermite"
INTERFACE_FLUX_TRAINING_MODE = "solver_embedded_interface_flux_rollout"
INTERFACE_FLUX_OBJECTIVE = "interface_flux_rollout"
INTERFACE_FLUX_LOSS_BACKEND = "regime_balanced_all_k_interface_flux"
INTERFACE_FLUX_CHECKPOINT_SCHEMA_VERSION = 4
_LEGACY_INTERFACE_FLUX_TRIPLE = (
    "exact_q_rollout",
    "q_rollout",
    "exact_fourier_hermite_q_rollout",
)


def normalize_teacher_backend_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    if str(name) == LEGACY_GRID_CUBIC_TEACHER_BACKEND:
        return GRID_CUBIC_SPLINE_TEACHER_BACKEND
    return str(name)


@dataclass(frozen=True)
class LearnedInterfaceClosure:
    """
    Solver-embedded learned interface-flux closure for the unresolved Hermite tail.

        x_k(t) = Std(Re z_k(t), Im z_k(t), |k|/k_scale, Nv/nv_scale, E(t), H_low(t))

    The network is a linear shortcut branch plus a nonlinear residual MLP branch.
    Supported legacy exact-q checkpoints are normalized to the canonical metadata
    identifiers when loaded.
    """
    params: Dict[str, Array]
    Nm: int
    k_scale: float
    nv_scale: float
    input_mean: Array
    input_std: Array
    target_mean: Array
    target_std: Array
    hidden_width: int = 128
    res_blocks: int = 2
    equilibrium_centered: bool = False
    complex_normalization_mode: str = "componentwise"
    translation_augmented: bool = False
    Nv_targets: Optional[Tuple[int, ...]] = None
    train_regimes: Optional[Tuple[str, ...]] = None
    teacher_backend: Optional[str] = None
    teacher_Lx: Optional[float] = None
    teacher_Nx: Optional[int] = None
    rollout_Nx: Optional[int] = None
    teacher_Nv: Optional[int] = None
    teacher_vmin: Optional[float] = None
    teacher_vmax: Optional[float] = None
    teacher_dt: Optional[float] = None
    teacher_proj_Nv: Optional[int] = None
    projection_quadrature_Nv: Optional[int] = None
    include_global_indicators: bool = True
    n_low: int = 2
    training_mode: str = INTERFACE_FLUX_TRAINING_MODE
    train_objective: str = INTERFACE_FLUX_OBJECTIVE
    context_mode: str = "none"
    context_lags: int = 0
    base_input_dim: Optional[int] = None
    rollout_horizon: int = 0
    loss_backend: str = INTERFACE_FLUX_LOSS_BACKEND
    ic_manifest_sha256: Optional[str] = None
    training_ic_count: Optional[int] = None
    heldout_ic_count: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.Nm) <= 0:
            raise ValueError("Nm must be positive")
        if int(self.hidden_width) <= 0:
            raise ValueError("hidden_width must be positive")
        if int(self.res_blocks) < 0:
            raise ValueError("res_blocks must be nonnegative")
        if str(self.complex_normalization_mode) not in {"componentwise", "phase_isotropic"}:
            raise ValueError(
                "complex_normalization_mode must be 'componentwise' or 'phase_isotropic'"
            )
        if float(self.k_scale) <= 0.0:
            raise ValueError("k_scale must be positive")
        if float(self.nv_scale) <= 0.0:
            raise ValueError("nv_scale must be positive")
        if int(self.n_low) < 0:
            raise ValueError("n_low must be nonnegative")
        if self.projection_quadrature_Nv is not None and int(self.projection_quadrature_Nv) <= 3:
            raise ValueError("projection_quadrature_Nv must exceed three when provided")
        if self.rollout_Nx is not None and int(self.rollout_Nx) <= 1:
            raise ValueError("rollout_Nx must exceed one when provided")
        if (
            self.rollout_Nx is not None
            and self.teacher_Nx is not None
            and int(self.rollout_Nx) > int(self.teacher_Nx)
        ):
            raise ValueError("rollout_Nx cannot exceed teacher_Nx")
        if str(self.training_mode) != INTERFACE_FLUX_TRAINING_MODE:
            raise ValueError(f"Unsupported training_mode={self.training_mode!r}")
        if str(self.context_mode) not in {"none", "lag1_delta"}:
            raise ValueError(f"Unsupported context_mode={self.context_mode!r}")
        if int(self.context_lags) < 0:
            raise ValueError("context_lags must be nonnegative")
        if int(self.rollout_horizon) < 0:
            raise ValueError("rollout_horizon must be nonnegative")
        if str(self.train_objective) != INTERFACE_FLUX_OBJECTIVE:
            raise ValueError(f"Unsupported train_objective={self.train_objective!r}")
        if str(self.loss_backend) != INTERFACE_FLUX_LOSS_BACKEND:
            raise ValueError(f"Unsupported loss_backend={self.loss_backend!r}")
        if self.training_ic_count is not None and int(self.training_ic_count) <= 0:
            raise ValueError("training_ic_count must be positive when provided")
        if self.heldout_ic_count is not None and int(self.heldout_ic_count) <= 0:
            raise ValueError("heldout_ic_count must be positive when provided")

        input_dim = self.input_dim
        input_mean = jnp.asarray(self.input_mean, dtype=jnp.float64)
        input_std = jnp.asarray(self.input_std, dtype=jnp.float64)
        target_mean = jnp.asarray(self.target_mean, dtype=jnp.float64)
        target_std = jnp.asarray(self.target_std, dtype=jnp.float64)

        if input_mean.shape != (input_dim,):
            raise ValueError(f"input_mean must have shape ({input_dim},), got {input_mean.shape}")
        if input_std.shape != (input_dim,):
            raise ValueError(f"input_std must have shape ({input_dim},), got {input_std.shape}")
        if target_mean.shape != (2,):
            raise ValueError(f"target_mean must have shape (2,), got {target_mean.shape}")
        if target_std.shape != (2,):
            raise ValueError(f"target_std must have shape (2,), got {target_std.shape}")

    @property
    def raw_base_dim(self) -> int:
        return 2 * int(self.Nm) + 2 + (2 if bool(self.include_global_indicators) else 0)

    @property
    def input_dim(self) -> int:
        if self.context_mode == "none":
            return int(self.raw_base_dim)
        if self.context_mode == "lag1_delta":
            return 3 * int(self.raw_base_dim)
        raise ValueError(f"Unsupported context_mode={self.context_mode!r}")

    def standardized_inputs(self, x: Array) -> Array:
        input_mean = jnp.asarray(self.input_mean)
        real_dtype = input_mean.dtype
        x = jnp.asarray(x, dtype=real_dtype)
        std = jnp.maximum(jnp.asarray(self.input_std, dtype=real_dtype), jnp.asarray(1e-12, dtype=real_dtype))
        return (x - input_mean) / std

    def unstandardized_targets(self, y: Array) -> Array:
        target_mean = jnp.asarray(self.target_mean)
        real_dtype = target_mean.dtype
        y = jnp.asarray(y, dtype=real_dtype)
        std = jnp.maximum(jnp.asarray(self.target_std, dtype=real_dtype), jnp.asarray(1e-12, dtype=real_dtype))
        return y * std + target_mean

    def predict_standardized_components(self, raw_features: Array) -> Array:
        feats = self.standardized_inputs(raw_features)
        pred = jax.vmap(
            lambda z: interface_closure_apply(
                self.params,
                z,
                hidden_width=int(self.hidden_width),
                res_blocks=int(self.res_blocks),
            )
        )(feats)
        if not bool(self.equilibrium_centered):
            return pred

        zero_raw = jnp.zeros_like(jnp.asarray(raw_features))
        base_dim = int(self.raw_base_dim)
        k_col = 2 * int(self.Nm)
        nv_col = k_col + 1
        if self.context_mode == "none":
            zero_raw = zero_raw.at[:, k_col : nv_col + 1].set(
                jnp.asarray(raw_features)[:, k_col : nv_col + 1]
            )
        elif self.context_mode == "lag1_delta":
            for offset in (0, base_dim):
                zero_raw = zero_raw.at[:, offset + k_col : offset + nv_col + 1].set(
                    jnp.asarray(raw_features)[:, offset + k_col : offset + nv_col + 1]
                )
        else:
            raise ValueError(f"Unsupported context_mode={self.context_mode!r}")
        zero_feats = self.standardized_inputs(zero_raw)
        zero_pred = jax.vmap(
            lambda z: interface_closure_apply(
                self.params,
                z,
                hidden_width=int(self.hidden_width),
                res_blocks=int(self.res_blocks),
            )
        )(zero_feats)
        return pred - zero_pred

    def predict_q_components(self, raw_features: Array) -> Array:
        pred = self.predict_standardized_components(raw_features)
        if bool(self.equilibrium_centered):
            real_dtype = jnp.asarray(self.target_std).dtype
            std = jnp.maximum(
                jnp.asarray(self.target_std, dtype=real_dtype),
                jnp.asarray(1e-12, dtype=real_dtype),
            )
            return jnp.asarray(pred, dtype=real_dtype) * std
        return self.unstandardized_targets(pred)


def _xavier_normal(key: Array, shape: Tuple[int, int]) -> Array:
    fan_in, fan_out = shape
    scale = jnp.sqrt(2.0 / float(fan_in + fan_out))
    return scale * jax.random.normal(key, shape, dtype=jnp.float64)


def init_interface_closure_params(
    key: Array,
    *,
    input_dim: int,
    hidden_width: int = 128,
    res_blocks: int = 2,
) -> Dict[str, Array]:
    """Initialize the linear-plus-residual interface-closure network."""
    if int(input_dim) <= 0:
        raise ValueError("input_dim must be positive")
    if int(hidden_width) <= 0:
        raise ValueError("hidden_width must be positive")
    if int(res_blocks) < 0:
        raise ValueError("res_blocks must be nonnegative")

    keys = jax.random.split(key, 2 + (2 * int(res_blocks)) + 1)
    idx = 0
    params: Dict[str, Array] = {}
    params["W_lin"] = _xavier_normal(keys[idx], (int(input_dim), 2)); idx += 1
    params["b_lin"] = jnp.zeros((2,), dtype=jnp.float64)
    params["W_in"] = _xavier_normal(keys[idx], (int(input_dim), int(hidden_width))); idx += 1
    params["b_in"] = jnp.zeros((int(hidden_width),), dtype=jnp.float64)
    for block_idx in range(int(res_blocks)):
        params[f"W1_{block_idx}"] = _xavier_normal(keys[idx], (int(hidden_width), int(hidden_width))); idx += 1
        params[f"b1_{block_idx}"] = jnp.zeros((int(hidden_width),), dtype=jnp.float64)
        params[f"W2_{block_idx}"] = _xavier_normal(keys[idx], (int(hidden_width), int(hidden_width))); idx += 1
        params[f"b2_{block_idx}"] = jnp.zeros((int(hidden_width),), dtype=jnp.float64)
    params["W_out"] = _xavier_normal(keys[idx], (int(hidden_width), 2))
    params["b_out"] = jnp.zeros((2,), dtype=jnp.float64)
    idx += 1
    return params


def interface_closure_apply(
    params: Dict[str, Array],
    x: Array,
    *,
    hidden_width: int = 128,
    res_blocks: int = 2,
) -> Array:
    """Apply the shared residual interface-closure network to one standardized feature vector."""
    del hidden_width  # encoded in parameter shapes
    param_dtype = jnp.asarray(params["W_lin"]).dtype
    x = jnp.asarray(x, dtype=param_dtype)
    y_lin = x @ params["W_lin"] + params["b_lin"]
    h = jax.nn.silu(x @ params["W_in"] + params["b_in"])
    for block_idx in range(int(res_blocks)):
        residual = jax.nn.silu(h @ params[f"W1_{block_idx}"] + params[f"b1_{block_idx}"])
        residual = residual @ params[f"W2_{block_idx}"] + params[f"b2_{block_idx}"]
        h = h + residual
    y_nl = h @ params["W_out"] + params["b_out"]
    return (y_lin + y_nl).astype(param_dtype)
