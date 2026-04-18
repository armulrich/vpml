"""
Core Fourier–Hermite operators + IMEX CNAB2 integrator in **JAX** for 1D1V Vlasov-type
models with periodic x and Hermite expansion in v.

This module is intentionally "physics-agnostic": it provides
  - Fourier grid utilities (rFFT in x)
  - Hermite ladder streaming operator (tridiagonal in Hermite index)
  - Poisson solver for E from density a0
  - Optional spectral dealiasing in x (2/3 rule)
  - Optional Hermite damping models:
        * Hyper-collisions (Palisso et al. 2024, arXiv:2412.07073)
        * Hou–Li filter interpreted as damping
  - Optional Hermite exponential filter (spectral viscosity) as a per-step multiplier
  - Optional Nm=1 Hammett–Perkins nonlocal closure (time-stepping support)
  - Optional learned interface-closure helpers for solver-embedded ML boundaries

The nonlinear physics (equilibrium coefficients, acceleration term, external control fields, etc.)
live in separate scripts that import this module.

Notes on conventions
--------------------
We evolve *Hermite coefficients of the perturbation* a_n(x,t) such that

    f(x,v,t) = sum_n a_n(x,t) φ_n(v)

where φ_n are normalized probabilists' Hermite functions with Gaussian weight.
Density is ρ(x,t) = a_0(x,t).

Poisson in 1D periodic domain is handled in Fourier space via
    E_k = poisson_sign * i * ρ_k / k   (k != 0),  E_0 = 0.
Hence
    i k E_k = -poisson_sign * ρ_k.
With poisson_sign = +1 this matches the perturbation-form convention used in the
provided nonlinear two-stream/bump-on-tail script and in Eq. (2.3) of the control paper.
Some papers use the opposite sign; you can flip it by passing poisson_sign=-1.

"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .jax_runtime import bootstrap_jax_runtime

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

try:
    import scipy.sparse.linalg as spla
except Exception:  # pragma: no cover
    spla = None

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass

Array = jnp.ndarray

LEGACY_GRID_CUBIC_TEACHER_BACKEND = "physical_grid_cubic_v1"
GRID_CUBIC_SPLINE_TEACHER_BACKEND = "grid_cubic_spline"
HIGHER_ORDER_HERMITE_TEACHER_BACKEND = "higher_order_hermite"


def normalize_teacher_backend_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    if str(name) == LEGACY_GRID_CUBIC_TEACHER_BACKEND:
        return GRID_CUBIC_SPLINE_TEACHER_BACKEND
    return str(name)


def init_real_mlp(key: Array, layer_sizes: Sequence[int]) -> Dict[str, Array]:
    """
    Initialize a real-valued MLP with Xavier-style Gaussian weights.
    """
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must contain at least an input and output size")

    params: Dict[str, Array] = {}
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if m <= 0 or n <= 0:
            raise ValueError("All layer sizes must be positive")
        scale = jnp.sqrt(2.0 / float(m + n))
        params[f"W{i}"] = scale * jax.random.normal(keys[i], (m, n), dtype=jnp.float64)
        params[f"b{i}"] = jnp.zeros((n,), dtype=jnp.float64)
    return params


def mlp_apply(
    params: Dict[str, Array],
    x: Array,
    *,
    activation: Callable[[Array], Array] = jnp.tanh,
) -> Array:
    """Apply a real-valued MLP to a single feature vector."""
    h = jnp.asarray(x, dtype=jnp.float64)
    num_layers = len(params) // 2
    for i in range(num_layers):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        if i < num_layers - 1:
            h = activation(h)
    return h


# =============================================================================
# Hermite-space dissipation / closure models
# =============================================================================

@dataclass(frozen=True)
class HyperCollisions:
    """
    Hyper-collisional damping in Hermite index n (Palisso et al. Eq. 19).

        dC_n/dt += -gamma_n * C_n

    with
        gamma_n = nu * n!/(n-2α+1)! * (Nv-2α)!/(Nv-1)!    for n >= 2α-1
                = 0                                     otherwise

    Parameters
    ----------
    alpha : int
        Hyper-collision order α.
    nu : float
        Collision frequency ν.
    Nv : int
        Number of Hermite modes (0..Nv-1).
    """
    alpha: int
    nu: float
    Nv: int

    def damping_rates(self) -> Array:
        n = jnp.arange(self.Nv, dtype=jnp.float64)
        alpha = int(self.alpha)
        Nv = int(self.Nv)

        cutoff = 2 * alpha - 1
        mask = n >= cutoff

        # term1 = n!/(n-2α+1)!  (use gammaln for stability)
        # n-2α+1 == n - (2α-1)
        term1 = jnp.exp(
            jsp.special.gammaln(n + 1.0)
            - jsp.special.gammaln(n - float(2 * alpha - 1) + 1.0)
        )

        # term2 = (Nv-2α)!/(Nv-1)!
        term2 = jnp.exp(
            jsp.special.gammaln(float(Nv - 2 * alpha) + 1.0)
            - jsp.special.gammaln(float(Nv))
        )

        gamma = float(self.nu) * term1 * term2
        return jnp.where(mask, gamma, 0.0)


@dataclass(frozen=True)
class HouLiFilter:
    """
    Hou–Li exponential filter interpreted as an equivalent damping rate (Palisso et al. Eq. 20).

        C_n <- C_n * exp( -χ (n/(Nv-1))^p )

    If applied once per time step, this is equivalent to a damping term

        dC_n/dt += -(χ/Δt) (n/(Nv-1))^p C_n

    We store chi_over_dt = χ/Δt.

    Parameters
    ----------
    chi_over_dt : float
        χ/Δt in the paper's notation.
    Nv : int
        Number of Hermite modes.
    p : int
        Filter exponent (paper uses p=36 in most plots).
    """
    chi_over_dt: float
    Nv: int
    p: int = 36

    def damping_rates(self) -> Array:
        n = jnp.arange(self.Nv, dtype=jnp.float64)
        frac = n / float(self.Nv - 1)
        return float(self.chi_over_dt) * frac ** float(self.p)

    def per_step_factors(self, dt: float) -> Array:
        """
        Multiplicative factors to apply once per time step of size dt:
            exp( -dt * (χ/Δt) (n/(Nv-1))^p )
        """
        n = jnp.arange(self.Nv, dtype=jnp.float64)
        frac = n / float(self.Nv - 1)
        return jnp.exp(-float(dt) * float(self.chi_over_dt) * frac ** float(self.p))


@dataclass(frozen=True)
class HermiteExponentialFilter:
    """
    Simple exponential "spectral viscosity" filter used frequently in nonlinear Hermite truncations:

        C_n <- C_n * exp( -alpha * (n/(Nv-1))^p )

    This matches the style used in the provided numpy two-stream/bump-on-tail script.
    """
    alpha: float
    Nv: int
    p: int = 8

    def factors(self) -> Array:
        n = jnp.arange(self.Nv, dtype=jnp.float64)
        frac = n / float(self.Nv - 1)
        return jnp.exp(-float(self.alpha) * frac ** float(self.p))


@dataclass(frozen=True)
class NonlocalClosure:
    """
    Hammett–Perkins / nonlocal closure "tail" coefficients.

    The paper's Eq. (21) is typically written as a closure for the unresolved tail coefficient C_{Nv}:

        C_{Nv} = sum_{j=0..Nm-1} i * mu_{Nv-Nm+j} * sign(k) * C_{Nv-Nm+j}

    This module stores the tail mu coefficients as an array mu_tail of shape (Nm,).

    IMPORTANT
    ---------
    For *time-stepping* with a tridiagonal implicit solve, only Nm=1 is supported
    (it remains tridiagonal after substitution).
    For response-function / eigenvalue benchmarks, Nm>1 is also useful and is handled
    in the benchmark script (dense linear algebra).
    """
    mu_tail: Array  # shape (Nm,)

    @property
    def Nm(self) -> int:
        return int(self.mu_tail.shape[0])


@dataclass(frozen=True)
class LearnedInterfaceClosure:
    """
    Solver-embedded learned interface closure for the unresolved Hermite tail.

    Previous version
    ----------------
    The original closure predicted the complex interface term q_k directly from the
    current resolved Hermite window:

        x_k(t) = Std(Re z_k(t), Im z_k(t), |k|/k_scale, Nv/nv_scale, E(t), H_low(t))

    and trained only with one-step q-supervision.

    Stability-aware version
    -----------------------
    The new trainer can also attach temporal context and use a hybrid objective. The
    deployed model remains a direct map from standardized features to q_k, but the
    feature builder may use a context mode such as

        x_k^ctx(t) = (x_k(t), x_k(t - dt), x_k(t) - x_k(t - dt)).

    The network is a linear shortcut branch plus a nonlinear residual MLP branch.
    Old q-only checkpoints remain loadable without migration.
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
    Nv_targets: Optional[Tuple[int, ...]] = None
    train_regimes: Optional[Tuple[str, ...]] = None
    teacher_backend: Optional[str] = None
    teacher_Lx: Optional[float] = None
    teacher_Nx: Optional[int] = None
    teacher_Nv: Optional[int] = None
    teacher_vmin: Optional[float] = None
    teacher_vmax: Optional[float] = None
    teacher_dt: Optional[float] = None
    teacher_proj_Nv: Optional[int] = None
    include_global_indicators: bool = True
    n_low: int = 2
    training_mode: str = "offline_rollout"
    train_objective: str = "q_only"
    context_mode: str = "none"
    context_lags: int = 0
    base_input_dim: Optional[int] = None
    rollout_horizon: int = 0
    tail_start_fraction: float = 2.0 / 3.0
    loss_backend: Optional[str] = None
    lambda_q: float = 1.0
    lambda_E: float = 0.0
    lambda_dist: float = 0.0
    lambda_tail: float = 0.0
    lambda_neg: float = 0.0
    lambda_reg: float = 0.0
    online_v_probes: int = 0
    stability_loss_definition: Optional[str] = None

    def __post_init__(self) -> None:
        if int(self.Nm) <= 0:
            raise ValueError("Nm must be positive")
        if int(self.hidden_width) <= 0:
            raise ValueError("hidden_width must be positive")
        if int(self.res_blocks) < 0:
            raise ValueError("res_blocks must be nonnegative")
        if float(self.k_scale) <= 0.0:
            raise ValueError("k_scale must be positive")
        if float(self.nv_scale) <= 0.0:
            raise ValueError("nv_scale must be positive")
        if int(self.n_low) < 0:
            raise ValueError("n_low must be nonnegative")
        if str(self.training_mode) not in {"offline_rollout", "online_rollout"}:
            raise ValueError(f"Unsupported training_mode={self.training_mode!r}")
        if str(self.context_mode) not in {"none", "lag1_delta"}:
            raise ValueError(f"Unsupported context_mode={self.context_mode!r}")
        if int(self.context_lags) < 0:
            raise ValueError("context_lags must be nonnegative")
        if int(self.rollout_horizon) < 0:
            raise ValueError("rollout_horizon must be nonnegative")
        if not (0.0 < float(self.tail_start_fraction) <= 1.0):
            raise ValueError("tail_start_fraction must lie in (0, 1]")
        if str(self.train_objective) not in {"q_only", "stability_aware", "trajectory"}:
            raise ValueError(f"Unsupported train_objective={self.train_objective!r}")
        if int(self.online_v_probes) < 0:
            raise ValueError("online_v_probes must be nonnegative")

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
        x = jnp.asarray(x, dtype=jnp.float64)
        std = jnp.maximum(jnp.asarray(self.input_std, dtype=jnp.float64), 1e-12)
        return (x - jnp.asarray(self.input_mean, dtype=jnp.float64)) / std

    def unstandardized_targets(self, y: Array) -> Array:
        y = jnp.asarray(y, dtype=jnp.float64)
        std = jnp.maximum(jnp.asarray(self.target_std, dtype=jnp.float64), 1e-12)
        return y * std + jnp.asarray(self.target_mean, dtype=jnp.float64)

    def predict_standardized_components(self, raw_features: Array) -> Array:
        feats = self.standardized_inputs(raw_features)
        return jax.vmap(
            lambda z: interface_closure_apply(
                self.params,
                z,
                hidden_width=int(self.hidden_width),
                res_blocks=int(self.res_blocks),
            )
        )(feats)

    def predict_q_components(self, raw_features: Array) -> Array:
        return self.unstandardized_targets(
            self.predict_standardized_components(raw_features)
        )


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
    x = jnp.asarray(x, dtype=jnp.float64)
    y_lin = x @ params["W_lin"] + params["b_lin"]
    h = jax.nn.silu(x @ params["W_in"] + params["b_in"])
    for block_idx in range(int(res_blocks)):
        residual = jax.nn.silu(h @ params[f"W1_{block_idx}"] + params[f"b1_{block_idx}"])
        residual = residual @ params[f"W2_{block_idx}"] + params[f"b2_{block_idx}"]
        h = h + residual
    y_nl = h @ params["W_out"] + params["b_out"]
    return (y_lin + y_nl).astype(jnp.float64)


def save_learned_interface_closure_npz(
    path: str | os.PathLike[str],
    learned: LearnedInterfaceClosure,
) -> None:
    """Save a learned interface-closure checkpoint in NPZ format."""
    payload: Dict[str, np.ndarray] = {
        "closure_kind": np.array(["interface_closure"], dtype=np.str_),
        "Nm": np.array([int(learned.Nm)], dtype=np.int32),
        "k_scale": np.array([float(learned.k_scale)], dtype=np.float64),
        "nv_scale": np.array([float(learned.nv_scale)], dtype=np.float64),
        "input_mean": np.asarray(learned.input_mean, dtype=np.float64),
        "input_std": np.asarray(learned.input_std, dtype=np.float64),
        "target_mean": np.asarray(learned.target_mean, dtype=np.float64),
        "target_std": np.asarray(learned.target_std, dtype=np.float64),
        "hidden_width": np.array([int(learned.hidden_width)], dtype=np.int32),
        "res_blocks": np.array([int(learned.res_blocks)], dtype=np.int32),
        "Nv_targets": np.array(
            [] if learned.Nv_targets is None else learned.Nv_targets,
            dtype=np.int32,
        ),
        "train_regimes": np.array(
            [] if learned.train_regimes is None else learned.train_regimes,
            dtype=np.str_,
        ),
        "teacher_backend": np.array(
            []
            if learned.teacher_backend is None
            else [normalize_teacher_backend_name(learned.teacher_backend)],
            dtype=np.str_,
        ),
        "include_global_indicators": np.array([int(bool(learned.include_global_indicators))], dtype=np.int32),
        "n_low": np.array([int(learned.n_low)], dtype=np.int32),
        "training_mode": np.array([str(learned.training_mode)], dtype=np.str_),
        "train_objective": np.array([str(learned.train_objective)], dtype=np.str_),
        "context_mode": np.array([str(learned.context_mode)], dtype=np.str_),
        "context_lags": np.array([int(learned.context_lags)], dtype=np.int32),
        "base_input_dim": np.array([int(learned.raw_base_dim if learned.base_input_dim is None else learned.base_input_dim)], dtype=np.int32),
        "rollout_horizon": np.array([int(learned.rollout_horizon)], dtype=np.int32),
        "tail_start_fraction": np.array([float(learned.tail_start_fraction)], dtype=np.float64),
        "loss_backend": np.array([] if learned.loss_backend is None else [str(learned.loss_backend)], dtype=np.str_),
        "lambda_q": np.array([float(learned.lambda_q)], dtype=np.float64),
        "lambda_E": np.array([float(learned.lambda_E)], dtype=np.float64),
        "lambda_dist": np.array([float(learned.lambda_dist)], dtype=np.float64),
        "lambda_tail": np.array([float(learned.lambda_tail)], dtype=np.float64),
        "lambda_neg": np.array([float(learned.lambda_neg)], dtype=np.float64),
        "lambda_reg": np.array([float(learned.lambda_reg)], dtype=np.float64),
        "online_v_probes": np.array([int(learned.online_v_probes)], dtype=np.int32),
    }
    if learned.stability_loss_definition is not None:
        payload["stability_loss_definition"] = np.array([str(learned.stability_loss_definition)], dtype=np.str_)
    if learned.teacher_Lx is not None:
        payload["teacher_Lx"] = np.array([float(learned.teacher_Lx)], dtype=np.float64)
    if learned.teacher_Nx is not None:
        payload["teacher_Nx"] = np.array([int(learned.teacher_Nx)], dtype=np.int32)
    if learned.teacher_Nv is not None:
        payload["teacher_Nv"] = np.array([int(learned.teacher_Nv)], dtype=np.int32)
    if learned.teacher_vmin is not None:
        payload["teacher_vmin"] = np.array([float(learned.teacher_vmin)], dtype=np.float64)
    if learned.teacher_vmax is not None:
        payload["teacher_vmax"] = np.array([float(learned.teacher_vmax)], dtype=np.float64)
    if learned.teacher_dt is not None:
        payload["teacher_dt"] = np.array([float(learned.teacher_dt)], dtype=np.float64)
    if learned.teacher_proj_Nv is not None:
        payload["teacher_proj_Nv"] = np.array([int(learned.teacher_proj_Nv)], dtype=np.int32)
    for name, value in learned.params.items():
        payload[name] = np.asarray(value, dtype=np.float64)
    np.savez(path, **payload)


def load_learned_interface_closure_npz(path: str | os.PathLike[str]) -> LearnedInterfaceClosure:
    """Load a learned interface-closure checkpoint saved by `save_learned_interface_closure_npz`."""
    with np.load(path) as data:
        closure_kind = None
        if "closure_kind" in data.files and data["closure_kind"].size:
            closure_kind = str(np.asarray(data["closure_kind"]).reshape(-1)[0])
        if closure_kind != "interface_closure":
            raise ValueError(
                f"Incompatible learned-closure checkpoint at {path}: expected interface_closure. "
                "Legacy mu-tail checkpoints are no longer supported and must be retrained."
            )

        params = {
            name: jnp.asarray(data[name], dtype=jnp.float64)
            for name in data.files
            if (
                name.startswith("W")
                or name.startswith("b")
                or name in {"W_lin", "b_lin", "W_in", "b_in", "W_out", "b_out"}
            )
        }
        if not params:
            raise ValueError(f"No learned-closure parameters found in checkpoint: {path}")

        Nm = int(np.asarray(data["Nm"]).reshape(-1)[0])
        k_scale = float(np.asarray(data["k_scale"]).reshape(-1)[0])
        nv_scale = float(np.asarray(data["nv_scale"]).reshape(-1)[0])
        input_mean = jnp.asarray(data["input_mean"], dtype=jnp.float64)
        input_std = jnp.asarray(data["input_std"], dtype=jnp.float64)
        target_mean = jnp.asarray(data["target_mean"], dtype=jnp.float64)
        target_std = jnp.asarray(data["target_std"], dtype=jnp.float64)
        hidden_width = int(np.asarray(data["hidden_width"]).reshape(-1)[0])
        res_blocks = int(np.asarray(data["res_blocks"]).reshape(-1)[0])

        Nv_targets = None
        if "Nv_targets" in data.files and data["Nv_targets"].size:
            Nv_targets = tuple(int(v) for v in np.asarray(data["Nv_targets"], dtype=np.int32).tolist())

        train_regimes = None
        if "train_regimes" in data.files and data["train_regimes"].size:
            train_regimes = tuple(str(v) for v in np.asarray(data["train_regimes"], dtype=np.str_).tolist())

        teacher_backend = None
        if "teacher_backend" in data.files and data["teacher_backend"].size:
            teacher_backend = normalize_teacher_backend_name(
                str(np.asarray(data["teacher_backend"], dtype=np.str_).reshape(-1)[0])
            )

        teacher_Lx = None if "teacher_Lx" not in data.files or not data["teacher_Lx"].size else float(np.asarray(data["teacher_Lx"]).reshape(-1)[0])
        teacher_Nx = None if "teacher_Nx" not in data.files or not data["teacher_Nx"].size else int(np.asarray(data["teacher_Nx"]).reshape(-1)[0])
        teacher_Nv = None if "teacher_Nv" not in data.files or not data["teacher_Nv"].size else int(np.asarray(data["teacher_Nv"]).reshape(-1)[0])
        teacher_vmin = None if "teacher_vmin" not in data.files or not data["teacher_vmin"].size else float(np.asarray(data["teacher_vmin"]).reshape(-1)[0])
        teacher_vmax = None if "teacher_vmax" not in data.files or not data["teacher_vmax"].size else float(np.asarray(data["teacher_vmax"]).reshape(-1)[0])
        teacher_dt = None if "teacher_dt" not in data.files or not data["teacher_dt"].size else float(np.asarray(data["teacher_dt"]).reshape(-1)[0])
        teacher_proj_Nv = None if "teacher_proj_Nv" not in data.files or not data["teacher_proj_Nv"].size else int(np.asarray(data["teacher_proj_Nv"]).reshape(-1)[0])
        include_global_indicators = (
            bool(np.asarray(data["include_global_indicators"]).reshape(-1)[0])
            if "include_global_indicators" in data.files and data["include_global_indicators"].size
            else input_mean.shape[0] == (2 * Nm + 4)
        )
        n_low = (
            int(np.asarray(data["n_low"]).reshape(-1)[0])
            if "n_low" in data.files and data["n_low"].size
            else 2
        )
        training_mode = (
            str(np.asarray(data["training_mode"], dtype=np.str_).reshape(-1)[0])
            if "training_mode" in data.files and data["training_mode"].size
            else "offline_rollout"
        )
        train_objective = (
            str(np.asarray(data["train_objective"], dtype=np.str_).reshape(-1)[0])
            if "train_objective" in data.files and data["train_objective"].size
            else "q_only"
        )
        context_mode = (
            str(np.asarray(data["context_mode"], dtype=np.str_).reshape(-1)[0])
            if "context_mode" in data.files and data["context_mode"].size
            else "none"
        )
        context_lags = (
            int(np.asarray(data["context_lags"]).reshape(-1)[0])
            if "context_lags" in data.files and data["context_lags"].size
            else (1 if context_mode == "lag1_delta" else 0)
        )
        base_input_dim = (
            int(np.asarray(data["base_input_dim"]).reshape(-1)[0])
            if "base_input_dim" in data.files and data["base_input_dim"].size
            else 2 * Nm + 2 + (2 if include_global_indicators else 0)
        )
        rollout_horizon = (
            int(np.asarray(data["rollout_horizon"]).reshape(-1)[0])
            if "rollout_horizon" in data.files and data["rollout_horizon"].size
            else 0
        )
        tail_start_fraction = (
            float(np.asarray(data["tail_start_fraction"]).reshape(-1)[0])
            if "tail_start_fraction" in data.files and data["tail_start_fraction"].size
            else 2.0 / 3.0
        )
        loss_backend = (
            str(np.asarray(data["loss_backend"], dtype=np.str_).reshape(-1)[0])
            if "loss_backend" in data.files and data["loss_backend"].size
            else None
        )
        lambda_q = (
            float(np.asarray(data["lambda_q"]).reshape(-1)[0])
            if "lambda_q" in data.files and data["lambda_q"].size
            else 1.0
        )
        lambda_E = (
            float(np.asarray(data["lambda_E"]).reshape(-1)[0])
            if "lambda_E" in data.files and data["lambda_E"].size
            else 0.0
        )
        lambda_dist = (
            float(np.asarray(data["lambda_dist"]).reshape(-1)[0])
            if "lambda_dist" in data.files and data["lambda_dist"].size
            else 0.0
        )
        lambda_tail = (
            float(np.asarray(data["lambda_tail"]).reshape(-1)[0])
            if "lambda_tail" in data.files and data["lambda_tail"].size
            else 0.0
        )
        lambda_neg = (
            float(np.asarray(data["lambda_neg"]).reshape(-1)[0])
            if "lambda_neg" in data.files and data["lambda_neg"].size
            else 0.0
        )
        lambda_reg = (
            float(np.asarray(data["lambda_reg"]).reshape(-1)[0])
            if "lambda_reg" in data.files and data["lambda_reg"].size
            else 0.0
        )
        online_v_probes = (
            int(np.asarray(data["online_v_probes"]).reshape(-1)[0])
            if "online_v_probes" in data.files and data["online_v_probes"].size
            else 0
        )
        stability_loss_definition = (
            str(np.asarray(data["stability_loss_definition"], dtype=np.str_).reshape(-1)[0])
            if "stability_loss_definition" in data.files and data["stability_loss_definition"].size
            else ("legacy_step_relative_v1" if train_objective == "stability_aware" else None)
        )

    return LearnedInterfaceClosure(
        params=params,
        Nm=Nm,
        k_scale=k_scale,
        nv_scale=nv_scale,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        hidden_width=hidden_width,
        res_blocks=res_blocks,
        Nv_targets=Nv_targets,
        train_regimes=train_regimes,
        teacher_backend=teacher_backend,
        teacher_Lx=teacher_Lx,
        teacher_Nx=teacher_Nx,
        teacher_Nv=teacher_Nv,
        teacher_vmin=teacher_vmin,
        teacher_vmax=teacher_vmax,
        teacher_dt=teacher_dt,
        teacher_proj_Nv=teacher_proj_Nv,
        include_global_indicators=include_global_indicators,
        n_low=n_low,
        training_mode=training_mode,
        train_objective=train_objective,
        context_mode=context_mode,
        context_lags=context_lags,
        base_input_dim=base_input_dim,
        rollout_horizon=rollout_horizon,
        tail_start_fraction=tail_start_fraction,
        loss_backend=loss_backend,
        lambda_q=lambda_q,
        lambda_E=lambda_E,
        lambda_dist=lambda_dist,
        lambda_tail=lambda_tail,
        lambda_neg=lambda_neg,
        lambda_reg=lambda_reg,
        online_v_probes=online_v_probes,
        stability_loss_definition=stability_loss_definition,
    )


def _hermitian_rfft_weights(nk: int) -> Array:
    if int(nk) <= 0:
        raise ValueError("nk must be positive")
    weights = jnp.ones((int(nk),), dtype=jnp.float64)
    if int(nk) > 2:
        weights = weights.at[1:-1].set(2.0)
    return weights


def learned_closure_global_indicators(
    a_hat: Array,
    k_arr: Array,
    *,
    n_low: int,
) -> Tuple[Array, Array]:
    """
    Return the low-cost global state summaries used by the learned interface closure:

        E = sum_{k != 0} |E_k|^2
        H_low = sum_k sum_{n=0}^{n_low} |C_{n,k}|^2

    The sums are evaluated on the stored rFFT half-spectrum with Hermitian weights so
    they remain comparable to the full-spectrum quantities for real-valued states.
    """
    a_hat = jnp.asarray(a_hat, dtype=jnp.complex128)
    k_arr = jnp.asarray(k_arr, dtype=jnp.float64)
    if a_hat.ndim != 2:
        raise ValueError(f"a_hat must have shape (Nv, Nk), got {a_hat.shape}")
    if k_arr.ndim != 1 or k_arr.shape[0] != a_hat.shape[1]:
        raise ValueError(f"k_arr must have shape ({a_hat.shape[1]},), got {k_arr.shape}")
    if int(n_low) < 0:
        raise ValueError("n_low must be nonnegative")

    weights = _hermitian_rfft_weights(a_hat.shape[1])

    e_hat = jnp.zeros((a_hat.shape[1],), dtype=jnp.complex128)
    if a_hat.shape[1] > 1:
        e_hat = e_hat.at[1:].set(1j * a_hat[0, 1:] / k_arr[1:])
    field_activity = jnp.sum(weights[1:] * (jnp.abs(e_hat[1:]) ** 2))

    n_hi = min(int(n_low), int(a_hat.shape[0]) - 1)
    low_energy = jnp.sum(weights[None, :] * (jnp.abs(a_hat[: n_hi + 1, :]) ** 2))
    return field_activity.astype(jnp.float64), low_energy.astype(jnp.float64)


def learned_closure_raw_base_features(
    a_hat: Array,
    k_arr: Array,
    Nv: int,
    learned: LearnedInterfaceClosure,
) -> Array:
    """Build the unstandardized current-state feature block for each nonzero rFFT mode."""
    Nv = int(Nv)
    Nm = int(learned.Nm)
    if Nm > Nv:
        raise ValueError(f"Learned closure Nm={Nm} exceeds Nv={Nv}")

    k_arr = jnp.asarray(k_arr, dtype=jnp.float64)
    if k_arr.shape[0] <= 1:
        return jnp.zeros((0, learned.raw_base_dim), dtype=jnp.float64)

    z = jnp.swapaxes(a_hat[Nv - Nm : Nv, 1:], 0, 1)
    feature_cols = [
        jnp.real(z),
        jnp.imag(z),
        jnp.abs(k_arr[1:])[:, None],
        jnp.full((z.shape[0], 1), float(Nv), dtype=jnp.float64),
    ]
    if learned.include_global_indicators:
        field_activity, low_energy = learned_closure_global_indicators(
            a_hat,
            k_arr,
            n_low=int(learned.n_low),
        )
        globals_mat = jnp.broadcast_to(
            jnp.array([field_activity, low_energy], dtype=jnp.float64)[None, :],
            (z.shape[0], 2),
        )
        feature_cols.append(globals_mat)
    return jnp.concatenate(feature_cols, axis=1)


def learned_closure_raw_features(
    a_hat: Array,
    k_arr: Array,
    Nv: int,
    learned: LearnedInterfaceClosure,
    *,
    a_hat_prev: Optional[Array] = None,
) -> Array:
    """Build the unstandardized learned-closure features for each nonzero rFFT mode."""
    current = learned_closure_raw_base_features(a_hat, k_arr, Nv, learned)
    if learned.context_mode == "none":
        return current
    if learned.context_mode == "lag1_delta":
        prev_state = a_hat if a_hat_prev is None else a_hat_prev
        previous = learned_closure_raw_base_features(prev_state, k_arr, Nv, learned)
        return jnp.concatenate([current, previous, current - previous], axis=1)
    raise ValueError(f"Unsupported context_mode={learned.context_mode!r}")


def scale_learned_closure_raw_features(
    raw_features: Array,
    learned: LearnedInterfaceClosure,
) -> Array:
    """Apply the k/Nv scaling used before global feature standardization."""
    raw_features = jnp.asarray(raw_features, dtype=jnp.float64)
    base_dim = int(learned.raw_base_dim)
    if learned.context_mode == "none":
        feats = raw_features
        feats = feats.at[:, 2 * int(learned.Nm)].divide(float(learned.k_scale))
        feats = feats.at[:, 2 * int(learned.Nm) + 1].divide(float(learned.nv_scale))
        return feats
    if learned.context_mode == "lag1_delta":
        current = raw_features[:, :base_dim]
        previous = raw_features[:, base_dim : 2 * base_dim]
        delta = raw_features[:, 2 * base_dim :]
        k_col = 2 * int(learned.Nm)
        nv_col = k_col + 1
        current = current.at[:, k_col].divide(float(learned.k_scale))
        current = current.at[:, nv_col].divide(float(learned.nv_scale))
        previous = previous.at[:, k_col].divide(float(learned.k_scale))
        previous = previous.at[:, nv_col].divide(float(learned.nv_scale))
        delta = current - previous
        return jnp.concatenate([current, previous, delta], axis=1)
    raise ValueError(f"Unsupported context_mode={learned.context_mode!r}")


def learned_interface_q_hat(
    a_hat: Array,
    k_arr: Array,
    Nv: int,
    learned: LearnedInterfaceClosure,
    *,
    a_hat_prev: Optional[Array] = None,
) -> Array:
    """
    Return the learned interface term q_k on the rFFT half-spectrum.

    In a real-valued simulation, the negative-k values are the conjugates of the
    stored positive modes, so only the nonnegative rFFT entries need to be modeled
    explicitly here.
    """
    Nv = int(Nv)
    Nm = int(learned.Nm)
    if Nm > Nv:
        raise ValueError(f"Learned closure Nm={Nm} exceeds Nv={Nv}")

    k_arr = jnp.asarray(k_arr, dtype=jnp.float64)
    q_hat = jnp.zeros((k_arr.shape[0],), dtype=jnp.complex128)
    if k_arr.shape[0] <= 1:
        return q_hat

    raw_features = learned_closure_raw_features(
        a_hat,
        k_arr,
        Nv,
        learned,
        a_hat_prev=a_hat_prev,
    )
    raw_features = scale_learned_closure_raw_features(raw_features, learned)
    pred = learned.predict_q_components(raw_features)
    if str(learned.training_mode) == "online_rollout":
        # Keep solver-in-the-loop optimization in a numerically stable regime
        # while preserving full JAX differentiability.
        clip = jnp.asarray([0.25, 0.75], dtype=jnp.float64)
        pred = clip * jnp.tanh(pred / clip)
    q_nonzero = (pred[:, 0] + 1j * pred[:, 1]).astype(jnp.complex128)
    return q_hat.at[1:].set(q_nonzero)


def learned_boundary_flux_hat(
    a_hat: Array,
    k_arr: Array,
    Nv: int,
    vth: float,
    learned: LearnedInterfaceClosure,
    *,
    a_hat_prev: Optional[Array] = None,
) -> Array:
    """Return the learned Hermite-boundary interface contribution, nonzero only on the last row."""
    del vth  # q_k is learned directly in physical units.
    q_hat = learned_interface_q_hat(a_hat, k_arr, Nv, learned, a_hat_prev=a_hat_prev)
    B_hat = jnp.zeros_like(a_hat, dtype=jnp.complex128)
    return B_hat.at[int(Nv) - 1].set(q_hat)


# =============================================================================
# Core numerical building blocks
# =============================================================================

def e_hat_from_rho_hat(
    rho_hat: Array,
    k_arr: Array,
    *,
    poisson_sign: float = +1.0,
) -> Array:
    """Compute electric-field Fourier coefficients from density coefficients."""
    rho_hat = jnp.asarray(rho_hat, dtype=jnp.complex128)
    k_arr = jnp.asarray(k_arr, dtype=jnp.float64)
    if rho_hat.ndim != 1:
        raise ValueError(f"rho_hat must have shape (Nk,), got {rho_hat.shape}")
    if k_arr.ndim != 1 or k_arr.shape[0] != rho_hat.shape[0]:
        raise ValueError(f"k_arr must have shape ({rho_hat.shape[0]},), got {k_arr.shape}")

    E_hat = jnp.zeros_like(rho_hat, dtype=jnp.complex128)
    return E_hat.at[1:].set((float(poisson_sign) * 1j) * rho_hat[1:] / k_arr[1:])


def e_hat_history_from_a_hat_history(
    a_hat_hist: Array,
    k_arr: Array,
    *,
    poisson_sign: float = +1.0,
) -> Array:
    """Convert an ``a_hat`` history of shape ``(Nt, Nv, Nk)`` into ``E_hat(t,k)``."""
    a_hat_hist = jnp.asarray(a_hat_hist, dtype=jnp.complex128)
    if a_hat_hist.ndim != 3:
        raise ValueError(f"a_hat_hist must have shape (Nt, Nv, Nk), got {a_hat_hist.shape}")
    return jax.vmap(
        lambda a_hat: e_hat_from_rho_hat(a_hat[0], k_arr, poisson_sign=poisson_sign)
    )(a_hat_hist)

def rfft_x(u_phys: Array) -> Array:
    """Real FFT over x axis=1, returning complex128."""
    return jnp.fft.rfft(u_phys, axis=1).astype(jnp.complex128)


def irfft_x(u_hat: Array, Nx: int) -> Array:
    """Inverse real FFT over x axis=1, returning float64."""
    return jnp.fft.irfft(u_hat, n=Nx, axis=1).astype(jnp.float64)
# Backwards-compatible aliases (older drafts used leading underscores)
_rfft_x = rfft_x
_irfft_x = irfft_x


def hermite_basis_phi_scaled(N: int, v: np.ndarray, vth: float = 1.0) -> np.ndarray:
    """
    Build the scaled AW Hermite basis φ_n^{(vth)}(v) on a velocity grid.

    The recurrence matches the visualization helper used by the nonlinear JAX demos:

        φ_n^{(vth)}(v) = (1 / vth) φ_n(v / vth)

    where φ_0 is the unit-mass Gaussian.
    """
    if N < 0:
        raise ValueError("N must be nonnegative")
    if vth <= 0.0:
        raise ValueError("vth must be positive")

    v = np.asarray(v, dtype=float)
    if N == 0:
        return np.zeros((0, v.size), dtype=float)

    xi = v / float(vth)
    w = np.exp(-0.5 * xi ** 2) / (math.sqrt(2.0 * math.pi) * float(vth))

    h = np.zeros((N, v.size), dtype=float)
    h[0] = 1.0
    if N > 1:
        h[1] = xi
    for n in range(1, N - 1):
        h[n + 1] = (xi / math.sqrt(n + 1)) * h[n] - math.sqrt(n / (n + 1)) * h[n - 1]
    return w * h


def hermite_basis_phi(N: int, v: np.ndarray) -> np.ndarray:
    """Build the unscaled AW Hermite basis φ_n(v) on a velocity grid."""
    return hermite_basis_phi_scaled(N, v, vth=1.0)



def _mask_23(Nx: int) -> Array:
    """
    2/3 dealias mask for an rFFT array of length Nk=Nx//2+1.

    We keep Fourier indices j <= floor(Nx/3), zero out j > floor(Nx/3).
    """
    Nk = Nx // 2 + 1
    cutoff = Nx // 3
    mask = jnp.ones((Nk,), dtype=bool)
    mask = mask.at[cutoff + 1 :].set(False)
    return mask


def tridiag_solve(sub: Array, diag: Array, sup: Array, rhs: Array) -> Array:
    """
    Thomas algorithm for a single tridiagonal system in JAX.

    Parameters
    ----------
    sub : (N-1,) complex
        Sub-diagonal entries A[i, i-1] for i=1..N-1.
    diag : (N,) complex
        Diagonal entries A[i,i].
    sup : (N-1,) complex
        Super-diagonal entries A[i, i+1] for i=0..N-2.
    rhs : (N,) complex
        Right-hand side.

    Returns
    -------
    x : (N,) complex
        Solution of Ax=rhs.
    """
    N = rhs.shape[0]
    z = jnp.zeros((), dtype=rhs.dtype)

    # Pad to length N for scan-friendly indexing.
    sub_p = jnp.concatenate([jnp.array([z], dtype=rhs.dtype), sub])      # (N,)
    sup_p = jnp.concatenate([sup, jnp.array([z], dtype=rhs.dtype)])      # (N,)

    # Forward sweep
    c0 = sup_p[0] / diag[0]
    d0 = rhs[0] / diag[0]

    def fwd(carry, i):
        c_prev, d_prev = carry
        den = diag[i] - sub_p[i] * c_prev
        c = sup_p[i] / den
        d = (rhs[i] - sub_p[i] * d_prev) / den
        return (c, d), (c, d)

    (_, _), hist = jax.lax.scan(fwd, (c0, d0), jnp.arange(1, N))
    c_full = jnp.concatenate([jnp.array([c0], dtype=rhs.dtype), hist[0]])
    d_full = jnp.concatenate([jnp.array([d0], dtype=rhs.dtype), hist[1]])

    # Back substitution
    xN = d_full[-1]

    def bwd(x_next, i):
        x_i = d_full[i] - c_full[i] * x_next
        return x_i, x_i

    _, x_rev = jax.lax.scan(bwd, xN, jnp.arange(N - 2, -1, -1))
    x0_to_N2 = x_rev[::-1]
    return jnp.concatenate([x0_to_N2, jnp.array([xN], dtype=rhs.dtype)])


def implicit_midpoint_jfnk_step(
    a_hat: Array,
    rhs_hat_fn: Callable[[Array], Array],
    dt: float,
    *,
    initial_guess: Optional[Array] = None,
    newton_tol: float = 1e-10,
    gmres_tol: float = 1e-10,
    max_newton_iter: int = 8,
    gmres_restart: Optional[int] = None,
    gmres_maxiter: Optional[int] = None,
    jvp_eps: float = 1e-8,
    post_step_filter: Optional[Array] = None,
) -> Tuple[Array, Dict[str, float]]:
    """
    One implicit-midpoint step solved with a Jacobian-free Newton-Krylov iteration.

    The nonlinear residual is

        R(y) = y - a^n - dt * F((a^n + y)/2),

    where ``F`` is provided by ``rhs_hat_fn``. The linear solve inside each Newton
    step uses GMRES with Jacobian-vector products approximated by directional
    finite differences of the residual.

    Notes
    -----
    This routine is intended for small benchmark problems where matching the
    paper's implicit-midpoint/JFNK algorithm matters more than raw throughput.
    """
    if spla is None:  # pragma: no cover
        raise RuntimeError(
            "implicit_midpoint_jfnk_step requires SciPy's sparse linear algebra support."
        )

    a0 = jnp.asarray(a_hat, dtype=jnp.complex128)
    shape = a0.shape
    dt = float(dt)

    def _flatten(arr: Array) -> np.ndarray:
        return np.asarray(arr, dtype=np.complex128).reshape(-1)

    def _reshape(vec: np.ndarray) -> Array:
        return jnp.asarray(np.asarray(vec, dtype=np.complex128).reshape(shape), dtype=jnp.complex128)

    def residual(state: Array) -> Array:
        midpoint = 0.5 * (a0 + state)
        return state - a0 - dt * rhs_hat_fn(midpoint)

    if initial_guess is None:
        y = a0 + dt * rhs_hat_fn(a0)
    else:
        y = jnp.asarray(initial_guess, dtype=jnp.complex128)

    residual_target = math.sqrt(float(np.prod(shape))) * max(
        float(newton_tol),
        100.0 * float(gmres_tol),
    )
    last_residual = float("inf")
    last_step_norm = float("inf")
    total_gmres_iters = 0

    for newton_iter in range(1, int(max_newton_iter) + 1):
        r = residual(y)
        r_np = _flatten(r)
        last_residual = float(np.linalg.norm(r_np))
        if last_residual <= residual_target:
            break

        def matvec(v_np: np.ndarray) -> np.ndarray:
            v_norm = max(1.0, float(np.linalg.norm(v_np)))
            if v_norm == 0.0:
                return np.zeros_like(v_np, dtype=np.complex128)
            v_j = _reshape(v_np)
            try:
                _, jv = jax.jvp(residual, (y,), (v_j,))
                return np.array(_flatten(jv), dtype=np.complex128, copy=True)
            except Exception:
                eps = float(jvp_eps) / v_norm
                rp = residual(y + eps * v_j)
                return np.array((_flatten(rp) - r_np) / eps, dtype=np.complex128, copy=True)

        linop = spla.LinearOperator(
            (r_np.size, r_np.size),
            matvec=matvec,
            dtype=np.complex128,
        )

        gmres_iters = [0]

        def gmres_callback(_residual: np.ndarray) -> None:
            gmres_iters[0] += 1

        try:
            delta_np, info = spla.gmres(
                linop,
                -r_np,
                restart=gmres_restart,
                maxiter=gmres_maxiter,
                rtol=float(gmres_tol),
                atol=0.0,
                callback=gmres_callback,
                callback_type="legacy",
            )
        except TypeError:  # pragma: no cover
            delta_np, info = spla.gmres(
                linop,
                -r_np,
                restart=gmres_restart,
                maxiter=gmres_maxiter,
                tol=float(gmres_tol),
                atol=0.0,
                callback=gmres_callback,
                callback_type="legacy",
            )

        total_gmres_iters += gmres_iters[0]
        if info != 0:
            raise RuntimeError(
                f"GMRES failed during implicit_midpoint_jfnk_step with info={info} "
                f"at Newton iteration {newton_iter}."
            )

        last_step_norm = float(np.linalg.norm(delta_np))
        y = y + _reshape(delta_np)

    final_residual = float(np.linalg.norm(_flatten(residual(y))))
    if final_residual > residual_target:
        raise RuntimeError(
            "implicit_midpoint_jfnk_step did not converge: "
            f"final residual={final_residual:.3e}, target={residual_target:.3e}"
        )

    if post_step_filter is not None:
        y = jnp.asarray(post_step_filter, dtype=jnp.float64)[:, None] * y

    info = {
        "newton_iters": float(newton_iter),
        "gmres_iters": float(total_gmres_iters),
        "residual_norm": final_residual,
        "residual_target": residual_target,
        "step_norm": last_step_norm,
    }
    return y, info


# =============================================================================
# Fourier–Hermite IMEX CNAB2 integrator
# =============================================================================

@dataclass
class FourierHermiteIMEX:
    """
    IMEX CNAB2 integrator for systems of the form:

        d/dt a_hat = L(a_hat) + N(a_hat, t)

    where:
      - a_hat has shape (Nv, Nk) with Nk = Nx//2+1 (rFFT in x).
      - L is the linear streaming operator, tridiagonal in Hermite index.
      - N is an explicit term (usually electric-field acceleration + optional damping).

    The CNAB2 update is:

        (I - dt/2 L) a^{n+1} = (I + dt/2 L) a^n + dt*( 3/2 N^n - 1/2 N^{n-1} )

    This class provides:
      - precomputed per-k tridiagonal matrices for the implicit solve
      - batched Thomas solve via jax.vmap(tridiag_solve)
      - utilities for Poisson solve E from density a0

    Parameters
    ----------
    Nx : int
        Number of x grid points.
    Nv : int
        Number of Hermite modes.
    Lx : float
        Domain length in x.
    dt : float
        Time step size.
    vth : float
        Hermite thermal scaling. vth=1 recovers the unscaled basis.
        Streaming operator scales by vth and acceleration term scales by 1/vth.
    dealias_23 : bool
        If True, apply 2/3 dealiasing mask to *all* Fourier-space arrays passed
        through `apply_mask_hat`.
    closure : Optional[NonlocalClosure]
        If provided with Nm=1, modifies the last Hermite mode via the closure substitution.
        (Time-stepping supports Nm=1 only.)
    """
    Nx: int
    Nv: int
    Lx: float
    dt: float
    vth: float = 1.0
    dealias_23: bool = True
    closure: Optional[NonlocalClosure] = None

    def __post_init__(self):
        if self.Nx <= 0 or self.Nv <= 0:
            raise ValueError("Nx and Nv must be positive")
        if self.vth <= 0:
            raise ValueError("vth must be positive")

        dx = float(self.Lx) / float(self.Nx)
        self.dx = dx

        self.x = jnp.linspace(0.0, float(self.Lx), int(self.Nx), endpoint=False)
        self.k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(int(self.Nx), d=dx).astype(jnp.float64)
        self.Nk = int(self.Nx // 2 + 1)

        self.sqrt_n = jnp.sqrt(jnp.arange(int(self.Nv), dtype=jnp.float64))
        self.sqrt_np1 = jnp.sqrt(jnp.arange(1, int(self.Nv) + 1, dtype=jnp.float64))

        # 2/3 dealias mask (rFFT length Nk)
        self.mask = _mask_23(self.Nx) if self.dealias_23 else None

        # --- Precompute tridiagonal coefficients for the implicit CN solve ---
        dt_half = 0.5 * float(self.dt)
        s = jnp.sqrt(jnp.arange(1, int(self.Nv), dtype=jnp.float64))              # sqrt(1..Nv-1)
        val = (1j * dt_half * self.k_arr * float(self.vth)).astype(jnp.complex128)  # (Nk,)

        # sub/sup: (Nk, Nv-1)
        self.sub = (val[:, None] * s[None, :]).astype(jnp.complex128)
        self.sup = self.sub.copy()
        self.diag = jnp.ones((self.Nk, int(self.Nv)), dtype=jnp.complex128)

        # Optional Nm=1 closure -> diagonal modification on last Hermite mode
        self._closure_Ldiag_last = None
        if self.closure is not None:
            Nm = int(self.closure.Nm)
            if Nm != 1:
                raise ValueError("Time-stepping integrator supports NonlocalClosure only for Nm=1.")
            mu = jnp.asarray(self.closure.mu_tail[0], dtype=jnp.float64)
            signk = jnp.sign(self.k_arr)
            # L_diag_last = k*vth*sqrt(Nv)*mu*sign(k)
            Ldiag_last = (self.k_arr * float(self.vth) * jnp.sqrt(float(self.Nv)) * mu * signk)
            self._closure_Ldiag_last = Ldiag_last.astype(jnp.complex128)
            self.diag = self.diag.at[:, int(self.Nv) - 1].set(1.0 - dt_half * self._closure_Ldiag_last)

        # Batched solver (vmap over k)
        self._solve_batched = jax.vmap(tridiag_solve, in_axes=(0, 0, 0, 0), out_axes=0)

    # -------------------------
    # Fourier-space utilities
    # -------------------------

    def apply_mask_hat(self, a_hat: Array) -> Array:
        """Apply the 2/3 dealias mask (if enabled) to an array shaped (..., Nk)."""
        if self.mask is None:
            return a_hat
        # broadcast mask onto leading axes
        return jnp.where(self.mask, a_hat, 0.0)

    # -------------------------
    # Linear streaming operator
    # -------------------------

    def streaming_hat(self, a_hat: Array) -> Array:
        """
        Streaming operator in Fourier-x / Hermite-v space:

            (L a)_n = -i k vth ( sqrt(n) a_{n-1} + sqrt(n+1) a_{n+1} )

        plus optional Nm=1 nonlocal closure contribution on the last mode.
        """
        a_down = jnp.concatenate([jnp.zeros_like(a_hat[:1]), a_hat[:-1]], axis=0)  # a_{n-1}
        a_up = jnp.concatenate([a_hat[1:], jnp.zeros_like(a_hat[:1])], axis=0)     # a_{n+1}

        out = -1j * (self.k_arr * float(self.vth))[None, :] * (
            self.sqrt_n[:, None] * a_down + self.sqrt_np1[:, None] * a_up
        )

        if self._closure_Ldiag_last is not None:
            out = out.at[int(self.Nv) - 1].add(self._closure_Ldiag_last * a_hat[int(self.Nv) - 1])

        return out

    # -------------------------
    # Poisson solver utilities
    # -------------------------

    def E_hat_from_rho_hat(self, rho_hat: Array, poisson_sign: float = +1.0) -> Array:
        """
        Compute E_hat from rho_hat using
            E_k = poisson_sign * i * rho_k / k,   k!=0
            E_0 = 0,
        so that i k E_k = -poisson_sign * rho_k.
        """
        E_hat = jnp.zeros_like(rho_hat)
        E_hat = E_hat.at[1:].set((poisson_sign * 1j) * rho_hat[1:] / self.k_arr[1:])
        return E_hat

    def E_phys_from_a_hat(self, a_hat: Array, poisson_sign: float = +1.0) -> Array:
        """Compute electric field E(x) in physical space from a_hat[0,k]."""
        rho_hat = a_hat[0]
        E_hat = self.E_hat_from_rho_hat(rho_hat, poisson_sign=poisson_sign)
        return jnp.fft.irfft(E_hat, n=int(self.Nx)).astype(jnp.float64)

    def electric_energy(self, a_hat: Array, poisson_sign: float = +1.0) -> Array:
        """Return 0.5 * ∫ E^2 dx (discrete trapezoid == dx * sum)."""
        E = self.E_phys_from_a_hat(a_hat, poisson_sign=poisson_sign)
        return 0.5 * float(self.dx) * jnp.sum(E * E)

    # -------------------------
    # IMEX CNAB2 step
    # -------------------------

    def implicit_solve(self, rhs_hat: Array) -> Array:
        """
        Solve (I - dt/2 L) a_new = rhs_hat for all Fourier modes k (batched).
        """
        # vmap solver expects (Nk, Nv), so transpose:
        rhs_b = jnp.swapaxes(rhs_hat, 0, 1)  # (Nk, Nv)
        sol_b = self._solve_batched(self.sub, self.diag, self.sup, rhs_b)  # (Nk, Nv)
        return jnp.swapaxes(sol_b, 0, 1)  # (Nv, Nk)

    def step_cnab2(
        self,
        a_hat: Array,
        N_hat: Array,
        N_hat_prev: Array,
        *,
        extra_hat: Optional[Array] = None,
        extra_hat_prev: Optional[Array] = None,
        hermite_filter: Optional[Array] = None,
    ) -> Array:
        """
        One CNAB2 step given current state a_hat and explicit terms N_hat, N_hat_prev.

        Parameters
        ----------
        a_hat : (Nv, Nk) complex
            Current state.
        N_hat : (Nv, Nk) complex
            Explicit term at the current time.
        N_hat_prev : (Nv, Nk) complex
            Explicit term at the previous time.
        extra_hat : Optional[(Nv, Nk)] complex
            Additional explicit term, typically a learned Hermite-boundary flux.
        extra_hat_prev : Optional[(Nv, Nk)] complex
            Previous-step value of `extra_hat`.
        hermite_filter : Optional[(Nv,)]
            If provided, multiplies the updated a_hat by this per-mode factor.

        Returns
        -------
        a_hat_new : (Nv, Nk) complex
        """
        dt = float(self.dt)
        dt_half = 0.5 * dt

        if extra_hat is None:
            extra_hat = jnp.zeros_like(a_hat)
        if extra_hat_prev is None:
            extra_hat_prev = jnp.zeros_like(a_hat)

        La = self.streaming_hat(a_hat)
        rhs = a_hat + dt_half * La + dt * (
            1.5 * (N_hat + extra_hat) - 0.5 * (N_hat_prev + extra_hat_prev)
        )
        rhs = self.apply_mask_hat(rhs)

        a_new = self.implicit_solve(rhs)
        a_new = self.apply_mask_hat(a_new)

        if hermite_filter is not None:
            a_new = hermite_filter[:, None] * a_new

        a_new = self.apply_mask_hat(a_new)
        return a_new


# =============================================================================
# Small helpers for explicit-term construction (shared by sims/benchmarks)
# =============================================================================

def hermite_damping_term(a_phys: Array, damping_rates: Array) -> Array:
    """
    Apply a Hermite-diagonal damping term in physical space:

        d/dt a_n += -gamma_n a_n

    This returns N_phys = -gamma[:,None] * a_phys.

    Using physical space here is convenient because a_phys is typically already available
    when forming nonlinear products E(x)*something(x).
    """
    return -(damping_rates[:, None] * a_phys)
