"""Interface closure construction, checkpoint compatibility, and statistics caches."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from model.train.interface_flux_data import sha256_json
from vpml.core import (
    Array,
    LearnedInterfaceClosure,
    load_learned_interface_closure_npz,
    normalize_teacher_backend_name,
)
from vpml.training.interface_configuration import (
    CANONICAL_NV_TARGETS,
    CANONICAL_REGIMES,
    EXACT_ROLLOUT_PRECISION_FLOAT64,
    INTERFACE_FLUX_ROLLOUT_LOSS_BACKEND,
    INTERFACE_FLUX_ROLLOUT_OBJECTIVE,
    INTERFACE_FLUX_ROLLOUT_TRAINING_MODE,
    exact_rollout_precision_dtypes,
)


def cast_learned_closure_for_rollout(
    learned: LearnedInterfaceClosure,
    *,
    precision: str,
) -> LearnedInterfaceClosure:
    real_dtype, _ = exact_rollout_precision_dtypes(str(precision))
    if str(precision) == EXACT_ROLLOUT_PRECISION_FLOAT64:
        return learned
    params = jax.tree_util.tree_map(lambda value: jnp.asarray(value, dtype=real_dtype), learned.params)
    return replace(
        learned,
        params=params,
        input_mean=jnp.asarray(learned.input_mean, dtype=real_dtype),
        input_std=jnp.asarray(learned.input_std, dtype=real_dtype),
        target_mean=jnp.asarray(learned.target_mean, dtype=real_dtype),
        target_std=jnp.asarray(learned.target_std, dtype=real_dtype),
    )


def default_exact_k_scale(k_arr: np.ndarray) -> float:
    k_nonzero = np.asarray(k_arr, dtype=np.float64)[1:]
    if int(k_nonzero.shape[0]) == 0:
        return 1.0
    return max(float(np.max(np.abs(k_nonzero))), 1.0)


def build_learned_interface_closure(
    *,
    params: Dict[str, Array],
    Nm: int,
    k_scale: float,
    nv_scale: float,
    stats: Dict[str, np.ndarray],
    hidden_width: int,
    res_blocks: int,
    equilibrium_centered: bool = False,
    complex_normalization_mode: str = "componentwise",
    translation_augmented: bool = False,
    Nv_targets: Sequence[int],
    train_regimes: Sequence[str],
    teacher_backend: str,
    teacher_Lx: float,
    teacher_Nx: int,
    teacher_Nv: int,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    teacher_proj_Nv: Optional[int],
    projection_quadrature_Nv: Optional[int],
    n_low: int,
    rollout_horizon: int,
    rollout_Nx: Optional[int] = None,
) -> LearnedInterfaceClosure:
    return LearnedInterfaceClosure(
        params=params,
        Nm=Nm,
        k_scale=k_scale,
        nv_scale=nv_scale,
        input_mean=jnp.asarray(stats["input_mean"], dtype=jnp.float64),
        input_std=jnp.asarray(stats["input_std"], dtype=jnp.float64),
        target_mean=jnp.asarray(stats["target_mean"], dtype=jnp.float64),
        target_std=jnp.asarray(stats["target_std"], dtype=jnp.float64),
        hidden_width=int(hidden_width),
        res_blocks=int(res_blocks),
        equilibrium_centered=bool(equilibrium_centered),
        complex_normalization_mode=str(complex_normalization_mode),
        translation_augmented=bool(translation_augmented),
        Nv_targets=tuple(int(v) for v in Nv_targets),
        train_regimes=tuple(str(v) for v in train_regimes),
        teacher_backend=str(normalize_teacher_backend_name(teacher_backend)),
        teacher_Lx=float(teacher_Lx),
        teacher_Nx=int(teacher_Nx),
        rollout_Nx=(
            int(teacher_Nx) if rollout_Nx is None else int(rollout_Nx)
        ),
        teacher_Nv=int(teacher_Nv),
        teacher_vmin=float(teacher_vmin),
        teacher_vmax=float(teacher_vmax),
        teacher_dt=float(teacher_dt),
        teacher_proj_Nv=None if teacher_proj_Nv is None else int(teacher_proj_Nv),
        projection_quadrature_Nv=(
            None
            if projection_quadrature_Nv is None
            else int(projection_quadrature_Nv)
        ),
        include_global_indicators=True,
        n_low=int(n_low),
        training_mode=INTERFACE_FLUX_ROLLOUT_TRAINING_MODE,
        train_objective=INTERFACE_FLUX_ROLLOUT_OBJECTIVE,
        context_mode="none",
        context_lags=0,
        base_input_dim=2 * int(Nm) + 4,
        rollout_horizon=int(rollout_horizon),
        loss_backend=INTERFACE_FLUX_ROLLOUT_LOSS_BACKEND,
    )


def _load_init_checkpoint_for_interface_closure(
    init_checkpoint: Path,
    *,
    Nm: int,
    hidden_width: int,
    res_blocks: int,
    Nv_targets: Sequence[int],
    context_mode: str,
    equilibrium_centered: Optional[bool] = None,
    complex_normalization_mode: Optional[str] = None,
    ic_manifest_sha256: Optional[str] = None,
    rollout_Nx: Optional[int] = None,
) -> Tuple[Dict[str, Array], Dict[str, np.ndarray], float, float]:
    learned = load_learned_interface_closure_npz(init_checkpoint)
    expected_targets = tuple(int(v) for v in Nv_targets)
    actual_targets = tuple(int(v) for v in learned.Nv_targets)
    if actual_targets != expected_targets:
        raise ValueError(
            f"--init-checkpoint Nv_targets={actual_targets} does not match requested Nv-targets={expected_targets}"
        )
    if int(learned.Nm) != int(Nm):
        raise ValueError(f"--init-checkpoint Nm={int(learned.Nm)} does not match requested Nm={int(Nm)}")
    if int(learned.hidden_width) != int(hidden_width):
        raise ValueError(
            f"--init-checkpoint hidden_width={int(learned.hidden_width)} does not match requested hidden_width={int(hidden_width)}"
        )
    if int(learned.res_blocks) != int(res_blocks):
        raise ValueError(
            f"--init-checkpoint res_blocks={int(learned.res_blocks)} does not match requested res_blocks={int(res_blocks)}"
        )
    if str(learned.context_mode) != str(context_mode):
        raise ValueError(
            f"--init-checkpoint context_mode={learned.context_mode!r} does not match requested context_mode={context_mode!r}"
        )
    if (
        equilibrium_centered is not None
        and bool(learned.equilibrium_centered) != bool(equilibrium_centered)
    ):
        raise ValueError(
            "--init-checkpoint equilibrium_centered metadata does not match the requested closure behavior"
        )
    if (
        complex_normalization_mode is not None
        and str(learned.complex_normalization_mode) != str(complex_normalization_mode)
    ):
        raise ValueError(
            "--init-checkpoint complex_normalization_mode metadata does not match the requested normalization"
        )
    if (
        ic_manifest_sha256 is not None
        and learned.ic_manifest_sha256 is not None
        and str(learned.ic_manifest_sha256) != str(ic_manifest_sha256)
    ):
        raise ValueError(
            "--init-checkpoint IC manifest does not match the requested training manifest"
        )
    if (
        rollout_Nx is not None
        and learned.rollout_Nx is not None
        and int(learned.rollout_Nx) != int(rollout_Nx)
    ):
        raise ValueError(
            f"--init-checkpoint rollout_Nx={int(learned.rollout_Nx)} does not "
            f"match requested rollout_Nx={int(rollout_Nx)}"
        )
    params = {
        key: jnp.asarray(value, dtype=jnp.float64)
        for key, value in learned.params.items()
    }
    stats = {
        "input_mean": np.asarray(learned.input_mean, dtype=np.float64),
        "input_std": np.asarray(learned.input_std, dtype=np.float64),
        "target_mean": np.asarray(learned.target_mean, dtype=np.float64),
        "target_std": np.asarray(learned.target_std, dtype=np.float64),
    }
    return params, stats, float(learned.k_scale), float(learned.nv_scale)


def _derived_statistics_path(
    reference_cache_dir: Path,
    *,
    manifest_sha256: str,
    rollout_horizon: int,
    history_stride: int,
    Nm: int,
    n_low: int,
    k_scale: float,
    nv_scale: float,
    rollout_Nx: int,
) -> Path:
    configuration = {
        "format": "interface_flux_training_statistics_v1",
        "manifest_sha256": str(manifest_sha256),
        "rollout_horizon": int(rollout_horizon),
        "history_stride": int(history_stride),
        "Nm": int(Nm),
        "n_low": int(n_low),
        "k_scale": float(k_scale),
        "nv_scale": float(nv_scale),
        "rollout_Nx": int(rollout_Nx),
        "Nv_targets": list(CANONICAL_NV_TARGETS),
        "normalization": "phase_isotropic",
        "regime_scaling": "fixed_training_only",
    }
    return (
        Path(reference_cache_dir)
        / "derived"
        / f"{sha256_json(configuration)[:20]}.npz"
    )


def _load_derived_statistics(
    path: Path,
) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, float]]]:
    if not Path(path).exists():
        return None
    with np.load(path) as payload:
        required = {
            "input_mean",
            "input_std",
            "target_mean",
            "target_std",
            "regimes",
            "regime_loss_stds",
        }
        if not required.issubset(payload.files):
            return None
        regimes = tuple(
            str(value)
            for value in np.asarray(payload["regimes"], dtype=np.str_).tolist()
        )
        if regimes != CANONICAL_REGIMES:
            return None
        stats = {
            key: np.asarray(payload[key], dtype=np.float64)
            for key in ("input_mean", "input_std", "target_mean", "target_std")
        }
        scale_values = np.asarray(payload["regime_loss_stds"], dtype=np.float64)
        if scale_values.shape != (len(CANONICAL_REGIMES),):
            return None
        scales = {
            regime: float(scale_values[index])
            for index, regime in enumerate(CANONICAL_REGIMES)
        }
    return stats, scales


def _save_derived_statistics(
    path: Path,
    *,
    stats: Dict[str, np.ndarray],
    regime_loss_stds: Dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            **{
                key: np.asarray(stats[key], dtype=np.float64)
                for key in ("input_mean", "input_std", "target_mean", "target_std")
            },
            regimes=np.asarray(CANONICAL_REGIMES, dtype=np.str_),
            regime_loss_stds=np.asarray(
                [regime_loss_stds[regime] for regime in CANONICAL_REGIMES],
                dtype=np.float64,
            ),
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
