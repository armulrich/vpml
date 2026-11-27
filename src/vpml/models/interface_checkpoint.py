"""NPZ serialization and legacy metadata adaptation for interface closures."""

from __future__ import annotations

from typing import Dict
import numpy as np
import os

from .._jax_precision import enable_x64

enable_x64()

import jax.numpy as jnp

from .interface_closure import (
    INTERFACE_FLUX_CHECKPOINT_SCHEMA_VERSION,
    INTERFACE_FLUX_LOSS_BACKEND,
    INTERFACE_FLUX_OBJECTIVE,
    INTERFACE_FLUX_TRAINING_MODE,
    LearnedInterfaceClosure,
    _LEGACY_INTERFACE_FLUX_TRIPLE,
    normalize_teacher_backend_name,
)


def save_learned_interface_closure_npz(
    path: str | os.PathLike[str],
    learned: LearnedInterfaceClosure,
) -> None:
    """Save a learned interface-closure checkpoint in NPZ format."""
    payload: Dict[str, np.ndarray] = {
        "closure_kind": np.array(["interface_closure"], dtype=np.str_),
        "metadata_schema_version": np.array(
            [INTERFACE_FLUX_CHECKPOINT_SCHEMA_VERSION], dtype=np.int32
        ),
        "Nm": np.array([int(learned.Nm)], dtype=np.int32),
        "k_scale": np.array([float(learned.k_scale)], dtype=np.float64),
        "nv_scale": np.array([float(learned.nv_scale)], dtype=np.float64),
        "input_mean": np.asarray(learned.input_mean, dtype=np.float64),
        "input_std": np.asarray(learned.input_std, dtype=np.float64),
        "target_mean": np.asarray(learned.target_mean, dtype=np.float64),
        "target_std": np.asarray(learned.target_std, dtype=np.float64),
        "hidden_width": np.array([int(learned.hidden_width)], dtype=np.int32),
        "res_blocks": np.array([int(learned.res_blocks)], dtype=np.int32),
        "equilibrium_centered": np.array([int(bool(learned.equilibrium_centered))], dtype=np.int32),
        "complex_normalization_mode": np.array([str(learned.complex_normalization_mode)], dtype=np.str_),
        "translation_augmented": np.array([int(bool(learned.translation_augmented))], dtype=np.int32),
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
        "loss_backend": np.array([str(learned.loss_backend)], dtype=np.str_),
    }
    if learned.ic_manifest_sha256 is not None:
        payload["ic_manifest_sha256"] = np.array(
            [str(learned.ic_manifest_sha256)], dtype=np.str_
        )
    if learned.training_ic_count is not None:
        payload["training_ic_count"] = np.array(
            [int(learned.training_ic_count)], dtype=np.int32
        )
    if learned.heldout_ic_count is not None:
        payload["heldout_ic_count"] = np.array(
            [int(learned.heldout_ic_count)], dtype=np.int32
        )
    if learned.teacher_Lx is not None:
        payload["teacher_Lx"] = np.array([float(learned.teacher_Lx)], dtype=np.float64)
    if learned.teacher_Nx is not None:
        payload["teacher_Nx"] = np.array([int(learned.teacher_Nx)], dtype=np.int32)
    if learned.rollout_Nx is not None:
        payload["rollout_Nx"] = np.array([int(learned.rollout_Nx)], dtype=np.int32)
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
    if learned.projection_quadrature_Nv is not None:
        payload["projection_quadrature_Nv"] = np.array(
            [int(learned.projection_quadrature_Nv)], dtype=np.int32
        )
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
        schema_version = (
            int(np.asarray(data["metadata_schema_version"]).reshape(-1)[0])
            if "metadata_schema_version" in data.files
            and data["metadata_schema_version"].size
            else 0
        )
        if schema_version > INTERFACE_FLUX_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema {schema_version} is newer than supported schema "
                f"{INTERFACE_FLUX_CHECKPOINT_SCHEMA_VERSION}"
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
        equilibrium_centered = (
            bool(np.asarray(data["equilibrium_centered"]).reshape(-1)[0])
            if "equilibrium_centered" in data.files and data["equilibrium_centered"].size
            else False
        )
        complex_normalization_mode = (
            str(np.asarray(data["complex_normalization_mode"], dtype=np.str_).reshape(-1)[0])
            if "complex_normalization_mode" in data.files and data["complex_normalization_mode"].size
            else "componentwise"
        )
        translation_augmented = (
            bool(np.asarray(data["translation_augmented"]).reshape(-1)[0])
            if "translation_augmented" in data.files and data["translation_augmented"].size
            else False
        )

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
        rollout_Nx = (
            int(np.asarray(data["rollout_Nx"]).reshape(-1)[0])
            if "rollout_Nx" in data.files and data["rollout_Nx"].size
            else teacher_Nx
        )
        teacher_Nv = None if "teacher_Nv" not in data.files or not data["teacher_Nv"].size else int(np.asarray(data["teacher_Nv"]).reshape(-1)[0])
        teacher_vmin = None if "teacher_vmin" not in data.files or not data["teacher_vmin"].size else float(np.asarray(data["teacher_vmin"]).reshape(-1)[0])
        teacher_vmax = None if "teacher_vmax" not in data.files or not data["teacher_vmax"].size else float(np.asarray(data["teacher_vmax"]).reshape(-1)[0])
        teacher_dt = None if "teacher_dt" not in data.files or not data["teacher_dt"].size else float(np.asarray(data["teacher_dt"]).reshape(-1)[0])
        teacher_proj_Nv = None if "teacher_proj_Nv" not in data.files or not data["teacher_proj_Nv"].size else int(np.asarray(data["teacher_proj_Nv"]).reshape(-1)[0])
        projection_quadrature_Nv = (
            None
            if "projection_quadrature_Nv" not in data.files
            or not data["projection_quadrature_Nv"].size
            else int(np.asarray(data["projection_quadrature_Nv"]).reshape(-1)[0])
        )
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
        loss_backend = (
            str(np.asarray(data["loss_backend"], dtype=np.str_).reshape(-1)[0])
            if "loss_backend" in data.files and data["loss_backend"].size
            else None
        )
        ic_manifest_sha256 = (
            str(np.asarray(data["ic_manifest_sha256"], dtype=np.str_).reshape(-1)[0])
            if "ic_manifest_sha256" in data.files
            and data["ic_manifest_sha256"].size
            else None
        )
        training_ic_count = (
            int(np.asarray(data["training_ic_count"]).reshape(-1)[0])
            if "training_ic_count" in data.files and data["training_ic_count"].size
            else None
        )
        heldout_ic_count = (
            int(np.asarray(data["heldout_ic_count"]).reshape(-1)[0])
            if "heldout_ic_count" in data.files and data["heldout_ic_count"].size
            else None
        )
        metadata_triple = (training_mode, train_objective, loss_backend)
        if metadata_triple == _LEGACY_INTERFACE_FLUX_TRIPLE:
            training_mode = INTERFACE_FLUX_TRAINING_MODE
            train_objective = INTERFACE_FLUX_OBJECTIVE
            loss_backend = INTERFACE_FLUX_LOSS_BACKEND
        elif metadata_triple != (
            INTERFACE_FLUX_TRAINING_MODE,
            INTERFACE_FLUX_OBJECTIVE,
            INTERFACE_FLUX_LOSS_BACKEND,
        ):
            raise ValueError(
                "Unsupported learned-closure trainer metadata "
                f"{metadata_triple!r}; only canonical interface-flux and legacy "
                "solver-embedded exact-q checkpoints are supported"
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
        equilibrium_centered=equilibrium_centered,
        complex_normalization_mode=complex_normalization_mode,
        translation_augmented=translation_augmented,
        Nv_targets=Nv_targets,
        train_regimes=train_regimes,
        teacher_backend=teacher_backend,
        teacher_Lx=teacher_Lx,
        teacher_Nx=teacher_Nx,
        rollout_Nx=rollout_Nx,
        teacher_Nv=teacher_Nv,
        teacher_vmin=teacher_vmin,
        teacher_vmax=teacher_vmax,
        teacher_dt=teacher_dt,
        teacher_proj_Nv=teacher_proj_Nv,
        projection_quadrature_Nv=projection_quadrature_Nv,
        include_global_indicators=include_global_indicators,
        n_low=n_low,
        training_mode=training_mode,
        train_objective=train_objective,
        context_mode=context_mode,
        context_lags=context_lags,
        base_input_dim=base_input_dim,
        rollout_horizon=rollout_horizon,
        loss_backend=loss_backend,
        ic_manifest_sha256=ic_manifest_sha256,
        training_ic_count=training_ic_count,
        heldout_ic_count=heldout_ic_count,
    )
