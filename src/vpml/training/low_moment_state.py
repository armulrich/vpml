"""Compatible low-moment parameter checkpoints and exact optimizer continuation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np

from vpml.models.moment_parameters import (
    init_burles_latent_fno_params,
    init_causal_spacetime_operator_params,
    init_explicit_window_params,
    init_spectral_memory_params,
    init_window_fno_params,
)


def _save_checkpoint(path: Path, params, metadata: Mapping[str, object], stats):
    payload = {f"param_{key}": np.asarray(value) for key, value in params.items()}
    payload.update({f"stat_{key}": np.asarray(value) for key, value in stats.items()})
    payload["metadata_json"] = np.array([json.dumps(dict(metadata), sort_keys=True)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _load_checkpoint(path: Path):
    with np.load(Path(path), allow_pickle=False) as payload:
        params = {
            key.removeprefix("param_"): jnp.asarray(payload[key])
            for key in payload.files
            if key.startswith("param_")
        }
        stats = {
            key.removeprefix("stat_"): np.asarray(payload[key])
            for key in payload.files
            if key.startswith("stat_")
        }
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).reshape(-1)[0]))
    if not params or not stats:
        raise ValueError(f"Incomplete low-moment checkpoint: {path}")
    return params, metadata, stats


def _save_training_state(
    path: Path,
    *,
    params,
    optimizer,
    rng: np.random.Generator,
    global_epoch: int,
    loss_ema: Optional[float],
    best_val: float,
    histories: Mapping[str, object],
    in_epoch: Optional[Mapping[str, object]] = None,
) -> None:
    payload = {f"param_{key}": np.asarray(value) for key, value in params.items()}
    payload.update(
        {f"optimizer_m_{key}": np.asarray(value) for key, value in optimizer["m"].items()}
    )
    payload.update(
        {f"optimizer_v_{key}": np.asarray(value) for key, value in optimizer["v"].items()}
    )
    payload["optimizer_step"] = np.asarray(optimizer["step"])
    payload["global_epoch"] = np.asarray(global_epoch, dtype=np.int64)
    payload["loss_ema"] = np.asarray(
        np.nan if loss_ema is None else loss_ema, dtype=np.float64
    )
    payload["best_val"] = np.asarray(best_val, dtype=np.float64)
    payload["rng_state_json"] = np.asarray(
        [json.dumps(rng.bit_generator.state, sort_keys=True)]
    )
    for key, value in histories.items():
        payload[f"history_{key}"] = np.asarray(value)
    if in_epoch is not None:
        for key, value in in_epoch.items():
            if key == "rng_state":
                payload["in_epoch_rng_state_json"] = np.asarray(
                    [json.dumps(value, sort_keys=True)]
                )
            else:
                payload[f"in_epoch_{key}"] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, path)


def _load_training_state(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        params = {
            key.removeprefix("param_"): jnp.asarray(payload[key])
            for key in payload.files
            if key.startswith("param_")
        }
        optimizer = {
            "step": jnp.asarray(payload["optimizer_step"]),
            "m": {
                key.removeprefix("optimizer_m_"): jnp.asarray(payload[key])
                for key in payload.files
                if key.startswith("optimizer_m_")
            },
            "v": {
                key.removeprefix("optimizer_v_"): jnp.asarray(payload[key])
                for key in payload.files
                if key.startswith("optimizer_v_")
            },
        }
        histories = {
            key.removeprefix("history_"): np.asarray(payload[key])
            for key in payload.files
            if key.startswith("history_")
        }
        state = {
            "global_epoch": int(np.asarray(payload["global_epoch"])),
            "loss_ema": float(np.asarray(payload["loss_ema"])),
            "best_val": float(np.asarray(payload["best_val"])),
            "rng_state": json.loads(str(np.asarray(payload["rng_state_json"]).reshape(-1)[0])),
        }
        if "in_epoch_number" in payload.files:
            state["in_epoch"] = {
                "number": int(np.asarray(payload["in_epoch_number"])),
                "completed_steps": int(
                    np.asarray(payload["in_epoch_completed_steps"])
                ),
                "rng_state": json.loads(
                    str(np.asarray(payload["in_epoch_rng_state_json"]).reshape(-1)[0])
                ),
                "losses": np.asarray(payload["in_epoch_losses"]),
                "regime_losses": np.asarray(payload["in_epoch_regime_losses"]),
                "grad_norms": np.asarray(payload["in_epoch_grad_norms"]),
                "update_norms": np.asarray(payload["in_epoch_update_norms"]),
                "update_scales": np.asarray(payload["in_epoch_update_scales"]),
            }
    if set(optimizer["m"]) != set(params) or set(optimizer["v"]) != set(params):
        raise ValueError(f"Incomplete optimizer state: {path}")
    return params, optimizer, histories, state


def initialize_model_parameters(args):
    """Initialize the selected closure with the established seed and float32 dtype."""
    if args.memory_backend == "causal_spacetime_operator":
        params = init_causal_spacetime_operator_params(
            jax.random.PRNGKey(args.seed),
            width=args.width,
            spectral_modes=args.spectral_modes,
            memory_steps=args.memory_steps,
            depth=args.spacetime_depth,
            temporal_kernel_size=args.temporal_kernel_size,
            input_channels=5,
            dtype=jnp.float32,
        )
    elif args.memory_backend == "burles_latent_fno":
        params = init_burles_latent_fno_params(
            jax.random.PRNGKey(args.seed),
            width=args.width,
            spectral_modes=args.spectral_modes,
            memory_steps=args.memory_steps,
            depth=args.fno_depth,
            latent_dim=args.latent_memory_dim,
            input_channels=5,
            dtype=jnp.float32,
        )
    elif args.memory_backend == "window_fno":
        params = init_window_fno_params(
            jax.random.PRNGKey(args.seed),
            width=args.width,
            spectral_modes=args.spectral_modes,
            memory_steps=args.memory_steps,
            depth=args.fno_depth,
            input_channels=5 if args.closure_history_input else 4,
            dtype=jnp.float32,
        )
    elif args.memory_backend == "explicit_window":
        params = init_explicit_window_params(
            jax.random.PRNGKey(args.seed),
            width=args.width,
            spectral_modes=args.spectral_modes,
            memory_steps=args.memory_steps,
            input_channels=5 if args.closure_history_input else 4,
            dtype=jnp.float32,
        )
    else:
        params = init_spectral_memory_params(
            jax.random.PRNGKey(args.seed),
            width=args.width,
            spectral_modes=args.spectral_modes,
            input_channels=5 if args.closure_history_input else 4,
            dtype=jnp.float32,
        )
    return params
