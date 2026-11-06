"""Finite-update guarded interface-flux minibatch training."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from vpml.core import Array
from vpml.training.interface_configuration import (
    EXACT_ROLLOUT_PRECISION_FLOAT64,
    exact_rollout_numpy_complex_dtype,
    exact_rollout_precision_dtypes,
)
from vpml.training.interface_sampling import (
    interface_flux_cutoff_for_step,
    prepare_interface_flux_rollout_sampling_state,
    sample_interface_flux_rollout_regime_batch,
    select_interface_flux_rollout_regime_indices,
)
from vpml.training.optimization import _tree_all_finite, adam_init, adam_step


def _format_train_loss_log(
    *,
    epoch: int,
    epochs: int,
    history: Dict[str, np.ndarray],
    components: Sequence[str],
) -> str:
    parts = [
        f"[train] epoch {int(epoch) + 1:04d}/{int(epochs):04d}",
        f"loss={history['total'][int(epoch)]:.6e}",
    ]
    for key in components:
        if key == "total":
            continue
        if key in history:
            parts.append(f"{key}={history[key][int(epoch)]:.6e}")
    return " ".join(parts)


def train_with_interface_flux_rollout_minibatch_loss(
    params: Dict[str, Array],
    reference_dataset: Dict[str, Dict[str, object]],
    interface_flux_dataset: Dict[str, Dict[str, np.ndarray]],
    batch_loss_fn,
    *,
    max_projection_order: int,
    active_regimes: Sequence[str],
    k_arr: np.ndarray,
    epochs: int,
    learning_rate: float,
    grad_clip: Optional[float],
    log_every: int,
    batch_size: int,
    steps_per_epoch: int,
    rollout_horizon: int,
    seed: int,
    source_Nx: int,
    rollout_Nx: int,
    log_components: Sequence[str] = (),
    profile_trace_dir: Optional[Path] = None,
    profile_train_steps: int = 0,
    profile_skip_steps: int = 1,
) -> Tuple[Dict[str, Array], Dict[str, np.ndarray]]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive for interface-flux rollout training")
    if int(steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be positive for interface-flux rollout training")

    optimizer_state = adam_init(params)
    history = {
        key: np.zeros((int(epochs),), dtype=np.float64)
        for key in ("total", "q")
    }

    def make_train_step(target_batch_loss_fn):
        @jax.jit
        def train_step(
            current_params: Dict[str, Array],
            current_state: Dict[str, object],
            regime_batches: Dict[str, Dict[str, Array]],
        ) -> Tuple[Dict[str, Array], Dict[str, object], Dict[str, Array], Array]:
            (loss, aux), grads = jax.value_and_grad(
                target_batch_loss_fn,
                has_aux=True,
            )(current_params, regime_batches)
            aux = dict(aux)
            aux["total"] = loss
            all_finite = _tree_all_finite(aux) & _tree_all_finite(grads)

            def apply_update(_: None) -> Tuple[Dict[str, Array], Dict[str, object]]:
                return adam_step(
                    current_params,
                    grads,
                    current_state,
                    learning_rate,
                    grad_clip=grad_clip,
                )

            def keep_state(_: None) -> Tuple[Dict[str, Array], Dict[str, object]]:
                return current_params, current_state

            next_params, next_state = jax.lax.cond(
                all_finite,
                apply_update,
                keep_state,
                operand=None,
            )
            return next_params, next_state, aux, all_finite

        return train_step

    target_nvs = tuple(int(value) for value in getattr(batch_loss_fn, "target_nvs", ()))
    target_loss_fns = getattr(batch_loss_fn, "target_loss_fns", None)
    if not target_nvs or not isinstance(target_loss_fns, dict):
        raise ValueError("Canonical interface-flux loss must provide per-cutoff loss functions")
    train_steps = {
        target_nv: make_train_step(target_loss_fns[target_nv])
        for target_nv in target_nvs
    }

    rng = np.random.default_rng(int(seed))
    rollout_precision = str(
        getattr(batch_loss_fn, "rollout_precision", EXACT_ROLLOUT_PRECISION_FLOAT64)
    )
    _, batch_complex_dtype = exact_rollout_precision_dtypes(rollout_precision)
    history_complex_dtype = exact_rollout_numpy_complex_dtype(batch_complex_dtype)
    translation_augmented = bool(
        getattr(batch_loss_fn, "translation_augmented", False)
    )
    translation_domain_length = getattr(batch_loss_fn, "teacher_Lx", None)
    sampling_state = prepare_interface_flux_rollout_sampling_state(
        reference_dataset,
        interface_flux_dataset,
        max_projection_order=int(max_projection_order),
        target_nvs=target_nvs,
        history_dtype=history_complex_dtype,
        fourier_count=int(np.asarray(k_arr).shape[0]),
        source_Nx=int(source_Nx),
        rollout_Nx=int(rollout_Nx),
    )

    profile_steps = max(int(profile_train_steps), 0)
    profile_skip = max(int(profile_skip_steps), 0)
    profile_dir = (
        None
        if profile_trace_dir is None or profile_steps <= 0
        else Path(profile_trace_dir)
    )
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[profile] interface-flux rollout tracing {profile_steps} train step(s) "
            f"after {profile_skip} warmup step(s) to {profile_dir}"
        )
    profiled_seconds: List[float] = []

    for epoch in range(int(epochs)):
        running = {
            key: jnp.asarray(0.0, dtype=jnp.float64)
            for key in history
        }
        for step_idx in range(int(steps_per_epoch)):
            global_step = int(epoch) * int(steps_per_epoch) + int(step_idx)
            target_nv = interface_flux_cutoff_for_step(target_nvs, global_step)
            selected_by_regime = {
                regime: select_interface_flux_rollout_regime_indices(
                    sampling_state,
                    regime=regime,
                    target_nv=target_nv,
                    batch_size=int(batch_size),
                    rng=rng,
                    all_k_loss=True,
                )
                for regime in active_regimes
            }
            regime_batches = {
                regime: sample_interface_flux_rollout_regime_batch(
                    sampling_state,
                    regime=regime,
                    target_nv=target_nv,
                    rollout_horizon=int(rollout_horizon),
                    batch_size=int(batch_size),
                    k_arr=np.asarray(k_arr, dtype=np.float64),
                    rng=rng,
                    complex_dtype=batch_complex_dtype,
                    all_k_loss=True,
                    selected_indices=selected_by_regime[regime],
                    translation_augmentation=translation_augmented,
                    domain_length=(
                        float(translation_domain_length)
                        if translation_augmented
                        and translation_domain_length is not None
                        else None
                    ),
                )
                for regime in active_regimes
            }

            should_profile = (
                profile_dir is not None
                and profile_skip <= global_step < profile_skip + profile_steps
            )
            started = time.perf_counter() if should_profile else 0.0
            if should_profile:
                with jax.profiler.trace(str(profile_dir)):
                    params, optimizer_state, aux, all_finite = train_steps[target_nv](
                        params,
                        optimizer_state,
                        regime_batches,
                    )
                    jax.block_until_ready(aux["total"])
                profiled_seconds.append(time.perf_counter() - started)
            else:
                params, optimizer_state, aux, all_finite = train_steps[target_nv](
                    params,
                    optimizer_state,
                    regime_batches,
                )
            if not bool(all_finite):
                raise FloatingPointError(
                    "interface-flux rollout produced non-finite loss/gradients at "
                    f"epoch {epoch + 1}, step {step_idx + 1}; "
                    "reduce TRAIN_LR, TRAIN_BATCH_SIZE, TRAIN_ROLLOUT_HORIZON, "
                    "or TRAIN_GRAD_CLIP."
                )
            for key in running:
                running[key] = running[key] + aux[key]

        for key in history:
            history[key][epoch] = float(running[key] / float(steps_per_epoch))
        if (
            epoch == 0
            or (epoch + 1) % max(int(log_every), 1) == 0
            or epoch + 1 == int(epochs)
        ):
            print(
                _format_train_loss_log(
                    epoch=epoch,
                    epochs=epochs,
                    history=history,
                    components=log_components,
                )
            )

    if profiled_seconds:
        mean_seconds = float(np.mean(np.asarray(profiled_seconds, dtype=np.float64)))
        print(
            f"[profile] interface-flux rollout traced {len(profiled_seconds)} step(s); "
            f"mean profiled wall time={mean_seconds:.3f}s/step"
        )
    return params, history
