"""Full-history anchor sampling and translation augmentation."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from vpml.datasets.interface_flux import grouped_history_gather
from vpml.solvers.implicit import Array
from vpml.training.interface_configuration import (
    exact_rollout_numpy_complex_dtype,
    exact_rollout_numpy_real_dtype,
    interface_flux_rollout_coeff_key,
)
from vpml.training.interface_reference import (
    _reference_case_histories,
    restrict_rfft_coefficients_to_grid,
)


def prepare_interface_flux_rollout_sampling_state(
    reference_dataset: Dict[str, Dict[str, object]],
    interface_flux_dataset: Dict[str, Dict[str, np.ndarray]],
    *,
    max_projection_order: int,
    target_nvs: Sequence[int],
    history_dtype: object = np.complex128,
    fourier_count: Optional[int] = None,
    source_Nx: Optional[int] = None,
    rollout_Nx: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
    history_np_dtype = exact_rollout_numpy_complex_dtype(history_dtype)
    sampling_state: Dict[str, Dict[str, object]] = {}
    if (source_Nx is None) != (rollout_Nx is None):
        raise ValueError("source_Nx and rollout_Nx must be provided together")
    for regime, arrays in interface_flux_dataset.items():
        histories = (
            np.asarray(
                reference_dataset[regime][coeff_key],
                dtype=history_np_dtype,
            )
            if isinstance(reference_dataset[regime][coeff_key], np.ndarray)
            else _reference_case_histories(reference_dataset[regime], coeff_key)
        )
        first_history = histories[0] if not isinstance(histories, np.ndarray) else histories[0]
        retained_k_count = (
            int(first_history.shape[-1])
            if fourier_count is None
            else int(fourier_count)
        )
        if rollout_Nx is not None:
            expected_k_count = int(rollout_Nx) // 2 + 1
            if retained_k_count != expected_k_count:
                raise ValueError(
                    f"rollout_Nx={int(rollout_Nx)} requires {expected_k_count} "
                    f"Fourier modes, got {retained_k_count}"
                )
        if retained_k_count <= 1 or retained_k_count > int(first_history.shape[-1]):
            raise ValueError(
                f"Invalid retained Fourier count {retained_k_count} for {regime} "
                f"history width {int(first_history.shape[-1])}"
            )
        train_target_nvs = np.asarray(arrays["train_target_nvs"], dtype=np.int32)
        target_pools = {
            int(target_nv): np.flatnonzero(train_target_nvs == int(target_nv)).astype(np.int32)
            for target_nv in target_nvs
        }
        train_anchor_target_nvs = np.asarray(arrays["train_anchor_target_nvs"], dtype=np.int32)
        anchor_target_pools = {
            int(target_nv): np.flatnonzero(train_anchor_target_nvs == int(target_nv)).astype(
                np.int32
            )
            for target_nv in target_nvs
        }
        state: Dict[str, object] = {
            "histories": histories,
            "fourier_count": retained_k_count,
            "source_Nx": source_Nx,
            "rollout_Nx": rollout_Nx,
            "train_case_indices": np.asarray(arrays["train_case_indices"], dtype=np.int32),
            "train_time_indices": np.asarray(arrays["train_time_indices"], dtype=np.int32),
            "train_k_indices": np.asarray(arrays["train_k_indices"], dtype=np.int32),
            "target_pools": target_pools,
            "train_anchor_case_indices": np.asarray(
                arrays["train_anchor_case_indices"], dtype=np.int32
            ),
            "train_anchor_time_indices": np.asarray(
                arrays["train_anchor_time_indices"], dtype=np.int32
            ),
            "anchor_target_pools": anchor_target_pools,
        }
        sampling_state[regime] = state
    return sampling_state


def select_interface_flux_rollout_regime_indices(
    sampling_state: Dict[str, Dict[str, object]],
    *,
    regime: str,
    target_nv: int,
    batch_size: int,
    rng: np.random.Generator,
    all_k_loss: bool = False,
) -> np.ndarray:
    regime_state = sampling_state[regime]
    target_pools = (
        regime_state["anchor_target_pools"] if bool(all_k_loss) else regime_state["target_pools"]
    )
    target_pool = target_pools[int(target_nv)]
    if int(target_pool.shape[0]) <= 0:
        raise ValueError(
            f"Exact q-rollout regime '{regime}' has no valid q-pairs for Nv={int(target_nv)}"
        )
    batch_n = int(min(int(batch_size), int(target_pool.shape[0])))
    return np.asarray(
        target_pool[rng.integers(0, int(target_pool.shape[0]), size=batch_n, endpoint=False)],
        dtype=np.int32,
    )


def translate_interface_flux_rollout_anchor_batch(
    anchor_stencils: np.ndarray,
    ref_q_windows: np.ndarray,
    *,
    k_arr: np.ndarray,
    shifts: np.ndarray,
    k_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Translate complete exact-q anchors using the repository rFFT convention."""
    stencils = np.asarray(anchor_stencils)
    q_windows = np.asarray(ref_q_windows)
    k_values = np.asarray(k_arr, dtype=np.float64)
    shift_values = np.asarray(shifts, dtype=np.float64)
    if stencils.ndim != 4:
        raise ValueError("anchor_stencils must have shape (B,S,Nv,Nk)")
    if shift_values.shape != (int(stencils.shape[0]),):
        raise ValueError("shifts must have shape (B,) matching anchor_stencils")
    if k_values.shape != (int(stencils.shape[-1]),):
        raise ValueError("k_arr must have shape (Nk,) matching anchor_stencils")

    phases = np.exp(-1j * shift_values[:, None] * k_values[None, :])
    translated_stencils = stencils * phases[:, None, None, :]
    if q_windows.ndim == 3:
        if tuple(q_windows.shape[::2]) != (int(stencils.shape[0]), int(stencils.shape[-1])):
            raise ValueError("all-k ref_q_windows must have shape (B,H,Nk)")
        translated_q = q_windows * phases[:, None, :]
    elif q_windows.ndim == 2:
        if k_indices is None:
            raise ValueError("selected-k ref_q_windows require k_indices")
        selected_k = np.asarray(k_indices, dtype=np.int32)
        if selected_k.shape != (int(stencils.shape[0]),):
            raise ValueError("k_indices must have shape (B,)")
        translated_q = q_windows * phases[np.arange(int(stencils.shape[0])), selected_k][:, None]
    else:
        raise ValueError("ref_q_windows must have shape (B,H) or (B,H,Nk)")
    return translated_stencils, translated_q


def sample_interface_flux_rollout_regime_batch(
    sampling_state: Dict[str, Dict[str, object]],
    *,
    regime: str,
    target_nv: int,
    rollout_horizon: int,
    batch_size: int,
    k_arr: np.ndarray,
    rng: np.random.Generator,
    complex_dtype: object = jnp.complex128,
    all_k_loss: bool = False,
    selected_indices: Optional[np.ndarray] = None,
    translation_augmentation: bool = False,
    domain_length: Optional[float] = None,
) -> Dict[str, Array]:
    regime_state = sampling_state[regime]
    histories = regime_state["histories"]
    complex_np_dtype = exact_rollout_numpy_complex_dtype(complex_dtype)
    real_np_dtype = exact_rollout_numpy_real_dtype(complex_dtype)
    real_jax_dtype = jnp.float32 if real_np_dtype == np.dtype(np.float32) else jnp.float64
    k_arr_np = np.asarray(k_arr, dtype=real_np_dtype)
    retained_k_count = int(regime_state["fourier_count"])
    if int(k_arr_np.shape[0]) != retained_k_count:
        raise ValueError(
            f"k_arr has {int(k_arr_np.shape[0])} modes but sampling retains "
            f"{retained_k_count}"
        )
    if selected_indices is None:
        selected = select_interface_flux_rollout_regime_indices(
            sampling_state,
            regime=regime,
            target_nv=int(target_nv),
            batch_size=int(batch_size),
            rng=rng,
            all_k_loss=bool(all_k_loss),
        )
    else:
        selected = np.asarray(selected_indices, dtype=np.int32)
        if selected.ndim != 1 or int(selected.shape[0]) <= 0:
            raise ValueError("selected_indices must be a nonempty 1D array")
    batch_n = int(selected.shape[0])
    if bool(all_k_loss):
        case_idx = regime_state["train_anchor_case_indices"][selected]
        time_idx = regime_state["train_anchor_time_indices"][selected]
        k_indices = np.zeros((batch_n,), dtype=np.int32)
    else:
        case_idx = regime_state["train_case_indices"][selected]
        time_idx = regime_state["train_time_indices"][selected]
        k_indices = regime_state["train_k_indices"][selected]
    stencil_times = np.stack(
        (
            time_idx,
            np.maximum(time_idx - 1, 0),
            np.maximum(time_idx - 2, 0),
        ),
        axis=1,
    )
    target_nv_i = int(target_nv)
    stencils = grouped_history_gather(
        histories,
        case_idx[:, None],
        stencil_times,
        hermite_slice=slice(0, target_nv_i),
        fourier_slice=slice(0, retained_k_count),
    )
    if regime_state["source_Nx"] is not None:
        stencils = restrict_rfft_coefficients_to_grid(
            stencils,
            source_Nx=int(regime_state["source_Nx"]),
            target_Nx=int(regime_state["rollout_Nx"]),
        )
    first_history = (
        histories[0]
        if not isinstance(histories, np.ndarray) or histories.ndim != 4
        else histories[0]
    )
    nk = retained_k_count
    if nk <= 1:
        raise ValueError("Exact q-rollout requires at least one nonzero Fourier mode")
    offsets = np.arange(int(rollout_horizon), dtype=np.int32)
    window_times = time_idx[:, None] + offsets[None, :]
    if bool(all_k_loss):
        q_coeff = grouped_history_gather(
            histories,
            case_idx[:, None],
            window_times,
            hermite_slice=target_nv_i,
            fourier_slice=slice(0, retained_k_count),
        )
        if regime_state["source_Nx"] is not None:
            q_coeff = restrict_rfft_coefficients_to_grid(
                q_coeff,
                source_Nx=int(regime_state["source_Nx"]),
                target_Nx=int(regime_state["rollout_Nx"]),
            )
        q_windows = (
            -1j
            * k_arr_np[None, None, :]
            * math.sqrt(float(target_nv_i))
            * q_coeff
        )
    else:
        q_coeff_all = grouped_history_gather(
            histories,
            case_idx[:, None],
            window_times,
            hermite_slice=target_nv_i,
            fourier_slice=slice(0, retained_k_count),
        )
        if regime_state["source_Nx"] is not None:
            q_coeff_all = restrict_rfft_coefficients_to_grid(
                q_coeff_all,
                source_Nx=int(regime_state["source_Nx"]),
                target_Nx=int(regime_state["rollout_Nx"]),
            )
        q_coeff = q_coeff_all[
            np.arange(batch_n, dtype=np.int32)[:, None],
            np.arange(int(rollout_horizon), dtype=np.int32)[None, :],
            k_indices[:, None],
        ]
        q_windows = (
            -1j
            * k_arr_np[k_indices][:, None]
            * math.sqrt(float(target_nv_i))
            * q_coeff
        )
    if bool(translation_augmentation):
        if domain_length is None or float(domain_length) <= 0.0:
            raise ValueError("translation augmentation requires a positive domain_length")
        shifts = rng.uniform(0.0, float(domain_length), size=batch_n)
        stencils, q_windows = translate_interface_flux_rollout_anchor_batch(
            stencils,
            q_windows,
            k_arr=k_arr_np,
            shifts=shifts,
            k_indices=k_indices,
        )
    batch = {
        "anchor_stencils": jnp.asarray(stencils, dtype=complex_dtype),
        "ref_q_windows": jnp.asarray(q_windows, dtype=complex_dtype),
        "k_indices": jnp.asarray(k_indices, dtype=jnp.int32),
    }
    return batch


def interface_flux_cutoff_for_step(
    target_nvs: Sequence[int],
    global_step: int,
) -> int:
    targets = tuple(int(value) for value in target_nvs)
    if not targets:
        raise ValueError("target_nvs must not be empty")
    if int(global_step) < 0:
        raise ValueError("global_step must be nonnegative")
    return targets[int(global_step) % len(targets)]
