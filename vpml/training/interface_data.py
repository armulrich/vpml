"""Interface-flux target pairs and training-only normalization statistics."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from model.train.interface_flux_data import (
    IC_SPLIT_HELDOUT,
    IC_SPLIT_TRAIN,
    StreamingMoments,
    grouped_history_gather,
)
from vpml.physical_grid import extract_interface_supervised_pairs_from_coeff_history
from vpml.training.interface_configuration import (
    REGIME_LINEAR,
    interface_flux_rollout_coeff_key,
)
from vpml.training.interface_reference import (
    _reference_case_histories,
    _reference_case_splits,
    restrict_rfft_coefficients_to_grid,
)


def append_pairs(
    accum: Dict[str, Dict[str, list]],
    regime: str,
    split: str,
    pairs_by_nv: Dict[int, Dict[str, np.ndarray]],
) -> None:
    for payload in pairs_by_nv.values():
        accum[regime][f"{split}_inputs_base"].append(payload["inputs_base"])
        accum[regime][f"{split}_targets"].append(payload["targets"])


def build_interface_flux_rollout_qpair_dataset(
    exact_dataset: Dict[str, Dict[str, object]],
    *,
    max_projection_order: int,
    Nv_targets: Sequence[int],
    Nm: int,
    k_arr: np.ndarray,
    linear_history_stride: int,
    nonlinear_history_stride: int,
    rollout_horizon: int,
    n_low: int,
    context_mode: str,
    store_training_pairs: bool = True,
    k_scale: Optional[float] = None,
    nv_scale: Optional[float] = None,
    precomputed_training_stats: Optional[Dict[str, np.ndarray]] = None,
    source_Nx: Optional[int] = None,
    rollout_Nx: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
    accum = {
        regime: {
            "train_inputs_base": [],
            "train_targets": [],
            "val_inputs_base": [],
            "val_targets": [],
            "train_case_indices": [],
            "train_time_indices": [],
            "train_k_indices": [],
            "train_target_nvs": [],
            "train_anchor_case_indices": [],
            "train_anchor_time_indices": [],
            "train_anchor_target_nvs": [],
            "val_case_indices": [],
            "val_time_indices": [],
            "val_k_indices": [],
            "val_target_nvs": [],
            "val_anchor_case_indices": [],
            "val_anchor_time_indices": [],
            "val_anchor_target_nvs": [],
        }
        for regime in exact_dataset
    }
    input_moments = StreamingMoments()
    target_moments = StreamingMoments()

    if not bool(store_training_pairs) and (k_scale is None or nv_scale is None):
        raise ValueError("k_scale and nv_scale are required when store_training_pairs=False")
    retained_k_count = int(np.asarray(k_arr).shape[0])
    if retained_k_count <= 1:
        raise ValueError("interface-flux training requires at least one positive Fourier mode")
    if (source_Nx is None) != (rollout_Nx is None):
        raise ValueError("source_Nx and rollout_Nx must be provided together")

    def sampled_indices(history_length: int, stride: int) -> np.ndarray:
        nsteps = int(history_length) - 1
        sampled = np.arange(0, nsteps + 1, max(int(stride), 1), dtype=np.int32)
        if sampled.size == 0 or int(sampled[-1]) != nsteps:
            sampled = np.concatenate([sampled, np.array([nsteps], dtype=np.int32)])
        return sampled[(sampled + int(rollout_horizon) - 1) < int(history_length)]

    def append_index_rows(
        *,
        regime: str,
        split: str,
        case_idx: int,
        time_indices: np.ndarray,
        k_count: int,
        include_flattened_k_rows: bool,
    ) -> None:
        if str(context_mode) == "lag1_delta":
            time_indices = np.asarray(time_indices, dtype=np.int32)[1:]
        else:
            time_indices = np.asarray(time_indices, dtype=np.int32)
        if int(time_indices.shape[0]) == 0:
            return
        for target_nv in Nv_targets:
            anchor_rows = int(time_indices.shape[0])
            accum[regime][f"{split}_anchor_case_indices"].append(
                np.full(anchor_rows, int(case_idx), dtype=np.int32)
            )
            accum[regime][f"{split}_anchor_time_indices"].append(time_indices.astype(np.int32))
            accum[regime][f"{split}_anchor_target_nvs"].append(
                np.full(anchor_rows, int(target_nv), dtype=np.int32)
            )
            if bool(include_flattened_k_rows):
                k_indices_one_time = np.arange(1, int(k_count), dtype=np.int32)
                if int(k_indices_one_time.shape[0]) == 0:
                    continue
                case_rows = np.full(
                    int(time_indices.shape[0]) * int(k_indices_one_time.shape[0]),
                    int(case_idx),
                    dtype=np.int32,
                )
                time_rows_base = np.repeat(
                    time_indices, int(k_indices_one_time.shape[0])
                ).astype(np.int32)
                k_rows_base = np.tile(k_indices_one_time, int(time_indices.shape[0])).astype(
                    np.int32
                )
                rows = int(time_rows_base.shape[0])
                accum[regime][f"{split}_case_indices"].append(case_rows)
                accum[regime][f"{split}_time_indices"].append(time_rows_base)
                accum[regime][f"{split}_k_indices"].append(k_rows_base)
                accum[regime][f"{split}_target_nvs"].append(
                    np.full(rows, int(target_nv), dtype=np.int32)
                )

    def accumulate_training_stats(pairs_by_nv: Dict[int, Dict[str, np.ndarray]]) -> None:
        assert k_scale is not None
        assert nv_scale is not None
        for payload in pairs_by_nv.values():
            inputs = build_model_inputs(
                payload["inputs_base"],
                Nm=int(Nm),
                k_scale=float(k_scale),
                nv_scale=float(nv_scale),
                context_mode=str(context_mode),
                include_global_indicators=True,
            )
            targets = np.asarray(payload["targets"], dtype=np.float64)
            input_moments.update(inputs)
            target_moments.update(targets)

    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive for interface-flux rollout q-pairs")
    for regime, group in exact_dataset.items():
        cases = _reference_case_histories(group, coeff_key)
        case_splits = _reference_case_splits(group, len(cases))
        stride = int(linear_history_stride) if regime == REGIME_LINEAR else int(nonlinear_history_stride)
        stride = max(stride, 1)
        for case_idx, case_hist in enumerate(cases):
            if int(case_hist.shape[-1]) < retained_k_count:
                raise ValueError(
                    f"Reference history for {regime} has {int(case_hist.shape[-1])} "
                    f"Fourier modes but training requires {retained_k_count}"
                )
            split_name = (
                "train"
                if str(case_splits[case_idx]) == IC_SPLIT_TRAIN
                else "val"
                if str(case_splits[case_idx]) == IC_SPLIT_HELDOUT
                else None
            )
            if split_name is None:
                raise ValueError(
                    f"Unsupported trajectory split {case_splits[case_idx]!r}"
                )
            original_times = sampled_indices(int(case_hist.shape[0]), stride)
            if int(original_times.shape[0]) == 0:
                continue
            needs_pairs = bool(store_training_pairs) or (
                split_name == "train" and precomputed_training_stats is None
            )
            if needs_pairs:
                hist = np.asarray(
                    case_hist[
                        np.asarray(original_times, dtype=np.int32),
                        :,
                        :retained_k_count,
                    ],
                    dtype=np.complex128,
                )
                if source_Nx is not None and rollout_Nx is not None:
                    hist = restrict_rfft_coefficients_to_grid(
                        hist,
                        source_Nx=int(source_Nx),
                        target_Nx=int(rollout_Nx),
                    )
                pairs = extract_interface_supervised_pairs_from_coeff_history(
                    hist,
                    Nv_targets=Nv_targets,
                    Nm=Nm,
                    k_arr=k_arr,
                    vth=1.0,
                    include_global_indicators=True,
                    n_low=int(n_low),
                    context_mode=context_mode,
                )
                if bool(store_training_pairs):
                    append_pairs(accum, regime, split_name, pairs)
                elif split_name == "train":
                    accumulate_training_stats(pairs)
            append_index_rows(
                regime=regime,
                split=split_name,
                case_idx=int(case_idx),
                time_indices=np.asarray(original_times, dtype=np.int32),
                k_count=retained_k_count,
                include_flattened_k_rows=bool(store_training_pairs),
            )
    raw_base_dim = 2 * int(Nm) + 4
    input_dim = raw_base_dim if str(context_mode) == "none" else 3 * raw_base_dim
    dataset: Dict[str, Dict[str, np.ndarray]] = {}
    for regime, payload in accum.items():
        has_any_rows = any(
            payload[f"{split}_anchor_case_indices"]
            for split in ("train", "val")
        )
        if not has_any_rows:
            continue

        def concat_or_empty(key: str, shape_tail: Tuple[int, ...], dtype) -> np.ndarray:
            values = payload[key]
            if values:
                return np.concatenate(values, axis=0).astype(dtype)
            return np.zeros((0, *shape_tail), dtype=dtype)

        dataset[regime] = {
            "train_inputs_base": concat_or_empty("train_inputs_base", (input_dim,), np.float64),
            "train_targets": concat_or_empty("train_targets", (2,), np.float64),
            "val_inputs_base": concat_or_empty("val_inputs_base", (input_dim,), np.float64),
            "val_targets": concat_or_empty("val_targets", (2,), np.float64),
        }
    for regime, arrays in dataset.items():
        for split in ("train", "val"):
            for suffix in ("case_indices", "time_indices", "k_indices", "target_nvs"):
                key = f"{split}_{suffix}"
                arrays[key] = (
                    np.concatenate(accum[regime][key], axis=0).astype(np.int32)
                    if accum[regime][key]
                    else np.zeros((0,), dtype=np.int32)
                )
            for suffix in ("anchor_case_indices", "anchor_time_indices", "anchor_target_nvs"):
                key = f"{split}_{suffix}"
                arrays[key] = (
                    np.concatenate(accum[regime][key], axis=0).astype(np.int32)
                    if accum[regime][key]
                    else np.zeros((0,), dtype=np.int32)
                )
    computed_stats: Optional[Dict[str, np.ndarray]] = None
    if not bool(store_training_pairs):
        if precomputed_training_stats is not None:
            computed_stats = {
                key: np.asarray(precomputed_training_stats[key], dtype=np.float64)
                for key in ("input_mean", "input_std", "target_mean", "target_std")
            }
        else:
            input_mean, input_std = input_moments.finalize()
            target_mean, target_std = target_moments.finalize()
            computed_stats = {
                "input_mean": np.asarray(input_mean, dtype=np.float64),
                "input_std": safe_feature_std(input_std),
                "target_mean": np.asarray(target_mean, dtype=np.float64),
                "target_std": safe_feature_std(target_std),
            }
    return dataset, computed_stats


def interface_flux_rollout_regime_loss_stds(
    exact_dataset: Dict[str, Dict[str, object]],
    qpair_dataset: Dict[str, Dict[str, np.ndarray]],
    *,
    max_projection_order: int,
    target_nvs: Sequence[int],
    k_arr: np.ndarray,
    rollout_horizon: int,
    chunk_size: int = 64,
    source_Nx: Optional[int] = None,
    rollout_Nx: Optional[int] = None,
) -> Dict[str, float]:
    """Compute fixed phase-isotropic q-loss scales for each training regime."""
    coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
    k_values = np.asarray(k_arr, dtype=np.float64)
    if int(k_values.shape[0]) <= 1:
        raise ValueError("Exact q-rollout regime scaling requires positive Fourier modes")
    horizon = int(rollout_horizon)
    if horizon <= 0:
        raise ValueError("Exact q-rollout regime scaling requires rollout_horizon > 0")
    chunk_n = max(int(chunk_size), 1)
    offsets = np.arange(horizon, dtype=np.int32)
    positive_k_sq = k_values[1:] ** 2
    if (source_Nx is None) != (rollout_Nx is None):
        raise ValueError("source_Nx and rollout_Nx must be provided together")
    scales: Dict[str, float] = {}

    for regime, arrays in qpair_dataset.items():
        if regime not in exact_dataset:
            continue
        histories = _reference_case_histories(exact_dataset[regime], coeff_key)
        anchor_cases = np.asarray(arrays["train_anchor_case_indices"], dtype=np.int32)
        anchor_times = np.asarray(arrays["train_anchor_time_indices"], dtype=np.int32)
        anchor_targets = np.asarray(arrays["train_anchor_target_nvs"], dtype=np.int32)
        sum_abs_sq = 0.0
        value_count = 0

        for target_nv in target_nvs:
            selected = np.flatnonzero(anchor_targets == int(target_nv)).astype(np.int32)
            for start in range(0, int(selected.shape[0]), chunk_n):
                chunk = selected[start : start + chunk_n]
                if int(chunk.shape[0]) == 0:
                    continue
                case_idx = anchor_cases[chunk]
                time_idx = anchor_times[chunk]
                window_times = time_idx[:, None] + offsets[None, :]
                if any(
                    int(np.max(window_times[case_idx == index])) >= int(histories[index].shape[0])
                    for index in np.unique(case_idx)
                ):
                    raise ValueError(
                        f"Exact q-rollout regime scale window exceeds history for {regime}"
                    )
                coeff = grouped_history_gather(
                    histories,
                    case_idx[:, None],
                    window_times,
                    hermite_slice=int(target_nv),
                    fourier_slice=slice(1, int(k_values.shape[0])),
                )
                if source_Nx is not None and rollout_Nx is not None:
                    with_zero_mode = np.zeros(
                        (*coeff.shape[:-1], int(k_values.shape[0])),
                        dtype=coeff.dtype,
                    )
                    with_zero_mode[..., 1:] = coeff
                    coeff = restrict_rfft_coefficients_to_grid(
                        with_zero_mode,
                        source_Nx=int(source_Nx),
                        target_Nx=int(rollout_Nx),
                    )[..., 1:]
                q_abs_sq = (
                    float(target_nv)
                    * positive_k_sq[None, None, :]
                    * np.abs(coeff) ** 2
                )
                sum_abs_sq += float(np.sum(q_abs_sq, dtype=np.float64))
                value_count += int(q_abs_sq.size)

        if value_count <= 0:
            raise ValueError(f"Exact q-rollout regime '{regime}' has no q targets for scaling")
        component_variance = 0.5 * sum_abs_sq / float(value_count)
        component_std = math.sqrt(max(component_variance, 0.0))
        if not math.isfinite(component_std) or component_std <= 1e-12:
            raise ValueError(
                f"Exact q-rollout regime '{regime}' has invalid q-loss scale {component_std}"
            )
        scales[str(regime)] = float(component_std)
    return scales


def build_model_inputs(
    inputs_base: np.ndarray,
    *,
    Nm: int,
    k_scale: float,
    nv_scale: float,
    context_mode: str,
    include_global_indicators: bool = True,
) -> np.ndarray:
    inputs = np.asarray(inputs_base, dtype=np.float64).copy()
    k_col = 2 * int(Nm)
    nv_col = k_col + 1
    base_dim = 2 * int(Nm) + (4 if bool(include_global_indicators) else 2)
    if context_mode == "none":
        inputs[:, k_col] = inputs[:, k_col] / float(k_scale)
        inputs[:, nv_col] = inputs[:, nv_col] / float(nv_scale)
        return inputs
    if context_mode == "lag1_delta":
        current = inputs[:, :base_dim]
        previous = inputs[:, base_dim : 2 * base_dim]
        current[:, k_col] = current[:, k_col] / float(k_scale)
        current[:, nv_col] = current[:, nv_col] / float(nv_scale)
        previous[:, k_col] = previous[:, k_col] / float(k_scale)
        previous[:, nv_col] = previous[:, nv_col] / float(nv_scale)
        delta = current - previous
        return np.concatenate([current, previous, delta], axis=1)
    raise ValueError(f"Unsupported context_mode={context_mode!r}")


def safe_feature_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=np.float64)
    return np.where(std > 1e-12, std, 1.0)


def phase_isotropic_complex_training_stats(
    stats: Dict[str, np.ndarray],
    *,
    Nm: int,
    context_mode: str,
) -> Dict[str, np.ndarray]:
    """Return phase-isotropic normalization for complex closure features and q."""
    input_mean = np.asarray(stats["input_mean"], dtype=np.float64).copy()
    input_std = np.asarray(stats["input_std"], dtype=np.float64).copy()
    target_mean = np.asarray(stats["target_mean"], dtype=np.float64).copy()
    target_std = np.asarray(stats["target_std"], dtype=np.float64).copy()
    base_dim = 2 * int(Nm) + 4
    if str(context_mode) == "none":
        block_offsets = (0,)
    elif str(context_mode) == "lag1_delta":
        block_offsets = (0, base_dim, 2 * base_dim)
    else:
        raise ValueError(f"Unsupported context_mode={context_mode!r}")

    for offset in block_offsets:
        for mode_idx in range(int(Nm)):
            real_idx = int(offset + mode_idx)
            imag_idx = int(offset + int(Nm) + mode_idx)
            second_moment = 0.5 * (
                input_std[real_idx] ** 2
                + input_mean[real_idx] ** 2
                + input_std[imag_idx] ** 2
                + input_mean[imag_idx] ** 2
            )
            shared_std = float(safe_feature_std(np.array([math.sqrt(second_moment)]))[0])
            input_mean[[real_idx, imag_idx]] = 0.0
            input_std[[real_idx, imag_idx]] = shared_std

    target_second_moment = 0.5 * float(
        np.sum(target_std * target_std + target_mean * target_mean)
    )
    shared_target_std = float(
        safe_feature_std(np.array([math.sqrt(target_second_moment)]))[0]
    )
    target_mean[:] = 0.0
    target_std[:] = shared_target_std
    return {
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }
