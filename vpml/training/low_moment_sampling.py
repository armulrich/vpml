"""Case-balanced windows, complete trajectories, and diagnostic sampling."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from vpml.datasets.interface_flux import IC_SPLIT_HELDOUT, IC_SPLIT_TRAIN
from vpml.training.low_moment_configuration import REGIMES
from vpml.training.low_moment_data import (
    _case_amplitudes,
    _central_heat_flux_numpy,
    _gather_coefficients,
    low_hermite_coefficients_to_conservative,
)


def _build_anchor_index(
    grouped: Mapping[str, Mapping[str, object]],
    coefficient_key: str,
    *,
    regimes: Sequence[str] = REGIMES,
    horizon: int,
    history_stride: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for regime in regimes:
        group = grouped[regime]
        histories = tuple(group[coefficient_key])
        splits = np.asarray(group["case_splits"], dtype=np.str_)
        payload: Dict[str, np.ndarray] = {}
        for split_name, split_value in (("train", IC_SPLIT_TRAIN), ("val", IC_SPLIT_HELDOUT)):
            case_rows = []
            time_rows = []
            for case_index, (history, split) in enumerate(zip(histories, splits)):
                if str(split) != split_value:
                    continue
                times = np.arange(
                    0,
                    int(history.shape[0]) - int(horizon),
                    max(int(history_stride), 1),
                    dtype=np.int32,
                )
                case_rows.append(np.full(times.shape, case_index, dtype=np.int32))
                time_rows.append(times)
            payload[f"{split_name}_cases"] = (
                np.concatenate(case_rows) if case_rows else np.zeros((0,), dtype=np.int32)
            )
            payload[f"{split_name}_times"] = (
                np.concatenate(time_rows) if time_rows else np.zeros((0,), dtype=np.int32)
            )
        result[regime] = payload
    return result


def limit_training_anchors(
    anchors: Mapping[str, Mapping[str, np.ndarray]],
    case_limits: Mapping[str, int],
    *,
    regimes: Sequence[str] = REGIMES,
    seed: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    limited: Dict[str, Dict[str, np.ndarray]] = {}
    selected_cases: Dict[str, np.ndarray] = {}
    for regime in regimes:
        regime_index = REGIMES.index(regime)
        payload = {
            key: np.asarray(value).copy()
            for key, value in anchors[regime].items()
        }
        available = np.unique(payload["train_cases"])
        count = min(int(case_limits.get(regime, available.size)), int(available.size))
        permutation = np.random.default_rng(
            int(seed) + 1009 * (regime_index + 1)
        ).permutation(available)
        selected = np.sort(permutation[:count]).astype(np.int32)
        mask = np.isin(payload["train_cases"], selected)
        payload["train_cases"] = payload["train_cases"][mask]
        payload["train_times"] = payload["train_times"][mask]
        limited[regime] = payload
        selected_cases[regime] = selected
    return limited, selected_cases


def sample_batch(
    rng: np.random.Generator,
    grouped,
    anchors,
    manifest,
    coefficient_key: str,
    *,
    regimes: Sequence[str] = REGIMES,
    split: str,
    batch_size_per_regime: int,
    horizon: int,
    memory_steps: int,
    memory_stride: int,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    translation_augmentation: bool,
    explicit_selection: Optional[
        Mapping[str, Tuple[np.ndarray, np.ndarray]]
    ] = None,
    time_blocks: int = 6,
    heat_flux_gradient_case_scale_by_id: Optional[Mapping[str, float]] = None,
    include_heat_flux_gradient_history: bool = False,
    autonomous_burnin_steps: int = 0,
    reference_steps_per_solver_step: float = 1.0,
) -> Dict[str, np.ndarray]:
    amplitudes_by_id = _case_amplitudes(manifest)
    memories = []
    initials = []
    targets = []
    amplitudes = []
    regime_indices = []
    heat_flux_gradient_targets = []
    heat_flux_gradient_histories = []
    heat_flux_gradient_case_scales = []
    start_indices = []
    for regime in regimes:
        regime_index = REGIMES.index(regime)
        group = grouped[regime]
        histories = tuple(group[coefficient_key])
        case_ids = np.asarray(group["case_ids"], dtype=np.str_)
        case_pool = anchors[regime][f"{split}_cases"]
        time_pool = anchors[regime][f"{split}_times"]
        if int(case_pool.size) == 0:
            raise ValueError(f"No {split} anchors for {regime}")
        if explicit_selection is not None:
            cases, times = explicit_selection[regime]
            cases = np.asarray(cases, dtype=np.int32)
            times = np.asarray(times, dtype=np.int32)
        else:
            unique_cases = np.unique(case_pool)
            cases = rng.choice(
                unique_cases,
                size=int(batch_size_per_regime),
                replace=int(batch_size_per_regime) > int(unique_cases.size),
            ).astype(np.int32)
            block_count = max(int(time_blocks), 1)
            block_offset = int(rng.integers(0, block_count))
            block_ids = (block_offset + np.arange(cases.size)) % block_count
            selected_times = []
            for case_index, block_index in zip(cases, block_ids):
                available = time_pool[case_pool == int(case_index)]
                edges = np.linspace(0, available.size, block_count + 1, dtype=np.int32)
                left, right = int(edges[block_index]), int(edges[block_index + 1])
                if right <= left:
                    left, right = 0, int(available.size)
                selected_times.append(available[int(rng.integers(left, right))])
            times = np.asarray(selected_times, dtype=np.int32)
        burnin_steps = int(autonomous_burnin_steps)
        reference_ratio = float(reference_steps_per_solver_step)
        if reference_ratio <= 0.0:
            raise ValueError("reference_steps_per_solver_step must be positive")
        if burnin_steps:
            # A random training reset supplies only the three physical moments.
            # The closure history is then generated by the model itself before
            # any scored target is observed.
            memory_times = np.broadcast_to(
                times[:, None], (times.size, int(memory_steps))
            )
        else:
            memory_offsets = float(memory_stride) * reference_ratio * np.arange(
                int(memory_steps), 0, -1, dtype=np.float64
            )
            raw_memory_times = times[:, None] - memory_offsets[None, :]
            padded_memory = raw_memory_times < 0
            memory_times = np.maximum(raw_memory_times, 0)
        target_times = times[:, None] + reference_ratio * (
            burnin_steps
            + np.arange(1, int(horizon) + 1, dtype=np.float64)[None]
        )
        memory_coefficient_count = 4 if include_heat_flux_gradient_history else 3
        memory_coeff = _gather_coefficients(
            histories,
            cases[:, None],
            memory_times,
            coefficient_count=memory_coefficient_count,
        )
        initial_coeff = _gather_coefficients(
            histories, cases, times, coefficient_count=4
        )
        target_coeff = _gather_coefficients(
            histories, cases[:, None], target_times, coefficient_count=3
        )
        memory_state = low_hermite_coefficients_to_conservative(
            memory_coeff, source_nx=source_nx, target_nx=rollout_nx
        )
        if burnin_steps:
            # A random reset exposes only its current physical state. Generate
            # the causal history from an equilibrium-padded buffer.
            memory_state[...] = 0.0
        else:
            # Before t=0 the causal history is the centered equilibrium state,
            # rather than a fictitious copy of the perturbed initial condition.
            memory_state[padded_memory] = 0.0
        memories.append(memory_state)
        initials.append(
            low_hermite_coefficients_to_conservative(
                initial_coeff, source_nx=source_nx, target_nx=rollout_nx
            )
        )
        targets.append(
            low_hermite_coefficients_to_conservative(
                target_coeff, source_nx=source_nx, target_nx=rollout_nx
            )
        )
        current_coeff = initial_coeff
        current_state = low_hermite_coefficients_to_conservative(
            current_coeff,
            source_nx=source_nx,
            target_nx=rollout_nx,
            dtype=np.float64,
        )
        central_heat_flux = _central_heat_flux_numpy(
            current_coeff,
            current_state,
            source_nx=source_nx,
            target_nx=rollout_nx,
        )
        k_arr = 2.0 * math.pi * np.fft.rfftfreq(
            int(rollout_nx), d=float(domain_length) / float(rollout_nx)
        )
        heat_flux_gradient_targets.append(
            np.fft.irfft(
                1j * k_arr * np.fft.rfft(central_heat_flux, axis=-1),
                n=int(rollout_nx),
                axis=-1,
            )
        )
        if include_heat_flux_gradient_history:
            memory_state = low_hermite_coefficients_to_conservative(
                memory_coeff,
                source_nx=source_nx,
                target_nx=rollout_nx,
                dtype=np.float64,
            )
            memory_heat_flux = _central_heat_flux_numpy(
                memory_coeff,
                memory_state,
                source_nx=source_nx,
                target_nx=rollout_nx,
            )
            memory_gradient = np.fft.irfft(
                1j * k_arr * np.fft.rfft(memory_heat_flux, axis=-1),
                n=int(rollout_nx),
                axis=-1,
            )
            if not burnin_steps:
                memory_gradient[padded_memory] = 0.0
            else:
                memory_gradient[...] = 0.0
            heat_flux_gradient_histories.append(memory_gradient)
        if heat_flux_gradient_case_scale_by_id is not None:
            heat_flux_gradient_case_scales.append(
                np.asarray(
                    [
                        heat_flux_gradient_case_scale_by_id[str(case_ids[index])]
                        for index in cases
                    ],
                    dtype=np.float64,
                )
            )
        amplitudes.append(
            np.asarray([amplitudes_by_id[str(case_ids[index])] for index in cases])
        )
        regime_indices.append(np.full((len(cases),), regime_index, dtype=np.int32))
        start_indices.append(
            np.rint(times / reference_ratio).astype(np.int32) + burnin_steps
        )
    batch = {
        "memory": np.concatenate(memories, axis=0).astype(np.float32),
        "initial": np.concatenate(initials, axis=0).astype(np.float32),
        "targets": np.concatenate(targets, axis=0).astype(np.float32),
        "amplitude": np.concatenate(amplitudes, axis=0).astype(np.float32),
        "regime_index": np.concatenate(regime_indices, axis=0),
        "start_index": np.concatenate(start_indices, axis=0).astype(np.int32),
        "heat_flux_gradient_target": np.concatenate(
            heat_flux_gradient_targets, axis=0
        ).astype(np.float32),
    }
    if include_heat_flux_gradient_history:
        batch["heat_flux_gradient_history"] = np.concatenate(
            heat_flux_gradient_histories, axis=0
        ).astype(np.float32)
    if heat_flux_gradient_case_scale_by_id is not None:
        batch["heat_flux_gradient_case_scale"] = np.concatenate(
            heat_flux_gradient_case_scales, axis=0
        ).astype(np.float32)
    if translation_augmentation:
        shifts = rng.integers(0, int(rollout_nx), size=batch["initial"].shape[0])
        for row, shift in enumerate(shifts):
            batch["memory"][row] = np.roll(batch["memory"][row], int(shift), axis=-1)
            batch["initial"][row] = np.roll(batch["initial"][row], int(shift), axis=-1)
            batch["targets"][row] = np.roll(batch["targets"][row], int(shift), axis=-1)
            batch["heat_flux_gradient_target"][row] = np.roll(
                batch["heat_flux_gradient_target"][row], int(shift), axis=-1
            )
            if include_heat_flux_gradient_history:
                batch["heat_flux_gradient_history"][row] = np.roll(
                    batch["heat_flux_gradient_history"][row], int(shift), axis=-1
                )
    return batch


def build_complete_trajectory_case_batches(
    rng: np.random.Generator,
    grouped,
    *,
    regimes: Sequence[str] = REGIMES,
    split: str,
    batch_size_per_regime: int,
    shuffle: bool,
) -> Sequence[Dict[str, np.ndarray]]:
    """Return balanced case batches that cover every IC in a split once."""
    split_value = IC_SPLIT_TRAIN if split == "train" else IC_SPLIT_HELDOUT
    rows_by_regime = {}
    for regime in regimes:
        rows = np.flatnonzero(
            np.asarray(grouped[regime]["case_splits"], dtype=np.str_) == split_value
        ).astype(np.int32)
        if rows.size == 0:
            raise ValueError(f"No {split} trajectories for {regime}")
        rows_by_regime[regime] = rng.permutation(rows) if shuffle else rows
    counts = {int(rows.size) for rows in rows_by_regime.values()}
    if len(counts) != 1:
        raise ValueError(
            f"Complete-trajectory training requires equal per-regime IC counts, got {counts}"
        )
    batch_size = max(int(batch_size_per_regime), 1)
    case_count = counts.pop()
    return tuple(
        {
            regime: rows_by_regime[regime][start : start + batch_size]
            for regime in regimes
        }
        for start in range(0, case_count, batch_size)
    )


def build_balanced_random_window_epoch(
    rng: np.random.Generator,
    anchors,
    *,
    regimes: Sequence[str] = REGIMES,
    split: str,
    batch_size_per_regime: int,
    time_blocks: int = 6,
    deterministic_epoch: Optional[int] = None,
) -> Sequence[Mapping[str, Tuple[np.ndarray, np.ndarray]]]:
    """Cover every available IC exactly once with balanced random windows."""
    batch_size = int(batch_size_per_regime)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    cases_by_regime = {}
    for regime in regimes:
        case_pool = np.asarray(anchors[regime][f"{split}_cases"], dtype=np.int32)
        unique = np.unique(case_pool)
        cases_by_regime[regime] = (
            unique if deterministic_epoch is not None else rng.permutation(unique)
        )
    counts = {len(value) for value in cases_by_regime.values()}
    if len(counts) != 1:
        raise ValueError(f"Balanced random windows require equal regime counts: {counts}")
    count = counts.pop()
    selections = []
    for start in range(0, count, batch_size):
        selection = {}
        for regime in regimes:
            cases = cases_by_regime[regime][start : start + batch_size]
            case_pool = np.asarray(anchors[regime][f"{split}_cases"], dtype=np.int32)
            time_pool = np.asarray(anchors[regime][f"{split}_times"], dtype=np.int32)
            times = []
            for offset, case_index in enumerate(cases):
                available = time_pool[case_pool == int(case_index)]
                if deterministic_epoch is not None:
                    position = (
                        int(deterministic_epoch) - 1 + 131 * int(case_index)
                    ) % int(available.size)
                    times.append(available[position])
                    continue
                block_count = max(int(time_blocks), 1)
                block_index = (start // batch_size + offset) % block_count
                edges = np.linspace(0, available.size, block_count + 1, dtype=np.int32)
                left, right = int(edges[block_index]), int(edges[block_index + 1])
                if right <= left:
                    left, right = 0, int(available.size)
                times.append(available[int(rng.integers(left, right))])
            selection[regime] = (cases, np.asarray(times, dtype=np.int32))
        selections.append(selection)
    return tuple(selections)


def _full_anchor_steps_per_epoch(anchors, regimes, batch_size, accumulation_steps):
    """Count complete selected-regime updates without consuming the training RNG."""
    counts = {int(anchors[regime]["train_cases"].size) for regime in regimes}
    if len(counts) != 1 or min(counts) <= 0:
        raise ValueError(f"Full-anchor sweep requires equal nonempty regime counts: {counts}")
    count = counts.pop()
    samples_per_update = int(batch_size) * int(accumulation_steps)
    if samples_per_update <= 0 or count % samples_per_update:
        raise ValueError(
            "Full-anchor count must divide by batch size * gradient accumulation; "
            f"got {count} and {samples_per_update}"
        )
    return count // samples_per_update


def build_balanced_full_anchor_epoch(
    rng: np.random.Generator,
    anchors,
    *,
    regimes: Sequence[str] = REGIMES,
    split: str,
    batch_size_per_regime: int,
) -> Sequence[Mapping[str, Tuple[np.ndarray, np.ndarray]]]:
    """Shuffle and consume every anchor exactly once in balanced minibatches."""
    batch_size = int(batch_size_per_regime)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    shuffled = {}
    counts = set()
    for regime in regimes:
        cases = np.asarray(anchors[regime][f"{split}_cases"], dtype=np.int32)
        times = np.asarray(anchors[regime][f"{split}_times"], dtype=np.int32)
        if cases.size != times.size or cases.size == 0:
            raise ValueError(f"Invalid {split} anchor table for {regime}")
        order = rng.permutation(cases.size)
        shuffled[regime] = (cases[order], times[order])
        counts.add(int(cases.size))
    if len(counts) != 1:
        raise ValueError(f"Balanced full sweep requires equal regime counts: {counts}")
    count = counts.pop()
    if count % batch_size:
        raise ValueError(
            "Full-anchor sweep requires the per-regime anchor count to be divisible "
            f"by batch size; got {count} and {batch_size}"
        )
    return tuple(
        {
            regime: (
                shuffled[regime][0][start : start + batch_size],
                shuffled[regime][1][start : start + batch_size],
            )
            for regime in regimes
        }
        for start in range(0, count, batch_size)
    )


def sample_complete_trajectory_batch(
    rng: np.random.Generator,
    grouped,
    manifest,
    coefficient_key: str,
    case_rows: Mapping[str, np.ndarray],
    *,
    regimes: Sequence[str] = REGIMES,
    memory_steps: int,
    memory_stride: int,
    source_nx: int,
    rollout_nx: int,
    translation_augmentation: bool,
    primitive_target_histories: Optional[
        Mapping[str, Sequence[np.ndarray]]
    ] = None,
    trajectory_target_norm_by_id: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Load complete, balanced trajectories initialized only at ``t=0``."""
    del memory_stride  # Every pre-initial memory sample is clamped to t=0.
    amplitudes_by_id = _case_amplitudes(manifest)
    memories = []
    initials = []
    targets = []
    target_fields = []
    target_norms = []
    amplitudes = []
    regime_indices = []
    trajectory_steps = None
    for regime in regimes:
        regime_index = REGIMES.index(regime)
        group = grouped[regime]
        histories = tuple(group[coefficient_key])
        case_ids = np.asarray(group["case_ids"], dtype=np.str_)
        cases = np.asarray(case_rows[regime], dtype=np.int32)
        available_steps = min(int(histories[index].shape[0]) - 1 for index in cases)
        trajectory_steps = (
            available_steps
            if trajectory_steps is None
            else min(int(trajectory_steps), available_steps)
        )
        initial_times = np.zeros((cases.size,), dtype=np.int32)
        initial_coeff = _gather_coefficients(
            histories, cases, initial_times, coefficient_count=4
        )
        memories.append(
            np.zeros(
                (cases.size, int(memory_steps), 3, int(rollout_nx)),
                dtype=np.float64,
            )
        )
        initials.append(
            low_hermite_coefficients_to_conservative(
                initial_coeff, source_nx=source_nx, target_nx=rollout_nx
            )
        )
        if primitive_target_histories is None:
            target_times = np.broadcast_to(
                np.arange(1, available_steps + 1, dtype=np.int32)[None, :],
                (cases.size, available_steps),
            )
            target_coeff = _gather_coefficients(
                histories, cases[:, None], target_times, coefficient_count=3
            )
            targets.append(
                low_hermite_coefficients_to_conservative(
                    target_coeff, source_nx=source_nx, target_nx=rollout_nx
                )
            )
        else:
            cached_histories = primitive_target_histories[regime]
            target_fields.append(
                np.stack(
                    [
                        np.asarray(cached_histories[int(index)][1 : available_steps + 1])
                        for index in cases
                    ],
                    axis=0,
                )
            )
            if trajectory_target_norm_by_id is None:
                raise ValueError("Cached primitive targets require trajectory norms")
            target_norms.append(
                np.stack(
                    [
                        np.asarray(
                            trajectory_target_norm_by_id[str(case_ids[index])],
                            dtype=np.float64,
                        )
                        for index in cases
                    ],
                    axis=0,
                )
            )
        amplitudes.append(
            np.asarray([amplitudes_by_id[str(case_ids[index])] for index in cases])
        )
        regime_indices.append(np.full((cases.size,), regime_index, dtype=np.int32))
    if trajectory_steps is None:
        raise ValueError("No complete trajectories selected")
    batch = {
        "memory": np.concatenate(memories, axis=0).astype(np.float32),
        "initial": np.concatenate(initials, axis=0).astype(np.float32),
        "amplitude": np.concatenate(amplitudes, axis=0).astype(np.float32),
        "regime_index": np.concatenate(regime_indices, axis=0),
    }
    if primitive_target_histories is None:
        batch["targets"] = np.concatenate(
            [value[:, :trajectory_steps] for value in targets], axis=0
        ).astype(np.float32)
    else:
        batch["target_fields"] = np.concatenate(
            [value[:, :trajectory_steps] for value in target_fields], axis=0
        ).astype(np.float32)
        batch["trajectory_target_norm"] = np.concatenate(target_norms, axis=0)
    if translation_augmentation:
        shifts = rng.integers(0, int(rollout_nx), size=batch["initial"].shape[0])
        for row, shift in enumerate(shifts):
            batch["memory"][row] = np.roll(
                batch["memory"][row], int(shift), axis=-1
            )
            batch["initial"][row] = np.roll(
                batch["initial"][row], int(shift), axis=-1
            )
            target_key = "target_fields" if "target_fields" in batch else "targets"
            batch[target_key][row] = np.roll(
                batch[target_key][row], int(shift), axis=-1
            )
    return batch


def build_diagnostic_panel(
    rng: np.random.Generator,
    grouped,
    anchors,
    manifest,
    coefficient_key: str,
    *,
    regimes: Sequence[str] = REGIMES,
    split: str,
    start_times: Sequence[float],
    cases_per_regime: Optional[int],
    horizon: int,
    dt: float,
    memory_steps: int,
    memory_stride: int,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    heat_flux_gradient_case_scale_by_id: Optional[Mapping[str, float]] = None,
    include_heat_flux_gradient_history: bool = False,
    autonomous_burnin_steps: int = 0,
    reference_dt: Optional[float] = None,
) -> Sequence[Dict[str, np.ndarray]]:
    amplitudes = _case_amplitudes(manifest)
    selections = {}
    for regime in regimes:
        group = grouped[regime]
        split_value = IC_SPLIT_TRAIN if split == "train" else IC_SPLIT_HELDOUT
        rows = np.flatnonzero(np.asarray(group["case_splits"]) == split_value)
        rows = np.asarray(
            sorted(rows, key=lambda row: amplitudes[str(group["case_ids"][row])]),
            dtype=np.int32,
        )
        if cases_per_regime is not None and int(cases_per_regime) < int(rows.size):
            positions = np.rint(
                np.linspace(0, rows.size - 1, int(cases_per_regime))
            ).astype(np.int32)
            rows = rows[positions]
        selections[regime] = rows
    panel = []
    reference_dt_value = float(dt if reference_dt is None else reference_dt)
    reference_ratio = float(dt) / reference_dt_value
    for start_time in start_times:
        time_index = int(round(float(start_time) / reference_dt_value))
        explicit = {}
        for regime in regimes:
            rows = selections[regime]
            histories = tuple(grouped[regime][coefficient_key])
            required_reference_steps = int(
                math.ceil(
                    (int(horizon) + int(autonomous_burnin_steps))
                    * reference_ratio
                )
            )
            if any(
                time_index + required_reference_steps >= int(histories[row].shape[0])
                for row in rows
            ):
                raise ValueError(
                    f"Diagnostic start t={start_time:g} exceeds the valid H={horizon} window"
                )
            explicit[regime] = (
                rows,
                np.full(rows.shape, time_index, dtype=np.int32),
            )
        panel.append(
            sample_batch(
                rng,
                grouped,
                anchors,
                manifest,
                coefficient_key,
                regimes=regimes,
                split=split,
                batch_size_per_regime=len(selections[regimes[0]]),
                horizon=horizon,
                memory_steps=memory_steps,
                memory_stride=memory_stride,
                source_nx=source_nx,
                rollout_nx=rollout_nx,
                domain_length=domain_length,
                translation_augmentation=False,
                explicit_selection=explicit,
                heat_flux_gradient_case_scale_by_id=(
                    heat_flux_gradient_case_scale_by_id
                ),
                include_heat_flux_gradient_history=(
                    include_heat_flux_gradient_history
                ),
                autonomous_burnin_steps=autonomous_burnin_steps,
                reference_steps_per_solver_step=reference_ratio,
            )
        )
    return panel
