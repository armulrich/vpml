"""Compare low-moment closure identifiability on single- and multimode ICs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vpml.jax_runtime import bootstrap_jax_runtime, print_jax_runtime_summary

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
from scipy.spatial import cKDTree

from model.train.interface_flux_data import load_ic_manifest
from model.train.low_moment_closure import (
    _central_heat_flux_numpy,
    low_hermite_coefficients_to_conservative,
)
from vpml.low_moment import primitive_fields


def _state_and_gradient(
    history: np.ndarray,
    indices: np.ndarray,
    *,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    coeff = np.asarray(history[np.asarray(indices, dtype=np.int32), :4])
    state = low_hermite_coefficients_to_conservative(
        coeff, source_nx=source_nx, target_nx=rollout_nx, dtype=np.float64
    )
    heat_flux = _central_heat_flux_numpy(
        coeff, state, source_nx=source_nx, target_nx=rollout_nx
    )
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(
        rollout_nx, d=domain_length / rollout_nx
    )
    gradient = np.fft.irfft(
        1j * k_arr * np.fft.rfft(heat_flux, axis=-1),
        n=rollout_nx,
        axis=-1,
    )
    return state, gradient


def _case_samples(
    history: np.ndarray,
    *,
    case_id: str,
    split: str,
    dt: float,
    start_time: float,
    end_time: float,
    sample_stride: int,
    memory_steps: int,
    memory_stride: int,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    modes: int,
    include_closure_history: bool,
) -> dict[str, np.ndarray]:
    first = max(int(round(start_time / dt)), memory_steps * memory_stride + 1)
    last = min(int(round(end_time / dt)), int(history.shape[0]) - 1)
    times = np.arange(first, last + 1, sample_stride, dtype=np.int32)
    offsets = memory_stride * np.arange(memory_steps - 1, -1, -1, dtype=np.int32)
    history_indices = times[:, None] - offsets[None]
    needed_parts = [history_indices.reshape(-1), times]
    if include_closure_history:
        needed_parts.append((history_indices - 1).reshape(-1))
    needed = np.unique(np.concatenate(needed_parts))
    state_flat, gradient_flat = _state_and_gradient(
        history,
        needed,
        source_nx=source_nx,
        rollout_nx=rollout_nx,
        domain_length=domain_length,
    )
    state = state_flat[np.searchsorted(needed, history_indices)]
    target_gradient = gradient_flat[np.searchsorted(needed, times)]
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(
        rollout_nx, d=domain_length / rollout_nx
    )
    flat_state = state.reshape((-1, 3, rollout_nx))
    fields = np.asarray(
        primitive_fields(jnp.asarray(flat_state), jnp.asarray(k_arr), poisson_sign=1.0)
    ).reshape((times.size, memory_steps, 4, rollout_nx))
    field_hat = np.fft.rfft(fields, axis=-1)[..., 1 : modes + 1] / rollout_nx
    target_hat = np.fft.rfft(target_gradient, axis=-1)[..., 1 : modes + 1] / rollout_nx

    current_density_hat = field_hat[:, -1, 0]
    phase = np.angle(current_density_hat[:, 0])
    mode_numbers = np.arange(1, modes + 1, dtype=np.float64)
    rotation = np.exp(-1j * phase[:, None] * mode_numbers[None])
    field_hat *= rotation[:, None, None, :]
    target_hat *= rotation

    amplitude = np.sqrt(np.mean(fields[:, -1, 0] ** 2, axis=-1))
    amplitude = np.maximum(amplitude, 1e-12)
    field_hat /= amplitude[:, None, None, None]
    target_hat /= amplitude[:, None]
    channels = [np.concatenate((field_hat.real, field_hat.imag), axis=-1)]

    if include_closure_history:
        gradient_history = gradient_flat[
            np.searchsorted(needed, history_indices - 1)
        ]
        gradient_hat = (
            np.fft.rfft(gradient_history, axis=-1)[..., 1 : modes + 1]
            / rollout_nx
        )
        gradient_hat *= rotation[:, None, :]
        gradient_hat /= amplitude[:, None, None]
        channels.append(
            np.concatenate((gradient_hat.real, gradient_hat.imag), axis=-1)[:, :, None]
        )

    history_features = np.concatenate(channels, axis=2).reshape(times.size, memory_steps, -1)
    target = np.concatenate((target_hat.real, target_hat.imag), axis=-1)
    return {
        "history": history_features,
        "target": target,
        "log_amplitude": np.log(amplitude)[:, None],
        "case_id": np.full((times.size,), case_id),
        "split": np.full((times.size,), split),
    }


def _combine(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([row[key] for row in rows], axis=0) for key in rows[0]}


def _features(dataset: dict[str, np.ndarray], temporal_rank: int) -> np.ndarray:
    history = dataset["history"]
    current = history[:, -1]
    if temporal_rank <= 0:
        return np.concatenate((current, dataset["log_amplitude"]), axis=1)
    coefficients = dct(history, axis=1, norm="ortho")[:, :temporal_rank]
    return np.concatenate(
        (current, coefficients.reshape(coefficients.shape[0], -1), dataset["log_amplitude"]),
        axis=1,
    )


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    neighbors: int,
    projection_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    active = scale > 1e-10
    train = (train_x[:, active] - center[active]) / scale[active]
    test = (test_x[:, active] - center[active]) / scale[active]
    if train.shape[1] > projection_dim:
        rng = np.random.default_rng(seed)
        projection = rng.normal(
            size=(train.shape[1], projection_dim)
        ) / math.sqrt(projection_dim)
        train = train @ projection
        test = test @ projection
    tree = cKDTree(train)
    distance, nearest = tree.query(test, k=min(neighbors, train.shape[0]))
    if nearest.ndim == 1:
        nearest = nearest[:, None]
        distance = distance[:, None]
    return np.mean(train_y[nearest], axis=1), np.mean(distance, axis=1)


def _score(
    dataset: dict[str, np.ndarray],
    temporal_rank: int,
    *,
    leave_one_case_out: bool,
    neighbors: int,
    projection_dim: int,
) -> dict[str, float]:
    features = _features(dataset, temporal_rank)
    target = dataset["target"]
    predictions = np.empty_like(target)
    distances = np.empty((target.shape[0],), dtype=np.float64)
    if leave_one_case_out:
        groups = np.unique(dataset["case_id"])
        masks = [(dataset["case_id"] != group, dataset["case_id"] == group) for group in groups]
    else:
        masks = [(dataset["split"] == "train", dataset["split"] == "heldout")]
    test_union = np.zeros((target.shape[0],), dtype=bool)
    for fold, (train_mask, test_mask) in enumerate(masks):
        prediction, distance = _fit_predict(
            features[train_mask],
            target[train_mask],
            features[test_mask],
            neighbors=neighbors,
            projection_dim=projection_dim,
            seed=1729 + fold,
        )
        predictions[test_mask] = prediction
        distances[test_mask] = distance
        test_union |= test_mask
    truth = target[test_union]
    prediction = predictions[test_union]
    train_target = target[~test_union] if not leave_one_case_out else target
    baseline_center = np.mean(train_target, axis=0)
    error = float(np.mean(np.sum((prediction - truth) ** 2, axis=1)))
    baseline = float(np.mean(np.sum((truth - baseline_center) ** 2, axis=1)))
    target_energy = float(np.sum(truth * truth))
    prediction_energy = float(np.sum(prediction * prediction))
    cross = float(np.sum(prediction * truth))
    return {
        "normalized_knn_mse": error / max(baseline, 1e-30),
        "relative_target_mse": float(np.sum((prediction - truth) ** 2)) / max(target_energy, 1e-30),
        "least_squares_gain": cross / max(target_energy, 1e-30),
        "prediction_target_cosine": cross / max(math.sqrt(prediction_energy * target_energy), 1e-30),
        "mean_neighbor_distance": float(np.mean(distances[test_union])),
        "test_samples": int(np.sum(test_union)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimode-cache", type=Path, required=True)
    parser.add_argument("--single-mode-history", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--temporal-ranks", default="0,1,4,8,16,32,50")
    parser.add_argument("--memory-steps", type=int, default=50)
    parser.add_argument("--memory-stride", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--start-time", type=float, default=20.0)
    parser.add_argument("--end-time", type=float, default=60.0)
    args = parser.parse_args()
    print_jax_runtime_summary(jax, context="single-vs-multimode identifiability")

    multimode_cache = args.multimode_cache.resolve()
    metadata = json.loads((multimode_cache / "metadata.json").read_text())
    configuration = metadata["configuration"]
    manifest = load_ic_manifest(multimode_cache / "ic_manifest.json")
    dt = float(configuration["teacher_dt"])
    domain_length = float(configuration["teacher_L"])
    rollout_nx = 256

    reports: dict[str, dict] = {}
    for include_closure_history in (False, True):
        suffix = "state_plus_closure_history" if include_closure_history else "state_history"
        multimode_rows = []
        for case in manifest["cases"]:
            if str(case["regime"]) != "nonlinear_landau_strong":
                continue
            history = np.load(
                multimode_cache / "cases" / f"{case['case_id']}.npy", mmap_mode="r"
            )
            multimode_rows.append(
                _case_samples(
                    history,
                    case_id=str(case["case_id"]),
                    split=str(case["split"]),
                    dt=dt,
                    start_time=args.start_time,
                    end_time=args.end_time,
                    sample_stride=args.sample_stride,
                    memory_steps=args.memory_steps,
                    memory_stride=args.memory_stride,
                    source_nx=int(configuration["teacher_Nx"]),
                    rollout_nx=rollout_nx,
                    domain_length=domain_length,
                    modes=args.modes,
                    include_closure_history=include_closure_history,
                )
            )
        single_history = np.load(args.single_mode_history.resolve(), mmap_mode="r")
        single_rows = [
            _case_samples(
                single_history[index],
                case_id=f"single_strong_{index:02d}",
                split="heldout" if index in (1, 3, 5) else "train",
                dt=0.01,
                start_time=args.start_time,
                end_time=args.end_time,
                sample_stride=args.sample_stride,
                memory_steps=args.memory_steps,
                memory_stride=args.memory_stride,
                source_nx=256,
                rollout_nx=rollout_nx,
                domain_length=domain_length,
                modes=args.modes,
                include_closure_history=include_closure_history,
            )
            for index in range(single_history.shape[0])
        ]
        datasets = {
            f"single_mode_{suffix}": (_combine(single_rows), False),
            f"multimode_{suffix}": (_combine(multimode_rows), False),
        }
        ranks = [int(value) for value in args.temporal_ranks.split(",")]
        for label, (dataset, leave_one_case_out) in datasets.items():
            print(f"[identifiability] {label}: samples={dataset['target'].shape[0]}")
            reports[label] = {
                str(rank): _score(
                    dataset,
                    rank,
                    leave_one_case_out=leave_one_case_out,
                    neighbors=args.neighbors,
                    projection_dim=args.projection_dim,
                )
                for rank in ranks
            }

    report = {
        "configuration": {
            "memory_steps": args.memory_steps,
            "memory_stride": args.memory_stride,
            "memory_span": args.memory_steps * args.memory_stride * dt,
            "modes": args.modes,
            "projection_dim": args.projection_dim,
            "single_mode_split": "heldout interpolation amplitudes 0.25, 0.35, 0.50",
            "multimode_split": "manifest train/heldout",
            "single_mode_interval": [args.start_time, args.end_time],
            "multimode_interval": [args.start_time, args.end_time],
        },
        "scores": reports,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    ranks = [int(value) for value in args.temporal_ranks.split(",")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    for ax, include_label in zip(axes, ("state_history", "state_plus_closure_history")):
        for family, color in (("single_mode", "#2b6cb0"), ("multimode", "#c53030")):
            values = reports[f"{family}_{include_label}"]
            ax.plot(
                ranks,
                [values[str(rank)]["normalized_knn_mse"] for rank in ranks],
                marker="o",
                color=color,
                label=family.replace("_", " "),
            )
        ax.set_xlabel("Temporal DCT coefficients retained")
        ax.set_ylabel("Held-out closure kNN NMSE")
        ax.set_title(include_label.replace("_", " "))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(args.outdir / "single_vs_multimode_identifiability.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
