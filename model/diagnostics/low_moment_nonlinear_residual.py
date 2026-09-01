"""Diagnose strong-nonlinear residuals of a trained low-moment closure."""

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

from model.train.interface_flux_data import load_ic_manifest
from model.train.low_moment_closure import (
    REGIMES,
    _central_heat_flux_numpy,
    _load_checkpoint,
    low_hermite_coefficients_to_conservative,
)
from vpml.low_moment import explicit_window_closure_step, low_moment_rhs, primitive_fields


def _case_map(cache_dir: Path) -> tuple[dict[str, dict], dict]:
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    manifest = load_ic_manifest(cache_dir / "ic_manifest.json")
    return {str(case["case_id"]): dict(case) for case in manifest["cases"]}, metadata


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
        1j * k_arr * np.fft.rfft(heat_flux, axis=-1), n=rollout_nx, axis=-1
    )
    return state, gradient


def _align_spectra(values: np.ndarray, density_hat: np.ndarray) -> np.ndarray:
    """Remove global translations using the phase of the first density mode."""
    modes = np.arange(values.shape[-1], dtype=np.float64)
    phase = np.angle(density_hat[..., 1])
    rotation = np.exp(-1j * phase[..., None] * modes)
    return values * rotation[..., None, :]


def _spectral_features(
    state: np.ndarray,
    gradient: np.ndarray,
    *,
    domain_length: float,
    modes: int,
    include_gradient: bool,
) -> tuple[np.ndarray, np.ndarray]:
    nx = state.shape[-1]
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(nx, d=domain_length / nx)
    fields = np.asarray(
        primitive_fields(jnp.asarray(state), jnp.asarray(k_arr), poisson_sign=1.0)
    )
    field_hat = np.fft.rfft(fields, axis=-1)[..., : modes + 1]
    gradient_hat = np.fft.rfft(gradient, axis=-1)[..., : modes + 1]
    density_hat = field_hat[..., 0, :]
    field_hat = _align_spectra(field_hat, density_hat)
    gradient_hat = _align_spectra(gradient_hat[..., None, :], density_hat)[..., 0, :]
    channels = [field_hat[..., 1 : modes + 1]]
    if include_gradient:
        channels.append(gradient_hat[..., None, 1 : modes + 1])
    joined = np.concatenate(channels, axis=-2)
    features = np.concatenate((joined.real, joined.imag), axis=-1).reshape(state.shape[0], -1)
    target = np.concatenate(
        (gradient_hat[..., 1 : modes + 1].real, gradient_hat[..., 1 : modes + 1].imag),
        axis=-1,
    )
    return features, target


def _history_dataset(
    cache_dir: Path,
    cases: dict[str, dict],
    *,
    split: str,
    regime: str,
    span: float,
    common_start: float,
    include_gradient: bool,
    dt: float,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    sample_stride: int,
    lag_count: int,
    modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_features, all_targets = [], []
    span_steps = int(round(span / dt))
    start = max(int(round(common_start / dt)), span_steps, 1)
    for case_id, case in cases.items():
        if str(case["split"]) != split or str(case["regime"]) != regime:
            continue
        history = np.load(cache_dir / "cases" / f"{case_id}.npy", mmap_mode="r")
        times = np.arange(start, history.shape[0], sample_stride, dtype=np.int32)
        if span_steps:
            offsets = np.rint(np.linspace(span_steps, 0, lag_count)).astype(np.int32)
        else:
            offsets = np.zeros((1,), dtype=np.int32)
        indices = times[:, None] - offsets[None]
        flat = np.unique(
            np.concatenate((indices.reshape(-1), (indices - 1).reshape(-1)))
        )
        state_flat, gradient_flat = _state_and_gradient(
            history,
            flat,
            source_nx=source_nx,
            rollout_nx=rollout_nx,
            domain_length=domain_length,
        )
        lookup = np.searchsorted(flat, indices)
        state = state_flat[lookup]
        # Closure history is causal: the newest available value is at t-dt.
        gradient = gradient_flat[np.searchsorted(flat, indices - 1)]
        feature_rows = []
        for lag in range(state.shape[1]):
            feature, _ = _spectral_features(
                state[:, lag], gradient[:, lag], domain_length=domain_length,
                modes=modes, include_gradient=include_gradient,
            )
            feature_rows.append(feature)
        target_gradient = gradient_flat[np.searchsorted(flat, times)]
        _, target = _spectral_features(
            state[:, -1], target_gradient, domain_length=domain_length,
            modes=modes, include_gradient=False,
        )
        feature_history = np.stack(feature_rows, axis=1)
        temporal_summary = np.concatenate(
            (
                feature_history[:, -1],
                np.mean(feature_history, axis=1),
                np.std(feature_history, axis=1),
                feature_history[:, -1] - feature_history[:, 0],
            ),
            axis=1,
        )
        amplitude = np.full((times.size, 1), math.log(float(case["epsilon"])))
        all_features.append(np.concatenate((temporal_summary, amplitude), axis=1))
        all_targets.append(target)
    return np.concatenate(all_features), np.concatenate(all_targets)


def _knn_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    neighbors: int,
    projection_dim: int,
    seed: int,
) -> dict[str, float]:
    center = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    active = scale > 1e-10
    train = (train_x[:, active] - center[active]) / scale[active]
    test = (test_x[:, active] - center[active]) / scale[active]
    rng = np.random.default_rng(seed)
    dim = min(projection_dim, train.shape[1])
    projection = rng.normal(size=(train.shape[1], dim)) / math.sqrt(dim)
    train = train @ projection
    test = test @ projection
    target_center = np.mean(train_y, axis=0)
    baseline = float(np.mean(np.sum((test_y - target_center) ** 2, axis=1)))
    error_sum = 0.0
    count = 0
    for start in range(0, test.shape[0], 128):
        query = test[start : start + 128]
        distance = (
            np.sum(query * query, axis=1)[:, None]
            + np.sum(train * train, axis=1)[None]
            - 2.0 * query @ train.T
        )
        nearest = np.argpartition(distance, neighbors - 1, axis=1)[:, :neighbors]
        prediction = np.mean(train_y[nearest], axis=1)
        truth = test_y[start : start + query.shape[0]]
        error_sum += float(np.sum(np.sum((prediction - truth) ** 2, axis=1)))
        count += query.shape[0]
    mse = error_sum / max(count, 1)
    return {"normalized_knn_mse": mse / baseline, "baseline_mse": baseline, "mse": mse}


def _teacher_forced_checkpoint_score(
    checkpoint: Path,
    cache_dir: Path,
    cases: dict[str, dict],
    *,
    dt: float,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    sample_stride: int,
    closure_history_mode: str = "exact",
    split: str = "heldout",
) -> dict:
    params, metadata, stats = _load_checkpoint(checkpoint)
    selected_case_ids = None
    case_split = split
    if split == "train_selected":
        case_split = "train"
        selected_by_regime = metadata.get("selected_training_case_ids")
        if not selected_by_regime:
            raise ValueError(
                f"Checkpoint does not record selected training cases: {checkpoint}"
            )
        selected_case_ids = {
            str(case_id)
            for regime_case_ids in selected_by_regime.values()
            for case_id in regime_case_ids
        }
    memory_steps = int(metadata["memory_steps"])
    memory_stride = int(metadata["memory_stride"])
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(rollout_nx, d=domain_length / rollout_nx)
    predict = jax.jit(
        lambda state, memory, amplitude, previous, gradient_history: explicit_window_closure_step(
            params,
            state,
            memory,
            amplitude,
            jnp.asarray(k_arr, dtype=jnp.float32),
            input_scale=jnp.asarray(stats["input_scale"], dtype=jnp.float32),
            heat_flux_gradient_scale=float(stats["heat_flux_gradient_scale"][0]),
            amplitude_center=float(stats["amplitude_center"][0]),
            amplitude_scale=float(stats["amplitude_scale"][0]),
            previous_heat_flux_gradient=(
                previous if bool(metadata.get("closure_history_input", False)) else None
            ),
            heat_flux_gradient_history=(
                gradient_history
                if bool(metadata.get("closure_history_input", False))
                else None
            ),
            poisson_sign=1.0,
            normalized_heat_flux_bound=float(metadata["normalized_heat_flux_bound"]),
            input_scaling=str(metadata.get("input_scaling", "fixed_training_scale")),
            dynamic_amplitude_floor=float(
                metadata.get("dynamic_amplitude_floor", 1e-6)
            ),
            allow_uniform_heating=bool(
                metadata.get("allow_uniform_heating", False)
            ),
        )
    )
    totals = {
        regime: {
            "error": 0.0, "target": 0.0, "prediction": 0.0, "cross": 0.0,
            "mean": 0.0, "residual": 0.0, "count": 0,
            "spectral_error": np.zeros((rollout_nx // 2 + 1,), dtype=np.float64),
            "spectral_target": np.zeros((rollout_nx // 2 + 1,), dtype=np.float64),
        }
        for regime in REGIMES
    }
    time_blocks = {regime: [] for regime in REGIMES}
    strong_growth, strong_residual = [], []
    for case_id, case in cases.items():
        if str(case["split"]) != case_split:
            continue
        if selected_case_ids is not None and case_id not in selected_case_ids:
            continue
        regime = str(case["regime"])
        history = np.load(cache_dir / "cases" / f"{case_id}.npy", mmap_mode="r")
        start = memory_steps * memory_stride + 1
        times = np.arange(start, history.shape[0], sample_stride, dtype=np.int32)
        block_values = []
        for block_start in range(0, times.size, 32):
            selected = times[block_start : block_start + 32]
            offsets = memory_stride * np.arange(memory_steps, 0, -1, dtype=np.int32)
            memory_indices = selected[:, None] - offsets[None]
            needed = np.unique(np.concatenate((selected, selected - 1, memory_indices.reshape(-1))))
            states, gradients = _state_and_gradient(
                history, needed, source_nx=source_nx, rollout_nx=rollout_nx,
                domain_length=domain_length,
            )
            current = states[np.searchsorted(needed, selected)]
            target = gradients[np.searchsorted(needed, selected)]
            previous = gradients[np.searchsorted(needed, selected - 1)]
            memory = states[np.searchsorted(needed, memory_indices)]
            gradient_history = gradients[np.searchsorted(needed, memory_indices)]
            amplitude = np.full((selected.size,), float(case["epsilon"]), dtype=np.float32)
            if closure_history_mode == "zero":
                previous = np.zeros_like(previous)
                gradient_history = np.zeros_like(gradient_history)
            elif closure_history_mode != "exact":
                raise ValueError(
                    f"Unsupported closure-history mode: {closure_history_mode}"
                )
            predicted = np.asarray(
                predict(current, memory, amplitude, previous, gradient_history)
            )
            residual = predicted - target
            target_energy = np.sum(target * target, axis=1)
            residual_energy = np.sum(residual * residual, axis=1)
            block_values.extend((residual_energy / np.maximum(target_energy, 1e-30)).tolist())
            entry = totals[regime]
            entry["error"] += float(np.sum(residual_energy))
            entry["target"] += float(np.sum(target_energy))
            entry["prediction"] += float(np.sum(predicted * predicted))
            entry["cross"] += float(np.sum(predicted * target))
            entry["mean"] += float(np.sum(np.square(np.mean(residual, axis=1))))
            entry["residual"] += float(np.sum(np.mean(residual * residual, axis=1)))
            entry["count"] += int(selected.size)
            residual_hat = np.fft.rfft(residual, axis=-1)
            target_hat = np.fft.rfft(target, axis=-1)
            entry["spectral_error"] += np.sum(np.abs(residual_hat) ** 2, axis=0)
            entry["spectral_target"] += np.sum(np.abs(target_hat) ** 2, axis=0)
            if regime == "nonlinear_landau_strong":
                fields = np.asarray(primitive_fields(jnp.asarray(current), jnp.asarray(k_arr)))
                energy = 0.5 * np.mean(fields[:, 3] ** 2, axis=1)
                # Compare residual magnitude with the teacher energy-envelope tendency.
                log_energy = np.log(np.maximum(energy, 1e-30))
                if log_energy.size > 1:
                    strong_growth.extend(np.gradient(log_energy).tolist())
                    strong_residual.extend(np.sqrt(residual_energy).tolist())
        thirds = np.array_split(np.asarray(block_values), 3)
        time_blocks[regime].append([float(np.mean(value)) for value in thirds])
    result = {
        "checkpoint": str(checkpoint),
        "closure_history_mode": closure_history_mode,
        "split": split,
        "regimes": {},
    }
    for regime, entry in totals.items():
        spectral_error = entry["spectral_error"]
        spectral_target = entry["spectral_target"]
        bands = {"k1_4": (1, 5), "k5_16": (5, 17), "k17_64": (17, 65), "k65_plus": (65, spectral_error.size)}
        result["regimes"][regime] = {
            "relative_heat_flux_gradient_mse": entry["error"] / max(entry["target"], 1e-30),
            "least_squares_gain": entry["cross"] / max(entry["target"], 1e-30),
            "prediction_target_cosine": entry["cross"] / max(
                math.sqrt(entry["prediction"] * entry["target"]), 1e-30
            ),
            "residual_spatial_mean_fraction": entry["mean"] / max(entry["residual"], 1e-30),
            "relative_mse_time_thirds": np.mean(time_blocks[regime], axis=0).tolist(),
            "relative_mse_spectral_bands": {
                name: float(np.sum(spectral_error[left:right]) / max(np.sum(spectral_target[left:right]), 1e-30))
                for name, (left, right) in bands.items()
            },
        }
    if len(strong_growth) > 2:
        result["strong_residual_growth_correlation"] = float(
            np.corrcoef(strong_growth, strong_residual)[0, 1]
        )
    return result


def _moment_equation_residual_score(
    cache_dir: Path,
    cases: dict[str, dict],
    *,
    split: str,
    dt: float,
    source_nx: int,
    rollout_nx: int,
    domain_length: float,
    sample_stride: int,
) -> dict[str, dict[str, float]]:
    """Measure whether the projected teacher requires a uniform effective source."""
    k_arr = 2.0 * math.pi * np.fft.rfftfreq(
        rollout_nx, d=domain_length / rollout_nx
    )
    totals = {
        regime: {
            "residual": 0.0,
            "derivative": 0.0,
            "uniform_residual": 0.0,
            "uniform_derivative": 0.0,
            "gradient_mean": 0.0,
            "count": 0,
        }
        for regime in REGIMES
    }
    for case_id, case in cases.items():
        if str(case["split"]) != split:
            continue
        regime = str(case["regime"])
        history = np.load(cache_dir / "cases" / f"{case_id}.npy", mmap_mode="r")
        times = np.arange(1, history.shape[0] - 1, sample_stride, dtype=np.int32)
        needed = np.unique(np.concatenate((times - 1, times, times + 1)))
        states, gradients = _state_and_gradient(
            history,
            needed,
            source_nx=source_nx,
            rollout_nx=rollout_nx,
            domain_length=domain_length,
        )
        previous = states[np.searchsorted(needed, times - 1)]
        current = states[np.searchsorted(needed, times)]
        following = states[np.searchsorted(needed, times + 1)]
        gradient = gradients[np.searchsorted(needed, times)]
        derivative = (following[:, 2] - previous[:, 2]) / (2.0 * dt)
        rhs = np.asarray(
            low_moment_rhs(
                jnp.asarray(current, dtype=jnp.float64),
                jnp.asarray(gradient, dtype=jnp.float64),
                jnp.asarray(k_arr, dtype=jnp.float64),
            )
        )[:, 2]
        residual = derivative - rhs
        entry = totals[regime]
        entry["residual"] += float(np.sum(residual * residual))
        entry["derivative"] += float(np.sum(derivative * derivative))
        entry["uniform_residual"] += float(
            rollout_nx * np.sum(np.square(np.mean(residual, axis=-1)))
        )
        entry["uniform_derivative"] += float(
            rollout_nx * np.sum(np.square(np.mean(derivative, axis=-1)))
        )
        entry["gradient_mean"] += float(np.sum(np.square(np.mean(gradient, axis=-1))))
        entry["count"] += int(times.size)
    result = {}
    for regime, entry in totals.items():
        result[regime] = {
            "relative_second_moment_rhs_residual": entry["residual"]
            / max(entry["derivative"], 1e-30),
            "uniform_fraction_of_rhs_residual": entry["uniform_residual"]
            / max(entry["residual"], 1e-30),
            "relative_uniform_rhs_residual": entry["uniform_residual"]
            / max(entry["uniform_derivative"], 1e-30),
            "exact_gradient_spatial_mean_rms": math.sqrt(
                entry["gradient_mean"] / max(entry["count"], 1)
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--history-spans", default="0,5,10,20,40")
    parser.add_argument("--history-lags", type=int, default=11)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--skip-history-identifiability", action="store_true")
    args = parser.parse_args()
    print_jax_runtime_summary(jax, context="low-moment nonlinear residual")
    cache_dir = args.cache.resolve()
    cases, cache_metadata = _case_map(cache_dir)
    config = cache_metadata["configuration"]
    source_nx = int(config["teacher_Nx"])
    rollout_nx = 256
    dt = float(config["teacher_dt"])
    domain_length = float(config["teacher_L"])
    report = {
        "cache": str(cache_dir),
        "teacher_forced": [],
        "history_identifiability": {},
        "projected_moment_equation_residual": {},
    }
    for split in ("train", "heldout"):
        report["projected_moment_equation_residual"][split] = (
            _moment_equation_residual_score(
                cache_dir,
                cases,
                split=split,
                dt=dt,
                source_nx=source_nx,
                rollout_nx=rollout_nx,
                domain_length=domain_length,
                sample_stride=args.sample_stride,
            )
        )
    for checkpoint in args.checkpoint:
        print(f"[diagnostic] scoring {checkpoint}")
        _, checkpoint_metadata, _ = _load_checkpoint(checkpoint.resolve())
        splits = ["train", "heldout"]
        if checkpoint_metadata.get("selected_training_case_ids"):
            splits.insert(0, "train_selected")
        for split in splits:
            for closure_history_mode in ("exact", "zero"):
                report["teacher_forced"].append(
                    _teacher_forced_checkpoint_score(
                        checkpoint.resolve(), cache_dir, cases, dt=dt,
                        source_nx=source_nx, rollout_nx=rollout_nx,
                        domain_length=domain_length,
                        sample_stride=args.sample_stride,
                        closure_history_mode=closure_history_mode,
                        split=split,
                    )
                )
    spans = [float(value) for value in args.history_spans.split(",")]
    if not args.skip_history_identifiability:
        common_start = max(spans)
        for include_gradient in (False, True):
            label = "state_plus_exact_closure_history" if include_gradient else "state_history_only"
            report["history_identifiability"][label] = {}
            for span in spans:
                print(f"[diagnostic] ambiguity {label} span={span:g}")
                train_x, train_y = _history_dataset(
                    cache_dir, cases, split="train", regime="nonlinear_landau_strong",
                    span=span, common_start=common_start,
                    include_gradient=include_gradient, dt=dt,
                    source_nx=source_nx, rollout_nx=rollout_nx,
                    domain_length=domain_length, sample_stride=args.sample_stride,
                    lag_count=args.history_lags, modes=args.modes,
                )
                test_x, test_y = _history_dataset(
                    cache_dir, cases, split="heldout", regime="nonlinear_landau_strong",
                    span=span, common_start=common_start,
                    include_gradient=include_gradient, dt=dt,
                    source_nx=source_nx, rollout_nx=rollout_nx,
                    domain_length=domain_length, sample_stride=args.sample_stride,
                    lag_count=args.history_lags, modes=args.modes,
                )
                report["history_identifiability"][label][str(span)] = _knn_score(
                    train_x, train_y, test_x, test_y, neighbors=args.neighbors,
                    projection_dim=48, seed=1729,
                )
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    if report["history_identifiability"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for label, values in report["history_identifiability"].items():
            y = [values[str(span)]["normalized_knn_mse"] for span in spans]
            ax.plot(spans, y, marker="o", label=label.replace("_", " "))
        ax.set_xlabel("Observed history span")
        ax.set_ylabel("Held-out strong closure kNN NMSE")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(args.outdir / "history_identifiability.png", dpi=180)
        plt.close(fig)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
