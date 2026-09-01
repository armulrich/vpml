"""Test whether persistent mode-resolved bounce exposure identifies closure targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model.diagnostics.low_moment_nonlinear_residual import (
    _case_map,
    _history_dataset,
    _knn_score,
)
from vpml.linear_landau import landau_omega, solve_landau_root_xi


def _cumulative_trapezoid(values: np.ndarray, dt: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    if values.shape[0] > 1:
        result[1:] = np.cumsum(
            0.5 * float(dt) * (values[1:] + values[:-1]), axis=0
        )
    return result


def mode_bounce_exposure(
    density_hat_history: np.ndarray,
    *,
    source_nx: int,
    domain_length: float,
    dt: float,
    modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return instantaneous bounce frequency and its cumulative phase per mode."""
    available_modes = min(int(modes), density_hat_history.shape[-1] - 1)
    mode_numbers = np.arange(1, available_modes + 1, dtype=np.float64)
    wave_numbers = (2.0 * math.pi / float(domain_length)) * mode_numbers
    density_hat = np.asarray(
        density_hat_history[:, 1 : available_modes + 1], dtype=np.complex128
    )
    field_hat = 1j * density_hat / wave_numbers[None, :]
    # For a real field, 2 |E_hat| / Nx is the cosine-mode amplitude.
    field_amplitude = 2.0 * np.abs(field_hat) / float(source_nx)
    bounce_frequency = np.sqrt(wave_numbers[None, :] * field_amplitude)
    return bounce_frequency, _cumulative_trapezoid(bounce_frequency, dt)


def _resonance_rows(
    cache_dir: Path,
    cases: dict[str, dict],
    *,
    split: str,
    regime: str,
    start_step: int,
    sample_stride: int,
    source_nx: int,
    domain_length: float,
    dt: float,
    modes: int,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {
        "time": [],
        "global_exposure": [],
        "mode_exposure": [],
        "mode_frequency": [],
    }
    for case_id, case in cases.items():
        if str(case["split"]) != split or str(case["regime"]) != regime:
            continue
        history = np.load(cache_dir / "cases" / f"{case_id}.npy", mmap_mode="r")
        times = np.arange(start_step, history.shape[0], sample_stride, dtype=np.int32)
        frequency, exposure = mode_bounce_exposure(
            history[:, 0, : modes + 1],
            source_nx=source_nx,
            domain_length=domain_length,
            dt=dt,
            modes=modes,
        )
        selected_exposure = exposure[times]
        rows["time"].append((times.astype(np.float64) * dt)[:, None])
        rows["global_exposure"].append(
            np.sum(selected_exposure, axis=1, keepdims=True)
        )
        rows["mode_exposure"].append(selected_exposure)
        rows["mode_frequency"].append(frequency[times])
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


def _phase_velocities(domain_length: float, modes: int) -> np.ndarray:
    wave_numbers = (2.0 * math.pi / float(domain_length)) * np.arange(1, modes + 1)
    return np.asarray(
        [
            np.real(landau_omega(float(k), solve_landau_root_xi(float(k)))) / k
            for k in wave_numbers
        ],
        dtype=np.float64,
    )


def _integrate_velocity_bands(
    f_hat: np.ndarray,
    velocity: np.ndarray,
    centers: np.ndarray,
    half_width: float,
) -> np.ndarray:
    result = np.empty((centers.size,), dtype=np.complex128)
    for mode, center in enumerate(centers):
        selected = np.abs(velocity - center) <= float(half_width)
        result[mode] = np.trapezoid(f_hat[selected, mode], velocity[selected])
    return result


def _sample_velocity_profiles(
    f_hat: np.ndarray,
    velocity: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    profiles = np.empty((centers.size, offsets.size), dtype=np.complex128)
    for mode, center in enumerate(centers):
        points = center + offsets
        profiles[mode] = np.interp(
            points, velocity, f_hat[:, mode].real
        ) + 1j * np.interp(points, velocity, f_hat[:, mode].imag)
    return profiles


def _direct_population_rows(
    cache_dir: Path,
    cases: dict[str, dict],
    *,
    split: str,
    regime: str,
    domain_length: float,
    dt: float,
    modes: int,
    half_widths: tuple[float, ...],
) -> dict[str, np.ndarray]:
    phase_velocities = _phase_velocities(domain_length, modes)
    rows: dict[str, list[np.ndarray]] = {}
    for width in half_widths:
        rows[f"resonant_width_{width:g}"] = []
        rows[f"zero_velocity_width_{width:g}"] = []
    profile_offsets = np.asarray((-1.0, -0.5, 0.0, 0.5, 1.0), dtype=np.float64)
    rows["resonant_profile"] = []
    rows["zero_velocity_profile"] = []
    mode_numbers = np.arange(1, modes + 1, dtype=np.float64)
    for case_id, case in cases.items():
        if str(case["split"]) != split or str(case["regime"]) != regime:
            continue
        history = np.load(cache_dir / "cases" / f"{case_id}.npy", mmap_mode="r")
        with np.load(cache_dir / "snapshots" / f"{case_id}.npz") as snapshot:
            snapshot_times = np.asarray(snapshot["snapshot_times"], dtype=np.float64)
            velocity = np.asarray(snapshot["v"], dtype=np.float64)
            x = np.asarray(snapshot["x"], dtype=np.float64)
            f_values = np.asarray(snapshot["snapshot_f"], dtype=np.float64)
        time_indices = np.rint(snapshot_times / dt).astype(np.int32)
        density_phase = np.angle(np.asarray(history[time_indices, 0, 1]))
        rotation = np.exp(-1j * density_phase[:, None] * mode_numbers[None, :])
        spatial_basis = np.exp(
            -1j
            * x[:, None]
            * ((2.0 * math.pi / domain_length) * mode_numbers)[None, :]
        )
        values_by_width = {name: [] for name in rows}
        for snapshot_index, f_value in enumerate(f_values):
            f_hat = (f_value @ spatial_basis) / x.size
            for width in half_widths:
                resonant = _integrate_velocity_bands(
                    f_hat, velocity, phase_velocities, width
                )
                nonresonant = _integrate_velocity_bands(
                    f_hat, velocity, np.zeros_like(phase_velocities), width
                )
                values_by_width[f"resonant_width_{width:g}"].append(
                    resonant * rotation[snapshot_index]
                )
                values_by_width[f"zero_velocity_width_{width:g}"].append(
                    nonresonant * rotation[snapshot_index]
                )
            resonant_profile = _sample_velocity_profiles(
                f_hat, velocity, phase_velocities, profile_offsets
            )
            zero_profile = _sample_velocity_profiles(
                f_hat, velocity, np.zeros_like(phase_velocities), profile_offsets
            )
            values_by_width["resonant_profile"].append(
                resonant_profile * rotation[snapshot_index, :, None]
            )
            values_by_width["zero_velocity_profile"].append(
                zero_profile * rotation[snapshot_index, :, None]
            )
        for name, values in values_by_width.items():
            complex_values = np.asarray(values).reshape(len(values), -1)
            rows[name].append(
                np.concatenate((complex_values.real, complex_values.imag), axis=1)
            )
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


def _score_feature_sets(
    train_state: np.ndarray,
    train_target: np.ndarray,
    heldout_state: np.ndarray,
    heldout_target: np.ndarray,
    train_resonance: dict[str, np.ndarray],
    heldout_resonance: dict[str, np.ndarray],
    *,
    neighbors: int,
    projection_dim: int,
    seeds: tuple[int, ...],
) -> dict[str, dict]:
    feature_sets = {
        "state_history": (train_state, heldout_state),
        "state_history_plus_time": (
            np.concatenate((train_state, train_resonance["time"]), axis=1),
            np.concatenate((heldout_state, heldout_resonance["time"]), axis=1),
        ),
        "state_history_plus_global_bounce_exposure": (
            np.concatenate(
                (train_state, train_resonance["global_exposure"]), axis=1
            ),
            np.concatenate(
                (heldout_state, heldout_resonance["global_exposure"]), axis=1
            ),
        ),
        "state_history_plus_mode_bounce_exposure": (
            np.concatenate((train_state, train_resonance["mode_exposure"]), axis=1),
            np.concatenate(
                (heldout_state, heldout_resonance["mode_exposure"]), axis=1
            ),
        ),
        "state_history_plus_mode_bounce_state": (
            np.concatenate(
                (
                    train_state,
                    train_resonance["mode_exposure"],
                    train_resonance["mode_frequency"],
                ),
                axis=1,
            ),
            np.concatenate(
                (
                    heldout_state,
                    heldout_resonance["mode_exposure"],
                    heldout_resonance["mode_frequency"],
                ),
                axis=1,
            ),
        ),
    }
    result: dict[str, dict] = {}
    for label, (train_x, heldout_x) in feature_sets.items():
        scores = [
            _knn_score(
                train_x,
                train_target,
                heldout_x,
                heldout_target,
                neighbors=neighbors,
                projection_dim=projection_dim,
                seed=seed,
            )
            for seed in seeds
        ]
        values = np.asarray([score["normalized_knn_mse"] for score in scores])
        result[label] = {
            "normalized_knn_mse_mean": float(np.mean(values)),
            "normalized_knn_mse_std": float(np.std(values)),
            "normalized_knn_mse_by_seed": values.tolist(),
            "mse_mean": float(np.mean([score["mse"] for score in scores])),
            "baseline_mse": float(scores[0]["baseline_mse"]),
        }
    baseline = result["state_history"]["normalized_knn_mse_mean"]
    for value in result.values():
        value["improvement_over_state_history"] = float(
            (baseline - value["normalized_knn_mse_mean"]) / max(baseline, 1e-30)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--history-span", type=float, default=5.0)
    parser.add_argument("--common-start", type=float, default=20.0)
    parser.add_argument("--history-lags", type=int, default=11)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=48)
    parser.add_argument("--seeds", default="1729,1730,1731,1732,1733")
    parser.add_argument("--direct-population-modes", type=int, default=4)
    parser.add_argument("--direct-population-widths", default="0.25,0.5,1.0")
    parser.add_argument("--skip-direct-population", action="store_true")
    args = parser.parse_args()

    cache_dir = args.cache.resolve()
    cases, metadata = _case_map(cache_dir)
    config = metadata["configuration"]
    dt = float(config["teacher_dt"])
    source_nx = int(config["teacher_Nx"])
    rollout_nx = 256
    domain_length = float(config["teacher_L"])
    start_step = max(
        int(round(args.common_start / dt)),
        int(round(args.history_span / dt)),
        1,
    )

    datasets = {}
    resonance = {}
    for split in ("train", "heldout"):
        datasets[split] = _history_dataset(
            cache_dir,
            cases,
            split=split,
            regime="nonlinear_landau_strong",
            span=args.history_span,
            common_start=args.common_start,
            include_gradient=False,
            dt=dt,
            source_nx=source_nx,
            rollout_nx=rollout_nx,
            domain_length=domain_length,
            sample_stride=args.sample_stride,
            lag_count=args.history_lags,
            modes=args.modes,
        )
        resonance[split] = _resonance_rows(
            cache_dir,
            cases,
            split=split,
            regime="nonlinear_landau_strong",
            start_step=start_step,
            sample_stride=args.sample_stride,
            source_nx=source_nx,
            domain_length=domain_length,
            dt=dt,
            modes=args.modes,
        )
        if datasets[split][0].shape[0] != resonance[split]["time"].shape[0]:
            raise RuntimeError(f"Feature row mismatch for {split}")

    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    scores = _score_feature_sets(
        datasets["train"][0],
        datasets["train"][1],
        datasets["heldout"][0],
        datasets["heldout"][1],
        resonance["train"],
        resonance["heldout"],
        neighbors=args.neighbors,
        projection_dim=args.projection_dim,
        seeds=seeds,
    )
    report = {
        "cache": str(cache_dir),
        "configuration": {
            "history_span": args.history_span,
            "common_start": args.common_start,
            "sample_stride": args.sample_stride,
            "modes": args.modes,
            "neighbors": args.neighbors,
            "projection_dim": args.projection_dim,
            "seeds": list(seeds),
            "train_samples": int(datasets["train"][0].shape[0]),
            "heldout_samples": int(datasets["heldout"][0].shape[0]),
        },
        "scores": scores,
    }
    if not args.skip_direct_population:
        direct_modes = int(args.direct_population_modes)
        direct_widths = tuple(
            float(value) for value in args.direct_population_widths.split(",") if value
        )
        direct_datasets = {}
        direct_resonance = {}
        direct_sample_stride = int(round(20.0 / dt))
        for split in ("train", "heldout"):
            direct_datasets[split] = _history_dataset(
                cache_dir,
                cases,
                split=split,
                regime="nonlinear_landau_strong",
                span=args.history_span,
                common_start=20.0,
                include_gradient=False,
                dt=dt,
                source_nx=source_nx,
                rollout_nx=rollout_nx,
                domain_length=domain_length,
                sample_stride=direct_sample_stride,
                lag_count=args.history_lags,
                modes=args.modes,
            )
            direct_resonance[split] = _direct_population_rows(
                cache_dir,
                cases,
                split=split,
                regime="nonlinear_landau_strong",
                domain_length=domain_length,
                dt=dt,
                modes=direct_modes,
                half_widths=direct_widths,
            )
        direct_scores = {}
        base_train, train_target = direct_datasets["train"]
        base_heldout, heldout_target = direct_datasets["heldout"]
        base_score = _score_feature_sets(
            base_train,
            train_target,
            base_heldout,
            heldout_target,
            {
                "time": np.zeros((base_train.shape[0], 1)),
                "global_exposure": np.zeros((base_train.shape[0], 1)),
                "mode_exposure": np.zeros((base_train.shape[0], 1)),
                "mode_frequency": np.zeros((base_train.shape[0], 1)),
            },
            {
                "time": np.zeros((base_heldout.shape[0], 1)),
                "global_exposure": np.zeros((base_heldout.shape[0], 1)),
                "mode_exposure": np.zeros((base_heldout.shape[0], 1)),
                "mode_frequency": np.zeros((base_heldout.shape[0], 1)),
            },
            neighbors=min(args.neighbors, 4),
            projection_dim=args.projection_dim,
            seeds=seeds,
        )["state_history"]
        direct_scores["state_history"] = base_score
        baseline = base_score["normalized_knn_mse_mean"]
        for name in direct_resonance["train"]:
            train_x = np.concatenate(
                (base_train, direct_resonance["train"][name]), axis=1
            )
            heldout_x = np.concatenate(
                (base_heldout, direct_resonance["heldout"][name]), axis=1
            )
            seed_scores = [
                _knn_score(
                    train_x,
                    train_target,
                    heldout_x,
                    heldout_target,
                    neighbors=min(args.neighbors, 4),
                    projection_dim=args.projection_dim,
                    seed=seed,
                )
                for seed in seeds
            ]
            values = np.asarray(
                [score["normalized_knn_mse"] for score in seed_scores]
            )
            direct_scores[f"state_history_plus_{name}"] = {
                "normalized_knn_mse_mean": float(np.mean(values)),
                "normalized_knn_mse_std": float(np.std(values)),
                "normalized_knn_mse_by_seed": values.tolist(),
                "mse_mean": float(np.mean([score["mse"] for score in seed_scores])),
                "baseline_mse": float(seed_scores[0]["baseline_mse"]),
                "improvement_over_state_history": float(
                    (baseline - np.mean(values)) / max(baseline, 1e-30)
                ),
            }
        report["direct_resonant_population"] = {
            "phase_velocities": _phase_velocities(
                domain_length, direct_modes
            ).tolist(),
            "train_samples": int(base_train.shape[0]),
            "heldout_samples": int(base_heldout.shape[0]),
            "scores": direct_scores,
        }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    labels = list(scores)
    means = [scores[label]["normalized_knn_mse_mean"] for label in labels]
    errors = [scores[label]["normalized_knn_mse_std"] for label in labels]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.bar(np.arange(len(labels)), means, yerr=errors, capsize=4)
    ax.set_xticks(np.arange(len(labels)), [label.replace("_", "\n") for label in labels])
    ax.set_ylabel("Held-out strong closure kNN NMSE")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(args.outdir / "resonance_identifiability.png", dpi=180)
    plt.close(fig)
    if "direct_resonant_population" in report:
        direct_scores = report["direct_resonant_population"]["scores"]
        direct_labels = list(direct_scores)
        direct_means = [
            direct_scores[label]["normalized_knn_mse_mean"]
            for label in direct_labels
        ]
        direct_errors = [
            direct_scores[label]["normalized_knn_mse_std"]
            for label in direct_labels
        ]
        fig, ax = plt.subplots(figsize=(10.8, 5.2), constrained_layout=True)
        ax.bar(
            np.arange(len(direct_labels)),
            direct_means,
            yerr=direct_errors,
            capsize=4,
        )
        ax.set_xticks(
            np.arange(len(direct_labels)),
            [
                label.replace("state_history_plus_", "").replace("_", "\n")
                for label in direct_labels
            ],
        )
        ax.set_ylabel("Sparse-snapshot held-out closure kNN NMSE")
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(args.outdir / "direct_resonant_identifiability.png", dpi=180)
        plt.close(fig)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
