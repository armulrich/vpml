"""Test whether train-only kinetic POD coordinates identify future heat flux."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from model.train.interface_flux_data import load_ic_manifest
from model.train.low_moment_closure import (
    _central_heat_flux_numpy,
    low_hermite_coefficients_to_conservative,
)


REGIMES = (
    "linear_landau",
    "nonlinear_landau_weak",
    "nonlinear_landau_strong",
)


def _complex_to_real(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.concatenate((values.real, values.imag), axis=-1).reshape(
        values.shape[0], -1
    ).astype(np.float32)


def _history_summary(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            values[:, -1],
            np.mean(values, axis=1),
            np.std(values.real, axis=1) + 1j * np.std(values.imag, axis=1),
            values[:, -1] - values[:, 0],
        ),
        axis=1,
    )


def _selected_cases(
    cases: Iterable[dict],
    *,
    regime: str,
    split: str,
) -> list[dict]:
    return [
        case
        for case in cases
        if str(case["regime"]) == regime and str(case["split"]) == split
    ]


def _train_velocity_pod(
    cache_dir: Path,
    cases: list[dict],
    *,
    first_step: int,
    last_step: int,
    sample_stride: int,
    basis_modes: int,
    first_unresolved_order: int,
    cached_orders: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    covariance_by_regime: Dict[str, np.ndarray] = {}
    sample_steps = np.arange(first_step, last_step + 1, sample_stride, dtype=np.int32)
    unresolved_orders = int(cached_orders) - int(first_unresolved_order)
    for regime in REGIMES:
        covariance = np.zeros(
            (unresolved_orders, unresolved_orders), dtype=np.float64
        )
        for case in _selected_cases(cases, regime=regime, split="train"):
            history = np.load(
                cache_dir / "cases" / f"{case['case_id']}.npy",
                mmap_mode="r",
            )
            values = np.asarray(
                history[
                    sample_steps,
                    first_unresolved_order:cached_orders,
                    1 : basis_modes + 1,
                ],
                dtype=np.complex128,
            )
            covariance += 2.0 * np.real(
                np.einsum("tnk,tmk->nm", values, values.conj(), optimize=True)
            )
        trace = float(np.trace(covariance))
        if not trace > 0.0:
            raise ValueError(f"Degenerate train covariance for {regime}")
        covariance_by_regime[regime] = covariance / trace

    covariance = sum(covariance_by_regime.values()) / float(len(REGIMES))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = eigenvectors[:, order]
    return basis, eigenvalues, covariance_by_regime


def _case_rows(
    history: np.ndarray,
    basis: np.ndarray,
    ranks: list[int],
    *,
    source_nx: int,
    target_nx: int,
    domain_length: float,
    history_steps: int,
    history_stride: int,
    sample_stride: int,
    first_step: int,
    last_step: int,
    future_offsets: np.ndarray,
    feature_modes: int,
    first_unresolved_order: int,
    latent_representation: str,
) -> tuple[Dict[int, np.ndarray], np.ndarray]:
    times = np.arange(first_step, last_step + 1, sample_stride, dtype=np.int32)
    lag_offsets = (
        np.arange(history_steps - 1, -1, -1, dtype=np.int32) * history_stride
    )
    history_indices = times[:, None] - lag_offsets[None]
    coefficients = np.asarray(
        history[
            history_indices,
            : first_unresolved_order + basis.shape[0],
            1 : feature_modes + 1,
        ],
        dtype=np.complex128,
    )

    phase = np.angle(coefficients[:, -1, 0, 0])
    mode_numbers = np.arange(1, feature_modes + 1, dtype=np.float64)
    rotation = np.exp(-1j * phase[:, None] * mode_numbers[None])
    coefficients *= rotation[:, None, None, :]
    density = coefficients[:, -1, 0]
    amplitude = np.sqrt(2.0 * np.sum(np.square(np.abs(density)), axis=1))
    amplitude = np.maximum(amplitude / float(source_nx), 1e-14)
    coefficients /= (amplitude * float(source_nx))[:, None, None, None]

    state = coefficients[:, :, :first_unresolved_order]
    state_features = _complex_to_real(_history_summary(state))
    log_amplitude = np.log(amplitude)[:, None].astype(np.float32)
    base_features = np.concatenate((state_features, log_amplitude), axis=1)

    unresolved = coefficients[:, :, first_unresolved_order:]
    features: Dict[int, np.ndarray] = {0: base_features}
    for rank in ranks:
        latent = np.einsum(
            "nr,shnk->shrk",
            basis[:, :rank],
            unresolved,
            optimize=True,
        )
        if latent_representation == "current":
            latent_features = _complex_to_real(latent[:, -1])
        elif latent_representation == "history_summary":
            latent_features = _complex_to_real(_history_summary(latent))
        else:
            raise ValueError(
                f"Unknown latent representation: {latent_representation}"
            )
        features[rank] = np.concatenate(
            (base_features, latent_features),
            axis=1,
        )

    target_indices = times[:, None] + future_offsets[None]
    target_coefficients = np.asarray(
        history[target_indices, :4], dtype=np.complex128
    ).reshape(-1, 4, history.shape[-1])
    target_state = low_hermite_coefficients_to_conservative(
        target_coefficients[:, :3],
        source_nx=source_nx,
        target_nx=target_nx,
        dtype=np.float64,
    )
    heat_flux = _central_heat_flux_numpy(
        target_coefficients,
        target_state,
        source_nx=source_nx,
        target_nx=target_nx,
    )
    heat_flux_hat = np.fft.rfft(heat_flux, axis=-1)[
        :, 1 : feature_modes + 1
    ]
    wave_numbers = (
        2.0 * math.pi / float(domain_length)
    ) * mode_numbers
    gradient = (1j * wave_numbers[None] * heat_flux_hat).reshape(
        times.size, future_offsets.size, feature_modes
    )
    gradient *= rotation[:, None]
    gradient /= (amplitude * float(target_nx))[:, None, None]
    return features, _complex_to_real(gradient)


def _collect_split(
    cache_dir: Path,
    cases: list[dict],
    basis: np.ndarray,
    ranks: list[int],
    *,
    regime: str,
    split: str,
    source_nx: int,
    target_nx: int,
    domain_length: float,
    history_steps: int,
    history_stride: int,
    sample_stride: int,
    first_step: int,
    last_step: int,
    future_offsets: np.ndarray,
    feature_modes: int,
    first_unresolved_order: int,
    latent_representation: str,
) -> tuple[Dict[int, np.ndarray], np.ndarray]:
    feature_rows: Dict[int, list[np.ndarray]] = {0: []}
    feature_rows.update({rank: [] for rank in ranks})
    target_rows = []
    for case in _selected_cases(cases, regime=regime, split=split):
        history = np.load(
            cache_dir / "cases" / f"{case['case_id']}.npy",
            mmap_mode="r",
        )
        features, target = _case_rows(
            history,
            basis,
            ranks,
            source_nx=source_nx,
            target_nx=target_nx,
            domain_length=domain_length,
            history_steps=history_steps,
            history_stride=history_stride,
            sample_stride=sample_stride,
            first_step=first_step,
            last_step=last_step,
            future_offsets=future_offsets,
            feature_modes=feature_modes,
            first_unresolved_order=first_unresolved_order,
            latent_representation=latent_representation,
        )
        for rank, values in features.items():
            feature_rows[rank].append(values)
        target_rows.append(target)
    return (
        {rank: np.concatenate(rows) for rank, rows in feature_rows.items()},
        np.concatenate(target_rows),
    )


def _current_flux_reconstruction_scores(
    cache_dir: Path,
    cases: list[dict],
    basis: np.ndarray,
    ranks: list[int],
    *,
    regime: str,
    split: str,
    source_nx: int,
    target_nx: int,
    domain_length: float,
    first_step: int,
    last_step: int,
    sample_stride: int,
    first_unresolved_order: int,
) -> Dict[str, float]:
    numerator = {rank: 0.0 for rank in ranks}
    denominator = 0.0
    sample_steps = np.arange(
        first_step,
        last_step + 1,
        sample_stride,
        dtype=np.int32,
    )
    wave_numbers = (2.0 * math.pi / float(domain_length)) * np.arange(
        target_nx // 2 + 1,
        dtype=np.float64,
    )
    for case in _selected_cases(cases, regime=regime, split=split):
        history = np.load(
            cache_dir / "cases" / f"{case['case_id']}.npy",
            mmap_mode="r",
        )
        coefficients = np.asarray(
            history[
                sample_steps,
                : first_unresolved_order + basis.shape[0],
            ],
            dtype=np.complex128,
        )
        state = low_hermite_coefficients_to_conservative(
            coefficients[:, :first_unresolved_order],
            source_nx=source_nx,
            target_nx=target_nx,
            dtype=np.float64,
        )
        exact_flux = _central_heat_flux_numpy(
            coefficients[:, : first_unresolved_order + 1],
            state,
            source_nx=source_nx,
            target_nx=target_nx,
        )
        exact_gradient = np.fft.irfft(
            1j * wave_numbers[None] * np.fft.rfft(exact_flux, axis=-1),
            n=target_nx,
            axis=-1,
        )
        denominator += float(np.sum(np.square(exact_gradient)))
        unresolved = coefficients[:, first_unresolved_order:]
        for rank in ranks:
            latent = np.einsum(
                "nr,snk->srk",
                basis[:, :rank],
                unresolved,
                optimize=True,
            )
            reconstructed = np.einsum(
                "nr,srk->snk",
                basis[:, :rank],
                latent,
                optimize=True,
            )
            approximate_coefficients = coefficients[
                :, : first_unresolved_order + 1
            ].copy()
            approximate_coefficients[:, first_unresolved_order] = (
                reconstructed[:, 0]
            )
            approximate_flux = _central_heat_flux_numpy(
                approximate_coefficients,
                state,
                source_nx=source_nx,
                target_nx=target_nx,
            )
            approximate_gradient = np.fft.irfft(
                1j
                * wave_numbers[None]
                * np.fft.rfft(approximate_flux, axis=-1),
                n=target_nx,
                axis=-1,
            )
            numerator[rank] += float(
                np.sum(np.square(approximate_gradient - exact_gradient))
            )
    return {
        str(rank): numerator[rank] / max(denominator, 1e-30)
        for rank in ranks
    }


def _knn_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    heldout_x: np.ndarray,
    heldout_y: np.ndarray,
    *,
    base_feature_count: int,
    neighbors: int,
    projection_dim: int,
    seed: int,
) -> dict[str, float]:
    center = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    active = scale > 1e-8
    train = (train_x[:, active] - center[active]) / scale[active]
    heldout = (heldout_x[:, active] - center[active]) / scale[active]

    # Standardizing coordinates alone gives a larger feature block proportionally
    # more influence on Euclidean distance. Balance the resolved-history and
    # kinetic-latent blocks by their active dimensions before comparing them.
    active_indices = np.flatnonzero(active)
    base_active = active_indices < int(base_feature_count)
    latent_active = ~base_active
    if np.any(base_active):
        factor = math.sqrt(float(np.sum(base_active)))
        train[:, base_active] /= factor
        heldout[:, base_active] /= factor
    if np.any(latent_active):
        factor = math.sqrt(float(np.sum(latent_active)))
        train[:, latent_active] /= factor
        heldout[:, latent_active] /= factor
    projected_dim = min(int(projection_dim), train.shape[1])
    if projected_dim < train.shape[1]:
        rng = np.random.default_rng(seed)
        projection = rng.normal(
            size=(train.shape[1], projected_dim)
        ).astype(np.float32) / math.sqrt(projected_dim)
        train = train @ projection
        heldout = heldout @ projection
    tree = cKDTree(train)
    _, nearest = tree.query(heldout, k=min(neighbors, train.shape[0]))
    if nearest.ndim == 1:
        nearest = nearest[:, None]
    prediction = np.mean(train_y[nearest], axis=1)
    center_y = np.mean(train_y, axis=0)
    error = float(np.mean(np.sum(np.square(prediction - heldout_y), axis=1)))
    baseline = float(np.mean(np.sum(np.square(heldout_y - center_y), axis=1)))
    target_energy = float(np.mean(np.sum(np.square(heldout_y), axis=1)))
    return {
        "normalized_mse": error / max(baseline, 1e-30),
        "relative_target_mse": error / max(target_energy, 1e-30),
        "active_features": int(np.sum(active)),
        "active_base_features": int(np.sum(base_active)),
        "active_latent_features": int(np.sum(latent_active)),
        "projected_features": int(projected_dim),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--ranks", default="16,32,48,62")
    parser.add_argument("--basis-modes", type=int, default=64)
    parser.add_argument("--feature-modes", type=int, default=8)
    parser.add_argument("--history-span", type=float, default=5.0)
    parser.add_argument("--history-cadence", type=float, default=0.5)
    parser.add_argument("--sample-cadence", type=float, default=1.0)
    parser.add_argument("--start-time", type=float, default=20.0)
    parser.add_argument("--end-time", type=float, default=100.0)
    parser.add_argument("--future-times", default="0.1,1,5")
    parser.add_argument("--target-nx", type=int, default=256)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument(
        "--latent-representation",
        choices=("current", "history_summary"),
        default="current",
    )
    parser.add_argument("--seeds", default="1729,1730,1731,1732,1733")
    args = parser.parse_args()

    cache_dir = args.reference_cache.resolve()
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    configuration = metadata["configuration"]
    manifest = load_ic_manifest(cache_dir / "ic_manifest.json")
    cases = [dict(case) for case in manifest["cases"]]
    dt = float(configuration["teacher_dt"])
    source_nx = int(configuration["teacher_Nx"])
    cached_orders = int(configuration["max_projection_order"])
    first_unresolved_order = 3
    maximum_rank = cached_orders - first_unresolved_order
    ranks = sorted({int(value) for value in args.ranks.split(",") if value})
    if not ranks or min(ranks) <= 0 or max(ranks) > maximum_rank:
        raise ValueError(f"ranks must lie in [1, {maximum_rank}]")
    if args.feature_modes > args.basis_modes:
        raise ValueError("feature-modes cannot exceed basis-modes")
    history_stride = int(round(args.history_cadence / dt))
    history_steps = int(round(args.history_span / args.history_cadence)) + 1
    sample_stride = int(round(args.sample_cadence / dt))
    future_times = [
        float(value) for value in args.future_times.split(",") if value.strip()
    ]
    future_offsets = np.asarray(
        [int(round(value / dt)) for value in future_times], dtype=np.int32
    )
    first_step = max(
        int(round(args.start_time / dt)),
        (history_steps - 1) * history_stride,
    )
    last_step = min(
        int(round(args.end_time / dt)),
        int(round(float(configuration["T_final"]) / dt))
        - int(np.max(future_offsets)),
    )
    seeds = [int(value) for value in args.seeds.split(",") if value]
    args.outdir.mkdir(parents=True, exist_ok=False)

    print("[latent-identifiability] building train-only velocity POD", flush=True)
    basis, eigenvalues, covariance_by_regime = _train_velocity_pod(
        cache_dir,
        cases,
        first_step=first_step,
        last_step=last_step,
        sample_stride=sample_stride,
        basis_modes=int(args.basis_modes),
        first_unresolved_order=first_unresolved_order,
        cached_orders=cached_orders,
    )
    np.savez_compressed(
        args.outdir / "train_only_velocity_pod.npz",
        basis=basis,
        eigenvalues=eigenvalues,
        **{f"covariance_{name}": value for name, value in covariance_by_regime.items()},
    )

    report: dict[str, object] = {
        "reference_cache": str(cache_dir),
        "basis": {
            "training_split_only": True,
            "first_unresolved_order": first_unresolved_order,
            "cached_orders": cached_orders,
            "basis_modes": int(args.basis_modes),
            "ranks": ranks,
            "cumulative_energy": {
                str(rank): float(np.sum(eigenvalues[:rank]) / np.sum(eigenvalues))
                for rank in ranks
            },
        },
        "estimator": {
            "features": "C0:C2 history plus fixed-POD kinetic latent history",
            "target": "future central heat-flux divergence at fixed offsets",
            "future_times": future_times,
            "history_span": float(args.history_span),
            "history_cadence": float(args.history_cadence),
            "sample_cadence": float(args.sample_cadence),
            "start_time": first_step * dt,
            "end_time": last_step * dt,
            "feature_modes": int(args.feature_modes),
            "latent_representation": str(args.latent_representation),
            "neighbors": int(args.neighbors),
            "projection_dim": int(args.projection_dim),
            "seeds": seeds,
        },
        "regimes": {},
    }

    for regime in REGIMES:
        print(f"[latent-identifiability] loading {regime}", flush=True)
        train_x, train_y = _collect_split(
            cache_dir,
            cases,
            basis,
            ranks,
            regime=regime,
            split="train",
            source_nx=source_nx,
            target_nx=int(args.target_nx),
            domain_length=float(configuration["teacher_L"]),
            history_steps=history_steps,
            history_stride=history_stride,
            sample_stride=sample_stride,
            first_step=first_step,
            last_step=last_step,
            future_offsets=future_offsets,
            feature_modes=int(args.feature_modes),
            first_unresolved_order=first_unresolved_order,
            latent_representation=str(args.latent_representation),
        )
        heldout_x, heldout_y = _collect_split(
            cache_dir,
            cases,
            basis,
            ranks,
            regime=regime,
            split="heldout",
            source_nx=source_nx,
            target_nx=int(args.target_nx),
            domain_length=float(configuration["teacher_L"]),
            history_steps=history_steps,
            history_stride=history_stride,
            sample_stride=sample_stride,
            first_step=first_step,
            last_step=last_step,
            future_offsets=future_offsets,
            feature_modes=int(args.feature_modes),
            first_unresolved_order=first_unresolved_order,
            latent_representation=str(args.latent_representation),
        )
        scores: Dict[str, object] = {}
        for rank in [0] + ranks:
            seed_scores = [
                _knn_score(
                    train_x[rank],
                    train_y,
                    heldout_x[rank],
                    heldout_y,
                    base_feature_count=train_x[0].shape[1],
                    neighbors=int(args.neighbors),
                    projection_dim=int(args.projection_dim),
                    seed=seed,
                )
                for seed in seeds
            ]
            normalized = np.asarray(
                [score["normalized_mse"] for score in seed_scores]
            )
            scores[str(rank)] = {
                "median_normalized_mse": float(np.median(normalized)),
                "minimum_normalized_mse": float(np.min(normalized)),
                "maximum_normalized_mse": float(np.max(normalized)),
                "median_relative_target_mse": float(
                    np.median([score["relative_target_mse"] for score in seed_scores])
                ),
                "seed_scores": seed_scores,
            }
        base = float(scores["0"]["median_normalized_mse"])
        for rank in ranks:
            scores[str(rank)]["ratio_to_moment_history"] = float(
                scores[str(rank)]["median_normalized_mse"] / max(base, 1e-30)
            )
        report["regimes"][regime] = {
            "train_samples": int(train_y.shape[0]),
            "heldout_samples": int(heldout_y.shape[0]),
            "heldout_current_flux_reconstruction_relative_mse": (
                _current_flux_reconstruction_scores(
                    cache_dir,
                    cases,
                    basis,
                    ranks,
                    regime=regime,
                    split="heldout",
                    source_nx=source_nx,
                    target_nx=int(args.target_nx),
                    domain_length=float(configuration["teacher_L"]),
                    first_step=first_step,
                    last_step=last_step,
                    sample_stride=sample_stride,
                    first_unresolved_order=first_unresolved_order,
                )
            ),
            "scores": scores,
        }
        print(
            f"[latent-identifiability] {regime} ratios="
            + ",".join(
                f"r{rank}:{scores[str(rank)]['ratio_to_moment_history']:.3f}"
                for rank in ranks
            ),
            flush=True,
        )

    (args.outdir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    x = np.asarray([0] + ranks)
    for regime in REGIMES:
        scores = report["regimes"][regime]["scores"]
        y = [scores[str(rank)]["median_normalized_mse"] for rank in x]
        axis.plot(x, y, marker="o", label=regime)
    axis.axhline(1.0, color="#6b7280", linewidth=1.0, linestyle="--")
    axis.set_xlabel("kinetic latent rank (0 = moment history only)")
    axis.set_ylabel("heldout normalized future-q MSE")
    axis.set_yscale("log")
    axis.grid(alpha=0.22)
    axis.legend()
    fig.savefig(args.outdir / "future_q_identifiability.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
