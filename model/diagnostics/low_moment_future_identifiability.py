"""Test whether retained kinetic state identifies the same future electric field."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
from scipy.spatial import cKDTree

from model.train.interface_flux_data import load_ic_manifest


REGIMES = ("linear_landau", "nonlinear_landau_strong")


def _complex_to_real(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values.real, values.imag), axis=-1).reshape(
        values.shape[0], -1
    ).astype(np.float32)


def _case_samples(
    history: np.ndarray,
    *,
    orders: list[int],
    history_steps: int,
    history_stride: int,
    sample_stride: int,
    first_index: int,
    last_index: int,
    future_offsets: np.ndarray,
    modes: int,
    temporal_rank: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    times = np.arange(first_index, last_index + 1, sample_stride, dtype=np.int32)
    lag_offsets = (
        np.arange(history_steps - 1, -1, -1, dtype=np.int32) * history_stride
    )
    history_indices = times[:, None] - lag_offsets[None]
    max_order = max(orders)
    coefficients = np.asarray(
        history[history_indices, : max_order + 1, 1 : modes + 1],
        dtype=np.complex64,
    )

    phase = np.angle(coefficients[:, -1, 0, 0])
    mode_numbers = np.arange(1, modes + 1, dtype=np.float32)
    rotation = np.exp(-1j * phase[:, None] * mode_numbers[None])
    coefficients *= rotation[:, None, None, :]

    current_density = coefficients[:, -1, 0]
    amplitude = np.sqrt(2.0 * np.sum(np.abs(current_density) ** 2, axis=1))
    amplitude = np.maximum(amplitude, 1e-12)
    coefficients /= amplitude[:, None, None, None]

    target_density = np.asarray(
        history[times[:, None] + future_offsets[None], 0, 1 : modes + 1],
        dtype=np.complex64,
    )
    target_density *= rotation[:, None, :]
    target_density /= amplitude[:, None, None]
    # Poisson contributes the fixed 1/k weighting; the overall sign is immaterial.
    target_field = target_density / mode_numbers[None, None, :]
    target = _complex_to_real(target_field)

    features: dict[int, np.ndarray] = {}
    log_amplitude = np.log(amplitude)[:, None].astype(np.float32)
    for order in orders:
        retained = coefficients[:, :, : order + 1]
        current = retained[:, -1]
        parts = [_complex_to_real(current)]
        if temporal_rank > 0:
            temporal = dct(retained, axis=1, norm="ortho")[:, :temporal_rank]
            parts.append(_complex_to_real(temporal))
        parts.append(log_amplitude)
        features[order] = np.concatenate(parts, axis=1)
    return features, target


def _collect_regime(
    cache_dir: Path,
    cases: list[dict],
    *,
    regime: str,
    split: str,
    orders: list[int],
    history_steps: int,
    history_stride: int,
    sample_stride: int,
    first_index: int,
    last_index: int,
    future_offsets: np.ndarray,
    modes: int,
    temporal_rank: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    feature_rows = {order: [] for order in orders}
    target_rows = []
    for case in cases:
        if str(case["regime"]) != regime or str(case["split"]) != split:
            continue
        history = np.load(
            cache_dir / "cases" / f"{case['case_id']}.npy", mmap_mode="r"
        )
        features, target = _case_samples(
            history,
            orders=orders,
            history_steps=history_steps,
            history_stride=history_stride,
            sample_stride=sample_stride,
            first_index=first_index,
            last_index=last_index,
            future_offsets=future_offsets,
            modes=modes,
            temporal_rank=temporal_rank,
        )
        for order in orders:
            feature_rows[order].append(features[order])
        target_rows.append(target)
    if not target_rows:
        raise ValueError(f"No {split} cases found for {regime}")
    return (
        {order: np.concatenate(rows) for order, rows in feature_rows.items()},
        np.concatenate(target_rows),
    )


def _knn_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    heldout_x: np.ndarray,
    heldout_y: np.ndarray,
    *,
    neighbors: int,
    projection_dim: int,
    seed: int,
) -> dict[str, float]:
    center = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    active = scale > 1e-8
    train = (train_x[:, active] - center[active]) / scale[active]
    heldout = (heldout_x[:, active] - center[active]) / scale[active]

    projected_dim = min(projection_dim, train.shape[1])
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
    target_center = np.mean(train_y, axis=0)
    error = float(np.mean(np.sum(np.square(prediction - heldout_y), axis=1)))
    baseline = float(np.mean(np.sum(np.square(heldout_y - target_center), axis=1)))
    target_energy = float(np.mean(np.sum(np.square(heldout_y), axis=1)))
    return {
        "normalized_mse": error / max(baseline, 1e-30),
        "relative_target_mse": error / max(target_energy, 1e-30),
        "mse": error,
        "baseline_mse": baseline,
        "active_feature_count": int(np.sum(active)),
        "projection_dim": int(projected_dim),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--orders", default="2,3,4,8,16,32,48,64")
    parser.add_argument("--history-span", type=float, default=5.0)
    parser.add_argument("--history-cadence", type=float, default=0.5)
    parser.add_argument("--sample-cadence", type=float, default=2.0)
    parser.add_argument("--start-time", type=float, default=20.0)
    parser.add_argument("--end-time", type=float, default=100.0)
    parser.add_argument("--future-times", default="5,10,20")
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--temporal-rank", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seeds", default="1729,1730,1731,1732,1733")
    args = parser.parse_args()

    orders = sorted({int(value) for value in args.orders.split(",") if value})
    future_times = [float(value) for value in args.future_times.split(",") if value]
    cache_dir = args.reference_cache.resolve()
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    manifest = load_ic_manifest(cache_dir / "ic_manifest.json")
    cases = [dict(case) for case in manifest["cases"]]
    dt = float(metadata["configuration"]["teacher_dt"])
    max_cached_order = int(metadata["configuration"]["max_projection_order"]) - 1
    if not orders or min(orders) < 2 or max(orders) > max_cached_order:
        raise ValueError(f"Orders must lie between 2 and {max_cached_order}")

    history_stride = int(round(args.history_cadence / dt))
    history_steps = int(round(args.history_span / args.history_cadence)) + 1
    sample_stride = int(round(args.sample_cadence / dt))
    future_offsets = np.asarray(
        [int(round(value / dt)) for value in future_times], dtype=np.int32
    )
    first_index = max(
        int(round(args.start_time / dt)), (history_steps - 1) * history_stride
    )
    last_index = min(
        int(round(args.end_time / dt)),
        int(round(float(metadata["configuration"]["T_final"]) / dt))
        - int(np.max(future_offsets)),
    )
    seeds = [int(value) for value in args.seeds.split(",") if value]
    args.outdir.mkdir(parents=True, exist_ok=False)

    report: dict[str, object] = {
        "reference_cache": str(cache_dir),
        "estimator": {
            "description": "heldout kNN prediction of a fixed future electric-field target",
            "target": "phase-aligned E modes at fixed future offsets",
            "orders": orders,
            "future_times": future_times,
            "history_span": args.history_span,
            "history_cadence": args.history_cadence,
            "sample_cadence": args.sample_cadence,
            "start_time": first_index * dt,
            "end_time": last_index * dt,
            "modes": args.modes,
            "temporal_rank": args.temporal_rank,
            "neighbors": args.neighbors,
            "projection_dim": args.projection_dim,
            "seeds": seeds,
        },
        "regimes": {},
    }

    for regime in REGIMES:
        print(f"[future-identifiability] loading {regime}", flush=True)
        train_x, train_y = _collect_regime(
            cache_dir,
            cases,
            regime=regime,
            split="train",
            orders=orders,
            history_steps=history_steps,
            history_stride=history_stride,
            sample_stride=sample_stride,
            first_index=first_index,
            last_index=last_index,
            future_offsets=future_offsets,
            modes=args.modes,
            temporal_rank=args.temporal_rank,
        )
        heldout_x, heldout_y = _collect_regime(
            cache_dir,
            cases,
            regime=regime,
            split="heldout",
            orders=orders,
            history_steps=history_steps,
            history_stride=history_stride,
            sample_stride=sample_stride,
            first_index=first_index,
            last_index=last_index,
            future_offsets=future_offsets,
            modes=args.modes,
            temporal_rank=args.temporal_rank,
        )
        regime_report = {}
        for order in orders:
            scores = [
                _knn_score(
                    train_x[order],
                    train_y,
                    heldout_x[order],
                    heldout_y,
                    neighbors=args.neighbors,
                    projection_dim=args.projection_dim,
                    seed=seed,
                )
                for seed in seeds
            ]
            values = np.asarray([score["normalized_mse"] for score in scores])
            regime_report[str(order)] = {
                "normalized_mse_mean": float(np.mean(values)),
                "normalized_mse_std": float(np.std(values)),
                "normalized_mse_by_seed": values.tolist(),
                "relative_target_mse_mean": float(
                    np.mean([score["relative_target_mse"] for score in scores])
                ),
                "train_samples": int(train_y.shape[0]),
                "heldout_samples": int(heldout_y.shape[0]),
                "feature_count": int(train_x[order].shape[1]),
                "active_feature_count": scores[0]["active_feature_count"],
                "projection_dim": scores[0]["projection_dim"],
            }
            print(
                f"[future-identifiability] {regime} C0:C{order} "
                f"NMSE={np.mean(values):.6f}+/-{np.std(values):.6f}",
                flush=True,
            )
        report["regimes"][regime] = regime_report

    strong = report["regimes"]["nonlinear_landau_strong"]
    baseline = float(strong["2"]["normalized_mse_mean"])
    report["strong_improvement_over_C0_C2"] = {
        str(order): float(
            (baseline - strong[str(order)]["normalized_mse_mean"])
            / max(baseline, 1e-30)
        )
        for order in orders
    }
    with (args.outdir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for regime, color in zip(REGIMES, ("#2563eb", "#dc2626")):
        values = report["regimes"][regime]
        means = [values[str(order)]["normalized_mse_mean"] for order in orders]
        errors = [values[str(order)]["normalized_mse_std"] for order in orders]
        axis.errorbar(orders, means, yerr=errors, marker="o", color=color, label=regime)
    axis.axhline(1.0, color="#6b7280", linewidth=1.0, linestyle="--")
    axis.set_xlabel("highest exposed Hermite coefficient")
    axis.set_ylabel("held-out normalized future-E prediction error")
    axis.set_title("Does kinetic state identify the same future electric field?")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "future_field_identifiability.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
