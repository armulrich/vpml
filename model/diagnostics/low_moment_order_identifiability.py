"""Estimate closure identifiability as the retained moment order increases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model.train.interface_flux_data import load_ic_manifest


REGIMES = ("linear_landau", "nonlinear_landau_strong")


def _history_summary(values: np.ndarray) -> np.ndarray:
    """Compress a causal complex history without mixing spatial modes."""
    return np.concatenate(
        (
            values[:, -1],
            np.mean(values, axis=1),
            np.std(values.real, axis=1) + 1j * np.std(values.imag, axis=1),
            values[:, -1] - values[:, 0],
        ),
        axis=1,
    )


def _complex_to_real(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values.real, values.imag), axis=-1).reshape(
        values.shape[0], -1
    ).astype(np.float32)


def _raw_moment_weights(moment_order: int) -> np.ndarray:
    """Map normalized probabilists' Hermite coefficients to a raw moment."""
    weights = np.zeros((moment_order + 1,), dtype=np.float64)
    for coefficient_order in range(moment_order + 1):
        remainder = moment_order - coefficient_order
        if remainder % 2:
            continue
        pairs = remainder // 2
        weights[coefficient_order] = math.factorial(moment_order) / (
            (2**pairs)
            * math.factorial(pairs)
            * math.sqrt(math.factorial(coefficient_order))
        )
    return weights


def _case_order_data(
    history: np.ndarray,
    *,
    max_order: int,
    orders: list[int],
    history_steps: int,
    history_stride: int,
    sample_stride: int,
    start_index: int,
    modes: int,
    domain_length: float,
    dt: float,
    history_representation: str,
    target_kind: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    times = np.arange(start_index, history.shape[0], sample_stride, dtype=np.int32)
    offsets = np.arange(history_steps - 1, -1, -1, dtype=np.int32) * history_stride
    indices = times[:, None] - offsets[None]
    coefficients = np.asarray(
        history[indices, : max_order + 2, 1 : modes + 1], dtype=np.complex128
    )
    wave_numbers = 2.0 * math.pi * np.arange(1, modes + 1) / float(domain_length)

    features: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    for order in orders:
        retained = coefficients[:, :, : order + 1]
        flattened = retained.reshape(retained.shape[0], retained.shape[1], -1)
        if history_representation == "full":
            feature = flattened.reshape(flattened.shape[0], -1)
        elif history_representation == "summary":
            feature = _history_summary(flattened)
        else:
            raise ValueError(f"Unknown history representation: {history_representation}")
        features[order] = _complex_to_real(feature)

        if target_kind == "raw_moment":
            weights = _raw_moment_weights(order + 1)
            boundary = np.einsum(
                "n,snk->sk", weights, coefficients[:, -1, : order + 2]
            )
        elif target_kind == "hermite_boundary":
            boundary = coefficients[:, -1, order + 1]
        else:
            raise ValueError(f"Unknown target kind: {target_kind}")
        target = 1j * wave_numbers[None] * boundary
        targets[order] = _complex_to_real(target)
    return features, targets


def _collect_regime(
    cache_dir: Path,
    cases: list[dict],
    *,
    regime: str,
    split: str,
    max_order: int,
    orders: list[int],
    history_steps: int,
    history_stride: int,
    sample_stride: int,
    common_start: float,
    modes: int,
    domain_length: float,
    dt: float,
    history_representation: str,
    target_kind: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    feature_rows = {order: [] for order in orders}
    target_rows = {order: [] for order in orders}
    history_extent = (history_steps - 1) * history_stride
    start_index = max(int(round(common_start / dt)), history_extent)

    for case in cases:
        if str(case["regime"]) != regime or str(case["split"]) != split:
            continue
        case_id = str(case["case_id"])
        history = np.load(cache_dir / "cases" / f"{case_id}.npy", mmap_mode="r")
        features, targets = _case_order_data(
            history,
            max_order=max_order,
            orders=orders,
            history_steps=history_steps,
            history_stride=history_stride,
            sample_stride=sample_stride,
            start_index=start_index,
            modes=modes,
            domain_length=domain_length,
            dt=dt,
            history_representation=history_representation,
            target_kind=target_kind,
        )
        for order in orders:
            feature_rows[order].append(features[order])
            target_rows[order].append(targets[order])

    if not feature_rows[orders[0]]:
        raise ValueError(f"No {split} cases found for {regime}")
    return (
        {order: np.concatenate(rows) for order, rows in feature_rows.items()},
        {order: np.concatenate(rows) for order, rows in target_rows.items()},
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
    active = scale > 1e-10
    train = (train_x[:, active] - center[active]) / scale[active]
    heldout = (heldout_x[:, active] - center[active]) / scale[active]

    rng = np.random.default_rng(seed)
    projected_dim = min(int(projection_dim), train.shape[1])
    projection = (
        rng.normal(size=(train.shape[1], projected_dim)).astype(np.float32)
        / math.sqrt(projected_dim)
    )
    train = train @ projection
    heldout = heldout @ projection

    target_center = np.mean(train_y, axis=0)
    baseline = float(np.mean(np.sum(np.square(heldout_y - target_center), axis=1)))
    target_energy = float(np.mean(np.sum(np.square(heldout_y), axis=1)))
    error = 0.0
    count = 0
    for start in range(0, heldout.shape[0], 128):
        query = heldout[start : start + 128]
        distances = (
            np.sum(np.square(query), axis=1)[:, None]
            + np.sum(np.square(train), axis=1)[None]
            - 2.0 * query @ train.T
        )
        nearest = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
        prediction = np.mean(train_y[nearest], axis=1)
        truth = heldout_y[start : start + query.shape[0]]
        error += float(np.sum(np.square(prediction - truth)))
        count += int(query.shape[0])
    mse = error / max(count, 1)
    return {
        "normalized_knn_mse": mse / max(baseline, 1e-30),
        "mse": mse,
        "baseline_mse": baseline,
        "heldout_target_rms": math.sqrt(target_energy / max(heldout_y.shape[1], 1)),
        "active_feature_count": int(np.sum(active)),
        "projection_dim": int(projected_dim),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-order", type=int, default=8)
    parser.add_argument(
        "--orders",
        default="",
        help="Comma-separated retained orders; defaults to every order through max-order",
    )
    parser.add_argument("--history-span", type=float, default=5.0)
    parser.add_argument("--history-cadence", type=float, default=0.1)
    parser.add_argument("--sample-cadence", type=float, default=1.0)
    parser.add_argument("--common-start", type=float, default=20.0)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument(
        "--history-representation", choices=("full", "summary"), default="full"
    )
    parser.add_argument(
        "--target-kind",
        choices=("raw_moment", "hermite_boundary"),
        default="raw_moment",
    )
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seeds", default="1729,1730,1731,1732,1733")
    args = parser.parse_args()

    orders = (
        sorted({int(value) for value in args.orders.split(",") if value.strip()})
        if args.orders
        else list(range(2, args.max_order + 1))
    )
    if not orders or min(orders) < 2:
        raise ValueError("max-order must be at least 2")
    max_order = max(orders)
    cache_dir = args.reference_cache.resolve()
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    manifest = load_ic_manifest(cache_dir / "ic_manifest.json")
    cases = [dict(case) for case in manifest["cases"]]
    configuration = dict(metadata["configuration"])
    dt = float(configuration["teacher_dt"])
    max_cached_order = int(configuration["max_projection_order"])
    if max_order + 1 >= max_cached_order:
        raise ValueError(
            f"Need C0:C{max_order + 1}, cache contains indices C0:C{max_cached_order - 1}"
        )
    history_stride = int(round(args.history_cadence / dt))
    sample_stride = int(round(args.sample_cadence / dt))
    history_steps = int(round(args.history_span / args.history_cadence)) + 1
    if min(history_stride, sample_stride, history_steps) <= 0:
        raise ValueError("History and sample cadences must be positive")
    seeds = [int(value) for value in str(args.seeds).split(",") if value.strip()]
    args.outdir.mkdir(parents=True, exist_ok=False)

    report: dict[str, object] = {
        "reference_cache": str(cache_dir),
        "estimator": {
            "description": "heldout kNN upper estimate of normalized conditional variance",
            "feature_basis": "C0:C_N history; invertibly equivalent to M0:M_N",
            "target": (
                "spatial derivative of full raw moment M_{N+1}"
                if args.target_kind == "raw_moment"
                else "spatial derivative of Hermite boundary coefficient C_{N+1}"
            ),
            "target_kind": args.target_kind,
            "orders": orders,
            "history_representation": args.history_representation,
            "history_span": float(args.history_span),
            "history_cadence": float(args.history_cadence),
            "sample_cadence": float(args.sample_cadence),
            "common_start": float(args.common_start),
            "modes": int(args.modes),
            "neighbors": int(args.neighbors),
            "projection_dim": int(args.projection_dim),
            "seeds": seeds,
        },
        "regimes": {},
    }

    for regime in REGIMES:
        print(f"[order-identifiability] loading {regime}", flush=True)
        train_x, train_y = _collect_regime(
            cache_dir,
            cases,
            regime=regime,
            split="train",
            max_order=max_order,
            orders=orders,
            history_steps=history_steps,
            history_stride=history_stride,
            sample_stride=sample_stride,
            common_start=args.common_start,
            modes=args.modes,
            domain_length=float(manifest["domain_length"]),
            dt=dt,
            history_representation=args.history_representation,
            target_kind=args.target_kind,
        )
        heldout_x, heldout_y = _collect_regime(
            cache_dir,
            cases,
            regime=regime,
            split="heldout",
            max_order=max_order,
            orders=orders,
            history_steps=history_steps,
            history_stride=history_stride,
            sample_stride=sample_stride,
            common_start=args.common_start,
            modes=args.modes,
            domain_length=float(manifest["domain_length"]),
            dt=dt,
            history_representation=args.history_representation,
            target_kind=args.target_kind,
        )
        order_report = {}
        for order in orders:
            scores = [
                _knn_score(
                    train_x[order],
                    train_y[order],
                    heldout_x[order],
                    heldout_y[order],
                    neighbors=args.neighbors,
                    projection_dim=args.projection_dim,
                    seed=seed,
                )
                for seed in seeds
            ]
            values = np.asarray([score["normalized_knn_mse"] for score in scores])
            order_report[str(order)] = {
                "normalized_knn_mse_mean": float(np.mean(values)),
                "normalized_knn_mse_std": float(np.std(values)),
                "normalized_knn_mse_by_seed": values.tolist(),
                "train_samples": int(train_x[order].shape[0]),
                "heldout_samples": int(heldout_x[order].shape[0]),
                "feature_count": int(train_x[order].shape[1]),
                **{
                    key: value
                    for key, value in scores[0].items()
                    if key != "normalized_knn_mse"
                },
            }
            print(
                f"[order-identifiability] {regime} N={order} "
                f"A_N={np.mean(values):.6f}+/-{np.std(values):.6f}",
                flush=True,
            )
        report["regimes"][regime] = order_report

    strong = report["regimes"]["nonlinear_landau_strong"]
    baseline = float(strong["2"]["normalized_knn_mse_mean"])
    relative_drops = {
        str(order): float(
            (baseline - strong[str(order)]["normalized_knn_mse_mean"])
            / max(baseline, 1e-30)
        )
        for order in orders
    }
    report["strong_improvement_over_N2"] = relative_drops

    with (args.outdir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for regime, color in zip(REGIMES, ("#2563eb", "#dc2626")):
        values = report["regimes"][regime]
        means = [values[str(order)]["normalized_knn_mse_mean"] for order in orders]
        errors = [values[str(order)]["normalized_knn_mse_std"] for order in orders]
        axis.errorbar(orders, means, yerr=errors, marker="o", color=color, label=regime)
    axis.axhline(1.0, color="#6b7280", linewidth=1.0, linestyle="--")
    axis.set_xlabel("highest retained moment N")
    axis.set_ylabel(r"empirical $A_N$ (held-out normalized kNN MSE)")
    axis.set_title("Closure identifiability versus retained moment order")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "moment_order_identifiability.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
