"""Test finite-history closure identifiability with cross-IC collision pairs.

The diagnostic compares held-out histories against training histories only.  It
asks whether nearby causal low-moment histories require different instantaneous
heat-flux gradients or different future electric-field trajectories, and whether
the unresolved Hermite state can disambiguate those local neighborhoods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


REGIMES = (
    "linear_landau",
    "nonlinear_landau_weak",
    "nonlinear_landau_strong",
)


@dataclass
class CaseSeries:
    case_id: str
    regime: str
    split: str
    cadence: float
    fields: np.ndarray
    closure_gradient: np.ndarray
    density_amplitude: np.ndarray
    field_energy: np.ndarray
    tail_coefficients: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restrict_coefficients(
    values: np.ndarray, *, source_nx: int, target_nx: int
) -> np.ndarray:
    target_nk = target_nx // 2 + 1
    result = np.asarray(values[..., :target_nk], dtype=np.complex128).copy()
    if source_nx != target_nx:
        result *= float(target_nx) / float(source_nx)
        result[..., -1] = 2.0 * np.real(result[..., -1])
    return result


def _load_case(
    path: Path,
    *,
    case_id: str,
    regime: str,
    split: str,
    source_nx: int,
    target_nx: int,
    domain_length: float,
    reference_stride: int,
    reference_dt: float,
) -> CaseSeries:
    history = np.load(path, mmap_mode="r")
    low = _restrict_coefficients(
        history[::reference_stride, :4],
        source_nx=source_nx,
        target_nx=target_nx,
    )
    physical = np.fft.irfft(low, n=target_nx, axis=-1)
    density_perturbation = physical[:, 0]
    momentum = physical[:, 1]
    density = 1.0 + density_perturbation
    velocity = momentum / np.maximum(density, 1.0e-8)
    second_perturbation = density_perturbation + math.sqrt(2.0) * physical[:, 2]
    pressure_perturbation = second_perturbation - momentum * velocity

    wave_numbers = 2.0 * math.pi * np.fft.rfftfreq(
        target_nx, d=domain_length / target_nx
    )
    density_hat = np.fft.rfft(
        density_perturbation
        - np.mean(density_perturbation, axis=-1, keepdims=True),
        axis=-1,
    )
    field_hat = np.zeros_like(density_hat)
    field_hat[:, 1:] = 1j * density_hat[:, 1:] / wave_numbers[1:]
    field = np.fft.irfft(field_hat, n=target_nx, axis=-1)

    pressure = 1.0 + pressure_perturbation
    raw_third = math.sqrt(6.0) * physical[:, 3] + 3.0 * momentum
    heat_flux = raw_third - 3.0 * velocity * pressure - density * velocity**3
    heat_flux_hat = np.fft.rfft(heat_flux, axis=-1)
    closure_gradient = np.fft.irfft(
        1j * wave_numbers[None] * heat_flux_hat,
        n=target_nx,
        axis=-1,
    )

    fields = np.stack(
        (density_perturbation, velocity, pressure_perturbation, field), axis=1
    ).astype(np.float32)
    amplitude = np.sqrt(np.mean(np.square(density_perturbation), axis=-1))
    energy = 0.5 * np.mean(np.square(field), axis=-1)

    # Current unresolved coefficients are read only at the one-unit sample grid.
    sample_step = int(round(1.0 / reference_dt))
    sample_indices = np.arange(
        int(round(20.0 / reference_dt)),
        int(round(110.0 / reference_dt)) + 1,
        sample_step,
        dtype=np.int32,
    )
    tail = _restrict_coefficients(
        history[sample_indices, 3:65],
        source_nx=source_nx,
        target_nx=target_nx,
    )
    return CaseSeries(
        case_id=case_id,
        regime=regime,
        split=split,
        cadence=reference_stride * reference_dt,
        fields=fields,
        closure_gradient=np.asarray(closure_gradient, dtype=np.float32),
        density_amplitude=np.asarray(amplitude, dtype=np.float64),
        field_energy=np.asarray(energy, dtype=np.float64),
        tail_coefficients=np.asarray(tail, dtype=np.complex64),
    )


def _complex_real(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values.real, values.imag), axis=-1).reshape(
        values.shape[0], -1
    )


def _phase_rotation(fields: np.ndarray, current_indices: np.ndarray, modes: int):
    density_hat = np.fft.rfft(fields[current_indices, 0], axis=-1)
    phase = np.angle(density_hat[:, 1])
    mode_numbers = np.arange(1, modes + 1, dtype=np.float64)
    return np.exp(-1j * phase[:, None] * mode_numbers[None])


def _history_features(
    case: CaseSeries,
    *,
    span: float,
    temporal_rank: int,
    modes: int,
    include_closure: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cadence = case.cadence
    history_steps = int(round(span / cadence))
    current_indices = np.arange(
        int(round(20.0 / cadence)),
        int(round(110.0 / cadence)) + 1,
        int(round(1.0 / cadence)),
        dtype=np.int32,
    )
    offsets = np.arange(history_steps - 1, -1, -1, dtype=np.int32)
    history_indices = current_indices[:, None] - offsets[None]
    amplitude = np.maximum(case.density_amplitude[current_indices], 1.0e-8)
    normalized_fields = np.arcsinh(
        case.fields[history_indices] / amplitude[:, None, None, None]
    )
    field_hat = (
        np.fft.rfft(normalized_fields, axis=-1)[..., 1 : modes + 1]
        / normalized_fields.shape[-1]
    )
    rotation = _phase_rotation(case.fields, current_indices, modes)
    field_hat *= rotation[:, None, None, :]
    channels = [field_hat]
    if include_closure:
        normalized_closure = np.arcsinh(
            case.closure_gradient[history_indices]
            / amplitude[:, None, None]
        )
        closure_hat = (
            np.fft.rfft(normalized_closure, axis=-1)[..., 1 : modes + 1]
            / normalized_closure.shape[-1]
        )
        closure_hat *= rotation[:, None, :]
        channels.append(closure_hat[:, :, None])
    history_hat = np.concatenate(channels, axis=2)
    real_history = np.concatenate((history_hat.real, history_hat.imag), axis=-1)
    retained = min(int(temporal_rank), int(real_history.shape[1]))
    temporal = dct(real_history, axis=1, norm="ortho")[:, :retained]
    current = real_history[:, -1]
    features = np.concatenate(
        (
            current.reshape(current.shape[0], -1),
            temporal.reshape(temporal.shape[0], -1),
            np.log(amplitude)[:, None],
        ),
        axis=1,
    ).astype(np.float32)

    closure_hat = (
        np.fft.rfft(case.closure_gradient[current_indices], axis=-1)[
            ..., 1 : modes + 1
        ]
        / case.closure_gradient.shape[-1]
    )
    closure_hat = closure_hat * rotation / amplitude[:, None]
    closure_target = _complex_real(closure_hat).astype(np.float32)

    future_times = np.asarray((0.1, 1.0, 5.0, 10.0), dtype=np.float64)
    future_offsets = np.rint(future_times / cadence).astype(np.int32)
    future_indices = current_indices[:, None] + future_offsets[None]
    future_field_hat = (
        np.fft.rfft(case.fields[future_indices, 3], axis=-1)[..., 1 : modes + 1]
        / case.fields.shape[-1]
    )
    future_field_hat *= rotation[:, None, :]
    future_field_hat /= amplitude[:, None, None]
    future_field = _complex_real(future_field_hat).astype(np.float32)
    future_energy = np.log(
        np.maximum(case.field_energy[future_indices], 1.0e-30)
        / np.maximum(case.field_energy[current_indices, None], 1.0e-30)
    ).astype(np.float32)
    future_target = np.concatenate((future_field, future_energy), axis=1)

    tail = np.asarray(
        case.tail_coefficients[..., 1 : modes + 1], dtype=np.complex128
    )
    tail *= rotation[:, None, :]
    # Late linear density can be many orders below roundoff in the unresolved
    # coefficients.  Match the bounded dynamic scaling used by the model rather
    # than forming an unbounded coefficient/amplitude ratio in complex64.
    tail_real = np.arcsinh(tail.real / amplitude[:, None, None])
    tail_imag = np.arcsinh(tail.imag / amplitude[:, None, None])
    c3 = np.concatenate(
        (tail_real[:, :1], tail_imag[:, :1]), axis=-1
    ).reshape(tail.shape[0], -1).astype(np.float32)
    low_tail = np.concatenate(
        (tail_real[:, 1:13], tail_imag[:, 1:13]), axis=-1
    ).reshape(tail.shape[0], -1).astype(np.float32)
    tail_energy = np.log1p(
        np.sum(
            np.square(tail_real[:, 13:]) + np.square(tail_imag[:, 13:]),
            axis=-1,
        )
    ).astype(np.float32)
    tail_summary = np.concatenate((low_tail, tail_energy), axis=1)

    metadata = {
        "closure_target": closure_target,
        "future_target": future_target,
        "c3": c3,
        "tail": tail_summary,
        "case_id": np.full(current_indices.shape, case.case_id),
        "regime": np.full(current_indices.shape, case.regime),
        "split": np.full(current_indices.shape, case.split),
        "time": current_indices.astype(np.float64) * cadence,
        "current_index": current_indices,
    }
    return features, metadata


def _combine(rows: Iterable[tuple[np.ndarray, dict[str, np.ndarray]]]):
    rows = list(rows)
    features = np.concatenate([row[0] for row in rows], axis=0)
    keys = rows[0][1]
    metadata = {
        key: np.concatenate([row[1][key] for row in rows], axis=0) for key in keys
    }
    return features, metadata


def _standardize(train: np.ndarray, heldout: np.ndarray):
    center = np.mean(train, axis=0, dtype=np.float64)
    scale = np.std(train, axis=0, dtype=np.float64)
    active = scale > 1.0e-8
    train_scaled = ((train[:, active] - center[active]) / scale[active]).astype(
        np.float32
    )
    heldout_scaled = (
        (heldout[:, active] - center[active]) / scale[active]
    ).astype(np.float32)
    active_indices = np.flatnonzero(active)
    shape_block = active_indices != train.shape[1] - 1
    if np.any(shape_block):
        block_scale = math.sqrt(float(np.sum(shape_block)))
        train_scaled[:, shape_block] /= block_scale
        heldout_scaled[:, shape_block] /= block_scale
    return train_scaled, heldout_scaled, int(np.sum(active))


def _target_scale(train: np.ndarray, heldout: np.ndarray):
    center = np.mean(train, axis=0, dtype=np.float64)
    scale = np.std(train, axis=0, dtype=np.float64)
    safe_scale = np.where(scale > 1.0e-10, scale, 1.0)
    return (
        ((train - center) / safe_scale).astype(np.float32),
        ((heldout - center) / safe_scale).astype(np.float32),
    )


def _target_scale_by_regime(
    train: np.ndarray,
    heldout: np.ndarray,
    train_regime: np.ndarray,
    heldout_regime: np.ndarray,
):
    train_scaled = np.empty_like(train, dtype=np.float32)
    heldout_scaled = np.empty_like(heldout, dtype=np.float32)
    for regime in REGIMES:
        train_mask = train_regime == regime
        heldout_mask = heldout_regime == regime
        scaled_train, scaled_heldout = _target_scale(
            train[train_mask], heldout[heldout_mask]
        )
        train_scaled[train_mask] = scaled_train
        heldout_scaled[heldout_mask] = scaled_heldout
    return train_scaled, heldout_scaled


def _row_mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.mean(np.square(left - right), axis=1)


def _analyze_projection(
    train_x: np.ndarray,
    heldout_x: np.ndarray,
    train_meta: dict[str, np.ndarray],
    heldout_meta: dict[str, np.ndarray],
    *,
    projection_dim: int,
    neighbors: int,
    seed: int,
):
    train, heldout, active_features = _standardize(train_x, heldout_x)
    rng = np.random.default_rng(seed)
    projected_dim = min(int(projection_dim), int(train.shape[1]))
    if projected_dim < train.shape[1]:
        projection = rng.normal(
            size=(train.shape[1], projected_dim)
        ).astype(np.float32) / math.sqrt(float(projected_dim))
        train = train @ projection
        heldout = heldout @ projection
    neighbor_count = min(
        int(neighbors),
        min(int(np.sum(train_meta["regime"] == regime)) for regime in REGIMES),
    )
    nearest = np.empty((heldout.shape[0], neighbor_count), dtype=np.int32)
    history_distance = np.empty((heldout.shape[0], neighbor_count), dtype=np.float64)
    for regime in REGIMES:
        train_rows = np.flatnonzero(train_meta["regime"] == regime)
        heldout_rows = np.flatnonzero(heldout_meta["regime"] == regime)
        tree = cKDTree(train[train_rows])
        distance, local_nearest = tree.query(heldout[heldout_rows], k=neighbor_count)
        if local_nearest.ndim == 1:
            local_nearest = local_nearest[:, None]
            distance = distance[:, None]
        nearest[heldout_rows] = train_rows[local_nearest]
        history_distance[heldout_rows] = distance

    closure_train, closure_heldout = _target_scale_by_regime(
        train_meta["closure_target"],
        heldout_meta["closure_target"],
        train_meta["regime"],
        heldout_meta["regime"],
    )
    future_train, future_heldout = _target_scale_by_regime(
        train_meta["future_target"],
        heldout_meta["future_target"],
        train_meta["regime"],
        heldout_meta["regime"],
    )
    c3_train, c3_heldout = _target_scale_by_regime(
        train_meta["c3"],
        heldout_meta["c3"],
        train_meta["regime"],
        heldout_meta["regime"],
    )
    tail_train, tail_heldout = _target_scale_by_regime(
        train_meta["tail"],
        heldout_meta["tail"],
        train_meta["regime"],
        heldout_meta["regime"],
    )

    nearest_first = nearest[:, 0]
    closure_distance = _row_mse(closure_heldout, closure_train[nearest_first])
    future_distance = _row_mse(future_heldout, future_train[nearest_first])
    c3_distance = _row_mse(c3_heldout, c3_train[nearest_first])
    tail_distance = _row_mse(tail_heldout, tail_train[nearest_first])

    # A random comparison from another amplitude regime is too easy and makes
    # low-amplitude histories look artificially identifiable.  Draw the null
    # pair from the same physical regime while retaining disjoint ICs.
    random_index = np.empty((heldout.shape[0],), dtype=np.int32)
    for regime in REGIMES:
        query_rows = np.flatnonzero(heldout_meta["regime"] == regime)
        candidate_rows = np.flatnonzero(train_meta["regime"] == regime)
        random_index[query_rows] = rng.choice(candidate_rows, size=query_rows.size)
    random_future = _row_mse(future_heldout, future_train[random_index])
    random_closure = _row_mse(closure_heldout, closure_train[random_index])

    neighbor_future = np.mean(
        np.square(future_train[nearest] - future_heldout[:, None]), axis=2
    )
    neighbor_closure = np.mean(
        np.square(closure_train[nearest] - closure_heldout[:, None]), axis=2
    )
    neighbor_c3 = np.mean(
        np.square(c3_train[nearest] - c3_heldout[:, None]), axis=2
    )
    neighbor_tail = np.mean(
        np.square(tail_train[nearest] - tail_heldout[:, None]), axis=2
    )
    c3_choice = np.argmin(neighbor_c3, axis=1)
    tail_choice = np.argmin(neighbor_tail, axis=1)
    oracle_choice = np.argmin(neighbor_future, axis=1)
    rows = np.arange(heldout.shape[0])

    prediction = np.mean(future_train[nearest[:, : min(8, nearest.shape[1])]], axis=1)
    future_knn_mse = float(np.mean(np.square(prediction - future_heldout)))
    future_baseline_mse = float(
        np.mean(np.square(future_heldout - np.mean(future_train, axis=0)))
    )
    closure_prediction = np.mean(
        closure_train[nearest[:, : min(8, nearest.shape[1])]], axis=1
    )
    closure_knn_mse = float(np.mean(np.square(closure_prediction - closure_heldout)))
    closure_baseline_mse = float(
        np.mean(np.square(closure_heldout - np.mean(closure_train, axis=0)))
    )

    history_first = history_distance[:, 0]
    collision = np.zeros((heldout.shape[0],), dtype=bool)
    for regime in REGIMES:
        mask = heldout_meta["regime"] == regime
        nearest_quartile = history_first[mask] <= np.quantile(
            history_first[mask], 0.25
        )
        future_random_median = np.median(random_future[mask])
        collision[mask] = nearest_quartile & (
            future_distance[mask] >= future_random_median
        )
    correlation = spearmanr(tail_distance, future_distance).statistic
    return {
        "arrays": {
            "nearest": nearest,
            "history_distance": history_first,
            "closure_distance": closure_distance,
            "future_distance": future_distance,
            "random_future": random_future,
            "random_closure": random_closure,
            "c3_distance": c3_distance,
            "tail_distance": tail_distance,
            "c3_selected_future": neighbor_future[rows, c3_choice],
            "tail_selected_future": neighbor_future[rows, tail_choice],
            "oracle_selected_future": neighbor_future[rows, oracle_choice],
            "collision": collision,
        },
        "summary": {
            "active_history_features": active_features,
            "projected_features": projected_dim,
            "future_knn8_normalized_mse": future_knn_mse
            / max(future_baseline_mse, 1.0e-30),
            "closure_knn8_normalized_mse": closure_knn_mse
            / max(closure_baseline_mse, 1.0e-30),
            "nearest_future_to_random_ratio": float(np.mean(future_distance))
            / max(float(np.mean(random_future)), 1.0e-30),
            "nearest_closure_to_random_ratio": float(np.mean(closure_distance))
            / max(float(np.mean(random_closure)), 1.0e-30),
            "collision_fraction": float(np.mean(collision)),
            "tail_future_spearman": float(correlation),
            "c3_rerank_future_ratio": float(
                np.mean(neighbor_future[rows, c3_choice])
                / max(np.mean(future_distance), 1.0e-30)
            ),
            "tail_rerank_future_ratio": float(
                np.mean(neighbor_future[rows, tail_choice])
                / max(np.mean(future_distance), 1.0e-30)
            ),
            "local_future_oracle_ratio": float(
                np.mean(neighbor_future[rows, oracle_choice])
                / max(np.mean(future_distance), 1.0e-30)
            ),
        },
    }


def _regime_summary(result, regimes: np.ndarray):
    summaries = {}
    arrays = result["arrays"]
    for regime in REGIMES:
        mask = regimes == regime
        if not np.any(mask):
            continue
        nearest_future = arrays["future_distance"][mask]
        summaries[regime] = {
            "heldout_samples": int(np.sum(mask)),
            "nearest_future_to_random_ratio": float(np.mean(nearest_future))
            / max(float(np.mean(arrays["random_future"][mask])), 1.0e-30),
            "nearest_closure_to_random_ratio": float(
                np.mean(arrays["closure_distance"][mask])
            )
            / max(float(np.mean(arrays["random_closure"][mask])), 1.0e-30),
            "collision_fraction": float(np.mean(arrays["collision"][mask])),
            "tail_future_spearman": float(
                spearmanr(
                    arrays["tail_distance"][mask], arrays["future_distance"][mask]
                ).statistic
            ),
            "c3_rerank_future_ratio": float(
                np.mean(arrays["c3_selected_future"][mask])
                / max(np.mean(nearest_future), 1.0e-30)
            ),
            "tail_rerank_future_ratio": float(
                np.mean(arrays["tail_selected_future"][mask])
                / max(np.mean(nearest_future), 1.0e-30)
            ),
            "local_future_oracle_ratio": float(
                np.mean(arrays["oracle_selected_future"][mask])
                / max(np.mean(nearest_future), 1.0e-30)
            ),
        }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--history-spans", default="5,10,20")
    parser.add_argument("--history-cadence", type=float, default=0.1)
    parser.add_argument("--temporal-rank", type=int, default=16)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--target-nx", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=192)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--seeds", default="1729,1730,1731")
    args = parser.parse_args()

    cache = args.reference_cache.resolve()
    metadata_path = cache / "metadata.json"
    manifest_path = cache / "ic_manifest.json"
    metadata = json.loads(metadata_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    configuration = metadata["configuration"]
    reference_dt = float(configuration["teacher_dt"])
    reference_stride = int(round(args.history_cadence / reference_dt))
    if not math.isclose(reference_stride * reference_dt, args.history_cadence):
        raise ValueError("history cadence must be an integer multiple of reference dt")
    args.outdir.mkdir(parents=True, exist_ok=False)

    cases = []
    for index, case in enumerate(manifest["cases"], start=1):
        print(
            f"[history-collision] loading {index}/{len(manifest['cases'])} "
            f"{case['case_id']}",
            flush=True,
        )
        cases.append(
            _load_case(
                cache / "cases" / f"{case['case_id']}.npy",
                case_id=str(case["case_id"]),
                regime=str(case["regime"]),
                split=str(case["split"]),
                source_nx=int(configuration["teacher_Nx"]),
                target_nx=int(args.target_nx),
                domain_length=float(configuration["teacher_L"]),
                reference_stride=reference_stride,
                reference_dt=reference_dt,
            )
        )

    spans = [float(value) for value in args.history_spans.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    report = {
        "scope": (
            "Finite-data heldout-to-training collision diagnostic; supports or "
            "falsifies observed identifiability but is not a mathematical proof."
        ),
        "reference": {
            "cache": str(cache),
            "metadata_sha256": _sha256(metadata_path),
            "manifest_sha256": _sha256(manifest_path),
        },
        "configuration": {
            "history_spans": spans,
            "history_cadence": args.history_cadence,
            "sample_cadence": 1.0,
            "sample_interval": [20.0, 110.0],
            "future_times": [0.1, 1.0, 5.0, 10.0],
            "temporal_rank": args.temporal_rank,
            "modes": args.modes,
            "target_nx": args.target_nx,
            "projection_dim": args.projection_dim,
            "neighbors": args.neighbors,
            "seeds": seeds,
            "match_policy": "heldout queries against disjoint training ICs",
            "normalization": "current-density RMS, arcsinh, phase aligned, train standardized",
        },
        "results": {},
    }
    plot_payload = {}
    example_payload = None
    for include_closure in (False, True):
        feature_name = "state_plus_exact_closure_history" if include_closure else "state_history"
        report["results"][feature_name] = {}
        plot_payload[feature_name] = {}
        for span in spans:
            print(
                f"[history-collision] analyzing {feature_name} span={span:g}",
                flush=True,
            )
            features, sample = _combine(
                _history_features(
                    case,
                    span=span,
                    temporal_rank=args.temporal_rank,
                    modes=args.modes,
                    include_closure=include_closure,
                )
                for case in cases
            )
            train_mask = sample["split"] == "train"
            heldout_mask = sample["split"] == "heldout"
            train_meta = {key: value[train_mask] for key, value in sample.items()}
            heldout_meta = {key: value[heldout_mask] for key, value in sample.items()}
            seed_results = []
            for seed in seeds:
                seed_results.append(
                    _analyze_projection(
                        features[train_mask],
                        features[heldout_mask],
                        train_meta,
                        heldout_meta,
                        projection_dim=args.projection_dim,
                        neighbors=args.neighbors,
                        seed=seed,
                    )
                )
            metric_names = seed_results[0]["summary"]
            aggregate = {
                name: {
                    "median": float(
                        np.median([result["summary"][name] for result in seed_results])
                    ),
                    "minimum": float(
                        np.min([result["summary"][name] for result in seed_results])
                    ),
                    "maximum": float(
                        np.max([result["summary"][name] for result in seed_results])
                    ),
                }
                for name in metric_names
            }
            per_regime_by_seed = [
                _regime_summary(result, heldout_meta["regime"])
                for result in seed_results
            ]
            per_regime = {}
            for regime in REGIMES:
                names = per_regime_by_seed[0][regime]
                per_regime[regime] = {
                    name: float(
                        np.median([values[regime][name] for values in per_regime_by_seed])
                    )
                    for name in names
                }
            report["results"][feature_name][str(span)] = {
                "train_samples": int(np.sum(train_mask)),
                "heldout_samples": int(np.sum(heldout_mask)),
                "aggregate": aggregate,
                "per_regime": per_regime,
            }
            plot_payload[feature_name][span] = {
                "result": seed_results[0],
                "heldout": heldout_meta,
                "train": train_meta,
                "cases": cases,
            }
            if include_closure and span == spans[0]:
                example_payload = plot_payload[feature_name][span]

    report_path = args.outdir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "nearest_future_to_random_ratio", "Nearest-history future divergence"),
        (axes[1], "tail_rerank_future_ratio", "Tail reranking within 32 neighbors"),
    ):
        for feature_name, linestyle in (
            ("state_history", "--"),
            ("state_plus_exact_closure_history", "-"),
        ):
            for regime, color in zip(REGIMES, ("#2563eb", "#ca8a04", "#dc2626")):
                values = [
                    report["results"][feature_name][str(span)]["per_regime"][regime][metric]
                    for span in spans
                ]
                axis.plot(
                    spans,
                    values,
                    marker="o",
                    linestyle=linestyle,
                    color=color,
                    label=f"{regime} | {feature_name}",
                )
        axis.axhline(1.0, color="#64748b", linewidth=1.0, linestyle=":")
        axis.set_xlabel("history span")
        axis.set_ylabel("error ratio (lower is better)")
        axis.set_title(title)
        axis.grid(alpha=0.22)
    axes[1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(args.outdir / "history_span_and_tail_reranking.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(REGIMES), len(spans), figsize=(13.5, 10.0), constrained_layout=True
    )
    for row, regime in enumerate(REGIMES):
        for column, span in enumerate(spans):
            payload = plot_payload["state_plus_exact_closure_history"][span]
            result = payload["result"]
            mask = payload["heldout"]["regime"] == regime
            history_distance = result["arrays"]["history_distance"][mask]
            future_distance = result["arrays"]["future_distance"][mask]
            random_scale = np.mean(result["arrays"]["random_future"][mask])
            color = result["arrays"]["tail_distance"][mask]
            axis = axes[row, column]
            points = axis.scatter(
                history_distance,
                future_distance / max(random_scale, 1.0e-30),
                c=np.log10(np.maximum(color, 1.0e-8)),
                s=14,
                alpha=0.72,
                cmap="viridis",
            )
            axis.set_yscale("log")
            axis.grid(alpha=0.18)
            axis.set_title(f"{regime}, H={span:g}")
            if row == len(REGIMES) - 1:
                axis.set_xlabel("nearest history distance")
            if column == 0:
                axis.set_ylabel("future divergence / random")
    fig.colorbar(points, ax=axes, label="log10 hidden-tail distance", shrink=0.7)
    fig.savefig(args.outdir / "nearest_history_future_collisions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(REGIMES), figsize=(13.0, 4.0), constrained_layout=True)
    payload = plot_payload["state_plus_exact_closure_history"][spans[0]]
    for axis, regime in zip(axes, REGIMES):
        mask = payload["heldout"]["regime"] == regime
        nearest = payload["result"]["arrays"]["future_distance"][mask]
        random_values = payload["result"]["arrays"]["random_future"][mask]
        scale = max(float(np.mean(random_values)), 1.0e-30)
        axis.hist(nearest / scale, bins=24, alpha=0.75, label="nearest H")
        axis.hist(random_values / scale, bins=24, alpha=0.45, label="random")
        axis.set_title(regime)
        axis.set_xlabel("future divergence / random mean")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("held-out windows")
    axes[-1].legend(frameon=False)
    fig.savefig(args.outdir / "collision_histograms_H5.png", dpi=180)
    plt.close(fig)

    assert example_payload is not None
    arrays = example_payload["result"]["arrays"]
    heldout = example_payload["heldout"]
    train = example_payload["train"]
    strong = heldout["regime"] == "nonlinear_landau_strong"
    strong_indices = np.flatnonzero(strong)
    history_order = np.argsort(arrays["history_distance"][strong_indices])
    candidate_pool = strong_indices[history_order[: max(8, len(history_order) // 4)]]
    selected = candidate_pool[
        np.argsort(arrays["future_distance"][candidate_pool])[-3:]
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), constrained_layout=True)
    case_by_id = {case.case_id: case for case in cases}
    for row, query_index in enumerate(selected):
        train_index = arrays["nearest"][query_index, 0]
        query_case = case_by_id[str(heldout["case_id"][query_index])]
        neighbor_case = case_by_id[str(train["case_id"][train_index])]
        query_time = float(heldout["time"][query_index])
        neighbor_time = float(train["time"][train_index])
        q_center = int(round(query_time / query_case.cadence))
        n_center = int(round(neighbor_time / neighbor_case.cadence))
        offsets = np.arange(-50, 101)
        q_energy = query_case.field_energy[q_center + offsets]
        n_energy = neighbor_case.field_energy[n_center + offsets]
        axes[row, 0].semilogy(offsets * query_case.cadence, q_energy, label="heldout")
        axes[row, 0].semilogy(offsets * neighbor_case.cadence, n_energy, label="train neighbor")
        axes[row, 0].axvline(0.0, color="#64748b", linestyle=":")
        axes[row, 0].set_title(
            f"{query_case.case_id}@{query_time:g} vs {neighbor_case.case_id}@{neighbor_time:g}"
        )
        axes[row, 0].set_xlabel("time relative to match")
        axes[row, 0].set_ylabel("electric energy")
        q_norm = np.sqrt(np.mean(np.square(query_case.closure_gradient), axis=1))
        n_norm = np.sqrt(np.mean(np.square(neighbor_case.closure_gradient), axis=1))
        axes[row, 1].plot(offsets * query_case.cadence, q_norm[q_center + offsets], label="heldout")
        axes[row, 1].plot(offsets * neighbor_case.cadence, n_norm[n_center + offsets], label="train neighbor")
        axes[row, 1].axvline(0.0, color="#64748b", linestyle=":")
        axes[row, 1].set_xlabel("time relative to match")
        axes[row, 1].set_ylabel("RMS heat-flux gradient")
        for axis in axes[row]:
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    fig.savefig(args.outdir / "strong_collision_examples.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
