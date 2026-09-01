"""Train a compact kinetic latent update before low-moment integration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from model.diagnostics.kinetic_latent_identifiability import _train_velocity_pod
from model.train.interface_flux_data import load_ic_manifest
from model.train.low_moment_closure import (
    _central_heat_flux_numpy,
    low_hermite_coefficients_to_conservative,
)
from vpml.jax_runtime import print_jax_runtime_summary
from vpml.kinetic_latent import (
    init_kinetic_latent_dynamics,
    init_state_conditioned_latent_operator,
    reconstruct_first_unresolved_field,
    rollout_state_conditioned_kinetic_latent_dynamics,
    stabilize_linear_latent_propagator,
)


REGIMES = (
    "linear_landau",
    "nonlinear_landau_weak",
    "nonlinear_landau_strong",
)


def _selected_cases(cases: Iterable[dict], regime: str, split: str) -> list[dict]:
    return [
        case
        for case in cases
        if str(case["regime"]) == regime and str(case["split"]) == split
    ]


def _cache_key(configuration: dict, args: argparse.Namespace) -> str:
    payload = {
        "manifest": configuration["manifest_sha256"],
        "rank": int(args.latent_rank),
        "nx": int(args.nx),
        "cadence": float(args.cadence),
        "basis_modes": int(args.basis_modes),
        "format": "kinetic_latent_projected_fields_v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _build_projected_cache(
    reference_cache: Path,
    projected_cache: Path,
    cases: list[dict],
    configuration: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = float(configuration["teacher_dt"])
    stride = int(round(float(args.cadence) / dt))
    if not np.isclose(stride * dt, float(args.cadence)):
        raise ValueError("cadence must be an integer multiple of teacher_dt")
    source_nx = int(configuration["teacher_Nx"])
    cached_orders = int(configuration["max_projection_order"])
    first_unresolved_order = 3
    maximum_latent_rank = cached_orders - first_unresolved_order
    if int(args.latent_rank) > maximum_latent_rank:
        raise ValueError(
            f"latent_rank={args.latent_rank} exceeds the {maximum_latent_rank} "
            "unresolved Hermite coordinates stored in the reference cache"
        )
    if int(args.basis_modes) < int(args.latent_rank):
        raise ValueError("basis_modes must be at least latent_rank")
    basis, eigenvalues, _ = _train_velocity_pod(
        reference_cache,
        cases,
        first_step=0,
        last_step=int(round(float(configuration["T_final"]) / dt)),
        sample_stride=max(stride * 10, 1),
        basis_modes=int(args.basis_modes),
        first_unresolved_order=first_unresolved_order,
        cached_orders=cached_orders,
    )
    basis = np.asarray(basis[:, : int(args.latent_rank)], dtype=np.float32)
    projected_cache.mkdir(parents=True, exist_ok=False)
    (projected_cache / "cases").mkdir()
    np.save(projected_cache / "basis.npy", basis)

    spatial_modes = int(args.nx) // 2 + 1
    indices = np.arange(
        0,
        int(round(float(configuration["T_final"]) / dt)) + 1,
        stride,
        dtype=np.int32,
    )
    for case_index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        print(
            f"[cache] [{case_index}/{len(cases)}] projecting {case_id}",
            flush=True,
        )
        history = np.load(
            reference_cache / "cases" / f"{case_id}.npy",
            mmap_mode="r",
        )
        coefficients = np.asarray(
            history[indices, :cached_orders, :spatial_modes],
            dtype=np.complex64,
        ) / np.float32(source_nx)
        resolved = np.fft.irfft(
            coefficients[:, :first_unresolved_order],
            n=int(args.nx),
            axis=-1,
            norm="forward",
        ).astype(np.float32)
        latent_hat = np.einsum(
            "nr,tnk->trk",
            basis,
            coefficients[:, first_unresolved_order:],
            optimize=True,
        )
        latent = np.fft.irfft(
            latent_hat,
            n=int(args.nx),
            axis=-1,
            norm="forward",
        ).astype(np.float32)
        np.save(
            projected_cache / "cases" / f"{case_id}.npy",
            np.concatenate((resolved, latent), axis=1),
        )

    channel_count = 3 + int(args.latent_rank)
    total = np.zeros((channel_count,), dtype=np.float64)
    total_square = np.zeros_like(total)
    count = 0
    for case in cases:
        if str(case["split"]) != "train":
            continue
        values = np.load(
            projected_cache / "cases" / f"{case['case_id']}.npy",
            mmap_mode="r",
        )
        total += np.sum(values, axis=(0, 2), dtype=np.float64)
        total_square += np.sum(np.square(values), axis=(0, 2), dtype=np.float64)
        count += int(values.shape[0] * values.shape[2])
    center = total / float(count)
    scale = np.sqrt(np.maximum(total_square / float(count) - center * center, 1e-16))
    resolved_center = center[:3].astype(np.float32)
    resolved_scale = scale[:3].astype(np.float32)
    latent_center = center[3:].astype(np.float32)
    latent_scale = scale[3:].astype(np.float32)
    np.savez_compressed(
        projected_cache / "statistics.npz",
        resolved_center=resolved_center,
        resolved_scale=resolved_scale,
        latent_center=latent_center,
        latent_scale=latent_scale,
    )
    metadata = {
        "format": "kinetic_latent_projected_fields_v1",
        "cache_key": _cache_key(configuration, args),
        "source_reference_cache": str(reference_cache.resolve()),
        "training_split_only_basis_and_statistics": True,
        "rank": int(args.latent_rank),
        "nx": int(args.nx),
        "cadence": float(args.cadence),
        "basis_modes": int(args.basis_modes),
        "time_samples": int(indices.size),
        "cumulative_basis_energy": float(
            np.sum(eigenvalues[: int(args.latent_rank)]) / np.sum(eigenvalues)
        ),
    }
    (projected_cache / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return basis, resolved_center, resolved_scale, latent_center, latent_scale


def _load_or_build_projected_cache(
    reference_cache: Path,
    projected_cache: Path,
    cases: list[dict],
    configuration: dict,
    args: argparse.Namespace,
):
    expected_key = _cache_key(configuration, args)
    if not projected_cache.exists():
        return _build_projected_cache(
            reference_cache,
            projected_cache,
            cases,
            configuration,
            args,
        )
    metadata = json.loads((projected_cache / "metadata.json").read_text())
    if metadata.get("cache_key") != expected_key:
        raise ValueError(
            f"Projected cache configuration mismatch: {projected_cache}"
        )
    print(f"[cache] reusing projected latent cache {projected_cache}", flush=True)
    basis = np.load(projected_cache / "basis.npy")
    with np.load(projected_cache / "statistics.npz") as payload:
        return (
            basis,
            np.asarray(payload["resolved_center"]),
            np.asarray(payload["resolved_scale"]),
            np.asarray(payload["latent_center"]),
            np.asarray(payload["latent_scale"]),
        )


def _adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": jnp.array(0, dtype=jnp.int32), "m": zeros, "v": zeros}


def _tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    maximum = jnp.max(
        jnp.stack([jnp.max(jnp.abs(value)) for value in leaves])
    )
    scale = jnp.where(maximum > 0.0, maximum, 1.0)
    scaled_square = sum(
        jnp.sum(jnp.square(value / scale)) for value in leaves
    )
    return jnp.where(maximum > 0.0, scale * jnp.sqrt(scaled_square), 0.0)


def _tree_all_finite(tree):
    return jnp.all(
        jnp.stack(
            [jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(tree)]
        )
    )


def _adam_step(params, grads, state, learning_rate: float, grad_clip: float):
    norm = _tree_l2_norm(grads)
    factor = jnp.minimum(1.0, jnp.asarray(grad_clip, norm.dtype) / norm)
    grads = jax.tree_util.tree_map(lambda value: value * factor, grads)
    step = state["step"] + 1
    m = jax.tree_util.tree_map(
        lambda old, grad: 0.9 * old + 0.1 * grad,
        state["m"],
        grads,
    )
    v = jax.tree_util.tree_map(
        lambda old, grad: 0.999 * old + 0.001 * grad * grad,
        state["v"],
        grads,
    )
    params = jax.tree_util.tree_map(
        lambda value, mean, variance: value
        - learning_rate
        * (mean / (1.0 - 0.9**step))
        / (jnp.sqrt(variance / (1.0 - 0.999**step)) + 1e-8),
        params,
        m,
        v,
    )
    return params, {"step": step, "m": m, "v": v}, norm


def _sample_batch(
    rng: np.random.Generator,
    projected_cache: Path,
    train_by_regime: Dict[str, list[dict]],
    case_evolution_scales: Dict[str, float],
    *,
    batch_per_regime: int,
    horizon: int,
    latent_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    scales = []
    for regime in REGIMES:
        for _ in range(batch_per_regime):
            case = train_by_regime[regime][rng.integers(len(train_by_regime[regime]))]
            values = np.load(
                projected_cache / "cases" / f"{case['case_id']}.npy",
                mmap_mode="r",
            )
            start = int(rng.integers(0, values.shape[0] - horizon))
            rows.append(np.asarray(values[start : start + horizon + 1]))
            scales.append(float(case_evolution_scales[str(case["case_id"])]))
    batch = np.stack(rows).astype(np.float32)
    return (
        batch[:, :-1, :3],
        batch[:, :, 3 : 3 + latent_rank],
        np.asarray(scales, dtype=np.float32),
    )


def _case_evolution_scales(
    projected_cache: Path,
    cases: list[dict],
    latent_scale: np.ndarray,
) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for case in cases:
        if str(case["split"]) != "train":
            continue
        values = np.load(
            projected_cache / "cases" / f"{case['case_id']}.npy",
            mmap_mode="r",
        )[:, 3:]
        difference = (values[1:] - values[:-1]) / latent_scale[None, :, None]
        scale = float(np.mean(np.square(difference), dtype=np.float64))
        if not scale > 0.0:
            raise ValueError(f"Stationary training trajectory: {case['case_id']}")
        scales[str(case["case_id"])] = scale
    return scales


def _fit_or_load_linear_propagator(
    projected_cache: Path,
    cases: list[dict],
    *,
    latent_rank: int,
    nx: int,
    ridge: float,
    maximum_spectral_radius: float,
) -> np.ndarray:
    raw_path = projected_cache / f"linear_propagator_ridge{ridge:.3e}.npy"
    stable_path = projected_cache / (
        f"linear_propagator_ridge{ridge:.3e}"
        f"_radius{maximum_spectral_radius:.6f}.npy"
    )
    if stable_path.exists():
        print(
            f"[data] reusing stabilized train-only linear propagator {stable_path}",
            flush=True,
        )
        return np.load(stable_path)
    if raw_path.exists():
        print(f"[data] reusing raw train-only linear propagator {raw_path}", flush=True)
        propagator = np.load(raw_path)
    else:
        propagator = None
    modes = nx // 2 + 1
    feature_count = 3 + latent_rank
    if propagator is None:
        gram = np.zeros((modes, feature_count, feature_count), dtype=np.complex128)
        cross = np.zeros((modes, feature_count, latent_rank), dtype=np.complex128)
        for case in _selected_cases(cases, "linear_landau", "train"):
            values = np.load(
                projected_cache / "cases" / f"{case['case_id']}.npy",
                mmap_mode="r",
            )
            coefficients = np.fft.rfft(np.asarray(values), axis=-1, norm="forward")
            inputs = coefficients[:-1]
            targets = coefficients[1:, 3 : 3 + latent_rank]
            inputs_by_mode = inputs.transpose(0, 2, 1)
            gram += np.einsum(
                "tkc,tkd->kcd",
                inputs_by_mode.conj(),
                inputs_by_mode,
                optimize=True,
            )
            cross += np.einsum(
                "tkc,tkr->kcr",
                inputs_by_mode.conj(),
                targets.transpose(0, 2, 1),
                optimize=True,
            )
        propagator = np.empty_like(cross)
        identity = np.eye(feature_count)
        for mode in range(modes):
            regularization = ridge * float(np.trace(gram[mode]).real) / feature_count
            propagator[mode] = np.linalg.solve(
                gram[mode] + regularization * identity,
                cross[mode],
            )
        propagator = propagator.astype(np.complex64)
        np.save(raw_path, propagator)
        print(f"[data] saved raw train-only linear propagator {raw_path}", flush=True)
    stabilized, diagnostics = stabilize_linear_latent_propagator(
        propagator,
        resolved_channels=3,
        maximum_spectral_radius=maximum_spectral_radius,
    )
    np.save(stable_path, stabilized)
    adjusted = ",".join(
        f"k{item['mode']}:{item['original_spectral_radius']:.6f}"
        f"->{maximum_spectral_radius:.6f}"
        for item in diagnostics
    )
    print(
        f"[data] saved stabilized train-only linear propagator {stable_path} "
        f"adjusted_modes={adjusted or 'none'}",
        flush=True,
    )
    return stabilized


def _heat_flux_gradient(
    resolved: np.ndarray,
    latent: np.ndarray,
    basis: np.ndarray,
    *,
    source_nx: int,
    nx: int,
    domain_length: float,
) -> np.ndarray:
    first_unresolved = np.asarray(
        reconstruct_first_unresolved_field(jnp.asarray(latent), jnp.asarray(basis))
    )
    fields = np.concatenate((resolved, first_unresolved[:, None]), axis=1)
    coefficients = (
        np.fft.rfft(fields, axis=-1, norm="forward") * float(source_nx)
    )
    state = low_hermite_coefficients_to_conservative(
        coefficients[:, :3],
        source_nx=source_nx,
        target_nx=nx,
        dtype=np.float64,
    )
    heat_flux = _central_heat_flux_numpy(
        coefficients,
        state,
        source_nx=source_nx,
        target_nx=nx,
    )
    wave_numbers = (2.0 * np.pi / float(domain_length)) * np.arange(nx // 2 + 1)
    return np.fft.irfft(
        1j * wave_numbers[None] * np.fft.rfft(heat_flux, axis=-1),
        n=nx,
        axis=-1,
    )


def _evaluate(
    params,
    projected_cache: Path,
    cases: list[dict],
    basis: np.ndarray,
    propagator: np.ndarray,
    resolved_center: np.ndarray,
    resolved_scale: np.ndarray,
    latent_center: np.ndarray,
    latent_scale: np.ndarray,
    *,
    depth: int,
    cadence: float,
    horizons: list[int],
    start_times: list[float],
    source_nx: int,
    nx: int,
    domain_length: float,
    dynamics_model: str,
) -> dict:
    report = {}
    rollout = jax.jit(
        lambda model_params, x, z: rollout_state_conditioned_kinetic_latent_dynamics(
            model_params,
            x,
            z,
            jnp.asarray(propagator),
            resolved_center=jnp.asarray(resolved_center),
            resolved_scale=jnp.asarray(resolved_scale),
            latent_center=jnp.asarray(latent_center),
            latent_scale=jnp.asarray(latent_scale),
            depth=depth,
            dynamics_model=dynamics_model,
        )
    )
    for horizon in horizons:
        by_regime = {}
        for regime in REGIMES:
            model_error = persistence_error = target_energy = latent_error = latent_energy = 0.0
            sample_count = 0
            for case in _selected_cases(cases, regime, "heldout"):
                values = np.load(
                    projected_cache / "cases" / f"{case['case_id']}.npy",
                    mmap_mode="r",
                )
                for start_time in start_times:
                    start = int(round(start_time / cadence))
                    if start + horizon >= values.shape[0]:
                        continue
                    window = np.asarray(values[start : start + horizon + 1])
                    resolved = window[:, :3]
                    latent = window[:, 3:]
                    predicted = np.asarray(
                        rollout(
                            params,
                            jnp.asarray(resolved[:-1][None]),
                            jnp.asarray(latent[0][None]),
                        )
                    )[0, -1]
                    target = latent[-1]
                    target_resolved = resolved[-1][None]
                    target_gradient = _heat_flux_gradient(
                        target_resolved,
                        target[None],
                        basis,
                        source_nx=source_nx,
                        nx=nx,
                        domain_length=domain_length,
                    )[0]
                    model_gradient = _heat_flux_gradient(
                        target_resolved,
                        predicted[None],
                        basis,
                        source_nx=source_nx,
                        nx=nx,
                        domain_length=domain_length,
                    )[0]
                    persistence_gradient = _heat_flux_gradient(
                        target_resolved,
                        latent[0][None],
                        basis,
                        source_nx=source_nx,
                        nx=nx,
                        domain_length=domain_length,
                    )[0]
                    model_error += float(np.sum(np.square(model_gradient - target_gradient)))
                    persistence_error += float(
                        np.sum(np.square(persistence_gradient - target_gradient))
                    )
                    target_energy += float(np.sum(np.square(target_gradient)))
                    latent_error += float(np.sum(np.square(predicted - target)))
                    latent_energy += float(np.sum(np.square(target)))
                    sample_count += 1
            by_regime[regime] = {
                "samples": sample_count,
                "model_relative_q_mse": model_error / max(target_energy, 1e-30),
                "persistence_relative_q_mse": persistence_error
                / max(target_energy, 1e-30),
                "model_to_persistence_q_ratio": model_error
                / max(persistence_error, 1e-30),
                "latent_relative_mse": latent_error / max(latent_energy, 1e-30),
            }
        report[str(horizon)] = {
            "physical_horizon": horizon * cadence,
            "regimes": by_regime,
        }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--latent-rank", type=int, default=16)
    parser.add_argument("--basis-modes", type=int, default=64)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--cadence", type=float, default=0.1)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=30)
    parser.add_argument("--batch-per-regime", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--linear-ridge", type=float, default=1e-4)
    parser.add_argument("--linear-max-spectral-radius", type=float, default=1.0)
    parser.add_argument(
        "--dynamics-model",
        choices=("additive_residual", "multiplicative_operator"),
        default="additive_residual",
    )
    parser.add_argument("--operator-rank", type=int, default=16)
    parser.add_argument("--operator-modes", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--validation-every", type=int, default=2)
    parser.add_argument("--validation-horizons", default="10,50")
    parser.add_argument("--validation-start-times", default="20,40,60,80")
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print_jax_runtime_summary(jax, context="kinetic latent dynamics probe")
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.outdir}")
    args.outdir.mkdir(parents=True)
    metadata = json.loads((args.reference_cache / "metadata.json").read_text())
    configuration = metadata["configuration"]
    manifest = load_ic_manifest(args.reference_cache / "ic_manifest.json")
    cases = [dict(case) for case in manifest["cases"]]
    basis, resolved_center, resolved_scale, latent_center, latent_scale = (
        _load_or_build_projected_cache(
            args.reference_cache,
            args.projected_cache,
            cases,
            configuration,
            args,
        )
    )
    train_by_regime = {
        regime: _selected_cases(cases, regime, "train") for regime in REGIMES
    }
    case_evolution_scales = _case_evolution_scales(
        args.projected_cache,
        cases,
        latent_scale,
    )
    propagator = _fit_or_load_linear_propagator(
        args.projected_cache,
        cases,
        latent_rank=int(args.latent_rank),
        nx=int(args.nx),
        ridge=float(args.linear_ridge),
        maximum_spectral_radius=float(args.linear_max_spectral_radius),
    )
    if args.dynamics_model == "multiplicative_operator":
        operator_modes = int(args.operator_modes) or (int(args.nx) // 2 + 1)
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(int(args.seed)),
            resolved_channels=3,
            latent_rank=int(args.latent_rank),
            width=int(args.width),
            depth=int(args.depth),
            spectral_modes=operator_modes,
            operator_rank=int(args.operator_rank),
            kernel_size=int(args.kernel_size),
        )
    else:
        operator_modes = 0
        params = init_kinetic_latent_dynamics(
            jax.random.PRNGKey(int(args.seed)),
            resolved_channels=3,
            latent_rank=int(args.latent_rank),
            width=int(args.width),
            depth=int(args.depth),
            kernel_size=int(args.kernel_size),
        )
    parameter_count = sum(value.size for value in jax.tree_util.tree_leaves(params))
    print(
        f"[model] dynamics={args.dynamics_model} parameters={parameter_count} "
        f"operator_rank={args.operator_rank if operator_modes else 0} "
        f"operator_modes={operator_modes}",
        flush=True,
    )
    if args.init_checkpoint is not None:
        with np.load(args.init_checkpoint) as payload:
            missing = [key for key in params if key not in payload.files]
            if missing:
                raise ValueError(f"Checkpoint is missing model parameters: {missing}")
            params = {
                key: jnp.asarray(payload[key], dtype=value.dtype)
                for key, value in params.items()
            }
        print(f"[train] initialized from {args.init_checkpoint}", flush=True)
    optimizer = _adam_init(params)

    def loss_function(model_params, resolved, latent, evolution_scale):
        predicted = rollout_state_conditioned_kinetic_latent_dynamics(
            model_params,
            resolved,
            latent[:, 0],
            jnp.asarray(propagator),
            resolved_center=jnp.asarray(resolved_center),
            resolved_scale=jnp.asarray(resolved_scale),
            latent_center=jnp.asarray(latent_center),
            latent_scale=jnp.asarray(latent_scale),
            depth=int(args.depth),
            dynamics_model=str(args.dynamics_model),
        )
        target = latent[:, 1:]
        normalized_error = (
            predicted - target
        ) / jnp.asarray(latent_scale)[None, None, :, None]
        error = jnp.mean(jnp.square(normalized_error), axis=(1, 2, 3))
        return jnp.mean(error / evolution_scale)

    def rollout_function(model_params, resolved, initial_latent):
        return rollout_state_conditioned_kinetic_latent_dynamics(
            model_params,
            resolved,
            initial_latent,
            jnp.asarray(propagator),
            resolved_center=jnp.asarray(resolved_center),
            resolved_scale=jnp.asarray(resolved_scale),
            latent_center=jnp.asarray(latent_center),
            latent_scale=jnp.asarray(latent_scale),
            depth=int(args.depth),
            dynamics_model=str(args.dynamics_model),
        )

    preflight_rng = np.random.default_rng(int(args.seed))
    preflight_resolved, preflight_latent, preflight_scale = _sample_batch(
        preflight_rng,
        args.projected_cache,
        train_by_regime,
        case_evolution_scales,
        batch_per_regime=1,
        horizon=int(args.rollout_steps),
        latent_rank=int(args.latent_rank),
    )
    preflight_prediction = jax.jit(rollout_function)(
        params,
        jnp.asarray(preflight_resolved),
        jnp.asarray(preflight_latent[:, 0]),
    )
    forward_finite = bool(jnp.all(jnp.isfinite(preflight_prediction)))
    forward_maximum = float(jnp.max(jnp.abs(preflight_prediction)))
    print(
        f"[preflight] forward_finite={int(forward_finite)} "
        f"maximum_abs_latent={forward_maximum:.6e}",
        flush=True,
    )
    if not forward_finite:
        raise FloatingPointError("Non-finite latent rollout during forward preflight")
    preflight_loss, preflight_grads = jax.jit(jax.value_and_grad(loss_function))(
        params,
        jnp.asarray(preflight_resolved),
        jnp.asarray(preflight_latent),
        jnp.asarray(preflight_scale),
    )
    gradient_finite = bool(_tree_all_finite(preflight_grads))
    preflight_gradient_norm = float(_tree_l2_norm(preflight_grads))
    print(
        f"[preflight] loss={float(preflight_loss):.6e} "
        f"gradient_finite={int(gradient_finite)} "
        f"gradient_norm={preflight_gradient_norm:.6e}",
        flush=True,
    )
    if not bool(jnp.isfinite(preflight_loss)) or not gradient_finite:
        raise FloatingPointError("Non-finite loss or gradient during preflight")

    @jax.jit
    def train_step(model_params, optimizer_state, resolved, latent, evolution_scale):
        loss, grads = jax.value_and_grad(loss_function)(
            model_params,
            resolved,
            latent,
            evolution_scale,
        )
        finite = jnp.isfinite(loss) & _tree_all_finite(grads)
        safe_grads = jax.tree_util.tree_map(
            lambda value: jnp.where(finite, value, jnp.zeros_like(value)),
            grads,
        )
        updated, optimizer_state, grad_norm = _adam_step(
            model_params,
            safe_grads,
            optimizer_state,
            float(args.learning_rate),
            float(args.grad_clip),
        )
        updated = jax.tree_util.tree_map(
            lambda new, old: jnp.where(finite, new, old),
            updated,
            model_params,
        )
        return updated, optimizer_state, loss, grad_norm, finite

    rng = np.random.default_rng(int(args.seed))
    validation_horizons = [int(value) for value in args.validation_horizons.split(",")]
    validation_start_times = [
        float(value) for value in args.validation_start_times.split(",")
    ]
    history = []
    best_checkpoint_score = float("inf")
    best_params = params
    for epoch in range(1, int(args.epochs) + 1):
        losses = []
        gradients = []
        for _ in range(int(args.steps_per_epoch)):
            resolved, latent, evolution_scale = _sample_batch(
                rng,
                args.projected_cache,
                train_by_regime,
                case_evolution_scales,
                batch_per_regime=int(args.batch_per_regime),
                horizon=int(args.rollout_steps),
                latent_rank=int(args.latent_rank),
            )
            params, optimizer, loss, grad_norm, finite = train_step(
                params,
                optimizer,
                jnp.asarray(resolved),
                jnp.asarray(latent),
                jnp.asarray(evolution_scale),
            )
            if not bool(finite):
                raise FloatingPointError(
                    f"Non-finite loss or gradient at epoch {epoch}"
                )
            losses.append(float(loss))
            gradients.append(float(grad_norm))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "grad_mean": float(np.mean(gradients)),
        }
        if epoch == 1 or epoch % int(args.validation_every) == 0:
            evaluation = _evaluate(
                params,
                args.projected_cache,
                cases,
                basis,
                propagator,
                resolved_center,
                resolved_scale,
                latent_center,
                latent_scale,
                depth=int(args.depth),
                cadence=float(args.cadence),
                horizons=validation_horizons,
                start_times=validation_start_times,
                source_nx=int(configuration["teacher_Nx"]),
                nx=int(args.nx),
                domain_length=float(configuration["teacher_L"]),
                dynamics_model=str(args.dynamics_model),
            )
            row["evaluation"] = evaluation
            longest = evaluation[str(max(validation_horizons))]["regimes"]
            checkpoint_score = max(
                values["latent_relative_mse"] for values in longest.values()
            )
            row["checkpoint_score"] = checkpoint_score
            if np.isfinite(checkpoint_score) and checkpoint_score < best_checkpoint_score:
                best_checkpoint_score = checkpoint_score
                best_params = params
                row["best_updated"] = True
        history.append(row)
        message = f"[train] epoch {epoch:03d}/{args.epochs:03d} loss={row['loss']:.6e} grad={row['grad_mean']:.3e}"
        if "evaluation" in row:
            parts = []
            for horizon in validation_horizons:
                strong = row["evaluation"][str(horizon)]["regimes"][
                    "nonlinear_landau_strong"
                ]
                parts.append(
                    f"H{horizon}_strong_q={strong['model_relative_q_mse']:.3e}"
                    f"/persist={strong['persistence_relative_q_mse']:.3e}"
                )
            message += " " + " ".join(parts)
        print(message, flush=True)

    final_evaluation = _evaluate(
        best_params,
        args.projected_cache,
        cases,
        basis,
        propagator,
        resolved_center,
        resolved_scale,
        latent_center,
        latent_scale,
        depth=int(args.depth),
        cadence=float(args.cadence),
        horizons=validation_horizons,
        start_times=validation_start_times,
        source_nx=int(configuration["teacher_Nx"]),
        nx=int(args.nx),
        domain_length=float(configuration["teacher_L"]),
        dynamics_model=str(args.dynamics_model),
    )
    checkpoint = {key: np.asarray(value) for key, value in best_params.items()}
    checkpoint.update(
        {
            "basis": basis,
            "resolved_center": resolved_center,
            "resolved_scale": resolved_scale,
            "latent_center": latent_center,
            "latent_scale": latent_scale,
            "linear_propagator": propagator,
            "dynamics_model": np.asarray(str(args.dynamics_model)),
            "operator_rank": np.asarray(int(args.operator_rank)),
            "operator_modes": np.asarray(operator_modes),
        }
    )
    np.savez_compressed(args.outdir / "best_kinetic_latent_dynamics.npz", **checkpoint)
    report = {
        "configuration": vars(args) | {"reference_cache": str(args.reference_cache), "projected_cache": str(args.projected_cache), "outdir": str(args.outdir)},
        "basis_cumulative_energy": json.loads(
            (args.projected_cache / "metadata.json").read_text()
        )["cumulative_basis_energy"],
        "history": history,
        "best_evaluation": final_evaluation,
    }
    (args.outdir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
