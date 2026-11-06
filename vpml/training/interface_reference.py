"""Kinetic teacher histories and Fourier-Hermite reference preparation."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from model.train.interface_flux_data import (
    IC_SPLIT_TRAIN,
    case_shard_is_complete,
    evaluate_manifest_case,
    initialize_reference_cache,
    load_or_create_ic_manifest,
    load_sharded_reference,
    reference_cache_directory,
    write_case_shard,
)
from vpml.core import Array, GRID_CUBIC_SPLINE_TEACHER_BACKEND
from vpml.fourier import restrict_rfft_numpy
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    build_cubic_spline_hermite_projection_matrix,
    gaussian_pdf,
    normalize_density_on_grid,
    project_distribution_snapshot_with_hermite_matrix,
    run_semilagrangian_vlasov_poisson,
)
from vpml.training.interface_configuration import (
    INTERFACE_FLUX_PROJECTION_SCHEME,
    interface_flux_rollout_coeff_key,
)


def restrict_rfft_coefficients_to_grid(
    coefficients: np.ndarray,
    *,
    source_Nx: int,
    target_Nx: int,
) -> np.ndarray:
    """Restrict unnormalized rFFT coefficients to a coarser periodic grid."""
    return restrict_rfft_numpy(
        coefficients, source_nx=source_Nx, target_nx=target_Nx
    )


def maxwellian_equilibrium(v: Array) -> Array:
    return normalize_density_on_grid(gaussian_pdf(v, mean=0.0, sigma=1.0), v)


def _projected_history_projector(
    v: Array,
    projection_order: int,
    *,
    projection_quadrature_Nv: int,
    equilibrium: Array,
    projection_matrix: Optional[Array] = None,
    vth: float = 1.0,
    storage_dtype: object = jnp.complex128,
):
    matrix = (
        build_cubic_spline_hermite_projection_matrix(
            v,
            int(projection_order),
            int(projection_quadrature_Nv),
            vth=float(vth),
        )
        if projection_matrix is None
        else jnp.asarray(projection_matrix, dtype=jnp.float64)
    )

    def projector(f_state: Array) -> Array:
        projected = project_distribution_snapshot_with_hermite_matrix(
            f_state,
            matrix,
            equilibrium=equilibrium,
        )
        return projected.astype(storage_dtype)

    return projector


def _run_landau_teacher_projected_history(
    config: PhysicalGridVlasovPoissonConfig,
    perturbation_x: Array,
    *,
    projection_order: int,
    projection_quadrature_Nv: int,
    history_stride: int,
    equilibrium: Optional[Array] = None,
    projection_matrix: Optional[Array] = None,
    storage_dtype: object = jnp.complex128,
    return_raw: bool = False,
    return_field_history: bool = False,
) -> tuple:
    v = config.v
    equilibrium = (
        maxwellian_equilibrium(v)
        if equilibrium is None
        else jnp.asarray(equilibrium, dtype=jnp.float64)
    )
    f0 = equilibrium[:, None] * (1.0 + jnp.asarray(perturbation_x, dtype=jnp.float64)[None, :])
    raw = run_semilagrangian_vlasov_poisson(
        config,
        f0,
        history_stride=history_stride,
        return_state_history=True,
        return_field_history=bool(return_field_history),
        history_projector=_projected_history_projector(
            v,
            int(projection_order),
            projection_quadrature_Nv=int(projection_quadrature_Nv),
            equilibrium=equilibrium,
            projection_matrix=projection_matrix,
            vth=1.0,
            storage_dtype=storage_dtype,
        ),
    )
    history = np.asarray(raw["state_history"], dtype=np.dtype(storage_dtype))
    k_arr = np.asarray(raw["k_arr"], dtype=np.float64)
    if return_raw:
        return history, k_arr, raw
    return history, k_arr


def build_interface_flux_rollout_reference_dataset(
    *,
    reference_cache_root: Path,
    ic_manifest_path: Path,
    regimes: Sequence[str],
    teacher_Nx: int,
    teacher_Nv: int,
    projection_quadrature_Nv: int,
    teacher_L: float,
    teacher_vmin: float,
    teacher_vmax: float,
    teacher_dt: float,
    T_final: float,
    teacher_poisson_sign: float,
    snapshot_times: Sequence[float],
    cases_per_regime: int,
    heldout_cases_per_regime: int,
    ic_generation_seed: int,
    ic_split_seed: int,
    ic_modes: Sequence[float],
    Nv_targets: Sequence[int],
    min_projection_order: Optional[int] = None,
    selected_case_ids: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Dict[str, object]], int, Dict[str, object], Path]:
    """Build or resume one complex64 projected-history shard per complete IC."""
    max_projection_order = max(max(int(v) for v in Nv_targets) + 1, int(min_projection_order or 0))
    manifest = load_or_create_ic_manifest(
        ic_manifest_path,
        cases_per_regime=int(cases_per_regime),
        heldout_per_regime=int(heldout_cases_per_regime),
        generation_seed=int(ic_generation_seed),
        split_seed=int(ic_split_seed),
        modes=tuple(float(value) for value in ic_modes),
        domain_length=float(teacher_L),
    )
    coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
    configuration = {
        "dataset_format": "landau_interface_flux_sharded_reference_v1",
        "manifest_sha256": str(manifest["sha256"]),
        "regimes": list(str(value) for value in regimes),
        "teacher_backend": GRID_CUBIC_SPLINE_TEACHER_BACKEND,
        "teacher_Nx": int(teacher_Nx),
        "teacher_Nv": int(teacher_Nv),
        "projection_quadrature_Nv": int(projection_quadrature_Nv),
        "projection_quadrature_scheme": INTERFACE_FLUX_PROJECTION_SCHEME,
        "teacher_L": float(teacher_L),
        "teacher_vmin": float(teacher_vmin),
        "teacher_vmax": float(teacher_vmax),
        "teacher_dt": float(teacher_dt),
        "T_final": float(T_final),
        "teacher_poisson_sign": float(teacher_poisson_sign),
        "max_projection_order": int(max_projection_order),
        "snapshot_times": [float(value) for value in snapshot_times],
        "solver_version": "semilagrangian_vlasov_poisson_strang_cubic_v1",
        "storage_dtype": np.dtype(np.complex64).name,
        "selected_case_ids": (
            sorted(str(value) for value in selected_case_ids)
            if selected_case_ids is not None
            else None
        ),
    }
    cache_dir = reference_cache_directory(reference_cache_root, configuration)
    initialize_reference_cache(
        cache_dir,
        configuration=configuration,
        manifest=manifest,
    )

    projection_v = jnp.linspace(
        float(teacher_vmin),
        float(teacher_vmax),
        int(teacher_Nv),
        dtype=jnp.float64,
    )
    projection_equilibrium = maxwellian_equilibrium(projection_v)
    projection_matrix = build_cubic_spline_hermite_projection_matrix(
        projection_v,
        int(max_projection_order),
        int(projection_quadrature_Nv),
        vth=1.0,
    )
    config = PhysicalGridVlasovPoissonConfig(
        Nx=int(teacher_Nx),
        Nv=int(teacher_Nv),
        Lx=float(teacher_L),
        vmin=float(teacher_vmin),
        vmax=float(teacher_vmax),
        dt=float(teacher_dt),
        T=float(T_final),
        poisson_sign=float(teacher_poisson_sign),
        snapshot_times=tuple(float(value) for value in snapshot_times),
    )
    expected_shape = (
        int(config.nsteps) + 1,
        int(max_projection_order),
        int(config.Nk),
    )
    x = np.asarray(config.x, dtype=np.float64)
    active = set(str(value) for value in regimes)
    selected_cases = [
        case
        for case in manifest["cases"]
        if str(case["regime"]) in active
        and (
            selected_case_ids is None
            or str(case["case_id"]) in set(str(value) for value in selected_case_ids)
        )
    ]
    for case_number, case in enumerate(selected_cases, start=1):
        case_id = str(case["case_id"])
        if case_shard_is_complete(
            cache_dir,
            case_id,
            expected_shape=expected_shape,
            expected_dtype=np.dtype(np.complex64),
        ):
            print(
                f"[cache] [{case_number}/{len(selected_cases)}] reusing {case_id}"
            )
            continue
        print(
            f"[cache] [{case_number}/{len(selected_cases)}] generating {case_id} "
            f"({case['split']})"
        )
        perturbation = evaluate_manifest_case(case, x)
        history, _, raw = _run_landau_teacher_projected_history(
            config,
            perturbation,
            projection_order=int(max_projection_order),
            projection_quadrature_Nv=int(projection_quadrature_Nv),
            history_stride=1,
            equilibrium=projection_equilibrium,
            projection_matrix=projection_matrix,
            storage_dtype=jnp.complex64,
            return_raw=True,
            return_field_history=True,
        )
        if tuple(history.shape) != expected_shape:
            raise ValueError(
                f"Projected history for {case_id} has shape {history.shape}, "
                f"expected {expected_shape}"
            )
        write_case_shard(
            cache_dir,
            case,
            history,
            history_times=np.asarray(raw["state_history_times"], dtype=np.float64),
            snapshots={
                "x": np.asarray(raw["x"], dtype=np.float64),
                "v": np.asarray(raw["v"], dtype=np.float64),
                "k_arr": np.asarray(raw["k_arr"], dtype=np.float64),
                "perturbation_x": np.asarray(perturbation, dtype=np.float64),
                "times": np.asarray(raw["times"], dtype=np.float64),
                "energy": np.asarray(raw["energy"], dtype=np.float64),
                "E_hat_hist_times": np.asarray(
                    raw["E_hat_hist_times"], dtype=np.float64
                ),
                "E_hat_hist": np.asarray(raw["E_hat_hist"], dtype=np.complex128),
                "snapshot_times": np.asarray(raw["snapshot_times"], dtype=np.float64),
                "snapshot_f": np.asarray(raw["snapshot_f"], dtype=np.float64),
                "equilibrium": np.asarray(
                    projection_equilibrium, dtype=np.float64
                ),
            },
        )
        del history, raw
        gc.collect()

    selected_manifest = dict(manifest)
    selected_manifest["cases"] = selected_cases
    dataset = load_sharded_reference(
        cache_dir,
        selected_manifest,
        coeff_key=coeff_key,
    )
    return dataset, max_projection_order, manifest, cache_dir


def _reference_case_histories(
    group: Dict[str, object],
    coeff_key: str,
) -> Tuple[np.ndarray, ...]:
    histories = group[coeff_key]
    if isinstance(histories, np.ndarray) and histories.ndim == 4:
        return tuple(histories[index] for index in range(int(histories.shape[0])))
    return tuple(histories)


def _reference_case_splits(
    group: Dict[str, object],
    case_count: int,
) -> np.ndarray:
    values = group.get("case_splits")
    if values is None:
        return np.full((int(case_count),), IC_SPLIT_TRAIN, dtype=np.str_)
    splits = np.asarray(values, dtype=np.str_)
    if splits.shape != (int(case_count),):
        raise ValueError(
            f"case_splits must have shape ({case_count},), got {splits.shape}"
        )
    return splits
