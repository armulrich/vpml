"""Derive primitive-field RMS floors from a paired physical-grid refinement."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from model.train.low_moment_closure import (
    REGIMES,
    _primitive_numpy,
    low_hermite_coefficients_to_conservative,
)
from vpml.physical_grid import (
    gaussian_pdf,
    hermite_dual_basis_scaled,
    normalize_density_on_grid,
    trapezoid_quadrature_weights,
)


CASES = (
    ("linear_landau", "linear_sample00"),
    ("nonlinear_landau_weak", "weak_eps0p1"),
    ("nonlinear_landau_strong", "strong_eps0p5"),
)


def _project_low_moments(case_path: Path, metadata: dict, rollout_nx: int) -> np.ndarray:
    nv = int(metadata["Nv"])
    nx = int(metadata["Nx"])
    v = np.linspace(float(metadata["vmin"]), float(metadata["vmax"]), nv)
    basis = np.asarray(hermite_dual_basis_scaled(3, v), dtype=np.float64)
    weights = np.asarray(trapezoid_quadrature_weights(v), dtype=np.float64)
    projector = basis * weights[None, :]
    equilibrium = np.asarray(
        normalize_density_on_grid(gaussian_pdf(v, mean=0.0, sigma=1.0), v),
        dtype=np.float64,
    )
    with np.load(case_path, allow_pickle=False) as payload:
        snapshots = np.asarray(payload["snapshot_f"], dtype=np.float64)
    coefficients = np.empty((snapshots.shape[0], 3, nx // 2 + 1), np.complex128)
    for index, snapshot in enumerate(snapshots):
        moments = projector @ (snapshot - equilibrium[:, None])
        coefficients[index] = np.fft.rfft(moments, axis=-1)
    del snapshots
    state = low_hermite_coefficients_to_conservative(
        coefficients, source_nx=nx, target_nx=rollout_nx, dtype=np.float64
    )
    return _primitive_numpy(state, float(metadata["L"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--coarse-Nx", type=int, default=1024)
    parser.add_argument("--coarse-Nv", type=int, default=8192)
    parser.add_argument("--refined-Nx", type=int, default=1024)
    parser.add_argument("--refined-Nv", type=int, default=16384)
    parser.add_argument("--rollout-Nx", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    floors = np.empty((len(REGIMES), 4), dtype=np.float64)
    snapshot_rms = np.empty((len(REGIMES), 7, 4), dtype=np.float64)
    for regime_index, (regime, case_name) in enumerate(CASES):
        projected = []
        metadata_rows = []
        for nx, nv in (
            (args.coarse_Nx, args.coarse_Nv),
            (args.refined_Nx, args.refined_Nv),
        ):
            pair = args.root / f"Nx{nx}_Nv{nv}"
            metadata = json.loads((pair / "metadata.json").read_text())
            metadata_rows.append(metadata)
            projected.append(
                _project_low_moments(
                    pair / "cases" / f"{case_name}.npz",
                    metadata,
                    args.rollout_Nx,
                )
            )
        if metadata_rows[0]["snapshot_times"] != metadata_rows[1]["snapshot_times"]:
            raise ValueError("Physical-grid snapshots use different times")
        difference = projected[0] - projected[1]
        snapshot_rms[regime_index] = np.sqrt(np.mean(difference**2, axis=-1))
        floors[regime_index] = np.max(snapshot_rms[regime_index], axis=0)
        print(
            f"[floor] {regime}: "
            + ", ".join(f"{value:.6e}" for value in floors[regime_index])
        )

    metadata = {
        "diagnostic": "low_moment_physical_grid_rms_floor",
        "regimes": list(REGIMES),
        "channels": ["density_perturbation", "velocity", "pressure_perturbation", "electric_field"],
        "coarse_grid": [args.coarse_Nx, args.coarse_Nv],
        "refined_grid": [args.refined_Nx, args.refined_Nv],
        "rollout_Nx": args.rollout_Nx,
        "aggregation": "maximum snapshot RMS disagreement",
        "snapshot_times": metadata_rows[1]["snapshot_times"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        floor_rms=floors,
        snapshot_rms=snapshot_rms,
        metadata_json=np.array([json.dumps(metadata, sort_keys=True)]),
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
