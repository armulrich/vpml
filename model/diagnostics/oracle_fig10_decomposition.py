#!/usr/bin/env python3
"""Separate low-order and tail-lift errors in a stage-2 Fig10 reconstruction.

The diagnostic writes new artifacts only. It never changes a training run,
checkpoint, cache, or sweep result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("VPML_JAX_BACKEND", "cpu")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from vpml.core import (
    learned_history_hermite_lift,
    load_learned_interface_closure_npz,
)
from vpml.nonlinear_landau import NonlinearLandauParams, _time_key, run_nonlinear_landau_rollout_raw
from model.diagnostics.phase_space import (
    FourierHermiteHistoryReader,
    phase_space_from_hermite_phys,
    resample_periodic_rows,
    select_nearest_case,
)


CHECKPOINT = Path()
HISTORY_CACHE = Path()
HISTORY_ARRAY = ""
HISTORY_READER: FourierHermiteHistoryReader | None = None

NX_TEACHER = 256
NX_DEPLOY = 200
NV_DEPLOY = 64
NV_REF = 512
LAGS = 16
DT = 0.005
DT_TEACHER = 0.01
T_FINAL = 60.0
TIMES = (20.0, 40.0, 60.0)
EPS = 0.5
K0 = 0.5
V_RANGE = (-4.0, 4.0)
NV_PLOT = 1000
Lx = 4.0 * math.pi


def _read_history_slice(case_idx: int, time_idx: int, n_min: int = 0, n_max: int = NV_REF) -> np.ndarray:
    if HISTORY_READER is None:
        raise RuntimeError("history reader is not configured")
    return HISTORY_READER.read_slice(case_idx, time_idx, n_min, n_max)


def _hr_phys_at(case_idx: int, time_idx: int) -> np.ndarray:
    a_hat = _read_history_slice(case_idx, time_idx, 0, NV_REF)
    a_phys_teacher = np.fft.irfft(a_hat, n=NX_TEACHER, axis=1).real.astype(np.float64)
    return resample_periodic_rows(a_phys_teacher, Lx=Lx, target_nx=NX_DEPLOY)


def _hr_low_history(case_idx: int, time_idx: int, lags: int) -> np.ndarray:
    hist = []
    for idx in range(int(time_idx) - int(lags), int(time_idx) + 1):
        idx = max(idx, 0)
        low_hat_teacher = _read_history_slice(case_idx, idx, 0, NV_DEPLOY)
        low_phys_teacher = np.fft.irfft(low_hat_teacher, n=NX_TEACHER, axis=1).real.astype(np.float64)
        low_phys_deploy = resample_periodic_rows(low_phys_teacher, Lx=Lx, target_nx=NX_DEPLOY)
        hist.append(np.fft.rfft(low_phys_deploy, axis=1).astype(np.complex128))
    return np.asarray(hist, dtype=np.complex128)


def _field_from_coeffs(a_phys: np.ndarray, v_plot: np.ndarray) -> np.ndarray:
    return phase_space_from_hermite_phys(a_phys, v_plot)


def _rel_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference.reshape(-1)))
    return float(np.linalg.norm((candidate - reference).reshape(-1)) / max(denom, 1e-30))


def _plot_variants(payload: Dict[str, np.ndarray], outdir: Path) -> Path:
    variant_labels = [
        ("learned_pred", rf"learned $C_{{<{NV_DEPLOY}}}$ + predicted tail"),
        ("learned_true", rf"learned $C_{{<{NV_DEPLOY}}}$ + true HR tail"),
        ("hr_pred", rf"HR $C_{{<{NV_DEPLOY}}}$ + predicted tail"),
        ("hr_true", rf"HR $C_{{<{NV_DEPLOY}}}$ + true HR tail"),
    ]
    times = tuple(float(t) for t in payload["times"])
    x = payload["x"]
    v = payload["v"]
    fig = plt.figure(figsize=(3.7 * len(times) + 3.0, 2.5 * len(variant_labels)), constrained_layout=True)
    grid = fig.add_gridspec(len(variant_labels), len(times), wspace=0.08, hspace=0.18)
    mesh = None
    for r, (variant, label) in enumerate(variant_labels):
        for c, t in enumerate(times):
            ax = fig.add_subplot(grid[r, c])
            key = f"{variant}_f_{_time_key(t)}"
            mesh = ax.pcolormesh(
                x,
                v,
                payload[key],
                shading="auto",
                cmap="Spectral_r",
                vmin=0.0,
                vmax=0.5,
            )
            if r == 0:
                ax.set_title(rf"$t={t:g}$")
            if c == 0:
                ax.set_ylabel(label + "\n" + r"$v$")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("x")
            ax.set_xticks([0.0, 2.0 * math.pi, 4.0 * math.pi], ["0", r"$2\pi$", r"$4\pi$"])
            ax.set_yticks([-4, -2, 0, 2, 4])
    if mesh is not None:
        fig.colorbar(mesh, ax=fig.axes, fraction=0.018, pad=0.01, label=r"$f(x,v,t)$")
    path = outdir / "fig10_oracle_decomposition.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _parse_float_tuple(text: str) -> Tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated float")
    return values


def _parse_v_range(text: str) -> Tuple[float, float]:
    values = _parse_float_tuple(text)
    if len(values) != 2 or values[0] >= values[1]:
        raise ValueError(f"--v-range must be min,max with min < max, got {text!r}")
    return values[0], values[1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose whether a stage-2 Fig10 error comes from the low-order rollout or predicted tail."
    )
    parser.add_argument("--run-root", type=Path, default=None, help="Stage-2 parent run directory for convenience.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--history-cache", type=Path, default=None)
    parser.add_argument("--history-array", type=str, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--Nx", type=int, default=200)
    parser.add_argument("--deploy-Nv", type=int, default=None)
    parser.add_argument("--teacher-Nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", type=int, default=512)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--teacher-dt", type=float, default=0.01)
    parser.add_argument("--T", type=float, default=60.0)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--k0", type=float, default=0.5)
    parser.add_argument("--snapshot-times", type=str, default="20.0,40.0,60.0")
    parser.add_argument("--history-lags", type=int, default=None)
    parser.add_argument("--v-range", type=str, default="-4.0,4.0")
    parser.add_argument("--Nv-plot", type=int, default=1000)
    return parser


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    run_root = None if args.run_root is None else Path(args.run_root)
    checkpoint = None if args.checkpoint is None else Path(args.checkpoint)
    cache = None if args.history_cache is None else Path(args.history_cache)
    outdir = None if args.outdir is None else Path(args.outdir)
    if run_root is not None:
        stage2_root = run_root / "stage2_history_lift"
        checkpoint = checkpoint or stage2_root / "models/nv64/interface_closure.npz"
        cache = cache or stage2_root / "models/nv64/interface_closure_exact_q_rollout_histories.npz"
        outdir = outdir or run_root / "oracle_fig10_decomposition"
    if checkpoint is None or cache is None or outdir is None:
        raise ValueError("pass --run-root or all of --checkpoint, --history-cache, and --outdir")
    return checkpoint, cache, outdir


def main(argv: Tuple[str, ...] | None = None) -> None:
    global CHECKPOINT, HISTORY_CACHE, HISTORY_ARRAY, HISTORY_READER
    global NX_TEACHER, NX_DEPLOY, NV_DEPLOY, NV_REF, LAGS
    global DT, DT_TEACHER, T_FINAL, TIMES, EPS, K0, V_RANGE, NV_PLOT

    args = build_arg_parser().parse_args(argv)
    CHECKPOINT, HISTORY_CACHE, outdir = _resolve_paths(args)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if not HISTORY_CACHE.exists():
        raise FileNotFoundError(HISTORY_CACHE)
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Diagnostic output exists: {outdir}; pass --overwrite to replace only this diagnostic")
    outdir.mkdir(parents=True, exist_ok=True)

    learned = load_learned_interface_closure_npz(CHECKPOINT)
    if not bool(learned.tail_history_lift_enabled):
        raise ValueError("oracle decomposition requires a checkpoint with tail-history lift enabled")
    NX_TEACHER = int(args.teacher_Nx)
    NX_DEPLOY = int(args.Nx)
    NV_DEPLOY = int(args.deploy_Nv or learned.tail_history_n_min)
    NV_REF = int(learned.tail_history_n_max)
    if NV_REF <= NV_DEPLOY:
        raise ValueError(f"tail-history range must be nonempty, got [{NV_DEPLOY},{NV_REF})")
    LAGS = int(learned.tail_history_lags if args.history_lags is None else args.history_lags)
    if LAGS < 0:
        raise ValueError("--history-lags must be nonnegative")
    DT = float(args.dt)
    DT_TEACHER = float(args.teacher_dt)
    T_FINAL = float(args.T)
    TIMES = _parse_float_tuple(args.snapshot_times)
    EPS = float(args.eps)
    K0 = float(args.k0)
    V_RANGE = _parse_v_range(args.v_range)
    NV_PLOT = int(args.Nv_plot)
    HISTORY_ARRAY = args.history_array or f"nonlinear_landau_strong_a_hat_ref_order{NV_REF}.npy"
    HISTORY_READER = FourierHermiteHistoryReader(HISTORY_CACHE, HISTORY_ARRAY)
    if NX_TEACHER <= 0 or NX_DEPLOY <= 0 or NV_PLOT <= 1:
        raise ValueError("Nx, teacher-Nx, and Nv-plot must be positive")
    if DT <= 0.0 or DT_TEACHER <= 0.0 or T_FINAL <= 0.0:
        raise ValueError("dt, teacher-dt, and T must be positive")

    params = NonlinearLandauParams(
        Nx=NX_DEPLOY,
        Nv=NV_DEPLOY,
        L=Lx,
        dt=DT,
        T=T_FINAL,
        eps=EPS,
        k0=K0,
        dealias_23=True,
        snapshot_times=TIMES,
        v_range=V_RANGE,
        Nv_plot=NV_PLOT,
        vmin=0.0,
        vmax=0.5,
    )
    print(f"[oracle] running learned Nv={NV_DEPLOY} rollout once to recover low state and predicted tail snapshots")
    learned_raw = run_nonlinear_landau_rollout_raw(
        params,
        "learned",
        learned_closure=learned,
        return_state_history=False,
    )
    learned_low = np.asarray(learned_raw["snapshot_a_phys"], dtype=np.float64)
    learned_pred = np.asarray(learned_raw["snapshot_recon_a_phys"], dtype=np.float64)
    k_arr = np.asarray(learned_raw["k_arr"], dtype=np.float64)

    case_idx, case_eps = select_nearest_case(HISTORY_CACHE, eps=EPS)
    print(f"[oracle] using nonlinear_landau_strong case {case_idx} with eps={case_eps:g}")

    v_plot = np.linspace(V_RANGE[0], V_RANGE[1], NV_PLOT, dtype=np.float64)
    x = np.linspace(0.0, Lx, NX_DEPLOY, endpoint=False, dtype=np.float64)
    payload: Dict[str, np.ndarray] = {"x": x, "v": v_plot, "times": np.asarray(TIMES, dtype=np.float64)}
    summary: Dict[str, object] = {
        "checkpoint": str(CHECKPOINT),
        "history_cache": str(HISTORY_CACHE),
        "case_idx": case_idx,
        "case_eps": case_eps,
        "history_lags": int(LAGS),
        "times": list(TIMES),
        "relative_l2_vs_hr_true": {},
    }

    for snap_idx, t in enumerate(TIMES):
        time_idx = int(round(float(t) / DT_TEACHER))
        hr_phys = _hr_phys_at(case_idx, time_idx)
        hr_low_hist = _hr_low_history(case_idx, time_idx, LAGS)
        hr_pred_hat = np.asarray(
            learned_history_hermite_lift(
                jnp.asarray(hr_low_hist, dtype=jnp.complex128),
                jnp.asarray(k_arr, dtype=jnp.float64),
                learned,
                n_min=NV_DEPLOY,
                n_max=NV_REF,
            ),
            dtype=np.complex128,
        )
        hr_pred_phys = np.fft.irfft(hr_pred_hat, n=NX_DEPLOY, axis=1).real.astype(np.float64)

        learned_true_phys = np.zeros((NV_REF, NX_DEPLOY), dtype=np.float64)
        learned_true_phys[:NV_DEPLOY] = learned_low[snap_idx]
        learned_true_phys[NV_DEPLOY:NV_REF] = hr_phys[NV_DEPLOY:NV_REF]

        hr_pred_mix_phys = np.zeros((NV_REF, NX_DEPLOY), dtype=np.float64)
        hr_pred_mix_phys[:NV_DEPLOY] = hr_phys[:NV_DEPLOY]
        hr_pred_mix_phys[NV_DEPLOY:NV_REF] = hr_pred_phys[NV_DEPLOY:NV_REF]

        fields = {
            "learned_pred": _field_from_coeffs(learned_pred[snap_idx], v_plot),
            "learned_true": _field_from_coeffs(learned_true_phys, v_plot),
            "hr_pred": _field_from_coeffs(hr_pred_mix_phys, v_plot),
            "hr_true": _field_from_coeffs(hr_phys, v_plot),
        }
        key = _time_key(t)
        for name, field in fields.items():
            payload[f"{name}_f_{key}"] = field
        summary["relative_l2_vs_hr_true"][str(t)] = {
            name: _rel_l2(field, fields["hr_true"]) for name, field in fields.items()
        }
        print(f"[oracle] t={t:g} rel-L2:", summary["relative_l2_vs_hr_true"][str(t)])

    fig_path = _plot_variants(payload, outdir)
    np.savez(outdir / "oracle_decomposition_payload.npz", **payload)
    with (outdir / "oracle_decomposition_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[oracle] wrote {fig_path}")
    print(f"[oracle] wrote {outdir / 'oracle_decomposition_summary.json'}")


if __name__ == "__main__":
    main()
