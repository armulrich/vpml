"""Sweep nonlinear learned-closure rollout quality across deployment Nv values."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))

from vpml.jax_runtime import bootstrap_jax_runtime, print_jax_runtime_summary

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np

from model.eval import run_physical_landau_reference
from vpml.nonlinear_landau import (
    NonlinearLandauParams,
    _time_key,
    run_nonlinear_landau_rollout_raw,
)
from vpml.core import (
    LearnedInterfaceClosure,
    e_hat_history_from_a_hat_history,
    hermite_basis_phi,
    load_learned_interface_closure_npz,
)
from vpml.physical_grid import project_distribution_snapshot_to_fourier_hermite
from vpml.metrics import (
    EarlyElectricFieldGrowthMetric,
    EarlyGrowthConfig,
    FieldErrorConfig,
    SelfGeneratedFieldErrorMetric,
)
from vpml.visualization import (
    FieldSweepCase,
    GrowthSweepCase,
    plot_field_metric_sweep,
    plot_growth_metric_sweep,
    save_fig10_learned_comparison_nv_sweep_phase_space,
    save_figure,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _electric_energy_from_ehat_history(E_hat_hist: np.ndarray, *, Nx: int, Lx: float) -> np.ndarray:
    E_hat_hist = np.asarray(E_hat_hist, dtype=np.complex128)
    nk = E_hat_hist.shape[1]
    weights = np.full((nk,), 2.0, dtype=np.float64)
    weights[0] = 1.0
    if int(Nx) % 2 == 0 and nk >= 2:
        weights[-1] = 1.0
    sqrt_max = float(np.sqrt(np.finfo(np.float64).max))
    mag = np.abs(E_hat_hist)
    invalid = ~np.isfinite(mag)
    overflow = mag > sqrt_max
    safe_mag = np.where(invalid | overflow, 0.0, mag)
    weighted_sq = np.sum(weights[None, :] * safe_mag * safe_mag, axis=1, dtype=np.float64)
    weighted_sq = np.where(np.any(invalid | overflow, axis=1), np.inf, weighted_sq)
    return (0.5 * float(Lx) / float(Nx) ** 2 * weighted_sq).astype(np.float64)


def _json_scalar(value: float) -> float | str:
    value = float(value)
    if np.isfinite(value):
        return value
    return "inf" if value > 0.0 else ("-inf" if value < 0.0 else "nan")


def _phase_space_payload_from_raw(
    raw: Dict[str, np.ndarray | jnp.ndarray],
    params: NonlinearLandauParams,
) -> Dict[str, np.ndarray]:
    x = np.asarray(raw["x"], dtype=np.float64)
    v = np.linspace(params.v_range[0], params.v_range[1], int(params.Nv_plot), dtype=np.float64)
    phi = np.asarray(hermite_basis_phi(int(params.Nv), v), dtype=np.float64)
    snaps_phys = np.asarray(raw["snapshot_a_phys"], dtype=np.float64)
    m_eq = np.asarray(raw["m_eq"], dtype=np.float64)

    payload: Dict[str, np.ndarray] = {
        "x": x,
        "v": v,
        "times": np.asarray(params.snapshot_times, dtype=np.float64),
    }
    for idx, t in enumerate(params.snapshot_times):
        full_f = (snaps_phys[idx] + m_eq[:, None]).T @ phi
        payload[f"f_{_time_key(float(t))}"] = full_f.T.astype(np.float64)
    return payload


def _resample_periodic_x(f_vx: np.ndarray, *, Lx: float, target_nx: int) -> np.ndarray:
    f_vx = np.asarray(f_vx, dtype=np.float64)
    source_nx = int(f_vx.shape[1])
    target_nx = int(target_nx)
    if source_nx == target_nx:
        return f_vx.copy()
    x_source = np.linspace(0.0, float(Lx), source_nx, endpoint=False, dtype=np.float64)
    x_target = np.linspace(0.0, float(Lx), target_nx, endpoint=False, dtype=np.float64)
    x_ext = np.concatenate([x_source, np.asarray([float(Lx)], dtype=np.float64)])
    values_ext = np.concatenate([f_vx, f_vx[:, :1]], axis=1)
    out = np.empty((f_vx.shape[0], target_nx), dtype=np.float64)
    for j in range(f_vx.shape[0]):
        out[j] = np.interp(x_target, x_ext, values_ext[j])
    return out


def _resample_v(f_vx: np.ndarray, *, source_v: np.ndarray, target_v: np.ndarray) -> np.ndarray:
    f_vx = np.asarray(f_vx, dtype=np.float64)
    source_v = np.asarray(source_v, dtype=np.float64)
    target_v = np.asarray(target_v, dtype=np.float64)
    if f_vx.shape[0] != source_v.shape[0]:
        raise ValueError("source_v length must match the velocity dimension of f_vx")
    if source_v.shape == target_v.shape and np.array_equal(source_v, target_v):
        return f_vx.copy()
    out = np.empty((target_v.shape[0], f_vx.shape[1]), dtype=np.float64)
    for x_idx in range(f_vx.shape[1]):
        out[:, x_idx] = np.interp(target_v, source_v, f_vx[:, x_idx])
    return out


def _raw_hr_reference_phase_payload(
    hr_payload: Dict[str, np.ndarray],
    params: NonlinearLandauParams,
) -> Dict[str, np.ndarray]:
    """Use spline/grid HR snapshots directly, resampling only to the plot grid."""
    if "snapshot_f" not in hr_payload:
        raise ValueError("Raw HR phase payload requires HR reference snapshot_f")
    source_v = np.asarray(hr_payload["v"], dtype=np.float64)
    snapshot_times = np.asarray(hr_payload["snapshot_times"], dtype=np.float64)
    snapshot_f = np.asarray(hr_payload["snapshot_f"], dtype=np.float64)
    x = np.linspace(0.0, float(params.L), int(params.Nx), endpoint=False, dtype=np.float64)
    v_plot = np.linspace(params.v_range[0], params.v_range[1], int(params.Nv_plot), dtype=np.float64)
    payload: Dict[str, np.ndarray] = {
        "x": x,
        "v": v_plot,
        "times": np.asarray(params.snapshot_times, dtype=np.float64),
    }
    for t in params.snapshot_times:
        matches = np.flatnonzero(np.isclose(snapshot_times, float(t), rtol=0.0, atol=1e-10))
        if matches.size == 0:
            raise ValueError(f"HR reference is missing snapshot time t={float(t):g}")
        f_hr = snapshot_f[int(matches[0])]
        f_on_solver_x = _resample_periodic_x(f_hr, Lx=float(params.L), target_nx=int(params.Nx))
        payload[f"f_{_time_key(float(t))}"] = _resample_v(
            f_on_solver_x,
            source_v=source_v,
            target_v=v_plot,
        )
    return payload


def _projected_hr_reference_phase_payload(
    hr_payload: Dict[str, np.ndarray],
    params: NonlinearLandauParams,
    *,
    reference_Nv: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if "snapshot_f" not in hr_payload:
        raise ValueError("Projected HR phase payload requires HR reference snapshot_f")
    v_teacher = np.asarray(hr_payload["v"], dtype=np.float64)
    equilibrium = np.asarray(hr_payload["equilibrium"], dtype=np.float64)
    snapshot_times = np.asarray(hr_payload["snapshot_times"], dtype=np.float64)
    snapshot_f = np.asarray(hr_payload["snapshot_f"], dtype=np.float64)
    phase_Nv = int(params.Nv if reference_Nv is None else reference_Nv)
    if phase_Nv <= 0:
        raise ValueError(f"phase reference Nv must be positive, got {phase_Nv}")

    x = np.linspace(0.0, float(params.L), int(params.Nx), endpoint=False, dtype=np.float64)
    v_plot = np.linspace(params.v_range[0], params.v_range[1], int(params.Nv_plot), dtype=np.float64)
    phi = np.asarray(hermite_basis_phi(phase_Nv, v_plot), dtype=np.float64)
    m_eq = np.zeros((phase_Nv,), dtype=np.float64)
    m_eq[0] = 1.0

    payload: Dict[str, np.ndarray] = {
        "x": x,
        "v": v_plot,
        "times": np.asarray(params.snapshot_times, dtype=np.float64),
    }
    for t in params.snapshot_times:
        matches = np.flatnonzero(np.isclose(snapshot_times, float(t), rtol=0.0, atol=1e-10))
        if matches.size == 0:
            raise ValueError(f"HR reference is missing snapshot time t={float(t):g}")
        f_hr = snapshot_f[int(matches[0])]
        f_on_solver_x = _resample_periodic_x(f_hr, Lx=float(params.L), target_nx=int(params.Nx))
        a_hat = np.asarray(
            project_distribution_snapshot_to_fourier_hermite(
                jnp.asarray(f_on_solver_x, dtype=jnp.float64),
                jnp.asarray(v_teacher, dtype=jnp.float64),
                phase_Nv,
                vth=float(params.vth),
                equilibrium=jnp.asarray(equilibrium, dtype=jnp.float64),
            ),
            dtype=np.complex128,
        )
        a_phys = np.fft.irfft(a_hat, n=int(params.Nx), axis=1).real.astype(np.float64)
        full_f = (a_phys + m_eq[:, None]).T @ phi
        payload[f"f_{_time_key(float(t))}"] = full_f.T.astype(np.float64)
    return payload


def _learned_nonlinear_payload_from_raw(
    raw: Dict[str, np.ndarray | jnp.ndarray],
    params: NonlinearLandauParams,
) -> Dict[str, np.ndarray]:
    a_hat_hist = np.asarray(raw["a_hat_hist"], dtype=np.complex128)
    times = np.asarray(raw["a_hat_hist_times"], dtype=np.float64)
    k_arr = np.asarray(raw["k_arr"], dtype=np.float64)
    E_hat_hist = np.asarray(
        e_hat_history_from_a_hat_history(
            jnp.asarray(a_hat_hist, dtype=jnp.complex128),
            jnp.asarray(k_arr, dtype=jnp.float64),
            poisson_sign=float(params.poisson_sign),
        ),
        dtype=np.complex128,
    )
    return {
        "times": times,
        "E_hat_hist": E_hat_hist,
        "energy": _electric_energy_from_ehat_history(E_hat_hist, Nx=int(params.Nx), Lx=float(params.L)),
        "k_arr": k_arr,
    }


def _write_summary(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep nonlinear learned-closure metrics across deployment Nv")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("out_bench") / "nv_sweep")
    parser.add_argument("--nv-list", type=str, default="8,64,256,300,512")
    parser.add_argument("--Nx", type=int, default=200)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--T", type=float, default=40.0)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--k0", type=float, default=0.5)
    parser.add_argument("--snapshot-times", type=str, default="20.0,40.0,60.0")
    parser.add_argument("--Nv-plot", dest="nv_plot", type=int, default=1000)
    parser.add_argument("--phase-vmin", dest="phase_vmin", type=float, default=0.0)
    parser.add_argument("--phase-vmax", dest="phase_vmax", type=float, default=0.5)
    parser.add_argument("--phase-vrange", dest="phase_vrange", type=str, default="-4.0,4.0")
    parser.add_argument(
        "--phase-reference-Nv",
        dest="phase_reference_nv",
        type=int,
        default=None,
        help="Project the HR phase-space reference with this Hermite order instead of each deployment Nv.",
    )
    parser.add_argument(
        "--phase-reference-mode",
        choices=("projected", "raw_hr_grid"),
        default="projected",
        help="Use a Hermite-projected or direct spline/grid HR left Fig10 column.",
    )
    parser.add_argument(
        "--initial-perturbation-npy",
        type=Path,
        default=None,
        help="Optional 1D perturbation sampled on the reduced solver x grid.",
    )
    parser.add_argument(
        "--teacher-initial-perturbation-npy",
        type=Path,
        default=None,
        help="Optional 1D perturbation sampled on the HR teacher x grid.",
    )
    parser.add_argument(
        "--teacher-reference-npz",
        type=Path,
        default=None,
        help="Reuse a cached physical-teacher field history and selected snapshots.",
    )
    parser.add_argument(
        "--case-label",
        type=str,
        default=None,
        help="Optional label recorded in the per-case evaluation summary.",
    )
    parser.add_argument("--dealias-23", action="store_true")
    parser.add_argument("--nonlocal-mu", type=float, default=-1.017234)

    parser.add_argument("--teacher-Nx", dest="teacher_nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", dest="teacher_nv", type=int, default=512)
    parser.add_argument("--teacher-dt", dest="teacher_dt", type=float, default=1e-2)
    parser.add_argument("--teacher-vmin", dest="teacher_vmin", type=float, default=-8.0)
    parser.add_argument("--teacher-vmax", dest="teacher_vmax", type=float, default=8.0)

    parser.add_argument("--growth-sample-selector", type=str, default="all", choices=["all", "local_maxima"])
    parser.add_argument("--field-num-low-modes", type=int, default=None)
    parser.add_argument("--field-k-max", type=float, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    print_jax_runtime_summary(jax, context="nv-sweep")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    nv_list = parse_int_tuple(args.nv_list)
    if not nv_list:
        raise ValueError("At least one Nv value must be provided")
    if args.checkpoint is None and args.checkpoint_dir is None:
        raise ValueError("Either --checkpoint or --checkpoint-dir must be provided")
    if args.checkpoint is not None and args.checkpoint_dir is not None:
        raise ValueError("Provide only one of --checkpoint or --checkpoint-dir")
    snapshot_times = parse_float_tuple(args.snapshot_times)
    if len(snapshot_times) < 1:
        raise ValueError("snapshot-times must contain at least one time")
    phase_vrange = parse_float_tuple(args.phase_vrange)
    if len(phase_vrange) != 2:
        raise ValueError("phase-vrange must contain exactly two values")
    initial_perturbation_x: Optional[np.ndarray] = None
    if args.initial_perturbation_npy is not None:
        initial_perturbation_x = np.asarray(np.load(args.initial_perturbation_npy), dtype=np.float64)
        if initial_perturbation_x.shape != (int(args.Nx),):
            raise ValueError(
                "--initial-perturbation-npy must contain shape "
                f"({int(args.Nx)},), got {initial_perturbation_x.shape}"
            )
    initial_perturbation_teacher_x: Optional[np.ndarray] = None
    if args.teacher_initial_perturbation_npy is not None:
        initial_perturbation_teacher_x = np.asarray(
            np.load(args.teacher_initial_perturbation_npy), dtype=np.float64
        )
        if initial_perturbation_teacher_x.shape != (int(args.teacher_nx),):
            raise ValueError(
                "--teacher-initial-perturbation-npy must contain shape "
                f"({int(args.teacher_nx)},), got {initial_perturbation_teacher_x.shape}"
            )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    shared_checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir is not None else None
    checkpoint_map: Dict[int, Path] = {}
    if shared_checkpoint is not None:
        learned = load_learned_interface_closure_npz(shared_checkpoint)
        train_nv_targets = tuple(int(v) for v in learned.Nv_targets)
        train_nv_min = min(train_nv_targets)
        train_nv_max = max(train_nv_targets)
        checkpoint_map = {int(Nv): shared_checkpoint for Nv in nv_list}
        print(f"[nv-sweep] loaded shared checkpoint {shared_checkpoint} for evaluation")
    else:
        train_nv_targets = tuple(int(v) for v in nv_list)
        train_nv_min = min(train_nv_targets)
        train_nv_max = max(train_nv_targets)
        for Nv in nv_list:
            ckpt = checkpoint_dir / f"nv{int(Nv)}" / "interface_closure.npz"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Expected checkpoint for Nv={int(Nv)} at {ckpt}. "
                    "Train the per-Nv models first or pass --checkpoint for a shared model."
                )
            checkpoint_map[int(Nv)] = ckpt
        print(f"[nv-sweep] loaded per-Nv checkpoints from {checkpoint_dir} for evaluation")
    print(
        "[nv-sweep] preparing one shared HR reference "
        f"(teacher Nx={int(args.teacher_nx)}, Nv={int(args.teacher_nv)}, T={float(args.T):g}) "
        "and reusing it for every deployment Nv"
    )

    Lx = 4.0 * math.pi
    effective_field_k_max = args.field_k_max
    if bool(args.dealias_23) and effective_field_k_max is None and args.field_num_low_modes is None:
        effective_field_k_max = (2.0 * math.pi / Lx) * float(args.Nx // 3)

    growth_metric = EarlyElectricFieldGrowthMetric(
        EarlyGrowthConfig(sample_selector=args.growth_sample_selector)
    )
    field_metric = SelfGeneratedFieldErrorMetric(
        FieldErrorConfig(
            num_low_modes=args.field_num_low_modes,
            k_max=effective_field_k_max,
        )
    )

    x_hr = np.linspace(0.0, Lx, int(args.teacher_nx), endpoint=False, dtype=np.float64)
    if initial_perturbation_teacher_x is not None:
        perturb_hr = initial_perturbation_teacher_x
    elif initial_perturbation_x is None:
        perturb_hr = float(args.eps) * np.cos(float(args.k0) * x_hr)
    else:
        perturb_hr = _resample_periodic_x(
            initial_perturbation_x[None, :],
            Lx=Lx,
            target_nx=int(args.teacher_nx),
        )[0]
    if args.teacher_reference_npz is None:
        hr_payload = run_physical_landau_reference(
            Nx=int(args.teacher_nx),
            Nv=int(args.teacher_nv),
            Lx=Lx,
            vmin=float(args.teacher_vmin),
            vmax=float(args.teacher_vmax),
            dt=float(args.teacher_dt),
            T=float(args.T),
            perturbation_x=perturb_hr,
            snapshot_times=snapshot_times,
        )
    else:
        with np.load(args.teacher_reference_npz) as cached:
            required = {
                "x",
                "v",
                "k_arr",
                "perturbation_x",
                "equilibrium",
                "energy",
                "E_hat_hist_times",
                "E_hat_hist",
                "snapshot_times",
                "snapshot_f",
            }
            missing = sorted(required - set(cached.files))
            if missing:
                raise ValueError(
                    f"Cached teacher reference {args.teacher_reference_npz} "
                    f"is missing {missing!r}"
                )
            hr_payload = {
                "times": np.asarray(cached["E_hat_hist_times"], dtype=np.float64),
                "E_hat_hist": np.asarray(cached["E_hat_hist"], dtype=np.complex128),
                "energy": np.asarray(cached["energy"], dtype=np.float64),
                "k_arr": np.asarray(cached["k_arr"], dtype=np.float64),
                "x": np.asarray(cached["x"], dtype=np.float64),
                "v": np.asarray(cached["v"], dtype=np.float64),
                "perturbation_x": np.asarray(
                    cached["perturbation_x"], dtype=np.float64
                ),
                "equilibrium": np.asarray(cached["equilibrium"], dtype=np.float64),
                "snapshot_times": np.asarray(cached["snapshot_times"], dtype=np.float64),
                "snapshot_f": np.asarray(cached["snapshot_f"], dtype=np.float64),
            }
        if hr_payload["x"].shape != (int(args.teacher_nx),):
            raise ValueError("Cached teacher reference Nx does not match --teacher-Nx")
        if hr_payload["v"].shape != (int(args.teacher_nv),):
            raise ValueError("Cached teacher reference Nv does not match --teacher-Nv")
        if not np.isclose(hr_payload["times"][-1], float(args.T)):
            raise ValueError("Cached teacher reference final time does not match --T")
        cached_perturbation = np.asarray(
            hr_payload["perturbation_x"], dtype=np.float64
        )
        if cached_perturbation.shape != perturb_hr.shape or not np.allclose(
            cached_perturbation,
            perturb_hr,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Cached teacher reference initial condition does not match the "
                "requested teacher perturbation"
            )
        print(f"[nv-sweep] reused cached HR reference {args.teacher_reference_npz}")

    growth_cases: List[GrowthSweepCase] = []
    field_cases: List[FieldSweepCase] = []
    summary_cases: List[Dict[str, object]] = []
    phase_payload: Dict[str, np.ndarray] = {}
    row_labels: List[str] = []
    case_npz_dir = outdir / "cases"
    case_npz_dir.mkdir(parents=True, exist_ok=True)

    for Nv in nv_list:
        learned = load_learned_interface_closure_npz(checkpoint_map[int(Nv)])
        case_train_nv_targets = tuple(int(v) for v in learned.Nv_targets)
        case_train_nv_min = min(case_train_nv_targets)
        case_train_nv_max = max(case_train_nv_targets)
        print(f"[nv-sweep] running truncation/learned rollout pair for Nv={int(Nv)}")
        params = NonlinearLandauParams(
            Nx=int(args.Nx),
            Nv=int(Nv),
            L=Lx,
            dt=float(args.dt),
            T=float(args.T),
            eps=float(args.eps),
            k0=float(args.k0),
            dealias_23=bool(args.dealias_23),
            snapshot_times=tuple(float(v) for v in snapshot_times),
            v_range=(float(phase_vrange[0]), float(phase_vrange[1])),
            Nv_plot=int(args.nv_plot),
            vmin=float(args.phase_vmin),
            vmax=float(args.phase_vmax),
            initial_perturbation_x=initial_perturbation_x,
        )

        truncation_raw = run_nonlinear_landau_rollout_raw(
            params,
            "truncation",
            return_state_history=True,
            history_stride=1,
        )
        truncation_payload = _learned_nonlinear_payload_from_raw(truncation_raw, params)
        truncation_field_comparison = field_metric.prepare_fourier_comparison(
            truncation_payload["times"],
            truncation_payload["E_hat_hist"],
            truncation_payload["k_arr"],
            hr_payload["times"],
            hr_payload["E_hat_hist"],
            hr_payload["k_arr"],
        )
        truncation_field = field_metric.evaluate_fourier(
            truncation_payload["times"],
            truncation_payload["E_hat_hist"],
            truncation_payload["k_arr"],
            hr_payload["times"],
            hr_payload["E_hat_hist"],
            hr_payload["k_arr"],
        )

        learned_raw = run_nonlinear_landau_rollout_raw(
            params,
            "learned",
            learned_closure=learned,
            return_state_history=True,
            history_stride=1,
        )
        theta_payload = _learned_nonlinear_payload_from_raw(learned_raw, params)
        growth = growth_metric.compare(
            theta_payload["times"],
            theta_payload["energy"],
            hr_payload["times"],
            hr_payload["energy"],
        )
        field_comparison = field_metric.prepare_fourier_comparison(
            theta_payload["times"],
            theta_payload["E_hat_hist"],
            theta_payload["k_arr"],
            hr_payload["times"],
            hr_payload["E_hat_hist"],
            hr_payload["k_arr"],
        )
        field = field_metric.evaluate_fourier(
            theta_payload["times"],
            theta_payload["E_hat_hist"],
            theta_payload["k_arr"],
            hr_payload["times"],
            hr_payload["E_hat_hist"],
            hr_payload["k_arr"],
        )

        reference_phase = (
            _raw_hr_reference_phase_payload(hr_payload, params)
            if args.phase_reference_mode == "raw_hr_grid"
            else _projected_hr_reference_phase_payload(
                hr_payload,
                params,
                reference_Nv=args.phase_reference_nv,
            )
        )
        truncation_phase = _phase_space_payload_from_raw(truncation_raw, params)
        learned_phase = _phase_space_payload_from_raw(learned_raw, params)

        if not phase_payload:
            phase_payload["x"] = np.asarray(learned_phase["x"], dtype=np.float64)
            phase_payload["v"] = np.asarray(learned_phase["v"], dtype=np.float64)
            phase_payload["times"] = np.asarray(snapshot_times, dtype=np.float64)
        for t in snapshot_times:
            phase_payload[f"nv{int(Nv)}_reference_f_{_time_key(float(t))}"] = np.asarray(
                reference_phase[f"f_{_time_key(float(t))}"],
                dtype=np.float64,
            )
            phase_payload[f"nv{int(Nv)}_truncation_f_{_time_key(float(t))}"] = np.asarray(
                truncation_phase[f"f_{_time_key(float(t))}"],
                dtype=np.float64,
            )
            phase_payload[f"nv{int(Nv)}_learned_f_{_time_key(float(t))}"] = np.asarray(
                learned_phase[f"f_{_time_key(float(t))}"],
                dtype=np.float64,
            )

        in_training_targets = int(Nv) in case_train_nv_targets
        beyond_training_range = int(Nv) > int(case_train_nv_max) or int(Nv) < int(case_train_nv_min)
        label = rf"$N_v={int(Nv)}$"
        if beyond_training_range:
            label += " (out of train range)"
        elif not in_training_targets:
            label += " (unseen)"
        row_labels.append(label)

        growth_cases.append(
            GrowthSweepCase(
                Nv=int(Nv),
                times_theta=np.asarray(theta_payload["times"], dtype=np.float64),
                energy_theta=np.asarray(theta_payload["energy"], dtype=np.float64),
                comparison=growth,
                in_training_targets=in_training_targets,
                beyond_training_range=beyond_training_range,
            )
        )
        field_cases.append(
            FieldSweepCase(
                Nv=int(Nv),
                comparison=field_comparison,
                epsilon_E=float(field.epsilon_E),
                baseline_comparison=truncation_field_comparison,
                baseline_epsilon_E=float(truncation_field.epsilon_E),
                baseline_label="truncation",
                theta_label="learned",
                in_training_targets=in_training_targets,
                beyond_training_range=beyond_training_range,
            )
        )
        case_path = case_npz_dir / f"nv{int(Nv)}_nonlinear_sweep_case.npz"
        case_payload = {
            "times_hr": np.asarray(hr_payload["times"], dtype=np.float64),
            "times_theta": np.asarray(theta_payload["times"], dtype=np.float64),
            "energy_hr": np.asarray(hr_payload["energy"], dtype=np.float64),
            "energy_theta": np.asarray(theta_payload["energy"], dtype=np.float64),
            "E_hat_hr": np.asarray(hr_payload["E_hat_hist"], dtype=np.complex128),
            "E_hat_theta": np.asarray(theta_payload["E_hat_hist"], dtype=np.complex128),
            "k_hr": np.asarray(hr_payload["k_arr"], dtype=np.float64),
            "k_theta": np.asarray(theta_payload["k_arr"], dtype=np.float64),
            "field_times": np.asarray(field_comparison.times, dtype=np.float64),
            "field_selected_k": np.asarray(field_comparison.selected_k, dtype=np.float64),
            "field_E_hat_hr": np.asarray(field_comparison.E_hat_hr, dtype=np.complex128),
            "field_E_hat_theta": np.asarray(field_comparison.E_hat_theta, dtype=np.complex128),
            "field_E_hat_truncation": np.asarray(truncation_field_comparison.E_hat_theta, dtype=np.complex128),
            "epsilon_grow": np.array([growth.epsilon_grow], dtype=np.float64),
            "gamma_grow_hr": np.array([growth.gamma_grow_hr], dtype=np.float64),
            "gamma_grow_theta": np.array([growth.gamma_grow_theta], dtype=np.float64),
            "epsilon_E": np.array([field.epsilon_E], dtype=np.float64),
            "epsilon_E_truncation": np.array([truncation_field.epsilon_E], dtype=np.float64),
            "in_training_targets": np.array([int(in_training_targets)], dtype=np.int32),
            "beyond_training_range": np.array([int(beyond_training_range)], dtype=np.int32),
            "x": np.asarray(learned_phase["x"], dtype=np.float64),
            "v": np.asarray(learned_phase["v"], dtype=np.float64),
        }
        for snap_idx, snap_t in enumerate(snapshot_times):
            snap_key = _time_key(float(snap_t))
            for prefix, phase in (
                ("reference", reference_phase),
                ("truncation", truncation_phase),
                ("learned", learned_phase),
            ):
                case_payload[f"{prefix}_f_t{snap_idx}"] = np.asarray(
                    phase[f"f_{snap_key}"],
                    dtype=np.float64,
                )
                case_payload[f"{prefix}_f_{snap_key}"] = np.asarray(
                    phase[f"f_{snap_key}"],
                    dtype=np.float64,
                )
        np.savez(case_path, **case_payload)

        summary_cases.append(
            {
                "Nv": int(Nv),
                "checkpoint": str(checkpoint_map[int(Nv)]),
                "epsilon_grow": _json_scalar(growth.epsilon_grow),
                "gamma_grow_hr": _json_scalar(growth.gamma_grow_hr),
                "gamma_grow_theta": _json_scalar(growth.gamma_grow_theta),
                "epsilon_E": _json_scalar(field.epsilon_E),
                "epsilon_E_truncation": _json_scalar(truncation_field.epsilon_E),
                "fit_t_a": _json_scalar(growth.t_a),
                "fit_t_b": _json_scalar(growth.t_b),
                "num_common_modes": int(field.num_modes),
                "in_training_targets": bool(in_training_targets),
                "beyond_training_range": bool(beyond_training_range),
                "train_nv_targets": list(case_train_nv_targets),
                "case_npz": str(case_path),
            }
        )
        print(
            f"[nv-sweep] Nv={int(Nv)}: epsilon_grow={growth.epsilon_grow:.4e} "
            f"gamma_hr={growth.gamma_grow_hr:.4e} gamma_theta={growth.gamma_grow_theta:.4e} "
            f"epsilon_E_truncation={truncation_field.epsilon_E:.4e} "
            f"epsilon_E_learned={field.epsilon_E:.4e}"
        )
    growth_fig = plot_growth_metric_sweep(
        np.asarray(hr_payload["times"], dtype=np.float64),
        np.asarray(hr_payload["energy"], dtype=np.float64),
        growth_cases,
        title=r"Metric 1: HR reference vs learned FH ($q_\theta$)",
    )
    growth_png = save_figure(growth_fig, outdir / "nv_sweep_metric1.png", dpi=220)

    field_fig = plot_field_metric_sweep(
        field_cases,
        title=r"Nonlinear Landau Metric 2 sweep across deployment $N_v$",
    )
    field_png = save_figure(field_fig, outdir / "nv_sweep_metric2.png", dpi=220)

    phase_path = save_fig10_learned_comparison_nv_sweep_phase_space(
        phase_payload,
        nv_list=nv_list,
        times=snapshot_times,
        row_labels=row_labels,
        reference_title=(
            "HR Grid Reference"
            if args.phase_reference_mode == "raw_hr_grid"
            else "Projected HR Reference"
            if args.phase_reference_nv is None
            else rf"Projected HR Reference ($N_v^{{ref}}={int(args.phase_reference_nv)}$)"
        ),
        vmin=float(args.phase_vmin),
        vmax=float(args.phase_vmax),
        time_key_fn=_time_key,
        outdir=outdir,
    )
    phase_npz = outdir / "nv_sweep_phase_space_payload.npz"
    np.savez(phase_npz, **phase_payload)

    artifacts = {
        "metric1_png": str(growth_png),
        "metric2_png": str(field_png),
        "phase_space_png": str(phase_path),
        "phase_space_npz": str(phase_npz),
    }

    summary = {
        "checkpoint": None if shared_checkpoint is None else str(shared_checkpoint),
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
        "outdir": str(outdir),
        "nv_list": list(int(v) for v in nv_list),
        "train_nv_targets": list(int(v) for v in train_nv_targets),
        "train_nv_min": int(train_nv_min),
        "train_nv_max": int(train_nv_max),
        "teacher": {
            "Nx": int(args.teacher_nx),
            "Nv": int(args.teacher_nv),
            "dt": float(args.teacher_dt),
            "vmin": float(args.teacher_vmin),
            "vmax": float(args.teacher_vmax),
        },
        "phase_reference_Nv": None if args.phase_reference_nv is None else int(args.phase_reference_nv),
        "phase_reference_mode": str(args.phase_reference_mode),
        "case_label": args.case_label,
        "initial_perturbation_npy": (
            None if args.initial_perturbation_npy is None else str(args.initial_perturbation_npy)
        ),
        "teacher_initial_perturbation_npy": (
            None
            if args.teacher_initial_perturbation_npy is None
            else str(args.teacher_initial_perturbation_npy)
        ),
        "teacher_reference_npz": (
            None
            if args.teacher_reference_npz is None
            else str(args.teacher_reference_npz)
        ),
        "nonlinear_case": {
            "Nx": int(args.Nx),
            "dt": float(args.dt),
            "T": float(args.T),
            "eps": float(args.eps),
            "k0": float(args.k0),
            "snapshot_times": list(float(v) for v in snapshot_times),
            "dealias_23": bool(args.dealias_23),
            "nonlocal_mu": float(args.nonlocal_mu),
            "field_k_max": None if effective_field_k_max is None else float(effective_field_k_max),
        },
        "artifacts": artifacts,
        "cases": summary_cases,
    }
    _write_summary(outdir / "summary.json", summary)
    print(f"Saved Nv sweep summary to {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
