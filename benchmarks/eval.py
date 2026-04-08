"""Post-train a posteriori evaluation for learned interface closures."""

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

from benchmarks.fh_benchmarks_2412_07073_jax import (
    NonlinearLandauParams,
    run_nonlinear_landau_rollout_raw,
)
from vpml.core import (
    LearnedInterfaceClosure,
    e_hat_history_from_a_hat_history,
    load_learned_interface_closure_npz,
)
from vpml.linear_landau import (
    LinearLandauConfig,
    run_linear_landau_rollout,
)
from vpml.metrics import (
    EarlyElectricFieldGrowthMetric,
    EarlyGrowthConfig,
    FieldErrorConfig,
    SelfGeneratedFieldErrorMetric,
)
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    compute_electric_field_from_distribution,
    gaussian_pdf,
    normalize_density_on_grid,
    run_semilagrangian_vlasov_poisson,
)
from vpml.visualization.metrics import (
    plot_field_metric,
    plot_growth_metric,
    plot_metric_summary,
)
from vpml.visualization.common import save_figure

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass

BUNDLE_HELDOUT = "heldout_landau"
BUNDLE_BENCHMARK = "benchmark_rollouts"
ALL_BUNDLES = (BUNDLE_HELDOUT, BUNDLE_BENCHMARK)


def parse_str_tuple(text: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _k_arr_for_grid(Nx: int, Lx: float) -> np.ndarray:
    dx = float(Lx) / float(Nx)
    return np.asarray(2.0 * math.pi * np.fft.rfftfreq(int(Nx), d=dx), dtype=np.float64)


def _electric_energy_from_ehat_history(E_hat_hist: np.ndarray, *, Nx: int, Lx: float) -> np.ndarray:
    e_phys = np.fft.irfft(np.asarray(E_hat_hist, dtype=np.complex128), n=int(Nx), axis=1).astype(np.float64)
    dx = float(Lx) / float(Nx)
    return (0.5 * dx * np.sum(e_phys * e_phys, axis=1)).astype(np.float64)


def _maxwellian_equilibrium(v: jnp.ndarray) -> jnp.ndarray:
    return normalize_density_on_grid(gaussian_pdf(v, mean=0.0, sigma=1.0), v)


def _physical_grid_ehat_projector(config: PhysicalGridVlasovPoissonConfig):
    def projector(f_state):
        E_phys = compute_electric_field_from_distribution(f_state, config)
        return jnp.fft.rfft(E_phys).astype(jnp.complex128)

    return projector


def run_physical_landau_reference(
    *,
    Nx: int,
    Nv: int,
    Lx: float,
    vmin: float,
    vmax: float,
    dt: float,
    T: float,
    perturbation_x: np.ndarray,
    poisson_sign: float = +1.0,
) -> Dict[str, np.ndarray]:
    config = PhysicalGridVlasovPoissonConfig(
        Nx=int(Nx),
        Nv=int(Nv),
        Lx=float(Lx),
        vmin=float(vmin),
        vmax=float(vmax),
        dt=float(dt),
        T=float(T),
        poisson_sign=float(poisson_sign),
        snapshot_times=(),
    )
    equilibrium = _maxwellian_equilibrium(config.v)
    perturb = jnp.asarray(perturbation_x, dtype=jnp.float64)
    f0 = equilibrium[:, None] * (1.0 + perturb[None, :])
    raw = run_semilagrangian_vlasov_poisson(
        config,
        f0,
        history_stride=1,
        return_state_history=True,
        history_projector=_physical_grid_ehat_projector(config),
    )
    return {
        "times": np.asarray(raw["state_history_times"], dtype=np.float64),
        "E_hat_hist": np.asarray(raw["state_history"], dtype=np.complex128),
        "energy": np.asarray(raw["energy"], dtype=np.float64),
        "k_arr": np.asarray(raw["k_arr"], dtype=np.float64),
    }


def run_learned_linear_case(
    config: LinearLandauConfig,
    learned_closure: LearnedInterfaceClosure,
) -> Dict[str, np.ndarray]:
    payload = run_linear_landau_rollout(
        config,
        learned_closure=learned_closure,
        solver_backend="cnab2",
        return_state_history=False,
    )
    E_hat_hist = np.asarray(payload["E_hat_hist"], dtype=np.complex128)
    return {
        "times": np.asarray(payload["times"], dtype=np.float64),
        "E_hat_hist": E_hat_hist,
        "energy": _electric_energy_from_ehat_history(E_hat_hist, Nx=int(config.Nx), Lx=float(config.L)),
        "k_arr": _k_arr_for_grid(int(config.Nx), float(config.L)),
    }


def run_learned_nonlinear_case(
    params: NonlinearLandauParams,
    learned_closure: LearnedInterfaceClosure,
) -> Dict[str, np.ndarray]:
    raw = run_nonlinear_landau_rollout_raw(
        params,
        "learned",
        learned_closure=learned_closure,
        return_state_history=True,
        history_stride=1,
    )
    a_hat_hist = np.asarray(raw["a_hat_hist"], dtype=np.complex128)
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
        "times": np.asarray(raw["a_hat_hist_times"], dtype=np.float64),
        "E_hat_hist": E_hat_hist,
        "energy": _electric_energy_from_ehat_history(E_hat_hist, Nx=int(params.Nx), Lx=float(params.L)),
        "k_arr": k_arr,
    }


def evaluate_case(
    *,
    case_name: str,
    outdir: Path,
    theta_payload: Dict[str, np.ndarray],
    hr_payload: Dict[str, np.ndarray],
    growth_metric: EarlyElectricFieldGrowthMetric,
    field_metric: SelfGeneratedFieldErrorMetric,
) -> Dict[str, object]:
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

    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / case_name
    np.savez(
        base.with_suffix(".npz"),
        times_hr=np.asarray(hr_payload["times"], dtype=np.float64),
        times_theta=np.asarray(theta_payload["times"], dtype=np.float64),
        energy_hr=np.asarray(hr_payload["energy"], dtype=np.float64),
        energy_theta=np.asarray(theta_payload["energy"], dtype=np.float64),
        E_hat_hr=np.asarray(hr_payload["E_hat_hist"], dtype=np.complex128),
        E_hat_theta=np.asarray(theta_payload["E_hat_hist"], dtype=np.complex128),
        k_hr=np.asarray(hr_payload["k_arr"], dtype=np.float64),
        k_theta=np.asarray(theta_payload["k_arr"], dtype=np.float64),
        epsilon_grow=np.array([growth.epsilon_grow], dtype=np.float64),
        gamma_grow_hr=np.array([growth.gamma_grow_hr], dtype=np.float64),
        gamma_grow_theta=np.array([growth.gamma_grow_theta], dtype=np.float64),
        epsilon_E=np.array([field.epsilon_E], dtype=np.float64),
        fit_t_a=np.array([growth.t_a], dtype=np.float64),
        fit_t_b=np.array([growth.t_b], dtype=np.float64),
        fit_hr_intercept=np.array([growth.fit_hr.intercept], dtype=np.float64),
        fit_theta_intercept=np.array([growth.fit_theta.intercept], dtype=np.float64),
        fit_hr_num_samples=np.array([growth.fit_hr.num_samples], dtype=np.int32),
        fit_theta_num_samples=np.array([growth.fit_theta.num_samples], dtype=np.int32),
        field_times=np.asarray(field_comparison.times, dtype=np.float64),
        field_selected_k=np.asarray(field_comparison.selected_k, dtype=np.float64),
        field_E_hat_hr=np.asarray(field_comparison.E_hat_hr, dtype=np.complex128),
        field_E_hat_theta=np.asarray(field_comparison.E_hat_theta, dtype=np.complex128),
    )

    growth_fig = plot_growth_metric(
        hr_payload["times"],
        hr_payload["energy"],
        theta_payload["times"],
        theta_payload["energy"],
        growth,
        title=f"{case_name} — Metric 1",
    )
    save_figure(growth_fig, base.with_name(f"{case_name}_metric1.png"), dpi=220)

    field_fig = plot_field_metric(
        field_comparison,
        title=f"{case_name} — Metric 2",
    )
    save_figure(field_fig, base.with_name(f"{case_name}_metric2.png"), dpi=220)

    summary_fig = plot_metric_summary(
        hr_payload["times"],
        hr_payload["energy"],
        theta_payload["times"],
        theta_payload["energy"],
        growth,
        field_comparison,
        title=f"{case_name} — Learned Closure Evaluation",
        epsilon_E=field.epsilon_E,
    )
    save_figure(summary_fig, base.with_name(f"{case_name}_summary.png"), dpi=220)

    summary = {
        "case_name": case_name,
        "epsilon_grow": float(growth.epsilon_grow),
        "gamma_grow_hr": float(growth.gamma_grow_hr),
        "gamma_grow_theta": float(growth.gamma_grow_theta),
        "epsilon_E": float(field.epsilon_E),
        "fit_t_a": float(growth.t_a),
        "fit_t_b": float(growth.t_b),
        "num_common_modes": int(field.num_modes),
        "npz": str(base.with_suffix(".npz")),
        "metric1_png": str(base.with_name(f"{case_name}_metric1.png")),
        "metric2_png": str(base.with_name(f"{case_name}_metric2.png")),
        "summary_png": str(base.with_name(f"{case_name}_summary.png")),
    }
    print(
        f"[eval] {case_name}: epsilon_grow={summary['epsilon_grow']:.4e} "
        f"gamma_hr={summary['gamma_grow_hr']:.4e} gamma_theta={summary['gamma_grow_theta']:.4e} "
        f"epsilon_E={summary['epsilon_E']:.4e}"
    )
    return summary


def run_heldout_landau_bundle(
    args: argparse.Namespace,
    learned_closure: LearnedInterfaceClosure,
    outdir: Path,
    growth_metric: EarlyElectricFieldGrowthMetric,
    field_metric: SelfGeneratedFieldErrorMetric,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    Lx = 4.0 * math.pi

    linear_cfg = LinearLandauConfig(
        method="learned",
        Nx=int(args.heldout_linear_nx),
        Nv=int(args.heldout_linear_nv),
        L=Lx,
        dt=float(args.heldout_linear_dt),
        T=float(args.heldout_linear_t),
        eps=float(args.heldout_linear_eps),
        modes=(float(args.heldout_linear_mode),),
    )
    x_linear = np.linspace(0.0, Lx, int(args.teacher_nx), endpoint=False, dtype=np.float64)
    perturb_linear_hr = float(args.heldout_linear_eps) * np.cos(float(args.heldout_linear_mode) * x_linear)
    hr_linear = run_physical_landau_reference(
        Nx=args.teacher_nx,
        Nv=args.teacher_nv,
        Lx=Lx,
        vmin=args.teacher_vmin,
        vmax=args.teacher_vmax,
        dt=args.teacher_dt,
        T=args.heldout_linear_t,
        perturbation_x=perturb_linear_hr,
    )
    theta_linear = run_learned_linear_case(linear_cfg, learned_closure)
    results.append(
        evaluate_case(
            case_name="linear_landau_heldout",
            outdir=outdir,
            theta_payload=theta_linear,
            hr_payload=hr_linear,
            growth_metric=growth_metric,
            field_metric=field_metric,
        )
    )

    for case_name, eps in [
        ("nonlinear_landau_weak_heldout", float(args.heldout_weak_eps)),
        ("nonlinear_landau_strong_heldout", float(args.heldout_strong_eps)),
    ]:
        params = NonlinearLandauParams(
            Nx=int(args.heldout_nonlinear_nx),
            Nv=int(args.heldout_nonlinear_nv),
            L=Lx,
            dt=float(args.heldout_nonlinear_dt),
            T=float(args.heldout_nonlinear_t),
            eps=float(eps),
            k0=float(args.heldout_k0),
            snapshot_times=(),
        )
        x_hr = np.linspace(0.0, Lx, int(args.teacher_nx), endpoint=False, dtype=np.float64)
        perturb_hr = float(eps) * np.cos(float(args.heldout_k0) * x_hr)
        hr_payload = run_physical_landau_reference(
            Nx=args.teacher_nx,
            Nv=args.teacher_nv,
            Lx=Lx,
            vmin=args.teacher_vmin,
            vmax=args.teacher_vmax,
            dt=args.teacher_dt,
            T=args.heldout_nonlinear_t,
            perturbation_x=perturb_hr,
        )
        theta_payload = run_learned_nonlinear_case(params, learned_closure)
        results.append(
            evaluate_case(
                case_name=case_name,
                outdir=outdir,
                theta_payload=theta_payload,
                hr_payload=hr_payload,
                growth_metric=growth_metric,
                field_metric=field_metric,
            )
        )
    return results


def run_benchmark_rollout_bundle(
    args: argparse.Namespace,
    learned_closure: LearnedInterfaceClosure,
    outdir: Path,
    growth_metric: EarlyElectricFieldGrowthMetric,
    field_metric: SelfGeneratedFieldErrorMetric,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    Lx = 4.0 * math.pi

    linear_cfg = LinearLandauConfig(
        method="learned",
        Nx=int(args.benchmark_linear_nx),
        Nv=int(args.benchmark_linear_nv),
        L=Lx,
        dt=float(args.benchmark_linear_dt),
        T=float(args.benchmark_linear_t),
        eps=float(args.benchmark_linear_eps),
        modes=tuple(float(v) for v in parse_float_tuple(args.benchmark_linear_modes)),
    )
    x_linear_hr = np.linspace(0.0, Lx, int(args.teacher_nx), endpoint=False, dtype=np.float64)
    perturb_linear_hr = np.zeros_like(x_linear_hr)
    for mode in linear_cfg.modes:
        perturb_linear_hr = perturb_linear_hr + np.cos(float(mode) * x_linear_hr)
    perturb_linear_hr = float(linear_cfg.eps) * perturb_linear_hr
    hr_linear = run_physical_landau_reference(
        Nx=args.teacher_nx,
        Nv=args.teacher_nv,
        Lx=Lx,
        vmin=args.teacher_vmin,
        vmax=args.teacher_vmax,
        dt=args.teacher_dt,
        T=args.benchmark_linear_t,
        perturbation_x=perturb_linear_hr,
    )
    theta_linear = run_learned_linear_case(linear_cfg, learned_closure)
    results.append(
        evaluate_case(
            case_name="linear_landau_learned_benchmark",
            outdir=outdir,
            theta_payload=theta_linear,
            hr_payload=hr_linear,
            growth_metric=growth_metric,
            field_metric=field_metric,
        )
    )

    nonlinear_params = NonlinearLandauParams(
        Nx=int(args.benchmark_nonlinear_nx),
        Nv=int(args.benchmark_nonlinear_nv),
        L=Lx,
        dt=float(args.benchmark_nonlinear_dt),
        T=float(args.benchmark_nonlinear_t),
        eps=float(args.benchmark_nonlinear_eps),
        k0=float(args.benchmark_nonlinear_k0),
        snapshot_times=(),
    )
    x_nl_hr = np.linspace(0.0, Lx, int(args.teacher_nx), endpoint=False, dtype=np.float64)
    perturb_nl_hr = float(nonlinear_params.eps) * np.cos(float(nonlinear_params.k0) * x_nl_hr)
    hr_nonlinear = run_physical_landau_reference(
        Nx=args.teacher_nx,
        Nv=args.teacher_nv,
        Lx=Lx,
        vmin=args.teacher_vmin,
        vmax=args.teacher_vmax,
        dt=args.teacher_dt,
        T=args.benchmark_nonlinear_t,
        perturbation_x=perturb_nl_hr,
    )
    theta_nonlinear = run_learned_nonlinear_case(nonlinear_params, learned_closure)
    results.append(
        evaluate_case(
            case_name="fig10_nonlinear_landau_learned_benchmark",
            outdir=outdir,
            theta_payload=theta_nonlinear,
            hr_payload=hr_nonlinear,
            growth_metric=growth_metric,
            field_metric=field_metric,
        )
    )
    return results


def write_bundle_summary(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate learned Vlasov-Poisson closure metrics after training")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("out_bench") / "eval")
    parser.add_argument("--bundles", type=str, default="heldout_landau,benchmark_rollouts")
    parser.add_argument("--growth-sample-selector", type=str, default="all", choices=["all", "local_maxima"])
    parser.add_argument("--field-num-low-modes", type=int, default=None)
    parser.add_argument("--field-k-max", type=float, default=None)

    parser.add_argument("--teacher-Nx", dest="teacher_nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", dest="teacher_nv", type=int, default=512)
    parser.add_argument("--teacher-dt", dest="teacher_dt", type=float, default=1e-2)
    parser.add_argument("--teacher-vmin", dest="teacher_vmin", type=float, default=-8.0)
    parser.add_argument("--teacher-vmax", dest="teacher_vmax", type=float, default=8.0)

    parser.add_argument("--heldout-linear-Nx", dest="heldout_linear_nx", type=int, default=32)
    parser.add_argument("--heldout-linear-Nv", dest="heldout_linear_nv", type=int, default=20)
    parser.add_argument("--heldout-linear-dt", dest="heldout_linear_dt", type=float, default=1e-2)
    parser.add_argument("--heldout-linear-T", dest="heldout_linear_t", type=float, default=20.0)
    parser.add_argument("--heldout-linear-eps", dest="heldout_linear_eps", type=float, default=7.5e-3)
    parser.add_argument("--heldout-linear-mode", dest="heldout_linear_mode", type=float, default=1.0)

    parser.add_argument("--heldout-nonlinear-Nx", dest="heldout_nonlinear_nx", type=int, default=64)
    parser.add_argument("--heldout-nonlinear-Nv", dest="heldout_nonlinear_nv", type=int, default=20)
    parser.add_argument("--heldout-nonlinear-dt", dest="heldout_nonlinear_dt", type=float, default=1e-2)
    parser.add_argument("--heldout-nonlinear-T", dest="heldout_nonlinear_t", type=float, default=20.0)
    parser.add_argument("--heldout-k0", dest="heldout_k0", type=float, default=0.5)
    parser.add_argument("--heldout-weak-eps", dest="heldout_weak_eps", type=float, default=0.075)
    parser.add_argument("--heldout-strong-eps", dest="heldout_strong_eps", type=float, default=0.375)

    parser.add_argument("--benchmark-linear-Nx", dest="benchmark_linear_nx", type=int, default=10)
    parser.add_argument("--benchmark-linear-Nv", dest="benchmark_linear_nv", type=int, default=20)
    parser.add_argument("--benchmark-linear-dt", dest="benchmark_linear_dt", type=float, default=1e-2)
    parser.add_argument("--benchmark-linear-T", dest="benchmark_linear_t", type=float, default=40.0)
    parser.add_argument("--benchmark-linear-eps", dest="benchmark_linear_eps", type=float, default=1e-2)
    parser.add_argument("--benchmark-linear-modes", dest="benchmark_linear_modes", type=str, default="0.5,1.5")

    parser.add_argument("--benchmark-nonlinear-Nx", dest="benchmark_nonlinear_nx", type=int, default=200)
    parser.add_argument("--benchmark-nonlinear-Nv", dest="benchmark_nonlinear_nv", type=int, default=300)
    parser.add_argument("--benchmark-nonlinear-dt", dest="benchmark_nonlinear_dt", type=float, default=1e-2)
    parser.add_argument("--benchmark-nonlinear-T", dest="benchmark_nonlinear_t", type=float, default=40.0)
    parser.add_argument("--benchmark-nonlinear-eps", dest="benchmark_nonlinear_eps", type=float, default=0.5)
    parser.add_argument("--benchmark-nonlinear-k0", dest="benchmark_nonlinear_k0", type=float, default=0.5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    print_jax_runtime_summary(jax, context="eval")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    bundles = tuple(bundle for bundle in parse_str_tuple(args.bundles) if bundle in ALL_BUNDLES)
    if not bundles:
        raise ValueError("At least one valid evaluation bundle must be selected")

    learned_closure = load_learned_interface_closure_npz(args.checkpoint)
    growth_metric = EarlyElectricFieldGrowthMetric(
        EarlyGrowthConfig(sample_selector=args.growth_sample_selector)
    )
    field_metric = SelfGeneratedFieldErrorMetric(
        FieldErrorConfig(
            num_low_modes=args.field_num_low_modes,
            k_max=args.field_k_max,
        )
    )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    root_summary: Dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "outdir": str(outdir),
        "bundles": {},
    }

    if BUNDLE_HELDOUT in bundles:
        bundle_dir = outdir / BUNDLE_HELDOUT
        results = run_heldout_landau_bundle(
            args,
            learned_closure,
            bundle_dir,
            growth_metric,
            field_metric,
        )
        bundle_summary = {"bundle": BUNDLE_HELDOUT, "cases": results}
        write_bundle_summary(bundle_dir / "summary.json", bundle_summary)
        root_summary["bundles"][BUNDLE_HELDOUT] = bundle_summary

    if BUNDLE_BENCHMARK in bundles:
        bundle_dir = outdir / BUNDLE_BENCHMARK
        results = run_benchmark_rollout_bundle(
            args,
            learned_closure,
            bundle_dir,
            growth_metric,
            field_metric,
        )
        bundle_summary = {"bundle": BUNDLE_BENCHMARK, "cases": results}
        write_bundle_summary(bundle_dir / "summary.json", bundle_summary)
        root_summary["bundles"][BUNDLE_BENCHMARK] = bundle_summary

    write_bundle_summary(outdir / "summary.json", root_summary)
    print(f"Saved eval summary to {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
