"""Evaluate an Nv sweep on the exact IC families used by rollout training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from model.eval_nv_sweep import main as run_nv_sweep
from model.train.interface_flux_data import (
    IC_SPLITS,
    evaluate_manifest_case,
    load_ic_manifest,
)


VALID_REGIMES = (
    "linear_landau",
    "nonlinear_landau_weak",
    "nonlinear_landau_strong",
)


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def parse_str_tuple(text: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _sample_initial_condition_pair(
    rng: np.random.Generator,
    x: np.ndarray,
    teacher_x: np.ndarray,
    modes: Sequence[float],
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    amplitudes = rng.uniform(0.5, 1.5, size=len(modes))
    phases = rng.uniform(0.0, 2.0 * math.pi, size=len(modes))
    perturbation = np.zeros_like(x)
    teacher_perturbation = np.zeros_like(teacher_x)
    for amplitude, phase, mode in zip(amplitudes, phases, modes):
        perturbation += amplitude * np.cos(float(mode) * x + phase)
        teacher_perturbation += amplitude * np.cos(float(mode) * teacher_x + phase)
    scale = float(eps) / max(len(modes), 1)
    return scale * perturbation, scale * teacher_perturbation


def _slug(text: str) -> str:
    return str(text).replace("-", "m").replace(".", "p").replace(" ", "_")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Metric 1, Metric 2, and Fig10 for every configured training IC."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--nv-list", type=str, default="64")
    parser.add_argument("--Nx", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--T", type=float, default=60.0)
    parser.add_argument("--k0", type=float, default=0.5)
    parser.add_argument("--snapshot-times", type=str, default="20.0,40.0,60.0")
    parser.add_argument("--Nv-plot", dest="nv_plot", type=int, default=1000)
    parser.add_argument("--phase-vmin", dest="phase_vmin", type=float, default=0.0)
    parser.add_argument("--phase-vmax", dest="phase_vmax", type=float, default=0.5)
    parser.add_argument("--phase-vrange", dest="phase_vrange", type=str, default="-4.0,4.0")
    parser.add_argument("--phase-reference-Nv", dest="phase_reference_nv", type=int, default=None)
    parser.add_argument(
        "--phase-reference-mode",
        choices=("projected", "raw_hr_grid"),
        default="raw_hr_grid",
    )
    parser.add_argument("--dealias-23", action="store_true")
    parser.add_argument("--nonlocal-mu", type=float, default=-1.017234)
    parser.add_argument("--teacher-Nx", dest="teacher_nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", dest="teacher_nv", type=int, default=512)
    parser.add_argument("--teacher-dt", dest="teacher_dt", type=float, default=0.01)
    parser.add_argument("--teacher-vmin", dest="teacher_vmin", type=float, default=-8.0)
    parser.add_argument("--teacher-vmax", dest="teacher_vmax", type=float, default=8.0)
    parser.add_argument("--growth-sample-selector", type=str, default="all", choices=("all", "local_maxima"))
    parser.add_argument("--field-num-low-modes", type=int, default=None)
    parser.add_argument("--field-k-max", type=float, default=None)
    parser.add_argument("--regimes", type=str, default=",".join(VALID_REGIMES))
    parser.add_argument("--ic-manifest", type=Path, default=None)
    parser.add_argument("--teacher-reference-dir", type=Path, default=None)
    parser.add_argument(
        "--ic-split",
        choices=(*IC_SPLITS, "all"),
        default="heldout",
    )
    parser.add_argument("--linear-eps", type=float, default=0.01)
    parser.add_argument("--linear-modes", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--linear-num-samples", type=int, default=8)
    parser.add_argument("--linear-seed", type=int, default=0)
    parser.add_argument("--weak-eps", type=str, default="0.03,0.05,0.07,0.1,0.15")
    parser.add_argument("--strong-eps", type=str, default="0.15,0.25,0.35,0.5,0.65")
    return parser


def _case_specs(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.ic_manifest is not None:
        manifest = load_ic_manifest(args.ic_manifest)
        x = np.linspace(
            0.0, 4.0 * math.pi, int(args.Nx), endpoint=False, dtype=np.float64
        )
        teacher_x = np.linspace(
            0.0,
            4.0 * math.pi,
            int(args.teacher_nx),
            endpoint=False,
            dtype=np.float64,
        )
        requested_split = str(args.ic_split)
        specs = []
        for case in manifest["cases"]:
            if requested_split != "all" and str(case["split"]) != requested_split:
                continue
            specs.append(
                {
                    "id": str(case["case_id"]),
                    "label": (
                        f"{case['regime']}, eps={float(case['epsilon']):.4g}, "
                        f"{case['split']}"
                    ),
                    "regime": str(case["regime"]),
                    "split": str(case["split"]),
                    "epsilon": float(case["epsilon"]),
                    "perturbation": evaluate_manifest_case(case, x),
                    "teacher_perturbation": evaluate_manifest_case(case, teacher_x),
                }
            )
        if not specs:
            raise ValueError(
                f"IC manifest {args.ic_manifest} has no cases for split {requested_split!r}"
            )
        return specs

    regimes = parse_str_tuple(args.regimes)
    unknown = sorted(set(regimes) - set(VALID_REGIMES))
    if unknown:
        raise ValueError(f"Unknown evaluation regimes: {unknown!r}")
    if not regimes:
        raise ValueError("--regimes must contain at least one regime")
    x = np.linspace(0.0, 4.0 * math.pi, int(args.Nx), endpoint=False, dtype=np.float64)
    teacher_x = np.linspace(0.0, 4.0 * math.pi, int(args.teacher_nx), endpoint=False, dtype=np.float64)
    specs: List[Dict[str, object]] = []
    if "linear_landau" in regimes:
        modes = parse_float_tuple(args.linear_modes)
        if not modes:
            raise ValueError("--linear-modes must contain at least one mode")
        rng = np.random.default_rng(int(args.linear_seed))
        for sample_idx in range(int(args.linear_num_samples)):
            perturbation, teacher_perturbation = _sample_initial_condition_pair(
                rng, x, teacher_x, modes, float(args.linear_eps)
            )
            specs.append(
                {
                    "id": f"linear_landau_sample{sample_idx:02d}",
                    "label": f"linear_landau sample {sample_idx}",
                    "regime": "linear_landau",
                    "split": "legacy",
                    "perturbation": perturbation,
                    "teacher_perturbation": teacher_perturbation,
                }
            )
    for regime, eps_text in (
        ("nonlinear_landau_weak", args.weak_eps),
        ("nonlinear_landau_strong", args.strong_eps),
    ):
        if regime not in regimes:
            continue
        for eps in parse_float_tuple(eps_text):
            specs.append(
                {
                    "id": f"{regime}_eps{_slug(f'{eps:g}')}",
                    "label": f"{regime}, eps={eps:g}",
                    "regime": regime,
                    "split": "legacy",
                    "epsilon": float(eps),
                    "perturbation": float(eps) * np.cos(float(args.k0) * x),
                    "teacher_perturbation": float(eps) * np.cos(float(args.k0) * teacher_x),
                }
            )
    return specs


def _sweep_args(
    args: argparse.Namespace,
    *,
    case_dir: Path,
    perturbation_path: Path,
    teacher_perturbation_path: Path,
    label: str,
    case_id: str,
) -> List[str]:
    command = [
        "--outdir", str(case_dir),
        "--nv-list", str(args.nv_list),
        "--Nx", str(int(args.Nx)),
        "--dt", str(float(args.dt)),
        "--T", str(float(args.T)),
        "--eps", "0.0",
        "--k0", str(float(args.k0)),
        "--snapshot-times", str(args.snapshot_times),
        "--Nv-plot", str(int(args.nv_plot)),
        "--phase-vmin", str(float(args.phase_vmin)),
        "--phase-vmax", str(float(args.phase_vmax)),
        f"--phase-vrange={args.phase_vrange}",
        "--phase-reference-mode", str(args.phase_reference_mode),
        "--nonlocal-mu", str(float(args.nonlocal_mu)),
        "--teacher-Nx", str(int(args.teacher_nx)),
        "--teacher-Nv", str(int(args.teacher_nv)),
        "--teacher-dt", str(float(args.teacher_dt)),
        "--teacher-vmin", str(float(args.teacher_vmin)),
        "--teacher-vmax", str(float(args.teacher_vmax)),
        "--growth-sample-selector", str(args.growth_sample_selector),
        "--initial-perturbation-npy", str(perturbation_path),
        "--teacher-initial-perturbation-npy", str(teacher_perturbation_path),
        "--case-label", label,
    ]
    if args.checkpoint is not None:
        command.extend(("--checkpoint", str(args.checkpoint)))
    else:
        command.extend(("--checkpoint-dir", str(args.checkpoint_dir)))
    if args.phase_reference_nv is not None:
        command.extend(("--phase-reference-Nv", str(int(args.phase_reference_nv))))
    if args.dealias_23:
        command.append("--dealias-23")
    if args.field_num_low_modes is not None:
        command.extend(("--field-num-low-modes", str(int(args.field_num_low_modes))))
    if args.field_k_max is not None:
        command.extend(("--field-k-max", str(float(args.field_k_max))))
    if args.teacher_reference_dir is not None:
        teacher_reference = args.teacher_reference_dir / f"{case_id}.npz"
        if not teacher_reference.exists():
            raise FileNotFoundError(
                f"Missing cached teacher reference for {case_id}: {teacher_reference}"
            )
        command.extend(("--teacher-reference-npz", str(teacher_reference)))
    return command


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    outdir = args.outdir
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing evaluation-case directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for spec in _case_specs(args):
        case_id = str(spec["id"])
        case_dir = outdir / case_id
        perturbation_path = case_dir / "initial_perturbation.npy"
        teacher_perturbation_path = case_dir / "teacher_initial_perturbation.npy"
        case_dir.mkdir(parents=True, exist_ok=False)
        np.save(perturbation_path, np.asarray(spec["perturbation"], dtype=np.float64))
        np.save(teacher_perturbation_path, np.asarray(spec["teacher_perturbation"], dtype=np.float64))
        print(f"[eval-training-cases] evaluating {case_id}")
        run_nv_sweep(
            _sweep_args(
                args,
                case_dir=case_dir,
                perturbation_path=perturbation_path,
                teacher_perturbation_path=teacher_perturbation_path,
                label=str(spec["label"]),
                case_id=case_id,
            )
        )
        with (case_dir / "summary.json").open("r", encoding="utf-8") as handle:
            case_summary = json.load(handle)
        nv_metrics = case_summary["cases"][0]
        summaries.append(
            {
                "id": case_id,
                "label": str(spec["label"]),
                "regime": str(spec["regime"]),
                "split": str(spec["split"]),
                "epsilon": (
                    None if "epsilon" not in spec else float(spec["epsilon"])
                ),
                "outdir": str(case_dir),
                "summary": str(case_dir / "summary.json"),
                "metrics": {
                    key: nv_metrics[key]
                    for key in (
                        "epsilon_grow",
                        "epsilon_E",
                        "epsilon_E_truncation",
                    )
                },
            }
        )

    regime_means: Dict[str, Dict[str, float]] = {}
    for regime in VALID_REGIMES:
        selected = [case for case in summaries if case["regime"] == regime]
        if not selected:
            continue
        regime_means[regime] = {}
        for key in ("epsilon_grow", "epsilon_E", "epsilon_E_truncation"):
            values = [
                float(case["metrics"][key])
                for case in selected
                if isinstance(case["metrics"][key], (int, float))
            ]
            regime_means[regime][key] = (
                float(np.mean(values)) if values else float("nan")
            )
    macro_mean = {
        key: float(
            np.mean(
                [
                    regime_means[regime][key]
                    for regime in regime_means
                    if np.isfinite(regime_means[regime][key])
                ]
            )
        )
        for key in ("epsilon_grow", "epsilon_E", "epsilon_E_truncation")
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "regimes": list(parse_str_tuple(args.regimes)),
                "ic_manifest": (
                    None if args.ic_manifest is None else str(args.ic_manifest)
                ),
                "ic_split": str(args.ic_split),
                "teacher_reference_dir": (
                    None
                    if args.teacher_reference_dir is None
                    else str(args.teacher_reference_dir)
                ),
                "phase_reference_mode": str(args.phase_reference_mode),
                "cases": summaries,
                "regime_means": regime_means,
                "equal_regime_macro_mean": macro_mean,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(f"Saved per-IC evaluation summary to {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
