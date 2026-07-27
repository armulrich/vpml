#!/usr/bin/env python3
"""Render synchronized raw-HR, truncated, and learned-closure phase-space video."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("VPML_JAX_BACKEND", "cpu")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
import numpy as np

from model.diagnostics.phase_space import phase_space_from_hermite_phys
from model.eval import run_physical_landau_reference
from vpml.core import load_learned_interface_closure_npz
from vpml.nonlinear_landau import NonlinearLandauParams, run_nonlinear_landau_rollout_raw


L_X = 4.0 * math.pi


def _parse_v_range(text: str) -> Tuple[float, float]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if len(values) != 2 or values[0] >= values[1]:
        raise ValueError(f"--v-range must be min,max with min < max, got {text!r}")
    return values[0], values[1]


def _frame_times(*, T: float, frame_dt: float) -> np.ndarray:
    if frame_dt <= 0.0:
        raise ValueError("--frame-dt must be positive")
    count = int(round(float(T) / float(frame_dt)))
    if not np.isclose(count * float(frame_dt), float(T), rtol=0.0, atol=1e-10):
        raise ValueError("T must be an integer multiple of frame-dt")
    return np.arange(count + 1, dtype=np.float64) * float(frame_dt)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render raw-HR, Nv64 truncation, and Nv64 learned-closure phase space."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--case-eps", type=float, default=0.5)
    parser.add_argument("--Nx", type=int, default=200)
    parser.add_argument("--teacher-Nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", type=int, default=512)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--teacher-dt", type=float, default=0.01)
    parser.add_argument("--teacher-vmin", type=float, default=-8.0)
    parser.add_argument("--teacher-vmax", type=float, default=8.0)
    parser.add_argument("--T", type=float, default=60.0)
    parser.add_argument("--k0", type=float, default=0.5)
    parser.add_argument("--frame-dt", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--v-range", type=str, default="-4.0,4.0")
    parser.add_argument("--v-points", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--save-frames", action="store_true")
    return parser


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    run_root = Path(args.run_root)
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else run_root / "models/nv64/interface_closure.npz"
    )
    if args.outdir is not None:
        outdir = Path(args.outdir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = run_root / "diagnostics" / f"phase_space_video_{stamp}"
    return checkpoint, outdir


def _configure_axes(
    axes: Sequence[plt.Axes],
    *,
    deployment_nv: int,
    teacher_nv: int,
    v_range: Tuple[float, float],
) -> None:
    titles = (
        rf"Raw HR grid teacher ($N_v={teacher_nv}$)",
        rf"Unclosed truncation ($N_v={deployment_nv}$)",
        rf"Learned interface flux ($N_v={deployment_nv}$)",
    )
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.set_xticks([0.0, 2.0 * math.pi, 4.0 * math.pi], ["0", r"$2\pi$", r"$4\pi$"])
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.set_ylim(v_range)
    axes[0].set_ylabel("v")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    checkpoint, outdir = _resolve_paths(args)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if outdir.exists():
        raise FileExistsError(f"Refusing to overwrite existing video diagnostics: {outdir}")
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("Matplotlib cannot find ffmpeg; install it or add it to PATH")
    if int(args.fps) <= 0 or int(args.v_points) < 2 or int(args.dpi) <= 0:
        raise ValueError("fps and dpi must be positive, and v-points must be at least 2")

    learned = load_learned_interface_closure_npz(checkpoint)
    deployment_nv = max(int(value) for value in learned.Nv_targets)
    v_range = _parse_v_range(args.v_range)
    v_grid = np.linspace(v_range[0], v_range[1], int(args.v_points), dtype=np.float64)
    frame_times = _frame_times(T=float(args.T), frame_dt=float(args.frame_dt))

    reduced_params = NonlinearLandauParams(
        Nx=int(args.Nx),
        Nv=deployment_nv,
        L=L_X,
        dt=float(args.dt),
        T=float(args.T),
        eps=float(args.case_eps),
        k0=float(args.k0),
        dealias_23=True,
        snapshot_times=tuple(float(t) for t in frame_times),
        v_range=v_range,
        Nv_plot=int(args.v_points),
        vmin=0.0,
        vmax=0.5,
    )
    truncation_raw = run_nonlinear_landau_rollout_raw(reduced_params, "truncation")
    learned_raw = run_nonlinear_landau_rollout_raw(
        reduced_params,
        "learned",
        learned_closure=learned,
    )
    truncation_frames = np.asarray(truncation_raw["snapshot_a_phys"], dtype=np.float64)
    learned_frames = np.asarray(learned_raw["snapshot_a_phys"], dtype=np.float64)

    teacher_x = np.linspace(0.0, L_X, int(args.teacher_Nx), endpoint=False, dtype=np.float64)
    teacher = run_physical_landau_reference(
        Nx=int(args.teacher_Nx),
        Nv=int(args.teacher_Nv),
        Lx=L_X,
        vmin=float(args.teacher_vmin),
        vmax=float(args.teacher_vmax),
        dt=float(args.teacher_dt),
        T=float(args.T),
        perturbation_x=float(args.case_eps) * np.cos(float(args.k0) * teacher_x),
        snapshot_times=tuple(float(t) for t in frame_times),
    )
    teacher_frames = np.asarray(teacher["snapshot_f"], dtype=np.float64)
    teacher_v = np.asarray(teacher["v"], dtype=np.float64)
    teacher_frames = np.stack(
        [
            np.stack(
                [np.interp(v_grid, teacher_v, frame[:, x_idx]) for x_idx in range(frame.shape[1])],
                axis=1,
            )
            for frame in teacher_frames
        ],
        axis=0,
    )
    if (
        truncation_frames.shape[0] != len(frame_times)
        or learned_frames.shape[0] != len(frame_times)
        or teacher_frames.shape[0] != len(frame_times)
    ):
        raise RuntimeError("a rollout did not return every requested video frame")

    outdir.mkdir(parents=True, exist_ok=False)
    if args.save_frames:
        (outdir / "frames").mkdir()
    cmap = plt.colormaps["Spectral_r"].copy()
    cmap.set_bad("#d1d5db")
    figure, axes_array = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    axes = tuple(axes_array)
    _configure_axes(
        axes,
        deployment_nv=deployment_nv,
        teacher_nv=int(args.teacher_Nv),
        v_range=v_range,
    )
    zero_frame = np.zeros((int(args.v_points), int(args.Nx)), dtype=np.float64)
    images = [
        ax.imshow(
            zero_frame,
            origin="lower",
            extent=(0.0, L_X, v_range[0], v_range[1]),
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=0.5,
            interpolation="nearest",
        )
        for ax in axes
    ]
    figure.colorbar(images[-1], ax=axes, fraction=0.025, pad=0.02, label=r"$f(x,v,t)$")
    time_label = figure.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=12)
    mp4_path = outdir / "phase_space_triptych.mp4"
    preview_index = int(np.argmin(np.abs(frame_times - 40.0)))
    preview_time = float(frame_times[preview_index])
    preview_path = outdir / f"preview_t{preview_time:g}.png"
    writer = FFMpegWriter(fps=int(args.fps), codec="libx264", extra_args=["-pix_fmt", "yuv420p"])

    with writer.saving(figure, str(mp4_path), dpi=int(args.dpi)):
        for frame_index, t in enumerate(frame_times):
            fields = (
                teacher_frames[frame_index],
                phase_space_from_hermite_phys(truncation_frames[frame_index], v_grid),
                phase_space_from_hermite_phys(learned_frames[frame_index], v_grid),
            )
            for image, field in zip(images, fields):
                image.set_data(np.ma.masked_invalid(field))
            time_label.set_text(rf"physical time $t={float(t):.1f}$")
            writer.grab_frame()
            if args.save_frames:
                figure.savefig(outdir / "frames" / f"frame_{frame_index:04d}.png", dpi=int(args.dpi))
            if frame_index == preview_index:
                figure.savefig(preview_path, dpi=int(args.dpi))
    plt.close(figure)

    metadata = {
        "checkpoint": str(checkpoint),
        "case_eps": float(args.case_eps),
        "deployment_Nv": deployment_nv,
        "teacher_Nv": int(args.teacher_Nv),
        "frame_count": int(len(frame_times)),
        "frame_dt": float(args.frame_dt),
        "fps": int(args.fps),
        "video_duration_seconds": float(len(frame_times) / int(args.fps)),
        "physical_duration_seconds": float(args.T),
        "columns": [
            "raw physical-grid HR teacher",
            "unclosed Fourier-Hermite truncation",
            "learned Fourier-Hermite interface-flux closure",
        ],
    }
    with (outdir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[phase-video] wrote {mp4_path}")
    print(f"[phase-video] wrote {preview_path}")


if __name__ == "__main__":
    main()
