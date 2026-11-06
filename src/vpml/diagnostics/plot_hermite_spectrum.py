"""Plot Hermite-index spectra from an online projected-reference cache."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from vpml.core import hermite_basis_phi_scaled


_COEFF_KEY_RE = re.compile(r"^(?P<regime>.+)_(?P<split>train|val)_a_hat_ref_nv(?P<nv>\d+)$")


def _scalar_from_npz(data: np.lib.npyio.NpzFile, key: str, default: float | int) -> float:
    if key not in data.files:
        return float(default)
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def _find_available_targets(data: np.lib.npyio.NpzFile) -> Tuple[int, ...]:
    targets = set()
    for name in data.files:
        match = _COEFF_KEY_RE.match(name)
        if match:
            targets.add(int(match.group("nv")))
    return tuple(sorted(targets))


def _selected_coeff_keys(
    data: np.lib.npyio.NpzFile,
    *,
    target_nv: int,
    regimes: Sequence[str] | None,
    splits: Sequence[str],
) -> Tuple[str, ...]:
    selected = []
    regime_filter = set(regimes) if regimes else None
    split_filter = set(splits)
    for name in data.files:
        match = _COEFF_KEY_RE.match(name)
        if not match:
            continue
        if int(match.group("nv")) != int(target_nv):
            continue
        if match.group("split") not in split_filter:
            continue
        if regime_filter is not None and match.group("regime") not in regime_filter:
            continue
        selected.append(name)
    return tuple(sorted(selected))


def hermite_spectrum_from_cache(
    cache_path: Path,
    *,
    target_nv: int | None = None,
    regimes: Sequence[str] | None = None,
    splits: Sequence[str] = ("train", "val"),
) -> Dict[str, object]:
    """Return raw and physical-v weighted Hermite spectra from a reference cache.

    The raw spectrum is
        S_n = sqrt(E_{case,t} sum_{k>0} |C_{n,k}(t)|^2).

    The x-v weighted spectrum multiplies each Hermite index by ||phi_n||_L2(v),
    using the deterministic primal reconstruction basis from the solver.
    """
    cache_path = Path(cache_path)
    with np.load(cache_path) as data:
        targets = _find_available_targets(data)
        if not targets:
            raise ValueError(f"No a_hat_ref_nv* arrays found in {cache_path}")
        if target_nv is None:
            if len(targets) != 1:
                raise ValueError(f"Multiple Nv targets found {targets}; pass --target-nv")
            target_nv = int(targets[0])
        if int(target_nv) not in targets:
            raise ValueError(f"Nv={target_nv} not found in {cache_path}; available={targets}")

        keys = _selected_coeff_keys(
            data,
            target_nv=int(target_nv),
            regimes=regimes,
            splits=splits,
        )
        if not keys:
            raise ValueError(
                f"No arrays selected for Nv={target_nv}, regimes={regimes or 'all'}, splits={tuple(splits)}"
            )

        numerator = None
        denom = 0
        key_shapes = {}
        for key in keys:
            arr = np.asarray(data[key], dtype=np.complex128)
            if arr.ndim != 4:
                raise ValueError(f"{key} must have shape (cases,time,Nv,Nk), got {arr.shape}")
            if arr.shape[2] != int(target_nv):
                raise ValueError(f"{key} has Nv dimension {arr.shape[2]}, expected {target_nv}")
            key_shapes[key] = tuple(int(v) for v in arr.shape)
            k_positive = arr[:, :, :, 1:]
            contribution = np.sum(np.abs(k_positive) ** 2, axis=(0, 1, 3))
            numerator = contribution if numerator is None else numerator + contribution
            denom += int(arr.shape[0]) * int(arr.shape[1])

        assert numerator is not None
        raw = np.sqrt(numerator / float(denom))

        teacher_nv = int(round(_scalar_from_npz(data, "teacher_Nv", 512)))
        teacher_lx = _scalar_from_npz(data, "teacher_L", 4.0 * np.pi)
        vmin = _scalar_from_npz(data, "teacher_vmin", -8.0)
        vmax = _scalar_from_npz(data, "teacher_vmax", 8.0)
        v_grid = np.linspace(vmin, vmax, teacher_nv, dtype=np.float64)
        phi = hermite_basis_phi_scaled(int(target_nv), v_grid, vth=1.0)
        phi_l2_sq = np.trapezoid(phi * phi, x=v_grid, axis=1)
        basis_l2_sq = float(teacher_lx) * phi_l2_sq
        xv_weighted = raw * np.sqrt(np.maximum(phi_l2_sq, 0.0))

    return {
        "cache_path": str(cache_path),
        "target_nv": int(target_nv),
        "keys": keys,
        "key_shapes": key_shapes,
        "num_case_time_states": int(denom),
        "n": np.arange(int(target_nv), dtype=np.int32),
        "S_raw": raw,
        "phi_l2_sq": phi_l2_sq,
        "basis_l2_sq": basis_l2_sq,
        "S_xv_weighted": xv_weighted,
    }


def _plot_bar(n: np.ndarray, y: np.ndarray, *, ylabel: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    ax.bar(n, y, width=0.82, color="#2f5d62", edgecolor="#153236", linewidth=0.7)
    ax.set_xlabel("Hermite index n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    finite_positive = np.asarray(y, dtype=np.float64)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
    if finite_positive.size and float(np.max(finite_positive) / np.min(finite_positive)) >= 10.0:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0.0)
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xticks(n)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_outputs(result: Dict[str, object], outdir: Path) -> Dict[str, str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target_nv = int(result["target_nv"])
    n = np.asarray(result["n"], dtype=np.int32)
    raw = np.asarray(result["S_raw"], dtype=np.float64)
    xv = np.asarray(result["S_xv_weighted"], dtype=np.float64)
    phi_l2_sq = np.asarray(result["phi_l2_sq"], dtype=np.float64)
    basis_l2_sq = np.asarray(result["basis_l2_sq"], dtype=np.float64)
    basis_l2 = np.sqrt(np.maximum(basis_l2_sq, 0.0))

    raw_png = outdir / f"hermite_spectrum_raw_nv{target_nv}.png"
    xv_png = outdir / f"hermite_spectrum_xv_weighted_nv{target_nv}.png"
    basis_png = outdir / f"hermite_basis_l2_weight_nv{target_nv}.png"
    csv_path = outdir / f"hermite_spectrum_nv{target_nv}.csv"
    json_path = outdir / f"hermite_spectrum_nv{target_nv}.json"

    _plot_bar(
        n,
        raw,
        ylabel=r"$S_n=\left(\mathbb{E}_{case,t}\sum_{k>0}|\hat C^{HR}_{n,k}|^2\right)^{1/2}$",
        title=f"Raw Hermite coefficient spectrum, Nv={target_nv}",
        output_path=raw_png,
    )
    _plot_bar(
        n,
        xv,
        ylabel=r"$S_n^{xv}=S_n\,\|\phi_n\|_{L_v^2}$",
        title=f"Physical-v weighted Hermite spectrum, Nv={target_nv}",
        output_path=xv_png,
    )
    _plot_bar(
        n,
        basis_l2,
        ylabel=r"$\|e^{ikx}\phi_n(v)\|_{L^2_{x,v}}$",
        title=f"Fourier-Hermite basis L2 norm by Hermite index, Nv={target_nv}",
        output_path=basis_png,
    )

    table = np.column_stack([n, raw, phi_l2_sq, basis_l2_sq, basis_l2, xv])
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="n,S_raw,phi_l2_sq,basis_l2_sq,basis_l2,S_xv_weighted",
        comments="",
    )
    summary = {
        "cache_path": result["cache_path"],
        "target_nv": target_nv,
        "keys": list(result["keys"]),
        "key_shapes": result["key_shapes"],
        "num_case_time_states": int(result["num_case_time_states"]),
        "raw_png": str(raw_png),
        "xv_weighted_png": str(xv_png),
        "basis_l2_png": str(basis_png),
        "csv": str(csv_path),
        "n": n.tolist(),
        "S_raw": raw.tolist(),
        "phi_l2_sq": phi_l2_sq.tolist(),
        "basis_l2_sq": basis_l2_sq.tolist(),
        "basis_l2": basis_l2.tolist(),
        "S_xv_weighted": xv.tolist(),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "raw_png": str(raw_png),
        "xv_weighted_png": str(xv_png),
        "basis_l2_png": str(basis_png),
        "csv": str(csv_path),
        "json": str(json_path),
    }


def _parse_csv(text: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Online reference cache .npz")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory for diagnostic artifacts")
    parser.add_argument("--target-nv", type=int, default=None, help="Nv target to plot; inferred if unique")
    parser.add_argument("--regimes", default="", help="Comma-separated regimes; default uses all")
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits, default train,val")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    regimes = _parse_csv(args.regimes) or None
    splits = _parse_csv(args.splits)
    if not splits:
        raise ValueError("--splits must select at least one split")
    result = hermite_spectrum_from_cache(
        args.cache,
        target_nv=args.target_nv,
        regimes=regimes,
        splits=splits,
    )
    outdir = args.outdir
    if outdir is None:
        target_nv = int(result["target_nv"])
        outdir = args.cache.parent / "diagnostics" / f"nv{target_nv}"
    outputs = write_outputs(result, outdir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
