"""Summarize the local Gaussian pilot without changing the research notes.

The effective-rank calculation is descriptive. It does not certify independent
histories, reference accuracy, or improvement of a learned closure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vpml_gaussian_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
from scipy.interpolate import interp1d


def effective_rank(matrix: np.ndarray) -> dict:
    """Roy--Vetterli entropy of singular values (not squared values)."""
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected a finite two-dimensional feature matrix")
    singular = np.linalg.svd(a, compute_uv=False)
    total = singular.sum()
    probabilities = singular / total if total > 0 else np.zeros_like(singular)
    positive = probabilities[probabilities > 0]
    rank = float(np.exp(-np.sum(positive * np.log(positive)))) if total > 0 else 0.0
    return {"effective_rank": rank, "shape": list(a.shape),
            "singular_values": singular.tolist(),
            "normalized_singular_values": probabilities.tolist()}


def spatial_features(case: dict, modes: int, target: bool = False) -> np.ndarray:
    """One row per time; each field contributes modes 1..m and real/imag parts."""
    names = ("g",) if target else ("n", "u", "p")
    features = []
    for name in names:
        field = np.asarray(case[name], dtype=np.float64)
        if field.ndim != 2 or field.shape[1] < 2 * modes + 1:
            raise ValueError(f"Insufficient spatial resolution for {modes} modes")
        if name in ("n", "p"):
            field = field - 1.0
        fourier = np.fft.rfft(field, axis=1)[:, 1:modes + 1] / field.shape[1]
        features.extend((fourier.real, fourier.imag))
    return np.concatenate(features, axis=1)


def history_features(case: dict, modes: int, target: bool = False) -> np.ndarray:
    snapshots = spatial_features(case, modes, target)
    times = np.asarray(case["times"], dtype=np.float64)
    anchors = np.arange(15.0, 35.0)
    sample_times = anchors[:, None] + np.array([-15.0, -7.5, 0.0])[None, :]
    distances = np.abs(times[None, None, :] - sample_times[:, :, None])
    indices = distances.argmin(axis=2)
    if np.max(np.abs(times[indices] - sample_times)) > 1e-8:
        raise ValueError("Saved times must contain beginning, midpoint, end of each history")
    return snapshots[indices].reshape(len(anchors), -1)


def common_standardization(a: np.ndarray, b: np.ndarray) -> tuple:
    """Balanced common calibration; no independent or per-case normalization."""
    if a.shape != b.shape:
        raise ValueError("Comparison requires equal rows and feature counts")
    calibration = np.vstack((a, b))
    mean = calibration.mean(axis=0)
    std = calibration.std(axis=0)
    # A common tiny floor avoids amplifying columns that are roundoff-only.
    floor = max(float(std.max()) * 1e-12, np.finfo(np.float64).tiny)
    scales = np.maximum(std, floor)
    return (a - mean) / scales, (b - mean) / scales, mean, scales


def load_cases(root: Path) -> list[dict]:
    cases = []
    for meta_path in sorted((root / "cases").glob("*.json")):
        metadata = json.loads(meta_path.read_text())
        path = meta_path.with_suffix(".npz")
        if not path.exists() or metadata.get("family") not in ("gaussian", "linear_landau"):
            continue
        expected = metadata.get("result_sha256")
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Result checksum mismatch: {path}")
        with np.load(path) as saved:
            names = ("times", "n", "u", "p", "E", "Q", "g", "full_energy",
                     "linear_energy", "linear_field", "final_f", "x", "v",
                     "diagnostic_times", "diagnostics")
            arrays = {name: saved[name] for name in names if name in saved}
        cases.append({**metadata, **arrays, "id": meta_path.stem})
    return sorted(cases, key=lambda c: (c["family"], c.get("seed", 0), c.get("strength", 0), c["id"]))


def primary_cases(cases: list[dict]) -> list[dict]:
    """Only the declared base collection contributes to population summaries."""
    return [case for case in cases if case["id"].endswith("_base")]


def _relative_l2(coarse, fine, times=None, velocity=None):
    difference = np.asarray(coarse) - np.asarray(fine)
    if times is not None:
        numerator = np.trapezoid(np.mean(difference ** 2, axis=1), x=times)
        denominator = np.trapezoid(np.mean(np.asarray(fine) ** 2, axis=1), x=times)
    elif velocity is not None:
        numerator = np.trapezoid(np.mean(difference ** 2, axis=1), x=velocity)
        denominator = np.trapezoid(np.mean(np.asarray(fine) ** 2, axis=1), x=velocity)
    else:
        numerator, denominator = np.sum(difference ** 2), np.sum(np.asarray(fine) ** 2)
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    return float(np.sqrt(numerator / denominator))


def common_spectral_band_errors(coarse: dict, fine: dict, coarse_indices, fine_indices,
                                times, max_mode: int) -> dict:
    """Compare matched coefficients before any fine-to-coarse nodal sampling.

    Zero mode is retained after subtracting the n/p background. Positive modes
    carry weight two, reproducing the real-field Parseval norm in this band.
    """
    result = {}
    weights = np.ones(max_mode + 1)
    weights[1:] = 2
    for field in ("E", "n", "u", "p", "Q", "g"):
        if field not in coarse or field not in fine:
            continue
        a, b = np.asarray(coarse[field])[coarse_indices], np.asarray(fine[field])[fine_indices]
        if max_mode >= min(a.shape[1], b.shape[1]) / 2:
            raise ValueError("Common band must lie strictly below both Nyquist modes")
        if field in ("n", "p"):
            a, b = a - 1, b - 1
        fa = np.fft.rfft(a, axis=1)[:, :max_mode + 1] / a.shape[1]
        fb = np.fft.rfft(b, axis=1)[:, :max_mode + 1] / b.shape[1]
        numerator = np.trapezoid(np.sum(weights * np.abs(fa - fb) ** 2, axis=1), x=times)
        denominator = np.trapezoid(np.sum(weights * np.abs(fb) ** 2, axis=1), x=times)
        error = 0.0 if denominator == numerator == 0 else None if denominator == 0 else float(np.sqrt(numerator / denominator))
        label = field + "_perturbation" if field in ("n", "p") else field
        result[label] = error
    return result


def refinement_comparisons(cases: list[dict], tolerances=None) -> list[dict]:
    """Compare each sensitivity solve to its base on common physical coordinates."""
    tolerances = tolerances or {"relative_field_L2": .01, "relative_moment_L2": .01,
                                "relative_closure_target_L2": .05}
    bases = {(c["family"], c["seed"], c["strength"], c["k0"]): c for c in primary_cases(cases)}
    comparisons = []
    for fine in cases:
        if fine["id"].endswith("_base"):
            continue
        coarse = bases.get((fine["family"], fine["seed"], fine["strength"], fine["k0"]))
        if coarse is None:
            continue
        nx_coarse, nx_fine = len(coarse["x"]), len(fine["x"])
        if nx_fine % nx_coarse:
            raise ValueError("Refinement comparison requires an integer spatial grid ratio")
        stride = nx_fine // nx_coarse
        if not np.allclose(fine["x"][::stride], coarse["x"], rtol=1e-12, atol=1e-12):
            raise ValueError("Spatial refinement coordinates do not coincide")
        common = coarse["times"] <= min(coarse["times"][-1], fine["times"][-1]) + 1e-10
        times = coarse["times"][common]
        distances = np.abs(fine["times"][None, :] - times[:, None])
        indices = distances.argmin(axis=1)
        if np.max(np.abs(fine["times"][indices] - times)) > 1e-8:
            raise ValueError("Sensitivity comparison requires matching saved times")
        errors = {}
        for field in ("E", "n", "u", "p", "Q", "g"):
            if field not in coarse or field not in fine:
                continue
            a, b = coarse[field][common], fine[field][indices, ::stride]
            errors[field] = _relative_l2(a, b, times=times)
            if field in ("n", "p"):
                errors[field + "_perturbation"] = _relative_l2(a - 1, b - 1, times=times)
        # Nv=4096 and 2048 on endpoint-inclusive [-8,8] are not nested.
        # Interpolate using actual velocity coordinates, never a stride-2 slice.
        final_f_error = None
        if (abs(coarse["times"][-1] - fine["times"][-1]) < 1e-8
                and fine["v"][0] <= coarse["v"][0] and fine["v"][-1] >= coarse["v"][-1]):
            fine_final_on_coarse = interp1d(fine["v"], fine["final_f"][:, ::stride],
                                            axis=0, kind="linear", bounds_error=True)(coarse["v"])
            final_f_error = _relative_l2(coarse["final_f"], fine_final_on_coarse, velocity=coarse["v"])
        thresholds = {"E": tolerances["relative_field_L2"],
                      "n_perturbation": tolerances["relative_moment_L2"],
                      "u": tolerances["relative_moment_L2"],
                      "p_perturbation": tolerances["relative_moment_L2"],
                      "Q": tolerances["relative_closure_target_L2"],
                      "g": tolerances["relative_closure_target_L2"]}
        failed = [name for name, threshold in thresholds.items()
                  if name in errors and (errors[name] is None or errors[name] > threshold)]
        coarse_grid, fine_grid = coarse.get("grid", {}), fine.get("grid", {})
        changed = {key: {"base": coarse_grid.get(key), "sensitivity": fine_grid.get(key)}
                   for key in sorted(set(coarse_grid) | set(fine_grid))
                   if coarse_grid.get(key) != fine_grid.get(key)}
        spectral_bands = {}
        for max_mode in (4, 63):
            if max_mode >= min(nx_coarse, nx_fine) / 2:
                continue
            spectral_bands[f"modes_0_to_{max_mode}"] = {
                "maximum_positive_mode": max_mode,
                "physical_wavenumber_maximum": max_mode * coarse["k0"],
                "relative_L2_errors": common_spectral_band_errors(
                    coarse, fine, common, indices, times, max_mode),
                "scope": "Information-metric low-mode band" if max_mode == 4 else
                    "Positive modes below the Nyquist mode of an Nx=128 resolved grid",
            }
        comparisons.append({
            "base_case": coarse["id"], "sensitivity_case": fine["id"], "changed_grid_parameters": changed,
            "common_time_interval": [float(times[0]), float(times[-1])],
            "relative_L2_errors": errors, "final_distribution_relative_L2": final_f_error,
            "common_spectral_band_comparisons": spectral_bands,
            "spectral_band_convention": "Fourier coefficients computed on each calculation's own grid and normalized by its Nx. Mode zero retained after subtracting the n/p unit backgrounds; positive modes weighted by two. Trapezoid integration in time. These diagnostics do not replace the existing nodal or positivity gates.",
            "relative_L2_reference": "Sensitivity calculation sampled at base spatial nodes and matching saved times; time integration uses trapezoid weights.",
            "final_distribution_sampling": "Fine x nodes coincide with base x; linear interpolation from actual fine v coordinates to base v coordinates.",
            "eta_NL_absolute_change": abs(fine["eta_NL"] - coarse["eta_NL"]),
            "tolerances": thresholds, "failed_refinement_checks": failed,
            "base_minimum_f": coarse["min_full_distribution"],
            "sensitivity_minimum_f": fine["min_full_distribution"],
            "base_max_negative_mass_fraction": coarse["max_negative_mass_fraction"],
            "sensitivity_max_negative_mass_fraction": fine["max_negative_mass_fraction"],
            "base_screen_pass": bool(coarse.get("qualification_pass", False)),
            "sensitivity_screen_pass": bool(fine.get("qualification_pass", False)),
            "sensitivity_screen_failures": fine.get("qualification_failures", []),
            "refinement_checks_pass": not failed,
            "sensitivity_screen_and_refinement_pass": not failed and bool(fine.get("qualification_pass", False)),
        })
    return comparisons


def information_comparison(cases: list[dict]) -> dict:
    cases = primary_cases(cases)
    gaussian = [c for c in cases if c["family"] == "gaussian"]
    landau = [c for c in cases if c["family"] == "linear_landau"]
    count = min(len(gaussian), len(landau))
    result = {
        "status": "exploratory" if any(not c.get("qualification_pass", False) for c in cases)
                  else "descriptive_reference_diagnostic",
        "interpretation": "Variation in the chosen features, not independent sample count or closure-prediction accuracy.",
        "common_calibration": "Equal numbers of Gaussian and Landau histories, stacked to determine shared column means and population standard deviations; no per-case amplitude normalization.",
        "history_anchors": list(range(15, 35)), "history_offsets": [-15, -7.5, 0],
        "spatial_coordinate": "x/L; modes are harmonic indices, and physical k0 is retained in case metadata",
        "selected_cases_per_family": count, "comparisons": {},
        "sampling_structure": {
            "gaussian_spatial_realizations": len({c["seed"] for c in gaussian}),
            "gaussian_paired_amplitudes": sorted({c["strength"] for c in gaussian}),
            "gaussian_trajectory_count": len(gaussian),
            "landau_distinct_wavenumbers": len({c["k0"] for c in landau}),
            "landau_trajectory_count": len(landau),
            "interpretation": "The Gaussian amplitudes share their seed's spatial shapes; the pooled trajectory count is not the count of independent Gaussian shapes. Landau cases vary physical wavenumber of one density Fourier mode.",
        },
        "within_amplitude_comparisons": {},
    }
    if count == 0:
        result["status"] = "awaiting_matched_landau_cases"
        return result
    # Use all cases when counts match. For partial outputs choose evenly spaced
    # sorted cases rather than selecting only one amplitude or one seed.
    def evenly(c):
        idx = np.rint(np.linspace(0, len(c) - 1, count)).astype(int)
        return [c[i] for i in idx]
    all_gaussian, all_landau = gaussian, landau
    gaussian, landau = evenly(gaussian), evenly(landau)
    result["gaussian_case_ids"] = [c["id"] for c in gaussian]
    result["landau_case_ids"] = [c["id"] for c in landau]
    for target, label in ((False, "moment_history"), (True, "closure_target_history")):
        if target and any("g" not in c for c in gaussian + landau):
            continue
        for modes in (2, 4, 8):
            a = np.vstack([history_features(c, modes, target) for c in gaussian])
            b = np.vstack([history_features(c, modes, target) for c in landau])
            aa, bb, mean, scales = common_standardization(a, b)
            result["comparisons"][f"{label}_modes_{modes}"] = {
                "gaussian": effective_rank(aa), "linear_landau": effective_rank(bb),
                "collection_centered_sensitivity": {
                    "gaussian": effective_rank(aa - aa.mean(axis=0)),
                    "linear_landau": effective_rank(bb - bb.mean(axis=0)),
                    "scaling": "Same column scales; each collection independently centered after the shared scaling.",
                },
                "column_means": mean.tolist(), "column_scales": scales.tolist(),
                "histories_per_case": 20, "cases_per_family": count,
            }
    for strength in sorted({c["strength"] for c in all_gaussian}):
        group = [c for c in all_gaussian if c["strength"] == strength]
        n_group = min(len(group), len(all_landau))
        group = group[:n_group]
        indices = np.rint(np.linspace(0, len(all_landau) - 1, n_group)).astype(int)
        matched_landau = [all_landau[i] for i in indices]
        entry = {"cases_per_family": n_group, "gaussian_case_ids": [c["id"] for c in group],
                 "landau_case_ids": [c["id"] for c in matched_landau],
                 "scope": "Small within-amplitude comparison; Landau cases spread across its wavenumber range. Shared calibration is recomputed within this pair.",
                 "comparisons": {}}
        for target, label in ((False, "moment_history"), (True, "closure_target_history")):
            if target and any("g" not in c for c in group + matched_landau):
                continue
            for modes in (2, 4, 8):
                a = np.vstack([history_features(c, modes, target) for c in group])
                b = np.vstack([history_features(c, modes, target) for c in matched_landau])
                aa, bb, mean, scales = common_standardization(a, b)
                entry["comparisons"][f"{label}_modes_{modes}"] = {
                    "gaussian": effective_rank(aa), "linear_landau": effective_rank(bb),
                    "collection_centered_sensitivity": {
                        "gaussian": effective_rank(aa - aa.mean(axis=0)),
                        "linear_landau": effective_rank(bb - bb.mean(axis=0)),
                        "scaling": "Same shared column scales; each collection independently centered.",
                    },
                    "column_means": mean.tolist(), "column_scales": scales.tolist(),
                }
        result["within_amplitude_comparisons"][str(strength)] = entry
    return result


def refinement_diagnostic_figure(cases: list[dict], destination: Path) -> str | None:
    """Show the actual negative-population history of the selected stress case."""
    selected = [case for case in cases if case["family"] == "gaussian"
                and case["seed"] == 28002 and case["strength"] == 1.0
                and "diagnostic_times" in case and "diagnostics" in case
                and any(case["id"].endswith("_" + variant)
                        for variant in ("base", "refined", "half_dt"))]
    if len(selected) < 2 or not any(c["id"].endswith("_base") for c in selected):
        return None
    order = {"base": 0, "refined": 1, "half_dt": 2}
    def variant(case):
        return next(label for label in order if case["id"].endswith("_" + label))
    selected.sort(key=lambda case: order[variant(case)])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    for case, color in zip(selected, ("#6f39a8", "#e27b21", "#1679b5")):
        times, stats, grid = case["diagnostic_times"], case["diagnostics"], case["grid"]
        label = rf"${grid['nx']}\times{grid['nv']},\ \Delta t={grid['dt']:g}$"
        names = case.get("diagnostic_names", ["electric", "linear_electric", "delta_e2",
                    "mass", "total_energy", "min_f", "negative_mass_fraction", "boundary_abs_fraction"])
        column = {name: i for i, name in enumerate(names)}
        axes[0].semilogy(times, stats[:, column["electric"]], color=color, label=label, lw=1.3)
        axes[1].plot(times, stats[:, column["min_f"]], color=color, lw=1.3)
        axes[2].semilogy(times, np.maximum(stats[:, column["negative_mass_fraction"]], 1e-20),
                         color=color, lw=1.3)
    axes[0].set_ylabel(r"Electric energy $W_E$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel(r"Minimum distribution $\min_{x,v}f$")
    axes[1].axhline(0, color="black", lw=.7)
    axes[2].set_ylabel("Negative population fraction")
    axes[2].axhline(1e-8, color=".35", ls="--", lw=1, label=r"Screen tolerance $10^{-8}$")
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    axes[2].yaxis.set_major_locator(LogLocator(numticks=6))
    for ax in axes:
        ax.set_xlabel("Time")
        ax.grid(alpha=.2)
    path = destination / "refinement_diagnostics.png"
    fig.savefig(path); plt.close(fig)
    return str(path)


def figures(cases: list[dict], info: dict, destination: Path) -> list[str]:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 170})
    gauss = [c for c in primary_cases(cases) if c["family"] == "gaussian"]
    if not gauss:
        return []
    paths = []
    seeds = sorted({c["seed"] for c in gauss})
    colors = plt.cm.viridis(np.linspace(.15, .85, len(seeds)))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5), constrained_layout=True)
    for seed, color in zip(seeds, colors):
        selected = sorted([c for c in gauss if c["seed"] == seed], key=lambda c: c["strength"])
        x = [c["strength"] for c in selected]
        axes[0].plot(x, [c["eta_NL"] for c in selected], "o-", color=color, label=str(seed))
        axes[1].plot(x, [max(-c["min_full_distribution"], 1e-18) for c in selected], "o-", color=color)
        axes[2].plot(x, [max(c["max_negative_mass_fraction"], 1e-20) for c in selected], "o-", color=color)
    axes[0].set_ylabel(r"Field departure $\eta_{\rm NL}$")
    axes[0].legend(title="Spatial seed", frameon=False, fontsize=8)
    axes[1].set_ylabel(r"Largest negative depth $\max(-\min f,0)$")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("Maximum negative population fraction")
    axes[2].set_yscale("log")
    for ax in axes[1:]:
        ax.axhline(1e-8, color="0.35", lw=.9, ls="--", label="Screening tolerance")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.yaxis.set_major_locator(LogLocator(numticks=6))
    for ax in axes:
        ax.set_xlabel(r"Amplitude $\lambda$")
        ax.grid(alpha=.2)
    path = destination / "amplitude_and_reference_checks.png"
    fig.savefig(path); plt.close(fig); paths.append(str(path))

    first = sorted([c for c in gauss if c["seed"] == seeds[0]], key=lambda c: c["strength"])
    fig, axes = plt.subplots(1, len(first), figsize=(3.05 * len(first), 3.25), squeeze=False,
                             sharey=True, constrained_layout=True)
    for ax, case in zip(axes[0], first):
        scale = case["strength"] ** 2
        ax.semilogy(case["times"], case["full_energy"] / scale, color="#6f39a8", lw=1.5, label="Kinetic")
        ax.semilogy(case["times"], case["linear_energy"] / scale, color="#e27b21", lw=1.1, ls="--", label="Linearized")
        ax.set_title(rf"$\lambda={case['strength']:g}$")
        ax.set_xlabel("Time")
        ax.grid(alpha=.18)
    axes[0, 0].set_ylabel(r"Electric energy $W_E/\lambda^2$")
    axes[0, -1].legend(frameon=False, fontsize=9)
    path = destination / "energy_histories.png"
    fig.savefig(path); plt.close(fig); paths.append(str(path))

    select = [first[0], first[len(first) // 2], first[-1]] if len(first) >= 3 else first
    fig, axes = plt.subplots(1, len(select), figsize=(4 * len(select), 3.3), squeeze=False,
                             sharey=True, constrained_layout=True)
    v_max = max(np.max(c["final_f"]) for c in select)
    v_min = min(0.0, min(np.min(c["final_f"]) for c in select))
    for ax, case in zip(axes[0], select):
        mask = np.abs(case["v"]) <= 4
        xnorm = case["x"] / (2 * np.pi / case["k0"])
        mesh = ax.pcolormesh(xnorm, case["v"][mask], case["final_f"][mask],
                             shading="auto", cmap="viridis", vmin=v_min, vmax=v_max,
                             rasterized=True)
        ax.contour(xnorm, case["v"][mask], case["final_f"][mask],
                   levels=[.1, .3, .5], colors="white", linewidths=.5, alpha=.6)
        ax.set_xlabel(r"$x/L$")
        ax.set_title(rf"$\lambda={case['strength']:g}$")
    axes[0, 0].set_ylabel("Velocity")
    fig.colorbar(mesh, ax=axes[0].tolist(), label=r"$f(x,v,T)$", shrink=.9)
    path = destination / "final_phase_space.png"
    fig.savefig(path); plt.close(fig); paths.append(str(path))

    comparison = info.get("comparisons", {}).get("moment_history_modes_4")
    if comparison:
        fig, axes = plt.subplots(1, 2, figsize=(11.3, 3.7), constrained_layout=True, sharey=True)
        for ax, values, title in zip(axes,
                (comparison, comparison["collection_centered_sensitivity"]),
                ("Shared centering", "Each collection centered")):
            for key, label, color in (("gaussian", "Gaussian family", "#6f39a8"),
                                      ("linear_landau", "Linear Landau", "#e27b21")):
                singular = np.asarray(values[key]["normalized_singular_values"])
                ax.semilogy(np.arange(1, len(singular) + 1), np.maximum(singular, 1e-18),
                            "o-", ms=2.5, lw=1.1, color=color,
                            label=f"{label}, effective rank {values[key]['effective_rank']:.2f}")
            ax.set_xlabel("Singular value index")
            ax.set_title(title)
            ax.set_ylim(1e-12, 1)
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=.2)
        axes[0].set_ylabel(r"Normalized singular value $\sigma_j/\sum_k\sigma_k$")
        path = destination / "information_spectrum.png"
        fig.savefig(path); plt.close(fig); paths.append(str(path))
    refinement_path = refinement_diagnostic_figure(cases, destination)
    if refinement_path:
        paths.append(refinement_path)
    return paths


def analyze(root: Path) -> dict:
    destination = root / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    cases = load_cases(root)
    if not cases:
        raise ValueError("No completed case files found")
    rows = []
    keys = ["id", "family", "seed", "strength", "k0", "eta_NL", "qualification_pass",
            "min_full_distribution", "max_negative_mass_fraction", "max_relative_mass_error",
            "max_relative_total_energy_error", "max_boundary_mass_fraction", "elapsed_seconds"]
    for case in cases:
        rows.append({key: case.get(key) for key in keys})
    with (destination / "case_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    base_cases = primary_cases(cases)
    info = information_comparison(base_cases)
    grouped = {}
    for strength in sorted({c["strength"] for c in base_cases if c["family"] == "gaussian"}):
        group = [c for c in base_cases if c["family"] == "gaussian" and c["strength"] == strength]
        eta = np.array([c["eta_NL"] for c in group])
        grouped[str(strength)] = {"count": len(group), "eta_median": float(np.median(eta)),
                                 "eta_min": float(eta.min()), "eta_max": float(eta.max()),
                                 "qualified_count": sum(bool(c["qualification_pass"]) for c in group)}
    summary = {
        "cases": rows, "amplitude_groups": grouped, "information_content": info,
        "primary_case_ids": [c["id"] for c in base_cases],
        "sensitivity_case_ids": [c["id"] for c in cases if not c["id"].endswith("_base")],
        "solver_scope": {
            "gaussian": "Nonlinear collisionless Vlasov-Poisson with cubic B-spline Strang splitting; comparison is with its linearization about the homogeneous unit Maxwellian from the same initial distribution.",
            "linear_landau": "Linearized kinetic evolution about the homogeneous unit Maxwellian, initialized with one small density Fourier mode at each physical wavenumber; this is not a nonlinear Gaussian-family trajectory.",
            "case_solver_metadata": {c["id"]: {key: c[key] for key in
                ("solver", "linearization", "evolution", "initialization") if key in c} for c in cases},
        },
        "refinement_comparisons": refinement_comparisons(cases),
        "qualification_failures": {c["id"]: c.get("qualification_failures", []) for c in cases
                                   if not c.get("qualification_pass", False)},
        "no_model_training_performed": True,
        "figures": figures(cases, info, destination),
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(destination / "feature_spectra.npz", **{
        f"{name}_{family}": result[family]["singular_values"]
        for name, result in info.get("comparisons", {}).items()
        for family in ("gaussian", "linear_landau")})
    print(json.dumps({"case_count": len(cases), "amplitude_groups": grouped,
                      "information_status": info["status"], "figures": summary["figures"]}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root.resolve())
