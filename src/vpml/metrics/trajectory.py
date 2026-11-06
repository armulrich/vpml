"""Whole-trajectory diagnostics with no prescribed transition count."""

from __future__ import annotations

import numpy as np


def matched_cadence_growth(times, model_rate, reference_times, reference_energy, *, fit_window):
    """Supplement native Metric 1 with a reference fit on the model's samples.

    Reuse the original reference-selected window and model rate. This isolates
    sampling-cadence bias without changing the original reported Metric 1.
    """
    from vpml.metrics.early_growth import EarlyElectricFieldGrowthMetric, EarlyGrowthConfig
    times = np.asarray(times, dtype=float)
    reference_times = np.asarray(reference_times, dtype=float)
    if times[0] < reference_times[0] or times[-1] > reference_times[-1] + 1e-9:
        raise ValueError("model samples must lie within the reference history")
    fit = EarlyElectricFieldGrowthMetric(EarlyGrowthConfig(sample_selector="local_maxima")).fit(
        times, np.interp(times, reference_times, reference_energy), time_window=tuple(fit_window))
    return {
        "epsilon_grow_matched_cadence": float(abs(model_rate - fit.gamma_grow)
                                               / max(abs(fit.gamma_grow), 1e-30)),
        "gamma_hr_matched_cadence": float(fit.gamma_grow),
        "metric1_fit_window": list(map(float, fit_window)),
    }


def trailing_mean(values, width: int):
    """Average complete trailing windows; never pad the ends with zeros."""
    values = np.asarray(values, dtype=np.float64)
    if not 1 <= width <= values.shape[-1]:
        raise ValueError("window must fit the trajectory")
    prefix = np.concatenate(
        (np.zeros_like(values[..., :1]), np.cumsum(values, axis=-1)), axis=-1
    )
    return (prefix[..., width:] - prefix[..., :-width]) / width


def envelope_diagnostics(model_energy, reference_energy, *, cadence: float,
                         window_times=(2.5, 5.0, 10.0), floor_ratio=1e-8):
    """Compare levels and signed changes at every complete envelope window.

    A reference-relative floor makes low-energy errors finite; the floor and
    fraction of reference samples below it are reported, not silently hidden.
    All windows and all directions contribute, including flat trajectories.
    """
    model = np.asarray(model_energy, dtype=np.float64)
    reference = np.asarray(reference_energy, dtype=np.float64)
    if model.ndim != 1 or model.shape != reference.shape or model.size < 3:
        raise ValueError("energies must be matching one-dimensional trajectories")
    if (not np.isfinite(cadence) or cadence <= 0 or not 0 < floor_ratio < 1
            or not np.all(np.isfinite(reference)) or np.any(reference < 0)):
        raise ValueError("invalid cadence, floor, or reference energy")
    if not window_times or any(not np.isfinite(t) or t <= 0 for t in window_times):
        raise ValueError("window times must be positive")
    floor = max(float(np.max(reference)) * floor_ratio, np.finfo(float).tiny)
    finite = bool(np.all(np.isfinite(model)) and np.all(model >= 0))
    result = {
        "finite": finite, "energy_floor": floor,
        "reference_floor_fraction": float(np.mean(reference <= floor)),
        "windows": [],
    }
    if not finite:
        result.update(log_envelope_rmse=None, signed_log_change_rmse=None)
        return result
    for window_time in window_times:
        width = max(1, int(round(window_time / cadence)))
        if 2 * width > reference.size:
            continue
        predicted = np.log(np.maximum(trailing_mean(model, width), 0) + floor)
        target = np.log(np.maximum(trailing_mean(reference, width), 0) + floor)
        change_error = (predicted[width:] - predicted[:-width]
                        - target[width:] + target[:-width])
        result["windows"].append({
            "window_time": width * cadence,
            "log_envelope_mse": float(np.mean((predicted - target) ** 2)),
            "signed_log_change_mse": float(np.mean(change_error ** 2)),
        })
    if not result["windows"]:
        raise ValueError("trajectory too short for the requested windows")
    result["log_envelope_rmse"] = float(np.sqrt(np.mean(
        [row["log_envelope_mse"] for row in result["windows"]])))
    result["signed_log_change_rmse"] = float(np.sqrt(np.mean(
        [row["signed_log_change_mse"] for row in result["windows"]])))
    return result


def latent_saturation_diagnostics(normalized_latent, *, bound: float,
                                  cadence: float, initial_time: float = 0.0):
    """Measure saturation of saved states, not unobserved internal substeps."""
    values = np.asarray(normalized_latent)
    if values.ndim != 3 or bound < 0 or not np.isfinite(bound) or cadence <= 0:
        raise ValueError("expected (time, channel, x), nonnegative bound, positive cadence")
    finite = bool(np.all(np.isfinite(values)))
    saturated = (np.abs(values) >= bound * (1.0 - 1e-6)) if bound else np.zeros(
        values.shape, dtype=bool)
    hits = np.flatnonzero(np.any(saturated, axis=(1, 2)))
    return {
        "finite": finite, "configured_bound": float(bound),
        "maximum_normalized_latent": float(np.max(np.abs(values))) if finite else None,
        "saved_state_saturation_fraction": float(np.mean(saturated)),
        "saved_time_saturation_fraction": float(np.mean(np.any(saturated, axis=(1, 2)))),
        "channel_saturation_fraction": np.mean(saturated, axis=(0, 2)).tolist(),
        "first_saved_saturation_time": (
            float(initial_time + hits[0] * cadence) if hits.size else None),
    }


def signed_hermite_flux(coefficients, k_arr, *, first_order: int = 3):
    """Outward flux of squared coefficient norm across each Hermite pair.

    For dC_m/dt=-ik sqrt(m+1) C_(m+1)+..., outward flux is
    -2 k sqrt(m+1) Im(conj(C_m) C_(m+1)). Positive means lower-to-higher
    Hermite order. Inputs are real physical-space coefficients (..., m, x).
    Spatial rFFT multiplicities implement Parseval for norm='forward'.
    This is the streaming contribution, not total nonlinear free-energy flux.
    """
    values = np.asarray(coefficients)
    if values.shape[-2] < 2 or np.iscomplexobj(values):
        raise ValueError("at least two real physical-space Hermite coefficients required")
    wave = np.asarray(k_arr)
    spectral = np.fft.rfft(values, axis=-1, norm="forward")
    if wave.shape != (spectral.shape[-1],):
        raise ValueError("wavenumbers must match the spatial grid")
    orders = np.arange(first_order, first_order + values.shape[-2] - 1)
    multiplicity = np.full(wave.size, 2.0)
    multiplicity[0] = 1.0
    if values.shape[-1] % 2 == 0:
        multiplicity[-1] = 1.0
    flux = (-2 * np.sqrt(orders + 1)[:, None] * wave * multiplicity
            * np.imag(np.conj(spectral[..., :-1, :]) * spectral[..., 1:, :]))
    return np.sum(flux, axis=-1)
