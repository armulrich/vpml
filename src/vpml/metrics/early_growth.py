"""Metric 1: early electric-field growth for a posteriori closure assessment.

This module fits an effective early-time growth rate from an electric-field-energy
trace, following the same validation style used in Ho et al. (2018) for Landau
damping and two-stream growth diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

SampleSelector = Literal["all", "local_maxima"]


@dataclass(frozen=True)
class EarlyGrowthConfig:
    """Configuration for the early electric-field growth metric.

    Parameters
    ----------
    time_window:
        Optional manual fit window ``(t_a, t_b)``. When omitted, the fit window is
        inferred from the supplied reference trace by ending at the first local
        maximum, or at the last valid sample if no local maximum is present.
    sample_selector:
        ``"all"`` fits all valid samples in the chosen window.
        ``"local_maxima"`` fits only local maxima, which is useful when the early
        energy trace oscillates around an exponential envelope.
    min_points:
        Minimum number of samples required for the least-squares fit.
    positive_floor:
        Samples at or below this threshold are discarded before taking the log.
    local_maxima_rel_prominence:
        Relative prominence threshold used when identifying local maxima.
    maxima_fallback_to_all:
        If ``True`` and too few local maxima are found, fall back to fitting all
        valid samples in the selected time window.
    denom_floor:
        Lower bound on ``|gamma_grow^HR|`` when forming the relative error.
    """

    time_window: Optional[Tuple[float, float]] = None
    sample_selector: SampleSelector = "all"
    min_points: int = 2
    positive_floor: float = 1e-30
    local_maxima_rel_prominence: float = 1e-6
    maxima_fallback_to_all: bool = True
    denom_floor: float = 1e-30

    def __post_init__(self) -> None:
        if self.time_window is not None:
            t_a, t_b = (float(v) for v in self.time_window)
            if t_b <= t_a:
                raise ValueError("time_window must satisfy t_b > t_a")
        if self.sample_selector not in {"all", "local_maxima"}:
            raise ValueError("sample_selector must be 'all' or 'local_maxima'")
        if int(self.min_points) < 2:
            raise ValueError("min_points must be at least 2")
        if float(self.positive_floor) <= 0.0:
            raise ValueError("positive_floor must be positive")
        if float(self.local_maxima_rel_prominence) < 0.0:
            raise ValueError("local_maxima_rel_prominence must be nonnegative")
        if float(self.denom_floor) <= 0.0:
            raise ValueError("denom_floor must be positive")


@dataclass(frozen=True)
class GrowthFitResult:
    """Least-squares fit result for ``log(series) = a + 2 gamma_grow t``."""

    gamma_grow: float
    intercept: float
    t_a: float
    t_b: float
    num_samples: int
    sample_selector: str


@dataclass(frozen=True)
class GrowthComparisonResult:
    """Comparison result for Metric 1.

    The notation follows the learned-closure write-up:
    ``gamma_grow^theta`` denotes the learned-rollout rate and ``gamma_grow^HR`` the
    reference high-resolution rate.
    """

    gamma_grow_theta: float
    gamma_grow_hr: float
    epsilon_grow: float
    fit_theta: GrowthFitResult
    fit_hr: GrowthFitResult
    t_a: float
    t_b: float


def _validate_1d_series(times: np.ndarray, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    if times.shape != series.shape:
        raise ValueError(f"times and series must have the same shape, got {times.shape} and {series.shape}")
    if times.size < 2:
        raise ValueError("At least two time samples are required")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")
    return times, series


def _local_maxima_indices(values: np.ndarray, rel_prominence: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 3:
        return np.zeros((0,), dtype=np.int64)

    prev_delta = values[1:-1] - values[:-2]
    next_delta = values[1:-1] - values[2:]
    candidate = (prev_delta >= 0.0) & (next_delta > 0.0)
    prominence = np.minimum(prev_delta, next_delta) / np.maximum(np.abs(values[1:-1]), 1e-30)
    return np.flatnonzero(candidate & (prominence >= float(rel_prominence))) + 1


def _strict_local_maxima_indices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 3:
        return np.zeros((0,), dtype=np.int64)
    prev_delta = values[1:-1] - values[:-2]
    next_delta = values[1:-1] - values[2:]
    return np.flatnonzero((prev_delta >= 0.0) & (next_delta > 0.0)) + 1


class EarlyElectricFieldGrowthMetric:
    """Metric 1: fit and compare early electric-field growth rates."""

    def __init__(self, config: EarlyGrowthConfig = EarlyGrowthConfig()) -> None:
        self.config = config

    def _infer_reference_window(self, times: np.ndarray, series: np.ndarray) -> Tuple[float, float]:
        valid = np.isfinite(series) & (series > float(self.config.positive_floor))
        if int(np.count_nonzero(valid)) < int(self.config.min_points):
            raise ValueError("Not enough positive finite samples to infer a fit window")

        times_valid = times[valid]
        series_valid = series[valid]
        maxima = _local_maxima_indices(series_valid, float(self.config.local_maxima_rel_prominence))
        t_a = float(times_valid[0])
        t_b = float(times_valid[maxima[0]]) if maxima.size else float(times_valid[-1])
        if t_b <= t_a:
            raise ValueError("Inferred fit window is empty")
        return t_a, t_b

    def _select_fit_samples(
        self,
        times: np.ndarray,
        series: np.ndarray,
        *,
        time_window: Tuple[float, float],
        sample_selector: SampleSelector,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        t_a, t_b = (float(v) for v in time_window)
        valid = (
            np.isfinite(series)
            & (series > float(self.config.positive_floor))
            & (times >= t_a)
            & (times <= t_b)
        )
        if int(np.count_nonzero(valid)) < int(self.config.min_points):
            raise ValueError("Not enough valid samples in the requested fit window")

        times_window = times[valid]
        series_window = series[valid]
        effective_selector = str(sample_selector)

        if sample_selector == "local_maxima":
            maxima = _local_maxima_indices(series_window, float(self.config.local_maxima_rel_prominence))
            if maxima.size < int(self.config.min_points):
                maxima = _strict_local_maxima_indices(series_window)
            if maxima.size >= int(self.config.min_points):
                return times_window[maxima], series_window[maxima], effective_selector
            if not bool(self.config.maxima_fallback_to_all):
                raise ValueError("Not enough local maxima in the selected fit window")
            effective_selector = "all"

        return times_window, series_window, effective_selector

    def fit(
        self,
        times: np.ndarray,
        series: np.ndarray,
        *,
        time_window: Optional[Tuple[float, float]] = None,
        sample_selector: Optional[SampleSelector] = None,
    ) -> GrowthFitResult:
        """Fit ``gamma_grow`` from a single electric-field-energy trace."""

        times, series = _validate_1d_series(times, series)
        effective_window = (
            tuple(float(v) for v in time_window)
            if time_window is not None
            else self.config.time_window
        )
        if effective_window is None:
            effective_window = self._infer_reference_window(times, series)
        selector = self.config.sample_selector if sample_selector is None else sample_selector

        fit_times, fit_series, effective_selector = self._select_fit_samples(
            times,
            series,
            time_window=effective_window,
            sample_selector=selector,
        )
        slope, intercept = np.polyfit(fit_times, np.log(fit_series), 1)
        return GrowthFitResult(
            gamma_grow=0.5 * float(slope),
            intercept=float(intercept),
            t_a=float(effective_window[0]),
            t_b=float(effective_window[1]),
            num_samples=int(fit_times.size),
            sample_selector=effective_selector,
        )

    def compare(
        self,
        times_theta: np.ndarray,
        series_theta: np.ndarray,
        times_hr: np.ndarray,
        series_hr: np.ndarray,
        *,
        time_window: Optional[Tuple[float, float]] = None,
        sample_selector: Optional[SampleSelector] = None,
    ) -> GrowthComparisonResult:
        """Compare ``gamma_grow^theta`` against ``gamma_grow^HR``."""

        times_theta, series_theta = _validate_1d_series(times_theta, series_theta)
        times_hr, series_hr = _validate_1d_series(times_hr, series_hr)
        effective_window = (
            tuple(float(v) for v in time_window)
            if time_window is not None
            else self.config.time_window
        )
        if effective_window is None:
            effective_window = self._infer_reference_window(times_hr, series_hr)

        fit_theta = self.fit(
            times_theta,
            series_theta,
            time_window=effective_window,
            sample_selector=sample_selector,
        )
        fit_hr = self.fit(
            times_hr,
            series_hr,
            time_window=effective_window,
            sample_selector=sample_selector,
        )
        denom = max(abs(float(fit_hr.gamma_grow)), float(self.config.denom_floor))
        epsilon_grow = abs(float(fit_theta.gamma_grow) - float(fit_hr.gamma_grow)) / denom
        return GrowthComparisonResult(
            gamma_grow_theta=float(fit_theta.gamma_grow),
            gamma_grow_hr=float(fit_hr.gamma_grow),
            epsilon_grow=float(epsilon_grow),
            fit_theta=fit_theta,
            fit_hr=fit_hr,
            t_a=float(effective_window[0]),
            t_b=float(effective_window[1]),
        )
