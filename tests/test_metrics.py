import unittest

import numpy as np

from vpml import (
    EarlyElectricFieldGrowthMetric,
    EarlyGrowthConfig,
    FieldErrorConfig,
    SelfGeneratedFieldErrorMetric,
)


class EarlyGrowthMetricTests(unittest.TestCase):
    def test_exact_exponential_growth_recovers_gamma_and_zero_relative_error(self) -> None:
        times = np.linspace(0.0, 5.0, 101)
        gamma = 0.35
        series = np.exp(1.2 + 2.0 * gamma * times)
        metric = EarlyElectricFieldGrowthMetric(
            EarlyGrowthConfig(time_window=(0.0, 5.0))
        )

        fit = metric.fit(times, series)
        comp = metric.compare(times, series, times, series)

        self.assertAlmostEqual(fit.gamma_grow, gamma, places=12)
        self.assertAlmostEqual(comp.gamma_grow_theta, gamma, places=12)
        self.assertAlmostEqual(comp.gamma_grow_hr, gamma, places=12)
        self.assertAlmostEqual(comp.epsilon_grow, 0.0, places=12)

    def test_manual_fit_window_changes_recovered_growth_rate(self) -> None:
        times = np.linspace(0.0, 10.0, 401)
        gamma_early = 0.15
        gamma_late = 0.45
        series = np.empty_like(times)
        early = times <= 5.0
        series[early] = np.exp(2.0 * gamma_early * times[early])
        series[~early] = np.exp(2.0 * gamma_early * 5.0) * np.exp(2.0 * gamma_late * (times[~early] - 5.0))

        metric_early = EarlyElectricFieldGrowthMetric(EarlyGrowthConfig(time_window=(0.0, 4.0)))
        metric_late = EarlyElectricFieldGrowthMetric(EarlyGrowthConfig(time_window=(6.0, 10.0)))

        fit_early = metric_early.fit(times, series)
        fit_late = metric_late.fit(times, series)

        self.assertAlmostEqual(fit_early.gamma_grow, gamma_early, places=3)
        self.assertAlmostEqual(fit_late.gamma_grow, gamma_late, places=3)
        self.assertGreater(fit_late.gamma_grow, fit_early.gamma_grow)

    def test_local_maxima_selector_recovers_oscillatory_growth_envelope(self) -> None:
        times = np.linspace(0.0, 6.0, 2001)
        gamma = 0.25
        series = np.exp(2.0 * gamma * times) * (1.0 + 0.2 * np.sin(4.0 * np.pi * times)) ** 2
        metric = EarlyElectricFieldGrowthMetric(
            EarlyGrowthConfig(
                time_window=(0.0, 6.0),
                sample_selector="local_maxima",
                local_maxima_rel_prominence=1e-3,
            )
        )

        fit = metric.fit(times, series)

        self.assertEqual(fit.sample_selector, "local_maxima")
        self.assertGreaterEqual(fit.num_samples, 2)
        self.assertAlmostEqual(fit.gamma_grow, gamma, delta=2e-2)


class FieldErrorMetricTests(unittest.TestCase):
    def test_identical_fourier_histories_have_zero_field_error(self) -> None:
        times = np.linspace(0.0, 2.0, 9)
        k = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        E_hat = np.zeros((times.size, k.size), dtype=np.complex128)
        E_hat[:, 1] = 1.0 + (0.5 - 0.25j) * times
        E_hat[:, 2] = -0.75j * (1.0 + times)
        metric = SelfGeneratedFieldErrorMetric()

        result = metric.evaluate_fourier(times, E_hat, k, times, E_hat, k)

        self.assertAlmostEqual(result.epsilon_E, 0.0, places=12)
        self.assertEqual(result.num_modes, 2)
        np.testing.assert_allclose(result.selected_k, np.array([1.0, 2.0]))

    def test_scaled_fourier_history_has_expected_relative_error(self) -> None:
        times = np.linspace(0.0, 2.0, 21)
        k = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        E_hat_hr = np.zeros((times.size, k.size), dtype=np.complex128)
        E_hat_hr[:, 1] = 2.0 - 0.5j * times
        E_hat_hr[:, 2] = 1.0 + (0.25 + 0.75j) * times
        E_hat_theta = 1.5 * E_hat_hr
        metric = SelfGeneratedFieldErrorMetric()

        result = metric.evaluate_fourier(times, E_hat_theta, k, times, E_hat_hr, k)

        self.assertAlmostEqual(result.epsilon_E, 0.5, places=12)

    def test_fourier_metric_handles_different_time_steps_via_interpolation(self) -> None:
        times_theta = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        times_hr = np.linspace(0.0, 3.0, 13)
        k = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        def analytic(times: np.ndarray) -> np.ndarray:
            out = np.zeros((times.size, k.size), dtype=np.complex128)
            out[:, 1] = 1.0 + (0.2 - 0.1j) * times
            out[:, 2] = -0.5 + (0.4 + 0.25j) * times
            return out

        metric = SelfGeneratedFieldErrorMetric()
        result = metric.evaluate_fourier(
            times_theta,
            analytic(times_theta),
            k,
            times_hr,
            analytic(times_hr),
            k,
        )

        self.assertAlmostEqual(result.epsilon_E, 0.0, places=12)

    def test_same_grid_physical_and_fourier_paths_agree(self) -> None:
        Nx = 8
        Lx = 2.0 * np.pi
        dx = Lx / Nx
        x = np.arange(Nx, dtype=np.float64) * dx
        times = np.linspace(0.0, 2.0, 17)
        k = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=dx)

        E_hat_hr = np.zeros((times.size, k.size), dtype=np.complex128)
        E_hat_hr[:, 1] = 1.0 + 0.25 * times
        E_hat_hr[:, 2] = 0.5 - 0.1j * times
        E_hat_theta = 1.1 * E_hat_hr

        E_hr = np.fft.irfft(E_hat_hr, n=Nx, axis=1)
        E_theta = np.fft.irfft(E_hat_theta, n=Nx, axis=1)

        metric = SelfGeneratedFieldErrorMetric(
            FieldErrorConfig(num_low_modes=2)
        )
        fourier_result = metric.evaluate_fourier(times, E_hat_theta, k, times, E_hat_hr, k)
        physical_result = metric.evaluate_physical(times, E_theta, x, times, E_hr, x)

        self.assertAlmostEqual(fourier_result.epsilon_E, physical_result.epsilon_E, places=12)

    def test_divergent_fourier_history_returns_infinite_error_not_nan(self) -> None:
        times = np.linspace(0.0, 2.0, 9)
        k = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        E_hat_hr = np.zeros((times.size, k.size), dtype=np.complex128)
        E_hat_hr[:, 1] = 1.0 + 0.1 * times
        E_hat_hr[:, 2] = 0.5 - 0.2j * times
        E_hat_theta = E_hat_hr.copy()
        E_hat_theta[-1, 1] = 1e309 + 0.0j

        metric = SelfGeneratedFieldErrorMetric()
        result = metric.evaluate_fourier(times, E_hat_theta, k, times, E_hat_hr, k)

        self.assertTrue(np.isinf(result.epsilon_E))


if __name__ == "__main__":
    unittest.main()
