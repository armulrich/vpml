import unittest

import numpy as np

from vpml.metrics.trajectory import (
    envelope_diagnostics, latent_saturation_diagnostics, signed_hermite_flux,
    trailing_mean, matched_cadence_growth,
)


class TrajectoryMetricsTest(unittest.TestCase):
    def test_matching_samples_isolates_rate_estimator_cadence_bias(self):
        from vpml.metrics import EarlyElectricFieldGrowthMetric, EarlyGrowthConfig
        fine = np.arange(301) * .01
        coarse = np.arange(31) * .1
        energy = np.exp(-.5 * fine) * (1 + .4 * np.cos(4 * fine))
        metric = EarlyElectricFieldGrowthMetric(EarlyGrowthConfig(sample_selector="local_maxima"))
        window = (0.0, 1.51)
        model_rate = metric.fit(coarse, np.interp(coarse, fine, energy), time_window=window).gamma_grow
        native_rate = metric.fit(fine, energy, time_window=window).gamma_grow
        self.assertGreater(abs(model_rate - native_rate), 1e-5)
        result = matched_cadence_growth(coarse, model_rate, fine, energy, fit_window=window)
        self.assertEqual(result["epsilon_grow_matched_cadence"], 0)

    def test_no_boundary_padding_or_false_growth(self):
        np.testing.assert_array_equal(trailing_mean(np.ones(21), 5), np.ones(17))
        result = envelope_diagnostics(np.ones(201), np.ones(201), cadence=0.1)
        self.assertEqual(result["log_envelope_rmse"], 0)
        self.assertEqual(result["signed_log_change_rmse"], 0)

    def test_levels_and_rates_are_distinct_and_cover_multiple_rebounds(self):
        t = np.arange(1201) * 0.1
        reference = np.exp(-0.015 * t + np.sin(t / 5))
        matched = envelope_diagnostics(reference, reference, cadence=0.1)
        self.assertEqual(matched["log_envelope_rmse"], 0)
        scaled = envelope_diagnostics(2 * reference, reference, cadence=0.1)
        self.assertAlmostEqual(scaled["log_envelope_rmse"], np.log(2), places=6)
        self.assertLess(scaled["signed_log_change_rmse"], 1e-6)
        lost = envelope_diagnostics(np.exp(-0.15 * t), reference, cadence=0.1)
        self.assertGreater(lost["signed_log_change_rmse"], 0.1)

    def test_nonfinite_fails_and_noise_floor_is_visible(self):
        result = envelope_diagnostics(np.full(201, np.nan), np.ones(201), cadence=0.1)
        self.assertFalse(result["finite"])
        self.assertIsNone(result["log_envelope_rmse"])
        zero = envelope_diagnostics(np.zeros(201), np.zeros(201), cadence=0.1)
        self.assertEqual(zero["reference_floor_fraction"], 1)
        self.assertEqual(zero["log_envelope_rmse"], 0)

    def test_bound_40_is_detected_without_above_96_excursions(self):
        values = np.zeros((10, 2, 4))
        values[3, 1, 0] = -40
        result = latent_saturation_diagnostics(values, bound=40, cadence=0.1)
        self.assertAlmostEqual(result["first_saved_saturation_time"], 0.3)
        self.assertEqual(result["saved_state_saturation_fraction"], 1 / 80)
        unbounded = latent_saturation_diagnostics(values, bound=0, cadence=0.1)
        self.assertIsNone(unbounded["first_saved_saturation_time"])

    def test_signed_flux_matches_direct_streaming_energy_balance(self):
        x = np.arange(32) * (2 * np.pi / 32)
        lower, upper = np.cos(x), np.sin(x)
        coefficients = np.stack((lower, upper))
        wave = np.fft.rfftfreq(x.size, d=1 / x.size)
        flux = signed_hermite_flux(coefficients, wave, first_order=3)[0]
        # dC3/dt=-sqrt(4)*d_x(C4); lost lower energy is outward flux.
        direct_loss = -2 * np.mean(lower * (-2 * np.cos(x)))
        self.assertAlmostEqual(flux, direct_loss)
        reverse = signed_hermite_flux(np.stack((lower, -upper)), wave)[0]
        self.assertAlmostEqual(reverse, -flux)


if __name__ == "__main__":
    unittest.main()
