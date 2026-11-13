"""Analytical checks for moment meanings and spectral output restriction."""

import unittest

import numpy as np

from vpml.datasets.observations import moment_fields, raw_to_primitives, restrict_fourier


class ObservationTests(unittest.TestCase):
    def test_fourier_restriction_preserves_output_nyquist_cosine(self):
        x = 2 * np.pi * np.arange(32) / 32
        values = 2 + np.cos(4 * x) + 0.3 * np.sin(3 * x) + 0.7 * np.cos(7 * x)
        y = 2 * np.pi * np.arange(8) / 8
        expected = 2 + np.cos(4 * y) + 0.3 * np.sin(3 * y)
        np.testing.assert_allclose(restrict_fourier(values, 8), expected, atol=2e-14, rtol=0)
        np.testing.assert_array_equal(restrict_fourier(values, 32), values)

    def test_mixture_has_known_nonzero_central_heat_flux(self):
        nx = 32
        length = 4 * np.pi
        x = length * np.arange(nx) / nx
        v = np.linspace(-10, 10, 257)
        k = 2 * np.pi / length
        weight = 0.3 + 0.1 * np.sin(k * x)
        derivative = 0.1 * k * np.cos(k * x)
        shift = 0.6
        f = (weight * np.exp(-0.5 * (v[:, None] - shift) ** 2)
             + (1 - weight) * np.exp(-0.5 * v[:, None] ** 2)) / np.sqrt(2 * np.pi)
        fields, raw = moment_fields(f, v, length)
        expected = np.stack((
            np.ones(nx), weight * shift, 1 + weight * (1 - weight) * shift ** 2,
            weight * (1 - weight) * (1 - 2 * weight) * shift ** 3,
            (1 - 6 * weight + 6 * weight ** 2) * derivative * shift ** 3,
        ))
        np.testing.assert_allclose(fields, expected, atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(raw_to_primitives(raw), fields[:3], atol=1e-13, rtol=1e-13)

    def test_filter_and_primitive_conversion_are_not_interchangeable(self):
        x = 2 * np.pi * np.arange(64) / 64
        density = 1 + 0.5 * np.cos(x)
        velocity = 0.4 * np.cos(3 * x)
        raw = np.stack((density, density * velocity, density * (1 + velocity ** 2)))
        desired = restrict_fourier(raw_to_primitives(raw), 8)
        converted_after_filter = raw_to_primitives(restrict_fourier(raw, 8))
        self.assertGreater(np.max(np.abs(desired - converted_after_filter)), 1e-3)

    def test_invalid_projection_size_is_rejected(self):
        for size in (3, 0, 64, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                restrict_fourier(np.ones(32), size)


if __name__ == "__main__":
    unittest.main()
