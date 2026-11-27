"""Analytical checks for the portable Gaussian initial conditions."""
import unittest

import numpy as np

from vpml.datasets.initial_conditions import (
    draw_coefficients, fields_from_coefficients, gaussian_distribution,
)


class InitialConditionTests(unittest.TestCase):
    def test_sampling_and_physical_constraints(self):
        coefficients = draw_coefficients(28001, 5, 1.2)
        self.assertEqual(coefficients.dtype, np.float64)
        np.testing.assert_allclose(0.5 * np.sum(coefficients ** 2, axis=(1, 2)), 1, atol=2e-15)
        np.testing.assert_array_equal(coefficients, draw_coefficients(28001, 5, 1.2))
        self.assertFalse(np.array_equal(coefficients, draw_coefficients(28002, 5, 1.2)))
        for strength in (0, 0.1, 0.3, 0.5, 0.7, 1):
            n, u, temperature = fields_from_coefficients(coefficients, 64, strength)
            self.assertTrue(np.all(n > 0) and np.all(temperature > 0))
            np.testing.assert_allclose([n.mean(), temperature.mean(), np.mean(n * u)], [1, 1, 0], atol=2e-15)
            if strength == 0:
                np.testing.assert_array_equal(n, np.ones(64))
                np.testing.assert_array_equal(u, np.zeros(64))

    def test_common_spatial_nodes_and_gaussian_moments(self):
        coefficients = draw_coefficients(28003, 3, 2.0)
        fields = fields_from_coefficients(coefficients, 64, 1)
        refined = fields_from_coefficients(coefficients, 256, 1)
        for coarse, fine in zip(fields, refined):
            np.testing.assert_allclose(coarse, fine[::4], atol=2e-14, rtol=2e-14)
        n, u, temperature = fields
        v = np.linspace(-12, 12, 2049)
        f = gaussian_distribution(n, u, temperature, v)
        central = v[:, None] - u
        np.testing.assert_allclose(np.trapezoid(f, x=v, axis=0), n, atol=2e-14)
        np.testing.assert_allclose(np.trapezoid(f * v[:, None], x=v, axis=0), n * u, atol=2e-14)
        np.testing.assert_allclose(np.trapezoid(f * central ** 2, x=v, axis=0), n * temperature, atol=2e-14)
        np.testing.assert_allclose(np.trapezoid(f * central ** 3, x=v, axis=0), 0, atol=2e-14)

    def test_rejects_invalid_or_unresolved_fields(self):
        coefficients = draw_coefficients(1, 5, 1.2)
        for args in [(True, 3, 1.2), (1, 0, 1.2), (1, 3, np.nan)]:
            with self.assertRaises(ValueError):
                draw_coefficients(*args)
        for nx, strength in [(8, 1), (64, np.inf), (64, -1), (64, 1e6)]:
            with self.assertRaises(ValueError):
                fields_from_coefficients(coefficients, nx, strength)
        with self.assertRaises(ValueError):
            gaussian_distribution([1], [0], [0], [-1, 1])
        with self.assertRaises(ValueError):
            gaussian_distribution([1], [0], [1], [1, -1])


if __name__ == "__main__":
    unittest.main()
