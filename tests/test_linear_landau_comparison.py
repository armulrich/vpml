import unittest

import numpy as np

from vpml.diagnostics.linear_landau_comparison import (
    fields,
    solve_modes,
)


class LinearLandauComparisonTests(unittest.TestCase):
    def test_initial_density_pressure_and_poisson_sign(self):
        epsilon, k = .001, .43
        result = solve_modes([k], nv=257, final_time=0., epsilon=epsilon)
        actual = fields(result, 0, nx=64)
        theta = 2 * np.pi * np.arange(64) / 64
        density = 1 + epsilon * np.cos(theta)
        np.testing.assert_allclose(actual["n"][0], density, rtol=0, atol=1e-13)
        np.testing.assert_allclose(actual["p"][0], density, rtol=0, atol=2e-12)
        np.testing.assert_allclose(actual["E"][0], -epsilon * np.sin(theta) / k,
                                   rtol=0, atol=1e-14)
        for q in ("u", "Q", "g"):
            np.testing.assert_allclose(actual[q], 0., rtol=0, atol=1e-13)

    def test_initial_density_acceleration_has_restoring_field_sign(self):
        k = .43
        tiny = solve_modes([k], nv=257, dt=1e-4, save_dt=1e-4, final_time=1e-4)
        n0, n1 = tiny["mode_moments"][:, 0, 0].real
        measured = 2 * (n1 - n0) / (1e-4)**2 / n0
        np.testing.assert_allclose(measured, -(1 + k**2), rtol=1e-6, atol=1e-6)

    def test_unperturbed_maxwellian_stays_stationary(self):
        solution = solve_modes([.3, .6], nv=129, final_time=.2, epsilon=0.)
        for index in range(2):
            actual = fields(solution, index, nx=32)
            for q in ("n", "p"):
                np.testing.assert_allclose(actual[q], 1., rtol=0, atol=1e-12)
            for q in ("u", "Q", "g", "E"):
                np.testing.assert_allclose(actual[q], 0., rtol=0, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
