"""Small numerical checks for Gaussian references, without evolving a dataset."""

import numpy as np
import unittest

from scripts.gaussian_dataset_pilot import Runner, realization


def check_initial_field_constraints_and_spectral_normalization(strength):
    n, u, temperature, coefficients = realization(28001, 64, strength)
    assert np.min(n) > 0.0
    assert np.min(temperature) > 0.0
    np.testing.assert_allclose(n.mean(), 1.0, atol=2e-15, rtol=0)
    np.testing.assert_allclose(temperature.mean(), 1.0, atol=2e-15, rtol=0)
    np.testing.assert_allclose(np.mean(n * u), 0.0, atol=2e-15, rtol=0)
    # Parseval's identity for these real sine/cosine coefficients.
    np.testing.assert_allclose(
        0.5 * np.sum(coefficients**2, axis=(1, 2)), 1.0,
        atol=2e-15, rtol=0,
    )
    if strength == 0:
        np.testing.assert_array_equal(n, np.ones_like(n))
        np.testing.assert_array_equal(u, np.zeros_like(u))
        np.testing.assert_array_equal(temperature, np.ones_like(temperature))


def check_same_realization_agrees_at_common_nodes_across_resolutions():
    coarse = realization(28003, 64, 1.0)
    fine = realization(28003, 256, 1.0)
    np.testing.assert_array_equal(coarse[3], fine[3])
    for field_coarse, field_fine in zip(coarse[:3], fine[:3]):
        np.testing.assert_allclose(field_coarse, field_fine[::4], rtol=2e-14, atol=2e-14)
    other = realization(28004, 64, 1.0)
    assert not np.allclose(coarse[3], other[3])


def check_initial_gaussian_moments_and_zero_central_heat_flux(moment_runner):
    runner = moment_runner
    n, u, temperature, _ = realization(28002, runner.cfg.Nx, 1.0)
    v = np.asarray(runner.cfg.v)[:, None]
    f0 = n[None, :] / np.sqrt(2 * np.pi * temperature[None, :]) * np.exp(
        -(v - u[None, :])**2 / (2 * temperature[None, :])
    )
    actual = np.asarray(runner.moments(f0))
    np.testing.assert_allclose(actual[0], n, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual[1], u, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual[2], n * temperature, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual[3:], 0.0, atol=1e-10, rtol=0)


def check_nonzero_heat_flux_and_its_spatial_derivative(moment_runner):
    # A positive mixture of two unit-variance Gaussians has known central
    # moments. This detects sign/factor errors that Q=0 cannot reveal.
    runner = moment_runner
    v = np.asarray(runner.cfg.v)[:, None]
    x = np.asarray(runner.cfg.x)
    k = 2 * np.pi / runner.cfg.Lx
    weight = 0.3 + 0.1 * np.sin(k * x)
    derivative = 0.1 * k * np.cos(k * x)
    shift = 0.6
    f = (
        weight[None, :] * np.exp(-0.5 * (v - shift)**2)
        + (1 - weight[None, :]) * np.exp(-0.5 * v**2)
    ) / np.sqrt(2 * np.pi)
    expected = np.stack([
        np.ones_like(weight),
        weight * shift,
        1 + weight * (1 - weight) * shift**2,
        weight * (1 - weight) * (1 - 2 * weight) * shift**3,
        (1 - 6 * weight + 6 * weight**2) * derivative * shift**3,
    ])
    np.testing.assert_allclose(np.asarray(runner.moments(f)), expected, rtol=1e-10, atol=1e-11)


def check_tangent_map_matches_centered_finite_difference():
    # Only one step on a tiny grid; no trajectory generation is needed.
    runner = Runner(nx=16, nv=65, dt=0.02)
    equilibrium = np.asarray(runner.equilibrium)
    n, u, temperature, _ = realization(28001, runner.cfg.Nx, 0.7)
    v = np.asarray(runner.cfg.v)[:, None]
    f0 = n[None, :] / np.sqrt(2 * np.pi * temperature[None, :]) * np.exp(
        -(v - u[None, :])**2 / (2 * temperature[None, :])
    )
    direction = f0 - equilibrium
    tangent = np.asarray(runner.linear_step(direction))
    epsilon = 1e-4
    finite_difference = (
        np.asarray(runner.full_step(equilibrium + epsilon * direction))
        - np.asarray(runner.full_step(equilibrium - epsilon * direction))
    ) / (2 * epsilon)
    relative_error = np.linalg.norm(finite_difference - tangent) / np.linalg.norm(tangent)
    assert relative_error < 1e-6
    np.testing.assert_allclose(
        np.asarray(runner.full_step(equilibrium)), equilibrium,
        rtol=0, atol=2e-13,
    )


class GaussianDatasetPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.moment_runner = Runner(nx=32, nv=257, dt=0.02, vmax=10.0)

    def test_initial_field_constraints(self):
        for strength in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
            with self.subTest(strength=strength):
                check_initial_field_constraints_and_spectral_normalization(strength)

    def test_cross_resolution(self):
        check_same_realization_agrees_at_common_nodes_across_resolutions()

    def test_gaussian_moments(self):
        check_initial_gaussian_moments_and_zero_central_heat_flux(self.moment_runner)

    def test_nonzero_heat_flux_derivative(self):
        check_nonzero_heat_flux_and_its_spatial_derivative(self.moment_runner)

    def test_tangent_finite_difference(self):
        check_tangent_map_matches_centered_finite_difference()


if __name__ == "__main__":
    unittest.main()
