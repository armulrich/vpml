import unittest

import jax
import jax.numpy as jnp
import numpy as np

from vpml.physical_grid import (
    build_cubic_spline_hermite_projection_matrix,
    cubic_bspline_velocity_resampling_matrix,
    project_distribution_snapshot_to_fourier_hermite,
    project_distribution_snapshot_with_hermite_matrix,
    trapezoid_quadrature_weights,
)


jax.config.update("jax_enable_x64", True)


class ProjectionQuadratureTests(unittest.TestCase):
    def test_trapezoid_weights_match_direct_integration(self) -> None:
        v = jnp.array([-2.0, -0.5, 0.25, 1.5, 3.0], dtype=jnp.float64)
        values = jnp.array([0.3, -0.2, 1.4, 0.7, -0.1], dtype=jnp.float64)
        weighted = jnp.sum(trapezoid_quadrature_weights(v) * values)
        direct = jnp.trapezoid(values, x=v)
        self.assertAlmostEqual(float(weighted), float(direct), places=14)

    def test_resampling_matrix_reproduces_native_spline_nodes(self) -> None:
        v = jnp.linspace(-6.0, 6.0, 32, dtype=jnp.float64)
        matrix = np.asarray(cubic_bspline_velocity_resampling_matrix(v, v))
        np.testing.assert_allclose(matrix, np.eye(32), rtol=1e-12, atol=1e-12)

    def test_native_projection_matrix_matches_direct_projection(self) -> None:
        v = jnp.linspace(-6.0, 6.0, 32, dtype=jnp.float64)
        x = np.linspace(0.0, 4.0 * np.pi, 24, endpoint=False)
        equilibrium = np.exp(-0.5 * np.asarray(v) ** 2) / np.sqrt(2.0 * np.pi)
        f_phys = equilibrium[:, None] * (
            1.0 + 0.1 * np.cos(0.5 * x)[None, :]
        )
        direct = project_distribution_snapshot_to_fourier_hermite(
            f_phys,
            v,
            9,
            equilibrium=equilibrium,
        )
        projection_matrix = build_cubic_spline_hermite_projection_matrix(
            v,
            9,
            32,
        )
        matrix_result = project_distribution_snapshot_with_hermite_matrix(
            f_phys,
            projection_matrix,
            equilibrium=equilibrium,
        )
        np.testing.assert_allclose(
            np.asarray(matrix_result),
            np.asarray(direct),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_equilibrium_subtraction_remains_exact_after_refinement(self) -> None:
        v = jnp.linspace(-8.0, 8.0, 48, dtype=jnp.float64)
        equilibrium = np.exp(-0.5 * np.asarray(v) ** 2) / np.sqrt(2.0 * np.pi)
        f_phys = np.repeat(equilibrium[:, None], 12, axis=1)
        projection_matrix = build_cubic_spline_hermite_projection_matrix(
            v,
            17,
            192,
        )
        projected = project_distribution_snapshot_with_hermite_matrix(
            f_phys,
            projection_matrix,
            equilibrium=equilibrium,
        )
        np.testing.assert_array_equal(
            np.asarray(projected),
            np.zeros((17, 7), dtype=np.complex128),
        )


if __name__ == "__main__":
    unittest.main()
