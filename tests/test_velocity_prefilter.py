"""Independent linear-system checks for the velocity spline optimization.

These checks establish agreement of the two prefilters, not positivity or
accuracy of cubic interpolation for an underresolved kinetic distribution.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_banded

from vpml.solvers.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    build_physical_grid_ops,
    cubic_bspline_interp_constant,
    cubic_bspline_prefilter_constant,
    cubic_bspline_prefilter_constant_batched,
    semilagrangian_vlasov_poisson_step,
)


def _positive_profiles(nv: int, sharp: bool) -> np.ndarray:
    """Five distinct columns, including non-negligible velocity boundary data."""
    v = np.linspace(-1.0, 1.0, nv)
    if sharp:
        profiles = np.full((nv, 5), 1e-8, dtype=np.float64)
        profiles[nv // 2 :, 0] += 1.0
        profiles[nv // 2, 1] += 2.0
        profiles[0, 2] += 0.75
        profiles[-1, 3] += 1.25
        profiles[::2, 4] += 0.5
        return profiles
    return np.stack(
        (
            np.exp(-2.0 * v**2),
            0.4 * np.exp(-3.0 * (v - 0.25) ** 2),
            0.1 + np.exp(-4.0 * (v + 1.0) ** 2),
            0.2 + np.exp(-4.0 * (v - 1.0) ** 2),
            0.3 + 0.1 * np.cos(np.pi * v),
        ),
        axis=1,
    )


def _normalized_residual(coefficients, values, sub, diag, sup):
    """Per-column backward error ||Ac-b||inf / (||A||inf ||c||inf + ||b||inf)."""
    product = diag[:, None] * coefficients
    product[1:] += sub[:, None] * coefficients[:-1]
    product[:-1] += sup[:, None] * coefficients[1:]
    row_sums = np.abs(diag).copy()
    row_sums[1:] += np.abs(sub)
    row_sums[:-1] += np.abs(sup)
    rhs = 6.0 * values
    denominator = (
        np.max(row_sums) * np.max(np.abs(coefficients), axis=0)
        + np.max(np.abs(rhs), axis=0)
    )
    return np.max(np.abs(product - rhs), axis=0) / denominator


class VelocityPrefilterTests(unittest.TestCase):
    def _check_against_independent_solve(self, sharp: bool) -> None:
        for nv in (4, 17, 64):
            values = _positive_profiles(nv, sharp)
            sub = np.ones(nv - 1, dtype=np.float64)
            diag = np.full(nv, 4.0, dtype=np.float64)
            sup = np.ones(nv - 1, dtype=np.float64)
            # SciPy's band convention is ab[u + i - j, j] = A[i, j].
            band = np.zeros((3, nv), dtype=np.float64)
            band[0, 1:] = sup
            band[1] = diag
            band[2, :-1] = sub
            expected = solve_banded((1, 1), band, 6.0 * values)
            for prefilter in (
                cubic_bspline_prefilter_constant,
                cubic_bspline_prefilter_constant_batched,
            ):
                with self.subTest(nv=nv, sharp=sharp, method=prefilter.__name__):
                    actual = np.asarray(
                        jax.jit(prefilter)(
                            jnp.asarray(values), jnp.asarray(sub),
                            jnp.asarray(diag), jnp.asarray(sup),
                        )
                    )
                    self.assertEqual(actual.dtype, np.dtype(np.float64))
                    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)
                    error = _normalized_residual(actual, values, sub, diag, sup)
                    self.assertLess(float(np.max(error)), 10 * np.finfo(np.float64).eps)

                    # The first/last equations have one neighbor, not wraparound.
                    np.testing.assert_allclose(
                        4.0 * actual[0] + actual[1], 6.0 * values[0],
                        rtol=3e-14, atol=3e-15,
                    )
                    np.testing.assert_allclose(
                        actual[-2] + 4.0 * actual[-1], 6.0 * values[-1],
                        rtol=3e-14, atol=3e-15,
                    )
                    coords = np.broadcast_to(np.arange(nv)[:, None], values.shape)
                    reconstructed = cubic_bspline_interp_constant(actual, coords)
                    np.testing.assert_allclose(
                        reconstructed, values, rtol=3e-14, atol=3e-15,
                    )

    def test_positive_smooth_columns_match_scipy_and_have_small_residuals(self):
        self._check_against_independent_solve(sharp=False)

    def test_positive_sharp_columns_and_boundary_rows_match_scipy(self):
        self._check_against_independent_solve(sharp=True)

    def test_short_split_step_distribution_and_field_histories_match(self):
        config = PhysicalGridVlasovPoissonConfig(
            Nx=16, Nv=65, Lx=2.0 * np.pi / 0.4,
            vmin=-6.0, vmax=6.0, dt=0.03, T=0.36,
        )
        phase = 0.4 * config.x
        density = 1.0 + 0.25 * jnp.cos(phase)
        velocity = 0.35 * jnp.sin(phase)
        temperature = 0.85 + 0.2 * jnp.cos(2.0 * phase)
        initial = (
            density[None, :] / jnp.sqrt(2.0 * jnp.pi * temperature[None, :])
            * jnp.exp(
                -0.5 * (config.v[:, None] - velocity[None, :]) ** 2
                / temperature[None, :]
            )
        )

        def evolve(batched: bool):
            ops = build_physical_grid_ops(config)
            ops["batched_v_prefilter"] = batched

            def step(distribution, _):
                updated, electric = semilagrangian_vlasov_poisson_step(
                    config, distribution, ops=ops,
                )
                return updated, (updated, electric)

            return jax.jit(lambda f: jax.lax.scan(step, f, None, length=12))(initial)

        final_original, history_original = evolve(False)
        final_batched, history_batched = evolve(True)
        for original, batched in (
            (final_original, final_batched),
            *zip(history_original, history_batched),
        ):
            self.assertEqual(np.asarray(batched).dtype, np.dtype(np.float64))
            self.assertTrue(np.isfinite(np.asarray(batched)).all())
            np.testing.assert_allclose(batched, original, rtol=2e-13, atol=3e-15)


if __name__ == "__main__":
    unittest.main()
