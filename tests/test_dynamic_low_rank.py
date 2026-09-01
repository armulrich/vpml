import unittest

import jax
import jax.numpy as jnp
import numpy as np

from vpml.dynamic_low_rank import (
    compress_macro_micro_state,
    positive_density_preserving_projection,
    randomized_low_rank_approximation,
    raw_velocity_moments,
    restore_first_three_moments,
)
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    build_physical_grid_ops,
    gaussian_pdf,
    normalize_density_on_grid,
    run_semilagrangian_vlasov_poisson,
    semilagrangian_vlasov_poisson_step,
)


class DynamicLowRankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PhysicalGridVlasovPoissonConfig(
            Nx=16,
            Nv=32,
            Lx=4.0 * np.pi,
            vmin=-6.0,
            vmax=6.0,
            dt=0.01,
            T=0.01,
        )
        equilibrium = normalize_density_on_grid(
            gaussian_pdf(self.config.v, mean=0.0, sigma=1.0), self.config.v
        )
        self.state = equilibrium[:, None] * (
            1.0
            + 0.08 * jnp.cos(0.5 * self.config.x)[None]
            + 0.02
            * self.config.v[:, None]
            * jnp.sin(self.config.x)[None]
        )

    def test_public_step_matches_one_step_runner(self) -> None:
        expected = run_semilagrangian_vlasov_poisson(
            self.config, self.state, return_final_state=True
        )["final_state"]
        actual, _ = semilagrangian_vlasov_poisson_step(
            self.config,
            self.state,
            ops=build_physical_grid_ops(self.config),
        )
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-15, atol=5e-16)

    def test_randomized_approximation_recovers_exact_low_rank_matrix(self) -> None:
        left = jax.random.normal(jax.random.PRNGKey(1), (32, 3), dtype=jnp.float64)
        right = jax.random.normal(jax.random.PRNGKey(2), (3, 16), dtype=jnp.float64)
        matrix = left @ right
        recovered = randomized_low_rank_approximation(
            matrix, 3, jax.random.PRNGKey(3), oversample=3, power_iterations=1
        )
        np.testing.assert_allclose(np.asarray(recovered), np.asarray(matrix), rtol=2e-11, atol=2e-11)

    def test_moment_restoration_matches_first_three_raw_moments(self) -> None:
        target = raw_velocity_moments(self.state, self.config.v)
        approximation = 0.97 * self.state
        restored = restore_first_three_moments(
            approximation, target, self.config.v
        )
        np.testing.assert_allclose(
            np.asarray(raw_velocity_moments(restored, self.config.v)),
            np.asarray(target),
            rtol=2e-13,
            atol=2e-13,
        )

    def test_macro_micro_compression_is_positive_and_moment_preserving(self) -> None:
        compressed, diagnostics = compress_macro_micro_state(
            self.state,
            self.config.v,
            rank=6,
            key=jax.random.PRNGKey(4),
            oversample=4,
            power_iterations=1,
        )
        np.testing.assert_allclose(
            np.asarray(raw_velocity_moments(compressed, self.config.v)),
            np.asarray(raw_velocity_moments(self.state, self.config.v)),
            rtol=2e-12,
            atol=2e-12,
        )
        self.assertGreaterEqual(float(jnp.min(compressed)), -1e-14)
        self.assertLess(float(diagnostics["maximum_relative_moment_error"]), 2e-11)

    def test_positive_projection_preserves_density(self) -> None:
        values = self.state.at[0, 0].set(-1e-8)
        before = np.asarray(jnp.trapezoid(values, x=self.config.v, axis=0))
        positive = positive_density_preserving_projection(values, self.config.v)
        after = np.asarray(jnp.trapezoid(positive, x=self.config.v, axis=0))
        self.assertGreaterEqual(float(jnp.min(positive)), 0.0)
        np.testing.assert_allclose(after, before, rtol=2e-14, atol=2e-14)


if __name__ == "__main__":
    unittest.main()
