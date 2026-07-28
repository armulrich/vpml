import unittest

import jax.numpy as jnp
import numpy as np

from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    compute_electric_field_from_distribution,
    gaussian_pdf,
    normalize_density_on_grid,
    run_semilagrangian_vlasov_poisson,
)


class PhysicalGridFieldHistoryTests(unittest.TestCase):
    def test_detached_field_history_matches_recomputed_fields(self) -> None:
        config = PhysicalGridVlasovPoissonConfig(
            Nx=8,
            Nv=16,
            Lx=4.0 * np.pi,
            vmin=-6.0,
            vmax=6.0,
            dt=0.01,
            T=0.02,
        )
        equilibrium = normalize_density_on_grid(
            gaussian_pdf(config.v, mean=0.0, sigma=1.0),
            config.v,
        )
        perturbation = 0.01 * jnp.cos(0.5 * config.x)
        f0 = equilibrium[:, None] * (1.0 + perturbation[None, :])
        raw = run_semilagrangian_vlasov_poisson(
            config,
            f0,
            return_state_history=True,
            return_field_history=True,
        )
        expected = np.stack(
            [
                np.fft.rfft(
                    np.asarray(
                        compute_electric_field_from_distribution(state, config),
                        dtype=np.float64,
                    )
                )
                for state in np.asarray(raw["state_history"])
            ],
            axis=0,
        )
        np.testing.assert_allclose(
            np.asarray(raw["E_hat_hist"]),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
