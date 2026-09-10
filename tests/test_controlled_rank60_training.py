import unittest

import jax.numpy as jnp
import numpy as np

from model.train.expert_balanced_metric import balanced_proposal
from model.train.random_september_experts import initialize


class ControlledRank60TrainingTests(unittest.TestCase):
    def config(self):
        return {
            "latent_rank": 60,
            "latent_readout_mode": "equilibrium_cnn",
            "width": 8,
            "depth": 1,
            "kernel_size": 5,
            "operator_modes": 3,
            "operator_rank": 2,
            "operator_output_init_scale": 1e-6,
            "seed": 1729,
        }

    def test_random_experts_are_distinct_and_zero_mean(self):
        params, _ = initialize(self.config())
        for key in ("expert_output_kernel", "expert_cyclic_output_kernel"):
            values = np.asarray(params[key])
            self.assertGreater(float(np.linalg.norm(values[0] - values[1])), 0.0)
            np.testing.assert_allclose(values.mean(axis=0), 0.0, atol=1e-20)

    def test_specialist_router_is_balanced_with_expert_group(self):
        gradient = {
            "shared": jnp.asarray([2.0]),
            "expert_output_kernel": jnp.asarray([1.0]),
            "specialist_router_gate_bias": jnp.asarray([1.0]),
        }
        metric = {key: (1.0, 1.0) for key in gradient}
        direction, _, _ = balanced_proposal(gradient, 1.0, metric)
        self.assertAlmostEqual(float(direction["expert_output_kernel"][0]),
                               float(direction["specialist_router_gate_bias"][0]))


if __name__ == "__main__":
    unittest.main()
