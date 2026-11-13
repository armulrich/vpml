"""Small CPU checks of the portable worker against the unchanged solver."""

import unittest

import jax
import numpy as np

from vpml.datasets.evolution import ReferenceEvolution
from vpml.datasets.initial_conditions import draw_coefficients
from vpml.datasets.manifest import Numerics
from vpml.solvers.physical_grid import semilagrangian_vlasov_poisson_step


AMPLITUDES = {"density": 0.18, "velocity": 0.25, "temperature": 0.10}


def make_evolution(strength=0.3, seed=28001, linearized=False):
    numerics = Numerics(nx=16, nv=65, dt=0.01, final_time=0.04,
                        output_nx=8, save_dt=0.02, block_steps=2)
    case = {"k0": 0.4, "strength": strength,
            "coefficients": draw_coefficients(seed, 3, 1.2).tolist()}
    return ReferenceEvolution(case, numerics, AMPLITUDES, linearized=linearized)


class EvolutionTests(unittest.TestCase):
    def test_block_matches_original_step_and_time_order(self):
        runner = make_evolution()
        initial = runner.initial_state()
        end, history = runner.advance(initial, 2)
        first, _ = semilagrangian_vlasov_poisson_step(runner.cfg, initial["f"], ops=runner.ops)
        second, _ = semilagrangian_vlasov_poisson_step(runner.cfg, first, ops=runner.ops)
        np.testing.assert_allclose(end["f"], second, atol=1e-13, rtol=1e-13)
        for index, state in enumerate(({"f": first}, {"f": second})):
            expected = runner.observe(state)
            for key in history:
                np.testing.assert_allclose(history[key][index], expected[key], atol=1e-12, rtol=1e-12)
        self.assertEqual(history["fields"].shape, (2, 5, 8))
        self.assertEqual(history["raw_moments"].shape, (2, 3, 8))
        self.assertEqual(set(history), {"fields", "raw_moments", "electric"})
        self.assertEqual(end["f"].dtype, np.dtype("float64"))

    def test_zero_strength_is_stationary(self):
        runner = make_evolution(strength=0)
        initial = runner.initial_state()
        end, history = runner.advance(initial, 2)
        np.testing.assert_allclose(end["f"], initial["f"], atol=2e-13, rtol=0)
        np.testing.assert_allclose(history["electric"], 0, atol=1e-14, rtol=0)

    def test_compilation_functions_are_reused_across_initial_conditions(self):
        first = make_evolution(strength=0.1)
        second = make_evolution(strength=1.0, seed=28002)
        self.assertIs(first._advance, second._advance)
        self.assertIs(first._observe, second._observe)
        self.assertFalse(np.array_equal(first.initial_state()["f"], second.initial_state()["f"]))

    def test_optional_linearized_state_uses_tangent_map(self):
        runner = make_evolution(strength=0.1, linearized=True)
        initial = runner.initial_state()
        end, history = runner.advance(initial, 1)
        def full_step(f):
            return semilagrangian_vlasov_poisson_step(runner.cfg, f, ops=runner.ops)[0]
        expected = jax.jvp(full_step, (runner.equilibrium,), (initial["h"],))[1]
        np.testing.assert_allclose(end["h"], expected, atol=1e-13, rtol=1e-13)
        self.assertEqual(history["linear_electric"].shape, (1, 8))
        self.assertNotIn("diagnostics", history)

    def test_invalid_state_precision_rejected(self):
        runner = make_evolution()
        with self.assertRaisesRegex(ValueError, "float64"):
            runner.observe({"f": np.asarray(runner.initial_state()["f"], dtype=np.float32)})

    def test_tangent_matches_centered_finite_difference(self):
        runner = make_evolution(strength=0.7, linearized=True)
        initial = runner.initial_state()
        direction = initial["h"]
        epsilon = 1e-4

        def full_step(f):
            return semilagrangian_vlasov_poisson_step(runner.cfg, f, ops=runner.ops)[0]

        difference = (full_step(runner.equilibrium + epsilon * direction)
                      - full_step(runner.equilibrium - epsilon * direction)) / (2 * epsilon)
        advanced, _ = runner.advance({"f": runner.equilibrium, "h": direction}, 1)
        relative = np.linalg.norm(advanced["h"] - difference) / np.linalg.norm(difference)
        self.assertLess(relative, 1e-6)


if __name__ == "__main__":
    unittest.main()
