"""Small CPU checks of the portable worker against the unchanged solver."""

import unittest

import jax
import numpy as np

from vpml.datasets.diagnostics import DIAGNOSTIC_NAMES
from vpml.datasets.evolution import ReferenceEvolution
from vpml.datasets.initial_conditions import draw_coefficients
from vpml.datasets.manifest import Numerics
from vpml.physical_grid import semilagrangian_vlasov_poisson_step


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
        self.assertEqual(history["diagnostics"].shape, (2, len(DIAGNOSTIC_NAMES)))
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
        expected = jax.jvp(full_step, (runner._equilibrium,), (initial["h"],))[1]
        np.testing.assert_allclose(end["h"], expected, atol=1e-13, rtol=1e-13)
        self.assertEqual(history["diagnostics"].shape[-1], len(DIAGNOSTIC_NAMES) + 2)
        self.assertIn("delta_e2", runner.diagnostic_names)

    def test_invalid_state_precision_rejected(self):
        runner = make_evolution()
        with self.assertRaisesRegex(ValueError, "float64"):
            runner.observe({"f": np.asarray(runner.initial_state()["f"], dtype=np.float32)})


if __name__ == "__main__":
    unittest.main()
