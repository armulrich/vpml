"""Check import boundaries and precision of canonical numerical modules."""

import os
import subprocess
import sys
import unittest


class SolverModelModuleTests(unittest.TestCase):
    def run_in_fresh_process(self, source: str) -> None:
        environment = dict(os.environ)
        environment.update(
            JAX_PLATFORMS="cpu",
            VPML_JAX_BACKEND="cpu",
            JAX_ENABLE_X64="0",
        )
        result = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_hermite_solver_preserves_precision_without_loading_learned_models(self):
        self.run_in_fresh_process(
            """
import sys
from vpml.solvers.hermite import FourierHermiteIMEX
import jax
import jax.numpy as jnp

solver = FourierHermiteIMEX(Nx=8, Nv=4, Lx=6.0, dt=0.01)
assert jax.config.x64_enabled
assert solver.diag.dtype == jnp.complex128
assert solver.k_arr.dtype == jnp.float64
assert 'vpml.core' not in sys.modules
assert 'vpml.models.interface_closure' not in sys.modules
assert 'vpml.models.interface_checkpoint' not in sys.modules
"""
        )

    def test_moment_equations_do_not_load_closure_architectures(self):
        self.run_in_fresh_process(
            """
import sys
from vpml.solvers.moments import low_moment_rhs
import jax
import jax.numpy as jnp

state = jnp.zeros((1, 3, 8), dtype=jnp.float32)
k_arr = jnp.arange(5, dtype=state.dtype)
rhs = low_moment_rhs(state, jnp.zeros((1, 8), dtype=state.dtype), k_arr)
assert jax.config.x64_enabled
assert rhs.dtype == jnp.float32
assert bool(jnp.all(rhs == 0))
assert 'vpml.low_moment' not in sys.modules
assert 'vpml.models.moment_closures' not in sys.modules
assert 'vpml.models.moment_parameters' not in sys.modules
"""
        )


if __name__ == "__main__":
    unittest.main()
