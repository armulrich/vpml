"""NumPy/JAX restriction parity, including edge-frequency conventions."""
import unittest

import jax
import numpy as np

from vpml.fourier import restrict_rfft_jax, restrict_rfft_numpy

jax.config.update("jax_enable_x64", True)


class FourierRestrictionTests(unittest.TestCase):
    def test_numpy_and_jax_agree_for_batched_even_and_odd_grids(self):
        rng = np.random.default_rng(723)
        for source_nx, target_nx in ((16, 8), (16, 7), (15, 8), (15, 7), (16, 16)):
            with self.subTest(source=source_nx, target=target_nx):
                coefficients = np.fft.rfft(rng.normal(size=(2, 3, source_nx)), axis=-1)
                numpy_result = restrict_rfft_numpy(coefficients, source_nx=source_nx, target_nx=target_nx)
                compiled = jax.jit(lambda c: restrict_rfft_jax(c, source_nx=source_nx, target_nx=target_nx))
                np.testing.assert_allclose(compiled(coefficients), numpy_result, atol=1e-14, rtol=1e-14)

    def test_target_nyquist_has_full_cosine_amplitude(self):
        theta = 2 * np.pi * np.arange(32) / 32
        coefficients = np.fft.rfft(np.cos(4 * theta) + np.sin(4 * theta))
        restricted = restrict_rfft_numpy(coefficients, source_nx=32, target_nx=8)
        expected = np.cos(2 * np.pi * 4 * np.arange(8) / 8)
        np.testing.assert_allclose(np.fft.irfft(restricted, n=8), expected, atol=2e-14, rtol=0)

    def test_equal_grid_crops_without_changing_coefficients(self):
        values = np.arange(15, dtype=np.float64).astype(np.complex128)
        result = restrict_rfft_numpy(values, source_nx=16, target_nx=16)
        np.testing.assert_array_equal(result, values[:9])
        self.assertTrue(np.shares_memory(result, values))

    def test_single_precision_is_preserved(self):
        values = np.ones((2, 17), dtype=np.complex64)
        for function in (restrict_rfft_numpy, restrict_rfft_jax):
            self.assertEqual(function(values, source_nx=32, target_nx=8).dtype, np.dtype("complex64"))

    def test_unsupported_grid_and_insufficient_coefficients_raise(self):
        for function in (restrict_rfft_numpy, restrict_rfft_jax):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(np.ones(9), source_nx=16, target_nx=32)
                with self.assertRaises(ValueError):
                    function(np.ones(2), source_nx=16, target_nx=8)


if __name__ == "__main__":
    unittest.main()
