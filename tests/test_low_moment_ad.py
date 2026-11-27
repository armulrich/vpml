"""Generic AD edge cases and forward-value regressions for low moments."""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from vpml.models import moment_features
from vpml.models.moment_features import (
    DYNAMIC_INPUT_SCALING,
    _normalized_closure_fields,
    _window_fno_features,
    encode_explicit_window_history,
)
from vpml.models.spectral_layers import _rms_with_floor
from vpml.models.moment_parameters import init_window_fno_params
from vpml.solvers.moments import limit_low_moment_state


def legacy_rms(values, floor):
    return jnp.maximum(
        jnp.sqrt(jnp.mean(jnp.square(values), axis=-1)),
        jnp.asarray(floor, dtype=values.dtype),
    )


class LowMomentADEdgeTests(unittest.TestCase):
    def setUp(self):
        original_x64 = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        self.addCleanup(jax.config.update, "jax_enable_x64", original_x64)

    def assertSameBits(self, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_rms_forward_bits_at_floor_and_neighbors(self):
        for dtype in (np.float32, np.float64):
            for evaluate in (lambda f: f, jax.jit):
                old, new = evaluate(legacy_rms), evaluate(_rms_with_floor)
                for floor in (0.0, 1e-6, 0.1, 0.125, 1e-30, 1e30):
                    floor = dtype(floor)
                    amplitudes = np.asarray([
                        0.0, np.nextafter(floor, dtype(0)), floor,
                        np.nextafter(floor, dtype(np.inf)), 2 * floor,
                    ], dtype=dtype)
                    values = jnp.asarray(amplitudes[:, None] * np.asarray(
                        [1, -1, 1, -1, 1, -1, 1, -1], dtype=dtype
                    ))
                    with self.subTest(dtype=dtype, floor=floor, evaluate=evaluate):
                        self.assertSameBits(new(values, floor), old(values, floor))
                values = jnp.asarray(np.random.default_rng(7).normal(size=(32, 8)), dtype=dtype)
                self.assertSameBits(new(values, dtype(0.1)), old(values, dtype(0.1)))

    def test_rms_gradients_below_at_above_and_zero_floor(self):
        for dtype in (jnp.float32, jnp.float64):
            for evaluate in (lambda f: f, jax.jit):
                gradient = evaluate(jax.grad(lambda x, f: jnp.sum(_rms_with_floor(x, f))))
                for value, floor, expected in (
                    (0.0, 0.125, 0.0), (0.0625, 0.125, 0.0),
                    (0.125, 0.125, 0.125), (0.25, 0.125, 0.25),
                    (0.0, 0.0, 0.0), (0.25, 0.0, 0.25),
                ):
                    x = jnp.full((1, 4), value, dtype=dtype)
                    with self.subTest(dtype=dtype, value=value, floor=floor):
                        np.testing.assert_array_equal(gradient(x, floor), np.full((1, 4), expected))

    def test_all_dynamic_rms_call_sites_have_finite_equilibrium_gradients(self):
        for dtype in (jnp.float32, jnp.float64):
            nx = 8
            state = jnp.zeros((1, 3, nx), dtype=dtype)
            history = jnp.zeros((1, 2, 3, nx), dtype=dtype)
            k = jnp.arange(nx // 2 + 1, dtype=dtype)
            params = init_window_fno_params(
                jax.random.PRNGKey(5), width=2, spectral_modes=2,
                memory_steps=2, depth=1, input_channels=4, dtype=dtype,
            )
            common = dict(input_scale=jnp.ones(4, dtype=dtype),
                          heat_flux_gradient_scale=1.0, input_scaling=DYNAMIC_INPUT_SCALING,
                          dynamic_amplitude_floor=1e-3)
            fields = lambda x: _normalized_closure_fields(
                {"amplitude_gain": jnp.zeros(4, dtype=dtype)}, x, jnp.ones(1, dtype=dtype), k,
                amplitude_center=0.0, amplitude_scale=1.0,
                previous_heat_flux_gradient=None, poisson_sign=1.0, **common,
            )
            features = lambda x: _window_fno_features(
                params, x, history, jnp.ones(1, dtype=dtype), k,
                amplitude_center=0.0, amplitude_scale=1.0, **common,
            )[0]
            encoded = lambda x: encode_explicit_window_history(params, x, k, **common)
            for function, value in ((fields, state), (features, state), (encoded, history)):
                gradient = jax.jit(jax.grad(lambda x: jnp.sum(function(x))))(value)
                self.assertTrue(np.isfinite(np.asarray(gradient)).all())
            np.testing.assert_allclose(
                jax.grad(lambda x: jnp.sum(fields(x)))(state),
                np.full(state.shape, 1.0 / np.asarray(1e-3, dtype=np.dtype(dtype))), rtol=2e-6,
            )
            # Compare each complete call path with the original RMS expression.
            for scale in (0.0, 1e-4, 1e-3, 0.05):
                probe = state.at[:, 0].set(scale * jnp.asarray([1, -1] * 4, dtype=dtype))
                probe_history = jnp.repeat(probe[:, None], 2, axis=1)
                for function, value in ((fields, probe), (features, probe), (encoded, probe_history)):
                    actual = jax.jit(function)(value)
                    with mock.patch.object(moment_features, "_rms_with_floor", legacy_rms):
                        expected = jax.jit(lambda x: function(x))(value)
                    self.assertSameBits(actual, expected)

    def test_density_rescaling_keeps_strict_epsilon_threshold(self):
        for dtype in (jnp.float32, jnp.float64):
            eps = np.finfo(np.dtype(dtype)).eps
            reference = jnp.zeros((1, 3, 4), dtype=dtype)
            for evaluate in (lambda f: f, jax.jit):
                limit = evaluate(lambda x: limit_low_moment_state(
                    x, reference_state=reference, density_floor=0.25,
                ))
                for ratio in (0.0, 0.5, 1.0, 2.0):
                    state = reference.at[:, 0].set(jnp.asarray(
                        [-1.0, -1.0, -1.0, -0.75 + 4 * ratio * eps], dtype=dtype
                    ))
                    expected = reference if ratio <= 1 else reference.at[:, 0].set(
                        jnp.asarray([-0.75, -0.75, -0.75, 2.25], dtype=dtype)
                    )
                    self.assertSameBits(limit(state), expected)
                    self.assertTrue(np.isfinite(np.asarray(
                        jax.grad(lambda x: jnp.sum(limit(x)))(state)
                    )).all())

    def test_density_fallback_gradients_and_zero_floor(self):
        for dtype in (jnp.float32, jnp.float64):
            reference = jnp.zeros((1, 3, 4), dtype=dtype)
            for floor in (0.0, 1e-4):
                state = reference.at[:, 0].set(-2.0)
                limit = lambda x: limit_low_moment_state(
                    x, reference_state=reference, density_floor=floor,
                )
                expected_gradient = jnp.ones_like(state).at[:, 0].set(0.0)
                self.assertSameBits(jax.jit(limit)(state), reference)
                np.testing.assert_array_equal(
                    jax.jit(jax.grad(lambda x: jnp.sum(limit(x))))(state), expected_gradient,
                )
                # Zero floor does not insert a positive density floor.
                valid = reference.at[:, 0].set(-0.5)
                self.assertSameBits(jax.jit(limit)(valid), valid)

    def test_active_limiter_directional_derivative_matches_finite_difference(self):
        for dtype, h, tolerance in ((jnp.float32, 1e-3, 2e-3), (jnp.float64, 1e-6, 1e-7)):
            reference = jnp.zeros((1, 3, 4), dtype=dtype)
            state = reference.at[:, 0].set(jnp.asarray([-1.2, -0.4, 0.2, 1.4], dtype=dtype))
            direction = jnp.asarray(np.random.default_rng(9).normal(size=state.shape), dtype=dtype)
            weights = jnp.arange(1, 13, dtype=dtype).reshape(state.shape)
            function = jax.jit(lambda x: jnp.sum(weights * limit_low_moment_state(
                x, reference_state=reference, density_floor=0.25,
            )))
            ad = jnp.vdot(jax.grad(function)(state), direction)
            fd = (function(state + h * direction) - function(state - h * direction)) / (2 * h)
            np.testing.assert_allclose(ad, fd, rtol=tolerance, atol=tolerance)


if __name__ == "__main__":
    unittest.main()
