import unittest

import jax
import jax.numpy as jnp
import numpy as np

from vpml.kinetic_latent import (
    _bounce_phase_features,
    _history_frequency_activation,
    advance_coupled_semilinear_strang,
    apply_coupled_linear_propagator,
    apply_linear_latent_propagator,
    bounded_normalized_closure_correction,
    bounded_normalized_latent_correction,
    closure_aligned_latent_correction,
    closure_orthogonal_latent_correction,
    coupled_low_moment_latent_step,
    equilibrium_preserving_latent_cnn_correction,
    gated_spectral_closure_correction,
    gated_spectral_latent_correction,
    init_kinetic_latent_dynamics,
    init_state_conditioned_latent_operator,
    kinetic_latent_dynamics_step,
    latent_heat_flux_gradient,
    low_moment_state_to_resolved_hermite,
    projected_hermite_latent_rhs,
    projected_hermite_electric_kick_rhs,
    reconstruct_first_unresolved_field,
    resolved_hermite_to_low_moment_state,
    rollout_kinetic_latent_dynamics,
    rollout_state_conditioned_kinetic_latent_dynamics,
    stabilize_coupled_linear_propagator,
    stabilize_linear_latent_propagator,
    state_conditioned_closure_correction,
    state_conditioned_latent_operator_correction,
    state_conditioned_latent_operator_features,
)
from vpml.low_moment import electric_field_from_density


class KineticLatentDynamicsTest(unittest.TestCase):
    def test_history_frequency_deadband_is_compact_and_causal(self):
        exposure = jnp.asarray(
            [
                [99.0, 98.0, 97.0, 96.0, 0.20, 0.25, 0.30, 0.35],
                [0.0, 0.0, 0.0, 0.0, 0.20, 0.25, 0.30, 0.36],
                [0.0, 0.0, 0.0, 0.0, 0.20, 0.25, 0.30, 0.37],
            ],
            dtype=jnp.float32,
        )
        activation = _history_frequency_activation(
            exposure, jnp.asarray([0.35, 0.37]), dtype=jnp.float32
        )
        np.testing.assert_allclose(activation, [0.0, 0.5, 1.0], atol=2e-6)

    def test_bounce_features_include_causal_cyclic_history_context(self):
        nx = 16
        x = 2.0 * jnp.pi * jnp.arange(nx) / float(nx)
        density = 0.2 * jnp.cos(x)
        momentum = 0.1 * jnp.sin(x)
        state = jnp.stack((density, momentum, density), axis=0)[None]
        k_arr = jnp.fft.rfftfreq(nx, d=1.0 / float(nx)).astype(jnp.float32)
        phase = 0.75 * jnp.pi
        low_history = _bounce_phase_features(
            state, jnp.asarray([[phase, 0.1]], dtype=jnp.float32), k_arr
        )
        high_history = _bounce_phase_features(
            state, jnp.asarray([[phase, 2.0]], dtype=jnp.float32), k_arr
        )
        self.assertEqual(low_history.shape, (1, 8))
        np.testing.assert_allclose(low_history[:, :4], high_history[:, :4])
        self.assertGreater(abs(float(high_history[0, 4])), abs(float(low_history[0, 4])))
        self.assertGreater(abs(float(high_history[0, 5])), abs(float(low_history[0, 5])))
        self.assertGreater(abs(float(high_history[0, 6])), abs(float(low_history[0, 6])))
        self.assertGreater(abs(float(high_history[0, 7])), abs(float(low_history[0, 7])))

    def test_bounce_features_include_periodic_pairwise_modal_phase(self):
        nx = 16
        x = 2.0 * jnp.pi * jnp.arange(nx) / float(nx)
        density = sum(0.05 * jnp.cos(mode * x) for mode in range(1, 5))
        momentum = sum(0.03 * jnp.sin(mode * x) for mode in range(1, 5))
        state = jnp.stack((density, momentum, density), axis=0)[None]
        k_arr = jnp.fft.rfftfreq(nx, d=1.0 / float(nx)).astype(jnp.float32)
        phase = jnp.asarray([[0.2, 0.7, 1.1, 1.9]], dtype=jnp.float32)
        maximum_frequency = jnp.ones_like(phase)
        exposure = jnp.concatenate((phase, maximum_frequency), axis=1)
        features = _bounce_phase_features(state, exposure, k_arr)
        shifted = _bounce_phase_features(
            state,
            jnp.concatenate((phase + 2.0 * jnp.pi, maximum_frequency), axis=1),
            k_arr,
        )
        self.assertEqual(features.shape, (1, 68))
        np.testing.assert_allclose(features, shifted, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(features[0, 32:34], [jnp.sin(-0.5), jnp.cos(-0.5) - 1.0])

    def test_electric_kick_matches_quadratic_moment_forces(self):
        nx = 24
        x = 2.0 * jnp.pi * jnp.arange(nx) / float(nx)
        density = 0.1 * jnp.cos(x) + 0.03 * jnp.cos(2.0 * x)
        momentum = 0.07 * jnp.sin(x)
        state = jnp.stack((density, momentum, 0.2 * jnp.cos(x)), axis=0)[None]
        latent = jnp.zeros((1, 3, nx), dtype=jnp.float32)
        basis = jnp.eye(3, dtype=jnp.float32)
        k_arr = jnp.fft.rfftfreq(nx, d=1.0 / float(nx)).astype(jnp.float32)
        fluid_rhs, _ = projected_hermite_electric_kick_rhs(
            state, latent, basis, k_arr
        )
        field = electric_field_from_density(density[None], k_arr)[0]

        def dealias(values):
            coefficients = jnp.fft.rfft(values)
            mask = jnp.arange(coefficients.shape[-1]) <= nx // 3
            return jnp.fft.irfft(coefficients * mask, n=nx)

        self.assertTrue(bool(jnp.allclose(fluid_rhs[:, 0], 0.0, atol=1e-7)))
        self.assertTrue(
            bool(jnp.allclose(fluid_rhs[0, 1], -dealias(density * field), atol=2e-6))
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    fluid_rhs[0, 2], -2.0 * dealias(momentum * field), atol=2e-6
                )
            )
        )

    def test_zero_kick_strang_step_composes_to_full_linear_map(self):
        nx = 16
        rank = 2
        channels = 3 + rank
        modes = nx // 2 + 1
        diagonal = jnp.asarray([0.99, 0.97, 0.95, 0.93, 0.91])
        half = jnp.broadcast_to(
            jnp.diag(diagonal).astype(jnp.complex64),
            (modes, channels, channels),
        )
        full = jnp.einsum("kac,kcb->kab", half, half)
        resolved = 0.05 * jax.random.normal(
            jax.random.PRNGKey(80), (1, 3, nx)
        )
        latent = 0.05 * jax.random.normal(
            jax.random.PRNGKey(81), (1, rank, nx)
        )
        state, updated_latent = advance_coupled_semilinear_strang(
            resolved_hermite_to_low_moment_state(resolved),
            latent,
            jnp.zeros_like(latent),
            half,
            jnp.eye(rank),
            jnp.fft.rfftfreq(nx, d=1.0 / float(nx)),
            resolved_scale=jnp.ones((3,)),
            latent_scale=jnp.ones((rank,)),
            fine_steps=1,
            fine_dt=0.01,
            kick_scale=0.0,
        )
        expected = apply_coupled_linear_propagator(
            full,
            resolved,
            latent,
            resolved_scale=jnp.ones((3,)),
            latent_scale=jnp.ones((rank,)),
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    low_moment_state_to_resolved_hermite(state),
                    expected[:, :3],
                    atol=2e-6,
                )
            )
        )
        self.assertTrue(bool(jnp.allclose(updated_latent, expected[:, 3:], atol=2e-6)))

    def test_endpoint_semilinear_correction_is_not_filtered_by_half_step(self):
        nx = 16
        rank = 2
        channels = 3 + rank
        modes = nx // 2 + 1
        half = jnp.broadcast_to(
            jnp.diag(jnp.asarray([0.9, 0.8, 0.7, 1e-3, 2e-3])).astype(
                jnp.complex64
            ),
            (modes, channels, channels),
        )
        resolved = jnp.zeros((1, 3, nx), dtype=jnp.float32)
        latent = jnp.zeros((1, rank, nx), dtype=jnp.float32)
        correction = jnp.stack(
            (jnp.cos(2.0 * jnp.pi * jnp.arange(nx) / nx), jnp.ones((nx,))),
            axis=0,
        )[None]
        state, updated_latent = advance_coupled_semilinear_strang(
            resolved_hermite_to_low_moment_state(resolved),
            latent,
            correction,
            half,
            jnp.eye(rank),
            jnp.fft.rfftfreq(nx, d=1.0 / float(nx)),
            resolved_scale=jnp.ones((3,)),
            latent_scale=jnp.ones((rank,)),
            fine_steps=1,
            fine_dt=0.01,
            kick_scale=0.0,
            correction_location="endpoint",
        )
        self.assertTrue(bool(jnp.allclose(updated_latent, correction, atol=2e-6)))
        self.assertTrue(
            bool(
                jnp.allclose(
                    low_moment_state_to_resolved_hermite(state),
                    resolved,
                    atol=2e-6,
                )
            )
        )

    def test_projected_hermite_tail_includes_nonlinear_acceleration(self):
        nx = 32
        x = 2.0 * jnp.pi * jnp.arange(nx) / float(nx)
        density = 0.2 * jnp.cos(x)
        c2 = jnp.full((nx,), 0.3, dtype=jnp.float32)
        state = jnp.stack(
            (
                density,
                jnp.zeros_like(density),
                density + jnp.sqrt(2.0) * c2,
            ),
            axis=0,
        )[None]
        latent = jnp.zeros((1, 4, nx), dtype=jnp.float32)
        basis = jnp.eye(4, dtype=jnp.float32)
        k_arr = jnp.fft.rfftfreq(nx, d=1.0 / float(nx)).astype(jnp.float32)
        positive = projected_hermite_latent_rhs(
            state,
            latent,
            basis,
            k_arr,
            tail_damping=0.0,
            tail_power=6.0,
            poisson_sign=1.0,
        )
        negative = projected_hermite_latent_rhs(
            state,
            latent,
            basis,
            k_arr,
            tail_damping=0.0,
            tail_power=6.0,
            poisson_sign=-1.0,
        )
        self.assertGreater(float(jnp.linalg.norm(positive[:, 0])), 1e-3)
        self.assertTrue(bool(jnp.allclose(positive, -negative, atol=2e-6)))

    def test_closure_aligned_decomposition_has_exact_c3_ownership(self):
        correction = jax.random.normal(jax.random.PRNGKey(50), (2, 5, 16))
        basis = jax.random.normal(jax.random.PRNGKey(51), (8, 5))
        closure = jax.random.normal(jax.random.PRNGKey(52), (2, 16))
        orthogonal = closure_orthogonal_latent_correction(correction, basis)
        aligned = closure_aligned_latent_correction(closure, basis)
        self.assertLess(
            float(jnp.max(jnp.abs(jnp.einsum("r,brx->bx", basis[0], orthogonal)))),
            2e-6,
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.einsum("r,brx->bx", basis[0], aligned),
                    closure,
                    atol=2e-6,
                )
            )
        )

    def test_closure_head_is_equilibrium_preserving(self):
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(53),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=2,
            spectral_modes=9,
            operator_rank=3,
            output_projection_init_scale=1e-3,
            closure_aligned_output=True,
        )
        resolved = jnp.zeros((1, 3, 16), dtype=jnp.float32)
        latent = jnp.zeros((1, 4, 16), dtype=jnp.float32)
        correction = state_conditioned_closure_correction(
            params,
            resolved,
            latent,
            depth=2,
            equilibrium_preserving=True,
        )
        self.assertTrue(bool(jnp.allclose(correction, 0.0, atol=1e-7)))

    def test_gated_closure_readout_vanishes_at_linear_order(self):
        nx = 16
        channels = 7
        params = {
            "operator_u_gated_closure_real": jnp.ones(
                (nx // 2 + 1, channels), dtype=jnp.float32
            ),
            "operator_u_gated_closure_imag": jnp.zeros(
                (nx // 2 + 1, channels), dtype=jnp.float32
            ),
        }
        resolved = jax.random.normal(jax.random.PRNGKey(58), (1, 3, nx))
        latent = jax.random.normal(jax.random.PRNGKey(59), (1, 4, nx))
        full = gated_spectral_closure_correction(
            params, resolved, latent, gate_scale=0.1
        )
        small = gated_spectral_closure_correction(
            params, 1e-3 * resolved, 1e-3 * latent, gate_scale=0.1
        )
        smaller = gated_spectral_closure_correction(
            params, 5e-4 * resolved, 5e-4 * latent, gate_scale=0.1
        )
        self.assertGreater(float(jnp.linalg.norm(full)), 0.0)
        self.assertLess(
            float(jnp.linalg.norm(smaller) / jnp.linalg.norm(small)),
            0.14,
        )
        zero = gated_spectral_closure_correction(
            params,
            jnp.zeros_like(resolved),
            jnp.zeros_like(latent),
            gate_scale=0.1,
        )
        self.assertTrue(bool(jnp.allclose(zero, 0.0, atol=1e-8)))

    def test_gated_latent_readout_is_equilibrium_preserving(self):
        nx = 16
        channels = 7
        rank = 4
        params = {
            "operator_u_gated_latent_real": jnp.ones(
                (nx // 2 + 1, channels, rank), dtype=jnp.float32
            ),
            "operator_u_gated_latent_imag": jnp.zeros(
                (nx // 2 + 1, channels, rank), dtype=jnp.float32
            ),
        }
        resolved = jnp.zeros((2, 3, nx), dtype=jnp.float32)
        latent = jnp.zeros((2, rank, nx), dtype=jnp.float32)
        correction = gated_spectral_latent_correction(
            params, resolved, latent, gate_scale=0.1
        )
        self.assertEqual(correction.shape, latent.shape)
        self.assertTrue(bool(jnp.allclose(correction, 0.0, atol=1e-8)))

    def test_channelwise_correction_bounds_match_smooth_formula(self):
        correction = jnp.asarray(
            [[[0.5, 2.0], [-3.0, 1.0]]], dtype=jnp.float32
        )
        bounds = jnp.asarray([1.0, 4.0], dtype=jnp.float32)
        actual = bounded_normalized_latent_correction(correction, bounds)
        expected = bounds[None, :, None] * jnp.tanh(
            correction / bounds[None, :, None]
        )
        self.assertTrue(bool(jnp.allclose(actual, expected)))
        self.assertTrue(
            bool(jnp.all(jnp.abs(actual) <= bounds[None, :, None]))
        )

    def test_closure_correction_bound_matches_smooth_formula(self):
        correction = jnp.asarray([[-30.0, -2.0, 0.0, 4.0, 40.0]])
        actual = bounded_normalized_closure_correction(correction, 40.0)
        expected = 40.0 * jnp.tanh(correction / 40.0)
        self.assertTrue(bool(jnp.allclose(actual, expected)))
        self.assertLessEqual(float(jnp.max(jnp.abs(actual))), 40.0)

    def test_resolved_and_conservative_states_round_trip(self):
        resolved = jax.random.normal(jax.random.PRNGKey(20), (2, 3, 16))
        state = resolved_hermite_to_low_moment_state(resolved)
        reconstructed = low_moment_state_to_resolved_hermite(state)
        self.assertTrue(bool(jnp.allclose(reconstructed, resolved, atol=1e-6)))

    def test_latent_heat_flux_gradient_has_zero_spatial_mean(self):
        state = jnp.zeros((2, 3, 16), dtype=jnp.float32)
        latent = jax.random.normal(jax.random.PRNGKey(21), (2, 4, 16))
        basis = jax.random.normal(jax.random.PRNGKey(22), (8, 4))
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(16, d=0.25)
        gradient = latent_heat_flux_gradient(state, latent, basis, k_arr)
        self.assertLess(float(jnp.max(jnp.abs(jnp.mean(gradient, axis=-1)))), 1e-6)

    def test_coupled_equilibrium_remains_equilibrium(self):
        rank = 4
        nx = 16
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(23),
            resolved_channels=3,
            latent_rank=rank,
            width=8,
            depth=2,
            spectral_modes=nx // 2 + 1,
            operator_rank=3,
        )
        propagator = jnp.broadcast_to(
            jnp.eye(3 + rank, dtype=jnp.complex64),
            (nx // 2 + 1, 3 + rank, 3 + rank),
        )
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32)
        latent = jnp.zeros((1, rank, nx), dtype=jnp.float32)
        basis = jnp.zeros((8, rank), dtype=jnp.float32)
        k_arr = (
            2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=0.25)
        ).astype(jnp.float32)
        updated_state, updated_latent = coupled_low_moment_latent_step(
            params,
            state,
            latent,
            propagator,
            basis,
            k_arr,
            resolved_center=jnp.zeros((3,)),
            resolved_scale=jnp.ones((3,)),
            latent_center=jnp.zeros((rank,)),
            latent_scale=jnp.ones((rank,)),
            depth=2,
            fine_steps=2,
            fine_dt=0.01,
        )
        self.assertTrue(bool(jnp.allclose(updated_state, 0.0, atol=1e-7)))
        self.assertTrue(bool(jnp.allclose(updated_latent, 0.0, atol=1e-7)))

    def test_closure_only_step_ignores_generic_latent_readout(self):
        rank = 4
        nx = 16
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(54),
            resolved_channels=3,
            latent_rank=rank,
            width=8,
            depth=2,
            spectral_modes=nx // 2 + 1,
            operator_rank=3,
            output_projection_init_scale=1e-3,
            closure_aligned_output=True,
        )
        changed = dict(params)
        changed["operator_u_real"] = params["operator_u_real"] + 100.0
        changed["operator_u_imag"] = params["operator_u_imag"] - 100.0
        propagator = jnp.broadcast_to(
            jnp.eye(3 + rank, dtype=jnp.complex64),
            (nx // 2 + 1, 3 + rank, 3 + rank),
        )
        state = 0.01 * jax.random.normal(jax.random.PRNGKey(55), (1, 3, nx))
        latent = 0.01 * jax.random.normal(jax.random.PRNGKey(56), (1, rank, nx))
        basis = jax.random.normal(jax.random.PRNGKey(57), (8, rank))
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=0.25)
        common = dict(
            propagator=propagator,
            basis=basis,
            k_arr=k_arr,
            resolved_center=jnp.zeros((3,)),
            resolved_scale=jnp.ones((3,)),
            latent_center=jnp.zeros((rank,)),
            latent_scale=jnp.ones((rank,)),
            depth=2,
            closure_aligned_output=True,
            closure_only_correction=True,
            fine_steps=1,
            fine_dt=0.001,
        )
        expected = coupled_low_moment_latent_step(
            params, state, latent, **common
        )
        actual = coupled_low_moment_latent_step(
            changed, state, latent, **common
        )
        self.assertTrue(bool(jnp.allclose(actual[0], expected[0], atol=1e-6)))
        self.assertTrue(bool(jnp.allclose(actual[1], expected[1], atol=1e-6)))

    def test_complete_modal_stabilization_clips_only_unstable_eigenvalues(self):
        propagator = np.zeros((2, 3, 3), dtype=np.complex64)
        propagator[0] = np.diag([1.0, 0.9, 1.2])
        propagator[1] = np.diag([0.8, 0.7, 0.6])
        stabilized, diagnostics = stabilize_coupled_linear_propagator(
            propagator,
            maximum_spectral_radius=1.0,
        )
        self.assertLessEqual(
            float(np.max(np.abs(np.linalg.eigvals(stabilized[0])))),
            1.0 + 1e-6,
        )
        self.assertEqual(diagnostics[0]["adjusted_eigenvalues"], 1)
        self.assertEqual(diagnostics[1]["adjusted_eigenvalues"], 0)

    def test_step_and_rollout_shapes(self):
        params = init_kinetic_latent_dynamics(
            jax.random.PRNGKey(0),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=2,
        )
        resolved = jnp.zeros((2, 3, 16), dtype=jnp.float32)
        latent = jnp.zeros((2, 4, 16), dtype=jnp.float32)
        updated = kinetic_latent_dynamics_step(params, resolved, latent, depth=2)
        self.assertEqual(updated.shape, latent.shape)

        history = jnp.zeros((2, 5, 3, 16), dtype=jnp.float32)
        trajectory = rollout_kinetic_latent_dynamics(
            params,
            history,
            latent,
            depth=2,
        )
        self.assertEqual(trajectory.shape, (2, 5, 4, 16))
        self.assertTrue(bool(jnp.all(jnp.isfinite(trajectory))))

    def test_periodic_translation_covariance(self):
        params = init_kinetic_latent_dynamics(
            jax.random.PRNGKey(1),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=2,
        )
        resolved = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 16))
        latent = jax.random.normal(jax.random.PRNGKey(3), (2, 4, 16))
        reference = kinetic_latent_dynamics_step(
            params,
            resolved,
            latent,
            depth=2,
        )
        shifted = kinetic_latent_dynamics_step(
            params,
            jnp.roll(resolved, 3, axis=-1),
            jnp.roll(latent, 3, axis=-1),
            depth=2,
        )
        self.assertLess(
            float(jnp.max(jnp.abs(shifted - jnp.roll(reference, 3, axis=-1)))),
            2e-6,
        )

    def test_first_unresolved_readout(self):
        latent = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
        basis = jnp.asarray([[1.0, -0.5, 0.25], [0.0, 1.0, 0.0]])
        expected = latent[:, 0] - 0.5 * latent[:, 1] + 0.25 * latent[:, 2]
        actual = reconstruct_first_unresolved_field(latent, basis)
        self.assertTrue(bool(jnp.allclose(actual, expected)))

    def test_structured_rollout_reduces_to_linear_propagator(self):
        params = init_kinetic_latent_dynamics(
            jax.random.PRNGKey(4),
            resolved_channels=3,
            latent_rank=2,
            width=6,
            depth=1,
        )
        params["output_kernel"] = jnp.zeros_like(params["output_kernel"])
        resolved = jax.random.normal(jax.random.PRNGKey(5), (1, 3, 8))
        latent = jax.random.normal(jax.random.PRNGKey(6), (1, 2, 8))
        propagator = jnp.zeros((5, 5, 2), dtype=jnp.complex64)
        propagator = propagator.at[:, 3, 0].set(1.0)
        propagator = propagator.at[:, 4, 1].set(1.0)
        expected = apply_linear_latent_propagator(
            propagator,
            resolved,
            latent,
        )
        actual = rollout_state_conditioned_kinetic_latent_dynamics(
            params,
            resolved[:, None],
            latent,
            propagator,
            resolved_center=jnp.zeros((3,)),
            resolved_scale=jnp.ones((3,)),
            latent_center=jnp.zeros((2,)),
            latent_scale=jnp.ones((2,)),
            depth=1,
        )[:, 0]
        self.assertTrue(bool(jnp.allclose(actual, expected, atol=1e-6)))

    def test_multiplicative_operator_is_translation_covariant(self):
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(11),
            resolved_channels=3,
            latent_rank=6,
            width=8,
            depth=2,
            spectral_modes=9,
            operator_rank=4,
            kernel_size=3,
        )
        resolved = jax.random.normal(jax.random.PRNGKey(12), (2, 3, 16))
        latent = jax.random.normal(jax.random.PRNGKey(13), (2, 6, 16))
        reference = state_conditioned_latent_operator_correction(
            params,
            resolved,
            latent,
            depth=2,
        )
        shifted = state_conditioned_latent_operator_correction(
            params,
            jnp.roll(resolved, 5, axis=-1),
            jnp.roll(latent, 5, axis=-1),
            depth=2,
        )
        self.assertLess(
            float(jnp.max(jnp.abs(shifted - jnp.roll(reference, 5, axis=-1)))),
            2e-5,
        )

    def test_latent_delay_is_causal_and_preserves_equilibrium(self):
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(90),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=2,
            spectral_modes=9,
            operator_rank=3,
            kernel_size=3,
            output_projection_init_scale=1e-2,
            latent_delay_input=True,
        )
        resolved = 0.1 * jax.random.normal(
            jax.random.PRNGKey(91), (1, 3, 16)
        )
        latent = 0.1 * jax.random.normal(
            jax.random.PRNGKey(92), (1, 4, 16)
        )
        zero_delta = jnp.zeros_like(latent)
        nonzero_delta = 0.1 * jax.random.normal(
            jax.random.PRNGKey(93), latent.shape
        )
        without_motion = state_conditioned_latent_operator_correction(
            params,
            resolved,
            latent,
            depth=2,
            equilibrium_preserving=True,
            normalized_latent_delta=zero_delta,
        )
        with_motion = state_conditioned_latent_operator_correction(
            params,
            resolved,
            latent,
            depth=2,
            equilibrium_preserving=True,
            normalized_latent_delta=nonzero_delta,
        )
        equilibrium = state_conditioned_latent_operator_correction(
            params,
            jnp.zeros_like(resolved),
            jnp.zeros_like(latent),
            depth=2,
            equilibrium_preserving=True,
            normalized_latent_delta=jnp.zeros_like(latent),
        )
        self.assertGreater(
            float(jnp.max(jnp.abs(with_motion - without_motion))), 1e-9
        )
        self.assertTrue(bool(jnp.allclose(equilibrium, 0.0, atol=1e-8)))

    def test_direct_latent_cnn_removes_constant_and_linear_response(self):
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(94),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=2,
            spectral_modes=9,
            operator_rank=3,
            kernel_size=3,
            conditioner_output_channels=4,
            latent_delay_input=True,
        )
        resolved = jax.random.normal(jax.random.PRNGKey(95), (1, 3, 16))
        latent = jax.random.normal(jax.random.PRNGKey(96), (1, 4, 16))
        delta = jax.random.normal(jax.random.PRNGKey(97), (1, 4, 16))

        def correction(scale):
            return equilibrium_preserving_latent_cnn_correction(
                params,
                scale * resolved,
                scale * latent,
                depth=2,
                normalized_latent_delta=scale * delta,
            )

        equilibrium = correction(0.0)
        small = float(jnp.linalg.norm(correction(1e-2)))
        doubled = float(jnp.linalg.norm(correction(2e-2)))
        self.assertTrue(bool(jnp.allclose(equilibrium, 0.0, atol=1e-8)))
        self.assertGreater(small, 0.0)
        self.assertGreater(doubled / small, 2.5)
        self.assertLess(doubled / small, 5.5)

        def compressed_correction(scale):
            return equilibrium_preserving_latent_cnn_correction(
                params,
                scale * resolved,
                scale * latent,
                depth=2,
                normalized_latent_delta=scale * delta,
                input_compression_scale=4.0,
            )

        compressed_small = float(jnp.linalg.norm(compressed_correction(1e-2)))
        compressed_doubled = float(jnp.linalg.norm(compressed_correction(2e-2)))
        self.assertTrue(
            bool(jnp.allclose(compressed_correction(0.0), 0.0, atol=1e-8))
        )
        self.assertGreater(compressed_small, 0.0)
        self.assertGreater(compressed_doubled / compressed_small, 2.5)
        self.assertLess(compressed_doubled / compressed_small, 5.5)

    def test_zero_initialized_experts_preserve_direct_cnn_exactly(self):
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(194),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=2,
            spectral_modes=9,
            operator_rank=3,
            kernel_size=3,
            conditioner_output_channels=4,
            conditioner_experts=4,
            latent_delay_input=True,
        )
        base_params = {
            key: value for key, value in params.items() if not key.startswith("expert_")
        }
        resolved = jax.random.normal(jax.random.PRNGKey(195), (2, 3, 16))
        latent = jax.random.normal(jax.random.PRNGKey(196), (2, 4, 16))
        delta = jax.random.normal(jax.random.PRNGKey(197), (2, 4, 16))
        expected = equilibrium_preserving_latent_cnn_correction(
            base_params,
            resolved,
            latent,
            depth=2,
            normalized_latent_delta=delta,
            input_compression_scale=4.0,
        )
        actual = equilibrium_preserving_latent_cnn_correction(
            params,
            resolved,
            latent,
            depth=2,
            normalized_latent_delta=delta,
            input_compression_scale=4.0,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_expert_amplitude_gate_selects_distinct_scales(self):
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(198),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=1,
            spectral_modes=9,
            operator_rank=3,
            kernel_size=3,
            conditioner_output_channels=4,
            conditioner_experts=4,
        )
        centers = np.asarray(params["expert_gate_log_amplitude_centers"])
        np.testing.assert_allclose(centers, (-3.0, -1.3, -0.7, 0.25))
        self.assertAlmostEqual(
            float(params["expert_gate_log_amplitude_width"]),
            0.30,
            places=6,
        )
        np.testing.assert_array_equal(params["expert_phase_output_kernel"], 0.0)
        np.testing.assert_array_equal(params["expert_phase_output_bias"], 0.0)
        np.testing.assert_array_equal(params["expert_phase_gain"], 0.0)

    def test_operator_features_decode_to_the_reported_correction(self):
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(14),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=2,
            spectral_modes=9,
            operator_rank=3,
            kernel_size=3,
            output_projection_init_scale=1e-2,
        )
        resolved = jax.random.normal(jax.random.PRNGKey(15), (3, 3, 16))
        latent = jax.random.normal(jax.random.PRNGKey(16), (3, 4, 16))
        features = state_conditioned_latent_operator_features(
            params,
            resolved,
            latent,
            depth=2,
            equilibrium_preserving=True,
        )
        operator_u = jax.lax.complex(
            params["operator_u_real"], params["operator_u_imag"]
        )
        decoded_hat = jnp.einsum("bjk,kjr->brk", features, operator_u)
        decoded = jnp.fft.irfft(decoded_hat, n=16, axis=-1, norm="forward")
        correction = state_conditioned_latent_operator_correction(
            params,
            resolved,
            latent,
            depth=2,
            equilibrium_preserving=True,
        )
        self.assertEqual(features.shape, (3, 3, 9))
        self.assertTrue(bool(jnp.allclose(decoded, correction, atol=1e-6)))

    def test_equilibrium_preserving_operator_has_zero_value_and_jacobian(self):
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(30),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=2,
            spectral_modes=9,
            operator_rank=3,
            kernel_size=3,
        )
        params["operator_u_real"] = jax.random.normal(
            jax.random.PRNGKey(31),
            params["operator_u_real"].shape,
            dtype=params["operator_u_real"].dtype,
        )
        resolved = jnp.zeros((1, 3, 16), dtype=jnp.float32)
        latent = jnp.zeros((1, 4, 16), dtype=jnp.float32)

        def correction(resolved_value, latent_value):
            return state_conditioned_latent_operator_correction(
                params,
                resolved_value,
                latent_value,
                depth=2,
                equilibrium_preserving=True,
            )

        value, tangent = jax.jvp(
            correction,
            (resolved, latent),
            (
                jnp.ones_like(resolved),
                jnp.ones_like(latent),
            ),
        )
        self.assertTrue(bool(jnp.allclose(value, 0.0, atol=1e-7)))
        self.assertTrue(bool(jnp.allclose(tangent, 0.0, atol=1e-6)))

    def test_zero_multiplicative_operator_reduces_to_linear_propagator(self):
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(14),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=2,
            spectral_modes=9,
            operator_rank=3,
            kernel_size=3,
        )
        params = {
            key: jnp.zeros_like(value) if key.startswith("operator_u_") else value
            for key, value in params.items()
        }
        resolved = jax.random.normal(jax.random.PRNGKey(15), (1, 3, 16))
        latent = jax.random.normal(jax.random.PRNGKey(16), (1, 4, 16))
        correction = state_conditioned_latent_operator_correction(
            params,
            resolved,
            latent,
            depth=2,
        )
        self.assertTrue(bool(jnp.allclose(correction, 0.0, atol=1e-7)))

    def test_operator_initialization_has_zero_nonlinear_readout(self):
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(17),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=2,
            spectral_modes=9,
            operator_rank=3,
        )
        self.assertTrue(bool(jnp.all(params["operator_u_real"] == 0.0)))
        self.assertTrue(bool(jnp.all(params["operator_u_imag"] == 0.0)))

    def test_nonzero_output_initialization_preserves_equilibrium_structure(self):
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(42),
            resolved_channels=3,
            latent_rank=4,
            width=8,
            depth=2,
            spectral_modes=9,
            operator_rank=3,
            output_projection_init_scale=1e-4,
        )
        self.assertGreater(float(jnp.linalg.norm(params["operator_u_real"])), 0.0)
        resolved = jnp.zeros((1, 3, 16), dtype=jnp.float32)
        latent = jnp.zeros((1, 4, 16), dtype=jnp.float32)

        def correction(resolved_value, latent_value):
            return state_conditioned_latent_operator_correction(
                params,
                resolved_value,
                latent_value,
                depth=2,
                equilibrium_preserving=True,
            )

        value, tangent = jax.jvp(
            correction,
            (resolved, latent),
            (jnp.ones_like(resolved), jnp.ones_like(latent)),
        )
        self.assertTrue(bool(jnp.allclose(value, 0.0, atol=1e-7)))
        self.assertTrue(bool(jnp.allclose(tangent, 0.0, atol=1e-6)))

    def test_linear_propagator_stabilization_preserves_resolved_forcing(self):
        propagator = np.zeros((3, 5, 2), dtype=np.complex64)
        propagator[:, :3] = 0.25
        propagator[:, 3, 0] = 1.2
        propagator[:, 4, 1] = 0.8
        stabilized, diagnostics = stabilize_linear_latent_propagator(
            propagator,
            resolved_channels=3,
            maximum_spectral_radius=1.0,
        )
        np.testing.assert_array_equal(stabilized[:, :3], propagator[:, :3])
        self.assertEqual(len(diagnostics), 3)
        for mode in range(3):
            radius = np.max(np.abs(np.linalg.eigvals(stabilized[mode, 3:, :])))
            self.assertLessEqual(float(radius), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
