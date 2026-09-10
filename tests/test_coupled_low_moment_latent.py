import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np

from model.train.coupled_low_moment_latent import (
    _aggregate_gradients,
    _atomic_savez,
    _calibrate_smooth_correction_bounds,
    _clip_parameter_update,
    _combine_gradients,
    _combine_conflict_safe_gradients,
    _combine_conflict_safe_updates,
    _combine_groupwise_conflict_safe_updates,
    _combine_optimizer_updates,
    _electric_chunk_log_growth_error,
    _electric_spectral_amplitude_terms,
    _first_nonfinite_step,
    _gradient_group_norms,
    _load_training_state,
    _make_functions,
    _matrix_square_root_propagator,
    _mask_tail_output_gradients,
    _mean_gradients,
    _scale_auxiliary_update,
    _sgd_step,
    _physical_sample_loss,
    _physical_trajectory_terms,
    _training_state_payload,
    _turnaround_indices,
)
from model.train.kinetic_latent_dynamics_probe import _adam_init
from vpml.kinetic_latent import (
    apply_coupled_linear_propagator,
    bounded_normalized_latent_correction,
    init_state_conditioned_latent_operator,
    resolved_hermite_to_low_moment_state,
    state_conditioned_latent_operator_correction,
)
from vpml.low_moment import primitive_fields


class CoupledLowMomentLatentLossTest(unittest.TestCase):
    def test_coordinate_median_gradient_rejects_single_outlier(self):
        gradients = [
            {"weight": jnp.asarray([1.0, 2.0])},
            {"weight": jnp.asarray([1.2, 1.8])},
            {"weight": jnp.asarray([-40.0, 50.0])},
        ]
        aggregated = _aggregate_gradients(
            gradients, normalize=False, method="coordinate_median"
        )
        np.testing.assert_allclose(aggregated["weight"], (1.0, 2.0))

    def test_turnaround_indices_select_strongest_local_regrowth(self):
        energy = jnp.concatenate(
            (
                jnp.linspace(5.0, 2.0, 51),
                jnp.asarray([1.0, 3.0, 2.0, 0.5, 4.0, 3.0]),
            )
        )[None, :]
        minimum, peak = _turnaround_indices(energy, minimum_start_index=50)
        self.assertEqual(int(minimum[0]), 54)
        self.assertEqual(int(peak[0]), 55)

    def test_tail_output_gradient_mask_freezes_core(self):
        gradients = {
            "output_kernel": jnp.ones((5, 2, 3)),
            "output_bias": jnp.ones((5,)),
            "block_0_kernel": jnp.ones((2, 2, 3)),
        }
        masked = _mask_tail_output_gradients(gradients, 3)
        np.testing.assert_array_equal(masked["output_kernel"][:3], 0.0)
        np.testing.assert_array_equal(masked["output_kernel"][3:], 1.0)
        np.testing.assert_array_equal(masked["output_bias"][:3], 0.0)
        np.testing.assert_array_equal(masked["output_bias"][3:], 1.0)
        np.testing.assert_array_equal(masked["block_0_kernel"], 0.0)

    def test_expert_gradient_mask_freezes_base_model(self):
        gradients = {
            "output_kernel": jnp.ones((5, 2, 3)),
            "expert_output_kernel": jnp.ones((4, 5, 2, 3)),
            "expert_gate_kernel": jnp.ones((4, 2)),
            "expert_gate_log_amplitude_centers": jnp.ones((4,)),
        }
        masked = _mask_tail_output_gradients(
            gradients, 0, expert_output_only=True
        )
        np.testing.assert_array_equal(masked["output_kernel"], 0.0)
        np.testing.assert_array_equal(masked["expert_output_kernel"], 1.0)
        np.testing.assert_array_equal(masked["expert_gate_kernel"], 1.0)
        np.testing.assert_array_equal(
            masked["expert_gate_log_amplitude_centers"], 0.0
        )

    def test_expert_gradient_mask_can_select_one_expert(self):
        gradients = {
            "expert_output_kernel": jnp.ones((4, 5, 2, 3)),
            "expert_output_bias": jnp.ones((4, 5)),
            "expert_gate_kernel": jnp.ones((4, 2)),
            "expert_gate_bias": jnp.ones((4,)),
        }
        masked = _mask_tail_output_gradients(
            gradients,
            0,
            expert_output_only=True,
            expert_indices=(1,),
        )
        np.testing.assert_array_equal(masked["expert_output_kernel"][1], 1.0)
        np.testing.assert_array_equal(
            np.asarray(masked["expert_output_kernel"])[[0, 2, 3]], 0.0
        )
        np.testing.assert_array_equal(masked["expert_output_bias"][1], 1.0)
        np.testing.assert_array_equal(
            np.asarray(masked["expert_output_bias"])[[0, 2, 3]], 0.0
        )
        np.testing.assert_array_equal(masked["expert_gate_kernel"], 0.0)
        np.testing.assert_array_equal(masked["expert_gate_bias"], 0.0)

    def test_modewise_matrix_square_root_recovers_propagator(self):
        propagator = np.stack(
            (
                np.diag([0.81, 0.64, 0.49]),
                np.asarray(
                    [[0.7, 0.1, 0.0], [0.0, 0.6, 0.2], [0.0, 0.0, 0.5]]
                ),
            )
        ).astype(np.complex64)
        half = _matrix_square_root_propagator(propagator)
        np.testing.assert_allclose(
            np.einsum("kac,kcb->kab", half, half),
            propagator,
            rtol=5e-5,
            atol=5e-6,
        )

    def test_sgd_preserves_global_gradient_direction(self):
        params = {"value": jnp.asarray([1.0, -2.0])}
        grads = {"value": jnp.asarray([3.0, 4.0])}
        state = _adam_init(params)
        updated, returned_state, norm = _sgd_step(
            params, grads, state, learning_rate=0.5, grad_clip=1.0
        )
        self.assertAlmostEqual(float(norm), 5.0, places=6)
        self.assertTrue(
            bool(
                jnp.allclose(
                    updated["value"],
                    params["value"] - 0.1 * grads["value"],
                )
            )
        )
        self.assertIs(returned_state, state)

    def test_normalized_gradient_mean_limits_outlier_magnitude(self):
        gradients = [
            {"value": jnp.asarray([1.0, 0.0])},
            {"value": jnp.asarray([0.0, 1000.0])},
        ]
        raw = _mean_gradients(gradients, normalize=False)
        normalized = _mean_gradients(gradients, normalize=True)
        self.assertTrue(
            bool(jnp.allclose(raw["value"], jnp.asarray([0.5, 500.0])))
        )
        self.assertTrue(
            bool(jnp.allclose(normalized["value"], jnp.asarray([0.5, 0.5])))
        )

    def test_direct_optimizer_update_keeps_conflicting_teacher_direction(self):
        physical = {"value": jnp.asarray([1.0, 0.0])}
        teacher = {"value": jnp.asarray([-0.5, 2.0])}
        combined, _, _, cosine, contribution, conflicting, alignment = (
            _combine_optimizer_updates(
                physical,
                teacher,
                mode="direct_sum",
                output_ratio=0.0,
                internal_ratio=0.0,
            )
        )
        np.testing.assert_allclose(combined["value"], np.asarray([0.5, 2.0]))
        self.assertLess(float(cosine), 0.0)
        self.assertAlmostEqual(float(contribution), np.sqrt(4.25), places=6)
        self.assertTrue(bool(conflicting))
        self.assertAlmostEqual(float(alignment), 0.5, places=6)

    def test_direct_joint_gradient_keeps_conflicting_teacher_direction(self):
        physical = {"value": jnp.asarray([1.0, 0.0])}
        teacher = {"value": jnp.asarray([-0.5, 2.0])}
        combined, _, _, cosine, contribution, conflicting, alignment = (
            _combine_gradients(
                physical,
                teacher,
                auxiliary_weight=1.0,
                maximum_auxiliary_ratio=0.0,
                mode="direct_sum",
            )
        )
        np.testing.assert_allclose(combined["value"], np.asarray([0.5, 2.0]))
        self.assertLess(float(cosine), 0.0)
        self.assertAlmostEqual(float(contribution), np.sqrt(4.25), places=6)
        self.assertTrue(bool(conflicting))
        self.assertAlmostEqual(float(alignment), 0.5, places=6)

    def test_zero_auxiliary_weight_disables_teacher_update(self):
        params = {"value": jnp.asarray([1.0, -2.0])}
        candidate = {"value": jnp.asarray([4.0, 3.0])}
        disabled = _scale_auxiliary_update(candidate, params, 0.0)
        half = _scale_auxiliary_update(candidate, params, 0.5)
        self.assertTrue(bool(jnp.allclose(disabled["value"], 0.0)))
        self.assertTrue(
            bool(jnp.allclose(half["value"], jnp.asarray([1.5, 2.5])))
        )

    def test_training_state_round_trip_restores_optimizer_and_epoch(self):
        params = {
            "weight": jnp.asarray([[1.0, -2.0]], dtype=jnp.float32),
            "bias": jnp.asarray([0.5], dtype=jnp.float32),
        }
        optimizers = {
            "physical": _adam_init(params),
            "teacher": _adam_init(params),
        }
        payload = _training_state_payload(params, optimizers)
        payload["completed_epoch"] = np.asarray(7)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.npz"
            _atomic_savez(path, payload)
            restored_params, restored_optimizers, epoch = _load_training_state(
                path,
                {key: jnp.zeros_like(value) for key, value in params.items()},
                {
                    name: _adam_init(params)
                    for name in ("physical", "teacher")
                },
            )
        self.assertEqual(epoch, 7)
        for key in params:
            self.assertTrue(np.array_equal(restored_params[key], params[key]))
            for objective in optimizers:
                self.assertTrue(
                    np.array_equal(
                        restored_optimizers[objective]["m"][key],
                        optimizers[objective]["m"][key],
                    )
                )

    def test_groupwise_update_allows_larger_internal_teacher_step(self):
        physical = {
            "operator_u_real": jnp.asarray([1.0]),
            "operator_v_real": jnp.asarray([1.0]),
            "conditioner": jnp.asarray([1.0]),
        }
        teacher = {
            "operator_u_real": jnp.asarray([4.0]),
            "operator_v_real": jnp.asarray([4.0]),
            "conditioner": jnp.asarray([4.0]),
        }
        combined, *_ = _combine_groupwise_conflict_safe_updates(
            physical,
            teacher,
            output_ratio=1.0,
            internal_ratio=4.0,
        )
        self.assertAlmostEqual(float(combined["operator_u_real"][0]), 2.0)
        self.assertAlmostEqual(float(combined["operator_v_real"][0]), 5.0)
        self.assertAlmostEqual(float(combined["conditioner"][0]), 5.0)

    def test_conflicting_teacher_update_is_projected_without_opposing_physics(self):
        physical = {"x": jnp.asarray([-1.0, 0.0])}
        teacher = {"x": jnp.asarray([2.0, -4.0])}
        (
            combined,
            physical_norm,
            teacher_norm,
            cosine,
            contribution_norm,
            conflicting,
            physical_alignment,
        ) = _combine_conflict_safe_updates(
            physical,
            teacher,
            maximum_teacher_ratio=0.5,
        )
        self.assertAlmostEqual(float(physical_norm), 1.0, places=6)
        self.assertAlmostEqual(float(teacher_norm), np.sqrt(20.0), places=6)
        self.assertLess(float(cosine), 0.0)
        self.assertTrue(bool(conflicting))
        self.assertAlmostEqual(float(contribution_norm), 0.5, places=6)
        self.assertAlmostEqual(float(physical_alignment), 1.0, places=6)
        self.assertTrue(
            bool(jnp.allclose(combined["x"], jnp.asarray([-1.0, -0.5])))
        )

    def test_parameter_update_clip_bounds_actual_displacement(self):
        params = {
            "operator_u_real": jnp.asarray([1.0]),
            "operator_v_real": jnp.asarray([2.0]),
            "lift_kernel": jnp.asarray([3.0]),
        }
        updated = {
            "operator_u_real": jnp.asarray([0.9]),
            "operator_v_real": jnp.asarray([1.8]),
            "lift_kernel": jnp.asarray([2.7]),
        }
        clipped, original_norm, clipped_norm = _clip_parameter_update(
            params, updated, 0.1
        )
        actual = jax.tree_util.tree_map(
            lambda new, old: new - old, clipped, params
        )
        actual_norm = float(
            jnp.sqrt(sum(jnp.sum(value**2) for value in actual.values()))
        )
        self.assertGreater(float(original_norm), 0.1)
        self.assertAlmostEqual(float(clipped_norm), 0.1, places=6)
        self.assertAlmostEqual(actual_norm, 0.1, places=6)

    def test_aligned_auxiliary_gradient_is_weighted_without_amplification(self):
        physical = {"x": jnp.asarray([3.0, 4.0])}
        auxiliary = {"x": jnp.asarray([0.0, 2.0])}
        (
            combined,
            physical_norm,
            auxiliary_norm,
            cosine,
            contribution_norm,
            conflicting,
            physical_alignment,
        ) = _combine_conflict_safe_gradients(
            physical,
            auxiliary,
            auxiliary_weight=0.05,
            maximum_auxiliary_ratio=0.25,
        )
        self.assertAlmostEqual(float(physical_norm), 5.0, places=6)
        self.assertAlmostEqual(float(auxiliary_norm), 2.0, places=6)
        self.assertAlmostEqual(float(contribution_norm), 0.1, places=6)
        self.assertAlmostEqual(float(cosine), 0.8, places=6)
        self.assertFalse(bool(conflicting))
        self.assertGreater(float(physical_alignment), 1.0)
        self.assertTrue(
            bool(jnp.allclose(combined["x"], physical["x"] + 0.05 * auxiliary["x"]))
        )

    def test_conflicting_auxiliary_gradient_is_projected_and_capped(self):
        physical = {"x": jnp.asarray([1.0, 0.0])}
        auxiliary = {"x": jnp.asarray([-2.0, 3.0])}
        (
            combined,
            _,
            _,
            cosine,
            contribution_norm,
            conflicting,
            physical_alignment,
        ) = _combine_conflict_safe_gradients(
            physical,
            auxiliary,
            auxiliary_weight=1.0,
            maximum_auxiliary_ratio=0.25,
        )
        self.assertTrue(bool(conflicting))
        self.assertLess(float(cosine), 0.0)
        self.assertAlmostEqual(float(contribution_norm), 0.25, places=6)
        self.assertAlmostEqual(float(physical_alignment), 1.0, places=6)
        self.assertTrue(bool(jnp.allclose(combined["x"], jnp.asarray([1.0, 0.25]))))

    def test_zero_auxiliary_weight_recovers_physical_gradient(self):
        physical = {"x": jnp.asarray([1.0, -2.0])}
        auxiliary = {"x": jnp.asarray([7.0, 3.0])}
        combined, *diagnostics = _combine_conflict_safe_gradients(
            physical,
            auxiliary,
            auxiliary_weight=0.0,
            maximum_auxiliary_ratio=0.25,
        )
        self.assertTrue(bool(jnp.allclose(combined["x"], physical["x"])))
        self.assertAlmostEqual(float(diagnostics[3]), 0.0, places=7)

    def test_atomic_checkpoint_replaces_complete_npz(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            _atomic_savez(path, {"value": np.asarray([1.0, 2.0])})
            with np.load(path, allow_pickle=False) as payload:
                self.assertTrue(np.array_equal(payload["value"], [1.0, 2.0]))
            self.assertEqual(list(Path(directory).glob("*.tmp*")), [])

    def test_first_nonfinite_step_reports_one_based_rollout_index(self):
        resolved = np.zeros((4, 3, 8), dtype=np.float32)
        latent = np.zeros((4, 2, 8), dtype=np.float32)
        latent[2, 1, 4] = np.nan
        self.assertEqual(_first_nonfinite_step(resolved, latent), 3)
        latent[2, 1, 4] = 0.0
        self.assertIsNone(_first_nonfinite_step(resolved, latent))

    def test_correction_bound_calibration_meets_channelwise_energy_tolerance(self):
        residual = np.zeros((5, 2, 4), dtype=np.float64)
        residual[:, 0] = np.linspace(-0.2, 0.2, 20).reshape(5, 4)
        residual[:, 1] = np.linspace(-8.0, 8.0, 20).reshape(5, 4)

        def blocks():
            yield residual

        bounds, diagnostics = _calibrate_smooth_correction_bounds(
            blocks,
            channels=2,
            energy_tolerance=1e-3,
            bisection_steps=16,
        )
        self.assertGreaterEqual(float(bounds[0]), 1.0)
        self.assertGreater(float(bounds[1]), float(bounds[0]))
        self.assertLessEqual(
            float(np.max(diagnostics["channel_distortion_fraction"])),
            1.01e-3,
        )

    def test_physical_terms_match_previous_four_channel_geometry(self):
        nx = 16
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=0.25)
        x = jnp.arange(nx, dtype=jnp.float32) * 0.25
        target = jnp.stack(
            (
                0.03 * jnp.cos(x),
                0.02 * jnp.sin(2.0 * x),
                0.01 * jnp.cos(3.0 * x),
            ),
            axis=0,
        )[None]
        predicted_state = resolved_hermite_to_low_moment_state(target).at[:, 0].add(
            0.005 * jnp.cos(2.0 * x)
        )
        numerator, denominator, _, _, _, _ = _physical_trajectory_terms(
            predicted_state,
            target,
            k_arr,
            poisson_sign=1.0,
        )

        predicted_fields = primitive_fields(predicted_state, k_arr)
        target_fields = primitive_fields(
            resolved_hermite_to_low_moment_state(target), k_arr
        )
        expected_numerator = jnp.sum(
            jnp.square(predicted_fields - target_fields), axis=-1
        )
        expected_denominator = jnp.sum(jnp.square(target_fields), axis=-1)

        self.assertTrue(bool(jnp.allclose(numerator, expected_numerator)))
        self.assertTrue(bool(jnp.allclose(denominator, expected_denominator)))
        self.assertAlmostEqual(
            float(_physical_sample_loss(numerator, denominator)[0]),
            float(jnp.mean(expected_numerator / expected_denominator)),
            places=6,
        )

    def test_zero_physical_error_has_zero_loss(self):
        nx = 16
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=0.25)
        x = jnp.arange(nx, dtype=jnp.float32) * 0.25
        target = jnp.stack(
            (
                0.03 * jnp.cos(x),
                0.02 * jnp.sin(2.0 * x),
                0.01 * jnp.cos(3.0 * x),
            ),
            axis=0,
        )[None]
        state = resolved_hermite_to_low_moment_state(target)
        numerator, denominator, spectral_numerator, _, _, _ = _physical_trajectory_terms(
            state,
            target,
            k_arr,
            poisson_sign=1.0,
        )
        self.assertTrue(bool(jnp.all(denominator > 0.0)))
        self.assertAlmostEqual(
            float(_physical_sample_loss(numerator, denominator)[0]), 0.0, places=7
        )
        self.assertAlmostEqual(float(spectral_numerator[0]), 0.0, places=7)

    def test_spectral_amplitude_loss_ignores_phase_but_detects_amplitude(self):
        x = 2.0 * jnp.pi * jnp.arange(32, dtype=jnp.float32) / 32.0
        target = jnp.cos(3.0 * x)[None]
        phase_shifted = jnp.sin(3.0 * x)[None]
        scaled = 0.5 * target
        phase_num, phase_den = _electric_spectral_amplitude_terms(
            phase_shifted, target
        )
        scaled_num, scaled_den = _electric_spectral_amplitude_terms(scaled, target)
        self.assertAlmostEqual(float(phase_num[0] / phase_den[0]), 0.0, places=6)
        self.assertAlmostEqual(float(scaled_num[0] / scaled_den[0]), 0.25, places=4)

    def test_time_relative_field_loss_can_restart_a_zero_field(self):
        nx = 16
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=0.25)
        x = jnp.arange(nx, dtype=jnp.float32) * 0.25
        target = jnp.zeros((1, 3, nx), dtype=jnp.float32).at[:, 0].set(
            0.03 * jnp.cos(x)
        )

        def losses(amplitude):
            predicted = resolved_hermite_to_low_moment_state(amplitude * target)
            terms = _physical_trajectory_terms(
                predicted,
                target,
                k_arr,
                poisson_sign=1.0,
                electric_energy_floor=jnp.asarray([1e-6]),
                electric_time_relative_floor=jnp.asarray([1e-6]),
            )
            return terms[4][0], terms[5][0]

        log_gradient = jax.grad(lambda value: losses(value)[0])(0.0)
        relative_gradient = jax.grad(lambda value: losses(value)[1])(0.0)
        self.assertAlmostEqual(float(log_gradient), 0.0, places=7)
        self.assertLess(float(relative_gradient), -1.0)

    def test_chunk_log_growth_loss_distinguishes_growth_from_damping(self):
        floor = jnp.asarray([1e-6])
        target_initial = jnp.asarray([1.0])
        target_final = jnp.asarray([4.0])
        matching = _electric_chunk_log_growth_error(
            jnp.asarray([1.0]),
            jnp.asarray([4.0]),
            target_initial,
            target_final,
            floor,
        )
        damping = _electric_chunk_log_growth_error(
            jnp.asarray([1.0]),
            jnp.asarray([0.25]),
            target_initial,
            target_final,
            floor,
        )
        self.assertAlmostEqual(float(matching[0]), 0.0, places=7)
        self.assertGreater(float(damping[0]), 7.0)

    def test_chunked_gradient_loss_keeps_exact_complete_trajectory_value(self):
        nx = 8
        rank = 2
        horizon = 4
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=0.5)
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(40),
            resolved_channels=3,
            latent_rank=rank,
            width=6,
            depth=1,
            spectral_modes=nx // 2 + 1,
            operator_rank=2,
            kernel_size=3,
        )
        propagator = jnp.broadcast_to(
            jnp.eye(3 + rank, dtype=jnp.complex64),
            (nx // 2 + 1, 3 + rank, 3 + rank),
        )
        trajectory = 0.01 * jax.random.normal(
            jax.random.PRNGKey(41),
            (1, horizon + 1, 3 + rank, nx),
            dtype=jnp.float32,
        )
        basis = jnp.asarray(
            [[1.0, 0.5], [0.2, -0.1], [0.0, 0.3], [-0.2, 0.4]],
            dtype=jnp.float32,
        )
        loss, rollout, components = _make_functions(
            propagator=propagator,
            basis=basis,
            k_arr=k_arr,
            resolved_center=jnp.zeros((3,)),
            resolved_scale=jnp.ones((3,)),
            latent_center=jnp.zeros((rank,)),
            latent_scale=jnp.ones((rank,)),
            depth=1,
            fine_steps=1,
            fine_dt=0.01,
            horizon=horizon,
            poisson_sign=1.0,
            gradient_chunk_steps=2,
            latent_residual_scale=jnp.ones((rank,)),
            latent_residual_weight=0.0,
        )
        predicted_resolved, predicted_latent = rollout(
            params,
            trajectory[:, 0, :3],
            trajectory[:, 0, 3:],
        )
        numerator = jnp.zeros((1, 4), dtype=jnp.float32)
        denominator = jnp.zeros_like(numerator)
        for step in range(horizon):
            step_num, step_den, _, _, _, _ = _physical_trajectory_terms(
                resolved_hermite_to_low_moment_state(
                    predicted_resolved[:, step]
                ),
                trajectory[:, step + 1, :3],
                k_arr,
                poisson_sign=1.0,
            )
            numerator += step_num
            denominator += step_den
        expected = jnp.mean(_physical_sample_loss(numerator, denominator))
        self.assertAlmostEqual(float(loss(params, trajectory)), float(expected), places=6)
        total, physical, latent = components(params, trajectory)
        self.assertAlmostEqual(float(total), float(physical), places=6)
        current = trajectory[:, :-1].reshape(-1, 3 + rank, nx)
        following = trajectory[:, 1:].reshape(-1, 3 + rank, nx)
        linear = apply_coupled_linear_propagator(
            propagator,
            current[:, :3],
            current[:, 3:],
            resolved_scale=jnp.ones((3,)),
            latent_scale=jnp.ones((rank,)),
        )
        target_correction = bounded_normalized_latent_correction(
            following[:, 3:] - linear[:, 3:],
            jnp.ones((rank,)),
        )
        predicted_correction = state_conditioned_latent_operator_correction(
            params,
            current[:, :3],
            current[:, 3:],
            depth=1,
            equilibrium_preserving=True,
        )
        predicted_correction = bounded_normalized_latent_correction(
            predicted_correction,
            jnp.ones((rank,)),
        )
        expected_latent = jnp.mean(
            jnp.square(predicted_correction - target_correction)
        )
        self.assertAlmostEqual(float(latent), float(expected_latent), places=6)

    def test_nonzero_output_initialization_connects_all_gradient_groups(self):
        nx = 8
        rank = 2
        horizon = 2
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=0.5)
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(50),
            resolved_channels=3,
            latent_rank=rank,
            width=6,
            depth=1,
            spectral_modes=nx // 2 + 1,
            operator_rank=2,
            kernel_size=3,
            output_projection_init_scale=1e-3,
        )
        propagator = jnp.broadcast_to(
            jnp.eye(3 + rank, dtype=jnp.complex64),
            (nx // 2 + 1, 3 + rank, 3 + rank),
        )
        trajectory = 0.05 * jax.random.normal(
            jax.random.PRNGKey(51),
            (1, horizon + 1, 3 + rank, nx),
            dtype=jnp.float32,
        )
        loss, _, components = _make_functions(
            propagator=propagator,
            basis=jax.random.normal(jax.random.PRNGKey(52), (4, rank)),
            k_arr=k_arr,
            resolved_center=jnp.zeros((3,)),
            resolved_scale=jnp.ones((3,)),
            latent_center=jnp.zeros((rank,)),
            latent_scale=jnp.ones((rank,)),
            correction_bounds=jnp.asarray([2.0, 3.0]),
            depth=1,
            fine_steps=1,
            fine_dt=0.01,
            horizon=horizon,
            poisson_sign=1.0,
            gradient_chunk_steps=1,
            latent_residual_scale=jnp.asarray([0.2, 0.3]),
            latent_residual_weight=0.05,
        )
        total, physical, latent = components(params, trajectory)
        self.assertAlmostEqual(
            float(total), float(physical + 0.05 * latent), places=6
        )
        self.assertGreater(float(latent), 0.0)
        _, gradients = jax.value_and_grad(loss)(params, trajectory)
        group_norms = _gradient_group_norms(gradients)
        for group_norm in group_norms:
            self.assertGreater(float(group_norm), 0.0)

    def test_closure_residual_weight_targets_decoded_c3_direction(self):
        nx = 8
        rank = 2
        horizon = 2
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(60),
            resolved_channels=3,
            latent_rank=rank,
            width=6,
            depth=1,
            spectral_modes=nx // 2 + 1,
            operator_rank=2,
            kernel_size=3,
            output_projection_init_scale=1e-3,
        )
        propagator = jnp.broadcast_to(
            jnp.eye(3 + rank, dtype=jnp.complex64),
            (nx // 2 + 1, 3 + rank, 3 + rank),
        )
        trajectory = 0.05 * jax.random.normal(
            jax.random.PRNGKey(61),
            (1, horizon + 1, 3 + rank, nx),
            dtype=jnp.float32,
        )
        common = dict(
            propagator=propagator,
            basis=jnp.asarray(
                [[1.0, -0.5], [0.0, 1.0], [0.5, 0.0]], dtype=jnp.float32
            ),
            k_arr=2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=0.5),
            resolved_center=jnp.zeros((3,)),
            resolved_scale=jnp.ones((3,)),
            latent_center=jnp.zeros((rank,)),
            latent_scale=jnp.ones((rank,)),
            depth=1,
            fine_steps=1,
            fine_dt=0.01,
            horizon=horizon,
            poisson_sign=1.0,
            gradient_chunk_steps=1,
            latent_residual_scale=jnp.ones((rank,)),
            latent_residual_weight=1.0,
            closure_residual_scale=0.1,
        )
        _, _, baseline_components = _make_functions(
            **common, closure_residual_weight=0.0
        )
        _, _, closure_components = _make_functions(
            **common, closure_residual_weight=1.0
        )
        baseline_teacher = baseline_components(params, trajectory)[2]
        closure_teacher = closure_components(params, trajectory)[2]
        self.assertGreater(float(closure_teacher), float(baseline_teacher))

    def test_closure_only_teacher_loss_does_not_train_hidden_readout(self):
        nx = 8
        rank = 2
        params = init_state_conditioned_latent_operator(
            jax.random.PRNGKey(70),
            resolved_channels=3,
            latent_rank=rank,
            width=6,
            depth=1,
            spectral_modes=nx // 2 + 1,
            operator_rank=2,
            kernel_size=3,
            output_projection_init_scale=1e-3,
            closure_aligned_output=True,
        )
        propagator = jnp.broadcast_to(
            jnp.eye(3 + rank, dtype=jnp.complex64),
            (nx // 2 + 1, 3 + rank, 3 + rank),
        )
        trajectory = 0.05 * jax.random.normal(
            jax.random.PRNGKey(71),
            (1, 3, 3 + rank, nx),
            dtype=jnp.float32,
        )
        _, _, components = _make_functions(
            propagator=propagator,
            basis=jnp.asarray(
                [[1.0, -0.5], [0.0, 1.0], [0.5, 0.0]], dtype=jnp.float32
            ),
            k_arr=2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=0.5),
            resolved_center=jnp.zeros((3,)),
            resolved_scale=jnp.ones((3,)),
            latent_center=jnp.zeros((rank,)),
            latent_scale=jnp.ones((rank,)),
            depth=1,
            fine_steps=1,
            fine_dt=0.01,
            horizon=2,
            poisson_sign=1.0,
            gradient_chunk_steps=1,
            latent_residual_scale=jnp.ones((rank,)),
            latent_residual_weight=1.0,
            latent_state_residual_weight=0.0,
            closure_residual_scale=0.1,
            closure_residual_weight=1.0,
            closure_aligned_output=True,
        )
        teacher_loss = lambda candidate: components(candidate, trajectory)[2]
        gradients = jax.grad(teacher_loss)(params)
        hidden_norm = jnp.sqrt(
            jnp.sum(jnp.square(gradients["operator_u_real"]))
            + jnp.sum(jnp.square(gradients["operator_u_imag"]))
        )
        closure_norm = jnp.sqrt(
            jnp.sum(jnp.square(gradients["operator_u_closure_real"]))
            + jnp.sum(jnp.square(gradients["operator_u_closure_imag"]))
        )
        self.assertLess(float(hidden_norm), 1e-7)
        self.assertGreater(float(closure_norm), 0.0)


if __name__ == "__main__":
    unittest.main()
