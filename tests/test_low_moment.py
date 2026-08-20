import math
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from model.train.low_moment_closure import (
    _block_relative_sample_loss,
    _electric_field_energy,
    build_complete_trajectory_case_batches,
    build_diagnostic_panel,
    low_hermite_coefficients_to_conservative,
    make_continuous_chunk_loss_function,
    parse_horizon_curriculum,
    sample_complete_trajectory_batch,
)
from vpml.low_moment import (
    explicit_window_closure_step,
    init_explicit_window_params,
    init_spectral_memory_params,
    low_moment_rk4_step,
    primitive_fields,
    spectral_memory_closure_step,
    rollout_explicit_window_closure,
)


class LowMomentClosureTests(unittest.TestCase):
    def test_explicit_window_is_history_sensitive_and_translation_equivariant(self) -> None:
        nx = 32
        width = 8
        memory_steps = 6
        shift = 5
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_explicit_window_params(
            jax.random.PRNGKey(41),
            width=width,
            spectral_modes=8,
            memory_steps=memory_steps,
        )
        params["output_local"] = jnp.ones((1, width), dtype=jnp.float32)
        x = jnp.linspace(0.0, 4.0 * jnp.pi, nx, endpoint=False)
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32)
        history_a = jnp.zeros((1, memory_steps, 3, nx), dtype=jnp.float32)
        history_b = history_a.at[:, 0, 0].set(0.05 * jnp.cos(x))
        kwargs = dict(
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
        )
        gradient_a = explicit_window_closure_step(
            params, state, history_a, jnp.asarray([0.1]), k_arr, **kwargs
        )
        gradient_b = explicit_window_closure_step(
            params, state, history_b, jnp.asarray([0.1]), k_arr, **kwargs
        )
        self.assertGreater(float(jnp.linalg.norm(gradient_b - gradient_a)), 1e-7)
        gradient_shifted = explicit_window_closure_step(
            params,
            jnp.roll(state, shift, axis=-1),
            jnp.roll(history_b, shift, axis=-1),
            jnp.asarray([0.1]),
            k_arr,
            **kwargs,
        )
        np.testing.assert_allclose(
            np.asarray(gradient_shifted),
            np.roll(np.asarray(gradient_b), shift, axis=-1),
            atol=3e-6,
        )

    def test_explicit_window_equilibrium_and_stride_update(self) -> None:
        nx = 16
        memory_steps = 4
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_explicit_window_params(
            jax.random.PRNGKey(42),
            width=6,
            spectral_modes=5,
            memory_steps=memory_steps,
        )
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32)
        history = jnp.zeros((1, memory_steps, 3, nx), dtype=jnp.float32)
        states, (history_new, encoded, counter, gradient) = rollout_explicit_window_closure(
            params,
            state,
            history,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray([0.1]),
            k_arr,
            horizon=3,
            dt=0.01,
            memory_stride=2,
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
        )
        np.testing.assert_array_equal(np.asarray(states), np.zeros_like(states))
        np.testing.assert_array_equal(np.asarray(history_new), np.zeros_like(history_new))
        np.testing.assert_array_equal(np.asarray(encoded), np.zeros_like(encoded))
        np.testing.assert_array_equal(np.asarray(gradient), np.zeros_like(gradient))
        self.assertEqual(int(counter), 1)

    def test_block_relative_loss_weights_fixed_time_blocks_equally(self) -> None:
        target = jnp.ones((1, 4, 4, 2), dtype=jnp.float32)
        predicted = target.at[:, :2].add(1.0).at[:, 2:].add(2.0)
        value = _block_relative_sample_loss(
            predicted,
            target,
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32),
            block_steps=2,
            block_count=2,
            floor_rms=jnp.full((3, 4), 0.1, dtype=jnp.float32),
        )
        np.testing.assert_allclose(np.asarray(value), [2.5], rtol=1e-6)

    def test_block_relative_loss_uses_convergence_floor_at_zero_signal(self) -> None:
        target = jnp.zeros((1, 2, 4, 2), dtype=jnp.float32)
        predicted = jnp.ones_like(target)
        value = _block_relative_sample_loss(
            predicted,
            target,
            jnp.asarray([2], dtype=jnp.int32),
            jnp.asarray([[1, 2]], dtype=jnp.int32),
            block_steps=2,
            block_count=1,
            floor_rms=jnp.ones((3, 4), dtype=jnp.float32),
        )
        np.testing.assert_allclose(np.asarray(value), [1.0], rtol=1e-6)

    def test_block_relative_chunk_losses_sum_to_complete_loss(self) -> None:
        target = jnp.ones((1, 4, 4, 2), dtype=jnp.float32)
        predicted = target.at[:, :2].add(1.0).at[:, 2:].add(2.0)
        full_norm = jnp.full((1, 2, 4), 4.0, dtype=jnp.float32)
        values = []
        for start in (0, 2):
            values.append(
                _block_relative_sample_loss(
                    predicted[:, start : start + 2],
                    target[:, start : start + 2],
                    jnp.asarray([0], dtype=jnp.int32),
                    jnp.asarray([[start + 1, start + 2]], dtype=jnp.int32),
                    block_steps=2,
                    block_count=2,
                    floor_rms=jnp.full((3, 4), 0.1, dtype=jnp.float32),
                    trajectory_target_norm=full_norm,
                )
            )
        np.testing.assert_allclose(np.asarray(values[0] + values[1]), [2.5], rtol=1e-6)

    def test_continuous_chunks_carry_state_and_memory_without_reset(self) -> None:
        nx = 16
        width = 6
        k_arr = 2.0 * np.pi * np.fft.rfftfreq(nx, d=4.0 * np.pi / nx)
        params = init_spectral_memory_params(
            jax.random.PRNGKey(8), width=width, spectral_modes=4
        )
        x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
        initial = np.zeros((3, 3, nx), dtype=np.float32)
        initial[:, 0] = 0.01 * np.cos(x)
        initial[:, 1] = 0.005 * np.sin(x)
        initial[:, 2] = 0.01 * np.sin(2.0 * x)
        memory = np.repeat(initial[:, None], 3, axis=1)
        targets = np.repeat(initial[:, None], 4, axis=1)
        amplitude = np.asarray([0.01, 0.1, 0.5], dtype=np.float32)
        regime_index = np.arange(3, dtype=np.int32)
        common = dict(
            k_arr=k_arr,
            width=width,
            dt=0.01,
            input_scale=np.ones((4,), dtype=np.float32),
            regime_scales=np.ones((3, 4), dtype=np.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
            poisson_sign=1.0,
            normalized_heat_flux_bound=8.0,
            density_floor=1e-4,
            pressure_floor=1e-4,
        )
        full = make_continuous_chunk_loss_function(
            horizon=4, warm_memory=True, **common
        )
        _, (_, full_state, full_hidden) = full(
            params, initial, memory, targets, amplitude, regime_index
        )
        first = make_continuous_chunk_loss_function(
            horizon=2, warm_memory=True, **common
        )
        _, (_, state, hidden) = first(
            params, initial, memory, targets[:, :2], amplitude, regime_index
        )
        second = make_continuous_chunk_loss_function(
            horizon=2, warm_memory=False, **common
        )
        _, (_, state, hidden) = second(
            params, state, hidden, targets[:, 2:], amplitude, regime_index
        )
        np.testing.assert_allclose(state, full_state, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(hidden, full_hidden, rtol=2e-6, atol=2e-6)

    def test_complete_trajectory_batches_cover_each_ic_once(self) -> None:
        grouped = {
            regime: {
                "case_splits": np.asarray(
                    ["train", "heldout", "train", "train", "heldout"]
                )
            }
            for regime in (
                "linear_landau",
                "nonlinear_landau_weak",
                "nonlinear_landau_strong",
            )
        }
        batches = build_complete_trajectory_case_batches(
            np.random.default_rng(7),
            grouped,
            split="train",
            batch_size_per_regime=2,
            shuffle=True,
        )
        self.assertEqual(len(batches), 2)
        for regime in grouped:
            selected = np.concatenate([batch[regime] for batch in batches])
            np.testing.assert_array_equal(np.sort(selected), [0, 2, 3])

    def test_complete_trajectory_batch_uses_t0_once_and_every_future_target(self) -> None:
        regimes = (
            "linear_landau",
            "nonlinear_landau_weak",
            "nonlinear_landau_strong",
        )
        nx = 4
        histories = []
        cases = []
        grouped = {}
        for regime_index, regime in enumerate(regimes):
            coefficient_history = np.zeros((6, 4, nx // 2 + 1), dtype=np.complex64)
            coefficient_history[:, 0, 0] = nx * np.arange(6, dtype=np.float32)
            histories.append(coefficient_history)
            case_id = f"{regime}_ic00"
            grouped[regime] = {
                "case_ids": np.asarray([case_id]),
                "case_splits": np.asarray(["train"]),
                "coefficients": (coefficient_history,),
            }
            cases.append({"case_id": case_id, "epsilon": 0.1 + regime_index})
        batch = sample_complete_trajectory_batch(
            np.random.default_rng(1),
            grouped,
            {"cases": cases},
            "coefficients",
            {regime: np.asarray([0], dtype=np.int32) for regime in regimes},
            memory_steps=3,
            memory_stride=2,
            source_nx=nx,
            rollout_nx=nx,
            translation_augmentation=False,
        )
        self.assertEqual(batch["memory"].shape, (3, 3, 3, nx))
        self.assertEqual(batch["targets"].shape, (3, 5, 3, nx))
        np.testing.assert_allclose(batch["initial"][:, 0], 0.0)
        np.testing.assert_allclose(batch["memory"][:, :, 0], 0.0)
        expected_density = np.broadcast_to(
            np.arange(1, 6, dtype=np.float32)[:, None], (5, nx)
        )
        np.testing.assert_allclose(batch["targets"][0, :, 0], expected_density)

    def test_diagnostic_panel_covers_every_heldout_ic_at_fixed_times(self) -> None:
        regimes = (
            "linear_landau",
            "nonlinear_landau_weak",
            "nonlinear_landau_strong",
        )
        grouped = {}
        cases = []
        for regime_index, regime in enumerate(regimes):
            case_ids = np.asarray([f"{regime}_ic{index:02d}" for index in range(4)])
            grouped[regime] = {
                "case_ids": case_ids,
                "case_splits": np.asarray(["train", "heldout", "heldout", "train"]),
                "coefficients": tuple(
                    np.zeros((64, 4, 3), dtype=np.complex64) for _ in range(4)
                ),
            }
            cases.extend(
                {
                    "case_id": case_id,
                    "epsilon": float(regime_index + 1) + 0.1 * index,
                }
                for index, case_id in enumerate(case_ids)
            )
        captured = []

        def capture_batch(*args, **kwargs):
            captured.append(kwargs["explicit_selection"])
            return {"selection_index": np.asarray([len(captured)])}

        with mock.patch(
            "model.train.low_moment_closure.sample_batch", side_effect=capture_batch
        ):
            panel = build_diagnostic_panel(
                np.random.default_rng(0),
                grouped,
                anchors={},
                manifest={"cases": cases},
                coefficient_key="coefficients",
                split="val",
                start_times=(0.0, 20.0),
                cases_per_regime=None,
                horizon=10,
                dt=1.0,
                memory_steps=2,
                memory_stride=1,
                source_nx=4,
                rollout_nx=4,
                domain_length=1.0,
            )

        self.assertEqual(len(panel), 2)
        for time_index, selections in zip((0, 20), captured):
            for regime in regimes:
                np.testing.assert_array_equal(selections[regime][0], [1, 2])
                np.testing.assert_array_equal(
                    selections[regime][1], [time_index, time_index]
                )

    def test_energy_diagnostic_preserves_large_finite_float32_fields(self) -> None:
        field_hat = np.zeros((1, 17), dtype=np.complex64)
        field_hat[0, 1] = np.complex64(1e30)
        energy = _electric_field_energy(field_hat, nx=32, dx=0.25)
        self.assertTrue(np.isfinite(energy[0]))
        self.assertGreater(energy[0], 1e50)

    def test_horizon_curriculum_ends_at_requested_horizon(self) -> None:
        self.assertEqual(
            parse_horizon_curriculum(
                "1:10,8:10,32:20,128:60", final_horizon=128, total_epochs=100
            ),
            ((1, 10), (8, 10), (32, 20), (128, 60)),
        )

    def test_low_hermite_coefficients_map_to_conservative_moments(self) -> None:
        nx = 32
        x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
        c0 = 0.1 * np.cos(x)
        c1 = 0.2 * np.sin(2.0 * x)
        c2 = -0.05 * np.cos(3.0 * x)
        coefficients = np.stack(
            [np.fft.rfft(c0), np.fft.rfft(c1), np.fft.rfft(c2)], axis=0
        )
        state = low_hermite_coefficients_to_conservative(
            coefficients, source_nx=nx, target_nx=nx, dtype=np.float64
        )
        np.testing.assert_allclose(state[0], c0, atol=1e-12)
        np.testing.assert_allclose(state[1], c1, atol=1e-12)
        np.testing.assert_allclose(
            state[2], c0 + math.sqrt(2.0) * c2, atol=1e-12
        )

    def test_centered_conversion_preserves_sub_float32_epsilon_perturbation(self) -> None:
        nx = 32
        x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
        perturbation = 1e-11 * np.cos(x)
        coefficients = np.zeros((3, nx // 2 + 1), dtype=np.complex64)
        coefficients[0] = np.fft.rfft(perturbation).astype(np.complex64)
        state = low_hermite_coefficients_to_conservative(
            coefficients, source_nx=nx, target_nx=nx, dtype=np.float32
        )
        np.testing.assert_allclose(state[0], perturbation, rtol=2e-6, atol=1e-18)

    def test_equilibrium_is_exact_fixed_point(self) -> None:
        nx = 32
        width = 8
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_spectral_memory_params(
            jax.random.PRNGKey(0), width=width, spectral_modes=6
        )
        state = jnp.zeros((2, 3, nx), dtype=jnp.float32)
        hidden = jnp.zeros((2, width, nx), dtype=jnp.float32)
        hidden_new, heat_flux_gradient = spectral_memory_closure_step(
            params,
            state,
            hidden,
            jnp.asarray([0.01, 0.5], dtype=jnp.float32),
            k_arr,
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
        )
        updated = low_moment_rk4_step(state, heat_flux_gradient, k_arr, 0.01)
        np.testing.assert_array_equal(np.asarray(hidden_new), np.zeros_like(hidden_new))
        np.testing.assert_array_equal(
            np.asarray(heat_flux_gradient), np.zeros_like(heat_flux_gradient)
        )
        np.testing.assert_array_equal(np.asarray(updated), np.asarray(state))

    def test_rk4_preserves_sub_epsilon_centered_uniform_state(self) -> None:
        nx = 32
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(
            nx, d=4.0 * jnp.pi / nx
        )
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32)
        state = state.at[:, 0].set(1e-11).at[:, 2].set(1e-11)
        updated = low_moment_rk4_step(
            state,
            jnp.zeros((1, nx), dtype=jnp.float32),
            k_arr,
            0.01,
        )
        np.testing.assert_allclose(
            np.asarray(updated), np.asarray(state), rtol=2e-6, atol=1e-18
        )

    def test_direct_heat_flux_gradient_has_zero_spatial_mean(self) -> None:
        nx = 32
        width = 6
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_spectral_memory_params(
            jax.random.PRNGKey(12),
            width=width,
            spectral_modes=8,
            input_channels=5,
        )
        params["output_local"] = jnp.ones((1, width), dtype=jnp.float32)
        x = jnp.linspace(
            0.0, 2.0 * jnp.pi, nx, endpoint=False, dtype=jnp.float32
        )
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32)
        state = state.at[:, 0].set(0.1 * jnp.cos(x))
        state = state.at[:, 2].set(0.1 * jnp.sin(x))
        hidden = jnp.zeros((1, width, nx), dtype=jnp.float32)
        common = dict(
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=0.0,
            amplitude_scale=1.0,
            previous_heat_flux_gradient=jnp.zeros((1, nx), dtype=jnp.float32),
        )
        _, gradient = spectral_memory_closure_step(
            params, state, hidden, jnp.ones((1,)), k_arr, **common
        )
        np.testing.assert_allclose(np.mean(np.asarray(gradient), axis=-1), 0.0, atol=1e-7)

    def test_relative_trajectory_loss_has_no_additive_denominator_floor(self) -> None:
        nx = 8
        width = 4
        k_arr = 2.0 * np.pi * np.fft.rfftfreq(nx, d=4.0 * np.pi / nx)
        params = init_spectral_memory_params(
            jax.random.PRNGKey(13), width=width, spectral_modes=3
        )
        equilibrium = np.zeros((3, 3, nx), dtype=np.float32)
        loss = make_continuous_chunk_loss_function(
            k_arr=k_arr,
            width=width,
            horizon=1,
            warm_memory=True,
            dt=0.01,
            input_scale=np.ones((4,), dtype=np.float32),
            regime_scales=np.ones((3, 4), dtype=np.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=0.0,
            amplitude_scale=1.0,
            poisson_sign=1.0,
            normalized_heat_flux_bound=8.0,
            density_floor=1e-4,
            pressure_floor=1e-4,
            relative_trajectory_loss=True,
        )
        value, _ = loss(
            params,
            equilibrium,
            equilibrium[:, None],
            equilibrium[:, None],
            np.ones((3,), dtype=np.float32),
            np.arange(3, dtype=np.int32),
        )
        self.assertTrue(np.isnan(float(value)))

    def test_rk4_step_restores_density_and_pressure_admissibility(self) -> None:
        nx = 32
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32)
        x = jnp.linspace(
            0.0, 2.0 * jnp.pi, nx, endpoint=False, dtype=jnp.float32
        )
        density = 1.0 + 1.2 * jnp.cos(x)
        state = state.at[:, 0].set(density - 1.0)
        state = state.at[:, 1].set(jnp.asarray(0.4, dtype=state.dtype))
        state = state.at[:, 2].set(jnp.asarray(-0.98, dtype=state.dtype))
        updated = low_moment_rk4_step(
            state,
            jnp.zeros((1, nx), dtype=jnp.float32),
            k_arr,
            0.0,
        )
        fields = primitive_fields(updated, k_arr)
        self.assertGreaterEqual(float(jnp.min(updated[:, 0] + 1.0)), 0.999e-4)
        self.assertGreaterEqual(float(jnp.min(fields[:, 2] + 1.0)), 0.999e-4)
        np.testing.assert_allclose(
            np.asarray(jnp.mean(updated[:, 0] + 1.0, axis=-1)),
            np.asarray(jnp.mean(state[:, 0] + 1.0, axis=-1)),
            rtol=1e-6,
        )

    def test_closure_output_remains_finite_for_extreme_amplitude_gain(self) -> None:
        nx = 32
        width = 6
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_spectral_memory_params(
            jax.random.PRNGKey(5), width=width, spectral_modes=8
        )
        params["amplitude_gain"] = jnp.full((4,), 1e6, dtype=jnp.float32)
        params["output_local"] = jnp.full((1, width), 1e6, dtype=jnp.float32)
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32)
        hidden = jnp.ones((1, width, nx), dtype=jnp.float32)
        _, gradient = spectral_memory_closure_step(
            params,
            state,
            hidden,
            jnp.asarray([0.65], dtype=jnp.float32),
            k_arr,
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-3.0,
            amplitude_scale=1.0,
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))

    def test_closure_is_equivariant_to_discrete_translation(self) -> None:
        nx = 32
        width = 6
        shift = 7
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_spectral_memory_params(
            jax.random.PRNGKey(3), width=width, spectral_modes=8
        )
        rng = np.random.default_rng(4)
        state = np.zeros((2, 3, nx), dtype=np.float32)
        state[:, 0] = 0.03 * rng.normal(size=(2, nx))
        state[:, 1] = 0.02 * rng.normal(size=(2, nx))
        state[:, 2] = 0.04 * rng.normal(size=(2, nx))
        hidden = 0.01 * rng.normal(size=(2, width, nx)).astype(np.float32)
        kwargs = dict(
            input_scale=jnp.asarray([0.1, 0.1, 0.1, 0.1], dtype=jnp.float32),
            heat_flux_gradient_scale=0.2,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
        )
        hidden_a, gradient_a = spectral_memory_closure_step(
            params,
            jnp.asarray(state),
            jnp.asarray(hidden),
            jnp.asarray([0.1, 0.3], dtype=jnp.float32),
            k_arr,
            **kwargs,
        )
        hidden_b, gradient_b = spectral_memory_closure_step(
            params,
            jnp.asarray(np.roll(state, shift, axis=-1)),
            jnp.asarray(np.roll(hidden, shift, axis=-1)),
            jnp.asarray([0.1, 0.3], dtype=jnp.float32),
            k_arr,
            **kwargs,
        )
        np.testing.assert_allclose(
            np.asarray(hidden_b), np.roll(np.asarray(hidden_a), shift, axis=-1), atol=2e-6
        )
        np.testing.assert_allclose(
            np.asarray(gradient_b),
            np.roll(np.asarray(gradient_a), shift, axis=-1),
            atol=2e-6,
        )


if __name__ == "__main__":
    unittest.main()
