import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from model.diagnostics.low_moment_resonance_identifiability import (
    mode_bounce_exposure,
)
from model.train.low_moment_closure import (
    _block_relative_sample_loss,
    _electric_field_energy,
    _load_or_build_primitive_target_cache,
    _primitive_numpy,
    build_complete_trajectory_case_batches,
    build_diagnostic_panel,
    limit_training_anchors,
    low_hermite_coefficients_to_conservative,
    make_loss_function,
    make_continuous_chunk_loss_function,
    parse_horizon_curriculum,
    parse_train_case_limits,
    sample_complete_trajectory_batch,
)
from vpml.low_moment import (
    DYNAMIC_INPUT_SCALING,
    _dealias,
    _causal_temporal_convolution,
    electric_field_from_density,
    explicit_window_closure_step,
    init_causal_spacetime_operator_params,
    init_explicit_window_params,
    init_spectral_memory_params,
    init_window_fno_params,
    low_moment_rhs,
    low_moment_rk4_step,
    primitive_fields,
    spectral_derivative,
    spectral_memory_closure_step,
    rollout_explicit_window_closure,
)


class LowMomentClosureTests(unittest.TestCase):
    def test_mode_bounce_exposure_integrates_constant_frequency(self) -> None:
        density_hat = np.zeros((4, 3), dtype=np.complex128)
        density_hat[:, 1] = 4.0
        frequency, exposure = mode_bounce_exposure(
            density_hat,
            source_nx=16,
            domain_length=4.0 * math.pi,
            dt=0.25,
            modes=2,
        )
        expected_frequency = math.sqrt(0.5)
        np.testing.assert_allclose(frequency[:, 0], expected_frequency)
        np.testing.assert_allclose(
            exposure[:, 0], expected_frequency * np.arange(4) * 0.25
        )
        np.testing.assert_array_equal(frequency[:, 1], 0.0)
        np.testing.assert_array_equal(exposure[:, 1], 0.0)

    def test_batched_rhs_matches_unfused_pseudospectral_operations(self) -> None:
        nx = 64
        batch = 3
        x = jnp.linspace(0.0, 4.0 * jnp.pi, nx, endpoint=False)
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        phase = jnp.arange(batch, dtype=jnp.float32)[:, None] * 0.23
        state = jnp.stack(
            (
                0.08 * jnp.cos(x[None] + phase),
                0.03 * jnp.sin(2.0 * x[None] - phase),
                0.05 * jnp.cos(3.0 * x[None] + 0.5 * phase),
            ),
            axis=1,
        )
        closure = 0.01 * jnp.sin(4.0 * x[None] + phase)

        density_perturbation, momentum, second_perturbation = (
            state[:, 0],
            state[:, 1],
            state[:, 2],
        )
        rho = 1.0 + density_perturbation
        velocity = momentum / jnp.maximum(rho, 1e-4)
        pressure = jnp.maximum(
            1.0 + second_perturbation - momentum * velocity, 1e-4
        )
        field = electric_field_from_density(density_perturbation, k_arr)
        third_flux = _dealias(rho * velocity**3 + 3.0 * velocity * pressure)
        expected = jnp.stack(
            (
                -spectral_derivative(momentum, k_arr),
                -spectral_derivative(second_perturbation, k_arr)
                - _dealias(rho * field),
                -spectral_derivative(third_flux, k_arr)
                - closure
                - 2.0 * _dealias(momentum * field),
            ),
            axis=1,
        )
        actual = low_moment_rhs(state, closure, k_arr)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_primitive_target_cache_matches_direct_conversion_and_reuses_files(self) -> None:
        nx = 16
        steps = 7
        k_count = nx // 2 + 1
        grouped = {}
        cases = []
        x_mode = np.arange(steps, dtype=np.float32)
        for regime_index, regime in enumerate(
            (
                "linear_landau",
                "nonlinear_landau_weak",
                "nonlinear_landau_strong",
            )
        ):
            case_id = f"{regime}_ic00"
            history = np.zeros((steps, 4, k_count), dtype=np.complex64)
            history[:, 0, 1] = (0.01 + 0.01 * regime_index) * np.exp(
                0.1j * x_mode
            )
            history[:, 1, 1] = 0.005j * (regime_index + 1)
            history[:, 2, 2] = 0.003 * (regime_index + 1)
            grouped[regime] = {
                "case_ids": np.asarray([case_id]),
                "case_splits": np.asarray(["train"]),
                "coefficients": (history,),
            }
            cases.append({"case_id": case_id, "epsilon": 0.1})
        manifest = {"sha256": "test-manifest", "cases": cases}

        with tempfile.TemporaryDirectory() as directory:
            cached, norms, root = _load_or_build_primitive_target_cache(
                Path(directory),
                grouped,
                manifest,
                "coefficients",
                source_nx=nx,
                rollout_nx=nx,
                domain_length=4.0 * math.pi,
                poisson_sign=1.0,
                chunk_steps=3,
            )
            for regime in grouped:
                direct_state = low_hermite_coefficients_to_conservative(
                    grouped[regime]["coefficients"][0][:, :3],
                    source_nx=nx,
                    target_nx=nx,
                    dtype=np.float32,
                )
                direct = _primitive_numpy(direct_state, 4.0 * math.pi)
                k_arr = 2.0 * np.pi * np.fft.rfftfreq(
                    nx, d=4.0 * math.pi / nx
                )
                jax_direct = np.asarray(
                    primitive_fields(jnp.asarray(direct_state), jnp.asarray(k_arr))
                )
                np.testing.assert_allclose(cached[regime][0], direct, rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(
                    cached[regime][0], jax_direct, rtol=2e-6, atol=2e-6
                )
                case_id = str(grouped[regime]["case_ids"][0])
                expected_norm = np.sum(np.square(direct[1:], dtype=np.float64), axis=(0, 2))
                np.testing.assert_allclose(norms[case_id], expected_norm, rtol=2e-6)
            self.assertTrue((root / "metadata.json").is_file())
            with mock.patch(
                "model.train.low_moment_closure.low_hermite_coefficients_to_conservative",
                side_effect=AssertionError("cache should be reused"),
            ):
                reused, reused_norms, reused_root = _load_or_build_primitive_target_cache(
                    Path(directory),
                    grouped,
                    manifest,
                    "coefficients",
                    source_nx=nx,
                    rollout_nx=nx,
                    domain_length=4.0 * math.pi,
                    poisson_sign=1.0,
                    chunk_steps=3,
                )
            self.assertEqual(reused_root, root)
            self.assertEqual(reused["linear_landau"][0].dtype, np.float32)
            np.testing.assert_allclose(
                reused_norms["linear_landau_ic00"], norms["linear_landau_ic00"]
            )

    def test_window_fno_is_equivariant_history_sensitive_and_zero_mean(self) -> None:
        nx = 32
        memory_steps = 6
        shift = 5
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_window_fno_params(
            jax.random.PRNGKey(104),
            width=8,
            spectral_modes=9,
            memory_steps=memory_steps,
            depth=3,
            input_channels=5,
        )
        params["output_local"] = jnp.ones((1, 8), dtype=jnp.float32) / 8.0
        x = jnp.linspace(0.0, 4.0 * jnp.pi, nx, endpoint=False)
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32).at[:, 0].set(
            0.2 * jnp.cos(x) + 0.05 * jnp.cos(2.0 * x + 0.3)
        )
        history_a = jnp.repeat(state[:, None], memory_steps, axis=1)
        history_b = history_a.at[:, 0, 1].set(0.04 * jnp.sin(3.0 * x))
        closure_history = jnp.zeros((1, memory_steps, nx), dtype=jnp.float32)
        previous = jnp.zeros((1, nx), dtype=jnp.float32)
        kwargs = dict(
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
            previous_heat_flux_gradient=previous,
            heat_flux_gradient_history=closure_history,
        )
        gradient_a = explicit_window_closure_step(
            params, state, history_a, jnp.asarray([0.2]), k_arr, **kwargs
        )
        gradient_b = explicit_window_closure_step(
            params, state, history_b, jnp.asarray([0.2]), k_arr, **kwargs
        )
        self.assertGreater(float(jnp.linalg.norm(gradient_b - gradient_a)), 1e-7)
        self.assertAlmostEqual(float(jnp.mean(gradient_b)), 0.0, places=6)
        shifted = explicit_window_closure_step(
            params,
            jnp.roll(state, shift, axis=-1),
            jnp.roll(history_b, shift, axis=-1),
            jnp.asarray([0.2]),
            k_arr,
            previous_heat_flux_gradient=jnp.roll(previous, shift, axis=-1),
            heat_flux_gradient_history=jnp.roll(
                closure_history, shift, axis=-1
            ),
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
        )
        np.testing.assert_allclose(
            np.asarray(shifted),
            np.roll(np.asarray(gradient_b), shift, axis=-1),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_window_fno_scan_unroll_preserves_rollout(self) -> None:
        nx = 16
        memory_steps = 4
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_window_fno_params(
            jax.random.PRNGKey(105),
            width=6,
            spectral_modes=5,
            memory_steps=memory_steps,
            depth=2,
            input_channels=5,
        )
        params["output_local"] = jnp.ones((1, 6), dtype=jnp.float32) / 12.0
        state = 1e-3 * jax.random.normal(
            jax.random.PRNGKey(106), (1, 3, nx), dtype=jnp.float32
        )
        history = jnp.repeat(state[:, None], memory_steps, axis=1)
        common = dict(
            horizon=8,
            dt=0.01,
            memory_stride=2,
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=0.1,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
            closure_history_input=True,
        )
        rollout_one = rollout_explicit_window_closure(
            params,
            state,
            history,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray([0.1]),
            k_arr,
            scan_unroll=1,
            **common,
        )
        rollout_four = rollout_explicit_window_closure(
            params,
            state,
            history,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray([0.1]),
            k_arr,
            scan_unroll=4,
            **common,
        )
        np.testing.assert_array_equal(
            np.asarray(rollout_four[0]), np.asarray(rollout_one[0])
        )
        for value_four, value_one in zip(rollout_four[1], rollout_one[1]):
            np.testing.assert_array_equal(np.asarray(value_four), np.asarray(value_one))

    def test_training_case_limits_are_nested_and_leave_validation_unchanged(self) -> None:
        anchors = {
            regime: {
                "train_cases": np.repeat(np.arange(6, dtype=np.int32), 3),
                "train_times": np.tile(np.arange(3, dtype=np.int32), 6),
                "val_cases": np.repeat(np.arange(6, 8, dtype=np.int32), 2),
                "val_times": np.tile(np.arange(2, dtype=np.int32), 2),
            }
            for regime in (
                "linear_landau",
                "nonlinear_landau_weak",
                "nonlinear_landau_strong",
            )
        }
        small, small_cases = limit_training_anchors(
            anchors,
            {"nonlinear_landau_strong": 2},
            seed=17,
        )
        large, large_cases = limit_training_anchors(
            anchors,
            {"nonlinear_landau_strong": 4},
            seed=17,
        )
        self.assertTrue(
            set(small_cases["nonlinear_landau_strong"]).issubset(
                set(large_cases["nonlinear_landau_strong"])
            )
        )
        np.testing.assert_array_equal(
            small["nonlinear_landau_strong"]["val_cases"],
            anchors["nonlinear_landau_strong"]["val_cases"],
        )
        self.assertEqual(
            np.unique(small["linear_landau"]["train_cases"]).size,
            6,
        )

    def test_parse_training_case_limits(self) -> None:
        self.assertEqual(
            parse_train_case_limits("nonlinear_landau_strong=4"),
            {"nonlinear_landau_strong": 4},
        )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            parse_train_case_limits("unknown=4")

    def test_temporal_convolution_is_strictly_causal(self) -> None:
        values = jnp.zeros((1, 8, 1, 1), dtype=jnp.float32)
        values = values.at[:, 6].set(1.0)
        kernel = jnp.ones((3, 1, 1), dtype=jnp.float32)
        output = _causal_temporal_convolution(values, kernel, dilation=2)
        np.testing.assert_array_equal(np.asarray(output[:, :6]), 0.0)
        self.assertGreater(float(output[0, 6, 0, 0]), 0.0)

    def test_spacetime_operator_is_history_sensitive_equivariant_and_finite(self) -> None:
        nx = 32
        memory_steps = 6
        width = 8
        shift = 7
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_causal_spacetime_operator_params(
            jax.random.PRNGKey(101),
            width=width,
            spectral_modes=8,
            memory_steps=memory_steps,
            depth=3,
            temporal_kernel_size=3,
        )
        params["output_local"] = jnp.ones((1, width), dtype=jnp.float32)
        x = jnp.linspace(0.0, 4.0 * jnp.pi, nx, endpoint=False)
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32).at[:, 0].set(
            0.03 * jnp.cos(x) + 0.01 * jnp.cos(2.0 * x + 0.4)
        )
        history_a = jnp.repeat(state[:, None], memory_steps, axis=1)
        history_b = history_a.at[:, 0, 1].set(0.02 * jnp.sin(3.0 * x))
        closure_history = jnp.zeros((1, memory_steps, nx), dtype=jnp.float32)
        previous = jnp.zeros((1, nx), dtype=jnp.float32)
        kwargs = dict(
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
            previous_heat_flux_gradient=previous,
            heat_flux_gradient_history=closure_history,
        )
        gradient_a = explicit_window_closure_step(
            params, state, history_a, jnp.asarray([0.1]), k_arr, **kwargs
        )
        gradient_b = explicit_window_closure_step(
            params, state, history_b, jnp.asarray([0.1]), k_arr, **kwargs
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient_b))))
        self.assertGreater(float(jnp.linalg.norm(gradient_b - gradient_a)), 1e-7)
        gradient_shifted = explicit_window_closure_step(
            params,
            jnp.roll(state, shift, axis=-1),
            jnp.roll(history_b, shift, axis=-1),
            jnp.asarray([0.1]),
            k_arr,
            previous_heat_flux_gradient=jnp.roll(previous, shift, axis=-1),
            heat_flux_gradient_history=jnp.roll(
                closure_history, shift, axis=-1
            ),
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
        )
        np.testing.assert_allclose(
            np.asarray(gradient_shifted),
            np.roll(np.asarray(gradient_b), shift, axis=-1),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_dynamic_window_scaling_is_amplitude_equivariant(self) -> None:
        nx = 32
        memory_steps = 5
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_window_fno_params(
            jax.random.PRNGKey(109),
            width=8,
            spectral_modes=9,
            memory_steps=memory_steps,
            depth=2,
            input_channels=5,
        )
        params["output_local"] = jnp.ones((1, 8), dtype=jnp.float32) / 8.0
        x = jnp.linspace(0.0, 4.0 * jnp.pi, nx, endpoint=False)
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32).at[:, 0].set(
            0.08 * jnp.cos(x) + 0.03 * jnp.cos(2.0 * x + 0.4)
        )
        history = jnp.repeat(state[:, None], memory_steps, axis=1)
        previous = 0.01 * jnp.sin(x)[None]
        closure_history = jnp.repeat(previous[:, None], memory_steps, axis=1)
        kwargs = dict(
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
            previous_heat_flux_gradient=previous,
            heat_flux_gradient_history=closure_history,
            input_scaling=DYNAMIC_INPUT_SCALING,
        )
        base = explicit_window_closure_step(
            params, state, history, jnp.asarray([0.08]), k_arr, **kwargs
        )
        factor = 3.0
        scaled = explicit_window_closure_step(
            params,
            factor * state,
            factor * history,
            jnp.asarray([factor * 0.08]),
            k_arr,
            **{
                **kwargs,
                "previous_heat_flux_gradient": factor * previous,
                "heat_flux_gradient_history": factor * closure_history,
            },
        )
        np.testing.assert_allclose(
            np.asarray(scaled), factor * np.asarray(base), rtol=3e-5, atol=3e-6
        )

    def test_uniform_heating_option_only_restores_zero_mode(self) -> None:
        nx = 32
        memory_steps = 5
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_window_fno_params(
            jax.random.PRNGKey(110),
            width=8,
            spectral_modes=9,
            memory_steps=memory_steps,
            depth=2,
            input_channels=5,
        )
        params["output_local"] = jnp.ones((1, 8), dtype=jnp.float32) / 8.0
        x = jnp.linspace(0.0, 4.0 * jnp.pi, nx, endpoint=False)
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32).at[:, 0].set(
            0.15 * jnp.cos(x) + 0.04 * jnp.cos(2.0 * x + 0.2)
        )
        history = jnp.repeat(state[:, None], memory_steps, axis=1)
        kwargs = dict(
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
            previous_heat_flux_gradient=jnp.zeros((1, nx)),
            heat_flux_gradient_history=jnp.zeros((1, memory_steps, nx)),
        )
        centered = explicit_window_closure_step(
            params, state, history, jnp.asarray([0.15]), k_arr, **kwargs
        )
        effective = explicit_window_closure_step(
            params,
            state,
            history,
            jnp.asarray([0.15]),
            k_arr,
            allow_uniform_heating=True,
            **kwargs,
        )
        self.assertAlmostEqual(float(jnp.mean(centered)), 0.0, places=6)
        difference = np.asarray(effective - centered)
        np.testing.assert_allclose(
            difference,
            np.broadcast_to(np.mean(difference, axis=-1, keepdims=True), difference.shape),
            atol=2e-6,
        )

    def test_spacetime_operator_preserves_equilibrium_and_chunk_boundaries(self) -> None:
        nx = 16
        memory_steps = 4
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_causal_spacetime_operator_params(
            jax.random.PRNGKey(102),
            width=6,
            spectral_modes=5,
            memory_steps=memory_steps,
            depth=2,
            temporal_kernel_size=3,
        )
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32)
        history = jnp.zeros((1, memory_steps, 3, nx), dtype=jnp.float32)
        common = dict(
            dt=0.01,
            memory_stride=2,
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
            closure_history_input=True,
        )
        full_states, full_memory = rollout_explicit_window_closure(
            params,
            state,
            history,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray([0.1]),
            k_arr,
            horizon=4,
            **common,
        )
        np.testing.assert_array_equal(np.asarray(full_states), np.zeros_like(full_states))
        first_states, first_memory = rollout_explicit_window_closure(
            params,
            state,
            history,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray([0.1]),
            k_arr,
            horizon=2,
            **common,
        )
        window, closure_history, encoded, counter, gradient = first_memory
        second_states, second_memory = rollout_explicit_window_closure(
            params,
            first_states[:, -1],
            window,
            counter,
            jnp.asarray([0.1]),
            k_arr,
            horizon=2,
            previous_heat_flux_gradient=gradient,
            heat_flux_gradient_history=closure_history,
            encoded_history=encoded,
            **common,
        )
        np.testing.assert_array_equal(
            np.asarray(jnp.concatenate((first_states, second_states), axis=1)),
            np.asarray(full_states),
        )
        for chunked, complete in zip(second_memory, full_memory):
            np.testing.assert_allclose(np.asarray(chunked), np.asarray(complete))

    def test_spacetime_operator_training_gradient_is_finite_and_nonzero(self) -> None:
        nx = 16
        memory_steps = 4
        horizon = 2
        width = 6
        k_arr = 2.0 * np.pi * np.fft.rfftfreq(nx, d=4.0 * np.pi / nx)
        params = init_causal_spacetime_operator_params(
            jax.random.PRNGKey(103),
            width=width,
            spectral_modes=5,
            memory_steps=memory_steps,
            depth=2,
            temporal_kernel_size=3,
        )
        x = jnp.linspace(0.0, 4.0 * jnp.pi, nx, endpoint=False)
        initial = jnp.zeros((3, 3, nx), dtype=jnp.float32).at[:, 0].set(
            jnp.stack(
                (
                    0.01 * jnp.cos(x),
                    0.08 * jnp.cos(x + 0.2),
                    0.4 * jnp.cos(x) + 0.1 * jnp.cos(2.0 * x + 0.5),
                )
            )
        )
        memory = jnp.repeat(initial[:, None], memory_steps, axis=1)
        targets = jnp.repeat(initial[:, None], horizon, axis=1).at[:, :, 1].add(
            0.01 * jnp.sin(x)
        )
        batch = {
            "initial": initial,
            "memory": memory,
            "targets": targets,
            "amplitude": jnp.asarray([0.01, 0.08, 0.4], dtype=jnp.float32),
            "regime_index": jnp.asarray([0, 1, 2], dtype=jnp.int32),
            "start_index": jnp.zeros((3,), dtype=jnp.int32),
        }
        loss_fn = make_loss_function(
            k_arr=k_arr,
            width=width,
            horizon=horizon,
            dt=0.01,
            input_scale=np.ones((4,), dtype=np.float32),
            regime_scales=np.ones((3, 4), dtype=np.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
            poisson_sign=1.0,
            normalized_heat_flux_bound=128.0,
            density_floor=1e-4,
            pressure_floor=1e-4,
            closure_history_input=True,
            memory_backend="causal_spacetime_operator",
            memory_stride=1,
        )
        (loss_value, _), gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, batch)
        leaves = jax.tree_util.tree_leaves(gradients)
        self.assertTrue(bool(jnp.isfinite(loss_value)))
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves))
        self.assertGreater(
            float(jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))),
            0.0,
        )

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
        states, (
            history_new,
            closure_history_new,
            encoded,
            counter,
            gradient,
        ) = rollout_explicit_window_closure(
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
        np.testing.assert_array_equal(
            np.asarray(closure_history_new), np.zeros_like(closure_history_new)
        )
        np.testing.assert_array_equal(np.asarray(encoded), np.zeros_like(encoded))
        np.testing.assert_array_equal(np.asarray(gradient), np.zeros_like(gradient))
        self.assertEqual(int(counter), 1)

    def test_explicit_window_uses_predicted_closure_history(self) -> None:
        nx = 32
        memory_steps = 5
        width = 8
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_explicit_window_params(
            jax.random.PRNGKey(43),
            width=width,
            spectral_modes=8,
            memory_steps=memory_steps,
            input_channels=5,
        )
        params["output_local"] = jnp.ones((1, width), dtype=jnp.float32)
        x = jnp.linspace(0.0, 4.0 * jnp.pi, nx, endpoint=False)
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32)
        history = jnp.zeros((1, memory_steps, 3, nx), dtype=jnp.float32)
        closure_history_a = jnp.zeros((1, memory_steps, nx), dtype=jnp.float32)
        closure_history_b = closure_history_a.at[:, 1].set(0.1 * jnp.cos(x))
        previous = jnp.zeros((1, nx), dtype=jnp.float32)
        kwargs = dict(
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
            previous_heat_flux_gradient=previous,
        )
        gradient_a = explicit_window_closure_step(
            params,
            state,
            history,
            jnp.asarray([0.1]),
            k_arr,
            heat_flux_gradient_history=closure_history_a,
            **kwargs,
        )
        gradient_b = explicit_window_closure_step(
            params,
            state,
            history,
            jnp.asarray([0.1]),
            k_arr,
            heat_flux_gradient_history=closure_history_b,
            **kwargs,
        )
        self.assertGreater(float(jnp.linalg.norm(gradient_b - gradient_a)), 1e-7)

    def test_explicit_window_rollout_populates_closure_history_causally(self) -> None:
        nx = 16
        memory_steps = 4
        width = 6
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_explicit_window_params(
            jax.random.PRNGKey(44),
            width=width,
            spectral_modes=5,
            memory_steps=memory_steps,
            input_channels=5,
        )
        params["output_local"] = jnp.ones((1, width), dtype=jnp.float32)
        x = jnp.linspace(0.0, 4.0 * jnp.pi, nx, endpoint=False)
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32).at[:, 0].set(
            0.02 * jnp.cos(x)
        )
        history = jnp.repeat(state[:, None], memory_steps, axis=1)
        _, (_, closure_history, _, counter, gradient) = rollout_explicit_window_closure(
            params,
            state,
            history,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray([0.1]),
            k_arr,
            horizon=2,
            dt=0.01,
            memory_stride=1,
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
            closure_history_input=True,
        )
        self.assertGreater(float(jnp.linalg.norm(closure_history[:, -2:])), 1e-7)
        np.testing.assert_array_equal(
            np.asarray(closure_history[:, :-2]),
            np.zeros_like(np.asarray(closure_history[:, :-2])),
        )
        self.assertGreater(float(jnp.linalg.norm(gradient)), 1e-7)
        self.assertEqual(int(counter), 0)

    def test_explicit_closure_history_survives_rollout_chunk_boundary(self) -> None:
        nx = 16
        memory_steps = 4
        width = 6
        k_arr = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=4.0 * jnp.pi / nx)
        params = init_explicit_window_params(
            jax.random.PRNGKey(45),
            width=width,
            spectral_modes=5,
            memory_steps=memory_steps,
            input_channels=5,
        )
        params["output_local"] = jnp.ones((1, width), dtype=jnp.float32)
        x = jnp.linspace(0.0, 4.0 * jnp.pi, nx, endpoint=False)
        state = jnp.zeros((1, 3, nx), dtype=jnp.float32).at[:, 0].set(
            0.02 * jnp.cos(x)
        )
        history = jnp.repeat(state[:, None], memory_steps, axis=1)
        common = dict(
            dt=0.01,
            memory_stride=2,
            input_scale=jnp.ones((4,), dtype=jnp.float32),
            heat_flux_gradient_scale=1.0,
            amplitude_center=-2.0,
            amplitude_scale=1.0,
            closure_history_input=True,
        )
        full_states, full_memory = rollout_explicit_window_closure(
            params,
            state,
            history,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray([0.1]),
            k_arr,
            horizon=4,
            **common,
        )
        first_states, first_memory = rollout_explicit_window_closure(
            params,
            state,
            history,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray([0.1]),
            k_arr,
            horizon=2,
            **common,
        )
        window, closure_history, encoded, counter, gradient = first_memory
        second_states, second_memory = rollout_explicit_window_closure(
            params,
            first_states[:, -1],
            window,
            counter,
            jnp.asarray([0.1]),
            k_arr,
            horizon=2,
            previous_heat_flux_gradient=gradient,
            heat_flux_gradient_history=closure_history,
            encoded_history=encoded,
            **common,
        )
        np.testing.assert_allclose(
            np.asarray(jnp.concatenate((first_states, second_states), axis=1)),
            np.asarray(full_states),
            rtol=2e-6,
            atol=2e-6,
        )
        for chunked, complete in zip(second_memory, full_memory):
            np.testing.assert_allclose(
                np.asarray(chunked), np.asarray(complete), rtol=2e-6, atol=2e-6
            )

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

        primitive_targets = {
            regime: (
                np.broadcast_to(
                    np.arange(6, dtype=np.float32)[:, None, None],
                    (6, 4, nx),
                ).copy(),
            )
            for regime in regimes
        }
        trajectory_norms = {
            case["case_id"]: np.full((4,), 10.0 + index, dtype=np.float64)
            for index, case in enumerate(cases)
        }
        cached_batch = sample_complete_trajectory_batch(
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
            primitive_target_histories=primitive_targets,
            trajectory_target_norm_by_id=trajectory_norms,
        )
        self.assertNotIn("targets", cached_batch)
        self.assertEqual(cached_batch["target_fields"].shape, (3, 5, 4, nx))
        np.testing.assert_allclose(
            cached_batch["target_fields"][0, :, 0, 0], np.arange(1, 6)
        )
        np.testing.assert_allclose(
            cached_batch["trajectory_target_norm"],
            np.stack([trajectory_norms[case["case_id"]] for case in cases]),
        )

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
