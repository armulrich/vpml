import unittest
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from model.train.fresh_coupled_latent import (
    backtracked_update, build_parser, curriculum_horizon, event_agnostic_loss, fresh_parameters,
    local_window_mean,
    replay_training_rng,
    evaluate_guard_first,
    descent_safe_adam_step,
)
from model.train.coupled_low_moment_latent import _atomic_savez, _training_state_payload, _load_training_state
from model.train.kinetic_latent_dynamics_probe import _adam_init, _adam_step
from vpml.kinetic_latent import coupled_low_moment_latent_step


class FreshCoupledLatentTest(unittest.TestCase):
    def test_stale_adam_direction_resets_only_first_moments(self):
        params = {"weight": jnp.asarray([0.], dtype=jnp.float32)}
        gradient = {"weight": jnp.asarray([1.], dtype=jnp.float32)}
        state = {"step": jnp.asarray(10, dtype=jnp.int32),
                 "m": {"weight": jnp.asarray([-1.], dtype=jnp.float32)},
                 "v": {"weight": jnp.asarray([1.], dtype=jnp.float32)}}
        old, old_state, _ = _adam_step(params, gradient, state, .001, 1.)
        new, new_state, _, audit = descent_safe_adam_step(params, gradient, state, .001)
        self.assertGreater(float(old["weight"][0]), 0)
        self.assertLess(float(new["weight"][0]), 0)
        self.assertTrue(audit["reset_first_moment"])
        np.testing.assert_array_equal(new_state["v"]["weight"], old_state["v"]["weight"])
        np.testing.assert_array_equal(new_state["step"], old_state["step"])
        state = _adam_init(params)
        expected = _adam_step(params, gradient, state, .001, 1.)
        actual = descent_safe_adam_step(params, gradient, state, .001)
        self.assertFalse(actual[3]["reset_first_moment"])
        for a,b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual[:3])):
            np.testing.assert_array_equal(a,b)

    def test_guard_short_circuit_preserves_backtracking_decision(self):
        params = {"weight": jnp.asarray([0.0])}
        proposal = {"weight": jnp.asarray([10.0])}
        calls = []
        def current(p):
            calls.append(float(p["weight"][0]))
            return (1-float(p["weight"][0]))**2
        def guard(p):
            return 2.0 if float(p["weight"][0]) > .6 else 1.0
        options = dict(initial_loss=1., initial_guard=1., step_limit=1., backtracks=4)
        old, old_record = backtracked_update(params, proposal, lambda p: (current(p), guard(p)), **options)
        old_count = len(calls)
        calls.clear()
        new, new_record = backtracked_update(params, proposal,
            lambda p: evaluate_guard_first(p, current, guard, 1.), **options)
        np.testing.assert_array_equal(old["weight"], new["weight"])
        self.assertEqual(old_record["scale"], new_record["scale"])
        self.assertLess(len(calls), old_count)
        self.assertIsNone(new_record["attempts"][0]["loss"])
        self.assertFalse(new_record["attempts"][0]["passed"])

    def test_resume_preserves_adam_next_update_exactly(self):
        params = {"weight": jnp.asarray([.1, -.2], dtype=jnp.float32)}
        optimizer = _adam_init(params)
        gradient = {"weight": jnp.asarray([.3, -.1], dtype=jnp.float32)}
        for _ in range(3):
            params, optimizer, _ = _adam_step(params, gradient, optimizer, 1e-4, 1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            _atomic_savez(path, {**_training_state_payload(params, {"physical": optimizer}),
                                 "completed_epoch": np.asarray(8)})
            restored, states, epoch = _load_training_state(path, params, {"physical": _adam_init(params)})
        expected = _adam_step(params, gradient, optimizer, 1e-4, 1.)
        actual = _adam_step(restored, gradient, states["physical"], 1e-4, 1.)
        self.assertEqual(epoch, 8)
        for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_array_equal(a, b)

    def test_sampling_cursor_replays_mid_epoch_and_full_horizon(self):
        rng = np.random.default_rng(17)
        for epoch in range(1, 10):
            orders = [rng.permutation(n) for n in (5, 6, 7)]
            horizon = curriculum_horizon(epoch, 1200)
            for step in range(4):
                if step % 2 and horizon < 1200:
                    for _ in range(3):
                        rng.integers(1200-horizon+1)
                replay, replay_orders = replay_training_rng(17, (5, 6, 7), 4, 1200, epoch, step+1)
                self.assertEqual(replay.bit_generator.state, rng.bit_generator.state)
                for a, b in zip(orders, replay_orders):
                    np.testing.assert_array_equal(a, b)

    def test_late_phase_error_is_detected_even_with_identical_energy(self):
        target = np.ones((1, 301, 4, 8), dtype=np.float32)
        target[:, 50:, 3] *= .001
        prediction = target.copy()
        prediction[:, 50:, 3] *= -1
        latent = jnp.zeros((1, 301, 1, 8), dtype=jnp.float32)
        value, components = event_agnostic_loss(jnp.asarray(prediction), jnp.asarray(target),
            latent, latent, cadence=.1, envelope_weight=.1, change_weight=.1,
            latent_weight=.1, floor_ratio=1e-8)
        self.assertGreater(float(value), .5)
        self.assertAlmostEqual(float(components[2]), 0, places=6)

    def test_late_window_preserves_small_energy_and_gradient_after_large_peak(self):
        values = jnp.concatenate((jnp.ones(100, dtype=jnp.float32),
                                  jnp.full(200, 1e-8, dtype=jnp.float32)))[None]
        expected = np.convolve(np.asarray(values[0], dtype=np.float64), np.ones(50)/50, 'valid')
        actual = local_window_mean(values, 50)
        np.testing.assert_allclose(actual[0], expected, rtol=2e-6, atol=1e-14)
        gradient = jax.jit(jax.grad(lambda x: jnp.log(local_window_mean(x, 50)[0, -1])))(values)
        self.assertTrue(np.all(np.isfinite(gradient)))
        np.testing.assert_allclose(gradient[0, -50:], np.full(50, 2e6), rtol=2e-6)

    def test_feature_scaling_preserves_physics_when_neural_output_is_zero(self):
        config = dict(seed=1729, latent_rank=4, width=4, depth=1,
                      operator_modes=5, operator_rank=2, kernel_size=3,
                      operator_output_init_scale=1e-6)
        params = fresh_parameters(config)
        params["output_kernel"] = jnp.zeros_like(params["output_kernel"])
        state = .001 * jax.random.normal(jax.random.PRNGKey(51), (1, 3, 8))
        latent = .001 * jax.random.normal(jax.random.PRNGKey(52), (1, 4, 8))
        scales = jnp.asarray([.1, .01, 1e-5, 1e-8])
        def step(feature_scale):
            return coupled_low_moment_latent_step(params, state, latent,
                jnp.broadcast_to(jnp.eye(7, dtype=jnp.complex64), (5, 7, 7)),
                jnp.eye(5, 4), jnp.arange(5, dtype=jnp.float32),
                previous_latent_state=latent, resolved_center=jnp.zeros(3),
                resolved_scale=jnp.ones(3), latent_center=jnp.zeros(4), latent_scale=scales,
                latent_input_scale=feature_scale, depth=1, fine_steps=1, fine_dt=.01,
                latent_readout_mode="equilibrium_cnn", linear_baseline="projected_hermite",
                equilibrium_input_compression_scale=4.0)
        legacy, explicit, conditioned = step(None), step(scales), step(jnp.maximum(scales, .01))
        for a, b, c in zip(legacy, explicit, conditioned):
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a, c)

    def test_backtracking_rejects_unstable_guard_without_changing_parameters(self):
        params = {"weight": jnp.asarray([0.0])}
        proposal = {"weight": jnp.asarray([10.0])}
        def evaluate(candidate):
            x = float(candidate["weight"][0])
            return (1 - x) ** 2, float("inf") if x > 0.6 else 1.0
        result, audit = backtracked_update(params, proposal, evaluate, initial_loss=1,
            initial_guard=1, step_limit=1, backtracks=4)
        self.assertTrue(audit["accepted"])
        self.assertAlmostEqual(float(result["weight"][0]), 0.5)
        self.assertEqual(float(params["weight"][0]), 0)
        rejected, audit = backtracked_update(params, proposal, lambda _: (2.0, 1.0),
            initial_loss=1, initial_guard=1, step_limit=1, backtracks=2)
        self.assertFalse(audit["accepted"])
        self.assertIs(rejected, params)

    def test_epoch_duration_does_not_change_initialization_or_curriculum(self):
        prefix = [curriculum_horizon(e, 1200) for e in range(1, 11)]
        longer = [curriculum_horizon(e, 1200) for e in range(1, 201)]
        self.assertEqual(prefix, longer[:10])
        self.assertEqual(prefix[-1], 1200)
        self.assertNotIn("--init-checkpoint", build_parser().format_help())
        config = dict(seed=1729, latent_rank=4, width=4, depth=1,
                      operator_modes=5, operator_rank=2, kernel_size=3,
                      operator_output_init_scale=1e-6)
        first, second = fresh_parameters(config), fresh_parameters(config)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
            self.assertFalse(key.startswith(("expert_", "specialist_")))
        other = fresh_parameters({**config, "seed": 1730})
        self.assertFalse(np.array_equal(first["output_kernel"], other["output_kernel"]))

    def test_matching_and_perturbed_trajectories_have_finite_gradients(self):
        time = jnp.arange(101) * 0.1
        energy_shape = jnp.exp(-0.1 * time + 0.2 * jnp.sin(time))
        target = jnp.broadcast_to(energy_shape[None, :, None, None], (2, 101, 4, 8))
        latent = target[:, :, :2]
        def loss(scale):
            return event_agnostic_loss(scale * target, target, scale * latent, latent,
                cadence=0.1, envelope_weight=0.1, change_weight=0.1,
                latent_weight=0.1, floor_ratio=1e-8)[0]
        self.assertAlmostEqual(float(loss(1.0)), 0, places=6)
        self.assertAlmostEqual(float(jax.grad(loss)(1.0)), 0, places=6)
        value, gradient = jax.jit(jax.value_and_grad(loss))(1.1)
        self.assertTrue(np.isfinite(float(gradient)))
        self.assertGreater(float(value), 0)
        self.assertGreater(float(gradient), 0)


if __name__ == "__main__":
    unittest.main()
