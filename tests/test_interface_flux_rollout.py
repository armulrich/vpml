import math
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import model.train.interface_flux_rollout as trainer
from vpml.core import (
    FourierHermiteIMEX,
    LearnedInterfaceClosure,
    learned_interface_q_hat,
    load_learned_interface_closure_npz,
    save_learned_interface_closure_npz,
)


jax.config.update("jax_enable_x64", True)


def _zero_params(input_dim: int, hidden_width: int = 8):
    params = trainer.init_interface_closure_params(
        jax.random.PRNGKey(0),
        input_dim=input_dim,
        hidden_width=hidden_width,
        res_blocks=0,
    )
    return jax.tree_util.tree_map(jnp.zeros_like, params)


def _closure(*, params=None, centered: bool = True) -> LearnedInterfaceClosure:
    if params is None:
        params = _zero_params(6)
    return LearnedInterfaceClosure(
        params=params,
        Nm=1,
        k_scale=1.0,
        nv_scale=4.0,
        input_mean=jnp.zeros((6,), dtype=jnp.float64),
        input_std=jnp.ones((6,), dtype=jnp.float64),
        target_mean=jnp.zeros((2,), dtype=jnp.float64),
        target_std=jnp.ones((2,), dtype=jnp.float64),
        hidden_width=8,
        res_blocks=0,
        equilibrium_centered=centered,
        complex_normalization_mode="phase_isotropic",
        translation_augmented=True,
        Nv_targets=(4,),
        train_regimes=trainer.CANONICAL_REGIMES,
        teacher_backend="grid_cubic_spline",
        teacher_Lx=4.0 * math.pi,
        teacher_Nx=4,
        teacher_Nv=64,
        teacher_vmin=-8.0,
        teacher_vmax=8.0,
        teacher_dt=0.05,
        teacher_proj_Nv=5,
        projection_quadrature_Nv=64,
        training_mode=trainer.INTERFACE_FLUX_ROLLOUT_TRAINING_MODE,
        train_objective=trainer.INTERFACE_FLUX_ROLLOUT_OBJECTIVE,
        rollout_horizon=1,
        loss_backend=trainer.INTERFACE_FLUX_ROLLOUT_LOSS_BACKEND,
    )


class InterfaceFluxRolloutTests(unittest.TestCase):
    def test_canonical_parser_does_not_expose_trainer_switches(self) -> None:
        parser = trainer.build_arg_parser()
        option_names = {flag for action in parser._actions for flag in action.option_strings}
        self.assertNotIn("--training-mode", option_names)
        self.assertNotIn("--train-objective", option_names)
        self.assertNotIn("--tail-history-lift", option_names)
        self.assertNotIn("--exact-q-grouped-relative-loss", option_names)

    def test_cutoff_cycle_is_per_optimizer_step(self) -> None:
        observed = [
            trainer.interface_flux_cutoff_for_step(
                trainer.CANONICAL_NV_TARGETS,
                step,
            )
            for step in range(8)
        ]
        self.assertEqual(observed, [6, 7, 12, 20, 36, 64, 6, 7])

    def test_zero_validation_fraction_uses_every_sampled_time_for_training(self) -> None:
        max_projection_order = 4
        coeff_key = trainer.interface_flux_rollout_coeff_key(max_projection_order)
        coeff_history = np.zeros((1, 5, max_projection_order + 1, 3), dtype=np.complex128)
        coeff_history[:, :, 3:, 1:] = 0.25 + 0.1j
        dataset, _ = trainer.build_interface_flux_rollout_qpair_dataset(
            {trainer.REGIME_LINEAR: {coeff_key: coeff_history}},
            max_projection_order=max_projection_order,
            Nv_targets=(4,),
            Nm=2,
            k_arr=np.array([0.0, 0.5, 1.0], dtype=np.float64),
            val_fraction=0.0,
            linear_history_stride=2,
            nonlinear_history_stride=2,
            rollout_horizon=1,
            n_low=2,
            context_mode="none",
        )
        linear = dataset[trainer.REGIME_LINEAR]
        np.testing.assert_array_equal(
            linear["train_anchor_time_indices"],
            np.array([0, 2, 4], dtype=np.int32),
        )
        self.assertEqual(linear["val_anchor_time_indices"].size, 0)
        self.assertEqual(linear["val_targets"].shape[0], 0)

    def test_h1_target_matches_direct_interface_flux_prediction(self) -> None:
        params = _zero_params(6)
        params["W_lin"] = params["W_lin"].at[0, 0].set(2.0)
        learned = _closure(params=params, centered=False)
        integ = FourierHermiteIMEX(
            Nx=4,
            Nv=4,
            Lx=4.0 * math.pi,
            dt=0.05,
            closure=None,
        )
        anchors = jnp.zeros((1, 3, 4, 3), dtype=jnp.complex128)
        anchors = anchors.at[0, 0, 3, 1].set(0.25 + 0.0j)
        direct = learned_interface_q_hat(
            anchors[0, 0],
            integ.k_arr,
            4,
            learned,
            a_hat_prev=anchors[0, 1],
        )
        loss = trainer.interface_flux_rollout_loss_for_anchor_batch(
            anchors,
            direct[None, None, :],
            jnp.zeros((1,), dtype=jnp.int32),
            learned=learned,
            forward_integ=integ,
            rollout_horizon=1,
            explicit_n_hat_fn=lambda state, *, integ: jnp.zeros_like(state),
        )
        self.assertEqual(float(loss), 0.0)

    def test_all_positive_k_and_regimes_are_equally_weighted(self) -> None:
        stats = {
            "input_mean": np.zeros((6,), dtype=np.float64),
            "input_std": np.ones((6,), dtype=np.float64),
            "target_mean": np.zeros((2,), dtype=np.float64),
            "target_std": np.ones((2,), dtype=np.float64),
        }
        loss_fn, _ = trainer.make_interface_flux_rollout_batch_loss(
            regime_weights={regime: 1.0 for regime in trainer.CANONICAL_REGIMES},
            Nm=1,
            k_scale=1.0,
            nv_scale=4.0,
            stats=stats,
            hidden_width=8,
            res_blocks=0,
            Nv_targets=(4,),
            train_regimes=trainer.CANONICAL_REGIMES,
            teacher_backend="grid_cubic_spline",
            teacher_Lx=4.0 * math.pi,
            teacher_Nx=4,
            teacher_Nv=64,
            teacher_vmin=-8.0,
            teacher_vmax=8.0,
            teacher_dt=0.05,
            teacher_proj_Nv=5,
            projection_quadrature_Nv=64,
            n_low=2,
            context_mode="none",
            rollout_horizon=1,
            poisson_sign=1.0,
            rollout_dealias_23=False,
            regime_q_loss_stds={
                regime: 1.0 for regime in trainer.CANONICAL_REGIMES
            },
            equilibrium_centered=True,
            complex_normalization_mode="phase_isotropic",
            translation_augmented=True,
        )
        batches = {}
        for amplitude, regime in zip((1.0, 2.0, 3.0), trainer.CANONICAL_REGIMES):
            reference = np.full((1, 1, 3), amplitude + 0.0j, dtype=np.complex128)
            reference[..., 0] = 1e6
            batches[regime] = {
                "anchor_stencils": jnp.zeros(
                    (1, 3, 4, 3), dtype=jnp.complex128
                ),
                "ref_q_windows": jnp.asarray(reference),
                "k_indices": jnp.zeros((1,), dtype=jnp.int32),
            }
        loss, aux = loss_fn.target_loss_fns[4](_zero_params(6), batches)
        expected = (1.0**2 + 2.0**2 + 3.0**2) / 6.0
        self.assertAlmostEqual(float(loss), expected)
        self.assertAlmostEqual(float(aux["q"]), expected)

    def test_equilibrium_centering_cancels_arbitrary_biases(self) -> None:
        params = _zero_params(6)
        params["b_lin"] = jnp.array([3.0, -4.0], dtype=jnp.float64)
        params["b_out"] = jnp.array([-1.5, 2.5], dtype=jnp.float64)
        learned = _closure(params=params, centered=True)
        q_hat = learned_interface_q_hat(
            jnp.zeros((4, 3), dtype=jnp.complex128),
            jnp.array([0.0, 0.5, 1.0], dtype=jnp.float64),
            4,
            learned,
        )
        np.testing.assert_array_equal(np.asarray(q_hat), np.zeros((3,)))

    def test_translation_uses_one_phase_for_stencil_and_target(self) -> None:
        rng = np.random.default_rng(7)
        stencils = rng.normal(size=(2, 3, 4, 5)) + 1j * rng.normal(
            size=(2, 3, 4, 5)
        )
        targets = rng.normal(size=(2, 6, 5)) + 1j * rng.normal(
            size=(2, 6, 5)
        )
        k_arr = np.arange(5, dtype=np.float64) * 0.5
        shifts = np.array([0.3, 1.1], dtype=np.float64)
        translated_stencils, translated_targets = (
            trainer.translate_interface_flux_rollout_anchor_batch(
                stencils,
                targets,
                k_arr=k_arr,
                shifts=shifts,
            )
        )
        phases = np.exp(-1j * shifts[:, None] * k_arr[None, :])
        np.testing.assert_allclose(
            translated_stencils,
            stencils * phases[:, None, None, :],
        )
        np.testing.assert_allclose(
            translated_targets,
            targets * phases[:, None, :],
        )

    def test_checkpoint_round_trip_and_legacy_adapter(self) -> None:
        learned = _closure()
        state = jnp.zeros((4, 3), dtype=jnp.complex128).at[3, 1].set(
            0.25 + 0.1j
        )
        k_arr = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float64)
        expected = np.asarray(learned_interface_q_hat(state, k_arr, 4, learned))
        with tempfile.TemporaryDirectory() as tmp:
            canonical_path = Path(tmp) / "canonical.npz"
            legacy_path = Path(tmp) / "legacy.npz"
            save_learned_interface_closure_npz(canonical_path, learned)
            loaded = load_learned_interface_closure_npz(canonical_path)
            self.assertEqual(
                loaded.loss_backend,
                trainer.INTERFACE_FLUX_ROLLOUT_LOSS_BACKEND,
            )
            self.assertEqual(loaded.projection_quadrature_Nv, 64)
            np.testing.assert_allclose(
                np.asarray(learned_interface_q_hat(state, k_arr, 4, loaded)),
                expected,
            )

            with np.load(canonical_path) as payload:
                legacy = {key: np.asarray(payload[key]) for key in payload.files}
            legacy.pop("metadata_schema_version")
            legacy["training_mode"] = np.array(["exact_q_rollout"])
            legacy["train_objective"] = np.array(["q_rollout"])
            legacy["loss_backend"] = np.array(
                ["exact_fourier_hermite_q_rollout"]
            )
            np.savez(legacy_path, **legacy)
            adapted = load_learned_interface_closure_npz(legacy_path)
            self.assertEqual(
                adapted.training_mode,
                trainer.INTERFACE_FLUX_ROLLOUT_TRAINING_MODE,
            )
            np.testing.assert_allclose(
                np.asarray(learned_interface_q_hat(state, k_arr, 4, adapted)),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
