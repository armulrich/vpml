"""Focused exposure, fixed-normalization, and checkpoint contract tests."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model.train import low_moment_closure as trainer
import jax
import jax.numpy as jnp
import numpy as np


LINEAR = ("linear_landau",)


def reference_fixture(steps=9):
    grouped, cases = {}, []
    for index, regime in enumerate(trainer.REGIMES):
        ids = [f"{regime}_{row}" for row in range(3)]
        histories = []
        for row in range(3):
            history = np.zeros((steps, 4, 5), dtype=np.complex64)
            history[:, 0, 0] = 8
            history[:, 0, 1] = 0.01 * (index + 1) * (row + 1)
            histories.append(history)
            cases.append({"case_id": ids[row], "epsilon": 0.01 * (index + 1)})
        grouped[regime] = {
            "case_ids": np.asarray(ids),
            "case_splits": np.asarray(["train", "train", "heldout"]),
            "coefficients": tuple(histories),
        }
    return grouped, {"sha256": "test-manifest", "cases": cases}


def fixed_stats(manifest):
    return {
        "input_scale": np.arange(1, 5, dtype=float),
        "regime_scales": np.arange(1, 13, dtype=float).reshape(3, 4),
        "heat_flux_gradient_scale": np.array([2.0]),
        "heat_flux_gradient_regime_scales": np.array([1., 2., 3.]),
        "heat_flux_gradient_max_abs": np.array([4.0]),
        "amplitude_center": np.array([-3.0]),
        "amplitude_scale": np.array([0.75]),
        "heat_flux_gradient_case_ids": np.array([c["case_id"] for c in manifest["cases"]]),
        "heat_flux_gradient_case_scales": np.ones(len(manifest["cases"])),
    }


def checkpoint_metadata(manifest):
    return {
        "schema_version": trainer.CHECKPOINT_SCHEMA,
        "manifest_sha256": manifest["sha256"],
        "source_Nx": 8, "rollout_Nx": 8,
        "regimes": list(trainer.REGIMES),
        "width": 2, "memory_steps": 2, "memory_stride": 1,
    }


class RegimeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.grouped, self.manifest = reference_fixture()
        self.batch_options = dict(
            horizon=2, memory_steps=2, memory_stride=1, source_nx=8,
            rollout_nx=8, domain_length=4 * np.pi, translation_augmentation=False,
        )

    def anchors(self, **kwargs):
        return trainer._build_anchor_index(
            self.grouped, "coefficients", horizon=3, history_stride=2, **kwargs
        )

    def test_default_and_selected_exposure_and_canonical_indices(self):
        parser = trainer.build_arg_parser()
        args = parser.parse_args(["--reference-cache", "/tmp/cache", "--outdir", "/tmp/run"])
        self.assertEqual(args.training_regimes, "all3")
        self.assertIsNone(args.normalization_checkpoint)
        self.assertEqual(trainer.parse_training_regimes("all3"), trainer.REGIMES)
        self.assertEqual(trainer.parse_training_regimes("linear_landau"), LINEAR)
        with self.assertRaises(ValueError):
            trainer.parse_training_regimes("all")
        for selection in (None, LINEAR, (trainer.REGIMES[2],)):
            kwargs = {} if selection is None else {"regimes": selection}
            anchors = self.anchors(**kwargs)
            selected = trainer.REGIMES if selection is None else selection
            self.assertEqual(tuple(anchors), selected)
            # A selected-only grouped mapping catches accidental access to other regimes.
            grouped = {r: self.grouped[r] for r in selected}
            batch = trainer.sample_batch(
                np.random.default_rng(7), grouped, anchors, self.manifest, "coefficients",
                split="train", batch_size_per_regime=2, **self.batch_options, **kwargs,
            )
            self.assertEqual(batch["initial"].shape, (2 * len(selected), 3, 8))
            np.testing.assert_array_equal(
                batch["regime_index"], np.repeat([trainer.REGIMES.index(r) for r in selected], 2)
            )
        self.assertEqual(len(trainer.REGIMES), 3)

    def test_full_sweep_8416_anchors_gives_526_batches_without_omissions(self):
        anchors = {LINEAR[0]: {"train_cases": np.repeat(np.arange(16), 526),
                               "train_times": np.tile(np.arange(526), 16)}}
        batches = trainer.build_balanced_full_anchor_epoch(
            np.random.default_rng(3), anchors, regimes=LINEAR,
            split="train", batch_size_per_regime=16,
        )
        self.assertEqual(len(batches), 526)
        self.assertEqual(trainer._full_anchor_steps_per_epoch(anchors, LINEAR, 16, 1), 526)
        self.assertEqual(trainer._full_anchor_steps_per_epoch(anchors, LINEAR, 16, 2), 263)
        with self.assertRaises(ValueError):
            trainer._full_anchor_steps_per_epoch(anchors, LINEAR, 16, 3)
        observed = [tuple(pair) for batch in batches for pair in zip(*batch[LINEAR[0]])]
        self.assertEqual(len(set(observed)), 8416)
        self.assertEqual(set(observed), {(c, t) for c in range(16) for t in range(526)})

    def test_random_cycle_limits_and_complete_batches_use_selected_only(self):
        anchors, cases = trainer.limit_training_anchors(
            self.anchors(regimes=LINEAR), {LINEAR[0]: 1}, seed=7, regimes=LINEAR
        )
        self.assertEqual(len(cases[LINEAR[0]]), 1)
        batches = [trainer.build_balanced_random_window_epoch(
            np.random.default_rng(3), anchors, regimes=LINEAR, split="train",
            batch_size_per_regime=1, deterministic_epoch=epoch,
        ) for epoch in (1, 2)]
        self.assertNotEqual(batches[0][0][LINEAR[0]][1][0], batches[1][0][LINEAR[0]][1][0])
        grouped = {LINEAR[0]: self.grouped[LINEAR[0]]}
        complete = trainer.build_complete_trajectory_case_batches(
            np.random.default_rng(3), grouped, regimes=LINEAR, split="train",
            batch_size_per_regime=2, shuffle=False,
        )
        batch = trainer.sample_complete_trajectory_batch(
            np.random.default_rng(3), grouped, self.manifest, "coefficients", complete[0],
            regimes=LINEAR, memory_steps=2, memory_stride=1, source_nx=8,
            rollout_nx=8, translation_augmentation=False,
        )
        self.assertEqual(batch["targets"].shape, (2, 8, 3, 8))
        np.testing.assert_array_equal(batch["regime_index"], [0, 0])

    def test_diagnostic_selection_bounds_and_burnin_targets(self):
        kwargs = dict(self.batch_options)
        kwargs.pop("translation_augmentation")
        panel = trainer.build_diagnostic_panel(
            np.random.default_rng(4), {LINEAR[0]: self.grouped[LINEAR[0]]},
            self.anchors(regimes=LINEAR), self.manifest, "coefficients",
            regimes=LINEAR, split="val", start_times=(0, 2), cases_per_regime=None,
            dt=1, autonomous_burnin_steps=2, include_heat_flux_gradient_history=True,
            **kwargs,
        )
        self.assertEqual(len(panel), 2)
        for batch, start in zip(panel, (2, 4)):
            np.testing.assert_array_equal(batch["regime_index"], [0])
            np.testing.assert_array_equal(batch["start_index"], [start])
            self.assertFalse(np.any(batch["memory"]))
            self.assertFalse(np.any(batch["heat_flux_gradient_history"]))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            trainer.build_diagnostic_panel(
                np.random.default_rng(4), self.grouped, self.anchors(regimes=LINEAR),
                self.manifest, "coefficients", regimes=LINEAR, split="val",
                start_times=(5,), cases_per_regime=None, dt=1,
                autonomous_burnin_steps=2, **kwargs,
            )

    def test_supervised_selected_loss_not_diluted_by_absent_regimes(self):
        options = dict(k_arr=np.arange(5), width=2, input_scale=np.ones(4),
                       heat_flux_gradient_scale=1., amplitude_center=0., amplitude_scale=1.,
                       poisson_sign=-1., normalized_heat_flux_bound=128., memory_backend="explicit_window")
        for regimes in (LINEAR, trainer.REGIMES):
            count = len(regimes)
            batch = dict(initial=jnp.zeros((count, 3, 8)), memory=jnp.zeros((count, 2, 3, 8)),
                         amplitude=jnp.ones(count), regime_index=jnp.arange(count),
                         heat_flux_gradient_target=jnp.ones((count, 8)) * 2,
                         heat_flux_gradient_case_scale=jnp.ones(count))
            with mock.patch.object(trainer, "explicit_window_closure_step", return_value=jnp.zeros((count, 8))):
                value, per_regime = trainer.make_supervised_heat_flux_loss(regimes=regimes, **options)({}, batch)
            self.assertEqual(float(value), 4.)
            np.testing.assert_array_equal(per_regime, np.full(count, 4.))


class NormalizationAndCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.grouped, self.manifest = reference_fixture()
        self.stats = fixed_stats(self.manifest)
        self.metadata = checkpoint_metadata(self.manifest)

    def write_checkpoint(self, path, stats=None, metadata=None):
        # An object-valued weight would fail if the stats-only reader touched it.
        np.savez(path, param_must_not_load=np.array([object()], dtype=object),
                 metadata_json=np.array([json.dumps(self.metadata if metadata is None else metadata)]),
                 **{f"stat_{k}": v for k, v in (self.stats if stats is None else stats).items()})

    def load_stats(self, path):
        return trainer._load_normalization_statistics(path, manifest=self.manifest, source_nx=8, rollout_nx=8)

    def test_stats_only_load_preserves_full_family_values_and_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "epoch000.npz"
            self.write_checkpoint(path)
            stats, digest = self.load_stats(path)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            for key in self.stats:
                np.testing.assert_array_equal(stats[key], self.stats[key])
            self.assertEqual(stats["regime_scales"].shape, (3, 4))
            self.assertEqual(stats["amplitude_center"][0], -3.)

    def test_incompatible_shapes_values_provenance_and_cases_rejected(self):
        corruptions = [
            ("regime_scales", np.ones((1, 4))), ("input_scale", np.ones(3)),
            ("amplitude_center", np.array([np.nan])), ("amplitude_scale", np.array([0.])),
            ("heat_flux_gradient_scale", np.array([-1.])),
            ("heat_flux_gradient_case_ids", np.array(["wrong"])),
            ("heat_flux_gradient_case_scales", np.array([np.inf] * 9)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "epoch000.npz"
            for key, value in corruptions:
                with self.subTest(stat=key):
                    self.write_checkpoint(path, stats={**self.stats, key: value})
                    with self.assertRaisesRegex(ValueError, "Normalization"):
                        self.load_stats(path)
            for key, value in (("manifest_sha256", "other"), ("rollout_Nx", 16),
                               ("regimes", list(LINEAR))):
                with self.subTest(metadata=key):
                    self.write_checkpoint(path, metadata={**self.metadata, key: value})
                    with self.assertRaisesRegex(ValueError, "Normalization"):
                        self.load_stats(path)
            self.write_checkpoint(path, metadata={**self.metadata, "regimes": list(LINEAR),
                                                  "normalization_regimes": list(trainer.REGIMES)})
            self.load_stats(path)

    def test_legacy_and_selected_checkpoint_metadata_and_resume_guards(self):
        self.assertEqual(trainer._checkpoint_regimes({}), trainer.REGIMES)
        selected = dict(regimes=list(LINEAR), training_regimes="linear_landau",
                        normalization_policy="fixed_all3_training_statistics",
                        normalization_regimes=list(trainer.REGIMES),
                        normalization_checkpoint="/old/epoch000.npz",
                        normalization_checkpoint_sha256="source-hash",
                        normalization_statistics_sha256="stats-hash", stats_stride=20)
        self.assertEqual(trainer._checkpoint_regimes(selected), LINEAR)
        trainer._validate_regime_resume_configuration(selected, selected)
        trainer._validate_regime_resume_configuration({}, {"regimes": list(trainer.REGIMES)})
        for key, value in (("regimes", list(trainer.REGIMES)), ("stats_stride", 40),
                           ("normalization_policy", "selected"),
                           ("normalization_checkpoint", "/other.npz"),
                           ("normalization_checkpoint_sha256", "changed"),
                           ("normalization_statistics_sha256", "changed")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                trainer._validate_regime_resume_configuration(selected, {**selected, key: value})
        with self.assertRaises(ValueError):
            trainer._validate_regime_resume_configuration({}, selected)

    def test_evaluate_checkpoint_uses_saved_selection_and_saved_stats(self):
        cache_metadata = {"configuration": dict(teacher_Nx=8, teacher_L=4 * np.pi,
                          teacher_dt=.01, teacher_poisson_sign=-1.)}
        for regimes in (LINEAR, trainer.REGIMES):
            with self.subTest(regimes=regimes), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "checkpoint.npz"
                trainer._save_checkpoint(path, {"test": np.ones(1)},
                                         {**self.metadata, "regimes": list(regimes)}, self.stats)
                with mock.patch.object(trainer, "_load_reference_cache", return_value=(
                    self.grouped, self.manifest, cache_metadata, "coefficients"
                )), mock.patch.object(trainer, "_evaluate_heldout") as evaluate, mock.patch.object(
                    trainer, "load_or_compute_training_statistics", side_effect=AssertionError("no recompute")
                ):
                    trainer.main(["--reference-cache", tmp, "--outdir", str(Path(tmp) / "eval"),
                                  "--rollout-Nx", "8", "--evaluate-checkpoint", str(path)])
                self.assertEqual(evaluate.call_args.kwargs["regimes"], regimes)
                np.testing.assert_array_equal(evaluate.call_args.kwargs["input_scale"], self.stats["input_scale"])
                self.assertEqual(evaluate.call_args.kwargs["amplitude_center"], -3.)

    def test_training_wires_fixed_checkpoint_stats_and_selected_metadata(self):
        """Stop at the first loss construction, after real data/metadata setup."""
        cache_metadata = {"configuration": dict(teacher_Nx=8, teacher_L=4 * np.pi,
                          teacher_dt=.01, teacher_poisson_sign=-1.)}
        class SetupComplete(Exception):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "epoch000.npz"
            self.write_checkpoint(path)
            outdir = Path(tmp) / "run"
            args = ["--reference-cache", tmp, "--outdir", str(outdir), "--rollout-Nx", "8",
                    "--training-regimes", "linear_landau", "--normalization-checkpoint", str(path),
                    "--memory-backend", "burles_latent_fno", "--closure-history-input",
                    "--memory-steps", "2", "--memory-stride", "1", "--rollout-horizon", "2",
                    "--autonomous-history-burnin-steps", "1", "--epochs", "1", "--planned-epochs", "2",
                    "--diagnostic-start-times", "0", "--width", "2", "--spectral-modes", "2"]
            with mock.patch.object(trainer, "_load_reference_cache", return_value=(
                self.grouped, self.manifest, cache_metadata, "coefficients"
            )), mock.patch.object(trainer, "init_burles_latent_fno_params", return_value={"fresh": jnp.ones(1)}), mock.patch.object(
                trainer, "load_or_compute_training_statistics", side_effect=AssertionError("no recompute")
            ), mock.patch.object(trainer, "_load_checkpoint", side_effect=AssertionError("no weights")), mock.patch.object(
                trainer, "make_loss_function", side_effect=SetupComplete
            ) as make_loss:
                with self.assertRaises(SetupComplete):
                    trainer.main(args)
                metadata = json.loads((outdir / "run_configuration.json").read_text())
                self.assertEqual(metadata["regimes"], list(LINEAR))
                self.assertEqual(metadata["regime_indices"], [0])
                self.assertEqual(list(metadata["selected_training_case_ids"]), list(LINEAR))
                self.assertEqual(metadata["normalization_checkpoint"], str(path.resolve()))
                self.assertEqual(metadata["normalization_checkpoint_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertTrue(metadata["autonomous_history_burnin_detached"])
                self.assertTrue(metadata["compact_latent_initializer_gradient_disconnected"])
                self.assertEqual(make_loss.call_args.kwargs["regimes"], LINEAR)
                np.testing.assert_array_equal(make_loss.call_args.kwargs["regime_scales"], self.stats["regime_scales"])
                with self.assertRaisesRegex(ValueError, "training regimes"):
                    changed = list(args)
                    changed[changed.index("linear_landau")] = "all3"
                    trainer.main(changed + ["--resume-run", str(outdir)])

    def test_e1_to_e20_resume_keeps_planned1000_and_frozen_statistics(self):
        """Exercise actual persistence/optimizer scheduling with a scalar toy loss."""
        cache_metadata = {"configuration": dict(teacher_Nx=8, teacher_L=4 * np.pi,
                          teacher_dt=.01, teacher_poisson_sign=-1.)}

        def toy_loss(**kwargs):
            self.assertEqual(kwargs["regimes"], LINEAR)
            def loss(params, batch):
                value = jnp.sum(jnp.square(params["fresh"]))
                return value, jnp.asarray([value])
            return loss

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "epoch000.npz"
            self.write_checkpoint(path)
            outdir = Path(tmp) / "run"
            args = ["--reference-cache", tmp, "--outdir", str(outdir), "--rollout-Nx", "8",
                    "--training-regimes", "linear_landau", "--normalization-checkpoint", str(path),
                    "--rollout-horizon", "2", "--memory-steps", "2", "--memory-stride", "1",
                    "--diagnostic-start-times", "0", "--epochs", "1", "--planned-epochs", "1000",
                    "--full-anchor-sweep", "--history-stride", "20", "--batch-size", "2",
                    "--gradient-accumulation-steps", "1", "--no-translation-augmentation",
                    "--skip-evaluation"]
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                trainer, "_load_reference_cache", return_value=(
                    self.grouped, self.manifest, cache_metadata, "coefficients"
                )
            ), mock.patch.object(trainer, "init_spectral_memory_params", return_value={"fresh": jnp.ones(1)}), mock.patch.object(
                trainer, "load_or_compute_training_statistics", side_effect=AssertionError("no cache writes")
            ), mock.patch.object(trainer, "make_loss_function", side_effect=toy_loss), mock.patch.object(trainer, "_plot_losses"):
                trainer.main(args)
                _, _, _, state = trainer._load_training_state(outdir / "training_state.npz")
                self.assertEqual(state["global_epoch"], 1)
                args[args.index("--epochs") + 1] = "20"
                trainer.main(args + ["--resume-run", str(outdir)])
                _, optimizer, histories, state = trainer._load_training_state(outdir / "training_state.npz")
                self.assertEqual(state["global_epoch"], 20)
                self.assertEqual(int(optimizer["step"]), 20)
                self.assertEqual(len(histories["train_loss"]), 20)
                self.assertEqual(histories["val_regime_loss"].shape[1], 1)
                for name in ("epoch000_low_moment_closure.npz", "epoch020_low_moment_closure.npz"):
                    _, metadata, stats = trainer._load_checkpoint(outdir / name)
                    self.assertEqual(metadata["planned_epochs"], 1000)
                    self.assertEqual(metadata["steps_per_epoch"], 1)
                    self.assertEqual(metadata["steps_per_epoch_source"], "selected_full_anchor_count")
                    self.assertEqual(metadata["regimes"], list(LINEAR))
                    for key in self.stats:
                        np.testing.assert_array_equal(stats[key], self.stats[key])



if __name__ == "__main__":
    unittest.main()
