"""Regression checks for accumulation and exact mid-epoch training continuation."""
from dataclasses import replace

import jax
import numpy as np
import pytest

from vpml.io.serialization import json_digest
from vpml.training.gaussian_training import E10_MODEL, train, training_protocol
from vpml.training.optimization import adam_init
from vpml.training.wavenumber_training import initialize, load_snapshot, update_factory


class TinyWindows:
    strengths = (0.1, 0.3, 1.0)
    manifest = {"cases": [{"strength": s} for s in strengths]}
    manifest_sha256 = json_digest(manifest)
    training_outputs_sha256 = "fixture-output-identity"

    def __init__(self, config):
        self.config = config
        self.starts = []

    def epoch_rows(self, epoch, seed):
        rows = np.array([(case, start) for case in range(3) for start in range(2)])
        return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(rows)

    def batch(self, rows):
        config = self.config
        self.starts.extend(map(tuple, rows))
        x = np.arange(config.nx) * (2*np.pi/config.nx)
        strength = np.asarray([self.strengths[case] for case, _ in rows])
        amplitude = 0.01 * strength
        profile = amplitude[:, None] * np.cos(x[None, :] + np.asarray(rows)[:, 1, None])
        initial = np.stack([profile, profile*.1, profile], axis=1)
        count = len(rows)
        return dict(initial=initial, targets=np.repeat(initial[:, None], config.horizon, axis=1),
                    memory=np.zeros((count, config.memory_steps, 3, config.nx)),
                    heat_flux_gradient_history=np.zeros((count, config.memory_steps, config.nx)),
                    amplitude=amplitude, regime_index=np.zeros(count, np.int32),
                    k_arr=np.tile(np.arange(config.nx//2+1)*.3, (count, 1)))


def fixture():
    jax.config.update("jax_enable_x64", True)
    config = replace(E10_MODEL, nx=8, memory_steps=2, width=2, depth=1,
                     modes=2, latent_dim=2, horizon=2)
    stats = dict(input_scale=np.ones(4), regime_scales=np.ones((3, 4)),
                 heat_flux_gradient_scale=np.array([1.]), amplitude_center=np.array([0.]),
                 amplitude_scale=np.array([1.]))
    return config, stats


def test_accumulated_update_matches_full_e10_step_with_short_last_group():
    config, stats = fixture()
    data = TinyWindows(config)
    params = initialize(config)
    opt = adam_init(params)
    batch = data.batch([(0, 0), (1, 1), (2, 1)])
    actual = update_factory(config, stats, microbatch_size=2)(params, opt, batch, .0001)
    full = update_factory(config, stats)(params, opt, batch, .0001)
    assert bool(actual[-1]) and int(actual[1]["step"]) == 1
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(full)):
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-12)


def test_resume_replays_next_window_and_optimizer_with_extendible_epoch_target(tmp_path):
    config, stats = fixture()
    settings = dict(batch_size=3, microbatch_size=2, normalization_sha256="fixture")
    direct, segmented = TinyWindows(config), TinyWindows(config)
    direct_record = train(direct, config, stats, tmp_path/"direct", max_updates=3, **settings)

    # Interrupt immediately after the first completed step via the validation hook.
    interrupt = []
    def validation(params):
        interrupt.append(True)
        return {"finite": True}

    first = train(segmented, config, stats, tmp_path/"resume", max_epochs=1,
                  signal_received=lambda: bool(interrupt), validate=validation, **settings)
    assert (first["epoch"], first["next_batch"], first["accepted_updates"]) == (1, 1, 1)
    resumed = train(segmented, config, stats, tmp_path/"resume", max_epochs=3,
                    max_updates=3, **settings)
    assert segmented.starts == direct.starts and resumed == direct_record
    digest = json_digest(training_protocol(direct, config, "fixture", 3, 2))
    for i in (0, 1, 3):  # weights, Adam moments/step, and full loss rows
        uninterrupted = load_snapshot(tmp_path/"direct"/"checkpoint.npz", digest)[i]
        restored = load_snapshot(tmp_path/"resume"/"checkpoint.npz", digest)[i]
        for a, b in zip(jax.tree_util.tree_leaves(uninterrupted), jax.tree_util.tree_leaves(restored)):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="protocol"):
        train(segmented, config, stats, tmp_path/"resume", max_updates=4,
              **{**settings, "normalization_sha256": "different"})
