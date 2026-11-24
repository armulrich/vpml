"""Regression checks for accumulation and exact partial-round continuation."""
import json
from dataclasses import replace

import jax
import numpy as np
import pytest

from vpml.io.serialization import json_digest
from vpml.training import gaussian_training
from vpml.training.gaussian_training import E10_MODEL, train, training_protocol, record_validation
from vpml.training.gaussian_checkpoints import TrainingCheckpoints
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

    def round_updates(self, batch_size):
        return 25

    def sample_rows(self, step, seed, batch_size):
        rows = np.array([(case, start) for case in range(3) for start in range(2)])
        return np.random.default_rng(np.random.SeedSequence([seed, step])).permutation(rows)[:batch_size]

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


def test_resume_replays_prefetched_batch_and_optimizer(tmp_path, monkeypatch):
    config, stats = fixture()
    settings = dict(batch_size=3, microbatch_size=2, normalization_sha256="fixture",
                    validate=lambda p: {"mean_relative_l2": 0.4})
    direct, segmented = TinyWindows(config), TinyWindows(config)
    direct_record = train(direct, config, stats, tmp_path/"direct", max_updates=3, **settings)

    # Update 3 has already been submitted to the reader when update 2 commits.
    interrupt = []
    def observe(event, **values):
        if event == "update" and values["step"] == 2:
            interrupt.append(True)
    monkeypatch.setattr(gaussian_training, "log_event", observe)
    first = train(segmented, config, stats, tmp_path/"resume", max_rounds=1,
                  signal_received=lambda: bool(interrupt), **settings)
    assert first["accepted_updates"] == 2 and first["stop_reason"] == "signal"
    assert len(segmented.starts) == 9
    resumed = train(segmented, config, stats, tmp_path/"resume", max_rounds=1, max_updates=3, **settings)
    assert resumed["accepted_updates"] == direct_record["accepted_updates"] == 3
    assert resumed["windows_by_strength"] == direct_record["windows_by_strength"]
    assert segmented.starts == direct.starts + direct.starts[-3:]
    assert (tmp_path/"direct"/"training-loss.jsonl").read_bytes() == (tmp_path/"resume"/"training-loss.jsonl").read_bytes()
    digest = json_digest(training_protocol(direct, config, "fixture", 3, 2))
    for i in (0, 1):  # the restored parameters and both Adam moments agree bit for bit
        uninterrupted = load_snapshot(tmp_path/"direct"/"checkpoint.npz", digest)[i]
        restored = load_snapshot(tmp_path/"resume"/"checkpoint.npz", digest)[i]
        for a, b in zip(jax.tree_util.tree_leaves(uninterrupted), jax.tree_util.tree_leaves(restored)):
            np.testing.assert_array_equal(a, b)
    assert load_snapshot(tmp_path/"resume"/"best-checkpoint.npz", digest)[2]["accepted_updates"] == 1
    with pytest.raises(ValueError, match="protocol"):
        train(segmented, config, stats, tmp_path/"resume", max_updates=4,
              **{**settings, "normalization_sha256": "different"})

    # A later torn/corrupt latest checkpoint can be recovered from a separate,
    # atomically retained previous update, even when the loss log has run ahead.
    (tmp_path/"resume"/"checkpoint.npz").write_bytes(b"unreadable")
    snapshots = TrainingCheckpoints(tmp_path/"resume", digest)
    _, _, previous = snapshots.restore()
    assert previous["accepted_updates"] == 2
    with snapshots.open_history(previous):
        pass
    loss_rows = (tmp_path/"resume"/"training-loss.jsonl").read_text().splitlines()
    assert [json.loads(line)[0] for line in loss_rows] == [1, 2]


def test_plateau_ignores_irregular_checks_and_accumulates_small_gains():
    state = dict(best_score=None, best_step=None, significant_best=None, checks_without_gain=0)
    def score(step, value):
        record_validation(state, dict(step=step, mean_relative_l2=value))
    score(28, 0.35)
    assert state["significant_best"] is None
    score(50, 0.340)
    score(75, 0.338)
    score(79, 0.337)
    assert state["checks_without_gain"] == 1
    score(100, 0.336)
    assert state["checks_without_gain"] == 0
    score(125, 0.3359)
    score(150, 0.3358)
    score(171, 0.3357)
    score(175, 0.3358)
    assert state["checks_without_gain"] == 3
    assert state["significant_best"] == 0.336 and state["best_step"] == 171


def test_rollback_restores_the_best_for_the_previous_update(tmp_path):
    checkpoints = TrainingCheckpoints(tmp_path, "test")
    params = {"a": np.zeros(1)}
    opt = adam_init(params)
    record = dict(protocol_sha256="test", accepted_updates=0, loss_history_bytes=0,
                  validation={"best_step": None})
    with checkpoints.open_history(record) as history:
        for step, best in ((0, None), (1, 1), (2, 1), (3, 3)):
            params["a"] = np.full(1, step)
            opt["step"] = step
            record.update(accepted_updates=step, validation={"best_step": best})
            checkpoints.save(params, opt, record, history)

    checkpoints.latest.write_bytes(b"interrupted write")
    recovered = TrainingCheckpoints(tmp_path, "test")
    _, _, state = recovered.restore()
    assert state["accepted_updates"] == 2 and state["validation"]["best_step"] == 1
    selected, _, _, _ = load_snapshot(recovered.best, "test")
    np.testing.assert_array_equal(selected["a"], [1])


def test_interrupted_validation_and_plateau_survive_restarts(tmp_path, monkeypatch):
    config, stats = fixture()
    data = TinyWindows(config)
    monkeypatch.setattr(gaussian_training, "initialize", lambda _: {"a": np.float64(0)})
    def make_update(*args, **kwargs):
        def update(p, opt, batch, rate):
            return ({"a": p["a"] + 1}, dict(opt, step=opt["step"] + 1),
                    1., 0., 0., 0., True)
        return jax.jit(update)
    monkeypatch.setattr(gaussian_training, "update_factory", make_update)
    checks = []
    def validate(p):
        checks.append(int(p["a"]))
        if len(checks) == 1:
            raise RuntimeError("interrupted validation")
        return {"mean_relative_l2": 0.5}
    args = dict(batch_size=3, microbatch_size=2, normalization_sha256="fixture",
                validate=validate)
    with pytest.raises(RuntimeError, match="interrupted validation"):
        train(data, config, stats, tmp_path, max_updates=150, **args)
    result = train(data, config, stats, tmp_path, max_updates=150, **args)
    assert result["accepted_updates"] == 125
    assert result["stop_reason"] == "validation_plateau"
    assert checks == [1, 1, 28, 50, 75, 100, 125]
    with pytest.raises(ValueError, match="resume-after-plateau"):
        train(data, config, stats, tmp_path, max_updates=150, **args)
    result = train(data, config, stats, tmp_path, max_updates=150, resume_after_plateau=True, **args)
    assert result["accepted_updates"] == 150 and result["stop_reason"] == "update_limit"
    assert result["validation"]["checks_without_gain"] == 1
