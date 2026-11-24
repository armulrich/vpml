from types import SimpleNamespace

import jax
import numpy as np
import pytest

from vpml.training.gaussian_validation import make_validation_report, validation_windows
from vpml.training.wavenumber_training import Configuration, batch_from_states, initialize


def test_panel_pairs_shapes_and_times_across_strengths_and_excludes_test_rows():
    cases = [
        dict(split=split, k0=k0, mode_count=3, decay=1.2, shape_id=shape, strength=strength)
        for split in ("train", "validation", "test")
        for k0 in (0.3, 0.5)
        for shape in ("second", "first")
        for strength in (0.1, 1.0)
    ]
    rows = validation_windows({"cases": cases}, 526)
    assert [(cases[i]["split"], cases[i]["shape_id"], strength, anchor)
            for strength, i, anchor in rows] == [
        ("validation", "first", 0.1, 0), ("validation", "first", 1.0, 0),
        ("validation", "first", 0.1, 525), ("validation", "first", 1.0, 525),
    ]
    cases.pop(rows[-1][1])
    with pytest.raises(ValueError, match="Incomplete strength"):
        validation_windows({"cases": cases}, 526)


def test_batched_validation_matches_individual_windows_and_keeps_divergence_visible():
    jax.config.update("jax_enable_x64", True)
    config = Configuration(nx=8, modes=2, width=2, depth=1, latent_dim=2, memory_steps=2,
                           preparation_steps=0, horizon=2, relative_loss_norm="l2",
                           compute_dtype="float64")
    stats = dict(input_scale=np.ones(4), regime_scales=np.ones((3, 4)),
                 heat_flux_gradient_scale=np.ones(1), amplitude_center=np.zeros(1),
                 amplitude_scale=np.ones(1))
    cases = []
    states = {}
    x = np.arange(config.nx) * 2 * np.pi / config.nx
    for k0 in (0.3, 0.5):
        for strength in (0.1, 1.0):
            case_id = f"case-{k0}-{strength}"
            cases.append(dict(case_id=case_id, split="validation", k0=k0, mode_count=3,
                              decay=1.2, shape_id=f"shape-{k0}", strength=strength,
                              epsilon=strength, domain_length=2*np.pi/k0, regime="linear_landau"))
            profile = strength * 0.02 * np.cos(x)
            states[case_id] = np.broadcast_to([profile, profile * .1, profile], (11, 3, config.nx))
    manifest = {"cases": cases}
    reads = []

    def batch(rows):
        reads.append(len(rows))
        return batch_from_states(manifest, rows, states, config)

    data = SimpleNamespace(manifest=manifest, anchors_per_case=2, batch=batch)
    params = initialize(config)
    scalar = make_validation_report(data, config, stats, batch_size=1)
    batched = make_validation_report(data, config, stats, batch_size=3)
    assert not reads
    reference = scalar(params)
    assert reads == [1, 1, 1, 1]
    reads.clear()
    actual = batched(params)
    assert reads == [3, 1]
    assert actual["panel_sha256"] == scalar.panel_sha256 == batched.panel_sha256
    assert actual["validation_windows"] == reference["validation_windows"] == 4
    assert actual["mean_relative_l2"] == pytest.approx(reference["mean_relative_l2"], rel=1e-12)
    for strength in ("0.1", "1.0"):
        scores = actual["by_strength"][strength]
        assert scores["windows"] == scores["finite"] == 2
        assert scores["relative_l2"] == pytest.approx(reference["by_strength"][strength]["relative_l2"])
        assert scores["mean_relative_l2"] == pytest.approx(np.mean(scores["relative_l2"]))

    # Zero target power yields an undefined relative error for just one window.
    states[cases[-1]["case_id"]] = np.zeros((11, 3, config.nx))
    divergent = batched(params)
    assert divergent["mean_relative_l2"] is None
    assert divergent["by_strength"]["0.1"] == actual["by_strength"]["0.1"]
    assert divergent["by_strength"]["1.0"]["finite"] == 1
    assert divergent["by_strength"]["1.0"]["mean_relative_l2"] is None
    assert divergent["by_strength"]["1.0"]["relative_l2"][-1] is None
