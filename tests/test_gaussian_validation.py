from dataclasses import asdict, replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from vpml.cli.gaussian_validation import rescore
from vpml.datasets.gaussian_training import GaussianTrainingData
from vpml.datasets.manifest import Numerics, build_manifest
from vpml.datasets.runner import run_dataset
from vpml.io.serialization import file_digest, json_digest, read_json
from vpml.training.gaussian_checkpoints import TrainingCheckpoints, write_json
from vpml.training.gaussian_validation import make_validation_report, validation_windows
from vpml.training.optimization import adam_init
from vpml.training.wavenumber_training import Configuration, batch_from_states, initialize


def test_external_panel_resolves_exact_windows_and_rejects_mismatched_references(monkeypatch):
    cases = [
        dict(case_id=f"case-{split}", split=split, strength=0.7, k0=0.4, mode_count=3,
             shape_id=f"shape-{split}", decay=1.2)
        for split in ("train", "validation", "test")
    ]
    config = Configuration()
    selections = []
    def batch(rows):
        selections.extend(rows)
        return np.array([anchor for _, anchor in rows], dtype=float)
    data = SimpleNamespace(manifest={"cases": cases}, cases=cases, anchors_per_case=7,
                           config=config, anchor_stride=8, batch=batch)
    monkeypatch.setattr("vpml.training.gaussian_validation.window_loss_factory",
                        lambda config, stats: lambda params, batch: batch)
    def evaluate(rows):
        return make_validation_report(data, config, {}, batch_size=2, panel={"rows": rows})

    row = dict(case_id="case-validation", case_index=1, anchor_index=4)
    report = evaluate([row, dict(row, anchor_index=1)])
    result = report(None)
    assert result["mean_relative_l2"] == 2.5
    assert selections == [(1, 4), (1, 1)]
    assert report.panel_sha256 != evaluate([row]).panel_sha256
    for invalid in ([row, row], [dict(row, case_id="case-test")], [dict(row, case_index=-1)],
                    [dict(row, case_id="case-test", case_index=2)],
                    [dict(row, case_id="case-train", case_index=0)],
                    [dict(row, anchor_index=7)], [dict(row, anchor_index=-1)]):
        with pytest.raises(ValueError):
            evaluate(invalid)


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


def test_rescore_nonbest_archive_on_a_different_panel(tmp_path):
    jax.config.update("jax_enable_x64", True)
    reference, run, output = [tmp_path / name for name in ("reference", "run", "new-scores")]
    definition = build_manifest(k0_values=(0.4,), mode_counts=(3,), decays=(2.0,),
                                draws_per_combination=2, split_counts=(1, 1, 0), strengths=(.7,),
                                numerics=Numerics(nx=16, nv=64, output_nx=8, dt=.025, save_dt=.025,
                                                  final_time=.25, storage_dtype="float64"))
    run_dataset(definition, reference, backend="cpu", max_seconds=60)
    config = Configuration(nx=8, width=2, modes=2, depth=1, latent_dim=2, memory_steps=2,
                           preparation_steps=0, horizon=2, relative_loss_norm="l2", compute_dtype="float64")
    data = GaussianTrainingData(reference, tmp_path / "cache", config)
    norm = tmp_path / "normalization.npz"
    stats = dict(input_scale=np.ones(4), regime_scales=np.ones((3, 4)), heat_flux_gradient_scale=np.ones(1),
                 amplitude_center=np.zeros(1), amplitude_scale=np.ones(1))
    np.savez(norm, **stats)
    protocol = dict(model=asdict(config), microbatch_size=1, dataset_sha256=data.manifest_sha256,
                    training_outputs_sha256=data.training_outputs_sha256, normalization_sha256=file_digest(norm))
    run.mkdir()
    write_json(run / "training-protocol.json", protocol)
    checkpoints = TrainingCheckpoints(run, json_digest(protocol), archive_validations=True)
    records = []
    with checkpoints.open_history({"loss_history_bytes": 0}) as history:
        for step in (50, 100):
            params = initialize(replace(config, seed=step))
            optimizer = dict(adam_init(params), step=step)
            report = dict(step=step, panel_sha256="training-time-panel")
            record = dict(protocol_sha256=json_digest(protocol), accepted_updates=step,
                          validation={"last": report, "best_step": 50})
            checkpoints.save(params, optimizer, record, history)
            records.append(params)
    case_index = next(i for i, case in enumerate(definition["cases"]) if case["split"] == "validation")
    panel = {"rows": [dict(case_id=definition["cases"][case_index]["case_id"], case_index=case_index,
                           anchor_index=1)]}
    panel_file = tmp_path / "new-panel.json"
    write_json(panel_file, panel)
    archived = run / "snapshots" / "step-00000100"
    original_sha = file_digest(archived / "checkpoint.npz")
    rescore(reference, run, norm, panel_file, [50, 100], output)
    scorer = make_validation_report(data, config, stats, batch_size=1, panel=panel)
    for step, params in zip((50, 100), records, strict=True):
        actual = read_json(output / f"step-{step:08d}.json")
        assert actual["mean_relative_l2"] == pytest.approx(scorer(params)["mean_relative_l2"])
        assert actual["selection_sha256"] == json_digest(panel)
        assert actual["checkpoint_sha256"] == file_digest(run / "snapshots" / f"step-{step:08d}" / "checkpoint.npz")
    assert file_digest(archived / "checkpoint.npz") == original_sha
    with pytest.raises(FileExistsError):
        rescore(reference, run, norm, panel_file, [100], output)
