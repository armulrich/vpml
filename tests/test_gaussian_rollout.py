"""Held-out rollout selection, physical energy, and autonomous memory propagation."""
from dataclasses import replace
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from vpml.diagnostics import gaussian_rollout
from vpml.datasets.manifest import Numerics, build_manifest
from vpml.datasets.reader import read_case
from vpml.datasets.runner import run_dataset
from vpml.io.serialization import file_digest, json_digest
from vpml.training.gaussian_training import E10_MODEL, training_protocol
from vpml.training.optimization import adam_init
from vpml.training.wavenumber_evaluation import autonomous_case, rollout_factory
from vpml.training.wavenumber_training import initialize, save_snapshot


def test_only_requested_strengths_of_one_validation_shape_are_selected():
    cases = [
        {"shape_id": shape, "split": split, "strength": strength, "case_id": f"{shape}-{strength}"}
        for shape, split in [("train", "train"), ("v1", "validation"),
                             ("v2", "validation"), ("test", "test")]
        for strength in (1.0, 0.5, 0.1)
    ]
    manifest = {"cases": cases}
    selected = gaussian_rollout.select_cases(manifest, "v1", [1.0, 0.1])
    assert [c["case_id"] for c in selected] == ["v1-0.1", "v1-1.0"]
    for shape, strengths in [("v2", []), ("train", [0.5]), ("test", [0.1]), ("v1", [0.3])]:
        with pytest.raises(ValueError):
            gaussian_rollout.select_cases(manifest, shape, strengths)


def test_periodic_poisson_energy_has_correct_wavenumber_and_integral():
    k0 = 0.4
    phase = 2 * np.pi * np.arange(16) / 16
    amplitudes = np.array([0.0, 0.2, 0.7])
    # For n = constant + A cos(2 k0 x), E = -A sin(2 k0 x)/(2 k0).
    density = 0.13 + amplitudes[:, None] * np.cos(2 * phase)
    result = gaussian_rollout.electric_energy(density, k0)
    expected = (2 * np.pi / k0) * amplitudes**2 / (4 * (2 * k0)**2)
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-15)


def test_chunked_forecast_matches_one_uninterrupted_solver_call():
    jax.config.update("jax_enable_x64", True)
    config = replace(E10_MODEL, nx=8, width=2, depth=1, modes=2,
                     latent_dim=2, memory_steps=3, memory_stride=2)
    stats = dict(input_scale=np.ones(4), heat_flux_gradient_scale=np.ones(1),
                 amplitude_center=np.zeros(1), amplitude_scale=np.ones(1))
    k0 = 0.4
    phase = 2 * np.pi * np.arange(config.nx) / config.nx
    initial = np.stack((.02 * np.cos(phase), .01 * np.sin(phase), .03 * np.cos(phase)))
    case = {"case_id": "v1", "epsilon": np.sqrt(np.mean(initial[0]**2)),
            "domain_length": 2 * np.pi / k0}
    params = initialize(config)
    direct, _ = autonomous_case(params, case, initial, config, rollout_factory(config, stats, 8),
                                horizon=8 * config.dt, chunk_steps=8)
    energy, first_nonfinite = gaussian_rollout.forecast_energy(
        params, initial, k0, config, rollout_factory(config, stats, 2), steps=8, chunk_steps=2,
    )
    assert first_nonfinite is None
    assert len(energy) == 9
    np.testing.assert_allclose(energy, gaussian_rollout.electric_energy(direct[:, 0], k0),
                               rtol=1e-10, atol=1e-14)


def test_invalid_moment_stops_energy_curve_at_first_failed_step():
    config = replace(E10_MODEL, nx=8, memory_steps=2)
    initial = np.zeros((3, config.nx))
    initial[0] = .02 * np.cos(2 * np.pi * np.arange(config.nx) / config.nx)

    def invalid_chunk(params, state, history, closure_history, counter, previous, latent, amp, k):
        states = np.repeat(np.asarray(state)[:, None], 4, axis=1)
        # An invalid momentum must stop the curve even if density still yields finite energy.
        states[:, 2, 1, :] = np.nan
        return states, (history, closure_history, None, counter, previous, np.zeros(1))

    curve, failed_step = gaussian_rollout.forecast_energy(
        {}, initial, .4, config, invalid_chunk, steps=8, chunk_steps=4,
    )
    assert failed_step == 3
    assert len(curve) == 3  # t=0 and two accepted advances
    np.testing.assert_allclose(curve, gaussian_rollout.electric_energy(initial[0], .4))


def test_evaluation_reads_only_selected_completed_case_and_preserves_inputs(tmp_path, monkeypatch):
    jax.config.update("jax_enable_x64", True)
    dataset, run, output = (tmp_path / name for name in ("dataset", "training", "evaluation"))
    run.mkdir()
    config = replace(E10_MODEL, nx=8, width=2, depth=1, modes=2, latent_dim=2, memory_steps=2)
    numerics = Numerics(nx=16, nv=64, output_nx=8, dt=config.dt, save_dt=config.dt,
                        final_time=4 * config.dt, storage_dtype="float64")
    manifest = build_manifest(k0_values=(.4,), mode_counts=(3,), decays=(2.,),
                              strengths=(.3, .5), draws_per_combination=3, split_counts=(1, 1, 1),
                              numerics=numerics)
    case = next(c for c in manifest["cases"] if c["split"] == "validation" and c["strength"] == .5)
    run_dataset(manifest, dataset, backend="cpu", max_seconds=60, case_ids=[case["case_id"]])
    normalization = tmp_path / "normalization.npz"
    np.savez(normalization, input_scale=np.ones(4), heat_flux_gradient_scale=np.ones(1),
             amplitude_center=np.zeros(1), amplitude_scale=np.ones(1))
    data = SimpleNamespace(manifest_sha256=json_digest(manifest), training_outputs_sha256="a"*64)
    protocol = training_protocol(data, config, file_digest(normalization), 48, 12)
    (run / "training-protocol.json").write_text(json.dumps(protocol))
    checkpoint = run / "checkpoint.npz"
    params = initialize(replace(config, seed=config.seed + 1))
    optimizer = adam_init(params)
    optimizer["step"] = np.int64(103)
    digest = save_snapshot(checkpoint, params, optimizer,
                           {"accepted_updates": 103, "protocol_sha256": json_digest(protocol)}, [])

    def input_digests():
        return {p: file_digest(p) for tree in (run, dataset) for p in tree.rglob("*") if p.is_file()}

    before = input_digests()
    gaussian_rollout.evaluate(dataset, run, normalization, checkpoint, digest, case["shape_id"],
                              [case["strength"]], output)
    assert before == input_digests()
    curve_file, = output.glob("*.npz")
    raw = read_case(dataset, case["case_id"])
    with np.load(curve_file) as curves:
        np.testing.assert_array_equal(curves["times"], raw["times"])
        expected = gaussian_rollout.electric_energy(raw["raw_moments"][:, 0] - 1., case["k0"])
        np.testing.assert_allclose(curves["reference_energy"], expected)
        for key in ("initial_energy", "trained_energy"):
            assert curves[key].shape == expected.shape
            assert curves[key][0] == pytest.approx(expected[0])
        assert not np.array_equal(curves["trained_energy"][1:], curves["initial_energy"][1:])
        baseline = curves["initial_energy"]

    # Additional checkpoints should require only one new forward prediction.
    forecast = gaussian_rollout.forecast_energy
    forecasts = []

    def record_forecast(*args, **kwargs):
        forecasts.append(True)
        return forecast(*args, **kwargs)

    monkeypatch.setattr(gaussian_rollout, "forecast_energy", record_forecast)
    args = (dataset, run, normalization, checkpoint, digest, case["shape_id"], [case["strength"]])
    reused = tmp_path / "reused"
    gaussian_rollout.evaluate(*args, reused, baseline_from=output)
    assert len(forecasts) == 1
    with np.load(reused / curve_file.name) as curves:
        np.testing.assert_array_equal(curves["initial_energy"], baseline)
    with curve_file.open("ab") as f:
        f.write(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        gaussian_rollout.evaluate(*args, tmp_path / "rejected", baseline_from=output)
    with pytest.raises(ValueError):
        gaussian_rollout.evaluate(*args, run / "rejected")
    assert before == input_digests()
