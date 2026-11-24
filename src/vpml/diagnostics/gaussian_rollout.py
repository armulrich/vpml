"""Compare autonomous electric-energy forecasts on a selected Gaussian validation shape."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vpml.datasets.manifest import read_manifest
from vpml.datasets.reader import _read_current_case
from vpml.datasets.training_index import read_completed_digest
from vpml.io.atomic import atomic_write
from vpml.io.serialization import canonical_json_bytes, file_digest, json_digest, read_json
from vpml.training.wavenumber_evaluation import rollout_factory
from vpml.training.wavenumber_training import Configuration, initialize


def select_cases(manifest, shape_id, strengths):
    """Resolve explicitly requested strengths of one validation shape."""
    strengths = sorted(strengths)
    if not strengths or len(set(strengths)) != len(strengths):
        raise ValueError("Specify one or more distinct strengths")
    by_strength = {case["strength"]: case for case in manifest["cases"]
                   if case["shape_id"] == shape_id and case["split"] == "validation"}
    if any(strength not in by_strength for strength in strengths):
        raise ValueError("Requested shape and strengths must belong to the validation split")
    return [by_strength[strength] for strength in strengths]


def electric_energy(density_perturbation, k0):
    """Periodic zero-mean Poisson solve and 1/2 integral E² dx on the retained grid."""
    density = np.asarray(density_perturbation, dtype=np.float64)
    coefficients = np.fft.rfft(density, axis=-1)
    coefficients[..., 0] = 0
    modes = np.arange(1, coefficients.shape[-1]) * k0
    coefficients[..., 1:] *= 1j / modes
    field = np.fft.irfft(coefficients, n=density.shape[-1], axis=-1)
    return (np.pi / k0) * np.mean(field**2, axis=-1)


def forecast_energy(params, initial, k0, config, rollout, *, steps, chunk_steps):
    """Return the finite prefix, including t=0, and the first failed step if observed.

    Check all three physical fields and electric energy at every step. The
    kernel returns its latent state at chunk boundaries. Reference samples
    after the initial condition are never inputs to the forecast.
    """
    if steps < 1 or chunk_steps < 1 or steps % chunk_steps:
        raise ValueError("Forecast must contain complete positive-length chunks")
    state = jnp.asarray(initial[None], dtype=config.compute_dtype)
    history = jnp.zeros((1, config.memory_steps, 3, config.nx), state.dtype)
    closure_history = jnp.zeros((1, config.memory_steps, config.nx), state.dtype)
    previous = jnp.zeros((1, config.nx), state.dtype)
    counter = jnp.asarray(0, jnp.int32)
    latent = None
    amplitude = jnp.asarray([np.sqrt(np.mean(initial[0]**2))], state.dtype)
    modes = jnp.arange(config.nx // 2 + 1, dtype=state.dtype) * k0
    energies = [np.atleast_1d(electric_energy(initial[0], k0))]
    for start in range(0, steps, chunk_steps):
        states, memory = rollout(params, state, history, closure_history, counter,
                                 previous, latent, amplitude, modes)
        history, closure_history, _, counter, previous, latent = memory
        state = states[:, -1]
        physical = np.asarray(states[0])
        with np.errstate(over="ignore", invalid="ignore"):
            energy = electric_energy(physical[:, 0], k0)
        finite = np.isfinite(physical).all(axis=(1, 2)) & np.isfinite(energy)
        finite[-1] &= np.isfinite(np.asarray(latent)).all()
        invalid = np.flatnonzero(~finite)
        if len(invalid):
            first = int(invalid[0])
            energies.append(energy[:first])
            return np.concatenate(energies), start + first + 1
        energies.append(energy)
    return np.concatenate(energies), None


def _read_parameters(checkpoint, expected_sha, protocol_sha, template):
    # Hash and read the same open inode, even if training publishes a newer one.
    with checkpoint.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha:
            raise ValueError("Checkpoint checksum mismatch")
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            record = json.loads(str(archive["record"]))
            step = record["accepted_updates"]
            if record["protocol_sha256"] != protocol_sha or int(archive["optimizer_step"]) != step:
                raise ValueError("Checkpoint disagrees with training protocol or optimizer step")
            params = {name[6:]: archive[name] for name in archive.files if name.startswith("param_")}
    if params.keys() != template.keys():
        raise ValueError("Checkpoint parameter names disagree with model")
    for name, value in params.items():
        if (value.shape != template[name].shape or value.dtype != template[name].dtype
                or not np.isfinite(value).all()):
            raise ValueError(f"Invalid checkpoint parameter: {name}")
    return jax.device_put(params), step


def _status(failed):
    return {"status": "complete" if failed is None else "nonfinite", "first_nonfinite_step": failed}


def _baseline_curve(directory, report, case_id, output_sha, times):
    case = next((c for c in report["cases"] if c["case_id"] == case_id), None)
    if case is None or case["reference_output_sha256"] != output_sha:
        raise ValueError(f"Baseline case or reference identity mismatch for {case_id}")
    path = directory / f"{case_id}.npz"
    if file_digest(path) != case["curves_sha256"]:
        raise ValueError(f"Baseline curves checksum mismatch for {case_id}")
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(times, saved["times"]):
            raise ValueError(f"Baseline time grid mismatch for {case_id}")
        energy = saved["initial_energy"]
    failed = case["initial"]["first_nonfinite_step"]
    expected_length = len(times) if failed is None else failed
    if energy.shape != (expected_length,) or not np.isfinite(energy).all():
        raise ValueError(f"Baseline curve disagrees with the recorded finite steps for {case_id}")
    return energy, failed


def evaluate(dataset, run, normalization, checkpoint, checkpoint_sha256, shape_id,
             strengths, output, *, require_gpu=False, baseline_from=None):
    """Publish paired forecasts into a new directory outside all input directories."""
    dataset, run, normalization, checkpoint, output = (
        Path(path).resolve() for path in (dataset, run, normalization, checkpoint, output)
    )
    if any(output.is_relative_to(path) for path in (dataset, run)):
        raise ValueError("Evaluation output must be outside the training run and dataset")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    backend = jax.default_backend()
    if require_gpu and backend != "gpu":
        raise RuntimeError("Evaluation requested a GPU")
    protocol = read_json(run / "training-protocol.json")
    protocol_sha = json_digest(protocol)
    manifest = read_manifest(dataset / "manifest.json")
    if json_digest(manifest) != protocol["dataset_sha256"]:
        raise ValueError("Dataset manifest differs from training")
    if file_digest(normalization) != protocol["normalization_sha256"]:
        raise ValueError("Normalization differs from training")
    cases = select_cases(manifest, shape_id, strengths)
    config = Configuration(**protocol["model"])
    numerics = manifest["numerics"]
    if (config.preparation_steps != 0 or config.nx != numerics["output_nx"]
            or config.dt != numerics["save_dt"]):
        raise ValueError("Expected aligned training grid, time samples and zero preparation steps")
    steps = round(numerics["final_time"] / config.dt)
    chunk_steps = math.gcd(steps, 200)
    times = np.arange(steps + 1) * config.dt
    initial_params = initialize(config)
    trained_params, step = _read_parameters(checkpoint, checkpoint_sha256, protocol_sha, initial_params)
    with np.load(normalization, allow_pickle=False) as data:
        stats = {name: data[name] for name in data.files}
    rollout = rollout_factory(config, stats, chunk_steps=chunk_steps)
    previous_report = None
    if baseline_from is not None:
        baseline_from = Path(baseline_from).resolve()
        if output.is_relative_to(baseline_from):
            raise ValueError("Evaluation output must be outside the previous baseline")
        previous_report = read_json(baseline_from / "report.json")
        if previous_report["protocol_sha256"] != protocol_sha or previous_report["shape_id"] != shape_id:
            raise ValueError("Baseline normalization, training protocol or shape does not match")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(shape_id=shape_id, checkpoint_step=step, checkpoint_sha256=checkpoint_sha256,
                  protocol_sha256=protocol_sha, dataset_sha256=protocol["dataset_sha256"],
                  normalization_sha256=protocol["normalization_sha256"], seed=config.seed,
                  dt=config.dt, final_time=numerics["final_time"], backend=backend, cases=[])
    print(json.dumps(dict(event="evaluation_start", steps=steps, chunk_steps=chunk_steps,
                          checkpoint_step=step, backend=backend, strengths=sorted(strengths))), flush=True)
    for case in cases:
        case_id, k0 = case["case_id"], case["k0"]
        reference_sha = read_completed_digest(dataset, case_id)
        arrays = _read_current_case(dataset, manifest, case, manifest_sha256=protocol["dataset_sha256"])
        if read_completed_digest(dataset, case_id) != reference_sha:
            raise ValueError(f"Reference changed while evaluating {case_id}")
        if not np.array_equal(times, arrays["times"]):
            raise ValueError("Reference and rollout time coordinates disagree")
        moments = arrays["raw_moments"]
        reference = electric_energy(moments[:, 0] - 1, k0)
        initial = moments[0].astype(np.float64, copy=True)
        initial[[0, 2]] -= 1
        del arrays, moments
        if previous_report is None:
            baseline, init_failed = forecast_energy(initial_params, initial, k0, config, rollout,
                                                      steps=steps, chunk_steps=chunk_steps)
        else:
            baseline, init_failed = _baseline_curve(baseline_from, previous_report, case_id,
                                                     reference_sha, times)
        print(json.dumps(dict(case_id=case_id, strength=case["strength"], model="initial",
                              saved_points=len(baseline), **_status(init_failed))), flush=True)
        trained, failed = forecast_energy(trained_params, initial, k0, config, rollout,
                                          steps=steps, chunk_steps=chunk_steps)
        path = output / f"{case_id}.npz"
        atomic_write(path, lambda f: np.savez_compressed(f, times=times, reference_energy=reference,
                                                         initial_energy=baseline, trained_energy=trained),
                     replace=False)
        case_record = dict(case_id=case_id, strength=case["strength"], k0=k0,
                           initial=_status(init_failed), trained=_status(failed),
                           reference_output_sha256=reference_sha, curves_sha256=file_digest(path))
        report["cases"].append(case_record)
        atomic_write(output / "report.json", lambda f: f.write(canonical_json_bytes(report, newline=True)))
        print(json.dumps(case_record), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "run", "normalization", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--strengths", type=float, nargs="+", required=True)
    parser.add_argument("--baseline-from", type=Path,
                        help="Reuse seeded curves from a previous compatible evaluation")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
