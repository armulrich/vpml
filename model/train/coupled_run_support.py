"""Durable experiment records and shared coupled-model construction."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import importlib.metadata

import numpy as np


def atomic_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_digest(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunRecord:
    """Write each stage immediately, with source identity and failure evidence."""

    def __init__(self, root: Path, configuration: dict, *, sources=()):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "run.json").exists():
            raise FileExistsError(f"Run already exists: {self.root}")
        self.started = time.monotonic()
        self.record = {
            "configuration": configuration, "pid": os.getpid(),
            "started_unix": time.time(), "status": "running",
            "sources": {str(Path(p).resolve()): file_digest(p) for p in sources},
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "jax", "jaxlib", "scipy")},
        }
        snapshots = self.root / "source_snapshot"
        snapshots.mkdir()
        for source, expected in self.record["sources"].items():
            data = Path(source).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Source changed while recording: {source}")
            (snapshots / expected).write_bytes(data)
        atomic_json(self.root / "run.json", self.record)
        self.stage("created")

    def stage(self, name: str, **values):
        row = {"stage": name, "elapsed_seconds": time.monotonic() - self.started,
               "time_unix": time.time(), **values}
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row, allow_nan=False), flush=True)

    def finish(self, status="complete", **values):
        self.record.update(status=status, elapsed_seconds=time.monotonic() - self.started,
                           **values)
        atomic_json(self.root / "run.json", self.record)
        self.stage(status, **values)


def make_model_functions(config, arrays, teacher, *, horizon=None, **overrides):
    """Use the existing compact model; pass only supported objective options."""
    from model.train.coupled_low_moment_latent import (
        _make_functions, _matrix_square_root_propagator,
    )
    config = {**config, **overrides}
    arguments = {
        key: config[key] for key in inspect.signature(_make_functions).parameters
        if key in config
    }
    for key in ("basis", "k_arr", "resolved_center", "resolved_scale",
                "latent_center", "latent_scale", "correction_bounds",
                "latent_residual_scale", "latent_input_scale"):
        if key in arrays:
            arguments[key] = arrays[key]
    propagator = arrays["linear_propagator"]
    if config.get("linear_baseline") == "semilinear_strang":
        propagator = _matrix_square_root_propagator(propagator)
    arguments.update(propagator=propagator,
                     fine_dt=float(teacher["teacher_dt"]),
                     poisson_sign=float(teacher["teacher_poisson_sign"]),
                     horizon=int(horizon or config["rollout_steps"]))
    return _make_functions(**arguments)


def field_from_resolved(resolved, k_arr, *, poisson_sign=1.0):
    """Match E=i rho_hat/k in the coupled solver, with a zero mean field."""
    density_hat = np.fft.rfft(np.asarray(resolved, dtype=np.float64)[..., 0, :], axis=-1)
    field_hat = np.zeros_like(density_hat)
    field_hat[..., 1:] = poisson_sign * 1j * density_hat[..., 1:] / np.asarray(k_arr)[1:]
    return np.fft.irfft(field_hat, n=resolved.shape[-1], axis=-1)
