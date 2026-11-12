"""Bounded, single-writer execution of a frozen Gaussian reference dataset."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import socket
import time
from typing import Callable
import uuid

import numpy as np

from .manifest import Numerics, read_manifest, validate_manifest, write_manifest
from .selection import select_cases
from .schema import OUTPUT_SCHEMA, validate_completed_progress
from .provenance import json_digest, runtime_record, source_record
from vpml.io.atomic import atomic_write
from vpml.io.serialization import canonical_json_bytes, read_json


@contextmanager
def case_lock(path: Path):
    """Claim one case; stale locks require explicit operator investigation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    owner = {"token": token, "pid": os.getpid(), "host": socket.gethostname(),
             "job_id": os.environ.get("SLURM_JOB_ID")}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(owner, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        # Never remove a replacement owner's lock.
        if path.exists() and json.loads(path.read_text()).get("token") == token:
            path.unlink()


@contextmanager
def stop_signals():
    """Signal handlers only set a flag; all checkpoint I/O runs normally."""
    requested = [False]

    def request_stop(signum, frame):
        requested[0] = True

    previous = {}
    try:
        for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
            if hasattr(signal, name):
                number = getattr(signal, name)
                previous[number] = signal.signal(number, request_stop)
        yield lambda: requested[0]
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def sampled_outputs(history: dict, numerics: Numerics) -> dict:
    """Interpolate fine-time observations; never confuse raw and central fields.

    Spatial projection has already been applied to each defined observable.
    For save_dt=.025 and dt=.01, odd saved samples interpolate neighboring
    numerical steps.
    """
    expected = numerics.nsteps + 1
    if any(value.shape[0] != expected for value in history.values()):
        raise ValueError("History length does not match the final solver step")
    count = round(numerics.final_time / numerics.save_dt) + 1
    times = np.arange(count, dtype=np.float64) * numerics.save_dt
    indices = np.clip(times / numerics.dt, 0, numerics.nsteps)
    # Snap roundoff-close integer indices before interpolation.
    nearest = np.rint(indices)
    indices = np.where(np.abs(indices - nearest) < 1e-10, nearest, indices)
    lower = np.floor(indices).astype(int)
    upper = np.minimum(lower + 1, numerics.nsteps)
    fraction = indices - lower
    names = ("fields", "raw_moments", "electric")
    if "linear_electric" in history:
        names += ("linear_electric",)
    if set(history) != set(names):
        raise ValueError("Unexpected observation arrays")
    output = {"times": times}
    for name in names:
        values = history[name]
        weight = fraction.reshape((-1,) + (1,) * (values.ndim - 1))
        result = (1 - weight) * values[lower] + weight * values[upper]
        output[name] = result.astype(numerics.storage_dtype)
    if not all(np.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("Nonfinite values in final output")
    return output


def _invalid_fields(fields: np.ndarray) -> list[str]:
    """Reject unusable moments already computed for the saved observations."""
    return [f"nonpositive_{name}" for name, index in (("density", 0), ("pressure", 2))
            if np.any(fields[:, index] <= 0)]


def _completion_report(outputs: dict) -> dict:
    """Describe output validity without claiming numerical convergence."""
    metrics = {f"min_saved_{name}": float(outputs["fields"][:, index].min())
               for name, index in (("density", 0), ("pressure", 2))}
    failures = [f"nonpositive_saved_{name}" for name in ("density", "pressure")
                if metrics[f"min_saved_{name}"] <= 0]
    return {"checks_passed": not failures, "failures": failures, "metrics": metrics,
            "validation_scope": "finite output and positive density/pressure"}


def dataset_status(root: Path, manifest_path: Path | None = None) -> dict:
    """Read lightweight progress; this summary does not certify data integrity."""
    counts = {"complete": 0, "partial": 0, "failed": 0, "pending": 0}
    manifest = read_manifest(manifest_path or root / "manifest.json")
    if manifest_path is not None and (root / "manifest.json").exists():
        if read_manifest(root / "manifest.json") != manifest:
            raise ValueError("Status manifest differs from the existing dataset")
    cases = []
    for case in manifest["cases"]:
        case_id = case["case_id"]
        progress_path = root / "cases" / case_id / "progress.json"
        progress = read_json(progress_path) if progress_path.exists() else {}
        status = "pending"
        if (root / "failures" / f"{case_id}.json").exists():
            status = "failed"
        elif progress:
            status = "complete" if progress["status"] == "complete" else "partial"
            if status == "complete" and not progress["report"]["checks_passed"]:
                status = "failed"
        counts[status] += 1
        cases.append({"case_id": case_id, "status": status, "step": progress.get("step", 0)})
    return {"counts": counts, "cases": cases, "integrity_verified": False}


def run_dataset(
    manifest: dict,
    output_root: Path,
    *,
    backend: str,
    max_seconds: float,
    case_ids: list[str] | None = None,
    linearized: bool = False,
    stop_on_failure: bool = False,
    source_transition: Path | None = None,
    source_transition_sha256: str | None = None,
    stop_requested: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Process cases until stopped, retaining exact committed continuation states.

    This function does not submit jobs, acquire GPUs, retry failed cases, or
    start training. The caller owns resource allocation and the time limit.
    With ``stop_on_failure``, the first recorded or newly detected failed case
    ends this invocation before the next case's evolution is constructed.
    """
    from .checkpoints import CheckpointStore, atomic_json
    from .evolution import ReferenceEvolution
    from .observations import observation_schema
    from .continuation import (
        continuation_protocol, validate_provenance, validate_transition, write_transition,
    )

    measured_start = time.perf_counter()
    timings = {name: 0.0 for name in (
        "advance_and_observe_seconds", "host_transfer_seconds",
        "checkpoint_seconds", "finalize_seconds",
    )}
    measured_blocks = 0
    validate_manifest(manifest)
    if manifest["output_schema"] != OUTPUT_SCHEMA:
        raise ValueError("Legacy output schemas are read-only; start a new dataset to evolve")
    if type(stop_on_failure) is not bool:
        raise ValueError("stop_on_failure must be a boolean")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    deadline = clock() + max_seconds
    numerics = Numerics(**manifest["numerics"])
    cases = select_cases(manifest, case_ids)
    source = source_record()
    transition = read_json(Path(source_transition)) if source_transition is not None else None
    if source_transition_sha256 is not None and (
            transition is None or json_digest(transition) != source_transition_sha256):
        raise ValueError("Source transition checksum mismatch")
    if transition is not None:
        validate_transition(transition, current_source=source["compatibility"],
                            manifest_sha256=json_digest(manifest))
    runtime = runtime_record(backend)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        try:
            write_manifest(manifest_path, manifest)
        except FileExistsError:
            pass  # Another worker may have published the identical manifest.
    if read_manifest(manifest_path) != manifest:
        raise ValueError("Output directory belongs to a different manifest")
    provenance_directory = root / "provenance"
    provenance_directory.mkdir(exist_ok=True)
    provenance = {"source": source, "runtime": runtime}
    if transition is not None:
        transition_digest = json_digest(transition)
        write_transition(root / "source-transitions" / f"{transition_digest}.json", transition)
        provenance["source_transition_sha256"] = transition_digest
    provenance_digest = json_digest(provenance)
    provenance_path = provenance_directory / f"{provenance_digest}.json"
    if not provenance_path.exists():
        try:
            atomic_write(provenance_path, lambda handle: handle.write(
                canonical_json_bytes(provenance, newline=True)), replace=False)
        except FileExistsError:
            pass
    if read_json(provenance_path) != provenance:
        raise ValueError("Source provenance record mismatch")
    (root / "failures").mkdir(exist_ok=True)
    outcomes = []
    for case in cases:
        if stop_on_failure and outcomes and outcomes[-1]["status"] == "failed":
            break
        if stop_requested() or clock() >= deadline:
            break
        case_id = case["case_id"]
        failure_path = root / "failures" / f"{case_id}.json"
        if failure_path.exists():
            outcomes.append({"case_id": case_id, "status": "failed"})
            continue
        try:
            with case_lock(root / "locks" / f"{case_id}.lock"):
                protocol = {"manifest_sha256": json_digest(manifest), "case": case,
                            "numerics": manifest["numerics"], "runtime": runtime,
                            "source": source["compatibility"], "linearized": linearized,
                            "output_schema": manifest["output_schema"],
                            "observations": observation_schema()}
                directory = root / "cases" / case_id
                if (directory / "protocol.json").exists():
                    protocol = continuation_protocol(
                        protocol, read_json(directory / "protocol.json"), transition)
                store = CheckpointStore(directory, protocol, provenance_sha256=provenance_digest)
                if (directory / "progress.json").exists():
                    state, progress = store.load()
                    bundles = [block["observations"] for block in progress["blocks"]]
                    bundles.extend(entry["state"] for entry in progress["states"])
                    if progress["status"] == "complete":
                        bundles.append(progress["outputs"])
                    for digest in {bundle.get("provenance_sha256") for bundle in bundles} - {None}:
                        validate_provenance(root, digest, protocol)
                    if progress["status"] == "complete":
                        validate_completed_progress(progress, numerics, linearized)
                        status = "complete" if progress["report"]["checks_passed"] else "failed"
                        if status == "failed":
                            atomic_json(failure_path, {"case_id": case_id, "reason": "saved_field_checks_failed",
                                                      "last_committed_step": progress["step"],
                                                      "report": progress["report"]})
                        outcomes.append({"case_id": case_id, "status": status, "reused": True})
                        continue
                else:
                    state, progress = None, None
                evolution = ReferenceEvolution(case, numerics, manifest["amplitudes"],
                                               linearized=linearized)
                if state is None:
                    state = evolution.initial_state()
                    first = {key: np.asarray(value)[None] for key, value in evolution.observe(state).items()}
                    if not all(np.isfinite(value).all() for value in first.values()):
                        raise FloatingPointError(f"Nonfinite initial observations: {case_id}")
                    if _invalid_fields(first["fields"]):
                        raise FloatingPointError(f"Nonpositive initial density or pressure: {case_id}")
                    progress = store.initialize({k: np.asarray(v) for k, v in state.items()}, first)
                history = store.histories()
                if (progress["step"] > numerics.nsteps
                        or any(value.shape[0] != progress["step"] + 1 for value in history.values())):
                    raise ValueError("Committed histories disagree with the numerical step")
                last_block_seconds = 0.0
                while progress["step"] < numerics.nsteps:
                    remaining = deadline - clock()
                    if stop_requested() or remaining <= max(1.0, 1.5 * last_block_seconds):
                        break
                    steps = min(numerics.block_steps, numerics.nsteps - progress["step"])
                    started = clock()
                    measured = time.perf_counter()
                    next_state, values = evolution.advance(state, steps)
                    # JAX dispatch is asynchronous. Synchronize before measuring
                    # transfers so kernel time is not mislabeled as checkpoint I/O.
                    for value in (*next_state.values(), *values.values()):
                        if hasattr(value, "block_until_ready"):
                            value.block_until_ready()
                    timings["advance_and_observe_seconds"] += time.perf_counter() - measured
                    measured = time.perf_counter()
                    arrays = {key: np.asarray(value) for key, value in values.items()}
                    checkpoint_state = {key: np.asarray(value) for key, value in next_state.items()}
                    timings["host_transfer_seconds"] += time.perf_counter() - measured
                    if not all(np.isfinite(value).all() for value in [*arrays.values(), *checkpoint_state.values()]):
                        atomic_json(failure_path, {"case_id": case_id, "reason": "nonfinite_evolution",
                                                  "last_committed_step": progress["step"]})
                        outcomes.append({"case_id": case_id, "status": "failed"})
                        break
                    invalid_fields = _invalid_fields(arrays["fields"])
                    if invalid_fields:
                        atomic_json(failure_path, {"case_id": case_id, "reason": "invalid_moments",
                                                  "failures": invalid_fields,
                                                  "last_committed_step": progress["step"]})
                        outcomes.append({"case_id": case_id, "status": "failed"})
                        break
                    measured = time.perf_counter()
                    progress = store.append(progress["step"] + steps, checkpoint_state, arrays)
                    timings["checkpoint_seconds"] += time.perf_counter() - measured
                    # Retain the device arrays for the next block. The NumPy
                    # checkpoint copy must not cause an unnecessary host re-upload.
                    state = next_state
                    measured_blocks += 1
                    last_block_seconds = clock() - started
                    print(json.dumps({"case_id": case_id, "step": progress["step"],
                                      "time": progress["step"] * numerics.dt,
                                      "final_time": numerics.final_time,
                                      "block_seconds": last_block_seconds}), flush=True)
                if failure_path.exists():
                    continue
                if progress["step"] == numerics.nsteps:
                    measured = time.perf_counter()
                    outputs = sampled_outputs(store.histories(), numerics)
                    report = _completion_report(outputs)
                    progress = store.finalize(outputs, report)
                    timings["finalize_seconds"] += time.perf_counter() - measured
                    if not report["checks_passed"]:
                        atomic_json(failure_path, {"case_id": case_id, "reason": "saved_field_checks_failed",
                                                  "last_committed_step": progress["step"], "report": report})
                    outcomes.append({"case_id": case_id,
                                     "status": progress["status"] if report["checks_passed"] else "failed"})
                else:
                    outcomes.append({"case_id": case_id, "status": "partial"})
                    break
        except FileExistsError as exc:
            # Only a lock collision is a normal scheduling outcome.
            if exc.filename != str(root / "locks" / f"{case_id}.lock"):
                raise
            outcomes.append({"case_id": case_id, "status": "busy"})
    stopped_on_failure = stop_on_failure and any(case["status"] == "failed" for case in outcomes)
    timings["total_seconds"] = time.perf_counter() - measured_start
    return {"outcomes": outcomes,
            "stopped": stopped_on_failure or stop_requested() or clock() >= deadline,
            "stopped_on_failure": stopped_on_failure,
            "timings": timings, "advanced_blocks": measured_blocks,
            "training_started": False}
