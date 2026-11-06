"""Transactional NumPy checkpoints for one externally locked dataset case.

``progress.json`` is the publication point: files not referenced by it are
uncommitted and never used for recovery. A caller must hold an exclusive case
lock for the entire lifetime of a store operation. This module has no JAX,
scheduler, or storage-provider dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np


_VERSION = 1
_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_BUNDLE = re.compile(r"(?:state|observations|outputs)-[0-9a-f]{32}\.npz\Z")
_TEMP = re.compile(
    r"\.(?:protocol\.json|progress\.json|(?:state|observations|outputs)-"
    r"[0-9a-f]{32}\.npz)\.[0-9a-f]{32}\.tmp\Z"
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, write) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    payload = _json_bytes(value)
    _atomic_write(path, lambda handle: handle.write(payload))


def atomic_json(path: Path, value: dict) -> None:
    """Publish JSON with a flushed temporary file and atomic replacement."""
    _atomic_json(Path(path), value)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    _atomic_write(path, lambda handle: np.savez(handle, **arrays))


def _arrays(values: dict, *, state: bool = False,
            observations: bool = False) -> dict[str, np.ndarray]:
    if not isinstance(values, dict) or not values:
        raise ValueError("An array bundle must be a nonempty dictionary")
    result = {}
    for key, value in values.items():
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError(f"Invalid array name: {key!r}")
        array = np.asarray(value)
        if array.dtype.kind not in ("buif" if state else "buifc"):
            raise ValueError(f"Unsupported dtype for {key}: {array.dtype}")
        if observations and (array.ndim == 0 or array.shape[0] == 0):
            raise ValueError(f"Observation {key} requires a nonempty sample axis")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite array: {key}")
        if state:
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.asarray(array, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"State cannot be represented in float64: {key}")
        result[key] = array
    return result


def _schema(arrays: dict[str, np.ndarray]) -> dict:
    return {key: {"shape": list(value.shape), "dtype": value.dtype.str}
            for key, value in arrays.items()}


def _same_arrays(actual: dict, expected: dict) -> bool:
    return (actual.keys() == expected.keys()
            and all(actual[k].dtype == expected[k].dtype
                    and actual[k].shape == expected[k].shape
                    and np.array_equal(actual[k], expected[k]) for k in actual))


class CheckpointStore:
    """Store full restart states and append-only observations for one case.

    State arrays are saved as float64. Observation and output arrays retain
    their numeric dtypes. Every observation key can have its own sampling
    cadence, but its trailing dimensions and dtype must remain unchanged.
    Completion records the supplied numerical report without interpreting it:
    finishing a trajectory is distinct from qualifying it for training.
    """

    def __init__(self, directory: Path, protocol: dict):
        if not isinstance(protocol, dict):
            raise ValueError("Protocol must be a dictionary")
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("Checkpoint directory must not be a symlink")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.protocol = json.loads(_json_bytes(protocol))
        self.protocol_sha256 = hashlib.sha256(_json_bytes(self.protocol)).hexdigest()
        self.progress_path = self.directory / "progress.json"
        self._validated_step: int | None = None
        for path in self.directory.iterdir():
            known = (path.name in {"protocol.json", "progress.json"}
                     or _BUNDLE.fullmatch(path.name) or _TEMP.fullmatch(path.name))
            if not known or path.is_symlink() or not path.is_file():
                raise ValueError(f"Unrecognized checkpoint entry: {path.name}")
        protocol_path = self.directory / "protocol.json"
        if protocol_path.exists():
            if _json_bytes(json.loads(protocol_path.read_text())) != _json_bytes(self.protocol):
                raise ValueError("Checkpoint protocol mismatch")
        else:
            # A crash during the first protocol write can leave only its temp.
            if any(not p.name.startswith(".protocol.json.")
                   for p in self.directory.iterdir()):
                raise ValueError("Checkpoint files exist without a protocol")
            _atomic_json(protocol_path, self.protocol)

    def _write_bundle(self, kind: str, arrays: dict) -> dict:
        path = self.directory / f"{kind}-{uuid.uuid4().hex}.npz"
        _atomic_npz(path, arrays)
        return {"file": path.name, "sha256": _file_sha256(path),
                "arrays": _schema(arrays)}

    def _read_bundle(self, record: dict) -> dict[str, np.ndarray]:
        name = record.get("file", "")
        if not isinstance(name, str) or not _BUNDLE.fullmatch(name):
            raise ValueError("Invalid checkpoint bundle path")
        path = self.directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe checkpoint bundle: {name}")
        if _file_sha256(path) != record.get("sha256"):
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
        with np.load(path, allow_pickle=False) as bundle:
            arrays = {key: bundle[key] for key in bundle.files}
        if _schema(arrays) != record.get("arrays"):
            raise ValueError(f"Checkpoint array schema mismatch: {name}")
        return _arrays(arrays)

    def read_progress(self) -> dict:
        """Read committed metadata; use ``load`` to verify its data files too."""
        protocol_path = self.directory / "protocol.json"
        if (protocol_path.is_symlink() or self.progress_path.is_symlink()
                or _json_bytes(json.loads(protocol_path.read_text())) != _json_bytes(self.protocol)):
            raise ValueError("Checkpoint protocol mismatch or unsafe metadata path")
        progress = json.loads(self.progress_path.read_text())
        if (progress.get("version") != _VERSION
                or progress.get("protocol_sha256") != self.protocol_sha256):
            raise ValueError("Checkpoint protocol/version mismatch")
        blocks, states = progress.get("blocks"), progress.get("states")
        if not isinstance(blocks, list) or not blocks or not isinstance(states, list) or not states:
            raise ValueError("Invalid checkpoint manifest")
        steps = [block.get("step") for block in blocks]
        if (any(type(step) is not int or step < 0 for step in steps)
                or steps[0] != 0 or any(a >= b for a, b in zip(steps, steps[1:]))
                or progress.get("step") != steps[-1]
                or [s.get("step") for s in states] != steps[-2:]
                or progress.get("status") not in {"running", "complete"}):
            raise ValueError("Invalid checkpoint steps/status")
        if progress["status"] == "complete" and not {"outputs", "report"} <= progress.keys():
            raise ValueError("Incomplete completion metadata")
        return progress

    def _validate(self, progress: dict, *, collect: bool = False) -> tuple[dict, list[dict]]:
        observations = []
        first = None
        for block in progress["blocks"]:
            chunk = self._read_bundle(block["observations"])
            if first is None:
                first = _schema(chunk)
            if chunk.keys() != first.keys():
                raise ValueError("Observation keys changed across checkpoints")
            for key in chunk:
                if (chunk[key].ndim == 0 or chunk[key].shape[0] == 0
                        or list(chunk[key].shape[1:]) != first[key]["shape"][1:]
                        or chunk[key].dtype.str != first[key]["dtype"]):
                    raise ValueError(f"Observation schema changed: {key}")
            if collect:
                observations.append(chunk)
        states = [self._read_bundle(entry["state"]) for entry in progress["states"]]
        for state in states:
            if any(value.dtype != np.dtype(np.float64) for value in state.values()):
                raise ValueError("Restart state must use float64")
            if _schema(state) != _schema(states[-1]):
                raise ValueError("Restart state schema changed")
        if progress["status"] == "complete":
            self._read_bundle(progress["outputs"])
            _json_bytes(progress["report"])
        return states[-1], observations

    def load(self) -> tuple[dict[str, np.ndarray], dict]:
        progress = self.read_progress()
        state, _ = self._validate(progress)
        self._validated_step = progress["step"]
        return state, progress

    def initialize(self, state: dict, observations: dict) -> dict:
        state = _arrays(state, state=True)
        observations = _arrays(observations, observations=True)
        if self.progress_path.exists():
            saved, progress = self.load()
            initial = self._read_bundle(progress["blocks"][0]["observations"])
            if progress["step"] != 0 or not _same_arrays(saved, state) or not _same_arrays(initial, observations):
                raise ValueError("Case is already initialized; use load to resume")
            return progress
        progress = {"version": _VERSION, "protocol_sha256": self.protocol_sha256,
                    "status": "running", "step": 0,
                    "states": [{"step": 0, "state": self._write_bundle("state", state)}],
                    "blocks": [{"step": 0, "observations": self._write_bundle("observations", observations)}]}
        _atomic_json(self.progress_path, progress)
        self._validated_step = 0
        return progress

    def append(self, step: int, state: dict, observations: dict) -> dict:
        # Resume/finalization perform full verification. During normal execution
        # do not reread all earlier history at every block (quadratic disk I/O).
        progress = self.read_progress()
        if self._validated_step != progress["step"]:
            _, progress = self.load()
        if progress["status"] == "complete":
            raise ValueError("Cannot append to a completed case")
        if type(step) is not int or step <= progress["step"]:
            raise ValueError("Checkpoint step must strictly increase")
        state = _arrays(state, state=True)
        observations = _arrays(observations, observations=True)
        if _schema(state) != progress["states"][-1]["state"]["arrays"]:
            raise ValueError("Restart state schema changed")
        first = progress["blocks"][0]["observations"]["arrays"]
        if observations.keys() != first.keys() or any(
                list(observations[k].shape[1:]) != first[k]["shape"][1:]
                or observations[k].dtype.str != first[k]["dtype"] for k in first):
            raise ValueError("Observation schema changed")
        all_states = progress["states"] + [{"step": step, "state": self._write_bundle("state", state)}]
        updated = dict(progress, step=step, states=all_states[-2:],
                       blocks=progress["blocks"] + [{"step": step,
                           "observations": self._write_bundle("observations", observations)}])
        _atomic_json(self.progress_path, updated)
        self._validated_step = step
        # No committed metadata references these obsolete states anymore.
        for entry in all_states[:-2]:
            (self.directory / entry["state"]["file"]).unlink(missing_ok=True)
        return updated

    def histories(self) -> dict[str, np.ndarray]:
        progress = self.read_progress()
        _, chunks = self._validate(progress, collect=True)
        return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
                for key in chunks[0]}

    def finalize(self, outputs: dict, report: dict) -> dict:
        _, progress = self.load()
        outputs = _arrays(outputs)
        if not isinstance(report, dict):
            raise ValueError("Numerical report must be a dictionary")
        report = json.loads(_json_bytes(report))
        if progress["status"] == "complete":
            if progress["report"] != report or not _same_arrays(self._read_bundle(progress["outputs"]), outputs):
                raise ValueError("Completed outputs/report differ")
            return progress
        updated = dict(progress, status="complete", report=report,
                       outputs=self._write_bundle("outputs", outputs))
        _atomic_json(self.progress_path, updated)
        return updated
