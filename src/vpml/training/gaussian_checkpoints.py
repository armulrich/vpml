"""Recover a complete Gaussian training update together with its metric history."""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile

from vpml.io.atomic import atomic_write, sync_directory
from vpml.io.serialization import canonical_json_bytes
from vpml.training.wavenumber_training import load_snapshot, save_snapshot


def write_json(path, value, *, replace=True):
    atomic_write(path, lambda f: f.write(canonical_json_bytes(value, newline=True)),
                 replace=replace)


def link_snapshot(source, destination):
    """Atomically retain an existing checkpoint without copying the weights."""
    if destination.exists() and os.path.samefile(source, destination):
        return
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
        sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


class TrainingCheckpoints:
    def __init__(self, run: Path, protocol_digest: str):
        self.run = run
        self.digest = protocol_digest
        self.latest = run / "checkpoint.npz"
        self.previous = run / "previous-checkpoint.npz"
        self.best = run / "best-checkpoint.npz"
        self.metrics = run / "training-loss.jsonl"
        self.saved_step = None
        self.saved_best_step = None
        self.previous_best_step = None

    def _publish_best(self, step, best_step):
        if best_step is None:
            self.best.unlink(missing_ok=True)
            return
        # Retain independent links to the best weights referenced by BOTH generations.
        # A rollback can therefore restore the historical choice of best model.
        selected = self.run / f".best-at-{best_step}.npz"
        if step == best_step:
            link_snapshot(self.latest, selected)
        link_snapshot(selected, self.best)

    def restore(self):
        """Recover the last intact commit, rolling back any uncommitted metrics."""
        errors = []
        for path in (self.latest, self.previous):
            if not path.exists():
                continue
            try:
                params, optimizer, record, _ = load_snapshot(path, self.digest)
                step = record["accepted_updates"]
                if int(optimizer["step"]) != step:
                    raise ValueError("Checkpoint optimizer and sampling cursor disagree")
                size = self.metrics.stat().st_size if self.metrics.exists() else 0
                if not 0 <= record["loss_history_bytes"] <= size:
                    raise ValueError("Checkpoint has missing training loss history")
            except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            if path == self.previous:
                print(json.dumps({"event": "checkpoint_recovery", "step": step,
                                  "path": str(path), "errors": errors}), flush=True)
                link_snapshot(self.previous, self.latest)
            self.saved_step = step
            self.saved_best_step = record["validation"]["best_step"]
            self._publish_best(step, self.saved_best_step)
            return params, optimizer, record

        if errors:
            raise ValueError("No recoverable training checkpoint: " + "; ".join(errors))
        if self.metrics.exists() and self.metrics.stat().st_size:
            raise ValueError("A training history exists without a recoverable checkpoint")
        return None

    def open_history(self, record):
        """Only updates acknowledged by the restored checkpoint are authoritative."""
        handle = self.metrics.open("ab")
        try:
            handle.truncate(record["loss_history_bytes"])
            handle.seek(0, os.SEEK_END)
        except BaseException:
            handle.close()
            raise
        return handle

    def save(self, params, optimizer, record, history):
        history.flush()
        os.fsync(history.fileno())
        record["loss_history_bytes"] = history.tell()
        step = record["accepted_updates"]
        rotated = self.saved_step is not None and self.saved_step != step
        if rotated:
            link_snapshot(self.latest, self.previous)
            self.previous_best_step = self.saved_best_step
        checksum = save_snapshot(self.latest, params, optimizer, record, [], replace=True)
        self.saved_step = step
        self.saved_best_step = record["validation"]["best_step"]
        self._publish_best(step, self.saved_best_step)
        if rotated:
            retained = {f".best-at-{i}.npz" for i in (self.saved_best_step, self.previous_best_step)}
            for path in self.run.glob(".best-at-*.npz"):
                if path.name not in retained:
                    path.unlink()
        print(json.dumps({"event": "checkpoint", "step": step, "sha256": checksum,
                          "path": str(self.latest)}), flush=True)
