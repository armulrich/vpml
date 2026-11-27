"""Recover a complete Gaussian training update together with its metric history."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import BadZipFile

from vpml.io.atomic import atomic_write, sync_directory
from vpml.io.serialization import canonical_json_bytes, file_digest, json_digest, read_json
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
    def __init__(self, run: Path, protocol_digest: str, *, archive_validations=False):
        self.run = run
        self.digest = protocol_digest
        self.latest = run / "checkpoint.npz"
        self.previous = run / "previous-checkpoint.npz"
        self.best = run / "best-checkpoint.npz"
        self.metrics = run / "training-loss.jsonl"
        self.saved_step = None
        self.saved_best_step = None
        self.previous_best_step = None
        self.archive_validations = archive_validations
        self.snapshots = run / "snapshots"
        self.verified_archive_step = None

    def import_phase(self, path, checksum, protocol, strengths):
        """Seed a separate run using the exact optimizer state and committed loss history."""
        path = Path(path).resolve()
        parent = next((d for d in path.parents if (d / "training-protocol.json").is_file()), None)
        if parent is None or self.run.resolve() == parent:
            raise ValueError("A validation phase needs a checkpoint from a separate training run")
        parent_protocol = read_json(parent / "training-protocol.json")
        validation_keys = {"validation", "validation_panel_sha256", "phase_parent_sha256"}
        if ({k: v for k, v in parent_protocol.items() if k not in validation_keys}
                != {k: v for k, v in protocol.items() if k not in validation_keys}):
            raise ValueError("Parent optimization protocol differs from the requested phase")
        # Hash and load through one descriptor even if another process rotates the parent latest.
        with path.open("rb") as source:
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != checksum:
                raise ValueError("Parent checkpoint checksum mismatch")
            source.seek(0)
            params, opt, record, _ = load_snapshot(source, json_digest(parent_protocol))
        step = record["accepted_updates"]
        windows = record["windows"]
        by_strength = record["windows_by_strength"]
        if (type(step) is not int or step < 0 or int(opt["step"]) != step
                or type(windows) is not int or not 0 <= windows <= step * protocol["batch_size"]
                or by_strength.keys() != {str(s) for s in strengths}
                or any(type(count) is not int or count < 0 for count in by_strength.values())
                or sum(by_strength.values()) != windows
                or record["validation"]["invalid_step"] is not None):
            raise ValueError("Invalid parent optimizer, sampling or validation state")

        def copy_history(dest):
            with (parent / self.metrics.name).open("rb") as history:
                for update in range(1, step + 1):
                    line = history.readline()
                    if not line.endswith(b"\n") or json.loads(line)[0] != update:
                        raise ValueError("Missing or out-of-order parent training loss history")
                    dest.write(line)
                if history.tell() != record["loss_history_bytes"]:
                    raise ValueError("Parent checkpoint disagrees with committed loss history")
        atomic_write(self.metrics, copy_history)
        return params, opt, record

    def _archive(self, record, checksum=None):
        report = record["validation"].get("last")
        if not self.archive_validations or report is None:
            return
        step = report["step"]
        if self.verified_archive_step == step:
            return
        if not self.snapshots.exists():
            self.snapshots.mkdir(exist_ok=True)
            sync_directory(self.run)
        dest = self.snapshots / f"step-{step:08d}"
        identity = dict(step=step, protocol_sha256=self.digest,
                        panel_sha256=report["panel_sha256"], validation_sha256=json_digest(report))
        if dest.exists():
            receipt = read_json(dest / "receipt.json")
            if (any(receipt.get(k) != v for k, v in identity.items())
                    or file_digest(dest / "checkpoint.npz") != receipt.get("sha256")):
                raise ValueError(f"Conflicting validation checkpoint archive: {dest}")
        else:
            if step != record["accepted_updates"]:
                raise ValueError(f"Missing validation checkpoint archive: {dest}")
            # A single rename commits weights and receipt together under the run's writer lock.
            # Hard-linking is safe: rolling checkpoints are always written by atomic replacement.
            with TemporaryDirectory(prefix=".validation-", dir=self.snapshots) as temp:
                staging = Path(temp) / "snapshot"
                staging.mkdir()
                os.link(self.latest, staging / "checkpoint.npz")
                write_json(staging / "receipt.json", dict(**identity, sha256=checksum or file_digest(self.latest)))
                staging.rename(dest)
                sync_directory(self.snapshots)
        self.verified_archive_step = step

    def _publish_best(self, step, best_step):
        if best_step is None:
            self.best.unlink(missing_ok=True)
            return
        # Retain independent links to the best weights referenced by BOTH generations.
        # A rollback can therefore restore the historical choice of best model.
        selected = self.run / f".best-at-{best_step}.npz"
        if step == best_step:
            link_snapshot(self.latest, selected)
        elif self.archive_validations and not selected.exists():
            archived = self.snapshots / f"step-{best_step:08d}"
            receipt = read_json(archived / "receipt.json")
            if (receipt.get("step") != best_step or receipt.get("protocol_sha256") != self.digest
                    or file_digest(archived / "checkpoint.npz") != receipt.get("sha256")):
                raise ValueError(f"Invalid archived best checkpoint: {archived}")
            link_snapshot(archived / "checkpoint.npz", selected)
        link_snapshot(selected, self.best)

    def restore(self):
        """Recover the last intact commit, rolling back any uncommitted metrics."""
        errors = []
        candidates = [self.latest, self.previous]
        archive_step = -1
        if self.archive_validations:
            steps = [int(d.name[5:]) for d in self.snapshots.glob("step-*") if d.name[5:].isdigit()]
            if steps:
                archive_step = max(steps)
                candidates.append(self.snapshots / f"step-{archive_step:08d}" / "checkpoint.npz")
        for path in candidates:
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
            # A completed archived check supersedes any rolling state that would replay it.
            if step < archive_step or (step == archive_step and record.get("pending_validation")):
                continue
            if path != self.latest:
                if path.parent.parent == self.snapshots:
                    if step != archive_step or record["validation"]["last"]["step"] != step:
                        raise ValueError(f"Validation archive has the wrong optimizer step: {path}")
                    self._archive(record)  # Verify digest before restoring an archived generation.
                print(json.dumps({"event": "checkpoint_recovery", "step": step,
                                  "path": str(path), "errors": errors}), flush=True)
                link_snapshot(path, self.latest)
            self.saved_step = step
            self.saved_best_step = record["validation"]["best_step"]
            self._publish_best(step, self.saved_best_step)
            self._archive(record)
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
        self._archive(record, checksum)
        if rotated:
            retained = {f".best-at-{i}.npz" for i in (self.saved_best_step, self.previous_best_step)}
            for path in self.run.glob(".best-at-*.npz"):
                if path.name not in retained:
                    path.unlink()
        print(json.dumps({"event": "checkpoint", "step": step, "sha256": checksum,
                          "path": str(self.latest)}), flush=True)
