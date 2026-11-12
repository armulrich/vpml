#!/usr/bin/env python3
"""Freeze a local submission intent without contacting Slurm or loading JAX.

The record describes the requested job, not scheduler acceptance or completion.
Source and data paths must remain immutable while a submitted job can use them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def git_identity(root: Path) -> dict:
    """Distinguish observed Git identity from an optional archive declaration."""
    declared = os.environ.get("VPML_SOURCE_COMMIT")
    if declared is not None and re.fullmatch(r"[0-9a-fA-F]{40}", declared) is None:
        raise ValueError("VPML_SOURCE_COMMIT must contain exactly 40 hexadecimal characters")
    declared = declared.lower() if declared else None
    commit = dirty = None
    try:
        def git(*arguments):
            return subprocess.run(
                ["git", "-C", str(root), *arguments], check=True,
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()

        if Path(git("rev-parse", "--show-toplevel")).resolve() == root.resolve():
            commit = git("rev-parse", "HEAD")
            dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError):
        pass  # A git archive has no .git; the source hashes remain authoritative.
    if commit is not None and declared is not None and commit != declared:
        raise ValueError("VPML_SOURCE_COMMIT does not match the checkout's Git commit")
    return {"git_commit": commit, "git_dirty": dirty, "declared_source_commit": declared}


def snapshot(path: Path) -> dict:
    payload = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
            "text": payload.decode("utf-8")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code_root", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("python", type=Path)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("run_parameters", type=Path)
    parser.add_argument("worker_seconds", type=int)
    parser.add_argument("wall_seconds", type=int)
    parser.add_argument("shutdown_margin_seconds", type=int)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    split = args.arguments.index("--")
    scheduler_arguments, worker_arguments = args.arguments[:split], args.arguments[split + 1:]
    selected = []
    stop_on_failure = False
    index = 0
    while index < len(worker_arguments):
        option = worker_arguments[index]
        if option == "--case-id":
            selected.append(worker_arguments[index + 1])
            index += 2
        elif option == "--stop-on-failure":
            stop_on_failure = True
            index += 1
        else:
            raise ValueError(f"Unsupported worker option: {option}")

    # Explicitly inspect this checkout; these utilities import no JAX modules.
    sys.path.insert(0, str(args.code_root / "src"))
    from vpml.datasets.provenance import source_record
    from vpml.io.atomic import atomic_write
    from vpml.io.serialization import canonical_json_bytes

    manifest_path = args.data_root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not selected:
        selected = [case["case_id"] for case in manifest["cases"]]
    launcher_directory = Path(__file__).resolve().parent
    launcher_sources = {
        name: snapshot(launcher_directory / name)["sha256"]
        for name in ("run_gaussian_amplitudes.sh", "gaussian_amplitudes.batch.sh",
                     "record_submission.py")
    }
    record = {
        "schema_version": 1,
        "record_type": "submission_intent",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(manifest_path),
                     "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "scientific_parameters": snapshot(args.parameters),
        "launcher_parameters": snapshot(args.run_parameters),
        "code": {"root": str(args.code_root), **git_identity(args.code_root),
                 "source": source_record(args.code_root / "src/vpml"),
                 "launcher_sha256": launcher_sources},
        "selected_case_ids": selected,
        "scheduler": {"arguments": scheduler_arguments, "wall_seconds": args.wall_seconds,
                      "shutdown_margin_seconds": args.shutdown_margin_seconds},
        "worker": {"python": str(args.python), "arguments": worker_arguments,
                   "max_seconds": args.worker_seconds, "stop_on_failure": stop_on_failure},
    }
    payload = canonical_json_bytes(record, newline=True)
    directory = args.data_root / "submissions"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = directory / f"{hashlib.sha256(payload).hexdigest()}.json"
    def write_record(handle):
        handle.write(payload)
        os.fchmod(handle.fileno(), 0o400)

    # Publish only the fully flushed read-only record, sync its directory and
    # refuse to replace an earlier submission's evidence.
    atomic_write(destination, write_record, replace=False)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
