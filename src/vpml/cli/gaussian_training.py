"""Train or resume the latent moment closure on the Gaussian dataset."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
import time

from vpml.io.serialization import file_digest, json_digest, read_json


@dataclass(frozen=True)
class ValidationPhase:
    parent: Path
    checkpoint_sha256: str
    panel_path: Path
    panel: dict
    panel_sha256: str
    baseline_loss: float


def add_phase_arguments(parser):
    group = parser.add_argument_group("continuation with a new validation panel")
    group.add_argument("--phase-from", type=Path, help="Parent training run with the source checkpoint")
    group.add_argument("--parent-checkpoint-sha256", help="Expected SHA-256 of the parent's checkpoint.npz")
    group.add_argument("--validation-panel", type=Path, help="JSON manifest of the fixed held-out windows")
    group.add_argument("--validation-panel-sha256", help="Optional SHA-256 of the canonical panel JSON")
    group.add_argument("--baseline-loss", type=float, help="Reference mean loss of the parent on this panel")


def validation_phase(args, run: Path) -> ValidationPhase | None:
    required = (args.phase_from, args.parent_checkpoint_sha256, args.validation_panel, args.baseline_loss)
    if all(value is None for value in required) and args.validation_panel_sha256 is None:
        return None
    if any(value is None for value in required):
        raise ValueError("A validation phase requires --phase-from, --parent-checkpoint-sha256, "
                         "--validation-panel and --baseline-loss")
    if args.resume_after_plateau:
        raise ValueError("--resume-after-plateau applies to the original validation protocol")
    parent = args.phase_from.resolve()
    if run.resolve() == parent:
        raise ValueError("Select a separate run directory for the new validation phase")
    digest = args.parent_checkpoint_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid parent checkpoint SHA-256")
    if not math.isfinite(args.baseline_loss) or args.baseline_loss < 0:
        raise ValueError("--baseline-loss must be finite and nonnegative")
    if not (run / "checkpoint.npz").is_file():
        if file_digest(parent / "checkpoint.npz") != digest:
            raise ValueError("Parent checkpoint SHA-256 mismatch")
    panel_path = args.validation_panel.resolve()
    panel = read_json(panel_path)
    panel_sha256 = json_digest(panel)
    if args.validation_panel_sha256 is not None and args.validation_panel_sha256.lower() != panel_sha256:
        raise ValueError("Validation panel SHA-256 mismatch")
    return ValidationPhase(parent, digest, panel_path, panel, panel_sha256, args.baseline_loss)


def resolve_microbatch_size(run: Path, requested: int | None, *, parent: Path | None = None) -> int:
    """Recover the accumulation layout before submitting or opening a run."""
    protocol = run / "training-protocol.json"
    if parent is not None and not protocol.is_file():
        protocol = parent / "training-protocol.json"
        if not protocol.is_file():
            raise ValueError("Parent training protocol is missing")
    if protocol.is_file():
        saved_protocol = read_json(protocol)
        if saved_protocol.get("sampler") == "strength_interleaved_shuffled_windows":
            raise ValueError("This full-epoch run requires its original source revision to resume. "
                             "Choose a new run directory for sampled rounds.")
        saved = saved_protocol["microbatch_size"]
        if requested is not None and requested != saved:
            raise ValueError(f"The existing run uses microbatch size {saved}")
        size = saved
    else:
        size = 12 if requested is None else requested
    if type(size) is not int or not 1 <= size <= 48:
        raise ValueError("Microbatch size must be between 1 and 48")
    return size


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True,
                        help="Frozen E10 normalization.npz")
    parser.add_argument("--cache", type=Path, help="Reusable conservative moment arrays")
    parser.add_argument("--worker-seconds", type=int, help="Wall time including setup and compilation")
    parser.add_argument("--microbatch-size", type=int,
                        help="Windows processed together (default 12, or the saved run setting)")
    parser.add_argument("--max-updates", type=int, help="Cumulative optimizer update target")
    parser.add_argument("--max-rounds", type=int, help="Cumulative limit in 25-update sampling rounds")
    parser.add_argument("--resume-after-plateau", action="store_true",
                        help="Continue a stopped run with a new validation observation period")
    parser.add_argument("--require-gpu", action="store_true")
    add_phase_arguments(parser)
    args = parser.parse_args(argv)
    if args.worker_seconds is not None and args.worker_seconds <= 0:
        parser.error("--worker-seconds must be positive")
    deadline = None if args.worker_seconds is None else time.monotonic() + args.worker_seconds

    from vpml.datasets.runner import stop_signals
    with stop_signals() as signal_received:
        import jax
        import numpy as np
        from vpml.datasets.gaussian_training import GaussianTrainingData
        from vpml.datasets.provenance import runtime_record, source_record
        from vpml.training.gaussian_training import (
            E10_MODEL, E10_NORMALIZATION_SHA256, log_event, train,
        )
        from vpml.training.gaussian_validation import make_validation_report

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        device = "gpu" if args.require_gpu else jax.default_backend()
        actual = runtime_record(device)
        log_event("runtime", **actual, source_sha256=source_record()["sha256"])
        digest = file_digest(args.normalization)
        if digest != E10_NORMALIZATION_SHA256:
            raise ValueError("Expected the frozen E10 normalization artifact")
        with np.load(args.normalization, allow_pickle=False) as archive:
            stats = {key: archive[key] for key in archive.files}
        root = args.dataset.resolve()
        run = args.run.resolve()
        cache = (args.cache or run / "cache").resolve()
        if run == root or cache == root or root in run.parents or root in cache.parents:
            raise ValueError("Training and cache directories must be outside the reference dataset")
        phase = validation_phase(args, run)
        microbatch = resolve_microbatch_size(run, args.microbatch_size, parent=phase.parent if phase else None)
        data = GaussianTrainingData.open_indexed(root, cache, E10_MODEL)
        validation = make_validation_report(data, E10_MODEL, stats, batch_size=microbatch,
                                            panel=phase.panel if phase else None)
        continuation = (dict(phase_checkpoint=phase.parent / "checkpoint.npz",
                             phase_checkpoint_sha256=phase.checkpoint_sha256,
                             validation_interval=50, baseline_loss=phase.baseline_loss)
                        if phase else {})
        train(data, E10_MODEL, stats, run, normalization_sha256=digest,
              deadline=deadline, signal_received=signal_received, microbatch_size=microbatch,
              max_updates=args.max_updates, max_rounds=args.max_rounds,
              resume_after_plateau=args.resume_after_plateau,
              validate=validation, **continuation)


if __name__ == "__main__":
    main()
