"""Train or resume the latent moment closure on the Gaussian dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import time


def resolve_microbatch_size(run: Path, requested: int | None) -> int:
    """Recover the accumulation layout before submitting or opening a run."""
    from vpml.io.serialization import read_json

    protocol = run / "training-protocol.json"
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
        from vpml.io.serialization import file_digest
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
        microbatch = resolve_microbatch_size(run, args.microbatch_size)
        data = GaussianTrainingData.open_indexed(root, cache, E10_MODEL)
        train(data, E10_MODEL, stats, run, normalization_sha256=digest,
              deadline=deadline, signal_received=signal_received, microbatch_size=microbatch,
              max_updates=args.max_updates, max_rounds=args.max_rounds,
              resume_after_plateau=args.resume_after_plateau,
              validate=make_validation_report(data, E10_MODEL, stats, batch_size=microbatch))


if __name__ == "__main__":
    main()
