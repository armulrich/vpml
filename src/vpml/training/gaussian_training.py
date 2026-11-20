"""Train the latent moment closure on Gaussian reference trajectories."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

import jax
import numpy as np

from vpml.io.atomic import atomic_write
from vpml.io.serialization import canonical_json_bytes, json_digest, read_json
from vpml.training.optimization import _cosine_learning_rate, adam_init
from vpml.training.wavenumber_training import (
    Configuration, initialize, load_snapshot, save_snapshot, update_factory,
)

# The historical decay was specified as 1,000 epochs of 592 optimizer updates.
E10_DECAY_UPDATES = 592_000
E10_NORMALIZATION_SHA256 = "fd00f37899503377ae23028e73a7cddb25109d5a5dbbc175997d9d7274c2adf1"
E10_MODEL = Configuration(preparation_steps=0, horizon=600, relative_loss_norm="l2",
                          learning_rate=1e-4, compute_dtype="float64")


def training_protocol(data, config, normalization_sha256, batch_size, microbatch_size):
    """Describe the update sequence independently of any particular allocation."""
    model = asdict(config)
    del model["planned_epochs"]  # The Gaussian schedule is measured in optimizer updates.
    return dict(model=model, dataset_sha256=data.manifest_sha256,
                training_outputs_sha256=data.training_outputs_sha256,
                normalization_sha256=normalization_sha256,
                sampler="strength_interleaved_shuffled_windows", batch_size=batch_size,
                microbatch_size=microbatch_size, decay_updates=E10_DECAY_UPDATES)


def log_event(event, **values):
    print(json.dumps(dict(event=event, **values), allow_nan=False), flush=True)


def write_json(path, value, *, replace=True):
    atomic_write(path, lambda f: f.write(canonical_json_bytes(value, newline=True)),
                 replace=replace)


def train(data, config, stats, run, *, normalization_sha256, batch_size=48,
          microbatch_size=4, max_updates=None, max_epochs=None, deadline=None,
          signal_received=lambda: False, validate=None):
    """Continue one training directory through its requested time or exposure limit.

    Epochs are one-based. max_updates and max_epochs are cumulative targets.
    Only the latest complete optimizer step is published to checkpoint.npz.
    """
    if not 1 <= microbatch_size <= batch_size:
        raise ValueError("Microbatch size must be between one and the effective batch size")
    if any(target is not None and target <= 0 for target in (max_updates, max_epochs)):
        raise ValueError("Training targets must be positive")
    if deadline is None and max_epochs is None and max_updates is None:
        raise ValueError("Specify a deadline, an epoch target or an update target")
    if not jax.config.x64_enabled:
        raise RuntimeError("Enable JAX float64 before starting Gaussian closure training")
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    protocol = training_protocol(data, config, normalization_sha256, batch_size, microbatch_size)
    digest = json_digest(protocol)
    with (run / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = run / "training-protocol.json"
        if protocol_path.exists():
            if read_json(protocol_path) != protocol:
                raise ValueError("Training protocol has changed; use the original configuration to resume")
        else:
            write_json(protocol_path, protocol, replace=False)

        checkpoint = run / "checkpoint.npz"
        if checkpoint.exists():
            params, optimizer, record, rows = load_snapshot(checkpoint, digest)
            if (int(optimizer["step"]) != record["accepted_updates"]
                    or len(rows) != record["accepted_updates"]):
                raise ValueError("Checkpoint optimizer or loss history disagrees with training cursor")
            last_saved = record["accepted_updates"]
        else:
            params = initialize(config)
            optimizer = adam_init(params)
            record = dict(protocol_sha256=digest, epoch=1, next_batch=0, accepted_updates=0,
                          windows=0, windows_by_strength={str(s): 0 for s in data.strengths})
            rows = []
            last_saved = -1

        def checkpoint_now():
            nonlocal last_saved
            if record["accepted_updates"] != last_saved:
                checksum = save_snapshot(checkpoint, params, optimizer, record, rows, replace=True)
                last_saved = record["accepted_updates"]
                log_event("checkpoint", step=last_saved, sha256=checksum, path=str(checkpoint))

        def stopping():
            return signal_received() or (deadline is not None and time.monotonic() >= deadline)

        checkpoint_now()
        log_event("training_start", **record)
        update = update_factory(config, stats, microbatch_size=microbatch_size)
        compiled = {}
        active_epoch = None
        last_checkpoint_time = time.monotonic()
        last_validation_time = last_checkpoint_time
        validation_seconds = None
        try:
            while (not stopping()
                   and (max_updates is None or record["accepted_updates"] < max_updates)
                   and (max_epochs is None or record["epoch"] <= max_epochs)):
                if active_epoch != record["epoch"]:
                    active_epoch = record["epoch"]
                    epoch_rows = data.epoch_rows(active_epoch, config.seed)
                    if len(epoch_rows) == 0:
                        raise ValueError("Training split has no eligible windows")
                start = record["next_batch"] * batch_size
                selected = epoch_rows[start:start + batch_size]
                if not len(selected):
                    raise ValueError("Checkpoint batch lies outside the deterministic epoch")
                began = time.monotonic()
                batch = data.batch(selected)
                loaded = time.monotonic()
                if stopping():
                    break
                step = record["accepted_updates"]
                rate = np.asarray(_cosine_learning_rate(
                    step, E10_DECAY_UPDATES, config.learning_rate, config.final_learning_rate),
                    dtype=config.compute_dtype)
                if len(selected) not in compiled:
                    log_event("compiling", step=step, batch_size=len(selected),
                              microbatch_size=microbatch_size)
                    executable = update.lower(params, optimizer, batch, rate).compile()
                    compiled[len(selected)] = executable
                    analysis = executable.memory_analysis()
                    log_event("compiled", seconds=time.monotonic()-loaded,
                              temporary_bytes=analysis.temp_size_in_bytes if analysis else None)
                if stopping():
                    break
                update_start = time.monotonic()
                p, opt, loss, grad, change, init_grad, finite = compiled[len(selected)](
                    params, optimizer, batch, rate)
                if not bool(finite):  # Synchronizes the device before committing the cursor.
                    raise FloatingPointError(f"Nonfinite candidate at optimizer update {step + 1}")
                update_seconds = time.monotonic() - update_start
                params, optimizer = p, opt
                record["accepted_updates"] = step + 1
                record["next_batch"] += 1
                record["windows"] += len(selected)
                cases = data.manifest["cases"]
                counts = Counter(str(cases[int(ci)]["strength"]) for ci, _ in selected)
                for strength, count in counts.items():
                    record["windows_by_strength"][strength] += count
                rows.append([step + 1, float(loss), float(grad), float(change), float(init_grad)])
                end_of_epoch = start + len(selected) == len(epoch_rows)
                if end_of_epoch:
                    record["epoch"] += 1
                    record["next_batch"] = 0
                metrics = dict(step=step+1, epoch=active_epoch, loss=float(loss),
                               learning_rate=float(rate), update_seconds=update_seconds,
                               load_seconds=loaded-began, windows=record["windows"],
                               windows_by_strength=record["windows_by_strength"])
                memory = jax.devices()[0].memory_stats() or {}
                if "peak_bytes_in_use" in memory:
                    metrics["peak_device_bytes"] = int(memory["peak_bytes_in_use"])
                log_event("update", **metrics)
                now = time.monotonic()
                remaining = float("inf") if deadline is None else deadline - now
                # The first panel includes its own compilation. Reserve enough
                # time for another panel plus a durable checkpoint near the limit.
                panel_budget = 60 if validation_seconds is None else 1.3 * validation_seconds + 5
                nearing_deadline = remaining <= panel_budget + max(update_seconds, 15) + 10
                hit_target = (max_updates == step + 1
                              or (max_epochs is not None and record["epoch"] > max_epochs))
                validation_due = (validate is not None and remaining > panel_budget
                                  and (step == 0 or (step + 1) % 100 == 0 or hit_target
                                       or nearing_deadline or now - last_validation_time >= 600))
                if (validation_due or end_of_epoch or step == 0
                        or now - last_checkpoint_time >= 120):
                    checkpoint_now()
                    last_checkpoint_time = time.monotonic()
                if validation_due and not stopping():
                    validation_start = time.monotonic()
                    result = dict(step=step + 1, **validate(params))
                    last_validation_time = time.monotonic()
                    validation_seconds = last_validation_time - validation_start
                    result["seconds"] = validation_seconds
                    write_json(run / "latest-validation.json", result)
                    log_event("validation", **result)
                    if nearing_deadline:
                        break
        finally:
            checkpoint_now()
        log_event("training_stop", **record)
        return record
