"""Train the latent moment closure on deterministic sampled Gaussian windows."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path
import time

import jax
import numpy as np

from vpml.datasets.gaussian_training import SAMPLED_ROUND_UPDATES
from vpml.io.serialization import canonical_json_bytes, json_digest, read_json
from vpml.training.gaussian_checkpoints import TrainingCheckpoints, write_json
from vpml.training.optimization import _cosine_learning_rate, adam_init
from vpml.training.wavenumber_training import Configuration, initialize, update_factory

# The historical decay was specified as 1,000 epochs of 592 optimizer updates.
E10_DECAY_UPDATES = 592_000
E10_NORMALIZATION_SHA256 = "fd00f37899503377ae23028e73a7cddb25109d5a5dbbc175997d9d7274c2adf1"
E10_MODEL = Configuration(preparation_steps=0, horizon=600, relative_loss_norm="l2",
                          learning_rate=1e-4, compute_dtype="float64")

VALIDATION_START = 50
VALIDATION_INTERVAL = 25
PLATEAU_RELATIVE_GAIN = 0.01
PLATEAU_PATIENCE = 3
CHECKPOINT_SECONDS = 300


def training_protocol(data, config, normalization_sha256, batch_size, microbatch_size,
                      panel_sha256=None):
    """Describe the update and model-selection sequence across allocations."""
    model = asdict(config)
    del model["planned_epochs"]
    return dict(model=model, dataset_sha256=data.manifest_sha256,
                training_outputs_sha256=data.training_outputs_sha256,
                normalization_sha256=normalization_sha256,
                sampler="strength_balanced_sampled_rounds", updates_per_round=SAMPLED_ROUND_UPDATES,
                batch_size=batch_size, microbatch_size=microbatch_size,
                decay_updates=E10_DECAY_UPDATES, validation_panel_sha256=panel_sha256,
                validation=dict(first_steps=[1, 28], start=VALIDATION_START,
                                interval=VALIDATION_INTERVAL, patience=PLATEAU_PATIENCE,
                                relative_gain=PLATEAU_RELATIVE_GAIN))


def log_event(event, **values):
    print(json.dumps(dict(event=event, **values), allow_nan=False), flush=True)


def scheduled_check(step):
    return step >= VALIDATION_START and (step - VALIDATION_START) % VALIDATION_INTERVAL == 0


def record_validation(state, report):
    """Track the true best across checks; count patience only on scheduled checks."""
    step = report["step"]
    value = report["mean_relative_l2"]
    state["last"] = report
    if value is None or not math.isfinite(value):
        state["invalid_step"] = step
        return
    if state["best_score"] is None or value < state["best_score"]:
        state.update(best_score=value, best_step=step)
    if not scheduled_check(step):
        return
    reference = state["significant_best"]
    if reference is None or (value < reference and value <= (1 - PLATEAU_RELATIVE_GAIN) * reference):
        state.update(significant_best=value, checks_without_gain=0)
    else:
        state["checks_without_gain"] += 1


class PrefetchedBatches:
    """One host producer, at most one batch ahead of the committed optimizer."""
    def __init__(self, data, seed, batch_size, executor):
        self.data = data
        self.seed = seed
        self.size = batch_size
        self.executor = executor
        self.future = None
        self.step = None

    def _read(self, step):
        started = time.monotonic()
        rows = self.data.sample_rows(step, self.seed, self.size)
        batch = self.data.batch(rows)
        cases = self.data.manifest["cases"]
        counts = Counter(str(cases[int(case)]["strength"]) for case, _ in rows)
        return batch, counts, time.monotonic() - started

    def prepare(self, step):
        if self.future is None:
            self.future = self.executor.submit(self._read, step)
            self.step = step
        elif self.step != step:
            raise RuntimeError("Prefetch and optimizer positions disagree")

    def take(self, step):
        self.prepare(step)
        result = self.future.result()
        self.future = None
        return result

    def drain(self):
        # Validation reads through the same corpus cache on the main thread.
        if self.future is not None:
            self.future.result()


def _new_record(digest, strengths):
    return dict(protocol_sha256=digest, accepted_updates=0, windows=0,
                windows_by_strength={str(s): 0 for s in strengths},
                loss_history_bytes=0, pending_validation=False, stop_reason=None,
                validation=dict(last=None, best_score=None, best_step=None, invalid_step=None,
                                significant_best=None, checks_without_gain=0))


def _publish_validation(run, report):
    if report is None:
        return
    directory = run / "validation"
    directory.mkdir(exist_ok=True)
    write_json(directory / f"step-{report['step']:08d}.json", report)
    write_json(run / "latest-validation.json", report)


def train(data, config, stats, run, *, normalization_sha256, batch_size=48,
          microbatch_size=4, max_updates=None, max_rounds=None, deadline=None,
          signal_received=lambda: False, validate=None, resume_after_plateau=False):
    """Continue up to a cumulative step/round cap or an allocation deadline."""
    if not 1 <= microbatch_size <= batch_size:
        raise ValueError("Microbatch size must be between one and the effective batch size")
    if any(target is not None and target <= 0 for target in (max_updates, max_rounds)):
        raise ValueError("Training targets must be positive")
    if deadline is None and max_rounds is None and max_updates is None:
        raise ValueError("Specify a deadline, a round target or an update target")
    if not jax.config.x64_enabled:
        raise RuntimeError("Enable JAX float64 before starting Gaussian closure training")
    limit = min(max_updates or math.inf, (max_rounds or math.inf) * data.round_updates(batch_size))
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    protocol = training_protocol(data, config, normalization_sha256, batch_size, microbatch_size,
                                 getattr(validate, "panel_sha256", None))
    digest = json_digest(protocol)
    with (run / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = run / "training-protocol.json"
        if protocol_path.exists():
            if read_json(protocol_path) != protocol:
                raise ValueError("Training protocol mismatch: resume with the original code and "
                                 "configuration or start a fresh run directory for the new sampler")
        else:
            write_json(protocol_path, protocol, replace=False)
        checkpoints = TrainingCheckpoints(run, digest)
        restored = checkpoints.restore()
        if restored is None:
            params = initialize(config)
            optimizer = adam_init(params)
            record = _new_record(digest, data.strengths)
        else:
            params, optimizer, record = restored
        state = record["validation"]
        if state["invalid_step"] is not None:
            raise ValueError(f"Validation was nonfinite at update {state['invalid_step']}; inspect the checkpoint")
        if state["checks_without_gain"] >= PLATEAU_PATIENCE:
            if not resume_after_plateau:
                raise ValueError("Training paused at a validation plateau; use --resume-after-plateau "
                                 "to grant another observation period")
            state["checks_without_gain"] = 0
        _publish_validation(run, state["last"])
        with checkpoints.open_history(record) as history, ThreadPoolExecutor(max_workers=1) as reader:
            return _run_updates(data, config, stats, batch_size, microbatch_size, limit, deadline,
                                signal_received, validate, params, optimizer, record,
                                checkpoints, history, reader)


def _run_updates(data, config, stats, batch_size, microbatch_size, limit, deadline,
                 signal_received, validate, params, optimizer, record, checkpoints, history, reader):
    batches = PrefetchedBatches(data, config.seed, batch_size, reader)
    make_update = update_factory(config, stats, microbatch_size=microbatch_size)
    compiled = {}
    steps_per_round = data.round_updates(batch_size)
    state = record["validation"]
    last_save = time.monotonic()
    last_report = state["last"]
    panel_seconds = 180 if last_report is None else last_report["seconds"]
    step_seconds = 30
    record["stop_reason"] = None
    step = record["accepted_updates"]

    def remaining():
        return math.inf if deadline is None else deadline - time.monotonic()

    def persist():
        nonlocal last_save
        checkpoints.save(params, optimizer, record, history)
        last_save = time.monotonic()

    def evaluate():
        nonlocal panel_seconds
        batches.drain()
        if signal_received() or remaining() <= 1.3 * panel_seconds + 5:
            return False
        began = time.monotonic()
        report = dict(step=step, **validate(params))
        panel_seconds = report["seconds"] = time.monotonic() - began
        record_validation(state, report)
        record["pending_validation"] = False
        persist()
        _publish_validation(checkpoints.run, report)
        log_event("validation", **report)
        return True

    if checkpoints.saved_step is None:
        persist()
    log_event("training_start", accepted_updates=step, windows=record["windows"],
              windows_by_strength=record["windows_by_strength"])
    reason = "update_limit"
    finish_after_validation = False
    try:
        while True:
            if signal_received() or remaining() <= 0:
                reason = "signal" if signal_received() else "deadline"
                break
            if record["pending_validation"]:
                if validate is None:
                    raise ValueError("Resume requires the validation evaluator for the pending check")
                if not evaluate():
                    reason = "signal" if signal_received() else "deadline"
                    break
                if state["invalid_step"] is not None:
                    reason = "nonfinite_validation"
                    break
                if state["checks_without_gain"] >= PLATEAU_PATIENCE:
                    reason = "validation_plateau"
                    break
                if finish_after_validation:
                    reason = "deadline"
                    break
            if step >= limit:
                break
            if remaining() <= step_seconds + 5:
                reason = "deadline"
                break
            wait_at = time.monotonic()
            batch, counts, preparation = batches.take(step)
            wait = time.monotonic() - wait_at
            if signal_received() or remaining() <= step_seconds + 5:
                reason = "signal" if signal_received() else "deadline"
                break
            count = len(batch["initial"])
            if count not in compiled:
                began = time.monotonic()
                log_event("compiling", step=step, batch_size=count, microbatch_size=microbatch_size)
                rate = np.asarray(config.learning_rate, dtype=config.compute_dtype)
                executable = make_update.lower(params, optimizer, batch, rate).compile()
                compiled[count] = executable
                analysis = executable.memory_analysis()
                log_event("compiled", seconds=time.monotonic()-began,
                          temporary_bytes=analysis.temp_size_in_bytes if analysis else None)
            if signal_received() or remaining() <= step_seconds + 5:
                reason = "signal" if signal_received() else "deadline"
                break
            next_step = step + 1
            periodic = next_step in (1, 28) or scheduled_check(next_step)
            if next_step < limit and not (periodic and validate is not None):
                batches.prepare(next_step)
            rate = np.asarray(_cosine_learning_rate(step, E10_DECAY_UPDATES, config.learning_rate,
                                                   config.final_learning_rate), dtype=config.compute_dtype)
            began = time.monotonic()
            p, opt, loss, grad, change, init_grad, finite = compiled[count](params, optimizer, batch, rate)
            if not bool(finite):
                raise FloatingPointError(f"Nonfinite candidate at optimizer update {next_step}")
            step_seconds = time.monotonic() - began
            # Convert device values before advancing any part of the committed state.
            row = [next_step, float(loss), float(grad), float(change), float(init_grad)]
            history.write(canonical_json_bytes(row, newline=True))
            params, optimizer, step = p, opt, next_step
            record["accepted_updates"] = step
            record["windows"] += count
            for strength, strength_count in counts.items():
                record["windows_by_strength"][strength] += strength_count
            metrics = dict(step=step, sampled_round=1 + (step-1)//steps_per_round,
                           loss=row[1], learning_rate=float(rate), update_seconds=step_seconds,
                           load_seconds=preparation, input_wait_seconds=wait, windows=record["windows"],
                           windows_by_strength=record["windows_by_strength"])
            memory = jax.devices()[0].memory_stats() or {}
            if "peak_bytes_in_use" in memory:
                metrics["peak_device_bytes"] = int(memory["peak_bytes_in_use"])
            log_event("update", **metrics)
            panel_budget = 1.3 * panel_seconds + 5
            near_deadline = remaining() <= panel_budget + max(step_seconds, 15) + wait + 15
            if validate is not None and (periodic or step >= limit or near_deadline):
                # A pending check is durable before it consumes time. A resumed job
                # can finish it using the identical weights before another update.
                record["pending_validation"] = periodic or remaining() > panel_budget
                if record["pending_validation"]:
                    finish_after_validation = near_deadline
                    persist()
            if time.monotonic() - last_save >= CHECKPOINT_SECONDS:
                persist()
    except BaseException:
        reason = "error"
        raise
    finally:
        record["stop_reason"] = reason
        persist()
        log_event("training_stop", **record)
    return record
