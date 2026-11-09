"""Resumable CPU comparisons of one frozen Gaussian initial condition.

Finite numerical failures remain recorded while a diagnostic trajectory reaches
its prescribed end time. These stores are deliberately separate from training
datasets; neither completion nor a pairwise comparison qualifies training data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time
from typing import Callable

import numpy as np

from vpml.datasets.checkpoints import CheckpointStore, atomic_json
from vpml.datasets.diagnostics import DiagnosticTolerances, qualification_report
from vpml.datasets.evolution import ReferenceEvolution
from vpml.datasets.manifest import Numerics, validate_manifest
from vpml.datasets.provenance import runtime_record, source_record
from vpml.datasets.runner import case_lock
from vpml.datasets.schema import DIAGNOSTIC_NAMES, OUTPUT_SCHEMA, observation_schema
from vpml.io.atomic import atomic_write
from vpml.io.serialization import canonical_json_bytes, file_digest, json_digest, read_json

_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_CONFIG_KEYS = {"schema_version", "case", "amplitudes", "baseline", "variants", "comparisons"}


def _observation_schema() -> dict:
    return dict(observation_schema(), temporal_order="Exact subsampling at integration-step times; no time interpolation")


def _cadence(numerics: Numerics) -> int:
    ratio = numerics.save_dt / numerics.dt
    if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
        raise ValueError("Refinement save_dt must be an integer multiple of dt")
    if numerics.storage_dtype != "float64":
        raise ValueError("Refinement observations require float64 storage")
    return round(ratio)


def validate_config(config: dict) -> dict:
    """Validate explicit variants/pairs and return an independent JSON copy."""
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError("Unexpected refinement configuration keys")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported refinement schema_version")
    variants = config["variants"]
    if (not isinstance(variants, dict) or not variants
            or any(not isinstance(name, str) or not _NAME.fullmatch(name) for name in variants)):
        raise ValueError("Variants require safe, unique names")
    normalized = json.loads(canonical_json_bytes(config))
    for name, settings in variants.items():
        manifest = {"schema_version": 1, "output_schema": OUTPUT_SCHEMA,
                    "numerics": settings, "amplitudes": config["amplitudes"],
                    "cases": [config["case"]]}
        validate_manifest(manifest)
        numerics = Numerics(**settings)
        _cadence(numerics)
        normalized["variants"][name] = numerics.to_dict()
    baseline = config["baseline"]
    if not isinstance(baseline, str) or baseline not in variants:
        raise ValueError("Unknown baseline variant")
    shared = ("save_dt", "output_nx", "final_time", "vmin", "vmax")
    for settings in normalized["variants"].values():
        if any(settings[key] != normalized["variants"][baseline][key] for key in shared):
            raise ValueError("Variants must share observation times, output grid, final time and velocity domain")
    pairs = config["comparisons"]
    if (not isinstance(pairs, list) or not pairs
            or any(not isinstance(pair, list) or len(pair) != 2
                   or pair[0] != baseline or pair[1] not in variants or pair[0] == pair[1]
                   for pair in pairs)
            or len({tuple(pair) for pair in pairs}) != len(pairs)
            or {pair[1] for pair in pairs} != set(variants) - {baseline}):
        raise ValueError("Comparisons must pair the baseline explicitly with every other variant once")
    return normalized


def _publish_config(root: Path, config: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("Refinement output must be separate from a training dataset")
    with case_lock(root / "locks" / "configuration.lock"):
        path = root / "config.json"
        if not path.exists():
            atomic_write(path, lambda handle: handle.write(canonical_json_bytes(config, newline=True)),
                         replace=False)
        if read_json(path) != config:
            raise ValueError("Output directory belongs to a different refinement configuration")


def _report(diagnostics: np.ndarray, numerics: Numerics) -> dict:
    report = qualification_report(diagnostics)
    report["qualification_scope"] = "diagnostic refinement calculation; excluded from training"
    limits = DiagnosticTolerances()
    columns = dict(zip(DIAGNOSTIC_NAMES, diagnostics.T, strict=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        failures = {
            "relative_mass": np.abs(columns["mass"] / columns["mass"][0] - 1) > limits.relative_mass,
            "relative_energy": np.abs(columns["total_energy"] / columns["total_energy"][0] - 1) > limits.relative_energy,
            "minimum_f": columns["min_f"] < limits.minimum_f,
            "negative_mass_fraction": columns["negative_mass_fraction"] > limits.negative_mass_fraction,
            "boundary_abs_fraction": columns["boundary_abs_fraction"] > limits.boundary_abs_fraction,
            "charge_imbalance": np.abs(columns["charge_imbalance"]) > limits.charge_imbalance,
            "nonpositive_density": columns["min_n"] <= 0,
        }
    first = {name: float(np.flatnonzero(mask)[0] * numerics.dt)
             for name, mask in failures.items() if np.any(mask)}
    report["first_failure_times"] = first
    report["first_failure_time"] = min(first.values(), default=None)
    report["first_minimum_f_failure_time"] = first.get("minimum_f")
    report["diagnostic_names"] = list(DIAGNOSTIC_NAMES)
    return report


def _history(store: CheckpointStore, numerics: Numerics, step: int) -> dict:
    history = store.histories()
    expected = {"fields": (step + 1, 5, numerics.output_nx),
                "raw_moments": (step + 1, 3, numerics.output_nx),
                "electric": (step + 1, numerics.output_nx),
                "diagnostics": (step + 1, len(DIAGNOSTIC_NAMES))}
    if step > numerics.nsteps or set(history) != set(expected) or any(
            history[name].shape != shape or history[name].dtype != np.dtype("float64")
            for name, shape in expected.items()):
        raise ValueError("Diagnostic checkpoint history disagrees with its numerical protocol")
    return history


def _sample(history: dict, numerics: Numerics) -> dict:
    stride = _cadence(numerics)
    count = len(history["diagnostics"])
    selected = np.arange(0, count, stride)
    return {"times": np.arange(len(selected), dtype=np.float64) * numerics.save_dt,
            "diagnostic_times": np.arange(count, dtype=np.float64) * numerics.dt,
            "diagnostics": history["diagnostics"],
            **{name: history[name][selected] for name in ("fields", "raw_moments", "electric")}}


def run_variant(
    config: dict, variant_name: str, output_root: Path, *, max_seconds: float,
    stop_requested: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Continue one bounded diagnostic trajectory, preserving numerical failures.

    This function never acquires cluster resources. Only finite states commit;
    interruption retains the last complete block for exact continuation.
    """
    config = validate_config(config)
    if variant_name not in config["variants"]:
        raise ValueError("Unknown refinement variant")
    if isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    deadline = clock() + max_seconds
    numerics = Numerics(**config["variants"][variant_name])
    runtime = runtime_record("cpu")
    source = source_record()
    root = Path(output_root).resolve()
    _publish_config(root, config)
    provenance = {"source": source, "runtime": runtime,
                  "diagnostic_runner_sha256": file_digest(Path(__file__))}
    digest = json_digest(provenance)
    provenance_path = root / "provenance" / f"{digest}.json"
    provenance_path.parent.mkdir(exist_ok=True)
    if not provenance_path.exists():
        try:
            atomic_write(provenance_path, lambda handle: handle.write(
                canonical_json_bytes(provenance, newline=True)), replace=False)
        except FileExistsError:
            pass
    if read_json(provenance_path) != provenance:
        raise ValueError("Refinement provenance mismatch")
    protocol = {"schema": "gaussian-refinement-v1", "config_sha256": json_digest(config),
                "variant": variant_name, "case": config["case"], "amplitudes": config["amplitudes"],
                "numerics": numerics.to_dict(), "runtime": runtime,
                "source": source["compatibility"],
                "diagnostic_runner_sha256": provenance["diagnostic_runner_sha256"],
                "observations": _observation_schema(), "training_qualified": False}
    with case_lock(root / "locks" / f"{variant_name}.lock"):
        directory = root / "variants" / variant_name
        store = CheckpointStore(directory, protocol, provenance_sha256=digest)
        if store.progress_path.exists():
            state, progress = store.load()
            history = _history(store, numerics, progress["step"])
            if progress["status"] == "complete":
                result = read_variant(root, variant_name)
                return {"variant": variant_name, "status": result["status"],
                        "step": progress["step"], "report": result["report"], "reused": True}
        else:
            state, progress = None, None
        evolution = ReferenceEvolution(config["case"], numerics, config["amplitudes"])
        if state is None:
            state = evolution.initial_state()
            first = {key: np.asarray(value)[None] for key, value in evolution.observe(state).items()}
            progress = store.initialize({key: np.asarray(value) for key, value in state.items()}, first)
            history = first
        diagnostic_chunks = [history["diagnostics"]]
        last_block_seconds = 0.0
        reason = None
        while progress["step"] < numerics.nsteps:
            if stop_requested() or deadline - clock() <= max(1.0, 1.5 * last_block_seconds):
                reason = "stop_requested" if stop_requested() else "deadline"
                break
            steps = min(numerics.block_steps, numerics.nsteps - progress["step"])
            started = clock()
            next_state, values = evolution.advance(state, steps)
            arrays = {key: np.asarray(value) for key, value in values.items()}
            next_state = {key: np.asarray(value) for key, value in next_state.items()}
            if not all(np.isfinite(value).all() for value in [*arrays.values(), *next_state.values()]):
                reason = "nonfinite_evolution"
                break
            progress = store.append(progress["step"] + steps, next_state, arrays)
            state = next_state
            diagnostic_chunks.append(arrays["diagnostics"])
            last_block_seconds = clock() - started
            report = _report(np.concatenate(diagnostic_chunks), numerics)
            print(json.dumps({"variant": variant_name, "step": progress["step"],
                              "time": progress["step"] * numerics.dt,
                              "final_time": numerics.final_time,
                              "block_seconds": last_block_seconds,
                              "checks_passed": report["checks_passed"],
                              "minimum_f": report["metrics"]["min_f"],
                              "first_failure_time": report["first_failure_time"]}, allow_nan=False), flush=True)
        report = _report(np.concatenate(diagnostic_chunks), numerics)
        if reason == "nonfinite_evolution":
            report["checks_passed"] = False
            report["failures"] = sorted(set(report["failures"]) | {"nonfinite_evolution"})
        status = "completed" if progress["step"] == numerics.nsteps else "partial"
        if status == "completed":
            history = _history(store, numerics, progress["step"])
            store.finalize(_sample(history, numerics), report)
        outcome = {"variant": variant_name, "status": status, "step": progress["step"],
                   "time": progress["step"] * numerics.dt, "reason": reason, "report": report,
                   "training_qualified": False}
        (root / "outcomes").mkdir(exist_ok=True)
        atomic_json(root / "outcomes" / f"{variant_name}.json", outcome)
        return outcome


def read_variant(output_root: Path, variant_name: str) -> dict:
    """Verify a complete or partial diagnostic store, including restart state."""
    root = Path(output_root)
    config = validate_config(read_json(root / "config.json"))
    if variant_name not in config["variants"]:
        raise ValueError("Unknown refinement variant")
    directory = root / "variants" / variant_name
    protocol = read_json(directory / "protocol.json")
    expected = {"schema": "gaussian-refinement-v1", "config_sha256": json_digest(config),
                "variant": variant_name, "case": config["case"], "amplitudes": config["amplitudes"],
                "numerics": config["variants"][variant_name], "observations": _observation_schema(),
                "training_qualified": False}
    if (any(protocol.get(key) != value for key, value in expected.items())
            or protocol.get("runtime", {}).get("backend") != "cpu"
            or protocol.get("runtime", {}).get("compute_dtype") != "float64"):
        raise ValueError("Refinement protocol disagrees with its configuration")
    store = CheckpointStore(directory, protocol, read_only=True)
    state, progress = store.load()
    numerics = Numerics(**protocol["numerics"])
    if (set(state) != {"f"} or state["f"].shape != (numerics.nv, numerics.nx)
            or state["f"].dtype != np.dtype("float64")):
        raise ValueError("Refinement state disagrees with its numerical protocol")
    history = _history(store, numerics, progress["step"])
    arrays, report = _sample(history, numerics), _report(history["diagnostics"], numerics)
    records = [entry["state"] for entry in progress["states"]]
    records.extend(entry["observations"] for entry in progress["blocks"])
    if progress["status"] == "complete":
        if progress["step"] != numerics.nsteps:
            raise ValueError("Completed refinement does not reach its final step")
        stored = store.outputs()
        if (stored.keys() != arrays.keys() or any(not np.array_equal(stored[key], value)
                                                  for key, value in arrays.items())
                or progress["report"] != report):
            raise ValueError("Completed refinement outputs disagree with their histories")
        records.append(progress["outputs"])
    for digest in {record.get("provenance_sha256") for record in records}:
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid refinement provenance digest")
        provenance = read_json(root / "provenance" / f"{digest}.json")
        if (json_digest(provenance) != digest or provenance["runtime"] != protocol["runtime"]
                or provenance["source"]["compatibility"] != protocol["source"]
                or provenance["diagnostic_runner_sha256"] != protocol["diagnostic_runner_sha256"]):
            raise ValueError("Refinement provenance disagrees with its protocol")
    outcome_path = root / "outcomes" / f"{variant_name}.json"
    if outcome_path.exists():
        outcome = read_json(outcome_path)
        if outcome.get("step") == progress["step"] and outcome.get("reason") == "nonfinite_evolution":
            report["checks_passed"] = False
            report["failures"] = sorted(set(report["failures"]) | {"nonfinite_evolution"})
    return {"variant": variant_name, "case": config["case"], "amplitudes": config["amplitudes"],
            "numerics": numerics.to_dict(), "arrays": arrays, "state": state, "report": report,
            "status": "completed" if progress["status"] == "complete" else "partial"}
