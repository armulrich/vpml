"""Read validated reference outputs through explicit format adapters.

The current format verifies completed output checksums and schemas. Historical
flat NPZ files retain their original metadata and optional checksums. Neither
adapter infers that a completed trajectory is qualified training data.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from vpml.io.serialization import file_digest, json_digest, read_json
from .checkpoints import CheckpointStore
from .continuation import validate_provenance
from .manifest import Numerics, read_manifest
from .schema import (
    DIAGNOSTIC_NAMES, LINEAR_DIAGNOSTIC_NAMES, OUTPUT_SCHEMA, LEGACY_OUTPUT_SCHEMA,
    observation_schema, validate_completed_progress,
)


def read_case(root: str | Path, case_id: str) -> dict:
    """Load one completed case, without loading its large kinetic states."""
    root = Path(root)
    manifest = read_manifest(root / "manifest.json")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    if case_id not in cases:
        raise ValueError("Unknown manifest case ID")
    return _read_current_case(root, manifest, cases[case_id])


def _read_current_case(root: Path, manifest: dict, case: dict) -> dict:
    directory = root / "cases" / case["case_id"]
    protocol = read_json(directory / "protocol.json")
    output_schema = manifest["output_schema"]
    if (protocol.get("manifest_sha256") != json_digest(manifest)
            or protocol.get("case") != case
            or protocol.get("numerics") != manifest["numerics"]
            or protocol.get("output_schema") != output_schema
            or type(protocol.get("linearized")) is not bool
            or protocol.get("observations") != observation_schema(output_schema)):
        raise ValueError("Case protocol disagrees with the dataset manifest")
    store = CheckpointStore(directory, protocol, read_only=True)
    progress = store.read_progress()
    if progress["status"] != "complete":
        raise ValueError("Case is not complete")
    numerics = Numerics(**manifest["numerics"])
    validate_completed_progress(progress, numerics, protocol["linearized"], output_schema)
    arrays = store.outputs()
    provenance_digest = progress["outputs"].get("provenance_sha256")
    if provenance_digest is not None:
        validate_provenance(root, provenance_digest, protocol)
    expected_times = np.arange(numerics.nsamples) * numerics.save_dt
    if not np.array_equal(arrays["times"], expected_times):
        raise ValueError("Saved time coordinates disagree with the numerical protocol")
    report = progress["report"]
    fields = arrays["fields"]
    result = {
        **case, **arrays, "id": case["case_id"], "family": "gaussian",
        "format": output_schema, "grid": manifest["numerics"],
        "x": np.arange(numerics.output_nx) * (2 * np.pi / case["k0"]) / numerics.output_nx,
        "n": fields[:, 0], "u": fields[:, 1], "p": fields[:, 2],
        "Q": fields[:, 3], "g": fields[:, 4], "E": arrays["electric"],
        "report": report,
        "qualification_pass": report.get("training_qualified", False),
        "qualification_failures": report["failures"],
        "qualification_scope": report.get("qualification_scope", "Numerical accuracy not assessed"),
        "output_integrity_verified": True, "eta_NL": None,
        "provenance_sha256": provenance_digest,
    }
    result.update(report["metrics"])
    if output_schema == OUTPUT_SCHEMA:
        return _add_moment_summaries(result, protocol["linearized"])
    if output_schema != LEGACY_OUTPUT_SCHEMA:
        raise ValueError("Unsupported output schema")
    expected_diagnostic_times = np.arange(numerics.nsteps + 1) * numerics.dt
    if not np.array_equal(arrays["diagnostic_times"], expected_diagnostic_times):
        raise ValueError("Saved time coordinates disagree with the numerical protocol")
    names = report["diagnostic_names"]
    expected_names = DIAGNOSTIC_NAMES + (LINEAR_DIAGNOSTIC_NAMES if protocol["linearized"] else ())
    if tuple(names) != expected_names:
        raise ValueError("Diagnostic names disagree with the numerical protocol")
    diagnostics = dict(zip(names, arrays["diagnostics"].T, strict=True))
    result["diagnostic_names"] = names
    result["min_full_distribution"] = report["metrics"]["min_f"]
    result["max_boundary_mass_fraction"] = report["metrics"]["max_boundary_abs_fraction"]
    result["full_energy"] = np.interp(arrays["times"], arrays["diagnostic_times"],
                                      diagnostics["electric_energy"])
    if protocol["linearized"]:
        result["linear_energy"] = np.interp(arrays["times"], arrays["diagnostic_times"],
                                            diagnostics["linear_electric_energy"])
        denominator = np.trapezoid(2 * diagnostics["electric_energy"], arrays["diagnostic_times"])
        if denominator > 0:
            result["eta_NL"] = float(np.sqrt(np.trapezoid(diagnostics["delta_e2"],
                                                         arrays["diagnostic_times"]) / denominator))
    return result


def _add_moment_summaries(case: dict, linearized: bool) -> dict:
    """Calculate descriptive summaries from saved fields, without kinetic scans.

    Electric energy uses the retained spatial grid; discarded electric-field
    modes cannot be reconstructed. No tolerance or accuracy decision is made.
    """
    length = 2 * np.pi / case["k0"]
    raw = np.asarray(case["raw_moments"], dtype=np.float64)
    electric = np.asarray(case["electric"], dtype=np.float64)
    mass = length * raw[:, 0].mean(axis=-1)
    energy = .5 * length * np.mean(electric ** 2, axis=-1)
    total = .5 * length * raw[:, 2].mean(axis=-1) + energy

    def relative_drift(values):
        if values[0] == 0:
            return None
        return float(np.max(np.abs(values / values[0] - 1)))

    case.update({
        "mass": mass, "full_energy": energy, "total_energy": total,
        "max_relative_mass_error": relative_drift(mass),
        "max_relative_total_energy_error": relative_drift(total),
        "conservation_scope": "Computed after evolution from saved raw moments and retained electric-field modes; no accuracy threshold applied.",
    })
    if linearized:
        linear = np.asarray(case["linear_electric"], dtype=np.float64)
        case["linear_energy"] = .5 * length * np.mean(linear ** 2, axis=-1)
        difference = length * np.mean((electric - linear) ** 2, axis=-1)
        denominator = np.trapezoid(2 * energy, case["times"])
        if denominator > 0:
            case["eta_NL"] = float(np.sqrt(np.trapezoid(difference, case["times"]) / denominator))
    return case


def _read_legacy_case(metadata_path: Path) -> dict | None:
    metadata = read_json(metadata_path)
    path = metadata_path.with_suffix(".npz")
    if not path.exists() or metadata.get("family") not in ("gaussian", "linear_landau"):
        return None
    expected = metadata.get("result_sha256")
    if expected and file_digest(path) != expected:
        raise ValueError(f"Result checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if any(value.dtype.kind not in "buifc" or not np.isfinite(value).all()
           for value in arrays.values()):
        raise ValueError(f"Invalid legacy numeric arrays: {path}")
    return {**metadata, **arrays, "id": metadata_path.stem,
            "format": "legacy-flat-npz", "output_integrity_verified": bool(expected)}


def iter_cases(root: str | Path) -> Iterator[dict]:
    """Yield completed cases one at a time; pending/partial cases are skipped.

    Completion includes cases whose numerical checks failed, with that failure
    retained in their report. Corrupt completed outputs raise an error.
    """
    root = Path(root)
    if (root / "manifest.json").exists():
        manifest = read_manifest(root / "manifest.json")
        for case in manifest["cases"]:
            progress_path = root / "cases" / case["case_id"] / "progress.json"
            if progress_path.exists() and read_json(progress_path).get("status") == "complete":
                yield _read_current_case(root, manifest, case)
    else:
        for path in sorted((root / "cases").glob("*.json")):
            case = _read_legacy_case(path)
            if case is not None:
                yield case
