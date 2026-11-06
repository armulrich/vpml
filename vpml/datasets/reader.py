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
from .manifest import Numerics, read_manifest
from .schema import (
    DIAGNOSTIC_NAMES, LINEAR_DIAGNOSTIC_NAMES, OUTPUT_SCHEMA,
    validate_completed_progress,
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
    if (protocol.get("manifest_sha256") != json_digest(manifest)
            or protocol.get("case") != case
            or protocol.get("numerics") != manifest["numerics"]
            or protocol.get("output_schema") != OUTPUT_SCHEMA
            or type(protocol.get("linearized")) is not bool):
        raise ValueError("Case protocol disagrees with the dataset manifest")
    store = CheckpointStore(directory, protocol, read_only=True)
    progress = store.read_progress()
    if progress["status"] != "complete":
        raise ValueError("Case is not complete")
    numerics = Numerics(**manifest["numerics"])
    validate_completed_progress(progress, numerics, protocol["linearized"])
    arrays = store.outputs()
    expected_times = np.arange(numerics.nsamples) * numerics.save_dt
    expected_diagnostic_times = np.arange(numerics.nsteps + 1) * numerics.dt
    if (not np.array_equal(arrays["times"], expected_times)
            or not np.array_equal(arrays["diagnostic_times"], expected_diagnostic_times)):
        raise ValueError("Saved time coordinates disagree with the numerical protocol")
    report = progress["report"]
    names = report["diagnostic_names"]
    expected_names = DIAGNOSTIC_NAMES + (LINEAR_DIAGNOSTIC_NAMES if protocol["linearized"] else ())
    if tuple(names) != expected_names:
        raise ValueError("Diagnostic names disagree with the numerical protocol")
    diagnostics = dict(zip(names, arrays["diagnostics"].T, strict=True))
    fields = arrays["fields"]
    result = {
        **case, **arrays, "id": case["case_id"], "family": "gaussian",
        "format": OUTPUT_SCHEMA, "grid": manifest["numerics"],
        "x": np.arange(numerics.output_nx) * (2 * np.pi / case["k0"]) / numerics.output_nx,
        "n": fields[:, 0], "u": fields[:, 1], "p": fields[:, 2],
        "Q": fields[:, 3], "g": fields[:, 4], "E": arrays["electric"],
        "diagnostic_names": names, "report": report,
        "qualification_pass": report["training_qualified"],
        "qualification_failures": report["failures"],
        "qualification_scope": report["qualification_scope"],
        "output_integrity_verified": True, "eta_NL": None,
    }
    result.update(report["metrics"])
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
