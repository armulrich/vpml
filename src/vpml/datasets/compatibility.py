"""Restart contracts, independent of source-file layout.

The supported unversioned ``gaussian-moments`` protocol is the original
float64 moment-history/checkpoint format. Its exact layout maps to revision 1
in memory; stored protocols and their checkpoint hashes are never rewritten.
Changing numerical evolution or observation semantics requires a new solver
revision. Source inventories identify producers, not restart compatibility.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re

from vpml.io.serialization import json_digest, read_json
from .checkpoints import CheckpointStore
from .manifest import Numerics, read_manifest, validate_manifest
from .schema import OUTPUT_SCHEMA, LEGACY_OUTPUT_SCHEMA, observation_schema
from .selection import select_cases


PROTOCOL_VERSION = 1
CHECKPOINT_VERSION = 1
SOLVER_REVISION = "vp1d1v-strang-cubic-1"
_CONTRACT = {"protocol_version": PROTOCOL_VERSION,
             "checkpoint_version": CHECKPOINT_VERSION,
             "solver_revision": SOLVER_REVISION}
_FIELDS = {"manifest_sha256", "case", "numerics", "runtime", "linearized",
           "output_schema", "observations"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _inventory(value: dict, *, legacy: bool = False) -> None:
    keys = {"sha256", "files"} | ({"format"} if legacy else set())
    if (not isinstance(value, dict) or set(value) != keys
            or (legacy and (type(value["format"]) is not int or value["format"] != 2))
            or not isinstance(value["files"], dict) or not value["files"]
            or value["sha256"] != json_digest(value["files"])):
        raise ValueError("Invalid source inventory")
    for path, digest in value["files"].items():
        if (not isinstance(path, str) or not path or PurePosixPath(path).is_absolute()
                or PurePosixPath(path).as_posix() != path
                or ".." in PurePosixPath(path).parts or not _digest(digest)
                or (legacy and not path.startswith("vpml/"))):
            raise ValueError("Invalid source file identity")


def _runtime(value: dict) -> None:
    fields = {"python", "packages", "backend", "compute_dtype", "devices", "driver_versions"}
    if (not isinstance(value, dict) or set(value) != fields
            or not isinstance(value["python"], str) or not value["python"]
            or not isinstance(value["backend"], str) or value["backend"] not in {"cpu", "gpu"}
            or value["compute_dtype"] != "float64"
            or not isinstance(value["packages"], dict)
            or not {"jax", "jaxlib", "numpy", "scipy"} <= value["packages"].keys()
            or not all(isinstance(k, str) and k and isinstance(v, str) and v
                       for k, v in value["packages"].items())
            or not isinstance(value["devices"], list) or not value["devices"]
            or not isinstance(value["driver_versions"], list)
            or not all(isinstance(v, str) and v
                       for v in value["devices"] + value["driver_versions"])
            or (value["backend"] == "gpu" and not value["driver_versions"])):
        raise ValueError("Invalid supported runtime record")


def normalize_protocol(protocol: dict) -> dict:
    """Validate a supported restart protocol and return its in-memory contract."""
    if not isinstance(protocol, dict):
        raise ValueError("Invalid checkpoint protocol")
    if set(protocol) == _FIELDS | {"source"}:
        _inventory(protocol["source"], legacy=True)
        normalized = {key: value for key, value in protocol.items() if key != "source"}
        normalized.update(_CONTRACT)
    elif set(protocol) == _FIELDS | _CONTRACT.keys():
        normalized = dict(protocol)
    else:
        raise ValueError("Unsupported checkpoint protocol layout")
    if (any(normalized[key] != value or type(normalized[key]) is not type(value)
            for key, value in _CONTRACT.items())
            or normalized["output_schema"] != OUTPUT_SCHEMA):
        raise ValueError("Unsupported protocol, checkpoint or solver revision")
    if (not _digest(normalized["manifest_sha256"])
            or not isinstance(normalized["case"], dict)
            or type(normalized["linearized"]) is not bool
            or normalized["observations"] != observation_schema()):
        raise ValueError("Invalid checkpoint scientific settings")
    try:
        numerics = Numerics(**normalized["numerics"])
        if set(normalized["numerics"]) != numerics.to_dict().keys():
            raise ValueError("Incomplete numerical settings")
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid checkpoint numerics") from error
    _runtime(normalized["runtime"])
    return normalized


def build_protocol(manifest: dict, case: dict, runtime: dict, linearized: bool = False) -> dict:
    """Build the current explicit solver/storage contract for one case."""
    protocol = {**_CONTRACT, "manifest_sha256": json_digest(manifest), "case": case,
                "numerics": manifest["numerics"], "runtime": runtime,
                "linearized": linearized, "output_schema": manifest["output_schema"],
                "observations": observation_schema(manifest["output_schema"])}
    normalize_protocol(protocol)
    return protocol


def resume_protocol(expected: dict, saved: dict) -> dict:
    """Require identical scientific/runtime contracts; retain saved bytes/hash."""
    if normalize_protocol(expected) != normalize_protocol(saved):
        raise ValueError("Checkpoint protocol mismatch: scientific settings or runtime changed")
    return saved


def _only_initial_write_temps(directory: Path, filename: str) -> bool:
    if directory.is_symlink() or not directory.is_dir():
        return False
    pattern = re.compile(r"\." + re.escape(filename) + r"\.[0-9a-f]{32}\.tmp\Z")
    return all(path.is_file() and not path.is_symlink() and pattern.fullmatch(path.name)
               for path in directory.iterdir())


def validate_existing_protocols(root: Path, manifest: dict,
                                case_ids: list[str] | None = None,
                                *, linearized: bool = False) -> int:
    """Preflight metadata without JAX or arrays; actual runtime is checked by the worker."""
    validate_manifest(manifest)
    if type(linearized) is not bool:
        raise ValueError("linearized must be a boolean")
    if manifest["output_schema"] != OUTPUT_SCHEMA:
        raise ValueError("Legacy output schemas are read-only")
    root = Path(root)
    if not (root / "manifest.json").exists():
        if (root.exists() or root.is_symlink()) and not _only_initial_write_temps(root, "manifest.json"):
            raise ValueError("Existing output directory has no dataset manifest")
        return 0
    if read_manifest(root / "manifest.json") != manifest:
        raise ValueError("Existing output directory belongs to a different manifest")
    checked = 0
    for case in select_cases(manifest, case_ids):
        directory = root / "cases" / case["case_id"]
        protocol_path = directory / "protocol.json"
        if not protocol_path.exists():
            if ((directory.exists() or directory.is_symlink())
                    and not _only_initial_write_temps(directory, "protocol.json")):
                raise ValueError("Existing case files have no protocol")
            continue
        saved = read_json(protocol_path)
        normalized = normalize_protocol(saved)
        expected = build_protocol(manifest, case, normalized["runtime"], linearized)
        resume_protocol(expected, saved)
        store = CheckpointStore(directory, saved, read_only=True)
        if (directory / "progress.json").exists():
            store.read_progress()
        checked += 1
    return checked


def validate_provenance(root: Path, digest: str, protocol: dict) -> dict:
    """Verify each producing source and runtime without requiring source equality."""
    if not _digest(digest):
        raise ValueError("Invalid array provenance digest")
    provenance = read_json(Path(root) / "provenance" / f"{digest}.json")
    if (not isinstance(provenance, dict) or json_digest(provenance) != digest
            or provenance.get("runtime") != protocol.get("runtime")):
        raise ValueError("Array provenance disagrees with its protocol")
    if protocol["output_schema"] == LEGACY_OUTPUT_SCHEMA:
        if provenance["source"]["compatibility"] != protocol["source"]:
            raise ValueError("Array provenance disagrees with its protocol")
        return provenance
    normalize_protocol(protocol)
    source = provenance.get("source")
    if set(provenance) == {"source", "runtime"}:
        if not isinstance(source, dict) or set(source) != {"sha256", "files", "compatibility"}:
            raise ValueError("Invalid historical source provenance")
        _inventory(source["compatibility"], legacy=True)
        _inventory({key: value for key, value in source.items() if key != "compatibility"})
        if "source" in protocol and source["compatibility"] != protocol["source"]:
            raise ValueError("Historical array provenance disagrees with its original protocol")
        if any(source["files"].get(path) != digest
               for path, digest in source["compatibility"]["files"].items()):
            raise ValueError("Historical source inventories disagree")
    elif set(provenance) == {"source", "runtime"} | _CONTRACT.keys():
        if any(provenance[key] != value or type(provenance[key]) is not type(value)
               for key, value in _CONTRACT.items()):
            raise ValueError("Unsupported array provenance solver contract")
        _inventory(source)
    else:
        raise ValueError("Unsupported array provenance format")
    return provenance
