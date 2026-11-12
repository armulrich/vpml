"""Explicit source changes for continuation with unchanged numerical kernels.

A transition authorizes exact reviewed orchestration changes. It is not a
numerical equivalence test. Original case protocols and array provenance stay
immutable; newly written arrays identify the source that actually wrote them.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re

from vpml.io.atomic import atomic_write
from vpml.io.serialization import canonical_json_bytes, json_digest, read_json
from .checkpoints import CheckpointStore
from .manifest import read_manifest
from .provenance import source_record
from .schema import observation_schema
from .selection import select_cases


# These files control entry points, provenance and execution, including runtime
# checks in provenance.py. Their changes require actual review to preserve
# numerical and runtime behavior; the allowlist does not prove equivalence.
# Numerical kernel, schema and storage files remain outside this mechanism.
_REVIEWABLE = frozenset({
    "vpml/__init__.py", "vpml/datasets/runner.py",
    "vpml/datasets/provenance.py", "vpml/datasets/continuation.py",
})
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _source(value: dict) -> None:
    if (not isinstance(value, dict) or set(value) != {"format", "sha256", "files"}
            or value["format"] != 2 or not isinstance(value["files"], dict)
            or not value["files"] or value["sha256"] != json_digest(value["files"])):
        raise ValueError("Invalid source compatibility record")
    for path, digest in value["files"].items():
        if (not isinstance(path, str) or not path.startswith("vpml/")
                or PurePosixPath(path).as_posix() != path
                or ".." in PurePosixPath(path).parts or not _digest(digest)):
            raise ValueError("Invalid source file identity")


def changed_sources(before: dict, after: dict) -> dict:
    """List exact changed, added and removed dependency files."""
    _source(before)
    _source(after)
    old, new = before["files"], after["files"]
    return {path: {"before": old.get(path), "after": new.get(path)}
            for path in sorted(old.keys() | new.keys()) if old.get(path) != new.get(path)}


def validate_transition(record: dict, *, current_source: dict | None = None,
                        manifest_sha256: str | None = None) -> None:
    """Reject implicit, stale or numerical source changes before continuation."""
    keys = {"format", "kind", "manifest_sha256", "from_source", "to_source",
            "changes", "reason"}
    if (not isinstance(record, dict) or set(record) != keys or record["format"] != 1
            or record["kind"] != "reviewed-source-transition"
            or not _digest(record["manifest_sha256"])
            or not isinstance(record["reason"], str) or not record["reason"].strip()):
        raise ValueError("Invalid source transition record")
    changes = changed_sources(record["from_source"], record["to_source"])
    if not changes or record["changes"] != changes:
        raise ValueError("Source transition must record every exact source change")
    if set(changes) - _REVIEWABLE:
        raise ValueError("Source transition changes protected numerical or storage dependencies")
    if current_source is not None and record["to_source"] != current_source:
        raise ValueError("Source transition does not match the current source")
    if manifest_sha256 is not None and record["manifest_sha256"] != manifest_sha256:
        raise ValueError("Source transition belongs to a different manifest")


def prepare_transition(root: Path, *, allow_source_changes: list[str], reason: str,
                       from_source_sha256: str | None = None) -> dict:
    """Prepare a review record from existing metadata, without evolving or writing.

    All differences must be named individually. If a dataset contains multiple
    old source identities, the caller must select one explicitly.
    """
    root = Path(root)
    manifest = read_manifest(root / "manifest.json")
    current = source_record()["compatibility"]
    parents = {}
    for case in manifest["cases"]:
        path = root / "cases" / case["case_id"] / "protocol.json"
        if not path.exists():
            continue
        protocol = read_json(path)
        if (protocol.get("manifest_sha256") != json_digest(manifest)
                or protocol.get("case") != case
                or protocol.get("numerics") != manifest["numerics"]
                or protocol.get("output_schema") != manifest["output_schema"]):
            raise ValueError("Existing case protocol disagrees with the manifest")
        parent = protocol["source"]
        _source(parent)
        if parent != current:
            parents[parent["sha256"]] = parent
    if from_source_sha256 is not None:
        if from_source_sha256 not in parents:
            raise ValueError("Requested parent source does not occur in the existing dataset")
        parent = parents[from_source_sha256]
    elif len(parents) == 1:
        parent = next(iter(parents.values()))
    else:
        raise ValueError("Expected one previous source; specify --from-source-sha256 when ambiguous")
    changes = changed_sources(parent, current)
    if (len(allow_source_changes) != len(set(allow_source_changes))
            or set(allow_source_changes) != set(changes)):
        raise ValueError("--allow-source-change must name exactly: " + ", ".join(changes))
    record = {"format": 1, "kind": "reviewed-source-transition",
              "manifest_sha256": json_digest(manifest), "from_source": parent,
              "to_source": current, "changes": changes, "reason": reason}
    validate_transition(record, current_source=current, manifest_sha256=json_digest(manifest))
    return record


def write_transition(path: Path, record: dict) -> str:
    """Publish an immutable transition; an identical existing record is reusable."""
    validate_transition(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(record, newline=True)
    if not path.exists():
        try:
            atomic_write(path, lambda handle: handle.write(payload), replace=False)
        except FileExistsError:
            pass
    if path.is_symlink() or read_json(path) != record:
        raise ValueError("Existing source transition differs or is a symlink")
    return json_digest(record)


def continuation_protocol(expected: dict, saved: dict, transition: dict | None) -> dict:
    """Reuse the original protocol only for an exact, explicit source transition."""
    if saved == expected:
        return saved
    if {k: v for k, v in saved.items() if k != "source"} != {
            k: v for k, v in expected.items() if k != "source"}:
        raise ValueError("Checkpoint protocol mismatch: scientific settings or runtime changed")
    if transition is None:
        raise ValueError("Checkpoint source changed; supply a reviewed --source-transition")
    validate_transition(transition, current_source=expected["source"],
                        manifest_sha256=expected["manifest_sha256"])
    if transition["from_source"] != saved["source"]:
        raise ValueError("Source transition does not match the checkpoint's original source")
    return saved


def _only_initial_write_temps(directory: Path, filename: str) -> bool:
    """Recognize an empty directory or unpublished initial metadata writes."""
    if directory.is_symlink() or not directory.is_dir():
        return False
    pattern = re.compile(r"\." + re.escape(filename) + r"\.[0-9a-f]{32}\.tmp\Z")
    return all(path.is_file() and not path.is_symlink() and pattern.fullmatch(path.name)
               for path in directory.iterdir())


def validate_existing_protocols(root: Path, manifest: dict, current_source: dict,
                                transition: dict | None = None,
                                case_ids: list[str] | None = None,
                                *, linearized: bool = False) -> int:
    """Check selected continuation metadata before resources are allocated.

    Arrays are not read and no compute backend is initialized. The saved GPU
    runtime is retained here; the worker checks it against its actual runtime.
    The requested linearized mode must match the saved protocol.
    A new output directory or interrupted initial manifest write is valid only
    without a source transition.
    """
    root = Path(root)
    if type(linearized) is not bool:
        raise ValueError("linearized must be a boolean")
    _source(current_source)
    manifest_digest = json_digest(manifest)
    if transition is not None:
        validate_transition(transition, current_source=current_source,
                            manifest_sha256=manifest_digest)
    if not (root / "manifest.json").exists():
        if transition is not None:
            raise ValueError("Source transition requires an existing dataset manifest")
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
        if (not isinstance(saved, dict) or type(saved.get("linearized")) is not bool
                or not isinstance(saved.get("runtime"), dict) or not saved["runtime"]):
            raise ValueError("Invalid existing case runtime or linearized mode")
        expected = {
            "manifest_sha256": manifest_digest, "case": case,
            "numerics": manifest["numerics"], "runtime": saved["runtime"],
            "source": current_source, "linearized": linearized,
            "output_schema": manifest["output_schema"],
            "observations": observation_schema(manifest["output_schema"]),
        }
        continuation_protocol(expected, saved, transition)
        store = CheckpointStore(directory, saved, read_only=True)
        if (directory / "progress.json").exists():
            store.read_progress()
        checked += 1
    return checked


def validate_provenance(root: Path, digest: str, protocol: dict) -> dict:
    """Verify array origin, including any recorded source transition."""
    if not _digest(digest):
        raise ValueError("Invalid array provenance digest")
    root = Path(root)
    provenance = read_json(root / "provenance" / f"{digest}.json")
    if json_digest(provenance) != digest or provenance["runtime"] != protocol["runtime"]:
        raise ValueError("Array provenance disagrees with its protocol")
    current = provenance["source"]["compatibility"]
    transition_digest = provenance.get("source_transition_sha256")
    transition = None
    if transition_digest is not None:
        if not _digest(transition_digest):
            raise ValueError("Invalid source transition digest")
        transition = read_json(root / "source-transitions" / f"{transition_digest}.json")
        if json_digest(transition) != transition_digest:
            raise ValueError("Source transition checksum mismatch")
        validate_transition(transition, current_source=current,
                            manifest_sha256=protocol["manifest_sha256"])
    if current != protocol["source"]:
        continuation_protocol(dict(protocol, source=current), protocol, transition)
    return provenance
