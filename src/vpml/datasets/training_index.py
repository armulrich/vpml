"""Read-only, memory-mapped case catalog for a published Gaussian dataset."""
from collections.abc import Sequence
import fcntl
import hashlib
from pathlib import Path
import re

import numpy as np

from vpml.io.atomic import atomic_write
from vpml.io.serialization import canonical_json_bytes, file_digest, json_digest, read_json
from .manifest import read_manifest


_CASE_DTYPE = np.dtype([
    ("case_id", "S29"), ("shape_id", "S30"), ("split", "S10"),
    ("strength", "<f8"), ("k0", "<f8"), ("mode_count", "<i4"), ("decay", "<f8"),
    ("case_sha256", "S64"), ("output_sha256", "S64"),
])


def read_completed_digest(root, case_id):
    progress = read_json(root / "cases" / case_id / "progress.json")
    if progress.get("status") != "complete":
        raise ValueError(f"Incomplete training reference: {case_id}")
    digest = progress["outputs"]["sha256"]
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("Invalid reference output digest")
    return digest


class IndexedCases(Sequence):
    """Manifest geometry and scientific identities, without the Fourier coefficients."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return IndexedCases(self.records[index])
        row = self.records[index]
        return {
            "case_id": row["case_id"].decode("ascii"),
            "shape_id": row["shape_id"].decode("ascii"),
            "split": row["split"].decode("ascii"),
            "strength": float(row["strength"]), "k0": float(row["k0"]),
            "mode_count": int(row["mode_count"]), "decay": float(row["decay"]),
        }


def _publish_index(root, directory, manifest_file_digest):
    manifest = read_manifest(root / "manifest.json")
    if file_digest(root / "manifest.json") != manifest_file_digest:
        raise ValueError("Reference manifest changed during indexing")
    records = np.empty(len(manifest["cases"]), dtype=_CASE_DTYPE)
    # Match the scientific identity used before the catalog existed, without
    # retaining a second list of every case ID and output digest in memory.
    digest = hashlib.sha256(b"[")
    separator = b""
    for i, case in enumerate(manifest["cases"]):
        case_id = case["case_id"]
        output = ""
        if case["split"] in ("train", "validation"):
            output = read_completed_digest(root, case_id)
            digest.update(separator + canonical_json_bytes([case_id, output]))
            separator = b","
        records[i] = (
            case_id, case["shape_id"], case["split"], case["strength"], case["k0"],
            case["mode_count"], case["decay"], json_digest(case), output,
        )
    digest.update(b"]")
    table = directory / "cases.npy"
    atomic_write(table, lambda handle: np.save(handle, records, allow_pickle=False))
    metadata = {
        "source_file_sha256": manifest_file_digest,
        "manifest_sha256": json_digest(manifest),
        "training_outputs_sha256": digest.hexdigest(),
        "cases_sha256": file_digest(table),
        "metadata": {key: value for key, value in manifest.items() if key != "cases"},
    }
    atomic_write(directory / "index.json",
                 lambda handle: handle.write(canonical_json_bytes(metadata, newline=True)))


def open_training_index(root, cache):
    """Build once from immutable solver output receipts, then open a checked catalog.

    Only sequential manifest and catalog reads are needed on restart. Selected
    outputs are verified on first access by the data reader. Changes to outputs
    of an already indexed manifest constitute a different scientific dataset.
    """
    root = Path(root)
    source = file_digest(root / "manifest.json")
    directory = Path(cache) / "training-index" / source
    directory.mkdir(exist_ok=True, parents=True)
    header = directory / "index.json"
    if not header.exists():
        with (directory / ".build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not header.exists():
                _publish_index(root, directory, source)
    metadata = read_json(header)
    table = directory / "cases.npy"
    if metadata["source_file_sha256"] != source or file_digest(table) != metadata["cases_sha256"]:
        raise ValueError("Training catalog checksum disagrees with its published index")
    records = np.load(table, mmap_mode="r", allow_pickle=False)
    if records.ndim != 1 or records.dtype != _CASE_DTYPE:
        raise ValueError("Invalid training catalog record layout")
    return metadata, IndexedCases(records)
