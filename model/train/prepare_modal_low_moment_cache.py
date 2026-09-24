"""Build a compact, lossless-for-low-moment copy of a reference cache.

The low-moment trainer reads only Hermite coefficients C0:C3.  Its Nx=128
held-out evaluator reads only the first 65 electric-field Fourier modes.  This
utility copies exactly those values, retains the immutable IC manifest, and
records the parent hashes so the smaller cache can be uploaded once to Modal.
It never modifies the source cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping, Optional, Sequence

import numpy as np

from model.train.interface_flux_data import load_ic_manifest, sha256_file, sha256_json


SCHEMA_VERSION = 1
HERMITE_COUNT = 4


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _copy_low_hermite_rows(source: Path, destination: Path) -> Mapping[str, object]:
    source_array = np.load(source, mmap_mode="r", allow_pickle=False)
    if source_array.ndim != 3 or source_array.shape[1] < HERMITE_COUNT:
        raise ValueError(f"Expected [time, Hermite, Fourier] data at {source}")
    if source_array.dtype != np.complex64:
        raise ValueError(f"Expected complex64 data at {source}, got {source_array.dtype}")
    expected_shape = (int(source_array.shape[0]), HERMITE_COUNT, int(source_array.shape[2]))
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.complex64,
        shape=expected_shape,
    )
    for start in range(0, expected_shape[0], 128):
        stop = min(start + 128, expected_shape[0])
        output[start:stop] = source_array[start:stop, :HERMITE_COUNT]
    output.flush()
    del output
    temporary.replace(destination)
    return {
        "shape": list(expected_shape),
        "dtype": "complex64",
        "sha256": sha256_file(destination),
    }


def _copy_evaluation_snapshot(
    source: Path,
    destination: Path,
    *,
    target_nx: int,
) -> Mapping[str, object]:
    target_modes = int(target_nx) // 2 + 1
    with np.load(source, allow_pickle=False) as payload:
        required = ("energy", "E_hat_hist_times", "E_hat_hist", "k_arr")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(f"Snapshot {source} is missing {missing}")
        field_hat = np.asarray(payload["E_hat_hist"][:, :target_modes])
        k_arr = np.asarray(payload["k_arr"][:target_modes])
        values = {
            "energy": np.asarray(payload["energy"]),
            "E_hat_hist_times": np.asarray(payload["E_hat_hist_times"]),
            "E_hat_hist": field_hat,
            "k_arr": k_arr,
        }
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return {
        "E_hat_hist_shape": list(field_hat.shape),
        "sha256": sha256_file(destination),
    }


def _copy_reusable_derived_data(source: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.npz")):
        shutil.copy2(path, destination / path.name)
        copied.append(path.name)
    for path in sorted(source.glob("low_moment_primitive_targets_*")):
        target = destination / path.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(path, target)
        copied.append(path.name)
    return copied


def prepare_cache(source: Path, destination: Path, *, target_nx: int = 128) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Source and destination caches must differ")
    completion_path = destination / "COMPACT_CACHE_MANIFEST.json"
    if completion_path.exists():
        print(f"[compact-cache] already complete: {completion_path}")
        return completion_path
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse incomplete nonempty compact cache {destination}"
        )

    source_metadata_path = source / "metadata.json"
    source_manifest_path = source / "ic_manifest.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    manifest = load_ic_manifest(source_manifest_path)
    configuration = dict(source_metadata["configuration"])
    if int(configuration["max_projection_order"]) < HERMITE_COUNT:
        raise ValueError("Source cache does not contain C0:C3")
    source_projection_order = int(configuration["max_projection_order"])
    configuration["max_projection_order"] = HERMITE_COUNT

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cases").mkdir(exist_ok=True)
    (destination / "snapshots").mkdir(exist_ok=True)
    shutil.copy2(source_manifest_path, destination / "ic_manifest.json")
    compact_metadata = {
        "format": source_metadata["format"],
        "configuration": configuration,
        "configuration_sha256": sha256_json(configuration),
        "manifest_sha256": source_metadata["manifest_sha256"],
        "compact_low_moment_cache": {
            "schema_version": SCHEMA_VERSION,
            "parent_configuration_sha256": source_metadata["configuration_sha256"],
            "parent_metadata_sha256": sha256_file(source_metadata_path),
            "parent_manifest_sha256": sha256_file(source_manifest_path),
            "source_projection_order": source_projection_order,
            "retained_hermite_rows": [0, 1, 2, 3],
            "evaluation_target_nx": int(target_nx),
            "training_values_changed": False,
        },
    }
    _write_json_atomic(destination / "metadata.json", compact_metadata)

    records = []
    for index, case in enumerate(manifest["cases"], start=1):
        case_id = str(case["case_id"])
        source_case = source / "cases" / f"{case_id}.npy"
        source_marker = source / "cases" / f"{case_id}.json"
        source_snapshot = source / "snapshots" / f"{case_id}.npz"
        destination_case = destination / "cases" / source_case.name
        destination_marker = destination / "cases" / source_marker.name
        destination_snapshot = destination / "snapshots" / source_snapshot.name
        source_marker_value = json.loads(source_marker.read_text(encoding="utf-8"))
        compact_case = _copy_low_hermite_rows(source_case, destination_case)
        compact_snapshot = _copy_evaluation_snapshot(
            source_snapshot, destination_snapshot, target_nx=target_nx
        )
        marker = {
            "case_id": case_id,
            "regime": str(case["regime"]),
            "split": str(case["split"]),
            **compact_case,
            "history_times_start": source_marker_value["history_times_start"],
            "history_times_stop": source_marker_value["history_times_stop"],
            "history_count": source_marker_value["history_count"],
            "parent_sha256": source_marker_value["sha256"],
        }
        _write_json_atomic(destination_marker, marker)
        records.append(
            {
                "case_id": case_id,
                "coefficient_sha256": compact_case["sha256"],
                "snapshot_sha256": compact_snapshot["sha256"],
            }
        )
        print(f"[compact-cache] {index:02d}/{len(manifest['cases'])} {case_id}")

    copied_derived = _copy_reusable_derived_data(
        source / "derived", destination / "derived"
    )
    completion = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "destination": str(destination),
        "manifest_sha256": str(manifest["sha256"]),
        "metadata_sha256": sha256_file(destination / "metadata.json"),
        "target_nx": int(target_nx),
        "retained_hermite_rows": [0, 1, 2, 3],
        "case_count": len(records),
        "cases": records,
        "copied_derived_entries": copied_derived,
    }
    _write_json_atomic(completion_path, completion)
    print(f"[compact-cache] complete: {completion_path}")
    return completion_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--target-nx", type=int, default=128)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    prepare_cache(args.source, args.destination, target_nx=args.target_nx)


if __name__ == "__main__":
    main()
