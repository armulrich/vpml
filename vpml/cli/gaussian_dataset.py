#!/usr/bin/env python3
"""Prepare, inspect, or explicitly execute Gaussian reference generation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_parameters(path: Path | None) -> dict:
    """Read data-only builder options, rejecting duplicate keys and NaN."""
    if path is None:
        return {}

    from vpml.io.serialization import read_json

    result = read_json(path)
    if not isinstance(result, dict):
        raise ValueError("Parameters must be a JSON object")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Write a manifest from dataset parameters; no evolution")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--parameters", type=Path)
    validate = commands.add_parser("validate", help="Validate manifest without allocating compute")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--parameters", type=Path)
    status = commands.add_parser("status", help="Summarize existing progress without loading JAX devices")
    status.add_argument("--out", type=Path, required=True)
    status.add_argument("--manifest", type=Path)
    run = commands.add_parser("run", help="Execute on resources already allocated by the caller")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    run.add_argument("--max-seconds", type=float, required=True)
    run.add_argument("--case-id", action="append", dest="case_ids")
    run.add_argument("--linearized", action="store_true")
    args = parser.parse_args(argv)

    # Backend selection precedes importing vpml, before any numerical module loads JAX.
    if args.command == "run":
        os.environ["VPML_JAX_BACKEND"] = args.backend
        os.environ["JAX_PLATFORMS"] = "cpu" if args.backend == "cpu" else "cuda"
        os.environ["JAX_ENABLE_X64"] = "1"
    from vpml.datasets.manifest import build_manifest, read_manifest, write_manifest

    if args.command == "prepare":
        manifest = build_manifest(**read_parameters(args.parameters))
        if args.manifest.exists():
            if read_manifest(args.manifest) != manifest:
                raise ValueError("Existing manifest differs; choose a new run directory")
        else:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            write_manifest(args.manifest, manifest)
        print(json.dumps({"manifest": str(args.manifest), "cases": len(manifest["cases"]),
                          "evolution_started": False}))
    elif args.command == "validate":
        manifest = read_manifest(args.manifest)
        if args.parameters is not None and manifest != build_manifest(**read_parameters(args.parameters)):
            raise ValueError("Manifest differs from configured parameters")
        print(json.dumps({"valid": True, "cases": len(manifest["cases"]),
                          "numerics": manifest["numerics"], "evolution_started": False}))
    elif args.command == "status":
        from vpml.datasets.runner import dataset_status
        print(json.dumps(dataset_status(args.out, args.manifest), indent=2))
    else:
        from vpml.datasets.runner import run_dataset, stop_signals
        with stop_signals() as stopped:
            report = run_dataset(read_manifest(args.manifest), args.out, backend=args.backend,
                                 max_seconds=args.max_seconds, case_ids=args.case_ids,
                                 linearized=args.linearized, stop_requested=stopped)
        print(json.dumps(report, indent=2))
        return 2 if any(case["status"] == "failed" for case in report["outcomes"]) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
