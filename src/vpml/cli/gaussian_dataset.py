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
    parser.add_argument("--code-root", type=Path,
                        help="Require imports from this staged source checkout")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Write a manifest from dataset parameters; no evolution")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--parameters", type=Path)
    validate = commands.add_parser("validate", help="Validate manifest without allocating compute")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--parameters", type=Path)
    validate.add_argument("--case-id", action="append", dest="case_ids")
    validate.add_argument("--out", type=Path)
    validate.add_argument("--linearized", action="store_true",
                          help="Validate existing cases for a linearized continuation")
    validate.add_argument("--source-transition", type=Path)
    validate.add_argument("--source-transition-sha256")
    continuation = commands.add_parser(
        "prepare-continuation", help="Record reviewed source changes; metadata only, no evolution")
    continuation.add_argument("--out", type=Path, required=True)
    continuation.add_argument("--transition", type=Path, required=True)
    continuation.add_argument("--allow-source-change", action="append", required=True)
    continuation.add_argument("--from-source-sha256")
    continuation.add_argument("--reason", required=True)
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
    run.add_argument("--stop-on-failure", action="store_true",
                     help="Stop after the first recorded or newly detected failed case")
    run.add_argument("--source-transition", type=Path)
    run.add_argument("--source-transition-sha256")
    args = parser.parse_args(argv)
    if args.code_root is not None:
        expected = args.code_root.resolve() / "src/vpml/cli/gaussian_dataset.py"
        if Path(__file__).resolve() != expected:
            raise ValueError("Python imported a different checkout; install the staged source in its prepared environment")
    if getattr(args, "source_transition_sha256", None) and args.source_transition is None:
        parser.error("--source-transition-sha256 requires --source-transition")

    # Backend selection precedes importing vpml, before any numerical module loads JAX.
    if args.command == "run":
        os.environ["VPML_JAX_BACKEND"] = args.backend
        os.environ["JAX_PLATFORMS"] = "cpu" if args.backend == "cpu" else "cuda"
        os.environ["JAX_ENABLE_X64"] = "1"
    from vpml.datasets.manifest import build_manifest, read_manifest, write_manifest

    if args.command == "prepare-continuation":
        from vpml.datasets.continuation import prepare_transition, write_transition

        record = prepare_transition(args.out, allow_source_changes=args.allow_source_change,
                                    reason=args.reason, from_source_sha256=args.from_source_sha256)
        digest = write_transition(args.transition, record)
        print(json.dumps({"transition": str(args.transition), "sha256": digest,
                          "changes": record["changes"], "evolution_started": False}, indent=2))
    elif args.command == "prepare":
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
        from vpml.datasets.selection import select_cases

        manifest = read_manifest(args.manifest)
        if args.parameters is not None and manifest != build_manifest(**read_parameters(args.parameters)):
            raise ValueError("Manifest differs from configured parameters")
        selected = select_cases(manifest, args.case_ids)
        checked_protocols = 0
        if args.out is not None or args.source_transition is not None:
            from vpml.datasets.continuation import validate_existing_protocols
            from vpml.datasets.provenance import source_record
            from vpml.io.serialization import json_digest, read_json

            if args.out is None:
                raise ValueError("Source transition validation requires the existing --out dataset")
            transition = read_json(args.source_transition) if args.source_transition is not None else None
            if (args.source_transition_sha256 is not None
                    and args.source_transition_sha256 != json_digest(transition)):
                raise ValueError("Source transition checksum mismatch")
            checked_protocols = validate_existing_protocols(
                args.out, manifest, source_record()["compatibility"], transition,
                case_ids=args.case_ids, linearized=args.linearized,
            )
        print(json.dumps({"valid": True, "cases": len(manifest["cases"]),
                          "selected_cases": len(selected),
                          "existing_case_protocols_checked": checked_protocols,
                          "case_ids": [case["case_id"] for case in selected] if args.case_ids is not None else None,
                          "numerics": manifest["numerics"], "evolution_started": False}))
    elif args.command == "status":
        from vpml.datasets.runner import dataset_status
        print(json.dumps(dataset_status(args.out, args.manifest), indent=2))
    else:
        from vpml.datasets.runner import run_dataset, stop_signals
        with stop_signals() as stopped:
            report = run_dataset(read_manifest(args.manifest), args.out, backend=args.backend,
                                 max_seconds=args.max_seconds, case_ids=args.case_ids,
                                 linearized=args.linearized, stop_on_failure=args.stop_on_failure,
                                 source_transition=args.source_transition,
                                 source_transition_sha256=args.source_transition_sha256,
                                 stop_requested=stopped)
        print(json.dumps(report, indent=2))
        return 2 if any(case["status"] == "failed" for case in report["outcomes"]) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
