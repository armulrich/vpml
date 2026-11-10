"""Execute or compare explicit Gaussian refinement variants on the local CPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _input_record(root: Path, variant: str) -> dict:
    from vpml.io.serialization import file_digest
    root = root.resolve()
    directory = root / "variants" / variant
    return {"root": str(root), "variant": variant,
            "config_file_sha256": file_digest(root / "config.json"),
            "protocol_file_sha256": file_digest(directory / "protocol.json"),
            "progress_file_sha256": file_digest(directory / "progress.json")}


def _require_unchanged(record: dict) -> None:
    if _input_record(Path(record["root"]), record["variant"]) != record:
        raise ValueError("Refinement input changed during comparison; retry with stable checkpoints")


def _load_baseline(config: dict, root: Path, *, external: bool,
                   current_runtime: bool = False) -> tuple[dict, dict, dict]:
    """Verify the declared baseline and retain the exact input snapshot identity."""
    from vpml.diagnostics import gaussian_refinement as refinement
    from vpml.io.serialization import read_json
    root = root.resolve()
    name = config["baseline"]
    record = _input_record(root, name)
    external_config = refinement.validate_config(read_json(root / "config.json"))
    if (external_config["case"] != config["case"]
            or external_config["amplitudes"] != config["amplitudes"]):
        raise ValueError("External baseline must use the declared initial condition and amplitudes")
    if external_config["variants"].get(name) != config["variants"][name]:
        raise ValueError("External baseline numerics must exactly match the declared baseline")
    baseline = refinement.read_variant(root, name)
    if external and baseline["status"] != "completed":
        raise ValueError("An external baseline must be completed before reuse")
    protocol = read_json(root / "variants" / name / "protocol.json")
    if current_runtime:
        expected = {"source": refinement.source_record()["compatibility"],
                    "diagnostic_runner_sha256": refinement.file_digest(Path(refinement.__file__)),
                    "runtime": refinement.runtime_record("cpu")}
        if any(protocol[key] != value for key, value in expected.items()):
            raise ValueError("External baseline must match the current numerical source, runner and runtime")
    _require_unchanged(record)
    return baseline, protocol, record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check the diagnostic configuration")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--baseline-root", type=Path,
                          help="Verify external baseline compatibility before any evolution")
    run = commands.add_parser("run", help="Evolve one diagnostic variant locally; never allocate a GPU")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--variant", required=True)
    run.add_argument("--max-seconds", type=float, required=True)
    compare = commands.add_parser("compare", help="Compare verified diagnostic checkpoints and histories")
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--baseline-root", type=Path,
                         help="Reuse the declared baseline variant from another diagnostic root")
    compare.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    os.environ["VPML_JAX_BACKEND"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "1"
    from vpml.io.serialization import file_digest, read_json
    from vpml.diagnostics.gaussian_refinement import validate_config

    if args.command == "validate":
        config_digest = file_digest(args.config)
        config = validate_config(read_json(args.config))
        report = {"valid": True, "variants": list(config["variants"]),
                  "case_id": config["case"]["case_id"], "evolution_started": False}
        if args.baseline_root is not None:
            _, _, record = _load_baseline(config, args.baseline_root, external=True, current_runtime=True)
            report["baseline_input"] = record
        if file_digest(args.config) != config_digest:
            raise ValueError("Refinement configuration changed during validation")
        print(json.dumps(report))
        return 0
    if args.command == "run":
        from vpml.datasets.runner import stop_signals
        from vpml.diagnostics.gaussian_refinement import run_variant
        with stop_signals() as stopped:
            result = run_variant(read_json(args.config), args.variant, args.out,
                                 max_seconds=args.max_seconds, stop_requested=stopped)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "completed" else 3

    from vpml.datasets.checkpoints import atomic_json
    from vpml.diagnostics.gaussian_refinement import read_variant
    from vpml.diagnostics.refinement_metrics import compare_variants
    args.out = args.out.resolve()
    config_digest = file_digest(args.out / "config.json")
    config = validate_config(read_json(args.out / "config.json"))
    baseline_root = args.baseline_root or args.out
    baseline_name = config["baseline"]
    # Process pairs sequentially, retaining at most two full kinetic states.
    comparisons = []
    baseline, base_protocol, record = _load_baseline(config, baseline_root,
                                                    external=args.baseline_root is not None)
    inputs = {baseline_name: record}
    light_variants = [dict(baseline, state={})]
    for base_name, other_name in config["comparisons"]:
        if base_name != config["baseline"]:
            raise ValueError("Each configured comparison must use the declared baseline")
        record = _input_record(args.out, other_name)
        other = read_variant(args.out, other_name)
        other_protocol = read_json(args.out / "variants" / other_name / "protocol.json")
        if any(base_protocol[key] != other_protocol[key] for key in
               ("source", "diagnostic_runner_sha256", "runtime")):
            raise ValueError("Refinement variants must share numerical source and runtime")
        comparisons.append(compare_variants(baseline, other))
        _require_unchanged(record)
        inputs[other_name] = record
        light_variants.append(dict(other, state={}))
        del other
    report = {"schema_version": 1, "case_id": config["case"]["case_id"],
              "training_qualified": False,
              "scope": "Diagnostic refinement sensitivity; two levels do not establish convergence order",
              "inputs": inputs,
              "comparisons": comparisons}
    for record in inputs.values():
        _require_unchanged(record)
    if file_digest(args.out / "config.json") != config_digest:
        raise ValueError("Refinement configuration changed during comparison")
    atomic_json(args.out / "comparison.json", report)
    if args.plot:
        from vpml.diagnostics.refinement_plot import plot_comparison
        plot_comparison(light_variants, comparisons, args.out / "comparison.png")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
