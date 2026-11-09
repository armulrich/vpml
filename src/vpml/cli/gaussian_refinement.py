"""Execute or compare explicit Gaussian refinement variants on the local CPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check the diagnostic configuration")
    validate.add_argument("--config", type=Path, required=True)
    run = commands.add_parser("run", help="Evolve one diagnostic variant locally; never allocate a GPU")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--variant", required=True)
    run.add_argument("--max-seconds", type=float, required=True)
    compare = commands.add_parser("compare", help="Compare verified diagnostic checkpoints and histories")
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    os.environ["VPML_JAX_BACKEND"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "1"
    from vpml.io.serialization import read_json
    from vpml.diagnostics.gaussian_refinement import validate_config

    if args.command == "validate":
        config = validate_config(read_json(args.config))
        print(json.dumps({"valid": True, "variants": list(config["variants"]),
                          "case_id": config["case"]["case_id"], "evolution_started": False}))
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
    config = validate_config(read_json(args.out / "config.json"))
    # Process pairs sequentially, retaining at most two full kinetic states.
    comparisons = []
    baseline = read_variant(args.out, config["baseline"])
    base_protocol = read_json(args.out / "variants" / config["baseline"] / "protocol.json")
    light_variants = [dict(baseline, state={})]
    for base_name, other_name in config["comparisons"]:
        if base_name != config["baseline"]:
            raise ValueError("Each configured comparison must use the declared baseline")
        other = read_variant(args.out, other_name)
        other_protocol = read_json(args.out / "variants" / other_name / "protocol.json")
        if any(base_protocol[key] != other_protocol[key] for key in
               ("source", "diagnostic_runner_sha256", "runtime")):
            raise ValueError("Refinement variants must share numerical source and runtime")
        comparisons.append(compare_variants(baseline, other))
        light_variants.append(dict(other, state={}))
        del other
    report = {"schema_version": 1, "case_id": config["case"]["case_id"],
              "training_qualified": False,
              "scope": "Diagnostic refinement sensitivity; two levels do not establish convergence order",
              "comparisons": comparisons}
    atomic_json(args.out / "comparison.json", report)
    if args.plot:
        from vpml.diagnostics.refinement_plot import plot_comparison
        plot_comparison(light_variants, comparisons, args.out / "comparison.png")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
