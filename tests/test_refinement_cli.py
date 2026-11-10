"""A refinement comparison must not silently mix different numerical codes."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest import mock

from vpml.cli.gaussian_refinement import main
from vpml.datasets.manifest import Numerics, build_manifest
from vpml.diagnostics.gaussian_refinement import run_variant
from vpml.diagnostics import gaussian_refinement as refinement
from vpml.io.serialization import file_digest


class RefinementCliTests(unittest.TestCase):
    def test_individually_valid_results_with_different_sources_cannot_be_paired(self):
        numerics = Numerics(nx=16, nv=65, output_nx=8, dt=.01,
                            final_time=.04, save_dt=.02, block_steps=2,
                            storage_dtype="float64")
        manifest = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2.,),
                                  strengths=(.1,), draws_per_combination=1,
                                  numerics=numerics)
        config = {"schema_version": 1, "case": manifest["cases"][0],
                  "amplitudes": manifest["amplitudes"], "baseline": "base",
                  "variants": {"base": numerics.to_dict(),
                               "half_dt": dict(numerics.to_dict(), dt=.005)},
                  "comparisons": [["base", "half_dt"]]}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            for variant in config["variants"]:
                source = {"sha256": variant, "compatibility": {"sha256": variant}}
                with mock.patch("vpml.diagnostics.gaussian_refinement.source_record",
                                return_value=source):
                    result = run_variant(config, variant, root, max_seconds=60)
                self.assertEqual(result["status"], "completed")
            with self.assertRaisesRegex(ValueError, "share numerical source and runtime"):
                main(["compare", "--out", str(root)])
            self.assertFalse((root / "comparison.json").exists())


class CrossRootRefinementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.old = Path(self.temporary.name) / "old"
        self.new = Path(self.temporary.name) / "new"
        numerics = Numerics(nx=16, nv=65, output_nx=8, dt=.01,
                            final_time=.04, save_dt=.02, block_steps=2,
                            storage_dtype="float64")
        manifest = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2.,),
                                  strengths=(.1,), draws_per_combination=1,
                                  numerics=numerics)
        self.old_config = {"schema_version": 1, "case": manifest["cases"][0],
                           "amplitudes": manifest["amplitudes"], "baseline": "base",
                           "variants": {"base": numerics.to_dict(),
                                        "finer_x": dict(numerics.to_dict(), nx=32)},
                           "comparisons": [["base", "finer_x"]]}
        self.new_config = dict(copy.deepcopy(self.old_config), baseline="finer_x",
                               variants={"finer_x": dict(numerics.to_dict(), nx=32),
                                         "spatial_64": dict(numerics.to_dict(), nx=64)},
                               comparisons=[["finer_x", "spatial_64"]])

    def run_tiny(self, root, config, variant, source="shared", stop=False):
        record = {"sha256": source, "compatibility": {"sha256": source}}
        with mock.patch("vpml.diagnostics.gaussian_refinement.source_record", return_value=record), \
                redirect_stdout(io.StringIO()):
            return run_variant(config, variant, root, max_seconds=60, stop_requested=lambda: stop)

    def write_new_config(self, config=None):
        self.new.mkdir(exist_ok=True)
        (self.new / "config.json").write_text(json.dumps(config or self.new_config))

    def compare(self):
        with redirect_stdout(io.StringIO()):
            return main(["compare", "--out", str(self.new), "--baseline-root", str(self.old)])

    def test_completed_external_variant_is_reused_without_reexecution(self):
        self.run_tiny(self.old, self.old_config, "finer_x")
        self.run_tiny(self.new, self.new_config, "spatial_64")
        with mock.patch("vpml.diagnostics.gaussian_refinement.ReferenceEvolution",
                        side_effect=AssertionError("comparison must not evolve")):
            self.assertEqual(self.compare(), 0)
        self.assertFalse((self.new / "variants" / "finer_x").exists())
        report = json.loads((self.new / "comparison.json").read_text())
        pair = report["comparisons"][0]
        self.assertEqual((pair["base_variant"], pair["other_variant"]), ("finer_x", "spatial_64"))
        self.assertFalse(report["training_qualified"])
        for variant, root in (("finer_x", self.old), ("spatial_64", self.new)):
            record = report["inputs"][variant]
            self.assertEqual(record["root"], str(root.resolve()))
            self.assertEqual(record["config_file_sha256"], file_digest(root / "config.json"))
            self.assertEqual(record["protocol_file_sha256"],
                             file_digest(root / "variants" / variant / "protocol.json"))
            self.assertEqual(record["progress_file_sha256"],
                             file_digest(root / "variants" / variant / "progress.json"))

    def test_external_baseline_must_match_declared_numerics(self):
        self.run_tiny(self.old, self.old_config, "finer_x")
        changed = copy.deepcopy(self.new_config)
        changed["variants"]["finer_x"]["dt"] = .005
        self.write_new_config(changed)
        with self.assertRaisesRegex(ValueError, "numerics must exactly match"):
            self.compare()
        self.assertFalse((self.new / "comparison.json").exists())

    def test_external_baseline_must_match_initial_condition(self):
        self.run_tiny(self.old, self.old_config, "finer_x")
        changed = copy.deepcopy(self.new_config)
        changed["case"] = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2.,),
                                         strengths=(.1,), draws_per_combination=1,
                                         base_seed=999, numerics=changed["variants"]["finer_x"])["cases"][0]
        self.write_new_config(changed)
        with self.assertRaisesRegex(ValueError, "declared initial condition"):
            self.compare()

    def test_cross_root_comparison_keeps_source_guard(self):
        self.run_tiny(self.old, self.old_config, "finer_x", source="old-code")
        self.run_tiny(self.new, self.new_config, "spatial_64", source="new-code")
        with self.assertRaisesRegex(ValueError, "share numerical source and runtime"):
            self.compare()
        self.assertFalse((self.new / "comparison.json").exists())

    def test_partial_external_baseline_cannot_be_substituted(self):
        self.assertEqual(self.run_tiny(self.old, self.old_config, "finer_x", stop=True)["status"],
                         "partial")
        self.write_new_config()
        with self.assertRaisesRegex(ValueError, "must be completed before reuse"):
            self.compare()

    def preflight(self):
        with redirect_stdout(io.StringIO()):
            return main(["validate", "--config", str(self.new / "config.json"),
                         "--baseline-root", str(self.old)])

    def test_preflight_verifies_completed_external_baseline_without_evolution(self):
        self.run_tiny(self.old, self.old_config, "finer_x")
        self.write_new_config()
        source = {"sha256": "shared", "compatibility": {"sha256": "shared"}}
        with mock.patch.object(refinement, "source_record", return_value=source), \
                mock.patch.object(refinement, "ReferenceEvolution", side_effect=AssertionError("no evolution")):
            self.assertEqual(self.preflight(), 0)
        self.assertFalse((self.new / "variants").exists())

    def test_preflight_rejects_incompatible_current_source_runner_or_runtime(self):
        self.run_tiny(self.old, self.old_config, "finer_x")
        self.write_new_config()
        source = {"sha256": "shared", "compatibility": {"sha256": "shared"}}
        runtime = refinement.runtime_record("cpu")
        runner_digest = file_digest(Path(refinement.__file__))
        for mismatch in ("source", "runner", "runtime"):
            with self.subTest(mismatch=mismatch):
                current_source = source if mismatch != "source" else {
                    "sha256": "changed", "compatibility": {"sha256": "changed"}}
                current_runtime = runtime if mismatch != "runtime" else dict(runtime, devices=["different CPU"])
                current_runner = runner_digest if mismatch != "runner" else "0" * 64
                with mock.patch.object(refinement, "source_record", return_value=current_source), \
                        mock.patch.object(refinement, "runtime_record", return_value=current_runtime), \
                        mock.patch.object(refinement, "file_digest", return_value=current_runner), \
                        mock.patch.object(refinement, "ReferenceEvolution", side_effect=AssertionError("no evolution")):
                    with self.assertRaisesRegex(ValueError, "current numerical source, runner and runtime"):
                        self.preflight()
                self.assertFalse((self.new / "variants").exists())

    def test_changed_checkpoint_during_read_cannot_publish_comparison(self):
        self.run_tiny(self.old, self.old_config, "finer_x")
        self.run_tiny(self.new, self.new_config, "spatial_64")
        read_variant = refinement.read_variant
        for changed_variant in ("finer_x", "spatial_64"):
            with self.subTest(changed_variant=changed_variant):
                def read_then_advance(root, name):
                    result = read_variant(root, name)
                    if name == changed_variant:
                        progress = Path(root) / "variants" / name / "progress.json"
                        # A valid JSON rewrite models a changed metadata snapshot.
                        progress.write_bytes(progress.read_bytes() + b"\n")
                    return result

                with mock.patch.object(refinement, "read_variant", side_effect=read_then_advance):
                    with self.assertRaisesRegex(ValueError, "input changed during comparison"):
                        self.compare()
                self.assertFalse((self.new / "comparison.json").exists())

    def test_wrapper_stops_at_failed_external_preflight_before_run(self):
        self.write_new_config()
        temporary = Path(self.temporary.name)
        calls = temporary / "calls.txt"
        fake_python = temporary / "python"
        fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$VPML_TEST_CALLS"\nexit 7\n')
        fake_python.chmod(0o755)
        parameters = temporary / "parameters.sh"
        parameters.write_text(
            f"VPML_REFINEMENT_CONFIG={shlex.quote(str(self.new / 'config.json'))}\n"
            "VPML_REFINEMENT_VARIANTS=(spatial_64)\nVPML_REFINEMENT_MAX_SECONDS=60\n")
        wrapper = Path(__file__).resolve().parents[1] / "scripts/local/run_gaussian_refinement_v1.sh"
        environment = dict(os.environ, VPML_PYTHON=str(fake_python),
                           VPML_REFINEMENT_ROOT=str(self.new), VPML_REFINEMENT_PARAMS=str(parameters),
                           VPML_REFINEMENT_BASELINE_ROOT=str(self.old), VPML_TEST_CALLS=str(calls))
        result = subprocess.run(["bash", str(wrapper), "run"], env=environment,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 7)
        commands = calls.read_text().splitlines()
        self.assertEqual(len(commands), 1)
        self.assertIn(" validate --config ", commands[0])
        self.assertIn(f"--baseline-root {self.old}", commands[0])
        self.assertFalse((self.new / "logs").exists())
        self.assertFalse((self.new / "variants").exists())


if __name__ == "__main__":
    unittest.main()
