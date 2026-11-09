"""A refinement comparison must not silently mix different numerical codes."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from vpml.cli.gaussian_refinement import main
from vpml.datasets.manifest import Numerics, build_manifest
from vpml.diagnostics.gaussian_refinement import run_variant


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


if __name__ == "__main__":
    unittest.main()
