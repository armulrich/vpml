"""Provenance checks without initializing any GPU."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import tempfile
import subprocess
import sys

import jax

from vpml.datasets.provenance import json_digest, runtime_record, source_record


class ProvenanceTests(unittest.TestCase):
    def test_cli_rejects_the_wrong_staged_checkout_before_reading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([
                sys.executable, "-m", "vpml.cli.gaussian_dataset", "--code-root", directory,
                "validate", "--manifest", str(Path(directory) / "missing.json"),
            ], capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("different checkout", result.stderr)
        self.assertNotIn("FileNotFoundError", result.stderr)

    def test_source_inventory_records_all_edits_without_defining_restart_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "vpml"
            (package / "datasets").mkdir(parents=True)
            (package / "datasets/runner.py").write_text("from ..solver import step\n")
            (package / "solver.py").write_text("from .quadrature import integrate\n")
            (package / "quadrature.py").write_text("def integrate(): return 1\n")
            plot = package / "plot.py"
            plot.write_text("color = 'blue'\n")
            original = source_record(package)
            plot.write_text("color = 'purple'\n")
            plotting_change = source_record(package)
            self.assertNotEqual(original["sha256"], plotting_change["sha256"])
            self.assertEqual(set(original), {"sha256", "files"})
            self.assertEqual(original["files"]["vpml/quadrature.py"],
                             plotting_change["files"]["vpml/quadrature.py"])
            (package / "quadrature.py").write_text("def integrate(): return 2\n")
            numerical_change = source_record(package)
            self.assertNotEqual(plotting_change["sha256"], numerical_change["sha256"])
            self.assertEqual(numerical_change["sha256"], json_digest(numerical_change["files"]))

    def test_source_covers_cli_solver_and_environment_lock(self):
        root = Path(__file__).resolve().parents[1]
        record = source_record()
        self.assertEqual(record["sha256"], json_digest(record["files"]))
        for relative in ("vpml/cli/gaussian_dataset.py", "vpml/solvers/physical_grid.py",
                         "environments/delta/requirements.lock"):
            self.assertEqual(record["files"][relative],
                             hashlib.sha256((root / ("src" if relative.startswith("vpml/") else "") / relative).read_bytes()).hexdigest())

    def test_gpu_runtime_records_bounded_driver_query_without_secrets(self):
        jax.config.update("jax_enable_x64", True)
        with mock.patch.object(jax, "default_backend", return_value="gpu"), \
             mock.patch.object(jax, "devices", return_value=[SimpleNamespace(device_kind="Test A100")]), \
             mock.patch("vpml.datasets.provenance.subprocess.run") as query:
            query.return_value.stdout = "580.65.06\n580.65.06\n"
            record = runtime_record("gpu")
        self.assertEqual(record["driver_versions"], ["580.65.06"])
        self.assertEqual(record["devices"], ["Test A100"])
        query.assert_called_once_with(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )

    def test_cpu_mismatch_fails_before_gpu_driver_query(self):
        with mock.patch.object(jax, "default_backend", return_value="cpu"), \
             mock.patch("vpml.datasets.provenance.subprocess.run") as query:
            with self.assertRaisesRegex(RuntimeError, "Required gpu"):
                runtime_record("gpu")
        query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
