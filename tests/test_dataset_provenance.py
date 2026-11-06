"""Provenance checks without initializing any GPU."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import jax

from vpml.datasets.provenance import json_digest, runtime_record, source_record


class ProvenanceTests(unittest.TestCase):
    def test_source_covers_cli_solver_and_environment_lock(self):
        root = Path(__file__).resolve().parents[1]
        record = source_record()
        self.assertEqual(record["sha256"], json_digest(record["files"]))
        for relative in ("scripts/gaussian_dataset.py", "vpml/physical_grid.py",
                         "environments/delta/requirements.lock"):
            self.assertEqual(record["files"][relative],
                             hashlib.sha256((root / relative).read_bytes()).hexdigest())

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
