"""Read actual tiny CPU trajectories and historical files without mutation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vpml.datasets.checkpoints import CheckpointStore
from vpml.datasets.manifest import Numerics, build_manifest
from vpml.datasets.reader import iter_cases, read_case
from vpml.datasets.runner import run_dataset


class DatasetReaderTests(unittest.TestCase):
    def test_completed_outputs_are_checked_and_read_without_changing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2.,),
                                      strengths=(.1,), draws_per_combination=1,
                                      numerics=Numerics(nx=16, nv=64, output_nx=8,
                                                       dt=.01, final_time=.02, save_dt=.01))
            run_dataset(manifest, root, backend="cpu", max_seconds=60)
            before = {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
            case = read_case(root, manifest["cases"][0]["case_id"])
            self.assertEqual(case["n"].shape, (3, 8))
            self.assertTrue(case["output_integrity_verified"])
            self.assertFalse(case["qualification_pass"])
            self.assertIsNone(case["eta_NL"])
            self.assertNotIn("final_f", case)
            self.assertEqual(len(list(iter_cases(root))), 1)
            after = {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            case_dir = root / "cases" / case["id"]
            progress = json.loads((case_dir / "progress.json").read_text())
            output = case_dir / progress["outputs"]["file"]
            output.write_bytes(output.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_case(root, case["id"])

    def test_legacy_reader_preserves_field_meanings_and_optional_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cases").mkdir()
            path = root / "cases/old_base.npz"
            np.savez(path, times=np.array([0., 1.]), n=np.ones((2, 8)),
                     g=np.full((2, 8), 3.0))
            metadata = {"family": "gaussian", "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            path.with_suffix(".json").write_text(json.dumps(metadata))
            case, = iter_cases(root)
            self.assertEqual(case["format"], "legacy-flat-npz")
            np.testing.assert_array_equal(case["g"], 3.)
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                list(iter_cases(root))

    def test_read_only_store_never_initializes_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(ValueError):
                CheckpointStore(missing, {}, read_only=True)
            self.assertFalse(missing.exists())
            store = CheckpointStore(Path(directory), {})
            reader = CheckpointStore(store.directory, {}, read_only=True)
            with self.assertRaises(PermissionError):
                reader.initialize({"f": np.ones(2)}, {"fields": np.ones((1, 2))})


if __name__ == "__main__":
    unittest.main()
