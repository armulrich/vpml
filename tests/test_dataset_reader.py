"""Read actual tiny CPU trajectories and historical files without mutation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vpml.datasets.checkpoints import CheckpointStore
from vpml.datasets.manifest import Numerics, build_manifest, write_manifest
from vpml.datasets.reader import iter_cases, read_case
from vpml.datasets.runner import run_dataset
from vpml.datasets.schema import DIAGNOSTIC_NAMES, LEGACY_OUTPUT_SCHEMA, observation_schema
from vpml.io.serialization import json_digest


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
            self.assertEqual(case["format"], "gaussian-moments")
            self.assertIsNone(case["eta_NL"])
            self.assertNotIn("final_f", case)
            for absent in ("diagnostics", "diagnostic_times", "min_full_distribution",
                           "max_negative_mass_fraction", "max_boundary_mass_fraction"):
                self.assertNotIn(absent, case)
            length = 2 * np.pi / case["k0"]
            np.testing.assert_allclose(case["mass"], length * case["raw_moments"][:, 0].mean(axis=1))
            np.testing.assert_allclose(case["total_energy"],
                                       .5 * length * case["raw_moments"][:, 2].mean(axis=1)
                                       + .5 * length * np.mean(case["E"] ** 2, axis=1))
            self.assertIn("retained electric-field", case["conservation_scope"])
            self.assertEqual(len(list(iter_cases(root))), 1)
            after = {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            case_dir = root / "cases" / case["id"]
            progress = json.loads((case_dir / "progress.json").read_text())
            output = case_dir / progress["outputs"]["file"]
            output.write_bytes(output.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_case(root, case["id"])

    def test_tangent_energy_is_computed_from_saved_linear_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2.,),
                                      strengths=(.1,), draws_per_combination=1,
                                      numerics=Numerics(nx=16, nv=64, output_nx=8,
                                                       dt=.01, final_time=.02, save_dt=.01))
            run_dataset(manifest, root, backend="cpu", max_seconds=60, linearized=True)
            case = read_case(root, manifest["cases"][0]["case_id"])
            length = 2 * np.pi / case["k0"]
            expected = .5 * length * np.mean(case["linear_electric"] ** 2, axis=1)
            np.testing.assert_allclose(case["linear_energy"], expected)
            ratio = np.trapezoid(np.mean((case["E"] - case["linear_electric"]) ** 2, axis=1), case["times"])
            ratio /= np.trapezoid(np.mean(case["E"] ** 2, axis=1), case["times"])
            self.assertAlmostEqual(case["eta_NL"], np.sqrt(ratio))

    def test_previous_checkpoint_format_preserves_saved_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            numerics = Numerics(nx=16, nv=64, output_nx=8, dt=.01, final_time=.02, save_dt=.01,
                                storage_dtype="float64")
            manifest = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2.,),
                                      strengths=(.1,), draws_per_combination=1, numerics=numerics)
            manifest["output_schema"] = LEGACY_OUTPUT_SCHEMA
            write_manifest(root / "manifest.json", manifest)
            case = manifest["cases"][0]
            protocol = {"manifest_sha256": json_digest(manifest), "case": case,
                        "numerics": manifest["numerics"], "output_schema": LEGACY_OUTPUT_SCHEMA,
                        "linearized": False, "observations": observation_schema(LEGACY_OUTPUT_SCHEMA)}
            store = CheckpointStore(root / "cases" / case["case_id"], protocol)
            diagnostics = np.zeros((3, len(DIAGNOSTIC_NAMES)))
            diagnostics[:, DIAGNOSTIC_NAMES.index("electric_energy")] = [3., 2., 1.]
            fields = np.ones((3, 5, 8))
            observations = {"fields": fields, "raw_moments": np.ones((3, 3, 8)),
                            "electric": np.zeros((3, 8)), "diagnostics": diagnostics}
            state = {"f": np.ones((64, 16))}
            store.initialize(state, {key: value[:1] for key, value in observations.items()})
            store.append(2, state, {key: value[1:] for key, value in observations.items()})
            outputs = {**observations, "times": np.arange(3) * .01,
                       "diagnostic_times": np.arange(3) * .01}
            report = {"checks_passed": True, "failures": [], "training_qualified": False,
                      "qualification_scope": "historical diagnostic checks",
                      "diagnostic_names": list(DIAGNOSTIC_NAMES),
                      "metrics": {"min_f": -.0001, "max_boundary_abs_fraction": .001}}
            store.finalize(outputs, report)
            loaded = read_case(root, case["case_id"])
            self.assertEqual(loaded["format"], LEGACY_OUTPUT_SCHEMA)
            np.testing.assert_array_equal(loaded["full_energy"], [3., 2., 1.])
            self.assertEqual(loaded["min_full_distribution"], -.0001)
            self.assertEqual(loaded["max_boundary_mass_fraction"], .001)
            self.assertNotIn("conservation_scope", loaded)

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
