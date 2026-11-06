"""CLI routing tests: preparation and status never execute trajectories."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jax

from vpml.cli.gaussian_dataset import main, read_parameters
from vpml.datasets.manifest import Numerics, build_manifest, write_manifest


class DatasetCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def invoke(self, argv):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(argv)
        return status, json.loads(output.getvalue())

    def test_prepare_validate_and_status_do_not_query_devices_or_execute(self):
        manifest_path = self.root / "manifest.json"
        with mock.patch.object(jax, "devices", side_effect=AssertionError("device initialization")), \
             mock.patch.object(jax, "default_backend", side_effect=AssertionError("backend initialization")), \
             mock.patch("vpml.datasets.runner.run_dataset", side_effect=AssertionError("evolution")):
            status, prepared = self.invoke(["prepare", "--manifest", str(manifest_path)])
            self.assertEqual(status, 0)
            self.assertEqual(prepared["cases"], 1200)
            self.assertFalse(prepared["evolution_started"])
            original = manifest_path.read_bytes()
            self.invoke(["prepare", "--manifest", str(manifest_path)])
            self.assertEqual(manifest_path.read_bytes(), original)
            status, validated = self.invoke(["validate", "--manifest", str(manifest_path)])
            self.assertTrue(validated["valid"])
            self.assertEqual(validated["cases"], 1200)
            self.assertFalse(validated["evolution_started"])
            status, progress = self.invoke(["status", "--out", str(self.root)])
            self.assertEqual(progress["counts"]["pending"], 1200)
            self.assertFalse(progress["integrity_verified"])
            not_started = self.root / "not-started"
            _, progress = self.invoke(["status", "--out", str(not_started),
                                       "--manifest", str(manifest_path)])
            self.assertEqual(progress["counts"]["pending"], 1200)
            self.assertFalse(not_started.exists())

    def test_prepare_refuses_to_replace_a_different_manifest(self):
        path = self.root / "manifest.json"
        manifest = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2,),
                                  strengths=(.1,), draws_per_combination=1)
        write_manifest(path, manifest)
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Existing manifest differs"):
            self.invoke(["prepare", "--manifest", str(path)])
        self.assertEqual(path.read_bytes(), original)

    def test_validate_rejects_modified_scientific_parameters(self):
        path = self.root / "manifest.json"
        manifest = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2,),
                                  strengths=(.1,), draws_per_combination=1)
        manifest["cases"][0]["strength"] = .5
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.invoke(["validate", "--manifest", str(path)])

    def test_parameter_file_drives_preparation_and_validation(self):
        path = self.root / "manifest.json"
        parameters = self.root / "parameters.json"
        settings = {"k0_values": [.5], "mode_counts": [3], "decays": [2.0],
                    "strengths": [.1, .3], "draws_per_combination": 2}
        parameters.write_text(json.dumps(settings))
        status, prepared = self.invoke(["prepare", "--manifest", str(path),
                                        "--parameters", str(parameters)])
        self.assertEqual(prepared["cases"], 4)
        self.assertFalse(prepared["evolution_started"])
        status, verified = self.invoke(["validate", "--manifest", str(path),
                                       "--parameters", str(parameters)])
        self.assertTrue(verified["valid"])
        settings["strengths"] = [.1, .5]
        parameters.write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, "configured parameters"):
            self.invoke(["validate", "--manifest", str(path), "--parameters", str(parameters)])

    def test_parameters_reject_duplicates_nonfinite_values_and_nonobjects(self):
        path = self.root / "parameters.json"
        for text in ('{"draws_per_combination":1,"draws_per_combination":2}',
                     '{"strengths":[NaN]}', '{"strengths":[Infinity]}', '[1,2]'):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_parameters(path)

    def test_run_requires_explicit_backend_and_time_budget(self):
        common = ["run", "--manifest", "manifest.json", "--out", "outputs"]
        for extra in ([], ["--backend", "cpu"], ["--max-seconds", "60"]):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(common + extra)
                self.assertEqual(error.exception.code, 2)

    def test_run_routes_arguments_and_sets_backend_before_call(self):
        path = self.root / "manifest.json"
        manifest = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2,),
                                  strengths=(.1,), draws_per_combination=1,
                                  numerics=Numerics(nx=32, nv=257, output_nx=16,
                                                    final_time=.04, save_dt=.02, block_steps=2))
        write_manifest(path, manifest)
        case_id = manifest["cases"][0]["case_id"]

        def fake_run(received, output_root, **kwargs):
            self.assertEqual(received, manifest)
            self.assertEqual(output_root, self.root / "data")
            self.assertEqual(kwargs["backend"], "gpu")
            self.assertEqual(kwargs["max_seconds"], 60)
            self.assertEqual(kwargs["case_ids"], [case_id])
            self.assertTrue(kwargs["linearized"])
            self.assertFalse(kwargs["stop_requested"]())
            self.assertEqual(os.environ["JAX_PLATFORMS"], "cuda")
            self.assertEqual(os.environ["VPML_JAX_BACKEND"], "gpu")
            self.assertEqual(os.environ["JAX_ENABLE_X64"], "1")
            return {"outcomes": [{"case_id": case_id, "status": "failed"}],
                    "training_started": False}

        with mock.patch.dict(os.environ), \
             mock.patch("vpml.datasets.runner.run_dataset", side_effect=fake_run):
            status, report = self.invoke([
                "run", "--manifest", str(path), "--out", str(self.root / "data"),
                "--backend", "gpu", "--max-seconds", "60", "--case-id", case_id,
                "--linearized",
            ])
        self.assertEqual(status, 2)
        self.assertFalse(report["training_started"])
        self.assertFalse((self.root / "data").exists())


if __name__ == "__main__":
    unittest.main()
