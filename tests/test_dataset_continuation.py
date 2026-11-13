"""Source-independent restart contracts preserve arrays and producer provenance."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import jax
import numpy as np

from vpml.datasets import compatibility, runner
from vpml.datasets.checkpoints import CheckpointStore
from vpml.datasets.evolution import ReferenceEvolution
from vpml.datasets.manifest import Numerics, build_manifest, write_manifest
from vpml.datasets.provenance import runtime_record
from vpml.datasets.reader import read_case
from vpml.datasets.schema import observation_schema
from vpml.diagnostics.dataset_accuracy import compare_dataset_cases
from vpml.io.serialization import json_digest, read_json


def source_record(label="parent"):
    files = {"vpml/__init__.py": hashlib.sha256(label.encode()).hexdigest(),
             "vpml/datasets/evolution.py": hashlib.sha256(b"same evolution").hexdigest()}
    return {"sha256": json_digest(files), "files": files}


def tiny_manifest(strengths=(.1,)):
    return build_manifest(
        k0_values=(.5,), mode_counts=(3,), decays=(2.,),
        strengths=strengths, draws_per_combination=1,
        numerics=Numerics(nx=32, nv=129, dt=.01, final_time=.06,
                          save_dt=.02, output_nx=16, block_steps=2,
                          storage_dtype="float64"),
    )


class DatasetContinuationTests(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "data"
        self.manifest = tiny_manifest()
        self.parent = source_record()
        self.current = source_record("cleaned")

    @property
    def case_id(self):
        return self.manifest["cases"][0]["case_id"]

    def run_worker(self, source=None, *, root=None, stop_after_block=False, **options):
        root = self.dataset if root is None else root
        stopped = [False]
        append = CheckpointStore.append

        def append_and_stop(store, *args, **kwargs):
            progress = append(store, *args, **kwargs)
            stopped[0] = True
            return progress

        with mock.patch.object(runner, "source_record", return_value=source or self.current), \
                mock.patch.object(CheckpointStore, "append",
                                  append_and_stop if stop_after_block else append), \
                redirect_stdout(io.StringIO()):
            return runner.run_dataset(
                self.manifest, root, backend="cpu", max_seconds=60,
                stop_requested=lambda: stopped[0], **options,
            )

    def legacy_checkpoint(self, *, linearized=False):
        """Write the original unversioned protocol, with real float64 CPU arrays."""
        self.dataset.mkdir()
        write_manifest(self.dataset / "manifest.json", self.manifest)
        case = self.manifest["cases"][0]
        runtime = runtime_record("cpu")
        source = dict(self.parent, compatibility=dict(self.parent, format=2))
        provenance = {"source": source, "runtime": runtime}
        digest = json_digest(provenance)
        (self.dataset / "provenance").mkdir()
        (self.dataset / "provenance" / f"{digest}.json").write_text(json.dumps(provenance))
        protocol = {"manifest_sha256": json_digest(self.manifest), "case": case,
                    "numerics": self.manifest["numerics"], "runtime": runtime,
                    "source": source["compatibility"], "linearized": linearized,
                    "output_schema": self.manifest["output_schema"],
                    "observations": observation_schema()}
        store = CheckpointStore(self.dataset / "cases" / self.case_id, protocol,
                                provenance_sha256=digest)
        evolution = ReferenceEvolution(case, Numerics(**self.manifest["numerics"]),
                                       self.manifest["amplitudes"], linearized=linearized)
        state = evolution.initial_state()
        first = {key: np.asarray(value)[None] for key, value in evolution.observe(state).items()}
        store.initialize({key: np.asarray(value) for key, value in state.items()}, first)
        state, values = evolution.advance(state, 2)
        store.append(2, {key: np.asarray(value) for key, value in state.items()},
                     {key: np.asarray(value) for key, value in values.items()})

    def progress(self, case_id=None):
        return read_json(self.dataset / "cases" / (case_id or self.case_id) / "progress.json")

    def case_files(self):
        return {str(path.relative_to(self.dataset)): path.read_bytes()
                for path in (self.dataset / "cases").rglob("*") if path.is_file()}

    def test_old_to_new_resume_is_bitwise_identical_and_preserves_original_records(self):
        for linearized in (False, True):
            with self.subTest(linearized=linearized), tempfile.TemporaryDirectory() as directory:
                self.dataset = Path(directory) / "data"
                baseline = Path(directory) / "baseline"
                self.run_worker(root=baseline, linearized=linearized)
                self.legacy_checkpoint(linearized=linearized)
                case_directory = self.dataset / "cases" / self.case_id
                protocol_bytes = (case_directory / "protocol.json").read_bytes()
                original = self.progress()
                original_blocks = copy.deepcopy(original["blocks"])
                old_observations = {block["observations"]["file"]:
                                    (case_directory / block["observations"]["file"]).read_bytes()
                                    for block in original_blocks}
                old_provenance = {path.name: path.read_bytes()
                                  for path in (self.dataset / "provenance").glob("*.json")}
                result = self.run_worker(linearized=linearized)
                self.assertEqual(result["outcomes"][0]["status"], "complete")
                self.assertEqual((case_directory / "protocol.json").read_bytes(), protocol_bytes)
                final = self.progress()
                self.assertEqual(final["protocol_sha256"], original["protocol_sha256"])
                self.assertEqual(final["blocks"][:len(original_blocks)], original_blocks)
                for name, content in old_observations.items():
                    self.assertEqual((case_directory / name).read_bytes(), content)
                for name, content in old_provenance.items():
                    self.assertEqual((self.dataset / "provenance" / name).read_bytes(), content)
                expected, actual = read_case(baseline, self.case_id), read_case(self.dataset, self.case_id)
                for name in ("times", "fields", "raw_moments", "electric") + (("linear_electric",) if linearized else ()):
                    np.testing.assert_array_equal(actual[name], expected[name])
                latest = compatibility.validate_provenance(self.dataset,
                    final["outputs"]["provenance_sha256"], read_json(case_directory / "protocol.json"))
                self.assertEqual(latest["source"], self.current)
                self.assertEqual(latest["solver_revision"], compatibility.SOLVER_REVISION)
                comparison = compare_dataset_cases(self.dataset, baseline, [self.case_id],
                                                   representation="observations-float64")
                self.assertTrue(comparison["all_tolerances_passed"])

    def test_repeated_source_changes_resume_partial_and_skip_completed_cases(self):
        self.manifest = tiny_manifest((.1, .3, .5))
        first, second, third = [case["case_id"] for case in self.manifest["cases"]]
        self.run_worker(self.parent, case_ids=[first])
        first_progress = self.progress(first)
        self.run_worker(self.parent, case_ids=[second], stop_after_block=True)
        paused = self.run_worker(stop_after_block=True)
        self.assertTrue(paused["outcomes"][0]["reused"])
        self.assertEqual(paused["outcomes"][1]["status"], "partial")
        self.assertEqual(self.progress(second)["step"], 4)
        self.assertFalse((self.dataset / "cases" / third).exists())
        final = self.run_worker(source_record("another cleanup"))
        self.assertEqual([row["status"] for row in final["outcomes"]], ["complete"] * 3)
        self.assertEqual(self.progress(first), first_progress)
        for case_id in (first, second, third):
            self.assertTrue(read_case(self.dataset, case_id)["output_integrity_verified"])
        with mock.patch("vpml.datasets.evolution.ReferenceEvolution",
                        side_effect=AssertionError("complete cases evolved again")):
            again = self.run_worker()
        self.assertTrue(all(row["reused"] for row in again["outcomes"]))

    def test_contract_runtime_and_scientific_changes_are_rejected(self):
        self.legacy_checkpoint()
        saved = read_json(self.dataset / "cases" / self.case_id / "protocol.json")
        expected = compatibility.build_protocol(self.manifest, self.manifest["cases"][0], saved["runtime"])
        self.assertIs(compatibility.resume_protocol(expected, saved), saved)
        for key, value in (("protocol_version", 2), ("checkpoint_version", 2),
                           ("solver_revision", "unknown"), ("linearized", True),
                           ("runtime", dict(saved["runtime"], devices=["different device"])),
                           ("observations", {}), ("output_schema", "different"),
                           ("numerics", dict(saved["numerics"], nx=64)),
                           ("case", {}), ("manifest_sha256", "a" * 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                compatibility.resume_protocol(dict(expected, **{key: value}), saved)
        for key in expected:
            missing = {name: value for name, value in expected.items() if name != key}
            with self.subTest(missing=key), self.assertRaises(ValueError):
                compatibility.resume_protocol(missing, saved)
        malformed = copy.deepcopy(saved)
        malformed["source"]["files"]["vpml/__init__.py"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "source inventory"):
            compatibility.resume_protocol(expected, malformed)
        before = self.case_files()
        with mock.patch.object(runner, "runtime_record", return_value=dict(saved["runtime"], devices=["other"])), \
                self.assertRaisesRegex(ValueError, "runtime changed"):
            self.run_worker()
        self.assertEqual(self.case_files(), before)

    def test_old_bundle_provenance_checked_before_resume(self):
        self.legacy_checkpoint()
        digest = self.progress()["blocks"][0]["observations"]["provenance_sha256"]
        path = self.dataset / "provenance" / f"{digest}.json"
        provenance = read_json(path)
        path.write_text(json.dumps(dict(provenance, runtime={})))
        before = self.case_files()
        with mock.patch("vpml.datasets.evolution.ReferenceEvolution",
                        side_effect=AssertionError("unverified history reached evolution")), \
                self.assertRaisesRegex(ValueError, "provenance"):
            self.run_worker()
        self.assertEqual(self.case_files(), before)

    def test_historical_provenance_must_match_its_original_protocol(self):
        self.legacy_checkpoint()
        protocol = read_json(self.dataset / "cases" / self.case_id / "protocol.json")
        source = dict(self.current, compatibility=dict(self.current, format=2))
        provenance = {"source": source, "runtime": protocol["runtime"]}
        digest = json_digest(provenance)
        (self.dataset / "provenance" / f"{digest}.json").write_text(json.dumps(provenance))
        with self.assertRaisesRegex(ValueError, "original protocol"):
            compatibility.validate_provenance(self.dataset, digest, protocol)

    def test_self_consistent_provenance_requires_supported_contract_and_valid_inventory(self):
        self.run_worker()
        protocol = read_json(self.dataset / "cases" / self.case_id / "protocol.json")
        digest = self.progress()["outputs"]["provenance_sha256"]
        original = read_json(self.dataset / "provenance" / f"{digest}.json")
        alternatives = []
        for key in ("protocol_version", "checkpoint_version", "solver_revision"):
            alternatives.append(dict(original, **{key: "unknown"}))
            alternatives.append({name: value for name, value in original.items() if name != key})
        alternatives.append(dict(original, source={"sha256": "0" * 64, "files": self.parent["files"]}))
        for provenance in alternatives:
            digest = json_digest(provenance)
            (self.dataset / "provenance" / f"{digest}.json").write_text(json.dumps(provenance))
            with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                compatibility.validate_provenance(self.dataset, digest, protocol)

    def test_metadata_preflight_validates_contract_mode_and_progress_without_jax(self):
        self.legacy_checkpoint(linearized=True)
        before = self.case_files()
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            compatibility.validate_existing_protocols(self.dataset, self.manifest)
        self.assertEqual(compatibility.validate_existing_protocols(
            self.dataset, self.manifest, linearized=True), 1)
        self.assertEqual(self.case_files(), before)
        program = """import sys
from pathlib import Path
from vpml.datasets.compatibility import validate_existing_protocols
from vpml.datasets.manifest import read_manifest
root=Path(sys.argv[1])
assert validate_existing_protocols(root, read_manifest(root/'manifest.json'), linearized=True)==1
assert 'jax' not in sys.modules
"""
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        result = subprocess.run([sys.executable, "-c", program, str(self.dataset)],
                                env=environment, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        path = self.dataset / "cases" / self.case_id / "progress.json"
        progress = read_json(path)
        path.write_text(json.dumps(dict(progress, protocol_sha256="0" * 64)))
        with self.assertRaisesRegex(ValueError, "protocol/version mismatch"):
            compatibility.validate_existing_protocols(self.dataset, self.manifest, linearized=True)

    def test_metadata_preflight_accepts_only_unpublished_initial_metadata_writes(self):
        self.assertEqual(compatibility.validate_existing_protocols(self.dataset, self.manifest), 0)
        self.dataset.mkdir()
        temporary = self.dataset / (".manifest.json." + "a" * 32 + ".tmp")
        temporary.write_bytes(b"interrupted manifest write")
        self.assertEqual(compatibility.validate_existing_protocols(self.dataset, self.manifest), 0)
        write_manifest(self.dataset / "manifest.json", self.manifest)
        directory = self.dataset / "cases" / self.case_id
        directory.mkdir(parents=True)
        temporary = directory / (".protocol.json." + "b" * 32 + ".tmp")
        temporary.write_bytes(b"interrupted protocol write")
        self.assertEqual(compatibility.validate_existing_protocols(self.dataset, self.manifest), 0)
        self.assertEqual(temporary.read_bytes(), b"interrupted protocol write")
        self.assertFalse((directory / "protocol.json").exists())
        (directory / "unexpected.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "no protocol"):
            compatibility.validate_existing_protocols(self.dataset, self.manifest)

    def test_preflight_rejects_unknown_contract_before_compute_and_respects_selection(self):
        self.manifest = tiny_manifest((.1, .3))
        first, second = [case["case_id"] for case in self.manifest["cases"]]
        self.run_worker(case_ids=[first], stop_after_block=True)
        self.run_worker(case_ids=[second], stop_after_block=True)
        path = self.dataset / "cases" / second / "protocol.json"
        protocol = read_json(path)
        path.write_text(json.dumps(dict(protocol, solver_revision="unknown")))
        self.assertEqual(compatibility.validate_existing_protocols(self.dataset, self.manifest, [first]), 1)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            compatibility.validate_existing_protocols(self.dataset, self.manifest)


if __name__ == "__main__":
    unittest.main()
