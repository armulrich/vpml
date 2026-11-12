"""Reviewed source transitions preserve real CPU trajectories and their origins."""
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

from vpml.datasets import continuation, runner
from vpml.datasets.checkpoints import CheckpointStore
from vpml.datasets.manifest import Numerics, build_manifest, write_manifest
from vpml.datasets.reader import read_case
from vpml.diagnostics.dataset_accuracy import compare_dataset_cases
from vpml.io.serialization import json_digest, read_json


def source_record(label="parent", **changed):
    """Use valid dependency hashes while keeping numerical kernels identical."""
    files = {
        "vpml/__init__.py": hashlib.sha256(label.encode()).hexdigest(),
        "vpml/datasets/runner.py": hashlib.sha256(b"runner").hexdigest(),
        "vpml/datasets/evolution.py": hashlib.sha256(b"same evolution").hexdigest(),
        "vpml/datasets/observations.py": hashlib.sha256(b"same moments").hexdigest(),
        "vpml/datasets/schema.py": hashlib.sha256(b"same schema").hexdigest(),
        "vpml/solvers/physical_grid.py": hashlib.sha256(b"same solver").hexdigest(),
        "vpml/_jax_precision.py": hashlib.sha256(b"same precision").hexdigest(),
        "vpml/jax_runtime.py": hashlib.sha256(b"same runtime setup").hexdigest(),
    }
    files.update(changed)
    return {"sha256": json_digest(files), "files": files,
            "compatibility": {"format": 2, "files": files,
                              "sha256": json_digest(files)}}


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

    def run_worker(self, source, *, root=None, stop_after_block=False, **options):
        root = self.dataset if root is None else root
        stopped = [False]
        append = CheckpointStore.append

        def append_and_stop(store, *args, **kwargs):
            progress = append(store, *args, **kwargs)
            stopped[0] = True
            return progress

        with mock.patch.object(runner, "source_record", return_value=source), \
                mock.patch.object(CheckpointStore, "append",
                                  append_and_stop if stop_after_block else append), \
                redirect_stdout(io.StringIO()):
            return runner.run_dataset(
                self.manifest, root, backend="cpu", max_seconds=60,
                stop_requested=lambda: stopped[0], **options,
            )

    def prepare(self, current=None, **options):
        source = self.current if current is None else current
        with mock.patch.object(continuation, "source_record", return_value=source):
            return continuation.prepare_transition(
                self.dataset, reason="Reviewed import cleanup; numerical kernels unchanged",
                allow_source_changes=options.pop("allow_source_changes", ["vpml/__init__.py"]),
                **options,
            )

    def transition_path(self, record=None):
        record = self.prepare() if record is None else record
        path = self.root / "reviewed-transition.json"
        continuation.write_transition(path, record)
        return path

    def progress(self, case_id=None):
        return read_json(self.dataset / "cases" / (case_id or self.case_id) / "progress.json")

    def case_files(self):
        return {str(path.relative_to(self.dataset)): path.read_bytes()
                for path in (self.dataset / "cases").rglob("*") if path.is_file()}

    def test_prepare_requires_exact_review_and_never_authorizes_numerical_changes(self):
        self.run_worker(self.parent, stop_after_block=True)
        before = self.case_files()
        for allowed in ([], ["vpml/__init__.py", "vpml/datasets/runner.py"],
                        ["vpml/__init__.py", "vpml/__init__.py"], ["../vpml/__init__.py"]):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                self.prepare(allow_source_changes=allowed)
        for path in ("vpml/datasets/evolution.py", "vpml/datasets/observations.py",
                     "vpml/datasets/schema.py", "vpml/solvers/physical_grid.py",
                     "vpml/_jax_precision.py", "vpml/jax_runtime.py"):
            changed = source_record("cleaned", **{path: "f" * 64})
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "protected"):
                self.prepare(changed, allow_source_changes=["vpml/__init__.py", path])
        record = self.prepare()
        self.assertEqual(record["from_source"], self.parent["compatibility"])
        self.assertEqual(record["to_source"], self.current["compatibility"])
        self.assertEqual(self.case_files(), before)
        self.assertFalse((self.dataset / "source-transitions").exists())

    def test_record_rejects_missing_changes_stale_targets_and_wrong_manifest(self):
        self.run_worker(self.parent, stop_after_block=True)
        record = self.prepare()
        changed = copy.deepcopy(record)
        changed["changes"] = {}
        with self.assertRaisesRegex(ValueError, "every exact"):
            continuation.validate_transition(changed)
        with self.assertRaisesRegex(ValueError, "current source"):
            continuation.validate_transition(record, current_source=source_record("later")["compatibility"])
        with self.assertRaisesRegex(ValueError, "different manifest"):
            continuation.validate_transition(record, manifest_sha256="a" * 64)
        changed = copy.deepcopy(record)
        changed["reason"] = " "
        with self.assertRaisesRegex(ValueError, "Invalid"):
            continuation.validate_transition(changed)
        changed = copy.deepcopy(record)
        changed["to_source"]["files"]["vpml/__init__.py"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "source compatibility"):
            continuation.validate_transition(changed)

    def test_transition_publication_is_immutable_and_idempotent(self):
        self.run_worker(self.parent, stop_after_block=True)
        record = self.prepare()
        path = self.transition_path(record)
        content = path.read_bytes()
        self.assertEqual(continuation.write_transition(path, record), json_digest(record))
        self.assertEqual(path.read_bytes(), content)
        changed = dict(record, reason="A different review")
        with self.assertRaisesRegex(ValueError, "differs"):
            continuation.write_transition(path, changed)
        link = self.root / "transition-link.json"
        link.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "symlink"):
            continuation.write_transition(link, record)

    def test_multiple_parent_sources_require_explicit_selection(self):
        self.manifest = tiny_manifest((.1, .3))
        first, second = [case["case_id"] for case in self.manifest["cases"]]
        self.run_worker(self.parent, case_ids=[first], stop_after_block=True)
        intermediate = source_record("intermediate")
        self.run_worker(intermediate, case_ids=[second], stop_after_block=True)
        with self.assertRaisesRegex(ValueError, "one previous source"):
            self.prepare()
        selected = self.prepare(from_source_sha256=self.parent["compatibility"]["sha256"])
        self.assertEqual(selected["from_source"], self.parent["compatibility"])
        with self.assertRaisesRegex(ValueError, "does not occur"):
            self.prepare(from_source_sha256="0" * 64)

    def test_actual_resume_is_identical_and_preserves_original_protocol_and_history(self):
        baseline = self.root / "baseline"
        self.run_worker(self.parent, root=baseline)
        self.run_worker(self.parent, stop_after_block=True)
        directory = self.dataset / "cases" / self.case_id
        protocol_bytes = (directory / "protocol.json").read_bytes()
        initial = self.progress()
        original_blocks = copy.deepcopy(initial["blocks"])
        old_observations = {block["observations"]["file"]:
                            (directory / block["observations"]["file"]).read_bytes()
                            for block in original_blocks}
        old_provenance = {path.name: path.read_bytes()
                          for path in (self.dataset / "provenance").glob("*.json")}
        transition = self.transition_path()
        result = self.run_worker(self.current, source_transition=transition)
        self.assertEqual(result["outcomes"][0]["status"], "complete")
        self.assertEqual((directory / "protocol.json").read_bytes(), protocol_bytes)
        final = self.progress()
        self.assertEqual(final["blocks"][:len(original_blocks)], original_blocks)
        for name, content in old_observations.items():
            self.assertEqual((directory / name).read_bytes(), content)
        for name, content in old_provenance.items():
            self.assertEqual((self.dataset / "provenance" / name).read_bytes(), content)
        expected, actual = read_case(baseline, self.case_id), read_case(self.dataset, self.case_id)
        for name in ("times", "fields", "raw_moments", "electric"):
            np.testing.assert_array_equal(actual[name], expected[name])
        protocol = read_json(directory / "protocol.json")
        latest_digest = final["outputs"]["provenance_sha256"]
        latest = continuation.validate_provenance(self.dataset, latest_digest, protocol)
        self.assertEqual(latest["source"], self.current)
        self.assertEqual(latest["source_transition_sha256"], json_digest(read_json(transition)))
        self.assertNotEqual(latest_digest, original_blocks[0]["observations"]["provenance_sha256"])
        comparison = compare_dataset_cases(self.dataset, baseline, [self.case_id],
                                           representation="observations-float64")
        self.assertTrue(comparison["all_tolerances_passed"])

    def test_repeated_resume_skips_complete_cases_and_starts_pending_cases(self):
        self.manifest = tiny_manifest((.1, .3, .5))
        first, second, third = [case["case_id"] for case in self.manifest["cases"]]
        self.run_worker(self.parent, case_ids=[first])
        first_progress = self.progress(first)
        self.run_worker(self.parent, case_ids=[second], stop_after_block=True)
        transition = self.transition_path()
        paused = self.run_worker(self.current, source_transition=transition, stop_after_block=True)
        self.assertTrue(paused["outcomes"][0]["reused"])
        self.assertEqual(paused["outcomes"][1]["status"], "partial")
        self.assertEqual(self.progress(second)["step"], 4)
        self.assertFalse((self.dataset / "cases" / third).exists())
        final = self.run_worker(self.current, source_transition=transition)
        self.assertEqual([row["status"] for row in final["outcomes"]], ["complete"] * 3)
        self.assertEqual(self.progress(first), first_progress)
        for case_id, expected_source in ((first, self.parent), (second, self.parent),
                                         (third, self.current)):
            protocol = read_json(self.dataset / "cases" / case_id / "protocol.json")
            self.assertEqual(protocol["source"], expected_source["compatibility"])
            self.assertTrue(read_case(self.dataset, case_id)["output_integrity_verified"])
        with mock.patch("vpml.datasets.evolution.ReferenceEvolution",
                        side_effect=AssertionError("complete cases evolved again")):
            again = self.run_worker(self.current, source_transition=transition)
        self.assertTrue(all(row["reused"] for row in again["outcomes"]))

    def test_missing_stale_transition_and_changed_runtime_cannot_resume(self):
        self.run_worker(self.parent, stop_after_block=True)
        before = self.case_files()
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_worker(self.current)
        transition = self.transition_path()
        with self.assertRaisesRegex(ValueError, "current source"):
            self.run_worker(source_record("later"), source_transition=transition)
        protocol = read_json(self.dataset / "cases" / self.case_id / "protocol.json")
        runtime = dict(protocol["runtime"], devices=["a different device"])
        with mock.patch.object(runner, "runtime_record", return_value=runtime), \
                self.assertRaisesRegex(ValueError, "runtime changed"):
            self.run_worker(self.current, source_transition=transition)
        self.assertEqual(self.case_files(), before)

    def test_wrong_transition_digest_rejects_before_runtime_initialization(self):
        self.run_worker(self.parent, stop_after_block=True)
        transition = self.transition_path()
        before = self.case_files()
        with mock.patch.object(runner, "runtime_record",
                               side_effect=AssertionError("runtime initialized before identity check")), \
                self.assertRaisesRegex(ValueError, "checksum"):
            self.run_worker(self.current, source_transition=transition,
                            source_transition_sha256="0" * 64)
        self.assertEqual(self.case_files(), before)

    def test_old_bundle_provenance_is_checked_before_resuming(self):
        self.run_worker(self.parent, stop_after_block=True)
        transition = self.transition_path()
        progress = self.progress()
        digest = progress["blocks"][0]["observations"]["provenance_sha256"]
        path = self.dataset / "provenance" / f"{digest}.json"
        provenance = read_json(path)
        path.write_text(json.dumps(dict(provenance, runtime={"backend": "forged"})))
        before = self.case_files()
        with mock.patch("vpml.datasets.evolution.ReferenceEvolution",
                        side_effect=AssertionError("unverified history reached evolution")), \
                self.assertRaisesRegex(ValueError, "provenance"):
            self.run_worker(self.current, source_transition=transition)
        self.assertEqual(self.case_files(), before)

    def test_reader_rejects_tampered_transition_and_missing_authorization(self):
        self.run_worker(self.parent, stop_after_block=True)
        transition = self.transition_path()
        self.run_worker(self.current, source_transition=transition)
        progress = self.progress()
        protocol = read_json(self.dataset / "cases" / self.case_id / "protocol.json")
        digest = progress["outputs"]["provenance_sha256"]
        provenance = read_json(self.dataset / "provenance" / f"{digest}.json")
        transition_path = self.dataset / "source-transitions" / (
            provenance["source_transition_sha256"] + ".json")
        record = read_json(transition_path)
        transition_path.write_text(json.dumps(dict(record, reason="tampered")))
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_case(self.dataset, self.case_id)
        # A self-consistent provenance digest still cannot authorize a changed
        # source when the explicit transition reference has been removed.
        forged = {key: value for key, value in provenance.items()
                  if key != "source_transition_sha256"}
        forged_digest = json_digest(forged)
        (self.dataset / "provenance" / f"{forged_digest}.json").write_text(
            json.dumps(forged))
        with self.assertRaisesRegex(ValueError, "source changed"):
            continuation.validate_provenance(self.dataset, forged_digest, protocol)

    def test_protocol_changes_remain_rejected_with_valid_source_transition(self):
        self.run_worker(self.parent, stop_after_block=True)
        record = self.prepare()
        saved = read_json(self.dataset / "cases" / self.case_id / "protocol.json")
        expected = dict(saved, source=self.current["compatibility"])
        self.assertEqual(continuation.continuation_protocol(expected, saved, record), saved)
        for key, value in (("linearized", True), ("runtime", {}), ("observations", {}),
                           ("output_schema", "different"), ("numerics", {}),
                           ("case", {}), ("manifest_sha256", "a" * 64)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "protocol mismatch"):
                continuation.continuation_protocol(dict(expected, **{key: value}), saved, record)

    def test_metadata_preflight_rejects_missing_or_wrong_parent_and_corrupt_progress(self):
        self.run_worker(self.parent, stop_after_block=True)
        record = self.prepare()
        current = self.current["compatibility"]
        before = self.case_files()
        with self.assertRaisesRegex(ValueError, "source changed"):
            continuation.validate_existing_protocols(self.dataset, self.manifest, current)
        wrong = dict(record, from_source=source_record("wrong parent")["compatibility"])
        wrong["changes"] = continuation.changed_sources(wrong["from_source"], current)
        with self.assertRaisesRegex(ValueError, "original source"):
            continuation.validate_existing_protocols(self.dataset, self.manifest, current, wrong)
        self.assertEqual(continuation.validate_existing_protocols(
            self.dataset, self.manifest, current, record), 1)
        self.assertEqual(self.case_files(), before)
        progress_path = self.dataset / "cases" / self.case_id / "progress.json"
        progress = read_json(progress_path)
        progress_path.write_text(json.dumps(dict(progress, protocol_sha256="0" * 64)))
        with self.assertRaisesRegex(ValueError, "protocol/version mismatch"):
            continuation.validate_existing_protocols(self.dataset, self.manifest, current, record)

    def test_metadata_preflight_respects_case_selection_and_fresh_directories(self):
        current = self.current["compatibility"]
        self.assertEqual(continuation.validate_existing_protocols(
            self.dataset, self.manifest, current), 0)
        self.manifest = tiny_manifest((.1, .3))
        first, second = [case["case_id"] for case in self.manifest["cases"]]
        self.run_worker(self.parent, case_ids=[first], stop_after_block=True)
        self.run_worker(source_record("another parent"), case_ids=[second], stop_after_block=True)
        record = self.prepare(from_source_sha256=self.parent["compatibility"]["sha256"])
        self.assertEqual(continuation.validate_existing_protocols(
            self.dataset, self.manifest, current, record, case_ids=[first]), 1)
        with self.assertRaisesRegex(ValueError, "original source"):
            continuation.validate_existing_protocols(self.dataset, self.manifest, current, record)
        with self.assertRaisesRegex(ValueError, "existing dataset"):
            continuation.validate_existing_protocols(self.root / "fresh", self.manifest,
                                                    current, record)

    def test_metadata_preflight_accepts_only_unpublished_initial_metadata_writes(self):
        current = self.current["compatibility"]
        self.dataset.mkdir()
        manifest_temp = self.dataset / (".manifest.json." + "a" * 32 + ".tmp")
        manifest_temp.write_bytes(b"interrupted manifest write")
        self.assertEqual(continuation.validate_existing_protocols(
            self.dataset, self.manifest, current), 0)
        self.assertEqual(manifest_temp.read_bytes(), b"interrupted manifest write")
        self.assertFalse((self.dataset / "manifest.json").exists())

        write_manifest(self.dataset / "manifest.json", self.manifest)
        directory = self.dataset / "cases" / self.case_id
        directory.mkdir(parents=True)
        protocol_temp = directory / (".protocol.json." + "b" * 32 + ".tmp")
        protocol_temp.write_bytes(b"interrupted protocol write")
        self.assertEqual(continuation.validate_existing_protocols(
            self.dataset, self.manifest, current), 0)
        self.assertEqual(protocol_temp.read_bytes(), b"interrupted protocol write")
        self.assertFalse((directory / "protocol.json").exists())

    def test_metadata_preflight_rejects_other_initial_files_and_symlinks(self):
        current = self.current["compatibility"]
        for scope in ("manifest", "protocol"):
            for kind in ("unexpected", "bad_uuid", "directory", "symlink", "directory_symlink"):
                with self.subTest(scope=scope, kind=kind), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary) / "dataset"
                    root.mkdir()
                    directory = root
                    if scope == "protocol":
                        write_manifest(root / "manifest.json", self.manifest)
                        directory = root / "cases" / self.case_id
                        directory.mkdir(parents=True)
                    name = "." + scope + ".json." + "a" * 32 + ".tmp"
                    if kind == "unexpected":
                        name = "unexpected.json"
                    elif kind == "bad_uuid":
                        name = "." + scope + ".json.not-a-uuid.tmp"
                    path = directory / name
                    if kind == "directory":
                        path.mkdir()
                    elif kind == "symlink":
                        path.symlink_to(Path(temporary) / "missing")
                    elif kind == "directory_symlink":
                        directory.rmdir()
                        directory.symlink_to(Path(temporary) / "missing", target_is_directory=True)
                    else:
                        path.write_bytes(b"unpublished")
                    with self.assertRaisesRegex(ValueError, "no dataset manifest|no protocol"):
                        continuation.validate_existing_protocols(root, self.manifest, current)

    def test_linearized_preflight_requires_matching_mode_without_importing_jax(self):
        self.run_worker(self.parent, stop_after_block=True, linearized=True)
        record = self.prepare()
        transition = self.transition_path(record)
        current = self.current["compatibility"]
        before = self.case_files()
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            continuation.validate_existing_protocols(self.dataset, self.manifest, current, record)
        self.assertEqual(continuation.validate_existing_protocols(
            self.dataset, self.manifest, current, record, linearized=True), 1)
        base = ["validate", "--manifest", str(self.dataset / "manifest.json"),
                "--out", str(self.dataset), "--source-transition", str(transition)]
        program = """import json, sys
from vpml.datasets import provenance
provenance.source_record = lambda: json.loads(sys.argv[1])
from vpml.cli.gaussian_dataset import main
try:
    result = main(json.loads(sys.argv[2]))
except ValueError as error:
    assert 'jax' not in sys.modules
    print(str(error), file=sys.stderr)
    raise SystemExit(3)
assert 'jax' not in sys.modules
raise SystemExit(result)
"""
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        for options, expected_status, message in (
                ([], 3, "protocol mismatch"),
                (["--linearized"], 0, "existing_case_protocols_checked")):
            with self.subTest(options=options):
                result = subprocess.run(
                    [sys.executable, "-c", program, json.dumps(self.current), json.dumps(base + options)],
                    env=environment, text=True, capture_output=True, timeout=30,
                )
                self.assertEqual(result.returncode, expected_status, result.stderr)
                self.assertIn(message, result.stdout + result.stderr)
        self.assertEqual(self.case_files(), before)

    def test_cli_metadata_validation_never_imports_jax(self):
        self.run_worker(self.parent, stop_after_block=True)
        record = self.prepare()
        transition = self.transition_path(record)
        wrong = dict(record, from_source=source_record("wrong parent")["compatibility"])
        wrong["changes"] = continuation.changed_sources(wrong["from_source"], wrong["to_source"])
        wrong_path = self.root / "wrong-parent.json"
        continuation.write_transition(wrong_path, wrong)
        base = ["validate", "--manifest", str(self.dataset / "manifest.json"),
                "--out", str(self.dataset)]
        program = """import json, sys
from vpml.datasets import provenance
provenance.source_record = lambda: json.loads(sys.argv[1])
from vpml.cli.gaussian_dataset import main
try:
    result = main(json.loads(sys.argv[2]))
except ValueError as error:
    assert 'jax' not in sys.modules
    print(str(error), file=sys.stderr)
    raise SystemExit(3)
assert 'jax' not in sys.modules
raise SystemExit(result)
"""
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        for path, expected_status, message in (
                (None, 3, "source changed"),
                (wrong_path, 3, "original source"),
                (transition, 0, "existing_case_protocols_checked")):
            with self.subTest(path=path):
                arguments = base + (["--source-transition", str(path)] if path else [])
                result = subprocess.run(
                    [sys.executable, "-c", program, json.dumps(self.current), json.dumps(arguments)],
                    env=environment, text=True, capture_output=True, timeout=30,
                )
                self.assertEqual(result.returncode, expected_status, result.stderr)
                self.assertIn(message, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
