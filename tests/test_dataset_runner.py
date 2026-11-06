"""Tiny CPU trajectory and failure tests for the dataset worker."""
from contextlib import redirect_stdout
import io
import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jax
import numpy as np

from vpml.datasets.checkpoints import CheckpointStore, atomic_json
from vpml.datasets.manifest import Numerics, build_manifest
from vpml.datasets.provenance import runtime_record
from vpml.datasets import runner


def tiny_manifest(**changes):
    numerical = dict(nx=32, nv=257, dt=0.01, final_time=0.04,
                     save_dt=0.02, output_nx=16, block_steps=2,
                     storage_dtype="float64")
    numerical.update(changes)
    return build_manifest(k0_values=(0.5,), mode_counts=(3,), decays=(2.0,),
                          strengths=(0.1,), draws_per_combination=1,
                          numerics=Numerics(**numerical))


def read_case(root, case_id):
    directory = root / "cases" / case_id
    protocol = json.loads((directory / "protocol.json").read_text())
    store = CheckpointStore(directory, protocol)
    state, progress = store.load()
    with np.load(directory / progress["outputs"]["file"], allow_pickle=False) as bundle:
        outputs = {key: bundle[key] for key in bundle.files}
    return state, outputs, progress


class DatasetRunnerTests(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = tiny_manifest()
        self.case_id = self.manifest["cases"][0]["case_id"]
        # Other development agents may edit source files during this test run.
        # Provenance enforcement itself has a separate negative test below.
        source = mock.patch.object(runner, "source_record", return_value={"sha256": "test-source", "compatibility": {"sha256": "test-numerics"}})
        source.start()
        self.addCleanup(source.stop)

    def run_worker(self, directory, **kwargs):
        with redirect_stdout(io.StringIO()):
            return runner.run_dataset(self.manifest, directory, backend="cpu",
                                      max_seconds=60, **kwargs)

    def test_actual_cpu_trajectory_resumes_without_changing_results(self):
        for linearized in (False, True):
            with self.subTest(linearized=linearized):
                uninterrupted = self.root / f"continuous-{linearized}"
                interrupted = self.root / f"resumed-{linearized}"
                self.assertEqual(self.run_worker(uninterrupted, linearized=linearized)
                                 ["outcomes"][0]["status"], "complete")
                stop = [False]
                append = CheckpointStore.append

                def append_and_stop(store, *args, **kwargs):
                    progress = append(store, *args, **kwargs)
                    stop[0] = True
                    return progress

                with mock.patch.object(CheckpointStore, "append", append_and_stop):
                    paused = self.run_worker(interrupted, linearized=linearized,
                                             stop_requested=lambda: stop[0])
                self.assertEqual(paused["outcomes"][0]["status"], "partial")
                progress_path = interrupted / "cases" / self.case_id / "progress.json"
                self.assertEqual(json.loads(progress_path.read_text())["step"], 2)
                self.assertTrue(paused["stopped"])
                resumed = self.run_worker(interrupted, linearized=linearized)
                self.assertEqual(resumed["outcomes"][0]["status"], "complete")
                expected_state, expected_outputs, expected_progress = read_case(uninterrupted, self.case_id)
                actual_state, actual_outputs, actual_progress = read_case(interrupted, self.case_id)
                self.assertEqual(expected_state.keys(), actual_state.keys())
                for key in expected_state:
                    np.testing.assert_array_equal(actual_state[key], expected_state[key])
                self.assertEqual(actual_outputs.keys(), expected_outputs.keys())
                for key in expected_outputs:
                    np.testing.assert_array_equal(actual_outputs[key], expected_outputs[key])
                self.assertEqual(actual_progress["report"], expected_progress["report"])
                self.assertFalse(actual_progress["report"]["training_qualified"])
                np.testing.assert_allclose(actual_outputs["times"], [0, .02, .04], rtol=0, atol=1e-16)
                self.assertEqual(actual_outputs["diagnostics"].shape[0], 5)

    def test_completed_case_is_verified_and_skipped_without_evolving_again(self):
        self.run_worker(self.root)
        with mock.patch("vpml.datasets.evolution.ReferenceEvolution", side_effect=AssertionError("unexpected evolution")):
            report = self.run_worker(self.root)
        self.assertTrue(report["outcomes"][0]["reused"])
        self.assertFalse(report["training_started"])
        status = runner.dataset_status(self.root)
        self.assertEqual(status["counts"], {"complete": 1, "partial": 0, "failed": 0, "pending": 0})
        self.assertFalse(status["integrity_verified"])

    def test_corrupted_history_prevents_reusing_completed_case(self):
        self.run_worker(self.root)
        directory = self.root / "cases" / self.case_id
        progress = json.loads((directory / "progress.json").read_text())
        history = directory / progress["blocks"][0]["observations"]["file"]
        history.write_bytes(history.read_bytes() + b"invalid")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_worker(self.root)
        self.assertFalse((self.root / "locks" / f"{self.case_id}.lock").exists())

    def test_changed_source_or_manifest_cannot_reuse_results(self):
        self.run_worker(self.root)
        with mock.patch.object(runner, "source_record", return_value={"sha256": "different-source", "compatibility": {"sha256": "changed-numerics"}}):
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                self.run_worker(self.root)
        self.manifest = tiny_manifest(final_time=0.06)
        with self.assertRaisesRegex(ValueError, "different manifest"):
            self.run_worker(self.root)

    def test_changed_runtime_cannot_reuse_results(self):
        self.run_worker(self.root)
        changed = dict(runtime_record("cpu"), devices=["different CPU runtime"])
        with mock.patch.object(runner, "runtime_record", return_value=changed):
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                self.run_worker(self.root)

    def test_unrelated_source_change_resumes_and_attributes_each_output_block(self):
        stopped = [False]
        append = CheckpointStore.append

        def append_and_stop(store, *args, **kwargs):
            result = append(store, *args, **kwargs)
            stopped[0] = True
            return result

        with mock.patch.object(CheckpointStore, "append", append_and_stop):
            self.assertEqual(self.run_worker(self.root, stop_requested=lambda: stopped[0])
                             ["outcomes"][0]["status"], "partial")
        changed_inventory = {"sha256": "changed-plot", "compatibility": {"sha256": "test-numerics"}}
        with mock.patch.object(runner, "source_record", return_value=changed_inventory):
            self.assertEqual(self.run_worker(self.root)["outcomes"][0]["status"], "complete")
        progress = json.loads((self.root / "cases" / self.case_id / "progress.json").read_text())
        first = progress["blocks"][0]["observations"]["provenance_sha256"]
        last = progress["blocks"][-1]["observations"]["provenance_sha256"]
        self.assertNotEqual(first, last)
        self.assertEqual(progress["outputs"]["provenance_sha256"], last)
        for digest in (first, last):
            record = json.loads((self.root / "provenance" / f"{digest}.json").read_text())
            self.assertEqual(runner.json_digest(record), digest)

    def test_deadline_stops_before_next_block_and_can_resume(self):
        readings = iter([0.0, 0.0, 61.0, 61.0])
        with mock.patch("vpml.datasets.evolution.ReferenceEvolution.advance",
                        side_effect=AssertionError("advanced after deadline")):
            report = self.run_worker(self.root, clock=lambda: next(readings))
        self.assertEqual(report["outcomes"][0]["status"], "partial")
        self.assertTrue(report["stopped"])
        progress_path = self.root / "cases" / self.case_id / "progress.json"
        self.assertEqual(json.loads(progress_path.read_text())["step"], 0)
        self.assertEqual(self.run_worker(self.root)["outcomes"][0]["status"], "complete")

    def test_committed_history_must_match_reported_solver_step(self):
        stopped = [False]
        append = CheckpointStore.append

        def append_and_stop(store, *args, **kwargs):
            result = append(store, *args, **kwargs)
            stopped[0] = True
            return result

        with mock.patch.object(CheckpointStore, "append", append_and_stop):
            self.run_worker(self.root, stop_requested=lambda: stopped[0])
        path = self.root / "cases" / self.case_id / "progress.json"
        progress = json.loads(path.read_text())
        # Integrity of files is unchanged, but the manifest claims fewer steps
        # than the verified observation history actually contains.
        progress["step"] = 1
        progress["states"][-1]["step"] = 1
        progress["blocks"][-1]["step"] = 1
        atomic_json(path, progress)
        with self.assertRaisesRegex(ValueError, "histories disagree"):
            self.run_worker(self.root)

    def test_finite_quality_failure_is_flagged_and_not_automatically_retried(self):
        from vpml.datasets.diagnostics import qualification_report

        def failed_report(*args, **kwargs):
            report = qualification_report(*args, **kwargs)
            report["checks_passed"] = False
            report["failures"] = ["relative_energy"]
            report["metrics"]["max_relative_total_energy_error"] = 0.2
            return report

        with mock.patch("vpml.datasets.diagnostics.qualification_report", side_effect=failed_report), \
             mock.patch("vpml.datasets.evolution.ReferenceEvolution.advance",
                        side_effect=AssertionError("advanced after numerical failure")):
            report = self.run_worker(self.root)
        self.assertEqual(report["outcomes"][0]["status"], "failed")
        failure_path = self.root / "failures" / f"{self.case_id}.json"
        failure = json.loads(failure_path.read_text())
        self.assertEqual(failure["reason"], "numerical_checks_failed")
        with mock.patch("vpml.datasets.evolution.ReferenceEvolution",
                        side_effect=AssertionError("retried failed case")):
            repeated = self.run_worker(self.root)
        self.assertEqual(repeated["outcomes"][0]["status"], "failed")
        self.assertEqual(runner.dataset_status(self.root)["counts"]["failed"], 1)

    def test_nonpositive_saved_fields_are_failed_even_when_fine_diagnostics_pass(self):
        original = runner.sampled_outputs

        def negative_saved_pressure(*args, **kwargs):
            outputs = original(*args, **kwargs)
            outputs["fields"] = outputs["fields"].copy()
            outputs["fields"][:, 2] = -np.abs(outputs["fields"][:, 2])
            return outputs

        with mock.patch.object(runner, "sampled_outputs", side_effect=negative_saved_pressure):
            report = self.run_worker(self.root)
        self.assertEqual(report["outcomes"][0]["status"], "failed")
        _, _, progress = read_case(self.root, self.case_id)
        self.assertEqual(progress["status"], "complete")
        self.assertFalse(progress["report"]["checks_passed"])
        self.assertIn("nonpositive_saved_pressure", progress["report"]["failures"])
        self.assertFalse(progress["report"]["training_qualified"])

    def test_completed_failed_report_recovers_missing_failure_marker(self):
        original_outputs = runner.sampled_outputs
        original_json = atomic_json

        def negative_saved_density(*args, **kwargs):
            outputs = original_outputs(*args, **kwargs)
            outputs["fields"] = outputs["fields"].copy()
            outputs["fields"][:, 0] = -np.abs(outputs["fields"][:, 0])
            return outputs

        def interrupt_failure_marker(path, value):
            if path.parent.name == "failures":
                raise RuntimeError("interrupted before failure marker")
            return original_json(path, value)

        with mock.patch.object(runner, "sampled_outputs", side_effect=negative_saved_density), \
             mock.patch("vpml.datasets.checkpoints.atomic_json", side_effect=interrupt_failure_marker):
            with self.assertRaisesRegex(RuntimeError, "failure marker"):
                self.run_worker(self.root)
        failure_path = self.root / "failures" / f"{self.case_id}.json"
        self.assertFalse(failure_path.exists())
        self.assertEqual(runner.dataset_status(self.root)["counts"]["failed"], 1)
        with mock.patch("vpml.datasets.evolution.ReferenceEvolution",
                        side_effect=AssertionError("recomputed completed trajectory")):
            report = self.run_worker(self.root)
        self.assertEqual(report["outcomes"][0]["status"], "failed")
        self.assertTrue(report["outcomes"][0]["reused"])
        self.assertEqual(json.loads(failure_path.read_text())["reason"], "saved_field_checks_failed")

    def test_another_owner_keeps_its_lock_and_case_is_reported_busy(self):
        lock_path = self.root / "locks" / f"{self.case_id}.lock"
        with runner.case_lock(lock_path):
            owner = lock_path.read_bytes()
            report = self.run_worker(self.root)
            self.assertEqual(report["outcomes"], [{"case_id": self.case_id, "status": "busy"}])
            self.assertEqual(lock_path.read_bytes(), owner)
        self.assertFalse(lock_path.exists())
        self.assertFalse((self.root / "cases" / self.case_id).exists())

    def test_completion_must_reach_requested_time_and_output_schema(self):
        self.run_worker(self.root)
        _, _, progress = read_case(self.root, self.case_id)
        numerics = Numerics(**self.manifest["numerics"])
        invalid = copy.deepcopy(progress)
        invalid["step"] -= 1
        with self.assertRaisesRegex(ValueError, "final solver step"):
            runner.validate_completed_progress(invalid, numerics, False)
        invalid = copy.deepcopy(progress)
        invalid["outputs"]["arrays"]["fields"]["shape"][0] -= 1
        with self.assertRaisesRegex(ValueError, "output schema"):
            runner.validate_completed_progress(invalid, numerics, False)

    def test_invalid_selection_or_budget_does_not_create_outputs(self):
        for seconds in (0, -1, float("inf"), float("nan")):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, "max_seconds"):
                runner.run_dataset(self.manifest, self.root / "absent", backend="cpu", max_seconds=seconds)
        for selected in ([], ["unknown"], [self.case_id, self.case_id]):
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, "unique known"):
                self.run_worker(self.root / "absent", case_ids=selected)
        self.assertFalse((self.root / "absent").exists())

    def test_gpu_requirement_cannot_silently_fall_back_to_cpu(self):
        with mock.patch.object(jax, "default_backend", return_value="cpu"):
            with self.assertRaisesRegex(RuntimeError, "Required gpu"):
                runtime_record("gpu")
            with self.assertRaisesRegex(RuntimeError, "Required gpu"):
                runner.run_dataset(self.manifest, self.root / "absent", backend="gpu", max_seconds=60)
        self.assertFalse((self.root / "absent").exists())

    def test_fractional_saved_times_interpolate_known_ramps(self):
        numerical = Numerics(nx=16, nv=33, output_nx=8, dt=.01,
                             final_time=.1, save_dt=.025, block_steps=2,
                             storage_dtype="float32")
        times = np.arange(11) * .01
        histories = {"fields": np.broadcast_to((2 + 3 * times)[:, None, None], (11, 5, 8)),
                     "raw_moments": np.broadcast_to((1 + 2 * times)[:, None, None], (11, 3, 8)),
                     "electric": np.broadcast_to((4 * times)[:, None], (11, 8)),
                     "diagnostics": times[:, None]}
        output = runner.sampled_outputs(histories, numerical)
        np.testing.assert_array_equal(output["times"], np.arange(5) * .025)
        np.testing.assert_allclose(output["fields"][:, 0, 0], 2 + 3 * output["times"], rtol=1e-7)
        np.testing.assert_allclose(output["raw_moments"][:, 0, 0], 1 + 2 * output["times"], rtol=1e-7)
        np.testing.assert_allclose(output["electric"][:, 0], 4 * output["times"], rtol=1e-7)
        np.testing.assert_array_equal(output["diagnostics"], histories["diagnostics"])
        self.assertEqual(output["fields"].dtype, np.float32)
        self.assertEqual(output["diagnostics"].dtype, np.float64)
        histories["fields"] = histories["fields"][:-1]
        with self.assertRaisesRegex(ValueError, "History length"):
            runner.sampled_outputs(histories, numerical)


if __name__ == "__main__":
    unittest.main()
