"""Recovery and integrity tests using small NumPy arrays, without GPU work."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from vpml.datasets import checkpoints
from vpml.datasets.checkpoints import CheckpointStore


def state(step):
    return {"f": np.full((3, 4), step + 1.0),
            "h": np.full((3, 4), step * 0.1)}


def observations(first, stop):
    time = np.arange(first, stop, dtype=np.float64)
    return {"fields": np.stack((time, time**2), axis=-1),
            "diagnostics": time[:, None].astype(np.float32)}


class CheckpointStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "case"
        self.protocol = {"case_id": "shape0001-lambda01", "dt": 0.01,
                         "solver_sha256": "fixture", "output_schema": 1}
        self.store = CheckpointStore(self.directory, self.protocol)

    def initialize(self):
        return self.store.initialize(state(0), observations(0, 1))

    def test_history_and_restart_match_uninterrupted_arrays(self):
        self.initialize()
        self.store.append(2, state(2), observations(1, 3))
        # Reconstructing a store imitates a new process, with no cached state.
        self.store = CheckpointStore(self.directory, self.protocol)
        restored, progress = self.store.load()
        self.assertEqual(progress["step"], 2)
        for key in restored:
            np.testing.assert_array_equal(restored[key], state(2)[key])
        self.store.append(4, state(4), observations(3, 5))
        for key, array in self.store.histories().items():
            np.testing.assert_array_equal(array, observations(0, 5)[key])
        progress = self.store.read_progress()
        self.assertEqual([entry["step"] for entry in progress["states"]], [2, 4])
        self.assertEqual(len(list(self.directory.glob("state-*.npz"))), 2)
        self.assertEqual(len(progress["blocks"]), 3)

    def test_initialization_retry_after_protocol_only_or_orphan_writes(self):
        real = checkpoints._atomic_json

        def fail_progress(path, value):
            if path.name == "progress.json":
                raise RuntimeError("interrupted before initial publication")
            return real(path, value)

        with mock.patch.object(checkpoints, "_atomic_json", side_effect=fail_progress):
            with self.assertRaisesRegex(RuntimeError, "initial publication"):
                self.initialize()
        self.assertFalse(self.store.progress_path.exists())
        self.store = CheckpointStore(self.directory, self.protocol)
        first = self.initialize()
        again = self.initialize()
        self.assertEqual(first, again)
        self.assertEqual(self.store.load()[1]["step"], 0)

    def test_append_failure_before_publication_keeps_previous_history(self):
        self.initialize()
        with mock.patch.object(checkpoints, "_atomic_json", side_effect=RuntimeError("stop")):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                self.store.append(2, state(2), observations(1, 3))
        self.store = CheckpointStore(self.directory, self.protocol)
        self.assertEqual(self.store.load()[1]["step"], 0)
        np.testing.assert_array_equal(self.store.histories()["fields"], observations(0, 1)["fields"])
        self.store.append(2, state(2), observations(1, 3))
        np.testing.assert_array_equal(self.store.histories()["fields"], observations(0, 3)["fields"])

    def test_failure_after_publication_resumes_new_checkpoint(self):
        self.initialize()
        real = checkpoints._atomic_json

        def publish_then_fail(path, value):
            real(path, value)
            raise RuntimeError("interrupted after publication")

        with mock.patch.object(checkpoints, "_atomic_json", side_effect=publish_then_fail):
            with self.assertRaisesRegex(RuntimeError, "after publication"):
                self.store.append(2, state(2), observations(1, 3))
        restored, progress = CheckpointStore(self.directory, self.protocol).load()
        self.assertEqual(progress["step"], 2)
        np.testing.assert_array_equal(restored["f"], state(2)["f"])

    def test_interrupted_finalization_can_be_retried(self):
        self.initialize()
        outputs = {"fields": self.store.histories()["fields"]}
        report = {"numerical_checks_pass": False, "refinement_status": "pending"}
        with mock.patch.object(checkpoints, "_atomic_json", side_effect=RuntimeError("stop")):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                self.store.finalize(outputs, report)
        self.store = CheckpointStore(self.directory, self.protocol)
        self.assertEqual(self.store.load()[1]["status"], "running")
        complete = self.store.finalize(outputs, report)
        self.assertEqual(complete["status"], "complete")
        self.assertFalse(complete["report"]["numerical_checks_pass"])
        self.assertEqual(self.store.finalize(outputs, report), complete)
        self.assertEqual(self.store.load()[1], complete)
        with self.assertRaisesRegex(ValueError, "differ"):
            self.store.finalize({"fields": outputs["fields"] + 1}, report)
        with self.assertRaisesRegex(ValueError, "completed"):
            self.store.append(2, state(2), observations(1, 3))

    def test_finalization_published_before_interruption_remains_complete(self):
        self.initialize()
        outputs = {"fields": self.store.histories()["fields"]}
        report = {"refinement_status": "pending"}
        real = checkpoints._atomic_json

        def publish_then_fail(path, value):
            real(path, value)
            raise RuntimeError("interrupted after final publication")

        with mock.patch.object(checkpoints, "_atomic_json", side_effect=publish_then_fail):
            with self.assertRaisesRegex(RuntimeError, "final publication"):
                self.store.finalize(outputs, report)
        store = CheckpointStore(self.directory, self.protocol)
        self.assertEqual(store.load()[1]["status"], "complete")
        self.assertEqual(store.finalize(outputs, report)["status"], "complete")

    def test_committed_history_and_final_outputs_are_checked(self):
        progress = self.initialize()
        path = self.directory / progress["blocks"][0]["observations"]["file"]
        original = path.read_bytes()
        path.write_bytes(original + b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.store.load()
        path.write_bytes(original)
        complete = self.store.finalize({"f": state(0)["f"]}, {})
        (self.directory / complete["outputs"]["file"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.store.load()

    def test_protocol_mismatch_and_unknown_files_fail_closed(self):
        self.initialize()
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            CheckpointStore(self.directory, dict(self.protocol, dt=0.02))
        (self.directory / "unrelated-results.npy").write_bytes(b"do not overwrite")
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            CheckpointStore(self.directory, self.protocol)
        self.assertEqual((self.directory / "unrelated-results.npy").read_bytes(), b"do not overwrite")

    def test_changed_protocol_and_missing_state_are_rejected(self):
        progress = self.initialize()
        protocol_path = self.directory / "protocol.json"
        checkpoints.atomic_json(protocol_path, dict(self.protocol, dt=0.02))
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            self.store.load()
        checkpoints.atomic_json(protocol_path, self.protocol)
        snapshot = self.directory / progress["states"][-1]["state"]["file"]
        snapshot.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.store.load()

    def test_schema_metadata_and_resume_before_append_are_verified(self):
        progress = self.initialize()
        progress["blocks"][0]["observations"]["arrays"]["fields"]["shape"] = [1, 7]
        checkpoints.atomic_json(self.store.progress_path, progress)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            self.store.load()
        resumed = CheckpointStore(self.directory, self.protocol)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            resumed.append(2, state(2), observations(1, 3))

    def test_partial_state_write_leaves_previous_publication_readable(self):
        self.initialize()
        real = checkpoints._atomic_npz

        def fail_after_state(path, arrays):
            real(path, arrays)
            if path.name.startswith("state-"):
                raise RuntimeError("state write interrupted")

        with mock.patch.object(checkpoints, "_atomic_npz", side_effect=fail_after_state):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.store.append(2, state(2), observations(1, 3))
        store = CheckpointStore(self.directory, self.protocol)
        self.assertEqual(store.load()[1]["step"], 0)
        store.append(2, state(2), observations(1, 3))
        self.assertEqual(store.load()[1]["step"], 2)

    def test_nonfinite_or_object_arrays_and_invalid_names_are_rejected(self):
        for bad in ({"f": np.array([np.nan])}, {"f": np.array([object()])},
                    {"../outside": np.ones(1)}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.store.initialize(bad, observations(0, 1))
        with self.assertRaisesRegex(ValueError, "sample axis"):
            self.store.initialize(state(0), {"fields": np.array(1.0)})

    def test_step_and_array_schema_cannot_change(self):
        self.initialize()
        for step in (0, -1, 1.5, True):
            with self.subTest(step=step), self.assertRaisesRegex(ValueError, "increase"):
                self.store.append(step, state(2), observations(1, 3))
        with self.assertRaisesRegex(ValueError, "state schema"):
            self.store.append(2, {"f": np.ones((5, 4))}, observations(1, 3))
        with self.assertRaisesRegex(ValueError, "Observation schema"):
            self.store.append(2, state(2), {"fields": np.ones((2, 3))})

    def test_each_observation_field_can_have_its_own_sample_cadence(self):
        self.initialize()
        chunk = observations(1, 3)
        chunk["diagnostics"] = np.arange(1, 5, dtype=np.float32)[:, None]
        self.store.append(4, state(4), chunk)
        histories = self.store.histories()
        self.assertEqual(histories["fields"].shape, (3, 2))
        self.assertEqual(histories["diagnostics"].shape, (5, 1))


if __name__ == "__main__":
    unittest.main()
