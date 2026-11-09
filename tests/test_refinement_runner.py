"""CPU checks for diagnostic continuation, failure retention and exact sampling."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from vpml.datasets.checkpoints import CheckpointStore
from vpml.datasets.evolution import ReferenceEvolution
from vpml.datasets.manifest import Numerics, build_manifest
from vpml.diagnostics import gaussian_refinement as refinement
from vpml.io.serialization import read_json


def configuration():
    numerical = Numerics(nx=16, nv=65, dt=.01, final_time=.04,
                         save_dt=.02, output_nx=8, block_steps=2, storage_dtype="float64")
    manifest = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2.,),
                              strengths=(.1,), draws_per_combination=1, numerics=numerical)
    baseline = numerical.to_dict()
    return {"schema_version": 1, "case": manifest["cases"][0],
            "amplitudes": manifest["amplitudes"], "baseline": "base",
            "variants": {"base": baseline, "half_dt": dict(baseline, dt=.005)},
            "comparisons": [["base", "half_dt"]]}


class RefinementRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = configuration()
        source = mock.patch.object(refinement, "source_record", return_value={
            "sha256": "test-source", "compatibility": {"sha256": "test-numerics"}})
        source.start()
        self.addCleanup(source.stop)

    def run_variant(self, root=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return refinement.run_variant(self.config, "base", root or self.root,
                                          max_seconds=60, **kwargs)

    def test_actual_cpu_resume_matches_uninterrupted_and_exact_sampling(self):
        full, paused = self.root / "full", self.root / "paused"
        self.assertEqual(self.run_variant(full)["status"], "completed")
        stopped = [False]
        append = CheckpointStore.append

        def stop_after_block(store, *args):
            progress = append(store, *args)
            stopped[0] = True
            return progress

        with mock.patch.object(CheckpointStore, "append", stop_after_block):
            outcome = self.run_variant(paused, stop_requested=lambda: stopped[0])
        self.assertEqual(outcome["status"], "partial")
        self.assertEqual(outcome["step"], 2)
        partial = refinement.read_variant(paused, "base")
        np.testing.assert_array_equal(partial["arrays"]["diagnostic_times"], [0, .01, .02])
        self.assertEqual(partial["state"]["f"].shape, (65, 16))
        self.assertEqual(self.run_variant(paused)["status"], "completed")
        a, b = (refinement.read_variant(path, "base") for path in (full, paused))
        np.testing.assert_array_equal(a["state"]["f"], b["state"]["f"])
        for key in a["arrays"]:
            np.testing.assert_array_equal(a["arrays"][key], b["arrays"][key])
        np.testing.assert_array_equal(a["arrays"]["times"], [0, .02, .04])
        directory = full / "variants/base"
        store = CheckpointStore(directory, read_json(directory / "protocol.json"), read_only=True)
        history = store.histories()
        np.testing.assert_array_equal(a["arrays"]["fields"], history["fields"][::2])
        self.assertEqual(a["report"], b["report"])
        self.assertFalse(a["report"]["training_qualified"])

    def test_finite_failure_continues_but_never_qualifies(self):
        advance = ReferenceEvolution.advance

        def failed_observation(evolution, state, steps):
            result, observations = advance(evolution, state, steps)
            observations = {key: np.asarray(value).copy() for key, value in observations.items()}
            observations["diagnostics"][:, 3] = -1e-5
            return result, observations

        with mock.patch.object(ReferenceEvolution, "advance", failed_observation):
            outcome = self.run_variant()
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(outcome["step"], 4)
        self.assertFalse(outcome["report"]["checks_passed"])
        self.assertIn("minimum_f", outcome["report"]["failures"])
        self.assertEqual(outcome["report"]["first_minimum_f_failure_time"], .01)
        saved = refinement.read_variant(self.root, "base")
        self.assertEqual(saved["report"], outcome["report"])
        self.assertFalse(saved["report"]["training_qualified"])

    def test_nonfinite_block_does_not_replace_last_finite_state(self):
        advance = ReferenceEvolution.advance

        def nonfinite_observation(evolution, state, steps):
            result, observations = advance(evolution, state, steps)
            observations = {key: np.asarray(value).copy() for key, value in observations.items()}
            observations["fields"][0, 0, 0] = np.nan
            return result, observations

        with mock.patch.object(ReferenceEvolution, "advance", nonfinite_observation):
            outcome = self.run_variant()
        self.assertEqual(outcome["status"], "partial")
        self.assertEqual(outcome["step"], 0)
        self.assertEqual(outcome["reason"], "nonfinite_evolution")
        saved = refinement.read_variant(self.root, "base")
        self.assertTrue(np.isfinite(saved["state"]["f"]).all())
        self.assertIn("nonfinite_evolution", saved["report"]["failures"])
        self.assertFalse(saved["report"]["checks_passed"])

    def test_invalid_cadence_pairs_and_dtype_are_rejected(self):
        for change in ("cadence", "dtype", "pair", "name", "domain"):
            with self.subTest(change=change):
                config = copy.deepcopy(self.config)
                if change == "cadence":
                    for settings in config["variants"].values():
                        settings.update(save_dt=.025, final_time=.1)
                elif change == "dtype":
                    config["variants"]["base"]["storage_dtype"] = "float32"
                elif change == "pair":
                    config["comparisons"] = [["base", "base"]]
                elif change == "name":
                    config["variants"]["../escape"] = config["variants"].pop("half_dt")
                else:
                    config["variants"]["half_dt"]["vmax"] = 9
                with self.assertRaises(ValueError):
                    refinement.validate_config(config)

    def test_configuration_and_source_changes_cannot_reuse_checkpoint(self):
        self.run_variant()
        self.config["variants"]["base"]["block_steps"] = 1
        with self.assertRaisesRegex(ValueError, "different refinement configuration"):
            self.run_variant()
        self.config = configuration()
        with mock.patch.object(refinement, "source_record", return_value={
                "sha256": "different-source", "compatibility": {"sha256": "different-numerics"}}):
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                self.run_variant()

    def test_completed_reuse_verifies_integrity_without_evolution(self):
        self.run_variant()
        with mock.patch.object(refinement, "ReferenceEvolution", side_effect=AssertionError("unexpected evolution")):
            self.assertTrue(self.run_variant()["reused"])
        directory = self.root / "variants/base"
        progress = read_json(directory / "progress.json")
        bundle = directory / progress["blocks"][0]["observations"]["file"]
        with bundle.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            refinement.read_variant(self.root, "base")

    def test_deadline_does_not_advance_and_preserves_original_state(self):
        times = iter([0., 61.])
        with mock.patch.object(ReferenceEvolution, "advance", side_effect=AssertionError("unexpected evolution")):
            outcome = self.run_variant(clock=lambda: next(times))
        self.assertEqual(outcome["step"], 0)
        self.assertEqual(outcome["reason"], "deadline")
        self.assertFalse(outcome["report"]["training_qualified"])


if __name__ == "__main__":
    unittest.main()
