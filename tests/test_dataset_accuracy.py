"""Known retained-field errors through real checksummed output readers; no evolution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vpml.datasets.checkpoints import CheckpointStore
from vpml.datasets.manifest import Numerics, build_manifest, write_manifest
from vpml.datasets.schema import DIAGNOSTIC_NAMES, OUTPUT_SCHEMA, observation_schema
from vpml.diagnostics.dataset_accuracy import compare_dataset_cases, main
from vpml.io.serialization import json_digest


def manifest(**changes):
    numerics = dict(nx=16, nv=16, dt=1.0, final_time=120.0, save_dt=1.0,
                    output_nx=8, block_steps=100, storage_dtype="float64")
    numerics.update(changes)
    return build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2.,),
                          strengths=(.1,), draws_per_combination=1,
                          numerics=Numerics(**numerics))


def fields_for(settings, factor=1.0):
    count = Numerics(**settings["numerics"]).nsamples
    wave = .1 * np.cos(2 * np.pi * np.arange(settings["numerics"]["output_nx"]) / 8)
    values = np.broadcast_to(factor * wave, (count, 6, len(wave))).copy()
    values[:, 0] += 1
    values[:, 2] += 1
    return values


def write_fixture(root, settings, values, *, times=None, complete=True, diagnostic_pass=True):
    """Publish synthetic arrays with the same protocol/provenance as real outputs."""
    write_manifest(root / "manifest.json", settings)
    numerical = Numerics(**settings["numerics"])
    case = settings["cases"][0]
    provenance = {"source": {"compatibility": {"sha256": "a" * 64}},
                  "runtime": {"backend": "synthetic-test"}}
    provenance_hash = json_digest(provenance)
    (root / "provenance").mkdir()
    (root / "provenance" / f"{provenance_hash}.json").write_text(json.dumps(provenance))
    protocol = {"manifest_sha256": json_digest(settings), "case": case,
                "numerics": settings["numerics"], "output_schema": OUTPUT_SCHEMA,
                "linearized": False, "observations": observation_schema(),
                "source": provenance["source"]["compatibility"], "runtime": provenance["runtime"]}
    store = CheckpointStore(root / "cases" / case["case_id"], protocol,
                            provenance_sha256=provenance_hash)
    diagnostics = np.zeros((numerical.nsteps + 1, len(DIAGNOSTIC_NAMES)))
    diagnostics[:, 0] = 1
    observations = {
        "fields": np.broadcast_to(values[0, :5], (numerical.nsteps + 1, 5, numerical.output_nx)).copy(),
        "raw_moments": np.zeros((numerical.nsteps + 1, 3, numerical.output_nx)),
        "electric": np.broadcast_to(values[0, 5], (numerical.nsteps + 1, numerical.output_nx)).copy(),
        "diagnostics": diagnostics,
    }
    state = {"f": np.ones((numerical.nv, numerical.nx))}
    store.initialize(state, {key: array[:1] for key, array in observations.items()})
    store.append(numerical.nsteps, state, {key: array[1:] for key, array in observations.items()})
    if complete:
        outputs = {"times": np.arange(numerical.nsamples) * numerical.save_dt if times is None else times,
                   "diagnostic_times": np.arange(numerical.nsteps + 1) * numerical.dt,
                   "fields": values[:, :5], "raw_moments": np.zeros((numerical.nsamples, 3, numerical.output_nx)),
                   "electric": values[:, 5], "diagnostics": diagnostics}
        report = {"diagnostic_names": list(DIAGNOSTIC_NAMES), "checks_passed": diagnostic_pass,
                  "training_qualified": False, "failures": [] if diagnostic_pass else ["minimum_f"],
                  "qualification_scope": "synthetic fixture only",
                  "metrics": {"min_f": 0., "max_boundary_abs_fraction": 0.}}
        store.finalize(outputs, report)
    return case["case_id"]


class DatasetAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.candidate, self.reference = self.root / "candidate", self.root / "reference"
        self.settings = manifest()
        self.case_id = self.settings["cases"][0]["case_id"]

    def compare(self):
        return compare_dataset_cases(self.candidate, self.reference, [self.case_id])

    def test_known_relative_error_uses_perturbations_and_allows_resolution_changes(self):
        candidate_settings = manifest(nx=32, nv=32, dt=.5)
        write_fixture(self.candidate, candidate_settings, fields_for(candidate_settings, 1.004))
        write_fixture(self.reference, self.settings, fields_for(self.settings))
        before = {str(path): path.stat().st_mtime_ns for path in self.root.rglob("*") if path.is_file()}
        result = self.compare()
        self.assertTrue(result["all_tolerances_passed"])
        self.assertFalse(result["training_qualified"])
        self.assertFalse(result["reference_uncertainty_assessed"])
        for window in result["cases"][0]["windows"]:
            for metric in window["fields"].values():
                self.assertAlmostEqual(metric["relative_L2"], .004)
                self.assertAlmostEqual(metric["reference_RMS"], .1 / np.sqrt(2))
                self.assertAlmostEqual(metric["absolute_RMS"], .0004 / np.sqrt(2))
                self.assertAlmostEqual(metric["absolute_max"], .0004)
        after = {str(path): path.stat().st_mtime_ns for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        json.dumps(result, allow_nan=False)

    def test_late_time_error_is_not_hidden_by_large_early_activity(self):
        reference = fields_for(self.settings)
        reference[:80, 1] *= 1000
        candidate = reference.copy()
        candidate[81:, 1] *= 1.02
        write_fixture(self.candidate, self.settings, candidate)
        write_fixture(self.reference, self.settings, reference)
        result = self.compare()
        windows = result["cases"][0]["windows"]
        self.assertTrue(windows[0]["fields"]["u"]["tolerance_passed"])
        # Keep the boundary sample small too, so the late window reflects its own scale.
        self.assertFalse(windows[-1]["fields"]["u"]["tolerance_passed"])
        self.assertFalse(result["all_tolerances_passed"])
        self.assertEqual(result["status"], "exceeds_tolerance")

    def test_zero_references_are_explicit_and_nonzero_errors_inconclusive(self):
        reference = fields_for(self.settings)
        reference[:, 3] = 0
        candidate = reference.copy()
        write_fixture(self.candidate, self.settings, candidate)
        write_fixture(self.reference, self.settings, reference)
        result = self.compare()
        metric = result["cases"][0]["windows"][0]["fields"]["Q"]
        self.assertEqual(metric["status"], "exact_zero_match")
        self.assertIsNone(metric["relative_L2"])
        self.assertIsNone(metric["max_per_time_relative_L2"])
        self.assertTrue(metric["tolerance_passed"])
        other = self.root / "different"
        candidate[100, 3] = 1e-20
        write_fixture(other, self.settings, candidate)
        result = compare_dataset_cases(other, self.reference, [self.case_id])
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["all_tolerances_passed"])
        json.dumps(result, allow_nan=False)

    def test_tiny_nonzero_reference_has_no_arbitrary_floor(self):
        reference = fields_for(self.settings)
        reference[:, 3] *= 1e-200
        candidate = reference.copy()
        candidate[:, 3] *= 1.01
        write_fixture(self.candidate, self.settings, candidate)
        write_fixture(self.reference, self.settings, reference)
        result = self.compare()
        metric = result["cases"][0]["windows"][0]["fields"]["Q"]
        self.assertAlmostEqual(metric["relative_L2"], .01)
        self.assertEqual(metric["status"], "exceeds_tolerance")

    def test_isolated_zero_reference_error_is_not_hidden_by_other_samples(self):
        reference = fields_for(self.settings)
        reference[60, 1] = 0
        candidate = reference.copy()
        candidate[60, 1] = 1e-20
        write_fixture(self.candidate, self.settings, candidate)
        write_fixture(self.reference, self.settings, reference)
        result = self.compare()
        metric = result["cases"][0]["windows"][0]["fields"]["u"]
        self.assertLess(metric["relative_L2"], .005)
        self.assertEqual(metric["per_time_zero_reference_nonzero_error_count"], 1)
        self.assertEqual(metric["per_time_nonzero_reference_count"], 120)
        self.assertEqual(metric["max_per_time_relative_L2"], 0.)
        self.assertEqual(result["status"], "inconclusive")

    def test_equal_windows_can_end_between_actual_saved_times(self):
        settings = manifest(final_time=121)
        write_fixture(self.candidate, settings, fields_for(settings, 1.004))
        write_fixture(self.reference, settings, fields_for(settings))
        result = self.compare()
        self.assertTrue(result["all_tolerances_passed"])
        for window in result["cases"][0]["windows"]:
            self.assertAlmostEqual(window["fields"]["u"]["relative_L2"], .004)

    def test_mismatched_initial_conditions_and_schema_are_rejected(self):
        write_fixture(self.reference, self.settings, fields_for(self.settings))
        different = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2.,),
                                   strengths=(.1,), draws_per_combination=1, base_seed=300,
                                   numerics=Numerics(**self.settings["numerics"]))
        write_fixture(self.candidate, different, fields_for(different))
        with self.assertRaisesRegex(ValueError, "initial conditions"):
            self.compare()
        broken = copy.deepcopy(self.settings)
        broken["output_schema"] = "wrong-schema"
        (self.candidate / "manifest.json").write_text(json.dumps(broken))
        with self.assertRaisesRegex(ValueError, "output_schema"):
            self.compare()

    def test_changed_physical_domain_or_saved_grid_is_rejected(self):
        write_fixture(self.reference, self.settings, fields_for(self.settings))
        for change in ({"vmax": 9}, {"save_dt": 2}, {"output_nx": 16}, {"final_time": 60}):
            with self.subTest(change=change):
                root = self.root / f"changed-{next(iter(change))}"
                settings = manifest(**change)
                write_fixture(root, settings, fields_for(settings))
                with self.assertRaisesRegex(ValueError, "only in nx, nv and dt"):
                    compare_dataset_cases(root, self.reference, [self.case_id])

    def test_time_coordinates_and_partial_completion_are_rejected(self):
        write_fixture(self.reference, self.settings, fields_for(self.settings))
        times = np.arange(121, dtype=np.float64)
        times[60] += .01
        write_fixture(self.candidate, self.settings, fields_for(self.settings), times=times)
        with self.assertRaisesRegex(ValueError, "time coordinates"):
            self.compare()
        partial = self.root / "partial"
        write_fixture(partial, self.settings, fields_for(self.settings), complete=False)
        with self.assertRaisesRegex(ValueError, "not complete"):
            compare_dataset_cases(partial, self.reference, [self.case_id])

    def test_corrupted_outputs_or_provenance_cannot_be_compared(self):
        write_fixture(self.reference, self.settings, fields_for(self.settings))
        write_fixture(self.candidate, self.settings, fields_for(self.settings))
        provenance, = (self.candidate / "provenance").glob("*.json")
        original = provenance.read_text()
        provenance.write_text(original.replace("synthetic-test", "modified"))
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.compare()
        provenance.write_text(original)
        directory = self.candidate / "cases" / self.case_id
        progress = json.loads((directory / "progress.json").read_text())
        output = directory / progress["outputs"]["file"]
        output.write_bytes(output.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.compare()

    def test_diagnostic_failure_is_retained_and_cli_writes_json(self):
        write_fixture(self.reference, self.settings, fields_for(self.settings), diagnostic_pass=False)
        write_fixture(self.candidate, self.settings, fields_for(self.settings))
        path = self.root / "comparison.json"
        status = main(["--candidate", str(self.candidate), "--reference", str(self.reference),
                       "--case-id", self.case_id, "--out-report", str(path)])
        self.assertEqual(status, 0)
        result = json.loads(path.read_text())
        self.assertFalse(result["cases"][0]["reference_diagnostic_checks_passed"])
        self.assertFalse(result["training_qualified"])


if __name__ == "__main__":
    unittest.main()
