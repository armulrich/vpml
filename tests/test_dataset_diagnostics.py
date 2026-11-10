"""Checks distinguish numerical failures from unverified refinement."""

import json
import unittest

import numpy as np

from vpml.datasets.diagnostics import (
    DIAGNOSTIC_NAMES, DiagnosticTolerances, diagnostic_vector, qualification_report,
)


class DiagnosticTests(unittest.TestCase):
    def history(self):
        return np.tile([2.0, 1.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0], (3, 1))

    def test_checks_pass_without_claiming_training_qualification(self):
        report = qualification_report(self.history())
        self.assertTrue(report["checks_passed"])
        self.assertFalse(report["training_qualified"])
        self.assertIn("refinement", report["qualification_scope"])

    def test_conservation_uses_original_initial_row(self):
        history = self.history()
        history[1:, DIAGNOSTIC_NAMES.index("mass")] *= 1.001
        report = qualification_report(history)
        self.assertIn("relative_mass", report["failures"])
        self.assertAlmostEqual(report["metrics"]["max_relative_mass_error"], 0.001)

    def test_each_quality_failure_is_reported(self):
        violations = {
            "total_energy": (2.0, "relative_energy"),
            "boundary_abs_fraction": (0.01, "boundary_abs_fraction"),
            "min_n": (0.0, "nonpositive_density"),
            "charge_imbalance": (-0.01, "charge_imbalance"),
        }
        for column, (value, failure) in violations.items():
            with self.subTest(column=column):
                history = self.history()
                history[-1, DIAGNOSTIC_NAMES.index(column)] = value
                self.assertIn(failure, qualification_report(history)["failures"])

    def test_finite_distribution_negativity_is_recorded_without_rejection(self):
        history = self.history()
        history[-1, DIAGNOSTIC_NAMES.index("min_f")] = -0.01
        history[-1, DIAGNOSTIC_NAMES.index("negative_mass_fraction")] = 0.02
        report = qualification_report(history)
        self.assertTrue(report["checks_passed"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["metrics"]["min_f"], -0.01)
        self.assertEqual(report["metrics"]["max_negative_mass_fraction"], 0.02)
        self.assertNotIn("minimum_f", report["tolerances"])
        self.assertNotIn("negative_mass_fraction", report["tolerances"])
        self.assertFalse(report["training_qualified"])

    def test_nonfinite_distribution_diagnostics_still_fail(self):
        for column in ("min_f", "negative_mass_fraction"):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(column=column, value=value):
                    history = self.history()
                    history[-1, DIAGNOSTIC_NAMES.index(column)] = value
                    report = qualification_report(history)
                    self.assertFalse(report["checks_passed"])
                    self.assertEqual(report["failures"], ["nonfinite"])
                    json.dumps(report, allow_nan=False)

    def test_nonfinite_report_remains_valid_json(self):
        history = self.history()
        history[-1] = np.nan
        report = qualification_report(history)
        self.assertIn("nonfinite", report["failures"])
        json.dumps(report, allow_nan=False)
        self.assertIsNone(report["metrics"]["min_f"])

    def test_velocity_truncation_is_exposed_not_repaired(self):
        v = np.linspace(-1, 1, 65)
        f = np.broadcast_to(np.exp(-0.5 * v ** 2)[:, None] / np.sqrt(2 * np.pi), (65, 8))
        stats = np.asarray(diagnostic_vector(f, v, 0.5, np.zeros(8)))
        self.assertLess(stats[DIAGNOSTIC_NAMES.index("charge_imbalance")], -0.3)
        self.assertIn("charge_imbalance", qualification_report(stats[None])["failures"])

    def test_invalid_tolerances_rejected(self):
        with self.assertRaises(ValueError):
            DiagnosticTolerances(relative_mass=-1)
        with self.assertRaises(ValueError):
            DiagnosticTolerances(relative_mass=np.nan)


if __name__ == "__main__":
    unittest.main()
