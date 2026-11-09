"""Scientific comparisons keep IC identity, sampling and failure status explicit."""

from copy import deepcopy
import json
import unittest

import numpy as np

from vpml.diagnostics.refinement_metrics import compare_variants


def _variant(name, *, nx=128, nv=9, dt=0.1, output_nx=128):
    times = np.arange(6, dtype=np.float64) * 0.2
    diagnostic_times = np.arange(round(1 / dt) + 1, dtype=np.float64) * dt
    theta = 2 * np.pi * np.arange(output_nx) / output_nx
    wave = (1 + times[:, None]) * np.cos(theta)[None, :]
    fields = np.stack((1 + 1e-4 * wave, 0.1 * wave, 1 + 2e-4 * wave,
                       0.02 * wave, 0.03 * wave), axis=1)
    velocity = np.linspace(-8.0, 8.0, nv)
    kinetic_theta = 2 * np.pi * np.arange(nx) / nx
    return {
        "variant": name,
        "case": {"case_id": "one_fixed_IC", "shape_id": "one_shape", "seed": 1,
                 "split": "train", "mode_count": 1, "decay": 1.2, "k0": 0.4,
                 "strength": 0.5,
                 "coefficients": [[[1.0], [0.0]], [[0.0], [1.0]], [[0.5], [0.0]]]},
        "amplitudes": {"density": 0.18, "velocity": 0.25, "temperature": 0.1},
        "numerics": {"nx": nx, "nv": nv, "dt": dt, "final_time": 1.0,
                     "vmin": -8.0, "vmax": 8.0, "output_nx": output_nx,
                     "save_dt": 0.2, "block_steps": 2, "storage_dtype": "float64"},
        "arrays": {"times": times, "fields": fields, "electric": 0.01 * wave,
                   "diagnostic_times": diagnostic_times,
                   "diagnostics": np.ones((len(diagnostic_times), 8), dtype=np.float64)},
        "state": {"f": 2 + 0.03 * velocity[:, None] + 0.02 * np.cos(kinetic_theta)[None, :]},
        "status": "completed",
        "report": {"checks_passed": True, "failures": [], "training_qualified": False},
    }


class RefinementMetricsTests(unittest.TestCase):
    def test_nested_velocity_and_spatial_grids_compare_same_physical_coordinates(self):
        result = compare_variants(_variant("base"), _variant("fine", nx=256, nv=17, dt=0.05))
        self.assertEqual(result["base_variant"], "base")
        self.assertEqual(result["other_variant"], "fine")
        self.assertEqual(result["final_distribution"]["relative_L2"], 0)
        self.assertIn("Coincident", result["final_distribution"]["velocity_sampling"])
        self.assertEqual(result["final_distribution"]["spatial_stride"], 2)
        self.assertTrue(result["initial_fields_shared_nodes"]["passed"])
        self.assertTrue(result["engineering_checks"]["passed"])
        self.assertFalse(result["training_qualified"])
        self.assertEqual(set(result["spectral_bands"]), {"modes_0_to_4", "modes_0_to_63"})
        for metrics in result["field_errors"].values():
            self.assertEqual(metrics["relative_L2"], 0)
        json.dumps(result, allow_nan=False)

    def test_nonnested_velocity_grids_use_coordinates_instead_of_stride(self):
        result = compare_variants(_variant("base", nv=9), _variant("fine", nv=18))
        self.assertIn("Linear interpolation", result["final_distribution"]["velocity_sampling"])
        self.assertLess(result["final_distribution"]["relative_L2"], 2e-16)

    def test_small_density_perturbation_and_closure_errors_are_not_hidden_by_background(self):
        base, other = _variant("base"), _variant("other")
        other["arrays"]["fields"][:, 0] = 1 + 1.1 * (other["arrays"]["fields"][:, 0] - 1)
        other["arrays"]["fields"][:, 3:] *= 1.1
        result = compare_variants(base, other)
        self.assertEqual(set(result["engineering_checks"]["failed_fields"]),
                         {"n_perturbation", "Q", "dQ_dx"})
        for field in ("n_perturbation", "Q", "dQ_dx"):
            self.assertAlmostEqual(result["field_errors"][field]["relative_L2"], 1 / 11, places=10)
            self.assertGreater(result["field_errors"][field]["absolute_L2"], 0)
            self.assertGreater(result["field_errors"][field]["max_time_rms_error"], 0)

    def test_time_integrated_norm_and_spectral_band_use_parseval_weights(self):
        base, other = _variant("base"), _variant("other")
        base["arrays"]["electric"] *= 1.5
        result = compare_variants(base, other)
        metrics = result["field_errors"]["E"]
        times = base["arrays"]["times"]
        expected = 0.005 * np.sqrt(np.trapezoid((1 + times)**2 / 2, x=times))
        self.assertAlmostEqual(metrics["absolute_L2"], expected, places=15)
        self.assertAlmostEqual(metrics["relative_L2"], 0.5, places=14)
        spectral = result["spectral_bands"]["modes_0_to_4"]["field_errors"]["E"]
        self.assertAlmostEqual(spectral["absolute_L2"], expected, places=15)

    def test_zero_reference_is_explicit_and_json_safe(self):
        base, other = _variant("base"), _variant("other")
        other["arrays"]["fields"][:, 3:] = 0
        base["arrays"]["fields"][:, 3] = 0
        result = compare_variants(base, other)
        self.assertEqual(result["field_errors"]["Q"]["relative_L2"], 0)
        self.assertIsNone(result["field_errors"]["dQ_dx"]["relative_L2"])
        self.assertTrue(result["field_errors"]["dQ_dx"]["zero_reference"])
        self.assertIn("dQ_dx", result["engineering_checks"]["failed_fields"])
        json.dumps(result, allow_nan=False)

    def test_different_coefficients_or_amplitudes_are_rejected(self):
        base = _variant("base")
        for changed in ("coefficients", "amplitudes"):
            other = _variant("other")
            if changed == "coefficients":
                other["case"]["coefficients"][0][0][0] += 1e-12
            else:
                other["amplitudes"]["density"] += 1e-12
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "same"):
                compare_variants(base, other)

    def test_different_sampling_and_interpolated_sample_times_are_rejected(self):
        base, other = _variant("base"), _variant("other")
        other["arrays"]["times"] = other["arrays"]["times"][::2]
        other["arrays"]["fields"] = other["arrays"]["fields"][::2]
        other["arrays"]["electric"] = other["arrays"]["electric"][::2]
        other["numerics"]["save_dt"] = 0.4
        other["status"] = "partial"
        with self.assertRaisesRegex(ValueError, "matching saved times"):
            compare_variants(base, other)
        other = _variant("other", dt=0.1)
        other["numerics"]["save_dt"] = 0.15
        other["arrays"]["times"] = np.arange(6, dtype=np.float64) * 0.15
        other["status"] = "partial"
        with self.assertRaisesRegex(ValueError, "integration-step times"):
            compare_variants(base, other)

    def test_partial_or_failed_diagnostics_never_become_qualified(self):
        base, other = _variant("base"), _variant("other")
        other["status"] = "partial"
        for name in ("times", "fields", "electric"):
            other["arrays"][name] = other["arrays"][name][:4]
        for name in ("diagnostic_times", "diagnostics"):
            other["arrays"][name] = other["arrays"][name][:8]
        other["report"] = {"checks_passed": False, "failures": ["minimum_f"],
                           "training_qualified": False}
        result = compare_variants(base, other)
        self.assertAlmostEqual(result["common_time_interval"][1], 0.6)
        self.assertEqual(result["final_distribution"]["status"], "not_compared")
        self.assertEqual(result["diagnostic_status"]["other"]["failures"], ["minimum_f"])
        self.assertTrue(result["engineering_checks"]["passed"])
        self.assertFalse(result["training_qualified"])
        other["status"] = "completed"
        with self.assertRaisesRegex(ValueError, "declared final time"):
            compare_variants(base, other)

    def test_initial_grid_normalization_drift_is_rejected(self):
        base, other = _variant("base", nx=16, output_nx=16), _variant("other", nx=32, output_nx=16)
        for variant in (base, other):
            variant["amplitudes"]["density"] = 40.0
        with self.assertRaisesRegex(ValueError, "grid-dependent normalization"):
            compare_variants(base, other)

    def test_nonfinite_states_and_malformed_diagnostics_are_rejected(self):
        base, other = _variant("base"), _variant("other")
        broken = deepcopy(other)
        broken["state"]["f"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            compare_variants(base, broken)
        broken = deepcopy(other)
        broken["arrays"]["diagnostics"] = broken["arrays"]["diagnostics"][:, :-1]
        with self.assertRaisesRegex(ValueError, "shape"):
            compare_variants(base, broken)


if __name__ == "__main__":
    unittest.main()
