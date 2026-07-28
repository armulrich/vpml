import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.diagnostics.landau_resolution_report import (
    _first_projection_grid_passing_all_cases,
    build_report,
)
from model.diagnostics.physical_velocity_grid_convergence import (
    _distribution_successive_change,
    _energy_block_changes,
    _save_teacher_artifact,
)
from model.diagnostics.projection_quadrature_convergence import (
    _load_teacher_snapshot_artifact,
)
from vpml.physical_grid import PhysicalGridVlasovPoissonConfig


class LandauResolutionDiagnosticTests(unittest.TestCase):
    def test_distribution_change_uses_direct_phase_space_geometry(self) -> None:
        refined = np.ones((2, 8, 4), dtype=np.float64)
        coarse = 0.5 * refined
        velocity = np.linspace(-4.0, 4.0, 8)
        change, by_snapshot = _distribution_successive_change(
            coarse,
            refined,
            coarse_equilibrium=np.zeros(8),
            refined_equilibrium=np.zeros(8),
            coarse_v=velocity,
            refined_v=velocity,
        )
        self.assertAlmostEqual(change, 0.5)
        np.testing.assert_allclose(by_snapshot, 0.5)

    def test_energy_block_changes_include_full_and_late_windows(self) -> None:
        times = np.arange(0.0, 5.0)
        refined = np.ones_like(times)
        coarse = 0.75 * refined
        changes = _energy_block_changes(
            coarse,
            refined,
            times=times,
            block_edges=(0.0, 2.0, 4.0),
        )
        self.assertAlmostEqual(changes["global_energy_refinement_change"], 0.25)
        self.assertAlmostEqual(
            changes["energy_refinement_change_t0_to_2"],
            0.25,
        )
        self.assertAlmostEqual(
            changes["energy_refinement_change_t2_to_4"],
            0.25,
        )

    def test_combined_report_does_not_certify_finest_physical_grid(self) -> None:
        physical_payload = {
            "teacher": {"T_final": 120.0},
            "recommendation": {
                "finest_physical_Nv_tested": 2048,
                "finest_pair_passes_tolerance_for_all_cases": False,
                "successive_change_gate_physical_Nv": None,
                "physical_Nv_for_followup": 2048,
                "qualification": "Finest tested grid only.",
            },
        }
        projection_payload = {
            "teacher": {"Nv": 2048},
            "projection_quadrature_Nv": [2048, 4096, 8192],
            "successive_refinement_summary": {
                "linear": {
                    "4096": {"passes_one_percent_change": True},
                    "8192": {"passes_one_percent_change": True},
                },
                "weak": {
                    "4096": {"passes_one_percent_change": True},
                    "8192": {"passes_one_percent_change": True},
                },
                "strong": {
                    "4096": {"passes_one_percent_change": True},
                    "8192": {"passes_one_percent_change": True},
                },
            },
        }
        report, markdown = build_report(
            physical_payload=physical_payload,
            projection_payload=projection_payload,
        )
        parameters = report["recommended_training_parameters"]
        self.assertEqual(parameters["TEACHER_NV"], 2048)
        self.assertEqual(parameters["TEACHER_PROJECTION_NV"], 4096)
        self.assertFalse(
            report["physical_velocity_grid"]["successive_change_gate_passes"]
        )
        self.assertIn("does not pass the successive-change gate", markdown)

    def test_combined_report_rejects_mismatched_projection_source(self) -> None:
        physical_payload = {
            "teacher": {"T_final": 120.0},
            "recommendation": {
                "finest_physical_Nv_tested": 4096,
                "finest_pair_passes_tolerance_for_all_cases": False,
                "successive_change_gate_physical_Nv": None,
                "physical_Nv_for_followup": 4096,
                "qualification": "Finest tested grid only.",
            },
        }
        projection_payload = {
            "teacher": {"Nv": 2048},
            "projection_quadrature_Nv": [2048, 4096],
            "successive_refinement_summary": {
                "linear": {
                    "4096": {"passes_one_percent_change": True},
                },
            },
        }
        with self.assertRaisesRegex(
            ValueError,
            "projection source Nv=2048, recommended physical Nv=4096",
        ):
            build_report(
                physical_payload=physical_payload,
                projection_payload=projection_payload,
            )

    def test_projection_selection_requires_all_finer_comparisons_to_pass(self) -> None:
        summary = {
            "linear": {
                "4096": {"passes_one_percent_change": True},
                "8192": {"passes_one_percent_change": False},
                "16384": {"passes_one_percent_change": True},
            },
            "strong": {
                "4096": {"passes_one_percent_change": True},
                "8192": {"passes_one_percent_change": True},
                "16384": {"passes_one_percent_change": True},
            },
        }
        self.assertEqual(
            _first_projection_grid_passing_all_cases(summary),
            16384,
        )

    def test_teacher_snapshot_artifact_round_trip(self) -> None:
        config = PhysicalGridVlasovPoissonConfig(
            Nx=8,
            Nv=8,
            Lx=4.0 * np.pi,
            vmin=-4.0,
            vmax=4.0,
            dt=0.01,
            T=0.01,
            snapshot_times=(0.0, 0.01),
        )
        raw = {
            "snapshot_f": np.arange(2 * 8 * 8, dtype=np.float64).reshape(2, 8, 8),
            "times": np.asarray([0.0, 0.01]),
            "energy": np.asarray([1.0, 0.5]),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "teacher.npz"
            _save_teacher_artifact(
                artifact_path=artifact_path,
                config=config,
                raw_by_case={"linear_sample00": raw},
            )
            loaded_config, snapshots, energy = _load_teacher_snapshot_artifact(
                artifact_path
            )
        self.assertEqual(loaded_config, config)
        np.testing.assert_array_equal(
            snapshots["linear_sample00"],
            raw["snapshot_f"],
        )
        np.testing.assert_array_equal(
            energy["linear_sample00_energy"],
            raw["energy"],
        )


if __name__ == "__main__":
    unittest.main()
