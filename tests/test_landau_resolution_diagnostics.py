import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
from model.diagnostics.physical_spatial_grid_convergence import (
    _distribution_successive_x_change,
)
from model.diagnostics.physical_grid_2d_convergence import (
    _parse_case_names,
    _projected_change,
    _run_case_segmented,
)
from model.diagnostics.projection_quadrature_convergence import (
    _load_teacher_snapshot_artifact,
)
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    gaussian_pdf,
    normalize_density_on_grid,
    run_semilagrangian_vlasov_poisson,
)


class LandauResolutionDiagnosticTests(unittest.TestCase):
    def test_case_selection_supports_all_and_explicit_subsets(self) -> None:
        available = ("linear", "weak", "strong")
        self.assertEqual(_parse_case_names("all", available), available)
        self.assertEqual(
            _parse_case_names("strong,linear", available),
            ("strong", "linear"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown case"):
            _parse_case_names("missing", available)

    def test_segmented_case_matches_monolithic_solver(self) -> None:
        config = PhysicalGridVlasovPoissonConfig(
            Nx=8,
            Nv=16,
            Lx=4.0 * np.pi,
            vmin=-6.0,
            vmax=6.0,
            dt=0.01,
            T=0.04,
            snapshot_times=(0.0, 0.02, 0.04),
        )
        equilibrium = normalize_density_on_grid(
            gaussian_pdf(config.v, mean=0.0, sigma=1.0),
            config.v,
        )
        f0 = np.asarray(equilibrium)[:, None] * (
            1.0 + 0.01 * np.cos(0.5 * np.asarray(config.x))[None, :]
        )
        monolithic = run_semilagrangian_vlasov_poisson(config, f0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            segmented = _run_case_segmented(
                config=config,
                f0=f0,
                case_name="linear",
                cases_dir=Path(temporary_directory),
                snapshot_times=config.snapshot_times,
                checkpoint_interval_time=0.02,
            )
        np.testing.assert_allclose(
            segmented["snapshot_f"],
            monolithic["snapshot_f"],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            segmented["energy"],
            monolithic["energy"],
            rtol=1e-13,
            atol=1e-13,
        )

    def test_segmented_case_resumes_from_completed_checkpoint(self) -> None:
        config = PhysicalGridVlasovPoissonConfig(
            Nx=8,
            Nv=16,
            Lx=4.0 * np.pi,
            vmin=-6.0,
            vmax=6.0,
            dt=0.01,
            T=0.04,
            snapshot_times=(0.0, 0.02, 0.04),
        )
        equilibrium = normalize_density_on_grid(
            gaussian_pdf(config.v, mean=0.0, sigma=1.0),
            config.v,
        )
        f0 = np.asarray(equilibrium)[:, None] * (
            1.0 + 0.01 * np.cos(0.5 * np.asarray(config.x))[None, :]
        )
        monolithic = run_semilagrangian_vlasov_poisson(config, f0)
        call_count = 0

        def interrupt_second_segment(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("planned interruption")
            return run_semilagrangian_vlasov_poisson(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            cases_dir = Path(temporary_directory)
            with mock.patch(
                "model.diagnostics.physical_grid_2d_convergence."
                "run_semilagrangian_vlasov_poisson",
                side_effect=interrupt_second_segment,
            ):
                with self.assertRaisesRegex(RuntimeError, "planned interruption"):
                    _run_case_segmented(
                        config=config,
                        f0=f0,
                        case_name="linear",
                        cases_dir=cases_dir,
                        snapshot_times=config.snapshot_times,
                        checkpoint_interval_time=0.02,
                    )
            checkpoint_path = (
                cases_dir / ".linear.progress" / "checkpoint.npz"
            )
            with np.load(checkpoint_path) as checkpoint:
                self.assertEqual(int(checkpoint["completed_steps"]), 2)
            resumed = _run_case_segmented(
                config=config,
                f0=f0,
                case_name="linear",
                cases_dir=cases_dir,
                snapshot_times=config.snapshot_times,
                checkpoint_interval_time=0.02,
            )
        np.testing.assert_allclose(
            resumed["snapshot_f"],
            monolithic["snapshot_f"],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            resumed["energy"],
            monolithic["energy"],
            rtol=1e-13,
            atol=1e-13,
        )

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

    def test_spatial_distribution_change_resamples_periodic_grid(self) -> None:
        coarse_x = np.arange(8, dtype=np.float64) * (2.0 * np.pi / 8.0)
        refined_x = np.arange(16, dtype=np.float64) * (2.0 * np.pi / 16.0)
        coarse = np.cos(2.0 * coarse_x)[None, None, :]
        refined = np.cos(2.0 * refined_x)[None, None, :]
        change, by_snapshot = _distribution_successive_x_change(
            coarse,
            refined,
            equilibrium=np.zeros(1),
            row_chunk=1,
        )
        self.assertLess(change, 1e-12)
        self.assertLess(float(by_snapshot[0]), 1e-12)

    def test_projected_change_is_invariant_to_common_complex_scaling(self) -> None:
        base = np.ones((2, 3, 4), dtype=np.complex128)
        change_c, change_q = _projected_change(
            base,
            2.0 * base,
            cutoffs=(1, 2),
            domain_length=4.0 * np.pi,
        )
        self.assertAlmostEqual(change_c, 0.5)
        self.assertAlmostEqual(change_q, 0.5)

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
