import os
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIG = _REPO_ROOT / ".mplconfig"
if _MPLCONFIG.exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))

from vpml.visualization.benchmarks import (
    save_fig2_damping_profiles,
    save_fig3_response_function,
    save_fig4_eigenvalue_scan,
    save_fig10_learned_comparison_phase_space,
    save_fig10_learned_comparison_nv_sweep_phase_space,
    save_fig10_nonlinear_landau_phase_space,
    save_linear_landau_comparison,
    save_linear_landau_time,
)
from vpml.visualization.metrics import (
    FieldSweepCase,
    GrowthSweepCase,
    plot_field_metric_sweep,
    plot_growth_metric_sweep,
)
from vpml.visualization.nonlinear import (
    save_bump_on_tail_energy_comparison,
    save_snapshot_panel,
)
from vpml.visualization.training import plot_training_loss, save_training_loss_plot
from vpml.metrics import FourierFieldComparison, GrowthComparisonResult, GrowthFitResult


class VisualizationTests(unittest.TestCase):
    def test_training_loss_plot_uses_objective_specific_ylabel(self) -> None:
        fig_q = plot_training_loss(np.array([1.0, 0.5], dtype=np.float64), train_objective="q_only")
        fig_stab = plot_training_loss(np.array([1.0, 0.5], dtype=np.float64), train_objective="stability_aware")
        fig_hybrid = plot_training_loss(np.array([1.0, 0.5], dtype=np.float64), train_objective="trajectory_q_hybrid")
        self.assertEqual(
            fig_q.axes[0].get_ylabel(),
            r"$\mathcal{L}(\theta)=\mathbb{E}_{\mathrm{regime}}\mathbb{E}_{t,k>0}\left[\left|q_k^\theta-q_k^\star\right|^2\right]$",
        )
        self.assertEqual(
            fig_stab.axes[0].get_ylabel(),
            r"$\mathcal{L}_{\mathrm{stab}}(\theta)=\lambda_q\mathcal{L}_q+\lambda_E\mathcal{L}_{E,\mathrm{window}}^{\mathrm{hyb}}+\lambda_{\mathrm{tail}}\mathcal{L}_{\mathrm{tail},\mathrm{window}}^{\mathrm{hyb}}+\lambda_{\mathrm{reg}}\|\theta\|_2^2$",
        )
        self.assertEqual(
            fig_hybrid.axes[0].get_ylabel(),
            r"$\mathcal{L}_{\mathrm{traj+q}}(\theta)=\lambda_q\mathcal{L}_q+\lambda_E\mathcal{L}_E+\lambda_{\mathrm{dist}}\mathcal{L}_{\delta f}+\lambda_{\mathrm{tail}}\mathcal{L}_{\mathrm{tail}}+\lambda_{\mathrm{neg}}\mathcal{L}_{\mathrm{neg}}+\lambda_{\mathrm{reg}}\|\theta\|_2^2$",
        )
        plt.close(fig_q)
        plt.close(fig_stab)
        plt.close(fig_hybrid)

    def test_training_and_nonlinear_visualizations_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            loss_path = save_training_loss_plot(
                np.array([1.0, 0.5, 0.25], dtype=np.float64),
                tmp / "loss.png",
                val_metrics={"val_q_mse_linear_landau": np.array([1e-3], dtype=np.float64)},
            )
            self.assertTrue(loss_path.exists())

            x = np.linspace(0.0, 1.0, 4)
            v = np.linspace(-1.0, 1.0, 3)
            times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            snaps = {t: np.outer(v + 2.0, x + 1.0 + 0.1 * t) for t in times}
            snap_path = save_snapshot_panel(
                x,
                v,
                snaps,
                tmp / "snapshots.png",
                title="snapshots",
            )
            self.assertTrue(snap_path.exists())

            energy_path = save_bump_on_tail_energy_comparison(
                np.linspace(0.0, 1.0, 5),
                np.linspace(0.1, 0.5, 5),
                np.linspace(0.2, 0.4, 5),
                tmp / "energy.png",
            )
            self.assertTrue(energy_path.exists())

    def test_benchmark_visualizations_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            fig2_payload = {
                "n_Nv20": np.arange(4, dtype=np.float64),
                "filt_Nv20": np.array([1e-3, 1e-2, 1e-1, 1.0], dtype=np.float64),
                "n_Nv1000": np.arange(5, dtype=np.float64),
                "filt_Nv1000": np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0], dtype=np.float64),
                "hyper_Nv20_alpha1": np.array([1e-4, 1e-3, 1e-2, 1e-1], dtype=np.float64),
                "hyper_Nv20_alpha2": np.array([5e-5, 5e-4, 5e-3, 5e-2], dtype=np.float64),
                "hyper_Nv1000_alpha1": np.array([1e-8, 1e-6, 1e-4, 1e-2, 1e-1], dtype=np.float64),
                "hyper_Nv1000_alpha2": np.array([2e-8, 2e-6, 2e-4, 2e-2, 2e-1], dtype=np.float64),
            }
            self.assertTrue(save_fig2_damping_profiles(fig2_payload, p=36, outdir=outdir).exists())

            paper_methods = ["hyper_a1", "hyper_a2", "hyper_a3", "hyper_a4", "nonlocal_nm1", "nonlocal_nm3", "filter"]
            fig3_payload = {
                "Nv_list": np.array([4, 6, 8, 10, 12], dtype=np.int32),
                "xi_plot": np.logspace(-2, 2, 17),
            }
            for idx, name in enumerate(paper_methods, start=1):
                fig3_payload[f"err_{name}"] = np.linspace(0.2, 0.02, 5) * (1.0 + 0.1 * idx)
                fig3_payload[f"abs_err_{name}"] = np.linspace(1e-2, 1e-4, 17) * (1.0 + 0.05 * idx)
            fig3_payload["err_learned"] = np.linspace(0.15, 0.015, 5)
            fig3_payload["abs_err_learned"] = np.linspace(8e-3, 8e-5, 17)
            fig3_outputs = save_fig3_response_function(
                fig3_payload,
                Nv_list=fig3_payload["Nv_list"],
                xi_plot=fig3_payload["xi_plot"],
                xi_min=1e-2,
                xi_max=1e2,
                plot_floor=1e-8,
                plot_ymax=5e-1,
                has_learned=True,
                outdir=outdir,
            )
            self.assertTrue(all(path.exists() for path in fig3_outputs.values()))

            k_list = (0.5, 1.0, 1.5, 2.0)
            mu_vals = np.linspace(-3.0, 0.0, 5)
            chi_vals = np.linspace(0.0, 15.0, 6)
            fig4_payload = {
                "Nv": np.array([20], dtype=np.int32),
                "k_list": np.array(k_list, dtype=np.float64),
                "mu_vals": mu_vals,
                "chi_vals": chi_vals,
                "mu_opt": np.array([-1.0], dtype=np.float64),
            }
            for k in k_list:
                fig4_payload[f"err_nonlocal_k{k}"] = np.linspace(-0.4, 0.4, mu_vals.size)
                fig4_payload[f"err_filter_k{k}"] = np.linspace(-0.3, 0.3, chi_vals.size)
                fig4_payload[f"opt_filter_k{k}"] = np.array([5.0], dtype=np.float64)
                for alpha in (1, 2, 3, 4):
                    nu_vals = np.linspace(0.0, 10.0 + 5.0 * (alpha - 1), 6)
                    fig4_payload[f"nu_vals_a{alpha}"] = nu_vals
                    fig4_payload[f"err_hyper_a{alpha}_k{k}"] = np.linspace(-0.2, 0.2, nu_vals.size)
                    fig4_payload[f"opt_hyper_a{alpha}_k{k}"] = np.array([nu_vals[2]], dtype=np.float64)
            fig4_outputs = save_fig4_eigenvalue_scan(fig4_payload, Nv=20, k_list=k_list, outdir=outdir)
            self.assertTrue(all(path.exists() for path in fig4_outputs.values()))

            x = np.linspace(0.0, 4.0 * np.pi, 5)
            v = np.linspace(-4.0, 4.0, 4)
            times = (20.0, 40.0)
            fig10_payload = {
                "x": x,
                "v": v,
                "times": np.array(times, dtype=np.float64),
            }
            panel_names = ["truncation", "nonlocal_nm1", "hyper_a1", "hyper_a2", "hyper_a3", "filter", "learned"]
            for name in panel_names:
                for t in times:
                    key = f"{name}_f_t{str(t).replace('.0', '')}"
                    fig10_payload[key] = np.outer(v + 5.0, x + 1.0 + 0.01 * t)
            fig10_path = save_fig10_nonlinear_landau_phase_space(
                fig10_payload,
                times=times,
                vmin=0.0,
                vmax=0.5,
                time_key_fn=lambda t: f"t{str(float(t)).replace('.0', '')}",
                outdir=outdir,
            )
            self.assertTrue(fig10_path.exists())

            fig10_learned_path = save_fig10_learned_comparison_phase_space(
                fig10_payload,
                times=times,
                vmin=0.0,
                vmax=0.5,
                time_key_fn=lambda t: f"t{str(float(t)).replace('.0', '')}",
                outdir=outdir,
            )
            self.assertTrue(fig10_learned_path.exists())

            fig10_sweep_payload = {
                "x": x,
                "v": v,
                "times": np.array(times, dtype=np.float64),
            }
            for Nv in (8, 64):
                for method in ("nonlocal", "learned"):
                    for t in times:
                        key = f"nv{Nv}_{method}_f_t{str(t).replace('.0', '')}"
                        fig10_sweep_payload[key] = np.outer(v + 5.0 + 0.01 * Nv, x + 1.0 + 0.01 * t)
            fig10_sweep_path = save_fig10_learned_comparison_nv_sweep_phase_space(
                fig10_sweep_payload,
                nv_list=(8, 64),
                times=times,
                row_labels=(r"$N_v=8$", r"$N_v=64$"),
                vmin=0.0,
                vmax=0.5,
                time_key_fn=lambda t: f"t{str(float(t)).replace('.0', '')}",
                outdir=outdir,
            )
            self.assertTrue(fig10_sweep_path.exists())

            fig10_sweep_bad_payload = dict(fig10_sweep_payload)
            fig10_sweep_bad_payload["nv64_learned_f_t20"] = np.full_like(
                fig10_sweep_bad_payload["nv64_learned_f_t20"],
                np.nan,
            )
            fig10_sweep_bad_path = save_fig10_learned_comparison_nv_sweep_phase_space(
                fig10_sweep_bad_payload,
                nv_list=(8, 64),
                times=times,
                row_labels=(r"$N_v=8$", r"$N_v=64$"),
                vmin=0.0,
                vmax=0.5,
                time_key_fn=lambda t: f"t{str(float(t)).replace('.0', '')}",
                outdir=outdir,
            )
            self.assertTrue(fig10_sweep_bad_path.exists())

            linear_payload = {
                "times": np.linspace(0.0, 1.0, 5),
                "E_abs_k0p5": np.exp(-0.1 * np.linspace(0.0, 1.0, 5)),
                "E_abs_k1p5": np.exp(-0.2 * np.linspace(0.0, 1.0, 5)),
                "gamma_k0p5": np.array([-0.1], dtype=np.float64),
                "gamma_k1p5": np.array([-0.2], dtype=np.float64),
            }
            self.assertTrue(save_linear_landau_time(linear_payload, method="learned", outdir=outdir).exists())

            results = {
                "truncation": SimpleNamespace(payload=linear_payload),
                "learned": SimpleNamespace(payload=linear_payload),
            }
            self.assertTrue(save_linear_landau_comparison(results, outdir).exists())

            times_hr = np.linspace(0.0, 2.0, 21)
            energy_hr = np.exp(0.2 - 0.6 * times_hr)
            growth_cases = [
                GrowthSweepCase(
                    Nv=8,
                    times_theta=times_hr,
                    energy_theta=np.exp(0.15 - 0.58 * times_hr),
                    comparison=GrowthComparisonResult(
                        gamma_grow_theta=-0.29,
                        gamma_grow_hr=-0.30,
                        epsilon_grow=0.033,
                        fit_theta=GrowthFitResult(-0.29, 0.15, 0.0, 2.0, times_hr.size, "all"),
                        fit_hr=GrowthFitResult(-0.30, 0.2, 0.0, 2.0, times_hr.size, "all"),
                        t_a=0.0,
                        t_b=2.0,
                    ),
                    in_training_targets=True,
                    beyond_training_range=False,
                ),
                GrowthSweepCase(
                    Nv=64,
                    times_theta=times_hr,
                    energy_theta=np.exp(0.14 - 0.595 * times_hr),
                    comparison=GrowthComparisonResult(
                        gamma_grow_theta=-0.2975,
                        gamma_grow_hr=-0.30,
                        epsilon_grow=0.0083,
                        fit_theta=GrowthFitResult(-0.2975, 0.14, 0.0, 2.0, times_hr.size, "all"),
                        fit_hr=GrowthFitResult(-0.30, 0.2, 0.0, 2.0, times_hr.size, "all"),
                        t_a=0.0,
                        t_b=2.0,
                    ),
                    in_training_targets=False,
                    beyond_training_range=False,
                ),
            ]
            growth_fig = plot_growth_metric_sweep(
                times_hr,
                energy_hr,
                growth_cases,
                title="growth sweep",
            )
            growth_fig.savefig(outdir / "growth_sweep.png")
            self.assertTrue((outdir / "growth_sweep.png").exists())

            divergent_growth_cases = list(growth_cases)
            divergent_growth_cases[0] = GrowthSweepCase(
                Nv=8,
                times_theta=times_hr,
                energy_theta=np.array([1.0, 0.8, 0.6, np.inf] + [np.inf] * (times_hr.size - 4), dtype=np.float64),
                comparison=growth_cases[0].comparison,
                in_training_targets=True,
                beyond_training_range=False,
            )
            divergent_growth_fig = plot_growth_metric_sweep(
                times_hr,
                energy_hr,
                divergent_growth_cases,
                title="growth sweep divergent",
            )
            divergent_growth_fig.savefig(outdir / "growth_sweep_divergent.png")
            self.assertTrue((outdir / "growth_sweep_divergent.png").exists())

            huge_growth_cases = list(growth_cases)
            huge_growth_cases[0] = GrowthSweepCase(
                Nv=8,
                times_theta=times_hr,
                energy_theta=np.array([1.0, 2.0, 1e250] + [1e250] * (times_hr.size - 3), dtype=np.float64),
                comparison=growth_cases[0].comparison,
                in_training_targets=True,
                beyond_training_range=False,
            )
            huge_growth_fig = plot_growth_metric_sweep(
                times_hr,
                energy_hr,
                huge_growth_cases,
                title="growth sweep huge finite",
            )
            huge_growth_fig.savefig(outdir / "growth_sweep_huge.png")
            self.assertTrue((outdir / "growth_sweep_huge.png").exists())

            comparison = FourierFieldComparison(
                times=np.linspace(0.0, 2.0, 6),
                E_hat_theta=np.ones((6, 3), dtype=np.complex128),
                E_hat_hr=0.9 * np.ones((6, 3), dtype=np.complex128),
                selected_k=np.array([0.5, 1.0, 1.5], dtype=np.float64),
            )
            field_fig = plot_field_metric_sweep(
                [
                    FieldSweepCase(8, comparison, 0.2, True, False),
                    FieldSweepCase(64, comparison, 0.1, False, False),
                ],
                title="field sweep",
            )
            field_fig.savefig(outdir / "field_sweep.png")
            self.assertTrue((outdir / "field_sweep.png").exists())

            divergent_comparison = FourierFieldComparison(
                times=np.linspace(0.0, 2.0, 6),
                E_hat_theta=np.array(
                    [
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [np.inf, 1.0, 1.0],
                        [np.inf, np.inf, 1.0],
                        [np.inf, np.inf, np.inf],
                        [np.inf, np.inf, np.inf],
                    ],
                    dtype=np.complex128,
                ),
                E_hat_hr=0.9 * np.ones((6, 3), dtype=np.complex128),
                selected_k=np.array([0.5, 1.0, 1.5], dtype=np.float64),
            )
            divergent_field_fig = plot_field_metric_sweep(
                [FieldSweepCase(64, divergent_comparison, float("inf"), True, False)],
                title="field sweep divergent",
            )
            divergent_field_fig.savefig(outdir / "field_sweep_divergent.png")
            self.assertTrue((outdir / "field_sweep_divergent.png").exists())


if __name__ == "__main__":
    unittest.main()
