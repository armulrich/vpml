import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from model.eval import main as eval_main
from model.train.train import main as train_main
from vpml.core import (
    LearnedInterfaceClosure,
    e_hat_history_from_a_hat_history,
    save_learned_interface_closure_npz,
)
from vpml.metrics import (
    EarlyElectricFieldGrowthMetric,
    EarlyGrowthConfig,
    FieldErrorConfig,
    SelfGeneratedFieldErrorMetric,
)
from vpml.visualization.metrics import (
    plot_field_metric,
    plot_growth_metric,
    plot_metric_summary,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


def _zero_interface_params(input_dim: int, hidden_width: int = 8, res_blocks: int = 1):
    params = {
        "W_lin": jnp.zeros((input_dim, 2), dtype=jnp.float64),
        "b_lin": jnp.zeros((2,), dtype=jnp.float64),
        "W_in": jnp.zeros((input_dim, hidden_width), dtype=jnp.float64),
        "b_in": jnp.zeros((hidden_width,), dtype=jnp.float64),
        "W_out": jnp.zeros((hidden_width, 2), dtype=jnp.float64),
        "b_out": jnp.zeros((2,), dtype=jnp.float64),
    }
    for block_idx in range(res_blocks):
        params[f"W1_{block_idx}"] = jnp.zeros((hidden_width, hidden_width), dtype=jnp.float64)
        params[f"b1_{block_idx}"] = jnp.zeros((hidden_width,), dtype=jnp.float64)
        params[f"W2_{block_idx}"] = jnp.zeros((hidden_width, hidden_width), dtype=jnp.float64)
        params[f"b2_{block_idx}"] = jnp.zeros((hidden_width,), dtype=jnp.float64)
    return params


def _make_closure(
    *,
    Nm: int = 1,
    hidden_width: int = 8,
    res_blocks: int = 1,
    target_bias=(0.0, 0.0),
) -> LearnedInterfaceClosure:
    input_dim = 2 * Nm + 4
    return LearnedInterfaceClosure(
        params=_zero_interface_params(input_dim, hidden_width=hidden_width, res_blocks=res_blocks),
        Nm=Nm,
        k_scale=2.0,
        nv_scale=8.0,
        input_mean=jnp.zeros((input_dim,), dtype=jnp.float64),
        input_std=jnp.ones((input_dim,), dtype=jnp.float64),
        target_mean=jnp.asarray(target_bias, dtype=jnp.float64),
        target_std=jnp.ones((2,), dtype=jnp.float64),
        hidden_width=hidden_width,
        res_blocks=res_blocks,
        Nv_targets=(4,),
        train_regimes=("linear_landau",),
        teacher_backend="grid_cubic_spline",
        teacher_Lx=4.0 * math.pi,
        teacher_Nx=16,
        teacher_Nv=32,
        teacher_vmin=-6.0,
        teacher_vmax=6.0,
        teacher_dt=0.05,
        teacher_proj_Nv=5,
        include_global_indicators=True,
        n_low=2,
    )


class EvalPipelineTests(unittest.TestCase):
    def test_ehat_history_helper_matches_manual_poisson(self) -> None:
        k_arr = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64)
        a_hat_hist = jnp.zeros((2, 4, 3), dtype=jnp.complex128)
        a_hat_hist = a_hat_hist.at[0, 0].set(jnp.array([0.0, 2.0 + 1.0j, -1.0 + 0.5j], dtype=jnp.complex128))
        a_hat_hist = a_hat_hist.at[1, 0].set(jnp.array([0.0, -0.5 + 2.0j, 1.5 - 0.25j], dtype=jnp.complex128))

        E_hat_hist = np.asarray(
            e_hat_history_from_a_hat_history(a_hat_hist, k_arr, poisson_sign=+1.0)
        )
        manual = np.zeros_like(E_hat_hist)
        manual[:, 1] = 1j * np.asarray(a_hat_hist[:, 0, 1]) / 1.0
        manual[:, 2] = 1j * np.asarray(a_hat_hist[:, 0, 2]) / 2.0
        np.testing.assert_allclose(E_hat_hist, manual, rtol=1e-12, atol=1e-12)

    def test_metric_plot_helpers_render_and_save(self) -> None:
        times = np.linspace(0.0, 2.0, 41)
        series_hr = np.exp(0.2 + 2.0 * 0.3 * times)
        series_theta = np.exp(0.1 + 2.0 * 0.27 * times)
        growth_metric = EarlyElectricFieldGrowthMetric(EarlyGrowthConfig(time_window=(0.0, 2.0)))
        growth = growth_metric.compare(times, series_theta, times, series_hr)

        k = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        E_hat_hr = np.zeros((times.size, k.size), dtype=np.complex128)
        E_hat_hr[:, 1] = 1.0 + 0.2 * times
        E_hat_hr[:, 2] = 0.5 - 0.1j * times
        E_hat_theta = 1.1 * E_hat_hr
        field_metric = SelfGeneratedFieldErrorMetric(FieldErrorConfig(num_low_modes=2))
        field_comparison = field_metric.prepare_fourier_comparison(times, E_hat_theta, k, times, E_hat_hr, k)
        field_result = field_metric.evaluate_fourier(times, E_hat_theta, k, times, E_hat_hr, k)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fig1 = plot_growth_metric(times, series_hr, times, series_theta, growth, title="growth")
            fig1.savefig(tmp / "growth.png")
            fig2 = plot_field_metric(field_comparison, title="field")
            fig2.savefig(tmp / "field.png")
            fig3 = plot_metric_summary(
                times,
                series_hr,
                times,
                series_theta,
                growth,
                field_comparison,
                title="summary",
                epsilon_E=field_result.epsilon_E,
            )
            fig3.savefig(tmp / "summary.png")
            self.assertGreaterEqual(len(fig1.axes), 1)
            self.assertGreaterEqual(len(fig2.axes), 3)
            self.assertGreaterEqual(len(fig3.axes), 4)
            self.assertTrue((tmp / "growth.png").exists())
            self.assertTrue((tmp / "field.png").exists())
            self.assertTrue((tmp / "summary.png").exists())

    def test_eval_main_writes_summary_npz_and_plots_on_tiny_cases(self) -> None:
        closure = _make_closure()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ckpt = tmp / "interface_closure.npz"
            outdir = tmp / "eval"
            save_learned_interface_closure_npz(ckpt, closure)

            eval_main(
                [
                    "--checkpoint",
                    str(ckpt),
                    "--outdir",
                    str(outdir),
                    "--bundles",
                    "heldout_landau",
                    "--teacher-Nx",
                    "8",
                    "--teacher-Nv",
                    "16",
                    "--teacher-dt",
                    "0.05",
                    "--teacher-vmin",
                    "-6",
                    "--teacher-vmax",
                    "6",
                    "--heldout-linear-Nx",
                    "8",
                    "--heldout-linear-Nv",
                    "4",
                    "--heldout-linear-dt",
                    "0.05",
                    "--heldout-linear-T",
                    "0.10",
                    "--heldout-nonlinear-Nx",
                    "8",
                    "--heldout-nonlinear-Nv",
                    "4",
                    "--heldout-nonlinear-dt",
                    "0.05",
                    "--heldout-nonlinear-T",
                    "0.10",
                ]
            )

            summary_path = outdir / "summary.json"
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text())
            self.assertIn("heldout_landau", payload["bundles"])
            case_npz = outdir / "heldout_landau" / "linear_landau_heldout.npz"
            case_png = outdir / "heldout_landau" / "linear_landau_heldout_summary.png"
            self.assertTrue(case_npz.exists())
            self.assertTrue(case_png.exists())

    def test_train_then_eval_pipeline_runs_on_tiny_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "out"
            checkpoint = outdir / "interface_closure.npz"
            train_main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--dataset-cache",
                    str(outdir / "interface_closure_dataset.npz"),
                    "--loss-plot",
                    str(outdir / "interface_closure.loss.png"),
                    "--epochs",
                    "1",
                    "--Nv-targets",
                    "4",
                    "--Nm",
                    "1",
                    "--hidden-width",
                    "8",
                    "--res-blocks",
                    "1",
                    "--teacher-Nx",
                    "8",
                    "--teacher-Nv",
                    "16",
                    "--teacher-dt",
                    "0.05",
                    "--teacher-vmin",
                    "-6",
                    "--teacher-vmax",
                    "6",
                    "--linear-T",
                    "0.10",
                    "--linear-num-samples",
                    "1",
                    "--linear-history-stride",
                    "1",
                    "--nonlinear-T",
                    "0.10",
                    "--nonlinear-history-stride",
                    "1",
                    "--weak-eps",
                    "0.05",
                    "--strong-eps",
                    "0.25",
                ]
            )

            eval_main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--outdir",
                    str(outdir / "eval"),
                    "--bundles",
                    "heldout_landau",
                    "--teacher-Nx",
                    "8",
                    "--teacher-Nv",
                    "16",
                    "--teacher-dt",
                    "0.05",
                    "--teacher-vmin",
                    "-6",
                    "--teacher-vmax",
                    "6",
                    "--heldout-linear-Nx",
                    "8",
                    "--heldout-linear-Nv",
                    "4",
                    "--heldout-linear-dt",
                    "0.05",
                    "--heldout-linear-T",
                    "0.10",
                    "--heldout-nonlinear-Nx",
                    "8",
                    "--heldout-nonlinear-Nv",
                    "4",
                    "--heldout-nonlinear-dt",
                    "0.05",
                    "--heldout-nonlinear-T",
                    "0.10",
                ]
            )
            self.assertTrue(checkpoint.exists())
            self.assertTrue((outdir / "eval" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
