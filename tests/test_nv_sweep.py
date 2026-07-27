import json
import math
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

from model.eval_nv_sweep import main as nv_sweep_main
from vpml.core import LearnedInterfaceClosure, save_learned_interface_closure_npz

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


def _make_closure() -> LearnedInterfaceClosure:
    nm = 1
    hidden_width = 8
    res_blocks = 1
    input_dim = 2 * nm + 4
    return LearnedInterfaceClosure(
        params=_zero_interface_params(input_dim, hidden_width, res_blocks),
        Nm=nm,
        k_scale=2.0,
        nv_scale=8.0,
        input_mean=jnp.zeros((input_dim,), dtype=jnp.float64),
        input_std=jnp.ones((input_dim,), dtype=jnp.float64),
        target_mean=jnp.zeros((2,), dtype=jnp.float64),
        target_std=jnp.ones((2,), dtype=jnp.float64),
        hidden_width=hidden_width,
        res_blocks=res_blocks,
        equilibrium_centered=True,
        complex_normalization_mode="phase_isotropic",
        translation_augmented=True,
        Nv_targets=(6, 8),
        train_regimes=("linear_landau", "nonlinear_landau_weak", "nonlinear_landau_strong"),
        teacher_backend="grid_cubic_spline",
        teacher_Lx=4.0 * math.pi,
        teacher_Nx=8,
        teacher_Nv=16,
        teacher_vmin=-6.0,
        teacher_vmax=6.0,
        teacher_dt=0.05,
        teacher_proj_Nv=9,
        include_global_indicators=True,
        n_low=2,
        rollout_horizon=1,
    )


class NvSweepTests(unittest.TestCase):
    def test_nv_sweep_writes_retained_metrics_and_raw_hr_fig10(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            checkpoint = tmp / "interface_closure.npz"
            outdir = tmp / "nv_sweep"
            save_learned_interface_closure_npz(checkpoint, _make_closure())

            nv_sweep_main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--outdir",
                    str(outdir),
                    "--nv-list",
                    "6,8",
                    "--Nx",
                    "8",
                    "--dt",
                    "0.05",
                    "--T",
                    "0.10",
                    "--eps",
                    "0.05",
                    "--k0",
                    "0.5",
                    "--snapshot-times",
                    "0.05,0.10",
                    "--Nv-plot",
                    "32",
                    "--phase-reference-mode",
                    "raw_hr_grid",
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
                ]
            )

            summary = json.loads((outdir / "summary.json").read_text())
            self.assertEqual(summary["nv_list"], [6, 8])
            self.assertEqual(summary["phase_reference_mode"], "raw_hr_grid")
            self.assertEqual(len(summary["cases"]), 2)
            self.assertTrue((outdir / "nv_sweep_metric1.png").is_file())
            self.assertTrue((outdir / "nv_sweep_metric2.png").is_file())
            self.assertTrue(
                (outdir / "fig10_learned_vs_nonlocal_nv_sweep_phase_space.png").is_file()
            )
            self.assertFalse((outdir / "nv_sweep_metric3_phase_reconstruction.png").exists())


if __name__ == "__main__":
    unittest.main()
