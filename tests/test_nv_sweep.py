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

from benchmarks.eval_nv_sweep import main as nv_sweep_main
from vpml.core import LearnedInterfaceClosure, load_learned_interface_closure_npz, save_learned_interface_closure_npz

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


def _make_closure(*, Nm: int = 1, hidden_width: int = 8, res_blocks: int = 1) -> LearnedInterfaceClosure:
    input_dim = 2 * Nm + 4
    return LearnedInterfaceClosure(
        params=_zero_interface_params(input_dim, hidden_width=hidden_width, res_blocks=res_blocks),
        Nm=Nm,
        k_scale=2.0,
        nv_scale=8.0,
        input_mean=jnp.zeros((input_dim,), dtype=jnp.float64),
        input_std=jnp.ones((input_dim,), dtype=jnp.float64),
        target_mean=jnp.zeros((2,), dtype=jnp.float64),
        target_std=jnp.ones((2,), dtype=jnp.float64),
        hidden_width=hidden_width,
        res_blocks=res_blocks,
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
    )


def _target_log_ladder(target: int, *, nm: int, levels: int = 5) -> tuple[int, ...]:
    if target < nm:
        raise ValueError(f"target Nv={target} must be at least Nm={nm}")
    if levels <= 0:
        raise ValueError("levels must be positive")
    if target == nm:
        return (nm,)
    ladder = []
    for idx in range(levels):
        t = 0.0 if levels == 1 else idx / float(levels - 1)
        value = math.exp(math.log(float(nm)) + t * (math.log(float(target)) - math.log(float(nm))))
        ladder.append(int(round(value)))
    ladder[0] = nm
    ladder[-1] = target
    dedup = []
    for value in ladder:
        value = max(nm, min(target, int(value)))
        if not dedup or dedup[-1] != value:
            dedup.append(value)
    return tuple(dedup)


def _fixed_ratio_ladder(target: int, *, nm: int, ratio: float = 1.8) -> tuple[int, ...]:
    if target < nm:
        raise ValueError(f"target Nv={target} must be at least Nm={nm}")
    if ratio <= 1.0:
        raise ValueError("ratio must be greater than 1")
    if target == nm:
        return (nm,)
    ladder = [target]
    current = target
    while True:
        next_value = int(math.ceil(float(current) / ratio))
        if next_value <= nm:
            ladder.append(nm)
            break
        ladder.append(next_value)
        current = next_value
    return tuple(sorted(set(max(nm, min(target, int(value))) for value in ladder)))


class NvSweepTests(unittest.TestCase):
    def test_nv_sweep_main_writes_summary_and_figures(self) -> None:
        closure = _make_closure()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ckpt = tmp / "interface_closure.npz"
            outdir = tmp / "nv_sweep"
            save_learned_interface_closure_npz(ckpt, closure)

            nv_sweep_main(
                [
                    "--checkpoint",
                    str(ckpt),
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

            summary_path = outdir / "summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["nv_list"], [6, 8])
            self.assertEqual(len(summary["cases"]), 2)
            self.assertTrue((outdir / "nv_sweep_metric1.png").exists())
            self.assertTrue((outdir / "nv_sweep_metric2.png").exists())
            self.assertTrue((outdir / "fig10_learned_vs_nonlocal_nv_sweep_phase_space.png").exists())

    def test_run_nv_sweep_wrapper_accepts_default_negative_phase_vrange(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outdir = tmp / "nv_sweep_wrapper"

            env = os.environ.copy()
            env.update(
                {
                    "PYTHON": sys.executable,
                    "NV_LIST": "6,8",
                    "NX": "8",
                    "DT": "0.05",
                    "T_FINAL": "0.10",
                    "EPS": "0.05",
                    "K0": "0.5",
                    "SNAPSHOT_TIMES": "0.05,0.10",
                    "NV_PLOT": "32",
                    "TEACHER_NX": "8",
                    "TEACHER_NV": "16",
                    "TEACHER_DT": "0.05",
                    "TEACHER_VMIN": "-6",
                    "TEACHER_VMAX": "6",
                    "TRAIN_NV_TARGETS": "6,8",
                    "TRAIN_NM": "1",
                    "TRAIN_HIDDEN_WIDTH": "8",
                    "TRAIN_RES_BLOCKS": "1",
                    "TRAIN_EPOCHS": "1",
                    "TRAIN_LOG_EVERY": "1",
                    "TRAIN_BATCH_SIZE": "32",
                    "TRAIN_ROLLOUT_BATCH_SIZE": "4",
                    "TRAIN_STEPS_PER_EPOCH": "1",
                    "TRAIN_REGIMES": "linear_landau,nonlinear_landau_weak",
                    "TRAIN_OBJECTIVE": "q_only",
                    "TRAIN_ROLLOUT_HORIZON": "0",
                    "TRAIN_LINEAR_T": "0.10",
                    "TRAIN_LINEAR_NUM_SAMPLES": "1",
                    "TRAIN_LINEAR_HISTORY_STRIDE": "1",
                    "TRAIN_NONLINEAR_T": "0.10",
                    "TRAIN_NONLINEAR_HISTORY_STRIDE": "1",
                    "TRAIN_WEAK_EPS": "0.05",
                    "TRAIN_STRONG_EPS": "0.10",
                    "TRAIN_TEACHER_NX": "8",
                    "TRAIN_TEACHER_NV": "16",
                    "TRAIN_TEACHER_DT": "0.05",
                    "TRAIN_TEACHER_VMIN": "-6",
                    "TRAIN_TEACHER_VMAX": "6",
                    "TRAIN_TEACHER_PROJ_NV": "9",
                }
            )
            subprocess.run(
                ["bash", "benchmarks/run_nv_sweep.sh", str(outdir)],
                cwd="/Users/armin/Documents/NYU/vpml",
                env=env,
                check=True,
            )

            self.assertTrue((outdir / "summary.json").exists())
            self.assertTrue((outdir / "nv_sweep_metric1.png").exists())
            self.assertTrue((outdir / "shared_interface_closure_dataset.npz").exists())
            self.assertTrue((outdir / "models" / "nv6" / "interface_closure.npz").exists())
            self.assertTrue((outdir / "models" / "nv8" / "interface_closure.npz").exists())

    def test_run_nv_sweep_wrapper_stability_aware_uses_per_nv_dataset_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outdir = tmp / "nv_sweep_wrapper_stability"

            env = os.environ.copy()
            env.update(
                {
                    "PYTHON": sys.executable,
                    "NV_LIST": "6,8",
                    "NX": "8",
                    "DT": "0.05",
                    "T_FINAL": "0.10",
                    "EPS": "0.05",
                    "K0": "0.5",
                    "SNAPSHOT_TIMES": "0.05,0.10",
                    "NV_PLOT": "32",
                    "TEACHER_NX": "8",
                    "TEACHER_NV": "16",
                    "TEACHER_DT": "0.05",
                    "TEACHER_VMIN": "-6",
                    "TEACHER_VMAX": "6",
                    "TRAIN_NV_TARGETS": "6,8",
                    "TRAIN_NM": "1",
                    "TRAIN_HIDDEN_WIDTH": "8",
                    "TRAIN_RES_BLOCKS": "1",
                    "TRAIN_EPOCHS": "1",
                    "TRAIN_LOG_EVERY": "1",
                    "TRAIN_BATCH_SIZE": "32",
                    "TRAIN_ROLLOUT_BATCH_SIZE": "4",
                    "TRAIN_STEPS_PER_EPOCH": "1",
                    "TRAIN_REGIMES": "linear_landau,nonlinear_landau_weak",
                    "TRAIN_OBJECTIVE": "stability_aware",
                    "TRAIN_ROLLOUT_HORIZON": "2",
                    "TRAIN_LINEAR_T": "0.10",
                    "TRAIN_LINEAR_NUM_SAMPLES": "1",
                    "TRAIN_LINEAR_HISTORY_STRIDE": "1",
                    "TRAIN_NONLINEAR_T": "0.10",
                    "TRAIN_NONLINEAR_HISTORY_STRIDE": "1",
                    "TRAIN_WEAK_EPS": "0.05",
                    "TRAIN_STRONG_EPS": "0.10",
                    "TRAIN_TEACHER_NX": "8",
                    "TRAIN_TEACHER_NV": "16",
                    "TRAIN_TEACHER_DT": "0.05",
                    "TRAIN_TEACHER_VMIN": "-6",
                    "TRAIN_TEACHER_VMAX": "6",
                    "TRAIN_TEACHER_PROJ_NV": "9",
                }
            )
            subprocess.run(
                ["bash", "benchmarks/run_nv_sweep.sh", str(outdir)],
                cwd="/Users/armin/Documents/NYU/vpml",
                env=env,
                check=True,
            )

            self.assertTrue((outdir / "summary.json").exists())
            self.assertFalse((outdir / "shared_interface_closure_dataset.npz").exists())
            self.assertTrue((outdir / "models" / "nv6" / "interface_closure_dataset.npz").exists())
            self.assertTrue((outdir / "models" / "nv8" / "interface_closure_dataset.npz").exists())
            self.assertTrue((outdir / "models" / "nv6" / "interface_closure.npz").exists())
            self.assertTrue((outdir / "models" / "nv8" / "interface_closure.npz").exists())

    def test_fixed_ratio_ladder_matches_default_targets(self) -> None:
        expected = {
            8: (6, 8),
            64: (6, 7, 12, 20, 36, 64),
            256: (6, 8, 14, 25, 45, 80, 143, 256),
            300: (6, 10, 17, 29, 52, 93, 167, 300),
            512: (6, 9, 16, 28, 50, 89, 159, 285, 512),
        }
        actual = {target: _fixed_ratio_ladder(target, nm=6, ratio=1.8) for target in expected}
        self.assertEqual(actual, expected)

    def test_run_nv_sweep_single_qloss_wrapper_uses_target_specific_ladders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outdir = tmp / "nv_sweep_single_qloss"

            env = os.environ.copy()
            env.update(
                {
                    "PYTHON": sys.executable,
                    "NV_LIST": "6,8",
                    "NX": "8",
                    "DT": "0.05",
                    "T_FINAL": "0.10",
                    "EPS": "0.05",
                    "K0": "0.5",
                    "SNAPSHOT_TIMES": "0.05,0.10",
                    "NV_PLOT": "32",
                    "TEACHER_NX": "8",
                    "TEACHER_NV": "16",
                    "TEACHER_DT": "0.05",
                    "TEACHER_VMIN": "-6",
                    "TEACHER_VMAX": "6",
                    "TRAIN_NM": "6",
                    "TRAIN_HIDDEN_WIDTH": "8",
                    "TRAIN_RES_BLOCKS": "1",
                    "TRAIN_EPOCHS": "1",
                    "TRAIN_LOG_EVERY": "1",
                    "TRAIN_BATCH_SIZE": "32",
                    "TRAIN_STEPS_PER_EPOCH": "1",
                    "TRAIN_REGIMES": "linear_landau,nonlinear_landau_weak",
                    "TRAIN_LINEAR_T": "0.10",
                    "TRAIN_LINEAR_NUM_SAMPLES": "1",
                    "TRAIN_LINEAR_HISTORY_STRIDE": "1",
                    "TRAIN_NONLINEAR_T": "0.10",
                    "TRAIN_NONLINEAR_HISTORY_STRIDE": "1",
                    "TRAIN_WEAK_EPS": "0.05",
                    "TRAIN_STRONG_EPS": "0.10",
                    "TRAIN_TEACHER_NX": "8",
                    "TRAIN_TEACHER_NV": "16",
                    "TRAIN_TEACHER_DT": "0.05",
                    "TRAIN_TEACHER_VMIN": "-6",
                    "TRAIN_TEACHER_VMAX": "6",
                }
            )
            subprocess.run(
                ["bash", "benchmarks/run_nv_sweep_single_qloss.sh", str(outdir)],
                cwd="/Users/armin/Documents/NYU/vpml",
                env=env,
                check=True,
            )

            self.assertTrue((outdir / "summary.json").exists())
            self.assertTrue((outdir / "nv_sweep_metric1.png").exists())
            self.assertTrue((outdir / "nv_sweep_metric2.png").exists())
            self.assertTrue((outdir / "fig10_learned_vs_nonlocal_nv_sweep_phase_space.png").exists())
            self.assertTrue((outdir / "models" / "nv6" / "interface_closure_dataset.npz").exists())
            self.assertTrue((outdir / "models" / "nv8" / "interface_closure_dataset.npz").exists())

            expected = {
                6: _target_log_ladder(6, nm=6),
                8: _target_log_ladder(8, nm=6),
            }
            for nv, expected_targets in expected.items():
                ckpt = outdir / "models" / f"nv{nv}" / "interface_closure.npz"
                self.assertTrue(ckpt.exists())
                learned = load_learned_interface_closure_npz(ckpt)
                self.assertEqual(tuple(int(v) for v in learned.Nv_targets), expected_targets)

            summary = json.loads((outdir / "summary.json").read_text())
            self.assertEqual(summary["nv_list"], [6, 8])
            case_targets = {int(case["Nv"]): tuple(int(v) for v in case["train_nv_targets"]) for case in summary["cases"]}
            self.assertEqual(case_targets, expected)

    def test_run_nv_sweep_single_qloss_fixed_ratio_wrapper_uses_fixed_ratio_ladders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outdir = tmp / "nv_sweep_single_qloss_fixed_ratio"

            env = os.environ.copy()
            env.update(
                {
                    "PYTHON": sys.executable,
                    "NV_LIST": "6,8",
                    "NX": "8",
                    "DT": "0.05",
                    "T_FINAL": "0.10",
                    "EPS": "0.05",
                    "K0": "0.5",
                    "SNAPSHOT_TIMES": "0.05,0.10",
                    "NV_PLOT": "32",
                    "TEACHER_NX": "8",
                    "TEACHER_NV": "16",
                    "TEACHER_DT": "0.05",
                    "TEACHER_VMIN": "-6",
                    "TEACHER_VMAX": "6",
                    "TRAIN_NM": "6",
                    "TRAIN_FIXED_RATIO": "1.8",
                    "TRAIN_HIDDEN_WIDTH": "8",
                    "TRAIN_RES_BLOCKS": "1",
                    "TRAIN_EPOCHS": "1",
                    "TRAIN_LOG_EVERY": "1",
                    "TRAIN_BATCH_SIZE": "32",
                    "TRAIN_STEPS_PER_EPOCH": "1",
                    "TRAIN_REGIMES": "linear_landau,nonlinear_landau_weak",
                    "TRAIN_LINEAR_T": "0.10",
                    "TRAIN_LINEAR_NUM_SAMPLES": "1",
                    "TRAIN_LINEAR_HISTORY_STRIDE": "1",
                    "TRAIN_NONLINEAR_T": "0.10",
                    "TRAIN_NONLINEAR_HISTORY_STRIDE": "1",
                    "TRAIN_WEAK_EPS": "0.05",
                    "TRAIN_STRONG_EPS": "0.10",
                    "TRAIN_TEACHER_NX": "8",
                    "TRAIN_TEACHER_NV": "16",
                    "TRAIN_TEACHER_DT": "0.05",
                    "TRAIN_TEACHER_VMIN": "-6",
                    "TRAIN_TEACHER_VMAX": "6",
                }
            )
            subprocess.run(
                ["bash", "benchmarks/run_nv_sweep_single_qloss_fixed_ratio.sh", str(outdir)],
                cwd="/Users/armin/Documents/NYU/vpml",
                env=env,
                check=True,
            )

            self.assertTrue((outdir / "summary.json").exists())
            self.assertTrue((outdir / "nv_sweep_metric1.png").exists())
            self.assertTrue((outdir / "nv_sweep_metric2.png").exists())
            self.assertTrue((outdir / "fig10_learned_vs_nonlocal_nv_sweep_phase_space.png").exists())
            self.assertTrue((outdir / "models" / "nv6" / "interface_closure_dataset.npz").exists())
            self.assertTrue((outdir / "models" / "nv8" / "interface_closure_dataset.npz").exists())

            expected = {
                6: _fixed_ratio_ladder(6, nm=6, ratio=1.8),
                8: _fixed_ratio_ladder(8, nm=6, ratio=1.8),
            }
            for nv, expected_targets in expected.items():
                ckpt = outdir / "models" / f"nv{nv}" / "interface_closure.npz"
                self.assertTrue(ckpt.exists())
                learned = load_learned_interface_closure_npz(ckpt)
                self.assertEqual(tuple(int(v) for v in learned.Nv_targets), expected_targets)

            summary = json.loads((outdir / "summary.json").read_text())
            self.assertEqual(summary["nv_list"], [6, 8])
            case_targets = {int(case["Nv"]): tuple(int(v) for v in case["train_nv_targets"]) for case in summary["cases"]}
            self.assertEqual(case_targets, expected)

    def test_run_nv_sweep_higher_order_hermite_fixed_ratio_wrapper_uses_fixed_ratio_ladders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outdir = tmp / "nv_sweep_higher_order_hermite_fixed_ratio"

            env = os.environ.copy()
            env.update(
                {
                    "PYTHON": sys.executable,
                    "NV_LIST": "6,8",
                    "NX": "8",
                    "DT": "0.05",
                    "T_FINAL": "0.10",
                    "EPS": "0.05",
                    "K0": "0.5",
                    "SNAPSHOT_TIMES": "0.05,0.10",
                    "NV_PLOT": "32",
                    "TEACHER_NX": "8",
                    "TEACHER_NV": "16",
                    "TEACHER_DT": "0.05",
                    "TEACHER_VMIN": "-6",
                    "TEACHER_VMAX": "6",
                    "HR_TEACHER_RATIO": "2.0",
                    "TRAIN_NM": "6",
                    "TRAIN_FIXED_RATIO": "1.8",
                    "TRAIN_HIDDEN_WIDTH": "8",
                    "TRAIN_RES_BLOCKS": "1",
                    "TRAIN_EPOCHS": "1",
                    "TRAIN_LOG_EVERY": "1",
                    "TRAIN_BATCH_SIZE": "32",
                    "TRAIN_STEPS_PER_EPOCH": "1",
                    "TRAIN_REGIMES": "linear_landau,nonlinear_landau_weak",
                    "TRAIN_LINEAR_T": "0.10",
                    "TRAIN_LINEAR_NUM_SAMPLES": "1",
                    "TRAIN_LINEAR_HISTORY_STRIDE": "1",
                    "TRAIN_NONLINEAR_T": "0.10",
                    "TRAIN_NONLINEAR_HISTORY_STRIDE": "1",
                    "TRAIN_WEAK_EPS": "0.05",
                    "TRAIN_STRONG_EPS": "0.10",
                    "TRAIN_TEACHER_NX": "8",
                    "TRAIN_TEACHER_DT": "0.05",
                    "TRAIN_TEACHER_VMIN": "-6",
                    "TRAIN_TEACHER_VMAX": "6",
                }
            )
            subprocess.run(
                ["bash", "benchmarks/run_nv_sweep_higher_order_hermite_fixed_ratio.sh", str(outdir)],
                cwd="/Users/armin/Documents/NYU/vpml",
                env=env,
                check=True,
            )

            self.assertTrue((outdir / "summary.json").exists())
            self.assertTrue((outdir / "nv_sweep_metric1.png").exists())
            self.assertTrue((outdir / "nv_sweep_metric2.png").exists())
            self.assertTrue((outdir / "fig10_learned_vs_nonlocal_nv_sweep_phase_space.png").exists())
            self.assertTrue((outdir / "models" / "nv6" / "interface_closure_dataset.npz").exists())
            self.assertTrue((outdir / "models" / "nv8" / "interface_closure_dataset.npz").exists())

            expected = {
                6: _fixed_ratio_ladder(6, nm=6, ratio=1.8),
                8: _fixed_ratio_ladder(8, nm=6, ratio=1.8),
            }
            expected_teacher_nv = {
                6: 12,
                8: 16,
            }
            for nv, expected_targets in expected.items():
                ckpt = outdir / "models" / f"nv{nv}" / "interface_closure.npz"
                self.assertTrue(ckpt.exists())
                learned = load_learned_interface_closure_npz(ckpt)
                self.assertEqual(learned.teacher_backend, "higher_order_hermite")
                self.assertEqual(tuple(int(v) for v in learned.Nv_targets), expected_targets)
                self.assertEqual(int(learned.teacher_Nv), expected_teacher_nv[nv])
                self.assertIsNone(learned.teacher_proj_Nv)

            summary = json.loads((outdir / "summary.json").read_text())
            self.assertEqual(summary["nv_list"], [6, 8])
            case_targets = {int(case["Nv"]): tuple(int(v) for v in case["train_nv_targets"]) for case in summary["cases"]}
            self.assertEqual(case_targets, expected)


if __name__ == "__main__":
    unittest.main()
