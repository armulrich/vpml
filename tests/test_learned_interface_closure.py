import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import map_coordinates

import model.train.train as train_mod
from benchmarks.fh_benchmarks_2412_07073_jax import (
    Fig3ResponseFunction,
    Fig4EigenvalueScan,
)
from model.train.train import main as train_main
from benchmarks.fh_nonlinear_sim_jax import (
    BumpOnTailParams,
    TwoStreamParams,
    simulate_bump_on_tail,
    simulate_two_stream,
)
from vpml.core import (
    FourierHermiteIMEX,
    LearnedInterfaceClosure,
    learned_boundary_flux_hat,
    load_learned_interface_closure_npz,
    save_learned_interface_closure_npz,
)
from vpml.jax_runtime import plan_jax_runtime
from vpml.linear_landau import (
    LinearLandauConfig,
    run_linear_landau_cnab2_raw,
    run_linear_landau_rollout_raw,
)
from vpml.nonlinear_landau import (
    NonlinearLandauParams,
    run_nonlinear_landau_rollout_raw,
)
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    advect_v_cubic,
    advect_x_cubic,
    compute_electric_field_from_distribution,
    cubic_bspline_interp_constant,
    cubic_bspline_interp_periodic,
    cubic_bspline_prefilter_constant,
    cubic_bspline_prefilter_periodic,
    equilibrium_coeffs_bump_on_tail,
    extract_interface_supervised_pairs_from_coeff_history,
    gaussian_pdf,
    hermite_dual_basis_scaled,
    normalize_density_on_grid,
    project_distribution_snapshot_to_fourier_hermite,
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
    params=None,
    target_bias=(0.0, 0.0),
    include_global_indicators: bool = True,
    n_low: int = 2,
    context_mode: str = "none",
    train_objective: str = "q_only",
    rollout_horizon: int = 0,
    lambda_q: float = 1.0,
    lambda_E: float = 0.0,
    lambda_dist: float = 0.0,
    lambda_tail: float = 0.0,
    lambda_neg: float = 0.0,
    lambda_reg: float = 0.0,
    training_mode: str = "offline_rollout",
    loss_backend: str | None = None,
    online_v_probes: int = 0,
    stability_loss_definition: str | None = None,
) -> LearnedInterfaceClosure:
    raw_base_dim = 2 * Nm + (4 if include_global_indicators else 2)
    input_dim = raw_base_dim if context_mode == "none" else 3 * raw_base_dim
    if params is None:
        params = _zero_interface_params(input_dim, hidden_width=hidden_width, res_blocks=res_blocks)
    target_mean = jnp.asarray(target_bias, dtype=jnp.float64)
    return LearnedInterfaceClosure(
        params=params,
        Nm=Nm,
        k_scale=2.0,
        nv_scale=8.0,
        input_mean=jnp.zeros((input_dim,), dtype=jnp.float64),
        input_std=jnp.ones((input_dim,), dtype=jnp.float64),
        target_mean=target_mean,
        target_std=jnp.ones((2,), dtype=jnp.float64),
        hidden_width=hidden_width,
        res_blocks=res_blocks,
        Nv_targets=(4,),
        train_regimes=("linear_landau",),
        teacher_backend="grid_cubic_spline",
        teacher_Lx=4.0 * math.pi,
        teacher_Nx=32,
        teacher_Nv=64,
        teacher_vmin=-8.0,
        teacher_vmax=8.0,
        teacher_dt=0.05,
        teacher_proj_Nv=5,
        include_global_indicators=include_global_indicators,
        n_low=n_low,
        training_mode=training_mode,
        train_objective=train_objective,
        context_mode=context_mode,
        context_lags=1 if context_mode == "lag1_delta" else 0,
        base_input_dim=raw_base_dim,
        rollout_horizon=rollout_horizon,
        loss_backend=loss_backend,
        lambda_q=lambda_q,
        lambda_E=lambda_E,
        lambda_dist=lambda_dist,
        lambda_tail=lambda_tail,
        lambda_neg=lambda_neg,
        lambda_reg=lambda_reg,
        online_v_probes=online_v_probes,
        stability_loss_definition=stability_loss_definition,
    )


class LearnedInterfaceClosureTests(unittest.TestCase):
    def test_jax_runtime_plan_prefers_cpu_on_macos_auto(self) -> None:
        plan = plan_jax_runtime({}, system="Darwin")
        self.assertEqual(plan.requested_backend, "auto")
        self.assertEqual(plan.jax_platforms, "cpu")
        self.assertTrue(plan.metal_disabled)

    def test_jax_runtime_plan_leaves_linux_auto_unset(self) -> None:
        plan = plan_jax_runtime({}, system="Linux")
        self.assertEqual(plan.requested_backend, "auto")
        self.assertIsNone(plan.jax_platforms)
        self.assertFalse(plan.metal_disabled)

    def test_jax_runtime_plan_respects_explicit_override(self) -> None:
        plan = plan_jax_runtime({"JAX_PLATFORMS": "cpu"}, system="Linux")
        self.assertTrue(plan.env_override)
        self.assertIsNone(plan.jax_platforms)

    def test_checkpoint_round_trip(self) -> None:
        closure = _make_closure(target_bias=(1.5, -0.25))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interface_closure.npz"
            save_learned_interface_closure_npz(path, closure)
            loaded = load_learned_interface_closure_npz(path)
        self.assertEqual(loaded.Nm, closure.Nm)
        self.assertEqual(loaded.hidden_width, closure.hidden_width)
        self.assertEqual(loaded.res_blocks, closure.res_blocks)
        self.assertEqual(loaded.teacher_backend, "grid_cubic_spline")
        self.assertEqual(loaded.teacher_proj_Nv, 5)
        self.assertTrue(loaded.include_global_indicators)
        self.assertEqual(loaded.n_low, 2)
        np.testing.assert_allclose(np.asarray(loaded.input_mean), np.asarray(closure.input_mean))
        np.testing.assert_allclose(np.asarray(loaded.target_mean), np.asarray(closure.target_mean))
        np.testing.assert_allclose(np.asarray(loaded.params["W_lin"]), np.asarray(closure.params["W_lin"]))

    def test_checkpoint_loader_normalizes_legacy_teacher_backend_name(self) -> None:
        closure = _make_closure(target_bias=(1.5, -0.25))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interface_closure.npz"
            save_learned_interface_closure_npz(path, closure)
            with np.load(path) as data:
                payload = {name: np.asarray(data[name]) for name in data.files}
            payload["teacher_backend"] = np.array(["physical_grid_cubic_v1"], dtype=np.str_)
            np.savez(path, **payload)
            loaded = load_learned_interface_closure_npz(path)
        self.assertEqual(loaded.teacher_backend, "grid_cubic_spline")

    def test_checkpoint_round_trip_preserves_stability_metadata(self) -> None:
        closure = _make_closure(
            context_mode="lag1_delta",
            train_objective="stability_aware",
            rollout_horizon=2,
            lambda_q=1.0,
            lambda_E=0.5,
            lambda_tail=0.05,
            lambda_reg=1e-6,
            stability_loss_definition=train_mod.STABILITY_LOSS_DEFINITION,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interface_closure.npz"
            save_learned_interface_closure_npz(path, closure)
            loaded = load_learned_interface_closure_npz(path)
        self.assertEqual(loaded.context_mode, "lag1_delta")
        self.assertEqual(loaded.train_objective, "stability_aware")
        self.assertEqual(loaded.rollout_horizon, 2)
        self.assertAlmostEqual(loaded.lambda_E, 0.5)
        self.assertAlmostEqual(loaded.lambda_tail, 0.05)
        self.assertEqual(loaded.stability_loss_definition, train_mod.STABILITY_LOSS_DEFINITION)
        self.assertEqual(loaded.input_dim, 18)

    def test_checkpoint_round_trip_preserves_online_metadata(self) -> None:
        closure = _make_closure(
            training_mode="online_rollout",
            train_objective="trajectory",
            loss_backend=train_mod.ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1,
            lambda_E=0.5,
            lambda_dist=1.0,
            lambda_tail=0.05,
            lambda_neg=0.025,
            lambda_reg=1e-6,
            online_v_probes=64,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interface_closure.npz"
            save_learned_interface_closure_npz(path, closure)
            loaded = load_learned_interface_closure_npz(path)
        self.assertEqual(loaded.training_mode, "online_rollout")
        self.assertEqual(loaded.train_objective, "trajectory")
        self.assertEqual(loaded.loss_backend, train_mod.ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1)
        self.assertAlmostEqual(loaded.lambda_E, 0.5)
        self.assertAlmostEqual(loaded.lambda_dist, 1.0)
        self.assertAlmostEqual(loaded.lambda_tail, 0.05)
        self.assertAlmostEqual(loaded.lambda_neg, 0.025)
        self.assertAlmostEqual(loaded.lambda_reg, 1e-6)
        self.assertEqual(loaded.online_v_probes, 64)

    def test_boundary_flux_only_touches_last_row_and_zero_mode(self) -> None:
        params = _zero_interface_params(6)
        params["b_lin"] = jnp.array([2.0, -1.0], dtype=jnp.float64)
        closure = _make_closure(params=params)
        a_hat = jnp.zeros((4, 3), dtype=jnp.complex128)
        B_hat = learned_boundary_flux_hat(
            a_hat,
            jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64),
            Nv=4,
            vth=1.0,
            learned=closure,
        )
        np.testing.assert_allclose(np.asarray(B_hat[:-1]), 0.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(B_hat[-1]),
            np.array([0.0 + 0.0j, 2.0 - 1.0j, 2.0 - 1.0j], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_zero_output_closure_matches_truncation_for_linear_cnab2(self) -> None:
        closure = _make_closure()
        config = LinearLandauConfig(Nv=4, Nx=8, dt=0.05, T=0.10)
        trunc = run_linear_landau_cnab2_raw(config, return_state_history=True)
        learned = run_linear_landau_rollout_raw(
            LinearLandauConfig(method="learned", Nv=4, Nx=8, dt=0.05, T=0.10),
            learned_closure=closure,
            solver_backend="cnab2",
            return_state_history=True,
        )
        np.testing.assert_allclose(
            np.asarray(learned["a_hat_hist"]),
            np.asarray(trunc["a_hat_hist"]),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_linear_branch_can_reproduce_affine_map(self) -> None:
        params = _zero_interface_params(6)
        params["W_lin"] = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        params["b_lin"] = jnp.array([0.25, -0.75], dtype=jnp.float64)
        closure = _make_closure(params=params)
        x = jnp.array([[2.0, -1.0, 4.0, 3.0, 0.5, 1.5]], dtype=jnp.float64)
        pred = np.asarray(closure.predict_q_components(x))[0]
        expected = np.array([2.0 + 0.5 * 4.0 + 0.25, -1.0 - 3.0 - 0.75], dtype=np.float64)
        np.testing.assert_allclose(pred, expected, rtol=1e-12, atol=1e-12)

    def test_extract_interface_pairs_matches_exact_q_target(self) -> None:
        a_hat_hist = np.zeros((2, 6, 3), dtype=np.complex128)
        a_hat_hist[:, 3, 1] = 1.0 + 2.0j
        a_hat_hist[:, 3, 2] = -0.5 + 0.25j
        a_hat_hist[:, 4, 1] = 3.0 - 4.0j
        a_hat_hist[:, 4, 2] = -2.0 + 1.0j
        pairs = extract_interface_supervised_pairs_from_coeff_history(
            a_hat_hist,
            Nv_targets=(4,),
            Nm=1,
            k_arr=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            vth=1.0,
        )[4]
        expected_q = -1j * np.array([1.0, 2.0]) * math.sqrt(4.0) * np.array([3.0 - 4.0j, -2.0 + 1.0j])
        expected_targets = np.stack([expected_q.real, expected_q.imag], axis=-1)
        np.testing.assert_allclose(pairs["targets"][:2], expected_targets, rtol=1e-12, atol=1e-12)
        self.assertEqual(pairs["inputs_base"].shape[1], 6)
        np.testing.assert_allclose(pairs["inputs_base"][:2, -2:], 0.0, rtol=1e-12, atol=1e-12)

    def test_context_mode_lag1_delta_builds_temporal_features(self) -> None:
        a_hat_hist = np.zeros((3, 6, 3), dtype=np.complex128)
        a_hat_hist[0, 3, 1] = 1.0 + 0.0j
        a_hat_hist[1, 3, 1] = 2.0 + 0.0j
        a_hat_hist[2, 3, 1] = 4.0 + 0.0j
        a_hat_hist[0, 4, 1] = 1.0 + 0.0j
        a_hat_hist[1, 4, 1] = 3.0 + 0.0j
        a_hat_hist[2, 4, 1] = 5.0 + 0.0j
        pairs = extract_interface_supervised_pairs_from_coeff_history(
            a_hat_hist,
            Nv_targets=(4,),
            Nm=1,
            k_arr=np.array([0.0, 2.0, 4.0], dtype=np.float64),
            vth=1.0,
            include_global_indicators=False,
            context_mode="lag1_delta",
        )[4]
        self.assertEqual(pairs["inputs_base"].shape[1], 12)
        scaled = train_mod.build_model_inputs(
            pairs["inputs_base"],
            Nm=1,
            k_scale=2.0,
            nv_scale=8.0,
            context_mode="lag1_delta",
            include_global_indicators=False,
        )
        np.testing.assert_allclose(
            scaled[0],
            np.array(
                [
                    2.0, 0.0, 1.0, 0.5,
                    1.0, 0.0, 1.0, 0.5,
                    1.0, 0.0, 0.0, 0.0,
                ],
                dtype=np.float64,
            ),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_rollout_window_loss_terms_are_zero_for_zero_dynamics(self) -> None:
        closure = _make_closure(context_mode="lag1_delta", train_objective="stability_aware", rollout_horizon=2)
        integ = FourierHermiteIMEX(Nx=8, Nv=4, Lx=4.0 * math.pi, dt=0.05, vth=1.0, dealias_23=True, closure=None)
        prev = jnp.zeros((4, integ.Nk), dtype=jnp.complex128)
        curr = jnp.zeros((4, integ.Nk), dtype=jnp.complex128)
        future = jnp.zeros((2, 4, integ.Nk), dtype=jnp.complex128)
        future_e = jnp.zeros((2, integ.Nk), dtype=jnp.complex128)
        m_eq = jnp.zeros((4,), dtype=jnp.float64).at[0].set(1.0)
        field_loss, tail_loss = train_mod.rollout_window_loss_terms(
            closure,
            prev_state=prev,
            curr_state=curr,
            future_state=future,
            future_E_hat=future_e,
            integ=integ,
            m_eq=m_eq,
            poisson_sign=1.0,
            rollout_weights=jnp.array([1.0, 0.5], dtype=jnp.float64),
            tail_start_fraction=2.0 / 3.0,
            field_ref_scale=jnp.asarray(0.0, dtype=jnp.float64),
            tail_ref_scale=jnp.asarray(0.0, dtype=jnp.float64),
        )
        self.assertAlmostEqual(float(field_loss), 0.0, places=12)
        self.assertAlmostEqual(float(tail_loss), 0.0, places=12)

    def test_window_hybrid_field_loss_matches_closed_form(self) -> None:
        pred_state = jnp.zeros((2, 4, 3), dtype=jnp.complex128)
        true_state = jnp.zeros((2, 4, 3), dtype=jnp.complex128)
        pred_e = jnp.array(
            [
                [0.0 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j],
                [0.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j],
            ],
            dtype=jnp.complex128,
        )
        true_e = jnp.array(
            [
                [0.0 + 0.0j, 1.0e-8 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            ],
            dtype=jnp.complex128,
        )
        weights = jnp.array([1.0, 0.5], dtype=jnp.float64)
        field_ref = 2.5
        field_loss, tail_loss = train_mod.rollout_window_hybrid_loss_from_sequences(
            pred_state,
            pred_e,
            true_state,
            true_e,
            rollout_weights=weights,
            tail_start_fraction=2.0 / 3.0,
            field_ref_scale=jnp.asarray(field_ref, dtype=jnp.float64),
            tail_ref_scale=jnp.asarray(0.0, dtype=jnp.float64),
        )
        k_weights = np.array([1.0, 2.0, 1.0], dtype=np.float64)
        field_num = (
            1.0 * (2.0 * abs(1.0 - 1.0e-8) ** 2 + 1.0 * abs(2.0 - 0.0) ** 2)
            + 0.5 * (2.0 * abs(3.0 - 1.0) ** 2 + 1.0 * abs(4.0 - 0.0) ** 2)
        )
        field_true = (
            1.0 * (k_weights[1] * abs(1.0e-8) ** 2 + k_weights[2] * 0.0)
            + 0.5 * (k_weights[1] * abs(1.0) ** 2 + k_weights[2] * 0.0)
        )
        expected = field_num / (field_true + field_ref + 1e-30)
        self.assertTrue(math.isfinite(float(field_loss)))
        self.assertAlmostEqual(float(field_loss), float(expected), places=12)
        self.assertAlmostEqual(float(tail_loss), 0.0, places=12)

    def test_window_hybrid_tail_loss_matches_closed_form(self) -> None:
        pred_state = jnp.zeros((2, 4, 3), dtype=jnp.complex128)
        true_state = jnp.zeros((2, 4, 3), dtype=jnp.complex128)
        pred_state = pred_state.at[0, 3, 1].set(3.0 + 0.0j)
        pred_state = pred_state.at[1, 3, 1].set(4.0 + 0.0j)
        true_state = true_state.at[0, 3, 1].set(1.0 + 0.0j)
        true_state = true_state.at[1, 3, 1].set(2.0 + 0.0j)
        pred_e = jnp.zeros((2, 3), dtype=jnp.complex128)
        true_e = jnp.zeros((2, 3), dtype=jnp.complex128)
        weights = jnp.array([1.0, 0.5], dtype=jnp.float64)
        tail_ref = 5.0
        field_loss, tail_loss = train_mod.rollout_window_hybrid_loss_from_sequences(
            pred_state,
            pred_e,
            true_state,
            true_e,
            rollout_weights=weights,
            tail_start_fraction=2.0 / 3.0,
            field_ref_scale=jnp.asarray(0.0, dtype=jnp.float64),
            tail_ref_scale=jnp.asarray(tail_ref, dtype=jnp.float64),
        )
        # Nv=4 and tail_start_fraction=2/3 isolate the last retained mode only.
        k_weights = np.array([1.0, 2.0, 1.0], dtype=np.float64)
        excess0 = max(abs(3.0) ** 2 - abs(1.0) ** 2, 0.0)
        excess1 = max(abs(4.0) ** 2 - abs(2.0) ** 2, 0.0)
        tail_num = 1.0 * (k_weights[1] * (excess0 ** 2)) + 0.5 * (k_weights[1] * (excess1 ** 2))
        true_tail = 1.0 * (k_weights[1] * (abs(1.0) ** 4)) + 0.5 * (k_weights[1] * (abs(2.0) ** 4))
        expected = tail_num / (true_tail + tail_ref + 1e-30)
        self.assertAlmostEqual(float(field_loss), 0.0, places=12)
        self.assertTrue(math.isfinite(float(tail_loss)))
        self.assertAlmostEqual(float(tail_loss), float(expected), places=12)

    def test_reference_scales_match_mean_teacher_window_energies(self) -> None:
        weights = jnp.array([1.0, 0.5], dtype=jnp.float64)
        future_state = jnp.zeros((2, 2, 4, 3), dtype=jnp.complex128)
        future_e = jnp.zeros((2, 2, 3), dtype=jnp.complex128)

        future_e = future_e.at[0, 0, 1].set(1.0 + 0.0j)
        future_e = future_e.at[0, 1, 1].set(2.0 + 0.0j)
        future_state = future_state.at[0, 0, 3, 1].set(1.0 + 0.0j)
        future_state = future_state.at[0, 1, 3, 1].set(2.0 + 0.0j)

        future_e = future_e.at[1, 0, 1].set(2.0 + 0.0j)
        future_e = future_e.at[1, 1, 1].set(1.0 + 0.0j)
        future_state = future_state.at[1, 0, 3, 1].set(2.0 + 0.0j)
        future_state = future_state.at[1, 1, 3, 1].set(1.0 + 0.0j)

        prepared = {
            train_mod.REGIME_LINEAR: {
                "train_rollout": {
                    4: {
                        "future_state": future_state,
                        "future_E_hat": future_e,
                    }
                }
            }
        }
        scales = train_mod.build_stability_rollout_reference_scales(
            prepared,
            active_regimes=(train_mod.REGIME_LINEAR,),
            regime_rollout_nvs={train_mod.REGIME_LINEAR: (4,)},
            rollout_weights=weights,
            tail_start_fraction=2.0 / 3.0,
        )
        expected = [
            train_mod.rollout_window_teacher_scales(
                future_state[idx],
                future_e[idx],
                rollout_weights=weights,
                tail_start_fraction=2.0 / 3.0,
            )
            for idx in range(2)
        ]
        expected_field = float(np.mean([float(val[0]) for val in expected]))
        expected_tail = float(np.mean([float(val[1]) for val in expected]))
        self.assertAlmostEqual(float(scales[train_mod.REGIME_LINEAR][4]["field"]), expected_field, places=12)
        self.assertAlmostEqual(float(scales[train_mod.REGIME_LINEAR][4]["tail"]), expected_tail, places=12)

    def test_cubic_periodic_matches_scipy_wrap(self) -> None:
        values = np.sin(np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False))[None, :]
        coords = np.linspace(2.1, 13.7, 25, dtype=np.float64)[None, :]
        coeffs = cubic_bspline_prefilter_periodic(jnp.asarray(values), jnp.asarray((4.0 + 2.0 * np.cos(2.0 * math.pi * np.arange(16) / 16.0)) / 6.0))
        ours = np.asarray(cubic_bspline_interp_periodic(coeffs, jnp.asarray(coords)))[0]
        scipy_vals = map_coordinates(values[0], [coords[0]], order=3, mode="wrap")
        np.testing.assert_allclose(ours, scipy_vals, rtol=2e-2, atol=2e-2)

    def test_cubic_constant_reasonably_matches_scipy_constant(self) -> None:
        values = np.exp(-0.5 * (np.linspace(-2.0, 2.0, 17) ** 2))
        coords = np.linspace(1.2, 14.3, 17, dtype=np.float64)[:, None]
        coeffs = cubic_bspline_prefilter_constant(
            jnp.asarray(values[:, None]),
            jnp.ones((16,), dtype=jnp.float64),
            4.0 * jnp.ones((17,), dtype=jnp.float64),
            jnp.ones((16,), dtype=jnp.float64),
        )
        ours = np.asarray(cubic_bspline_interp_constant(coeffs, jnp.asarray(coords), cval=0.0)).ravel()
        scipy_vals = map_coordinates(values, [coords[:, 0]], order=3, mode="constant", cval=0.0)
        np.testing.assert_allclose(ours, scipy_vals, rtol=8e-2, atol=8e-2)

    def test_x_advection_matches_exact_shift_for_rowwise_sine(self) -> None:
        config = PhysicalGridVlasovPoissonConfig(Nx=16, Nv=9, Lx=4.0 * math.pi, vmin=-2.0, vmax=2.0, dt=0.05, T=0.05)
        ops_den = jnp.asarray((4.0 + 2.0 * np.cos(2.0 * math.pi * np.arange(config.Nx) / float(config.Nx))) / 6.0, dtype=jnp.float64)
        ops = {
            "periodic_den": ops_den,
            "v": config.v,
            "x_index": jnp.arange(config.Nx, dtype=jnp.float64)[None, :],
        }
        f = jnp.sin(config.x)[None, :].repeat(config.Nv, axis=0)
        shifted = np.asarray(advect_x_cubic(f, config, ops, 0.2))
        exact = np.sin(np.asarray(config.x)[None, :] - np.asarray(config.v)[:, None] * 0.2)
        np.testing.assert_allclose(shifted, exact, rtol=5e-3, atol=5e-3)

    def test_v_advection_matches_exact_shift_for_constant_field(self) -> None:
        config = PhysicalGridVlasovPoissonConfig(Nx=7, Nv=33, Lx=2.0 * math.pi, vmin=-4.0, vmax=4.0, dt=0.05, T=0.05)
        ops = {
            "v": config.v,
            "v_prefilter_sub": jnp.ones((config.Nv - 1,), dtype=jnp.float64),
            "v_prefilter_diag": 4.0 * jnp.ones((config.Nv,), dtype=jnp.float64),
            "v_prefilter_sup": jnp.ones((config.Nv - 1,), dtype=jnp.float64),
        }
        profile = np.exp(-0.5 * (np.asarray(config.v) - 0.4) ** 2)
        f = jnp.asarray(profile[:, None] * np.ones((1, config.Nx), dtype=np.float64))
        advected = np.asarray(advect_v_cubic(f, config, ops, jnp.full((config.Nx,), 0.3), 0.25))
        exact = np.exp(-0.5 * (np.asarray(config.v)[:, None] + 0.3 * 0.25 - 0.4) ** 2)
        exact = np.repeat(exact, config.Nx, axis=1)
        np.testing.assert_allclose(advected, exact, rtol=4e-2, atol=4e-2)

    def test_physical_grid_poisson_single_mode(self) -> None:
        config = PhysicalGridVlasovPoissonConfig(Nx=32, Nv=64, Lx=4.0 * math.pi, vmin=-8.0, vmax=8.0, dt=0.05, T=0.05)
        rho_mode = 0.05 * np.cos(0.5 * np.asarray(config.x))
        equilibrium = normalize_density_on_grid(gaussian_pdf(config.v, 0.0, 1.0), config.v)
        f = equilibrium[:, None] * (1.0 + rho_mode[None, :])
        E = np.asarray(compute_electric_field_from_distribution(f, config))
        exact = -(0.05 / 0.5) * np.sin(0.5 * np.asarray(config.x))
        np.testing.assert_allclose(E, exact, rtol=5e-3, atol=5e-3)

    def test_projection_recovers_maxwellian_and_scaled_bump_coeffs(self) -> None:
        Nx = 8
        v = jnp.linspace(-16.0, 16.0, 2001, dtype=jnp.float64)
        maxwell = gaussian_pdf(v, 0.0, 1.0)
        maxwell = normalize_density_on_grid(maxwell, v)
        f_maxwell = maxwell[:, None] * jnp.ones((1, Nx), dtype=jnp.float64)
        a_hat = np.asarray(project_distribution_snapshot_to_fourier_hermite(f_maxwell, v, 5, vth=1.0))
        a0 = a_hat[:, 0] / Nx
        self.assertAlmostEqual(float(np.real(a0[0])), 1.0, places=3)
        np.testing.assert_allclose(np.real(a0[1:]), 0.0, atol=3e-3)

        vth = 3.0
        bump = 0.9 * gaussian_pdf(v, -3.0, 1.0) + 0.1 * gaussian_pdf(v, 4.5, 0.5)
        f_bump = bump[:, None] * jnp.ones((1, Nx), dtype=jnp.float64)
        a_hat_bump = np.asarray(project_distribution_snapshot_to_fourier_hermite(f_bump, v, 6, vth=vth))
        coeffs = np.real(a_hat_bump[:, 0] / Nx)
        expected = np.asarray(equilibrium_coeffs_bump_on_tail(6, -3.0, 4.5, vth=vth))
        np.testing.assert_allclose(coeffs[:4], expected[:4], rtol=2e-2, atol=2e-2)

    def test_trainer_writes_checkpoint_and_metrics_on_tiny_physical_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            cache = Path(tmpdir) / "shared_dataset.npz"
            train_main(
                [
                    "--checkpoint",
                    str(ckpt),
                    "--dataset-cache",
                    str(cache),
                    "--Nv-targets",
                    "4",
                    "--Nm",
                    "1",
                    "--hidden-width",
                    "8",
                    "--res-blocks",
                    "1",
                    "--epochs",
                    "1",
                    "--log-every",
                    "1",
                    "--teacher-Nx",
                    "8",
                    "--teacher-Nv",
                    "16",
                    "--teacher-vmin",
                    "-6",
                    "--teacher-vmax",
                    "6",
                    "--teacher-dt",
                    "0.05",
                    "--teacher-proj-Nv",
                    "5",
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
            self.assertTrue(ckpt.exists())
            self.assertTrue(cache.exists())
            self.assertTrue(ckpt.with_suffix(".metrics.npz").exists())
            self.assertTrue(ckpt.with_suffix(".loss.png").exists())
            loaded = load_learned_interface_closure_npz(ckpt)
            self.assertEqual(loaded.teacher_backend, "grid_cubic_spline")
            self.assertTrue(loaded.include_global_indicators)
            self.assertEqual(loaded.input_dim, 6)

    def test_stability_aware_trainer_writes_finite_component_histories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface_stability.npz"
            cache = Path(tmpdir) / "shared_dataset_stability.npz"
            train_main(
                [
                    "--checkpoint",
                    str(ckpt),
                    "--dataset-cache",
                    str(cache),
                    "--Nv-targets",
                    "4",
                    "--Nm",
                    "1",
                    "--hidden-width",
                    "8",
                    "--res-blocks",
                    "1",
                    "--epochs",
                    "1",
                    "--log-every",
                    "1",
                    "--batch-size",
                    "4",
                    "--rollout-batch-size",
                    "2",
                    "--steps-per-epoch",
                    "1",
                    "--train-objective",
                    "stability_aware",
                    "--context-mode",
                    "lag1_delta",
                    "--rollout-horizon",
                    "2",
                    "--teacher-Nx",
                    "8",
                    "--teacher-Nv",
                    "16",
                    "--teacher-vmin",
                    "-6",
                    "--teacher-vmax",
                    "6",
                    "--teacher-dt",
                    "0.05",
                    "--teacher-proj-Nv",
                    "5",
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
            self.assertTrue(ckpt.exists())
            metrics_path = ckpt.with_suffix(".metrics.npz")
            self.assertTrue(metrics_path.exists())
            loaded = load_learned_interface_closure_npz(ckpt)
            self.assertEqual(loaded.stability_loss_definition, train_mod.STABILITY_LOSS_DEFINITION)
            with np.load(metrics_path) as data:
                self.assertEqual(
                    str(np.asarray(data["stability_loss_definition"], dtype=np.str_).reshape(-1)[0]),
                    train_mod.STABILITY_LOSS_DEFINITION,
                )
                for key in ("train_loss", "train_loss_q", "train_loss_field", "train_loss_tail", "train_loss_reg"):
                    self.assertTrue(np.isfinite(np.asarray(data[key], dtype=np.float64)).all(), msg=key)

    def test_online_rollout_loss_is_jax_differentiable_on_tiny_episode(self) -> None:
        target_nv = 4
        teacher_Nx = 8
        teacher_Nv = 16
        teacher_L = 4.0 * math.pi
        teacher_dt = 0.05
        teacher_vmin = -6.0
        teacher_vmax = 6.0
        online_v_probes = 8

        online_dataset, _ = train_mod.build_online_reference_dataset(
            regimes=(train_mod.REGIME_LINEAR,),
            teacher_Nx=teacher_Nx,
            teacher_Nv=teacher_Nv,
            teacher_L=teacher_L,
            teacher_vmin=teacher_vmin,
            teacher_vmax=teacher_vmax,
            teacher_dt=teacher_dt,
            linear_T=0.10,
            linear_eps=1e-2,
            linear_modes=(0.5,),
            linear_num_samples=1,
            linear_seed=0,
            linear_poisson_sign=1.0,
            nonlinear_T=0.10,
            nonlinear_k0=0.5,
            nonlinear_poisson_sign=1.0,
            weak_eps=(0.05,),
            strong_eps=(0.25,),
            val_fraction=0.2,
            online_v_probes=online_v_probes,
        )
        stats = train_mod.build_identity_training_stats(Nm=1, context_mode="none")
        params = train_mod.init_interface_closure_params(
            jax.random.PRNGKey(0),
            input_dim=int(stats["input_mean"].shape[0]),
            hidden_width=8,
            res_blocks=1,
        )
        integ = FourierHermiteIMEX(
            Nx=teacher_Nx,
            Nv=target_nv,
            Lx=teacher_L,
            dt=teacher_dt,
            vth=1.0,
            dealias_23=False,
            closure=None,
        )
        loss_fn, active_regimes = train_mod.make_online_trajectory_batch_loss(
            online_dataset=online_dataset,
            regime_weights={train_mod.REGIME_LINEAR: 1.0},
            Nm=1,
            k_scale=float(jnp.max(jnp.asarray(integ.k_arr[1:], dtype=jnp.float64))),
            nv_scale=float(target_nv),
            stats=stats,
            hidden_width=8,
            res_blocks=1,
            Nv_targets=(target_nv,),
            train_regimes=(train_mod.REGIME_LINEAR,),
            teacher_backend="grid_cubic_spline",
            teacher_Lx=teacher_L,
            teacher_Nx=teacher_Nx,
            teacher_Nv=teacher_Nv,
            teacher_vmin=teacher_vmin,
            teacher_vmax=teacher_vmax,
            teacher_dt=teacher_dt,
            n_low=2,
            context_mode="none",
            tail_start_fraction=2.0 / 3.0,
            loss_backend=train_mod.ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1,
            lambda_E=0.5,
            lambda_dist=1.0,
            lambda_tail=0.05,
            lambda_neg=0.01,
            lambda_reg=1e-6,
            online_v_probes=online_v_probes,
            nonlinear_T=0.10,
            nonlinear_k0=0.5,
            poisson_sign=1.0,
            rollout_dealias_23=False,
        )
        self.assertEqual(tuple(active_regimes), (train_mod.REGIME_LINEAR,))
        regime_batches = {
            regime: online_dataset[regime]["train"]
            for regime in active_regimes
        }
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, regime_batches)
        self.assertTrue(np.isfinite(float(loss)))
        for key, value in aux.items():
            self.assertTrue(np.isfinite(np.asarray(value, dtype=np.float64)).all(), msg=key)
        for leaf in jax.tree_util.tree_leaves(grads):
            self.assertTrue(np.isfinite(np.asarray(leaf, dtype=np.float64)).all())

    def test_online_reconstruction_and_penalties_are_finite_and_nontrivial(self) -> None:
        Nx = 4
        Nv = 6
        integ = FourierHermiteIMEX(
            Nx=Nx,
            Nv=Nv,
            Lx=4.0 * math.pi,
            dt=0.05,
            vth=1.0,
            dealias_23=False,
            closure=None,
        )
        v_probe = jnp.linspace(-4.0, 4.0, 9, dtype=jnp.float64)
        eq_probe = train_mod.maxwellian_equilibrium(v_probe)
        nk = int(integ.k_arr.shape[0])
        a_hat_hist = jnp.zeros((2, Nv, nk), dtype=jnp.complex128)
        a_hat_hist = a_hat_hist.at[:, 0, 0].set(complex(-4.0 * Nx, 0.0))
        a_hat_hist = a_hat_hist.at[:, -1, 0].set(complex(1.0 * Nx, 0.0))
        delta_f = train_mod.reconstruct_delta_f_from_a_hat_history(
            a_hat_hist,
            Nx=Nx,
            v_probe=v_probe,
            vth=1.0,
        )
        self.assertEqual(tuple(delta_f.shape), (2, int(v_probe.shape[0]), Nx))
        self.assertTrue(np.isfinite(np.asarray(delta_f, dtype=np.float64)).all())

        field_loss, dist_loss, tail_loss, neg_loss = train_mod.online_trajectory_loss_terms(
            a_hat_hist,
            k_arr=integ.k_arr,
            ref_E_hat=jnp.zeros((2, nk), dtype=jnp.complex128),
            ref_delta_f=0.5 * delta_f,
            Nx=Nx,
            v_probe=v_probe,
            eq_probe=eq_probe,
            tail_start_fraction=2.0 / 3.0,
            poisson_sign=1.0,
        )
        self.assertAlmostEqual(float(field_loss), 0.0, places=12)
        self.assertTrue(np.isfinite(float(dist_loss)))
        self.assertTrue(np.isfinite(float(tail_loss)))
        self.assertTrue(np.isfinite(float(neg_loss)))
        self.assertGreater(float(dist_loss), 0.0)
        self.assertGreater(float(tail_loss), 0.0)
        self.assertGreater(float(neg_loss), 0.0)

    def test_incompatible_dataset_cache_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "shared_dataset.npz"
            np.savez(
                cache,
                dataset_format=np.array(["landau_interface_dataset_physical_teacher_v2"], dtype=np.str_),
                regimes=np.array([train_mod.REGIME_LINEAR], dtype=np.str_),
                n_low=np.array([2], dtype=np.int32),
            )
            Nm = 6
            dummy_dataset = {
                "train_inputs_base": np.zeros((2, 2 * Nm + 4), dtype=np.float64),
                "train_targets": np.zeros((2, 2), dtype=np.float64),
                "val_inputs_base": np.zeros((1, 2 * Nm + 4), dtype=np.float64),
                "val_targets": np.zeros((1, 2), dtype=np.float64),
            }
            with mock.patch.object(train_mod, "build_linear_landau_regime", return_value=dummy_dataset) as patched:
                dataset = train_mod.build_mixed_landau_dataset(
                    dataset_cache=cache,
                    regimes=(train_mod.REGIME_LINEAR,),
                    teacher_backend="grid_cubic_spline",
                    teacher_Nx=8,
                    teacher_Nv=16,
                    teacher_L=4.0 * math.pi,
                    teacher_vmin=-6.0,
                    teacher_vmax=6.0,
                    teacher_dt=0.05,
                    teacher_proj_Nv=301,
                    linear_T=0.1,
                    linear_eps=1e-2,
                    linear_modes=(0.5, 1.0),
                    linear_num_samples=1,
                    linear_seed=0,
                    linear_poisson_sign=1.0,
                    linear_history_stride=1,
                    nonlinear_T=0.1,
                    nonlinear_k0=0.5,
                    nonlinear_poisson_sign=1.0,
                    nonlinear_history_stride=1,
                    weak_eps=(0.05,),
                    strong_eps=(0.25,),
                    Nv_targets=(4, 6, 8, 10, 12, 20, 40, 80, 160, 300),
                    Nm=Nm,
                    val_fraction=0.2,
                    n_low=2,
                )
            patched.assert_called_once()
            self.assertIn(train_mod.REGIME_LINEAR, dataset)
            with np.load(cache) as data:
                self.assertEqual(str(np.asarray(data["dataset_format"]).reshape(-1)[0]), train_mod.CACHE_FORMAT)
                np.testing.assert_array_equal(
                    np.asarray(data["Nv_targets"], dtype=np.int32),
                    np.array([4, 6, 8, 10, 12, 20, 40, 80, 160, 300], dtype=np.int32),
                )
                self.assertEqual(int(np.asarray(data["Nm"]).reshape(-1)[0]), Nm)

    def test_cached_nv_superset_can_be_reused_for_single_nv_training(self) -> None:
        Nm = 1
        nv_col = 2 * Nm + 1
        shared_dataset = {
            train_mod.REGIME_LINEAR: {
                "train_inputs_base": np.array(
                    [
                        [0.1, 0.2, 0.5, 6.0, 1.0, 2.0],
                        [0.3, 0.4, 0.5, 8.0, 1.1, 2.1],
                    ],
                    dtype=np.float64,
                ),
                "train_targets": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                "val_inputs_base": np.array(
                    [
                        [0.5, 0.6, 0.5, 6.0, 1.2, 2.2],
                        [0.7, 0.8, 0.5, 8.0, 1.3, 2.3],
                    ],
                    dtype=np.float64,
                ),
                "val_targets": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
            }
        }
        metadata = train_mod.build_dataset_cache_metadata(
            regimes=(train_mod.REGIME_LINEAR,),
            teacher_backend="grid_cubic_spline",
            teacher_Nx=8,
            teacher_Nv=16,
            teacher_L=4.0 * math.pi,
            teacher_vmin=-6.0,
            teacher_vmax=6.0,
            teacher_dt=0.05,
            teacher_proj_Nv=9,
            linear_T=0.1,
            linear_eps=1e-2,
            linear_modes=(0.5,),
            linear_num_samples=1,
            linear_seed=0,
            linear_poisson_sign=1.0,
            linear_history_stride=1,
            nonlinear_T=0.1,
            nonlinear_k0=0.5,
            nonlinear_poisson_sign=1.0,
            nonlinear_history_stride=1,
            weak_eps=(0.05,),
            strong_eps=(0.25,),
            Nv_targets=(6, 8),
            Nm=Nm,
            val_fraction=0.2,
            n_low=2,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "shared_dataset.npz"
            train_mod.save_dataset_cache(cache, shared_dataset, metadata=metadata)
            filtered = train_mod.build_mixed_landau_dataset(
                dataset_cache=cache,
                regimes=(train_mod.REGIME_LINEAR,),
                teacher_backend="grid_cubic_spline",
                teacher_Nx=8,
                teacher_Nv=16,
                teacher_L=4.0 * math.pi,
                teacher_vmin=-6.0,
                teacher_vmax=6.0,
                teacher_dt=0.05,
                teacher_proj_Nv=9,
                linear_T=0.1,
                linear_eps=1e-2,
                linear_modes=(0.5,),
                linear_num_samples=1,
                linear_seed=0,
                linear_poisson_sign=1.0,
                linear_history_stride=1,
                nonlinear_T=0.1,
                nonlinear_k0=0.5,
                nonlinear_poisson_sign=1.0,
                nonlinear_history_stride=1,
                weak_eps=(0.05,),
                strong_eps=(0.25,),
                Nv_targets=(8,),
                Nm=Nm,
                val_fraction=0.2,
                n_low=2,
                allow_cached_nv_superset=True,
            )
        train_inputs = filtered[train_mod.REGIME_LINEAR]["train_inputs_base"]
        val_inputs = filtered[train_mod.REGIME_LINEAR]["val_inputs_base"]
        self.assertEqual(train_inputs.shape[0], 1)
        self.assertEqual(val_inputs.shape[0], 1)
        self.assertEqual(int(round(train_inputs[0, nv_col])), 8)
        self.assertEqual(int(round(val_inputs[0, nv_col])), 8)

    def test_trainer_rejects_nv_targets_smaller_than_nm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            with self.assertRaisesRegex(ValueError, r"Nv >= Nm"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--Nv-targets",
                        "4,6,8",
                        "--Nm",
                        "6",
                    ]
                )

    def test_stability_aware_training_requires_minibatch_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            with self.assertRaisesRegex(ValueError, r"batch-size > 0"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                        "--hidden-width",
                        "8",
                        "--res-blocks",
                        "1",
                        "--epochs",
                        "1",
                        "--teacher-Nx",
                        "8",
                        "--teacher-Nv",
                        "16",
                        "--teacher-vmin",
                        "-6",
                        "--teacher-vmax",
                        "6",
                        "--teacher-dt",
                        "0.05",
                        "--teacher-proj-Nv",
                        "5",
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
                        "--train-objective",
                        "stability_aware",
                    ]
                )

    def test_grid_teacher_accepts_projection_options(self) -> None:
        captured = {}

        def fake_build_mixed_landau_dataset(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after dataset build")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            with mock.patch.object(
                train_mod,
                "build_mixed_landau_dataset",
                side_effect=fake_build_mixed_landau_dataset,
            ):
                with self.assertRaisesRegex(RuntimeError, r"stop after dataset build"):
                    train_main(
                        [
                            "--checkpoint",
                            str(ckpt),
                            "--teacher-backend",
                            "grid_cubic_spline",
                            "--Nv-targets",
                            "4",
                            "--Nm",
                            "1",
                            "--teacher-proj-Nv",
                            "5",
                            "--per-target-projection-orders",
                        ]
                    )

        self.assertEqual(captured["teacher_backend"], "grid_cubic_spline")
        self.assertEqual(captured["teacher_proj_Nv"], 5)
        self.assertTrue(captured["per_target_projection_orders"])

    def test_higher_order_hermite_rejects_projection_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            with self.assertRaisesRegex(ValueError, r"does not use --teacher-proj-Nv"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--teacher-backend",
                        "higher_order_hermite",
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                        "--teacher-Nv",
                        "8",
                        "--teacher-proj-Nv",
                        "5",
                    ]
                )
            with self.assertRaisesRegex(ValueError, r"does not support --per-target-projection-orders"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--teacher-backend",
                        "higher_order_hermite",
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                        "--teacher-Nv",
                        "8",
                        "--per-target-projection-orders",
                    ]
                )

    def test_higher_order_hermite_dataset_builds_landau_regimes(self) -> None:
        dataset = train_mod.build_mixed_landau_dataset(
            dataset_cache=None,
            regimes=(
                train_mod.REGIME_LINEAR,
                train_mod.REGIME_WEAK,
                train_mod.REGIME_STRONG,
            ),
            teacher_backend="higher_order_hermite",
            teacher_Nx=8,
            teacher_Nv=6,
            teacher_L=4.0 * math.pi,
            teacher_vmin=-6.0,
            teacher_vmax=6.0,
            teacher_dt=0.05,
            teacher_proj_Nv=None,
            linear_T=0.1,
            linear_eps=1e-2,
            linear_modes=(0.5,),
            linear_num_samples=1,
            linear_seed=0,
            linear_poisson_sign=1.0,
            linear_history_stride=1,
            nonlinear_T=0.1,
            nonlinear_k0=0.5,
            nonlinear_poisson_sign=1.0,
            nonlinear_history_stride=1,
            weak_eps=(0.05,),
            strong_eps=(0.25,),
            Nv_targets=(4,),
            Nm=1,
            val_fraction=0.2,
            n_low=2,
        )
        for regime in (train_mod.REGIME_LINEAR, train_mod.REGIME_WEAK, train_mod.REGIME_STRONG):
            self.assertIn(regime, dataset)
            self.assertGreater(dataset[regime]["train_inputs_base"].shape[0], 0, msg=regime)
            self.assertGreater(dataset[regime]["val_inputs_base"].shape[0], 0, msg=regime)

    def test_q_only_dataset_build_ignores_rollout_horizon(self) -> None:
        captured = {}

        def fake_build_mixed_landau_dataset(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after dataset build")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            with mock.patch.object(
                train_mod,
                "build_mixed_landau_dataset",
                side_effect=fake_build_mixed_landau_dataset,
            ):
                with self.assertRaisesRegex(RuntimeError, r"stop after dataset build"):
                    train_main(
                        [
                            "--checkpoint",
                            str(ckpt),
                            "--Nv-targets",
                            "4",
                            "--Nm",
                            "1",
                            "--train-objective",
                            "q_only",
                            "--rollout-horizon",
                            "5",
                        ]
                    )

        self.assertEqual(captured["rollout_horizon"], 0)

    def test_online_rollout_rejects_offline_cache_and_projection_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            cache = Path(tmpdir) / "shared_dataset.npz"
            with self.assertRaisesRegex(ValueError, r"does not support --dataset-cache"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory",
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                        "--dataset-cache",
                        str(cache),
                    ]
                )
            with self.assertRaisesRegex(ValueError, r"does not support --build-dataset-only"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory",
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                        "--build-dataset-only",
                    ]
                )
            with self.assertRaisesRegex(ValueError, r"does not support --allow-dataset-cache-nv-superset"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory",
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                        "--allow-dataset-cache-nv-superset",
                    ]
                )
            with self.assertRaisesRegex(ValueError, r"does not support --per-target-projection-orders"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory",
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                        "--per-target-projection-orders",
                    ]
                )

    def test_online_rollout_rejects_higher_order_hermite_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            with self.assertRaisesRegex(ValueError, r"only supports teacher_backend=grid_cubic_spline"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory",
                        "--teacher-backend",
                        "higher_order_hermite",
                        "--teacher-Nv",
                        "8",
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                    ]
                )

    def test_online_rollout_requires_exactly_one_target_nv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            with self.assertRaisesRegex(ValueError, r"requires exactly one target Nv"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory",
                        "--Nv-targets",
                        "4,6",
                        "--Nm",
                        "1",
                    ]
                )

    def test_learned_rollout_runs_for_linear_and_nonlinear_landau(self) -> None:
        closure = _make_closure()
        linear = run_linear_landau_rollout_raw(
            LinearLandauConfig(method="learned", Nv=4, Nx=8, dt=0.05, T=0.10),
            learned_closure=closure,
            solver_backend="cnab2",
            return_state_history=True,
        )
        self.assertIn("a_hat_hist", linear)

        params = NonlinearLandauParams(Nx=8, Nv=6, dt=0.05, T=0.10, snapshot_times=(0.0,))
        nonlinear = run_nonlinear_landau_rollout_raw(
            params,
            "learned",
            learned_closure=closure,
            return_state_history=True,
            history_stride=1,
        )
        self.assertIn("a_hat_hist", nonlinear)
        self.assertEqual(np.asarray(nonlinear["a_hat_hist"]).shape[1], 6)

    def test_zero_output_closure_matches_truncation_for_nonlinear_cnab2(self) -> None:
        closure = _make_closure()
        params = NonlinearLandauParams(Nx=8, Nv=6, dt=0.05, T=0.10, snapshot_times=(0.0,))
        trunc = run_nonlinear_landau_rollout_raw(
            params,
            "truncation",
            return_state_history=True,
            history_stride=1,
        )
        learned = run_nonlinear_landau_rollout_raw(
            params,
            "learned",
            learned_closure=closure,
            return_state_history=True,
            history_stride=1,
        )
        np.testing.assert_allclose(
            np.asarray(learned["a_hat_hist"]),
            np.asarray(trunc["a_hat_hist"]),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_nonlinear_physical_grid_runtimes_run(self) -> None:
        x, v, snaps, times, energy = simulate_two_stream(TwoStreamParams(Nx=16, Nv=16, T=0.2, snapshot_times=(0.0, 0.1, 0.2)))
        self.assertEqual(x.shape[0], 16)
        self.assertEqual(v.shape[0], 16)
        self.assertEqual(len(snaps), 3)
        self.assertEqual(times.shape[0], 3)
        self.assertEqual(energy.shape[0], 3)

        x, v, snaps, times, energy = simulate_bump_on_tail(
            BumpOnTailParams(Nx=16, Nv=24, T=0.2, snapshot_times=(0.0, 0.1, 0.2)),
            system="C",
        )
        self.assertEqual(x.shape[0], 16)
        self.assertEqual(v.shape[0], 24)
        self.assertEqual(len(snaps), 3)
        self.assertEqual(times.shape[0], 3)
        self.assertEqual(energy.shape[0], 3)

    def test_fig3_and_fig4_reject_learned_checkpoints(self) -> None:
        with self.assertRaisesRegex(ValueError, "state-dependent"):
            Fig3ResponseFunction(learned_checkpoint="dummy.npz").run()
        with self.assertRaisesRegex(ValueError, "state-dependent"):
            Fig4EigenvalueScan(learned_checkpoint="dummy.npz").run()


if __name__ == "__main__":
    unittest.main()
