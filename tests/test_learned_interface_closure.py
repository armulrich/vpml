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
    learned_interface_q_hat,
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
    rollout_anchor_samples: int = 0,
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
    equilibrium_centered: bool = False,
    complex_normalization_mode: str = "componentwise",
    translation_augmented: bool = False,
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
        equilibrium_centered=equilibrium_centered,
        complex_normalization_mode=complex_normalization_mode,
        translation_augmented=translation_augmented,
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
        rollout_anchor_samples=rollout_anchor_samples,
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
        closure = _make_closure(
            target_bias=(1.5, -0.25),
            equilibrium_centered=True,
            complex_normalization_mode="phase_isotropic",
            translation_augmented=True,
        )
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
        self.assertTrue(loaded.equilibrium_centered)
        self.assertEqual(loaded.complex_normalization_mode, "phase_isotropic")
        self.assertTrue(loaded.translation_augmented)
        np.testing.assert_allclose(np.asarray(loaded.input_mean), np.asarray(closure.input_mean))
        np.testing.assert_allclose(np.asarray(loaded.target_mean), np.asarray(closure.target_mean))
        np.testing.assert_allclose(np.asarray(loaded.params["W_lin"]), np.asarray(closure.params["W_lin"]))

    def test_checkpoint_loader_defaults_new_symmetry_metadata_for_legacy_npz(self) -> None:
        params = _zero_interface_params(6)
        params["W_lin"] = params["W_lin"].at[0, 0].set(0.75).at[1, 1].set(-0.5)
        params["b_lin"] = jnp.array([0.25, -0.125], dtype=jnp.float64)
        closure = _make_closure(params=params, target_bias=(1.0, -2.0))
        features = jnp.array([[0.4, -0.3, 0.5, 0.25, 0.1, 0.2]], dtype=jnp.float64)
        expected = np.asarray(closure.predict_q_components(features))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interface_closure.npz"
            save_learned_interface_closure_npz(path, closure)
            with np.load(path) as data:
                payload = {
                    name: np.asarray(data[name])
                    for name in data.files
                    if name
                    not in {
                        "equilibrium_centered",
                        "complex_normalization_mode",
                        "translation_augmented",
                    }
                }
            np.savez(path, **payload)
            loaded = load_learned_interface_closure_npz(path)
        self.assertFalse(loaded.equilibrium_centered)
        self.assertEqual(loaded.complex_normalization_mode, "componentwise")
        self.assertFalse(loaded.translation_augmented)
        np.testing.assert_array_equal(np.asarray(loaded.predict_q_components(features)), expected)

    def test_warm_start_rejects_symmetry_metadata_mismatch(self) -> None:
        closure = _make_closure(
            equilibrium_centered=True,
            complex_normalization_mode="phase_isotropic",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interface_closure.npz"
            save_learned_interface_closure_npz(path, closure)
            common = {
                "Nm": closure.Nm,
                "hidden_width": closure.hidden_width,
                "res_blocks": closure.res_blocks,
                "Nv_targets": closure.Nv_targets,
                "context_mode": closure.context_mode,
            }
            with self.assertRaisesRegex(ValueError, "equilibrium_centered metadata"):
                train_mod._load_init_checkpoint_for_interface_closure(
                    path,
                    equilibrium_centered=False,
                    complex_normalization_mode="phase_isotropic",
                    **common,
                )
            with self.assertRaisesRegex(ValueError, "complex_normalization_mode metadata"):
                train_mod._load_init_checkpoint_for_interface_closure(
                    path,
                    equilibrium_centered=True,
                    complex_normalization_mode="componentwise",
                    **common,
                )

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

    def test_checkpoint_round_trip_preserves_fourier_hermite_bidir_metadata(self) -> None:
        closure = _make_closure(
            training_mode="online_rollout",
            train_objective="trajectory",
            loss_backend=train_mod.ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR,
            rollout_horizon=3,
            rollout_anchor_samples=2,
            online_v_probes=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interface_closure.npz"
            save_learned_interface_closure_npz(path, closure)
            loaded = load_learned_interface_closure_npz(path)
        self.assertEqual(loaded.training_mode, "online_rollout")
        self.assertEqual(loaded.train_objective, "trajectory")
        self.assertEqual(loaded.loss_backend, train_mod.ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR)
        self.assertEqual(loaded.rollout_horizon, 3)
        self.assertEqual(loaded.rollout_anchor_samples, 2)
        self.assertEqual(loaded.online_v_probes, 0)
        self.assertAlmostEqual(loaded.lambda_E, 0.0)
        self.assertAlmostEqual(loaded.lambda_dist, 0.0)
        self.assertAlmostEqual(loaded.lambda_tail, 0.0)
        self.assertAlmostEqual(loaded.lambda_neg, 0.0)
        self.assertAlmostEqual(loaded.lambda_reg, 0.0)

    def test_checkpoint_round_trip_preserves_online_hybrid_metadata(self) -> None:
        closure = _make_closure(
            training_mode="online_rollout",
            train_objective="trajectory_q_hybrid",
            loss_backend=train_mod.ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1,
            lambda_q=1.0,
            lambda_E=0.5,
            lambda_dist=1.0,
            lambda_tail=0.05,
            lambda_neg=0.025,
            lambda_reg=1e-6,
            online_v_probes=64,
            stability_loss_definition=train_mod.ONLINE_HYBRID_LOSS_DEFINITION,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interface_closure.npz"
            save_learned_interface_closure_npz(path, closure)
            loaded = load_learned_interface_closure_npz(path)
        self.assertEqual(loaded.training_mode, "online_rollout")
        self.assertEqual(loaded.train_objective, "trajectory_q_hybrid")
        self.assertEqual(loaded.loss_backend, train_mod.ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1)
        self.assertAlmostEqual(loaded.lambda_q, 1.0)
        self.assertAlmostEqual(loaded.lambda_E, 0.5)
        self.assertAlmostEqual(loaded.lambda_dist, 1.0)
        self.assertAlmostEqual(loaded.lambda_tail, 0.05)
        self.assertAlmostEqual(loaded.lambda_neg, 0.025)
        self.assertAlmostEqual(loaded.lambda_reg, 1e-6)
        self.assertEqual(loaded.online_v_probes, 64)
        self.assertEqual(loaded.stability_loss_definition, train_mod.ONLINE_HYBRID_LOSS_DEFINITION)

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

    def test_online_field_distribution_boundary_flux_is_clipped(self) -> None:
        params = _zero_interface_params(6)
        params["b_lin"] = jnp.array([10.0, -10.0], dtype=jnp.float64)
        closure = _make_closure(
            params=params,
            training_mode="online_rollout",
            train_objective="trajectory",
            loss_backend=train_mod.ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1,
        )
        a_hat = jnp.zeros((4, 3), dtype=jnp.complex128)
        B_hat = learned_boundary_flux_hat(
            a_hat,
            jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64),
            Nv=4,
            vth=1.0,
            learned=closure,
        )
        expected = 0.25 * np.tanh(10.0 / 0.25) + 1j * 0.75 * np.tanh(-10.0 / 0.75)
        np.testing.assert_allclose(
            np.asarray(B_hat[-1, 1:]),
            np.full((2,), expected, dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_online_fourier_hermite_bidir_boundary_flux_is_not_clipped(self) -> None:
        params = _zero_interface_params(6)
        params["b_lin"] = jnp.array([10.0, -10.0], dtype=jnp.float64)
        closure = _make_closure(
            params=params,
            training_mode="online_rollout",
            train_objective="trajectory",
            loss_backend=train_mod.ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR,
        )
        a_hat = jnp.zeros((4, 3), dtype=jnp.complex128)
        B_hat = learned_boundary_flux_hat(
            a_hat,
            jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64),
            Nv=4,
            vth=1.0,
            learned=closure,
        )
        np.testing.assert_allclose(
            np.asarray(B_hat[-1, 1:]),
            np.full((2,), 10.0 - 10.0j, dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_online_full_state_loss_terms_upweight_tail_and_late_steps(self) -> None:
        ref = jnp.ones((3, 6, 3), dtype=jnp.complex128)

        pred_mid = ref.at[:, 3, 1].add(1.0)
        pred_tail = ref.at[:, 5, 1].add(1.0)
        pred_early = ref.at[0, 3, 1].add(1.0)
        pred_late = ref.at[2, 3, 1].add(1.0)

        num_mid, _ = train_mod.online_full_state_loss_terms(pred_mid, ref)
        num_tail, _ = train_mod.online_full_state_loss_terms(pred_tail, ref)
        num_early, _ = train_mod.online_full_state_loss_terms(pred_early, ref)
        num_late, _ = train_mod.online_full_state_loss_terms(pred_late, ref)

        self.assertGreater(float(num_tail), float(num_mid))
        self.assertGreater(float(num_late), float(num_early))

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

    def test_equilibrium_centering_makes_zero_state_flux_exactly_zero(self) -> None:
        params = _zero_interface_params(6)
        params["W_lin"] = params["W_lin"].at[0, 0].set(2.0)
        params["b_lin"] = jnp.array([3.0, -4.0], dtype=jnp.float64)
        params["b_out"] = jnp.array([-1.5, 2.5], dtype=jnp.float64)
        closure = _make_closure(
            params=params,
            target_bias=(7.0, -9.0),
            equilibrium_centered=True,
            complex_normalization_mode="phase_isotropic",
        )
        k_arr = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float64)
        zero_state = jnp.zeros((4, 3), dtype=jnp.complex128)
        zero_q = np.asarray(learned_interface_q_hat(zero_state, k_arr, 4, closure))
        np.testing.assert_array_equal(zero_q, np.zeros_like(zero_q))

        nonzero_state = zero_state.at[3, 1].set(0.25 + 0.0j)
        nonzero_q = np.asarray(learned_interface_q_hat(nonzero_state, k_arr, 4, closure))
        self.assertGreater(abs(nonzero_q[1]), 0.0)

    def test_phase_isotropic_stats_pair_phase_rotated_complex_components(self) -> None:
        rng = np.random.default_rng(19)
        phases = rng.uniform(-math.pi, math.pi, size=(64, 2))
        amplitudes = np.column_stack(
            [rng.uniform(0.5, 1.5, size=64), rng.uniform(1.5, 2.5, size=64)]
        )
        coeff = amplitudes * np.exp(1j * phases)
        invariant_features = rng.normal(size=(64, 4))
        inputs = np.concatenate([coeff.real, coeff.imag, invariant_features], axis=1)
        q = rng.uniform(0.25, 1.25, size=64) * np.exp(
            1j * rng.uniform(-math.pi, math.pi, size=64)
        )
        targets = np.column_stack([q.real, q.imag])
        stats = {
            "input_mean": np.mean(inputs, axis=0),
            "input_std": np.std(inputs, axis=0),
            "target_mean": np.mean(targets, axis=0),
            "target_std": np.std(targets, axis=0),
        }
        result = train_mod.phase_isotropic_complex_training_stats(
            stats,
            Nm=2,
            context_mode="none",
        )
        expected = np.sqrt(0.5 * np.mean(np.abs(coeff) ** 2, axis=0))
        expected_q = math.sqrt(0.5 * float(np.mean(np.abs(q) ** 2)))
        np.testing.assert_allclose(result["input_mean"][:4], 0.0)
        np.testing.assert_allclose(result["input_std"][:4], [expected[0], expected[1], expected[0], expected[1]])
        np.testing.assert_allclose(result["input_mean"][4:], stats["input_mean"][4:])
        np.testing.assert_allclose(result["input_std"][4:], stats["input_std"][4:])
        np.testing.assert_allclose(result["target_mean"], 0.0)
        np.testing.assert_allclose(result["target_std"], expected_q)

    def test_exact_q_regime_loss_std_uses_full_rollout_and_ladder(self) -> None:
        histories = np.zeros((1, 4, 3, 3), dtype=np.complex128)
        histories[0, :2, 1, 1:] = np.array([[1.0, 2.0], [3.0, 4.0]])
        histories[0, :2, 2, 1:] = np.array([[2.0, 1.0], [4.0, 3.0]])
        exact_dataset = {
            train_mod.REGIME_LINEAR: {
                train_mod.exact_q_rollout_coeff_key(2): histories,
            }
        }
        qpair_dataset = {
            train_mod.REGIME_LINEAR: {
                "train_anchor_case_indices": np.array([0, 0], dtype=np.int32),
                "train_anchor_time_indices": np.array([0, 0], dtype=np.int32),
                "train_anchor_target_nvs": np.array([1, 2], dtype=np.int32),
            }
        }
        k_arr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        scales = train_mod.exact_q_rollout_regime_loss_stds(
            exact_dataset,
            qpair_dataset,
            max_projection_order=2,
            target_nvs=(1, 2),
            k_arr=k_arr,
            rollout_horizon=2,
            chunk_size=1,
        )
        q_abs_sq = []
        for target_nv in (1, 2):
            coeff = histories[0, :2, target_nv, 1:]
            q_abs_sq.extend(
                (
                    float(target_nv)
                    * k_arr[1:][None, :] ** 2
                    * np.abs(coeff) ** 2
                ).reshape(-1)
            )
        expected = math.sqrt(0.5 * float(np.mean(q_abs_sq)))
        self.assertAlmostEqual(scales[train_mod.REGIME_LINEAR], expected)

    def test_exact_q_loss_scale_override_changes_only_error_denominator(self) -> None:
        closure = _make_closure(
            training_mode=train_mod.EXACT_Q_ROLLOUT_TRAINING_MODE,
            train_objective=train_mod.EXACT_Q_ROLLOUT_OBJECTIVE,
            rollout_horizon=1,
        )
        integ = FourierHermiteIMEX(
            Nx=4,
            Nv=4,
            Lx=4.0 * math.pi,
            dt=0.05,
            vth=1.0,
            dealias_23=False,
            closure=None,
        )
        anchors = jnp.zeros((1, 3, 4, 3), dtype=jnp.complex128)
        ref_q = jnp.array(
            [[[0.0 + 0.0j, 1.0 + 2.0j, 3.0 + 4.0j]]],
            dtype=jnp.complex128,
        )
        common = {
            "learned": closure,
            "forward_integ": integ,
            "rollout_horizon": 1,
            "explicit_n_hat_fn": lambda state, *, integ: jnp.zeros_like(state),
            "rollout_precision": train_mod.EXACT_ROLLOUT_PRECISION_FLOAT64,
        }
        default_loss = train_mod.exact_q_rollout_loss_for_anchor_batch(
            anchors,
            ref_q,
            jnp.zeros((1,), dtype=jnp.int32),
            **common,
        )
        scaled_loss = train_mod.exact_q_rollout_loss_for_anchor_batch(
            anchors,
            ref_q,
            jnp.zeros((1,), dtype=jnp.int32),
            loss_target_std=jnp.full((2,), 2.0, dtype=jnp.float64),
            **common,
        )
        self.assertAlmostEqual(float(scaled_loss), float(default_loss) / 4.0)

    def test_exact_q_translation_uses_one_phase_for_full_anchor(self) -> None:
        rng = np.random.default_rng(7)
        stencils = rng.normal(size=(2, 3, 4, 5)) + 1j * rng.normal(size=(2, 3, 4, 5))
        q_windows = rng.normal(size=(2, 6, 5)) + 1j * rng.normal(size=(2, 6, 5))
        k_arr = np.arange(5, dtype=np.float64) * 0.5
        shifts = np.array([0.3, 1.1], dtype=np.float64)
        translated_stencils, translated_q = train_mod.translate_exact_q_rollout_anchor_batch(
            stencils,
            q_windows,
            k_arr=k_arr,
            shifts=shifts,
        )
        phases = np.exp(-1j * shifts[:, None] * k_arr[None, :])
        np.testing.assert_allclose(translated_stencils, stencils * phases[:, None, None, :])
        np.testing.assert_allclose(translated_q, q_windows * phases[:, None, :])

    def test_exact_q_translation_matches_periodic_grid_shift(self) -> None:
        nx = 16
        length = 4.0 * math.pi
        x = np.arange(nx, dtype=np.float64) * length / float(nx)
        values = np.cos(0.5 * x) + 0.3 * np.sin(1.5 * x)
        shift_cells = 3
        shift = shift_cells * length / float(nx)
        coeff = np.fft.rfft(values)
        stencils = coeff[None, None, None, :]
        q_windows = coeff[None, None, :]
        k_arr = 2.0 * math.pi * np.fft.rfftfreq(nx, d=length / float(nx))
        translated, _ = train_mod.translate_exact_q_rollout_anchor_batch(
            stencils,
            q_windows,
            k_arr=k_arr,
            shifts=np.array([shift]),
        )
        expected = np.fft.rfft(np.roll(values, shift_cells))
        np.testing.assert_allclose(translated[0, 0, 0], expected, rtol=1e-12, atol=1e-12)

    def test_seeded_exact_q_translation_sampling_is_reproducible(self) -> None:
        histories = np.arange(1 * 6 * 3 * 3, dtype=np.float64).reshape(1, 6, 3, 3)
        histories = histories.astype(np.complex128) * (1.0 + 0.25j)
        sampling_state = {
            train_mod.REGIME_LINEAR: {
                "histories": histories,
                "train_case_indices": np.array([0], dtype=np.int32),
                "train_time_indices": np.array([2], dtype=np.int32),
                "train_k_indices": np.array([1], dtype=np.int32),
                "target_pools": {2: np.array([0], dtype=np.int32)},
                "train_anchor_case_indices": np.array([0], dtype=np.int32),
                "train_anchor_time_indices": np.array([2], dtype=np.int32),
                "anchor_target_pools": {2: np.array([0], dtype=np.int32)},
            }
        }

        def sample(seed: int):
            return train_mod.sample_exact_q_rollout_regime_batch(
                sampling_state,
                regime=train_mod.REGIME_LINEAR,
                target_nv=2,
                rollout_horizon=2,
                batch_size=1,
                k_arr=np.array([0.0, 0.5, 1.0]),
                rng=np.random.default_rng(seed),
                all_k_loss=True,
                selected_indices=np.array([0], dtype=np.int32),
                translation_augmentation=True,
                domain_length=4.0 * math.pi,
            )

        first = sample(11)
        second = sample(11)
        np.testing.assert_array_equal(first["anchor_stencils"], second["anchor_stencils"])
        np.testing.assert_array_equal(first["ref_q_windows"], second["ref_q_windows"])

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
            dataset_cache=None,
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
            online_loss_backend=train_mod.ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1,
            Nv_targets=(target_nv,),
            rollout_horizon=0,
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

    def test_online_fourier_hermite_bidir_loss_is_jax_differentiable_on_tiny_episode(self) -> None:
        target_nv = 4
        teacher_Nx = 8
        teacher_Nv = 16
        teacher_L = 4.0 * math.pi
        teacher_dt = 0.05
        teacher_vmin = -6.0
        teacher_vmax = 6.0

        online_dataset, _ = train_mod.build_online_reference_dataset(
            dataset_cache=None,
            regimes=(train_mod.REGIME_LINEAR,),
            teacher_Nx=teacher_Nx,
            teacher_Nv=teacher_Nv,
            teacher_L=teacher_L,
            teacher_vmin=teacher_vmin,
            teacher_vmax=teacher_vmax,
            teacher_dt=teacher_dt,
            linear_T=0.20,
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
            online_v_probes=0,
            online_loss_backend=train_mod.ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR,
            Nv_targets=(target_nv,),
            rollout_horizon=1,
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
        loss_fn, active_regimes = train_mod.make_online_fourier_hermite_bidir_batch_loss(
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
            rollout_horizon=1,
            rollout_anchor_samples=1,
            loss_backend=train_mod.ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR,
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
        self.assertTrue(np.isfinite(float(aux["q"])))
        self.assertEqual(float(aux["q"]), 0.0)
        self.assertTrue(np.isfinite(float(aux["state"])))
        self.assertGreaterEqual(float(aux["state"]), 0.0)
        for leaf in jax.tree_util.tree_leaves(grads):
            self.assertTrue(np.isfinite(np.asarray(leaf, dtype=np.float64)).all())

    def test_online_fourier_hermite_projected_xv_bidir_loss_is_jax_differentiable_on_tiny_episode(self) -> None:
        target_nv = 4
        teacher_Nx = 8
        teacher_Nv = 16
        teacher_L = 4.0 * math.pi
        teacher_dt = 0.05
        teacher_vmin = -6.0
        teacher_vmax = 6.0

        online_dataset, _ = train_mod.build_online_reference_dataset(
            dataset_cache=None,
            regimes=(train_mod.REGIME_LINEAR,),
            teacher_Nx=teacher_Nx,
            teacher_Nv=teacher_Nv,
            teacher_L=teacher_L,
            teacher_vmin=teacher_vmin,
            teacher_vmax=teacher_vmax,
            teacher_dt=teacher_dt,
            linear_T=0.20,
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
            online_v_probes=0,
            online_loss_backend=train_mod.ONLINE_LOSS_BACKEND_FOURIER_HERMITE_PROJECTED_XV_BIDIR,
            Nv_targets=(target_nv,),
            rollout_horizon=1,
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
        loss_fn, active_regimes = train_mod.make_online_fourier_hermite_bidir_batch_loss(
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
            rollout_horizon=1,
            rollout_anchor_samples=1,
            loss_backend=train_mod.ONLINE_LOSS_BACKEND_FOURIER_HERMITE_PROJECTED_XV_BIDIR,
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
        self.assertTrue(np.isfinite(float(aux["state"])))
        self.assertGreaterEqual(float(aux["state"]), 0.0)
        self.assertEqual(float(aux["q"]), 0.0)
        self.assertTrue(np.isfinite(float(aux["q_diag"])))
        self.assertGreaterEqual(float(aux["q_diag"]), 0.0)
        for leaf in jax.tree_util.tree_leaves(grads):
            self.assertTrue(np.isfinite(np.asarray(leaf, dtype=np.float64)).all())

    def test_online_projected_xv_loss_terms_are_relative_to_tail_scale(self) -> None:
        Nx = 8
        Nv = 4
        Nk = Nx // 2 + 1
        v_grid = jnp.linspace(-6.0, 6.0, 48, dtype=jnp.float64)
        ref = jnp.zeros((1, Nv, Nk), dtype=jnp.complex128).at[0, Nv - 1, 1].set(2.0 + 1.0j)
        pred_zero = jnp.zeros_like(ref)

        loss = train_mod.online_projected_xv_loss_terms(
            pred_zero,
            ref,
            Nx=Nx,
            Lx=4.0 * math.pi,
            v_grid=v_grid,
            tail_window=1,
        )
        scaled_loss = train_mod.online_projected_xv_loss_terms(
            pred_zero,
            7.0 * ref,
            Nx=Nx,
            Lx=4.0 * math.pi,
            v_grid=v_grid,
            tail_window=1,
        )
        zero_loss = train_mod.online_projected_xv_loss_terms(
            ref,
            ref,
            Nx=Nx,
            Lx=4.0 * math.pi,
            v_grid=v_grid,
            tail_window=1,
        )

        self.assertAlmostEqual(float(loss), 1.0, places=10)
        self.assertAlmostEqual(float(scaled_loss), 1.0, places=10)
        self.assertAlmostEqual(float(zero_loss), 0.0, places=12)

    def test_online_fourier_hermite_closure_bidir_loss_is_jax_differentiable_on_tiny_episode(self) -> None:
        target_nv = 4
        teacher_Nx = 8
        teacher_Nv = 16
        teacher_L = 4.0 * math.pi
        teacher_dt = 0.05
        teacher_vmin = -6.0
        teacher_vmax = 6.0

        online_dataset, _ = train_mod.build_online_reference_dataset(
            dataset_cache=None,
            regimes=(train_mod.REGIME_LINEAR,),
            teacher_Nx=teacher_Nx,
            teacher_Nv=teacher_Nv,
            teacher_L=teacher_L,
            teacher_vmin=teacher_vmin,
            teacher_vmax=teacher_vmax,
            teacher_dt=teacher_dt,
            linear_T=0.20,
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
            online_v_probes=0,
            online_loss_backend=train_mod.ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_BIDIR,
            Nv_targets=(target_nv,),
            rollout_horizon=1,
        )
        train_payload = online_dataset[train_mod.REGIME_LINEAR]["train"]
        self.assertIn(train_mod.online_reference_coeff_key(target_nv), train_payload)
        self.assertNotIn(train_mod.online_reference_coeff_key(target_nv + 1), train_payload)
        self.assertIn(train_mod.online_reference_q_key(target_nv), train_payload)

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
        loss_fn, active_regimes = train_mod.make_online_fourier_hermite_bidir_batch_loss(
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
            rollout_horizon=1,
            rollout_anchor_samples=1,
            loss_backend=train_mod.ONLINE_LOSS_BACKEND_FOURIER_HERMITE_CLOSURE_BIDIR,
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
        self.assertTrue(np.isfinite(float(aux["q"])))
        self.assertGreaterEqual(float(aux["q"]), 0.0)
        self.assertTrue(np.isfinite(float(aux["state"])))
        self.assertEqual(float(aux["state"]), 0.0)
        for leaf in jax.tree_util.tree_leaves(grads):
            self.assertTrue(np.isfinite(np.asarray(leaf, dtype=np.float64)).all())

    def test_online_hybrid_loss_is_jax_differentiable_on_tiny_episode(self) -> None:
        target_nv = 4
        teacher_Nx = 8
        teacher_Nv = 16
        teacher_L = 4.0 * math.pi
        teacher_dt = 0.05
        teacher_vmin = -6.0
        teacher_vmax = 6.0
        online_v_probes = 8

        online_dataset, _ = train_mod.build_online_reference_dataset(
            dataset_cache=None,
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
            online_loss_backend=train_mod.ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1,
            Nv_targets=(target_nv,),
            rollout_horizon=0,
        )
        dataset_base = train_mod.build_mixed_landau_dataset(
            dataset_cache=None,
            regimes=(train_mod.REGIME_LINEAR,),
            teacher_backend="grid_cubic_spline",
            teacher_Nx=teacher_Nx,
            teacher_Nv=teacher_Nv,
            teacher_L=teacher_L,
            teacher_vmin=teacher_vmin,
            teacher_vmax=teacher_vmax,
            teacher_dt=teacher_dt,
            teacher_proj_Nv=target_nv + 1,
            linear_T=0.10,
            linear_eps=1e-2,
            linear_modes=(0.5,),
            linear_num_samples=1,
            linear_seed=0,
            linear_poisson_sign=1.0,
            linear_history_stride=1,
            nonlinear_T=0.10,
            nonlinear_k0=0.5,
            nonlinear_poisson_sign=1.0,
            nonlinear_history_stride=1,
            weak_eps=(0.05,),
            strong_eps=(0.25,),
            Nv_targets=(target_nv,),
            Nm=1,
            val_fraction=0.2,
            n_low=2,
            context_mode="none",
            allow_cached_nv_superset=False,
            per_target_projection_orders=False,
        )
        k_scale = train_mod.choose_k_scale(dataset_base, Nm=1)
        nv_scale = train_mod.choose_nv_scale(dataset_base, Nm=1)
        prepared, stats = train_mod.prepare_training_dataset(
            dataset_base,
            Nm=1,
            k_scale=k_scale,
            nv_scale=nv_scale,
            context_mode="none",
        )
        params = train_mod.init_online_rollout_params(
            jax.random.PRNGKey(0),
            input_dim=int(stats["input_mean"].shape[0]),
            hidden_width=8,
            res_blocks=1,
            target_mean=stats["target_mean"],
            target_std=stats["target_std"],
        )
        loss_fn, active_regimes = train_mod.make_online_hybrid_batch_loss(
            prepared=prepared,
            online_dataset=online_dataset,
            regime_weights={train_mod.REGIME_LINEAR: 1.0},
            Nm=1,
            k_scale=float(k_scale),
            nv_scale=float(nv_scale),
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
            teacher_proj_Nv=target_nv + 1,
            n_low=2,
            context_mode="none",
            tail_start_fraction=2.0 / 3.0,
            loss_backend=train_mod.ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1,
            lambda_q=1.0,
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
        q_batches = {
            regime: {
                "inputs": prepared[regime]["train_inputs"],
                "targets_std": prepared[regime]["train_targets_std"],
            }
            for regime in active_regimes
        }
        regime_batches = {
            regime: online_dataset[regime]["train"]
            for regime in active_regimes
        }
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, q_batches, regime_batches)
        self.assertTrue(np.isfinite(float(loss)))
        for key, value in aux.items():
            self.assertTrue(np.isfinite(np.asarray(value, dtype=np.float64)).all(), msg=key)
        self.assertGreater(float(aux["q"]), 0.0)
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

    def test_online_rollout_training_raises_clear_error_on_nonfinite_step(self) -> None:
        params = {"w": jnp.zeros((1,), dtype=jnp.float64)}
        online_dataset = {
            train_mod.REGIME_LINEAR: {
                "train": {
                    "E_hat_ref": jnp.zeros((1, 1, 1), dtype=jnp.complex128),
                }
            }
        }

        def bad_batch_loss_fn(current_params, regime_batches):
            del regime_batches
            nan_value = current_params["w"][0] * jnp.asarray(0.0, dtype=jnp.float64) + jnp.asarray(
                jnp.nan,
                dtype=jnp.float64,
            )
            return nan_value, {
                "field": nan_value,
                "dist": nan_value,
                "tail": nan_value,
                "neg": nan_value,
                "reg": jnp.asarray(0.0, dtype=jnp.float64),
            }

        with self.assertRaisesRegex(FloatingPointError, r"online rollout produced non-finite loss/gradients"):
            train_mod.train_with_online_trajectory_minibatch_loss(
                params,
                online_dataset,
                bad_batch_loss_fn,
                active_regimes=(train_mod.REGIME_LINEAR,),
                epochs=1,
                learning_rate=1e-4,
                grad_clip=1.0,
                log_every=1,
                online_case_batch_size=1,
                steps_per_epoch=1,
                seed=0,
            )

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

    def test_removed_stability_aware_objective_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            with self.assertRaises(SystemExit):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
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

    def test_online_rollout_supports_reference_cache_but_rejects_offline_cache_flags_and_projection_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            cache = Path(tmpdir) / "shared_dataset.npz"
            online_cache = Path(tmpdir) / "online_reference_dataset.npz"
            with self.assertRaisesRegex(ValueError, r"--build-dataset-only requires --dataset-cache"):
                train_main(
                    [
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
            with self.assertRaisesRegex(ValueError, r"requires --online-reference-cache"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory_q_hybrid",
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                    ]
                )
            with self.assertRaisesRegex(ValueError, r"requires --lambda-q > 0"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory_q_hybrid",
                        "--online-reference-cache",
                        str(online_cache),
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                        "--lambda-q",
                        "0.0",
                    ]
                )
            with mock.patch.object(
                train_mod,
                "build_online_reference_dataset",
                return_value=(
                    {
                        train_mod.REGIME_LINEAR: {
                            "train": {"E_hat_ref": np.zeros((1, 2, 2), dtype=np.complex128)},
                            "val": {"E_hat_ref": np.zeros((1, 2, 2), dtype=np.complex128)},
                        }
                    },
                    jnp.linspace(-6.0, 6.0, 4, dtype=jnp.float64),
                ),
            ) as patched:
                train_main(
                    [
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory",
                        "--Nv-targets",
                        "4,8",
                        "--Nm",
                        "1",
                        "--dataset-cache",
                        str(cache),
                        "--build-dataset-only",
                    ]
                )
            patched.assert_called_once()
            with mock.patch.object(
                train_mod,
                "build_online_reference_dataset",
                return_value=(
                    {
                        train_mod.REGIME_LINEAR: {
                            "train": {"E_hat_ref": np.zeros((1, 2, 2), dtype=np.complex128)},
                            "val": {"E_hat_ref": np.zeros((1, 2, 2), dtype=np.complex128)},
                        }
                    },
                    jnp.linspace(-6.0, 6.0, 4, dtype=jnp.float64),
                ),
            ) as patched_online, mock.patch.object(
                train_mod,
                "build_mixed_landau_dataset",
                return_value={
                    train_mod.REGIME_LINEAR: {
                        "train_inputs_base": np.zeros((2, 2 * 1 + 4), dtype=np.float64),
                        "train_targets": np.zeros((2, 2), dtype=np.float64),
                        "val_inputs_base": np.zeros((1, 2 * 1 + 4), dtype=np.float64),
                        "val_targets": np.zeros((1, 2), dtype=np.float64),
                    }
                },
            ) as patched_q:
                train_main(
                    [
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory_q_hybrid",
                        "--Nv-targets",
                        "4,8",
                        "--Nm",
                        "1",
                        "--dataset-cache",
                        str(cache),
                        "--online-reference-cache",
                        str(online_cache),
                        "--lambda-q",
                        "1.0",
                        "--build-dataset-only",
                    ]
                )
            patched_online.assert_called_once()
            patched_q.assert_called_once()
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

    def test_fourier_hermite_bidir_rejects_nonzero_observable_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            with self.assertRaisesRegex(ValueError, r"requires lambda_E=lambda_dist=lambda_tail=lambda_neg=lambda_reg=0"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory",
                        "--online-loss-backend",
                        train_mod.ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR,
                        "--online-v-probes",
                        "0",
                        "--rollout-horizon",
                        "1",
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                        "--lambda-E",
                        "1.0",
                    ]
                )

    def test_fourier_hermite_bidir_rejects_nonzero_probe_grid_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
            with self.assertRaisesRegex(ValueError, r"requires --online-v-probes 0"):
                train_main(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--training-mode",
                        "online_rollout",
                        "--train-objective",
                        "trajectory",
                        "--online-loss-backend",
                        train_mod.ONLINE_LOSS_BACKEND_FOURIER_HERMITE_BIDIR,
                        "--online-v-probes",
                        "8",
                        "--rollout-horizon",
                        "1",
                        "--Nv-targets",
                        "4",
                        "--Nm",
                        "1",
                        "--lambda-E",
                        "0.0",
                        "--lambda-dist",
                        "0.0",
                        "--lambda-tail",
                        "0.0",
                        "--lambda-neg",
                        "0.0",
                        "--lambda-reg",
                        "0.0",
                    ]
                )

    def test_cached_online_reference_dataset_is_reused(self) -> None:
        v_probe = np.linspace(-6.0, 6.0, 8, dtype=np.float64)
        shared_dataset = {
            train_mod.REGIME_LINEAR: {
                "train": {
                    "times": np.array([[0.0, 0.05, 0.10]], dtype=np.float64),
                    "E_hat_ref": np.zeros((1, 3, 5), dtype=np.complex128),
                    "delta_f_ref": np.zeros((1, 3, 8, 4), dtype=np.float64),
                    "perturbation_x": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64),
                },
                "val": {
                    "times": np.array([[0.0, 0.05, 0.10]], dtype=np.float64),
                    "E_hat_ref": np.ones((1, 3, 5), dtype=np.complex128),
                    "delta_f_ref": np.ones((1, 3, 8, 4), dtype=np.float64),
                    "perturbation_x": np.array([[0.5, 0.6, 0.7, 0.8]], dtype=np.float64),
                },
            }
        }
        metadata = train_mod.build_online_reference_cache_metadata(
            regimes=(train_mod.REGIME_LINEAR,),
            teacher_Nx=8,
            teacher_Nv=16,
            teacher_L=4.0 * math.pi,
            teacher_vmin=-6.0,
            teacher_vmax=6.0,
            teacher_dt=0.05,
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
            online_v_probes=8,
            online_loss_backend=train_mod.ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1,
            Nv_targets=None,
            rollout_horizon=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "online_reference_dataset.npz"
            train_mod.save_online_reference_cache(
                cache,
                shared_dataset,
                v_probe=v_probe,
                metadata=metadata,
            )
            with mock.patch.object(
                train_mod,
                "build_physical_reference_episode",
                side_effect=RuntimeError("should reuse cached online dataset"),
            ):
                cached_dataset, cached_v_probe = train_mod.build_online_reference_dataset(
                    dataset_cache=cache,
                    regimes=(train_mod.REGIME_LINEAR,),
                    teacher_Nx=8,
                    teacher_Nv=16,
                    teacher_L=4.0 * math.pi,
                    teacher_vmin=-6.0,
                    teacher_vmax=6.0,
                    teacher_dt=0.05,
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
                    online_v_probes=8,
                    online_loss_backend=train_mod.ONLINE_LOSS_BACKEND_FIELD_DISTRIBUTION_V1,
                    Nv_targets=(4,),
                    rollout_horizon=0,
                )
            np.testing.assert_allclose(np.asarray(cached_v_probe, dtype=np.float64), v_probe)
            np.testing.assert_allclose(
                np.asarray(cached_dataset[train_mod.REGIME_LINEAR]["train"]["times"], dtype=np.float64),
                shared_dataset[train_mod.REGIME_LINEAR]["train"]["times"],
            )
            np.testing.assert_allclose(
                np.asarray(cached_dataset[train_mod.REGIME_LINEAR]["val"]["perturbation_x"], dtype=np.float64),
                shared_dataset[train_mod.REGIME_LINEAR]["val"]["perturbation_x"],
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

    def test_online_rollout_accepts_multiple_target_nv_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "shared_interface.npz"
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
                    "--hidden-width",
                    "8",
                    "--res-blocks",
                    "1",
                    "--epochs",
                    "1",
                    "--lr",
                    "1e-3",
                    "--grad-clip",
                    "1.0",
                    "--log-every",
                    "1",
                    "--steps-per-epoch",
                    "1",
                    "--online-case-batch-size",
                    "1",
                    "--regimes",
                    "linear_landau",
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
                    "--linear-eps",
                    "0.01",
                    "--linear-modes",
                    "0.5",
                    "--linear-num-samples",
                    "1",
                    "--linear-seed",
                    "0",
                    "--online-v-probes",
                    "8",
                ]
            )
            learned = load_learned_interface_closure_npz(ckpt)
            self.assertEqual(learned.training_mode, "online_rollout")
            self.assertEqual(learned.train_objective, "trajectory")
            self.assertEqual(tuple(int(v) for v in learned.Nv_targets), (4, 6))

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
