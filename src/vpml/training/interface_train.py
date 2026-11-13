"""Canonical solver-embedded interface-flux rollout trainer."""

from __future__ import annotations

from dataclasses import replace
from typing import (
    Dict,
    Optional,
    Sequence,
)

from vpml.jax_runtime import bootstrap_jax_runtime, print_jax_runtime_summary

bootstrap_jax_runtime()

import jax
import numpy as np

from vpml.datasets.interface_flux import (
    IC_SPLIT_HELDOUT,
    IC_SPLIT_TRAIN,
)

from vpml.solvers.implicit import Array
from vpml.solvers.hermite import FourierHermiteIMEX
from vpml.models.interface_closure import (
    GRID_CUBIC_SPLINE_TEACHER_BACKEND,
    init_interface_closure_params,
)
from vpml.models.interface_checkpoint import save_learned_interface_closure_npz
from vpml.visualization.training import save_training_loss_plot

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


from vpml.training.interface_configuration import (
    build_arg_parser,
    validate_training_options,
    CANONICAL_NV_TARGETS,
    CANONICAL_REGIMES,
    parse_float_tuple,
    interface_flux_rollout_coeff_key,
)
from vpml.training.interface_reference import (
    build_interface_flux_rollout_reference_dataset,
    _reference_case_histories,
    _reference_case_splits,
)
from vpml.training.interface_data import (
    build_interface_flux_rollout_qpair_dataset,
    interface_flux_rollout_regime_loss_stds,
    phase_isotropic_complex_training_stats,
)
from vpml.training.interface_losses import make_interface_flux_rollout_batch_loss
from vpml.training.interface_state import (
    default_exact_k_scale,
    build_learned_interface_closure,
    _load_init_checkpoint_for_interface_closure,
    _derived_statistics_path,
    _load_derived_statistics,
    _save_derived_statistics,
)
from vpml.training.interface_optimization import train_with_interface_flux_rollout_minibatch_loss
from vpml.training.interface_evaluation import (
    evaluate_heldout_interface_flux_metrics,
    _canonical_metrics_payload,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    print_jax_runtime_summary(jax, context="interface-flux training")
    args = build_arg_parser().parse_args(argv)
    projection_quadrature_Nv = validate_training_options(args)
    ic_modes = parse_float_tuple(args.ic_modes)
    cache_snapshot_times = parse_float_tuple(args.cache_snapshot_times)
    reference, max_projection_order, ic_manifest, reference_cache_dir = (
        build_interface_flux_rollout_reference_dataset(
        reference_cache_root=args.reference_cache_dir,
        ic_manifest_path=args.ic_manifest,
        regimes=CANONICAL_REGIMES,
        teacher_Nx=args.teacher_Nx,
        teacher_Nv=args.teacher_Nv,
        projection_quadrature_Nv=projection_quadrature_Nv,
        teacher_L=args.teacher_L,
        teacher_vmin=args.teacher_vmin,
        teacher_vmax=args.teacher_vmax,
        teacher_dt=args.teacher_dt,
        T_final=args.T_final,
        teacher_poisson_sign=args.teacher_poisson_sign,
        snapshot_times=cache_snapshot_times,
        cases_per_regime=args.ic_cases_per_regime,
        heldout_cases_per_regime=args.ic_heldout_per_regime,
        ic_generation_seed=args.ic_generation_seed,
        ic_split_seed=args.ic_split_seed,
        ic_modes=ic_modes,
        Nv_targets=CANONICAL_NV_TARGETS,
        min_projection_order=None,
        )
    )
    coeff_key = interface_flux_rollout_coeff_key(max_projection_order)
    print(
        "[data] physical teacher, rollout, and projection grids: "
        f"teacher_Nx={int(args.teacher_Nx)} "
        f"rollout_Nx={int(args.rollout_Nx)} "
        f"teacher_Nv={int(args.teacher_Nv)} "
        f"projection_quadrature_Nv={projection_quadrature_Nv}"
    )
    for regime in CANONICAL_REGIMES:
        cases = _reference_case_histories(reference[regime], coeff_key)
        case_splits = _reference_case_splits(reference[regime], len(cases))
        first_case = cases[0]
        print(
            f"[data] interface-flux rollout {regime}: "
            f"cases={len(cases)} train={int(np.sum(case_splits == IC_SPLIT_TRAIN))} "
            f"heldout={int(np.sum(case_splits == IC_SPLIT_HELDOUT))} "
            f"history={first_case.shape[0]} order={first_case.shape[1]}"
        )
    if args.build_dataset_only:
        print(f"Saved interface-flux reference cache to {reference_cache_dir}")
        return

    k_arr = np.asarray(
        FourierHermiteIMEX(
            Nx=int(args.rollout_Nx),
            Nv=max(CANONICAL_NV_TARGETS),
            Lx=float(args.teacher_L),
            dt=float(args.teacher_dt),
            vth=1.0,
            dealias_23=False,
            closure=None,
        ).k_arr,
        dtype=np.float64,
    )
    init_params: Optional[Dict[str, Array]] = None
    init_stats: Optional[Dict[str, np.ndarray]] = None
    init_k_scale: Optional[float] = None
    init_nv_scale: Optional[float] = None
    if args.init_checkpoint is not None:
        init_params, init_stats, init_k_scale, init_nv_scale = (
            _load_init_checkpoint_for_interface_closure(
                args.init_checkpoint,
                Nm=args.Nm,
                hidden_width=args.hidden_width,
                res_blocks=args.res_blocks,
                Nv_targets=CANONICAL_NV_TARGETS,
                context_mode="none",
                equilibrium_centered=True,
                complex_normalization_mode="phase_isotropic",
                ic_manifest_sha256=str(ic_manifest["sha256"]),
                rollout_Nx=int(args.rollout_Nx),
            )
        )
        print(
            "[train] initialized interface-flux parameters from "
            f"{args.init_checkpoint}"
        )
    k_scale = (
        float(args.k_scale)
        if args.k_scale is not None
        else float(init_k_scale)
        if init_k_scale is not None
        else default_exact_k_scale(k_arr)
    )
    nv_scale = (
        float(args.nv_scale)
        if args.nv_scale is not None
        else float(init_nv_scale)
        if init_nv_scale is not None
        else float(max(CANONICAL_NV_TARGETS))
    )
    derived_statistics_path = _derived_statistics_path(
        reference_cache_dir,
        manifest_sha256=str(ic_manifest["sha256"]),
        rollout_horizon=args.rollout_horizon,
        history_stride=args.history_stride,
        Nm=args.Nm,
        n_low=args.n_low,
        k_scale=k_scale,
        nv_scale=nv_scale,
        rollout_Nx=int(args.rollout_Nx),
    )
    cached_derived_statistics = _load_derived_statistics(derived_statistics_path)
    cached_stats: Optional[Dict[str, np.ndarray]] = None
    cached_regime_loss_stds: Optional[Dict[str, float]] = None
    if cached_derived_statistics is not None:
        cached_stats, cached_regime_loss_stds = cached_derived_statistics
        print(
            "[data] reusing training-only normalization and regime scales from "
            f"{derived_statistics_path}"
        )
    dataset_base, precomputed_stats = build_interface_flux_rollout_qpair_dataset(
        reference,
        max_projection_order=max_projection_order,
        Nv_targets=CANONICAL_NV_TARGETS,
        Nm=args.Nm,
        k_arr=k_arr,
        linear_history_stride=args.history_stride,
        nonlinear_history_stride=args.history_stride,
        rollout_horizon=args.rollout_horizon,
        n_low=args.n_low,
        context_mode="none",
        store_training_pairs=False,
        k_scale=k_scale,
        nv_scale=nv_scale,
        precomputed_training_stats=(
            init_stats if init_stats is not None else cached_stats
        ),
        source_Nx=int(args.teacher_Nx),
        rollout_Nx=int(args.rollout_Nx),
    )
    if init_stats is not None:
        stats = init_stats
    elif cached_stats is not None:
        stats = cached_stats
    else:
        if precomputed_stats is None:
            raise RuntimeError("interface-flux training statistics are unavailable")
        stats = phase_isotropic_complex_training_stats(
            precomputed_stats,
            Nm=args.Nm,
            context_mode="none",
        )
    for regime in CANONICAL_REGIMES:
        train_anchors = int(
            dataset_base[regime]["train_anchor_time_indices"].shape[0]
        )
        heldout_anchors = int(
            dataset_base[regime]["val_anchor_time_indices"].shape[0]
        )
        print(
            f"[data] {regime}: train_anchors={train_anchors} "
            f"heldout_anchors={heldout_anchors}"
        )

    regime_loss_stds = cached_regime_loss_stds
    if regime_loss_stds is None:
        regime_loss_stds = interface_flux_rollout_regime_loss_stds(
            reference,
            dataset_base,
            max_projection_order=max_projection_order,
            target_nvs=CANONICAL_NV_TARGETS,
            k_arr=k_arr,
            rollout_horizon=args.rollout_horizon,
            source_Nx=int(args.teacher_Nx),
            rollout_Nx=int(args.rollout_Nx),
        )
    if cached_derived_statistics is None:
        _save_derived_statistics(
            derived_statistics_path,
            stats=stats,
            regime_loss_stds=regime_loss_stds,
        )
        print(
            "[data] saved training-only normalization and regime scales to "
            f"{derived_statistics_path}"
        )
    print(
        "[data] fixed regime-balanced interface-flux scales: "
        + " ".join(
            f"{regime}={regime_loss_stds[regime]:.6e}"
            for regime in CANONICAL_REGIMES
        )
    )
    print(
        "[data] canonical constraints: all_positive_k=1 "
        "equilibrium_centered=1 complex_normalization=phase_isotropic "
        "translation_augmentation=1 "
        f"cutoff_cycle={','.join(str(value) for value in CANONICAL_NV_TARGETS)}"
    )

    params = (
        init_params
        if init_params is not None
        else init_interface_closure_params(
            jax.random.PRNGKey(args.seed),
            input_dim=int(stats["input_mean"].shape[0]),
            hidden_width=int(args.hidden_width),
            res_blocks=int(args.res_blocks),
        )
    )
    loss_fn, active_regimes = make_interface_flux_rollout_batch_loss(
        regime_weights={regime: 1.0 for regime in CANONICAL_REGIMES},
        Nm=args.Nm,
        k_scale=k_scale,
        nv_scale=nv_scale,
        stats=stats,
        hidden_width=args.hidden_width,
        res_blocks=args.res_blocks,
        Nv_targets=CANONICAL_NV_TARGETS,
        train_regimes=CANONICAL_REGIMES,
        teacher_backend=GRID_CUBIC_SPLINE_TEACHER_BACKEND,
        teacher_Lx=args.teacher_L,
        teacher_Nx=args.teacher_Nx,
        teacher_Nv=args.teacher_Nv,
        teacher_vmin=args.teacher_vmin,
        teacher_vmax=args.teacher_vmax,
        teacher_dt=args.teacher_dt,
        teacher_proj_Nv=max_projection_order,
        projection_quadrature_Nv=projection_quadrature_Nv,
        n_low=args.n_low,
        context_mode="none",
        rollout_horizon=args.rollout_horizon,
        poisson_sign=args.teacher_poisson_sign,
        rollout_dealias_23=False,
        rollout_precision=args.precision,
        regime_q_loss_stds=regime_loss_stds,
        equilibrium_centered=True,
        complex_normalization_mode="phase_isotropic",
        translation_augmented=True,
        rollout_Nx=int(args.rollout_Nx),
    )
    params, component_history = train_with_interface_flux_rollout_minibatch_loss(
        params,
        reference,
        dataset_base,
        loss_fn,
        max_projection_order=max_projection_order,
        active_regimes=active_regimes,
        k_arr=k_arr,
        epochs=args.epochs,
        learning_rate=args.lr,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        rollout_horizon=args.rollout_horizon,
        seed=args.seed,
        source_Nx=int(args.teacher_Nx),
        rollout_Nx=int(args.rollout_Nx),
        log_components=("q",),
        profile_trace_dir=args.profile_trace_dir,
        profile_train_steps=args.profile_train_steps,
        profile_skip_steps=args.profile_skip_steps,
    )
    learned = build_learned_interface_closure(
        params=params,
        Nm=args.Nm,
        k_scale=k_scale,
        nv_scale=nv_scale,
        stats=stats,
        hidden_width=args.hidden_width,
        res_blocks=args.res_blocks,
        equilibrium_centered=True,
        complex_normalization_mode="phase_isotropic",
        translation_augmented=True,
        Nv_targets=CANONICAL_NV_TARGETS,
        train_regimes=CANONICAL_REGIMES,
        teacher_backend=GRID_CUBIC_SPLINE_TEACHER_BACKEND,
        teacher_Lx=args.teacher_L,
        teacher_Nx=args.teacher_Nx,
        teacher_Nv=args.teacher_Nv,
        teacher_vmin=args.teacher_vmin,
        teacher_vmax=args.teacher_vmax,
        teacher_dt=args.teacher_dt,
        teacher_proj_Nv=max_projection_order,
        projection_quadrature_Nv=projection_quadrature_Nv,
        n_low=args.n_low,
        rollout_horizon=args.rollout_horizon,
        rollout_Nx=int(args.rollout_Nx),
    )
    learned = replace(
        learned,
        ic_manifest_sha256=str(ic_manifest["sha256"]),
        training_ic_count=sum(
            str(case["split"]) == IC_SPLIT_TRAIN
            for case in ic_manifest["cases"]
        ),
        heldout_ic_count=sum(
            str(case["split"]) == IC_SPLIT_HELDOUT
            for case in ic_manifest["cases"]
        ),
    )
    val_metrics = evaluate_heldout_interface_flux_metrics(
        learned,
        reference,
        dataset_base,
        max_projection_order=max_projection_order,
        target_nvs=CANONICAL_NV_TARGETS,
        Nm=args.Nm,
        k_arr=k_arr,
        k_scale=k_scale,
        nv_scale=nv_scale,
        n_low=args.n_low,
        source_Nx=int(args.teacher_Nx),
        rollout_Nx=int(args.rollout_Nx),
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_learned_interface_closure_npz(args.checkpoint, learned)
    metrics_path = args.checkpoint.with_suffix(".metrics.npz")
    loss_history = np.asarray(component_history["total"], dtype=np.float64)
    metrics_payload = _canonical_metrics_payload(
        args=args,
        loss_history=loss_history,
        component_history=component_history,
        stats=stats,
        k_scale=k_scale,
        nv_scale=nv_scale,
        teacher_projection_order=max_projection_order,
        projection_quadrature_Nv=projection_quadrature_Nv,
        regime_loss_stds=regime_loss_stds,
        val_metrics=val_metrics,
        ic_manifest=ic_manifest,
        reference_cache_dir=reference_cache_dir,
    )
    np.savez(metrics_path, **metrics_payload)
    loss_plot_path = (
        args.loss_plot
        if args.loss_plot is not None
        else args.checkpoint.with_suffix(".loss.png")
    )
    save_training_loss_plot(
        loss_history,
        loss_plot_path,
        loss_metadata=metrics_payload,
        val_metrics=val_metrics,
    )
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved loss plot to {loss_plot_path}")
    for key in sorted(val_metrics):
        print(f"{key}: {float(np.asarray(val_metrics[key]).reshape(-1)[0]):.6e}")


if __name__ == "__main__":
    main()
