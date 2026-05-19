"""Rollout backends that are separate from the Fourier-Hermite closure path."""

from .spline_fem import (
    init_spline_residual_params,
    restrict_history_to_grid,
    restrict_state_to_grid,
    spline_fem_base_step_dt,
    spline_fem_lr_teacher_defect_loss,
    spline_fem_lr_teacher_rollout_loss,
    spline_fem_rollout_loss,
    spline_fem_teacher_lifted_rollout_loss,
    spline_fem_step_with_residual,
    spline_fem_step_with_residual_dt,
)

__all__ = [
    "init_spline_residual_params",
    "restrict_history_to_grid",
    "restrict_state_to_grid",
    "spline_fem_base_step_dt",
    "spline_fem_lr_teacher_defect_loss",
    "spline_fem_lr_teacher_rollout_loss",
    "spline_fem_rollout_loss",
    "spline_fem_teacher_lifted_rollout_loss",
    "spline_fem_step_with_residual",
    "spline_fem_step_with_residual_dt",
]
