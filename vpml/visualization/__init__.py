"""Reusable plotting helpers for vpml."""

from .benchmarks import (
    save_fig2_damping_profiles,
    save_fig3_response_function,
    save_fig4_eigenvalue_scan,
    save_fig10_learned_comparison_phase_space,
    save_fig10_learned_comparison_nv_sweep_phase_space,
    save_fig10_nonlinear_landau_phase_space,
    save_linear_landau_comparison,
    save_linear_landau_time,
)
from .common import save_figure
from .metrics import (
    FieldSweepCase,
    GrowthSweepCase,
    plot_field_metric,
    plot_field_metric_sweep,
    plot_growth_metric,
    plot_growth_metric_sweep,
    plot_metric_summary,
)
from .nonlinear import (
    plot_bump_on_tail_energy_comparison,
    plot_snapshot_panel,
    save_bump_on_tail_energy_comparison,
    save_snapshot_panel,
)
from .training import (
    plot_training_loss,
    save_training_loss_plot,
)

__all__ = [
    "save_fig2_damping_profiles",
    "save_fig3_response_function",
    "save_fig4_eigenvalue_scan",
    "save_fig10_learned_comparison_phase_space",
    "save_fig10_learned_comparison_nv_sweep_phase_space",
    "save_fig10_nonlinear_landau_phase_space",
    "save_linear_landau_comparison",
    "save_linear_landau_time",
    "save_figure",
    "FieldSweepCase",
    "GrowthSweepCase",
    "plot_field_metric",
    "plot_field_metric_sweep",
    "plot_growth_metric",
    "plot_growth_metric_sweep",
    "plot_metric_summary",
    "plot_bump_on_tail_energy_comparison",
    "plot_snapshot_panel",
    "save_bump_on_tail_energy_comparison",
    "save_snapshot_panel",
    "plot_training_loss",
    "save_training_loss_plot",
]
