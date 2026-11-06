"""Compatibility entry point for :mod:`vpml.diagnostics.physical_velocity_grid_convergence`."""

from vpml.diagnostics.physical_velocity_grid_convergence import (
    _parse_int_tuple,
    _parse_float_tuple,
    _relative_l2,
    _json_float,
    _case_perturbations,
    _resample_velocity_snapshots,
    _distribution_successive_change,
    _energy_block_changes,
    _display_case,
    _save_energy_plot,
    _save_convergence_plot,
    _save_teacher_artifact,
    _build_arg_parser,
    main,
)


if __name__ == "__main__":
    main()
