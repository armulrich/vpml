"""Compatibility entry point for :mod:`vpml.diagnostics.physical_grid_2d_convergence`."""

from vpml.diagnostics.physical_grid_2d_convergence import (
    _atomic_savez,
    _atomic_save_npy,
    _pair_directory,
    _simulation_metadata,
    _parse_case_names,
    _run_case_segmented,
    _simulate_pair,
    _load_pair,
    _project_snapshots,
    _projected_change,
    _joint_distribution_change,
    _max_energy_change,
    _comparison_metrics,
    _config_from_metadata,
    _save_summary_plot,
    _analyze,
    _parse_int_tuple,
    _build_arg_parser,
    main,
)


if __name__ == "__main__":
    main()
