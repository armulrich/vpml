"""Compatibility entry point for :mod:`vpml.diagnostics.physical_spatial_grid_convergence`."""

from vpml.diagnostics.physical_spatial_grid_convergence import (
    _distribution_successive_x_change,
    _save_energy_plot,
    _save_convergence_plot,
    _build_arg_parser,
    main,
)


if __name__ == "__main__":
    main()
