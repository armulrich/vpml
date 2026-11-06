"""Compatibility entry point for :mod:`vpml.diagnostics.projection_quadrature_convergence`."""

from vpml.diagnostics.projection_quadrature_convergence import (
    _parse_int_tuple,
    _parse_float_tuple,
    _relative_l2,
    _snapshot_digest,
    _project_snapshots,
    _case_perturbations,
    _load_teacher_snapshot_artifact,
    _build_arg_parser,
    _json_float,
    _max_finite,
    _save_convergence_plot,
    main,
)


if __name__ == "__main__":
    main()
