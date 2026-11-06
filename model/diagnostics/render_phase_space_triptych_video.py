"""Compatibility entry point for :mod:`vpml.diagnostics.render_phase_space_triptych_video`."""

from vpml.diagnostics.render_phase_space_triptych_video import (
    L_X,
    _parse_v_range,
    _frame_times,
    build_arg_parser,
    _resolve_paths,
    _configure_axes,
    main,
)


if __name__ == "__main__":
    main()
