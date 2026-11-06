"""Compatibility entry point for :mod:`vpml.diagnostics.plot_hermite_spectrum`."""

from vpml.diagnostics.plot_hermite_spectrum import (
    _COEFF_KEY_RE,
    _scalar_from_npz,
    _find_available_targets,
    _selected_coeff_keys,
    hermite_spectrum_from_cache,
    _plot_bar,
    write_outputs,
    _parse_csv,
    build_arg_parser,
    main,
)


if __name__ == "__main__":
    main()
