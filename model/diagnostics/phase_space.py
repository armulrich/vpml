"""Compatibility entry point for :mod:`vpml.diagnostics.phase_space`."""

from vpml.diagnostics.phase_space import (
    _read_npy_header_from_npz,
    FourierHermiteHistoryReader,
    select_nearest_case,
    resample_periodic_rows,
    phase_space_from_hermite_phys,
)
