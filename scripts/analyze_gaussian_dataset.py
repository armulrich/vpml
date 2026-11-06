"""Compatibility entry point for :mod:`vpml.diagnostics.gaussian_dataset`."""

from vpml.diagnostics.gaussian_dataset import (
    effective_rank,
    spatial_features,
    history_features,
    common_standardization,
    load_cases,
    primary_cases,
    _relative_l2,
    common_spectral_band_errors,
    refinement_comparisons,
    information_comparison,
    refinement_diagnostic_figure,
    figures,
    analyze,
    main,
)


if __name__ == "__main__":
    main()
