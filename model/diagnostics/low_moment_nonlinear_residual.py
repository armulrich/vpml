"""Compatibility entry point for :mod:`vpml.diagnostics.low_moment_nonlinear_residual`."""

from vpml.diagnostics.low_moment_nonlinear_residual import (
    _case_map,
    _state_and_gradient,
    _align_spectra,
    _spectral_features,
    _history_dataset,
    _knn_score,
    _teacher_forced_checkpoint_score,
    _moment_equation_residual_score,
    main,
)


if __name__ == "__main__":
    main()
