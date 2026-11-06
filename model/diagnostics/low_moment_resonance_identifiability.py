"""Compatibility entry point for :mod:`vpml.diagnostics.low_moment_resonance_identifiability`."""

from vpml.diagnostics.low_moment_resonance_identifiability import (
    _cumulative_trapezoid,
    mode_bounce_exposure,
    _resonance_rows,
    _phase_velocities,
    _integrate_velocity_bands,
    _sample_velocity_profiles,
    _direct_population_rows,
    _score_feature_sets,
    main,
)


if __name__ == "__main__":
    main()
