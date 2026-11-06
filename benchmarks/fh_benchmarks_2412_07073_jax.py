"""Compatibility entry point for :mod:`vpml.benchmarks.fh_benchmarks_2412_07073_jax`."""

from vpml.benchmarks.fh_benchmarks_2412_07073_jax import (
    build_linear_Q,
    least_damped_gamma_star,
    select_paper_optimal_parameter,
    advection_matrix_Abarbar,
    modify_A_for_method,
    response_function_aw_Nv,
    response_function_aw_Nv_and_deriv,
    discrete_root_from_response,
    BenchmarkResult,
    Benchmark,
    Fig2DampingProfiles,
    Fig3ResponseFunction,
    Fig4EigenvalueScan,
    simulate_nonlinear_landau_method,
    Fig10NonlinearLandauPhaseSpace,
    Fig10LearnedComparisonPhaseSpace,
    LinearLandauTimeBenchmark,
    plot_linear_landau_comparison,
    main,
)


if __name__ == "__main__":
    main()
