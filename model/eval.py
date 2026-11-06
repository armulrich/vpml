"""Compatibility entry point for :mod:`vpml.evaluation.interface_flux`."""

from vpml.evaluation.interface_flux import (
    BUNDLE_HELDOUT,
    BUNDLE_BENCHMARK,
    ALL_BUNDLES,
    parse_str_tuple,
    parse_float_tuple,
    _k_arr_for_grid,
    _electric_energy_from_ehat_history,
    _maxwellian_equilibrium,
    _physical_grid_ehat_projector,
    run_physical_landau_reference,
    run_learned_linear_case,
    run_learned_nonlinear_case,
    evaluate_case,
    run_heldout_landau_bundle,
    run_benchmark_rollout_bundle,
    write_bundle_summary,
    build_arg_parser,
    main,
)


if __name__ == "__main__":
    main()
