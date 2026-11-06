"""Vlasov–Poisson solvers and learned closures.

Public names are loaded on first use. Manifest, storage and configuration tools
can therefore be imported without loading JAX or selecting a compute backend.
"""
from importlib import import_module

_EXPORTS = {
    **dict.fromkeys((
        "bootstrap_jax_runtime",
        "format_jax_runtime_summary",
        "plan_jax_runtime",
        "print_jax_runtime_summary",
    ), "jax_runtime"),
    **dict.fromkeys((
        "EarlyElectricFieldGrowthMetric",
        "EarlyGrowthConfig",
        "FieldErrorConfig",
        "FourierFieldComparison",
        "FieldErrorResult",
        "GrowthComparisonResult",
        "GrowthFitResult",
        "SelfGeneratedFieldErrorMetric",
    ), "metrics"),
    **dict.fromkeys((
        "Array",
        "FourierHermiteIMEX",
        "HermiteExponentialFilter",
        "HouLiFilter",
        "HyperCollisions",
        "LearnedInterfaceClosure",
        "NonlocalClosure",
        "e_hat_from_rho_hat",
        "e_hat_history_from_a_hat_history",
        "hermite_damping_term",
        "init_interface_closure_params",
        "init_real_mlp",
        "interface_closure_apply",
        "implicit_midpoint_jfnk_step",
        "irfft_x",
        "learned_boundary_flux_hat",
        "learned_interface_q_hat",
        "load_learned_interface_closure_npz",
        "mlp_apply",
        "rfft_x",
        "save_learned_interface_closure_npz",
        "tridiag_solve",
    ), "core"),
    **dict.fromkeys((
        "PhysicalGridVlasovPoissonConfig",
        "advect_v_cubic",
        "advect_x_cubic",
        "build_cubic_spline_hermite_projection_matrix",
        "compute_electric_field_from_distribution",
        "cubic_bspline_interp_constant",
        "cubic_bspline_interp_periodic",
        "cubic_bspline_prefilter_constant",
        "cubic_bspline_prefilter_periodic",
        "cubic_bspline_velocity_resampling_matrix",
        "electric_energy_from_field",
        "equilibrium_coeffs_bump_on_tail",
        "extract_interface_supervised_pairs_from_coeff_history",
        "gaussian_pdf",
        "gaussian_vhat",
        "hermite_coeffs_gaussian",
        "hermite_dual_basis",
        "hermite_dual_basis_scaled",
        "normalize_density_on_grid",
        "project_distribution_history_to_fourier_hermite",
        "project_distribution_snapshot_to_fourier_hermite",
        "project_distribution_snapshot_with_hermite_matrix",
        "run_semilagrangian_vlasov_poisson",
        "trapezoid_quadrature_weights",
    ), "physical_grid"),
    **dict.fromkeys((
        "landau_gamma",
        "landau_gamma_mag",
        "plasma_dispersion_Z",
        "response_function_R",
        "solve_landau_root_xi",
    ), "linear_landau"),
    **dict.fromkeys((
        "NonlinearLandauParams",
        "run_nonlinear_landau_rollout_raw",
    ), "nonlinear_landau"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
