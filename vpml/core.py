"""Compatibility imports for the historical :mod:`vpml.core` API.

Numerical operators are maintained in :mod:`vpml.solvers`; closure models,
features, and checkpoint I/O are maintained in :mod:`vpml.models`.
Import these focused modules in new code.
"""

from ._jax_precision import enable_x64

enable_x64()

import jax.numpy as jnp

Array = jnp.ndarray

from .models.mlp import (
    init_real_mlp as init_real_mlp,
    mlp_apply as mlp_apply,
)

from .models.hermite_closures import (
    HyperCollisions as HyperCollisions,
    HouLiFilter as HouLiFilter,
    HermiteExponentialFilter as HermiteExponentialFilter,
    NonlocalClosure as NonlocalClosure,
)

from .models.interface_closure import (
    LEGACY_GRID_CUBIC_TEACHER_BACKEND as LEGACY_GRID_CUBIC_TEACHER_BACKEND,
    GRID_CUBIC_SPLINE_TEACHER_BACKEND as GRID_CUBIC_SPLINE_TEACHER_BACKEND,
    HIGHER_ORDER_HERMITE_TEACHER_BACKEND as HIGHER_ORDER_HERMITE_TEACHER_BACKEND,
    INTERFACE_FLUX_TRAINING_MODE as INTERFACE_FLUX_TRAINING_MODE,
    INTERFACE_FLUX_OBJECTIVE as INTERFACE_FLUX_OBJECTIVE,
    INTERFACE_FLUX_LOSS_BACKEND as INTERFACE_FLUX_LOSS_BACKEND,
    INTERFACE_FLUX_CHECKPOINT_SCHEMA_VERSION as INTERFACE_FLUX_CHECKPOINT_SCHEMA_VERSION,
    _LEGACY_INTERFACE_FLUX_TRIPLE as _LEGACY_INTERFACE_FLUX_TRIPLE,
    normalize_teacher_backend_name as normalize_teacher_backend_name,
    LearnedInterfaceClosure as LearnedInterfaceClosure,
    _xavier_normal as _xavier_normal,
    init_interface_closure_params as init_interface_closure_params,
    interface_closure_apply as interface_closure_apply,
)

from .models.interface_checkpoint import (
    save_learned_interface_closure_npz as save_learned_interface_closure_npz,
    load_learned_interface_closure_npz as load_learned_interface_closure_npz,
)

from .models.interface_inference import (
    _hermitian_rfft_weights as _hermitian_rfft_weights,
    learned_closure_global_indicators as learned_closure_global_indicators,
    learned_closure_raw_base_features as learned_closure_raw_base_features,
    learned_closure_raw_features as learned_closure_raw_features,
    scale_learned_closure_raw_features as scale_learned_closure_raw_features,
    learned_interface_q_hat as learned_interface_q_hat,
    _standardized_learned_closure_features as _standardized_learned_closure_features,
    learned_boundary_flux_hat as learned_boundary_flux_hat,
)

from .solvers.implicit import (
    tridiag_solve as tridiag_solve,
    implicit_midpoint_jfnk_step as implicit_midpoint_jfnk_step,
)

from .solvers.hermite import (
    e_hat_from_rho_hat as e_hat_from_rho_hat,
    e_hat_history_from_a_hat_history as e_hat_history_from_a_hat_history,
    rfft_x as rfft_x,
    irfft_x as irfft_x,
    _rfft_x as _rfft_x,
    _irfft_x as _irfft_x,
    hermite_basis_phi_scaled as hermite_basis_phi_scaled,
    hermite_basis_phi as hermite_basis_phi,
    _mask_23 as _mask_23,
    FourierHermiteIMEX as FourierHermiteIMEX,
    hermite_damping_term as hermite_damping_term,
)
