"""Compatibility imports for the historical :mod:`vpml.low_moment` API.

Numerical operators are maintained in :mod:`vpml.solvers`; closure models,
features, and checkpoint I/O are maintained in :mod:`vpml.models`.
Import these focused modules in new code.
"""

from ._jax_precision import enable_x64

enable_x64()

import jax

Array = jax.Array

from .solvers.moments import (
    DEFAULT_DENSITY_FLOOR as DEFAULT_DENSITY_FLOOR,
    DEFAULT_PRESSURE_FLOOR as DEFAULT_PRESSURE_FLOOR,
    primitive_fields as primitive_fields,
    electric_field_from_density as electric_field_from_density,
    spectral_derivative as spectral_derivative,
    _dealias as _dealias,
    _batched_low_moment_flux_terms as _batched_low_moment_flux_terms,
    low_moment_rhs as low_moment_rhs,
    low_moment_rk4_step as low_moment_rk4_step,
    limit_low_moment_state as limit_low_moment_state,
)

from .models.spectral_layers import (
    _complex_weights as _complex_weights,
    spectral_channel_operator as spectral_channel_operator,
    _causal_temporal_convolution as _causal_temporal_convolution,
    _rms_with_floor as _rms_with_floor,
)

from .models.moment_parameters import (
    init_spectral_memory_params as init_spectral_memory_params,
    init_explicit_window_params as init_explicit_window_params,
    init_window_fno_params as init_window_fno_params,
    init_burles_latent_fno_params as init_burles_latent_fno_params,
    init_causal_spacetime_operator_params as init_causal_spacetime_operator_params,
)

from .models.moment_features import (
    DEFAULT_NORMALIZED_HEAT_FLUX_BOUND as DEFAULT_NORMALIZED_HEAT_FLUX_BOUND,
    FIXED_INPUT_SCALING as FIXED_INPUT_SCALING,
    DYNAMIC_INPUT_SCALING as DYNAMIC_INPUT_SCALING,
    INPUT_SCALING_KINDS as INPUT_SCALING_KINDS,
    _normalized_closure_fields as _normalized_closure_fields,
    _window_fno_features as _window_fno_features,
    encode_explicit_window_history as encode_explicit_window_history,
)

from .models.moment_closures import (
    explicit_window_closure_step as explicit_window_closure_step,
    initialize_burles_latent_state as initialize_burles_latent_state,
    burles_latent_closure_rhs as burles_latent_closure_rhs,
    spectral_memory_closure_step as spectral_memory_closure_step,
)

from .solvers.moment_rollout import (
    warm_spectral_memory as warm_spectral_memory,
    rollout_low_moment_closure as rollout_low_moment_closure,
    rollout_explicit_window_closure as rollout_explicit_window_closure,
    rollout_burles_latent_closure as rollout_burles_latent_closure,
)
