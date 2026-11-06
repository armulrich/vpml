"""Compatibility imports for :mod:`vpml.training.wavenumber_evaluation`.

New callers should import the maintained implementation directly.
"""

from vpml.training.wavenumber_evaluation import (
    base,
    write_new_json,
    rollout_burles_latent_closure,
    SelfGeneratedFieldErrorMetric,
    FieldErrorConfig,
    EarlyElectricFieldGrowthMetric,
    EarlyGrowthConfig,
    rollout_factory,
    autonomous_case,
    evaluate,
)
