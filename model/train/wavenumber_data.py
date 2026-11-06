"""Compatibility imports for :mod:`vpml.datasets.wavenumber`.

New callers should import the maintained implementation directly.
"""

from vpml.datasets.wavenumber import (
    evaluate_manifest_case,
    sha256_json,
    REGIMES,
    AMPLITUDES,
    FUNDAMENTALS,
    HELDOUT_FUNDAMENTALS,
    TRAIN_FUNDAMENTALS,
    ANCHORS,
    SMALL_TRAIN_FUNDAMENTALS,
    SMALL_HELDOUT_FUNDAMENTALS,
    build_five_epoch_manifest,
    build_manifest,
    validate_manifest,
    epoch_batches,
    write_new_json,
)
