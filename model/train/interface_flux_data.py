"""Compatibility imports for :mod:`vpml.datasets.interface_flux`.

New callers should import the maintained implementation directly.
"""

from vpml.datasets.interface_flux import (
    IC_MANIFEST_FORMAT,
    REFERENCE_CACHE_FORMAT,
    IC_GENERATOR_VERSION,
    IC_SPLIT_TRAIN,
    IC_SPLIT_HELDOUT,
    IC_SPLITS,
    DEFAULT_AMPLITUDE_RANGES,
    sample_initial_condition,
    atomic_write,
    _canonical_json_bytes,
    sha256_file,
    sha256_json,
    _write_json_atomic,
    evaluate_manifest_case,
    _heldout_indices,
    build_ic_manifest,
    validate_ic_manifest,
    load_ic_manifest,
    load_or_create_ic_manifest,
    reference_cache_directory,
    initialize_reference_cache,
    case_shard_paths,
    case_shard_is_complete,
    write_case_shard,
    load_sharded_reference,
    StreamingMoments,
    grouped_history_gather,
)
