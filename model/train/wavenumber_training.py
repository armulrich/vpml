"""Compatibility imports for :mod:`vpml.training.wavenumber_training`.

New callers should import the maintained implementation directly.
"""

from vpml.training.wavenumber_training import (
    base,
    sha256_json,
    sha256_file,
    REGIMES,
    epoch_batches,
    write_new_json,
    init_burles_latent_fno_params,
    rollout_burles_latent_closure,
    Configuration,
    initialize,
    loss_factory,
    load_stats,
    batch_from_states,
    convert_old_case,
    tree_hash,
    save_snapshot,
    load_snapshot,
    update_factory,
    train_segment,
)
