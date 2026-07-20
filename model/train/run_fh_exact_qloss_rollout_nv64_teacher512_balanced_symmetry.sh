#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Balanced exact-q experiment with physical symmetry constraints and a fixed
# per-regime q scale so equal relative errors have comparable training weight.
export TRAIN_EXACT_Q_REGIME_BALANCED_LOSS="${TRAIN_EXACT_Q_REGIME_BALANCED_LOSS:-1}"
export TRAIN_EQUILIBRIUM_CENTERED_CLOSURE="${TRAIN_EQUILIBRIUM_CENTERED_CLOSURE:-1}"
export TRAIN_COMPLEX_ISOTROPIC_NORMALIZATION="${TRAIN_COMPLEX_ISOTROPIC_NORMALIZATION:-1}"
export TRAIN_EXACT_TRANSLATION_AUGMENTATION="${TRAIN_EXACT_TRANSLATION_AUGMENTATION:-1}"
export TRAIN_ROLLOUT_HORIZON="${TRAIN_ROLLOUT_HORIZON:-128}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-30}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
export TRAIN_LR="${TRAIN_LR:-1e-4}"
export TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-0.5}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}"

OUTDIR="${1:-${REPO_ROOT}/out_bench/fh_exact_qloss_rollout_H${TRAIN_ROLLOUT_HORIZON}_nv64_teacher512_balanced8x3_symmetry_B${TRAIN_BATCH_SIZE}_steps${TRAIN_STEPS_PER_EPOCH}}"

exec "${SCRIPT_DIR}/run_fh_exact_qloss_rollout_nv64_teacher512_balanced_regimes.sh" "${OUTDIR}"
