#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Reduced-model experiment:
# train a deployment Nv=64 learned closure against an HR teacher projected at Nv=512.
export NV_LIST="${NV_LIST:-64}"
export TEACHER_NV="${TEACHER_NV:-512}"
export TRAIN_TEACHER_NV="${TRAIN_TEACHER_NV:-512}"
export T_FINAL="${T_FINAL:-60.0}"
export TRAIN_NONLINEAR_T="${TRAIN_NONLINEAR_T:-60.0}"
export SNAPSHOT_TIMES="${SNAPSHOT_TIMES:-20.0,40.0,60.0}"
export TRAIN_REGIMES="${TRAIN_REGIMES:-nonlinear_landau_strong}"
export TRAIN_NV_LADDER_MODE="${TRAIN_NV_LADDER_MODE:-fixed_ratio}"
export TRAIN_EXACT_STORE_TRAIN_QPAIRS="${TRAIN_EXACT_STORE_TRAIN_QPAIRS:-0}"
export TRAIN_EXACT_TARGET_SAMPLING="${TRAIN_EXACT_TARGET_SAMPLING:-cycle}"
export TRAIN_VAL_FRACTION="${TRAIN_VAL_FRACTION:-0}"

# Show the full HR512 teacher in fig10's reference column while keeping the
# truncated/learned columns at the deployment Nv=64 resolution.
export PHASE_REFERENCE_NV="${PHASE_REFERENCE_NV:-512}"

# Runnable defaults for the next reduced-model sweep. Override these from the
# environment when sweeping H or compute budget.
export TRAIN_ROLLOUT_HORIZON="${TRAIN_ROLLOUT_HORIZON:-128}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-30}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
export TRAIN_LR="${TRAIN_LR:-1e-4}"
export TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-0.5}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}"
export TRAIN_TAIL_CHAIN="${TRAIN_TAIL_CHAIN:-${TRAIN_TAIL_DECODER:-0}}"
export TRAIN_TAIL_CHAIN_NV="${TRAIN_TAIL_CHAIN_NV:-${TRAIN_TAIL_DECODER_NV:-512}}"
export TRAIN_TAIL_CHAIN_N_MIN="${TRAIN_TAIL_CHAIN_N_MIN:-}"
export TRAIN_TAIL_CHAIN_N_MAX="${TRAIN_TAIL_CHAIN_N_MAX:-${TRAIN_TAIL_DECODER_N_MAX:-512}}"
export TRAIN_TAIL_CHAIN_CHUNK_SIZE="${TRAIN_TAIL_CHAIN_CHUNK_SIZE:-16}"
export TRAIN_TAIL_CHAIN_RECURSIVE_LIFT="${TRAIN_TAIL_CHAIN_RECURSIVE_LIFT:-0}"
export TRAIN_LAMBDA_TAIL_CHAIN="${TRAIN_LAMBDA_TAIL_CHAIN:-${TRAIN_LAMBDA_TAIL_DECODER:-1.0}}"

OUTDIR="${1:-${REPO_ROOT}/out_bench/fh_exact_qloss_rollout_H${TRAIN_ROLLOUT_HORIZON}_nv64_teacher512_strong_ladder_B${TRAIN_BATCH_SIZE}_steps${TRAIN_STEPS_PER_EPOCH}}"

exec "${SCRIPT_DIR}/run_fh_exact_qloss_rollout.sh" "${OUTDIR}"
