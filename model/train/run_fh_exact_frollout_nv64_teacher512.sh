#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Pure direct-distribution rollout experiment. The deployed solver remains
# Nv=64; the spline HR512 field is only the physical-space training target.
export NV_LIST=64
export TRAIN_NV_LADDER_MODE="${TRAIN_NV_LADDER_MODE:-fixed_ratio}"
export TRAIN_FIXED_RATIO="${TRAIN_FIXED_RATIO:-1.8}"
export TRAIN_NV_TARGETS_CSV="${TRAIN_NV_TARGETS_CSV:-}"
export TRAIN_EXACT_ROLLOUT_OBJECTIVE="f_rollout"
export TEACHER_NV="${TEACHER_NV:-512}"
export TRAIN_TEACHER_NV="${TRAIN_TEACHER_NV:-512}"
export T_FINAL="${T_FINAL:-60.0}"
export TRAIN_NONLINEAR_T="${TRAIN_NONLINEAR_T:-60.0}"
export SNAPSHOT_TIMES="${SNAPSHOT_TIMES:-20.0,40.0,60.0}"
export TRAIN_REGIMES="${TRAIN_REGIMES:-nonlinear_landau_strong}"
export TRAIN_VAL_FRACTION="${TRAIN_VAL_FRACTION:-0}"
export TRAIN_EXACT_STORE_TRAIN_QPAIRS="${TRAIN_EXACT_STORE_TRAIN_QPAIRS:-0}"
export TRAIN_EXACT_TARGET_SAMPLING="${TRAIN_EXACT_TARGET_SAMPLING:-cycle}"
export PHASE_REFERENCE_NV="${PHASE_REFERENCE_NV:-512}"

# Keep this first direct-f run computationally comparable to the established
# Nv64 HR512 exact-q H=128 experiment.
export TRAIN_ROLLOUT_HORIZON="${TRAIN_ROLLOUT_HORIZON:-128}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-30}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
export TRAIN_LR="${TRAIN_LR:-1e-4}"
export TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-0.5}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}"

# This is a closure-only experiment: no tail chain or post-hoc history lift
# can enter training or alter the normal Metric 1/2 solver evaluation.
export TRAIN_TAIL_CHAIN=0
export TRAIN_TAIL_HISTORY_LIFT=0
export RUN_ORACLE_FIG10_DECOMPOSITION=0
export RUN_PHASE_SPACE_VIDEO=0

OUTDIR="${1:-${REPO_ROOT}/out_bench/fh_exact_frollout_H${TRAIN_ROLLOUT_HORIZON}_nv64_teacher512_strong_B${TRAIN_BATCH_SIZE}_steps${TRAIN_STEPS_PER_EPOCH}}"

exec "${SCRIPT_DIR}/run_fh_exact_qloss_rollout.sh" "${OUTDIR}"
