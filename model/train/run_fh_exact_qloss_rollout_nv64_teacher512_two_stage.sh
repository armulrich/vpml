#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

OUTDIR="${1:-${REPO_ROOT}/out_bench/fh_exact_qloss_rollout_two_stage_nv64_teacher512}"
STAGE1_OUTDIR="${TRAIN_STAGE1_OUTDIR:-${OUTDIR}/stage1_dyn}"
STAGE2_OUTDIR="${TRAIN_STAGE2_OUTDIR:-${OUTDIR}/stage2_history_lift}"

STAGE1_NV_LIST="${TRAIN_STAGE1_NV_LIST:-512}"
STAGE1_INIT_NV="${TRAIN_STAGE1_INIT_NV:-512}"
STAGE2_NV_LIST="${TRAIN_STAGE2_NV_LIST:-64}"
STAGE1_NV_LIST_COMPACT="${STAGE1_NV_LIST//[[:space:]]/}"
if [[ -n "${TRAIN_STAGE1_NV_TARGETS_CSV:-}" ]]; then
  STAGE1_NV_TARGETS_CSV="${TRAIN_STAGE1_NV_TARGETS_CSV}"
elif [[ "${STAGE1_NV_LIST_COMPACT}" == "512" ]]; then
  STAGE1_NV_TARGETS_CSV="6,9,16,28,50,64,89,159,285,512"
else
  STAGE1_NV_TARGETS_CSV=""
fi

STAGE1_H="${TRAIN_STAGE1_ROLLOUT_HORIZON:-${TRAIN_ROLLOUT_HORIZON:-128}}"
STAGE2_H="${TRAIN_STAGE2_ROLLOUT_HORIZON:-8}"

STAGE1_BATCH_SIZE="${TRAIN_STAGE1_BATCH_SIZE:-64}"
STAGE2_BATCH_SIZE="${TRAIN_STAGE2_BATCH_SIZE:-16}"
STAGE1_STEPS_PER_EPOCH="${TRAIN_STAGE1_STEPS_PER_EPOCH:-${TRAIN_STEPS_PER_EPOCH:-30}}"
STAGE2_STEPS_PER_EPOCH="${TRAIN_STAGE2_STEPS_PER_EPOCH:-30}"
STAGE1_EPOCHS="${TRAIN_STAGE1_EPOCHS:-${TRAIN_EPOCHS:-100}}"
STAGE2_EPOCHS="${TRAIN_STAGE2_EPOCHS:-50}"
STAGE1_LR="${TRAIN_STAGE1_LR:-${TRAIN_LR:-1e-4}}"
STAGE2_LR="${TRAIN_STAGE2_LR:-${TRAIN_LR:-1e-4}}"
STAGE1_GRAD_CLIP="${TRAIN_STAGE1_GRAD_CLIP:-${TRAIN_GRAD_CLIP:-0.5}}"
STAGE2_GRAD_CLIP="${TRAIN_STAGE2_GRAD_CLIP:-${TRAIN_GRAD_CLIP:-0.5}}"
STAGE2_HISTORY_NV="${TRAIN_STAGE2_TAIL_HISTORY_NV:-${TRAIN_TAIL_HISTORY_NV:-512}}"
STAGE2_HISTORY_N_MAX="${TRAIN_STAGE2_TAIL_HISTORY_N_MAX:-${TRAIN_TAIL_HISTORY_N_MAX:-512}}"
STAGE2_HISTORY_LAGS="${TRAIN_STAGE2_TAIL_HISTORY_LAGS:-${TRAIN_TAIL_HISTORY_LAGS:-8}}"

RUN_STAGE1="${RUN_STAGE1:-auto}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_STAGE2_EVAL="${RUN_STAGE2_EVAL:-1}"
STAGE1_CHECKPOINT="${TRAIN_STAGE2_INIT_CHECKPOINT:-${STAGE1_OUTDIR}/models/nv${STAGE1_INIT_NV}/interface_closure.npz}"

stage1_checkpoints_exist() {
  local nv_values nv trimmed
  IFS=',' read -r -a nv_values <<< "${STAGE1_NV_LIST}"
  for nv in "${nv_values[@]}"; do
    trimmed="${nv//[[:space:]]/}"
    if [[ -z "${trimmed}" ]]; then
      continue
    fi
    if [[ ! -f "${STAGE1_OUTDIR}/models/nv${trimmed}/interface_closure.npz" ]]; then
      return 1
    fi
  done
  return 0
}

if [[ "${RUN_STAGE1}" == "auto" ]]; then
  if stage1_checkpoints_exist; then
    RUN_STAGE1="0"
  else
    RUN_STAGE1="1"
  fi
fi

if [[ "${RUN_STAGE1}" == "0" && "${RUN_STAGE2}" != "0" ]] && [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
  echo "Stage 1 is disabled, but the required stage-2 init checkpoint is missing: ${STAGE1_CHECKPOINT}" >&2
  exit 1
fi

echo "[fh-exact-qloss-two-stage] [1/2] Stage 1: exact q-rollout dynamics only"
if [[ "${RUN_STAGE1}" == "0" ]]; then
  echo "[fh-exact-qloss-two-stage] [1/2] Reusing existing stage 1 checkpoint(s) from ${STAGE1_OUTDIR}/models"
else
  env \
    RUN_TRAIN="${RUN_STAGE1}" \
    RUN_EVAL=0 \
    CHECKPOINT_ROOT="${STAGE1_OUTDIR}/models" \
    NV_LIST="${STAGE1_NV_LIST}" \
    TRAIN_NV_LADDER_MODE="${TRAIN_STAGE1_NV_LADDER_MODE:-fixed_ratio}" \
    TRAIN_NV_TARGETS_CSV="${STAGE1_NV_TARGETS_CSV}" \
    TRAIN_TAIL_CHAIN=0 \
    TRAIN_ROLLOUT_HORIZON="${STAGE1_H}" \
    TRAIN_BATCH_SIZE="${STAGE1_BATCH_SIZE}" \
    TRAIN_STEPS_PER_EPOCH="${STAGE1_STEPS_PER_EPOCH}" \
    TRAIN_EPOCHS="${STAGE1_EPOCHS}" \
    TRAIN_LR="${STAGE1_LR}" \
    TRAIN_GRAD_CLIP="${STAGE1_GRAD_CLIP}" \
    "${SCRIPT_DIR}/run_fh_exact_qloss_rollout_nv64_teacher512.sh" \
    "${STAGE1_OUTDIR}"
fi

echo "[fh-exact-qloss-two-stage] [2/2] Stage 2: history lift initialized from stage 1"
env \
  RUN_TRAIN="${RUN_STAGE2}" \
  RUN_EVAL="${RUN_STAGE2_EVAL}" \
  CHECKPOINT_ROOT="${STAGE2_OUTDIR}/models" \
  INIT_CHECKPOINT_PATH="${STAGE1_CHECKPOINT}" \
  NV_LIST="${STAGE2_NV_LIST}" \
  TRAIN_NV_LADDER_MODE=target_only \
  TRAIN_TAIL_CHAIN=0 \
  TRAIN_TAIL_CHAIN_RECURSIVE_LIFT=0 \
  TRAIN_TAIL_HISTORY_LIFT=1 \
  TRAIN_TAIL_HISTORY_NV="${STAGE2_HISTORY_NV}" \
  TRAIN_TAIL_HISTORY_N_MAX="${STAGE2_HISTORY_N_MAX}" \
  TRAIN_TAIL_HISTORY_LAGS="${STAGE2_HISTORY_LAGS}" \
  TRAIN_ROLLOUT_HORIZON="${STAGE2_H}" \
  TRAIN_BATCH_SIZE="${STAGE2_BATCH_SIZE}" \
  TRAIN_STEPS_PER_EPOCH="${STAGE2_STEPS_PER_EPOCH}" \
  TRAIN_EPOCHS="${STAGE2_EPOCHS}" \
  TRAIN_LR="${STAGE2_LR}" \
  TRAIN_GRAD_CLIP="${STAGE2_GRAD_CLIP}" \
  "${SCRIPT_DIR}/run_fh_exact_qloss_rollout_nv64_teacher512.sh" \
  "${STAGE2_OUTDIR}"

cat <<EOF

Two-stage exact q-rollout complete.

Stage 1 dynamics checkpoint root:
  ${STAGE1_OUTDIR}/models

Stage 2 history-lift checkpoint root:
  ${STAGE2_OUTDIR}/models

Stage 1 Nv list:    ${STAGE1_NV_LIST}
Stage 1 Nv targets: ${STAGE1_NV_TARGETS_CSV:-<auto ${TRAIN_STAGE1_NV_LADDER_MODE:-fixed_ratio}>}
Stage 2 Nv list:    ${STAGE2_NV_LIST}
Stage 1 H:          ${STAGE1_H}
Stage 2 H:          ${STAGE2_H}
Stage 2 hist Nv:    ${STAGE2_HISTORY_NV}
Stage 2 hist n max: ${STAGE2_HISTORY_N_MAX}
Stage 2 hist lags:  ${STAGE2_HISTORY_LAGS}
Stage 2 init ckpt:  ${STAGE1_CHECKPOINT}
EOF
