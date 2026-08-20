#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

REFERENCE_CACHE="${TRAIN_LOW_MOMENT_REFERENCE_CACHE:-}"
if [[ -z "${REFERENCE_CACHE}" ]]; then
  echo "Set TRAIN_LOW_MOMENT_REFERENCE_CACHE to a completed T=120 reference-cache directory." >&2
  exit 2
fi
if [[ ! -f "${REFERENCE_CACHE}/metadata.json" ]]; then
  echo "Reference cache is incomplete: ${REFERENCE_CACHE}" >&2
  exit 2
fi

ROLLOUT_NX="${TRAIN_ROLLOUT_NX:-256}"
HORIZON="${TRAIN_ROLLOUT_HORIZON:-1024}"
TRAINING_SCHEDULE="${TRAIN_TRAJECTORY_SCHEDULE:-random_windows}"
HORIZON_CURRICULUM="${TRAIN_HORIZON_CURRICULUM:-}"
MEMORY_STEPS="${TRAIN_MEMORY_STEPS:-50}"
MEMORY_STRIDE="${TRAIN_MEMORY_STRIDE:-10}"
MEMORY_BACKEND="${TRAIN_MEMORY_BACKEND:-latent_recurrent}"
HISTORY_STRIDE="${TRAIN_HISTORY_STRIDE:-20}"
BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EPOCHS="${TRAIN_EPOCHS:-100}"
SUPERVISED_WARMUP_EPOCHS="${TRAIN_SUPERVISED_WARMUP_EPOCHS:-0}"
STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-30}"
GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-2}"
VALIDATION_EVERY="${TRAIN_VALIDATION_EVERY:-5}"
TRAINING_DIAGNOSTIC_CASES="${TRAIN_DIAGNOSTIC_CASES_PER_REGIME:-4}"
DIAGNOSTIC_START_TIMES="${TRAIN_DIAGNOSTIC_START_TIMES:-0,20,40,60,80,100}"
LOSS_EMA_DECAY="${TRAIN_LOSS_EMA_DECAY:-0.95}"
NONFINITE_TRAJECTORY_PENALTY="${TRAIN_NONFINITE_TRAJECTORY_PENALTY:-1e3}"
LEARNING_RATE="${TRAIN_LR:-1e-4}"
GRAD_CLIP="${TRAIN_GRAD_CLIP:-1.0}"
WIDTH="${TRAIN_MODEL_WIDTH:-24}"
SPECTRAL_MODES="${TRAIN_SPECTRAL_MODES:-16}"
CLOSURE_HISTORY_INPUT="${TRAIN_CLOSURE_HISTORY_INPUT:-0}"
RELATIVE_TRAJECTORY_LOSS="${TRAIN_RELATIVE_TRAJECTORY_LOSS:-0}"
RELATIVE_TIME_BLOCK="${TRAIN_RELATIVE_TIME_BLOCK:-0}"
CONVERGENCE_FLOOR_FILE="${TRAIN_CONVERGENCE_FLOOR_FILE:-}"
HEAT_FLUX_BOUND="${TRAIN_NORMALIZED_HEAT_FLUX_BOUND:-128.0}"
DENSITY_FLOOR="${TRAIN_DENSITY_FLOOR:-1e-4}"
PRESSURE_FLOOR="${TRAIN_PRESSURE_FLOOR:-1e-4}"
SEED="${TRAIN_SEED:-1729}"
EVAL_AFTER_TRAINING="${EVAL_AFTER_TRAINING:-1}"
EVALUATE_CHECKPOINT="${EVAL_LOW_MOMENT_CHECKPOINT:-}"
INIT_CHECKPOINT="${TRAIN_LOW_MOMENT_INIT_CHECKPOINT:-}"

OUTDIR="${1:-${REPO_ROOT}/out_bench/low_moment_spectral_memory_H${HORIZON}_B${BATCH_SIZE}_E${EPOCHS}}"

ARGS=(
  --reference-cache "${REFERENCE_CACHE}"
  --outdir "${OUTDIR}"
  --rollout-Nx "${ROLLOUT_NX}"
  --rollout-horizon "${HORIZON}"
  --training-schedule "${TRAINING_SCHEDULE}"
  --horizon-curriculum "${HORIZON_CURRICULUM}"
  --memory-steps "${MEMORY_STEPS}"
  --memory-stride "${MEMORY_STRIDE}"
  --memory-backend "${MEMORY_BACKEND}"
  --history-stride "${HISTORY_STRIDE}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --supervised-warmup-epochs "${SUPERVISED_WARMUP_EPOCHS}"
  --steps-per-epoch "${STEPS_PER_EPOCH}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --validation-every "${VALIDATION_EVERY}"
  --training-diagnostic-cases-per-regime "${TRAINING_DIAGNOSTIC_CASES}"
  --diagnostic-start-times "${DIAGNOSTIC_START_TIMES}"
  --loss-ema-decay "${LOSS_EMA_DECAY}"
  --nonfinite-trajectory-penalty "${NONFINITE_TRAJECTORY_PENALTY}"
  --learning-rate "${LEARNING_RATE}"
  --grad-clip "${GRAD_CLIP}"
  --width "${WIDTH}"
  --spectral-modes "${SPECTRAL_MODES}"
  --relative-time-block "${RELATIVE_TIME_BLOCK}"
  --normalized-heat-flux-bound "${HEAT_FLUX_BOUND}"
  --density-floor "${DENSITY_FLOOR}"
  --pressure-floor "${PRESSURE_FLOOR}"
  --seed "${SEED}"
)

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  ARGS+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi
if [[ "${CLOSURE_HISTORY_INPUT}" == "1" ]]; then
  ARGS+=(--closure-history-input)
fi
if [[ "${RELATIVE_TRAJECTORY_LOSS}" == "1" ]]; then
  ARGS+=(--relative-trajectory-loss)
fi
if [[ -n "${CONVERGENCE_FLOOR_FILE}" ]]; then
  ARGS+=(--convergence-floor-file "${CONVERGENCE_FLOOR_FILE}")
fi
if [[ "${EVAL_AFTER_TRAINING}" == "0" ]]; then
  ARGS+=(--skip-evaluation)
fi
if [[ -n "${EVALUATE_CHECKPOINT}" ]]; then
  ARGS+=(--evaluate-checkpoint "${EVALUATE_CHECKPOINT}")
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m model.train.low_moment_closure "${ARGS[@]}"
