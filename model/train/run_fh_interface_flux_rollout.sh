#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-True}"
export VPML_JAX_BACKEND="${VPML_JAX_BACKEND:-cpu}"

TRAIN_ROLLOUT_HORIZON="${TRAIN_ROLLOUT_HORIZON:-128}"
TRAIN_T_FINAL="${TRAIN_T_FINAL:-60.0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-30}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-0.5}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}"
TRAIN_PRECISION="${TRAIN_PRECISION:-float32}"
TRAIN_SEED="${TRAIN_SEED:-0}"
TRAIN_NM="${TRAIN_NM:-6}"
TRAIN_HIDDEN_WIDTH="${TRAIN_HIDDEN_WIDTH:-128}"
TRAIN_RES_BLOCKS="${TRAIN_RES_BLOCKS:-2}"
TRAIN_N_LOW="${TRAIN_N_LOW:-2}"
TRAIN_HISTORY_STRIDE="${TRAIN_HISTORY_STRIDE:-20}"
TRAIN_INIT_CHECKPOINT="${TRAIN_INIT_CHECKPOINT:-}"

TEACHER_NX="${TEACHER_NX:-256}"
TEACHER_NV="${TEACHER_NV:-512}"
TEACHER_DT="${TEACHER_DT:-0.01}"
TEACHER_VMIN="${TEACHER_VMIN:--8.0}"
TEACHER_VMAX="${TEACHER_VMAX:-8.0}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_TRAINING_CASES="${EVAL_TRAINING_CASES:-1}"
EVAL_NX="${EVAL_NX:-200}"
EVAL_DT="${EVAL_DT:-0.005}"
EVAL_EPS="${EVAL_EPS:-0.5}"
EVAL_K0="${EVAL_K0:-0.5}"
EVAL_SNAPSHOT_TIMES="${EVAL_SNAPSHOT_TIMES:-20.0,40.0,60.0}"
EVAL_NV_PLOT="${EVAL_NV_PLOT:-1000}"
EVAL_PHASE_VMIN="${EVAL_PHASE_VMIN:-0.0}"
EVAL_PHASE_VMAX="${EVAL_PHASE_VMAX:-0.5}"
EVAL_PHASE_VRANGE="${EVAL_PHASE_VRANGE:--4.0,4.0}"
EVAL_NONLOCAL_MU="${EVAL_NONLOCAL_MU:--1.017234}"

OUTDIR="${1:-${REPO_ROOT}/out_bench/fh_interface_flux_H${TRAIN_ROLLOUT_HORIZON}_T${TRAIN_T_FINAL}_B${TRAIN_BATCH_SIZE}_steps${TRAIN_STEPS_PER_EPOCH}}"
MODEL_DIR="${OUTDIR}/models/nv64"
CHECKPOINT="${MODEL_DIR}/interface_closure.npz"
DATASET_CACHE="${MODEL_DIR}/interface_closure_interface_flux_histories.npz"
LOSS_PLOT="${MODEL_DIR}/interface_closure.loss.png"

mkdir -p "${MODEL_DIR}"
cd "${REPO_ROOT}"

if [[ "${RUN_TRAIN}" != "0" ]]; then
  TRAIN_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --dataset-cache "${DATASET_CACHE}"
    --loss-plot "${LOSS_PLOT}"
    --rollout-horizon "${TRAIN_ROLLOUT_HORIZON}"
    --precision "${TRAIN_PRECISION}"
    --batch-size "${TRAIN_BATCH_SIZE}"
    --steps-per-epoch "${TRAIN_STEPS_PER_EPOCH}"
    --epochs "${TRAIN_EPOCHS}"
    --lr "${TRAIN_LR}"
    --grad-clip "${TRAIN_GRAD_CLIP}"
    --log-every "${TRAIN_LOG_EVERY}"
    --seed "${TRAIN_SEED}"
    --T-final "${TRAIN_T_FINAL}"
    --Nm "${TRAIN_NM}"
    --hidden-width "${TRAIN_HIDDEN_WIDTH}"
    --res-blocks "${TRAIN_RES_BLOCKS}"
    --n-low "${TRAIN_N_LOW}"
    --history-stride "${TRAIN_HISTORY_STRIDE}"
    --teacher-Nx "${TEACHER_NX}"
    --teacher-Nv "${TEACHER_NV}"
    --teacher-dt "${TEACHER_DT}"
    --teacher-vmin "${TEACHER_VMIN}"
    --teacher-vmax "${TEACHER_VMAX}"
  )
  if [[ -n "${TRAIN_INIT_CHECKPOINT}" ]]; then
    if [[ ! -f "${TRAIN_INIT_CHECKPOINT}" ]]; then
      echo "TRAIN_INIT_CHECKPOINT does not exist: ${TRAIN_INIT_CHECKPOINT}" >&2
      exit 1
    fi
    TRAIN_ARGS+=(--init-checkpoint "${TRAIN_INIT_CHECKPOINT}")
  fi
  echo "[interface-flux] [1/2] Training canonical Nv=64 closure with cutoff cycle 6,7,12,20,36,64"
  "${PYTHON_BIN}" -m model.train.interface_flux_rollout "${TRAIN_ARGS[@]}"
else
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "RUN_TRAIN=0 requires ${CHECKPOINT}" >&2
    exit 1
  fi
  echo "[interface-flux] [1/2] Reusing ${CHECKPOINT}"
fi

if [[ "${RUN_EVAL}" != "0" ]]; then
  COMMON_EVAL_ARGS=(
    --checkpoint-dir "${OUTDIR}/models"
    --nv-list 64
    --Nx "${EVAL_NX}"
    --dt "${EVAL_DT}"
    --T "${TRAIN_T_FINAL}"
    --k0 "${EVAL_K0}"
    --snapshot-times "${EVAL_SNAPSHOT_TIMES}"
    --Nv-plot "${EVAL_NV_PLOT}"
    --phase-vmin "${EVAL_PHASE_VMIN}"
    --phase-vmax "${EVAL_PHASE_VMAX}"
    "--phase-vrange=${EVAL_PHASE_VRANGE}"
    --phase-reference-mode raw_hr_grid
    --nonlocal-mu "${EVAL_NONLOCAL_MU}"
    --teacher-Nx "${TEACHER_NX}"
    --teacher-Nv "${TEACHER_NV}"
    --teacher-dt "${TEACHER_DT}"
    --teacher-vmin "${TEACHER_VMIN}"
    --teacher-vmax "${TEACHER_VMAX}"
    --dealias-23
  )
  if [[ "${EVAL_TRAINING_CASES}" != "0" ]]; then
    echo "[interface-flux] [2/2] Evaluating every canonical training IC"
    "${PYTHON_BIN}" -m model.eval_training_cases \
      "${COMMON_EVAL_ARGS[@]}" \
      --outdir "${OUTDIR}/evaluation_cases" \
      --regimes linear_landau,nonlinear_landau_weak,nonlinear_landau_strong \
      --linear-eps 0.01 \
      --linear-modes 0.5,1.0,1.5,2.0 \
      --linear-num-samples 8 \
      --linear-seed 0 \
      --weak-eps 0.02,0.03,0.05,0.07,0.10,0.12,0.15,0.18 \
      --strong-eps 0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.65
  else
    echo "[interface-flux] [2/2] Running Metric 1/2 and raw-HR Fig. 10 evaluation"
    "${PYTHON_BIN}" -m model.eval_nv_sweep \
      "${COMMON_EVAL_ARGS[@]}" \
      --outdir "${OUTDIR}" \
      --eps "${EVAL_EPS}"
  fi
else
  echo "[interface-flux] [2/2] Evaluation disabled"
fi

cat <<EOF

Canonical interface-flux run complete.
  checkpoint: ${CHECKPOINT}
  metrics:    ${MODEL_DIR}/interface_closure.metrics.npz
  loss plot:  ${LOSS_PLOT}
  H:          ${TRAIN_ROLLOUT_HORIZON}
  T final:    ${TRAIN_T_FINAL}
  precision:  ${TRAIN_PRECISION}
  backend:    regime_balanced_all_k_interface_flux
EOF
