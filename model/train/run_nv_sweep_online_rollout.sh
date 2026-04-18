#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_OUTDIR="${REPO_ROOT}/out_bench/nv_sweep_online_rollout"
DEFAULT_NV_LIST="8,64,256,300,512"  # match the legacy Nv sweep wrappers
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON:-python}"
fi

OUTDIR="${1:-${DEFAULT_OUTDIR}}"
NV_LIST="${NV_LIST:-${DEFAULT_NV_LIST}}"
NX="${NX:-200}"
DT="${DT:-0.01}"
T_FINAL="${T_FINAL:-40.0}"
EPS="${EPS:-0.5}"
K0="${K0:-0.5}"
SNAPSHOT_TIMES="${SNAPSHOT_TIMES:-20.0,40.0}"
NV_PLOT="${NV_PLOT:-1000}"
PHASE_VMIN="${PHASE_VMIN:-0.0}"
PHASE_VMAX="${PHASE_VMAX:-0.5}"
PHASE_VRANGE="${PHASE_VRANGE:--4.0,4.0}"
DEALIAS_23="${DEALIAS_23:-1}"
NONLOCAL_MU="${NONLOCAL_MU:--1.017234}"

TEACHER_NX="${TEACHER_NX:-200}"
TEACHER_NV="${TEACHER_NV:-512}"
TEACHER_DT="${TEACHER_DT:-0.01}"
TEACHER_VMIN="${TEACHER_VMIN:--8.0}"
TEACHER_VMAX="${TEACHER_VMAX:-8.0}"

FIELD_NUM_LOW_MODES="${FIELD_NUM_LOW_MODES:-}"
FIELD_K_MAX="${FIELD_K_MAX:-}"
RUN_TRAIN="${RUN_TRAIN:-1}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTDIR}/models}"
TRAIN_NM="${TRAIN_NM:-6}"
TRAIN_HIDDEN_WIDTH="${TRAIN_HIDDEN_WIDTH:-128}"
TRAIN_RES_BLOCKS="${TRAIN_RES_BLOCKS:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-300}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-1.0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-0}"
TRAIN_SEED="${TRAIN_SEED:-0}"
TRAIN_N_LOW="${TRAIN_N_LOW:-2}"
TRAIN_VAL_FRACTION="${TRAIN_VAL_FRACTION:-0.2}"
TRAIN_REGIMES="${TRAIN_REGIMES:-linear_landau,nonlinear_landau_weak,nonlinear_landau_strong}"
TRAIN_CONTEXT_MODE="${TRAIN_CONTEXT_MODE:-none}"
TRAIN_TAIL_START_FRACTION="${TRAIN_TAIL_START_FRACTION:-0.6666666666666666}"
TRAIN_LAMBDA_E="${TRAIN_LAMBDA_E:-0.5}"
TRAIN_LAMBDA_DIST="${TRAIN_LAMBDA_DIST:-1.0}"
TRAIN_LAMBDA_TAIL="${TRAIN_LAMBDA_TAIL:-0.05}"
TRAIN_LAMBDA_NEG="${TRAIN_LAMBDA_NEG:-0.05}"
TRAIN_LAMBDA_REG="${TRAIN_LAMBDA_REG:-1e-6}"
TRAIN_ROLLOUT_DEALIAS_23="${TRAIN_ROLLOUT_DEALIAS_23:-1}"
TRAIN_ONLINE_LOSS_BACKEND="${TRAIN_ONLINE_LOSS_BACKEND:-field_distribution_v1}"
TRAIN_ONLINE_V_PROBES="${TRAIN_ONLINE_V_PROBES:-64}"
TRAIN_ONLINE_CASE_BATCH_SIZE="${TRAIN_ONLINE_CASE_BATCH_SIZE:-1}"

TRAIN_TEACHER_NX="${TRAIN_TEACHER_NX:-${TEACHER_NX}}"
TRAIN_TEACHER_NV="${TRAIN_TEACHER_NV:-${TEACHER_NV}}"
TRAIN_TEACHER_DT="${TRAIN_TEACHER_DT:-${TEACHER_DT}}"
TRAIN_TEACHER_VMIN="${TRAIN_TEACHER_VMIN:-${TEACHER_VMIN}}"
TRAIN_TEACHER_VMAX="${TRAIN_TEACHER_VMAX:-${TEACHER_VMAX}}"
TRAIN_LINEAR_T="${TRAIN_LINEAR_T:-20.0}"
TRAIN_LINEAR_EPS="${TRAIN_LINEAR_EPS:-0.01}"
TRAIN_LINEAR_MODES="${TRAIN_LINEAR_MODES:-0.5,1.0,1.5,2.0}"
TRAIN_LINEAR_NUM_SAMPLES="${TRAIN_LINEAR_NUM_SAMPLES:-8}"
TRAIN_LINEAR_SEED="${TRAIN_LINEAR_SEED:-0}"
TRAIN_NONLINEAR_T="${TRAIN_NONLINEAR_T:-20.0}"
TRAIN_NONLINEAR_K0="${TRAIN_NONLINEAR_K0:-${K0}}"
TRAIN_WEAK_EPS="${TRAIN_WEAK_EPS:-0.05,0.1}"
TRAIN_STRONG_EPS="${TRAIN_STRONG_EPS:-0.25,0.5}"

mkdir -p "${OUTDIR}"
cd "${REPO_ROOT}"

ARGS=(
  --outdir "${OUTDIR}"
  --nv-list "${NV_LIST}"
  --Nx "${NX}"
  --dt "${DT}"
  --T "${T_FINAL}"
  --eps "${EPS}"
  --k0 "${K0}"
  --snapshot-times "${SNAPSHOT_TIMES}"
  --Nv-plot "${NV_PLOT}"
  --phase-vmin "${PHASE_VMIN}"
  --phase-vmax "${PHASE_VMAX}"
  "--phase-vrange=${PHASE_VRANGE}"
  --nonlocal-mu "${NONLOCAL_MU}"
  --teacher-Nx "${TEACHER_NX}"
  --teacher-Nv "${TEACHER_NV}"
  --teacher-dt "${TEACHER_DT}"
  --teacher-vmin "${TEACHER_VMIN}"
  --teacher-vmax "${TEACHER_VMAX}"
)

if [[ "${DEALIAS_23}" != "0" ]]; then
  ARGS+=(--dealias-23)
fi
if [[ -n "${FIELD_NUM_LOW_MODES}" ]]; then
  ARGS+=(--field-num-low-modes "${FIELD_NUM_LOW_MODES}")
fi
if [[ -n "${FIELD_K_MAX}" ]]; then
  ARGS+=(--field-k-max "${FIELD_K_MAX}")
fi

mkdir -p "${CHECKPOINT_ROOT}"
IFS=',' read -r -a NV_VALUES <<< "${NV_LIST}"
TOTAL_NV="${#NV_VALUES[@]}"

if [[ "${RUN_TRAIN}" != "0" ]]; then
  echo "[nv-sweep-online-rollout] [1/2] Training one online_rollout checkpoint per deployment Nv"
  for idx in "${!NV_VALUES[@]}"; do
    NV_RAW="${NV_VALUES[idx]}"
    NV="$(echo "${NV_RAW}" | tr -d '[:space:]')"
    if [[ -z "${NV}" ]]; then
      continue
    fi
    MODEL_DIR="${CHECKPOINT_ROOT}/nv${NV}"
    CHECKPOINT_NV="${MODEL_DIR}/interface_closure.npz"
    LOSS_PLOT_NV="${MODEL_DIR}/interface_closure.loss.png"
    mkdir -p "${MODEL_DIR}"

    TRAIN_ARGS=(
      --checkpoint "${CHECKPOINT_NV}"
      --loss-plot "${LOSS_PLOT_NV}"
      --training-mode online_rollout
      --train-objective trajectory
      --online-loss-backend "${TRAIN_ONLINE_LOSS_BACKEND}"
      --online-v-probes "${TRAIN_ONLINE_V_PROBES}"
      --online-case-batch-size "${TRAIN_ONLINE_CASE_BATCH_SIZE}"
      --teacher-backend grid_cubic_spline
      --Nv-targets "${NV}"
      --Nm "${TRAIN_NM}"
      --hidden-width "${TRAIN_HIDDEN_WIDTH}"
      --res-blocks "${TRAIN_RES_BLOCKS}"
      --epochs "${TRAIN_EPOCHS}"
      --lr "${TRAIN_LR}"
      --grad-clip "${TRAIN_GRAD_CLIP}"
      --log-every "${TRAIN_LOG_EVERY}"
      --steps-per-epoch "${TRAIN_STEPS_PER_EPOCH}"
      --seed "${TRAIN_SEED}"
      --n-low "${TRAIN_N_LOW}"
      --val-fraction "${TRAIN_VAL_FRACTION}"
      --context-mode "${TRAIN_CONTEXT_MODE}"
      --tail-start-fraction "${TRAIN_TAIL_START_FRACTION}"
      --lambda-E "${TRAIN_LAMBDA_E}"
      --lambda-dist "${TRAIN_LAMBDA_DIST}"
      --lambda-tail "${TRAIN_LAMBDA_TAIL}"
      --lambda-neg "${TRAIN_LAMBDA_NEG}"
      --lambda-reg "${TRAIN_LAMBDA_REG}"
      --regimes "${TRAIN_REGIMES}"
      --teacher-Nx "${TRAIN_TEACHER_NX}"
      --teacher-Nv "${TRAIN_TEACHER_NV}"
      --teacher-dt "${TRAIN_TEACHER_DT}"
      --teacher-vmin "${TRAIN_TEACHER_VMIN}"
      --teacher-vmax "${TRAIN_TEACHER_VMAX}"
      --linear-T "${TRAIN_LINEAR_T}"
      --linear-eps "${TRAIN_LINEAR_EPS}"
      --linear-modes "${TRAIN_LINEAR_MODES}"
      --linear-num-samples "${TRAIN_LINEAR_NUM_SAMPLES}"
      --linear-seed "${TRAIN_LINEAR_SEED}"
      --nonlinear-T "${TRAIN_NONLINEAR_T}"
      --nonlinear-k0 "${TRAIN_NONLINEAR_K0}"
      --weak-eps "${TRAIN_WEAK_EPS}"
      --strong-eps "${TRAIN_STRONG_EPS}"
    )
    if [[ "${TRAIN_ROLLOUT_DEALIAS_23}" != "0" ]]; then
      TRAIN_ARGS+=(--rollout-dealias-23)
    fi

    echo "[nv-sweep-online-rollout] [1/2] Training closure $((idx + 1))/${TOTAL_NV} for Nv=${NV}"
    "${PYTHON_BIN}" -m model.train.train "${TRAIN_ARGS[@]}"
  done
else
  for NV_RAW in "${NV_VALUES[@]}"; do
    NV="$(echo "${NV_RAW}" | tr -d '[:space:]')"
    if [[ -z "${NV}" ]]; then
      continue
    fi
    CHECKPOINT_NV="${CHECKPOINT_ROOT}/nv${NV}/interface_closure.npz"
    if [[ ! -f "${CHECKPOINT_NV}" ]]; then
      echo "RUN_TRAIN=0 requires an existing checkpoint at ${CHECKPOINT_NV}" >&2
      exit 1
    fi
  done
  echo "[nv-sweep-online-rollout] [1/2] Reusing existing online checkpoints in ${CHECKPOINT_ROOT}"
fi

ARGS+=(--checkpoint-dir "${CHECKPOINT_ROOT}")
echo "[nv-sweep-online-rollout] [2/2] Running nonlinear Nv sweep"
"${PYTHON_BIN}" -m model.eval_nv_sweep "${ARGS[@]}"

cat <<EOF

Done.

Artifacts:
  mode:           online_rollout_field_distribution_v1
  checkpoint dir: ${CHECKPOINT_ROOT}
  summary:        ${OUTDIR}/summary.json
  metric 1:       ${OUTDIR}/nv_sweep_metric1.png
  metric 2:       ${OUTDIR}/nv_sweep_metric2.png
  phase space:    ${OUTDIR}/fig10_learned_vs_nonlocal_nv_sweep_phase_space.png
  phase payload:  ${OUTDIR}/nv_sweep_phase_space_payload.npz

Defaults:
  objective:      trajectory
  loss backend:   ${TRAIN_ONLINE_LOSS_BACKEND}
  v probes:       ${TRAIN_ONLINE_V_PROBES}
  case batch:     ${TRAIN_ONLINE_CASE_BATCH_SIZE}
  train Nm:       ${TRAIN_NM}
  steps/epoch:    ${TRAIN_STEPS_PER_EPOCH}
  Nv list:        ${NV_LIST}
EOF
