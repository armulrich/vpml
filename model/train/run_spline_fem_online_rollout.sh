#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_OUTDIR="${REPO_ROOT}/out_bench/spline_fem_online_rollout"
DEFAULT_VGRID_LIST="32,64,128,256"
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON:-python}"
fi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-True}"
export VPML_JAX_BACKEND="${VPML_JAX_BACKEND:-cpu}"

OUTDIR="${1:-${DEFAULT_OUTDIR}}"
VGRID_LIST="${VGRID_LIST:-${NV_LIST:-${DEFAULT_VGRID_LIST}}}"
LOW_NX="${LOW_NX:-200}"

TEACHER_NX="${TEACHER_NX:-256}"
TEACHER_NV="${TEACHER_NV:-512}"
TEACHER_DT="${TEACHER_DT:-0.01}"
TEACHER_VMIN="${TEACHER_VMIN:--8.0}"
TEACHER_VMAX="${TEACHER_VMAX:-8.0}"
TEACHER_L="${TEACHER_L:-12.566370614359172}"

TRAIN_LINEAR_T="${TRAIN_LINEAR_T:-10.0}"
TRAIN_NONLINEAR_T="${TRAIN_NONLINEAR_T:-10.0}"
TRAIN_LINEAR_EPS="${TRAIN_LINEAR_EPS:-0.01}"
TRAIN_LINEAR_MODES="${TRAIN_LINEAR_MODES:-0.5,1.0,1.5,2.0}"
TRAIN_LINEAR_NUM_SAMPLES="${TRAIN_LINEAR_NUM_SAMPLES:-8}"
TRAIN_LINEAR_SEED="${TRAIN_LINEAR_SEED:-0}"
TRAIN_NONLINEAR_K0="${TRAIN_NONLINEAR_K0:-0.5}"
TRAIN_WEAK_EPS="${TRAIN_WEAK_EPS:-0.03,0.05,0.07,0.1,0.15}"
TRAIN_STRONG_EPS="${TRAIN_STRONG_EPS:-0.15,0.25,0.35,0.5,0.65}"
TRAIN_REGIMES="${TRAIN_REGIMES:-linear_landau,nonlinear_landau_weak,nonlinear_landau_strong}"
TRAIN_VAL_FRACTION="${TRAIN_VAL_FRACTION:-0.2}"

TRAIN_HIDDEN_WIDTH="${TRAIN_HIDDEN_WIDTH:-64}"
TRAIN_RES_BLOCKS="${TRAIN_RES_BLOCKS:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-300}"
TRAIN_LR="${TRAIN_LR:-1e-5}"
TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-0.25}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-5}"
TRAIN_ONLINE_CASE_BATCH_SIZE="${TRAIN_ONLINE_CASE_BATCH_SIZE:-1}"
TRAIN_ROLLOUT_HORIZON="${TRAIN_ROLLOUT_HORIZON:-5}"
TRAIN_ROLLOUT_ANCHOR_SAMPLES="${TRAIN_ROLLOUT_ANCHOR_SAMPLES:-32}"
TRAIN_BACKWARD_WEIGHT="${TRAIN_BACKWARD_WEIGHT:-1.0}"
TRAIN_LOSS_EVAL_BATCH_SIZE="${TRAIN_LOSS_EVAL_BATCH_SIZE:-1}"
TRAIN_SEED="${TRAIN_SEED:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

EVAL_OUTDIR="${EVAL_OUTDIR:-${OUTDIR}/eval}"
EVAL_T="${EVAL_T:-40.0}"
EVAL_EPS="${EVAL_EPS:-0.5}"
EVAL_K0="${EVAL_K0:-0.5}"
EVAL_SNAPSHOT_TIMES="${EVAL_SNAPSHOT_TIMES:-20.0,40.0}"
EVAL_FIELD_K_MAX="${EVAL_FIELD_K_MAX:-}"
EVAL_FIELD_NUM_LOW_MODES="${EVAL_FIELD_NUM_LOW_MODES:-}"
EVAL_PHASE_VMIN="${EVAL_PHASE_VMIN:-0.0}"
EVAL_PHASE_VMAX="${EVAL_PHASE_VMAX:-0.5}"
EVAL_PLOT_VMIN="${EVAL_PLOT_VMIN:--4.0}"
EVAL_PLOT_VMAX="${EVAL_PLOT_VMAX:-4.0}"

mkdir -p "${OUTDIR}"
cd "${REPO_ROOT}"

IFS=',' read -r -a VGRID_VALUES <<< "${VGRID_LIST}"
SUMMARY_JSON="${OUTDIR}/summary.jsonl"
: > "${SUMMARY_JSON}"

for VGRID_RAW in "${VGRID_VALUES[@]}"; do
  VGRID="$(echo "${VGRID_RAW}" | tr -d '[:space:]')"
  if [[ -z "${VGRID}" ]]; then
    continue
  fi
  MODEL_DIR="${OUTDIR}/vgrid${VGRID}"
  CHECKPOINT="${MODEL_DIR}/spline_fem_residual.npz"
  LOSS_PLOT="${MODEL_DIR}/spline_fem_residual.loss.png"
  DATASET_CACHE="${OUTDIR}/spline_fem_lr_teacher_reference_vgrid${VGRID}.npz"

  if [[ "${RUN_TRAIN}" == "0" ]]; then
    if [[ ! -f "${CHECKPOINT}" ]]; then
      echo "RUN_TRAIN=0 requires ${CHECKPOINT}" >&2
      exit 1
    fi
    echo "[spline-fem-sweep] Reusing v-grid=${VGRID} checkpoint at ${CHECKPOINT}"
  else
    echo "[spline-fem-sweep] Training v-grid=${VGRID} (no ladder)"
    "${PYTHON_BIN}" -m model.train.train_spline_fem_rollout \
      --outdir "${MODEL_DIR}" \
      --checkpoint "${CHECKPOINT}" \
      --loss-plot "${LOSS_PLOT}" \
      --dataset-cache "${DATASET_CACHE}" \
      --target-vgrid "${VGRID}" \
      --low-Nx "${LOW_NX}" \
      --hidden-width "${TRAIN_HIDDEN_WIDTH}" \
      --res-blocks "${TRAIN_RES_BLOCKS}" \
      --epochs "${TRAIN_EPOCHS}" \
      --lr "${TRAIN_LR}" \
      --grad-clip "${TRAIN_GRAD_CLIP}" \
      --log-every "${TRAIN_LOG_EVERY}" \
      --steps-per-epoch "${TRAIN_STEPS_PER_EPOCH}" \
      --online-case-batch-size "${TRAIN_ONLINE_CASE_BATCH_SIZE}" \
      --seed "${TRAIN_SEED}" \
      --rollout-horizon "${TRAIN_ROLLOUT_HORIZON}" \
      --rollout-anchor-samples "${TRAIN_ROLLOUT_ANCHOR_SAMPLES}" \
      --backward-weight "${TRAIN_BACKWARD_WEIGHT}" \
      --loss-eval-batch-size "${TRAIN_LOSS_EVAL_BATCH_SIZE}" \
      --regimes "${TRAIN_REGIMES}" \
      --val-fraction "${TRAIN_VAL_FRACTION}" \
      --teacher-Nx "${TEACHER_NX}" \
      --teacher-Nv "${TEACHER_NV}" \
      --teacher-L "${TEACHER_L}" \
      --teacher-vmin "${TEACHER_VMIN}" \
      --teacher-vmax "${TEACHER_VMAX}" \
      --teacher-dt "${TEACHER_DT}" \
      --linear-T "${TRAIN_LINEAR_T}" \
      --linear-eps "${TRAIN_LINEAR_EPS}" \
      --linear-modes "${TRAIN_LINEAR_MODES}" \
      --linear-num-samples "${TRAIN_LINEAR_NUM_SAMPLES}" \
      --linear-seed "${TRAIN_LINEAR_SEED}" \
      --nonlinear-T "${TRAIN_NONLINEAR_T}" \
      --nonlinear-k0 "${TRAIN_NONLINEAR_K0}" \
      --weak-eps "${TRAIN_WEAK_EPS}" \
      --strong-eps "${TRAIN_STRONG_EPS}"
  fi

  if [[ -f "${MODEL_DIR}/summary.json" ]]; then
    "${PYTHON_BIN}" - <<'PY' "${MODEL_DIR}/summary.json" "${SUMMARY_JSON}"
import json
import sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    payload = json.load(f)
with open(dst, "a") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")
PY
  fi
done

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[spline-fem-sweep] Evaluating no-correction and learned-correction rollouts"
  EVAL_ARGS=(
    -m model.eval_spline_fem_rollout
    --checkpoint-dir "${OUTDIR}"
    --outdir "${EVAL_OUTDIR}"
    --vgrid-list "${VGRID_LIST}"
    --low-Nx "${LOW_NX}"
    --teacher-Nx "${TEACHER_NX}"
    --teacher-Nv "${TEACHER_NV}"
    --teacher-L "${TEACHER_L}"
    --teacher-vmin "${TEACHER_VMIN}"
    --teacher-vmax "${TEACHER_VMAX}"
    --dt "${TEACHER_DT}"
    --T "${EVAL_T}"
    --eps "${EVAL_EPS}"
    --k0 "${EVAL_K0}"
    --snapshot-times "${EVAL_SNAPSHOT_TIMES}"
    --phase-vmin "${EVAL_PHASE_VMIN}"
    --phase-vmax "${EVAL_PHASE_VMAX}"
    --plot-vmin "${EVAL_PLOT_VMIN}"
    --plot-vmax "${EVAL_PLOT_VMAX}"
  )
  if [[ -n "${EVAL_FIELD_K_MAX}" ]]; then
    EVAL_ARGS+=(--field-k-max "${EVAL_FIELD_K_MAX}")
  fi
  if [[ -n "${EVAL_FIELD_NUM_LOW_MODES}" ]]; then
    EVAL_ARGS+=(--field-num-low-modes "${EVAL_FIELD_NUM_LOW_MODES}")
  fi
  "${PYTHON_BIN}" "${EVAL_ARGS[@]}"
fi

cat <<EOF

Done.

Artifacts:
  mode:          spline_fem_online_rollout
  outdir:        ${OUTDIR}
  summary jsonl: ${SUMMARY_JSON}
  eval outdir:   ${EVAL_OUTDIR}
  v-grid list:   ${VGRID_LIST}

Defaults:
  no ladder:     one independent spline/FEM residual per v-grid target
  low Nx:        ${LOW_NX}
  teacher grid:  Nx=${TEACHER_NX}, Nv=${TEACHER_NV}
  train dt:      ${TEACHER_DT}
  train T:       linear=${TRAIN_LINEAR_T}, nonlinear=${TRAIN_NONLINEAR_T}
  eval T:        ${EVAL_T}
  rollout horiz: ${TRAIN_ROLLOUT_HORIZON}
  anchors:       ${TRAIN_ROLLOUT_ANCHOR_SAMPLES}
  backward wt:   ${TRAIN_BACKWARD_WEIGHT}
  eval plots:    RUN_EVAL=${RUN_EVAL}
EOF
