#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON:-python}"
fi

OUTDIR="${1:-${REPO_ROOT}/out_bench/nv_sweep}"
NV_LIST="${NV_LIST:-8,64,256,300,512}"
SWEEP_TRAIN_MODE="${SWEEP_TRAIN_MODE:-per_nv}"
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

TEACHER_NX="${TEACHER_NX:-256}"
TEACHER_NV="${TEACHER_NV:-512}"
TEACHER_DT="${TEACHER_DT:-0.01}"
TEACHER_VMIN="${TEACHER_VMIN:--8.0}"
TEACHER_VMAX="${TEACHER_VMAX:-8.0}"

FIELD_NUM_LOW_MODES="${FIELD_NUM_LOW_MODES:-}"
FIELD_K_MAX="${FIELD_K_MAX:-}"
RUN_TRAIN="${RUN_TRAIN:-1}"

CHECKPOINT="${CHECKPOINT:-${OUTDIR}/interface_closure.npz}"
DATASET_CACHE="${DATASET_CACHE:-${OUTDIR}/interface_closure_dataset.npz}"
LOSS_PLOT="${LOSS_PLOT:-${OUTDIR}/interface_closure.loss.png}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTDIR}/models}"
SHARED_DATASET_CACHE="${SHARED_DATASET_CACHE:-${OUTDIR}/shared_interface_closure_dataset.npz}"
TRAIN_NV_TARGETS="${TRAIN_NV_TARGETS:-6,8,10,12,20,40,80,160,300,512}"
TRAIN_NM="${TRAIN_NM:-6}"
TRAIN_HIDDEN_WIDTH="${TRAIN_HIDDEN_WIDTH:-128}"
TRAIN_RES_BLOCKS="${TRAIN_RES_BLOCKS:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-300}"
TRAIN_LR="${TRAIN_LR:-0.001}"
TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-1.0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-262144}"
TRAIN_ROLLOUT_BATCH_SIZE="${TRAIN_ROLLOUT_BATCH_SIZE:-32}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-8}"
TRAIN_SEED="${TRAIN_SEED:-0}"
TRAIN_N_LOW="${TRAIN_N_LOW:-2}"
TRAIN_VAL_FRACTION="${TRAIN_VAL_FRACTION:-0.2}"
TRAIN_REGIMES="${TRAIN_REGIMES:-linear_landau,nonlinear_landau_weak,nonlinear_landau_strong}"
TRAIN_OBJECTIVE="${TRAIN_OBJECTIVE:-stability_aware}"
TRAIN_CONTEXT_MODE="${TRAIN_CONTEXT_MODE:-lag1_delta}"
TRAIN_ROLLOUT_HORIZON="${TRAIN_ROLLOUT_HORIZON:-2}"
TRAIN_TAIL_START_FRACTION="${TRAIN_TAIL_START_FRACTION:-0.6666666666666666}"
TRAIN_LAMBDA_Q="${TRAIN_LAMBDA_Q:-1.0}"
TRAIN_LAMBDA_E="${TRAIN_LAMBDA_E:-0.5}"
TRAIN_LAMBDA_TAIL="${TRAIN_LAMBDA_TAIL:-0.05}"
TRAIN_LAMBDA_REG="${TRAIN_LAMBDA_REG:-1e-6}"
TRAIN_ROLLOUT_DEALIAS_23="${TRAIN_ROLLOUT_DEALIAS_23:-1}"

TRAIN_TEACHER_NX="${TRAIN_TEACHER_NX:-${TEACHER_NX}}"
TRAIN_TEACHER_DT="${TRAIN_TEACHER_DT:-${TEACHER_DT}}"
TRAIN_TEACHER_VMIN="${TRAIN_TEACHER_VMIN:-${TEACHER_VMIN}}"
TRAIN_TEACHER_VMAX="${TRAIN_TEACHER_VMAX:-${TEACHER_VMAX}}"
TRAIN_TEACHER_PROJ_NV="${TRAIN_TEACHER_PROJ_NV:-}"
TRAIN_LINEAR_T="${TRAIN_LINEAR_T:-20.0}"
TRAIN_LINEAR_EPS="${TRAIN_LINEAR_EPS:-0.01}"
TRAIN_LINEAR_MODES="${TRAIN_LINEAR_MODES:-0.5,1.0,1.5,2.0}"
TRAIN_LINEAR_NUM_SAMPLES="${TRAIN_LINEAR_NUM_SAMPLES:-8}"
TRAIN_LINEAR_SEED="${TRAIN_LINEAR_SEED:-0}"
TRAIN_LINEAR_HISTORY_STRIDE="${TRAIN_LINEAR_HISTORY_STRIDE:-2}"
TRAIN_NONLINEAR_T="${TRAIN_NONLINEAR_T:-20.0}"
TRAIN_NONLINEAR_K0="${TRAIN_NONLINEAR_K0:-${K0}}"
TRAIN_NONLINEAR_HISTORY_STRIDE="${TRAIN_NONLINEAR_HISTORY_STRIDE:-20}"
TRAIN_WEAK_EPS="${TRAIN_WEAK_EPS:-0.05,0.1}"
TRAIN_STRONG_EPS="${TRAIN_STRONG_EPS:-0.25,0.5}"

mkdir -p "${OUTDIR}"
cd "${REPO_ROOT}"

case "${SWEEP_TRAIN_MODE}" in
  shared)
    TRAIN_TARGET_LIST="${TRAIN_NV_TARGETS}"
    ;;
  per_nv)
    TRAIN_TARGET_LIST="${NV_LIST}"
    ;;
  *)
    echo "Unknown SWEEP_TRAIN_MODE='${SWEEP_TRAIN_MODE}'. Expected 'shared' or 'per_nv'." >&2
    exit 1
    ;;
esac

TRAIN_MAX_NV="$("${PYTHON_BIN}" - <<'PY' "${TRAIN_TARGET_LIST}"
import sys
vals = [int(part.strip()) for part in sys.argv[1].split(",") if part.strip()]
if not vals:
    raise SystemExit("training target list must contain at least one Nv value")
print(max(vals))
PY
)"
TRAIN_TEACHER_NV="${TRAIN_TEACHER_NV:-${TEACHER_NV}}"
if [[ -z "${TRAIN_TEACHER_PROJ_NV}" ]]; then
  TRAIN_TEACHER_PROJ_NV="$(( TRAIN_MAX_NV + 1 ))"
fi
PER_NV_DATASET_CACHE_MODE="shared"
if [[ "${SWEEP_TRAIN_MODE}" == "per_nv" && "${TRAIN_OBJECTIVE}" == "stability_aware" && "${TRAIN_ROLLOUT_HORIZON}" != "0" ]]; then
  PER_NV_DATASET_CACHE_MODE="per_nv"
fi
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

if [[ "${SWEEP_TRAIN_MODE}" == "shared" ]]; then
  if [[ "${RUN_TRAIN}" != "0" ]]; then
    echo "[nv-sweep] [1/2] Training one shared closure for deployment sweep"
    TRAIN_ARGS=(
      --checkpoint "${CHECKPOINT}"
      --dataset-cache "${DATASET_CACHE}"
      --loss-plot "${LOSS_PLOT}"
      --Nv-targets "${TRAIN_NV_TARGETS}"
      --Nm "${TRAIN_NM}"
      --hidden-width "${TRAIN_HIDDEN_WIDTH}"
      --res-blocks "${TRAIN_RES_BLOCKS}"
      --epochs "${TRAIN_EPOCHS}"
      --lr "${TRAIN_LR}"
      --grad-clip "${TRAIN_GRAD_CLIP}"
      --log-every "${TRAIN_LOG_EVERY}"
      --batch-size "${TRAIN_BATCH_SIZE}"
      --rollout-batch-size "${TRAIN_ROLLOUT_BATCH_SIZE}"
      --steps-per-epoch "${TRAIN_STEPS_PER_EPOCH}"
      --seed "${TRAIN_SEED}"
      --n-low "${TRAIN_N_LOW}"
      --val-fraction "${TRAIN_VAL_FRACTION}"
      --train-objective "${TRAIN_OBJECTIVE}"
      --context-mode "${TRAIN_CONTEXT_MODE}"
      --rollout-horizon "${TRAIN_ROLLOUT_HORIZON}"
      --tail-start-fraction "${TRAIN_TAIL_START_FRACTION}"
      --lambda-q "${TRAIN_LAMBDA_Q}"
      --lambda-E "${TRAIN_LAMBDA_E}"
      --lambda-tail "${TRAIN_LAMBDA_TAIL}"
      --lambda-reg "${TRAIN_LAMBDA_REG}"
      --regimes "${TRAIN_REGIMES}"
      --teacher-Nx "${TRAIN_TEACHER_NX}"
      --teacher-Nv "${TRAIN_TEACHER_NV}"
      --teacher-dt "${TRAIN_TEACHER_DT}"
      --teacher-vmin "${TRAIN_TEACHER_VMIN}"
      --teacher-vmax "${TRAIN_TEACHER_VMAX}"
      --teacher-proj-Nv "${TRAIN_TEACHER_PROJ_NV}"
      --linear-T "${TRAIN_LINEAR_T}"
      --linear-eps "${TRAIN_LINEAR_EPS}"
      --linear-modes "${TRAIN_LINEAR_MODES}"
      --linear-num-samples "${TRAIN_LINEAR_NUM_SAMPLES}"
      --linear-seed "${TRAIN_LINEAR_SEED}"
      --linear-history-stride "${TRAIN_LINEAR_HISTORY_STRIDE}"
      --nonlinear-T "${TRAIN_NONLINEAR_T}"
      --nonlinear-k0 "${TRAIN_NONLINEAR_K0}"
      --nonlinear-history-stride "${TRAIN_NONLINEAR_HISTORY_STRIDE}"
      --weak-eps "${TRAIN_WEAK_EPS}"
      --strong-eps "${TRAIN_STRONG_EPS}"
    )
    if [[ "${TRAIN_ROLLOUT_DEALIAS_23}" != "0" ]]; then
      TRAIN_ARGS+=(--rollout-dealias-23)
    fi
    "${PYTHON_BIN}" benchmarks/fh_ml_tail_closure_train_jax.py "${TRAIN_ARGS[@]}"
  else
    if [[ ! -f "${CHECKPOINT}" ]]; then
      echo "RUN_TRAIN=0 requires an existing shared checkpoint at ${CHECKPOINT}" >&2
      exit 1
    fi
    echo "[nv-sweep] [1/2] Reusing shared checkpoint ${CHECKPOINT}"
  fi
  ARGS+=(--checkpoint "${CHECKPOINT}")
  echo "[nv-sweep] [2/2] Running nonlinear Nv sweep"
else
  BUILD_DATASET_ARGS=()
  if [[ "${PER_NV_DATASET_CACHE_MODE}" == "shared" ]]; then
    BUILD_DATASET_ARGS=(
      --dataset-cache "${SHARED_DATASET_CACHE}"
      --build-dataset-only
      --per-target-projection-orders
      --Nv-targets "${NV_LIST}"
      --Nm "${TRAIN_NM}"
      --context-mode "${TRAIN_CONTEXT_MODE}"
      --rollout-horizon "${TRAIN_ROLLOUT_HORIZON}"
      --teacher-Nx "${TRAIN_TEACHER_NX}"
      --teacher-Nv "${TRAIN_TEACHER_NV}"
      --teacher-dt "${TRAIN_TEACHER_DT}"
      --teacher-vmin "${TRAIN_TEACHER_VMIN}"
      --teacher-vmax "${TRAIN_TEACHER_VMAX}"
      --teacher-proj-Nv "${TRAIN_TEACHER_PROJ_NV}"
      --n-low "${TRAIN_N_LOW}"
      --val-fraction "${TRAIN_VAL_FRACTION}"
      --regimes "${TRAIN_REGIMES}"
      --linear-T "${TRAIN_LINEAR_T}"
      --linear-eps "${TRAIN_LINEAR_EPS}"
      --linear-modes "${TRAIN_LINEAR_MODES}"
      --linear-num-samples "${TRAIN_LINEAR_NUM_SAMPLES}"
      --linear-seed "${TRAIN_LINEAR_SEED}"
      --linear-history-stride "${TRAIN_LINEAR_HISTORY_STRIDE}"
      --nonlinear-T "${TRAIN_NONLINEAR_T}"
      --nonlinear-k0 "${TRAIN_NONLINEAR_K0}"
      --nonlinear-history-stride "${TRAIN_NONLINEAR_HISTORY_STRIDE}"
      --weak-eps "${TRAIN_WEAK_EPS}"
      --strong-eps "${TRAIN_STRONG_EPS}"
    )
  fi

  if [[ "${RUN_TRAIN}" != "0" ]]; then
    if [[ "${PER_NV_DATASET_CACHE_MODE}" == "shared" ]]; then
      echo "[nv-sweep] [1/3] Building shared extracted dataset for Nv targets ${NV_LIST}"
      "${PYTHON_BIN}" benchmarks/fh_ml_tail_closure_train_jax.py "${BUILD_DATASET_ARGS[@]}"
    else
      echo "[nv-sweep] [1/3] Using per-Nv dataset caches to avoid a giant shared rollout cache"
    fi

    for idx in "${!NV_VALUES[@]}"; do
      NV_RAW="${NV_VALUES[idx]}"
      NV="$(echo "${NV_RAW}" | tr -d '[:space:]')"
      if [[ -z "${NV}" ]]; then
        continue
      fi
      MODEL_DIR="${CHECKPOINT_ROOT}/nv${NV}"
      CHECKPOINT_NV="${MODEL_DIR}/interface_closure.npz"
      LOSS_PLOT_NV="${MODEL_DIR}/interface_closure.loss.png"
      DATASET_CACHE_NV="${MODEL_DIR}/interface_closure_dataset.npz"
      mkdir -p "${MODEL_DIR}"

      TRAIN_DATASET_CACHE="${SHARED_DATASET_CACHE}"
      ALLOW_CACHE_SUPERSET=1
      if [[ "${PER_NV_DATASET_CACHE_MODE}" == "per_nv" ]]; then
        TRAIN_DATASET_CACHE="${DATASET_CACHE_NV}"
        ALLOW_CACHE_SUPERSET=0
      fi

      TRAIN_ARGS=(
        --checkpoint "${CHECKPOINT_NV}"
        --dataset-cache "${TRAIN_DATASET_CACHE}"
        --loss-plot "${LOSS_PLOT_NV}"
        --Nv-targets "${NV}"
        --per-target-projection-orders
        --Nm "${TRAIN_NM}"
        --hidden-width "${TRAIN_HIDDEN_WIDTH}"
        --res-blocks "${TRAIN_RES_BLOCKS}"
        --epochs "${TRAIN_EPOCHS}"
        --lr "${TRAIN_LR}"
        --grad-clip "${TRAIN_GRAD_CLIP}"
        --log-every "${TRAIN_LOG_EVERY}"
        --batch-size "${TRAIN_BATCH_SIZE}"
        --rollout-batch-size "${TRAIN_ROLLOUT_BATCH_SIZE}"
        --steps-per-epoch "${TRAIN_STEPS_PER_EPOCH}"
        --seed "${TRAIN_SEED}"
        --n-low "${TRAIN_N_LOW}"
        --val-fraction "${TRAIN_VAL_FRACTION}"
        --train-objective "${TRAIN_OBJECTIVE}"
        --context-mode "${TRAIN_CONTEXT_MODE}"
        --rollout-horizon "${TRAIN_ROLLOUT_HORIZON}"
        --tail-start-fraction "${TRAIN_TAIL_START_FRACTION}"
        --lambda-q "${TRAIN_LAMBDA_Q}"
        --lambda-E "${TRAIN_LAMBDA_E}"
        --lambda-tail "${TRAIN_LAMBDA_TAIL}"
        --lambda-reg "${TRAIN_LAMBDA_REG}"
        --regimes "${TRAIN_REGIMES}"
        --teacher-Nx "${TRAIN_TEACHER_NX}"
        --teacher-Nv "${TRAIN_TEACHER_NV}"
        --teacher-dt "${TRAIN_TEACHER_DT}"
        --teacher-vmin "${TRAIN_TEACHER_VMIN}"
        --teacher-vmax "${TRAIN_TEACHER_VMAX}"
        --teacher-proj-Nv "$(( NV + 1 ))"
        --linear-T "${TRAIN_LINEAR_T}"
        --linear-eps "${TRAIN_LINEAR_EPS}"
        --linear-modes "${TRAIN_LINEAR_MODES}"
        --linear-num-samples "${TRAIN_LINEAR_NUM_SAMPLES}"
        --linear-seed "${TRAIN_LINEAR_SEED}"
        --linear-history-stride "${TRAIN_LINEAR_HISTORY_STRIDE}"
        --nonlinear-T "${TRAIN_NONLINEAR_T}"
        --nonlinear-k0 "${TRAIN_NONLINEAR_K0}"
        --nonlinear-history-stride "${TRAIN_NONLINEAR_HISTORY_STRIDE}"
        --weak-eps "${TRAIN_WEAK_EPS}"
        --strong-eps "${TRAIN_STRONG_EPS}"
      )
      if [[ "${ALLOW_CACHE_SUPERSET}" != "0" ]]; then
        TRAIN_ARGS+=(--allow-dataset-cache-nv-superset)
      fi
      if [[ "${TRAIN_ROLLOUT_DEALIAS_23}" != "0" ]]; then
        TRAIN_ARGS+=(--rollout-dealias-23)
      fi

      echo "[nv-sweep] [2/3] Training closure $((idx + 1))/${TOTAL_NV} for Nv=${NV}"
      "${PYTHON_BIN}" benchmarks/fh_ml_tail_closure_train_jax.py "${TRAIN_ARGS[@]}"
    done
  else
    if [[ "${PER_NV_DATASET_CACHE_MODE}" == "shared" && -f "${SHARED_DATASET_CACHE}" ]]; then
      "${PYTHON_BIN}" - <<'PY' "${SHARED_DATASET_CACHE}" "${NV_LIST}" "${TRAIN_NONLINEAR_T}" "${TRAIN_TEACHER_NV}" "${TRAIN_CONTEXT_MODE}" "${TRAIN_ROLLOUT_HORIZON}"
import sys
import numpy as np
from pathlib import Path

cache = Path(sys.argv[1])
expected_nv = tuple(int(part.strip()) for part in sys.argv[2].split(",") if part.strip())
expected_nonlinear_t = float(sys.argv[3])
expected_teacher_nv = int(sys.argv[4])
expected_context_mode = str(sys.argv[5])
expected_rollout_horizon = int(sys.argv[6])
with np.load(cache) as data:
    cached_nv = tuple(int(v) for v in np.asarray(data["Nv_targets"], dtype=np.int32).reshape(-1))
    cached_nonlinear_t = float(np.asarray(data["nonlinear_T"], dtype=np.float64).reshape(-1)[0])
    cached_teacher_nv = int(np.asarray(data["teacher_Nv"], dtype=np.int32).reshape(-1)[0])
    cached_context_mode = (
        str(np.asarray(data["context_mode"], dtype=np.str_).reshape(-1)[0])
        if "context_mode" in data.files and data["context_mode"].size
        else "none"
    )
    cached_rollout_horizon = (
        int(np.asarray(data["rollout_horizon"], dtype=np.int32).reshape(-1)[0])
        if "rollout_horizon" in data.files and data["rollout_horizon"].size
        else 0
    )
    cached_projection_mode = (
        str(np.asarray(data["projection_mode"], dtype=np.str_).reshape(-1)[0])
        if "projection_mode" in data.files and data["projection_mode"].size
        else "shared_max"
    )
if (
    cached_nv != expected_nv
    or abs(cached_nonlinear_t - expected_nonlinear_t) > 1e-12
    or cached_teacher_nv != expected_teacher_nv
    or cached_context_mode != expected_context_mode
    or cached_rollout_horizon != expected_rollout_horizon
    or cached_projection_mode != "per_target"
):
    raise SystemExit(
        f"RUN_TRAIN=0 cannot reuse stale sweep cache {cache}. "
        "Expected "
        f"Nv_targets={expected_nv}, nonlinear_T={expected_nonlinear_t:g}, "
        f"teacher_Nv={expected_teacher_nv}, context_mode='{expected_context_mode}', "
        f"rollout_horizon={expected_rollout_horizon}, projection_mode='per_target'; "
        "found "
        f"Nv_targets={cached_nv}, nonlinear_T={cached_nonlinear_t:g}, "
        f"teacher_Nv={cached_teacher_nv}, context_mode='{cached_context_mode}', "
        f"rollout_horizon={cached_rollout_horizon}, projection_mode='{cached_projection_mode}'. "
        "Rerun without RUN_TRAIN=0 to rebuild the sweep models."
    )
PY
    fi
    echo "[nv-sweep] [1/3] Skipping dataset build and model training because RUN_TRAIN=${RUN_TRAIN}"
    echo "[nv-sweep] [2/3] Reusing existing checkpoints in ${CHECKPOINT_ROOT}"
  fi
  ARGS+=(--checkpoint-dir "${CHECKPOINT_ROOT}")
  echo "[nv-sweep] [3/3] Running nonlinear Nv sweep"
fi
"${PYTHON_BIN}" benchmarks/eval_nv_sweep.py "${ARGS[@]}"

if [[ "${SWEEP_TRAIN_MODE}" == "per_nv" ]]; then
  PRIMARY_CHECKPOINT_BLOCK="  checkpoint dir: ${CHECKPOINT_ROOT}"
  if [[ "${PER_NV_DATASET_CACHE_MODE}" == "shared" ]]; then
    PRIMARY_DATASET_BLOCK="  shared cache:   ${SHARED_DATASET_CACHE}"
  else
    PRIMARY_DATASET_BLOCK="  dataset caches: ${CHECKPOINT_ROOT}/nv*/interface_closure_dataset.npz"
  fi
else
  PRIMARY_CHECKPOINT_BLOCK=$'  checkpoint:     '"${CHECKPOINT}"$'\n  checkpoint dir: '"${CHECKPOINT_ROOT}"
  PRIMARY_DATASET_BLOCK=$'  dataset cache:  '"${DATASET_CACHE}"$'\n  shared cache:   '"${SHARED_DATASET_CACHE}"
fi

cat <<EOF

Done.

Artifacts:
  mode:           ${SWEEP_TRAIN_MODE}
${PRIMARY_CHECKPOINT_BLOCK}
${PRIMARY_DATASET_BLOCK}
  summary:        ${OUTDIR}/summary.json
  metric 1:       ${OUTDIR}/nv_sweep_metric1.png
  metric 2:       ${OUTDIR}/nv_sweep_metric2.png
  phase space:    ${OUTDIR}/fig10_learned_vs_nonlocal_nv_sweep_phase_space.png
  phase payload:  ${OUTDIR}/nv_sweep_phase_space_payload.npz

Defaults:
  train mode:     ${SWEEP_TRAIN_MODE}
  train Nv list:  ${TRAIN_TARGET_LIST}
  batch size:     ${TRAIN_BATCH_SIZE}
  rollout batch:  ${TRAIN_ROLLOUT_BATCH_SIZE}
  steps/epoch:    ${TRAIN_STEPS_PER_EPOCH}
  objective:      ${TRAIN_OBJECTIVE}
  context:        ${TRAIN_CONTEXT_MODE}
  Nv list:        ${NV_LIST}
  nonlinear case: Nx=${NX}, T=${T_FINAL}, dt=${DT}, eps=${EPS}, k0=${K0}
EOF
