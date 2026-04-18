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

OUTDIR="${1:-${REPO_ROOT}/out_bench/nv_sweep_single_qloss}"
NV_LIST="${NV_LIST:-8,64,256,300,512}"
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

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTDIR}/models}"
TRAIN_LADDER_LEVELS=5
TRAIN_NM="${TRAIN_NM:-6}"
TRAIN_HIDDEN_WIDTH="${TRAIN_HIDDEN_WIDTH:-128}"
TRAIN_RES_BLOCKS="${TRAIN_RES_BLOCKS:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-300}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-1.0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-0}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-0}"
TRAIN_SEED="${TRAIN_SEED:-0}"
TRAIN_N_LOW="${TRAIN_N_LOW:-2}"
TRAIN_VAL_FRACTION="${TRAIN_VAL_FRACTION:-0.2}"
TRAIN_REGIMES="${TRAIN_REGIMES:-linear_landau,nonlinear_landau_weak,nonlinear_landau_strong}"
TRAIN_OBJECTIVE="${TRAIN_OBJECTIVE:-q_only}"
TRAIN_CONTEXT_MODE="${TRAIN_CONTEXT_MODE:-none}"
TRAIN_ROLLOUT_HORIZON="${TRAIN_ROLLOUT_HORIZON:-0}"

TRAIN_TEACHER_NX="${TRAIN_TEACHER_NX:-${TEACHER_NX}}"
TRAIN_TEACHER_NV="${TRAIN_TEACHER_NV:-${TEACHER_NV}}"
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

if [[ "${TRAIN_OBJECTIVE}" != "q_only" ]]; then
  echo "run_nv_sweep_single_qloss.sh only supports TRAIN_OBJECTIVE=q_only; got '${TRAIN_OBJECTIVE}'." >&2
  exit 1
fi
if [[ "${TRAIN_CONTEXT_MODE}" != "none" ]]; then
  echo "run_nv_sweep_single_qloss.sh only supports TRAIN_CONTEXT_MODE=none; got '${TRAIN_CONTEXT_MODE}'." >&2
  exit 1
fi
if [[ "${TRAIN_ROLLOUT_HORIZON}" != "0" ]]; then
  echo "run_nv_sweep_single_qloss.sh only supports TRAIN_ROLLOUT_HORIZON=0; got '${TRAIN_ROLLOUT_HORIZON}'." >&2
  exit 1
fi

ladder_csv_for_target() {
  local target="$1"
  "${PYTHON_BIN}" - <<'PY' "${target}" "${TRAIN_NM}" "${TRAIN_LADDER_LEVELS}"
import math
import sys

target = int(sys.argv[1])
nm = int(sys.argv[2])
levels = int(sys.argv[3])
if levels <= 0:
    raise SystemExit("TRAIN_LADDER_LEVELS must be positive")
if target < nm:
    raise SystemExit(f"target Nv={target} must be at least TRAIN_NM={nm}")
if target == nm:
    ladder = [nm]
else:
    ladder = []
    for idx in range(levels):
        t = 0.0 if levels == 1 else idx / float(levels - 1)
        value = math.exp(math.log(float(nm)) + t * (math.log(float(target)) - math.log(float(nm))))
        ladder.append(int(round(value)))
    ladder[0] = nm
    ladder[-1] = target
    dedup = []
    for value in ladder:
        value = max(nm, min(target, int(value)))
        if not dedup or dedup[-1] != value:
            dedup.append(value)
    ladder = dedup
print(",".join(str(value) for value in ladder))
PY
}

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
  echo "[nv-sweep-single-qloss] [1/3] Using target-specific q_only Nv ladders and per-model dataset caches"
  for idx in "${!NV_VALUES[@]}"; do
    NV_RAW="${NV_VALUES[idx]}"
    NV="$(echo "${NV_RAW}" | tr -d '[:space:]')"
    if [[ -z "${NV}" ]]; then
      continue
    fi
    TRAIN_LADDER_CSV="$(ladder_csv_for_target "${NV}")"
    MODEL_DIR="${CHECKPOINT_ROOT}/nv${NV}"
    CHECKPOINT_NV="${MODEL_DIR}/interface_closure.npz"
    LOSS_PLOT_NV="${MODEL_DIR}/interface_closure.loss.png"
    DATASET_CACHE_NV="${MODEL_DIR}/interface_closure_dataset.npz"
    mkdir -p "${MODEL_DIR}"

    TEACHER_PROJ_NV_TARGET="${TRAIN_TEACHER_PROJ_NV:-$(( NV + 1 ))}"
    TRAIN_ARGS=(
      --checkpoint "${CHECKPOINT_NV}"
      --dataset-cache "${DATASET_CACHE_NV}"
      --loss-plot "${LOSS_PLOT_NV}"
      --training-mode offline_rollout
      --Nv-targets "${TRAIN_LADDER_CSV}"
      --Nm "${TRAIN_NM}"
      --hidden-width "${TRAIN_HIDDEN_WIDTH}"
      --res-blocks "${TRAIN_RES_BLOCKS}"
      --epochs "${TRAIN_EPOCHS}"
      --lr "${TRAIN_LR}"
      --grad-clip "${TRAIN_GRAD_CLIP}"
      --log-every "${TRAIN_LOG_EVERY}"
      --batch-size "${TRAIN_BATCH_SIZE}"
      --steps-per-epoch "${TRAIN_STEPS_PER_EPOCH}"
      --seed "${TRAIN_SEED}"
      --n-low "${TRAIN_N_LOW}"
      --val-fraction "${TRAIN_VAL_FRACTION}"
      --train-objective q_only
      --context-mode none
      --rollout-horizon 0
      --regimes "${TRAIN_REGIMES}"
      --teacher-Nx "${TRAIN_TEACHER_NX}"
      --teacher-Nv "${TRAIN_TEACHER_NV}"
      --teacher-dt "${TRAIN_TEACHER_DT}"
      --teacher-vmin "${TRAIN_TEACHER_VMIN}"
      --teacher-vmax "${TRAIN_TEACHER_VMAX}"
      --teacher-proj-Nv "${TEACHER_PROJ_NV_TARGET}"
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

    echo "[nv-sweep-single-qloss] [2/3] Training closure $((idx + 1))/${TOTAL_NV} for Nv=${NV} with Nv-targets=${TRAIN_LADDER_CSV}"
    "${PYTHON_BIN}" benchmarks/fh_ml_tail_closure_train_jax.py "${TRAIN_ARGS[@]}"
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
  echo "[nv-sweep-single-qloss] [1/3] Skipping model training because RUN_TRAIN=${RUN_TRAIN}"
  echo "[nv-sweep-single-qloss] [2/3] Reusing existing checkpoints in ${CHECKPOINT_ROOT}"
fi

ARGS+=(--checkpoint-dir "${CHECKPOINT_ROOT}")
echo "[nv-sweep-single-qloss] [3/3] Running nonlinear Nv sweep"
"${PYTHON_BIN}" benchmarks/eval_nv_sweep.py "${ARGS[@]}"

cat <<EOF

Done.

Artifacts:
  mode:           offline_rollout_target_aware_single_qloss
  checkpoint dir: ${CHECKPOINT_ROOT}
  dataset caches: ${CHECKPOINT_ROOT}/nv*/interface_closure_dataset.npz
  summary:        ${OUTDIR}/summary.json
  metric 1:       ${OUTDIR}/nv_sweep_metric1.png
  metric 2:       ${OUTDIR}/nv_sweep_metric2.png
  phase space:    ${OUTDIR}/fig10_learned_vs_nonlocal_nv_sweep_phase_space.png
  phase payload:  ${OUTDIR}/nv_sweep_phase_space_payload.npz

Defaults:
  ladder mode:    target-specific log-spaced q_only
  ladder levels:  ${TRAIN_LADDER_LEVELS}
  train Nm:       ${TRAIN_NM}
  batch size:     ${TRAIN_BATCH_SIZE}
  steps/epoch:    ${TRAIN_STEPS_PER_EPOCH}
  objective:      q_only
  context:        none
  Nv list:        ${NV_LIST}
  nonlinear case: Nx=${NX}, T=${T_FINAL}, dt=${DT}, eps=${EPS}, k0=${K0}
EOF
