#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_OUTDIR="${REPO_ROOT}/out_bench/nv_sweep_online_rollout_q_hybrid_fixed_ratio"
DEFAULT_NV_LIST="8,64,256,300,512"
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON:-python}"
fi

OUTDIR="${1:-${DEFAULT_OUTDIR}}"
NV_LIST="${NV_LIST:-${DEFAULT_NV_LIST}}"
ONLINE_REFERENCE_CACHE="${ONLINE_REFERENCE_CACHE:-${OUTDIR}/online_reference_dataset.npz}"
NX="${NX:-200}"
DT="${DT:-0.005}"
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
TEACHER_DT="${TEACHER_DT:-0.005}"
TEACHER_VMIN="${TEACHER_VMIN:--8.0}"
TEACHER_VMAX="${TEACHER_VMAX:-8.0}"

FIELD_NUM_LOW_MODES="${FIELD_NUM_LOW_MODES:-}"
FIELD_K_MAX="${FIELD_K_MAX:-}"
RUN_TRAIN="${RUN_TRAIN:-1}"
TRAIN_PARALLEL_JOBS="${TRAIN_PARALLEL_JOBS:-1}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTDIR}/models}"
TRAIN_FIXED_RATIO="${TRAIN_FIXED_RATIO:-1.8}"
TRAIN_NM="${TRAIN_NM:-6}"
TRAIN_HIDDEN_WIDTH="${TRAIN_HIDDEN_WIDTH:-128}"
TRAIN_RES_BLOCKS="${TRAIN_RES_BLOCKS:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-300}"
TRAIN_LR="${TRAIN_LR:-5e-5}"
TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-0.25}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-0}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-1}"
TRAIN_SEED="${TRAIN_SEED:-0}"
TRAIN_N_LOW="${TRAIN_N_LOW:-2}"
TRAIN_VAL_FRACTION="${TRAIN_VAL_FRACTION:-0.2}"
TRAIN_REGIMES="${TRAIN_REGIMES:-linear_landau,nonlinear_landau_weak,nonlinear_landau_strong}"
TRAIN_CONTEXT_MODE="${TRAIN_CONTEXT_MODE:-none}"
TRAIN_TAIL_START_FRACTION="${TRAIN_TAIL_START_FRACTION:-0.6666666666666666}"
TRAIN_LAMBDA_Q="${TRAIN_LAMBDA_Q:-1.0}"
TRAIN_LAMBDA_E="${TRAIN_LAMBDA_E:-0.5}"
TRAIN_LAMBDA_DIST="${TRAIN_LAMBDA_DIST:-1.0}"
TRAIN_LAMBDA_TAIL="${TRAIN_LAMBDA_TAIL:-0.005}"
TRAIN_LAMBDA_NEG="${TRAIN_LAMBDA_NEG:-0.05}"
TRAIN_LAMBDA_REG="${TRAIN_LAMBDA_REG:-1e-6}"
TRAIN_ROLLOUT_DEALIAS_23="${TRAIN_ROLLOUT_DEALIAS_23:-1}"
TRAIN_ONLINE_LOSS_BACKEND="${TRAIN_ONLINE_LOSS_BACKEND:-field_distribution_v1}"
TRAIN_ONLINE_V_PROBES="${TRAIN_ONLINE_V_PROBES:-128}"
TRAIN_ONLINE_CASE_BATCH_SIZE="${TRAIN_ONLINE_CASE_BATCH_SIZE:-2}"

TRAIN_TEACHER_NX="${TRAIN_TEACHER_NX:-${TEACHER_NX}}"
TRAIN_TEACHER_NV="${TRAIN_TEACHER_NV:-${TEACHER_NV}}"
TRAIN_TEACHER_DT="${TRAIN_TEACHER_DT:-0.01}"
TRAIN_TEACHER_VMIN="${TRAIN_TEACHER_VMIN:-${TEACHER_VMIN}}"
TRAIN_TEACHER_VMAX="${TRAIN_TEACHER_VMAX:-${TEACHER_VMAX}}"
TRAIN_LINEAR_T="${TRAIN_LINEAR_T:-10.0}"
TRAIN_LINEAR_EPS="${TRAIN_LINEAR_EPS:-0.01}"
TRAIN_LINEAR_MODES="${TRAIN_LINEAR_MODES:-0.5,1.0,1.5,2.0}"
TRAIN_LINEAR_NUM_SAMPLES="${TRAIN_LINEAR_NUM_SAMPLES:-8}"
TRAIN_LINEAR_SEED="${TRAIN_LINEAR_SEED:-0}"
TRAIN_LINEAR_HISTORY_STRIDE="${TRAIN_LINEAR_HISTORY_STRIDE:-2}"
TRAIN_NONLINEAR_T="${TRAIN_NONLINEAR_T:-10.0}"
TRAIN_NONLINEAR_K0="${TRAIN_NONLINEAR_K0:-${K0}}"
TRAIN_NONLINEAR_HISTORY_STRIDE="${TRAIN_NONLINEAR_HISTORY_STRIDE:-20}"
TRAIN_WEAK_EPS="${TRAIN_WEAK_EPS:-0.03,0.05,0.07,0.1,0.15}"
TRAIN_STRONG_EPS="${TRAIN_STRONG_EPS:-0.15,0.25,0.35,0.5,0.65}"

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

ACTIVE_PIDS=()
ACTIVE_NVS=()
ACTIVE_LOGS=()

cancel_active_jobs() {
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}

reap_finished_jobs() {
  local -a next_pids=()
  local -a next_nvs=()
  local -a next_logs=()
  local idx pid nv log_path
  for idx in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[idx]}"
    nv="${ACTIVE_NVS[idx]}"
    log_path="${ACTIVE_LOGS[idx]}"
    if kill -0 "${pid}" 2>/dev/null; then
      next_pids+=("${pid}")
      next_nvs+=("${nv}")
      next_logs+=("${log_path}")
      continue
    fi
    if wait "${pid}"; then
      echo "[nv-sweep-online-rollout-q-hybrid] [1/2] Completed Nv=${nv}"
    else
      echo "[nv-sweep-online-rollout-q-hybrid] [1/2] Training failed for Nv=${nv}. Log: ${log_path}" >&2
      if [[ -f "${log_path}" ]]; then
        tail -n 40 "${log_path}" >&2 || true
      fi
      cancel_active_jobs
      return 1
    fi
  done
  ACTIVE_PIDS=()
  ACTIVE_NVS=()
  ACTIVE_LOGS=()
  if [[ "${#next_pids[@]}" -gt 0 ]]; then
    ACTIVE_PIDS=("${next_pids[@]}")
    ACTIVE_NVS=("${next_nvs[@]}")
    ACTIVE_LOGS=("${next_logs[@]}")
  fi
  return 0
}

wait_for_available_slot() {
  while [[ "${#ACTIVE_PIDS[@]}" -ge "${TRAIN_PARALLEL_JOBS}" ]]; do
    sleep 1
    reap_finished_jobs || return 1
  done
  return 0
}

wait_for_all_jobs() {
  while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
    sleep 1
    reap_finished_jobs || return 1
  done
  return 0
}

ladder_csv_for_target() {
  local target="$1"
  "${PYTHON_BIN}" - <<'PY' "${target}" "${TRAIN_NM}" "${TRAIN_FIXED_RATIO}"
import math
import sys

target = int(sys.argv[1])
nm = int(sys.argv[2])
ratio = float(sys.argv[3])
if target < nm:
    raise SystemExit(f"target Nv={target} must be at least TRAIN_NM={nm}")
if ratio <= 1.0:
    raise SystemExit(f"TRAIN_FIXED_RATIO must be greater than 1; got {ratio}")
if target == nm:
    ladder = [nm]
else:
    ladder = [target]
    current = target
    while True:
        next_value = int(math.ceil(float(current) / ratio))
        if next_value <= nm:
            ladder.append(nm)
            break
        ladder.append(next_value)
        current = next_value
    ladder = sorted(set(max(nm, min(target, int(value))) for value in ladder))
print(",".join(str(value) for value in ladder))
PY
}

prepare_online_reference_cache() {
  local -a cache_args=(
    --training-mode online_rollout
    --train-objective trajectory
    --build-dataset-only
    --dataset-cache "${ONLINE_REFERENCE_CACHE}"
    --online-loss-backend "${TRAIN_ONLINE_LOSS_BACKEND}"
    --online-v-probes "${TRAIN_ONLINE_V_PROBES}"
    --teacher-backend grid_cubic_spline
    --Nv-targets "${NV_LIST}"
    --Nm "${TRAIN_NM}"
    --regimes "${TRAIN_REGIMES}"
    --val-fraction "${TRAIN_VAL_FRACTION}"
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
  echo "[nv-sweep-online-rollout-q-hybrid] [1/2] Preparing shared online reference cache at ${ONLINE_REFERENCE_CACHE}"
  "${PYTHON_BIN}" -m model.train.train "${cache_args[@]}"
}

train_one_nv() {
  local idx="$1"
  local nv_raw="$2"
  local nv model_dir checkpoint_nv loss_plot_nv log_path train_ladder_csv q_dataset_cache teacher_proj_nv_target
  nv="$(echo "${nv_raw}" | tr -d '[:space:]')"
  if [[ -z "${nv}" ]]; then
    return 0
  fi
  model_dir="${CHECKPOINT_ROOT}/nv${nv}"
  checkpoint_nv="${model_dir}/interface_closure.npz"
  loss_plot_nv="${model_dir}/interface_closure.loss.png"
  log_path="${model_dir}/interface_closure.train.log"
  q_dataset_cache="${model_dir}/interface_closure_dataset.npz"
  train_ladder_csv="$(ladder_csv_for_target "${nv}")"
  teacher_proj_nv_target="$(( nv + 1 ))"
  mkdir -p "${model_dir}"

  local -a train_args=(
    --checkpoint "${checkpoint_nv}"
    --dataset-cache "${q_dataset_cache}"
    --online-reference-cache "${ONLINE_REFERENCE_CACHE}"
    --loss-plot "${loss_plot_nv}"
    --training-mode online_rollout
    --train-objective trajectory_q_hybrid
    --online-loss-backend "${TRAIN_ONLINE_LOSS_BACKEND}"
    --online-v-probes "${TRAIN_ONLINE_V_PROBES}"
    --online-case-batch-size "${TRAIN_ONLINE_CASE_BATCH_SIZE}"
    --teacher-backend grid_cubic_spline
    --teacher-proj-Nv "${teacher_proj_nv_target}"
    --Nv-targets "${train_ladder_csv}"
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
    --context-mode "${TRAIN_CONTEXT_MODE}"
    --tail-start-fraction "${TRAIN_TAIL_START_FRACTION}"
    --lambda-q "${TRAIN_LAMBDA_Q}"
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
    --linear-history-stride "${TRAIN_LINEAR_HISTORY_STRIDE}"
    --nonlinear-T "${TRAIN_NONLINEAR_T}"
    --nonlinear-k0 "${TRAIN_NONLINEAR_K0}"
    --nonlinear-history-stride "${TRAIN_NONLINEAR_HISTORY_STRIDE}"
    --weak-eps "${TRAIN_WEAK_EPS}"
    --strong-eps "${TRAIN_STRONG_EPS}"
  )
  if [[ "${TRAIN_ROLLOUT_DEALIAS_23}" != "0" ]]; then
    train_args+=(--rollout-dealias-23)
  fi

  if [[ "${TRAIN_PARALLEL_JOBS}" -le 1 ]]; then
    echo "[nv-sweep-online-rollout-q-hybrid] [1/2] Training closure $((idx + 1))/${TOTAL_NV} for Nv=${nv} with Nv-targets=${train_ladder_csv}"
    "${PYTHON_BIN}" -m model.train.train "${train_args[@]}"
    return 0
  fi

  wait_for_available_slot || return 1
  echo "[nv-sweep-online-rollout-q-hybrid] [1/2] Launching closure $((idx + 1))/${TOTAL_NV} for Nv=${nv} with Nv-targets=${train_ladder_csv} (log: ${log_path})"
  (
    "${PYTHON_BIN}" -m model.train.train "${train_args[@]}"
  ) >"${log_path}" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_NVS+=("${nv}")
  ACTIVE_LOGS+=("${log_path}")
}

if [[ "${RUN_TRAIN}" != "0" ]]; then
  echo "[nv-sweep-online-rollout-q-hybrid] [1/2] Training one online_rollout hybrid checkpoint per deployment Nv"
  prepare_online_reference_cache
  for idx in "${!NV_VALUES[@]}"; do
    train_one_nv "${idx}" "${NV_VALUES[idx]}"
  done
  wait_for_all_jobs
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
  echo "[nv-sweep-online-rollout-q-hybrid] [1/2] Reusing existing hybrid checkpoints in ${CHECKPOINT_ROOT}"
fi

ARGS+=(--checkpoint-dir "${CHECKPOINT_ROOT}")
echo "[nv-sweep-online-rollout-q-hybrid] [2/2] Running nonlinear Nv sweep"
"${PYTHON_BIN}" -m model.eval_nv_sweep "${ARGS[@]}"

cat <<EOF

Done.

Artifacts:
  mode:                online_rollout_q_trajectory_fixed_ratio
  checkpoint dir:      ${CHECKPOINT_ROOT}
  online ref cache:    ${ONLINE_REFERENCE_CACHE}
  q dataset caches:    ${CHECKPOINT_ROOT}/nv*/interface_closure_dataset.npz
  summary:             ${OUTDIR}/summary.json
  metric 1:            ${OUTDIR}/nv_sweep_metric1.png
  metric 2:            ${OUTDIR}/nv_sweep_metric2.png
  phase space:         ${OUTDIR}/fig10_learned_vs_nonlocal_nv_sweep_phase_space.png
  phase payload:       ${OUTDIR}/nv_sweep_phase_space_payload.npz

Defaults:
  objective:           trajectory_q_hybrid
  loss backend:        ${TRAIN_ONLINE_LOSS_BACKEND}
  fixed ratio:         ${TRAIN_FIXED_RATIO}
  train dt:            ${TRAIN_TEACHER_DT}
  train linear T:      ${TRAIN_LINEAR_T}
  train nonlin T:      ${TRAIN_NONLINEAR_T}
  lambda q:            ${TRAIN_LAMBDA_Q}
  lambda E:            ${TRAIN_LAMBDA_E}
  lambda dist:         ${TRAIN_LAMBDA_DIST}
  lambda tail:         ${TRAIN_LAMBDA_TAIL}
  lambda neg:          ${TRAIN_LAMBDA_NEG}
  q batch size:        ${TRAIN_BATCH_SIZE}
  v probes:            ${TRAIN_ONLINE_V_PROBES}
  case batch:          ${TRAIN_ONLINE_CASE_BATCH_SIZE}
  parallel jobs:       ${TRAIN_PARALLEL_JOBS}
  train Nm:            ${TRAIN_NM}
  steps/epoch:         ${TRAIN_STEPS_PER_EPOCH}
  Nv list:             ${NV_LIST}
EOF
