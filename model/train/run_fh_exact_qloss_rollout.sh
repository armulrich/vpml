#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON:-python}"
fi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-True}"
export VPML_JAX_BACKEND="${VPML_JAX_BACKEND:-cpu}"

TRAIN_ROLLOUT_HORIZON="${TRAIN_ROLLOUT_HORIZON:-256}"
OUTDIR="${1:-${REPO_ROOT}/out_bench/fh_exact_qloss_rollout_H${TRAIN_ROLLOUT_HORIZON}}"
NV_LIST="${NV_LIST:-8,64,256,300,512}"
NX="${NX:-200}"
DT="${DT:-0.005}"
T_FINAL="${T_FINAL:-60.0}"
EPS="${EPS:-0.5}"
K0="${K0:-0.5}"
SNAPSHOT_TIMES="${SNAPSHOT_TIMES:-20.0,40.0,60.0}"
NV_PLOT="${NV_PLOT:-1000}"
PHASE_VMIN="${PHASE_VMIN:-0.0}"
PHASE_VMAX="${PHASE_VMAX:-0.5}"
PHASE_VRANGE="${PHASE_VRANGE:--4.0,4.0}"
PHASE_REFERENCE_NV="${PHASE_REFERENCE_NV:-}"
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
RUN_EVAL="${RUN_EVAL:-1}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTDIR}/models}"
INIT_CHECKPOINT_ROOT="${INIT_CHECKPOINT_ROOT:-${TRAIN_INIT_CHECKPOINT_ROOT:-}}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${TRAIN_INIT_CHECKPOINT_PATH:-}}"
TRAIN_TAIL_CHAIN_ONLY_EFFECTIVE="0"
TRAIN_NV_LADDER_MODE="${TRAIN_NV_LADDER_MODE:-fixed_ratio}"
TRAIN_NV_TARGETS_CSV="${TRAIN_NV_TARGETS_CSV:-}"
TRAIN_FIXED_RATIO="${TRAIN_FIXED_RATIO:-1.8}"
TRAIN_NM="${TRAIN_NM:-6}"
TRAIN_HIDDEN_WIDTH="${TRAIN_HIDDEN_WIDTH:-128}"
TRAIN_RES_BLOCKS="${TRAIN_RES_BLOCKS:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-300}"
if [[ -z "${TRAIN_LR:-}" ]]; then
  if [[ "${TRAIN_ROLLOUT_HORIZON}" == "1" ]]; then
    TRAIN_LR="1e-3"
  else
    TRAIN_LR="3e-4"
  fi
fi
TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-1.0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
if [[ -z "${TRAIN_BATCH_SIZE:-}" ]]; then
  if [[ "${TRAIN_ROLLOUT_HORIZON}" == "1" ]]; then
    TRAIN_BATCH_SIZE="8192"
  else
    TRAIN_BATCH_SIZE="512"
  fi
fi
if [[ -z "${TRAIN_STEPS_PER_EPOCH:-}" ]]; then
  if [[ "${TRAIN_ROLLOUT_HORIZON}" == "1" ]]; then
    TRAIN_STEPS_PER_EPOCH="50"
  else
    TRAIN_STEPS_PER_EPOCH="50"
  fi
fi
TRAIN_SEED="${TRAIN_SEED:-0}"
TRAIN_N_LOW="${TRAIN_N_LOW:-2}"
TRAIN_VAL_FRACTION="${TRAIN_VAL_FRACTION:-0.2}"
TRAIN_REGIMES="${TRAIN_REGIMES:-linear_landau,nonlinear_landau_weak,nonlinear_landau_strong}"
TRAIN_CONTEXT_MODE="${TRAIN_CONTEXT_MODE:-none}"
TRAIN_ROLLOUT_DEALIAS_23="${TRAIN_ROLLOUT_DEALIAS_23:-1}"
if [[ -z "${TRAIN_EXACT_ROLLOUT_PRECISION:-}" ]]; then
  if [[ "${TRAIN_ROLLOUT_HORIZON}" == "1" ]]; then
    TRAIN_EXACT_ROLLOUT_PRECISION="float64"
  else
    TRAIN_EXACT_ROLLOUT_PRECISION="float32"
  fi
fi
TRAIN_EXACT_TARGET_SAMPLING="${TRAIN_EXACT_TARGET_SAMPLING:-cycle}"
TRAIN_EXACT_STORE_TRAIN_QPAIRS="${TRAIN_EXACT_STORE_TRAIN_QPAIRS:-0}"
TRAIN_EXACT_STORE_TRAIN_QPAIRS_EFFECTIVE="${TRAIN_EXACT_STORE_TRAIN_QPAIRS}"
TRAIN_TAIL_CHAIN="${TRAIN_TAIL_CHAIN:-${TRAIN_TAIL_DECODER:-0}}"
TRAIN_TAIL_CHAIN_NV="${TRAIN_TAIL_CHAIN_NV:-${TRAIN_TAIL_DECODER_NV:-512}}"
TRAIN_TAIL_CHAIN_N_MIN="${TRAIN_TAIL_CHAIN_N_MIN:-}"
TRAIN_TAIL_CHAIN_N_MAX="${TRAIN_TAIL_CHAIN_N_MAX:-${TRAIN_TAIL_DECODER_N_MAX:-}}"
TRAIN_TAIL_CHAIN_CHUNK_SIZE="${TRAIN_TAIL_CHAIN_CHUNK_SIZE:-16}"
TRAIN_TAIL_CHAIN_LIFT_HORIZONS="${TRAIN_TAIL_CHAIN_LIFT_HORIZONS:-}"
TRAIN_TAIL_CHAIN_RECURSIVE_LIFT="${TRAIN_TAIL_CHAIN_RECURSIVE_LIFT:-0}"
TRAIN_LAMBDA_TAIL_CHAIN="${TRAIN_LAMBDA_TAIL_CHAIN:-${TRAIN_LAMBDA_TAIL_DECODER:-1.0}}"
TRAIN_TAIL_HISTORY_LIFT="${TRAIN_TAIL_HISTORY_LIFT:-0}"
TRAIN_TAIL_HISTORY_NV="${TRAIN_TAIL_HISTORY_NV:-512}"
TRAIN_TAIL_HISTORY_N_MAX="${TRAIN_TAIL_HISTORY_N_MAX:-${TRAIN_TAIL_HISTORY_NV}}"
TRAIN_TAIL_HISTORY_LAGS="${TRAIN_TAIL_HISTORY_LAGS:-8}"
TRAIN_TAIL_HISTORY_LOSS="${TRAIN_TAIL_HISTORY_LOSS:-coeff}"
TRAIN_TAIL_HISTORY_XV_GRID="${TRAIN_TAIL_HISTORY_XV_GRID:-512}"
if [[ "${TRAIN_TAIL_CHAIN}" != "0" && -n "${INIT_CHECKPOINT_ROOT}${INIT_CHECKPOINT_PATH}" && "${TRAIN_TAIL_CHAIN_RECURSIVE_LIFT}" == "0" ]]; then
  TRAIN_TAIL_CHAIN_ONLY_EFFECTIVE="1"
fi
TRAIN_PROFILE_TRACE_DIR="${TRAIN_PROFILE_TRACE_DIR:-}"
TRAIN_PROFILE_STEPS="${TRAIN_PROFILE_STEPS:-0}"
TRAIN_PROFILE_SKIP_STEPS="${TRAIN_PROFILE_SKIP_STEPS:-1}"

TRAIN_TEACHER_NX="${TRAIN_TEACHER_NX:-${TEACHER_NX}}"
TRAIN_TEACHER_NV="${TRAIN_TEACHER_NV:-${TEACHER_NV}}"
TRAIN_TEACHER_DT="${TRAIN_TEACHER_DT:-0.01}"
TRAIN_TEACHER_VMIN="${TRAIN_TEACHER_VMIN:-${TEACHER_VMIN}}"
TRAIN_TEACHER_VMAX="${TRAIN_TEACHER_VMAX:-${TEACHER_VMAX}}"
TRAIN_LINEAR_T="${TRAIN_LINEAR_T:-60.0}"
TRAIN_LINEAR_EPS="${TRAIN_LINEAR_EPS:-0.01}"
TRAIN_LINEAR_MODES="${TRAIN_LINEAR_MODES:-0.5,1.0,1.5,2.0}"
TRAIN_LINEAR_NUM_SAMPLES="${TRAIN_LINEAR_NUM_SAMPLES:-8}"
TRAIN_LINEAR_SEED="${TRAIN_LINEAR_SEED:-0}"
TRAIN_LINEAR_HISTORY_STRIDE="${TRAIN_LINEAR_HISTORY_STRIDE:-2}"
TRAIN_NONLINEAR_T="${TRAIN_NONLINEAR_T:-60.0}"
TRAIN_NONLINEAR_K0="${TRAIN_NONLINEAR_K0:-${K0}}"
TRAIN_NONLINEAR_HISTORY_STRIDE="${TRAIN_NONLINEAR_HISTORY_STRIDE:-20}"
TRAIN_WEAK_EPS="${TRAIN_WEAK_EPS:-0.03,0.05,0.07,0.1,0.15}"
TRAIN_STRONG_EPS="${TRAIN_STRONG_EPS:-0.15,0.25,0.35,0.5,0.65}"

if [[ "${TRAIN_CONTEXT_MODE}" != "none" ]]; then
  echo "run_fh_exact_qloss_rollout.sh only supports TRAIN_CONTEXT_MODE=none; got '${TRAIN_CONTEXT_MODE}'." >&2
  exit 1
fi

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

ladder_csv_for_mode() {
  local target="$1"
  case "${TRAIN_NV_LADDER_MODE}" in
    target_only)
      printf '%s\n' "${target}"
      ;;
    fixed_ratio)
      ladder_csv_for_target "${target}"
      ;;
    *)
      echo "Unsupported TRAIN_NV_LADDER_MODE='${TRAIN_NV_LADDER_MODE}'. Expected 'target_only' or 'fixed_ratio'." >&2
      exit 1
      ;;
  esac
}

normalize_nv_targets_csv() {
  local csv="$1"
  local target="$2"
  "${PYTHON_BIN}" - <<'PY' "${csv}" "${target}" "${TRAIN_NM}"
import sys

raw = sys.argv[1]
target = int(sys.argv[2])
nm = int(sys.argv[3])
values = []
for part in raw.split(","):
    part = part.strip()
    if not part:
        continue
    try:
        value = int(part)
    except ValueError as exc:
        raise SystemExit(f"TRAIN_NV_TARGETS_CSV contains a non-integer value: {part!r}") from exc
    values.append(value)
if not values:
    raise SystemExit("TRAIN_NV_TARGETS_CSV must contain at least one Nv target")
if any(value < nm for value in values):
    raise SystemExit(f"TRAIN_NV_TARGETS_CSV values must be >= TRAIN_NM={nm}: {values}")
if any(value > target for value in values):
    raise SystemExit(f"TRAIN_NV_TARGETS_CSV values must be <= current NV={target}: {values}")
if max(values) != target:
    raise SystemExit(
        f"TRAIN_NV_TARGETS_CSV must include current NV={target} as its maximum target: {values}"
    )
print(",".join(str(value) for value in sorted(set(values))))
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
if [[ -n "${PHASE_REFERENCE_NV}" ]]; then
  ARGS+=(--phase-reference-Nv "${PHASE_REFERENCE_NV}")
fi

mkdir -p "${CHECKPOINT_ROOT}"
IFS=',' read -r -a NV_VALUES <<< "${NV_LIST}"
TOTAL_NV="${#NV_VALUES[@]}"

if [[ "${RUN_TRAIN}" != "0" ]]; then
  echo "[fh-exact-qloss-rollout] [1/3] Training exact q-rollout checkpoints with full-history caches and no anchor pool"
  for idx in "${!NV_VALUES[@]}"; do
    NV_RAW="${NV_VALUES[idx]}"
    NV="$(echo "${NV_RAW}" | tr -d '[:space:]')"
    if [[ -z "${NV}" ]]; then
      continue
    fi
    if [[ -n "${TRAIN_NV_TARGETS_CSV}" ]]; then
      TRAIN_LADDER_CSV="$(normalize_nv_targets_csv "${TRAIN_NV_TARGETS_CSV}" "${NV}")"
    else
      TRAIN_LADDER_CSV="$(ladder_csv_for_mode "${NV}")"
    fi
    MODEL_DIR="${CHECKPOINT_ROOT}/nv${NV}"
    CHECKPOINT_NV="${MODEL_DIR}/interface_closure.npz"
    LOSS_PLOT_NV="${MODEL_DIR}/interface_closure.loss.png"
    DATASET_CACHE_NV="${MODEL_DIR}/interface_closure_exact_q_rollout_histories.npz"
    mkdir -p "${MODEL_DIR}"

    TRAIN_ARGS=(
      --checkpoint "${CHECKPOINT_NV}"
      --dataset-cache "${DATASET_CACHE_NV}"
      --loss-plot "${LOSS_PLOT_NV}"
      --training-mode exact_q_rollout
      --train-objective q_rollout
      --rollout-horizon "${TRAIN_ROLLOUT_HORIZON}"
      --exact-rollout-precision "${TRAIN_EXACT_ROLLOUT_PRECISION}"
      --exact-target-sampling "${TRAIN_EXACT_TARGET_SAMPLING}"
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
      --context-mode none
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
    if [[ "${TRAIN_EXACT_STORE_TRAIN_QPAIRS}" != "0" ]]; then
      TRAIN_ARGS+=(--exact-store-train-qpairs)
    fi
    if [[ -n "${INIT_CHECKPOINT_PATH}" ]]; then
      INIT_CHECKPOINT_NV="${INIT_CHECKPOINT_PATH}"
      if [[ ! -f "${INIT_CHECKPOINT_NV}" ]]; then
        echo "INIT_CHECKPOINT_PATH requires an existing checkpoint at ${INIT_CHECKPOINT_NV}" >&2
        exit 1
      fi
      TRAIN_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT_NV}")
    elif [[ -n "${INIT_CHECKPOINT_ROOT}" ]]; then
      INIT_CHECKPOINT_NV="${INIT_CHECKPOINT_ROOT}/nv${NV}/interface_closure.npz"
      if [[ ! -f "${INIT_CHECKPOINT_NV}" ]]; then
        echo "INIT_CHECKPOINT_ROOT requires an existing checkpoint at ${INIT_CHECKPOINT_NV}" >&2
        exit 1
      fi
      TRAIN_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT_NV}")
    fi
    if [[ "${TRAIN_TAIL_CHAIN}" != "0" ]]; then
      TRAIN_ARGS+=(
        --tail-chain
        --tail-chain-Nv "${TRAIN_TAIL_CHAIN_NV}"
        --tail-chain-chunk-size "${TRAIN_TAIL_CHAIN_CHUNK_SIZE}"
      )
      if [[ -n "${TRAIN_TAIL_CHAIN_LIFT_HORIZONS}" ]]; then
        TRAIN_ARGS+=(--tail-chain-lift-horizons "${TRAIN_TAIL_CHAIN_LIFT_HORIZONS}")
      fi
      if [[ "${TRAIN_TAIL_CHAIN_RECURSIVE_LIFT}" != "0" ]]; then
        TRAIN_ARGS+=(--tail-chain-recursive-lift --lambda-tail-chain "${TRAIN_LAMBDA_TAIL_CHAIN}")
      elif [[ -n "${INIT_CHECKPOINT_ROOT}${INIT_CHECKPOINT_PATH}" ]]; then
        TRAIN_TAIL_CHAIN_ONLY_EFFECTIVE="1"
      else
        TRAIN_ARGS+=(--lambda-tail-chain "${TRAIN_LAMBDA_TAIL_CHAIN}")
      fi
      if [[ -n "${TRAIN_TAIL_CHAIN_N_MIN}" ]]; then
        TRAIN_ARGS+=(--tail-chain-n-min "${TRAIN_TAIL_CHAIN_N_MIN}")
      fi
      if [[ -n "${TRAIN_TAIL_CHAIN_N_MAX}" ]]; then
        TRAIN_ARGS+=(--tail-chain-n-max "${TRAIN_TAIL_CHAIN_N_MAX}")
      fi
    fi
    if [[ "${TRAIN_TAIL_HISTORY_LIFT}" != "0" ]]; then
      TRAIN_ARGS+=(
        --tail-history-lift
        --tail-history-Nv "${TRAIN_TAIL_HISTORY_NV}"
        --tail-history-n-max "${TRAIN_TAIL_HISTORY_N_MAX}"
        --tail-history-lags "${TRAIN_TAIL_HISTORY_LAGS}"
        --tail-history-loss "${TRAIN_TAIL_HISTORY_LOSS}"
        --tail-history-xv-grid "${TRAIN_TAIL_HISTORY_XV_GRID}"
      )
    fi
    if [[ -n "${TRAIN_PROFILE_TRACE_DIR}" && "${TRAIN_PROFILE_STEPS}" != "0" ]]; then
      TRAIN_ARGS+=(
        --profile-trace-dir "${TRAIN_PROFILE_TRACE_DIR}"
        --profile-train-steps "${TRAIN_PROFILE_STEPS}"
        --profile-skip-steps "${TRAIN_PROFILE_SKIP_STEPS}"
      )
    fi
    if [[ "${TRAIN_ROLLOUT_DEALIAS_23}" != "0" ]]; then
      TRAIN_ARGS+=(--rollout-dealias-23)
    fi

    echo "[fh-exact-qloss-rollout] [2/3] Training closure $((idx + 1))/${TOTAL_NV} for Nv=${NV} with Nv-targets=${TRAIN_LADDER_CSV}"
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
  echo "[fh-exact-qloss-rollout] [1/3] Skipping model training because RUN_TRAIN=${RUN_TRAIN}"
  echo "[fh-exact-qloss-rollout] [2/3] Reusing existing checkpoints in ${CHECKPOINT_ROOT}"
fi

ARGS+=(--checkpoint-dir "${CHECKPOINT_ROOT}")
if [[ "${RUN_EVAL}" != "0" ]]; then
  echo "[fh-exact-qloss-rollout] [3/3] Running nonlinear Nv sweep with HR/truncation/learned evaluation panels"
  "${PYTHON_BIN}" -m model.eval_nv_sweep "${ARGS[@]}"
else
  echo "[fh-exact-qloss-rollout] [3/3] Skipping nonlinear Nv sweep because RUN_EVAL=${RUN_EVAL}"
fi

cat <<EOF

Done.

Artifacts:
  mode:           exact_q_rollout_${TRAIN_NV_LADDER_MODE}_grid_teacher
  checkpoint dir: ${CHECKPOINT_ROOT}
  init checkpoints:${INIT_CHECKPOINT_PATH:-${INIT_CHECKPOINT_ROOT:-<none>}}
  history caches: ${CHECKPOINT_ROOT}/nv*/interface_closure_exact_q_rollout_histories.npz
  summary:        ${OUTDIR}/summary.json
  metric 1:       ${OUTDIR}/nv_sweep_metric1.png
  metric 2:       ${OUTDIR}/nv_sweep_metric2.png
  phase space:    ${OUTDIR}/fig10_learned_vs_nonlocal_nv_sweep_phase_space.png
  phase payload:  ${OUTDIR}/nv_sweep_phase_space_payload.npz

Defaults:
  objective:      q_rollout
  ladder mode:    ${TRAIN_NV_LADDER_MODE}
  fixed ratio:    ${TRAIN_FIXED_RATIO}
  rollout horiz:  ${TRAIN_ROLLOUT_HORIZON}
  anchor pool:    none
  eval dt:        ${DT}
  eval teacher dt:${TEACHER_DT}
  phase ref Nv:  ${PHASE_REFERENCE_NV:-deployment}
  train teacher dt:${TRAIN_TEACHER_DT}
  train linear T: ${TRAIN_LINEAR_T}
  train nonlin T: ${TRAIN_NONLINEAR_T}
  train Nm:       ${TRAIN_NM}
  batch size:     ${TRAIN_BATCH_SIZE}
  steps/epoch:    ${TRAIN_STEPS_PER_EPOCH}
  lr:             ${TRAIN_LR}
  precision:      ${TRAIN_EXACT_ROLLOUT_PRECISION}
  target sampling:${TRAIN_EXACT_TARGET_SAMPLING}
  store q-pairs:  ${TRAIN_EXACT_STORE_TRAIN_QPAIRS_EFFECTIVE}
  tail chain:     ${TRAIN_TAIL_CHAIN}
  tail chain Nv:  ${TRAIN_TAIL_CHAIN_NV}
  chain n range:  $(if [[ "${TRAIN_TAIL_CHAIN_RECURSIVE_LIFT}" != "0" ]]; then echo "${TRAIN_TAIL_CHAIN_N_MIN:-auto(target Nv)}"; else echo "${TRAIN_TAIL_CHAIN_N_MIN:-auto(target Nv+1)}"; fi)..${TRAIN_TAIL_CHAIN_N_MAX:-tail_Nv}
  chain n batch:  all
  chain chunk:    ${TRAIN_TAIL_CHAIN_CHUNK_SIZE}
  lift horizons:  ${TRAIN_TAIL_CHAIN_LIFT_HORIZONS:-full}
  chain objective:$(if [[ "${TRAIN_TAIL_CHAIN_RECURSIVE_LIFT}" != "0" ]]; then echo " recursive lift from frozen base"; elif [[ "${TRAIN_TAIL_CHAIN_ONLY_EFFECTIVE}" == "1" ]]; then echo " chain-only continuation"; elif [[ "${TRAIN_TAIL_CHAIN}" != "0" ]]; then echo " dyn + lambda*chain"; else echo " disabled"; fi)
  lambda chain:   $(if [[ "${TRAIN_TAIL_CHAIN}" == "0" || "${TRAIN_TAIL_CHAIN_ONLY_EFFECTIVE}" == "1" ]]; then echo "not used"; else echo "${TRAIN_LAMBDA_TAIL_CHAIN}"; fi)
  history lift:   ${TRAIN_TAIL_HISTORY_LIFT}
  hist lift Nv:   ${TRAIN_TAIL_HISTORY_NV}
  hist n range:   target Nv..${TRAIN_TAIL_HISTORY_N_MAX}
  hist lags:      ${TRAIN_TAIL_HISTORY_LAGS}
  hist loss:      ${TRAIN_TAIL_HISTORY_LOSS}
  hist xv grid:   ${TRAIN_TAIL_HISTORY_XV_GRID}
  context:        none
  Nv list:        ${NV_LIST}
EOF
