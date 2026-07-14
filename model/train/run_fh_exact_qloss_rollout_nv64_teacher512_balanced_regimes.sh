#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Balanced broad-regime exact-q experiment. Every regime contributes eight
# distinct teacher trajectories with the same physical duration and cache
# stride; this changes data exposure only, not the exact q-rollout objective.
export NV_LIST="${NV_LIST:-64}"
export TEACHER_NV="${TEACHER_NV:-512}"
export TRAIN_TEACHER_NV="${TRAIN_TEACHER_NV:-512}"
export T_FINAL="${T_FINAL:-60.0}"
export TRAIN_LINEAR_T="${TRAIN_LINEAR_T:-60.0}"
export TRAIN_NONLINEAR_T="${TRAIN_NONLINEAR_T:-60.0}"
export TRAIN_TEACHER_DT="${TRAIN_TEACHER_DT:-0.01}"
export TRAIN_ROLLOUT_HORIZON="${TRAIN_ROLLOUT_HORIZON:-128}"
export TRAIN_REGIMES="${TRAIN_REGIMES:-linear_landau,nonlinear_landau_weak,nonlinear_landau_strong}"
export TRAIN_LINEAR_NUM_SAMPLES="${TRAIN_LINEAR_NUM_SAMPLES:-8}"
export TRAIN_LINEAR_HISTORY_STRIDE="${TRAIN_LINEAR_HISTORY_STRIDE:-20}"
export TRAIN_NONLINEAR_HISTORY_STRIDE="${TRAIN_NONLINEAR_HISTORY_STRIDE:-20}"

# Retain the prior amplitudes, add distinct cases to reach eight per nonlinear
# regime, and remove the prior duplicate eps=0.15 shared by weak and strong.
export TRAIN_WEAK_EPS="${TRAIN_WEAK_EPS:-0.02,0.03,0.05,0.07,0.10,0.12,0.15,0.18}"
export TRAIN_STRONG_EPS="${TRAIN_STRONG_EPS:-0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.65}"

# Keep exact-q training untouched while making the broad IC ensemble directly
# inspectable in Metric 1, Metric 2, and Fig10 for every configured case.
export TRAIN_NV_LADDER_MODE="${TRAIN_NV_LADDER_MODE:-fixed_ratio}"
export TRAIN_EXACT_STORE_TRAIN_QPAIRS="${TRAIN_EXACT_STORE_TRAIN_QPAIRS:-0}"
export TRAIN_EXACT_TARGET_SAMPLING="${TRAIN_EXACT_TARGET_SAMPLING:-cycle}"
export TRAIN_VAL_FRACTION="${TRAIN_VAL_FRACTION:-0}"
export TRAIN_TAIL_CHAIN="${TRAIN_TAIL_CHAIN:-0}"
export TRAIN_TAIL_HISTORY_LIFT="${TRAIN_TAIL_HISTORY_LIFT:-0}"
export EVAL_TRAINING_CASES="${EVAL_TRAINING_CASES:-1}"
export EVAL_PHASE_REFERENCE_MODE="${EVAL_PHASE_REFERENCE_MODE:-raw_hr_grid}"
export PHASE_REFERENCE_NV="${PHASE_REFERENCE_NV:-512}"

csv_count() {
  local csv="$1"
  local count=0
  local value
  IFS=',' read -r -a values <<< "${csv}"
  for value in "${values[@]}"; do
    [[ -n "${value//[[:space:]]/}" ]] && ((count += 1))
  done
  printf '%s\n' "${count}"
}

if [[ "$(csv_count "${TRAIN_WEAK_EPS}")" != "8" || "$(csv_count "${TRAIN_STRONG_EPS}")" != "8" ]]; then
  echo "Balanced regimes require exactly eight weak and eight strong amplitudes." >&2
  exit 2
fi

OUTDIR="${1:-${REPO_ROOT}/out_bench/fh_exact_qloss_rollout_H${TRAIN_ROLLOUT_HORIZON}_nv64_teacher512_balanced8x3_stride20_B${TRAIN_BATCH_SIZE:-64}_steps${TRAIN_STEPS_PER_EPOCH:-30}}"

exec "${SCRIPT_DIR}/run_fh_exact_qloss_rollout_nv64_teacher512.sh" "${OUTDIR}"
