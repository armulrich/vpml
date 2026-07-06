#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

runs=(
  "1 512 120 B512_steps120"
  "1 64 30 B64_steps30"
  "256 64 30 B64_steps30"
)

outdir_for() {
  local horizon="$1"
  local tag="$4"
  printf 'out_bench/fh_exact_qloss_rollout_H%s_allk_nv64_strong_ladder_%s\n' "${horizon}" "${tag}"
}

for spec in "${runs[@]}"; do
  # shellcheck disable=SC2086
  set -- ${spec}
  outdir="$(outdir_for "$@")"
  if [[ -e "${outdir}" ]]; then
    echo "Refusing to overwrite existing output folder: ${outdir}" >&2
    exit 1
  fi
done

for spec in "${runs[@]}"; do
  # shellcheck disable=SC2086
  set -- ${spec}
  horizon="$1"
  batch_size="$2"
  steps_per_epoch="$3"
  outdir="$(outdir_for "$@")"

  echo "[fh-exact-h1-h256-budget] H=${horizon} batch=${batch_size} steps/epoch=${steps_per_epoch} -> ${outdir}"
  TRAIN_VAL_FRACTION=0 \
  NV_LIST=64 \
  TRAIN_NV_LADDER_MODE=fixed_ratio \
  TRAIN_REGIMES=nonlinear_landau_strong \
  TRAIN_ROLLOUT_HORIZON="${horizon}" \
  TRAIN_BATCH_SIZE="${batch_size}" \
  TRAIN_STEPS_PER_EPOCH="${steps_per_epoch}" \
  TRAIN_EPOCHS=100 \
  TRAIN_LR=1e-4 \
  TRAIN_GRAD_CLIP=0.5 \
  TRAIN_LOG_EVERY=1 \
  ./model/train/run_fh_exact_qloss_rollout.sh "${outdir}"
done
