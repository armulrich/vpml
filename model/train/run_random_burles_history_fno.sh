#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/armin/Documents/NYU/vpml/.venv/bin/python}"
SHARED_OUTPUT_ROOT="${VPML_SHARED_OUTPUT_ROOT:-/Users/armin/Documents/NYU/vpml/out_bench}"
REFERENCE_CACHE="${TRAIN_LOW_MOMENT_REFERENCE_CACHE:-${SHARED_OUTPUT_ROOT}/reference_cache/interface_flux_landau_T120_Nx1024_Nv8192_M4096/e376aa1efa28e754b5f6}"
OUTDIR="${1:-${SHARED_OUTPUT_ROOT}/low_moment_burles_history_dt0025_autoburn5_deterministic_random1729_E1000}"
STOP_EPOCH="${STOP_EPOCH:-10}"
PLANNED_EPOCHS="${PLANNED_EPOCHS:-1000}"

ARGS=(
  --reference-cache "${REFERENCE_CACHE}"
  --outdir "${OUTDIR}"
  --rollout-Nx 128
  --solver-dt 0.025
  --rollout-horizon 400
  --training-schedule random_windows
  --memory-backend burles_latent_fno
  --latent-memory-dim 0
  --memory-steps 50
  --memory-stride 4
  --closure-history-input
  --autonomous-history-burnin-steps 200
  --input-scaling current_density_rms_arcsinh
  --allow-uniform-heating
  --batch-size 4
  --steps-per-epoch 1
  --training-passes-per-epoch 1
  --gradient-accumulation-steps 4
  --deterministic-anchor-cycle
  --no-translation-augmentation
  --width 64
  --spectral-modes 32
  --fno-depth 8
  --epochs "${STOP_EPOCH}"
  --planned-epochs "${PLANNED_EPOCHS}"
  --learning-rate 1e-3
  --final-learning-rate 1e-5
  --weight-decay 1e-4
  --grad-clip 0
  --require-first-update-descent
  --report-every-update-descent
  --global-relative-trajectory-loss
  --validation-every 1
  --diagnostic-start-times 0,20,40,60,80,100
  --seed 1729
  --skip-evaluation
)

if [[ -f "${OUTDIR}/training_state.npz" ]]; then
  ARGS+=(--resume-run "${OUTDIR}")
fi

mkdir -p "$(dirname "${OUTDIR}")"
cd "${REPO_ROOT}"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m model.train.low_moment_closure "${ARGS[@]}" \
  2>&1 | tee -a "${OUTDIR}.training.log"
