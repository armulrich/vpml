#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/armin/Documents/NYU/vpml/.venv/bin/python}"
SHARED_OUTPUT_ROOT="${VPML_SHARED_OUTPUT_ROOT:-/Users/armin/Documents/NYU/vpml/out_bench}"
REFERENCE_CACHE="${TRAIN_LOW_MOMENT_REFERENCE_CACHE:-${SHARED_OUTPUT_ROOT}/reference_cache/interface_flux_landau_T120_Nx1024_Nv8192_M4096/e376aa1efa28e754b5f6}"
CONVERGENCE_FLOOR="${TRAIN_CONVERGENCE_FLOOR_FILE:-${SHARED_OUTPUT_ROOT}/low_moment_convergence_floor_T120_Nx1024_Nv8192_16384_rolloutNx128/low_moment_convergence_floors.npz}"
OUTDIR="${1:-${SHARED_OUTPUT_ROOT}/low_moment_burles_latent6_dt0025_qhistory_random1729_E1000}"
STOP_EPOCH="${STOP_EPOCH:-20}"
PLANNED_EPOCHS="${PLANNED_EPOCHS:-1000}"

ARGS=(
  --reference-cache "${REFERENCE_CACHE}"
  --outdir "${OUTDIR}"
  --rollout-Nx 128
  --solver-dt 0.025
  --rollout-horizon 400
  --training-schedule random_windows
  --horizon-curriculum 40:2,100:3,200:5,400:990
  --memory-backend burles_latent_fno
  --memory-steps 50
  --memory-stride 4
  --closure-history-input
  --input-scaling current_density_rms_arcsinh
  --batch-size 4
  --steps-per-epoch 4
  --training-passes-per-epoch 4
  --gradient-accumulation-steps 4
  --width 64
  --spectral-modes 32
  --fno-depth 8
  --latent-memory-dim 6
  --epochs "${STOP_EPOCH}"
  --planned-epochs "${PLANNED_EPOCHS}"
  --learning-rate 1e-3
  --final-learning-rate 1e-5
  --weight-decay 1e-4
  --grad-clip 1
  --require-first-update-descent
  --relative-trajectory-loss
  --relative-time-block 2.0
  --convergence-floor-file "${CONVERGENCE_FLOOR}"
  --log-energy-weight 0.01
  --log-growth-weight 0.05
  --validation-every 5
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
