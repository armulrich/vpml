#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/armin/Documents/NYU/vpml/.venv/bin/python}"
RUN_DIR="${1:?usage: evaluate_burles_latent_fno.sh RUN_DIR EPOCH}"
EPOCH="${2:?usage: evaluate_burles_latent_fno.sh RUN_DIR EPOCH}"
SHARED_OUTPUT_ROOT="${VPML_SHARED_OUTPUT_ROOT:-/Users/armin/Documents/NYU/vpml/out_bench}"
REFERENCE_CACHE="${TRAIN_LOW_MOMENT_REFERENCE_CACHE:-${SHARED_OUTPUT_ROOT}/reference_cache/interface_flux_landau_T120_Nx1024_Nv8192_M4096/e376aa1efa28e754b5f6}"
CHECKPOINT="${RUN_DIR}/epoch$(printf '%03d' "${EPOCH}")_low_moment_closure.npz"
EVAL_DIR="${RUN_DIR}/evaluations/epoch$(printf '%03d' "${EPOCH}")"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m model.train.low_moment_closure \
  --reference-cache "${REFERENCE_CACHE}" \
  --outdir "${EVAL_DIR}" \
  --evaluate-checkpoint "${CHECKPOINT}" \
  --evaluation-chunk-steps 250
