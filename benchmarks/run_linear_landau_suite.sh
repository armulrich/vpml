#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
OUTDIR="${1:-${REPO_ROOT}/out_bench}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

cd "${REPO_ROOT}"

for method in truncation hyper filter nonlocal; do
  "${PYTHON_BIN}" -m benchmarks.fh_benchmarks_2412_07073_jax linear_landau --method "${method}" --outdir "${OUTDIR}"
done
