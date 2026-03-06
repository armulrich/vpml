#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
OUTDIR="${1:-${REPO_ROOT}/out_bench}"
NV="${NV:-20}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m benchmarks.fh_benchmarks_2412_07073_jax fig2 --outdir "${OUTDIR}"
"${PYTHON_BIN}" -m benchmarks.fh_benchmarks_2412_07073_jax fig3 --outdir "${OUTDIR}"
"${PYTHON_BIN}" -m benchmarks.fh_benchmarks_2412_07073_jax fig4 --outdir "${OUTDIR}" --Nv "${NV}"
"${SCRIPT_DIR}/run_linear_landau_suite.sh" "${OUTDIR}"
