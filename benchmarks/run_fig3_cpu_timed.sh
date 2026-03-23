#!/usr/bin/env bash
# Run Fig. 3 benchmark on CPU (wrapper around run_fig_cpu_timed.sh).
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTDIR="${1:-${REPO_ROOT}/out_bench}"
exec bash "${SCRIPT_DIR}/run_fig_cpu_timed.sh" fig3 "${OUTDIR}"
