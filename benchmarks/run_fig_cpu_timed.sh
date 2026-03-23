#!/usr/bin/env bash
# Run a single benchmark (fig2, fig3, fig4, or linear_landau) on CPU and print wall-clock time.
# Usage: run_fig_cpu_timed.sh BENCHMARK OUTDIR [EXTRA_ARGS...]
#   BENCHMARK: fig2 | fig3 | fig4 | linear_landau
#   OUTDIR: output directory for the benchmark
set -euo pipefail

BENCHMARK="${1:?Usage: run_fig_cpu_timed.sh BENCHMARK OUTDIR [EXTRA_ARGS...]}"
OUTDIR="${2:?Usage: run_fig_cpu_timed.sh BENCHMARK OUTDIR [EXTRA_ARGS...]}"
shift 2

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

cd "${REPO_ROOT}"

echo "Running ${BENCHMARK} on ${JAX_PLATFORM_NAME} (outdir=${OUTDIR})..."
"${PYTHON_BIN}" -m benchmarks.fh_benchmarks_2412_07073_jax "${BENCHMARK}" --outdir "${OUTDIR}" "$@"
echo "Done."
