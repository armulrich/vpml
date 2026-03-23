#!/bin/bash
#SBATCH --job-name=fig4_cpu
#SBATCH --output=fig4_cpu_%A.out
#SBATCH --error=fig4_cpu_%A.err
#SBATCH --account=torch_pr_292_general
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

# ============================================================================
# Run Fig. 4 benchmark (eigenvalue scan) on CPU and print wall-clock time.
# ============================================================================
#
# Usage (from vpml repo root):
#   sbatch benchmarks/job_fig4_cpu_timed.sh
#   sbatch benchmarks/job_fig4_cpu_timed.sh /path/to/outdir
# Optional: NV=24 sbatch benchmarks/job_fig4_cpu_timed.sh  (default Nv=20)
# ============================================================================

CONDA_ENV="${CONDA_ENV:-vpml}"
NV="${NV:-20}"

# --- Find repo root ---
if [ -n "${SLURM_SUBMIT_DIR}" ] && [ -d "${SLURM_SUBMIT_DIR}/benchmarks" ]; then
    VPML_ROOT="${SLURM_SUBMIT_DIR}"
elif [ -d "$(pwd)/benchmarks" ]; then
    VPML_ROOT="$(pwd)"
elif [ -d "$HOME/vpml/benchmarks" ]; then
    VPML_ROOT="$HOME/vpml"
elif [ -d "/scratch/$USER/vpml/benchmarks" ]; then
    VPML_ROOT="/scratch/$USER/vpml"
else
    echo "ERROR: Cannot find vpml repo (benchmarks/ not found)"
    exit 1
fi
cd "$VPML_ROOT" || { echo "ERROR: Cannot cd to $VPML_ROOT"; exit 1; }

# --- Environment: conda ---
module purge
module load anaconda3/2025.06
if [ -f "/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh" ]; then
    source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
elif [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
else
    source ~/.bashrc
fi
eval "$(conda shell.bash hook)"

if [ -d "/scratch/$USER/.conda/envs/${CONDA_ENV}" ]; then
    VPML_ENV_PATH="/scratch/$USER/.conda/envs/${CONDA_ENV}"
elif VPML_ENV_PATH=$(conda env list | awk -v e="$CONDA_ENV" '$1==e {print $NF; exit}') && [ -n "$VPML_ENV_PATH" ] && [ -d "$VPML_ENV_PATH" ]; then
    :
elif [ -d "$HOME/.conda/envs/${CONDA_ENV}" ]; then
    VPML_ENV_PATH="$HOME/.conda/envs/${CONDA_ENV}"
else
    echo "ERROR: conda env '${CONDA_ENV}' not found"; exit 1
fi
export PYTHON="${VPML_ENV_PATH}/bin/python"
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

OUT_DIR="${1:-${VPML_ROOT}/out_bench}"
mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Fig. 4 benchmark (CPU, timed), Nv=$NV"
echo "Repo: $VPML_ROOT"
echo "Python: $PYTHON"
echo "Output: $OUT_DIR"
echo "=============================================="

time bash "$VPML_ROOT/benchmarks/run_fig_cpu_timed.sh" fig4 "$OUT_DIR" --Nv "$NV"
EXIT_CODE=$?
echo ""
echo "Job finished at $(date)"
exit $EXIT_CODE
