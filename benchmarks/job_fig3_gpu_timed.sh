#!/bin/bash
#SBATCH --job-name=fig3_gpu
#SBATCH --output=fig3_gpu_%A.out
#SBATCH --error=fig3_gpu_%A.err
#SBATCH --account=torch_pr_292_general
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# ============================================================================
# Run Fig. 3 benchmark (response-function) on GPU and print wall-clock time.
# ============================================================================
#
# Usage (from vpml repo root):
#   sbatch benchmarks/job_fig3_gpu_timed.sh
#   sbatch benchmarks/job_fig3_gpu_timed.sh /path/to/outdir
#
# Requires a conda env with vpml deps + JAX GPU (e.g. jax[cuda12] or jax for GPU).
# On Torch you may need: --partition=gpu or a different --gres (e.g. gpu:a100:1).
# ============================================================================

CONDA_ENV="${CONDA_ENV:-vpml}"

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
    echo "  Run from vpml directory or set SLURM_SUBMIT_DIR to vpml root."
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
    VPML_ENV_PATH=""
fi

if [ -z "$VPML_ENV_PATH" ] || [ ! -d "$VPML_ENV_PATH" ]; then
    echo "ERROR: conda env '${CONDA_ENV}' not found"
    echo "  Create with: conda create -n ${CONDA_ENV} python=3.11 && conda activate ${CONDA_ENV} && pip install numpy jax matplotlib scipy"
    exit 1
fi
export PYTHON="${VPML_ENV_PATH}/bin/python"
export JAX_PLATFORM_NAME=gpu
export JAX_PLATFORMS=gpu
export PYTHONUNBUFFERED=1

OUT_DIR="${1:-${VPML_ROOT}/out_bench}"
mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Fig. 3 benchmark (GPU, timed)"
echo "Repo: $VPML_ROOT"
echo "Python: $PYTHON"
echo "Output: $OUT_DIR"
echo "=============================================="

time bash "$VPML_ROOT/benchmarks/run_fig3_cpu_timed.sh" "$OUT_DIR"
EXIT_CODE=$?

echo ""
echo "Job finished at $(date)"
exit $EXIT_CODE
