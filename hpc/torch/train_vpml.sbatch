#!/bin/bash
#SBATCH --job-name=vpml-train
#SBATCH --nodes=1
#SBATCH --account=torch_pr_292_general
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/vpml/logs/%x_%A.out
#SBATCH --error=/scratch/%u/vpml/logs/%x_%A.err
#
# Full interface-flux train + eval on Torch GPU.
#
# Repo checkout:
#   Default /scratch/${USER}/vpml, else SLURM submit dir.
#   Override: VPML_REPO_ROOT=/path/to/vpml
#
# Python:
#   Override PYTHON_BIN, else first found mrx_gpu / mrx env python.
#
# Optional env (passed through to run_fh_interface_flux_rollout.sh):
#   OUTDIR, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_ROLLOUT_HORIZON,
#   RUN_TRAIN, RUN_EVAL, TEACHER_PROJECTION_NV, etc.
#
# Submit:
#   mkdir -p /scratch/$USER/vpml/logs
#   cd /scratch/$USER/vpml && git checkout vpml_js
#   sbatch hpc/torch/train_vpml.sbatch
#   OUTDIR=$SCRATCH/my-run TRAIN_EPOCHS=100 sbatch hpc/torch/train_vpml.sbatch

set -euo pipefail

mkdir -p "/scratch/${USER}/vpml/logs"

REPO_ROOT="${VPML_REPO_ROOT:-}"
if [[ -z "${REPO_ROOT}" ]]; then
  if [[ -d "/scratch/${USER}/vpml" ]]; then
    REPO_ROOT="/scratch/${USER}/vpml"
  else
    REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
  fi
fi
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "vpml repo not found: set VPML_REPO_ROOT or create /scratch/${USER}/vpml" >&2
  exit 1
fi
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  for _p in \
    "/scratch/${USER}/.conda/envs/mrx_gpu/bin/python" \
    "${HOME}/.conda/envs/mrx_gpu/bin/python" \
    "/scratch/${USER}/.conda/envs/mrx/bin/python" \
    "${HOME}/.conda/envs/mrx/bin/python"
  do
    if [[ -x "${_p}" ]]; then
      PYTHON_BIN="${_p}"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "No Python found. Install mrx_gpu or set PYTHON_BIN." >&2
  exit 1
fi
export PYTHON="${PYTHON_BIN}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/hpc/torch/env_vpml.sh"

echo "[vpml] REPO_ROOT=${REPO_ROOT}"
echo "[vpml] PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import jax; print('[vpml] jax devices:', jax.devices())"

OUTDIR="${OUTDIR:-${REPO_ROOT}/out_bench/fh_interface_flux_H${TRAIN_ROLLOUT_HORIZON:-128}}"
exec bash "${REPO_ROOT}/model/train/run_fh_interface_flux_rollout.sh" "${OUTDIR}"
