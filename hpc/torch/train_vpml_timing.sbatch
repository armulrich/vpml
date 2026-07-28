#!/bin/bash
#SBATCH --job-name=vpml-time
#SBATCH --nodes=1
#SBATCH --account=torch_pr_292_general
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/vpml/logs/%x_%A.out
#SBATCH --error=/scratch/%u/vpml/logs/%x_%A.err
#
# GPU wall-clock timing for interface-flux training on Torch (vpml_js / new main).
#
# Sweeps rollout horizon H by default (comma-separated):
#   HORIZON_TIMING_LIST=1,128,256
#
# Train-only timing (skip eval, faster):
#   TIMING_RUN_EVAL=0
#
# Short smoke test:
#   TRAIN_EPOCHS=2 TRAIN_STEPS_PER_EPOCH=2 HORIZON_TIMING_LIST=128 TIMING_RUN_EVAL=0 \
#     sbatch hpc/torch/train_vpml_timing.sbatch
#
# Outputs:
#   ${TIMING_OUT_BASE}/hardware.log
#   ${TIMING_CSV}   one row per horizon with wall time + GPU fields
#
# Submit:
#   mkdir -p /scratch/$USER/vpml/logs
#   cd /scratch/$USER/vpml && git checkout vpml_js
#   sbatch hpc/torch/train_vpml_timing.sbatch

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
  echo "No Python found. Set PYTHON_BIN or install mrx_gpu (CUDA JAX)." >&2
  exit 1
fi
export PYTHON="${PYTHON_BIN}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/hpc/torch/env_vpml.sh"

SWEEP_SCRIPT="${REPO_ROOT}/model/train/run_fh_interface_flux_rollout.sh"
if [[ ! -x "${SWEEP_SCRIPT}" ]]; then
  echo "Missing trainer script: ${SWEEP_SCRIPT}" >&2
  exit 1
fi

HORIZON_TIMING_LIST="${HORIZON_TIMING_LIST:-128}"
TIMING_RUN_EVAL="${TIMING_RUN_EVAL:-0}"
TIMING_JOB="${SLURM_JOB_ID:-local}"
TIMING_OUT_BASE="${TIMING_OUT_BASE:-${REPO_ROOT}/out_bench/interface_flux_timing_${TIMING_JOB}}"
TIMING_CSV="${TIMING_CSV:-${TIMING_OUT_BASE}/horizon_wall_times_${TIMING_JOB}.csv}"
mkdir -p "${TIMING_OUT_BASE}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/hpc/torch/log_torch_hardware.sh"

HARDWARE_LOG="${TIMING_OUT_BASE}/hardware.log"
vpml_log_hardware | tee "${HARDWARE_LOG}"

IFS='|' read -r HW_GPU HW_JAX HW_NODE HW_PART <<< "$(vpml_hardware_csv_fields)"

echo "[vpml-timing] REPO_ROOT=${REPO_ROOT}"
echo "[vpml-timing] PYTHON_BIN=${PYTHON_BIN}"
echo "[vpml-timing] TIMING_OUT_BASE=${TIMING_OUT_BASE}"
echo "[vpml-timing] TIMING_CSV=${TIMING_CSV}"
echo "[vpml-timing] HORIZON_TIMING_LIST=${HORIZON_TIMING_LIST}"
echo "[vpml-timing] TIMING_RUN_EVAL=${TIMING_RUN_EVAL}"
echo "[vpml-timing] gpu=${HW_GPU}"
echo "[vpml-timing] jax=${HW_JAX}"
echo "[vpml-timing] node=${HW_NODE} partition=${HW_PART}"

{
  echo "horizon,wall_seconds,run_eval,slurm_job_id,gpu_name,jax_devices,slurm_node,partition,hostname,utc_end"
} > "${TIMING_CSV}"

IFS=',' read -r -a H_ARR <<< "${HORIZON_TIMING_LIST}"
for H_RAW in "${H_ARR[@]}"; do
  horizon="$(echo "${H_RAW}" | tr -d '[:space:]')"
  if [[ -z "${horizon}" ]]; then
    continue
  fi
  ONE_OUT="${TIMING_OUT_BASE}/H${horizon}"
  echo "[vpml-timing] --- H=${horizon} OUTDIR=${ONE_OUT} ---"
  t0=$(date +%s)
  TRAIN_ROLLOUT_HORIZON="${horizon}" \
    RUN_EVAL="${TIMING_RUN_EVAL}" \
    bash "${SWEEP_SCRIPT}" "${ONE_OUT}"
  t1=$(date +%s)
  wall=$((t1 - t0))
  utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[vpml-timing] H=${horizon} wall_seconds=${wall}"
  echo "${horizon},${wall},${TIMING_RUN_EVAL},${TIMING_JOB},${HW_GPU},${HW_JAX},${HW_NODE},${HW_PART},$(hostname -s 2>/dev/null || hostname),${utc_end}" >> "${TIMING_CSV}"
done

echo "[vpml-timing] Wrote ${TIMING_CSV}"
echo "[vpml-timing] Wrote ${HARDWARE_LOG}"
cat "${TIMING_CSV}"
