#!/usr/bin/env bash
# Log Slurm node + GPU + JAX device info for vpml timing runs on Torch.
#
# Usage:
#   source hpc/torch/log_torch_hardware.sh
#   vpml_log_hardware >> "${TIMING_OUT_BASE}/hardware.log"
#   vpml_hardware_csv_fields   # prints CSV-safe single-line summary
#
# Requires: PYTHON_BIN or python on PATH for JAX probe (optional).

set -euo pipefail

vpml_log_hardware() {
  local py="${PYTHON_BIN:-${PYTHON:-python}}"
  echo "=== vpml hardware snapshot $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "hostname=$(hostname -s 2>/dev/null || hostname)"
  echo "user=${USER:-}"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-}"
  echo "slurm_partition=${SLURM_JOB_PARTITION:-}"
  echo "slurm_node=${SLURMD_NODENAME:-${SLURM_NODELIST:-}}"
  echo "slurm_cpus=${SLURM_CPUS_ON_NODE:-}"
  echo "slurm_mem_mb=${SLURM_MEM_PER_NODE:-}"
  echo "slurm_gres=${SLURM_JOB_GRES:-}"
  echo "slurm_tres=${SLURM_JOB_TRES_ALLOCATE:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "vpml_jax_backend=${VPML_JAX_BACKEND:-}"
  echo "jax_enable_x64=${JAX_ENABLE_X64:-}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi -L ---"
    nvidia-smi -L 2>/dev/null || true
    echo "--- nvidia-smi (name, driver, mem) ---"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap \
      --format=csv,noheader 2>/dev/null || true
  else
    echo "nvidia-smi=not_found"
  fi

  if [[ -x "${py}" ]] || command -v "${py}" >/dev/null 2>&1; then
    echo "--- jax ---"
    "${py}" - <<'PY' 2>&1 || echo "jax_probe_failed"
import os
try:
    import jax
    print("jax_version=", jax.__version__)
    print("jax_devices=", [str(d) for d in jax.devices()])
    print("jax_default_backend=", jax.default_backend())
except Exception as exc:
    print("jax_import_error=", exc)
PY
  fi
  echo "=== end hardware snapshot ==="
}

# One CSV-safe line: gpu_names|jax_devices|slurm_node|partition
vpml_hardware_csv_fields() {
  local py="${PYTHON_BIN:-${PYTHON:-python}}"
  local gpu_names="cpu"
  local jax_devs="none"
  local node="${SLURMD_NODENAME:-${SLURM_NODELIST:-$(hostname -s 2>/dev/null || hostname)}}"
  local part="${SLURM_JOB_PARTITION:-unknown}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ';' - || echo unknown_gpu)"
    if [[ -z "${gpu_names// /}" ]]; then
      gpu_names="no_gpu"
    fi
  fi

  if [[ -x "${py}" ]] || command -v "${py}" >/dev/null 2>&1; then
    jax_devs="$("${py}" - <<'PY' 2>/dev/null || echo jax_error
import jax
print(";".join(str(d) for d in jax.devices()))
PY
)"
  fi

  # Escape commas for CSV (replace with semicolon)
  gpu_names="${gpu_names//,/;}"
  jax_devs="${jax_devs//,/;}"
  node="${node//,/;}"
  part="${part//,/;}"

  printf '%s|%s|%s|%s' "${gpu_names}" "${jax_devs}" "${node}" "${part}"
}
