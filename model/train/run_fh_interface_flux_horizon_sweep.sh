#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

HORIZONS="${1:-1,128,256}"
OUTROOT="${2:-${REPO_ROOT}/out_bench/interface_flux_horizon_sweep}"

IFS=',' read -r -a VALUES <<< "${HORIZONS}"
for raw_value in "${VALUES[@]}"; do
  horizon="$(echo "${raw_value}" | tr -d '[:space:]')"
  if [[ -z "${horizon}" || ! "${horizon}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid rollout horizon in CSV: ${raw_value@Q}" >&2
    exit 1
  fi
  echo "[interface-flux-sweep] H=${horizon}"
  TRAIN_ROLLOUT_HORIZON="${horizon}" \
    "${SCRIPT_DIR}/run_fh_interface_flux_rollout.sh" \
    "${OUTROOT}/H${horizon}"
done
