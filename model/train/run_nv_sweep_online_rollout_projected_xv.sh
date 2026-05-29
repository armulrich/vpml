#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_PROJECTED_XV_METRIC="${TRAIN_PROJECTED_XV_METRIC:-gram_riesz}"

exec "${SCRIPT_DIR}/run_fh_online_projected_xv_rollout.sh" "$@"
