#!/usr/bin/env bash
set -euo pipefail

SHARED_OUTPUT_ROOT="${VPML_SHARED_OUTPUT_ROOT:-/Users/armin/Documents/NYU/vpml/out_bench}"
VOLUME_NAME="${MODAL_VPML_VOLUME:-vpml-low-moment-burles}"
REMOTE_RUN="/runs/history_fno_random1729_full_anchor_E1000"
LOCAL_DESTINATION="${1:-${SHARED_OUTPUT_ROOT}/modal_downloads/history_fno_random1729_full_anchor_E1000}"

mkdir -p "$(dirname "${LOCAL_DESTINATION}")"
modal volume get --force "${VOLUME_NAME}" "${REMOTE_RUN}" "${LOCAL_DESTINATION}"
