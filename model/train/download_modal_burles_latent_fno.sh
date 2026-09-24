#!/usr/bin/env bash
set -euo pipefail

SHARED_OUTPUT_ROOT="${VPML_SHARED_OUTPUT_ROOT:-/Users/armin/Documents/NYU/vpml/out_bench}"
VOLUME_NAME="${MODAL_VPML_VOLUME:-vpml-low-moment-burles}"
REMOTE_RUN="/runs/history_latent6_random1729_full_anchor_resumable_E1000"
LOCAL_DESTINATION="${1:-${SHARED_OUTPUT_ROOT}/modal_downloads/history_latent6_random1729_full_anchor_resumable_E1000}"

if [[ -e "${LOCAL_DESTINATION}" ]]; then
  echo "Refusing to overwrite existing destination: ${LOCAL_DESTINATION}" >&2
  exit 1
fi
mkdir -p "$(dirname "${LOCAL_DESTINATION}")"
REMOTE_BASENAME="$(basename "${REMOTE_RUN}")"
if [[ "$(basename "${LOCAL_DESTINATION}")" != "${REMOTE_BASENAME}" ]]; then
  echo "Local destination basename must match remote run basename: ${REMOTE_BASENAME}" >&2
  exit 1
fi
modal volume get "${VOLUME_NAME}" "${REMOTE_RUN}" "$(dirname "${LOCAL_DESTINATION}")"
