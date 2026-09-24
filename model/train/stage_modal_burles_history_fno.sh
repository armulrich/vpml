#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/armin/Documents/NYU/vpml/.venv/bin/python}"
SHARED_OUTPUT_ROOT="${VPML_SHARED_OUTPUT_ROOT:-/Users/armin/Documents/NYU/vpml/out_bench}"
SOURCE_CACHE="${TRAIN_LOW_MOMENT_REFERENCE_CACHE:-${SHARED_OUTPUT_ROOT}/reference_cache/interface_flux_landau_T120_Nx1024_Nv8192_M4096/e376aa1efa28e754b5f6}"
COMPACT_CACHE="${MODAL_LOW_MOMENT_COMPACT_CACHE:-${SHARED_OUTPUT_ROOT}/modal_staging/e376aa1efa28e754b5f6-low-moment-nx128}"
VOLUME_NAME="${MODAL_VPML_VOLUME:-vpml-low-moment-burles}"
REMOTE_ROOT="/reference/"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m model.train.prepare_modal_low_moment_cache \
  --source "${SOURCE_CACHE}" \
  --destination "${COMPACT_CACHE}" \
  --target-nx 128

if ! modal volume list --json | "${PYTHON_BIN}" -c \
  'import json,sys; name=sys.argv[1]; raise SystemExit(0 if any(row.get("Name") == name for row in json.load(sys.stdin)) else 1)' \
  "${VOLUME_NAME}"
then
  modal volume create --version=2 "${VOLUME_NAME}"
fi

REMOTE_MANIFEST="/reference/$(basename "${COMPACT_CACHE}")/COMPACT_CACHE_MANIFEST.json"
if modal volume ls "${VOLUME_NAME}" "${REMOTE_MANIFEST}" >/dev/null 2>&1
then
  LOCAL_HASH="$(shasum -a 256 "${COMPACT_CACHE}/COMPACT_CACHE_MANIFEST.json" | awk '{print $1}')"
  REMOTE_HASH="$(modal volume get "${VOLUME_NAME}" "${REMOTE_MANIFEST}" - | shasum -a 256 | awk '{print $1}')"
  if [[ "${LOCAL_HASH}" != "${REMOTE_HASH}" ]]; then
    echo "Remote compact-cache manifest differs; refusing to overwrite ${REMOTE_MANIFEST}" >&2
    exit 1
  fi
  echo "Modal cache already staged with matching manifest: ${REMOTE_MANIFEST}"
  exit 0
fi

modal volume put "${VOLUME_NAME}" "${COMPACT_CACHE}" "${REMOTE_ROOT}"
modal volume ls "${VOLUME_NAME}" "${REMOTE_MANIFEST}"
