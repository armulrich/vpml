#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-True}"
export VPML_JAX_BACKEND="${VPML_JAX_BACKEND:-cpu}"

TEACHER_NX="${TEACHER_NX:-256}"
TEACHER_NV="${TEACHER_NV:-512}"
TEACHER_DT="${TEACHER_DT:-0.01}"
TEACHER_VMIN="${TEACHER_VMIN:--8.0}"
TEACHER_VMAX="${TEACHER_VMAX:-8.0}"
T_FINAL="${T_FINAL:-120.0}"
SNAPSHOT_TIMES="${SNAPSHOT_TIMES:-0,20,40,60,80,100,120}"
PROJECTION_QUADRATURE_NV_LIST="${PROJECTION_QUADRATURE_NV_LIST:-512,1024,2048,4096,8192,16384}"
REFERENCE_PROJECTION_NV="${REFERENCE_PROJECTION_NV:-16384}"
PROJECTION_ORDER="${PROJECTION_ORDER:-65}"
CUTOFFS="${CUTOFFS:-6,7,12,20,36,64}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${1:-${REPO_ROOT}/out_bench/projection_quadrature_convergence_teacherNv${TEACHER_NV}_${STAMP}}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m model.diagnostics.projection_quadrature_convergence \
  --outdir "${OUTDIR}" \
  --teacher-Nx "${TEACHER_NX}" \
  --teacher-Nv "${TEACHER_NV}" \
  --teacher-dt "${TEACHER_DT}" \
  --teacher-vmin "${TEACHER_VMIN}" \
  --teacher-vmax "${TEACHER_VMAX}" \
  --T-final "${T_FINAL}" \
  --snapshot-times "${SNAPSHOT_TIMES}" \
  --projection-quadrature-Nv-list "${PROJECTION_QUADRATURE_NV_LIST}" \
  --reference-projection-Nv "${REFERENCE_PROJECTION_NV}" \
  --projection-order "${PROJECTION_ORDER}" \
  --cutoffs "${CUTOFFS}"

echo
echo "Projection quadrature diagnostic complete."
echo "  physical teacher Nv: ${TEACHER_NV}"
echo "  projection grids:    ${PROJECTION_QUADRATURE_NV_LIST}"
echo "  finest tested grid:  ${REFERENCE_PROJECTION_NV}"
echo "  output:              ${OUTDIR}"
