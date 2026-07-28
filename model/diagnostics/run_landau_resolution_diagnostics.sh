#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-True}"
export VPML_JAX_BACKEND="${VPML_JAX_BACKEND:-cpu}"

TEACHER_NX="${TEACHER_NX:-256}"
PHYSICAL_NV_LIST="${PHYSICAL_NV_LIST:-512,1024,2048,4096}"
PROJECTION_SOURCE_PHYSICAL_NV="${PROJECTION_SOURCE_PHYSICAL_NV:-${PHYSICAL_NV_LIST##*,}}"
COARSE_TEACHER_SNAPSHOTS="${COARSE_TEACHER_SNAPSHOTS:-}"
TEACHER_DT="${TEACHER_DT:-0.01}"
TEACHER_VMIN="${TEACHER_VMIN:--8.0}"
TEACHER_VMAX="${TEACHER_VMAX:-8.0}"
T_FINAL="${T_FINAL:-120.0}"
SNAPSHOT_TIMES="${SNAPSHOT_TIMES:-0,20,40,60,80,100,120}"
PROJECTION_QUADRATURE_NV_LIST="${PROJECTION_QUADRATURE_NV_LIST:-2048,4096,8192,16384}"
REFERENCE_PROJECTION_NV="${REFERENCE_PROJECTION_NV:-16384}"
PROJECTION_ORDER="${PROJECTION_ORDER:-65}"
CUTOFFS="${CUTOFFS:-6,7,12,20,36,64}"
RELATIVE_TOLERANCE="${RELATIVE_TOLERANCE:-0.01}"
LINEAR_EPS="${LINEAR_EPS:-0.01}"
LINEAR_MODES="${LINEAR_MODES:-0.5,1.0,1.5,2.0}"
LINEAR_SEED="${LINEAR_SEED:-0}"
WEAK_EPS="${WEAK_EPS:-0.1}"
STRONG_EPS="${STRONG_EPS:-0.5}"
NONLINEAR_K0="${NONLINEAR_K0:-0.5}"
POISSON_SIGN="${POISSON_SIGN:-1.0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${1:-${REPO_ROOT}/out_bench/landau_resolution_T${T_FINAL}_${STAMP}}"
PHYSICAL_OUTDIR="${OUTDIR}/physical_velocity_grid"
PROJECTION_OUTDIR="${OUTDIR}/projection_quadrature"
TEACHER_SNAPSHOTS="${PHYSICAL_OUTDIR}/physical_teacher_nv${PROJECTION_SOURCE_PHYSICAL_NV}_snapshots.npz"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing diagnostic directory: ${OUTDIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

PHYSICAL_REUSE_ARGS=()
if [[ -n "${COARSE_TEACHER_SNAPSHOTS}" ]]; then
  PHYSICAL_REUSE_ARGS+=(--coarse-teacher-snapshots "${COARSE_TEACHER_SNAPSHOTS}")
fi

echo "[landau-resolution] [1/3] Physical velocity-grid self-convergence"
"${PYTHON_BIN}" -m model.diagnostics.physical_velocity_grid_convergence \
  --outdir "${PHYSICAL_OUTDIR}" \
  --teacher-Nx "${TEACHER_NX}" \
  --physical-Nv-list "${PHYSICAL_NV_LIST}" \
  --snapshot-artifact-Nv "${PROJECTION_SOURCE_PHYSICAL_NV}" \
  --teacher-dt "${TEACHER_DT}" \
  --teacher-vmin "${TEACHER_VMIN}" \
  --teacher-vmax "${TEACHER_VMAX}" \
  --T-final "${T_FINAL}" \
  --snapshot-times "${SNAPSHOT_TIMES}" \
  --relative-tolerance "${RELATIVE_TOLERANCE}" \
  --linear-eps "${LINEAR_EPS}" \
  --linear-modes "${LINEAR_MODES}" \
  --linear-seed "${LINEAR_SEED}" \
  --weak-eps "${WEAK_EPS}" \
  --strong-eps "${STRONG_EPS}" \
  --nonlinear-k0 "${NONLINEAR_K0}" \
  --poisson-sign "${POISSON_SIGN}" \
  "${PHYSICAL_REUSE_ARGS[@]}"

echo "[landau-resolution] [2/3] Spline-to-Hermite projection-quadrature self-convergence"
"${PYTHON_BIN}" -m model.diagnostics.projection_quadrature_convergence \
  --outdir "${PROJECTION_OUTDIR}" \
  --teacher-snapshots "${TEACHER_SNAPSHOTS}" \
  --snapshot-times "${SNAPSHOT_TIMES}" \
  --projection-quadrature-Nv-list "${PROJECTION_QUADRATURE_NV_LIST}" \
  --reference-projection-Nv "${REFERENCE_PROJECTION_NV}" \
  --projection-order "${PROJECTION_ORDER}" \
  --cutoffs "${CUTOFFS}"

echo "[landau-resolution] [3/3] Combined parameter recommendation"
"${PYTHON_BIN}" -m model.diagnostics.landau_resolution_report \
  --physical-json "${PHYSICAL_OUTDIR}/physical_velocity_grid_convergence.json" \
  --projection-json "${PROJECTION_OUTDIR}/projection_quadrature_convergence.json" \
  --outdir "${OUTDIR}"

echo
echo "Landau resolution diagnostics complete."
echo "  physical grids:      ${PHYSICAL_NV_LIST}"
echo "  projection source:   physical Nv=${PROJECTION_SOURCE_PHYSICAL_NV}"
if [[ -n "${COARSE_TEACHER_SNAPSHOTS}" ]]; then
  echo "  reused coarse grid:  ${COARSE_TEACHER_SNAPSHOTS}"
fi
echo "  projection grids:    ${PROJECTION_QUADRATURE_NV_LIST}"
echo "  T final:             ${T_FINAL}"
echo "  output:              ${OUTDIR}"
echo "  recommendation:      ${OUTDIR}/README.md"
