#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON:-python}"
fi

OUTDIR="${1:-${REPO_ROOT}/out_bench}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export VPML_JAX_BACKEND="${VPML_JAX_BACKEND:-auto}"

CHECKPOINT="${CHECKPOINT:-${OUTDIR}/interface_closure.npz}"
DATASET_CACHE="${DATASET_CACHE:-${OUTDIR}/interface_closure_dataset.npz}"
LOSS_PLOT="${LOSS_PLOT:-${OUTDIR}/interface_closure.loss.png}"
EVAL_OUTDIR="${EVAL_OUTDIR:-${OUTDIR}/eval}"
EVAL_BUNDLES="${EVAL_BUNDLES:-heldout_landau,benchmark_rollouts}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_STANDARD_BENCHMARKS="${RUN_STANDARD_BENCHMARKS:-1}"
RUN_FIG10="${RUN_FIG10:-1}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"

# Default training/model parameters for the physical-grid Landau teacher.
NV_TARGETS="${NV_TARGETS:-6,8,10,12,20,40,80,160,300}"
NM="${NM:-6}"
HIDDEN_WIDTH="${HIDDEN_WIDTH:-128}"
RES_BLOCKS="${RES_BLOCKS:-2}"
EPOCHS="${EPOCHS:-300}"
LR="${LR:-1e-3}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_EVERY="${LOG_EVERY:-10}"
REGIMES="${REGIMES:-linear_landau,nonlinear_landau_weak,nonlinear_landau_strong}"

TEACHER_NX="${TEACHER_NX:-256}"
TEACHER_NV="${TEACHER_NV:-512}"
TEACHER_DT="${TEACHER_DT:-0.01}"
TEACHER_VMIN="${TEACHER_VMIN:--8.0}"
TEACHER_VMAX="${TEACHER_VMAX:-8.0}"
TEACHER_PROJ_NV="${TEACHER_PROJ_NV:-}"

LINEAR_T="${LINEAR_T:-20.0}"
LINEAR_EPS="${LINEAR_EPS:-1e-2}"
LINEAR_NUM_SAMPLES="${LINEAR_NUM_SAMPLES:-8}"
LINEAR_HISTORY_STRIDE="${LINEAR_HISTORY_STRIDE:-2}"

NONLINEAR_T="${NONLINEAR_T:-20.0}"
NONLINEAR_HISTORY_STRIDE="${NONLINEAR_HISTORY_STRIDE:-20}"
WEAK_EPS="${WEAK_EPS:-0.05,0.1}"
STRONG_EPS="${STRONG_EPS:-0.25,0.5}"

mkdir -p "${OUTDIR}"
cd "${REPO_ROOT}"

TRAIN_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --dataset-cache "${DATASET_CACHE}"
  --loss-plot "${LOSS_PLOT}"
  --Nv-targets "${NV_TARGETS}"
  --Nm "${NM}"
  --hidden-width "${HIDDEN_WIDTH}"
  --res-blocks "${RES_BLOCKS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --grad-clip "${GRAD_CLIP}"
  --log-every "${LOG_EVERY}"
  --regimes "${REGIMES}"
  --teacher-Nx "${TEACHER_NX}"
  --teacher-Nv "${TEACHER_NV}"
  --teacher-dt "${TEACHER_DT}"
  --teacher-vmin "${TEACHER_VMIN}"
  --teacher-vmax "${TEACHER_VMAX}"
  --linear-T "${LINEAR_T}"
  --linear-eps "${LINEAR_EPS}"
  --linear-num-samples "${LINEAR_NUM_SAMPLES}"
  --linear-history-stride "${LINEAR_HISTORY_STRIDE}"
  --nonlinear-T "${NONLINEAR_T}"
  --nonlinear-history-stride "${NONLINEAR_HISTORY_STRIDE}"
  --weak-eps "${WEAK_EPS}"
  --strong-eps "${STRONG_EPS}"
)
if [[ -n "${TEACHER_PROJ_NV}" ]]; then
  TRAIN_ARGS+=(--teacher-proj-Nv "${TEACHER_PROJ_NV}")
fi

EVAL_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --outdir "${EVAL_OUTDIR}"
  --bundles "${EVAL_BUNDLES}"
  --teacher-Nx "${TEACHER_NX}"
  --teacher-Nv "${TEACHER_NV}"
  --teacher-dt "${TEACHER_DT}"
  --teacher-vmin "${TEACHER_VMIN}"
  --teacher-vmax "${TEACHER_VMAX}"
)
if [[ -n "${EVAL_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_EVAL_ARGS=( ${EVAL_EXTRA_ARGS} )
  EVAL_ARGS+=("${EXTRA_EVAL_ARGS[@]}")
fi

echo "[1/4] Generating cached data if needed and training the shared interface closure"
"${PYTHON_BIN}" benchmarks/fh_ml_tail_closure_train_jax.py "${TRAIN_ARGS[@]}"

echo "[2/4] Running post-train metrics evaluation"
if [[ "${RUN_EVAL}" != "0" ]]; then
  "${PYTHON_BIN}" benchmarks/eval.py "${EVAL_ARGS[@]}"
else
  echo "Skipping eval.py because RUN_EVAL=${RUN_EVAL}"
fi

echo "[3/4] Running the standard benchmark suite plus learned linear Landau rollout"
if [[ "${RUN_STANDARD_BENCHMARKS}" != "0" ]]; then
  LEARNED_CHECKPOINT="${CHECKPOINT}" "${SCRIPT_DIR}/run_all_benchmarks.sh" "${OUTDIR}"
else
  echo "Skipping benchmark suite because RUN_STANDARD_BENCHMARKS=${RUN_STANDARD_BENCHMARKS}"
fi

echo "[4/4] Rendering nonlinear Landau phase-space figures"
if [[ "${RUN_FIG10}" != "0" ]]; then
  "${PYTHON_BIN}" -m benchmarks.fh_benchmarks_2412_07073_jax \
    fig10 \
    --outdir "${OUTDIR}"
  "${PYTHON_BIN}" -m benchmarks.fh_benchmarks_2412_07073_jax \
    fig10_learned_comparison \
    --outdir "${OUTDIR}" \
    --learned-checkpoint "${CHECKPOINT}"
else
  echo "Skipping fig10 render because RUN_FIG10=${RUN_FIG10}"
fi

cat <<EOF

Done.

Artifacts:
  checkpoint:      ${CHECKPOINT}
  dataset cache:   ${DATASET_CACHE}
  loss plot:       ${LOSS_PLOT}
  metrics:         ${CHECKPOINT%.npz}.metrics.npz
  eval:            ${EVAL_OUTDIR}
  benchmarks:      ${OUTDIR}

Open these first:
  ${LOSS_PLOT}
  ${EVAL_OUTDIR}/summary.json
  ${OUTDIR}/linear_landau_comparison.png
  ${OUTDIR}/fig10_nonlinear_landau_phase_space.png
  ${OUTDIR}/fig10_learned_vs_nonlocal_phase_space.png
EOF
