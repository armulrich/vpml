#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="${1:?usage: run_dynamic_low_rank_probe.sh OUTDIR}"
cd "${ROOT_DIR}"

PYTHON_BIN="${VPML_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
exec "${PYTHON_BIN}" -m model.train.dynamic_low_rank_probe \
  --reference-cache "${LOW_RANK_REFERENCE_CACHE:?set LOW_RANK_REFERENCE_CACHE}" \
  --outdir "${OUTDIR}" \
  --case-ids "${LOW_RANK_CASE_IDS:-linear_landau_ic09,nonlinear_landau_strong_ic12}" \
  --start-times "${LOW_RANK_START_TIMES:-100,60}" \
  --ranks "${LOW_RANK_RANKS:-16,32,48}" \
  --horizon "${LOW_RANK_HORIZON:-10}" \
  --cadence-steps "${LOW_RANK_CADENCE_STEPS:-10}" \
  --nx "${LOW_RANK_NX:-256}" \
  --high-nv "${LOW_RANK_HIGH_NV:-1024}" \
  --coarse-nv "${LOW_RANK_COARSE_NV:-512}" \
  --oversample "${LOW_RANK_OVERSAMPLE:-8}" \
  --power-iterations "${LOW_RANK_POWER_ITERATIONS:-1}" \
  --seed "${LOW_RANK_SEED:-1729}" \
  --rank-projection-schedule "${LOW_RANK_PROJECTION_SCHEDULE:-repeated}"
