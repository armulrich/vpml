#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 BASE_RUN [OUTDIR]" >&2
  exit 2
fi

ROOT="${VPML_ROOT:-/Users/armin/Documents/NYU/vpml-rank60-moe}"
PYTHON="${VPML_PYTHON:-/Users/armin/Documents/NYU/vpml/.venv/bin/python}"
BASE_RUN="$1"
OUT="${2:-out_bench/controlled_rank60_moe_E10_20260910}"

cd "$ROOT"
exec "$PYTHON" -m model.train.balanced_family_coupled \
  --recipe-run out_bench/diagnostics/fresh_family_20260908/stationary_joint_E10 \
  --normalization out_bench/diagnostics/fresh_family_20260908/fixed_balanced_objective_screen.json \
  --outdir "$OUT" \
  --architecture september_experts \
  --parameter-metric expert_balanced \
  --shared-run "$BASE_RUN" \
  --epochs 10 \
  --seed 1729
