#!/usr/bin/env bash
set -euo pipefail

ROOT="${VPML_ROOT:-/Users/armin/Documents/NYU/vpml-rank60-moe}"
PYTHON="${VPML_PYTHON:-/Users/armin/Documents/NYU/vpml/.venv/bin/python}"
OUT="${1:-out_bench/controlled_rank60_base_E20_20260910}"

cd "$ROOT"
exec "$PYTHON" -m model.train.balanced_family_coupled \
  --recipe-run out_bench/diagnostics/fresh_family_20260908/stationary_joint_E10 \
  --normalization out_bench/diagnostics/fresh_family_20260908/fixed_balanced_objective_screen.json \
  --outdir "$OUT" \
  --architecture shared \
  --epochs 20 \
  --seed 1729
