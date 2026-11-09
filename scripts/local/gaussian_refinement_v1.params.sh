# shellcheck shell=bash disable=SC2034
# Numerical values and the exact original IC are in the adjacent JSON.
VPML_REFINEMENT_CONFIG="$VPML_CODE_ROOT/scripts/local/gaussian_refinement_v1.json"
VPML_REFINEMENT_VARIANTS=(baseline finer_x finer_v half_dt)
# Per invocation, with resumable checkpoints between blocks.
VPML_REFINEMENT_MAX_SECONDS=10800
