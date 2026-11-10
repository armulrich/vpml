# shellcheck shell=bash disable=SC2034
# Compare the new 4096-point trajectory with the verified, completed 2048-point one.
: "${VPML_REFINEMENT_BASELINE_ROOT:?Set the existing diagnostic output directory containing finer_x}"
VPML_REFINEMENT_CONFIG="$VPML_CODE_ROOT/scripts/local/gaussian_spatial_refinement_v2.json"
VPML_REFINEMENT_VARIANTS=(spatial_4096)
VPML_REFINEMENT_MAX_SECONDS=10800
