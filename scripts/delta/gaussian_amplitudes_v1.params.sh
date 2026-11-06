# shellcheck shell=bash disable=SC2034
# Sourced settings consumed by the launcher, not standalone commands.
# Reviewed settings for this run. Scientific values live in the adjacent JSON.
# Changing this version requires reviewing the manifest and run provenance.
VPML_PARAMETERS_FILE="$VPML_CODE_ROOT/scripts/delta/gaussian_amplitudes_v1.json"
VPML_SLURM_ACCOUNT=bhvw-delta-gpu
VPML_SLURM_PARTITION=gpuA100x4
VPML_SLURM_CPUS=16
VPML_SLURM_MEMORY=60G
VPML_SLURM_TIME=01:00:00
VPML_MAX_SECONDS=3300
