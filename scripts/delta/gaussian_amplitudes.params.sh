# shellcheck shell=bash disable=SC2034
# Full 1200-IC dataset at Nx=2048; this invocation selects its first five cases.
VPML_PARAMETERS_FILE="$VPML_CODE_ROOT/scripts/delta/gaussian_amplitudes.json"
VPML_SLURM_ACCOUNT=bhvw-delta-gpu
VPML_SLURM_PARTITION=gpuA100x4
VPML_SLURM_CPUS=16
VPML_SLURM_MEMORY=60G
VPML_SLURM_TIME=01:00:00
VPML_MAX_SECONDS=3300
VPML_SHUTDOWN_MARGIN_SECONDS=180
VPML_SLURM_JOB_NAME=vpml-gaussian
VPML_STOP_ON_FAILURE=1
VPML_CASE_IDS=(
    case_0f256f2e1b1d12274df7d726
    case_85d6c8ae79fcf33b5d4878d4
    case_398a3fbaeed975eec57b46d0
    case_047207d6581b94d72a9bd823
    case_8ef5689ddc47917db7b30b97
)
