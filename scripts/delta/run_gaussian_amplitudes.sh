#!/usr/bin/env bash
# Every operation is explicit. Only the "submit" action can submit a real job.
set -euo pipefail
umask 077

usage() {
    printf 'Usage: bash %s {prepare|validate|status|test-submit|submit}\n' "$0" >&2
    printf 'Required: VPML_DATA_ROOT and VPML_PYTHON, both absolute paths.\n' >&2
    printf 'Optional: VPML_RUN_PARAMS selects an absolute parameter file; the adjacent gaussian_amplitudes.params.sh is the default.\n' >&2
}
if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi
action=$1
case $action in
    prepare|validate|status|test-submit|submit) ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
esac

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
VPML_CODE_ROOT=${VPML_CODE_ROOT:-$(cd -- "$script_directory/../.." && pwd -P)}
if [[ $VPML_CODE_ROOT != /* || ! -f $VPML_CODE_ROOT/scripts/gaussian_dataset.py ]]; then
    printf 'VPML_CODE_ROOT must be an absolute path to the reviewed VPML checkout.\n' >&2
    exit 2
fi
if [[ ${VPML_DATA_ROOT:-} != /* || ${VPML_DATA_ROOT:-} == / \
        || ${VPML_DATA_ROOT:-} == *%* || ${VPML_DATA_ROOT:-} == *$'\n'* ]]; then
    printf 'VPML_DATA_ROOT must be an absolute directory other than /, without %% or newlines.\n' >&2
    exit 2
fi
if [[ ${VPML_PYTHON:-} != /* || ! -x ${VPML_PYTHON:-} ]]; then
    printf 'VPML_PYTHON must name the absolute executable in the prepared environment.\n' >&2
    exit 2
fi
run_parameters=${VPML_RUN_PARAMS:-"$script_directory/gaussian_amplitudes.params.sh"}
if [[ $run_parameters != /* || ! -f $run_parameters || ! -r $run_parameters ]]; then
    printf 'VPML_RUN_PARAMS must name an absolute readable parameter file.\n' >&2
    exit 2
fi
VPML_CASE_IDS=()
VPML_STOP_ON_FAILURE=0
VPML_SLURM_JOB_NAME=vpml-gaussian
VPML_SHUTDOWN_MARGIN_SECONDS=180
# shellcheck source=/dev/null
source "$run_parameters"
if [[ ${VPML_PARAMETERS_FILE:-} != /* || ! -r $VPML_PARAMETERS_FILE ]]; then
    printf 'Cannot read run parameters: %s\n' "${VPML_PARAMETERS_FILE:-<unset>}" >&2
    exit 2
fi
# This launcher bounds a single A100 allocation even when selecting another
# parameter file. Reject larger requests before contacting the scheduler.
if [[ ${VPML_SLURM_PARTITION:-} != gpuA100x4 || ${VPML_SLURM_MEMORY:-} != 60G \
        || ! ${VPML_SLURM_CPUS:-} =~ ^([1-9]|1[0-6])$ \
        || ! ${VPML_SLURM_TIME:-} =~ ^([0-9]{2}):([0-5][0-9]):([0-5][0-9])$ ]]; then
    printf 'Require gpuA100x4, at most 16 CPUs, 60G memory and HH:MM:SS wall time.\n' >&2
    exit 2
fi
wall_seconds=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
if [[ ! ${VPML_MAX_SECONDS:-} =~ ^[0-9]{1,4}$ \
        || ! $VPML_SHUTDOWN_MARGIN_SECONDS =~ ^[0-9]{2,3}$ \
        || ! $VPML_STOP_ON_FAILURE =~ ^[01]$ \
        || ! $VPML_SLURM_JOB_NAME =~ ^[a-zA-Z0-9_-]+$ ]]; then
    printf 'Invalid worker budget, failure policy or job name.\n' >&2
    exit 2
fi
worker_seconds=$((10#$VPML_MAX_SECONDS))
shutdown_margin=$((10#$VPML_SHUTDOWN_MARGIN_SECONDS))
if (( shutdown_margin < 60 || shutdown_margin > 180 || wall_seconds <= shutdown_margin \
        || wall_seconds > 3600 || worker_seconds <= 0 || worker_seconds > 3300 \
        || worker_seconds > wall_seconds - shutdown_margin )); then
    printf 'Require wall time <=3600s, worker time <=3300s and a shutdown margin between 60s and 180s.\n' >&2
    exit 2
fi
selection=()
run_options=()
for case_id in ${VPML_CASE_IDS[@]+"${VPML_CASE_IDS[@]}"}; do
    selection+=(--case-id "$case_id")
    run_options+=(--case-id "$case_id")
done
if [[ $VPML_STOP_ON_FAILURE == 1 ]]; then
    run_options+=(--stop-on-failure)
fi
manifest="$VPML_DATA_ROOT/manifest.json"
output="$VPML_DATA_ROOT/reference"
cli="$VPML_CODE_ROOT/scripts/gaussian_dataset.py"

# Login-node actions only inspect or prepare metadata. Avoid inherited CUDA
# library paths; the pinned environment supplies its own CUDA wheel libraries.
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
export VPML_JAX_BACKEND=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
case $action in
    prepare)
        exec "$VPML_PYTHON" "$cli" "$action" --manifest "$manifest" --parameters "$VPML_PARAMETERS_FILE"
        ;;
    validate)
        exec "$VPML_PYTHON" "$cli" validate --manifest "$manifest" --parameters "$VPML_PARAMETERS_FILE" \
            ${selection[@]+"${selection[@]}"}
        ;;
    status)
        exec "$VPML_PYTHON" "$cli" status --out "$output" --manifest "$manifest"
        ;;
esac

"$VPML_PYTHON" "$cli" validate --manifest "$manifest" --parameters "$VPML_PARAMETERS_FILE" \
    ${selection[@]+"${selection[@]}"}
command -v sbatch >/dev/null
# Do not inherit an unrelated array, exclusive allocation or other sbatch
# option from the caller's environment; this run's request is explicit below.
for name in "${!SBATCH_@}"; do
    unset "$name"
done
arguments=(
    --parsable
    "--account=$VPML_SLURM_ACCOUNT"
    "--partition=$VPML_SLURM_PARTITION"
    --nodes=1 --ntasks=1 "--cpus-per-task=$VPML_SLURM_CPUS" --gpus-per-node=1
    "--mem=$VPML_SLURM_MEMORY" "--time=$VPML_SLURM_TIME"
    "--signal=USR1@$shutdown_margin" --no-requeue --constraint=work
    --export=NIL
    "--job-name=$VPML_SLURM_JOB_NAME"
    "--chdir=$VPML_CODE_ROOT"
    "--output=$VPML_DATA_ROOT/logs/slurm-%j.out"
    "--error=$VPML_DATA_ROOT/logs/slurm-%j.err"
)
if [[ $action == test-submit ]]; then
    arguments+=(--test-only)
else
    mkdir -p -- "$VPML_DATA_ROOT/logs"
    intent=$("$VPML_PYTHON" "$script_directory/record_submission.py" \
        "$VPML_CODE_ROOT" "$VPML_DATA_ROOT" "$VPML_PYTHON" "$VPML_PARAMETERS_FILE" \
        "$run_parameters" "$worker_seconds" "$wall_seconds" "$shutdown_margin" "${arguments[@]}" -- \
        ${run_options[@]+"${run_options[@]}"})
    printf 'Submission intent (not scheduler acceptance): %s\n' "$intent" >&2
fi
exec sbatch "${arguments[@]}" "$script_directory/gaussian_amplitudes.batch.sh" \
    "$VPML_CODE_ROOT" "$VPML_DATA_ROOT" "$VPML_PYTHON" "$VPML_PARAMETERS_FILE" "$worker_seconds" \
    ${run_options[@]+"${run_options[@]}"}
