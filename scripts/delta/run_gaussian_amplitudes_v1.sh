#!/usr/bin/env bash
# Every operation is explicit. Only the "submit" action can submit a real job.
set -euo pipefail
umask 077

usage() {
    printf 'Usage: bash %s {prepare|validate|status|test-submit|submit}\n' "$0" >&2
    printf 'Required: VPML_DATA_ROOT and VPML_PYTHON, both absolute paths.\n' >&2
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
# shellcheck source=scripts/delta/gaussian_amplitudes_v1.params.sh
source "$script_directory/gaussian_amplitudes_v1.params.sh"
if [[ ! -r $VPML_PARAMETERS_FILE ]]; then
    printf 'Cannot read run parameters: %s\n' "$VPML_PARAMETERS_FILE" >&2
    exit 2
fi
manifest="$VPML_DATA_ROOT/manifest.json"
output="$VPML_DATA_ROOT/reference"
cli="$VPML_CODE_ROOT/scripts/gaussian_dataset.py"

# Login-node actions only inspect or prepare metadata. Avoid inherited CUDA
# library paths; the pinned environment supplies its own CUDA wheel libraries.
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
export VPML_JAX_BACKEND=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
case $action in
    prepare|validate)
        exec "$VPML_PYTHON" "$cli" "$action" --manifest "$manifest" --parameters "$VPML_PARAMETERS_FILE"
        ;;
    status)
        exec "$VPML_PYTHON" "$cli" status --out "$output" --manifest "$manifest"
        ;;
esac

"$VPML_PYTHON" "$cli" validate --manifest "$manifest" --parameters "$VPML_PARAMETERS_FILE"
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
    --signal=USR1@180 --no-requeue --constraint=work
    --export=NIL
    --job-name=vpml-gaussian-v1
    "--chdir=$VPML_CODE_ROOT"
    "--output=$VPML_DATA_ROOT/logs/slurm-%j.out"
    "--error=$VPML_DATA_ROOT/logs/slurm-%j.err"
)
if [[ $action == test-submit ]]; then
    arguments+=(--test-only)
else
    mkdir -p -- "$VPML_DATA_ROOT/logs"
fi
exec sbatch "${arguments[@]}" "$script_directory/gaussian_amplitudes_v1.batch.sh" \
    "$VPML_CODE_ROOT" "$VPML_DATA_ROOT" "$VPML_PYTHON" "$VPML_PARAMETERS_FILE" "$VPML_MAX_SECONDS"
