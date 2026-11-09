#!/usr/bin/env bash
# Sequential CPU processes release each variant's JAX compilation cache on exit.
set -euo pipefail
umask 077

if [[ $# -ne 1 || ! $1 =~ ^(validate|run|compare)$ ]]; then
    printf 'Usage: bash %s {validate|run|compare}\n' "$0" >&2
    printf 'Set VPML_PYTHON and VPML_REFINEMENT_ROOT to absolute paths.\n' >&2
    exit 2
fi
action=$1
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
VPML_CODE_ROOT=$(cd -- "$script_directory/../.." && pwd -P)
if [[ ${VPML_PYTHON:-} != /* || ! -x ${VPML_PYTHON:-} ]]; then
    printf 'VPML_PYTHON must be an absolute executable path.\n' >&2
    exit 2
fi
if [[ ${VPML_REFINEMENT_ROOT:-} != /* || ${VPML_REFINEMENT_ROOT:-} == / ]]; then
    printf 'VPML_REFINEMENT_ROOT must be an absolute output directory other than /.\n' >&2
    exit 2
fi
# shellcheck source=scripts/local/gaussian_refinement_v1.params.sh
source "$script_directory/gaussian_refinement_v1.params.sh"
export PYTHONPATH="$VPML_CODE_ROOT/src"
export VPML_JAX_BACKEND=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
cli=("$VPML_PYTHON" -m vpml.cli.gaussian_refinement)
if [[ $action == validate ]]; then
    exec "${cli[@]}" validate --config "$VPML_REFINEMENT_CONFIG"
fi
if [[ $action == run ]]; then
    "${cli[@]}" validate --config "$VPML_REFINEMENT_CONFIG"
    mkdir -p -- "$VPML_REFINEMENT_ROOT/logs"
    for variant in "${VPML_REFINEMENT_VARIANTS[@]}"; do
        "${cli[@]}" run --config "$VPML_REFINEMENT_CONFIG" \
            --out "$VPML_REFINEMENT_ROOT" --variant "$variant" \
            --max-seconds "$VPML_REFINEMENT_MAX_SECONDS" \
            2>&1 | tee -a "$VPML_REFINEMENT_ROOT/logs/$variant.log"
    done
fi
exec "${cli[@]}" compare --out "$VPML_REFINEMENT_ROOT" --plot
