#!/usr/bin/env bash
# Invoked by Slurm after an explicit submission through the run launcher.
set -euo pipefail
umask 077

if [[ $# -lt 5 || -z ${SLURM_JOB_ID:-} ]]; then
    printf 'This batch entrypoint requires a Slurm allocation and at least five launcher arguments.\n' >&2
    exit 2
fi
code=$1
data=$2
python=$3
parameters=$4
max_seconds=$5
shift 5
selection=()
run_options=()
stop_on_failure=0
transition_path=
transition_sha256=
while (( $# )); do
    case $1 in
        --case-id)
            if [[ $# -lt 2 || -z $2 ]]; then
                printf 'Each --case-id requires a value.\n' >&2
                exit 2
            fi
            selection+=(--case-id "$2")
            run_options+=(--case-id "$2")
            shift 2
            ;;
        --stop-on-failure)
            if [[ $stop_on_failure == 1 ]]; then
                printf 'Duplicate --stop-on-failure option.\n' >&2
                exit 2
            fi
            stop_on_failure=1
            run_options+=(--stop-on-failure)
            shift
            ;;
        --source-transition|--source-transition-sha256)
            if [[ $# -lt 2 || -z $2 ]]; then
                printf 'Each source transition option requires a value.\n' >&2
                exit 2
            fi
            if [[ $1 == --source-transition ]]; then
                [[ -z $transition_path ]] || exit 2
                transition_path=$2
            else
                [[ -z $transition_sha256 ]] || exit 2
                transition_sha256=$2
            fi
            run_options+=("$1" "$2")
            shift 2
            ;;
        *)
            printf 'Unsupported worker option: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done
cpus=${SLURM_CPUS_PER_TASK:-}
if [[ $code != /* || $data != /* || $data == / || $python != /* || $parameters != /* \
        || ! -x $python || ! -r $code/src/vpml/cli/gaussian_dataset.py || ! -r $parameters \
        || ! $max_seconds =~ ^[0-9]{1,4}$ \
    || ! $cpus =~ ^([1-9]|1[0-6])$ ]]; then
    printf 'Invalid batch paths or runtime limit.\n' >&2
    exit 2
fi
if [[ -n $transition_path || -n $transition_sha256 ]]; then
    if [[ $transition_path != /* || ! -r $transition_path || ! $transition_sha256 =~ ^[0-9a-f]{64}$ ]]; then
        printf 'Invalid source transition path or digest.\n' >&2
        exit 2
    fi
    selection+=(--source-transition "$transition_path" \
        --source-transition-sha256 "$transition_sha256")
fi
max_seconds=$((10#$max_seconds))
if (( max_seconds <= 0 || max_seconds > 3300 )); then
    printf 'Worker runtime must be between 1 and 3300 seconds.\n' >&2
    exit 2
fi

# sbatch --export=NIL prevents forwarding unrelated login credentials and
# environment variables. srun gets only this explicit environment and Slurm's
# own allocation variables. Do not change CUDA_VISIBLE_DEVICES.
export PATH=/usr/bin:/bin
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
export VPML_JAX_BACKEND=gpu JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
cd -- "$code"
"$python" -m vpml.cli.gaussian_dataset --code-root "$code" validate \
    --manifest "$data/manifest.json" --parameters "$parameters" --out "$data/reference" \
    ${selection[@]+"${selection[@]}"}
exec /usr/bin/srun --ntasks=1 "--cpus-per-task=$cpus" --gpus-per-task=1 --export=ALL \
    --kill-on-bad-exit=1 "$python" -m vpml.cli.gaussian_dataset --code-root "$code" run \
    --manifest "$data/manifest.json" --out "$data/reference" --backend gpu \
    --max-seconds "$max_seconds" ${run_options[@]+"${run_options[@]}"}
