#!/usr/bin/env bash
# Invoked by Slurm after an explicit submission through the run launcher.
set -euo pipefail
umask 077

if [[ $# -ne 5 || -z ${SLURM_JOB_ID:-} ]]; then
    printf 'This batch entrypoint requires a Slurm allocation and five launcher arguments.\n' >&2
    exit 2
fi
code=$1
data=$2
python=$3
parameters=$4
max_seconds=$5
cpus=${SLURM_CPUS_PER_TASK:-}
if [[ $code != /* || $data != /* || $data == / || $python != /* || $parameters != /* \
        || ! -x $python || ! -r $code/scripts/gaussian_dataset.py || ! -r $parameters \
        || ! $max_seconds =~ ^[0-9]+$ || $max_seconds -le 0 || $max_seconds -gt 3300 \
        || ! $cpus =~ ^[0-9]+$ || $cpus -le 0 ]]; then
    printf 'Invalid batch paths or runtime limit.\n' >&2
    exit 2
fi

# sbatch --export=NIL prevents forwarding unrelated login credentials and
# environment variables. srun gets only this explicit environment and Slurm's
# own allocation variables. Do not change CUDA_VISIBLE_DEVICES.
export PATH=/usr/bin:/bin
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
export VPML_JAX_BACKEND=gpu JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
cd -- "$code"
"$python" "$code/scripts/gaussian_dataset.py" validate \
    --manifest "$data/manifest.json" --parameters "$parameters"
exec /usr/bin/srun --ntasks=1 "--cpus-per-task=$cpus" --gpus-per-task=1 --export=ALL \
    --kill-on-bad-exit=1 "$python" "$code/scripts/gaussian_dataset.py" run \
    --manifest "$data/manifest.json" --out "$data/reference" --backend gpu \
    --max-seconds "$max_seconds"
