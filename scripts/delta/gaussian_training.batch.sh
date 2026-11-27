#!/usr/bin/env bash
# Slurm copies this script. The source release and training data live on shared storage.
set -euo pipefail
umask 077

if [[ -z ${SLURM_JOB_ID:-} || $# -lt 9 || $6 != --shutdown-margin ]]; then
    echo "Training requires a Slurm allocation and the bounded launcher's arguments." >&2
    exit 2
fi
code=$1
run=$2
python=$3
normalization=$4
requested_seconds=$5
shutdown_margin=$7
shift 7

unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME
export PATH=/usr/bin:/bin
export PYTHONPATH="$code/src" PYTHONUNBUFFERED=1
export VPML_JAX_BACKEND=gpu JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true'
export JAX_COMPILATION_CACHE_DIR="$run/jax-cache"
cd -- "$code"
seconds=$("$python" "$code/scripts/delta/budget.py" worker-seconds "$requested_seconds" "$shutdown_margin")
# Slurm sends USR1 to this job step before timeout. The worker finishes or
# discards its in-flight update and checkpoints the last complete update.
exec /usr/bin/srun --ntasks=1 "--cpus-per-task=$SLURM_CPUS_PER_TASK" --gpus-per-task=1 \
    --kill-on-bad-exit=1 --export=ALL "$python" -m vpml.cli.gaussian_training \
    --run "$run" --normalization "$normalization" --worker-seconds "$seconds" --require-gpu "$@"
