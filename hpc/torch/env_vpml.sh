#!/usr/bin/env bash
# Shared environment for vpml training on a Linux HPC GPU node.
# Source after activating conda, before training or sbatch:
#   source /path/to/vpml/hpc/torch/env_vpml.sh
#
# Install CUDA JAX matching the cluster driver (example):
#   pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-True}"
export VPML_JAX_BACKEND="${VPML_JAX_BACKEND:-gpu}"
export TRAIN_PRECISION="${TRAIN_PRECISION:-float64}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
