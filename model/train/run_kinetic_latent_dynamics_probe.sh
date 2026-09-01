#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="${1:?usage: run_kinetic_latent_dynamics_probe.sh OUTDIR}"
cd "${ROOT_DIR}"

PYTHON_BIN="${VPML_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
ARGS=(
  -m model.train.kinetic_latent_dynamics_probe
  --reference-cache "${KINETIC_LATENT_REFERENCE_CACHE:?set KINETIC_LATENT_REFERENCE_CACHE}" \
  --projected-cache "${KINETIC_LATENT_PROJECTED_CACHE:?set KINETIC_LATENT_PROJECTED_CACHE}" \
  --outdir "${OUTDIR}" \
  --latent-rank "${KINETIC_LATENT_RANK:-16}" \
  --basis-modes "${KINETIC_LATENT_BASIS_MODES:-64}" \
  --nx "${KINETIC_LATENT_NX:-64}" \
  --cadence "${KINETIC_LATENT_CADENCE:-0.1}" \
  --rollout-steps "${KINETIC_LATENT_ROLLOUT_STEPS:-10}" \
  --epochs "${KINETIC_LATENT_EPOCHS:-10}" \
  --steps-per-epoch "${KINETIC_LATENT_STEPS_PER_EPOCH:-30}" \
  --batch-per-regime "${KINETIC_LATENT_BATCH_PER_REGIME:-2}" \
  --width "${KINETIC_LATENT_WIDTH:-32}" \
  --depth "${KINETIC_LATENT_DEPTH:-4}" \
  --kernel-size "${KINETIC_LATENT_KERNEL_SIZE:-5}" \
  --learning-rate "${KINETIC_LATENT_LR:-3e-4}" \
  --grad-clip "${KINETIC_LATENT_GRAD_CLIP:-1.0}" \
  --linear-ridge "${KINETIC_LATENT_LINEAR_RIDGE:-1e-4}" \
  --linear-max-spectral-radius "${KINETIC_LATENT_LINEAR_MAX_SPECTRAL_RADIUS:-1.0}" \
  --dynamics-model "${KINETIC_LATENT_DYNAMICS_MODEL:-additive_residual}" \
  --operator-rank "${KINETIC_LATENT_OPERATOR_RANK:-16}" \
  --operator-modes "${KINETIC_LATENT_OPERATOR_MODES:-0}" \
  --validation-every "${KINETIC_LATENT_VALIDATION_EVERY:-2}" \
  --validation-horizons "${KINETIC_LATENT_VALIDATION_HORIZONS:-10,50}" \
  --validation-start-times "${KINETIC_LATENT_VALIDATION_START_TIMES:-20,40,60,80}" \
  --seed "${KINETIC_LATENT_SEED:-1729}"
)
if [[ -n "${KINETIC_LATENT_INIT_CHECKPOINT:-}" ]]; then
  ARGS+=(--init-checkpoint "${KINETIC_LATENT_INIT_CHECKPOINT}")
fi
exec "${PYTHON_BIN}" "${ARGS[@]}"
