#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="${1:?usage: run_coupled_low_moment_latent.sh OUTDIR}"
cd "${ROOT_DIR}"

PYTHON_BIN="${VPML_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
set --
case "${COUPLED_LATENT_EVAL_AFTER_TRAINING:-1}" in
  0|false|FALSE|no|NO) set -- --skip-evaluation ;;
esac
if [[ -n "${COUPLED_LATENT_RESUME_TRAINING_STATE:-}" ]]; then
  set -- "$@" --resume-training-state "${COUPLED_LATENT_RESUME_TRAINING_STATE}"
fi
case "${COUPLED_LATENT_CLOSURE_ALIGNED_OUTPUT:-0}" in
  1|true|TRUE|yes|YES) set -- "$@" --closure-aligned-output ;;
esac
case "${COUPLED_LATENT_CLOSURE_ONLY_CORRECTION:-0}" in
  1|true|TRUE|yes|YES) set -- "$@" --closure-only-correction ;;
esac
case "${COUPLED_LATENT_NORMALIZE_ACCUMULATED_GRADIENTS:-0}" in
  1|true|TRUE|yes|YES) set -- "$@" --normalize-accumulated-gradients ;;
esac
case "${COUPLED_LATENT_FIT_MULTIPLICATIVE_OUTPUT_READOUT:-0}" in
  1|true|TRUE|yes|YES) set -- "$@" --fit-multiplicative-output-readout ;;
esac
case "${COUPLED_LATENT_DELAY_INPUT:-0}" in
  1|true|TRUE|yes|YES) set -- "$@" --latent-delay-input ;;
esac

exec "${PYTHON_BIN}" -m model.train.coupled_low_moment_latent \
  --reference-cache "${COUPLED_LATENT_REFERENCE_CACHE:?set COUPLED_LATENT_REFERENCE_CACHE}" \
  --projected-cache "${COUPLED_LATENT_PROJECTED_CACHE:?set COUPLED_LATENT_PROJECTED_CACHE}" \
  --outdir "${OUTDIR}" \
  --latent-rank "${COUPLED_LATENT_RANK:-32}" \
  --basis-modes "${COUPLED_LATENT_BASIS_MODES:-64}" \
  --nx "${COUPLED_LATENT_NX:-64}" \
  --cadence "${COUPLED_LATENT_CADENCE:-0.1}" \
  --fine-steps "${COUPLED_LATENT_FINE_STEPS:-10}" \
  --epochs "${COUPLED_LATENT_EPOCHS:-200}" \
  --steps-per-epoch "${COUPLED_LATENT_STEPS_PER_EPOCH:-16}" \
  --gradient-accumulation-steps "${COUPLED_LATENT_GRADIENT_ACCUMULATION_STEPS:-1}" \
  --width "${COUPLED_LATENT_WIDTH:-48}" \
  --depth "${COUPLED_LATENT_DEPTH:-4}" \
  --kernel-size "${COUPLED_LATENT_KERNEL_SIZE:-5}" \
  --operator-rank "${COUPLED_LATENT_OPERATOR_RANK:-16}" \
  --operator-modes "${COUPLED_LATENT_OPERATOR_MODES:-33}" \
  --operator-output-init-scale "${COUPLED_LATENT_OPERATOR_OUTPUT_INIT_SCALE:-1e-3}" \
  --multiplicative-output-ridge "${COUPLED_LATENT_MULTIPLICATIVE_OUTPUT_RIDGE:-1e-4}" \
  --multiplicative-output-initial-scale "${COUPLED_LATENT_MULTIPLICATIVE_OUTPUT_INITIAL_SCALE:-1.0}" \
  --latent-residual-weight "${COUPLED_LATENT_RESIDUAL_WEIGHT:-1.0}" \
  --latent-state-residual-weight "${COUPLED_LATENT_STATE_RESIDUAL_WEIGHT:-1.0}" \
  --closure-residual-weight "${COUPLED_LATENT_CLOSURE_RESIDUAL_WEIGHT:-0.0}" \
  --closure-correction-bound "${COUPLED_LATENT_CLOSURE_CORRECTION_BOUND:-40.0}" \
  --closure-readout-mode "${COUPLED_LATENT_CLOSURE_READOUT_MODE:-multiplicative}" \
  --closure-gate-scale "${COUPLED_LATENT_CLOSURE_GATE_SCALE:-0.1}" \
  --closure-gate-power "${COUPLED_LATENT_CLOSURE_GATE_POWER:-2}" \
  --closure-readout-ridge "${COUPLED_LATENT_CLOSURE_READOUT_RIDGE:-1e-4}" \
  --closure-readout-initial-scale "${COUPLED_LATENT_CLOSURE_READOUT_INITIAL_SCALE:-1.0}" \
  --closure-readout-trajectory-exponent "${COUPLED_LATENT_CLOSURE_READOUT_TRAJECTORY_EXPONENT:-0.75}" \
  --latent-readout-mode "${COUPLED_LATENT_DYNAMICS_READOUT_MODE:-multiplicative}" \
  --latent-gate-scale "${COUPLED_LATENT_DYNAMICS_GATE_SCALE:-0.1}" \
  --latent-gate-power "${COUPLED_LATENT_DYNAMICS_GATE_POWER:-2}" \
  --latent-readout-ridge "${COUPLED_LATENT_DYNAMICS_READOUT_RIDGE:-1e-4}" \
  --latent-readout-initial-scale "${COUPLED_LATENT_DYNAMICS_READOUT_INITIAL_SCALE:-1.0}" \
  --latent-readout-trajectory-exponent "${COUPLED_LATENT_DYNAMICS_READOUT_TRAJECTORY_EXPONENT:-0.75}" \
  --autonomous-latent-weight "${COUPLED_LATENT_AUTONOMOUS_LATENT_WEIGHT:-0.0}" \
  --electric-spectrum-weight "${COUPLED_LATENT_ELECTRIC_SPECTRUM_WEIGHT:-0.0}" \
  --electric-log-energy-weight "${COUPLED_LATENT_ELECTRIC_LOG_ENERGY_WEIGHT:-0.0}" \
  --electric-log-energy-floor-ratio "${COUPLED_LATENT_ELECTRIC_LOG_ENERGY_FLOOR_RATIO:-1e-8}" \
  --electric-chunk-log-growth-weight "${COUPLED_LATENT_ELECTRIC_CHUNK_LOG_GROWTH_WEIGHT:-0.0}" \
  --electric-growth-window-steps "${COUPLED_LATENT_ELECTRIC_GROWTH_WINDOW_STEPS:-100}" \
  --electric-time-relative-weight "${COUPLED_LATENT_ELECTRIC_TIME_RELATIVE_WEIGHT:-0.0}" \
  --electric-time-relative-floor-ratio "${COUPLED_LATENT_ELECTRIC_TIME_RELATIVE_FLOOR_RATIO:-1e-4}" \
  --latent-gradient-ratio "${COUPLED_LATENT_GRADIENT_RATIO:-1.0}" \
  --teacher-internal-update-ratio "${COUPLED_LATENT_INTERNAL_UPDATE_RATIO:-10.0}" \
  --teacher-residual-stride "${COUPLED_LATENT_TEACHER_RESIDUAL_STRIDE:-50}" \
  --teacher-rollout-steps "${COUPLED_LATENT_TEACHER_ROLLOUT_STEPS:-10}" \
  --training-objective "${COUPLED_LATENT_TRAINING_OBJECTIVE:-joint}" \
  --correction-energy-tolerance "${COUPLED_LATENT_CORRECTION_ENERGY_TOLERANCE:-1e-3}" \
  --learning-rate "${COUPLED_LATENT_LR:-1e-4}" \
  --teacher-learning-rate "${COUPLED_LATENT_TEACHER_LR:-1e-3}" \
  --update-combination "${COUPLED_LATENT_UPDATE_COMBINATION:-conflict_safe}" \
  --optimizer "${COUPLED_LATENT_OPTIMIZER:-adam}" \
  --grad-clip "${COUPLED_LATENT_GRAD_CLIP:-1.0}" \
  --parameter-update-clip "${COUPLED_LATENT_PARAMETER_UPDATE_CLIP:-0.05}" \
  --trust-region-backtracks "${COUPLED_LATENT_TRUST_REGION_BACKTRACKS:-0}" \
  --trust-region-tolerance "${COUPLED_LATENT_TRUST_REGION_TOLERANCE:-1e-3}" \
  --trust-region-objective "${COUPLED_LATENT_TRUST_REGION_OBJECTIVE:-total}" \
  --trust-region-batches "${COUPLED_LATENT_TRUST_REGION_BATCHES:-1}" \
  --linear-ridge "${COUPLED_LATENT_LINEAR_RIDGE:-1e-4}" \
  --linear-max-spectral-radius "${COUPLED_LATENT_LINEAR_MAX_SPECTRAL_RADIUS:-1.0}" \
  --linear-baseline "${COUPLED_LATENT_LINEAR_BASELINE:-fitted_propagator}" \
  --hermite-tail-damping "${COUPLED_LATENT_HERMITE_TAIL_DAMPING:-10.0}" \
  --hermite-tail-power "${COUPLED_LATENT_HERMITE_TAIL_POWER:-6.0}" \
  --semilinear-kick-scale "${COUPLED_LATENT_SEMILINEAR_KICK_SCALE:-1.0}" \
  --semilinear-correction-location "${COUPLED_LATENT_SEMILINEAR_CORRECTION_LOCATION:-midpoint}" \
  --latent-excursion-limit "${COUPLED_LATENT_EXCURSION_LIMIT:-100.0}" \
  --linear-nonregression-limit "${COUPLED_LATENT_LINEAR_NONREGRESSION_LIMIT:-0.0}" \
  --gradient-chunk-steps "${COUPLED_LATENT_GRADIENT_CHUNK_STEPS:-300}" \
  --validation-every "${COUPLED_LATENT_VALIDATION_EVERY:-5}" \
  --loss-ema-decay "${COUPLED_LATENT_LOSS_EMA_DECAY:-0.95}" \
  --seed "${COUPLED_LATENT_SEED:-1729}" \
  "$@"
