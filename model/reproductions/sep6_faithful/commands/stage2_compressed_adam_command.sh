#!/usr/bin/env bash
set -euo pipefail
/Users/armin/Documents/NYU/vpml/.venv/bin/python model/train/coupled_low_moment_latent.py \
  --reference-cache /Users/armin/Documents/NYU/vpml/out_bench/reference_cache/interface_flux_landau_T120_Nx1024_Nv8192_M4096/e376aa1efa28e754b5f6 \
  --projected-cache /Users/armin/Documents/NYU/vpml/out_bench/kinetic_latent_projected_cache_rank32_nx64_dt0p1 \
  --outdir /Users/armin/Documents/NYU/vpml/out_bench/reproductions/sep6_faithful_20260909/stage2_compressed_adam \
  --latent-rank 32 --basis-modes 64 --nx 64 --cadence 0.1 --fine-steps 10 \
  --epochs 3 --steps-per-epoch 4 --gradient-accumulation-steps 4 \
  --width 48 --depth 4 --kernel-size 5 --operator-rank 16 --operator-modes 33 \
  --operator-output-init-scale 1e-6 --latent-delay-input \
  --latent-readout-mode equilibrium_cnn --equilibrium-input-compression-scale 4 \
  --latent-residual-weight 1 --latent-state-residual-weight 1 \
  --closure-residual-weight 0 --autonomous-latent-weight 0 \
  --electric-spectrum-weight 0 --electric-log-energy-weight 0.01 \
  --electric-log-energy-floor-ratio 1e-8 --electric-chunk-log-growth-weight 0.05 \
  --electric-growth-window-steps 100 --electric-time-relative-weight 0.05 \
  --electric-time-relative-floor-ratio 1e-4 --latent-gradient-ratio 1 \
  --teacher-internal-update-ratio 10 --teacher-residual-stride 50 --teacher-rollout-steps 10 \
  --training-objective joint --correction-energy-tolerance 0.001 \
  --learning-rate 1e-4 --teacher-learning-rate 1e-3 --update-combination conflict_safe \
  --optimizer adam --normalize-accumulated-gradients --grad-clip 1 \
  --parameter-update-clip 0.05 --trust-region-backtracks 14 \
  --trust-region-tolerance 0 --trust-region-objective total --trust-region-batches 4 \
  --linear-ridge 1e-4 --linear-max-spectral-radius 1 \
  --linear-baseline projected_hermite --hermite-tail-damping 10 --hermite-tail-power 6 \
  --semilinear-kick-scale 1 --semilinear-correction-location midpoint \
  --latent-excursion-limit 1000 --linear-nonregression-limit 1e-4 \
  --gradient-chunk-steps 1200 --validation-every 1 --loss-ema-decay 0.95 \
  --init-checkpoint /Users/armin/Documents/NYU/vpml/out_bench/reproductions/sep6_faithful_20260909/stage1_e3/best_coupled_low_moment_latent.npz \
  --seed 1729
