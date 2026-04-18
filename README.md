# JAX Fourier-Hermite Vlasov-Poisson

This repo is split into three layers:

- `vpml/`: reusable package code
- `benchmarks/`: paper-result regeneration scripts
- `model/`: learned-closure training and learned-model evaluation

That means the benchmark app and the model app both depend on `vpml`, but they do not depend on each other.

## Repo Map

- `vpml/core.py`: Fourier-Hermite operators, closures, implicit/CNAB2 solvers, learned-closure runtime
- `vpml/linear_landau.py`: shared linear-Landau rollout helpers and dispersion/root-finding utilities
- `vpml/nonlinear_landau.py`: shared nonlinear-Landau rollout runtime for benchmarks and learned-model eval
- `vpml/physical_grid.py`: physical-grid semi-Lagrangian teacher solver and projection helpers
- `vpml/metrics/`: reusable rollout metrics
- `vpml/visualization/`: reusable plotting helpers
- `benchmarks/fh_benchmarks_2412_07073_jax.py`: paper benchmark regeneration for Palisso et al. (arXiv:2412.07073)
- `benchmarks/run_all_benchmarks.sh`: full benchmark shell entrypoint
- `benchmarks/run_linear_landau_suite.sh`: linear Landau benchmark shell entrypoint
- `benchmarks/fh_nonlinear_sim_jax.py`: standalone nonlinear physical-grid simulations
- `model/model.py`: thin learned-model surface built on top of `vpml`
- `model/train/train.py`: learned interface-closure training entrypoint
- `model/train/data.py`: dataset/cache/reference-building surface for learned-closure workflows
- `model/eval.py`: post-train learned-model evaluation
- `model/eval_nv_sweep.py`: learned-model nonlinear `N_v` sweep evaluation
- `model/train/run_nv_sweep_single_qloss.sh`: per-`N_v` offline `q_only` sweep wrapper
- `model/train/run_nv_sweep_single_qloss_fixed_ratio.sh`: per-`N_v` fixed-ratio offline `q_only` sweep wrapper
- `model/train/run_nv_sweep_online_rollout.sh`: per-`N_v` pure `online_rollout` sweep wrapper
- `model/train/run_nv_sweep_higher_order_hermite_fixed_ratio.sh`: per-`N_v` higher-order-Hermite teacher sweep wrapper

## Requirements

- Python 3.10+
- `jax`
- `jaxlib`
- `numpy`
- `matplotlib`
- `scipy`

For better eigenvalue/root-finding accuracy:

```bash
export JAX_ENABLE_X64=True
```

## Backend Selection

`vpml` bootstraps JAX before import and prints the active backend when the main
benchmark or model scripts start.

- On Linux, `VPML_JAX_BACKEND=auto` leaves backend selection to JAX.
- On macOS, `vpml` defaults to CPU rather than `jax-metal` because this repo relies heavily on `float64` and complex dtypes.

Overrides:

```bash
export VPML_JAX_BACKEND=cpu
export VPML_JAX_BACKEND=gpu
```

If you actually want CUDA, install a CUDA-enabled JAX build:

```bash
pip install -U "jax[cuda13]"
```

## Install

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -e .
```

Or just install dependencies:

```bash
pip install -r requirements.txt
```

Editable install sanity check:

```bash
python -c "import vpml; print(vpml.__version__)"
```

## Running The Benchmark App

### Nonlinear Simulations

```bash
python -m benchmarks.fh_nonlinear_sim_jax two_stream --outdir out_nl
python -m benchmarks.fh_nonlinear_sim_jax bump_on_tail --system AC --outdir out_nl --vmin -12 --vmax 12
```

Outputs go to `out_nl/`.

### Paper Benchmarks

```bash
python -m benchmarks.fh_benchmarks_2412_07073_jax fig2 --outdir out_bench
python -m benchmarks.fh_benchmarks_2412_07073_jax fig3 --outdir out_bench
python -m benchmarks.fh_benchmarks_2412_07073_jax fig4 --outdir out_bench --Nv 20
python -m benchmarks.fh_benchmarks_2412_07073_jax linear_landau --method truncation --outdir out_bench
./benchmarks/run_all_benchmarks.sh out_bench
```

When a learned checkpoint is available:

```bash
python -m benchmarks.fh_benchmarks_2412_07073_jax linear_landau --method learned --outdir out_bench --learned-checkpoint out_model/interface_closure.npz
python -m benchmarks.fh_benchmarks_2412_07073_jax fig10_learned_comparison --outdir out_bench --learned-checkpoint out_model/interface_closure.npz
LEARNED_CHECKPOINT=out_model/interface_closure.npz ./benchmarks/run_all_benchmarks.sh out_bench
```

The learned closure is intentionally not supported in `fig3` response-function or `fig4`
eigenvalue benchmarks because it is state-dependent rather than a fixed modified Hermite matrix.

## Running The Model App

### Train A Learned Closure

```bash
python -m model.train.train \
  --checkpoint out_model/interface_closure.npz \
  --dataset-cache out_model/interface_closure_dataset.npz
```

This writes:

- `out_model/interface_closure.npz`
- `out_model/interface_closure.metrics.npz`
- `out_model/interface_closure_dataset.npz`
- optionally `out_model/interface_closure.loss.png` if `--loss-plot` is passed

The main offline training lane is `q_only`. The pure online lane is still kept separate as `online_rollout`.

### Evaluate A Learned Closure

```bash
python -m model.eval \
  --checkpoint out_model/interface_closure.npz \
  --outdir out_model/eval
```

This writes:

- `out_model/eval/summary.json`
- `out_model/eval/heldout_landau/*.npz`
- `out_model/eval/heldout_landau/*_summary.png`
- `out_model/eval/benchmark_rollouts/*.npz`
- `out_model/eval/benchmark_rollouts/*_summary.png`

### Sweep Learned Closures Across Deployment `N_v`

Offline target-specific `q_only` sweep:

```bash
./model/train/run_nv_sweep_single_qloss.sh out_bench/nv_sweep_single_qloss
```

Offline fixed-ratio `q_only` sweep:

```bash
./model/train/run_nv_sweep_single_qloss_fixed_ratio.sh out_bench/nv_sweep_single_qloss_fixed_ratio
```

Pure online rollout sweep:

```bash
./model/train/run_nv_sweep_online_rollout.sh out_bench/nv_sweep_online_rollout
```

Higher-order-Hermite teacher fixed-ratio sweep:

```bash
./model/train/run_nv_sweep_higher_order_hermite_fixed_ratio.sh out_bench/nv_sweep_higher_order_hermite_fixed_ratio
```

All of these wrappers train one checkpoint per deployment `N_v` and then run:

```bash
python -m model.eval_nv_sweep ...
```

The sweep outputs include:

- `summary.json`
- `nv_sweep_metric1.png`
- `nv_sweep_metric2.png`
- `fig10_learned_vs_nonlocal_nv_sweep_phase_space.png`
- `cases/*.npz`

For the offline wrappers, dataset caches live under each per-`N_v` model directory:

- `models/nv*/interface_closure_dataset.npz`

For the pure online wrapper, no dataset cache is written.

## Design Boundary

The intended dependency direction is:

```text
benchmarks  -> vpml
model       -> vpml
```

Not:

```text
vpml -> benchmarks
vpml -> model
```

So shared math, solver code, rollout runtimes, checkpoint I/O, metrics, and plotting primitives belong in `vpml`.
Training-program choices such as dataset construction, curriculum ladders, and offline-vs-online objectives belong in `model/`.
