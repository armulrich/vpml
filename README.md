# JAX Fourier–Hermite Vlasov–Poisson 

This repo exposes the reusable Fourier–Hermite implementation and the shared physical-grid semi-Lagrangian solver as an importable `vpml` package and keeps the runnable scripts under `benchmarks/`.

- `vpml/`: importable Fourier–Hermite core operators plus the shared physical-grid semi-Lagrangian solver
- `benchmarks/fh_nonlinear_sim_jax.py`: nonlinear two-stream and bump-on-tail runs on the shared physical-grid solver
- `benchmarks/fh_benchmarks_2412_07073_jax.py`: linear benchmarks reproducing key figures from Palisso et al. (arXiv:2412.07073)
- `benchmarks/fh_ml_tail_closure_train_jax.py`: training entrypoint for the learned interface closure using a physical-grid Landau teacher
- `benchmarks/eval.py`: post-train a posteriori metric evaluation for learned closures
- `benchmarks/run_all_benchmarks.sh`: shell entry point for the full benchmark suite

## Requirements

- Python 3.10+ (this repo is currently using Python 3.12)
- `jax`, `jaxlib`, `numpy`, `matplotlib`
- `scipy` (required for the benchmark script’s plasma-dispersion function and spline-regression tests in this environment)

For better eigenvalue/root-finding accuracy, enable 64-bit:

```bash
export JAX_ENABLE_X64=True
```

## Backend selection

`vpml` now bootstraps JAX before import and prints the active backend when the main
training or benchmark scripts start.

- On Linux, the default `VPML_JAX_BACKEND=auto` leaves backend selection to JAX. Per the
  JAX docs, that means JAX defaults to GPU or TPU if available and falls back to CPU
  otherwise.
- On macOS, `vpml` forces `JAX_PLATFORMS=cpu` by default instead of using `jax-metal`.
  This repo relies heavily on `float64` and complex dtypes, while the current Apple
  `jax-metal` docs still list `np.float64`, `np.complex64`, and `np.complex128` as
  unsupported.

You can override the policy with:

```bash
export VPML_JAX_BACKEND=cpu   # force CPU
export VPML_JAX_BACKEND=gpu   # prefer a CUDA/ROCm-style backend; vpml will warn if JAX still lands on CPU
```

To actually use NVIDIA CUDA, install a CUDA-enabled JAX build in your environment, e.g.

```bash
pip install -U "jax[cuda13]"
```

## Install

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -e .
```

If you only want to install the dependencies (without an editable install), you can do:

```bash
pip install -r requirements.txt
```

After the editable install, this should work:

```bash
python -c "import vpml; print(vpml.__version__)"
```

## How to run

### Nonlinear simulations

```bash
python benchmarks/fh_nonlinear_sim_jax.py two_stream --outdir out_nl
python benchmarks/fh_nonlinear_sim_jax.py bump_on_tail --system AC --outdir out_nl --vmin -12 --vmax 12
```

Outputs go to `out_nl/` (PNGs + `.npz` energy time series).

### Benchmarks (arXiv:2412.07073)

```bash
python benchmarks/fh_benchmarks_2412_07073_jax.py fig2 --outdir out_bench
python benchmarks/fh_benchmarks_2412_07073_jax.py fig3 --outdir out_bench
python benchmarks/fh_benchmarks_2412_07073_jax.py fig4 --outdir out_bench --Nv 20
python benchmarks/fh_benchmarks_2412_07073_jax.py linear_landau --method truncation --outdir out_bench
./benchmarks/run_all_benchmarks.sh out_bench
```

Outputs go to `out_bench/` (PNGs + `.npz` data dumps).

When a learned interface-closure checkpoint is available, pass it explicitly to rollout-style benchmarks:

```bash
python benchmarks/fh_benchmarks_2412_07073_jax.py linear_landau --method learned --outdir out_bench --learned-checkpoint out_bench/interface_closure.npz
python benchmarks/fh_benchmarks_2412_07073_jax.py fig10 --outdir out_bench
python benchmarks/fh_benchmarks_2412_07073_jax.py fig10_learned_comparison --outdir out_bench --learned-checkpoint out_bench/interface_closure.npz
LEARNED_CHECKPOINT=out_bench/interface_closure.npz ./benchmarks/run_all_benchmarks.sh out_bench
```

The learned closure is intentionally not supported in `fig3` response-function or `fig4`
eigenvalue benchmarks because it is state-dependent rather than a fixed modified Hermite
matrix.

### Learned Interface-Closure Training

If you want the whole workflow in one command, use the wrapper script:

```bash
cd /Users/armin/Documents/NYU/vpml
./benchmarks/train.sh out_bench
```

That single script will:

- generate or reuse the cached mixed Landau-family training dataset
- train the shared interface-closure model with the repo defaults
- run `benchmarks/eval.py` to compute post-train rollout metrics and metric figures
- save the checkpoint, metrics, and training-loss plot
- rerun the benchmark suite
- render the classical nonlinear Landau phase-space figure plus a separate learned-vs-nonlocal comparison figure

You can override the defaults with environment variables, for example:

```bash
EPOCHS=100 NM=4 HIDDEN_WIDTH=128 ./benchmarks/train.sh out_bench
```

The current wrapper defaults train a less-local learned closure by using:

- `NM=6`
- `NV_TARGETS=6,8,10,12,20,40,80,160,300`

`train.sh` also now rebuilds an incompatible cached dataset automatically, so rerunning it after these default changes is enough even if `out_bench/interface_closure_dataset.npz` was created by an older low-`N_v` configuration.

The most useful outputs after the script finishes are:

- `out_bench/interface_closure.loss.png`
- `out_bench/eval/summary.json`
- `out_bench/linear_landau_comparison.png`
- `out_bench/fig10_nonlinear_landau_phase_space.png`
- `out_bench/fig10_learned_vs_nonlocal_phase_space.png`

```bash
python benchmarks/fh_ml_tail_closure_train_jax.py \
  --checkpoint out_bench/interface_closure.npz \
  --dataset-cache out_bench/interface_closure_dataset.npz
```

This writes:

- `interface_closure.npz`: learned interface-closure weights, normalization stats, and metadata
- `interface_closure.metrics.npz`: single-stage training loss plus split validation metrics
- `interface_closure_dataset.npz`: optional cached mixed Landau-family supervised pairs generated from the physical-grid teacher

To run the post-train a posteriori evaluation directly:

```bash
python benchmarks/eval.py \
  --checkpoint out_bench/interface_closure.npz \
  --outdir out_bench/eval
```

This writes:

- `eval/summary.json`: aggregate scalar metric summary across bundles
- `eval/heldout_landau/*.npz`: per-case times, energies, field histories, and scalar metrics
- `eval/heldout_landau/*_summary.png`: combined Metric 1 + Metric 2 figures
- `eval/benchmark_rollouts/*.npz`: per-case benchmark-style learned-rollout metric payloads
- `eval/benchmark_rollouts/*_summary.png`: benchmark-style metric figures

To sweep the nonlinear learned closure across deployment `N_v` values without rerunning the full benchmark suite:

```bash
./benchmarks/run_nv_sweep.sh out_bench/nv_sweep
```

The sweep defaults to `NV_LIST=8,64,256,300,512` and writes:

- `nv_sweep/shared_interface_closure_dataset.npz`: one shared extracted dataset cache built once for the full `N_v` list
- `nv_sweep/models/nv*/interface_closure.npz`: separately trained learned-closure checkpoints, one per `N_v`
- `nv_sweep/summary.json`: scalar metric summary per `N_v`
- `nv_sweep/nv_sweep_metric1.png`: all nonlinear energy traces plus fitted early-time lines on one figure
- `nv_sweep/nv_sweep_metric2.png`: `\varepsilon_E` summary plus stacked HR/learned field panels across `N_v`
- `nv_sweep/fig10_learned_vs_nonlocal_nv_sweep_phase_space.png`: stacked nonlocal-vs-learned nonlinear phase-space panels across `N_v`
- `nv_sweep/cases/*.npz`: per-`N_v` saved metric payloads

By default this sweep now trains a separate learned closure for each deployment `N_v`
from the shared extracted cache. That keeps the experiment aligned with the question
"how does the learned closure improve as `N_v` changes?" rather than mixing all
deployment resolutions into one shared checkpoint.

The first training pass uses only Landau-family data from the shared physical-grid teacher:

- linear Landau damping
- weakly nonlinear Landau damping
- strongly nonlinear Landau damping

The teacher defaults are:

- `teacher-Nx = 256`
- `teacher-Nv = 512`
- `teacher-vmin = -8`
- `teacher-vmax = 8`
- `teacher-dt = 0.01`
- `teacher-proj-Nv = max(Nv-targets) + 1` unless overridden

The runtime closure API can be reused elsewhere, but a checkpoint should only be trusted
for regimes represented in its training data.

## Package map

- `vpml`: public package namespace
- `vpml.core`: Fourier/Hermite utilities, CNAB2 integrator, implicit-midpoint/JFNK stepper, damping, and closure operators
- `vpml.physical_grid`: shared JAX semi-Lagrangian cubic-spline Vlasov-Poisson solver and Fourier-Hermite projection helpers
- `vpml.visualization`: shared plotting/output helpers for training, benchmarks, nonlinear runs, and post-train metrics
- `benchmarks/`: runnable scripts and shell helpers that import `vpml`
