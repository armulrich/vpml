```
 ██╗   ██╗██████╗ ███╗   ███╗██╗
 ██║   ██║██╔══██╗████╗ ████║██║
 ██║   ██║██████╔╝██╔████╔██║██║
 ╚██╗ ██╔╝██╔═══╝ ██║╚██╔╝██║██║
  ╚████╔╝ ██║     ██║ ╚═╝ ██║███████╗
   ╚═══╝  ╚═╝     ╚═╝     ╚═╝╚══════╝

    Fourier-Hermite · Vlasov-Poisson · JAX
```

> **JAX Fourier-Hermite Vlasov-Poisson solver with a learned interface closure.**

`vpml` solves the 1D1V collisionless Vlasov-Poisson system using Fourier modes
in space, orthonormal Hermite functions in velocity, and an IMEX CNAB2 time
integrator. It includes classical Hermite-truncation closures and a canonical
solver-embedded interface-flux trainer. The classical benchmark suite
reproduces results from Palisso et al.,
[arXiv:2412.07073](https://arxiv.org/abs/2412.07073).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Regenerate the linear-Landau benchmark.
python -m benchmarks.fh_benchmarks_2412_07073_jax \
  linear_landau --outdir out_bench
```

The core requirements are Python 3.10 or newer, JAX, NumPy, SciPy, and
Matplotlib. Benchmark outputs are written below the requested output directory.

## Supported Workflows

The recommended public surface is organized by task rather than by source-file
layout:

| Task | Recommended command | Main output |
| --- | --- | --- |
| Classical benchmark suite | `./benchmarks/run_all_benchmarks.sh out_bench` | Benchmark figures and metrics |
| Train and evaluate one interface closure | `./model/train/run_fh_interface_flux_rollout.sh <outdir>` | Checkpoint, loss, Metric 1/2, and Fig. 10 |
| Compare rollout horizons | `./model/train/run_fh_interface_flux_horizon_sweep.sh <H-csv> <outdir>` | One comparable run per horizon |
| Check spline-to-Hermite quadrature convergence | `./model/diagnostics/run_projection_quadrature_convergence.sh <outdir>` | Refinement plots and numerical summaries |

Direct Python modules expose lower-level controls and `--help`, but the shell
wrappers above set the compatible defaults needed for reproducible workflows.

## Runtime Backend

The benchmark and model entry points print the active JAX backend at startup.
On macOS, `vpml` defaults to CPU because the numerical pipeline relies on
`float64` and complex dtypes. On Linux, `VPML_JAX_BACKEND=auto` leaves backend
selection to JAX.

```bash
export VPML_JAX_BACKEND=cpu
# or, with a CUDA-enabled JAX installation:
export VPML_JAX_BACKEND=gpu
```

The wrappers enable JAX x64 support. The learned rollout precision remains an
explicit training control and defaults to `float32`.

## Classical Benchmarks

Run individual paper benchmarks:

```bash
python -m benchmarks.fh_benchmarks_2412_07073_jax fig2          --outdir out_bench
python -m benchmarks.fh_benchmarks_2412_07073_jax fig3          --outdir out_bench
python -m benchmarks.fh_benchmarks_2412_07073_jax fig4          --outdir out_bench --Nv 20
python -m benchmarks.fh_benchmarks_2412_07073_jax linear_landau --method truncation --outdir out_bench
```

Run standalone nonlinear physical-grid simulations:

```bash
python -m benchmarks.fh_nonlinear_sim_jax two_stream \
  --outdir out_nl
python -m benchmarks.fh_nonlinear_sim_jax bump_on_tail \
  --system AC --outdir out_nl --vmin -12 --vmax 12
```

Use a canonical learned checkpoint in supported benchmark evaluations:

```bash
CHECKPOINT=out_bench/interface_flux_H128_T60/models/nv64/interface_closure.npz

python -m benchmarks.fh_benchmarks_2412_07073_jax linear_landau \
  --method learned --outdir out_bench \
  --learned-checkpoint "${CHECKPOINT}"

python -m benchmarks.fh_benchmarks_2412_07073_jax fig10_learned_comparison \
  --outdir out_bench --learned-checkpoint "${CHECKPOINT}"
```

The learned closure is state-dependent, so it is not supported in the `fig3`
response-function or `fig4` fixed-matrix eigenvalue benchmarks.

## Interface-Flux Closure

The canonical trainer advances the reduced Fourier-Hermite solver
autonomously and minimizes the complex interface-flux error. Its physical and
statistical invariants are fixed:

- all positive Fourier modes participate in the loss;
- linear, weakly nonlinear, and strongly nonlinear regimes have equal weight;
- normalization is fixed and phase-isotropic;
- the closure is equilibrium-centered;
- training applies spatial-translation augmentation;
- virtual cutoffs cycle per optimizer step through `6,7,12,20,36,64`.

For batch size $B$, rollout horizon $H$, positive-mode set
$\mathcal K_+$, and regime weights $w_r=1/3$, the objective is

$$
\mathcal L_{\mathrm{IF}}^H(\theta)
=
\sum_r \frac{w_r}{2BH|\mathcal K_+|}
\sum_{i=1}^{B}\sum_{h=0}^{H-1}\sum_{k\in\mathcal K_+}
\frac{|q^\theta_{r,i,h,k}-q^\star_{r,i,h,k}|^2}{\sigma_r^2}.
$$

### Train and evaluate

```bash
TRAIN_ROLLOUT_HORIZON=128 \
TRAIN_T_FINAL=60 \
./model/train/run_fh_interface_flux_rollout.sh \
  out_bench/interface_flux_H128_T60
```

The default run trains one shared `Nv=64` closure and evaluates every
configured training initial condition. Every sampled time that can start a
complete \(H\)-step window is eligible for optimization; there is no held-out
validation split. The per-IC evaluations are therefore in-sample scientific
diagnostics, not a generalization estimate.

The teacher evolves a physical velocity grid with `TEACHER_NV=512`. Hermite
coefficients are then computed from the reconstructed spline using the finer
projection quadrature `TEACHER_PROJECTION_NV=4096`. These are distinct grids:
the latter improves the integral used to form training targets without changing
the physical teacher simulation.

Principal artifacts:

```text
models/nv64/interface_closure.npz
models/nv64/interface_closure.metrics.npz
models/nv64/interface_closure.loss.png
models/nv64/interface_closure_interface_flux_histories.npz
evaluation_cases/<case>/nv_sweep_metric1.png
evaluation_cases/<case>/nv_sweep_metric2.png
evaluation_cases/<case>/fig10_learned_vs_nonlocal_nv_sweep_phase_space.png
```

Common numerical controls:

| Environment variable | Meaning | Default |
| --- | --- | ---: |
| `TRAIN_ROLLOUT_HORIZON` | Differentiated solver steps per window | `128` |
| `TRAIN_T_FINAL` | Final teacher time for every regime | `60` |
| `TRAIN_BATCH_SIZE` | Anchors per optimizer step | `64` |
| `TRAIN_STEPS_PER_EPOCH` | Optimizer steps per epoch | `30` |
| `TRAIN_EPOCHS` | Number of epochs | `100` |
| `TRAIN_LR` | Learning rate | `1e-4` |
| `TRAIN_PRECISION` | Learned rollout precision | `float32` |
| `TRAIN_SEED` | Training and augmentation seed | `0` |
| `TRAIN_HISTORY_STRIDE` | Teacher-anchor stride | `20` |
| `TEACHER_NV` | Physical teacher velocity points | `512` |
| `TEACHER_PROJECTION_NV` | Spline-to-Hermite quadrature points | `4096` |

Reuse the checkpoint in an existing run directory:

```bash
RUN_TRAIN=0 \
./model/train/run_fh_interface_flux_rollout.sh \
  out_bench/interface_flux_H128_T60
```

Set `RUN_EVAL=0` to train without post-training evaluation. Set
`EVAL_TRAINING_CASES=0` to evaluate one configured nonlinear case at the run
root instead of generating the complete per-IC set.

### Controlled horizon sweep

Use the sweep wrapper when \(H\) should be the only changed training control:

```bash
./model/train/run_fh_interface_flux_horizon_sweep.sh \
  1,128,256 \
  out_bench/interface_flux_horizon_sweep
```

Each run is stored under `H<horizon>/`.

## Diagnostics

### Projection-quadrature convergence

Validate the spline-to-Hermite projection independently of model training:

```bash
./model/diagnostics/run_projection_quadrature_convergence.sh \
  out_bench/projection_quadrature_convergence
```

The default diagnostic keeps the physical teacher at `Nv=512`, runs to
`T=120`, and compares projection quadratures from 512 through 16,384 points.
It does not train or modify a closure checkpoint.

### Regenerate a loss figure

```bash
python -m model.diagnostics.plot_training_loss \
  --metrics out_bench/interface_flux_H128_T60/models/nv64/interface_closure.metrics.npz \
  --output out_bench/interface_flux_H128_T60/models/nv64/interface_closure.loss.regenerated.png
```

The plot derives its loss equation and scales from saved metadata. Canonical
loading also maps retained solver-embedded exact-q checkpoints to the current
runtime identifiers; checkpoints from removed trainer families are unsupported.

### Inspect the Hermite spectrum

```bash
python -m model.diagnostics.plot_hermite_spectrum \
  --cache out_bench/interface_flux_H128_T60/models/nv64/interface_closure_interface_flux_histories.npz \
  --target-nv 64 \
  --splits train
```

Use `python -m model.diagnostics.plot_hermite_spectrum --help` for regime,
split, and output controls.

### Render a phase-space video

The optional video diagnostic reruns the raw-HR teacher, unclosed `Nv=64`
solver, and learned-closure solver before encoding an MP4. It requires
`ffmpeg`, is substantially more expensive than plot-only diagnostics, and
refuses to overwrite existing output.

```bash
python -m model.diagnostics.render_phase_space_triptych_video \
  --run-root out_bench/interface_flux_H128_T60
```

## Verification

Run the test suite from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests
```

For command-specific options:

```bash
python -m model.train.interface_flux_rollout --help
python -m model.diagnostics.projection_quadrature_convergence --help
```
