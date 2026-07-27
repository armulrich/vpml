```
 ██╗   ██╗██████╗ ███╗   ███╗██╗
 ██║   ██║██╔══██╗████╗ ████║██║
 ██║   ██║██████╔╝██╔████╔██║██║
 ╚██╗ ██╔╝██╔═══╝ ██║╚██╔╝██║██║
  ╚████╔╝ ██║     ██║ ╚═╝ ██║███████╗
   ╚═══╝  ╚═╝     ╚═╝     ╚═╝╚══════╝

    Fourier-Hermite · Vlasov-Poisson · JAX
```

> **JAX Fourier–Hermite Vlasov–Poisson solver with a learned interface closure.**

`vpml` is a 1D1V collisionless-plasma solver that discretises the Vlasov–Poisson
system with **Fourier modes in space** and **orthonormal Hermite functions in
velocity**, advanced with an IMEX CNAB2 scheme on top of JAX. Classical closures
for the Hermite truncation boundary (hypercollisions, Hou–Li filtering, nonlocal
closure) are included so that results from **Palisso et al.,
[arXiv:2412.07073](https://arxiv.org/abs/2412.07073)** can be reproduced end-to-end.


**At a glance**

- Python ≥ 3.10 · JAX / `jax.numpy` · JAX x64 enabled, selectable rollout precision
- CPU by default (including on macOS); CUDA is opt-in
- Three sibling packages: `vpml/` (library), `benchmarks/` (paper figures), `model/` (learned closure)
- CLI entry points: `fh-nonlinear-sim`, `fh-benchmarks-2412-07073`, `fh-interface-flux-train`, `fh-learned-closure-eval`

---

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -e .

# Regenerate the linear-Landau benchmark (classical truncation closure)
python -m benchmarks.fh_benchmarks_2412_07073_jax linear_landau --outdir out_bench
```

Outputs land in `out_bench/` as `linear_landau_*.png`.

## Requirements

- `jax`, `jaxlib`, `numpy`, `matplotlib`, `scipy`

For better eigenvalue / root-finding accuracy, enable 64-bit JAX:

```bash
export JAX_ENABLE_X64=True
```

## Backend Selection

`vpml` bootstraps JAX before import and prints the active backend when the
main benchmark or model scripts start.

- On Linux, `VPML_JAX_BACKEND=auto` leaves backend selection to JAX.
- On macOS, `vpml` defaults to CPU rather than `jax-metal`, because this repo
  relies heavily on `float64` and complex dtypes.

Overrides:

```bash
export VPML_JAX_BACKEND=cpu
export VPML_JAX_BACKEND=gpu
```

If you actually want CUDA, install a CUDA-enabled JAX build:

```bash
pip install -U "jax[cuda13]"
```

---

## Benchmarks

### Nonlinear simulations

```bash
python -m benchmarks.fh_nonlinear_sim_jax two_stream --outdir out_nl
python -m benchmarks.fh_nonlinear_sim_jax bump_on_tail --system AC --outdir out_nl --vmin -12 --vmax 12
```

### Paper benchmarks (Palisso et al., arXiv:2412.07073)

```bash
python -m benchmarks.fh_benchmarks_2412_07073_jax fig2           --outdir out_bench
python -m benchmarks.fh_benchmarks_2412_07073_jax fig3           --outdir out_bench
python -m benchmarks.fh_benchmarks_2412_07073_jax fig4           --outdir out_bench --Nv 20
python -m benchmarks.fh_benchmarks_2412_07073_jax linear_landau  --method truncation --outdir out_bench
./benchmarks/run_all_benchmarks.sh out_bench
```

### With a learned checkpoint

```bash
python -m benchmarks.fh_benchmarks_2412_07073_jax linear_landau \
  --method learned --outdir out_bench \
  --learned-checkpoint out_model/interface_closure.npz

python -m benchmarks.fh_benchmarks_2412_07073_jax fig10_learned_comparison \
  --outdir out_bench --learned-checkpoint out_model/interface_closure.npz

LEARNED_CHECKPOINT=out_model/interface_closure.npz ./benchmarks/run_all_benchmarks.sh out_bench
```

The learned closure is intentionally **not** supported in the `fig3`
response-function or `fig4` eigenvalue benchmarks: it is state-dependent, not a
fixed modified-Hermite matrix.

---

## Interface-Flux Closure Workflow

The canonical trainer advances the reduced Fourier--Hermite solver
autonomously and minimizes error in its complex boundary flux \(q\). It uses all
positive Fourier modes, equal weights for the linear, weakly nonlinear, and
strongly nonlinear regimes, fixed phase-isotropic scales, equilibrium
centering, spatial-translation augmentation, and the per-step cutoff cycle
`6,7,12,20,36,64`.

For batch size \(B\), rollout horizon \(H\), positive-mode set
\(\mathcal K_+\), and regime weight \(w_r=1/3\), the retained objective is

\[
\mathcal L_{\mathrm{IF}}^H(\theta)
=
\sum_r \frac{w_r}{2BH|\mathcal K_+|}
\sum_{i=1}^{B}\sum_{h=0}^{H-1}\sum_{k\in\mathcal K_+}
\frac{|q^\theta_{r,i,h,k}-q^\star_{r,i,h,k}|^2}{\sigma_r^2}.
\]

### Train and evaluate

```bash
TRAIN_ROLLOUT_HORIZON=128 \
TRAIN_T_FINAL=60 \
./model/train/run_fh_interface_flux_rollout.sh \
  out_bench/interface_flux_H128_T60
```

The wrapper trains the canonical `Nv=64` model and evaluates every configured
training IC by default. Its principal artifacts are:

- `models/nv64/interface_closure.npz`
- `models/nv64/interface_closure.metrics.npz`
- `models/nv64/interface_closure.loss.png`
- `models/nv64/interface_closure_interface_flux_histories.npz`
- `evaluation_cases/<case>/nv_sweep_metric1.png`
- `evaluation_cases/<case>/nv_sweep_metric2.png`
- `evaluation_cases/<case>/fig10_learned_vs_nonlocal_nv_sweep_phase_space.png`

Set `EVAL_TRAINING_CASES=0` to produce one Metric 1/2 and raw-grid HR Fig. 10
comparison at the output root. Reuse an existing checkpoint without training:

```bash
RUN_TRAIN=0 \
./model/train/run_fh_interface_flux_rollout.sh \
  out_bench/interface_flux_H128_T60
```

The configurable numerical controls include `TRAIN_ROLLOUT_HORIZON`,
`TRAIN_T_FINAL`, `TRAIN_BATCH_SIZE`, `TRAIN_STEPS_PER_EPOCH`, `TRAIN_EPOCHS`,
`TRAIN_LR`, `TRAIN_PRECISION`, and `TRAIN_SEED`. The canonical physical and
normalization constraints are not exposed as competing training modes.

### Horizon sweep

Use the controlled sweep wrapper when \(H\) is the only intended difference:

```bash
./model/train/run_fh_interface_flux_horizon_sweep.sh \
  1,128,256 \
  out_bench/interface_flux_horizon_sweep
```

Each horizon is written to a separate `H<horizon>` directory.

### Plot-only regeneration

Regenerate a training-loss figure from saved metrics without retraining:

```bash
python -m model.diagnostics.plot_training_loss \
  --metrics out_bench/interface_flux_H128_T60/models/nv64/interface_closure.metrics.npz \
  --output out_bench/interface_flux_H128_T60/models/nv64/interface_closure.loss.regenerated.png
```

The figure derives its complete loss equation and scales from checkpoint
metadata. Retained solver-embedded exact-q checkpoints are mapped to the
canonical identifiers when loaded; checkpoints from removed trainer families
are intentionally unsupported.

### Phase-space video

Create a reusable raw-HR, unclosed `Nv=64`, and learned-interface-flux
triptych:

```bash
python -m model.diagnostics.render_phase_space_triptych_video \
  --run-root out_bench/interface_flux_H128_T60
```

---

<details>
<summary><b>Repo map &amp; design boundary</b></summary>

### Repo map

- `vpml/core.py` — Fourier–Hermite operators, closures, implicit/CNAB2 solvers, learned-closure runtime
- `vpml/linear_landau.py` — shared linear-Landau rollout helpers and dispersion / root-finding utilities
- `vpml/nonlinear_landau.py` — shared nonlinear-Landau rollout runtime for benchmarks and learned-model eval
- `vpml/physical_grid.py` — physical-grid semi-Lagrangian teacher solver and projection helpers
- `vpml/metrics/` — reusable rollout metrics
- `vpml/visualization/` — reusable plotting helpers
- `benchmarks/fh_benchmarks_2412_07073_jax.py` — paper benchmark regeneration for Palisso et al. (arXiv:2412.07073)
- `benchmarks/run_all_benchmarks.sh` — full benchmark shell entrypoint
- `benchmarks/run_linear_landau_suite.sh` — linear Landau benchmark shell entrypoint
- `benchmarks/fh_nonlinear_sim_jax.py` — standalone nonlinear physical-grid simulations
- `model/model.py` — thin learned-model surface built on top of `vpml`
- `model/train/interface_flux_rollout.py` — canonical solver-embedded interface-flux trainer
- `model/train/run_fh_interface_flux_rollout.sh` — canonical training and evaluation wrapper
- `model/train/run_fh_interface_flux_horizon_sweep.sh` — controlled rollout-horizon sweep
- `model/eval.py` — post-train learned-model evaluation
- `model/eval_nv_sweep.py` — learned-model nonlinear `N_v` sweep evaluation
- `model/eval_training_cases.py` — per-IC Metric 1/2 and raw-HR Fig. 10 evaluation
- `model/diagnostics/plot_training_loss.py` — metadata-driven plot-only loss regeneration
- `model/diagnostics/render_phase_space_triptych_video.py` — raw-HR/truncation/learned phase-space video
- `model/diagnostics/plot_hermite_spectrum.py` — Hermite-spectrum diagnostics

</details>
