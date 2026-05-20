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

- Python ≥ 3.10 · JAX / `jax.numpy` · `float64` throughout
- CPU by default (including on macOS); CUDA is opt-in
- Three sibling packages: `vpml/` (library), `benchmarks/` (paper figures), `model/` (learned closure)
- CLI entry points: `fh-nonlinear-sim`, `fh-benchmarks-2412-07073`, `fh-ml-tail-closure-train`, `fh-learned-closure-eval`

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

## Learned Closure Workflow

### Train

```bash
python -m model.train.train \
  --checkpoint     out_model/interface_closure.npz \
  --dataset-cache  out_model/interface_closure_dataset.npz
```

Writes:

- `out_model/interface_closure.npz`
- `out_model/interface_closure.metrics.npz`
- `out_model/interface_closure_dataset.npz`
- `out_model/interface_closure.loss.png` (if `--loss-plot` is passed)

The main offline lane is `q_only`; the pure-online lane is kept separate as `online_rollout`.

### Evaluate

```bash
python -m model.eval \
  --checkpoint out_model/interface_closure.npz \
  --outdir     out_model/eval
```

Writes `summary.json`, per-case `*.npz` rollouts, and `*_summary.png` plots
under `heldout_landau/` and `benchmark_rollouts/`.

### Sweep deployment `N_v`

| Wrapper                                                         | What it sweeps                             |
| --------------------------------------------------------------- | ------------------------------------------ |
| `run_fh_offline_qloss_ladder.sh`                                | Fourier--Hermite offline fixed-ratio `q_only` baseline |
| `run_fh_online_q_finetune.sh`                                   | Offline `q_only` pretraining plus online `q` fine-tuning |
| `run_fh_online_projected_xv_rollout.sh`                         | Fourier--Hermite projected `x,v` online rollout loss |
| `run_spline_grid_online_residual_rollout.sh`                    | Physical spline-grid online residual rollout loss |
| `run_nv_sweep_online_rollout.sh`                                | Shared online-rollout driver used by wrappers |

Example:

```bash
./model/train/run_fh_offline_qloss_ladder.sh out_bench/fh_offline_qloss_ladder
```

The Fourier--Hermite wrappers train one checkpoint per deployment `N_v` and then
call `python -m model.eval_nv_sweep ...`, emitting the same evaluation layout:
Metric 1 plots HR against learned rollouts, while Metric 2 and Fig. 10 include
the truncation (`q=0`) baseline beside the learned closure.

- `summary.json`
- `nv_sweep_metric1.png`, `nv_sweep_metric2.png`
- `fig10_learned_vs_nonlocal_nv_sweep_phase_space.png`
- `cases/*.npz`

For the offline wrapper, per-`N_v` dataset caches live under
`models/nv*/interface_closure_dataset.npz`. The spline-grid wrapper writes its
own spline-grid metrics and phase-space plots under its output directory.

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
- `model/train/train.py` — learned interface-closure training entrypoint
- `model/train/data.py` — dataset / cache / reference-building surface for learned-closure workflows
- `model/eval.py` — post-train learned-model evaluation
- `model/eval_nv_sweep.py` — learned-model nonlinear `N_v` sweep evaluation
- `model/train/run_fh_offline_qloss_ladder.sh` — per-`N_v` fixed-ratio offline `q_only` baseline wrapper
- `model/train/run_fh_online_q_finetune.sh` — offline `q_only` pretraining followed by online `q` fine-tuning wrapper
- `model/train/run_fh_online_projected_xv_rollout.sh` — projected `x,v` online rollout wrapper
- `model/train/run_spline_grid_online_residual_rollout.sh` — physical spline-grid online residual rollout wrapper
- `model/train/run_nv_sweep_online_rollout.sh` — shared per-`N_v` online rollout driver used by the Fourier--Hermite online wrappers
  Default recipe: denser nonlinear amplitude coverage, `TEACHER_NX=256`, `TEACHER_DT=0.005`, and `TRAIN_ONLINE_V_PROBES=256` for the top-end `N_v` comparison lane
  Runtime note: the wrapper prebuilds a shared `online_reference_dataset.npz`, reuses it across `N_v`, and enables bounded per-`N_v` parallel training on larger CPU boxes
