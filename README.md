# JAX Fourier–Hermite Vlasov–Poisson 

This folder contains a small, self-contained JAX implementation of 1D1V Fourier–Hermite Vlasov–Poisson
along with:

- **Nonlinear simulations** (two-stream + bump-on-tail)
- **Linear benchmarks** reproducing key figures from Palisso et al. (arXiv:2412.07073)

`fh_core_jax.py` is the shared “core” (operators + IMEX time stepper). The other two files are runnable CLIs.

## Requirements

- Python 3.10+ (this repo is currently using Python 3.12)
- `jax`, `jaxlib`, `numpy`, `matplotlib`
- `scipy` (required for the benchmark script’s plasma-dispersion function in this environment)

For better eigenvalue/root-finding accuracy, enable 64-bit:

```bash
export JAX_ENABLE_X64=True
```

## How to run

### Nonlinear simulations

```bash
python fh_nonlinear_sim_jax.py two_stream --outdir out_nl
python fh_nonlinear_sim_jax.py bump_on_tail --system AC --outdir out_nl
```

Outputs go to `out_nl/` (PNGs + `.npz` energy time series).

### Benchmarks (arXiv:2412.07073)

```bash
python fh_benchmarks_2412_07073_jax.py fig2 --outdir out_bench
python fh_benchmarks_2412_07073_jax.py fig3 --outdir out_bench
python fh_benchmarks_2412_07073_jax.py fig4 --outdir out_bench --Nv 20
python fh_benchmarks_2412_07073_jax.py linear_landau --method truncation --outdir out_bench
```

Outputs go to `out_bench/` (PNGs + `.npz` data dumps).

## Script map (quick)

- `fh_core_jax.py`: Fourier/Hermite utilities + IMEX CNAB2 integrator + damping/closure operators.
- `fh_nonlinear_sim_jax.py`: nonlinear runs (two-stream, bump-on-tail A/C/AC), saves snapshots + energies.
- `fh_benchmarks_2412_07073_jax.py`: Fig. 2/3/4-style scans + linear Landau damping time-domain run.

