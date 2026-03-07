# JAX Fourier–Hermite Vlasov–Poisson 

This repo exposes the reusable Fourier–Hermite implementation as an importable `vpml` package and keeps the runnable scripts under `benchmarks/`.

- `vpml/`: importable core operators, IMEX time stepper, and implicit-midpoint/JFNK helper
- `benchmarks/fh_nonlinear_sim_jax.py`: nonlinear two-stream and bump-on-tail runs
- `benchmarks/fh_benchmarks_2412_07073_jax.py`: linear benchmarks reproducing key figures from Palisso et al. (arXiv:2412.07073)
- `benchmarks/run_all_benchmarks.sh`: shell entry point for the full benchmark suite

## Requirements

- Python 3.10+ (this repo is currently using Python 3.12)
- `jax`, `jaxlib`, `numpy`, `matplotlib`
- `scipy` (required for the benchmark script’s plasma-dispersion function in this environment)

For better eigenvalue/root-finding accuracy, enable 64-bit:

```bash
export JAX_ENABLE_X64=True
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
python benchmarks/fh_nonlinear_sim_jax.py bump_on_tail --system AC --outdir out_nl
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

## Package map

- `vpml`: public package namespace
- `vpml.core`: Fourier/Hermite utilities, CNAB2 integrator, implicit-midpoint/JFNK stepper, damping, and closure operators
- `benchmarks/`: runnable scripts and shell helpers that import `vpml`
