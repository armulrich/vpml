# VPML

VPML provides numerical solvers and learned closures for the collisionless
Vlasov–Poisson system in one spatial and one velocity dimension. It includes
physical-grid kinetic reference generation, Fourier–Hermite benchmarks and
low-moment models with history or persistent neural memory.

## Install

Use Python 3.12 for the documented environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For GPU work on Delta, use the pinned environment and instructions in the
[Delta guide](docs/delta.rst). An editable CPU installation is not a substitute
for a verified CUDA environment.

## Choose a workflow

| Task | Documentation |
| --- | --- |
| Install and inspect initial fields | [Getting started](docs/getting_started.rst) |
| Run benchmarks or train a closure | [Simulation and training](docs/workflows.rst) |
| Prepare Gaussian reference data | [Dataset definition and continuation](docs/datasets.rst) |
| Submit and monitor a Delta job | [Delta usage](docs/delta.rst) |
| Use Python interfaces | [API reference](docs/api.rst) |
| Run tests or change the code | [Development](docs/development.rst) |

For example, run a local linear-Landau benchmark:

```bash
VPML_JAX_BACKEND=cpu JAX_PLATFORMS=cpu \
  python -m vpml.benchmarks.fh_benchmarks_2412_07073_jax \
  linear_landau --outdir out_bench/linear_landau
```

The benchmark performs a calculation and writes its outputs under `out_bench`.
Dataset preparation, validation, reference generation and model training are
separate commands. See the corresponding guide before launching a workload.

## Read the documentation

The Sphinx site contains usage, numerical conventions and API documentation:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. The [documentation index](docs/index.rst)
is also readable directly in the repository.

Keep run logs, generated manifests, checkpoints, figures and numerical reports
with their run output, outside tracked documentation. Input parameter files,
source code, tests and dependency locks remain in version control.
