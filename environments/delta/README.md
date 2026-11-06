# Delta environment

This environment targets **Python 3.12 on Linux x86_64**, with JAX 0.10.0,
NumPy 2.4.4, SciPy 1.17.1 and Matplotlib 3.10.9. CUDA 12 runtime libraries are
installed from pinned wheels. Delta GPU execution is **not yet verified**.

The CUDA 12 JAX build requires an NVIDIA driver of at least 525 on Linux.
Check the allocated node's driver and test actual JAX computation before
claiming GPU compatibility. Installing this environment or checking scheduler
eligibility does not establish that compatibility. See the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).

`requirements.in` lists the reviewed direct dependencies. `requirements.lock`
also pins every resolved transitive dependency and its allowed SHA-256 hashes.
The lock was resolved with uv 0.10.2 for the target platform without installing
packages or allocating hardware:

```bash
uv --no-config --no-progress --no-python-downloads \
  --cache-dir /tmp/vpml-delta-uv-cache pip compile \
  environments/delta/requirements.in \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --only-binary :all: \
  --default-index https://pypi.org/simple --no-header \
  --output-file environments/delta/requirements.lock
```

After choosing an existing Python 3.12 executable and a new environment path on
Delta, explicit setup is:

```bash
bash environments/delta/setup.sh /absolute/path/to/python3.12 /absolute/path/to/new-venv
```

The helper refuses an existing environment, installs only hash-verified wheels
from PyPI and runs `pip check`. It does not install the project, submit a job,
or run a GPU check. The dataset CLI imports the reviewed checkout directly.
If installation fails, inspect the incomplete environment before choosing a
new path; the helper does not delete or silently repair environments.

Do not load a second CUDA toolkit over the wheel-provided CUDA libraries.
The setup and batch helpers clear `LD_LIBRARY_PATH`, `LD_PRELOAD`, `PYTHONPATH`
and `PYTHONHOME` within their own processes to avoid inherited libraries and
Python installations. They preserve Slurm's `CUDA_VISIBLE_DEVICES` selection.
Source code, dependency lock and manifest should remain fixed during a run;
the worker records run provenance separately from the scientific manifest.
