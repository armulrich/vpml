# Gaussian amplitude dataset on Delta

The launcher has no default action. It does not log in, provision hardware,
install packages, or automatically retry a job. Use it from a Delta login
shell after transferring a fixed, reviewed checkout and explicitly preparing
the environment described in `environments/delta/README.md`.

`gaussian_amplitudes_v1.params.sh` selects the resources and the adjacent
`gaussian_amplitudes_v1.json` contains the scientific parameters. The JSON
defines 240 initial shapes and five strengths per shape, giving 1,200 cases.
Each parameter combination contributes 16/2/2 train/validation/test shapes;
all five strengths remain in the same split. Preparation records the actual
coefficients in an immutable manifest. Validation compares it against this
run's parameters; changing settings requires reviewing a new dataset version.

Set these paths explicitly. They are not SSH credentials:

```bash
export VPML_CODE_ROOT=/absolute/path/to/reviewed/vpml
export VPML_PYTHON=/absolute/path/to/prepared-venv/bin/python
export VPML_DATA_ROOT=/work/hdd/bhvw/aulrich/vpml/gaussian-amplitudes-v1
```

The data path is a suggestion, not a verified allocation. Check the actual
project work path, permissions and `quota` first. Dataset histories and full
kinetic checkpoints need substantially more space than compact training
arrays alone. Do not use node-local `/tmp` as the only checkpoint copy, and
do not infer your project quota from `df` free space.

These actions do not submit jobs or evolve trajectories:

```bash
bash scripts/delta/run_gaussian_amplitudes_v1.sh prepare
bash scripts/delta/run_gaussian_amplitudes_v1.sh validate
bash scripts/delta/run_gaussian_amplitudes_v1.sh status
bash scripts/delta/run_gaussian_amplitudes_v1.sh test-submit
```

`test-submit` runs `sbatch --test-only` after validating the scientific
manifest. It may print a hypothetical job ID and start time; it neither
submits that job nor verifies GPU execution.

**Only the following action submits a real, chargeable allocation:**

```bash
bash scripts/delta/run_gaussian_amplitudes_v1.sh submit
```

The fixed request uses account `bhvw-delta-gpu`, partition `gpuA100x4`, one
A100, 16 CPUs and 60 GB host RAM for at most one hour. The worker's 3,300-second
budget leaves time for checkpointing and shutdown; Slurm also sends USR1
before the allocation expires. There is no array, automatic requeue or
automatic resubmission. The job ID returned by Slurm identifies the job for
`squeue`, `sacct` and `scancel`; cancel only that specific job.

The manifest is stored at `$VPML_DATA_ROOT/manifest.json`, worker output at
`$VPML_DATA_ROOT/reference`, and Slurm logs at `$VPML_DATA_ROOT/logs`.
Resubmitting explicitly uses the same manifest and output directory so the
worker can validate completed cases and resume partial cases. Verify the
previous job has ended before resubmitting. Closing SSH does not stop an
already submitted batch job.

The shell scripts and mocked submission behavior are checked locally. The
environment has not been installed on Delta, the batch job has not been
submitted, and CUDA/JAX execution on Delta has not yet been tested. Numerical
qualification of generated cases is a separate requirement before training.
