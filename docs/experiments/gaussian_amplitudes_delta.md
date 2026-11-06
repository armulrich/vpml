# Gaussian reference generation on Delta

This workflow prepares and resumes the formal Gaussian reference dataset.
It does not train a model. The physical-grid Vlasov–Poisson solver is unchanged;
scheduling and checkpoint storage are outside the numerical kernels.

## Fixed scientific configuration

| Parameter | Values |
| --- | --- |
| Fundamental wavenumber | 0.3, 0.4, 0.5 |
| Fourier mode count | 3, 5 |
| Coefficient decay exponent | 1.2, 2.0 |
| Independent shapes per combination | 20 |
| Strength λ | 0.1, 0.3, 0.5, 0.7, 1.0 |
| Density, velocity, temperature amplitudes | 0.18, 0.25, 0.10 |
| Kinetic grid | 1024 spatial points, 8192 velocity points |
| Velocity interval | −8 to 8 |
| Integration | float64, dt = 0.01, final time = 120 |
| Output grid and interval | 128 points, 0.025 |
| Checkpoint interval | 100 integration steps |
| Compact output precision | float32 |

There are 240 shapes and 1,200 cases. Each wavenumber/mode/decay combination
contributes 16 training, two validation and two test shapes: **960/120/120
cases**. All five λ variants of a shape stay in the same split. These assignments
do not mean the references have already been qualified for training.

The versioned parameters are in scripts/delta/gaussian_amplitudes_v1.json.
Preparation records the actual normalized Fourier coefficients, seeds and
complete case identities in an immutable manifest. Execution reads that manifest.
Changed parameters require a new dataset directory; failed cases are not
silently replaced by new draws.

Density and temperature are positive exponential profiles normalized to mean
one. Velocity is centered using density weights. Their local Gaussian is not
renormalized on the finite velocity grid: tail loss and charge imbalance are
measured explicitly.

## Code organization

The vpml/datasets package separates initial conditions, manifests, observations,
evolution, diagnostics, checkpoints, provenance and the bounded runner.
scripts/gaussian_dataset.py exposes explicit prepare, validate, status and run
commands; scripts/delta contains the scheduler wrapper and run parameters.

Compiled kernels are reused across ICs sharing the numerical settings and
domain. A GPU worker checks actual JAX placement and float64 support. The older
Modal entrypoints remain available for their earlier experiments.

## Output definitions

The new schema is **gaussian-fields-v1**, not the older states.npy schema.
Velocity integration uses trapezoid weights. Quantities use the solver's
nondimensional units.

| Array | Axes and meaning |
| --- | --- |
| fields | (time, 5, position): n, u, p, Q, dQ/dx |
| raw_moments | (time, 3, position): M0, M1, M2, where Mj = ∫vʲf dv |
| electric | (time, position) |
| times | Output times, including 0 and 120 |
| diagnostics, diagnostic_times | Checks at every integration step, float64 |

Here u = M1/M0, p = ∫(v−u)²f dv and Q = ∫(v−u)³f dv, without a factor 1/2.
Raw moments have no equilibrium subtraction. Primitive/central moments are
evaluated on the kinetic grid before Fourier restriction; raw moments are
restricted separately. Converting filtered raw moments into primitive fields
generally produces a different quantity. The Fourier derivative uses physical
spatial coordinates, with explicit target-grid Nyquist handling.

Output times between integration steps use linear interpolation of observed
fields. This defines the 0.025 sampling protocol despite dt = 0.01. Saved
density and pressure are checked for positivity as well as fine-grid diagnostics.
The optional linearized mode evolves the tangent map about a normalized
homogeneous Maxwellian; it is disabled in the formal run configuration.

## Checkpoints and continuation

Each case has an immutable protocol.json and atomic progress.json publication
point. The protocol records the manifest, numerical settings, output definitions,
source hashes and runtime versions. Numeric NPZ bundles contain restart states,
observation chunks and final outputs; progress records their filenames, hashes,
shapes and dtypes. Loading never enables pickle.

New files are flushed and atomically renamed before progress is published.
The previous kinetic state remains recoverable. Files not referenced by progress
are uncommitted and never used for recovery. Interrupted temporary/bundle files
are tolerated only with the store's known naming format; no automatic orphan
cleanup is performed.

Continuation verifies all committed histories, retained states and final output
when present. Verified complete cases are skipped, partial cases resume at the
next step, and untouched cases initialize from recorded coefficients. For
example, t = 37 at dt = 0.01 resumes at step 3,701. The original initial row
remains the conservation baseline. Finalization can be repeated after a crash
between writing outputs and publishing completion.

An exclusive per-case lock records host, PID, job ID and an owner token.
Stale locks require operator investigation: confirm the owner is no longer
running before removing that one lock. The worker never steals an existing lock.

Finite numerical failures are checkpointed and flagged in failures/. A nonfinite
block is not published; the preceding valid checkpoint remains. Failed cases
are not retried automatically. A changed numerical/source protocol needs a
new dataset version, not edited checkpoint metadata.

## Numerical qualification

Reports check conservation, negative distribution values and negative mass,
boundary population, nonpositive density, charge imbalance and saved-field
positivity. Thresholds are declared engineering tolerances; no clipping repairs
failed distributions.

**Completion is separate from training acceptance.** Even a passing report
retains training_qualified = false until representative grid, timestep and
velocity-interval comparisons establish adequate accuracy. Local software
tests do not establish convergence of the full dataset.

## Storage and throughput

Default uncompressed payload per case is approximately:

| Retained data | Decimal MB |
| --- | ---: |
| Every-step float64 spatial observations | 110.6 |
| Two float64 kinetic restart states | 134.2 |
| Sampled float32 spatial outputs | 22.1 |
| Diagnostics, time arrays, metadata | Additional |

The full 1,200 cases therefore need roughly **321 GB plus overhead**, not the
earlier 20 GB estimate for a smaller compact training representation.
Interrupted orphan writes and optional tangent states need extra room.
Recovery data is deliberately retained; archival cleanup requires a separately
agreed verification and retention policy.

Use project work storage for active I/O and check the actual quota before
submission. Home storage is for code/small files; node-local temporary storage
must not be the only copy of checkpoints.
[Delta storage documentation](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html)

Older solver timings are not measurements of this pipeline. Additional
observables, diagnostics, compilation and checkpoint I/O change throughput.
Measure elapsed time and Slurm charges before extrapolating the full run.

## Preparation and compute boundary

The environments/delta recipe pins a Linux Python 3.12 CUDA JAX environment.
It is separate from the local CPU environment. Driver compatibility and actual
GPU execution still require validation; installation alone does not establish
either. [JAX installation guidance](https://docs.jax.dev/en/latest/installation.html)

Use a fixed source checkout. Set VPML_PYTHON to the absolute environment Python
path and VPML_DATA_ROOT to a dedicated absolute run directory. The launcher
derives the code root from its location. Do not edit source while a job runs.

    bash scripts/delta/run_gaussian_amplitudes_v1.sh prepare
    bash scripts/delta/run_gaussian_amplitudes_v1.sh validate
    bash scripts/delta/run_gaussian_amplitudes_v1.sh test-submit
    bash scripts/delta/run_gaussian_amplitudes_v1.sh status

Preparation only writes the manifest. Validation compares it with the run
parameters. The test-submit action uses Slurm --test-only and does not submit
or execute a job. Status reads metadata; it is not a full integrity check.

**The submit action is the separate, explicit compute boundary.** It requests
one A100 and one worker for at most one hour. The worker has an internal
3,300-second budget, and Slurm sends an advance signal before its hard limit.
Signal handlers only request stopping; checkpoint writes occur in normal code.
Compilation or slow I/O can still hit the hard limit, so recovery never relies
solely on graceful shutdown.

There is no automatic requeue, resubmission or retry. A submitted batch job
survives laptop/SSH disconnection. Inspect it using squeue/sacct. Request an
early graceful stop with Slurm USR1 to the job steps; use scancel for immediate
cancellation, accepting recovery from the last committed checkpoint.
Resubmitting the same configuration continues the same dataset.

Neither setup nor a default script invocation submits a job. Passwords, Duo
codes and SSH control sockets are never part of code, parameters or run records.

## Validation

Tests use analytical initial conditions, Gaussian-mixture moments, Fourier
Nyquist examples, unchanged-solver comparisons and tiny CPU trajectories.
Interrupted and uninterrupted trajectories are compared, including full restart
states and exact time indices. Storage tests inject publication failures and
corruption; launcher tests use a fake scheduler only.

    JAX_PLATFORMS=cpu VPML_JAX_BACKEND=cpu python -m unittest discover -s tests -p 'test_dataset_*.py' -v
    python -m unittest discover -s tests -p 'test_delta_launcher.py' -v
    shellcheck scripts/delta/*.sh

Actual GPU execution, full-grid numerical qualification and production throughput
remain separate, unperformed checks.

### Local verification on 2026-11-06

The local Python 3.12 CPU environment passed 63 dataset tests, seven mocked
launcher tests and five existing Gaussian regression tests (75 total).
ShellCheck, Bash syntax and Git whitespace checks passed. The actual prepare,
validate and status actions were also exercised locally with the versioned run
parameters: 1,200 pending cases, zero complete, partial or failed cases, and no
reference evolution.

The prepared local manifest is under
out_bench/gaussian_amplitudes_delta_20261106/manifest.json (ignored generated
output). It can be regenerated from the committed parameters. The environment
lock was resolved for Linux without installing it. No SSH operation, Delta
environment installation, GPU allocation, job submission or model training was
part of this implementation.
