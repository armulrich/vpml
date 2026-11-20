Using Delta
===========

Environment
-----------

Use a fixed checkout on Delta and a separate Python 3.12 environment. The
reviewed direct dependencies are in ``environments/delta/requirements.in``;
``requirements.lock`` pins exact versions of all direct and transitive
dependencies. It changes only when dependencies are deliberately updated,
not for each dataset run.

.. code-block:: bash

   bash environments/delta/setup.sh \
     /absolute/path/to/python3.12 /absolute/path/to/new-venv

The helper requires Linux x86_64, refuses an existing environment and installs
wheels at the pinned versions and the reviewed checkout as an editable package.
The launcher checks that its Python imports that checkout. Setup does not submit
a job. The CUDA 12 JAX wheels require
an NVIDIA Linux driver of at least 525; confirm the allocated node's driver and
actual JAX execution before relying on GPU compatibility. See the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html#nvidia-gpu>`_.

The helpers clear inherited CUDA library and Python paths within their own
processes so the wheel-provided libraries are used. Slurm's
``CUDA_VISIBLE_DEVICES`` selection is preserved.

Preparation and scheduling
--------------------------

Choose an existing project work directory and check its permissions and quota.
Set the environment executable and a dedicated dataset directory:

.. code-block:: bash

   export VPML_PYTHON=/absolute/path/to/prepared-venv/bin/python
   export VPML_DATA_ROOT=/absolute/path/to/project-work/gaussian-amplitudes
   export VPML_RUN_PARAMS=/absolute/path/to/vpml/scripts/delta/gaussian_amplitudes.params.sh

The launcher derives the code root from its location; ``VPML_CODE_ROOT`` can
select another fixed checkout explicitly. ``VPML_RUN_PARAMS`` selects a
fixed ``.params.sh`` file containing scheduler resources, a scientific
parameter JSON, optional case IDs and the failure policy. Omitting it selects
the adjacent ``gaussian_amplitudes.params.sh``.

The supplied JSON contains all 1,200 initial conditions with 512 spatial points,
4,096 velocity points and a timestep of 0.0125.
Its parameter file leaves ``VPML_CASE_IDS`` empty, so the worker processes the
full manifest in order within its time budget and stops on the first failure.
Explicit case IDs can restrict the selection; they are checked before submission
and again in the worker. This selection does not establish numerical accuracy.
Changing resolution requires a new dataset directory and evolution from the
original initial conditions; an old-grid checkpoint is not resumed.

The ``gaussian-moments`` output format stores the required fields, raw moments
and electric field. Generation checks finite values, positive density and
pressure, and checkpoint integrity. It does not compute extra per-step
diagnostics, issue numerical-quality warnings or impose conservation and
refinement tolerances. Conservation summaries can be computed afterward from
saved moments and electric fields. Completion records output validity;
numerical accuracy is assessed separately.

.. code-block:: bash

   bash scripts/delta/run_gaussian_amplitudes.sh prepare
   bash scripts/delta/run_gaussian_amplitudes.sh validate
   bash scripts/delta/run_gaussian_amplitudes.sh status
   bash scripts/delta/run_gaussian_amplitudes.sh budget
   bash scripts/delta/run_gaussian_amplitudes.sh test-submit

Preparation writes the manifest, validation compares it with the configured
parameters, and status reads progress metadata. Status is not a full integrity
audit. ``budget`` reads the allocation allowance and Slurm accounting without
submitting work. ``test-submit`` checks the billing configuration and uses
``sbatch --test-only``: a hypothetical job ID does not indicate that a job was
submitted or a GPU tested.

Submission is a separate action:

.. code-block:: bash

   bash scripts/delta/run_gaussian_amplitudes.sh submit

Before submission, the launcher freezes a read-only record under
``$VPML_DATA_ROOT/submissions`` with parameter texts, manifest and source hashes,
selected cases and resource limits. It separately records the accepted Slurm job
ID in the dataset's budget record. An intent alone does not prove acceptance or
completion. When using a Git archive,
set ``VPML_SOURCE_COMMIT`` to its full commit SHA. The record distinguishes that
declaration from a commit observed in a checkout. Stage and preserve the source,
environment and parameters before submission: ``sbatch`` does not transfer them.

This requests a real allocation: account ``bhvw-delta-gpu``, partition
``gpuA100x4``, one GPU, 16 CPUs and 60 GB host RAM. The default requests at most
25 hours 55 minutes within a cumulative allowance of 26 additional SU.
Allocation time is charged even while the GPU is idle. These settings are
specific to Delta; another account or cluster needs its own reviewed launcher.
See `Delta job accounting <https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html>`_.

The launcher verifies the live billing configuration before submission. For
these resources the expected rate is at most 1 SU per allocated hour. It
reserves five minutes of the allowance for scheduler termination and accounting
headroom. Completion or a configured stop on failure releases the allocation
early; requesting time does not charge for time after the job ends.

The default worker budget is 93,000 seconds (25 hours 50 minutes). It is also
bounded by the actual allocation end time, leaving the configured shutdown
margin after setup and validation. Slurm sends an advance USR1 signal and
enforces the outer wall-time limit even if the client disconnects. There is no
automatic requeue, retry or resubmission. A time allowance is not a guarantee
that every selected trajectory reaches its prescribed simulation time.

The first submission under this controller creates ``$VPML_DATA_ROOT/budget.json``.
The allowance applies to jobs recorded there; historical jobs from before its
creation are not retroactively charged against this new allowance. Each later
submission reconciles recorded job IDs with Slurm accounting, including failed
and cancelled jobs, and reduces the available wall time accordingly. It does
not grant a fresh allowance on restart or infer spending from changes in a
shared project balance. Preserve this record alongside the dataset.

Only one submission can update the budget at a time. An active job, missing
accounting or an ambiguous submission result blocks another submission. If the
connection fails during submission, inspect the recorded intent and Slurm queue
before resolving the uncertainty; do not delete the record and submit again.
Compilation, slow I/O or a crash can still interrupt checkpoint publication,
so recovery does not depend solely on graceful shutdown.

Monitoring and continuation
---------------------------

Use the job ID returned by ``sbatch`` with ``squeue`` and ``sacct``. Slurm logs
are written under ``$VPML_DATA_ROOT/logs``; worker data are under
``$VPML_DATA_ROOT/reference``. Closing the SSH connection does not stop an
already submitted batch job.

The ``status`` action reports completed and partial cases and their committed
steps; the ``budget`` action reports job states, charges and the remaining
allowance. A partial checkpoint can retain a ``running`` label after its job
ends: Slurm, rather than that label, determines whether computation is active.
Use committed steps to include partial trajectories in progress estimates.

To request a checkpointed stop, send USR1 to that job's steps. ``scancel JOBID``
cancels the specific job immediately; recovery then uses the last committed
checkpoint. Confirm that the previous job has ended before submitting again.
The same manifest and output directory continue the same dataset. Completed
cases are verified and skipped; partial cases resume from the last committed
checkpoint. An abrupt stop can lose the current uncommitted block, but not the
preceding published checkpoint. A crash may leave a case lock: confirm that its
owning job and process have ended before removing it. Preserve and investigate
any failure record before explicitly retrying the affected case.

.. code-block:: bash

   squeue -j JOBID
   sacct -j JOBID
   scancel --signal=USR1 JOBID

Continuing from a new source revision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage the reviewed commit in a new checkout and prepare an environment for
that checkout. Preserve the previous checkout, environment and submission
records. Reusing an editable environment that points at the old checkout does
not stage the new code, and ``sbatch`` does not copy source to a worker.
Keep ``VPML_DATA_ROOT`` set to the existing dataset directory and retain its
manifest and scientific parameters.

Compatible code revisions use the ordinary validation and submission commands
with the same dataset directory. Configure the launcher with the new fixed
checkout and environment paths:

.. code-block:: bash

   export VPML_CODE_ROOT=/absolute/path/to/new-reviewed-checkout
   export VPML_PYTHON=/absolute/path/to/new-prepared-venv/bin/python
   export VPML_RUN_PARAMS="$VPML_CODE_ROOT/scripts/delta/gaussian_amplitudes.params.sh"
   bash "$VPML_CODE_ROOT/scripts/delta/run_gaussian_amplitudes.sh" validate

For a staged Git archive, also set ``VPML_SOURCE_COMMIT`` to the new full commit
SHA. Review the scientific parameters in the selected parameter file against
the existing manifest. Login-node validation checks the saved protocol and
maintained restart contract before submission. The worker also checks its
actual runtime against the saved runtime. Source hashes record the code used
for each part of the run; source-byte equality is not the restart gate.
Supported existing protocols without an explicit numerical contract identifier
are handled by the ordinary reader, without rewriting their records.

After the previous worker has ended, a separate ``submit`` command requests
the next allocation. Completed cases are verified and skipped; partial cases
continue from their committed state. Original source records, case protocols
and provenance are retained, and new arrays record their actual source lineage.
This procedure describes the supported launcher workflow; local validation
does not establish that a revision has been staged or executed on Delta.

Storage planning
----------------

Checkpoint storage is substantially larger than compact training output.
At ``dt=0.0125``, ``final_time=120``, ``output_nx=128``, ``save_dt=0.025``
and float64 output, each case has 44.3 MB of sampled fields, raw moments,
electric field and time coordinates: 53.1 GB across 1,200 cases.
Every-step observations take another 88.5 MB per case. Two float64 kinetic
states need :math:`2N_xN_v\times8` bytes: 33.6 MB at ``nx=512, nv=4096``.
Retaining all observations, sampled outputs and restart states for 1,200 cases
therefore needs approximately 200 GB, plus file headers, metadata and
interrupted writes. These sizes use decimal MB and GB. Optional tangent states
and their electric fields require more space.

Use project work storage for active I/O; do not use node-local temporary
storage as the only checkpoint copy. Check project quota rather than inferring
it from filesystem free space. Preserve complete results before any cleanup.
See `Delta storage documentation <https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html>`_.

Measure throughput with the actual observation and checkpoint
settings. Timing a solver kernel alone does not estimate all data-generation
costs. Worker results separate advancement and observation, host transfer,
checkpoint writing and finalization times. Advancement
timing waits for device completion and includes compilation on the first block.

Training the Gaussian low moment closure
---------------------------------------

Stage the reviewed source revision and use the Delta environment's Python
interpreter. Give training a new durable run directory and point the input
at the completed reference directory. The batch step imports the staged
source and reads the existing reference trajectories in place.

.. code-block:: bash

   export VPML_REFERENCE_ROOT=/absolute/path/to/completed/reference
   export VPML_RUN_ROOT=/absolute/path/to/training/run
   export VPML_NORMALIZATION=/absolute/path/to/E10/normalization.npz
   launcher="$VPML_CODE_ROOT/scripts/delta/run_gaussian_training.py"
   "$VPML_PYTHON" "$launcher" test-submit
   "$VPML_PYTHON" "$launcher" submit
   "$VPML_PYTHON" "$launcher" budget

The default training allowance is 1 GPU SU. At 1 SU per allocated hour,
the first job requests at most 55 minutes with up to 50 minutes of worker
time. Slurm enforces the allocation deadline even if SSH disconnects. Each
submission starts one A100 step and writes its billing record, source
revision, input hashes and Slurm job ID under the training run directory.
The job writes progress in ``logs/slurm-JOBID.out`` and errors in
``logs/slurm-JOBID.err``. Reconnect, check ``squeue -j JOBID``, and read
those files to follow the same job.

On a new allocation, the worker opens the same run directory and resumes
its last published optimizer update and sampling position. New runs use
12 windows per gradient microbatch by default, with one optimizer step
per 48 windows. A resumed run uses the microbatch size in its saved
training protocol. The launcher checks an explicit override before submission. The
``--max-updates`` and ``--max-epochs`` options set absolute targets across
allocations. An explicit increase to the cumulative SU allowance preserves
all previous charges. For example, after approval for a total of 3 SU:

.. code-block:: bash

   "$VPML_PYTHON" "$launcher" extend-budget 3
   "$VPML_PYTHON" "$launcher" submit --wall-minutes 90 --max-epochs 2

The allowance change alone does not submit a job. Read completed-step counts
and validation from the training output; query ``budget`` for the separate
Slurm charges. Reuse the same data, normalization and training configuration
when continuing. The ``--cache`` option can select a reusable trajectory
cache on shared storage.
