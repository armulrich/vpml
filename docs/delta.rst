Using Delta
===========

Environment
-----------

Use a fixed checkout on Delta and a separate Python 3.12 environment. The
reviewed direct dependencies are in ``environments/delta/requirements.in``;
``requirements.lock`` pins the complete resolved environment with hashes.

.. code-block:: bash

   bash environments/delta/setup.sh \
     /absolute/path/to/python3.12 /absolute/path/to/new-venv

The helper requires Linux x86_64, refuses an existing environment and installs
hash-verified wheels. It does not submit a job. The CUDA 12 JAX wheels require
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
   export VPML_DATA_ROOT=/absolute/path/to/project-work/gaussian-amplitudes-v1

The launcher derives the code root from its location; ``VPML_CODE_ROOT`` can
select another fixed checkout explicitly. The adjacent ``.params.sh`` file
sets scheduler resources and points to the scientific parameter JSON.

.. code-block:: bash

   bash scripts/delta/run_gaussian_amplitudes_v1.sh prepare
   bash scripts/delta/run_gaussian_amplitudes_v1.sh validate
   bash scripts/delta/run_gaussian_amplitudes_v1.sh status
   bash scripts/delta/run_gaussian_amplitudes_v1.sh test-submit

Preparation writes the manifest, validation compares it with the configured
parameters, and status reads progress metadata. Status is not a full integrity
audit. ``test-submit`` uses ``sbatch --test-only``: a hypothetical job ID does
not indicate that a job was submitted or a GPU tested.

Submission is a separate action:

.. code-block:: bash

   bash scripts/delta/run_gaussian_amplitudes_v1.sh submit

This requests a real allocation: account ``bhvw-delta-gpu``, partition
``gpuA100x4``, one GPU, 16 CPUs and 60 GB host RAM for at most one hour.
Allocation time is charged even while the GPU is idle. These settings are
specific to Delta; another account or cluster needs its own reviewed launcher.
See `Delta job accounting <https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html>`_.

The worker's 3,300-second budget leaves a shutdown margin; Slurm also sends an
advance USR1 signal. There is no automatic requeue, retry or resubmission.
Compilation or slow I/O can still reach the hard limit, so recovery does not
depend solely on graceful shutdown.

Monitoring and continuation
---------------------------

Use the job ID returned by ``sbatch`` with ``squeue`` and ``sacct``. Slurm logs
are written under ``$VPML_DATA_ROOT/logs``; worker data are under
``$VPML_DATA_ROOT/reference``. Closing the SSH connection does not stop an
already submitted batch job.

To request a checkpointed stop, send USR1 to that job's steps. ``scancel JOBID``
cancels the specific job immediately; recovery then uses the last committed
checkpoint. Confirm that the previous job has ended before submitting again.
The same manifest and output directory continue the same dataset.

.. code-block:: bash

   squeue -j JOBID
   sacct -j JOBID
   scancel --signal=USR1 JOBID

Storage planning
----------------

Checkpoint storage is substantially larger than compact training output.
For the versioned parameters, each case retains about 110.6 MB of every-step
observations, 134.2 MB for two kinetic states and 22.1 MB of sampled outputs.
The 1,200-case dataset therefore needs approximately 321 GB plus diagnostics,
metadata and interrupted writes. Optional tangent states require more space.

Use project work storage for active I/O; do not use node-local temporary
storage as the only checkpoint copy. Check project quota rather than inferring
it from filesystem free space. Preserve complete results before any cleanup.
See `Delta storage documentation <https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html>`_.

Measure throughput with the actual observation, diagnostic and checkpoint
settings. Timing a solver kernel alone does not estimate all data-generation
costs.
