Simulation, training and plotting
=================================

Kinetic simulations and benchmarks
----------------------------------

The benchmark suite reproduces the Fourier–Hermite examples in
`Issan et al. <https://arxiv.org/abs/2412.07073>`_. Run the suite or inspect
the available problems:

.. code-block:: bash

   bash benchmarks/run_all_benchmarks.sh out_bench/benchmarks
   python -m vpml.benchmarks.fh_benchmarks_2412_07073_jax --help

For physical-grid two-stream or bump-on-tail simulations:

.. code-block:: bash

   python -m vpml.benchmarks.fh_nonlinear_sim_jax two_stream --outdir out_bench/two_stream
   python -m vpml.benchmarks.fh_nonlinear_sim_jax bump_on_tail --help

Use :doc:`datasets` when the goal is a reproducible collection of Gaussian
reference trajectories rather than one benchmark calculation.

Closure training
----------------

The Fourier–Hermite interface-flux trainer and the low-moment FNO trainer use
different state representations and data formats. Select the appropriate
entry point before preparing training inputs:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Model
     - Entry point
   * - Fourier–Hermite interface closure
     - ``python -m vpml.training.interface_train --help``
   * - History or persistent-memory low-moment FNO
     - ``python -m vpml.training.low_moment_train --help``
   * - Persistent-memory low-moment FNO on Gaussian moment trajectories
     - ``python -m vpml.cli.gaussian_training --help``

For the interface closure, the maintained workflow script configures the solver,
reference cache, training and evaluation:

.. code-block:: bash

   bash model/train/run_fh_interface_flux_rollout.sh out_bench/interface_closure

The low-moment option ``--memory-backend burles_latent_fno`` supports a history-only
model with ``--latent-memory-dim 0`` or persistent memory with
``--latent-memory-dim 6``. Its expected inputs are defined by the trainer;
the Gaussian ``gaussian-moments`` and historical ``gaussian-fields-v1`` outputs
are not interchangeable with an older ``states.npy`` array.

Training on Gaussian trajectories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Gaussian trainer reads the manifest from a completed reference directory.
The train, validation and test split is assigned by initial field shape and is
shared by all five strengths. Each 48-window training batch draws from shuffled
trajectory and start-time pairs. Every five batches give each strength equal
exposure. A cache stores each trajectory's conserved moments once on disk.

The model initializes its six latent channels from the reference moments at
the start of each window and an equilibrium-padded history. It predicts 600
steps at a spacing of 0.025. The loss averages the per-window relative L2
error in density, velocity, pressure and electric field over the whole
15-unit interval. Validation uses a fixed set of windows paired across
strengths. The test split is reserved for final evaluation.

Use the original E10 normalization archive with fresh model weights. A
continuation uses the same run directory, dataset and normalization. Each
checkpoint records the parameters, optimizer state and next batch. Time
limits and cumulative update or epoch targets control how long the worker
runs. Extending the target preserves the learning-rate schedule in optimizer
updates. For submission and accounting on Delta, see :doc:`delta`.

Historical Modal runs
~~~~~~~~~~~~~~~~~~~~~

The retained Modal launchers reproduce their own fixed configurations. They
require the referenced cache and Modal authentication before use. Staging
uploads data; remote preflight allocates a GPU; launch commands train models.
These are distinct operations:

.. code-block:: bash

   bash model/train/stage_modal_burles_history_fno.sh
   modal run model/train/modal_burles_history_fno.py::preflight
   bash model/train/run_modal_burles_history_fno_E20.sh
   bash model/train/download_modal_burles_history_fno.sh

The latent-model counterpart is
``model/train/run_modal_burles_latent_fno_E20.sh``. These launchers resume their
recorded lineage; changing a scientific setting requires a new configuration
and output location. Keep their archived source snapshots available: a
maintained package refactor does not replace a frozen experiment's code.
Delta reference generation does not start training.

For Python integrations, import model implementations from ``vpml.models``,
rollouts from ``vpml.solvers`` and reusable training functions from their
owning ``vpml.training`` modules. The trainer entry points expose their command
interfaces; they are not aggregate import locations for shared helpers.

Resolution diagnostics
----------------------

.. code-block:: bash

   bash model/diagnostics/run_landau_resolution_diagnostics.sh out_bench/resolution
   python -m vpml.diagnostics.physical_velocity_grid_convergence --help
   python -m vpml.diagnostics.projection_quadrature_convergence --help

Use convergence comparisons alongside conservation and positivity diagnostics.
A finite trajectory or a conserved integral alone does not establish accuracy.

Plot saved energy trajectories
------------------------------

``scripts/plot_wavenumber_energy_comparison.py`` reads saved arrays without
training or evaluating a model:

.. code-block:: bash

   python scripts/plot_wavenumber_energy_comparison.py \
     --root /path/to/saved/run --output /path/to/new/energy.png

This renderer expects a 21-case held-out manifest and evaluation directories
for updates 2960, 4736 and 5920. Each array contains ``times``,
``teacher_energy`` and ``model_energy``. It checks agreement of the saved
reference trajectories and writes a provenance sidecar. Use a new output path;
existing figures are not overwritten. ``--style families`` and ``--footer``
select the optional labels. This is a renderer for that saved format, not a
generic dataset evaluator.
