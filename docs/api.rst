Python API
==========

These interfaces separate initial conditions and data definitions from
evolution, persistence and scheduling. Array axes and units are defined in
:doc:`datasets`.

Initial conditions and manifests
--------------------------------

.. automodule:: vpml.datasets.initial_conditions
   :members: draw_coefficients, fields_from_coefficients, gaussian_distribution

.. automodule:: vpml.datasets.manifest
   :members: Numerics, build_manifest, validate_manifest, write_manifest, read_manifest

Observations and diagnostics
----------------------------

.. automodule:: vpml.datasets.observations
   :members: moment_fields, raw_to_primitives, restrict_fourier, observation_schema

.. automodule:: vpml.datasets.diagnostics
   :members: DiagnosticTolerances, diagnostic_vector, qualification_report

Evolution and continuation
--------------------------

.. autoclass:: vpml.datasets.evolution.ReferenceEvolution
   :members: initial_state, observe, advance

.. autoclass:: vpml.datasets.checkpoints.CheckpointStore
   :members: initialize, append, load, histories, finalize

.. automodule:: vpml.datasets.runner
   :members: run_dataset, dataset_status

Reading completed data
----------------------

.. automodule:: vpml.datasets.reader
   :members: iter_cases, read_case

Diagnostic refinement comparisons
---------------------------------

.. automodule:: vpml.diagnostics.gaussian_refinement
   :members: validate_config, run_variant, read_variant

.. automodule:: vpml.diagnostics.refinement_metrics
   :members: compare_variants

Kinetic solver
--------------

.. automodule:: vpml.solvers.physical_grid
   :members: PhysicalGridVlasovPoissonConfig, semilagrangian_vlasov_poisson_step
