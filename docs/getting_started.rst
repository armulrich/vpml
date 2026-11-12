Installation and first use
==========================

Local installation
------------------

Use Python 3.12 for the documented scientific environment. From the repository
root, create an environment and install the package:

.. code-block:: bash

   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .

The package depends on JAX, NumPy, SciPy and Matplotlib. The editable install
uses the source in the checkout. Keep that source fixed for each dataset
worker. Continuing an existing dataset after a reviewed source
refactor requires the explicit transition described in :doc:`datasets`.

For Delta, use the pinned Linux CUDA environment described in :doc:`delta`
instead of assuming that a local CPU environment will work on the cluster.

Backend and precision
---------------------

Select the backend before importing JAX:

.. code-block:: bash

   export VPML_JAX_BACKEND=cpu
   export JAX_PLATFORMS=cpu

macOS uses CPU because the solver requires float64 and complex arithmetic.
The reference generator computes in float64; saved observation precision is
specified separately by its manifest. A GPU worker verifies that JAX actually
selected a GPU and rejects a CPU fallback.

Inspect an initial condition
----------------------------

This example constructs fields without evolving a trajectory:

.. code-block:: python

   from vpml.datasets.initial_conditions import (
       draw_coefficients, fields_from_coefficients,
   )

   coefficients = draw_coefficients(seed=28001, modes=3, decay=2.0)
   n, u, temperature = fields_from_coefficients(
       coefficients, nx=128, strength=0.5,
   )

The fields have mean density and temperature one and density-weighted mean
velocity zero. Their spatial coordinate is :math:`x/L`; the physical domain
length is :math:`L=2\pi/k_0`. See :doc:`datasets` for the distribution and
normalization conventions.

Run a benchmark
---------------

.. code-block:: bash

   python -m vpml.benchmarks.fh_benchmarks_2412_07073_jax \
     linear_landau --outdir out_bench/linear_landau

This command performs a numerical calculation and writes figures and metrics
under the chosen output directory. Command-specific options are available
through ``--help``.

Import numerical interfaces directly from their owning modules:

.. code-block:: python

   from vpml.solvers.hermite import FourierHermiteIMEX
   from vpml.solvers.physical_grid import PhysicalGridVlasovPoissonConfig
   from vpml.models.interface_checkpoint import load_learned_interface_closure_npz

Use installed ``python -m vpml...`` commands for package workflows. The
retained shell scripts and experiment controllers described in :doc:`workflows`
configure or coordinate those workflows.

The complete initial-field example is ``examples/gaussian_initial_fields.py``.
It constructs one distribution without evolving it or writing run outputs.
