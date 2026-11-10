Gaussian reference datasets
===========================

Initial conditions
------------------

Each case starts from a local Gaussian,

.. math::

   f_0(x,v)=\frac{n_0(x)}{\sqrt{2\pi T_0(x)}}
   \exp\!\left[-\frac{(v-u_0(x))^2}{2T_0(x)}\right].

Independent sine and cosine coefficients define three smooth shapes
:math:`g_n,g_u,g_T`. The coefficient of harmonic :math:`j` is weighted by
:math:`j^{-s}`, and each shape is normalized to unit spatial RMS. With
:math:`\langle\cdot\rangle` denoting the spatial mean, the fields are

.. math::

   n_0=\frac{\exp(\lambda a_n g_n)}{\langle\exp(\lambda a_n g_n)\rangle},
   \qquad
   T_0=\frac{\exp(\lambda a_T g_T)}{\langle\exp(\lambda a_T g_T)\rangle},

.. math::

   u_0=\lambda a_u\left(g_u-
   \frac{\langle n_0g_u\rangle}{\langle n_0\rangle}\right).

Density and temperature are positive with mean one, and
:math:`\langle n_0u_0\rangle=0`. The Gaussian is evaluated on the finite velocity
interval without renormalizing away its truncated tails. Diagnostics measure
the resulting charge imbalance and boundary population.

Parameters and preparation
--------------------------

The parameters in ``scripts/delta/gaussian_amplitudes.json``
define the wavenumbers, mode counts, decay exponents, amplitudes, strengths,
numerical grid and data split. They produce 240 shapes and five strengths per
shape, giving 1,200 cases. All strengths of one shape remain in the same split.
The split contains 960 training, 120 validation and 120 test cases; these labels
do not imply that numerical references are already qualified for training.

Preparation records actual coefficients and validated identities in a manifest:

.. code-block:: bash

   vpml-dataset prepare \
     --parameters scripts/delta/gaussian_amplitudes.json \
     --manifest out_bench/gaussian/manifest.json
   vpml-dataset validate \
     --parameters scripts/delta/gaussian_amplitudes.json \
     --manifest out_bench/gaussian/manifest.json

These commands perform no trajectory evolution. Reusing a path with different
parameters is rejected. Keep the manifest and source fixed for all workers
that contribute to a dataset. :doc:`delta` describes execution on allocated
compute resources; the underlying Python runner also supports CPU execution.
``python -m vpml.cli.gaussian_dataset`` provides the same command interface.

Pass ``--case-id`` repeatedly to ``validate`` and ``run`` to select cases from
the full manifest without changing its contents. Validation rejects duplicate
or unknown IDs before a worker starts. ``run --stop-on-failure`` ends the
invocation after the first recorded or newly detected numerical rejection,
retaining its checkpoint and failure report. Without this option, the runner
continues to other selected cases.

Saved quantities
----------------

The ``gaussian-fields-v1`` schema uses the solver's nondimensional units.
Velocity integrals use trapezoid quadrature:

.. math::

   M_j=\int v^j f\,dv,\qquad n=M_0,\qquad u=M_1/M_0,
   \qquad p=\int(v-u)^2f\,dv,\qquad Q=\int(v-u)^3f\,dv.

There is no factor :math:`1/2` in :math:`p` or :math:`Q`, and raw moments have
no equilibrium subtraction.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Array
     - Shape and meaning
   * - ``fields``
     - ``(time, 5, position)``: :math:`n,u,p,Q,\partial_xQ`
   * - ``raw_moments``
     - ``(time, 3, position)``: :math:`M_0,M_1,M_2`
   * - ``electric``
     - ``(time, position)``: electric field
   * - ``times``
     - Saved output times, including the initial and final time
   * - ``diagnostics``, ``diagnostic_times``
     - Float64 diagnostic values at every integration step

Primitive and central moments are evaluated on the kinetic grid before Fourier
restriction. Raw moments are restricted independently. Filtering raw moments
and then converting to primitive fields generally gives a different quantity.
Spatial derivatives use physical coordinates; Fourier restriction handles the
target Nyquist mode explicitly.

Output times between integration steps use linear interpolation of observed
fields. In the supplied configuration, :math:`\Delta t=0.01` and the output
spacing is :math:`0.025`. Output precision is distinct from float64 integration.

Read completed cases
--------------------

Read one case at a time when inspecting a large dataset:

.. code-block:: python

   from vpml.datasets.reader import iter_cases

   for case in iter_cases("/path/to/dataset/reference"):
       density = case["n"]       # (saved time, position)
       report = case["report"]
       print(case["case_id"], density.shape, report["failures"])

The reader verifies completed output checksums and schema without loading
kinetic restart states. It skips pending and partial cases; corrupted completed
outputs raise an error. A completed case may still have failed numerical
checks, so inspect its report before using it for training. ``read_case`` loads
a single named case. The separate adapter for historical flat NPZ files keeps
their original metadata; it does not supply missing qualification evidence.

Avoid collecting all cases into a list merely to compute a running statistic.
The older analysis script materializes trajectories and is intended for small
comparisons, not the full 1,200-case dataset.

Continuation
------------

Each case records an immutable protocol, restart states, observation blocks
and progress. Progress is published only after the referenced data are durable.
Recovery verifies hashes, shapes, dtypes and the protocol before using a state.
Complete cases are verified and skipped; partial cases resume at the next
integration step. For example, a state at :math:`t=37` with
:math:`\Delta t=0.01` resumes at step 3,701. The original initial observation
remains the conservation baseline.

Resume compatibility includes the numerical code dependencies, manifest and
runtime. The complete source inventory is stored separately, and each saved
array bundle records the inventory used to produce it. Changes to unrelated
plotting code are recorded without invalidating a numerical checkpoint.
Older protocols with a different compatibility definition are not silently
upgraded; use their recorded code revision for continuation.

An exclusive case lock prevents two workers from writing the same case.
Investigate the recorded owner before removing a stale lock; workers never
steal locks. Uncommitted files are excluded from recovery. Nonfinite blocks do
not replace the last valid checkpoint, and failed cases are not retried or
replaced by new random draws automatically.

Numerical checks
----------------

Reports measure mass and total energy, negative distribution values, negative
mass, boundary population, density, charge imbalance and saved-field positivity.
Finite distribution undershoots and their negative mass are recorded without
automatically rejecting a trajectory. Nonfinite values, nonpositive density or
saved pressure, and failures of the remaining configured checks still reject it.
The tolerances are engineering thresholds, not a physical admissibility theorem.
Failed distributions are not repaired by clipping.

Completion and training qualification are separate. A passing report retains
``training_qualified = false`` until grid, timestep and velocity-domain
comparisons establish adequate accuracy for the intended use. The optional
linearized evolution follows the tangent map about a homogeneous Maxwellian;
it is disabled in the supplied dataset configuration.

Compare retained fields
-----------------------

Use completed cases from two datasets with identical initial conditions,
velocity interval and output coordinates. Only ``nx``, ``nv`` and ``dt`` may
differ between their frozen manifests:

.. code-block:: bash

   python -m vpml.diagnostics.dataset_accuracy \
     --candidate /path/to/candidate/reference \
     --reference /path/to/refined/reference \
     --case-id CASE_ID --tolerance 0.005 --out comparison.json

The comparison verifies checksums and measures :math:`n-1,u,p-1,Q,\partial_xQ,E`
at the actual saved times, including storage rounding. Relative L2 errors use
the time integral of spatial mean square, separately over the full horizon and
each third of it. Absolute errors and the largest per-time relative errors are
also reported. Zero reference norms receive no artificial denominator floor;
a nonzero difference against a zero reference is inconclusive.

A passing comparison establishes agreement at the chosen tolerance, not the
accuracy of either reference. Diagnostic failures are reported separately.
Reference convergence, velocity-domain adequacy and coverage of the intended
initial-condition family remain necessary before choosing production settings.
