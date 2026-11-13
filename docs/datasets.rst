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
interval without renormalizing away its truncated tails.

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
parameters is rejected. Keep the manifest and scientific parameters fixed for
all workers that contribute to a dataset, and keep each worker's checkout
fixed while it runs. :doc:`delta` describes execution on allocated
compute resources; the underlying Python runner also supports CPU execution.
``python -m vpml.cli.gaussian_dataset`` provides the same command interface.

Pass ``--case-id`` repeatedly to ``validate`` and ``run`` to select cases from
the full manifest without changing its contents. Validation rejects duplicate
or unknown IDs before a worker starts. ``run --stop-on-failure`` ends the
invocation after the first recorded or newly detected invalid output or integrity failure,
retaining its checkpoint and failure report. Without this option, the runner
continues to other selected cases.

Saved quantities
----------------

The ``gaussian-moments`` schema uses the solver's nondimensional units.
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
   * - ``linear_electric`` (optional)
     - ``(time, position)``: electric field of the tangent evolution

Primitive and central moments are evaluated on the kinetic grid before Fourier
restriction. Raw moments are restricted independently. Filtering raw moments
and then converting to primitive fields generally gives a different quantity.
Spatial derivatives use physical coordinates; Fourier restriction handles the
target Nyquist mode explicitly.

Output times between integration steps use linear interpolation of observed
fields. Integration and retained observations use float64; ``storage_dtype``
controls the precision of the final fields, raw moments and electric field.
Float32 rounding near a unit background can erase small :math:`n-1` and
:math:`p-1` perturbations. Use float64 output when these perturbations must be
retained, and subtract the background before any lower-precision conversion
for analysis or training. Float64 storage does not establish grid or timestep
convergence.

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
checks in a historical dataset; the reader preserves that report. ``read_case``
loads a single named case. The historical ``gaussian-fields-v1`` and flat NPZ
adapters preserve their original metadata. Reading either format does not
establish numerical accuracy.

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
:math:`\Delta t=0.0125` resumes at step 2,961. The original initial observation
is retained.

Resume compatibility uses a maintained restart contract: the supported saved
format and numerical semantics, the manifest, scientific parameters and
runtime must agree. Output precision, checkpoint cadence and the optional
linearized evolution are part of that contract. Changing ``storage_dtype`` or
``block_steps`` requires a separate dataset; comparison between datasets does
not permit resuming with changed settings.

The complete source inventory and its hashes are provenance. They identify
which code produced each saved array bundle without requiring byte-identical
source for a compatible restart. New protocols record ``protocol_version=1``,
``checkpoint_version=1`` and ``solver_revision="vp1d1v-strang-cubic-1"``.
The ordinary reader also recognizes the supported existing protocol layout
that predates those fields and contains a historical format-2 source record.
It interprets that state under the corresponding maintained contract and
preserves the original file bytes. Unknown formats or numerical contracts are
rejected.

Continue with the existing manifest and reference directory. Validation reads
metadata; execution uses resources already allocated for the chosen backend:

.. code-block:: bash

   python -m vpml.cli.gaussian_dataset validate \
     --manifest /dataset/manifest.json --out /dataset/reference
   python -m vpml.cli.gaussian_dataset run \
     --manifest /dataset/manifest.json --out /dataset/reference \
     --backend gpu --max-seconds 3300 --stop-on-failure

For a dataset with linearized evolution, pass ``--linearized`` to both commands.
The worker checks its actual runtime against the saved runtime before evolving.
Compatible code revisions use these same commands. Original case protocols,
observation blocks and provenance remain immutable; newly written arrays
identify the code that actually wrote them. Completed cases are verified and
skipped, and partial cases resume from their last committed state. There is no
separate migration record to prepare. :doc:`development` describes the
developer responsibility for maintaining the contract and restart tests.

An exclusive case lock prevents two workers from writing the same case.
Investigate the recorded owner before removing a stale lock; workers never
steal locks. Uncommitted files are excluded from recovery. Nonfinite blocks do
not replace the last valid checkpoint, and failed cases are not retried or
replaced by new random draws automatically.

Output validity and analysis
----------------------------

Generation checks finite state and output values, positive density and
pressure, and storage integrity. It stops on invalid output and records the
failure. It neither clips the distribution nor applies conservation,
boundary-population or refinement tolerances. Extra diagnostic reductions and
numerical-quality warnings are absent from the integration loop.

The completion report records these validity checks and its scope. Accuracy
assessment is separate from completion. Saved raw moments and electric fields
support conservation analysis without rerunning the kinetic evolution:

.. math::

   M(t)=\Delta x\sum_x M_0(x,t),\qquad
   H(t)=\frac{\Delta x}{2}\sum_x\left[M_2(x,t)+E(x,t)^2\right].

Here :math:`\Delta x` is the spacing of the saved spatial grid. This energy
uses retained electric-field modes; discarded-mode energy is unavailable,
so it is not the full kinetic-grid energy diagnostic. Full-distribution
minima and velocity-boundary populations are not reconstructed from moments;
they require a retained kinetic state. Historical output keeps its original
diagnostic arrays and reports. The optional linearized evolution follows the
tangent map about a homogeneous Maxwellian and saves its electric field; it
is disabled in the supplied dataset configuration.

Compare retained fields
-----------------------

Use completed cases from two datasets with identical initial conditions,
velocity interval and output coordinates. Only ``nx``, ``nv``, ``dt``,
``block_steps`` and ``storage_dtype`` may differ between their frozen manifests.
The report lists these differences:

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

To compare the evolution before output rounding, add
``--representation observations-float64`` to the same command. This mode
verifies the retained float64 observation bundles and their provenance, then
samples them at the same output times using the original interpolation rule.
It uses existing observations without rerunning a trajectory or modifying its
files. A pass in this mode does not establish that lower-precision saved
outputs retain the same accuracy; the default ``saved`` mode includes their
rounding error.

A passing comparison establishes agreement at the chosen tolerance, not the
accuracy of either reference. Historical diagnostic failures are reported
separately. The comparison is an analysis tool, not a dataset-generation gate;
its tolerance should reflect the quantities and use being assessed.
