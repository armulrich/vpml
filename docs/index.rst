VPML
====

VPML provides numerical solvers and learned closures for the collisionless
Vlasov–Poisson system in one spatial and one velocity dimension. Its main
workflows are kinetic reference generation, Fourier–Hermite benchmarks and
training reduced models from saved reference trajectories.

The physical-grid solver uses cubic spline semi-Lagrangian advection. The
Fourier–Hermite solver represents velocity dependence in Hermite functions.
Both use periodic spatial domains. The low-moment models evolve density,
velocity and pressure with a learned closure.

Start with :doc:`getting_started`, choose a task in :doc:`workflows`, or use
:doc:`api` for the Python interfaces. :doc:`datasets` defines the Gaussian
reference data and :doc:`delta` explains how to generate it on Delta.

.. toctree::
   :maxdepth: 2

   getting_started
   workflows
   datasets
   delta
   api
   development
