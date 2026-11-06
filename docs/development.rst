Development
===========

Code and outputs
----------------

Numerical algorithms live in the ``vpml`` package. Dataset modules separate
initial conditions, manifests, observations, evolution, diagnostics,
checkpoints and execution. CLI and scheduler wrappers call these interfaces;
they do not implement a second numerical solver.

Keep source code, tests, input parameter files and dependency locks in version
control. Generated manifests, provenance, progress, logs, metrics, checkpoints
and figures belong with the corresponding run output, such as the ignored
``out_bench`` directory or an external project data directory. Documentation
explains how to use and interpret the code; it does not accumulate run diaries.

Checks
------

Use CPU tests for deterministic initialization, analytical moments, sampling,
restart equivalence and failure handling:

.. code-block:: bash

   JAX_PLATFORMS=cpu VPML_JAX_BACKEND=cpu \
     python -m unittest discover -s tests -p 'test_dataset_*.py'
   python -m unittest discover -s tests -p 'test_delta_launcher.py'
   shellcheck -x scripts/delta/*.sh environments/delta/setup.sh

Launcher tests use a fake scheduler. They do not reserve GPUs. Broader tests
are available with ``python -m unittest discover -s tests``. Passing software
tests does not replace numerical convergence checks for a chosen dataset.

Build documentation
-------------------

Install the documentation requirements into a development environment that
already contains the scientific package, then build HTML:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html``. Sphinx imports API signatures and
docstrings with the CPU backend selected. The build does not execute the
example commands or evolve trajectories. NumPy/Google-style docstrings are
rendered with Napoleon, and mathematical expressions use MathJax.

Keep a usage topic in one canonical page and link to it from local README
files. Use docstrings for function contracts and nearby comments for numerical
choices that are not apparent from the implementation.

Update the Delta lock
---------------------

After reviewing changes to ``environments/delta/requirements.in``, resolve the
Linux Python 3.12 dependency set with hashes:

.. code-block:: bash

   uv --no-config --no-progress --no-python-downloads \
     --cache-dir /tmp/vpml-delta-uv-cache pip compile \
     environments/delta/requirements.in \
     --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
     --generate-hashes --only-binary :all: \
     --default-index https://pypi.org/simple --no-header \
     --output-file environments/delta/requirements.lock

Review the resolved versions and install into a new environment. A successful
resolution establishes neither GPU execution nor numerical equivalence; check
both before using changed dependencies for reference data.
