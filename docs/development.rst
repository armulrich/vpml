Development
===========

Code and outputs
----------------

Numerical algorithms live in ``src/vpml``. The installed package contains
solvers, models, training, datasets, diagnostics and evaluation. Import reusable
functions from their owning modules, such as ``vpml.solvers.hermite``,
``vpml.models.interface_checkpoint`` and ``vpml.training.low_moment_data``.
Training entry points contain workflow orchestration; shared helpers belong in
the corresponding configuration, data, loss, state or optimization module.
The package root retains its lazy public API without loading a numerical
backend for metadata-only commands.

The retained files under ``model/train``, ``model/diagnostics``, ``benchmarks``
and ``scripts`` implement experiment controllers, launch procedures and saved-run
analysis. Their shell commands call installed ``vpml`` modules. Preserve a
frozen experiment's recorded source and configuration when continuing it.
Dataset modules separate initial conditions, manifests, observations,
evolution, diagnostics, checkpoints and execution. CLI and scheduler code call
these interfaces without implementing a second numerical solver.

Keep source code, tests, input parameter files and dependency locks in version
control. Generated manifests, provenance, progress, logs, metrics, checkpoints
and figures belong with the corresponding run output, such as the ignored
``out_bench`` directory or an external project data directory. Documentation
explains how to use and interpret the code; it does not accumulate run diaries.

Maintaining restart compatibility
---------------------------------

The dataset restart contract describes the saved-state format and numerical
semantics supported by the implementation. Source revisions and full file
hashes are recorded as provenance; an import refactor or documentation change
does not by itself change that contract. The ordinary reader recognizes the
supported existing protocol format without rewriting its saved metadata.

Change ``solver_revision`` when an implementation change alters the evolution,
observation definitions or interpretation of saved state. Change
``protocol_version`` or ``checkpoint_version`` when the corresponding stored
representation changes. Review compatibility
explicitly and provide a reader for a supported older format, or reject it.
Keeping an identifier unchanged is a developer assertion of compatibility,
not something a source hash or version label can prove.

Restart tests must cover uninterrupted versus interrupted evolution, continued
loading of supported older checkpoints, preservation of original records and
rejection of incompatible scientific parameters, runtime or state. Use an
archived implementation when verifying continuation across a refactor. A tiny
CPU comparison checks that continuation path; it does not establish grid
convergence or bitwise agreement across GPU environments.

Checks
------

Use CPU tests for deterministic initialization, analytical moments, sampling,
restart equivalence and failure handling:

.. code-block:: bash

   JAX_PLATFORMS=cpu VPML_JAX_BACKEND=cpu \
     python -m unittest discover -s tests -p 'test_dataset_*.py'
   python -m unittest discover -s tests -p 'test_delta_launcher.py'
   shellcheck -x scripts/delta/*.sh environments/delta/setup.sh

Launcher tests use a fake scheduler. They do not reserve GPUs. For the full suite, including function-based tests, install ``.[test]`` and use
``python -m pytest``. The test extra includes the Modal SDK only to import
legacy controller definitions; those tests do not submit remote work. Passing software
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
Linux Python 3.12 dependency set into one exact version pin per package:

.. code-block:: bash

   uv --no-config --no-progress --no-python-downloads \
     --cache-dir /tmp/vpml-delta-uv-cache pip compile \
     environments/delta/requirements.in \
     --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
     --only-binary :all: \
     --default-index https://pypi.org/simple --no-header --no-annotate \
     --output-file environments/delta/requirements.lock

The lock includes direct and transitive dependencies without distribution
hashes. Regenerate it only for a deliberate dependency update. Review the
resolved versions and install into a new environment. A successful resolution
establishes neither GPU execution nor numerical equivalence; check both before
using changed dependencies for reference data.
