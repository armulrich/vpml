# Delta launchers

See the [Delta usage guide](../../docs/delta.rst) for environment setup,
preparation, validation, submission, monitoring and continuation.

`VPML_RUN_PARAMS` selects a versioned parameter file; v1 is the default.
The adjacent JSON supplies scientific parameters. The v2 parameter file
selects five cases from the full 2048-point dataset and stops on the first
numerical rejection. Both versions share the same launcher and batch script.
The launcher requires an explicit action; only `submit` requests a real allocation. Array definitions and
numerical checks are documented in the [dataset guide](../../docs/datasets.rst).
