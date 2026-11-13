# Delta launchers

See the [Delta usage guide](../../docs/delta.rst) for environment setup,
preparation, validation, submission, monitoring and continuation.

`run_gaussian_amplitudes.sh` defaults to the adjacent
`gaussian_amplitudes.params.sh`. `VPML_RUN_PARAMS` optionally selects another
absolute parameter-file path. The canonical `gaussian_amplitudes.json` supplies
scientific parameters for the full 1,200-case dataset with 512 spatial points,
4,096 velocity points and a timestep of 0.0125.
The canonical launcher parameters leave `VPML_CASE_IDS` empty to process the
full manifest within the time budget and stop on the first invalid output or
integrity failure. Generation retains finite-value and positive density/pressure
checks. Conservation and other accuracy measures are assessed separately from
saved output, without extra per-step diagnostic calculations or warning messages.

After setting absolute `VPML_DATA_ROOT` and `VPML_PYTHON` paths:

```bash
bash scripts/delta/run_gaussian_amplitudes.sh prepare
bash scripts/delta/run_gaussian_amplitudes.sh validate
bash scripts/delta/run_gaussian_amplitudes.sh status
bash scripts/delta/run_gaussian_amplitudes.sh test-submit
```

Only the explicit `submit` action requests a real allocation. The launcher
bounds every parameter-file choice to one A100, at most 16 CPUs, 60G memory,
one hour of allocation time and a 3,300-second worker budget. The default
shutdown margin is 180 seconds. An explicit parameter file can set
`VPML_SHUTDOWN_MARGIN_SECONDS` between 60 and 180; the worker budget must still
fit inside the allocation time minus that margin. Array definitions and output
validation are documented in the [dataset guide](../../docs/datasets.rst).

Before a real submission, the launcher writes a read-only, content-addressed
JSON record in `$VPML_DATA_ROOT/submissions/`. It preserves the manifest hash,
scientific and launcher parameter snapshots, selected case IDs, scheduler and
worker budgets, and source hashes. Git identity is recorded when available;
for an archived checkout, optional `VPML_SOURCE_COMMIT` supplies a validated
40-digit hexadecimal commit declaration alongside the actual source hashes.
The record is a submission intent, not evidence of scheduler acceptance or job
completion. Keep the staged code and manifest unchanged while a job can use them.
