# Delta environment

See the [Delta environment instructions](../../docs/delta.rst#environment)
for setup and CUDA requirements.

`requirements.in` lists the reviewed direct dependencies. `requirements.lock`
pins exact versions of all direct and transitive dependencies for Linux
x86_64/Python 3.12, without distribution hashes.
`setup.sh` creates a new environment only when explicitly invoked and never
submits a job. Lock regeneration is described in the
[development guide](../../docs/development.rst#update-the-delta-lock).
