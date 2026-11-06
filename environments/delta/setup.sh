#!/usr/bin/env bash
# Explicit environment installation only. This script never submits a job.
set -euo pipefail
umask 077

if [[ $# -ne 2 ]]; then
    printf 'Usage: bash %s /absolute/python3.12 /absolute/new-venv\n' "$0" >&2
    exit 2
fi
python=$1
destination=$2
if [[ $python != /* || ! -x $python || $destination != /* || $destination == / ]]; then
    printf 'Provide an absolute executable Python path and an absolute new environment path.\n' >&2
    exit 2
fi
if [[ -e $destination || -L $destination ]]; then
    printf 'Refusing to replace an existing environment: %s\n' "$destination" >&2
    exit 2
fi
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    printf 'This environment lock is for Linux x86_64 only.\n' >&2
    exit 2
fi
"$python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Python 3.12 is required")'
directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
"$python" -m venv "$destination"
"$destination/bin/python" -m pip --isolated install --require-hashes --only-binary=:all: \
    --index-url https://pypi.org/simple -r "$directory/requirements.lock"
code=$(cd -- "$directory/../.." && pwd -P)
"$destination/bin/python" -m pip --isolated install --no-deps --no-build-isolation --editable "$code"
"$destination/bin/python" -m pip check
"$destination/bin/python" --version
printf 'Environment installed at %s. No GPU verification or job submission was performed.\n' "$destination"
