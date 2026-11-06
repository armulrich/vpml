#!/usr/bin/env python3
"""Compatibility entry point; install VPML and use ``vpml-dataset``."""
from vpml.cli.gaussian_dataset import main, read_parameters

if __name__ == "__main__":
    raise SystemExit(main())
