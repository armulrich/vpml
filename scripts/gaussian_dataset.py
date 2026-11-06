#!/usr/bin/env python3
"""Compatibility entry point; install VPML and use ``vpml-dataset``."""
from pathlib import Path

from vpml.cli import gaussian_dataset
from vpml.cli.gaussian_dataset import main, read_parameters

if __name__ == "__main__":
    expected = Path(__file__).resolve().parents[1] / "src/vpml"
    actual = Path(gaussian_dataset.__file__).resolve().parents[1]
    if actual != expected:
        raise RuntimeError("Install this checkout with 'python -m pip install -e .' before using its launcher")
    raise SystemExit(main())
