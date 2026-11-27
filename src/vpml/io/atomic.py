"""Publish flushed files on one filesystem, then sync their directory."""
from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
from typing import BinaryIO
import uuid


def sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: str | Path, write: Callable[[BinaryIO], object], *,
                 replace: bool = True) -> None:
    """Publish one file; ``replace=False`` refuses any existing destination.

    The parent directory must exist. The callback writes bytes to an exclusive
    temporary file beside the destination. A failed write leaves the previously
    published destination intact. Concurrent updates still require caller locks.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
