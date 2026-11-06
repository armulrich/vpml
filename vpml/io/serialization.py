"""Stable JSON identities and streaming file checksums.

Canonical identities exclude a trailing newline. Storage formats that include
one must request it explicitly; changing these bytes changes existing hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False)
    return (payload + ("\n" if newline else "")).encode("utf-8")


def json_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_digest(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _invalid_constant(value: str):
    raise ValueError(f"Nonfinite JSON value: {value}")


def read_json(path: str | Path):
    """Read JSON, rejecting duplicate keys and nonfinite numeric values."""
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=_unique_object,
                       parse_constant=_invalid_constant)
    # JSON exponent overflow (1e999) also produces inf in Python's parser.
    canonical_json_bytes(value)
    return value
