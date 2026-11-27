"""Explicit source and runtime identities for reproducible reference data.

No credentials, environment-variable dumps, or remote service calls are needed.
Source inventories identify the code that produced each array bundle.
Restart compatibility is defined separately by the solver and storage contract.
"""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import platform
import subprocess


from vpml.io.serialization import file_digest, json_digest


def source_record(package: Path | None = None) -> dict:
    """Record package sources and the environment lock without importing them."""
    package = Path(package) if package is not None else Path(__file__).resolve().parents[1]
    files = {str(Path("vpml") / path.relative_to(package)): file_digest(path)
             for path in sorted(package.rglob("*.py"))}
    repository = next((parent for parent in package.parents
                       if (parent / "pyproject.toml").is_file()), None)
    if repository is not None:
        relative = "environments/delta/requirements.lock"
        path = repository / relative
        if path.is_file():
            files[relative] = file_digest(path)
    return {"sha256": json_digest(files), "files": files}


def runtime_record(required_backend: str) -> dict:
    """Check actual JAX placement and precision before any trajectory work."""
    if required_backend not in {"cpu", "gpu"}:
        raise ValueError("An explicit cpu or gpu backend is required")
    import jax
    if jax.default_backend() != required_backend:
        raise RuntimeError(f"Required {required_backend}; JAX selected {jax.default_backend()}")
    if not jax.config.x64_enabled:
        raise RuntimeError("Reference evolution requires JAX float64 support")
    names = ("jax", "jaxlib", "numpy", "scipy", "jax-cuda12-plugin", "jax-cuda13-plugin",
             "jax-cuda12-pjrt", "jax-cuda13-pjrt", "nvidia-cudnn-cu12", "nvidia-cudnn")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name in {"jax", "jaxlib", "numpy", "scipy"}:
                raise
    driver_versions = []
    if required_backend == "gpu":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        driver_versions = sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip()))
        if not driver_versions:
            raise RuntimeError("Cannot record the GPU driver version")
    return {"python": platform.python_version(), "packages": versions,
            "backend": required_backend, "compute_dtype": "float64",
            "devices": [device.device_kind for device in jax.devices()],
            "driver_versions": driver_versions}
