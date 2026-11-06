"""Explicit source and runtime identities for reproducible reference data.

No credentials, environment-variable dumps, or remote service calls are needed.
The compatibility fingerprint follows numerical dependencies; the broader
inventory also records unrelated package files without constraining restart.
"""
from __future__ import annotations

import ast
import importlib.metadata
from pathlib import Path
import platform
import subprocess


from vpml.io.serialization import file_digest, json_digest


def _imported_package_modules(module: str, path: Path, modules: dict[str, Path]) -> set[str]:
    """Resolve static local imports without importing the numerical modules."""
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    result = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                components = package.split(".")
                base = ".".join(components[:len(components) - node.level + 1])
                base += ("." + node.module) if node.module else ""
            else:
                base = node.module or ""
            candidates = [base, *(base + "." + alias.name for alias in node.names)]
        else:
            continue
        for candidate in candidates:
            if candidate in modules:
                result.add(candidate)
    return result


def source_record(package: Path | None = None) -> dict:
    """Record all package sources and a separate reference-compatibility digest.

    Compatibility follows the static VPML import dependencies of the dataset
    runner, including package initializers. Runtime-selected numerical modules
    must be made explicit imports so their code participates in this identity.
    An unrelated plotting edit is recorded but does not prevent continuation.
    """
    package = Path(package) if package is not None else Path(__file__).resolve().parents[1]
    modules = {}
    paths = sorted(package.rglob("*.py"))
    for path in paths:
        parts = list(path.relative_to(package).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(["vpml", *parts])] = path
    files = {str(Path("vpml") / path.relative_to(package)): file_digest(path) for path in paths}
    pending = ["vpml.datasets.runner"]
    dependencies = set()
    while pending:
        module = pending.pop()
        if module in dependencies:
            continue
        if module not in modules:
            raise ValueError(f"Missing numerical source module: {module}")
        dependencies.add(module)
        pending.extend(_imported_package_modules(module, modules[module], modules) - dependencies)
        # Importing a child executes each package initializer too.
        parts = module.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts))
                       if ".".join(parts[:i]) in modules)
    numerical_files = {str(Path("vpml") / modules[name].relative_to(package)):
                       files[str(Path("vpml") / modules[name].relative_to(package))]
                       for name in sorted(dependencies)}
    # Checkout inputs are useful provenance but not numerical compatibility.
    repository = next((parent for parent in package.parents if (parent / "pyproject.toml").is_file()), None)
    if repository is not None:
        for relative in ("scripts/gaussian_dataset.py", "environments/delta/requirements.lock"):
            path = repository / relative
            if path.is_file():
                files[relative] = file_digest(path)
    return {"sha256": json_digest(files), "files": files,
            "compatibility": {"format": 2, "sha256": json_digest(numerical_files),
                              "files": numerical_files}}


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
