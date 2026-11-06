"""Preserve the numerical modules' historical double-precision JAX policy."""

from .jax_runtime import bootstrap_jax_runtime


def enable_x64() -> None:
    """Select the configured backend before JAX import and enable 64-bit arrays.

    Fourier-Hermite solvers and interface models have always requested float64
    and complex128. Keep that policy when importing their canonical modules,
    including when the compatibility facade has not been imported first.
    Explicit float32 arrays in low-moment models retain their requested dtype.
    """
    bootstrap_jax_runtime()
    import jax

    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        # Preserve the legacy import behavior if runtime configuration is locked.
        pass
