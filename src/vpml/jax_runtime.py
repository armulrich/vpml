"""Pre-import JAX runtime selection helpers for vpml."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from typing import Mapping, Optional

_VALID_BACKENDS = frozenset({"auto", "cpu", "gpu"})
_METAL_DISABLED_REASON = (
    "vpml disables the experimental Apple Metal backend by default because this codebase "
    "requires float64 and complex dtypes, while Apple's jax-metal docs list np.float64, "
    "np.complex64, and np.complex128 as unsupported."
)

_BOOTSTRAP_STATE: Optional["JaxRuntimePlan"] = None
_EMITTED_CONTEXTS: set[str] = set()


@dataclass(frozen=True)
class JaxRuntimePlan:
    """Backend-selection plan applied before importing JAX."""

    requested_backend: str
    jax_platforms: Optional[str]
    env_override: bool
    reason: Optional[str]
    metal_disabled: bool


def plan_jax_runtime(
    env: Optional[Mapping[str, str]] = None,
    *,
    system: Optional[str] = None,
) -> JaxRuntimePlan:
    """Return the backend-selection plan without mutating the environment."""
    env_map = os.environ if env is None else env
    requested = env_map.get("VPML_JAX_BACKEND", "auto").strip().lower()
    if requested not in _VALID_BACKENDS:
        choices = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"VPML_JAX_BACKEND must be one of {{{choices}}}; got {requested!r}")

    if "JAX_PLATFORMS" in env_map or "JAX_PLATFORM_NAME" in env_map:
        return JaxRuntimePlan(
            requested_backend=requested,
            jax_platforms=None,
            env_override=True,
            reason="Respecting explicit JAX platform override from the environment.",
            metal_disabled=False,
        )

    system_name = platform.system() if system is None else system
    if requested == "cpu":
        return JaxRuntimePlan(
            requested_backend=requested,
            jax_platforms="cpu",
            env_override=False,
            reason="VPML_JAX_BACKEND=cpu",
            metal_disabled=False,
        )
    if requested == "gpu":
        if system_name == "Darwin":
            return JaxRuntimePlan(
                requested_backend=requested,
                jax_platforms="cpu",
                env_override=False,
                reason=_METAL_DISABLED_REASON,
                metal_disabled=True,
            )
        return JaxRuntimePlan(
            requested_backend=requested,
            jax_platforms=None,
            env_override=False,
            reason=(
                "VPML_JAX_BACKEND=gpu leaves JAX on automatic platform selection. JAX will "
                "default to GPU if a supported accelerator backend is installed and available."
            ),
            metal_disabled=False,
        )
    if system_name == "Darwin":
        return JaxRuntimePlan(
            requested_backend=requested,
            jax_platforms="cpu",
            env_override=False,
            reason=_METAL_DISABLED_REASON,
            metal_disabled=True,
        )
    return JaxRuntimePlan(
        requested_backend=requested,
        jax_platforms=None,
        env_override=False,
        reason=(
            "Leaving JAX platform selection on automatic. JAX defaults to GPU or TPU if "
            "available and falls back to CPU otherwise."
        ),
        metal_disabled=False,
    )


def bootstrap_jax_runtime() -> JaxRuntimePlan:
    """Apply the pre-import runtime plan exactly once."""
    global _BOOTSTRAP_STATE
    if _BOOTSTRAP_STATE is not None:
        return _BOOTSTRAP_STATE

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    plan = plan_jax_runtime()
    if plan.jax_platforms is not None and not plan.env_override:
        os.environ.setdefault("JAX_PLATFORMS", plan.jax_platforms)
    _BOOTSTRAP_STATE = plan
    return plan


def _has_nvidia_gpu_hint() -> bool:
    return bool(
        shutil.which("nvidia-smi")
        or os.path.exists("/dev/nvidiactl")
        or os.path.exists("/proc/driver/nvidia/version")
    )


def format_jax_runtime_summary(jax_module, *, context: Optional[str] = None) -> str:
    """Format a concise backend summary after JAX has been imported."""
    plan = _BOOTSTRAP_STATE or bootstrap_jax_runtime()
    label = "JAX runtime" if context is None else f"JAX runtime ({context})"
    devices = ", ".join(str(device) for device in jax_module.devices())
    message = f"[vpml] {label}: backend={jax_module.default_backend()} devices=[{devices}]"
    if plan.metal_disabled and plan.reason:
        message = f"{message}; Metal disabled: {plan.reason}"
    elif plan.reason and plan.requested_backend != "auto":
        message = f"{message}; {plan.reason}"
    if (
        jax_module.default_backend() == "cpu"
        and not plan.metal_disabled
        and plan.requested_backend != "cpu"
        and _has_nvidia_gpu_hint()
    ):
        message = (
            f"{message}; NVIDIA GPU detected but JAX is running on CPU. Install a CUDA-enabled "
            'JAX build such as `pip install -U "jax[cuda13]"`.'
        )
    return message


def print_jax_runtime_summary(jax_module, *, context: Optional[str] = None) -> None:
    """Print the runtime summary once per named context."""
    key = context or "__default__"
    if key in _EMITTED_CONTEXTS:
        return
    print(format_jax_runtime_summary(jax_module, context=context))
    _EMITTED_CONTEXTS.add(key)
