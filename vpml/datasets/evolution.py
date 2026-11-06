"""Portable single-case evolution using the existing physical-grid solver."""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig,
    build_physical_grid_ops,
    compute_electric_field_from_distribution,
    gaussian_pdf,
    normalize_density_on_grid,
    semilagrangian_vlasov_poisson_step,
)
from .diagnostics import DIAGNOSTIC_NAMES, LINEAR_DIAGNOSTIC_NAMES, diagnostic_vector
from .initial_conditions import fields_from_coefficients, gaussian_distribution
from .manifest import Numerics
from .observations import moment_fields, restrict_fourier


@lru_cache(maxsize=16)
def _build_kernels(numerics: Numerics, k0: float, linearized: bool):
    """Share compiled kernels across ICs with the same numerical settings."""
    cfg = PhysicalGridVlasovPoissonConfig(
        Nx=numerics.nx, Nv=numerics.nv, dt=numerics.dt, T=numerics.final_time,
        Lx=2 * np.pi / k0, vmin=numerics.vmin, vmax=numerics.vmax,
    )
    ops = build_physical_grid_ops(cfg)
    ops["batched_v_prefilter"] = True
    # Normalize only the tangent reference, never the sampled Gaussian IC.
    equilibrium = jnp.broadcast_to(
        normalize_density_on_grid(gaussian_pdf(cfg.v, 0, 1), cfg.v)[:, None],
        (numerics.nv, numerics.nx),
    )

    def full_step(f):
        return semilagrangian_vlasov_poisson_step(cfg, f, ops=ops)[0]

    def observe(state):
        f = state["f"]
        fields, raw = moment_fields(f, cfg.v, cfg.Lx)
        electric = compute_electric_field_from_distribution(f, cfg, ops=ops)
        linear_electric = (
            compute_electric_field_from_distribution(state["h"], cfg, ops=ops)
            if linearized else None
        )
        return {
            "fields": restrict_fourier(fields, numerics.output_nx),
            "raw_moments": restrict_fourier(raw, numerics.output_nx),
            "electric": restrict_fourier(electric, numerics.output_nx),
            "diagnostics": diagnostic_vector(f, cfg.v, cfg.dx, electric, linear_electric=linear_electric),
        }

    @partial(jax.jit, static_argnames=("steps",))
    def advance(state, steps):
        def one_step(current, _):
            result = {"f": full_step(current["f"])}
            if linearized:
                result["h"] = jax.jvp(full_step, (equilibrium,), (current["h"],))[1]
            return result, observe(result)
        return jax.lax.scan(one_step, state, None, length=steps)

    return cfg, ops, equilibrium, jax.jit(observe), advance


class ReferenceEvolution:
    """JIT-compiled block evolution; filesystem and scheduler independent.

    The state maps ``f`` to a float64 ``(nv,nx)`` kinetic distribution. Optional
    ``h`` evolves the tangent map about the homogeneous normalized Maxwellian.
    Each block returns observations at every integration step, excluding its
    starting state; callers save the initial observation exactly once.
    """

    def __init__(self, case: dict, numerics: Numerics, amplitudes: dict, linearized: bool = False):
        if not jax.config.x64_enabled:
            raise RuntimeError("Gaussian reference evolution requires JAX float64")
        if not isinstance(linearized, bool):
            raise ValueError("linearized must be a boolean")
        if not np.isfinite(case["k0"]) or case["k0"] <= 0:
            raise ValueError("k0 must be finite and positive")
        self.numerics = numerics
        self.linearized = bool(linearized)
        self.diagnostic_names = DIAGNOSTIC_NAMES + (LINEAR_DIAGNOSTIC_NAMES if linearized else ())
        self.cfg, self.ops, self._equilibrium, self._observe, self._advance = _build_kernels(
            numerics, float(case["k0"]), self.linearized,
        )
        n, u, temperature = fields_from_coefficients(
            case["coefficients"], numerics.nx, case["strength"], amplitudes=amplitudes,
        )
        self._f0 = jnp.asarray(gaussian_distribution(n, u, temperature, np.asarray(self.cfg.v)), dtype=jnp.float64)

    def initial_state(self) -> dict:
        state = {"f": self._f0}
        if self.linearized:
            state["h"] = self._f0 - self._equilibrium
        return state

    @property
    def equilibrium(self):
        """Normalized homogeneous Maxwellian used by the optional tangent map."""
        return self._equilibrium

    def _validate_state(self, state):
        expected_keys = {"f", "h"} if self.linearized else {"f"}
        if set(state) != expected_keys:
            raise ValueError(f"state must contain exactly {sorted(expected_keys)}")
        for name, value in state.items():
            if value.shape != (self.numerics.nv, self.numerics.nx):
                raise ValueError(f"invalid state shape for {name}: {value.shape}")
            if value.dtype != np.dtype("float64"):
                raise ValueError(f"state {name} must be float64")

    def observe(self, state) -> dict:
        self._validate_state(state)
        return self._observe(state)

    def advance(self, state, steps: int):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        self._validate_state(state)
        return self._advance(state, steps=steps)
