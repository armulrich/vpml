"""Numerical checks, distinct from completion and refinement qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from vpml._jax_precision import enable_x64

enable_x64()

import jax.numpy as jnp
import numpy as np

from .schema import DIAGNOSTIC_NAMES, LINEAR_DIAGNOSTIC_NAMES

@dataclass(frozen=True)
class DiagnosticTolerances:
    """Engineering thresholds, not a theorem of physical admissibility."""

    relative_mass: float = 1e-9
    relative_energy: float = 1e-4
    minimum_f: float = -1e-8
    negative_mass_fraction: float = 1e-8
    boundary_abs_fraction: float = 1e-8
    charge_imbalance: float = 1e-8

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "minimum_f":
                if value > 0:
                    raise ValueError("minimum_f must be nonpositive")
            elif value < 0:
                raise ValueError(f"{name} must be nonnegative")


def diagnostic_vector(distribution, velocity_grid, dx: float, electric, *, linear_electric=None):
    """Evaluate diagnostics on the kinetic grid with trapezoid velocity weights.

    The boundary statistic uses the outer 0.5 velocity units on either side.
    ``charge_imbalance`` is mean numerical density minus the unit ion background;
    it exposes finite-velocity truncation instead of silently renormalizing ``f``.
    """
    f = jnp.asarray(distribution, dtype=jnp.float64)
    v = jnp.asarray(velocity_grid, dtype=jnp.float64)
    e = jnp.asarray(electric, dtype=jnp.float64)
    density = jnp.trapezoid(f, x=v, axis=0)
    mass = dx * jnp.sum(density)
    electric_energy = 0.5 * dx * jnp.sum(e ** 2)
    kinetic_energy = 0.5 * dx * jnp.sum(jnp.trapezoid(v[:, None] ** 2 * f, x=v, axis=0))
    negative_mass = dx * jnp.sum(jnp.trapezoid(jnp.maximum(-f, 0), x=v, axis=0))
    boundary = (v <= v[0] + 0.5) | (v >= v[-1] - 0.5)
    boundary_mass = dx * jnp.sum(jnp.trapezoid(boundary[:, None] * jnp.abs(f), x=v, axis=0))
    entries = [
        mass, kinetic_energy + electric_energy, electric_energy, jnp.min(f),
        negative_mass / jnp.abs(mass), boundary_mass / jnp.abs(mass),
        jnp.min(density), jnp.mean(density) - 1,
    ]
    if linear_electric is not None:
        linear_e = jnp.asarray(linear_electric, dtype=jnp.float64)
        entries.extend((0.5 * dx * jnp.sum(linear_e ** 2), dx * jnp.sum((e - linear_e) ** 2)))
    return jnp.stack(entries)


def qualification_report(
    history,
    *,
    names=DIAGNOSTIC_NAMES,
    tolerances: DiagnosticTolerances = DiagnosticTolerances(),
) -> dict:
    """Summarize a diagnostic history whose first row is the original IC.

    Nonfinite input is reported with JSON ``null`` metrics, never NaN/Infinity.
    Passing these checks does not establish grid or velocity-domain convergence.
    """
    data = np.asarray(history, dtype=np.float64)
    names = tuple(names)
    if data.ndim != 2 or not len(data) or data.shape[1] != len(names):
        raise ValueError("diagnostic history must have shape (nonzero samples, len(names))")
    if len(set(names)) != len(names) or not set(DIAGNOSTIC_NAMES).issubset(names):
        raise ValueError("diagnostic names must include each required column exactly once")
    columns = {name: data[:, index] for index, name in enumerate(names)}
    failures = []
    if not np.isfinite(data).all():
        failures.append("nonfinite")

    def finite_extreme(values, operation):
        if not np.isfinite(values).all():
            return None
        value = float(operation(values))
        return value if math.isfinite(value) else None

    def relative_change(values):
        if not np.isfinite(values).all() or values[0] <= 0:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            return finite_extreme(np.abs(values / values[0] - 1), np.max)

    metrics = {
        "max_relative_mass_error": relative_change(columns["mass"]),
        "max_relative_total_energy_error": relative_change(columns["total_energy"]),
        "min_f": finite_extreme(columns["min_f"], np.min),
        "max_negative_mass_fraction": finite_extreme(columns["negative_mass_fraction"], np.max),
        "max_boundary_abs_fraction": finite_extreme(columns["boundary_abs_fraction"], np.max),
        "min_n": finite_extreme(columns["min_n"], np.min),
        "max_abs_charge_imbalance": finite_extreme(np.abs(columns["charge_imbalance"]), np.max),
    }
    limits = (
        ("relative_mass", "max_relative_mass_error", tolerances.relative_mass),
        ("relative_energy", "max_relative_total_energy_error", tolerances.relative_energy),
        ("negative_mass_fraction", "max_negative_mass_fraction", tolerances.negative_mass_fraction),
        ("boundary_abs_fraction", "max_boundary_abs_fraction", tolerances.boundary_abs_fraction),
        ("charge_imbalance", "max_abs_charge_imbalance", tolerances.charge_imbalance),
    )
    for name, metric, limit in limits:
        if metrics[metric] is None or metrics[metric] > limit:
            failures.append(name)
    if metrics["min_f"] is None or metrics["min_f"] < tolerances.minimum_f:
        failures.append("minimum_f")
    if metrics["min_n"] is None or metrics["min_n"] <= 0:
        failures.append("nonpositive_density")
    return {
        "checks_passed": not failures,
        "failures": failures,
        "training_qualified": False,
        "qualification_scope": "diagnostics only; refinement required",
        "metrics": metrics,
        "tolerances": asdict(tolerances),
    }
