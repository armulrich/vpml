"""Explicit, NumPy-only comparisons of two Gaussian solver variants.

These comparisons measure numerical sensitivity. Passing the engineering
thresholds does not establish convergence, positivity, or training suitability.
The caller supplies verified histories and the kinetic state at the final
diagnostic time; this module does not read checkpoints or infer variant pairs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from vpml.datasets.initial_conditions import fields_from_coefficients

FIELD_THRESHOLDS = {
    "E": 0.01,
    "n_perturbation": 0.01,
    "u": 0.01,
    "p_perturbation": 0.01,
    "Q": 0.05,
    "dQ_dx": 0.05,
}
_FIELD_LABELS = ("n_perturbation", "u", "p_perturbation", "Q", "dQ_dx")


def _plain(value):
    """Convert metadata to finite JSON values without discarding any keys."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metadata_signature(value) -> str:
    try:
        return json.dumps(_plain(value), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Comparison metadata must contain finite JSON values") from exc


def _array(value, name: str, shape=None) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind != "f" or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} has shape {result.shape}; expected {shape}")
    return np.asarray(result, dtype=np.float64)


def _validate_variant(variant: dict) -> dict:
    name = variant["variant"]
    if not isinstance(name, str) or not name:
        raise ValueError("variant must be a nonempty name")
    status = variant["status"]
    if status not in ("completed", "partial"):
        raise ValueError("variant status must be completed or partial")
    numerics, arrays = variant["numerics"], variant["arrays"]
    for key in ("nx", "nv", "output_nx"):
        value = numerics[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError(f"{key} must be an integer of at least two")
    if numerics["output_nx"] > numerics["nx"]:
        raise ValueError("output_nx cannot exceed nx")
    for key in ("dt", "final_time", "save_dt"):
        if not np.isfinite(numerics[key]) or numerics[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if (not np.isfinite([numerics["vmin"], numerics["vmax"]]).all()
            or numerics["vmin"] >= numerics["vmax"]):
        raise ValueError("Velocity endpoints must be finite and increasing")
    if not np.isfinite(variant["case"]["k0"]) or variant["case"]["k0"] <= 0:
        raise ValueError("k0 must be positive and finite")

    times = _array(arrays["times"], "times")
    diagnostic_times = _array(arrays["diagnostic_times"], "diagnostic_times")
    for values, label, interval in (
        (times, "times", numerics["save_dt"]),
        (diagnostic_times, "diagnostic_times", numerics["dt"]),
    ):
        if values.ndim != 1 or len(values) < 2 or not np.all(np.diff(values) > 0):
            raise ValueError(f"{label} must contain at least two increasing times")
        if not np.allclose(values, np.arange(len(values)) * interval, rtol=0, atol=1e-10):
            raise ValueError(f"{label} must start at zero and follow its numerical interval")
    if (times[-1] > diagnostic_times[-1] + 1e-10
            or diagnostic_times[-1] > numerics["final_time"] + 1e-10):
        raise ValueError("Saved times must lie within the evolved interval")
    if status == "completed" and not np.allclose(
        [times[-1], diagnostic_times[-1]], numerics["final_time"], rtol=0, atol=1e-10,
    ):
        raise ValueError("A completed variant must reach its declared final time")
    # Do not conflate interpolation of saved observables with timestep error.
    steps = times / numerics["dt"]
    if not np.allclose(steps, np.round(steps), rtol=0, atol=1e-8):
        raise ValueError("Refinement samples must coincide with integration-step times")
    output_nx = numerics["output_nx"]
    fields = _array(arrays["fields"], "fields", (len(times), 5, output_nx))
    electric = _array(arrays["electric"], "electric", (len(times), output_nx))
    _array(arrays["diagnostics"], "diagnostics", (len(diagnostic_times), 8))
    f = _array(variant["state"]["f"], "state.f", (numerics["nv"], numerics["nx"]))
    report = variant["report"]
    if report.get("training_qualified") is not False:
        raise ValueError("Diagnostic comparisons require training_qualified=False")
    if (not isinstance(report.get("checks_passed"), bool)
            or not isinstance(report.get("failures"), list)
            or any(not isinstance(item, str) for item in report["failures"])):
        raise ValueError("The diagnostic report must record checks_passed and failures")
    if report["checks_passed"] != (not report["failures"]):
        raise ValueError("Diagnostic checks_passed disagrees with failures")
    return {
        "times": times, "diagnostic_times": diagnostic_times,
        "fields": fields, "electric": electric, "f": f,
    }


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """A stable spatial norm for real nodal values or complex coefficients."""
    magnitude = np.abs(values)
    scales = np.max(magnitude, axis=1)
    scaled = np.divide(magnitude, scales[:, None], out=np.zeros_like(magnitude),
                       where=scales[:, None] > 0)
    return scales * np.sqrt(np.sum(weights[None, :] * scaled**2, axis=1))


def _time_norm(rms: np.ndarray, times: np.ndarray) -> float:
    scale = float(np.max(rms))
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.trapezoid((rms / scale)**2, x=times)))


def _error_metrics(base: np.ndarray, other: np.ndarray, times: np.ndarray,
                   weights: np.ndarray | None = None) -> dict:
    if weights is None:
        weights = np.full(base.shape[1], 1.0 / base.shape[1])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        error_rms = _weighted_rms(base - other, weights)
        reference_rms = _weighted_rms(other, weights)
        absolute = _time_norm(error_rms, times)
        reference = _time_norm(reference_rms, times)
    return {
        "relative_L2": absolute / reference if reference > 0 else (0.0 if absolute == 0 else None),
        "absolute_L2": absolute,
        "rms_error": absolute / float(np.sqrt(times[-1] - times[0])),
        "max_time_rms_error": float(np.max(error_rms)),
        "reference_L2": reference,
        "zero_reference": reference == 0,
    }


def _initial_fields_check(base: dict, other: dict) -> dict:
    """Check grid-dependent normalization of the same analytic coefficients."""
    nx_a, nx_b = base["numerics"]["nx"], other["numerics"]["nx"]
    if nx_b % nx_a:
        raise ValueError("Initial-field comparison requires nested periodic spatial nodes")
    stride = nx_b // nx_a
    case = base["case"]
    a = fields_from_coefficients(case["coefficients"], nx_a, case["strength"], base["amplitudes"])
    b = fields_from_coefficients(case["coefficients"], nx_b, case["strength"], other["amplitudes"])
    errors = {}
    for name, x, y in zip(("n", "u", "temperature"), a, b, strict=True):
        shared = y[::stride]
        errors[name] = float(np.max(np.abs(x - shared)))
        if not np.allclose(x, shared, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Initial {name} differs at shared nodes after grid-dependent normalization")
    return {"passed": True, "maximum_absolute_errors": errors, "rtol": 1e-12,
            "atol": 1e-12, "spatial_stride": stride}


def _final_distribution(base: dict, other: dict, a: dict, b: dict) -> dict:
    time_a, time_b = (float(item["diagnostic_times"][-1]) for item in (a, b))
    if not np.isclose(time_a, time_b, rtol=0, atol=1e-10):
        return {"status": "not_compared", "reason": "Kinetic states have different times",
                "base_time": time_a, "other_time": time_b}
    na, nb = base["numerics"], other["numerics"]
    if nb["nx"] % na["nx"]:
        raise ValueError("Final-state comparison requires nested periodic spatial nodes")
    stride = nb["nx"] // na["nx"]
    length = 2 * np.pi / base["case"]["k0"]
    xa = np.arange(na["nx"]) * length / na["nx"]
    xb = np.arange(nb["nx"]) * length / nb["nx"]
    if not np.allclose(xb[::stride], xa, rtol=0, atol=1e-12):
        raise ValueError("Final-state spatial coordinates do not coincide")
    va = np.linspace(na["vmin"], na["vmax"], na["nv"])
    vb = np.linspace(nb["vmin"], nb["vmax"], nb["nv"])
    if va[0] < vb[0] - 1e-12 or va[-1] > vb[-1] + 1e-12:
        raise ValueError("Other velocity domain must cover the baseline domain")
    coordinates = (va - vb[0]) / (vb[1] - vb[0])
    indices = np.rint(coordinates).astype(np.int64)
    nested = (np.all((indices >= 0) & (indices < len(vb)))
              and np.allclose(vb[indices.clip(0, len(vb) - 1)], va, rtol=0, atol=1e-11))
    source = b["f"][:, ::stride]
    if nested:
        reference = source[indices]
        sampling = "Coincident velocity nodes; no interpolation"
    else:
        reference = np.column_stack([np.interp(va, vb, source[:, column])
                                     for column in range(source.shape[1])])
        sampling = "Linear interpolation using actual velocity coordinates"

    def norm(values):
        rms = _weighted_rms(values, np.full(na["nx"], 1 / na["nx"]))
        return _time_norm(rms, va)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        absolute = norm(a["f"] - reference)
        reference_norm = norm(reference)
    return {
        "status": "compared", "time": time_a,
        "relative_L2": absolute / reference_norm if reference_norm > 0 else (0.0 if absolute == 0 else None),
        "absolute_L2": absolute, "reference_L2": reference_norm,
        "zero_reference": reference_norm == 0,
        "maximum_absolute_error": float(np.max(np.abs(a["f"] - reference))),
        "velocity_sampling": sampling, "spatial_stride": stride,
        "norm_convention": "Spatial mean and trapezoid velocity integral on baseline coordinates",
    }


def compare_variants(base: dict, other: dict) -> dict:
    """Compare explicit variants of exactly the same initial condition.

    Each input contains ``variant``, ``case``, ``amplitudes``, ``numerics``,
    ``arrays``, ``state``, ``report`` and ``status``. Arrays are ``times``,
    ``fields`` (time,5,output_nx), ``electric`` (time,output_nx),
    ``diagnostic_times`` and ``diagnostics`` (all steps,8). ``state['f']``
    belongs to the final diagnostic time. Output samples must be actual
    integration times, share the same output grid and match throughout their
    common interval. Partial histories are compared only over that interval.

    Relative norms use ``other`` as the reference. A nonzero difference from a
    zero reference has an undefined relative error, represented by JSON null.
    """
    if _metadata_signature(base["case"]) != _metadata_signature(other["case"]):
        raise ValueError("Variants must have the same complete initial-condition case")
    if _metadata_signature(base["amplitudes"]) != _metadata_signature(other["amplitudes"]):
        raise ValueError("Variants must have the same initial-condition amplitudes")
    a, b = _validate_variant(base), _validate_variant(other)
    if base["variant"] == other["variant"]:
        raise ValueError("Comparison requires distinct variant names")
    if base["numerics"]["output_nx"] != other["numerics"]["output_nx"]:
        raise ValueError("Variants must use the same output_nx")
    initial_check = _initial_fields_check(base, other)
    stop = min(float(a["times"][-1]), float(b["times"][-1]))
    ia, ib = a["times"] <= stop + 1e-10, b["times"] <= stop + 1e-10
    times, other_times = a["times"][ia], b["times"][ib]
    if (len(times) < 2 or times.shape != other_times.shape
            or not np.allclose(times, other_times, rtol=0, atol=1e-10)):
        raise ValueError("Variants must have matching saved times over their common interval")

    fields = {"E": (a["electric"][ia], b["electric"][ib])}
    for index, label in enumerate(_FIELD_LABELS):
        background = 1.0 if index in (0, 2) else 0.0
        fields[label] = (a["fields"][ia, index] - background,
                         b["fields"][ib, index] - background)
    errors = {label: _error_metrics(x, y, times) for label, (x, y) in fields.items()}
    nx = base["numerics"]["output_nx"]
    bands = {}
    for highest_mode in (4, 63):
        if highest_mode >= nx / 2:
            continue
        weights = np.full(highest_mode + 1, 2.0)
        weights[0] = 1.0
        bands[f"modes_0_to_{highest_mode}"] = {
            "maximum_positive_mode": highest_mode,
            "physical_wavenumber_maximum": highest_mode * base["case"]["k0"],
            "field_errors": {
                label: _error_metrics(np.fft.rfft(x, axis=1)[:, :highest_mode + 1] / nx,
                                      np.fft.rfft(y, axis=1)[:, :highest_mode + 1] / nx,
                                      times, weights)
                for label, (x, y) in fields.items()
            },
        }
    failed = [label for label, threshold in FIELD_THRESHOLDS.items()
              if errors[label]["relative_L2"] is None or errors[label]["relative_L2"] > threshold]
    na, nb = base["numerics"], other["numerics"]
    changed = {key: {"base": na.get(key), "other": nb.get(key)}
               for key in sorted(set(na) | set(nb)) if na.get(key) != nb.get(key)}
    result = {
        "base_variant": base["variant"], "other_variant": other["variant"],
        "changed_numerics": changed,
        "initial_fields_shared_nodes": initial_check,
        "common_time_interval": [float(times[0]), float(times[-1])],
        "common_sample_count": len(times), "field_errors": errors,
        "spectral_bands": bands,
        "final_distribution": _final_distribution(base, other, a, b),
        "engineering_checks": {
            "thresholds": dict(FIELD_THRESHOLDS), "failed_fields": failed,
            "passed": not failed,
            "scope": "Field sensitivity over the common interval; engineering thresholds, not proof of convergence",
        },
        "diagnostic_status": {
            variant["variant"]: {
                "status": variant["status"],
                "state_time": float(values["diagnostic_times"][-1]),
                "checks_passed": variant["report"]["checks_passed"],
                "failures": list(variant["report"]["failures"]),
                "training_qualified": False,
            }
            for variant, values in ((base, a), (other, b))
        },
        "training_qualified": False,
        "norm_convention": "Time integral of spatial mean squared error; other variant is the reference. Density and pressure use unit-background perturbations. Absolute_L2 is the square root of that integral; rms_error divides by square root of duration.",
        "spectral_convention": "Fourier coefficients normalized by output_nx; mode zero retained, positive modes weighted by two; Nyquist excluded",
    }
    # Fail rather than emitting nonfinite numbers into a scientific report.
    result = _plain(result)
    json.dumps(result, allow_nan=False)
    return result
