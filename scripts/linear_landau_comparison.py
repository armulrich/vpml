"""Single-mode linearized kinetic Landau comparators for the Gaussian pilot.

Each independent periodic domain has one complex positive spatial mode h(v,t).
The real perturbation is 2 Re[h exp(i k x)]. The linearized equation is
    dh/dt = -i k v h + E_hat M'(v),  E_hat = i integral(h dv) / k.
Exact free streaming and the exact linear force subflow are Strang split.
These are linearized comparators, not nonlinear kinetic training references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solve_modes(wavenumbers, nv=2048, dt=.02, final_time=40., save_dt=.1, epsilon=.001):
    """Solve independent single-mode problems; return retained raw moments."""
    k = np.asarray(wavenumbers, dtype=float)
    if k.ndim != 1 or len(k) == 0 or not np.all(k > 0):
        raise ValueError("Positive one-dimensional wavenumber list required")
    if nv < 4 or dt <= 0 or save_dt <= 0 or final_time < 0:
        raise ValueError("Invalid numerical parameters")
    inner = round(save_dt / dt)
    saves = round(final_time / save_dt)
    if inner < 1 or not np.isclose(inner * dt, save_dt) or not np.isclose(saves * save_dt, final_time):
        raise ValueError("Saved intervals must contain whole timesteps")
    v = np.linspace(-8., 8., nv)
    dv = v[1] - v[0]
    w = np.full(nv, dv); w[[0, -1]] *= .5
    maxwellian = np.exp(-.5 * v**2) / np.sqrt(2 * np.pi)
    maxwellian /= np.sum(w * maxwellian)
    derivative = -v * maxwellian
    weighted_powers = w[:, None] * v[:, None] ** np.arange(4)[None, :]
    equilibrium_moments = maxwellian @ weighted_powers
    h = np.broadcast_to(.5 * epsilon * maxwellian, (len(k), nv)).astype(np.complex128).copy()
    half_drift = np.exp(-.5j * k[:, None] * v[None, :] * dt)
    mode_moments = [h @ weighted_powers]
    for _ in range(saves):
        for _ in range(inner):
            h *= half_drift
            electric = 1j * (h @ w) / k
            h += dt * electric[:, None] * derivative[None, :]
            h *= half_drift
        mode_moments.append(h @ weighted_powers)
    return dict(times=np.arange(saves + 1) * save_dt, k=k, v=v,
                equilibrium=maxwellian, equilibrium_moments=equilibrium_moments,
                mode_moments=np.asarray(mode_moments), final_mode=h,
                dt=dt, epsilon=epsilon, save_dt=save_dt)


def fields(solution, index, nx=256):
    """Apply the same moment definitions to M + 2 Re[h exp(i theta)]."""
    phase = np.exp(2j * np.pi * np.arange(nx) / nx)
    raw = (solution["equilibrium_moments"][None, :, None]
           + 2 * np.real(solution["mode_moments"][:, index, :, None] * phase[None, None, :]))
    n, momentum, second, third = np.moveaxis(raw, 1, 0)
    u = momentum / n
    p = second - momentum**2 / n
    Q = third - 3 * u * second + 2 * n * u**3
    k = solution["k"][index]
    harmonic = np.arange(nx // 2 + 1) * k
    g = np.fft.irfft(1j * harmonic * np.fft.rfft(Q, axis=1), n=nx, axis=1)
    ehat = 1j * solution["mode_moments"][:, index, 0] / k
    E = 2 * np.real(ehat[:, None] * phase[None, :])
    return dict(n=n, u=u, p=p, Q=Q, g=g, E=E)


def relative_error(actual, reference):
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(actual - reference) / denominator) if denominator else float(np.linalg.norm(actual - reference))


def refinement_validation(base):
    """Separate timestep and velocity-grid refinements on every wavenumber."""
    k = base["k"]
    dt_refined = solve_modes(k, len(base["v"]), base["dt"] / 2,
                             base["times"][-1], base["save_dt"], base["epsilon"])
    v_refined = solve_modes(k, 2 * len(base["v"]), base["dt"],
                            base["times"][-1], base["save_dt"], base["epsilon"])
    records = []
    for name, refined in (("halved_dt", dt_refined), ("doubled_nv", v_refined)):
        for index, wavenumber in enumerate(k):
            a, b = fields(base, index), fields(refined, index)
            errors = {"field_relative_L2": relative_error(a["E"], b["E"])}
            for q in ("n", "u", "p", "Q", "g"):
                # Do not let the unit equilibrium conceal perturbation errors.
                offset = 1. if q in ("n", "p") else 0.
                errors[q + "_perturbation_relative_L2"] = relative_error(a[q] - offset, b[q] - offset)
            records.append(dict(refinement=name, k0=float(wavenumber), **errors))
    tolerance = 1e-3
    passed = all(value < tolerance for record in records for key, value in record.items()
                 if key.endswith("relative_L2"))
    return dict(passed=passed, relative_L2_tolerance=tolerance, records=records,
                interpretation="Numerical agreement for the linearized comparator only; not a nonlinear reference qualification.")


def spline_jvp_validation():
    """Calibrate against the maintained kinetic map's tangent on a small grid."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from scripts.gaussian_dataset_pilot import Runner
    import jax.numpy as jnp
    runner = Runner(nx=64, nv=512, dt=.02, k0=.43)
    f = runner.equilibrium
    h = .001 * f * jnp.cos(.43 * runner.cfg.x)[None, :]
    reference_times = [0.]
    reference_fields = [np.asarray(runner.diagnostics(f, h)[2])]
    reference_moments = [np.asarray(runner.moments(f + h))]
    for block in range(2):
        # Full f remains the homogeneous Maxwellian. h follows the exact JVP
        # of the spline timestep, with its own discretization of M'(v).
        (f, h), (_, _, _, ehist) = runner.advance(f, h)
        reference_fields.extend(np.asarray(ehist))
        # Retain the final moment sample in each one-unit block for comparison.
        reference_moments.append(np.asarray(runner.moments(f + h)))
        reference_times.extend(np.arange(1, 11) * .1 + block)
    spectral = solve_modes([.43], nv=512, dt=.02, final_time=2.)
    actual = fields(spectral, 0, nx=64)
    field_error = relative_error(actual["E"], np.asarray(reference_fields))
    moments = np.stack([actual[q][[0, 10, 20]] for q in ("n", "u", "p", "Q", "g")], axis=1)
    reference = np.asarray(reference_moments)
    errors = {}
    for j, q in enumerate(("n", "u", "p", "Q", "g")):
        offset = 1. if q in ("n", "p") else 0.
        errors[q + "_perturbation_relative_L2"] = relative_error(moments[:, j] - offset, reference[:, j] - offset)
    tolerance = 1e-3
    passed = field_error < tolerance and all(error < tolerance for error in errors.values())
    return dict(passed=passed, field_relative_L2=field_error, **errors,
                relative_L2_tolerance=tolerance, nx=64, nv=512, dt=.02, T=2., k0=.43,
                interpretation="Short comparison with maintained cubic-spline kinetic JVP; spectral drift and analytic M' need not reproduce its interpolation error exactly.")


def save_cases(out, solution, validation):
    destination = out / "cases"
    destination.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(__file__)
    nx = 256
    qualified = validation["refinement"]["passed"] and validation["spline_jvp"]["passed"]
    for index, k in enumerate(solution["k"]):
        identity = f"landau_{index + 1:02d}_base"
        dest = destination / f"{identity}.npz"
        meta_path = dest.with_suffix(".json")
        arrays = fields(solution, index, nx)
        L = 2 * np.pi / k
        energy = .5 * L / nx * np.sum(arrays["E"]**2, axis=1)
        grid = dict(nx=nx, nv=len(solution["v"]), dt=solution["dt"], L=float(L),
                    k0=float(k), vmin=-8., vmax=8., final_time=float(solution["times"][-1]),
                    save_dt=solution["save_dt"])
        if meta_path.exists():
            previous = json.loads(meta_path.read_text())
            if (previous.get("source_sha256") != source_hash or previous["grid"] != grid
                    or previous["result_sha256"] != sha256(dest)):
                raise ValueError(f"Existing comparator differs: {identity}")
            continue
        temporary = dest.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, times=solution["times"], **arrays,
                            full_energy=energy, linear_energy=energy, linear_field=arrays["E"],
                            x=np.arange(nx) * L / nx, v=solution["v"])
        temporary.replace(dest)
        report = dict(case_id=identity, family="linear_landau", seed=index + 1,
                      strength=.001, k0=float(k), grid=grid, eta_NL=None,
                      solver="single positive Fourier mode; exact drift and linear force kick; Strang split",
                      equation="dh/dt=-ikvh+E_hat*M_prime; E_hat=i*integral(h dv)/k",
                      reference_kind="Linearized kinetic comparator, not nonlinear training reference",
                      moment_convention="Exact nonlinear moment transform of the linearized distribution; same n,u,p,Q definitions as Gaussian cases",
                      max_relative_mass_error=float(np.max(np.abs(arrays["n"].mean(axis=1) - 1))),
                      max_relative_total_energy_error=None, min_full_distribution=None,
                      max_negative_mass_fraction=None, max_boundary_mass_fraction=None,
                      energy_note="The full nonlinear physical-energy gate is inapplicable to linearized evolution; quadratic perturbation terms are omitted from its dynamics.",
                      qualification_pass=qualified,
                      qualification_failures=[] if qualified else ["linearized_numerical_validation"],
                      qualification_scope="Linearized comparator numerical agreement only",
                      source_sha256=source_hash, result_sha256=sha256(dest),
                      compute_dtype="complex128/float64", numpy_version=np.__version__)
        meta_path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    solution = solve_modes(np.linspace(.30, .60, 15))
    validation = dict(refinement=refinement_validation(solution), spline_jvp=spline_jvp_validation(),
                      source_sha256=sha256(__file__))
    validation["elapsed_seconds"] = time.monotonic() - started
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "linear_landau_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if not validation["refinement"]["passed"] or not validation["spline_jvp"]["passed"]:
        print(json.dumps(validation, indent=2), flush=True)
        raise RuntimeError("Linearized comparison failed numerical validation")
    save_cases(args.out, solution, validation)
    print(json.dumps(dict(cases=15, linearized_only=True,
                          elapsed_seconds=time.monotonic() - started,
                          refinement_passed=True, spline_jvp_passed=True,
                          validation=str(args.out / "linear_landau_validation.json")), indent=2), flush=True)


if __name__ == "__main__":
    main()
