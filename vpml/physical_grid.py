"""
Shared JAX physical-grid semi-Lagrangian 1D1V Vlasov-Poisson solvers and projection helpers.

This module provides:
  - a reusable physical (x, v) grid configuration
  - JAX-native cubic B-spline interpolation helpers
  - Strang-split semi-Lagrangian Vlasov-Poisson time stepping
  - Fourier-Hermite projection helpers for teacher-data generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

from .jax_runtime import bootstrap_jax_runtime

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .core import Array, learned_closure_global_indicators, tridiag_solve

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


@dataclass(frozen=True)
class PhysicalGridVlasovPoissonConfig:
    Nx: int
    Nv: int
    Lx: float
    vmin: float
    vmax: float
    dt: float
    T: float
    poisson_sign: float = +1.0
    snapshot_times: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if int(self.Nx) <= 1:
            raise ValueError("Nx must exceed 1")
        if int(self.Nv) <= 3:
            raise ValueError("Nv must exceed 3 for cubic interpolation")
        if float(self.Lx) <= 0.0:
            raise ValueError("Lx must be positive")
        if float(self.vmax) <= float(self.vmin):
            raise ValueError("vmax must exceed vmin")
        if float(self.dt) <= 0.0:
            raise ValueError("dt must be positive")
        if float(self.T) < 0.0:
            raise ValueError("T must be nonnegative")

    @property
    def dx(self) -> float:
        return float(self.Lx) / float(self.Nx)

    @property
    def dv(self) -> float:
        return float(self.vmax - self.vmin) / float(self.Nv - 1)

    @property
    def nsteps(self) -> int:
        return int(round(float(self.T) / float(self.dt)))

    @property
    def x(self) -> Array:
        return (jnp.arange(int(self.Nx), dtype=jnp.float64) * self.dx).astype(jnp.float64)

    @property
    def v(self) -> Array:
        return jnp.linspace(float(self.vmin), float(self.vmax), int(self.Nv), dtype=jnp.float64)

    @property
    def k_arr(self) -> Array:
        return (2.0 * math.pi * jnp.fft.rfftfreq(int(self.Nx), d=self.dx)).astype(jnp.float64)

    @property
    def Nk(self) -> int:
        return int(self.Nx) // 2 + 1


ExternalFieldFn = Callable[[float, Array, Array], Array]
HistoryProjectorFn = Callable[[Array], Array]


def _snapshot_indices(snapshot_times: Sequence[float], dt: float) -> np.ndarray:
    return np.asarray([int(round(float(t) / float(dt))) for t in snapshot_times], dtype=np.int32)


def gaussian_pdf(v: Array, mean: float, sigma: float) -> Array:
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    v = jnp.asarray(v, dtype=jnp.float64)
    return (
        jnp.exp(-0.5 * ((v - float(mean)) / sigma) ** 2)
        / (math.sqrt(2.0 * math.pi) * sigma)
    ).astype(jnp.float64)


def normalize_density_on_grid(profile: Array, v: Array) -> Array:
    profile = jnp.asarray(profile, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    mass = jnp.maximum(jnp.trapezoid(profile, x=v), 1e-30)
    return (profile / mass).astype(jnp.float64)


def gaussian_vhat(eta: Array, mean: float, sigma: float, amplitude: float = 1.0) -> Array:
    return float(amplitude) * jnp.exp(
        -1j * jnp.asarray(eta, dtype=jnp.float64) * float(mean)
        - 0.5 * (float(sigma) ** 2) * (jnp.asarray(eta, dtype=jnp.float64) ** 2)
    )


def hermite_coeffs_gaussian(N: int, mean: float, sigma: float) -> Array:
    if int(N) <= 0:
        return jnp.zeros((0,), dtype=jnp.float64)

    mean = float(mean)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    s2m1 = sigma * sigma - 1.0
    H = jnp.zeros((int(N),), dtype=jnp.float64).at[0].set(1.0)
    if int(N) > 1:
        H = H.at[1].set(mean)

    def body(i, H_):
        val = mean * H_[i] + (i.astype(H_.dtype) * s2m1) * H_[i - 1]
        return H_.at[i + 1].set(val)

    if int(N) > 2:
        H = jax.lax.fori_loop(1, int(N) - 1, body, H)

    n = jnp.arange(int(N), dtype=jnp.float64)
    return (H * jnp.exp(-0.5 * jsp.special.gammaln(n + 1.0))).astype(jnp.float64)


def equilibrium_coeffs_bump_on_tail(N: int, vbar1: float, vbar2: float, vth: float = 1.0) -> Array:
    if float(vth) <= 0.0:
        raise ValueError("vth must be positive")
    c1 = hermite_coeffs_gaussian(int(N), float(vbar1) / float(vth), 1.0 / float(vth))
    c2 = hermite_coeffs_gaussian(int(N), float(vbar2) / float(vth), 0.5 / float(vth))
    return (0.9 * c1 + 0.1 * c2).astype(jnp.float64)


def _physical_grid_ops(config: PhysicalGridVlasovPoissonConfig) -> Dict[str, Array]:
    Nx = int(config.Nx)
    Nv = int(config.Nv)
    x = config.x
    v = config.v
    k_arr = config.k_arr
    x_index = jnp.arange(Nx, dtype=jnp.float64)[None, :]

    # Periodic cubic B-spline prefilter in x is diagonal in Fourier space.
    m = jnp.arange(Nx, dtype=jnp.float64)
    periodic_den = ((4.0 + 2.0 * jnp.cos(2.0 * math.pi * m / float(Nx))) / 6.0).astype(jnp.float64)

    # Zero-extension cubic B-spline coefficients in v satisfy a tridiagonal relation.
    sub = jnp.full((Nv - 1,), 1.0, dtype=jnp.float64)
    diag = jnp.full((Nv,), 4.0, dtype=jnp.float64)
    sup = jnp.full((Nv - 1,), 1.0, dtype=jnp.float64)

    return {
        "x": x,
        "v": v,
        "k_arr": k_arr,
        "x_index": x_index,
        "periodic_den": periodic_den,
        "v_prefilter_sub": sub,
        "v_prefilter_diag": diag,
        "v_prefilter_sup": sup,
    }


def _bspline_weights(frac: Array) -> Tuple[Array, Array, Array, Array]:
    frac = jnp.asarray(frac, dtype=jnp.float64)
    w0 = ((1.0 - frac) ** 3) / 6.0
    w1 = (3.0 * frac ** 3 - 6.0 * frac ** 2 + 4.0) / 6.0
    w2 = (-3.0 * frac ** 3 + 3.0 * frac ** 2 + 3.0 * frac + 1.0) / 6.0
    w3 = (frac ** 3) / 6.0
    return w0, w1, w2, w3


def cubic_bspline_prefilter_periodic(values: Array, periodic_den: Array) -> Array:
    values = jnp.asarray(values, dtype=jnp.float64)
    periodic_den = jnp.asarray(periodic_den, dtype=jnp.float64)
    coeff_hat = jnp.fft.fft(values, axis=-1) / periodic_den[None, :]
    return jnp.fft.ifft(coeff_hat, axis=-1).real.astype(jnp.float64)


def cubic_bspline_prefilter_constant(
    values: Array,
    sub: Array,
    diag: Array,
    sup: Array,
) -> Array:
    values = jnp.asarray(values, dtype=jnp.float64)
    solve_col = lambda rhs: tridiag_solve(sub, diag, sup, 6.0 * rhs)
    return jax.vmap(solve_col, in_axes=1, out_axes=1)(values).astype(jnp.float64)


def cubic_bspline_interp_periodic(coeffs: Array, coords: Array) -> Array:
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
    coords = jnp.asarray(coords, dtype=jnp.float64)
    N = coeffs.shape[1]
    base = jnp.floor(coords).astype(jnp.int32)
    frac = coords - base.astype(jnp.float64)
    w0, w1, w2, w3 = _bspline_weights(frac)

    idx0 = jnp.mod(base - 1, N)
    idx1 = jnp.mod(base, N)
    idx2 = jnp.mod(base + 1, N)
    idx3 = jnp.mod(base + 2, N)

    return (
        jnp.take_along_axis(coeffs, idx0, axis=1) * w0
        + jnp.take_along_axis(coeffs, idx1, axis=1) * w1
        + jnp.take_along_axis(coeffs, idx2, axis=1) * w2
        + jnp.take_along_axis(coeffs, idx3, axis=1) * w3
    ).astype(jnp.float64)


def cubic_bspline_interp_constant(coeffs: Array, coords: Array, cval: float = 0.0) -> Array:
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
    coords = jnp.asarray(coords, dtype=jnp.float64)
    coeffs_t = coeffs.T
    coords_t = coords.T
    N = coeffs_t.shape[1]

    base = jnp.floor(coords_t).astype(jnp.int32)
    frac = coords_t - base.astype(jnp.float64)
    w0, w1, w2, w3 = _bspline_weights(frac)

    def gather(idx: Array, weight: Array) -> Array:
        valid = (idx >= 0) & (idx < N)
        idx_clip = jnp.clip(idx, 0, N - 1)
        vals = jnp.take_along_axis(coeffs_t, idx_clip, axis=1)
        return vals * weight * valid.astype(jnp.float64)

    interp_t = (
        gather(base - 1, w0)
        + gather(base, w1)
        + gather(base + 1, w2)
        + gather(base + 2, w3)
    )
    inside = (coords_t >= 0.0) & (coords_t <= float(N - 1))
    interp_t = jnp.where(inside, interp_t, float(cval))
    return interp_t.T.astype(jnp.float64)


def compute_electric_field_from_distribution(
    f_phys: Array,
    config: PhysicalGridVlasovPoissonConfig,
    *,
    ops: Optional[Dict[str, Array]] = None,
) -> Array:
    f_phys = jnp.asarray(f_phys, dtype=jnp.float64)
    v = config.v if ops is None else ops["v"]
    k_arr = config.k_arr if ops is None else ops["k_arr"]
    rho = jnp.trapezoid(f_phys, x=v, axis=0)
    rho_p = rho - jnp.mean(rho)
    rho_hat = jnp.fft.rfft(rho).astype(jnp.complex128)
    rho_p_hat = jnp.fft.rfft(rho_p).astype(jnp.complex128)
    del rho_hat
    E_hat = jnp.zeros_like(rho_p_hat, dtype=jnp.complex128)
    E_hat = E_hat.at[1:].set(float(config.poisson_sign) * 1j * rho_p_hat[1:] / k_arr[1:])
    return jnp.fft.irfft(E_hat, n=int(config.Nx)).astype(jnp.float64)


def electric_energy_from_field(E_phys: Array, config: PhysicalGridVlasovPoissonConfig) -> Array:
    E_phys = jnp.asarray(E_phys, dtype=jnp.float64)
    return (0.5 * float(config.dx) * jnp.sum(E_phys * E_phys)).astype(jnp.float64)


def advect_x_cubic(
    f_phys: Array,
    config: PhysicalGridVlasovPoissonConfig,
    ops: Dict[str, Array],
    tau: float,
) -> Array:
    coeffs = cubic_bspline_prefilter_periodic(f_phys, ops["periodic_den"])
    coords = jnp.mod(ops["x_index"] - ops["v"][:, None] * float(tau) / float(config.dx), float(config.Nx))
    return cubic_bspline_interp_periodic(coeffs, coords)


def advect_v_cubic(
    f_phys: Array,
    config: PhysicalGridVlasovPoissonConfig,
    ops: Dict[str, Array],
    E_phys: Array,
    tau: float,
) -> Array:
    coeffs = cubic_bspline_prefilter_constant(
        f_phys,
        ops["v_prefilter_sub"],
        ops["v_prefilter_diag"],
        ops["v_prefilter_sup"],
    )
    coords = (ops["v"][:, None] + jnp.asarray(E_phys, dtype=jnp.float64)[None, :] * float(tau) - float(config.vmin)) / float(config.dv)
    return cubic_bspline_interp_constant(coeffs, coords, cval=0.0)


def run_semilagrangian_vlasov_poisson(
    config: PhysicalGridVlasovPoissonConfig,
    f0: Array,
    *,
    external_field_fn: Optional[ExternalFieldFn] = None,
    history_stride: int = 1,
    return_state_history: bool = False,
    history_projector: Optional[HistoryProjectorFn] = None,
) -> Dict[str, np.ndarray | Array]:
    f0 = jnp.asarray(f0, dtype=jnp.float64)
    if f0.shape != (int(config.Nv), int(config.Nx)):
        raise ValueError(f"f0 must have shape ({config.Nv}, {config.Nx}), got {f0.shape}")

    ops = _physical_grid_ops(config)
    nsteps = int(config.nsteps)
    history_stride = max(int(history_stride), 1)
    snap_steps = _snapshot_indices(config.snapshot_times, config.dt)
    snaps0 = jnp.zeros((len(snap_steps), int(config.Nv), int(config.Nx)), dtype=jnp.float64)
    if 0 in snap_steps:
        snap0_idx = int(np.where(snap_steps == 0)[0][0])
        snaps0 = snaps0.at[snap0_idx].set(f0)

    history_data0 = None
    hist_steps = np.arange(0, nsteps + 1, history_stride, dtype=np.int32)
    if hist_steps[-1] != nsteps:
        hist_steps = np.concatenate([hist_steps, np.array([nsteps], dtype=np.int32)])
    history_value0 = None
    if return_state_history:
        history_value0 = f0 if history_projector is None else jnp.asarray(history_projector(f0))
        history_data0 = jnp.zeros((len(hist_steps),) + history_value0.shape, dtype=history_value0.dtype)
        history_data0 = history_data0.at[0].set(history_value0)

    E0 = compute_electric_field_from_distribution(f0, config, ops=ops)
    energy0 = electric_energy_from_field(E0, config)

    def default_external_field(t: Array, x: Array, k_arr: Array) -> Array:
        del t, x, k_arr
        return jnp.zeros((int(config.Nx),), dtype=jnp.float64)

    ext_fn = external_field_fn if external_field_fn is not None else default_external_field

    def maybe_store_snapshot(snaps: Array, step_i: Array, f_state: Array) -> Array:
        for j, snap_step in enumerate(snap_steps):
            snaps = jax.lax.cond(
                step_i == int(snap_step),
                lambda s, arr=f_state, idx=j: s.at[idx].set(arr),
                lambda s: s,
                snaps,
            )
        return snaps

    def maybe_store_history(history: Array, step_i: Array, f_state: Array) -> Array:
        if not return_state_history:
            return history
        projected = f_state if history_projector is None else jnp.asarray(history_projector(f_state))
        history = jax.lax.cond(
            (step_i % history_stride) == 0,
            lambda h, arr=projected: h.at[step_i // history_stride].set(arr),
            lambda h: h,
            history,
        )
        if int(hist_steps[-1]) != nsteps:
            history = jax.lax.cond(
                step_i == int(nsteps),
                lambda h, arr=projected: h.at[len(hist_steps) - 1].set(arr),
                lambda h: h,
                history,
            )
        return history

    def step(carry, step_i):
        f_state, snaps, history = carry
        t_mid = (step_i.astype(jnp.float64) - 0.5) * float(config.dt)
        f_half = advect_x_cubic(f_state, config, ops, 0.5 * float(config.dt))
        E_mid = compute_electric_field_from_distribution(f_half, config, ops=ops)
        H_mid = jnp.asarray(ext_fn(t_mid, ops["x"], ops["k_arr"]), dtype=jnp.float64)
        f_vel = advect_v_cubic(f_half, config, ops, E_mid + H_mid, float(config.dt))
        f_new = advect_x_cubic(f_vel, config, ops, 0.5 * float(config.dt))
        E_new = compute_electric_field_from_distribution(f_new, config, ops=ops)
        en = electric_energy_from_field(E_new, config)
        snaps = maybe_store_snapshot(snaps, step_i, f_new)
        history = maybe_store_history(history, step_i, f_new)
        return (f_new, snaps, history), en

    (f_last, snaps_out, hist_out), energy_hist = jax.lax.scan(
        step,
        (f0, snaps0, history_data0),
        jnp.arange(1, nsteps + 1, dtype=jnp.int32),
    )
    del f_last

    raw: Dict[str, np.ndarray | Array] = {
        "x": np.asarray(ops["x"]),
        "v": np.asarray(ops["v"]),
        "k_arr": np.asarray(ops["k_arr"]),
        "snapshot_times": np.asarray(config.snapshot_times, dtype=float),
        "snapshot_f": np.asarray(snaps_out),
        "times": np.linspace(0.0, nsteps * float(config.dt), nsteps + 1),
        "energy": np.asarray(jnp.concatenate([jnp.array([energy0], dtype=jnp.float64), energy_hist], axis=0)),
    }
    if return_state_history and hist_out is not None:
        raw["state_history"] = hist_out
        raw["state_history_times"] = hist_steps.astype(float) * float(config.dt)
    return raw


def hermite_dual_basis_scaled(N: int, v: Array, vth: float = 1.0) -> Array:
    if int(N) < 0:
        raise ValueError("N must be nonnegative")
    if float(vth) <= 0.0:
        raise ValueError("vth must be positive")
    v = jnp.asarray(v, dtype=jnp.float64)
    if int(N) == 0:
        return jnp.zeros((0, v.size), dtype=jnp.float64)

    xi = v / float(vth)
    h = jnp.zeros((int(N), v.size), dtype=jnp.float64).at[0].set(1.0)
    if int(N) > 1:
        h = h.at[1].set(xi)

    def body(i, h_):
        i_f = i.astype(jnp.float64)
        next_row = (xi / jnp.sqrt(i_f + 1.0)) * h_[i] - jnp.sqrt(i_f / (i_f + 1.0)) * h_[i - 1]
        return h_.at[i + 1].set(next_row)

    if int(N) > 2:
        h = jax.lax.fori_loop(1, int(N) - 1, body, h)
    return h.astype(jnp.float64)


def hermite_dual_basis(N: int, v: Array) -> Array:
    return hermite_dual_basis_scaled(int(N), v, vth=1.0)


def project_distribution_snapshot_to_fourier_hermite(
    f_phys: Array,
    v: Array,
    projection_order: int,
    *,
    vth: float = 1.0,
    equilibrium: Optional[Array] = None,
    dual_basis: Optional[Array] = None,
) -> Array:
    f_phys = jnp.asarray(f_phys, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    if equilibrium is not None:
        f_phys = f_phys - jnp.asarray(equilibrium, dtype=jnp.float64)[:, None]
    H = dual_basis if dual_basis is not None else hermite_dual_basis_scaled(int(projection_order), v, vth=vth)
    a_phys = jnp.trapezoid(
        f_phys[None, :, :] * H[:, :, None],
        x=v,
        axis=1,
    )
    return jnp.fft.rfft(a_phys, axis=1).astype(jnp.complex128)


def project_distribution_history_to_fourier_hermite(
    f_hist: Array,
    v: Array,
    projection_order: int,
    *,
    vth: float = 1.0,
    equilibrium: Optional[Array] = None,
    dual_basis: Optional[Array] = None,
) -> Array:
    projector = lambda f_state: project_distribution_snapshot_to_fourier_hermite(
        f_state,
        v,
        int(projection_order),
        vth=vth,
        equilibrium=equilibrium,
        dual_basis=dual_basis,
    )
    return jax.vmap(projector)(jnp.asarray(f_hist, dtype=jnp.float64))


def extract_interface_supervised_pairs_from_coeff_history(
    a_hat_hist: np.ndarray,
    *,
    Nv_targets: Sequence[int],
    Nm: int,
    k_arr: np.ndarray,
    vth: float,
    include_global_indicators: bool = True,
    n_low: int = 2,
    context_mode: str = "none",
) -> Dict[int, Dict[str, np.ndarray]]:
    a_hat_hist = np.asarray(a_hat_hist, dtype=np.complex128)
    k_arr = np.asarray(k_arr, dtype=np.float64)
    if a_hat_hist.ndim != 3:
        raise ValueError(f"a_hat_hist must have shape (T, Nv_ref, Nk), got {a_hat_hist.shape}")
    if int(n_low) < 0:
        raise ValueError("n_low must be nonnegative")
    if str(context_mode) not in {"none", "lag1_delta"}:
        raise ValueError(f"Unsupported context_mode={context_mode!r}")

    out: Dict[int, Dict[str, np.ndarray]] = {}
    k_nonzero = k_arr[1:]
    abs_k = np.abs(k_nonzero)[None, :, None]
    global_features = None
    if include_global_indicators:
        field_activity, low_energy = jax.vmap(
            lambda a_hat: learned_closure_global_indicators(
                a_hat,
                jnp.asarray(k_arr, dtype=jnp.float64),
                n_low=int(n_low),
            )
        )(jnp.asarray(a_hat_hist, dtype=jnp.complex128))
        global_features = np.stack(
            [
                np.asarray(field_activity, dtype=np.float64),
                np.asarray(low_energy, dtype=np.float64),
            ],
            axis=-1,
        )

    for Nv in Nv_targets:
        if int(Nm) > int(Nv):
            raise ValueError(f"Nm={Nm} exceeds target Nv={Nv}")
        if int(Nv) >= a_hat_hist.shape[1]:
            raise ValueError(
                f"Target Nv={Nv} requires coefficient index {Nv}, but projection order={a_hat_hist.shape[1]}"
            )

        tail = np.transpose(a_hat_hist[:, int(Nv) - int(Nm) : int(Nv), 1:], (0, 2, 1))
        q = (-1j * k_nonzero[None, :] * float(vth) * math.sqrt(float(Nv)) * a_hat_hist[:, int(Nv), 1:])
        nv_col = np.full((tail.shape[0], tail.shape[1], 1), float(Nv), dtype=np.float64)
        base_pieces = [
            np.real(tail),
            np.imag(tail),
            np.broadcast_to(abs_k, (tail.shape[0], tail.shape[1], 1)),
            nv_col,
        ]
        if global_features is not None:
            base_pieces.append(
                np.broadcast_to(
                    global_features[:, None, :],
                    (tail.shape[0], tail.shape[1], 2),
                )
            )
        base_inputs = np.concatenate(base_pieces, axis=-1)
        targets = np.stack([np.real(q), np.imag(q)], axis=-1)
        base_dim = 2 * int(Nm) + (4 if include_global_indicators else 2)
        if context_mode == "none":
            inputs = base_inputs
            targets_out = targets
            input_dim = base_dim
        else:
            if base_inputs.shape[0] < 2:
                inputs = np.zeros((0, base_inputs.shape[1], 3 * base_dim), dtype=np.float64)
                targets_out = np.zeros((0, targets.shape[1], 2), dtype=np.float64)
            else:
                current = base_inputs[1:]
                previous = base_inputs[:-1]
                inputs = np.concatenate([current, previous, current - previous], axis=-1)
                targets_out = targets[1:]
            input_dim = 3 * base_dim
        out[int(Nv)] = {
            "inputs_base": inputs.reshape(-1, input_dim).astype(np.float64),
            "targets": targets_out.reshape(-1, 2).astype(np.float64),
        }
    return out


def extract_interface_rollout_windows_from_coeff_history(
    a_hat_hist: np.ndarray,
    *,
    Nv_targets: Sequence[int],
    Nm: int,
    k_arr: np.ndarray,
    vth: float,
    rollout_horizon: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Extract short retained-state rollout windows for stability-aware closure training.

    For each target Nv, windows are built from consecutive projected states

        (a^{n-1}, a^n) -> {a^{n+1}, ..., a^{n+H}}

    using the retained hierarchy only. Electric-field targets are stored explicitly
    because they are used by the rollout-aware field penalty.
    """
    del Nm, vth  # retained-state windows depend only on projected coefficient history.
    a_hat_hist = np.asarray(a_hat_hist, dtype=np.complex128)
    k_arr = np.asarray(k_arr, dtype=np.float64)
    H = int(rollout_horizon)
    if a_hat_hist.ndim != 3:
        raise ValueError(f"a_hat_hist must have shape (T, Nv_ref, Nk), got {a_hat_hist.shape}")
    if H < 1:
        raise ValueError("rollout_horizon must be positive")

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for Nv in Nv_targets:
        Nv = int(Nv)
        if Nv >= a_hat_hist.shape[1]:
            raise ValueError(
                f"Target Nv={Nv} requires coefficient index {Nv}, but projection order={a_hat_hist.shape[1]}"
            )
        max_start = a_hat_hist.shape[0] - H
        if max_start <= 1:
            out[Nv] = {
                "prev_state": np.zeros((0, Nv, a_hat_hist.shape[2]), dtype=np.complex128),
                "curr_state": np.zeros((0, Nv, a_hat_hist.shape[2]), dtype=np.complex128),
                "future_state": np.zeros((0, H, Nv, a_hat_hist.shape[2]), dtype=np.complex128),
                "future_E_hat": np.zeros((0, H, a_hat_hist.shape[2]), dtype=np.complex128),
            }
            continue

        prev_state = []
        curr_state = []
        future_state = []
        future_E_hat = []
        for n in range(1, max_start):
            prev = a_hat_hist[n - 1, :Nv, :]
            curr = a_hat_hist[n, :Nv, :]
            future = a_hat_hist[n + 1 : n + 1 + H, :Nv, :]
            e_hat_future = np.zeros((H, curr.shape[1]), dtype=np.complex128)
            if curr.shape[1] > 1:
                e_hat_future[:, 1:] = (1j * future[:, 0, 1:]) / k_arr[None, 1:]
            prev_state.append(prev)
            curr_state.append(curr)
            future_state.append(future)
            future_E_hat.append(e_hat_future)
        out[Nv] = {
            "prev_state": np.asarray(prev_state, dtype=np.complex128),
            "curr_state": np.asarray(curr_state, dtype=np.complex128),
            "future_state": np.asarray(future_state, dtype=np.complex128),
            "future_E_hat": np.asarray(future_E_hat, dtype=np.complex128),
        }
    return out
