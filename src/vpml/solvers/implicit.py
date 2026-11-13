"""Tridiagonal solves and implicit-midpoint Newton-Krylov integration."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple
import math
import numpy as np

from .._jax_precision import enable_x64

enable_x64()

import jax
import jax.numpy as jnp

try:
    import scipy.sparse.linalg as spla
except Exception:  # pragma: no cover
    spla = None

Array = jnp.ndarray


def tridiag_solve(sub: Array, diag: Array, sup: Array, rhs: Array) -> Array:
    """
    Thomas algorithm for a single tridiagonal system in JAX.

    Parameters
    ----------
    sub : (N-1,) complex
        Sub-diagonal entries A[i, i-1] for i=1..N-1.
    diag : (N,) complex
        Diagonal entries A[i,i].
    sup : (N-1,) complex
        Super-diagonal entries A[i, i+1] for i=0..N-2.
    rhs : (N,) complex
        Right-hand side.

    Returns
    -------
    x : (N,) complex
        Solution of Ax=rhs.
    """
    N = rhs.shape[0]
    z = jnp.zeros((), dtype=rhs.dtype)

    # Pad to length N for scan-friendly indexing.
    sub_p = jnp.concatenate([jnp.array([z], dtype=rhs.dtype), sub])      # (N,)
    sup_p = jnp.concatenate([sup, jnp.array([z], dtype=rhs.dtype)])      # (N,)

    # Forward sweep
    c0 = sup_p[0] / diag[0]
    d0 = rhs[0] / diag[0]

    def fwd(carry, i):
        c_prev, d_prev = carry
        den = diag[i] - sub_p[i] * c_prev
        c = sup_p[i] / den
        d = (rhs[i] - sub_p[i] * d_prev) / den
        return (c, d), (c, d)

    (_, _), hist = jax.lax.scan(fwd, (c0, d0), jnp.arange(1, N))
    c_full = jnp.concatenate([jnp.array([c0], dtype=rhs.dtype), hist[0]])
    d_full = jnp.concatenate([jnp.array([d0], dtype=rhs.dtype), hist[1]])

    # Back substitution
    xN = d_full[-1]

    def bwd(x_next, i):
        x_i = d_full[i] - c_full[i] * x_next
        return x_i, x_i

    _, x_rev = jax.lax.scan(bwd, xN, jnp.arange(N - 2, -1, -1))
    x0_to_N2 = x_rev[::-1]
    return jnp.concatenate([x0_to_N2, jnp.array([xN], dtype=rhs.dtype)])


def implicit_midpoint_jfnk_step(
    a_hat: Array,
    rhs_hat_fn: Callable[[Array], Array],
    dt: float,
    *,
    initial_guess: Optional[Array] = None,
    newton_tol: float = 1e-10,
    gmres_tol: float = 1e-10,
    max_newton_iter: int = 8,
    gmres_restart: Optional[int] = None,
    gmres_maxiter: Optional[int] = None,
    jvp_eps: float = 1e-8,
    post_step_filter: Optional[Array] = None,
) -> Tuple[Array, Dict[str, float]]:
    """
    One implicit-midpoint step solved with a Jacobian-free Newton-Krylov iteration.

    The nonlinear residual is

        R(y) = y - a^n - dt * F((a^n + y)/2),

    where ``F`` is provided by ``rhs_hat_fn``. The linear solve inside each Newton
    step uses GMRES with Jacobian-vector products approximated by directional
    finite differences of the residual.

    Notes
    -----
    This routine is intended for small benchmark problems where matching the
    paper's implicit-midpoint/JFNK algorithm matters more than raw throughput.
    """
    if spla is None:  # pragma: no cover
        raise RuntimeError(
            "implicit_midpoint_jfnk_step requires SciPy's sparse linear algebra support."
        )

    a0 = jnp.asarray(a_hat, dtype=jnp.complex128)
    shape = a0.shape
    dt = float(dt)

    def _flatten(arr: Array) -> np.ndarray:
        return np.asarray(arr, dtype=np.complex128).reshape(-1)

    def _reshape(vec: np.ndarray) -> Array:
        return jnp.asarray(np.asarray(vec, dtype=np.complex128).reshape(shape), dtype=jnp.complex128)

    def residual(state: Array) -> Array:
        midpoint = 0.5 * (a0 + state)
        return state - a0 - dt * rhs_hat_fn(midpoint)

    if initial_guess is None:
        y = a0 + dt * rhs_hat_fn(a0)
    else:
        y = jnp.asarray(initial_guess, dtype=jnp.complex128)

    residual_target = math.sqrt(float(np.prod(shape))) * max(
        float(newton_tol),
        100.0 * float(gmres_tol),
    )
    last_residual = float("inf")
    last_step_norm = float("inf")
    total_gmres_iters = 0

    for newton_iter in range(1, int(max_newton_iter) + 1):
        r = residual(y)
        r_np = _flatten(r)
        last_residual = float(np.linalg.norm(r_np))
        if last_residual <= residual_target:
            break

        def matvec(v_np: np.ndarray) -> np.ndarray:
            v_norm = max(1.0, float(np.linalg.norm(v_np)))
            if v_norm == 0.0:
                return np.zeros_like(v_np, dtype=np.complex128)
            v_j = _reshape(v_np)
            try:
                _, jv = jax.jvp(residual, (y,), (v_j,))
                return np.array(_flatten(jv), dtype=np.complex128, copy=True)
            except Exception:
                eps = float(jvp_eps) / v_norm
                rp = residual(y + eps * v_j)
                return np.array((_flatten(rp) - r_np) / eps, dtype=np.complex128, copy=True)

        linop = spla.LinearOperator(
            (r_np.size, r_np.size),
            matvec=matvec,
            dtype=np.complex128,
        )

        gmres_iters = [0]

        def gmres_callback(_residual: np.ndarray) -> None:
            gmres_iters[0] += 1

        try:
            delta_np, info = spla.gmres(
                linop,
                -r_np,
                restart=gmres_restart,
                maxiter=gmres_maxiter,
                rtol=float(gmres_tol),
                atol=0.0,
                callback=gmres_callback,
                callback_type="legacy",
            )
        except TypeError:  # pragma: no cover
            delta_np, info = spla.gmres(
                linop,
                -r_np,
                restart=gmres_restart,
                maxiter=gmres_maxiter,
                tol=float(gmres_tol),
                atol=0.0,
                callback=gmres_callback,
                callback_type="legacy",
            )

        total_gmres_iters += gmres_iters[0]
        if info != 0:
            raise RuntimeError(
                f"GMRES failed during implicit_midpoint_jfnk_step with info={info} "
                f"at Newton iteration {newton_iter}."
            )

        last_step_norm = float(np.linalg.norm(delta_np))
        y = y + _reshape(delta_np)

    final_residual = float(np.linalg.norm(_flatten(residual(y))))
    if final_residual > residual_target:
        raise RuntimeError(
            "implicit_midpoint_jfnk_step did not converge: "
            f"final residual={final_residual:.3e}, target={residual_target:.3e}"
        )

    if post_step_filter is not None:
        y = jnp.asarray(post_step_filter, dtype=jnp.float64)[:, None] * y

    info = {
        "newton_iters": float(newton_iter),
        "gmres_iters": float(total_gmres_iters),
        "residual_norm": final_residual,
        "residual_target": residual_target,
        "step_norm": last_step_norm,
    }
    return y, info
