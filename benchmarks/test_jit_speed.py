"""
Quick test that JIT is working for the nonlinear simulation.

Runs simulate_two_stream twice with the same parameters. The second reuses compiled code and should be noticibly faster.
If the second run is at least 1.5x faster, it reports that JIT appears to be working.

Usage:
    python -m benchmarks.test_jit_speed
"""
from __future__ import annotations

import time

from benchmarks.fh_nonlinear_sim_jax import TwoStreamParams, simulate_two_stream


def main() -> None:
    # (Nx=64, Nv=40, T=20, 200 time steps)
    p = TwoStreamParams(Nx=64, Nv=40, T=20.0, dt=0.1) #Parameters

    print("Running simulate_two_stream twice (same params)...")
    print("  First run: compile + execute")
    print("  Second run: execute only (reuses JIT)\n")
    # First run
    t0 = time.perf_counter()
    simulate_two_stream(p)
    t1 = time.perf_counter()
    first_s = t1 - t0
    # Second run
    t0 = time.perf_counter()
    simulate_two_stream(p)
    t1 = time.perf_counter()
    second_s = t1 - t0

    print(f"First run:  {first_s:.3f} s")
    print(f"Second run: {second_s:.3f} s")

    if second_s > 0 and first_s / second_s >= 1.5:
        print("\nJIT appears to be working (second run is noticeably faster).")
    elif second_s > 0 and first_s > second_s:
        print("\nSecond run was faster, but JIT may or may not be working.")
    else:
        print("\nSecond run was not faster.")


if __name__ == "__main__":
    main()
