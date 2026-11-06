"""Compatibility entry point for :mod:`vpml.benchmarks.fh_nonlinear_sim_jax`."""

from vpml.benchmarks.fh_nonlinear_sim_jax import (
    TwoStreamParams,
    BumpOnTailParams,
    simulate_two_stream,
    simulate_bump_on_tail,
    plot_snapshot_panel,
    main,
)


if __name__ == "__main__":
    main()
