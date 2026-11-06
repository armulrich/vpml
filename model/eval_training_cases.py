"""Compatibility entry point for :mod:`vpml.evaluation.training_cases`."""

from vpml.evaluation.training_cases import (
    VALID_REGIMES,
    parse_float_tuple,
    parse_str_tuple,
    _sample_initial_condition_pair,
    _slug,
    build_arg_parser,
    _case_specs,
    _sweep_args,
    main,
)


if __name__ == "__main__":
    main()
