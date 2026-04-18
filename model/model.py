"""Thin learned-model surface layered on top of vpml runtime types."""

from vpml.core import (
    LearnedInterfaceClosure,
    init_interface_closure_params,
    load_learned_interface_closure_npz,
    save_learned_interface_closure_npz,
)

__all__ = [
    "LearnedInterfaceClosure",
    "init_interface_closure_params",
    "load_learned_interface_closure_npz",
    "save_learned_interface_closure_npz",
]
