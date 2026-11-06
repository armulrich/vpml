"""Compatibility exports for learned interface-closure models."""

from vpml.models.interface_closure import (
    LearnedInterfaceClosure,
    init_interface_closure_params,
)
from vpml.models.interface_checkpoint import (
    load_learned_interface_closure_npz,
    save_learned_interface_closure_npz,
)

__all__ = [
    "LearnedInterfaceClosure",
    "init_interface_closure_params",
    "load_learned_interface_closure_npz",
    "save_learned_interface_closure_npz",
]
