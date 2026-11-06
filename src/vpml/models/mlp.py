"""Real-valued MLP initialization and evaluation for learned closures."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

from .._jax_precision import enable_x64

enable_x64()

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def init_real_mlp(key: Array, layer_sizes: Sequence[int]) -> Dict[str, Array]:
    """
    Initialize a real-valued MLP with Xavier-style Gaussian weights.
    """
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must contain at least an input and output size")

    params: Dict[str, Array] = {}
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if m <= 0 or n <= 0:
            raise ValueError("All layer sizes must be positive")
        scale = jnp.sqrt(2.0 / float(m + n))
        params[f"W{i}"] = scale * jax.random.normal(keys[i], (m, n), dtype=jnp.float64)
        params[f"b{i}"] = jnp.zeros((n,), dtype=jnp.float64)
    return params


def mlp_apply(
    params: Dict[str, Array],
    x: Array,
    *,
    activation: Callable[[Array], Array] = jnp.tanh,
) -> Array:
    """Apply a real-valued MLP to a single feature vector."""
    h = jnp.asarray(x, dtype=jnp.float64)
    num_layers = len(params) // 2
    for i in range(num_layers):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        if i < num_layers - 1:
            h = activation(h)
    return h
