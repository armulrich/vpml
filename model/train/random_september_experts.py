"""Random split-sign-family topology, without fitted signs or case thresholds.

Returns trainable weights separately from structural constants. No checkpoint
weights are read. Historical mechanisms remain subject to ablation/preflight.
"""
import math

import jax
import jax.numpy as jnp

from model.train.fresh_coupled_latent import fresh_parameters


def initialize(config):
    if config['latent_rank'] != 60 or config['latent_readout_mode'] != 'equilibrium_cnn':
        raise ValueError('Require the compact rank60 equilibrium CNN')
    shared = fresh_parameters(config)
    # Only operator_v_real.shape is used by the equilibrium CNN step to infer
    # the delay-input layout. Spectral projection weights are inactive here.
    constants = {'operator_v_real': jnp.zeros_like(shared['operator_v_real']),
                 'output_bias': jnp.zeros_like(shared['output_bias']),
                 'cyclic_contract_before_convolution': jnp.asarray(1., dtype=jnp.float64),
                 'specialist_unrestricted_activation': jnp.asarray(1., dtype=jnp.float64)}
    trainable = {k: jnp.asarray(v, dtype=jnp.float64) for k,v in shared.items()
                 if not k.startswith('operator_') and k != 'output_bias'}
    key = jax.random.fold_in(jax.random.PRNGKey(config['seed']), 609)

    def normal(shape, scale):
        nonlocal key
        key, draw = jax.random.split(key)
        return scale * jax.random.normal(draw, shape, dtype=jnp.float64)

    width, kernel, channels, experts, features = config['width'], config['kernel_size'], 60, 4, 68
    small = float(config['operator_output_init_scale'])
    for k,v in list(trainable.items()):
        if k.endswith('_bias'):
            trainable[k] = normal(v.shape, .01 / math.sqrt(width))
    ordinary = normal((experts,channels,width,kernel), small/math.sqrt(width*kernel))
    cyclic = normal((experts,features,channels,width,kernel), small/math.sqrt(features*width*kernel))
    # Preserve the shared function at conversion while breaking expert
    # symmetry.  Uniform routing initially contracts these residuals to zero,
    # but each expert still has a distinct nonzero routing derivative.
    trainable['expert_output_kernel'] = ordinary - jnp.mean(ordinary, axis=0, keepdims=True)
    trainable['expert_cyclic_output_kernel'] = cyclic - jnp.mean(cyclic, axis=0, keepdims=True)
    constants['expert_output_bias'] = jnp.zeros((experts,channels), dtype=jnp.float64)
    constants['expert_cyclic_output_bias'] = jnp.zeros((experts,features,channels), dtype=jnp.float64)
    # With unrestricted specialist activation these preliminary gates are
    # replaced exactly, so keep only the required structural placeholders.
    constants['expert_gate_kernel'] = jnp.zeros((experts,width), dtype=jnp.float64)
    constants['expert_gate_bias'] = jnp.zeros((experts,), dtype=jnp.float64)
    # Preserve the two causal affine routing branches. No fitted router values,
    # predetermined expert identities, deadband, or signed slice scales enter.
    for prefix in ('specialist_router', 'specialist_high_router'):
        trainable[prefix+'_bounce_gate_gain'] = normal((experts,features), .1/math.sqrt(features))
        trainable[prefix+'_gate_bias'] = normal((experts,), .01)
        trainable[prefix+'_null_gate_bias'] = normal((1,), .01)
    trainable['specialist_router_amplitude_split'] = jnp.stack((normal((),.1), jax.nn.softplus(normal((),.1))))
    trainable['expert_cyclic_low_mode_closure_gain'] = normal((features,2), small/math.sqrt(features))
    return trainable, constants
