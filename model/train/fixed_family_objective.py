"""The existing event-agnostic objective on full autonomous trajectories."""
import jax.numpy as jnp
import numpy as np

from model.train.coupled_run_support import make_model_functions
from model.train.fresh_coupled_latent import event_agnostic_loss, local_window_mean
from vpml.kinetic_latent import resolved_hermite_to_low_moment_state
from vpml.low_moment import primitive_fields


def make_fixed_objective(config, arrays, teacher, *, field_components=False):
    _, rollout, _ = make_model_functions(config, arrays, teacher)
    nx = config["nx"]
    def fields(values):
        physical = primitive_fields(resolved_hermite_to_low_moment_state(values.reshape(-1, 3, nx)),
            jnp.asarray(arrays["k_arr"]), poisson_sign=float(teacher["teacher_poisson_sign"]))
        return physical.reshape(values.shape[0], values.shape[1], 4, nx)
    def objective(params, target):
        resolved, latent = rollout(params, target[:, 0, :3], target[:, 0, 3:])
        resolved = jnp.concatenate((target[:, :1, :3], resolved), axis=1)
        latent = jnp.concatenate((target[:, :1, 3:], latent), axis=1)
        predicted, truth = fields(resolved), fields(target[:, :, :3])
        result = event_agnostic_loss(predicted, truth, latent, target[:, :, 3:],
            cadence=config["cadence"], envelope_weight=config["envelope_weight"],
            change_weight=config["envelope_change_weight"], latent_weight=config["latent_weight"],
            floor_ratio=config["energy_floor_ratio"])
        if not field_components:
            return result
        energy = jnp.mean(truth[:, :, 3]**2, axis=-1)
        error = jnp.mean((predicted[:, :, 3]-truth[:, :, 3])**2, axis=-1)
        whole = jnp.sum(error, axis=1)/jnp.maximum(jnp.sum(energy, axis=1), 1e-20/config["nx"])
        floor = jnp.maximum(config["energy_floor_ratio"]*jnp.max(energy, axis=1, keepdims=True), 1e-30)
        local = []
        for duration in (2.5, 5., 10.):
            width = max(1, int(round(duration/config["cadence"])))
            if 2*width <= energy.shape[1]:
                local.append(jnp.mean(local_window_mean(error,width)/(local_window_mean(energy,width)+floor),axis=1))
        return jnp.stack((whole,jnp.mean(jnp.stack(local),axis=0)),axis=1).reshape(-1), result
    return objective


def make_regime_balanced_objective(config, arrays, teacher, denominators):
    """Fixed six regime-field means plus envelope, using balanced triplets.

    Every batch must contain one linear, one strong, then one weak case.
    Denominators are frozen before training and never updated across epochs.
    Original physical/latent/envelope/transfer components remain diagnostics.
    """
    scales = np.asarray(denominators, dtype=np.float64)
    if scales.shape != (7,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Require seven positive finite frozen denominators")
    base = make_fixed_objective(config, arrays, teacher, field_components=True)
    def objective(params, target):
        if target.shape[0] != 3:
            raise ValueError("Balanced objective requires linear/strong/weak triplets")
        field, (legacy_total, components) = base(params, target)
        values = jnp.concatenate((field, components[2:3]))
        score = jnp.mean(values / jnp.asarray(scales, dtype=values.dtype))
        return score, (values, legacy_total, components)
    return objective
