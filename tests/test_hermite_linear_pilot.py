"""Check that pilot instrumentation preserves historical optimizer behavior."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from model.diagnostics.hermite_linear_pilot import make_update
from model.train.interface_flux_rollout import adam_init, adam_step


class PilotUpdateTests(unittest.TestCase):
    def test_same_clipped_adam_update(self):
        params={"w":jnp.array([2.,-3.],dtype=jnp.float64)}
        state=adam_init(params)
        def objective(p,b):
            loss=jnp.sum((p["w"]-b)**2)
            return loss,{"q":loss}
        target=jnp.array([.1,.2],dtype=jnp.float64)
        actual,new_state,loss,norm,delta,finite=make_update(objective)(params,state,target)
        grads=jax.grad(lambda p:objective(p,target)[0])(params)
        expected,expected_state=adam_step(params,grads,state,1e-4,grad_clip=.5)
        for a,b in zip(jax.tree.leaves((actual,new_state)),jax.tree.leaves((expected,expected_state))):
            np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-15)
        self.assertTrue(bool(finite));self.assertGreater(float(norm),.5)

    def test_nonfinite_gradient_preserves_params_and_counter(self):
        params={"w":jnp.array([2.],dtype=jnp.float64)};state=adam_init(params)
        def objective(p,b):
            loss=jnp.sum(p["w"])*b
            return loss,{"q":loss}
        actual,new_state,_,_,_,finite=make_update(objective)(params,state,jnp.nan)
        self.assertFalse(bool(finite))
        for a,b in zip(jax.tree.leaves((actual,new_state)),jax.tree.leaves((params,state))):
            np.testing.assert_array_equal(a,b)


if __name__=="__main__":unittest.main()
