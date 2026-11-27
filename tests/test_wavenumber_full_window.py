"""Full-window supervision preserves the model and the complete forward state."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from vpml.training.wavenumber_training import (
    Configuration,
    initialize,
    batch_from_states,
    loss_factory,
    update_factory,
    save_snapshot,
    load_snapshot,
)
from vpml.training.optimization import adam_init
from vpml.solvers.moment_rollout import rollout_burles_latent_closure
from vpml.solvers.moments import primitive_fields


class FullWindowTests(unittest.TestCase):
    def setUp(self):
        jax.config.update('jax_enable_x64', True)
        self.old=Configuration(nx=8,width=2,depth=1,modes=2,latent_dim=2,
            memory_steps=3,memory_stride=1,preparation_steps=2,horizon=4,
            detach_preparation=True,relative_loss_norm='l2',compute_dtype='float64')
        self.new=replace(self.old,preparation_steps=0,horizon=6,detach_preparation=False)
        self.manifest={'cases':[dict(case_id='case',epsilon=.01,regime='linear_landau',domain_length=4*np.pi)]}
        x=np.arange(8)*2*np.pi/8
        states=np.zeros((16,3,8),np.float64)
        states[:,0]=(.01*np.exp(-np.arange(16)*.02))[:,None]*np.cos(x)
        states[:,2]=states[:,0]
        self.states={'case':states}
        self.stats=dict(input_scale=np.ones(4,np.float32),regime_scales=np.ones((3,4),np.float32),
            heat_flux_gradient_scale=np.array([1.]),amplitude_center=np.array([0.]),amplitude_scale=np.array([1.]))
        self.params=initialize(self.new)
        self.batch=batch_from_states(self.manifest,np.array([[0,0]]),self.states,self.new)
        self.kw=dict(dt=self.new.dt,memory_stride=1,input_scale=jnp.ones(4,jnp.float64),
            heat_flux_gradient_scale=1.,amplitude_center=0.,amplitude_scale=1.,
            input_scaling='current_density_rms_arcsinh',dynamic_amplitude_floor=1e-6,
            allow_uniform_heating=True,normalized_heat_flux_bound=128.,density_floor=1e-4,pressure_floor=1e-4)

    def rollout(self,p,steps,state=None,memory=None):
        b=self.batch
        if memory is None:
            h=b['memory'];ch=b['heat_flux_gradient_history'];count=jnp.array(0,jnp.int32);prev=None;z=None
        else:h,ch,_,count,prev,z=memory
        return rollout_burles_latent_closure(p,b['initial'] if state is None else state,h,count,
            b['amplitude'],b['k_arr'][0],horizon=steps,heat_flux_gradient_history=ch,
            previous_heat_flux_gradient=prev,compact_latent=z,**self.kw)

    def test_same_initial_model_and_full_target_interval(self):
        oldparams=initialize(self.old)
        for k in self.params:np.testing.assert_array_equal(self.params[k],oldparams[k])
        oldbatch=batch_from_states(self.manifest,np.array([[0,0]]),self.states,self.old)
        np.testing.assert_array_equal(self.batch['targets'][0],self.states['case'][1:7])
        np.testing.assert_array_equal(self.batch['targets'][:,2:],oldbatch['targets'])
        np.testing.assert_array_equal(self.batch['initial'],oldbatch['initial'])

    def test_full_loss_and_chunked_forward_parity(self):
        full,mem=self.rollout(self.params,6)
        first,prepared=self.rollout(self.params,2)
        last,splitmem=self.rollout(self.params,4,first[:,-1],prepared)
        np.testing.assert_allclose(full,jnp.concatenate([first,last],axis=1),rtol=1e-11,atol=1e-13)
        for a,b in zip(jax.tree_util.tree_leaves(mem),jax.tree_util.tree_leaves(splitmem)):
            np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-13)
        pf=primitive_fields(full[0],self.batch['k_arr'][0]);tf=primitive_fields(self.batch['targets'][0],self.batch['k_arr'][0])
        expected=jnp.sqrt(jnp.sum((pf-tf)**2)/jnp.sum(tf**2))
        actual=loss_factory(self.new,self.stats)(self.params,self.batch)
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)

    def test_initializer_receives_gradient_and_full_derivative_agrees(self):
        f=jax.jit(loss_factory(self.new,self.stats));val,g=jax.jit(jax.value_and_grad(f))(self.params,self.batch)
        self.assertTrue(np.isfinite(float(val)))
        init=sum(float(jnp.sum(v*v)) for k,v in g.items() if k.startswith('compact_latent_init_'))
        self.assertGreater(init,0.)
        oldbatch=batch_from_states(self.manifest,np.array([[0,0]]),self.states,self.old)
        oldg=jax.jit(jax.grad(loss_factory(self.old,self.stats)))(self.params,oldbatch)
        self.assertEqual(sum(float(jnp.sum(v*v)) for k,v in oldg.items() if k.startswith('compact_latent_init_')),0.)
        gn=jnp.sqrt(sum(jnp.sum(x*x) for x in g.values()));h=1e-5
        plus=jax.tree_util.tree_map(lambda p,d:p+h*d/gn,self.params,g)
        minus=jax.tree_util.tree_map(lambda p,d:p-h*d/gn,self.params,g)
        fd=(f(plus,self.batch)-f(minus,self.batch))/(2*h)
        np.testing.assert_allclose(fd,gn,rtol=2e-4,atol=1e-8)

    def test_two_update_resume_parity(self):
        step=update_factory(self.new,self.stats);o=adam_init(self.params)
        first=step(self.params,o,self.batch,.0001)
        self.assertTrue(bool(first[-1]));self.assertGreater(float(first[-2]),0)
        direct=step(first[0],first[1],self.batch,.0001)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'state.npz'
            save_snapshot(p,first[0],first[1],{'protocol_sha256':'full-window-test'},[[1,*map(float,first[2:6])]])
            pp,oo,_,_=load_snapshot(p,'full-window-test');resumed=step(pp,oo,self.batch,.0001)
        for a,b in zip(jax.tree_util.tree_leaves(direct),jax.tree_util.tree_leaves(resumed)):
            np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
