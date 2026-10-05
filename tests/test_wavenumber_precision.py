"""Precision is uniform across data, neural parameters, optimizer and rollout."""
import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from model.train.wavenumber_training import Configuration,initialize,batch_from_states,update_factory,base,save_snapshot,load_snapshot
from model.train.wavenumber_evaluation import rollout_factory


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        jax.config.update('jax_enable_x64',True)
        self.config=Configuration(nx=8,width=2,depth=1,modes=2,latent_dim=2,memory_steps=3,
            memory_stride=1,preparation_steps=2,horizon=4,detach_preparation=True,relative_loss_norm='l2')
        self.manifest={'cases':[dict(case_id='case',epsilon=.01,regime='linear_landau',domain_length=4*np.pi)]}
        x=np.arange(8)*2*np.pi/8
        a=np.zeros((8,3,8),np.float32);a[:,0]=.01*np.cos(x);a[:,2]=a[:,0]
        self.states={'case':a}
        self.stats=dict(input_scale=np.ones(4,np.float32),regime_scales=np.ones((3,4),np.float32),
            heat_flux_gradient_scale=np.array([1.]),amplitude_center=np.array([0.]),amplitude_scale=np.array([1.]))

    def test_seed_values_preserved_when_precision_changes(self):
        p32=initialize(self.config);p64=initialize(replace(self.config,compute_dtype='float64'))
        for k in p32:
            self.assertEqual(p32[k].dtype,jnp.float32)
            self.assertEqual(p64[k].dtype,jnp.float64)
            np.testing.assert_array_equal(np.asarray(p32[k]).astype(np.float64),p64[k])
        with self.assertRaises(ValueError):replace(self.config,compute_dtype='float16')

    def test_data_update_and_carried_state_do_not_downcast(self):
        for dtype in ('float32','float64'):
            c=replace(self.config,compute_dtype=dtype)
            b=batch_from_states(self.manifest,np.array([[0,0]]),self.states,c)
            for k,v in b.items():
                if k!='regime_index':self.assertEqual(v.dtype,np.dtype(dtype))
            p=initialize(c);o=base._adam_init(p)
            updated=update_factory(c,self.stats)(p,o,b,.0001)
            self.assertTrue(bool(updated[-1]))
            for x in jax.tree_util.tree_leaves(updated[:2]):
                if jnp.issubdtype(x.dtype,jnp.floating):self.assertEqual(x.dtype,np.dtype(dtype))
            roll=rollout_factory(c,self.stats,chunk_steps=2)
            arr,mem=roll(p,jnp.asarray(b['initial']),jnp.asarray(b['memory']),jnp.asarray(b['heat_flux_gradient_history']),
                jnp.array(0,jnp.int32),jnp.zeros((1,8),dtype=dtype),None,jnp.asarray(b['amplitude']),jnp.asarray(b['k_arr'][0]))
            self.assertEqual(arr.dtype,np.dtype(dtype))
            for x in jax.tree_util.tree_leaves(mem):
                if jnp.issubdtype(x.dtype,jnp.floating):self.assertEqual(x.dtype,np.dtype(dtype))

    def test_float64_resume_with_nonzero_adam_state(self):
        c=replace(self.config,compute_dtype='float64')
        b=batch_from_states(self.manifest,np.array([[0,0]]),self.states,c)
        step=update_factory(c,self.stats);p=initialize(c);o=base._adam_init(p)
        first=step(p,o,b,.0001)
        uninterrupted=step(first[0],first[1],b,.0001)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.npz'
            save_snapshot(path,first[0],first[1],{'protocol_sha256':'test'},[[1,float(first[2]),float(first[3]),float(first[4]),float(first[5])]])
            restored_p,restored_o,_,_=load_snapshot(path,'test')
            resumed=step(restored_p,restored_o,b,.0001)
        for a,d in zip(jax.tree_util.tree_leaves(uninterrupted),jax.tree_util.tree_leaves(resumed)):
            np.testing.assert_array_equal(a,d)


if __name__=='__main__':unittest.main()
