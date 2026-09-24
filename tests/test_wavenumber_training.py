import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from model.train.wavenumber_data import build_manifest,epoch_batches,REGIMES,write_new_json
from model.train.wavenumber_training import (Configuration,initialize,loss_factory,tree_hash,
    save_snapshot,load_snapshot,update_factory,base)
import jax
import jax.numpy as jnp


def original():
    return {'cases':[dict(case_id=f'{r}_{i}',regime=r,epsilon=.01,modes=[.5,1,1.5,2],
        mode_weights=[.25]*4,relative_phases=[0.]*4,shape_normalization=1.,split='train' if i<16 else 'heldout')
        for r in REGIMES for i in range(20)]}


def tiny():
    c=Configuration(nx=16,width=2,depth=1,modes=3,memory_steps=3,memory_stride=1,preparation_steps=2,horizon=3)
    x=np.arange(16)*2*np.pi/16
    state=np.stack([.01*np.cos(x),np.zeros(16),.01*np.cos(x)]).astype(np.float32)
    batch=dict(initial=np.stack([state]*2),targets=np.tile(state,(2,3,1,1)),memory=np.zeros((2,3,3,16),np.float32),
        heat_flux_gradient_history=np.zeros((2,3,16),np.float32),amplitude=np.array([.01,.01],np.float32),
        regime_index=np.array([0,1],np.int32),k_arr=np.array([np.arange(9)*.3,np.arange(9)*.5],np.float32))
    stats=dict(input_scale=np.ones(4),regime_scales=np.ones((3,4)),heat_flux_gradient_scale=np.ones(1),amplitude_center=np.zeros(1),amplitude_scale=np.ones(1))
    return c,batch,stats


class WavenumberTests(unittest.TestCase):
    def test_complete_sweep_and_domain_split(self):
        m=build_manifest(original());batches=list(epoch_batches(m,1))
        self.assertEqual(len(batches),2302);self.assertEqual(len(batches[-1]),12)
        rows=np.concatenate(batches);self.assertEqual(len(rows),110460)
        self.assertEqual(len(set(map(tuple,rows))),len(rows))
        self.assertFalse(any(m['cases'][i]['split']!='train' for i in rows[:,0]))
        self.assertTrue(all(sum(m['cases'][i]['regime']==r for i,a in b)==len(b)//3 for b in batches for r in REGIMES))
        self.assertEqual(m,build_manifest(original()))
        np.testing.assert_array_equal(rows,np.concatenate(list(epoch_batches(m,1))))
        self.assertFalse(np.array_equal(rows,np.concatenate(list(epoch_batches(m,2)))))

    def test_forward_parity_and_initializer_gradient(self):
        c,b,s=tiny();p=initialize(c)
        live=loss_factory(c,s); detached=loss_factory(replace(c,detach_preparation=True),s)
        lv,lg=jax.jit(jax.value_and_grad(live))(p,b)
        dv,dg=jax.jit(jax.value_and_grad(detached))(p,b)
        np.testing.assert_allclose(lv,dv,rtol=0,atol=1e-10)
        keys=[k for k in p if k.startswith('compact_latent_init_')]
        self.assertEqual(sum(float(jnp.sum(dg[k]**2)) for k in keys),0.)
        self.assertGreater(sum(float(jnp.sum(lg[k]**2)) for k in keys),0.)
        self.assertTrue(np.isfinite(float(lv)))
        # Batched geometry must agree with independent per-domain evaluations.
        singles=[float(live(p,{k:v[i:i+1] for k,v in b.items()})) for i in range(2)]
        self.assertAlmostEqual(float(lv),np.mean(singles),places=7)

    def test_exact_resume_and_overwrite_refusal(self):
        c,b,s=tiny();p=initialize(c);o=base._adam_init(p);step=update_factory(c,s)
        p,o,*rest=step(p,o,b,.001);self.assertTrue(bool(rest[-1]))
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'state.npz';record=dict(protocol_sha256='test',epoch=1,next_batch=1)
            save_snapshot(f,p,o,record,[[1,0.,0.,0.,0.]])
            pr,orr,rr,_=load_snapshot(f,'test')
            p2,o2,*_=step(p,o,b,.001);pr2,or2,*_=step(pr,orr,b,.001)
            self.assertEqual(tree_hash(p2),tree_hash(pr2))
            for part in ('m','v'):self.assertEqual(tree_hash(o2[part]),tree_hash(or2[part]))
            with self.assertRaises(FileExistsError):save_snapshot(f,p,o,record,[])
            with self.assertRaises(ValueError):load_snapshot(f,'other')
            write_new_json(Path(d)/'x.json',{})
            with self.assertRaises(FileExistsError):write_new_json(Path(d)/'x.json',{})
