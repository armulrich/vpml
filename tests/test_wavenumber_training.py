import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from model.train.wavenumber_data import build_manifest,build_five_epoch_manifest,epoch_batches,REGIMES,write_new_json
from model.train.wavenumber_training import (Configuration,initialize,loss_factory,tree_hash,
    save_snapshot,load_snapshot,update_factory,batch_from_states,base)
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
    def test_initial_rms_loss_matches_formula_and_gradient(self):
        c,b,s=tiny();c=replace(c,detach_preparation=True,relative_loss_norm='l2')
        p=initialize(c); candidate=replace(c,relative_normalization='initial_rms')
        self.assertEqual(tree_hash(p),tree_hash(initialize(candidate)))
        b={**b,'initial_reference_rms':np.array([.02,.03],np.float32)}
        old=loss_factory(c,s); new=loss_factory(candidate,s)
        expected=[]
        for i in range(2):
            sample={k:v[i:i+1] for k,v in b.items()}
            target=base.primitive_fields(jnp.asarray(sample['targets'][0]),jnp.asarray(sample['k_arr'][0]))
            factor=float(jnp.sqrt(jnp.mean(target*target)))/float(b['initial_reference_rms'][i])
            value,g=jax.jit(jax.value_and_grad(old))(p,sample)
            actual,ng=jax.jit(jax.value_and_grad(new))(p,sample)
            np.testing.assert_allclose(actual,value*factor,rtol=3e-6)
            for key in g:np.testing.assert_allclose(ng[key],g[key]*factor,rtol=3e-4,atol=2e-5)
            expected.append(float(value)*factor)
        np.testing.assert_allclose(new(p,b),np.mean(expected),rtol=3e-6)
        doubled={**b,'initial_reference_rms':2*b['initial_reference_rms']}
        np.testing.assert_allclose(new(p,doubled),new(p,b)/2,rtol=2e-6)
        changed={**b,'regime_index':np.array([2,2],np.int32)}
        s2={**s,'regime_scales':np.full((3,4),12345.)}
        np.testing.assert_allclose(loss_factory(candidate,s2)(p,changed),new(p,b),rtol=0,atol=0)
        # A vanished reference window is valid when the IC scale remains positive.
        self.assertTrue(np.isfinite(float(new(p,{**b,'targets':np.zeros_like(b['targets'])}))))
        self.assertFalse(np.isfinite(float(new(p,{**b,'initial_reference_rms':np.zeros(2,np.float32)}))))

    def test_initial_scale_uses_t0_not_anchor_or_future(self):
        c,b,s=tiny();c=replace(c,relative_normalization='initial_rms',relative_loss_norm='l2')
        case=dict(case_id='test',regime=REGIMES[0],epsilon=.01,domain_length=2*np.pi/.3)
        history=np.stack([b['initial'][0]*np.exp(-i*.1) for i in range(32)])
        first=batch_from_states({'cases':[case]},[(0,0),(0,1)],{'test':history},c)
        np.testing.assert_array_equal(first['initial_reference_rms'][0],first['initial_reference_rms'][1])
        modified=history.copy();modified[1:]*=10
        other=batch_from_states({'cases':[case]},[(0,1)],{'test':modified},c)
        np.testing.assert_array_equal(first['initial_reference_rms'][:1],other['initial_reference_rms'])
        with self.assertRaises(ValueError):
            batch_from_states({'cases':[case]},[(0,0)],{'test':np.zeros_like(history)},c)

    def test_initial_rms_exact_resume(self):
        c,b,s=tiny();c=replace(c,detach_preparation=True,relative_loss_norm='l2',relative_normalization='initial_rms')
        b={**b,'initial_reference_rms':np.array([.02,.03],np.float32)}
        p=initialize(c);opt=base._adam_init(p);step=update_factory(c,s)
        p,opt,*rest=step(p,opt,b,.001);self.assertTrue(bool(rest[-1]))
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'state.npz';save_snapshot(f,p,opt,dict(protocol_sha256='initial'),[])
            pr,orr,_,_=load_snapshot(f,'initial')
            p2,o2,*_=step(p,opt,b,.001);pr2,or2,*_=step(pr,orr,b,.001)
            self.assertEqual(tree_hash(p2),tree_hash(pr2))
            for part in ('m','v'):self.assertEqual(tree_hash(o2[part]),tree_hash(or2[part]))

    def test_relative_l2_is_per_example_and_does_not_use_regime_weights(self):
        c,b,s=tiny(); c=replace(c,detach_preparation=True)
        p=initialize(c)
        candidate=replace(c,relative_loss_norm='l2')
        self.assertEqual(tree_hash(p),tree_hash(initialize(candidate)))
        old=loss_factory(c,s); new=loss_factory(candidate,s)
        expected=[]
        for i in range(2):
            sample={k:v[i:i+1] for k,v in b.items()}
            value,g=jax.jit(jax.value_and_grad(old))(p,sample)
            actual,ng=jax.jit(jax.value_and_grad(new))(p,sample)
            np.testing.assert_allclose(actual,jnp.sqrt(value),rtol=2e-6)
            for key in g:
                np.testing.assert_allclose(ng[key],g[key]/(2*jnp.sqrt(value)),rtol=3e-4,atol=2e-5)
            expected.append(float(jnp.sqrt(value)))
        np.testing.assert_allclose(new(p,b),np.mean(expected),rtol=2e-6)
        changed={**b,'regime_index':np.array([2,2],np.int32)}
        s2={**s,'regime_scales':np.full((3,4),12345.)}
        np.testing.assert_allclose(loss_factory(candidate,s2)(p,changed),new(p,b),rtol=0,atol=0)

    def test_l2_zero_error_and_invalid_denominator(self):
        root=base._sqrt_with_zero_subgradient
        f=lambda x:root(jnp.sum(x*x)/jnp.asarray(2.))
        value,g=jax.jit(jax.value_and_grad(f))(jnp.zeros(4))
        self.assertEqual(float(value),0.)
        np.testing.assert_array_equal(g,np.zeros(4))
        np.testing.assert_allclose(jax.grad(f)(jnp.array([3.,4.])),np.array([3.,4.])/(5*np.sqrt(2)),rtol=1e-6)
        self.assertFalse(bool(jnp.isfinite(root(jnp.nan))))
        c,b,s=tiny(); c=replace(c,relative_loss_norm='l2')
        p=initialize(c)
        bad={**b,'targets':np.zeros_like(b['targets'])}
        self.assertFalse(np.isfinite(float(loss_factory(c,s)(p,bad))))
        with self.assertRaises(ValueError):loss_factory(replace(c,relative_loss_norm='unknown'),s)

    def test_relative_l2_exact_resume(self):
        c,b,s=tiny();c=replace(c,detach_preparation=True,relative_loss_norm='l2')
        p=initialize(c);opt=base._adam_init(p);step=update_factory(c,s)
        p,opt,*rest=step(p,opt,b,.001)
        self.assertTrue(bool(rest[-1]))
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'state.npz'
            save_snapshot(file,p,opt,dict(protocol_sha256='l2'),[])
            pr,orr,_,_=load_snapshot(file,'l2')
            p2,o2,*_=step(p,opt,b,.001);pr2,or2,*_=step(pr,orr,b,.001)
            self.assertEqual(tree_hash(p2),tree_hash(pr2))
            for part in ('m','v'):self.assertEqual(tree_hash(o2[part]),tree_hash(or2[part]))

    def test_five_epoch_manifest_and_sweep(self):
        m=build_five_epoch_manifest(original())
        self.assertEqual((m['train_count'],m['development_count']),(54,21))
        self.assertEqual(len(list(epoch_batches(m,1))),592)
        rows=np.concatenate(list(epoch_batches(m,1)))
        self.assertEqual(len(rows),54*526)
        self.assertEqual(len(set(map(tuple,rows))),len(rows))
        self.assertEqual({m['cases'][i]['fundamental'] for i,a in rows},{.3,.4,.5})
        self.assertEqual({c['fundamental'] for c in m['cases'] if c['panel']=='unseen_domain'},{.35,.45})
        self.assertEqual(m,build_five_epoch_manifest(original()))

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

    def test_continuous_evaluation_chunk_parity(self):
        from model.train.wavenumber_evaluation import rollout_factory,autonomous_case
        c,b,s=tiny();p=initialize(c)
        case=dict(case_id='small',domain_length=2*np.pi/.3,epsilon=.01)
        a,z=autonomous_case(p,case,b['initial'][0],c,rollout_factory(c,s,2),horizon=.1,chunk_steps=2)
        full,_=autonomous_case(p,case,b['initial'][0],c,rollout_factory(c,s,4),horizon=.1,chunk_steps=4)
        np.testing.assert_allclose(a,full,rtol=1e-6,atol=1e-8)
        self.assertEqual(a.shape,(5,3,16))

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
