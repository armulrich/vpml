"""Measure saved-cadence causal routing on autonomous training trajectories."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from model.eval_coupled_low_moment_latent import _load_saved_model
from model.train.coupled_run_support import RunRecord, atomic_json, make_model_functions
from vpml.kinetic_latent import _bounce_phase_features, resolved_hermite_to_low_moment_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    checkpoint=args.run/f'epoch{args.epoch:03d}_coupled_low_moment_latent.npz'
    _,config,params,arrays=_load_saved_model(args.run,checkpoint.name)
    origin=args.run/'initial_coupled_low_moment_latent.npz'
    with np.load(origin) as data:
        initial={k:jnp.asarray(data[k]) for k in config['trainable_parameter_keys'] if k.startswith('specialist_')}
    teacher=json.loads((Path(config['reference_cache'])/'metadata.json').read_text())['configuration']
    _,rollout,_=make_model_functions(config,arrays,teacher)
    cases=json.loads((args.run/'training_cases.json').read_text())['cases']
    batches=list(zip(*[sorted(c['case_id'] for c in cases if c['regime']==r) for r in ('linear_landau','nonlinear_landau_strong','nonlinear_landau_weak')]))
    assert len(batches)*3==len(cases)
    k=jnp.asarray(arrays['k_arr']);nx=config['nx'];dt=config['cadence']
    assert np.isclose(dt,config['cadence'])
    def gates(p,features,initial_frequency):
        branches=[]
        for prefix in ('specialist_router','specialist_high_router'):
            logits=features@p[prefix+'_bounce_gate_gain'].T+p[prefix+'_gate_bias']
            null=jnp.broadcast_to(p[prefix+'_null_gate_bias'],logits.shape[:-1]+(1,))
            branches.append(jax.nn.softmax(jnp.concatenate((null,logits),axis=-1),axis=-1))
        split=p['specialist_router_amplitude_split']
        mix=jax.nn.sigmoid((jnp.max(initial_frequency,axis=-1)-split[0])/jnp.maximum(split[1],1e-6))
        return branches[0]+mix[...,None]*(branches[1]-branches[0]),mix
    @jax.jit
    def calculate(target):
        resolved,_=rollout(params,target[:,0,:3],target[:,0,3:])
        resolved=jnp.concatenate((target[:,:1,:3],resolved),axis=1)
        states=resolved_hermite_to_low_moment_state(resolved.reshape(-1,3,nx)).reshape(resolved.shape)
        density=jnp.fft.rfft(states[:,:,0],axis=-1,norm='forward')
        frequency=jnp.sqrt(jnp.maximum(2*jnp.abs(density[:,:,1:5]),0))
        phase=jnp.concatenate((jnp.zeros_like(frequency[:,:1]),jnp.cumsum(dt*frequency[:,:-1],axis=1)),axis=1)
        initial_frequency=jnp.broadcast_to(frequency[:,:1],frequency.shape)
        exposure=jnp.concatenate((phase,initial_frequency),axis=-1)
        features=_bounce_phase_features(states.reshape(-1,3,nx),exposure.reshape(-1,8),k).reshape(states.shape[0],states.shape[1],-1)
        final,mix=gates(params,features,initial_frequency)
        original,_=gates(initial,features,initial_frequency)
        return final,original,mix,features
    record=RunRecord(args.outdir,{'scope':'Saved-cadence autonomous training-state gates; origin routing on final states is a conditional comparison, not origin rollout','epoch':args.epoch,'cases':batches},sources=[Path(__file__),checkpoint,origin,Path('vpml/kinetic_latent.py')])
    rows=[]
    try:
        for bi,names in enumerate(batches):
            target=jnp.asarray(np.stack([np.load(Path(config['projected_cache'])/'cases'/f'{n}.npy')[:config['rollout_steps']+1] for n in names]),dtype=jnp.float64)
            final,original,mix,features=map(np.asarray,jax.block_until_ready(calculate(target)))
            if not all(np.isfinite(x).all() for x in (final,original,mix,features)):
                raise ValueError('Nonfinite telemetry')
            for i,name in enumerate(names):
                g=final[i]
                rows.append({'case_id':name,'gate_mean':g.mean(axis=0).tolist(),'gate_min':g.min(axis=0).tolist(),'gate_max':g.max(axis=0).tolist(),'gate_std':g.std(axis=0).tolist(),'entropy_mean':float(np.mean(-np.sum(g*np.log(np.maximum(g,1e-300)),axis=-1))),'max_probability_above_099_fraction':float(np.mean(g.max(axis=-1)>.99)),'routing_learning_gate_rms':float(np.sqrt(np.mean((g-original[i])**2))),'high_mix':float(mix[i,0]),'feature_rms':np.sqrt(np.mean(features[i]**2,axis=0)).tolist()})
            atomic_json(args.outdir/'partial.json',rows)
            record.stage('batch_saved',batch=bi)
        atomic_json(args.outdir/'result.json',{'cases':rows,'scope':'Activation telemetry only; not loss gradients, identifiability or generalization proof'})
        record.finish()
    except BaseException as exc:
        record.finish('failed',error=repr(exc));raise


if __name__=='__main__':main()
