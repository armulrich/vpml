"""Multi-domain adapter around the preserved low-moment model, loss and AdamW.

The architecture and objective live in the common modules. This module supplies
per-example physical geometry, complete exposure, and immutable resume snapshots.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from model.train import low_moment_closure as base
from model.train.interface_flux_data import sha256_json, sha256_file
from model.train.wavenumber_data import REGIMES, epoch_batches, write_new_json
from vpml.low_moment import init_burles_latent_fno_params, rollout_burles_latent_closure
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class Configuration:
    seed: int = 1729
    nx: int = 128
    dt: float = .025
    width: int = 64
    depth: int = 8
    modes: int = 32
    latent_dim: int = 6
    memory_steps: int = 50
    memory_stride: int = 4
    preparation_steps: int = 200
    horizon: int = 400
    detach_preparation: bool = False
    planned_epochs: int = 1000
    learning_rate: float = .001
    final_learning_rate: float = .00001
    weight_decay: float = .0001


def initialize(config):
    return init_burles_latent_fno_params(jax.random.PRNGKey(config.seed),
        width=config.width, spectral_modes=config.modes, memory_steps=config.memory_steps,
        depth=config.depth, latent_dim=config.latent_dim)


def loss_factory(config, stats):
    one_loss = base.make_loss_function(k_arr=np.arange(config.nx//2+1,dtype=np.float32),
        width=config.width, horizon=config.horizon, dt=config.dt,
        input_scale=stats['input_scale'], regime_scales=stats['regime_scales'],
        heat_flux_gradient_scale=float(stats['heat_flux_gradient_scale'][0]),
        amplitude_center=float(stats['amplitude_center'][0]),
        amplitude_scale=float(stats['amplitude_scale'][0]), poisson_sign=1.,
        normalized_heat_flux_bound=128., density_floor=1e-4,pressure_floor=1e-4,
        global_relative_trajectory_loss=True, closure_history_input=True,
        memory_backend='burles_latent_fno', memory_stride=config.memory_stride,
        input_scaling='current_density_rms_arcsinh', dynamic_amplitude_floor=1e-6,
        allow_uniform_heating=True, autonomous_burnin_steps=config.preparation_steps,
        detach_autonomous_burnin=config.detach_preparation)
    def sample(params, row):
        # The core solver sees one geometry and a batch dimension of one.
        b={key:(value if key=='k_arr' else value[None]) for key,value in row.items()}
        return one_loss(params,b)[0]
    def loss(params,batch):
        values=jax.vmap(sample,in_axes=(None,0))(params,batch)
        return jnp.mean(values)
    return loss


def load_stats(checkpoint):
    # Only normalization is inherited. Neural weights always use initialize().
    with np.load(checkpoint,allow_pickle=False) as z:
        return {k[5:]:np.asarray(z[k]) for k in z.files if k.startswith('stat_')}


def batch_from_states(manifest, rows, states, config):
    initials=[]; targets=[]; amplitudes=[]; regimes=[]; wavenumbers=[]
    for ci, ai in rows:
        case=manifest['cases'][int(ci)]
        history=states[case['case_id']]
        # References are sampled at solver dt; anchors are exactly 0.2 apart.
        s=int(round(int(ai)*.2/config.dt))
        stop=s+config.preparation_steps+config.horizon+1
        if stop>len(history): raise ValueError('Reference does not cover scored window')
        initials.append(history[s])
        targets.append(history[s+config.preparation_steps+1:stop])
        amplitudes.append(case['epsilon']); regimes.append(REGIMES.index(case['regime']))
        wavenumbers.append(2*np.pi*np.fft.rfftfreq(config.nx,d=case['domain_length']/config.nx))
    b=len(rows)
    return dict(initial=np.asarray(initials,np.float32), targets=np.asarray(targets,np.float32),
        memory=np.zeros((b,config.memory_steps,3,config.nx),np.float32),
        heat_flux_gradient_history=np.zeros((b,config.memory_steps,config.nx),np.float32),
        amplitude=np.asarray(amplitudes,np.float32),regime_index=np.asarray(regimes,np.int32),
        k_arr=np.asarray(wavenumbers,np.float32))


def convert_old_case(cache, case, config):
    coeff=np.load(Path(cache)/'cases'/f"{case['case_id']}.npy",mmap_mode='r')
    # Use the original reference interpolation, including half reference steps.
    source_nx=2*(coeff.shape[-1]-1)
    result=[]
    for s in range(0,4801,128):
        indices=np.arange(s,min(s+128,4801))*config.dt/.01
        lo=np.floor(indices).astype(int); hi=np.minimum(lo+1,len(coeff)-1)
        fraction=(indices-lo).astype(np.float32)[:,None,None]
        part=(1-fraction)*coeff[lo,:3]+fraction*coeff[hi,:3]
        result.append(base.low_hermite_coefficients_to_conservative(part,source_nx=source_nx,target_nx=config.nx))
    return np.concatenate(result)


def tree_hash(tree):
    import hashlib
    h=hashlib.sha256()
    for key,value in sorted(tree.items()):
        a=np.asarray(value); h.update(key.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def save_snapshot(path,params,optimizer,record,loss_rows):
    """Create immutable snapshot with optimizer, exposure cursor and loss prefix."""
    payload={f'param_{k}':np.asarray(v) for k,v in params.items()}
    for part in ('m','v'):
        payload.update({f'optimizer_{part}_{k}':np.asarray(v) for k,v in optimizer[part].items()})
    payload['optimizer_step']=np.asarray(optimizer['step'])
    payload['record']=np.asarray(json.dumps(record,sort_keys=True))
    payload['loss_rows']=np.asarray(loss_rows,dtype=np.float64).reshape(-1,5)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Write-exclusive: a duplicate snapshot is an error, never an overwrite.
    with path.open('xb') as f: np.savez(f,**payload)
    return sha256_file(path)


def load_snapshot(path, expected_protocol):
    with np.load(path,allow_pickle=False) as z:
        record=json.loads(str(z['record']))
        if record['protocol_sha256']!=expected_protocol: raise ValueError('Resume protocol mismatch')
        params={k[6:]:jnp.asarray(z[k]) for k in z.files if k.startswith('param_')}
        optimizer={part:{k[len('optimizer_'+part+'_'):]:jnp.asarray(z[k]) for k in z.files if k.startswith('optimizer_'+part+'_')} for part in ('m','v')}
        optimizer['step']=jnp.asarray(z['optimizer_step'])
        rows=z['loss_rows'].tolist()
    return params,optimizer,record,rows


def update_factory(config,stats):
    loss=loss_factory(config,stats)
    @jax.jit
    def update(params,opt,batch,rate):
        value,grad=jax.value_and_grad(loss)(params,batch)
        p,o,gn,un,_=base._adam_step(params,grad,opt,jnp.asarray(rate,jnp.float32),0.,config.weight_decay,0.)
        p=jax.tree_util.tree_map(lambda new,old:new.astype(old.dtype),p,params)
        for part in ("m","v"):
            o[part]=jax.tree_util.tree_map(lambda new,old:new.astype(old.dtype),o[part],opt[part])
        finite=jnp.isfinite(value)&jnp.isfinite(gn)&jnp.isfinite(un)
        finite=finite & jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves((p,o))]))
        init_norm=jnp.sqrt(sum(jnp.sum(g*g) for k,g in grad.items() if k.startswith('compact_latent_init_')))
        return p,o,value,gn,un,init_norm,finite
    return update


def train_segment(manifest,states,stats,config,outdir,protocol,*,resume=None,max_updates=50,stop_epoch=20):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    digest=sha256_json(protocol)
    params=initialize(config);opt=base._adam_init(params)
    record=dict(protocol_sha256=digest,epoch=1,next_batch=0,accepted_updates=0,windows=0)
    rows=[]
    if resume: params,opt,record,rows=load_snapshot(resume,digest)
    elif list(out.glob('state_*.npz')): raise FileExistsError('Fresh run cannot reuse checkpoint directory')
    step=update_factory(config,stats); started=time.monotonic();count=0
    while record['epoch']<=stop_epoch and count<max_updates:
        batches=list(epoch_batches(manifest,record['epoch'],config.seed))
        for bi in range(record['next_batch'],len(batches)):
            b=batch_from_states(manifest,batches[bi],states,config)
            progress=(record['epoch']-1)+bi/len(batches)
            lr=base._cosine_learning_rate(record["accepted_updates"],len(batches)*config.planned_epochs,config.learning_rate,config.final_learning_rate)
            p,o,l,gn,un,ig,finite=step(params,opt,b,lr)
            if not bool(finite): raise FloatingPointError(f'Nonfinite attempted update at {record}')
            params,opt=p,o;count+=1
            record['accepted_updates']+=1;record['windows']+=len(batches[bi]);record['next_batch']=bi+1
            rows.append([float(record['accepted_updates']),float(l),float(gn),float(un),float(ig)])
            print(json.dumps(dict(event='update',epoch=record['epoch'],batch=bi+1,updates=record['accepted_updates'],loss=float(l),initializer_grad=float(ig),elapsed=time.monotonic()-started)),flush=True)
            if record['next_batch']==len(batches):record['epoch']+=1;record['next_batch']=0
            if count%50==0 or count==max_updates or record['next_batch']==0:
                save_snapshot(out/f"state_{record['accepted_updates']:08d}.npz",params,opt,record,rows)
            if count>=max_updates or record['next_batch']==0:break
    return record
