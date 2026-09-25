"""Stream compact low-moment targets from the common kinetic reference solver."""
from __future__ import annotations
import time
import json
import os
import hashlib
from pathlib import Path
import numpy as np
from vpml.physical_grid import (PhysicalGridVlasovPoissonConfig,build_physical_grid_ops,
    semilagrangian_vlasov_poisson_step,normalize_density_on_grid,gaussian_pdf)
from model.train.interface_flux_data import evaluate_manifest_case
import jax
import jax.numpy as jnp
from model.train.interface_flux_data import sha256_json, sha256_file


def reference_kernel(case,*,nx=1024,nv=8192,dt=.01,steps=10,batched_v_prefilter=False):
    cfg=PhysicalGridVlasovPoissonConfig(Nx=nx,Nv=nv,Lx=case['domain_length'],vmin=-8,vmax=8,dt=dt,T=120)
    ops=build_physical_grid_ops(cfg)
    if batched_v_prefilter:ops['batched_v_prefilter']=True
    fv=normalize_density_on_grid(gaussian_pdf(cfg.v,0.,1.),cfg.v)
    f0=fv[:,None]*(1+jnp.asarray(evaluate_manifest_case(case,np.asarray(cfg.x)))[None,:])
    def moments(f):
        perturb=f-fv[:,None]
        m=jnp.stack([jnp.trapezoid(perturb*cfg.v[:,None]**j,x=cfg.v,axis=0) for j in range(3)])
        mh=jnp.fft.rfft(m,axis=-1)[:,:65]*(128/nx)
        mh=mh.at[:,-1].set(jnp.real(mh[:,-1]))
        return jnp.fft.irfft(mh,n=128,axis=-1).astype(jnp.float32)
    @jax.jit
    def advance(f):
        def body(f,_):
            new,_=semilagrangian_vlasov_poisson_step(cfg,f,ops=ops)
            return new,moments(new)
        return jax.lax.scan(body,f,None,length=steps)
    return f0,moments,advance


def benchmark_reference(case,*,nx=1024,nv=8192,steps=10,batched_v_prefilter=False):
    started=time.monotonic();f,moments,advance=reference_kernel(case,nx=nx,nv=nv,steps=steps,
        batched_v_prefilter=batched_v_prefilter)
    f,values=advance(f);jax.block_until_ready(f)
    compile_seconds=time.monotonic()-started
    started=time.monotonic();f,values=advance(f);jax.block_until_ready(f)
    seconds=time.monotonic()-started
    if not np.isfinite(np.asarray(values)).all():raise FloatingPointError('Nonfinite kinetic reference')
    return dict(nx=nx,nv=nv,steps=steps,compile_seconds=compile_seconds,seconds=seconds,seconds_per_step=seconds/steps,batched_v_prefilter=batched_v_prefilter)


def generate_reference(case,destination,*,nx=1024,nv=8192,dt=.01):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=False)
    f,moments,advance=reference_kernel(case,nx=nx,nv=nv,dt=dt,steps=100)
    raw=np.lib.format.open_memmap(dest/'moments_dt001.npy',mode='w+',dtype=np.float32,shape=(12001,3,128))
    raw[0]=np.asarray(moments(f))
    for i in range(120):
        f,values=advance(f);arr=np.asarray(values)
        if not np.isfinite(arr).all():raise FloatingPointError(f'Nonfinite reference at block{i}')
        raw[1+i*100:1+(i+1)*100]=arr;raw.flush()
    # The same linear temporal interpolation used by the original trainer.
    idx=np.arange(4801)*.025/.01;lo=np.floor(idx).astype(int);hi=np.minimum(lo+1,12000)
    frac=(idx-lo).astype(np.float32)[:,None,None]
    with (dest/'states.npy').open('xb') as handle:np.save(handle,(1-frac)*raw[lo]+frac*raw[hi])
    with (dest/'kinetic_final.npy').open('xb') as handle:np.save(handle,np.asarray(f))
    return dest/'states.npy'


def _atomic_npy(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        np.save(f,np.asarray(value));f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def _atomic_json(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def generate_reference_resumable(case,destination,*,nx=1024,nv=8192,dt=.01,
                                 block_steps=100,on_block=None,max_blocks=None,
                                 batched_v_prefilter=False):
    """Advance one full kinetic case, committing a recoverable state per block.

    The callback must commit the backing persistent volume. Two kinetic states are
    retained, so the previous committed state remains available while writing the
    next one. Reference arrays and training targets are never silently replaced.
    """
    if dt != .01 or block_steps != 100:
        raise ValueError('Frozen reference protocol requires dt=.01 and 100-step blocks')
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    protocol=dict(case_sha256=sha256_json(case),nx=nx,nv=nv,dt=dt,
                  block_steps=block_steps,batched_v_prefilter=batched_v_prefilter)
    protocol_sha=sha256_json(protocol)
    meta_path=dest/'progress.json'
    f0,moments,advance=reference_kernel(case,nx=nx,nv=nv,dt=dt,
        steps=block_steps,batched_v_prefilter=batched_v_prefilter)
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['protocol_sha256']!=protocol_sha:
            raise ValueError('Reference resume protocol mismatch')
        block=int(meta['completed_blocks'])
        if block:
            checkpoint=dest/f'kinetic_{block:04d}.npy'
            if sha256_file(checkpoint)!=meta['kinetic_sha256']:
                raise ValueError('Reference kinetic checkpoint mismatch')
            f=jnp.asarray(np.load(checkpoint,mmap_mode='r'))
        else:f=f0
    else:
        if any(dest.iterdir()):raise FileExistsError('Unrecognized partial reference directory')
        block=0;f=f0
        _atomic_npy(dest/'moments_0000.npy',np.asarray(moments(f)))
        meta=dict(protocol_sha256=protocol_sha,completed_blocks=0,kinetic_sha256=None,status='running')
        _atomic_json(meta_path,meta)
        if on_block:on_block(meta)
    limit=120 if max_blocks is None else min(120,block+max_blocks)
    while block<limit:
        f,values=advance(f);arr=np.asarray(values)
        if not np.isfinite(arr).all():raise FloatingPointError(f'Nonfinite reference block {block+1}')
        next_block=block+1
        moment_path=dest/f'moments_{next_block:04d}.npy'
        _atomic_npy(moment_path,arr)
        checkpoint=dest/f'kinetic_{next_block:04d}.npy'
        _atomic_npy(checkpoint,np.asarray(f))
        meta=dict(protocol_sha256=protocol_sha,completed_blocks=next_block,
                  kinetic_sha256=sha256_file(checkpoint),status='running')
        _atomic_json(meta_path,meta)
        if on_block:on_block(meta)
        # Retain the preceding checkpoint until the new state is durably committed.
        if next_block>1:
            old=dest/f'kinetic_{next_block-2:04d}.npy'
            if old.exists():old.unlink()
        block=next_block
    if block==120 and not (dest/'states.npy').exists():
        raw=np.empty((12001,3,128),np.float32)
        raw[0]=np.load(dest/'moments_0000.npy')
        for i in range(120):raw[1+i*100:1+(i+1)*100]=np.load(dest/f'moments_{i+1:04d}.npy')
        idx=np.arange(4801)*.025/dt;lo=np.floor(idx).astype(int);hi=np.minimum(lo+1,12000)
        frac=(idx-lo).astype(np.float32)[:,None,None]
        _atomic_npy(dest/'states.npy',(1-frac)*raw[lo]+frac*raw[hi])
        final=dict(meta,status='complete',states_sha256=sha256_file(dest/'states.npy'))
        _atomic_json(meta_path,final)
        if on_block:on_block(final)
    return json.loads(meta_path.read_text())
