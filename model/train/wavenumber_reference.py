"""Stream compact low-moment targets from the common kinetic reference solver."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from vpml.physical_grid import (PhysicalGridVlasovPoissonConfig,build_physical_grid_ops,
    semilagrangian_vlasov_poisson_step,normalize_density_on_grid,gaussian_pdf)
from model.train.interface_flux_data import evaluate_manifest_case
import jax
import jax.numpy as jnp


def reference_kernel(case,*,nx=1024,nv=8192,dt=.01,steps=10):
    cfg=PhysicalGridVlasovPoissonConfig(Nx=nx,Nv=nv,Lx=case['domain_length'],vmin=-8,vmax=8,dt=dt,T=120)
    ops=build_physical_grid_ops(cfg)
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


def benchmark_reference(case,*,nx=1024,nv=8192,steps=10):
    started=time.monotonic();f,moments,advance=reference_kernel(case,nx=nx,nv=nv,steps=steps)
    f,values=advance(f);jax.block_until_ready(f)
    compile_seconds=time.monotonic()-started
    started=time.monotonic();f,values=advance(f);jax.block_until_ready(f)
    seconds=time.monotonic()-started
    if not np.isfinite(np.asarray(values)).all():raise FloatingPointError('Nonfinite kinetic reference')
    return dict(nx=nx,nv=nv,steps=steps,compile_seconds=compile_seconds,seconds=seconds,seconds_per_step=seconds/steps)


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
