"""Reproducible local Gaussian-family reference screen; never changes the notes.

The cubic kinetic map is deliberately unchanged. Failed reference gates are
recorded, rather than repairing negative distributions by clipping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("VPML_JAX_BACKEND", "cpu")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.special import erfc
from vpml.physical_grid import (
    PhysicalGridVlasovPoissonConfig, build_physical_grid_ops,
    compute_electric_field_from_distribution, gaussian_pdf,
    normalize_density_on_grid, semilagrangian_vlasov_poisson_step,
)

LAMBDAS = (0.1, 0.3, 0.5, 0.7, 1.0)
BASE_SCALES = (0.18, 0.10, 0.25)
DIAGNOSTIC_NAMES = ["electric", "linear_electric", "delta_e2", "mass",
                    "total_energy", "min_f", "negative_mass_fraction",
                    "boundary_abs_fraction"]
# Screening decisions, declared before examining new trajectories. They are
# engineering tolerances for this pilot, not a physical admissibility theorem.
GATES = dict(relative_mass=1e-9, relative_energy=1e-4,
             minimum_f=-1e-8, negative_mass_fraction=1e-8,
             boundary_abs_fraction=1e-8)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def realization(seed, nx, strength, k0=.43, J=5, s=1.2):
    rng = np.random.default_rng(seed)
    coeff = rng.standard_normal((3, 2, J)) * np.arange(1, J + 1)[None, None, :] ** (-s)
    coeff /= np.sqrt(.5 * np.sum(coeff ** 2, axis=(1, 2)))[:, None, None]
    theta = 2 * np.pi * np.arange(nx) / nx
    phase = np.arange(1, J + 1)[:, None] * theta[None, :]
    shapes = coeff[:, 0] @ np.cos(phase) + coeff[:, 1] @ np.sin(phase)
    gn, gu, gt = shapes
    n = np.exp(strength * BASE_SCALES[0] * gn); n /= n.mean()
    temp = np.exp(strength * BASE_SCALES[1] * gt); temp /= temp.mean()
    u = strength * BASE_SCALES[2] * (gu - np.mean(n * gu) / n.mean())
    return n, u, temp, coeff


def protocol(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)
    p = dict(version=1, base_commit="9d5c3fa41231db66e18beadc61e7c776f5513715",
             family="local Gaussian with independent random n,u,T shapes",
             seeds=list(range(28001, 28011)), screening_seeds=[28001,28002,28003],
             lambdas=LAMBDAS, scales=BASE_SCALES, k0=.43, J=5, s=1.2, Tbar=1.,
             base_grid=dict(nx=256,nv=2048,dt=.02,vmin=-8.,vmax=8.,save_dt=.1),
             screening_T=40., training_T=120., screening_gates=GATES,
             refinement_gates=dict(relative_field_L2=.01,relative_moment_L2=.01,
                                   relative_closure_target_L2=.05),
             split_by_seed={str(s):("train" if s<=28006 else "validation" if s<=28008 else "test")
                            for s in range(28001,28011)},
             comparison="matched IC count, common feature scaling; exploratory if references fail",
             linear_initialization="same f0 minus homogeneous unit Maxwellian",
             notes_document="unchanged")
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(p)):
        raise ValueError("Output protocol differs; use a fresh output directory")
    path.write_text(json.dumps(p, indent=2)+"\n")
    return p


class Runner:
    def __init__(self, nx=256, nv=2048, dt=.02, k0=.43, vmax=8., save_dt=.1):
        self.cfg = cfg = PhysicalGridVlasovPoissonConfig(
            Nx=nx,Nv=nv,Lx=2*np.pi/k0,vmin=-vmax,vmax=vmax,dt=dt,T=40.)
        self.ops = ops = build_physical_grid_ops(cfg)
        ops["batched_v_prefilter"] = True
        self.save_dt = save_dt
        ninner = round(save_dt/dt)
        if not np.isclose(ninner*dt, save_dt):
            raise ValueError("save_dt must be an integer number of steps")
        v = cfg.v
        w = jnp.ones(nv).at[0].set(.5).at[-1].set(.5)*cfg.dv
        self.equilibrium = eq = jnp.broadcast_to(
            normalize_density_on_grid(gaussian_pdf(v,0.,1.),v)[:,None], (nv,nx))

        def full_step(f):
            return semilagrangian_vlasov_poisson_step(cfg,f,ops=ops)[0]

        def linear_step(h):
            return jax.jvp(full_step,(eq,),(h,))[1]

        def diagnostics(f,h):
            ef = compute_electric_field_from_distribution(f,cfg,ops=ops)
            el = compute_electric_field_from_distribution(h,cfg,ops=ops)
            mass = cfg.dx*jnp.sum(w[:,None]*f)
            electric = .5*cfg.dx*jnp.sum(ef**2)
            total = electric + .5*cfg.dx*jnp.sum((w*v*v)[:,None]*f)
            neg = cfg.dx*jnp.sum(w[:,None]*jnp.maximum(-f,0.))/mass
            edge = cfg.dx*jnp.sum((w*(jnp.abs(v)>=vmax-.5))[:,None]*jnp.abs(f))/mass
            stats = jnp.stack([electric,.5*cfg.dx*jnp.sum(el**2),
                               cfg.dx*jnp.sum((ef-el)**2),mass,total,jnp.min(f),neg,edge])
            return stats,ef,el

        def moments(f):
            n = jnp.sum(w[:,None]*f,axis=0)
            u = jnp.sum((w*v)[:,None]*f,axis=0)/n
            central = v[:,None]-u[None,:]
            p = jnp.sum(w[:,None]*central**2*f,axis=0)
            Q = jnp.sum(w[:,None]*central**3*f,axis=0)
            g = jnp.fft.irfft(1j*ops["k_arr"]*jnp.fft.rfft(Q),n=nx)
            return jnp.stack([n,u,p,Q,g])

        def inner(carry,_):
            f,h = carry
            f,h = full_step(f),linear_step(h)
            return (f,h),diagnostics(f,h)[0]

        def saved(carry,_):
            (f,h),stats = jax.lax.scan(inner,carry,None,length=ninner)
            _,ef,el = diagnostics(f,h)
            return (f,h),(stats,moments(f),ef,el)

        # One physical time unit per call gives regular progress and allows
        # exact continuation from an interruption checkpoint.
        block_saves = round(1./save_dt)
        self.block_time = block_saves*save_dt
        self.advance = jax.jit(lambda f,h:jax.lax.scan(saved,(f,h),None,length=block_saves))
        self.diagnostics = jax.jit(diagnostics)
        self.moments = jax.jit(moments)
        self.full_step = jax.jit(full_step)
        self.linear_step = jax.jit(linear_step)

    def execute(self, out, seed=28001, strength=1., final_time=40., family="gaussian",label="base"):
        cfg = self.cfg
        k0 = 2*np.pi/cfg.Lx
        identity = f"s{seed}_l{strength:g}_{label}" if family=="gaussian" else f"landau_{seed:02d}_{label}"
        dest = out/"cases"/f"{identity}.npz"
        meta_path = dest.with_suffix(".json")
        grid = dict(nx=cfg.Nx,nv=cfg.Nv,dt=cfg.dt,L=cfg.Lx,k0=k0,
                    vmin=cfg.vmin,vmax=cfg.vmax,final_time=final_time,save_dt=self.save_dt)
        if meta_path.exists():
            report=json.loads(meta_path.read_text())
            if report["grid"] != grid or report["result_sha256"] != sha256(dest):
                raise ValueError("Existing result identity/checksum mismatch")
            print(f"REUSE {identity}",flush=True)
            return report
        if family=="gaussian":
            n,u,temp,coeff=realization(seed,cfg.Nx,strength,k0)
        else:
            n=1+1e-3*np.cos(np.asarray(cfg.x)*k0)
            u=np.zeros(cfg.Nx); temp=np.ones(cfg.Nx); coeff=np.empty((0,))
        f0=jnp.asarray(n[None,:]/np.sqrt(2*np.pi*temp[None,:]) *
                      np.exp(-(np.asarray(cfg.v)[:,None]-u[None,:])**2/(2*temp[None,:])))
        f,h=f0,f0-self.equilibrium
        initial=self.diagnostics(f,h)
        stats=[np.asarray(initial[0])[None,:]]
        fields=[np.asarray(initial[1])[None,:]]
        lfields=[np.asarray(initial[2])[None,:]]
        moment_history=[np.asarray(self.moments(f))[None,:,:]]
        # Initial and final distribution only; moments/targets at every .1.
        started=time.monotonic()
        blocks=round(final_time/self.block_time)
        if not np.isclose(blocks*self.block_time,final_time):
            raise ValueError("final_time must divide into whole output blocks")
        checkpoint=dest.with_name(dest.stem+"_checkpoint.npz")
        first=0
        if checkpoint.exists():
            with np.load(checkpoint) as old:
                if str(old["source_sha256"])!=sha256(__file__) or str(old["grid_json"])!=json.dumps(grid,sort_keys=True):
                    raise ValueError("Checkpoint protocol/source mismatch")
                f,h=jnp.asarray(old["f"]),jnp.asarray(old["h"])
                first=int(old["blocks"])
                stats=[old["diagnostics"]]; fields=[old["E"]]; lfields=[old["linear_field"]]
                moment_history=[old["moment_history"]]
        for block in range(first,blocks):
            (f,h),outputs=self.advance(f,h)
            ds,mom,ef,el=map(np.asarray,outputs)
            if not all(np.isfinite(a).all() for a in (ds,mom,ef,el)):
                raise FloatingPointError(f"Nonfinite output in {identity} block {block}")
            stats.append(ds.reshape(-1,len(DIAGNOSTIC_NAMES)))
            fields.append(ef); lfields.append(el); moment_history.append(mom)
            if block<2 or (block+1)%10==0:
                print(f"{identity}: t={(block+1)*self.block_time:g}/{final_time:g}, "
                      f"elapsed={time.monotonic()-started:.1f}s min_f={ds[...,5].min():.3g}",flush=True)
            if (block+1)%20==0 and block+1<blocks:
                np.savez(checkpoint,f=np.asarray(f),h=np.asarray(h),blocks=block+1,
                         diagnostics=np.concatenate(stats),E=np.concatenate(fields),
                         linear_field=np.concatenate(lfields),moment_history=np.concatenate(moment_history),
                         source_sha256=sha256(__file__),grid_json=json.dumps(grid,sort_keys=True))
        ds=np.concatenate(stats); ef=np.concatenate(fields); el=np.concatenate(lfields)
        mh=np.concatenate(moment_history)
        tdiag=np.arange(len(ds))*cfg.dt
        times=np.arange(len(ef))*self.save_dt
        eta=float(np.sqrt(np.trapezoid(ds[:,2],x=tdiag)/np.trapezoid(2*ds[:,0],x=tdiag)))
        tail=float(np.mean(n*.5*(erfc((cfg.vmax-u)/np.sqrt(2*temp))+
                                  erfc((u-cfg.vmin)/np.sqrt(2*temp)))))
        report=dict(case_id=identity,family=family,seed=seed,strength=strength,k0=k0,grid=grid,
                    eta_NL=eta,initial_tail_mass_fraction=tail,
                    max_relative_mass_error=float(np.max(np.abs(ds[:,3]/ds[0,3]-1))),
                    max_relative_total_energy_error=float(np.max(np.abs(ds[:,4]/ds[0,4]-1))),
                    min_full_distribution=float(ds[:,5].min()),max_negative_mass_fraction=float(ds[:,6].max()),
                    max_boundary_mass_fraction=float(ds[:,7].max()),
                    elapsed_seconds=time.monotonic()-started,jax_version=jax.__version__,
                    device=str(jax.devices()),compute_dtype="float64",source_sha256=sha256(__file__),
                    diagnostic_names=DIAGNOSTIC_NAMES)
        failures=[]
        for name,key in [("relative_mass","max_relative_mass_error"),("relative_energy","max_relative_total_energy_error"),
                         ("negative_mass_fraction","max_negative_mass_fraction"),("boundary_abs_fraction","max_boundary_mass_fraction")]:
            if report[key]>GATES[name]: failures.append(name)
        if report["min_full_distribution"]<GATES["minimum_f"]: failures.append("minimum_f")
        report.update(qualification_pass=not failures,qualification_failures=failures,
                      qualification_scope="screen only; refinement needed before training acceptance")
        tmp=dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,times=times,n=mh[:,0],u=mh[:,1],p=mh[:,2],Q=mh[:,3],g=mh[:,4],E=ef,
                            full_energy=.5*cfg.dx*np.sum(ef**2,axis=1),linear_energy=.5*cfg.dx*np.sum(el**2,axis=1),
                            linear_field=el,diagnostic_times=tdiag,diagnostics=ds,initial_f=np.asarray(f0),
                            final_f=np.asarray(f),x=np.asarray(cfg.x),v=np.asarray(cfg.v),fourier_coefficients=coeff)
        tmp.replace(dest)
        report["result_sha256"]=sha256(dest)
        meta_path.write_text(json.dumps(report,indent=2)+"\n")
        if checkpoint.exists(): checkpoint.unlink()
        print("COMPLETE "+json.dumps(report),flush=True)
        return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",default=[28001,28002,28003])
    ap.add_argument("--strengths",type=float,nargs="+",default=LAMBDAS)
    ap.add_argument("--nx",type=int,default=256); ap.add_argument("--nv",type=int,default=2048)
    ap.add_argument("--dt",type=float,default=.02); ap.add_argument("--vmax",type=float,default=8.)
    ap.add_argument("--final-time",type=float,default=40.); ap.add_argument("--label",default="base")
    ap.add_argument("--landau-count",type=int,default=0)
    args=ap.parse_args()
    protocol(args.out)
    if args.landau_count:
        for i,k in enumerate(np.linspace(.30,.60,args.landau_count)):
            Runner(args.nx,args.nv,args.dt,float(k),args.vmax).execute(
                args.out,i+1,.001,args.final_time,"linear_landau",args.label)
    else:
        runner=Runner(args.nx,args.nv,args.dt,.43,args.vmax)
        for seed in args.seeds:
            for strength in args.strengths:
                runner.execute(args.out,seed,strength,args.final_time,label=args.label)


if __name__=="__main__":
    main()
