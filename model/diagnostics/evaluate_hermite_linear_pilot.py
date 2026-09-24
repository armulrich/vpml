"""Full nonlinear CNAB2 autonomous evaluation for the linear Hermite pilot.

Only Fourier electric fields are retained, avoiding full Hermite trajectory storage.
Physical fields are normalized by their own FFT grid size before comparisons.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import jax
import jax.numpy as jnp
from model.train import interface_flux_rollout as tr
from model.train.interface_flux_data import evaluate_manifest_case
from model.diagnostics.hermite_linear_pilot import digest, write_json
from vpml.core import FourierHermiteIMEX, learned_boundary_flux_hat


def make_rollout(template, *, closure=True, nonlinear=True, nsteps=12000, dt=.01, dealias=False):
    integ=FourierHermiteIMEX(Nx=256,Nv=64,Lx=4*math.pi,dt=dt,vth=1.,
                            dealias_23=dealias,closure=None)
    explicit=(tr._nonlinear_explicit_n_hat_for_state if nonlinear
              else tr._linear_explicit_n_hat_for_state)
    @jax.jit
    def rollout(params, perturbation):
        learned=replace(template,params=params)
        a=integ.apply_mask_hat(jnp.zeros((64,129),dtype=jnp.complex128).at[0].set(jnp.fft.rfft(perturbation)))
        def rhs(a):return explicit(a,integ=integ,poisson_sign=1.)
        def boundary(a):
            return (learned_boundary_flux_hat(a,integ.k_arr,64,1.,learned)
                    if closure else jnp.zeros_like(a))
        def field(a):
            # E_k = i delta-n_k/k, with zero mean electric field.
            return jnp.zeros(129,dtype=jnp.complex128).at[1:].set(1j*a[0,1:]/integ.k_arr[1:])
        def step(carry,_):
            a,prev_n,prev_b=carry
            n,b=rhs(a),boundary(a)
            nxt=integ.step_cnab2(a,n,prev_n,extra_hat=b,extra_hat_prev=prev_b)
            return (nxt,n,b),field(nxt)
        _,fields=jax.lax.scan(step,(a,rhs(a),boundary(a)),None,length=nsteps)
        return jnp.concatenate([field(a)[None],fields],axis=0)
    return rollout


def energy(f):
    # irfft discards an imaginary Nyquist component for a real physical field.
    return 4*math.pi*(np.sum(abs(f[:,1:-1])**2,axis=1)+.5*f[:,-1].real**2)


def metrics(times, f, r, e, er):
    result=[]
    for lo,hi in [(0,20),(20,60),(60,120),(0,120)]:
        ix=(times>=lo)&(times<=hi)
        def integral(v):return np.trapezoid(v[ix],times[ix])
        for band,stop in [("modes1_4",5),("all_common_modes",min(f.shape[1],r.shape[1]))]:
            ff,rr=f[:,1:stop],r[:,1:stop]
            finite=bool(np.all(np.isfinite(ff[ix])) and np.all(np.isfinite(e[ix])))
            if not finite:
                result.append(dict(start=lo,end=hi,band=band,finite=False));continue
            den=integral(np.sum(abs(rr)**2,axis=1))
            phase=np.angle(ff*np.conj(rr))
            mask=(abs(rr)>1e-6*np.max(abs(rr),axis=0))&(abs(ff)>1e-6*np.max(abs(rr),axis=0))
            w=(abs(rr)**2*mask)[ix]
            row=dict(start=lo,end=hi,band=band,finite=True,
                epsilon_E=float(np.sqrt(integral(np.sum(abs(ff-rr)**2,axis=1))/den)),
                amplitude_error=float(np.sqrt(integral(np.sum((abs(ff)-abs(rr))**2,axis=1))/den)),
                phase_rmse=float(np.sqrt(np.sum(w*phase[ix]**2)/max(w.sum(),1e-300))),
                log_energy_rmse=float(np.sqrt(np.mean((np.log10(np.maximum(e[ix],1e-30))-
                                                       np.log10(np.maximum(er[ix],1e-30)))**2))))
            if not all(np.isfinite(row[key]) for key in ("epsilon_E","amplitude_error","phase_rmse","log_energy_rmse")):
                row=dict(start=lo,end=hi,band=band,finite=False,reason="metric overflow",trajectory_finite=True)
            result.append(row)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--epochs",default="0,1,2,5")
    p.add_argument("--wait-checkpoints",action="store_true")
    p.add_argument("--eval-dt",type=float,default=.01)
    p.add_argument("--dealias",action="store_true")
    p.add_argument("--name",default="evaluations")
    p.add_argument("--linearized",action="store_true")
    p.add_argument("--skip-control",action="store_true")
    args=p.parse_args();out=args.run;config=json.loads((out/"configuration.json").read_text())
    dest=out/args.name;dest.mkdir(exist_ok=False)
    cache=Path(config["reference_cache"])
    cases=[c for c in json.loads((out/"selected_manifest.json").read_text())["cases"] if c["split"]=="heldout"]
    template=tr.load_learned_interface_closure_npz(out/"epoch000/interface_closure.npz")
    steps=round(120/args.eval_dt)
    roll=make_rollout(template,nsteps=steps,dt=args.eval_dt,dealias=args.dealias,nonlinear=not args.linearized)
    off=make_rollout(template,closure=False,nsteps=steps,dt=args.eval_dt,dealias=args.dealias,nonlinear=not args.linearized)
    times=np.arange(steps+1)*args.eval_dt
    specs=([] if args.skip_control else [("closure_disabled",None)])+[(f"E{int(e)}",out/f"epoch{int(e):03d}/interface_closure.npz") for e in args.epochs.split(",")]
    records=[];provenance=[]
    for label,path in specs:
        if path is not None and args.wait_checkpoints:
            # The RNG sidecar is written only after checkpoint and optimizer files.
            deadline=time.monotonic()+14400
            while not (path.parent/"rng_state.json").exists():
                if time.monotonic()>deadline:raise TimeoutError(f"Waiting for {path}")
                time.sleep(5)
        learned=template if path is None else tr.load_learned_interface_closure_npz(path)
        if path: provenance.append(dict(path=str(path),sha256=digest(path)))
        for case in cases:
            cid=case["case_id"];started=time.perf_counter()
            perturbation=evaluate_manifest_case(case,np.arange(256)*4*math.pi/256)
            f=np.asarray((off if path is None else roll)(learned.params,jnp.asarray(perturbation)))/256
            with np.load(cache/"snapshots"/(cid+".npz")) as data:
                rt=data["E_hat_hist_times"];rf=data["E_hat_hist"][:,:129]/1024
                r=np.stack([np.interp(times,rt,rf[:,k].real)+1j*np.interp(times,rt,rf[:,k].imag) for k in range(129)],axis=1)
                er=np.interp(times,data["times"],data["energy"])
            e=energy(f)
            np.savez_compressed(dest/f"{label}_{cid}.npz",times=times,field=f,reference_field=r,
                                energy=e,reference_energy=er)
            rows=metrics(times,f,r,e,er)
            for row in rows:row.update(model=label,case=cid)
            records.extend(rows)
            write_json(dest/"metrics.json",records)
            print(f"[eval] {label} {cid} finite={np.isfinite(f).all()} seconds={time.perf_counter()-started:.2f}",flush=True)
    write_json(dest/"provenance.json",provenance)
    write_json(dest/"definition.json",dict(Nx=256,Nv=64,dt=args.eval_dt,T=120,precision="float64",
         dynamics="linearized Vlasov-Poisson" if args.linearized else "full nonlinear Vlasov-Poisson retained Hermite equations",dealias=args.dealias,
         normalization="Fourier coefficients divided by each grid Nx",
         comparison="Same reduced grid at train/eval. Canonical linearized training RHS; full nonlinear autonomous RHS, as in historical evaluation.",
         energy="L sum positive non-Nyquist |Ehat/Nx|^2 + L/2 real Nyquist squared",
         note="Log RMSE uses display/comparison floor 1e-30; no field clipping in rollout."))


if __name__=="__main__":main()
