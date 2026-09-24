"""Autonomous full-interval evaluation with per-case geometry and carried memory."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from model.train.wavenumber_training import base
from model.train.wavenumber_data import write_new_json
from vpml.low_moment import rollout_burles_latent_closure
from vpml.metrics import (SelfGeneratedFieldErrorMetric,FieldErrorConfig,
                         EarlyElectricFieldGrowthMetric,EarlyGrowthConfig)


def rollout_factory(config,stats,chunk_steps=200):
    def run(params,state,history,closure_history,counter,previous,z,amplitude,k):
        return rollout_burles_latent_closure(params,state,history,counter,amplitude,k,
            horizon=chunk_steps,dt=config.dt,memory_stride=config.memory_stride,
            input_scale=jnp.asarray(stats['input_scale'],jnp.float32),
            heat_flux_gradient_scale=float(stats['heat_flux_gradient_scale'][0]),
            amplitude_center=float(stats['amplitude_center'][0]),amplitude_scale=float(stats['amplitude_scale'][0]),
            compact_latent=z,previous_heat_flux_gradient=previous,heat_flux_gradient_history=closure_history,
            input_scaling='current_density_rms_arcsinh',dynamic_amplitude_floor=1e-6,
            allow_uniform_heating=True,normalized_heat_flux_bound=128.,density_floor=1e-4,pressure_floor=1e-4)
    return jax.jit(run)


def autonomous_case(params,case,initial,config,rollout,*,horizon=120.,chunk_steps=200):
    total=int(round(horizon/config.dt))
    if total%chunk_steps:raise ValueError('Evaluation horizon must fit complete chunks')
    state=jnp.asarray(initial[None],jnp.float32)
    h=jnp.zeros((1,config.memory_steps,3,config.nx),jnp.float32)
    ch=jnp.zeros((1,config.memory_steps,config.nx),jnp.float32)
    count=jnp.asarray(0,jnp.int32);prev=jnp.zeros((1,config.nx),jnp.float32);z=None
    k=jnp.asarray(2*np.pi*np.fft.rfftfreq(config.nx,d=case['domain_length']/config.nx),jnp.float32)
    a=jnp.asarray([case['epsilon']],jnp.float32)
    pieces=[np.asarray(initial)[None]];zn=[]
    for _ in range(total//chunk_steps):
        states,memory=rollout(params,state,h,ch,count,prev,z,a,k)
        h,ch,_,count,prev,z=memory;state=states[:,-1]
        arr=np.asarray(states[0])
        if not np.isfinite(arr).all() or not np.isfinite(np.asarray(z)).all():raise FloatingPointError(case['case_id'])
        pieces.append(arr);zn.append(float(jnp.sqrt(jnp.mean(z*z))))
    return np.concatenate(pieces),zn


def evaluate(params,manifest,states,stats,config,destination,case_ids=None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    rollout=rollout_factory(config,stats)
    times=np.arange(4801)*config.dt;rows=[];curves=[]
    selected=[c for c in manifest['cases'] if c['split']=='heldout' and (case_ids is None or c['case_id'] in case_ids)]
    for c in selected:
        target=states[c['case_id']]
        predicted,zn=autonomous_case(params,c,target[0],config,rollout)
        pp=base._primitive_numpy(predicted,c['domain_length']);tp=base._primitive_numpy(target,c['domain_length'])
        pe=.5*c['domain_length']*np.mean(pp[:,3]**2,axis=1)
        te=.5*c['domain_length']*np.mean(tp[:,3]**2,axis=1)
        k=2*np.pi*np.fft.rfftfreq(config.nx,d=c['domain_length']/config.nx)
        metric=SelfGeneratedFieldErrorMetric(FieldErrorConfig(final_time=120.)).evaluate_fourier(times,np.fft.rfft(pp[:,3]),k,times,np.fft.rfft(tp[:,3]),k)
        growth=EarlyElectricFieldGrowthMetric(EarlyGrowthConfig(sample_selector='local_maxima')).compare(times,pe,times,te)
        row=dict(case_id=c['case_id'],regime=c['regime'],panel=c['panel'],fundamental=c['fundamental'],
                 epsilon_E=float(metric.epsilon_E),log_energy_rmse=float(np.sqrt(np.mean((np.log10(np.maximum(pe,1e-30))-np.log10(np.maximum(te,1e-30)))**2))),
                 epsilon_grow=base._json_float(growth.epsilon_grow),finite=True,
                 minimum_density=float(np.min(1+predicted[:,0])),minimum_pressure=float(np.min(1+pp[:,2])),max_latent_rms=max(zn))
        rows.append(row);curves.append((c,pe,te))
        with (destination/f"{c['case_id']}.npz").open('xb') as f:
            np.savez_compressed(f,times=times,model_energy=pe,teacher_energy=te,model_E_hat=np.fft.rfft(pp[:,3]),teacher_E_hat=np.fft.rfft(tp[:,3]),latent_rms=zn)
        print('evaluation',row,flush=True)
    # Keep panels readable: twelve rows or fewer per page.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for panel in sorted({c['panel'] for c,_,_ in curves}):
        subset=[r for r in curves if r[0]['panel']==panel]
        for page,start in enumerate(range(0,len(subset),12),1):
            group=subset[start:start+12]
            fig,axes=plt.subplots(len(group),1,figsize=(10,2.0*len(group)),sharex=True,squeeze=False,layout='constrained')
            for ax,(c,pe,te) in zip(axes[:,0],group):
                ax.semilogy(times,te,color='#2463eb',label='kinetic reference',lw=1)
                ax.semilogy(times,pe,color='#6f3cc3',label='latent FNO',lw=1)
                ax.set_ylabel(c['case_id'],fontsize=7);ax.set_xlim(0,120);ax.grid(alpha=.2)
            axes[0,0].legend(loc='upper right',fontsize=8);axes[-1,0].set_xlabel('t')
            fig.suptitle('Autonomous electric-field energy, T=0 to 120')
            fig.savefig(destination/f'metric1_energy_{panel}_{page}.png',dpi=170);plt.close(fig)
    report=dict(cases=rows,mean_epsilon_E=float(np.mean([r['epsilon_E'] for r in rows])),mean_log_energy_rmse=float(np.mean([r['log_energy_rmse'] for r in rows])),finite=all(r['finite'] for r in rows))
    write_new_json(destination/'report.json',report)
    return report
