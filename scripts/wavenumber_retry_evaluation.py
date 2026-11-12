"""Shared robust autonomous evaluation for the detached-preparation control.

Numerical evaluation body preserved from evaluate_wavenumber_e5.py.
This file is frozen into a run's setup and imports that run's original
model.train modules. Keep those archive-specific paths and numerical behavior.
"""
from pathlib import Path
import json

def evaluate_checkpoint(params, manifest, config, stats, reference_root, dest, epoch,
                        *, accepted_updates=None, validation_configs=None):
    import numpy as np, jax, jax.numpy as jnp
    from model.train.wavenumber_training import loss_factory,batch_from_states,base
    from model.train.wavenumber_evaluation import rollout_factory
    from vpml.metrics import SelfGeneratedFieldErrorMetric,FieldErrorConfig,EarlyElectricFieldGrowthMetric,EarlyGrowthConfig
    dest=Path(dest)
    if (dest/'report.json').exists():
        return json.loads((dest/'report.json').read_text())
    if dest.exists():
        raise FileExistsError(f'Preserving partial evaluation at {dest}')
    dest.mkdir(parents=True)
    states={c['case_id']:np.load(reference_root/'reference'/c['case_id']/'states.npy',mmap_mode='r')
            for c in manifest['cases']}
    rollout=rollout_factory(config,stats);times=np.arange(4801)*config.dt
    rows=[];curves=[]
    dtype=getattr(jnp,config.compute_dtype)
    for c in (x for x in manifest['cases'] if x['split']=='heldout'):
        target=states[c['case_id']]
        state=jnp.asarray(target[0][None],dtype)
        h=jnp.zeros((1,config.memory_steps,3,config.nx),dtype)
        ch=jnp.zeros((1,config.memory_steps,config.nx),dtype)
        count=jnp.asarray(0,jnp.int32);prev=jnp.zeros((1,config.nx),dtype);z=None
        k=jnp.asarray(2*np.pi*np.fft.rfftfreq(config.nx,d=c['domain_length']/config.nx),dtype)
        a=jnp.asarray([c['epsilon']],dtype)
        parts=[np.asarray(target[0])[None]];failure_t=None;zn=[]
        for block in range(24):
            output,memory=rollout(params,state,h,ch,count,prev,z,a,k)
            array=np.asarray(output[0]);valid=np.isfinite(array).all(axis=(1,2))
            if not valid.all():
                index=int(np.where(~valid)[0][0]);failure_t=float((block*200+index+1)*config.dt)
                if index:parts.append(array[:index])
                break
            parts.append(array);h,ch,_,count,prev,z=memory;state=output[:,-1]
            if not np.isfinite(np.asarray(z)).all():
                failure_t=float((block+1)*200*config.dt);break
            zn.append(float(jnp.sqrt(jnp.mean(z*z))))
        predicted=np.concatenate(parts)
        tprim=base._primitive_numpy(target,c['domain_length'])
        teacher_energy=.5*c['domain_length']*np.mean(tprim[:,3]**2,axis=1)
        pprim=base._primitive_numpy(predicted,c['domain_length'])
        density_spectrum=np.abs(np.fft.rfft(predicted[:,0],axis=-1))/config.nx
        teacher_density_spectrum=np.abs(np.fft.rfft(target[:,0],axis=-1))/config.nx
        predicted_energy=.5*c['domain_length']*np.mean(pprim[:,3]**2,axis=1)
        row=dict(case_id=c['case_id'],regime=c['regime'],panel=c['panel'],
                 fundamental=c['fundamental'],finite=failure_t is None,
                 failure_t=failure_t,epsilon_E=None,log_energy_rmse=None,
                 epsilon_grow=None,max_latent_rms=max(zn) if zn else None)
        row['maximum_density_mode19_amplitude']=float(np.max(density_spectrum[:,19]))
        row['teacher_maximum_density_mode19_amplitude']=float(np.max(teacher_density_spectrum[:,19]))
        if failure_t is None:
            metric=SelfGeneratedFieldErrorMetric(FieldErrorConfig(final_time=120.)).evaluate_fourier(
                times,np.fft.rfft(pprim[:,3]),np.asarray(k),times,np.fft.rfft(tprim[:,3]),np.asarray(k))
            growth=EarlyElectricFieldGrowthMetric(EarlyGrowthConfig(sample_selector='local_maxima')).compare(
                times,predicted_energy,times,teacher_energy)
            row['epsilon_E']=float(metric.epsilon_E)
            row['log_energy_rmse']=float(np.sqrt(np.mean((np.log10(np.maximum(predicted_energy,1e-30))-
                np.log10(np.maximum(teacher_energy,1e-30)))**2)))
            row['epsilon_grow']=base._json_float(growth.epsilon_grow)
        rows.append(row);curves.append((c,predicted_energy,teacher_energy,failure_t))
        with (dest/f"{c['case_id']}.npz").open('xb') as f:
            np.savez_compressed(f,times=times,model_energy=predicted_energy,
                                teacher_energy=teacher_energy,failure_t=failure_t if failure_t is not None else np.nan,
                                density_spectrum=density_spectrum.astype(np.float32),
                                teacher_density_spectrum=teacher_density_spectrum.astype(np.float32))
        print(json.dumps(row),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for panel in sorted({c['panel'] for c,_,_,_ in curves}):
        subset=[r for r in curves if r[0]['panel']==panel]
        for page,start in enumerate(range(0,len(subset),12),1):
            group=subset[start:start+12]
            fig,axes=plt.subplots(len(group),1,figsize=(10,2*len(group)),sharex=True,squeeze=False,layout='constrained')
            for ax,(c,pe,te,failure_t) in zip(axes[:,0],group):
                ax.semilogy(times,te,color='#2463eb',label='kinetic reference',lw=1)
                ax.semilogy(times[:len(pe)],pe,color='#6f3cc3',label='latent FNO',lw=1)
                if failure_t is not None:ax.text(.98,.88,f'nonfinite at t={failure_t:.2f}',
                    transform=ax.transAxes,ha='right',va='top',color='#b91c1c',fontsize=7)
                ax.set_ylabel(c['case_id'],fontsize=7);ax.set_xlim(0,120);ax.grid(alpha=.2)
            axes[0,0].legend(loc='upper right',fontsize=8);axes[-1,0].set_xlabel('t')
            label=f'E{epoch}' if accepted_updates is None else f'{accepted_updates} accepted updates'
            fig.suptitle(f'Autonomous electric-field energy, {label}')
            fig.savefig(dest/f'metric1_energy_{panel}_{page}.png',dpi=170);plt.close(fig)
    anchors=(0,100,200,300,400,500)
    selected=np.asarray([(i,a) for i,c in enumerate(manifest['cases']) if c['split']=='heldout'
                         for a in anchors],np.int32)
    losses={}
    for name,vc in (validation_configs or {'training_objective':config}).items():
        validation=jax.jit(loss_factory(vc,stats));values=[]
        for start in range(0,len(selected),42):
            b=batch_from_states(manifest,selected[start:start+42],states,vc)
            values.append((float(validation(params,b)),len(b['amplitude'])))
        value=sum(v*n for v,n in values)/sum(n for _,n in values)
        losses[name]=value if np.isfinite(value) else None
    fixed=next(iter(losses.values()))
    finite=[r['epsilon_E'] for r in rows if r['finite']]
    report=dict(epoch=epoch,accepted_updates=epoch*592 if accepted_updates is None else accepted_updates,cases=rows,finite_count=len(finite),
        nonfinite_count=len(rows)-len(finite),mean_epsilon_E=None if len(finite)<len(rows) else float(np.mean(finite)),
        mean_epsilon_E_finite_only=float(np.mean(finite)) if finite else None,
        mean_log_energy_rmse=float(np.mean([r['log_energy_rmse'] for r in rows])) if len(finite)==len(rows) else None,
        validation_losses=losses,fixed_validation_window_loss=fixed)
    with (dest/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    return report
