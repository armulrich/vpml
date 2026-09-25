"""Classify nonfinite autonomous cases at saved epochs without changing training."""
from __future__ import annotations
import json
from pathlib import Path
import modal

ROOT=Path('/root/vpml')
NAME='wavenumber_generalization_v2_e5_20260925'
OUT=Path('/mnt/new')/NAME
OLD=Path('/mnt/old')
NORMALIZATION=OLD/'runs/history_fno_random1729_full_anchor_E1000/training/epoch000_low_moment_closure.npz'
SOURCE='ad8cabb58635341993b24385b550f950f6d7398c69c7193b50ea820c4b19e88e'
app=modal.App('vpml-wavenumber-e5-robust-evaluation')
volume=modal.Volume.from_name('vpml-wavenumber-generalization-v2-20260925',create_if_missing=False,version=2)
old=modal.Volume.from_name('vpml-low-moment-burles',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda13]==0.10.0','numpy==2.4.4','matplotlib==3.10.9','scipy')
    .add_local_dir('vpml',str(ROOT/'vpml'),copy=True)
    .add_local_dir('model',str(ROOT/'model'),copy=True)
    .env({'PYTHONPATH':str(ROOT),'VPML_JAX_BACKEND':'gpu','PYTHONUNBUFFERED':'1'}))


@app.function(image=image,gpu='A100-80GB',cpu=16,memory=65536,timeout=7200,
              volumes={'/mnt/new':volume,'/mnt/old':old},max_containers=1)
def evaluate_saved(epoch:int):
    import numpy as np,jax,jax.numpy as jnp
    from dataclasses import asdict
    from model.train.modal_burles_latent_fno import _source_tree_sha256
    from model.train.wavenumber_training import (Configuration,load_stats,initialize,
        tree_hash,load_snapshot,loss_factory,batch_from_states,base)
    from model.train.wavenumber_evaluation import rollout_factory
    from model.train.interface_flux_data import sha256_json
    from vpml.metrics import (SelfGeneratedFieldErrorMetric,FieldErrorConfig,
        EarlyElectricFieldGrowthMetric,EarlyGrowthConfig)
    if _source_tree_sha256(ROOT)!=SOURCE:raise ValueError('Training source changed')
    manifest=json.loads((OUT/'manifest.json').read_text())
    config=Configuration();stats=load_stats(NORMALIZATION)
    protocol=dict(source_sha256=SOURCE,manifest_sha256=manifest['manifest_sha256'],
        config=asdict(config),normalization_sha256='3b9483428177a4c4671c9197a5d4c7225e2a52b27a1a92b90d6bf195ddfdf34e',
        init_sha256=tree_hash(initialize(config)))
    if epoch==0:params=initialize(config)
    else:
        path=OUT/'training'/f'state_{epoch*592:08d}.npz'
        params,_,record,_=load_snapshot(path,sha256_json(protocol))
        if record['accepted_updates']!=epoch*592:raise ValueError('Epoch update mismatch')
    dest=OUT/'evaluation_robust'/f'epoch_{epoch:04d}'
    dest.mkdir(parents=True,exist_ok=False)
    states={c['case_id']:np.load(OUT/'reference'/c['case_id']/'states.npy',mmap_mode='r')
            for c in manifest['cases']}
    rollout=rollout_factory(config,stats);times=np.arange(4801)*config.dt
    rows=[];curves=[]
    for c in (x for x in manifest['cases'] if x['split']=='heldout'):
        target=states[c['case_id']]
        state=jnp.asarray(target[0][None],jnp.float32)
        h=jnp.zeros((1,config.memory_steps,3,config.nx),jnp.float32)
        ch=jnp.zeros((1,config.memory_steps,config.nx),jnp.float32)
        count=jnp.asarray(0,jnp.int32);prev=jnp.zeros((1,config.nx),jnp.float32);z=None
        k=jnp.asarray(2*np.pi*np.fft.rfftfreq(config.nx,d=c['domain_length']/config.nx),jnp.float32)
        a=jnp.asarray([c['epsilon']],jnp.float32)
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
        predicted_energy=.5*c['domain_length']*np.mean(pprim[:,3]**2,axis=1)
        row=dict(case_id=c['case_id'],regime=c['regime'],panel=c['panel'],
                 fundamental=c['fundamental'],finite=failure_t is None,
                 failure_t=failure_t,epsilon_E=None,log_energy_rmse=None,
                 epsilon_grow=None,max_latent_rms=max(zn) if zn else None)
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
                                teacher_energy=teacher_energy,failure_t=failure_t if failure_t is not None else np.nan)
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
            fig.suptitle(f'Autonomous electric-field energy, E{epoch}')
            fig.savefig(dest/f'metric1_energy_{panel}_{page}.png',dpi=170);plt.close(fig)
    validation=jax.jit(loss_factory(config,stats));anchors=(0,100,200,300,400,500)
    selected=np.asarray([(i,a) for i,c in enumerate(manifest['cases']) if c['split']=='heldout'
                         for a in anchors],np.int32)
    values=[]
    for start in range(0,len(selected),42):
        b=batch_from_states(manifest,selected[start:start+42],states,config)
        values.append((float(validation(params,b)),len(b['amplitude'])))
    fixed=sum(v*n for v,n in values)/sum(n for _,n in values)
    finite=[r['epsilon_E'] for r in rows if r['finite']]
    report=dict(epoch=epoch,accepted_updates=epoch*592,cases=rows,finite_count=len(finite),
        nonfinite_count=len(rows)-len(finite),mean_epsilon_E=None if len(finite)<len(rows) else float(np.mean(finite)),
        mean_epsilon_E_finite_only=float(np.mean(finite)) if finite else None,
        fixed_validation_window_loss=fixed if np.isfinite(fixed) else None)
    with (dest/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    volume.commit()
    return {k:report[k] for k in ('epoch','accepted_updates','finite_count','nonfinite_count',
                                  'mean_epsilon_E','fixed_validation_window_loss')}


@app.local_entrypoint()
def run(epoch:int):
    print(json.dumps(evaluate_saved.remote(epoch),indent=2))
