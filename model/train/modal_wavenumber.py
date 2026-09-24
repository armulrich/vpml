"""Bounded pilot for broader-wavelength training. All writes use a new volume."""
from __future__ import annotations
import json
from pathlib import Path
import modal

NAME='wavenumber_generalization_20260924_v1'
ROOT=Path('/root/vpml')
OUT=Path('/mnt/new')/NAME
OLD=Path('/mnt/old')
CACHE=OLD/'reference/e376aa1efa28e754b5f6-low-moment-nx128'
NORMALIZATION=OLD/'runs/history_fno_random1729_full_anchor_E1000/training/epoch000_low_moment_closure.npz'
NORMALIZATION_SHA='3b9483428177a4c4671c9197a5d4c7225e2a52b27a1a92b90d6bf195ddfdf34e'
RATE=3.764448
app=modal.App('vpml-wavenumber-generalization-20260924')
volume=modal.Volume.from_name('vpml-wavenumber-generalization-20260924',create_if_missing=True,version=2)
old=modal.Volume.from_name('vpml-low-moment-burles',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda13]==0.10.0','numpy==2.4.4','matplotlib==3.10.9','scipy')
    .add_local_dir('vpml',str(ROOT/'vpml'),copy=True)
    .add_local_dir('model',str(ROOT/'model'),copy=True)
    .env({'PYTHONPATH':str(ROOT),'VPML_JAX_BACKEND':'gpu','PYTHONUNBUFFERED':'1'}))

@app.function(image=image,gpu='A100-80GB',cpu=16,memory=65536,timeout=7200,
    volumes={'/mnt/new':volume,'/mnt/old':old},max_containers=1,retries=0)
def pilot(source_sha: str):
    import time
    from dataclasses import asdict,replace
    import numpy as np
    import jax
    from model.train.interface_flux_data import sha256_file,sha256_json
    from model.train.wavenumber_data import build_manifest,write_new_json,REGIMES
    from model.train.wavenumber_training import (Configuration,initialize,load_stats,convert_old_case,
        batch_from_states,loss_factory,update_factory,tree_hash,base,save_snapshot,load_snapshot)
    from model.train.wavenumber_reference import benchmark_reference
    from model.train.modal_burles_latent_fno import _source_tree_sha256
    if _source_tree_sha256(ROOT)!=source_sha:raise ValueError('Uploaded source hash mismatch')
    if sha256_file(NORMALIZATION)!=NORMALIZATION_SHA:raise ValueError('Normalization mismatch')
    if jax.default_backend()!='gpu':raise RuntimeError('Pilot requires GPU')
    pilot_dir=OUT/'pilot';pilot_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    manifest=build_manifest(json.loads((CACHE/'ic_manifest.json').read_text()))
    write_new_json(pilot_dir/'manifest.json',manifest)
    config=Configuration();stats=load_stats(NORMALIZATION);params=initialize(config)
    report=dict(source_sha256=source_sha,config=asdict(config),initial_parameter_sha256=tree_hash(params),
        manifest_sha256=manifest['manifest_sha256'],status='started',devices=[str(d) for d in jax.devices()])
    write_new_json(pilot_dir/'started.json',report);volume.commit()
    # Use real preserved training windows to test the gradient path and throughput.
    selected=[next(i for i,c in enumerate(manifest['cases']) if c['provenance']=='original' and c['split']=='train' and c['regime']==r) for r in REGIMES]
    states={manifest['cases'][i]['case_id']:convert_old_case(CACHE,manifest['cases'][i],config) for i in selected}
    rows=np.array([(i,j*20) for i in selected for j in range(16)],np.int32)
    batch=batch_from_states(manifest,rows,states,config)
    opt=base._adam_init(params); update=update_factory(config,stats)
    t=time.monotonic();p1,o1,l,gn,un,ig,finite=update(params,opt,batch,.001);jax.block_until_ready(l)
    report['compile_and_first_update_seconds']=time.monotonic()-t
    if not bool(finite) or float(ig)<=0:raise RuntimeError('Finite update/initializer-gradient gate failed')
    report.update(loss_before=float(l),grad_norm=float(gn),update_norm=float(un),initializer_gradient=float(ig))
    frozen=jax.jit(loss_factory(config,stats));report['loss_after']=float(frozen(p1,batch))
    timings=[]
    for _ in range(3):
        t=time.monotonic();_,_,v,*_=update(params,opt,batch,.001);jax.block_until_ready(v);timings.append(time.monotonic()-t)
    report['update_seconds']=float(np.median(timings))
    # Stop-gradient should affect derivatives only, not the objective value.
    detached=jax.jit(loss_factory(replace(config,detach_preparation=True),stats))
    report['detached_loss']=float(detached(params,batch))
    if not np.isclose(report['detached_loss'],float(l),rtol=2e-6,atol=1e-10):raise RuntimeError('Forward parity failed')
    protocol='pilot';record=dict(protocol_sha256=protocol,epoch=1,next_batch=1)
    save_snapshot(pilot_dir/'state_00000001.npz',p1,o1,record,[[1,float(l),float(gn),float(un),float(ig)]])
    pr,orr,_,_=load_snapshot(pilot_dir/'state_00000001.npz',protocol)
    p2,o2,*_=update(p1,o1,batch,.001);pr2,or2,*_=update(pr,orr,batch,.001)
    report['resume_bitwise']=tree_hash(p2)==tree_hash(pr2) and all(tree_hash(o2[k])==tree_hash(or2[k]) for k in ('m','v'))
    if not report['resume_bitwise']:raise RuntimeError('Resume parity failed')
    report['memory_stats']=jax.devices()[0].memory_stats()
    write_new_json(pilot_dir/'training_preflight.json',report);volume.commit()
    # Reference timing extrapolates the unchanged full-resolution kinetic solver.
    case=next(c for c in manifest['cases'] if c['provenance']=='expanded' and c['regime']=='nonlinear_landau_strong')
    report['reference_benchmark']=benchmark_reference(case)
    report['training_E20_hours']=20*2302*report['update_seconds']/3600
    report['reference_hours']=213*12000*report['reference_benchmark']['seconds_per_step']/3600
    report['pilot_hours']=(time.monotonic()-started)/3600
    # Reserve 20 percent plus $50 for compilation, convergence, evaluation and storage.
    report['projected_dollars_excluding_reserve']=RATE*(report['training_E20_hours']+report['reference_hours']+report['pilot_hours'])
    report['projected_dollars_with_reserve']=1.2*report['projected_dollars_excluding_reserve']+50
    report['within_budget']=report['projected_dollars_with_reserve']<=750 and RATE*report['pilot_hours']<=25
    report['status']='pilot_complete'
    write_new_json(pilot_dir/'report.json',report);volume.commit()
    return report

@app.local_entrypoint()
def run_pilot(source_sha: str):
    print(json.dumps(pilot.remote(source_sha),indent=2,default=str))
