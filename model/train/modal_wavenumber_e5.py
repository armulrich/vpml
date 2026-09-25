"""Recoverable 54-case multi-domain latent FNO experiment on Modal.

All writes are confined to a dated v2 volume; the v1 pilot and historic
low-moment volume are mounted read-only by convention and never modified.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import modal

ROOT=Path('/root/vpml')
NAME='wavenumber_generalization_v2_e5_20260925'
VOLUME_NAME='vpml-wavenumber-generalization-v2-20260925'
OUT=Path('/mnt/new')/NAME
OLD=Path('/mnt/old')
CACHE=OLD/'reference/e376aa1efa28e754b5f6-low-moment-nx128'
NORMALIZATION=OLD/'runs/history_fno_random1729_full_anchor_E1000/training/epoch000_low_moment_closure.npz'
NORMALIZATION_SHA='3b9483428177a4c4671c9197a5d4c7225e2a52b27a1a92b90d6bf195ddfdf34e'
app=modal.App('vpml-wavenumber-generalization-v2-e5')
volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=True,version=2)
old=modal.Volume.from_name('vpml-low-moment-burles',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda13]==0.10.0','numpy==2.4.4','matplotlib==3.10.9','scipy')
    .add_local_dir('vpml',str(ROOT/'vpml'),copy=True)
    .add_local_dir('model',str(ROOT/'model'),copy=True)
    .env({'PYTHONPATH':str(ROOT),'VPML_JAX_BACKEND':'gpu','PYTHONUNBUFFERED':'1'}))


def _verify_source(source_sha):
    from model.train.modal_burles_latent_fno import _source_tree_sha256
    if _source_tree_sha256(ROOT)!=source_sha:raise ValueError('Uploaded source hash mismatch')


@app.function(image=image,cpu=2,memory=4096,timeout=600,
    volumes={'/mnt/new':volume,'/mnt/old':old},max_containers=1)
def prepare(source_sha):
    from model.train.wavenumber_data import build_five_epoch_manifest,write_new_json
    from model.train.interface_flux_data import sha256_file
    _verify_source(source_sha)
    if sha256_file(NORMALIZATION)!=NORMALIZATION_SHA:raise ValueError('Normalization mismatch')
    original=json.loads((CACHE/'ic_manifest.json').read_text())
    manifest=build_five_epoch_manifest(original)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:raise ValueError('Existing manifest differs')
    else:
        write_new_json(path,manifest)
        write_new_json(OUT/'provenance.json',dict(source_sha256=source_sha,
            original_manifest_sha256=manifest['original_manifest_sha256'],
            normalization_sha256=NORMALIZATION_SHA,manifest_sha256=manifest['manifest_sha256']))
        volume.commit()
    return dict(manifest_sha256=manifest['manifest_sha256'],cases=len(manifest['cases']),
                train=manifest['train_count'],heldout=manifest['development_count'])


@app.function(image=image,gpu='A100-80GB',cpu=16,memory=65536,timeout=2400,
    volumes={'/mnt/new':volume},max_containers=1)
def profile(source_sha):
    import jax,numpy as np
    from model.train.wavenumber_reference import reference_kernel
    from model.train.wavenumber_data import write_new_json
    _verify_source(source_sha)
    manifest=json.loads((OUT/'manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['split']=='train' and c['regime']=='nonlinear_landau_strong')
    timings={};outputs={}
    for fast in (False,True):
        f,_,advance=reference_kernel(case,steps=100,batched_v_prefilter=fast)
        started=time.monotonic();f,values=advance(f);jax.block_until_ready(f)
        compile_time=time.monotonic()-started
        started=time.monotonic();f,values=advance(f);jax.block_until_ready(f)
        timings[str(fast)]=dict(compile_seconds=compile_time,seconds_per_100=time.monotonic()-started)
        outputs[str(fast)]=(np.asarray(f),np.asarray(values))
    maximum_f=float(np.max(np.abs(outputs['False'][0]-outputs['True'][0])))
    maximum_m=float(np.max(np.abs(outputs['False'][1]-outputs['True'][1])))
    report=dict(timings=timings,max_distribution_difference=maximum_f,
        max_moment_difference=maximum_m,devices=[str(x) for x in jax.devices()])
    path=OUT/'reference_profile.json'
    write_new_json(path,report);volume.commit()
    return report


@app.function(image=image,gpu='A100-80GB',cpu=16,memory=65536,timeout=7200,
    volumes={'/mnt/new':volume},max_containers=2,retries=0)
def reference_case(source_sha,case_id,fast):
    import jax
    from model.train.wavenumber_reference import generate_reference_resumable
    _verify_source(source_sha)
    if jax.default_backend()!='gpu':raise RuntimeError('GPU reference required')
    manifest=json.loads((OUT/'manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['case_id']==case_id)
    dest=OUT/'reference'/case_id
    def commit(progress):
        volume.commit()
        print(json.dumps(dict(event='reference_block',case_id=case_id,
                              completed_blocks=progress['completed_blocks'])),flush=True)
    result=generate_reference_resumable(case,dest,on_block=commit,
        batched_v_prefilter=fast)
    return dict(case_id=case_id,status=result['status'],blocks=result['completed_blocks'],
                states_sha256=result.get('states_sha256'))


@app.function(image=image,gpu='A100-80GB',cpu=16,memory=65536,timeout=7200,
    volumes={'/mnt/new':volume,'/mnt/old':old},max_containers=1,retries=0)
def train_chunk(source_sha,stop_epoch,max_updates=50):
    import numpy as np,jax
    from dataclasses import asdict
    from model.train.wavenumber_training import (Configuration,load_stats,
        train_segment,tree_hash,initialize,load_snapshot)
    from model.train.interface_flux_data import sha256_json
    _verify_source(source_sha)
    if jax.default_backend()!='gpu':raise RuntimeError('GPU training required')
    manifest=json.loads((OUT/'manifest.json').read_text())
    states={}
    for c in manifest['cases']:
        path=OUT/'reference'/c['case_id']/'states.npy'
        if not path.exists():raise FileNotFoundError(path)
        states[c['case_id']]=np.load(path,mmap_mode='r')
    stats=load_stats(NORMALIZATION);config=Configuration()
    protocol=dict(source_sha256=source_sha,manifest_sha256=manifest['manifest_sha256'],
        config=asdict(config),normalization_sha256=NORMALIZATION_SHA,
        init_sha256=tree_hash(initialize(config)))
    out=OUT/'training';out.mkdir(exist_ok=True)
    prior=sorted(out.glob('state_*.npz'))
    resume=prior[-1] if prior else None
    result=train_segment(manifest,states,stats,config,out,protocol,resume=resume,
                         max_updates=max_updates,stop_epoch=stop_epoch)
    volume.commit()
    return result


@app.function(image=image,gpu='A100-80GB',cpu=16,memory=65536,timeout=7200,
    volumes={'/mnt/new':volume,'/mnt/old':old},max_containers=1,retries=0)
def evaluate_epoch(source_sha,epoch):
    import numpy as np,jax
    from dataclasses import asdict
    from model.train.wavenumber_training import (Configuration,load_stats,initialize,
        tree_hash,load_snapshot,loss_factory,batch_from_states)
    from model.train.wavenumber_evaluation import evaluate
    from model.train.interface_flux_data import sha256_json
    _verify_source(source_sha)
    manifest=json.loads((OUT/'manifest.json').read_text())
    config=Configuration();stats=load_stats(NORMALIZATION)
    states={c['case_id']:np.load(OUT/'reference'/c['case_id']/'states.npy',mmap_mode='r')
            for c in manifest['cases']}
    protocol=dict(source_sha256=source_sha,manifest_sha256=manifest['manifest_sha256'],
        config=asdict(config),normalization_sha256=NORMALIZATION_SHA,
        init_sha256=tree_hash(initialize(config)))
    if epoch==0:params=initialize(config)
    else:
        candidates=sorted((OUT/'training').glob('state_*.npz'))
        wanted=epoch*592
        match=next((p for p in candidates if p.stem==f'state_{wanted:08d}'),None)
        if match is None:raise FileNotFoundError(f'No snapshot for epoch {epoch}, update {wanted}')
        params,_,_,_=load_snapshot(match,sha256_json(protocol))
    destination=OUT/'evaluation'/f'epoch_{epoch:04d}'
    report=evaluate(params,manifest,states,stats,config,destination)
    loss=jax.jit(loss_factory(config,stats))
    anchors=(0,100,200,300,400,500)
    rows=np.asarray([(i,a) for i,c in enumerate(manifest['cases']) if c['split']=='heldout'
                     for a in anchors],np.int32)
    values=[]
    for start in range(0,len(rows),42):
        batch=batch_from_states(manifest,rows[start:start+42],states,config)
        values.append((float(loss(params,batch)),len(batch['amplitude'])))
    fixed_validation=sum(v*n for v,n in values)/sum(n for _,n in values)
    report['fixed_validation_window_loss']=fixed_validation
    (destination/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    volume.commit()
    return dict(epoch=epoch,mean_epsilon_E=report['mean_epsilon_E'],
                fixed_validation_window_loss=fixed_validation)


@app.local_entrypoint()
def inspect_profile(source_sha: str):
    print(json.dumps(prepare.remote(source_sha),indent=2))
    print(json.dumps(profile.remote(source_sha),indent=2))


def _mirror(relative,local_root):
    import subprocess
    path=Path(local_root)/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return path
    remote=f'{NAME}/{relative}'
    subprocess.run(['modal','volume','get',VOLUME_NAME,remote,str(path)],check=True)
    return path


def _local_sha256(path):
    import hashlib
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


@app.local_entrypoint()
def run_references(source_sha: str,local_root: str):
    info=prepare.remote(source_sha)
    _mirror('manifest.json',local_root)
    _mirror('provenance.json',local_root)
    _mirror('reference_profile.json',local_root)
    profile_result=json.loads((Path(local_root)/'reference_profile.json').read_text())
    speed=profile_result['timings']['True']['seconds_per_100']
    old_speed=profile_result['timings']['False']['seconds_per_100']
    if not (speed<old_speed and profile_result['max_distribution_difference']<1e-10
            and profile_result['max_moment_difference']<1e-7):
        raise RuntimeError('Fast reference numerical or performance gate failed')
    manifest=json.loads((Path(local_root)/'manifest.json').read_text())
    cases=manifest['cases']
    # At most two A100 reference workers. Already completed cases are verified
    # against their saved SHA before being skipped on any resumed invocation.
    incomplete=[]
    for c in cases:
        case_id=c['case_id'];relative=f'reference/{case_id}/states.npy'
        path=Path(local_root)/relative
        if path.exists():
            progress=json.loads((Path(local_root)/f'reference/{case_id}/progress.json').read_text())
            if _local_sha256(path)!=progress['states_sha256']:raise ValueError(f'Local reference hash mismatch: {case_id}')
        else:incomplete.append(case_id)
    print(json.dumps(dict(event='reference_start',completed=len(cases)-len(incomplete),
                          remaining=len(incomplete),estimated_gpu_hours=len(incomplete)*120*speed/3600)),flush=True)
    for result in reference_case.map([source_sha]*len(incomplete),incomplete,[True]*len(incomplete),
                                     order_outputs=False):
        case_id=result['case_id']
        local_state=_mirror(f'reference/{case_id}/states.npy',local_root)
        _mirror(f'reference/{case_id}/progress.json',local_root)
        if _local_sha256(local_state)!=result['states_sha256']:
            raise ValueError(f'Downloaded reference hash mismatch: {case_id}')
        print(json.dumps(dict(event='reference_mirrored',case_id=case_id,
                              states_sha256=result['states_sha256'])),flush=True)
    print(json.dumps(dict(event='all_references_complete',cases=len(cases))),flush=True)


@app.local_entrypoint()
def run_training(source_sha: str,local_root: str,stop_epoch: int=5):
    if stop_epoch!=5:raise ValueError('The approved run stops at E5')
    manifest=json.loads((Path(local_root)/'manifest.json').read_text())
    for c in manifest['cases']:
        path=Path(local_root)/'reference'/c['case_id']/'states.npy'
        if not path.exists():raise FileNotFoundError(path)
    if not (Path(local_root)/'evaluation/epoch_0000/report.json').exists():
        print(json.dumps(dict(event='evaluation',result=evaluate_epoch.remote(source_sha,0))),flush=True)
        _mirror('evaluation/epoch_0000/report.json',local_root)
    last_epoch=0
    while last_epoch<stop_epoch:
        result=train_chunk.remote(source_sha,stop_epoch,50)
        print(json.dumps(dict(event='training_chunk',record=result)),flush=True)
        update=result['accepted_updates']
        _mirror(f'training/state_{update:08d}.npz',local_root)
        if result['next_batch']==0:
            finished=result['epoch']-1
            if finished>last_epoch:
                last_epoch=finished
                print(json.dumps(dict(event='evaluation',result=evaluate_epoch.remote(source_sha,finished))),flush=True)
                _mirror(f'evaluation/epoch_{finished:04d}/report.json',local_root)
                # All plot and per-case arrays are downloaded when their epoch is complete.
                import subprocess
                dest=Path(local_root)/'evaluation'/f'epoch_{finished:04d}'
                subprocess.run(['modal','volume','get',VOLUME_NAME,
                    f'{NAME}/evaluation/epoch_{finished:04d}',str(dest),'--force'],check=True)
    print(json.dumps(dict(event='E5_complete',epoch=last_epoch)),flush=True)
