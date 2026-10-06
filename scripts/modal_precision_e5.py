"""Protocol-configured random-start training with immutable, resumable saves.

VPML_RUN_NAME selects an isolated, already prepared setup directory. Historical
defaults remain usable, while each new setup freezes its own scientific source.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import modal

REPO = Path(__file__).resolve().parents[1]
NAME = os.environ.get('VPML_RUN_NAME', 'wavenumber_fp64_e5_20261004')
if not NAME.replace('_', '').replace('-', '').isalnum():
    raise ValueError('Run name must be a single safe namespace')
APP = 'vpml-' + NAME.replace('_', '-')
LOCAL = Path('/Users/armin/Documents/NYU/vpml/out_bench') / NAME
FROZEN = LOCAL / 'frozen_source'
SETUP = LOCAL / 'setup'
REMOTE = Path('/mnt/new')
OUT = REMOTE / NAME
REF = REMOTE / 'wavenumber_generalization_v2_e5_20260925'
ORIGIN = REMOTE / 'wavenumber_initial_rms_e5_20261002'
DEFAULT_GATES = (0, 3, 50, 150, 300, 592, 1184, 1776, 2368, 2960)
volume = modal.Volume.from_name('vpml-wavenumber-generalization-v2-20260925', create_if_missing=False, version=2)
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda13]==0.10.0', 'numpy==2.4.4', 'matplotlib==3.10.9', 'scipy')
    .add_local_dir(str(FROZEN/'model'), '/root/vpml/model', copy=True)
    .add_local_dir(str(FROZEN/'vpml'), '/root/vpml/vpml', copy=True)
    .add_local_file(str(SETUP/'wavenumber_retry_evaluation.py'), '/root/vpml/wavenumber_retry_evaluation.py', copy=True)
    .add_local_dir(str(SETUP), '/root/setup', copy=True)
    .env({'PYTHONPATH': '/root/vpml', 'VPML_JAX_BACKEND': 'gpu', 'PYTHONUNBUFFERED': '1',
          'VPML_RUN_NAME': NAME, 'JAX_ENABLE_X64': 'true',
          'JAX_DEFAULT_MATMUL_PRECISION': 'highest',
          'XLA_FLAGS': '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true'}))
RUNTIME = {}


def source_hash(root):
    h = hashlib.sha256()
    for package in ('model', 'vpml'):
        for p in sorted((root/package).rglob('*.py')):
            h.update(p.relative_to(root).as_posix().encode()); h.update(b'\0')
            h.update(p.read_bytes()); h.update(b'\0')
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, indent=2, allow_nan=False)


def evaluation_configuration(config):
    """Optimizer rates do not affect a fixed-weight trajectory or loss."""
    return {k:v for k,v in config.items() if k not in ('learning_rate','final_learning_rate')}


def stopping_updates(protocol, stop_epoch, stop_update=0):
    if not 1<=stop_epoch<=protocol['stop_epoch']:raise ValueError('Unapproved stopping epoch')
    target=stop_update or stop_epoch*protocol['updates_per_epoch']
    if not 0<target<=stop_epoch*protocol['updates_per_epoch']:raise ValueError('Unapproved stopping update')
    gates=[g for g in protocol.get('autonomous_gate_updates',DEFAULT_GATES) if g<=target]
    if not gates or gates[-1]!=target:raise ValueError('Missing final evaluation gate')
    return target,gates


def validate_continuation(parent, child):
    """Allow a longer endpoint, never a changed scientific training prefix."""
    fixed = ('config', 'source_sha256', 'evaluation_source_sha256',
        'manifest_sha256', 'normalization_sha256', 'random_float32_init_sha256',
        'validation_configurations', 'runtime', 'updates_per_epoch',
        'windows_per_epoch', 'planned_epochs', 'reference_root',
        'normalization_origin', 'train_count', 'heldout_count',
        'initial_descent_check_batches', 'stop_on_nonfinite', 'combined_cost_cap')
    for key in fixed:
        if child.get(key) != parent.get(key):
            raise ValueError(f'Continuation changes {key}')
    resume = child['resume_from']
    name = resume['run']
    if not name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Unsafe parent namespace')
    count = parent['updates_per_epoch']
    start = parent['stop_epoch'] * count
    if resume['accepted_updates'] != start:
        raise ValueError('Continuation must restore the completed parent endpoint')
    if not parent['stop_epoch'] < child['stop_epoch'] <= parent['config']['planned_epochs']:
        raise ValueError('Invalid continuation endpoint')
    expected = parent['autonomous_gate_updates'] + [
        epoch * count for epoch in range(parent['stop_epoch'] + 1, child['stop_epoch'] + 1)]
    if child['autonomous_gate_updates'] != expected:
        raise ValueError('Continuation must retain every autonomous epoch gate')
    return start


def prepare_continuation(ctx):
    """Copy a completed lineage into a new namespace, retaining every state array."""
    import shutil, numpy as np
    from model.train.interface_flux_data import sha256_file, sha256_json
    from model.train.wavenumber_training import load_snapshot
    resume = ctx['protocol']['resume_from']; parent = REMOTE / resume['run']
    if parent == OUT:
        raise ValueError('Continuation cannot overwrite its parent')
    old = json.loads((parent/'protocol.json').read_text())
    if sha256_json(old) != resume['protocol_sha256']:
        raise ValueError('Parent protocol hash mismatch')
    start = validate_continuation(old, ctx['protocol'])
    complete = json.loads((parent/'COMPLETE.json').read_text())
    if (parent/'STOPPED.json').exists() or complete['accepted_updates'] != start:
        raise ValueError('Parent is not a completed successful qualification')
    checkpoint = parent/'training'/f'state_{start:08d}.npz'
    if sha256_file(checkpoint) != resume['checkpoint_sha256']:
        raise ValueError('Parent checkpoint hash mismatch')
    p,o,r,rows = load_snapshot(checkpoint, resume['protocol_sha256'])
    if r['accepted_updates'] != start:
        raise ValueError('Parent checkpoint is not the requested endpoint')
    r = dict(r, protocol_sha256=ctx['digest'])
    validate(r,o,rows)
    # Hash every reused evaluation and the original initialization qualification.
    for rel,digest in resume['reused_files'].items():
        path = Path(rel)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe parent artifact path')
        if sha256_file(parent/path) != digest:
            raise ValueError(f'Parent artifact changed: {rel}')
    for gate in old['autonomous_gate_updates']:
        rel = f'evaluation/update_{gate:08d}/report.json'
        if rel not in resume['reused_files']:
            raise ValueError(f'Missing parent evaluation: {gate}')
        report = json.loads((parent/rel).read_text())
        if report['nonfinite_count'] or any(v is None for v in report['validation_losses'].values()):
            raise ValueError('Failed parent autonomous gate')
    OUT.mkdir(exist_ok=False)
    shutil.copytree('/root/setup',OUT/'setup')
    for rel in resume['reused_files']:
        target = OUT/rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent/rel,target)
    write(OUT/'protocol.json',ctx['protocol'])
    snapshot(p,o,r,rows)
    with np.load(checkpoint,allow_pickle=False) as before, np.load(
            OUT/'training'/checkpoint.name,allow_pickle=False) as after:
        if set(before.files) != set(after.files):
            raise ValueError('Continuation checkpoint fields changed')
        for key in before.files:
            if key == 'record': continue
            a,b = before[key],after[key]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError(f'Continuation changed checkpoint array {key}')
    result = dict(status='ready',accepted_updates=start,parent_run=resume['run'],
        parent_checkpoint_sha256=resume['checkpoint_sha256'],
        exact_array_and_optimizer_parity=True,preserved_loss_rows=len(rows),
        next_epoch=r['epoch'],next_batch=r['next_batch'],
        note='Only the provenance protocol hash changes. Scientific config, optimizer, data cursor and loss prefix are unchanged.')
    write(OUT/'CONTINUATION_PARITY.json',result)
    write(OUT/'READY.json',result)
    volume.commit(); return result


def context():
    import numpy as np
    import jax
    from model.train.wavenumber_training import Configuration
    from model.train.interface_flux_data import sha256_json, sha256_file
    jax.config.update('jax_default_matmul_precision', 'highest')
    if not RUNTIME:
        protocol = json.loads(Path('/root/setup/protocol.json').read_text())
        if protocol.get('controller_sha256') and sha256_file(__file__) != protocol['controller_sha256']:
            raise ValueError('Frozen controller changed')
        if source_hash(Path('/root/vpml')) != protocol['source_sha256']:
            raise ValueError('Frozen scientific source changed')
        if sha256_file('/root/vpml/wavenumber_retry_evaluation.py') != protocol['evaluation_source_sha256']:
            raise ValueError('Frozen evaluator changed')
        manifest = json.loads((REF/'manifest.json').read_text())
        if ((manifest['train_count'], manifest['development_count']) !=
                (protocol.get('train_count',54),protocol.get('heldout_count',21))
                or manifest['manifest_sha256'] != protocol['manifest_sha256']
                or any(c['provenance'] != 'expanded_v2' for c in manifest['cases'])):
            raise ValueError('Wrong dataset lineage')
        if sha256_file(ORIGIN/'normalization.npz') != protocol['normalization_sha256']:
            raise ValueError('Normalization changed')
        RUNTIME.update(protocol=protocol, digest=sha256_json(protocol), manifest=manifest,
            config=Configuration(**protocol['config']), stats=dict(np.load(ORIGIN/'normalization.npz')),
            states={c['case_id']:np.load(REF/'reference'/c['case_id']/'states.npy', mmap_mode='r') for c in manifest['cases']})
    return RUNTIME


def validate(record, opt, rows):
    from model.train.wavenumber_data import epoch_batches
    ctx=context(); n = record['accepted_updates']; count=ctx['protocol']['updates_per_epoch']
    batches=list(epoch_batches(ctx['manifest'],record['epoch'],ctx['config'].seed))
    windows=(n//count)*ctx['protocol']['windows_per_epoch']+sum(len(b) for b in batches[:n%count])
    if (int(opt['step']) != n or len(rows) != n or record['epoch'] != n//count+1
            or record['next_batch'] != n%count or record['windows'] != windows):
        raise ValueError('Optimizer/exposure cursor mismatch')
    if n and (rows[0][0] != 1 or rows[-1][0] != n): raise ValueError('Loss prefix mismatch')


def snapshot(p, o, record, rows):
    from model.train.wavenumber_training import save_snapshot
    from model.train.interface_flux_data import sha256_file
    validate(record, o, rows)
    path = OUT/'training'/f"state_{record['accepted_updates']:08d}.npz"
    if path.exists(): raise FileExistsError(f'Immutable checkpoint exists: {path}')
    digest = save_snapshot(path, p, o, record, rows)
    write(path.with_suffix('.json'), dict(sha256=digest, **record))
    return digest


def load_latest():
    from model.train.wavenumber_training import load_snapshot
    from model.train.interface_flux_data import sha256_file
    path = sorted((OUT/'training').glob('state_*.npz'))[-1]
    if sha256_file(path) != json.loads(path.with_suffix('.json').read_text())['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    p,o,r,rows = load_snapshot(path, context()['digest']); validate(r,o,rows)
    return p,o,r,rows


@app.function(image=image, gpu='A100-80GB', cpu=4, memory=8192, timeout=600, volumes={'/mnt/new':volume}, max_containers=1, retries=0)
def prepare():
    import shutil, numpy as np
    from dataclasses import replace
    from model.train.wavenumber_training import initialize, tree_hash, base, load_snapshot
    from model.train.interface_flux_data import sha256_file
    from model.train.wavenumber_data import epoch_batches
    ctx=context(); c=ctx['config']
    if OUT.exists():
        if not (OUT/'READY.json').exists(): raise ValueError('Partial setup: preserve and inspect')
        if json.loads((OUT/'protocol.json').read_text()) != ctx['protocol']: raise ValueError('Resume protocol mismatch')
        return dict(status='existing')
    if 'resume_from' in ctx['protocol']:
        return prepare_continuation(ctx)
    hashes=json.loads((ORIGIN/'reference_hashes.json').read_text())
    for cid, a in ctx['states'].items():
        if a.shape != (4801,3,128) or not np.isfinite(a).all() or sha256_file(REF/'reference'/cid/'states.npy') != hashes[cid]:
            raise ValueError(f'Reference mismatch {cid}')
    batches=list(epoch_batches(ctx['manifest'],1,c.seed))
    if (len(batches)!=ctx['protocol']['updates_per_epoch'] or
            sum(map(len,batches))!=ctx['protocol']['windows_per_epoch']): raise ValueError('Exposure mismatch')
    p=initialize(c); original=initialize(replace(c,compute_dtype='float32'))
    if tree_hash(original)!=ctx['protocol']['random_float32_init_sha256']: raise ValueError('Random initialization changed')
    for k in p: np.testing.assert_array_equal(np.asarray(p[k]),np.asarray(original[k]).astype(np.float64))
    OUT.mkdir(exist_ok=False)
    shutil.copytree('/root/setup',OUT/'setup')
    for name in ('manifest.json','normalization.npz','reference_hashes.json'): shutil.copy2(ORIGIN/name,OUT/name)
    write(OUT/'protocol.json',ctx['protocol'])
    r=dict(protocol_sha256=ctx['digest'],accepted_updates=0,epoch=1,next_batch=0,windows=0)
    snapshot(p,base._adam_init(p),r,[])
    cache=ctx['protocol'].get('reuse_e0_evaluation')
    if cache:
        cached=REMOTE/cache['run']
        old=json.loads((cached/'protocol.json').read_text())
        from model.train.interface_flux_data import sha256_json
        for key in ('source_sha256','evaluation_source_sha256','manifest_sha256','normalization_sha256'):
            if old[key]!=ctx['protocol'][key]:raise ValueError(f'E0 cache {key} mismatch')
        if evaluation_configuration(old['config'])!=evaluation_configuration(ctx['protocol']['config']):
            raise ValueError('E0 cache model configuration mismatch')
        if {k:evaluation_configuration(v) for k,v in old['validation_configurations'].items()}!={k:evaluation_configuration(v) for k,v in ctx['protocol']['validation_configurations'].items()}:
            raise ValueError('E0 cache loss configuration mismatch')
        cp,co,cr,crows=load_snapshot(cached/'training/state_00000000.npz',sha256_json(old))
        if tree_hash(cp)!=tree_hash(p) or int(co['step']) or cr['accepted_updates'] or crows:
            raise ValueError('E0 cache weights mismatch')
        source=cached/'evaluation/update_00000000'
        for rel,digest in cache['files'].items():
            if sha256_file(source/rel)!=digest:raise ValueError(f'E0 cache artifact mismatch: {rel}')
        shutil.copytree(source,OUT/'evaluation/update_00000000')
        write(OUT/'E0_EVALUATION_REUSED.json',dict(source=str(source),verified=True,weights_sha256=tree_hash(p)))
    write(OUT/'READY.json',dict(status='ready',references_verified=len(ctx['states']),random_seed=c.seed,
        random_float32_init_sha256=tree_hash(original),float64_init_sha256=tree_hash(p),
        training_windows_per_epoch=sum(map(len,batches)),updates_per_epoch=len(batches)))
    volume.commit(); return dict(status='ready',accepted_updates=0)


@app.function(image=image, gpu='A100-80GB',cpu=16,memory=65536,timeout=1800,volumes={'/mnt/new':volume},max_containers=1,retries=0)
def train_chunk(stop: int):
    import numpy as np, jax
    from model.train.wavenumber_training import update_factory,batch_from_states,loss_factory,load_snapshot,base,tree_hash
    from model.train.wavenumber_data import epoch_batches
    ctx=context();c=ctx['config']
    if (OUT/'STOPPED.json').exists():return json.loads((OUT/'STOPPED.json').read_text())
    gates=ctx['protocol'].get('autonomous_gate_updates',DEFAULT_GATES)
    count=ctx['protocol']['updates_per_epoch']
    if stop not in gates:raise ValueError('Unapproved endpoint')
    p,o,r,rows=load_latest();start=r['accepted_updates']
    for gate in gates:
        if gate<=start:
            f=OUT/'evaluation'/f'update_{gate:08d}'/'report.json'
            if not f.exists():raise ValueError(f'Missing autonomous gate {gate}')
            report=json.loads(f.read_text())
            if report['nonfinite_count'] or any(v is None for v in report['validation_losses'].values()):raise ValueError('Failed autonomous gate')
    if start>=stop:return dict(status='saved',**r)
    limit=min(stop,(start//50+1)*50,(start//count+1)*count)
    if 'step' not in ctx:ctx['step']=update_factory(c,ctx['stats'])
    step=ctx['step'];batches=list(epoch_batches(ctx['manifest'],r['epoch'],c.seed))
    checks=ctx['protocol'].get('initial_descent_check_batches',[])
    if checks and not (OUT/'rate_preflight.json').exists():
        if start:raise ValueError('Missing initial rate qualification')
        value=jax.jit(loss_factory(c,ctx['stats']));results=[]
        for batch_number in checks:
            check_path=OUT/'preflight'/f'batch_{batch_number:04d}.json'
            if check_path.exists():
                check=json.loads(check_path.read_text())
                if check['initial_model_sha256']!=tree_hash(p) or check['optimizer_step']!=0 or check['learning_rate']!=c.learning_rate:
                    raise ValueError('Partial preflight identity mismatch')
                results.append(check);continue
            b=batch_from_states(ctx['manifest'],batches[batch_number-1],ctx['states'],c)
            trial=step(p,o,b,c.learning_rate)
            after=float(value(trial[0],b));before=float(trial[2])
            check=dict(batch_number=batch_number,before=before,after=after,finite=bool(trial[-1]),
                gradient_norm=float(trial[3]),update_norm=float(trial[4]),initializer_gradient_norm=float(trial[5]),
                initial_model_sha256=tree_hash(p),learning_rate=c.learning_rate,
                optimizer_step=int(o['step']),passed=bool(trial[-1]) and np.isfinite(after) and after<before and float(trial[5])>0)
            write(check_path,check);results.append(check)
            volume.commit();print(json.dumps(dict(event='rate_preflight',**check)),flush=True)
        summary=dict(passed=all(v['passed'] for v in results),checks=results,accepted_updates=0,
            note='Every trial starts from the same E0 and zero Adam state. Trial weights are discarded.')
        write(OUT/'rate_preflight.json',summary)
        if not summary['passed']:
            write(OUT/'STOPPED.json',dict(status='initial_rate_preflight_failed',accepted_updates=0));volume.commit()
            return dict(status='initial_rate_preflight_failed',accepted_updates=0)
        volume.commit()
    elif checks and not json.loads((OUT/'rate_preflight.json').read_text())['passed']:
        raise ValueError('Initial rate qualification failed')
    started=time.monotonic();timings=[]
    for bi in range(r['next_batch'],count):
        b=batch_from_states(ctx['manifest'],batches[bi],ctx['states'],c)
        lr=base._cosine_learning_rate(r['accepted_updates'],count*c.planned_epochs,c.learning_rate,c.final_learning_rate)
        tick=time.monotonic();out=step(p,o,b,lr)
        pp,oo,loss,gn,un,ig,finite=out;finite=bool(finite);seconds=time.monotonic()-tick;timings.append(seconds)
        if not finite:
            if not (OUT/'training'/f"state_{r['accepted_updates']:08d}.npz").exists():snapshot(p,o,r,rows)
            fail=dict(status='nonfinite_update',attempted_update=r['accepted_updates']+1,accepted_updates=r['accepted_updates'])
            with (OUT/f"rejected_update_{r['accepted_updates']+1:08d}.npz").open('xb') as f:
                payload={f'param_{k}':np.asarray(v) for k,v in pp.items()}
                payload.update({f'optimizer_{part}_{k}':np.asarray(v) for part in ('m','v') for k,v in oo[part].items()})
                np.savez(f,**payload,loss=np.asarray(loss),gradient_norm=np.asarray(gn))
            write(OUT/'STOPPED.json',fail);volume.commit();return fail
        resume = ctx['protocol'].get('resume_from')
        if resume and r['accepted_updates'] == resume['accepted_updates']:
            rp,ro,_,_ = load_latest()
            repeated = step(rp,ro,b,lr)
            equal = all(np.array_equal(np.asarray(a),np.asarray(d)) for a,d in
                zip(jax.tree_util.tree_leaves(out),jax.tree_util.tree_leaves(repeated)))
            check = dict(bitwise_repeat_resume=equal,from_update=r['accepted_updates'],
                attempted_update=r['accepted_updates']+1,learning_rate=float(lr),
                loss=float(loss),finite=finite,epoch=r['epoch'],batch=bi+1)
            write(OUT/'continuation_first_update_check.json',check)
            if not equal:
                write(OUT/'STOPPED.json',dict(status='continuation_replay_failed',
                    accepted_updates=r['accepted_updates'],check=check))
                volume.commit();return dict(status='continuation_replay_failed',**r)
        if start==0 and bi==0:
            # Same random model and update, repeated after exact disk restore.
            rp,ro,rr,_=load_snapshot(OUT/'training/state_00000000.npz',ctx['digest'])
            repeated=step(rp,ro,b,lr)
            equal=all(np.array_equal(np.asarray(a),np.asarray(d)) for a,d in zip(jax.tree_util.tree_leaves(out),jax.tree_util.tree_leaves(repeated)))
            after=float(jax.jit(loss_factory(c,ctx['stats']))(pp,b))
            check=dict(bitwise_repeat_resume=equal,before=float(loss),after=after,finite=finite,
                batch_regimes=np.unique(b['regime_index']).tolist(),gradient_norm=float(gn),update_norm=float(un),
                initializer_gradient_norm=float(ig),scored_steps=c.horizon,unscored_steps=c.preparation_steps)
            write(OUT/'first_update_check.json',check)
            init_ok=c.preparation_steps>0 or float(ig)>0
            if not equal or not np.isfinite(after) or after>=float(loss) or not init_ok:
                write(OUT/'STOPPED.json',dict(status='preflight_failed',accepted_updates=0,check=check));volume.commit()
                return dict(status='preflight_failed',accepted_updates=0)
        p,o=pp,oo;r['accepted_updates']+=1;r['windows']+=len(batches[bi]);r['next_batch']=bi+1
        rows.append([r['accepted_updates'],float(loss),float(gn),float(un),float(ig)])
        print(json.dumps(dict(event='update',updates=r['accepted_updates'],loss=float(loss),gradient_norm=float(gn),update_norm=float(un),seconds=seconds)),flush=True)
        if r['next_batch']==count:r['epoch']+=1;r['next_batch']=0
        if r['accepted_updates']<=3:
            snapshot(p,o,r,rows);volume.commit()
            # The next update uses the exact persisted model and optimizer.
            p,o,r,rows=load_latest()
        if r['accepted_updates']>=limit:break
    if not (OUT/'training'/f"state_{r['accepted_updates']:08d}.npz").exists():snapshot(p,o,r,rows)
    result=dict(status='saved',**r,elapsed_seconds=time.monotonic()-started,step_seconds=timings)
    write(OUT/'timings'/f'chunk_{start:08d}_{r["accepted_updates"]:08d}.json',result)
    volume.commit();return result


@app.function(image=image,gpu='A100-80GB',cpu=16,memory=65536,timeout=1800,volumes={'/mnt/new':volume},max_containers=1,retries=0)
def evaluate(updates: int):
    from model.train.wavenumber_training import load_snapshot,Configuration
    from wavenumber_retry_evaluation import evaluate_checkpoint
    ctx=context();p,o,r,rows=load_snapshot(OUT/'training'/f'state_{updates:08d}.npz',ctx['digest']);validate(r,o,rows)
    configs={name:Configuration(**values) for name,values in ctx['protocol'].get('validation_configurations',{}).items()}
    report=evaluate_checkpoint(p,ctx['manifest'],ctx['config'],ctx['stats'],REF,
        OUT/'evaluation'/f'update_{updates:08d}',updates/ctx['protocol']['updates_per_epoch'],accepted_updates=updates,
        validation_configs=configs or {'window_rms_l2':ctx['config']})
    if (report['nonfinite_count'] or any(v is None for v in report['validation_losses'].values())) and not (OUT/'STOPPED.json').exists():
        write(OUT/'STOPPED.json',dict(status='nonfinite_evaluation',accepted_updates=updates,
            finite_cases=report['finite_count'],nonfinite_cases=report['nonfinite_count']))
    volume.commit();return {k:v for k,v in report.items() if k!='cases'}


@app.function(image=image,cpu=1,memory=1024,timeout=120,volumes={'/mnt/new':volume},max_containers=1,retries=0)
def complete(epoch: int, updates: int = 0):
    protocol=json.loads((OUT/'protocol.json').read_text())
    if not 1<=epoch<=protocol['stop_epoch']:raise ValueError('Unapproved epoch')
    n,_=stopping_updates(protocol,epoch,updates)
    r=json.loads((OUT/'evaluation'/f'update_{n:08d}'/'report.json').read_text())
    if (OUT/'STOPPED.json').exists() or r['nonfinite_count'] or any(v is None for v in r['validation_losses'].values()):raise ValueError('Failed qualification')
    final=n==protocol['stop_epoch']*protocol['updates_per_epoch']
    name='COMPLETE.json' if final else f'REACHED_UPDATE_{n:08d}.json'
    result=dict(status='complete' if final else 'awaiting_trajectory_review',epoch=n/protocol['updates_per_epoch'],accepted_updates=n)
    if not (OUT/name).exists():write(OUT/name,result)
    volume.commit();return result


def mirror():
    index_path=LOCAL/'mirror_index.json';index=json.loads(index_path.read_text()) if index_path.exists() else {}
    for e in volume.iterdir(NAME,recursive=True):
        if not (e.type==1 or str(e.type).endswith('FILE')):continue
        rel=e.path.lstrip('/')[len(NAME)+1:];p=LOCAL/rel
        if rel in index and p.exists():
            if hashlib.sha256(p.read_bytes()).hexdigest()!=index[rel]:raise ValueError(f'Local artifact changed {p}')
            continue
        p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.download');h=hashlib.sha256()
        with tmp.open('wb') as f:
            for b in volume.read_file(e.path):h.update(b);f.write(b)
        if p.exists():
            if hashlib.sha256(p.read_bytes()).hexdigest()!=h.hexdigest():raise ValueError(f'Mirror conflict {p}')
            tmp.unlink()
        else:tmp.rename(p)
        index[rel]=h.hexdigest()
    temp=index_path.with_suffix('.tmp');temp.write_text(json.dumps(index,indent=2));temp.replace(index_path)


@app.local_entrypoint()
def run(stop_epoch: int = 5, stop_update: int = 0):
    import subprocess,fcntl,os
    from datetime import datetime,timezone,timedelta
    LOCAL.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((SETUP/'protocol.json').read_text())
    count=protocol['updates_per_epoch']
    target,gates=stopping_updates(protocol,stop_epoch,stop_update)
    lock=(LOCAL/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    def budget():
        end=(datetime.now(timezone.utc).date()+timedelta(days=1)).isoformat()
        charges=json.loads(subprocess.check_output(['modal','billing','report','--start','2026-09-25','--end',end,'--json'],timeout=40))
        total=sum(float(x['Cost']) for x in charges if x.get('Description','').startswith('vpml-'))
        # Reserve $50 for billing lag and the in-flight bounded call.
        status=dict(reported_combined=total,authorized_cap=350,stop_threshold=300,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat())
        (LOCAL/'budget_status.json').write_text(json.dumps(status,indent=2))
        if total>=300 or time.monotonic()-started>16*3600:raise RuntimeError('Budget/runtime guard')
    def call(fn,*args):
        budget();r=fn.remote(*args);mirror();print(json.dumps(dict(event='stage',result=r)),flush=True);return r
    try:
        call(prepare)
        if (LOCAL/'STOPPED.json').exists() or (LOCAL/'COMPLETE.json').exists():return
        for gate in gates:
            while True:
                n=int(sorted((LOCAL/'training').glob('state_*.npz'))[-1].stem.split('_')[-1])
                if n>=gate:break
                r=call(train_chunk,gate)
                if r['status']!='saved':
                    if r['accepted_updates']>0:call(evaluate,r['accepted_updates'])
                    return
                if n==0:
                    steps=r['step_seconds'][1:]
                    if steps:
                        estimate=sum(steps)/len(steps)*target
                        write(LOCAL/'runtime_estimate.json',dict(warm_seconds_per_update=sum(steps)/len(steps),training_seconds_to_stop=estimate,stop_epoch=stop_epoch,stop_update=target,
                            excludes_evaluation_compile_transfer=True))
            cached=LOCAL/'evaluation'/f'update_{gate:08d}'/'report.json'
            r=json.loads(cached.read_text()) if cached.exists() else call(evaluate,gate)
            if r['nonfinite_count'] or any(v is None for v in r['validation_losses'].values()):return
        call(complete,stop_epoch,target)
    except Exception as exc:
        try:mirror()
        except Exception as mex:print(f'Mirror failed: {mex}',flush=True)
        write(LOCAL/f'INTERRUPTED_{time.time_ns()}.json',dict(error=repr(exc),instruction='Resume exact saved state after infrastructure inspection only. Never retry scientific failure.'))
        raise
