"""Random-start full-family training with a fixed balanced trajectory loss."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from model.eval_coupled_low_moment_latent import _load_saved_model
from model.train.fresh_coupled_latent import fresh_parameters
from model.train.fixed_family_objective import make_regime_balanced_objective
from model.train.coupled_run_support import RunRecord, atomic_json, file_digest
from model.train.coupled_low_moment_latent import _atomic_savez
from model.train.parameter_metric import initial_metric, proposal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe-run',type=Path,required=True)
    parser.add_argument('--normalization',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--architecture',choices=('shared','september_experts'),default='shared')
    parser.add_argument('--cached-first-step',type=Path)
    parser.add_argument('--cached-direction',type=Path)
    parser.add_argument('--cached-epoch-one',type=Path)
    parser.add_argument('--cached-initial-gradient',type=Path)
    parser.add_argument('--parameter-metric',choices=('euclidean','initial_rms','expert_balanced'),default='euclidean')
    parser.add_argument('--guard-reference',type=Path)
    parser.add_argument('--cached-second-trials',type=Path)
    parser.add_argument('--shared-run',type=Path,
        help='Qualified shared-base run used to initialize expert post-training')
    parser.add_argument('--shared-checkpoint',default='last_coupled_low_moment_latent.npz')
    parser.add_argument('--resume-run',type=Path,
        help='Continue this exact training lineage in a new output directory')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    from model.train.expert_balanced_metric import balanced_proposal
    propose = balanced_proposal if args.parameter_metric == "expert_balanced" else proposal
    if args.epochs<1:raise ValueError('Positive epoch count required')
    if bool(args.cached_first_step)!=bool(args.cached_direction):raise ValueError('Supply both cache paths')
    if args.architecture!='shared' and args.cached_first_step:
        raise ValueError('Shared-network cached updates cannot initialize experts')
    if args.cached_epoch_one and args.cached_first_step:
        raise ValueError('Use only one prefix cache')
    if args.cached_initial_gradient and (args.cached_epoch_one or args.cached_first_step):
        raise ValueError('Initial-gradient reuse is separate from update replay')
    if args.parameter_metric!='euclidean' and args.architecture!='september_experts':
        raise ValueError('RMS metric requires the audited nonzero expert initialization')
    if bool(args.shared_run) != (args.architecture == 'september_experts'):
        raise ValueError('Expert post-training requires exactly one qualified shared run')
    if args.resume_run and any((args.cached_first_step,args.cached_epoch_one,
                                args.cached_initial_gradient,args.cached_second_trials)):
        raise ValueError('Resume cannot be combined with prefix caches')
    if args.cached_second_trials and (not args.cached_epoch_one or not args.guard_reference):
        raise ValueError('Second-trial replay requires an exact first-epoch cache and calibrated guards')
    jax.config.update('jax_enable_x64',True)
    origin=args.recipe_run/'initial_coupled_low_moment_latent.npz'
    _,config,stored,arrays=_load_saved_model(args.recipe_run,origin.name)
    if config['latent_rank']>60:raise ValueError('Compact rank limit exceeded')
    config={**config,'seed':args.seed,'initialization':'random','init_checkpoint':None,
            'optimizer_protocol':'fixed_balanced_scalar_v1','precision':'float64',
            'parameter_metric':args.parameter_metric}
    constants={}
    if args.architecture=='september_experts':
        from model.train.random_september_experts import initialize
        params,constants=initialize(config)
        config.update(conditioner_experts=4,neural_architecture='random_september_experts_v1',
                      trainable_parameter_keys=sorted(params),fixed_parameter_keys=sorted(constants))
    else:
        params=fresh_parameters(config)
    def promote(v):
        a=np.asarray(v)
        return a.astype(np.complex128) if np.iscomplexobj(a) else a.astype(np.float64)
    params={k:jnp.asarray(promote(v)) for k,v in params.items()}
    constants={k:jnp.asarray(promote(v)) for k,v in constants.items()}
    if args.shared_run:
        shared_path=args.shared_run/args.shared_checkpoint
        _,shared_config,shared_params,shared_arrays=_load_saved_model(args.shared_run,args.shared_checkpoint)
        for key in ('latent_rank','nx','basis_modes','cadence','rollout_steps'):
            if shared_config[key]!=config[key]:raise ValueError(f'Shared-base {key} differs')
        for key in arrays:
            if key in shared_arrays and not np.array_equal(promote(arrays[key]),promote(shared_arrays[key])):
                raise ValueError(f'Shared-base numerical array differs: {key}')
        missing=[k for k in params if not k.startswith(('expert_','specialist_')) and k not in shared_params]
        if missing:raise ValueError(f'Shared-base tensors missing: {missing}')
        for key in params:
            if not key.startswith(('expert_','specialist_')):
                params[key]=jnp.asarray(promote(shared_params[key]))
        for key in ('output_bias','operator_v_real'):
            if key in shared_params:constants[key]=jnp.asarray(promote(shared_params[key]))
        config.update(shared_parent_checkpoint=str(shared_path),
                      shared_parent_sha256=file_digest(shared_path),
                      post_training_epoch_origin=0)
    metric=initial_metric(params,"initial_rms" if args.parameter_metric=="expert_balanced" else args.parameter_metric)
    arrays={k:promote(v) for k,v in arrays.items()}
    normalization=json.loads(args.normalization.read_text())
    scales=np.asarray(normalization['denominators'])
    regimes=['linear_landau','nonlinear_landau_strong','nonlinear_landau_weak']
    if normalization['regime_order']!=regimes:raise ValueError('Normalization order mismatch')
    config['balanced_denominators']=scales.tolist()
    cases=json.loads((args.recipe_run/'training_cases.json').read_text())['cases']
    groups=[sorted(c['case_id'] for c in cases if c['regime']==r) for r in regimes]
    if any(len(g)!=16 for g in groups):raise ValueError('Expected48 balanced training cases')
    reference=None
    if args.guard_reference:
        reference=json.loads(args.guard_reference.read_text())
        for regime,index in (('linear_landau',0),('nonlinear_landau_weak',2)):
            values=np.asarray(reference[regime]['mean_global_local'])
            if reference[regime]['case_ids']!=groups[index] or values.shape!=(2,) or not np.all(np.isfinite(values)) or np.any(values<=0):
                raise ValueError('Historical guard calibration does not match this family')
        config['guard_reference_sha256']=file_digest(args.guard_reference)
    def guard_limits(origin_values):
        limits=np.array(origin_values,copy=True)
        if reference is not None:
            for regime,indices in (('linear_landau',[0,1]),('nonlinear_landau_weak',[4,5])):
                limits[indices]=np.maximum(limits[indices],reference[regime]['mean_global_local'])
        return limits
    def same_model_config(other):
        return {k:v for k,v in other.items() if k!='guard_reference_sha256'}=={k:v for k,v in config.items() if k!='guard_reference_sha256'}
    batches=list(zip(*groups));cache=Path(config['projected_cache'])
    teacher=json.loads((Path(config['reference_cache'])/'metadata.json').read_text())['configuration']
    complete_objective=make_regime_balanced_objective(config,arrays,teacher,scales)
    def objective(p,target):
        return complete_objective({**constants,**p},target)
    value=jax.jit(objective);grad=jax.jit(jax.value_and_grad(objective,has_aux=True))
    sources=[Path(__file__),Path('model/train/fixed_family_objective.py'),Path('model/train/fresh_coupled_latent.py'),Path('model/train/coupled_low_moment_latent.py'),Path('model/train/coupled_run_support.py'),Path('vpml/kinetic_latent.py'),Path('vpml/low_moment.py'),origin,args.normalization,args.recipe_run/'report.json',args.recipe_run/'training_cases.json']
    if args.cached_first_step:sources += [args.cached_first_step/'trials.json',args.cached_first_step/'diagnostic_candidate.npz',args.cached_direction]
    if args.architecture=='september_experts':sources.append(Path('model/train/random_september_experts.py'))
    if args.shared_run:sources += [args.shared_run/args.shared_checkpoint,args.shared_run/'report.json',args.shared_run/'training_cases.json']
    if args.resume_run:sources += [args.resume_run/'last_coupled_low_moment_latent.npz',args.resume_run/'report.json',args.resume_run/'run.json']
    sources.append(Path('model/train/parameter_metric.py'))
    if args.parameter_metric=='expert_balanced':sources.append(Path('model/train/expert_balanced_metric.py'))
    if args.guard_reference:sources.append(args.guard_reference)
    if args.cached_second_trials:
        sources += [args.cached_second_trials/n for n in ('run.json','report.json',
            'epoch001_coupled_low_moment_latent.npz','epoch002_coupled_low_moment_latent.npz',
            'epoch002_work/gradient_progress.npz')]
        sources += sorted((args.cached_second_trials/'epoch002_work').glob('trial*.json'))
    if args.cached_initial_gradient:
        sources += [args.cached_initial_gradient/n for n in ('run.json','report.json',
            'initial_coupled_low_moment_latent.npz','epoch001_work/gradient_progress.npz')]
    if args.cached_epoch_one:
        sources += [args.cached_epoch_one/n for n in ('run.json','report.json',
            'initial_coupled_low_moment_latent.npz','epoch001_coupled_low_moment_latent.npz',
            'epoch001_work/gradient_progress.npz')]
    record=RunRecord(args.outdir,{'seed':args.seed,'epochs':args.epochs,'architecture':args.architecture,'case_order':[n for b in batches for n in b],'denominators':scales.tolist(),'scope':'Training only; frontier and E200 unvalidated'},sources=sources)
    history=[]
    def save(name):_atomic_savez(args.outdir/name,{**arrays,**{k:np.asarray(v) for k,v in {**constants,**params}.items()}})
    def report():atomic_json(args.outdir/'report.json',{'configuration':config,'history':history,'scope':'Unvalidated fixed-family training'})
    def data(names):
        t=np.stack([np.load(cache/'cases'/(n+'.npy')) for n in names])
        if t.shape[1]!=config['rollout_steps']+1:raise ValueError('Horizon changed')
        return jnp.asarray(t,dtype=jnp.float64)
    def row(result):
        score,(seven,legacy,components)=result
        return np.r_[float(score),np.asarray(seven),float(legacy),np.asarray(components)]
    try:
        save('initial_coupled_low_moment_latent.npz');report()
        atomic_json(args.outdir/'training_cases.json',{'cases':cases})
        if args.preflight_only:
            names=batches[0]
            result,gradient=grad(params,data(names))
            values=row(result)
            finite=bool(np.all(np.isfinite(values)) and all(
                np.all(np.isfinite(np.asarray(g))) for g in gradient.values()))
            groups={
                'shared':sum(float(np.sum(np.asarray(g)**2)) for k,g in gradient.items()
                    if not k.startswith(('expert_','specialist_'))),
                'expert':sum(float(np.sum(np.asarray(g)**2)) for k,g in gradient.items()
                    if k.startswith('expert_')),
                'router':sum(float(np.sum(np.asarray(g)**2)) for k,g in gradient.items()
                    if k.startswith('specialist_')),
            }
            if not finite or groups['shared']<=0 or (args.architecture=='september_experts'
                    and min(groups['expert'],groups['router'])<=0):
                raise FloatingPointError('Preflight requires finite nonzero active gradients')
            direction,step,slope=propose(gradient,values[0],metric)
            candidate={k:params[k]+min(step,1e-8)*jnp.asarray(direction[k]) for k in params}
            candidate_values=row(value(candidate,data(names)))
            if not np.all(np.isfinite(candidate_values)):
                raise FloatingPointError('Preflight candidate is nonfinite')
            atomic_json(args.outdir/'preflight.json',{'case_ids':list(names),
                'values':values.tolist(),'candidate_values':candidate_values.tolist(),
                'gradient_squared_norms':groups,'proposed_step':step,'slope':slope,
                'scope':'Mechanical wiring check; not training or frontier evidence'})
            record.finish('preflight_passed',gradient_squared_norms=groups)
            return
        start=1
        if args.resume_run:
            prior=json.loads((args.resume_run/'report.json').read_text())
            prior_config=prior['configuration']
            if prior_config!=config:raise ValueError('Resume configuration differs')
            with np.load(args.resume_run/'initial_coupled_low_moment_latent.npz') as z:
                if set(z.files)!=set(arrays)|set(constants)|set(params):raise ValueError('Resume initial keys differ')
                if any(not np.array_equal(np.asarray(v),z[k]) for k,v in {**arrays,**constants,**params}.items()):
                    raise ValueError('Resume initial checkpoint differs')
            with np.load(args.resume_run/'last_coupled_low_moment_latent.npz') as z:
                params={k:jnp.asarray(z[k]) for k in params}
            history=list(prior['history'])
            completed=max(int(row['epoch']) for row in history)
            start=completed+1
            save('last_coupled_low_moment_latent.npz');report()
            record.stage('resume_state_restored',completed_epoch=completed,next_epoch=start)
        if args.cached_epoch_one:
            cache_root=args.cached_epoch_one
            cache_meta=json.loads((cache_root/'run.json').read_text())
            cache_report=json.loads((cache_root/'report.json').read_text())
            if not same_model_config(cache_report['configuration']):raise ValueError('Cached training configuration differs')
            for path,digest in cache_meta['sources'].items():
                if args.guard_reference and Path(path).resolve()==Path(__file__).resolve():continue
                if file_digest(Path(path))!=digest:raise ValueError(f'Cached source changed: {path}')
            if cache_meta['configuration']['case_order']!=[n for b in batches for n in b]:
                raise ValueError('Cached training family differs')
            if cache_meta['status']!='training_complete_unvalidated' or len(cache_report['history'])!=2:
                raise ValueError('Require a completed one-epoch prefix')
            with np.load(cache_root/'initial_coupled_low_moment_latent.npz') as z:
                if set(z.files)!=set(arrays)|set(constants)|set(params):raise ValueError('Cached initial keys differ')
                if any(not np.array_equal(np.asarray(v),z[k]) for k,v in {**arrays,**constants,**params}.items()):
                    raise ValueError('Cached random initialization differs')
            with np.load(cache_root/'epoch001_work/gradient_progress.npz') as z:
                if int(z['completed_batches'])!=len(batches):raise ValueError('Incomplete cached gradient')
                gradient={k:np.asarray(z[k]) for k in params}
            step=cache_report['history'][1]['step']
            direction,_,_=propose(gradient,cache_report['history'][0]['values'][0],metric)
            updated={k:params[k]+step*jnp.asarray(direction[k]) for k in params}
            with np.load(cache_root/'epoch001_coupled_low_moment_latent.npz') as z:
                if any(not np.array_equal(np.asarray(v),z[k]) for k,v in {**arrays,**constants,**updated}.items()):
                    raise ValueError('Cached first update replay differs')
            history=cache_report['history']
            if not(history[1]['values'][0]<history[0]['values'][0] and np.all(
                np.asarray(history[1]['values'][1:8])<=guard_limits(history[0]['values'][1:8]))):
                raise ValueError('Cached update violates objective guards')
            params=updated
            save('epoch001_coupled_low_moment_latent.npz');save('last_coupled_low_moment_latent.npz');report()
            record.stage('epoch_one_replayed',loss=history[1]['values'][0]);start=2
        if args.cached_second_trials:
            cache_root=args.cached_second_trials
            cache_meta=json.loads((cache_root/'run.json').read_text())
            cache_report=json.loads((cache_root/'report.json').read_text())
            if cache_meta['status']=='running':raise ValueError('Second-trial cache must be stopped')
            if not same_model_config(cache_report['configuration']):raise ValueError('Second-trial model configuration differs')
            if cache_meta['configuration']['case_order']!=[n for b in batches for n in b]:raise ValueError('Second-trial family differs')
            for path,digest in cache_meta['sources'].items():
                if Path(path).resolve()==Path(__file__).resolve():continue
                if file_digest(Path(path))!=digest:raise ValueError(f'Second-trial numerical source changed: {path}')
            with np.load(cache_root/'epoch001_coupled_low_moment_latent.npz') as z:
                if set(z.files)!=set(arrays)|set(constants)|set(params):raise ValueError('Second-trial origin keys differ')
                if any(not np.array_equal(np.asarray(v),z[k]) for k,v in {**arrays,**constants,**params}.items()):raise ValueError('Second-trial origin differs')
            with np.load(cache_root/'epoch002_work/gradient_progress.npz') as z:
                if int(z['completed_batches'])!=len(batches):raise ValueError('Incomplete second gradient')
                gradient={k:np.asarray(z[k]) for k in params};rows=np.asarray(z['values'])
            if rows.shape!=(len(batches),13) or not np.isfinite(rows).all() or any(not np.isfinite(v).all() for v in gradient.values()):raise ValueError('Invalid second gradient')
            before=rows.mean(axis=0)
            if not np.allclose(before,history[-1]['values'],rtol=2e-5,atol=1e-10):raise ValueError('Second-trial loss is not stationary')
            direction,first_step,slope=propose(gradient,before[0],metric)
            old_second=next(h for h in cache_report['history'] if h['epoch']==2)
            with np.load(cache_root/'epoch002_coupled_low_moment_latent.npz') as z:
                if any(not np.array_equal(np.asarray(params[k]+old_second['step']*jnp.asarray(direction[k])),z[k]) for k in params):raise ValueError('Original second-update replay differs')
            for attempt,path in enumerate(sorted((cache_root/'epoch002_work').glob('trial*.json'))):
                trial=json.loads(path.read_text());step=trial['step'];after=np.asarray(trial['values']);measured=np.asarray(trial['batch_values'])
                if step!=first_step/(2**attempt):raise ValueError('Second-trial step sequence differs')
                if measured.shape!=(len(batches),13) or not np.array_equal(measured.mean(axis=0),after):raise ValueError('Second-trial aggregate differs')
                accepted=bool(np.isfinite(after).all() and after[0]<=before[0]-.1*step*slope and np.all(after[1:8]<=guard_limits(history[0]['values'][1:8])))
                if not accepted:continue
                params={k:params[k]+step*jnp.asarray(direction[k]) for k in params}
                history.append({'epoch':2,'values':after.tolist(),'step':step,'replayed_trial_source':str(path)})
                folder=args.outdir/'epoch002_work';folder.mkdir()
                _atomic_savez(folder/'gradient_progress.npz',{**gradient,'completed_batches':np.asarray(len(batches)),'values':rows})
                atomic_json(folder/'replayed_trial.json',{'source':str(path),'source_accepted':trial['accepted'],'accepted_under_calibrated_guard':True,'limits':guard_limits(history[0]['values'][1:8]).tolist()})
                save('epoch002_coupled_low_moment_latent.npz');save('last_coupled_low_moment_latent.npz');report()
                record.stage('second_trial_replayed',loss=after[0],source=str(path));start=3
                break
            else:raise ValueError('No recorded second trial passes calibrated guards')
        if args.cached_first_step:
            if any(not np.array_equal(np.asarray(params[k]),np.asarray(stored[k],dtype=np.float64)) for k in params):raise ValueError('Random origin differs')
            meta=json.loads((args.cached_first_step/'run.json').read_text())
            if meta['configuration']['origin_sha256']!=file_digest(origin):raise ValueError('Cache origin mismatch')
            t=json.loads((args.cached_first_step/'trials.json').read_text());a=t['attempts'][-1]
            if t['case_order']!=[n for b in batches for n in b] or not a['accepted']:raise ValueError('Cache family/acceptance mismatch')
            base7=np.r_[np.asarray(t['base_fields']).reshape(16,3,2).mean(axis=0).ravel(),t['baseline'][3]]
            if not np.array_equal(base7,scales):raise ValueError('Cache normalization differs')
            with np.load(args.cached_direction) as d:updated={k:params[k]+a['step']*jnp.asarray(d[k]) for k in params}
            with np.load(args.cached_first_step/'diagnostic_candidate.npz') as z:
                if any(not np.array_equal(np.asarray(v),z[k]) for k,v in {**arrays,**updated}.items()):raise ValueError('Cache replay differs')
            before=np.r_[1.,base7,t['baseline']];after=np.r_[a['balanced_loss'],a['regime_values'],a['losses']]
            if not(after[0]<before[0] and np.all(after[1:8]<=before[1:8])):raise ValueError('Cache step violates guard')
            params=updated;history=[{'epoch':0,'values':before.tolist()},{'epoch':1,'values':after.tolist(),'step':a['step'],'exact_cached_replay':True}]
            save('epoch001_coupled_low_moment_latent.npz');save('last_coupled_low_moment_latent.npz');report();start=2
            record.stage('random_first_step_replayed',loss=after[0])
        for epoch in range(start,args.epochs+1):
            folder=args.outdir/f'epoch{epoch:03d}_work';folder.mkdir()
            gradient={k:np.zeros_like(v) for k,v in params.items()};rows=[]
            if epoch==1 and args.cached_initial_gradient:
                cache_root=args.cached_initial_gradient
                cache_meta=json.loads((cache_root/'run.json').read_text())
                cache_report=json.loads((cache_root/'report.json').read_text())
                if cache_meta['status'] not in ('training_complete_unvalidated','no_accepted_step_unvalidated'):
                    raise ValueError('Wait for the gradient source run to terminate')
                # The optimizer/controller may change when reusing a gradient;
                # its numerical objective, model and input sources may not.
                for path,digest in cache_meta['sources'].items():
                    if Path(path).resolve()==Path(__file__).resolve():continue
                    if file_digest(Path(path))!=digest:raise ValueError(f'Gradient source changed: {path}')
                without_metric=lambda d:{k:v for k,v in d.items() if k not in ('parameter_metric','guard_reference_sha256')}
                if without_metric(cache_report['configuration'])!=without_metric(config):
                    raise ValueError('Cached gradient configuration differs')
                if cache_meta['configuration']['case_order']!=[n for b in batches for n in b]:
                    raise ValueError('Cached gradient case order differs')
                with np.load(cache_root/'initial_coupled_low_moment_latent.npz') as z:
                    if set(z.files)!=set(arrays)|set(constants)|set(params):raise ValueError('Gradient origin keys differ')
                    if any(not np.array_equal(np.asarray(v),z[k]) for k,v in {**arrays,**constants,**params}.items()):
                        raise ValueError('Gradient random origin differs')
                with np.load(cache_root/'epoch001_work/gradient_progress.npz') as z:
                    if int(z['completed_batches'])!=len(batches):raise ValueError('Incomplete gradient cache')
                    gradient={k:np.asarray(z[k]) for k in params};rows=list(np.asarray(z['values']))
                if np.asarray(rows).shape!=(len(batches),13) or not np.all(np.isfinite(rows)) or any(not np.all(np.isfinite(v)) for v in gradient.values()):
                    raise ValueError('Invalid cached gradient')
                if not np.array_equal(np.mean(rows,axis=0),np.asarray(cache_report['history'][0]['values'])):
                    raise ValueError('Cached origin loss differs')
                _atomic_savez(folder/'gradient_progress.npz',{**gradient,'completed_batches':np.asarray(len(batches)),'values':np.asarray(rows)})
                record.stage('initial_gradient_reused',source=str(cache_root))
            for i,names in enumerate(batches if not rows else []):
                result,g=grad(params,data(names));rows.append(row(result))
                for k in gradient:gradient[k]+=np.asarray(g[k])/len(batches)
                if not np.all(np.isfinite(rows[-1])) or any(not np.all(np.isfinite(v)) for v in gradient.values()):raise FloatingPointError('Nonfinite gradient or loss')
                _atomic_savez(folder/'gradient_progress.npz',{**gradient,'completed_batches':np.asarray(i+1),'values':np.asarray(rows)})
                record.stage('gradient_saved',epoch=epoch,batch=i)
            before=np.mean(rows,axis=0)
            if not history:history.append({'epoch':0,'values':before.tolist()});report()
            elif not np.allclose(before,history[-1]['values'],rtol=2e-5,atol=1e-10):raise ValueError('Stationary loss mismatch')
            direction,step,slope=propose(gradient,before[0],metric)
            accepted=False
            for attempt in range(3):
                candidate={k:params[k]+step*jnp.asarray(direction[k]) for k in params}
                measured=[row(value(candidate,data(names))) for names in batches]
                after=np.mean(measured,axis=0)
                accepted=bool(np.all(np.isfinite(after)) and after[0]<=before[0]-.1*step*slope and np.all(after[1:8]<=guard_limits(history[0]['values'][1:8])))
                atomic_json(folder/f'trial{attempt}.json',{'step':step,'values':after.tolist(),'batch_values':np.asarray(measured).tolist(),'accepted':accepted})
                record.stage('trial',epoch=epoch,step=step,loss=after[0],accepted=accepted)
                if accepted:break
                step*=.5
            if not accepted:record.finish('no_accepted_step_unvalidated',epoch=epoch);return
            params=candidate;history.append({'epoch':epoch,'values':after.tolist(),'step':step})
            save(f'epoch{epoch:03d}_coupled_low_moment_latent.npz');save('last_coupled_low_moment_latent.npz');report()
            record.stage('epoch_saved',epoch=epoch,loss=after[0])
        record.finish('training_complete_unvalidated')
    except BaseException as exc:
        record.finish('failed',error=repr(exc));raise

if __name__=='__main__':main()
