"""Bounded local reproduction of the historical interface trainer on linear ICs.

Uses the canonical loss, sampler, initialization and Adam implementation directly.
Reference shards are opened read-only. Every output directory must be new.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from model.train import interface_flux_rollout as tr
from model.train.interface_flux_data import load_sharded_reference, evaluate_manifest_case
import jax
import jax.numpy as jnp
from vpml.core import FourierHermiteIMEX, learned_boundary_flux_hat

REGIME = "linear_landau"
CUTOFFS = (6, 7, 12, 20, 36, 64)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_update(loss_fn):
    """Same historical update, with finite checks and norm telemetry."""
    @jax.jit
    def update(params, state, batch):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        norm = jnp.sqrt(sum(jnp.sum(jnp.abs(x)**2) for x in jax.tree.leaves(grads)))
        finite = tr._tree_all_finite((loss, grads))
        candidate, candidate_state = tr.adam_step(params, grads, state, 1e-4, grad_clip=.5)
        finite = finite & tr._tree_all_finite((candidate, candidate_state))
        result, next_state = jax.lax.cond(
            finite, lambda _: (candidate, candidate_state), lambda _: (params, state), None)
        delta = jnp.sqrt(sum(jnp.sum(jnp.abs(result[k]-params[k])**2) for k in params))
        return result, next_state, loss, norm, delta, finite
    return update


def setup(cache, out, normalization_path=None):
    manifest = json.loads((cache / "ic_manifest.json").read_text())
    selected = {**manifest, "cases": [c for c in manifest["cases"] if c["regime"] == REGIME]}
    assert sum(c["split"] == "train" for c in selected["cases"]) == 16
    assert sum(c["split"] == "heldout" for c in selected["cases"]) == 4
    coeff_key = tr.interface_flux_rollout_coeff_key(65)
    reference = load_sharded_reference(cache, selected, coeff_key=coeff_key)
    k = np.arange(129, dtype=float)*.5
    stats=None
    if normalization_path is not None:
        with np.load(normalization_path) as data:
            stats={key:data[key] for key in ("input_mean","input_std","target_mean","target_std")}
            scales={REGIME:float(data["linear_q_scale"])}
    print("[setup] reusing frozen normalization" if stats is not None else
          "[setup] computing linear training-only statistics", flush=True)
    dataset, stats_raw = tr.build_interface_flux_rollout_qpair_dataset(
        reference, max_projection_order=65, Nv_targets=CUTOFFS, Nm=6, k_arr=k,
        linear_history_stride=20, nonlinear_history_stride=20, rollout_horizon=128,
        n_low=2, context_mode="none", store_training_pairs=False, k_scale=64., nv_scale=64.,
        source_Nx=1024, rollout_Nx=256, precomputed_training_stats=stats)
    if stats is None:
        stats = tr.phase_isotropic_complex_training_stats(stats_raw, Nm=6, context_mode="none")
        scales = tr.interface_flux_rollout_regime_loss_stds(
            reference, dataset, max_projection_order=65, target_nvs=CUTOFFS, k_arr=k,
            rollout_horizon=128, source_Nx=1024, rollout_Nx=256)
    np.savez(out / "normalization.npz", **stats, linear_q_scale=scales[REGIME])
    write_json(out / "selected_manifest.json", selected)
    kwargs = dict(Nm=6, k_scale=64., nv_scale=64., stats=stats, hidden_width=128,
                  res_blocks=2, equilibrium_centered=True, complex_normalization_mode="phase_isotropic",
                  translation_augmented=True, Nv_targets=CUTOFFS, train_regimes=(REGIME,),
                  teacher_backend="grid_cubic_spline", teacher_Lx=4*math.pi, teacher_Nx=1024,
                  teacher_Nv=8192, teacher_vmin=-8., teacher_vmax=8., teacher_dt=.01,
                  teacher_proj_Nv=65, projection_quadrature_Nv=4096, n_low=2,
                  rollout_horizon=128, rollout_Nx=256)
    loss, _ = tr.make_interface_flux_rollout_batch_loss(
        **kwargs, regime_weights={REGIME: 1.}, context_mode="none", poisson_sign=1.,
        rollout_dealias_23=False, rollout_precision="float32", regime_q_loss_stds=scales)
    sampling = tr.prepare_interface_flux_rollout_sampling_state(
        reference, dataset, max_projection_order=65, target_nvs=CUTOFFS,
        history_dtype=np.complex64, fourier_count=129, source_Nx=1024, rollout_Nx=256)
    print("[setup] statistics complete", scales, flush=True)
    return selected, kwargs, loss, sampling, k


def restore(directory):
    """Restore parameters, full Adam state, exposure and sampler state."""
    with np.load(directory/"training_state.npz") as data:
        params={key[8:]:jnp.asarray(data[key]) for key in data.files if key.startswith("params__")}
        state={"step":jnp.asarray(data["accepted_updates"],dtype=jnp.int32),
               "m":{key[3:]:jnp.asarray(data[key]) for key in data.files if key.startswith("m__")},
               "v":{key[3:]:jnp.asarray(data[key]) for key in data.files if key.startswith("v__")}}
        exposure=data["exposure"].copy()
    learned=tr.load_learned_interface_closure_npz(directory/"interface_closure.npz")
    for key in params:np.testing.assert_array_equal(params[key],learned.params[key])
    rng=np.random.default_rng()
    rng.bit_generator.state=json.loads((directory/"rng_state.json").read_text())
    rows=json.loads((directory/"training_rows.json").read_text())
    assert len(rows)==int(state["step"])
    return params,state,rng,exposure,rows


def sample(sampling,k,rng,step):
    n=CUTOFFS[step%len(CUTOFFS)]
    indices=tr.select_interface_flux_rollout_regime_indices(
        sampling,regime=REGIME,target_nv=n,batch_size=64,rng=rng,all_k_loss=True)
    case_indices=sampling[REGIME]["train_anchor_case_indices"][indices]
    batch=tr.sample_interface_flux_rollout_regime_batch(
        sampling,regime=REGIME,target_nv=n,rollout_horizon=128,batch_size=64,
        k_arr=k,rng=rng,complex_dtype=jnp.complex64,all_k_loss=True,
        selected_indices=indices,translation_augmentation=True,domain_length=4*math.pi)
    return n,indices,case_indices,{REGIME:batch}


def verify_resume(source,target,updates,sampling,k):
    p,s,rng,exposure,_=restore(source)
    expected_p,expected_s,expected_rng,expected_exposure,rows=restore(target)
    first,last=int(s["step"]),int(expected_s["step"])
    for i in range(first,last):
        n,indices,cases,batch=sample(sampling,k,rng,i)
        np.testing.assert_array_equal(indices,rows[i]["anchor_indices"])
        p,s,value,_,_,finite=updates[n](p,s,batch)
        if not bool(finite):raise FloatingPointError("Resume replay became nonfinite")
        np.testing.assert_array_equal(np.asarray(value),np.asarray(rows[i]["loss"]))
        exposure+=np.bincount(cases,minlength=20)
    for actual,expected in zip(jax.tree.leaves((p,s)),jax.tree.leaves((expected_p,expected_s))):
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(exposure,expected_exposure)
    assert rng.bit_generator.state==expected_rng.bit_generator.state
    return dict(source=str(source),target=str(target),replayed_updates=last-first,
                parameters_bitwise_equal=True,adam_bitwise_equal=True,rng_equal=True,
                sampled_anchors_equal=True,losses_bitwise_equal=True,exposure_equal=True)


def train(args):
    out, cache = args.outdir.resolve(), args.reference_cache.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    resume=None if args.resume_from is None else args.resume_from.resolve()
    if args.epochs<1:raise ValueError("epochs must be positive")
    source_files = [Path(__file__), root/"model/train/interface_flux_rollout.py",
                    root/"model/train/interface_flux_data.py", root/"vpml/core.py"]
    config = dict(seed=0, epochs=args.epochs, steps_per_epoch=30, batch_size=64, horizon=128,
                  learning_rate=.0001, gradient_clip=.5, update_norm_cap=None,
                  teacher_Nx=1024, teacher_Nv=8192, projection_quadrature_Nv=4096,
                  rollout_Nx=256, dt=.01, evaluation_dt=.01, T=120, cutoffs=list(CUTOFFS),
                  training_precision="float32", parameter_precision="float64",
                  initialization="canonical random seed 0, no trained weights",
                  reference_cache=str(cache), backend=jax.default_backend(),
                  source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                  source_hashes={str(p):digest(p) for p in source_files})
    if resume is not None:
        prior=json.loads((resume.parent/"configuration.json").read_text())
        for key in ("seed","steps_per_epoch","batch_size","horizon","learning_rate",
                    "gradient_clip","update_norm_cap","teacher_Nx","teacher_Nv",
                    "projection_quadrature_Nv","rollout_Nx","dt","T","cutoffs",
                    "training_precision","parameter_precision","reference_cache"):
            if config[key]!=prior[key]:raise ValueError(f"Resume configuration mismatch: {key}")
        for filename in ("model/train/interface_flux_rollout.py","model/train/interface_flux_data.py","vpml/core.py"):
            path=str(root/filename)
            if prior["source_hashes"][path]!=digest(path):raise ValueError(f"Scientific source changed: {filename}")
        config.update(resume_from=str(resume),initialization="exact continuation of saved optimizer and sampler state",
                      original_random_seed=0,source_run=str(resume.parent))
        originals_prefix=[f for f in resume.parent.rglob("*") if f.is_file()]
        write_json(out/"prefix_hashes.json",{str(f):digest(f) for f in originals_prefix})
        # Read-only ancestry copy supplies the original initialization to evaluators.
        shutil.copytree(resume.parent/"epoch000",out/"epoch000")
    write_json(out/"configuration.json", config)
    # Small metadata hashes and full projected-shard fingerprints. No rewriting cache.
    manifest = json.loads((cache/"ic_manifest.json").read_text())
    originals = [cache/"metadata.json", cache/"ic_manifest.json"] + [
        cache/"cases"/(c["case_id"]+".npy") for c in manifest["cases"] if c["regime"] == REGIME]
    print("[setup] hashing preserved linear reference shards", flush=True)
    hashes = {str(p): digest(p) for p in originals}
    write_json(out/"source_hashes.json", hashes)
    selected, kwargs, loss, sampling, k = setup(cache, out,
        None if resume is None else resume.parent/"normalization.npz")
    params = tr.init_interface_closure_params(jax.random.PRNGKey(0), input_dim=16,
                                             hidden_width=128, res_blocks=2)
    replay = tr.init_interface_closure_params(jax.random.PRNGKey(0), input_dim=16,
                                             hidden_width=128, res_blocks=2)
    assert all(np.array_equal(params[key], replay[key]) for key in params)
    state = tr.adam_init(params)
    rng = np.random.default_rng(0)
    updates = {n: make_update(loss.target_loss_fns[n]) for n in CUTOFFS}
    exposure = np.zeros(20, dtype=int)
    rows = []
    if resume is not None:
        params,state,rng,exposure,rows=restore(resume)
        if int(state["step"])%30:raise ValueError("Resume must be at a complete epoch")
        if int(state["step"])>=args.epochs*30:raise ValueError("Stop epoch must exceed resume epoch")
    if args.verify_resume_from is not None:
        if resume is None:raise ValueError("Resume parity requires --resume-from")
        print("[preflight] replaying saved epoch for exact resume parity",flush=True)
        check=verify_resume(args.verify_resume_from.resolve(),resume,updates,sampling,k)
        write_json(out/"resume_parity.json",check)
        print("[preflight] exact resume parity passed",flush=True)

    def save(epoch):
        dest=out/f"epoch{epoch:03d}"
        dest.mkdir(exist_ok=False)
        learned=tr.build_learned_interface_closure(params=params, **kwargs)
        learned=replace(learned, ic_manifest_sha256=selected["sha256"],
                        training_ic_count=16, heldout_ic_count=4)
        tr.save_learned_interface_closure_npz(dest/"interface_closure.npz", learned)
        np.savez(dest/"training_state.npz", **{f"params__{k}":np.asarray(v) for k,v in params.items()},
                 **{f"m__{k}":np.asarray(v) for k,v in state["m"].items()},
                 **{f"v__{k}":np.asarray(v) for k,v in state["v"].items()},
                 accepted_updates=int(state["step"]), exposure=exposure)
        write_json(dest/"rng_state.json", rng.bit_generator.state)
        write_json(dest/"training_rows.json", rows)
        print(f"[checkpoint] E{epoch}, accepted={int(state['step'])}", flush=True)
    start_step=int(state["step"])
    save(start_step//30)
    previous_elapsed=rows[-1]["elapsed_seconds"] if rows else 0.
    start=time.perf_counter()
    with (out/"updates.jsonl").open("x") as log:
        for i in range(start_step,args.epochs*30):
            n,indices,case_indices,batch=sample(sampling,k,rng,i)
            assert all(selected["cases"][int(c)]["split"]=="train" for c in case_indices)
            started=time.perf_counter()
            params,state,value,norm,delta,finite=updates[n](params,state,batch)
            value,norm,delta,finite=map(np.asarray,(value,norm,delta,finite))
            if not finite:
                raise FloatingPointError(f"Nonfinite update {i+1}; original state retained")
            exposure+=np.bincount(case_indices,minlength=20)
            row=dict(update=i+1,epoch=i//30+1,cutoff=n,loss=float(value),
                     gradient_norm=float(norm),gradient_scale=min(1.,.5/max(float(norm),1e-300)),
                     update_norm=float(delta),seconds=time.perf_counter()-started,
                     elapsed_seconds=previous_elapsed+time.perf_counter()-start,
                     anchor_indices=indices.tolist(),case_indices=case_indices.tolist())
            rows.append(row);log.write(json.dumps(row)+"\n");log.flush()
            print(f"[update] {i+1}/{args.epochs*30} N={n} loss={value:.7g} grad={norm:.3g} "
                  f"delta={delta:.3g} seconds={row['seconds']:.2f}",flush=True)
            if (i+1)%30==0:
                save((i+1)//30)
                write_json(out/"status.json",dict(pid=os.getpid(),state="training",completed_epoch=(i+1)//30,
                    accepted_updates=i+1,stop_epoch=args.epochs,last_loss=float(value)))
    write_json(out/"integrity.json", {p: digest(p)==h for p,h in hashes.items()})
    if resume is not None:
        hashes=json.loads((out/"prefix_hashes.json").read_text())
        write_json(out/"prefix_integrity.json",{p:digest(p)==h for p,h in hashes.items()})
    write_json(out/"status.json",dict(pid=os.getpid(),state="complete",completed_epoch=args.epochs,
        accepted_updates=int(state["step"]),stop_epoch=args.epochs))
    print(f"[complete] {args.epochs} epochs, {int(state['step'])} accepted updates",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir",type=Path,required=True)
    p.add_argument("--reference-cache",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--resume-from",type=Path)
    p.add_argument("--verify-resume-from",type=Path)
    train(p.parse_args())


if __name__=="__main__":
    main()
