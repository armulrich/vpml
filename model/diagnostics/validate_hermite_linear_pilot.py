"""Fixed heldout interface-window losses for the bounded Hermite pilot."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from model.train import interface_flux_rollout as tr
from model.train.interface_flux_data import load_sharded_reference
from model.diagnostics.hermite_linear_pilot import CUTOFFS, REGIME, write_json
import jax
import jax.numpy as jnp


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--epochs",default="0,1,2,3,4,5")
    args=parser.parse_args();out=args.run
    config=json.loads((out/"configuration.json").read_text())
    selected=json.loads((out/"selected_manifest.json").read_text())
    reference=load_sharded_reference(Path(config["reference_cache"]),selected,coeff_key=tr.interface_flux_rollout_coeff_key(65))
    with np.load(out/"normalization.npz") as f:
        stats={k:f[k] for k in ("input_mean","input_std","target_mean","target_std")}
        scale=float(f["linear_q_scale"])
    k=np.arange(129)*.5
    dataset,_=tr.build_interface_flux_rollout_qpair_dataset(
        reference,max_projection_order=65,Nv_targets=CUTOFFS,Nm=6,k_arr=k,
        linear_history_stride=20,nonlinear_history_stride=20,rollout_horizon=128,
        n_low=2,context_mode="none",store_training_pairs=False,k_scale=64,nv_scale=64,
        precomputed_training_stats=stats,source_Nx=1024,rollout_Nx=256)
    # Reuse the sampler with explicitly heldout indices. No optimizer is invoked.
    val={REGIME:{key:(dataset[REGIME][key.replace("train_","val_",1)]
                     if key.startswith("train_") else value)
                 for key,value in dataset[REGIME].items()}}
    sampling=tr.prepare_interface_flux_rollout_sampling_state(reference,val,
        max_projection_order=65,target_nvs=CUTOFFS,history_dtype=np.complex64,
        fourier_count=129,source_Nx=1024,rollout_Nx=256)
    loss,_=tr.make_interface_flux_rollout_batch_loss(
        regime_weights={REGIME:1.},Nm=6,k_scale=64.,nv_scale=64.,stats=stats,
        hidden_width=128,res_blocks=2,Nv_targets=CUTOFFS,train_regimes=(REGIME,),
        teacher_backend="grid_cubic_spline",teacher_Lx=4*math.pi,teacher_Nx=1024,
        teacher_Nv=8192,teacher_vmin=-8.,teacher_vmax=8.,teacher_dt=.01,
        teacher_proj_Nv=65,projection_quadrature_Nv=4096,n_low=2,context_mode="none",
        rollout_horizon=128,poisson_sign=1.,rollout_dealias_23=False,rollout_precision="float32",
        regime_q_loss_stds={REGIME:scale},equilibrium_centered=True,
        complex_normalization_mode="phase_isotropic",translation_augmented=True,rollout_Nx=256)
    batches={};functions={};indices_by_cutoff={}
    for n in CUTOFFS:
        idx=np.flatnonzero((val[REGIME]["train_anchor_target_nvs"]==n)&
                          np.isin(val[REGIME]["train_anchor_time_indices"],np.arange(0,10001,2000)))
        ci=sampling[REGIME]["train_anchor_case_indices"][idx]
        assert len(idx)==24 and all(selected["cases"][int(c)]["split"]=="heldout" for c in ci)
        indices_by_cutoff[n]=idx.tolist()
        batches[n]={REGIME:tr.sample_interface_flux_rollout_regime_batch(sampling,
            regime=REGIME,target_nv=n,rollout_horizon=128,batch_size=24,k_arr=k,
            rng=np.random.default_rng(20260921),complex_dtype=jnp.complex64,all_k_loss=True,
            selected_indices=idx,translation_augmentation=False)}
        functions[n]=jax.jit(loss.target_loss_fns[n])
    rows=[]
    for epoch in [int(e) for e in args.epochs.split(",")]:
        deadline=time.monotonic()+14400
        while not (out/f"epoch{epoch:03d}/rng_state.json").exists():
            if time.monotonic()>deadline:raise TimeoutError(f"Waiting for E{epoch}")
            time.sleep(5)
        learned=tr.load_learned_interface_closure_npz(out/f"epoch{epoch:03d}/interface_closure.npz")
        values={n:float(functions[n](learned.params,batches[n])[0]) for n in CUTOFFS}
        row=dict(epoch=epoch,loss=float(np.mean(list(values.values()))),by_cutoff=values)
        rows.append(row);print("[validation]",row,flush=True)
        write_json(out/"validation_progress.json",rows)
    write_json(out/"validation_windows.json",dict(rows=rows,
        definition="Mean canonical standardized q-window MSE over all six cutoffs; four heldout ICs at t=0,20,40,60,80,100. H128 dt0.01. Fixed unshifted windows, no updates.",
        anchor_indices=indices_by_cutoff))


if __name__=="__main__":main()
