"""Plot preserved Hermite pilot and existing linear-only low-moment trajectories."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from model.diagnostics.hermite_linear_pilot import write_json, digest


def interp_complex(t,ts,f):
    return np.stack([np.interp(t,ts,f[:,k].real)+1j*np.interp(t,ts,f[:,k].imag)
                     for k in range(f.shape[1])],axis=1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True)
    out=parser.parse_args().run
    figdir=out/"figures";figdir.mkdir(exist_ok=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False})
    updates=[json.loads(s) for s in (out/"updates.jsonl").read_text().splitlines()]
    val=json.loads((out/"validation_windows.json").read_text())["rows"]
    epochmeans=[np.mean([r["loss"] for r in updates if r["epoch"]==e]) for e in range(1,6)]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.semilogy(range(1,6),epochmeans,"o-",color="#151b29",lw=2,label="Training: epoch mean")
    ax.semilogy([r["epoch"] for r in val],[r["loss"] for r in val],"s-",color="#863baa",lw=2,label="Validation: fixed heldout windows")
    ax.set(xlabel="Epoch (30 updates per epoch)",ylabel=r"$\mathcal{L}_q=\langle|\widehat q_\theta-\widehat q^*|^2\rangle/(2s_q^2)$",
           title="Linear-only Hermite interface closure",xticks=range(6))
    ax.yaxis.set_major_locator(LogLocator(base=10));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f"{v:g}" if v>=1 else f"{v:.8f}".rstrip("0").rstrip(".")))
    ax.grid(which="major",alpha=.22);ax.legend();fig.tight_layout()
    for ext in ["png","pdf"]:fig.savefig(figdir/f"loss_curve.{ext}",dpi=190)
    plt.close(fig)
    cases=[c["case_id"] for c in json.loads((out/"selected_manifest.json").read_text())["cases"] if c["split"]=="heldout"]
    traces={};sources=[]
    for label in ["closure_disabled","E0","E1","E2","E5"]:
        for cid in cases:
            path=out/"evaluations_historical_settings"/f"{label}_{cid}.npz"
            with np.load(path) as d:traces[(label,cid)]={k:d[k] for k in d.files}
    base=out.parent/"linear_only_pair_20260921_v1/local_evaluations"
    for label,arm,ep in [("History E2","history",2),("Latent E15","latent",15)]:
        for cid in cases:
            path=base/arm/f"epoch{ep:03d}/heldout_cases"/cid/"trajectory.npz"
            sources.append(dict(path=str(path),sha256=digest(path)))
            with np.load(path) as d:
                traces[(label,cid)]=dict(times=d["times"],field=d["model_E_hat"]/128,
                      energy=d["model_energy"],reference_energy=d["teacher_energy"],
                      reference_times=d["teacher_times"],reference_field=d["teacher_E_hat"]/128)
    # Historical fitted model provides context, not random-start evidence.
    histroot=out.parent/"interface_flux_H128_T120_landau60_split16x4_teacherNx1024_rolloutNx256_B64_steps30_E400/evaluation_cases"
    for cid in cases:
        path=histroot/cid/"cases/nv64_nonlinear_sweep_case.npz"
        sources.append(dict(path=str(path),sha256=digest(path)))
        with np.load(path) as d:
            nx=2*(d["E_hat_theta"].shape[1]-1)
            traces[("Historical mixed E400",cid)]=dict(times=d["times_theta"],field=d["E_hat_theta"]/nx,
                energy=d["energy_theta"])
    records=[];t=np.arange(2401)*.05
    for (label,cid),d in traces.items():
        ref=traces[("E5",cid)]
        f=interp_complex(t,d["times"],d["field"][:,1:5]);r=interp_complex(t,ref["times"],ref["reference_field"][:,1:5])
        e=np.interp(t,d["times"],d["energy"]);er=np.interp(t,ref["times"],ref["reference_energy"])
        for lo,hi in [(0,20),(20,60),(60,120),(0,120)]:
            ix=(t>=lo)&(t<=hi);finite=bool(np.isfinite(f[ix]).all() and np.isfinite(e[ix]).all())
            row=dict(model=label,case=cid,start=lo,end=hi,finite=finite)
            if finite:
                den=np.trapezoid(np.sum(abs(r[ix])**2,axis=1),t[ix])
                row.update(epsilon_E=float(np.sqrt(np.trapezoid(np.sum(abs(f[ix]-r[ix])**2,axis=1),t[ix])/den)),
                    log_energy_rmse=float(np.sqrt(np.mean((np.log10(np.maximum(e[ix],1e-30))-np.log10(np.maximum(er[ix],1e-30)))**2))))
            records.append(row)
    summary=[]
    for label in dict.fromkeys(k[0] for k in traces):
        for lo,hi in [(0,20),(20,60),(60,120),(0,120)]:
            rows=[r for r in records if r["model"]==label and r["start"]==lo and r["end"]==hi]
            row=dict(model=label,start=lo,end=hi,finite_cases=sum(r["finite"] for r in rows))
            if all(r["finite"] for r in rows):
                row.update({k:float(np.mean([r[k] for r in rows])) for k in ["epsilon_E","log_energy_rmse"]})
            summary.append(row)
    write_json(out/"common_comparison.json",dict(summary=summary,per_case=records,sources=sources,
        definition="Common t spacing 0.05 through T120; Fourier fields normalized by native Nx, modes1-4; full saved energy. Arithmetic mean over four linear heldout ICs. Different model objectives/exposure.",
        train_epoch_means=epochmeans,validation=val))
    palette={"closure_disabled":"#888888","E0":"#ccb5df","E1":"#ec9755","E2":"#36a193","E5":"#783aa8",
             "History E2":"#db8240","Latent E15":"#248b83","Historical mixed E400":"#aa5555"}
    specs=[("autonomous_training_progress",["closure_disabled","E0","E1","E2","E5"],"Hermite linear-only pilot: autonomous electric energy"),
           ("autonomous_comparison",["E5","History E2","Latent E15"],"Linear heldouts: Hermite pilot and preserved low-moment models"),
           ("autonomous_hermite_E5",["E5"],"Hermite E5: autonomous electric energy, 150 updates"),
           ("historical_comparison",["E5","Historical mixed E400"],"New linear pilot and historical mixed-regime Hermite model")]
    for filename,labels,title in specs:
        fig,axs=plt.subplots(4,1,figsize=(11,11),sharex=True)
        for ax,cid in zip(axs,cases):
            d=traces[("E5",cid)]
            ax.semilogy(d["times"],d["reference_energy"],color="#2865ee",lw=1.5,label="Kinetic reference")
            for label in labels:
                d=traces[(label,cid)]
                ax.semilogy(d["times"],np.where(np.isfinite(d["energy"]),np.maximum(d["energy"],1e-30),np.nan),
                            color=palette[label],lw=1.3,ls="--" if label=="closure_disabled" else "-",label=label.replace("closure_disabled","Closure disabled"))
            ax.set_xlim(0,120);ax.set_ylim(1e-28,1e-1);ax.grid(alpha=.18)
            ax.set_ylabel(r"$\mathcal{E}(t)$");ax.text(.012,.92,cid,transform=ax.transAxes,fontsize=10)
            e5=traces[("E5",cid)]
            bad=np.flatnonzero(~np.isfinite(e5["energy"]))
            if len(bad):ax.text(.98,.05,f"E5 becomes nonfinite at t={e5['times'][bad[0]]:.2f}",transform=ax.transAxes,ha="right",color=palette["E5"],fontsize=9)
        axs[0].legend(loc="upper right",fontsize=8,ncol=2)
        axs[-1].set_xlabel("Time");axs[-1].set_xticks(np.arange(0,121,20))
        fig.suptitle(title,y=.994,fontsize=14);fig.tight_layout(rect=[0,0,1,.975])
        for ext in ["png","pdf"]:fig.savefig(figdir/f"{filename}.{ext}",dpi=180)
        plt.close(fig)
    print(json.dumps([r for r in summary if r["start"]==0 and r["end"]==120],indent=2))


if __name__=="__main__":main()
