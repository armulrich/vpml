"""Build an E0-to-final record from a completed, unmodified Hermite continuation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,FuncFormatter
from model.diagnostics.hermite_linear_pilot import write_json,digest
from model.diagnostics.evaluate_hermite_linear_pilot import metrics
from model.diagnostics.plot_hermite_linear_pilot import interp_complex


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    out=p.parse_args().run
    cfg=json.loads((out/"configuration.json").read_text());source=Path(cfg["source_run"])
    stop=cfg["epochs"];rows=json.loads((out/f"epoch{stop:03d}/training_rows.json").read_text())
    validation=json.loads((source/"validation_windows.json").read_text())["rows"]+json.loads((out/"validation_windows.json").read_text())["rows"]
    epochs=[5]+[int(x.name.split("_")[0][1:]) for x in (out/"evaluations_historical_settings").glob("E*_linear_landau_ic01.npz")]
    epochs=sorted(set(epochs));cases=[c["case_id"] for c in json.loads((out/"selected_manifest.json").read_text())["cases"] if c["split"]=="heldout"]
    curves={};scores=[];t=np.arange(2401)*.05
    for e in epochs:
        folder=(source if e==5 else out)/"evaluations_historical_settings"
        for cid in cases:
            with np.load(folder/f"E{e}_{cid}.npz") as d:curves[e,cid]={k:d[k] for k in d.files}
            d=curves[e,cid]
            f=interp_complex(t,d["times"],d["field"][:,:5]);r=interp_complex(t,d["times"],d["reference_field"][:,:5])
            energy=np.interp(t,d["times"],d["energy"]);reference=np.interp(t,d["times"],d["reference_energy"])
            row=metrics(t,f,r,energy,reference)[-2]
            assert row["start"]==0 and row["end"]==120 and row["band"]=="modes1_4"
            row.update(epoch=e,case=cid);scores.append(row)
    aggregate=[]
    for e in epochs:
        group=[r for r in scores if r["epoch"]==e]
        row=dict(epoch=e,finite_cases=sum(r["finite"] for r in group))
        if row["finite_cases"]==4:
            for k in ["epsilon_E","log_energy_rmse","amplitude_error","phase_rmse"]:
                row[k]=float(np.mean([r[k] for r in group]))
        aggregate.append(row)
    finite=[r for r in aggregate if r["finite_cases"]==4]
    best_energy=min(finite,key=lambda r:r["log_energy_rmse"])["epoch"]
    best_field=min(finite,key=lambda r:r["epsilon_E"])["epoch"]
    write_json(out/"continuation_summary.json",dict(aggregate=aggregate,per_case=scores,
        best_energy_epoch=best_energy,best_field_epoch=best_field,
        selection="One checkpoint across all four ICs; mean full T120 log-energy RMSE; compare field rank separately.",
        metric_grid_dt=.05,energy_floor_for_scoring=1e-30,
        training_epoch_means=[float(np.mean([r["loss"] for r in rows if r["epoch"]==e])) for e in range(1,stop+1)],
        validation=validation))
    figdir=out/"figures";figdir.mkdir(exist_ok=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
    fig,ax=plt.subplots(figsize=(11,5))
    ax.semilogy(range(1,stop+1),[np.mean([r["loss"] for r in rows if r["epoch"]==e]) for e in range(1,stop+1)],color="#182030",label="Training: epoch mean")
    ax.semilogy([v["epoch"] for v in validation],[v["loss"] for v in validation],"o-",color="#803aa0",label="Validation: fixed heldout windows")
    ax.set(xlabel="Epoch (30 updates per epoch)",ylabel=r"$\mathcal{L}_q=\langle|\widehat q_\theta-\widehat q^*|^2\rangle/(2s_q^2)$",title=f"Linear-only Hermite closure: E0–E{stop}")
    ax.yaxis.set_major_locator(LogLocator(base=10));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:.10f}".rstrip("0").rstrip(".")))
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    for ext in ["png","pdf"]:fig.savefig(figdir/f"loss_E0_E{stop}.{ext}",dpi=180)
    plt.close(fig)
    for filename,selected,title in [("autonomous_last",[stop],f"Final Hermite E{stop}"),
        ("autonomous_best_energy",[best_energy],f"Best saved energy envelope: Hermite E{best_energy}"),
        ("autonomous_progress",[e for e in [5,20,60,stop] if e in epochs],"Hermite continuation: autonomous progress")]:
        fig,axs=plt.subplots(4,1,figsize=(11,11),sharex=True)
        for ax,cid in zip(axs,cases):
            d=curves[5,cid];ax.semilogy(d["times"],d["reference_energy"],color="#2865ee",lw=1.3,label="Kinetic reference")
            for e in selected:
                d=curves[e,cid];ax.semilogy(d["times"],np.where(np.isfinite(d["energy"]),np.maximum(d["energy"],1e-30),np.nan),lw=1.2,label=f"Hermite E{e}")
            ax.set(xlim=(0,120),ylim=(1e-28,.1),ylabel=r"$\mathcal{E}(t)$")
            ax.text(.01,.92,cid,transform=ax.transAxes);ax.grid(alpha=.18)
        axs[0].legend(fontsize=9,ncol=2);axs[-1].set_xlabel("Time")
        fig.suptitle(title);fig.tight_layout(rect=[0,0,1,.97])
        for ext in ["png","pdf"]:fig.savefig(figdir/f"{filename}.{ext}",dpi=180)
        plt.close(fig)
    lines=["# Hermite linear-only continuation through E100","",
        "Completed locally from the exact E5 optimizer and RNG state. No Modal resources used. No training settings changed.","",
        "E4-to-E5 replay passed bitwise equality for weights, Adam state, sampled anchors, losses, exposure and RNG. The original E5 run is preserved.","",
        "| Epoch | Finite cases | Mean field error, modes 1–4 | Mean log-energy RMSE |","|---|---:|---:|---:|"]
    for r in aggregate:lines.append(f"| {r['epoch']} | {r['finite_cases']}/4 | {r.get('epsilon_E','nonfinite')} | {r.get('log_energy_rmse','nonfinite')} |")
    lines.extend(["",f"Best saved energy checkpoint: E{best_energy}. Best saved field checkpoint: E{best_field}.",
        "These ranks use complete T120 autonomous rollouts across all four heldouts. The short q-window training objective is distinct.","",
        "Original reference Nx1024/Nv8192 and projection quadrature4096 retained. Reduced Nx256, training dt0.01/H128, canonical cutoff cycle, Adam1e-4 and gradient clip0.5 retained. Evaluation dt0.005 with dealiasing retained.","",
        f"Total accepted updates: {stop*30}. All epochs retain weights, optimizer state and RNG state. The E0–E5 prefix was not retrained. The numerical replay was a reproducibility preflight, not additional optimization exposure.","",
        "Figures: loss_E0_E100.png, autonomous_last.png, autonomous_best_energy.png and autonomous_progress.png in figures/.","",
        "See continuation_summary.json for per-case measurements, status.json for training completion, and integrity.json/prefix_integrity.json for preservation checks."])
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    write_json(out/"COMPLETE.json",dict(completed_epoch=stop,best_energy_epoch=best_energy,best_field_epoch=best_field,
        report=str(out/"REPORT.md"),figures=str(figdir),checkpoint=str(out/f"epoch{stop:03d}/interface_closure.npz")))


if __name__=="__main__":main()
