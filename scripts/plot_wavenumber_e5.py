"""Plot saved E0-E5 window losses from the immutable v2 run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    run=args.run
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    snapshots=sorted((run/'training').glob('state_*.npz'))
    if not snapshots:raise FileNotFoundError('No saved training snapshot')
    with np.load(snapshots[-1],allow_pickle=False) as state:
        rows=state['loss_rows']
    epochs=sorted({int((update-1)//592)+1 for update in rows[:,0]})
    train=[];val=[];xs=[]
    e0=run/'evaluation/epoch_0000/report.json'
    if e0.exists():
        xs.append(0);train.append(np.nan)
        val.append(json.loads(e0.read_text())['fixed_validation_window_loss'])
    for epoch in epochs:
        selected=(rows[:,0]>(epoch-1)*592)&(rows[:,0]<=epoch*592)
        if not selected.any():continue
        xs.append(epoch);train.append(float(np.mean(rows[selected,1])))
        path=run/'evaluation'/f'epoch_{epoch:04d}'/'report.json'
        val.append(json.loads(path.read_text())['fixed_validation_window_loss'] if path.exists() else np.nan)
    fig,ax=plt.subplots(figsize=(10,5.8),layout='constrained')
    ax.plot(xs,train,color='#111827',marker='o',linewidth=2,label='Training, complete epoch mean')
    ax.plot(xs,val,color='#7037a5',marker='s',linewidth=2,label='Fixed held-out windows')
    ax.set_yscale('log');ax.set_xlabel('Epoch');ax.set_ylabel('Global relative trajectory loss')
    ax.set_xticks(xs)
    observed=np.asarray(train+val,float);observed=observed[np.isfinite(observed)&(observed>0)]
    if len(observed):
        lo,hi=np.log10(observed.min()),np.log10(observed.max())
        ticks=np.power(10,np.linspace(np.floor(lo)-.1,np.ceil(hi)+.1,6))
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.3f}' if x<1 else f'{x:.2f}'))
    ax.grid(alpha=.22);ax.legend(frameon=False)
    ax.set_title('Broader-wavenumber latent FNO: training and validation')
    out=run/'loss_train_validation_E5.png';fig.savefig(out,dpi=180);plt.close(fig)
    print(out)


if __name__=='__main__':main()
