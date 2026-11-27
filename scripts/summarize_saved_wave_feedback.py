"""Summarize all saved spatial modes and late energy, without model execution."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(folder, case):
    with np.load(folder/f"{case['case_id']}.npz",allow_pickle=False) as z:
        t=z['times'];pe=z['model_energy'];te=z['teacher_energy']
        ps=z['density_spectrum'].astype(np.float64);ts=z['teacher_density_spectrum'].astype(np.float64)
        fail=float(z['failure_t'])
    times=t[:len(pe)]
    result={'case_id':case['case_id'],'regime':case['regime'],'family':case['family'],
            'fundamental':case['fundamental'],'finite':not np.isfinite(fail),
            'failure_t':fail if np.isfinite(fail) else None}
    weights=1/np.arange(1,ps.shape[-1],dtype=np.float64)**2
    weights[-1]*=.5
    for start,end in [(20,40),(40,60),(60,80),(80,100),(100,120)]:
        mask=(times>=start)&(times<=end)
        if times[-1]<end or not mask.any():continue
        ms=ps[mask,1:];rs=ts[:len(pe)][mask,1:]
        if not np.isfinite(ms).all():continue
        with np.errstate(over='ignore',invalid='ignore'):
            pm=np.mean(ms**2,axis=0)*weights;tm=np.mean(rs**2,axis=0)*weights
        if not np.isfinite(pm).all():continue
        mode=int(np.argmax(pm))+1;refmode=int(np.argmax(tm))+1
        result[f'interval_{start}_{end}']={
            'model_mean_energy':float(np.mean(pe[mask])),
            'reference_mean_energy':float(np.mean(te[:len(pe)][mask])),
            'excess_energy_relative_to_initial':float(np.mean(pe[mask]-te[:len(pe)][mask])/max(te[0],1e-300)),
            'model_dominant_harmonic':mode,'reference_dominant_harmonic':refmode,
            'model_dominant_mode_fraction':float(pm[mode-1]/max(np.sum(pm),1e-300)),
            'reference_fraction_at_model_dominant_mode':float(tm[mode-1]/max(np.sum(tm),1e-300)),
            'top_model_modes':[{'harmonic':int(k+1),'physical_k':float((k+1)*case['fundamental']),
                                'fraction':float(pm[k]/max(np.sum(pm),1e-300))}
                               for k in np.argsort(pm)[-5:][::-1]]}
    if times[-1]>=120:
        mids=[];pm=[];tm=[]
        for start in range(60,120,10):
            mask=(times>=start)&(times<start+10)
            mids.append(start+5);pm.append(float(np.mean(pe[mask])));tm.append(float(np.mean(te[mask])))
        if all(np.isfinite(pm)) and min(pm)>0 and min(tm)>0:
            result['late_log_energy_slope']={'model':float(np.polyfit(mids,np.log(pm),1)[0]),
                                            'reference':float(np.polyfit(mids,np.log(tm),1)[0]),
                                            'interval':[60,120],'averaging_bin':10}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--updates',type=int,required=True);a=parser.parse_args()
    cases=[c for c in json.loads((a.root/'manifest.json').read_text())['cases'] if c['split']=='heldout']
    folder=a.root/'evaluation'/f'update_{a.updates:08d}'
    rows=[summarize(folder,c) for c in cases]
    out=a.root/'analysis';out.mkdir(exist_ok=True)
    path=out/f'spectral_feedback_{a.updates:08d}.json'
    with path.open('x') as f:
        json.dump({'updates':a.updates,'cases':rows,
                   'note':'All modes examined. Spectra were saved as float32 by the unchanged evaluator; dynamics use float64. These amplitude diagnostics omit phase and are not a linearized eigenvalue analysis.'},f,indent=2,allow_nan=False)
    for r in rows:
        if r['regime']=='linear_landau':
            print(json.dumps({'case_id':r['case_id'],'finite':r['finite'],'late':r.get('interval_100_120'),'slope':r.get('late_log_energy_slope')}))
    print(path)


if __name__=='__main__':main()
