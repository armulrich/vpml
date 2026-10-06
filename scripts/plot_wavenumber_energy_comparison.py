"""Reproduce E5/E8/E10 energy panels from saved trajectories, without inference.

Each column is a fundamental wavenumber. Formula titles use normalized weights
and phases from the run manifest. Three displayed decimals do not alter data.
Existing output figures and provenance records are never overwritten.
"""
import argparse
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator


def render(root, output, style='formulas', footer=False):
    ROOT = Path(root).resolve()
    TARGET = Path(output).resolve()
    if TARGET.suffix.lower() != '.png':
        raise ValueError('Output must be a PNG path')
    if style not in ('formulas', 'families'):
        raise ValueError('Unknown label style')
    OUT = TARGET.parent
    PROVENANCE = TARGET.with_name(TARGET.stem + '_provenance.json')
    for path in (TARGET, PROVENANCE):
        if path.exists():
            raise FileExistsError(f'Refusing to replace {path}')
    OUT.mkdir(parents=True, exist_ok=True)
    formulas = style == 'formulas'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    preserved = {str(p): sha(p) for p in OUT.glob('*.png')}
    cases = [c for c in json.loads((ROOT/'manifest.json').read_text())['cases'] if c['split']=='heldout']
    assert len(cases) == 21
    ks = [.30, .35, .40, .45, .50]
    regimes = ['linear_landau', 'nonlinear_landau_weak', 'nonlinear_landau_strong']
    updates = [(2960, 'E5', '#b68435'), (4736, 'E8', '#169c80'), (5920, 'E10', '#743b9d')]
    lookup = {(c['fundamental'],c['family'],c['regime']): c for c in cases}
    assert len(lookup)==21
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':'#666666','axes.linewidth':.7,'xtick.color':'#444444','ytick.color':'#444444'})
    fig, axes = plt.subplots(6,5,figsize=(26,21) if formulas else (22,16))
    if formulas:
        fig.subplots_adjust(left=.085,right=.985,bottom=.075,top=.815,wspace=.27,hspace=.66)
    else:
        fig.subplots_adjust(left=.115,right=.985,bottom=.09,top=.83,wspace=.28,hspace=.40)
    fig.text(.5,.978,'Autonomous held-out trajectories by fundamental wavenumber',ha='center',va='top',fontsize=24,weight='bold')
    handles = [Line2D([0],[0],color='#2463eb',lw=3,label='Kinetic reference')]+[Line2D([0],[0],color=color,lw=3,label=name) for _,name,color in updates]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.954),ncol=4,frameon=False,fontsize=17,handlelength=3.2,columnspacing=3)
    if formulas:
        fig.text(.5,.914,r'$f_0(x,v) = [1+\varepsilon\,s(\theta)]\,e^{-v^2/2}/\sqrt{2\pi},\qquad \theta=k_0x$',ha='center',fontsize=20,color='#303440')
        fig.text(.5,.888,'Each panel shows its spatial perturbation s(θ)  •  Electric energy, logarithmic scale  •  T = 0–120',ha='center',fontsize=13,color='#555555')
    else:
        fig.text(.5,.914,'21 held-out initial conditions  •  Full trajectories to T = 120  •  Electric energy on a logarithmic scale',ha='center',fontsize=12,color='#555555')
    inputs={}
    placements=[]
    for col,k in enumerate(ks):
        mid=sum([axes[0,col].get_position().x0,axes[0,col].get_position().x1])/2
        unseen=k in [.35,.45]
        fig.text(mid,.859 if formulas else .884,rf'$k_0 = {k:.2f}$',ha='center',fontsize=20,weight='bold',color='#c7671a' if unseen else '#222222')
        fig.text(mid,.841 if formulas else .861,'Unseen fundamental' if unseen else 'Training fundamental',ha='center',fontsize=12,color='#c7671a' if unseen else '#626571')
        for row in range(6):
            ax=axes[row,col]
            family='isolated' if row<3 else 'mixture'
            regime=regimes[row%3]
            c=lookup.get((k,family,regime))
            if c is None:
                ax.set_axis_off()
                continue
            harmonics=[int(round(m/k)) for m in c['modes']]
            assert harmonics == ([1] if family=='isolated' else ([1,2,3,4] if unseen else [1,2,4]))
            ref_t=ref_e=None
            for u,name,color in updates:
                p=ROOT/f'evaluation/update_{u:08d}'/f"{c['case_id']}.npz"
                inputs[str(p.relative_to(ROOT))]=sha(p)
                with np.load(p) as d:
                    t=d['times']; teacher=d['teacher_energy']; model=d['model_energy']
                    assert np.all(np.isfinite(model)) and np.all(model>0)
                    if ref_t is None:
                        ref_t=t.copy(); ref_e=teacher.copy()
                        ax.semilogy(t,teacher,color='#2463eb',lw=1.15)
                    else:
                        np.testing.assert_array_equal(ref_t,t)
                        np.testing.assert_array_equal(ref_e,teacher)
                    ax.semilogy(t,model,color=color,lw=.8,alpha=.85)
            ax.set_xlim(0,120)
            ax.set_xticks([0,40,80,120])
            ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
            ax.tick_params(labelsize=9,length=3)
            ax.grid(alpha=.15)
            if row==5: ax.set_xlabel('Time, t',fontsize=11)
            weights=np.array(c['mode_weights'],dtype=float)
            norm=1/np.sum(np.abs(weights))
            np.testing.assert_allclose(norm,c['shape_normalization'],rtol=1e-14)
            coeff=weights*norm
            phases=c['relative_phases']
            if family=='isolated':
                assert harmonics==[1] and coeff[0]==1 and phases[0]==0
                formula=r'$s(\theta)=\cos(\theta)$'
            else:
                terms=[]
                for a,j,phase in zip(coeff,harmonics,phases):
                    arg=r'\theta' if j==1 else str(j)+r'\theta'
                    terms.append(rf'{a:.3f}\cos({arg}+{phase:.3f})')
                formula=r'$s(\theta)\approx '+ '+'.join(terms[:2])+'$'+'\n'+r'$\qquad+'+'+'.join(terms[2:])+'$'
            if formulas:
                ax.set_title(formula,fontsize=10.5,pad=9,linespacing=1.5)
            elif row==3:
                ax.set_title(r'$J = \{'+','.join(map(str,harmonics))+r'\}$',fontsize=11,pad=7)

            placements.append({'row':row,'column':col,'case_id':c['case_id'],'fundamental':k,'family':family,'regime':regime,'harmonics':harmonics,'normalized_coefficients':coeff.tolist(),'phases_radians':phases,'display_formula':formula})
    labels=['Linear','Weak','Strong']
    eps=['0.001','0.03','0.3']
    for row in range(6):
        p=axes[row,0].get_position()
        y=(p.y0+p.y1)/2
        if not formulas:
            fig.text(.018,y+.024,'ISOLATED' if row<3 else 'MIXTURE',fontsize=12,weight='bold',color='#303440')
        fig.text(.018,y,labels[row%3],fontsize=16)
        fig.text(.018,y-.023,rf'$\varepsilon = {eps[row%3]}$',fontsize=12,color='#626571')
    for col in [0,2,4]:
        p=axes[1,col].get_position()
        fig.text((p.x0+p.x1)/2,(p.y0+p.y1)/2,('No held-out cases with\n'+r'$s(\theta)=\cos(\theta)$'+'\nat this fundamental') if formulas else 'No isolated held-out cases\nat this fundamental',ha='center',va='center',fontsize=11,color='#9a9da5',linespacing=1.6)
    y=(axes[2,0].get_position().y0+axes[3,0].get_position().y1)/2
    fig.add_artist(Line2D([.015,.985],[y,y],transform=fig.transFigure,color='#d5d8de',lw=1))
    if footer:
        if formulas:
            fig.text(.085,.037,r'Panel coefficients are $a_j=w_j/\sum_\ell|w_\ell|$. Displayed coefficients and phases are rounded to 3 decimals; phases are in radians.',fontsize=12,color='#454954')
            fig.text(.085,.019,'All curves use the original saved data. Vertical limits are set per panel; compare curves using the tick values.',fontsize=12,color='#626571')
        else:
            fig.text(.115,.047,r'Training fundamentals: unseen mixture $J=\{1,2,4\}$.  Unseen fundamentals: isolated $j=1$ and mixture $J=\{1,2,3,4\}$.',fontsize=12,color='#454954')
            fig.text(.115,.025,'Mixture structure, weights and phases vary across cases. Vertical limits are set per panel; compare curves using the tick values.',fontsize=11,color='#626571')
    assert len(placements)==21
    fig.savefig(TARGET,dpi=180,facecolor='white',bbox_inches=(None if footer else matplotlib.transforms.Bbox.from_extents(0,.9,26,21) if formulas else matplotlib.transforms.Bbox.from_extents(0,.85,22,16)))
    plt.close(fig)
    assert all(sha(Path(p))==digest for p,digest in preserved.items())
    record={'figure':TARGET.name,'figure_sha256':sha(TARGET),'all_21_cases_plotted_once':True,'model_evaluation_performed':False,'original_figures_unchanged':True,'placements':placements,'source_sha256':inputs,'manifest_sha256':sha(ROOT/'manifest.json'),'generator_sha256':sha(Path(__file__)), 'style':style,'footer':footer,'formula_display_decimals':3,'phases_unit':'radians'}
    PROVENANCE.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'figure':str(TARGET),'cases':len(placements),'source_files':len(inputs),'existing_figures_unchanged':True},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Saved E10 run root, including inherited E5 evaluation')
    parser.add_argument('--output', required=True, type=Path, help='New PNG path; existing files are refused')
    parser.add_argument('--style', choices=('formulas', 'families'), default='formulas')
    parser.add_argument('--footer', action='store_true', help='Include explanatory footer text')
    args = parser.parse_args()
    render(args.root, args.output, args.style, args.footer)


if __name__ == '__main__':
    main()
