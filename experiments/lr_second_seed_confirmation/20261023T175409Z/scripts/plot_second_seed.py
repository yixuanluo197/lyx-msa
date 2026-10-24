"""CPU-only train/validation diagnostic curves with legends outside the axes."""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
COLORS={1e-4:'#333333',2e-4:'#D55E00'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def canvas(title,ylabel,footnote):
    fig,axis=plt.subplots(figsize=(9.2,4.8))
    fig.subplots_adjust(left=.10,right=.72,top=.88,bottom=.23)
    axis.set_title(title,loc='left',fontsize=12,pad=12)
    axis.set_xlabel('Epoch');axis.set_ylabel(ylabel)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=8))
    axis.grid(axis='y',alpha=.22,linewidth=.6)
    axis.spines[['top','right']].set_visible(False)
    axis.margins(x=.015)
    fig.text(.10,.045,footnote,fontsize=8,color='#444444',va='bottom')
    return fig,axis


def line(axis,records,key,lr,split=None):
    selected=sorted([r for r in records if float(r['lr'])==lr],key=lambda r:int(r['epoch']))
    assert len(selected)==40
    label=('1e-4 (XP04)' if lr==1e-4 else '2e-4')+(' '+split if split else '')
    axis.plot([int(r['epoch']) for r in selected],[float(r[key]) for r in selected],
              color=COLORS[lr],linestyle='--' if split=='train' else '-',
              linewidth=1.2 if split=='train' else 1.6,alpha=.74 if split=='train' else 1.,label=label)


def main():
    summary=json.loads((ROOT/'results/metrics_summary.json').read_text())
    assert summary['formal_training_complete'] and summary['new_runs_completed']==1
    curves=csv_read(ROOT/'results/curves.csv');losses=csv_read(ROOT/'results/loss_curves.csv')
    output=ROOT/'results/figures';output.mkdir(parents=True,exist_ok=True);artifacts=[]
    def save(fig,axis,name):
        axis.legend(fontsize=8.5,loc='center left',bbox_to_anchor=(1.025,.5),frameon=False,borderaxespad=0)
        for extension in ('png','svg'):
            path=output/(name+'.'+extension);fig.savefig(path,dpi=180 if extension=='png' else None,facecolor='white')
            artifacts.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size})
        plt.close(fig)
    for seed in (67137,2024):
        selected=[r for r in curves if int(r['seed'])==seed]
        fig,axis=canvas(f'MELD | seed {seed} | full CMERC','Weighted F1 (%)',
                        'Dashed: online train. Solid: fixed-epoch validation. Dots: first maximum validation epoch.\n'
                        'No test evaluation. Seed 67137 selected the LR; seed 2024 is the confirmation run.')
        for lr in (1e-4,2e-4):
            for split in ('train','validation'):line(axis,selected,split+'_weighted_f1',lr,split)
            subset=[r for r in selected if float(r['lr'])==lr];best=max(subset,key=lambda r:float(r['validation_weighted_f1']))
            axis.scatter([int(best['epoch'])],[float(best['validation_weighted_f1'])],s=26,color=COLORS[lr],edgecolor='white',linewidth=.6,zorder=5)
        save(fig,axis,f'MELD_seed{seed}_wf1')
        fig,axis=canvas(f'MELD | seed {seed} | classification loss','Classification loss',
                        'Train: online batch mean. Validation: fixed model in eval mode.\n'
                        'Different modes and parameter timing: curve separation is not a strict generalization gap.')
        for lr in (1e-4,2e-4):
            for split in ('train','validation'):line(axis,selected,'train_loss' if split=='train' else 'validation_loss',lr,split)
        save(fig,axis,f'MELD_seed{seed}_classification_loss')
        train=[r for r in losses if int(r['seed'])==seed and r['split']=='train']
        for filename,key,title in [('weighted_loss','total_loss','Total weighted training loss'),
                                   ('weighted_cc_loss','weighted_cc_loss','Weighted CC loss'),
                                   ('weighted_mscl_loss','weighted_mscl_loss','Weighted MSCL loss'),
                                   ('weighted_cscl_loss','weighted_cscl_loss','Weighted CSCL loss'),
                                   ('raw_cc_loss','cc_loss','Raw CC loss'),
                                   ('raw_mscl_loss','mscl_loss','Raw MSCL loss'),
                                   ('raw_cscl_loss','cscl_loss','Raw CSCL loss')]:
            fig,axis=canvas(f'MELD | seed {seed} | {title}',title,
                            'Training batch means; original fixed coefficients and mathematical loss formulas.\n'
                            'Same L2 coefficient does not imply the same effective Adam regularization trajectory.')
            for lr in (1e-4,2e-4):line(axis,train,key,lr)
            save(fig,axis,f'MELD_seed{seed}_{filename}')
    manifest={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'formal_training_complete':True,
              'matplotlib_version':matplotlib.__version__,'legends':'outside axes, reserved right margin',
              'scope':'CPU CSV plotting, train/validation only; no model or test calls',
              'input_sha256':{name:sha(ROOT/name) for name in ['results/curves.csv','results/loss_curves.csv','results/metrics_summary.json','scripts/plot_second_seed.py']},'artifacts':artifacts}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'plots':len(artifacts)//2,'artifacts':len(artifacts)}))


if __name__=='__main__':main()
