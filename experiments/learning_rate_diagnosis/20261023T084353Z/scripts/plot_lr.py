"""CPU-only single-panel diagnostic plots from validated train/val CSV exports."""
from __future__ import annotations
import argparse
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
COLORS={5e-5:'#0072B2',1e-4:'#222222',2e-4:'#D55E00'}
LABELS={5e-5:'5e-5 (new)',1e-4:'1e-4 (XP04)',2e-4:'2e-4 (new)'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def canvas(title,ylabel,footnote):
    fig,axis=plt.subplots(figsize=(10.4,4.8))
    fig.subplots_adjust(left=.105,right=.74,top=.88,bottom=.22)
    axis.set_title(title,loc='left',fontsize=12,pad=12)
    axis.set_xlabel('Epoch')
    axis.set_ylabel(ylabel)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=8))
    axis.grid(axis='y',alpha=.23,linewidth=.6)
    axis.spines[['top','right']].set_visible(False)
    axis.margins(x=.015)
    fig.text(.105,.045,footnote,fontsize=8,color='#444444',va='bottom')
    return fig,axis


def plot_line(axis,rows,key,lr,split=None):
    if not rows:
        return
    rows=sorted(rows,key=lambda r:int(r['epoch']))
    label=LABELS[lr]+(' '+split if split else '')
    axis.plot([int(r['epoch']) for r in rows],[float(r[key]) for r in rows],
              color=COLORS[lr],linestyle='--' if split=='train' else '-',
              linewidth=1.25 if split=='train' else 1.65,alpha=.74 if split=='train' else 1.,label=label)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete',action='store_true')
    opts=parser.parse_args()
    summary=json.loads((ROOT/'results/metrics_summary.json').read_text())
    assert summary['formal_training_complete'] or opts.allow_incomplete,'Refuse final plots from incomplete training'
    curves=read_csv(ROOT/'results/curves.csv')
    losses=read_csv(ROOT/'results/loss_curves.csv')
    output=ROOT/'results/figures';output.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    def save(fig,name):
        for suffix in ('png','svg'):
            path=output/(name+'.'+suffix)
            fig.savefig(path,dpi=180 if suffix=='png' else None,facecolor='white')
            artifacts.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size})
        plt.close(fig)
    for dataset in ('IEMOCAP','MELD'):
        selected=[r for r in curves if r['dataset']==dataset]
        fig,axis=canvas(dataset+' | full CMERC, validation-selected protocol','Weighted F1 (%)',
                        'Dashed: online train; solid: validation at fixed epoch weights. No test curves.\n'
                        'XP04 1e-4 is a historical control on another physical GPU; one seed only.')
        for lr in (5e-5,1e-4,2e-4):
            rows=[r for r in selected if float(r['lr'])==lr]
            for split in ('train','validation'):
                plot_line(axis,rows,split+'_weighted_f1',lr,split)
            if rows:
                best=max(rows,key=lambda r:float(r['validation_weighted_f1']))
                axis.scatter([int(best['epoch'])],[float(best['validation_weighted_f1'])],s=24,
                             marker='o',color=COLORS[lr],edgecolor='white',linewidth=.6,zorder=5)
        axis.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1),framealpha=.9)
        save(fig,dataset+'_wf1')
        fig,axis=canvas(dataset+' | classification loss','Classification loss',
                        'Dashed: online train batch mean; solid: validation eval batch mean.\n'
                        'Mode/parameter timing differs; curve separation is not a strict generalization gap.')
        for lr in (5e-5,1e-4,2e-4):
            rows=[r for r in selected if float(r['lr'])==lr]
            for split in ('train','validation'):
                plot_line(axis,rows,'train_loss' if split=='train' else 'validation_loss',lr,split)
        axis.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1),framealpha=.9)
        save(fig,dataset+'_classification_loss')
        chosen=[r for r in losses if r['dataset']==dataset and r['split']=='train']
        specs=[('weighted_loss','total_loss','Total weighted training loss'),
               ('weighted_cc_loss','weighted_cc_loss','Weighted CC loss'),
               ('weighted_mscl_loss','weighted_mscl_loss','Weighted MSCL loss'),
               ('weighted_cscl_loss','weighted_cscl_loss','Weighted CSCL loss')]
        for filename,key,title in specs:
            fig,axis=canvas(dataset+' | '+title,title,
                            'Original fixed coefficients; batch-mean training statistics. No loss formula changes.\n'
                            'Adam L2 coefficient stays fixed; changing LR changes its effective update trajectory.')
            for lr in (5e-5,1e-4,2e-4):
                plot_line(axis,[r for r in chosen if float(r['lr'])==lr],key,lr)
            axis.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1),framealpha=.9)
            save(fig,dataset+'_'+filename)
    manifest={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'formal_training_complete':summary['formal_training_complete'],'matplotlib_version':matplotlib.__version__,
              'scope':'CPU CSV plots only, train/validation, no model calls or test artifacts',
              'input_sha256':{name:sha(ROOT/name) for name in ['results/curves.csv','results/loss_curves.csv','results/metrics_summary.json','scripts/plot_lr.py']},
              'artifacts':artifacts}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figures':len(artifacts)//2,'artifacts':len(artifacts),'complete':summary['formal_training_complete']}))


if __name__=='__main__':
    main()
