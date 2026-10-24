"""CPU-only single-panel batch-size diagnostics, with external legends."""
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
COLORS={16:'#333333',8:'#007F73'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def canvas(title,ylabel,note):
    fig,axis=plt.subplots(figsize=(9.3,4.8))
    fig.subplots_adjust(left=.10,right=.72,top=.88,bottom=.23)
    axis.set_title(title,loc='left',fontsize=12,pad=12)
    axis.set_xlabel('Epoch');axis.set_ylabel(ylabel)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=8))
    axis.grid(axis='y',alpha=.22,linewidth=.6)
    axis.spines[['top','right']].set_visible(False);axis.margins(x=.015)
    fig.text(.10,.045,note,fontsize=8,color='#444444',va='bottom')
    return fig,axis


def line(axis,data,key,batch,split=None):
    selected=sorted([r for r in data if int(r['batch_size'])==batch],key=lambda r:int(r['epoch']))
    assert len(selected)==40
    label=('Batch16 (XP04)' if batch==16 else 'Batch8 (new)')+(' '+split if split else '')
    axis.plot([int(r['epoch']) for r in selected],[float(r[key]) for r in selected],
              color=COLORS[batch],linestyle='--' if split=='train' else '-',
              linewidth=1.2 if split=='train' else 1.6,alpha=.75 if split=='train' else 1.,label=label)


def main():
    summary=json.loads((ROOT/'results/metrics_summary.json').read_text())
    assert summary['formal_training_complete'] and summary['new_runs_completed']==1
    assert summary['test_forward_passes']==0
    curves=csv_read(ROOT/'results/curves.csv');losses=csv_read(ROOT/'results/loss_curves.csv')
    output=ROOT/'results/figures';output.mkdir(parents=True,exist_ok=True);artifacts=[]
    def save(fig,axis,name):
        axis.legend(fontsize=8.5,loc='center left',bbox_to_anchor=(1.025,.5),frameon=False,borderaxespad=0)
        for extension in ('png','svg'):
            path=output/(name+'.'+extension);fig.savefig(path,dpi=180 if extension=='png' else None,facecolor='white')
            artifacts.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size})
        plt.close(fig)
    fig,axis=canvas('MELD | seed 2024 | LR fixed at 1e-4','Weighted F1 (%)',
                    'Dashed: online train. Solid: full-pool validation. Dots: first maximum validation epoch.\n'
                    'Both training and validation grouping change with --batch-size; no test evaluation.')
    for batch in (16,8):
        for split in ('train','validation'):line(axis,curves,split+'_weighted_f1',batch,split)
        selected=[r for r in curves if int(r['batch_size'])==batch];best=max(selected,key=lambda r:float(r['validation_weighted_f1']))
        axis.scatter([int(best['epoch'])],[float(best['validation_weighted_f1'])],s=26,color=COLORS[batch],edgecolor='white',linewidth=.6,zorder=5)
    save(fig,axis,'MELD_batch_size_wf1')
    fig,axis=canvas('MELD | seed 2024 | classification loss','Classification loss',
                    'Original batch-mean losses: cross-batch-size values are not one sample-normalized objective.\n'
                    'Train is online; validation is fixed-model eval. Curve separation is not a strict generalization gap.')
    for batch in (16,8):
        for split in ('train','validation'):line(axis,curves,'train_loss' if split=='train' else 'validation_loss',batch,split)
    save(fig,axis,'MELD_batch_size_classification_loss')
    training=[r for r in losses if r['split']=='train']
    for name,key,title in [('weighted_loss','total_loss','Total weighted training loss'),
                           ('weighted_cc_loss','weighted_cc_loss','Weighted CC loss'),
                           ('weighted_mscl_loss','weighted_mscl_loss','Weighted MSCL loss'),
                           ('weighted_cscl_loss','weighted_cscl_loss','Weighted CSCL loss'),
                           ('raw_cc_loss','cc_loss','Raw CC loss'),
                           ('raw_mscl_loss','mscl_loss','Raw MSCL loss'),
                           ('raw_cscl_loss','cscl_loss','Raw CSCL loss')]:
        fig,axis=canvas('MELD | seed 2024 | '+title,title,
                        'Original coefficients and loss formulas; training batch means, not sample-normalized totals.\n'
                        'Batch changes BN, contrastive pairs, CC scale and optimizer steps; LR remains 1e-4.')
        for batch in (16,8):line(axis,training,key,batch)
        save(fig,axis,'MELD_batch_size_'+name)
    fig,axis=canvas('MELD | seed 2024 | optimizer steps per epoch','Optimizer steps',
                    'Batch8: recorded after real optimizer steps. Batch16: CPU reconstruction of historical batching.\n'
                    'Same epoch budget and LR do not imply the same number of updates or regularization trajectory.')
    for batch in (16,8):line(axis,summary['optimizer_step_records'],'optimizer_steps',batch)
    save(fig,axis,'MELD_batch_size_optimizer_steps')
    manifest={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'formal_training_complete':True,
              'matplotlib_version':matplotlib.__version__,'legend_location':'outside axes, reserved right margin',
              'scope':'CPU train/validation CSV plotting only; no model, test or scheduler calls',
              'input_sha256':{name:sha(ROOT/name) for name in ['results/per_run.csv','results/comparison.csv','results/curves.csv','results/loss_curves.csv','results/metrics_summary.json','scripts/plot_batch_size.py']},
              'artifacts':artifacts}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'plots':len(artifacts)//2,'artifacts':len(artifacts)}))


if __name__=='__main__':main()
