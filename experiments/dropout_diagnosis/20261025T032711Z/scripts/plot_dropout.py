"""CPU-only diagnostic plots from validated dropout train/validation CSVs."""
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
COLORS={.4:'#333333',.5:'#007F73'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def canvas(title,ylabel,note):
    fig,axis=plt.subplots(figsize=(9.5,4.8));fig.subplots_adjust(left=.10,right=.71,top=.88,bottom=.23)
    axis.set_title(title,loc='left',fontsize=12,pad=12);axis.set_xlabel('Epoch');axis.set_ylabel(ylabel)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=8));axis.grid(axis='y',alpha=.22,linewidth=.6)
    axis.spines[['top','right']].set_visible(False);axis.margins(x=.015)
    fig.text(.10,.045,note,fontsize=8,color='#444444',va='bottom')
    return fig,axis


def line(axis,rows,key,dropout,split=None):
    selected=sorted([row for row in rows if float(row['dropout'])==dropout],key=lambda row:int(row['epoch']))
    if not selected:return
    assert len(selected)==40
    label=('p=0.4 (XP04)' if dropout==.4 else 'p=0.5 (new)')+(' '+split if split else '')
    axis.plot([int(row['epoch']) for row in selected],[float(row[key]) for row in selected],
              color=COLORS[dropout],linestyle='--' if split=='train' else '-',
              linewidth=1.2 if split=='train' else 1.7,alpha=.75 if split=='train' else 1.,label=label)


def main():
    summary=json.loads((ROOT/'results/metrics_summary.json').read_text())
    assert summary['formal_training_complete'] and summary['new_runs_completed']==1 and summary['test_forward_passes']==0
    curves=csv_read(ROOT/'results/curves.csv');losses=csv_read(ROOT/'results/loss_curves.csv')
    output=ROOT/'results/figures';output.mkdir(parents=True,exist_ok=True);artifacts=[]
    def save(fig,axis,name):
        axis.legend(fontsize=8.3,loc='center left',bbox_to_anchor=(1.025,.5),frameon=False,borderaxespad=0)
        for extension in ('png','svg'):
            path=output/(name+'.'+extension);fig.savefig(path,dpi=180 if extension=='png' else None,facecolor='white')
            artifacts.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size})
        plt.close(fig)
    for filename,field,ylabel in [('wf1','weighted_f1','Weighted F1 (%)'),('accuracy','accuracy','Accuracy (%)')]:
        fig,axis=canvas('MELD | seed 2024 | dropout 0.4 vs 0.5',ylabel,
                        'Dashed: online train; solid: full-pool validation. LR=1e-4, train/validation batch=16.\n'
                        'No test evaluation. Dropout probability also controls the active text GRU inter-layer dropout.')
        for p in (.4,.5):
            for split in ('train','validation'):line(axis,curves,split+'_'+field,p,split)
            if field=='weighted_f1':
                selected=[row for row in curves if float(row['dropout'])==p];best=max(selected,key=lambda row:float(row['validation_weighted_f1']))
                axis.scatter([int(best['epoch'])],[float(best['validation_weighted_f1'])],s=26,color=COLORS[p],edgecolor='white',linewidth=.6,zorder=5)
        save(fig,axis,'MELD_dropout_'+filename)
    fig,axis=canvas('MELD | seed 2024 | classification loss','Classification loss',
                    'Train is online; validation uses fixed epoch weights in eval mode.\n'
                    'Curve separation is not a strict generalization gap; historical validation loss is rounded to 4 decimals.')
    for p in (.4,.5):
        for split in ('train','validation'):line(axis,curves,'train_loss' if split=='train' else 'validation_loss',p,split)
    save(fig,axis,'MELD_dropout_classification_loss')
    for filename,key,title in [('total_loss','total_loss','Total weighted loss'),
                               ('raw_cc_loss','cc_loss','Raw CC loss'),('raw_mscl_loss','mscl_loss','Raw MSCL loss'),('raw_cscl_loss','cscl_loss','Raw CSCL loss'),
                               ('weighted_cc_loss','weighted_cc_loss','Weighted CC loss'),('weighted_mscl_loss','weighted_mscl_loss','Weighted MSCL loss'),('weighted_cscl_loss','weighted_cscl_loss','Weighted CSCL loss')]:
        fig,axis=canvas('MELD | seed 2024 | '+title,title,
                        'Original fixed coefficients; batch means. Train auxiliary models use epoch-entry snapshots.\n'
                        'Historical validation auxiliary/total losses were not recorded: absent curves are not zero values.')
        for p in (.4,.5):
            for split in ('train','validation'):line(axis,[row for row in losses if row['split']==split],key,p,split)
        save(fig,axis,'MELD_dropout_'+filename)
    fig,axis=canvas('MELD | seed 2024 | optimizer steps','Optimizer steps per epoch',
                    'p=0.5: observed after actual optimizer steps; p=0.4: CPU reconstruction of historical batching.\n'
                    'Same train/validation grouping and curriculum schedule; overlapping curves are expected.')
    for p in (.4,.5):line(axis,summary['optimizer_step_records'],'optimizer_steps',p)
    save(fig,axis,'MELD_dropout_optimizer_steps')
    manifest={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'formal_training_complete':True,'matplotlib_version':matplotlib.__version__,
              'legend_location':'outside axes, reserved right margin','scope':'CPU train/validation CSV plotting only; no model or test calls',
              'input_sha256':{name:sha(ROOT/name) for name in ['results/per_run.csv','results/comparison.csv','results/curves.csv','results/loss_curves.csv','results/metrics_summary.json','scripts/plot_dropout.py']},'artifacts':artifacts}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'plots':len(artifacts)//2,'artifacts':len(artifacts)}))


if __name__=='__main__':main()
