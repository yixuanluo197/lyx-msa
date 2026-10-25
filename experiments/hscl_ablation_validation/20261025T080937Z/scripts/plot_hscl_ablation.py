"""CPU Matplotlib diagnostic plots; consumes only completed train/validation CSVs."""
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
VARIANTS=('XP04_full','XP06_wo_cc','XP12_wo_hscl')
COLORS={'XP04_full':'#333333','XP06_wo_cc':'#0072B2','XP12_wo_hscl':'#D55E00'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def canvas(title,ylabel,note,height=5.0):
    fig,axis=plt.subplots(figsize=(10.1,height));fig.subplots_adjust(left=.10,right=.69,top=.87,bottom=.23)
    axis.set_title(title,loc='left',fontsize=12,pad=12);axis.set_xlabel('Epoch');axis.set_ylabel(ylabel)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=8));axis.grid(axis='y',alpha=.22,linewidth=.6)
    axis.spines[['top','right']].set_visible(False);axis.margins(x=.015)
    fig.text(.10,.045,note,fontsize=8,color='#444444',va='bottom')
    return fig,axis


def line(axis,data,key,variant,split=None,label=None,style=None):
    selected=sorted([r for r in data if r['variant']==variant and r.get(key) not in (None,'')],key=lambda r:int(r['epoch']))
    if not selected:return
    assert len(selected)==40
    axis.plot([int(r['epoch']) for r in selected],[float(r[key]) for r in selected],color=COLORS[variant],
              linestyle=style or ('--' if split=='train' else '-'),linewidth=1.15 if split=='train' else 1.7,
              alpha=.75 if split=='train' else 1.,label=label or variant+(' '+split if split else ''))


def main():
    summary=json.loads((ROOT/'results/metrics_summary.json').read_text())
    assert summary['formal_training_complete'] and summary['new_runs_completed']==1 and summary['test_forward_passes']==0
    assert len(summary['per_run'])==3 and {r['variant'] for r in summary['per_run']}==set(VARIANTS)
    curves=csv_read(ROOT/'results/curves.csv');losses=csv_read(ROOT/'results/loss_curves.csv')
    output=ROOT/'results/figures';output.mkdir(parents=True,exist_ok=True);artifacts=[]
    def save(fig,axis,name):
        axis.legend(fontsize=8,loc='center left',bbox_to_anchor=(1.025,.5),frameon=False,borderaxespad=0)
        for extension in ('png','svg'):
            path=output/(name+'.'+extension);fig.savefig(path,dpi=180 if extension=='png' else None,facecolor='white')
            artifacts.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size})
        plt.close(fig)
    for filename,field,ylabel in [('wf1','weighted_f1','Weighted F1 (%)'),('accuracy','accuracy','Accuracy (%)')]:
        fig,axis=canvas('MELD | seed 67137 | HSCL ablation',ylabel,
                        'Dashed: online train; solid: fixed full-pool validation. No new test evaluation.\n'
                        'XP12 vs XP04 is primary; XP06 w/o CC is a descriptive historical reference only.')
        for variant in VARIANTS:
            for split in ('train','validation'):line(axis,curves,split+'_'+field,variant,split)
            if field=='weighted_f1':
                selected=[r for r in curves if r['variant']==variant];best=max(selected,key=lambda r:float(r['validation_weighted_f1']))
                axis.scatter([int(best['epoch'])],[float(best['validation_weighted_f1'])],s=26,color=COLORS[variant],edgecolor='white',linewidth=.6,zorder=5)
        save(fig,axis,'MELD_hscl_'+filename)
    fig,axis=canvas('MELD | seed 67137 | classification loss','Classification loss',
                    'Train is online; validation uses fixed epoch weights in eval mode, not a strict generalization gap.\n'
                    'New validation loss: watcher precision. Historical validation loss: original 4 decimal places.')
    for variant in VARIANTS:
        for split in ('train','validation'):line(axis,curves,split+'_loss',variant,split)
    save(fig,axis,'MELD_hscl_classification_loss')
    for filename,key,title in [('total_loss','total_loss','Total effective loss'),('raw_cc_loss','cc_loss','Raw CC loss'),
                               ('raw_mscl_loss','mscl_loss','Raw MSCL loss'),('raw_cscl_loss','cscl_loss','Raw CSCL loss')]:
        fig,axis=canvas('MELD | seed 67137 | '+title,title,
                        'XP12 still computes raw CC/MSCL/CSCL; effective MSCL/CSCL are zero.\n'
                        'Historical validation auxiliary/total losses were not recorded: absent curves are not zero.')
        for variant in VARIANTS:
            for split in ('train','validation'):line(axis,[r for r in losses if r['split']==split],key,variant,split)
        save(fig,axis,'MELD_hscl_'+filename)
    fig,axis=canvas('MELD | seed 67137 | effective auxiliary train losses','Weighted auxiliary loss',
                    'Train-only overlay of fixed-coefficient effective contributions; full train/validation values in CSV.\n'
                    'XP12 MSCL/CSCL=0 despite nonzero raw values; XP06 CC=0. Overlapping zero curves are expected.',height=5.8)
    for variant in VARIANTS:
        for key,style in [('weighted_cc_loss','-'),('weighted_mscl_loss','--'),('weighted_cscl_loss',':')]:
            line(axis,[r for r in losses if r['split']=='train'],key,variant,label=variant+' '+key.replace('weighted_',''),style=style)
    save(fig,axis,'MELD_hscl_effective_auxiliary_loss')
    fig,axis=canvas('MELD | seed 67137 | optimizer steps','Optimizer steps per epoch',
                    'XP12: actual observed steps. XP04/XP06: CPU reconstruction, not historical step logs.\n'
                    'Unchanged batch16 and curriculum; overlapping curves are expected.')
    for variant in VARIANTS:line(axis,summary['optimizer_step_records'],'optimizer_steps',variant)
    save(fig,axis,'MELD_hscl_optimizer_steps')
    assert len(artifacts)==18
    manifest={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'formal_training_complete':True,
              'matplotlib_version':matplotlib.__version__,'legend_location':'outside axes, reserved right margin',
              'scope':'CPU train/validation plotting; no model or test calls; XP06 descriptive only',
              'input_sha256':{name:sha(ROOT/name) for name in ['results/per_run.csv','results/comparison.csv','results/curves.csv','results/loss_curves.csv','results/metrics_summary.json','scripts/plot_hscl_ablation.py']},
              'analysis_inputs_sha256':summary['inputs_sha256'],'artifacts':artifacts}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'plots':9,'artifacts':18}))


if __name__=='__main__':main()
