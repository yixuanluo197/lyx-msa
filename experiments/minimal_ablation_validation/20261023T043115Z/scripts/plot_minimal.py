"""Plot diagnostic single-panel curves from existing CSVs; never access GPUs."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def xy(rows, key):
    return [int(row['epoch']) for row in rows], [float(row[key]) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete',action='store_true')
    opts = parser.parse_args()
    summary = json.loads((ROOT/'results/metrics_summary.json').read_text())
    assert opts.allow_incomplete or summary['formal_training_complete']
    curves = read_csv(ROOT/'results/epoch_curves.csv')
    losses = read_csv(ROOT/'results/epoch_loss_curves.csv')
    output = ROOT/'results/figures'
    output.mkdir(exist_ok=True)
    files = {}
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False,'axes.axisbelow':True})

    def save(fig, name, footnote):
        fig.text(.5,.015,footnote,ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.055,1,1))
        for suffix in ['png','svg']:
            path=output/(name+'.'+suffix)
            fig.savefig(path,dpi=180,metadata={'Creator':'CMERC diagnostic plotting'})
            files[str(path.relative_to(ROOT))]=sha(path)
        plt.close(fig)

    by_id = {}
    for run in summary['per_run']:
        ident=run['run_id'];by_id[ident]={}
        for source in ['new','XP04']:
            by_id[ident][source]=sorted([r for r in curves if r['run_id']==ident and r['source']==source],key=lambda r:int(r['epoch']))
        fig,ax=plt.subplots(figsize=(8,4.8))
        for source,kind,color,style in [('new','validation','#0072B2','-'),('XP04','validation','#009E73','--'),
                                      ('new','train','#D55E00',':'),('XP04','train','#E69F00','-.')]:
            rows=by_id[ident][source]
            if rows:
                ax.plot(*xy(rows,kind+'_weighted_f1'),label=source+' '+kind,color=color,linestyle=style,lw=1.8)
        if run['complete']:
            ax.scatter([run['selected_epoch']],[run['selected_validation_weighted_f1']],s=28,color='#0072B2',zorder=5)
        ax.set(title=ident,xlabel='Epoch',ylabel='Weighted F1 (%)',ylim=(0,100))
        ax.grid(alpha=.18);ax.legend(loc='lower right',fontsize=9)
        save(fig,ident+'_wf1','Validation selection in both arms; one seed; historical control and hardware limits apply.')
        fig,ax=plt.subplots(figsize=(8,4.8))
        new=sorted([r for r in losses if r['run_id']==ident and r['source']=='new' and r['split']=='train'],key=lambda r:int(r['epoch']))
        old=sorted([r for r in losses if r['run_id']==ident and r['source']=='XP04' and r['split']=='train'],key=lambda r:int(r['epoch']))
        for key,label,color in [('classification_loss','New classification','#0072B2'),('weighted_cc','New weighted CC','#D55E00'),
                                ('weighted_mscl','New weighted MSCL','#009E73'),('weighted_cscl','New weighted CSCL','#CC79A7'),
                                ('total_loss','New total','#333333')]:
            if new:
                ax.plot(*xy(new,key),label=label,color=color,lw=1.8)
        if old:
            ax.plot(*xy(old,'total_loss'),label='XP04 total',color='#666666',linestyle='--',lw=1.6)
        ax.set(title=ident,xlabel='Epoch',ylabel='Training loss (batch mean)')
        ax.grid(alpha=.18);ax.legend(fontsize=8,ncol=2)
        save(fig,ident+'_loss','Original coefficients; raw CC is measured even when its weighted contribution is zero.')
    base='IEMOCAP_baseline_source_order_seed1475';full='IEMOCAP_cmerc_source_order_seed1475'
    fig,ax=plt.subplots(figsize=(8,4.8))
    for source,color,style in [('new','#0072B2','-'),('XP04','#009E73','--')]:
        a={int(r['epoch']):float(r['validation_weighted_f1']) for r in by_id[base][source]}
        b={int(r['epoch']):float(r['validation_weighted_f1']) for r in by_id[full][source]}
        common=sorted(set(a)&set(b))
        if common:
            ax.plot(common,[b[e]-a[e] for e in common],label=source+' CMERC - baseline',color=color,linestyle=style)
    ax.axhline(0,color='#777777',lw=.8)
    ax.set(title='IEMOCAP / seed 1475 / matched-epoch validation differences',xlabel='Epoch',ylabel='Validation W-F1 difference (pp)')
    ax.grid(alpha=.18);ax.legend(fontsize=9)
    save(fig,'IEMOCAP_validation_gain','Same-epoch trajectory description; differs from the separately selected best-checkpoint gains.')
    manifest={'formal_training_complete':summary['formal_training_complete'],'purpose':'diagnostic single-panel training curves',
        'python':sys.version,'executable':sys.executable,'matplotlib':matplotlib.__version__,'host':platform.node(),
        'inputs':{n:sha(ROOT/'results'/n) for n in ['metrics_summary.json','epoch_curves.csv','epoch_loss_curves.csv']},'files':files}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'figures':len(files),'formal_training_complete':summary['formal_training_complete']}))


if __name__=='__main__':
    main()
