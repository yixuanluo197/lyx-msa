"""CPU publication plots from completed, verified public diagnostic aggregates."""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CP=('XP04_best','XP04_latest','XP12_best','XP12_latest')
AUX=('cc','mscl','cscl')
GROUPS=('all','shared_backbone','text_gru','audio_projection','visual_projection','graph_shared','classification_head','other')
RELATIONS=('t','a','v','intra','inter')
COLORS={'cc':'#0072B2','mscl':'#D55E00','cscl':'#009E73'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,
                     'font.size':9,'axes.spines.right':False,'axes.spines.top':False,'legend.frameon':False})


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readcsv(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))
def num(value):
    try:
        value=float(value);return value if math.isfinite(value) else None
    except (ValueError,TypeError):return None


def canvas(title,ylabel,note,width=10.0,height=5.2):
    fig,axis=plt.subplots(figsize=(width,height));fig.subplots_adjust(left=.10,right=.79,top=.88,bottom=.28)
    axis.set_title(title,loc='left',fontsize=11,pad=12);axis.set_ylabel(ylabel);axis.grid(axis='y',alpha=.18,zorder=0)
    fig.text(.10,.025,note,fontsize=8,va='bottom',color='#444444')
    return fig,axis


def boxplot(axis,series,labels,colors,zero=False):
    usable=[(index,values) for index,values in enumerate(series) if values]
    for index,values in enumerate(series):axis.text(index,1.01,'n='+str(len(values)),transform=axis.get_xaxis_transform(),ha='center',fontsize=7)
    for index,values in usable:
        drawing=axis.boxplot([values],positions=[index],widths=.55,patch_artist=True,showfliers=False,
                             medianprops={'color':'black','linewidth':1},whiskerprops={'linewidth':.8},capprops={'linewidth':.8})
        drawing['boxes'][0].set(facecolor=colors[index],alpha=.65,linewidth=.8)
    axis.set_xticks(range(len(labels)));axis.set_xticklabels(labels,rotation=55,ha='right',fontsize=7)
    if zero:axis.axhline(0,color='#777777',linewidth=.8,linestyle=':')
    if not usable:axis.text(.5,.5,'N/A: no valid common support',transform=axis.transAxes,ha='center')


def main():
    summary=json.loads((ROOT/'results/metrics_summary.json').read_text())
    assert summary['completed'] and summary['verification_status']=='passed' and summary['case_count']==128
    assert summary['execution_counts']['training_runs']==summary['execution_counts']['optimizer_steps']==summary['execution_counts']['test_forwards']==0
    names=('analysis_gradient_pairs','gradient_summary','norm_summary','paired_checkpoint_deltas','pseudo_summary','class_relation_summary','branch_summary','pair_counts')
    data={name:readcsv(ROOT/'results'/(name+'.csv')) for name in names}
    output=ROOT/'results/figures';output.mkdir(parents=True,exist_ok=True);artifacts=[]
    def save(fig,name):
        for extension in ('png','svg'):
            path=output/(name+'.'+extension);fig.savefig(path,dpi=240 if extension=='png' else None,facecolor='white')
            artifacts.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size})
        plt.close(fig)
    pairs=[r for r in data['analysis_gradient_pairs'] if r['status']=='valid' and num(r['cosine']) is not None]
    labels=[cp.replace('XP','')+'\n'+loss.upper() for cp in CP for loss in AUX];colors=[COLORS[loss] for cp in CP for loss in AUX]
    for key,title,ylabel,filename,zero in [('cosine','Common-backbone gradient alignment','Cosine with classification','01_cosine_distributions',True),
                                         ('norm_ratio_reference_weighted','Reference-weighted common-support gradient scale','Reference-weighted norm / classification norm','02_reference_norm_ratios',False)]:
        fig,axis=canvas(title,ylabel,'Fixed eval snapshots; 32 predefined batches per checkpoint, not independent seeds.\n'
                              'Common finite nonzero parameter tensors only; box median/IQR, whiskers 1.5 IQR; fliers omitted.',height=5.8)
        series=[[num(r[key]) for r in pairs if r['checkpoint_id']==cp and r['loss_b']==loss and r['group']=='shared_backbone' and num(r.get(key)) is not None] for cp in CP for loss in AUX]
        boxplot(axis,series,labels,colors,zero)
        save(fig,filename)
    fig,axis=canvas('Negative-cosine frequency by parameter group','Parameter group',
                    'Each cell: frequency among valid batch pairs; annotation is valid n. Missing = N/A, not zero conflict.\n'
                    'Classification-head HSCL without common gradient support must remain N/A.',width=12.,height=6.3)
    matrix=np.full((len(GROUPS),12),np.nan);counts=np.zeros((len(GROUPS),12),dtype=int)
    for i,group in enumerate(GROUPS):
        for j,(cp,loss) in enumerate((cp,loss) for cp in CP for loss in AUX):
            row=next((r for r in data['gradient_summary'] if r['checkpoint_id']==cp and r['loss_b']==loss and r['group']==group),None)
            if row:
                counts[i,j]=int(float(row['valid_n']));value=num(row['conflict_frequency'])
                if value is not None:matrix[i,j]=value
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#E4E4E4');image=axis.imshow(matrix,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    axis.set_xticks(range(12));axis.set_xticklabels(labels,rotation=55,ha='right',fontsize=7);axis.set_yticks(range(8));axis.set_yticklabels(GROUPS,fontsize=8)
    for i in range(8):
        for j in range(12):axis.text(j,i,'N/A' if math.isnan(matrix[i,j]) else str(counts[i,j]),ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>.6 else 'black')
    fig.colorbar(image,ax=axis,fraction=.025,pad=.03,label='Conflict / valid pairs');save(fig,'03_group_conflict_frequency')
    fig,axis=canvas('Five binary relationship pseudo-labels','Checkpoint',
                    'These binary labels describe original/removal model relations, NOT emotion classes.\n'
                    'Cells show label-1 proportion; n is pooled utterance observations within this fixed checkpoint sample.',height=4.8)
    matrix=np.full((4,5),np.nan);counts=np.zeros((4,5),int)
    for i,cp in enumerate(CP):
        for j,rel in enumerate(RELATIONS):
            row=next(r for r in data['pseudo_summary'] if r['checkpoint_id']==cp and r['relation']==rel)
            matrix[i,j]=float(row['label1_fraction']);counts[i,j]=int(float(row['n_utterances']))
    image=axis.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto');axis.set_xticks(range(5));axis.set_xticklabels(RELATIONS);axis.set_yticks(range(4));axis.set_yticklabels(CP)
    for i in range(4):
        for j in range(5):axis.text(j,i,f'{matrix[i,j]:.2f}\nn={counts[i,j]}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.6 else 'black')
    fig.colorbar(image,ax=axis,fraction=.04,pad=.03,label='Relationship label-1 fraction');save(fig,'04_pseudo_label_proportions')
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.));fig.subplots_adjust(left=.10,right=.87,top=.91,bottom=.16,wspace=.40,hspace=.42)
    for axis,cp in zip(axes.flat,CP):
        matrix=np.full((7,5),np.nan);den=np.zeros((7,5))
        for i in range(7):
            for j,rel in enumerate(RELATIONS):
                rows=[r for r in data['class_relation_summary'] if r['checkpoint_id']==cp and r['relation']==rel and int(r['emotion_class'])==i]
                n=sum(float(r['n_utterances']) for r in rows);n1=sum(float(r['n_utterances']) for r in rows if int(r['pseudo_label'])==1)
                if n:matrix[i,j]=n1/n;den[i,j]=n
        image=axis.imshow(matrix,vmin=0,vmax=1,cmap=cmap,aspect='auto');axis.set_title(cp,loc='left',fontsize=10)
        axis.set_xticks(range(5));axis.set_xticklabels(RELATIONS,fontsize=8);axis.set_yticks(range(7));axis.set_yticklabels(range(7),fontsize=8);axis.set_ylabel('Emotion class ID')
        for i in range(7):
            for j in range(5):axis.text(j,i,'N/A' if math.isnan(matrix[i,j]) else f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>.6 else 'black')
    fig.colorbar(image,ax=list(axes.flat),fraction=.025,pad=.025,label='Relationship label-1 fraction')
    fig.text(.10,.035,'Emotion IDs are separate from binary relationship labels. Conditional fractions are not emotion accuracy.\n'
             'Per-cell denominators and absent emotion/relationship cells remain explicit in class_relation_summary.csv.',fontsize=8)
    save(fig,'05_emotion_relationship_heatmap')
    fig,axis=canvas('Same-batch checkpoint changes in common-backbone cosine','Latest minus best cosine',
                    'Matched batch and loss only; both checkpoint pairs must be valid. n is matched batches, not seeds.\n'
                    'XP04 is primary; XP12 is descriptive and is not a causal trajectory comparison.',height=5.6)
    labels=[group+'\n'+loss.upper() for group in ('XP04','XP12') for loss in AUX]
    series=[[float(r['delta_cosine']) for r in data['paired_checkpoint_deltas'] if r['checkpoint_from']==group+'_best' and r['loss_b']==loss and r['group']=='shared_backbone'] for group in ('XP04','XP12') for loss in AUX]
    boxplot(axis,series,labels,[COLORS[loss] for group in ('XP04','XP12') for loss in AUX],True);save(fig,'06_paired_checkpoint_cosine_changes')
    fig,axis=canvas('Correctness and confidence branches behind pseudo-labels','Utterance-observation fraction',
                    'Five mutually exclusive implementation branches; both_wrong_default0 is an unassigned default 0.\n'
                    'Relationship labels are not emotion categories. All fractions use the same checkpoint/relation denominator.',width=12.,height=6.0)
    branches=sorted({r['branch'] for r in data['branch_summary']});palette=['#777777','#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00']
    labels=[cp.replace('XP','')+'\n'+rel for cp in CP for rel in RELATIONS];bottom=np.zeros(20)
    for index,branch in enumerate(branches):
        values=[]
        for cp in CP:
            for rel in RELATIONS:
                part=[r for r in data['branch_summary'] if r['checkpoint_id']==cp and r['relation']==rel];total=sum(float(r['n_utterances']) for r in part)
                values.append(sum(float(r['n_utterances']) for r in part if r['branch']==branch)/total if total else 0)
        axis.bar(range(20),values,bottom=bottom,width=.8,label=branch,color=palette[index%len(palette)]);bottom+=values
    axis.set_xticks(range(20));axis.set_xticklabels(labels,rotation=65,ha='right',fontsize=6.5);axis.set_ylim(0,1)
    axis.legend(loc='center left',bbox_to_anchor=(1.02,.5),fontsize=7);save(fig,'07_pseudo_branch_coverage')
    fig,axis=canvas('True positive / negative relationship pair coverage','Ordered off-diagonal pair fraction',
                    'Counts use exact binary-label equality, separate from epsilon-adjusted mask masses.\n'
                    'Zero-positive / zero-negative anchor coverage and mask numeric details remain in pair_counts.csv.',width=12.,height=5.5)
    positive=[];negative=[]
    for cp in CP:
        for rel in RELATIONS:
            part=[r for r in data['pair_counts'] if r['checkpoint_id']==cp and r['relation']==rel]
            denominator=sum(float(r['ordered_offdiag_pairs']) for r in part)
            positive.append(sum(float(r['true_positive_pairs']) for r in part)/denominator if denominator else 0)
            negative.append(sum(float(r['true_negative_pairs']) for r in part)/denominator if denominator else 0)
    axis.bar(range(20),positive,label='True equal-label positive pairs',color='#0072B2');axis.bar(range(20),negative,bottom=positive,label='True different-label negative pairs',color='#E69F00')
    axis.set_xticks(range(20));axis.set_xticklabels(labels,rotation=65,ha='right',fontsize=6.5);axis.set_ylim(0,1);axis.legend(loc='center left',bbox_to_anchor=(1.02,.5),fontsize=7)
    save(fig,'08_relationship_pair_coverage')
    fig,axis=canvas('Actual versus reference-weighted local gradient scale','Median norm / classification norm',
                    'Common-backbone valid pairs only. XP12 MSCL/CSCL actual=0; reference weights are counterfactual.\n'
                    'Large norm does not establish harm. Exact valid n and missingness are in gradient_summary.csv.',height=5.7)
    ref=[];actual=[];labels=[]
    for cp in CP:
        for loss in AUX:
            row=next(r for r in data['gradient_summary'] if r['checkpoint_id']==cp and r['group']=='shared_backbone' and r['loss_b']==loss)
            ref.append(num(row['norm_ratio_reference_weighted_median']));actual.append(num(row['norm_ratio_actual_weighted_median']));labels.append(cp.replace('XP','')+'\n'+loss.upper())
    x=np.arange(12);axis.bar(x-.18,[np.nan if v is None else v for v in ref],.34,label='Reference coefficient',color='#777777');axis.bar(x+.18,[np.nan if v is None else v for v in actual],.34,label='Actual training coefficient',color='#0072B2')
    axis.set_xticks(x);axis.set_xticklabels(labels,rotation=55,ha='right',fontsize=7);axis.legend(loc='center left',bbox_to_anchor=(1.02,.5),fontsize=8);save(fig,'09_actual_reference_gradient_scale')
    assert len(artifacts)==18
    inputs=[ROOT/'results'/(name+'.csv') for name in names]+[ROOT/'results/metrics_summary.json',ROOT/'scripts/plot_diagnostic.py',ROOT/'provenance/figure_contract.md']
    manifest={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'completed':True,'matplotlib_version':matplotlib.__version__,
              'svg_text_editable':True,'png_dpi':240,'sample_unit':'fixed batch; not independent seed','source_data':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
              'analysis_input_sha256':summary['inputs_sha256'],'artifacts':artifacts}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'plots':9,'artifacts':18}))


if __name__=='__main__':main()
