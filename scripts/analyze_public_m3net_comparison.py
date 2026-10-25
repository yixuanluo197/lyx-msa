#!/usr/bin/env python3
"""CPU-only IEMOCAP behavior comparison of two fixed author-public checkpoints."""
import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.high_confidence_calibration import summarize_calibration
from scripts.high_confidence_followup_stats import _validate
from scripts.summarize_m3net_verification import independent_metrics

THRESHOLDS=(.8,.9,.95)
TRANSITIONS=('both_correct','baseline_wrong_cmerc_correct','baseline_correct_cmerc_wrong','both_wrong')
NAMES=('happy','sad','neutral','angry','excited','frustrated')
DEFAULT_OUTPUT=Path('checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis/experiment_2_public_m3net_comparison')


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''):value.update(part)
    return value.hexdigest()


def write_csv(path,rows,fields=None):
    if not rows and not fields:raise ValueError('Empty CSV needs explicit columns')
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]));writer.writeheader()
        writer.writerows({k:('N/A' if v is None else v) for k,v in r.items()} for r in rows)


def write_json(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_records(path,model):
    with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for key in ('utterance_index','true_label','predicted_label'):row[key]=int(row[key])
        for key in row:
            if key.startswith('probability_') or key in ('predicted_confidence','true_label_probability'):row[key]=float(row[key])
        correct=row['true_label']==row['predicted_label']
        if 'correct' in row and row['correct']!=str(correct):raise ValueError('Stored correctness differs')
        row['correct']=correct
    if [k for k in rows[0] if k.startswith('probability_')]!=['probability_'+n for n in NAMES]:raise ValueError('Class mapping changed')
    _validate(rows,('IEMOCAP',model),'full')
    for row in rows:
        if NAMES[row['true_label']]!=row['true_emotion'] or NAMES[row['predicted_label']]!=row['predicted_emotion']:raise ValueError('Emotion name mismatch')
    return rows


def pair_records(baseline,cmerc):
    key=lambda r:(r['dataset'],r['dialogue_id'],r['utterance_index'])
    bkeys=[key(r) for r in baseline];ckeys=[key(r) for r in cmerc]
    if len(set(bkeys))!=len(bkeys) or len(set(ckeys))!=len(ckeys):raise ValueError('Duplicate sample IDs')
    if set(bkeys)!=set(ckeys):raise ValueError('Different test sample sets')
    if bkeys!=ckeys:raise ValueError('Original test utterance order differs')
    lookup={key(r):r for r in cmerc};paired=[]
    for base in baseline:
        cm=lookup[key(base)]
        for field in ('true_label','true_emotion','utterance_text','source_utterance_id'):
            if base[field]!=cm[field]:raise ValueError(f'Identity mismatch: {field}')
        transition=('both_correct' if cm['correct'] else 'baseline_correct_cmerc_wrong') if base['correct'] else ('baseline_wrong_cmerc_correct' if cm['correct'] else 'both_wrong')
        row={k:base[k] for k in ('dataset','dialogue_id','utterance_index','source_utterance_id','utterance_text','true_label','true_emotion')}
        row['transition']=transition
        for prefix,source in [('baseline',base),('cmerc',cm)]:
            for field in ('model','predicted_label','predicted_emotion','predicted_confidence','true_label_probability','correct'):
                row[prefix+'_'+field]=source[field]
            row.update({prefix+'_probability_'+n:source['probability_'+n] for n in NAMES})
        row['predicted_class_changed']=base['predicted_label']!=cm['predicted_label']
        row['confidence_delta']=cm['predicted_confidence']-base['predicted_confidence']
        row['true_label_probability_delta']=cm['true_label_probability']-base['true_label_probability']
        row['cmerc_probability_of_baseline_predicted_class']=cm['probability_'+NAMES[base['predicted_label']]]
        row['baseline_predicted_class_probability_delta']=row['cmerc_probability_of_baseline_predicted_class']-base['predicted_confidence']
        for threshold in THRESHOLDS:
            suffix=str(threshold).replace('.','p')
            bh=not base['correct'] and base['predicted_confidence']>=threshold
            ch=not cm['correct'] and cm['predicted_confidence']>=threshold
            row['baseline_hc_error_ge_'+suffix]=bh;row['cmerc_hc_error_ge_'+suffix]=ch
            row['new_cmerc_hc_error_ge_'+suffix]=ch and not bh
            row['baseline_hc_outcome_ge_'+suffix]=('corrected' if cm['correct'] else 'still_wrong_confidence_lower' if row['confidence_delta']<0 else 'still_wrong_confidence_not_lower') if bh else 'not_baseline_hc_error'
        paired.append(row)
    return paired


def ratio(n,d):return n/d if d else None


def summarize_high_confidence(paired,threshold):
    bh=[r for r in paired if not r['baseline_correct'] and r['baseline_predicted_confidence']>=threshold]
    ch=[r for r in paired if not r['cmerc_correct'] and r['cmerc_predicted_confidence']>=threshold]
    bp=sum(r['baseline_predicted_confidence']>=threshold for r in paired)
    cp=sum(r['cmerc_predicted_confidence']>=threshold for r in paired)
    corrected=sum(r['cmerc_correct'] for r in bh)
    lower=sum(not r['cmerc_correct'] and r['confidence_delta']<0 for r in bh)
    not_lower=sum(not r['cmerc_correct'] and r['confidence_delta']>=0 for r in bh)
    remain=sum(not r['cmerc_correct'] and r['cmerc_predicted_confidence']>=threshold for r in bh)
    new=[r for r in ch if r['baseline_correct'] or r['baseline_predicted_confidence']<threshold]
    from_correct=sum(r['baseline_correct'] for r in new)
    assert corrected+lower+not_lower==len(bh)
    assert remain+len(new)==len(ch)
    return dict(dataset='IEMOCAP',threshold=threshold,valid_samples=len(paired),
        baseline_high_confidence_predictions=bp,baseline_high_confidence_errors=len(bh),baseline_high_confidence_error_rate=ratio(len(bh),bp),
        cmerc_high_confidence_predictions=cp,cmerc_high_confidence_errors=len(ch),cmerc_high_confidence_error_rate=ratio(len(ch),cp),
        corrected=corrected,still_wrong_confidence_lower=lower,still_wrong_confidence_not_lower=not_lower,still_wrong_above_threshold=remain,
        corrected_ratio=ratio(corrected,len(bh)),still_wrong_confidence_lower_ratio=ratio(lower,len(bh)),
        still_wrong_confidence_not_lower_ratio=ratio(not_lower,len(bh)),still_wrong_above_threshold_ratio=ratio(remain,len(bh)),
        new_cmerc_hc_errors=len(new),eligible_non_baseline_hc_errors=len(paired)-len(bh),
        new_cmerc_hc_error_ratio_of_eligible=ratio(len(new),len(paired)-len(bh)),
        new_cmerc_hc_errors_from_baseline_correct=from_correct,
        new_cmerc_hc_errors_from_baseline_wrong_below_threshold=len(new)-from_correct)


def select_typical_cases(paired):
    cases=[]
    for threshold in THRESHOLDS:
        for row in paired:
            bh=not row['baseline_correct'] and row['baseline_predicted_confidence']>=threshold
            ch=not row['cmerc_correct'] and row['cmerc_predicted_confidence']>=threshold
            types=[]
            if bh and row['cmerc_correct']:types.append('corrected_baseline_hc_error')
            if bh and not row['cmerc_correct'] and row['confidence_delta']<0:types.append('still_wrong_confidence_lower')
            if row['baseline_correct'] and ch:types.append('baseline_correct_new_cmerc_hc_error')
            if bh and ch:types.append('both_high_confidence_wrong')
            for name in types:
                severity=row['cmerc_predicted_confidence'] if name=='baseline_correct_new_cmerc_hc_error' else min(row['baseline_predicted_confidence'],row['cmerc_predicted_confidence']) if name=='both_high_confidence_wrong' else row['baseline_predicted_confidence']
                cases.append(dict(case_type=name,threshold=threshold,selection_score=severity,**row))
    return sorted(cases,key=lambda r:(r['case_type'],r['threshold'],-r['selection_score'],r['dialogue_id'],r['utterance_index']))


def high_confidence_explanation(row):
    total=int(row['baseline_high_confidence_errors']);corrected=int(row['corrected'])
    remain=int(row['still_wrong_above_threshold']);new=int(row['new_cmerc_hc_errors'])
    left_but_wrong=total-corrected-remain
    return (f"在阈值{float(row['threshold']):.2f}下，原{total}条高置信错误中，{corrected}条被纠正、"
            f"{left_but_wrong}条仍然错误但降到阈值以下、{remain}条仍为高置信错误；"
            f"另外新增{new}条高置信错误，所以最终是{remain}+{new}={remain+new}条。"
            "高置信错误数量下降与分类纠错不是同一个统计量。")


def md_table(rows,fields):
    def show(v):
        if v is None:return 'N/A'
        if isinstance(v,float):return f'{v:.6f}'
        return str(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']+['| '+' | '.join(show(r.get(k)) for k in fields)+' |' for r in rows])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=PROJECT);p.add_argument('--output-dir',type=Path)
    args=p.parse_args();root=args.project_root.resolve();out=(args.output_dir or root/DEFAULT_OUTPUT).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists():raise ValueError('Refusing to overwrite a completed statistical run; use an independent directory')
    bpath=root/'checkpoints/analysis/m3net_official_checkpoint_verification/all_predictions.csv'
    cpath=root/'checkpoints/analysis/official_checkpoint_verification/IEMOCAP_predictions.csv'
    sources={str(path.relative_to(root)):sha(path) for path in [bpath,cpath]}
    base=load_records(bpath,'Official M³Net');cm=load_records(cpath,'Official CMERC')
    paired=pair_records(base,cm)
    if len(paired)!=1623:raise ValueError('Wrong test set size')
    counts=Counter(r['transition'] for r in paired)
    transitions=[dict(dataset='IEMOCAP',transition=t,count=counts[t],ratio=counts[t]/len(paired)) for t in TRANSITIONS]
    hc=[summarize_high_confidence(paired,t) for t in THRESHOLDS]
    classes=[]
    for i,n in enumerate(NAMES):
        rows=[r for r in paired if r['true_label']==i];counts_i=Counter(r['transition'] for r in rows)
        be=sum(not r['baseline_correct'] for r in rows);ce=sum(not r['cmerc_correct'] for r in rows)
        classes.append(dict(dataset='IEMOCAP',true_label=i,true_emotion=n,support=len(rows),baseline_errors=be,cmerc_errors=ce,
            corrected=counts_i['baseline_wrong_cmerc_correct'],regressed=counts_i['baseline_correct_cmerc_wrong'],
            both_correct=counts_i['both_correct'],both_wrong=counts_i['both_wrong'],net_error_change=ce-be))
    calibration=[];bins=[]
    for name,rows,path in [('baseline',base,bpath),('cmerc',cm,cpath)]:
        summary,model_bins=summarize_calibration(rows,10);metrics,_,_=independent_metrics(rows,NAMES)
        errors=[r for r in rows if not r['correct']]
        summary.update(model_key=name,weighted_f1=metrics['weighted_f1']/100,macro_f1=metrics['macro_f1']/100,total_errors=len(errors),
            mean_error_confidence=math.fsum(r['predicted_confidence'] for r in errors)/len(errors),prediction_sha256=sha(path))
        calibration.append(summary);bins+=model_bins
    cases=select_typical_cases(paired)
    tables={'paired_predictions':paired,'classification_transitions':transitions,'high_confidence_transitions':hc,'classwise_transitions':classes,
        'calibration_comparison':calibration,'calibration_bins':bins,'typical_cases':cases}
    for name,rows in tables.items():write_csv(out/f'{name}.csv',rows)
    bweight=json.loads((root/'checkpoints/analysis/m3net_official_checkpoint_verification/checkpoint_audit.json').read_text())
    cweight=json.loads((root/'checkpoints/analysis/official_checkpoint_verification/IEMOCAP_checkpoint_inventory.json').read_text())
    for path,digest in sources.items():assert sha(root/path)==digest
    manifest=dict(status='statistics_completed',created_at=datetime.now(timezone.utc).isoformat(),dataset='IEMOCAP',valid_samples=len(paired),
        experiment='2 fixed public M3Net-CMERC behavior comparison',training=False,gpu_inference=False,model_forward_calls=0,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        input_predictions_sha256=sources,checkpoint_sources={k:{f:v[f] for f in ('path','sha256','tensor_keys','training_epoch','training_seed','selection_protocol')} for k,v in [('baseline',bweight),('cmerc',cweight)]},
        class_names=list(NAMES),sample_key=['dataset','dialogue_id','utterance_index'],sample_order='Original verified test order; exact source metadata match',
        threshold_comparison='>=',thresholds=list(THRESHOLDS),confidence='Maximum class probability; true-label probability saved separately',
        calibration={'ece_mce':'10 equal-width left-closed/right-open bins; p=1 in last bin; exclude empty bin observations',
            'brier':'Mean of sum over all classes of squared probability minus one-hot error',
            'nll':'Negative mean natural log original true-label probability; no clipping or renormalization'},
        missing_values='N/A for undefined ratios or empty-bin observations',
        high_confidence_denominators='HC error rate: all predictions >=threshold; correction outcomes: baseline HC errors; new error ratio: all non-baseline-HC-error samples',
        high_confidence_overlapping_measure='still_wrong_above_threshold is a separate overlapping flag, not a fourth mutually exclusive outcome',
        typical_cases='All matching rows for all four case categories and all three thresholds; categories/thresholds can overlap; Word top2 by selection_score within each type at .90 then stable sample ID tie-break',
        metric_units='All rates, F1, confidence, ECE/MCE and signed gaps stored as fractions [0,1]',
        public_weight_behavior_comparison=True,strict_training_pairing_confirmed=False,
        interpretation_limit='Separately published fixed author weights; training seed/epoch/selection/pairing unknown; no module causal attribution',
        python=platform.python_version(),host=platform.node(),
        output_csv_sha256={n+'.csv':sha(out/(n+'.csv')) for n in tables})
    write_json(out/'run_manifest.json',manifest)
    text='''# CMERC 官方公开权重实验二：分类纠错与概率校准对照

本实验只读取 IEMOCAP 的两份既有官方公开权重预测，共1623条有效话语。不训练、不做GPU推理、不重新选checkpoint、不修改输入。以 dataset/dialogue_id/utterance_index 唯一匹配，重新核对原文、真实标签、source utterance ID、原始顺序和六类概率映射。

**这是作者分别公开的固定权重行为对照，严格训练配对尚未确认。** 实际训练seed、轮次、选模方式及训练配对关系未知，不能将差异归因于CL、HSCL或CC；不是严格模块消融实验。

所有CSV的Accuracy、F1、置信度、ECE、MCE、差值与比率均为0–1分数；Brier/NLL按原定义。Word会明确百分数或原分数单位。CSV的N/A代表分母为零或空箱，不是0。

## 一、全部话语的分类变化

'''+md_table(transitions,['transition','count','ratio'])+'\n\n'
    text+=f"CMERC纠正{counts['baseline_wrong_cmerc_correct']}条原错误，同时将{counts['baseline_correct_cmerc_wrong']}条原正确话语变错，错误净增加{counts['baseline_correct_cmerc_wrong']-counts['baseline_wrong_cmerc_correct']}条。纠错数不能脱离新增错误单独解释。\n\n"
    text+='## 二、高置信预测和错误率\n\n'+md_table(hc,['threshold','baseline_high_confidence_predictions','baseline_high_confidence_errors','baseline_high_confidence_error_rate','cmerc_high_confidence_predictions','cmerc_high_confidence_errors','cmerc_high_confidence_error_rate'])+'\n\n'
    text+='高置信错误率分母是该模型所有达到阈值的预测数，不是错误数。以下前三种outcome互斥，合计各阈值原Baseline高置信错误；最后“仍超过原阈值”是与两种仍错状态可能重叠的独立统计，不可作为第四类相加。\n\n'+md_table(hc,['threshold','baseline_high_confidence_errors','corrected','still_wrong_confidence_lower','still_wrong_confidence_not_lower','still_wrong_above_threshold'])+'\n\n'
    text+='这些outcome各自比例见CSV，分母固定为对应Baseline高置信错误数。CMERC新增高置信错误定义为Baseline原本不是高置信错误、CMERC变为高置信错误，包含原本正确与原本低置信错误两种来源。\n\n'+md_table(hc,['threshold','new_cmerc_hc_errors','eligible_non_baseline_hc_errors','new_cmerc_hc_error_ratio_of_eligible','new_cmerc_hc_errors_from_baseline_correct','new_cmerc_hc_errors_from_baseline_wrong_below_threshold'])+'\n\n'
    text+=high_confidence_explanation(hc[1])+'\n\n'
    text+='## 三、按真实情绪类别\n\n'+md_table(classes,['true_emotion','support','baseline_errors','cmerc_errors','corrected','regressed','both_wrong','net_error_change'])+'\n\n'
    text+='net_error_change=CMERC errors−Baseline errors=regressed−corrected，正值表示错误增加。\n\n## 四、概率校准与分类性能\n\n'+md_table(calibration,['model','valid_utterances','accuracy','weighted_f1','total_errors','mean_confidence','mean_error_confidence','ece','mce','brier_score','nll'])+'\n\n'
    b,c=calibration
    text+=f"两份权重的W-F1分别为{100*b['weighted_f1']:.6f}%和{100*c['weighted_f1']:.6f}%，CMERC低{100*(b['weighted_f1']-c['weighted_f1']):.6f}个百分点；ECE、MCE、Brier、NLL的CMERC−Baseline变化依次为{c['ece']-b['ece']:+.9f}、{c['mce']-b['mce']:+.9f}、{c['brier_score']-b['brier_score']:+.9f}、{c['nll']-b['nll']:+.9f}。这组数据表现为分类略差、所列校准/概率指标略好，不能把校准改善写成整体分类纠错提高。没有跨seed重复，不能据此宣称统计显著或模块因果收益。\n\n"
    text+='错误平均置信度分别基于各自447条和456条错误集合，不能单凭均值差推断同一批错误均降低了置信度；同话语变化应以配对记录为准。\n\n'
    text+='定义与原实验六完全一致：ECE按10等宽箱的样本权重加权绝对置信度–准确率差；MCE是非空箱最大绝对差；Brier为各类别平方误差之和的样本均值，不除类别数或2；NLL用原真实类别概率的自然对数，不裁剪、不归一化。Signed Gap=mean confidence−accuracy；空箱不绘制观测点，可靠性图同时显示理想对角线及样本量。\n\n'
    text+='## 五、典型样本与完整概率\n\n`paired_predictions.csv`保留全部1623条与两模型全部12个概率、最大置信度差、真实类别概率差、原Baseline预测类别在CMERC中的概率差和预测类别是否改变。最大置信度下降不意味着同一错误类别的概率下降，也不等于已纠正。\n\n`typical_cases.csv`按三个阈值保留四种案例的全部匹配样本：原高置信错误被纠正；仍错但最大置信度下降；Baseline正确而CMERC新增高置信错误；两者均高置信错误。类别与阈值重叠时可重复。Word每类在0.90阈值按selection_score降序摘录前2条，稳定ID破同分，正反案例同规则展示；全量原文和概率始终在CSV中，不以Top2代替全部样本。\n\n'
    text+='## 六、复核、来源与范围\n\n两份原始预测和权重的SHA-256、统计源码、协议见`run_manifest.json`；独立重算与参考值差异检查见`independent_verification.json`及`verification.json`。参考值只用于计算后核对，不参与生成。旧实验文件SHA由`preserved_inputs.json`保护。无MELD分析、无新的模型推理。图和Word均为本轮新增，不覆盖旧报告。\n'
    (out/'SUMMARY.md').write_text(text,encoding='utf-8')
    print(json.dumps({'classification_transitions':transitions,'high_confidence_transitions':hc,'calibration':calibration,'typical_case_rows':len(cases)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
