#!/usr/bin/env python3
"""CPU-only MELD comparison of the audited local test-selected baseline and official CMERC."""
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
AUXILIARY_THRESHOLDS=(.4,.5,.6,.7)
ALL_THRESHOLDS=THRESHOLDS+AUXILIARY_THRESHOLDS
TRANSITIONS=('both_correct','baseline_wrong_cmerc_correct','baseline_correct_cmerc_wrong','both_wrong')
NAMES=('neutral','surprise','fear','sadness','joy','disgust','anger')
DEFAULT_OUTPUT=Path('checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis/experiment_2_meld_baseline_comparison')


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
    _validate(rows,('MELD',model),'full')
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
        for threshold in ALL_THRESHOLDS:
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
    return dict(dataset='MELD',threshold=threshold,valid_samples=len(paired),
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
    for threshold in ALL_THRESHOLDS:
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
                cases.append(dict(case_type=name,threshold_group='primary' if threshold in THRESHOLDS else 'auxiliary',threshold=threshold,selection_score=severity,**row))
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
    bpath=out/'baseline_predictions.csv'
    cpath=root/'checkpoints/analysis/official_checkpoint_verification/MELD_predictions.csv'
    sources={str(path.relative_to(root)):sha(path) for path in [bpath,cpath]}
    base=load_records(bpath,'Local README Baseline (test-selected)');cm=load_records(cpath,'Official CMERC')
    paired=pair_records(base,cm)
    if len(paired)!=2610:raise ValueError('Wrong test set size')
    counts=Counter(r['transition'] for r in paired)
    transitions=[dict(dataset='MELD',transition=t,count=counts[t],ratio=counts[t]/len(paired)) for t in TRANSITIONS]
    hc=[summarize_high_confidence(paired,t) for t in THRESHOLDS]
    aux=[summarize_high_confidence(paired,t) for t in AUXILIARY_THRESHOLDS]
    classes=[]
    for i,n in enumerate(NAMES):
        rows=[r for r in paired if r['true_label']==i];counts_i=Counter(r['transition'] for r in rows)
        be=sum(not r['baseline_correct'] for r in rows);ce=sum(not r['cmerc_correct'] for r in rows)
        classes.append(dict(dataset='MELD',true_label=i,true_emotion=n,support=len(rows),baseline_errors=be,cmerc_errors=ce,
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
    tables={'paired_predictions':paired,'classification_transitions':transitions,'high_confidence_transitions':hc,'auxiliary_threshold_transitions':aux,'classwise_transitions':classes,
        'calibration_comparison':calibration,'calibration_bins':bins,'typical_cases':cases}
    for name,rows in tables.items():write_csv(out/f'{name}.csv',rows)
    bweight=json.loads((out/'baseline_inference/MELD_checkpoint_inventory.json').read_text())
    audit=json.loads((out/'baseline_checkpoint_audit.json').read_text())
    cweight=json.loads((root/'checkpoints/analysis/official_checkpoint_verification/MELD_checkpoint_inventory.json').read_text())
    for path,digest in sources.items():assert sha(root/path)==digest
    manifest=dict(status='statistics_completed',created_at=datetime.now(timezone.utc).isoformat(),dataset='MELD',valid_samples=len(paired),
        experiment='2 fixed local README MELD baseline versus official CMERC behavior comparison',training=False,gpu_inference=True,model_forward_calls=18,statistics_gpu_inference=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        input_predictions_sha256=sources,checkpoint_sources={k:{f:v[f] for f in ('path','sha256','tensor_keys','training_epoch','training_seed','selection_protocol')} for k,v in [('baseline',bweight),('cmerc',cweight)]},
        class_names=list(NAMES),sample_key=['dataset','dialogue_id','utterance_index'],sample_order='Original verified test order; exact source metadata match',
        threshold_comparison='>=',thresholds=list(THRESHOLDS),auxiliary_thresholds=list(AUXILIARY_THRESHOLDS),auxiliary_thresholds_predeclared=True,
        baseline_source='Local official-README reproduction; not author-released baseline',
        baseline_training_checkpoint=audit['checkpoint'],baseline_test_selection_risk=True,
        baseline_training_server='local user23 TITAN Xp',evaluation_server='biggpu user22 V100; user requested common official protocol',confidence='Maximum class probability; true-label probability saved separately',
        calibration={'ece_mce':'10 equal-width left-closed/right-open bins; p=1 in last bin; exclude empty bin observations',
            'brier':'Mean of sum over all classes of squared probability minus one-hot error',
            'nll':'Negative mean natural log original true-label probability; no clipping or renormalization'},
        missing_values='N/A for undefined ratios or empty-bin observations',
        high_confidence_denominators='HC error rate: all predictions >=threshold; correction outcomes: baseline HC errors; new error ratio: all non-baseline-HC-error samples',
        high_confidence_overlapping_measure='still_wrong_above_threshold is a separate overlapping flag, not a fourth mutually exclusive outcome',
        typical_cases='All matching rows for all four categories and seven predeclared thresholds, primary and auxiliary explicitly separated; Word illustrations at auxiliary .60 top2/type by selection_score and stable sample ID tie-break',
        metric_units='All rates, F1, confidence, ECE/MCE and signed gaps stored as fractions [0,1]',
        fixed_weight_behavior_comparison=True,strict_training_pairing_confirmed=False,
        interpretation_limit='Local baseline seed67137 epoch5 official_test_selection, official CMERC training history unknown; fixed-weight behavior comparison, not strict module attribution',
        python=platform.python_version(),host=platform.node(),
        output_csv_sha256={n+'.csv':sha(out/(n+'.csv')) for n in tables})
    write_json(out/'run_manifest.json',manifest)
    run=json.loads((out/'baseline_inference/MELD_run_manifest.json').read_text())
    ref=json.loads((root/'checkpoints/analysis/official_checkpoint_verification/MELD_run_manifest.json').read_text())
    assert run['official_argv']==ref['official_argv'] and run['effective_args']==ref['effective_args']
    assert run['audit']['batches']==ref['audit']['batches']
    assert run['audit']['forward_calls']==18 and run['audit']['strict_load_calls']==1
    assert run['audit']['optimizer_steps']==run['audit']['masking_forward_calls']==run['cmerc_inference_calls']==0
    assert run['model_state_unchanged'] and run['all_parameter_grads_none'] and run['original_inputs_unchanged']
    assert not any(run['audit']['gradient_enabled_observations']) and not any(run['audit']['model_training_observations'])
    for key in ['cudnn_enabled','cudnn_deterministic','cudnn_version','torch','cuda_runtime']:
        assert run[key]==ref[key],key
    performance=[]
    for rows,reference in [(base,run['classification_metrics']),(cm,ref['classification_metrics'])]:
        computed,_,_=independent_metrics(rows,NAMES)
        for key in computed:assert math.isclose(computed[key],reference[key],abs_tol=1e-10),key
        performance.append(dict(dataset='MELD',model=rows[0]['model'],**{k:reference[k] for k in computed}))
    write_csv(out/'baseline_evaluation_results.csv',performance)
    manifest.update(baseline_inference_manifest='baseline_inference/MELD_run_manifest.json',
        baseline_model_forward_calls=18,cmerc_model_forward_calls=0,statistics_model_forward_calls=0,
        baseline_evaluation_metrics=performance[0],official_cmerc_metrics=performance[1],
        baseline_source_weights_sha256=audit['checkpoint']['sha256'],
        derived_state_checkpoint_sha256=bweight['sha256'],same_evaluation_protocol_verified=True)
    manifest['output_csv_sha256']['baseline_evaluation_results.csv']=sha(out/'baseline_evaluation_results.csv')
    write_json(out/'run_manifest.json',manifest)
    text='''# MELD 实验二：本地 README Baseline 与作者官方 CMERC 对照

本轮先审计再固定首选 `experiments/official_readme_reproduction/checkpoints/MELD_baseline_seed67137/best.pt`，依据身份和技术兼容性选择，没有按测试成绩比较候选。Baseline 是本地按官方 README 复现的M³Net结构，**不是作者公开的MELD Baseline**；训练seed=67137、epoch=5、历史选模=`official_test_selection`。它有测试集选模偏差，不能称为独立Validation选择。

源权重SHA `a6ab85cb6c0c12054dade9c6bcdf01cfaa6abc417625059a927de5ba386d3738`。83张量与官方MELD结构strict=True完全匹配，calibrate/contrastlearning/courselearning均关闭。纯模型state导出SHA `1d446d95fc21d14fd550e20bed939e0510d7ac343284507dad9f40a4e2217c6d`，张量内容与源model逐值一致，原始完整权重未修改。详见`MELD_baseline_checkpoint_audit.md`。

训练发生于本地XP；本轮按用户要求统一到已成功验证官方CMERC的biggpu V100/cuDNN、原冻结代码和两处既有PyG兼容补丁。仅首选Baseline进行了18批次完整T+A+V前向；CMERC直接复用既有官方MELD预测，未训练、未改参数、未重新选权重、未删除模态。模型参数/BN缓冲区前后哈希不变，optimizer step=0。

## 一、真实性能和全量对齐

以下Accuracy和F1单位为百分数；该表对应`baseline_evaluation_results.csv`，其他统计CSV统一使用0–1分数。

'''+md_table(performance,['model','accuracy','weighted_f1','macro_f1','valid_samples'])+'\n\n'
    text+='两模型2610条有效话语的dataset/dialogue_id/utterance_index、原文、真实标签、source utterance ID、原始测试顺序和七类概率字段严格一致。batch16、num_workers2、原Padding/Mask、GRU、hyper、concat_DHT、dropout.4、L3/K3、BN、avl、cuDNN后端和测试argv一致；评估seed1475不等于Baseline训练seed67137。\n\n## 二、分类纠错\n\n'
    text+=md_table(transitions,['transition','count','ratio'])+'\n\n'
    text+=f"CMERC纠正{counts['baseline_wrong_cmerc_correct']}条原错误，新增{counts['baseline_correct_cmerc_wrong']}条分类错误；错误净变化为{counts['baseline_correct_cmerc_wrong']-counts['baseline_wrong_cmerc_correct']:+d}条。纠错和新增错误必须同时解释。\n\n"
    text+='## 三、原始高置信阈值：0.80、0.90、0.95\n\n'
    cols=['threshold','baseline_high_confidence_predictions','baseline_high_confidence_errors','baseline_high_confidence_error_rate','cmerc_high_confidence_predictions','cmerc_high_confidence_errors','cmerc_high_confidence_error_rate']
    outcomes=['threshold','corrected','still_wrong_confidence_lower','still_wrong_confidence_not_lower','still_wrong_above_threshold','new_cmerc_hc_errors','new_cmerc_hc_errors_from_baseline_correct']
    text+=md_table(hc,cols)+'\n\n'+md_table(hc,outcomes)+'\n\n'
    text+='达到阈值的预测数为零时，错误率是N/A，不是0%。原CMERC在MELD上≥0.80只有2条预测且均正确、≥0.90与≥0.95无预测；这些阈值没有高置信错误不能证明其校准完美或鲁棒。\n\n## 四、预先固定辅助阈值：0.40、0.50、0.60、0.70\n\n'
    text+=md_table(aux,cols)+'\n\n'+md_table(aux,outcomes)+'\n\n'
    text+='辅助阈值在读取本轮Baseline预测前按用户要求固定，与主阈值分表，不替代原始三阈值。corrected/lower/not_lower是针对各阈值原Baseline高置信错误的互斥三类；still_wrong_above_threshold是独立重叠统计。高置信错误率的分母是所有达到阈值的预测，不是错误总数。新增CMERC高置信错误来自原Baseline正确或原低置信错误，两种来源与明确分母完整保留在CSV。\n\n'
    text+=high_confidence_explanation(aux[2])+'\n\n## 五、七类真实情绪\n\n'
    text+=md_table(classes,['true_emotion','support','baseline_errors','cmerc_errors','corrected','regressed','both_wrong','net_error_change'])+'\n\n'
    text+='net_error_change=CMERC errors−Baseline errors，正值为错误增加。fear/sadness/disgust均按实际数据呈现，不预设改善。\n\n## 六、全量概率校准\n\n'
    text+=md_table(calibration,['model','valid_utterances','accuracy','weighted_f1','mean_confidence','mean_error_confidence','ece','mce','brier_score','nll'])+'\n\n'
    b,c=calibration
    text+=f"CMERC−Baseline的W-F1为{100*(c['weighted_f1']-b['weighted_f1']):+.6f}个百分点；ECE/MCE/Brier/NLL的差依次为{c['ece']-b['ece']:+.9f}/{c['mce']-b['mce']:+.9f}/{c['brier_score']-b['brier_score']:+.9f}/{c['nll']-b['nll']:+.9f}。各错误均值置信度分别基于自己的{b['total_errors']}/{c['total_errors']}条错误，不能替代同话语配对分析。\n\n"
    text+='沿用此前实验六：全部有效话语10等宽左闭右开箱，最后包含1；ECE按箱占比加权绝对平均置信度与准确率差，MCE取非空箱最大绝对差；Brier为所有类别平方误差之和再样本平均，不除类别数或2；NLL为原真实类别概率的自然对数负均值，不裁剪、不温度缩放、不归一化。Signed Gap=mean confidence−accuracy，空箱观测N/A，不画虚假点。\n\n'
    text+=f"官方CMERC平均置信度{100*c['mean_confidence']:.4f}%而准确率{100*c['accuracy']:.4f}%，明显置信度不足；应结合辅助阈值与可靠性图，不能把无≥0.90预测当成完美可靠性。\n\n## 七、典型样本与完整明细\n\n"
    text+='`paired_predictions.csv`保留2610条两模型全部14个类别概率，最大置信度变化、真实类别概率变化和原预测类别概率变化独立保存；置信度下降不等于分类已纠正。`typical_cases.csv`保存四种案例、全部三个主阈值与四个辅助阈值的全部匹配；阈值及案例类型可以重叠。Word案例预先固定辅助0.60阈值，每种类型按selection_score降序前2条，稳定ID破同分；空类型如实标N/A，不以摘录代替全量数据。\n\n## 八、可比性与综合报告范围\n\n'
    text+='第一层：相同测试协议下的固定权重描述性行为对比技术可行。第二层：两份模型缺少严格训练配对，Baseline又有历史测试集选模风险，不能将分类或校准差异直接归因于CL/HSCL/CC。作者官方CMERC实际训练seed/epoch/选模历史未知，不补写成Validation选模。\n\n'
    text+='IEMOCAP实验二仍使用作者公开的Roberta_IEMOCAP3.pth.tar；MELD采用本轮明确来源的本地README复现Baseline，二者来源不同，不能合称同一组严格训练配对实验。综合Word保留原官方CMERC实验一/三/四/五/六的全部核实内容，追加此前IEMOCAP实验二和本轮MELD实验二，另存新文件，不修改旧Word。\n\n'
    text+='## 九、复核和文件\n\n`verification.json`汇总原始标签/文本/ID、概率/分类指标/分箱、独立逐表复算、旧报告和原模型权重SHA不变以及Word数字一致性。`baseline_classification_report.csv`和`baseline_confusion_matrix.csv`保存七类实际性能；`baseline_predictions.csv/.npz`是本次新推理，`baseline_evaluation_results.csv`为与官方CMERC性能对照，其百分数单位独立注明。模型推理清单、真实日志和统计代码版本见`baseline_inference/`、`evaluation.log`、`run_manifest.json`与`source_code/`。\n'
    (out/'SUMMARY.md').write_text(text,encoding='utf-8')
    print(json.dumps({'performance':performance,'classification_transitions':transitions,'primary':hc,'auxiliary':aux,'calibration':calibration,'typical_case_rows':len(cases)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
