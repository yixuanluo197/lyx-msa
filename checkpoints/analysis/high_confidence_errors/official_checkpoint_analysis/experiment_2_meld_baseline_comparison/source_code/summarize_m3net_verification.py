#!/usr/bin/env python3
"""Independently verify one fixed M3NET checkpoint run; no model/calibration/error analysis."""
import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''):value.update(part)
    return value.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def independent_metrics(rows,names):
    labels=[int(r['true_label']) for r in rows];preds=[int(r['predicted_label']) for r in rows]
    counts=Counter(zip(labels,preds));reports=[];matrix=[]
    for i,name in enumerate(names):
        support=sum(counts[i,j] for j in range(len(names)))
        predicted=sum(counts[j,i] for j in range(len(names)))
        tp=counts[i,i];p=tp/predicted if predicted else 0;r=tp/support if support else 0
        f=2*p*r/(p+r) if p+r else 0
        reports.append(dict(class_id=i,class_name=name,precision=100*p,recall=100*r,f1=100*f,support=support))
        matrix.append([counts[i,j] for j in range(len(names))])
    return dict(accuracy=100*sum(t==p for t,p in zip(labels,preds))/len(rows),
        weighted_f1=math.fsum(r['f1']*r['support'] for r in reports)/len(rows),
        macro_f1=math.fsum(r['f1'] for r in reports)/len(reports),valid_samples=len(rows)),reports,matrix


def verify_arrays(rows,path,names):
    fields=['probability_'+n for n in names]
    assert [k for k in rows[0] if k.startswith('probability_')]==fields
    ids=[(r['dataset'],r['dialogue_id'],int(r['utterance_index'])) for r in rows]
    assert len(ids)==len(set(ids))==1623
    values=np.array([[float(r[k]) for k in fields] for r in rows])
    labels=np.array([int(r['true_label']) for r in rows]);preds=np.array([int(r['predicted_label']) for r in rows])
    assert np.isfinite(values).all() and (values>=0).all() and (values<=1).all()
    assert np.allclose(values.sum(1),1,atol=1e-5,rtol=0)
    assert np.array_equal(values.argmax(1),preds)
    assert np.array_equal(values[np.arange(len(rows)),preds],[float(r['predicted_confidence']) for r in rows])
    assert np.array_equal(values[np.arange(len(rows)),labels],[float(r['true_label_probability']) for r in rows])
    for row in rows:
        assert row['dataset']=='IEMOCAP' and row['input_condition']=='full'
        assert names[int(row['true_label'])]==row['true_emotion']
        assert names[int(row['predicted_label'])]==row['predicted_emotion']
        if 'correct' in row:assert row['correct']==str(row['true_label']==row['predicted_label'])
    with np.load(path,allow_pickle=False) as arrays:
        assert set(arrays.files)=={'labels','predictions','log_probabilities','probabilities'}
        assert np.array_equal(arrays['labels'],labels) and np.array_equal(arrays['predictions'],preds)
        assert np.array_equal(arrays['probabilities'],values)
        assert np.isfinite(arrays['log_probabilities']).all()
        assert np.allclose(np.exp(arrays['log_probabilities'].astype(np.float64)),values,atol=2e-15,rtol=0)
    return float(np.abs(values.sum(1)-1).max())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=PROJECT)
    args=parser.parse_args();project=args.project_root.resolve()
    out=project/'checkpoints/analysis/m3net_official_checkpoint_verification'
    source=project/'checkpoints/analysis/official_checkpoint_verification'
    read=lambda p:json.loads(p.read_text())
    run=read(out/'inference/IEMOCAP_run_manifest.json');ref=read(source/'IEMOCAP_run_manifest.json')
    names=run['classification_metrics']['class_names']
    assert names==ref['classification_metrics']['class_names']
    rows=read_csv(out/'all_predictions.csv');original=read_csv(source/'IEMOCAP_predictions.csv')
    assert all(r['model']=='Official M³Net' for r in rows)
    m3dev=verify_arrays(rows,out/'predictions.npz',names)
    cmdev=verify_arrays(original,source/'IEMOCAP_predictions.npz',names)
    align_fields=['dataset','dialogue_id','utterance_index','source_utterance_id','utterance_text','true_label','true_emotion']
    assert len(rows)==len(original)==1623
    for a,b in zip(rows,original):assert all(a[k]==b[k] for k in align_fields)
    preflight=read(source/'preflight.json')
    for relative,digest in preflight['input_sha256'].items():assert sha(project/relative)==digest,relative
    rawpath=project/'data/IEMOCAP_features.pkl'
    with rawpath.open('rb') as stream:raw=pickle.load(stream,encoding='latin1')
    ids,labels,texts,test=raw[0],raw[2],raw[6],raw[8]
    expected=[(str(v),i,str(ids[v][i]),int(labels[v][i]),texts[v][i]) for v in test for i in range(len(labels[v]))]
    actual=[(r['dialogue_id'],int(r['utterance_index']),r['source_utterance_id'],int(r['true_label']),r['utterance_text']) for r in rows]
    assert actual==expected
    assert run['official_argv']==ref['official_argv']
    for key in ['torch','numpy','sklearn','cuda_runtime','cudnn_version','cudnn_enabled','cudnn_deterministic']:
        assert run[key]==ref[key],key
    assert run['audit']['batches']==ref['audit']['batches']
    assert len(run['audit']['batches'])==run['audit']['forward_calls']==2
    assert run['audit']['strict_load_calls']==1 and run['audit']['masking_forward_calls']==0
    assert run['audit']['optimizer_steps']==run['cmerc_inference_calls']==0
    assert not any(run['audit']['gradient_enabled_observations']) and not any(run['audit']['model_training_observations'])
    assert run['all_parameter_grads_none'] and run['model_state_unchanged'] and run['original_inputs_unchanged']
    assert run['model_state_sha256_before']==run['model_state_sha256_after']
    assert run['checkpoint_sha256']=='fb939de1f88855fda46a9d491e5cce4400c88c7187f3b792bb40f61c4ed03faa'
    assert run['actual_gpu_uuids']==['GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049']
    audit=read(out/'checkpoint_audit.json')
    assert audit['tensor_keys']==85 and not any(audit['compatibility'].values())
    results=[]
    for model,recordset,metrics in [('Official M³Net',rows,run['classification_metrics']),('Official CMERC',original,read(source/'IEMOCAP_metrics.json'))]:
        independent,report,matrix=independent_metrics(recordset,names)
        for k in independent:assert math.isclose(independent[k],metrics[k],rel_tol=0,abs_tol=1e-10),k
        if model=='Official M³Net':
            for observed,recalculated in zip(read_csv(out/'classification_report.csv'),report):
                assert observed['class_name']==recalculated['class_name']
                for k in ['class_id','precision','recall','f1','support']:assert math.isclose(float(observed[k]),recalculated[k],abs_tol=1e-10),k
            saved=read_csv(out/'confusion_matrix.csv')
            assert [r['true_emotion'] for r in saved]==names
            assert [[int(r[k]) for k in names] for r in saved]==matrix
        results.append(dict(dataset='IEMOCAP',model=model,**{k:metrics[k] for k in independent}))
    write_csv(out/'evaluation_results.csv',results)
    preserved=read(out/'preserved_inputs.json')
    for p,digest in preserved.items():assert sha(project/p)==digest,p
    verification=dict(status='passed',verified_at=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha256_verified=True,strict_load_complete=True,parameter_tensor_count=85,
        parameter_and_buffer_hash_unchanged=True,optimizer_steps=0,full_m3net_forwards=2,cmerc_forwards=0,masked_forwards=0,
        valid_samples=1623,dialogues=len(test),raw_labels_ids_text_order_verified=True,padding_excluded=True,
        class_mapping=names,class_probability_fields=['probability_'+n for n in names],
        model_probability_values_expected_to_differ=True,probability_sum_max_deviation={'m3net':m3dev,'cmerc':cmdev},
        finite_normalized_probabilities_argmax_correct=True,csv_npz_values_exactly_equal=True,
        same_test_utterance_alignment=True,same_batch_padding_eval_backend=True,
        classification_metrics_and_confusion_independently_recomputed=True,
        independent_metric_method='Counter contingency counts; Precision/Recall/F1 and aggregate metrics recomputed without sklearn',
        preserved_original_files=len(preserved),preserved_original_files_unchanged=True,
        public_weight_behavior_comparison_feasible=True,strict_training_pairing_confirmed=False,
        training_seed=None,training_epoch=None,checkpoint_selection=None,evaluation_seed=1475,
        tests_passed=17,six_diagnostic_experiments_executed=False,experiment2_full_error_analysis_executed=False,
        source_prediction_sha256={'m3net':sha(out/'all_predictions.csv'),'cmerc':sha(source/'IEMOCAP_predictions.csv')})
    write_json(out/'verification.json',verification)
    manifest=dict(run,source_revision='4d0fa88cd4f92c72b053527e877763f54e29ccb2',
        evaluation_code_revision='815c7c060914f5b55db816c0ba3662b11d08552f',
        comparison_table='evaluation_results.csv',cmerc_predictions_reused=True,
        verification='verification.json',public_weight_behavior_comparison_feasible=True,
        strict_training_pairing_confirmed=False,local_summary_host=platform.node(),
        local_summary_python=platform.python_version(),local_summary_numpy=np.__version__)
    write_json(out/'run_manifest.json',manifest)
    gap=results[1]['weighted_f1']-results[0]['weighted_f1']
    summary=f'''# M³Net 官方 IEMOCAP Checkpoint 性能验证

本轮只评估预先固定的作者发布权重 `Roberta_IEMOCAP3.pth.tar`，不训练、不模态删除、不计算六项诊断指标、不运行实验二完整错误分析、不评估其他两个候选。官方 CMERC 的结果直接复用原始预测与指标，未重跑或重新选择权重。

## 权重来源、结构及配置

- 权重路径：`/data2/yb/paper/cmerc_reproduction_20261013/checkpoints/baseline_candidates/m3net_official_20261025/Roberta_IEMOCAP3.pth.tar`。
- SHA-256：`{run['checkpoint_sha256']}`。
- 来源：[M³Net 作者 README](https://github.com/feiyuchen7/M3NET/tree/4d0fa88cd4f92c72b053527e877763f54e29ccb2#testing-and-checkpoints) 指向的 [官方公开 Dropbox](https://www.dropbox.com/sh/gd32s36v7l3c3u9/AACOipUURd7gEbEcdYSrmP-0a?dl=0)。沿用已有下载文件，无重复下载。
- CPU `torch.load(weights_only=True)` 加载为纯 OrderedDict；85 个张量，无缺失、额外或形状不兼容参数，`strict=True` 完整加载。
- 有效模型为两仓库一致的三模态 GRU + concat_DHT 超图结构：D_g=512、graph_hidden=512、num_L=3、num_K=4、dropout=0.5、norm=BN。复用冻结 CMERC 官方代码及既有两处 PyG 兼容补丁，未修改模型或增加新补丁。
- 为明确更换预训练权重，官方 train.py 原有 checkpoint 请求由只读加载器映射到上述唯一固定 M³Net 路径；清单同时记录 requested 与 actual_loaded_checkpoint。没有加载 CMERC 权重执行前向。
- 纯 state_dict 不包含实际训练 seed、epoch、选模方式或与发布 CMERC 权重的训练配对关系；不能从文件名、参数形状或当前测试成绩补造这些历史。

## 环境和测试协议

服务器 `ssh biggpu` / user22，宿主机 GPU 1（UUID `{run['actual_gpu_uuids'][0]}`），逻辑 cuda:0；GPU 4 禁用。Python {run['python']}、PyTorch {run['torch']}、CUDA {run['cuda_runtime']}、cuDNN {run['cudnn_version']}。使用原 `/data2/yb/conda_envs/cmerc-repro` 环境。

完整文本/音频/视觉，batch=16，num_workers=2，不打乱测试对话，原 Padding/Mask 及 dialogue-major flatten；六类顺序 happy/sad/neutral/angry/excited/frustrated。cuDNN enabled=True、deterministic=True，评估 seed=1475，仅为评估种子。实际两批次 forward，model.eval() 和 no_grad，optimizer step=0，无模态删除。

## 实际分类结果

单位：Accuracy / F1 为百分数；CSV 保留精度。

| Dataset | Model | Accuracy (%) | Weighted F1 (%) | Macro F1 (%) | Samples |
|---|---|---:|---:|---:|---:|
'''
    for r in results:summary+=f"| IEMOCAP | {r['model']} | {r['accuracy']:.10f} | {r['weighted_f1']:.10f} | {r['macro_f1']:.10f} | {r['valid_samples']} |\n"
    summary+=f'''
CMERC − M³Net 的 Weighted F1 差为 **{gap:+.10f} 个百分点**。本次固定公开 M³Net 权重的实测 F1 更高；不得因此测试另两份候选、更换 checkpoint、调整概率或以测试分数择优。该差值描述两份发布权重，既不能归因于某个训练模块，也不能据此否定其严格受控实验中的效果。

## 样本对齐及质量验证

两模型全部 **1623 条**有效话语的 dataset、dialogue ID、utterance index、source utterance ID、真实标签、原文和测试顺序完全一致；类别概率字段顺序和标签映射一致。概率数值不要求两模型相等，而是各自有效并对应同一类别/话语。

已对可信 SHA 的原始测试特征重新核验标签、原文、ID 和话语顺序；Padding 排除。所有概率有限、在 [0,1]、各行和约等于 1，预测为 argmax；CSV/NPZ 逐值一致。Accuracy、六类 Precision/Recall/F1、Weighted/Macro F1、Confusion Matrix 均由独立 Counter 统计复算通过。参数和 BN buffers 推理前后哈希不变；原 checkpoint/特征哈希不变；此前 {len(preserved)} 个实验文件 SHA 不变。详见 `verification.json`。

## 可比性结论与下一步边界

**A. 公开权重行为对照可行。** 结构、特征、测试样本、batch/Padding/Mask、cuDNN 及有效评估路径已核验一致，可基于这两份固定公开权重开展后续实验二的描述性逐样本行为对比。本轮仅保存准备数据，没有执行纠错类型统计。

**B. 严格配对的模块效果对照尚未确认。** 两份公开权重的训练 seed、轮次、checkpoint 选择准则与训练配对关系未知，无法把预测差异归因于 CMERC 的课程学习、对比学习或置信度约束。公开训练脚本存在测试集选模风险，但不能据此断言这两份发布权重实际采用了该方式；本轮未用测试集选模或调参。

细项证据与已核实差异见 `pairing_feasibility.md` / `protocol_audit/`，历史来源审计保持原样。仅限 IEMOCAP，不据此补造 MELD Baseline。

## 文件与复现

`evaluation_results.csv` 为两模型性能表，`classification_report.csv` / `confusion_matrix.csv` 为本次 M³Net 全类别结果，`all_predictions.csv` / `predictions.npz` 为全部话语原始概率。`checkpoint_audit.json` 为实际载入权重参数清单，`run_manifest.json` 与 `evaluation.log` 记录命令和真实执行，原始观察产物完整保存在 `inference/`。

运行入口 `run_evaluation.sh` 固定上述唯一权重和健康卡，已完成产物不需重复推理，脚本拒绝覆盖完成的运行。CPU 独立复算：`python scripts/summarize_m3net_verification.py`。运行脚本及其源码副本不改变已有模型文件、标签或预测。
'''
    (out/'SUMMARY.md').write_text(summary,encoding='utf-8')
    print(json.dumps(dict(status='passed',metrics=results,cmerc_minus_m3net_weighted_f1_pp=gap,preserved_files=len(preserved)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
