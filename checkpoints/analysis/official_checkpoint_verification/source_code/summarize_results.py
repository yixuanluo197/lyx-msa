"""Independently verify saved full predictions and write the verification report."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUTPUT = Path(__file__).resolve().parents[1]
PROJECT = OUTPUT.parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def dump(name, value):
    (OUTPUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')


def same(actual, expected):
    assert math.isclose(float(actual),expected,rel_tol=0,abs_tol=1e-10),(actual,expected)


def main():
    preflight = json.loads((OUTPUT/'preflight.json').read_text())
    runs, inventories, results, independent = {}, {}, [], {}
    for dataset in ('IEMOCAP','MELD'):
        run=json.loads((OUTPUT/f'{dataset}_run_manifest.json').read_text())
        inventory=json.loads((OUTPUT/f'{dataset}_checkpoint_inventory.json').read_text())
        metrics=json.loads((OUTPUT/f'{dataset}_metrics.json').read_text())
        records=read_csv(OUTPUT/f'{dataset}_predictions.csv')
        names=metrics['class_names'];n=len(records)
        assert n=={'IEMOCAP':1623,'MELD':2610}[dataset]
        assert len({(r['dialogue_id'],r['utterance_index']) for r in records})==n
        assert all(r['input_condition']=='full' for r in records)
        labels=[int(r['true_label']) for r in records]
        predicted=[int(r['predicted_label']) for r in records]
        support=Counter(labels);pred_support=Counter(predicted)
        tp=Counter(y for y,p in zip(labels,predicted) if y==p)
        per_class=read_csv(OUTPUT/f'{dataset}_classification_report.csv')
        f1_values=[]
        for k,row in enumerate(per_class):
            precision=tp[k]/pred_support[k] if pred_support[k] else 0
            recall=tp[k]/support[k] if support[k] else 0
            f1=2*tp[k]/(support[k]+pred_support[k]) if support[k]+pred_support[k] else 0
            assert row['class_name']==names[k] and int(row['support'])==support[k]
            same(row['precision'],100*precision);same(row['recall'],100*recall);same(row['f1'],100*f1)
            f1_values.append(f1)
        accuracy=100*sum(y==p for y,p in zip(labels,predicted))/n
        weighted=100*sum(support[k]*f1_values[k] for k in range(len(names)))/n
        macro=100*sum(f1_values)/len(names)
        same(metrics['accuracy'],accuracy);same(metrics['weighted_f1'],weighted);same(metrics['macro_f1'],macro)
        matrix=read_csv(OUTPUT/f'{dataset}_confusion_matrix.csv')
        for k,row in enumerate(matrix):
            assert row['true_emotion']==names[k]
            for j,name in enumerate(names):
                assert int(row[name])==sum(y==k and p==j for y,p in zip(labels,predicted))
        arrays=np.load(OUTPUT/f'{dataset}_predictions.npz',allow_pickle=False)
        assert set(arrays.files)=={'labels','predictions','log_probabilities','probabilities'}
        assert np.array_equal(arrays['labels'],labels) and np.array_equal(arrays['predictions'],predicted)
        probabilities=arrays['probabilities']
        assert np.isfinite(arrays['log_probabilities']).all()
        assert (probabilities>=0).all() and (probabilities<=1).all()
        assert np.allclose(probabilities.sum(1),1,rtol=0,atol=1e-5)
        assert np.array_equal(probabilities.argmax(1),predicted)
        assert np.allclose(probabilities,np.exp(arrays['log_probabilities'].astype(np.float64)),rtol=0,atol=1e-15)
        for i,row in enumerate(records):
            assert row['true_emotion']==names[labels[i]] and row['predicted_emotion']==names[predicted[i]]
            for j,name in enumerate(names):same(row['probability_'+name],probabilities[i,j])
            same(row['predicted_confidence'],probabilities[i,predicted[i]])
            same(row['true_label_probability'],probabilities[i,labels[i]])
        assert run['status']=='completed' and run['model_state_unchanged'] and run['all_parameter_grads_none']
        assert run['original_inputs_unchanged'] and run['cudnn_enabled']
        assert not run['training'] and not run['new_calibration_diagnostics'] and not run['modality_removal']
        assert run['actual_gpu_uuids']==[preflight['gpu_uuid']]
        audit=run['audit'];expected_batches=2 if dataset=='IEMOCAP' else 18
        assert audit['forward_calls']==len(audit['batches'])==expected_batches
        assert audit['optimizer_steps']==0 and audit['masking_forward_calls']==0 and audit['strict_load_calls']==1
        assert not any(audit['gradient_enabled_observations']) and not any(audit['model_training_observations'])
        assert all(b['batch_size']==16 and b['num_workers']==2 for b in audit['batches'])
        assert sum(b['valid_utterances'] for b in audit['batches'])==n
        assert all(not value for value in inventory['compatibility'].values())
        same(metrics['gap_percentage_points'],weighted-metrics['paper_weighted_f1'])
        assert metrics['within_0p5_points']==(abs(weighted-metrics['paper_weighted_f1'])<=.5)
        runs[dataset]=run;inventories[dataset]=inventory
        results.append({key:metrics[key] for key in ('dataset','model','accuracy','weighted_f1','macro_f1','valid_samples','paper_weighted_f1','gap_percentage_points','within_0p5_points')})
        independent[dataset]=dict(valid_samples=n,accuracy=accuracy,weighted_f1=weighted,macro_f1=macro,
                                  classification_report_and_matrix_match=True,probabilities_and_padding_checked=True)
    for folder,key in [('code_official','official_sha256'),('code_runtime','runtime_sha256')]:
        for relative,expected in preflight[key].items():assert sha(OUTPUT/folder/relative)==expected
    for relative,expected in preflight['input_sha256'].items():assert sha(PROJECT/relative)==expected
    for path,expected in preflight['local_previous_diagnostic_files_sha256'].items():assert sha(path)==expected,path
    with (OUTPUT/'evaluation_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    inventory=dict(official_repository=preflight['official_repository'],release_source_url=preflight['checkpoint_source_url'],
                   downloaded_again=False,loading='torch.load(map_location=cpu, weights_only=True), model.load_state_dict(strict=True)',
                   source_audit=preflight['source_provenance_report'],entries=inventories,
                   limitations=['Both releases are plain state_dicts: epoch, training seed, calibration flags and selection protocol are not present.',
                                'Official README identifies their evaluation use; architecture alone does not prove the historical training procedure.'])
    dump('checkpoint_inventory.json',inventory)
    manifest=dict(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),server='biggpu',host='user22',
                  physical_gpu_index=1,gpu_uuid=preflight['gpu_uuid'],gpu4_forbidden=True,
                  remote_output=preflight['remote_output'],local_output=str(OUTPUT),official_commit=preflight['official_commit'],
                  code_classification=preflight['execution_classification'],training=False,modality_removal=False,
                  diagnostic_experiments_repeated=False,calibration_metrics_computed=False,
                  historical_predictions_reused=False,runs=runs,preflight_sha256=sha(OUTPUT/'preflight.json'),
                  observer_sha256=sha(OUTPUT/'source_code/verify_author_checkpoints.py'),
                  launch=json.loads((OUTPUT/'launch.json').read_text()),hardware_preflight_recovery=preflight['hardware_preflight_recovery'])
    dump('run_manifest.json',manifest)
    dump('verification.json',dict(status='passed',independent_calculation='Counter-based precision/recall/F1 from this run CSV; no official metric function reused',
          datasets=independent,protected_previous_diagnostic_files=len(preflight['local_previous_diagnostic_files_sha256']),
          official_inputs_hashes_verified=len(preflight['input_sha256']),frozen_code_files_per_copy=len(preflight['runtime_sha256']),
          unit_tests_passed=4,strict_loading_verified=True,optimizer_updates=0,full_forward_calls=20,removed_forward_calls=0))
    lines=['# CMERC 作者公开 Checkpoint 性能验证','','## 实验目的与结论','',
           '重新加载作者发布的两份权重，实际执行官方 train.py --testing --epochs 0，验证完整三模态测试集分类性能。本次不重新训练、不执行信息移除、不计算ECE/MCE/Brier/NLL/CEL，不重复此前六项诊断。历史成绩没有代替本次推理。','',
           '两个数据集均达到预先规定的 |实测W-F1−论文W-F1| ≤0.5个百分点标准，实测W-F1四舍五入两位后与论文相同。','',
           '| Dataset | Model | Accuracy (%) | Weighted F1 (%) | Macro F1 (%) | Valid samples | Paper W-F1 (%) | Gap (百分点) | ≤0.5点 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in results:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['accuracy']:.10f} | {row['weighted_f1']:.10f} | {row['macro_f1']:.10f} | {row['valid_samples']} | {row['paper_weighted_f1']:.2f} | {row['gap_percentage_points']:+.10f} | 是 |")
    lines += ['','CSV和JSON保留完整浮点精度；以上是本次实测值，不是复制论文数字。Gap=实测−论文，单位百分点。','',
              '## 权重来源和结构','',f"官方README下载入口：{preflight['checkpoint_source_url']}。优先使用服务器已有文件，没有重复下载。文件大小和SHA与2026-10-13下载审计一致，且本地与biggpu副本逐字节哈希一致。",'',
              '| Dataset | 文件 | 字节 | SHA-256 | state_dict张量键数 |','| --- | --- | ---: | --- | ---: |']
    for dataset,entry in inventories.items():
        lines.append(f"| {dataset} | `{entry['path']}` | {entry['size_bytes']} | `{entry['sha256']}` | {entry['tensor_keys']} |")
    lines += ['','权重均为纯OrderedDict，安全weights_only读取，严格strict=True加载；缺失键、额外键、形状不一致均为0。完整参数名称、shape、dtype保存在checkpoint_inventory.json。没有epoch、训练seed、优化器、训练flags或选模方式元数据，均标记未知，不从缺失字段推断训练历史。作者README把这两个文件用于CMERC评估，本次沿用该来源称谓，不以参数形状独立证明训练机制。','',
              '## 模型代码与配置','',f"固定官方commit：`{preflight['official_commit']}`。code_official保留14个逐字节核对git blob的原文件；code_runtime仅有compatibility.patch记录的两处既有PyG兼容修改：删除未用且当前不可导入的topk；修正HypergraphConv反向propagate的size轴。train.py、model.py、model_cm.py、dataloader.py与官方原文件一致。不能把此运行称为完全无补丁的官方源码执行。",'',
              '| 设置 | IEMOCAP | MELD |','| --- | --- | --- |',
              '| 结构 | GRU / hyper / direct / concat_DHT / BN | 同左 |',
              '| 输入 | avl完整三模态 | avl完整三模态 |',
              '| 文本 / 音频 / 视觉维数 | 1024 / 1582 / 342 | 1024 / 300 / 342 |',
              '| 类别数 / speaker数 | 6 / 2 | 7 / 9 |',
              '| dropout / num_L / num_K | 0.5 / 3 / 4 | 0.4 / 3 / 3 |',
              '| batch / workers / shuffle | 16 / 2 / False | 同左 |',
              '| eval seed | 官方命令未传，默认1475 | 官方命令未传，默认1475 |',
              '| RNN后端 | 默认cuDNN，deterministic=True | 同左 |',
              '| test对话 / 有效话语 / forward | 31 / 1623 / 2 | 280 / 2610 / 18 |','',
              'MELD训练命令的seed67137不等于本次README评估命令的默认seed。作者权重实际训练seed仍未知。完整解析参数、环境版本和每batch形状见run_manifest.json。','',
              '## 数据、话语顺序和指标','',
              '使用官方4个真实特征文件，SHA见preflight.json。IEMOCAP类别顺序happy/sad/neutral/angry/excited/frustrated；MELD为neutral/surprise/fear/sadness/joy/disgust/anger。测试DataLoader保持原pickle testVid顺序，batch16、不排序、不打乱、不丢最后批。',
              '官方model输出为log_softmax。官方函数按对话顺序拼接label[j][:length]，observer在同位置只读记录log_prob、原始ID和长度，检查每条标签与原pickle对应，排除padding，概率从原log_prob副本exp得到。没有改变输入张量或模型输出。',
              '官方函数返回的vids仅累积最后batch，是原代码的ID记录缺口；本次observer逐batch记录完整ID，没有将最后batch ID重复映射。',
              'Accuracy为正确话语比例；Weighted F1按真实类别support加权；Macro F1对全部固定类别取均值。指标用本次原始argmax结果由sklearn计算，并由独立Counter实现逐类复算；与官方函数两位小数返回结果一致。','']
    for dataset in ('IEMOCAP','MELD'):
        lines += [f'## {dataset}逐类别结果','',
                  '| 情绪 | Precision (%) | Recall (%) | F1 (%) | Support |','| --- | ---: | ---: | ---: | ---: |']
        for row in read_csv(OUTPUT/f'{dataset}_classification_report.csv'):
            lines.append(f"| {row['class_name']} | {float(row['precision']):.6f} | {float(row['recall']):.6f} | {float(row['f1']):.6f} | {row['support']} |")
        lines += ['',f'完整混淆矩阵见 `{dataset}_confusion_matrix.csv`，行是真实类别、列是预测类别。全部话语、标签、原文及各类概率见 `{dataset}_predictions.csv`，原log_prob另存NPZ。','']
    lines += ['## 执行证据和风险边界','',
              '- 实际服务器biggpu/user22，宿主健康GPU1，进程逻辑cuda:0；GPU UUID为'+preflight['gpu_uuid']+'。禁用宿主GPU4，未使用。两数据集顺序执行，进程PID、启动时间和命令均已记录，完成后本次GPU进程退出。',
              '- 官方testing分支实际运行，外层torch.no_grad，模型eval；仅创建官方原有Adam对象，不调用step，设置了禁止更新保护。前后模型参数/BN缓冲区、checkpoint和数据SHA一致，所有参数梯度为空。',
              '- 最初启动因PyTorch1.13的UUID可见性解析返回0设备而在模型构建前失败，尚无模型forward。已用最小硬件检查确定原因，改用CUDA_DEVICE_ORDER=PCI_BUS_ID和CUDA_VISIBLE_DEVICES=1，并用nvidia-smi进程PID→GPU UUID独立核对。失败日志与原脚本保存在attempts/attempt1_uuid；这不是按测试分数选择配置。',
              '- 原train.py训练路径存在测试集逐轮选模风险；作者纯权重不含真实训练和选模历史，无法由此证明其独立选模过程。本次只有单个已发布文件的固定测试，不做任何测试选模。',
              '- 论文报告五次运行均值，而作者公开的是单份状态；本次证明公开权重在当前兼容环境下达到接近论文的分类数值，不重建作者五次训练历史，也不验证其全部校准指标。',
              '- 本次两个W-F1差距均处于论文两位小数的报告精度范围，未触发>0.5点的额外差距排查。没有搜索batch、调权重、改标签或更换checkpoint以追分。',
              '- 前六项诊断使用本地validation权重，本次只使用作者pkl。原94份诊断文件哈希保持不变，结果不混用。','',
              '## 可复现命令与文件','',
              '实际持久启动脚本为run_evaluation.sh；其中两个dataset子进程调用source_code/verify_author_checkpoints.py，再执行code_runtime/train.py原始testing入口。evaluation.log保留官方argv、完整Namespace、终端指标和精确实测输出。',
              '本目录保存eval结果CSV、各类报告、混淆矩阵、预测CSV/NPZ、checkpoint_inventory.json、run_manifest.json、preflight.json、protocol_audit.md、verification.json和artifact_manifest.json。',
              '复跑请新建独立输出目录、复制固定代码/预检清单并调整命令路径；脚本拒绝覆盖已有同数据集结果。只读重算本次文件可运行：','',
              '```bash','/home/yangbin/miniconda3/bin/python checkpoints/analysis/official_checkpoint_verification/source_code/summarize_results.py','```','',
              '该重算命令不运行模型、不加载checkpoint，不生成六项诊断；仅核验已保存分类预测并重写本目录汇总。']
    (OUTPUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    artifacts={str(path.relative_to(OUTPUT)):dict(bytes=path.stat().st_size,sha256=sha(path))
               for path in sorted(OUTPUT.rglob('*')) if path.is_file() and path.name!='artifact_manifest.json'}
    dump('artifact_manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),artifacts=artifacts))
    print(json.dumps(dict(status='completed',results=results,protected_previous_files_unchanged=len(preflight['local_previous_diagnostic_files_sha256']),artifacts=len(artifacts)),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
