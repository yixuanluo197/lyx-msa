#!/usr/bin/env python3
"""Experiment 6: reuse fixed predictions; write calibration tables and provenance."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.high_confidence_calibration import summarize_calibration
from scripts.high_confidence_followup_stats import load_full_records, _align


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, records):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows({key: 'N/A' if value is None else value
                          for key, value in record.items()} for record in records)


def methods_text():
    return '''## 方法与范围

只复用实验一至五固定的本地 validation checkpoint 对应完整三模态测试预测；不换成作者权重，不训练，不重新推理，不用测试集调参或选模。IEMOCAP 每模型1623条有效话语，MELD 每模型2610条。

令 c_i=max_k p_ik，a_i=1[argmax_k p_ik=y_i]。按 c_i 分成10个等宽区间：[0,0.1)、[0.1,0.2)、…、[0.9,1.0]；内部边界归右侧区间，1归最后区间。

- 各箱平均置信度 conf_b=mean(c_i)，实际准确率 acc_b=mean(a_i)。signed_gap=conf_b−acc_b；正值表示过置信，负值表示欠置信。
- ECE=sum_b (n_b/N)×|conf_b−acc_b|；MCE=max_{n_b>0}|conf_b−acc_b|。两者仅在非空箱定义差值。
- 多类别 Brier Score=mean_i sum_k (p_ik−1[y_i=k])²，不除类别数、不除2，范围0–2。
- NLL=mean_i −ln(p_i,y_i)，使用自然对数，单位nats。原概率不裁剪、不重新归一化。本次真实类概率均大于0，没有∞或NaN。
- 空箱保留count=0，平均置信度、实际准确率、差值为N/A；ECE贡献为0，MCE不纳入空箱。图中不为其画0点。
- CSV中概率、ECE/MCE和差值均以0–1表示；Word表中ECE/MCE、置信度、准确率和差值乘100显示。差值是百分点，Brier和NLL保持原尺度。四个指标均以较低为优；Brier/NLL反映综合概率预测质量，不是只度量校准误差。

方法来源：[Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)；[多类别Brier定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)；[自然对数Log loss定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)。这里只采用数学定义，代码使用现有Python标准库，不依赖或安装scikit-learn。
'''


def write_summary(output, summaries, bins):
    lines = ['# 实验六：完整输入预测的概率校准', '', methods_text(),
             '## 总体对比', '',
             '| 数据集 | 模型 | 有效话语 | Accuracy (%) | 平均置信度 (%) | ECE (%) ↓ | MCE (%) ↓ | Brier ↓ | NLL (nats) ↓ |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    by_group = {(r['dataset'],r['model']):r for r in summaries}
    for row in summaries:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['valid_utterances']} | {100*row['accuracy']:.4f} | {100*row['mean_confidence']:.4f} | {100*row['ece']:.4f} | {100*row['mce']:.4f} | {row['brier_score']:.6f} | {row['nll']:.6f} |")
    lines += ['', '## 主要发现', '']
    for dataset in ('IEMOCAP','MELD'):
        b,c = by_group[dataset,'baseline'],by_group[dataset,'cmerc']
        lines.append(f"- {dataset}：ECE由{100*b['ece']:.4f}%降至{100*c['ece']:.4f}%，减少{100*(b['ece']-c['ece']):.4f}个百分点；MCE由{100*b['mce']:.4f}%降至{100*c['mce']:.4f}%。Brier由{b['brier_score']:.6f}降至{c['brier_score']:.6f}，NLL由{b['nll']:.6f}降至{c['nll']:.6f}。四项指标在本固定checkpoint对比中均改善。")
    lines += [
        '- IEMOCAP Baseline总体偏过置信：平均置信度83.2917%，实际准确率66.8515%；CMERC为71.7960%和68.0838%，差距缩小。图中非空区间的曲线整体更接近理想对角线，但不表示每条样本或每个区间都改善。',
        '- MELD主要表现为欠置信：Baseline平均置信度40.2129%，实际准确率66.5134%；CMERC平均置信度47.6384%，实际准确率66.7050%。CMERC提高整体置信度后，平均把握更接近实际正确率。',
        '- 这补充了前五项实验的结论：MELD错误样本平均置信度从0.346139升至0.408199，不能单独据此判断整体校准变差。实验六把正确与错误预测都纳入，按本口径ECE/MCE/Brier/NLL均降低；分类纠错收益有限与概率校准改善可以同时发生。',
        '- 可靠性图的横轴是每个非空区间的平均预测置信度，纵轴是该区间实际准确率。点在对角线上方表示欠置信，下方表示过置信；竖向距离即该箱绝对校准差。下方柱形展示样本数，避免将稀疏箱与大样本箱等量解读。',
        '', '## 十区间明细', '']
    for dataset in ('IEMOCAP','MELD'):
        for model in ('baseline','cmerc'):
            lines += [f'### {dataset} / {model}', '',
                      '| 区间 | 数量 | 平均置信度 (%) | 实际准确率 (%) | 置信度−准确率 (百分点) |',
                      '| --- | ---: | ---: | ---: | ---: |']
            for row in bins:
                if (row['dataset'],row['model']) != (dataset,model): continue
                interval=f"[{row['lower_bound']:.1f}, {row['upper_bound']:.1f}{']' if row['bin_index']==9 else ')'}"
                values=['N/A' if row[key] is None else f'{100*row[key]:.4f}' for key in ('mean_confidence','accuracy','signed_gap')]
                lines.append(f"| {interval} | {row['count']} | " + ' | '.join(values) + ' |')
            lines.append('')
    lines += ['## 局限性与来源', '',
              '- 本次为同一seed、固定权重的描述性测试分析，不是多seed均值或显著性检验；不能由四项指标改善直接证明某个训练模块的因果贡献。',
              '- ECE/MCE依赖分箱约定，MCE尤其容易受少量样本区间影响。10等宽箱在计算前固定，不依据测试结果选择分箱；没有为获得更好指标调整温度或参数。',
              '- 本轮对比是本地复现validation权重，不是作者发布checkpoint；不将当前F1或校准数值替换成论文数字。两模型的样本ID、真实标签、原文、顺序和类别映射严格一致。',
              '- 实验六直接以现有五实验Word为母版追加，保留其全部内容及原有“实验一Word母版缺失、使用Markdown转录”的说明；本轮不追溯性改写原结论。',
              '- 官方代码历史测试集选模风险、训练协议差异及均值替换限制沿用前五项报告说明。本轮不做模态删除，也不把本实验的校准指标与CEL混用。', '']
    (output/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp1-root',type=Path,default=ROOT/'checkpoints/analysis/high_confidence_errors')
    parser.add_argument('--previous-results',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    exp1=args.exp1_root.resolve()
    previous=(args.previous_results or exp1/'experiments_2_to_5').resolve()
    output=(args.output_dir or exp1/'experiment_6_calibration').resolve()
    if output in (exp1,previous) or output.is_relative_to(previous):
        parser.error('experiment 6 requires a separate output directory')
    if output.exists() and any(output.iterdir()):
        parser.error('refusing to overwrite an existing experiment')
    mother=previous/'CMERC_高置信度错误诊断报告_完整版.docx'
    if not mother.is_file(): raise FileNotFoundError(mother)
    original_files=[p for p in exp1.rglob('*') if p.is_file() and not p.is_relative_to(output)]
    before={str(p):sha256(p) for p in original_files}
    full=load_full_records(exp1)
    for dataset in ('IEMOCAP','MELD'):
        _align(full[dataset,'baseline'],full[dataset,'cmerc'],strict_order=True)
    summaries,bins=[],[]
    for group,records in full.items():
        summary,detail=summarize_calibration(records,10)
        summaries.append(summary);bins.extend(detail)
    if any(row['zero_true_probability_count'] for row in summaries):
        raise ValueError('A true probability is zero; infinite NLL requires explicit reporting')
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'exp6_overall_summary.csv',summaries)
    write_csv(output/'exp6_confidence_bins.csv',bins)
    for dataset in ('IEMOCAP','MELD'):
        write_csv(output/f'{dataset}_confidence_bins.csv',[row for row in bins if row['dataset']==dataset])
    write_summary(output,summaries,bins)
    old=json.loads((exp1/'run_manifest.json').read_text())
    manifest=dict(experiment=6,status='statistics_completed',started_at=datetime.now(timezone.utc).isoformat(),
                  split='test',input_condition='full',n_bins=10,new_training=False,new_inference=False,
                  checkpoint_source='same four local validation checkpoints as experiments 1–5, not author releases',
                  exp1_root=str(exp1),previous_results=str(previous),output_dir=str(output),
                  mother_docx=str(mother),mother_sha256=before[str(mother)],
                  previous_inputs_before_sha256=before,models={key:{name:value[name] for name in ('checkpoint','checkpoint_sha256','training_args','class_names')} for key,value in old['models'].items()},
                  metric_conventions=dict(ece='sum(nonempty_bin_count/N * abs(mean_confidence-accuracy))',
                    mce='max(abs(mean_confidence-accuracy)) over nonempty bins',
                    bins='[i/10,(i+1)/10), i=0..8; [0.9,1.0] last; empty statistics N/A',
                    signed_gap='mean_confidence - accuracy',brier='mean(sum_class((p-onehot)^2)), range [0,2]',
                    nll='mean(-ln(p_true)), no clipping, no renormalization',csv_scale='ECE/MCE, accuracy/confidence/gap in [0,1]; Brier unscaled; NLL nats'),
                  source_predictions={f'{dataset}/{model}':{'path':str(exp1/dataset/model/'all_predictions.csv'),'sha256':sha256(exp1/dataset/model/'all_predictions.csv'),'rows':len(records)} for (dataset,model),records in full.items()},
                  python=platform.python_version(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  row_counts={'exp6_overall_summary.csv':len(summaries),'exp6_confidence_bins.csv':len(bins)},
                  references=['https://proceedings.mlr.press/v70/guo17a.html','https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html','https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html'])
    for path,expected in before.items():
        if sha256(path)!=expected:raise ValueError(f'Previous artifact changed: {path}')
    manifest['previous_artifacts_unchanged']=True
    write_json(output/'run_manifest.json',manifest)
    readme=f'''# 实验六：概率校准分析

结果解释与全部10箱明细见 SUMMARY.md。输入为实验一至五同一批本地validation权重的完整输入预测，仅CPU统计，不训练、不重新推理、不换checkpoint。

## 文件

- exp6_overall_summary.csv：两个数据集、两模型的ECE/MCE/Brier/NLL及平均置信度/准确率，4行。
- exp6_confidence_bins.csv：每模型10个等宽区间，共40行，含数量、平均置信度、实际准确率、有符号/绝对差与ECE贡献。
- IEMOCAP_confidence_bins.csv、MELD_confidence_bins.csv：每个数据集20行。
- figures/reliability_IEMOCAP、reliability_MELD：PNG嵌入Word，另保存PDF和SVG矢量版本。
- CMERC_高置信度错误诊断报告_实验一至六.docx：前五实验Word内容完整保留，后附实验六三线表、图及解释。
- CMERC_实验六_概率校准结果.xlsx：本轮指标和10箱完整明细。
- run_manifest.json、render_manifest.json、verification.json、artifact_manifest.json：来源、绘图及表格记录、独立核验、文件哈希。
- source_code/：本轮分析、渲染、核验脚本及测试的完整快照。

## 复现

工作目录为仓库根目录。本次使用现有Python/matplotlib/python-docx/xlsxwriter，无新增依赖。重复运行统计时须指定新的空输出目录，保护既有结果。

```bash
/home/yangbin/miniconda3/bin/python scripts/run_calibration_experiment.py \\
  --output-dir checkpoints/analysis/high_confidence_errors/experiment_6_calibration_rerun
```

随后对新的输出目录调用渲染器，母版路径保持前五实验Word：

```bash
/home/yangbin/miniconda3/bin/python scripts/render_calibration_experiment.py \\
  --output-dir checkpoints/analysis/high_confidence_errors/experiment_6_calibration_rerun \\
  --mother-docx {mother}
```

只读核验现有结果：

```bash
/home/yangbin/miniconda3/bin/python scripts/verify_calibration_experiment.py \\
  --output-dir checkpoints/analysis/high_confidence_errors/experiment_6_calibration
```

所有分箱和分母事先固定；空箱为N/A，不作0准确率解释。ECE/MCE是top-label校准指标；Brier和NLL评价整个概率预测。只描述本批固定checkpoint，不声明多seed显著性或严格重现论文校准分数。
'''
    (output/'README.md').write_text(readme,encoding='utf-8')
    print(json.dumps({'output':str(output),'summary_rows':len(summaries),'bin_rows':len(bins),'new_inference':False},ensure_ascii=False))


if __name__=='__main__':
    main()
