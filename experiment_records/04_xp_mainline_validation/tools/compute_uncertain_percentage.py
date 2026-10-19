#!/usr/bin/env python3
"""CPU-only prevalence analysis of accepted XP04 predictions; no inference."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import numpy as np

SEEDS = [1475, 67137, 2024, 3407, 42]
DATASETS = {'IEMOCAP': (1623, 6), 'MELD': (2610, 7)}
REMOVALS = ['text', 'audio', 'visual', 'intra_context', 'inter_context']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prevalence(labels, full, removed):
    """Strict true-class comparison and per-category sample unions."""
    index = np.arange(len(labels))
    true_full = full.astype(np.float64)[index, labels]
    flags = []
    for values in removed:
        true_removed = values.astype(np.float64)[index, labels]
        flag = true_removed > true_full
        # Independent probability-coordinate check, avoiding float32 exp rounding.
        assert np.array_equal(flag, np.exp(true_removed) > np.exp(true_full))
        flags.append(flag)
    modality = np.logical_or.reduce(flags[:3])
    context = np.logical_or.reduce(flags[3:])
    return {
        'n_samples': len(labels),
        'modality_count': int(modality.sum()),
        'context_count': int(context.sum()),
        'modality_percentage': 100.0 * int(modality.sum()) / len(labels),
        'context_percentage': 100.0 * int(context.sum()) / len(labels),
        'individual_removal_counts': {name: int(flag.sum()) for name, flag in zip(REMOVALS, flags)},
    }


def analyze(archive, variant):
    acceptance_path = archive / 'results/acceptance.json'
    acceptance = json.loads(acceptance_path.read_text())
    assert acceptance['status'] == 'passed' and acceptance['runs_completed'] == 20
    assert acceptance['planned_seeds'] == SEEDS
    selected = [r for r in acceptance['runs'] if r['variant'] == variant]
    assert {(r['dataset'], r['seed']) for r in selected} == {(d, s) for d in DATASETS for s in SEEDS}
    by_identity = {(r['dataset'], r['seed']): r for r in selected}
    references, rows = {}, []
    for dataset, (n, classes) in DATASETS.items():
        for seed in SEEDS:
            record = by_identity[dataset, seed]
            source = Path(record['queue_job']['result_dir']) / 'predictions.npz'
            metrics_path = source.parent / 'metrics.json'
            assert sha256(source) == record['artifact_sha256']['predictions.npz'], source
            assert sha256(metrics_path) == record['artifact_sha256']['independent_metrics.json'], metrics_path
            metadata = json.loads(metrics_path.read_text())
            assert metadata['selected_epoch'] == record['selected_epoch']
            assert metadata['seed'] == seed and metadata['dataset'] == dataset
            assert Path(metadata['checkpoint']) == Path(record['queue_job']['run_dir']) / 'best.pt'
            with np.load(source, allow_pickle=False) as z:
                labels = z['labels']
                full = z['log_probabilities']
                removed = [z[f'removed_{name}_log_probabilities'] for name in REMOVALS]
                assert labels.shape == (n,) and np.issubdtype(labels.dtype, np.integer)
                assert np.all((labels >= 0) & (labels < classes))
                for values in [full, *removed]:
                    assert values.shape == (n, classes) and np.isfinite(values).all()
                    assert np.allclose(np.exp(values.astype(np.float64)).sum(1), 1, rtol=1e-4, atol=1e-5)
                assert np.allclose(z['probabilities'], np.exp(full.astype(np.float64)), rtol=1e-6, atol=1e-7)
                assert np.array_equal(z['predictions'], full.argmax(1))
                ids = z['dialogue_ids'].astype(str)
                positions = z['utterance_indices']
                assert ids.shape == positions.shape == (n,)
                assert len(set(zip(ids, positions.tolist()))) == n
                identity = (labels.copy(), ids.copy(), positions.copy())
                if dataset in references:
                    assert all(np.array_equal(x, y) for x, y in zip(identity, references[dataset]))
                else:
                    references[dataset] = identity
                values = prevalence(labels, full, removed)
            rows.append({'dataset': dataset, 'seed': seed, 'variant': variant, **values,
                         'selected_epoch': record['selected_epoch'], 'split_hash': record['split_hash'],
                         'prediction_path': str(source), 'prediction_sha256': sha256(source),
                         'metrics_sha256': sha256(metrics_path), 'checkpoint_path': metadata['checkpoint'],
                         'checkpoint_sha256_at_acceptance': record['checkpoint_checks']['best.pt']['sha256']})
    summary = []
    for dataset in DATASETS:
        subset = [r for r in rows if r['dataset'] == dataset]
        assert len(subset) == 5 and len({r['split_hash'] for r in subset}) == 1
        result = {'dataset': dataset}
        for category in ['modality', 'context']:
            values = [r[category + '_percentage'] for r in subset]
            result[category + '_mean'] = statistics.mean(values)
            result[category + '_std'] = statistics.stdev(values)
            assert np.isclose(result[category + '_std'], np.std(values, ddof=1), rtol=0, atol=1e-12)
        summary.append(result)
    return {'status': 'completed', 'source_protocol': 'XP04 validation-selected, noncausal, fixed_author',
            'variant': variant, 'training_or_inference_executed': False,
            'definition': 'strict removed true-class log-probability > full true-class log-probability; OR within category',
            'modality_removals': REMOVALS[:3], 'context_removals': REMOVALS[3:],
            'epsilon': 0.0, 'denominator': 'all test utterances, including both correct and incorrect predictions',
            'aggregation': 'unweighted mean of five per-seed percentages; sample std ddof=1',
            'acceptance_path': str(acceptance_path), 'acceptance_sha256': sha256(acceptance_path),
            'script_sha256': sha256(__file__), 'numpy_version': np.__version__,
            'per_seed': rows, 'summary': summary}


def outputs(result):
    summary, rows = result['summary'], result['per_seed']
    csv_text = io.StringIO(newline='')
    writer = csv.writer(csv_text, lineterminator='\n')
    writer.writerow(['Dataset', 'Mean Modality (%)', 'Std Modality (%)', 'Mean Context (%)', 'Std Context (%)'])
    for row in summary:
        writer.writerow([row['dataset'], *(format(row[key], '.12f') for key in
                        ['modality_mean', 'modality_std', 'context_mean', 'context_std'])])
    model = result['variant'].upper()
    lines = [f'# XP04 {model} 异常置信度上升样本占比', '',
             f'主表模型：**{model}**。XP04另有配对模型；本表只统计所列模型，未将不同模型合并。', '',
             '直接复用五个正式seed的validation-best对应测试集predictions.npz，不进行训练、推理或checkpoint重选。', '',
             '## 统计口径', '',
             '- 样本单位为测试话语；IEMOCAP每seed为1623条，MELD每seed为2610条。分母包含预测正确和错误的全部样本。',
             '- 比较真实类别y的置信度，不使用argmax类别或最大概率代替。',
             '- Modality：text/audio/visual任一路移除后p(y)>full即计入，对三路取并集；一个样本最多一次。',
             '- Context：intra-context/inter-context任一路移除后p(y)>full即计入，对两路取并集；一个样本最多一次。',
             '- 两类别分别计数，一个样本可以同时属于Modality和Context。相等不计入；严格>，不设额外epsilon或幅度门槛。',
             '- 以保存的float32 log-probs转换为float64后直接比较；log为严格单调函数。另用float64 exp后的概率独立复核，所有判断一致。',
             '- 每seed百分比=并集样本数/该seed测试样本总数×100；五seed等权均值，样本标准差ddof=1，不是标准误。',
             '- 这是用户定义的置信度上升事件比例，不是CEL幅度、分类错误率或统计显著异常比例。', '',
             '## 逐seed结果', '',
             '| Dataset | Seed | Modality (%) | Context (%) |',
             '|---------|------|--------------|-------------|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['modality_percentage']:.6f} | {row['context_percentage']:.6f} |")
    lines += ['', '## 五seed汇总', '',
              '| Dataset | Mean Modality (%) | Std | Mean Context (%) | Std |',
              '|---------|-------------------|-----|------------------|-----|']
    for row in summary:
        lines.append(f"| {row['dataset']} | {row['modality_mean']:.6f} | {row['modality_std']:.6f} | {row['context_mean']:.6f} | {row['context_std']:.6f} |")
    lines += ['', '## 绘图四个主值（百分比均值）', '']
    for row in summary:
        lines += [f"- {row['dataset']} modality：**{row['modality_mean']:.6f}%**（std {row['modality_std']:.6f}个百分点）",
                  f"- {row['dataset']} context：**{row['context_mean']:.6f}%**（std {row['context_std']:.6f}个百分点）"]
    lines += ['', '## 整数计数及所用checkpoint', '',
              '| Dataset | Seed | N | Modality union | Context union | Selected epoch |',
              '|---------|------|---|----------------|---------------|----------------|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['n_samples']} | {row['modality_count']} | {row['context_count']} | {row['selected_epoch']} |")
    lines += ['', '## 来源与复算', '',
              '十份NPZ及独立metrics.json的SHA-256均与XP04原验收记录一致；checkpoint路径和validation选中epoch一致。跨seed测试ID、话语位置及真实标签逐项相同，五路数组完整、有限、维度及概率归一化通过检查。', '',
              '原始逐样本文件仍保留在本机原批次/独立备份，不复制进公开档案。本次结果是已完成XP04的派生分析，不改原训练、验收或输入文件。输入路径、哈希、各路计数及checkpoint哈希记录见 [分析来源](../provenance/uncertain_percentage_analysis.json)。', '',
              '在CMERC工程根目录运行（需要原始NPZ可访问及NumPy）：', '',
              '```bash',
              '/data2/yb/conda_envs/cmerc-xp-repro/bin/python experiment_records/04_xp_mainline_validation/tools/compute_uncertain_percentage.py --archive experiment_records/04_xp_mainline_validation --output-root /tmp/xp04_uncertain_recompute',
              '```', '',
              '复算CSV的列名区分两类Std；报告和CSV保留六位及十二位小数，统计使用未舍入值。']
    return csv_text.getvalue(), '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'cmerc'], default='cmerc')
    args = parser.parse_args()
    result = analyze(args.archive.resolve(), args.variant)
    csv_text, report = outputs(result)
    for relative, text in [('results/uncertain_percentage_summary.csv', csv_text),
                           ('reports/UNCERTAIN_SAMPLE_PERCENTAGE.md', report),
                           ('provenance/uncertain_percentage_analysis.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n')]:
        target = args.output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(target)
    print(json.dumps({'status': 'completed', 'variant': args.variant, 'per_seed_rows': len(result['per_seed']),
                      'summary': result['summary']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
