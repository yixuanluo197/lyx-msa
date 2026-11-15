"""Audit repeated-measurement reliability without creating new perturbations."""

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostic'
CONFIG = json.loads((ROOT / 'preregistration.json').read_text())
SOURCE = Path(CONFIG['source_run'])
PILOT = Path(CONFIG['seed1475_pilot'])
KEY = ['dataset', 'seed', 'dialogue_id', 'utterance_index']
MEASURES = ['mu', 'delta', 'D', 'A', 'JS']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4194304), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def random_control(dataset, dialogue, utterance):
    key = [dataset, str(dialogue), int(utterance), 'capacity_control_seed1475']
    encoded = json.dumps(key, separators=(',', ':'), ensure_ascii=True).encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big')
    return np.random.default_rng(seed).normal(size=4)


def measure(probabilities, original, predicted):
    confidence = float(original.max())
    fixed = probabilities[:, predicted]
    maxima = probabilities.max(axis=1)
    flips = probabilities.argmax(axis=1) != predicted
    midpoint = (probabilities + original[None, :]) / 2
    eps = 1e-300
    js = .5 * np.sum(original * (np.log(np.maximum(original, eps)) - np.log(np.maximum(midpoint, eps))), axis=1)
    js += .5 * np.sum(probabilities * (np.log(np.maximum(probabilities, eps)) - np.log(np.maximum(midpoint, eps))), axis=1)
    return {'mu': float(np.mean(flips * maxima)), 'delta': float(abs(confidence - maxima.mean())),
            'D': float(confidence - fixed.mean()), 'A': float(np.sqrt(np.mean((fixed - confidence) ** 2))),
            'JS': float(js.mean()), 'flip_rate': float(flips.mean()),
            'D_mc_se': float(fixed.std(ddof=1) / np.sqrt(len(fixed)))}


def correlation(a, b, rank=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) <= CONFIG['zero_threshold'] or np.ptp(b) <= CONFIG['zero_threshold']:
        return None
    return float(spearmanr(a, b).statistic if rank else np.corrcoef(a, b)[0, 1])


def main():
    assert not (OUT / 'verification.json').exists(), 'Refusing to overwrite completed diagnostic'
    assert json.loads((SOURCE / 'provenance/joint_verification.json').read_text())['all_passed']
    started = time.time()
    originals = pd.read_csv(SOURCE / 'baseline/original_predictions.csv', dtype={'dialogue_id': str})
    old_features = pd.read_csv(SOURCE / 'behavior/feature_matrix.csv', dtype={'dialogue_id': str}).set_index(KEY)
    assert len(originals) == 8180 and not originals.duplicated(KEY).any()
    paths = [ROOT / 'preregistration.json', Path(__file__), SOURCE / 'baseline/original_predictions.csv',
             SOURCE / 'behavior/feature_matrix.csv', SOURCE / 'provenance/joint_verification.json']
    features, responses, half_rows, summaries, split_summaries = [], [], [], [], []
    checked = 0
    for dataset, spec in CONFIG['datasets'].items():
        for seed in CONFIG['seeds']:
            folder = PILOT / dataset if seed == 1475 else SOURCE / f'runs/{dataset}/seed{seed}'
            path = folder / 'perturbed_predictions.csv'
            paths.append(path)
            frame = pd.read_csv(path, dtype={'dialogue_id': str})
            frame = frame[frame.modality.eq('joint')]
            assert len(frame) == spec['n'] * 20
            orig = originals[originals.dataset.eq(dataset) & originals.seed.eq(seed)].copy()
            assert len(orig) == spec['n'] and orig.dialogue_id.nunique() == spec['dialogues']
            lookup = orig.set_index(['dialogue_id', 'utterance_index'])
            records = {}
            per_kind, halves = {k: [] for k in ['gaussian', 'dropout']}, {k: [] for k in ['gaussian', 'dropout']}
            for (dialogue, utterance, kind), group in frame.groupby(['dialogue_id', 'utterance_index', 'corruption'], sort=False):
                group = group.sort_values('repeat_id')
                assert group.repeat_id.tolist() == list(range(10))
                source = lookup.loc[(dialogue, utterance)]
                original = np.array(json.loads(source.probabilities), float)
                probabilities = np.array([json.loads(p) for p in group.probabilities], float)
                predicted = int(source.prediction)
                assert probabilities.shape == (10, spec['classes']) and np.isfinite(probabilities).all()
                assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)
                assert predicted == original.argmax()
                # Prior max() values were stored in float32; preserve their canonical anchor.
                assert abs(float(original.max()) - float(source.confidence)) < 1e-12
                full = measure(probabilities, original, predicted)
                ident = {'dataset': dataset, 'seed': seed, 'dialogue_id': dialogue, 'utterance_index': int(utterance)}
                historical = old_features.loc[(dataset, seed, dialogue, utterance)]
                for name in ['mu', 'delta', 'D', 'A']:
                    assert abs(full[name] - historical[f'{kind}_{name}']) < 1e-12
                checked += 1
                uid = (dialogue, utterance)
                if uid not in records:
                    clipped = np.clip(float(source.confidence), 1e-6, 1 - 1e-6)
                    ordered = np.sort(original)
                    record = {**ident, 'label': int(source.label), 'prediction': predicted,
                              'correct': int(source.label == predicted), 'confidence': float(source.confidence),
                              'logit_C': float(np.log(clipped / (1 - clipped))),
                              'margin': float(ordered[-1] - ordered[-2]),
                              'normalized_entropy': float(-np.sum(original * np.log(np.maximum(original, 1e-300))) / np.log(spec['classes'])),
                              'visual_missing': bool(historical.visual_missing)}
                    record.update({f'pred_class_{c}': int(predicted == c) for c in range(1, 7)})
                    record.update({f'rand_{i}': float(v) for i, v in enumerate(random_control(dataset, dialogue, utterance))})
                    records[uid] = record
                records[uid].update({f'{kind}_{name}': full[name] for name in MEASURES})
                responses.append({**ident, 'corruption': kind, **full})
                per_kind[kind].append({**full, 'confidence': float(source.confidence),
                                       'margin': records[uid]['margin'], 'entropy': records[uid]['normalized_entropy']})
                for split_id, (left, right) in enumerate(CONFIG['repeat_splits']):
                    a, b = measure(probabilities[left], original, predicted), measure(probabilities[right], original, predicted)
                    half = {**ident, 'corruption': kind, 'split': split_id}
                    half.update({f'{m}_{side}': values[m] for m in MEASURES for side, values in [('left', a), ('right', b)]})
                    half_rows.append(half)
                    halves[kind].append(half)
            assert len(records) == spec['n']
            features.extend(records.values())
            for kind in ['gaussian', 'dropout']:
                data, half = pd.DataFrame(per_kind[kind]), pd.DataFrame(halves[kind])
                variance = float(data.D.var(ddof=1))
                noise = float(np.mean(data.D_mc_se ** 2))
                for name in MEASURES:
                    row = {'dataset': dataset, 'seed': seed, 'corruption': kind, 'measure': name, 'n': len(data),
                           'mean': float(data[name].mean()), 'sample_sd': float(data[name].std(ddof=1)),
                           'zero_fraction': float(np.mean(np.abs(data[name]) <= CONFIG['zero_threshold'])),
                           'confidence_spearman': correlation(data[name], data.confidence, True),
                           'entropy_spearman': correlation(data[name], data.entropy, True),
                           'margin_spearman': correlation(data[name], data.margin, True)}
                    summaries.append(row)
                summaries.append({'dataset': dataset, 'seed': seed, 'corruption': kind, 'measure': 'D_measurement_noise',
                    'n': len(data), 'mean': float(data.D_mc_se.mean()), 'sample_sd': float(data.D_mc_se.std(ddof=1)),
                    'observed_between_utterance_variance': variance, 'mean_mc_variance': noise,
                    'noise_corrected_reliability_ratio': float(max(0, variance - noise) / variance) if variance > 0 else None,
                    'fraction_abs_D_gt_1_96_mcse': float(np.mean(np.abs(data.D) > 1.96 * data.D_mc_se)),
                    'no_flip_utterance_fraction': float(np.mean(data.flip_rate == 0)),
                    'mean_flip_rate': float(data.flip_rate.mean())})
                for split_id, h in half.groupby('split'):
                    for name in MEASURES:
                        a, b = h[name + '_left'], h[name + '_right']
                        pearson, spearman = correlation(a, b), correlation(a, b, True)
                        split_summaries.append({'dataset': dataset, 'seed': seed, 'corruption': kind, 'split': int(split_id),
                            'measure': name, 'n': len(h), 'pearson': pearson, 'spearman': spearman,
                            'correlation_estimable': pearson is not None and spearman is not None,
                            'left_zero_fraction': float(np.mean(abs(a) <= CONFIG['zero_threshold'])),
                            'right_zero_fraction': float(np.mean(abs(b) <= CONFIG['zero_threshold']))})
            print(json.dumps({'dataset': dataset, 'seed': seed, 'diagnosed': len(records)}), flush=True)
    feature_frame = pd.DataFrame(features)
    assert len(feature_frame) == 8180 and not feature_frame.duplicated(KEY).any()
    numeric = feature_frame.select_dtypes(include=np.number)
    assert np.isfinite(numeric.to_numpy()).all()
    feature_frame.to_csv(OUT / 'feature_matrix.csv', index=False)
    pd.DataFrame(responses).to_csv(OUT / 'utterance_responses.csv', index=False)
    pd.DataFrame(half_rows).to_csv(OUT / 'split_responses.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT / 'response_summary.csv', index=False)
    splits = pd.DataFrame(split_summaries)
    splits.to_csv(OUT / 'split_reliability.csv', index=False)
    aggregate = splits.groupby(['dataset', 'corruption', 'measure']).agg(
        estimable_splits=('correlation_estimable', 'sum'), spearman_mean=('spearman', 'mean'),
        spearman_min=('spearman', 'min'), spearman_max=('spearman', 'max'),
        pearson_mean=('pearson', 'mean')).reset_index()
    aggregate.to_csv(OUT / 'aggregate_reliability.csv', index=False)
    hashes = {str(p): sha(p) for p in paths}
    dump(OUT / 'source_hashes.json', hashes)
    dump(OUT / 'environment.json', {'python': sys.executable, 'numpy': np.__version__, 'pandas': pd.__version__,
                                   'host': platform.node(), 'cpu_only': True})
    report = '# XP04 随机扰动行为的重复测量可靠性\n\n'
    report += '本轮只读取已经核验的五Seed163600条joint扰动记录，没有模型推理或新扰动。8180条原话语各形成一条特征记录。\n\n'
    report += '对10次扰动使用预登记的三种5/5切分，全部报告。μ为flip×扰动最大置信度的均值，δ为原置信度与扰动最大置信度均值之差的绝对值；D/A取固定原预测类别的概率。JS为辅助扩展，不属原CaliDist核心。\n\n'
    report += '| Dataset | Corruption | Measure | Mean Spearman | Min | Max | Estimable /15 |\n|---|---|---|---:|---:|---:|---:|\n'
    for row in aggregate.itertuples():
        report += f'| {row.dataset} | {row.corruption} | {row.measure} | {row.spearman_mean:.4f} | {row.spearman_min:.4f} | {row.spearman_max:.4f} | {row.estimable_splits}/15 |\n'
    report += '\n这里的15项是5个模型Seed×3种切分的描述性汇总，彼此不独立，不是15份数据。常量或全零向量的相关性标记为不可估计，CSV留空；模型特征矩阵没有NaN/Inf。\n\n'
    report += '## Monte Carlo 噪声\n\nD的MC标准误为10次固定类概率的样本标准差/√10。response_summary.csv同时给出D跨话语方差、平均MC方差及max(0,var(D)−mean(SE²))/var(D)估计比例。这是描述性测量误差分解，受异方差与数值误差影响，不能据此判断某条预测错误。|D|>1.96SE比例也不是经过多重校正的逐话语显著检验。\n\n'
    report += '完整半样本结果见split_reliability.csv及split_responses.csv；原始响应与置信度、margin、entropy的相关见response_summary.csv。高重复性只代表敏感性可测，不证明它在静态概率之外有可靠性信息；低重复性说明10次估计噪声可能掩盖信号，不证明增加重复必然改善校准。\n\n'
    report += '本报告先于后续校准结果生成，不据诊断删特征、挑切分或重新设置扰动。后续固定强静态对照与联合拟合检验增量。所有数据仍来自参与过模型/方法选择的Validation，仅作探索性诊断。\n'
    (OUT / 'REPEATABILITY_REPORT.md').write_text(report)
    assert all(sha(p) == expected for p, expected in hashes.items())
    dump(OUT / 'verification.json', {'status': 'completed', 'independent_verification_pending': True,
        'feature_rows': len(feature_frame), 'response_rows': len(responses), 'half_response_rows': len(half_rows),
        'prior_behavior_groups_matched': checked, 'all_model_features_finite': True, 'source_hashes_unchanged': True,
        'no_gpu_or_new_perturbations': True, 'no_test_labels': True, 'seconds': time.time() - started})


if __name__ == '__main__':
    main()
