"""Check Pilot artifacts and build source-grounded reports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/data2/yb/paper/lyx-msa')
OUT = Path(__file__).resolve().parent
OFFICIAL = 'https://github.com/anas-jawad/CaliDist/blob/aa490893831c5c27a891e8d28c9da38b7e7f6057/src/calidist/evaluation/calidist.py'
METRICS = ('ece15', 'binary_brier', 'binary_nll')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


behavior_summaries = []
all_metrics = []
verification = {'status': 'completed', 'datasets': {}, 'official_calidist_commit':
                'aa490893831c5c27a891e8d28c9da38b7e7f6057',
                'no_training': True, 'no_test_labels_used': True, 'no_temperature_scaling': True}
files = {}
for dataset, n, dialogues in [('IEMOCAP', 527, 12), ('MELD', 1109, 114)]:
    folder = OUT / dataset
    parity = json.loads((folder / 'validation_parity.json').read_text())
    assert parity['completed'] and parity['parity']['passed']
    files[parity['checkpoint']] = sha(parity['checkpoint'])
    files[parity['prior_validation_file']] = sha(parity['prior_validation_file'])
    assert files[parity['checkpoint']] == parity['checkpoint_sha256']
    assert files[parity['prior_validation_file']] == parity['prior_sha256']
    original = pd.read_csv(folder / 'original_predictions.csv', dtype={'dialogue_id': str})
    pert = pd.read_csv(folder / 'perturbed_predictions.csv', dtype={'dialogue_id': str})
    behavior = pd.read_csv(folder / 'behavior_features.csv', dtype={'dialogue_id': str})
    reliability = pd.read_csv(folder / 'reliability_scores.csv', dtype={'dialogue_id': str})
    oof = pd.read_csv(folder / 'oof_calibration_predictions.csv', dtype={'dialogue_id': str})
    metrics = pd.read_csv(folder / 'calibration_metrics.csv')
    folds = json.loads((folder / 'fold_parameters.json').read_text())
    assert len(original) == n and original.dialogue_id.nunique() == dialogues
    assert len(pert) == n * 80 and len(behavior) == n * 8
    assert len(reliability) == n * 8 and len(oof) == n * 10
    assert not pert[['confidence']].isna().any().any()
    assert not behavior[['mu', 'delta', 'js']].isna().any().any()
    assert set(pert.repeat_id) == set(range(10))
    assert all(not (set(f['train_dialogues']) & set(f['held_out_dialogues'])) for f in folds)
    assert all(len(f['held_out_dialogues']) == 1 for f in folds) if dataset == 'IEMOCAP' else True
    visual_zero = 1 if dataset == 'IEMOCAP' else 95
    assert int(behavior.query("modality == 'visual' and corruption == 'gaussian'").visual_missing_skipped.sum()) == visual_zero
    assert len(pert.query("modality == 'visual' and visual_missing_skipped == True")) == visual_zero * 20
    p0 = original.set_index(['dialogue_id', 'utterance_index'])
    for row in pert.query("modality == 'visual' and visual_missing_skipped == True").itertuples():
        baseline = p0.loc[(row.dialogue_id, row.utterance_index)]
        assert np.allclose(json.loads(row.probabilities), json.loads(baseline.probabilities), atol=1e-5, rtol=1e-5)
    assert len(metrics) == 10
    group_summary = behavior.groupby(['corruption', 'modality']).agg(
        mean_mu=('mu', 'mean'), mean_delta=('delta', 'mean'), mean_js=('js', 'mean'),
        mean_flip_rate=('flip_rate', 'mean'), mean_confidence_change=('confidence_change_mean', 'mean'),
        nonzero_flip_fraction=('flip_rate', lambda x: float((x > 0).mean()))).reset_index()
    group_summary.insert(0, 'dataset', dataset)
    behavior_summaries.append(group_summary)
    all_metrics.append(metrics)
    verification['datasets'][dataset] = dict(utterances=n, dialogues=dialogues,
        original_rows=len(original), perturbed_rows=len(pert), behavior_rows=len(behavior),
        reliability_rows=len(reliability), oof_rows=len(oof),
        visual_missing_utterances=visual_zero, oof_folds=12 if dataset == 'IEMOCAP' else 5,
        max_original_probability_error=parity['parity']['checks']['probabilities']['max_abs_error'],
        max_original_log_probability_error=parity['parity']['checks']['log_probabilities']['max_abs_error'],
        checkpoint_sha256=parity['checkpoint_sha256'], gpu=parity['gpu']['gpu'],
        seconds=parity['seconds'])
    for path in folder.iterdir():
        if path.is_file() and path.name != 'progress.json':
            files[str(path.relative_to(OUT))] = sha(path)
for path in [OUT / 'run_behavior.py', OUT / 'analyze_calibration.py', OUT / 'build_reports.py',
             OUT / 'repair_rng_metadata.py', OUT / 'verify_final.py',
             OUT / 'provenance/official_calidist.py',
             OUT / 'provenance/calidist_paper.pdf']:
    files[str(path.relative_to(OUT))] = sha(path)
for name in ('IEMOCAP', 'MELD'):
    for file in [ROOT / f'CMERC/data/{"IEMOCAP_features.pkl" if name == "IEMOCAP" else "MELD_features_raw1.pkl"}',
                 ROOT / f'CMERC/data/{"iemocap_features_roberta.pkl" if name == "IEMOCAP" else "meld_features_roberta.pkl"}']:
        files[str(file.resolve())] = sha(file)
    stats_file = OUT / 'provenance' / f'{name}_train_statistics.npz'
    files[str(stats_file.relative_to(OUT))] = sha(stats_file)
prior_hashes = json.loads((OUT.parent.parent / '20261112T174957Z/provenance/source_hashes.json').read_text())
for path, digest in files.items():
    if path in prior_hashes:
        assert digest == prior_hashes[path], f'XP04 source changed: {path}'
(OUT / 'provenance/source_hashes.json').write_text(json.dumps(files, indent=2) + '\n')
(OUT / 'verification.json').write_text(json.dumps(verification, indent=2, allow_nan=False) + '\n')
behavior_summary = pd.concat(behavior_summaries, ignore_index=True)
behavior_summary.to_csv(OUT / 'behavior_summary.csv', index=False)
all_metrics = pd.concat(all_metrics, ignore_index=True)
all_metrics.to_csv(OUT / 'calibration_metrics.csv', index=False)
for name in ('original_predictions', 'perturbed_predictions', 'behavior_features',
             'reliability_scores', 'oof_calibration_predictions'):
    pd.concat([pd.read_csv(OUT / ds / f'{name}.csv', dtype={'dialogue_id': str})
               for ds in ('IEMOCAP', 'MELD')], ignore_index=True).to_csv(OUT / f'{name}.csv', index=False)

pilot = ['# XP04 random-feature Pilot', '',
         'Seed 1475, true Validation only. IEMOCAP: 527 utterances/12 dialogues; MELD: 1109/114.',
         'Fresh XP04 `best.pt` inference matched the independently saved true Validation probabilities.'
         ' Historical XP04 `validation/` prediction files were excluded because their rows are Test.',
         '', 'Gaussian uses `x + 0.05 × Std_train × ε`; dropout replaces 5% of selected coordinates'
         ' with the corresponding Training mean. Exactly 5% of nonconstant dimensions (rounded to the'
         ' nearest integer) are selected for each feature array. T1–T4 are perturbed together for Text.'
         ' All conditions use 10 independent deterministic repeats. The original model and source features were read-only.',
         '', 'The model returns log-probabilities. Probabilities were computed with `exp`, with no second softmax.',
         'Each target variant is evaluated in its original mixed Dialogue batch and retains the dialogue,'
         ' labels, `qmask`, `umask`, and other utterances.'
         ' Visual all-zero inputs were skipped for that modality and retained in the dataset.',
         '', '## Verification', '',
         '| Dataset | Max original probability error | Perturbed rows | Behavior rows | Skipped visual utterances |',
         '|---|---:|---:|---:|---:|']
for ds, v in verification['datasets'].items():
    pilot.append(f"| {ds} | {v['max_original_probability_error']:.3g} | {v['perturbed_rows']} | {v['behavior_rows']} | {v['visual_missing_utterances']} |")
pilot += ['', 'See `verification.json`, per-dataset `validation_parity.json`, and `provenance/source_hashes.json`.'
          ' This is a single-seed technical and calibration Pilot, not five-seed evidence.', '',
          '## Execution provenance', '',
          'Earlier local attempts were explicitly excluded: the first used a Bernoulli mask instead of exactly 5% coordinates;'
          ' the second exceeded graph-index capacity when duplicating long dialogues; the third showed that duplicate-dialogue'
          ' batches change XP04 predictions. This run perturbed one target at a time inside the original mixed-dialogue batch.'
          ' MELD was restarted on idle local GPU 0 after two simultaneous jobs saturated local GPU 3; IEMOCAP continued on GPU 3.'
          ' The only source edit during IEMOCAP execution widened the permitted local device assertion from GPU 3 to GPU 0/3'
          ' so MELD could run; no model, feature, perturbation, or analysis logic changed.']
(OUT / 'PILOT_REPORT.md').write_text('\n'.join(pilot) + '\n')

behavior_lines = ['# Behavioral metrics', '',
                  f'Original CaliDist equations were checked against [the ICML paper](https://proceedings.mlr.press/v306/jawad26a.html) (Section 3, Fig. 2) and [official code]({OFFICIAL}):',
                  '`μ = mean(1[pred_r ≠ pred_0] × conf_r)`, '
                  '`δ = |conf_0 − mean(conf_r)|`, and `λ_raw = (1−μ)/(δ+1e−10)`.',
                  'The equations are original; random perturbation of CMERC features is this Pilot’s adaptation of the paper’s semantic distractors. JS is an added diagnostic and does not enter the original Reliability Score.',
                  '', '| Dataset | Corruption | Modality | Mean μ | Mean δ | Mean JS | Mean flip rate | Mean confidence change |',
                  '|---|---|---|---:|---:|---:|---:|---:|']
for row in behavior_summary.itertuples():
    behavior_lines.append(f'| {row.dataset} | {row.corruption} | {row.modality} | {row.mean_mu:.6g} | {row.mean_delta:.6g} | {row.mean_js:.6g} | {row.mean_flip_rate:.6g} | {row.mean_confidence_change:.6g} |')
behavior_lines += ['', 'A low flip rate does not imply correctness; perturbation sensitivity is diagnostic only.'
                   ' The full 10-repeat probabilities and per-utterance metrics are in the CSV files.']
(OUT / 'BEHAVIOR_METRICS.md').write_text('\n'.join(behavior_lines) + '\n')

calibration_lines = ['# Top-label OOF calibration comparison', '',
    'All ECE15, Brier, and NLL values use binary prediction correctness; they are not multiclass scores.',
    'E1 fits an L2 logistic mapping from original confidence in each training fold.'
    ' E2 follows the [official CaliDist equations](' + OFFICIAL + '): training-fold λ min/max,'
    ' 100×100 α/β grid minimizing training-fold Brier, and `C_cal = C × sigmoid(β(λ−α))`.'
    ' The held-out dialogue does not contribute labels or λ bounds. No temperature scaling was used.',
    '', '| Dataset | Arm | Corruption | Modality | ECE15 | Binary Brier | Binary NLL | Accuracy | Weighted F1 | Macro F1 |',
    '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
for row in all_metrics.itertuples():
    calibration_lines.append(f'| {row.dataset} | {row.method} | {row.corruption} | {row.modality} | {row.ece15:.5f} | {row.binary_brier:.5f} | {row.binary_nll:.5f} | {row.accuracy:.5f} | {row.weighted_f1:.5f} | {row.macro_f1:.5f} |')
calibration_lines += ['', 'E2 can only lower the original confidence. Fold membership and fitted parameters'
                      ' are saved in each dataset’s `fold_parameters.json`; every held-out dialogue is disjoint'
                      ' from its training dialogues.']
(OUT / 'CALIBRATION_COMPARISON.md').write_text('\n'.join(calibration_lines) + '\n')

decision = ['# Final Pilot decision', '']
qualifying = []
for ds in ('IEMOCAP', 'MELD'):
    table = all_metrics[all_metrics.dataset == ds]
    original_rows = pd.read_csv(OUT / ds / 'original_predictions.csv')
    mean_confidence = float(original_rows.confidence.mean())
    accuracy = float((original_rows.prediction == original_rows.label).mean())
    score_rows = pd.read_csv(OUT / ds / 'reliability_scores.csv')
    scale = score_rows.calibrated_confidence / score_rows.original_confidence
    e1 = table[table.method == 'E1'].iloc[0]
    e2 = table[table.method == 'E2'].copy()
    e2['all_better'] = np.logical_and.reduce([e2[m] < float(e1[m]) for m in METRICS])
    wins = e2[e2.all_better]
    qualifying.extend([(ds, r.corruption, r.modality) for r in wins.itertuples()])
    best = e2.sort_values(['binary_brier', 'binary_nll', 'ece15']).iloc[0]
    decision += [f'## {ds}', '',
                 f'Original accuracy={accuracy:.5f}; mean top-label confidence={mean_confidence:.5f}; '
                 f'accuracy−confidence={accuracy - mean_confidence:+.5f}.',
                 f'E1: ECE15={e1.ece15:.5f}, Brier={e1.binary_brier:.5f}, NLL={e1.binary_nll:.5f}.',
                 f'Lowest-Brier E2 ({best.corruption}/{best.modality}): ECE15={best.ece15:.5f}, '
                 f'Brier={best.binary_brier:.5f}, NLL={best.binary_nll:.5f}.',
                 f'E2 conditions improving all three metrics over E1: {len(wins)}/8.',
                 f'E2 scale factor ≥0.999 in {float((scale >= .999).mean()):.1%} of utterance-condition records.'
                 ' The original CaliDist scaling factor is at most 1, so it cannot correct underconfidence by increasing C.', '']
if not qualifying:
    verdict = '暂无收益：暂不进入五 Seed 或原始数据干扰。'
elif len({x[0] for x in qualifying}) == 1:
    verdict = '部分有效：仅保留出现一致改善的数据集及模态，需五 Seed 复核。'
else:
    verdict = '值得继续：两个数据集均有单 Seed 一致改善，可进入五 Seed 完整实验验证。'
decision += ['## Decision', '', verdict, '',
             'Qualifying conditions: ' + (', '.join('/'.join(x) for x in qualifying) if qualifying else 'none') + '.',
             'This single-seed Pilot has no uncertainty interval and does not establish cross-seed robustness.'
             ' The full experiment remains subject to researcher review.']
decision += ['', '## Questions answered', '',
             '1. Both Gaussian noise and feature dropout changed some predictions and confidences;'
             ' dropout was stronger in both datasets, while most predictions remained stable.',
             '2. Under dropout, IEMOCAP audio was the most responsive single modality;'
             ' MELD text was the most responsive single modality. Joint perturbation had the largest flip rate in both.',
             '3. μ and δ formed a numerical Reliability Score, but E2 did not show consistent calibration value beyond C-only E1.',
             '4. No E2 condition improved ECE15, binary Brier, and binary NLL together relative to E1.',
             '5. IEMOCAP had small E2 ECE/Brier gains in some conditions, with worse NLL;'
             ' MELD E2 was essentially E0 and worse than E1 on all three metrics.',
             '6. The datasets differ: IEMOCAP is mildly overconfident on average and responds most to audio;'
             ' MELD is strongly underconfident on average and responds most to text.'
             ' CaliDist’s original scaling can only reduce confidence, explaining its MELD limitation.']
(OUT / 'FINAL_PILOT_DECISION.md').write_text('\n'.join(decision) + '\n')
print(verdict)
