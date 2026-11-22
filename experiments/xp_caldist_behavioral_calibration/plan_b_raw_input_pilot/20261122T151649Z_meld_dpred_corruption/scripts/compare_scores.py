#!/usr/bin/env python3
"""Fixed CPU E0--E3 exploratory reliability comparison; no predictive model.

Primary: identical jointly-covered score_fit and score_eval for all four models.
Secondary deployment: common-fit enhanced models where required inputs exist;
otherwise exact fallback to E0 fitted on the entire score_fit queue.
Only scalar CSV/JSON split/provenance files and the pinned scorer are read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '8'

import numpy as np

RUN = Path(__file__).resolve().parents[1]
SCORER_PATH = RUN.parent / '20261121T175233Z_reliability_readiness/scripts/score_heldout.py'
SCORER_SHA = '22acb70db16cd1013fbc5b60dccc707d7382653a39cfedbd7d442b51aa8d9049'
if hashlib.sha256(SCORER_PATH.read_bytes()).hexdigest() != SCORER_SHA:
    raise RuntimeError('Pinned readiness scorer changed')
spec = importlib.util.spec_from_file_location('_corruption_fixed_score_heldout', SCORER_PATH)
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)
METHODS = {'E0': ('logit_confidence',), 'E1': ('logit_confidence', 'D_pred'),
           'E2': ('logit_confidence', 'corruption_d_pred'),
           'E3': ('logit_confidence', 'D_pred', 'corruption_d_pred')}
ROLE_COUNTS = {'score_fit': 1569, 'score_eval': 1407}
ROLE_DIALOGUES = {'score_fit': 155, 'score_eval': 155}
MISSING = {'', 'na', 'nan', 'null', 'none'}


def validate_protocol(protocol):
    scoring = protocol['scoring']
    if (protocol['dataset'] != 'MELD' or protocol['test_access'] is not False
            or protocol['official_validation_new_inference'] is not False
            or scoring['comparison'] != {k: list(v) for k, v in METHODS.items()}
            or scoring['C'] != shared.LOGISTIC['C']
            or scoring['scorer_sha256'] != SCORER_SHA
            or scoring['bootstrap']['seed'] != shared.SEED
            or scoring['bootstrap']['replicates'] != shared.BOOTSTRAPS):
        raise ValueError('Frozen E0--E3 method/bootstrap/provenance protocol mismatch')


def validate_inputs(raw, dialogue_manifest=None, utterance_manifest=None,
                    *, _expected_counts=ROLE_COUNTS, _expected_dialogues=ROLE_DIALOGUES):
    rows, validation = shared.validate_rows(raw, dialogue_manifest, utterance_manifest)
    by_id = {r['sample_id']: r for r in raw}
    for row in rows:
        source = by_id[row['sample_id']]
        if 'corruption_d_pred' not in source:
            raise ValueError('Missing corruption_d_pred column; never replace semantic D_pred')
        value = source['corruption_d_pred']
        number = None if str(value).strip().lower() in MISSING else shared.finite_number(value, 'corruption_d_pred')
        if number is not None and not -1 - 1e-6 <= number <= 1 + 1e-6:
            raise ValueError('Corruption D_pred must be a signed probability difference in [-1,1]')
        row['corruption_d_pred'] = number
        if 'corruption_k2_available' in source:
            flag = str(source['corruption_k2_available']).strip().lower()
            if flag not in {'true', 'false', '1', '0'} or (flag in {'true', '1'}) != (number is not None):
                raise ValueError('Corruption K2 availability marker disagrees with scalar presence')
    for role, expected in _expected_counts.items():
        selected = [r for r in rows if r['role'] == role]
        if len(selected) != expected:
            raise ValueError('Exact frozen role count mismatch: ' + role)
        if len({r['dialogue_id'] for r in selected}) != _expected_dialogues[role]:
            raise ValueError('Exact frozen role Dialogue count mismatch: ' + role)
    validation.update(exact_role_counts_verified=True, exact_role_dialogue_counts_verified=True,
                      semantic_D_pred_not_replaced=True, corruption_difference_range_verified=True)
    return rows, validation


def support(rows):
    y = [r['original_correct'] for r in rows]
    return dict(n=len(rows), dialogues=len({r['dialogue_id'] for r in rows}),
                correct=sum(y), errors=len(y) - sum(y), two_outcomes=len(set(y)) == 2)


def available(row, fields):
    return all(row.get(field) is not None for field in fields)


def fit_fixed(rows, fields, clean_reference=None):
    """Same pinned estimator/scaler/settings; generalize shared clean prefix.

    The parent clean-reference guard only permits one last column named D_pred.
    E2/E3 require a generalized prefix guard. All original settings and fitting
    operations remain identical, and the actual same-fit E0 scaler is copied
    before optimization (including mean/variance/scale), never after training.
    """
    if clean_reference is None:
        return shared.fit_logistic(rows, fields)
    x = np.asarray([[r[field] for field in fields] for r in rows], dtype=float)
    y = np.asarray([r['original_correct'] for r in rows], dtype=int)
    if (len(np.unique(y)) != 2 or tuple(fields[:1]) != clean_reference['fields']
            or tuple(r['sample_id'] for r in rows) != clean_reference['fit_sample_ids']):
        raise ValueError('Generalized shared scaler requires both outcomes and exact same Fit rows')
    scaler = shared.StandardScaler().fit(x)
    for name in ('mean_', 'var_', 'scale_'):
        target, reference = getattr(scaler, name), getattr(clean_reference['scaler'], name)
        if not np.allclose(target[:1], reference, atol=1e-12, rtol=1e-12):
            raise ValueError('Same-Fit clean scaler mismatch beyond roundoff')
        target[:1] = reference
    with warnings.catch_warnings():
        warnings.simplefilter('error', shared.ConvergenceWarning)
        model = shared.LogisticRegression(**shared.LOGISTIC).fit(scaler.transform(x), y)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError('Nonfinite CPU logistic parameters')
    return dict(fields=tuple(fields), scaler=scaler, model=model,
                clean_scaler_reused_from_same_fit=True,
                fit_sample_ids=tuple(r['sample_id'] for r in rows),
                fit_dialogue_ids=tuple(sorted({r['dialogue_id'] for r in rows})))


def scoring(rows, *, _bootstrap_repeats=shared.BOOTSTRAPS):
    fit = [r for r in rows if r['role'] == 'score_fit']
    evaluation = [r for r in rows if r['role'] == 'score_eval']
    common_fit = [r for r in fit if available(r, METHODS['E3'])]
    common_eval = np.asarray([available(r, METHODS['E3']) for r in evaluation], dtype=bool)
    y = np.asarray([r['original_correct'] for r in evaluation])
    weights, bootstrap = shared.bootstrap_weights([r['dialogue_id'] for r in evaluation], _bootstrap_repeats)
    fitted, records, failures = {}, {}, {}

    def attempt(key, selected, fields, reference=None):
        if not support(selected)['two_outcomes']:
            failures[key] = dict(status='UNAVAILABLE', reason='Fit requires both correct and error outcomes', support=support(selected))
            return
        try:
            fitted[key] = fit_fixed(selected, fields, reference)
            records[key] = dict(status='FITTED', **shared.model_record(fitted[key]))
        except (ValueError, FloatingPointError, shared.ConvergenceWarning) as error:
            failures[key] = dict(status='FAILED_FIXED_FIT', error_type=type(error).__name__,
                                 reason=str(error), support=support(selected))

    attempt('E0_full_fit', fit, METHODS['E0'])
    attempt('E0_common_fit', common_fit, METHODS['E0'])
    for name in ('E1', 'E2', 'E3'):
        if 'E0_common_fit' in fitted:
            attempt(name + '_common_fit', common_fit, METHODS[name], fitted['E0_common_fit'])
        else:
            failures[name + '_common_fit'] = dict(status='UNAVAILABLE', reason='Common-fit E0/scaler unavailable', support=support(common_fit))
    full = shared.predict(fitted['E0_full_fit'], evaluation) if 'E0_full_fit' in fitted else None
    common_predictions, deployment, used = {}, {}, {}
    for name, fields in METHODS.items():
        selector = np.asarray([available(r, fields) for r in evaluation], dtype=bool)
        values = np.full(len(evaluation), np.nan)
        key = name + '_common_fit'
        if key in fitted and selector.any():
            selected = [r for r, flag in zip(evaluation, selector) if flag]
            values[selector] = shared.predict(fitted[key], selected)
        common_predictions[name] = values
        if full is not None:
            policy = full.copy()
            applied = selector & np.isfinite(values) if name != 'E0' else np.zeros(len(evaluation), bool)
            policy[applied] = values[applied]
            if not np.array_equal(policy[~applied], full[~applied]):
                raise AssertionError('Missing/unsupported enhancement must use exact E0-full fallback')
            deployment[name], used[name] = policy, applied
    comparisons = []

    def contrast(scope, reference, new, selector, predictions):
        first, second = predictions.get(reference), predictions.get(new)
        data = [r for r, flag in zip(evaluation, selector) if flag]
        valid = (first is not None and second is not None
                 and np.isfinite(first[selector]).all() and np.isfinite(second[selector]).all())
        base = dict(scope=scope, reference=reference, new=new, evaluation_support=support(data),
                    comparison=reference + '_to_' + new)
        if not valid:
            base.update(status='UNAVAILABLE_MODEL', reason='At least one fixed fitted model is unavailable; no substitute fit/cohort')
        else:
            result = shared.compare(y, first, second, weights, selector)
            base.update(status='COMPUTED' if selector.any() else 'NO_EVAL_SUPPORT', **result)
            base['new_minus_reference'] = {metric: (result['augmented'][metric] - result['baseline'][metric]
                                                   if result['augmented'][metric] is not None and result['baseline'][metric] is not None else None)
                                           for metric in shared.METRICS}
        comparisons.append(base)

    contrasts = [('E0', 'E1'), ('E0', 'E2'), ('E0', 'E3'), ('E1', 'E2')]
    for first, second in contrasts:
        contrast('PRIMARY_COMMON_FIT_COMMON_EVAL', first, second, common_eval, common_predictions)
    for first, second in contrasts:
        contrast('SECONDARY_FULL_QUEUE_FALLBACK_POLICY', first, second,
                 np.ones(len(evaluation), bool), deployment)
    coverage = {}
    for name, fields in METHODS.items():
        coverage[name] = {role: support([r for r in subset if available(r, fields)])
                          for role, subset in [('score_fit', fit), ('score_eval', evaluation)]}
        coverage[name]['fit_fraction'] = coverage[name]['score_fit']['n'] / len(fit)
        coverage[name]['eval_fraction'] = coverage[name]['score_eval']['n'] / len(evaluation)
        if name in used:
            coverage[name]['deployment_applied'] = int(used[name].sum())
            coverage[name]['deployment_fallback'] = int((~used[name]).sum()) if name != 'E0' else 0
            if name == 'E0': coverage[name]['direct_baseline_rows'] = len(evaluation)
    predictions = []
    for i, row in enumerate(evaluation):
        record = dict(row, common_eval_eligible=bool(common_eval[i]),
                      E0_full_fit_R=float(full[i]) if full is not None else None)
        for name in METHODS:
            value = common_predictions[name][i]
            record[name + '_common_fit_R'] = float(value) if np.isfinite(value) else None
            record[name + '_policy_R'] = float(deployment[name][i]) if name in deployment else None
            record[name + '_enhancement_used'] = bool(used[name][i]) if name in used else False
            record[name + '_fallback_to_E0_full'] = bool(not used[name][i]) if name in used and name != 'E0' else False if name == 'E0' and full is not None else None
        predictions.append(record)
    report = dict(status='COMPLETE_CPU_SCORING' if not failures else 'COMPLETED_WITH_UNAVAILABLE_MODELS',
                  outcome='R=P(original correct=1); AUROC/AP positive=error, score=1-R',
                  methods={k: list(v) for k, v in METHODS.items()},
                  primary='E2 versus E1 Binary NLL; identical jointly covered Fit/Eval for E0--E3',
                  secondary='All method-versus-E0 contrasts; E3; full-queue exact-fallback policies',
                  point_estimator='Utterance-weighted; equal-weight original rows, Dialogue cluster bootstrap for uncertainty',
                  fit_support=support(fit), eval_support=support(evaluation),
                  common_fit_support=support(common_fit),
                  common_eval_support=support([r for r, flag in zip(evaluation, common_eval) if flag]),
                  common_fit_sample_ids=[r['sample_id'] for r in common_fit],
                  common_eval_sample_ids=[r['sample_id'] for r, flag in zip(evaluation, common_eval) if flag],
                  coverage=coverage, bootstrap=bootstrap, comparisons=comparisons, failures=failures,
                  settings=dict(logistic=shared.LOGISTIC, scaler='StandardScaler Fit only, ddof=0; constant scale=1; exact same-fit E0 clean prefix shared before enhanced fitting',
                                confidence_clip=shared.CONFIDENCE_EPS, binary_nll_clip=1e-15,
                                hyperparameter_search=False, predictive_model_training=False, cpu_only=True),
                  support_condition='Common-fit models need both outcomes. NLL/Brier need nonempty Eval; AUROC/AP also need both Eval outcomes. Empty/single-class bootstrap draws remain explicitly unavailable. No arbitrary sample deletion or substitute cohort.',
                  limitations=['Reused score_eval has already been viewed; exploratory, not independent confirmation.',
                               'Coverage is the joint semantic/corruption eligibility subset, not a representative random cohort.',
                               'Deployment changes fitting coverage relative to E0-full: it cannot isolate feature increment.',
                               '2000 paired Dialogue draws hold fitted scoring parameters fixed; conditional evaluation uncertainty, no score-fit or model-training bootstrap.',
                               '95% intervals are descriptive, unadjusted for multiple comparisons; E3 is secondary.',
                               'AI semantic review and automatic QC do not establish human-confirmed label preservation.'],
                  versions=dict(numpy=np.__version__, sklearn=shared.sklearn.__version__))
    return report, predictions, dict(records, **failures)


def flattened(report):
    rows = []
    for result in report['comparisons']:
        for metric in shared.METRICS:
            stats = result.get('paired_dialogue_improvement', {}).get(metric, {})
            rows.append(dict(scope=result['scope'], reference=result['reference'], new=result['new'],
                             metric=metric, comparison_status=result['status'], n=result['evaluation_support']['n'],
                             dialogues=result['evaluation_support']['dialogues'],
                             baseline=result.get('baseline', {}).get(metric),
                             augmented=result.get('augmented', {}).get(metric),
                             new_minus_reference=result.get('new_minus_reference', {}).get(metric),
                             improvement=stats.get('improvement'),
                             improvement_ci_low=stats.get('improvement_ci_low'),
                             improvement_ci_high=stats.get('improvement_ci_high'),
                             valid_draws=stats.get('valid_draws', 0),
                             requested_draws=report['bootstrap']['requested'],
                             positive_means='new method improves reference',
                             uncertainty_status=stats.get('status', 'UNAVAILABLE_MODEL')))
    return rows


def write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = shared.csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def markdown(report):
    common = report['common_fit_support'], report['common_eval_support']
    text = ['# E0–E3 探索性评分比较', '',
            f"状态：{report['status']}。模型仅使用 score_fit 拟合，score_eval 已在历史分析中查看，本结果不能作为独立确认。", '',
            f"完整队列 Fit/Eval={report['fit_support']['n']}/{report['eval_support']['n']}；共同覆盖 Fit/Eval={common[0]['n']}/{common[1]['n']}，Dialogue={common[0]['dialogues']}/{common[1]['dialogues']}。", '',
            'E0=logit(C)；E1=E0+语义 D_pred；E2=E0+拼写干扰 D_pred；E3=E0+两种 D_pred（次要）。标准化只在同一共同 Fit 拟合，固定 L2/C=1，不调参。', '',
            '| 范围 | 对比（新−参照） | NLL 差值 | NLL 改善 95% CI | 有效抽样 |',
            '|---|---|---:|---|---:|']
    fmt = lambda value: 'NA' if value is None else f'{value:.6g}'
    for comparison in report['comparisons']:
        stats = comparison.get('paired_dialogue_improvement', {}).get('binary_nll', {})
        raw = comparison.get('new_minus_reference', {}).get('binary_nll')
        text.append(f"| {comparison['scope']} | {comparison['new']}−{comparison['reference']} | {fmt(raw)} | [{fmt(stats.get('improvement_ci_low'))}, {fmt(stats.get('improvement_ci_high'))}] | {stats.get('valid_draws', 0)}/{report['bootstrap']['requested']} |")
    text += ['', 'NLL/Brier 的原始差值小于零更好；改善值及其区间大于零更好。AUROC/AP 使用错误为正类、1−R 排序；AP 为阶梯式 average precision。完整指标和失败记录见 scoring_report.json 与 comparisons.csv。', '',
             '主比较为共同覆盖样本上的 E2 对 E1，E1/E2 对同一 E0 说明各自增量，E3 为次要。所有方法共享一次 2000 次配对 Dialogue Bootstrap，不重新拟合参数；区间未做多重比较校正。', '',
             '完整队列部署范围中，E0 用全部 score_fit 重新拟合；增强模型使用共同 Fit，在其所需特征齐全时应用，否则逐条精确回退同一 E0-full。该比较同时改变拟合覆盖，不能据此单独归因特征增益。', '',
             '| 方法 | 可用 Fit（占全部） | 可用 Eval（占全部） | 实际应用增强 | 精确回退 |', '|---|---:|---:|---:|---:|']
    for name, coverage in report['coverage'].items():
        text.append(f"| {name} | {coverage['score_fit']['n']} ({coverage['fit_fraction']:.2%}) | {coverage['score_eval']['n']} ({coverage['eval_fraction']:.2%}) | {coverage.get('deployment_applied', 'NA')} | {coverage.get('deployment_fallback', 'NA')} |")
    primary = next(c for c in report['comparisons'] if c['scope'] == 'PRIMARY_COMMON_FIT_COMMON_EVAL' and c['reference'] == 'E1' and c['new'] == 'E2')
    stat = primary.get('paired_dialogue_improvement', {}).get('binary_nll', {})
    effect, low, high = stat.get('improvement'), stat.get('improvement_ci_low'), stat.get('improvement_ci_high')
    if effect is None:
        conclusion = '共同覆盖主比较没有可估计的 NLL 结果，不能判断两类行为信息谁更好。'
    elif low is not None and low > 0:
        conclusion = '共同覆盖点估计中 E2 的 NLL 较 E1 更低，条件配对区间位于改善方向；这是复用评估集上的探索性信号，不能视为独立确认或部署增益。'
    elif high is not None and high < 0:
        conclusion = '共同覆盖点估计中 E2 的 NLL 较 E1 更高，条件配对区间位于恶化方向；当前探索结果不支持拼写干扰行为特征优于语义行为特征。'
    else:
        conclusion = '共同覆盖 NLL 的条件配对区间未排除零，当前结果不足以判断拼写干扰行为特征优于语义行为特征。'
    text += ['', conclusion]
    if stat and stat['valid_draws'] != stat['requested_draws']:
        text += ['该主比较存在缺少支持的 Bootstrap 抽样，区间仅使用有效抽样，不能作为完整支持的区间解释。']
    text += ['', '支持条件：共同 Fit 必须同时包含正确与错误；NLL/Brier 需要非空 Eval，AUROC/AP 还需要两类。缺失、无模型、空抽样和单类抽样均保留；没有替换目标、缩小 K 或按结果删条件。', '',
             '覆盖稀疏、AI 审核语义不确定、复用既有评估和固定拟合参数的条件区间均限制结论。本实现不读取媒体、文本对话、Test 或官方 Validation，也不训练或推理 CMERC。', '']
    if report['failures']:
        text += ['失败/不可用模型：', '```json', json.dumps(report['failures'], ensure_ascii=False, indent=2), '```', '']
    return '\n'.join(text)


def verify_sources(args):
    declaration = json.loads(args.provenance.read_text())
    declared = shared.provenance_check(declaration, args.input, args.manifest,
                                       args.utterance_manifest, True, args.split_lock)
    if declaration['data_kind'] != 'declared_real':
        raise ValueError('Production execution requires attributed declared_real; use --self-test for synthetic')
    protocol = json.loads(args.protocol.read_text()); validate_protocol(protocol)
    if declaration.get('protocol_sha256') != shared.sha(args.protocol):
        raise ValueError('Scoring protocol provenance hash mismatch')
    bindings = declaration.get('source_bindings', {})
    if not bindings:
        raise ValueError('Required inference verification and source CSV bindings absent')
    for path, digest in bindings.items():
        if Path(path).suffix.lower() not in {'.csv', '.json', '.py'}:
            raise ValueError('Scorer may not open private dialogue/media/features/weights: ' + path)
        if not Path(path).is_absolute() or shared.sha(path) != digest:
            raise ValueError('Scoring source binding mismatch: ' + path)
    verification_path = args.input.parent / 'inference_verification.json'
    if str(verification_path.resolve()) not in bindings:
        raise ValueError('Inference verification must be explicitly hash-bound')
    verification = json.loads(verification_path.read_text())
    expected = dict(status='COMPLETE', no_model_fit=True, no_test=True, no_new_validation=True,
                    clean_utterances=2976, corruption_targets=192, planned_corruption_versions=384,
                    clean_semantic_rows_bit_exact=True, semantic_probabilities_reused=True,
                    unexpected_failures=0)
    if any(verification.get(key) != value for key, value in expected.items()):
        raise ValueError('Incomplete/failed inference verification cannot enter scoring')
    if verification.get('protocol_sha256') != shared.sha(args.protocol):
        raise ValueError('Inference verification protocol differs')
    # Complete engine lineage also contains private dataset/review/features and
    # weights. Keep those declared hashes in the bound verification; the engine
    # and root audit validate them. This CPU scorer never opens those assets.
    for stem in ('source_score_inputs', 'source_semantic_variants', 'source_prediction_verification'):
        path, digest = Path(verification[stem + '_path']), verification[stem + '_sha256']
        if path.suffix.lower() not in {'.csv', '.json'} or bindings.get(str(path.resolve())) != digest:
            raise ValueError('Numeric source/metadata is not explicitly hash-bound: ' + stem)
        if shared.sha(path) != digest:
            raise ValueError('Numeric inference source changed: ' + stem)
    for name, digest in verification.get('outputs', {}).items():
        path = args.input.parent / name
        if path.suffix.lower() not in {'.csv', '.json'}:
            raise ValueError('Unexpected private inference output requested by scorer')
        if shared.sha(path) != digest:
            raise ValueError('Inference output changed: ' + name)
    if verification.get('outputs', {}).get(args.input.name) != shared.sha(args.input):
        raise ValueError('Inference verification does not bind exact scoring CSV')
    return declared, declaration


def run(args):
    report_path = RUN / 'SCORING_COMPARISON.md'
    staging = args.output.with_name(args.output.name + '.tmp')
    if args.output.exists() or staging.exists() or report_path.exists():
        raise FileExistsError('Preserve existing scoring outputs and report')
    declared, declaration = verify_sources(args)
    raw = shared.read_csv(args.input)
    rows, validation = validate_inputs(raw, shared.read_csv(args.manifest), shared.read_csv(args.utterance_manifest))
    report, predictions, parameters = scoring(rows)
    report.update(data_provenance=declared, scalar_validation=validation,
                  source_declaration=declaration,
                  input_status_counts={key: dict(Counter(row.get(key, '') for row in raw))
                                       for key in raw[0]
                                       if key.endswith(('_missing_reason', '_available', '_status'))},
                  private_lineage_policy='Bound inference verification records private lineage; engine/root audit validate it. Scorer independently rehashes only supplied numeric CSV/code/JSON metadata, never private dialogue/review/features/weights.',
                  bindings={str(p.resolve()): shared.sha(p) for p in (args.input, args.provenance, args.manifest, args.utterance_manifest, args.split_lock, args.protocol, Path(__file__), SCORER_PATH)})
    # Recheck after fitting; editing inputs during scoring invalidates completion.
    verify_sources(args)
    for path, digest in report['bindings'].items():
        if shared.sha(path) != digest:
            raise RuntimeError('Scoring input/code changed while running')
    staging.mkdir(parents=True, exist_ok=False)
    for name, content in [('scoring_report.json', report), ('fitted_parameters.json', parameters)]:
        (staging / name).write_text(json.dumps(content, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
    original = {r['sample_id']: r for r in raw}
    write_csv(staging / 'evaluation_predictions.csv', [dict(original[r['sample_id']], **r) for r in predictions])
    write_csv(staging / 'comparisons.csv', flattened(report))
    narrative = markdown(report)
    staging.rename(args.output)
    with report_path.open('x', encoding='utf-8') as stream:
        stream.write(narrative)
    completion = dict(status=report['status'], script_sha256=shared.sha(__file__),
                      files={str(p.resolve()): shared.sha(p) for p in [report_path, *sorted(args.output.iterdir())]},
                      no_predictive_inference=True, no_test=True, no_new_validation=True)
    (args.output / 'completion.json').write_text(json.dumps(completion, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], common_fit=report['common_fit_support'],
                          common_eval=report['common_eval_support'], output=str(args.output))), flush=True)


def self_test():
    """Bounded synthetic checks only; no real CSV, labels or predictions read."""
    import unittest
    from sklearn.metrics import average_precision_score, roc_auc_score

    class Contracts(unittest.TestCase):
        def rows(self):
            rows = []
            for role in ('score_fit', 'score_eval'):
                for i in range(16):
                    confidence = .51 + .025 * i
                    rows.append(dict(sample_id=f'{role}-{i}', dialogue_id=f'{role}-d{i//2}', role=role,
                                     original_correct=str(i % 2), original_confidence=str(confidence),
                                     clean_margin='0.1', clean_entropy='1.2',
                                     D_pred='' if i % 5 == 0 else str(.01 * (i - 7)),
                                     corruption_d_pred='' if i % 7 == 0 else str(.02 * (i - 7))))
            return validate_inputs(rows, _expected_counts={'score_fit': 16, 'score_eval': 16},
                                   _expected_dialogues={'score_fit': 8, 'score_eval': 8})[0]

        def test_shared_scaler_and_old_single_behavior_equivalence(self):
            rows = [r for r in self.rows() if r['role'] == 'score_fit' and available(r, METHODS['E3'])]
            clean = fit_fixed(rows, METHODS['E0'])
            mine = fit_fixed(rows, METHODS['E1'], clean)
            old = shared.fit_logistic(rows, METHODS['E1'], clean_reference=clean)
            np.testing.assert_array_equal(mine['scaler'].mean_[:1], clean['scaler'].mean_)
            np.testing.assert_allclose(shared.predict(mine, rows), shared.predict(old, rows), atol=1e-14)

        def test_common_cohort_and_exact_fallback(self):
            rows = self.rows(); report, predictions, models = scoring(rows, _bootstrap_repeats=32)
            common = tuple(report['common_fit_sample_ids'])
            self.assertTrue(common)
            for name in METHODS:
                self.assertEqual(tuple(models[name + '_common_fit']['fit_sample_ids']), common)
            self.assertEqual(len({report['bootstrap']['draw_indices_sha256']}), 1)
            for row in predictions:
                for name in METHODS:
                    if row[name + '_fallback_to_E0_full']:
                        self.assertEqual(row[name + '_policy_R'], row['E0_full_fit_R'])
            self.assertEqual(len(report['comparisons']), 8)

        def test_evaluation_changes_do_not_change_fitted_scalers_or_models(self):
            rows = self.rows(); _, _, first = scoring(rows, _bootstrap_repeats=8)
            altered = [dict(r, original_correct=1-r['original_correct'], corruption_d_pred=.99,
                            D_pred=-.99, logit_confidence=50) if r['role'] == 'score_eval' else dict(r) for r in rows]
            _, _, second = scoring(altered, _bootstrap_repeats=8)
            self.assertEqual(first, second)

        def test_no_joint_fit_keeps_queue_and_failures(self):
            rows = self.rows()
            for row in rows:
                if row['role'] == 'score_fit': row['corruption_d_pred'] = None
            report, predictions, models = scoring(rows, _bootstrap_repeats=8)
            self.assertEqual(report['common_fit_support']['n'], 0)
            self.assertEqual(len(predictions), 16)
            self.assertIn('E3_common_fit', report['failures'])
            for row in predictions:
                self.assertEqual(row['E2_policy_R'], row['E0_full_fit_R'])

        def test_error_positive_ranking_and_sparse_draw_support(self):
            y = np.asarray([1, 0, 1, 0]); reliability = np.asarray([.9, .3, .7, .1])
            result = shared.metric_point(y, reliability)
            self.assertAlmostEqual(result['auroc_error'], roc_auc_score(1-y, 1-reliability))
            self.assertAlmostEqual(result['auprc_error'], average_precision_score(1-y, 1-reliability))
            self.assertIsNone(shared.metric_point(np.ones(2), np.asarray([.2, .8]))['auroc_error'])

        def test_real_count_guard_and_temp_only_csv(self):
            rows = self.rows()
            with self.assertRaises(ValueError):
                validate_inputs(rows)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'fixture.csv'
                write_csv(path, rows)
                self.assertEqual(len(shared.read_csv(path)), 32)

        def test_report_failure_and_csv_serialization(self):
            rows = self.rows()
            for empty in (False, True):
                if empty:
                    for row in rows:
                        if row['role'] == 'score_fit': row['corruption_d_pred'] = None
                report, predictions, models = scoring(rows, _bootstrap_repeats=8)
                table = flattened(report)
                self.assertEqual(len(table), 8 * 4)
                self.assertTrue(all('comparison_status' in r and 'uncertainty_status' in r for r in table))
                json.dumps(report, allow_nan=False); json.dumps(models, allow_nan=False)
                self.assertIn('E2−E1', markdown(report))
                with tempfile.TemporaryDirectory() as tmp:
                    write_csv(Path(tmp) / 'comparisons.csv', table)
                    write_csv(Path(tmp) / 'predictions.csv', predictions)

        def test_lineage_checks_numeric_sources_but_never_private_assets(self):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                def put(name, value):
                    path = root / name
                    path.write_text(json.dumps(value) if isinstance(value, dict) else value)
                    return path
                manifests = [put('SPLIT_DIALOGUES.csv', 'dialogue_id,role\n'),
                             put('SPLIT_UTTERANCES.csv', 'sample_id,dialogue_id,role\n')]
                lock = put('SPLIT_LOCK.json', {'hashes': {p.name: shared.sha(p) for p in manifests}})
                protocol = put('EXPERIMENT_PROTOCOL.json', dict(dataset='MELD', test_access=False,
                    official_validation_new_inference=False,
                    scoring=dict(comparison={k: list(v) for k, v in METHODS.items()}, C=1,
                                 scorer_sha256=SCORER_SHA,
                                 bootstrap=dict(seed=shared.SEED, replicates=shared.BOOTSTRAPS))))
                source_inputs = put('source_score_inputs.csv', 'numeric\n1\n')
                source_variants = put('source_semantic_variants.csv', 'numeric\n1\n')
                source_verification = put('source_prediction_verification.json', {'status': 'COMPLETE'})
                score_input = put('score_inputs.csv', 'numeric\n1\n')
                verification = dict(status='COMPLETE', no_model_fit=True, no_test=True,
                    no_new_validation=True, clean_utterances=2976, corruption_targets=192,
                    planned_corruption_versions=384, clean_semantic_rows_bit_exact=True,
                    semantic_probabilities_reused=True, unexpected_failures=0,
                    protocol_sha256=shared.sha(protocol),
                    source_score_inputs_path=str(source_inputs), source_score_inputs_sha256=shared.sha(source_inputs),
                    source_semantic_variants_path=str(source_variants), source_semantic_variants_sha256=shared.sha(source_variants),
                    source_prediction_verification_path=str(source_verification), source_prediction_verification_sha256=shared.sha(source_verification),
                    source_bindings={'/never-open-this-private-path/dataset.pt': 'metadata-only'},
                    outputs={score_input.name: shared.sha(score_input)})
                verified_path = put('inference_verification.json', verification)
                declaration = dict(data_kind='declared_real', declarant='synthetic test',
                    declaration_note='Temporary synthetic provenance fixture; not real data',
                    confidence_definition='original_max_softmax_probability', outcome_definition='original_correct_1',
                    split_frozen=True, input_sha256=shared.sha(score_input),
                    dialogue_manifest_sha256=shared.sha(manifests[0]),
                    utterance_manifest_sha256=shared.sha(manifests[1]), split_lock_sha256=shared.sha(lock),
                    protocol_sha256=shared.sha(protocol),
                    source_bindings={str(p): shared.sha(p) for p in [verified_path, source_inputs, source_variants, source_verification]})
                provenance = put('SCORING_INPUT_PROVENANCE.json', declaration)
                args = argparse.Namespace(provenance=provenance, input=score_input, manifest=manifests[0],
                                          utterance_manifest=manifests[1], split_lock=lock, protocol=protocol)
                verify_sources(args)  # nonexistent private path was not opened.
                verification['unexpected_failures'] = 1
                put(verified_path.name, verification)
                declaration['source_bindings'][str(verified_path)] = shared.sha(verified_path)
                put(provenance.name, declaration)
                with self.assertRaises(ValueError): verify_sources(args)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful(): raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--input', type=Path, default=RUN / 'results/text_comparison/score_inputs.csv')
    parser.add_argument('--provenance', type=Path, default=RUN / 'provenance/SCORING_INPUT_PROVENANCE.json')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--utterance-manifest', type=Path)
    parser.add_argument('--split-lock', type=Path)
    parser.add_argument('--protocol', type=Path, default=RUN / 'EXPERIMENT_PROTOCOL.json')
    parser.add_argument('--output', type=Path, default=RUN / 'results/scoring')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if any(path is None for path in (args.manifest, args.utterance_manifest, args.split_lock)):
        parser.error('Real scoring requires --manifest, --utterance-manifest and --split-lock')
    run(args)


if __name__ == '__main__':
    main()
