#!/usr/bin/env python3
"""Numeric-only, Train exploratory D_pred versus D_abs CPU scoring.

No source text, model/weights, Test or official Validation is opened. The exact
prior fixed scorer supplies all estimators, calibration metrics and bootstrap
semantics. All four primary arms share K2-eligible fitting/evaluation rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
COMPARISON = RUN.parent / '20261122T151649Z_meld_dpred_corruption/scripts/compare_scores.py'
COMPARISON_SHA = 'da49fe2453a3b63d1d7a2d9d374d2263f1316a43e3b8b45da71855423d373e97'
if hashlib.sha256(COMPARISON.read_bytes()).hexdigest() != COMPARISON_SHA:
    raise RuntimeError('Pinned comparison implementation changed')
spec = importlib.util.spec_from_file_location('_pilot_dabs_fixed_comparison', COMPARISON)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
np, shared = core.np, core.shared
METHODS = {'E0': ('logit_confidence',), 'E1': ('logit_confidence', 'D_pred'),
           'E2': ('logit_confidence', 'D_abs'), 'E3': ('logit_confidence', 'D_pred', 'D_abs')}
core.METHODS = METHODS  # isolated imported module; no historical source is edited.
ROLES = {'pilot_fit': 'score_fit', 'pilot_eval': 'score_eval'}
DISPLAY_ROLES = {v: k for k, v in ROLES.items()}
TOL = 1e-7


def validate_protocol(protocol):
    scoring = protocol['scoring']
    if (protocol['dataset'] != 'MELD' or protocol['source_role'] != 'score_fit'
            or protocol['test_access'] is not False or protocol['official_validation_access'] is not False
            or protocol['historical_score_eval_used'] is not False
            or protocol['new_independent_evaluation'] is not False
            or scoring['arms'] != {k: list(v) for k, v in METHODS.items()}
            or scoring['logistic_l2_C'] != shared.LOGISTIC['C']
            or scoring['bootstrap_dialogue_replicates'] != shared.BOOTSTRAPS
            or scoring['bootstrap_seed'] != shared.SEED or scoring['scorer_sha256'] != core.SCORER_SHA
            or protocol['cohort']['pilot_fit_dialogues'] != 75
            or protocol['cohort']['pilot_eval_dialogues'] != 60):
        raise ValueError('Wrong fixed Train-only D_pred/D_abs protocol')


def map_role(row):
    if row.get('role') not in ROLES:
        raise ValueError('Only exact pilot_fit/pilot_eval roles are permitted')
    return dict(row, role=ROLES[row['role']])


def validate_inputs(raw, cohort, *, _dialogues=(75, 60), _utterances=(741, 671)):
    mapped, manifest = [map_role(r) for r in raw], [map_role(r) for r in cohort]
    rows, audit = shared.validate_rows(mapped, utterance_manifest=manifest)
    source = {r['sample_id']: r for r in raw}
    for row in rows:
        original = source[row['sample_id']]
        if 'original_role' in original and original['original_role'] != 'score_fit':
            raise ValueError('Pilot rows may derive only from historical score_fit')
        if not original.get('original_probabilities', '').strip():
            raise ValueError('Every pilot row requires its complete clean probability vector')
        if len(json.loads(original['original_probabilities'])) != 7:
            raise ValueError('MELD clean probabilities must contain seven classes')
        if 'D_abs' not in original:
            raise ValueError('Missing D_abs column')
        text = original['D_abs']
        absolute = None if str(text).strip().lower() in core.MISSING else shared.finite_number(text, 'D_abs')
        signed = row['D_pred']
        if (signed is None) != (absolute is None):
            raise ValueError('The same K2 must produce both D_pred/D_abs or neither')
        if 'k2_available' in original:
            flag = str(original['k2_available']).lower().strip()
            if flag not in {'true', 'false', '0', '1'} or (flag in {'true', '1'}) != (absolute is not None):
                raise ValueError('K2 eligibility disagrees with joint scalar presence')
        if absolute is not None:
            confidence = row['original_confidence']
            envelope = (2 * confidence - 1) * signed + 2 * confidence * (1 - confidence)
            if (not 0 <= absolute <= 1 + TOL or absolute + TOL < abs(signed)
                    or not confidence - 1 - TOL <= signed <= confidence + TOL
                    or absolute > max(confidence, 1 - confidence) + TOL
                    or absolute > envelope + TOL):
                raise ValueError('D_pred/D_abs violate probability bounds or Jensen inequality')
        row.update(D_abs=absolute, cancellation=absolute - abs(signed) if absolute is not None else None)
    for role, expected, utterances in zip(('score_fit', 'score_eval'), _dialogues, _utterances):
        selected = [r for r in rows if r['role'] == role]
        if len({r['dialogue_id'] for r in selected}) != expected or len(selected) != utterances:
            raise ValueError('Exact frozen pilot Dialogue/utterance count mismatch')
    audit.update(exact_pilot_cohort_verified=True, joint_K2_missingness_verified=True,
                 probability_bounds_verified=True, roundoff_tolerance=TOL,
                 cancellation_roundoff='D_abs-abs(D_pred) retained, including tiny negatives within tolerance')
    return rows, audit


def compare_extra(report, predictions, rows):
    evaluation = [r for r in rows if r['role'] == 'score_eval']
    y = np.asarray([r['original_correct'] for r in evaluation])
    weights, boot = shared.bootstrap_weights([r['dialogue_id'] for r in evaluation], report['bootstrap']['requested'])
    if boot != report['bootstrap']:
        raise AssertionError('Additional contrasts must use the exact same shared Dialogue draws')
    for scope, common in [('PRIMARY_COMMON_FIT_COMMON_EVAL', True),
                          ('SECONDARY_FULL_QUEUE_FALLBACK_POLICY', False)]:
        selector = np.asarray([r['common_eval_eligible'] for r in predictions], bool) if common else np.ones(len(evaluation), bool)
        suffix = '_common_fit_R' if common else '_policy_R'
        for reference in ('E1', 'E2'):
            base = dict(scope=scope, reference=reference, new='E3', comparison=reference + '_to_E3',
                        evaluation_support=core.support([r for r, flag in zip(evaluation, selector) if flag]))
            first = np.asarray([r[reference + suffix] if r[reference + suffix] is not None else np.nan for r in predictions])
            second = np.asarray([r['E3' + suffix] if r['E3' + suffix] is not None else np.nan for r in predictions])
            if np.isfinite(first[selector]).all() and np.isfinite(second[selector]).all():
                result = shared.compare(y, first, second, weights, selector)
                base.update(status='COMPUTED' if selector.any() else 'NO_EVAL_SUPPORT', **result)
                base['new_minus_reference'] = {key: result['augmented'][key] - result['baseline'][key]
                    if result['augmented'][key] is not None and result['baseline'][key] is not None else None
                    for key in shared.METRICS}
            else:
                base.update(status='UNAVAILABLE_MODEL', reason='No substitute fit/cohort when an arm is unavailable')
            report['comparisons'].append(base)
    return weights


def distribution(rows):
    result = []
    for role in ('score_fit', 'score_eval'):
        for outcome in ('all', 'correct', 'error'):
            selected = [r for r in rows if r['role'] == role and (outcome == 'all' or r['original_correct'] == int(outcome == 'correct'))]
            for feature in ('D_pred', 'D_abs', 'cancellation'):
                values = np.asarray([r[feature] for r in selected if r[feature] is not None], float)
                item = dict(role=DISPLAY_ROLES[role], actual=outcome, feature=feature,
                            n_total=len(selected), n_valid=len(values), n_missing=len(selected)-len(values))
                stats = dict(mean=None, std_ddof0=None, median=None, minimum=None, maximum=None,
                             q05=None, q25=None, q75=None, q95=None, negative=0, zero=0, positive=0)
                if len(values):
                    qs = np.quantile(values, [.05, .25, .5, .75, .95])
                    stats.update(mean=float(values.mean()), std_ddof0=float(values.std()), median=float(qs[2]),
                                 minimum=float(values.min()), maximum=float(values.max()),
                                 q05=float(qs[0]), q25=float(qs[1]), q75=float(qs[3]), q95=float(qs[4]),
                                 negative=int((values < 0).sum()), zero=int((values == 0).sum()), positive=int((values > 0).sum()))
                result.append(dict(item, **stats))
    return result


def draw_indices(dialogues, repeats):
    universe = sorted(set(dialogues))
    draws = np.random.RandomState(shared.SEED).randint(0, len(universe), (repeats, len(universe))) if universe else np.zeros((repeats, 0), dtype=np.int64)
    return universe, draws


def coefficient_bootstrap(fit, *, repeats=shared.BOOTSTRAPS):
    """2000 shared Fit-Dialogue draws; re-fit scaler and all four fixed arms.

    These are coefficient uncertainty diagnostics, not tuning and not the Eval
    performance interval. No evaluation rows or labels are accepted here.
    """
    if any(r['role'] != 'score_fit' for r in fit):
        raise ValueError('Coefficient bootstrap may receive eligible pilot_fit only')
    universe, draws = draw_indices([r['dialogue_id'] for r in fit], repeats)
    grouped = {did: [r for r in fit if r['dialogue_id'] == did] for did in universe}
    keys = [(arm, field) for arm, fields in METHODS.items() for field in fields]
    arrays = {arm + '__' + field: np.full((repeats, 2), np.nan) for arm, field in keys}
    statuses, started = Counter(), time.monotonic()
    for replicate, draw in enumerate(draws):
        selected = [r for i in draw for r in grouped[universe[i]]]
        if not core.support(selected)['two_outcomes']:
            statuses['NO_TWO_CLASS_FIT_SUPPORT'] += 1
            continue
        clean = None
        for arm, fields in METHODS.items():
            try:
                model = core.fit_fixed(selected, fields, clean if arm != 'E0' else None)
                if arm == 'E0': clean = model
                if arm != 'E0' and clean is None: raise ValueError('Shared E0 scaler unavailable')
                coefficient = model['model'].coef_[0]
                raw_coefficient = coefficient / model['scaler'].scale_
                for field, standardized, raw in zip(fields, coefficient, raw_coefficient):
                    arrays[arm + '__' + field][replicate] = (float(standardized), float(raw))
                statuses[arm + ':FITTED'] += 1
            except (ValueError, FloatingPointError, shared.ConvergenceWarning) as error:
                statuses[arm + ':' + type(error).__name__ + ':' + str(error)] += 1
                if arm == 'E0': break
        if (replicate + 1) % 100 == 0:
            print(json.dumps(dict(stage='fit_coefficient_bootstrap', completed=replicate + 1,
                                  requested=repeats, elapsed_seconds=round(time.monotonic()-started, 2))), flush=True)
    summary = []
    for (arm, field), values in zip(keys, arrays.values()):
        valid = np.isfinite(values).all(1)
        record = dict(model=arm, feature=field, valid_draws=int(valid.sum()), requested_draws=repeats,
                      negative_fraction=float((values[valid, 0] < 0).mean()) if valid.any() else None,
                      zero_fraction=float((values[valid, 0] == 0).mean()) if valid.any() else None,
                      positive_fraction=float((values[valid, 0] > 0).mean()) if valid.any() else None,
                      standardized_ci_low=None, standardized_ci_high=None, raw_ci_low=None, raw_ci_high=None)
        if valid.any():
            ci = np.quantile(values[valid], [.025, .975], axis=0)
            record.update(standardized_ci_low=float(ci[0, 0]), standardized_ci_high=float(ci[1, 0]),
                          raw_ci_low=float(ci[0, 1]), raw_ci_high=float(ci[1, 1]))
        summary.append(record)
    return dict(seed=shared.SEED, requested=repeats, unit='eligible pilot_fit Dialogue, shared across arms',
                dialogue_universe=universe, draw_indices_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
                statuses=dict(statuses), elapsed_seconds=time.monotonic()-started,
                interpretation='Refitted Fit-only standardization and fixed logistic; signs describe conditional associations, not causal effects or a tuning rule'), summary, arrays, draws


def coefficient_points(models):
    rows = []
    for arm in METHODS:
        model = models.get(arm + '_common_fit', {})
        if model.get('status') != 'FITTED':
            rows.append(dict(model=arm, feature=None, status=model.get('status', 'UNAVAILABLE'), reason=model.get('reason', '')))
            continue
        for feature, coefficient, scale in zip(model['features'], model['coefficient'][0], model['scaler_scale']):
            rows.append(dict(model=arm, feature=feature, status='FITTED', standardized_coefficient=coefficient,
                             raw_coefficient=coefficient/scale, direction='higher feature increases R(correct)' if coefficient > 0
                             else 'higher feature decreases R(correct), increases error risk' if coefficient < 0 else 'zero coefficient'))
    return rows


def evaluate_draws(report, predictions, rows, weights):
    evaluation = [r for r in rows if r['role'] == 'score_eval']
    y = np.asarray([r['original_correct'] for r in evaluation])
    arrays = {}
    for scope, common in [('common_eval', True), ('full_queue_policy', False)]:
        mask = np.asarray([r['common_eval_eligible'] for r in predictions], bool) if common else np.ones(len(y), bool)
        suffix = '_common_fit_R' if common else '_policy_R'
        for arm in METHODS:
            r = np.asarray([row[arm + suffix] if row[arm + suffix] is not None else np.nan for row in predictions])
            if np.isfinite(r[mask]).all():
                metrics = shared.metric_batch(y[mask], r[mask], weights[:, mask])
            else:
                metrics = {key: np.full(len(weights), np.nan) for key in shared.METRICS}
            for metric, values in metrics.items(): arrays[scope + '__' + arm + '__' + metric] = values
    return arrays


def verify_sources(args):
    protocol = json.loads(args.protocol.read_text()); validate_protocol(protocol)
    verification = json.loads(args.verification.read_text())
    expected = dict(status='COMPLETE', unexpected_failures=0, no_model_fit=True, no_test=True,
                    no_new_validation=True, old_score_eval_used=False, source_original_role='score_fit',
                    pilot_fit_dialogues=75, pilot_eval_dialogues=60, targets=135, planned_versions=270,
                    source_clean_rows_unchanged=True, new_independent_evaluation=False)
    if any(verification.get(k) != value for k, value in expected.items()):
        raise ValueError('Inference verification failed its frozen Train-only completeness gates')
    if verification['protocol_sha256'] != shared.sha(args.protocol):
        raise ValueError('Inference/scoring protocol hash mismatch')
    source = Path(verification['source_clean_path'])
    if source.suffix != '.csv' or shared.sha(source) != verification['source_clean_sha256']:
        raise ValueError('Numeric source-clean CSV hash mismatch')
    for name, digest in verification['outputs'].items():
        path = args.input.parent / name
        if path.suffix not in {'.csv', '.json'} or shared.sha(path) != digest:
            raise ValueError('Inference scalar output hash mismatch: ' + name)
    if verification['outputs'].get(args.input.name) != shared.sha(args.input):
        raise ValueError('Verification does not bind exact score_inputs.csv')
    raw, clean, cohort = shared.read_csv(args.input), shared.read_csv(source), shared.read_csv(args.cohort)
    if len(raw) != verification['clean_utterances'] or len(raw) != len(clean):
        raise ValueError('Natural pilot cohort count mismatch')
    for before, after in zip(clean, raw):
        if any(after.get(key) != value for key, value in before.items()):
            raise ValueError('Original clean numeric metadata/order changed')
    if args.cohort.name in verification.get('source_bindings', {}):
        raise ValueError('Use resolved absolute IDs for cohort provenance')
    cohort_hash = verification.get('source_bindings', {}).get(str(args.cohort.resolve()))
    if cohort_hash != shared.sha(args.cohort):
        raise ValueError('Inference verification must hash-bind the exact pilot cohort manifest')
    # Other full-lineage bindings may refer to private texts/features/weights.
    # They are declared engine/root provenance; this scalar script never opens them.
    return raw, cohort, verification, source


def narrative(report, distributions, points, signs):
    fmt = lambda x: 'NA' if x is None else f'{x:.6g}'
    text = ['# D_pred 与 D_abs 的探索性评分比较', '',
            '此实验仅使用历史 score_fit 内重新分出的 pilot_fit/pilot_eval，两个组的来源均已被查看。本轮不访问旧 score_eval、Test 或官方 Validation，不构成独立确认。', '',
            f"完整 Fit/Eval={report['fit_support']['n']}/{report['eval_support']['n']}；共同 K2 有效 Fit/Eval={report['common_fit_support']['n']}/{report['common_eval_support']['n']}。四个主比较模型使用完全相同的有效 Fit 行和 Eval 行。", '',
            'E0=logit(C)，E1=E0+D_pred，E2=E0+D_abs，E3=E0+两者。C 裁剪至 [1e−6,1−1e−6]；只在 Fit 标准化；固定 L2/C=1，无调参。D_pred 为原预测类概率的平均有符号下降，D_abs 为同两个版本下降量绝对值的平均；抵消量为 D_abs−|D_pred|。', '',
            '| 范围 | 新−参照 | Binary NLL 差值 | 改善 95% CI | 有效抽样 |', '|---|---|---:|---|---:|']
    for result in report['comparisons']:
        stat = result.get('paired_dialogue_improvement', {}).get('binary_nll', {})
        text.append(f"| {result['scope']} | {result['new']}−{result['reference']} | {fmt(result.get('new_minus_reference', {}).get('binary_nll'))} | [{fmt(stat.get('improvement_ci_low'))}, {fmt(stat.get('improvement_ci_high'))}] | {stat.get('valid_draws', 0)}/{report['bootstrap']['requested']} |")
    primary = next(r for r in report['comparisons'] if r['scope'].startswith('PRIMARY') and r['reference'] == 'E1' and r['new'] == 'E2')
    ci = primary.get('paired_dialogue_improvement', {}).get('binary_nll', {})
    if ci.get('improvement') is None:
        finding = '主比较支持不足，不能判断 D_abs 与 D_pred 的优劣。'
    elif ci.get('improvement_ci_low') is not None and ci['improvement_ci_low'] > 0:
        finding = '主比较中 E2 的 NLL 低于 E1，条件配对区间位于改善方向；这是复用数据上的探索性关联，仍需未查看数据确认。'
    elif ci.get('improvement_ci_high') is not None and ci['improvement_ci_high'] < 0:
        finding = '主比较中 E2 的 NLL 高于 E1，条件配对区间位于恶化方向；当前结果不支持 D_abs 优于 D_pred。'
    else:
        finding = '主比较的条件配对区间未排除零，当前结果不足以证明 D_abs 优于 D_pred。'
    text += ['', finding, '', 'NLL/Brier 原始差值越小越好；改善区间大于零表示新模型更好。AUROC/AP 以错误为正类、1−R 排序。E3 对 E1/E2 的比较为次要，未做多重比较校正。', '',
             '| 模型 | 有效 Fit | 有效 Eval | 完整队列增强应用 | 精确回退 |', '|---|---:|---:|---:|---:|']
    for arm, c in report['coverage'].items():
        text.append(f"| {arm} | {c['score_fit']['n']} ({c['fit_fraction']:.1%}) | {c['score_eval']['n']} ({c['eval_fraction']:.1%}) | {c.get('deployment_applied', 'NA')} | {c.get('deployment_fallback', 'NA')} |")
    text += ['', '完整队列范围：E0 在全部 pilot_fit 拟合；增强模型只在共同有效 Fit 拟合，对无 K2 特征的 Eval 行精确回退同一 E0。此比较同时改变拟合覆盖，不能单独归因特征增益。', '',
             '| 角色 | 原预测实际对错 | 行为量 | 有效/全部 | 均值 | 中位数 | 标准差 |', '|---|---|---|---:|---:|---:|---:|']
    for r in distributions:
        text.append(f"| {r['role']} | {r['actual']} | {r['feature']} | {r['n_valid']}/{r['n_total']} | {fmt(r['mean'])} | {fmt(r['median'])} | {fmt(r['std_ddof0'])} |")
    by_key = {(r['model'], r['feature']): r for r in signs}
    text += ['', '| 模型 | 特征 | 标准化点系数 | 原尺度系数 95% Fit CI | 负号比例 | 正号比例 | 有效 Fit 抽样 |', '|---|---|---:|---|---:|---:|---:|']
    for p in points:
        s = by_key.get((p['model'], p['feature']), {})
        text.append(f"| {p['model']} | {p['feature']} | {fmt(p.get('standardized_coefficient'))} | [{fmt(s.get('raw_ci_low'))}, {fmt(s.get('raw_ci_high'))}] | {fmt(s.get('negative_fraction'))} | {fmt(s.get('positive_fraction'))} | {s.get('valid_draws', 0)}/{report['fit_coefficient_bootstrap']['requested']} |")
    text += ['', 'R 表示预测正确概率：负系数表示该行为量越高，条件正确概率越低；正系数方向相反。Fit 系数 Bootstrap 每次重采样 Dialogue、重新拟合标准化和固定逻辑回归，所有方法共享抽样；无调参、无 Eval 信息。系数符号只描述条件关联；D_pred/D_abs 的共线性可能使 E3 符号不稳定。', '',
             'Eval 指标 Bootstrap 为固定拟合参数下的 2000 次共享 Dialogue 抽样；Fit 系数区间为另一组仅 Fit 的 2000 次抽样，不是性能区间。两组抽样均保存原索引与逐次结果。空支持、单类、非收敛和缺失全部保留，不替换目标或删除条件。', '',
             '没有原始文本或权重被此评分器加载。AI 审核与 Token QC 不证明人类确认的标签保持；有限覆盖及已查看数据限制任何校准增益解释。', '']
    return '\n'.join(text)


def run(args):
    output, report_path = args.output, RUN / 'D_PRED_D_ABS_COMPARISON.md'
    staging = output.with_name(output.name + '.tmp')
    if output.exists() or staging.exists() or report_path.exists():
        raise FileExistsError('Preserve existing scoring evidence')
    raw, cohort, verification, source = verify_sources(args)
    rows, audit = validate_inputs(raw, cohort)
    bindings = {str(p.resolve()): shared.sha(p) for p in (args.input, args.cohort, args.verification,
                 args.protocol, source, Path(__file__), COMPARISON, core.SCORER_PATH)}
    report, predictions, models = core.scoring(rows)
    weights = compare_extra(report, predictions, rows)
    distribution_rows = distribution(rows); points = coefficient_points(models)
    eligible_fit = [r for r in rows if r['role'] == 'score_fit' and core.available(r, METHODS['E3'])]
    coefficient_info, signs, coefficient_arrays, fit_draws = coefficient_bootstrap(eligible_fit)
    eval_metric_arrays = evaluate_draws(report, predictions, rows, weights)
    evaluation = [r for r in rows if r['role'] == 'score_eval']
    eval_universe, eval_draws = draw_indices([r['dialogue_id'] for r in evaluation], shared.BOOTSTRAPS)
    if hashlib.sha256(eval_draws.astype('<i8').tobytes()).hexdigest() != report['bootstrap']['draw_indices_sha256']:
        raise AssertionError('Saved evaluation draws differ from comparison draws')
    report.update(primary='E2(D_abs) versus E1(D_pred) Binary NLL on identical jointly eligible Fit/Eval',
                  secondary='All E0 increments; combined E3 versus each single behavior; full queue fallback',
                  role_mapping=ROLES, scalar_validation=audit, source_verification=verification,
                  fit_coefficient_bootstrap=coefficient_info, bindings=bindings,
                  artifact_schemas=dict(evaluation_bootstrap_draws='draw_indices[2000,60], sorted dialogue_universe, row_dialogue_ids, row_sample_ids, row_weights[2000,671]',
                    evaluation_metric_draws='32 arrays of 2000: {common_eval|full_queue_policy}__{E0|E1|E2|E3}__{binary_nll|brier|auroc_error|auprc_error}; NaN denotes unsupported draw',
                    fit_coefficient_draws='draw_indices[2000,n_eligible_fit_dialogues]; eight E{0..3}__feature arrays[2000,2], columns standardized/raw coefficient; NaN denotes unsupported/failed fit'),
                  logistic_scoring_models_fitted=True, cmerc_encoder_loaded=False,
                  input_status_counts={key: dict(Counter(r.get(key, '') for r in raw)) for key in raw[0]
                                       if key in {'k2_available', 'missing_reason'}},
                  limitations=['Both new groups derive from historically viewed score_fit; Train exploratory, not independent evaluation.',
                               'D_pred and D_abs use the same two reviewed variants. Coverage and outcome groups are fully retained.',
                               'Full-queue baseline/augmented fitting coverage differs; only common-fit/common-eval contrasts isolate adding behavior fields.',
                               'Intervals are descriptive and unadjusted; paired Eval draws condition on fitted scoring parameters.',
                               'Private inference lineage is recorded from verified engine/root metadata, not opened by this numeric scorer.'])
    # All model fitting and reporting occur before exposing a completed directory.
    verify_sources(args)
    if any(shared.sha(path) != digest for path, digest in bindings.items()):
        raise RuntimeError('Scalar source/code changed during scoring')
    staging.mkdir(parents=True, exist_ok=False)
    for name, obj in [('scoring_report.json', report), ('fitted_parameters.json', models)]:
        (staging / name).write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
    source_rows = {r['sample_id']: r for r in raw}
    for p in predictions: p['role'] = DISPLAY_ROLES[p['role']]
    core.write_csv(staging / 'evaluation_predictions.csv', [dict(source_rows[r['sample_id']], **r) for r in predictions])
    core.write_csv(staging / 'comparisons.csv', core.flattened(report))
    core.write_csv(staging / 'behavior_distributions.csv', distribution_rows)
    core.write_csv(staging / 'coefficient_points.csv', points)
    core.write_csv(staging / 'coefficient_sign_bootstrap.csv', signs)
    np.savez_compressed(staging / 'evaluation_bootstrap_draws.npz', draw_indices=eval_draws,
                        dialogue_universe=np.asarray(eval_universe), row_dialogue_ids=np.asarray([r['dialogue_id'] for r in evaluation]),
                        row_sample_ids=np.asarray([r['sample_id'] for r in evaluation]), row_weights=weights)
    np.savez_compressed(staging / 'evaluation_metric_draws.npz', **eval_metric_arrays)
    np.savez_compressed(staging / 'fit_coefficient_draws.npz', draw_indices=fit_draws,
                        dialogue_universe=np.asarray(coefficient_info['dialogue_universe']), **coefficient_arrays)
    text = narrative(report, distribution_rows, points, signs)
    staging.rename(output)
    with report_path.open('x', encoding='utf-8') as stream: stream.write(text)
    completion = dict(status=report['status'], files={str(p.resolve()): shared.sha(p)
                      for p in [report_path, *sorted(output.iterdir())]}, no_cmerc_fit=True,
                      scalar_logistic_fit=True, no_encoder_or_cmerc_loaded=True,
                      no_test=True, no_official_validation=True, no_old_score_eval=True)
    (output / 'completion.json').write_text(json.dumps(completion, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], fit=report['fit_support'], evaluation=report['eval_support'],
                          eligible_fit=report['common_fit_support'], eligible_eval=report['common_eval_support'])), flush=True)


def self_test():
    import unittest
    class Contracts(unittest.TestCase):
        def raw(self):
            rows = []
            for role in ROLES:
                for i in range(12):
                    c = .55 + i * .025
                    p = [c] + [(1-c)/6] * 6
                    signed, absolute = (.04, .06) if i % 2 else (-.03, .07)
                    missing = i % 5 == 0
                    rows.append(dict(sample_id=f'{role}-{i}', dialogue_id=f'{role}-d{i//3}',
                        role=role, original_role='score_fit', utterance_index=str(i % 3), utterance_id=str(i % 3),
                        original_correct=str(i % 2), original_confidence=str(c), clean_margin=str(c-(1-c)/6),
                        clean_entropy=str(-sum(x*np.log(x) for x in p)), original_probabilities=json.dumps(p),
                        D_pred='' if missing else str(signed), D_abs='' if missing else str(absolute),
                        k2_available=str(not missing)))
            return rows
        def parsed(self):
            raw = self.raw(); return validate_inputs(raw, raw, _dialogues=(4, 4), _utterances=(12, 12))[0]
        def test_exact_cohort_and_joint_missingness(self):
            raw = self.raw()
            validate_inputs(raw, raw, _dialogues=(4, 4), _utterances=(12, 12))
            for key, value in [('D_abs', '.01'), ('D_abs', '.9'), ('D_abs', ''), ('k2_available', 'False')]:
                altered = [dict(r) for r in raw]; altered[1][key] = value
                with self.assertRaises(ValueError): validate_inputs(altered, raw, _dialogues=(4, 4), _utterances=(12, 12))
            with self.assertRaises(ValueError): validate_inputs(raw[:-1], raw, _dialogues=(4, 4), _utterances=(12, 12))
        def test_common_arms_extra_contrasts_and_exact_fallback(self):
            rows = self.parsed(); report, predictions, models = core.scoring(rows, _bootstrap_repeats=16)
            weights = compare_extra(report, predictions, rows)
            self.assertEqual(len(report['comparisons']), 12)
            ids = report['common_fit_sample_ids']
            for arm in METHODS: self.assertEqual(models[arm+'_common_fit']['fit_sample_ids'], ids)
            for p in predictions:
                if not p['common_eval_eligible']:
                    for arm in ('E1', 'E2', 'E3'): self.assertEqual(p[arm+'_policy_R'], p['E0_full_fit_R'])
            self.assertEqual(len(evaluate_draws(report, predictions, rows, weights)), 32)
        def test_fit_bootstrap_and_distributions_have_no_eval_dependency(self):
            rows = self.parsed(); fit = [r for r in rows if r['role']=='score_fit' and r['D_abs'] is not None]
            info, summary, arrays, draws = coefficient_bootstrap(fit, repeats=8)
            self.assertEqual(draws.shape, (8, 4)); self.assertEqual(len(summary), 8)
            self.assertEqual(len(distribution(rows)), 18)
            with self.assertRaises(ValueError): coefficient_bootstrap(rows, repeats=8)
            for a in arrays.values(): self.assertEqual(a.shape, (8, 2))
        def test_eval_outcome_and_feature_changes_cannot_affect_fit(self):
            rows = self.parsed()
            first = core.scoring(rows, _bootstrap_repeats=8)[2]
            changed = [dict(r, original_correct=1-r['original_correct'], D_pred=.01, D_abs=.09,
                            cancellation=.08, logit_confidence=20) if r['role']=='score_eval' else dict(r) for r in rows]
            second = core.scoring(changed, _bootstrap_repeats=8)[2]
            self.assertEqual(first, second)
        def test_numeric_lineage_and_no_private_asset_access(self):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                raw = self.raw()
                clean = [{k:v for k,v in r.items() if k not in {'D_pred','D_abs','k2_available'}} for r in raw]
                paths = {name:root/name for name in ('score_inputs.csv','source_clean.csv','pilot_cohort.csv')}
                core.write_csv(paths['score_inputs.csv'],raw)
                core.write_csv(paths['source_clean.csv'],clean)
                core.write_csv(paths['pilot_cohort.csv'],clean)
                protocol = dict(dataset='MELD', source_role='score_fit', test_access=False,
                    official_validation_access=False, historical_score_eval_used=False,
                    new_independent_evaluation=False, cohort=dict(pilot_fit_dialogues=75,pilot_eval_dialogues=60),
                    scoring=dict(arms={k:list(v) for k,v in METHODS.items()}, logistic_l2_C=1,
                        bootstrap_dialogue_replicates=2000,bootstrap_seed=shared.SEED,scorer_sha256=core.SCORER_SHA))
                pp=root/'EXPERIMENT_PROTOCOL.json';pp.write_text(json.dumps(protocol))
                verification=dict(status='COMPLETE',unexpected_failures=0,no_model_fit=True,no_test=True,
                    no_new_validation=True,old_score_eval_used=False,source_original_role='score_fit',
                    pilot_fit_dialogues=75,pilot_eval_dialogues=60,targets=135,planned_versions=270,
                    source_clean_rows_unchanged=True,new_independent_evaluation=False,
                    clean_utterances=len(raw),protocol_sha256=shared.sha(pp),
                    source_clean_path=str(paths['source_clean.csv']),source_clean_sha256=shared.sha(paths['source_clean.csv']),
                    source_bindings={str(paths['pilot_cohort.csv']):shared.sha(paths['pilot_cohort.csv']),
                        '/never-read-this-private-asset/source_data.pt':'engine-metadata-only'},
                    outputs={'score_inputs.csv':shared.sha(paths['score_inputs.csv'])})
                vp=root/'inference_verification.json';vp.write_text(json.dumps(verification))
                args=argparse.Namespace(input=paths['score_inputs.csv'],cohort=paths['pilot_cohort.csv'],
                    verification=vp,protocol=pp)
                self.assertEqual(len(verify_sources(args)[0]),24)
                verification['unexpected_failures']=1;vp.write_text(json.dumps(verification))
                with self.assertRaises(ValueError):verify_sources(args)
        def test_serialization_and_sparse_support(self):
            rows = self.parsed()
            for r in rows:
                if r['role']=='score_fit': r['D_pred']=r['D_abs']=r['cancellation']=None
            report, predictions, models=core.scoring(rows, _bootstrap_repeats=8)
            compare_extra(report,predictions,rows)
            info, signs, arrays, draws=coefficient_bootstrap([],repeats=8)
            report['fit_coefficient_bootstrap']=info
            json.dumps(report,allow_nan=False)
            narrative(report,distribution(rows),coefficient_points(models),signs)
            with tempfile.TemporaryDirectory() as tmp:
                core.write_csv(Path(tmp)/'comparison.csv',core.flattened(report))
                np.savez_compressed(Path(tmp)/'fit.npz',draw_indices=draws,**arrays)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful(): raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--input',type=Path,default=RUN/'results/inference/score_inputs.csv')
    parser.add_argument('--cohort',type=Path,default=RUN/'local_only/pilot_cohort.csv')
    parser.add_argument('--verification',type=Path,default=RUN/'results/inference/inference_verification.json')
    parser.add_argument('--protocol',type=Path,default=RUN/'EXPERIMENT_PROTOCOL.json')
    parser.add_argument('--output',type=Path,default=RUN/'results/scoring')
    args=parser.parse_args()
    if args.self_test: self_test()
    else: run(args)


if __name__=='__main__': main()
