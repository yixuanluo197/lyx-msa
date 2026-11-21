#!/usr/bin/env python3
"""Offline K=2 decomposition of frozen predictions; no text/model/media access.

Incomplete planned families remain present with unavailable formal metrics.
Correlations and response patterns are descriptive on previously exposed Val.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {'neutral_append': ('neutral_a', 'neutral_b'),
            'semantic_paraphrase': ('paraphrase_a', 'paraphrase_b')}
REQUIRED = ['sample_id', 'dialogue_id', 'perturbation_type', 'original_correct',
    'original_prediction', 'original_confidence', 'flip_count', 'flip_rate',
    'no_flip_count', 'confidence_increase_count', 'confidence_decrease_count',
    'mean_delta_C', 'mean_abs_delta_C', 'std_confidence', 'mean_delta_P',
    'mean_abs_delta_P', 'std_original_class_probability', 'mu', 'delta', 'JS',
    'D_pred', 'D_margin', 'D_C', 'V_C', 'max_abs_delta_C']
METRICS = ['mu', 'delta', 'JS', 'D_pred', 'D_margin', 'D_C', 'V_C',
           'mean_abs_delta_C', 'max_abs_delta_C', 'mean_abs_delta_P',
           'std_original_class_probability', 'mean_delta_other',
           'D_margin_fixed_clean_competitor', 'competitor_reselection_gain']
PATTERNS = ['BOTH_INCREASE', 'BOTH_DECREASE', 'MIXED_INCREASE_DECREASE',
            'INCREASE_AND_UNCHANGED', 'DECREASE_AND_UNCHANGED', 'BOTH_UNCHANGED']
SCOPES = ('ALL_COMPLETE_DIAGNOSTIC', 'AUTOMATIC_QC_ELIGIBLE', 'AI_REVIEWED_EXPLORATORY')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def yes(value):
    return value is True or str(value).lower() in ('1', 'true')


def write_csv(path, rows, first=()):
    rows = list(rows)
    if not rows:
        raise ValueError('Refuse an undocumented empty output: ' + str(path))
    fields = list(dict.fromkeys(list(first) + [k for row in rows for k in row]))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def close(value, expected):
    return value not in (None, '') and math.isclose(float(value), float(expected), abs_tol=1e-12, rel_tol=1e-10)


def probability(value):
    result = np.asarray(json.loads(value), dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all() or (result < 0).any() or not np.isclose(result.sum(), 1, atol=1e-5):
        raise ValueError('Invalid frozen probability vector')
    return result


def js_divergence(p, q):
    middle = (p + q) / 2
    left, right = p > 0, q > 0
    return float(.5 * np.sum(p[left] * np.log(p[left] / middle[left])) +
                 .5 * np.sum(q[right] * np.log(q[right] / middle[right])))


def response_pattern(values, threshold):
    values = list(values)
    if len(values) != 2 or not np.isfinite(values).all():
        raise ValueError('Response pattern requires exactly two finite values')
    up = sum(x > threshold for x in values)
    down = sum(x < -threshold for x in values)
    if up == 2: return 'BOTH_INCREASE'
    if down == 2: return 'BOTH_DECREASE'
    if up == down == 1: return 'MIXED_INCREASE_DECREASE'
    if up == 1: return 'INCREASE_AND_UNCHANGED'
    if down == 1: return 'DECREASE_AND_UNCHANGED'
    return 'BOTH_UNCHANGED'


def from_probabilities(p, q):
    """Require frozen clean original class, strongest-other competitor, and K=2."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (2, len(p)):
        raise ValueError('Formal metrics require exactly K=2 vectors')
    original = int(p.argmax()); other = np.arange(len(p)) != original
    competitor = int(np.flatnonzero(other)[p[other].argmax()])
    maxima = q.max(axis=1); original_probs = q[:, original]
    dc = maxima - p[original]; dp = original_probs - p[original]
    flip = q.argmax(axis=1) != original
    clean_margin = p[original] - p[other].max()
    other_max = q[:, other].max(axis=1)
    margin = original_probs - other_max
    fixed_margin = original_probs - q[:, competitor]
    js = np.asarray([js_divergence(p, row) for row in q])
    result = dict(original_prediction=original, original_confidence=float(p[original]),
        flip_count=int(flip.sum()), no_flip_count=int((~flip).sum()), flip_rate=float(flip.mean()),
        confidence_increase_count=int((dc > 0).sum()), confidence_decrease_count=int((dc < 0).sum()),
        confidence_unchanged_count=int((dc == 0).sum()), mean_delta_C=float(dc.mean()),
        mean_abs_delta_C=float(np.abs(dc).mean()), max_abs_delta_C=float(np.abs(dc).max()),
        std_confidence=float(maxima.std(ddof=1)), mean_delta_P=float(dp.mean()),
        mean_abs_delta_P=float(np.abs(dp).mean()), max_abs_delta_P=float(np.abs(dp).max()),
        std_original_class_probability=float(original_probs.std(ddof=1)),
        mu=float(np.mean(flip * maxima)), delta=float(abs(p[original] - maxima.mean())),
        JS=float(js.mean()), js_mean=float(js.mean()), js_max=float(js.max()),
        D_pred=float(p[original] - original_probs.mean()),
        D_margin=float(clean_margin - margin.mean()), D_C=float(maxima.mean() - p[original]),
        V_C=float(maxima.std(ddof=1)), clean_margin=float(clean_margin),
        mean_perturbed_margin=float(margin.mean()), mean_delta_other=float(other_max.mean() - p[other].max()),
        clean_competitor_class=competitor,
        competitor_switch_count=int((np.flatnonzero(other)[q[:, other].argmax(axis=1)] != competitor).sum()),
        mean_delta_fixed_clean_competitor=float(q[:, competitor].mean() - p[competitor]),
        D_margin_fixed_clean_competitor=float(clean_margin - fixed_margin.mean()),
        competitor_reselection_gain=float(other_max.mean() - q[:, competitor].mean()),
        mean_confidence_change=float(dc.mean()), mean_abs_confidence_change=float(np.abs(dc).mean()),
        max_abs_confidence_change=float(np.abs(dc).max()),
        mean_original_class_probability_change=float(dp.mean()))
    for i in range(2):
        result.update({f'delta_C_{i+1}': float(dc[i]), f'delta_P_{i+1}': float(dp[i]),
                       f'C_perturbed_{i+1}': float(maxima[i]), f'P_perturbed_{i+1}': float(original_probs[i]),
                       f'JS_{i+1}': float(js[i]), f'flip_{i+1}': int(flip[i]),
                       f'perturbed_margin_{i+1}': float(margin[i])})
    return result


def load_unified(args):
    parent = read_csv(args.parent / 'matched_analysis/target_behavior.csv')
    predictions = read_csv(args.parent / 'perturbed_predictions.csv')
    events = read_csv(args.events)
    parent_keys = [(r['sample_id'], r['perturbation_type']) for r in parent]
    if len(parent) != 812 or len(set(parent_keys)) != len(parent):
        raise ValueError('Expected 812 unique original target-by-family rows')
    indexed = lambda rows: {(r['sample_id'], r['perturbation_type'], r['variant_id']): r for r in rows}
    event_lookup, raw_lookup = indexed(events), indexed(predictions)
    if len(event_lookup) != len(events) or len(raw_lookup) != len(predictions) or set(event_lookup) != set(raw_lookup):
        raise ValueError('Duplicate, missing or extra frozen event keys')
    expected = {(sid, family, version) for sid, family in parent_keys for version in FAMILIES[family]}
    if set(event_lookup) != expected or len(events) != 1624:
        raise ValueError('All 1624 fixed versions, including failures, must be retained')
    comparison_count = 0; result = []; pattern_rows = []
    for row in parent:
        sid, family = row['sample_id'], row['perturbation_type']; versions = FAMILIES[family]
        ev = [event_lookup[sid, family, version] for version in versions]
        raw = [raw_lookup[sid, family, version] for version in versions]
        p = probability(raw[0]['original_probabilities'])
        if any(not np.array_equal(p, probability(r['original_probabilities'])) for r in raw):
            raise ValueError('Clean prediction differs across fixed versions')
        out = dict(row)
        out.update(original_prediction=int(p.argmax()), planned_k=2, recorded_k=len(ev),
            successful_k=sum(r['status'] == 'SUCCESS' for r in raw),
            metric_available_k=sum(yes(e['metric_available']) for e in ev),
            automatic_qc_eligible=int(yes(row['automatic_qc_eligible'])),
            ai_reviewed_exploratory_eligible=int(yes(row['ai_reviewed_exploratory_eligible'])),
            human_confirmed_eligible='', human_review_status='PENDING_TWO_REAL_HUMAN_REVIEWS',
            variant_id_1=versions[0], variant_id_2=versions[1],
            event_status_1=ev[0]['status'], event_status_2=ev[1]['status'],
            std_ddof=1, JS_logarithm='natural', material_threshold=0.02)
        for i, (e, r) in enumerate(zip(ev, raw), 1):
            if (str(e['dialogue_id']) != str(row['dialogue_id']) or e['status'] != r['status'] or
                yes(e['automatic_qc_pass']) != yes(r['automatic_qc_pass']) or
                yes(e['label_preserving_main_eligible']) != yes(r['label_preserving_main_eligible']) or
                int(e['original_correct']) != int(row['original_correct'])):
                raise ValueError('Event/source identity or eligibility differs')
            if e['status'] != 'SUCCESS':
                if yes(e['metric_available']): raise ValueError('Failed event cannot have metrics')
                continue
            if not yes(e['metric_available']): raise ValueError('Successful event lacks finite metrics')
            q = probability(r['perturbed_probabilities']); a = int(p.argmax()); other = np.arange(len(p)) != a
            direct = dict(C_clean=p[a], C_perturbed=q.max(), delta_C=q.max()-p[a],
                          P_clean=p[a], P_perturbed=q[a], delta_P=q[a]-p[a], JS=js_divergence(p,q),
                          clean_margin=p[a]-p[other].max(), perturbed_margin=q[a]-q[other].max(),
                          flip=int(a != q.argmax()), original_prediction=a, original_confidence=p[a])
            for key, value in direct.items():
                if not close(e[key], value): raise ValueError('Independent event mismatch: '+key)
                comparison_count += 1
        complete = all(r['status'] == 'SUCCESS' for r in raw) and all(yes(e['metric_available']) for e in ev)
        if complete != yes(row['complete_planned_family']):
            raise ValueError('Parent K=2 completeness differs')
        automatic = complete and all(yes(e['automatic_qc_pass']) for e in ev)
        ai = automatic and all(yes(e['label_preserving_main_eligible']) for e in ev)
        if automatic != yes(row['automatic_qc_eligible']) or ai != yes(row['ai_reviewed_exploratory_eligible']):
            raise ValueError('Parent automatic/AI eligibility differs')
        out['complete_planned_family'] = int(complete)
        out['formal_metrics_available'] = int(complete)
        if complete:
            values = from_probabilities(p, np.asarray([probability(r['perturbed_probabilities']) for r in raw]))
            for key in ('mu', 'delta', 'js_mean', 'flip_rate', 'mean_confidence_change',
                        'mean_abs_confidence_change', 'max_abs_confidence_change',
                        'mean_original_class_probability_change'):
                if not close(row[key], values[key]): raise ValueError('Parent formal metric differs: '+key)
                comparison_count += 1
            out.update(values)
            out['flip_stratum'] = 'ZERO_FLIP' if values['flip_count'] == 0 else 'AT_LEAST_ONE_FLIP'
            for signal in ('C', 'P'):
                deltas = [values[f'delta_{signal}_1'], values[f'delta_{signal}_2']]
                out[f'{signal}_pattern_strict'] = response_pattern(deltas, 0.)
                out[f'{signal}_pattern_material'] = response_pattern(deltas, .02)
        else:
            out['flip_stratum'] = 'INCOMPLETE_K2'
            for key in REQUIRED + METRICS:
                if key not in ('sample_id','dialogue_id','perturbation_type','original_correct','original_prediction','original_confidence'):
                    out[key] = ''
            for signal in ('C', 'P'):
                out[f'{signal}_pattern_strict'] = out[f'{signal}_pattern_material'] = 'INCOMPLETE_K2'
        result.append(out)
        keys = ('sample_id','pair_id','dialogue_id','utterance_index','perturbation_type','original_correct',
                'original_prediction','original_confidence','complete_planned_family','formal_metrics_available',
                'automatic_qc_eligible','ai_reviewed_exploratory_eligible','human_confirmed_eligible','human_review_status',
                'label_review_statuses','successful_k','metric_available_k','flip_count','flip_rate','flip_stratum',
                'C_pattern_strict','C_pattern_material','P_pattern_strict','P_pattern_material',
                'delta_C_1','delta_C_2','delta_P_1','delta_P_2')
        pattern_rows.append({key:out.get(key,'') for key in keys})
    if args.experiment1.exists():
        exp = {(r['sample_id'],r['perturbation_type']):r for r in read_csv(args.experiment1)}
        if set(exp) != set(parent_keys): raise ValueError('Experiment1 target keys differ')
        for row in result:
            if not row['formal_metrics_available']: continue
            reference = exp[row['sample_id'],row['perturbation_type']]
            for key in ('mu','delta','js_mean','flip_rate','D_pred','D_margin','D_C','V_C','clean_margin'):
                if not close(reference[key],row[key]):raise ValueError('Experiment1 metric differs: '+key)
                comparison_count += 1
    else:
        raise FileNotFoundError('Frozen Experiment1 comparison is required')
    return result, pattern_rows, comparison_count


def eligible(row, scope):
    return bool(row['formal_metrics_available']) and (scope == 'ALL_COMPLETE_DIAGNOSTIC' or
        scope == 'AUTOMATIC_QC_ELIGIBLE' and yes(row['automatic_qc_eligible']) or
        scope == 'AI_REVIEWED_EXPLORATORY' and yes(row['ai_reviewed_exploratory_eligible']))


def distribution(rows):
    output = []
    for family, scope, group, signal, scheme in itertools.product(FAMILIES, SCOPES, ('ALL','CORRECT','ERROR'), ('C','P'), ('strict','material')):
        planned = [r for r in rows if r['perturbation_type'] == family and
                   (group == 'ALL' or int(r['original_correct']) == int(group == 'CORRECT'))]
        selected = [r for r in planned if eligible(r,scope)]
        counts = Counter(r[f'{signal}_pattern_{scheme}'] for r in selected)
        for pattern in PATTERNS:
            members = [r for r in selected if r[f'{signal}_pattern_{scheme}'] == pattern]
            output.append(dict(perturbation_type=family,analysis_scope=scope,original_correct_group=group,
                signal=signal,pattern_scheme=scheme,material_threshold=0 if scheme=='strict' else .02,
                stable_interval='exact zero' if scheme=='strict' else '[-0.02,+0.02] inclusive',
                response_pattern=pattern,planned_target_families=len(planned),eligible_complete_target_families=len(selected),
                pattern_count=counts[pattern],pattern_fraction=counts[pattern]/len(selected) if selected else '',
                dialogues=len({r['dialogue_id'] for r in members}),
                human_confirmed_eligible='',population_claim=False))
    return output


def relationships(rows):
    output = []
    for family, scope, group, stratum in itertools.product(FAMILIES, SCOPES, ('ALL','CORRECT','ERROR'), ('ALL','ZERO_FLIP','AT_LEAST_ONE_FLIP')):
        selected = [r for r in rows if r['perturbation_type']==family and eligible(r,scope) and
                    (group=='ALL' or int(r['original_correct'])==int(group=='CORRECT')) and
                    (stratum=='ALL' or r['flip_stratum']==stratum)]
        base=dict(perturbation_type=family,analysis_scope=scope,original_correct_group=group,flip_stratum=stratum,
            target_families=len(selected),dialogues=len({r['dialogue_id'] for r in selected}),
            interpretation='Descriptive; clustered targets and exposed Validation; no p-values or confirmation claim')
        for left,right in itertools.combinations(METRICS,2):
            a=np.asarray([r[left] for r in selected],float);b=np.asarray([r[right] for r in selected],float)
            valid=len(a)>=3 and len(np.unique(a))>1 and len(np.unique(b))>1
            output.append(dict(base,row_type='CORRELATION',metric_x=left,metric_y=right,
                pearson=float(np.corrcoef(a,b)[0,1]) if valid else '',
                spearman=float(np.corrcoef(rankdata(a,method='average'),rankdata(b,method='average'))[0,1]) if valid else '',
                status='DESCRIPTIVE' if valid else 'INSUFFICIENT_OR_CONSTANT',n_pairs=len(a)))
        formulas={
            'delta=abs(D_C)':lambda r:r['delta']-abs(r['D_C']),
            'D_margin=D_pred+mean_delta_other':lambda r:r['D_margin']-r['D_pred']-r['mean_delta_other'],
            'D_margin=D_margin_fixed_clean_competitor+competitor_reselection_gain':lambda r:r['D_margin']-r['D_margin_fixed_clean_competitor']-r['competitor_reselection_gain'],
            'V_C=std_confidence_ddof1':lambda r:r['V_C']-r['std_confidence'],
            'K2_V_C=abs(delta_C_1-delta_C_2)/sqrt2':lambda r:r['V_C']-abs(r['delta_C_1']-r['delta_C_2'])/math.sqrt(2),
            'mean_abs_delta_C>=delta':lambda r:min(0,r['mean_abs_delta_C']-r['delta']),
            'mean_abs_delta_P>=abs(D_pred)':lambda r:min(0,r['mean_abs_delta_P']-abs(r['D_pred']))}
        if stratum=='ZERO_FLIP':formulas['no_flip_D_pred=-D_C']=lambda r:r['D_pred']+r['D_C']
        for name,fn in formulas.items():
            residual=np.asarray([fn(r) for r in selected],float)
            output.append(dict(base,row_type='FORMULA_CHECK',formula=name,n_pairs=len(residual),
                max_abs_residual=float(np.abs(residual).max()) if len(residual) else '',
                mean_abs_residual=float(np.abs(residual).mean()) if len(residual) else '',
                tolerance=1e-12,status='PASS' if len(residual) and np.max(np.abs(residual))<=1e-12 else 'NA' if not len(residual) else 'FAIL'))
        for name,fn in {'D_pred_minus_delta':lambda r:r['D_pred']-r['delta'],
                        'abs_D_pred_minus_delta':lambda r:abs(r['D_pred'])-r['delta'],
                        'mean_abs_delta_P_minus_std_original_class_probability':lambda r:r['mean_abs_delta_P']-r['std_original_class_probability'],
                        'competitor_reselection_gain':lambda r:r['competitor_reselection_gain']}.items():
            values=np.asarray([fn(r) for r in selected],float)
            output.append(dict(base,row_type='DECOMPOSITION_SUMMARY',comparison=name,n_pairs=len(values),
                mean=float(values.mean()) if len(values) else '',median=float(np.median(values)) if len(values) else '',
                minimum=float(values.min()) if len(values) else '',maximum=float(values.max()) if len(values) else '',
                positive_count=int((values>0).sum()),negative_count=int((values<0).sum()),zero_count=int((values==0).sum()),
                status='DESCRIPTIVE' if len(values) else 'NA'))
    return output


def self_test():
    p=np.asarray([.8,.2]);q=np.asarray([[.9,.1],[.7,.3]])
    values=from_probabilities(p,q)
    assert values['mu']==0 and abs(values['D_pred']+values['D_C'])<1e-12
    assert values['delta']==abs(values['D_C']) and math.isclose(values['V_C'],math.sqrt(.02))
    for values,name in [([.1,.2],'BOTH_INCREASE'),([-.1,-.2],'BOTH_DECREASE'),([-.1,.2],'MIXED_INCREASE_DECREASE'),([0,.1],'INCREASE_AND_UNCHANGED'),([0,-.1],'DECREASE_AND_UNCHANGED'),([0,0],'BOTH_UNCHANGED')]:
        assert response_pattern(values,0)==name
    assert response_pattern([-.02,.02],.02)=='BOTH_UNCHANGED'
    try:from_probabilities(p,q[:1])
    except ValueError:pass
    else:raise AssertionError('K1 must never produce formal metrics')
    flipped=from_probabilities(np.asarray([.6,.3,.1]),np.asarray([[.2,.7,.1],[.2,.1,.7]]))
    assert flipped['flip_count']==2 and flipped['competitor_switch_count']==1
    assert abs(flipped['D_margin']-flipped['D_pred']-flipped['mean_delta_other'])<1e-12
    assert abs(flipped['D_margin']-flipped['D_margin_fixed_clean_competitor']-flipped['competitor_reselection_gain'])<1e-12
    print(json.dumps(dict(status='SYNTHETIC_PASS',real_data_opened=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--parent',type=Path,default=ROOT.parent/'20261120T170835Z/results')
    parser.add_argument('--events',type=Path)
    parser.add_argument('--experiment1',type=Path,default=ROOT.parent/'20261121T115624Z_metric_diagnosis/results/ai_exploratory/targets.csv')
    parser.add_argument('--stage',choices=['unified','all'],default='all')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    args.events=args.events or args.root/'EVENT_RESPONSES.csv'
    rows,patterns,checks=load_unified(args)
    write_csv(args.root/'UNIFIED_BEHAVIOR_FEATURES.csv',rows,REQUIRED)
    log=dict(status='UNIFIED_COMPLETE',target_family_rows=len(rows),unique_targets=len({r['sample_id'] for r in rows}),
        complete_families=sum(r['formal_metrics_available'] for r in rows),
        incomplete_families=sum(not r['formal_metrics_available'] for r in rows),
        automatic_eligible_families=sum(r['automatic_qc_eligible'] for r in rows),
        ai_eligible_families=sum(r['ai_reviewed_exploratory_eligible'] for r in rows),
        human_confirmed_eligible=None,human_status='PENDING_REAL_TWO_HUMAN_REVIEWS',
        independent_source_metric_comparisons=checks,formal_k=2,std_ddof=1,natural_log_js=True,
        no_fitting=True,no_model_forward=True,no_original_text_read=True,no_new_inference=True,
        no_test=True,posthoc_exposed_validation=True,material_stable_interval='[-0.02,+0.02] inclusive',
        input_sha256={str(path):digest(path) for path in (args.events,args.parent/'perturbed_predictions.csv',args.parent/'matched_analysis/target_behavior.csv',args.experiment1)},
        script_sha256=digest(__file__),outputs={'UNIFIED_BEHAVIOR_FEATURES.csv':digest(args.root/'UNIFIED_BEHAVIOR_FEATURES.csv')})
    if args.stage=='all':
        write_csv(args.root/'SAMPLE_RESPONSE_PATTERN.csv',patterns)
        write_csv(args.root/'PATTERN_DISTRIBUTION.csv',distribution(rows))
        relation=relationships(rows)
        failures=[r for r in relation if r['row_type']=='FORMULA_CHECK' and r['status']=='FAIL']
        if failures:raise ValueError('A decomposition identity failed: '+str(failures[:1]))
        write_csv(args.root/'METRIC_RELATIONSHIP_ANALYSIS.csv',relation)
        log.update(status='COMPLETE',formula_checks=sum(r['row_type']=='FORMULA_CHECK' and r['status']=='PASS' for r in relation),
            formula_failures=len(failures),correlation_rows=sum(r['row_type']=='CORRELATION' for r in relation))
        for name in ('SAMPLE_RESPONSE_PATTERN.csv','PATTERN_DISTRIBUTION.csv','METRIC_RELATIONSHIP_ANALYSIS.csv'):
            log['outputs'][name]=digest(args.root/name)
    write_json(args.root/'logs/targets.json',log)
    print(json.dumps({key:log[key] for key in ('status','target_family_rows','unique_targets','complete_families','incomplete_families','independent_source_metric_comparisons')}))


if __name__=='__main__':main()
