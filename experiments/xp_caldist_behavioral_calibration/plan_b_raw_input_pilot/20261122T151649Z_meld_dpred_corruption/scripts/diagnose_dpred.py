#!/usr/bin/env python3
"""Offline, Train-score-cohort-only diagnosis of the saved signed K2 response.

No encoder, downstream model, tensor checkpoint, raw dialogue, Validation or Test
is loaded. Logistic refits use only the old score_fit cohort and fixed settings.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROLES = ('score_fit', 'score_eval')
VARIANTS = ('paraphrase_a', 'paraphrase_b')
GROUPS = ('all', 'correct', 'wrong')
METRICS = ('delta_a', 'delta_b', 'abs_delta_a', 'abs_delta_b', 'D_pred', 'D_abs',
           'cancellation_abs', 'cancellation_relative', 'opposite_sign')
CORR_RESPONSES = ('D_pred', 'D_abs', 'cancellation_relative')
CLEAN = ('original_confidence', 'clean_margin', 'clean_entropy')
FEATURES = {'weak': ('logit_confidence',),
            'strong': ('logit_confidence', 'clean_margin', 'clean_entropy')}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csvrows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def csvwrite(path, rows):
    if not rows:
        raise ValueError('Do not silently omit an empty result: '+str(path))
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def assert_close(a, b, message, atol=1e-10):
    if not np.allclose(a, b, atol=atol, rtol=1e-10):
        raise ValueError(message)


def probability(text):
    p = np.asarray(json.loads(text), dtype=np.float64)
    if p.shape != (7,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or abs(p.sum()-1) > 1e-6:
        raise ValueError('Invalid saved full seven-class Softmax')
    return p


def number(value):
    return float(value) if value not in ('', None) else float('nan')


def printable(value):
    return float(value) if np.isfinite(value) else ''


def interval(values):
    a = np.asarray(values, dtype=float)
    valid = a[np.isfinite(a)]
    return {'ci_low': float(np.quantile(valid, .025)) if len(valid) else None,
            'ci_high': float(np.quantile(valid, .975)) if len(valid) else None,
            'bootstrap_valid': int(len(valid)), 'bootstrap_na': int(len(a)-len(valid))}


def subset(row, group):
    return group == 'all' or row['original_correct'] == int(group == 'correct')


def pearson(a, b):
    if len(a) < 3:
        return float('nan')
    x, y = np.asarray(a, float), np.asarray(b, float)
    x, y = x-x.mean(), y-y.mean()
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/denominator) if denominator > 0 else float('nan')


def correlations(x, y):
    if len(x) < 3:
        return float('nan'), float('nan')
    return pearson(x, y), pearson(rankdata(x, method='average'), rankdata(y, method='average'))


def verify_sources(parent, protocol_path):
    source = parent/'results'
    verification = read(source/'score_prediction_verification.json')
    report = read(source/'scoring/scoring_report.json')
    protocol = read(protocol_path)
    if verification['status'] != 'COMPLETE' or not all(verification[k] for k in
            ('no_test', 'no_model_fit', 'fp32', 'target_only_replacement', 'full_dialogue_context', 'text_modality_only')):
        raise ValueError('Source inference guard is incomplete')
    if verification['model_state_before'] != verification['model_state_after']:
        raise ValueError('Source model changed during original inference')
    anchors = {}
    for filename in ('score_inputs.csv', 'score_transformations.csv'):
        path = source/filename
        if sha(path) != verification['outputs'][filename]:
            raise ValueError('Saved source probability CSV hash mismatch')
        anchors[str(path.resolve())] = sha(path)
    for filename, expected in report['input_sha256'].items():
        path = Path(filename)
        if path.suffix not in ('.json', '.csv') or not path.is_relative_to(parent):
            raise ValueError('Scoring provenance points outside the permitted parent metadata')
        if sha(path) != expected:
            raise ValueError('Original CPU scorer input binding changed: '+str(path))
        anchors[str(path.resolve())] = expected
    scorer = parent.parent/'20261121T175233Z_reliability_readiness/scripts/score_heldout.py'
    if sha(scorer) != report['script_sha256'] or sha(scorer) != protocol['scoring']['scorer_sha256']:
        raise ValueError('Original fixed scoring source differs')
    for path in (source/'score_prediction_verification.json', source/'scoring/scoring_report.json',
                 source/'scoring/fitted_parameters.json', protocol_path, scorer):
        anchors[str(path.resolve())] = sha(path)
    if protocol['scoring']['bootstrap'] != {'replicates':2000, 'seed':20261121,
                                           'unit':'Paired Dialogue cluster', 'interval':'95% percentile'}:
        raise ValueError('Use exactly the frozen bootstrap protocol')
    if report['versions'] != {'numpy':np.__version__, 'sklearn':sklearn.__version__}:
        raise ValueError('Use the original NumPy/sklearn CPU environment for exact coefficient replay')
    return anchors, verification, report, protocol


def construct_rows(parent, verification):
    clean = csvrows(parent/'results/score_inputs.csv')
    transformed = csvrows(parent/'results/score_transformations.csv')
    if len(clean) != 2976 or len(transformed) != 5952:
        raise ValueError('Unexpected full frozen score cohort size')
    byid = {row['sample_id']:row for row in clean}
    variants = {(row['sample_id'],row['variant_id']):row for row in transformed}
    if len(byid) != len(clean) or len(variants) != len(transformed) or set(variants) != {(sid,v) for sid in byid for v in VARIANTS}:
        raise ValueError('Duplicate or missing source identities')
    per_version, samples = [], []
    for original in clean:
        sid = original['sample_id']; role = original['role']
        if role not in ROLES:
            raise ValueError('No base, Validation or Test role is allowed')
        p = probability(original['original_probabilities'])
        c = int(p.argmax()); y = int(original['original_correct'])
        if c != int(original['original_prediction']) or y not in (0,1) or y != int(c == int(original['original_label'])):
            raise ValueError('Original prediction/correctness does not match saved vector/label')
        ordered = np.sort(p); entropy = float(-np.sum(p[p>0]*np.log(p[p>0])))
        assert_close([p[c],ordered[-1]-ordered[-2],entropy],
                     [float(original[k]) for k in CLEAN], 'Saved clean scalars differ from probabilities')
        row = {key:original[key] for key in ('sample_id','dialogue_id','utterance_id','utterance_index','role')}
        row.update(original_correct=y, **{key:float(original[key]) for key in CLEAN})
        clipped = np.clip(row['original_confidence'],1e-6,1-1e-6)
        row['logit_confidence'] = float(np.log(clipped/(1-clipped)))
        row.update({metric:float('nan') for metric in METRICS})
        eligible, deltas = 0, []
        reasons = []
        for version, metric in zip(VARIANTS, ('delta_a','delta_b')):
            v = variants[sid,version]
            if any(v[k] != original[k] for k in ('role','dialogue_id','utterance_id','utterance_index')):
                raise ValueError('Version/target identity mismatch')
            assert_close(probability(v['original_probabilities']),p,'Version original probability changed')
            valid = v['eligible'] == '1'
            delta = float('nan')
            if valid:
                if v['review_status'] != 'PASS' or v['inference_status'] != 'PREDICTED':
                    raise ValueError('An unqualified version was marked eligible')
                q = probability(v['perturbed_probabilities'])
                assert_close(q[c],float(v['perturbed_original_class_probability']),'Original-class probability mismatch')
                if int(q.argmax()) != int(v['perturbed_prediction']):
                    raise ValueError('Perturbed class does not match full probability')
                delta = float(p[c]-q[c]); eligible += 1; deltas.append(delta)
            elif v['perturbed_probabilities']:
                raise ValueError('Ineligible version has an unexpected probability')
            row[metric] = delta; row['abs_'+metric] = abs(delta)
            reasons.append(v['reason'])
            per_version.append({key:row[key] for key in ('sample_id','dialogue_id','utterance_id','utterance_index','role','original_correct')} |
                {'variant_id':version,'eligible':int(valid),'review_status':v['review_status'],
                 'encoding_status':v['encoding_status'],'inference_status':v['inference_status'],
                 'reason':v['reason'],'error':v['error'],'p_clean_original_class':float(p[c]),
                 'p_perturbed_original_class':float(p[c]-delta) if valid else '',
                 'signed_delta':printable(delta),'absolute_delta':printable(abs(delta))})
        if eligible == 2:
            row['D_pred'] = float(np.mean(deltas)); row['D_abs'] = float(np.mean(np.abs(deltas)))
            row['cancellation_abs'] = row['D_abs']-abs(row['D_pred'])
            if row['cancellation_abs'] < -1e-14:
                raise ValueError('Triangle inequality violated')
            row['cancellation_abs'] = max(0.0,row['cancellation_abs'])
            row['cancellation_relative'] = row['cancellation_abs']/row['D_abs'] if row['D_abs']>0 else float('nan')
            row['opposite_sign'] = float(deltas[0]*deltas[1]<0)
            if original['k2_available'] != '1' or not original['D_pred']:
                raise ValueError('Double-eligible source pair lost D_pred')
            assert_close(row['D_pred'],float(original['D_pred']),'D_pred cannot be reproduced from full vectors')
            if row['opposite_sign'] == 0 and row['cancellation_abs']>1e-14:
                raise ValueError('Cancellation without opposite signs')
        elif original['D_pred'] or original['k2_available'] != '0':
            raise ValueError('Missing K2 was silently imputed')
        row.update(available_versions=eligible,k2_available=int(eligible==2),missing_reason=original['D_pred_missing_reason'],
                   version_a_reason=reasons[0],version_b_reason=reasons[1],
                   relative_cancellation_na_reason='ZERO_D_ABS' if eligible==2 and row['D_abs']==0 else ('NO_K2' if eligible!=2 else ''))
        samples.append(row)
    for role,n in [('score_fit',1569),('score_eval',1407)]:
        role_rows = [row for row in samples if row['role']==role]
        if len(role_rows)!=n or len({row['dialogue_id'] for row in role_rows})!=155:
            raise ValueError('Unexpected frozen role size')
        covered = [row for row in role_rows if row['k2_available']]
        if len({row['dialogue_id'] for row in covered}) != len(covered):
            raise ValueError('Expected at most one originally targeted utterance per Dialogue')
    if sum(row['k2_available'] for row in samples) != verification['available_D_pred']:
        raise ValueError('Source coverage count changed')
    return samples, per_version


def fit(rows, fields, settings, clean_reference=None):
    x = np.asarray([[row[key] for key in fields] for row in rows],float)
    y = np.asarray([row['original_correct'] for row in rows],int)
    if len(np.unique(y)) != 2:
        raise ValueError('ONE_OUTCOME_ONLY')
    scaler = StandardScaler().fit(x)
    if clean_reference is not None:
        for attribute in ('mean_','var_','scale_'):
            current = getattr(scaler,attribute); reference = getattr(clean_reference,attribute)
            assert_close(current[:-1],reference,'Matched clean scaler differs',atol=1e-12)
            current[:-1] = reference
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model = LogisticRegression(**settings).fit(scaler.transform(x),y)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError('NONFINITE_COEFFICIENT')
    return scaler,model


def reproduce_models(samples, parameters, settings):
    fit_all = [row for row in samples if row['role']=='score_fit']
    fit_covered = [row for row in fit_all if row['k2_available']]
    checks, point_models = [], {}
    for strength,fields in FEATURES.items():
        matched_scaler = None
        for suffix,rows,augmented in [('baseline_all_fit',fit_all,False),
                                      ('baseline_covered_fit',fit_covered,False),
                                      ('augmented_covered_fit',fit_covered,True)]:
            name = strength+'_'+suffix; saved = parameters[name]
            used = fields+('D_pred',) if augmented else fields
            scaler,model = fit(rows,used,settings,matched_scaler if augmented else None)
            if list(used)!=saved['features'] or [row['sample_id'] for row in rows]!=saved['fit_sample_ids']:
                raise ValueError('Original model order/coverage mismatch')
            assert_close(model.coef_,saved['coefficient'],'Original coefficient could not be reproduced',atol=1e-9)
            assert_close(model.intercept_,saved['intercept'],'Original intercept differs',atol=1e-9)
            assert_close(scaler.mean_,saved['scaler_mean'],'Original scaler means differ',atol=1e-12)
            assert_close(scaler.scale_,saved['scaler_scale'],'Original scaler scales differ',atol=1e-12)
            if model.classes_.tolist()!=[0,1] or saved['fixed_logistic_settings']!=settings:
                raise ValueError('Original target direction or C/settings changed')
            checks.append({'model':name,'fit_rows':len(rows),'max_coefficient_absolute_error':float(np.abs(model.coef_-saved['coefficient']).max()),
                           'max_intercept_absolute_error':float(np.abs(model.intercept_-saved['intercept']).max()),
                           'max_scaler_absolute_error':float(np.abs(scaler.scale_-saved['scaler_scale']).max()),'status':'PASS'})
            if suffix=='baseline_covered_fit':matched_scaler=scaler
            if augmented:point_models[strength]=(scaler,model)
    return checks,point_models


def bootstrap_analysis(samples, settings, repeats, seed, out):
    point_stats, corr_stats, contrasts = [], [], []
    stat_draws, corr_draws, coefficient_draws = [], [], []
    coefficients = {'weak':[],'strong':[]}
    for role in ROLES:
        rows = [row for row in samples if row['role']==role]
        dialogues = sorted({row['dialogue_id'] for row in rows})
        by_dialogue = {did:[i for i,row in enumerate(rows) if row['dialogue_id']==did] for did in dialogues}
        rng = np.random.default_rng(seed)
        indices = [np.asarray(by_dialogue[did],int) for did in dialogues]
        selections = [np.concatenate([indices[i] for i in rng.integers(0,len(dialogues),len(dialogues))]) for _ in range(repeats)]
        metric_boot = {(group,metric):{'mean':[],'median':[]} for group in GROUPS for metric in METRICS}
        correlation_boot = {(group,response,feature,method):[] for group in GROUPS for response in CORR_RESPONSES
                            for feature in CLEAN for method in ('pearson','spearman')}
        for draw,selection in enumerate(selections):
            sampled = [rows[i] for i in selection]
            for group in GROUPS:
                group_rows = [row for row in sampled if subset(row,group)]
                for metric in METRICS:
                    values = np.asarray([row[metric] for row in group_rows],float)
                    values = values[np.isfinite(values)]
                    mean = float(values.mean()) if len(values) else float('nan')
                    median = float(np.median(values)) if len(values) else float('nan')
                    metric_boot[group,metric]['mean'].append(mean);metric_boot[group,metric]['median'].append(median)
                    stat_draws.append({'role':role,'draw':draw,'group':group,'metric':metric,'n_weighted':len(values),
                                      'mean':printable(mean),'median':printable(median),'status':'OK' if len(values) else 'NO_SUPPORT'})
                for response in CORR_RESPONSES:
                    supported = [row for row in group_rows if np.isfinite(row[response])]
                    for feature in CLEAN:
                        p,s = correlations([row[response] for row in supported],[row[feature] for row in supported])
                        for method,value in [('pearson',p),('spearman',s)]:
                            correlation_boot[group,response,feature,method].append(value)
                            corr_draws.append({'role':role,'draw':draw,'group':group,'response':response,'clean_feature':feature,
                                               'method':method,'n_weighted':len(supported),'correlation':printable(value),
                                               'status':'OK' if np.isfinite(value) else 'NO_SUPPORT_OR_CONSTANT'})
            if role=='score_fit':
                paired = [row for row in sampled if row['k2_available']]
                for strength,fields in FEATURES.items():
                    entry={'draw':draw,'model':strength,'fit_role':'score_fit','n_rows_weighted':len(paired),
                           'n_unique_dialogues':len({row['dialogue_id'] for row in paired}),
                           'n_correct_weighted':sum(row['original_correct'] for row in paired),
                           'status':'OK','reason':'','beta_D_pred_standardized':'','beta_D_pred_raw':'','D_pred_scale':'',
                           'intercept_standardized':'','intercept_raw':'','all_standardized_coefficients':'','all_scaler_means':'','all_scaler_scales':''}
                    try:
                        if not paired:raise ValueError('NO_COVERED_FIT_SUPPORT')
                        clean_x=np.asarray([[row[key] for key in fields] for row in paired],float)
                        reference=StandardScaler().fit(clean_x)
                        scaler,model=fit(paired,fields+('D_pred',),settings,reference)
                        if scaler.var_[-1]==0:raise ValueError('CONSTANT_D_PRED_NOT_IDENTIFIED')
                        beta=float(model.coef_[0,-1]);raw=beta/float(scaler.scale_[-1])
                        entry.update(beta_D_pred_standardized=beta,beta_D_pred_raw=raw,D_pred_scale=float(scaler.scale_[-1]),
                            intercept_standardized=float(model.intercept_[0]),
                            intercept_raw=float(model.intercept_[0]-model.coef_[0]@(scaler.mean_/scaler.scale_)),
                            all_standardized_coefficients=json.dumps(model.coef_[0].tolist()),
                            all_scaler_means=json.dumps(scaler.mean_.tolist()),all_scaler_scales=json.dumps(scaler.scale_.tolist()))
                    except Exception as error:
                        entry.update(status='NA',reason=type(error).__name__+':'+str(error))
                    coefficient_draws.append(entry);coefficients[strength].append(entry)
            if draw in (0,499,999,1499,1999):
                print(json.dumps({'role':role,'draw_completed':draw+1,'total_draws':repeats}),flush=True)
        for group in GROUPS:
            group_rows=[row for row in rows if subset(row,group)]
            for metric in METRICS:
                values=np.asarray([row[metric] for row in group_rows],float);values=values[np.isfinite(values)]
                if not len(values):
                    summary={'n':0,'n_missing':len(group_rows),'mean':None,'median':None,'std_ddof1':None,
                             'min':None,'p05':None,'p25':None,'p75':None,'p95':None,'max':None}
                else:
                    quant=np.quantile(values,[.05,.25,.5,.75,.95])
                    summary={'n':len(values),'n_missing':len(group_rows)-len(values),'mean':float(values.mean()),'median':float(quant[2]),
                             'std_ddof1':float(values.std(ddof=1)) if len(values)>1 else None,'min':float(values.min()),
                             'p05':float(quant[0]),'p25':float(quant[1]),'p75':float(quant[3]),'p95':float(quant[4]),'max':float(values.max())}
                mean_ci=interval(metric_boot[group,metric]['mean']);median_ci=interval(metric_boot[group,metric]['median'])
                point_stats.append({'role':role,'group':group,'metric':metric,**summary,
                    **{'mean_'+key:value for key,value in mean_ci.items()},**{'median_'+key:value for key,value in median_ci.items()}})
            for response in CORR_RESPONSES:
                supported=[row for row in group_rows if np.isfinite(row[response])]
                for feature in CLEAN:
                    p,s=correlations([row[response] for row in supported],[row[feature] for row in supported])
                    for method,value in [('pearson',p),('spearman',s)]:
                        corr_stats.append({'role':role,'group':group,'response':response,'clean_feature':feature,'method':method,
                                           'n':len(supported),'correlation':printable(value),
                                           **interval(correlation_boot[group,response,feature,method])})
        for metric in METRICS:
            correct=np.asarray(metric_boot['correct',metric]['mean']);wrong=np.asarray(metric_boot['wrong',metric]['mean'])
            c=[row[metric] for row in rows if row['original_correct']==1 and np.isfinite(row[metric])]
            w=[row[metric] for row in rows if row['original_correct']==0 and np.isfinite(row[metric])]
            contrasts.append({'role':role,'metric':metric,'contrast':'wrong_minus_correct_mean','n_correct':len(c),'n_wrong':len(w),
                              'difference':float(np.mean(w)-np.mean(c)) if c and w else None,**interval(wrong-correct)})
    csvwrite(out/'distribution_summary.csv',point_stats)
    csvwrite(out/'correct_wrong_differences.csv',contrasts)
    csvwrite(out/'correlations.csv',corr_stats)
    csvwrite(out/'bootstrap_distribution_draws.csv',stat_draws)
    csvwrite(out/'bootstrap_correlation_draws.csv',corr_draws)
    csvwrite(out/'bootstrap_coefficient_draws.csv',coefficient_draws)
    return point_stats,corr_stats,contrasts,coefficients


def report(run, samples, per_version, stats, corr, contrasts, coefficients, models, reproduction, inputs, protocol):
    out=run/'results/dpred'
    beta_summary=[]
    for name,(scaler,model) in models.items():
        draws=coefficients[name];valid=[row for row in draws if row['status']=='OK']
        betas=[float(row['beta_D_pred_standardized']) for row in valid]
        raw=[float(row['beta_D_pred_raw']) for row in valid]
        summary={'model':name,'original_beta_D_pred_standardized':float(model.coef_[0,-1]),
                 'original_beta_D_pred_raw':float(model.coef_[0,-1]/scaler.scale_[-1]),
                 'original_D_pred_mean':float(scaler.mean_[-1]),'original_D_pred_sd_ddof0':float(scaler.scale_[-1]),
                 'replicates':len(draws),'valid':len(valid),'failed_or_na':len(draws)-len(valid),
                 'beta_standardized_mean':float(np.mean(betas)) if betas else None,
                 'beta_standardized_median':float(np.median(betas)) if betas else None,
                 'beta_standardized_ci_low':float(np.quantile(betas,.025)) if betas else None,
                 'beta_standardized_ci_high':float(np.quantile(betas,.975)) if betas else None,
                 'beta_raw_ci_low':float(np.quantile(raw,.025)) if raw else None,
                 'beta_raw_ci_high':float(np.quantile(raw,.975)) if raw else None,
                 'fraction_positive':float(np.mean(np.asarray(betas)>0)) if betas else None,
                 'fraction_negative':float(np.mean(np.asarray(betas)<0)) if betas else None,
                 'failure_reasons':json.dumps(dict(Counter(row['reason'] for row in draws if row['status']!='OK')),ensure_ascii=False)}
        beta_summary.append(summary)
    csvwrite(out/'coefficient_stability.csv',beta_summary)
    coverage={}
    for role in ROLES:
        rows=[row for row in samples if row['role']==role];covered=[row for row in rows if row['k2_available']]
        coverage[role]={'all_rows':len(rows),'all_dialogues':len({row['dialogue_id'] for row in rows}),
            'K2_rows':len(covered),'K2_correct':sum(row['original_correct'] for row in covered),
            'K2_wrong':sum(not row['original_correct'] for row in covered),
            'one_version_only_rows':sum(row['available_versions']==1 for row in rows),
            'zero_version_rows':sum(row['available_versions']==0 for row in rows),
            'opposite_sign_pairs':sum(row['opposite_sign']==1 for row in covered),
            'both_zero_pairs':sum(row['D_abs']==0 for row in covered),
            'relative_cancellation_ge_half':sum(np.isfinite(row['cancellation_relative']) and row['cancellation_relative']>=.5 for row in covered)}
    fmt=lambda x:'NA' if x is None or x=='' else f'{float(x):.6g}'
    get=lambda role,group,metric:next(row for row in stats if row['role']==role and row['group']==group and row['metric']==metric)
    lines=['# D_pred 离线诊断','',
        '本轮只复算父实验已保存的完整概率与固定逻辑回归参数，没有重新运行特征提取器或情绪模型。score_eval 已在前一轮查看，本结果属于探索性诊断，不是新的独立确认。','',
        '## 口径与完整性','',
        '- 固定原 clean argmax 类别 c；每版 d_k=p_clean(c)−p_k(c)，正值表示该原类别概率下降，负值表示上升。D_pred=(d_a+d_b)/2；D_abs=(|d_a|+|d_b|)/2。',
        '- 绝对抵消量=D_abs−|D_pred|；相对抵消量=绝对抵消量/D_abs。D_abs=0时相对量为NA，不补成0；相反符号严格定义为d_a*d_b<0，零值不算反向。',
        '- 全部2976条clean及5952条计划版本保留；单版本成功也记录其d_k，但不能构成K2，不用单版或绝对值替代原D_pred。',
        '- 已核对来源CSV哈希、原完整向量/类别/正确性/标量、每版原类别概率、完整D_pred与覆盖数，并从score_fit逐一复现6套原标准化逻辑回归参数。没有更改原参数或数据。',
        '- 2000次Dialogue cluster bootstrap，种子20261121；每个role从完整155个Dialogue中有放回抽155次，带上被抽Dialogue的所有话语，再限制所需有效指标及正确/错误组。相同抽样用于该role的各指标和配对组差。缺支持、常量相关和拟合失败保留NA及原因。',
        '- 95%区间为百分位区间，描述性、未校正多重比较。点估计按有效话语加权；不将同一Dialogue中的话语当作独立样本。','',
        '## 覆盖与符号抵消','',
        '| Role | 全clean | K2（正确/错误） | 相反符号 | 相对抵消≥50% | 单版成功 |','|---|---:|---:|---:|---:|---:|']
    for role,c in coverage.items():
        lines.append(f"| {role} | {c['all_rows']} | {c['K2_rows']}（{c['K2_correct']}/{c['K2_wrong']}） | {c['opposite_sign_pairs']} | {c['relative_cancellation_ge_half']} | {c['one_version_only_rows']} |")
    lines += ['','| Role | 指标 | 均值 [95% cluster CI] | 中位数 |','|---|---|---|---:|']
    for role in ROLES:
        for metric in ('D_pred','D_abs','cancellation_abs','cancellation_relative'):
            row=get(role,'all',metric)
            lines.append(f"| {role} | {metric} | {fmt(row['mean'])} [{fmt(row['mean_ci_low'])}, {fmt(row['mean_ci_high'])}] | {fmt(row['median'])} |")
    lines += ['','绝对响应幅度与有方向的平均响应是不同统计量。存在异号及抵消不等于它导致了评分失败，也不证明改用D_abs就能获得收益；本轮不拟合新的替代行为评分。','',
        '## 正确与错误样本','',
        '| Role | 指标 | 正确均值 | 错误均值 | 错误−正确 [95% cluster CI] |','|---|---|---:|---:|---|']
    for role in ROLES:
        for metric in ('D_pred','D_abs','cancellation_relative','opposite_sign'):
            a=get(role,'correct',metric);b=get(role,'wrong',metric)
            diff=next(row for row in contrasts if row['role']==role and row['metric']==metric)
            lines.append(f"| {role} | {metric} | {fmt(a['mean'])} | {fmt(b['mean'])} | {fmt(diff['difference'])} [{fmt(diff['ci_low'])}, {fmt(diff['ci_high'])}] |")
    lines += ['','完整范围、分位数、均值及中位数CI见`results/dpred/distribution_summary.csv`；单版本采用各自可用支持，K2指标仅对完整配对定义，不将这些有限覆盖结果当作总体数据集表现。','',
        '## 与clean置信度、margin及entropy的相关','',
        '| Role | Clean变量 | Pearson(D_pred) [95% CI] | Spearman(D_pred) [95% CI] |','|---|---|---|']
    for role in ROLES:
        for feature in CLEAN:
            values=[next(row for row in corr if row['role']==role and row['group']=='all' and row['response']=='D_pred' and row['clean_feature']==feature and row['method']==method) for method in ('pearson','spearman')]
            cells=[f"{fmt(row['correlation'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]" for row in values]
            lines.append(f'| {role} | {feature} | {cells[0]} | {cells[1]} |')
    lines += ['','Pearson为线性相关，Spearman每次在重采样的重复行上重新按平均秩处理并列值。分正确/错误以及D_abs、相对抵消的额外描述性结果均保留在`correlations.csv`。相关性不能单独证明控制clean变量后的增量价值。','',
        '## D_pred 系数的重采样稳定性','',
        '只在score_fit重拟合，score_eval没有进入拟合、标准化或模型选择。弱模型为logit(confidence)+D_pred；强模型另含clean margin与entropy。每次抽样后重新拟合各列StandardScaler（总体方差ddof=0），清洁列使用同一抽样行得到的clean参考scaler，再以与原实验完全相同的L2 C=1、lbfgs及固定数值设置拟合。没有搜索C或挑选抽样。','',
        '模型输出R=P(原预测正确)，因此正系数表示控制所列clean变量后D_pred越大、估计正确率越高。标准化系数表示D_pred增加该次score_fit样本1个标准差时的log-odds变化；各次标准差会变化，另外保存raw系数=标准化系数/对应SD。系数大小不能在不同缩放口径间直接比较。','',
        '| 模型 | 原标准化β | 原D_pred SD | Bootstrap β 95% CI | β>0比例 | 有效/2000 |','|---|---:|---:|---|---:|---:|']
    for b in beta_summary:
        lines.append(f"| {b['model']} | {fmt(b['original_beta_D_pred_standardized'])} | {fmt(b['original_D_pred_sd_ddof0'])} | [{fmt(b['beta_standardized_ci_low'])}, {fmt(b['beta_standardized_ci_high'])}] | {fmt(b['fraction_positive'])} | {b['valid']}/2000 |")
    lines += ['','系数符号比例是bootstrap描述量，不是检验p值；区间是否跨0及符号稳定性见上表。每次拟合的全部系数、均值/SD、有效样本数、原始尺度系数及所有NA原因见`bootstrap_coefficient_draws.csv`。本重拟合仅诊断原有score_fit估计的不确定性，不产生部署用的新模块。','',
        '## 产物与限制','',
        '- `per_version_response.csv`：5952条计划版本的原类别有方向变化及失败原因；`per_sample_response.csv`：2976条全队列，保留所有缺失与单版情况。',
        '- `distribution_summary.csv`、`correct_wrong_differences.csv`、`correlations.csv`：分角色与正确性分析和cluster CI。',
        '- `bootstrap_distribution_draws.csv`、`bootstrap_correlation_draws.csv`、`bootstrap_coefficient_draws.csv`：完整2000次结果及NA，不筛选好看的重复。',
        '- `coefficient_reproduction.csv`、`coefficient_stability.csv`、`verification.json`：原6模型复现、系数CI与独立数值核验。',
        '- 没有访问原始媒体、完整文本、官方Validation/Test或模型权重，没有重新运行情绪模型；没有新增超参数搜索、正式校准训练或公开上传。','']
    (run/'D_PRED_OFFLINE_DIAGNOSTIC.md').write_text('\n'.join(lines))
    summary={'coverage':coverage,'coefficient_stability':beta_summary,'source_inputs':inputs,
             'bootstrap':protocol['scoring']['bootstrap'],'scope':'exploratory old Train score cohorts; no new independent evaluation'}
    dump(out/'diagnostic_summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path(__file__).resolve().parent.parent)
    args=parser.parse_args();run=args.run_dir.resolve();parent=run.parent/'20261122T041838Z_meld_real_scoring'
    out=run/'results/dpred';out.mkdir(parents=True,exist_ok=False)
    anchors,source_verification,old_report,protocol=verify_sources(parent,run/'EXPERIMENT_PROTOCOL.json')
    samples,per_version=construct_rows(parent,source_verification)
    parameters=read(parent/'results/scoring/fitted_parameters.json')
    settings=old_report['settings']['logistic']
    if settings['C']!=1 or settings['random_state']!=20261121 or settings['class_weight'] is not None:
        raise ValueError('Fixed original C/seed/class direction is required')
    with threadpool_limits(limits=1):
        reproduction,models=reproduce_models(samples,parameters,settings)
        csvwrite(out/'coefficient_reproduction.csv',reproduction)
        sanitized=[{k:printable(v) if isinstance(v,float) and not np.isfinite(v) else v for k,v in row.items()} for row in samples]
        csvwrite(out/'per_sample_response.csv',sanitized);csvwrite(out/'per_version_response.csv',per_version)
        stats,corr,contrasts,coefficients=bootstrap_analysis(samples,settings,2000,20261121,out)
    summary=report(run,samples,per_version,stats,corr,contrasts,coefficients,models,reproduction,anchors,protocol)
    for filename,digest in anchors.items():
        if sha(filename)!=digest:raise RuntimeError('Original offline input changed during analysis')
    # Separate formulas and persisted row counts, not just flags set by report code.
    pv=csvrows(out/'per_version_response.csv');ps=csvrows(out/'per_sample_response.csv')
    if len(pv)!=5952 or len(ps)!=2976 or len(csvrows(out/'bootstrap_coefficient_draws.csv'))!=4000:
        raise ValueError('Persisted diagnosis coverage failed')
    for row in ps:
        if row['k2_available']=='1':
            a,b=number(row['delta_a']),number(row['delta_b'])
            assert_close(float(row['D_pred']),(a+b)/2,'Persisted signed response formula failed')
            assert_close(float(row['D_abs']),(abs(a)+abs(b))/2,'Persisted absolute response formula failed')
            assert_close(float(row['cancellation_abs']),min(abs(a),abs(b)) if a*b<0 else 0,
                         'Independent minimum-magnitude cancellation identity failed')
    verification={'status':'PASS','completed_at':datetime.now(timezone.utc).isoformat(),
        'script_sha256':sha(__file__),'source_inputs_unchanged':True,'source_inputs':anchors,
        'counts':{'all_clean':2976,'all_planned_versions':5952,'original_models_reproduced':len(reproduction),
                  'coefficient_bootstrap_rows':4000,'coefficient_bootstrap_draws_per_model':2000},
        'source_role_coverage':summary['coverage'],'all_missing_and_failed_rows_retained':True,
        'D_pred_D_abs_recomputed_from_saved_full_probabilities':True,'cancellation_independent_algebraic_check':True,
        'official_validation_accessed':False,'test_accessed':False,'new_emotion_model_inference':False,
        'new_downstream_training':False,'coefficient_fitting_uses_score_fit_only':True,
        'environment':{'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'python':sys.version},
        'outputs_sha256':{str(path.relative_to(run)):sha(path) for path in sorted(out.iterdir()) if path.is_file()}}
    verification['outputs_sha256']['D_PRED_OFFLINE_DIAGNOSTIC.md']=sha(run/'D_PRED_OFFLINE_DIAGNOSTIC.md')
    dump(out/'verification.json',verification)
    print(json.dumps({'status':'COMPLETE','coverage':summary['coverage'],'coefficient_stability':summary['coefficient_stability']}),flush=True)


if __name__=='__main__':
    main()
