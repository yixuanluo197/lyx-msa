#!/usr/bin/env python3
"""Fixed CPU reliability scoring for a future frozen natural cohort.

Current execution is restricted to explicitly declared synthetic input by default.
No model/feature extractor, source media, Test set, tuning, or split search is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 20261121
BOOTSTRAPS = 2000
CONFIDENCE_EPS = 1e-6
LOGISTIC = dict(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000,
                tol=1e-10, random_state=SEED, class_weight=None, fit_intercept=True)
METRICS = ('binary_nll', 'brier', 'auroc_error', 'auprc_error')
FEATURES = {'weak': ('logit_confidence',),
            'strong': ('logit_confidence', 'clean_margin', 'clean_entropy')}
REQUIRED = ('sample_id', 'dialogue_id', 'original_correct', 'original_confidence',
            'clean_margin', 'clean_entropy', 'D_pred', 'role')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError('CSV has no header')
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('Duplicate CSV columns')
        return list(reader)


def identity(value, name):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(name + ' must be nonempty and have no surrounding whitespace')
    return value


def finite_number(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + ' must be numeric') from error
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return value


def split_manifest_check(rows, manifest, utterance=False):
    """Exact frozen identities, including roles; never infer a role or drop rows."""
    required = ('sample_id', 'dialogue_id', 'role') if utterance else ('dialogue_id', 'role')
    expected = []; all_identities = []; all_roles = {}
    for row in manifest:
        if not all(key in row for key in required):
            raise ValueError('Split manifest missing fields: ' + ','.join(required))
        key = tuple(identity(row[field], field) for field in required)
        all_identities.append(key)
        did = row['dialogue_id']
        if did in all_roles and all_roles[did] != row['role']:
            raise ValueError('Frozen manifest assigns one Dialogue to multiple roles')
        all_roles[did] = row['role']
        if row['role'] in ('score_fit', 'score_eval'):
            expected.append(key)
    if len(all_identities) != len(set(all_identities)):
        raise ValueError('Duplicate manifest identities')
    if utterance and len({key[0] for key in all_identities}) != len(all_identities):
        raise ValueError('Duplicate sample_id in frozen utterance manifest')
    observed = {tuple(str(row[field]) for field in required) for row in rows}
    if set(expected) != observed:
        raise ValueError('Frozen manifest exact coverage/role mismatch')
    # A dialogue manifest cannot itself assign the same dialogue to both roles.
    if not utterance and len({key[0] for key in expected}) != len(expected):
        raise ValueError('Manifest contains Dialogue role overlap')
    if utterance:
        lookup = {row['sample_id']: row for row in manifest}
        for row in rows:
            for field in ('utterance_id', 'utterance_index'):
                if field in row and field in lookup[row['sample_id']] and str(row[field]) != str(lookup[row['sample_id']][field]):
                    raise ValueError('Frozen utterance identity mismatch: ' + field)


def validate_rows(raw, dialogue_manifest=None, utterance_manifest=None):
    if not raw:
        raise ValueError('No input rows')
    rows = []; seen = set(); dialogue_roles = {}; probabilities_checked = 0
    for row in raw:
        if any(key not in row for key in REQUIRED):
            raise ValueError('Input CSV must contain: ' + ','.join(REQUIRED))
        sample = identity(row['sample_id'], 'sample_id'); did = identity(row['dialogue_id'], 'dialogue_id')
        if sample in seen:
            raise ValueError('Duplicate sample_id, including across roles')
        seen.add(sample)
        role = row['role']
        if role not in ('score_fit', 'score_eval'):
            raise ValueError('Unknown role; no implicit split is permitted')
        if did in dialogue_roles and dialogue_roles[did] != role:
            raise ValueError('Dialogue overlaps score_fit and score_eval')
        dialogue_roles[did] = role
        if str(row['original_correct']) not in ('0', '1'):
            raise ValueError('original_correct must be exactly 0 or 1; correct=1')
        confidence = finite_number(row['original_confidence'], 'original_confidence')
        margin = finite_number(row['clean_margin'], 'clean_margin')
        entropy = finite_number(row['clean_entropy'], 'clean_entropy')
        if not 0 < confidence <= 1:
            raise ValueError('original_confidence must be a probability in (0,1], not a logit/percent')
        if not 0 <= margin <= confidence + 1e-6 or entropy < 0:
            raise ValueError('Clean margin/entropy inconsistent with an original maximum probability')
        if entropy + 1e-6 < -math.log(confidence):
            raise ValueError('Entropy is smaller than the max-probability lower bound')
        missing = str(row['D_pred']).strip().lower() in ('', 'na', 'nan', 'null', 'none')
        d_pred = None if missing else finite_number(row['D_pred'], 'D_pred')
        if d_pred is not None and not -1 - 1e-6 <= d_pred <= 1 + 1e-6:
            raise ValueError('D_pred must be a signed probability difference in [-1,1]')
        # Optional stronger scalar audit, without requiring a separate raw asset.
        if row.get('original_probabilities', '').strip():
            values = np.asarray(json.loads(row['original_probabilities']), dtype=float)
            if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-5):
                raise ValueError('Invalid optional clean probability vector')
            ordered = np.sort(values); nonzero = values > 0
            implied = (values.max(), ordered[-1] - ordered[-2], -np.sum(values[nonzero] * np.log(values[nonzero])))
            if not np.allclose([confidence, margin, entropy], implied, atol=1e-6, rtol=1e-5):
                raise ValueError('Confidence/margin/entropy differs from optional clean probability vector')
            probabilities_checked += 1
        clipped = float(np.clip(confidence, CONFIDENCE_EPS, 1 - CONFIDENCE_EPS))
        parsed = dict(sample_id=sample, dialogue_id=did, role=role,
                         original_correct=int(row['original_correct']), original_confidence=confidence,
                         clean_margin=margin, clean_entropy=entropy, D_pred=d_pred,
                         logit_confidence=math.log(clipped / (1 - clipped)))
        for field in ('utterance_id', 'utterance_index'):
            if field in row:
                parsed[field] = identity(str(row[field]), field)
        rows.append(parsed)
    for role in ('score_fit', 'score_eval'):
        selected = [r for r in rows if r['role'] == role]
        if not selected:
            raise ValueError('Both score_fit and score_eval are required')
        if role == 'score_fit' and len({r['original_correct'] for r in selected}) != 2:
            raise ValueError('score_fit must contain correct and error outcomes')
    if dialogue_manifest is not None:
        split_manifest_check(rows, dialogue_manifest)
    if utterance_manifest is not None:
        split_manifest_check(rows, utterance_manifest, utterance=True)
    return rows, dict(confidence_scalar_range_validated=True,
                      confidence_definition_requires_source_provenance=True,
                      optional_clean_probability_vectors_checked=probabilities_checked,
                      max_softmax_definition_numerically_verified_for_all_rows=probabilities_checked == len(rows),
                      dialogue_role_manifest_verified=dialogue_manifest is not None,
                      utterance_role_manifest_verified=utterance_manifest is not None)


def provenance_check(provenance, input_path, dialogue_manifest=None, utterance_manifest=None,
                     allow_declared_real=False, split_lock=None):
    """Declarations are auditable assertions, never proof of real blind reviews."""
    for key in ('declarant', 'declaration_note'):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError('Provenance needs an attributed ' + key)
    if provenance.get('confidence_definition') != 'original_max_softmax_probability':
        raise ValueError('Provenance must declare original max-softmax confidence')
    if provenance.get('outcome_definition') != 'original_correct_1' or provenance.get('split_frozen') is not True:
        raise ValueError('Provenance must declare correct=1 and a frozen split')
    if split_lock is not None:
        lock = json.loads(Path(split_lock).read_text())
        for manifest in (dialogue_manifest, utterance_manifest):
            if manifest is None or lock.get('hashes', {}).get(Path(manifest).name) != sha(manifest):
                raise ValueError('SPLIT_LOCK.json manifest hash mismatch')
    kind = provenance.get('data_kind')
    if kind != 'synthetic':
        if not allow_declared_real or kind != 'declared_real':
            raise ValueError('Default guard permits declared synthetic data only; real execution is not authorized here')
        if dialogue_manifest is None or utterance_manifest is None or split_lock is None:
            raise ValueError('Declared-real mode requires both frozen manifests and SPLIT_LOCK.json')
        expected = {'input_sha256': input_path, 'dialogue_manifest_sha256': dialogue_manifest,
                    'utterance_manifest_sha256': utterance_manifest, 'split_lock_sha256': split_lock}
        for key, path in expected.items():
            if provenance.get(key) != sha(path):
                raise ValueError('Declared-real provenance hash mismatch: ' + key)
    return dict(data_kind=kind, declaration_validation='Attributed declarations and file hashes only; not independent human-review evidence',
                genuine_blind_review_verified_by_this_script=False,
                strict_natural_cohort_review_gate='Must be independently supplied and enforced by root before real execution')


def fit_logistic(rows, fields, clean_reference=None):
    x = np.asarray([[r[field] for field in fields] for r in rows], dtype=float)
    y = np.asarray([r['original_correct'] for r in rows], dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError('A fitted logistic model requires both outcomes')
    scaler = StandardScaler().fit(x)  # population variance, ddof=0; score_fit only.
    if clean_reference is not None:
        if (tuple(fields[:-1]) != clean_reference['fields'] or fields[-1] != 'D_pred' or
            tuple(r['sample_id'] for r in rows) != clean_reference['fit_sample_ids']):
            raise ValueError('Shared clean scaler requires the exact same fit rows and clean features')
        reference = clean_reference['scaler']
        for attribute in ('mean_', 'var_', 'scale_'):
            values = getattr(scaler, attribute)
            clean_values = getattr(reference, attribute)
            if not np.allclose(values[:-1], clean_values, atol=1e-12, rtol=1e-12):
                raise ValueError('Matched-fit scaler differs beyond floating roundoff')
            # Feature-count-dependent summation can alter one-column means by
            # 1 ulp. Share the actual same-Fit clean scaler, rather than treating
            # equal coverage as proof of bit-identical independent reductions.
            values[:-1] = clean_values
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = LogisticRegression(**LOGISTIC).fit(scaler.transform(x), y)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError('Non-finite fitted CPU logistic parameters')
    return dict(fields=tuple(fields), scaler=scaler, model=model,
                clean_scaler_reused_from_same_fit=clean_reference is not None,
                fit_sample_ids=tuple(r['sample_id'] for r in rows),
                fit_dialogue_ids=tuple(sorted({r['dialogue_id'] for r in rows})))


def predict(fitted, rows):
    x = np.asarray([[r[field] for field in fitted['fields']] for r in rows], dtype=float)
    return fitted['model'].predict_proba(fitted['scaler'].transform(x))[:, 1]


def model_record(fitted):
    return dict(features=list(fitted['fields']), fit_sample_ids=list(fitted['fit_sample_ids']),
                fit_dialogue_ids=list(fitted['fit_dialogue_ids']),
                scaler_mean=fitted['scaler'].mean_.tolist(), scaler_scale=fitted['scaler'].scale_.tolist(),
                scaler_variance=fitted['scaler'].var_.tolist(), scaler_ddof=0,
                coefficient=fitted['model'].coef_.tolist(), intercept=fitted['model'].intercept_.tolist(),
                classes=fitted['model'].classes_.tolist(), optimizer_iterations=fitted['model'].n_iter_.tolist(),
                clean_scaler_reused_from_same_fit=fitted['clean_scaler_reused_from_same_fit'],
                fixed_logistic_settings=LOGISTIC)


def metric_batch(y_correct, reliability, weights):
    """Weighted repeated-cluster metrics; exact score ties use grouped ranks.

    AUROC/AP use error=positive and score=1-R. AP is sklearn-style average
    precision, not trapezoidal PR area. Both need positive and negative support.
    """
    y = np.asarray(y_correct, dtype=float); r = np.asarray(reliability, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != len(y) or len(y) != len(r) or (weights < 0).any():
        raise ValueError('Invalid weighted metric shapes/weights')
    if not np.isfinite(r).all() or (r < 0).any() or (r > 1).any():
        raise ValueError('Reliability must be finite probabilities')
    total = weights.sum(axis=1); clipped = np.clip(r, 1e-15, 1 - 1e-15)
    nll = -(y * np.log(clipped) + (1-y) * np.log(1-clipped))
    divide = lambda a,b: np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=b>0)
    output = dict(binary_nll=divide(weights @ nll,total), brier=divide(weights @ ((r-y)**2),total))
    if len(y) == 0:
        output.update(auroc_error=np.full(len(weights),np.nan),auprc_error=np.full(len(weights),np.nan))
        return output
    error = 1-y; order = np.argsort(1-r, kind='mergesort'); risk = (1-r)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(risk) != 0) + 1]
    positive = np.add.reduceat(weights[:,order] * error[order], starts, axis=1)
    negative = np.add.reduceat(weights[:,order] * (1-error[order]), starts, axis=1)
    p,n = positive.sum(1),negative.sum(1)
    area = np.sum(positive * (np.cumsum(negative,axis=1)-negative+.5*negative),axis=1)
    output['auroc_error'] = divide(area,p*n)
    pd,nd = positive[:,::-1],negative[:,::-1]
    precision = np.divide(np.cumsum(pd,axis=1),np.cumsum(pd+nd,axis=1),
                          out=np.zeros_like(pd),where=np.cumsum(pd+nd,axis=1)>0)
    ap = divide(np.sum(pd*precision,axis=1),p)
    ap[(p<=0)|(n<=0)] = np.nan
    output['auprc_error'] = ap
    return output


def metric_point(y, r):
    values = metric_batch(y,r,np.ones((1,len(y)),dtype=float))
    return {key:float(value[0]) if np.isfinite(value[0]) else None for key,value in values.items()}


def bootstrap_weights(dialogues, repeats=BOOTSTRAPS):
    universe = sorted(set(dialogues)); position = {did:i for i,did in enumerate(universe)}
    draws = np.random.RandomState(SEED).randint(0,len(universe),(repeats,len(universe)))
    counts = np.zeros((repeats,len(universe)),dtype=np.int64)
    np.add.at(counts,(np.repeat(np.arange(repeats),len(universe)),draws.ravel()),1)
    return counts[:,[position[did] for did in dialogues]],dict(seed=SEED,requested=repeats,
        resampling_unit='Dialogue, paired predictions and shared draws across models/scopes',
        rng='numpy RandomState MT19937; sorted Dialogue universe',
        interval='95% percentile [2.5%,97.5%], NumPy linear quantiles',
        dialogue_universe=universe,draw_indices_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest())


def compare(y, baseline, augmented, weights, selector):
    y=np.asarray(y)[selector];baseline=np.asarray(baseline)[selector];augmented=np.asarray(augmented)[selector]
    w=weights[:,selector]
    first,second=metric_point(y,baseline),metric_point(y,augmented)
    first_boot,second_boot=metric_batch(y,baseline,w),metric_batch(y,augmented,w)
    intervals={}
    for key in METRICS:
        direction=1 if key in ('binary_nll','brier') else -1
        valid=np.isfinite(first_boot[key])&np.isfinite(second_boot[key])
        improvement=direction*(first_boot[key][valid]-second_boot[key][valid])
        ci=np.quantile(improvement,[.025,.975]).tolist() if len(improvement) else [None,None]
        intervals[key]=dict(improvement=direction*(first[key]-second[key]) if first[key] is not None and second[key] is not None else None,
            improvement_ci_low=ci[0],improvement_ci_high=ci[1],valid_draws=int(valid.sum()),
            requested_draws=len(weights),positive_means='augmentation improves',
            status='DESCRIPTIVE' if len(improvement)==len(weights) else 'PARTIAL_CLUSTER_SUPPORT' if len(improvement) else 'NO_TWO_CLASS_OR_NONEMPTY_SUPPORT')
    return dict(n=len(y),correct=int(y.sum()),errors=int(len(y)-y.sum()),
                baseline=first,augmented=second,paired_dialogue_improvement=intervals)


def score(rows, scalar_validation, declaration, *, _bootstrap_repeats=BOOTSTRAPS):
    """CPU-only fitting. Private repeat override exists solely for bounded tests."""
    fit=[r for r in rows if r['role']=='score_fit'];evaluation=[r for r in rows if r['role']=='score_eval']
    covered_fit=[r for r in fit if r['D_pred'] is not None]
    covered_eval=np.asarray([r['D_pred'] is not None for r in evaluation],dtype=bool)
    y=np.asarray([r['original_correct'] for r in evaluation]);weights,boot=bootstrap_weights([r['dialogue_id'] for r in evaluation],_bootstrap_repeats)
    models={};comparisons={};predictions=[dict(r) for r in evaluation]
    available=len({r['original_correct'] for r in covered_fit})==2
    reason=None if available else 'D_pred-covered score_fit has no rows or only one outcome; no augmented model fit'
    for strength,fields in FEATURES.items():
        full=fit_logistic(fit,fields);baseline=predict(full,evaluation);augmented=baseline.copy()
        models[strength+'_baseline_all_fit']=model_record(full)
        if available:
            matched=fit_logistic(covered_fit,fields);aug=fit_logistic(covered_fit,tuple(fields)+('D_pred',),clean_reference=matched)
            if not np.array_equal(matched['scaler'].mean_,aug['scaler'].mean_[:-1]) or not np.array_equal(matched['scaler'].scale_,aug['scaler'].scale_[:-1]):
                raise AssertionError('Matched-fit common clean feature scaling differs')
            eval_covered=[r for r in evaluation if r['D_pred'] is not None]
            if eval_covered:augmented[covered_eval]=predict(aug,eval_covered)
            matched_predictions=baseline.copy()
            if eval_covered:matched_predictions[covered_eval]=predict(matched,eval_covered)
            models[strength+'_baseline_covered_fit']=model_record(matched)
            models[strength+'_augmented_covered_fit']=model_record(aug)
            result=compare(y,matched_predictions,augmented,weights,covered_eval)
            result.update(comparison_kind='MATCHED_FIT_COVERAGE_FEATURE_INCREMENT',fit_rows=len(covered_fit),
                          status='COMPUTED' if covered_eval.any() else 'NO_COVERED_EVAL_ROWS')
            comparisons[strength+'_matched_fit_covered_eval']=result
        else:
            comparisons[strength+'_matched_fit_covered_eval']=dict(status='NOT_AVAILABLE',reason=reason,n=int(covered_eval.sum()))
        if not np.array_equal(augmented[~covered_eval],baseline[~covered_eval]):
            raise AssertionError('Missing D_pred must fall back bit-exactly to corresponding all-fit baseline')
        for scope,selector in [('all_eval',np.ones(len(evaluation),bool)),('covered_eval',covered_eval)]:
            result=compare(y,baseline,augmented,weights,selector)
            result.update(comparison_kind='DEPLOYMENT_POLICY_ALL_FIT_BASELINE',status='COMPUTED' if available else 'FALLBACK_ONLY_NO_AUGMENTED_MODEL',
                          baseline_fit_rows=len(fit),augmented_fit_rows=len(covered_fit) if available else 0,
                          fit_cohort_confounding='Deployment comparison changes training coverage; use matched-fit comparison for feature increment')
            comparisons[strength+'_'+scope]=result
        for i,row in enumerate(predictions):
            row[strength+'_baseline_R']=float(baseline[i]);row[strength+'_augmented_policy_R']=float(augmented[i])
            row[strength+'_augmented_used']=bool(available and covered_eval[i]);row[strength+'_fallback_to_baseline']=bool(not available or not covered_eval[i])
            if available:row[strength+'_matched_fit_baseline_R']=float(matched_predictions[i]) if covered_eval[i] else None
    raw=np.asarray([r['original_confidence'] for r in evaluation])
    report=dict(status='COMPLETE_CPU_SCORING',data_provenance=declaration,scalar_validation=scalar_validation,
        outcome='original correct=1; model R=P(correct)',ranking='error positive; score=1-R',
        primary_metric='binary_nll',secondary_metrics=['brier','auroc_error','auprc_error'],
        average_precision_definition='Stepwise average precision, not trapezoidal PR-AUC; requires both error/correct support',
        point_estimator='Utterance-weighted natural-cohort mean; clusters resampled for uncertainty, no Dialogue equalization of point metrics',
        settings=dict(logistic=LOGISTIC,scaler='StandardScaler score_fit rows only, ddof=0; constant scale=1',
                      logit_confidence_clip=CONFIDENCE_EPS,binary_nll_probability_clip=1e-15,
                      no_hyperparameter_selection=True,device='CPU'),
        fit_rows=len(fit),eval_rows=len(evaluation),fit_dialogues=len({r['dialogue_id'] for r in fit}),
        eval_dialogues=len({r['dialogue_id'] for r in evaluation}),covered_fit_rows=len(covered_fit),
        covered_eval_rows=int(covered_eval.sum()),uncovered_eval_rows=int((~covered_eval).sum()),
        D_pred_eval_coverage=float(covered_eval.mean()),augmentation_available=available,augmentation_unavailable_reason=reason,
        missing_policy='No D_pred imputation. Fit augmentations only on covered score_fit; missing eval rows use exact all-fit baseline predictions.',
        matched_fit_control='Clean-only and +D_pred models share the exact covered score_fit rows; compare only covered score_eval',
        bootstrap=boot,comparisons=comparisons,
        raw_original_confidence_reference=dict(all_eval=metric_point(y,raw),covered_eval=metric_point(y[covered_eval],raw[covered_eval])),
        versions=dict(numpy=np.__version__,sklearn=sklearn.__version__),
        limitations=['Manifest coverage/hash verification is not independent proof of human blind reviews or a natural cohort.',
                     'Bootstrap intervals are descriptive and not multiplicity-corrected; no scalar score calibration gain is assumed.',
                     'No evaluation labels/features are used for fitting, scaling, tuning or model selection.'])
    return report,predictions,models


def run(args):
    if args.output.exists():raise FileExistsError('Refuse existing output directory')
    declaration=json.loads(args.provenance.read_text())
    declared=provenance_check(declaration,args.input,args.manifest,args.utterance_manifest,args.allow_declared_real,args.split_lock)
    rows,validation=validate_rows(read_csv(args.input),read_csv(args.manifest) if args.manifest else None,
                                  read_csv(args.utterance_manifest) if args.utterance_manifest else None)
    # All source, role and scalar guards run before any fitting/output creation.
    report,predictions,parameters=score(rows,validation,declared)
    report['input_sha256']={str(path.resolve()):sha(path) for path in (args.input,args.provenance,args.manifest,args.utterance_manifest,args.split_lock) if path is not None}
    report['script_sha256']=sha(__file__)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'scoring_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.output/'fitted_parameters.json').write_text(json.dumps(parameters,indent=2,allow_nan=False)+'\n')
    with (args.output/'evaluation_predictions.csv').open('w',newline='') as stream:
        keys=list(dict.fromkeys(k for row in predictions for k in row));writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(predictions)
    print(json.dumps(dict(status=report['status'],fit_rows=report['fit_rows'],eval_rows=report['eval_rows'],
                          D_pred_eval_coverage=report['D_pred_eval_coverage'],output=str(args.output))))


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Required input CSV columns:
  sample_id,dialogue_id,original_correct,original_confidence,clean_margin,
  clean_entropy,D_pred,role  (role: score_fit | score_eval)
D_pred may be blank/NA; all other features must be finite. C must be the
original maximum softmax probability in (0,1], never a logit or percentage.
Optional original_probabilities JSON vectors permit direct scalar audits.

Required provenance JSON declaration:
  {"data_kind":"synthetic", "declarant":"NAME", "declaration_note":"NOTE",
   "confidence_definition":"original_max_softmax_probability",
   "outcome_definition":"original_correct_1", "split_frozen":true}
This is an attributed declaration, not proof of genuine human blind review.

Example (synthetic only; uses only installed CPU sklearn):
  python score_heldout.py --input synthetic.csv --provenance provenance.json \\
    --manifest SPLIT_DIALOGUES.csv --utterance-manifest SPLIT_UTTERANCES.csv \\
    --split-lock SPLIT_LOCK.json \\
    --output NEW_LOCAL_DIRECTORY

--manifest requires exact dialogue_id,role coverage; --utterance-manifest
requires exact sample_id,dialogue_id,role coverage. Optional utterance_id/index
are checked when supplied in both tables. The full frozen Train manifest may
include non-scoring roles; its score_fit/score_eval slice must match exactly.
SPLIT_LOCK.json hashes must bind the two manifest basenames. Input sample IDs must be
globally unique; no Dialogue may occur in both roles. No split is generated.
Future declared-real execution needs explicit --allow-declared-real, BOTH
manifests, --split-lock, and matching input_sha256/dialogue_manifest_sha256/
utterance_manifest_sha256/split_lock_sha256 in provenance. Root must separately enforce actual
human-review and natural-cohort eligibility; this flag is not authorization.
No real cohort execution is part of the current readiness task.

Models: fixed C=1 L2 logistic, lbfgs, max_iter=2000, tol=1e-10, no balancing;
standardization learned only from each model's score_fit rows. Confidence
logit clips to [1e-6,1-1e-6]; raw confidence remains separately reported.
NLL is primary; Brier, error-positive AUROC/average precision secondary.
Exactly 2000 shared paired Dialogue bootstrap draws; no tuning or refitting.
Missing D_pred: baseline fallback; covered-fit clean-only controls isolate
feature increment from the deployment comparison's training-coverage change.
Existing output, invalid sources/roles/scalars, or nonconvergence fail closed.
''')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--provenance',type=Path,required=True)
    p.add_argument('--manifest',type=Path,help='Frozen exact Dialogue-role CSV')
    p.add_argument('--utterance-manifest',type=Path,help='Frozen exact sample/Dialogue/role CSV')
    p.add_argument('--split-lock',type=Path,help='Root frozen SPLIT_LOCK.json, hashes keyed by manifest basename')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-declared-real',action='store_true',help='Future explicit guard override; requires both hash-bound manifests, not current authorization')
    return p


if __name__=='__main__':run(parser().parse_args())
