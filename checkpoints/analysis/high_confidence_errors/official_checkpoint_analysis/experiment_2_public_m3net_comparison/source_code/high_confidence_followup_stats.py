"""Pure CPU, read-only prediction validation and experiment 2–5 accounting.

Ratios and accuracy/F1 are fractions, never percentages. Undefined ratios are
None. Threshold detail tables intentionally repeat a sample at every qualifying
threshold. Class probability columns preserve the source CSV class order.
"""
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

THRESHOLDS = (.8, .9, .95)
CONDITIONS = ('w/o_text', 'w/o_audio', 'w/o_visual')
BINS = ((0,.5),( .5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,.95),(.95,1.0))
META = ['dataset','dialogue_id','utterance_index','source_utterance_id','utterance_text','true_label','true_emotion']
PRED = ['predicted_label','predicted_emotion','predicted_confidence','true_label_probability','correct']
THRESHOLD_FIELDS = [f'{name}_ge_{str(t).replace(".","p")}' for t in THRESHOLDS
                    for name in ('errors','high_confidence_predictions','high_confidence_error_rate')]
PAIRED_FIELDS = META + [f'{model}_{field}' for model in ('baseline','cmerc') for field in PRED]
CANONICAL_FIELDS = META + ['model','input_condition'] + PRED
TABLE_FIELDS = {
    'exp2_paired_outcomes': PAIRED_FIELDS + ['threshold','outcome','wrong_class_changed'],
    'exp2_summary': ['dataset','threshold','valid_test_utterances','baseline_high_confidence_errors'] +
        [f'{outcome}_{suffix}' for outcome in ('corrected','still_wrong_confidence_lower','still_wrong_confidence_not_lower') for suffix in ('count','ratio')] +
        ['new_cmerc_high_confidence_errors','new_error_eligible_count','new_error_ratio'],
    'exp2_new_high_confidence_errors': PAIRED_FIELDS + ['threshold'],
    'exp3_cm_remaining_high_confidence_errors': PAIRED_FIELDS + ['threshold','error_type'],
    'exp3_class_summary': ['dataset','threshold','true_label','true_emotion','class_support','total_errors','class_error_rate','high_confidence_error_count','high_confidence_prediction_count','high_confidence_error_rate','high_confidence_error_fraction_of_class'],
    'exp4_modality_ablation_predictions': META + ['model','input_condition','removed_modality'] + [f'{condition}_{field}' for condition in ('full','masked') for field in PRED],
    'exp4_modality_ablation_summary': ['dataset','model','input_condition','valid_utterances','total_errors','mean_error_confidence'] + THRESHOLD_FIELDS,
    'exp4_full_correct_to_masked_error': ['dataset','model','input_condition','removed_modality','threshold','full_correct_samples','new_high_confidence_error_count','ratio','full_wrong_samples','full_wrong_still_wrong_count','full_wrong_still_wrong_high_confidence_count','full_wrong_still_wrong_high_confidence_ratio'],
    'exp4_new_high_confidence_errors': META + ['model','input_condition','removed_modality','threshold'] + [f'{condition}_{field}' for condition in ('full','masked') for field in PRED],
    'exp4_existing_errors': META + ['model','input_condition','removed_modality','masked_ge_0p8','masked_ge_0p9','masked_ge_0p95'] + [f'{condition}_{field}' for condition in ('full','masked') for field in PRED],
    'exp5_error_confidence_distribution': ['dataset','confidence_bin','baseline_error_count','baseline_ratio_of_errors','cmerc_error_count','cmerc_ratio_of_errors'],
    'exp5_overall_summary': ['dataset','model','valid_test_utterances','total_errors','accuracy','weighted_f1','mean_error_confidence','median_error_confidence'] + THRESHOLD_FIELDS,
}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _boolean(value):
    if value in (True, 'True', 'true', '1'): return True
    if value in (False, 'False', 'false', '0'): return False
    raise ValueError(f'Invalid boolean {value!r}')


def load_full_records(exp1_root):
    """Load only original all_predictions CSVs without modifying any input."""
    result = {}
    for path in sorted(Path(exp1_root).glob('*/*/all_predictions.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = []
            for raw in csv.DictReader(stream):
                row = {field: raw[field] for field in META if field not in ('utterance_text','utterance_index','true_label')}
                row.update(utterance_text=raw['text'], utterance_index=int(raw['utterance_index']),
                           true_label=int(raw['true_label']), model=raw['model'], input_condition='full',
                           predicted_label=int(raw['predicted_label']), predicted_emotion=raw['predicted_emotion'],
                           predicted_confidence=float(raw['max_probability']),
                           true_label_probability=float(raw['true_class_probability']), correct=not _boolean(raw['is_error']))
                row.update({key: float(value) for key,value in raw.items() if key.startswith('probability_')})
                rows.append(row)
        key = (path.parent.parent.name, path.parent.name)
        _validate(rows, key, 'full')
        result[key] = rows
    if not result:
        raise ValueError(f'No all_predictions.csv under {exp1_root}')
    return result


def _key(row):
    return row['dataset'], str(row['dialogue_id']), row['utterance_index']


def _probability_fields(rows):
    return [field for field in rows[0] if field.startswith('probability_')] if rows else []


def _validate(rows, group, condition):
    seen = set()
    fields = _probability_fields(rows)
    for row in rows:
        if (row['dataset'],row['model']) != group or row['input_condition'] != condition:
            raise ValueError('Prediction group or input condition mismatch')
        if isinstance(row['utterance_index'],bool) or not isinstance(row['utterance_index'],int) or row['utterance_index'] < 0:
            raise ValueError('Invalid utterance index')
        if _key(row) in seen: raise ValueError('Duplicate sample ID')
        seen.add(_key(row))
        if _probability_fields([row]) != fields or len(fields) < 2:
            raise ValueError('Missing or inconsistent class probability columns')
        probabilities = [row[f] for f in fields]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or not math.isclose(sum(probabilities),1,rel_tol=0,abs_tol=1e-5):
            raise ValueError('Invalid class probabilities')
        true, pred = row['true_label'], row['predicted_label']
        if any(isinstance(label,bool) or not isinstance(label,int) or not 0 <= label < len(fields) for label in (true,pred)):
            raise ValueError('Invalid class label')
        if pred != max(range(len(fields)),key=probabilities.__getitem__): raise ValueError('Prediction is not probability argmax')
        if row['true_emotion'] != fields[true][12:] or row['predicted_emotion'] != fields[pred][12:]:
            raise ValueError('Emotion names do not match class probability order')
        if not math.isclose(row['predicted_confidence'], probabilities[pred],abs_tol=1e-7) or not math.isclose(row['true_label_probability'],probabilities[true],abs_tol=1e-7):
            raise ValueError('Confidence does not match class probabilities')
        if not isinstance(row['correct'],bool) or row['correct'] != (true == pred): raise ValueError('Incorrect correctness flag')


def _align(reference, other, strict_order=False):
    if set(_key(r) for r in reference) != set(_key(r) for r in other):
        raise ValueError('Sample sets differ')
    other_by_key = {_key(r):r for r in other}
    if len(other_by_key) != len(other): raise ValueError('Duplicate aligned sample')
    if strict_order and [_key(r) for r in reference] != [_key(r) for r in other]:
        raise ValueError('Masked sample order differs from full input')
    ordered = [other_by_key[_key(r)] for r in reference]
    for left,right in zip(reference,ordered):
        for field in META + ['source_utterance_id']:
            if left[field] != right[field]: raise ValueError(f'Sample metadata differ: {field}')
        if _probability_fields([left]) != _probability_fields([right]): raise ValueError('Class probability mappings differ')
    return ordered


def _high_error(row, threshold):
    return not row['correct'] and row['predicted_confidence'] >= threshold


def _prefix(row, prefix):
    return {f'{prefix}_{field}': value for field,value in row.items() if field in PRED or field.startswith('probability_')}


def _paired(baseline, cmerc):
    return {**{f:baseline[f] for f in META}, **_prefix(baseline,'baseline'), **_prefix(cmerc,'cmerc')}


def _masked_pair(full, masked):
    return {**{f:full[f] for f in META}, 'model':full['model'], 'input_condition':masked['input_condition'],
            'removed_modality':masked['input_condition'][4:], **_prefix(full,'full'), **_prefix(masked,'masked')}


def _threshold_counts(rows):
    result = {}
    for threshold in THRESHOLDS:
        suffix = str(threshold).replace('.','p')
        count = sum(r['predicted_confidence'] >= threshold for r in rows)
        errors = sum(_high_error(r,threshold) for r in rows)
        result.update({f'errors_ge_{suffix}':errors, f'high_confidence_predictions_ge_{suffix}':count,
                       f'high_confidence_error_rate_ge_{suffix}':ratio(errors,count)})
    return result


def _weighted_f1(rows):
    support = Counter(r['true_label'] for r in rows)
    predicted = Counter(r['predicted_label'] for r in rows)
    correct = Counter(r['true_label'] for r in rows if r['correct'])
    return ratio(sum(n * ratio(2*correct[c], n+predicted[c]) for c,n in support.items()),len(rows))


def _sorted(rows, confidence):
    return sorted(rows, key=lambda r: (-r[confidence],r['dataset'],r['dialogue_id'],r['utterance_index'],r.get('threshold',0)))


def generate_tables(full_records, masked_records=None):
    """Validate all inputs, then return CSV-ready tables; None defers experiment 4.

    When masked_records is nonempty every full group must have all three complete
    masked conditions in the original per-condition sample order. Baseline may
    be absent; no paired rows are fabricated in that case.
    """
    for group,rows in full_records.items(): _validate(rows,group,'full')
    if masked_records and set(masked_records) != set(full_records): raise ValueError('Masked model groups differ')
    masks = {}
    for group,rows in (masked_records or {}).items():
        if any(r['input_condition'] not in CONDITIONS for r in rows): raise ValueError('Unknown masked condition')
        for condition in CONDITIONS:
            selected = [r for r in rows if r['input_condition']==condition]
            _validate(selected,group,condition)
            masks[group,condition] = _align(full_records[group],selected,strict_order=True)
    tables = {name:[] for name in TABLE_FIELDS}
    datasets = sorted({dataset for dataset,_ in full_records})
    for dataset in datasets:
        baseline = full_records.get((dataset,'baseline'))
        cmerc = full_records.get((dataset,'cmerc'))
        if baseline is not None and cmerc is not None:
            paired = list(zip(baseline,_align(baseline,cmerc)))
            for threshold in THRESHOLDS:
                selected = [(b,c) for b,c in paired if _high_error(b,threshold)]
                outcomes = Counter()
                for b,c in selected:
                    outcome = ('corrected' if c['correct'] else 'still_wrong_confidence_lower' if c['predicted_confidence']<b['predicted_confidence'] else 'still_wrong_confidence_not_lower')
                    outcomes[outcome] += 1
                    tables['exp2_paired_outcomes'].append(dict(_paired(b,c),threshold=threshold,outcome=outcome,wrong_class_changed=not c['correct'] and b['predicted_label'] != c['predicted_label']))
                new = [(b,c) for b,c in paired if not _high_error(b,threshold) and _high_error(c,threshold)]
                tables['exp2_new_high_confidence_errors'].extend(dict(_paired(b,c),threshold=threshold) for b,c in new)
                stat = dict(dataset=dataset,threshold=threshold,valid_test_utterances=len(paired),baseline_high_confidence_errors=len(selected),new_cmerc_high_confidence_errors=len(new),new_error_eligible_count=len(paired)-len(selected),new_error_ratio=ratio(len(new),len(paired)-len(selected)))
                for outcome in ('corrected','still_wrong_confidence_lower','still_wrong_confidence_not_lower'):
                    stat[f'{outcome}_count']=outcomes[outcome]; stat[f'{outcome}_ratio']=ratio(outcomes[outcome],len(selected))
                tables['exp2_summary'].append(stat)
                for b,c in paired:
                    if _high_error(c,threshold):
                        error_type = ('baseline_correct_cmerc_new_error' if b['correct'] else 'both_high_confidence_wrong' if _high_error(b,threshold) else 'baseline_wrong_not_high_confidence')
                        tables['exp3_cm_remaining_high_confidence_errors'].append(dict(_paired(b,c),threshold=threshold,error_type=error_type))
        elif cmerc is not None:
            for threshold in THRESHOLDS:
                for c in cmerc:
                    if _high_error(c,threshold):
                        tables['exp3_cm_remaining_high_confidence_errors'].append({**{f:c[f] for f in META},**_prefix(c,'cmerc'),'threshold':threshold,'error_type':'baseline_unavailable'})
        if cmerc is not None:
            for threshold in THRESHOLDS:
                for label,emotion in sorted({(r['true_label'],r['true_emotion']) for r in cmerc}):
                    rows=[r for r in cmerc if r['true_label']==label]
                    errors=sum(not r['correct'] for r in rows)
                    hc_errors=sum(_high_error(r,threshold) for r in rows)
                    hc_predictions=sum(r['predicted_confidence']>=threshold for r in rows)
                    tables['exp3_class_summary'].append(dict(dataset=dataset,threshold=threshold,true_label=label,true_emotion=emotion,class_support=len(rows),total_errors=errors,class_error_rate=ratio(errors,len(rows)),high_confidence_error_count=hc_errors,high_confidence_prediction_count=hc_predictions,high_confidence_error_rate=ratio(hc_errors,hc_predictions),high_confidence_error_fraction_of_class=ratio(hc_errors,len(rows))))
        for lo,hi in BINS:
            stat=dict(dataset=dataset,confidence_bin=f'[{lo:g}, {hi:g}{"]" if hi==1 else ")"}')
            for model in ('baseline','cmerc'):
                rows=full_records.get((dataset,model))
                errors=[r for r in rows if not r['correct']] if rows is not None else None
                count=sum(lo<=r['predicted_confidence'] and (r['predicted_confidence']<=hi if hi==1 else r['predicted_confidence']<hi) for r in errors) if errors is not None else None
                stat[f'{model}_error_count']=count
                stat[f'{model}_ratio_of_errors']=ratio(count,len(errors)) if errors is not None else None
            tables['exp5_error_confidence_distribution'].append(stat)
    for (dataset,model),full in full_records.items():
        errors = [r['predicted_confidence'] for r in full if not r['correct']]
        tables['exp5_overall_summary'].append(dict(dataset=dataset,model=model,valid_test_utterances=len(full),total_errors=len(errors),accuracy=ratio(len(full)-len(errors),len(full)),weighted_f1=_weighted_f1(full),mean_error_confidence=mean(errors) if errors else None,median_error_confidence=median(errors) if errors else None,**_threshold_counts(full)))
        for condition in (('full',)+CONDITIONS if masked_records is not None else ()):
            rows=full if condition=='full' else masks.get(((dataset,model),condition))
            if rows is None: continue
            errors = [r['predicted_confidence'] for r in rows if not r['correct']]
            tables['exp4_modality_ablation_summary'].append(dict(dataset=dataset,model=model,input_condition=condition,valid_utterances=len(rows),total_errors=len(errors),mean_error_confidence=mean(errors) if errors else None,**_threshold_counts(rows)))
            if condition=='full': continue
            pairs=list(zip(full,rows))
            for f,m in pairs:
                detail=_masked_pair(f,m)
                tables['exp4_modality_ablation_predictions'].append(detail)
                if not f['correct'] and not m['correct']:
                    tables['exp4_existing_errors'].append(dict(detail,**{f'masked_ge_{str(t).replace(".","p")}':m['predicted_confidence']>=t for t in THRESHOLDS}))
            for threshold in THRESHOLDS:
                full_correct=sum(r['correct'] for r in full)
                new=[(f,m) for f,m in pairs if f['correct'] and _high_error(m,threshold)]
                existing=[(f,m) for f,m in pairs if not f['correct'] and not m['correct']]
                high_existing=sum(_high_error(m,threshold) for f,m in existing)
                tables['exp4_full_correct_to_masked_error'].append(dict(dataset=dataset,model=model,input_condition=condition,removed_modality=condition[4:],threshold=threshold,full_correct_samples=full_correct,new_high_confidence_error_count=len(new),ratio=ratio(len(new),full_correct),full_wrong_samples=len(full)-full_correct,full_wrong_still_wrong_count=len(existing),full_wrong_still_wrong_high_confidence_count=high_existing,full_wrong_still_wrong_high_confidence_ratio=ratio(high_existing,len(full)-full_correct)))
                tables['exp4_new_high_confidence_errors'].extend(dict(_masked_pair(f,m),threshold=threshold) for f,m in new)
    for name,confidence in (('exp2_paired_outcomes','baseline_predicted_confidence'),('exp2_new_high_confidence_errors','cmerc_predicted_confidence'),('exp3_cm_remaining_high_confidence_errors','cmerc_predicted_confidence'),('exp4_new_high_confidence_errors','masked_predicted_confidence'),('exp4_existing_errors','masked_predicted_confidence')):
        tables[name]=_sorted(tables[name],confidence)
    # Dynamic union keeps all emotion probabilities while supporting empty CSVs.
    probability_fields=list(dict.fromkeys(field for rows in full_records.values() for field in _probability_fields(rows)))
    for name in ('exp2_paired_outcomes','exp2_new_high_confidence_errors','exp3_cm_remaining_high_confidence_errors'):
        for prefix in ('baseline','cmerc'):
            for field in probability_fields:
                key=f'{prefix}_{field}'
                if key not in TABLE_FIELDS[name]: TABLE_FIELDS[name].append(key)
    for name in ('exp4_modality_ablation_predictions','exp4_new_high_confidence_errors','exp4_existing_errors'):
        for prefix in ('full','masked'):
            for field in probability_fields:
                key=f'{prefix}_{field}'
                if key not in TABLE_FIELDS[name]: TABLE_FIELDS[name].append(key)
    return tables
