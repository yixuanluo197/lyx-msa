#!/usr/bin/env python3
"""Offline decomposition of frozen MELD response events, without model access.

Original-label transitions are reference calculations, not verified post-edit
correctness. Every fixed event, including failed/missing probability rows, stays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


RUN = Path(__file__).resolve().parents[1]
PARENT = RUN.parent / '20261120T170835Z'
FAMILIES = {'neutral_append': ('neutral_a', 'neutral_b'),
            'semantic_paraphrase': ('paraphrase_a', 'paraphrase_b')}
THRESHOLD = .02
NA = 'NA'
SCALAR_ATOL = SCALAR_RTOL = 1e-6


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError('Malformed CSV record')
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refuse an undocumented empty table')
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def truth(value):
    if value is True or str(value).strip().lower() in {'true', '1'}:
        return True
    if value is False or str(value).strip().lower() in {'false', '0'}:
        return False
    raise ValueError('Expected a stored boolean')


def vector(value):
    if value is None or str(value).strip() in {'', 'NA', 'None', 'null'}:
        return None
    p = np.asarray(json.loads(value), dtype=np.float64)
    if p.shape != (7,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Expected seven finite nonnegative MELD probabilities')
    if not np.isclose(p.sum(), 1., atol=1e-5, rtol=0):
        raise ValueError('Invalid probability normalization; do not silently renormalize')
    return p


def scalar_equal(value, expected, field):
    if str(value).strip() in {'', 'NA', 'None', 'null'}:
        raise ValueError('Missing stored scalar: ' + field)
    if not np.isclose(float(value), expected, atol=SCALAR_ATOL, rtol=SCALAR_RTOL):
        raise ValueError('Stored scalar conflicts with complete probability: ' + field)


def sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def response_bin(value):
    if value < -THRESHOLD:
        return 'DECREASE_GT_0.02'
    if value > THRESHOLD:
        return 'INCREASE_GT_0.02'
    return 'WITHIN_0.02'


def seven_bin(value):
    if value < -.20:return 1
    if value < -.10:return 2
    if value < -.02:return 3
    if value <= .02:return 4
    if value <= .10:return 5
    if value <= .20:return 6
    return 7


def delta_bins(selected, field, aggregation):
    applicable=field in ('delta_C','delta_P')
    result=dict(seven_bin_scheme='FIXED_7_DELTA_BINS' if applicable else NA,
        bin_count_unit='INDIVIDUAL_EVENTS' if applicable else NA,
        bin_fraction_unit='Mean within-Dialogue event fractions' if aggregation=='EQUAL_DIALOGUE' and applicable else 'Event fractions' if applicable else NA)
    for index in range(1,8):
        result[f'bin_{index}_count']=sum(seven_bin(float(r[field]))==index for r in selected) if applicable else NA
        result[f'bin_{index}_fraction']=rate(selected,lambda r,i=index:seven_bin(float(r[field]))==i,aggregation) if applicable else NA
    return result


def js(p, q):
    m = (p + q) / 2
    a, b = p > 0, q > 0
    return float(.5 * np.sum(p[a] * np.log(p[a] / m[a]))
                 + .5 * np.sum(q[b] * np.log(q[b] / m[b])))


def margin(p, original_class):
    return float(p[original_class] - np.max(np.delete(p, original_class)))


def event_values(p, q, original_class):
    if p is None or q is None:
        return dict(metric_available=0, flip=NA, C_clean=float(p.max()) if p is not None else NA,
            C_perturbed=NA, delta_C=NA, P_clean=float(p[original_class]) if p is not None else NA,
            P_perturbed=NA, delta_P=NA, JS=NA,
            clean_margin=margin(p, original_class) if p is not None else NA,
            perturbed_margin=NA, confidence_response_state='NA_MISSING_PROBABILITY',
            delta_C_bin=NA, delta_P_bin=NA, delta_C_sign=NA, delta_P_sign=NA,
            delta_C_strict_state=NA, delta_P_strict_state=NA,
            perturbed_winner=NA, winner_is_new=NA, winner_probability_clean=NA,
            winner_probability_perturbed=NA, winner_probability_delta=NA,
            winner_probability_delta_sign=NA, perturbed_winner_margin=NA,
            winner_lead_over_original_class=NA, delta_C_minus_delta_P_identity_error=NA)
    cc, cq = float(p.max()), float(q.max())
    pp, pq = float(p[original_class]), float(q[original_class])
    dc, dp = cq-cc, pq-pp
    winner = int(q.argmax())
    flip = int(winner != original_class)
    state = ('FLIP' if flip else 'NO_FLIP') + '_C_' + response_bin(dc)
    difference = float(q[winner]-p[winner])
    result = dict(metric_available=1, flip=flip, C_clean=cc, C_perturbed=cq, delta_C=dc,
        P_clean=pp, P_perturbed=pq, delta_P=dp, JS=js(p,q), clean_margin=margin(p,original_class),
        perturbed_margin=margin(q,original_class), confidence_response_state=state,
        delta_C_bin=response_bin(dc), delta_P_bin=response_bin(dp),
        delta_C_sign=sign(dc), delta_P_sign=sign(dp),
        delta_C_strict_state={-1:'DECREASE',0:'EXACT_ZERO',1:'INCREASE'}[sign(dc)],
        delta_P_strict_state={-1:'DECREASE',0:'EXACT_ZERO',1:'INCREASE'}[sign(dp)],
        perturbed_winner=winner, winner_is_new=flip,
        winner_probability_clean=float(p[winner]), winner_probability_perturbed=float(q[winner]),
        winner_probability_delta=difference, winner_probability_delta_sign=sign(difference),
        perturbed_winner_margin=margin(q,winner), winner_lead_over_original_class=cq-pq,
        delta_C_minus_delta_P_identity_error=(dc-dp)-(cq-pq))
    if pp != cc or abs(result['delta_C_minus_delta_P_identity_error']) > 1e-12:
        raise AssertionError('Fixed original-class support/max-confidence identity failed')
    if flip and result['perturbed_margin'] > 1e-12:
        raise AssertionError('Flipped class has an impossible positive original-class margin')
    if not flip and result['perturbed_margin'] < -1e-12:
        raise AssertionError('Kept class has an impossible negative original-class margin')
    return result


def key(row):
    return row['sample_id'], row['variant_id']


def produce_events(run, parent):
    sources = [parent/'results/perturbed_predictions.csv',
               parent/'results/prediction_flips_and_confidence_changes.csv',
               parent/'results/matched_sample_manifest.csv']
    before = {str(path): sha(path) for path in sources}
    predictions, prior_flips, samples = [read_csv(path) for path in sources]
    by_sample = {row['sample_id']: row for row in samples}
    flips = {key(row): row for row in prior_flips}
    expected = {(sid, v) for sid in by_sample for variants in FAMILIES.values() for v in variants}
    if len(samples)!=406 or len(by_sample)!=406 or len(predictions)!=1624 or len(flips)!=1624:
        raise ValueError('Fixed source counts changed')
    if len({key(row) for row in predictions})!=1624 or {key(row) for row in predictions}!=expected or set(flips)!=expected:
        raise ValueError('Missing, duplicated or unexpected fixed events')
    events = []
    missing = []
    for row in predictions:
        sample = by_sample[row['sample_id']]
        prior = flips[key(row)]
        for field in sample:
            if row.get(field) != sample[field]:
                raise ValueError('Event original metadata differs from frozen sample: '+field)
        for field in ('status','original_correct','original_confidence','label_review_status','automatic_qc_pass'):
            if row[field] != prior[field]:
                raise ValueError('Parent source tables conflict: '+field)
        if row['modality']!='text' or row['variant_id'] not in FAMILIES[row['perturbation_type']]:
            raise ValueError('Only the two fixed text families are permitted')
        p, q = vector(row['original_probabilities']), vector(row['perturbed_probabilities'])
        original = int(row['original_prediction'])
        if p is not None:
            if original != int(p.argmax()):raise ValueError('Original class is not complete-probability argmax')
            scalar_equal(row['original_confidence'],float(p.max()),'original_confidence')
        correct = int(row['original_correct'])
        if correct not in (0,1) or correct != int(original==int(row['true_label'])):
            raise ValueError('Original correctness metadata conflicts')
        metrics = event_values(p,q,original)
        if metrics['metric_available']:
            if row['status']!='SUCCESS':raise ValueError('Failed event unexpectedly has usable perturbed probabilities')
            if int(row['perturbed_prediction'])!=int(q.argmax()):raise ValueError('Perturbed argmax conflicts')
            for field, value in [('perturbed_max_confidence',float(q.max())),
                ('perturbed_original_class_probability',float(q[original])),
                ('perturbed_original_label_probability',float(q[int(row['true_label'])]))]:
                scalar_equal(row[field],value,field)
            if truth(prior['prediction_flip']) != bool(metrics['flip']):raise ValueError('Saved flip differs')
            scalar_equal(prior['max_confidence_change'],metrics['delta_C'],'saved max-confidence change')
            scalar_equal(prior['original_class_probability_change'],metrics['delta_P'],'saved original-class support change')
            c2w = int(correct==1 and metrics['flip']==1)
            w2c = int(correct==0 and int(q.argmax())==int(row['true_label']))
            if truth(prior['correct_to_wrong']) != bool(c2w) or truth(prior['wrong_to_correct']) != bool(w2c):
                raise ValueError('Reference-label transition differs from saved result')
        else:
            missing.append(dict(sample_id=row['sample_id'],variant_id=row['variant_id'],status=row['status']))
            c2w=w2c=NA
        automatic = int(metrics['metric_available']==1 and truth(row['automatic_qc_pass']))
        ai = int(automatic and truth(row['label_preserving_main_eligible']) and row['label_review_status']=='PASS')
        event = dict(row,original_correct=correct,original_prediction=original,
            **metrics,input_qc_scope_available=automatic,ai_version_eligible=ai,
            saved_scalar_validation='VERIFIED_FROM_COMPLETE_7D_PROBABILITIES' if metrics['metric_available'] else 'NOT_VERIFIABLE_MISSING_PROBABILITY',
            reference_correct_to_wrong=c2w,reference_wrong_to_correct=w2c,
            original_label_transition_interpretation='ORIGINAL_LABEL_REFERENCE_NOT_CONFIRMED_POST_EDIT_CORRECTNESS',
            human_review_scope='NA_NO_ACTUAL_HUMAN_COMPLETION',
            original_prediction_source=str(sources[0]),parent_flip_source=str(sources[1]))
        events.append(event)
    grouped=defaultdict(list)
    for row in events:grouped[row['sample_id'],row['perturbation_type']].append(row)
    for group in grouped.values():
        if len(group)!=2:raise ValueError('A fixed family lost a version')
        for row in group:
            row['qc_complete_family_eligible']=int(all(x['input_qc_scope_available'] for x in group))
            row['ai_complete_family_eligible']=int(all(x['ai_version_eligible'] for x in group))
    after={str(path):sha(path) for path in sources}
    if before!=after:raise RuntimeError('Frozen parent sources changed')
    write_csv(run/'EVENT_RESPONSES.csv',events)
    return dict(status='ALL_FIXED_EVENTS_EXPORTED',events=1624,metric_available=sum(x['metric_available'] for x in events),
        missing_probability_events=len(missing),status_counts=dict(Counter(x['status'] for x in events)),
        family_counts=dict(Counter(x['perturbation_type'] for x in events)),
        source_sha256_before=before,source_sha256_after=after,source_files_unchanged=before==after,
        output_sha256=sha(run/'EVENT_RESPONSES.csv'),missing_probability_records=missing)


SCOPES = ('ALL_PLANNED','INPUT_QC_AVAILABLE','AI_VERSION_ELIGIBLE',
          'QC_COMPLETE_FAMILY','AI_COMPLETE_FAMILY')
GROUPS = ('ALL','ORIGINAL_CORRECT','ORIGINAL_ERROR')
STATES = [(prefix+'_C_'+state) for prefix in ('NO_FLIP','FLIP')
          for state in ('DECREASE_GT_0.02','WITHIN_0.02','INCREASE_GT_0.02')]
STRICT_STATES = [(prefix+'_C_STRICT_'+state) for prefix in ('NO_FLIP','FLIP')
                 for state in ('DECREASE','EXACT_ZERO','INCREASE')]
NUMERIC = ('C_clean','C_perturbed','delta_C','P_clean','P_perturbed','delta_P','JS',
           'clean_margin','perturbed_margin','winner_probability_clean',
           'winner_probability_perturbed','winner_probability_delta',
           'delta_C_minus_delta_P_identity_error')


def scope_member(row, scope):
    field = {'INPUT_QC_AVAILABLE':'input_qc_scope_available',
             'AI_VERSION_ELIGIBLE':'ai_version_eligible',
             'QC_COMPLETE_FAMILY':'qc_complete_family_eligible',
             'AI_COMPLETE_FAMILY':'ai_complete_family_eligible'}.get(scope)
    return field is None or row[field]==1


def correct_group(row, group):
    return group=='ALL' or row['original_correct']==int(group=='ORIGINAL_CORRECT')


def collect_groups(events):
    for family in FAMILIES:
        family_events=[row for row in events if row['perturbation_type']==family]
        for scope in SCOPES:
            for group in GROUPS:
                planned=[row for row in family_events if correct_group(row,group)]
                selected=[row for row in planned if scope_member(row,scope)]
                yield family,scope,group,planned,selected


def units(planned, selected):
    yield 'EVENT_WEIGHTED','ALL',planned,selected
    yield 'EQUAL_DIALOGUE','ALL',planned,selected
    dids=sorted({row['dialogue_id'] for row in selected})
    for did in dids:
        yield 'PER_DIALOGUE',did,[r for r in planned if r['dialogue_id']==did],[r for r in selected if r['dialogue_id']==did]


def rate(selected, predicate, aggregation):
    if not selected:return NA
    if aggregation!='EQUAL_DIALOGUE':
        return sum(bool(predicate(row)) for row in selected)/len(selected)
    groups=defaultdict(list)
    for row in selected:groups[row['dialogue_id']].append(row)
    return float(np.mean([sum(bool(predicate(row)) for row in group)/len(group)
                          for group in groups.values()]))


def base_row(family, scope, group, aggregation, did, planned, selected):
    valid=[row for row in selected if row['metric_available']==1]
    return dict(perturbation_type=family,analysis_scope=scope,original_correctness_group=group,
        aggregation=aggregation,dialogue_id=did,planned_reference_events=len(planned),
        selected_events=len(selected),metric_available_events=len(valid),
        missing_probability_events=len(selected)-len(valid),
        unique_targets=len({r['sample_id'] for r in selected}),
        selected_dialogues=len({r['dialogue_id'] for r in selected}),
        metric_available_dialogues=len({r['dialogue_id'] for r in valid}),
        scope_status='DESCRIPTIVE_CONDITIONAL_ON_FIXED_SCOPE',
        independence='Repeated versions/targets in a Dialogue are not independent',
        human_review_scope='NA_NO_ACTUAL_HUMAN_COMPLETION')


def strict_state(row):
    if not row['metric_available']:return 'NA_MISSING_PROBABILITY'
    return ('FLIP' if row['flip'] else 'NO_FLIP')+'_C_STRICT_'+row['delta_C_strict_state']


def state_counts(selected, state):
    missing=state=='NA_MISSING_PROBABILITY'
    return dict(original_correct_count=sum(r['original_correct']==1 for r in selected),
        original_wrong_count=sum(r['original_correct']==0 for r in selected),
        reference_correct_to_wrong_count=NA if missing else sum(str(r['reference_correct_to_wrong'])=='1' for r in selected),
        reference_wrong_to_correct_count=NA if missing else sum(str(r['reference_wrong_to_correct'])=='1' for r in selected),
        state_count_unit='EVENTS_NOT_UNIQUE_TARGETS',
        reference_transition_interpretation='Original-label reference only; not confirmed post-edit correctness')


def cross_table(events):
    result=[]
    for family,scope,group,planned,selected in collect_groups(events):
        for aggregation,did,reference,subset in units(planned,selected):
            valid=[r for r in subset if r['metric_available']==1]
            # Strict sign is available at all resolutions in EVENT_RESPONSES;
            # avoid duplicating its per-Dialogue cross table with the threshold table.
            definitions=[('THRESHOLD_0.02',STATES,lambda r:r['confidence_response_state'])]
            if aggregation!='PER_DIALOGUE':
                definitions.append(('STRICT_SIGN_NO_TOLERANCE',STRICT_STATES,strict_state))
            for definition,states,function in definitions:
                for state in [*states,'NA_MISSING_PROBABILITY']:
                    condition=lambda r,state=state:function(r)==state
                    row=base_row(family,scope,group,aggregation,did,reference,subset)
                    row.update(confidence_definition=definition,confidence_response_state=state,
                        state_count=sum(condition(r) for r in subset),
                        fraction_all_selected=rate(subset,condition,aggregation),
                        fraction_metric_available=rate(valid,condition,aggregation) if state!='NA_MISSING_PROBABILITY' else NA,
                        fraction_unit='Mean within-Dialogue fractions over represented Dialogues' if aggregation=='EQUAL_DIALOGUE' else 'Fraction of events',
                        threshold=THRESHOLD,zero_band_inclusive=True)
                    row.update(state_counts([r for r in subset if condition(r)],state))
                    result.append(row)
    return result


def describe_values(selected, field, aggregation):
    if not selected:
        return dict(mean=NA,mean_abs=NA,std_population=NA,q00=NA,q01=NA,q05=NA,q10=NA,q25=NA,
                    q50=NA,q75=NA,q90=NA,q95=NA,q99=NA,q100=NA,
                    fraction_strict_decrease=NA,fraction_exact_zero=NA,fraction_strict_increase=NA,
                    fraction_decrease_gt_0_02=NA,fraction_within_0_02=NA,fraction_increase_gt_0_02=NA,
                    **delta_bins(selected,field,aggregation))
    if aggregation=='EQUAL_DIALOGUE':
        grouped=defaultdict(list)
        for row in selected:grouped[row['dialogue_id']].append(float(row[field]))
        values=np.array([np.mean(v) for v in grouped.values()])
        absolute=float(np.mean([np.mean(np.abs(v)) for v in grouped.values()]))
    else:
        values=np.array([float(r[field]) for r in selected]);absolute=float(np.abs(values).mean())
    quantiles=np.quantile(values,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1],method='linear')
    result=dict(mean=float(values.mean()),mean_abs=absolute,std_population=float(values.std(ddof=0)),
        **dict(zip(['q00','q01','q05','q10','q25','q50','q75','q90','q95','q99','q100'],map(float,quantiles))))
    predicates={'fraction_strict_decrease':lambda x:x<0,'fraction_exact_zero':lambda x:x==0,
        'fraction_strict_increase':lambda x:x>0,'fraction_decrease_gt_0_02':lambda x:x<-THRESHOLD,
        'fraction_within_0_02':lambda x:abs(x)<=THRESHOLD,'fraction_increase_gt_0_02':lambda x:x>THRESHOLD}
    for name,predicate in predicates.items():
        result[name]=rate(selected,lambda r,p=predicate:p(float(r[field])),aggregation)
    result.update(delta_bins(selected,field,aggregation))
    return result


def distribution_table(events):
    result=[]
    fields=('delta_C','delta_P','JS','winner_probability_delta','clean_margin','perturbed_margin','fixed_original_margin_change')
    for family,scope,group,planned,selected in collect_groups(events):
        for aggregation,did,reference,subset in units(planned,selected):
            valid=[r for r in subset if r['metric_available']==1]
            for flipgroup in ('ALL','FLIP','NO_FLIP'):
                take=[r for r in valid if flipgroup=='ALL' or r['flip']==int(flipgroup=='FLIP')]
                for field in fields:
                    row=base_row(family,scope,group,aggregation,did,reference,subset)
                    row.update(flip_group=flipgroup,metric=field,distribution_events=len(take),
                        distribution_dialogues=len({r['dialogue_id'] for r in take}),
                        distribution_unit='Dialogue mean of metric; quantiles across supported Dialogue means' if aggregation=='EQUAL_DIALOGUE' else 'Individual event values',
                        sign_fractions_unit='Mean of within-Dialogue event sign fractions' if aggregation=='EQUAL_DIALOGUE' else 'Event sign fractions',
                        quantile_method='numpy linear',**describe_values(take,field,aggregation))
                    result.append(row)
    return result


RELATIONSHIPS = {
    'ALL_METRIC_AVAILABLE':('All selected events with complete probabilities',lambda r:True),
    'FLIP':('Perturbed argmax differs from original argmax',lambda r:r['flip']==1),
    'NO_FLIP':('Perturbed argmax retains original argmax',lambda r:r['flip']==0),
    'FLIP_C_INCREASE_STRICT':('flip and delta_C>0',lambda r:r['flip']==1 and r['delta_C']>0),
    'FLIP_C_DECREASE_STRICT':('flip and delta_C<0',lambda r:r['flip']==1 and r['delta_C']<0),
    'FLIP_C_WITHIN_0.02':('flip and abs(delta_C)<=0.02',lambda r:r['flip']==1 and abs(r['delta_C'])<=THRESHOLD),
    'FLIP_C_INCREASE_GT_0.02':('flip and delta_C>0.02',lambda r:r['flip']==1 and r['delta_C']>THRESHOLD),
    'FLIP_C_DECREASE_GT_0.02':('flip and delta_C<-0.02',lambda r:r['flip']==1 and r['delta_C']<-THRESHOLD),
    'NO_FLIP_C_INCREASE_STRICT':('no flip and delta_C>0',lambda r:r['flip']==0 and r['delta_C']>0),
    'NO_FLIP_C_DECREASE_STRICT':('no flip and delta_C<0',lambda r:r['flip']==0 and r['delta_C']<0),
    'C_INCREASE_P_DECREASE_STRICT':('delta_C>0 and delta_P<0; max confidence can rise while original class loses support',lambda r:r['delta_C']>0 and r['delta_P']<0),
    'FLIP_P_INCREASE_STRICT':('flip and delta_P>0',lambda r:r['flip']==1 and r['delta_P']>0),
    'FLIP_P_DECREASE_STRICT':('flip and delta_P<0',lambda r:r['flip']==1 and r['delta_P']<0),
    'FLIP_NEW_WINNER_PROBABILITY_INCREASE_STRICT':('flip and q[new_winner]-p[new_winner]>0; same class compared before and after',lambda r:r['flip']==1 and r['winner_probability_delta']>0),
    'FLIP_NEW_WINNER_PROBABILITY_DECREASE_STRICT':('flip and q[new_winner]-p[new_winner]<0; original class may fall faster',lambda r:r['flip']==1 and r['winner_probability_delta']<0),
    'FLIP_NEW_WINNER_PROBABILITY_EXACT_ZERO':('flip and q[new_winner]-p[new_winner]==0',lambda r:r['flip']==1 and r['winner_probability_delta']==0),
    'P_DECREASE_GT_0.02':('delta_P<-0.02',lambda r:r['delta_P']<-THRESHOLD),
    'ORIGINAL_MARGIN_NEGATIVE':('q[original]-max_other(q)<0',lambda r:r['perturbed_margin']<0),
}


def support_table(events):
    result=[]
    for family,scope,group,planned,selected in collect_groups(events):
        for aggregation,did,reference,subset in units(planned,selected):
            valid=[r for r in subset if r['metric_available']==1]
            flips=[r for r in valid if r['flip']==1]
            for name,(definition,predicate) in RELATIONSHIPS.items():
                take=[r for r in valid if predicate(r)]
                row=base_row(family,scope,group,aggregation,did,reference,subset)
                row.update(relationship=name,definition=definition,condition_events=len(take),
                    condition_dialogues=len({r['dialogue_id'] for r in take}),
                    fraction_all_metric_available=rate(valid,predicate,aggregation),
                    fraction_within_flipped=rate(flips,predicate,aggregation) if name.startswith('FLIP') else NA,
                    denominator='Same family, fixed scope, original-correctness group and aggregation; complete-probability events',
                    reference_label_transitions='Reference to original label only; no verified post-edit correctness')
                for field in ('C_clean','C_perturbed','delta_C','P_clean','P_perturbed','delta_P','JS',
                              'clean_margin','perturbed_margin','winner_probability_clean',
                              'winner_probability_perturbed','winner_probability_delta'):
                    if not take:
                        value=NA
                    elif aggregation=='EQUAL_DIALOGUE':
                        groups=defaultdict(list)
                        for event in take:groups[event['dialogue_id']].append(event[field])
                        value=float(np.mean([np.mean(v) for v in groups.values()]))
                    else:
                        value=float(np.mean([event[field] for event in take]))
                    row['mean_'+field]=value
                row['max_abs_algebra_identity_error']=max((abs(r['delta_C_minus_delta_P_identity_error']) for r in take),default=NA)
                result.append(row)
    return result


def human_placeholder(table_kind):
    rows=[]
    for family in FAMILIES:
        for group in GROUPS:
            rows.append(dict(perturbation_type=family,analysis_scope='HUMAN_CONFIRMED',
                original_correctness_group=group,aggregation='UNAVAILABLE',dialogue_id='ALL',
                selected_events=NA,metric_available_events=NA,scope_status='NA_NO_ACTUAL_HUMAN_COMPLETION',
                **({'confidence_response_state':NA,'state_count':NA,'fraction_all_selected':NA}
                   if table_kind=='cross' else {'metric':NA,'mean':NA,'q50':NA}
                   if table_kind=='distribution' else {'relationship':NA,'condition_events':NA,'fraction_all_metric_available':NA})))
    return rows


def produce_summaries(run, parent):
    metadata=json.loads((run/'logs/events.json').read_text())
    event_path=run/'EVENT_RESPONSES.csv'
    if sha(event_path)!=metadata['output_sha256']:raise ValueError('Frozen event table changed')
    sources={path:sha(path) for path in metadata['source_sha256_before']}
    if sources!=metadata['source_sha256_before']:raise ValueError('Parent source changed since event export')
    events=read_csv(event_path)
    for row in events:
        for field in ('original_correct','metric_available','input_qc_scope_available','ai_version_eligible',
                      'qc_complete_family_eligible','ai_complete_family_eligible'):
            row[field]=int(row[field])
        if row['metric_available']:
            row['flip']=int(row['flip'])
            for field in NUMERIC:row[field]=float(row[field])
            row['fixed_original_margin_change']=row['perturbed_margin']-row['clean_margin']
    cross=cross_table(events)+human_placeholder('cross')
    distribution=distribution_table(events)+human_placeholder('distribution')
    support=support_table(events)+human_placeholder('support')
    outputs={'FLIP_CONFIDENCE_CROSS_TAB.csv':cross,
             'CONFIDENCE_CHANGE_DISTRIBUTION.csv':distribution,
             'ORIGINAL_CLASS_SUPPORT_ANALYSIS.csv':support}
    for filename,rows in outputs.items():write_csv(run/filename,rows)
    if sha(event_path)!=metadata['output_sha256'] or {path:sha(path) for path in sources}!=sources:
        raise RuntimeError('Events or parent source changed during summaries')
    core={}
    for family in FAMILIES:
        valid=[r for r in events if r['perturbation_type']==family and r['metric_available']]
        flipped=[r for r in valid if r['flip']]
        core[family]=dict(available_events=len(valid),flipped_events=len(flipped),
            flip_rate=len(flipped)/len(valid),threshold_states=dict(Counter(r['confidence_response_state'] for r in valid)),
            strict_flip_confidence_increase=sum(r['delta_C']>0 for r in flipped),
            strict_flip_confidence_decrease=sum(r['delta_C']<0 for r in flipped),
            strict_flip_confidence_exact_zero=sum(r['delta_C']==0 for r in flipped),
            strict_C_increase_P_decrease=sum(r['delta_C']>0 and r['delta_P']<0 for r in valid),
            flipped_new_winner_probability_increases=sum(r['winner_probability_delta']>0 for r in flipped),
            flipped_new_winner_probability_decreases=sum(r['winner_probability_delta']<0 for r in flipped),
            strict_no_flip_probability_change_identity_max_abs=max((abs(r['delta_C']-r['delta_P']) for r in valid if not r['flip']),default=NA))
    metadata['summaries']=dict(completed_at_utc=now(),code_sha256=sha(__file__),
        output_rows={name:len(rows) for name,rows in outputs.items()},
        output_sha256={name:sha(run/name) for name in outputs},
        core_metric_available_scope=core,source_sha256_before=sources,
        source_sha256_after={path:sha(path) for path in sources},
        event_table_unchanged=True,all_results_descriptive_not_independent_event_tests=True,
        equal_dialogue_distribution='Quantiles across within-Dialogue event means; sign fractions averaged within then across supported Dialogues',
        human_confirmed_scope='NA_NO_ACTUAL_HUMAN_COMPLETION')
    temp=run/'logs/events.json.tmp'
    write_json(temp,metadata);temp.replace(run/'logs/events.json')
    return metadata['summaries']


def amend_required_columns(run):
    """Append the originally required seven bins/counts; keep every old cell.

    This bounded schema correction uses fixed original-value cutoffs, not a
    result-selected analysis. All prior output hashes remain in the run log.
    """
    log_path=run/'logs/events.json';metadata=json.loads(log_path.read_text())
    if 'required_column_amendment' in metadata:
        raise FileExistsError('Required schema amendment already completed')
    if sha(run/'EVENT_RESPONSES.csv')!=metadata['output_sha256']:
        raise ValueError('Frozen event table changed')
    events=read_csv(run/'EVENT_RESPONSES.csv')
    for r in events:
        for field in ('original_correct','metric_available','input_qc_scope_available','ai_version_eligible',
                      'qc_complete_family_eligible','ai_complete_family_eligible'):
            r[field]=int(r[field])
        if r['metric_available']:
            r['flip']=int(r['flip'])
            for field in NUMERIC:r[field]=float(r[field])
    previous=dict(metadata['summaries']['output_sha256'])
    for filename,digest in previous.items():
        if sha(run/filename)!=digest:raise ValueError('Existing summary changed before schema amendment')
    old_cross=read_csv(run/'FLIP_CONFIDENCE_CROSS_TAB.csv')
    old_distribution=read_csv(run/'CONFIDENCE_CHANGE_DISTRIBUTION.csv')
    cache={}
    def subset(row, available=False):
        key=tuple(row[k] for k in ('perturbation_type','analysis_scope','original_correctness_group','dialogue_id'))
        if key not in cache:
            cache[key]=[r for r in events if r['perturbation_type']==row['perturbation_type']
                and scope_member(r,row['analysis_scope']) and correct_group(r,row['original_correctness_group'])
                and (row['dialogue_id']=='ALL' or r['dialogue_id']==row['dialogue_id'])]
        return [r for r in cache[key] if not available or r['metric_available']]
    cross=[]
    for row in old_cross:
        if row['analysis_scope']=='HUMAN_CONFIRMED':
            extra={k:NA for k in ('original_correct_count','original_wrong_count','reference_correct_to_wrong_count','reference_wrong_to_correct_count')}
        else:
            function=strict_state if row['confidence_definition']=='STRICT_SIGN_NO_TOLERANCE' else lambda r:r['confidence_response_state']
            selected=[r for r in subset(row) if function(r)==row['confidence_response_state']]
            extra=state_counts(selected,row['confidence_response_state'])
        cross.append(dict(row,**extra))
    distribution=[]
    for row in old_distribution:
        if row['analysis_scope']=='HUMAN_CONFIRMED':
            extra={f'bin_{i}_{kind}':NA for i in range(1,8) for kind in ('count','fraction')}
        else:
            selected=[r for r in subset(row,available=True) if row['flip_group']=='ALL' or r['flip']==int(row['flip_group']=='FLIP')]
            extra=delta_bins(selected,row['metric'],row['aggregation'])
        distribution.append(dict(row,**extra))
    preserved_cells=0
    for old,new in [(old_cross,cross),(old_distribution,distribution)]:
        for before,after in zip(old,new):
            if any(after[k]!=v for k,v in before.items()):raise AssertionError('An existing summary cell changed')
            preserved_cells+=len(before)
    updates={'FLIP_CONFIDENCE_CROSS_TAB.csv':cross,'CONFIDENCE_CHANGE_DISTRIBUTION.csv':distribution}
    for filename,rows in updates.items():write_csv(run/(filename+'.schema_amendment.tmp'),rows)
    if any(sha(p)!=digest for p,digest in metadata['source_sha256_before'].items()):
        raise RuntimeError('Parent source changed during amendment')
    for filename in updates:(run/(filename+'.schema_amendment.tmp')).replace(run/filename)
    metadata['required_column_amendment']=dict(completed_at_utc=now(),
        reason='Complete user-required seven fixed change bins and per-state original/reference transition counts',
        thresholds_selected_from_results=False,previous_output_sha256=previous,
        all_previous_cells_preserved=True,previous_cells_preserved=preserved_cells,
        event_table_unchanged=True,original_parent_sources_unchanged=True,
        seven_bins=['(-inf,-0.20)','[-0.20,-0.10)','[-0.10,-0.02)','[-0.02,0.02]','(0.02,0.10]','(0.10,0.20]','(0.20,inf)'],
        seven_bin_count_columns=[f'bin_{i}_count' for i in range(1,8)],
        seven_bin_fraction_columns=[f'bin_{i}_fraction' for i in range(1,8)],
        bins_apply_to='delta_C/delta_P individual unrounded event values; not Dialogue means',
        cross_counts_are='Events, not unique targets; original-label transitions remain references',
        final_code_sha256=sha(__file__))
    metadata['summaries']['output_sha256']={name:sha(run/name) for name in previous}
    metadata['summaries']['code_sha256']=sha(__file__)
    temp=log_path.with_suffix('.json.amendment.tmp');write_json(temp,metadata);temp.replace(log_path)
    return metadata['required_column_amendment']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    parser.add_argument('--parent-dir',type=Path,default=PARENT)
    parser.add_argument('--stage',choices=['events','summaries','amend-required-columns'],default='events')
    args=parser.parse_args()
    run,parent=args.run_dir.resolve(),args.parent_dir.resolve()
    if args.stage=='amend-required-columns':
        print(json.dumps(amend_required_columns(run),indent=2))
        return
    if args.stage=='summaries':
        print(json.dumps(produce_summaries(run,parent),indent=2))
        return
    metadata=produce_events(run,parent)
    metadata.update(created_at_utc=now(),stage=args.stage,code_sha256=sha(__file__),numpy=np.__version__,
        threshold=THRESHOLD,threshold_boundary='abs(delta)<=0.02 inclusive; tails strict',
        saved_scalar_tolerance=dict(atol=SCALAR_ATOL,rtol=SCALAR_RTOL),
        original_label_reference_only=True,human_review_available=False,no_models_loaded=True,
        no_training=True,no_new_seeds=True,no_Test=True)
    write_json(run/'logs/events.json',metadata)
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['missing_probability_records','source_sha256_before','source_sha256_after']},indent=2))


if __name__=='__main__':main()
