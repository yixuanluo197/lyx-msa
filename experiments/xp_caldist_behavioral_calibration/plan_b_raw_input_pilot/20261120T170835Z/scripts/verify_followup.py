#!/usr/bin/env python3
"""Independent actual-artifact verification; no imports of the analysis engine."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RUN=Path(__file__).resolve().parents[1]
ANALYSIS=RUN/'results/matched_analysis'


def rows(path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    checks=Counter()
    def check(condition,description):
        if not bool(condition):raise AssertionError(description)
        checks[description]+=1
    for name in ('configs/COHORT_LOCK.json','provenance/TEXT_INPUT_LOCK.json'):
        obj=json.loads((RUN/name).read_text())
        for category in ('sources_sha256','outputs_sha256'):
            for path,expected in obj[category].items():check(sha(path)==expected,'frozen_source_output_hash')
    p=rows(RUN/'results/perturbed_predictions.csv')
    check(len(p)==1624,'all_fixed_conditions_retained')
    check(len({(r['sample_id'],r['variant_id']) for r in p})==1624,'unique_conditions')
    controls=rows(RUN/'results/clean_controls.csv')
    check(len(controls)==406,'all_clean_controls_retained')
    for r in controls:
        for field in ('clean_feature_allclose','clean_probability_allclose','original_probability_allclose','clean_prediction_unchanged'):
            check(r[field]=='True','actual_clean_negative_control')
    for name in ('encode','infer'):
        state=json.loads((RUN/f'provenance/text_engine_{name}.json').read_text())
        check(state['encoder_before']==state['encoder_after'],'encoder_frozen_state')
        check(not state['test_accessed'],'no_test_access')
        if name=='infer':
            check(state['model_before']==state['model_after'],'cmerc_frozen_state')
            check(state['original_context_before']==state['original_context_after'],'original_context_bit_exact')
            check(sha(RUN/'results/perturbed_predictions.csv')==state['output_sha256'],'inference_csv_output_hash')
            check(sha(RUN/'local_only/dialogue_predictions.pt')==state['dialogue_artifact_sha256'],'full_dialogue_prediction_artifact_hash')
            check(sha(RUN/'local_only/encoder_features.pt')==state['encoder_feature_sha256'],'inference_feature_cache_hash')
        else:
            check(sha(RUN/'local_only/encoder_features.pt')==state['feature_file_sha256'],'encoder_output_cache_hash')
        for path,expected in state['bindings'].items():check(sha(path)==expected,'engine_bound_input_hash')
    blind=rows(RUN/'TEXT_REVIEW_PACKAGE.csv')
    check(len(blind)==1932,'complete_old_plus_new_review_package')
    forbidden=('true_label','confidence','probabilit','correct','pair_id','review_notes','ai_review')
    check(not any(any(x in k.lower() for x in forbidden) for k in blind[0]),'blind_columns_hide_outcomes_and_ai')
    for r in blind:
        check(r['human_consensus_status']=='PENDING','no_fabricated_human_reviews')
        check(all(not v for k,v in r.items() if k.startswith(('reviewer_1_','reviewer_2_'))),'human_fields_blank')
        for turn in json.loads(r['full_original_dialogue_context']):
            check(set(turn)=={'utterance_index','speaker_id','text'},'blind_context_has_no_labels_or_predictions')
    original=jsonl(RUN.parent/'20261120T143131Z/local_only/pilot/MELD/text_reviewed.jsonl')
    old={(r['sample_id'],r['variant_id']):r for r in original}
    matched=jsonl(RUN/'local_only/matched_text_reviewed.jsonl')
    check(sum((r['sample_id'],r['variant_id']) in old for r in matched)==172,'historical_overlap_172')
    for r in matched:
        key=(r['sample_id'],r['variant_id'])
        if key in old:check(r==old[key],'historical_failed_uncertain_records_unchanged')
    targets={(r['sample_id'],r['perturbation_type']):r for r in rows(ANALYSIS/'target_behavior.csv')}
    analysis_audit=json.loads((ANALYSIS/'analysis_verification.json').read_text())
    check(analysis_audit['predictions_sha256']==sha(RUN/'results/perturbed_predictions.csv'),'analysis_prediction_input_hash')
    for filename,expected in analysis_audit['outputs_sha256'].items():
        check(sha(ANALYSIS/filename)==expected,'analysis_output_hash')
    grouped=defaultdict(list)
    for r in p:grouped[r['sample_id'],r['perturbation_type']].append(r)
    for key,conditions in grouped.items():
        target=targets[key];check(len(conditions)==2,'family_k_fixed_two')
        check(target['human_confirmed_eligible']=='False','human_analysis_unavailable')
        if any(r['status']!='SUCCESS' for r in conditions):
            check(not target['mu'],'failed_k_not_silently_reduced');continue
        clean=np.asarray(json.loads(conditions[0]['original_probabilities']),dtype=np.float64)
        q=np.array([json.loads(r['perturbed_probabilities']) for r in conditions],dtype=np.float64)
        flip=(q.argmax(1)!=clean.argmax()).astype(float)
        maxq=q.max(1)
        js=[]
        for v in q:
            m=(clean+v)/2
            a=clean>0;b=v>0
            js.append(.5*np.sum(clean[a]*np.log(clean[a]/m[a]))+.5*np.sum(v[b]*np.log(v[b]/m[b])))
        truth=dict(mu=np.mean(flip*maxq),delta=abs(clean.max()-np.mean(maxq)),js_mean=np.mean(js),flip_rate=np.mean(flip))
        for k,v in truth.items():check(np.isclose(float(target[k]),v,rtol=1e-12,atol=1e-14),'independent_probability_behavior_formula')
    pairrows=rows(ANALYSIS/'pair_behavior_and_eligibility.csv')
    frozen_pairs={r['pair_id']:r for r in rows(RUN/'results/matched_pair_manifest.csv')}
    gates={'AUTOMATIC_INPUT_QC_DIAGNOSTIC':'automatic_qc_eligible',
           'AI_REVIEWED_EXPLORATORY':'ai_reviewed_exploratory_eligible',
           'HUMAN_CONFIRMED':'human_confirmed_eligible'}
    def rank(value):return float(value>0)+.5*float(value==0)
    for row in pairrows:
        fixed=frozen_pairs[row['pair_id']]
        for field in ('dialogue_id','correct_sample_id','error_sample_id'):
            check(row[field]==fixed[field],'independent_pair_endpoint_identity')
        c=targets[(fixed['correct_sample_id'],row['perturbation_type'])]
        e=targets[(fixed['error_sample_id'],row['perturbation_type'])]
        eligible=c[gates[row['analysis_scope']]]=='True' and e[gates[row['analysis_scope']]]=='True'
        check((row['eligible']=='True')==eligible,'independent_pair_complete_scope_eligibility')
        confidence_rank=rank(float(c['original_confidence'])-float(e['original_confidence']))
        check(float(row['confidence_baseline_concordance'])==confidence_rank,'independent_samepair_confidence_baseline')
        for metric in ('mu','delta','js_mean','flip_rate'):
            for side,target in [('correct',c),('error',e)]:
                if target[metric]:
                    check(float(row[f'{metric}_{side}'])==float(target[metric]),'independent_pair_behavior_endpoint')
                else:check(not row[f'{metric}_{side}'],'missing_endpoint_preserved')
            if eligible:
                difference=float(e[metric])-float(c[metric])
                check(np.isclose(float(row[metric+'_difference']),difference,atol=1e-14,rtol=1e-12),'independent_error_minus_correct_direction')
                check(float(row[metric+'_concordance'])==rank(difference),'independent_pair_behavior_concordance')
                check(float(row[metric+'_concordance_minus_confidence'])==rank(difference)-confidence_rank,'independent_conditional_ranking_difference')
            else:check(not row[metric+'_difference'],'ineligible_pairs_have_no_contrast')
    drawfile=np.load(ANALYSIS/'bootstrap_draw_indices.npz')
    # Stored names are explicit; demand the actual master draw array and ID order.
    draws=next(v for v in drawfile.values() if v.shape==(2000,80))
    dialogue_array=next(v for v in drawfile.values() if v.shape==(80,))
    dialogues=[str(x) for x in dialogue_array]
    check(np.array_equal(draws,np.random.RandomState(20261120).randint(0,80,(2000,80))),'shared_2000_dialogue_draws')
    # Count matrix is an independent bootstrap implementation, preserving duplicate clusters.
    weights=np.zeros((2000,80),dtype=np.int64)
    for i,draw in enumerate(draws):weights[i]=np.bincount(draw,minlength=80)
    pos={d:i for i,d in enumerate(dialogues)}
    replicates=np.load(ANALYSIS/'bootstrap_replicates.npz')
    contrasts=rows(ANALYSIS/'paired_metric_contrasts.csv')
    ranks=rows(ANALYSIS/'conditional_pair_ranking.csv')
    def selection(condition):
        take=[r for r in pairrows if r['perturbation_type']==condition['perturbation_type'] and r['analysis_scope']==condition['analysis_scope'] and r['eligible']=='True']
        cal=condition['confidence_caliper']
        if cal!='all':
            limit=.02 if '.02' in cal or '0p02' in cal else .01
            take=[r for r in take if float(r['confidence_abs_gap'])<=limit]
        group=condition['residual_confidence_group']
        if group!='all':take=[r for r in take if r['confidence_gap_group']==group]
        return take
    def verify_stat(condition,field,estimand,value_field):
        selected=selection(condition)
        sums=np.zeros(80);counts=np.zeros(80,dtype=np.int64)
        for row in selected:
            i=pos[row['dialogue_id']];sums[i]+=float(row[field]);counts[i]+=1
        if estimand=='equal_pair':
            numerator=weights@sums;denominator=weights@counts
            point=sums.sum()/counts.sum() if counts.sum() else np.nan
        else:
            valid=counts>0;means=np.divide(sums,counts,out=np.zeros(80),where=valid)
            numerator=weights@means;denominator=weights@valid.astype(np.int64)
            point=np.mean(means[valid]) if valid.any() else np.nan
        expected=np.divide(numerator,denominator,out=np.full(2000,np.nan),where=denominator>0)
        check(np.allclose(expected,replicates[condition['replicate_key']],atol=1e-12,rtol=1e-10,equal_nan=True),'independent_cluster_bootstrap_replicates')
        if np.isfinite(point):
            check(np.isclose(point,float(condition[value_field]),atol=1e-12,rtol=1e-10),'independent_point_estimand')
            lo,hi=np.quantile(expected[np.isfinite(expected)],[.025,.975])
            check(np.allclose([lo,hi],[float(condition['ci95_low']),float(condition['ci95_high'])],atol=1e-12,rtol=1e-10),'independent_95_percentile_interval')
        else:check(not condition[value_field],'unavailable_scope_has_no_estimate')
        if condition.get('priority_two_family_js')=='True':
            good=expected[np.isfinite(expected)]
            if len(good):
                lo,hi=np.quantile(good,[.0125,.9875])
                check(np.allclose([lo,hi],[float(condition['bonferroni_ci97_5_low']),float(condition['bonferroni_ci97_5_high'])],atol=1e-12,rtol=1e-10),'independent_priority_97_5_interval')
            else:
                check(not condition['bonferroni_ci97_5_low'] and not condition['bonferroni_ci97_5_high'],'unavailable_priority_has_no_interval')
    for r in contrasts:verify_stat(r,r['metric']+'_difference',r['estimand'],'error_minus_correct')
    for r in ranks:
        field={'behavior':r['metric']+'_concordance','confidence_only':'confidence_baseline_concordance','behavior_minus_confidence':r['metric']+'_concordance_minus_confidence'}[r['ranking']]
        verify_stat(r,field,'equal_dialogue','estimate')
    verification=dict(status='PASS',checks=sum(checks.values()),check_categories=dict(checks),
        fixed_pairs=203,matched_targets=406,dialogues=80,conditions=1624,clean_controls=406,
        human_review_rows_completed=0,human_confirmed_results_available=False,
        exploratory_only=True,no_training=True,no_test=True,no_added_seeds=True,
        independent_bootstrap_contrasts=len(contrasts),independent_bootstrap_rankings=len(ranks),
        actual_status_counts=dict(Counter(r['status'] for r in p)),
        source_and_artifact_hashes_verified=True)
    with (RUN/'verification.json').open('x') as f:json.dump(verification,f,indent=2);f.write('\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
