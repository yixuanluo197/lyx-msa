#!/usr/bin/env python3
"""Task4: CSV/JSON-only historical audio diagnosis; no torch/media/model access."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'20261120T143131Z'
DATASETS=('IEMOCAP','MELD')
VARIANTS={'pitch_shift':('pitch_plus_0.5','pitch_minus_0.5','pitch_plus_1','pitch_minus_1'),
          'timbre_tilt':('timbre_light','timbre_moderate')}
METRICS=('D_pred','D_abs','cancellation','signed_delta_max_confidence','mean_abs_delta_max_confidence','JS','flip_rate',
         'feature_rmse','feature_relative_l2','feature_cosine_distance')
COMPARISON_METRICS=METRICS[:7]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def write_csv(path,rows,empty_fields=('dataset','condition','status')):
    fields=list(dict.fromkeys(k for r in rows for k in r)) or list(empty_fields)
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:'' if v is None else v for k,v in row.items()})


def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=True,allow_nan=False)+'\n')
def yes(x):return x is True or str(x).lower() in ('true','1')
def number(x):return None if x in ('',None) else float(x)


def probability(value,classes):
    p=np.asarray(json.loads(value),dtype=np.float64)
    if p.shape!=(classes,) or not np.isfinite(p).all() or (p<0).any() or (p>1+1e-6).any() or abs(p.sum()-1)>1e-5:
        raise ValueError('Invalid stored complete probability distribution')
    return p


def js(p,q):
    m=(p+q)/2
    def kl(a):
        mask=a>0
        return float(np.sum(a[mask]*np.log(a[mask]/m[mask])))
    value=.5*(kl(p)+kl(q))
    if value< -1e-14 or value>np.log(2)+1e-6:raise ValueError('JS out of range')
    return max(0.,value)


def response(p,q):
    c=int(p.argmax());d=float(p[c]-q[c]);change=float(q.max()-p.max())
    return dict(D_pred=d,D_abs=abs(d),cancellation=0.,signed_delta_max_confidence=change,
                mean_abs_delta_max_confidence=abs(change),JS=js(p,q),flip_rate=float(q.argmax()!=c))


def unique(rows,key):
    result={}
    for r in rows:
        k=key(r)
        if k in result:raise ValueError('Duplicate source identity: '+str(k))
        result[k]=r
    return result


def bootstrap_counts(dialogues,reps,seed):
    rng=np.random.default_rng(seed);n=len(dialogues)
    draws=rng.integers(0,n,size=(reps,n))
    return np.stack([np.bincount(row,minlength=n) for row in draws]).astype(float)


def divide(a,b):
    a,b=np.broadcast_arrays(a,b);out=np.full(a.shape,np.nan)
    np.divide(a,b,out=out,where=b>0);return out


def interval(values,support):
    finite=np.asarray(values)[np.isfinite(values)]
    if support<2 or not len(finite):return dict(ci_low=None,ci_high=None,valid_bootstraps=len(finite),ci_status='INSUFFICIENT_DIALOGUE_SUPPORT')
    lo,hi=np.quantile(finite,[.025,.975])
    return dict(ci_low=float(lo),ci_high=float(hi),valid_bootstraps=len(finite),
                ci_status='VERY_SMALL_DIALOGUE_SUPPORT' if support<5 else 'DESCRIPTIVE_POINTWISE')


def group_arrays(rows,metric,dialogues):
    lookup={d:i for i,d in enumerate(dialogues)};sums=np.zeros(len(dialogues));counts=np.zeros(len(dialogues))
    for r in rows:
        v=r.get(metric)
        if v is not None:
            j=lookup[r['dialogue_id']];sums[j]+=v;counts[j]+=1
    return sums,counts


def summary_rows(condition_rows,dialogues,draws):
    output=[]
    for correctness in ('all','correct','wrong'):
        planned=[r for r in condition_rows if correctness=='all' or r['original_correct']==int(correctness=='correct')]
        complete=[r for r in planned if r['complete']]
        for metric in METRICS:
            values=np.array([r[metric] for r in complete if r[metric] is not None],float)
            sums,counts=group_arrays(complete,metric,dialogues);present=counts>0
            row=dict(correctness=correctness,metric=metric,n_planned=len(planned),n_valid=len(values),n_missing=len(planned)-len(values),
                     n_dialogues=int(present.sum()),mean_utterance=float(values.mean()) if len(values) else None,
                     mean_dialogue=float((sums[present]/counts[present]).mean()) if present.any() else None)
            row.update({k:float(np.quantile(values,q)) if len(values) else None for k,q in [('p05',.05),('median',.5),('p95',.95)]})
            row.update(minimum=float(values.min()) if len(values) else None,maximum=float(values.max()) if len(values) else None,
                       negative_fraction=float((values<0).mean()) if len(values) else None,
                       zero_fraction=float((values==0).mean()) if len(values) else None,
                       positive_fraction=float((values>0).mean()) if len(values) else None)
            utterance=interval(divide(draws@sums,draws@counts),int(present.sum()))
            means=np.zeros_like(sums);np.divide(sums,counts,out=means,where=present)
            equal=interval(divide(draws@means,draws@present.astype(float)),int(present.sum()))
            row.update({f'utterance_{k}':v for k,v in utterance.items()})
            row.update({f'dialogue_{k}':v for k,v in equal.items()});output.append(row)
    return output


def comparison_rows(rows,dialogues,draws):
    output=[];cells=[]
    complete=[r for r in rows if r['complete']]
    correct=[r for r in complete if r['original_correct']];wrong=[r for r in complete if not r['original_correct']]
    for metric in COMPARISON_METRICS:
        sc,nc=group_arrays(correct,metric,dialogues);sw,nw=group_arrays(wrong,metric,dialogues)
        both=(nc>0)&(nw>0);dc=divide(sc,nc);dw=divide(sw,nw)
        common=dict(metric=metric,n_correct=int(nc.sum()),n_wrong=int(nw.sum()),
                    correct_dialogues=int((nc>0).sum()),wrong_dialogues=int((nw>0).sum()))
        global_diff=float(sw.sum()/nw.sum()-sc.sum()/nc.sum()) if nc.sum() and nw.sum() else None
        estimate=divide(draws@sw,draws@nw)-divide(draws@sc,draws@nc)
        supported=min(int((nc>0).sum()),int((nw>0).sum()))
        output.append(dict(common,estimand='cluster_global_error_minus_correct',difference=global_diff,support_dialogues=int(((nc>0)|(nw>0)).sum()),
                           paired_cells=None,**interval(estimate,supported)))
        difference=np.where(both,dw-dc,0)
        output.append(dict(common,estimand='within_dialogue_error_minus_correct',difference=float(difference[both].mean()) if both.any() else None,
                           support_dialogues=int(both.sum()),paired_cells=int(both.sum()),
                           **interval(divide(draws@difference,draws@both.astype(float)),int(both.sum()))))
        # Exact Dialogue x clean-confidence width 0.1 cells, equal fixed cell weight.
        bycell=defaultdict(list)
        for r in complete:bycell[(r['dialogue_id'],min(9,int(r['original_confidence']*10)))].append(r)
        cell_sum=np.zeros(len(dialogues));cell_count=np.zeros(len(dialogues));lookup={d:i for i,d in enumerate(dialogues)}
        for (did,bin_id),members in sorted(bycell.items()):
            c=[r[metric] for r in members if r['original_correct']];w=[r[metric] for r in members if not r['original_correct']]
            if not c or not w:continue
            diff=float(np.mean(w)-np.mean(c));j=lookup[did];cell_sum[j]+=diff;cell_count[j]+=1
            cells.append(dict(dialogue_id=did,confidence_bin=f'{bin_id/10:.1f}-{(bin_id+1)/10:.1f}',metric=metric,
                              n_correct=len(c),n_wrong=len(w),correct_mean=float(np.mean(c)),wrong_mean=float(np.mean(w)),difference=diff))
        supported=int((cell_count>0).sum())
        output.append(dict(common,estimand='within_dialogue_confidence_cell_error_minus_correct',
                           difference=float(cell_sum.sum()/cell_count.sum()) if cell_count.sum() else None,
                           support_dialogues=supported,paired_cells=int(cell_count.sum()),
                           **interval(divide(draws@cell_sum,draws@cell_count),supported)))
    return output,cells


def weighted_correlation(x,y,weights):
    total=weights.sum(1);mx=divide(weights@x,total);my=divide(weights@y,total)
    dx=x[None,:]-mx[:,None];dy=y[None,:]-my[:,None]
    numerator=(weights*dx*dy).sum(1);denom=np.sqrt((weights*dx*dx).sum(1)*(weights*dy*dy).sum(1))
    return divide(numerator,denom)


def weighted_ranks(x,weights):
    order=np.argsort(x,kind='stable');sorted_x=x[order];cumulative=np.cumsum(weights[:,order],axis=1)
    ends=np.r_[np.flatnonzero(np.diff(sorted_x)),len(x)-1]
    group_cumulative=cumulative[:,ends];group_weights=np.diff(np.c_[np.zeros(len(weights)),group_cumulative],axis=1)
    group_rank=group_cumulative-.5*group_weights+.5
    groups=np.cumsum(np.r_[False,np.diff(sorted_x)!=0]).astype(int)
    sorted_ranks=group_rank[:,groups];ranks=np.empty_like(sorted_ranks);ranks[:,order]=sorted_ranks
    return ranks


def spearman_draws(x,y,weights):
    rx=weighted_ranks(x,weights);ry=weighted_ranks(y,weights);total=weights.sum(1)
    mx=divide((weights*rx).sum(1),total);my=divide((weights*ry).sum(1),total)
    dx=rx-mx[:,None];dy=ry-my[:,None]
    return divide((weights*dx*dy).sum(1),np.sqrt((weights*dx*dx).sum(1)*(weights*dy*dy).sum(1)))


def correlation_rows(rows,dialogues,draws):
    output=[];lookup={d:i for i,d in enumerate(dialogues)}
    for feature in ('feature_rmse','feature_relative_l2','feature_cosine_distance'):
        for metric in ('D_pred','D_abs','JS'):
            members=[r for r in rows if r['complete'] and r.get(feature) is not None and r.get(metric) is not None]
            base=dict(feature=feature,response=metric,n=len(members),n_dialogues=len({r['dialogue_id'] for r in members}))
            if len(members)<3:
                output.append(dict(base,correlation=None,ci_low=None,ci_high=None,valid_bootstraps=0,ci_status='INSUFFICIENT_DATA'));continue
            x=np.array([r[feature] for r in members]);y=np.array([r[metric] for r in members]);w=draws[:,[lookup[r['dialogue_id']] for r in members]]
            corr=spearman_draws(x,y,np.ones((1,len(x))))[0]
            values=spearman_draws(x,y,w)
            output.append(dict(base,correlation=float(corr) if np.isfinite(corr) else None,**interval(values,base['n_dialogues'])))
    return output


def self_test():
    p=np.array([.6,.3,.02,.02,.02,.02,.02]);q=np.array([.2,.7,.02,.02,.02,.02,.02])
    assert abs(response(p,q)['D_pred']-.4)<1e-12 and response(p,q)['flip_rate']==1
    assert js(p,p)==0 and abs(js(p,q)-js(q,p))<1e-14
    assert abs(response(p,q)['signed_delta_max_confidence']-.1)<1e-12
    # Weighted tied ranks agree with explicitly duplicated bootstrap observations.
    x=np.array([1.,1.,3.,4.]);y=np.array([2.,4.,1.,3.]);weights=np.array([[2.,1.,0.,3.]])
    indices=np.repeat(np.arange(4),weights[0].astype(int))
    expanded=spearman_draws(x[indices],y[indices],np.ones((1,len(indices))))[0]
    assert abs(expanded-spearman_draws(x,y,weights)[0])<1e-12
    assert np.array_equal(bootstrap_counts(['a','b'],20,3),bootstrap_counts(['a','b'],20,3))
    assert interval(np.ones(2000),1)['ci_low'] is None
    return dict(probability_and_JS=True,protected_clean_class=True,signed_confidence_delta=True,
                weighted_spearman_ties_vs_expansion=True,bootstrap_determinism=True,single_cluster_not_false_precision=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.output_root.resolve();out=root/'results/audio';out.mkdir(parents=True,exist_ok=True)
    protocol_path=root/'EXPERIMENT_PROTOCOL.json';protocol=json.loads(protocol_path.read_text())
    reps=protocol['scoring']['bootstrap']['replicates'];seed=protocol['scoring']['bootstrap']['seed']
    if reps!=2000 or seed!=20261121:raise ValueError('Unexpected frozen bootstrap parameters')
    checks=self_test();inputs={str(protocol_path):sha(protocol_path),str(Path(__file__).resolve()):sha(__file__)};audit=[]
    final_path=PARENT/'provenance/FINAL_ARTIFACT_MANIFEST.json';final=json.loads(final_path.read_text());inputs[str(final_path)]=sha(final_path)
    historical_verification=PARENT/'verification.json'
    if sha(historical_verification)!=final['verification_sha256']:raise ValueError('Historical final verification chain changed')
    inputs[str(historical_verification)]=sha(historical_verification)
    anchored=dict(final['code_configs_reports_and_statistics'])
    for path,digest in json.loads(historical_verification.read_text())['artifacts_sha256'].items():
        if path in anchored and anchored[path]!=digest:raise ValueError('Historical manifest hash conflict')
        anchored[path]=digest
    def load(path,kind):
        path=Path(path);digest=sha(path);inputs[str(path.resolve())]=digest
        relative=str(path.relative_to(PARENT));expected=anchored.get(relative)
        if expected is not None and digest!=expected:raise ValueError('Historical source hash mismatch: '+relative)
        audit.append(dict(path=relative,source_sha256=digest,existing_manifest_sha256=expected,
                          existing_hash_match=digest==expected if expected else None,status='MATCH' if expected else 'NEW_SNAPSHOT_ONLY'))
        return read_csv(path) if kind=='csv' else json.loads(path.read_text())
    feature=load(PARENT/'results/analysis/feature_response_per_variant.csv','csv')
    feature=unique([r for r in feature if r['modality']=='audio'],lambda r:(r['dataset'],r['sample_id'],r['variant_id']))
    repeat=load(PARENT/'results/analysis/feature_clean_repeat_distances.csv','csv')
    repeat=unique([r for r in repeat if r['modality']=='audio'],lambda r:(r['dataset'],r['sample_id']))
    analysis_proof=load(PARENT/'results/analysis/analysis_verification.json','json')
    per_variant=[];controls=[];conditions=[];failures=[];dataset_support={};universes={};bootstrap={}
    for ds in DATASETS:
        classes=6 if ds=='IEMOCAP' else 7;base=PARENT/'results/pilot'/ds
        original=load(base/'original_predictions.csv','csv');sample=load(base/'sample_manifest.csv','csv')
        perturbed=load(base/'perturbed_predictions_both.csv','csv');transform=load(base/'transformation_manifest_both.csv','csv')
        control=load(base/'clean_reextraction_controls_both.csv','csv');selection=load(base/'selection_frozen.json','json')
        inference=load(base/'inference_both.json','json');lock_path=PARENT/f'configs/BASELINE_LOCK_{ds}.json';load(lock_path,'json')
        if sha(lock_path)!=selection['baseline_lock_sha256'] or sha(base/'sample_manifest.csv')!=selection['sample_manifest_sha256']:
            raise ValueError('Frozen original/sample binding mismatch')
        if inference['sample_manifest_sha256']!=sha(base/'sample_manifest.csv') or not inference['model_state_bit_exact'] or inference['test_accessed']:
            raise ValueError('Historical inference state/split verification failed')
        probability_path=(base/'perturbed_predictions_both.csv').resolve()
        project=PARENT.parents[3]
        expected=next((digest for name,digest in analysis_proof['input_sha256'].items()
                       if (project/Path(name)).resolve()==probability_path),None)
        if expected is None or expected!=sha(base/'perturbed_predictions_both.csv'):raise ValueError('Prior analysis probability hash binding mismatch')
        original=unique(original,lambda r:r['sample_id']);samples=unique(sample,lambda r:r['sample_id'])
        if len(samples)!=120:raise ValueError('Historical pilot must retain fixed120')
        universes[ds]=sorted({r['dialogue_id'] for r in sample});bootstrap[ds]=bootstrap_counts(universes[ds],reps,seed)
        control=unique([r for r in control if r['modality']=='audio'],lambda r:r['sample_id'])
        tr=unique([r for r in transform if r['modality']=='audio'],lambda r:(r['sample_id'],r['variant_id']))
        perturbed=[r for r in perturbed if r['modality']=='audio'];pr=unique(perturbed,lambda r:(r['sample_id'],r['variant_id']))
        expected_keys={(sid,v) for sid in samples for vs in VARIANTS.values() for v in vs}
        if set(pr)!=expected_keys or set(tr)!=expected_keys:raise ValueError('Missing or extra acoustic planned variant')
        for sid,s in samples.items():
            o=original[sid]
            for field,value in s.items():
                if o[field]!=value:raise ValueError('Selected sample differs from original full Validation prediction')
            p=probability(o['original_probabilities'],classes);c=int(p.argmax())
            if c!=int(o['original_prediction']) or abs(p.max()-float(o['original_confidence']))>1e-12 or (c==int(o['true_label']))!=yes(o['original_correct']):
                raise ValueError('Clean prediction/label consistency mismatch')
            ctrl=control.get(sid);fr=repeat.get((ds,sid))
            cr=dict(dataset=ds,sample_id=sid,dialogue_id=s['dialogue_id'],control_status='PRESENT' if ctrl else 'MISSING',
                    clean_probability_max_abs=number(ctrl['clean_probability_max_abs']) if ctrl else None,
                    clean_prediction_unchanged=yes(ctrl['clean_prediction_unchanged']) if ctrl else None,
                    clean_feature_rmse=number(ctrl['clean_feature_rmse']) if ctrl else None,
                    feature_repeat_rmse=number(fr['rmse']) if fr else None,
                    feature_repeat_relative_l2=number(fr['relative_l2']) if fr else None,
                    feature_repeat_cosine=number(fr['cosine']) if fr else None)
            if ctrl and fr and abs(float(ctrl['clean_feature_rmse'])-float(fr['rmse']))>1e-14:raise ValueError('Clean control feature distance mismatch')
            controls.append(cr)
            for family,variants in VARIANTS.items():
                for variant in variants:
                    raw=pr[(sid,variant)];tm=tr[(sid,variant)]
                    for field in set(raw)&set(tm):
                        if raw[field]!=tm[field]:raise ValueError('Prediction/transformation metadata mismatch: '+field)
                    for field in ('dataset','seed','dialogue_id','utterance_index','utterance_id','speaker_id','true_label','original_prediction','original_correct','original_confidence','confidence_bin'):
                        if raw[field]!=o[field]:raise ValueError('Audio target does not exactly match clean utterance')
                    if not np.array_equal(probability(raw['original_probabilities'],classes),p):raise ValueError('Clean vector differs in perturbation row')
                    if raw['precomputed_feature_file_sha256']!=inference['precomputed_features_sha256']:raise ValueError('Precomputed feature hash binding mismatch')
                    if raw['perturbation_type']!=family or int(raw['planned_k'])!=len(variants):raise ValueError('Family/K mismatch')
                    qc=json.loads(raw['quality_control']) if raw['quality_control'] else {}
                    record=dict(dataset=ds,sample_id=sid,dialogue_id=s['dialogue_id'],utterance_id=s['utterance_id'],utterance_index=int(s['utterance_index']),
                        family=family,variant_id=variant,intensity='0.5_semitone' if '0.5' in variant else '1_semitone' if family=='pitch_shift' else variant.removeprefix('timbre_'),
                        original_correct=int(o['original_correct']),original_confidence=float(o['original_confidence']),original_prediction=c,
                        status=raw['status'],label_review_status=raw['label_review_status'],label_preserving_main_eligible=yes(raw['label_preserving_main_eligible']),
                        saved_automatic_qc_pass=raw['automatic_qc_pass'],saved_signal_qc_pass=qc.get('automatic_signal_qc_pass'),
                        non_target_features_bit_exact=raw['non_target_features_bit_exact'],dialogue_context_hashes_unchanged=raw['dialogue_context_hashes_unchanged'],
                        original_input_sha256=raw['original_input_sha256'],processed_input_sha256=raw['processed_input_sha256'],
                        original_feature_sha256=raw['original_feature_sha256'],target_feature_sha256=raw['target_feature_sha256'],artifact_sha256=raw['artifact_sha256'],
                        original_probabilities=raw['original_probabilities'],perturbed_probabilities=raw['perturbed_probabilities'],error=raw['error'],
                        rms_change_db=qc.get('rms_change_db'),raw_length_relative_difference=qc.get('raw_length_relative_difference'),
                        clipping_fraction=qc.get('clipping_fraction'),clean_clipping_fraction=qc.get('clean_clipping_fraction'),
                        control_status=cr['control_status'],clean_probability_repeat_bound=cr['clean_probability_max_abs'],
                        feature_rmse=None,feature_relative_l2=None,feature_cosine_distance=None,
                        D_pred=None,D_abs=None,cancellation=None,signed_delta_max_confidence=None,mean_abs_delta_max_confidence=None,JS=None,flip_rate=None,
                        abs_D_above_clean_repeat=None,abs_max_delta_above_clean_repeat=None)
                    if raw['status']=='SUCCESS':
                        if not yes(raw['automatic_qc_pass']) or not yes(qc.get('automatic_signal_qc_pass')) or not yes(raw['non_target_features_bit_exact']) or not yes(raw['dialogue_context_hashes_unchanged']):
                            raise ValueError('Successful prediction lacks frozen input/signal QC')
                        q=probability(raw['perturbed_probabilities'],classes)
                        if int(q.argmax())!=int(raw['perturbed_prediction']) or abs(q.max()-float(raw['perturbed_max_confidence']))>1e-12 or abs(q[c]-float(raw['perturbed_original_class_probability']))>1e-12:
                            raise ValueError('Protected probability or perturbed argmax mismatch')
                        if abs(q[int(raw['true_label'])]-float(raw['perturbed_original_label_probability']))>1e-12:raise ValueError('Saved true-label probability mismatch')
                        fm=feature.get((ds,sid,variant))
                        if fm is None or fm['perturbation_type']!=family or fm['label_review_status']!=raw['label_review_status']:raise ValueError('Missing/mismatched saved feature movement')
                        for field in ('original_input_sha256','processed_input_sha256','original_feature_sha256','target_feature_sha256','artifact_sha256'):
                            if len(raw[field])!=64:raise ValueError('Missing successful SHA256 field')
                        record.update(response(p,q),feature_rmse=float(fm['rmse']),feature_relative_l2=float(fm['relative_l2']),feature_cosine_distance=1-float(fm['cosine']))
                        if cr['clean_probability_max_abs'] is not None:
                            record['abs_D_above_clean_repeat']=record['D_abs']>cr['clean_probability_max_abs']
                            record['abs_max_delta_above_clean_repeat']=abs(record['signed_delta_max_confidence'])>cr['clean_probability_max_abs']
                    elif raw['status']=='FAILED':
                        if raw['perturbed_probabilities']:raise ValueError('Failed row unexpectedly contains usable probability')
                        failures.append(record)
                    else:raise ValueError('Unknown saved audio status')
                    per_variant.append(record)
        dataset_support[ds]=dict(planned=len(perturbed),successful=sum(r['status']=='SUCCESS' for r in perturbed),failed=sum(r['status']=='FAILED' for r in perturbed),
                                  selected_utterances=len(samples),selected_dialogues=len(universes[ds]),correct=sum(yes(s['original_correct']) for s in sample),wrong=sum(not yes(s['original_correct']) for s in sample),
                                  label_review_counts=dict(Counter(r['label_review_status'] for r in perturbed)),main_eligible=sum(yes(r['label_preserving_main_eligible']) for r in perturbed),
                                  clean_controls=len(control),full_validation_rows=len(original))
    if set(feature)!={(r['dataset'],r['sample_id'],r['variant_id']) for r in per_variant if r['status']=='SUCCESS'}:
        raise ValueError('Saved audio feature movement coverage differs from successful acoustic predictions')
    # Ten fixed per-sample conditions: 6 versions, 2 full families, 2 pitch magnitudes.
    grouped=defaultdict(list)
    for r in per_variant:grouped[(r['dataset'],r['sample_id'])].append(r)
    for (ds,sid),records in grouped.items():
        specs=[('version',r['variant_id'],r['family'],[r['variant_id']]) for r in records]
        specs += [('family',f'{f}:ALL_K{len(vs)}',f,list(vs)) for f,vs in VARIANTS.items()]
        specs += [('intensity','pitch_abs_'+mag,'pitch_shift',[f'pitch_plus_{mag}',f'pitch_minus_{mag}']) for mag in ('0.5','1')]
        indexed={r['variant_id']:r for r in records}
        for scope,condition,family,versions in specs:
            members=[indexed[v] for v in versions];base=members[0];valid=[r for r in members if r['status']=='SUCCESS'];complete=len(valid)==len(members)
            row={k:base[k] for k in ('dataset','sample_id','dialogue_id','original_correct','original_confidence')}
            row.update(scope=scope,condition=condition,family=family,expected_k=len(members),successful_k=len(valid),complete=complete,
                       status='COMPLETE_ENGINEERING_ONLY' if complete else 'INCOMPLETE_RETAINED_NA',
                       failed_versions='|'.join(r['variant_id'] for r in members if r['status']!='SUCCESS'),label_preserving_main_eligible=all(r['label_preserving_main_eligible'] for r in members))
            for metric in METRICS:row[metric]=float(np.mean([r[metric] for r in valid])) if complete else None
            if complete:row['cancellation']=max(0.,row['D_abs']-abs(row['D_pred']))
            conditions.append(row)
    summaries=[];comparisons=[];cells=[];correlations=[];dialogue_summary=[]
    condition_groups=defaultdict(list)
    for r in conditions:condition_groups[(r['dataset'],r['scope'],r['condition'],r['family'])].append(r)
    for (ds,scope,condition,family),rows in sorted(condition_groups.items()):
        identity=dict(dataset=ds,scope=scope,condition=condition,family=family)
        summaries.extend(dict(identity,**r) for r in summary_rows(rows,universes[ds],bootstrap[ds]))
        comp,cell=comparison_rows(rows,universes[ds],bootstrap[ds]);comparisons.extend(dict(identity,**r) for r in comp);cells.extend(dict(identity,**r) for r in cell)
        correlations.extend(dict(identity,**r) for r in correlation_rows(rows,universes[ds],bootstrap[ds]))
        bydialogue=defaultdict(list)
        for r in rows:bydialogue[r['dialogue_id']].append(r)
        for did,members in bydialogue.items():
            valid=[r for r in members if r['complete']]
            summary=dict(identity,dialogue_id=did,n_planned=len(members),n_valid=len(valid),n_missing=len(members)-len(valid),
                         n_correct=sum(r['original_correct'] for r in valid),n_wrong=sum(not r['original_correct'] for r in valid))
            for metric in COMPARISON_METRICS:
                for name,label in [('all',None),('correct',1),('wrong',0)]:
                    values=[r[metric] for r in valid if label is None or r['original_correct']==label]
                    summary[name+'_'+metric]=float(np.mean(values)) if values else None
            dialogue_summary.append(summary)
    outputs={'variant_responses.csv':per_variant,'condition_features.csv':conditions,'response_distributions.csv':summaries,
             'correct_wrong_comparisons.csv':comparisons,'within_dialogue_confidence_cells.csv':cells,
             'feature_response_correlations.csv':correlations,'clean_control_comparison.csv':controls,
             'dialogue_response_summary.csv':dialogue_summary,'failed_variants.csv':failures,'source_integrity_audit.csv':audit}
    for name,rows in outputs.items():write_csv(out/name,rows)
    # Summarize engineering changes, keeping lack of audio label review explicit.
    def getmean(ds,scope,condition,metric,group='all'):
        return next(r['mean_utterance'] for r in summaries if r['dataset']==ds and r['scope']==scope and r['condition']==condition and r['metric']==metric and r['correctness']==group)
    lines=['# 历史音频响应离线诊断','',
      '本轮只重算历史 Plan B 真正 Validation 音频 Pilot 的已保存统计，没有打开媒体、权重或特征张量，没有新推理、训练或 Test 访问。历史两数据集各固定120条分层样本；它们不是数据集总体的随机抽样。',
      '', '## 结论与可用证据','',
      '音频工程干扰的响应可以从保存概率中独立复算；但全部音频版本仍为 PENDING 标签审核，标签保持主分析资格为0。当前不建议把声学 D_pred 加入正式可靠性评分或以本轮选择最佳音频条件。以下正误区别仅按原句预测是否正确分组，不计算干扰后准确率或假定情绪标签仍成立。',
      '', '## 覆盖与保留失败','', '| 数据集 | 固定样本/Dialogue | 成功/计划版本 | 失败 | clean重提取控制 |', '|---|---:|---:|---:|---:|']
    for ds,s in dataset_support.items():lines.append(f"| {ds} | {s['selected_utterances']}/{s['selected_dialogues']} | {s['successful']}/{s['planned']} | {s['failed']} | {s['clean_controls']}/120 |")
    lines += ['', '1440个原计划版本全部保留；失败版本概率和响应为空。整类指标只在固定全部K版成功时计算，不混用不同K，也不以可用版本均值替代缺失版本。六个单版本、完整pitch K=4、完整timbre K=2及两个pitch强度配对各自分开报告。',
      '', '## 指标口径','',
      '- 固定原句预测类 c=argmax(p_clean)。单版本 d=p_clean(c)-p_perturbed(c)，家族 D_pred=mean(d)，有正负号；D_abs=mean(abs(d))，cancellation=D_abs-abs(D_pred)。',
      '- signed_delta_max_confidence=mean(max(p_perturbed)-max(p_clean))。这是有符号最大置信度变化，既不是固定原类概率变化，也不是既有正式δ的替代定义。',
      '- JS为每版与原概率的自然对数Jensen–Shannon散度均值，零概率项贡献0；flip_rate只比较两次argmax，不把翻转率称为μ。',
      '- 分布同时给出话语等权点估计及Dialogue等权点估计。正确/错误的总体差使用相同Dialogue抽样；随后单列同Dialogue配对、同Dialogue×0.1原置信度格配对。普通cluster区间本身不消除Dialogue或置信度混杂。',
      f'- 使用冻结协议的{reps}次Dialogue有放回Bootstrap，seed={seed}，95%百分位区间。同一数据集各条件共用固定120样本的Dialogue抽样矩阵，重复话语不视为独立抽样单位。少于2个有效Dialogue不生成区间，2–4个Dialogue仅标为极小支持；同时报告有效重采样次数。所有区间是探索性的逐项区间，不作多重比较显著性筛选。',
      '', '## 六个版本的观察响应','', '| 数据集 | 版本 | n成功 | D_pred均值 | D_abs均值 | 有符号max置信度变化 | JS均值 | 翻转率 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        for vs in VARIANTS.values():
            for version in vs:
                n=sum(r['status']=='SUCCESS' for r in per_variant if r['dataset']==ds and r['variant_id']==version)
                vals=[getmean(ds,'version',version,k) for k in ('D_pred','D_abs','signed_delta_max_confidence','JS','flip_rate')]
                lines.append(f"| {ds} | {version} | {n} | "+' | '.join(f'{v:.6g}' for v in vals)+' |')
    lines += ['', '## 正误区分与Dialogue支持','', '| 数据集 | 完整家族 | 比较口径 | 错误减正确 D_pred | 95%区间 | 支持Dialogue/格 |', '|---|---|---|---:|---|---:|']
    for r in comparisons:
        if r['scope']!='family' or r['metric']!='D_pred':continue
        point='NA' if r['difference'] is None else f"{r['difference']:.6g}"
        ci='NA' if r['ci_low'] is None else f"[{r['ci_low']:.6g}, {r['ci_high']:.6g}]"
        lines.append(f"| {r['dataset']} | {r['condition']} | {r['estimand']} | {point} | {ci} | {r['support_dialogues']}/{r['paired_cells'] if r['paired_cells'] is not None else '—'} |")
    lines += ['', 'MELD的120条分布在114个Dialogue，重复话语很少；正误同Dialogue及同置信度格的支持尤其有限。没有共享支持时明确保留NA，不能把总体正误差或较窄的cluster区间解释为控制混杂后的增量价值。D_pred有符号，负值意味着原预测类概率上升；较大绝对响应只说明受影响，不自动指示原预测错误。',
      '', '## clean控制与特征移动','',
      '两数据集各119条现存音频clean重提取控制的特征及完整概率最大绝对误差均为0，预测均未变。每数据集另有1条缺少概率控制；单列缺失，未补成0。重提取误差为0时不计算响应/噪声比；直接列出响应是否超过保存的概率最大绝对误差界。该控制只能排除这些已观测clean重复的数值漂移，不能证明变换情绪保持，也不能证明所有潜在非目标影响都已因果排除。',
      '', '特征RMSE、relative L2和1−cosine直接复用既有描述CSV；与D_pred、D_abs、JS的Spearman关系分别按数据集、家族、版本及pitch强度计算，区间按Dialogue聚类。相关说明当前编码变化和模型响应的关联，不能把它当成音频质量、情绪标签保持或对错误的独立诊断。逐项结果见 feature_response_correlations.csv，不按相关大小追选条件。',
      '', '## 完整性与边界','',
      '逐行核对固定清单与全Validation原记录的身份、标签、原预测及完整概率；再核对扰动行的原概率、固定原类概率、最大置信度、argmax和目标话语身份。保存的transform清单与prediction清单共有字段逐字对应，成功行的非目标特征/上下文保持标记必须为真；特征移动行覆盖必须与成功音频行完全相等。来源CSV/JSON与历史最终产物清单、冻结样本及历史分析输入哈希交叉验证。',
      '', '这里重新哈希的是允许读取的CSV/JSON证据，音频/权重/大型特征文件未打开；其中列出的媒体、特征和模型哈希只核对保存记录之间的一致性，并不宣称本轮重新验证了这些二进制文件。详细来源和断言见 source_integrity_audit.csv 与 verification.json。',
      '', '下一步如单独获授权，先做独立音频标签保持审核和变换失真检查，再在不接触已查看评估结果的预先冻结新样本上检验可靠性增量。本轮停止于离线诊断，不自动执行审核、评分或新实验。','']
    extra=['## 完整家族幅度与特征关联','',
        '单版本平均绝对原类概率变化约0.069–0.253个百分点。IEMOCAP pitch为476个成功版本中5次翻转，涉及3条不同话语；MELD pitch为472个成功版本中1次翻转，涉及1条话语。两个timbre家族均未翻转。这些版本计数只用于描述，同一话语的多个版本不视为独立样本。全部1414个成功版本的绝对原类概率变化均超过对应0误差clean控制；超出数值噪声不等于具有可靠性价值。','',
        '| 数据集 | 家族 | 完整K样本 | D_abs均值 [Dialogue聚类95% CI] | relative L2与D_abs Spearman [95% CI] | relative L2与JS Spearman [95% CI] |',
        '|---|---|---:|---|---|---|']
    for ds in DATASETS:
        for family in VARIANTS:
            r=next(x for x in summaries if x['dataset']==ds and x['scope']=='family' and x['family']==family and x['correctness']=='all' and x['metric']=='D_abs')
            values=[f"{r['mean_utterance']:.6g} [{r['utterance_ci_low']:.6g}, {r['utterance_ci_high']:.6g}]"]
            for metric in ('D_abs','JS'):
                c=next(x for x in correlations if x['dataset']==ds and x['scope']=='family' and x['family']==family and x['feature']=='feature_relative_l2' and x['response']==metric)
                values.append(f"{c['correlation']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
            extra.append(f"| {ds} | {family} | {r['n_valid']} | "+' | '.join(values)+' |')
    extra += ['', 'IEMOCAP两个家族控制Dialogue×原置信度格后的D_pred正误差区间均覆盖0，本轮没有给出稳定分离证据，但这不证明真实效应必为0。MELD两个完整家族同Dialogue正误配对均为0，不能据其总体差宣称控制混杂后的价值。特征移动与无符号响应的关联较清楚，但与有符号D_pred的四个家族相关区间均覆盖0；可能存在正负抵消，仍不能由此推出错误可分。',
        '', '26个失败版本按保存QC数值检查：24个存在绝对RMS变化超过0.5 dB，另2个MELD pitch版本原始变换时长差超过5%。这是既存QC失败的描述，没有重新处理或替换条件。','']
    position=lines.index('## 完整性与边界')
    lines[position:position]=extra
    (root/'AUDIO_RESPONSE_DIAGNOSTIC.md').write_text('\n'.join(lines),encoding='utf-8')
    if len(per_variant)!=1440 or len(failures)!=26 or len(conditions)!=2400:raise ValueError('Coverage regression')
    if any(r[m] is not None for r in failures for m in METRICS):raise ValueError('Failed status was silently filled')
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Source changed during offline analysis')
    verification=dict(status='COMPLETE_OFFLINE_ONLY',task=4,no_model_or_media_opened=True,no_inference=True,no_training=True,no_test=True,
        historical_validation_only=True,replicates=reps,seed=seed,bootstrap_unit='paired Dialogue cluster',no_mixed_K=True,
        all_planned_audio_variants_retained=True,failed_metrics_remain_NA=True,datasets=dataset_support,
        source_file_hashes=inputs,self_tests=checks,
        output_rows={name:len(rows) for name,rows in outputs.items()},
        output_sha256={name:sha(out/name) for name in outputs},report_sha256=sha(root/'AUDIO_RESPONSE_DIAGNOSTIC.md'),
        binary_hash_limit='Saved binary/media/weight hashes cross-checked between metadata only; no binary asset opened',
        scoring_recommendation='DO_NOT_SCORE_YET_ALL_AUDIO_LABEL_REVIEWS_PENDING')
    dump(out/'verification.json',verification)
    print(json.dumps(dict(status=verification['status'],datasets=dataset_support,outputs=verification['output_rows'])),flush=True)


if __name__=='__main__':main()
