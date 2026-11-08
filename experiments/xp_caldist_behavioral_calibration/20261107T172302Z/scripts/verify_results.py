"""Independent artifact, identity, provenance and metric checks; no GPU inference."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def independent_metrics(p,q):
    p,q=p.astype(np.float64),q.astype(np.float64)
    c,cj=p.max(1),q.max(2);flip=q.argmax(2)!=p.argmax(1)[:,None];mid=(p[:,None]+q)/2
    with np.errstate(divide='ignore',invalid='ignore'):
        kl1=np.where(p[:,None]>0,p[:,None]*np.log(p[:,None]/mid),0).sum(2)
        kl2=np.where(q>0,q*np.log(q/mid),0).sum(2)
    return dict(flip_rate=flip.mean(1),mu=(flip*cj).mean(1),delta_original=np.abs(c-cj.mean(1)),
                confidence_mad=np.abs(c[:,None]-cj).mean(1),js_divergence=((kl1+kl2)/2).mean(1),confidence_change=cj.mean(1)-c)


def independent_classification(labels,p):
    p=np.asarray(p,np.float64);confidence=p.max(1);correct=p.argmax(1)==labels
    bins=np.searchsorted(np.linspace(0,1,16)[1:],confidence,side='left').clip(0,14)
    ece=sum(abs(confidence[bins==b].mean()-correct[bins==b].mean())*np.mean(bins==b) for b in range(15) if np.any(bins==b))
    return dict(ece15=ece*100,brier=float(((p-np.eye(p.shape[1])[labels])**2).sum(1).mean()),
                nll=float(-np.log(p[np.arange(len(p)),labels]).mean()),auroc_success=float(roc_auc_score(correct,confidence))*100,accuracy=float(correct.mean()))


def independent_success(y,score):
    y=np.asarray(y,int);score=np.asarray(score,float);clipped=np.clip(score,1e-15,1-1e-15);n1=y.sum();n0=len(y)-n1
    auc=(rankdata(score)[y==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan
    return dict(auroc=auc,brier=float(((score-y)**2).mean()),nll=float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()))


def verify_conditional(root):
    records=list(csv.DictReader((root/'results/conditional_information_diagnostics.csv').open()))
    oof=list(csv.DictReader((root/'results/conditional_oof_predictions.csv').open()))
    for tag in sorted({(r['dataset'],r['modality'],r['perturbation_type']) for r in oof}):
        group=[r for r in oof if (r['dataset'],r['modality'],r['perturbation_type'])==tag]
        y=np.asarray([r['correct']=='True' for r in group],int);a=np.asarray([float(r['baseline_oof_success']) for r in group]);b=np.asarray([float(r['behavior_oof_success']) for r in group]);dialogues=np.asarray([r['dialogue_id'] for r in group])
        assert np.isfinite(a).all() and np.isfinite(b).all()
        ma,mb=independent_success(y,a),independent_success(y,b);units=[np.flatnonzero(dialogues==d) for d in np.unique(dialogues)]
        rng=np.random.default_rng(20261107);draws={k:[] for k in ma}
        for _ in range(2000):
            ix=np.concatenate([units[i] for i in rng.integers(0,len(units),len(units))]);da=independent_success(y[ix],a[ix]);db=independent_success(y[ix],b[ix])
            for k in draws:
                if np.isfinite(da[k]) and np.isfinite(db[k]):draws[k].append(db[k]-da[k])
        for rec in [r for r in records if (r['dataset'],r['modality'],r['perturbation_type'])==tag]:
            k=rec['metric'];lo,hi=np.quantile(draws[k],[.025,.975])
            for field,value in [('base',ma[k]),('behavior',mb[k]),('difference',mb[k]-ma[k]),('ci_low',lo),('ci_high',hi)]:assert np.isclose(float(rec[field]),value,atol=1e-12,rtol=1e-10),(tag,k,field)
    return len(records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root.resolve()
    workspace=root.parents[2];base=workspace/'CMERC/experiments/xp_mainline_validation/20261016T050823Z';code=base/'code'
    config=json.loads((root/'configs/perturbation_config.json').read_text());sources=json.loads((root/'provenance/source_hashes.json').read_text())
    checks={};checks['historical_inputs_unchanged']=all(digest(p)==h for p,h in sources['historical_inputs'].items());assert checks['historical_inputs_unchanged']
    alignment=json.loads((root/'results/alignment_summary.json').read_text())
    assert alignment['passed'] and len(alignment['runs'])==10
    rows=list(csv.DictReader((root/'results/utterance_behavior_metrics.csv').open()));summaries=list(csv.DictReader((root/'results/modality_pilot_summary.csv').open()))
    indexed={(r['dataset'],r['modality'],r['perturbation_type'],r['dialogue_id'],int(r['utterance_index'])):r for r in rows}
    assert len(rows)==len(indexed)==(527+1109)*9
    assert len(summaries)==18
    summary_index={(r['dataset'],r['modality'],r['perturbation_type']):r for r in summaries}
    nprob=0;n_cache_checks=0;run_checks=[]
    for ds,raw in [('IEMOCAP','IEMOCAP_features.pkl'),('MELD','MELD_features_raw1.pkl')]:
        cfg=json.loads((base/f'configs/effective/{ds}_cmerc_validation_seed1475.json').read_text());split=cfg['provenance']['split']
        with (code/'data'/raw).open('rb') as f:payload=pickle.load(f,encoding='latin1')
        for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not(set(split[a+'_ids'])&set(split[b+'_ids']))
        expected_n=527 if ds=='IEMOCAP' else 1109;expected_d=12 if ds=='IEMOCAP' else 114
        for seed in config['alignment_seeds']:
            d=root/f'runs/{ds}/seed{seed}';new=read_npz(d/'test_alignment_predictions.npz');old=read_npz(base/f'results/cmerc/{ds}/validation/seed{seed}/predictions.npz')
            for k in ['labels','predictions','dialogue_ids','utterance_indices']:assert np.array_equal(new[k],old[k]),(ds,seed,k)
            for k in ['probabilities','log_probabilities']:assert np.allclose(new[k],old[k],atol=1e-6,rtol=1e-5),(ds,seed,k)
            val=read_npz(d/'validation_predictions.npz');ids=[];uis=[];labels=[];speakers=[];lengths=[]
            for key in split['validation_ids']:
                n=len(payload[2][key]);ids.extend([str(key)]*n);uis.extend(range(n));labels.extend(payload[2][key]);lengths.extend([n]*n)
                if ds=='IEMOCAP':speakers.extend([0 if s=='M' else 1 for s in payload[1][key]])
                else:speakers.extend(np.asarray(payload[1][key]).argmax(-1).tolist())
            for k,v in [('labels',labels),('dialogue_ids',ids),('utterance_indices',uis),('speaker_ids',speakers),('dialogue_lengths',lengths)]:assert np.array_equal(val[k],v),(ds,seed,k)
            assert len(val['labels'])==expected_n and len(set(val['dialogue_ids']))==expected_d
            assert np.array_equal(val['predictions'],val['probabilities'].argmax(1))
            assert np.isfinite(val['log_probabilities']).all() and np.array_equal(np.exp(val['log_probabilities']),val['probabilities'])
            run_checks.append(dict(dataset=ds,seed=seed,test_alignment=True,true_validation_identity=True,utterances=expected_n,dialogues=expected_d))
            if seed!=config['pilots'][ds]:
                assert not (d/'perturbations').exists();continue
            for modality in config['modality_order']:
                complete=json.loads((d/f'{modality}_complete.json').read_text())
                assert complete['passed'] and complete['unchanged_model_state'] and complete['restored_full_input_exact'] and complete['padding_untouched']
                assert complete['counts']=={'gaussian':expected_n*5,'dropout':expected_n*5,'attenuation':expected_n}
                assert complete['graph_index_hook_checks']>=expected_n*11*2
                for kind,spec in config['perturbations'].items():
                    x=read_npz(d/f'perturbations/{modality}_{kind}.npz');p=x['probabilities'];assert p.shape==(expected_n,spec['repeats'],val['probabilities'].shape[1])
                    assert np.isfinite(x['log_probabilities']).all() and np.array_equal(np.exp(x['log_probabilities']),p)
                    assert np.array_equal(x['perturbed_predictions'],p.argmax(-1))
                    assert np.array_equal(x['original_predictions'],val['predictions'])
                    for k in ['labels','dialogue_ids','utterance_indices','speaker_ids','batch_indices','dialogue_lengths']:assert np.array_equal(x[k],val[k])
                    assert np.array_equal(x['original_probabilities'],val['probabilities'])
                    b=independent_metrics(val['probabilities'],p)
                    for i in range(expected_n):
                        record=indexed[(ds,modality,kind,str(val['dialogue_ids'][i]),int(val['utterance_indices'][i]))]
                        for k,v in b.items():assert np.isclose(float(record[k]),v[i],atol=1e-12,rtol=1e-10),(ds,modality,kind,i,k)
                    summary=summary_index[(ds,modality,kind)]
                    originals=independent_classification(val['labels'],val['probabilities'])
                    repeated=[independent_classification(val['labels'],p[:,repeat]) for repeat in range(spec['repeats'])]
                    for key,value in originals.items():
                        assert np.isclose(float(summary['original_'+key]),value,atol=1e-10,rtol=1e-10)
                        assert np.isclose(float(summary['perturbed_'+key]),np.mean([m[key] for m in repeated]),atol=1e-10,rtol=1e-10)
                    nprob+=expected_n*spec['repeats']
                cachepath=root/f'provenance/cache_parity_{ds}_{modality}.json'
                expected_cached=not(ds=='MELD' and modality=='audio')
                assert cachepath.exists()==expected_cached
                if expected_cached:
                    cache=json.loads(cachepath.read_text());assert cache['passed'];assert all(c['passed'] for c in cache['parity_checks']);assert len(cache['parity_checks'])==4*((expected_d+15)//16)
                    signature_calls={}
                    for check in cache['parity_checks']:
                        signature_calls.setdefault(tuple(check['batch_lengths']),set()).add(check['call_in_batch'])
                        assert check['max_probability_error']<=1e-6 or check.get('probabilities_passed',False)
                    assert len(signature_calls)==((expected_d+15)//16)
                    assert all(calls=={1,2,7,12} for calls in signature_calls.values())
                    assert cache['cache_sha256']==digest(root/'scripts/graph_index_cache.py')
                    assert cache['runner_sha256']==digest(root/'scripts/run_pilot.py')
                    wrapper_path=root/('provenance/run_cached_pilot_audio.py' if modality=='audio' else 'scripts/run_cached_pilot.py')
                    assert cache['wrapper_sha256']==digest(wrapper_path)
                    n_cache_checks+=len(cache['parity_checks'])
        del payload
        # Recompute all layer statistics from effective train IDs; padding and other splits never enter.
        feature_name='iemocap_features_roberta.pkl' if ds=='IEMOCAP' else 'meld_features_roberta.pkl'
        with (code/'data'/feature_name).open('rb') as f:text=pickle.load(f,encoding='latin1')
        with (code/'data'/raw).open('rb') as f:payload=pickle.load(f,encoding='latin1')
        features=[payload[4],payload[5]]+[text[i] for i in ([2,3,4,5] if ds=='IEMOCAP' else [3,4,5,6])]
        stats=read_npz(root/f'provenance/{ds}_train_statistics.npz')
        for name,features in zip(['audio','visual','text0','text1','text2','text3'],features):
            values=np.concatenate([np.asarray(features[k],np.float64) for k in split['train_ids']])
            assert np.array_equal(values.std(0,ddof=0).astype(np.float32),stats[name+'_std'])
            assert np.array_equal(values.mean(0).astype(np.float32),stats[name+'_mean'])
            assert len(values)==int(stats[name+'_count'])
        del text,payload,values
    assert nprob==53988
    conditional_count=verify_conditional(root)
    for filename in ['GPU_ALIGNMENT_REPORT.md','VALIDATION_INFERENCE_REPORT.md','SINGLE_MODAL_PILOT_REPORT.md']:assert (root/'reports'/filename).is_file()
    for p in (root/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    sources['new_scripts_final']={str(p):digest(p) for p in (root/'scripts').glob('*.py')}
    sources['alignment_validation_runner_snapshot']={'path':str(root/'provenance/run_pilot_alignment_validation.py'),'sha256':digest(root/'provenance/run_pilot_alignment_validation.py')}
    sources['config_unchanged']=sources['config_sha256']==digest(root/'configs/perturbation_config.json');assert sources['config_unchanged']
    (root/'provenance/source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
    result=dict(status='passed',historical_inputs_unchanged=True,config_unchanged=True,gpu_alignment_runs=10,true_validation_runs=10,pilot_runs=6,
                perturbation_files=18,perturbed_probability_vectors=nprob,behavior_rows=len(rows),summary_rows=len(summaries),
                cached_vs_original_full_vector_checks=n_cache_checks,all_behavior_metrics_independently_recomputed=True,classification_summary_metrics_independently_recomputed=True,conditional_oof_metrics_and_bootstrap_ci_independently_recomputed=conditional_count==54,
                training_statistics_independently_recomputed=True,input_and_model_invariants_passed=True,graph_indices_invariants_passed=True,
                no_training=True,no_joint_modalities=True,no_extra_seed_perturbations=True,analysis_split='validation',runs=run_checks,
                limitations=['Two anchor seeds only; exploratory validation diagnostics, not generalization or calibration improvement',
                             'Bootstrap uses fixed LODO scores and does not cover refit uncertainty',
                             'Feature-space perturbations do not demonstrate physical or semantic robustness'])
    (root/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
