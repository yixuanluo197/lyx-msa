"""Independent serialized-parameter and metric verification of CPU baselines."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[4]
OUT = RUN/'baseline'
KEY = ['dataset','seed','dialogue_id','utterance_index']


def main():
    original = pd.read_csv(OUT/'original_predictions.csv', dtype={'dialogue_id': str})
    oof = pd.read_csv(OUT/'oof_predictions.csv', dtype={'dialogue_id': str})
    scores = pd.read_csv(OUT/'per_seed_metrics.csv')
    params = json.loads((OUT/'fold_parameters.json').read_text())
    assert len(original) == 8180 and len(oof) == 24540 and len(scores) == 30 and len(params) == 85
    assert not original.duplicated(KEY).any()
    assert not oof.duplicated(KEY+['method']).any()
    checked = 0
    max_error = 0.
    xp = oof[oof.method=='XP04'].merge(original[KEY+['confidence']],on=KEY,validate='one_to_one')
    assert len(xp)==8180 and np.allclose(xp.oof_confidence,xp.confidence,atol=1e-15,rtol=0)
    for f in params:
        all_rows = original[(original.dataset == f['dataset'])&(original.seed == f['seed'])]
        train = all_rows[all_rows.dialogue_id.isin(f['train_dialogues'])]
        held = all_rows[all_rows.dialogue_id.isin(f['held_dialogues'])]
        assert set(f['train_dialogues']).isdisjoint(f['held_dialogues'])
        assert len(train) == f['train_n'] and len(held) == f['held_n']
        assert len(train)+len(held) == len(all_rows)
        for method, p in f['methods'].items():
            if method == 'L0':
                xtrain = logit(np.clip(train.confidence.to_numpy(), 1e-6,1-1e-6))
                assert np.allclose(p['mean'], xtrain.mean(), atol=1e-12)
                assert np.allclose(p['scale'], xtrain.std(ddof=0), atol=1e-12)
            else:
                assert p['mean'] == [0.] and p['scale'] == [1.]
            saved = oof[(oof.dataset == f['dataset'])&(oof.seed == f['seed'])&
                        (oof.method == method)&(oof.fold == f['fold'])]
            frame = held.merge(saved[KEY+['oof_confidence']],on=KEY,validate='one_to_one')
            x = frame.confidence.to_numpy()
            if method == 'L0':
                x = logit(np.clip(x, 1e-6,1-1e-6))
            pred = expit(p['intercept']+p['coefficient']*(x-p['mean'][0])/p['scale'][0])
            error = np.abs(pred-frame.oof_confidence.to_numpy()).max()
            max_error = max(error,max_error)
            assert error < 2e-12
            checked += len(frame)
    for row in scores.to_dict('records'):
        s = oof[(oof.dataset == row['dataset'])&(oof.seed == row['seed'])&(oof.method == row['method'])]
        p,y = s.oof_confidence.to_numpy(), s.correct.to_numpy()
        assert np.array_equal(y,(s.label.to_numpy()==s.prediction.to_numpy()).astype(int))
        edges = np.linspace(0,1,16)
        ece = 0.
        for lo,hi in zip(edges[:-1],edges[1:]):
            use = (p>=lo)&((p<hi) if hi<1 else (p<=hi))
            if use.any(): ece += use.mean()*abs(y[use].mean()-p[use].mean())
        assert abs(ece-row['ece15']) < 1e-12
        assert abs(brier_score_loss(y,p)-row['binary_brier']) < 1e-12
        assert abs(log_loss(y,p,labels=[0,1])-row['binary_nll']) < 1e-12
    # Match historical true-Validation OOF for both baseline choices.
    history_path = ROOT/'experiments/xp_caldist_behavioral_calibration/xp04_random_feature_behavior_calibration/unified_module_pilot/20261115T145425Z/oof_predictions.csv'
    history = pd.read_csv(history_path,dtype={'dialogue_id':str})
    old = history[history.method.isin(['XP04','E1','L0'])]
    now = oof[oof.seed == 1475]
    merged = old.merge(now,on=KEY+['method'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(merged) == 4908
    assert np.array_equal(merged.fold_old,merged.fold_new)
    hist_error = np.abs(merged.oof_confidence_old-merged.oof_confidence_new).max()
    assert hist_error < 1e-12, hist_error
    for dataset, size, dialogues in [('IEMOCAP',527,12),('MELD',1109,114)]:
        canonical = original[(original.dataset == dataset)&(original.seed == 1475)]
        for seed in (1475,67137,2024,3407,42):
            s = original[(original.dataset == dataset)&(original.seed == seed)]
            assert len(s) == size and s.dialogue_id.nunique() == dialogues
            merged = canonical.merge(s,on=['dialogue_id','utterance_index'],validate='one_to_one',suffixes=('_a','_b'))
            assert np.array_equal(merged.label_a,merged.label_b)
    for file,digest in json.loads((OUT/'source_hashes.json').read_text()).items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
    aggregate = pd.read_csv(OUT/'aggregate_metrics.csv')
    for r in aggregate.to_dict('records'):
        s = scores[(scores.dataset == r['dataset'])&(scores.method == r['method'])][r['metric']]
        assert abs(s.mean()-r['mean']) < 1e-12
        assert abs(s.std(ddof=1)-r['sample_sd']) < 1e-12
    # Independently recompute the first shared-dialogue bootstrap draw with expanded rows.
    for dataset in ('IEMOCAP','MELD'):
        a = np.load(OUT/'bootstrap'/f'{dataset}_dialogue_draws.npz')
        assert a['draws'].shape == (2000,len(a['dialogue_ids']))
        draw = a['dialogue_ids'][a['draws'][0]]
        expected = []
        for seed in (1475,67137,2024,3407,42):
            methods = {}
            for method in ('XP04','E1','L0'):
                s = oof[(oof.dataset==dataset)&(oof.seed==seed)&(oof.method==method)]
                sampled = pd.concat([s[s.dialogue_id==d] for d in draw])
                p,y = sampled.oof_confidence.to_numpy(),sampled.correct.to_numpy()
                bins = np.minimum((p*15).astype(int),14)
                ece = sum((bins==k).mean()*abs(p[bins==k].mean()-y[bins==k].mean()) for k in range(15) if (bins==k).any())
                methods[method] = np.array([ece,brier_score_loss(y,p),log_loss(y,p,labels=[0,1])])
            expected.append(methods)
        for left,right in [('E1','XP04'),('L0','XP04'),('L0','E1')]:
            actual = np.load(OUT/'bootstrap'/f'{dataset}_{left}_vs_{right}_metric_differences.npy')[0]
            assert np.allclose(actual,np.mean([s[left]-s[right] for s in expected],axis=0),atol=1e-12)
    verification = {'status':'completed', 'all_passed':True,
        'original_rows':len(original),'oof_rows':len(oof),'fold_parameter_records':len(params),
        'serialized_parameter_predictions_recomputed':checked,'XP04_identity_predictions_verified':len(xp),'max_prediction_error':max_error,
        'historical_seed1475_matching_rows':4908,'historical_max_error':hist_error,
        'metrics_independently_recomputed':len(scores),'same_ids_labels_all_five_seeds':True,
        'group_disjointness_all_folds':True,'train_only_scaler_verified':True,
        'source_hashes_verified':True,'bootstrap_shared_draw_recomputation':True,
        'GPU_used':False,'Test_used':False,'note':'fresh checkpoint parity for non1475 seeds is a separate subsequent perturbation audit'}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    hashes = {str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    (OUT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__ == '__main__':
    main()
