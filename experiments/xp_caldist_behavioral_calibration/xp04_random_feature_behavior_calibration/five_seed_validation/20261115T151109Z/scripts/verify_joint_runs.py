"""Independent audit of all completed GPU records and cross-model RNG pairing."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[4]
PILOT = ROOT.parents[1] / 'pilot/20261113T025053Z'
KEY = ['dataset','seed','dialogue_id','utterance_index']
NAMES = ['text0','text1','text2','text3','audio','visual']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT/'provenance/source_hashes.json').read_text())
    for path, expected in manifest.items():
        assert sha(path) == expected, path
    baseline = pd.read_csv(ROOT/'baseline/original_predictions.csv',dtype={'dialogue_id':str})
    expected_per_dataset = {'IEMOCAP':527,'MELD':1109}
    records, pairing = [], {}
    for dataset in ['IEMOCAP','MELD']:
        digests = {}
        for seed in [1475,67137,2024,3407,42]:
            if seed == 1475:
                source = PILOT/dataset
                old = pd.read_csv(source/'perturbed_predictions.csv',dtype={'dialogue_id':str})
                pert = old[old.modality.eq('joint')].copy()
                origin = pd.read_csv(source/'original_predictions.csv',dtype={'dialogue_id':str})
                evidence = json.loads((source/'validation_parity.json').read_text())
                assert evidence['completed']
            else:
                source = ROOT/f'runs/{dataset}/seed{seed}'
                evidence = json.loads((source/'complete.json').read_text())
                assert evidence['status'] == 'completed' and evidence['weights_unchanged'] and evidence['batch_inputs_restored']
                assert evidence['parity']['passed']
                assert evidence['target_mask_checks'] >= 2 * (12 if dataset=='IEMOCAP' else 114)
                assert evidence['cached_uncached_parity_checks'] >= 3 * (1 if dataset=='IEMOCAP' else 8)
                for path, expected in evidence['source_sha256'].items():
                    assert sha(path) == expected
                origin = pd.read_csv(source/'original_predictions.csv',dtype={'dialogue_id':str})
                pert = pd.read_csv(source/'perturbed_predictions.csv',dtype={'dialogue_id':str})
            assert len(origin)==expected_per_dataset[dataset]
            assert len(pert)==20*len(origin) and not pert.duplicated(KEY+['corruption','repeat_id']).any()
            assert set(pert.seed)=={seed} and set(pert.modality)=={'joint'} and np.allclose(pert.strength,.05)
            assert set(pert.corruption)=={'gaussian','dropout'}
            matched=origin.merge(baseline[baseline.dataset.eq(dataset)&baseline.seed.eq(seed)],on=KEY,validate='one_to_one',suffixes=('_run','_baseline'))
            assert len(matched)==len(origin)
            for col in ['label','prediction']:
                assert np.array_equal(matched[col+'_run'],matched[col+'_baseline'])
            assert np.allclose(matched.confidence_run,matched.confidence_baseline,atol=1e-12,rtol=0)
            origin_lookup=origin.set_index(KEY)
            key_digest=hashlib.sha256()
            missing_ids=set()
            for key, group in pert.groupby(KEY+['corruption'],sort=True):
                group=group.sort_values('repeat_id')
                assert group.repeat_id.tolist()==list(range(10))
                orig=origin_lookup.loc[key[:4]]
                for row in group.itertuples():
                    p=np.asarray(json.loads(row.probabilities))
                    assert np.isfinite(p).all() and np.all(p>=0) and abs(p.sum()-1)<1e-5
                    assert p.argmax()==row.prediction and abs(p.max()-row.confidence)<1e-12
                    if hasattr(row,'log_probabilities'):
                        assert np.allclose(np.exp(json.loads(row.log_probabilities)),p,atol=1e-12,rtol=0)
                    names=NAMES[:-1] if row.visual_missing_skipped else NAMES
                    expected_keys=[[dataset,1475,str(row.dialogue_id),int(row.utterance_index),n,row.corruption,.05,int(row.repeat_id)] for n in names]
                    assert json.loads(row.rng_key)==expected_keys
                    key_digest.update(json.dumps(expected_keys,separators=(',',':')).encode())
                    if row.visual_missing_skipped:missing_ids.add((row.dialogue_id,int(row.utterance_index)))
                assert int(orig.label)>=0
            assert len(missing_ids)==(1 if dataset=='IEMOCAP' else 95)
            digest=key_digest.hexdigest();digests[str(seed)]=digest
            records.append({'dataset':dataset,'seed':seed,'original_rows':len(origin),'joint_perturbed_rows':len(pert),
                            'visual_missing_retained':len(missing_ids),'rng_digest':digest,
                            'source':str(source),'checkpoint_parity_passed':True})
        assert len(set(digests.values()))==1,'Feature random streams differ across model seeds'
        pairing[dataset]=digests
    result={'status':'completed','all_passed':True,'runs':records,'joint_perturbed_rows':sum(x['joint_perturbed_rows'] for x in records),
            'frozen_source_files_rehashed':len(manifest),'cross_seed_random_streams_identical':True,'rng_digests':pairing,
            'note':'Seed1475 records reused; eight new runs passed fresh checkpoint parity and runtime state/mask checks'}
    (ROOT/'provenance/joint_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_passed':True,'runs':len(records),'joint_rows':result['joint_perturbed_rows']},indent=2))


if __name__=='__main__':
    main()
