from pathlib import Path
import hashlib,json,platform,shutil,subprocess,sys
import numpy as np
import pandas as pd
import scipy,sklearn
ROOT=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    cfg=json.loads((ROOT/'preregistration.json').read_text());old=Path(cfg['source_mainline']);five=Path(cfg['source_five_seed'])
    manifest=json.loads((old/'provenance/delivery_hashes.json').read_text())
    assert all(sha(old/k)==v for k,v in manifest.items()),'Historical delivery changed'
    assert json.loads((old/'verification.json').read_text())['all_passed']
    assert json.loads((old/'joint/independent_verification.json').read_text())['all_passed']
    sources=[old/'provenance/delivery_hashes.json',old/'verification.json',old/'inputs/joint_features.csv',old/'inputs/verification.json',old/'inputs/source_hashes.json',old/'joint/oof_predictions.csv',old/'joint/independent_verification.json',old/'scripts/mainline_core.py',five/'baseline/fold_parameters.json']+[five/f'baseline/bootstrap/{ds}_dialogue_draws.npz' for ds in cfg['datasets']]
    before={str(p):sha(p) for p in sources}
    src=old/'inputs/joint_features.csv';shutil.copy2(src,ROOT/'inputs/features.csv')
    f=pd.read_csv(src,dtype={'dataset':str,'dialogue_id':str},low_memory=False)
    key=['dataset','seed','dialogue_id','utterance_index']; B=cfg['behavior_features']
    assert len(f)==8180 and not f.duplicated(key).any() and set(f.modality)=={'joint'}
    assert set(f.seed)==set(cfg['seeds']) and set(f.dataset)==set(cfg['datasets'])
    assert np.isfinite(f[['confidence']+B].to_numpy(float)).all()
    assert ((f.prediction==f.label).astype(int)==f.correct).all()
    assert f.confidence.between(0,1).all() and (f[B]>=0).all().all() and (f[B]<=1).all().all()
    groups=[]
    for (ds,seed),g in f.groupby(['dataset','seed']):
        assert len(g)==cfg['datasets'][ds]['utterances'] and g.dialogue_id.nunique()==cfg['datasets'][ds]['dialogues']
        missing=int(g.visual_missing.astype(bool).sum());assert missing=={'IEMOCAP':1,'MELD':95}[ds]
        groups.append({'dataset':ds,'seed':int(seed),'utterances':len(g),'dialogues':g.dialogue_id.nunique(),'visual_missing_retained':missing})
    oldpred=pd.read_csv(old/'joint/oof_predictions.csv',dtype={'dataset':str,'dialogue_id':str},low_memory=False)
    keep=oldpred.method.isin(['E0','E1','M_joint'])
    reference=oldpred.loc[keep,key+['method','fold','label','prediction','correct','oof_confidence']]
    assert len(reference)==24540 and not reference.duplicated(key+['method']).any()
    reference.to_csv(ROOT/'inputs/reference_oof.csv',index=False)
    for method,g in reference.groupby('method'):
        merged=f[key+['label','prediction','correct']].merge(g,on=key,validate='one_to_one',suffixes=('_in','_ref'))
        assert len(merged)==8180
        for c in ['label','prediction','correct']: assert (merged[c+'_in']==merged[c+'_ref']).all()
    shutil.copy2(five/'baseline/fold_parameters.json',ROOT/'inputs/fold_definitions.json')
    for ds in cfg['datasets']: shutil.copy2(five/f'baseline/bootstrap/{ds}_dialogue_draws.npz',ROOT/f'inputs/{ds}_dialogue_draws.npz')
    assert sha(ROOT/'vendor/mainline_core.py')==sha(old/'scripts/mainline_core.py')
    assert all(sha(p)==v for p,v in before.items())
    dump(ROOT/'inputs/source_hashes.json',before)
    dump(ROOT/'inputs/verification.json',{'all_passed':True,'rows':len(f),'unique_original_utterance_records':True,'historical_delivery_files_verified':len(manifest),'historical_independent_audit_reused':True,'source_files_unchanged':True,'reference_rows':len(reference),'scoring_primitives':B,'groups':groups,'new_raw_probability_inference':False})
    repo=Path('/data2/yb/paper/lyx-msa/CMERC')
    dump(ROOT/'provenance/environment.json',{'python':sys.executable,'python_version':sys.version,'host':platform.node(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'device':'CPU','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'preexisting_git_status':subprocess.check_output(['git','status','--short'],cwd=repo,text=True).splitlines(),'argv':sys.argv})
    print(json.dumps({'all_passed':True,'rows':len(f),'historical_files_verified':len(manifest),'reference_rows':len(reference)}))
if __name__=='__main__': main()
