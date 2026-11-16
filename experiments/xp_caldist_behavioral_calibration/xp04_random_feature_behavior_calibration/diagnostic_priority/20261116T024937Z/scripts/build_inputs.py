from pathlib import Path
import hashlib,json,platform,shutil,subprocess,sys
import numpy as np,pandas as pd,scipy,sklearn
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 c=json.loads((ROOT/'diagnostic_protocol.json').read_text());inter=Path(c['source_interaction']);mechanism=Path(c['source_mechanism']);mainline=Path(c['source_mainline']);five=Path(c['source_five_seed'])
 counts={}
 for source,manifest_name in [(inter,'provenance/delivery_hashes.json'),(mechanism,'provenance/artifact_hashes.json'),(mainline,'provenance/delivery_hashes.json')]:
  manifest=json.loads((source/manifest_name).read_text())
  assert all(sha(source/k)==v for k,v in manifest.items()),str(source)
  counts[str(source)]=len(manifest)
 sources=[inter/'inputs/features.csv',inter/'results/aggregate_metrics.csv',inter/'results/per_seed_metrics.csv',inter/'results/bootstrap_results.csv',inter/'results/independent_verification.json',mechanism/'diagnostic/split_responses.csv',mechanism/'diagnostic/split_reliability.csv',mechanism/'diagnostic/verification.json',mainline/'modality/bootstrap_results.csv']+[five/f'baseline/bootstrap/{d}_dialogue_draws.npz' for d in c['datasets']]
 before={str(p):sha(p) for p in sources}
 shutil.copy2(inter/'inputs/features.csv',ROOT/'inputs/features.csv')
 f=pd.read_csv(ROOT/'inputs/features.csv',dtype={'dialogue_id':str},low_memory=False);key=['dataset','seed','dialogue_id','utterance_index'];assert len(f)==8180 and not f.duplicated(key).any()
 assert set(f.seed)==set(c['seeds']) and set(f.dataset)==set(c['datasets']) and set(f.modality)=={'joint'}
 assert ((f.prediction==f.label).astype(int)==f.correct).all()
 for (d,s),g in f.groupby(['dataset','seed']):
  assert len(g)==c['datasets'][d]['utterances'] and g.dialogue_id.nunique()==c['datasets'][d]['dialogues']
  assert int(g.visual_missing.sum())=={'IEMOCAP':1,'MELD':95}[d]
 for corruption in c['corruptions']:
  cols=[corruption+'_'+m for m in ['mu','delta','flip_rate','confidence_change_mean','confidence_change_std']]
  assert np.isfinite(f[cols]).all().all()
  assert np.allclose(f[corruption+'_delta'],f[corruption+'_confidence_change_mean'].abs(),atol=1e-12,rtol=0)
 split=pd.read_csv(mechanism/'diagnostic/split_responses.csv',dtype={'dialogue_id':str},low_memory=False)
 keep=key+['corruption','split','mu_left','mu_right','delta_left','delta_right'];split=split[keep]
 assert len(split)==49080 and not split.duplicated(key+['corruption','split']).any()
 assert len(split.merge(f[key],on=key,validate='many_to_one'))==49080
 assert set(split['split'])=={0,1,2} and set(split.corruption)==set(c['corruptions'])
 split.to_csv(ROOT/'inputs/split_responses.csv',index=False)
 rel=pd.read_csv(mechanism/'diagnostic/split_reliability.csv');rel=rel[rel.measure.isin(c['measures'])];assert len(rel)==120
 rel.to_csv(ROOT/'inputs/reference_split_reliability.csv',index=False)
 for src,name in [(inter/'results/aggregate_metrics.csv','prior_aggregate_metrics.csv'),(inter/'results/per_seed_metrics.csv','prior_per_seed_metrics.csv'),(inter/'results/bootstrap_results.csv','prior_bootstrap_results.csv'),(mainline/'modality/bootstrap_results.csv','prior_modality_bootstrap.csv')]:shutil.copy2(src,ROOT/'inputs'/name)
 for d in c['datasets']:shutil.copy2(five/f'baseline/bootstrap/{d}_dialogue_draws.npz',ROOT/f'inputs/{d}_dialogue_draws.npz')
 assert all(sha(p)==v for p,v in before.items())
 dump(ROOT/'inputs/source_hashes.json',before)
 dump(ROOT/'inputs/verification.json',{'all_passed':True,'canonical_rows':8180,'split_rows':49080,'split_reference_rows':120,'source_files_unchanged':True,'historical_manifests_verified':counts,'std_convention':'population ddof0 over10, SE signed mean=stdpop/sqrt9','visual_missing_retained':True,'no_new_inference':True})
 repo=Path('/data2/yb/paper/lyx-msa/CMERC')
 dump(ROOT/'provenance/environment.json',{'python':sys.executable,'python_version':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'host':platform.node(),'device':'CPU','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'preexisting_git_status':subprocess.check_output(['git','status','--short'],cwd=repo,text=True).splitlines()})
 print(json.dumps({'all_passed':True,'input_rows':len(f),'split_rows':len(split),'historical_manifests':counts}))
if __name__=='__main__':main()
