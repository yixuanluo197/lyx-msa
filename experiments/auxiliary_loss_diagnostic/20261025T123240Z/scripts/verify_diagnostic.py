"""CPU-only verification of captured fixed-checkpoint diagnostic artifacts."""
from pathlib import Path
import csv,json,hashlib,os,math,collections,datetime
B=Path(__file__).resolve().parents[1];R=B.parents[2]
def read(p):return json.loads(Path(p).read_text())
def rows(n):
 with (B/'results'/f'{n}.csv').open(newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def yes(v):return v is True or str(v)=='True'
def close(a,b):return math.isclose(float(a),float(b),rel_tol=2e-6,abs_tol=2e-8)
def main():
 s=read(B/'state/status.json');reg=read(B/'configs/registry.json');assert s['phase']=='completed' and not s['active']
 assert s['completed_cases']==128 and s['model_forward_calls']==768 and s['autograd_calls']<=512
 assert s['training_runs']==s['optimizer_steps']==s['test_forwards']==s['tensor_backward_calls']==0 and not s['failed_cases']
 for n,d in read(B/'provenance/frozen_execution.json')['files'].items():assert sha(B/n)==d,n
 for n,d in read(B/'provenance/analysis_frozen.json')['files'].items():assert sha(B/n)==d,n
 inventory=read(B/'preflight/checkpoint_inventory.json');assert inventory['status']=='passed'
 for x in inventory['checkpoints']:assert sha(Path(x['path']))==x['sha256'],x['checkpoint_id']
 hist={}
 for rel,old in read(B/'provenance/history_before.json')['roots'].items():
  files={};links={};base=R/rel
  for folder,_,names in os.walk(base,followlinks=False):
   for n in names:
    p=Path(folder)/n;k=str(p.relative_to(base))
    if p.is_symlink():links[k]=os.readlink(p)
    elif p.is_file():files[k]=sha(p)
  assert files==old['files'] and links==old['symlinks'],rel
  hist[rel]=len(files)
 fixed=read(B/'private/fixed_batches.json')['batches'];ref=read(B/'provenance/xp04_validation_reference.json')
 selected=[v for b in fixed for v in b['dialogue_ids']];assert len(set(selected))==len(selected)==512
 assert set(selected).issubset(ref['split']['train_ids']) and not set(selected)&set(ref['split']['validation_ids']+ref['split']['test_ids'])
 cp={c['checkpoint_id'] for c in reg['checkpoints']};expected={(c,b['batch_id']) for c in cp for b in fixed};assert len(expected)==128
 cases=rows('case_summary');assert len(cases)==128 and {(x['checkpoint_id'],x['batch_id']) for x in cases}==expected
 assert sum(int(x['model_forward_calls']) for x in cases)==s['model_forward_calls']
 assert sum(int(x['autograd_calls']) for x in cases)==s['autograd_calls']
 for x in cases:
  assert all(yes(x[k]) for k in ['parameters_buffers_unchanged','RNG_unchanged','grad_buffers_unchanged'])
  assert [int(x['forward_'+k]) for k in ['model','model_mm','model_intra','model_inter']]==[1,3,1,1]
  assert int(x['n_dialogues'])==16 and int(x['n_utterances'])>0
  assert float(x['actual_mscl_coefficient'])==float(x['actual_cscl_coefficient'])==(.15 if x['checkpoint_id'].startswith('XP04') else 0.)
 audit=rows('state_audit');assert len(audit)==128
 for x in audit:
  assert x['rng_before']==x['rng_after'] and yes(x['all_parameter_grad_buffers_none'])
  assert all(x['before_'+k]==x['after_'+k] for k in ['model','model_mm','model_intra','model_inter'])
 pairs=rows('gradient_pairs');assert len(pairs)==128*5*8
 valid=0
 for x in pairs:
  if x['cosine']!='':
   a,b,c,d=map(float,[x['norm_a_common'],x['norm_b_common'],x['cosine'],x['dot_raw']]);assert a>0 and b>0 and int(x['common_tensors'])>0
   assert close(c,max(-1.,min(1.,d/(a*b)))) and yes(x['negative_cosine'])==(c<0)
   assert close(x['dot_reference_weighted'],float(x['reference_coefficient'])*d)
   assert close(x['norm_ratio_reference_weighted'],float(x['reference_coefficient'])*b/a)
   valid+=1
  else:assert int(x['common_tensors'])==0 or x['status']=='nonfinite_aggregate'
 norms=rows('gradient_norms');assert len(norms)==128*6*8
 for x in norms:
  if x['raw_l2']!='':
   assert math.isfinite(float(x['raw_l2'])) and float(x['raw_l2'])>=0
   assert close(x['reference_weighted_l2'],float(x['raw_l2'])*float(x['reference_coefficient']))
   assert close(x['actual_weighted_l2'],float(x['raw_l2'])*float(x['actual_coefficient']))
 public={name:rows(name) for name in ['pseudo_counts','pair_counts','class_relations','branch_counts','confidence_summary','contribution_summary','ckscl_numeric','ckscl_wrapper_audit']}
 keys=lambda x:(x['checkpoint_id'],x['batch_id'],x['relation'])
 pc=collections.defaultdict(dict)
 for x in public['pseudo_counts']:pc[keys(x)][int(x['pseudo_label'])]=x
 bc=collections.defaultdict(dict)
 for x in public['branch_counts']:bc[keys(x)][x['branch']]=x
 cc=collections.defaultdict(dict)
 for x in public['class_relations']:cc[keys(x)][(int(x['emotion_class']),int(x['pseudo_label']))]=x
 ac=collections.defaultdict(dict)
 for x in public['pair_counts']:ac[keys(x)]=x
 conf=collections.defaultdict(list)
 for x in public['confidence_summary']:conf[keys(x)].append(x)
 contrib=collections.defaultdict(list)
 for x in public['contribution_summary']:
  if x['relation'] in ['t','a','v','intra','inter']:contrib[keys(x)].append(x)
 private_count=0
 for checkpoint,batch in sorted(expected):
  p=B/'private'/checkpoint/(batch+'.json');d=read(p);assert set(d['gradient_sha256'])=={'classification','cc','mscl','cscl'}
  pr=d['utterance_records'];private_count+=len(pr)
  assert d['dialogue_ids']==next(b['dialogue_ids'] for b in fixed if b['batch_id']==batch)
  for relation in ['t','a','v','intra','inter']:
   part=[r for r in pr if r['relation']==relation];n=len(part);assert n>0
   key=(checkpoint,batch,relation)
   for pseudo in [0,1]:
    chosen=[r for r in part if r['pseudo_label']==pseudo];row=pc[key][pseudo]
    assert int(row['n_pseudo'])==len(chosen) and close(row['fraction'],len(chosen)/n)
    assert int(row['n_default_zero'])==sum(r['default_zero'] for r in chosen)
    assert close(row['class_weight'],n/len(chosen) if chosen else 0.)
    for emotion in range(7):assert int(cc[key][(emotion,pseudo)]['n_utterances'])==sum(r['emotion_class']==emotion for r in chosen)
   for branch,row in bc[key].items():assert int(row['n_utterances'])==sum(r['branch']==branch for r in part)
   pair=ac[key];n0=sum(r['pseudo_label']==0 for r in part);n1=n-n0
   assert int(pair['true_positive_pairs'])==n0*(n0-1)+n1*(n1-1)
   assert int(pair['true_negative_pairs'])==2*n0*n1
   for row in conf[key]:
    subset=row['subset'];chosen=[r for r in part if subset=='all' or subset=='default0' and r['default_zero'] or subset=='explicit0' and r['pseudo_label']==0 and not r['default_zero'] or subset=='explicit1' and r['pseudo_label']==1]
    assert int(row['n_utterances'])==len(chosen)
    if chosen:
     assert close(row['mean_main_gt_probability'],sum(r['main_gt_probability'] for r in chosen)/len(chosen))
     assert close(row['mean_gt_probability_delta'],sum(r['gt_probability_delta'] for r in chosen)/len(chosen))
    else:assert row['mean_gt_probability_delta']==''
   for row in contrib[key]:
    chosen=[r for r in part if (row['emotion_class']=='' or r['emotion_class']==int(row['emotion_class'])) and (row['pseudo_label']=='' or r['pseudo_label']==int(row['pseudo_label'])) and (row['branch']=='' or r['branch']==row['branch'])]
    assert len(chosen)==int(row['n_utterances'])
    good=[r['contribution'] for r in chosen if r['contribution'] is not None]
    assert close(row['finite_contribution_sum'],sum(good))
    assert int(row['n_nonfinite_contributions'])==len(chosen)-len(good)
 assert len(public['ckscl_numeric'])==640 and len(public['ckscl_wrapper_audit'])==256
 for x in public['ckscl_numeric']:
  key=keys(x);assert int(x['negative_mask_cells'])==int(ac[key]['true_positive_pairs'])
 for x in public['ckscl_wrapper_audit']:
  assert yes(x['fallback_observed'])==yes(x['nonfinite_before_fallback'])
  if yes(x['fallback_observed']):assert float(x['returned_loss'])==0
  else:assert close(x['pre_fallback_loss'],x['returned_loss'])
 # Public tables have aggregates only; raw anchor records remain private.
 forbidden={'dialogue_id','dialogue_ids','utterance_id','utterance_index','utterance_text','main_pred','removed_pred'}
 for name in ['case_summary','gradient_norms','gradient_pairs',*public]:
  data=rows(name);assert not forbidden&set(data[0]),name
 outputs={str(p.relative_to(B)):sha(p) for p in (B/'results').glob('*.csv')}
 result={'status':'passed','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':128,'checkpoint_count':4,'training_runs':0,'optimizer_steps':0,'test_forwards':0,'tensor_backward_calls':0,'model_forward_calls':s['model_forward_calls'],'autograd_calls':s['autograd_calls'],'frozen_sources_inputs_checkpoints_unchanged':True,'case_parameter_buffer_rng_grad_states_unchanged':True,'same_frozen_training_batches_all_checkpoints':True,'private_utterance_relation_records_verified':private_count,'valid_gradient_pairs':valid,'historical_files_unchanged':hist,'output_csv_sha256':outputs,'scope':'CPU aggregates/private reconciliation and input hashes only; no model forward, no optimizer, no CUDA initialization'}
 (B/'provenance/final_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'passed','cases':128,'historical_files':sum(hist.values()),'private_records':private_count}))
if __name__=='__main__':main()
