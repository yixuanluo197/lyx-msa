"""Compare already-generated full versus interrupted epoch-boundary checkpoints."""
import json,sys
from pathlib import Path
import numpy as np
import torch
p=Path(sys.argv[1]);full=p/'full/cmerc/IEMOCAP_validation_seed2024'; resumed=p/'resumed/cmerc/IEMOCAP_validation_seed2024'
a=torch.load(full/'latest.pt',map_location='cpu');b=torch.load(resumed/'latest.pt',map_location='cpu')
def compare(x,y,exact=False):
 if torch.is_tensor(x):
  if exact: assert torch.equal(x,y)
  else: torch.testing.assert_close(x,y,rtol=1e-3,atol=1e-4)
 elif isinstance(x,np.ndarray):assert np.array_equal(x,y)
 elif isinstance(x,dict):
  assert x.keys()==y.keys()
  for k in x:compare(x[k],y[k],exact)
 elif isinstance(x,(tuple,list)):
  assert len(x)==len(y)
  for u,v in zip(x,y):compare(u,v,exact)
 else:assert x==y,(x,y)
assert a['epoch']==b['epoch']==1
assert a['provenance']==b['provenance']
compare(a['rng_state'],b['rng_state'],exact=True)
compare(a['model_state_dict'],b['model_state_dict'])
compare(a['optimizer_state_dict'],b['optimizer_state_dict'])
compare(a['sampling_metadata'],b['sampling_metadata'],exact=True)
assert a['best_epoch']==b['best_epoch'] and a['best_metric']==b['best_metric']
ra=[json.loads(l) for l in (full/'metrics.jsonl').read_text().splitlines()]
rb=[json.loads(l) for l in (resumed/'metrics.jsonl').read_text().splitlines()]
for x,y in zip(ra,rb):
 for k in ('epoch','batch_policy','sampler_retries','max_batch_edges','max_batch_nodes','sample_ratio','train_dialogues','selection_split','selection_weighted_f1'):assert x[k]==y[k],k
assert len(ra)==len(rb)==2
for d in (full,resumed):
 f=json.loads((d/'final_test.json').read_text());assert f['selected_epoch']==a['best_epoch']+1
max_diff=max(float((v-b['model_state_dict'][k]).abs().max()) for k,v in a['model_state_dict'].items() if v.is_floating_point())
payload={'status':'passed','epochs':2,'rtol':1e-3,'atol':1e-4,'model_and_optimizer_close':True,'all_rng_exact':True,'sampling_progress_exact':True,'selection_records_exact':True,'max_parameter_difference':max_diff,'same_physical_gpu':True,'scope':'engineering preflight, not formal results'}
(p/'verification.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload))
